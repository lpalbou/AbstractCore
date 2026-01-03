"""
vLLM provider implementation with advanced features.

vLLM-specific features:
- Guided Decoding: guided_regex, guided_json, guided_grammar
- Multi-LoRA: load_adapter, unload_adapter, list_adapters
- Beam Search: best_of, use_beam_search
"""

import os
import httpx
import json
import time
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Type

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError, format_model_error, format_provider_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType


class VLLMProvider(BaseProvider):
    """vLLM provider for high-throughput GPU inference with advanced features."""

    def __init__(self, model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "vllm"

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Base URL: parameter > VLLM_BASE_URL > default
        self.base_url = (
            base_url or
            os.getenv("VLLM_BASE_URL") or
            "http://localhost:8000/v1"
        ).rstrip('/')

        # API key: parameter > VLLM_API_KEY > "EMPTY"
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"

        # Get timeout value - None means unlimited timeout
        timeout_value = getattr(self, '_timeout', None)
        if timeout_value is not None and timeout_value <= 0:
            timeout_value = None  # Invalid timeout becomes unlimited

        try:
            self.client = httpx.Client(timeout=timeout_value)
        except Exception as e:
            try:
                fallback_timeout = None
                try:
                    from ..config.manager import get_config_manager

                    fallback_timeout = float(get_config_manager().get_default_timeout())
                except Exception:
                    fallback_timeout = 7200.0
                if isinstance(fallback_timeout, (int, float)) and float(fallback_timeout) <= 0:
                    fallback_timeout = None
                self.client = httpx.Client(timeout=fallback_timeout)
            except Exception:
                raise RuntimeError(f"Failed to create HTTP client for vLLM: {e}")

        self._async_client = None  # Lazy-loaded async client

        # Validate model exists in vLLM
        self._validate_model()

    @property
    def async_client(self):
        """Lazy-load async HTTP client for native async operations."""
        if self._async_client is None:
            timeout_value = getattr(self, '_timeout', None)
            if timeout_value is not None and timeout_value <= 0:
                timeout_value = None
            self._async_client = httpx.AsyncClient(timeout=timeout_value)
        return self._async_client

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers including API key if configured."""
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _validate_model(self):
        """Validate that the model exists in vLLM."""
        try:
            available_models = self.list_available_models(base_url=self.base_url)
            if available_models and self.model not in available_models:
                error_message = format_model_error("vLLM", self.model, available_models)
                raise ModelNotFoundError(error_message)
        except httpx.ConnectError:
            if hasattr(self, 'logger'):
                self.logger.debug(f"vLLM server not accessible at {self.base_url} - model validation skipped")
            pass
        except ModelNotFoundError:
            raise
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Model validation failed with error: {e} - continuing anyway")
            pass

    def unload(self) -> None:
        """
        Close HTTP client connection.

        Note: vLLM manages model memory automatically.
        This method only closes the HTTP client connection for cleanup.
        """
        try:
            if hasattr(self, 'client') and self.client is not None:
                self.client.close()

            if self._async_client is not None:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._async_client.aclose())
                except RuntimeError:
                    import asyncio
                    asyncio.run(self._async_client.aclose())

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry."""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          execute_tools: Optional[bool] = None,
                          tool_call_tags: Optional[str] = None,
                          # vLLM-specific parameters:
                          guided_regex: Optional[str] = None,
                          guided_json: Optional[Dict] = None,
                          guided_grammar: Optional[str] = None,
                          best_of: Optional[int] = None,
                          use_beam_search: bool = False,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using vLLM with advanced features."""

        # Build messages for chat completions with tool support
        chat_messages = []

        # Add tools to system prompt if provided
        final_system_prompt = system_prompt
        # Prefer native tools when the model supports them. Only inject a prompted tool list
        # when native tool calling is not available.
        if tools and self.tool_handler.supports_prompted and not self.tool_handler.supports_native:
            include_tool_list = True
            if final_system_prompt and "## Tools (session)" in final_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if final_system_prompt:
                final_system_prompt += f"\n\n{tool_prompt}"
            else:
                final_system_prompt = tool_prompt

        # Add system message if provided
        if final_system_prompt:
            chat_messages.append({
                "role": "system",
                "content": final_system_prompt
            })

        # Add conversation history
        if messages:
            chat_messages.extend(messages)

        # Handle media content if provided
        if media:
            user_message_text = prompt.strip() if prompt else ""
            if not user_message_text and chat_messages:
                for msg in reversed(chat_messages):
                    if msg.get("role") == "user" and msg.get("content"):
                        user_message_text = msg["content"]
                        break
            try:
                processed_media = self._process_media_content(media)
                media_handler = self._get_media_handler_for_model(self.model)
                multimodal_message = media_handler.create_multimodal_message(user_message_text, processed_media)

                if isinstance(multimodal_message, str):
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1]["content"] = multimodal_message
                    else:
                        chat_messages.append({"role": "user", "content": multimodal_message})
                else:
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1] = multimodal_message
                    else:
                        chat_messages.append(multimodal_message)
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install abstractcore[media]")
                if user_message_text:
                    chat_messages.append({"role": "user", "content": user_message_text})
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                if user_message_text:
                    chat_messages.append({"role": "user", "content": user_message_text})

        # Add prompt as separate message if provided
        elif prompt and prompt.strip():
            chat_messages.append({"role": "user", "content": prompt})

        # Build request payload using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": max_output_tokens,
            "top_p": kwargs.get("top_p", 0.9),
        }

        # Native tools (OpenAI-compatible): send structured tools/tool_choice when supported.
        if tools and self.tool_handler.supports_native:
            payload["tools"] = self.tool_handler.prepare_tools_for_native(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add additional generation parameters if provided
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]

        # Add seed if provided
        seed_value = kwargs.get("seed", self.seed)
        if seed_value is not None:
            payload["seed"] = seed_value

        # Build extra_body for vLLM-specific features
        extra_body = {}

        # Guided decoding
        if guided_regex:
            extra_body["guided_regex"] = guided_regex
        if guided_json:
            extra_body["guided_json"] = guided_json
        if guided_grammar:
            extra_body["guided_grammar"] = guided_grammar

        # Beam search
        if use_beam_search or best_of:
            extra_body["use_beam_search"] = use_beam_search
            if best_of:
                extra_body["best_of"] = best_of

        # Add structured output support (standard OpenAI-compatible format)
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        # Add extra_body if we have vLLM-specific parameters
        if extra_body:
            payload["extra_body"] = extra_body

        if stream:
            return self._stream_generate(payload)
        else:
            response = self._single_generate(payload)

            # Execute tools if enabled and tools are present
            if self.execute_tools and tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools, execute_tools)

            return response

    def _single_generate(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Generate single response."""
        try:
            if not hasattr(self, 'client') or self.client is None:
                raise ProviderAPIError("HTTP client not initialized")

            start_time = time.time()
            request_url = f"{self.base_url}/chat/completions"
            response = self.client.post(
                request_url,
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            gen_time = round((time.time() - start_time) * 1000, 1)

            result = response.json()

            # Extract response from OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    message = {}

                content = message.get("content", "")
                tool_calls = message.get("tool_calls")
                if tool_calls is None:
                    tool_calls = choice.get("tool_calls")
                finish_reason = choice.get("finish_reason", "stop")
            else:
                content = "No response generated"
                tool_calls = None
                finish_reason = "error"

            # Extract usage info
            usage = result.get("usage", {})

            return GenerateResponse(
                content=content,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=result,
                tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                metadata={
                    "_provider_request": {
                        "url": request_url,
                        "payload": payload,
                    }
                },
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                },
                gen_time=gen_time
            )

        except AttributeError as e:
            if "'NoneType'" in str(e):
                raise ProviderAPIError(f"vLLM provider not properly initialized: {str(e)}")
            else:
                raise ProviderAPIError(f"vLLM configuration error: {str(e)}")
        except Exception as e:
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str or 'model' in error_str) and ('not found' in error_str):
                try:
                    available_models = self.list_available_models(base_url=self.base_url)
                    error_message = format_model_error("vLLM", self.model, available_models)
                    raise ModelNotFoundError(error_message)
                except Exception:
                    raise ModelNotFoundError(f"Model '{self.model}' not found in vLLM and could not fetch available models")
            else:
                raise

    def _stream_generate(self, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Generate streaming response."""
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._get_headers()
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        if isinstance(line, bytes):
                            line = line.decode('utf-8')
                        line = line.strip()

                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix

                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)

                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    choice = chunk["choices"][0]
                                    delta = choice.get("delta", {})
                                    if not isinstance(delta, dict):
                                        delta = {}
                                    content = delta.get("content", "")
                                    tool_calls = delta.get("tool_calls") or choice.get("tool_calls")
                                    finish_reason = choice.get("finish_reason")

                                    yield GenerateResponse(
                                        content=content,
                                        model=self.model,
                                        finish_reason=finish_reason,
                                        tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                                        raw_response=chunk
                                    )

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    async def _agenerate_internal(self,
                                   prompt: str,
                                   messages: Optional[List[Dict[str, str]]] = None,
                                   system_prompt: Optional[str] = None,
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   media: Optional[List['MediaContent']] = None,
                                   stream: bool = False,
                                   response_model: Optional[Type[BaseModel]] = None,
                                   execute_tools: Optional[bool] = None,
                                   tool_call_tags: Optional[str] = None,
                                   # vLLM-specific parameters:
                                   guided_regex: Optional[str] = None,
                                   guided_json: Optional[Dict] = None,
                                   guided_grammar: Optional[str] = None,
                                   best_of: Optional[int] = None,
                                   use_beam_search: bool = False,
                                   **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Native async implementation with vLLM features."""

        # Build messages (same logic as sync)
        chat_messages = []

        final_system_prompt = system_prompt
        # Prefer native tools when available; only inject prompted tool syntax as fallback.
        if tools and self.tool_handler.supports_prompted and not self.tool_handler.supports_native:
            include_tool_list = True
            if final_system_prompt and "## Tools (session)" in final_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if final_system_prompt:
                final_system_prompt += f"\n\n{tool_prompt}"
            else:
                final_system_prompt = tool_prompt

        if final_system_prompt:
            chat_messages.append({"role": "system", "content": final_system_prompt})

        if messages:
            chat_messages.extend(messages)

        if media:
            user_message_text = prompt.strip() if prompt else ""
            if not user_message_text and chat_messages:
                for msg in reversed(chat_messages):
                    if msg.get("role") == "user" and msg.get("content"):
                        user_message_text = msg["content"]
                        break
            try:
                processed_media = self._process_media_content(media)
                media_handler = self._get_media_handler_for_model(self.model)
                multimodal_message = media_handler.create_multimodal_message(user_message_text, processed_media)

                if isinstance(multimodal_message, str):
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1]["content"] = multimodal_message
                    else:
                        chat_messages.append({"role": "user", "content": multimodal_message})
                else:
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1] = multimodal_message
                    else:
                        chat_messages.append(multimodal_message)
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install abstractcore[media]")
                if user_message_text:
                    chat_messages.append({"role": "user", "content": user_message_text})
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                if user_message_text:
                    chat_messages.append({"role": "user", "content": user_message_text})

        elif prompt and prompt.strip():
            chat_messages.append({"role": "user", "content": prompt})

        # Build request payload
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": max_output_tokens,
            "top_p": kwargs.get("top_p", 0.9),
        }

        # Native tools (OpenAI-compatible): send structured tools/tool_choice when supported.
        if tools and self.tool_handler.supports_native:
            payload["tools"] = self.tool_handler.prepare_tools_for_native(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]

        seed_value = kwargs.get("seed", self.seed)
        if seed_value is not None:
            payload["seed"] = seed_value

        # vLLM-specific features
        extra_body = {}

        if guided_regex:
            extra_body["guided_regex"] = guided_regex
        if guided_json:
            extra_body["guided_json"] = guided_json
        if guided_grammar:
            extra_body["guided_grammar"] = guided_grammar

        if use_beam_search or best_of:
            extra_body["use_beam_search"] = use_beam_search
            if best_of:
                extra_body["best_of"] = best_of

        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        if extra_body:
            payload["extra_body"] = extra_body

        if stream:
            return self._async_stream_generate(payload)
        else:
            response = await self._async_single_generate(payload)

            if self.execute_tools and tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools, execute_tools)

            return response

    async def _async_single_generate(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Native async single response generation."""
        try:
            start_time = time.time()
            response = await self.async_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            gen_time = round((time.time() - start_time) * 1000, 1)

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason", "stop")
            else:
                content = "No response generated"
                finish_reason = "error"

            usage = result.get("usage", {})

            return GenerateResponse(
                content=content,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=result,
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                },
                gen_time=gen_time
            )

        except Exception as e:
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str or 'model' in error_str) and ('not found' in error_str):
                try:
                    available_models = self.list_available_models(base_url=self.base_url)
                    error_message = format_model_error("vLLM", self.model, available_models)
                    raise ModelNotFoundError(error_message)
                except Exception:
                    raise ModelNotFoundError(f"Model '{self.model}' not found in vLLM")
            else:
                raise ProviderAPIError(f"vLLM API error: {str(e)}")

    async def _async_stream_generate(self, payload: Dict[str, Any]) -> AsyncIterator[GenerateResponse]:
        """Native async streaming response generation."""
        try:
            async with self.async_client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._get_headers()
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        line = line.strip()

                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix

                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)

                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    choice = chunk["choices"][0]
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                    finish_reason = choice.get("finish_reason")

                                    yield GenerateResponse(
                                        content=content,
                                        model=self.model,
                                        finish_reason=finish_reason,
                                        raw_response=chunk
                                    )

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    # vLLM-specific methods

    def load_adapter(self, adapter_name: str, adapter_path: str) -> str:
        """
        Load a LoRA adapter dynamically without restarting the server.

        Args:
            adapter_name: Name to identify the adapter (e.g., "sql-expert")
            adapter_path: Path to the LoRA adapter weights

        Returns:
            Success message

        Usage:
            llm.load_adapter("sql-expert", "/models/adapters/sql-lora")
            response = llm.generate("Query...", model="sql-expert")
        """
        management_url = self.base_url.rstrip('/').replace('/v1', '')

        response = self.client.post(
            f"{management_url}/v1/load_lora_adapter",
            json={"lora_name": adapter_name, "lora_path": adapter_path},
            headers=self._get_headers()
        )
        response.raise_for_status()
        return f"Adapter '{adapter_name}' loaded successfully"

    def unload_adapter(self, adapter_name: str) -> str:
        """Unload a LoRA adapter from memory."""
        management_url = self.base_url.rstrip('/').replace('/v1', '')

        response = self.client.post(
            f"{management_url}/v1/unload_lora_adapter",
            json={"lora_name": adapter_name},
            headers=self._get_headers()
        )
        response.raise_for_status()
        return f"Adapter '{adapter_name}' unloaded successfully"

    def list_adapters(self) -> List[str]:
        """List currently loaded LoRA adapters."""
        management_url = self.base_url.rstrip('/').replace('/v1', '')

        response = self.client.get(
            f"{management_url}/v1/lora_adapters",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json().get("adapters", [])

    # Standard AbstractCore methods

    def get_capabilities(self) -> List[str]:
        """Get vLLM capabilities."""
        capabilities = ["streaming", "chat", "tools", "structured_output"]
        # vLLM-specific capabilities
        capabilities.extend(["guided_decoding", "multi_lora", "beam_search"])
        return capabilities

    def validate_config(self) -> bool:
        """Validate vLLM connection."""
        try:
            response = self.client.get(f"{self.base_url}/models", headers=self._get_headers())
            return response.status_code == 200
        except:
            return False

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for vLLM API."""
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _update_http_client_timeout(self) -> None:
        """Update HTTP client timeout when timeout is changed."""
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.close()

                timeout_value = getattr(self, '_timeout', None)
                if timeout_value is not None and timeout_value <= 0:
                    timeout_value = None

                self.client = httpx.Client(timeout=timeout_value)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Failed to update HTTP client timeout: {e}")
                try:
                    fallback_timeout = None
                    try:
                        from ..config.manager import get_config_manager

                        fallback_timeout = float(get_config_manager().get_default_timeout())
                    except Exception:
                        fallback_timeout = 7200.0
                    if isinstance(fallback_timeout, (int, float)) and float(fallback_timeout) <= 0:
                        fallback_timeout = None
                    self.client = httpx.Client(timeout=fallback_timeout)
                except Exception:
                    pass

    def _normalize_model_name(self, model_name: str) -> str:
        """Remove common provider prefixes from model name."""
        for prefix in ["vllm/", "qwen/", "ollama/", "huggingface/"]:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):]
        return model_name

    def _get_media_handler_for_model(self, model_name: str):
        """Get appropriate media handler based on model vision capabilities."""
        from ..media.handlers import OpenAIMediaHandler, LocalMediaHandler

        clean_model_name = self._normalize_model_name(model_name)

        try:
            from ..architectures.detection import supports_vision
            use_vision_handler = supports_vision(clean_model_name)
        except Exception as e:
            self.logger.debug(f"Vision detection failed: {e}, defaulting to LocalMediaHandler")
            use_vision_handler = False

        if use_vision_handler:
            handler = OpenAIMediaHandler(self.model_capabilities, model_name=model_name)
            self.logger.debug(f"Using OpenAIMediaHandler for vision model: {clean_model_name}")
        else:
            handler = LocalMediaHandler("vllm", self.model_capabilities, model_name=model_name)
            self.logger.debug(f"Using LocalMediaHandler for model: {clean_model_name}")

        return handler

    def list_available_models(self, **kwargs) -> List[str]:
        """
        List available models from vLLM server.

        Args:
            **kwargs: Optional parameters including:
                - base_url: vLLM server URL
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            from .model_capabilities import filter_models_by_capabilities

            base_url = kwargs.get('base_url', self.base_url)

            response = self.client.get(f"{base_url}/models", headers=self._get_headers(), timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data.get("data", [])]
                models = sorted(models)

                # Apply capability filtering if provided
                input_capabilities = kwargs.get('input_capabilities')
                output_capabilities = kwargs.get('output_capabilities')

                if input_capabilities or output_capabilities:
                    models = filter_models_by_capabilities(
                        models,
                        input_capabilities=input_capabilities,
                        output_capabilities=output_capabilities
                    )

                return models
            else:
                self.logger.warning(f"vLLM API returned status {response.status_code}")
                return []
        except Exception as e:
            self.logger.warning(f"Failed to list vLLM models: {e}")
            return []

    def embed(self, input_text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings using vLLM's OpenAI-compatible embedding API.

        Args:
            input_text: Single string or list of strings to embed
            **kwargs: Additional parameters (encoding_format, dimensions, user, etc.)

        Returns:
            Dict with embeddings in OpenAI-compatible format:
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...],
                "model": "model-name",
                "usage": {"prompt_tokens": N, "total_tokens": N}
            }
        """
        try:
            payload = {
                "input": input_text,
                "model": self.model
            }

            if "encoding_format" in kwargs:
                payload["encoding_format"] = kwargs["encoding_format"]
            if "dimensions" in kwargs and kwargs["dimensions"]:
                payload["dimensions"] = kwargs["dimensions"]
            if "user" in kwargs:
                payload["user"] = kwargs["user"]

            response = self.client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()

            result = response.json()
            result["model"] = self.model

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise ProviderAPIError(f"vLLM embedding error: {str(e)}")
