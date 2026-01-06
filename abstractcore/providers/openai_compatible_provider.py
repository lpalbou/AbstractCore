"""
Generic OpenAI-compatible provider for any OpenAI-compatible API endpoint.

Supports any server implementing the OpenAI API format:
- llama.cpp server
- text-generation-webui (with OpenAI extension)
- LocalAI
- FastChat
- Aphrodite
- SGLang
- Custom deployments and proxies
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


class OpenAICompatibleProvider(BaseProvider):
    """
    Generic provider for any OpenAI-compatible API endpoint.

    Works with any server implementing the OpenAI API format:
    - llama.cpp server
    - text-generation-webui (OpenAI extension)
    - LocalAI
    - FastChat
    - Aphrodite
    - SGLang
    - Custom deployments and proxies

    Usage:
        # Basic usage
        llm = create_llm("openai-compatible",
                        base_url="http://localhost:8080/v1",
                        model="llama-3.1-8b")

        # With API key (optional for many local servers)
        llm = create_llm("openai-compatible",
                        base_url="http://localhost:8080/v1",
                        model="my-model",
                        api_key="your-key")

        # Environment variable configuration
        export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8080/v1"
        export OPENAI_COMPATIBLE_API_KEY="your-key"  # Optional
        llm = create_llm("openai-compatible", model="my-model")
    """

    def __init__(self, model: str = "default", base_url: Optional[str] = None,
                 api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "openai-compatible"

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Base URL priority: parameter > OPENAI_COMPATIBLE_BASE_URL > default
        self.base_url = (
            base_url or
            os.getenv("OPENAI_COMPATIBLE_BASE_URL") or
            "http://localhost:8080/v1"
        ).rstrip('/')

        # API key: OPTIONAL (many local servers don't require authentication)
        # Priority: parameter > OPENAI_COMPATIBLE_API_KEY > None
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")

        # Get timeout value - None means unlimited timeout
        timeout_value = getattr(self, '_timeout', None)
        # Validate timeout if provided (None is allowed for unlimited)
        if timeout_value is not None and timeout_value <= 0:
            timeout_value = None  # Invalid timeout becomes unlimited

        try:
            self.client = httpx.Client(timeout=timeout_value)
        except Exception as e:
            # Fallback with default timeout if client creation fails
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
                raise RuntimeError(f"Failed to create HTTP client for OpenAI-compatible provider: {e}")

        self._async_client = None  # Lazy-loaded async client

        # Validate model exists on server
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
        """Get HTTP headers with optional API key authentication."""
        headers = {"Content-Type": "application/json"}
        # Only add Authorization header if api_key is provided and truthy
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _validate_model(self):
        """Validate that the model exists on the OpenAI-compatible server"""
        # Skip validation for "default" placeholder (used by registry for model listing)
        if self.model == "default":
            return

        try:
            # Use base_url as-is (should include /v1) for model discovery
            available_models = self.list_available_models(base_url=self.base_url)
            if available_models and self.model not in available_models:
                error_message = format_model_error("OpenAI-compatible server", self.model, available_models)
                raise ModelNotFoundError(error_message)
        except httpx.ConnectError:
            # Server not running - will fail later when trying to generate
            if hasattr(self, 'logger'):
                self.logger.debug(f"OpenAI-compatible server not accessible at {self.base_url} - model validation skipped")
            pass
        except ModelNotFoundError:
            # Re-raise model not found errors
            raise
        except Exception as e:
            # Other errors (like timeout, None type errors) - continue, will fail later if needed
            if hasattr(self, 'logger'):
                self.logger.debug(f"Model validation failed with error: {e} - continuing anyway")
            pass

    def unload(self) -> None:
        """
        Close HTTP client connection.

        Note: Most OpenAI-compatible servers manage model memory automatically.
        This method only closes the HTTP client connection for cleanup.
        """
        try:
            # Close the HTTP client connection
            if hasattr(self, 'client') and self.client is not None:
                self.client.close()

            # Close async client if it was created
            if self._async_client is not None:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._async_client.aclose())
                except RuntimeError:
                    # No running loop
                    import asyncio
                    asyncio.run(self._async_client.aclose())

        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
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
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using OpenAI-compatible server"""

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

        # Handle media content regardless of prompt (media can be used with messages too)
        if media:
            # Get the last user message content to combine with media
            user_message_text = prompt.strip() if prompt else ""
            if not user_message_text and chat_messages:
                # If no prompt, try to get text from the last user message
                for msg in reversed(chat_messages):
                    if msg.get("role") == "user" and msg.get("content"):
                        user_message_text = msg["content"]
                        break
            try:
                # Process media files into MediaContent objects first
                processed_media = self._process_media_content(media)

                # Use capability-based media handler selection
                media_handler = self._get_media_handler_for_model(self.model)

                # Create multimodal message combining text and processed media
                multimodal_message = media_handler.create_multimodal_message(user_message_text, processed_media)

                # For OpenAI-compatible servers, we might get a string (embedded text) or dict (structured)
                if isinstance(multimodal_message, str):
                    # Replace the last user message with the multimodal message, or add new one
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1]["content"] = multimodal_message
                    else:
                        chat_messages.append({
                            "role": "user",
                            "content": multimodal_message
                        })
                else:
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        # Replace last user message with structured multimodal message
                        chat_messages[-1] = multimodal_message
                    else:
                        chat_messages.append(multimodal_message)
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install abstractcore[media]")
                if user_message_text:
                    chat_messages.append({
                        "role": "user",
                        "content": user_message_text
                    })
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                if user_message_text:
                    chat_messages.append({
                        "role": "user",
                        "content": user_message_text
                    })

        # Add prompt as separate message if provided (for backward compatibility)
        elif prompt and prompt.strip():
            chat_messages.append({
                "role": "user",
                "content": prompt
            })

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

        # Add additional generation parameters if provided (OpenAI-compatible)
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "repetition_penalty" in kwargs:
            # Some models support repetition_penalty directly
            payload["repetition_penalty"] = kwargs["repetition_penalty"]

        # Add seed if provided (many servers support seed via OpenAI-compatible API)
        seed_value = kwargs.get("seed", self.seed)
        if seed_value is not None:
            payload["seed"] = seed_value

        # Add structured output support (OpenAI-compatible format)
        # Many servers support native structured outputs using the response_format parameter
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        if stream:
            # Return streaming response - BaseProvider will handle tag rewriting via UnifiedStreamProcessor
            return self._stream_generate(payload)
        else:
            response = self._single_generate(payload)

            # Execute tools if enabled and tools are present
            if self.execute_tools and tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools, execute_tools)

            return response

    def _single_generate(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Generate single response"""
        try:
            # Ensure client is available
            if not hasattr(self, 'client') or self.client is None:
                raise ProviderAPIError("HTTP client not initialized")

            # Track generation time
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
                    # Some servers surface tool calls at the choice level.
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
                    # Keep legacy keys for backward compatibility
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                },
                gen_time=gen_time
            )

        except AttributeError as e:
            # Handle None type errors specifically
            if "'NoneType'" in str(e):
                raise ProviderAPIError(f"OpenAI-compatible provider not properly initialized: {str(e)}")
            else:
                raise ProviderAPIError(f"OpenAI-compatible provider configuration error: {str(e)}")
        except Exception as e:
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str or 'model' in error_str) and ('not found' in error_str):
                # Model not found - show available models
                try:
                    available_models = self.list_available_models(base_url=self.base_url)
                    error_message = format_model_error("OpenAI-compatible server", self.model, available_models)
                    raise ModelNotFoundError(error_message)
                except Exception:
                    # If model discovery also fails, provide a generic error
                    raise ModelNotFoundError(f"Model '{self.model}' not found on OpenAI-compatible server and could not fetch available models")
            else:
                raise

    def _stream_generate(self, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Generate streaming response"""
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
                        # Decode bytes to string if necessary
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
                                   **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Native async implementation using httpx.AsyncClient - 3-10x faster for batch operations."""

        # Build messages for chat completions with tool support (same logic as sync)
        chat_messages = []

        # Add tools to system prompt if provided
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

        # Add system message if provided
        if final_system_prompt:
            chat_messages.append({
                "role": "system",
                "content": final_system_prompt
            })

        # Add conversation history
        if messages:
            chat_messages.extend(messages)

        # Handle media content
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

        # Add additional parameters
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]

        # Add seed if provided
        seed_value = kwargs.get("seed", self.seed)
        if seed_value is not None:
            payload["seed"] = seed_value

        # Add structured output support
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        if stream:
            return self._async_stream_generate(payload)
        else:
            response = await self._async_single_generate(payload)

            # Execute tools if enabled
            if self.execute_tools and tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools, execute_tools)

            return response

    async def _async_single_generate(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Native async single response generation."""
        try:
            # Track generation time
            start_time = time.time()
            request_url = f"{self.base_url}/chat/completions"
            response = await self.async_client.post(
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
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason", "stop")
            else:
                content = "No response generated"
                finish_reason = "error"

            # Extract usage info
            usage = result.get("usage", {})

            return GenerateResponse(
                content=content,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=result,
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

        except Exception as e:
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str or 'model' in error_str) and ('not found' in error_str):
                try:
                    available_models = self.list_available_models(base_url=self.base_url)
                    error_message = format_model_error("OpenAI-compatible server", self.model, available_models)
                    raise ModelNotFoundError(error_message)
                except Exception:
                    raise ModelNotFoundError(f"Model '{self.model}' not found on OpenAI-compatible server")
            else:
                raise ProviderAPIError(f"OpenAI-compatible server API error: {str(e)}")

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

    def get_capabilities(self) -> List[str]:
        """Get OpenAI-compatible server capabilities"""
        return ["streaming", "chat", "tools"]

    def validate_config(self) -> bool:
        """Validate OpenAI-compatible server connection"""
        try:
            response = self.client.get(f"{self.base_url}/models", headers=self._get_headers())
            return response.status_code == 200
        except:
            return False

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for OpenAI-compatible API"""
        # For OpenAI-compatible servers, max_tokens is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _update_http_client_timeout(self) -> None:
        """Update HTTP client timeout when timeout is changed."""
        if hasattr(self, 'client') and self.client is not None:
            try:
                # Create new client with updated timeout
                self.client.close()

                # Get timeout value - None means unlimited timeout
                timeout_value = getattr(self, '_timeout', None)
                # Validate timeout if provided (None is allowed for unlimited)
                if timeout_value is not None and timeout_value <= 0:
                    timeout_value = None  # Invalid timeout becomes unlimited

                self.client = httpx.Client(timeout=timeout_value)
            except Exception as e:
                # Log error but don't fail - timeout update is not critical
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Failed to update HTTP client timeout: {e}")
                # Try to create a new client with default timeout
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
                    pass  # Best effort - don't fail the operation

    def _normalize_model_name(self, model_name: str) -> str:
        """Remove common provider prefixes from model name."""
        for prefix in ["openai-compatible/", "lmstudio/", "qwen/", "ollama/", "huggingface/"]:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):]
        return model_name

    def _get_media_handler_for_model(self, model_name: str):
        """Get appropriate media handler based on model vision capabilities."""
        from ..media.handlers import OpenAIMediaHandler, LocalMediaHandler

        # Normalize model name by removing provider prefixes
        clean_model_name = self._normalize_model_name(model_name)

        # Determine if model supports vision
        try:
            from ..architectures.detection import supports_vision
            use_vision_handler = supports_vision(clean_model_name)
        except Exception as e:
            self.logger.debug(f"Vision detection failed: {e}, defaulting to LocalMediaHandler")
            use_vision_handler = False

        # Create appropriate handler
        if use_vision_handler:
            handler = OpenAIMediaHandler(self.model_capabilities, model_name=model_name)
            self.logger.debug(f"Using OpenAIMediaHandler for vision model: {clean_model_name}")
        else:
            handler = LocalMediaHandler("openai-compatible", self.model_capabilities, model_name=model_name)
            self.logger.debug(f"Using LocalMediaHandler for model: {clean_model_name}")

        return handler

    def list_available_models(self, **kwargs) -> List[str]:
        """
        List available models from OpenAI-compatible server.

        Args:
            **kwargs: Optional parameters including:
                - base_url: Server URL
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            from .model_capabilities import filter_models_by_capabilities

            # Use provided base_url or fall back to instance base_url
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
                self.logger.warning(f"OpenAI-compatible server API returned status {response.status_code}")
                return []
        except Exception as e:
            self.logger.warning(f"Failed to list models from OpenAI-compatible server: {e}")
            return []

    def embed(self, input_text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings using OpenAI-compatible embedding API.

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
            # Prepare request payload for OpenAI-compatible API
            payload = {
                "input": input_text,
                "model": self.model
            }

            # Add optional parameters if provided
            if "encoding_format" in kwargs:
                payload["encoding_format"] = kwargs["encoding_format"]
            if "dimensions" in kwargs and kwargs["dimensions"]:
                payload["dimensions"] = kwargs["dimensions"]
            if "user" in kwargs:
                payload["user"] = kwargs["user"]

            # Call server's embeddings API (OpenAI-compatible)
            response = self.client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()

            # Server returns OpenAI-compatible format
            result = response.json()

            # Ensure the model field uses our provider-prefixed format
            result["model"] = self.model

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise ProviderAPIError(f"OpenAI-compatible server embedding error: {str(e)}")
