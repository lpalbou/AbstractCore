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


def _inline_json_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline local $defs/$ref references in a JSON Schema dict.

    Some OpenAI-compatible servers only partially support `$defs`/`$ref` inside
    `response_format: {type:'json_schema'}`. Inlining keeps schemas simple and
    improves compatibility for structured outputs.
    """

    defs = schema.get("$defs")
    if not isinstance(defs, dict) or not defs:
        return schema

    def _resolve(node: Any, *, seen: set[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref[len("#/$defs/"):]
                target = defs.get(key)
                if isinstance(key, str) and key and isinstance(target, dict):
                    if key in seen:
                        return node
                    seen.add(key)
                    resolved_target = _resolve(dict(target), seen=seen)
                    seen.remove(key)
                    if isinstance(resolved_target, dict):
                        merged: Dict[str, Any] = dict(resolved_target)
                        for k, v in node.items():
                            if k == "$ref":
                                continue
                            merged[k] = _resolve(v, seen=seen)
                        return merged

            out: Dict[str, Any] = {}
            for k, v in node.items():
                if k == "$defs":
                    continue
                out[k] = _resolve(v, seen=seen)
            return out

        if isinstance(node, list):
            return [_resolve(x, seen=seen) for x in node]

        return node

    try:
        base = {k: v for k, v in schema.items() if k != "$defs"}
        inlined = _resolve(base, seen=set())
        return inlined if isinstance(inlined, dict) and inlined else schema
    except Exception:
        return schema
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import (
    ProviderAPIError,
    ModelNotFoundError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    format_model_error,
)
from ..tools import UniversalToolHandler


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
                        base_url="http://127.0.0.1:1234/v1",
                        model="llama-3.1-8b")

        # With API key (optional for many local servers)
        llm = create_llm("openai-compatible",
                        base_url="http://127.0.0.1:1234/v1",
                        model="my-model",
                        api_key="your-key")

        # Environment variable configuration
        export OPENAI_COMPATIBLE_BASE_URL="http://127.0.0.1:1234/v1"
        export OPENAI_COMPATIBLE_API_KEY="your-key"  # Optional
        llm = create_llm("openai-compatible", model="my-model")
    """

    PROVIDER_ID = "openai-compatible"
    PROVIDER_DISPLAY_NAME = "OpenAI-compatible server"
    BASE_URL_ENV_VAR = "OPENAI_COMPATIBLE_BASE_URL"
    API_KEY_ENV_VAR = "OPENAI_COMPATIBLE_API_KEY"
    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(self, model: str = "default", base_url: Optional[str] = None,
                 api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.provider = self.PROVIDER_ID

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        self.base_url = self._resolve_base_url(base_url)

        self.api_key = self._resolve_api_key(api_key)

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
                raise RuntimeError(f"Failed to create HTTP client for {self.PROVIDER_DISPLAY_NAME}: {e}")

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
        # Only add Authorization header if api_key is provided and meaningful.
        api_key = None if self.api_key is None else str(self.api_key).strip()
        if api_key and api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _mutate_payload(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Provider-specific payload hook (default: no-op)."""
        return payload

    def _resolve_base_url(self, base_url: Optional[str]) -> str:
        """Resolve base URL with parameter > env var > default precedence."""
        if base_url is not None:
            resolved = str(base_url).strip()
            if not resolved:
                raise ValueError("base_url cannot be empty")
            return resolved.rstrip("/")

        env_var = getattr(self, "BASE_URL_ENV_VAR", None)
        env_val = os.getenv(env_var) if isinstance(env_var, str) and env_var else None
        if isinstance(env_val, str) and env_val.strip():
            return env_val.strip().rstrip("/")

        default = getattr(self, "DEFAULT_BASE_URL", None) or ""
        return str(default).strip().rstrip("/")

    def _resolve_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """Resolve API key with parameter > env var > config fallback."""
        if api_key is not None:
            # Allow callers to explicitly disable auth by passing an empty string.
            return api_key

        env_var = getattr(self, "API_KEY_ENV_VAR", None)
        env_val = os.getenv(env_var) if isinstance(env_var, str) and env_var else None
        if env_val is not None:
            return env_val

        return self._get_api_key_from_config()

    def _get_api_key_from_config(self) -> Optional[str]:
        """Optional config-manager fallback for subclasses (default: none)."""
        return None

    def _extract_error_detail(self, response: Optional[httpx.Response]) -> Optional[str]:
        """Extract a useful error message from an HTTPX response, if possible."""
        if response is None:
            return None

        try:
            data = response.json()
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    for k in ("message", "error", "detail"):
                        v = err.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                for k in ("message", "detail"):
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            # If it's JSON but not a dict, stringify it.
            if data is not None:
                return json.dumps(data, ensure_ascii=False)
        except Exception:
            pass

        try:
            text = response.text
            if isinstance(text, str) and text.strip():
                # Bound size to avoid dumping huge error bodies.
                return text.strip()[:2000]
        except Exception:
            pass

        return None

    def _raise_for_status(self, response: httpx.Response, *, request_url: Optional[str] = None) -> None:
        """Raise rich provider exceptions on HTTP errors."""
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            # Unit tests sometimes stub the HTTP response with only `.raise_for_status()`/`.json()`.
            # Treat as success if `.raise_for_status()` does not raise.
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            return

        if int(status_code) < 400:
            return

        detail = self._extract_error_detail(response)
        prefix = f"{self.PROVIDER_DISPLAY_NAME} API error ({status_code})"
        msg = f"{prefix}: {detail}" if detail else prefix

        status = int(status_code)
        if status in (401, 403):
            raise AuthenticationError(msg)
        if status == 429:
            raise RateLimitError(msg)
        if status == 400:
            # Many OpenAI-compatible servers use 400 for schema/model errors.
            if detail and ("model" in detail.lower()) and ("not found" in detail.lower()):
                self._raise_model_not_found()
            raise InvalidRequestError(msg)
        if status == 404:
            # Could be endpoint misconfiguration (missing /v1) or an unknown model.
            if detail and ("model" in detail.lower()) and ("not found" in detail.lower()):
                self._raise_model_not_found()
            raise ProviderAPIError(msg if request_url is None else f"{msg} [{request_url}]")

        raise ProviderAPIError(msg if request_url is None else f"{msg} [{request_url}]")

    def _raise_model_not_found(self) -> None:
        """Raise ModelNotFoundError with a best-effort available-model list."""
        try:
            available_models = self.list_available_models(base_url=self.base_url)
        except Exception:
            available_models = []
        raise ModelNotFoundError(format_model_error(self.PROVIDER_DISPLAY_NAME, self.model, available_models))

    def _validate_model(self):
        """Validate that the model exists on the server (best-effort)."""
        # Skip validation for "default" placeholder (used by registry for model listing)
        if self.model == "default":
            return

        try:
            # Use base_url as-is (should include /v1) for model discovery
            available_models = self.list_available_models(base_url=self.base_url)
            if available_models and self.model not in available_models:
                error_message = format_model_error(self.PROVIDER_DISPLAY_NAME, self.model, available_models)
                raise ModelNotFoundError(error_message)
        except httpx.ConnectError:
            # Server not running - will fail later when trying to generate
            if hasattr(self, 'logger'):
                self.logger.debug(f"{self.PROVIDER_DISPLAY_NAME} not accessible at {self.base_url} - model validation skipped")
            pass
        except ModelNotFoundError:
            # Re-raise model not found errors
            raise
        except Exception as e:
            # Other errors (like timeout, None type errors) - continue, will fail later if needed
            if hasattr(self, 'logger'):
                self.logger.debug(f"Model validation failed with error: {e} - continuing anyway")
            pass

    def unload_model(self, model_name: str) -> None:
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
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "max_tokens": max_output_tokens,
            "top_p": kwargs.get("top_p", 0.9),
        }

        # Prompt caching (best-effort): pass through `prompt_cache_key` when provided.
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            payload["prompt_cache_key"] = prompt_cache_key.strip()

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
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            payload["seed"] = seed_value

        # Add structured output support (OpenAI-compatible format)
        # Many servers support native structured outputs using the response_format parameter
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            if isinstance(json_schema, dict) and json_schema:
                json_schema = _inline_json_schema_refs(json_schema)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        # Provider-specific request extensions (vLLM extra_body, OpenRouter headers, etc.)
        payload = self._mutate_payload(payload, **kwargs)

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
            self._raise_for_status(response, request_url=request_url)
            gen_time = round((time.time() - start_time) * 1000, 1)

            result = response.json()

            # Extract response from OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    message = {}

                content = message.get("content", "")
                reasoning = message.get("reasoning")
                tool_calls = message.get("tool_calls")
                if tool_calls is None:
                    # Some servers surface tool calls at the choice level.
                    tool_calls = choice.get("tool_calls")
                finish_reason = choice.get("finish_reason", "stop")
            else:
                content = "No response generated"
                reasoning = None
                tool_calls = None
                finish_reason = "error"

            # Extract usage info
            usage = result.get("usage", {})

            metadata: Dict[str, Any] = {
                "_provider_request": {
                    "url": request_url,
                    "payload": payload,
                }
            }
            if isinstance(reasoning, str) and reasoning.strip():
                metadata["reasoning"] = reasoning

            return GenerateResponse(
                content=content,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=result,
                tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                metadata=metadata,
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
                raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} not properly initialized: {str(e)}")
            else:
                raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} configuration error: {str(e)}")
        except Exception as e:
            error_str = str(e).lower()
            if ("not found" in error_str) and ("model" in error_str):
                self._raise_model_not_found()
            raise

    def _stream_generate(self, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Generate streaming response"""
        request_url = f"{self.base_url}/chat/completions"

        with self.client.stream(
            "POST",
            request_url,
            json=payload,
            headers=self._get_headers()
        ) as response:
            self._raise_for_status(response, request_url=request_url)

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
                                reasoning = delta.get("reasoning")
                                tool_calls = delta.get("tool_calls") or choice.get("tool_calls")
                                finish_reason = choice.get("finish_reason")

                                metadata = {}
                                if isinstance(reasoning, str) and reasoning.strip():
                                    metadata["reasoning"] = reasoning

                                yield GenerateResponse(
                                    content=content,
                                    model=self.model,
                                    finish_reason=finish_reason,
                                    tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                                    metadata=metadata or None,
                                    raw_response=chunk
                                )

                        except json.JSONDecodeError:
                            continue

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
            "temperature": generation_kwargs.get("temperature", self.temperature),
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
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            payload["seed"] = seed_value

        # Add structured output support
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            if isinstance(json_schema, dict) and json_schema:
                json_schema = _inline_json_schema_refs(json_schema)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        # Provider-specific request extensions (vLLM extra_body, OpenRouter headers, etc.)
        payload = self._mutate_payload(payload, **kwargs)

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
            self._raise_for_status(response, request_url=request_url)
            gen_time = round((time.time() - start_time) * 1000, 1)

            result = response.json()

            # Extract response from OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    message = {}

                content = message.get("content", "")
                reasoning = message.get("reasoning")
                tool_calls = message.get("tool_calls")
                if tool_calls is None:
                    tool_calls = choice.get("tool_calls")
                finish_reason = choice.get("finish_reason", "stop")
            else:
                content = "No response generated"
                reasoning = None
                tool_calls = None
                finish_reason = "error"

            # Extract usage info
            usage = result.get("usage", {})

            metadata: Dict[str, Any] = {
                "_provider_request": {
                    "url": request_url,
                    "payload": payload,
                }
            }
            if isinstance(reasoning, str) and reasoning.strip():
                metadata["reasoning"] = reasoning

            return GenerateResponse(
                content=content,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=result,
                tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                metadata=metadata,
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                },
                gen_time=gen_time
            )

        except (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError, ProviderAPIError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            if ("not found" in error_str) and ("model" in error_str):
                self._raise_model_not_found()
            raise

    async def _async_stream_generate(self, payload: Dict[str, Any]) -> AsyncIterator[GenerateResponse]:
        """Native async streaming response generation."""
        request_url = f"{self.base_url}/chat/completions"

        async with self.async_client.stream(
            "POST",
            request_url,
            json=payload,
            headers=self._get_headers()
        ) as response:
            self._raise_for_status(response, request_url=request_url)

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
                                if not isinstance(delta, dict):
                                    delta = {}
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning")
                                tool_calls = delta.get("tool_calls") or choice.get("tool_calls")
                                finish_reason = choice.get("finish_reason")

                                metadata = {}
                                if isinstance(reasoning, str) and reasoning.strip():
                                    metadata["reasoning"] = reasoning

                                yield GenerateResponse(
                                    content=content,
                                    model=self.model,
                                    finish_reason=finish_reason,
                                    tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                                    metadata=metadata or None,
                                    raw_response=chunk
                                )

                        except json.JSONDecodeError:
                            continue

    def supports_prompt_cache(self) -> bool:
        """Best-effort: forward `prompt_cache_key` to OpenAI-compatible servers that support it."""
        return True

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

    def _get_media_handler_for_model(self, model_name: str):
        """Get appropriate media handler based on model vision capabilities."""
        from ..media.handlers import OpenAIMediaHandler, LocalMediaHandler

        # Determine if model supports vision
        try:
            from ..architectures.detection import supports_vision
            use_vision_handler = supports_vision(model_name)
        except Exception as e:
            self.logger.debug(f"Vision detection failed: {e}, defaulting to LocalMediaHandler")
            use_vision_handler = False

        # Create appropriate handler
        if use_vision_handler:
            handler = OpenAIMediaHandler(self.model_capabilities, model_name=model_name)
            self.logger.debug(f"Using OpenAIMediaHandler for vision model: {model_name}")
        else:
            handler = LocalMediaHandler(self.provider, self.model_capabilities, model_name=model_name)
            self.logger.debug(f"Using LocalMediaHandler for model: {model_name}")

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
                detail = self._extract_error_detail(response)
                suffix = f": {detail}" if detail else ""
                self.logger.warning(f"{self.PROVIDER_DISPLAY_NAME} /models returned {response.status_code}{suffix}")
                return []
        except Exception as e:
            self.logger.warning(f"Failed to list models from {self.PROVIDER_DISPLAY_NAME}: {e}")
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
            self._raise_for_status(response, request_url=f"{self.base_url}/embeddings")

            # Server returns OpenAI-compatible format
            result = response.json()

            # Ensure the model field uses our provider-prefixed format
            result["model"] = self.model

            return result

        except (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError, ProviderAPIError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} embedding error: {str(e)}")
