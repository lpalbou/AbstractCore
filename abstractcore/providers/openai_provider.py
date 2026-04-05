"""
OpenAI provider implementation.
"""

import os
import json
import time
import warnings
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Type

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider, ThinkingControlHandling
from ..core.types import GenerateResponse
from ..exceptions import AuthenticationError, ProviderAPIError, ModelNotFoundError, format_model_error, format_auth_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIProvider(BaseProvider):
    """OpenAI API provider with full integration"""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "openai"

        # Track which generation params were explicitly provided at init so we can warn
        # only when callers intentionally requested unsupported behaviour.
        self._explicit_init_params = frozenset(
            key
            for key, value in kwargs.items()
            if key in {"temperature", "top_p", "frequency_penalty", "presence_penalty", "seed"} and value is not None
        )

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        # Get API key from param or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        # Get base URL from param or environment
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        # Initialize client with timeout and optional base_url
        client_kwargs = {"api_key": self.api_key, "timeout": self._timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = openai.OpenAI(**client_kwargs)
        self._async_client = None  # Lazy-loaded async client

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Preflight check: validate model exists
        self._validate_model_exists()

        # Store provider-specific configuration
        self.top_p = kwargs.get("top_p", 1.0)
        self.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _apply_provider_thinking_kwargs(
        self,
        *,
        enabled: Optional[bool],
        level: Optional[str],
        kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], ThinkingControlHandling]:
        # OpenAI Chat Completions exposes reasoning controls via `reasoning_effort`.
        #
        # Keep this mapping provider-local (BaseProvider normalizes thinking="none" -> enabled=False).
        reasoning_levels = self._model_reasoning_levels()
        if not reasoning_levels:
            return kwargs, ThinkingControlHandling()
        if enabled is None and level is None:
            return kwargs, ThinkingControlHandling()

        effort: Optional[str] = None

        # Explicit level wins.
        if level is not None:
            effort = level
        elif enabled is False:
            # Prefer true disable when supported; otherwise best-effort minimum.
            if "none" in reasoning_levels:
                effort = "none"
            else:
                fallback = next((x for x in ("minimal", "low", "medium", "high", "xhigh") if x in reasoning_levels), None)
                if not fallback:
                    return kwargs, ThinkingControlHandling()
                warnings.warn(
                    f"thinking='off' requested for model '{self.model}', but reasoning_effort cannot be fully "
                    f"disabled (supported: {reasoning_levels}); using reasoning_effort={fallback!r}.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                effort = fallback
        elif enabled is True:
            # Make the default explicit when the caller opts-in.
            if "medium" in reasoning_levels:
                effort = "medium"
            elif "high" in reasoning_levels:
                effort = "high"
            elif "low" in reasoning_levels:
                effort = "low"
            else:
                return kwargs, ThinkingControlHandling()

        if not effort:
            return kwargs, ThinkingControlHandling()

        new_kwargs = dict(kwargs)
        new_kwargs["reasoning_effort"] = effort
        return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=True)

    @property
    def async_client(self):
        """Lazy-load AsyncOpenAI client for native async operations."""
        if self._async_client is None:
            client_kwargs = {"api_key": self.api_key, "timeout": self._timeout}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._async_client = openai.AsyncOpenAI(**client_kwargs)
        return self._async_client

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with OpenAI API"""

        # Build messages array
        api_messages = []

        # Add system message if provided
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        if messages:
            for msg in messages:
                # Skip system messages as they're handled separately
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "system":
                    continue
                if not isinstance(role, str) or not role.strip():
                    continue
                content = msg.get("content")
                if content is None:
                    content = ""
                entry: Dict[str, Any] = {"role": role, "content": content}
                if role == "assistant":
                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        entry["tool_calls"] = tool_calls
                elif role == "tool":
                    tool_call_id = msg.get("tool_call_id")
                    tool_call_id_s = str(tool_call_id).strip() if tool_call_id is not None else ""
                    if tool_call_id_s:
                        entry["tool_call_id"] = tool_call_id_s
                    else:
                        # OpenAI requires tool messages to reference a preceding assistant tool call.
                        entry = {"role": "user", "content": f"[TOOL RESULT unknown]: {content}"}
                elif role == "function":
                    name = msg.get("name")
                    if isinstance(name, str) and name.strip():
                        entry["name"] = name.strip()
                api_messages.append(entry)

        media_enrichment = None

        # Handle media content regardless of prompt (media can be used with messages too)
        if media:
            # Get the last user message content to combine with media
            user_message_text = prompt.strip() if prompt else ""
            if not user_message_text and api_messages:
                # If no prompt, try to get text from the last user message
                for msg in reversed(api_messages):
                    if msg.get("role") == "user" and msg.get("content"):
                        user_message_text = msg["content"]
                        break

            replace_last_user = False
            if api_messages and api_messages[-1].get("role") == "user":
                last_content = api_messages[-1].get("content")
                # Only replace the last user message when prompt is empty (prompt already in messages)
                # or when the prompt is the same as the last user content (avoid duplication).
                if (not prompt.strip()) or (last_content == user_message_text):
                    replace_last_user = True

            try:
                # Process media files into MediaContent objects first
                processed_media = self._process_media_content(media)

                # Use capability-based media handler selection (vision vs fallback)
                media_handler = self._get_media_handler_for_model(self.model)

                # Create multimodal message combining text and processed media
                multimodal_message = media_handler.create_multimodal_message(user_message_text, processed_media)
                media_enrichment = getattr(media_handler, "media_enrichment", None)

                if isinstance(multimodal_message, str):
                    if replace_last_user:
                        api_messages[-1]["content"] = multimodal_message
                    else:
                        api_messages.append({"role": "user", "content": multimodal_message})
                else:
                    if replace_last_user:
                        api_messages[-1] = multimodal_message
                    else:
                        api_messages.append(multimodal_message)
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                if user_message_text and not replace_last_user:
                    api_messages.append({"role": "user", "content": user_message_text})
            except Exception as e:
                # Do not silently drop user-supplied media. Fail loudly so callers can
                # choose an explicit fallback policy (e.g. audio_policy='speech_to_text').
                from ..exceptions import UnsupportedFeatureError

                raise UnsupportedFeatureError(
                    f"OpenAI provider could not format attached media for model '{self.model}': {e}"
                ) from e

        # Add prompt as separate message if provided (for backward compatibility)
        elif prompt and prompt not in [msg.get("content") for msg in (messages or [])]:
            api_messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        call_params = {
            "model": self.model,
            "messages": api_messages,
            "stream": stream
        }

        # Prompt caching (OpenAI): best-effort pass-through via `prompt_cache_key`.
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            call_params["prompt_cache_key"] = prompt_cache_key.strip()
        prompt_cache_retention = kwargs.get("prompt_cache_retention")
        if isinstance(prompt_cache_retention, str) and prompt_cache_retention.strip():
            # Docs sometimes format this as `in-memory`; the API expects `in_memory`.
            retention = prompt_cache_retention.strip().lower().replace("-", "_")
            call_params["prompt_cache_retention"] = retention

        # Add generation parameters supported by this model (driven by model_capabilities.json).
        # Unsupported parameters are silently dropped — upstream callers (runtime, gateway)
        # always pass standard sampling params as defaults; warning on every call is noise.
        reasoning_effort = kwargs.get("reasoning_effort")
        reasoning_effort_s = (
            reasoning_effort.strip().lower() if isinstance(reasoning_effort, str) and reasoning_effort.strip() else None
        )
        reasoning_mode = bool(reasoning_effort_s and reasoning_effort_s != "none")

        explicit_sampling = self._explicit_init_params | {
            k for k, v in kwargs.items() if k in {"temperature", "top_p", "frequency_penalty", "presence_penalty", "seed"} and v is not None
        }

        sampling_params = {
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
        }
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            sampling_params["seed"] = seed_value

        for param, value in sampling_params.items():
            if self._is_parameter_supported(param):
                if reasoning_mode and param in {"temperature", "top_p", "frequency_penalty", "presence_penalty"}:
                    allowed_defaults = {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                    }
                    try:
                        allowed = allowed_defaults[param]
                        v = float(value)
                        if abs(v - float(allowed)) > 1e-9:
                            if param in explicit_sampling:
                                warnings.warn(
                                    f"OpenAI model '{self.model}' does not support setting {param} when "
                                    f"reasoning_effort={reasoning_effort_s!r}; dropping {param}={value!r}.",
                                    RuntimeWarning,
                                    stacklevel=3,
                                )
                            continue
                    except Exception:
                        if param in explicit_sampling:
                            warnings.warn(
                                f"OpenAI model '{self.model}' does not support setting {param} when "
                                f"reasoning_effort={reasoning_effort_s!r}; dropping {param}={value!r}.",
                                RuntimeWarning,
                                stacklevel=3,
                            )
                        continue
                call_params[param] = value

        if reasoning_effort_s:
            call_params["reasoning_effort"] = reasoning_effort_s

        # Output token parameter name (driven by model_capabilities.json)
        call_params[self._get_token_param_name()] = max_output_tokens

        # Add tools if provided (convert to native format)
        if tools:
            # Convert tools to native format for OpenAI API
            if self.tool_handler.supports_native:
                call_params["tools"] = self.tool_handler.prepare_tools_for_native(tools)
                call_params["tool_choice"] = kwargs.get("tool_choice", "auto")
            else:
                # Fallback to manual formatting
                call_params["tools"] = self._format_tools_for_openai(tools)
                call_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add structured output support (OpenAI native)
        if response_model and PYDANTIC_AVAILABLE:
            if self._supports_structured_output():
                json_schema = response_model.model_json_schema()

                # OpenAI requires additionalProperties: false for strict mode
                self._ensure_strict_schema(json_schema)

                call_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": json_schema
                    }
                }

        # Make API call with proper exception handling
        try:
            if stream:
                return self._stream_response(call_params, tools)
            else:
                # Track generation time
                start_time = time.time()
                response = self.client.chat.completions.create(**call_params)
                gen_time = round((time.time() - start_time) * 1000, 1)
                
                formatted = self._format_response(response)
                # Add generation time to response
                formatted.gen_time = gen_time
                # Runtime observability: capture the exact client payload we sent.
                formatted.metadata = dict(formatted.metadata or {})
                formatted.metadata["_provider_request"] = {"call_params": call_params}
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    formatted.metadata = merge_enrichment_metadata(formatted.metadata, media_enrichment)

                # Handle tool execution for OpenAI native responses
                if tools and formatted.has_tool_calls():
                    formatted = self._handle_tool_execution(formatted, tools)

                return formatted
        except Exception:
            # Let BaseProvider normalize (timeouts/auth/rate limits) consistently.
            raise

    async def _agenerate_internal(self,
                                   prompt: str,
                                   messages: Optional[List[Dict[str, str]]] = None,
                                   system_prompt: Optional[str] = None,
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   media: Optional[List['MediaContent']] = None,
                                   stream: bool = False,
                                   response_model: Optional[Type[BaseModel]] = None,
                                   **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Native async implementation using AsyncOpenAI - 3-10x faster for batch operations."""

        # Build messages array (same logic as sync)
        api_messages = []

        # Add system message if provided
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        if messages:
            for msg in messages:
                # Skip system messages as they're handled separately
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "system":
                    continue
                if not isinstance(role, str) or not role.strip():
                    continue
                content = msg.get("content")
                if content is None:
                    content = ""
                entry: Dict[str, Any] = {"role": role, "content": content}
                if role == "assistant":
                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        entry["tool_calls"] = tool_calls
                elif role == "tool":
                    tool_call_id = msg.get("tool_call_id")
                    tool_call_id_s = str(tool_call_id).strip() if tool_call_id is not None else ""
                    if tool_call_id_s:
                        entry["tool_call_id"] = tool_call_id_s
                    else:
                        entry = {"role": "user", "content": f"[TOOL RESULT unknown]: {content}"}
                elif role == "function":
                    name = msg.get("name")
                    if isinstance(name, str) and name.strip():
                        entry["name"] = name.strip()
                api_messages.append(entry)

        media_enrichment = None

        # Handle media content regardless of prompt (media can be used with messages too)
        if media:
            # Get the last user message content to combine with media
            user_message_text = prompt.strip() if prompt else ""
            if not user_message_text and api_messages:
                # If no prompt, try to get text from the last user message
                for msg in reversed(api_messages):
                    if msg.get("role") == "user" and msg.get("content"):
                        user_message_text = msg["content"]
                        break

            replace_last_user = False
            if api_messages and api_messages[-1].get("role") == "user":
                last_content = api_messages[-1].get("content")
                if (not prompt.strip()) or (last_content == user_message_text):
                    replace_last_user = True

            try:
                processed_media = self._process_media_content(media)
                media_handler = self._get_media_handler_for_model(self.model)
                multimodal_message = media_handler.create_multimodal_message(user_message_text, processed_media)
                media_enrichment = getattr(media_handler, "media_enrichment", None)

                if isinstance(multimodal_message, str):
                    if replace_last_user:
                        api_messages[-1]["content"] = multimodal_message
                    else:
                        api_messages.append({"role": "user", "content": multimodal_message})
                else:
                    if replace_last_user:
                        api_messages[-1] = multimodal_message
                    else:
                        api_messages.append(multimodal_message)

            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                if user_message_text and not replace_last_user:
                    api_messages.append({"role": "user", "content": user_message_text})
            except Exception as e:
                from ..exceptions import UnsupportedFeatureError

                raise UnsupportedFeatureError(
                    f"OpenAI provider could not format attached media for model '{self.model}': {e}"
                ) from e

        # Add prompt as separate message if provided (for backward compatibility)
        elif prompt and prompt not in [msg.get("content") for msg in (messages or [])]:
            api_messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters using unified system (same logic as sync)
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        call_params = {
            "model": self.model,
            "messages": api_messages,
            "stream": stream
        }

        # Prompt caching (OpenAI): best-effort pass-through via `prompt_cache_key`.
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            call_params["prompt_cache_key"] = prompt_cache_key.strip()
        prompt_cache_retention = kwargs.get("prompt_cache_retention")
        if isinstance(prompt_cache_retention, str) and prompt_cache_retention.strip():
            retention = prompt_cache_retention.strip().lower().replace("-", "_")
            call_params["prompt_cache_retention"] = retention

        # Add generation parameters supported by this model (driven by model_capabilities.json).
        # Unsupported parameters are silently dropped (same rationale as sync path).
        reasoning_effort = kwargs.get("reasoning_effort")
        reasoning_effort_s = (
            reasoning_effort.strip().lower() if isinstance(reasoning_effort, str) and reasoning_effort.strip() else None
        )
        reasoning_mode = bool(reasoning_effort_s and reasoning_effort_s != "none")

        explicit_sampling = self._explicit_init_params | {
            k for k, v in kwargs.items() if k in {"temperature", "top_p", "frequency_penalty", "presence_penalty", "seed"} and v is not None
        }

        sampling_params = {
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
        }
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            sampling_params["seed"] = seed_value

        for param, value in sampling_params.items():
            if self._is_parameter_supported(param):
                if reasoning_mode and param in {"temperature", "top_p", "frequency_penalty", "presence_penalty"}:
                    allowed_defaults = {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                    }
                    try:
                        allowed = allowed_defaults[param]
                        v = float(value)
                        if abs(v - float(allowed)) > 1e-9:
                            if param in explicit_sampling:
                                warnings.warn(
                                    f"OpenAI model '{self.model}' does not support setting {param} when "
                                    f"reasoning_effort={reasoning_effort_s!r}; dropping {param}={value!r}.",
                                    RuntimeWarning,
                                    stacklevel=3,
                                )
                            continue
                    except Exception:
                        if param in explicit_sampling:
                            warnings.warn(
                                f"OpenAI model '{self.model}' does not support setting {param} when "
                                f"reasoning_effort={reasoning_effort_s!r}; dropping {param}={value!r}.",
                                RuntimeWarning,
                                stacklevel=3,
                            )
                        continue
                call_params[param] = value

        if reasoning_effort_s:
            call_params["reasoning_effort"] = reasoning_effort_s

        # Output token parameter name (driven by model_capabilities.json)
        call_params[self._get_token_param_name()] = max_output_tokens

        # Add tools if provided (convert to native format)
        if tools:
            # Convert tools to native format for OpenAI API
            if self.tool_handler.supports_native:
                call_params["tools"] = self.tool_handler.prepare_tools_for_native(tools)
                call_params["tool_choice"] = kwargs.get("tool_choice", "auto")
            else:
                # Fallback to manual formatting
                call_params["tools"] = self._format_tools_for_openai(tools)
                call_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add structured output support (OpenAI native)
        if response_model and PYDANTIC_AVAILABLE:
            if self._supports_structured_output():
                json_schema = response_model.model_json_schema()

                # OpenAI requires additionalProperties: false for strict mode
                self._ensure_strict_schema(json_schema)

                call_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": json_schema
                    }
                }

        # Make async API call with proper exception handling
        try:
            if stream:
                return self._async_stream_response(call_params, tools)
            else:
                # Track generation time
                start_time = time.time()
                response = await self.async_client.chat.completions.create(**call_params)
                gen_time = round((time.time() - start_time) * 1000, 1)

                formatted = self._format_response(response)
                # Add generation time to response
                formatted.gen_time = gen_time
                formatted.metadata = dict(formatted.metadata or {})
                formatted.metadata["_provider_request"] = {"call_params": call_params}
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    formatted.metadata = merge_enrichment_metadata(formatted.metadata, media_enrichment)

                # Handle tool execution for OpenAI native responses
                if tools and formatted.has_tool_calls():
                    formatted = self._handle_tool_execution(formatted, tools)

                return formatted
        except Exception:
            raise

    async def _async_stream_response(self, call_params: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[GenerateResponse]:
        """Native async streaming responses from OpenAI."""
        try:
            stream = await self.async_client.chat.completions.create(**call_params)
        except Exception:
            raise

        # For streaming with tools, we need to collect the complete response
        collected_content = ""
        collected_tool_calls = {}  # Use dict to merge streaming chunks by tool call ID
        final_response = None

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta
            content = getattr(delta, 'content', None) or ""
            collected_content += content

            # Handle tool calls in streaming - merge incomplete chunks
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_id = getattr(tc, 'id', None) or getattr(tc, 'index', 0)

                    # Initialize or get existing tool call
                    if tc_id not in collected_tool_calls:
                        collected_tool_calls[tc_id] = {
                            "id": getattr(tc, 'id', None),
                            "type": getattr(tc, 'type', 'function'),
                            "name": None,
                            "arguments": ""
                        }

                    # Update with new data from this chunk
                    if hasattr(tc, 'function'):
                        if hasattr(tc.function, 'name') and tc.function.name:
                            collected_tool_calls[tc_id]["name"] = tc.function.name
                        if hasattr(tc.function, 'arguments') and tc.function.arguments:
                            collected_tool_calls[tc_id]["arguments"] += tc.function.arguments

            # Create chunk response
            chunk_response = GenerateResponse(
                content=content,
                raw_response=chunk,
                model=chunk.model,
                finish_reason=choice.finish_reason,
                tool_calls=None  # Don't include incomplete tool calls in chunks
            )

            # If this is the final chunk and we have tools, handle tool execution
            if choice.finish_reason and tools and collected_tool_calls:
                # Convert dict to list and filter out incomplete tool calls
                complete_tool_calls = []
                for tc in collected_tool_calls.values():
                    if tc["name"] and tc["arguments"] is not None:  # Include tool calls with empty args
                        complete_tool_calls.append(tc)

                # Create complete response for tool processing
                complete_response = GenerateResponse(
                    content=collected_content,
                    raw_response=chunk,
                    model=chunk.model,
                    finish_reason=choice.finish_reason,
                    tool_calls=complete_tool_calls if complete_tool_calls else None
                )

                # Handle tool execution
                final_response = self._handle_tool_execution(complete_response, tools)

                # If tools were executed, yield the tool results as final chunk
                if final_response.content != collected_content:
                    tool_results_content = final_response.content[len(collected_content):]
                    yield GenerateResponse(
                        content=tool_results_content,
                        raw_response=chunk,
                        model=chunk.model,
                        finish_reason=choice.finish_reason,
                        tool_calls=None
                    )
                else:
                    # No tools executed but response was processed - yield final response content
                    yield GenerateResponse(
                        content=final_response.content,
                        raw_response=chunk,
                        model=chunk.model,
                        finish_reason=choice.finish_reason,
                        tool_calls=complete_tool_calls if complete_tool_calls else None
                    )
            else:
                yield chunk_response

    def _format_tools_for_openai(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI API format"""
        formatted_tools = []
        for tool in tools:
            # Clean parameters for OpenAI compatibility
            parameters = self._clean_parameters_for_openai(tool.get("parameters", {}))

            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": parameters
                }
            }
            formatted_tools.append(formatted_tool)
        return formatted_tools

    def _format_response(self, response) -> GenerateResponse:
        """Format OpenAI response to GenerateResponse"""
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls if present
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })

        # Build usage dict with consistent terminology
        usage = None
        if hasattr(response, 'usage'):
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                # Keep legacy keys for backward compatibility
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }

            # Add detailed token breakdown for reasoning models
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                usage["completion_tokens_details"] = {
                    "reasoning_tokens": getattr(details, 'reasoning_tokens', 0),
                    "accepted_prediction_tokens": getattr(details, 'accepted_prediction_tokens', 0),
                    "rejected_prediction_tokens": getattr(details, 'rejected_prediction_tokens', 0),
                    "audio_tokens": getattr(details, 'audio_tokens', 0)
                }

            if hasattr(response.usage, 'prompt_tokens_details'):
                prompt_details = response.usage.prompt_tokens_details
                usage["prompt_tokens_details"] = {
                    "cached_tokens": getattr(prompt_details, 'cached_tokens', 0),
                    "audio_tokens": getattr(prompt_details, 'audio_tokens', 0)
                }

        return GenerateResponse(
            content=message.content,
            raw_response=response,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            tool_calls=tool_calls
        )

    def _handle_tool_execution(self, response: GenerateResponse, tools: List[Dict[str, Any]]) -> GenerateResponse:
        """Handle tool execution for OpenAI native responses"""
        if not response.has_tool_calls():
            return response

        # Convert OpenAI tool calls to standard format using base method
        tool_calls = self._convert_native_tool_calls_to_standard(response.tool_calls)

        # Execute with events using base method
        return self._execute_tools_with_events(response, tool_calls)

    def _stream_response(self, call_params: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[GenerateResponse]:
        """Stream responses from OpenAI"""
        try:
            stream = self.client.chat.completions.create(**call_params)
        except Exception as e:
            # Model validation is done at initialization, so this is likely an API error
            raise ProviderAPIError(f"OpenAI API error: {str(e)}")

        # For streaming with tools, we need to collect the complete response
        collected_content = ""
        collected_tool_calls = {}  # Use dict to merge streaming chunks by tool call ID
        final_response = None

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta
            content = getattr(delta, 'content', None) or ""
            collected_content += content

            # Handle tool calls in streaming - merge incomplete chunks
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_id = getattr(tc, 'id', None) or getattr(tc, 'index', 0)

                    # Initialize or get existing tool call
                    if tc_id not in collected_tool_calls:
                        collected_tool_calls[tc_id] = {
                            "id": getattr(tc, 'id', None),
                            "type": getattr(tc, 'type', 'function'),
                            "name": None,
                            "arguments": ""
                        }

                    # Update with new data from this chunk
                    if hasattr(tc, 'function'):
                        if hasattr(tc.function, 'name') and tc.function.name:
                            collected_tool_calls[tc_id]["name"] = tc.function.name
                        if hasattr(tc.function, 'arguments') and tc.function.arguments:
                            collected_tool_calls[tc_id]["arguments"] += tc.function.arguments

            # Create chunk response
            chunk_response = GenerateResponse(
                content=content,
                raw_response=chunk,
                model=chunk.model,
                finish_reason=choice.finish_reason,
                tool_calls=None  # Don't include incomplete tool calls in chunks
            )

            # If this is the final chunk and we have tools, handle tool execution
            if choice.finish_reason and tools and collected_tool_calls:
                # Convert dict to list and filter out incomplete tool calls
                complete_tool_calls = []
                for tc in collected_tool_calls.values():
                    if tc["name"] and tc["arguments"] is not None:  # Include tool calls with empty args
                        complete_tool_calls.append(tc)

                # Create complete response for tool processing
                complete_response = GenerateResponse(
                    content=collected_content,
                    raw_response=chunk,
                    model=chunk.model,
                    finish_reason=choice.finish_reason,
                    tool_calls=complete_tool_calls if complete_tool_calls else None
                )

                # Handle tool execution
                final_response = self._handle_tool_execution(complete_response, tools)

                # If tools were executed, yield the tool results as final chunk
                if final_response.content != collected_content:
                    tool_results_content = final_response.content[len(collected_content):]
                    yield GenerateResponse(
                        content=tool_results_content,
                        raw_response=chunk,
                        model=chunk.model,
                        finish_reason=choice.finish_reason,
                        tool_calls=None
                    )
                else:
                    # No tools executed but response was processed - yield final response content
                    yield GenerateResponse(
                        content=final_response.content,
                        raw_response=chunk,
                        model=chunk.model,
                        finish_reason=choice.finish_reason,
                        tool_calls=complete_tool_calls if complete_tool_calls else None
                    )
            else:
                yield chunk_response

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

    def supports_prompt_cache(self) -> bool:
        """OpenAI supports prompt caching via `prompt_cache_key` (server-managed)."""
        return True

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        capabilities = [
            "chat",
            "streaming",
            "system_prompt"
        ]

        # Add tools if supported
        if self.tool_handler.supports_native or self.tool_handler.supports_prompted:
            capabilities.append("tools")

        # Add vision for capable models
        if "gpt-4o" in self.model or "gpt-4-turbo" in self.model:
            capabilities.append("vision")

        return capabilities


    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if not self.api_key:
            return False
        return True

    def unload_model(self, model_name: str) -> None:
        """Close async client if it was created."""
        if self._async_client is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_client.close())
            except RuntimeError:
                # No running loop, close synchronously
                import asyncio
                asyncio.run(self._async_client.close())

    def _validate_model_exists(self):
        """Preflight check to validate model exists before any generation"""
        try:
            # Use the models.list() API to check if model exists
            models = self.client.models.list()
            available_model_ids = [model.id for model in models.data]

            if self.model not in available_model_ids:
                # Model not found - provide helpful error
                error_message = format_model_error("OpenAI", self.model, available_model_ids)
                raise ModelNotFoundError(error_message)

        except Exception as e:
            # If it's already a ModelNotFoundError, re-raise it
            if isinstance(e, ModelNotFoundError):
                raise
            # For other errors (like API failures), handle gracefully
            error_str = str(e).lower()
            if 'api_key' in error_str or 'authentication' in error_str:
                raise AuthenticationError(format_auth_error("openai", str(e)))
            # For other API errors during preflight, continue (model might work)
            # This allows for cases where models.list() fails but generation works

    # Removed overrides - using BaseProvider methods with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for OpenAI API"""
        # For OpenAI, max_tokens in the API is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _supports_structured_output(self) -> bool:
        """Return True when the registry says this model supports native structured outputs."""
        caps = getattr(self, "model_capabilities", None)
        if isinstance(caps, dict):
            level = caps.get("structured_output")
            if isinstance(level, str) and level.strip().lower() == "native":
                return True

        # Backward-compatible fallback for older registries.
        model_lower = self.model.lower()
        return "gpt-4o" in model_lower

    def _ensure_strict_schema(self, schema: Dict[str, Any]) -> None:
        """
        Ensure schema is compatible with OpenAI strict mode.

        OpenAI requires:
        - additionalProperties: false for all objects
        - required array must include all properties
        """
        def make_strict(obj):
            if isinstance(obj, dict):
                # Add additionalProperties: false to all objects
                if "type" in obj and obj["type"] == "object":
                    obj["additionalProperties"] = False

                    # Ensure all properties are in required array for strict mode
                    if "properties" in obj:
                        all_props = list(obj["properties"].keys())
                        obj["required"] = all_props

                # Recursively process nested objects
                for value in obj.values():
                    make_strict(value)
            elif isinstance(obj, list):
                for item in obj:
                    make_strict(item)

        make_strict(schema)

    def _update_http_client_timeout(self) -> None:
        """Update OpenAI client timeout when timeout is changed."""
        # Create new client with updated timeout
        self.client = openai.OpenAI(api_key=self.api_key, timeout=self._timeout)

    def _clean_parameters_for_openai(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean tool parameters for OpenAI compatibility.

        OpenAI doesn't accept 'default' properties in function parameters.
        """
        if not parameters:
            return parameters

        def remove_defaults(obj):
            if isinstance(obj, dict):
                # Remove 'default' keys and process nested objects
                cleaned = {}
                for key, value in obj.items():
                    if key != "default":
                        cleaned[key] = remove_defaults(value)
                return cleaned
            elif isinstance(obj, list):
                return [remove_defaults(item) for item in obj]
            else:
                return obj

        return remove_defaults(parameters)

    @classmethod
    def list_available_models(cls, **kwargs) -> List[str]:
        """
        List available models from OpenAI API.

        Args:
            **kwargs: Optional parameters including:
                - api_key: OpenAI API key
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            import openai
            from .model_capabilities import filter_models_by_capabilities

            # Get API key from kwargs or environment
            api_key = kwargs.get('api_key') or os.getenv("OPENAI_API_KEY")
            if not api_key:
                return []

            # Create temporary client
            client = openai.OpenAI(api_key=api_key, timeout=5.0)
            models = client.models.list()

            # Extract model IDs and filter to chat models only
            # Filter based on actual OpenAI chat completion model patterns
            available_models = [model.id for model in models.data]
            chat_models = []

            for model_id in available_models:
                # Include GPT models for chat completions
                if any(pattern in model_id for pattern in [
                    "gpt-3.5", "gpt-4", "gpt-5", "gpt-o1", "o1-",  # Standard patterns
                    "text-davinci", "code-davinci"  # Legacy but still chat-capable
                ]):
                    # Exclude embedding, fine-tuning, and other non-chat models
                    if not any(exclude in model_id for exclude in [
                        "embedding", "similarity", "search", "edit",
                        "insert", "davinci-002", "babbage", "ada", "curie"
                    ]):
                        chat_models.append(model_id)

            chat_models = sorted(chat_models, reverse=True)  # Latest models first

            # Apply new capability filtering if provided
            input_capabilities = kwargs.get('input_capabilities')
            output_capabilities = kwargs.get('output_capabilities')
            
            if input_capabilities or output_capabilities:
                chat_models = filter_models_by_capabilities(
                    chat_models, 
                    input_capabilities=input_capabilities,
                    output_capabilities=output_capabilities
                )


            return chat_models

        except Exception:
            return []

    # ------------------------------------------------------------------
    # Embeddings (OpenAI embeddings API)
    # ------------------------------------------------------------------

    def embed(self, input_text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Generate embeddings using the OpenAI embeddings API.

        Args:
            input_text: Single string or list of strings to embed.
            **kwargs: Extra parameters forwarded to ``client.embeddings.create``
                      (e.g. ``dimensions``, ``encoding_format``).

        Returns:
            Dict in OpenAI-compatible format (same shape as Ollama/LMStudio):
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": N, "total_tokens": N}
            }
        """
        try:
            texts = [input_text] if isinstance(input_text, str) else input_text

            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                **kwargs,
            )

            # The OpenAI SDK already returns the right shape; normalise to plain dicts.
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": item.embedding,
                        "index": item.index,
                    }
                    for item in response.data
                ],
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            raise ProviderAPIError(f"OpenAI embedding error: {str(e)}")
