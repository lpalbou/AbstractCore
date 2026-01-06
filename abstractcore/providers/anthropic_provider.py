"""
Anthropic provider implementation.
"""

import os
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
from ..media import MediaHandler
from ..exceptions import AuthenticationError, ProviderAPIError, ModelNotFoundError, format_model_error, format_auth_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider with full integration"""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "anthropic"

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

        # Get API key from param or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        # Get base URL from param or environment
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

        # Initialize client with timeout and optional base_url
        client_kwargs = {"api_key": self.api_key, "timeout": self._timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        self._async_client = None  # Lazy-loaded async client

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Store provider-specific configuration
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", None)

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    @property
    def async_client(self):
        """Lazy-load AsyncAnthropic client for native async operations."""
        if self._async_client is None:
            client_kwargs = {"api_key": self.api_key, "timeout": self._timeout}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._async_client = anthropic.AsyncAnthropic(**client_kwargs)
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
        """Internal generation with Anthropic API"""

        # Build messages array
        api_messages = []

        # Add conversation history
        if messages:
            for msg in messages:
                # Skip system messages as they're handled separately
                if msg.get("role") != "system":
                    # Convert assistant role if needed
                    role = msg["role"]
                    if role == "assistant":
                        api_messages.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })
                    elif role == "tool":
                        # Anthropic Messages API represents tool outputs as `tool_result`
                        # content blocks inside a USER message (there is no `role="tool"`).
                        meta = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
                        tool_use_id = meta.get("call_id") or meta.get("tool_use_id") or meta.get("id")
                        tool_text = msg.get("content", "")
                        tool_text = "" if tool_text is None else str(tool_text)

                        if isinstance(tool_use_id, str) and tool_use_id.strip():
                            api_messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_use_id.strip(),
                                            "content": tool_text,
                                        }
                                    ],
                                }
                            )
                        else:
                            # Fallback: preserve as plain user text when no tool_use_id is available.
                            api_messages.append({"role": "user", "content": tool_text})
                    else:
                        api_messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })

        # Add current prompt as user message
        if prompt and prompt not in [msg.get("content") for msg in (messages or [])]:
            # Handle multimodal message with media content
            if media:
                try:
                    from ..media.handlers import AnthropicMediaHandler
                    media_handler = AnthropicMediaHandler(self.model_capabilities)

                    # Create multimodal message combining text and media
                    multimodal_message = media_handler.create_multimodal_message(prompt, media)
                    api_messages.append(multimodal_message)
                except ImportError:
                    self.logger.warning("Media processing not available. Install with: pip install abstractcore[media]")
                    api_messages.append({"role": "user", "content": prompt})
                except Exception as e:
                    self.logger.warning(f"Failed to process media content: {e}")
                    api_messages.append({"role": "user", "content": prompt})
            else:
                api_messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        call_params = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_output_tokens,  # This is max_output_tokens for Anthropic
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": stream
        }

        # Add system prompt if provided
        if system_prompt:
            call_params["system"] = system_prompt

        # Add top_p if specified
        if kwargs.get("top_p") or self.top_p < 1.0:
            call_params["top_p"] = kwargs.get("top_p", self.top_p)

        # Add top_k if specified
        if kwargs.get("top_k") or self.top_k:
            call_params["top_k"] = kwargs.get("top_k", self.top_k)

        # Handle seed parameter (Anthropic doesn't support seed natively)
        seed_value = kwargs.get("seed", self.seed)
        if seed_value is not None:
            import warnings
            warnings.warn(
                f"Seed parameter ({seed_value}) is not supported by Anthropic Claude API. "
                f"For deterministic outputs, use temperature=0.0 which may provide more consistent results, "
                f"though true determinism is not guaranteed.",
                UserWarning,
                stacklevel=3
            )
            self.logger.warning(f"Seed {seed_value} requested but not supported by Anthropic API")

        # Handle structured output using the "tool trick"
        structured_tool_name = None
        if response_model and PYDANTIC_AVAILABLE:
            # Create a synthetic tool for structured output
            structured_tool = self._create_structured_output_tool(response_model)

            # Add to existing tools or create new tools list
            if tools:
                tools = list(tools) + [structured_tool]
            else:
                tools = [structured_tool]

            structured_tool_name = structured_tool["name"]

            # Modify the prompt to instruct the model to use the structured tool
            if api_messages and api_messages[-1]["role"] == "user":
                api_messages[-1]["content"] += f"\n\nPlease use the {structured_tool_name} tool to provide your response."

        # Add tools if provided (convert to native format)
        if tools:
            if self.tool_handler.supports_native:
                # Use Anthropic-specific tool formatting instead of universal handler
                call_params["tools"] = self._format_tools_for_anthropic(tools)

                # Force tool use for structured output
                if structured_tool_name:
                    call_params["tool_choice"] = {"type": "tool", "name": structured_tool_name}
                elif kwargs.get("tool_choice"):
                    call_params["tool_choice"] = {"type": kwargs.get("tool_choice", "auto")}
            else:
                # Add tools as system prompt for prompted models
                system_text = call_params.get("system") if isinstance(call_params.get("system"), str) else ""
                include_tool_list = "## Tools (session)" not in system_text
                tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
                if call_params.get("system"):
                    call_params["system"] += f"\n\n{tool_prompt}"
                else:
                    call_params["system"] = tool_prompt

        # Make API call with proper exception handling
        try:
            if stream:
                return self._stream_response(call_params, tools)
            else:
                # Track generation time
                start_time = time.time()
                response = self.client.messages.create(**call_params)
                gen_time = round((time.time() - start_time) * 1000, 1)
                
                formatted = self._format_response(response)
                # Add generation time to response
                formatted.gen_time = gen_time
                formatted.metadata = dict(formatted.metadata or {})
                formatted.metadata["_provider_request"] = {"call_params": call_params}

                # Handle tool execution for Anthropic responses
                if tools and (formatted.has_tool_calls() or
                             (self.tool_handler.supports_prompted and formatted.content)):
                    formatted = self._handle_tool_execution(formatted, tools)

                return formatted
        except Exception as e:
            # Use proper exception handling from base
            error_str = str(e).lower()

            if 'api_key' in error_str or 'authentication' in error_str:
                raise AuthenticationError(format_auth_error("anthropic", str(e)))
            elif ('not_found_error' in error_str and 'model:' in error_str) or '404' in error_str:
                # Model not found - show available models
                available_models = self.list_available_models(api_key=self.api_key)
                error_message = format_model_error("Anthropic", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
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
        """Native async implementation using AsyncAnthropic - 3-10x faster for batch operations."""

        # Build messages array (same logic as sync)
        api_messages = []

        # Add conversation history
        if messages:
            for msg in messages:
                # Skip system messages as they're handled separately
                if msg.get("role") != "system":
                    # Convert assistant role if needed
                    role = msg["role"]
                    if role == "assistant":
                        api_messages.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })
                    elif role == "tool":
                        # Anthropic Messages API represents tool outputs as `tool_result`
                        # content blocks inside a USER message (there is no `role="tool"`).
                        meta = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
                        tool_use_id = meta.get("call_id") or meta.get("tool_use_id") or meta.get("id")
                        tool_text = msg.get("content", "")
                        tool_text = "" if tool_text is None else str(tool_text)

                        if isinstance(tool_use_id, str) and tool_use_id.strip():
                            api_messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_use_id.strip(),
                                            "content": tool_text,
                                        }
                                    ],
                                }
                            )
                        else:
                            # Fallback: preserve as plain user text when no tool_use_id is available.
                            api_messages.append({"role": "user", "content": tool_text})
                    else:
                        api_messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })

        # Add current prompt as user message
        if prompt and prompt not in [msg.get("content") for msg in (messages or [])]:
            # Handle multimodal message with media content
            if media:
                try:
                    from ..media.handlers import AnthropicMediaHandler
                    media_handler = AnthropicMediaHandler(self.model_capabilities)

                    # Create multimodal message combining text and media
                    multimodal_message = media_handler.create_multimodal_message(prompt, media)
                    api_messages.append(multimodal_message)
                except ImportError:
                    self.logger.warning("Media processing not available. Install with: pip install abstractcore[media]")
                    api_messages.append({"role": "user", "content": prompt})
                except Exception as e:
                    self.logger.warning(f"Failed to process media content: {e}")
                    api_messages.append({"role": "user", "content": prompt})
            else:
                api_messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters (same logic as sync)
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        call_params = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_output_tokens,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": stream
        }

        # Add system prompt if provided (Anthropic-specific: separate parameter)
        if system_prompt:
            call_params["system"] = system_prompt

        # Add top_p if specified
        if kwargs.get("top_p") or self.top_p < 1.0:
            call_params["top_p"] = kwargs.get("top_p", self.top_p)

        # Add top_k if specified
        if kwargs.get("top_k") or self.top_k:
            call_params["top_k"] = kwargs.get("top_k", self.top_k)

        # Handle seed parameter (Anthropic doesn't support seed natively)
        seed_value = kwargs.get("seed", self.seed)
        if seed_value is not None:
            import warnings
            warnings.warn(
                f"Seed parameter ({seed_value}) is not supported by Anthropic Claude API. "
                f"For deterministic outputs, use temperature=0.0 which may provide more consistent results, "
                f"though true determinism is not guaranteed.",
                UserWarning,
                stacklevel=3
            )
            self.logger.warning(f"Seed {seed_value} requested but not supported by Anthropic API")

        # Handle structured output using the "tool trick"
        structured_tool_name = None
        if response_model and PYDANTIC_AVAILABLE:
            structured_tool = self._create_structured_output_tool(response_model)

            if tools:
                tools = list(tools) + [structured_tool]
            else:
                tools = [structured_tool]

            structured_tool_name = structured_tool["name"]

            if api_messages and api_messages[-1]["role"] == "user":
                api_messages[-1]["content"] += f"\n\nPlease use the {structured_tool_name} tool to provide your response."

        # Add tools if provided
        if tools:
            if self.tool_handler.supports_native:
                call_params["tools"] = self._format_tools_for_anthropic(tools)

                if structured_tool_name:
                    call_params["tool_choice"] = {"type": "tool", "name": structured_tool_name}
                elif kwargs.get("tool_choice"):
                    call_params["tool_choice"] = {"type": kwargs.get("tool_choice", "auto")}
            else:
                system_text = call_params.get("system") if isinstance(call_params.get("system"), str) else ""
                include_tool_list = "## Tools (session)" not in system_text
                tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
                if call_params.get("system"):
                    call_params["system"] += f"\n\n{tool_prompt}"
                else:
                    call_params["system"] = tool_prompt

        # Make async API call
        try:
            if stream:
                return self._async_stream_response(call_params, tools)
            else:
                start_time = time.time()
                response = await self.async_client.messages.create(**call_params)
                gen_time = round((time.time() - start_time) * 1000, 1)

                formatted = self._format_response(response)
                formatted.gen_time = gen_time
                formatted.metadata = dict(formatted.metadata or {})
                formatted.metadata["_provider_request"] = {"call_params": call_params}

                if tools and (formatted.has_tool_calls() or
                             (self.tool_handler.supports_prompted and formatted.content)):
                    formatted = self._handle_tool_execution(formatted, tools)

                return formatted
        except Exception as e:
            error_str = str(e).lower()

            if 'api_key' in error_str or 'authentication' in error_str:
                raise AuthenticationError(format_auth_error("anthropic", str(e)))
            elif ('not_found_error' in error_str and 'model:' in error_str) or '404' in error_str:
                available_models = self.list_available_models(api_key=self.api_key)
                error_message = format_model_error("Anthropic", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise

    async def _async_stream_response(self, call_params: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[GenerateResponse]:
        """Native async streaming with Anthropic's context manager pattern."""
        stream_params = {k: v for k, v in call_params.items() if k != 'stream'}

        try:
            async with self.async_client.messages.stream(**stream_params) as stream:
                async for chunk in stream:
                    yield GenerateResponse(
                        content=getattr(chunk, 'content', ''),
                        model=self.model,
                        finish_reason=getattr(chunk, 'finish_reason', None),
                        raw_response=chunk
                    )
        except Exception as e:
            raise

    def unload(self) -> None:
        """Close async client if it was created."""
        if self._async_client is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_client.close())
            except RuntimeError:
                import asyncio
                asyncio.run(self._async_client.close())

    def _format_tools_for_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic API format"""
        formatted_tools = []
        for tool in tools:
            # Anthropic expects `input_schema` to be a JSON Schema object:
            # https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use
            #
            # Our internal tool representation typically uses:
            #   tool["parameters"] = { "arg": {"type": "...", "default": ...?}, ... }
            # or, less commonly:
            #   tool["parameters"] = {"type":"object","properties":{...},"required":[...]}
            params = tool.get("parameters", {})

            properties: Dict[str, Any] = {}
            required: List[str] = []

            if isinstance(params, dict) and "properties" in params:
                # Treat as already-schema-like.
                raw_props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
                properties = dict(raw_props)
                raw_required = params.get("required")
                if isinstance(raw_required, list):
                    required = [str(x) for x in raw_required if isinstance(x, (str, int))]
            elif isinstance(params, dict):
                # Treat as compact parameter dict; infer required args by absence of `default`.
                properties = dict(params)
                for k, v in params.items():
                    if isinstance(v, dict) and "default" not in v:
                        required.append(str(k))

            input_schema: Dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }
            if required:
                input_schema["required"] = required

            formatted_tool = {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "input_schema": input_schema
            }
            formatted_tools.append(formatted_tool)
        return formatted_tools

    def _format_response(self, response) -> GenerateResponse:
        """Format Anthropic response to GenerateResponse"""

        # Extract content from response
        content = ""
        tool_calls = None

        # Handle different content types
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": content_block.id,
                    "type": "tool_use",
                    "name": content_block.name,
                    "arguments": json.dumps(content_block.input) if not isinstance(content_block.input, str) else content_block.input
                })

        # Build usage dict
        usage = None
        if hasattr(response, 'usage'):
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

        return GenerateResponse(
            content=content,
            raw_response=response,
            model=response.model,
            finish_reason=response.stop_reason,
            usage=usage,
            tool_calls=tool_calls
        )

    def _handle_tool_execution(self, response: GenerateResponse, tools: List[Dict[str, Any]]) -> GenerateResponse:
        """Handle tool execution for Anthropic responses"""
        # Check for native tool calls first
        if response.has_tool_calls():
            # Convert Anthropic tool calls to standard format using base method
            tool_calls = self._convert_native_tool_calls_to_standard(response.tool_calls)
            # Execute with events using base method
            return self._execute_tools_with_events(response, tool_calls)
        elif self.tool_handler.supports_prompted and response.content:
            # Handle prompted tool calls using base method
            return self._handle_prompted_tool_execution(response, tools)

        return response

    def _stream_response(self, call_params: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[GenerateResponse]:
        """Stream responses from Anthropic"""
        # Remove stream parameter for streaming API
        stream_params = {k: v for k, v in call_params.items() if k != 'stream'}
        with self.client.messages.stream(**stream_params) as stream:
            current_tool_call = None
            accumulated_input = ""

            # For tool execution, collect complete response
            collected_content = ""
            collected_tool_calls = []

            for chunk in stream:
                # Handle different event types
                if chunk.type == "content_block_start":
                    # Start of a new content block (could be text or tool_use)
                    if hasattr(chunk, 'content_block') and chunk.content_block.type == "tool_use":
                        # Starting a tool call
                        current_tool_call = {
                            "id": chunk.content_block.id,
                            "type": "tool_use",
                            "name": chunk.content_block.name,
                            "arguments": ""
                        }
                        accumulated_input = ""

                elif chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        # Text content
                        text_content = chunk.delta.text
                        collected_content += text_content
                        yield GenerateResponse(
                            content=text_content,
                            raw_response=chunk,
                            model=call_params.get("model")
                        )
                    elif hasattr(chunk.delta, 'partial_json'):
                        # Tool call arguments coming in chunks
                        if current_tool_call:
                            accumulated_input += chunk.delta.partial_json
                            # Yield partial tool call
                            tool_call_partial = current_tool_call.copy()
                            tool_call_partial["arguments"] = accumulated_input
                            yield GenerateResponse(
                                content="",
                                raw_response=chunk,
                                model=call_params.get("model"),
                                tool_calls=[tool_call_partial]
                            )

                elif chunk.type == "content_block_stop":
                    # End of a content block
                    if current_tool_call and accumulated_input:
                        # Finalize the tool call with complete arguments
                        current_tool_call["arguments"] = accumulated_input
                        collected_tool_calls.append(current_tool_call)
                        yield GenerateResponse(
                            content="",
                            raw_response=chunk,
                            model=call_params.get("model"),
                            tool_calls=[current_tool_call]
                        )
                        current_tool_call = None
                        accumulated_input = ""

                elif chunk.type == "message_stop":
                    # Final chunk with usage info and tool execution
                    usage = None
                    if hasattr(stream, 'response') and hasattr(stream.response, 'usage'):
                        usage = {
                            "prompt_tokens": stream.response.usage.input_tokens,
                            "completion_tokens": stream.response.usage.output_tokens,
                            "total_tokens": stream.response.usage.input_tokens + stream.response.usage.output_tokens
                        }

                    # Handle tool execution if we have tools and collected calls
                    if tools and (collected_tool_calls or
                                 (self.tool_handler.supports_prompted and collected_content)):
                        # Create complete response for tool processing
                        complete_response = GenerateResponse(
                            content=collected_content,
                            raw_response=chunk,
                            model=call_params.get("model"),
                            finish_reason="stop",
                            usage=usage,
                            tool_calls=collected_tool_calls
                        )

                        # Handle tool execution
                        final_response = self._handle_tool_execution(complete_response, tools)

                        # If tools were executed, yield the tool results as final chunk
                        if final_response.content != collected_content:
                            tool_results_content = final_response.content[len(collected_content):]
                            yield GenerateResponse(
                                content=tool_results_content,
                                raw_response=chunk,
                                model=call_params.get("model"),
                                finish_reason="stop",
                                usage=usage,
                                tool_calls=None
                            )

                    # Always yield final chunk
                    yield GenerateResponse(
                        content="",
                        raw_response=chunk,
                        model=call_params.get("model"),
                        finish_reason="stop",
                        usage=usage
                    )

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        capabilities = [
            "chat",
            "streaming",
            "system_prompt",
            "tools",
            "vision"  # All Claude 3 models support vision
        ]
        return capabilities


    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if not self.api_key:
            return False
        return True

    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for Anthropic API"""
        # For Anthropic, max_tokens in the API is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _create_structured_output_tool(self, response_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Create a synthetic tool for structured output using Anthropic's tool calling.

        Args:
            response_model: Pydantic model to create tool for

        Returns:
            Tool definition dict for Anthropic API
        """
        schema = response_model.model_json_schema()
        tool_name = f"extract_{response_model.__name__.lower()}"

        return {
            "name": tool_name,
            "description": f"Extract structured data in {response_model.__name__} format",
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
                "additionalProperties": False
            }
        }

    def _update_http_client_timeout(self) -> None:
        """Update Anthropic client timeout when timeout is changed."""
        # Create new client with updated timeout
        self.client = anthropic.Anthropic(api_key=self.api_key, timeout=self._timeout)
    def list_available_models(self, **kwargs) -> List[str]:
        """
        List available models from Anthropic API.

        Args:
            **kwargs: Optional parameters including:
                - api_key: Anthropic API key
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            import httpx
            from .model_capabilities import filter_models_by_capabilities

            # Use provided API key or instance API key
            api_key = kwargs.get('api_key', self.api_key)
            if not api_key:
                self.logger.debug("No Anthropic API key available for model listing")
                return []

            # Make API call to list models
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }

            response = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers=headers,
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data.get("data", [])]
                self.logger.debug(f"Retrieved {len(models)} models from Anthropic API")
                models = sorted(models, reverse=True)  # Latest models first

                # Apply new capability filtering if provided
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
                self.logger.warning(f"Anthropic API returned status {response.status_code}")
                return []

        except Exception as e:
            self.logger.debug(f"Failed to fetch Anthropic models from API: {e}")
            return []
