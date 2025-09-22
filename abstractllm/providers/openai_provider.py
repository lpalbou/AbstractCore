"""
OpenAI provider implementation.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union, Iterator, Type

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..media import MediaHandler
from ..exceptions import AuthenticationError, ProviderAPIError, ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models, format_model_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIProvider(BaseProvider):
    """OpenAI API provider with full integration"""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        # Get API key from param or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        # Initialize client
        self.client = openai.OpenAI(api_key=self.api_key)

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Preflight check: validate model exists
        self._validate_model_exists()

        # Store configuration (remove duplicate max_tokens)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
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
                if msg.get("role") != "system":
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # Add current prompt as user message
        if prompt and prompt not in [msg.get("content") for msg in (messages or [])]:
            api_messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        call_params = {
            "model": self.model,
            "messages": api_messages,
            "stream": stream
        }

        # Add parameters that are supported by this model
        if not self._is_reasoning_model():
            # Reasoning models (o1, gpt-5) don't support many parameters
            call_params["temperature"] = kwargs.get("temperature", self.temperature)
            call_params["top_p"] = kwargs.get("top_p", self.top_p)
            call_params["frequency_penalty"] = kwargs.get("frequency_penalty", self.frequency_penalty)
            call_params["presence_penalty"] = kwargs.get("presence_penalty", self.presence_penalty)

        # Handle different token parameter names for different model families
        if self._uses_max_completion_tokens():
            call_params["max_completion_tokens"] = max_output_tokens
        else:
            call_params["max_tokens"] = max_output_tokens

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
                response = self.client.chat.completions.create(**call_params)
                formatted = self._format_response(response)

                # Handle tool execution for OpenAI native responses
                if tools and formatted.has_tool_calls():
                    formatted = self._handle_tool_execution(formatted, tools)

                return formatted
        except Exception as e:
            # Model validation is done at initialization, so this is likely an API error
            raise ProviderAPIError(f"OpenAI API error: {str(e)}")

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

        # Build usage dict with detailed breakdown
        usage = None
        if hasattr(response, 'usage'):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
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
                raise AuthenticationError(f"OpenAI authentication failed: {str(e)}")
            # For other API errors during preflight, continue (model might work)
            # This allows for cases where models.list() fails but generation works

    # Removed overrides - using BaseProvider methods with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for OpenAI API"""
        # For OpenAI, max_tokens in the API is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _uses_max_completion_tokens(self) -> bool:
        """Check if this model uses max_completion_tokens instead of max_tokens"""
        # OpenAI o1 series and newer models use max_completion_tokens
        model_lower = self.model.lower()
        return (
            model_lower.startswith("o1") or
            "gpt-5" in model_lower or
            model_lower.startswith("gpt-o1")
        )

    def _is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model with limited parameter support"""
        # Reasoning models (o1, gpt-5) have restricted parameter support
        model_lower = self.model.lower()
        return (
            model_lower.startswith("o1") or
            "gpt-5" in model_lower or
            model_lower.startswith("gpt-o1")
        )

    def _supports_structured_output(self) -> bool:
        """Check if this model supports native structured output"""
        # Only specific OpenAI models support structured outputs
        model_lower = self.model.lower()
        return (
            "gpt-4o-2024-08-06" in model_lower or
            "gpt-4o-mini-2024-07-18" in model_lower or
            "gpt-4o-mini" in model_lower or
            "gpt-4o" in model_lower
        )

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