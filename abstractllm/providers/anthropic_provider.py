"""
Anthropic provider implementation.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union, Iterator
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..media import MediaHandler
from ..exceptions import AuthenticationError, ProviderAPIError, ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models, format_model_error

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider with full integration"""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

        # Get API key from param or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Store configuration
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", None)

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          stream: bool = False,
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
                    else:
                        api_messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })

        # Add current prompt as user message
        if prompt and prompt not in [msg.get("content") for msg in (messages or [])]:
            api_messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters
        call_params = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
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

        # Add tools if provided
        if tools:
            call_params["tools"] = self._format_tools_for_anthropic(tools)
            # Anthropic uses tool_choice differently than OpenAI
            if kwargs.get("tool_choice"):
                call_params["tool_choice"] = {"type": kwargs.get("tool_choice", "auto")}

        # Make API call with proper exception handling
        try:
            if stream:
                return self._stream_response(call_params)
            else:
                response = self.client.messages.create(**call_params)
                formatted = self._format_response(response)

                # Track tool calls if present
                if formatted.has_tool_calls():
                    for call in formatted.tool_calls:
                        self._track_tool_call(
                            tool_name=call.get('name'),
                            arguments=json.loads(call.get('arguments')) if isinstance(call.get('arguments'), str) else call.get('arguments', {}),
                            success=True
                        )

                return formatted
        except Exception as e:
            # Use proper exception handling from base
            error_str = str(e).lower()

            if 'api_key' in error_str or 'authentication' in error_str:
                raise AuthenticationError(f"Anthropic authentication failed: {str(e)}")
            elif ('not_found_error' in error_str and 'model:' in error_str) or '404' in error_str:
                # Model not found - show available models
                available_models = get_available_models("anthropic", api_key=self.api_key)
                error_message = format_model_error("Anthropic", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise ProviderAPIError(f"Anthropic API error: {str(e)}")

    def _format_tools_for_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic API format"""
        formatted_tools = []
        for tool in tools:
            formatted_tool = {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
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
                content = content_block.text
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

    def _stream_response(self, call_params: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Stream responses from Anthropic"""
        # Remove stream parameter for streaming API
        stream_params = {k: v for k, v in call_params.items() if k != 'stream'}
        with self.client.messages.stream(**stream_params) as stream:
            current_tool_call = None
            accumulated_input = ""

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
                        yield GenerateResponse(
                            content=chunk.delta.text,
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
                        yield GenerateResponse(
                            content="",
                            raw_response=chunk,
                            model=call_params.get("model"),
                            tool_calls=[current_tool_call]
                        )
                        current_tool_call = None
                        accumulated_input = ""

                elif chunk.type == "message_stop":
                    # Final chunk with usage info
                    usage = None
                    if hasattr(stream, 'response') and hasattr(stream.response, 'usage'):
                        usage = {
                            "prompt_tokens": stream.response.usage.input_tokens,
                            "completion_tokens": stream.response.usage.output_tokens,
                            "total_tokens": stream.response.usage.input_tokens + stream.response.usage.output_tokens
                        }
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

    def get_token_limit(self) -> Optional[int]:
        """Get maximum token limit for this model"""
        token_limits = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000
        }
        return token_limits.get(self.model, 200000)  # Default to 200k for newer models

    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if not self.api_key:
            return False
        return True