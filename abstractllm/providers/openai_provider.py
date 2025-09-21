"""
OpenAI provider implementation.
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

        # Store configuration
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 2048)
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

        # Prepare API call parameters
        call_params = {
            "model": self.model,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "stream": stream
        }

        # Add tools if provided
        if tools:
            call_params["tools"] = self._format_tools_for_openai(tools)
            call_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Make API call with proper exception handling
        try:
            if stream:
                return self._stream_response(call_params)
            else:
                response = self.client.chat.completions.create(**call_params)
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
                raise AuthenticationError(f"OpenAI authentication failed: {str(e)}")
            elif 'model' in error_str and ('not found' in error_str or 'does not exist' in error_str or '404' in error_str):
                # Model not found - provide helpful error
                available_models = get_available_models("openai", api_key=self.api_key)
                error_message = format_model_error("OpenAI", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise ProviderAPIError(f"OpenAI API error: {str(e)}")

    def _format_tools_for_openai(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI API format"""
        formatted_tools = []
        for tool in tools:
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
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

        # Build usage dict
        usage = None
        if hasattr(response, 'usage'):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        return GenerateResponse(
            content=message.content,
            raw_response=response,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            tool_calls=tool_calls
        )

    def _stream_response(self, call_params: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Stream responses from OpenAI"""
        stream = self.client.chat.completions.create(**call_params)

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta
            content = getattr(delta, 'content', None) or ""

            # Handle tool calls in streaming
            tool_calls = None
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                tool_calls = []
                for tc in delta.tool_calls:
                    tool_call = {
                        "id": getattr(tc, 'id', None),
                        "type": getattr(tc, 'type', 'function'),
                    }
                    if hasattr(tc, 'function'):
                        tool_call["name"] = getattr(tc.function, 'name', None)
                        tool_call["arguments"] = getattr(tc.function, 'arguments', None)
                    tool_calls.append(tool_call)

            # Yield chunk if it has content or tool calls or finish reason
            if content or tool_calls or choice.finish_reason:
                yield GenerateResponse(
                    content=content,
                    raw_response=chunk,
                    model=chunk.model,
                    finish_reason=choice.finish_reason,
                    tool_calls=tool_calls
                )

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        capabilities = [
            "chat",
            "streaming",
            "system_prompt",
            "tools"
        ]

        # Add vision for capable models
        if "gpt-4o" in self.model or "gpt-4-turbo" in self.model:
            capabilities.append("vision")

        return capabilities

    def get_token_limit(self) -> Optional[int]:
        """Get maximum token limit for this model"""
        token_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000
        }
        return token_limits.get(self.model)

    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if not self.api_key:
            return False
        return True