"""
BasicSession for conversation tracking.
Target: <500 lines maximum.
"""

from typing import List, Optional, Dict, Any, Union, Iterator, Callable
from datetime import datetime
from pathlib import Path
import json
import uuid
from collections.abc import Generator

from .interface import AbstractLLMInterface
from .types import GenerateResponse, Message
from .enums import MessageRole


class BasicSession:
    """
    Minimal session for conversation management.
    ONLY handles:
    - Message history
    - Basic generation
    - Simple persistence
    """

    def __init__(self,
                 provider: Optional[AbstractLLMInterface] = None,
                 system_prompt: Optional[str] = None,
                 tools: Optional[List[Callable]] = None,
                 timeout: Optional[float] = None,
                 tool_timeout: Optional[float] = None,
                 recovery_timeout: Optional[float] = None):
        """Initialize basic session"""

        self.provider = provider
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.messages: List[Message] = []
        self.system_prompt = system_prompt
        self.tools = self._register_tools(tools) if tools else []

        # Apply timeout configurations to provider if specified and provider exists
        if self.provider and hasattr(self.provider, 'set_timeout'):
            if timeout is not None:
                self.provider.set_timeout(timeout)
            if tool_timeout is not None:
                self.provider.set_tool_timeout(tool_timeout)
            if recovery_timeout is not None:
                self.provider.set_recovery_timeout(recovery_timeout)

        # Add system message if provided
        if system_prompt:
            self.add_message(MessageRole.SYSTEM.value, system_prompt)

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to conversation history"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message

    def get_messages(self) -> List[Message]:
        """Get all messages"""
        return self.messages.copy()

    def get_history(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """Get conversation history as dicts"""
        if include_system:
            return [m.to_dict() for m in self.messages]
        return [m.to_dict() for m in self.messages if m.role != 'system']

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history"""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == 'system']
        else:
            self.messages = []

    def generate(self, prompt: str, **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using provider"""
        if not self.provider:
            raise ValueError("No provider configured")

        # Add user message
        self.add_message('user', prompt)

        # Format messages for provider (exclude the current user message since provider will add it)
        messages = self._format_messages_for_provider_excluding_current()

        # Use session tools if not provided in kwargs
        if 'tools' not in kwargs and self.tools:
            kwargs['tools'] = self.tools

        # Call provider
        response = self.provider.generate(
            prompt=prompt,
            messages=messages,
            system_prompt=self.system_prompt,
            **kwargs
        )

        # Handle streaming vs non-streaming responses
        if isinstance(response, (Generator, Iterator)) or hasattr(response, '__iter__') and not hasattr(response, 'content'):
            # Streaming response - wrap it to collect content
            return self._handle_streaming_response(response)
        else:
            # Non-streaming response
            if hasattr(response, 'content') and response.content:
                self.add_message('assistant', response.content)
            return response

    def _handle_streaming_response(self, response_iterator: Iterator[GenerateResponse]) -> Iterator[GenerateResponse]:
        """Handle streaming response and collect content for history"""
        collected_content = ""

        for chunk in response_iterator:
            # Yield the chunk for the caller
            yield chunk

            # Collect content for history
            if hasattr(chunk, 'content') and chunk.content:
                collected_content += chunk.content

        # After streaming is complete, add the full response to history
        if collected_content:
            self.add_message('assistant', collected_content)

    def _format_messages_for_provider(self) -> List[Dict[str, str]]:
        """Format messages for provider API"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
            if m.role != 'system'  # System handled separately
        ]

    def _format_messages_for_provider_excluding_current(self) -> List[Dict[str, str]]:
        """Format messages for provider API, excluding the most recent user message"""
        # Exclude the last message if it's a user message (the current prompt)
        messages_to_send = self.messages[:-1] if self.messages and self.messages[-1].role == 'user' else self.messages
        return [
            {"role": m.role, "content": m.content}
            for m in messages_to_send
            if m.role != 'system'  # System handled separately
        ]

    def save(self, filepath: Union[str, Path]):
        """Save session to file"""
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BasicSession':
        """Load session from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        session = cls(system_prompt=data.get("system_prompt"))
        session.id = data["id"]
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.messages = [Message.from_dict(m) for m in data["messages"]]

        return session

    def _register_tools(self, tools: List[Callable]) -> List:
        """Register tools and return tool definitions"""
        from ..tools import ToolDefinition, register_tool

        registered_tools = []
        for func in tools:
            tool_def = ToolDefinition.from_function(func)
            register_tool(tool_def)
            registered_tools.append(tool_def)
        return registered_tools

    # Timeout management methods
    def get_timeout(self) -> Optional[float]:
        """Get the current HTTP request timeout in seconds."""
        if self.provider and hasattr(self.provider, 'get_timeout'):
            return self.provider.get_timeout()
        return None

    def set_timeout(self, timeout: float) -> None:
        """Set the HTTP request timeout in seconds."""
        if self.provider and hasattr(self.provider, 'set_timeout'):
            self.provider.set_timeout(timeout)

    def get_recovery_timeout(self) -> Optional[float]:
        """Get the current circuit breaker recovery timeout in seconds."""
        if self.provider and hasattr(self.provider, 'get_recovery_timeout'):
            return self.provider.get_recovery_timeout()
        return None

    def set_recovery_timeout(self, timeout: float) -> None:
        """Set the circuit breaker recovery timeout in seconds."""
        if self.provider and hasattr(self.provider, 'set_recovery_timeout'):
            self.provider.set_recovery_timeout(timeout)

    def get_tool_timeout(self) -> Optional[float]:
        """Get the current tool execution timeout in seconds."""
        if self.provider and hasattr(self.provider, 'get_tool_timeout'):
            return self.provider.get_tool_timeout()
        return None

    def set_tool_timeout(self, timeout: float) -> None:
        """Set the tool execution timeout in seconds."""
        if self.provider and hasattr(self.provider, 'set_tool_timeout'):
            self.provider.set_tool_timeout(timeout)