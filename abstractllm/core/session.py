"""
BasicSession for conversation tracking.
Target: <500 lines maximum.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import json
import uuid

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
                 system_prompt: Optional[str] = None):
        """Initialize basic session"""

        self.provider = provider
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.messages: List[Message] = []
        self.system_prompt = system_prompt

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

    def generate(self, prompt: str, **kwargs) -> GenerateResponse:
        """Generate response using provider"""
        if not self.provider:
            raise ValueError("No provider configured")

        # Add user message
        self.add_message('user', prompt)

        # Format messages for provider
        messages = self._format_messages_for_provider()

        # Call provider
        response = self.provider.generate(
            prompt=prompt,
            messages=messages,
            system_prompt=self.system_prompt,
            **kwargs
        )

        # Add assistant response
        if hasattr(response, 'content') and response.content:
            self.add_message('assistant', response.content)

        return response

    def _format_messages_for_provider(self) -> List[Dict[str, str]]:
        """Format messages for provider API"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
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