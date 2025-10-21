"""
Core types for AbstractCore.
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


@dataclass
class Message:
    """Represents a single message in a conversation"""
    role: str
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    @property
    def name(self) -> Optional[str]:
        """Get the username from metadata"""
        return self.metadata.get('name') if self.metadata else None

    @name.setter
    def name(self, value: Optional[str]):
        """Set the username in metadata"""
        if self.metadata is None:
            self.metadata = {}
        if value is not None:
            self.metadata['name'] = value
        elif 'name' in self.metadata:
            del self.metadata['name']

    @property
    def location(self) -> Optional[str]:
        """Get the location from metadata"""
        return self.metadata.get('location') if self.metadata else None

    @location.setter
    def location(self, value: Optional[str]):
        """Set the location in metadata"""
        if self.metadata is None:
            self.metadata = {}
        if value is not None:
            self.metadata['location'] = value
        elif 'location' in self.metadata:
            del self.metadata['location']

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        # Handle backward compatibility: if 'name' exists as separate field, move to metadata
        metadata = data.get("metadata", {}).copy() if data.get("metadata") else {}
        
        # Backward compatibility: migrate old 'name' field to metadata
        if "name" in data and data["name"] is not None:
            metadata["name"] = data["name"]

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=metadata if metadata else None
        )


@dataclass
class GenerateResponse:
    """Response from LLM generation"""
    content: Optional[str] = None
    raw_response: Any = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    gen_time: Optional[float] = None  # Generation time in milliseconds

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls"""
        return bool(self.tool_calls)

    def get_tools_executed(self) -> List[str]:
        """Get list of tools that were called"""
        if not self.tool_calls:
            return []
        return [call.get('name', '') for call in self.tool_calls]

    def get_summary(self) -> str:
        """Get a summary of the response"""
        parts = []
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.usage:
            parts.append(f"Tokens: {self.usage.get('total_tokens', 'unknown')}")
        if self.gen_time:
            parts.append(f"Time: {self.gen_time:.1f}ms")
        if self.tool_calls:
            parts.append(f"Tools: {len(self.tool_calls)} executed")
        return " | ".join(parts)
    
    @property
    def input_tokens(self) -> Optional[int]:
        """Get input tokens with consistent terminology (prompt_tokens or input_tokens)."""
        if not self.usage:
            return None
        return self.usage.get('input_tokens') or self.usage.get('prompt_tokens')
    
    @property
    def output_tokens(self) -> Optional[int]:
        """Get output tokens with consistent terminology (completion_tokens or output_tokens)."""
        if not self.usage:
            return None
        return self.usage.get('output_tokens') or self.usage.get('completion_tokens')
    
    @property
    def total_tokens(self) -> Optional[int]:
        """Get total tokens."""
        if not self.usage:
            return None
        return self.usage.get('total_tokens')