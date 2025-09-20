"""
Core types for AbstractLLM.
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
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "name": self.name,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            name=data.get("name"),
            metadata=data.get("metadata")
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
        if self.tool_calls:
            parts.append(f"Tools: {len(self.tool_calls)} executed")
        return " | ".join(parts)