"""
Core tool definitions and abstractions.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by LLM"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None

    @classmethod
    def from_function(cls, func: Callable) -> 'ToolDefinition':
        """Create tool definition from a function"""
        import inspect

        # Extract function name and docstring
        name = func.__name__
        description = func.__doc__ or "No description provided"

        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = {}

        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type

            # Try to infer type from annotation
            if param.annotation != param.empty:
                if param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"

            if param.default != param.empty:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            function=func
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class ToolCall:
    """Represents a tool call from the LLM"""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result of tool execution"""
    call_id: str
    output: Any
    error: Optional[str] = None
    success: bool = True


@dataclass
class ToolCallResponse:
    """Response containing content and tool calls"""
    content: str
    tool_calls: List[ToolCall]
    raw_response: Any = None

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls"""
        return bool(self.tool_calls)


def function_to_tool_definition(func: Callable) -> ToolDefinition:
    """
    Convert a function to a ToolDefinition.

    This is a convenience function that wraps ToolDefinition.from_function()
    """
    return ToolDefinition.from_function(func)