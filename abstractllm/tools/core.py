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


def tool(func=None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Simple decorator to convert a function into a tool.

    Usage:
        @tool
        def my_function(param: str) -> str:
            "Does something"
            return result

        # Or with custom name/description
        @tool(name="custom", description="Custom tool")
        def my_function(param: str) -> str:
            return result

        # Pass to generate like this:
        llm.generate("Do something", tools=[my_function])
    """
    def decorator(f):
        tool_name = name or f.__name__
        tool_description = description or f.__doc__ or f"Execute {tool_name}"

        # Create tool definition from function and customize
        tool_def = ToolDefinition.from_function(f)
        tool_def.name = tool_name
        tool_def.description = tool_description

        # Attach tool definition to function for easy access
        f._tool_definition = tool_def
        f.tool_name = tool_name

        return f

    if func is None:
        # Called with arguments: @tool(name="custom")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)