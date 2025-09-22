"""
Tool registry for managing and executing tools.

This module provides a centralized registry for managing available tools
and executing them safely.
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Union
from functools import wraps

from .core import ToolDefinition, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: Union[ToolDefinition, Callable]) -> ToolDefinition:
        """
        Register a tool in the registry.

        Args:
            tool: Tool definition or callable function

        Returns:
            The registered ToolDefinition
        """
        if callable(tool):
            tool_def = ToolDefinition.from_function(tool)
        elif isinstance(tool, ToolDefinition):
            tool_def = tool
        else:
            raise ValueError(f"Tool must be ToolDefinition or callable, got {type(tool)}")

        self._tools[tool_def.name] = tool_def
        logger.debug(f"Registered tool: {tool_def.name}")
        return tool_def

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """
        Get all registered tools.

        Returns:
            List of all ToolDefinitions
        """
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """
        Get names of all registered tools.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: Tool call to execute

        Returns:
            ToolResult with execution result
        """
        tool_def = self.get_tool(tool_call.name)
        if not tool_def:
            return ToolResult(
                call_id=tool_call.call_id or "",
                output="",
                error=f"Tool '{tool_call.name}' not found",
                success=False
            )

        if not tool_def.function:
            return ToolResult(
                call_id=tool_call.call_id or "",
                output="",
                error=f"Tool '{tool_call.name}' has no executable function",
                success=False
            )

        try:
            # Execute the function with the provided arguments
            result = tool_def.function(**tool_call.arguments)
            return ToolResult(
                call_id=tool_call.call_id or "",
                output=result,
                success=True
            )

        except TypeError as e:
            error_msg = f"Invalid arguments for tool '{tool_call.name}': {e}"
            logger.warning(error_msg)
            return ToolResult(
                call_id=tool_call.call_id or "",
                output="",
                error=error_msg,
                success=False
            )

        except Exception as e:
            error_msg = f"Error executing tool '{tool_call.name}': {e}"
            logger.error(error_msg)
            return ToolResult(
                call_id=tool_call.call_id or "",
                output="",
                error=error_msg,
                success=False
            )

    def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of ToolResults
        """
        return [self.execute_tool(call) for call in tool_calls]

    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        logger.debug("Cleared all tools from registry")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


# Global registry instance
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry.

    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


def register_tool(tool: Union[ToolDefinition, Callable]) -> ToolDefinition:
    """
    Register a tool in the global registry.

    Args:
        tool: Tool definition or callable function

    Returns:
        The registered ToolDefinition
    """
    return _global_registry.register(tool)


def execute_tool(tool_call: ToolCall) -> ToolResult:
    """
    Execute a tool call using the global registry.

    Args:
        tool_call: Tool call to execute

    Returns:
        ToolResult with execution result
    """
    return _global_registry.execute_tool(tool_call)


def execute_tools(tool_calls: List[ToolCall]) -> List[ToolResult]:
    """
    Execute multiple tool calls using the global registry.

    Args:
        tool_calls: List of tool calls to execute

    Returns:
        List of ToolResults
    """
    return _global_registry.execute_tools(tool_calls)


def tool(func: Callable) -> Callable:
    """
    Decorator to register a function as a tool.

    Args:
        func: Function to register as a tool

    Returns:
        The original function (unchanged)
    """
    register_tool(func)
    return func


# Convenience decorator alias
register = tool