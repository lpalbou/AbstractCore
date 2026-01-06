"""
Tool registry for managing and executing tools.

This module provides a centralized registry for managing available tools
and executing them safely.
"""

import time
import warnings
from typing import Dict, List, Any, Callable, Optional, Union
from functools import wraps

from .core import ToolDefinition, ToolCall, ToolResult
from ..events import EventType, emit_global, create_tool_event
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


def _error_from_output(value: Any) -> Optional[str]:
    """Detect tool failures reported as string outputs (instead of exceptions)."""
    # Allow tools to return structured outputs while still communicating failure
    # without raising exceptions. We only treat this as an error when the tool
    # explicitly marks itself as unsuccessful.
    if isinstance(value, dict):
        success = value.get("success")
        ok = value.get("ok")
        if success is False or ok is False:
            err = value.get("error") or value.get("message") or "Tool reported failure"
            text = str(err).strip()
            return text or "Tool reported failure"
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("Error:"):
        cleaned = text[len("Error:") :].strip()
        return cleaned or text
    if text.startswith(("❌", "🚫", "⏰")):
        cleaned = text.lstrip("❌🚫⏰").strip()
        if cleaned.startswith("Error:"):
            cleaned = cleaned[len("Error:") :].strip()
        return cleaned or text
    return None


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
        start_time = time.time()

        # Note: Removing TOOL_CALLED event - redundant with BEFORE_TOOL_EXECUTION
        # BEFORE_TOOL_EXECUTION is emitted by the provider and covers this

        tool_def = self.get_tool(tool_call.name)
        if not tool_def:
            duration_ms = (time.time() - start_time) * 1000
            error_result = ToolResult(
                call_id=tool_call.call_id or "",
                output="",
                error=f"Tool '{tool_call.name}' not found",
                success=False
            )

            # Emit tool result event
            result_data = create_tool_event(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                success=False,
                error=error_result.error
            )
            emit_global(EventType.TOOL_COMPLETED, result_data,
                       source="ToolRegistry", duration_ms=duration_ms)

            return error_result

        if not tool_def.function:
            duration_ms = (time.time() - start_time) * 1000
            error_result = ToolResult(
                call_id=tool_call.call_id or "",
                output="",
                error=f"Tool '{tool_call.name}' has no executable function",
                success=False
            )

            # Emit tool result event
            result_data = create_tool_event(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                success=False,
                error=error_result.error
            )
            emit_global(EventType.TOOL_COMPLETED, result_data,
                       source="ToolRegistry", duration_ms=duration_ms)

            return error_result

        try:
            from .arg_canonicalizer import canonicalize_tool_arguments

            arguments = canonicalize_tool_arguments(tool_call.name, tool_call.arguments)

            # Execute the function with the provided arguments
            result = tool_def.function(**arguments)
            duration_ms = (time.time() - start_time) * 1000

            implied_error = _error_from_output(result)
            if implied_error is not None:
                error_result = ToolResult(
                    call_id=tool_call.call_id or "",
                    # Preserve structured outputs for post-mortem evidence/provenance.
                    # For string-only error outputs, store the message in `error` and keep output empty.
                    output=result if not isinstance(result, str) else "",
                    error=implied_error,
                    success=False,
                )

                # Emit tool error event
                result_data = create_tool_event(
                    tool_name=tool_call.name,
                    arguments=arguments,
                    success=False,
                    error=implied_error,
                )
                emit_global(
                    EventType.TOOL_COMPLETED,
                    result_data,
                    source="ToolRegistry",
                    duration_ms=duration_ms,
                )

                return error_result

            success_result = ToolResult(
                call_id=tool_call.call_id or "",
                output=result,
                success=True
            )

            # Emit successful tool result event
            result_data = create_tool_event(
                tool_name=tool_call.name,
                arguments=arguments,
                result=result,
                success=True
            )
            emit_global(EventType.TOOL_COMPLETED, result_data,
                       source="ToolRegistry", duration_ms=duration_ms)

            return success_result

        except TypeError as e:
            # Some models include wrapper/meta keys ("name", nested "arguments") or
            # stray extras in tool kwargs. Retry once with a sanitized argument dict.
            try:
                wrapper_keys = {"name", "arguments", "call_id", "id"}
                from .arg_canonicalizer import canonicalize_tool_arguments

                args = canonicalize_tool_arguments(tool_call.name, tool_call.arguments)
                for _ in range(4):
                    inner = args.get("arguments")
                    if not isinstance(inner, dict):
                        break
                    extras = {k: v for k, v in args.items() if k not in wrapper_keys}
                    merged = dict(inner)
                    for k, v in extras.items():
                        merged.setdefault(k, v)
                    args = merged

                allowed = set(tool_def.parameters.keys()) if isinstance(tool_def.parameters, dict) else set()
                if allowed:
                    args = {k: v for k, v in args.items() if k in allowed}

                if args != dict(tool_call.arguments or {}):
                    result = tool_def.function(**args)
                    duration_ms = (time.time() - start_time) * 1000
                    success_result = ToolResult(
                        call_id=tool_call.call_id or "",
                        output=result,
                        success=True,
                    )
                    result_data = create_tool_event(
                        tool_name=tool_call.name,
                        arguments=args,
                        result=result,
                        success=True,
                    )
                    emit_global(
                        EventType.TOOL_COMPLETED,
                        result_data,
                        source="ToolRegistry",
                        duration_ms=duration_ms,
                    )
                    return success_result
            except Exception:
                pass

            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Invalid arguments for tool '{tool_call.name}': {e}"
            logger.warning(error_msg)

            error_result = ToolResult(
                call_id=tool_call.call_id or "",
                output="",
                error=error_msg,
                success=False
            )

            # Emit tool error event
            result_data = create_tool_event(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                success=False,
                error=error_msg,
                error_type="TypeError"
            )
            emit_global(EventType.TOOL_COMPLETED, result_data,
                       source="ToolRegistry", duration_ms=duration_ms)

            return error_result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Error executing tool '{tool_call.name}': {e}"
            logger.error(error_msg)

            error_result = ToolResult(
                call_id=tool_call.call_id or "",
                output="",
                error=error_msg,
                success=False
            )

            # Emit tool error event
            result_data = create_tool_event(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                success=False,
                error=error_msg,
                error_type=type(e).__name__
            )
            emit_global(EventType.TOOL_COMPLETED, result_data,
                       source="ToolRegistry", duration_ms=duration_ms)

            return error_result

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
    warnings.warn(
        "Global tool registration is deprecated. Prefer passing tools explicitly to generate() "
        "and executing tool calls via a host-configured ToolExecutor.",
        DeprecationWarning,
        stacklevel=2,
    )
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


def clear_registry():
    """
    Clear all tools from the global registry.

    Useful for testing and resetting tool state.
    """
    return _global_registry.clear()


__all__ = [
    "ToolRegistry",
    "get_registry",
    "register_tool",
    "execute_tool",
    "execute_tools",
    "clear_registry",
]
