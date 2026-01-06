"""
Core tool definitions and abstractions.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


_MAX_TOOL_DESCRIPTION_CHARS = 200
_MAX_TOOL_WHEN_TO_USE_CHARS = 240
_MAX_TOOL_EXAMPLES = 3


def _first_non_empty_line(text: Optional[str]) -> str:
    if not text:
        return ""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _normalize_one_line(text: Optional[str]) -> str:
    """Collapse whitespace (including newlines) into a single, prompt-friendly line."""
    return " ".join(str(text or "").split()).strip()


def _validate_tool_metadata(*, name: str, description: str, when_to_use: Optional[str], examples: List[Dict[str, Any]]) -> None:
    if not description:
        raise ValueError(f"Tool '{name}': description must be a non-empty string")
    if len(description) > _MAX_TOOL_DESCRIPTION_CHARS:
        raise ValueError(
            f"Tool '{name}': description is too long ({len(description)} chars; max {_MAX_TOOL_DESCRIPTION_CHARS}). "
            "Keep it to a single short sentence; put detailed guidance in `when_to_use` or docs."
        )
    if when_to_use is not None and len(when_to_use) > _MAX_TOOL_WHEN_TO_USE_CHARS:
        raise ValueError(
            f"Tool '{name}': when_to_use is too long ({len(when_to_use)} chars; max {_MAX_TOOL_WHEN_TO_USE_CHARS}). "
            "Keep it to a single short sentence."
        )
    if len(examples) > _MAX_TOOL_EXAMPLES:
        raise ValueError(f"Tool '{name}': too many examples ({len(examples)}; max {_MAX_TOOL_EXAMPLES}).")


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by LLM"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None

    # Enhanced metadata for better LLM guidance
    tags: List[str] = field(default_factory=list)
    when_to_use: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Normalize to a single line for prompt-friendly catalogs.
        self.name = str(self.name or "").strip()
        self.description = _normalize_one_line(self.description)
        self.when_to_use = _normalize_one_line(self.when_to_use) if self.when_to_use else None
        self.tags = list(self.tags) if isinstance(self.tags, list) else []
        self.examples = list(self.examples) if isinstance(self.examples, list) else []
        _validate_tool_metadata(
            name=self.name,
            description=self.description,
            when_to_use=self.when_to_use,
            examples=self.examples,
        )

    @classmethod
    def from_function(cls, func: Callable) -> 'ToolDefinition':
        """Create tool definition from a function"""
        import inspect

        # Extract function name and docstring
        name = func.__name__
        # Tool `description` must be short; use the first docstring line (not the whole docstring).
        description = _first_non_empty_line(func.__doc__) or "No description provided"
        description = _normalize_one_line(description)
        if description != "No description provided":
            _validate_tool_metadata(name=name, description=description, when_to_use=None, examples=[])

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
        # Preserve a human/LLM-friendly field ordering (also improves UX when dumped as JSON).
        result: Dict[str, Any] = {"name": self.name, "description": self.description}

        if self.when_to_use:
            result["when_to_use"] = self.when_to_use

        result["parameters"] = self.parameters

        # Expose required args explicitly for hosts that render ToolDefinitions directly
        # (e.g. AbstractRuntime prompt payloads / debug traces). This is additive metadata:
        # providers still rely on JSON Schema `required` when building native tool payloads.
        try:
            required: List[str] = []
            for param_name, meta in (self.parameters or {}).items():
                if not isinstance(param_name, str) or not param_name.strip():
                    continue
                # Convention in this repo: absence of `default` means "required".
                if not isinstance(meta, dict) or "default" not in meta:
                    required.append(param_name)
            required.sort()
            if required:
                result["required_args"] = required
        except Exception:
            # Best-effort only; never break tool serialization.
            pass

        # Include enhanced metadata if available
        if self.tags:
            result["tags"] = self.tags
        if self.examples:
            result["examples"] = self.examples

        return result


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


def tool(
    func=None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    when_to_use: Optional[str] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    hide_args: Optional[List[str]] = None,
):
    """
    Enhanced decorator to convert a function into a tool with rich metadata.

    Usage:
        @tool
        def my_function(param: str) -> str:
            "Does something"
            return result

        # Or with enhanced metadata
        @tool(
            name="custom",
            description="Custom tool",
            tags=["utility", "helper"],
            when_to_use="When you need to perform X operation",
            examples=[
                {
                    "description": "Basic usage",
                    "arguments": {"param": "value"}
                }
            ]
        )
        def my_function(param: str) -> str:
            return result

        # Pass to generate like this:
        llm.generate("Do something", tools=[my_function])
    """
    def decorator(f):
        tool_name = name or f.__name__
        tool_description = description or _first_non_empty_line(f.__doc__) or f"Execute {tool_name}"
        tool_description = _normalize_one_line(tool_description)

        # Create tool definition from function and customize
        tool_def = ToolDefinition.from_function(f)
        tool_def.name = tool_name
        tool_def.description = tool_description

        # Add enhanced metadata
        tool_def.tags = tags or []
        tool_def.when_to_use = _normalize_one_line(when_to_use) if when_to_use else None
        tool_def.examples = list(examples) if isinstance(examples, list) else []

        # Optionally hide parameters from the exported schema (LLM-facing), while
        # keeping them accepted by the underlying Python callable for backwards
        # compatibility (e.g. legacy callers still passing deprecated kwargs).
        hidden = [str(a).strip() for a in (hide_args or []) if str(a).strip()]
        if hidden:
            for arg in hidden:
                if arg not in tool_def.parameters:
                    continue
                # Avoid hiding required args (no default), which would make the
                # tool schema incomplete for tool-call generation.
                if "default" not in tool_def.parameters.get(arg, {}):
                    raise ValueError(f"Tool '{tool_def.name}': cannot hide required arg '{arg}'")
                tool_def.parameters.pop(arg, None)

        _validate_tool_metadata(
            name=tool_def.name,
            description=tool_def.description,
            when_to_use=tool_def.when_to_use,
            examples=tool_def.examples,
        )

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
