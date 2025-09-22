"""
Universal tool support for AbstractLLM.

This package provides a unified tool system that works across all models
and providers, whether they have native tool APIs or require prompting.

Key components:
- Core types (ToolDefinition, ToolCall, ToolResult)
- Universal handler for all models
- Architecture-based parsing and formatting
- Tool registry for managing available tools

Example usage:
```python
from abstractllm.tools import ToolDefinition, UniversalToolHandler, register_tool

# Define a tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    '''List files in a directory.'''
    import os, fnmatch
    files = [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]
    return "\n".join(files)

# Register the tool
tool_def = ToolDefinition.from_function(list_files)

# Create handler for a model
handler = UniversalToolHandler("qwen3-coder:30b")

# Get tool prompt for prompted models
if handler.supports_prompted:
    tool_prompt = handler.format_tools_prompt([tool_def])
    print("Add this to your system prompt:")
    print(tool_prompt)

# Parse response for tool calls
response = "I'll list the files. <|tool_call|>{'name': 'list_files', 'arguments': {'directory': '.'}}"
tool_calls = handler.parse_response(response, mode="prompted")

if tool_calls.has_tool_calls():
    print("Tool calls found:", tool_calls.tool_calls)
```
"""

# Core types
from .core import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolCallResponse,
    tool
)

# Handler
from .handler import (
    UniversalToolHandler,
    create_handler
)

# Parser functions
from .parser import (
    detect_tool_calls,
    parse_tool_calls,
    format_tool_prompt
)

# Registry
from .registry import (
    ToolRegistry,
    register_tool,
    get_registry,
    execute_tool,
    execute_tools,
    clear_registry
)

__all__ = [
    # Core types
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolCallResponse",
    "tool",

    # Handler
    "UniversalToolHandler",
    "create_handler",

    # Parser
    "detect_tool_calls",
    "parse_tool_calls",
    "format_tool_prompt",

    # Registry
    "ToolRegistry",
    "register_tool",
    "get_registry",
    "execute_tool",
    "execute_tools",
    "clear_registry",
]