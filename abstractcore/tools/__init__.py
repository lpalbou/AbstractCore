"""
Universal tool support for AbstractCore.

This package provides a unified tool system that works across all models
and providers, whether they have native tool APIs or require prompting.

Tool Execution Modes
--------------------

AbstractCore supports two tool execution modes:

**Passthrough Mode (Default)** - execute_tools=False
    AbstractCore detects/parses tool calls (native API fields or prompted tags)
    and returns structured `GenerateResponse.tool_calls` plus cleaned `content`.
    Downstream runtimes (AbstractRuntime, Codex, Claude Code) execute tools.
    Use case: Agent loops, custom orchestration, multi-step workflows.

**Direct Execution Mode** - execute_tools=True
    AbstractCore parses and executes tools internally using the
    global registry. Requires register_tool() for each tool.
    Use case: Simple scripts, single-turn tool use.

Key Components
--------------
- Core types (ToolDefinition, ToolCall, ToolResult)
- Universal handler for all models
- Architecture-based parsing and formatting
- Tool registry for managing available tools

Example: Passthrough Mode (Default)
-----------------------------------
```python
from abstractcore import create_llm
from abstractcore.tools import tool

@tool(name="get_weather", description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

llm = create_llm("ollama", model="qwen3:4b")
response = llm.generate("Weather in Paris?", tools=[get_weather])
# response.tool_calls contains structured tool calls; response.content is cleaned
```

Example: Direct Execution Mode
------------------------------
```python
from abstractcore import create_llm
from abstractcore.tools import tool, register_tool

@tool(name="get_weather", description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

register_tool(get_weather)  # Required for direct execution

llm = create_llm("ollama", model="qwen3:4b", execute_tools=True)
response = llm.generate("Weather in Paris?", tools=[get_weather])
# response.content has executed tool results
```

Note: The @tool decorator creates metadata but does NOT auto-register.
Tools are passed explicitly to generate(). Use register_tool() only
when using direct execution mode.
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
