# Tool Call Tag Rewriting

## Overview

AbstractLLM Core now supports **real-time tool call tag rewriting** to accommodate different agentic CLI requirements. This feature allows you to rewrite tool call tags during generation, including streaming scenarios.

## Why This Feature?

Different agentic CLIs expect different tool call formats:

- **Codex**: Uses `<|tool_call|>...JSON...</|tool_call|>` format
- **Crush**: Uses `<function_call>...JSON...</function_call>` format  
- **Gemini CLI**: Uses `<tool_call>...JSON...</tool_call>` format
- **Custom CLIs**: May require completely custom formats

## Quick Start

### Basic Usage

```python
from abstractllm import create_llm
from abstractllm.tools.core import tool

@tool(description="Get weather information")
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

# Create LLM
llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")

# Default format (no rewriting)
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)

# Rewrite to LLaMA3 format for Crush CLI
response_crush = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather],
    tool_call_tags="llama3"  # Rewrites to <function_call>...JSON...</function_call>
)

# Rewrite to XML format for Gemini CLI
response_gemini = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather],
    tool_call_tags="xml"  # Rewrites to <tool_call>...JSON...</tool_call>
)
```

### Session Integration

```python
from abstractllm.core.session import BasicSession

session = BasicSession(provider=llm)

# Session also supports tool call tag rewriting
response = session.generate(
    "What's the weather in Paris?",
    tools=[get_weather],
    tool_call_tags="llama3"
)
```

### Streaming Support

```python
# Streaming with tag rewriting
for chunk in llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather],
    stream=True,
    tool_call_tags="llama3"
):
    print(chunk.content, end="")
```

## Supported Formats

### Predefined Formats

| CLI | Format | Example | Use Case |
|-----|--------|---------|----------|
| `qwen3` | `<|tool_call|>...JSON...</|tool_call|>` | `<|tool_call|>{"name": "get_weather", "arguments": {"location": "Paris"}}</|tool_call|>` | Qwen3 models, Codex CLI, OpenAI CLI |
| `llama3` | `<function_call>...JSON...</function_call>` | `<function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>` | LLaMA3 models, Crush CLI, Anthropic CLI |
| `xml` | `<tool_call>...JSON...</tool_call>` | `<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>` | XML format, Gemini CLI |
| `gemma` | ````tool_code...JSON...```` | ````tool_code\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n``` | Gemma models |
| `codex` | `<|tool_call|>...JSON...</|tool_call|>` | `<|tool_call|>{"name": "get_weather", "arguments": {"location": "Paris"}}</|tool_call|>` | Codex CLI (same as Qwen3) |
| `crush` | `<function_call>...JSON...</function_call>` | `<function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>` | Crush CLI (same as LLaMA3) |
| `gemini` | `<tool_call>...JSON...</tool_call>` | `<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>` | Gemini CLI (same as XML) |
| `openai` | `<|tool_call|>...JSON...</|tool_call|>` | `<|tool_call|>{"name": "get_weather", "arguments": {"location": "Paris"}}</|tool_call|>` | OpenAI CLI (same as Qwen3) |
| `anthropic` | `<function_call>...JSON...</function_call>` | `<function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>` | Anthropic CLI (same as LLaMA3) |

### Custom Formats

```python
from abstractllm.tools.tag_rewriter import ToolCallTags

# Custom format
custom_tags = ToolCallTags("<my_tool>", "</my_tool>")

response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather],
    tool_call_tags=custom_tags  # Rewrites to <my_tool>...JSON...</my_tool>
)
```

## Default Format

**The default tool call format is Qwen3**: `<|tool_call|>...JSON...</|tool_call|>`

This means:
- When no `tool_call_tags` parameter is specified, the system uses Qwen3 format
- This format is compatible with Codex CLI, OpenAI CLI, and most modern agentic CLIs
- The system automatically detects and rewrites tool calls to this format

## How It Works

1. **Pattern Detection**: The system detects existing tool call patterns in the response
2. **Tag Rewriting**: Replaces the detected patterns with the target format
3. **Streaming Support**: Handles partial tool calls that may be split across chunks
4. **Zero Configuration**: Works with any model and provider
5. **Multiple Tool Calls**: Handles multiple tool calls in a single response
6. **Mixed Content**: Works with tool calls embedded in regular text

## Architecture Support

The feature works with all supported architectures:

- **Qwen3**: Rewrites from `<|tool_call|>` to target format
- **LLaMA3**: Rewrites from `<function_call>` to target format  
- **Gemma3**: Rewrites from ````tool_code```` to target format
- **Generic**: Rewrites from plain JSON to target format

## Error Handling

- **Graceful Fallback**: If rewriting fails, the original response is returned
- **No Breaking Changes**: Existing code continues to work without modification
- **Logging**: Warnings are logged if rewriting encounters issues

## Event Monitoring

The tool call tag rewriting system integrates with AbstractLLM's event system for monitoring and observability:

```python
from abstractllm.events import EventType, on_global

def monitor_tool_calls(event):
    if event.type == EventType.TOOL_STARTED:
        print(f"🔧 Tool started: {event.data.get('tool_name', 'unknown')}")
    elif event.type == EventType.TOOL_COMPLETED:
        print(f"✅ Tool completed: {event.data.get('tool_name', 'unknown')}")

# Register event handler
on_global(EventType.TOOL_STARTED, monitor_tool_calls)
on_global(EventType.TOOL_COMPLETED, monitor_tool_calls)

# Use with tool call tag rewriting
llm = create_llm("openai", model="gpt-4o-mini", tool_call_tags="llama3")
response = llm.generate("What's the weather in Paris?", tools=[get_weather])
```

## Best Practices

1. **Use Predefined Formats**: Prefer predefined formats (`"llama3"`, `"xml"`, etc.) over custom formats
2. **Monitor Events**: Use the event system to track tool call execution and performance
3. **Test with Real CLIs**: Test your tool call formats with actual agentic CLIs to ensure compatibility
4. **Handle Errors Gracefully**: The system gracefully falls back to original content if rewriting fails
5. **Streaming Considerations**: For streaming, ensure your CLI can handle partial tool calls
6. **Error Handling**: Always handle cases where tool calls might not be generated

## Examples

### Codex CLI Integration

```python
from abstractllm import create_llm
from abstractllm.tools.core import tool

@tool(description="Get weather information")
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

# Codex CLI expects Qwen3 format (default)
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)
# Output: <|tool_call|>{"name": "get_weather", "arguments": {"location": "Paris"}}</|tool_call|>
```

### Crush CLI Integration

```python
# Crush CLI expects LLaMA3 format
llm = create_llm("openai", model="gpt-4o-mini", tool_call_tags="llama3")
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)
# Output: <function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>
```

### Gemini CLI Integration

```python
# Gemini CLI expects XML format
llm = create_llm("openai", model="gpt-4o-mini", tool_call_tags="xml")
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)
# Output: <tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>
```

### Streaming with Tool Call Rewriting

```python
# Streaming works seamlessly with tool call rewriting
for chunk in llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather],
    stream=True,
    tool_call_tags="llama3"
):
    print(chunk.content, end="")
# Output: <function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>
```

### Custom Format

```python
from abstractllm.tools.tag_rewriter import ToolCallTags

# Custom format for your specific CLI
custom_tags = ToolCallTags("<my_tool>", "</my_tool>")
llm = create_llm("openai", model="gpt-4o-mini", tool_call_tags=custom_tags)
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)
# Output: <my_tool>{"name": "get_weather", "arguments": {"location": "Paris"}}</my_tool>
```

### Multiple Tool Calls

```python
@tool(description="Calculate mathematical expressions")
def calculate(expression: str) -> str:
    return f"Result: {eval(expression)}"

# Multiple tool calls are handled correctly
response = llm.generate(
    "What's 2+2 and what's the weather in Paris?",
    tools=[calculate, get_weather],
    tool_call_tags="llama3"
)
# Output: <function_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</function_call>
#         <function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>
```

### Mixed Content

```python
# Tool calls embedded in regular text work correctly
response = llm.generate(
    "I'll help you with that. Let me get the weather for you.",
    tools=[get_weather],
    tool_call_tags="llama3"
)
# Output: I'll help you with that. Let me get the weather for you.
#         <function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>
```

## Technical Details

- **Real-time Processing**: Tag rewriting happens during generation, not post-processing
- **Streaming Support**: Handles partial tool calls across streaming chunks
- **Pattern Matching**: Uses regex patterns to detect and rewrite tool calls
- **Memory Efficient**: Minimal overhead, processes only when needed
- **Provider Agnostic**: Works with all AbstractLLM providers
- **Event Integration**: Full integration with AbstractLLM's event system for monitoring

This feature makes AbstractLLM Core compatible with any agentic CLI that has specific tool call format requirements, providing a clean and simple solution for real-time tag rewriting.