# Tool Call Tag Rewriting

> **Real-time tool call format conversion for agentic CLI compatibility**

## Why Tool Call Tag Rewriting?

Different agentic CLIs expect different tool call formats. AbstractCore automatically converts tool calls to the format your CLI needs:

- **Codex CLI**: `<|tool_call|>...JSON...</|tool_call|>` (default)
- **Crush CLI**: `<function_call>...JSON...</function_call>`
- **Gemini CLI**: `<tool_call>...JSON...</tool_call>`
- **Custom CLIs**: Any format you need

## Quick Start

```python
from abstractllm import create_llm

# Define your tool function (no decorators needed)
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

llm = create_llm("ollama", model="qwen3-coder:30b")

# 1. Default format (Qwen3/Codex compatible)
response = llm.generate(
    "What's the weather in Paris?",
    tools=[{
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }]
)
# Output: <|tool_call|>{"name": "get_weather", "arguments": {"location": "Paris"}}</|tool_call|>

# 2. Crush CLI format
response = llm.generate(
    "What's the weather in Paris?",
    tools=[tool_def],
    tool_call_tags="llama3"
)
# Output: <function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>

# 3. Custom format with comma-separated tags
response = llm.generate(
    "What's the weather in Paris?",
    tools=[tool_def],
    tool_call_tags="START,END"  # Results in: START...JSON...END
)
```

> **📋 Related Documentation**: [Tool Calling Examples](examples.md#tool-calling-examples) | [Getting Started](getting-started.md#tool-calling-llm-with-superpowers)

## Streaming Support

**Real-time tool call rewriting during streaming** - Works perfectly with the unified streaming architecture:

```python
# Streaming with automatic tag rewriting
print("AI: ", end="", flush=True)
for chunk in llm.generate(
    "What's the weather in Paris?",
    tools=[tool_def],
    stream=True,
    tool_call_tags="llama3"
):
    print(chunk.content, end="", flush=True)

    # Tool calls are detected and rewritten in real-time
    if chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            result = tool_call.execute()
            print(f"\n🛠️ Tool executed: {result}")
```

> **⚡ Performance**: Streaming with tag rewriting delivers first chunk in <10ms with zero buffering overhead.

## Session Integration

Sessions work seamlessly with tool call tag rewriting:

```python
from abstractllm import BasicSession

session = BasicSession(provider=llm)
response = session.generate(
    "What's the weather in Paris?",
    tools=[tool_def],
    tool_call_tags="llama3"
)
```

## Supported Formats

### Predefined Formats (Common CLIs)

| Format | Tags | Best For |
|--------|------|----------|
| **Default** (qwen3) | `<|tool_call|>...JSON...</|tool_call|>` | Codex CLI, OpenAI tools |
| `llama3` | `<function_call>...JSON...</function_call>` | Crush CLI, Anthropic tools |
| `xml` | `<tool_call>...JSON...</tool_call>` | Gemini CLI, XML-based tools |
| `gemma` | ````tool_code...JSON...```` | Gemma models |

### Custom Formats

```python
# Option 1: Simple comma-separated tags
tool_call_tags="START,END"  # Results in: START...JSON...END

# Option 2: Advanced configuration
from abstractllm.tools.tag_rewriter import ToolCallTags
custom_tags = ToolCallTags("<my_tool>", "</my_tool>", auto_format=False)
response = llm.generate("...", tools=[...], tool_call_tags=custom_tags)
```

## How It Works

**Automatic & Zero-Configuration**:

1. **Default Format**: Qwen3 (`<|tool_call|>...JSON...</|tool_call|>`) - compatible with Codex CLI and most modern tools
2. **Auto-Detection**: Detects tool calls from any model architecture (Qwen, LLaMA, Gemma, etc.)
3. **Real-Time Rewriting**: Converts to your target format during generation (streaming or non-streaming)
4. **Graceful Fallback**: Returns original content if rewriting fails

> **🔧 Technical**: Works with all providers (OpenAI, Anthropic, Ollama, MLX) and handles multiple tool calls, partial streaming chunks, and mixed content.

## Common Integration Examples

### Codex CLI (Default - No Configuration Needed)

```python
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
response = llm.generate("List files in current directory", tools=[list_files_tool])
# Automatic qwen3 format: <|tool_call|>{"name": "list_files"}...</|tool_call|>
```

### Crush CLI

```python
llm = create_llm("ollama", model="qwen3-coder:30b", tool_call_tags="llama3")
response = llm.generate("Calculate 15*23", tools=[calculator_tool])
# LLaMA3 format: <function_call>{"name": "calculate"}...</function_call>
```

### Custom Agentic CLI

```python
# Your CLI expects: [TOOL]...JSON...[/TOOL]
llm = create_llm("openai", model="gpt-4o-mini", tool_call_tags="[TOOL],[/TOOL]")
response = llm.generate("Get weather", tools=[weather_tool])
# Custom format: [TOOL]{"name": "get_weather"}...[/TOOL]
```

## Event Monitoring (Optional)

```python
from abstractllm.events import EventType, on_global

def log_tool_usage(event):
    print(f"🔧 Tool: {event.data.get('tool_calls', [])[0].name}")

on_global(EventType.TOOL_STARTED, log_tool_usage)
```

## Best Practices

✅ **Use predefined formats** (`"llama3"`, `"xml"`) when possible
✅ **Test with your actual CLI** to verify compatibility
✅ **Use events for monitoring** in production
✅ **Default format works** with most modern agentic CLIs

> **📚 Next Steps**: [Unified Streaming Architecture](architecture.md#unified-streaming-architecture) | [Tool System Integration](architecture.md#tool-system-architecture) | [Production Examples](examples.md#production-patterns)