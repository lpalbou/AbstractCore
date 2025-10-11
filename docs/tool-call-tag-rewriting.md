# Universal Tool Call Conversion System

> **Seamless, Zero-Configuration Tool Call Format Normalization**

## Overview

The AbstractCore Universal Tool Call Conversion System detects tool calls from various model formats and dynamically rewrites them to match the required format for different agentic tools. This enables your CLIs to work consistently across different language models with minimal configuration.

## Supported Input Formats

| Model Family | Input Format | Example |
|-------------|--------------|---------|
| Qwen3 | `<|tool_call|>...JSON...</|tool_call|>` | `<|tool_call|>{"name": "get_weather", "arguments": {...}}</|tool_call|>` |
| LLaMA | `<function_call>...JSON...</function_call>` | `<function_call>{"name": "calculate", "arguments": {...}}</function_call>` |
| Gemma | `` `tool_code...JSON...` `` | `` `tool_code{"name": "web_search", "arguments": {...}}`tool_code` `` |
| Generic XML | `<tool_call>...JSON...</tool_call>` | `<tool_call>{"name": "translate", "arguments": {...}}</tool_call>` |

## Conversion Capabilities

The conversion system works transparently:

1. **Input Detection**: Identifies tool calls across multiple formats
2. **Dynamic Rewriting**: Converts tool calls to the format required by the target system
3. **Flexible Configuration**: Adapts to different providers and models

### Example

```python
from abstractllm import create_llm

# Works identically across different models and formats
llm = create_llm("ollama", model="qwen3-coder:30b")
response = llm.generate(
    "What's the temperature in Paris?",
    tools=[{
        "name": "get_weather",
        "description": "Fetch current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }]
)

# Normalized tool call format
print(response.tool_calls)
# [{"id": "...", "type": "function", "function": {...}}]
# Note: Actual format may vary based on target system
```

## Advanced Tag Customization

### Custom Tag Configuration

You can customize tool call tags for precise control:

```python
# Custom tag configuration
llm = create_llm(
    "openai",
    model="gpt-4o-mini",
    tool_call_tags="ojlk,dfsd"  # Exact custom tags
)
```

#### Tag Handling Modes

1. **Automatic Formatting (Default)**
   ```python
   # When using a single tag
   llm = create_llm("model", tool_call_tags="mytag")
   # Becomes: <mytag>...JSON...</mytag>
   ```

2. **Exact Tag Matching**
   ```python
   # Use comma-separated tags for exact matching
   llm = create_llm("model", tool_call_tags="ojlk,dfsd")
   # Becomes: ojlk...JSON...dfsd
   ```

### Key Features

- **Exact Tag Preservation**: Precise control over tool call tagging
- **No Auto-Formatting**: Prevent automatic angle bracket wrapping
- **Multi-Format Support**: Works across Qwen, LLaMA, Gemma, and XML formats
- **Streaming Compatible**: Handles tool calls in real-time streaming

## Streaming Support

The conversion system works seamlessly in both streaming and non-streaming modes:

```python
# Streaming with automatic conversion
for chunk in llm.generate(
    "List files in my project",
    tools=[list_files_tool],
    stream=True
):
    print(chunk.content, end="")

    # Tool calls converted in real-time
    if chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            result = tool_call.execute()
```

## Performance Characteristics

- **First Chunk Latency**: <10ms
- **Conversion Overhead**: Negligible (<1ms)
- **Memory Usage**: Constant, bounded
- **Scalability**: Handles 1000+ chunks efficiently

## Best Practices

✅ Trust the default configuration
✅ Test with your specific agentic CLI
✅ Monitor tool execution via events
✅ No manual parsing required

## Compatibility Matrix

| Provider | Models | Tool Call Support | Conversion |
|----------|--------|-------------------|------------|
| OpenAI | GPT-4o, GPT-3.5 | Full | Automatic |
| Anthropic | Claude 3.5 Haiku | Full | Automatic |
| Ollama | Qwen, LLaMA, Gemma | Full | Automatic |
| MLX | Community Models | Full | Automatic |

## Troubleshooting

If you encounter issues:
1. Verify tool definition matches expected schema
2. Check model's specific tool call syntax
3. Use `verbose=True` in `create_llm()` for detailed logs

> **📚 Related Docs**:
> - [Unified Streaming Architecture](architecture.md#unified-streaming-architecture)
> - [Tool System Integration](architecture.md#tool-system-architecture)
> - [Production Examples](examples.md#production-patterns)