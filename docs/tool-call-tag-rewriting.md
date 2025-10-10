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

| CLI | Format | Example |
|-----|--------|---------|
| `qwen3` | `<|tool_call|>...JSON...</|tool_call|>` | Qwen3 models |
| `llama3` | `<function_call>...JSON...</function_call>` | LLaMA3 models, Crush CLI |
| `xml` | `<tool_call>...JSON...</tool_call>` | XML format, Gemini CLI |
| `gemma` | ````tool_code...JSON...```` | Gemma models |
| `codex` | `<|tool_call|>...JSON...</|tool_call|>` | Codex CLI |
| `crush` | `<function_call>...JSON...</function_call>` | Crush CLI |
| `gemini` | `<tool_call>...JSON...</tool_call>` | Gemini CLI |
| `openai` | `<|tool_call|>...JSON...</|tool_call|>` | OpenAI CLI |
| `anthropic` | `<function_call>...JSON...</function_call>` | Anthropic CLI |

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

## How It Works

1. **Pattern Detection**: The system detects existing tool call patterns in the response
2. **Tag Rewriting**: Replaces the detected patterns with the target format
3. **Streaming Support**: Handles partial tool calls that may be split across chunks
4. **Zero Configuration**: Works with any model and provider

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

## Best Practices

1. **Use Predefined Formats**: Prefer predefined formats (`"llama3"`, `"xml"`, etc.) over custom formats
2. **Test with Your CLI**: Verify the rewritten format works with your specific agentic CLI
3. **Streaming Considerations**: For streaming, ensure your CLI can handle partial tool calls
4. **Error Handling**: Always handle cases where tool calls might not be generated

## Examples

### Codex CLI Integration

```python
# Codex expects Qwen3 format
response = llm.generate(
    prompt,
    tools=tools,
    tool_call_tags="codex"  # Uses <|tool_call|> format
)
```

### Crush CLI Integration

```python
# Crush expects LLaMA3 format
response = llm.generate(
    prompt,
    tools=tools,
    tool_call_tags="crush"  # Uses <function_call> format
)
```

### Custom CLI Integration

```python
# Custom CLI with specific requirements
custom_tags = ToolCallTags("<my_custom_tool>", "</my_custom_tool>")
response = llm.generate(
    prompt,
    tools=tools,
    tool_call_tags=custom_tags
)
```

## Technical Details

- **Real-time Processing**: Tag rewriting happens during generation, not post-processing
- **Streaming Support**: Handles partial tool calls across streaming chunks
- **Pattern Matching**: Uses regex patterns to detect and rewrite tool calls
- **Memory Efficient**: Minimal overhead, processes only when needed
- **Provider Agnostic**: Works with all AbstractLLM providers

This feature makes AbstractLLM Core compatible with any agentic CLI that has specific tool call format requirements, providing a clean and simple solution for real-time tag rewriting.