# Tool Call Syntax Rewriting

> **Seamless Tool Call Format Conversion Across Models and Environments**

AbstractCore provides two complementary systems for tool call format conversion, serving different use cases:

1. **Tag Rewriter** (`tag_rewriter.py`) - For **Core Library** (Python API)
2. **Syntax Rewriter** (`syntax_rewriter.py`) - For **Server** (REST API)

## Overview

Different LLM models use different formats for tool calls:
- **Qwen3**: `<|tool_call|>...JSON...</|tool_call|>`
- **LLaMA3**: `<function_call>...JSON...</function_call>`
- **Gemma**: `` `tool_code...JSON...` ``
- **OpenAI**: Structured JSON in API response
- **Codex/Agentic CLIs**: Custom formats

AbstractCore automatically detects and converts between these formats.

---

## Core Library (Python API) - Tag Rewriter

Used when calling LLMs directly via Python. Handles real-time tag rewriting for streaming responses.

### Quick Start

```python
from abstractllm import create_llm

# Automatic format detection and conversion
llm = create_llm("ollama", model="qwen3-coder:30b")

response = llm.generate(
    "What's the weather in Paris?",
    tools=[{
        "name": "get_weather",
        "description": "Fetch current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }]
)

# Tool calls automatically normalized
print(response.tool_calls)
```

### Custom Tag Configuration

For specific agentic CLI requirements:

```python
from abstractllm import create_llm

# Option 1: Simple tag name (auto-formatted with angle brackets)
llm = create_llm(
    "ollama",
    model="qwen3-coder:30b",
    tool_call_tags="mytag"  # Becomes: <mytag>...JSON...</mytag>
)

# Option 2: Exact tag control (comma-separated, no auto-formatting)
llm = create_llm(
    "ollama", 
    model="qwen3-coder:30b",
    tool_call_tags="ojlk,dfsd"  # Becomes: ojlk...JSON...dfsd (exact)
)

# Option 3: ToolCallTags object for full control
from abstractllm.tools.tag_rewriter import ToolCallTags

custom_tags = ToolCallTags(
    start_tag="<custom_tool>",
    end_tag="</custom_tool>",
    preserve_json=True,
    auto_format=False  # Use tags exactly as specified
)

llm = create_llm(
    "ollama",
    model="qwen3-coder:30b",
    tool_call_tags=custom_tags
)
```

### Streaming Support

Tag rewriting works seamlessly with streaming:

```python
for chunk in llm.generate(
    "List files in my project",
    tools=[list_files_tool],
    stream=True
):
    print(chunk.content, end="")
    
    # Tool calls converted in real-time
    if chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            result = execute_tool(tool_call)
```

### Predefined CLI Formats

Built-in support for popular agentic CLIs:

```python
from abstractllm.tools.tag_rewriter import create_tag_rewriter

# Codex CLI
rewriter = create_tag_rewriter("codex")  # Qwen3 format

# Crush CLI  
rewriter = create_tag_rewriter("crush")  # LLaMA3 format

# Gemini CLI
rewriter = create_tag_rewriter("gemini")  # XML format

# Available: "qwen3", "llama3", "xml", "gemma", "codex", "crush", "gemini", "openai", "anthropic"
```

### Manual Tag Rewriting

For advanced use cases:

```python
from abstractllm.tools.tag_rewriter import ToolCallTagRewriter, ToolCallTags

# Create custom rewriter
tags = ToolCallTags("<start>", "<end>")
rewriter = ToolCallTagRewriter(tags)

# Rewrite text
original = '<|tool_call|>{"name": "get_weather"}</|tool_call|>'
rewritten = rewriter.rewrite_text(original)
# Result: <start>{"name": "get_weather"}<end>

# Check for tool calls
has_tool_call = rewriter.is_tool_call(original)

# Extract tool calls
tool_calls = rewriter.extract_tool_calls(original)
```

---

## Server (REST API) - Syntax Rewriter

Used by the AbstractCore server to convert tool call formats for HTTP API clients.

### How It Works

The server automatically detects the target format and converts tool calls:

1. **Format Detection**: Based on model, user-agent, or explicit `agent_format` parameter
2. **Content Rewriting**: Converts tool call syntax in response content
3. **OpenAI Format**: Converts to OpenAI API structure when needed

### Agent Format Parameter

Specify the target format explicitly:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Get weather in Paris"}],
    "tools": [...],
    "agent_format": "codex"
  }'
```

**Supported agent_format values:**
- `"passthrough"` - No conversion (OpenAI models)
- `"openai"` - OpenAI format
- `"codex"` - Codex CLI format
- `"qwen3"` - Qwen3 format
- `"llama3"` - LLaMA3 format
- `"gemma"` - Gemma format
- `"xml"` - XML format

### Automatic Format Detection

The server auto-detects format based on:

1. **Custom Headers**: `X-Agent-Type: codex`
2. **User Agent**: Detects Codex and other CLIs
3. **Model Name**: Matches model patterns (qwen → qwen3, llama → llama3)
4. **Default**: OpenAI format for maximum compatibility

```bash
# Using custom header
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-Agent-Type: codex" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### OpenAI Format Conversion

For OpenAI-compatible clients:

```python
import openai

# Point to AbstractCore server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Works with any model via AbstractCore
response = client.chat.completions.create(
    model="ollama/qwen3-coder:30b",  # Non-OpenAI model
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[...]
)

# Tool calls automatically in OpenAI format
print(response.choices[0].message.tool_calls)
```

### Programmatic Usage

Use the syntax rewriter directly:

```python
from abstractllm.tools.syntax_rewriter import (
    ToolCallSyntaxRewriter,
    SyntaxFormat,
    create_openai_rewriter,
    create_codex_rewriter
)

# Create rewriter
rewriter = ToolCallSyntaxRewriter(
    target_format=SyntaxFormat.CODEX,
    model_name="qwen3-coder"
)

# Rewrite content
original = '<|tool_call|>{"name": "get_weather"}</|tool_call|>'
rewritten = rewriter.rewrite_content(original)

# Convert to OpenAI format
from abstractllm.tools.core import ToolCall

tool_calls = [ToolCall(name="get_weather", arguments={"location": "Paris"})]
openai_format = rewriter.convert_to_openai_format(tool_calls)
```

---

## Comparison: Tag Rewriter vs Syntax Rewriter

| Feature | Tag Rewriter (Core) | Syntax Rewriter (Server) |
|---------|---------------------|--------------------------|
| **Use Case** | Python API, Streaming | REST API, HTTP clients |
| **Target** | Providers, LLMs | API responses, Agentic CLIs |
| **When Used** | Direct Python usage | Server/HTTP endpoints |
| **Streaming** | ✅ Real-time | ✅ Supported |
| **Format Detection** | Manual via parameters | Automatic + manual |
| **OpenAI Conversion** | ❌ Not applicable | ✅ Full support |
| **Custom Tags** | ✅ Full control | ✅ Via agent_format |
| **File** | `tag_rewriter.py` | `syntax_rewriter.py` |

---

## Supported Input Formats

Both systems detect and convert from these formats:

| Model Family | Input Format | Example |
|-------------|--------------|---------|
| Qwen3 | `<|tool_call|>...JSON...</|tool_call|>` | Qwen3-Coder, Qwen3-Chat |
| LLaMA3 | `<function_call>...JSON...</function_call>` | LLaMA3, Mistral |
| Gemma | `` `tool_code...JSON...` `` | Gemma-2, CodeGemma |
| XML | `<tool_call>...JSON...</tool_call>` | Claude, custom |
| OpenAI | Structured API response | GPT-4, GPT-3.5 |

---

## Performance Characteristics

### Tag Rewriter (Core Library)
- **First Chunk Latency**: <5ms
- **Conversion Overhead**: <1ms per chunk
- **Memory**: Constant, O(1)
- **Streaming**: Real-time, zero buffering for simple tags

### Syntax Rewriter (Server)
- **Conversion Time**: <10ms per response
- **Batch Support**: ✅ Handles multiple tool calls
- **Memory**: O(n) where n = content length
- **Caching**: Pattern compilation cached

---

## Best Practices

### For Python API Users
✅ Use default settings for most cases  
✅ Only customize tags when integrating with specific CLIs  
✅ Test tag format with your agentic CLI first  
✅ Use predefined formats when available (`create_tag_rewriter("codex")`)  

### For Server Users
✅ Use `agent_format` parameter for explicit control  
✅ Set `X-Agent-Type` header for automatic detection  
✅ Test with OpenAI client for compatibility  
✅ Monitor logs for format detection issues  

### For Both
✅ Avoid double-tagging by checking output format  
✅ Enable verbose logging (`verbose=True`) for debugging  
✅ Validate tool call JSON structure  
✅ Test streaming scenarios thoroughly  

---

## Compatibility Matrix

| Provider | Models | Python API | REST API | Auto-Detect |
|----------|--------|------------|----------|-------------|
| OpenAI | GPT-4o, GPT-3.5 | ✅ | ✅ | ✅ |
| Anthropic | Claude 3.5 | ✅ | ✅ | ✅ |
| Ollama | Qwen, LLaMA, Gemma | ✅ | ✅ | ✅ |
| LMStudio | Community Models | ✅ | ✅ | ✅ |
| MLX | Apple Silicon Models | ✅ | ✅ | ✅ |
| HuggingFace | Transformers | ✅ | ✅ | ✅ |

---

## Troubleshooting

### Issue: Tool calls not detected

**Solution:**
```python
# Enable verbose logging
llm = create_llm("ollama", model="qwen3", verbose=True)

# Check tool call format in response
response = llm.generate("...", tools=[...])
print(response.raw_response)  # Inspect raw format
```

### Issue: Double-tagging (tags appear twice)

**Cause**: Format already matches target, rewriter applies tags again.

**Solution:**
```python
# Use passthrough format for OpenAI models
from abstractllm.tools.syntax_rewriter import SyntaxFormat

rewriter = ToolCallSyntaxRewriter(SyntaxFormat.PASSTHROUGH)
```

### Issue: Custom tags not working with CLI

**Solution:**
```python
# Disable auto-formatting for exact tag control
from abstractllm.tools.tag_rewriter import ToolCallTags

tags = ToolCallTags(
    start_tag="ojlk",
    end_tag="dfsd",
    auto_format=False  # Use exact tags
)
```

### Issue: Server returns wrong format

**Solution:**
```bash
# Explicitly set agent_format parameter
curl ... -d '{
  "agent_format": "codex",
  ...
}'

# Or use custom header
curl -H "X-Agent-Type: codex" ...
```

---

## Related Documentation

**Core Library:**
- **[Python API Reference](api-reference.md)** - Complete Python API
- **[Getting Started](getting-started.md)** - Basic usage examples
- **[Providers](providers.md)** - Provider-specific details

**Server:**
- **[Server Guide](server.md)** - Server setup and configuration
- **[Server API Reference](server.md)** - REST API endpoints
- **[Architecture](architecture.md)** - System architecture

---

**Remember**: Both systems work transparently in most cases. Customize only when integrating with specific agentic CLIs or when you need explicit format control.

