# Unified Streaming Architecture

## Overview

AbstractLLM Core's Unified Streaming Architecture provides a single streaming implementation that works across different models and tool call formats, simplifying streaming interactions.

> **📊 Visual Documentation**: For detailed architectural diagrams, data flow visualizations, and performance charts, see the comprehensive [Streaming Architecture Visual Guide](streaming-architecture-visual-guide.md).

## Key Design Principles

1. **Single Streaming Path**: Replace complex dual-mode system with a unified approach
2. **Real-time Tool Execution**: Execute tools as they're detected during streaming
3. **Format Agnostic**: Support multiple tool call formats seamlessly
4. **Performance Optimized**: Minimize latency and memory overhead

## Performance Characteristics

| Metric | Result | Description |
|--------|--------|-------------|
| First Chunk Latency | <10ms | Initial response time |
| Tool Detection Overhead | <1ms per chunk | Processing overhead |
| Memory Efficiency | Linear, bounded | Predictable memory allocation |
| Large Stream Handling | 1000+ chunks | High volume streaming capability |

## Streaming Implementation

### Core Components

1. **IncrementalToolDetector**
   - Detects tool calls across different formats
   - Preserves tool calls for tag rewriting
   - Handles partial and complete tool calls

2. **UnifiedStreamProcessor**
   - Single processing pipeline for all streaming scenarios
   - Supports tag rewriting and tool detection
   - Compatible with multiple model formats

### Streaming Workflow

```python
def process_stream(self, response_stream):
    for chunk in response_stream:
        # Detect tool calls, preserving content
        streamable_content, completed_tools = self.detector.process_chunk(chunk.content)

        # Apply tag rewriting if needed
        if streamable_content and self.tag_rewriter:
            streamable_content = self._apply_tag_rewriting_direct(streamable_content)

        # Yield processed chunk
        yield GenerateResponse(content=streamable_content, ...)

        # Tools can be processed separately by the client
        if completed_tools:
            for tool_call in completed_tools:
                tool_call.execute()
```

## Supported Tool Call Formats

| Model Family | Format | Example |
|-------------|--------|---------|
| Qwen3 | `<|tool_call|>...JSON...</|tool_call|>` | Full support |
| LLaMA | `<function_call>...JSON...</function_call>` | Full support |
| Gemma | `` `tool_code...JSON...` `` | Full support |
| Generic XML | `<tool_call>...JSON...</tool_call>` | Full support |
| Custom Tags | Exact matching | `ojlk...JSON...dfsd` |

## Advanced Features

### Custom Tag Handling

```python
# Exact tag matching
llm = create_llm("ollama",
    model="qwen3-coder:30b",
    tool_call_tags="ojlk,dfsd"  # Exact custom tags
)
```

### Real-Time Tool Execution

- Tools execute immediately when complete
- No waiting for entire stream to finish
- Supports multiple sequential tool calls
- Works across different model formats

## Edge Case Handling

- Empty content chunks
- Malformed JSON auto-repair
- Incomplete tool calls
- Multiple sequential tools
- Network interruptions

## Migration from Dual-Mode System

### Before (Problematic)
- Separate streaming paths
- Inconsistent tool handling
- Complex configuration
- High latency

### After (Unified Architecture)
- Single streaming implementation
- Consistent tool execution
- Zero configuration
- <10ms first chunk latency

## Best Practices

✅ Trust the default configuration
✅ Use with any supported model
✅ No manual tool parsing required
✅ Monitor tool execution via events

## Compatibility Matrix

| Provider | Models | Streaming Support | Tool Call Conversion |
|----------|--------|-------------------|---------------------|
| OpenAI | GPT-4o, GPT-3.5 | Full | Automatic |
| Anthropic | Claude 3.5 Haiku | Full | Automatic |
| Ollama | Qwen, LLaMA, Gemma | Full | Automatic |
| MLX | Community Models | Full | Automatic |

## Troubleshooting

1. Verify tool definitions match expected schema
2. Check model-specific tool call syntax
3. Use `verbose=True` for detailed logs

> **📚 Related Docs**:
> - [Tool Call Conversion](tool-syntax-rewriting.md)
> - [Tool System Integration](tool-system-architecture.md)
> - [Production Examples](examples.md#streaming-patterns)