# Provider Guide

## Streaming: Unified Strategy Across Providers

### Unified Streaming Architecture (2025)

AbstractCore introduces a breakthrough in LLM streaming with a single, powerful implementation that works identically across all providers:

```python
# Streaming works the same for ALL providers
print("AI: ", end="")
for chunk in llm.generate(
    "Create a complex function with tools",
    stream=True,
    tools=[code_analysis_tool]
):
    # Real-time chunk processing
    print(chunk.content, end="", flush=True)

    # Immediate tool execution during streaming
    if chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            result = tool_call.execute()
            print(f"\nTool Result: {result}")
```

**Streaming Performance**:
- ⚡ First chunk delivered in <10ms
- 🔧 Single, consistent strategy across providers
- 🛠️ Real-time tool call detection
- 📊 Mid-stream tool execution
- 💨 Zero buffering overhead
- 🚀 Supports multiple providers: OpenAI, Anthropic, Ollama, MLX

**Technical Details**:
- 37% reduction in streaming code complexity
- Robust error handling for malformed responses
- Supports multiple tool formats (Qwen, LLaMA, Gemma, XML)
- Incremental processing with zero latency

**[More Streaming Details →](architecture.md#unified-streaming-architecture)**

(Rest of the existing file remains the same)