# AbstractLLM Core: Streaming and Real-Time Interactions

## What You'll Learn

- 🌊 Implement streaming text generation
- ⚡ Real-time tool execution
- 🚀 Create interactive experiences

### Learning Objectives

1. Understand character-by-character streaming
2. Handle streaming with tool calls
3. Create responsive CLI applications
4. Optimize generation performance

### Example Walkthrough

This example explores advanced streaming capabilities:
- Character-by-character generation
- Mid-stream tool execution
- Building interactive CLI experiences

### Key Code Snippet

```python
from abstractllm import AbstractLLM, tool

@tool
def get_current_time() -> str:
    """Retrieve the current system time"""
    return datetime.now().isoformat()

llm = AbstractLLM(
    provider='ollama',
    model='llama3',
    streaming=True,
    tools=[get_current_time]
)

# Interactive streaming generation
for chunk in llm.generate_stream("Tell me a story and include the current time"):
    print(chunk, end='', flush=True)
```

### Advanced Streaming Features

- Preserving tool calls during streaming
- Low-latency generation
- Handling partial and complete tool calls

### Performance Insights

- Streaming reduces perceived latency
- Enables real-time, interactive experiences
- Supports complex, dynamic interactions

### Next Steps

Progress to `example_05_server` to learn about creating production-ready API servers with streaming support.