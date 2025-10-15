# AbstractCore Core: Tools and Function Calling

## What You'll Learn

- 🛠️ Create custom tools
- 🔗 Implement function calling
- 🤖 Build agent-like interactions

### Learning Objectives

1. Define custom tool functions
2. Enable function calling in model
3. Create complex multi-tool workflows
4. Handle tool execution and error scenarios

### Example Walkthrough

This example demonstrates advanced interaction capabilities:
- Creating tools with input/output specifications
- Dynamically executing tools based on model decision
- Building interactive problem-solving workflows

### Key Code Snippet

```python
from abstractcore import create_llm, tool

# Custom tools
@tool
def search_web(query: str) -> str:
    """Search the internet for information"""
    return f"Search results for: {query}"

@tool
def calculate_math(expression: str) -> float:
    """Perform mathematical calculations"""
    return eval(expression)

# Multi-tool workflow
llm = create_llm(
    provider='ollama',
    model='phi3',
    tools=[search_web, calculate_math]
)

response = llm.generate(
    "What's the square root of 16 multiplied by the population of Paris?"
)
```

### Advanced Concepts

- Tool input/output type hints
- Dynamic tool selection
- Error handling in tool execution

### Next Steps

After mastering tools, explore `example_04_streaming` to learn about real-time interactions and streaming generation.