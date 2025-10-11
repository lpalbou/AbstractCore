# AbstractLLM Core: Basic Introduction

## What You'll Learn

- 🌟 Initialize AbstractLLM Core
- 📝 Configure basic text generation
- 🔍 Understand core library concepts

### Learning Objectives

1. Import AbstractLLM library
2. Select and configure a basic model
3. Generate simple text responses
4. Explore basic configuration options

### Prerequisites

- Python 3.9+
- Basic Python programming knowledge
- Ollama or other supported provider installed

### Example Walkthrough

This example demonstrates the simplest way to use AbstractLLM Core. You'll learn how to:
- Set up a basic text generation environment
- Make your first API call
- Handle basic configurations

### Installation

```bash
# Ensure you have the required dependencies
pip install abstractllm ollama

# Verify Ollama is running
ollama serve
```

### Code Example

```python
from abstractllm import AbstractLLM

# Initialize a basic model
llm = AbstractLLM(provider='ollama', model='llama3')

# Generate a simple response
response = llm.generate("Tell me a short joke.")
print(response)
```

### Expected Output

```
A typical AI-generated humorous response about a programmer walking into a bar.
```

### Configuration Options

```python
# Advanced configuration example
llm = AbstractLLM(
    provider='ollama',
    model='llama3',
    max_tokens=100,      # Limit response length
    temperature=0.7,     # Control randomness
    stream=True          # Enable streaming
)
```

### Troubleshooting

#### Common Issues
- **Model Not Found**: Ensure the model is installed via `ollama pull llama3`
- **Connection Error**: Verify Ollama is running (`ollama serve`)
- **Dependency Problems**: Use `pip install -U abstractllm ollama`

#### Debugging Tips
- Check Ollama logs: `ollama logs`
- Validate model availability: `ollama list`
- Use verbose mode for detailed errors

### Best Practices

- Always specify a max token limit
- Use temperature to control response creativity
- Handle potential network or model errors

### Performance Considerations

- First call might be slower due to model loading
- Subsequent calls will be faster
- Consider model size and your hardware capabilities

### Reference Documentation

- [AbstractLLM Core Documentation](/docs/getting-started.md)
- [Provider Configuration](/docs/providers.md)
- [Troubleshooting Guide](/docs/troubleshooting.md)

### Next Steps

After completing this example, move to `example_02_providers` to explore more advanced provider configurations.

### Contribute & Feedback

- Found an issue? [Open an Issue](https://github.com/your-org/abstractllm_core/issues)
- Want to improve the example? Submit a pull request!

---

**Learning Time**: ~15 minutes
**Complexity**: Beginner
**Last Updated**: 2025-10-11