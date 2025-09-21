# AbstractLLM

A unified interface to all LLM providers with essential infrastructure for building LLM-powered applications.

## Features

- 🔌 **Universal Interface**: Single API for OpenAI, Anthropic, Ollama, MLX, and more
- 🛠️ **Tool Calling**: Standardized tool/function calling across providers
- 💬 **Session Management**: Built-in conversation context and history
- 🏠 **Local Models**: Support for Ollama, MLX, and LM Studio
- 🔄 **Streaming**: Real-time response streaming
- 🎯 **Type Safety**: Full typing support throughout
- 🧪 **Testing**: Comprehensive test suite with real implementations

## Installation

### Basic Installation

```bash
pip install abstractllm
```

### With Specific Providers

```bash
# For OpenAI
pip install abstractllm[openai]

# For Anthropic
pip install abstractllm[anthropic]

# For all providers
pip install abstractllm[all]
```

## Quick Start

### Simple Generation

```python
from abstractllm import create_llm

# Create a provider
llm = create_llm("openai", model="gpt-3.5-turbo")

# Generate a response
response = llm.generate("Explain quantum computing in simple terms")
print(response.content)
```

### With Session Management

```python
from abstractllm import create_llm, BasicSession

# Create a session with conversation history
llm = create_llm("anthropic", model="claude-3-haiku-20240307")
session = BasicSession(
    provider=llm,
    system_prompt="You are a helpful physics tutor"
)

# Have a conversation
session.generate("What is quantum entanglement?")
session.generate("Can you give me an example?")  # Has context of previous question

# Save conversation
session.save("physics_lesson.json")
```

### Tool Calling

```python
# Define tools
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    }
}]

# Generate with tools
response = llm.generate(
    "What's the weather in Paris?",
    tools=tools
)

# Check for tool calls
if response.has_tool_calls():
    for call in response.tool_calls:
        print(f"Tool: {call['name']}")
        print(f"Arguments: {call['arguments']}")
```

### Local Models with Ollama

```python
# Use Ollama for local execution
llm = create_llm("ollama", model="qwen3-coder:30b")
response = llm.generate("Write a Python function to sort a list")
```

### Apple Silicon with MLX

```python
# Optimized for Apple Silicon
llm = create_llm("mlx", model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
response = llm.generate("Explain how transformers work")
```

## Supported Providers

| Provider | Models | Tool Support | Streaming | Local |
|----------|--------|--------------|-----------|-------|
| OpenAI | GPT-3.5, GPT-4, GPT-4V | ✅ | ✅ | ❌ |
| Anthropic | Claude 2, Claude 3 | ✅ | ✅ | ❌ |
| Ollama | Llama, Mistral, Qwen, etc | ⚠️ | ✅ | ✅ |
| MLX | Quantized models | ❌ | ⚠️ | ✅ |
| LM Studio | Any GGUF model | ⚠️ | ✅ | ✅ |

## Advanced Usage

### Streaming Responses

```python
for chunk in llm.generate("Tell me a long story", stream=True):
    print(chunk.content, end="", flush=True)
```

### Custom Parameters

```python
response = llm.generate(
    "Be creative",
    temperature=1.5,      # Higher = more creative
    max_tokens=500,       # Limit response length
    top_p=0.9,           # Nucleus sampling
)
```

### Error Handling

```python
from abstractllm.exceptions import ProviderAPIError, RateLimitError

try:
    response = llm.generate("Hello")
except RateLimitError:
    print("Rate limited, waiting...")
    time.sleep(60)
except ProviderAPIError as e:
    print(f"API error: {e}")
```

## Testing

The project includes comprehensive tests with real provider implementations:

```bash
# Run all tests
pytest tests/

# Test specific provider
pytest tests/test_providers.py -k "openai"

# Test tool calling
python tests/test_tool_calling.py
```

### Testing Your Code

Use the mock provider for testing without API calls:

```python
# No API calls, immediate responses
llm = create_llm("mock")
response = llm.generate("Test prompt")
assert "Mock response" in response.content
```

## Architecture

AbstractLLM follows a clean architecture with clear separation of concerns:

```
abstractllm/
├── core/           # Core interfaces and types
│   ├── interface.py    # AbstractLLMInterface
│   ├── session.py      # BasicSession
│   └── types.py        # Data classes
├── providers/      # Provider implementations
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── ...
└── tools/          # Tool system
    └── core.py         # Tool definitions
```

See [Architecture Documentation](docs/architecture.md) for details.

## Documentation

- [Architecture Guide](docs/architecture.md) - System design and components
- [Provider Guide](docs/providers.md) - Detailed provider documentation
- [API Reference](docs/api_reference.md) - Complete API documentation

## Roadmap

### Current (v2.0)
- ✅ Core provider abstraction
- ✅ 6 provider implementations
- ✅ Tool calling support
- ✅ Session management
- ✅ Comprehensive testing

### Planned
- [ ] Async support
- [ ] Response caching
- [ ] Token usage tracking
- [ ] Provider fallback chains
- [ ] Middleware system

### Future Packages
- **AbstractMemory**: Temporal knowledge graphs
- **AbstractAgent**: Agent orchestration
- **AbstractSwarm**: Multi-agent systems

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) file.

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/abstractllm/abstractllm/issues)
- Discord: [Join our community](https://discord.gg/abstractllm)

## Citation

If you use AbstractLLM in your research, please cite:

```bibtex
@software{abstractllm2024,
  title = {AbstractLLM: Unified Interface to LLM Providers},
  year = {2024},
  url = {https://github.com/abstractllm/abstractllm}
}
```