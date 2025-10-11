# AbstractCore

A unified, powerful Python library for seamless interaction with multiple Large Language Model (LLM) providers.

## Quick Overview

AbstractCore simplifies LLM interactions by providing a consistent, intuitive interface across different providers. Write once, run everywhere.

## Core Library Usage

### Basic Generation

```python
from abstractllm import create_llm

# Easily switch between providers without changing code
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
response = llm.generate("What is the capital of France?")
print(response.content)
```

### Advanced Tool Calling

```python
from abstractllm import create_llm, tool

@tool
def get_current_weather(city: str):
    """Fetch current weather for a given city."""
    return f"Weather in {city}: 72°F, Sunny"

llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "What's the weather like in San Francisco today?",
    tools=[get_current_weather]
)

# Automatically executes tools and integrates results
print(response.content)
```

## Key Features

- **Provider Agnostic**: Seamlessly switch between OpenAI, Anthropic, Ollama, MLX
- **Unified Tools**: Consistent tool calling across all providers
- **Structured Responses**: Clean, predictable output formats
- **Intelligent Retry Mechanisms**: Automatic handling of rate limits and transient errors
- **Streaming Support**: Real-time token generation for interactive experiences

## Provider Support

| Provider | Status | Setup |
|----------|--------|-------|
| OpenAI | Full | [Get API key](docs/prerequisites.md#openai-setup) |
| Anthropic | Full | [Get API key](docs/prerequisites.md#anthropic-setup) |
| Ollama | Full | [Install guide](docs/prerequisites.md#ollama-setup) |
| MLX | Full | [Setup guide](docs/prerequisites.md#mlx-setup-apple-silicon) |

## Server Compatibility Layer

While AbstractCore is primarily a Python library, we provide an optional server mode for enhanced compatibility with OpenAI-style APIs and agentic CLI tools.

### Quick Server Setup

```bash
# Install with server dependencies
pip install abstractcore[server]

# Start the compatibility server
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000
```

#### OpenAI-Compatible Client Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Documentation

- [Prerequisites & Setup](docs/prerequisites.md)
- [Getting Started](docs/getting-started.md)
- [Server Documentation](docs/server.md)
- [Tool Call Tag Rewriting](docs/tool-call-tag-rewriting.md)

## Testing Status

All tests passing as of October 11th, 2025

### Test Environment
- Hardware: MacBook Pro (14-inch, Nov 2024)
- Chip: Apple M4 Max
- Memory: 128 GB
- Python: 3.12.2

## License

MIT License - see [LICENSE](LICENSE) file for details.