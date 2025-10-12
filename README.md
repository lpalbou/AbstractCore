# AbstractCore

A unified, powerful Python library for seamless interaction with multiple Large Language Model (LLM) providers.

**Write once, run everywhere.**

## Quick Start

### Installation

```bash
pip install abstractcore[all]
```

### Basic Usage

```python
from abstractllm import create_llm

# Works with any provider - just change the provider name
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
response = llm.generate("What is the capital of France?")
print(response.content)
```

### Tool Calling

```python
from abstractllm import create_llm, tool

@tool
def get_current_weather(city: str):
    """Fetch current weather for a given city."""
    return f"Weather in {city}: 72°F, Sunny"

llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "What's the weather like in San Francisco?",
    tools=[get_current_weather]
)
print(response.content)
```

## Key Features

- **Provider Agnostic**: Seamlessly switch between OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace
- **Unified Tools**: Consistent tool calling across all providers
- **Structured Responses**: Clean, predictable output formats with Pydantic
- **Streaming Support**: Real-time token generation for interactive experiences
- **Embeddings**: Built-in support for semantic search and RAG applications
- **Universal Server**: Optional OpenAI-compatible API server

## Supported Providers

| Provider | Status | Setup |
|----------|--------|-------|
| **OpenAI** | ✅ Full | [Get API key](docs/prerequisites.md#openai-setup) |
| **Anthropic** | ✅ Full | [Get API key](docs/prerequisites.md#anthropic-setup) |
| **Ollama** | ✅ Full | [Install guide](docs/prerequisites.md#ollama-setup) |
| **LMStudio** | ✅ Full | [Install guide](docs/prerequisites.md#lmstudio-setup) |
| **MLX** | ✅ Full | [Setup guide](docs/prerequisites.md#mlx-setup) |
| **HuggingFace** | ✅ Full | [Setup guide](docs/prerequisites.md#huggingface-setup) |

## Server Mode (Optional HTTP REST API)

AbstractCore is **primarily a Python library**. The server is an **optional component** that provides OpenAI-compatible HTTP endpoints:

```bash
# Install with server support
pip install abstractcore[server]

# Start the server
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000
```

Use with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Server Features:**
- OpenAI-compatible REST endpoints (`/v1/chat/completions`, `/v1/embeddings`, etc.)
- Multi-provider support through one HTTP API
- Agentic CLI integration (Codex, Crush, Gemini CLI)
- Streaming responses
- Tool call format conversion
- Interactive API docs at `/docs`

**When to use the server:**
- Integrating with existing OpenAI-compatible tools
- Using agentic CLIs (Codex, Crush, Gemini CLI)
- Building web applications that need HTTP API
- Multi-language access (not just Python)

## Internal CLI (Optional Interactive Testing Tool)

AbstractCore includes a **built-in CLI** for interactive testing, development, and conversation management. This is an internal testing tool, distinct from external agentic CLIs.

```bash
# Start interactive CLI
python -m abstractllm.utils.cli --provider ollama --model qwen3-coder:30b

# With streaming enabled
python -m abstractllm.utils.cli --provider openai --model gpt-4o-mini --stream

# Single prompt execution
python -m abstractllm.utils.cli --provider anthropic --model claude-3-5-haiku-latest --prompt "What is Python?"
```

**Key Features:**
- Interactive REPL with conversation history
- Chat history compaction and management
- Fact extraction from conversations
- Conversation quality evaluation (LLM-as-a-judge)
- Tool call testing and debugging
- System prompt management
- Multiple provider support

**Popular Commands:**
- `/compact` - Compress chat history while preserving context
- `/facts [file]` - Extract structured facts from conversation
- `/judge` - Evaluate conversation quality with feedback
- `/history [n]` - View conversation history
- `/stream` - Toggle real-time streaming
- `/system [prompt]` - Show or change system prompt
- `/status` - Show current provider, model, and capabilities

**Full Documentation:** [Internal CLI Guide](docs/internal-cli.md)

**When to use the CLI:**
- Interactive development and testing
- Debugging tool calls and provider behavior
- Conversation management experiments
- Quick prototyping with different models
- Learning AbstractCore capabilities

## Documentation

**📚 Complete Documentation:** [docs/](docs/) - Full documentation index and navigation guide

### Getting Started
- **[Prerequisites & Setup](docs/prerequisites.md)** - Install and configure providers (OpenAI, Anthropic, Ollama, etc.)
- **[Getting Started Guide](docs/getting-started.md)** - 5-minute quick start with core concepts
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

### Core Library (Python)
- **[Python API Reference](docs/api-reference.md)** - Complete Python API documentation
- **[Embeddings Guide](docs/embeddings.md)** - Semantic search, RAG, and vector embeddings
- **[Code Examples](examples/)** - Working examples for all features
- **[Capabilities](docs/capabilities.md)** - What AbstractCore can and cannot do

### Server (Optional HTTP REST API)
- **[Server Documentation](docs/server.md)** - Complete server setup, API reference, and deployment

### Architecture & Advanced
- **[Architecture](docs/architecture.md)** - System design and architecture overview
- **[Tool Syntax Rewriting](docs/tool-syntax-rewriting.md)** - Format conversion for agentic CLIs

## Use Cases

### 1. Provider Flexibility

```python
# Same code works with any provider
providers = ["openai", "anthropic", "ollama"]

for provider in providers:
    llm = create_llm(provider, model="gpt-4o-mini")  # Auto-selects appropriate model
    response = llm.generate("Hello!")
```

### 2. Local Development, Cloud Production

```python
# Development (free, local)
llm_dev = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")

# Production (high quality, cloud)
llm_prod = create_llm("openai", model="gpt-4o-mini")
```

### 3. Embeddings & RAG

```python
from abstractllm.embeddings import EmbeddingManager

# Create embeddings for semantic search
embedder = EmbeddingManager()
docs_embeddings = embedder.embed_batch([
    "Python is great for data science",
    "JavaScript powers the web",
    "Rust ensures memory safety"
])

# Find most similar document
query_embedding = embedder.embed("Tell me about web development")
similarity = embedder.compute_similarity(query, docs[0])
```

### 4. Structured Output

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: int  # 1-5
    summary: str

llm = create_llm("openai", model="gpt-4o-mini")
review = llm.generate(
    "Review the movie Inception",
    response_model=MovieReview
)
print(f"{review.title}: {review.rating}/5")
```

### 5. Universal API Server

```bash
# Start server once
uvicorn abstractllm.server.app:app --port 8000

# Use with any OpenAI client
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

## Why AbstractCore?

✅ **Unified Interface**: One API for all LLM providers  
✅ **Production Ready**: Robust error handling, retries, timeouts  
✅ **Type Safe**: Full Pydantic integration for structured outputs  
✅ **Local & Cloud**: Run models locally or use cloud APIs  
✅ **Tool Calling**: Consistent function calling across providers  
✅ **Streaming**: Real-time responses for better UX  
✅ **Embeddings**: Built-in vector embeddings for RAG  
✅ **Server Mode**: Optional OpenAI-compatible API server  
✅ **Well Documented**: Comprehensive guides and examples  

## Installation Options

```bash
# Minimal core
pip install abstractcore

# With specific providers
pip install abstractcore[openai]
pip install abstractcore[anthropic]
pip install abstractcore[ollama]

# With server support
pip install abstractcore[server]

# With embeddings
pip install abstractcore[embeddings]

# Everything
pip install abstractcore[all]
```

## Testing Status

All tests passing as of October 12th, 2025.

**Test Environment:**
- Hardware: MacBook Pro (14-inch, Nov 2024)
- Chip: Apple M4 Max
- Memory: 128 GB
- Python: 3.12.2

## Quick Links

- **[📚 Documentation Index](docs/)** - Complete documentation navigation guide
- **[🚀 Getting Started](docs/getting-started.md)** - 5-minute quick start
- **[⚙️ Prerequisites](docs/prerequisites.md)** - Provider setup (OpenAI, Anthropic, Ollama, etc.)
- **[📖 Python API](docs/api-reference.md)** - Complete Python API reference
- **[🌐 Server Guide](docs/server.md)** - HTTP API server setup
- **[🔧 Troubleshooting](docs/troubleshooting.md)** - Fix common issues
- **[💻 Examples](examples/)** - Working code examples
- **[🐛 Issues](https://github.com/lpalbou/AbstractCore/issues)** - Report bugs
- **[💬 Discussions](https://github.com/lpalbou/AbstractCore/discussions)** - Get help

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**AbstractCore** - One interface, all LLM providers. Focus on building, not managing API differences. 🚀
