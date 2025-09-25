# AbstractCore

**Universal LLM Infrastructure: One API for All Models** 🚀

AbstractCore provides a unified interface to all LLM providers (OpenAI, Anthropic, Ollama, MLX, and more) with production-grade reliability AND a universal API server that makes any model OpenAI-compatible.

## 🎯 New: AbstractCore Server - Universal LLM Gateway

```bash
# Start the universal server
pip install abstractcore[server]
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Use ANY provider through simple URLs or OpenAI clients!
```

### Super Simple Usage
```bash
# Just URL parameters - no JSON needed!
curl "http://localhost:8000/chat?message=Hello&provider=anthropic"

# Or use any OpenAI client
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Use Claude with OpenAI's client! 🤯
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Hello Claude!"}]
)
```

**[Server documentation →](docs/server.md)**

## Python Library Usage

```python
from abstractllm import create_llm

# Direct Python interface - same code for any provider
llm = create_llm("openai", model="gpt-4o-mini")  # or "anthropic", "ollama"...
response = llm.generate("What is the capital of France?")
print(response.content)  # "The capital of France is Paris."
```

## What Is AbstractCore?

AbstractCore is **focused infrastructure** for LLM applications. It handles the messy details of different APIs so you can focus on building features.

**Core Philosophy**: Provide a unified interface with production-grade reliability, not a full-featured framework.

### ✅ What AbstractCore Does Well

- **🌐 Universal API Server**: OpenAI-compatible endpoints for ALL providers
- **🔌 Universal Provider Support**: Same API for OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace
- **🛠️ Tool Calling**: Native support across all providers with automatic execution
- **🔍 Web Search**: Real-time DuckDuckGo search with time filtering and regional results
- **📊 Structured Output**: Type-safe JSON responses with Pydantic validation
- **⚡ Streaming**: Real-time responses with proper tool handling
- **🔄 Retry & Circuit Breakers**: Production-grade error handling and recovery
- **🔔 Event System**: Comprehensive observability and monitoring hooks
- **🔢 Vector Embeddings**: SOTA open-source embeddings for RAG applications
- **💬 Simple Sessions**: Conversation memory without complexity
- **⌨️ Basic CLI**: Interactive command-line tool for testing and demonstration

### ❌ What AbstractCore Doesn't Do

AbstractCore is **infrastructure, not application logic**. For more advanced capabilities:

- **Complex Workflows**: Use [AbstractAgent](https://github.com/lpalbou/AbstractAgent) for autonomous agents
- **Advanced Memory**: Use [AbstractMemory](https://github.com/lpalbou/AbstractMemory) for temporal knowledge graphs
- **Multi-Agent Systems**: Use specialized orchestration frameworks
- **RAG Pipelines**: Built-in embeddings, but you build the pipeline
- **Prompt Templates**: Bring your own templating system

## Quick Start

### Installation

```bash
# Quick start with server and common providers
pip install abstractcore[server,openai,anthropic]

# Or install everything
pip install abstractcore[all]

# Minimal installation options
pip install abstractcore                     # Core only
pip install abstractcore[openai,anthropic]  # API providers
pip install abstractcore[ollama,mlx]        # Local providers
pip install abstractcore[embeddings]        # Vector embeddings
pip install abstractcore[server]            # API server
```

### 30-Second Example

```python
from abstractllm import create_llm

# Pick any provider - same code works everywhere
llm = create_llm("openai", model="gpt-4o-mini")  # or anthropic, ollama...

# Generate text
response = llm.generate("Explain Python decorators")
print(response.content)

# Structured output with automatic validation
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = llm.generate(
    "Extract: John Doe is 25 years old",
    response_model=Person
)
print(f"{person.name} is {person.age}")  # John Doe is 25
```

### Basic CLI Tool

AbstractCore includes a simple CLI tool for quick testing and demonstration:

```bash
# Interactive chat with any provider
python -m abstractllm.utils.cli --provider ollama --model qwen3-coder:30b

# Single prompt execution
python -m abstractllm.utils.cli --provider openai --model gpt-4o-mini --prompt "What is Python?"

# With streaming
python -m abstractllm.utils.cli --provider anthropic --model claude-3-5-haiku-20241022 --stream

# Commands: /help /quit /clear /stream /debug /history /model <spec>
# Built-in tools: list_files, read_file, execute_command, web_search
```

**Capabilities & Limitations:**
- ✅ Basic chat interactions with conversation history
- ✅ Built-in tools for file operations and web search
- ✅ All AbstractCore providers supported
- ✅ Streaming mode and debug output
- ❌ No ReAct patterns or complex reasoning chains
- ❌ No adaptive actions or advanced agent behaviors
- ❌ Limited to basic demonstration purposes

**Note**: This CLI is a basic demonstrator. For production applications requiring advanced reasoning, multi-step tool chains, or complex agent behaviors, build custom solutions using the AbstractCore framework directly.

## Provider Support

| Provider | Status | Best For | Setup |
|----------|--------|----------|-------|
| **OpenAI** | ✅ Full | Production APIs, latest models | `OPENAI_API_KEY` |
| **Anthropic** | ✅ Full | Claude models, long context | `ANTHROPIC_API_KEY` |
| **Ollama** | ✅ Full | Local/private, no costs | Install Ollama |
| **MLX** | ✅ Full | Apple Silicon optimization | Built-in |
| **LMStudio** | ✅ Full | Local with GUI | Start LMStudio |
| **HuggingFace** | ✅ Full | Open source models | Built-in |

## Framework Comparison

| Feature | AbstractCore | LiteLLM | LangChain | LangGraph |
|---------|-------------|----------|-----------|-----------|
| **Focus** | Clean LLM interface + API server | API compatibility | Full framework | Agent workflows |
| **API Server** | ✅ Built-in OpenAI-compatible | ✅ Proxy server | ❌ None | ❌ None |
| **Size** | Lightweight (~10k LOC) | Lightweight | Heavy (100k+ LOC) | Medium |
| **Tool Calling** | ✅ Universal execution | ⚠️ Pass-through only | ✅ Via integrations | ✅ Native |
| **Streaming** | ✅ With tool support | ✅ Basic | ✅ Basic | ❌ Limited |
| **Structured Output** | ✅ With retry logic | ❌ None | ⚠️ Via parsers | ⚠️ Basic |
| **Production Ready** | ✅ Retry + circuit breakers | ⚠️ Basic | ✅ Via LangSmith | ✅ Via LangSmith |

**Choose AbstractCore if**: You want clean LLM infrastructure with a universal API server.
**Choose LangChain if**: You need pre-built RAG/agent components and don't mind complexity.
**Choose LiteLLM if**: You only need basic API compatibility without advanced features.

## Core Features

### Tool Calling (Universal)

Tools work the same across all providers. Use the `@tool` decorator to teach the LLM when and how to use your functions:

```python
from abstractllm.tools import tool

@tool(
    description="Get current weather information for any city worldwide",
    tags=["weather", "location", "temperature"],
    when_to_use="When user asks about weather, temperature, or climate conditions",
    examples=[
        {
            "description": "Get weather for major city",
            "arguments": {"city": "London"}
        },
        {
            "description": "Check weather with country",
            "arguments": {"city": "Tokyo, Japan"}
        }
    ]
)
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In production, call a real weather API
    return f"The weather in {city} is sunny, 72°F"

# The decorator creates rich tool metadata that helps LLMs understand:
# - What the tool does (description)
# - When to use it (when_to_use)
# - How to use it (examples with proper arguments)
response = llm.generate("What's the weather in Paris?", tools=[get_weather])
# LLM automatically calls function with proper arguments

# See docs/examples.md for calculator, file operations, and advanced tool examples
```

### Structured Output with Retry

Automatic validation and retry when models return invalid JSON:

```python
from pydantic import BaseModel, field_validator

class Product(BaseModel):
    name: str
    price: float

    @field_validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

# Automatically retries with error feedback if validation fails
product = llm.generate(
    "Extract: Gaming laptop for $1200",
    response_model=Product
)
```

### Real-Time Streaming

```python
print("AI: ", end="")
for chunk in llm.generate("Write a haiku about coding", stream=True):
    print(chunk.content, end="", flush=True)
print()  # Code flows like rain / Logic blooms in endless loops / Beauty in the bugs
```

### Built-in Web Search

AbstractCore includes a powerful web search tool with DuckDuckGo integration:

```python
from abstractllm.tools.common_tools import web_search

# Recent news (past 24 hours)
results = web_search("AI developments news", time_range="h")

# Research from past week
results = web_search("Python tutorials", time_range="w", region="us-en")

# Academic papers from past month
results = web_search("machine learning research", time_range="m", safe_search="strict")
```

**Time Range Options:**
- `"h"` - Past 24 hours
- `"d"` - Past day
- `"w"` - Past week
- `"m"` - Past month
- `"y"` - Past year

**Features:**
- Real web results with titles, URLs, and descriptions
- Time range filtering for current content
- Regional results (us-en, uk-en, etc.)
- Safe search controls
- No API key required

### Timeout Configuration

Configure timeouts for HTTP requests, tools, and circuit breaker recovery:

```python
# Basic timeout configuration
llm = create_llm(
    "openai",
    model="gpt-4o-mini",
    timeout=180,            # HTTP request timeout (default: 180s)
    tool_timeout=300,       # Tool execution timeout (default: 300s)
)

# Runtime timeout adjustments
llm.set_timeout(120)              # Change HTTP timeout
llm.set_tool_timeout(600)         # Change tool timeout  
llm.set_recovery_timeout(30)      # Change circuit breaker recovery (default: 60s)

# Get current timeouts
print(f"HTTP timeout: {llm.get_timeout()}s")
print(f"Tool timeout: {llm.get_tool_timeout()}s")
print(f"Recovery timeout: {llm.get_recovery_timeout()}s")

# Session-level timeout configuration
from abstractllm.core.session import BasicSession

session = BasicSession(
    provider=llm,
    timeout=60,              # Override provider HTTP timeout
    tool_timeout=120,        # Override provider tool timeout
    recovery_timeout=30      # Override provider recovery timeout
)

# Session timeout management
session.set_timeout(90)
print(f"Session timeout: {session.get_timeout()}s")
```

### Production Reliability

Built-in retry logic and circuit breakers handle provider issues automatically:

```python
from abstractllm.core.retry import RetryConfig

# Production-grade retry configuration
llm = create_llm(
    "openai",
    model="gpt-4o-mini",
    retry_config=RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        use_jitter=True
    )
)
```

### Event System & Monitoring

Hook into every operation for monitoring and control:

```python
from abstractllm.events import EventType, on_global

def monitor_costs(event):
    if event.cost_usd and event.cost_usd > 0.10:
        print(f"💰 High cost alert: ${event.cost_usd:.4f}")

on_global(EventType.AFTER_GENERATE, monitor_costs)
```

### Vector Embeddings

Built-in SOTA embeddings for semantic search:

```python
from abstractllm.embeddings import EmbeddingManager

embedder = EmbeddingManager()  # Uses Google's EmbeddingGemma by default
similarity = embedder.compute_similarity(
    "machine learning",
    "artificial intelligence"
)
print(f"Similarity: {similarity:.3f}")  # 0.847
```

## AbstractCore Server

Turn AbstractCore into a universal API gateway that makes ANY LLM provider OpenAI-compatible:

### Key Server Features

- **🔄 Universal Compatibility**: Use OpenAI clients with Claude, Llama, or any model
- **📊 Dynamic Discovery**: Auto-detects available models without hardcoding
- **🛠️ Tool Management**: Register and execute tools via API
- **💾 Session Management**: Maintain conversation context across requests
- **📡 Real-time Events**: Stream events via SSE for monitoring
- **🔌 Drop-in Replacement**: Works with any OpenAI-compatible application

### Server Quick Start

```bash
# Start the server
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# With reload for development
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000 --reload
```

**Two ways to use:**

1. **Super Simple** (URL parameters)
```bash
curl "http://localhost:8000/chat?message=Hello&provider=anthropic"
```

2. **OpenAI Compatible** (any programming language)
```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'not-needed'
});

const response = await client.chat.completions.create({
  model: 'ollama/qwen3-coder:30b',
  messages: [{ role: 'user', content: 'Hello!' }]
});
```

**[Server documentation →](docs/server.md)**

## Advanced Capabilities

AbstractCore is designed as infrastructure. For advanced AI applications, combine with:

### [AbstractMemory](https://github.com/lpalbou/AbstractMemory)
Temporal knowledge graphs and advanced memory systems for AI agents.

### [AbstractAgent](https://github.com/lpalbou/AbstractAgent)
Autonomous agents with planning, tool execution, and self-improvement capabilities.

## Documentation

- **[🌐 Server Guide](docs/server.md)** - Universal API server documentation
- **[Getting Started](docs/getting-started.md)** - Your first AbstractCore program
- **[Capabilities](docs/capabilities.md)** - What AbstractCore can and cannot do
- **[Providers](docs/providers.md)** - Complete provider guide
- **[Examples](docs/examples.md)** - Real-world use cases
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Architecture](docs/architecture.md)** - How it works internally
- **[Framework Comparison](docs/comparison.md)** - Detailed comparison with alternatives

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for a complete list of dependencies and contributors.

---

**AbstractCore** - Clean infrastructure for LLM applications 🚀