# AbstractCore

**A unified Python interface to all LLM providers with production-grade reliability**

AbstractCore (preview of AbstractLLM v2) provides a clean, consistent API for OpenAI, Anthropic, Ollama, MLX, and more - with built-in retry logic, streaming, tool calling, and structured output.

```python
from abstractllm import create_llm

# Works with any provider - same simple interface
llm = create_llm("openai", model="gpt-4o-mini")  # or "anthropic", "ollama"...
response = llm.generate("What is the capital of France?")
print(response.content)  # "The capital of France is Paris."
```

## What Is AbstractCore?

AbstractCore is **focused infrastructure** for LLM applications. It handles the messy details of different APIs so you can focus on building features.

**Core Philosophy**: Provide a unified interface with production-grade reliability, not a full-featured framework.

### ✅ What AbstractCore Does Well

- **🔌 Universal Provider Support**: Same API for OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace
- **🛠️ Tool Calling**: Native support across all providers with automatic execution
- **📊 Structured Output**: Type-safe JSON responses with Pydantic validation
- **⚡ Streaming**: Real-time responses with proper tool handling
- **🔄 Retry & Circuit Breakers**: Production-grade error handling and recovery
- **🔔 Event System**: Comprehensive observability and monitoring hooks
- **🔢 Vector Embeddings**: SOTA open-source embeddings for RAG applications
- **💬 Simple Sessions**: Conversation memory without complexity

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
# Core package
pip install abstractcore

# With providers you need
pip install abstractcore[openai,anthropic]  # API providers
pip install abstractcore[ollama,mlx]        # Local providers
pip install abstractcore[embeddings]        # Vector embeddings
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
| **Focus** | Clean LLM interface | API compatibility | Full framework | Agent workflows |
| **Size** | Lightweight (~8k LOC) | Lightweight | Heavy (100k+ LOC) | Medium |
| **Tool Calling** | ✅ Universal execution | ⚠️ Pass-through only | ✅ Via integrations | ✅ Native |
| **Streaming** | ✅ With tool support | ✅ Basic | ✅ Basic | ❌ Limited |
| **Structured Output** | ✅ With retry logic | ❌ None | ⚠️ Via parsers | ⚠️ Basic |
| **Production Ready** | ✅ Retry + circuit breakers | ⚠️ Basic | ✅ Via LangSmith | ✅ Via LangSmith |

**Choose AbstractCore if**: You want clean LLM infrastructure without framework lock-in.
**Choose LangChain if**: You need pre-built RAG/agent components and don't mind complexity.
**Choose LiteLLM if**: You only need API compatibility without advanced features.

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

## Advanced Capabilities

AbstractCore is designed as infrastructure. For advanced AI applications, combine with:

### [AbstractMemory](https://github.com/lpalbou/AbstractMemory)
Temporal knowledge graphs and advanced memory systems for AI agents.

### [AbstractAgent](https://github.com/lpalbou/AbstractAgent)
Autonomous agents with planning, tool execution, and self-improvement capabilities.

## Documentation

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