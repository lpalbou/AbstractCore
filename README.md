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

**🚀 Built-in Production Applications**
- **📝 Text Summarization**: Ready-to-use summarizer with 6 styles (executive, analytical, conversational, etc.) and CLI
- **⚖️ LLM-as-a-Judge**: Production-ready objective evaluation with structured assessments, multiple file support, and critical analysis
- **🕸️ Knowledge Graph Extraction**: Extract entities/relationships with JSON-LD and RDF triple outputs, perfect for semantic applications

**🔌 Universal LLM Infrastructure**
- **🌐 Universal API Server**: OpenAI-compatible endpoints for ALL providers
- **🤖 Agentic CLI Compatibility** (In Progress): Initial support for Codex, Gemini CLI, Crush
- **🔌 Universal Provider Support**: Same API for OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace
- **🛠️ Tool Calling**: Native support across all providers with automatic execution
- **📊 Structured Output**: Type-safe JSON responses with Pydantic validation
- **⚡ Streaming**: Real-time responses with proper tool handling

**🔧 Production-Ready Features**
- **🔄 Retry & Circuit Breakers**: Production-grade error handling and recovery
- **🔔 Event System**: Comprehensive observability and monitoring hooks
- **🔢 Vector Embeddings**: SOTA embeddings with similarity matrices, clustering, and performance optimization
- **🔍 Web Search**: Real-time DuckDuckGo search with time filtering and regional results
- **💬 Simple Sessions**: Conversation memory without complexity
- **🗜️ Chat Compaction**: SOTA conversation summarization for unlimited chat length
- **⌨️ Basic CLI**: Interactive command-line tool for testing and demonstration

**🔧 Experimental**
- **🔧 MCP Support** (Planned): Model Context Protocol endpoints (stub implementation)

### ❌ What AbstractCore Doesn't Do

AbstractCore is **infrastructure, not application logic**. For more advanced capabilities:

- **Advanced Memory** (WIP): Use [AbstractMemory](https://github.com/lpalbou/AbstractMemory) for temporal knowledge graphs
- **Complex Workflows** (WIP): Use [AbstractAgent](https://github.com/lpalbou/AbstractAgent) for autonomous agents
- **Multi-Agent Systems** (WIP): Use [AbstractSwarm](https://github.com/lpalbou/AbstractSwarm) to distribute tasks across agents
- **RAG Pipelines**: (WIP): Use [RAGnarok](https://github.com/lpalbou/RAGnarok) to leverage GraphRAG in your applications
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

**Need help with setup?** See **[📋 Prerequisites & Setup Guide](docs/prerequisites.md)** for detailed instructions on:
- Getting OpenAI/Anthropic API keys
- Installing and configuring Ollama with models
- Setting up LMStudio, MLX, and HuggingFace
- Troubleshooting common issues

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

# Built-in Text Processing Applications

# 1. Text Summarization with multiple styles and lengths
from abstractllm.processing import BasicSummarizer
from abstractllm.processing.basic_summarizer import SummaryStyle, SummaryLength

# Use qwen3-coder:30b for better quality (if you have 20GB+ RAM)
# Or use qwen3:4b-instruct-2507-q4_K_M for balanced performance
local_llm = create_llm("ollama", model="qwen3-coder:30b")  # Good for complex tasks
summarizer = BasicSummarizer(local_llm)
result = summarizer.summarize(
    long_text,
    focus="business implications",
    style=SummaryStyle.EXECUTIVE,
    length=SummaryLength.BRIEF
)
print(result.summary)
print(f"Confidence: {result.confidence:.2f}")

# CLI: summarizer report.txt --style=executive --length=brief --output=summary.md
# See docs/basic-summarizer.md for full documentation

# 2. Simple Knowledge Extraction
# For complex JSON-LD knowledge graphs, use cloud providers like OpenAI/Anthropic

simple_extraction_prompt = """Extract key facts as JSON:
Google created TensorFlow. Microsoft uses TensorFlow for Azure AI.

Return: [{"entity": "Google", "action": "created", "object": "TensorFlow"}]"""

facts = local_llm.generate(simple_extraction_prompt)
print(facts.content)  # Valid JSON with extracted facts

# For advanced knowledge graphs:
# CLI: python -m abstractllm.apps.extractor document.txt --provider=openai --model=gpt-4o-mini
# See docs/basic-extractor.md for full documentation

# 3. LLM-as-a-Judge for Objective Evaluation
from abstractllm.processing import BasicJudge

# Use same local model for consistency
judge = BasicJudge(local_llm)

# Evaluate content with structured assessment
assessment = judge.evaluate(
    "This function calculates the sum efficiently with clear documentation.",
    context="code review",
    include_criteria=True  # Include detailed criteria explanations
)

print(f"Overall Score: {assessment['overall_score']}/5")
print(f"Judge Summary: {assessment['judge_summary']}")
print("Recommendations:", assessment['actionable_feedback'])

# Evaluate multiple files sequentially (avoids context overflow)
results = judge.evaluate_files(
    ["src/main.py", "src/utils.py", "tests/test_main.py"],
    context="Python code review"
)
for result in results:
    print(f"File: {result['source_reference']} - Score: {result['overall_score']}/5")

# CLI: judge file1.py file2.py --context="code review" --format=json
# See docs/basic-judge.md for full documentation

# Chat history compaction for unlimited conversation length
session = BasicSession(
    local_llm,
    system_prompt="You are helpful",
    auto_compact=True,           # Enable automatic compaction
    auto_compact_threshold=6000  # Compact when >6000 tokens
)
# Conversation continues indefinitely with automatic compaction
# Or manually: session.force_compact(preserve_recent=6, focus="key decisions")
```

### CLI Tools

AbstractCore includes a simple CLI tool for quick testing and demonstration:

**[📋 Complete Internal CLI Guide →](docs/internal-cli.md)**

```bash
# Interactive chat with any provider
python -m abstractllm.utils.cli --provider ollama --model qwen3-coder:30b

# Single prompt execution
python -m abstractllm.utils.cli --provider openai --model gpt-4o-mini --prompt "What is Python?"

# With streaming
python -m abstractllm.utils.cli --provider anthropic --model claude-3-5-haiku-20241022 --stream

# Commands: /help /quit /clear /stream /debug /history /model <spec> /compact /facts [file] /system [prompt]
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
| **OpenAI** | ✅ Full | Production APIs, latest models | [Get API key →](docs/prerequisites.md#openai-setup) |
| **Anthropic** | ✅ Full | Claude models, long context | [Get API key →](docs/prerequisites.md#anthropic-setup) |
| **Ollama** | ✅ Full | Local/private, no costs | [Install guide →](docs/prerequisites.md#ollama-setup) |
| **MLX** | ✅ Full | Apple Silicon optimization | [Setup guide →](docs/prerequisites.md#mlx-setup-apple-silicon) |
| **LMStudio** | ✅ Full | Local with GUI | [Setup guide →](docs/prerequisites.md#lmstudio-setup) |
| **HuggingFace** | ✅ Full | Open source models | [Setup guide →](docs/prerequisites.md#huggingface-setup) |

**📋 Complete setup instructions for all providers: [Prerequisites & Setup Guide](docs/prerequisites.md)**

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

### Memory Management for Local Models

Free memory explicitly when working with multiple large models (critical for test suites and memory-constrained environments):

```python
# Load and use a large local model
llm = create_llm("ollama", model="qwen3-coder:30b")
response = llm.generate("Hello world")

# Explicitly free memory when done
llm.unload()  # Unloads from Ollama server
del llm

# Load a different model without running out of memory
llm2 = create_llm("mlx", model="mlx-community/Qwen3-30B-4bit")
response2 = llm2.generate("Hello again")
llm2.unload()  # Frees GPU memory
del llm2
```

**Provider-specific behavior:**
- **Ollama**: Sends `keep_alive=0` to immediately unload from server memory
- **MLX**: Clears model/tokenizer and forces garbage collection
- **HuggingFace**: Closes llama.cpp resources (GGUF) or clears model references
- **LMStudio**: Closes HTTP connection (models auto-unload via TTL)
- **OpenAI/Anthropic**: No-op (cloud providers manage memory)

**Best practices:**
- Call `unload()` when switching between large local models
- Always call in test teardown to prevent OOM errors
- Combine with `del` + `gc.collect()` for immediate cleanup

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
    timeout=300,            # HTTP request timeout (default: 300s, 5 minutes)
    tool_timeout=300,       # Tool execution timeout (default: 300s, 5 minutes)
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

### Vector Embeddings & Similarity Matrix

Built-in SOTA embeddings with advanced similarity computation and clustering:

```python
from abstractllm.embeddings import EmbeddingManager
import numpy as np

embedder = EmbeddingManager()  # Uses all-MiniLM-L6-v2 (perfect clustering, 318K sentences/sec)

# Basic similarity between two texts
similarity = embedder.compute_similarity(
    "machine learning",
    "artificial intelligence"
)
print(f"Similarity: {similarity:.3f}")  # 0.847

# NEW: Batch similarity - compare one text against many
queries = ["What is Python?", "How does AI work?"]
docs = ["Python programming guide", "Machine learning basics", "Web development"]
similarities = embedder.compute_similarities(queries[0], docs)
print(f"Query matches: {similarities}")  # [0.652, 0.234, 0.123]

# NEW: Similarity matrix - compare all texts against all texts (L×C matrix)
texts = ["Python programming", "JavaScript development", "Python data science", "Web frameworks"]
matrix = embedder.compute_similarities_matrix(texts)
print(f"Matrix shape: {matrix.shape}")  # (4, 4) symmetric matrix

# NEW: Asymmetric matrix for query-document matching
queries = ["Learn Python", "Web development guide"]
knowledge_base = ["Python tutorial", "JavaScript guide", "React framework", "Python for beginners"]
search_matrix = embedder.compute_similarities_matrix(queries, knowledge_base)
print(f"Search matrix: {search_matrix.shape}")  # (2, 4) - 2 queries × 4 documents

# NEW: Automatic clustering for content organization
documents = [
    "Python programming tutorial",
    "Learn Python for data science",
    "JavaScript web development",
    "React framework guide",
    "Python machine learning",
    "JavaScript frontend development"
]
clusters = embedder.find_similar_clusters(documents, threshold=0.6, min_cluster_size=2)
print(f"Found {len(clusters)} clusters")  # [[0,1,4], [2,3,5]] - Python cluster & JS cluster

# Performance optimizations
stats = embedder.get_cache_stats()
print(f"Cache: {stats['persistent_cache_size']} regular + {stats['normalized_cache_size']} normalized")
```

**Benchmark-Optimized Model Selection:**
- **all-minilm-l6-v2** (default): Perfect 1.000 clustering purity, 318K sentences/sec, 90MB
- **granite-278m**: Perfect clustering + multilingual support, 567K sentences/sec, 278MB
- **qwen3-embedding**: Near-perfect 0.944 purity, highest coverage, 485K sentences/sec, 600MB
- **granite-30m**: Ultra-efficient 0.856 purity, 520K sentences/sec, only 30MB

```python
# Production multilingual clustering
embedder = EmbeddingManager(model="granite-278m")  # Best balance of quality + multilingual

# Resource-constrained environments
embedder = EmbeddingManager(model="granite-30m")   # Smallest footprint with good quality

# High-coverage applications
embedder = EmbeddingManager(model="qwen3-embedding")  # Clusters more diverse content
```

**Key Matrix Features:**
- **SOTA Performance**: 140x faster than loops using vectorized NumPy operations
- **Memory Efficient**: Automatic chunking for large matrices (handles millions of comparisons)
- **Smart Caching**: Dual-layer caching (memory + disk) with normalized embedding cache for 2x speedup
- **Clustering**: Automatic content organization using similarity thresholds
- **Production Ready**: Event system integration, comprehensive error handling, progress tracking

**[📖 Complete Embeddings Guide →](docs/embeddings.md)** - Detailed documentation, examples, and best practices

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

## 🤖 Agentic CLI Compatibility (Work in Progress)

**Experimental support for using AbstractCore with agentic CLIs and open-source models**

AbstractCore is actively developing compatibility with agentic command-line interfaces:

- **Codex CLI** (OpenAI) - Partial compatibility, ongoing testing
- **Gemini CLI** (Google) - Experimental support
- **Crush** (Charmbracelet) - Experimental support

**Current Status:** Infrastructure is in place (`/v1/responses`, `/v1/chat/completions`, `/v1/messages` endpoints), adaptive message conversion working, but full integration with agentic CLIs requires additional testing and potentially more capable models.

### Quick Setup (Experimental)

```bash
# 1. Start AbstractCore server
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# 2. Configure your agentic CLI
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# 3. Test with your agentic CLI (results may vary)
# Note: May require more capable models for full functionality
codex --model "anthropic/claude-3-5-haiku-latest" "Review this code"
```

### What Works

- ✅ **OpenAI-Compatible Server** - All endpoints functional
- ✅ **Message Format Conversion** - Adapts messages for local vs cloud models
- ✅ **Tool Role Support** - Handles `role: "tool"` messages correctly
- ✅ **Multi-turn Conversations** - Maintains conversation context
- ✅ **Streaming Responses** - Real-time SSE streaming
- ✅ **Structured Output** - JSON schema validation with Pydantic

### Known Limitations

- ⚠️ **Partial Codex Compatibility** - Tested with limited success, may need better models
- ⚠️ **Gemini CLI/Crush** - Not yet tested with real-world usage
- ⚠️ **Model Capabilities** - Some features may require GPT-4 class models
- ⚠️ **Tool Calling** - Works but complex multi-turn tool scenarios need more testing

**[📖 Server Documentation →](docs/server.md)** (Complete server setup, agentic CLI integration, and troubleshooting)

## Advanced Capabilities

AbstractCore is designed as infrastructure. For advanced AI applications, combine with:

### [AbstractMemory](https://github.com/lpalbou/AbstractMemory)
Temporal knowledge graphs and advanced memory systems for AI agents.

### [AbstractAgent](https://github.com/lpalbou/AbstractAgent)
Autonomous agents with planning, tool execution, and self-improvement capabilities.

## Documentation

### Built-in Applications
- **[📝 Basic Summarizer](docs/basic-summarizer.md)** - Production-ready text summarization with multiple styles and formats
- **[⚖️ Basic Judge](docs/basic-judge.md)** - LLM-as-a-judge for objective evaluation with structured assessments and multiple file support
- **[🕸️ Basic Extractor](docs/basic-extractor.md)** - Knowledge graph extraction with JSON-LD and RDF triple outputs

### Core Features
- **[🌐 Server Guide](docs/server.md)** - Universal API server documentation
- **[🤖 Agentic CLI Compatibility](docs/server.md)** - Use with Codex, Gemini CLI, Crush (see Server Documentation)
- **[🔢 Vector Embeddings](docs/embeddings.md)** - Similarity matrices, clustering, and semantic search
- **[💬 Chat Compaction](docs/chat-compaction.md)** - SOTA conversation history summarization

### Getting Started
- **[📋 Prerequisites & Setup](docs/prerequisites.md)** - Complete setup guide for all providers (API keys, Ollama, LMStudio, etc.)
- **[Getting Started](docs/getting-started.md)** - Your first AbstractCore program
- **[⌨️ Internal CLI](docs/internal-cli.md)** - Built-in CLI for testing and interactive conversations
- **[Capabilities](docs/capabilities.md)** - What AbstractCore can and cannot do
- **[Providers](docs/providers.md)** - Complete provider guide
- **[Examples](docs/examples.md)** - Real-world use cases

### Reference
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Architecture](docs/architecture.md)** - How it works internally
- **[Framework Comparison](docs/comparison.md)** - Detailed comparison with alternatives

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Testing Status

**✅ All tests passing as of October 9th, 2025**

**Test Environment:**
- **Hardware**: MacBook Pro (14-inch, Nov 2024)
- **Chip**: Apple M4 Max
- **Memory**: 128 GB
- **OS**: macOS Sequoia 15.3.1

**Software Environment:**
- **Python**: 3.12.2
- **Ollama**: 0.12.3 (with 22 models including gpt-oss:120b, qwen3-coder:30b, embedding models)
- **LM Studio**: 0.3.30

**Key Dependencies (Virtual Environment):**
- **Core**: pydantic 2.11.9, httpx 0.28.1, tiktoken 0.11.0
- **LLM Providers**: openai 1.108.1, anthropic 0.68.0
- **Local ML**: transformers 4.56.2, torch 2.8.0, mlx 0.29.2, mlx-lm 0.28.1, llama-cpp-python 0.3.16
- **Embeddings**: sentence-transformers 5.1.1, numpy 2.3.3
- **Server**: fastapi 0.117.1, uvicorn 0.37.0, sse-starlette 3.0.2

The full test suite has been validated on this configuration, ensuring compatibility with Apple Silicon M4 Max and high-memory environments. All core features, providers, tools, embeddings, and server functionality are working correctly. Dependencies match pyproject.toml requirements with no conflicts detected.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for a complete list of dependencies and contributors.

---

**AbstractCore** - Clean infrastructure for LLM applications 🚀