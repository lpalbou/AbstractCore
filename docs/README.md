# AbstractCore Documentation Index

Complete navigation guide for AbstractCore documentation.

## Getting Started

**New to AbstractCore? Start here:**

1. **[Prerequisites](prerequisites.md)** - Install and configure providers (OpenAI, Anthropic, Ollama, etc.)
2. **[Getting Started](getting-started.md)** - 5-minute quick start with core concepts and examples
3. **[Troubleshooting](troubleshooting.md)** - Fix common issues quickly

## Core Library (Python API)

**AbstractCore is primarily a Python library for programmatic LLM usage.**

### Essential Guides

- **[Getting Started](getting-started.md)** - Quick start, core concepts, common patterns
- **[Prerequisites](prerequisites.md)** - Provider setup (OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace)
- **[Async/Await Support](api-reference.md#agenerate)** - Concurrent requests for faster batch operations
- **[Interaction Tracing](interaction-tracing.md)** - Programmatic observability for prompts, responses, timing, and usage
- **[Glyph Visual-Text Compression](glyphs.md)** - Experimental vision-based compression that can reduce token usage for long documents (results vary)
- **[Structured Output](structured-output.md)** - Pydantic models, schema validation, native vs prompted strategies
- **[Session Management](session.md)** - Persistent conversations, serialization, and analytics
- **[Embeddings](embeddings.md)** - Vector embeddings, semantic search, RAG applications
- **[Examples](examples.md)** - Real-world use cases and code samples

### Python API Reference

- **[Python API Reference](api-reference.md)** - Complete Python API: functions, classes, methods

### Provider Documentation

- **[Prerequisites](prerequisites.md)** - Provider setup and configuration (OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace)
- **[Capabilities](capabilities.md)** - What AbstractCore can and cannot do

### Advanced Topics

- **[Media Handling System](media-handling-system.md)** - Images, PDFs, documents with vision model optimization
- **[Concurrency & Throughput](concurrency.md)** - How to measure and reason about concurrency (incl. MLX continuous batching)
- **[Glyph Visual-Text Compression](glyphs.md)** - Advanced compression techniques for large documents
- **[Vision Capabilities](vision-capabilities.md)** - Vision model integration and optimization
- **[Tool Calling](tool-calling.md)** - Universal tool system and format conversion
- **[Tool Syntax Rewriting](tool-syntax-rewriting.md)** - Convert tool-call syntax for different runtimes/clients
- **[MCP (Model Context Protocol)](mcp.md)** - Discover tools from MCP servers (HTTP/stdio) and route tool calls
- **[AbstractCore CLI](acore-cli.md)** - Built-in CLI tool for testing and exploration

## Server (Optional HTTP REST API)

**The server is an optional component that provides OpenAI-compatible HTTP endpoints.**

### Server Documentation

- **[Server Documentation](server.md)** - Complete guide including:
  - Quick start (5 minutes)
  - Configuration and environment variables
  - API endpoints: chat completions, embeddings, models, providers
  - Use cases and examples
  - Agentic CLI integration (Codex, Crush, Gemini CLI)
  - Deployment (Docker, production, cloud)

## Built-in Applications (CLI Tools)

**AbstractCore includes five production-ready command-line applications for common LLM tasks.**

### Quick Start with Apps

```bash
# Install and use immediately (pick one)
pip install "abstractcore[all-apple]"    # macOS/Apple Silicon (includes MLX, excludes vLLM)
pip install "abstractcore[all-non-mlx]"  # Linux/Windows/Intel Mac (excludes MLX and vLLM)
pip install "abstractcore[all-gpu]"      # Linux NVIDIA GPU (includes vLLM, excludes MLX)

# Direct terminal usage (no Python code needed)
summarizer document.pdf --provider openai --model gpt-4o-mini
extractor research_paper.pdf --format json-ld --provider anthropic  
judge essay.txt --criteria clarity,accuracy --provider ollama
```

### Application Documentation

- **[Summarizer Guide](apps/basic-summarizer.md)** - Document summarization with multiple strategies
- **[Extractor Guide](apps/basic-extractor.md)** - Entity and relationship extraction from text
- **[Judge Guide](apps/basic-judge.md)** - Text evaluation and scoring systems
- **[Intent Guide](apps/basic-intent.md)** - Intent analysis & deception detection
- **[DeepSearch Guide](apps/basic-deepsearch.md)** - Autonomous web-assisted research

**Key Features:**
- **Direct CLI usage**: `summarizer`, `extractor`, `judge`, `intent`, `deepsearch`
- **Provider agnostic**: Works with any configured LLM provider
- **Multiple formats** (with `pip install "abstractcore[media]"`): PDF, TXT, MD, DOCX support
- **Batch processing**: Handle multiple files at once
- **Production ready**: Robust error handling and logging

## Specialized Topics

### Architecture & Design

- **[Architecture](architecture.md)** - System architecture overview
- **[Comparison](comparison.md)** - Compare AbstractCore with alternatives
- **[Chat Compaction](chat-compaction.md)** - Manage conversation history efficiently

## Troubleshooting & Help

- **[Troubleshooting](troubleshooting.md)** - Comprehensive troubleshooting guide:
  - Installation issues
  - Core library issues
  - Server issues
  - Provider-specific issues
  - Performance issues
  - Debug techniques

## Documentation Structure

```
docs/
├── README.md                   # This file - navigation guide
│
├── Getting Started/
│   ├── prerequisites.md        # Provider setup
│   ├── getting-started.md      # Quick start guide
│   └── troubleshooting.md      # Common issues
│
├── Core Library (Python)/
│   ├── api-reference.md        # Python API reference
│   ├── structured-output.md    # Structured output with Pydantic
│   ├── embeddings.md           # Embeddings guide
│   ├── examples.md             # Code examples
│   ├── capabilities.md         # What AbstractCore can do
│   ├── interaction-tracing.md  # Observability for prompts/responses/timing
│   ├── mcp.md                  # MCP tool servers (HTTP/stdio)
│   └── tool-syntax-rewriting.md  # Tool format conversion
│
├── Server (Optional HTTP REST API)/
│   └── server.md               # Complete server documentation 
│
├── Built-in Applications (CLI Tools)/
│   ├── apps/                   # Production-ready CLI applications
│   │   ├── basic-summarizer.md     # Document summarization
│   │   ├── basic-extractor.md      # Entity/relationship extraction
│   │   ├── basic-judge.md          # Text evaluation and scoring
│   │   ├── basic-intent.md         # Intent analysis & deception detection
│   │   └── basic-deepsearch.md     # Multi-stage research with web search
│
├── Specialized/
│   ├── acore-cli.md           # Interactive CLI tool for development
│   ├── architecture.md         # System architecture
│   ├── comparison.md           # vs alternatives
│   └── chat-compaction.md      # History management
│
├── reports/                    # Technical reports (see reports/README.md)
├── research/                   # Research notes / experiments (see research/README.md)
└── Archive/
    └── README.md               # Superseded documentation (historical)
```

**Key Distinction:**
- **`api-reference.md`** = Python library API (functions, classes)
- **`server.md`** = HTTP REST API (endpoints, requests)

## Quick Navigation

### I want to...

**Get Started:**
- Install AbstractCore → [Prerequisites](prerequisites.md)
- Make my first LLM call → [Getting Started](getting-started.md)
- Fix installation issues → [Troubleshooting](troubleshooting.md#installation-issues)

**Use Built-in Apps (CLI Tools):**
- Summarize documents → [Summarizer Guide](apps/basic-summarizer.md)
- Extract entities → [Extractor Guide](apps/basic-extractor.md)  
- Evaluate text → [Judge Guide](apps/basic-judge.md)
- Analyze intent → [Intent Guide](apps/basic-intent.md)
- Run deep research → [DeepSearch Guide](apps/basic-deepsearch.md)
- Quick start → `pip install "abstractcore[all-apple]"` / `"abstractcore[all-non-mlx]"` / `"abstractcore[all-gpu]"` then `summarizer --help`

**Use Core Library (Python):**
- Switch between providers → [Getting Started](getting-started.md#providers-and-models)
- Use tool calling → [Getting Started](getting-started.md#tool-calling)
- Get structured outputs → [Structured Output](structured-output.md)
- Stream responses → [Getting Started](getting-started.md#streaming)
- Generate embeddings → [Embeddings](embeddings.md)
- See Python API → [Python API Reference](api-reference.md)

**Set Up Server (HTTP REST API):**
- Start the server → [Server Documentation](server.md#quick-start)
- Configure environment → [Server Documentation](server.md#configuration)
- Use with OpenAI client → [Server Documentation](server.md#quick-start)
- Integrate with Codex CLI → [Server Documentation](server.md#agentic-cli-integration)
- Deploy to production → [Server Documentation](server.md#deployment)

**REST API Integration:**
- See all HTTP endpoints → [Server Documentation](server.md#api-endpoints)
- Use chat completions → [Server Documentation](server.md#chat-completions)
- Create embeddings → [Server Documentation](server.md#embeddings)
- List models → [Server Documentation](server.md#model-discovery)
- Check providers → [Server Documentation](server.md#provider-status)

**Troubleshoot:**
- Fix authentication errors → [Troubleshooting](troubleshooting.md#authentication-errors)
- Solve connection issues → [Troubleshooting](troubleshooting.md#connection-errors)
- Debug server problems → [Troubleshooting](troubleshooting.md#server-issues)
- Improve performance → [Troubleshooting](troubleshooting.md#performance-issues)

### By Provider

**OpenAI:**
- Setup → [Prerequisites](prerequisites.md#openai-setup)
- Issues → [Troubleshooting](troubleshooting.md#openai)

**Anthropic:**
- Setup → [Prerequisites](prerequisites.md#anthropic-setup)
- Issues → [Troubleshooting](troubleshooting.md#anthropic)

**Ollama:**
- Setup → [Prerequisites](prerequisites.md#ollama-setup)
- Issues → [Troubleshooting](troubleshooting.md#ollama)

**LMStudio:**
- Setup → [Prerequisites](prerequisites.md#lmstudio-setup)
- Issues → [Troubleshooting](troubleshooting.md#lmstudio)

**MLX:**
- Setup → [Prerequisites](prerequisites.md#mlx-setup)
- HuggingFace Setup → [Prerequisites](prerequisites.md#huggingface-setup)

## Reading Paths

### For Beginners
1. [Prerequisites](prerequisites.md) - Setup
2. [Getting Started](getting-started.md) - Learn basics
3. [Built-in Apps](apps/) - Use CLI tools immediately
4. [Examples](examples.md) - See real code
5. [Troubleshooting](troubleshooting.md) - Fix issues

### For Server/API Users
1. [Server Documentation](server.md) - Setup server and learn API endpoints
2. [Prerequisites](prerequisites.md) - Configure providers
3. [Troubleshooting](troubleshooting.md) - Fix server issues

### For Advanced Users
1. [Architecture](architecture.md) - Understand system
2. [Tool Call Syntax Rewriting](tool-syntax-rewriting.md) - Format conversion
3. [AbstractCore CLI](acore-cli.md) - Advanced CLI usage
4. [Capabilities](capabilities.md) - Deep dive into features

## External Links

- **GitHub Repository**: [lpalbou/AbstractCore](https://github.com/lpalbou/AbstractCore)
- **Issues**: [Report bugs](https://github.com/lpalbou/AbstractCore/issues)
- **Discussions**: [Get help](https://github.com/lpalbou/AbstractCore/discussions)

## Document Status

Core docs in `docs/` aim to reflect the current codebase.

Non-authoritative folders:
- `docs/reports/` (engineering notes) → see `docs/reports/README.md`
- `docs/research/` (experiments) → see `docs/research/README.md`
- `docs/archive/` (historical) → see `docs/archive/README.md`

If you find a mismatch between docs and code, please open an issue.

**Key Files:**
- **`api-reference.md`** - For Python programmers using AbstractCore library
- **`server.md`** - For HTTP/REST API integration with the server

---

**Start your journey:** [Prerequisites](prerequisites.md) → [Getting Started](getting-started.md) → Build your applications!
