# AbstractCore Documentation Index

Complete navigation guide for AbstractCore documentation.

## 🚀 Getting Started

**New to AbstractCore? Start here:**

1. **[Prerequisites](prerequisites.md)** - Install and configure providers (OpenAI, Anthropic, Ollama, etc.)
2. **[Getting Started](getting-started.md)** - 5-minute quick start with core concepts and examples
3. **[Troubleshooting](troubleshooting.md)** - Fix common issues quickly

## 📚 Core Library (Python API)

**AbstractCore is primarily a Python library for programmatic LLM usage.**

### Essential Guides

- **[Getting Started](getting-started.md)** - Quick start, core concepts, common patterns
- **[Prerequisites](prerequisites.md)** - Provider setup (OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace)
- **[Embeddings](embeddings.md)** - Vector embeddings, semantic search, RAG applications
- **[Examples](examples.md)** - Real-world use cases and code samples

### Python API Reference

- **[Python API Reference](api-reference.md)** - Complete Python API: functions, classes, methods

### Provider Documentation

- **[Providers](providers.md)** - Deep dive into each provider's capabilities
- **[Capabilities](capabilities.md)** - What AbstractCore can and cannot do

### Advanced Topics

- **[Tool Call Tag Rewriting](tool-call-tag-rewriting.md)** - Tool format conversion for agentic CLIs
- **[Internal CLI](internal-cli.md)** - Built-in CLI tool for testing and exploration
- **[Common Mistakes](common-mistakes.md)** - Pitfalls to avoid

## 🌐 Server (Optional HTTP REST API)

**The server is an optional component that provides OpenAI-compatible HTTP endpoints.**

### Server Documentation

- **[Server Guide](server.md)** - Complete server setup and deployment:
  - Quick start (5 minutes)
  - Configuration
  - Use cases
  - Agentic CLI integration
  - Deployment

- **[Server API Reference](server-api-reference.md)** - Complete REST API documentation:
  - Chat completions (`/v1/chat/completions`)
  - Embeddings (`/v1/embeddings`)
  - Models (`/v1/models`)
  - Providers (`/providers`)
  - All HTTP request/response formats
  - Agentic CLI integration (Codex, Crush, Gemini CLI)

## 🔧 Specialized Topics

### Application Guides

- **[Extractor](apps/basic-extractor.md)** - Extract structured data from text
- **[Judge](apps/basic-judge.md)** - Evaluate and score text
- **[Summarizer](apps/basic-summarizer.md)** - Generate summaries

### Architecture & Design

- **[Architecture](architecture.md)** - System architecture overview
- **[Comparison](comparison.md)** - Compare AbstractCore with alternatives
- **[Chat Compaction](chat-compaction.md)** - Manage conversation history efficiently

## 🐛 Troubleshooting & Help

- **[Troubleshooting](troubleshooting.md)** - Comprehensive troubleshooting guide:
  - Installation issues
  - Core library issues
  - Server issues
  - Provider-specific issues
  - Performance issues
  - Debug techniques

## 📁 Documentation Structure

```
docs/
├── INDEX.md                    # This file - navigation guide
│
├── Getting Started/
│   ├── prerequisites.md        # Provider setup
│   ├── getting-started.md      # Quick start guide
│   └── troubleshooting.md      # Common issues
│
├── Core Library (Python)/
│   ├── api-reference.md        # Python API reference ⭐
│   ├── embeddings.md           # Embeddings guide
│   ├── examples.md             # Code examples
│   ├── providers.md            # Provider details
│   ├── capabilities.md         # What AbstractCore can do
│   └── tool-call-tag-rewriting.md  # Tool format conversion
│
├── Server (Optional HTTP REST API)/
│   ├── server.md               # Server guide (consolidated)
│   └── server-api-reference.md # REST API reference ⭐
│
├── Specialized/
│   ├── apps/                   # Built-in applications
│   │   ├── basic-extractor.md      # Data extraction
│   │   ├── basic-judge.md          # Text evaluation
│   │   └── basic-summarizer.md     # Summarization
│   ├── internal-cli.md         # Built-in CLI tool
│   ├── architecture.md         # System architecture
│   ├── comparison.md           # vs alternatives
│   ├── common-mistakes.md      # Pitfalls
│   └── chat-compaction.md      # History management
│
└── Archive/
    └── README.md               # Superseded documentation
```

**⭐ Key Distinction:**
- **`api-reference.md`** = Python library API (functions, classes)
- **`server-api-reference.md`** = HTTP REST API (endpoints, requests)

## 🎯 Quick Navigation

### I want to...

**Get Started:**
- Install AbstractCore → [Prerequisites](prerequisites.md)
- Make my first LLM call → [Getting Started](getting-started.md)
- Fix installation issues → [Troubleshooting](troubleshooting.md#installation-issues)

**Use Core Library (Python):**
- Switch between providers → [Getting Started](getting-started.md#providers-and-models)
- Use tool calling → [Getting Started](getting-started.md#tool-calling)
- Get structured outputs → [Getting Started](getting-started.md#structured-output)
- Stream responses → [Getting Started](getting-started.md#streaming)
- Generate embeddings → [Embeddings](embeddings.md)
- See Python API → [Python API Reference](api-reference.md)

**Set Up Server (HTTP REST API):**
- Start the server → [Server Guide](server.md#quick-start-5-minutes)
- Configure environment → [Server Guide](server.md#configuration)
- Use with OpenAI client → [Server Guide](server.md#use-cases)
- Integrate with Codex CLI → [Server Guide](server.md#agentic-cli-integration)
- Deploy to production → [Server Guide](server.md#deployment)

**REST API Integration:**
- See all HTTP endpoints → [Server API Reference](server-api-reference.md)
- Use chat completions → [Server API Reference](server-api-reference.md#chat-completions-endpoints)
- Create embeddings → [Server API Reference](server-api-reference.md#embeddings-endpoint)
- List models → [Server API Reference](server-api-reference.md#models-endpoint)
- Check providers → [Server API Reference](server-api-reference.md#providers-endpoint)

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

## 📖 Reading Paths

### For Beginners
1. [Prerequisites](prerequisites.md) - Setup
2. [Getting Started](getting-started.md) - Learn basics
3. [Examples](examples.md) - See real code
4. [Troubleshooting](troubleshooting.md) - Fix issues

### For API Users
1. [Server Guide](server.md) - Setup server
2. [API Reference](api-reference.md) - Learn endpoints
3. [Codex CLI Integration](codex-cli-integration.md) - CLI integration
4. [Troubleshooting](troubleshooting.md) - Fix server issues

### For Advanced Users
1. [Architecture](architecture.md) - Understand system
2. [Tool Call Tag Rewriting](tool-call-tag-rewriting.md) - Format conversion
3. [Internal CLI](internal-cli.md) - Advanced CLI usage
4. [Capabilities](capabilities.md) - Deep dive into features

## 🔗 External Links

- **GitHub Repository**: [lpalbou/AbstractCore](https://github.com/lpalbou/AbstractCore)
- **Issues**: [Report bugs](https://github.com/lpalbou/AbstractCore/issues)
- **Discussions**: [Get help](https://github.com/lpalbou/AbstractCore/discussions)

## 📝 Document Status

| Document | Type | Status | Last Updated |
|----------|------|--------|--------------|
| README.md | Overview | ✅ Updated | Oct 12, 2025 |
| getting-started.md | Core Library | ✅ Current | Oct 12, 2025 |
| prerequisites.md | Core Library | ✅ Current | Oct 12, 2025 |
| api-reference.md | Python API | ✅ Current | Oct 12, 2025 |
| embeddings.md | Core Library | ✅ Current | Oct 12, 2025 |
| server.md | Server | ✅ Consolidated | Oct 12, 2025 |
| server-api-reference.md | REST API | ✅ Consolidated | Oct 12, 2025 |
| troubleshooting.md | Both | ✅ Consolidated | Oct 12, 2025 |

**All documents cross-referenced and up-to-date as of October 12, 2025.**

**Key Files:**
- **`api-reference.md`** - For Python programmers using AbstractCore library
- **`server-api-reference.md`** - For HTTP/REST API integration with the server

---

**Start your journey:** [Prerequisites](prerequisites.md) → [Getting Started](getting-started.md) → Build amazing things! 🚀

