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

- **[Prerequisites](prerequisites.md)** - Provider setup and configuration (OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace)
- **[Capabilities](capabilities.md)** - What AbstractCore can and cannot do

### Advanced Topics

- **[Tool Call Syntax Rewriting](tool-syntax-rewriting.md)** - Tool format conversion for agentic CLIs
- **[Internal CLI](internal-cli.md)** - Built-in CLI tool for testing and exploration

## 🌐 Server (Optional HTTP REST API)

**The server is an optional component that provides OpenAI-compatible HTTP endpoints.**

### Server Documentation

- **[Server Documentation](server.md)** - Complete guide including:
  - Quick start (5 minutes)
  - Configuration and environment variables
  - API endpoints: chat completions, embeddings, models, providers
  - Use cases and examples
  - Agentic CLI integration (Codex, Crush, Gemini CLI)
  - Deployment (Docker, production, cloud)

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
├── README.md                   # This file - navigation guide
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
│   ├── capabilities.md         # What AbstractCore can do
│   └── tool-syntax-rewriting.md  # Tool format conversion
│
├── Server (Optional HTTP REST API)/
│   └── server.md               # Complete server documentation ⭐
│
├── Specialized/
│   ├── apps/                   # Built-in applications
│   │   ├── basic-extractor.md      # Data extraction
│   │   ├── basic-judge.md          # Text evaluation
│   │   └── basic-summarizer.md     # Summarization
│   ├── internal-cli.md         # Built-in CLI tool
│   ├── architecture.md         # System architecture
│   ├── comparison.md           # vs alternatives
│   └── chat-compaction.md      # History management
│
└── Archive/
    └── README.md               # Superseded documentation
```

**⭐ Key Distinction:**
- **`api-reference.md`** = Python library API (functions, classes)
- **`server.md`** = HTTP REST API (endpoints, requests)

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

## 📖 Reading Paths

### For Beginners
1. [Prerequisites](prerequisites.md) - Setup
2. [Getting Started](getting-started.md) - Learn basics
3. [Examples](examples.md) - See real code
4. [Troubleshooting](troubleshooting.md) - Fix issues

### For Server/API Users
1. [Server Documentation](server.md) - Setup server and learn API endpoints
2. [Prerequisites](prerequisites.md) - Configure providers
3. [Troubleshooting](troubleshooting.md) - Fix server issues

### For Advanced Users
1. [Architecture](architecture.md) - Understand system
2. [Tool Call Syntax Rewriting](tool-syntax-rewriting.md) - Format conversion
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
| server.md | Server + REST API | ✅ Consolidated | Oct 12, 2025 |
| troubleshooting.md | Core + Server | ✅ Consolidated | Oct 12, 2025 |

**All documents cross-referenced and up-to-date as of October 12, 2025.**

**Key Files:**
- **`api-reference.md`** - For Python programmers using AbstractCore library
- **`server.md`** - For HTTP/REST API integration with the server

---

**Start your journey:** [Prerequisites](prerequisites.md) → [Getting Started](getting-started.md) → Build amazing things! 🚀

