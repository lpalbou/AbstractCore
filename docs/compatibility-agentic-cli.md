# Agentic CLI Compatibility Guide

**Making AbstractCore work with popular agentic CLIs using powerful open-source models**

AbstractCore provides full compatibility with leading agentic command-line interfaces, enabling you to use powerful open-source models like Qwen3-Coder 30B, Llama 3, and others with tools like Codex, Gemini CLI, Crush, and more.

## 🎯 Quick Start

### 1. Start AbstractCore Server

```bash
# Install with server support
pip install abstractcore[server,ollama,openai,anthropic]

# Start the server
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000
```

### 2. Configure Your Agentic CLI

Point your agentic CLI to AbstractCore's server using the OpenAI-compatible endpoint:

```bash
# Standard OpenAI endpoint (works for all supported CLIs)
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
```

## 🔧 Supported Agentic CLIs

### Codex (OpenAI)

OpenAI's Codex CLI supports both Responses API and Chat Completions API patterns.

**Configuration:**
```bash
# Standard configuration
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Run codex
codex --model "ollama/qwen3-coder:30b" "Write a Python function to sort a list"
```

**Tool Calling Support:**
```bash
# Codex with tool calling (automatic detection)
codex --model "anthropic/claude-3-5-haiku-latest" --tools "file_operations,web_search"
```

**GitHub**: [github.com/openai/codex](https://github.com/openai/codex)

### Gemini CLI (Google)

Google's Gemini CLI provides powerful AI capabilities for development workflows.

**Configuration:**
```bash
# Set up base URL and API key
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Run Gemini CLI
gemini-cli --model "ollama/qwen3-coder:30b" "Analyze this codebase"
```

**Features:**
- ✅ Code generation and analysis
- ✅ Multi-file operations
- ✅ Codebase understanding
- ✅ Streaming responses

**GitHub**: [github.com/google/gemini-cli](https://github.com/google/gemini-cli)

### Crush CLI (Charmbracelet)

Crush (by Charmbracelet) provides a glamorous terminal AI experience with excellent UX.

**Configuration:**
```bash
# Set up base URL and API key
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Run crush
crush --model "ollama/qwen3-coder:30b" "Write a FastAPI server"
```

**Features:**
- ✅ Beautiful terminal UI
- ✅ Interactive conversations
- ✅ File operations
- ✅ Code generation

**GitHub**: [github.com/charmbracelet/crush](https://github.com/charmbracelet/crush)

## 🎨 Supported Features

### Core Capabilities

- ✅ **Full OpenAI Compatibility** - Works with any OpenAI-compatible client
- ✅ **Tool Calling & Function Calling** - Native support across all providers
- ✅ **Streaming Responses** - Real-time SSE streaming with tool call deltas
- ✅ **Parallel Tool Calls** - Execute multiple tools simultaneously
- ✅ **Structured Output** - JSON schema validation and Pydantic models
- ✅ **MCP Support** - Model Context Protocol for advanced tool integration
- ✅ **Multiple Routing Patterns** - `/v1/chat/completions` and `/{provider}/v1/chat/completions`

### Provider Support

AbstractCore seamlessly routes to any provider:

```bash
# Use Ollama (local)
codex --model "ollama/qwen3-coder:30b"

# Use OpenAI
codex --model "openai/gpt-4o-mini"

# Use Anthropic
codex --model "anthropic/claude-3-5-haiku-latest"

# Use LMStudio (local)
codex --model "lmstudio/qwen/qwen3-next-80b"

# Use MLX (Apple Silicon)
codex --model "mlx/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
```

## 🎛️ Server Endpoints

AbstractCore provides these endpoints for maximum CLI compatibility:

### Standard OpenAI Endpoints
```
POST /v1/chat/completions          # Main chat endpoint
POST /v1/completions               # Legacy text completions
POST /v1/responses                 # OpenAI Responses API (Codex)
GET  /v1/models                    # List all models
POST /v1/embeddings                # Generate embeddings
GET  /v1/capabilities              # Server capabilities
```

### Anthropic Endpoints
```
POST /v1/messages                  # Anthropic Messages API
```

### Provider-Specific Endpoints
```
POST /{provider}/v1/chat/completions    # Provider routing
GET  /{provider}/v1/models              # Provider models
POST /{provider}/v1/embeddings          # Provider embeddings
```

### MCP (Model Context Protocol) Endpoints ⚠️ **Stub Implementation**
```
POST /v1/mcp/servers               # Register MCP server (stub)
GET  /v1/mcp/servers               # List MCP servers (stub)
DELETE /v1/mcp/servers/{id}        # Unregister server (stub)
```
**Note:** MCP endpoints exist but are placeholder implementations. Full MCP support is planned for a future release.

### Utility Endpoints
```
GET  /                             # Server info
GET  /health                       # Health check
GET  /providers                    # Available providers
```

## 🚀 Recommended Models

### For Coding Tasks

**Large Models (30B+):**
- `ollama/qwen3-coder:30b` - Excellent code generation and analysis
- `ollama/deepseek-coder:33b` - Strong reasoning and debugging
- `lmstudio/qwen/qwen3-next-80b` - Best quality for complex tasks

**Medium Models (7-13B):**
- `ollama/codellama:13b` - Fast code completion
- `ollama/starcoder2:7b` - Lightweight code generation

**Apple Silicon Optimized:**
- `mlx/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` - Optimized for M-series chips
- `mlx/mlx-community/GLM-4.5-Air-4bit` - Fast and efficient

### For General Tasks

**Cloud Models:**
- `anthropic/claude-3-5-haiku-latest` - Fast and intelligent
- `openai/gpt-4o-mini` - Reliable and cost-effective

**Local Models:**
- `ollama/llama3:8b` - Local and private
- `ollama/mistral:7b` - Fast general purpose

## 📝 Example Workflows

### Code Generation with Codex

```bash
# Simple code generation
codex --model "ollama/qwen3-coder:30b" "Create a REST API with authentication"

# With file operations
codex --model "ollama/qwen3-coder:30b" --tools "file_operations" "Refactor this Python module"

# Multi-step workflow
codex --model "ollama/qwen3-coder:30b" "Analyze the codebase, fix bugs, and write tests"
```

### Development with Gemini CLI

```bash
# Codebase analysis
gemini-cli --model "ollama/qwen3-coder:30b" "What does this project do?"

# Feature implementation
gemini-cli --model "anthropic/claude-3-5-haiku-latest" "Add user authentication"

# Code review
gemini-cli --model "ollama/deepseek-coder:33b" "Review changes in git diff"
```

### Interactive Coding with Crush

```bash
# Start interactive session
crush --model "ollama/qwen3-coder:30b"

# Within Crush:
> Write a FastAPI endpoint for user registration
> Add input validation
> Write unit tests
```

## 🔍 Troubleshooting

### Connection Issues

**Problem:** CLI can't connect to AbstractCore
```bash
# Check if server is running
curl http://localhost:8000/health

# Check correct base URL
echo $OPENAI_BASE_URL  # Should be http://localhost:8000/v1
```

### Model Not Found

**Problem:** Model name not recognized
```bash
# List available models
curl http://localhost:8000/v1/models

# Use correct format: provider/model
codex --model "ollama/qwen3-coder:30b"  # Correct
codex --model "qwen3-coder:30b"         # Wrong
```

### Tool Calling Issues

**Problem:** Tools not executing

**Solution:** Check that AbstractCore server is running with tool execution enabled:
```bash
# Server logs should show tool calls
tail -f logs/abstractllm_*.log
```

### Performance Issues

**Problem:** Slow responses

**Solutions:**
- Use smaller models for faster responses (`7b` instead of `30b`)
- Enable GPU acceleration for Ollama: `OLLAMA_GPU_LAYERS=-1`
- Use MLX provider for Apple Silicon
- Check network latency to AbstractCore server

## 🎓 Advanced Usage

### Custom Provider Configuration

```bash
# Use specific provider endpoint
export OPENAI_BASE_URL="http://localhost:8000/ollama/v1"

# This forces all requests to use Ollama provider
codex --model "qwen3-coder:30b"  # Auto-routes to Ollama
```

### Environment-Specific Setup

**Development:**
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dev-key"
```

**Production:**
```bash
export OPENAI_BASE_URL="https://abstractcore.example.com/v1"
export OPENAI_API_KEY="prod-key-from-secrets"
```

### Multi-Model Workflows

```bash
# Use different models for different tasks
codex --model "ollama/qwen3-coder:30b" "Write code"  # Best for code
gemini-cli --model "anthropic/claude-3-5-haiku-latest" "Write docs"  # Best for prose
crush --model "ollama/llama3:8b" "Quick answer"  # Fast for simple tasks
```

## 📊 Performance Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| qwen3-coder:30b | 30B | Medium | Excellent | Complex coding tasks |
| deepseek-coder:33b | 33B | Medium | Excellent | Debugging, analysis |
| codellama:13b | 13B | Fast | Good | Code completion |
| llama3:8b | 8B | Very Fast | Good | General tasks |
| gpt-4o-mini | Cloud | Fast | Excellent | Production use |
| claude-3-5-haiku | Cloud | Very Fast | Excellent | Fast intelligent tasks |

## 🔗 Resources

### Official Documentation

- **AbstractCore Docs**: [github.com/lpalbou/abstractllm_core](https://github.com/lpalbou/abstractllm_core)
- **Codex GitHub**: [github.com/openai/codex](https://github.com/openai/codex)
- **Gemini CLI GitHub**: [github.com/google/gemini-cli](https://github.com/google/gemini-cli)
- **Crush GitHub**: [github.com/charmbracelet/crush](https://github.com/charmbracelet/crush)

### Model Resources

- **Ollama Models**: [ollama.com/library](https://ollama.com/library)
- **MLX Models**: [huggingface.co/mlx-community](https://huggingface.co/mlx-community)
- **Anthropic Models**: [docs.anthropic.com/models](https://docs.anthropic.com/models)
- **OpenAI Models**: [platform.openai.com/docs/models](https://platform.openai.com/docs/models)

## 💡 Tips & Best Practices

### 1. Model Selection
- **Start with 7-8B models** for quick iterations
- **Use 30B+ models** for production-quality results
- **Cloud models** (GPT-4o-mini, Claude) for best quality when cost isn't a concern

### 2. Performance Optimization
- **Local models**: Prefer Ollama or MLX for low latency
- **Cloud models**: Better for complex reasoning
- **Hybrid approach**: Local for drafts, cloud for refinement

### 3. Tool Usage
- **Enable tools** for file operations and web search
- **Limit tool scope** to prevent unwanted modifications
- **Review tool outputs** before applying changes

### 4. Cost Management
- **Local models**: Free but require hardware
- **Cloud models**: Paid but more capable
- **Cache frequently used prompts** to reduce costs

---

**Ready to start?** Install AbstractCore and configure your favorite agentic CLI to unlock powerful AI coding capabilities with open-source models!

```bash
pip install abstractcore[server,ollama]
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000
```
