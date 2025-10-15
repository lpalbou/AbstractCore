# AbstractCore Server - 5-Minute Quick Start

## What You'll Achieve

In 5 minutes, you'll have a universal LLM gateway running locally that:
- Works with ANY OpenAI-compatible client
- Supports local models (Ollama, LMStudio) and cloud providers
- Automatically converts tool calls between different formats
- Powers agentic CLIs like Codex, Crush, and Gemini CLI

## Prerequisites

- **Python 3.9+** - Check: `python --version`
- **pip** - Check: `python -m pip --version`
- **Port 8000 available** - Check: `lsof -i :8000` (should show nothing)

## Step 1: Install (30 seconds)

```bash
pip install abstractcore[server]
```

## Step 2: Start Server (10 seconds)

```bash
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 3: Quick Test (20 seconds)

Open a new terminal and test the server:

```bash
# Check server health
curl http://localhost:8000/health
# Expected: {"status":"healthy"}

# List available models
curl http://localhost:8000/v1/models
# Expected: JSON list of available models
```

## Step 4: Your First Generation (1 minute)

### Option A: Using curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello! Write a haiku about coding."}]
  }'
```

### Option B: Using Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello! Write a haiku about coding."}]
)
print(response.choices[0].message.content)
```

## Step 5: Choose Your Model (2 minutes)

### Local Models (No API Key Required)

**Ollama** - Run powerful models locally:
```bash
# Install Ollama first: https://ollama.ai
ollama pull qwen3-coder:30b

# Use with AbstractCore
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

**LMStudio** - GUI for local models:
```bash
# Start LMStudio on localhost:1234, then:
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio",
    "messages": [{"role": "user", "content": "Explain recursion"}]
  }'
```

### Cloud Models (API Key Required)

Set your API keys first:
```bash
export OPENAI_API_KEY="sk-..."          # For OpenAI
export ANTHROPIC_API_KEY="sk-ant-..."   # For Anthropic
```

Then use any cloud model:
```bash
# OpenAI GPT-4
"model": "openai/gpt-4o-mini"

# Anthropic Claude
"model": "anthropic/claude-3-5-haiku-latest"
```

## Bonus: Power an Agentic CLI (1 minute)

### For Codex CLI

```bash
# Configure environment
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
export ABSTRACTCORE_API_KEY="unused"

# Use Codex with any model
codex --model "ollama/qwen3-coder:30b" "Write a factorial function"
```

### For Crush CLI

```bash
# Same environment setup, then:
crush --model "anthropic/claude-3-5-haiku-latest" "Explain this code"
```

## Quick Validation Checklist

✅ Server started successfully
✅ Health check returns `{"status":"healthy"}`
✅ Models endpoint lists available models
✅ Basic generation works with curl or Python
✅ Your chosen model responds correctly

## What's Next?

**Ready for more?**
- [Full Configuration Guide](server-configuration.md) - All settings and options
- [Troubleshooting Guide](server-troubleshooting.md) - Fix common issues
- [Agentic CLI Integration](codex-cli-integration.md) - Advanced CLI setup

**Popular Models to Try:**
- `ollama/qwen3-coder:30b` - Best for coding
- `openai/gpt-4o-mini` - Reliable and fast
- `anthropic/claude-3-5-haiku-latest` - Intelligent and efficient

## Quick Commands Reference

```bash
# Start server
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl http://localhost:8000/providers

# Basic generation
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MODEL_NAME", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

Server has been configured. For further assistance, refer to the [Troubleshooting Guide](server-troubleshooting.md).