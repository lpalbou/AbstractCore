# AbstractCore Server - Universal LLM Gateway

Transform AbstractCore into a universal OpenAI-compatible API server that works with ANY LLM provider. One server, all models, any client.

## Table of Contents

- [Quick Start](#quick-start-5-minutes)
- [Configuration](#configuration)
- [Use Cases](#use-cases)
- [Agentic CLI Integration](#agentic-cli-integration)
- [Deployment](#deployment)
- [Related Documentation](#related-documentation)

---

## Quick Start (5 Minutes)

### Prerequisites

- Python 3.9+ (`python --version`)
- pip (`python -m pip --version`)
- Port 8000 available (`lsof -i :8000` should show nothing)

### Step 1: Install (30 seconds)

```bash
pip install abstractcore[server]
```

### Step 2: Start Server (10 seconds)

```bash
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 3: Quick Test (20 seconds)

```bash
# Check server health
curl http://localhost:8000/health
# Expected: {"status":"healthy"}

# List available models
curl http://localhost:8000/v1/models
```

### Step 4: Your First Generation (1 minute)

**Using curl:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Write a haiku about coding"}]
  }'
```

**Using Python:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a haiku about coding"}]
)
print(response.choices[0].message.content)
```

✅ **Success!** You now have a universal LLM gateway running.

---

## Configuration

### Environment Variables

#### Core Server Settings

```bash
# Default provider and model
export ABSTRACTCORE_DEFAULT_PROVIDER=openai
export ABSTRACTCORE_DEFAULT_MODEL=gpt-4o-mini

# Debug mode (logs all requests/responses)
export ABSTRACTCORE_DEBUG=true
```

#### Provider API Keys

```bash
# Cloud providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Local providers
export OLLAMA_HOST="http://localhost:11434"
export LMSTUDIO_HOST="http://localhost:1234"
```

#### Tool Call Configuration (for Agentic CLIs)

```bash
# Set tool call format for CLI compatibility
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=qwen3    # Default, Codex CLI
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3   # Crush CLI
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml      # Gemini CLI

# Control tool execution
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=true   # Server executes tools
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false  # Return tools to client
```

### Server Startup Options

```bash
# Development mode with auto-reload
uvicorn abstractllm.server.app:app --reload --log-level debug

# Production with multiple workers
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000 --workers 4

# Custom port
uvicorn abstractllm.server.app:app --port 3000

# With SSL
uvicorn abstractllm.server.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-keyfile=./key.pem \
  --ssl-certfile=./cert.pem
```

---

## Use Cases

### 1. OpenAI Client Compatibility

Use any OpenAI-compatible client with any provider:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Use Claude with OpenAI's client!
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

### 2. Local Model Gateway

Run powerful models locally for privacy:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-coder:30b

# Use via AbstractCore server
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

### 3. Multi-Provider Fallback

```python
import requests

providers = [
    ("ollama/qwen3-coder:30b", "Free local model"),
    ("openai/gpt-4o-mini", "Cheap cloud fallback"),
    ("anthropic/claude-3-5-haiku-latest", "Quality fallback")
]

def generate_with_fallback(prompt):
    for model, description in providers:
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            if response.status_code == 200:
                print(f"✓ Using {description}")
                return response.json()
        except Exception as e:
            print(f"✗ {description} failed: {e}")
            continue
    raise Exception("All providers failed")

result = generate_with_fallback("Hello world")
```

### 4. Streaming Responses

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

stream = client.chat.completions.create(
    model="ollama/qwen3-coder:30b",
    messages=[{"role": "user", "content": "Write a long story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 5. Embeddings Generation

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Machine learning is fascinating",
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
  }'
```

---

## Agentic CLI Integration

AbstractCore server is optimized for integration with agentic CLI tools like Codex, Crush, and Gemini CLI.

### Codex CLI Setup

```bash
# Set environment variables
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
export ABSTRACTCORE_API_KEY="unused"

# Use Codex with any model
codex --model "ollama/qwen3-coder:30b" "Write a factorial function"
```

### Crush CLI Setup

```bash
# Configure server for Crush (llama3 format)
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Configure CLI
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Use Crush
crush --model "anthropic/claude-3-5-haiku-latest" "Explain this code"
```

### Gemini CLI Setup

```bash
# Configure server for Gemini (xml format)
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Configure CLI
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Use Gemini CLI
gemini-cli --model "ollama/qwen3-coder:30b" "Review this project"
```

### Recommended Models for CLIs

**For Coding:**
- `ollama/qwen3-coder:30b` - Excellent code generation
- `ollama/deepseek-coder:33b` - Strong reasoning
- `lmstudio` - Whatever model you've loaded

**For General Tasks:**
- `anthropic/claude-3-5-haiku-latest` - Fast and intelligent
- `openai/gpt-4o-mini` - Reliable and cost-effective
- `ollama/qwen3:4b-instruct-2507-q4_K_M` - Local and private

---

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

RUN pip install abstractcore[server]

ENV ABSTRACTCORE_DEFAULT_PROVIDER=openai
ENV ABSTRACTCORE_DEFAULT_MODEL=gpt-4o-mini

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "abstractllm.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Build and Run:**
```bash
docker build -t abstractcore-server .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  abstractcore-server
```

### Docker Compose

```yaml
version: '3.8'

services:
  abstractcore:
    image: abstractcore-server:latest
    ports:
      - "8000:8000"
    environment:
      - ABSTRACTCORE_DEFAULT_PROVIDER=anthropic
      - ABSTRACTCORE_DEFAULT_MODEL=claude-3-5-haiku-latest
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

**Start:**
```bash
docker-compose up -d
```

### Production with Gunicorn

```bash
# Install Gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn abstractllm.server.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000
```

### Cloud Deployment

**Railway/Vercel/Fly.io:**
1. Install: `pip install abstractcore[server]`
2. Start command: `uvicorn abstractllm.server.app:app --host 0.0.0.0 --port $PORT`
3. Set environment variables via platform UI

---

## Debug Logging

Enable comprehensive logging for troubleshooting:

```bash
# Enable debug mode
export ABSTRACTCORE_DEBUG=true

# Start server
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Logs are created in logs/ directory
tail -f logs/abstractllm_*.log
```

**Log Files:**
- `logs/abstractllm_TIMESTAMP.log` - Structured events
- `logs/YYYYMMDD-payloads.jsonl` - Full request bodies
- `logs/verbatim_TIMESTAMP.jsonl` - Complete I/O

**Useful Commands:**
```bash
# Find errors
grep '"level": "error"' logs/abstractllm_*.log

# Track token usage
cat logs/verbatim_*.jsonl | jq '.metadata.tokens | .input + .output' | \
  awk '{sum+=$1} END {print "Total tokens:", sum}'

# Monitor specific model
grep '"model": "qwen3-coder:30b"' logs/verbatim_*.jsonl
```

---

## REST API Endpoints

AbstractCore server provides these HTTP endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Standard OpenAI chat endpoint |
| `/{provider}/v1/chat/completions` | POST | Provider-specific chat endpoint |
| `/v1/responses` | POST | Simplified endpoint for CLIs |
| `/v1/embeddings` | POST | Create embeddings |
| `/v1/models` | GET | List available models |
| `/providers` | GET | List providers and their status |
| `/docs` | GET | Interactive API documentation |

**Complete REST API reference:** [Server API Reference](server-api-reference.md)

---

## Quick Validation

Before using in production, verify:

```bash
# 1. Server starts
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# 2. Health check passes
curl http://localhost:8000/health
# Expected: {"status":"healthy"}

# 3. Models available
curl http://localhost:8000/v1/models | jq '.data | length'
# Expected: > 0

# 4. Basic generation works
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'

# 5. Streaming works
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Count to 5"}], "stream": true}'
```

---

## Why Choose AbstractCore Server?

✅ **Universal**: Works with all providers through one API  
✅ **OpenAI Compatible**: Drop-in replacement for OpenAI API  
✅ **Simple**: Clean, focused endpoints  
✅ **Fast**: Lightweight implementation  
✅ **Debuggable**: Comprehensive logging  
✅ **CLI Ready**: Full Codex/Gemini CLI/Crush support  
✅ **Production Ready**: Docker, multi-worker, health checks  

---

## Related Documentation

**Server Documentation:**
- **[Server API Reference](server-api-reference.md)** - Complete REST API documentation for all HTTP endpoints

**Core Library Documentation:**
- **[Python API Reference](api-reference.md)** - AbstractCore library functions and classes
- **[Getting Started](getting-started.md)** - Core library quick start
- **[Embeddings Guide](embeddings.md)** - Embeddings deep dive and use cases
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Prerequisites](prerequisites.md)** - Provider setup instructions

---

**AbstractCore Server** - One server, all models, any language, any tool format, any agentic CLI. 🚀
