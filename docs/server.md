# AbstractCore Server

Transform AbstractCore into an OpenAI-compatible API server. One server, all models, any client.

## Quick Start

### Install and Run (2 minutes)

```bash
# Install
pip install abstractcore[server]

# Start server
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/health
# Response: {"status":"healthy"}
```

### First Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or with Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

---

## Configuration

### Environment Variables

```bash
# Provider API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Local providers
export OLLAMA_HOST="http://localhost:11434"
export LMSTUDIO_HOST="http://localhost:1234"

# Default settings
export ABSTRACTCORE_DEFAULT_PROVIDER=openai
export ABSTRACTCORE_DEFAULT_MODEL=gpt-4o-mini

# Debug mode
export ABSTRACTCORE_DEBUG=true
```

### Startup Options

```bash
# Development with auto-reload
uvicorn abstractllm.server.app:app --reload

# Production with multiple workers
uvicorn abstractllm.server.app:app --workers 4

# Custom port
uvicorn abstractllm.server.app:app --port 3000
```

---

## API Endpoints

### Chat Completions

**Endpoint:** `POST /v1/chat/completions`

Standard OpenAI-compatible endpoint. Works with all providers.

**Request:**
```json
{
  "model": "provider/model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

**Key Parameters:**
- `model` (required): Format `"provider/model-name"` (e.g., `"openai/gpt-4o-mini"`)
- `messages` (required): Array of message objects
- `stream` (optional): Enable streaming responses
- `tools` (optional): Tools for function calling
- `temperature`, `max_tokens`, `top_p`: Standard LLM parameters

**Example with streaming:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

stream = client.chat.completions.create(
    model="ollama/qwen3-coder:30b",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

### Embeddings

**Endpoint:** `POST /v1/embeddings`

Generate embedding vectors for semantic search, RAG, and similarity analysis.

**Request:**
```json
{
  "input": "Text to embed",
  "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
}
```

**Supported Providers:**
- **HuggingFace**: Local models with ONNX acceleration
- **Ollama**: `ollama/granite-embedding:278m`, etc.
- **LMStudio**: Any loaded embedding model

**Batch Embedding:**
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["text 1", "text 2", "text 3"],
    "model": "ollama/granite-embedding:278m"
  }'
```

---

### Model Discovery

**Endpoint:** `GET /v1/models`

List all available models from configured providers.

**Query Parameters:**
- `provider`: Filter by provider (e.g., `ollama`, `openai`)
- `type`: Filter by type (`text-generation` or `text-embedding`)

**Examples:**
```bash
# All models
curl http://localhost:8000/v1/models

# Ollama models only
curl http://localhost:8000/v1/models?provider=ollama

# Embedding models only
curl http://localhost:8000/v1/models?type=text-embedding

# Ollama embeddings
curl http://localhost:8000/v1/models?provider=ollama&type=text-embedding
```

---

### Provider Status

**Endpoint:** `GET /providers`

List all available providers and their status.

**Response:**
```json
{
  "providers": [
    {
      "name": "ollama",
      "type": "llm",
      "model_count": 15,
      "status": "available"
    }
  ]
}
```

---

### Health Check

**Endpoint:** `GET /health`

Server health check for monitoring.

**Response:** `{"status": "healthy"}`

---

## Agentic CLI Integration

Use AbstractCore server with agentic CLI tools like Codex, Crush, and Gemini CLI.

### Codex CLI

```bash
# Setup
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
export ABSTRACTCORE_API_KEY="unused"

# Use with any model
codex --model "ollama/qwen3-coder:30b" "Write a factorial function"
```

### Crush CLI (LLaMA3 format)

```bash
# Configure server
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Configure CLI
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Use
crush --model "anthropic/claude-3-5-haiku-latest" "Explain this code"
```

### Gemini CLI (XML format)

```bash
# Configure server
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Configure CLI
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Use
gemini-cli --model "ollama/qwen3-coder:30b" "Review this project"
```

### Tool Call Format Configuration

```bash
# Set format for your CLI
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=qwen3    # Codex CLI
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3   # Crush CLI
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml      # Gemini CLI

# Control tool execution
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=true   # Server executes
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false  # Return to client
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

RUN pip install abstractcore[server]

ENV ABSTRACTCORE_DEFAULT_PROVIDER=openai
ENV ABSTRACTCORE_DEFAULT_MODEL=gpt-4o-mini

EXPOSE 8000

CMD ["uvicorn", "abstractllm.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Run:**
```bash
docker build -t abstractcore-server .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY abstractcore-server
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
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

### Production with Gunicorn

```bash
pip install gunicorn

gunicorn abstractllm.server.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000
```

---

## Debug and Monitoring

### Enable Debug Logging

```bash
export ABSTRACTCORE_DEBUG=true
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000
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
  awk '{sum+=$1} END {print "Total:", sum}'

# Monitor specific model
grep '"model": "qwen3-coder:30b"' logs/verbatim_*.jsonl
```

### Interactive Documentation

Visit while server is running:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Common Patterns

### Multi-Provider Fallback

```python
import requests

providers = [
    "ollama/qwen3-coder:30b",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-haiku-latest"
]

def generate_with_fallback(prompt):
    for model in providers:
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            continue
    raise Exception("All providers failed")
```

### Local Model Gateway

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

---

## Troubleshooting

### Server Won't Start

```bash
# Check port availability
lsof -i :8000

# Use different port
uvicorn abstractllm.server.app:app --port 3000
```

### No Models Available

```bash
# Check providers
curl http://localhost:8000/providers

# Check API keys
echo $OPENAI_API_KEY

# Start Ollama
ollama serve
ollama list
```

### Authentication Errors

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Restart server after setting keys
```

---

## Why AbstractCore Server?

✅ **Universal**: One API for all providers  
✅ **OpenAI Compatible**: Drop-in replacement  
✅ **Simple**: Clean, focused endpoints  
✅ **Fast**: Lightweight, high-performance  
✅ **Debuggable**: Comprehensive logging  
✅ **CLI Ready**: Codex, Gemini CLI, Crush support  
✅ **Production Ready**: Docker, multi-worker, health checks  

---

## Related Documentation

- **[Getting Started](getting-started.md)** - Core library quick start
- **[Architecture](architecture.md)** - System architecture including server
- **[Python API Reference](api-reference.md)** - Core library API
- **[Embeddings Guide](embeddings.md)** - Embeddings deep dive
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

**AbstractCore Server** - One server, all models, any client. 🚀
