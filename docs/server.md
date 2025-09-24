# AbstractCore Server - Universal LLM Gateway

## What It Does

Turn AbstractCore into a **universal API server** that works with ANY LLM provider through simple, clean endpoints. One server, all models, any language.

## Quick Start

```bash
# Install and start (uses simplified server by default)
pip install abstractcore[server]
abstractcore-server

# Now ANY OpenAI client can use ANY provider!
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Use Claude with OpenAI's client!
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Super Simple Usage

### 1. URL Chat (No JSON needed!)
```bash
# Just URL parameters - works in any browser
http://localhost:8000/chat?message=Hello%20world
http://localhost:8000/chat?message=Write%20code&provider=anthropic

# Streaming responses
http://localhost:8000/chat?message=Tell%20a%20story&stream=true
```

### 2. POST Chat (Simple JSON)
```bash
# Regular response
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "provider": "anthropic"}'

# Streaming response
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "stream": true}'
```

### 3. OpenAI Compatible
```bash
# Regular response
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming response (OpenAI format)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-5-haiku-latest",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

## All Endpoints (Just 8!)

| Endpoint | Method | What It Does |
|----------|---------|-------------|
| `/` | GET | Server info + quick examples |
| `/chat` | GET/POST | Universal chat endpoint |
| `/models` | GET | List all available models |
| `/providers` | GET | Show provider status |
| `/health` | GET | Health check |
| `/test` | GET | Test all providers quickly |
| `/v1/chat/completions` | POST | OpenAI compatible endpoint |
| `/docs` | GET | Interactive API documentation |

That's it! No complexity, no confusion.

## Examples That Actually Work

### Test All Providers Instantly
```bash
# See which providers are working
curl http://localhost:8000/test
```

### List Models by Type
```bash
# Chat models only
curl "http://localhost:8000/models?type=chat"

# Embedding models only
curl "http://localhost:8000/models?type=embedding"

# From specific provider
curl "http://localhost:8000/models?provider=ollama&type=chat"
```

### Different Creativity Levels
```bash
# Focused (temperature=0.1)
curl "http://localhost:8000/chat?message=Write%20code&temperature=0.1"

# Creative (temperature=0.9)
curl "http://localhost:8000/chat?message=Tell%20a%20story&temperature=0.9"
```

## Server Features

```bash
# Start the server
abstractcore-server
```

**What you get:**
- 8 clean, focused endpoints
- 280 lines of code
- All providers supported
- OpenAI compatibility
- Interactive documentation

## Configuration

```bash
# Different provider
abstractcore-server --provider anthropic --model claude-3-5-haiku-latest

# Development mode
abstractcore-server --reload --log-level debug

# Custom port
abstractcore-server --port 3000
```

## Environment Variables

```bash
# Set defaults
export ABSTRACTCORE_DEFAULT_PROVIDER=anthropic
export ABSTRACTCORE_DEFAULT_MODEL=claude-3-5-haiku-latest

# Provider API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

## Language Examples

### Python
```python
import requests

response = requests.get("http://localhost:8000/chat", params={
    "message": "What is Python?",
    "provider": "openai"
})
print(response.json()["response"])
```

### JavaScript
```javascript
// Regular response
fetch('http://localhost:8000/chat?message=Hello&provider=anthropic')
  .then(r => r.json())
  .then(data => console.log(data.response));

// Streaming response
fetch('http://localhost:8000/chat?message=Tell%20a%20story&stream=true')
  .then(response => {
    const reader = response.body.getReader();
    function readStream() {
      return reader.read().then(({done, value}) => {
        if (done) return;
        const text = new TextDecoder().decode(value);
        console.log(text);
        return readStream();
      });
    }
    return readStream();
  });
```

### curl
```bash
curl "http://localhost:8000/chat?message=Hello&provider=ollama&model=llama3:8b"
```

## Response Format

All endpoints return consistent, simple responses:

```json
{
  "message": "Hello!",
  "response": "Hello! How can I help you today?",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 12,
    "total_tokens": 20
  }
}
```

## Deploy Anywhere

### Docker
```dockerfile
FROM python:3.9-slim
RUN pip install abstractcore[server]
EXPOSE 8000
CMD ["abstractcore-server"]
```

### Railway/Vercel/Fly.io
Just deploy with `pip install abstractcore[server]` and `abstractcore-server`

### Local Development
```bash
abstractcore-server --reload --log-level debug
```

## Why Choose AbstractCore Server?

✅ **Simple** - 8 endpoints, not 20+
✅ **Fast** - 280 lines of clean code
✅ **Universal** - Works with all providers
✅ **Compatible** - OpenAI clients work
✅ **Examples** - Actually working examples
✅ **Reliable** - Less code = fewer bugs

## Getting Started

1. **Install**: `pip install abstractcore[server]`
2. **Start**: `abstractcore-server`
3. **Test**: Visit `http://localhost:8000`
4. **Use**: Point any OpenAI client to your server

That's it! No complexity, no confusion, just a working universal LLM API server.

## Summary

The AbstractCore server transforms any machine into a **universal LLM gateway** that works with:
- ✅ Any LLM provider (OpenAI, Anthropic, Ollama, etc.)
- ✅ Any programming language (Python, JS, Go, etc.)
- ✅ Any OpenAI-compatible client
- ✅ Simple URL parameters or JSON
- ✅ Zero configuration needed

**One server, all models, any language.** 🚀