# AbstractCore Server - Universal LLM API Gateway

## Overview

The AbstractCore Server is a **unified API gateway** that provides OpenAI-compatible endpoints for ALL LLM providers supported by AbstractCore (OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace). This means any application that works with OpenAI's API can now work with Claude, Llama, or any local model - without changing a single line of code.

**Key Value**: Write once, run with any LLM provider.

## Why Use AbstractCore Server?

### 1. **Universal Compatibility**
- Use OpenAI's Python/JS/Ruby clients with Anthropic's Claude
- Connect ChatGPT-compatible apps to local Ollama models
- Switch between providers without changing client code

### 2. **Cost Optimization**
- Route simple queries to cheaper models
- Use local models for development, cloud for production
- Monitor costs across all providers in one place

### 3. **No Vendor Lock-in**
- Switch providers instantly via configuration
- Fallback chains (OpenAI → Anthropic → Ollama)
- Provider-agnostic tool calling

### 4. **Enterprise Features**
- Built-in retry logic and circuit breakers
- Real-time event streaming for monitoring
- Session management for stateful conversations
- Dynamic model discovery without hardcoding

## Installation

```bash
# Install with server support
pip install abstractcore[server]

# Or install all providers and server
pip install abstractcore[all]
```

## Quick Start

### 1. Start the Server

```bash
# Using the CLI
abstractcore-server

# With custom provider
abstractcore-server --provider anthropic --model claude-3-5-haiku-latest

# Development mode with auto-reload
abstractcore-server --reload --log-level debug

# Custom host and port
abstractcore-server --host localhost --port 3000
```

### 2. Use with OpenAI Client

Now ANY OpenAI-compatible client can use ANY provider:

```python
from openai import OpenAI

# Point to AbstractCore server instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Auth handled by AbstractCore
)

# Use Anthropic's Claude via OpenAI client!
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",  # Provider prefix
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.choices[0].message.content)

# Or use local Ollama models
response = client.chat.completions.create(
    model="ollama/llama3:8b",
    messages=[
        {"role": "user", "content": "Hello, Llama!"}
    ]
)
```

### 3. JavaScript/TypeScript Example

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'not-needed',
});

// Use ANY provider through OpenAI SDK
const completion = await openai.chat.completions.create({
  model: 'anthropic/claude-3-5-sonnet-latest',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

## API Endpoints

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/v1/chat/completions` | POST | Main chat endpoint (works with ALL providers) |
| `/v1/models` | GET | List all available models from all providers |
| `/v1/completions` | POST | Legacy completions endpoint |

### Provider Management

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/v1/providers` | GET | List all providers and their status |
| `/v1/providers/{name}/models` | GET | Get models for specific provider |
| `/v1/providers/test` | POST | Test provider configuration |

### Session Management

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/v1/sessions` | POST | Create conversation session |
| `/v1/sessions/{id}` | GET | Get session info |
| `/v1/sessions/{id}/chat` | POST | Chat with session context |
| `/v1/sessions/{id}` | DELETE | Delete session |

### Tool Management

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/v1/tools` | GET | List registered tools |
| `/v1/tools/register` | POST | Register new tool |

### Advanced Features

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/v1/generate/structured` | POST | Generate with Pydantic models |
| `/v1/events/stream` | GET | Real-time event stream (SSE) |
| `/v1/status` | GET | Server status and metrics |

## Advanced Usage

### Dynamic Provider Selection

The server can route requests to different providers based on the model name:

```python
# Format: provider/model
models = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-haiku-latest",
    "ollama/qwen3-coder:30b",
    "mlx/Qwen3-4B"
]

for model in models:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(f"{model}: {response.choices[0].message.content}")
```

### Streaming Responses

```python
# Streaming works with ALL providers
stream = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Tool Calling

Tools work uniformly across ALL providers, even those without native support:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="ollama/llama3:8b",  # Works even with local models!
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)
```

### Session-Based Conversations

Maintain conversation context across multiple requests:

```bash
# Create a session
curl -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model": "claude-3-5-haiku-latest",
    "system_prompt": "You are a helpful assistant"
  }'

# Returns: {"id": "session-123", ...}

# Chat with the session
curl -X POST http://localhost:8000/v1/sessions/session-123/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Remember that my name is Alice"}]
  }'

# Subsequent messages remember context
curl -X POST http://localhost:8000/v1/sessions/session-123/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is my name?"}]
  }'
# Response: "Your name is Alice"
```

### Real-Time Event Streaming

Monitor all LLM operations in real-time:

```javascript
// Connect to event stream
const eventSource = new EventSource('http://localhost:8000/v1/events/stream');

eventSource.addEventListener('generation_started', (e) => {
  const data = JSON.parse(e.data);
  console.log('Generation started:', data.model);
});

eventSource.addEventListener('tool_started', (e) => {
  const data = JSON.parse(e.data);
  console.log('Tool executing:', data.tool_name);
});

eventSource.addEventListener('generation_completed', (e) => {
  const data = JSON.parse(e.data);
  console.log('Tokens used:', data.tokens_output);
  console.log('Cost:', data.cost_usd);
});
```

### Structured Output Generation

Generate type-safe responses using Pydantic models:

```python
import httpx

response = httpx.post("http://localhost:8000/v1/generate/structured", json={
    "model": "gpt-4o-mini",
    "prompt": "Extract: John Doe is 25 years old and lives in New York",
    "response_model": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
    }
})

data = response.json()
# {"name": "John Doe", "age": 25, "city": "New York"}
```

## Configuration

### Environment Variables

```bash
# Default provider and model
export ABSTRACTCORE_DEFAULT_PROVIDER=anthropic
export ABSTRACTCORE_DEFAULT_MODEL=claude-3-5-haiku-latest

# Provider API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Local provider settings
export OLLAMA_HOST=http://localhost:11434
export LMSTUDIO_HOST=http://localhost:1234
```

### Provider Configuration

Test and configure providers dynamically:

```python
import httpx

# Test provider configuration
response = httpx.post("http://localhost:8000/v1/providers/test", json={
    "name": "openai",
    "api_key": "sk-...",
    "base_url": "https://api.openai.com/v1"  # Optional custom endpoint
})
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN pip install abstractcore[server]

EXPOSE 8000

CMD ["abstractcore-server", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t abstractcore-server .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  abstractcore-server
```

### Production Deployment

For production, use a process manager like systemd or supervisor:

```ini
# /etc/supervisor/conf.d/abstractcore.conf
[program:abstractcore]
command=/usr/local/bin/abstractcore-server --host 0.0.0.0 --port 8000
directory=/opt/abstractcore
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/abstractcore/server.log
environment=
    OPENAI_API_KEY="sk-...",
    ANTHROPIC_API_KEY="sk-ant-..."
```

### Reverse Proxy with nginx

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;

        # For SSE support
        proxy_set_header X-Accel-Buffering no;
        proxy_read_timeout 86400;
    }
}
```

## Performance & Monitoring

### Built-in Metrics

Access server metrics at `/v1/status`:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "total_requests": 1523,
  "active_sessions": 12,
  "providers": [
    {
      "name": "openai",
      "status": "healthy",
      "models": ["gpt-4o-mini", "gpt-4-turbo"]
    },
    {
      "name": "anthropic",
      "status": "healthy",
      "models": ["claude-3-5-haiku-latest"]
    }
  ]
}
```

### Prometheus Integration

(Coming soon - export metrics in Prometheus format)

## Troubleshooting

### Common Issues

1. **Provider not available**
   - Check API keys are set correctly
   - Verify network connectivity to provider
   - Check `/v1/providers` for provider status

2. **Model not found**
   - Use `/v1/models` to list available models
   - Format: `provider/model` for explicit routing
   - Check provider documentation for model names

3. **Streaming not working**
   - Ensure client supports SSE/streaming
   - Check reverse proxy configuration
   - Verify `stream=true` in request

4. **Session context lost**
   - Sessions are in-memory by default
   - Use session ID consistently
   - Check session exists with GET `/v1/sessions/{id}`

## API Compatibility

The server is designed to be a drop-in replacement for OpenAI's API:

| OpenAI SDK | Supported | Notes |
|------------|-----------|--------|
| Python | ✅ Full | All features work |
| JavaScript/Node | ✅ Full | All features work |
| Ruby | ✅ Full | All features work |
| Go | ✅ Full | All features work |
| .NET | ✅ Full | All features work |
| curl/HTTP | ✅ Full | Direct API access |

## Unique Features

### 1. **Universal Tool Calling**
Even providers without native tool support (Ollama, MLX) can use tools through AbstractCore's universal tool system.

### 2. **Dynamic Model Discovery**
No hardcoded model lists - the server discovers available models dynamically from each provider.

### 3. **Structured Output Everywhere**
Pydantic-validated structured output works with ALL providers, with automatic retry on validation failures.

### 4. **Real-Time Events**
Monitor costs, performance, and errors across all providers in real-time via SSE.

### 5. **Session Management**
Maintain conversation context with built-in session support, perfect for chatbots and assistants.

## Comparison with Alternatives

| Feature | AbstractCore Server | LiteLLM Proxy | OpenAI API |
|---------|-------------------|---------------|------------|
| OpenAI Compatibility | ✅ Full | ✅ Full | ✅ Native |
| Multiple Providers | ✅ All | ✅ Most | ❌ OpenAI only |
| Dynamic Model Discovery | ✅ Yes | ❌ Hardcoded | ❌ N/A |
| Universal Tools | ✅ Yes | ❌ Pass-through | ❌ N/A |
| Structured Output | ✅ All providers | ❌ Limited | ⚠️ Beta |
| Event Streaming | ✅ Yes | ❌ No | ❌ No |
| Session Management | ✅ Yes | ❌ No | ❌ No |
| Retry & Circuit Breakers | ✅ Built-in | ⚠️ Basic | ❌ No |
| Self-Hosted | ✅ Yes | ✅ Yes | ❌ No |
| Cost | ✅ Free | ✅ Free | 💰 Per token |

## Next Steps

1. **Install and Start**: `pip install abstractcore[server] && abstractcore-server`
2. **View API Docs**: Visit `http://localhost:8000/docs` for interactive API documentation
3. **Connect Your App**: Point any OpenAI client to `http://localhost:8000/v1`
4. **Monitor Events**: Connect to `/v1/events/stream` for real-time monitoring
5. **Explore Providers**: Try different models with `/v1/models`

## Summary

The AbstractCore Server transforms AbstractCore from a Python library into a **universal LLM infrastructure platform**. It provides:

- **One API for all LLMs** - OpenAI, Anthropic, Ollama, and more
- **Zero code changes** - Works with existing OpenAI clients
- **Enterprise features** - Retry, monitoring, sessions, tools
- **Cost optimization** - Route to cheapest/best provider
- **No vendor lock-in** - Switch providers instantly

This is the server that should have existed from the beginning - a truly universal LLM API gateway that just works.