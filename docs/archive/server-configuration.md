# AbstractCore Server - Complete Configuration Reference

## Environment Variables

### Core Server Configuration

```bash
# Default Provider and Model
export ABSTRACTCORE_DEFAULT_PROVIDER=openai       # Default: "openai"
export ABSTRACTCORE_DEFAULT_MODEL=gpt-4o-mini    # Default: "gpt-4o-mini"

# Debug and Logging
export ABSTRACTCORE_DEBUG=true                   # Enable debug logging (default: false)

# Tool Execution Behavior
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=true   # Auto-execute tools (default: true)
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false  # Return tool calls for external handling

# Tool Call Tag Rewriting (for Agentic CLIs)
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=qwen3  # Default format
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3 # For Crush CLI compatibility
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml    # For Gemini CLI compatibility
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS="START,END" # Custom format
```

### Provider API Keys

```bash
# Cloud Provider Keys
export OPENAI_API_KEY="sk-..."                   # OpenAI models
export ANTHROPIC_API_KEY="sk-ant-..."           # Anthropic Claude models
export GOOGLE_API_KEY="..."                      # Google Gemini models
export COHERE_API_KEY="..."                      # Cohere models

# Local Model Configuration
export OLLAMA_HOST="http://localhost:11434"      # Ollama server location
export LMSTUDIO_HOST="http://localhost:1234"     # LMStudio server location
```

## Server Endpoints Reference

### Core Endpoints

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/` | GET | Server info and examples | None |
| `/health` | GET | Health check | None |
| `/docs` | GET | Interactive API documentation | None |
| `/providers` | GET | List provider status | None |

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description | Key Parameters |
|----------|--------|-------------|----------------|
| `/v1/chat/completions` | POST | Standard OpenAI chat endpoint | `model`, `messages`, `tools`, `stream` |
| `/{provider}/v1/chat/completions` | POST | Provider-specific routing | Same as above, provider in path |
| `/v1/models` | GET | List all available models | Query: `type` (optional) |
| `/v1/responses` | POST | Codex CLI preferred endpoint | `prompt`, `tools`, `max_tokens` |
| `/v1/messages` | POST | Anthropic Messages API | `messages`, `model`, `max_tokens` |
| `/v1/embeddings` | POST | Create embeddings | `input`, `model` |
| `/v1/completions` | POST | Legacy completions API | `prompt`, `model`, `max_tokens` |

### Advanced Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/engines` | GET | List available engines (legacy) |
| `/v1/capabilities` | GET | Server capabilities info |
| `/v1/mcp/servers` | POST/GET/DELETE | MCP server management |

## Tool Call Tag Rewriting

### Format Specifications

| Format | Start Tag | End Tag | Use Case |
|--------|-----------|---------|----------|
| `qwen3` | `<|tool_call|>` | `</|tool_call|>` | Qwen models, default |
| `openai` | None | None | Native JSON tool calls |
| `llama3` | `<function_call>` | `</function_call>` | LLaMA models, Crush CLI |
| `xml` | `<tool_call>` | `</tool_call>` | XML-based, Gemini CLI |
| `gemma` | ` ```tool_code` | ` ``` ` | Gemma models |
| Custom | User-defined | User-defined | Any format: `"START,END"` |

### Configuration Examples

#### Server-Level Configuration (for Agentic CLIs)

```bash
# Configure server for Codex CLI (default qwen3 format)
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Configure server for Crush CLI
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Configure server for custom format
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS="[TOOL],[/TOOL]"
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000
```

#### Per-Request Override (when you control the client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Override server defaults with request parameter
response = client.chat.completions.create(
    model="ollama/qwen3-coder:30b",
    messages=[{"role": "user", "content": "Get weather"}],
    tools=[{"type": "function", "function": {"name": "get_weather"}}],
    tool_call_tags="llama3",  # Override server default
    execute_tools=False        # Don't execute, just return tool calls
)
```

```javascript
// JavaScript example
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model: 'ollama/qwen3-coder:30b',
    messages: [{role: 'user', content: 'List files'}],
    tools: [{type: 'function', function: {name: 'list_files'}}],
    tool_call_tags: 'xml',  // Override to XML format
    stream: true            // Enable streaming
  })
});
```

## Advanced Parameters

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | Required | Model identifier (e.g., "openai/gpt-4o-mini") |
| `messages` | array | Required | Conversation messages |
| `tools` | array | None | Available tools for the model |
| `tool_call_tags` | string | Server default | Tool call format override |
| `execute_tools` | boolean | true | Whether to execute tools automatically |
| `stream` | boolean | false | Enable streaming responses |
| `temperature` | float | 0.7 | Response randomness (0-2) |
| `max_tokens` | integer | Model default | Maximum response length |
| `top_p` | float | 1.0 | Nucleus sampling parameter |
| `frequency_penalty` | float | 0.0 | Repetition penalty (-2 to 2) |
| `presence_penalty` | float | 0.0 | Topic penalty (-2 to 2) |

### Streaming Configuration

```python
# Python streaming example
stream = client.chat.completions.create(
    model="ollama/qwen3-coder:30b",
    messages=[{"role": "user", "content": "Write a long story"}],
    stream=True,
    tool_call_tags="llama3"  # Tool calls work in streaming mode
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Debug Logging Configuration

### Enable Comprehensive Logging

```bash
# Enable debug mode
export ABSTRACTCORE_DEBUG=true

# Start server with debug logging
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# With additional uvicorn debugging
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000 --log-level debug
```

### Log File Locations

| Log Type | Location | Format | Contents |
|----------|----------|--------|----------|
| Structured Logs | `logs/abstractllm_TIMESTAMP.log` | JSON | Events, errors, metadata |
| Request Payloads | `logs/YYYYMMDD-payloads.jsonl` | JSONL | Full request bodies |
| Verbatim I/O | `logs/verbatim_TIMESTAMP.jsonl` | JSONL | Complete prompts/responses |

### Log Analysis Commands

```bash
# Find all errors
grep '"level": "error"' logs/abstractllm_*.log | jq -r '.event + " | " + .error'

# Calculate average latency by provider
cat logs/verbatim_*.jsonl | jq -r 'select(.provider=="ollama") | .metadata.latency_ms' | \
  awk '{sum+=$1; count++} END {print "Average:", sum/count "ms"}'

# Track token usage
cat logs/verbatim_*.jsonl | jq -r '.metadata.tokens | .input + .output' | \
  awk '{sum+=$1} END {print "Total tokens:", sum}'

# Monitor specific model performance
grep '"model": "qwen3-coder:30b"' logs/verbatim_*.jsonl | \
  jq '.metadata.latency_ms' | sort -n
```

## Deployment Configurations

### Production Docker Setup

```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install abstractcore[server,all-providers]

# Configure environment
ENV ABSTRACTCORE_DEFAULT_PROVIDER=openai
ENV ABSTRACTCORE_DEFAULT_MODEL=gpt-4o-mini
ENV ABSTRACTCORE_DEBUG=false

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "abstractllm.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose Example

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
      - ABSTRACTCORE_DEBUG=false
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: abstractcore-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: abstractcore
  template:
    metadata:
      labels:
        app: abstractcore
    spec:
      containers:
      - name: abstractcore
        image: abstractcore-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: ABSTRACTCORE_DEFAULT_PROVIDER
          value: "openai"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
```

## Performance Optimization

### Server Tuning

```bash
# Multiple workers for production
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn for better performance
gunicorn abstractllm.server.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000

# Development with auto-reload
uvicorn abstractllm.server.app:app --reload --log-level debug
```

### Model-Specific Optimization

```python
# Configure for high-throughput
response = client.chat.completions.create(
    model="ollama/qwen3-coder:30b",
    messages=[{"role": "user", "content": "Quick response"}],
    temperature=0.3,      # Lower for consistency
    max_tokens=500,       # Limit response size
    top_p=0.9,           # Focused sampling
    stream=True          # Stream for perceived speed
)
```

## Security Configuration

### Basic Access Control

```bash
# Set API key requirement
export ABSTRACTCORE_API_KEY="your-secret-key"

# Clients must provide the key
curl -H "Authorization: Bearer your-secret-key" \
  http://localhost:8000/v1/chat/completions
```

### HTTPS with SSL

```bash
# Generate self-signed certificate for testing
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with SSL
uvicorn abstractllm.server.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-keyfile=./key.pem \
  --ssl-certfile=./cert.pem
```

### Rate Limiting

```python
# In your deployment, use a reverse proxy like nginx:
# /etc/nginx/sites-available/abstractcore

limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    listen 443 ssl;

    location /v1/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://localhost:8000;
    }
}
```

## Related Documentation

- [Quick Start Guide](server-quickstart.md) - Get running in 5 minutes
- [Troubleshooting Guide](server-troubleshooting.md) - Debug common issues
- [Tool Call Tag Rewriting](tool-syntax-rewriting.md) - Deep dive into tool formats
- [Agentic CLI Integration](codex-cli-integration.md) - CLI-specific setup

---

AbstractCore Server Configuration