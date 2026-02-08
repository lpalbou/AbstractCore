# AbstractCore Server

Transform AbstractCore into an OpenAI-compatible API server. One server, all models, any client.

## Interactive API docs (start here)

Visit while the server is running:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Quick Start

### Install and Run (2 minutes)

```bash
# Install
pip install "abstractcore[server]"

# Start server
python -m abstractcore.server.app

# Or with uvicorn directly
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

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
    model="anthropic/claude-haiku-4-5",
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
export OPENROUTER_API_KEY="sk-or-..."

# Local providers
export OLLAMA_BASE_URL="http://localhost:11434"          # (or legacy: OLLAMA_HOST)
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"
export VLLM_BASE_URL="http://localhost:8000/v1"

# Default settings
export ABSTRACTCORE_DEFAULT_PROVIDER=openai
export ABSTRACTCORE_DEFAULT_MODEL=gpt-4o-mini

# Debug mode
export ABSTRACTCORE_DEBUG=true

# Dangerous (multi-tenant hazard): allow unload_after for providers that can unload shared server state (e.g. Ollama)
export ABSTRACTCORE_ALLOW_UNSAFE_UNLOAD_AFTER=1
```

### Startup Options

```bash
# Using AbstractCore's built-in CLI
python -m abstractcore.server.app --help                    # View all options
python -m abstractcore.server.app --debug                   # Debug mode
python -m abstractcore.server.app --host 127.0.0.1 --port 8080  # Custom host/port
python -m abstractcore.server.app --debug --port 8001       # Debug on custom port

# Using uvicorn directly
uvicorn abstractcore.server.app:app --reload                # Development with auto-reload
uvicorn abstractcore.server.app:app --workers 4             # Production with multiple workers
uvicorn abstractcore.server.app:app --port 3000             # Custom port
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
- `api_key` (optional, AbstractCore extension): Provider API key for per-request authentication. Falls back to environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`)
- `base_url` (optional, AbstractCore extension): Override the provider endpoint (include `/v1` for OpenAI-compatible servers like LM Studio / vLLM / OpenRouter)
- `unload_after` (optional, AbstractCore extension): If `true`, calls `llm.unload_model(model)` after the request completes. Disabled for `ollama/*` unless `ABSTRACTCORE_ALLOW_UNSAFE_UNLOAD_AFTER=1`.
- `thinking` (optional, AbstractCore extension): Unified thinking/reasoning control (`null|"auto"|"on"|"off"` or `"low"|"medium"|"high"` when supported)
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

#### Provider `base_url` override (AbstractCore extension)

Route a provider to a specific endpoint (useful for remote OpenAI-compatible servers):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio/qwen/qwen3-4b-2507",
    "base_url": "http://localhost:1234/v1",
    "messages": [{"role": "user", "content": "Hello from a remote LM Studio endpoint"}]
  }'
```

#### Per-request `api_key` (AbstractCore extension)

Pass API keys directly in requests (useful for multi-tenant scenarios or OpenRouter):

```bash
# OpenRouter with per-request API key
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openrouter/anthropic/claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}],
    "api_key": "sk-or-v1-your-openrouter-key"
  }'

# OpenAI-compatible endpoint with custom auth
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-compatible/my-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "api_key": "your-api-key",
    "base_url": "https://my-custom-endpoint.com/v1"
  }'
```

If `api_key` is not provided, AbstractCore falls back to environment variables.

### Media generation endpoints (optional)

AbstractCore Server can optionally expose OpenAI-compatible **image generation** and **audio** endpoints.

Important notes:
- These are **interoperability-first** endpoints (return `b64_json` or raw bytes), not an artifact-first durability contract.
- If the required plugin/backend is not available, the server returns `501` with actionable messaging.

#### Images (generate/edit) — requires `abstractvision`

Endpoints:
- `POST /v1/images/generations`
- `POST /v1/images/edits`

Install:
```bash
pip install "abstractcore[server]"
pip install abstractvision
```

#### Audio (STT/TTS) — requires an audio/voice capability plugin (typically `abstractvoice`)

Endpoints:
- `POST /v1/audio/transcriptions` (multipart; `file=...`)
- `POST /v1/audio/speech` (json; `input=...`, optional `voice`, optional `format`)

Install:
```bash
pip install "abstractcore[server]"
pip install abstractvoice
```

Notes:
- `/v1/audio/transcriptions` requires `python-multipart` for form parsing (included in the server extra).

Examples:

```bash
# Speech-to-text (STT)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@speech.wav" \
  -F "language=en"

# Text-to-speech (TTS)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello!","format":"wav"}' \
  --output hello.wav
```

If you want to “ask a model about an audio file”, prefer one of:
- Run STT first (`/v1/audio/transcriptions`) then send the transcript to `POST /v1/chat/completions`, or
- Configure the server’s default audio strategy (`config.audio.strategy`) to enable STT fallback for audio attachments, then attach audio in chat requests.

### Multimodal Requests (Images, Documents, Files)

AbstractCore server supports comprehensive file attachments using OpenAI-compatible multimodal message format, plus AbstractCore's convenient `@filename` syntax.

#### Supported File Types

- **Images**: PNG, JPEG, GIF, WEBP, BMP, TIFF
- **Documents**: PDF, DOCX, XLSX, PPTX
- **Data/Text**: CSV, TSV, TXT, MD, JSON, XML
- **Size Limits**: 10MB per file, 32MB total per request

#### Method 1: @filename Syntax (AbstractCore Extension)

Simple syntax that works with all providers:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [
      {"role": "user", "content": "What is in this document? @/path/to/report.pdf"}
    ]
  }'
```

#### Method 2: OpenAI Vision API Format (Image URLs)

Standard OpenAI format for images:

```json
{
  "model": "anthropic/claude-haiku-4-5",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.jpg"
          }
        }
      ]
    }
  ]
}
```

**Base64 Images:**
```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
  }
}
```

#### Method 3: OpenAI File Format (Forward-Compatible)

AbstractCore supports OpenAI's planned file format with simplified structure (consistent with image_url):

**File URL Format (Recommended - Same Pattern as image_url):**
```json
{
  "model": "ollama/qwen3:4b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this document"},
        {
          "type": "file",
          "file_url": {
            "url": "https://example.com/documents/report.pdf"
          }
        }
      ]
    }
  ]
}
```

**Local File Path:**
```json
{
  "type": "file",
  "file_url": {
    "url": "/Users/username/documents/data.csv"
  }
}
```

**Base64 Data URL:**
```json
{
  "type": "file",
  "file_url": {
    "url": "data:application/pdf;base64,JVBERi0xLjQKMSAwIG9iago<PAovVHlwZS..."
  }
}
```

**Filename Extraction:**
- **URLs/Paths**: Extracted automatically (`/path/file.pdf` → `file.pdf`)
- **Base64**: Generated from MIME type (`data:application/pdf;base64,...` → `document.pdf`)

#### Mixed Content Example

Combine text, images, and documents in a single request:

```json
{
  "model": "openai/gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Compare this chart with the data in the spreadsheet"},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS..."}
        },
        {
          "type": "file",
          "file_url": {
            "url": "https://example.com/data/sales_data.xlsx"
          }
        }
      ]
    }
  ]
}
```

#### Python Client Examples

**Using OpenAI Client:**
```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Method 1: @filename syntax
response = client.chat.completions.create(
    model="anthropic/claude-haiku-4-5",
    messages=[{"role": "user", "content": "Summarize @document.pdf"}]
)

# Method 2: File URL (HTTP/HTTPS)
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the key findings?"},
            {
                "type": "file",
                "file_url": {
                    "url": "https://example.com/documents/report.pdf"
                }
            }
        ]
    }]
)

# Method 3: Local file path
response = client.chat.completions.create(
    model="anthropic/claude-haiku-4-5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this local document"},
            {
                "type": "file",
                "file_url": {
                    "url": "/Users/username/documents/report.pdf"
                }
            }
        ]
    }]
)

# Method 4: Base64 data URL
with open("report.pdf", "rb") as f:
    file_data = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="lmstudio/qwen/qwen3-next-80b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the key findings?"},
            {
                "type": "file",
                "file_url": {
                    "url": f"data:application/pdf;base64,{file_data}"
                }
            }
        ]
    }]
)
```

**Universal Provider Support:**
```python
# Same syntax works across all providers
providers_models = [
    "openai/gpt-4o",
    "anthropic/claude-haiku-4-5",
    "ollama/qwen2.5vl:7b",
    "lmstudio/qwen/qwen2.5-vl-7b"
]

for model in providers_models:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Analyze @data.csv and @chart.png"}]
    )
    print(f"{model}: {response.choices[0].message.content[:100]}...")
```

---

### OpenAI Responses API

**Endpoint:** `POST /v1/responses`

AbstractCore implements an OpenAI-compatible Responses-style API, including `input_file` support.

#### Why Use /v1/responses?

- **OpenAI Compatible**: Drop-in replacement for OpenAI's Responses API
- **Native File Support**: `input_file` type designed specifically for document attachments
- **Cleaner API**: Explicit separation between text (`input_text`) and files (`input_file`)
- **Backward Compatible**: Existing `messages` format still works alongside new `input` format
- **Optional Streaming**: Streaming opt-in with `"stream": true` (defaults to `false`)

#### Request Format

**OpenAI Responses API Format (Recommended):**
```json
{
  "model": "gpt-4o",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Analyze this document"},
        {"type": "input_file", "file_url": "https://example.com/report.pdf"}
      ]
    }
  ],
  "stream": false,
  "max_tokens": 2000,
  "temperature": 0.7
}
```

**Legacy Format (Still Supported):**
```json
{
  "model": "openai/gpt-4",
  "messages": [
    {"role": "user", "content": "Tell me a story"}
  ],
  "stream": false
}
```

#### Automatic Format Detection

The server automatically detects which format you're using:
- **OpenAI Format**: Presence of `input` field → converts to internal format
- **Legacy Format**: Presence of `messages` field → processes directly
- **Error**: Missing both fields → returns 400 error with clear message

#### Examples

**Simple Text Request:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio/qwen/qwen3-next-80b",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "What is Python?"}
        ]
      }
    ]
  }'
```

**File Analysis:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Analyze the letter and summarize key points"},
          {"type": "input_file", "file_url": "https://www.berkshirehathaway.com/letters/2024ltr.pdf"}
        ]
      }
    ]
  }'
```

**Multiple Files:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-haiku-4-5",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Compare these documents"},
          {"type": "input_file", "file_url": "https://example.com/report1.pdf"},
          {"type": "input_file", "file_url": "https://example.com/report2.pdf"},
          {"type": "input_file", "file_url": "https://example.com/chart.png"}
        ]
      }
    ],
    "max_tokens": 2000
  }'
```

**Streaming Response:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Summarize this document"},
          {"type": "input_file", "file_url": "https://example.com/document.pdf"}
        ]
      }
    ],
    "stream": true
  }' --no-buffer
```

#### Supported Media Types

All file types supported via URL, local path, or base64:

- **Documents**: PDF, DOCX, XLSX, PPTX
- **Data Files**: CSV, TSV, JSON, XML
- **Text Files**: TXT, MD
- **Images**: PNG, JPEG, GIF, WEBP, BMP, TIFF
- **Size Limits**: 10MB per file, 32MB total per request

**Source Options:**
```json
// HTTP/HTTPS URL
{"type": "input_file", "file_url": "https://example.com/report.pdf"}

// Local file path
{"type": "input_file", "file_url": "/path/to/document.xlsx"}

// Base64 data URL
{"type": "input_file", "file_url": "data:application/pdf;base64,JVBERi0x..."}
```

#### Python Client Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Direct request to /v1/responses endpoint
import requests

response = requests.post(
    "http://localhost:8000/v1/responses",
    json={
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze this document"},
                    {"type": "input_file", "file_url": "https://example.com/report.pdf"}
                ]
            }
        ]
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
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
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# Configure CLI
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Use
crush --model "anthropic/claude-haiku-4-5" "Explain this code"
```

### Gemini CLI (XML format)

```bash
# Configure server
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

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

RUN pip install "abstractcore[server]"

ENV ABSTRACTCORE_DEFAULT_PROVIDER=openai
ENV ABSTRACTCORE_DEFAULT_MODEL=gpt-4o-mini

EXPOSE 8000

CMD ["uvicorn", "abstractcore.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
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

gunicorn abstractcore.server.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000
```

---

## Debug and Monitoring

### Enable Debug Mode

Debug mode provides comprehensive logging and detailed error reporting for troubleshooting API issues.

```bash
# Method 1: Using command line flag (recommended)
python -m abstractcore.server.app --debug

# Method 2: Using environment variable
export ABSTRACTCORE_DEBUG=true
python -m abstractcore.server.app

# Method 3: With uvicorn directly
export ABSTRACTCORE_DEBUG=true
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

### Debug Features

**Enhanced Error Reporting:**
- **Before**: Uninformative "422 Unprocessable Entity" messages
- **After**: Detailed field validation errors with request body capture

**Example Debug Output:**
```json
🔴 Request Validation Error (422) | method=POST | error_count=2 | errors=[
  {"field": "body -> model", "message": "Field required", "type": "missing"},
  {"field": "body -> messages", "message": "Field required", "type": "missing"}
] | client=127.0.0.1

📋 Request Body (Validation Error) | body={"invalid": "data"}
```

**Request/Response Tracking:**
- Full HTTP request details (method, URL, headers, client IP)
- Response status codes and processing times
- Structured JSON logging for machine processing

**Log Files:**
- `logs/abstractcore_TIMESTAMP.log` - Structured events
- `logs/YYYYMMDD-payloads.jsonl` - Full request bodies
- `logs/verbatim_TIMESTAMP.jsonl` - Complete I/O

**Useful Commands:**
```bash
# Find errors
grep '"level": "error"' logs/abstractcore_*.log

# Track token usage
cat logs/verbatim_*.jsonl | jq '.metadata.tokens | .input + .output' | \
  awk '{sum+=$1} END {print "Total:", sum}'

# Monitor specific model
grep '"model": "qwen3-coder:30b"' logs/verbatim_*.jsonl
```

## Common Patterns

### Multi-Provider Fallback

```python
import requests

providers = [
    "ollama/qwen3-coder:30b",
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4-5"
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
uvicorn abstractcore.server.app:app --port 3000
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

- **Universal**: One API for all providers  
- **OpenAI Compatible**: Drop-in replacement  
- **Simple**: Clean, focused endpoints  
- **Fast**: Lightweight, high-performance  
- **Debuggable**: Comprehensive logging  
- **CLI Ready**: Codex, Gemini CLI, Crush support  
- **Production Ready**: Docker, multi-worker, health checks  

---

## Related Documentation

- **[Getting Started](getting-started.md)** - Core library quick start
- **[Architecture](architecture.md)** - System architecture including server
- **[Python API Reference](api-reference.md)** - Core library API
- **[Embeddings Guide](embeddings.md)** - Embeddings deep dive
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

**AbstractCore Server** - One server, all models, any client.
