# AbstractCore Server

Transform AbstractCore into an OpenAI-compatible API server. One server, all models, any client.

If you want a dedicated **single-model** `/v1` server (one provider/model per worker), see [Endpoint](endpoint.md).

## Interactive API docs (start here)

Visit while the server is running:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Quick Start

### Install and Run (2 minutes)

```bash
# Install
pip install "abstractcore[server]"

# Configure server auth and provider keys
export ABSTRACTCORE_SERVER_API_KEY="acore-server-secret"
export OPENAI_API_KEY="sk-..."

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
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or with Python:

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["ABSTRACTCORE_SERVER_API_KEY"])

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
export PORTKEY_API_KEY="pk_..."         # optional (Portkey)
export PORTKEY_CONFIG="pcfg_..."        # required for Portkey routing

# Server master key. Authenticated clients can use all server-configured providers.
export ABSTRACTCORE_SERVER_API_KEY="acore-server-secret"

# Local providers
export OLLAMA_BASE_URL="http://localhost:11434"          # (or legacy: OLLAMA_HOST)
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"
export VLLM_BASE_URL="http://localhost:8000/v1"

# Server bind (only used by `python -m abstractcore.server.app`)
export HOST="0.0.0.0"
export PORT="8000"

# Debug mode
export ABSTRACTCORE_DEBUG=true

# Dangerous (multi-tenant hazard): allow unload_after for providers that can unload shared server state (e.g. Ollama)
export ABSTRACTCORE_ALLOW_UNSAFE_UNLOAD_AFTER=1

# Server security controls (recommended)
#
# - Request-level base_url overrides are loopback-only by default.
#   URL entries match scheme + exact host + default/explicit port + path-segment prefix.
#   Bare entries match hostname globs, e.g. "*.example.com".
export ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST="https://api.openai.com,https://example.com/v1"
#
# - Remote URL fetches for attachments are blocked for private/loopback/link-local targets by default (SSRF protection).
#   To allow specific hosts/prefixes, use the same structured allowlist syntax:
export ABSTRACTCORE_SERVER_URL_FETCH_ALLOWLIST="https://www.berkshirehathaway.com"
#
# - Local file paths in HTTP requests are disabled by default (including @/path/to/file in message strings).
#   To allow local file paths safely, restrict them under a single directory:
export ABSTRACTCORE_SERVER_MEDIA_ROOT="/srv/abstractcore-media"
#
# - Unsafe escape hatch: allow arbitrary local file paths from HTTP requests (not recommended)
export ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1
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

Server auth:
- If `ABSTRACTCORE_SERVER_API_KEY` is configured, every non-health endpoint requires
  `Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY`. Authenticated clients can use all
  provider keys/endpoints configured on the server.
- If `ABSTRACTCORE_SERVER_API_KEY` is not configured, `Authorization: Bearer <provider-key>`
  may be used as a bring-your-own upstream provider key. That key is forwarded only to the
  requested provider and never unlocks server-configured provider keys.
- Health checks (`GET /health`) are always unauthenticated.

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
- `model` (required): Prefer `"provider/model-name"` (e.g., `"openai/gpt-4o-mini"`). If you pass a bare model name (no `/`), the server will best-effort auto-detect a provider.
- `messages` (required): Array of message objects
- `stream` (optional): Enable streaming responses
- `tools` (optional): Tools for function calling
- `agent_format` (optional, AbstractCore extension): Tool-call syntax output format for agentic clients (`"auto"|"openai"|"codex"|"qwen3"|"llama3"|"gemma"|"xml"|"passthrough"`). When omitted, the server auto-detects from user-agent + model heuristics.
- `api_key` (deprecated/disabled, AbstractCore extension): Provider API keys are no longer accepted in request bodies or query strings. Configure provider keys on the server, use `X-AbstractCore-Provider-API-Key` for a per-request provider override, or use `Authorization` as a provider key only when `ABSTRACTCORE_SERVER_API_KEY` is not configured.
- `base_url` (optional, AbstractCore extension): Override the provider endpoint (include `/v1` for OpenAI-compatible servers like LM Studio / vLLM / OpenRouter)
- `unload_after` (optional, AbstractCore extension): If `true`, calls `llm.unload_model(model)` after the request completes. Disabled for `ollama/*` unless `ABSTRACTCORE_ALLOW_UNSAFE_UNLOAD_AFTER=1`.
- `prompt_cache_key` (optional, AbstractCore extension): Best-effort prompt caching key (semantics depend on provider/backend). See `docs/prompt-caching.md`.
- `prompt_cache_retention` (optional, AbstractCore extension): Prompt cache retention policy (OpenAI: `"in_memory"` or `"24h"`; ignored by other providers). See `docs/prompt-caching.md`.
- `thinking` (optional, AbstractCore extension): Unified thinking/reasoning control (`null|"auto"|"on"|"off"|"none"` or `"low"|"medium"|"high"|"xhigh"` when supported). Note: `"none"` is treated as an alias for `"off"`.
- `temperature`, `max_tokens`, `top_p`: Standard LLM parameters

#### Thinking (AbstractCore extension)

The server forwards `thinking` to the underlying provider using AbstractCore’s unified thinking mapping (see [Generation Parameters](generation-parameters.md)).

Example (route to LM Studio + Qwen3.5, disable thinking):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio/qwen3.5-27b@q4_k_m",
    "base_url": "http://localhost:1234/v1",
    "messages": [{"role": "user", "content": "Compute 17*23 - 19*11. Reply with the integer only."}],
    "thinking": "none",
    "max_tokens": 64
  }'
```

Notes:
- For **Qwen3 / Qwen3.5 on LM Studio**, `thinking="none"` maps to LM Studio’s template variables (`enable_thinking` / `enableThinking`) plus a Qwen template “hard switch” fallback (empty `<think></think>`) when needed. This avoids injecting “reasoning effort” instructions into the system prompt.
- Not every backend supports per-effort budgets for `low|medium|high`; when unavailable, levels degrade to “thinking enabled”.

**Example with streaming:**

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["ABSTRACTCORE_SERVER_API_KEY"])

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

Security notes:
- Request-level `base_url` overrides are **loopback-only by default**. To allow additional
  origins or host globs, set `ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST`. URL entries are parsed
  and matched on scheme, exact host, effective port, and path-segment prefix.
- If the server has an environment provider key set (e.g. `OPENAI_API_KEY`) and you route to a **non-loopback** `base_url`, the request is refused unless the provider key was supplied explicitly with `X-AbstractCore-Provider-API-Key`, or with `Authorization` when server auth is disabled.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio/qwen/qwen3-4b-2507",
    "base_url": "http://localhost:1234/v1",
    "messages": [{"role": "user", "content": "Hello from a remote LM Studio endpoint"}]
  }'
```

#### Provider Authentication

Do not put provider keys in request bodies or query strings. Those fields are disabled because
they leak through logs, shell history, browser history, and reverse proxies.

```bash
# Preferred: configure provider keys on the server and authenticate to AbstractCore.
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

When `ABSTRACTCORE_SERVER_API_KEY` is not configured, `Authorization: Bearer <provider-key>` may
be used as an upstream provider key. Once server auth is enabled, `Authorization` is reserved for
the AbstractCore server key and is never forwarded upstream.

To override a single upstream provider while still using the server master key, send the provider
key in `X-AbstractCore-Provider-API-Key`. The override applies only to the requested provider:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -H "X-AbstractCore-Provider-API-Key: $ANTHROPIC_API_KEY" \
  -d '{
    "model": "anthropic/claude-haiku-4-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

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

Security note (HTTP server): local file paths are disabled by default (including `@/path/to/file` and `{"url": "/path/to/file"}`).
Use `http(s)` URLs or `data:` base64, or enable local paths via `ABSTRACTCORE_SERVER_MEDIA_ROOT` (safe) / `ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1` (unsafe).

#### Supported File Types

- **Images**: PNG, JPEG, GIF, WEBP, BMP, TIFF
- **Documents**: PDF, DOCX, XLSX, PPTX
- **Data/Text**: CSV, TSV, TXT, MD, JSON, XML
- **Size Limits**: 10MB per file, 32MB total per request

#### Method 1: @filename Syntax (AbstractCore Extension)

Simple syntax that works with all providers (requires local paths enabled via `ABSTRACTCORE_SERVER_MEDIA_ROOT` or `ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1`):

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

Note: local file paths require `ABSTRACTCORE_SERVER_MEDIA_ROOT` (safe) or `ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1` (unsafe) on the server.

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
import os
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["ABSTRACTCORE_SERVER_API_KEY"])

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
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["ABSTRACTCORE_SERVER_API_KEY"])

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

## Agentic CLI integration

AbstractCore Server is **OpenAI-compatible**. Most OpenAI-compatible CLIs/SDKs can be pointed at it by setting:

- `OPENAI_BASE_URL="http://localhost:8000/v1"` (or an equivalent flag)
- `OPENAI_API_KEY="unused"` (many clients require a non-empty key even for local servers)

### Tool calling interoperability

- The server **does not execute tools** (it always returns tool calls; your host/runtime executes them).
- It can emit tool calls either as structured `tool_calls` (OpenAI/Codex style) **or** as tagged content for clients that parse tool calls from assistant text.
- Control the output format with `agent_format` (request body, AbstractCore extension), or rely on auto-detection (user-agent + model heuristics).

Supported `agent_format` values: `auto`, `openai`, `codex`, `qwen3`, `llama3`, `gemma`, `xml`, `passthrough`.

### Codex CLI (example)

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

codex --model "ollama/qwen3-coder:30b" "Write a factorial function"
```

### Forcing a format (curl)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3:4b-instruct-2507-q4_K_M",
    "messages": [{"role": "user", "content": "Use the tool."}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather by city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ],
    "agent_format": "llama3"
  }'
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

RUN pip install "abstractcore[server]"

EXPOSE 8000

CMD ["uvicorn", "abstractcore.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Run:**
```bash
docker build -t abstractcore-server .
docker run -p 8000:8000 \
  -e ABSTRACTCORE_SERVER_API_KEY=$ABSTRACTCORE_SERVER_API_KEY \
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
      - ABSTRACTCORE_SERVER_API_KEY=${ABSTRACTCORE_SERVER_API_KEY}
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
export ABSTRACTCORE_SERVER_API_KEY="acore-server-secret"
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
