# Server Module

## Purpose and Architecture Position

The Server Module provides a production-ready FastAPI REST server that exposes AbstractCore's capabilities through OpenAI-compatible HTTP endpoints. It acts as a universal LLM gateway with comprehensive media processing, multi-agent tool call syntax support, and provider-agnostic model access.

**Architecture Position**: The Server Module sits at the top of the AbstractCore stack, providing HTTP/REST access to all underlying capabilities. It transforms AbstractCore's Python API into web-accessible endpoints for integration with any application, regardless of programming language.

## Quick Reference

### API Endpoints Quick Reference

| Endpoint | Method | Purpose | Key Parameters |
|----------|--------|---------|----------------|
| `/health` | GET | Server health check | - |
| `/v1/models` | GET | List available models | `type` (optional) |
| `/providers` | GET | List providers with metadata | - |
| `/v1/chat/completions` | POST | Chat completions | `model`, `messages`, `stream` |
| `/v1/embeddings` | POST | Generate embeddings | `model`, `input` |
| `/v1/responses` | POST | OpenAI Responses API | `model`, `input` |
| `/v1/images/generations` | POST | Image generation (optional) | `prompt`, `model` |
| `/v1/images/edits` | POST | Image editing (optional) | `prompt`, `image`, `mask` |
| `/{provider}/v1/chat/completions` | POST | Provider-specific endpoint | `model` (no prefix) |

### Common Request Patterns

| Pattern | Endpoint | Key Fields | Example |
|---------|----------|------------|---------|
| **Simple Chat** | `/v1/chat/completions` | `model`, `messages` | Text conversation |
| **Streaming** | `/v1/chat/completions` | `stream: true` | Real-time responses |
| **Vision** | `/v1/chat/completions` | `content: [text, image_url]` | Image analysis |
| **Image Generation** | `/v1/images/generations` | `prompt` | Create images (optional) |
| **Image Editing** | `/v1/images/edits` | `prompt`, `image` | Edit images (optional) |
| **Documents** | `/v1/chat/completions` | `content: [text, file_url]` | PDF/CSV processing |
| **Tools** | `/v1/chat/completions` | `tools`, `tool_choice` | Function calling |
| **Embeddings** | `/v1/embeddings` | `model`, `input` | Text embeddings |

### Media Support

| Media Type | MIME Types | Max Size | Processing |
|------------|-----------|----------|------------|
| **Images** | `image/*` | 10MB | Vision models |
| **Documents** | `application/pdf`, `docx`, `xlsx` | 10MB | Text extraction |
| **Data** | `text/csv`, `application/json` | 10MB | Rendering |
| **Text** | `text/plain`, `text/html` | 10MB | Direct |

## Common Tasks

- **How do I start the server?** → See [Local Development](#local-development)
- **How do I list models?** → See [List Models](#2-list-models)
- **How do I make a chat request?** → See [Chat Completions](#4-chat-completions)
- **How do I process images?** → See [Vision API](#5-vision-api-image-analysis)
- **How do I process documents?** → See [Document Processing](#6-document-processing)
- **How do I generate images?** → See [Image Generation](#image-generation-optional)
- **How do I enable streaming?** → See [Streaming Response](#streaming-response)
- **How do I use function calling?** → See [With Function Calling](#with-function-calling)
- **How do I deploy to production?** → See [Production Deployment](#production-deployment)
- **How do I troubleshoot errors?** → See [Troubleshooting](#troubleshooting)

```
┌─────────────────────────────────────────────┐
│         Server Module (server/)             │  ← You are here
│   (FastAPI REST API, OpenAI-compatible)     │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│       Processing Module (processing/)       │
│  (Specialized NLP operations & workflows)   │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│        Core Layer (core/, providers/)       │
│  (LLM providers, factory, base interfaces)  │
└─────────────────────────────────────────────┘
```

## Component Structure

The Server Module consists of a single, comprehensive FastAPI application:

```
server/
├── __init__.py                    # Module exports
└── app.py                         # FastAPI server (2000+ lines)
```

### Design Philosophy

- **OpenAI Compatibility**: 100% compatible with OpenAI Chat Completions API
- **Universal Gateway**: Access any AbstractCore provider through unified API
- **Media Processing**: Built-in support for images, documents, PDFs, CSV, Excel, etc.
- **Tool Call Flexibility**: Auto-detection and conversion of tool call syntaxes
- **Clean Delegation**: Minimal server logic, delegates to AbstractCore
- **Production-Ready**: Comprehensive error handling, logging, CORS support
- **Zero Configuration**: Works out-of-the-box with sensible defaults

---

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Purpose**: Server health and version information.

**Response**:
```json
{
  "status": "healthy",
  "version": "2.5.2",
  "timestamp": "2025-11-06T12:00:00Z"
}
```

**Usage**:
```bash
curl http://localhost:8000/health
```

---

### 2. List Models

**Endpoint**: `GET /v1/models`

**Purpose**: List all available models across all providers.

**Query Parameters**:
- `input_type` (optional): Filter by input capability (`text`, `image`, `audio`, `video`)
- `output_type` (optional): Filter by output capability (`text`, `embeddings`)
- `provider` (optional): Filter by specific provider

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/gpt-4o-mini",
      "object": "model",
      "created": 1234567890,
      "owned_by": "openai"
    },
    {
      "id": "ollama/llama3.2:3b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "ollama"
    }
  ]
}
```

**Usage**:
```bash
# List all models
curl http://localhost:8000/v1/models

# List models that can analyze images (vision models)
curl http://localhost:8000/v1/models?input_type=image

# List embedding models
curl http://localhost:8000/v1/models?output_type=embeddings

# List text-only models that generate text
curl http://localhost:8000/v1/models?input_type=text&output_type=text

# List Ollama vision models
curl http://localhost:8000/v1/models?provider=ollama&input_type=image

# Combine multiple filters
curl http://localhost:8000/v1/models?provider=openai&input_type=image&output_type=text
```

**Python Client**:
```python
import requests

# List all models
response = requests.get("http://localhost:8000/v1/models")
models = response.json()["data"]

# Filter by capabilities
response = requests.get("http://localhost:8000/v1/models?input_type=image")
vision_models = response.json()["data"]

response = requests.get("http://localhost:8000/v1/models?output_type=embeddings")
embedding_models = response.json()["data"]
```

---

### 3. List Providers

**Endpoint**: `GET /providers`

**Purpose**: List all registered providers with their metadata and available models.

**Response**:
```json
{
  "providers": [
    {
      "name": "openai",
      "description": "OpenAI Language Models",
      "requires_api_key": true,
      "api_key_env": "OPENAI_API_KEY",
      "supported_features": [
        "streaming",
        "function_calling",
        "vision",
        "structured_output"
      ],
      "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", ...],
      "installation": "pip install openai"
    },
    {
      "name": "ollama",
      "description": "Local LLM Serving with Ollama",
      "requires_api_key": false,
      "supported_features": [
        "streaming",
        "function_calling",
        "structured_output"
      ],
      "models": ["llama3.2:3b", "qwen3:4b", "gemma3:1b-it-qat", ...],
      "installation": "Install from https://ollama.com/"
    }
  ],
  "total_providers": 6,
  "total_models": 137,
  "registry_version": "2.0"
}
```

**Usage**:
```bash
curl http://localhost:8000/providers
```

---

### 4. Chat Completions

**Endpoint**: `POST /v1/chat/completions`

**Purpose**: Generate chat completions using any AbstractCore provider.

**Request Schema**:
```json
{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "top_p": 1.0,
  "stream": false,
  "tools": [...],
  "tool_choice": "auto",
  "stop": ["END"],
  "seed": 12345,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "agent_format": "auto",
  "api_key": null,
  "base_url": null
}
```

**Key Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | **required** | Model ID in `provider/model` format |
| `messages` | array | **required** | Conversation history |
| `temperature` | float | 0.7 | Sampling temperature (0-2) |
| `max_tokens` | integer | null | Max tokens to generate |
| `top_p` | float | 1.0 | Nucleus sampling (0-1) |
| `stream` | boolean | false | Enable streaming responses |
| `tools` | array | null | Available function tools |
| `tool_choice` | string/object | "auto" | Tool calling strategy |
| `agent_format` | string | null | Tool syntax format |
| `api_key` | string | null | Provider API key (falls back to env vars) |
| `base_url` | string | null | Custom API endpoint URL |

**Response (Non-Streaming)**:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "openai/gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

**Response (Streaming)**:
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"openai/gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"openai/gpt-4o-mini","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

data: [DONE]
```

**Usage Examples**:

**Basic Text Chat**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7
  }'
```

#### Streaming response
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "ollama/llama3.2:3b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

#### With function calling
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

**Python Client**:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

**OpenAI Python Client**:
```python
from openai import OpenAI

# Point to AbstractCore server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # AbstractCore manages API keys
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

**With Per-Request API Key** (OpenRouter, OpenAI-compatible, etc.):
```bash
# Pass API key directly in request (useful for multi-tenant scenarios)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openrouter/anthropic/claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}],
    "api_key": "sk-or-v1-your-openrouter-key",
    "temperature": 0.7
  }'
```

```python
# Python example with per-request API key
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "openai-compatible/my-model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "api_key": "your-api-key",
        "base_url": "https://my-custom-endpoint.com/v1"
    }
)
```

Note: If `api_key` is not provided, AbstractCore falls back to environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`).

---

### 5. Vision API (Image Analysis)

**Endpoint**: `POST /v1/chat/completions` (multimodal messages)

**Purpose**: Analyze images using vision-capable models.

**Request with Image**:
```json
{
  "model": "ollama/qwen2.5vl:7b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
          }
        }
      ]
    }
  ]
}
```

**Supported Image Formats**:
- PNG, JPEG, GIF, WEBP, BMP, TIFF
- Base64 data URLs: `data:image/jpeg;base64,...`
- HTTP(S) URLs: `https://example.com/image.jpg`
- Size limit: 10MB per image

**Usage**:
```bash
# Using base64 encoded image
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen2.5vl:7b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}
        }
      ]
    }]
  }'

# Using HTTP URL
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see?"},
        {
          "type": "image_url",
          "image_url": {"url": "https://example.com/photo.jpg"}
        }
      ]
    }]
  }'
```

**Python with Image**:
```python
import requests
import base64

# Read and encode image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama/qwen2.5vl:7b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

---

### Image Generation (optional)

AbstractCore Server can optionally expose OpenAI-compatible image endpoints:
- `POST /v1/images/generations`
- `POST /v1/images/edits`

These return `501` only if the server can't infer a backend (e.g. no `model` in the request and no relevant env vars are set).

**Backends (env vars)**:

- **Default**: `auto` (picks a backend per request)
  - If `model` looks like a GGUF file/path, uses stable-diffusion.cpp (`sdcpp`)
  - If `model` looks like a Hugging Face id/path (e.g. `runwayml/stable-diffusion-v1-5`), uses Diffusers
  - If `ABSTRACTCORE_VISION_UPSTREAM_BASE_URL` is set and `model` is a bare name (e.g. `dall-e-3`), uses the proxy

- **Proxy**: set `ABSTRACTCORE_VISION_BACKEND=openai_compatible_proxy` (proxy to an upstream OpenAI-compatible image server)
  - `ABSTRACTCORE_VISION_UPSTREAM_BASE_URL` (required) — base URL (include `/v1`)
  - `ABSTRACTCORE_VISION_UPSTREAM_API_KEY` (optional)
  - `ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID` (optional)

- **Local Diffusers**: set `ABSTRACTCORE_VISION_BACKEND=diffusers`
  - `ABSTRACTCORE_VISION_MODEL_ID` (required) — local model id/path (Diffusers)
  - `ABSTRACTCORE_VISION_DEVICE` (optional, default `auto`; chooses `cuda`/`mps` when available, else `cpu`)
  - `ABSTRACTCORE_VISION_TORCH_DTYPE` (optional, e.g. `float16`; for very large models you typically need `float16` or you may run out of memory)
  - `ABSTRACTCORE_VISION_ALLOW_DOWNLOAD` (optional, default true; set to `0` for cache-only/offline)
  - Note: if you set `ABSTRACTCORE_VISION_DEVICE=mps` or `cuda`, your PyTorch must actually support it (`torch.backends.mps.is_available()` / `torch.cuda.is_available()`).
  - Install: `pip install "abstractcore[server]"`

- **Local stable-diffusion.cpp**: set `ABSTRACTCORE_VISION_BACKEND=sdcpp`
  - Recommended (pip-only): `pip install "abstractcore[server]"`
  - Alternative (external executable): install `sd-cli`: https://github.com/leejet/stable-diffusion.cpp/releases
  - `ABSTRACTCORE_VISION_SDCPP_BIN` (optional, default `sd-cli`)
  - Configure either:
    - **Full model**: `ABSTRACTCORE_VISION_SDCPP_MODEL`
    - **Component mode**: `ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL` (and optional components like `ABSTRACTCORE_VISION_SDCPP_VAE`, `ABSTRACTCORE_VISION_SDCPP_LLM`, ...)
  - `ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS` (optional) — extra `sd-cli` flags (e.g. `--diffusion-fa --sampling-method euler --flow-shift 3`)

**Text-to-image (JSON)**:
```bash
curl http://localhost:8000/v1/images/generations \\
  -H "Content-Type: application/json" \\
  -d '{"prompt":"a red fox in snow","response_format":"b64_json"}'
```

**Image edit (multipart/form-data)**:
```bash
curl http://localhost:8000/v1/images/edits \\
  -F "prompt=make it watercolor" \\
  -F "image=@./input.png"
```

Notes:
- The server returns `b64_json` outputs, matching the OpenAI image API shape.
- Endpoints delegate to AbstractVision internally; install it in the same env as the server (and prefer `python -m uvicorn ...`):
  - `pip install "abstractcore[server]"`
  - or manage it directly: `pip install abstractvision`

---

### 6. Document Processing

**Endpoint**: `POST /v1/chat/completions` (with document attachments)

**Purpose**: Process PDF, CSV, Excel, JSON, XML, and text documents.

**Supported Document Types**:
- **Documents**: PDF, DOCX, XLSX, PPTX
- **Data**: CSV, TSV, JSON, XML
- **Text**: TXT, MD, HTML
- Size limit: 10MB per file, 32MB total

**Request with CSV File**:
```json
{
  "model": "lmstudio/qwen3-next-80b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Analyze this CSV and calculate total sales"},
      {
        "type": "image_url",
        "image_url": {
          "url": "data:text/csv;base64,RGF0ZSxQcm9kdWN0LFNhbGVzCjIwMjQtMDEtMDEsUHJvZHVjdCBBLDEwMDAwCjIwMjQtMDEtMDIsUHJvZHVjdCBCLDE1MDAwCjIwMjQtMDEtMDMsUHJvZHVjdCBDLDI1MDAw"
        }
      }
    ]
  }]
}
```

**Request with PDF**:
```json
{
  "model": "openai/gpt-4o",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Summarize this PDF report"},
      {
        "type": "image_url",
        "image_url": {
          "url": "data:application/pdf;base64,JVBERi0xLjQKJdPr6eEK..."
        }
      }
    ]
  }]
}
```

**Python with CSV**:
```python
import requests
import base64

# Read and encode CSV
with open("sales.csv", "rb") as f:
    csv_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "lmstudio/qwen3-next-80b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Calculate total sales from this data"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:text/csv;base64,{csv_base64}"}
                }
            ]
        }]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

---

### 7. Embeddings

**Endpoint**: `POST /v1/embeddings`

**Purpose**: Generate text embeddings using embedding models.

**Request**:
```json
{
  "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
  "input": "The quick brown fox jumps over the lazy dog",
  "encoding_format": "float",
  "dimensions": 0
}
```

**Request (Batch)**:
```json
{
  "model": "ollama/granite-embedding:278m",
  "input": [
    "First text to embed",
    "Second text to embed",
    "Third text to embed"
  ]
}
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

**Usage**:
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

**Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
        "input": "The quick brown fox jumps over the lazy dog"
    }
)

embedding = response.json()["data"][0]["embedding"]
print(f"Embedding dimension: {len(embedding)}")
```

---

### 8. OpenAI Responses API

**Endpoint**: `POST /v1/responses`

**Purpose**: OpenAI Responses API compatible endpoint with file URL support.

**Request**:
```json
{
  "model": "openai/gpt-4o",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Analyze this file"},
        {"type": "input_file", "file_url": "https://example.com/data.csv"}
      ]
    }
  ],
  "stream": false
}
```

**Note**: This endpoint accepts both OpenAI Responses format (with `input` field) and legacy format (with `messages` field). Format is auto-detected.

---

### 9. Provider-Specific Endpoints

**Endpoint**: `POST /{provider}/v1/chat/completions`

**Purpose**: Route requests to specific provider without provider prefix in model name.

**Example**:
```bash
# Standard endpoint with provider prefix
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model": "openai/gpt-4o-mini", "messages": [...]}'

# Provider-specific endpoint without prefix
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -d '{"model": "gpt-4o-mini", "messages": [...]}'
```

**Supported Providers**:
- `/openai/v1/chat/completions`
- `/anthropic/v1/chat/completions`
- `/ollama/v1/chat/completions`
- `/lmstudio/v1/chat/completions`
- `/huggingface/v1/chat/completions`
- `/mlx/v1/chat/completions`

---

## Media Processing Features

### Automatic Format Detection

The server automatically detects and processes:

| MIME Type | Extensions | Processing |
|-----------|------------|------------|
| `image/*` | png, jpg, gif, webp, bmp, tiff | Vision models |
| `application/pdf` | pdf | Text extraction |
| `text/csv` | csv | Table rendering |
| `application/vnd.ms-excel` | xls, xlsx | Table rendering |
| `application/json` | json | JSON formatting |
| `text/plain` | txt, md | Text rendering |
| `text/html` | html | Text extraction |

### Data URL Support

Supports both base64 data URLs and HTTP(S) URLs:

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

```json
{
  "type": "image_url",
  "image_url": {
    "url": "https://example.com/image.png"
  }
}
```

### Size Limits

- Per file: 10MB
- Total per request: 32MB
- Automatic validation and error messages

---

## Tool Call Syntax Support

The server includes advanced tool call syntax conversion:

### Supported Formats

| Format | Description | Detection |
|--------|-------------|-----------|
| `openai` | OpenAI function calling | Default for OpenAI models |
| `codex` | Codex-style function calls | Auto-detect from User-Agent |
| `qwen3` | Qwen3 tool syntax | Auto-detect from model name |
| `llama3` | LLaMA3 tool syntax | Auto-detect from model name |
| `passthrough` | No conversion | Explicit opt-in |
| `auto` | Automatic detection | Recommended (default) |

### Agent Format Parameter

Control tool syntax with `agent_format` parameter:

```json
{
  "model": "anthropic/claude-3-opus-20240229",
  "messages": [...],
  "tools": [...],
  "agent_format": "auto"
}
```

### Auto-Detection Rules

1. **User-Agent Detection**: Codex clients get Codex format
2. **Model Name Detection**: qwen3 → Qwen3, llama3 → LLaMA3
3. **Provider Default**: OpenAI models → OpenAI format
4. **Fallback**: OpenAI format (most compatible)

---

## Deployment

### Local Development

**Start Server**:
```bash
# Standard mode
python -m abstractcore.server

# Debug mode (verbose logging)
python -m abstractcore.server --debug

# Custom host/port
python -m abstractcore.server --host 0.0.0.0 --port 8080

# All options
python -m abstractcore.server --host 0.0.0.0 --port 8080 --debug
```

**Environment Variables**:
```bash
# Provider API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Debug mode
export ABSTRACTCORE_DEBUG="true"

# Start server
python -m abstractcore.server
```

### Production Deployment

**Using Uvicorn Directly**:
```bash
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Using Gunicorn + Uvicorn Workers**:
```bash
gunicorn abstractcore.server.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000
```

**Docker Deployment**:
```dockerfile
FROM python:3.12-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "abstractcore.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t abstractcore-server .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  abstractcore-server
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming support
        proxy_buffering off;
        proxy_cache off;
    }
}
```

---

## Configuration

### API Keys

The server automatically reads API keys from environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or use AbstractCore's configuration system:
```bash
abstractcore --set-api-key openai sk-...
abstractcore --set-api-key anthropic sk-ant-...
```

### Logging

Configure logging via environment or code:

**Environment**:
```bash
export ABSTRACTCORE_DEBUG="true"
python -m abstractcore.server
```

**Code** (`abstractcore/server/app.py`):
```python
from abstractcore.utils.structured_logging import configure_logging
import logging

configure_logging(
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    log_dir="logs",
    console_json=False,
    file_json=True
)
```

### CORS

CORS is enabled by default for all origins. To restrict:

```python
# In app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid JSON or missing fields |
| 422 | Validation Error | Invalid parameter values |
| 404 | Not Found | Unknown endpoint or model |
| 500 | Server Error | Internal processing failure |
| 503 | Service Unavailable | Provider unavailable |

### Error Response Format

```json
{
  "error": {
    "message": "Request validation failed",
    "type": "validation_error",
    "details": [
      {
        "field": "model",
        "message": "field required",
        "type": "value_error.missing"
      }
    ]
  }
}
```

### Common Errors

**Invalid Model Format**:
```json
{
  "error": {
    "message": "Invalid model format. Use 'provider/model' format (e.g., 'openai/gpt-4o-mini')",
    "type": "validation_error"
  }
}
```

**Provider Not Available**:
```json
{
  "error": {
    "message": "Provider 'xyz' not found. Available providers: openai, anthropic, ollama, ...",
    "type": "provider_error"
  }
}
```

**Media Processing Error**:
```json
{
  "error": {
    "message": "File size exceeds maximum limit of 10MB",
    "type": "media_processing_error"
  }
}
```

---

## Client Integration

### Python (requests)

```python
import requests

class AbstractCoreClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def chat(self, model, messages, **kwargs):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": model, "messages": messages, **kwargs}
        )
        response.raise_for_status()
        return response.json()

    def embed(self, model, input_text):
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": model, "input": input_text}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = AbstractCoreClient()
result = client.chat(
    "openai/gpt-4o-mini",
    [{"role": "user", "content": "Hello!"}]
)
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### JavaScript/TypeScript

```typescript
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

async function chat(model: string, messages: ChatMessage[]) {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model, messages})
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  return response.json();
}

// Usage
const result = await chat('openai/gpt-4o-mini', [
  {role: 'user', content: 'Hello!'}
]);
console.log(result.choices[0].message.content);
```

### cURL

```bash
# Simple chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{"model":"ollama/llama3.2:3b","messages":[{"role":"user","content":"Hello!"}],"stream":true}'

# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"huggingface/sentence-transformers/all-MiniLM-L6-v2","input":"Hello world"}'
```

---

## Best Practices

### 1. Use Provider/Model Format

Always specify provider prefix:
```json
{"model": "openai/gpt-4o-mini"}  // Correct
{"model": "gpt-4o-mini"}         // May fail
```

### 2. Handle Streaming Properly

For streaming responses, process server-sent events:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={"model": "ollama/llama3.2:3b", "messages": [...], "stream": True},
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                break
            import json
            chunk = json.loads(data)
            print(chunk['choices'][0]['delta'].get('content', ''), end='')
```

### 3. Optimize Media Attachments

- Compress images before sending
- Use appropriate image resolution (don't send 10MP when 1MP suffices)
- Consider HTTP URLs instead of base64 for large files

### 4. Implement Retry Logic

```python
import time
import requests

def chat_with_retry(model, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={"model": model, "messages": messages},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
```

### 5. Monitor Token Usage

Track usage for cost management:
```python
result = client.chat(model, messages)
usage = result["usage"]
print(f"Tokens: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
```

---

## Performance Considerations

### Concurrent Requests

The server handles concurrent requests efficiently:

```python
from concurrent.futures import ThreadPoolExecutor

def process_request(text):
    return client.chat("openai/gpt-4o-mini", [{"role": "user", "content": text}])

texts = ["Query 1", "Query 2", "Query 3", ...]

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_request, texts))
```

### Streaming for Long Responses

Use streaming for better UX with long responses:
```python
response = client.chat(
    "openai/gpt-4o-mini",
    messages=[...],
    stream=True,
    max_tokens=4096
)
```

### Batch Embeddings

Process multiple texts in one request:
```python
client.embed(
    "huggingface/sentence-transformers/all-MiniLM-L6-v2",
    ["Text 1", "Text 2", "Text 3", ...]
)
```

---

## Monitoring and Observability

### Structured Logging

All requests are logged with structured data:

```json
{
  "timestamp": "2025-11-06T12:00:00Z",
  "level": "INFO",
  "message": "HTTP Request",
  "method": "POST",
  "url": "/v1/chat/completions",
  "status_code": 200,
  "process_time_ms": 1234.56,
  "model": "openai/gpt-4o-mini",
  "provider": "openai",
  "usage": {"total_tokens": 150}
}
```

### Debug Mode

Enable comprehensive debugging:
```bash
python -m abstractcore.server --debug
```

Debug mode logs:
- Request headers and body
- Provider API calls
- Media processing steps
- Tool call conversions
- Full stack traces

### Health Monitoring

Monitor server health:
```bash
# Check health
curl http://localhost:8000/health

# Expected response
{"status":"healthy","version":"2.5.2","timestamp":"..."}
```

---

## Troubleshooting

### Server Won't Start

**Problem**: Import errors or port conflicts.

**Solution**:
```bash
# Check dependencies
pip install -r requirements.txt

# Check port availability
lsof -i :8000

# Use different port
python -m abstractcore.server --port 8080
```

### Model Not Found

**Problem**: Model not listed in `/v1/models`.

**Solution**:
1. Check provider is installed: `pip list | grep openai`
2. Verify API key is set: `echo $OPENAI_API_KEY`
3. Check provider status: `curl http://localhost:8000/providers`

### Streaming Not Working

**Problem**: Streaming responses not received.

**Solution**:
- Disable proxy buffering (Nginx: `proxy_buffering off;`)
- Use `stream=True` in requests library
- Process lines individually, not full response

### Media Processing Fails

**Problem**: Images/documents not processed.

**Solution**:
1. Check file size (max 10MB per file)
2. Verify base64 encoding is correct
3. Check MIME type in data URL
4. Ensure model supports vision/document processing

---

## Future Enhancements

Planned improvements:
- WebSocket support for real-time streaming
- GraphQL API alongside REST
- Built-in rate limiting and quotas
- Caching layer for repeated requests
- Metrics endpoint (Prometheus format)
- OpenTelemetry tracing
- Multi-tenancy support
- Request replay and debugging UI

---

For detailed implementation, see `/Users/albou/projects/abstractcore/abstractcore/server/app.py` (2000+ lines).

## Related Modules

**Direct dependencies**:
- [`core/`](../core/README.md) - LLM factory and generation
- [`providers/`](../providers/README.md) - Provider registry and health checks
- [`config/`](../config/README.md) - Server configuration management
- [`processing/`](../processing/README.md) - High-level processors
- [`media/`](../media/README.md) - Media upload and processing
- [`structured/`](../structured/README.md) - API response schemas
- [`exceptions/`](../exceptions/README.md) - Error response formatting
- [`utils/`](../utils/README.md) - Logging, metrics

**Exposes**:
- [`architectures/`](../architectures/README.md) - Model capabilities via API
- [`assets/`](../assets/README.md) - Provider and model listings
- [`tools/`](../tools/README.md) - Tool execution endpoints
- [`embeddings/`](../embeddings/README.md) - Embedding generation endpoints

**Related systems**:
- [`events/`](../events/README.md) - Server event emission
