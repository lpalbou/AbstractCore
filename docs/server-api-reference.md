# AbstractCore Server API Reference

**REST API documentation for the AbstractCore HTTP server.**

The AbstractCore server is an **optional component** that provides OpenAI-compatible HTTP endpoints built on top of the AbstractCore library. It enables integration with existing tools and agentic CLIs.

**Base URL**: `http://localhost:8000` (or your deployed server URL)

## Quick Navigation

- [Health & Discovery](#health--discovery-endpoints)
- [Chat Completions](#chat-completions-endpoints)
- [Embeddings](#embeddings-endpoint)
- [Models](#models-endpoint)
- [Providers](#providers-endpoint)

---

## Health & Discovery Endpoints

### GET /health

Health check endpoint for monitoring server status.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /

Server information and quick start examples.

**Response:** HTML page with server information and usage examples.

---

## Chat Completions Endpoints

### POST /v1/chat/completions

Standard OpenAI-compatible chat completions endpoint. Works with all providers.

**Request Body:**

```json
{
  "model": "provider/model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "tools": []
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier in `provider/model-name` format |
| `messages` | array | Yes | Array of message objects with `role` and `content` |
| `temperature` | float | No | Sampling temperature (0-2). Default: 0.7 |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `stream` | boolean | No | Enable streaming responses. Default: false |
| `tools` | array | No | Available tools for function calling |
| `top_p` | float | No | Nucleus sampling parameter. Default: 1.0 |
| `frequency_penalty` | float | No | Repetition penalty (-2 to 2) |
| `presence_penalty` | float | No | Topic penalty (-2 to 2) |
| `user` | string | No | Unique user identifier for tracking |

**Response (Non-streaming):**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "provider/model-name",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

**Response (Streaming):**

Server-sent events with `data:` prefix. Each chunk:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1677652288,
  "model": "provider/model-name",
  "choices": [{
    "index": 0,
    "delta": {
      "content": "Hello"
    },
    "finish_reason": null
  }]
}
```

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-5-haiku-latest",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 500
  }'
```

### POST /{provider}/v1/chat/completions

Provider-specific chat completions endpoint. Same as `/v1/chat/completions` but with provider in the path.

**Path Parameters:**
- `provider`: Provider name (e.g., `ollama`, `openai`, `anthropic`)

**Request Body:**
Same as `/v1/chat/completions` but `model` field doesn't need provider prefix.

**Example:**

```bash
curl -X POST http://localhost:8000/ollama/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

### POST /v1/responses

Alternative endpoint optimized for agentic CLI tools. Simpler request format.

**Request Body:**

```json
{
  "prompt": "Your question here",
  "model": "provider/model-name",
  "max_tokens": 1000,
  "tools": []
}
```

**Key Differences:**
- Uses `prompt` instead of `messages` array
- Simpler request structure
- Optimized for CLI tool integration
- Preferred by Codex CLI

**Example:**

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "List files in current directory",
    "model": "ollama/qwen3-coder:30b",
    "max_tokens": 500
  }'
```

---

## Embeddings Endpoint

### POST /v1/embeddings

Creates embedding vectors representing input text. Useful for semantic search, RAG, document similarity, and clustering.

**Request Body:**

```json
{
  "input": "this is the story of starship lost in space",
  "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
  "encoding_format": "float",
  "dimensions": 0,
  "user": "user-123"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string or array | Yes | Text(s) to embed. Can be single string or array of strings |
| `model` | string | Yes | Embedding model in `provider/model-name` format |
| `encoding_format` | string | No | Format: `"float"` or `"base64"`. Default: `"float"` |
| `dimensions` | integer | No | Output dimensions (for truncation). `0` = model default |
| `user` | string | No | User identifier for tracking and abuse monitoring |

**Supported Providers:**
- **HuggingFace**: Local sentence-transformers models with ONNX acceleration
- **Ollama**: Local embedding models via Ollama API
- **LMStudio**: Local embedding models via LMStudio API

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023064255, -0.009327292, ...],
      "index": 0
    }
  ],
  "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

**Multiple Inputs Example:**

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "The food was delicious",
      "I love programming",
      "Machine learning rocks"
    ],
    "model": "ollama/granite-embedding:278m"
  }'
```

**Use Cases:**
- **Semantic Search**: Find relevant documents by meaning
- **RAG (Retrieval-Augmented Generation)**: Retrieve context for LLM prompts
- **Document Clustering**: Group similar content together
- **Similarity Analysis**: Compare documents for relevance

**Discovering Embedding Models:**

```bash
# List all embedding models
curl http://localhost:8000/v1/models?type=text-embedding

# List by provider
curl http://localhost:8000/v1/models?provider=ollama&type=text-embedding
```

---

## Models Endpoint

### GET /v1/models

Lists all available models from all configured providers.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | string | Filter by provider (e.g., `ollama`, `openai`) |
| `type` | enum | Filter by type: `text-generation` or `text-embedding` |

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "ollama/qwen3-coder:30b",
      "object": "model",
      "created": 1677610602,
      "owned_by": "ollama"
    },
    {
      "id": "openai/gpt-4o-mini",
      "object": "model",
      "created": 1677610602,
      "owned_by": "openai"
    }
  ]
}
```

**Examples:**

```bash
# List all models
curl http://localhost:8000/v1/models

# Filter by provider
curl http://localhost:8000/v1/models?provider=ollama

# Filter by type
curl http://localhost:8000/v1/models?type=text-embedding

# Combine filters
curl http://localhost:8000/v1/models?provider=huggingface&type=text-embedding
```

**Model Type Detection:**

Models are automatically classified as `text-generation` or `text-embedding` based on naming patterns:

**Embedding Model Patterns:**
- Contains "embed", "all-minilm", "nomic-embed"
- Contains "bert-", "-bert", "bge-", "gte-", "e5-"
- Contains "instructor-", "granite-embedding"

---

## Providers Endpoint

### GET /providers

Lists all available AbstractCore providers and their capabilities.

**Response:**

```json
{
  "providers": [
    {
      "name": "ollama",
      "type": "llm",
      "model_count": 15,
      "status": "available",
      "description": "Ollama provider with 15 available models"
    },
    {
      "name": "openai",
      "type": "llm",
      "model_count": 8,
      "status": "available",
      "description": "OpenAI provider with 8 available models"
    }
  ]
}
```

**Provider Information:**

| Provider | Type | Setup | Best For |
|----------|------|-------|----------|
| **OpenAI** | Commercial API | Requires `OPENAI_API_KEY` | Production, highest quality |
| **Anthropic** | Commercial API | Requires `ANTHROPIC_API_KEY` | Long context, reasoning |
| **Ollama** | Local | Install Ollama | Privacy, offline use |
| **LMStudio** | Local | Install LMStudio GUI | Easy local setup |
| **MLX** | Local | Apple Silicon only | Mac optimization |
| **HuggingFace** | Local/API | Optional auth token | Research, custom models |
| **Mock** | Testing | No setup | Testing, CI/CD |

**Example:**

```bash
curl http://localhost:8000/providers
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "detail": {
    "error": {
      "message": "Detailed error message",
      "type": "error_type"
    }
  }
}
```

**Common Error Types:**

| Error Type | HTTP Code | Description |
|------------|-----------|-------------|
| `unsupported_provider` | 400 | Invalid provider specified |
| `model_not_found` | 404 | Model not available |
| `embedding_error` | 500 | Embedding generation failed |
| `invalid_request` | 400 | Malformed request |
| `authentication_error` | 401 | Invalid API key |
| `rate_limit_exceeded` | 429 | Too many requests |

---

## Best Practices

### 1. Model Selection

```bash
# Always use provider/model format
"model": "ollama/qwen3-coder:30b"  # ✅ Correct
"model": "qwen3-coder:30b"          # ❌ Wrong
```

### 2. Streaming for Long Responses

```json
{
  "model": "anthropic/claude-3-5-sonnet-latest",
  "messages": [...],
  "stream": true  // Better UX for long responses
}
```

### 3. Batch Embedding Requests

```json
{
  "input": ["text1", "text2", "text3"],  // ✅ Efficient
  "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
}
```

Instead of:
```bash
# ❌ Inefficient - multiple requests
for text in texts:
    request("/v1/embeddings", {"input": text})
```

### 4. Error Handling

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": "ollama/qwen3-coder:30b", "messages": [...]},
        timeout=30
    )
    response.raise_for_status()
    data = response.json()
except requests.exceptions.Timeout:
    # Handle timeout
    pass
except requests.exceptions.HTTPError as e:
    # Handle HTTP errors
    error_detail = e.response.json()
    print(f"Error: {error_detail['detail']['error']['message']}")
```

### 5. Provider Discovery

```python
# Check available providers before making requests
providers_resp = requests.get("http://localhost:8000/providers")
available = {p["name"] for p in providers_resp.json()["providers"]}

if "ollama" in available:
    # Use Ollama
    pass
else:
    # Fallback to OpenAI
    pass
```

---

## Interactive Documentation

When the server is running, visit these URLs for interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide:
- Interactive "Try it out" buttons
- Request/response examples
- Schema validation
- Parameter descriptions

---

## Related Documentation

- **[Server Guide](server.md)** - Server setup, configuration, and deployment
- **[Python API Reference](api-reference.md)** - Core AbstractCore library API
- **[Getting Started](getting-started.md)** - Core library quick start
- **[Embeddings Guide](embeddings.md)** - Embeddings deep dive
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

**Note**: This REST API is an **optional server component**. For programmatic Python usage, see the [Python API Reference](api-reference.md).

**OpenAI Compatibility**: All endpoints follow OpenAI API conventions, making the server compatible with existing OpenAI clients and tools.

