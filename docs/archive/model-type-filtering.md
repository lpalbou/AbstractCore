# Model Type Filtering

## Overview

The `/v1/models` endpoint supports filtering models by type to distinguish between **text generation** models and **text embedding** models. This makes it easier to discover and work with models for specific use cases.

## API Endpoint

```
GET /v1/models
```

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | string | No | Filter by provider (e.g., 'ollama', 'lmstudio', 'openai') |
| `type` | enum | No | Filter by model type: `text-generation` or `text-embedding` |

## Model Type Enum

```
text-generation    - Models for text generation (chat, completion)
text-embedding     - Models for embedding text into vector representations
```

## Usage Examples

### 1. List All Models (No Filter)

```bash
curl http://localhost:8000/v1/models
```

Returns all available models from all providers.

### 2. List Only Embedding Models

```bash
curl http://localhost:8000/v1/models?type=text-embedding
```

Returns only embedding models across all providers:
```json
{
  "object": "list",
  "data": [
    {"id": "ollama/granite-embedding:278m", "object": "model", ...},
    {"id": "lmstudio/all-MiniLM-L6-v2-embedding", "object": "model", ...},
    {"id": "huggingface/nomic-embed-text-v1.5", "object": "model", ...},
    {"id": "openai/text-embedding-ada-002", "object": "model", ...}
  ]
}
```

### 3. List Only Text Generation Models

```bash
curl http://localhost:8000/v1/models?type=text-generation
```

Returns only text generation models:
```json
{
  "object": "list",
  "data": [
    {"id": "ollama/llama3:latest", "object": "model", ...},
    {"id": "ollama/qwen3-coder:30b", "object": "model", ...},
    {"id": "openai/gpt-4", "object": "model", ...},
    {"id": "anthropic/claude-3-opus-20240229", "object": "model", ...}
  ]
}
```

### 4. Combine Provider and Type Filters

```bash
# Ollama embedding models only
curl http://localhost:8000/v1/models?provider=ollama&type=text-embedding

# OpenAI text generation models only
curl http://localhost:8000/v1/models?provider=openai&type=text-generation
```

## Embedding Model Detection

The server uses intelligent heuristics to detect embedding models:

### Detection Patterns

Models are classified as embedding models if their name contains:

- `embed` - Most embedding models
- `all-minilm` - Sentence-transformers MiniLM models
- `all-mpnet` - Sentence-transformers MPNet models
- `nomic-embed` - Nomic embedding models
- `bert-` - BERT models (e.g., bert-base-uncased)
- `-bert` - BERT-based embedding models (e.g., nomic-bert-2048)
- `bge-` - BAAI BGE embedding models
- `gte-` - GTE embedding models
- `e5-` - E5 embedding models
- `instructor-` - Instructor embedding models
- `granite-embedding` - IBM Granite embedding models

### Examples

| Model Name | Detected Type |
|------------|---------------|
| `granite-embedding:278m` | text-embedding |
| `all-MiniLM-L6-v2` | text-embedding |
| `nomic-embed-text-v1.5` | text-embedding |
| `bert-base-uncased` | text-embedding |
| `nomic-bert-2048` | text-embedding |
| `text-embedding-ada-002` | text-embedding |
| `llama3:latest` | text-generation |
| `qwen3-coder:30b` | text-generation |
| `gpt-4` | text-generation |
| `claude-3-opus` | text-generation |

## Python Client Example

```python
import requests

def get_models(type_filter=None, provider=None):
    """Get models with optional filtering."""
    params = {}
    if type_filter:
        params['type'] = type_filter
    if provider:
        params['provider'] = provider
    
    response = requests.get('http://localhost:8000/v1/models', params=params)
    return response.json()['data']

# Get all embedding models
embedding_models = get_models(type_filter='text-embedding')
print(f"Found {len(embedding_models)} embedding models")

# Get Ollama text generation models
ollama_gen_models = get_models(
    type_filter='text-generation',
    provider='ollama'
)
print(f"Found {len(ollama_gen_models)} Ollama text generation models")
```

## Use Cases

### 1. Building Embedding UIs

Filter to show only embedding models when building interfaces for:
- Semantic search
- Document similarity
- Clustering
- RAG (Retrieval-Augmented Generation)

```bash
curl "http://localhost:8000/v1/models?type=text-embedding"
```

### 2. Building Chat/Completion UIs

Filter to show only text generation models for:
- Chat interfaces
- Text completion
- Code generation
- Conversational AI

```bash
curl "http://localhost:8000/v1/models?type=text-generation"
```

### 3. Provider-Specific Discovery

Discover what embedding models are available locally:

```bash
# Local Ollama embedding models
curl "http://localhost:8000/v1/models?provider=ollama&type=text-embedding"

# Local LMStudio embedding models
curl "http://localhost:8000/v1/models?provider=lmstudio&type=text-embedding"
```

### 4. Automated Model Selection

Programmatically select the right model type:

```python
def get_best_model(task_type: str):
    """Get best available model for task type."""
    models = get_models(type_filter=task_type)
    
    # Prefer local providers for privacy/cost
    local_models = [m for m in models if m['owned_by'] in ['ollama', 'lmstudio']]
    
    if local_models:
        return local_models[0]['id']
    else:
        return models[0]['id'] if models else None

# For embedding tasks
embedding_model = get_best_model('text-embedding')
print(f"Using embedding model: {embedding_model}")

# For generation tasks  
generation_model = get_best_model('text-generation')
print(f"Using generation model: {generation_model}")
```

## Response Format

The response follows OpenAI's model list format:

```json
{
  "object": "list",
  "data": [
    {
      "id": "provider/model-name",
      "object": "model",
      "owned_by": "provider",
      "created": 1234567890,
      "permission": [
        {
          "allow_create_engine": false,
          "allow_sampling": true
        }
      ]
    }
  ]
}
```

## Benefits

1. **Simplified Discovery**: Easily find models for specific use cases
2. **Better UX**: Build cleaner interfaces with appropriate model lists
3. **Type Safety**: Prevent using embedding models for generation and vice versa
4. **Flexibility**: Combine with provider filtering for precise model selection
5. **OpenAI Compatible**: Works with existing OpenAI-compatible tools

## Future Enhancements

Potential improvements:
- Add model capabilities metadata (context length, languages, etc.)
- Support for additional model types (image generation, audio, etc.)
- Model performance/quality metrics
- Provider-specific model recommendations

---

**Related Documentation:**
- [Server Configuration](server-configuration.md)
- [Embeddings](../embeddings.md)
- [Getting Started](../getting-started.md)
