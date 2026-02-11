# Embeddings Endpoint Documentation

## Overview

The `/v1/embeddings` endpoint creates embedding vectors representing input text. These embeddings are useful for semantic search, document similarity, clustering, classification, and Retrieval-Augmented Generation (RAG).

## Endpoint

```
POST /v1/embeddings
```

## Request Body

### Parameters

#### `input` (string or array, **Required**)

Input text to embed, encoded as a string or array of strings.

**Details:**
- To embed multiple inputs in a single request, pass an array of strings
- The input must not exceed the max input tokens for the model (8192 tokens for most embedding models)
- Cannot be an empty string
- Any array must be 2048 dimensions or less

**Examples:**
```json
// Single string
"input": "this is the story of starship lost in space"

// Array of strings
"input": [
  "The food was delicious and the waiter was friendly",
  "I love programming in Python",
  "Machine learning is fascinating"
]
```

#### `model` (string, **Required**)

ID of the model to use. Use `provider/model` format.

**Details:**
- Format: `provider/model-name`
- You can use the List models API to see all available models
- Filter embedding models: `GET /v1/models?type=text-embedding`

**Supported Providers:**
- **HuggingFace**: Local sentence-transformers models with ONNX acceleration
- **Ollama**: Local embedding models via Ollama API
- **LMStudio**: Local embedding models via LMStudio API

**Examples:**
```json
"model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
"model": "ollama/granite-embedding:278m"
"model": "lmstudio/text-embedding-all-minilm-l6-v2"
```

#### `encoding_format` (string, **Optional**)

The format to return the embeddings in.

**Options:**
- `float` (default) - Returns embeddings as an array of floats
- `base64` - Returns embeddings in base64-encoded format

**Default:** `"float"`

**Example:**
```json
"encoding_format": "float"
```

#### `dimensions` (integer, **Optional**)

The number of dimensions the resulting output embeddings should have.

**Details:**
- Only supported in some models (e.g., models with Matryoshka support)
- If specified, embeddings will be truncated to this dimension
- Set to `0` or `null` to use the model's default dimension
- Useful for reducing storage/compute requirements while maintaining reasonable accuracy

**Default:** `null` (model's default dimension)

**Example:**
```json
"dimensions": 0  // Use model default
"dimensions": 256  // Truncate to 256 dimensions
```

#### `user` (string, **Optional**)

A unique identifier representing your end-user.

**Details:**
- Can help providers to monitor and detect abuse
- Optional but recommended for production applications
- Use consistent identifiers for the same user across requests

**Default:** `null`

**Example:**
```json
"user": "user-123"
```

## Example Request

```bash
curl -X POST https://api.yourdomain.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "this is the story of starship lost in space",
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
    "encoding_format": "float",
    "dimensions": 0,
    "user": "user-123"
  }'
```

## Response Format

Returns a list of embedding objects.

### Response Structure

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.0023064255,
        -0.009327292,
        -0.0028842222,
        // ... (1536 floats total for ada-002)
      ],
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

### Response Fields

#### `object` (string)
Always `"list"`

#### `data` (array)
Array of embedding objects, one for each input text

**Embedding Object:**
- `object` (string): Always `"embedding"`
- `embedding` (array): The embedding vector (list of floats)
- `index` (integer): The index of the embedding in the list

#### `model` (string)
The model used to generate embeddings (with provider prefix)

#### `usage` (object)
Token usage information:
- `prompt_tokens` (integer): Number of tokens in the input
- `total_tokens` (integer): Total tokens used

## Multiple Inputs Example

```bash
curl -X POST https://api.yourdomain.com/v1/embeddings \
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

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.002, -0.009, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.015, -0.003, ...],
      "index": 1
    },
    {
      "object": "embedding",
      "embedding": [-0.001, 0.012, ...],
      "index": 2
    }
  ],
  "model": "ollama/granite-embedding:278m",
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 15
  }
}
```

## Use Cases

### 1. Semantic Search

```python
import requests

# Embed documents
docs = ["Python is great", "JavaScript is versatile", "Rust is fast"]
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": docs,
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
})
doc_embeddings = [item["embedding"] for item in response.json()["data"]]

# Embed query
query_response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Tell me about Python",
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
})
query_embedding = query_response.json()["data"][0]["embedding"]

# Calculate similarity (simplified)
# Use cosine similarity to find most relevant document
```

### 2. Document Clustering

```python
# Embed multiple documents
documents = [
    "AI and machine learning",
    "Deep learning models",
    "Web development with React",
    "Frontend frameworks",
    "Neural networks"
]

response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": documents,
    "model": "ollama/granite-embedding:278m"
})

embeddings = [item["embedding"] for item in response.json()["data"]]
# Apply clustering algorithm (e.g., K-means) to group similar documents
```

### 3. RAG (Retrieval-Augmented Generation)

```python
# Embed knowledge base
knowledge_base = ["Fact 1...", "Fact 2...", "Fact 3..."]
kb_response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": knowledge_base,
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
})
kb_embeddings = [item["embedding"] for item in kb_response.json()["data"]]

# Embed user query
query = "What is...?"
query_response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": query,
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
})
query_embedding = query_response.json()["data"][0]["embedding"]

# Find most relevant facts using similarity
# Use retrieved facts to augment LLM prompt
```

## Discovering Available Models

### List All Embedding Models

```bash
curl http://localhost:8000/v1/models?type=text-embedding
```

### List Provider-Specific Embedding Models

```bash
# Ollama embedding models
curl http://localhost:8000/v1/models?provider=ollama&type=text-embedding

# HuggingFace embedding models
curl http://localhost:8000/v1/models?provider=huggingface&type=text-embedding

# LMStudio embedding models
curl http://localhost:8000/v1/models?provider=lmstudio&type=text-embedding
```

## Error Responses

### Invalid Provider

```json
{
  "detail": {
    "error": {
      "message": "Embedding provider 'invalid' not supported. Supported providers: huggingface, ollama, lmstudio",
      "type": "unsupported_provider"
    }
  }
}
```

### Model Not Found

```json
{
  "detail": {
    "error": {
      "message": "Model 'non-existent-model' not found",
      "type": "model_not_found"
    }
  }
}
```

### Empty Input

```json
{
  "detail": {
    "error": {
      "message": "Input cannot be empty",
      "type": "invalid_request"
    }
  }
}
```

## Best Practices

### 1. Batch Processing

For multiple texts, use batch processing (single request with array) instead of multiple requests:

✅ **Good:**
```python
response = requests.post("/v1/embeddings", json={
    "input": ["text1", "text2", "text3"],
    "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
})
```

❌ **Avoid:**
```python
# Multiple requests - slower
for text in ["text1", "text2", "text3"]:
    requests.post("/v1/embeddings", json={"input": text, "model": "..."})
```

### 2. Choose Appropriate Model

- **Fast & Small**: `huggingface/sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **Balanced**: `ollama/granite-embedding:107m` (768 dims)
- **High Quality**: `ollama/granite-embedding:278m` (768 dims)

### 3. Use Dimension Truncation

If supported by your model, use dimension truncation to save storage:

```python
response = requests.post("/v1/embeddings", json={
    "input": "text",
    "model": "huggingface/nomic-ai/nomic-embed-text-v1.5",
    "dimensions": 256  # Truncate from 768 to 256
})
```

### 4. Monitor Usage

Use the `user` parameter to track usage per user:

```python
response = requests.post("/v1/embeddings", json={
    "input": "text",
    "model": "...",
    "user": f"user-{user_id}"  # Track by user
})
```

### 5. Cache Embeddings

Embeddings for the same text are always the same. Cache them to avoid redundant API calls:

```python
embedding_cache = {}

def get_embedding(text, model):
    cache_key = f"{text}:{model}"
    if cache_key not in embedding_cache:
        response = requests.post("/v1/embeddings", json={
            "input": text,
            "model": model
        })
        embedding_cache[cache_key] = response.json()["data"][0]["embedding"]
    return embedding_cache[cache_key]
```

## Related Endpoints

- **List Models**: `GET /v1/models?type=text-embedding`
- **Chat Completions**: `POST /v1/chat/completions`
- **Health Check**: `GET /health`

---

**OpenAI Compatibility**: This endpoint follows the OpenAI Embeddings API format, making it compatible with existing tools and libraries that support OpenAI's API.

