# Model Type Filtering Implementation Summary

## Overview

Added intelligent model type filtering to the `/v1/models` endpoint to distinguish between **text generation** and **text embedding** models, making it easier to discover and select appropriate models for specific use cases.

## Changes Made

### 1. Core Server (`abstractllm/server/app.py`)

**Added Model Type Enum**:
```python
class ModelType(str, Enum):
    TEXT_GENERATION = "text-generation"
    TEXT_EMBEDDING = "text-embedding"
```

**Added Embedding Detection Function**:
```python
def is_embedding_model(model_name: str) -> bool:
    """Detect if a model is an embedding model based on naming heuristics."""
```

Detection patterns include:
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

**Updated `/v1/models` Endpoint**:
- Added optional `type` parameter (enum: `text-generation`, `text-embedding`)
- Uses FastAPI's `Query` for proper documentation and enum display
- Filters models based on type when parameter is provided
- Works in combination with existing `provider` filter
- Default behavior (no filter) unchanged
- **Appears as dropdown enum in /docs and /redoc** ✨

### 2. Documentation

**Created**: `docs/model-type-filtering.md`
- Comprehensive API documentation
- Usage examples for all filter combinations
- Detection heuristics explanation
- Use case examples
- Python client code samples

**Created**: `docs/fastapi-docs-enum-display.md`
- Guide to viewing enum in FastAPI interactive docs
- Visual representation of how the dropdown appears
- OpenAPI schema details
- Testing instructions for /docs and /redoc

### 3. Demo

**Created**: `examples/model_filtering_demo.py`
- Interactive demo showing all filtering scenarios
- Model classification examples
- API usage documentation

## API Usage

### Basic Filtering

```bash
# All models (default behavior - unchanged)
GET /v1/models

# Only embedding models
GET /v1/models?type=text-embedding

# Only text generation models
GET /v1/models?type=text-generation
```

### Combined Filters

```bash
# Ollama embedding models
GET /v1/models?provider=ollama&type=text-embedding

# OpenAI text generation models
GET /v1/models?provider=openai&type=text-generation
```

### Example Response

```bash
curl "http://localhost:8000/v1/models?type=text-embedding"
```

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

## Detection Examples

| Model Name | Detected Type | Reason |
|------------|---------------|---------|
| `granite-embedding:278m` | text-embedding | Contains "granite-embedding" |
| `all-MiniLM-L6-v2` | text-embedding | Contains "all-minilm" |
| `nomic-embed-text-v1.5` | text-embedding | Contains "nomic-embed" |
| `bert-base-uncased` | text-embedding | Contains "bert-" |
| `nomic-bert-2048` | text-embedding | Contains "-bert" |
| `text-embedding-ada-002` | text-embedding | Contains "embed" |
| `llama3:latest` | text-generation | No embedding patterns |
| `qwen3-coder:30b` | text-generation | No embedding patterns |
| `gpt-4` | text-generation | No embedding patterns |
| `claude-3-opus` | text-generation | No embedding patterns |

## Use Cases

### 1. Building Embedding-Specific UIs
```bash
# Get only embedding models for RAG/semantic search interfaces
curl "http://localhost:8000/v1/models?type=text-embedding"
```

### 2. Building Chat/Completion UIs
```bash
# Get only text generation models for chat interfaces
curl "http://localhost:8000/v1/models?type=text-generation"
```

### 3. Provider-Specific Model Discovery
```bash
# What embedding models are available locally on Ollama?
curl "http://localhost:8000/v1/models?provider=ollama&type=text-embedding"
```

### 4. Automated Model Selection

```python
import requests

def get_best_model(task_type: str, prefer_local: bool = True):
    """Intelligently select best model for task."""
    response = requests.get(
        'http://localhost:8000/v1/models',
        params={'type': task_type}
    )
    models = response.json()['data']
    
    if prefer_local:
        # Prefer local providers (ollama, lmstudio)
        local_models = [m for m in models 
                       if m['owned_by'] in ['ollama', 'lmstudio']]
        if local_models:
            return local_models[0]['id']
    
    return models[0]['id'] if models else None

# Use it
embedding_model = get_best_model('text-embedding')
generation_model = get_best_model('text-generation')
```

## Design Principles Applied

✅ **Simple and Clean**: Single optional parameter, intuitive naming
✅ **Backward Compatible**: Default behavior unchanged (no filter)
✅ **Robust General-Purpose**: Heuristics work across all providers
✅ **Composable**: Works with existing provider filter
✅ **OpenAI Compatible**: Follows OpenAI API conventions

## Benefits

### For Users
- **Easy Discovery**: Find the right model type quickly
- **Better UX**: Build cleaner interfaces with relevant models only
- **Type Safety**: Avoid using wrong model types for tasks
- **Flexibility**: Combine filters for precise selection

### For Developers
- **Clean API**: Single endpoint with composable filters
- **Extensible**: Easy to add more detection patterns
- **Well-Documented**: Comprehensive docs and examples
- **Maintainable**: Clear separation of concerns

### For the Project
- **Feature Parity**: Matches capabilities of major AI platforms
- **User Friendly**: Reduces friction in model selection
- **Future Ready**: Foundation for additional model metadata

## Testing

Run the demo to see filtering in action:
```bash
python examples/model_filtering_demo.py
```

Manual testing:
```bash
# Start server
python -m abstractllm.server.app

# Test filters
curl "http://localhost:8000/v1/models?type=text-embedding"
curl "http://localhost:8000/v1/models?type=text-generation"
curl "http://localhost:8000/v1/models?provider=ollama&type=text-embedding"
```

## Future Enhancements

Potential improvements:
1. **Model Capabilities**: Add metadata about context length, languages, etc.
2. **Additional Types**: Support image-generation, text-to-speech, etc.
3. **Performance Metrics**: Include model quality/speed ratings
4. **Smart Recommendations**: Suggest best model for specific tasks
5. **Model Tags**: User-defined tags for custom organization

## Related Files

- `abstractllm/server/app.py` - Core implementation
- `docs/model-type-filtering.md` - Full documentation
- `examples/model_filtering_demo.py` - Interactive demo
- `abstractllm/embeddings/models.py` - HuggingFace model configs (reference)

## About `models.py`

The `abstractllm/embeddings/models.py` file contains curated HuggingFace embedding model configurations. It's used by `EmbeddingManager` for:
- Model aliasing (e.g., "all-minilm-l6-v2" → full HF model ID)
- Matryoshka dimension support detection
- Model metadata (dimensions, languages, etc.)

**Note**: It's HuggingFace-specific and doesn't need modification for this feature. The server's detection heuristics work independently across all providers.

---

**Result**: Clean, intuitive model type filtering that makes it easy to discover and select appropriate models for specific tasks while maintaining full backward compatibility! 🎯

