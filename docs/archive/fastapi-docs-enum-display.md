# FastAPI Documentation - Enum Display

## Overview

The `/v1/models` endpoint now displays the `type` parameter as a proper enum in the FastAPI interactive documentation (`/docs` and `/redoc`).

## What You'll See

### Interactive Documentation (/docs)

When you navigate to `http://localhost:8000/docs`, the `/v1/models` endpoint will show:

**Query Parameters:**

1. **provider** (string, optional)
   - Description: "Filter by provider (e.g., 'ollama', 'openai', 'anthropic', 'lmstudio')"
   - Example: "ollama"
   - Type: Free text input

2. **type** (ModelType enum, optional)
   - Description: "Filter by model type: 'text-generation' for chat/completion models, 'text-embedding' for embedding models"
   - Example: "text-embedding"
   - Type: **Dropdown with enum values:**
     - `text-generation`
     - `text-embedding`

### Visual Representation

```
GET /v1/models

Query Parameters:
┌─────────────────────────────────────────────────────────────┐
│ provider (string, optional)                                  │
│ Filter by provider (e.g., 'ollama', 'openai', ...)         │
│ Example: ollama                                             │
│ [              ]                                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ type (ModelType, optional)                                   │
│ Filter by model type: 'text-generation' for chat/...       │
│ Example: text-embedding                                     │
│ [▼ Select...        ]  ← Dropdown with enum values         │
│   - text-generation                                         │
│   - text-embedding                                          │
└─────────────────────────────────────────────────────────────┘
```

## Testing the Documentation

### 1. Start the Server

```bash
python -m abstractllm.server.app
```

### 2. Access Interactive Docs

**Swagger UI** (Try it out interface):
```
http://localhost:8000/docs
```

**ReDoc** (Clean reference documentation):
```
http://localhost:8000/redoc
```

### 3. Using the Enum in Swagger UI

1. Navigate to `http://localhost:8000/docs`
2. Find the `GET /v1/models` endpoint
3. Click "Try it out"
4. Click the `type` dropdown
5. You'll see two options:
   - `text-generation`
   - `text-embedding`
6. Select one and click "Execute"

### 4. Schema Definition

In the OpenAPI schema (visible at `http://localhost:8000/openapi.json`), the enum is defined as:

```json
{
  "parameters": [
    {
      "name": "provider",
      "in": "query",
      "required": false,
      "schema": {
        "type": "string",
        "title": "Provider",
        "example": "ollama"
      },
      "description": "Filter by provider..."
    },
    {
      "name": "type",
      "in": "query",
      "required": false,
      "schema": {
        "allOf": [{"$ref": "#/components/schemas/ModelType"}],
        "example": "text-embedding"
      },
      "description": "Filter by model type..."
    }
  ]
}
```

With the schema definition:

```json
{
  "ModelType": {
    "type": "string",
    "enum": ["text-generation", "text-embedding"],
    "title": "ModelType"
  }
}
```

## Benefits

### 1. Type Safety in UI
Users can't enter invalid values - they must select from the dropdown

### 2. Discoverability
Users immediately see what options are available without reading docs

### 3. API Validation
FastAPI automatically validates that only valid enum values are accepted

### 4. Auto-Generated Client Code
When generating client libraries, the enum is properly typed:

**Python Client:**
```python
from enum import Enum

class ModelType(str, Enum):
    TEXT_GENERATION = "text-generation"
    TEXT_EMBEDDING = "text-embedding"

# Usage with autocomplete
response = client.get("/v1/models", params={"type": ModelType.TEXT_EMBEDDING})
```

**TypeScript Client:**
```typescript
enum ModelType {
  TEXT_GENERATION = "text-generation",
  TEXT_EMBEDDING = "text-embedding"
}

// Usage with type safety
const response = await client.get("/v1/models", {
  params: { type: ModelType.TEXT_EMBEDDING }
});
```

## Comparison: Before vs After

### Before (no Query annotation)
```python
async def list_models(
    provider: Optional[str] = None,
    type: Optional[ModelType] = None
):
```

**Documentation shows:**
- Parameter type: `ModelType` (not very clear)
- No description
- No example
- Might not show enum values in dropdown

### After (with Query annotation)
```python
async def list_models(
    provider: Optional[str] = Query(None, description="...", example="ollama"),
    type: Optional[ModelType] = Query(None, description="...", example="text-embedding")
):
```

**Documentation shows:**
- Clear description of what each filter does
- Example values for guidance
- Dropdown with enum values for `type`
- Better UX in interactive docs

## Testing Examples

### Via Swagger UI

1. Go to `http://localhost:8000/docs`
2. Try these combinations:
   - No filters → See all models
   - `type: text-embedding` → See only embedding models
   - `type: text-generation` → See only generation models
   - `provider: ollama, type: text-embedding` → See Ollama embedding models

### Via curl (validates enum)

```bash
# Valid - works
curl "http://localhost:8000/v1/models?type=text-embedding"

# Valid - works
curl "http://localhost:8000/v1/models?type=text-generation"

# Invalid - FastAPI returns validation error
curl "http://localhost:8000/v1/models?type=invalid-type"
# Returns: 422 Unprocessable Entity
# {"detail": [{"loc": ["query", "type"], "msg": "value is not a valid enumeration member..."}]}
```

## Additional Features

### 1. Auto-Validation
FastAPI automatically validates enum values:
- Valid values pass through
- Invalid values return 422 error with helpful message

### 2. OpenAPI Schema
The enum is properly documented in the OpenAPI schema, making it available to:
- API documentation tools
- Client code generators
- API testing tools
- Integration platforms

### 3. Type Hints
The enum provides type hints in Python:
```python
from abstractllm.server.app import ModelType

# IDE autocomplete works
model_type = ModelType.TEXT_EMBEDDING  # ✓ Autocompletes
model_type = ModelType.TEXT_GENERATION  # ✓ Autocompletes
model_type = ModelType.INVALID  # ✗ Type error
```

---

**Summary**: The `type` parameter now appears as a proper dropdown enum in FastAPI's interactive documentation, making it easy for users to discover and use the model filtering feature! 🎉

