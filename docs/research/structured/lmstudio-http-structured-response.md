# LMStudio HTTP API: Structured Response Guide

## Overview

This document demonstrates how to enforce structured responses when communicating directly with LMStudio through its HTTP endpoints, bypassing the Python SDK entirely. LMStudio provides an OpenAI-compatible API that supports structured output generation using JSON Schema.

## Data Flow Diagram

```
┌─────────────────┐    HTTP POST     ┌─────────────────┐    WebSocket     ┌─────────────────┐
│   HTTP Client   │ ──────────────► │   LMStudio      │ ──────────────► │   Model         │
│                 │  /v1/chat/      │   HTTP Server   │   Internal       │   Inference     │
│                 │  completions    │   (Port 1234)   │   Protocol       │   Engine        │
└─────────────────┘                 └─────────────────┘                  └─────────────────┘
         │                                   │                                   │
         │ JSON Schema                       │ KV Config                         │ Structured
         │ in request                        │ Translation                       │ Generation
         │                                   │                                   │
         ▼                                   ▼                                   ▼
┌─────────────────┐    HTTP Response ┌─────────────────┐    JSON Output    ┌─────────────────┐
│   Parsed JSON   │ ◄────────────── │   Response      │ ◄────────────── │   Schema        │
│   Object        │                 │   Formatter     │                 │   Validator     │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
```

## API Endpoints

### Base URL
```
http://127.0.0.1:1234
```

### Key Endpoints
- **Health Check**: `GET /lmstudio-greeting`
- **List Models**: `GET /v1/models`
- **Chat Completions**: `POST /v1/chat/completions`

## Structured Response Implementation

### 1. Basic Setup

First, verify LMStudio is running and list available models:

```bash
# Check if LMStudio is running
curl -s http://127.0.0.1:1234/lmstudio-greeting

# List available models
curl -s http://127.0.0.1:1234/v1/models | jq '.data[].id'
```

### 2. Simple Structured Response

**Request Format:**
```bash
curl -X POST http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Tell me about The Hobbit book"}
    ],
    "max_tokens": 200,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "book_info",
        "schema": {
          "type": "object",
          "properties": {
            "title": {
              "type": "string",
              "description": "The title of the book"
            },
            "author": {
              "type": "string",
              "description": "The author of the book"
            },
            "publication_year": {
              "type": "integer",
              "description": "The year the book was published"
            },
            "genre": {
              "type": "string",
              "description": "The genre of the book"
            },
            "main_character": {
              "type": "string",
              "description": "The main character of the book"
            }
          },
          "required": ["title", "author", "publication_year", "genre", "main_character"],
          "additionalProperties": false
        }
      }
    }
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-xyz123",
  "object": "chat.completion",
  "created": 1761422819,
  "model": "mistralai/mistral-small-3.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"title\": \"The Hobbit\", \"author\": \"J.R.R. Tolkien\", \"publication_year\": 1937, \"genre\": \"Fantasy\", \"main_character\": \"Bilbo Baggins\"}",
        "reasoning_content": "",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  }
}
```

**Parsed JSON Content:**
```json
{
  "title": "The Hobbit",
  "author": "J.R.R. Tolkien",
  "publication_year": 1937,
  "genre": "Fantasy",
  "main_character": "Bilbo Baggins"
}
```

### 3. Complex Nested Schema

**Request with Arrays and Nested Objects:**
```bash
curl -X POST http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/mistral-small-3.2",
    "messages": [
      {"role": "user", "content": "List 3 main characters from The Hobbit with their roles"}
    ],
    "max_tokens": 300,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "characters",
        "schema": {
          "type": "object",
          "properties": {
            "book_title": {
              "type": "string"
            },
            "characters": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {
                    "type": "string"
                  },
                  "role": {
                    "type": "string"
                  },
                  "traits": {
                    "type": "array",
                    "items": {
                      "type": "string"
                    }
                  }
                },
                "required": ["name", "role", "traits"],
                "additionalProperties": false
              }
            }
          },
          "required": ["book_title", "characters"],
          "additionalProperties": false
        }
      }
    }
  }'
```

### 4. Streaming Structured Responses

LMStudio supports streaming with structured responses:

```bash
curl -X POST http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/mistral-small-3.2",
    "messages": [
      {"role": "user", "content": "Tell me about Python programming"}
    ],
    "max_tokens": 150,
    "stream": true,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "programming_info",
        "schema": {
          "type": "object",
          "properties": {
            "language": {
              "type": "string"
            },
            "description": {
              "type": "string"
            },
            "key_features": {
              "type": "array",
              "items": {
                "type": "string"
              }
            }
          },
          "required": ["language", "description", "key_features"],
          "additionalProperties": false
        }
      }
    }
  }'
```

**Streaming Response Format:**
```
data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","created":1761422917,"model":"mistralai/mistral-small-3.2","choices":[{"index":0,"delta":{"role":"assistant","content":"{"},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","created":1761422917,"model":"mistralai/mistral-small-3.2","choices":[{"index":0,"delta":{"content":" \"language\""},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","created":1761422917,"model":"mistralai/mistral-small-3.2","choices":[{"index":0,"delta":{"content":": \"Python\""},"finish_reason":null}]}

...

data: [DONE]
```

## Schema Requirements and Validation

### Supported Schema Types
- `object`: JSON objects with defined properties
- `array`: Arrays with item schemas
- `string`: Text values
- `integer`: Whole numbers
- `number`: Floating-point numbers
- `boolean`: True/false values

### Schema Validation Rules
1. **Required Fields**: Use `"required": ["field1", "field2"]` to enforce mandatory fields
2. **Additional Properties**: Set `"additionalProperties": false` to prevent extra fields
3. **Array Items**: Define item schemas using `"items": {...}`
4. **Descriptions**: Add `"description"` fields for better model understanding

### Error Handling

**Invalid Schema Example:**
```bash
curl -X POST http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/mistral-small-3.2",
    "messages": [{"role": "user", "content": "Test"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "invalid",
        "schema": {
          "type": "invalid_type"
        }
      }
    }
  }'
```

**Error Response:**
```json
{
  "error": "Invalid structured output configuration: data/type must be equal to one of the allowed values, data/type must be array, data/type must match a schema in anyOf"
}
```

## Implementation Details

### HTTP to WebSocket Translation

When you make an HTTP request to `/v1/chat/completions` with structured output:

1. **HTTP Layer**: LMStudio's HTTP server receives the request
2. **Schema Processing**: The JSON schema is validated and converted to internal format
3. **WebSocket Protocol**: The request is translated to LMStudio's internal WebSocket protocol
4. **Model Inference**: The model generates structured output based on the schema constraints
5. **Response Formatting**: The structured JSON is returned via HTTP response

### Key Differences from SDK

| Aspect | HTTP API | Python SDK |
|--------|----------|------------|
| **Connection** | Stateless HTTP requests | Persistent WebSocket connections |
| **Schema Definition** | Raw JSON Schema | Python `BaseModel` classes |
| **Error Handling** | HTTP status codes + JSON errors | Python exceptions |
| **Streaming** | Server-Sent Events | AsyncIterator/Iterator |
| **Type Safety** | Manual JSON parsing | Automatic deserialization |

### Performance Characteristics

- **Latency**: HTTP requests have connection overhead per request
- **Throughput**: WebSocket (SDK) is more efficient for multiple requests
- **Memory**: HTTP API has lower memory footprint
- **Scalability**: HTTP API is stateless and easier to load balance

## Practical Examples

### Example 1: Data Extraction

Extract structured data from unstructured text:

```bash
curl -X POST http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-4b-2507",
    "messages": [
      {"role": "user", "content": "John Smith, age 30, works as a software engineer at Google. He lives in San Francisco and has 5 years of experience."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person_info",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "job_title": {"type": "string"},
            "company": {"type": "string"},
            "location": {"type": "string"},
            "experience_years": {"type": "integer"}
          },
          "required": ["name", "age", "job_title", "company", "location", "experience_years"],
          "additionalProperties": false
        }
      }
    }
  }'
```

### Example 2: Classification Task

Classify text into predefined categories:

```bash
curl -X POST http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/mistral-small-3.2",
    "messages": [
      {"role": "user", "content": "I am really unhappy with this product. It broke after just one day and customer service was terrible."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "sentiment_analysis",
        "schema": {
          "type": "object",
          "properties": {
            "sentiment": {
              "type": "string",
              "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
              "type": "number",
              "minimum": 0,
              "maximum": 1
            },
            "key_phrases": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["sentiment", "confidence", "key_phrases"],
          "additionalProperties": false
        }
      }
    }
  }'
```

### Example 3: Code Generation

Generate structured code documentation:

```bash
curl -X POST http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-coder-30b",
    "messages": [
      {"role": "user", "content": "def calculate_fibonacci(n): return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "code_documentation",
        "schema": {
          "type": "object",
          "properties": {
            "function_name": {"type": "string"},
            "description": {"type": "string"},
            "parameters": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "type": {"type": "string"},
                  "description": {"type": "string"}
                },
                "required": ["name", "type", "description"]
              }
            },
            "return_type": {"type": "string"},
            "time_complexity": {"type": "string"},
            "space_complexity": {"type": "string"}
          },
          "required": ["function_name", "description", "parameters", "return_type", "time_complexity", "space_complexity"],
          "additionalProperties": false
        }
      }
    }
  }'
```

## Best Practices

### 1. Schema Design
- Keep schemas as specific as possible
- Use `enum` for limited value sets
- Add meaningful descriptions for better model understanding
- Set appropriate constraints (min/max for numbers, etc.)

### 2. Error Handling
- Always validate the JSON schema before sending requests
- Handle HTTP errors gracefully
- Parse and validate the returned JSON content
- Implement retry logic for transient failures

### 3. Performance Optimization
- Use appropriate `max_tokens` limits to avoid truncation
- Consider streaming for long responses
- Batch multiple requests when possible
- Cache schemas to avoid repeated validation

### 4. Model Selection
- Different models have varying structured output capabilities
- Test with your specific schema before production use
- Consider model size vs. accuracy trade-offs

## Limitations and Considerations

### Current Limitations
1. **Token Limits**: Complex schemas may be truncated if `max_tokens` is too low
2. **Model Capabilities**: Not all models handle complex schemas equally well
3. **Validation**: LMStudio validates schema format but not semantic correctness
4. **Error Recovery**: Limited ability to recover from malformed JSON generation

### Comparison with SDK
- **Pros**: Direct HTTP access, no Python dependencies, language-agnostic
- **Cons**: Manual JSON parsing, no type safety, connection overhead per request

## Conclusion

LMStudio's HTTP API provides robust support for structured responses through JSON Schema, offering a direct alternative to the Python SDK. While it requires more manual handling of JSON parsing and validation, it provides greater flexibility and can be used from any programming language or tool that supports HTTP requests.

The structured output capability works consistently across different models and supports both streaming and non-streaming responses, making it suitable for a wide range of applications from data extraction to code generation.
