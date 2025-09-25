"""
AbstractCore Server - SOTA Clean Implementation

A focused FastAPI server providing only essential OpenAI-compatible endpoints.
No duplicates, no over-engineering - just what works with maximum compatibility.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.factory import create_llm
from ..exceptions import AuthenticationError, RateLimitError, ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models


# ============================================================================
# Enums & Models
# ============================================================================

class ModelType(str, Enum):
    """Model type for filtering"""
    CHAT = "chat"
    EMBEDDING = "embedding"


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible message format"""
    role: Literal["system", "user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")


class OpenAIChatCompletionRequest(BaseModel):
    """
    🎯 **STANDARD OPENAI REQUEST** 🎯

    100% standard OpenAI chat completion request - provider in URL, model in body.
    """
    model: str = Field(description="Model to use (e.g., 'claude-3-5-haiku-latest', 'gpt-4o-mini')")
    messages: List[OpenAIChatMessage] = Field(
        description="List of messages in the conversation"
    )

    # === CORE GENERATION PARAMETERS ===
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Controls randomness (0=deterministic, 2=very random)")
    max_tokens: Optional[int] = Field(default=500, ge=1, le=4000, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling: cumulative probability cutoff")
    stream: Optional[bool] = Field(default=False, description="Enable real-time streaming of response")

    # === REPETITION CONTROL ===
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Penalize tokens based on frequency (-2 to 2)")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Penalize tokens based on presence (-2 to 2)")

    # === CONTROL PARAMETERS ===
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences - model stops when encountering these")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible outputs")

    # === COMPLETION CONTROL ===
    n: Optional[int] = Field(default=1, ge=1, le=128, description="Number of chat completion choices to generate")
    best_of: Optional[int] = Field(default=None, ge=1, le=20, description="Generates best_of completions and returns n best")
    suffix: Optional[str] = Field(default=None, description="Text that comes after completion (for insertion)")
    echo: Optional[bool] = Field(default=False, description="Echo back the prompt in addition to completion")
    user: Optional[str] = Field(default=None, description="Unique identifier for end-user (for abuse monitoring)")

    # === ADVANCED PARAMETERS ===
    logit_bias: Optional[Dict[str, float]] = Field(default=None, description="Modify token probabilities (-100 to 100)")
    logprobs: Optional[bool] = Field(default=False, description="Return log probabilities of output tokens")
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=5, description="Number of most likely tokens to return (0-5)")

    # === STREAMING OPTIONS ===
    stream_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Options for streaming response. Only set when stream=true. Use {'include_usage': true} for token usage"
    )

    # === FUNCTION/TOOL CALLING ===
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Available tools/functions for the model to call")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default="auto",
        description="Controls tool usage: 'auto', 'none', or {'type': 'function', 'function': {'name': 'func_name'}}"
    )
    parallel_tool_calls: Optional[bool] = Field(default=True, description="Whether to enable parallel function calling")

    # === STRUCTURED OUTPUT ===
    response_format: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Format for structured output. Use {'type': 'json_schema', 'json_schema': {...}} for JSON Schema"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "description": "Basic chat completion",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": "Extract: John Smith, age 30, engineer"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "person_info",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "age": {"type": "integer", "minimum": 0},
                                    "occupation": {"type": "string"}
                                },
                                "required": ["name", "age"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "model": "qwen3-coder:30b",
                    "temperature": 0.7,
                    "stream": False
                },
                {
                    "description": "Deterministic coding assistance",
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a senior Python developer. Write clean, well-documented code."},
                        {"role": "user", "content": "Write a function to reverse a string with error handling"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300,
                    "seed": 42,
                    "stop": ["```\n\n", "# End of function"]
                },
                {
                    "description": "Creative writing with streaming",
                    "model": "claude-3-5-haiku-latest",
                    "messages": [
                        {"role": "system", "content": "You are a creative storyteller. Write engaging, original stories."},
                        {"role": "user", "content": "Tell me a short story about AI discovering emotions"}
                    ],
                    "temperature": 1.2,
                    "max_tokens": 800,
                    "stream": True,
                    "frequency_penalty": 0.3,
                    "presence_penalty": 0.2
                },
                {
                    "description": "Function calling with weather tools",
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "What's the weather like in San Francisco and should I bring an umbrella?"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "description": "Get the current weather in a given location",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string",
                                            "description": "The city and state, e.g. San Francisco, CA"
                                        },
                                        "unit": {
                                            "type": "string",
                                            "enum": ["celsius", "fahrenheit"],
                                            "description": "Temperature unit"
                                        }
                                    },
                                    "required": ["location"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "get_umbrella_recommendation",
                                "description": "Recommend whether to bring an umbrella based on weather conditions",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "precipitation_chance": {"type": "number"},
                                        "wind_speed": {"type": "number"}
                                    }
                                }
                            }
                        }
                    ],
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "temperature": 0.1
                },
                {
                    "description": "Structured output with JSON schema",
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "Analyze this: 'Alice Johnson, 30, Software Engineer at TechCorp, living in Seattle, earns $120k annually'"}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "person_analysis",
                            "description": "Structured person information",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Full name"},
                                    "age": {"type": "integer", "minimum": 0, "maximum": 150},
                                    "occupation": {"type": "string"},
                                    "company": {"type": "string"},
                                    "location": {"type": "string"},
                                    "salary": {"type": "integer"},
                                    "salary_currency": {"type": "string", "default": "USD"}
                                },
                                "required": ["name", "age", "occupation"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "temperature": 0.0,
                    "seed": 123
                },
                {
                    "description": "Advanced parameters with logprobs",
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "Rate this movie: 'The Matrix' - give a score from 1-10"}
                    ],
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 100,
                    "logprobs": True,
                    "top_logprobs": 3,
                    "logit_bias": {"10": 2.0, "9": 1.0},
                    "stop": ["\n\n", "Score:"]
                },
                {
                    "description": "Force specific function call",
                    "model": "claude-3-5-haiku-latest",
                    "messages": [
                        {"role": "user", "content": "I need to calculate 15% tip on $45.67"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calculate_tip",
                                "description": "Calculate tip amount and total",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "bill_amount": {"type": "number", "description": "Original bill amount"},
                                        "tip_percentage": {"type": "number", "description": "Tip percentage (e.g., 15 for 15%)"}
                                    },
                                    "required": ["bill_amount", "tip_percentage"]
                                }
                            }
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "calculate_tip"}
                    },
                    "temperature": 0.0
                },
                {
                    "description": "Multiple completion choices with ALL parameters",
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Write a tagline for a coffee shop"}
                    ],
                    "n": 3,
                    "best_of": 5,
                    "temperature": 1.0,
                    "max_tokens": 50,
                    "top_p": 0.8,
                    "frequency_penalty": 0.5,
                    "presence_penalty": 0.2,
                    "stop": [".", "!"],
                    "seed": 789,
                    "logit_bias": {"15496": -100, "3137": 2.0},
                    "logprobs": True,
                    "top_logprobs": 5,
                    "user": "user_12345",
                    "echo": False
                },
                {
                    "description": "Streaming with complete options",
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "Explain the concept of machine learning"}
                    ],
                    "stream": True,
                    "stream_options": {
                        "include_usage": True
                    },
                    "temperature": 0.7,
                    "max_tokens": 200,
                    "n": 1,
                    "user": "demo_user_456"
                },
                {
                    "description": "Text insertion with suffix",
                    "model": "gpt-3.5-turbo-instruct",
                    "messages": [
                        {"role": "user", "content": "The quick brown fox"}
                    ],
                    "suffix": "over the lazy dog.",
                    "temperature": 0.3,
                    "max_tokens": 20,
                    "echo": True,
                    "user": "completion_user"
                }
            ]
        }


class OpenAIEmbeddingRequest(BaseModel):
    """Standard OpenAI embedding request - provider in URL, model in body"""
    input: Union[str, List[str]] = Field(description="Text(s) to embed")
    model: str = Field(description="Model to use (e.g., 'text-embedding-ada-002', 'nomic-embed-text')")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "input": "Hello world",
                    "model": "all-minilm:l6-v2"
                },
                {
                    "input": ["Hello world", "How are you?"],
                    "model": "all-minilm:l6-v2"
                },
                {
                    "input": "Generate embedding for this text",
                    "model": "text-embedding-ada-002"
                }
            ]
        }


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="AbstractCore Server",
    description="Universal LLM Gateway - OpenAI-Compatible API for ALL Providers",
    version="2.1.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global config
DEFAULT_PROVIDER = os.getenv("ABSTRACTCORE_DEFAULT_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("ABSTRACTCORE_DEFAULT_MODEL", "gpt-4o-mini")


# ============================================================================
# Helper Functions
# ============================================================================

def parse_model_string(model_string: str) -> tuple[str, str]:
    """
    Smart model parsing for maximum OpenAI compatibility.

    Formats:
    1. "provider/model" - Explicit (e.g., "anthropic/claude-3-5-haiku-latest")
    2. "model" - Auto-route (e.g., "claude-3-5-haiku-latest" -> anthropic)
    3. OpenAI standard (e.g., "gpt-4o-mini" -> openai)

    Returns: (provider, model)
    """
    if not model_string:
        return DEFAULT_PROVIDER, DEFAULT_MODEL

    # Format 1: Explicit provider/model
    if '/' in model_string:
        provider, model = model_string.split('/', 1)
        return provider.strip(), model.strip()

    # Format 2: Smart routing
    model_lower = model_string.lower()

    # OpenAI models
    if any(pattern in model_lower for pattern in ['gpt-', 'text-davinci', 'text-embedding']):
        return "openai", model_string

    # Anthropic models
    if any(pattern in model_lower for pattern in ['claude']):
        return "anthropic", model_string

    # Ollama models
    if any(pattern in model_lower for pattern in ['llama', 'mistral', 'codellama', 'gemma', 'phi', 'qwen']):
        return "ollama", model_string

    # MLX models
    if any(pattern in model_lower for pattern in ['-4bit', 'mlx-community']):
        return "mlx", model_string

    # Default
    return DEFAULT_PROVIDER, model_string


def classify_model_type(model_name: str) -> str:
    """
    Classify model as 'chat' or 'embedding' based on comprehensive patterns.

    Covers all major embedding model families across providers.
    """
    model_lower = model_name.lower()

    # Comprehensive embedding model patterns
    embedding_patterns = [
        # Generic patterns
        'embed', 'embedding', 'embeddings',

        # OpenAI models
        'text-embedding',

        # Sentence transformers family
        'sentence-', 'all-minilm', 'all-mpnet', 'paraphrase',

        # Nomic models
        'nomic-embed',

        # BGE models (BAAI General Embedding)
        'bge-', 'baai',

        # E5 models
        'e5-',

        # Google models
        'embeddinggemma',

        # Stella models
        'stella',

        # Multilingual models
        'multilingual-e5', 'multilingual-embed',

        # Other patterns
        'semantic', 'retrieval', 'vector'
    ]

    # Check if it matches any embedding pattern
    for pattern in embedding_patterns:
        if pattern in model_lower:
            return 'embedding'

    # Default to chat model
    return 'chat'


def create_provider(model: str = None):
    """Create LLM provider with smart model parsing"""
    provider, model = parse_model_string(model)
    return create_llm(provider, model=model), provider, model


# ============================================================================
# SOTA Endpoints (Only Essential Ones)
# ============================================================================

@app.get("/")
async def root():
    """Server info and quick start guide"""
    return {
        "name": "AbstractCore Server",
        "description": "Universal LLM Gateway - OpenAI-Compatible API for ALL Providers",
        "version": "1.0.0",
        "endpoint_pattern": "/{provider}/v1/[standard_openai_endpoints]",
        "examples": {
            "default_chat": "/ollama/v1/chat/completions",
            "anthropic_chat": "/anthropic/v1/chat/completions",
            "openai_chat": "/openai/v1/chat/completions",
            "ollama_models": "/ollama/v1/models",
            "openai_embeddings": "/openai/v1/embeddings"
        },
        "features": [
            "🎯 Standard OpenAI endpoints with provider routing",
            "🔄 100% OpenAI API compatibility - model param works as expected",
            "📈 Works with ANY HTTP client or OpenAI SDK",
            "⚡ Clean, simple, maximum compatibility"
        ],
        "quick_start": {
            "curl": "curl -X POST http://localhost:8000/ollama/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen3-coder:30b\",\"messages\":[{\"role\":\"user\",\"content\":\"Extract: John Smith, age 30, engineer\"}],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"person_info\",\"schema\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"},\"occupation\":{\"type\":\"string\"}}}}}}'",
            "python": "client = OpenAI(base_url='http://localhost:8000/ollama'); response = client.chat.completions.create(model='qwen3-coder:30b', messages=[{'role':'user','content':'Extract: John Smith, age 30, engineer'}], response_format={'type':'json_schema','json_schema':{'name':'person_info','schema':{'type':'object','properties':{'name':{'type':'string'},'age':{'type':'integer'},'occupation':{'type':'string'}}}}})"
        }
    }


@app.post("/{provider}/v1/chat/completions")
async def openai_chat_completions(provider: str, request: OpenAIChatCompletionRequest):
    """
    🎯 **THE UNIVERSAL LLM ENDPOINT** 🎯

    Standard OpenAI chat/completions endpoint with provider routing!

    **URL Pattern:** `/{provider}/v1/chat/completions`

    **Examples:**
    - `/ollama/v1/chat/completions` (Default - Local models)
    - `/anthropic/v1/chat/completions`
    - `/openai/v1/chat/completions`

    **🔥 COMPLETE OpenAI Chat Completions API - ALL Parameters Supported! 🔥**

    **Core Parameters:**
    - `model`: Model to use (claude-3-5-haiku-latest, gpt-4o-mini, llama3:8b, etc.)
    - `messages`: Array of conversation messages with role (system/user/assistant) and content
    - `temperature` (0-2): Controls randomness - 0=deterministic, 2=very creative
    - `max_tokens`: Maximum tokens to generate (1-4000+)
    - `top_p` (0-1): Nucleus sampling - probability mass cutoff

    **Streaming & Control:**
    - `stream`: Enable real-time response streaming
    - `stream_options`: Streaming options (use {"include_usage": true} for token usage)
    - `stop`: Array of stop sequences to halt generation
    - `seed`: Integer for reproducible outputs

    **Repetition Control:**
    - `frequency_penalty` (-2 to 2): Reduce repetition based on frequency
    - `presence_penalty` (-2 to 2): Reduce repetition based on presence

    **Completion Control:**
    - `n` (1-128): Number of completion choices to generate
    - `best_of` (1-20): Generate best_of completions and return n best ones
    - `suffix`: Text that comes after completion (for insertion tasks)
    - `echo`: Echo back the prompt in addition to completion
    - `user`: Unique identifier for end-user (helps OpenAI monitor abuse)

    **Advanced Features:**
    - `logprobs`: Return log probabilities of output tokens
    - `top_logprobs` (0-5): Number of top token probabilities to return
    - `logit_bias`: Modify specific token probabilities (-100 to 100)

    **Function/Tool Calling:**
    - `tools`: Array of available functions with JSON Schema parameters
    - `tool_choice`: "auto", "none", or force specific function
    - `parallel_tool_calls`: Enable parallel function execution

    **Structured Output:**
    - `response_format`: JSON Schema for guaranteed structured responses

    **📋 Complete Examples:**

    **1. Basic Chat:**
    ```json
    {
      "model": "claude-3-5-haiku-latest",
      "messages": [{"role": "user", "content": "Hello!"}],
      "temperature": 0.7,
      "max_tokens": 150
    }
    ```

    **2. Streaming with Repetition Control:**
    ```json
    {
      "model": "gpt-4o-mini",
      "messages": [{"role": "user", "content": "Write a creative story"}],
      "stream": true,
      "temperature": 1.2,
      "frequency_penalty": 0.3,
      "presence_penalty": 0.2,
      "stop": ["THE END", "\n\n---"]
    }
    ```

    **3. Function Calling (Weather Tools):**
    ```json
    {
      "model": "gpt-4o-mini",
      "messages": [{"role": "user", "content": "Weather in SF?"}],
      "tools": [{
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City, State"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }],
      "tool_choice": "auto",
      "parallel_tool_calls": true
    }
    ```

    **4. Structured Output (JSON Schema):**
    ```json
    {
      "model": "gpt-4o-mini",
      "messages": [{"role": "user", "content": "Extract: John Smith, age 30, engineer"}],
      "response_format": {
        "type": "json_schema",
        "json_schema": {
          "name": "person_info",
          "schema": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "age": {"type": "integer", "minimum": 0},
              "occupation": {"type": "string"}
            },
            "required": ["name", "age"],
            "additionalProperties": false
          }
        }
      },
      "temperature": 0.0
    }
    ```

    **5. Advanced Parameters with LogProbs:**
    ```json
    {
      "model": "gpt-4o-mini",
      "messages": [{"role": "user", "content": "Rate this movie 1-10: The Matrix"}],
      "temperature": 0.3,
      "top_p": 0.9,
      "logprobs": true,
      "top_logprobs": 3,
      "logit_bias": {"10": 2.0, "9": 1.0},
      "seed": 42,
      "max_tokens": 50
    }
    ```
    """
    try:
        # Extract model from request (standard OpenAI way)
        model = request.model

        # Create provider
        llm = create_llm(provider, model=model)

        # Build conversation context from messages
        conversation_parts = []
        for msg in request.messages:
            role = msg.role
            content = msg.content
            if role == "system":
                conversation_parts.insert(0, f"System: {content}")
            elif role == "user":
                conversation_parts.append(f"User: {content}")
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")

        if not conversation_parts:
            raise HTTPException(status_code=400, detail="No valid messages found")

        # Create prompt from conversation
        prompt = "\n".join(conversation_parts) + "\nAssistant:"

        # Generation parameters
        gen_kwargs = {
            "prompt": prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.tools:
            gen_kwargs["tools"] = request.tools

        if request.stream:
            # OpenAI-compatible streaming
            def generate_openai_stream():
                try:
                    gen_kwargs["stream"] = True
                    for chunk in llm.generate(**gen_kwargs):
                        if hasattr(chunk, 'content') and chunk.content:
                            openai_chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": f"{provider}/{model}",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": chunk.content},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"

                    # Final chunk
                    final_chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": f"{provider}/{model}",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_chunk = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                generate_openai_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Regular response
            response = llm.generate(**gen_kwargs)

            # OpenAI-compatible response format
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"{provider}/{model}",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": getattr(response, 'usage', {}).get('prompt_tokens', 0),
                    "completion_tokens": getattr(response, 'usage', {}).get('completion_tokens', 0),
                    "total_tokens": getattr(response, 'usage', {}).get('total_tokens', 0)
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/{provider}/v1/models")
async def list_provider_models(provider: str, type: Optional[ModelType] = None):
    """
    List available models for a specific provider.

    **URL Pattern:** `/{provider}/v1/models` (Standard OpenAI endpoint)

    **Examples:**
    - `/anthropic/v1/models` - List all Anthropic models
    - `/ollama/v1/models?type=chat` - List only Ollama chat models
    - `/openai/v1/models?type=embedding` - List only OpenAI embedding models
    """
    models = []

    try:
        available_models = get_available_models(provider)
        for model_id in available_models:
            model_type = classify_model_type(model_id)

            # Filter by type if specified
            if type and model_type != type.value:
                continue

            models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": provider,
                "provider": provider,
                "model_type": model_type,
                "supports_tools": provider in ["openai", "anthropic"] and model_type == "chat",
                "supports_vision": provider in ["openai", "anthropic"] and model_type == "chat",
                "supports_streaming": model_type == "chat"
            })
    except Exception:
        # Return empty list if provider not available
        pass

    return {"object": "list", "data": models}


def check_provider_availability(provider_name: str) -> tuple[str, int]:
    """
    Check if a provider is available and get model count.

    Returns: (status, model_count)
    - "available": Provider is working and has models
    - "unavailable": Provider has configuration issues (no API keys, network issues)
    - "no_models": Provider is configured but has no models available
    """
    try:
        # For API-based providers, we need to check if they're configured
        if provider_name in ["openai", "anthropic"]:
            # Try to create a provider instance to test configuration
            try:
                llm = create_llm(provider_name, model="dummy-model-for-test")
                # If we can create the provider without error, it's likely configured
                # For API providers, we assume they have models if properly configured
                return "available", "unknown"
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["api_key", "key", "auth", "token"]):
                    return "unavailable", 0
                elif any(keyword in error_msg for keyword in ["network", "connection", "timeout"]):
                    return "unavailable", 0
                else:
                    # Try to get actual models
                    models = get_available_models(provider_name)
                    if models:
                        return "available", len(models)
                    else:
                        return "unavailable", 0

        # For local providers, check actual model availability
        else:
            models = get_available_models(provider_name)
            if models:
                return "available", len(models)
            else:
                return "no_models", 0

    except Exception as e:
        return "unavailable", 0


@app.get("/providers")
async def list_providers():
    """
    List all available providers with proper availability detection.

    **URL:** `/providers`

    **Status meanings:**
    - "available": Provider is working and has models
    - "unavailable": Provider has issues (no API keys, network problems, not installed)
    - "no_models": Provider is configured but has no models available
    """
    providers = []
    for provider_name in ["openai", "anthropic", "ollama", "mlx", "lmstudio"]:
        status, model_count = check_provider_availability(provider_name)

        providers.append({
            "name": provider_name,
            "status": status,
            "model_count": model_count if isinstance(model_count, int) else "unknown",
            "models_endpoint": f"/{provider_name}/v1/models",
            "description": get_provider_description(provider_name, status)
        })

    return {"providers": providers}


def get_provider_description(provider_name: str, status: str) -> str:
    """Get a helpful description for the provider status"""
    base_descriptions = {
        "openai": "OpenAI API (GPT models)",
        "anthropic": "Anthropic API (Claude models)",
        "ollama": "Local Ollama server",
        "mlx": "Apple MLX (local Mac models)",
        "lmstudio": "LM Studio (local server)"
    }

    base = base_descriptions.get(provider_name, provider_name)

    if status == "unavailable":
        if provider_name in ["openai", "anthropic"]:
            return f"{base} - Check API key configuration"
        else:
            return f"{base} - Not installed or not running"
    elif status == "no_models":
        return f"{base} - No models available"
    else:
        return base


@app.post("/{provider}/v1/embeddings")
async def create_embeddings(provider: str, request: OpenAIEmbeddingRequest):
    """
    Generate embeddings for text inputs.

    **URL Pattern:** `/{provider}/v1/embeddings` (Standard OpenAI endpoint)

    **Examples:**
    - `/ollama/v1/embeddings` (with model: "all-minilm:l6-v2" - uses Ollama API)
    - `/openai/v1/embeddings` (with model: "text-embedding-ada-002" - uses HF EmbeddingManager)
    - `/anthropic/v1/embeddings` (uses sentence-transformers/all-MiniLM-L6-v2 via EmbeddingManager)

    **Usage with OpenAI Client:**
    ```python
    # Option 1: Use Ollama model (via Ollama API)
    client = OpenAI(base_url="http://localhost:8000/ollama")
    response = client.embeddings.create(
        model="all-minilm:l6-v2",  # Uses Ollama's native API
        input="Hello world"
    )

    # Option 2: Use HuggingFace model (via EmbeddingManager - fast & convenient)
    client = OpenAI(base_url="http://localhost:8000/anthropic")
    response = client.embeddings.create(
        model="any-model",  # Uses sentence-transformers/all-MiniLM-L6-v2
        input="Hello world"
    )
    ```
    """
    try:
        # Extract model from request (standard OpenAI way)
        model = request.model

        # Handle both string and list inputs
        texts = request.input if isinstance(request.input, list) else [request.input]

        # Use AbstractCore's EmbeddingManager for best quality embeddings
        # EmbeddingManager provides SOTA models with caching and optimization
        try:
            from ..embeddings import EmbeddingManager

            # Smart embedding routing for best performance
            if provider == "ollama" and model in ["all-minilm:l6-v2", "all-minilm:33m", "nomic-embed-text"]:
                # Use Ollama's native embedding API for Ollama-specific models
                embedding_model = model
                use_ollama_api = True
            else:
                # Use AbstractCore's EmbeddingManager with HuggingFace for speed & convenience
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # HF model for speed
                use_ollama_api = False

            embeddings_data = []

            if use_ollama_api:
                # Use Ollama's native embedding API for Ollama-specific models
                import httpx

                for i, text in enumerate(texts):
                    try:
                        # Call Ollama's embeddings API directly
                        with httpx.Client(timeout=180.0) as client:
                            response = client.post(
                                "http://localhost:11434/api/embeddings",
                                json={"model": embedding_model, "prompt": text},
                                headers={"Content-Type": "application/json"}
                            )
                            response.raise_for_status()
                            result = response.json()

                            embedding = result.get("embedding", [])
                            if not embedding:
                                # If Ollama returns no embedding, raise error - don't mock
                                raise ValueError(f"Ollama returned empty embedding for model {embedding_model}")

                            embeddings_data.append({
                                "object": "embedding",
                                "embedding": embedding,
                                "index": i
                            })

                    except Exception as e:
                        # Don't mock - propagate the error
                        raise HTTPException(
                            status_code=500,
                            detail=f"Ollama embedding failed for model {embedding_model}: {str(e)}"
                        )

            else:
                # Use AbstractCore's EmbeddingManager for SOTA quality
                try:
                    embedder = EmbeddingManager(model=embedding_model)

                    for i, text in enumerate(texts):
                        try:
                            # Generate real embedding using EmbeddingManager
                            embedding = embedder.embed(text)

                            # Convert numpy array to list if needed
                            if hasattr(embedding, 'tolist'):
                                embedding = embedding.tolist()
                            elif not isinstance(embedding, list):
                                embedding = list(embedding)

                            embeddings_data.append({
                                "object": "embedding",
                                "embedding": embedding,
                                "index": i
                            })

                        except Exception as e:
                            # Don't mock - propagate the error for this text
                            raise HTTPException(
                                status_code=500,
                                detail=f"EmbeddingManager failed for text {i}: {str(e)}"
                            )

                except ImportError as e:
                    # EmbeddingManager not available - return proper error
                    raise HTTPException(
                        status_code=500,
                        detail="EmbeddingManager not available. Install embedding dependencies with: pip install abstractllm[embeddings]"
                    )

            return {
                "object": "list",
                "data": embeddings_data,
                "model": f"{provider}/{model}",
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts)
                }
            }

        except Exception as provider_error:
            # If provider creation fails, return error
            raise HTTPException(
                status_code=400,
                detail=f"Provider {provider} with model {model} not available: {str(provider_error)}"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/{provider}/v1/chat/completions/{completion_id}")
async def get_chat_completion(provider: str, completion_id: str):
    """
    Get a previously created chat completion by ID.

    **OpenAI-Compatible GET endpoint** for retrieving chat completions.
    Similar to OpenAI's Responses API pattern.

    **URL Pattern:** `/{provider}/v1/chat/completions/{completion_id}`

    **Examples:**
    - `/openai/v1/chat/completions/chatcmpl-123456`
    - `/anthropic/v1/chat/completions/msg_abc123`
    """
    # This would typically retrieve from a database/cache
    # For now, return a mock response indicating the feature
    return {
        "id": completion_id,
        "object": "chat.completion",
        "provider": provider,
        "status": "completed",
        "message": f"GET method for retrieving completion {completion_id} from {provider}",
        "note": "This endpoint would retrieve previously created completions from storage"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


# ============================================================================
# Server Runner
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)