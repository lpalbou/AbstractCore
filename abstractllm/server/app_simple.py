"""
AbstractCore Server - Simplified Clean Implementation

A focused FastAPI server providing only essential OpenAI-compatible endpoints.
Delegates all business logic to AbstractCore - no reimplementation of existing functionality.
"""

import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Literal, Union, Iterator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.factory import create_llm
from ..utils.structured_logging import get_logger, configure_logging

# ============================================================================
# Configuration
# ============================================================================

# Configure structured logging
configure_logging(
    console_level="INFO",
    file_level="DEBUG",
    log_dir="logs",
    verbatim_enabled=True,
    console_json=False,
    file_json=True
)

# Create FastAPI app
app = FastAPI(
    title="AbstractCore Server",
    description="Universal LLM Gateway - OpenAI-Compatible API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global config (following same pattern as original app.py)
DEFAULT_TOOL_CALL_TAGS = os.getenv("ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS", None)

# Get logger
logger = get_logger("server")
logger.info("🚀 AbstractCore Server Starting - Simplified Architecture v3.0.0")

# ============================================================================
# Models
# ============================================================================

class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible message format"""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None

class OpenAIChatCompletionRequest(BaseModel):
    """Standard OpenAI chat completion request"""
    model: str
    messages: List[OpenAIChatMessage]

    # Core parameters
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"

    # Structured output
    response_format: Optional[Dict[str, Any]] = None

    # Tool call tag rewriting (AbstractCore feature)
    tool_call_tags: Optional[str] = None

    # Other OpenAI parameters (passed through as kwargs)
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0

# ============================================================================
# Helper Functions
# ============================================================================

def parse_model_string(model_string: str) -> tuple[str, str]:
    """
    Parse model string to extract provider and model.

    Formats:
    - "ollama/qwen3-coder:30b" -> ("ollama", "qwen3-coder:30b")
    - "lmstudio/qwen/qwen3-next-80b" -> ("lmstudio", "qwen/qwen3-next-80b")
    - "anthropic/claude-3-5-haiku-latest" -> ("anthropic", "claude-3-5-haiku-latest")
    - "gpt-4o-mini" -> ("openai", "gpt-4o-mini") [auto-detected]
    - "qwen3-coder:30b" -> ("ollama", "qwen3-coder:30b") [auto-detected]
    """
    if not model_string:
        return "ollama", "qwen3-coder:30b"  # Default

    # Explicit provider/model format - split on FIRST "/" only
    if '/' in model_string:
        parts = model_string.split('/', 1)  # Split on first "/" only
        provider = parts[0].strip()
        model = parts[1].strip()
        return provider, model

    # Auto-detect provider from model name (no "/" present)
    model_lower = model_string.lower()

    if any(pattern in model_lower for pattern in ['gpt-', 'text-davinci', 'text-embedding']):
        return "openai", model_string
    elif any(pattern in model_lower for pattern in ['claude']):
        return "anthropic", model_string
    elif any(pattern in model_lower for pattern in ['llama', 'mistral', 'gemma', 'phi', 'qwen']):
        return "ollama", model_string
    elif any(pattern in model_lower for pattern in ['-4bit', 'mlx-community']):
        return "mlx", model_string
    else:
        return "ollama", model_string  # Default to ollama

def convert_to_abstractcore_messages(openai_messages: List[OpenAIChatMessage]) -> List[Dict[str, Any]]:
    """Convert OpenAI messages to AbstractCore format"""
    messages = []
    for msg in openai_messages:
        message_dict = {"role": msg.role}

        if msg.content is not None:
            message_dict["content"] = msg.content

        if msg.tool_calls:
            message_dict["tool_calls"] = msg.tool_calls

        if msg.tool_call_id:
            message_dict["tool_call_id"] = msg.tool_call_id

        if msg.name:
            message_dict["name"] = msg.name

        messages.append(message_dict)

    return messages

def convert_structured_output(response_format: Dict[str, Any]) -> Optional[Any]:
    """Convert OpenAI response_format to AbstractCore response_model"""
    if not response_format or response_format.get("type") != "json_schema":
        return None

    try:
        from pydantic import create_model
        from typing import Optional, List, Dict

        json_schema = response_format.get("json_schema", {})
        schema = json_schema.get("schema", {})

        if not schema.get("properties"):
            return None

        # Convert JSON schema to Pydantic fields
        fields = {}
        for prop_name, prop_def in schema["properties"].items():
            field_type = str  # Default

            if prop_def.get("type") == "integer":
                field_type = int
            elif prop_def.get("type") == "number":
                field_type = float
            elif prop_def.get("type") == "boolean":
                field_type = bool
            elif prop_def.get("type") == "array":
                field_type = List[str]
            elif prop_def.get("type") == "object":
                field_type = Dict

            # Check if required
            required = prop_name in schema.get("required", [])
            if required:
                fields[prop_name] = (field_type, ...)
            else:
                fields[prop_name] = (Optional[field_type], None)

        model_name = json_schema.get("name", "DynamicModel")
        return create_model(model_name, **fields)

    except Exception as e:
        logger.warning(f"Failed to convert structured output: {e}")
        return None

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "3.0.0"}

@app.get("/v1/models")
async def list_models():
    """List available models - simplified version"""
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
            {"id": "claude-3-5-haiku-latest", "object": "model", "owned_by": "anthropic"},
            {"id": "qwen3-coder:30b", "object": "model", "owned_by": "ollama"},
        ]
    }

@app.post("/v1/chat/completions")
async def standard_chat_completions(request: OpenAIChatCompletionRequest):
    """Standard OpenAI chat completions endpoint"""
    # Parse provider from model string
    provider, model = parse_model_string(request.model)
    return await chat_completions(provider, model, request)

@app.post("/{provider}/v1/chat/completions")
async def provider_chat_completions(provider: str, request: OpenAIChatCompletionRequest):
    """Provider-specific chat completions endpoint"""
    _, model = parse_model_string(request.model)
    return await chat_completions(provider, model, request)

async def chat_completions(provider: str, model: str, request: OpenAIChatCompletionRequest):
    """
    Core chat completions logic - delegates everything to AbstractCore
    """
    try:
        logger.info(f"📥 Chat Request | {provider}/{model} | messages={len(request.messages)} | tools={'YES' if request.tools else 'NO'}")

        # Create LLM instance
        llm = create_llm(provider, model=model)

        # Convert messages to AbstractCore format
        messages = convert_to_abstractcore_messages(request.messages)

        # Convert structured output if specified
        response_model = convert_structured_output(request.response_format) if request.response_format else None

        # Handle tool call tag rewriting (following same pattern as original app.py)
        tool_call_tags = None
        if request.tool_call_tags:
            tool_call_tags = request.tool_call_tags
        elif DEFAULT_TOOL_CALL_TAGS:
            tool_call_tags = DEFAULT_TOOL_CALL_TAGS

        # Prepare generation parameters - let AbstractCore handle everything
        gen_kwargs = {
            "prompt": "",  # Empty when using messages
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "tools": request.tools,
            "tool_choice": request.tool_choice if request.tools else None,
            "response_model": response_model,
            "tool_call_tags": tool_call_tags,
            "execute_tools": False,  # Server mode - don't execute tools
        }

        # Add other OpenAI parameters
        if request.stop:
            gen_kwargs["stop"] = request.stop
        if request.seed:
            gen_kwargs["seed"] = request.seed
        if request.frequency_penalty:
            gen_kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty:
            gen_kwargs["presence_penalty"] = request.presence_penalty

        # Generate response using AbstractCore
        if request.stream:
            return StreamingResponse(
                generate_openai_stream(llm, gen_kwargs, provider, model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            response = llm.generate_with_telemetry(**gen_kwargs)
            return convert_to_openai_response(response, provider, model, request)

    except Exception as e:
        logger.error(f"❌ Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})

def generate_openai_stream(llm, gen_kwargs: Dict[str, Any], provider: str, model: str) -> Iterator[str]:
    """Generate OpenAI-compatible streaming response"""
    try:
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_time = int(time.time())

        # Trust AbstractCore's unified streaming processor to handle everything
        for chunk in llm.generate_with_telemetry(**gen_kwargs):
            openai_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": f"{provider}/{model}",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }]
            }

            # Content streaming
            if hasattr(chunk, 'content') and chunk.content:
                openai_chunk["choices"][0]["delta"]["content"] = chunk.content
                yield f"data: {json.dumps(openai_chunk)}\n\n"

            # Tool calls - AbstractCore handles detection and formatting
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    tool_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": f"{provider}/{model}",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": 0,
                                    "id": tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.name,
                                        "arguments": json.dumps(tool_call.arguments) if isinstance(tool_call.arguments, dict) else str(tool_call.arguments)
                                    }
                                }]
                            },
                            "finish_reason": "tool_calls"  # Critical for Codex: must be "tool_calls" not None
                        }]
                    }
                    yield f"data: {json.dumps(tool_chunk)}\n\n"

        # Final chunk
        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
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
        logger.error(f"❌ Streaming failed: {e}")
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"

def convert_to_openai_response(response, provider: str, model: str, request: OpenAIChatCompletionRequest) -> Dict[str, Any]:
    """Convert AbstractCore response to OpenAI format"""
    # Handle None response case
    if response is None:
        response_dict = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"{provider}/{model}",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Error: No response generated"
                },
                "finish_reason": "error"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        return response_dict

    response_dict = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"{provider}/{model}",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.content if hasattr(response, 'content') else str(response)
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": getattr(response, 'usage', {}).get('prompt_tokens', 0) if hasattr(response, 'usage') else 0,
            "completion_tokens": getattr(response, 'usage', {}).get('completion_tokens', 0) if hasattr(response, 'usage') else 0,
            "total_tokens": getattr(response, 'usage', {}).get('total_tokens', 0) if hasattr(response, 'usage') else 0
        }
    }

    # Add tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        response_dict["choices"][0]["message"]["tool_calls"] = [
            {
                "id": tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments) if isinstance(tool_call.arguments, dict) else str(tool_call.arguments)
                }
            }
            for tool_call in response.tool_calls
        ]
        response_dict["choices"][0]["finish_reason"] = "tool_calls"

    return response_dict

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)