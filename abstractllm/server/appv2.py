"""
AbstractCore Server V2 - Clean Architecture with Universal Tool Call Syntax Support

A focused FastAPI server that provides OpenAI-compatible endpoints with support for
multiple agent formats through the enhanced syntax rewriter.

Key Features:
- Universal tool call syntax conversion (OpenAI, Codex, Qwen3, LLaMA3, custom)
- Auto-detection of target agent format
- Clean delegation to AbstractCore
- Proper ReAct loop support
- Comprehensive model listing from AbstractCore providers
"""

import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Literal, Union, Iterator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.factory import create_llm
from ..utils.structured_logging import get_logger, configure_logging
from ..utils.simple_model_discovery import get_available_models
from ..tools.syntax_rewriter import (
    ToolCallSyntaxRewriter,
    SyntaxFormat,
    auto_detect_format,
    create_openai_rewriter,
    create_codex_rewriter,
    create_passthrough_rewriter
)

# ============================================================================
# Configuration
# ============================================================================

# Configure structured logging
debug_mode = os.getenv("ABSTRACTCORE_DEBUG", "false").lower() == "true"
configure_logging(
    console_level="DEBUG" if debug_mode else "INFO",
    file_level="DEBUG",
    log_dir="logs",
    verbatim_enabled=True,
    console_json=False,
    file_json=True
)

# Create FastAPI app
app = FastAPI(
    title="AbstractCore Server V2",
    description="Universal LLM Gateway with Multi-Agent Tool Call Syntax Support",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get logger
logger = get_logger("appv2")
logger.info("🚀 AbstractCore Server V2 Starting", version="2.0.0", debug_mode=debug_mode)

# ============================================================================
# Models
# ============================================================================

class ChatMessage(BaseModel):
    """OpenAI-compatible message format"""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(description="Model identifier (provider/model or auto-detected)")
    messages: List[ChatMessage] = Field(description="Conversation messages")

    # Core parameters
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = Field(default=False)

    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default="auto")

    # Other OpenAI parameters
    stop: Optional[List[str]] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

    # Agent format control (AppV2 feature)
    agent_format: Optional[str] = Field(
        default=None,
        description="Target agent format: 'auto', 'openai', 'codex', 'qwen3', 'llama3', 'passthrough'"
    )

# ============================================================================
# Helper Functions
# ============================================================================

def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse model string to extract provider and model."""
    if not model_string:
        return "ollama", "qwen3-coder:30b"  # Default

    # Explicit provider/model format
    if '/' in model_string:
        parts = model_string.split('/', 1)
        provider = parts[0].strip()
        model = parts[1].strip()
        return provider, model

    # Auto-detect provider from model name
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
        return "ollama", model_string  # Default

def detect_target_format(
    model: str,
    request: ChatCompletionRequest,
    http_request: Request
) -> SyntaxFormat:
    """
    Detect the target format for tool call syntax conversion.

    Args:
        model: Model identifier
        request: Chat completion request
        http_request: HTTP request object

    Returns:
        Target syntax format
    """
    # Explicit format override
    if request.agent_format:
        try:
            return SyntaxFormat(request.agent_format.lower())
        except ValueError:
            logger.warning(f"Invalid agent_format '{request.agent_format}', using auto-detection")

    # Auto-detect from headers and model
    user_agent = http_request.headers.get("user-agent", "")
    return auto_detect_format(model, user_agent)

def convert_to_abstractcore_messages(openai_messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert OpenAI messages to AbstractCore format."""
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

def create_syntax_rewriter(target_format: SyntaxFormat, model_name: str) -> ToolCallSyntaxRewriter:
    """Create appropriate syntax rewriter for target format."""
    if target_format == SyntaxFormat.PASSTHROUGH:
        return create_passthrough_rewriter()
    elif target_format == SyntaxFormat.CODEX:
        return create_codex_rewriter(model_name)
    elif target_format == SyntaxFormat.OPENAI:
        return create_openai_rewriter(model_name)
    else:
        return ToolCallSyntaxRewriter(target_format, model_name=model_name)

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": [
            "multi-agent-syntax-support",
            "auto-format-detection",
            "universal-tool-calls",
            "abstractcore-integration"
        ]
    }

@app.get("/v1/models")
async def list_models():
    """List available models from all AbstractCore providers."""
    try:
        # Get models from AbstractCore's discovery system
        available_models = get_available_models()

        # Convert to OpenAI format
        models_data = []
        for provider, models in available_models.items():
            for model in models:
                model_id = f"{provider}/{model}" if provider != "openai" else model
                models_data.append({
                    "id": model_id,
                    "object": "model",
                    "owned_by": provider,
                    "created": int(time.time()),
                    "permission": [{"allow_create_engine": False, "allow_sampling": True}]
                })

        return {
            "object": "list",
            "data": sorted(models_data, key=lambda x: x["id"])
        }

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        # Fallback to basic model list
        return {
            "object": "list",
            "data": [
                {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
                {"id": "claude-3-5-haiku-latest", "object": "model", "owned_by": "anthropic"},
                {"id": "qwen3-coder:30b", "object": "model", "owned_by": "ollama"},
                {"id": "lmstudio/qwen/qwen3-next-80b", "object": "model", "owned_by": "lmstudio"},
            ]
        }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """Standard OpenAI chat completions endpoint with multi-agent support."""
    provider, model = parse_model_string(request.model)
    return await process_chat_completion(provider, model, request, http_request)

@app.post("/{provider}/v1/chat/completions")
async def provider_chat_completions(
    provider: str,
    request: ChatCompletionRequest,
    http_request: Request
):
    """Provider-specific chat completions endpoint."""
    _, model = parse_model_string(request.model)
    return await process_chat_completion(provider, model, request, http_request)

async def process_chat_completion(
    provider: str,
    model: str,
    request: ChatCompletionRequest,
    http_request: Request
):
    """
    Core chat completion processing with syntax rewriting support.
    """
    request_id = uuid.uuid4().hex[:8]

    try:
        logger.info(
            "📥 Chat Completion Request",
            request_id=request_id,
            provider=provider,
            model=model,
            messages=len(request.messages),
            has_tools=bool(request.tools),
            stream=request.stream
        )

        # Detect target format for tool call syntax
        target_format = detect_target_format(f"{provider}/{model}", request, http_request)
        logger.info(
            "🎯 Target Format Detected",
            request_id=request_id,
            target_format=target_format.value,
            user_agent=http_request.headers.get("user-agent", "")[:50]
        )

        # Create LLM instance
        llm = create_llm(provider, model=model)

        # Convert messages
        messages = convert_to_abstractcore_messages(request.messages)

        # Create syntax rewriter
        syntax_rewriter = create_syntax_rewriter(target_format, f"{provider}/{model}")

        # Prepare generation parameters
        gen_kwargs = {
            "prompt": "",  # Empty when using messages
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "tools": request.tools,
            "tool_choice": request.tool_choice if request.tools else None,
            "execute_tools": False,  # Server mode - don't execute tools
        }

        # Add optional parameters
        if request.stop:
            gen_kwargs["stop"] = request.stop
        if request.seed:
            gen_kwargs["seed"] = request.seed
        if request.frequency_penalty:
            gen_kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty:
            gen_kwargs["presence_penalty"] = request.presence_penalty

        # Generate response
        if request.stream:
            return StreamingResponse(
                generate_streaming_response(
                    llm, gen_kwargs, provider, model, syntax_rewriter, request_id
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            response = llm.generate(**gen_kwargs)
            return convert_to_openai_response(
                response, provider, model, syntax_rewriter, request_id
            )

    except Exception as e:
        logger.error(
            "❌ Chat completion failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "server_error"}}
        )

def generate_streaming_response(
    llm,
    gen_kwargs: Dict[str, Any],
    provider: str,
    model: str,
    syntax_rewriter: ToolCallSyntaxRewriter,
    request_id: str
) -> Iterator[str]:
    """Generate OpenAI-compatible streaming response with syntax rewriting."""
    try:
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_time = int(time.time())
        has_tool_calls = False

        for chunk in llm.generate(**gen_kwargs):
            # Content streaming
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content

                # For OpenAI/Codex format: clean content (remove tool call syntax)
                # For other formats: apply syntax rewriting
                if syntax_rewriter.target_format in [SyntaxFormat.OPENAI, SyntaxFormat.CODEX]:
                    # Clean content - remove tool call syntax entirely
                    content = syntax_rewriter.remove_tool_call_patterns(content)
                elif syntax_rewriter.target_format != SyntaxFormat.PASSTHROUGH:
                    # Apply format-specific rewriting for non-OpenAI formats
                    content = syntax_rewriter.rewrite_content(content)

                # Only send content if it's meaningful (not just whitespace)
                if content.strip():
                    openai_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": f"{provider}/{model}",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(openai_chunk)}\n\n"

            # Tool calls - always convert to OpenAI format for streaming
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                has_tool_calls = True
                openai_tool_calls = syntax_rewriter.convert_to_openai_format(chunk.tool_calls)

                for i, openai_tool_call in enumerate(openai_tool_calls):
                    tool_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": f"{provider}/{model}",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": i,  # Proper indexing for multiple tools
                                    "id": openai_tool_call["id"],
                                    "type": "function",
                                    "function": openai_tool_call["function"]
                                }]
                            },
                            "finish_reason": "tool_calls"  # Critical for Codex
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
                "finish_reason": "tool_calls" if has_tool_calls else "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(
            "✅ Streaming completed",
            request_id=request_id,
            has_tool_calls=has_tool_calls
        )

    except Exception as e:
        logger.error(
            "❌ Streaming failed",
            request_id=request_id,
            error=str(e)
        )
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"

def convert_to_openai_response(
    response,
    provider: str,
    model: str,
    syntax_rewriter: ToolCallSyntaxRewriter,
    request_id: str
) -> Dict[str, Any]:
    """Convert AbstractCore response to OpenAI format with syntax rewriting."""

    if response is None:
        logger.warning("Received None response", request_id=request_id)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"{provider}/{model}",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Error: No response generated"},
                "finish_reason": "error"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # Apply syntax rewriting to content
    content = response.content if hasattr(response, 'content') else str(response)

    # For OpenAI/Codex format: clean content (remove tool call syntax)
    # For other formats: apply syntax rewriting
    if syntax_rewriter.target_format in [SyntaxFormat.OPENAI, SyntaxFormat.CODEX]:
        # Clean content - remove tool call syntax entirely
        content = syntax_rewriter.remove_tool_call_patterns(content)
    elif syntax_rewriter.target_format != SyntaxFormat.PASSTHROUGH:
        # Apply format-specific rewriting for non-OpenAI formats
        content = syntax_rewriter.rewrite_content(content)

    response_dict = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"{provider}/{model}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
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
        openai_tool_calls = syntax_rewriter.convert_to_openai_format(response.tool_calls)
        response_dict["choices"][0]["message"]["tool_calls"] = openai_tool_calls
        response_dict["choices"][0]["finish_reason"] = "tool_calls"

        logger.info(
            "🔧 Tool calls converted",
            request_id=request_id,
            tool_count=len(response.tool_calls),
            target_format=syntax_rewriter.target_format.value
        )

    return response_dict

# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(
        "🚀 Starting AbstractCore Server V2",
        host=host,
        port=port,
        debug=debug_mode
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if debug_mode else "info"
    )