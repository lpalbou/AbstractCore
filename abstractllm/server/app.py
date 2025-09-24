"""
AbstractCore Server - FastAPI application

A universal LLM API gateway that provides OpenAI-compatible endpoints
for all AbstractCore providers (OpenAI, Anthropic, Ollama, MLX, etc.)
"""

import os
import json
import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response, Depends, status, Field
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import httpx

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    Message,
    Usage,
    ModelsResponse,
    ModelInfo,
    ModelType,
    ProviderInfo,
    ProviderConfig,
    SessionConfig,
    SessionInfo,
    ServerStatus,
    SimpleChatRequest,
    SimpleChatResponse,
    ErrorResponse,
    ToolRegistration,
    StructuredOutputRequest,
    ChatCompletionStreamChoice
)

from ..core.factory import create_llm
from ..core.session import BasicSession
from ..core.types import GenerateResponse
from ..exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderAPIError
)
from ..tools import get_registry, ToolDefinition
from ..events import EventType, on_global, emit_global
from ..utils.simple_model_discovery import (
    get_available_models,
    format_model_error
)

# Server configuration
DEFAULT_PROVIDER = os.getenv("ABSTRACTCORE_DEFAULT_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("ABSTRACTCORE_DEFAULT_MODEL", "gpt-4o-mini")
SERVER_VERSION = "1.0.0"

# Global state
sessions: Dict[str, BasicSession] = {}
providers_cache: Dict[str, Any] = {}
server_start_time = time.time()
request_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the FastAPI app
    """
    # Startup
    print(f"🚀 AbstractCore Server v{SERVER_VERSION} starting...")
    print(f"📦 Default provider: {DEFAULT_PROVIDER}")
    print(f"🤖 Default model: {DEFAULT_MODEL}")

    # Register event handlers for monitoring
    def log_generation(event):
        print(f"📝 Generation: {event.data.get('model')} - {event.data.get('tokens_output', 0)} tokens")

    on_global(EventType.GENERATION_COMPLETED, log_generation)

    yield

    # Shutdown
    print("👋 AbstractCore Server shutting down...")
    # Clean up sessions
    sessions.clear()


# Create FastAPI app
app = FastAPI(
    title="AbstractCore Server",
    description="Universal LLM API Gateway - OpenAI-compatible endpoints for all providers",
    version=SERVER_VERSION,
    lifespan=lifespan
)

# Add CORS middleware for web UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def classify_model_type(model_name: str) -> str:
    """
    Classify a model as 'chat' or 'embedding' based on naming patterns.

    Args:
        model_name: Name of the model

    Returns:
        'chat' for text generation models, 'embedding' for embedding models
    """
    model_lower = model_name.lower()

    # Embedding model patterns
    embedding_patterns = [
        'embed', 'embedding', 'embeddings',
        'nomic-embed', 'text-embedding',
        'embeddinggemma', 'e5-', 'bge-',
        'sentence-', 'all-minilm', 'paraphrase'
    ]

    # Check if it's an embedding model
    for pattern in embedding_patterns:
        if pattern in model_lower:
            return 'embedding'

    # Default to chat model
    return 'chat'


def convert_to_openai_message(msg: Dict[str, Any]) -> Message:
    """Convert AbstractCore message to OpenAI format"""
    return Message(
        role=msg.get("role", "assistant"),
        content=msg.get("content", ""),
        name=msg.get("name"),
        tool_calls=msg.get("tool_calls"),
        tool_call_id=msg.get("tool_call_id")
    )


def convert_from_openai_message(msg: Message) -> Dict[str, str]:
    """Convert OpenAI message to AbstractCore format"""
    result = {"role": msg.role, "content": msg.content}
    if msg.name:
        result["name"] = msg.name
    if msg.tool_calls:
        result["tool_calls"] = msg.tool_calls
    return result


async def generate_streaming_response(
    provider,
    prompt: str,
    messages: List[Dict[str, str]],
    request: ChatCompletionRequest
) -> AsyncIterator[str]:
    """
    Generate streaming response chunks in OpenAI format.
    Runs sync generation in thread pool for compatibility.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    # Convert tools if provided
    tools = None
    if request.tools:
        tools = []
        for tool in request.tools:
            tool_def = ToolDefinition(
                name=tool.function.name,
                description=tool.function.description,
                parameters=tool.function.parameters
            )
            tools.append(tool_def)

    # Run sync generation in thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        # Create generator in thread
        stream_gen = await loop.run_in_executor(
            executor,
            lambda: provider.generate(
                prompt=prompt,
                messages=messages,
                stream=True,
                tools=tools,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        )

        # Convert generator to list in thread
        chunks = await loop.run_in_executor(
            executor,
            lambda: list(stream_gen)
        )

    # Stream chunks
    full_content = ""
    for chunk in chunks:
        if hasattr(chunk, 'content') and chunk.content:
            full_content += chunk.content
            # Create streaming response
            stream_response = ChatCompletionStreamResponse(
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=Message(role="assistant", content=chunk.content),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {stream_response.json()}\n\n"

    # Send final chunk with finish reason
    final_response = ChatCompletionStreamResponse(
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=Message(role="assistant", content=""),
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {final_response.json()}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================================
# Simple Endpoints (Easy to Use)
# ============================================================================

@app.get("/v1/simple/chat", response_model=SimpleChatResponse)
async def simple_chat(
    message: str = Field(
        description="Your message to the AI",
        examples=["Hello world!", "What is Python?", "Write a haiku about code"]
    ),
    provider: str = Field(
        default="openai",
        description="AI provider to use",
        examples=["openai", "anthropic", "ollama"]
    ),
    model: str = Field(
        default="gpt-4o-mini",
        description="Model to use",
        examples=["gpt-4o-mini", "claude-3-5-haiku-latest", "llama3:8b"]
    ),
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Creativity level (0.1 = focused, 0.9 = creative)"
    ),
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=4000,
        description="Maximum response length"
    )
):
    """
    🚀 EASIEST way to chat with ANY AI - just use URL parameters!

    Perfect for:
    - Quick testing: Just click "Try it out" and modify the message
    - Simple integrations: One GET request, no JSON needed
    - Experimenting: Try different providers/models instantly

    Examples (click these in your browser):
    - Basic: /v1/simple/chat?message=Hello%20world
    - Use Claude: /v1/simple/chat?message=What%20is%20Python?&provider=anthropic&model=claude-3-5-haiku-latest
    - Use Ollama: /v1/simple/chat?message=Write%20code&provider=ollama&model=qwen3-coder:30b
    - More creative: /v1/simple/chat?message=Tell%20me%20a%20story&temperature=0.9
    """
    try:
        # Create provider
        llm = create_llm(provider, model=model)

        # Generate response
        response = llm.generate(
            prompt=message,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return SimpleChatResponse(
            message=message,
            response=response.content,
            provider=provider,
            model=model,
            settings={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": None
            },
            usage=response.usage if hasattr(response, 'usage') else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/{provider}/{model}/chat", response_model=SimpleChatResponse)
async def provider_model_chat(
    provider: str,
    model: str,
    request: SimpleChatRequest
):
    """
    Provider-specific chat endpoint with model in URL.
    Clean and simple - just specify your message and optional parameters!

    Examples:
    - POST /v1/openai/gpt-4o-mini/chat
    - POST /v1/anthropic/claude-3-5-haiku-latest/chat
    - POST /v1/ollama/llama3:8b/chat

    The request body will be pre-filled with working examples in the Swagger UI!
    """
    try:
        # Create provider
        llm = create_llm(provider, model=model)

        # Generate response (with optional system prompt)
        response = llm.generate(
            prompt=request.message,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return SimpleChatResponse(
            message=request.message,
            response=response.content,
            provider=provider,
            model=model,
            settings={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "system_prompt": request.system_prompt
            },
            usage=response.usage if hasattr(response, 'usage') else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/quick/test")
async def quick_test():
    """
    Quick endpoint to test if the server is working with available providers.
    Returns a test response from each available provider.
    """
    results = {}
    test_message = "Say 'Hello from [your provider name]' in one sentence."

    # Test each provider
    providers_to_test = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-latest"),
        ("ollama", "llama3:8b")
    ]

    for provider, model in providers_to_test:
        try:
            llm = create_llm(provider, model=model)
            response = llm.generate(test_message, max_tokens=50)
            results[provider] = {
                "status": "✅ working",
                "model": model,
                "response": response.content
            }
        except Exception as e:
            results[provider] = {
                "status": "❌ error",
                "model": model,
                "error": str(e)
            }

    return {
        "server_status": "AbstractCore Server is running!",
        "test_message": test_message,
        "provider_tests": results
    }


# ============================================================================
# OpenAI-Compatible Endpoints (Advanced Users)
# ============================================================================

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Works with ANY provider configured in AbstractCore.
    """
    global request_count
    request_count += 1

    # Determine provider and model
    provider_name = request.provider or DEFAULT_PROVIDER
    model_name = request.model

    # Special handling for model names with provider prefix (e.g., "anthropic/claude-3")
    if "/" in model_name and not request.provider:
        parts = model_name.split("/", 1)
        provider_name = parts[0]
        model_name = parts[1]

    try:
        # Create provider instance
        provider = create_llm(provider_name, model=model_name)

        # Convert messages
        messages = [convert_from_openai_message(msg) for msg in request.messages]

        # Extract system message if present
        system_prompt = None
        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        # Get the last user message as prompt
        prompt = ""
        if messages and messages[-1]["role"] == "user":
            prompt = messages[-1]["content"]
            messages = messages[:-1] if len(messages) > 1 else None

        # Handle streaming
        if request.stream:
            return StreamingResponse(
                generate_streaming_response(provider, prompt, messages, request),
                media_type="text/event-stream"
            )

        # Convert tools if provided
        tools = None
        if request.tools:
            tools = []
            for tool in request.tools:
                tool_def = ToolDefinition(
                    name=tool.function.name,
                    description=tool.function.description,
                    parameters=tool.function.parameters
                )
                tools.append(tool_def)

        # Generate response
        response = provider.generate(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Convert to OpenAI format
        message = Message(
            role="assistant",
            content=response.content
        )

        # Add tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            message.tool_calls = response.tool_calls

        # Calculate usage
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = Usage(
                prompt_tokens=response.usage.get('prompt_tokens', 0),
                completion_tokens=response.usage.get('completion_tokens', 0),
                total_tokens=response.usage.get('total_tokens', 0)
            )

        return ChatCompletionResponse(
            model=f"{provider_name}/{model_name}",
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason="stop"
                )
            ],
            usage=usage
        )

    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(provider: Optional[str] = None, type: Optional[ModelType] = None):
    """
    List available models from all providers or a specific provider.

    Args:
        provider: Filter by specific provider (openai, anthropic, ollama, etc.)
        type: Filter by model type ('chat', 'embedding', or None for all)

    Model Type Detection:
        - 'chat': Text generation models (default for /v1/chat/completions)
        - 'embedding': Vector embedding models (for /v1/embeddings)
        - None: All models (backwards compatible)
    """
    models = []

    if provider:
        # Get models for specific provider
        providers_to_check = [provider]
    else:
        # Get models for all providers
        providers_to_check = ["openai", "anthropic", "ollama", "lmstudio", "mlx", "huggingface"]

    for provider_name in providers_to_check:
        try:
            # Get API keys from environment
            api_key = None
            if provider_name == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider_name == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")

            # Discover available models
            available_models = get_available_models(provider_name, api_key=api_key)

            for model_id in available_models:
                # Classify model type
                model_type = classify_model_type(model_id)

                # Filter by type if specified
                if type and model_type != type.value:
                    continue

                models.append(
                    ModelInfo(
                        id=f"{provider_name}/{model_id}" if not provider else model_id,
                        owned_by=provider_name,
                        provider=provider_name,
                        model_type=model_type,
                        supports_tools=provider_name in ["openai", "anthropic"] and model_type == "chat",
                        supports_vision=provider_name in ["openai", "anthropic"] and model_type == "chat",
                        supports_streaming=model_type == "chat"
                    )
                )
        except Exception as e:
            # Log error but continue with other providers
            print(f"Error fetching models for {provider_name}: {e}")

    return ModelsResponse(data=models)


@app.get("/v1/embeddings/models", response_model=ModelsResponse)
async def list_embedding_models(provider: Optional[str] = None):
    """
    List only embedding models - convenience endpoint.
    Equivalent to /v1/models?type=embedding
    """
    return await list_models(provider=provider, type=ModelType.EMBEDDING)


# ============================================================================
# Provider Management Endpoints
# ============================================================================

@app.get("/v1/providers")
async def list_providers():
    """
    List all available providers and their status.
    """
    providers = []

    for provider_name in ["openai", "anthropic", "ollama", "lmstudio", "mlx", "huggingface"]:
        try:
            # Try to get models to check if provider is available
            api_key = None
            if provider_name == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider_name == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")

            available_models = get_available_models(provider_name, api_key=api_key)

            status = "healthy" if available_models else "unavailable"
            error = None if available_models else "No models found"

            providers.append(
                ProviderInfo(
                    name=provider_name,
                    status=status,
                    models=available_models[:10],  # Limit to first 10
                    capabilities={
                        "tools": provider_name in ["openai", "anthropic"],
                        "vision": provider_name in ["openai", "anthropic"],
                        "streaming": True,
                        "structured_output": True  # All support via AbstractCore
                    },
                    error=error
                )
            )
        except Exception as e:
            providers.append(
                ProviderInfo(
                    name=provider_name,
                    status="unavailable",
                    models=[],
                    capabilities={},
                    error=str(e)
                )
            )

    return {"providers": providers}


@app.get("/v1/providers/{provider}/models")
async def get_provider_models(provider: str):
    """
    Get all available models for a specific provider.
    """
    try:
        # Get API key if needed
        api_key = None
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

        models = get_available_models(provider, api_key=api_key)

        if not models:
            raise HTTPException(
                status_code=404,
                detail=f"No models found for provider '{provider}'"
            )

        return {
            "provider": provider,
            "models": models,
            "count": len(models)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/providers/test")
async def test_provider(config: ProviderConfig):
    """
    Test a provider configuration.
    """
    try:
        # Create provider with config
        provider = create_llm(
            config.name,
            api_key=config.api_key,
            base_url=config.base_url
        )

        # Try a simple generation
        response = provider.generate("Say 'test successful' if you can read this.")

        return {
            "status": "success",
            "provider": config.name,
            "response": response.content,
            "model": provider.model
        }

    except Exception as e:
        return {
            "status": "error",
            "provider": config.name,
            "error": str(e)
        }


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.post("/v1/sessions")
async def create_session(config: SessionConfig):
    """
    Create a new conversation session.
    """
    session_id = config.id or str(uuid.uuid4())

    try:
        # Create provider
        provider = create_llm(config.provider, model=config.model)

        # Create session
        session = BasicSession(provider=provider, system_prompt=config.system_prompt)
        sessions[session_id] = session

        return SessionInfo(
            id=session_id,
            provider=config.provider,
            model=config.model,
            created_at=datetime.now(),
            message_count=len(session.messages)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get session information.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return {
        "id": session_id,
        "messages": session.get_messages(),
        "message_count": len(session.messages)
    }


@app.post("/v1/sessions/{session_id}/chat")
async def session_chat(session_id: str, request: ChatCompletionRequest):
    """
    Chat using a session (maintains conversation history).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Get the last message as prompt
    prompt = request.messages[-1].content if request.messages else ""

    try:
        # Generate response using session
        response = session.generate(
            prompt,
            stream=request.stream,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Convert to OpenAI format
        message = Message(
            role="assistant",
            content=response.content
        )

        return ChatCompletionResponse(
            model=session.provider.model,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason="stop"
                )
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


# ============================================================================
# Tool Management Endpoints
# ============================================================================

@app.get("/v1/tools")
async def list_tools():
    """
    List all registered tools.
    """
    registry = get_registry()
    tools = []

    for tool_def in registry.list_tools():
        tools.append({
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters": tool_def.parameters
        })

    return {"tools": tools, "count": len(tools)}


@app.post("/v1/tools/register")
async def register_tool(registration: ToolRegistration):
    """
    Register a new tool dynamically.
    """
    try:
        # Create tool definition
        tool_def = ToolDefinition(
            name=registration.name,
            description=registration.description,
            parameters=registration.parameters
        )

        # Register in global registry
        registry = get_registry()
        registry.register(tool_def)

        return {
            "status": "registered",
            "tool": {
                "name": tool_def.name,
                "description": tool_def.description
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AbstractCore-Specific Features
# ============================================================================

@app.post("/v1/generate/structured")
async def generate_structured(request: StructuredOutputRequest):
    """
    Generate structured output using Pydantic models.
    """
    try:
        # Create provider
        provider_name = request.provider or DEFAULT_PROVIDER
        provider = create_llm(provider_name, model=request.model)

        # Convert JSON schema to Pydantic model dynamically
        from pydantic import create_model
        ResponseModel = create_model('ResponseModel', **request.response_model)

        # Generate with structured output
        result = provider.generate(
            prompt=request.prompt,
            messages=[convert_from_openai_message(m) for m in request.messages] if request.messages else None,
            system_prompt=request.system_prompt,
            response_model=ResponseModel
        )

        return {
            "status": "success",
            "data": result.dict() if hasattr(result, 'dict') else result,
            "model": request.model
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/events/stream")
async def event_stream(request: Request):
    """
    Stream real-time events via Server-Sent Events.
    """
    async def generate_events():
        """Generate events as they occur"""
        event_queue = []

        def capture_event(event):
            event_queue.append({
                "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                "data": event.data,
                "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else None
            })

        # Register handlers for all event types
        on_global(EventType.GENERATION_STARTED, capture_event)
        on_global(EventType.GENERATION_COMPLETED, capture_event)
        on_global(EventType.TOOL_STARTED, capture_event)
        on_global(EventType.TOOL_COMPLETED, capture_event)

        while True:
            if await request.is_disconnected():
                break

            # Send queued events
            while event_queue:
                event = event_queue.pop(0)
                yield {
                    "event": event["type"],
                    "data": json.dumps(event)
                }

            await asyncio.sleep(0.1)

    return EventSourceResponse(generate_events())


# ============================================================================
# Server Status & Health
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint with server information.
    """
    return {
        "name": "AbstractCore Server",
        "version": SERVER_VERSION,
        "description": "Universal LLM API Gateway - Simple endpoints + OpenAI compatibility",
        "quick_start": {
            "test_server": "/v1/quick/test",
            "simple_chat": "/v1/simple/chat?message=Hello%20world",
            "try_anthropic": "/v1/simple/chat?message=Hello&provider=anthropic&model=claude-3-5-haiku-latest",
            "try_ollama": "/v1/simple/chat?message=Hello&provider=ollama&model=llama3:8b"
        },
        "simple_endpoints": {
            "get_chat": "/v1/simple/chat?message=your_message&provider=openai&model=gpt-4o-mini",
            "post_chat": "POST /v1/{provider}/{model}/chat with {\"message\": \"hello\"}",
            "quick_test": "/v1/quick/test"
        },
        "openai_compatible": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "streaming": "/v1/chat/completions?stream=true"
        },
        "management": {
            "providers": "/v1/providers",
            "tools": "/v1/tools",
            "sessions": "/v1/sessions",
            "events": "/v1/events/stream"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/v1/status", response_model=ServerStatus)
async def server_status():
    """
    Get detailed server status.
    """
    uptime = time.time() - server_start_time

    # Get provider statuses
    providers = []
    for provider_name in ["openai", "anthropic", "ollama"]:
        try:
            models = get_available_models(provider_name)
            status = "healthy" if models else "unavailable"
        except:
            status = "unavailable"
            models = []

        providers.append(
            ProviderInfo(
                name=provider_name,
                status=status,
                models=models[:5],
                capabilities={}
            )
        )

    return ServerStatus(
        status="healthy",
        version=SERVER_VERSION,
        uptime_seconds=uptime,
        providers=providers,
        total_requests=request_count,
        active_sessions=len(sessions),
        circuit_breakers={}
    )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions in OpenAI format.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": exc.status_code
            }
        }
    )


# ============================================================================
# Server Runner
# ============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
):
    """
    Run the AbstractCore server.
    """
    import uvicorn

    uvicorn.run(
        "abstractllm.server.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


def create_app() -> FastAPI:
    """
    Create and return the FastAPI app instance.
    """
    return app


if __name__ == "__main__":
    run_server()