"""
AbstractCore Server - Clean Architecture with Universal Tool Call Syntax Support

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
from enum import Enum
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.factory import create_llm
from ..utils.structured_logging import get_logger, configure_logging
from ..utils.version import __version__
# Removed simple_model_discovery import - now using provider methods directly
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
    title="AbstractCore Server",
    description="Universal LLM Gateway with Multi-Agent Tool Call Syntax Support",
    version=__version__
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get logger
logger = get_logger("server")
logger.info("🚀 AbstractCore Server Starting", version=__version__, debug_mode=debug_mode)

# ============================================================================
# Model Type Detection
# ============================================================================

class ModelType(str, Enum):
    """Model type enumeration for filtering"""
    TEXT_GENERATION = "text-generation"
    TEXT_EMBEDDING = "text-embedding"

def is_embedding_model(model_name: str) -> bool:
    """
    Detect if a model is an embedding model based on naming heuristics.
    
    Args:
        model_name: The model name to check
        
    Returns:
        True if the model appears to be an embedding model
    """
    model_lower = model_name.lower()
    
    # Heuristics for embedding models
    embedding_patterns = [
        "embed",           # Most embedding models contain "embed"
        "all-minilm",      # Sentence-transformers MiniLM models
        "all-mpnet",       # Sentence-transformers MPNet models
        "nomic-embed",     # Nomic embedding models
        "bert-",           # BERT models (e.g., bert-base-uncased)
        "-bert",           # BERT-based embedding models (e.g., nomic-bert-2048)
        "bge-",            # BAAI BGE embedding models
        "gte-",            # GTE embedding models
        "e5-",             # E5 embedding models
        "instructor-",     # Instructor embedding models
        "granite-embedding", # IBM Granite embedding models
    ]
    
    return any(pattern in model_lower for pattern in embedding_patterns)

# ============================================================================
# Provider Model Discovery
# ============================================================================

def get_models_from_provider(provider_name: str) -> List[str]:
    """Get available models from a specific provider using their list_available_models() method."""
    try:
        if provider_name == "openai":
            from ..providers.openai_provider import OpenAIProvider
            return OpenAIProvider.list_available_models()
        elif provider_name == "anthropic":
            from ..providers.anthropic_provider import AnthropicProvider
            # Need minimal instance for API key access
            try:
                provider = AnthropicProvider(model="claude-3-haiku-20240307")
                return provider.list_available_models()
            except Exception:
                return []
        elif provider_name == "ollama":
            from ..providers.ollama_provider import OllamaProvider
            # Need minimal instance for HTTP client
            try:
                provider = OllamaProvider(model="llama2")
                return provider.list_available_models()
            except Exception:
                return []
        elif provider_name == "lmstudio":
            from ..providers.lmstudio_provider import LMStudioProvider
            # Need minimal instance for HTTP client
            try:
                provider = LMStudioProvider(model="local-model")
                return provider.list_available_models()
            except Exception:
                return []
        elif provider_name == "mlx":
            from ..providers.mlx_provider import MLXProvider
            return MLXProvider.list_available_models()
        elif provider_name == "huggingface":
            from ..providers.huggingface_provider import HuggingFaceProvider
            return HuggingFaceProvider.list_available_models()
        elif provider_name == "mock":
            # Mock provider for testing
            return ["mock-model-1", "mock-model-2", "mock-embedding-1"]
        else:
            return []
    except Exception as e:
        logger.debug(f"Failed to get models from provider {provider_name}: {e}")
        return []

# ============================================================================
# Models
# ============================================================================

class ChatMessage(BaseModel):
    """OpenAI-compatible message format"""
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message author. One of 'system', 'user', 'assistant', or 'tool'.",
        example="user"
    )
    content: Optional[str] = Field(
        default=None,
        description="The contents of the message. Can be null for assistant messages with tool calls.",
        example="What is the capital of France?"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool call that this message is responding to (required for role='tool').",
        example="call_abc123"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="The tool calls generated by the model (only for assistant messages)."
    )
    name: Optional[str] = Field(
        default=None,
        description="An optional name for the participant. Provides the model information to differentiate between participants of the same role.",
        example="User1"
    )

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(
        description="ID of the model to use. Use provider/model format (e.g., 'openai/gpt-4', 'ollama/llama3:latest', "
                    "'anthropic/claude-3-opus-20240229'). You can use the List models API to see all available models, "
                    "or filter by type=text-generation.",
        example="openai/gpt-4"
    )
    messages: List[ChatMessage] = Field(
        description="A list of messages comprising the conversation so far. Each message has a role (system/user/assistant/tool) "
                    "and content. System messages set the assistant's behavior, user messages are from the end user, "
                    "assistant messages are from the AI, and tool messages contain function call results."
    )

    # Core parameters
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, "
                    "while lower values like 0.2 will make it more focused and deterministic. "
                    "We generally recommend altering this or top_p but not both.",
        example=0.7
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="The maximum number of tokens that can be generated in the chat completion. "
                    "The total length of input tokens and generated tokens is limited by the model's context length. "
                    "If not specified, uses the model's default maximum output tokens.",
        example=2048
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="An alternative to sampling with temperature, called nucleus sampling. "
                    "The model considers the results of the tokens with top_p probability mass. "
                    "For example, 0.1 means only the tokens comprising the top 10% probability mass are considered. "
                    "We generally recommend altering this or temperature but not both.",
        example=1.0
    )
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent as server-sent events. "
                    "Tokens will be sent as data-only server-sent events as they become available, "
                    "with the stream terminated by a 'data: [DONE]' message.",
        example=False
    )

    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of tools the model may call. Currently, only functions are supported as a tool. "
                    "Use this to provide a list of functions the model may generate JSON inputs for. "
                    "Maximum of 128 functions supported."
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default="auto",
        description="Controls which (if any) tool is called by the model. "
                    "'none' means the model will not call any tool and instead generates a message. "
                    "'auto' means the model can pick between generating a message or calling one or more tools. "
                    "'required' means the model must call one or more tools. "
                    "Specifying a particular tool via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}} forces the model to call that tool.",
        example="auto"
    )

    # Other OpenAI parameters
    stop: Optional[List[str]] = Field(
        default=None,
        description="Up to 4 sequences where the API will stop generating further tokens. "
                    "The returned text will not contain the stop sequence.",
        example=None
    )
    seed: Optional[int] = Field(
        default=None,
        description="If specified, the system will make a best effort to sample deterministically, "
                    "such that repeated requests with the same seed and parameters should return the same result. "
                    "Determinism is not guaranteed.",
        example=None
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, "
                    "decreasing the model's likelihood to repeat the same line verbatim.",
        example=0.0
    )
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, "
                    "increasing the model's likelihood to talk about new topics.",
        example=0.0
    )

    # Agent format control (AppV2 feature)
    agent_format: Optional[str] = Field(
        default=None,
        description="Target agent format for tool call syntax conversion (AbstractCore-specific feature). "
                    "Options: 'auto' (auto-detect), 'openai', 'codex', 'qwen3', 'llama3', 'passthrough'. "
                    "Use 'auto' for automatic format detection based on model and user-agent.",
        example="auto"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "model": "openai/gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 150,
                "stream": False
            }
        }

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: Union[str, List[str]] = Field(
        description="Input text to embed, encoded as a string or array of strings. "
                    "To embed multiple inputs in a single request, pass an array of strings. "
                    "The input must not exceed the max input tokens for the model (8192 tokens for most embedding models), "
                    "cannot be an empty string, and any array must be 2048 dimensions or less.",
        example="this is the story of starship lost in space"
    )
    model: str = Field(
        description="ID of the model to use. Use provider/model format (e.g., 'huggingface/sentence-transformers/all-MiniLM-L6-v2', "
                    "'ollama/granite-embedding:278m', 'lmstudio/text-embedding-all-minilm-l6-v2'). "
                    "You can use the List models API to see all available models, or filter by type=text-embedding.",
        example="huggingface/sentence-transformers/all-MiniLM-L6-v2"
    )
    encoding_format: Optional[str] = Field(
        default="float",
        description="The format to return the embeddings in. Can be either 'float' or 'base64'. Defaults to 'float'.",
        example="float"
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="The number of dimensions the resulting output embeddings should have. "
                    "Only supported in some models (e.g., text-embedding-3 and later models). "
                    "If specified, embeddings will be truncated to this dimension. "
                    "Set to 0 or None to use the model's default dimension.",
        example=0
    )
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user, which can help OpenAI/providers to monitor and detect abuse. "
                    "This is optional but recommended for production applications.",
        example="user-123"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "input": "this is the story of starship lost in space",
                "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
                "encoding_format": "float",
                "dimensions": 0,
                "user": "user-123"
            }
        }

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
        "version": __version__,
        "features": [
            "multi-agent-syntax-support",
            "auto-format-detection",
            "universal-tool-calls",
            "abstractcore-integration"
        ]
    }

@app.get("/v1/models")
async def list_models(
    provider: Optional[str] = Query(
        None,
        description="Filter by provider (e.g., 'ollama', 'openai', 'anthropic', 'lmstudio')",
        example=""
    ),
    type: Optional[ModelType] = Query(
        None,
        description="Filter by model type: 'text-generation' for chat/completion models, 'text-embedding' for embedding models",
        example="text-generation"
    )
):
    """
    List available models from AbstractCore providers.
    
    Returns a list of all available models, optionally filtered by provider and/or model type.
    
    **Filters:**
    - `provider`: Limit results to a specific provider
    - `type`: Limit results to a specific model type (text-generation or text-embedding)
    
    **Examples:**
    - `/v1/models` - All models from all providers
    - `/v1/models?type=text-embedding` - Only embedding models
    - `/v1/models?type=text-generation` - Only text generation models
    - `/v1/models?provider=ollama` - Only Ollama models
    - `/v1/models?provider=ollama&type=text-embedding` - Ollama embedding models only
    """
    try:
        models_data = []

        if provider:
            # Get models from specific provider
            models = get_models_from_provider(provider.lower())
            for model in models:
                # Apply type filter if specified
                if type:
                    is_embedding = is_embedding_model(model)
                    if type == ModelType.TEXT_EMBEDDING and not is_embedding:
                        continue  # Skip non-embedding models
                    if type == ModelType.TEXT_GENERATION and is_embedding:
                        continue  # Skip embedding models
                
                model_id = f"{provider.lower()}/{model}"
                models_data.append({
                    "id": model_id,
                    "object": "model",
                    "owned_by": provider.lower(),
                    "created": int(time.time()),
                    "permission": [{"allow_create_engine": False, "allow_sampling": True}]
                })
            
            filter_msg = f" (type={type.value})" if type else ""
            logger.info(f"Listed {len(models_data)} models for provider {provider}{filter_msg}")
        else:
            # Get models from all providers
            providers = ["openai", "anthropic", "ollama", "lmstudio", "mlx", "huggingface", "mock"]
            for prov in providers:
                models = get_models_from_provider(prov)
                for model in models:
                    # Apply type filter if specified
                    if type:
                        is_embedding = is_embedding_model(model)
                        if type == ModelType.TEXT_EMBEDDING and not is_embedding:
                            continue  # Skip non-embedding models
                        if type == ModelType.TEXT_GENERATION and is_embedding:
                            continue  # Skip embedding models
                    
                    model_id = f"{prov}/{model}"
                    models_data.append({
                        "id": model_id,
                        "object": "model",
                        "owned_by": prov,
                        "created": int(time.time()),
                        "permission": [{"allow_create_engine": False, "allow_sampling": True}]
                    })
            
            filter_msg = f" (type={type.value})" if type else ""
            logger.info(f"Listed {len(models_data)} models from all providers{filter_msg}")

        return {
            "object": "list",
            "data": sorted(models_data, key=lambda x: x["id"])
        }

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {
            "object": "list",
            "data": []
        }

@app.get("/providers")
async def list_providers():
    """
    List all available AbstractCore providers and their capabilities.
    
    Returns information about all registered LLM providers, including:
    - Provider name and type
    - Number of available models
    - Current availability status
    - Provider description
    
    **Supported Providers:**
    - **OpenAI**: Commercial API with GPT-4, GPT-3.5, and embedding models
    - **Anthropic**: Commercial API with Claude 3 family models
    - **Ollama**: Local LLM server for running open-source models
    - **LMStudio**: Local model development and testing platform
    - **MLX**: Apple Silicon optimized local inference
    - **HuggingFace**: Access to HuggingFace models (transformers and embeddings)
    - **Mock**: Testing provider for development
    
    **Use Cases:**
    - Discover available providers before making requests
    - Check provider availability and model counts
    - Build dynamic provider selection UIs
    - Monitor provider status
    
    **Note:** Only providers with available models are included in the response.
    
    **Returns:** A list of provider objects with name, type, model count, status, and description.
    """
    try:
        providers_info = []
        providers = ["openai", "anthropic", "ollama", "lmstudio", "mlx", "huggingface", "mock"]

        for provider_name in providers:
            models = get_models_from_provider(provider_name)
            if models:  # Only include providers that have models
                providers_info.append({
                    "name": provider_name,
                    "type": "llm",  # Could be extended to include "embedding" type
                    "model_count": len(models),
                    "status": "available",
                    "description": f"{provider_name.title()} provider with {len(models)} available models"
                })

        logger.info(f"Listed {len(providers_info)} available providers")

        return {
            "providers": sorted(providers_info, key=lambda x: x["name"])
        }

    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        return {
            "providers": []
        }

@app.post("/v1/responses")
async def create_response(request: ChatCompletionRequest, http_request: Request):
    """
    Create a real-time streaming response for the given chat conversation.
    
    This endpoint provides real-time conversation capabilities optimized for streaming interaction.
    It's similar to OpenAI's Realtime/Responses API, automatically enabling streaming for immediate token delivery.
    
    **Key Features:**
    - **Always Streams**: Streaming is automatically enabled for real-time interaction
    - **Lower Latency**: Optimized for quick first-token delivery
    - **Same Parameters**: Uses the same request format as `/v1/chat/completions`
    - **Multi-Provider**: Supports all providers (OpenAI, Anthropic, Ollama, etc.)
    
    **Use Cases:**
    - Real-time chat interfaces
    - Voice-to-text streaming
    - Live coding assistants
    - Interactive agents
    
    **Differences from `/v1/chat/completions`:**
    - Streaming is always enabled (ignores `stream: false`)
    - Optimized for immediate response delivery
    - Better for user-facing real-time applications
    
    **Example:**
    ```json
    {
      "model": "openai/gpt-4",
      "messages": [
        {"role": "user", "content": "Tell me a story"}
      ]
    }
    ```
    
    **Returns:** Server-sent events stream of chat completion chunks, terminated by `data: [DONE]`.
    """
    # For now, delegate to chat completions with streaming enabled
    # The OpenAI Responses API is essentially streaming chat completions with enhanced real-time features
    request.stream = True  # Force streaming for responses API

    provider, model = parse_model_string(request.model)

    logger.info(
        "📡 Responses API Request",
        provider=provider,
        model=model,
        messages=len(request.messages),
        has_tools=bool(request.tools)
    )

    return await process_chat_completion(provider, model, request, http_request)

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embedding vectors representing the input text.
    
    Creates an embedding vector representing the input text. Embeddings are useful for:
    - Semantic search
    - Document similarity
    - Clustering and classification
    - Retrieval-Augmented Generation (RAG)
    
    **Supported Providers:**
    - **HuggingFace**: Local sentence-transformers models with ONNX acceleration
    - **Ollama**: Local embedding models via Ollama API
    - **LMStudio**: Local embedding models via LMStudio API
    
    **Model Format:** Use `provider/model` format:
    - `huggingface/sentence-transformers/all-MiniLM-L6-v2`
    - `ollama/granite-embedding:278m`
    - `lmstudio/text-embedding-all-minilm-l6-v2`
    
    **To see available embedding models:** `GET /v1/models?type=text-embedding`
    
    **Returns:** A list of embedding objects containing the embedding vector and metadata.
    """
    try:
        # Parse provider and model
        provider, model = parse_model_string(request.model)

        logger.info(
            "🔢 Embedding Request",
            provider=provider,
            model=model,
            input_type=type(request.input).__name__,
            input_count=len(request.input) if isinstance(request.input, list) else 1
        )

        # Route to EmbeddingManager with provider parameter
        # EmbeddingManager handles all embedding logic for all providers
        from ..embeddings.manager import EmbeddingManager

        # Validate provider
        provider_lower = provider.lower()
        if provider_lower not in ["huggingface", "ollama", "lmstudio"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Embedding provider '{provider}' not supported. Supported providers: huggingface, ollama, lmstudio",
                        "type": "unsupported_provider"
                    }
                }
            )

        # Create embedding manager with provider specification
        embedder = EmbeddingManager(
            model=model,
            provider=provider_lower,
            output_dims=request.dimensions
        )

        # Process input - handle both string and list
        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input

        # Generate embeddings
        embeddings = embedder.embed_batch(inputs)

        # Convert to OpenAI format
        embedding_objects = []
        for i, embedding in enumerate(embeddings):
            embedding_objects.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })

        # Calculate usage using centralized token utilities
        # Calculate total tokens using centralized utility
        from ..utils.token_utils import TokenUtils
        model_name = getattr(embedder, 'model_name', None)
        total_tokens = sum(TokenUtils.estimate_tokens(text, model_name) for text in inputs)

        response = {
            "object": "list",
            "data": embedding_objects,
            "model": f"{provider}/{model}",
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }

        logger.info(
            "✅ Embeddings generated",
            provider=provider_lower,
            count=len(embedding_objects),
            dimensions=len(embeddings[0]) if embeddings else 0,
            total_tokens=total_tokens
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": str(e), "type": "embedding_error"}}
        )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    Create a model response for the given chat conversation.
    
    Given a list of messages comprising a conversation, the model will return a response. 
    This endpoint supports streaming, tool calling, and multiple providers.
    
    **Key Features:**
    - Multi-provider support (OpenAI, Anthropic, Ollama, LMStudio, etc.)
    - Streaming responses with server-sent events
    - Tool/function calling with automatic syntax conversion
    - OpenAI-compatible format
    
    **Provider Format:** Use `provider/model` format in the model field:
    - `openai/gpt-4` - OpenAI GPT-4
    - `ollama/llama3:latest` - Ollama LLaMA 3
    - `anthropic/claude-3-opus-20240229` - Anthropic Claude 3 Opus
    
    **To see available models:** `GET /v1/models?type=text-generation`
    
    **Returns:** A chat completion object, or a stream of chat completion chunks if streaming is enabled.
    """
    provider, model = parse_model_string(request.model)
    return await process_chat_completion(provider, model, request, http_request)

@app.post("/{provider}/v1/chat/completions")
async def provider_chat_completions(
    provider: str,
    request: ChatCompletionRequest,
    http_request: Request
):
    """
    Provider-specific chat completions endpoint.
    
    Same functionality as `/v1/chat/completions` but allows specifying the provider in the URL path.
    Useful when you want explicit provider routing or when the model name doesn't include the provider prefix.
    
    **Examples:**
    - `POST /ollama/v1/chat/completions` with `"model": "llama3:latest"`
    - `POST /openai/v1/chat/completions` with `"model": "gpt-4"`
    
    **Note:** Provider in the URL takes precedence over provider in the model name.
    """
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

                # For OpenAI/Codex format: only clean if content contains tool calls
                # For other formats: apply syntax rewriting
                if syntax_rewriter.target_format in [SyntaxFormat.OPENAI, SyntaxFormat.CODEX]:
                    # Only clean content if it contains tool call patterns
                    # This prevents stripping spaces from regular text chunks
                    if any(pattern in content for pattern in ['<function_call>', '<tool_call>', '<|tool_call|>', '```tool_code']):
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

    # For OpenAI/Codex format: only clean if content contains tool calls
    # For other formats: apply syntax rewriting
    if syntax_rewriter.target_format in [SyntaxFormat.OPENAI, SyntaxFormat.CODEX]:
        # Only clean content if it contains tool call patterns
        # This prevents stripping spaces from regular text
        if any(pattern in content for pattern in ['<function_call>', '<tool_call>', '<|tool_call|>', '```tool_code']):
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
# Server Runner
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(
        "🚀 Starting AbstractCore Server",
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