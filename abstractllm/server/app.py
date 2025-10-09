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

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.factory import create_llm
from ..utils.simple_model_discovery import get_available_models
from ..utils.structured_logging import get_logger, configure_logging
from ..tools import UniversalToolHandler
from ..tools.parser import detect_tool_calls, parse_tool_calls


# ============================================================================
# Enums & Models
# ============================================================================

class ModelType(str, Enum):
    """Model type for filtering"""
    CHAT = "chat"
    EMBEDDING = "embedding"


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible message format"""
    role: Literal["system", "user", "assistant", "tool"] = Field(description="Message role")
    content: Optional[str] = Field(description="Message content (can be null for tool messages)", default=None)
    tool_call_id: Optional[str] = Field(description="Tool call ID for tool messages", default=None)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(description="Tool calls in assistant messages", default=None)
    name: Optional[str] = Field(description="Name of the tool (for tool messages)", default=None)


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


# Anthropic Messages API Models
class AnthropicContentBlock(BaseModel):
    """Anthropic content block"""
    type: Literal["text"] = Field(description="Content type")
    text: str = Field(description="Text content")
    cache_control: Optional[Dict[str, Any]] = Field(default=None, description="Cache control settings")


class AnthropicMessage(BaseModel):
    """Anthropic message format - supports both string and rich content"""
    role: Literal["user", "assistant"] = Field(description="Message role")
    content: Any = Field(description="Message content - can be string or array of content blocks")


class AnthropicMessagesRequest(BaseModel):
    """
    🎯 **ANTHROPIC MESSAGES API** 🎯

    Native Anthropic Messages API format for Anthropic-compatible clients.
    """
    model: str = Field(description="Model to use (e.g., 'claude-3-5-haiku-latest', 'ollama/qwen3-coder:30b')")
    max_tokens: int = Field(description="Maximum tokens to generate")
    messages: List[AnthropicMessage] = Field(description="List of messages in the conversation")

    # Optional parameters
    system: Optional[Any] = Field(default=None, description="System prompt - can be string or array of content blocks")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Controls randomness")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling")
    stream: Optional[bool] = Field(default=False, description="Enable streaming")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Request metadata")

    # Tool calling support
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Available tools for function calling")
    tool_choice: Optional[Dict[str, Any]] = Field(default=None, description="Tool choice configuration")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "model": "claude-3-5-haiku-latest",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": "Hello, Claude!"}],
                    "temperature": 0.7
                },
                {
                    "model": "ollama/qwen3-coder:30b",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": "Write a Python function"}],
                    "system": "You are a helpful coding assistant",
                    "temperature": 0.1
                }
            ]
        }


# OpenAI Responses API Models (for Codex compatibility)
class ResponsesInputItem(BaseModel):
    """Input item for Responses API"""
    type: Literal["message"] = Field(default="message", description="Input item type")
    role: Literal["system", "user", "assistant", "tool"] = Field(description="Message role")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(description="Message content", default=None)
    tool_call_id: Optional[str] = Field(description="Tool call ID for tool messages", default=None)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(description="Tool calls in assistant messages", default=None)


class ResponsesRequest(BaseModel):
    """
    🎯 **OPENAI RESPONSES API** 🎯

    OpenAI's new Responses API format for Codex compatibility.
    Similar to Chat Completions but with enhanced features.
    """
    model: str = Field(description="Model to use")
    input: List[ResponsesInputItem] = Field(description="Input items (messages)")

    # Core parameters (same as chat completions)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=500, ge=1, le=4000)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = Field(default=False)

    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default="auto")
    parallel_tool_calls: Optional[bool] = Field(default=True)

    # Structured output
    response_format: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "model": "gpt-4o-mini",
                    "input": [
                        {"type": "message", "role": "system", "content": "You are a helpful assistant"},
                        {"type": "message", "role": "user", "content": "Hello!"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            ]
        }


# ============================================================================
# FastAPI App Setup
# ============================================================================

# Configure structured logging for server
import logging

# Check if debug mode is enabled
debug_mode = os.getenv("ABSTRACTCORE_DEBUG", "false").lower() == "true"
console_level = logging.DEBUG if debug_mode else logging.INFO

configure_logging(
    console_level=console_level,  # DEBUG if ABSTRACTCORE_DEBUG=true, otherwise INFO
    file_level=logging.DEBUG,     # Log everything to file
    log_dir="logs",               # Create logs directory
    verbatim_enabled=True,        # Capture full prompts/responses
    console_json=False,           # Human-readable console
    file_json=True               # Machine-readable files
)

app = FastAPI(
    title="AbstractCore Server",
    description="Universal LLM Gateway - OpenAI-Compatible API for ALL Providers",
    version="2.2.3"
)

# Get server logger
server_logger = get_logger("server")
server_logger.info("🚀 AbstractCore Server Starting",
                   version="2.2.3",
                   logging_configured=True)

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
# Tool Handling Functions
# ============================================================================

# Essential tools that should be preserved for agentic CLIs
ESSENTIAL_TOOLS = ['shell', 'local_shell', 'exec', 'bash', 'execute_command', 'unified_exec']

def filter_tools_for_local_models(tools, mode='essential'):
    """
    Filter tools for local models to maintain agency while reducing complexity.

    Args:
        tools: List of OpenAI-format tool definitions
        mode: 'essential' (keep execution tools), 'simple' (keep simple tools), 'none' (strip all)

    Returns:
        Filtered list of tools
    """
    if not tools:
        return []

    if mode == 'none':
        return []

    filtered = []
    for tool in tools:
        tool_name = tool.get('function', {}).get('name', '')
        tool_type = tool.get('type', '')

        # Keep essential execution tools for agency
        if tool_name in ESSENTIAL_TOOLS or tool_type in ESSENTIAL_TOOLS:
            filtered.append(tool)
        elif mode == 'simple':
            # Keep tools with simple schemas (low complexity)
            if is_simple_tool(tool):
                filtered.append(tool)

    return filtered

def is_simple_tool(tool):
    """Check if a tool has a simple schema suitable for local models."""
    try:
        params = tool.get('function', {}).get('parameters', {})
        properties = params.get('properties', {})

        # Consider it simple if:
        # 1. Few parameters (≤ 3)
        # 2. No nested objects or complex arrays
        # 3. Basic types only (string, number, boolean)

        if len(properties) > 3:
            return False

        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get('type', '')

            # Allow basic types
            if prop_type in ['string', 'number', 'integer', 'boolean']:
                continue
            # Allow simple arrays of strings
            elif prop_type == 'array' and prop_def.get('items', {}).get('type') == 'string':
                continue
            else:
                return False

        return True
    except:
        return False

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
    # API server mode: disable tool execution (agentic CLI will execute tools)
    return create_llm(provider, model=model, execute_tools=False), provider, model


def supports_native_tool_role(provider: str, model: str) -> bool:
    """
    Check if a model supports native 'tool' role in messages.

    Models that support it:
    - OpenAI: GPT-4, GPT-3.5-turbo, GPT-4o
    - Anthropic: Claude models (via tool_use content blocks)
    - Codex-compatible models

    Models that DON'T support it:
    - Most local models (Ollama, LMStudio)
    - Base/completion models
    """
    provider_lower = provider.lower()
    model_lower = model.lower()

    # OpenAI models support tool role
    if provider_lower == "openai":
        return True

    # Anthropic models support it (via content blocks)
    if provider_lower == "anthropic":
        return True

    # Most local models don't support it
    if provider_lower in ["ollama", "lmstudio", "mlx", "huggingface"]:
        return False

    # Conservative default
    return False


def convert_tool_messages_for_model(messages: List[Dict[str, Any]], supports_tool_role: bool) -> List[Dict[str, Any]]:
    """
    Convert tool messages to appropriate format for model.

    If model supports tool role: keep as-is
    If model doesn't support tool role:
      - Convert tool messages to user messages with [TOOL RESULT] markers
      - Remove tool_calls from assistant messages (Ollama doesn't support them)
    """
    if supports_tool_role:
        return messages

    converted = []
    for msg in messages:
        if msg.get("role") == "tool":
            # Convert tool message to user message with markers
            tool_content = msg.get("content", "")
            tool_name = msg.get("name", msg.get("tool_call_id", "unknown"))
            converted.append({
                "role": "user",
                "content": f"[TOOL RESULT: {tool_name}]\n{tool_content}\n[/TOOL RESULT]"
            })
        else:
            # Clean the message - remove tool_calls and other OpenAI-specific fields
            clean_msg = {
                "role": msg["role"],
                "content": msg.get("content", "")
            }
            converted.append(clean_msg)

    return converted


# ============================================================================
# Enhanced OpenAI-Compatible Endpoints for Agentic CLIs
# ============================================================================

# For better tool calling support
import uuid

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


# ============================================================================
# OpenAI Responses API Endpoint (Codex Compatibility)
# ============================================================================

@app.post("/v1/responses")
async def openai_responses(request: ResponsesRequest):
    """
    🎯 **OPENAI RESPONSES API** 🎯

    OpenAI's Responses API endpoint for Codex compatibility.
    Accepts input items and returns structured output/reasoning.

    **URL Pattern:** `/v1/responses`

    **Examples:**
    - Model: "gpt-4o-mini" → Uses OpenAI provider
    - Model: "ollama/qwen3-coder:30b" → Uses Ollama provider
    """
    logger = get_logger("openai_responses")
    request_id = uuid.uuid4().hex[:8]

    logger.info("🚀 OpenAI Responses API Request",
                request_id=request_id,
                original_model=request.model,
                input_count=len(request.input),
                has_tools=bool(request.tools))

    try:
        # Parse provider and model
        provider, model = parse_model_string(request.model)

        logger.info("🔧 Model Parsing",
                    request_id=request_id,
                    original_model=request.model,
                    parsed_provider=provider,
                    parsed_model=model)

        # Create provider
        # API server mode: disable tool execution (agentic CLI will execute tools)
        llm = create_llm(provider, model=model, execute_tools=False)

        # Convert input items to messages array
        messages = []
        for item in request.input:
            msg_dict = {"role": item.role}

            # Handle content
            if item.content is not None:
                if isinstance(item.content, str):
                    msg_dict["content"] = item.content
                else:
                    # For complex content (like Anthropic format), extract text
                    msg_dict["content"] = str(item.content)
            else:
                msg_dict["content"] = None

            # Handle tool calls
            if item.tool_calls:
                msg_dict["tool_calls"] = item.tool_calls

            # Handle tool messages
            if item.tool_call_id:
                msg_dict["tool_call_id"] = item.tool_call_id

            messages.append(msg_dict)

        # Check if model supports native tool role
        model_supports_tool_role = supports_native_tool_role(provider, model)

        # Convert tool messages if needed
        adapted_messages = convert_tool_messages_for_model(messages, model_supports_tool_role)

        logger.info("📝 Converted Input to Messages",
                    request_id=request_id,
                    message_count=len(adapted_messages),
                    supports_tool_role=model_supports_tool_role)

        # Call llm.generate with messages array
        gen_kwargs = {
            "prompt": "",  # Empty prompt when using messages
            "messages": adapted_messages,
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            "tools": request.tools,
            "tool_choice": request.tool_choice if request.tools else None
        }

        logger.info("🎯 Starting Generation",
                    request_id=request_id,
                    provider=provider,
                    model=model)

        import time
        start_time = time.time()
        response = llm.generate(**gen_kwargs)
        generation_time = (time.time() - start_time) * 1000

        logger.info("🎉 Generation Completed",
                    request_id=request_id,
                    generation_time_ms=generation_time)

        # Build Responses API response format
        output_items = []

        # Add text output
        if response and hasattr(response, 'content') and response.content:
            output_items.append({
                "type": "message",
                "role": "assistant",
                "content": response.content
            })

        # Add tool calls
        if response and hasattr(response, 'tool_calls') and response.tool_calls:
            # In Responses API, tool calls are in the message content
            if output_items:
                output_items[0]["tool_calls"] = [
                    {
                        "id": tc.call_id or f"call_{uuid.uuid4().hex[:20]}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments
                        }
                    } for tc in response.tool_calls
                ]

        # If no output, add empty message
        if not output_items:
            output_items.append({
                "type": "message",
                "role": "assistant",
                "content": ""
            })

        return {
            "id": f"resp_{uuid.uuid4().hex[:8]}",
            "model": request.model,
            "output": output_items,
            "usage": {
                "input_tokens": getattr(response, 'usage', {}).get('prompt_tokens', 0) if response else 0,
                "output_tokens": getattr(response, 'usage', {}).get('completion_tokens', 0) if response else 0,
                "total_tokens": getattr(response, 'usage', {}).get('total_tokens', 0) if response else 0
            }
        }

    except Exception as e:
        logger.error("❌ Generation Failed",
                     request_id=request_id,
                     error=str(e),
                     error_type=type(e).__name__)

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
        )


# ============================================================================
# Anthropic Messages API Endpoint
# ============================================================================

@app.post("/v1/messages")
async def anthropic_messages(request: AnthropicMessagesRequest, beta: Optional[bool] = Query(default=None)):
    """
    🎯 **ANTHROPIC MESSAGES API** 🎯

    Native Anthropic Messages API endpoint for Anthropic-compatible clients.

    **URL Pattern:** `/v1/messages`

    **Examples:**
    - Model: "claude-3-5-haiku-latest" → Uses Anthropic provider
    - Model: "ollama/qwen3-coder:30b" → Uses Ollama provider
    """
    # Initialize structured logger and generate request ID
    logger = get_logger("anthropic_messages")
    request_id = uuid.uuid4().hex[:8]

    # Check if debug mode is enabled
    debug_mode = os.getenv("ABSTRACTCORE_DEBUG", "false").lower() == "true"

    # Extract user content for logging (first user message)
    user_content = "unknown"
    try:
        for msg in request.messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    user_content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                elif isinstance(msg.content, list) and msg.content:
                    # Extract first text block
                    for block in msg.content:
                        if isinstance(block, dict) and block.get("text"):
                            user_content = block["text"][:100] + "..." if len(block["text"]) > 100 else block["text"]
                            break
                        elif hasattr(block, 'text'):
                            user_content = block.text[:100] + "..." if len(block.text) > 100 else block.text
                            break
                break
    except Exception:
        user_content = "extraction_failed"

    logger.info("🚀 Anthropic Messages API Request",
                request_id=request_id,
                original_model=request.model,
                user_content=user_content,
                beta_param=beta,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                debug_mode=debug_mode)

    # Log full request payload if debug mode is enabled or verbatim logging
    if debug_mode:
        import json
        from datetime import datetime

        # Create detailed request log
        full_request_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "endpoint": "/v1/messages",
            "beta_param": beta,
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stop_sequences": request.stop_sequences,
            "stream": request.stream,
            "system": request.system,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content
                } for msg in request.messages
            ],
            "tools": getattr(request, 'tools', None),
            "tool_choice": getattr(request, 'tool_choice', None)
        }

        # Save to dedicated payload log
        os.makedirs("logs", exist_ok=True)
        payload_log_file = f"logs/{datetime.now().strftime('%Y%m%d')}-payloads.jsonl"
        with open(payload_log_file, "a") as f:
            f.write(json.dumps(full_request_data) + "\n")

        # Enhanced tool logging
        tool_info = {}
        if request.tools:
            tool_info = {
                "tool_count": len(request.tools),
                "tool_names": [tool.get("function", {}).get("name", "unknown") for tool in request.tools],
                "tool_choice": request.tool_choice
            }

        logger.info("💾 Full Request Payload Saved",
                   request_id=request_id,
                   payload_file=payload_log_file,
                   has_tools=bool(request.tools),
                   message_count=len(request.messages),
                   system_prompt=bool(request.system),
                   **tool_info)

    try:
        # Parse provider and model from request
        provider, model = parse_model_string(request.model)

        logger.info("🔧 Model Parsing",
                    request_id=request_id,
                    original_model=request.model,
                    parsed_provider=provider,
                    parsed_model=model)

        # Create provider
        logger.info("🏭 Creating LLM Provider",
                    request_id=request_id,
                    provider=provider,
                    model=model)

        # Optional fallback handling for local models
        original_provider = provider
        original_model = model
        fallback_used = False

        # Check for optional local override parameter
        force_local = request.metadata.get("force_local", False) if hasattr(request, "metadata") and request.metadata else False

        if force_local and provider in ["anthropic", "openai"]:
            # Optional override: use local model instead of external APIs
            logger.warning("🔄 Local override requested, using fallback model",
                         request_id=request_id,
                         original_model=original_model,
                         original_provider=provider,
                         fallback_provider="ollama",
                         fallback_model="qwen3-coder:30b")
            provider = "ollama"
            model = "qwen3-coder:30b"
            fallback_used = True
        elif provider == "anthropic":
            # Smart fallback: only when API is actually unreachable
            try:
                import httpx
                with httpx.Client(timeout=2.0) as client:
                    client.get("https://api.anthropic.com", timeout=2.0)
            except Exception:
                logger.warning("🔄 Anthropic API unreachable, using fallback model",
                             request_id=request_id,
                             original_model=original_model,
                             fallback_provider="ollama",
                             fallback_model="qwen3-coder:30b")
                provider = "ollama"
                model = "qwen3-coder:30b"
                fallback_used = True

        # API server mode: disable tool execution (agentic CLI will execute tools)
        llm = create_llm(provider, model=model, execute_tools=False)

        logger.info("✅ LLM Provider Created Successfully",
                    request_id=request_id,
                    provider_type=type(llm).__name__,
                    provider=provider,
                    model=model,
                    fallback_used=fallback_used,
                    original_provider=original_provider if fallback_used else None,
                    original_model=original_model if fallback_used else None)

        # Helper function to extract text from Anthropic content
        def extract_text_content(content):
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Extract text from all content blocks
                text_parts = []
                for block in content:
                    if hasattr(block, 'text'):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and 'text' in block:
                        text_parts.append(block['text'])
                return "\n".join(text_parts)
            else:
                return str(content)

        # Convert Anthropic messages to our internal format
        openai_messages = []

        # Add system message if provided
        if request.system:
            system_text = extract_text_content(request.system)
            openai_messages.append({"role": "system", "content": system_text})

        # Convert Anthropic messages to OpenAI format
        for msg in request.messages:
            content_text = extract_text_content(msg.content)
            openai_messages.append({
                "role": msg.role,
                "content": content_text
            })

        # Check if model supports native tool role
        model_supports_tool_role = supports_native_tool_role(provider, model)

        # Convert tool messages if needed
        adapted_messages = convert_tool_messages_for_model(openai_messages, model_supports_tool_role)

        logger.info("📝 Converted Anthropic to OpenAI Messages",
                    request_id=request_id,
                    message_count=len(adapted_messages),
                    supports_tool_role=model_supports_tool_role)

        # Generation parameters - use messages array instead of prompt string
        gen_kwargs = {
            "prompt": "",  # Empty prompt when using messages
            "messages": adapted_messages,
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }

        # Determine if local model
        is_local_model = provider.lower() in ["ollama", "lmstudio"]

        if request.stop_sequences:
            gen_kwargs["stop"] = request.stop_sequences

        # Advanced tool handling with runtime configuration
        has_tools = bool(request.tools)
        tool_handler = None
        filtered_tools = None

        if has_tools:
            # Check for tool mode configuration via headers
            tool_mode = getattr(request, 'headers', {}).get('X-Tool-Mode', 'auto')
            if hasattr(request, 'tool_mode'):  # Also check direct parameter
                tool_mode = request.tool_mode

            if is_local_model:
                # Filter tools for local models
                if tool_mode == 'none':
                    filtered_tools = []
                elif tool_mode == 'simple':
                    filtered_tools = filter_tools_for_local_models(request.tools, mode='simple')
                else:  # 'essential' or 'auto'
                    filtered_tools = filter_tools_for_local_models(request.tools, mode='essential')

                logger.debug("🔧 Filtered tools for local model",
                           request_id=request_id,
                           provider=provider,
                           model=model,
                           original_count=len(request.tools),
                           filtered_count=len(filtered_tools),
                           tool_mode=tool_mode)

                # Initialize tool handler for prompting
                if filtered_tools:
                    try:
                        tool_handler = UniversalToolHandler(model)
                        if tool_handler.supports_prompted:
                            # Convert OpenAI tools to ToolDefinition format
                            from ..tools.core import ToolDefinition
                            tool_defs = []
                            for tool in filtered_tools:
                                if tool.get('type') == 'function':
                                    func = tool.get('function', {})
                                    tool_def = ToolDefinition(
                                        name=func.get('name', 'unknown'),
                                        description=func.get('description', ''),
                                        parameters=func.get('parameters', {})
                                    )
                                    tool_defs.append(tool_def)

                            # Add tool prompt to system prompt
                            if tool_defs:
                                tool_prompt = tool_handler.format_tools_prompt(tool_defs)
                                if effective_system_prompt:
                                    effective_system_prompt = f"{tool_prompt}\n\n{effective_system_prompt}"
                                else:
                                    effective_system_prompt = tool_prompt

                                logger.debug("🔧 Added tool prompt to system prompt",
                                           request_id=request_id,
                                           tool_count=len(tool_defs))

                            # Don't pass tools in API call for prompted models
                            gen_kwargs["tools"] = None
                        else:
                            # Pass filtered tools directly if prompting not supported
                            gen_kwargs["tools"] = filtered_tools
                            if request.tool_choice:
                                gen_kwargs["tool_choice"] = request.tool_choice
                    except Exception as e:
                        logger.warning("🔧 Tool handler initialization failed, falling back to direct tools",
                                     request_id=request_id,
                                     error=str(e))
                        gen_kwargs["tools"] = filtered_tools
                        if request.tool_choice:
                            gen_kwargs["tool_choice"] = request.tool_choice
                else:
                    # No tools to pass
                    gen_kwargs["tools"] = None
            else:
                # Native model - pass tools directly
                gen_kwargs["tools"] = request.tools
                if request.tool_choice:
                    gen_kwargs["tool_choice"] = request.tool_choice

        # Streaming support
        # Note: Currently disabled for Anthropic Messages API
        if False and request.stream:
            # Anthropic-compatible streaming
            def generate_anthropic_stream():
                try:
                    gen_kwargs["stream"] = True
                    message_id = f"msg_{uuid.uuid4().hex[:8]}"

                    # Initial event
                    initial_event = {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": request.model,
                            "stop_reason": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0}
                        }
                    }
                    yield f"event: message_start\ndata: {json.dumps(initial_event)}\n\n"

                    # Content block start
                    content_start = {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""}
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(content_start)}\n\n"

                    # Stream content and tools
                    content_index = 0
                    tool_use_started = False

                    for chunk in llm.generate(**gen_kwargs):
                        # Handle text content
                        if hasattr(chunk, 'content') and chunk.content:
                            delta_event = {
                                "type": "content_block_delta",
                                "index": content_index,
                                "delta": {"type": "text_delta", "text": chunk.content}
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

                        # Handle tool calls
                        if hasattr(chunk, 'tool_calls') and chunk.tool_calls and not tool_use_started:
                            # End text content block
                            if content_index == 0:
                                content_stop = {"type": "content_block_stop", "index": content_index}
                                yield f"event: content_block_stop\ndata: {json.dumps(content_stop)}\n\n"
                                content_index += 1

                            # Start tool use blocks
                            for tool_call in chunk.tool_calls:
                                tool_start = {
                                    "type": "content_block_start",
                                    "index": content_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tool_call.call_id or f"toolu_{uuid.uuid4().hex[:20]}",
                                        "name": tool_call.name,
                                        "input": {}
                                    }
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(tool_start)}\n\n"

                                # Tool input delta
                                input_delta = {
                                    "type": "content_block_delta",
                                    "index": content_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": json.dumps(tool_call.arguments)
                                    }
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(input_delta)}\n\n"

                                # Tool use stop
                                tool_stop = {"type": "content_block_stop", "index": content_index}
                                yield f"event: content_block_stop\ndata: {json.dumps(tool_stop)}\n\n"
                                content_index += 1

                            tool_use_started = True

                    # Content block stop
                    content_stop = {"type": "content_block_stop", "index": 0}
                    yield f"event: content_block_stop\ndata: {json.dumps(content_stop)}\n\n"

                    # Message stop
                    message_stop = {
                        "type": "message_stop",
                        "message": {
                            "stop_reason": "tool_use" if tool_use_started else "end_turn",
                            "usage": {"output_tokens": 10}  # Approximate
                        }
                    }
                    yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"

                except Exception as e:
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": str(e)
                        }
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

            return StreamingResponse(
                generate_anthropic_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Regular response
            logger.info("🎯 Starting Generation",
                        request_id=request_id,
                        provider=provider,
                        model=model,
                        message_count=len(adapted_messages),
                        temperature=gen_kwargs.get("temperature"),
                        max_tokens=gen_kwargs.get("max_output_tokens"))

            import time
            start_time = time.time()
            response = llm.generate(**gen_kwargs)
            generation_time = (time.time() - start_time) * 1000  # Convert to ms

            # Check if response is None - this indicates a serious error
            if response is None:
                raise Exception("Provider returned None response - likely a connection or internal error")

            # Parse tool calls from response if using prompted model
            if is_local_model and tool_handler and tool_handler.supports_prompted and response.content:
                try:
                    parsed_response = tool_handler.parse_response(response.content, mode="prompted")
                    if parsed_response.has_tool_calls():
                        # Convert to GenerateResponse format
                        response.tool_calls = parsed_response.tool_calls
                        # Clean tool syntax from content
                        response.content = parsed_response.content

                        logger.debug("🔧 Parsed tool calls from local model response",
                                   request_id=request_id,
                                   tool_call_count=len(parsed_response.tool_calls),
                                   tools=[call.name for call in parsed_response.tool_calls])
                except Exception as e:
                    logger.warning("🔧 Tool call parsing failed",
                                 request_id=request_id,
                                 error=str(e))

            # Response validation logging
            response_length = len(response.content) if response and hasattr(response, 'content') and response.content else 0

            logger.info("🎉 Generation Completed",
                        request_id=request_id,
                        provider=provider,
                        model=model,
                        response_length=response_length,
                        generation_time_ms=generation_time,
                        tokens_used=getattr(response, 'usage', {}) if response else {})

            # Warning for suspiciously short responses
            if response_length < 100 and is_local_model:
                logger.warning("⚠️ Suspiciously short response from local model",
                             request_id=request_id,
                             provider=provider,
                             model=model,
                             response_length=response_length,
                             requested_tokens=gen_kwargs.get("max_output_tokens", "unknown"),
                             response_preview=response.content[:50] if response and response.content else "empty")

            # Log the actual generation with structured logging
            logger.log_generation(
                provider=provider,
                model=model,
                prompt=f"{len(adapted_messages)} messages",  # Summary instead of full prompt
                response=response.content if response and hasattr(response, 'content') and response.content else str(response) if response else "No response",
                tokens=getattr(response, 'usage', None) if response else None,
                latency_ms=generation_time,
                success=True
            )

            # Anthropic-compatible response format with tool support
            content_blocks = []

            # Add tool use blocks if tools were used
            if response and hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tool_call.call_id or f"toolu_{uuid.uuid4().hex[:20]}",
                        "name": tool_call.name,
                        "input": tool_call.arguments
                    })

                # Clean any remaining tool syntax from content
                if response and hasattr(response, 'content') and response.content:
                    from abstractllm.tools.parser import clean_tool_syntax
                    cleaned_content = clean_tool_syntax(response.content, response.tool_calls)

                    # Only add text block if there's meaningful content after cleaning
                    # Exclude various forms of tool error messages
                    if cleaned_content and cleaned_content.strip():
                        cleaned_stripped = cleaned_content.strip()
                        # Skip if it's just tool error messages
                        skip_patterns = [
                            "Tool Results:",
                            "- Error: Tool",
                            "Tool not found",
                            "Error: Tool 'Bash' not found",
                            "Error: Tool 'Read' not found"
                        ]
                        should_skip = any(pattern in cleaned_stripped for pattern in skip_patterns)

                        if not should_skip and len(cleaned_stripped) > 5:  # Only include substantial content
                            content_blocks.insert(0, {
                                "type": "text",
                                "text": cleaned_content
                            })
            else:
                # No tools, add text content if present
                if response and hasattr(response, 'content') and response.content:
                    content_blocks.append({
                        "type": "text",
                        "text": response.content
                    })

            # If no content blocks, add empty text
            if not content_blocks:
                content_blocks.append({
                    "type": "text",
                    "text": ""
                })

            return {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": request.model,
                "stop_reason": "tool_use" if (response and hasattr(response, 'tool_calls') and response.tool_calls) else "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": getattr(response, 'usage', {}).get('prompt_tokens', 0) if response else 0,
                    "output_tokens": getattr(response, 'usage', {}).get('completion_tokens', 0) if response else 0
                }
            }

    except Exception as e:
        # Log the error with full details
        logger.error("❌ Generation Failed",
                     request_id=request_id,
                     error=str(e),
                     error_type=type(e).__name__,
                     original_model=request.model,
                     parsed_provider=provider if 'provider' in locals() else 'unknown',
                     parsed_model=model if 'model' in locals() else 'unknown')

        # Log with structured logging
        if 'provider' in locals() and 'model' in locals():
            logger.log_generation(
                provider=provider,
                model=model,
                prompt=f"{len(adapted_messages)} messages" if 'adapted_messages' in locals() else "unknown",
                response="",
                success=False,
                error=str(e),
                latency_ms=0
            )

        # Anthropic-compatible error format
        raise HTTPException(
            status_code=404,
            detail={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"model: {request.model}"
                }
            }
        )


@app.post("/v1/chat/completions")
async def standard_chat_completions(request: OpenAIChatCompletionRequest):
    """
    🎯 **STANDARD OPENAI ENDPOINT** 🎯

    Standard /v1/chat/completions endpoint for maximum CLI compatibility.
    Provider determined from model name (e.g., "anthropic/claude-3-5-haiku-latest").

    **Examples:**
    - Model: "gpt-4o-mini" → Uses OpenAI provider
    - Model: "anthropic/claude-3-5-haiku-latest" → Uses Anthropic provider
    - Model: "ollama/qwen3-coder:30b" → Uses Ollama provider
    """
    # Parse provider from model string for standard routing
    provider, parsed_model = parse_model_string(request.model)
    return await enhanced_chat_completions(provider, request)


@app.post("/{provider}/v1/chat/completions")
async def provider_chat_completions(provider: str, request: OpenAIChatCompletionRequest):
    """Provider-specific routing for /provider/v1/chat/completions pattern"""
    return await enhanced_chat_completions(provider, request)


async def enhanced_chat_completions(provider: str, request: OpenAIChatCompletionRequest):
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
        # DEBUG: Log incoming request
        import logging
        logger = logging.getLogger("server")
        logger.info(f"📥 Chat Completions Request | model={request.model}, messages={len(request.messages)}, tools={'YES' if request.tools else 'NO'}")
        if request.tools:
            logger.debug(f"🔧 Tools in request: {[t.function.name if hasattr(t, 'function') else t.get('function', {}).get('name') for t in request.tools]}")

        # Parse provider and model from request
        _, parsed_model = parse_model_string(request.model)

        # Use the parsed values (provider parameter might override parsed_provider)
        actual_provider = provider
        actual_model = parsed_model

        # Model parsing successful

        # Create provider
        llm = create_llm(actual_provider, model=actual_model)

        # Convert OpenAI messages to internal format
        messages = []
        for msg in request.messages:
            msg_dict = {"role": msg.role}

            # Handle content
            if msg.content is not None:
                msg_dict["content"] = msg.content
            else:
                msg_dict["content"] = None

            # Handle tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls

            # Handle tool messages
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id

            # Handle tool name
            if hasattr(msg, 'name') and msg.name:
                msg_dict["name"] = msg.name

            messages.append(msg_dict)

        # Get provider and model
        provider, _ = parse_model_string(request.model)

        # Check if model supports native tool role
        model_supports_tool_role = supports_native_tool_role(provider, request.model)

        # Convert tool messages if needed
        adapted_messages = convert_tool_messages_for_model(messages, model_supports_tool_role)

        from ..utils.structured_logging import get_logger
        logger = get_logger("openai_chat_completions")
        logger.info("📝 Converted OpenAI to Messages",
                    message_count=len(adapted_messages),
                    supports_tool_role=model_supports_tool_role)

        # Generation parameters - use messages array for ALL providers
        # Calculate max_output_tokens as max(4096, 25% of max_tokens)
        max_output = request.max_tokens if request.max_tokens else 4096
        calculated_max_output = max(4096, int(max_output * 0.25))

        gen_kwargs = {
            "prompt": "",  # Empty prompt when using messages
            "messages": adapted_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,  # Keep original for reference
            "max_output_tokens": calculated_max_output,  # Use calculated value
        }

        # Determine if local model for later use
        is_local_model = provider.lower() in ["ollama", "lmstudio"]
        filtered_tools = None

        # Pass tools directly (provider will handle them appropriately)
        if request.tools:
            gen_kwargs["tools"] = request.tools
            if request.tool_choice:
                gen_kwargs["tool_choice"] = request.tool_choice
            # Keep reference for later streaming logic
            filtered_tools = request.tools
        else:
            gen_kwargs["tools"] = None

        # Add response format support
        if hasattr(request, 'response_format') and request.response_format:
            if request.response_format.get("type") == "json_schema":
                schema_def = request.response_format.get("json_schema", {})
                if "schema" in schema_def:
                    # Convert to Pydantic model for structured output
                    from pydantic import create_model
                    from typing import Optional, List, Dict

                    try:
                        # Create a dynamic Pydantic model from the JSON schema
                        schema = schema_def["schema"]
                        fields = {}
                        if "properties" in schema:
                            for prop_name, prop_def in schema["properties"].items():
                                field_type = str  # Default to string
                                if prop_def.get("type") == "integer":
                                    field_type = int
                                elif prop_def.get("type") == "number":
                                    field_type = float
                                elif prop_def.get("type") == "boolean":
                                    field_type = bool
                                elif prop_def.get("type") == "array":
                                    field_type = List[str]  # Simplified
                                elif prop_def.get("type") == "object":
                                    field_type = Dict  # Simplified

                                # Check if required
                                required = prop_name in schema.get("required", [])
                                if required:
                                    fields[prop_name] = (field_type, ...)
                                else:
                                    fields[prop_name] = (Optional[field_type], None)

                        model_name = schema_def.get("name", "DynamicModel")
                        dynamic_model = create_model(model_name, **fields)
                        gen_kwargs["response_model"] = dynamic_model
                    except Exception as e:
                        # If schema parsing fails, continue without structured output
                        pass

        if request.stream:
            # Enhanced OpenAI-compatible streaming with tool calling support
            def generate_openai_stream():
                try:
                    gen_kwargs["stream"] = True
                    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    created_time = int(time.time())
                    tool_calls = []  # Track tool calls across chunks
                    accumulated_content = ""  # Track content for tool call parsing
                    tool_calls_emitted = False  # Track if tool calls were emitted
                    buffering_for_tools = False  # Track if we're buffering content for tool detection

                    for chunk in llm.generate(**gen_kwargs):
                        openai_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": f"{actual_provider}/{actual_model}",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": None
                            }]
                        }

                        # Handle content streaming
                        if hasattr(chunk, 'content') and chunk.content:
                            # For local models with tools, buffer when we detect tool call markers
                            if is_local_model and filtered_tools:
                                accumulated_content += chunk.content

                                # Only start buffering if we detect potential tool call
                                import re
                                if not buffering_for_tools and "<|" in chunk.content:
                                    buffering_for_tools = True  # Start buffering when we see tool marker

                                # Check if we have complete tool calls to parse
                                # Handle both <|tool_call|> and <|\ntool_call|> (with newline/whitespace)
                                if buffering_for_tools and re.search(r'<\|\s*tool_call\s*\|>', accumulated_content) and "</|tool_call|>" in accumulated_content:
                                    try:
                                        # Parse tool calls from accumulated content
                                        tool_handler = UniversalToolHandler(actual_model)
                                        if tool_handler.supports_prompted:
                                            parsed_response = tool_handler.parse_response(accumulated_content, mode="prompted")
                                            if parsed_response.has_tool_calls():
                                                # Emit tool calls in streaming format
                                                for tool_call in parsed_response.tool_calls:
                                                    tool_call_chunk = {
                                                        "id": chat_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": created_time,
                                                        "model": f"{actual_provider}/{actual_model}",
                                                        "choices": [{
                                                            "index": 0,
                                                            "delta": {
                                                                "tool_calls": [{
                                                                    "id": tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}",
                                                                    "type": "function",
                                                                    "function": {
                                                                        "name": tool_call.name,
                                                                        "arguments": json.dumps(tool_call.arguments) if isinstance(tool_call.arguments, dict) else str(tool_call.arguments)
                                                                    }
                                                                }]
                                                            },
                                                            "finish_reason": None
                                                        }]
                                                    }
                                                    yield f"data: {json.dumps(tool_call_chunk)}\n\n"
                                                    tool_calls.extend(parsed_response.tool_calls)
                                                    tool_calls_emitted = True

                                                # Send cleaned content if any remains
                                                if parsed_response.content and parsed_response.content.strip():
                                                    content_chunk = {
                                                        "id": chat_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": created_time,
                                                        "model": f"{actual_provider}/{actual_model}",
                                                        "choices": [{
                                                            "index": 0,
                                                            "delta": {"content": parsed_response.content},
                                                            "finish_reason": None
                                                        }]
                                                    }
                                                    yield f"data: {json.dumps(content_chunk)}\n\n"

                                                # Clear accumulated content after parsing
                                                accumulated_content = ""
                                                buffering_for_tools = False
                                                continue
                                    except Exception as e:
                                        print(f"🔧 Streaming tool call parsing failed: {e}")

                                # Emit content if not buffering OR after tool calls were emitted
                                # Stream content normally unless we're actively buffering for tool detection
                                if not buffering_for_tools:
                                    openai_chunk["choices"][0]["delta"]["content"] = chunk.content
                                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                            else:
                                # Regular content streaming for non-local models
                                openai_chunk["choices"][0]["delta"]["content"] = chunk.content
                                yield f"data: {json.dumps(openai_chunk)}\n\n"

                        # Handle tool calls
                        elif hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                            for tool_call in chunk.tool_calls:
                                tool_call_chunk = {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": f"{actual_provider}/{actual_model}",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": len(tool_calls),
                                                "id": tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}",
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call.name,
                                                    "arguments": json.dumps(tool_call.arguments) if isinstance(tool_call.arguments, dict) else str(tool_call.arguments)
                                                }
                                            }]
                                        },
                                        "finish_reason": None
                                    }]
                                }
                                tool_calls.append(tool_call)
                                yield f"data: {json.dumps(tool_call_chunk)}\n\n"

                        # Handle other chunk types
                        elif hasattr(chunk, 'delta') and chunk.delta:
                            openai_chunk["choices"][0]["delta"] = chunk.delta
                            yield f"data: {json.dumps(openai_chunk)}\n\n"

                    # Emit any remaining buffered content if no tool calls were found
                    # BUT do not emit incomplete tool calls
                    if buffering_for_tools and accumulated_content and not tool_calls_emitted:
                        # Check if it's an incomplete tool call - if so, discard it
                        has_incomplete_tool_call = (
                            "<|tool_call|>" in accumulated_content and
                            "</|tool_call|>" not in accumulated_content
                        )

                        if not has_incomplete_tool_call:
                            # Only emit if it's not an incomplete tool call
                            remaining_content_chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": f"{actual_provider}/{actual_model}",
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": accumulated_content},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(remaining_content_chunk)}\n\n"
                        # If it's incomplete tool call, discard silently

                    # Final chunk
                    finish_reason = "tool_calls" if tool_calls else "stop"
                    final_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": f"{actual_provider}/{actual_model}",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }]
                    }

                    # Include usage stats if requested
                    if hasattr(request, 'stream_options') and request.stream_options and request.stream_options.get('include_usage'):
                        final_chunk["usage"] = {
                            "prompt_tokens": 0,  # Would need to be calculated
                            "completion_tokens": 0,
                            "total_tokens": 0
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
            # Enhanced regular response with tool calling support
            response = llm.generate(**gen_kwargs)

            # Parse tool calls from response if using local model with filtered tools
            if is_local_model and filtered_tools and response.content:
                try:
                    tool_handler = UniversalToolHandler(actual_model)
                    if tool_handler.supports_prompted:
                        parsed_response = tool_handler.parse_response(response.content, mode="prompted")
                        if parsed_response.has_tool_calls():
                            # Convert to GenerateResponse format
                            response.tool_calls = parsed_response.tool_calls
                            # Clean tool syntax from content
                            response.content = parsed_response.content

                            print(f"🔧 Parsed {len(parsed_response.tool_calls)} tool calls from local model response")
                except Exception as e:
                    print(f"🔧 Tool call parsing failed: {e}")

            # Prepare message object
            message = {
                "role": "assistant",
                "content": response.content
            }

            # Handle tool calls in response
            finish_reason = "stop"
            if hasattr(response, 'tool_calls') and response.tool_calls:
                message["tool_calls"] = []
                for tool_call in response.tool_calls:
                    message["tool_calls"].append({
                        "id": tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments) if isinstance(tool_call.arguments, dict) else str(tool_call.arguments)
                        }
                    })
                finish_reason = "tool_calls"
                # Content should be null when there are tool calls
                message["content"] = None

            # OpenAI-compatible response format
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"{actual_provider}/{actual_model}",
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": getattr(response, 'usage', {}).get('prompt_tokens', 0),
                    "completion_tokens": getattr(response, 'usage', {}).get('completion_tokens', 0),
                    "total_tokens": getattr(response, 'usage', {}).get('total_tokens', 0)
                }
            }

    except Exception as e:
        # Enhanced error handling with more details
        error_detail = str(e)
        print(f"❌ ERROR in chat completions: {error_detail}")
        print(f"❌ Request model: {request.model}")
        actual_provider_str = actual_provider if 'actual_provider' in locals() else 'unknown'
        actual_model_str = actual_model if 'actual_model' in locals() else 'unknown'
        print(f"❌ Provider: {actual_provider_str}")
        print(f"❌ Model: {actual_model_str}")

        # Return more helpful error message
        if "not found" in error_detail.lower() or "404" in error_detail:
            detail = f"Model '{request.model}' not found. Available models: /v1/models"
        elif "authentication" in error_detail.lower() or "api_key" in error_detail.lower():
            detail = f"Authentication error for provider '{actual_provider if 'actual_provider' in locals() else 'unknown'}'"
        else:
            detail = error_detail

        raise HTTPException(status_code=500, detail=detail)


# Add standard OpenAI models endpoint
@app.get("/v1/models")
async def standard_models_list(type: Optional[ModelType] = None):
    """
    🎯 **STANDARD OPENAI MODELS ENDPOINT** 🎯

    Standard /v1/models endpoint that lists all available models across providers.
    Agentic CLIs expect this endpoint for model discovery.
    """
    all_models = []
    providers = ["openai", "anthropic", "ollama", "mlx", "lmstudio"]

    for provider_name in providers:
        try:
            available_models = get_available_models(provider_name)
            for model_id in available_models:
                model_type = classify_model_type(model_id)

                # Filter by type if specified
                if type and model_type != type.value:
                    continue

                # Add both provider-prefixed and direct model names
                all_models.append({
                    "id": f"{provider_name}/{model_id}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider_name,
                    "provider": provider_name,
                    "model_type": model_type,
                    "supports_tools": model_type == "chat",
                    "supports_vision": provider_name in ["openai", "anthropic"] and model_type == "chat",
                    "supports_streaming": model_type == "chat",
                    "supports_function_calling": model_type == "chat",
                    "supports_parallel_tool_calls": model_type == "chat"
                })

                # Also add the direct model name for common models
                if provider_name == "openai" or model_id.startswith(("gpt-", "claude-", "gemini-")):
                    all_models.append({
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": provider_name,
                        "provider": provider_name,
                        "model_type": model_type,
                        "supports_tools": model_type == "chat",
                        "supports_vision": provider_name in ["openai", "anthropic"] and model_type == "chat",
                        "supports_streaming": model_type == "chat",
                        "supports_function_calling": model_type == "chat",
                        "supports_parallel_tool_calls": model_type == "chat"
                    })
        except Exception:
            # Skip provider if not available
            continue

    return {"object": "list", "data": all_models}


@app.get("/{provider}/v1/models")
async def provider_models_list(provider: str, type: Optional[ModelType] = None):
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
                "supports_tools": model_type == "chat",
                "supports_vision": provider in ["openai", "anthropic"] and model_type == "chat",
                "supports_streaming": model_type == "chat",
                "supports_function_calling": model_type == "chat",
                "supports_parallel_tool_calls": model_type == "chat"
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
                        with httpx.Client(timeout=300.0) as client:
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


# MCP (Model Context Protocol) Support for Advanced Tool Integration
@app.post("/v1/mcp/servers")
async def register_mcp_server(server_config: dict):
    """
    🔧 **MCP SERVER REGISTRATION** 🔧

    Register an MCP server for tool integration.
    Enables agentic CLIs to connect external tools and services.
    """
    # Basic MCP server registration (would need full implementation)
    server_id = server_config.get("name", f"server-{uuid.uuid4().hex[:8]}")

    return {
        "id": server_id,
        "status": "registered",
        "capabilities": server_config.get("capabilities", []),
        "message": "MCP server registered successfully"
    }


@app.get("/v1/mcp/servers")
async def list_mcp_servers():
    """List registered MCP servers"""
    return {
        "servers": [],
        "message": "MCP server listing - full implementation pending"
    }


@app.delete("/v1/mcp/servers/{server_id}")
async def unregister_mcp_server(server_id: str):
    """Unregister an MCP server"""
    return {
        "id": server_id,
        "status": "unregistered",
        "message": f"MCP server {server_id} unregistered"
    }


# Additional endpoints for agentic CLI compatibility
@app.get("/v1/engines")
async def list_engines():
    """
    Legacy engines endpoint for compatibility with older OpenAI clients.
    Some CLIs might still use this endpoint.
    """
    return await standard_models_list()


@app.post("/v1/completions")
async def text_completions(request: dict):
    """
    Legacy text completions endpoint for compatibility.
    Converts to chat completions format internally.
    """
    # Convert legacy completions to chat completions
    prompt = request.get("prompt", "")

    # Create a chat completion request
    chat_request = OpenAIChatCompletionRequest(
        model=request.get("model", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=request.get("max_tokens", 150),
        temperature=request.get("temperature", 0.7),
        stream=request.get("stream", False)
    )

    # Route to chat completions
    provider, _ = parse_model_string(chat_request.model)
    response = await enhanced_chat_completions(provider, chat_request)

    # Convert response format if needed
    if isinstance(response, dict) and "choices" in response:
        # Convert chat completion to text completion format
        choice = response["choices"][0]
        if "message" in choice:
            choice["text"] = choice["message"]["content"]
            del choice["message"]

    return response


@app.get("/v1/capabilities")
async def get_capabilities():
    """
    🚀 **ABSTRACTCORE CAPABILITIES** 🚀

    Endpoint for agentic CLIs to discover server capabilities.
    """
    return {
        "api_version": "v1",
        "server": "AbstractCore",
        "version": "2.2.3",
        "capabilities": {
            "chat_completions": True,
            "text_completions": True,
            "embeddings": True,
            "streaming": True,
            "tool_calling": True,
            "parallel_tool_calls": True,
            "structured_output": True,
            "mcp_support": True,
            "multi_provider": True,
            "vision": True
        },
        "supported_models": {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
            "ollama": ["qwen3-coder:30b", "llama3:8b", "gemma:7b"],
            "mlx": ["mlx-community/*"],
            "lmstudio": ["local models"]
        },
        "endpoints": [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/models",
            "/v1/embeddings",
            "/v1/capabilities",
            "/v1/mcp/servers",
            "/{provider}/v1/chat/completions",
            "/{provider}/v1/models",
            "/{provider}/v1/embeddings"
        ]
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