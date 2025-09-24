"""
OpenAI-compatible request/response models for AbstractCore server.

These models ensure compatibility with any OpenAI client library while
supporting AbstractCore's extended features.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import time


# ============================================================================
# Enums for API Parameters
# ============================================================================

class ModelType(str, Enum):
    """Model type for filtering models endpoint"""
    CHAT = "chat"
    EMBEDDING = "embedding"


# ============================================================================
# OpenAI-Compatible Request Models
# ============================================================================

class Message(BaseModel):
    """Standard message format"""
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Role of the message sender",
        examples=["user", "assistant", "system"]
    )
    content: Union[str, List[Dict[str, Any]]] = Field(
        description="Content of the message",
        examples=["Hello, how can I help you today?", "What is the weather like?"]
    )
    name: Optional[str] = Field(default=None, description="Name of the participant")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool calls made by the assistant")
    tool_call_id: Optional[str] = Field(default=None, description="ID of the tool call being responded to")

    class Config:
        schema_extra = {
            "examples": [
                {"role": "user", "content": "Hello!"},
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": "Hello! How can I help you today?"}
            ]
        }


class Function(BaseModel):
    """Function definition for tool calling"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition"""
    type: Literal["function"] = "function"
    function: Function


class ResponseFormat(BaseModel):
    """Response format specification"""
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(
        description="Model to use (e.g., 'gpt-4o-mini', 'anthropic/claude-3-5-haiku-latest', 'ollama/llama3:8b')",
        examples=["gpt-4o-mini", "anthropic/claude-3-5-haiku-latest", "ollama/llama3:8b"]
    )
    messages: List[Message] = Field(
        description="List of messages in the conversation",
        examples=[[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you today?"}
        ]]
    )

    # Optional parameters
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature (0-2)")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(default=1, ge=1, description="Number of completions to generate")
    stream: Optional[bool] = Field(default=False, description="Enable streaming responses")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(default=None, description="Modify likelihood of specific tokens")
    user: Optional[str] = Field(default=None, description="User identifier for tracking")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    tools: Optional[List[Tool]] = Field(default=None, description="Available tools for the model")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="How to choose tools")
    response_format: Optional[ResponseFormat] = Field(default=None, description="Response format specification")

    # AbstractCore extensions
    provider: Optional[str] = Field(
        default=None,
        description="Override provider (e.g., 'anthropic' to force Claude even with 'gpt-4' model)",
        examples=["openai", "anthropic", "ollama"]
    )
    retry_config: Optional[Dict[str, Any]] = Field(default=None, description="Custom retry settings")
    response_model: Optional[str] = Field(default=None, description="Pydantic model name for structured output")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "Hello! What is the capital of France?"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100
                },
                {
                    "model": "anthropic/claude-3-5-haiku-latest",
                    "messages": [
                        {"role": "system", "content": "You are a helpful coding assistant."},
                        {"role": "user", "content": "Write a Python function to reverse a string"}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                {
                    "model": "ollama/llama3:8b",
                    "messages": [
                        {"role": "user", "content": "Explain quantum computing in simple terms"}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 300
                }
            ]
        }


# ============================================================================
# OpenAI-Compatible Response Models
# ============================================================================

class Choice(BaseModel):
    """Response choice"""
    index: int
    message: Message
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    """Streaming response choice"""
    index: int
    delta: Message
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming response chunk"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[Usage] = None


# ============================================================================
# Model List Response
# ============================================================================

class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str

    # AbstractCore extensions
    provider: Optional[str] = None
    model_type: Optional[Literal["chat", "embedding"]] = None
    max_tokens: Optional[int] = None
    supports_tools: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_streaming: Optional[bool] = None


class ModelsResponse(BaseModel):
    """List of available models"""
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# ============================================================================
# AbstractCore-Specific Models
# ============================================================================

class ProviderConfig(BaseModel):
    """Provider configuration"""
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = 30.0
    retry_config: Optional[Dict[str, Any]] = None
    circuit_breaker: Optional[Dict[str, Any]] = None


class ProviderInfo(BaseModel):
    """Provider information and status"""
    name: str
    status: Literal["healthy", "degraded", "unavailable"]
    models: List[str]
    capabilities: Dict[str, bool]
    error: Optional[str] = None
    last_check: datetime = Field(default_factory=datetime.now)


class SessionConfig(BaseModel):
    """Session configuration"""
    id: Optional[str] = None
    provider: str
    model: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


class SessionInfo(BaseModel):
    """Session information"""
    id: str
    provider: str
    model: str
    created_at: datetime
    message_count: int
    total_tokens: Optional[int] = None
    total_cost: Optional[float] = None


class EventStreamResponse(BaseModel):
    """Real-time event stream response"""
    event: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolRegistration(BaseModel):
    """Tool registration request"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function_code: Optional[str] = None  # Python code as string


class StructuredOutputRequest(BaseModel):
    """Structured output generation request"""
    model: str
    prompt: str
    response_model: Dict[str, Any]  # Pydantic model as JSON schema
    messages: Optional[List[Message]] = None
    system_prompt: Optional[str] = None
    provider: Optional[str] = None
    max_retries: Optional[int] = 3


class ServerStatus(BaseModel):
    """Server status and health information"""
    status: Literal["healthy", "degraded", "error"]
    version: str
    uptime_seconds: float
    providers: List[ProviderInfo]
    total_requests: int
    active_sessions: int
    circuit_breakers: Dict[str, str]  # provider -> state
    metrics: Optional[Dict[str, Any]] = None


class SimpleChatRequest(BaseModel):
    """Simple chat request for provider-specific endpoints"""
    message: str = Field(
        description="Your message to the AI",
        examples=["Hello!", "What is Python?", "Write a haiku about coding", "Explain quantum physics simply"]
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Creativity level (0.1 = focused, 0.9 = creative)"
    )
    max_tokens: Optional[int] = Field(
        default=500,
        ge=1,
        le=4000,
        description="Maximum length of response"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System instructions for the AI",
        examples=["You are a helpful coding assistant", "You are a creative writing partner"]
    )

    class Config:
        schema_extra = {
            "examples": [
                {
                    "message": "Hello! How are you today?",
                    "temperature": 0.7,
                    "max_tokens": 100
                },
                {
                    "message": "Write a Python function to reverse a string",
                    "temperature": 0.1,
                    "max_tokens": 300,
                    "system_prompt": "You are a helpful coding assistant. Always include comments."
                },
                {
                    "message": "Tell me a creative story about a robot",
                    "temperature": 0.9,
                    "max_tokens": 500,
                    "system_prompt": "You are a creative writing assistant."
                },
                {
                    "message": "Explain machine learning in simple terms",
                    "temperature": 0.3,
                    "max_tokens": 250
                }
            ]
        }


class SimpleChatResponse(BaseModel):
    """Response from simple chat endpoints"""
    message: str = Field(description="The original message sent")
    response: str = Field(description="AI's response to your message")
    provider: str = Field(description="Provider used (openai, anthropic, etc.)")
    model: str = Field(description="Model used for generation")
    settings: Dict[str, Any] = Field(description="Settings used for generation")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage information")

    class Config:
        schema_extra = {
            "example": {
                "message": "Hello! How are you?",
                "response": "Hello! I'm doing great, thank you for asking. How can I help you today?",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "system_prompt": None
                },
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 18,
                    "total_tokens": 30
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "message": "Model not found",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            }
        }