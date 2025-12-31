"""
AbstractCore Server - Universal LLM Gateway with Media Processing

A focused FastAPI server that provides OpenAI-compatible endpoints with support for
multiple agent formats, tool calling, and comprehensive media processing capabilities.

Key Features:
- Universal tool call syntax conversion (OpenAI, Codex, Qwen3, LLaMA3, custom)
- Auto-detection of target agent format
- Media processing for images, documents, and data files
- OpenAI Vision API compatible format support
- Streaming responses with media attachments
- Clean delegation to AbstractCore
- Proper ReAct loop support
- Comprehensive model listing from AbstractCore providers

Media Support:
- Images: PNG, JPEG, GIF, WEBP, BMP, TIFF
- Documents: PDF, DOCX, XLSX, PPTX
- Data: CSV, TSV, JSON, XML, TXT, MD
- Size limits: 10MB per file, 32MB total per request
- Both base64 data URLs and HTTP URLs supported
"""

import os
import json
import time
import uuid
import base64
import tempfile
import urllib.request
import urllib.parse
import argparse
import sys
import logging
from typing import List, Dict, Any, Optional, Literal, Union, Iterator, Tuple, Annotated
from enum import Enum
from fastapi import FastAPI, HTTPException, Request, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.factory import create_llm
from ..utils.structured_logging import get_logger, configure_logging
from ..utils.version import __version__
from ..utils.message_preprocessor import MessagePreprocessor
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

# Initialize with default logging configuration (can be overridden later)
debug_mode = os.getenv("ABSTRACTCORE_DEBUG", "false").lower() == "true"

# Initial logging setup (will be reconfigured if --debug is used)
# Check environment variable for debug mode
initial_console_level = logging.DEBUG if debug_mode else logging.INFO
configure_logging(
    console_level=initial_console_level,
    file_level=logging.DEBUG,
    log_dir="logs",
    verbatim_enabled=True,
    console_json=False,
    file_json=True
)

# Get initial logger
logger = get_logger("server")

# Log initial startup with debug mode status
logger.info("🚀 AbstractCore Server Initializing", version=__version__, debug_mode=debug_mode)

def reconfigure_for_debug():
    """Reconfigure logging for debug mode when --debug flag is used."""
    global debug_mode, logger

    debug_mode = True

    # Reconfigure with debug levels
    configure_logging(
        console_level=logging.DEBUG,
        file_level=logging.DEBUG,
        log_dir="logs",
        verbatim_enabled=True,
        console_json=False,
        file_json=True
    )

    # Update logger instance
    logger = get_logger("server")

    return logger

# Create FastAPI app (will be initialized after argument parsing)
app = FastAPI(
    title="AbstractCore Server",
    description="Universal LLM Gateway with Multi-Agent Tool Call Syntax Support and Media Processing",
    version=__version__
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Enhanced Error Handling and Logging Middleware
# ============================================================================

@app.middleware("http")
async def debug_logging_middleware(request: Request, call_next):
    """Enhanced logging middleware for debug mode."""
    start_time = time.time()

    # Log request details in debug mode
    if debug_mode:
        logger.debug(
            "📥 HTTP Request",
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            client=request.client.host if request.client else "unknown"
        )

    response = await call_next(request)

    process_time = time.time() - start_time

    # Log response details
    log_data = {
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time_ms": round(process_time * 1000, 2)
    }

    if response.status_code >= 400:
        logger.error("❌ HTTP Error Response", **log_data)
    elif debug_mode:
        logger.debug("📤 HTTP Response", **log_data)
    else:
        logger.info("✅ HTTP Request", **log_data)

    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Enhanced handler for 422 validation errors with detailed logging."""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })

    # Log detailed validation error information
    logger.error(
        "🔴 Request Validation Error (422)",
        method=request.method,
        url=str(request.url),
        error_count=len(error_details),
        errors=error_details,
        client=request.client.host if request.client else "unknown"
    )

    # In debug mode, also try to log the request body if possible
    if debug_mode:
        try:
            # Try to get the request body for debugging
            body = await request.body()
            if body:
                try:
                    import json
                    body_json = json.loads(body)
                    logger.debug(
                        "📋 Request Body (Validation Error)",
                        body=body_json
                    )
                except json.JSONDecodeError:
                    logger.debug(
                        "📋 Request Body (Validation Error)",
                        body_text=body.decode('utf-8', errors='replace')[:1000]  # Limit to 1000 chars
                    )
        except Exception as e:
            logger.debug(f"Could not read request body for debugging: {e}")

    # Return detailed error response
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "type": "validation_error",
                "details": error_details
            }
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Enhanced handler for HTTP exceptions with detailed logging."""
    logger.error(
        "🔴 HTTP Exception",
        method=request.method,
        url=str(request.url),
        status_code=exc.status_code,
        detail=str(exc.detail),
        client=request.client.host if request.client else "unknown"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "http_error"
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler for unexpected exceptions with detailed logging."""
    logger.error(
        "💥 Unexpected Server Error",
        method=request.method,
        url=str(request.url),
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        client=request.client.host if request.client else "unknown",
        exc_info=True  # This will include the full stack trace
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error"
            }
        }
    )

# ============================================================================
# Model Type Detection
# ============================================================================

# Import the core capability enums directly
from ..providers.model_capabilities import ModelInputCapability, ModelOutputCapability


# ============================================================================
# Provider Model Discovery (Using Centralized Registry)
# ============================================================================

def get_models_from_provider(
    provider_name: str, 
    input_capabilities=None, 
    output_capabilities=None
) -> List[str]:
    """
    Get available models from a specific provider using the centralized provider registry.

    Args:
        provider_name: Name of the provider
        input_capabilities: Optional list of ModelInputCapability enums
        output_capabilities: Optional list of ModelOutputCapability enums

    Returns:
        List of model names from the provider, optionally filtered
    """
    try:
        from ..providers.registry import get_available_models_for_provider
        return get_available_models_for_provider(
            provider_name, 
            input_capabilities=input_capabilities,
            output_capabilities=output_capabilities
        )
    except Exception as e:
        logger.debug(f"Failed to get models from provider {provider_name}: {e}")
        return []



# ============================================================================
# OpenAI Responses API Models (100% Compatible)
# ============================================================================

class OpenAIInputContent(BaseModel):
    """OpenAI Responses API content item"""
    type: Literal["input_text", "input_file"] = Field(
        description="Content type - 'input_text' for text or 'input_file' for files"
    )
    text: Optional[str] = Field(
        default=None,
        description="Text content (required when type='input_text')"
    )
    file_url: Optional[str] = Field(
        default=None,
        description="Direct file URL (required when type='input_file')"
    )

class OpenAIResponsesInput(BaseModel):
    """OpenAI Responses API input message"""
    role: Literal["user"] = Field(
        description="Message role (OpenAI responses only supports 'user')"
    )
    content: List[OpenAIInputContent] = Field(
        description="Array of input content items"
    )

class OpenAIResponsesRequest(BaseModel):
    """OpenAI Responses API request format (100% compatible)"""
    model: str = Field(
        description="Model identifier",
        example="gpt-4o"
    )
    input: List[OpenAIResponsesInput] = Field(
        description="Array of input messages"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Enable streaming (false by default, set to true for real-time responses)"
    )

# ============================================================================
# Models
# ============================================================================

class ContentItem(BaseModel):
    """Individual content item within a message (OpenAI Vision API format with file support)"""
    type: Literal["text", "image_url", "file"] = Field(
        description="Content type - 'text' for text content, 'image_url' for images, or 'file' for file attachments"
    )
    text: Optional[str] = Field(
        default=None,
        description="Text content (required when type='text')"
    )
    image_url: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Image URL object (required when type='image_url'). Should contain 'url' field with base64 data URL or HTTP(S) URL"
    )
    file_url: Optional[Dict[str, Any]] = Field(
        default=None,
        description="File URL object (required when type='file'). Should contain 'url' field with HTTP(S) URL, local path, or base64 data URL"
    )

class ChatMessage(BaseModel):
    """Enhanced OpenAI-compatible message format with media support"""
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message author. One of 'system', 'user', 'assistant', or 'tool'.",
        example="user"
    )
    content: Optional[Union[str, List[ContentItem]]] = Field(
        default=None,
        description="Message content - can be a string or array of content objects for multimodal messages. "
                   "For multimodal messages, use array format with text, image_url, and file objects.",
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

    # Provider-specific parameters (AbstractCore-specific feature)
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the provider API endpoint (AbstractCore-specific feature). "
                    "Useful for openai-compatible provider to connect to custom endpoints. "
                    "Example: 'http://localhost:1234/v1' for LMStudio, 'http://localhost:8080/v1' for llama.cpp. "
                    "If not specified, uses provider's default or environment variable.",
        example="http://localhost:1234/v1"
    )

    # Runtime/orchestrator policy (AbstractCore-specific feature)
    timeout_s: Optional[float] = Field(
        default=None,
        description="Per-request provider HTTP timeout in seconds (AbstractCore-specific feature). "
                    "Intended for orchestrators (e.g. AbstractRuntime) to enforce execution policy. "
                    "If omitted, the server uses its own defaults. "
                    "Values <= 0 are treated as unlimited.",
        example=7200.0,
    )

    class Config:
        schema_extra = {
            "examples": {
                "basic_text": {
                    "summary": "Basic Text Chat",
                    "description": "Simple text-based conversation",
                    "value": {
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
                },
                "vision_image": {
                    "summary": "Image Analysis",
                    "description": "Analyze images using vision-capable models with OpenAI Vision API format",
                    "value": {
                        "model": "ollama/qwen2.5vl:7b",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "What's in this image?"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                                        }
                                    }
                                ]
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 200
                    }
                },
                "document_analysis": {
                    "summary": "Document Analysis",
                    "description": "Process documents (PDF, CSV, Excel, etc.) with file attachments",
                    "value": {
                        "model": "lmstudio/qwen/qwen3-next-80b",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Analyze this CSV file and calculate the total sales"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:text/csv;base64,RGF0ZSxQcm9kdWN0LFNhbGVzCjIwMjQtMDEtMDEsUHJvZHVjdCBBLDEwMDAwCjIwMjQtMDEtMDIsUHJvZHVjdCBCLDE1MDAwCjIwMjQtMDEtMDMsUHJvZHVjdCBDLDI1MDAw"
                                        }
                                    }
                                ]
                            }
                        ],
                        "temperature": 0.3,
                        "max_tokens": 300
                    }
                },
                "mixed_media": {
                    "summary": "Mixed Media Analysis",
                    "description": "Process multiple file types in a single request",
                    "value": {
                        "model": "ollama/qwen2.5vl:7b",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Compare this chart image with the data in this PDF report"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                                        }
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:application/pdf;base64,JVBERi0xLjQKJdPr6eEKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDEKPJ4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovTWVkaWFCb3ggWzAgMCA2MTIgNzkyXQo+PgplbmRvYmoKeHJlZgowIDQKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNTggMDAwMDAgbiAKMDAwMDAwMDExNSAwMDAwMCBuIAp0cmFpbGVyCjw8Ci9TaXplIDQKL1Jvb3QgMSAwIFIKPj4Kc3RhcnR4cmVmCjE5NQolJUVPRgo="
                                        }
                                    }
                                ]
                            }
                        ],
                        "temperature": 0.5,
                        "max_tokens": 500,
                        "stream": False
                    }
                },
                "tools_with_media": {
                    "summary": "Tools + Media",
                    "description": "Combine tool usage with file attachments for complex workflows",
                    "value": {
                        "model": "openai/gpt-4",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Analyze this financial data and create a summary chart"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:text/csv;base64,Q29tcGFueSxRMSxRMixRMyxRNApBY21lIEluYywyMDAsMjUwLDMwMCwzNTAKVGVjaCBDb3JwLDE1MCwyMDAsMjUwLDMwMApCaXogTHRkLDEwMCwxMjAsMTQwLDE2MA=="
                                        }
                                    }
                                ]
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2048,
                        "stream": False,
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "create_chart",
                                    "description": "Create a chart from data",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "chart_type": {"type": "string"},
                                            "data": {"type": "array"}
                                        }
                                    }
                                }
                            }
                        ],
                        "tool_choice": "auto"
                    }
                },
                "complete_request": {
                    "summary": "Complete Request with Media",
                    "description": "Full example showing all possible fields with file attachment",
                    "value": {
                        "model": "openai/gpt-4",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Analyze this CSV file and provide insights"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:text/csv;base64,RGF0ZSxQcm9kdWN0LFNhbGVzCjIwMjQtMDEtMDEsUHJvZHVjdCBBLDEwMDAwCjIwMjQtMDEtMDIsUHJvZHVjdCBCLDE1MDAwCjIwMjQtMDEtMDMsUHJvZHVjdCBDLDI1MDAw"
                                        }
                                    }
                                ],
                                "tool_call_id": None,
                                "tool_calls": None,
                                "name": "DataAnalyst"
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2048,
                        "top_p": 1,
                        "stream": False,
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "analyze_data",
                                    "description": "Analyze data and generate insights",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "analysis_type": {"type": "string"},
                                            "metrics": {"type": "array"}
                                        }
                                    }
                                }
                            }
                        ],
                        "tool_choice": "auto",
                        "stop": ["END"],
                        "seed": 12345,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                        "agent_format": "auto"
                    }
                }
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
# Union Request Model for /v1/responses endpoint
# ============================================================================

class ResponsesAPIRequest(BaseModel):
    """
    Union request model for /v1/responses endpoint supporting both OpenAI and legacy formats.

    The endpoint automatically detects the format based on the presence of 'input' vs 'messages' field.
    """
    class Config:
        schema_extra = {
            "oneOf": [
                {
                    "title": "OpenAI Responses API Format",
                    "description": "OpenAI-compatible responses format with input_file support",
                    "$ref": "#/components/schemas/OpenAIResponsesRequest"
                },
                {
                    "title": "Legacy Format (ChatCompletionRequest)",
                    "description": "Backward-compatible format using messages array",
                    "$ref": "#/components/schemas/ChatCompletionRequest"
                }
            ],
            "examples": {
                "openai_format": {
                    "summary": "OpenAI Responses API Format",
                    "description": "Use input array with input_text and input_file objects",
                    "value": {
                        "model": "gpt-4o",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Analyze this document"},
                                    {"type": "input_file", "file_url": "https://example.com/doc.pdf"}
                                ]
                            }
                        ],
                        "stream": False
                    }
                },
                "legacy_format": {
                    "summary": "Legacy Format (Backward Compatible)",
                    "description": "Use messages array like standard chat completions",
                    "value": {
                        "model": "openai/gpt-4",
                        "messages": [
                            {"role": "user", "content": "Tell me a story"}
                        ],
                        "stream": False
                    }
                }
            }
        }

# ============================================================================
# OpenAI Responses API Compatibility
# ============================================================================

def convert_openai_responses_to_chat_completion(openai_request: OpenAIResponsesRequest) -> ChatCompletionRequest:
    """
    Convert OpenAI Responses API format to internal ChatCompletionRequest format.

    Transforms:
    - input -> messages
    - input_text -> text
    - input_file -> file with file_url

    Args:
        openai_request: OpenAI responses API request

    Returns:
        ChatCompletionRequest compatible with our internal processing
    """
    # Convert input messages to chat messages
    messages = []

    for input_msg in openai_request.input:
        # Build content array as list of dicts (not ContentItem objects)
        content_items = []

        for content in input_msg.content:
            if content.type == "input_text":
                content_items.append({
                    "type": "text",
                    "text": content.text
                })
            elif content.type == "input_file":
                content_items.append({
                    "type": "file",
                    "file_url": {"url": content.file_url}  # Convert to our format
                })

        # Create chat message with list content (not ContentItem objects)
        message_dict = {
            "role": input_msg.role,
            "content": content_items
        }
        messages.append(ChatMessage(**message_dict))

    # Build ChatCompletionRequest
    return ChatCompletionRequest(
        model=openai_request.model,
        messages=messages,
        max_tokens=openai_request.max_tokens,
        temperature=openai_request.temperature,
        top_p=openai_request.top_p,
        stream=openai_request.stream
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
    input_type: Optional[ModelInputCapability] = Query(
        None,
        description="Filter by input capability: 'text', 'image', 'audio', 'video'"
    ),
    output_type: Optional[ModelOutputCapability] = Query(
        None,
        description="Filter by output capability: 'text', 'embeddings'"
    ),
):
    """
    List available models from AbstractCore providers.

    Returns a list of all available models, optionally filtered by provider and/or capabilities.

    **Filtering System:**
    - `input_type`: Filter by what INPUT the model can process (text, image, audio, video)
    - `output_type`: Filter by what OUTPUT the model generates (text, embeddings)

    **Examples:**
    - `/v1/models` - All models from all providers
    - `/v1/models?output_type=embeddings` - Only embedding models
    - `/v1/models?input_type=text&output_type=text` - Text-only models that generate text
    - `/v1/models?input_type=image` - Models that can analyze images
    - `/v1/models?provider=ollama&input_type=image` - Ollama vision models only
    """
    try:
        models_data = []

        # Use the capability enums directly
        input_capabilities = [input_type] if input_type else None
        output_capabilities = [output_type] if output_type else None
        

        if provider:
            # Get models from specific provider with optional filtering
            models = get_models_from_provider(
                provider.lower(), 
                input_capabilities=input_capabilities,
                output_capabilities=output_capabilities
            )
            for model in models:
                model_id = f"{provider.lower()}/{model}"
                models_data.append({
                    "id": model_id,
                    "object": "model",
                    "owned_by": provider.lower(),
                    "created": int(time.time()),
                    "permission": [{"allow_create_engine": False, "allow_sampling": True}]
                })

            filter_parts = []
            if input_type:
                filter_parts.append(f"input_type={input_type.value}")
            if output_type:
                filter_parts.append(f"output_type={output_type.value}")
            
            filter_msg = f" ({', '.join(filter_parts)})" if filter_parts else ""
            logger.info(f"Listed {len(models_data)} models for provider {provider}{filter_msg}")
        else:
            # Get models from all providers using centralized registry
            from ..providers.registry import list_available_providers
            providers = list_available_providers()
            for prov in providers:
                models = get_models_from_provider(
                    prov, 
                    input_capabilities=input_capabilities,
                    output_capabilities=output_capabilities
                )
                for model in models:
                    model_id = f"{prov}/{model}"
                    models_data.append({
                        "id": model_id,
                        "object": "model",
                        "owned_by": prov,
                        "created": int(time.time()),
                        "permission": [{"allow_create_engine": False, "allow_sampling": True}]
                    })

            filter_parts = []
            if input_type:
                filter_parts.append(f"input_type={input_type.value}")
            if output_type:
                filter_parts.append(f"output_type={output_type.value}")
            
            filter_msg = f" ({', '.join(filter_parts)})" if filter_parts else ""
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
async def list_providers(
    include_models: bool = Query(
        False,
        description="Include model lists for each provider. Set to true for full information (slower)."
    )
):
    """
    List all available AbstractCore providers and their capabilities.

    Returns comprehensive information about all registered LLM providers, including:
    - Provider name, display name, and type
    - Number of available models and sample models (if include_models=True)
    - Current availability status and detailed error information
    - Provider description and supported features
    - Authentication requirements and installation instructions
    - Local vs. cloud provider designation

    **Query Parameters:**
    - `include_models` (bool, default=False): Include model lists for each provider.
      Set to `true` for full information (slower).

    **Performance:**
    - `include_models=false`: Metadata only (very fast, ~15ms) - **DEFAULT**
    - `include_models=true`: Full information including model lists (slower, ~800ms)

    **Supported Providers:**
    - **OpenAI**: Commercial API with GPT-4, GPT-3.5, and embedding models
    - **Anthropic**: Commercial API with Claude 3 family models
    - **Ollama**: Local LLM server for running open-source models
    - **LMStudio**: Local model development and testing platform
    - **MLX**: Apple Silicon optimized local inference
    - **HuggingFace**: Access to HuggingFace models (transformers and embeddings)

    **Use Cases:**
    - Fast provider discovery: `GET /providers` (default, very fast)
    - Full provider information: `GET /providers?include_models=true`
    - Build dynamic provider selection UIs
    - Monitor provider status and troubleshoot issues
    - Get installation instructions for missing dependencies

    **Returns:** A list of provider objects with comprehensive metadata.
    """
    try:
        from ..providers.registry import get_all_providers_with_models, get_all_providers_status

        # Get providers with models (available providers)
        available_providers = get_all_providers_with_models(include_models=include_models)

        # Optionally include all providers (even those with issues) for debugging
        # Uncomment the next line if you want to see providers with errors too:
        # all_providers = get_all_providers_status()

        logger.info(f"Listed {len(available_providers)} available providers with models")

        return {
            "providers": available_providers,
            "total_providers": len(available_providers),
            "registry_version": "2.0",  # Indicate this is using the new registry system
            "note": "Provider information from centralized AbstractCore registry"
        }

    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        return {
            "providers": [],
            "total_providers": 0,
            "error": str(e),
            "registry_version": "2.0"
        }

@app.post("/v1/responses")
async def create_response(
    http_request: Request,
    request_body: Annotated[
        Dict[str, Any],
        Body(
            ...,
            examples={
                "openai_format": {
                    "summary": "OpenAI Responses API Format",
                    "description": "Use input array with input_text and input_file objects",
                    "value": {
                        "model": "gpt-4o",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Analyze this document"},
                                    {"type": "input_file", "file_url": "https://example.com/doc.pdf"}
                                ]
                            }
                        ],
                        "stream": False
                    }
                },
                "legacy_format": {
                    "summary": "Legacy Format (Backward Compatible)",
                    "description": "Use messages array like standard chat completions",
                    "value": {
                        "model": "openai/gpt-4",
                        "messages": [
                            {"role": "user", "content": "Tell me a story"}
                        ],
                        "stream": False
                    }
                },
                "file_analysis": {
                    "summary": "Document Analysis",
                    "description": "Analyze files using OpenAI format",
                    "value": {
                        "model": "openai/gpt-4",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "What's the key information in this CSV?"},
                                    {"type": "input_file", "file_url": "data:text/csv;base64,RGF0ZSxQcm9kdWN0LFNhbGVzCjIwMjQtMDEtMDEsUHJvZHVjdCBBLDEwMDAwCjIwMjQtMDEtMDIsUHJvZHVjdCBCLDE1MDAwCjIwMjQtMDEtMDMsUHJvZHVjdCBDLDI1MDAw"}
                                ]
                            }
                        ]
                    }
                },
                "streaming_example": {
                    "summary": "Streaming Response",
                    "description": "Enable streaming for real-time responses",
                    "value": {
                        "model": "lmstudio/qwen/qwen3-next-80b",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Analyze the letter and provide a summary of the key points."},
                                    {"type": "input_file", "file_url": "https://www.berkshirehathaway.com/letters/2024ltr.pdf"}
                                ]
                            }
                        ],
                        "stream": True
                    }
                }
            }
        )
    ]
):
    """
    OpenAI Responses API (100% Compatible) + Backward Compatibility

    Supports both OpenAI's responses format and our legacy format for seamless migration.
    Streaming can be enabled by setting "stream": true for real-time interaction.

    **OpenAI Format (input_file support):**
    ```json
    {
      "model": "gpt-4o",
      "input": [
        {
          "role": "user",
          "content": [
            {"type": "input_text", "text": "Analyze this document"},
            {"type": "input_file", "file_url": "https://example.com/doc.pdf"}
          ]
        }
      ]
    }
    ```

    **Legacy Format (backward compatibility):**
    ```json
    {
      "model": "openai/gpt-4",
      "messages": [
        {"role": "user", "content": "Tell me a story"}
      ]
    }
    ```

    **Key Features:**
    - **100% OpenAI Compatible**: Supports input_file with file_url
    - **Universal File Support**: PDF, DOCX, XLSX, CSV, images, and more
    - **Multi-Provider**: Works with all providers (OpenAI, Anthropic, Ollama, etc.)
    - **Optional Streaming**: Set "stream": true for real-time responses
    - **Backward Compatible**: Existing clients continue to work

    **Returns:** Chat completion object, or server-sent events stream if streaming is enabled.
    """
    try:
        # Use the parsed request body directly
        request_data = request_body

        # Detect OpenAI responses format vs legacy format
        if "input" in request_data:
            # OpenAI Responses API format
            logger.info("📡 OpenAI Responses API format detected")

            # Parse as OpenAI format
            openai_request = OpenAIResponsesRequest(**request_data)

            # Convert to internal format
            chat_request = convert_openai_responses_to_chat_completion(openai_request)

        elif "messages" in request_data:
            # Legacy format (backward compatibility)
            logger.info("📡 Legacy responses format detected")

            # Parse as ChatCompletionRequest
            chat_request = ChatCompletionRequest(**request_data)

        else:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "Request must contain either 'input' (OpenAI format) or 'messages' (legacy format)", "type": "invalid_request"}}
            )

        # Respect user's streaming preference (defaults to False)

        # Process using our standard pipeline
        provider, model = parse_model_string(chat_request.model)

        logger.info(
            "📡 Responses API Request",
            provider=provider,
            model=model,
            format="openai" if "input" in request_data else "legacy",
            messages=len(chat_request.messages)
        )

        return await process_chat_completion(provider, model, chat_request, http_request)

    except Exception as e:
        logger.error(f"Responses API error: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(e), "type": "processing_error"}}
        )

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

# ============================================================================
# Media Processing Utilities
# ============================================================================

def handle_base64_image(data_url: str) -> str:
    """
    Process base64 data URL and save to temporary file.

    Args:
        data_url: Base64 data URL (e.g., "data:image/jpeg;base64,..." or "data:application/pdf;base64,...")

    Returns:
        Path to temporary file
    """
    try:
        # Parse data URL
        if not data_url.startswith("data:"):
            raise ValueError("Invalid data URL format")

        # Extract media type and base64 data
        header, data = data_url.split(",", 1)
        media_type = header.split(";")[0].split(":")[1]

        # Determine file extension for all supported media types
        ext_map = {
            # Images
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            # Documents
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            # Data files
            "text/csv": ".csv",
            "text/tab-separated-values": ".tsv",
            "application/json": ".json",
            "application/xml": ".xml",
            "text/xml": ".xml",
            "text/plain": ".txt",
            "text/markdown": ".md",
            # Generic fallback
            "application/octet-stream": ".bin"
        }
        extension = ext_map.get(media_type, ".bin")

        # Decode base64 data
        file_data = base64.b64decode(data)

        # Save to temporary file with request-specific prefix for better isolation
        import hashlib
        data_hash = hashlib.md5(data[:100].encode() if len(data) > 100 else data.encode()).hexdigest()[:8]
        request_id = uuid.uuid4().hex[:8]
        prefix = f"abstractcore_b64_{data_hash}_{request_id}_"

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension, prefix=prefix) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name

        # Log the temporary file creation for debugging
        logger.debug(f"Processed base64 media to temporary file: {temp_file_path} (size: {len(file_data)} bytes)")
        return temp_file_path

    except Exception as e:
        logger.error(f"Failed to process base64 media: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Invalid base64 media data: {e}", "type": "media_error"}}
        )

def download_file_temporarily(url: str) -> str:
    """
    Download file from URL to temporary file (supports images, documents, data files).

    Args:
        url: HTTP(S) URL to file

    Returns:
        Path to temporary file
    """
    try:
        # Validate URL
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are allowed")

        # Create request with browser-like headers to avoid 403 Forbidden errors
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        # Generic accept header for all file types
        request.add_header('Accept', '*/*')
        request.add_header('Accept-Language', 'en-US,en;q=0.9')
        request.add_header('Accept-Encoding', 'gzip, deflate, br')
        request.add_header('Connection', 'keep-alive')
        request.add_header('Upgrade-Insecure-Requests', '1')
        request.add_header('Sec-Fetch-Dest', 'document')  # More generic than 'image'
        request.add_header('Sec-Fetch-Mode', 'no-cors')
        request.add_header('Sec-Fetch-Site', 'cross-site')

        # Download with size limit (10MB)
        response = urllib.request.urlopen(request, timeout=30)
        if response.getheader('content-length'):
            size = int(response.getheader('content-length'))
            if size > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("File too large (max 10MB)")

        # Read data with size check
        data = b""
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            data += chunk
            if len(data) > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("File too large (max 10MB)")

        # Determine extension from content-type or URL
        content_type = response.getheader('content-type', '').lower()
        ext_map = {
            # Images
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            # Documents
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            # Data files
            "text/csv": ".csv",
            "text/tab-separated-values": ".tsv",
            "application/json": ".json",
            "application/xml": ".xml",
            "text/xml": ".xml",
            "text/plain": ".txt",
            "text/markdown": ".md",
            # Generic fallback
            "application/octet-stream": ".bin"
        }

        # Try to get extension from content-type first, then URL
        extension = ext_map.get(content_type)
        if not extension:
            # Try to get extension from URL
            url_path = parsed.path.lower()
            if url_path.endswith('.pdf'):
                extension = '.pdf'
            elif url_path.endswith('.jpg') or url_path.endswith('.jpeg'):
                extension = '.jpg'
            elif url_path.endswith('.png'):
                extension = '.png'
            elif url_path.endswith('.docx'):
                extension = '.docx'
            elif url_path.endswith('.xlsx'):
                extension = '.xlsx'
            elif url_path.endswith('.csv'):
                extension = '.csv'
            else:
                extension = '.bin'  # Generic fallback

        # Save to temporary file with request-specific prefix for better isolation
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        request_id = uuid.uuid4().hex[:8]
        prefix = f"abstractcore_file_{url_hash}_{request_id}_"

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension, prefix=prefix) as temp_file:
            temp_file.write(data)
            temp_file_path = temp_file.name

        # Log the temporary file creation for debugging
        logger.info(f"Downloaded file to temporary file: {temp_file_path} (size: {len(data)} bytes, type: {content_type})")
        return temp_file_path

    except Exception as e:
        logger.error(f"Failed to download file from URL {url}: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": f"Failed to download file: {e}", "type": "media_error"}}
        )

def download_image_temporarily(url: str) -> str:
    """
    Download image from URL to temporary file (backward compatibility wrapper).

    Args:
        url: HTTP(S) URL to image

    Returns:
        Path to temporary file
    """
    return download_file_temporarily(url)

def process_image_url_object(image_url_obj: Dict[str, Any]) -> Optional[str]:
    """
    Process OpenAI image_url object and return local file path.

    Args:
        image_url_obj: Image URL object with 'url' field

    Returns:
        Local file path or None if processing failed
    """
    try:
        url = image_url_obj.get("url", "")
        if not url:
            return None

        if url.startswith("data:"):
            # Base64 encoded image
            return handle_base64_image(url)
        elif url.startswith(("http://", "https://")):
            # Download from URL
            return download_image_temporarily(url)
        else:
            # Assume local file path
            if os.path.exists(url):
                return url
            else:
                logger.warning(f"Local file not found: {url}")
                return None

    except Exception as e:
        logger.error(f"Failed to process image URL object: {e}")
        return None

def process_file_url_object(file_url_obj: Dict[str, Any]) -> Optional[str]:
    """
    Process OpenAI file_url object and return local file path.

    Simplified format (consistent with image_url):
    {"url": "https://example.com/file.pdf"} or
    {"url": "/local/path/file.pdf"} or
    {"url": "data:application/pdf;base64,..."}

    Args:
        file_url_obj: File URL object with 'url' field (same as image_url)

    Returns:
        Local file path or None if processing failed
    """
    try:
        # Reuse existing image URL processing logic - works perfectly for any file type
        return process_image_url_object(file_url_obj)

    except Exception as e:
        logger.error(f"Failed to process file URL object: {e}")
        return None

def process_message_content(message: ChatMessage) -> Tuple[str, List[str]]:
    """
    Extract media files from message content and return clean text + media list.

    Supports both OpenAI formats:
    - content as string: "Analyze this @image.jpg"
    - content as array: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}, {"type": "file", "file_url": {...}}]

    Args:
        message: ChatMessage with content to process

    Returns:
        Tuple of (clean_text, media_file_paths)
    """
    if message.content is None:
        return "", []

    if isinstance(message.content, str):
        # Legacy format: extract @filename references
        clean_text, media_files = MessagePreprocessor.parse_file_attachments(
            message.content,
            validate_existence=True,
            verbose=False
        )
        return clean_text, media_files

    elif isinstance(message.content, list):
        # OpenAI array format: extract image_url objects
        text_parts = []
        media_files = []

        for item in message.content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text" and item.get("text"):
                    text_parts.append(item["text"])
                elif item_type == "image_url" and item.get("image_url"):
                    media_file = process_image_url_object(item["image_url"])
                    if media_file:
                        media_files.append(media_file)
                elif item_type == "file" and item.get("file_url"):
                    media_file = process_file_url_object(item["file_url"])
                    if media_file:
                        media_files.append(media_file)
            elif hasattr(item, 'type'):
                # Pydantic ContentItem object
                if item.type == "text" and item.text:
                    text_parts.append(item.text)
                elif item.type == "image_url" and item.image_url:
                    media_file = process_image_url_object(item.image_url)
                    if media_file:
                        media_files.append(media_file)
                elif item.type == "file" and item.file_url:
                    media_file = process_file_url_object(item.file_url)
                    if media_file:
                        media_files.append(media_file)

        return " ".join(text_parts), media_files

    return str(message.content), []

def adapt_prompt_for_media_types(text: str, media_files: List[str]) -> str:
    """
    Intelligently adapt prompts based on attached media file types.

    Fixes common mismatches like "What is in this image?" when sending documents.

    Args:
        text: Original text content
        media_files: List of media file paths

    Returns:
        Adapted text content
    """
    if not media_files or not text:
        return text

    # Analyze media file types
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
    document_extensions = {'.pdf', '.docx', '.xlsx', '.pptx'}
    data_extensions = {'.csv', '.tsv', '.json', '.xml'}
    text_extensions = {'.txt', '.md'}

    has_images = False
    has_documents = False
    has_data = False
    has_text = False

    for file_path in media_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in image_extensions:
            has_images = True
        elif ext in document_extensions:
            has_documents = True
        elif ext in data_extensions:
            has_data = True
        elif ext in text_extensions:
            has_text = True

    # Common prompt adaptations
    adapted_text = text

    # Fix "What is in this image?" when not dealing with images
    if "what is in this image" in text.lower():
        if has_documents and not has_images:
            adapted_text = text.replace("What is in this image?", "What is in this document?")
            adapted_text = adapted_text.replace("what is in this image?", "what is in this document?")
            adapted_text = adapted_text.replace("What is in this image", "What is in this document")
            adapted_text = adapted_text.replace("what is in this image", "what is in this document")
        elif has_data and not has_images:
            adapted_text = text.replace("What is in this image?", "What data is in this file?")
            adapted_text = adapted_text.replace("what is in this image?", "what data is in this file?")
            adapted_text = adapted_text.replace("What is in this image", "What data is in this file")
            adapted_text = adapted_text.replace("what is in this image", "what data is in this file")
        elif has_text and not has_images:
            adapted_text = text.replace("What is in this image?", "What is in this text file?")
            adapted_text = adapted_text.replace("what is in this image?", "what is in this text file?")
            adapted_text = adapted_text.replace("What is in this image", "What is in this text file")
            adapted_text = adapted_text.replace("what is in this image", "what is in this text file")

    # Fix "What is in this document?" when dealing with images
    elif "what is in this document" in text.lower() and has_images and not (has_documents or has_data or has_text):
        adapted_text = text.replace("What is in this document?", "What is in this image?")
        adapted_text = adapted_text.replace("what is in this document?", "what is in this image?")
        adapted_text = adapted_text.replace("What is in this document", "What is in this image")
        adapted_text = adapted_text.replace("what is in this document", "what is in this image")

    # Handle mixed content with specific naming
    if adapted_text != text:
        # Count media types for better description
        total_files = len(media_files)
        if total_files > 1:
            types = []
            if has_images:
                types.append("image(s)")
            if has_documents:
                types.append("document(s)")
            if has_data:
                types.append("data file(s)")
            if has_text:
                types.append("text file(s)")

            if len(types) > 1:
                adapted_text = adapted_text.replace("this document", f"these {' and '.join(types)}")
                adapted_text = adapted_text.replace("this image", f"these {' and '.join(types)}")
                adapted_text = adapted_text.replace("this file", f"these {' and '.join(types)}")

    if adapted_text != text:
        logger.info(f"Adapted prompt for media types: '{text}' → '{adapted_text}'")

    return adapted_text

def validate_media_files(files: List[str]) -> None:
    """
    Validate media files for security and size limits.

    Args:
        files: List of file paths to validate

    Raises:
        HTTPException: If validation fails
    """
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff',
                         '.pdf', '.docx', '.xlsx', '.pptx', '.csv', '.tsv', '.txt', '.md',
                         '.json', '.xml'}

    total_size = 0
    max_total_size = 32 * 1024 * 1024  # 32MB total limit

    for file_path in files:
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"File not found: {file_path}", "type": "file_not_found"}}
            )

        # Check extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"File type {ext} not allowed", "type": "invalid_file_type"}}
            )

        # Check individual file size (10MB per file)
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"File too large: {file_path} (max 10MB per file)", "type": "file_too_large"}}
            )

        total_size += file_size

        # Check total size across all files
        if total_size > max_total_size:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "Total file size exceeds 32MB limit", "type": "total_size_exceeded"}}
            )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    Create a model response for the given chat conversation with optional media attachments.

    Given a list of messages comprising a conversation, the model will return a response.
    This endpoint supports streaming, tool calling, media attachments, and multiple providers.

    **Key Features:**
    - Multi-provider support (OpenAI, Anthropic, Ollama, LMStudio, etc.)
    - Streaming responses with server-sent events
    - Tool/function calling with automatic syntax conversion
    - Media attachments (images, documents, data files)
    - OpenAI Vision API compatible format
    
    **Provider Format:** Use `provider/model` format in the model field:
    - `openai/gpt-4` - OpenAI GPT-4
    - `ollama/llama3:latest` - Ollama LLaMA 3
    - `anthropic/claude-3-opus-20240229` - Anthropic Claude 3 Opus

    **Media Attachments:** Support for OpenAI Vision API compatible format:
    - String content: "Analyze this @image.jpg" (AbstractCore @filename syntax)
    - Array content: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
    - Supported formats: Images (PNG, JPEG, GIF, WEBP), Documents (PDF, DOCX, XLSX, PPTX), Data (CSV, TSV, TXT, MD)
    - Size limits: 10MB per file, 32MB total per request

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


def _extract_trace_metadata(http_request: Request) -> Dict[str, Any]:
    """Extract trace metadata from request headers (schema-safe)."""
    meta: Dict[str, Any] = {}

    raw = (
        http_request.headers.get("x-abstractcore-trace-metadata")
        or http_request.headers.get("x-abstract-trace-metadata")
    )
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                meta.update(parsed)
        except Exception:
            # Ignore invalid metadata payloads; tracing is best-effort.
            pass

    header_map = {
        "actor_id": "x-abstractcore-actor-id",
        "session_id": "x-abstractcore-session-id",
        "run_id": "x-abstractcore-run-id",
        "parent_run_id": "x-abstractcore-parent-run-id",
    }
    for key, header in header_map.items():
        val = http_request.headers.get(header)
        if val is not None and key not in meta:
            meta[key] = val

    # Never log or return these directly; they are for internal correlation only.
    return meta


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

        # Process media from messages
        all_media_files = []
        processed_messages = []

        for message in request.messages:
            clean_text, media_files = process_message_content(message)
            all_media_files.extend(media_files)

            # Adapt prompt based on media file types to avoid confusion
            if media_files:
                adapted_text = adapt_prompt_for_media_types(clean_text, media_files)
            else:
                adapted_text = clean_text

            # Create processed message with adapted text
            processed_message = message.model_copy()
            processed_message.content = adapted_text
            processed_messages.append(processed_message)

        # Validate media files if any were found
        if all_media_files:
            validate_media_files(all_media_files)
            logger.info(
                "📎 Media Files Processed",
                request_id=request_id,
                file_count=len(all_media_files),
                files=[os.path.basename(f) for f in all_media_files[:5]]  # Log first 5 filenames
            )

        # Create LLM instance
        # Prepare provider-specific kwargs
        provider_kwargs = {}
        trace_metadata = _extract_trace_metadata(http_request)
        if trace_metadata:
            # Enable trace capture (trace_id) without retaining full trace buffers by default.
            provider_kwargs["enable_tracing"] = True
            provider_kwargs.setdefault("max_traces", 0)
        if request.base_url:
            provider_kwargs["base_url"] = request.base_url
            logger.info(
                "🔗 Custom Base URL",
                request_id=request_id,
                base_url=request.base_url
            )
        if request.timeout_s is not None:
            # Orchestrator policy: allow the caller to specify the provider timeout.
            # Note: BaseProvider treats non-positive values as "unlimited".
            provider_kwargs["timeout"] = request.timeout_s

        llm = create_llm(provider, model=model, **provider_kwargs)

        # Convert messages
        messages = convert_to_abstractcore_messages(processed_messages)

        # Create syntax rewriter
        syntax_rewriter = create_syntax_rewriter(target_format, f"{provider}/{model}")

        # Prepare generation parameters
        gen_kwargs = {
            "prompt": "",  # Empty when using messages
            "messages": messages,
            "media": all_media_files if all_media_files else None,  # Add media files
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "tools": request.tools,
            "tool_choice": request.tool_choice if request.tools else None,
            "execute_tools": False,  # Server mode - don't execute tools
        }
        if trace_metadata:
            gen_kwargs["trace_metadata"] = trace_metadata

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
        # Only cleanup files created by this request (with our specific prefixes)
        temp_files_to_cleanup = [
            f for f in all_media_files
            if f.startswith("/tmp/") and (
                "abstractcore_img_" in f or
                "abstractcore_file_" in f or
                "abstractcore_b64_" in f or
                "temp" in f
            )
        ]

        try:
            if request.stream:
                return StreamingResponse(
                    generate_streaming_response(
                        llm, gen_kwargs, provider, model, syntax_rewriter, request_id, temp_files_to_cleanup
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                response = llm.generate(**gen_kwargs)
                openai_response = convert_to_openai_response(
                    response, provider, model, syntax_rewriter, request_id
                )
                trace_id = None
                if hasattr(response, "metadata") and isinstance(getattr(response, "metadata"), dict):
                    trace_id = response.metadata.get("trace_id")
                if trace_id:
                    return JSONResponse(
                        content=openai_response,
                        headers={"X-AbstractCore-Trace-Id": str(trace_id)},
                    )
                return openai_response
        finally:
            # Cleanup temporary files (base64 and downloaded images) with delay to avoid race conditions
            import threading

            def delayed_cleanup():
                """Cleanup temporary files after a short delay to avoid race conditions"""
                time.sleep(1)  # Short delay to ensure generation is complete
                for temp_file in temp_files_to_cleanup:
                    try:
                        if os.path.exists(temp_file):
                            # Additional check: only delete files created by this session
                            if ("abstractcore_img_" in temp_file or "abstractcore_file_" in temp_file or "abstractcore_b64_" in temp_file):
                                os.unlink(temp_file)
                                logger.debug(f"Cleaned up temporary file: {temp_file}")
                            else:
                                logger.debug(f"Skipped cleanup of non-AbstractCore file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temporary file {temp_file}: {e}")

            # Run cleanup in background thread to avoid blocking response
            cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
            cleanup_thread.start()

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
    request_id: str,
    temp_files_to_cleanup: List[str] = None
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

            # Tool calls
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                has_tool_calls = True
                # OpenAI/Codex clients expect structured tool_calls deltas.
                if syntax_rewriter.target_format in [SyntaxFormat.OPENAI, SyntaxFormat.CODEX]:
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
                # Tag-based clients parse tool calls from assistant content.
                elif syntax_rewriter.target_format != SyntaxFormat.PASSTHROUGH:
                    tool_text = syntax_rewriter.rewrite_content("", detected_tool_calls=chunk.tool_calls)
                    if tool_text and tool_text.strip():
                        openai_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": f"{provider}/{model}",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": tool_text},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n"

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

        # Cleanup temporary files for streaming with delay to avoid race conditions
        if temp_files_to_cleanup:
            import threading

            def delayed_streaming_cleanup():
                """Cleanup temporary files after streaming completes"""
                time.sleep(2)  # Longer delay for streaming to ensure all chunks are sent
                for temp_file in temp_files_to_cleanup:
                    try:
                        if os.path.exists(temp_file):
                            # Additional check: only delete files created by this session
                            if ("abstractcore_img_" in temp_file or "abstractcore_file_" in temp_file or "abstractcore_b64_" in temp_file):
                                os.unlink(temp_file)
                                logger.debug(f"Cleaned up temporary file during streaming: {temp_file}")
                            else:
                                logger.debug(f"Skipped cleanup of non-AbstractCore streaming file: {temp_file}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary file {temp_file}: {cleanup_error}")

            # Run cleanup in background thread
            cleanup_thread = threading.Thread(target=delayed_streaming_cleanup, daemon=True)
            cleanup_thread.start()

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
    tool_calls = getattr(response, "tool_calls", None) if hasattr(response, "tool_calls") else None

    # For OpenAI/Codex format: only clean if content contains tool calls
    # For other formats: apply syntax rewriting
    if syntax_rewriter.target_format in [SyntaxFormat.OPENAI, SyntaxFormat.CODEX]:
        # Only clean content if it contains tool call patterns
        # This prevents stripping spaces from regular text
        if any(pattern in content for pattern in ['<function_call>', '<tool_call>', '<|tool_call|>', '```tool_code']):
            content = syntax_rewriter.remove_tool_call_patterns(content)
    elif syntax_rewriter.target_format != SyntaxFormat.PASSTHROUGH:
        # Apply format-specific rewriting for non-OpenAI formats.
        # Prefer structured tool_calls when present (content may be cleaned upstream).
        if tool_calls:
            content = syntax_rewriter.rewrite_content(content, detected_tool_calls=tool_calls)
        else:
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
    if tool_calls:
        openai_tool_calls = syntax_rewriter.convert_to_openai_format(tool_calls)
        response_dict["choices"][0]["message"]["tool_calls"] = openai_tool_calls
        response_dict["choices"][0]["finish_reason"] = "tool_calls"

        logger.info(
            "🔧 Tool calls converted",
            request_id=request_id,
            tool_count=len(tool_calls),
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
# Server Runner Function
# ============================================================================

def run_server_with_args():
    """Run the server with argument parsing for CLI usage."""
    parser = argparse.ArgumentParser(
        description="AbstractCore Server - Universal LLM Gateway with Media Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m abstractcore.server.app                    # Start server with defaults
  python -m abstractcore.server.app --debug           # Start with debug logging
  python -m abstractcore.server.app --host 127.0.0.1 --port 8080  # Custom host/port
  python -m abstractcore.server.app --debug --port 8080           # Debug on custom port

Environment Variables:
  ABSTRACTCORE_DEBUG=true    # Enable debug mode (equivalent to --debug)
  HOST=127.0.0.1            # Server host (overridden by --host)
  PORT=8080                  # Server port (overridden by --port)

Debug Mode:
  The --debug flag enables verbose logging and better error reporting, including:
  - Detailed HTTP request/response logging
  - Full error traces for 422 Unprocessable Entity errors
  - Media processing diagnostics
  - Provider initialization details
        """
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging and show detailed diagnostics (overrides centralized config)'
    )
    parser.add_argument(
        '--host',
        default=os.getenv("HOST", "0.0.0.0"),
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help='Port to bind the server to (default: 8000)'
    )

    args = parser.parse_args()

    # Reconfigure logging if debug mode is requested (--debug overrides config defaults)
    if args.debug:
        reconfigure_for_debug()
        print("🐛 Debug mode enabled - detailed logging active")

    logger.info(
        "🚀 Starting AbstractCore Server",
        host=args.host,
        port=args.port,
        debug=debug_mode,
        version=__version__
    )

    # Enhanced uvicorn configuration for debug mode
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": "debug" if debug_mode else "info"
    }

    # In debug mode, enable more detailed uvicorn logging
    if debug_mode:
        uvicorn_config.update({
            "access_log": True,
            "use_colors": True,
        })

    import uvicorn
    uvicorn.run(**uvicorn_config)

# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    run_server_with_args()
