#!/usr/bin/env python3
"""
Minimal echo server for testing Codex tool call format expectations.

This server will:
1. Log exactly what Codex sends
2. Return manually crafted responses for testing
3. Allow us to test different tool call formats
"""

import json
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Minimal Echo Server for Codex Testing")

# Global variables to control responses
RESPONSE_MODE = "chat_completions"  # or "responses_api"
TOOL_CALL_FORMAT = "openai_correct"  # or "openai_malformed", "responses_api"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint."""

    print("="*80)
    print("📥 RECEIVED REQUEST FROM CODEX")
    print("="*80)

    # Log the raw request
    body = await raw_request.body()
    print(f"Raw body: {body.decode()}")

    # Log the parsed request
    print(f"Model: {request.model}")
    print(f"Stream: {request.stream}")
    print(f"Messages: {len(request.messages)}")
    print(f"Tools: {len(request.tools) if request.tools else 0}")

    for i, msg in enumerate(request.messages):
        print(f"  Message {i}: {msg.role} - {msg.content[:100]}...")

    if request.tools:
        for i, tool in enumerate(request.tools):
            print(f"  Tool {i}: {tool.get('function', {}).get('name', 'unknown')}")

    print("="*80)
    print("📤 SENDING RESPONSE")
    print("="*80)

    if request.stream:
        return StreamingResponse(
            generate_streaming_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        return generate_non_streaming_response()

def generate_streaming_response():
    """Generate a streaming response with tool calls."""

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())

    print(f"🔄 Generating streaming response (format: {TOOL_CALL_FORMAT})")

    # First chunk - some content
    first_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "I'll help you with that. Let me run the command."
            },
            "finish_reason": None
        }]
    }

    print(f"Chunk 1: {json.dumps(first_chunk)}")
    yield f"data: {json.dumps(first_chunk)}\n\n"

    # Tool call chunk
    tool_chunk = generate_tool_call_chunk(chat_id, created_time)
    print(f"Tool chunk: {json.dumps(tool_chunk)}")
    yield f"data: {json.dumps(tool_chunk)}\n\n"

    # Final chunk
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    }

    print(f"Final chunk: {json.dumps(final_chunk)}")
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

def generate_tool_call_chunk(chat_id: str, created_time: int) -> Dict[str, Any]:
    """Generate tool call chunk in different formats for testing."""

    if TOOL_CALL_FORMAT == "openai_correct":
        # Correct OpenAI format with properly escaped JSON string
        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": json.dumps({"command": ["ls", "-la"], "workdir": "/tmp"})  # Properly encoded
            }
        }

    elif TOOL_CALL_FORMAT == "openai_malformed":
        # Malformed OpenAI format (what user might be seeing)
        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": '{"command": ["ls", "-la"], "workdir": "/tmp"}'  # Unescaped quotes
            }
        }

    elif TOOL_CALL_FORMAT == "responses_api":
        # Responses API format (what user asked Codex to echo)
        return {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": json.dumps({
                        "type": "function_call",
                        "name": "shell",
                        "arguments": json.dumps({"command": ["ls", "-la"], "workdir": "/tmp"}),
                        "call_id": f"call_{uuid.uuid4().hex[:8]}"
                    })
                },
                "finish_reason": None
            }]
        }

    # Default to OpenAI format
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    **tool_call
                }]
            },
            "finish_reason": None
        }]
    }

def generate_non_streaming_response() -> Dict[str, Any]:
    """Generate non-streaming response for testing."""

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())

    print(f"📝 Generating non-streaming response (format: {TOOL_CALL_FORMAT})")

    if TOOL_CALL_FORMAT == "openai_correct":
        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": json.dumps({"command": ["ls", "-la"], "workdir": "/tmp"})
            }
        }
    elif TOOL_CALL_FORMAT == "openai_malformed":
        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": '{"command": ["ls", "-la"], "workdir": "/tmp"}'
            }
        }

    response = {
        "id": chat_id,
        "object": "chat.completion",
        "created": created_time,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I'll help you with that command.",
                "tool_calls": [tool_call]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60
        }
    }

    print(f"Response: {json.dumps(response, indent=2)}")
    return response

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{
            "id": "test-model",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "test"
        }]
    }

# Control endpoints for testing
@app.post("/control/set_format")
async def set_format(format_type: str):
    """Set the tool call format for testing."""
    global TOOL_CALL_FORMAT

    valid_formats = ["openai_correct", "openai_malformed", "responses_api"]
    if format_type not in valid_formats:
        return {"error": f"Invalid format. Must be one of: {valid_formats}"}

    TOOL_CALL_FORMAT = format_type
    print(f"🔧 Set format to: {TOOL_CALL_FORMAT}")
    return {"status": "ok", "format": TOOL_CALL_FORMAT}

@app.get("/control/status")
async def get_status():
    """Get current server status."""
    return {
        "format": TOOL_CALL_FORMAT,
        "response_mode": RESPONSE_MODE
    }

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Minimal Echo Server for Codex Testing")
    print("="*80)
    print("USAGE:")
    print("1. Start server: python minimal_echo_server.py")
    print("2. Configure Codex to use: http://localhost:8000")
    print("3. Change format: curl -X POST http://localhost:8000/control/set_format -d 'openai_correct'")
    print("4. Check status: curl http://localhost:8000/control/status")
    print("")
    print("Available formats:")
    print("  - openai_correct: Properly escaped JSON arguments")
    print("  - openai_malformed: Unescaped arguments (what you might be seeing)")
    print("  - responses_api: Format you asked Codex to echo")
    print("="*80)

    uvicorn.run(app, host="0.0.0.0", port=8000)