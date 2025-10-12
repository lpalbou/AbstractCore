#!/usr/bin/env python3
"""
TRUE echo server - logs what it receives and echoes it back exactly.
"""

import json
import time
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="True Echo Server")

@app.post("/v1/chat/completions")
async def echo_chat_completions(request: Request):
    """Echo back exactly what we receive."""

    # Get raw request body
    body = await request.body()
    headers = dict(request.headers)

    print("="*80)
    print("📥 RAW REQUEST FROM CODEX")
    print("="*80)
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers: {json.dumps(dict(headers), indent=2)}")
    print(f"Body: {body.decode()}")
    print("="*80)

    try:
        # Parse the JSON to understand what Codex sent
        parsed = json.loads(body.decode())
        print("📋 PARSED REQUEST:")
        print(json.dumps(parsed, indent=2))
        print("="*80)

        # Check if it's streaming
        is_streaming = parsed.get("stream", False)
        print(f"🔄 Streaming: {is_streaming}")

        if is_streaming:
            # Return a simple streaming response
            return StreamingResponse(
                echo_streaming_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Return simple non-streaming response
            return echo_non_streaming_response()

    except Exception as e:
        print(f"❌ Error parsing request: {e}")
        return JSONResponse({"error": "Invalid request"}, status_code=400)

def echo_streaming_response():
    """Return minimal streaming response."""
    print("📤 SENDING STREAMING RESPONSE")

    # Just send a simple text response, no tool calls
    chunk = {
        "id": "test-123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "echo-model",
        "choices": [{
            "index": 0,
            "delta": {"content": "This is a simple echo response - no tool calls."},
            "finish_reason": None
        }]
    }

    print(f"Chunk: {json.dumps(chunk)}")
    yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final = {
        "id": "test-123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "echo-model",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }

    print(f"Final: {json.dumps(final)}")
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

def echo_non_streaming_response():
    """Return minimal non-streaming response."""
    print("📤 SENDING NON-STREAMING RESPONSE")

    response = {
        "id": "test-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "echo-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a simple echo response - no tool calls."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    }

    print(f"Response: {json.dumps(response, indent=2)}")
    return response

@app.get("/v1/models")
async def echo_models():
    """Return minimal models list."""
    return {
        "object": "list",
        "data": [{
            "id": "echo-model",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "echo"
        }]
    }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    """Catch all other requests and log them."""
    body = await request.body()

    print("="*80)
    print(f"📥 UNEXPECTED REQUEST: {request.method} /{path}")
    print("="*80)
    print(f"Headers: {dict(request.headers)}")
    print(f"Body: {body.decode() if body else 'No body'}")
    print("="*80)

    return JSONResponse({"error": f"Endpoint /{path} not implemented"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TRUE Echo Server")
    print("📝 This server will log EXACTLY what it receives from Codex")
    print("🔍 Use this to see what Codex is actually sending")
    uvicorn.run(app, host="0.0.0.0", port=8000)