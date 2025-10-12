#!/usr/bin/env python3
"""
Echo server that returns exactly what Codex showed us in the conversation.
"""

import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Echo Codex Format")

# Which format to echo back (change this to test different formats)
ECHO_FORMAT = "chat_completions"  # or "responses_api"

@app.post("/v1/chat/completions")
async def echo_codex_format(request: Request):
    """Echo back the exact format Codex showed us."""

    body = await request.body()
    parsed = json.loads(body.decode())

    print("="*80)
    print("📥 REQUEST FROM CODEX")
    print("="*80)
    print(f"Stream: {parsed.get('stream', False)}")
    print(f"Messages: {len(parsed.get('messages', []))}")

    if parsed.get("stream", False):
        return StreamingResponse(
            echo_streaming_format(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        return echo_non_streaming_format()

def echo_streaming_format():
    """Echo back the exact streaming format Codex expects."""

    print(f"📤 ECHOING FORMAT: {ECHO_FORMAT}")

    if ECHO_FORMAT == "responses_api":
        # Echo back the Responses API format Codex showed us
        content = '{"type":"response.output_item.done","item":{"type":"function_call","name":"shell","arguments":"{\\"command\\":[\\"ls\\",\\"-la\\"]}","call_id":"call_abc123"}}'

        chunk = {
            "id": "test-123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "echo-model",
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        }

        print(f"Echoing content: {content}")
        yield f"data: {json.dumps(chunk)}\n\n"

    elif ECHO_FORMAT == "chat_completions":
        # Echo back the Chat Completions API format Codex showed us
        tool_chunk = {
            "id": "test-123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "echo-model",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command":["ls","-la"]}'  # EXACTLY as Codex showed
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

        print(f"Echoing tool chunk: {json.dumps(tool_chunk)}")
        yield f"data: {json.dumps(tool_chunk)}\n\n"

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

    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

def echo_non_streaming_format():
    """Echo back non-streaming format."""

    if ECHO_FORMAT == "chat_completions":
        return {
            "id": "test-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "echo-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command":["ls","-la"]}'  # EXACTLY as Codex showed
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{
            "id": "echo-model",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "echo"
        }]
    }

# Control endpoint to switch formats
@app.post("/control/format/{format_name}")
async def set_format(format_name: str):
    global ECHO_FORMAT
    if format_name in ["chat_completions", "responses_api"]:
        ECHO_FORMAT = format_name
        print(f"🔧 Switched to format: {ECHO_FORMAT}")
        return {"status": "ok", "format": ECHO_FORMAT}
    return {"error": "Invalid format"}

@app.get("/control/status")
async def status():
    return {"format": ECHO_FORMAT}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Echo Server - Returns EXACTLY what Codex showed us")
    print("📋 Available formats:")
    print("  - chat_completions: Tool calls in delta.tool_calls")
    print("  - responses_api: Tool calls as content")
    print(f"🔧 Current format: {ECHO_FORMAT}")
    print("💡 Change format: curl -X POST http://localhost:8000/control/format/responses_api")
    uvicorn.run(app, host="0.0.0.0", port=8000)