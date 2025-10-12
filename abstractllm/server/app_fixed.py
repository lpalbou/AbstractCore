"""
Fixed server implementation - Fragment showing the corrected streaming logic.

This replaces the problematic dual detection system with proper reliance on
the UnifiedStreamProcessor which already handles all edge cases correctly.
"""

def generate_openai_stream_fixed(llm, gen_kwargs, actual_provider, actual_model, request):
    """
    Fixed streaming implementation that properly relies on UnifiedStreamProcessor.

    Key changes:
    1. Remove duplicate tool detection logic
    2. Let UnifiedStreamProcessor handle ALL tool detection
    3. Trust the tool_calls from the streaming processor
    """
    import uuid
    import time
    import json

    try:
        gen_kwargs["stream"] = True
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_time = int(time.time())

        # The UnifiedStreamProcessor already handles tool detection correctly
        # We should NOT duplicate this logic in the server

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

            # Simply pass through the content from the processor
            # The UnifiedStreamProcessor has already:
            # 1. Detected tool calls
            # 2. Applied tag rewriting if configured
            # 3. Handled all edge cases (split chunks, etc.)

            if hasattr(chunk, 'content') and chunk.content:
                # Stream the content as-is - processor has already handled everything
                openai_chunk["choices"][0]["delta"]["content"] = chunk.content
                yield f"data: {json.dumps(openai_chunk)}\n\n"

            # Handle tool calls that were properly extracted by the processor
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
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
                                    "index": 0,  # Would need proper indexing
                                    "id": tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.name,
                                        "arguments": tool_call.arguments if isinstance(tool_call.arguments, str) else json.dumps(tool_call.arguments)
                                    }
                                }]
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(tool_call_chunk)}\n\n"

            # Handle finish reason
            if hasattr(chunk, 'finish_reason') and chunk.finish_reason:
                openai_chunk["choices"][0]["finish_reason"] = chunk.finish_reason
                yield f"data: {json.dumps(openai_chunk)}\n\n"

        # Final chunk
        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": f"{actual_provider}/{actual_model}",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


# ============================================================================
# PROPER CONFIGURATION OF UNIFIED PROCESSOR
# ============================================================================

def create_properly_configured_processor(model_name, tool_call_tags=None):
    """
    Create a properly configured UnifiedStreamProcessor.

    This ensures:
    1. Tool detection works for all formats
    2. Tag rewriting is applied when configured
    3. All edge cases are handled
    """
    from ..providers.streaming import UnifiedStreamProcessor

    # If tool_call_tags is specified, use it for rewriting
    # Otherwise, use the default format for the model
    if tool_call_tags:
        processor = UnifiedStreamProcessor(
            model_name=model_name,
            tool_call_tags=tool_call_tags,
            execute_tools=False  # Server doesn't execute, just converts
        )
    else:
        # Default to converting to OpenAI format for Codex compatibility
        processor = UnifiedStreamProcessor(
            model_name=model_name,
            tool_call_tags="openai",  # Convert to OpenAI format
            execute_tools=False
        )

    return processor