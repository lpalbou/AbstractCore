# Critical Fix: `<function_call>` Tag Detection and Conversion

## Problem Summary

The user reported that `<function_call>` tags are appearing in the output instead of being detected and converted to OpenAI format. Some tool calls work while others fail inconsistently in the same session.

## Root Cause

The issue is in **`abstractllm/server/app.py`** where the server implements its own **duplicate and flawed tool detection logic** that interferes with the proper `UnifiedStreamProcessor`.

### Key Problems:

1. **Dual Detection Systems**:
   - `UnifiedStreamProcessor` in `streaming.py` (works correctly)
   - Server's regex detection in `app.py` lines 1924-1982 (buggy)

2. **Flawed Buffering Logic**:
   ```python
   # Current buggy code in app.py
   if has_tool_start or in_tool_call:
       tool_call_buffer += content
       in_tool_call = True
   ```
   This fails when:
   - Tool calls are split across many small chunks
   - Character-by-character streaming occurs
   - Partial tags arrive in separate chunks

3. **State Management Issues**: The `in_tool_call` flag doesn't properly track state across streaming chunks.

## The Fix

Replace the streaming logic in `app.py` (lines 1911-2049) with this corrected version:

```python
def generate_openai_stream():
    try:
        gen_kwargs["stream"] = True
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_time = int(time.time())
        tool_calls_collected = []  # Track all tool calls for proper indexing

        # REMOVE all the duplicate detection logic (lines 1902-1910, 1924-1982)
        # The UnifiedStreamProcessor already handles everything correctly

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

            # Simply pass through content - processor has already handled detection
            if hasattr(chunk, 'content') and chunk.content:
                openai_chunk["choices"][0]["delta"]["content"] = chunk.content
                yield f"data: {json.dumps(openai_chunk)}\n\n"

            # Handle properly extracted tool calls from the processor
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
                                    "index": len(tool_calls_collected),
                                    "id": tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.name,
                                        "arguments": json.dumps(tool_call.arguments)
                                    }
                                }]
                            },
                            "finish_reason": None
                        }]
                    }
                    tool_calls_collected.append(tool_call)
                    yield f"data: {json.dumps(tool_call_chunk)}\n\n"

        # Final chunk
        finish_reason = "tool_calls" if tool_calls_collected else "stop"
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
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
```

## Why This Fixes the Issue

1. **Single Source of Truth**: Only the `UnifiedStreamProcessor` handles tool detection, eliminating conflicts
2. **Proper Buffering**: The processor already handles split chunks, partial tags, and character-by-character streaming
3. **Consistent State**: The processor maintains proper state across all chunks
4. **Format Preservation**: When configured with `tool_call_tags`, the processor correctly preserves/rewrites tags

## Additional Configuration

Ensure the `UnifiedStreamProcessor` is properly configured in the `BaseProvider`:

```python
# In BaseProvider.generate() when streaming is enabled
if stream and tool_call_tags:
    # Use UnifiedStreamProcessor for proper detection and rewriting
    from .streaming import UnifiedStreamProcessor
    processor = UnifiedStreamProcessor(
        model_name=self.model,
        tool_call_tags=tool_call_tags,
        execute_tools=execute_tools
    )
    # Process the stream through the unified processor
    return processor.process_stream(response_stream, converted_tools)
```

## Testing

To verify the fix works:

1. Test with character-by-character streaming
2. Test with tool calls split across many chunks
3. Test with multiple sequential tool calls
4. Test with different models (LLaMA, Qwen, etc.)

## Immediate Workaround

If you can't modify the server code immediately, ensure:
1. Set `ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=openai` environment variable
2. This forces conversion to OpenAI format at the provider level

## Long-term Solution

The server should be refactored to:
1. Remove ALL duplicate tool detection logic
2. Rely entirely on the `UnifiedStreamProcessor`
3. Trust the processor's output for both content and tool calls

This ensures consistent behavior across all scenarios and eliminates the bug where `<function_call>` tags appear in the output.