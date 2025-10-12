# Server Streaming Architecture Fix

## Problem Summary

The AbstractLLM server had **duplicate tool detection logic** that was causing inconsistent behavior, particularly with the `<function_call>` format used by LLaMA models. This was breaking Codex CLI integration.

### Root Cause

The server (`app.py`) was attempting to detect and parse tool calls **in addition to** the detection already performed by the `UnifiedStreamProcessor` in the core library. This created several issues:

1. **Inconsistent Detection**: The server's regex patterns didn't match all formats correctly
2. **Buffering Issues**: The server's manual buffering logic didn't handle split chunks properly
3. **Double Processing**: Tool calls were being processed twice, leading to errors
4. **Format Limitations**: The server only recognized specific hardcoded formats

## The Fix

### Architecture Before Fix

```
User Request
    ↓
Server (app.py)
    ↓
[DUPLICATE DETECTION LAYER]  ← Problem: Manual regex patterns
    - tool_call_buffer
    - in_tool_call state
    - Manual chunk assembly
    ↓
BaseProvider.generate()
    ↓
UnifiedStreamProcessor  ← Already handles ALL detection correctly
    ↓
Response to User
```

### Architecture After Fix

```
User Request
    ↓
Server (app.py)
    ↓
[REMOVED DUPLICATE LAYER]  ← Fix: Trust the core library
    ↓
BaseProvider.generate()
    ↓
UnifiedStreamProcessor  ← Single source of truth for detection
    ↓
Response to User
```

## Implementation Details

### 1. Streaming Path Fix

**Before** (Lines 1892-1984 in app.py):
```python
# Problematic duplicate detection
tool_call_buffer = ""
in_tool_call = False
tool_call_patterns = [
    (r'<\|tool_call\|>(.*?)</\|tool_call\|>', 'qwen3'),
    (r'<function_call>(.*?)</function_call>', 'llama'),
    (r'<tool_call>(.*?)</tool_call>', 'generic'),
]

# Manual buffering and detection
if has_tool_start or in_tool_call:
    tool_call_buffer += content
    in_tool_call = True
    # Complex regex matching and JSON parsing...
```

**After**:
```python
# Trust the UnifiedStreamProcessor
for chunk in llm.generate(**gen_kwargs):
    # Simply pass through content from the processor
    if hasattr(chunk, 'content') and chunk.content:
        # Stream as-is - processor already handled everything
        openai_chunk["choices"][0]["delta"]["content"] = chunk.content
        yield f"data: {json.dumps(openai_chunk)}\n\n"

    # Use tool calls extracted by the processor
    if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
        # Convert to OpenAI format and stream
```

### 2. Non-Streaming Path Fix

**Before** (Lines 1996-2057):
```python
# Duplicate detection in non-streaming
tool_call_patterns = [...]
for pattern, format_type in tool_call_patterns:
    matches = re.findall(pattern, content, re.DOTALL)
    # Manual extraction and parsing...
```

**After**:
```python
# Trust the BaseProvider's detection
response = llm.generate(**gen_kwargs)

# Use tool calls from the provider
if hasattr(response, 'tool_calls') and response.tool_calls:
    # Convert to OpenAI format
```

## Benefits of the Fix

### 1. **Consistency**
- All tool call formats now work identically
- No difference between streaming and non-streaming behavior
- Unified detection logic in one place

### 2. **Reliability**
- Proper handling of split chunks
- No lost tool calls at chunk boundaries
- Robust JSON parsing with auto-repair

### 3. **Performance**
- Eliminated redundant processing
- Reduced memory usage (no duplicate buffering)
- Faster streaming with less overhead

### 4. **Maintainability**
- ~100 lines of complex code removed
- Single source of truth for tool detection
- Easier to add new tool formats

## Supported Tool Call Formats

The UnifiedStreamProcessor (which the server now properly delegates to) supports:

1. **Qwen Format**: `<|tool_call|>{...}</|tool_call|>`
2. **LLaMA Format**: `<function_call>{...}</function_call>` ✅ Fixed!
3. **Generic Format**: `<tool_call>{...}</tool_call>`
4. **XML Format**: `<tool>...</tool>`
5. **Gemma Format**: `tool_code{...}`
6. **Custom Formats**: Via tool_call_tags parameter

## Testing the Fix

The fix ensures that all these scenarios work correctly:

```python
# Scenario 1: Complete tool call in single chunk
"<function_call>{\"name\": \"list_files\", \"arguments\": {}}</function_call>"
# ✅ Detected and converted

# Scenario 2: Tool call split across chunks
Chunk 1: "<function_call>{\"name\": \"lis"
Chunk 2: "t_files\", \"argumen"
Chunk 3: "ts\": {}}</function_call>"
# ✅ Properly buffered and detected

# Scenario 3: Multiple sequential tool calls
"<function_call>{...}</function_call>\n<function_call>{...}</function_call>"
# ✅ Both detected correctly

# Scenario 4: Mixed content and tool calls
"Let me check.\n<function_call>{...}</function_call>\nHere are the results."
# ✅ Content preserved, tool extracted
```

## Codex CLI Integration

This fix resolves the Codex CLI integration issue where `<function_call>` format wasn't being detected properly. The CLI can now:

1. Send requests with any tool call format
2. Receive consistent OpenAI-formatted responses
3. Handle streaming with proper tool detection
4. Work with all supported LLM providers

## Migration Notes

**No API changes required!** The fix is entirely internal to the server. Existing clients will automatically benefit from:

- Better tool detection accuracy
- Improved streaming performance
- Support for more tool formats
- More consistent behavior

## Architecture Principles

This fix reinforces important architectural principles:

1. **Don't Repeat Yourself (DRY)**: Tool detection logic exists in one place
2. **Single Responsibility**: Each component has one clear job
3. **Trust the Framework**: The core library handles the complexity
4. **Separation of Concerns**: Server focuses on API translation, not tool detection

## Conclusion

By removing the duplicate tool detection logic from the server and properly delegating to the `UnifiedStreamProcessor`, we've:

- ✅ Fixed the `<function_call>` detection issue
- ✅ Improved overall reliability
- ✅ Simplified the codebase
- ✅ Enabled better Codex CLI integration
- ✅ Maintained full backward compatibility

The server now properly acts as a thin API translation layer, delegating all complex tool detection to the battle-tested core library components.