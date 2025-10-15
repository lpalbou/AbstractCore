# Universal Tool Call Conversion Fix

## Problem Description

The AbstractCore server was not properly converting tool calls to OpenAI's JSON format for all models. This caused issues with agentic CLIs like Codex/Cline that expect tool calls in the standard OpenAI format.

### Symptoms

1. **qwen3-coder models**: Tool calls worked (accidentally, due to previous local model conversion)
2. **qwen3-next-80b models**: Tool calls appeared as raw XML tags in the output
3. **Codex/Cline**: Could not recognize or execute tool calls from certain models

### Root Cause

The server was receiving tool calls as raw text content with XML-like tags (`<|tool_call|>`, `<function_call>`, etc.) but was not detecting and converting them to the OpenAI JSON format that Codex expects.

## Solution

Implemented universal tool call detection and conversion in the server that:

1. **Detects tool calls** in any format during streaming
2. **Buffers content** until complete tool call is received
3. **Parses JSON** from within the tool call tags
4. **Converts to OpenAI format** with proper `tool_calls` structure
5. **Works for all models** regardless of their native tool call format

## Technical Details

### Streaming Mode Changes

```python
# Before: Only handled pre-extracted tool_calls
if hasattr(chunk, 'content') and chunk.content:
    # Stream raw content including tool call tags
    openai_chunk["choices"][0]["delta"]["content"] = chunk.content

# After: Universal detection and conversion
if hasattr(chunk, 'content') and chunk.content:
    # Detect tool call markers
    if '<|tool_call|>' in content or '<function_call>' in content:
        # Buffer and parse complete tool calls
        # Convert to OpenAI format
        openai_chunk["choices"][0]["delta"]["tool_calls"] = [...]
```

### Non-Streaming Mode Changes

```python
# Extract tool calls from content
tool_call_patterns = [
    (r'<\|tool_call\|>(.*?)</\|tool_call\|>', 'qwen3'),
    (r'<function_call>(.*?)</function_call>', 'llama'),
    (r'<tool_call>(.*?)</tool_call>', 'generic'),
]

# Parse and convert to OpenAI format
for pattern, format_type in tool_call_patterns:
    matches = re.findall(pattern, content)
    # Convert matches to tool_calls array
```

## OpenAI Tool Call Format

The server now outputs tool calls in the standard OpenAI format:

```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "index": 0,
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "shell",
          "arguments": "{\"command\": [\"ls\", \"-la\"]}"
        }
      }]
    }
  }]
}
```

## Files Modified

1. **`abstractcore/server/app.py`**
   - Lines 1893-1989: Added universal tool call detection in streaming
   - Lines 2055-2117: Added universal tool call detection in non-streaming

## Testing

### Manual Test

```bash
# Start the server
python -m abstractcore.server

# Run the test script
chmod +x test_manual_tool_conversion.sh
./test_manual_tool_conversion.sh
```

### Python Test

```bash
# Start the server
python -m abstractcore.server

# Run the Python test
python test_server_tool_conversion.py
```

### Expected Results

✅ **Success**: Tool calls appear in OpenAI JSON format with `tool_calls` field
❌ **Failure**: Raw tags like `<|tool_call|>` appear in content

## Compatibility

This fix maintains full backward compatibility:

- **Models that output OpenAI format**: Continue to work as before
- **Models with native tool_calls**: Continue to work as before
- **Models with raw XML tags**: Now properly converted to OpenAI format
- **All agentic CLIs**: Now receive consistent OpenAI-formatted tool calls

## Performance Impact

- **Minimal overhead**: Pattern matching only when tool markers detected
- **Efficient buffering**: Only buffers when inside tool call
- **No impact on regular content**: Regular text streams immediately

## Future Improvements

1. **Add more tool call formats** as new models are released
2. **Optimize regex patterns** for better performance
3. **Add configuration option** to disable conversion if needed
4. **Enhanced error handling** for malformed tool calls