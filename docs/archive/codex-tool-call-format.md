# Codex Tool Call Format Requirements

## Exact Format Required

Codex requires tool calls in this **exact format** for streaming responses:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "shell",
          "arguments": "{\"command\":[\"ls\",\"-la\"]}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Critical Requirements

1. **Arguments MUST be a JSON string**: `"{\"command\":[\"ls\",\"-la\"]}"`
   - NOT a JSON object: `{"command":["ls","-la"]}`
   - NOT unescaped: `"{"command":["ls","-la"]}"`

2. **Tool calls MUST be in `delta.tool_calls` array**

3. **Finish reason MUST be `"tool_calls"`**

## Verified Working Example

This exact format was tested and confirmed to work with Codex:

```json
"function": {
  "name": "shell",
  "arguments": "{\"command\":[\"ls\",\"-la\"]}"
}
```

**Status**: ✅ VERIFIED - Codex executed this 50+ times successfully