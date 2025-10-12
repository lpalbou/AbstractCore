# OllamaProvider Tool Role Message Support - Bug Report

## Issue Summary

**Title**: OllamaProvider fails to handle OpenAI `role: "tool"` messages, causing 400 Bad Request errors in agentic CLI workflows

**Priority**: High
**Component**: `abstractllm/providers/ollama_provider.py`
**Affects**: Codex CLI integration, multi-turn tool conversations

## Problem Description

The OllamaProvider directly passes OpenAI-format messages with `role: "tool"` to Ollama's API, but Ollama only supports roles: `["system", "user", "assistant"]`. This causes 400 Bad Request errors when agentic CLIs like Codex send tool result messages in follow-up requests.

## Root Cause Analysis

**File**: `abstractllm/providers/ollama_provider.py`
**Line**: 119
**Code**: `payload["messages"].extend(messages)`

The provider directly forwards all messages without checking if Ollama supports the message roles, specifically the OpenAI `"tool"` role used for tool execution results.

## Reproduction Steps

### Test Case 1: Direct Tool Role Message
```bash
# 1. Start AbstractCore server
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=openai
python -m abstractllm.server.app_simple

# 2. Send request with tool role message
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3-coder:30b",
    "messages": [
      {"role": "user", "content": "List files"},
      {"role": "assistant", "content": "I will list the files.", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "shell", "arguments": "{\"command\": [\"ls\"]}"}}]},
      {"role": "tool", "content": "file1.txt\nfile2.txt", "tool_call_id": "call_123"}
    ]
  }'
```

**Expected**: Server should handle tool result message gracefully
**Actual**: 500 Internal Server Error due to Ollama 400 Bad Request

### Test Case 2: Real Codex Workflow
```bash
# 1. Start server with OpenAI tool call tags
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=openai
python -m abstractllm.server.app_simple

# 2. Configure Codex to use the server
export OPENAI_BASE_URL="http://localhost:8000/v1"

# 3. Execute tool command
ABSTRACTCORE_API_KEY=dummy codex exec "list the local files" --model "ollama/qwen3-coder:30b"
```

**Expected**: Complete tool execution workflow
**Actual**:
- ✅ Initial tool call generation works
- ✅ Tool execution succeeds
- ❌ Follow-up request with tool results fails with 400 Bad Request

## Error Details

### Server Logs
```
[INFO] Chat Request | ollama/qwen3-coder:30b | messages=6 | tools=YES
[INFO] Tool execution disabled - tools will be generated but not executed
[INFO] HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 400 Bad Request"
[ERROR] Chat completion failed: 'NoneType' object has no attribute 'get'
```

### Ollama Response
```
HTTP/1.1 400 Bad Request
```

## Expected Behavior

The OllamaProvider should convert unsupported message roles to formats Ollama understands:

```python
# Convert tool messages to user messages with clear markers
{"role": "tool", "content": "file1.txt\nfile2.txt", "tool_call_id": "call_123"}
# Should become:
{"role": "user", "content": "[TOOL RESULT call_123]: file1.txt\nfile2.txt"}
```

## Proposed Solution

Add message role conversion logic to OllamaProvider, similar to the `convert_tool_messages_for_model()` function from the original server implementation:

```python
def _convert_messages_for_ollama(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI messages to Ollama-compatible format"""
    converted = []
    for msg in messages:
        if msg.get("role") == "tool":
            # Convert tool message to user message with markers
            tool_content = msg.get("content", "")
            tool_call_id = msg.get("tool_call_id", "unknown")
            converted.append({
                "role": "user",
                "content": f"[TOOL RESULT {tool_call_id}]: {tool_content}"
            })
        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Remove tool_calls from assistant messages (Ollama doesn't support them)
            converted.append({
                "role": "assistant",
                "content": msg.get("content", "")
            })
        else:
            # Keep supported roles as-is
            converted.append(msg)
    return converted
```

## Implementation Location

**File**: `abstractllm/providers/ollama_provider.py`
**Method**: `_generate_internal()` around line 118-119
**Change**: Replace `payload["messages"].extend(messages)` with converted messages

## Test Cases for Validation

### Test 1: Tool Role Conversion
```python
def test_tool_role_conversion():
    messages = [
        {"role": "user", "content": "List files"},
        {"role": "assistant", "content": "I'll list files", "tool_calls": [...]},
        {"role": "tool", "content": "file1.txt", "tool_call_id": "call_123"}
    ]
    converted = provider._convert_messages_for_ollama(messages)
    assert converted[2]["role"] == "user"
    assert "TOOL RESULT call_123" in converted[2]["content"]
```

### Test 2: Assistant Tool Calls Removal
```python
def test_assistant_tool_calls_removal():
    messages = [
        {"role": "assistant", "content": "I'll help", "tool_calls": [...]}
    ]
    converted = provider._convert_messages_for_ollama(messages)
    assert "tool_calls" not in converted[0]
    assert converted[0]["content"] == "I'll help"
```

### Test 3: Supported Roles Preserved
```python
def test_supported_roles_preserved():
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    converted = provider._convert_messages_for_ollama(messages)
    assert converted == messages  # Should be unchanged
```

## Related Issues

- Affects all agentic CLI integrations that use tool execution
- Impacts multi-turn conversations with tool usage
- Breaks the tool execution feedback loop

## Compatibility Notes

- OpenAI Provider: Supports `role: "tool"` natively
- Anthropic Provider: Has similar conversion logic
- Other providers: May need similar fixes

## Workaround

Currently, users must avoid multi-turn tool conversations with Ollama provider. Initial tool generation works, but follow-up conversations fail.

## References

- Original server implementation: `abstractllm/server/app.py` lines 747-785 (`convert_tool_messages_for_model()`)
- Codex tool detection documentation: `/Users/albou/projects/gh/codex/codex-tool-detection.md`
- OpenAI API specification: https://platform.openai.com/docs/api-reference/chat/create

---

**Reported**: 2025-10-12
**Severity**: High - Breaks agentic CLI integration
**Estimated Effort**: Medium - Requires message conversion logic implementation
**Testing**: Comprehensive test cases provided above