# Codex Tool Call Detection and ReAct Loops

## How Codex Works

### 1. Tool Call Detection

Codex detects tool calls from LLM responses in two ways:

#### Chat Completions API
- **Trigger**: `finish_reason: "tool_calls"` in streaming response
- **Process**: Accumulates tool call data across multiple chunks using `fn_call_state`
- **Limitation**: Only processes the first tool call in the array (`tool_calls.first()`)
- **Code Reference**: `codex-rs/core/src/chat_completions.rs:544-570`

#### Responses API  
- **Trigger**: `type: "response.output_item.done"` SSE events
- **Process**: Each event processed immediately via `serde_json::from_value::<ResponseItem>`
- **Advantage**: Supports parallel tool calls
- **Code Reference**: `codex-rs/core/src/client.rs:745-756`

### 2. ReAct Loop

Codex implements a ReAct (Reasoning and Acting) loop:

1. **User input** → Added to conversation history
2. **LLM processes** → May generate tool calls or final response
3. **Tool calls executed** → Results added to conversation history via `record_conversation_items()`
4. **Loop continues** → Until no more tool call results need to be sent back to LLM

**Loop ends when**: `responses.is_empty()` (no more tool call results to send back to LLM)
**Code Reference**: `codex-rs/core/src/codex.rs:1866-1877`

**Key insight**: The `responses` vector contains `ResponseInputItem` (tool call results). When not empty, the loop continues and sends the full conversation history (including tool results) back to the LLM via `turn_input_with_history()`.

---

## Exact Payload Formats

### Tool Call Detection (Chat Completions)

```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_123",
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

**Code Reference**: `codex-rs/core/src/chat_completions.rs:545-570`

### Tool Call Detection (Responses API)

```json
{
  "type": "response.output_item.done",
  "item": {
    "type": "function_call",
    "call_id": "call_123",
    "name": "shell",
    "arguments": "{\"command\":[\"ls\",\"-la\"]}"
  }
}
```

**Code Reference**: `codex-rs/protocol/src/models.rs:68-78` (with `#[serde(tag = "type", rename_all = "snake_case")]`)

### Tool Execution Result

```json
{
  "type": "function_call_output",
  "call_id": "call_123",
  "output": {
    "content": "file1.txt\nfile2.txt",
    "success": true
  }
}
```

**Code Reference**: `codex-rs/protocol/src/models.rs:85-87` and `codex-rs/protocol/src/models.rs:260-264`

---

## Multi-Step ReAct Example

### Step 1: User Input
```json
{
  "type": "message",
  "role": "user",
  "content": [{"type": "input_text", "text": "Create a file and write to it"}]
}
```

### Step 2: LLM Generates Tool Call
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_create_001",
        "type": "function",
        "function": {
          "name": "shell",
          "arguments": "{\"command\":[\"touch\",\"/tmp/test.txt\"]}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Step 3: Tool Execution Result
```json
{
  "type": "function_call_output",
  "call_id": "call_create_001",
  "output": {
    "content": "File created",
    "success": true
  }
}
```

### Step 4: LLM Generates Another Tool Call
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_write_001",
        "type": "function",
        "function": {
          "name": "shell",
          "arguments": "{\"command\":[\"echo\",\"Hello World\",\">\",\"/tmp/test.txt\"]}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Step 5: Tool Execution Result
```json
{
  "type": "function_call_output",
  "call_id": "call_write_001",
  "output": {
    "content": "Content written",
    "success": true
  }
}
```

### Step 6: LLM Generates Final Response
```json
{
  "choices": [{
    "delta": {
      "content": "Task completed! I created the file and wrote 'Hello World' to it."
    },
    "finish_reason": "stop"
  }]
}
```

**Loop ends** because `responses.is_empty()` returns true (no more tool call results to send back)

---

## Implementation Requirements

### Critical Requirements

1. **Arguments must be JSON strings**:
   ```python
   arguments = json.dumps({"command": ["ls"]})  # ✅ Correct
   arguments = {"command": ["ls"]}              # ❌ Wrong
   ```
   **Code Reference**: `codex-rs/protocol/src/models.rs:72-76`

2. **Unique call IDs required**:
   ```python
   call_id = f"call_{uuid.uuid4().hex[:8]}"
   ```
   **Code Reference**: `codex-rs/protocol/src/models.rs:77`

3. **Correct finish reason for Chat Completions**:
   ```python
   "finish_reason": "tool_calls"  # Required for execution
   ```
   **Code Reference**: `codex-rs/core/src/chat_completions.rs:575`

4. **Streaming support essential** for real-time detection

### API Selection

- **Chat Completions API**: Single tool calls, simpler integration
- **Responses API**: Parallel tool calls, advanced features

### Tool Routing

Codex routes tool calls based on name patterns:
- **MCP tools**: `server.tool_name` format → `ToolPayload::Mcp`
- **Unified exec**: `unified_exec` → `ToolPayload::UnifiedExec`  
- **Other functions**: → `ToolPayload::Function`

**Code Reference**: `codex-rs/core/src/tools/router.rs:68-89`

### ReAct Loop Details

The ReAct loop works as follows:

1. **Tool calls detected** → Added to `items_to_record_in_conversation_history`
2. **Tool calls executed** → Results added to `responses` vector as `ResponseInputItem`
3. **Results recorded** → Added to conversation history via `record_conversation_items()`
4. **Loop continues** → Until `responses.is_empty()` (no more results to send back)
5. **Next iteration** → Full conversation history sent to LLM via `turn_input_with_history()`

**Code Reference**: 
- Main loop: `codex-rs/core/src/codex.rs:1670-1878`
- Tool call processing: `codex-rs/core/src/codex.rs:2145-2161`
- History recording: `codex-rs/core/src/codex.rs:1837-1838`
- History retrieval: `codex-rs/core/src/codex.rs:1697-1698`

---

## Summary

**How Codex recognizes tool calls**:
- Chat Completions: `finish_reason: "tool_calls"` triggers execution (`codex-rs/core/src/chat_completions.rs:575`)
- Responses API: `"response.output_item.done"` events processed immediately (`codex-rs/core/src/client.rs:745`)

**How Codex implements ReAct loops**:
- Continuous loop until `responses.is_empty()` returns true (`codex-rs/core/src/codex.rs:1866`)
- Full conversation history maintained across turns via `record_conversation_items()` and `turn_input_with_history()`
- Tool calls executed immediately and results added to history
- Loop ends when no more tool call results need to be sent back to LLM

**Key requirements for integration**:
- Arguments as JSON strings (`codex-rs/protocol/src/models.rs:76`)
- Unique call IDs (`codex-rs/protocol/src/models.rs:77`)
- Streaming support
- Proper error handling