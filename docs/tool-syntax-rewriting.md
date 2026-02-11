# Tool Call Syntax Rewriting

AbstractCore can **convert tool-call syntax** to help different runtimes/clients consume tool calls consistently.

There are two related but distinct features:

1. **Python API (`tool_call_tags`)**: preserve and rewrite *tool-call markup inside assistant content* (mostly for prompted-tool models).
2. **HTTP Server (`agent_format`)**: convert/synthesize tool-call syntax for HTTP clients (Codex, other agentic CLIs), while keeping `tool_calls` structured.

## 1) Python API: `tool_call_tags` (per-call)

`tool_call_tags` is passed to `generate()` / `agenerate()` / `BasicSession.generate()` as a **per-call kwarg**.

### Default behavior (recommended)

- When `tool_call_tags is None` (default):
  - `response.tool_calls` is populated when tool calls are detected (native tools or prompted tags).
  - Tool-call markup is stripped from `response.content` for clean UX/history.

### When to set `tool_call_tags`

Set `tool_call_tags` when you want **tool-call markup kept in `content`** so a downstream consumer can parse it from text.

This is most useful for **prompted-tool** providers (tool calls are emitted in assistant content), e.g.:
- `ollama`
- `lmstudio`
- `mlx`
- `huggingface`
- `openai-compatible` (and compatible endpoints like vLLM / LM Studio)

For **native tool** providers (OpenAI/Anthropic), tool calls are primarily consumed from `response.tool_calls` (structured), not from tags embedded in `content`.

### Supported values

- Predefined formats:
  - `qwen3` → `<|tool_call|>...JSON...</|tool_call|>`
  - `llama3` → `<function_call>...JSON...</function_call>`
  - `xml` → `<tool_call>...JSON...</tool_call>`
  - `gemma` → ```tool_code\n...JSON...\n```
- Custom tags:
  - Comma-separated start/end: `"START,END"` or `"[TOOL],[/TOOL]"`
  - Single tag name: `"MYTAG"` → `<MYTAG>...JSON...</MYTAG>`

### Example (non-streaming)

```python
from abstractcore import create_llm

tool = {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}

llm = create_llm("ollama", model="qwen3:4b-instruct")
response = llm.generate(
    "Weather in Paris?",
    tools=[tool],
    tool_call_tags="llama3",
)

print(response.content)     # contains <function_call>...</function_call>
print(response.tool_calls)  # always structured dicts for host/runtime execution
```

### Example (streaming)

```python
tool_calls = []
for chunk in llm.generate(
    "Weather in Paris?",
    tools=[tool],
    stream=True,
    tool_call_tags="llama3",
):
    print(chunk.content, end="", flush=True)
    if chunk.tool_calls:
        tool_calls.extend(chunk.tool_calls)
```

## 2) HTTP Server: `agent_format`

When using the AbstractCore server (`/v1/chat/completions`), you can request a target tool-call syntax via `agent_format`.

- `agent_format` affects how tool calls are represented in the response for a given client.
- The server always runs in passthrough mode (`execute_tools=False`): it returns tool calls; it does not execute them.

### Supported values

- `auto` (default): auto-detect based on `User-Agent` + model name patterns
- `openai`
- `codex`
- `qwen3`
- `llama3`
- `xml`
- `gemma`
- `passthrough`

### Example

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3:4b-instruct",
    "messages": [{"role": "user", "content": "Weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
      }
    }],
    "agent_format": "codex"
  }'
```

## Notes

- `tool_call_tags` is **formatting**, not execution: it only changes how tool calls are represented in `content`.
- The canonical machine-readable representation remains `GenerateResponse.tool_calls` (Python) or `message.tool_calls` (server/OpenAI format).

