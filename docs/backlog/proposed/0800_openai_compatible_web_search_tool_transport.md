# Proposed: OpenAI-compatible web search tool transport

## Metadata
- Created: 2026-05-21
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None identified in this pass
- ADR impact: None unless AbstractCore changes from pass-through transport into server-side tool execution.

## Context

Codex Memory uses AbstractCore's OpenAI-compatible gateway for the `mlx`, `transformer`, and
`gguf` routes. In that integration, Codex is the agent runtime: Codex defines the available tools,
Codex receives tool calls, and Codex executes tool calls. AbstractCore should not be required to
execute `web_search` for Codex.

The current problem is transport and shape compatibility. Codex currently has a Responses-native
`web_search` tool shape, but its Chat Completions conversion path drops non-function tools before
they reach AbstractCore. If Codex chooses to use AbstractCore's `/v1/chat/completions` route, the
primary fix belongs in Codex: expose `web_search` as an OpenAI-compatible function tool and register
a real Codex-side `web_search` handler.

This AbstractCore backlog item exists only for the endpoint side of the contract: if Codex, or any
other OpenAI-compatible client, uses AbstractCore's `/v1/responses` compatibility endpoint with
Responses-native tools, AbstractCore must not silently discard those tool declarations.

## Current code reality

Inspected on 2026-05-21:

- Codex-side evidence:
  - `codex-rs/core/src/tools/spec.rs` builds `ToolSpec::WebSearch` from `web_search_mode`.
  - `create_tools_json_for_chat_completions_api(...)` drops every non-function tool, so
    `ToolSpec::WebSearch` is never sent to Chat providers such as AbstractCore `mlx`,
    `transformer`, and `gguf`.
  - Codex has no local `web_search` function handler in its tool registry; the existing
    `web_search` path is a provider-hosted Responses tool represented by `ResponseItem::WebSearchCall`.
- AbstractCore-side evidence:
  - `abstractcore/server/app.py` `ChatCompletionRequest.tools` accepts OpenAI-compatible function
    tool dictionaries.
  - `/v1/chat/completions` forwards `request.tools` to provider generation and uses
    `execute_tools=False`, which is correct for Codex-owned tool execution.
  - `OpenAIResponsesRequest` does not define a `tools` field.
  - `convert_openai_responses_to_chat_completion(...)` cannot preserve Responses-native tool
    declarations such as `{"type": "web_search"}` because those declarations are absent from the
    parsed request model.

## Current broken examples

### Codex Chat payload today

When Codex routes through Chat Completions, the native Responses web-search tool has already been
filtered out. AbstractCore receives a request with no `web_search` tool definition:

```json
{
  "model": "mlx/mlx-community/Qwen3.6-35B-A3B-4bit",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "research current game development strategies"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "exec_command",
        "description": "...",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ],
  "tool_choice": "auto",
  "stream": true
}
```

The model is then correct to say that `web_search` is unavailable.

### AbstractCore `/v1/responses` payload today

If a client sends a Responses-style request with native web search, AbstractCore's current
`OpenAIResponsesRequest` model has no `tools` field. That means the endpoint cannot preserve this
declaration into the internal chat request:

```json
{
  "model": "mlx/mlx-community/Qwen3.6-35B-A3B-4bit",
  "input": [
    {
      "role": "user",
      "content": [{"type": "input_text", "text": "research current game development strategies"}]
    }
  ],
  "tools": [
    {"type": "web_search", "external_web_access": true}
  ],
  "stream": true
}
```

## Required working examples

### Preferred Codex-owned Chat contract

For `/v1/chat/completions`, Codex should provide `web_search` as a function tool and execute it
after AbstractCore returns a standard tool call:

```json
{
  "model": "mlx/mlx-community/Qwen3.6-35B-A3B-4bit",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "research current game development strategies"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string"},
            "recency": {"type": "integer"}
          },
          "required": ["query"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "stream": true
}
```

Expected AbstractCore/OpenAI-compatible response shape:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_web_1",
            "type": "function",
            "function": {
              "name": "web_search",
              "arguments": "{\"query\":\"current video game development strategies 2026\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

Then Codex, not AbstractCore, executes `web_search` and sends the tool result back in the next Chat
request:

```json
{
  "messages": [
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_web_1", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\":\"current video game development strategies 2026\"}"}}]},
    {"role": "tool", "tool_call_id": "call_web_1", "content": "{\"results\":[...]}"}
  ]
}
```

### Optional AbstractCore `/v1/responses` transport contract

If AbstractCore's `/v1/responses` endpoint is used, it should accept the same native Responses
`tools` field and either preserve it losslessly or convert known built-in/native tools into
OpenAI-compatible function tools for provider prompting. It must still leave execution to the host
unless an explicit server-side execution mode is requested.

Minimal required preservation:

```json
{
  "model": "mlx/mlx-community/Qwen3.6-35B-A3B-4bit",
  "input": [{"role": "user", "content": [{"type": "input_text", "text": "research current game development strategies"}]}],
  "tools": [{"type": "web_search", "external_web_access": true}],
  "stream": true
}
```

Internal equivalent after conversion:

```json
{
  "model": "mlx/mlx-community/Qwen3.6-35B-A3B-4bit",
  "messages": [{"role": "user", "content": [{"type": "text", "text": "research current game development strategies"}]}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

## Problem or opportunity

Codex can fix the current AbstractCore Chat route by using a function-shaped `web_search` tool and
a Codex-side handler. AbstractCore only needs backlog work if the `/v1/responses` compatibility
endpoint is expected to carry native Responses tools. In that case, silently ignoring `tools` makes
the endpoint look compatible while removing the key agent capability.

## Proposed direction

Keep the boundaries explicit:

- Codex owns the first implementation path:
  - convert `ToolSpec::WebSearch` to a Chat-compatible function schema for Chat providers;
  - register a Codex-side `web_search` handler;
  - execute returned `web_search` function calls in Codex and continue the tool loop.
- AbstractCore owns only endpoint transport compatibility if `/v1/responses` is used:
  - add `tools` to `OpenAIResponsesRequest`;
  - preserve or normalize native `{"type":"web_search"}` tools during
    `convert_openai_responses_to_chat_completion(...)`;
  - never execute web/network tools server-side unless an explicit request/config contract enables
    that mode.

## Why it might matter

Local open-source models behind AbstractCore should have access to the same model-visible tool
definitions that Codex exposes elsewhere. If the tool disappears in transport, the model cannot act
and may end the turn asking for unavailable capabilities.

## Promotion criteria

Promote this AbstractCore item only if Codex chooses to rely on AbstractCore `/v1/responses`, or if
another OpenAI-compatible client needs native Responses `tools` to survive that endpoint.

Do not promote this item merely to fix Codex's current Chat route; that fix belongs in Codex.

## Validation ideas

- Server unit test showing `OpenAIResponsesRequest` accepts a `tools` field.
- Conversion test showing native `{"type":"web_search"}` is preserved or normalized into a
  function-shaped `web_search` tool.
- Server test proving `/v1/responses` no longer drops `tools`.
- Negative test proving unsupported native tools fail loudly or are preserved explicitly, not
  silently ignored.
- End-to-end Codex test should live in Codex, not AbstractCore: Chat request includes function
  `web_search`, model returns standard `tool_calls`, Codex executes the handler, and the follow-up
  request includes a `role="tool"` result.

## Non-goals

- Do not teach clients to replace missing `web_search` with shell commands.
- Do not make AbstractCore responsible for Codex tool execution.
- Do not make all built-in AbstractCore tools globally available by default without an exposure
  policy.
- Do not silently execute web/network tools server-side.

## Guidance for future agents

First check whether Codex still uses `/v1/chat/completions` for AbstractCore. If yes, fix Codex and
leave this item proposed. Only edit AbstractCore if `/v1/responses` transport compatibility is
needed. In that case, start in `OpenAIResponsesRequest` and
`convert_openai_responses_to_chat_completion(...)`; keep execution out of AbstractCore unless a
separate explicit execution policy is accepted.
