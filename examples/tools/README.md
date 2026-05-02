# Tools / Function Calling Examples

## What This Folder Teaches

How to connect an LLM to deterministic code via tools (“function calling”):
- define tool schemas,
- run tools safely,
- stream tool-call results,
- and observe tool lifecycle events.

## Prereqs

- Some scripts default to hosted models. For local-only usage, switch to Ollama or LM Studio models.
- Web/skim benchmarks require: `pip install "abstractcore[tools]"`

## Key AbstractCore Concepts

- Tools are *structured IO*: you describe inputs/outputs, and AbstractCore normalizes how providers call them.
- Tool execution should be treated like untrusted input: validate args, time out, and log.
- Events (tool start/end) are a better UI signal than parsing model tokens.

## How Tool Calling Works (Conceptually)

1) You pass `tools=[...]` (JSON schema-like definitions or function adapters).  
2) The model decides to call a tool and returns a structured “tool call”.  
3) Your app executes the tool and returns the tool result.  
4) The model uses the tool result to produce the final answer.

## Scripts

- `tool_usage_basic.py`
  - Demonstrates: defining tools with `@tool`, receiving `response.tool_calls`, executing them in the host, then asking the model to answer using the tool results.
  - Takeaway: AbstractCore normalizes tool-call extraction; your app controls execution and safety.
  - Tip: run with `--executor manual` to see how little code a custom dispatcher needs.

- `tool_usage_advanced.py`
  - Demonstrates: host-side tool policy (denylist/allowlist) + argument sanitization before execution.
  - Takeaway: production tool calling is mostly “policy + validation + observability”.

- `tooltag_cli_demo.py`
  - Demonstrates: `/tooltag` in the AbstractCore CLI (rewriting tool-call tags for model compatibility).
  - Takeaway: some models require tag/format tweaks; AbstractCore gives you a controlled place to do it.

- `skim_tools_benchmark.py` (benchmark/dev-oriented)
  - Demonstrates: measuring tool output size/latency for skim/fetch/web tools.
  - Takeaway: tool payload size matters; keep tool results compact and structured.

## Key Takeaways

- Tools make your system reliable by moving “actions” out of free-form text.
- Always validate tool args and set timeouts; treat model output as untrusted input.
