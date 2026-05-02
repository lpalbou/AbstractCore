# CLI Examples

## What This Folder Teaches

How to build a command-line app around AbstractCore sessions:
- async/non-async generation,
- streaming output,
- tool calling,
- and progress/telemetry events you can surface in UI.

## Key AbstractCore Concepts

- `create_llm(...)`: selects a backend (Ollama/LM Studio/etc) behind a unified interface.
- `BasicSession`: holds the system prompt, tool schema, and message history.
- `GlobalEventBus`: emits lifecycle events (tool start/end, retries, etc) so a UI can react without parsing model text.

## How The Example Works

`async_cli_demo.py` is a reference implementation of “event-driven CLI”:

1) create provider → 2) create session with tools → 3) run an input loop  
4) for each user message: call `agenerate(...)` (optionally streaming)  
5) render tool progress using events from `GlobalEventBus`

## Scripts

- `async_cli_demo.py`
  - Demonstrates: async generation + async streaming, tool progress events, and a minimal “spinner UI”.
  - How it works: follow `AsyncCLIDemo._setup_event_handlers()` to see how tool events are wired, then `run()`/`handle_input()` for the async loop.

## Key Takeaways

- Prefer sessions (`BasicSession`) over “raw provider calls” once you have tools + history.
- Prefer events (`GlobalEventBus`) for UI status; don’t try to infer tool phases from tokens.
