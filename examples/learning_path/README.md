# Learning Path (Long-Form)

This folder is the “guided track”: 6 scripts meant to be read in order. Each file is intentionally verbose and heavily commented so you can understand *why* the code is structured the way it is.

## What This Folder Teaches

How to go from “hello world” to “production-shaped” code using AbstractCore’s core abstractions:
- providers (`create_llm(...)`),
- sessions (system prompt + history + tools),
- streaming,
- tool calling,
- and operational patterns (retries, observability, configuration).

## How To Use It

Start with:
- `01_basic_generation.py`

Then continue in order:
- `02_provider_configuration.py`
- `03_tool_calling.py`
- `04_unified_streaming.py`
- `05_server_agentic_cli.py`
- `06_production_patterns.py`

Most scripts can run against local providers (Ollama/LM Studio) or hosted providers.
If a script defaults to a hosted provider, switch it by editing the `create_llm(...)` call.

## What Each Part Demonstrates

- `01_basic_generation.py`
  - Demonstrates: minimal provider + generation call.
  - Teaches: what a “provider” is in AbstractCore.

- `02_provider_configuration.py`
  - Demonstrates: configuration knobs (timeouts, retries, base URLs, token limits).
  - Teaches: how to make provider choice/config a runtime concern.

- `03_tool_calling.py`
  - Demonstrates: tool schemas and tool execution flow.
  - Teaches: how tools become structured IO instead of “prompt hacks”.

- `04_unified_streaming.py`
  - Demonstrates: streaming across providers with a consistent interface.
  - Teaches: how to build “fast feedback” UX without coupling to a specific backend.

- `05_server_agentic_cli.py`
  - Demonstrates: server mode + agentic CLI workflow patterns.
  - Teaches: how to expose capabilities over HTTP and keep a clean boundary between client/server.

- `06_production_patterns.py`
  - Demonstrates: production concerns (observability, safer defaults, error handling patterns).
  - Teaches: what to measure/log, and how to keep behavior predictable across models/providers.

## Key Takeaways

- AbstractCore’s value is composability: provider ↔ session ↔ tools ↔ observability stay decoupled.
- Prefer “thin examples you can copy” over black-box demos you can’t adapt.
