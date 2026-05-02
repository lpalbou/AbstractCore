# Observability / Tracing Examples

## What This Folder Teaches

How to instrument an LLM application so you can answer:
- “What did we send to the model?”
- “How long did prefill vs decode take?”
- “Which tools were called, and how long did they run?”
- “Why did a request fail (rate limit, network, invalid tool args)?”

## Key AbstractCore Concepts

- Interaction tracing: a normalized record of the request/response lifecycle.
- Events vs logs: events are structured signals you can aggregate; logs are for human inspection.

## How The Example Works

`interaction_tracing_demo.py` runs a small interaction and emits/export traces.
The point is to show the shape of telemetry you can build around AbstractCore without
vendor-specific parsing.

## Scripts

- `interaction_tracing_demo.py`
  - Demonstrates: capturing an interaction trace and exporting it for later inspection.
  - Takeaway: make traces part of your “debug toolbox” before you ship.

## Key Takeaways

- Observability is not optional in production: build it alongside features.
- Prefer structured traces/events over regexing model output.
