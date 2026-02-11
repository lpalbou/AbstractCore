# Event System

AbstractCore emits lightweight events for generation, tool execution, retries, sessions, and compaction. You can subscribe to these events for observability, progress reporting, or custom telemetry.

## Global event handlers

Use `on_global()` to register a process-wide handler:

```python
from abstractcore import create_llm
from abstractcore.events import EventType, on_global

def on_event(event):
    print(event.type.value, event.source, event.data)

on_global(EventType.GENERATION_COMPLETED, on_event)

llm = create_llm("ollama", model="qwen3:4b")
llm.generate("Say 'ok' and nothing else.")
```

## Common event types

- `EventType.GENERATION_STARTED` / `EventType.GENERATION_COMPLETED`
- `EventType.TOOL_STARTED` / `EventType.TOOL_PROGRESS` / `EventType.TOOL_COMPLETED`
- `EventType.RETRY_ATTEMPTED` / `EventType.RETRY_EXHAUSTED`
- `EventType.VALIDATION_FAILED`
- `EventType.COMPACTION_STARTED` / `EventType.COMPACTION_COMPLETED`

For the full list, see `abstractcore/events/__init__.py` and the [API Reference](api-reference.md#eventtype).

## Async handlers

If you need async handlers, use `GlobalEventBus.on_async(...)` (or per-emitter `on_async(...)`) and emit events with the async emission methods. Most application code only needs the sync hooks above.
