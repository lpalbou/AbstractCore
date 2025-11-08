# Events Module

## Purpose

This module provides a lightweight, OpenTelemetry-compatible event system for tracking LLM operations, tool calls, structured output validation, and performance metrics throughout AbstractCore. It enables observability and monitoring without adding complexity.

## Architecture Position

- **Layer**: Foundation (Layer 0)
- **Dependencies**: None (standard library only: `typing`, `enum`, `dataclasses`, `datetime`, `uuid`)
- **Used By**: Providers, tools, structured output handlers, session management, compression

## Philosophy

The event system follows **SOTA minimal design principles**:
- **12 event types total** (vs. 30+ in complex frameworks)
- **No streaming events** (use GENERATION_STARTED/COMPLETED)
- **No token-level events** (too granular)
- **No separate chunk events** (capture in GENERATION_COMPLETED)
- **Compatible with OpenTelemetry** semantic conventions
- **Zero dependencies** beyond Python standard library

## Component Structure

### Core Classes

1. **EventType** (Enum) - 12 standardized event types
2. **Event** (Dataclass) - Simple event structure
3. **EventEmitter** (Mixin) - Instance-level event emission
4. **GlobalEventBus** (Singleton) - System-wide event bus
5. **EventLogger** - Basic event logging
6. **PerformanceTracker** - Metrics aggregation

### Helper Functions

- `on_global()` - Register global event handler
- `emit_global()` - Emit global event
- `create_generation_event()` - Standardized generation event data
- `create_tool_event()` - Standardized tool event data
- `create_structured_output_event()` - Standardized structured output event data

## Event Types

### Core Events (4)

#### GENERATION_STARTED
**When**: Before LLM generation begins (both streaming and non-streaming)

**Data**:
```python
{
    "model_name": str,
    "provider_name": str,
    "prompt": str,  # Optional
    "messages": List[Dict],  # Optional
    "temperature": float,  # Optional
    "max_tokens": int,  # Optional
}
```

**Usage**:
```python
self.emit(EventType.GENERATION_STARTED, {
    "model_name": "gpt-4",
    "provider_name": "OpenAI",
    "prompt": "Hello, world!"
})
```

#### GENERATION_COMPLETED
**When**: After LLM generation completes (includes all metrics)

**Data**:
```python
{
    "model_name": str,
    "provider_name": str,
    "tokens_input": int,
    "tokens_output": int,
    "duration_ms": float,
    "cost_usd": float,  # Optional
    "response": str,  # Optional
    "finish_reason": str,  # Optional
}
```

**Usage**:
```python
self.emit(EventType.GENERATION_COMPLETED, {
    "model_name": "gpt-4",
    "provider_name": "OpenAI",
    "tokens_input": 100,
    "tokens_output": 50,
    "duration_ms": 1250.5,
    "cost_usd": 0.003
})
```

#### TOOL_STARTED
**When**: Before tool execution begins

**Data**:
```python
{
    "tool_name": str,
    "arguments": Dict[str, Any],
    "tool_id": str,  # Optional
}
```

**Usage**:
```python
self.emit(EventType.TOOL_STARTED, {
    "tool_name": "search_web",
    "arguments": {"query": "Python async", "limit": 5}
})
```

#### TOOL_COMPLETED
**When**: After tool execution completes

**Data**:
```python
{
    "tool_name": str,
    "arguments": Dict[str, Any],
    "result": str,  # Truncated to 500 chars
    "success": bool,
    "error": Optional[str],
    "duration_ms": float,  # Optional
}
```

**Usage**:
```python
self.emit(EventType.TOOL_COMPLETED, {
    "tool_name": "search_web",
    "arguments": {"query": "Python async"},
    "result": "Found 5 results...",
    "success": True,
    "duration_ms": 342.1
})
```

### Error Handling (1)

#### ERROR
**When**: Any error occurs in the system

**Data**:
```python
{
    "error": str,  # Error message
    "error_type": str,  # Exception class name
    "context": Dict[str, Any],  # Additional context
}
```

**Usage**:
```python
self.emit_error(
    error=exception,
    context={"operation": "generate", "model": "gpt-4"}
)
```

### Retry and Resilience (2)

#### RETRY_ATTEMPTED
**When**: Retry process starts (after failure)

**Data**:
```python
{
    "attempt": int,  # Current attempt number (1-indexed)
    "max_attempts": int,
    "reason": str,  # Why retry is needed
    "delay_ms": float,  # Optional backoff delay
}
```

**Usage**:
```python
self.emit(EventType.RETRY_ATTEMPTED, {
    "attempt": 2,
    "max_attempts": 3,
    "reason": "RateLimitError",
    "delay_ms": 2000
})
```

#### RETRY_EXHAUSTED
**When**: All retry attempts fail (critical for alerting)

**Data**:
```python
{
    "total_attempts": int,
    "final_error": str,
    "operation": str,
}
```

**Usage**:
```python
self.emit(EventType.RETRY_EXHAUSTED, {
    "total_attempts": 3,
    "final_error": "RateLimitError: Too many requests",
    "operation": "generate"
})
```

### Specialized Events (5)

#### VALIDATION_FAILED
**When**: Structured output validation fails (triggers retry)

**Data**:
```python
{
    "response_model": str,  # Model class name
    "validation_error": str,
    "validation_attempt": int,
}
```

#### SESSION_CREATED
**When**: New conversation session is created

**Data**:
```python
{
    "session_id": str,
    "initial_context": Dict,  # Optional
}
```

#### SESSION_CLEARED
**When**: Session history is cleared

**Data**:
```python
{
    "session_id": str,
    "messages_cleared": int,
}
```

#### COMPACTION_STARTED
**When**: Chat history compaction begins

**Data**:
```python
{
    "original_messages": int,
    "target_messages": int,
    "compaction_strategy": str,
}
```

#### COMPACTION_COMPLETED
**When**: Compaction finishes

**Data**:
```python
{
    "original_messages": int,
    "compacted_messages": int,
    "tokens_saved": int,
}
```

## Core Classes

### Event (Dataclass)

**Purpose**: Simple, immutable event structure.

**Fields**:
```python
@dataclass
class Event:
    type: EventType          # Event type enum
    timestamp: datetime      # When event occurred
    data: Dict[str, Any]     # Event-specific data
    source: Optional[str]    # Source component name
```

**Example**:
```python
event = Event(
    type=EventType.GENERATION_STARTED,
    timestamp=datetime.now(),
    data={"model_name": "gpt-4", "provider_name": "OpenAI"},
    source="OpenAIProvider"
)
```

### EventEmitter (Mixin)

**Purpose**: Add event emission capabilities to any class.

**Methods**:

#### on(event_type, handler)
Register an event handler for this instance.

```python
def my_handler(event: Event):
    print(f"Event: {event.type.value}")

llm.on(EventType.GENERATION_STARTED, my_handler)
```

#### off(event_type, handler)
Unregister an event handler.

```python
llm.off(EventType.GENERATION_STARTED, my_handler)
```

#### emit(event_type, data, source=None, **kwargs)
Emit an event to all registered handlers.

```python
self.emit(
    EventType.GENERATION_COMPLETED,
    data={
        "model_name": "gpt-4",
        "tokens_input": 100,
        "tokens_output": 50
    }
)
```

#### emit_error(error, context=None)
Convenience method for emitting errors.

```python
try:
    result = risky_operation()
except Exception as e:
    self.emit_error(e, context={"operation": "risky_operation"})
```

**Usage Pattern**:
```python
from abstractcore.events import EventEmitter, EventType, Event

class MyComponent(EventEmitter):
    def __init__(self):
        super().__init__()

    def do_work(self):
        self.emit(EventType.GENERATION_STARTED, {
            "model_name": "gpt-4",
            "provider_name": "OpenAI"
        })

        # Do work...

        self.emit(EventType.GENERATION_COMPLETED, {
            "model_name": "gpt-4",
            "tokens_output": 50
        })
```

### GlobalEventBus (Singleton)

**Purpose**: System-wide event bus for monitoring across all components.

**Methods**:

#### GlobalEventBus.on(event_type, handler)
Register a global event handler.

```python
def global_handler(event: Event):
    print(f"Global event: {event.type.value}")

GlobalEventBus.on(EventType.GENERATION_COMPLETED, global_handler)
```

#### GlobalEventBus.off(event_type, handler)
Unregister a global event handler.

#### GlobalEventBus.emit(event_type, data, source=None, **kwargs)
Emit a global event.

#### GlobalEventBus.clear()
Clear all global event handlers (useful for testing).

**Convenience Functions**:
```python
from abstractcore.events import on_global, emit_global

# Register global handler
on_global(EventType.ERROR, error_handler)

# Emit global event
emit_global(EventType.GENERATION_STARTED, {
    "model_name": "gpt-4"
})
```

### EventLogger

**Purpose**: Basic event logger for debugging and monitoring.

**Usage**:
```python
from abstractcore.events import EventLogger, on_global, EventType

logger = EventLogger(log_level="INFO")

# Log all events
on_global(EventType.GENERATION_STARTED, logger.log_event)
on_global(EventType.GENERATION_COMPLETED, logger.log_event)
on_global(EventType.ERROR, logger.log_event)
```

**Output**:
```
[2025-10-25 14:30:15] generation_started from OpenAIProvider: {'model_name': 'gpt-4', ...}
[2025-10-25 14:30:17] generation_completed from OpenAIProvider: {'tokens_output': 50, ...}
```

### PerformanceTracker

**Purpose**: Aggregate performance metrics from events.

**Tracked Metrics**:
- Total requests
- Total latency (ms)
- Total input/output tokens
- Total cost (USD)
- Tool calls count
- Errors count

**Usage**:
```python
from abstractcore.events import PerformanceTracker, on_global, EventType

tracker = PerformanceTracker()

# Register trackers
on_global(EventType.GENERATION_COMPLETED, tracker.track_generation)
on_global(EventType.TOOL_COMPLETED, tracker.track_tool_call)
on_global(EventType.ERROR, tracker.track_error)

# Later, get metrics
metrics = tracker.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Total cost: ${metrics['total_cost_usd']:.4f}")
print(f"Avg latency: {metrics['total_latency_ms'] / metrics['total_requests']:.2f}ms")
```

## Helper Functions

### create_generation_event()

**Purpose**: Create standardized generation event data.

**Signature**:
```python
def create_generation_event(
    model_name: str,
    provider_name: str,
    tokens_input: int = None,
    tokens_output: int = None,
    duration_ms: float = None,
    cost_usd: float = None,
    **data
) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

**Returns**: Tuple of (event_data, kwargs) for emission.

**Usage**:
```python
event_data, kwargs = create_generation_event(
    model_name="gpt-4",
    provider_name="OpenAI",
    tokens_input=100,
    tokens_output=50,
    duration_ms=1250.5,
    prompt="Hello"
)

self.emit(EventType.GENERATION_COMPLETED, event_data, **kwargs)
```

### create_tool_event()

**Purpose**: Create standardized tool event data.

**Signature**:
```python
def create_tool_event(
    tool_name: str,
    arguments: Dict[str, Any],
    result: Any = None,
    success: bool = True,
    error: str = None,
    **data
) -> Dict[str, Any]
```

**Usage**:
```python
event_data = create_tool_event(
    tool_name="search_web",
    arguments={"query": "Python"},
    result="Found 5 results",
    success=True
)

self.emit(EventType.TOOL_COMPLETED, event_data)
```

### create_structured_output_event()

**Purpose**: Create standardized structured output event data.

**Signature**:
```python
def create_structured_output_event(
    response_model: str,
    validation_attempt: int = None,
    validation_error: str = None,
    retry_count: int = None,
    **data
) -> Dict[str, Any]
```

**Usage**:
```python
event_data = create_structured_output_event(
    response_model="PersonInfo",
    validation_attempt=2,
    validation_error="Field 'age' must be positive"
)

self.emit(EventType.VALIDATION_FAILED, event_data)
```

## Usage Patterns

### Pattern 1: Instance-Level Events

Monitor events from a specific LLM instance:

```python
from abstractcore import create_llm
from abstractcore.events import EventType, Event

llm = create_llm("openai", model="gpt-4")

def on_generation_complete(event: Event):
    print(f"Generated {event.data['tokens_output']} tokens")

llm.on(EventType.GENERATION_COMPLETED, on_generation_complete)

response = llm.generate("Hello!")  # Triggers event
```

### Pattern 2: Global Event Monitoring

Monitor all events across the entire system:

```python
from abstractcore.events import on_global, EventType, Event

def log_all_generations(event: Event):
    print(f"Generation: {event.data['model_name']}")

on_global(EventType.GENERATION_STARTED, log_all_generations)
on_global(EventType.GENERATION_COMPLETED, log_all_generations)

# Now all LLM generations are logged
llm1 = create_llm("openai", model="gpt-4")
llm2 = create_llm("anthropic", model="claude-3-5-sonnet-20241022")

llm1.generate("Hello")  # Logged
llm2.generate("World")  # Logged
```

### Pattern 3: Performance Monitoring

Track performance metrics across all operations:

```python
from abstractcore.events import PerformanceTracker, on_global, EventType

tracker = PerformanceTracker()
on_global(EventType.GENERATION_COMPLETED, tracker.track_generation)
on_global(EventType.TOOL_COMPLETED, tracker.track_tool_call)
on_global(EventType.ERROR, tracker.track_error)

# Run operations...

# Get metrics
metrics = tracker.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Total tokens: {metrics['total_tokens_input'] + metrics['total_tokens_output']}")
print(f"Total cost: ${metrics['total_cost_usd']:.4f}")
print(f"Error rate: {metrics['errors'] / metrics['total_requests']:.2%}")
```

### Pattern 4: Custom Event Handlers

Build custom monitoring, logging, or alerting:

```python
from abstractcore.events import Event, EventType, on_global
import logging

class CustomMonitor:
    def __init__(self):
        self.logger = logging.getLogger("abstractcore")
        self.error_count = 0
        self.max_errors = 10

    def handle_error(self, event: Event):
        self.error_count += 1
        self.logger.error(f"Error: {event.data['error']}")

        if self.error_count >= self.max_errors:
            self.logger.critical("Too many errors! Shutting down...")
            # Trigger alert, shutdown, etc.

    def handle_retry_exhausted(self, event: Event):
        self.logger.critical(
            f"Retry exhausted after {event.data['total_attempts']} attempts"
        )
        # Send alert to monitoring system

monitor = CustomMonitor()
on_global(EventType.ERROR, monitor.handle_error)
on_global(EventType.RETRY_EXHAUSTED, monitor.handle_retry_exhausted)
```

### Pattern 5: OpenTelemetry Integration

Integrate with OpenTelemetry for distributed tracing:

```python
from abstractcore.events import Event, EventType, on_global
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("abstractcore")

def trace_generation(event: Event):
    """Export AbstractCore events to OpenTelemetry spans"""
    if event.type == EventType.GENERATION_STARTED:
        span = tracer.start_span(
            "llm.generate",
            attributes={
                "llm.model": event.data["model_name"],
                "llm.provider": event.data["provider_name"],
            }
        )
        # Store span in context for GENERATION_COMPLETED

    elif event.type == EventType.GENERATION_COMPLETED:
        # Retrieve span from context
        span.set_attribute("llm.tokens.input", event.data["tokens_input"])
        span.set_attribute("llm.tokens.output", event.data["tokens_output"])
        span.set_attribute("llm.duration_ms", event.data["duration_ms"])
        span.end()

on_global(EventType.GENERATION_STARTED, trace_generation)
on_global(EventType.GENERATION_COMPLETED, trace_generation)
```

## Integration Points

### Providers
All providers emit events:
- **GENERATION_STARTED**: Before calling provider API
- **GENERATION_COMPLETED**: After receiving response
- **ERROR**: On API errors, network issues, etc.

### Tools
Tool handlers emit events:
- **TOOL_STARTED**: Before tool execution
- **TOOL_COMPLETED**: After tool execution (success or failure)
- **ERROR**: On tool execution errors

### Structured Output
Structured output handlers emit events:
- **VALIDATION_FAILED**: When Pydantic validation fails
- **RETRY_ATTEMPTED**: Before retry attempt
- **RETRY_EXHAUSTED**: When all retries fail

### Sessions
Session managers emit events:
- **SESSION_CREATED**: New session created
- **SESSION_CLEARED**: History cleared
- **COMPACTION_STARTED/COMPLETED**: History compaction

## Best Practices

### DO:
✅ **Use instance events for component-specific monitoring**
```python
llm.on(EventType.GENERATION_COMPLETED, my_handler)
```

✅ **Use global events for system-wide monitoring**
```python
on_global(EventType.ERROR, error_handler)
```

✅ **Handle exceptions in event handlers** (already done by EventEmitter)
```python
def safe_handler(event: Event):
    try:
        process_event(event)
    except Exception as e:
        logger.error(f"Handler error: {e}")
```

✅ **Use PerformanceTracker for metrics aggregation**
```python
tracker = PerformanceTracker()
on_global(EventType.GENERATION_COMPLETED, tracker.track_generation)
```

✅ **Clear global handlers in tests**
```python
def teardown():
    GlobalEventBus.clear()
```

✅ **Use helper functions for standardized event data**
```python
event_data = create_tool_event(tool_name, arguments, result)
```

### DON'T:
❌ **Don't emit events in event handlers** (can cause infinite loops)
```python
def bad_handler(event: Event):
    # BAD: This will trigger more events!
    emit_global(EventType.ERROR, {"msg": "handling"})
```

❌ **Don't perform heavy operations in handlers** (blocks event emission)
```python
def slow_handler(event: Event):
    # BAD: This will slow down all operations
    time.sleep(10)
    expensive_db_query()
```

❌ **Don't mutate event data** (events should be immutable)
```python
def bad_handler(event: Event):
    # BAD: Don't mutate
    event.data["modified"] = True
```

❌ **Don't forget to unregister handlers** (can cause memory leaks)
```python
# BAD: Handler never removed
llm.on(EventType.GENERATION_COMPLETED, temp_handler)
# GOOD: Remove when done
llm.off(EventType.GENERATION_COMPLETED, temp_handler)
```

## Common Pitfalls

### Pitfall 1: Event Handler Infinite Loops
```python
# BAD: Emits event in handler for same event type
def bad_handler(event: Event):
    emit_global(EventType.GENERATION_COMPLETED, {})  # INFINITE LOOP!

on_global(EventType.GENERATION_COMPLETED, bad_handler)

# GOOD: Only process, don't emit same event type
def good_handler(event: Event):
    process_metrics(event.data)  # No emission
```

### Pitfall 2: Heavy Operations in Handlers
```python
# BAD: Slow handler blocks everything
def slow_handler(event: Event):
    result = expensive_api_call()  # BLOCKS!
    save_to_db(result)

# GOOD: Offload to background thread
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)

def fast_handler(event: Event):
    executor.submit(process_async, event.data)
```

### Pitfall 3: Memory Leaks from Unremoved Handlers
```python
# BAD: Creates new handler on every call
def create_llm_with_logging():
    llm = create_llm("openai", model="gpt-4")
    llm.on(EventType.GENERATION_COMPLETED, lambda e: print(e))
    return llm  # Handler never removed!

# GOOD: Manage handler lifecycle
def create_llm_with_logging():
    llm = create_llm("openai", model="gpt-4")
    handler = lambda e: print(e)
    llm.on(EventType.GENERATION_COMPLETED, handler)

    # Store handler reference for cleanup
    llm._logging_handler = handler
    return llm

def cleanup_llm(llm):
    if hasattr(llm, '_logging_handler'):
        llm.off(EventType.GENERATION_COMPLETED, llm._logging_handler)
```

## Testing Strategy

### Testing Event Emission

```python
import pytest
from abstractcore.events import EventEmitter, EventType, Event

def test_event_emission():
    """Test that events are emitted correctly."""
    emitter = EventEmitter()
    events_received = []

    def handler(event: Event):
        events_received.append(event)

    emitter.on(EventType.GENERATION_STARTED, handler)

    emitter.emit(EventType.GENERATION_STARTED, {
        "model_name": "gpt-4"
    })

    assert len(events_received) == 1
    assert events_received[0].type == EventType.GENERATION_STARTED
    assert events_received[0].data["model_name"] == "gpt-4"

def test_handler_removal():
    """Test that handlers can be removed."""
    emitter = EventEmitter()
    events_received = []

    def handler(event: Event):
        events_received.append(event)

    emitter.on(EventType.GENERATION_STARTED, handler)
    emitter.emit(EventType.GENERATION_STARTED, {})

    emitter.off(EventType.GENERATION_STARTED, handler)
    emitter.emit(EventType.GENERATION_STARTED, {})

    # Only first emission should be received
    assert len(events_received) == 1
```

### Testing Global Events

```python
from abstractcore.events import GlobalEventBus, on_global, EventType

def test_global_events():
    """Test global event bus."""
    events_received = []

    def handler(event: Event):
        events_received.append(event)

    try:
        on_global(EventType.ERROR, handler)

        GlobalEventBus.emit(EventType.ERROR, {
            "error": "Test error",
            "error_type": "TestError"
        })

        assert len(events_received) == 1
        assert events_received[0].data["error"] == "Test error"

    finally:
        # Always clean up global handlers in tests
        GlobalEventBus.clear()
```

### Testing Performance Tracker

```python
from abstractcore.events import PerformanceTracker, Event, EventType

def test_performance_tracker():
    """Test metrics aggregation."""
    tracker = PerformanceTracker()

    # Simulate generation event
    event = Event(
        type=EventType.GENERATION_COMPLETED,
        timestamp=datetime.now(),
        data={
            "tokens_input": 100,
            "tokens_output": 50,
            "duration_ms": 1250.5,
            "cost_usd": 0.003
        },
        source="test"
    )

    tracker.track_generation(event)

    metrics = tracker.get_metrics()
    assert metrics["total_requests"] == 1
    assert metrics["total_tokens_input"] == 100
    assert metrics["total_tokens_output"] == 50
    assert metrics["total_latency_ms"] == 1250.5
    assert metrics["total_cost_usd"] == 0.003
```

## Public API

All event system components are exported:

```python
from abstractcore.events import (
    # Event types
    EventType,

    # Core classes
    Event,
    EventEmitter,
    GlobalEventBus,
    EventLogger,
    PerformanceTracker,

    # Convenience functions
    on_global,
    emit_global,

    # Helper functions
    create_generation_event,
    create_tool_event,
    create_structured_output_event,
)
```

## Summary

The events module provides a minimal, efficient event system for observability throughout AbstractCore. With only 12 event types and zero external dependencies, it enables comprehensive monitoring without adding complexity.

**Key Features**:
- **Instance-level events**: Monitor specific components
- **Global event bus**: System-wide monitoring
- **OpenTelemetry compatible**: Seamless integration with observability platforms
- **Performance tracking**: Built-in metrics aggregation
- **Error handling**: Robust error tracking and alerting
- **Zero dependencies**: Pure Python standard library

**Key Takeaways**:
- Use `EventEmitter` mixin for components that emit events
- Use `GlobalEventBus` for system-wide monitoring
- Use `PerformanceTracker` for metrics aggregation
- Always clean up global handlers in tests with `GlobalEventBus.clear()`
- Keep event handlers fast and non-blocking
- Use helper functions for standardized event data

## Related Modules

**Used by (emits events)**:
- [`providers/`](../providers/README.md) - Generation, streaming, and tool execution events
- [`media/`](../media/README.md) - Media processing events
- [`compression/`](../compression/README.md) - Compression operation events
- [`structured/`](../structured/README.md) - Validation and retry events
- [`tools/`](../tools/README.md) - Tool execution lifecycle events

**Related infrastructure**:
- [`exceptions/`](../exceptions/README.md) - Error events use exception types
- [`utils/`](../utils/README.md) - Event logging integration
- [`config/`](../config/README.md) - Event system configuration
