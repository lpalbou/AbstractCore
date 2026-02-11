# Interaction Tracing for LLM Observability

**Interaction tracing** provides programmatic access to complete LLM interaction history for debugging, observability, and compliance purposes.

## Overview

AbstractCore's interaction tracing captures the complete context of every LLM interaction:
- **Input**: Prompts, system prompts, messages, parameters
- **Output**: Content, tool calls, usage metrics, timing
- **Metadata**: Custom tags, session info, step types

This is essential for:
- **Debugging**: Understanding why generation succeeded or failed
- **Trust**: Seeing the LLM's complete reasoning chain
- **Optimization**: Identifying inefficient prompts or patterns
- **Compliance**: Audit trails for AI-generated content

## Quick Start

### Provider-Level Tracing

```python
from abstractcore import create_llm

# Enable tracing on provider
llm = create_llm(
    'ollama',
    model='qwen3:4b',
    enable_tracing=True,
    max_traces=100  # Ring buffer size (default: 100)
)

# Generate with custom metadata
response = llm.generate(
    "Write a Python function to add two numbers",
    temperature=0,
    trace_metadata={
        'step': 'code_generation',
        'attempt': 1,
        'user_id': 'user_123'
    }
)

# Access trace ID from response
trace_id = response.metadata['trace_id']

# Retrieve specific trace
trace = llm.get_traces(trace_id=trace_id)

print(f"Prompt: {trace['prompt']}")
print(f"Response: {trace['response']['content']}")
print(f"Tokens: {trace['response']['usage']}")
print(f"Time: {trace['response']['generation_time_ms']}ms")
print(f"Custom metadata: {trace['metadata']}")
```

### Session-Level Tracing

```python
from abstractcore import create_llm
from abstractcore.core.session import BasicSession

# Create provider with tracing
llm = create_llm('ollama', model='qwen3:4b', enable_tracing=True)

# Create session with tracing
session = BasicSession(provider=llm, enable_tracing=True)

# Normal conversation
response1 = session.generate("What is Python?")
response2 = session.generate("Give me an example")

# Get all interaction traces for this session
traces = session.get_interaction_history()

print(f"Captured {len(traces)} interactions")
for i, trace in enumerate(traces, 1):
    print(f"\nInteraction {i}:")
    print(f"  Session ID: {trace['metadata']['session_id']}")
    print(f"  Prompt: {trace['prompt']}")
    print(f"  Tokens: {trace['response']['usage']['total_tokens']}")
```

## Trace Structure

Each trace contains:

```python
{
    'trace_id': 'uuid-string',
    'timestamp': '2025-11-08T12:34:56.789',
    'provider': 'OllamaProvider',
    'model': 'qwen3:4b',

    # Input
    'system_prompt': 'You are a helpful assistant',
    'prompt': 'What is Python?',
    'messages': [...],  # Conversation history
    'tools': [...],     # Available tools

    # Parameters
    'parameters': {
        'temperature': 0.7,
        'max_tokens': 8000,
        'max_output_tokens': 2048,
        'seed': 42,
        'top_p': 0.9,
        'top_k': 50
    },

    # Output
    'response': {
        'content': 'Python is...',
        'raw_response': {...},  # If verbatim=True
        'tool_calls': [...],
        'finish_reason': 'stop',
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 50,
            'total_tokens': 60
        },
        'generation_time_ms': 1234.56
    },

    # Custom metadata
    'metadata': {
        'session_id': 'uuid',
        'step_type': 'chat',
        'user_id': 'user_123',
        # ... any custom fields
    }
}
```

## Retrieving Traces

### Get All Traces

```python
# Get all traces from provider
all_traces = llm.get_traces()

for trace in all_traces:
    print(f"{trace['timestamp']}: {trace['prompt'][:50]}...")
```

### Get Specific Trace by ID

```python
# Generate and get trace ID
response = llm.generate("Test", trace_metadata={'step': 'test'})
trace_id = response.metadata['trace_id']

# Retrieve specific trace
trace = llm.get_traces(trace_id=trace_id)
```

### Get Last N Traces

```python
# Get most recent 10 traces
recent_traces = llm.get_traces(last_n=10)

for trace in recent_traces:
    print(f"Tokens: {trace['response']['usage']['total_tokens']}")
```

## Exporting Traces

AbstractCore provides utilities to export traces to various formats:

### JSONL (JSON Lines)

```python
from abstractcore.utils import export_traces

traces = llm.get_traces()
export_traces(traces, format='jsonl', file_path='traces.jsonl')
```

### JSON (Pretty-Printed)

```python
export_traces(traces, format='json', file_path='traces.json')
```

### Markdown Report

```python
# Generate human-readable markdown report
export_traces(traces, format='markdown', file_path='trace_report.md')
```

### Export as String

```python
# Get formatted string without writing to file
json_string = export_traces(traces, format='json')
print(json_string)
```

## Trace Summarization

Get summary statistics across multiple traces:

```python
from abstractcore.utils import summarize_traces

traces = session.get_interaction_history()
summary = summarize_traces(traces)

print(f"Total interactions: {summary['total_interactions']}")
print(f"Total tokens used: {summary['total_tokens']}")
print(f"Average tokens per interaction: {summary['avg_tokens_per_interaction']:.0f}")
print(f"Average generation time: {summary['avg_time_ms']:.2f}ms")
print(f"Providers used: {summary['providers']}")
print(f"Models used: {summary['models']}")
print(f"Date range: {summary['date_range']['first']} to {summary['date_range']['last']}")
```

## Use Case: Multi-Step Code Generation with Retries

Perfect for debugging workflows like Digital Article's computational notebook:

```python
from abstractcore import create_llm
from abstractcore.core.session import BasicSession

llm = create_llm('ollama', model='qwen3:4b', enable_tracing=True)
session = BasicSession(provider=llm, enable_tracing=True)

# Step 1: Generate code
response = session.generate(
    "Create a histogram of ages",
    system_prompt="You are a Python code generator",
    step_type='code_generation',
    attempt_number=1
)

code = response.content

# Step 2: Execute code (simulated)
try:
    exec(code)
    success = True
except Exception as e:
    success = False
    error = str(e)

    # Retry with error context
    for attempt in range(2, 4):
        response = session.generate(
            f"Previous code failed with error: {error}. Fix the code.",
            step_type='code_generation',
            attempt_number=attempt
        )
        code = response.content
        try:
            exec(code)
            success = True
            break
        except Exception as e:
            error = str(e)

# Step 3: Generate methodology text
if success:
    response = session.generate(
        "Generate scientific methodology text for the histogram analysis",
        step_type='methodology_generation'
    )

# Complete observability
traces = session.get_interaction_history()
print(f"\nWorkflow Summary:")
print(f"Total LLM calls: {len(traces)}")

for trace in traces:
    print(f"\nStep: {trace['metadata']['step_type']}")
    print(f"Attempt: {trace['metadata']['attempt_number']}")
    print(f"Tokens: {trace['response']['usage']['total_tokens']}")
    print(f"Time: {trace['response']['generation_time_ms']}ms")

# Export for analysis
from abstractcore.utils import export_traces
export_traces(traces, format='markdown', file_path='workflow_trace.md')
```

## Memory Management

### Ring Buffer

Traces are stored in a **ring buffer** (deque with max size) for memory efficiency:

```python
llm = create_llm(
    'ollama',
    model='qwen3:4b',
    enable_tracing=True,
    max_traces=50  # Keep only last 50 traces
)

# After 100 generations, only last 50 are kept
for i in range(100):
    llm.generate(f"Test {i}")

traces = llm.get_traces()
assert len(traces) == 50  # Oldest 50 were dropped
```

### Session Isolation

Each session maintains its own trace list:

```python
llm = create_llm('ollama', model='qwen3:4b', enable_tracing=True)

session1 = BasicSession(provider=llm, enable_tracing=True)
session2 = BasicSession(provider=llm, enable_tracing=True)

session1.generate("Question 1")
session2.generate("Question 2")

# Traces are isolated per session
assert len(session1.get_interaction_history()) == 1
assert len(session2.get_interaction_history()) == 1

# Provider still has both traces
assert len(llm.get_traces()) == 2
```

## Best Practices

### 1. Enable Tracing Only When Needed

```python
# Development/debugging
llm = create_llm('ollama', model='qwen3:4b', enable_tracing=True)

# Production (default - no overhead)
llm = create_llm('ollama', model='qwen3:4b')
```

### 2. Use Custom Metadata for Context

```python
response = llm.generate(
    prompt,
    trace_metadata={
        'user_id': user.id,
        'workflow': 'code_generation',
        'step': 'initial_generation',
        'attempt': 1,
        'environment': 'production'
    }
)
```

### 3. Export Regularly for Long Sessions

```python
# Export and clear every 100 interactions
if len(session.interaction_traces) >= 100:
    export_traces(
        session.get_interaction_history(),
        format='jsonl',
        file_path=f'traces_{datetime.now().isoformat()}.jsonl'
    )
    session.interaction_traces.clear()
```

### 4. Filter Raw Responses for Privacy

By default, `raw_response` is only included if `verbatim=True` on the provider. This prevents accidentally logging sensitive data.

## Performance Impact

- **When disabled (default)**: Zero overhead
- **When enabled**: Minimal overhead (~1-2% for typical workloads)
  - Trace capture: <1ms per interaction
  - Memory: ~1-5KB per trace (depends on response size)
  - Ring buffer: O(1) append, automatic eviction

## Comparison with Existing Logging

| Feature | VerbatimCapture | Event System | Interaction Tracing |
|---------|----------------|--------------|---------------------|
| **Access** | File-only | Event handlers | Programmatic (in-memory) |
| **Completeness** | Prompt + response | Metrics only | Full interaction context |
| **Retrieval** | Parse files | Listen to events | Direct API (get_traces()) |
| **Filtering** | N/A | By event type | By trace_id or last_n |
| **Export** | JSONL | N/A | JSONL/JSON/Markdown |
| **Use Case** | Audit logs | Real-time monitoring | Debugging, observability |

## Example: Debugging Failed Generation

```python
llm = create_llm('ollama', model='qwen3:4b', enable_tracing=True)

try:
    response = llm.generate(
        "Complex prompt...",
        temperature=0.7,
        max_output_tokens=2000
    )
except Exception as e:
    # Get trace even if generation failed
    traces = llm.get_traces(last_n=1)
    if traces:
        trace = traces[0]
        print("Failed generation details:")
        print(f"  Prompt: {trace['prompt']}")
        print(f"  Parameters: {trace['parameters']}")
        print(f"  Error: {e}")
```

## API Reference

### Provider Methods

- `llm.get_traces()` → List[Dict]: Get all traces
- `llm.get_traces(trace_id='...')` → Dict: Get specific trace
- `llm.get_traces(last_n=10)` → List[Dict]: Get last N traces

### Session Methods

- `session.get_interaction_history()` → List[Dict]: Get all session traces

### Utility Functions

- `export_traces(traces, format='jsonl|json|markdown', file_path=None)` → str
- `summarize_traces(traces)` → Dict: Get summary statistics

## Related Documentation

- [Structured Logging](./structured-logging.md) - File-based logging
- [Event System](./events.md) - Real-time event monitoring
- [Session Management](./session.md) - BasicSession usage

## FAQ

**Q: Does tracing work with streaming?**
A: Currently, tracing is only supported for non-streaming responses. Streaming support is planned for a future release.

**Q: Are traces thread-safe?**
A: Traces are stored per-provider-instance. If you share a provider across threads, use separate provider instances or add your own synchronization.

**Q: Can I disable raw_response in traces?**
A: Yes, raw_response is only included if the provider has `verbatim=True`. By default, it's `None` to save memory and avoid logging sensitive data.

**Q: What happens to traces when provider is garbage collected?**
A: Traces are stored in memory and will be lost when the provider is garbage collected. Export traces if you need persistence.
