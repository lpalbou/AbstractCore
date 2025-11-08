# Feature Request: In-Memory Interaction Tracing for LLM Observability

## Problem Statement

Currently, AbstractCore provides excellent logging capabilities (`VerbatimCapture`, `StructuredLogger`) that write LLM interactions to log files. However, there's no way to **programmatically access** the complete trace of LLM interactions for observability purposes in applications.

### Use Case: Digital Article

Digital Article is a computational notebook application where users describe analyses in natural language, and the system:
1. **Generates code** from the prompt (LLM call #1)
2. Executes the code
3. If errors occur, **retries up to 3 times** with error context (LLM calls #2-4)
4. **Generates scientific methodology** text (LLM call #5)

**Users need complete observability** of what the LLM did at each step:
- What prompts were sent (system + user)
- What the LLM returned (raw response, thinking traces if available, tool calls)
- What was extracted/parsed from the response
- Token usage and timing for each step

This is critical for:
- **Debugging**: Understanding why code generation failed or succeeded
- **Trust**: Seeing the LLM's reasoning process
- **Optimization**: Identifying inefficient prompts or patterns
- **Compliance**: Audit trails for AI-generated code

## Proposed Solution

### 1. New `InteractionTrace` Data Structure

```python
@dataclass
class InteractionTrace:
    """Complete trace of a single LLM interaction."""

    # Metadata
    trace_id: str  # Unique identifier
    timestamp: datetime
    step_type: str  # e.g., "code_generation", "retry_1", "methodology"
    attempt_number: int  # 1-based counter

    # Input
    system_prompt: Optional[str]
    user_prompt: str
    messages_context: Optional[List[Dict[str, str]]]  # For conversation history
    parameters: Dict[str, Any]  # temperature, max_tokens, seed, etc.

    # Output
    raw_response: Any  # Full raw response object from provider
    content: Optional[str]  # Extracted content
    thinking: Optional[str]  # Thinking traces (if supported by model)
    tool_calls: Optional[List[Dict[str, Any]]]
    finish_reason: Optional[str]

    # Metrics
    usage: Optional[Dict[str, int]]  # Token counts
    generation_time_ms: Optional[float]

    # Processing
    extracted_output: Optional[str]  # What was parsed/extracted (e.g., code block)
    parsing_metadata: Optional[Dict[str, Any]]  # How extraction was done

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractionTrace':
        """Deserialize from dictionary."""
        ...
```

### 2. Enhance `GenerateResponse` with Tracing

```python
@dataclass
class GenerateResponse:
    """Response from LLM generation"""
    content: Optional[str] = None
    raw_response: Any = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    gen_time: Optional[float] = None

    # NEW: Interaction trace
    interaction_trace: Optional[InteractionTrace] = None  # Full trace if enabled
```

### 3. Add Tracing to `AbstractCoreInterface`

```python
class AbstractCoreInterface:
    """Base interface for LLM providers with tracing support."""

    def __init__(self, ..., enable_tracing: bool = False, trace_storage: Optional['TraceStorage'] = None):
        ...
        self.enable_tracing = enable_tracing
        self.trace_storage = trace_storage or InMemoryTraceStorage()

    def generate(self, prompt: str, system_prompt: Optional[str] = None, ...) -> GenerateResponse:
        """Generate with optional tracing."""

        # Create trace if enabled
        trace = None
        if self.enable_tracing:
            trace = InteractionTrace(
                trace_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                step_type="generation",  # Can be customized via metadata
                attempt_number=1,
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_context=messages,
                parameters={
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'seed': seed,
                    ...
                }
            )

        # Call provider
        response = self._provider_generate(...)

        # Complete trace
        if trace:
            trace.raw_response = response.raw_response
            trace.content = response.content
            trace.thinking = response.metadata.get('thinking') if response.metadata else None
            trace.tool_calls = response.tool_calls
            trace.finish_reason = response.finish_reason
            trace.usage = response.usage
            trace.generation_time_ms = response.gen_time

            # Store trace
            self.trace_storage.add_trace(trace)

            # Attach to response
            response.interaction_trace = trace

        return response
```

### 4. Trace Storage Interface

```python
class TraceStorage(ABC):
    """Abstract interface for trace storage."""

    @abstractmethod
    def add_trace(self, trace: InteractionTrace) -> None:
        """Store a trace."""
        pass

    @abstractmethod
    def get_traces(self, trace_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   step_type: Optional[str] = None) -> List[InteractionTrace]:
        """Retrieve traces by filters."""
        pass

    @abstractmethod
    def clear_traces(self) -> None:
        """Clear all stored traces."""
        pass


class InMemoryTraceStorage(TraceStorage):
    """In-memory trace storage (default)."""

    def __init__(self, max_traces: int = 1000):
        self.traces: List[InteractionTrace] = []
        self.max_traces = max_traces

    def add_trace(self, trace: InteractionTrace) -> None:
        self.traces.append(trace)
        # Keep only last N traces
        if len(self.traces) > self.max_traces:
            self.traces = self.traces[-self.max_traces:]

    def get_traces(self, ...) -> List[InteractionTrace]:
        # Filter implementation
        ...


class FileTraceStorage(TraceStorage):
    """File-based trace storage (JSONL)."""

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def add_trace(self, trace: InteractionTrace) -> None:
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(trace.to_dict()) + '\n')

    ...
```

### 5. Integration with `BasicSession`

```python
class BasicSession:
    """Session with interaction tracing."""

    def __init__(self, ..., enable_tracing: bool = False):
        ...
        self.enable_tracing = enable_tracing
        self.traces: List[InteractionTrace] = []  # Session-specific traces

    def generate(self, prompt: str, ...) -> GenerateResponse:
        """Generate with automatic trace capture."""

        # Set metadata for this interaction
        if self.enable_tracing and self.provider.enable_tracing:
            # You can tag the step type via metadata
            kwargs['trace_metadata'] = {
                'step_type': kwargs.get('step_type', 'chat'),
                'attempt_number': kwargs.get('attempt_number', 1),
                'session_id': self.id
            }

        response = self.provider.generate(prompt, ...)

        # Collect trace in session
        if response.interaction_trace:
            self.traces.append(response.interaction_trace)

        return response

    def get_interaction_history(self) -> List[InteractionTrace]:
        """Get all interaction traces for this session."""
        return self.traces.copy()

    def export_traces(self, file_path: Path, format: str = 'jsonl'):
        """Export traces to file."""
        ...
```

## Usage Example

```python
from abstractcore import create_llm
from abstractcore.core.tracing import InMemoryTraceStorage

# Create LLM with tracing enabled
llm = create_llm('lmstudio', model='qwen/qwen3-next-80b',
                 enable_tracing=True)

# Generate code
response = llm.generate(
    "Create a histogram of ages",
    system_prompt="You are a Python code generator...",
    trace_metadata={'step_type': 'code_generation', 'attempt_number': 1}
)

# Access the trace
trace = response.interaction_trace

print(f"System Prompt: {trace.system_prompt}")
print(f"User Prompt: {trace.user_prompt}")
print(f"Raw Response: {trace.raw_response}")
print(f"Extracted Content: {trace.content}")
print(f"Tokens: {trace.usage}")
print(f"Time: {trace.generation_time_ms}ms")

# Get all traces
all_traces = llm.trace_storage.get_traces()

# Export traces for analysis
llm.trace_storage.export_to_file('traces.jsonl')
```

## Benefits

1. **Zero Breaking Changes**: Tracing is opt-in via `enable_tracing=True`
2. **Flexible Storage**: In-memory (default) or file-based
3. **Complete Transparency**: Every LLM interaction is fully traceable
4. **Framework Agnostic**: Works with all providers
5. **Debugging Power**: Full visibility into prompts, responses, parsing
6. **Compliance Ready**: Audit trail for AI-generated content

## Implementation Priority

**High Priority**: This feature is critical for production applications that need:
- Debugging capabilities for LLM-generated code/content
- User trust through transparency
- Compliance and audit trails
- Performance optimization through trace analysis

## Additional Considerations

### Performance
- In-memory storage with configurable max size (default: 1000 traces)
- Optional file-based storage for long-running sessions
- Traces can be disabled per-call if needed

### Privacy
- Allow filtering of sensitive data from traces
- Optional trace encryption for file storage
- Configurable trace retention policies

### API Design
- Keep consistent with existing AbstractCore patterns
- Follow dataclass pattern for serialization
- Support both dict and object access patterns

## Questions for AbstractCore Maintainers

1. Would you prefer `enable_tracing` as a global setting or per-call parameter?
2. Should traces be attached to `GenerateResponse` or retrieved separately?
3. Any concerns about memory usage for long-running sessions?
4. Should we include prompt/response compression for large traces?

---

**Submitted by**: Digital Article Team
**Date**: 2025-11-08
**Contact**: (your contact info)
