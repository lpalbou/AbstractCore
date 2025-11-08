# Core Module

## Purpose

The **Core Module** is the foundational layer of AbstractCore, providing essential abstractions, types, and patterns that enable provider-agnostic LLM interactions. It establishes a unified interface for all LLM providers, standardizes token management, implements production-ready retry strategies, and manages conversation state. This module serves as the architectural bedrock upon which all provider implementations and high-level features are built.

## Architecture Position

**Layer**: Foundation / Abstraction Layer

**Dependencies**:
- Python standard library (abc, dataclasses, typing, datetime, enum)
- Internal: `abstractcore.exceptions`, `abstractcore.utils.token_utils`, `abstractcore.events`, `abstractcore.tools`, `abstractcore.processing`

**Used By**:
- All provider implementations (`providers/*_provider.py`)
- High-level processing modules (`processing/`)
- CLI and server components
- Application code using AbstractCore

**Role in System**:
- Defines contracts that all providers must implement
- Provides reusable patterns for retry logic, token management, and sessions
- Ensures consistency across the entire framework
- Acts as the single source of truth for core types and behaviors

---

## Component Structure

### File Overview (6 files)

| File | Purpose | Lines | Key Exports |
|------|---------|-------|-------------|
| `enums.py` | Enumeration types for roles, parameters, capabilities | 35 | `MessageRole`, `ModelParameter`, `ModelCapability` |
| `types.py` | Core data structures for messages and responses | 138 | `Message`, `GenerateResponse` |
| `interface.py` | Abstract base class defining provider contract | 357 | `AbstractCoreInterface` |
| `retry.py` | Production-ready retry and circuit breaker logic | 373 | `RetryManager`, `RetryConfig`, `CircuitBreaker` |
| `factory.py` | Factory function for creating provider instances | 73 | `create_llm()` |
| `session.py` | Conversation management and history tracking | 940 | `BasicSession` |

**Total**: ~1,916 lines of foundational code

---

## Core Components

### enums.py

Defines enumeration types for standardized communication across the framework.

#### MessageRole (4 values)

Represents the role of a message in a conversation:

```python
class MessageRole(Enum):
    SYSTEM = "system"      # System instructions and context
    USER = "user"          # User input
    ASSISTANT = "assistant"  # LLM responses
    TOOL = "tool"          # Tool execution results
```

**Usage**:
```python
from abstractcore.core.enums import MessageRole

# Creating messages with explicit roles
session.add_message(MessageRole.SYSTEM.value, "You are a helpful assistant.")
session.add_message(MessageRole.USER.value, "What is Python?")
```

#### ModelParameter (7 values)

Standard parameters supported across providers:

```python
class ModelParameter(Enum):
    MODEL = "model"                        # Model identifier
    TEMPERATURE = "temperature"            # Randomness (0.0-1.0)
    MAX_TOKENS = "max_tokens"              # Context window budget
    TOP_P = "top_p"                        # Nucleus sampling
    TOP_K = "top_k"                        # Top-K sampling
    FREQUENCY_PENALTY = "frequency_penalty"  # Repetition penalty
    PRESENCE_PENALTY = "presence_penalty"    # Topic diversity
    SEED = "seed"                          # Deterministic generation
```

**Purpose**: Normalize parameter names across different provider APIs.

#### ModelCapability (6 values)

Capabilities that models/providers can support:

```python
class ModelCapability(Enum):
    CHAT = "chat"              # Conversational interactions
    TOOLS = "tools"            # Function/tool calling
    VISION = "vision"          # Image understanding
    STREAMING = "streaming"    # Token-by-token streaming
    ASYNC = "async"            # Asynchronous operations
    JSON_MODE = "json_mode"    # JSON-formatted responses
```

**Usage**:
```python
# Check provider capabilities
capabilities = llm.get_capabilities()
if ModelCapability.VISION.value in capabilities:
    response = llm.generate(prompt="Describe this", media=["image.jpg"])
```

---

### types.py

Core data structures for messages and responses with comprehensive metadata support.

#### Message Dataclass

Represents a single message with metadata and serialization support.

```python
@dataclass
class Message:
    role: str                           # Message role (user, assistant, system, tool)
    content: str                        # Message content
    timestamp: Optional[datetime] = None  # Auto-generated if not provided
    metadata: Optional[Dict[str, Any]] = None  # Extensible metadata
```

**Key Features**:

1. **Automatic Timestamp**: Defaults to `datetime.now()` if not provided
2. **Metadata Properties**: Convenience accessors for common fields
   - `name`: Username (accessible via property)
   - `location`: Geographical/contextual location
3. **Serialization**: `to_dict()` and `from_dict()` for persistence
4. **Backward Compatibility**: Handles legacy format with separate `name` field

**Usage Examples**:

```python
from abstractcore.core.types import Message

# Basic message
msg = Message(role="user", content="Hello!")
# Auto-generated: msg.timestamp = datetime.now()

# Message with metadata
msg = Message(
    role="user",
    content="What's the weather?",
    metadata={"name": "alice", "location": "Paris"}
)

# Using convenience properties
msg.name = "alice"
msg.location = "Paris"
print(msg.name)  # "alice"

# Serialization
data = msg.to_dict()
restored = Message.from_dict(data)
```

#### GenerateResponse Dataclass

Comprehensive response object from LLM generation.

```python
@dataclass
class GenerateResponse:
    content: Optional[str] = None              # Generated text
    raw_response: Any = None                   # Provider-specific raw response
    model: Optional[str] = None                # Model used
    finish_reason: Optional[str] = None        # "stop", "length", "tool_calls", etc.
    usage: Optional[Dict[str, int]] = None     # Token usage statistics
    tool_calls: Optional[List[Dict]] = None    # Tool invocations
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    gen_time: Optional[float] = None           # Generation time (ms)
```

**Key Methods**:

```python
# Check for tool calls
has_tools = response.has_tool_calls()

# Get executed tool names
tools = response.get_tools_executed()  # ["search", "calculator"]

# Get formatted summary
summary = response.get_summary()
# "Model: gpt-4o | Tokens: 523 | Time: 1234.5ms | Tools: 2 executed"
```

**Token Usage Properties** (handles provider variations):

```python
# Unified properties (handles OpenAI vs Anthropic naming)
input_tokens = response.input_tokens    # prompt_tokens OR input_tokens
output_tokens = response.output_tokens  # completion_tokens OR output_tokens
total_tokens = response.total_tokens    # total_tokens
```

**Usage Example**:

```python
response = llm.generate("Explain quantum computing")

print(f"Generated: {response.content}")
print(f"Model: {response.model}")
print(f"Tokens: {response.total_tokens}")
print(f"Time: {response.gen_time}ms")

if response.has_tool_calls():
    print(f"Tools used: {response.get_tools_executed()}")
```

---

### interface.py

The abstract base class that all LLM providers must implement. Defines the unified contract and comprehensive token management.

#### AbstractCoreInterface

```python
class AbstractCoreInterface(ABC):
    """Abstract base class for all LLM providers with unified token management."""

    def __init__(self,
                 model: str,
                 max_tokens: Optional[int] = None,
                 max_input_tokens: Optional[int] = None,
                 max_output_tokens: int = 2048,
                 temperature: float = 0.7,
                 seed: Optional[int] = None,
                 debug: bool = False,
                 **kwargs):
        # ...
```

#### Token Management

**Unified Parameters**: `max_tokens` (total budget), `max_output_tokens` (generation limit), `max_input_tokens` (auto-calculated)
**Constraint**: `max_input_tokens + max_output_tokens ≤ max_tokens`
**Helper Methods**: `get_token_configuration_summary()`, `validate_token_constraints()`, `calculate_token_budget()`

→ See [utils/token_utils.py](../utils/README.md#token-counting) for token counting strategies

AbstractCore automatically maps to provider-specific APIs (OpenAI, Anthropic, Google, HuggingFace).

#### Abstract Methods (Provider Contract)

All providers must implement:

**1. generate() - Core Generation Method**

```python
@abstractmethod
def generate(self,
            prompt: str,
            messages: Optional[List[Dict[str, str]]] = None,
            system_prompt: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            media: Optional[List[Union[str, Dict, MediaContent]]] = None,
            stream: bool = False,
            **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
    """Generate response from the LLM."""
    pass
```

**Parameters**:
- `prompt`: User input (required)
- `messages`: Conversation history (optional)
- `system_prompt`: System instructions (optional)
- `tools`: Available tools for function calling (optional)
- `media`: Images/files for vision models (optional)
- `stream`: Enable streaming (optional)
- `**kwargs`: Provider-specific parameters

**2. get_capabilities() - Capability Discovery**

```python
@abstractmethod
def get_capabilities(self) -> List[str]:
    """Return list of supported capabilities."""
    pass

# Example implementation:
def get_capabilities(self) -> List[str]:
    return ["chat", "tools", "vision", "streaming"]
```

**3. validate_config() - Configuration Validation**

```python
def validate_config(self) -> bool:
    """Validate provider configuration (optional override)."""
    return True
```

---

### retry.py

Production-ready retry with exponential backoff (full jitter) and circuit breaker patterns (SOTA 2025).

#### RetryConfig

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3                   # Retry attempts
    initial_delay: float = 1.0              # Initial delay (seconds)
    max_delay: float = 60.0                 # Maximum delay cap
    exponential_base: float = 2.0           # Backoff base
    use_jitter: bool = True                 # AWS-recommended jitter
    failure_threshold: int = 5              # Circuit breaker threshold
    recovery_timeout: float = 60.0          # Recovery period
    half_open_max_calls: int = 3            # Test calls in half-open
```

**Delay**: `random(0, min(cap, base^attempt))` with full jitter

#### CircuitBreaker

**3-State Pattern** (Netflix Hystrix): CLOSED (normal) → OPEN (fail-fast) → HALF_OPEN (testing) → CLOSED
**Methods**: `can_execute()`, `record_success()`, `record_failure()`, `get_state_info()`

#### RetryManager

Central retry orchestration with smart error classification:

**Smart Error Classification**:

```python
class RetryableErrorType(Enum):
    RATE_LIMIT = "rate_limit"          # Always retry with backoff
    TIMEOUT = "timeout"                # Retry with backoff
    NETWORK = "network"                # Retry with backoff
    API_ERROR = "api_error"            # Retry once
    VALIDATION_ERROR = "validation_error"  # Retry up to max_attempts
    UNKNOWN = "unknown"                # No retry
```

**Retry Decision Logic**:

| Error Type | Max Retries | Strategy |
|------------|-------------|----------|
| `RATE_LIMIT` | `max_attempts` | Exponential backoff + jitter |
| `TIMEOUT` | `max_attempts` | Exponential backoff + jitter |
| `NETWORK` | `max_attempts` | Exponential backoff + jitter |
| `VALIDATION_ERROR` | `max_attempts` | Retry with feedback |
| `API_ERROR` | 1 | Single retry for transient issues |
| `UNKNOWN` | 0 | No retry (fail immediately) |

**Usage Example**:

```python
from abstractcore.core.retry import RetryManager, RetryConfig

# Configure retry behavior
config = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    failure_threshold=5,
    recovery_timeout=60.0
)

retry_manager = RetryManager(config)

# Execute with retry protection
def risky_api_call():
    return external_api.call()

result = retry_manager.execute_with_retry(
    risky_api_call,
    provider_key="openai:gpt-4o"  # Circuit breaker key
)

# Automatic handling of:
# 1. Exponential backoff with jitter
# 2. Circuit breaker protection
# 3. Smart error classification
# 4. Event emission for observability
```

**Event Emission** (Observability):

The RetryManager emits events for monitoring and alerting:

```python
# Events emitted:
# - RETRY_ATTEMPTED: Each retry with context
# - RETRY_EXHAUSTED: All retries failed (critical)

# Event data includes:
# {
#     "provider_key": "openai:gpt-4o",
#     "current_attempt": 2,
#     "max_attempts": 3,
#     "error_type": "rate_limit",
#     "delay_seconds": 2.34,
#     "circuit_breaker_state": {...}
# }
```

**Global Instance**:

```python
from abstractcore.core.retry import default_retry_manager

# Use the default global instance
result = default_retry_manager.execute_with_retry(func, provider_key="...")
```

---

### factory.py

Factory function for creating LLM provider instances with unified configuration.

#### create_llm()

The primary entry point for creating provider instances:

```python
def create_llm(provider: str,
               model: Optional[str] = None,
               **kwargs) -> AbstractCoreInterface:
    """
    Create an LLM provider instance with unified token parameter support.

    Args:
        provider: Provider name (openai, anthropic, ollama, huggingface, mlx, lmstudio, google)
        model: Model name (optional, will use provider default)
        **kwargs: Additional configuration (token parameters, temperature, etc.)

    Returns:
        Configured LLM provider instance

    Raises:
        ImportError: Provider dependencies not installed
        ValueError: Provider not supported
        ModelNotFoundError: Model not available
        AuthenticationError: Invalid API credentials
    """
```

**Basic Usage**:

```python
from abstractcore import create_llm

# Minimal setup
llm = create_llm("openai", "gpt-4o")

# With token configuration
llm = create_llm(
    provider="anthropic",
    model="claude-3.5-sonnet",
    max_tokens=8000,
    max_output_tokens=2000,
    temperature=0.7
)

# With custom parameters
llm = create_llm(
    provider="ollama",
    model="qwen3:4b",
    temperature=0.0,
    seed=42,
    top_p=0.9
)
```

**Provider Auto-Detection**:

The factory intelligently routes models to the correct provider:

```python
# MLX models → MLX provider
llm = create_llm("huggingface", "mlx-community/Qwen3-4B")
# Auto-routed to: create_llm("mlx", "mlx-community/Qwen3-4B")

# GGUF models → HuggingFace GGUF backend
llm = create_llm("mlx", "unsloth/Qwen3-4B-GGUF")
# Auto-routed to: create_llm("huggingface", "unsloth/Qwen3-4B-GGUF")
```

**Token Parameter Examples**:

```python
# Strategy 1: Budget + Output Reserve
llm = create_llm(
    provider="openai",
    model="gpt-4o",
    max_tokens=8000,        # Total budget
    max_output_tokens=2000  # Reserve for output
)

# Strategy 2: Explicit Input + Output
llm = create_llm(
    provider="anthropic",
    model="claude-3.5-sonnet",
    max_input_tokens=6000,   # Explicit input limit
    max_output_tokens=2000   # Explicit output limit
)

# Get configuration help
print(llm.get_token_configuration_summary())
warnings = llm.validate_token_constraints()
```

**Integration with Provider Registry**:

The factory delegates to the centralized provider registry:

```python
# Internal implementation:
from abstractcore.providers.registry import create_provider
return create_provider(provider, model, **kwargs)

# Registry provides:
# - Centralized provider metadata
# - Lazy loading of provider classes
# - Automatic capability detection
# - Installation guidance
```

---

### session.py

Conversation management with history, auto-compaction, persistence, and analytics.

#### BasicSession

```python
class BasicSession:
    """Conversation management: history, generation, persistence, auto-compaction, analytics."""

    def __init__(self, provider, system_prompt=None, tools=None, timeout=None,
                 tool_timeout=None, recovery_timeout=None, auto_compact=False,
                 auto_compact_threshold=6000, temperature=None, seed=None):
```

#### Key Features

**Core Methods**: `generate()`, `add_message()`, `get_history()`, `clear_history()`, `save()`, `load()`
**Metadata Support**: Attach `name`, `location`, custom fields to messages
**Tool Registration**: Pass `tools=[func1, func2]` for automatic tool calling
**Streaming**: `session.generate(prompt, stream=True)` returns iterator
**Glyph Compression** (⚠️ EXPERIMENTAL): `session.generate(prompt, media=["file.txt"], glyph_compression="auto")` - vision models only
**History**: Auto-tracks all interactions with timestamps

#### Auto-Compaction (SOTA 2025)

Automatic summarization when tokens exceed threshold. Preserves system + recent messages, summarizes older context.

**Enable**: `BasicSession(auto_compact=True, auto_compact_threshold=6000)`
**Manual**: `session.force_compact(preserve_recent=8, focus="key points")`
**Strategy**: `[SYSTEM] + [SUMMARY] + [Recent N messages]`

#### Persistence & Analytics

**Save/Load**: `session.save("file.json")` / `BasicSession.load("file.json", provider=llm)`
**Format**: JSON with schema `session-archive/v1` (messages, metadata, tool registry, settings)
**Analytics**: `generate_summary()`, `generate_assessment()`, `extract_facts()`, `analyze_intents()`
**Timeouts**: Configure `timeout` (HTTP), `tool_timeout` (tools), `recovery_timeout` (circuit breaker)
**Token Estimation**: `get_token_estimate()`, `should_compact(limit)`

#### Glyph Compression (⚠️ EXPERIMENTAL)

Visual-text compression for large documents using vision models. Converts text to optimized images for 3-4x token savings.

**Parameter Passing**: The `glyph_compression` parameter passes through `**kwargs` to the provider:

```python
session = BasicSession(provider=llm)

# Auto mode (default): compress if beneficial
response = session.generate(
    "Summarize this",
    media=["long_document.txt"]
    # glyph_compression="auto" is default
)

# Force compression (requires vision model)
response = session.generate(
    "Summarize this",
    media=["long_document.txt"],
    glyph_compression="always"  # Raises UnsupportedFeatureError if model lacks vision
)

# Disable compression
response = session.generate(
    "Summarize this",
    media=["long_document.txt"],
    glyph_compression="never"
)
```

**Vision Model Requirement**: ONLY works with vision-capable models (gpt-4o, claude-3-5-sonnet, llama3.2-vision, etc.)

**Error Handling**:
- `glyph_compression="always"` + non-vision model → `UnsupportedFeatureError`
- `glyph_compression="auto"` + non-vision model → Warning logged, falls back to text

See [Compression Module](../compression/README.md) for detailed documentation.

---

## Integration Points

### How Components Work Together

**1. Factory → Interface → Provider**

```python
# User calls factory
llm = create_llm("openai", "gpt-4o", max_tokens=8000)

# Factory delegates to registry
provider = create_provider("openai", "gpt-4o", max_tokens=8000)

# Provider inherits from AbstractCoreInterface
class OpenAIProvider(AbstractCoreInterface):
    def generate(self, prompt, **kwargs):
        # Token management handled by base class
        # Provider implements generation logic
        ...
```

**2. Session → Provider → Types**

```python
# Session manages conversation
session = BasicSession(provider=llm)

# Session calls provider.generate()
response = session.generate("Hello")  # Returns GenerateResponse

# Session adds Message to history
message = Message(role="assistant", content=response.content)
session.messages.append(message)
```

**3. RetryManager → Provider → Events**

```python
# Provider uses RetryManager internally
retry_manager = RetryManager(config)

# Execute with retry protection
response = retry_manager.execute_with_retry(
    func=self._call_api,
    provider_key=f"{self.provider}:{self.model}"
)

# RetryManager emits events
emit_global(EventType.RETRY_ATTEMPTED, {...})
emit_global(EventType.RETRY_EXHAUSTED, {...})
```

**4. Token Management Flow**

```
create_llm(max_tokens=8000, max_output_tokens=2000)
    ↓
AbstractCoreInterface.__init__()
    ↓
_validate_token_parameters()  # Ensure constraints
    ↓
_calculate_effective_token_limits()  # max_input = 8000 - 2000
    ↓
Provider.generate() uses calculated limits
    ↓
Provider maps to API-specific parameters
```

---

## Usage Patterns

| Pattern | Key Code | Description |
|---------|----------|-------------|
| **Basic LLM** | `llm = create_llm("openai", "gpt-4o")`<br>`response = llm.generate("...")` | Simple generation |
| **Session** | `session = BasicSession(llm, system_prompt="...")`<br>`session.generate("...")` | Multi-turn conversations |
| **Token Config** | `llm = create_llm("openai", "gpt-4o", max_tokens=8000, max_output_tokens=2000)`<br>`llm.validate_token_constraints()` | Budget management |
| **Retry** | `RetryManager(RetryConfig(max_attempts=5))`<br>`retry_manager.execute_with_retry(func)` | Resilient execution |
| **Auto-Compact** | `BasicSession(llm, auto_compact=True, auto_compact_threshold=6000)` | Long conversations |
| **Persistence** | `session.save("file.json")`<br>`BasicSession.load("file.json", provider=llm)` | Save/restore sessions |

---

## Best Practices

### DO
1. **Use factory**: `create_llm()` instead of direct provider imports
2. **Configure tokens**: Set explicit `max_tokens` and `max_output_tokens`
3. **Use sessions**: `BasicSession` for automatic history management
4. **Enable auto-compact**: `auto_compact=True` for long conversations
5. **Validate config**: Call `validate_token_constraints()` after creation
6. **Add metadata**: Use `name`, `location` in messages for context
7. **Use helpers**: `get_token_estimate()`, `should_compact()` for monitoring

### DON'T
1. **Instantiate providers directly**: Bypasses registry and validation
2. **Ignore token warnings**: Fix configuration or adjust limits
3. **Mix manual + session history**: Use `add_message()` consistently
4. **Ignore circuit breaker**: Use `RetryManager` for resilience
5. **Skip session saves**: Call `session.save()` periodically

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Token constraints** | `max_input + max_output > max_tokens` | Ensure sum ≤ total |
| **Context overflow** | Long conversations exceed window | Enable `auto_compact=True` |
| **Lost metadata** | Messages without context | Add `name`, `location` fields |
| **Session persistence** | Provider not re-attached on load | Pass `provider=llm` to `load()` |
| **Retry exhaustion** | Uncontrolled retries | Use `RetryManager` with circuit breaker |
| **Stream abandonment** | Not consuming full stream | Always iterate to completion |

---

## Testing Strategy

**Unit Tests**: Enums (value checks), Types (serialization), Interface (token validation via mock), RetryManager (attempt counting), Session (history + persistence)

**Integration Tests**: Full conversation flow with real LLM (create → generate → save → load → continue)

---

## Public API

### Primary Exports

**From `abstractcore`**:
```python
from abstractcore import create_llm  # Factory function
```

**From `abstractcore.core`**:
```python
from abstractcore.core import (
    # Factory
    create_llm,

    # Types
    Message,
    GenerateResponse,

    # Enums
    MessageRole,
    ModelParameter,
    ModelCapability,

    # Session
    BasicSession,

    # Interface (for custom providers)
    AbstractCoreInterface,

    # Retry (for advanced usage)
    RetryManager,
    RetryConfig,
    CircuitBreaker,
)
```

### Recommended Imports for Applications

**Basic Usage**:
```python
from abstractcore import create_llm
from abstractcore.core import BasicSession
```

**Advanced Usage**:
```python
from abstractcore import create_llm
from abstractcore.core import (
    BasicSession,
    Message,
    GenerateResponse,
    MessageRole,
    RetryManager,
    RetryConfig,
)
```

**Custom Provider Development**:
```python
from abstractcore.core import (
    AbstractCoreInterface,
    GenerateResponse,
    Message,
    MessageRole,
    ModelCapability,
)
```

---

## Summary

The **Core Module** provides the essential building blocks for AbstractCore:

1. **Unified Abstractions**: `AbstractCoreInterface` ensures consistency across all providers
2. **Token Management**: Comprehensive vocabulary and helpers for context window control
3. **Retry Patterns**: Production-ready exponential backoff with circuit breakers
4. **Type System**: Robust data structures with metadata and serialization
5. **Session Management**: Conversation tracking with auto-compaction and analytics
6. **Factory Pattern**: Simple, consistent provider creation

**Key Strengths**:
- Provider-agnostic design
- SOTA 2025 best practices
- Comprehensive token management
- Production-ready retry logic
- Conversation continuity via compaction
- Rich metadata support
- Complete observability

**When to Use**:
- Use `create_llm()` for all provider instantiation
- Use `BasicSession` for conversational applications
- Use `RetryManager` for robust production systems
- Use token helpers for capacity planning
- Use auto-compaction for long-running conversations

This module is the foundation upon which all AbstractCore functionality is built, ensuring consistency, reliability, and ease of use across the entire framework.

## Related Modules

**Direct dependencies**:
- [`providers/`](../providers/README.md) - Factory creates provider instances
- [`exceptions/`](../exceptions/README.md) - Error handling throughout core
- [`events/`](../events/README.md) - Event emission from factory operations
- [`config/`](../config/README.md) - Configuration integration

**Used by**:
- [`architectures/`](../architectures/README.md) - Model capability abstractions
- [`media/`](../media/README.md) - Media processing base classes
- [`compression/`](../compression/README.md) - Compression orchestration
- [`structured/`](../structured/README.md) - Response model handling
- [`tools/`](../tools/README.md) - Tool execution framework
- [`processing/`](../processing/README.md) - High-level processors
- [`apps/`](../apps/README.md) - Application-level integrations
- [`server/`](../server/README.md) - API endpoints
