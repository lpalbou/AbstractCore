# AbstractCore Architecture

AbstractCore provides a unified interface to all major LLM providers with production-grade reliability. This document explains how it works internally and why it's designed this way.

## System Overview

AbstractCore operates as both a Python library and an optional HTTP server:

```mermaid
graph TD
    A[Your Application] --> B[AbstractCore API]
    AA[HTTP Clients] --> BB[AbstractCore Server]
    BB --> B
    
    B --> C[Provider Interface]
    C --> D[Event System]
    C --> E[Tool System]
    C --> F[Retry System]
    C --> G[Provider Implementations]

    G --> H[OpenAI Provider]
    G --> HH[OpenAI-Compatible Provider]
    G --> I[Anthropic Provider]
    G --> J[Ollama Provider]
    G --> K[MLX Provider]
    G --> L[LMStudio Provider]
    G --> M[HuggingFace Provider]
    G --> MM[vLLM Provider]
    G --> MN[OpenRouter Provider]

    H --> N[OpenAI API]
    HH --> NN[OpenAI-Compatible /v1 Endpoint]
    I --> O[Anthropic API]
    J --> P[Ollama Server]
    K --> Q[MLX Models]
    L --> R[LMStudio Server]
    M --> S[HuggingFace Models]
    MM --> RR[vLLM Server]
    MN --> RO[OpenRouter API]

    style B fill:#e1f5fe
    style BB fill:#4caf50
    style C fill:#f3e5f5
    style G fill:#fff3e0
```

## Design Principles

### 1. Provider Abstraction
**Goal**: Same interface for all providers
**Implementation**: Common interface with provider-specific implementations

### 2. Production Reliability
**Goal**: Handle real-world failures gracefully
**Implementation**: Built-in retry logic, circuit breakers, comprehensive error handling

### 3. Universal Tool Support
**Goal**: Tools work everywhere, even with providers that don't support them natively
**Implementation**: Native support where available, intelligent prompting as fallback

### 4. Simplicity Over Features
**Goal**: Clean, focused API that's easy to understand
**Implementation**: Minimal core with clear extension points

### 5. Optional HTTP Access
**Goal**: Flexible deployment as library or server
**Implementation**: OpenAI-compatible REST API built on core library

## Core Components

### 1. Factory Pattern (`create_llm`)

The main entry point uses the factory pattern for clean provider instantiation:

```mermaid
graph LR
    A[create_llm] --> B{Provider Type}
    B --> C[OpenAI Provider]
    B --> D[Anthropic Provider]
    B --> E[Ollama Provider]
    B --> F[Other Providers...]

    C --> G[Configured Instance]
    D --> G
    E --> G
    F --> G

    style A fill:#4caf50
    style G fill:#2196f3
```

```python
from abstractcore import create_llm

# Factory creates the right provider with proper configuration
llm = create_llm("openai", model="gpt-4o-mini", temperature=0.7)

# OpenAI-compatible /v1 endpoints (LMStudio, vLLM, custom proxies)
llm_local = create_llm("lmstudio", model="qwen/qwen3-4b-2507", base_url="http://localhost:1234/v1")
llm_openrouter = create_llm("openrouter", model="openai/gpt-4o-mini")  # requires OPENROUTER_API_KEY
```

### 2. Provider Interface

All providers implement `AbstractCoreInterface`:

```python
class AbstractCoreInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerateResponse:
        """Generate response from LLM"""

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get provider capabilities"""

    def unload_model(self, model_name: str) -> None:
        """Unload/cleanup resources for a specific model (best-effort)"""
```

This ensures:
- **Consistency**: Same methods across all providers
- **Reliability**: Standardized error handling
- **Extensibility**: Easy to add new providers
- **Memory Management**: Explicit control over model lifecycle

#### Response Normalization (Model Output Cleanup)

`BaseProvider` also applies **asset-driven response normalization** so downstream code sees clean, consistent output across providers:

- **Output wrappers**: Strip configured leading/trailing wrapper tokens (e.g., GLM `<|begin_of_box|>…<|end_of_box|>`)
- **Harmony transcripts (GPT-OSS)**: Extract `<|channel|>final` into `GenerateResponse.content` and capture `<|channel|>analysis` as `GenerateResponse.metadata["reasoning"]` (non-streaming)
- **Thinking tags**: Extract inline `<think>...</think>` blocks into `GenerateResponse.metadata["reasoning"]` (when configured)

**Why this belongs in `BaseProvider` (even for streaming):**
- These artifacts are **model/template-specific**, not provider-specific (the same model can be served via Ollama, vLLM, LMStudio, HF, or MLX)
- In streaming mode, wrappers often appear in the first/last chunks; stripping them incrementally avoids leaking markup into UIs and tool parsers without buffering the full response

Configuration comes from `abstractcore/assets/architecture_formats.json` and `abstractcore/assets/model_capabilities.json`; implementation lives in `abstractcore/architectures/response_postprocessing.py`.

#### Memory Management

The `unload_model(model_name)` method is a **best-effort resource cleanup hook**.

- **API providers** (OpenAI, Anthropic): typically a no-op (safe to call).
- **Local / self-hosted providers**: behavior is provider-specific:
  - some can actively release memory (or request server-side eviction),
  - others can only close client connections and rely on server-side TTL/auto-eviction.
  - Example: **LMStudio** does not expose an explicit “unload model” API; `unload_model()` closes HTTP clients and relies on LMStudio TTL/auto-evict.

In the OpenAI-compatible AbstractCore server (`abstractcore.server.app`), requests can set `unload_after` (default `false`)
to call `llm.unload_model(model)` after the request completes. For providers that can unload shared server state (e.g. Ollama),
this is disabled by default and must be explicitly enabled by the server operator.

```python
# Load model, use it, then free memory
llm = create_llm("ollama", model="large-model")
response = llm.generate("Hello")
llm.unload_model(llm.model)  # Explicitly free memory
del llm
```

This is critical for:
- Test suites that load multiple models sequentially
- Memory-constrained environments (<32GB RAM)
- Production systems serving different models sequentially

### 3. Media Handling System

AbstractCore includes a production-ready media processing system that enables universal file attachment across all providers:

```mermaid
graph TD
    A[User Input: @file.pdf] --> B[MessagePreprocessor]
    B --> C[Extract Files + Clean Text]
    C --> D[AutoMediaHandler]
    D --> E{File Type Detection}
    E -->|Images| F[ImageProcessor]
    E -->|PDFs| G[PDFProcessor]
    E -->|Office| H[OfficeProcessor]
    E -->|Text/CSV| I[TextProcessor]

    F --> J[MediaContent Objects]
    G --> J
    H --> J
    I --> J

    J --> K{Provider Type}
    K -->|OpenAI| L[OpenAI Format]
    K -->|Anthropic| M[Anthropic Format]
    K -->|Local| N[Text Embedding]

    L --> O[Provider API Call]
    M --> O
    N --> O

    style D fill:#4caf50
    style J fill:#2196f3
    style O fill:#ff9800
```

#### Media System Architecture

**Core Components:**
- **MessagePreprocessor**: Parses `@filename` syntax in CLI and extracts file references
- **AutoMediaHandler**: Intelligent coordinator that selects appropriate processors
- **Specialized Processors**:
  - `ImageProcessor` (PIL-based for images)
  - `PDFProcessor` (PyMuPDF4LLM for documents)
  - `OfficeProcessor` (Unstructured for DOCX/XLSX/PPTX)
  - `TextProcessor` (pandas for CSV/TSV data analysis)
- **Provider Handlers**: Format media content for each provider's API requirements

**Provider-Specific Formatting:**
```python
# Same MediaContent gets formatted differently:

# OpenAI (JSON with image_url):
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Analyze this"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}

# Anthropic (Messages API with source):
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Analyze this"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
  ]
}

# Local (Text embedding):
"Analyze this\n\nImage description: A chart showing quarterly trends..."
```

**Graceful Fallback Strategy:**
1. **Advanced Processing**: PyMuPDF4LLM, Unstructured libraries
2. **Basic Processing**: Simple text extraction
3. **Metadata Fallback**: File information and properties
4. **Degrades gracefully for documents**: PDFs/Office/text aim to return best-effort extracted text/metadata rather than crashing.
5. **Policy-driven for true multimodal inputs (planned)**: for image/audio/video message parts, behavior is policy-driven; unsupported requests should fail loudly unless an explicit enrichment fallback is configured (see ADR-0028).

#### Unified Media API

The same `media=[]` parameter works across all providers:

```python
# Universal API - works with any provider
llm = create_llm("openai", model="gpt-4o")  # or "anthropic", "ollama", etc.
response = llm.generate(
    "Analyze these files",
    media=["report.pdf", "chart.png", "data.xlsx"]
)
```

**CLI Integration:**
```bash
# Simple @filename syntax works everywhere
python -m abstractcore.utils.cli --prompt "What's in @document.pdf and @image.jpg"
```

#### Planned: Capabilities plugins (voice/vision)
To keep `abstractcore` dependency-light (ADR-0001) while still enabling “must work” TTS/STT and vision generation:
- multimodal transforms (TTS/STT/T2I/…) are planned to integrate via optional **capability plugins** (ADR-0028),
- and input enrichment fallbacks (image/audio/video → short observations injected into the main LLM request) are planned to be **explicit and config-driven** (no silent semantic change).

### 4. Request Lifecycle

```mermaid
sequenceDiagram
    participant App as Your App
    participant Core as AbstractCore
    participant Events as Event System
    participant Retry as Retry Logic
    participant Provider as LLM Provider
    participant Tools as Tool System

    App->>Core: generate("prompt", tools=tools)
    Core->>Events: emit(GENERATION_STARTED)
    Core->>Retry: wrap_with_retry()

    alt Provider Call Success
        Retry->>Provider: API call
        Provider->>Retry: response
        Retry->>Core: successful response
    else Provider Call Fails
        Retry->>Provider: API call (attempt 1)
        Provider->>Retry: rate limit error
        Retry->>Retry: wait with backoff
        Retry->>Provider: API call (attempt 2)
        Provider->>Retry: success
        Retry->>Core: successful response
    end

    alt Has Tool Calls
        Core->>Events: emit(TOOL_STARTED)
        Core->>Tools: execute_tools()
        Tools->>Core: tool results
        Core->>Events: emit(TOOL_COMPLETED)
    end

    Core->>Events: emit(GENERATION_COMPLETED)
    Core->>App: GenerateResponse
```

### 4. Tool System Architecture

The tool system provides universal tool execution across all providers:

```mermaid
graph TD
    A[LLM Response] --> B{Has Tool Calls?}
    B -->|No| C[Return Response]
    B -->|Yes| D[Parse Tool Calls]
    D --> E[Event: TOOL_STARTED]
    E --> F{Event Prevented?}
    F -->|Yes| G[Skip Tool Execution]
    F -->|No| H[Execute Tools]
    H --> I[Collect Results]
    I --> J[Event: TOOL_COMPLETED]
    J --> K[Append Results to Response]
    K --> C

    style D fill:#ffeb3b
    style H fill:#4caf50
    style E fill:#ff9800
```

#### Tool Execution Flow

1. **Tool Detection**: Parse tool calls from LLM response
2. **Event Emission**: Emit `TOOL_STARTED` (preventable)
3. **Local Execution**: Execute tools in AbstractCore (not by provider)
4. **Result Collection**: Gather results and error information
5. **Event Emission**: Emit `TOOL_COMPLETED` with results
6. **Response Integration**: Append tool results to original response

#### Provider-Specific Tool Handling with Tag Rewriting

```mermaid
graph LR
    A[Tool Definition] --> B{Provider Type}
    B --> C[OpenAI: Native JSON]
    B --> D[Anthropic: Native XML]
    B --> E[Ollama: Architecture-specific]
    B --> F[Others: Prompted Format]

    C --> G[LLM Generation]
    D --> G
    E --> G
    F --> G

    G --> H[Tool Call Tag Rewriter]
    H --> I[Target Format Conversion]
    I --> J[Universal Tool Parser]
    J --> K[Local Tool Execution]

    style A fill:#e1f5fe
    style H fill:#ff9800
    style I fill:#9c27b0
    style K fill:#4caf50
```

#### Tool Call Tag Rewriting System

AbstractCore includes a sophisticated tag rewriting system that enables compatibility with any agentic CLI:

**Rewriting Pipeline**:

```mermaid
graph TD
    A[Raw LLM Response] --> B[Pattern Detection]
    B --> C{Tag Format Needed?}
    C -->|No| D[Default Qwen3 Format]
    C -->|Yes| E[Target Format Conversion]

    E --> F{Format Type}
    F -->|Predefined| G[llama3, xml, gemma, etc.]
    F -->|Custom| H[User-defined Tags]

    G --> I[Rewritten Tool Call]
    H --> I
    D --> I

    I --> J[Tool Execution]

    style B fill:#2196f3
    style E fill:#ff9800
    style I fill:#4caf50
```

**Supported Formats**:
- **Default (Qwen3)**: `<|tool_call|>...JSON...</|tool_call|>` - Compatible with Codex CLI
- **LLaMA3**: `<function_call>...JSON...</function_call>` - Compatible with Crush CLI
- **XML**: `<tool_call>...JSON...</tool_call>` - Compatible with Gemini CLI
- **Gemma**: ````tool_code...JSON...```` - Compatible with Gemma models
- **Custom**: Any user-defined format (e.g., `[TOOL]...JSON...[/TOOL]`)

**Real-Time Integration**:
- **Streaming Compatible**: Works seamlessly with unified streaming architecture
- **Zero Latency**: No additional processing delays
- **Universal Detection**: Automatically detects source format from any model
- **Graceful Fallback**: Returns original content if rewriting fails

### 5. Retry and Reliability System

Production-grade error handling with multiple layers:

```mermaid
graph TD
    A[LLM Request] --> B[Retry Manager]
    B --> C{Error Type}
    C -->|Rate Limit| D[Exponential Backoff]
    C -->|Network Error| D
    C -->|Timeout| D
    C -->|Auth Error| E[Fail Fast]
    C -->|Invalid Request| E

    D --> F{Max Attempts?}
    F -->|No| G[Wait + Jitter]
    G --> H[Retry Request]
    H --> B
    F -->|Yes| I[Circuit Breaker]

    I --> J{Failure Threshold?}
    J -->|No| K[Return Error]
    J -->|Yes| L[Open Circuit]
    L --> M[Fail Fast for Duration]

    style D fill:#ff9800
    style I fill:#f44336
    style L fill:#d32f2f
```

#### Retry Configuration

```python
from abstractcore import create_llm
from abstractcore.core.retry import RetryConfig

config = RetryConfig(
    max_attempts=3,           # Try up to 3 times
    initial_delay=1.0,        # Start with 1 second delay
    max_delay=60.0,           # Cap at 1 minute
    use_jitter=True,          # Add randomness
    failure_threshold=5,      # Circuit breaker after 5 failures
    recovery_timeout=60.0     # Test recovery after 1 minute
)

llm = create_llm("openai", model="gpt-4o-mini", retry_config=config)
```

### 6. Event System

Comprehensive observability and control through events:

```mermaid
graph TD
    A[LLM Operation] --> B[Event Emission]
    B --> C[Global Event Bus]
    C --> D[Event Listeners]

    D --> E[Monitoring]
    D --> F[Logging]
    D --> G[Cost Tracking]
    D --> H[Tool Control]
    D --> I[Custom Logic]

    E --> J[Metrics Dashboard]
    F --> K[Log Files]
    G --> L[Cost Alerts]
    H --> M[Security Gates]
    I --> N[Business Logic]

    style B fill:#9c27b0
    style C fill:#673ab7
    style H fill:#f44336
```

#### Event Types and Use Cases

```python
from abstractcore.events import EventType, on_global

# Cost monitoring
def monitor_costs(event):
    if event.cost_usd and event.cost_usd > 0.10:
        alert(f"High cost request: ${event.cost_usd}")

# Security control
def prevent_dangerous_tools(event):
    for call in event.data.get('tool_calls', []):
        if call.name in ['delete_file', 'system_command']:
            event.prevent()  # Stop tool execution

# Performance tracking
def track_performance(event):
    if event.duration_ms > 10000:
        log(f"Slow request: {event.duration_ms}ms")

on_global(EventType.GENERATION_COMPLETED, monitor_costs)
on_global(EventType.TOOL_STARTED, prevent_dangerous_tools)
on_global(EventType.GENERATION_COMPLETED, track_performance)
```

### 7. Structured Output System with Streaming Integration

Type-safe responses with automatic validation, retry, and unified streaming:

```mermaid
graph TD
    A[LLM Generate] --> B{Streaming Mode?}
    B -->|Yes| C[Unified Streaming Processor]
    B -->|No| D[Standard JSON Parsing]

    C --> E[Incremental Tool Detector]
    E --> F[Real-time Chunk Processing]
    F --> G[Tool Call Detection]
    G --> H[Mid-Stream Tool Execution]

    D --> I[Parse JSON]
    I --> J{Valid JSON?}
    J -->|No| K[Retry with Error Feedback]
    J -->|Yes| L[Pydantic Validation]

    L --> M{Valid Model?}
    M -->|No| K
    M -->|Yes| N[Return Typed Object]

    K --> O{Max Retries?}
    O -->|No| A
    O -->|Yes| P[Raise ValidationError]

    style C fill:#4caf50
    style E fill:#2196f3
    style F fill:#ff9800
    style G fill:#9c27b0
    style K fill:#f44336
```

#### Unified Streaming Architecture

AbstractCore's streaming system provides high-performance, character-by-character streaming with real-time tool detection:

**Architecture Components**:

```mermaid
graph TD
    A[Stream Input] --> B[UnifiedStreamProcessor]
    B --> C[IncrementalToolDetector]
    C --> D[Tag Rewriter]
    D --> E[Tool Execution]
    E --> F[Stream Output]

    B --> G[Character-by-Character Handling]
    G --> H[Intelligent Buffering]
    H --> C

    style B fill:#4caf50
    style C fill:#2196f3
    style D fill:#ff9800
    style E fill:#9c27b0
```

**Key Features**:

1. **Unified Streaming Strategy**
   - Single consistent approach across all providers
   - First chunk delivery in <10ms
   - Minimal code complexity

2. **Incremental Tool Detection**
   - Real-time tool call detection during streaming
   - Emits `chunk.tool_calls` as soon as a full tool call is detected
   - Handles partial tool calls across chunk boundaries

3. **Character-by-Character Streaming**
   - Handles micro-chunking from providers (22+ tiny chunks per tool call)
   - Intelligent buffering for partial tool calls
   - Robust parsing with auto-repair for malformed JSON

4. **Tool Call Tag Rewriting Integration**
   - Real-time format conversion during streaming
   - Support for multiple formats (Qwen3, LLaMA3, Gemma, XML, custom)
   - Zero buffering overhead for tag conversion

**Streaming with Tag Rewriting Example**:
```python
# Real-time streaming with automatic tool call format conversion
for chunk in llm.generate(
    "Create a Python function and analyze it",
    stream=True,
    tools=[code_analysis_tool],
    tool_call_tags="llama3"  # Convert to Crush CLI format
):
    # Immediate character-by-character output
    print(chunk.content, end="", flush=True)

    # Tool calls are surfaced as structured dicts; execute them in your host/runtime.
    if chunk.tool_calls:
        print(f"\nTool calls: {chunk.tool_calls}")

# Output format: <function_call>{"name": "analyze_code"}...</function_call>
```

**Performance Characteristics**:
- **First Chunk Latency**: <10ms across all providers
- **Tool Detection Overhead**: <1ms per chunk
- **Memory Efficiency**: Linear, bounded growth
- **Character-by-Character Support**: Handles extreme micro-chunking
- **Tag Rewriting**: Zero additional latency

#### Automatic Error Feedback

When validation fails, AbstractCore provides detailed feedback to the LLM:

```python
# If LLM returns invalid data, AbstractCore automatically retries with:
"""
IMPORTANT: Your previous response had validation errors:
• Field 'age': Age must be positive (got -25)
• Field 'email': Invalid email format

Please correct these errors and provide valid JSON.
"""
```

### 8. Session Management

Simple conversation memory without complexity:

```mermaid
graph LR
    A[BasicSession] --> B[Message History]
    A --> C[System Prompt]
    A --> D[Provider Reference]

    B --> E[generate()]
    C --> E
    D --> E

    E --> F[Add to History]
    F --> G[Return Response]

    A --> H[save()/load()]
    H --> I[JSON Persistence]

    style A fill:#2196f3
    style B fill:#4caf50
```

### 9. Server Architecture (Optional Component)

The AbstractCore server provides OpenAI-compatible HTTP endpoints built on top of the core library:

```mermaid
graph TD
    A[HTTP Client] --> B[FastAPI Server]
    B --> C{Endpoint Router}
    
    C --> D[/v1/chat/completions]
    C --> E[/v1/embeddings]
    C --> F[/v1/models]
    C --> G[/providers]
    
    D --> H[Request Validation]
    E --> H
    F --> I[Provider Discovery]
    G --> I
    
    H --> J[AbstractCore Library]
    I --> J
    
    J --> K[Provider Interface]
    K --> L[LLM Providers]
    
    style B fill:#4caf50
    style J fill:#e1f5fe
    style K fill:#f3e5f5
```

**Architecture Layers**:

1. **HTTP Layer**: FastAPI-based REST API with request validation
2. **Translation Layer**: Converts HTTP requests to AbstractCore library calls
3. **Core Layer**: Uses the full AbstractCore provider system
4. **Response Layer**: Transforms responses to OpenAI-compatible format

**Key Capabilities**:

- **OpenAI Compatibility**: Drop-in replacement for OpenAI API clients
- **Universal Provider Access**: Single API for all providers (OpenAI, Anthropic, Ollama, etc.)
- **Format Conversion**: Automatic tool call format conversion for agentic CLIs
- **Streaming Support**: Server-sent events for real-time responses
- **Model Discovery**: Dynamic model listing across all providers
- **Embedding Support**: Multi-provider embedding generation (HuggingFace, Ollama, LMStudio)

**Request Flow Example**:

```mermaid
sequenceDiagram
    participant Client
    participant Server as FastAPI Server
    participant Core as AbstractCore
    participant Provider as LLM Provider
    
    Client->>Server: POST /v1/chat/completions
    Server->>Server: Validate Request
    Server->>Core: create_llm(provider, model)
    Server->>Core: llm.generate(messages, tools)
    Core->>Provider: API call with retry logic
    Provider->>Core: Response
    Core->>Core: Execute tools if needed
    Core->>Server: GenerateResponse
    Server->>Server: Convert to OpenAI format
    Server->>Client: HTTP Response (streaming or complete)
```

**Server Features**:

- **Automatic Retry**: Built-in retry logic from core library
- **Event System**: Full observability through events
- **Debug Logging**: Comprehensive request/response logging
- **Health Checks**: `/health` endpoint for monitoring
- **Interactive Docs**: Auto-generated Swagger UI at `/docs`
- **Multi-Worker Support**: Production deployment with multiple workers

## Architecture Benefits

### 1. Provider Agnostic
- **Same code works everywhere**: Switch providers by changing one line
- **No vendor lock-in**: Easy migration between cloud and local providers
- **Consistent behavior**: Tools, streaming, structured output work identically

### 2. Production Ready
- **Automatic reliability**: Built-in retry logic and circuit breakers
- **Comprehensive observability**: Events for every operation
- **Error handling**: Proper error classification and handling

### 3. Extensible
- **Event system**: Hook into any operation
- **Tool system**: Add new tools easily
- **Provider system**: Add new providers with minimal code

### 4. Performance Optimized
- **Lazy loading**: Providers loaded only when needed
- **Connection pooling**: Reuse HTTP connections
- **Efficient parsing**: Optimized JSON and tool parsing

## Extension Points

AbstractCore is designed to be extended:

### Adding a New Provider

```python
from abstractcore.providers.base import BaseProvider

class MyProvider(BaseProvider):
    def generate(self, prompt: str, **kwargs) -> GenerateResponse:
        # Implement provider-specific logic
        return GenerateResponse(content="...")

    def get_capabilities(self) -> List[str]:
        return ["text_generation", "streaming"]
```

### Adding Custom Events

```python
from abstractcore.events import EventType, emit_global

class EventType(Enum):  # Extend the enum
    CUSTOM_EVENT = "custom_event"

# Emit custom events
emit_global(EventType.CUSTOM_EVENT, data={"custom": "data"})
```

### Adding Tools

```python
from abstractcore.tools import register_tool

@register_tool
def my_custom_tool(param: str) -> str:
    """Custom tool that does something useful."""
    return f"Processed: {param}"
```

## Performance Characteristics

### Memory Usage
- **Core**: ~15MB base memory
- **Per Provider**: ~2-5MB additional
- **Scaling**: Linear with number of concurrent requests

### Latency Overhead
- **Provider abstraction**: ~1-2ms overhead
- **Event system**: ~0.5ms per event
- **Tool parsing**: ~1-5ms depending on complexity
- **Retry logic**: Only on failures

### Throughput
- **Single instance**: 100+ requests/second
- **Bottleneck**: Usually the LLM provider, not AbstractCore
- **Scaling**: Horizontal scaling through multiple instances

## Security Considerations

### 1. Tool Execution Safety
- **Local execution**: Tools run in AbstractCore, not by providers
- **Event prevention**: Stop dangerous tools before execution
- **Input validation**: Validate tool parameters

### 2. API Key Management
- **Environment variables**: Secure key storage
- **No logging**: Keys never appear in logs
- **Provider isolation**: Keys scoped to specific providers

### 3. Data Privacy
- **Local options**: Support for local providers (Ollama, MLX)
- **No data retention**: AbstractCore doesn't store conversation data
- **Transparent processing**: All operations are observable through events

## Testing Strategy

### 1. No Mocking Philosophy
- **Real implementations**: Test against actual providers
- **Real models**: Use actual LLM models in tests
- **Real scenarios**: Test real-world usage patterns

### 2. Provider Coverage
- **All providers tested**: Every provider has comprehensive tests
- **Cross-provider consistency**: Same tests run across all providers
- **Feature parity**: Ensure consistent behavior

### 3. Reliability Testing
- **Failure scenarios**: Test retry logic and error handling
- **Performance tests**: Measure latency and throughput
- **Integration tests**: Test with real external dependencies

## Integration with Abstract Framework

AbstractCore is the foundation layer for the Abstract Framework stack:

```mermaid
graph TD
    subgraph "UI Layer (peers)"
        A[AbstractCode<br/>Terminal CLI]
        B[AbstractFlow Visual Editor<br/>React + ReactFlow]
    end

    A -.->|optional| F[AbstractFlow Engine]
    B --> F

    F --> C[AbstractAgent]
    A --> C
    C --> D[AbstractRuntime]
    D --> E[AbstractCore]
    E --> G[LLM Providers]

    style E fill:#e1f5fe
    style A fill:#fff3e0
    style B fill:#fff3e0
    style F fill:#f3e5f5
    style C fill:#f3e5f5
    style D fill:#f3e5f5
```

### Framework Layers
- **UI Layer** (peers):
  - AbstractCode: Terminal CLI for interactive sessions
  - AbstractFlow Visual Editor: Web-based diagram editor (React + ReactFlow + FastAPI)
- **AbstractFlow**: Multi-agent orchestration engine + visual editor
- **AbstractAgent**: Agent patterns (ReactAgent, CodeActAgent) with durable execution
- **AbstractRuntime**: Effect system, workflows, state persistence

AbstractCode can optionally use AbstractFlow for running flows. AbstractFlow includes its own visual editor for designing workflows.

## Summary

AbstractCore's architecture prioritizes:

1. **Reliability** - Production-grade error handling and retry logic
2. **Simplicity** - Clean APIs that are easy to understand and use
3. **Universality** - Same interface and features across all providers
4. **Extensibility** - Clear extension points for advanced features
5. **Observability** - Comprehensive events for monitoring and control
6. **Flexibility** - Deploy as Python library or OpenAI-compatible HTTP server

The result is a foundation that works reliably in production while remaining simple enough to learn quickly and flexible enough to build advanced applications on top of.
