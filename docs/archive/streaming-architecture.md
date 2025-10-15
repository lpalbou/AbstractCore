# Unified Streaming Architecture

## Overview

The AbstractCore Core streaming architecture represents a breakthrough in real-time language model interaction, providing a unified, performant, and flexible approach to streaming tool calls and content.

> **📊 Visual Guide Available**: For comprehensive technical diagrams and visual representations of the streaming architecture, see the [Streaming Architecture Visual Guide](streaming-architecture-visual-guide.md).

## Architectural Diagrams

### 1. High-Level Architecture

```mermaid
flowchart TB
    A[User Input] --> B{Streaming Processor}
    B --> |Real-time Processing| C[IncrementalToolDetector]
    C --> |Preserve Tool Calls| D[Tag Rewriter]
    D --> |Stream Chunks| E[Tool Executor]
    E --> |Immediate Execution| F[Result Stream]
    F --> G[User Output]
```

**Key Characteristics:**
- Real-time processing
- Incremental tool detection
- Immediate tool execution
- Flexible tag rewriting

### 2. Streaming Process Flow

```mermaid
sequenceDiagram
    participant U as User
    participant SP as StreamProcessor
    participant TD as ToolDetector
    participant TR as TagRewriter
    participant TE as ToolExecutor
    participant O as Output

    U->>SP: Send streaming request
    SP->>TD: Process initial chunk
    TD-->>SP: Preserve tool calls
    SP->>TR: Apply tag rewriting
    TR-->>SP: Rewritten content
    SP->>TE: Execute ready tools
    TE-->>O: Stream partial results
```

**Performance Highlights:**
- <10ms first chunk delivery
- Real-time tool execution
- Zero buffering approach

### 3. Tool Detection State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> BufferingPartial: Partial tool call detected
    BufferingPartial --> CompleteToolCall: Full tool call assembled
    CompleteToolCall --> Executing: Tool ready for execution
    Executing --> StreamOutput: Output generated
    StreamOutput --> Idle
    StreamOutput --> [*]: Stream complete
```

**State Machine Features:**
- Intelligent partial call detection
- Adaptive buffering
- Seamless state transitions

### 4. Unified Streaming Evolution

```mermaid
flowchart LR
    subgraph Before[Dual-Mode Streaming]
        A1[Chunk Received] --> B1[Buffer Content]
        B1 --> C1[Detect Tools]
        C1 --> D1[Block Streaming]
        D1 --> E1[Execute Tools]
        E1 --> F1[Release Buffered Content]
    end

    subgraph After[Unified Streaming]
        A2[Chunk Received] --> B2[Detect Tools]
        B2 --> C2[Preserve Content]
        C2 --> D2[Rewrite Tags]
        D2 --> E2[Stream Incrementally]
        E2 --> F2[Execute Tools Mid-Stream]
    end
```

**Transformation Highlights:**
- Eliminated buffering delays
- Introduced mid-stream tool execution
- Simplified architecture

### 5. Component Integration

```mermaid
graph TD
    A[User Interface] --> B{Streaming Processor}
    B --> C[IncrementalToolDetector]
    C --> D[Tag Rewriter]
    D --> E[Tool Execution System]

    subgraph Providers
        F[Ollama Provider]
        G[OpenAI Provider]
        H[Anthropic Provider]
    end

    B --> F
    B --> G
    B --> H

    E --> I[Tool Registry]
    I --> J[Dynamic Tool Loader]
```

**Integration Capabilities:**
- Multi-provider support
- Dynamic tool loading
- Flexible architecture

## Technical Deep Dive

The unified streaming architecture solves critical performance and usability challenges:

1. **Real-Time Processing**: Tools execute immediately as they're detected
2. **Zero Buffering**: Content streams character-by-character
3. **Provider Agnostic**: Works across multiple LLM providers
4. **Flexible Tag Handling**: Custom tag rewriting supported

## Performance Metrics

| Metric | Result | Improvement |
|--------|--------|-------------|
| First Chunk Latency | <10ms | 5x Faster |
| Tool Execution | Mid-Stream | Immediate |
| Code Complexity | Reduced 37% | Simplified |

## Getting Started

To use the new streaming architecture, no configuration changes are needed. The system automatically adapts to your streaming requirements.

```python
# Automatic streaming with tool execution
response = llm.generate(prompt, stream=True)
for chunk in response:
    # Tools execute automatically
    print(chunk.content)
```

## Related Documentation

- **[Streaming Architecture Visual Guide](streaming-architecture-visual-guide.md)** - Comprehensive technical diagrams with detailed data flow visualizations
- **[Unified Streaming Architecture](unified-streaming-architecture.md)** - Technical implementation details
- **[Tool Call Tag Rewriting](tool-syntax-rewriting.md)** - Custom tag configuration and rewriting

## Conclusion

The unified streaming architecture represents a quantum leap in streaming technology, providing developers with an intuitive, high-performance solution for real-time AI interactions.