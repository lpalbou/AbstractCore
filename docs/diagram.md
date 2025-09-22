# AbstractLLM Core - Visual Architecture

This document provides visual diagrams to complement the [Architecture Documentation](architecture.md).

## Table of Contents

- [System Overview](#system-overview)
- [Provider Architecture](#provider-architecture)
- [Tool Execution Flow](#tool-execution-flow)
- [Event System](#event-system)
- [Component Relationships](#component-relationships)
- [Data Flow Diagrams](#data-flow-diagrams)

## System Overview

```mermaid
graph TB
    User[User Application] --> AbstractLLM[AbstractLLM Core]
    AbstractLLM --> Providers[Provider Layer]
    AbstractLLM --> Tools[Tool System]
    AbstractLLM --> Events[Event System]
    AbstractLLM --> Session[Session Management]

    Providers --> OpenAI[OpenAI Provider]
    Providers --> Anthropic[Anthropic Provider]
    Providers --> Ollama[Ollama Provider]
    Providers --> HuggingFace[HuggingFace Provider]
    Providers --> MLX[MLX Provider]
    Providers --> LMStudio[LMStudio Provider]

    Tools --> Handler[UniversalToolHandler]
    Tools --> Parser[ToolCallParser]
    Tools --> Executor[Tool Execution]

    Events --> BEFORE[BEFORE_TOOL_EXECUTION]
    Events --> AFTER[AFTER_TOOL_EXECUTION]
    Events --> Prevention[Event Prevention]

    OpenAI --> Models1[GPT-4, GPT-3.5]
    Anthropic --> Models2[Claude-3, Claude-3.5]
    Ollama --> Models3[Llama, Qwen, Mistral]
    HuggingFace --> Models4[Transformers, GGUF]
    MLX --> Models5[MLX Models]
    LMStudio --> Models6[Local Models]
```

## Provider Architecture

```mermaid
classDiagram
    AbstractLLMInterface <|-- BaseProvider
    BaseProvider <|-- OpenAIProvider
    BaseProvider <|-- AnthropicProvider
    BaseProvider <|-- OllamaProvider
    BaseProvider <|-- HuggingFaceProvider
    BaseProvider <|-- MLXProvider
    BaseProvider <|-- LMStudioProvider

    BaseProvider --> EventEmitter
    BaseProvider --> ModelCapabilities
    BaseProvider --> UniversalToolHandler

    class AbstractLLMInterface {
        +generate()
        +get_capabilities()
    }

    class BaseProvider {
        +model: str
        +architecture: Architecture
        +model_capabilities: Dict
        +tool_handler: UniversalToolHandler
        +_generate_internal()
        +_handle_tool_execution()
    }

    class OpenAIProvider {
        +client: OpenAI
        +_format_tools_for_openai()
        +_handle_tool_execution()
        +_stream_response()
    }

    class AnthropicProvider {
        +client: Anthropic
        +_format_tools_for_anthropic()
        +_handle_tool_execution()
        +_stream_response()
    }

    class OllamaProvider {
        +client: httpx.Client
        +_handle_tool_execution()
        +_stream_response()
    }
```

## Tool Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Provider
    participant ToolHandler
    participant EventSystem
    participant ToolExecutor
    participant LLM

    User->>Provider: generate(prompt, tools)
    Provider->>ToolHandler: format_tools_prompt()
    Provider->>LLM: send_request(enhanced_prompt)
    LLM-->>Provider: response_with_tool_calls

    Provider->>ToolHandler: parse_response(content)
    ToolHandler-->>Provider: tool_calls[]

    alt Tool calls found
        Provider->>EventSystem: emit(BEFORE_TOOL_EXECUTION)
        EventSystem-->>Provider: event (can be prevented)

        alt Not prevented
            Provider->>ToolExecutor: execute_tools(tool_calls)
            ToolExecutor-->>Provider: tool_results[]
            Provider->>EventSystem: emit(AFTER_TOOL_EXECUTION)
            Provider->>Provider: merge_results_with_response()
        end
    end

    Provider-->>User: final_response_with_tool_results
```

## Tool System Architecture

```mermaid
graph TB
    subgraph "Tool Definition Layer"
        ToolDef[Tool Definition]
        Schema[JSON Schema]
        Params[Parameters]
    end

    subgraph "Universal Tool Handler"
        Handler[UniversalToolHandler]
        Native[Native Tool Support]
        Prompted[Prompted Tool Support]
        Formatter[Tool Formatter]
    end

    subgraph "Parser Layer"
        Parser[ToolCallParser]
        Strategy1[OpenAI Parser]
        Strategy2[Anthropic Parser]
        Strategy3[Qwen Parser]
        Strategy4[Generic Parser]
        Fallback[Fallback Strategies]
    end

    subgraph "Execution Layer"
        Executor[Tool Executor]
        Registry[Tool Registry]
        Results[Tool Results]
        Metrics[Execution Metrics]
    end

    ToolDef --> Handler
    Handler --> Native
    Handler --> Prompted
    Handler --> Formatter

    Handler --> Parser
    Parser --> Strategy1
    Parser --> Strategy2
    Parser --> Strategy3
    Parser --> Strategy4
    Parser --> Fallback

    Parser --> Executor
    Executor --> Registry
    Executor --> Results
    Executor --> Metrics
```

## Event System

```mermaid
graph LR
    subgraph "Event Flow"
        Start[Tool Call Detected] --> Before[BEFORE_TOOL_EXECUTION]
        Before --> Check{Event Prevented?}
        Check -->|No| Execute[Execute Tools]
        Check -->|Yes| Skip[Skip Execution]
        Execute --> After[AFTER_TOOL_EXECUTION]
        Skip --> Return[Return Original Response]
        After --> Return
    end

    subgraph "Event Data"
        BeforeData["`tool_calls: ToolCall[]
        model: str
        can_prevent: bool`"]

        AfterData["`tool_calls: ToolCall[]
        results: ToolResult[]
        model: str`"]
    end

    subgraph "Prevention Examples"
        Security[Security Check]
        RateLimit[Rate Limiting]
        TimeLimit[Time Restrictions]
        UserCheck[User Permissions]
    end

    Before --> BeforeData
    After --> AfterData
    Before --> Security
    Before --> RateLimit
    Before --> TimeLimit
    Before --> UserCheck
```

## Provider Tool Support Matrix

```mermaid
graph TB
    subgraph "OpenAI Provider"
        OAI_Native[Native JSON Tools ✓]
        OAI_Stream[Streaming + Tools ✓]
        OAI_Execute[Local Execution ✓]
        OAI_Events[Event System ✓]
    end

    subgraph "Anthropic Provider"
        ANT_Native[Native Tools ✓]
        ANT_Prompted[Prompted Fallback ✓]
        ANT_Stream[Streaming + Tools ✓]
        ANT_Execute[Local Execution ✓]
        ANT_Events[Event System ✓]
    end

    subgraph "HuggingFace Provider"
        HF_GGUF[GGUF Native ✓]
        HF_Transformers[Prompted Tools ✓]
        HF_Dual[Dual Mode ✓]
        HF_Stream[Streaming + Tools ✓]
        HF_Execute[Local Execution ✓]
        HF_Events[Event System ✓]
    end

    subgraph "MLX Provider"
        MLX_Prompted[Prompted Tools ✓]
        MLX_Qwen[Qwen Support ✓]
        MLX_Stream[Streaming + Tools ✓]
        MLX_Execute[Local Execution ✓]
        MLX_Events[Event System ✓]
    end

    subgraph "LMStudio Provider"
        LMS_Prompted[Prompted Tools ✓]
        LMS_OpenAI[OpenAI Compatible ✓]
        LMS_Stream[Streaming + Tools ✓]
        LMS_Execute[Local Execution ✓]
        LMS_Events[Event System ✓]
    end

    subgraph "Ollama Provider"
        OLL_Arch[Architecture-Specific ✓]
        OLL_Robust[Robust Parsing ✓]
        OLL_Stream[Streaming + Tools ✓]
        OLL_Execute[Local Execution ✓]
        OLL_Events[Event System ✓]
    end
```

## Data Flow: Non-Streaming vs Streaming

### Non-Streaming Flow

```mermaid
sequenceDiagram
    participant App
    participant Provider
    participant LLM
    participant Tools

    App->>Provider: generate(prompt, tools, stream=False)
    Provider->>LLM: send_complete_request()
    LLM-->>Provider: complete_response
    Provider->>Tools: parse_and_execute()
    Tools-->>Provider: tool_results
    Provider->>Provider: merge_response()
    Provider-->>App: final_response

    Note over App,Provider: ✓ Complete response available immediately<br/>✓ Easy to process entire response<br/>✓ All tool calls visible at once
```

### Streaming Flow

```mermaid
sequenceDiagram
    participant App
    participant Provider
    participant LLM
    participant Tools

    App->>Provider: generate(prompt, tools, stream=True)
    loop Streaming Chunks
        Provider->>LLM: stream_request()
        LLM-->>Provider: chunk
        Provider->>Provider: collect_content()
        Provider-->>App: yield chunk
    end
    Provider->>Tools: parse_collected_content()
    Tools-->>Provider: tool_results
    Provider-->>App: yield tool_results

    Note over App,Provider: ✓ Real-time response display<br/>✓ Lower perceived latency<br/>✓ Progressive rendering possible
```

## Component Interaction Diagram

```mermaid
graph TB
    subgraph "Application Layer"
        UserApp[User Application]
        Examples[Example Scripts]
    end

    subgraph "Core AbstractLLM"
        Interface[AbstractLLM Interface]
        Session[Session Management]
        Types[Type System]
    end

    subgraph "Provider Layer"
        BaseProvider[Base Provider]
        Providers[Specific Providers]
        Capabilities[Model Capabilities]
    end

    subgraph "Tool System"
        ToolHandler[Universal Tool Handler]
        ToolParser[Tool Call Parser]
        ToolExecutor[Tool Executor]
        ToolRegistry[Tool Registry]
    end

    subgraph "Event System"
        EventEmitter[Event Emitter]
        EventTypes[Event Types]
        Prevention[Prevention System]
    end

    subgraph "Architecture System"
        ArchDetection[Architecture Detection]
        ModelAssets[Model Assets JSON]
        FormatHandlers[Format Handlers]
    end

    UserApp --> Interface
    Examples --> Interface
    Interface --> Session
    Interface --> Types

    Session --> BaseProvider
    BaseProvider --> Providers
    BaseProvider --> Capabilities

    Providers --> ToolHandler
    ToolHandler --> ToolParser
    ToolHandler --> ToolExecutor
    ToolExecutor --> ToolRegistry

    Providers --> EventEmitter
    EventEmitter --> EventTypes
    EventEmitter --> Prevention

    BaseProvider --> ArchDetection
    ArchDetection --> ModelAssets
    ArchDetection --> FormatHandlers

    ToolHandler --> ArchDetection
```

## Tool Parsing Strategies

```mermaid
graph TB
    subgraph "Robust Parsing Pipeline"
        Input[LLM Response Text]

        subgraph "Native Parsing"
            OpenAIFormat[OpenAI JSON Format]
            AnthropicFormat[Anthropic XML Format]
            GGUFFormat[GGUF Function Calling]
        end

        subgraph "Prompted Parsing"
            QwenStyle[Qwen <|tool_call|> Format]
            LlamaStyle[Llama JSON Format]
            MistralStyle[Mistral Function Format]
            GenericJSON[Generic JSON Detection]
            XMLFormat[XML Tool Tags]
        end

        subgraph "Fallback Strategies"
            PatternMatch[Pattern Matching]
            BraceCount[Brace Counting]
            OverlapDetect[Overlap Detection]
            ErrorRecovery[Error Recovery]
        end

        subgraph "Output"
            ToolCalls[Parsed Tool Calls]
            ValidationErrors[Validation Errors]
        end
    end

    Input --> OpenAIFormat
    Input --> AnthropicFormat
    Input --> GGUFFormat
    Input --> QwenStyle
    Input --> LlamaStyle
    Input --> MistralStyle
    Input --> GenericJSON
    Input --> XMLFormat

    OpenAIFormat --> PatternMatch
    AnthropicFormat --> PatternMatch
    QwenStyle --> BraceCount
    LlamaStyle --> OverlapDetect
    GenericJSON --> ErrorRecovery

    PatternMatch --> ToolCalls
    BraceCount --> ToolCalls
    OverlapDetect --> ToolCalls
    ErrorRecovery --> ToolCalls

    PatternMatch --> ValidationErrors
    BraceCount --> ValidationErrors
    OverlapDetect --> ValidationErrors
    ErrorRecovery --> ValidationErrors
```

## Model Capabilities Integration

```mermaid
graph LR
    subgraph "Model Assets"
        JSON1[gpt-models.json]
        JSON2[claude-models.json]
        JSON3[ollama-models.json]
        JSON4[huggingface-models.json]
    end

    subgraph "Capability Detection"
        Detection[get_model_capabilities()]
        ContextLength[context_length]
        MaxOutput[max_output_tokens]
        ToolSupport[supports_tools]
        Vision[supports_vision]
    end

    subgraph "Provider Configuration"
        BaseProvider[Base Provider Init]
        DefaultTokens[Default Token Limits]
        ToolStrategy[Tool Support Strategy]
        Features[Feature Enablement]
    end

    JSON1 --> Detection
    JSON2 --> Detection
    JSON3 --> Detection
    JSON4 --> Detection

    Detection --> ContextLength
    Detection --> MaxOutput
    Detection --> ToolSupport
    Detection --> Vision

    ContextLength --> BaseProvider
    MaxOutput --> BaseProvider
    ToolSupport --> BaseProvider
    Vision --> BaseProvider

    BaseProvider --> DefaultTokens
    BaseProvider --> ToolStrategy
    BaseProvider --> Features
```

## Performance and Monitoring

```mermaid
graph TB
    subgraph "Performance Metrics"
        ToolMetrics[Tool Execution Metrics]
        ResponseTime[Response Time]
        SuccessRate[Success Rate]
        TokenUsage[Token Usage]
    end

    subgraph "Monitoring Events"
        BeforeEvent[BEFORE_TOOL_EXECUTION]
        AfterEvent[AFTER_TOOL_EXECUTION]
        ErrorEvent[ERROR_OCCURRED]
    end

    subgraph "Optimization Strategies"
        BatchTools[Batch Tool Execution]
        CachingResults[Result Caching]
        StreamingOpt[Streaming Optimization]
        ProviderSelection[Provider Selection]
    end

    subgraph "Observability"
        Logging[Structured Logging]
        Telemetry[Telemetry Data]
        Analytics[Usage Analytics]
        Alerts[Error Alerts]
    end

    BeforeEvent --> ToolMetrics
    AfterEvent --> ToolMetrics
    ErrorEvent --> ToolMetrics

    ToolMetrics --> ResponseTime
    ToolMetrics --> SuccessRate
    ToolMetrics --> TokenUsage

    ToolMetrics --> BatchTools
    ToolMetrics --> CachingResults
    ToolMetrics --> StreamingOpt
    ToolMetrics --> ProviderSelection

    ToolMetrics --> Logging
    ToolMetrics --> Telemetry
    ToolMetrics --> Analytics
    ToolMetrics --> Alerts
```

## Security and Control

```mermaid
graph TB
    subgraph "Security Layers"
        InputValidation[Input Validation]
        ToolWhitelist[Tool Whitelist]
        PermissionCheck[Permission Checking]
        RateLimiting[Rate Limiting]
    end

    subgraph "Event-Based Control"
        BeforeHook[Before Execution Hook]
        Prevention[Event Prevention]
        ConditionalLogic[Conditional Logic]
        SecurityGates[Security Gates]
    end

    subgraph "Monitoring & Logging"
        AuditLog[Audit Logging]
        SecurityEvents[Security Events]
        ThreatDetection[Threat Detection]
        AlertSystem[Alert System]
    end

    InputValidation --> BeforeHook
    ToolWhitelist --> BeforeHook
    PermissionCheck --> BeforeHook
    RateLimiting --> BeforeHook

    BeforeHook --> Prevention
    BeforeHook --> ConditionalLogic
    BeforeHook --> SecurityGates

    Prevention --> AuditLog
    ConditionalLogic --> SecurityEvents
    SecurityGates --> ThreatDetection

    AuditLog --> AlertSystem
    SecurityEvents --> AlertSystem
    ThreatDetection --> AlertSystem
```

---

## Diagram Usage Notes

1. **Mermaid Rendering**: These diagrams use Mermaid syntax and can be rendered in:
   - GitHub README files
   - GitLab documentation
   - Notion pages
   - VS Code with Mermaid extension
   - Online Mermaid editors

2. **Interactive Elements**: In supported viewers, diagram elements may be clickable or hoverable for additional information.

3. **Maintenance**: Keep diagrams updated when architecture changes. Consider generating some diagrams programmatically from code structure.

4. **Accessibility**: Each diagram includes descriptive text and can be supplemented with detailed explanations for screen readers.

For implementation details, see [Architecture Documentation](architecture.md).