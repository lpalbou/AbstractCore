# AbstractLLM Architecture

## Overview

AbstractLLM provides a unified interface to all major LLM providers with essential infrastructure for building LLM-powered applications. This document describes the core architecture of the refactored AbstractLLM package.

## Design Principles

1. **Provider Abstraction**: Single interface for all LLM providers
2. **Minimal Core**: Keep the core lean (~8,000 lines)
3. **Extensibility**: Event-driven architecture for extensions
4. **Type Safety**: Strong typing throughout the codebase
5. **No Vendor Lock-in**: Easy to switch between providers

## Core Components

### 1. Provider Interface (`core/interface.py`)

The `AbstractLLMInterface` defines the contract all providers must implement:

```python
class AbstractLLMInterface(ABC):
    @abstractmethod
    def generate(...) -> GenerateResponse:
        """Generate response from LLM"""

    @abstractmethod
    def get_capabilities() -> List[str]:
        """Get provider capabilities"""
```

### 2. Providers (`providers/`)

Each provider implements the interface with provider-specific logic:

- **OpenAI Provider**: Native OpenAI API support with tools
- **Anthropic Provider**: Claude models with XML tool format
- **Ollama Provider**: Local models via Ollama server
- **MLX Provider**: Apple Silicon optimized models
- **LMStudio Provider**: OpenAI-compatible local models
- **Mock Provider**: For testing without API calls

### 3. Session Management (`core/session.py`)

`BasicSession` provides conversation management:

- Message history tracking
- System prompt handling
- Conversation persistence
- Context management

### 4. Tool System (`tools/`)

Universal tool execution system with comprehensive provider support:

#### Core Components
- **`UniversalToolHandler`**: Central orchestrator for all tool operations
- **`ToolDefinition`**: Standard format for defining tools across providers
- **`ToolCall`**: Represents tool invocations with arguments and metadata
- **`ToolResult`**: Handles execution results, errors, and timing
- **`ToolCallParser`**: Robust parsing for various LLM response formats

#### Tool Execution Architecture
The tool system follows the **Execute Locally** principle: regardless of provider API capabilities, all tool execution happens within AbstractLLM Core for consistency and control.

#### Supported Tool Formats

**Native Tool Support**:
- **OpenAI**: JSON function calling format
- **Anthropic**: Native tool format with XML content
- **HuggingFace GGUF**: OpenAI-compatible for chatml-function-calling models

**Prompted Tool Support**:
- **All Providers**: Universal prompted format for models without native support
- **Architecture-Specific**: Qwen, Llama, Mistral, and 80+ architectures
- **Fallback Parsing**: Multiple robust parsing strategies

#### Tool Execution Flow

1. **Tool Preparation**: Format tools for provider (native or prompted)
2. **Response Generation**: LLM generates response with tool calls
3. **Tool Detection**: Parse tool calls from response using robust parsing
4. **Event Emission**: `BEFORE_TOOL_EXECUTION` with prevention capability
5. **Tool Execution**: Execute tools locally with error handling
6. **Result Integration**: Append tool results to response
7. **Event Emission**: `AFTER_TOOL_EXECUTION` with results and metrics

#### Event-Driven Tool System

```python
# Prevention Example
def prevent_dangerous_tools(event):
    for call in event.data['tool_calls']:
        if call.name in ['delete_file', 'system_command']:
            event.prevent()

llm.add_event_listener(EventType.BEFORE_TOOL_EXECUTION, prevent_dangerous_tools)
```

#### Streaming Tool Support

All providers support streaming with tool execution:
- **Real-time Content**: Stream response content as generated
- **Tool Collection**: Collect tool calls during streaming
- **End-of-Stream Execution**: Execute tools when streaming completes
- **Result Streaming**: Stream tool results as final chunks

### 5. Type System (`core/types.py`)

Strong typing for all components:

- `Message`: Conversation messages
- `GenerateResponse`: LLM responses
- `ToolCall`: Tool invocations
- Enums for roles, parameters, capabilities

## Provider-Specific Handling

### Tool Calling

Comprehensive tool support across all providers with universal execution:

**OpenAI Provider**:
- **Native Format**: JSON function calling format
- **Tool Execution**: Local execution with event emission
- **Streaming Support**: Tool execution at end of stream
- **Prevention**: Event-based tool execution prevention

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "parameters": {"location": "Paris", "unit": "celsius"}
  }
}
```

**Anthropic Provider**:
- **Native Format**: Anthropic tool format with input_schema
- **Prompted Fallback**: Universal prompted format
- **Tool Execution**: Local execution with robust parsing
- **Streaming Support**: Complete streaming + tool execution

```json
{
  "name": "get_weather",
  "description": "Get weather information",
  "input_schema": {"type": "object", "properties": {...}}
}
```

**HuggingFace Provider**:
- **GGUF Native**: OpenAI-compatible for function-calling models
- **Transformers Prompted**: Prompted format with tool parsing
- **Dual Mode**: Automatic selection based on model capabilities
- **Robust Parsing**: Multiple fallback strategies

**MLX Provider**:
- **Prompted Tools**: Universal prompted format
- **Architecture-Aware**: Qwen-specific chat template support
- **Local Execution**: Complete tool execution pipeline

**LMStudio Provider**:
- **OpenAI-Compatible**: Standard format for compatible models
- **Prompted Fallback**: Universal prompted format
- **Tool Execution**: Full execution pipeline

**Ollama Provider** (Architecture-specific):
- **Qwen**: `<|tool_call|>{"name": "func", "arguments": {...}}`
- **Llama**: Various JSON and XML formats
- **Mistral**: Function calling syntax
- **Universal**: Prompted format for all architectures
- **Robust Parsing**: Multiple strategies with overlap detection

### Media Handling

Provider-specific image/file processing:

- **OpenAI**: Base64 encoded images in messages
- **Anthropic**: Native image support in content blocks
- **Local Models**: Depends on architecture capabilities

## Event System

Extensible event system for monitoring, plugins, and tool control:

### Core Events

```python
# Generation Events
- BEFORE_GENERATE: Before LLM generation starts
- AFTER_GENERATE: After LLM generation completes
- ERROR_OCCURRED: When errors occur

# Tool Events (New)
- BEFORE_TOOL_EXECUTION: Before tools execute (with prevention)
- AFTER_TOOL_EXECUTION: After tools execute (with results)

# Provider Events
- PROVIDER_CREATED: When provider is initialized
- MODEL_LOADED: When model is loaded
```

### Tool Event System

The tool event system provides comprehensive control and monitoring:

**Before Tool Execution**:
```python
def tool_security_handler(event):
    tool_calls = event.data['tool_calls']
    model = event.data['model']
    can_prevent = event.data['can_prevent']

    # Security check
    for call in tool_calls:
        if call.name in DANGEROUS_TOOLS:
            event.prevent()  # Prevent execution
            break

llm.add_event_listener(EventType.BEFORE_TOOL_EXECUTION, tool_security_handler)
```

**After Tool Execution**:
```python
def tool_metrics_handler(event):
    tool_calls = event.data['tool_calls']
    results = event.data['results']
    model = event.data['model']

    # Log metrics
    for call, result in zip(tool_calls, results):
        log_tool_usage(call.name, result.success, result.execution_time)

llm.add_event_listener(EventType.AFTER_TOOL_EXECUTION, tool_metrics_handler)
```

### Event Prevention

Events can prevent default behavior:
- **Tool Execution Prevention**: Stop tools from executing
- **Conditional Logic**: Prevent based on context, time, user, etc.
- **Security Gates**: Block dangerous operations
- **Rate Limiting**: Prevent too many tool calls

## Architecture Detection

Automatic detection of 80+ model architectures with model capabilities:

```python
def detect_architecture(model_name: str) -> Architecture:
    # Detect from model name patterns
    # Return architecture-specific configuration

def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    # Return model-specific capabilities from JSON assets
    # Including context_length, max_output_tokens, tool_support
```

### Model Capabilities Integration

The system now uses JSON asset files to provide model-specific defaults:

**Automatic Parameter Selection**:
- **Context Length**: Auto-detected from model capabilities
- **Max Output Tokens**: Model-specific defaults
- **Tool Support**: Automatic native vs prompted detection
- **Provider-Specific Features**: Streaming, vision, function calling

**Asset-Driven Configuration**:
```json
{
  "model_name": "gpt-4",
  "context_length": 8192,
  "max_output_tokens": 4096,
  "supports_tools": true,
  "supports_vision": true,
  "architecture": "gpt-4"
}
```

**BaseProvider Integration**:
All providers now automatically use model capabilities for:
- Setting appropriate context windows
- Configuring output token limits
- Determining tool support approach
- Optimizing provider-specific features

## Session Lifecycle

1. **Creation**: Initialize with provider and system prompt
2. **Generation**: Send prompts and receive responses
3. **Context Management**: Maintain conversation history
4. **Persistence**: Save/load sessions to disk
5. **Cleanup**: Clear history when needed

## Error Handling

Consistent error handling across providers:

```python
try:
    response = provider.generate(prompt)
except ProviderAPIError as e:
    # Handle API errors
except RateLimitError as e:
    # Handle rate limits
except AuthenticationError as e:
    # Handle auth issues
```

## Performance Considerations

- **Lazy Loading**: Providers loaded only when needed
- **Connection Pooling**: Reuse HTTP connections
- **Streaming Support**: Stream responses for better UX
- **Token Counting**: Track usage across providers
- **Caching**: Optional response caching

## Future Extensions

The architecture supports future extensions:

- **AbstractMemory**: Temporal knowledge graphs
- **AbstractAgent**: Agent orchestration layer
- **AbstractSwarm**: Multi-agent systems
- **Custom Providers**: Easy to add new providers

## Testing Strategy

- **Unit Tests**: Test each component in isolation
- **Integration Tests**: Test provider interactions
- **No Mocking**: Test against real implementations
- **Multiple Providers**: Test same functionality across providers