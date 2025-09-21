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

Universal tool abstraction across providers:

- `ToolDefinition`: Define tools in a standard format
- `ToolCall`: Represent tool invocations
- `ToolResult`: Handle tool execution results
- Provider-specific formatting (OpenAI JSON, Anthropic XML, etc.)

### 5. Type System (`core/types.py`)

Strong typing for all components:

- `Message`: Conversation messages
- `GenerateResponse`: LLM responses
- `ToolCall`: Tool invocations
- Enums for roles, parameters, capabilities

## Provider-Specific Handling

### Tool Calling

Different providers handle tools differently:

**OpenAI**:
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "parameters": {...}
  }
}
```

**Anthropic**:
```xml
<tool_use>
  <name>get_weather</name>
  <input>{...}</input>
</tool_use>
```

**Ollama (Architecture-specific)**:
- Qwen: `<|tool_call|>name\n{args}<|tool_call_end|>`
- Llama: JSON format
- Mistral: Function calling syntax

### Media Handling

Provider-specific image/file processing:

- **OpenAI**: Base64 encoded images in messages
- **Anthropic**: Native image support in content blocks
- **Local Models**: Depends on architecture capabilities

## Event System

Extensible event system for monitoring and plugins:

```python
# Event types
- BEFORE_GENERATE
- AFTER_GENERATE
- TOOL_CALLED
- ERROR_OCCURRED

# Usage
provider.on('BEFORE_GENERATE', handler)
```

## Architecture Detection

Automatic detection of 80+ model architectures:

```python
def detect_architecture(model_name: str) -> Architecture:
    # Detect from model name patterns
    # Return architecture-specific configuration
```

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