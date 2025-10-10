# Architecture & Model Detection System

## Overview

The AbstractLLM Core architecture and model detection system automatically identifies the correct communication protocols and capabilities for different LLM models. This ensures optimal compatibility and performance across all supported model families.

## Quick Start

```python
from abstractllm.architectures.detection import detect_architecture, get_model_capabilities
from abstractllm.tools.handler import UniversalToolHandler
from abstractllm.structured.handler import StructuredOutputHandler

# Detect model architecture and capabilities
model_name = "qwen3-4b-instruct"
architecture = detect_architecture(model_name)  # Returns: "qwen3"
capabilities = get_model_capabilities(model_name)

# Create handlers with automatic architecture detection
tool_handler = UniversalToolHandler(model_name)
structured_handler = StructuredOutputHandler()

# Use with any model - architecture detection is automatic
llm = create_llm("ollama", model=model_name)
```

## How It Works

### 1. Architecture Detection

**Purpose**: Identifies the architectural family of a model based on its name.

**Process**:
1. **Pattern Matching**: Check model name against architecture patterns
2. **Exact Match**: Look for exact model name in capabilities  
3. **Partial Match**: Try partial name matching
4. **Architecture Fallback**: Use architecture-based defaults
5. **Generic Fallback**: Use generic defaults if no match found

**Example**:
```python
# Different models, same architecture family
detect_architecture("qwen3-4b-instruct")     # Returns: "qwen3"
detect_architecture("qwen3:30b-a3b")         # Returns: "qwen3_moe" 
detect_architecture("qwen/qwen3-next-80b")   # Returns: "qwen3_next"
```

### 2. Capability Detection

**Purpose**: Determines what a model can do (tools, structured output, context limits, etc.).

**Key Capabilities**:
- **Tool Support**: `native`, `prompted`, or `none`
- **Structured Output**: `native`, `prompted`, or `none`
- **Context Length**: Maximum input tokens
- **Multimodal**: Vision, audio, video support
- **Special Features**: Thinking support, FIM, etc.

**Example**:
```python
capabilities = get_model_capabilities("qwen3-4b-instruct")
print(capabilities["tool_support"])        # "prompted"
print(capabilities["context_length"])      # 32768
print(capabilities["vision_support"])      # False
```

### 3. Automatic Integration

**Tool Handler**: Automatically chooses the right tool strategy
```python
handler = UniversalToolHandler("qwen3-4b-instruct")
# Automatically uses prompted tool calling for Qwen3
# Automatically uses native tool calling for Gemma3
```

**Structured Output**: Automatically chooses the right output strategy
```python
handler = StructuredOutputHandler()
# Automatically uses prompted for Qwen3
# Automatically uses native for Gemma3
```

## Supported Architectures

### Qwen3 Family
- **Qwen3**: `qwen3-4b-instruct`, `qwen3:4b-instruct`
- **Qwen3-MoE**: `qwen3-30b-a3b`, `qwen3-coder:30b`
- **Qwen3-Next**: `qwen3-next-80b`
- **Qwen3-VL**: `qwen3-vl-*`

**Message Format**: ChatML with `<|im_start|>` tags
**Tool Format**: Special token with `<|tool_call|>` markers

### LLaMA Family
- **LLaMA3**: `llama-3-*`, `llama3-*`
- **LLaMA3.1**: `llama-3.1-*`
- **LLaMA3.2**: `llama-3.2-*`
- **LLaMA4**: `llama-4-*`

**Message Format**: Header format with `<|start_header_id|>` tags
**Tool Format**: Function call with `<function_call>` tags

### Gemma Family
- **Gemma3**: `gemma3-*`, `gemma3:4b`
- **Gemma3n**: `gemma3n:*`

**Message Format**: Basic Human/Assistant format
**Tool Format**: Native API tool calling

### Mistral Family
- **Mistral**: `mistral-*`
- **Mixtral**: `mixtral-*`
- **Mistral Large**: `mistral-large-*`
- **Codestral**: `codestral-*`

**Message Format**: Instruction format with `[INST]` tags
**Tool Format**: Native API tool calling

### Other Architectures
- **GLM4**: `glm-4-*`, `chatglm4-*`
- **Granite**: `granite-*`, `granite3.3:*`
- **Claude**: `claude-*`
- **GPT**: `gpt-*`

## Real-World Usage

### Tool Calling Example

```python
from abstractllm import create_llm
from abstractllm.tools.handler import UniversalToolHandler
from abstractllm.tools.core import ToolDefinition

# Define a tool
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

tool_def = ToolDefinition(
    name="get_weather",
    description="Get weather for a location",
    parameters={"location": {"type": "string", "description": "The city"}}
)

# Create LLM and tool handler
llm = create_llm("ollama", model="qwen3-4b-instruct")
handler = UniversalToolHandler("qwen3-4b-instruct")

# Format tools for the model
tool_prompt = handler.format_tools_prompt([tool_def])
system_prompt = f"You are a helpful assistant.\n\n{tool_prompt}"

# Generate with tools
response = llm.generate(
    "What's the weather like in Tokyo?",
    system_prompt=system_prompt
)

# Parse tool calls
parsed = handler.parse_response(response.content, mode="prompted")
for tool_call in parsed.tool_calls:
    print(f"Tool: {tool_call.name}({tool_call.arguments})")
    # Execute the tool...
```

### Structured Output Example

```python
from abstractllm.structured.handler import StructuredOutputHandler
from pydantic import BaseModel

class PersonInfo(BaseModel):
    name: str
    age: int
    city: str
    occupation: str

# Create handler
handler = StructuredOutputHandler()

# Generate structured output
result = handler.generate_structured(
    provider=llm,
    prompt="Create a profile for Alice, 28, from Tokyo, software engineer",
    response_model=PersonInfo
)

print(result)  # PersonInfo(name='Alice', age=28, city='Tokyo', occupation='software engineer')
```

### Message Formatting Example

```python
from abstractllm.architectures.detection import format_messages

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

# Different architectures use different formats
qwen3_formatted = format_messages(messages, "qwen3")
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Hello!<|im_end|>

llama3_formatted = format_messages(messages, "llama3")
# <|start_header_id|>system<|end_header_id|>
# You are a helpful assistant.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Hello!<|eot_id|>
```

## Architecture-Specific Details

### Qwen3 Architecture

**Message Format**: ChatML
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello! How are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you!<|im_end|>
```

**Tool Format**: Special Token
```
<|tool_call|>
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
</|tool_call|>
```

### LLaMA3 Architecture

**Message Format**: Header Format
```
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello! How are you?<|eot_id|>
```

**Tool Format**: Function Call
```
<function_call>
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
</function_call>
```

### Gemma3 Architecture

**Message Format**: Basic Human/Assistant
```
You are a helpful assistant.

Human: Hello! How are you?
Assistant: I'm doing well, thank you!
```

**Tool Format**: Native API (handled by provider)

### Mistral Architecture

**Message Format**: Instruction Format
```
You are a helpful assistant.

[INST] Hello! How are you? [/INST]I'm doing well, thank you!
```

**Tool Format**: Native API (handled by provider)

## Configuration Files

### Architecture Formats (`abstractllm/assets/architecture_formats.json`)

Defines communication protocols for each architecture:

```json
{
  "architectures": {
    "qwen3": {
      "description": "Alibaba's Qwen3 architecture",
      "message_format": "im_start_end",
      "system_prefix": "<|im_start|>system\n",
      "system_suffix": "<|im_end|>\n",
      "user_prefix": "<|im_start|>user\n",
      "user_suffix": "<|im_end|>\n",
      "assistant_prefix": "<|im_start|>assistant\n",
      "assistant_suffix": "<|im_end|>\n",
      "tool_format": "special_token",
      "tool_prefix": "<|tool_call|>",
      "patterns": ["qwen3-*", "qwen3:*"]
    }
  }
}
```

### Model Capabilities (`abstractllm/assets/model_capabilities.json`)

Stores individual model capabilities:

```json
{
  "models": {
    "qwen3-4b-instruct": {
      "context_length": 32768,
      "max_output_tokens": 8192,
      "tool_support": "prompted",
      "structured_output": "prompted",
      "vision_support": false
    }
  },
  "default_capabilities": {
    "context_length": 16384,
    "max_output_tokens": 4096,
    "tool_support": "none",
    "structured_output": "none"
  }
}
```

## Adding New Models

### 1. Add Model Capabilities

```json
{
  "models": {
    "new-model-7b": {
      "context_length": 32768,
      "max_output_tokens": 4096,
      "tool_support": "native",
      "structured_output": "native",
      "vision_support": true
    }
  }
}
```

### 2. Update Architecture Patterns

```json
{
  "architectures": {
    "existing_arch": {
      "patterns": ["existing-*", "new-model-*"]
    }
  }
}
```

### 3. Test Detection

```python
from abstractllm.architectures.detection import detect_architecture, get_model_capabilities

# Test detection
architecture = detect_architecture("new-model-7b")
capabilities = get_model_capabilities("new-model-7b")
print(f"Architecture: {architecture}")
print(f"Capabilities: {capabilities}")
```

## Troubleshooting

### Common Issues

**Architecture not detected**:
- Check if model name matches any patterns in `architecture_formats.json`
- Add new pattern if needed

**Wrong tool format**:
- Verify architecture tool format configuration
- Check if model supports the expected format

**Parsing failures**:
- Check tool call format matches expected pattern
- Enable debug logging to see parsing details

**Capability mismatches**:
- Verify model capabilities are correctly configured
- Check if using architecture defaults vs model-specific

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now you'll see detailed detection logs
architecture = detect_architecture("model-name")
```

## Best Practices

1. **Use specific patterns** - Avoid overly broad patterns
2. **Test with real models** - Verify detection works with actual model names
3. **Be conservative with capabilities** - Underestimate rather than overestimate
4. **Handle edge cases** - Provide fallbacks for unknown models
5. **Keep patterns updated** - Update when new model variants are released

## Integration Points

- **Tool Handler**: Automatically uses architecture detection
- **Structured Output**: Automatically chooses the right strategy  
- **Providers**: Automatically format messages correctly
- **All Components**: Work together seamlessly with zero configuration