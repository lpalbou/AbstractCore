# API Reference

## Core Classes

### AbstractLLMInterface

Base interface for all LLM providers.

```python
class AbstractLLMInterface(ABC):
    def __init__(self, model: str, **kwargs)
    def generate(prompt: str, messages: List[Dict], system_prompt: str,
                tools: List[Dict], stream: bool, **kwargs) -> GenerateResponse
    def get_capabilities() -> List[str]
    def validate_config() -> bool
    def get_token_limit() -> Optional[int]
```

### BasicSession

Manages conversation context and history.

```python
class BasicSession:
    def __init__(provider: AbstractLLMInterface, system_prompt: str)
    def add_message(role: str, content: str) -> Message
    def get_messages() -> List[Message]
    def get_history(include_system: bool) -> List[Dict]
    def clear_history(keep_system: bool)
    def generate(prompt: str, **kwargs) -> GenerateResponse
    def save(filepath: str)
    def load(filepath: str) -> BasicSession
```

### GenerateResponse

Response from LLM generation.

```python
@dataclass
class GenerateResponse:
    content: Optional[str]
    raw_response: Any
    model: Optional[str]
    finish_reason: Optional[str]
    usage: Optional[Dict[str, int]]
    tool_calls: Optional[List[Dict]]
    metadata: Optional[Dict]

    def has_tool_calls() -> bool
    def get_tools_executed() -> List[str]
    def get_summary() -> str
```

### Message

Represents a conversation message.

```python
@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[datetime]
    name: Optional[str]
    metadata: Optional[Dict]

    def to_dict() -> Dict
    def from_dict(data: Dict) -> Message
```

### ToolDefinition

Defines a tool that can be called by the LLM.

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable]

    def from_function(func: Callable) -> ToolDefinition
    def to_dict() -> Dict
```

## Factory Functions

### create_llm

Creates an LLM provider instance.

```python
def create_llm(
    provider: str,
    model: Optional[str] = None,
    **kwargs
) -> AbstractLLMInterface
```

**Parameters**:
- `provider`: Provider name ("openai", "anthropic", "ollama", etc.)
- `model`: Model identifier (defaults to provider's default model)
- `**kwargs`: Provider-specific configuration

**Returns**: Configured provider instance

**Example**:
```python
llm = create_llm("openai", model="gpt-4", temperature=0.5)
```

## Enums

### MessageRole

Standard message roles.

```python
class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
```

### ModelParameter

Standard model parameters.

```python
class ModelParameter(Enum):
    MODEL = "model"
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    TOP_P = "top_p"
    TOP_K = "top_k"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    SEED = "seed"
```

### ModelCapability

Model capabilities.

```python
class ModelCapability(Enum):
    CHAT = "chat"
    TOOLS = "tools"
    VISION = "vision"
    STREAMING = "streaming"
    ASYNC = "async"
    JSON_MODE = "json_mode"
```

## Provider-Specific Classes

### OpenAIProvider

```python
class OpenAIProvider(AbstractLLMInterface):
    def __init__(
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    )
```

### AnthropicProvider

```python
class AnthropicProvider(AbstractLLMInterface):
    def __init__(
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    )
```

### OllamaProvider

```python
class OllamaProvider(AbstractLLMInterface):
    def __init__(
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    )
```

### MLXProvider

```python
class MLXProvider(AbstractLLMInterface):
    def __init__(
        model: str = "mlx-community/Mistral-7B-Instruct-v0.1-4bit",
        **kwargs
    )
```

### LMStudioProvider

```python
class LMStudioProvider(AbstractLLMInterface):
    def __init__(
        model: str = "local-model",
        base_url: str = "http://localhost:1234",
        **kwargs
    )
```

## Generation Parameters

Common parameters for `generate()` method:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | Required | Input prompt |
| `messages` | List[Dict] | None | Conversation history |
| `system_prompt` | str | None | System instructions |
| `tools` | List[Dict] | None | Available tools |
| `stream` | bool | False | Stream response |
| `temperature` | float | 0.7 | Randomness (0-2) |
| `max_tokens` | int | Provider default | Max response tokens |
| `top_p` | float | 1.0 | Nucleus sampling |
| `top_k` | int | None | Top-k sampling |
| `frequency_penalty` | float | 0.0 | Reduce repetition |
| `presence_penalty` | float | 0.0 | Encourage diversity |
| `seed` | int | None | Reproducible output |

## Tool Format

Tools should follow this format:

```python
tool = {
    "name": "function_name",
    "description": "What the function does",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            },
            "param2": {
                "type": "integer",
                "description": "Another parameter"
            }
        },
        "required": ["param1"]
    }
}
```

## Usage Examples

### Basic Generation

```python
from abstractllm import create_llm

llm = create_llm("openai")
response = llm.generate("Explain quantum computing")
print(response.content)
```

### With Session

```python
from abstractllm import create_llm, BasicSession

llm = create_llm("anthropic")
session = BasicSession(provider=llm, system_prompt="You are a tutor")

response1 = session.generate("What is Python?")
response2 = session.generate("Give me an example")  # Has context
```

### Tool Calling

```python
tools = [{
    "name": "calculate",
    "description": "Perform calculation",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        }
    }
}]

response = llm.generate("What is 15 * 24?", tools=tools)

if response.has_tool_calls():
    for call in response.tool_calls:
        print(f"{call['name']}: {call['arguments']}")
```

### Streaming

```python
for chunk in llm.generate("Write a poem", stream=True):
    print(chunk.content, end="")
```

### Error Handling

```python
try:
    response = llm.generate("Hello")
except Exception as e:
    print(f"Error: {e}")
```