# API Reference

Complete reference for the AbstractCore API. All examples work across any provider.

## Table of Contents

- [Core Functions](#core-functions)
- [Classes](#classes)
- [Event System](#event-system)
- [Retry Configuration](#retry-configuration)
- [Embeddings](#embeddings)
- [Exceptions](#exceptions)

## Core Functions

### create_llm()

Creates an LLM provider instance.

```python
def create_llm(
    provider: str,
    model: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None,
    **kwargs
) -> AbstractLLMInterface
```

**Parameters:**
- `provider` (str): Provider name ("openai", "anthropic", "ollama", "mlx", "lmstudio", "huggingface")
- `model` (str, optional): Model name. If not provided, uses provider default
- `retry_config` (RetryConfig, optional): Custom retry configuration
- `**kwargs`: Provider-specific parameters

**Provider-specific parameters:**
- `api_key` (str): API key for cloud providers
- `base_url` (str): Custom endpoint URL
- `temperature` (float): Sampling temperature (0-2)
- `max_tokens` (int): Maximum output tokens
- `timeout` (int): Request timeout in seconds
- `top_p` (float): Nucleus sampling parameter

**Returns:** AbstractLLMInterface instance

**Example:**
```python
from abstractllm import create_llm

# Basic usage
llm = create_llm("openai", model="gpt-4o-mini")

# With configuration
llm = create_llm(
    "anthropic",
    model="claude-3-5-haiku-latest",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)

# Local provider
llm = create_llm("ollama", model="qwen2.5-coder:7b", base_url="http://localhost:11434")
```

## Classes

### AbstractLLMInterface

Base interface for all LLM providers. All providers implement this interface.

#### generate()

Generate text response from the LLM.

```python
def generate(
    self,
    prompt: str,
    messages: Optional[List[Dict]] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    response_model: Optional[BaseModel] = None,
    retry_strategy: Optional[Retry] = None,
    stream: bool = False,
    **kwargs
) -> Union[GenerateResponse, Iterator[GenerateResponse]]
```

**Parameters:**
- `prompt` (str): Text prompt to generate from
- `messages` (List[Dict], optional): Conversation messages in OpenAI format
- `system_prompt` (str, optional): System prompt to set context
- `tools` (List[Dict], optional): Tools the LLM can call
- `response_model` (BaseModel, optional): Pydantic model for structured output
- `retry_strategy` (Retry, optional): Custom retry strategy for structured output
- `stream` (bool): Enable streaming response
- `**kwargs`: Additional generation parameters

**Returns:**
- If `stream=False`: GenerateResponse
- If `stream=True`: Iterator[GenerateResponse]

**Examples:**

**Basic Generation:**
```python
response = llm.generate("What is machine learning?")
print(response.content)
```

**With System Prompt:**
```python
response = llm.generate(
    "Explain Python decorators",
    system_prompt="You are a Python expert. Always provide code examples."
)
```

**Structured Output:**
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = llm.generate(
    "Extract: John Doe is 25 years old",
    response_model=Person
)
print(f"{person.name}, age {person.age}")
```

**Tool Calling:**
```python
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"

tools = [{
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}]

response = llm.generate("What's the weather in Paris?", tools=tools)
```

**Streaming (Unified 2025 Implementation):**
```python
# Streaming works identically across ALL providers
print("AI: ", end="")
for chunk in llm.generate(
    "Create a Python function with a tool",
    stream=True,
    tools=[code_analysis_tool]
):
    # Real-time chunk processing
    print(chunk.content, end="", flush=True)

    # Real-time tool call detection and execution
    if chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            result = tool_call.execute()
            print(f"\nTool Result: {result}")
```

**Streaming Features**:
- ⚡ First chunk in <10ms
- 🔧 Unified strategy across providers
- 🛠️ Real-time tool call detection
- 📊 Mid-stream tool execution
- 💨 Zero buffering overhead
- 🚀 Supports: OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace
- 🔒 Robust error handling for malformed responses

#### get_capabilities()

Get provider capabilities.

```python
def get_capabilities(self) -> List[str]
```

**Returns:** List of capability strings

**Example:**
```python
capabilities = llm.get_capabilities()
print(capabilities)  # ['text_generation', 'tool_calling', 'streaming', 'vision']
```

#### unload()

Unload the model from memory (local providers only).

```python
def unload(self) -> None
```

For local providers (Ollama, MLX, HuggingFace, LMStudio), this explicitly frees model memory. For API providers (OpenAI, Anthropic), this is a no-op but safe to call.

**Provider-specific behavior:**
- **Ollama**: Sends `keep_alive=0` to immediately unload from server
- **MLX**: Clears model/tokenizer references and forces garbage collection
- **HuggingFace**: Closes llama.cpp resources (GGUF) or clears model references
- **LMStudio**: Closes HTTP connection (server auto-manages via TTL)
- **OpenAI/Anthropic**: No-op (safe to call)

**Example:**
```python
# Load and use a large model
llm = create_llm("ollama", model="qwen3-coder:30b")
response = llm.generate("Hello world")

# Explicitly free memory when done
llm.unload()
del llm

# Now safe to load another large model
llm2 = create_llm("mlx", model="mlx-community/Qwen3-30B-4bit")
```

**Use cases:**
- Test suites testing multiple models sequentially
- Memory-constrained environments (<32GB RAM)
- Sequential model loading in production systems

### GenerateResponse

Response object from LLM generation.

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
```

**Attributes:**
- `content` (str): Generated text content
- `raw_response` (Any): Raw provider response
- `model` (str): Model used for generation
- `finish_reason` (str): Why generation stopped ("stop", "length", "tool_calls")
- `usage` (Dict): Token usage information
- `tool_calls` (List[Dict]): Tools called by the LLM
- `metadata` (Dict): Additional metadata

**Methods:**

#### has_tool_calls()
```python
def has_tool_calls(self) -> bool
```
Returns True if the response contains tool calls.

#### get_tools_executed()
```python
def get_tools_executed(self) -> List[str]
```
Returns list of tool names that were executed.

**Example:**
```python
response = llm.generate("What's 2+2?", tools=[calculator_tool])

print(f"Content: {response.content}")
print(f"Model: {response.model}")
print(f"Tokens: {response.usage}")

if response.has_tool_calls():
    print(f"Tools used: {response.get_tools_executed()}")
```

### BasicSession

Manages conversation context and history.

```python
class BasicSession:
    def __init__(
        self,
        provider: AbstractLLMInterface,
        system_prompt: Optional[str] = None
    ):
```

**Parameters:**
- `provider` (AbstractLLMInterface): LLM provider instance
- `system_prompt` (str, optional): System prompt for the conversation

**Attributes:**
- `messages` (List[Message]): Conversation history
- `provider` (AbstractLLMInterface): LLM provider
- `system_prompt` (str): System prompt

**Methods:**

#### generate()
```python
def generate(self, prompt: str, **kwargs) -> GenerateResponse
```
Generate response and add to conversation history.

#### add_message()
```python
def add_message(self, role: str, content: str, **metadata) -> Message
```
Add message to conversation history.

#### clear_history()
```python
def clear_history(self, keep_system: bool = True) -> None
```
Clear conversation history, optionally keeping system prompt.

#### save()
```python
def save(self, filepath: Path) -> None
```
Save session to JSON file.

#### load()
```python
@classmethod
def load(cls, filepath: Path, provider: AbstractLLMInterface) -> "BasicSession"
```
Load session from JSON file.

**Example:**
```python
from abstractllm import create_llm, BasicSession

llm = create_llm("openai", model="gpt-4o-mini")
session = BasicSession(
    provider=llm,
    system_prompt="You are a helpful coding tutor."
)

# Multi-turn conversation
response1 = session.generate("What are Python decorators?")
response2 = session.generate("Show me an example")

print(f"Conversation has {len(session.messages)} messages")

# Save session
session.save(Path("conversation.json"))

# Load later
loaded_session = BasicSession.load(Path("conversation.json"), llm)
```

### Message

Represents a conversation message.

```python
@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[datetime] = None
    name: Optional[str] = None
    metadata: Optional[Dict] = None
```

**Methods:**

#### to_dict()
```python
def to_dict(self) -> Dict
```
Convert message to dictionary.

#### from_dict()
```python
@classmethod
def from_dict(cls, data: Dict) -> "Message"
```
Create message from dictionary.

## Event System

### EventType

Available event types for monitoring.

```python
class EventType(Enum):
    # Generation events
    BEFORE_GENERATE = "before_generate"
    AFTER_GENERATE = "after_generate"

    # Tool events
    BEFORE_TOOL_EXECUTION = "before_tool_execution"
    AFTER_TOOL_EXECUTION = "after_tool_execution"

    # Structured output events
    STRUCTURED_OUTPUT_REQUESTED = "structured_output_requested"
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_SUCCEEDED = "validation_succeeded"

    # Retry events
    RETRY_ATTEMPTED = "retry_attempted"
    RETRY_EXHAUSTED = "retry_exhausted"

    # Session events
    SESSION_CREATED = "session_created"
    SESSION_SAVED = "session_saved"
    SESSION_LOADED = "session_loaded"
    SESSION_CLEARED = "session_cleared"

    # System events
    PROVIDER_CREATED = "provider_created"
    ERROR_OCCURRED = "error_occurred"
```

### on_global()

Register global event handler.

```python
def on_global(event_type: EventType, handler: Callable[[Event], None]) -> None
```

**Parameters:**
- `event_type` (EventType): Event type to listen for
- `handler` (Callable): Function to call when event occurs

**Example:**
```python
from abstractllm.events import EventType, on_global

def cost_monitor(event):
    if hasattr(event, 'cost_usd') and event.cost_usd:
        print(f"Cost: ${event.cost_usd:.4f}")

def tool_monitor(event):
    tool_calls = event.data.get('tool_calls', [])
    for call in tool_calls:
        print(f"Tool called: {call['name']}")

# Register handlers
on_global(EventType.AFTER_GENERATE, cost_monitor)
on_global(EventType.BEFORE_TOOL_EXECUTION, tool_monitor)

# Now all LLM operations will trigger these handlers
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("Hello world")
```

### Event

Event object passed to handlers.

```python
@dataclass
class Event:
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    duration_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    model_name: Optional[str] = None
    provider_name: Optional[str] = None
```

**Methods:**

#### prevent()
```python
def prevent(self) -> None
```
Prevent default behavior (only works for preventable events like `BEFORE_TOOL_EXECUTION`).

## Retry Configuration

### RetryConfig

Configuration for provider-level retry behavior.

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    use_jitter: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 2
```

**Parameters:**
- `max_attempts` (int): Maximum retry attempts
- `initial_delay` (float): Initial delay in seconds
- `max_delay` (float): Maximum delay in seconds
- `exponential_base` (float): Base for exponential backoff
- `use_jitter` (bool): Add randomness to delays
- `failure_threshold` (int): Circuit breaker failure threshold
- `recovery_timeout` (float): Circuit breaker recovery timeout
- `half_open_max_calls` (int): Max calls in half-open state

**Example:**
```python
from abstractllm import create_llm
from abstractllm.core.retry import RetryConfig

config = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    use_jitter=True,
    failure_threshold=3
)

llm = create_llm("openai", model="gpt-4o-mini", retry_config=config)
```

### FeedbackRetry

Retry strategy for structured output validation failures.

```python
class FeedbackRetry:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
```

**Example:**
```python
from abstractllm.structured import FeedbackRetry
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

custom_retry = FeedbackRetry(max_attempts=5)

user = llm.generate(
    "Extract user: John Doe, 25",
    response_model=User,
    retry_strategy=custom_retry
)
```

## Embeddings

### EmbeddingManager

Manages text embeddings using SOTA models.

```python
class EmbeddingManager:
    def __init__(
        self,
        model: str = "embeddinggemma",
        backend: str = "auto",
        output_dims: Optional[int] = None,
        cache_size: int = 1000,
        cache_dir: Optional[str] = None
    ):
```

**Parameters:**
- `model` (str): Model name ("embeddinggemma", "granite", "stella-400m")
- `backend` (str): Backend ("auto", "pytorch", "onnx")
- `output_dims` (int, optional): Truncate output dimensions
- `cache_size` (int): Memory cache size
- `cache_dir` (str, optional): Disk cache directory

**Methods:**

#### embed()
```python
def embed(self, text: str) -> List[float]
```
Generate embedding for single text.

#### embed_batch()
```python
def embed_batch(self, texts: List[str]) -> List[List[float]]
```
Generate embeddings for multiple texts (more efficient).

#### compute_similarity()
```python
def compute_similarity(self, text1: str, text2: str) -> float
```
Compute cosine similarity between two texts.

**Example:**
```python
from abstractllm.embeddings import EmbeddingManager

embedder = EmbeddingManager(model="embeddinggemma")

# Single embedding
embedding = embedder.embed("Hello world")
print(f"Embedding dimension: {len(embedding)}")

# Batch embeddings
embeddings = embedder.embed_batch(["Hello", "World", "AI"])

# Similarity
similarity = embedder.compute_similarity("cat", "kitten")
print(f"Similarity: {similarity:.3f}")
```

## Exceptions

### Base Exceptions

#### AbstractLLMError
```python
class AbstractLLMError(Exception):
    """Base exception for AbstractLLM."""
```

#### ProviderAPIError
```python
class ProviderAPIError(AbstractLLMError):
    """Provider API error."""
```

#### ModelNotFoundError
```python
class ModelNotFoundError(AbstractLLMError):
    """Model not found error."""
```

#### AuthenticationError
```python
class AuthenticationError(ProviderAPIError):
    """Authentication error."""
```

#### RateLimitError
```python
class RateLimitError(ProviderAPIError):
    """Rate limit error."""
```

### Usage

```python
from abstractllm.exceptions import ProviderAPIError, RateLimitError

try:
    response = llm.generate("Hello world")
except RateLimitError:
    print("Rate limited, wait and retry")
except ProviderAPIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Usage Patterns

### Custom Provider Configuration

```python
# Provider with all options
llm = create_llm(
    provider="openai",
    model="gpt-4o-mini",
    api_key="your-key",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    timeout=30,
    retry_config=RetryConfig(max_attempts=5)
)
```

### Multi-Provider Setup

```python
providers = {
    "fast": create_llm("openai", model="gpt-4o-mini"),
    "smart": create_llm("openai", model="gpt-4o"),
    "long_context": create_llm("anthropic", model="claude-3-5-sonnet-latest"),
    "local": create_llm("ollama", model="qwen2.5-coder:7b")
}

def route_request(prompt, task_type="general"):
    if task_type == "simple":
        return providers["fast"].generate(prompt)
    elif task_type == "complex":
        return providers["smart"].generate(prompt)
    elif len(prompt) > 50000:
        return providers["long_context"].generate(prompt)
    else:
        return providers["local"].generate(prompt)
```

### Production Monitoring

```python
from abstractllm.events import EventType, on_global
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cost tracking
total_cost = 0.0

def production_monitor(event):
    global total_cost

    if event.type == EventType.AFTER_GENERATE:
        if event.cost_usd:
            total_cost += event.cost_usd
            logger.info(f"Request cost: ${event.cost_usd:.4f}, Total: ${total_cost:.4f}")

        if event.duration_ms and event.duration_ms > 10000:
            logger.warning(f"Slow request: {event.duration_ms:.0f}ms")

    elif event.type == EventType.ERROR_OCCURRED:
        logger.error(f"Error: {event.data.get('error')}")

    elif event.type == EventType.RETRY_ATTEMPTED:
        logger.info(f"Retrying due to: {event.data.get('error_type')}")

on_global(EventType.AFTER_GENERATE, production_monitor)
on_global(EventType.ERROR_OCCURRED, production_monitor)
on_global(EventType.RETRY_ATTEMPTED, production_monitor)
```

---

For more examples and use cases, see:
- [Getting Started](getting-started.md) - Basic setup and usage
- [Examples](examples.md) - Practical use cases
- [Providers](providers.md) - Provider-specific documentation
- [Capabilities](capabilities.md) - What AbstractCore can do