# AbstractCore

**Preview of AbstractLLM v2** - A unified interface to all LLM providers with essential infrastructure for tool calling, streaming, structured output, and model management.

> **Note**: This is a preview release published as `abstractcore`. The final version will be published as `abstractllm` v2.0. All import statements remain the same for seamless migration.

## ✨ Features

- **🔌 Universal Provider Support**: OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace
- **🛠️ Tool Calling**: Native tool calling with automatic execution
- **📊 Structured Output**: Type-safe JSON responses using Pydantic models
- **⚡ Streaming**: Real-time response streaming across all providers
- **🔔 Event System**: Comprehensive events for monitoring, debugging, and UI integration
- **🔄 Session Management**: Conversation memory and context management
- **🔄 Retry & Resilience**: Production-grade retry with exponential backoff and circuit breakers
- **🔢 Vector Embeddings**: SOTA open-source embeddings for semantic search and RAG
- **🎯 Zero Configuration**: Works out of the box with sensible defaults
- **🏗️ Production Ready**: Comprehensive error handling and telemetry

## 🚀 Quick Start

### Installation

```bash
# Install core package (preview release)
pip install abstractcore

# Install with specific providers
pip install abstractcore[openai,anthropic]  # API providers
pip install abstractcore[ollama,lmstudio]   # Local providers
pip install abstractcore[mlx]               # Apple Silicon
pip install abstractcore[embeddings]        # Vector embeddings
pip install abstractcore[all]               # Everything
```

> **Migration Note**: When AbstractLLM v2.0 is released, simply replace `abstractcore` with `abstractllm` in your installation commands. No code changes required!

### Basic Usage

```python
from abstractllm import create_llm

# Create an LLM instance
llm = create_llm("openai", model="gpt-4o-mini")

# Generate a response
response = llm.generate("What is the capital of France?")
print(response.content)
```

## 🔌 Supported Providers

### OpenAI
```python
from abstractllm import create_llm

# Standard models
llm = create_llm("openai", model="gpt-4o-mini")
llm = create_llm("openai", model="gpt-4-turbo")

# With API key (or use OPENAI_API_KEY env var)
llm = create_llm("openai", model="gpt-4o-mini", api_key="your-key")

response = llm.generate("Explain quantum computing in simple terms")
print(response.content)
```

### Anthropic
```python
from abstractllm import create_llm

# Latest models
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
llm = create_llm("anthropic", model="claude-3-5-sonnet-latest")

# With API key (or use ANTHROPIC_API_KEY env var)
llm = create_llm("anthropic", model="claude-3-5-haiku-latest", api_key="your-key")

response = llm.generate("Write a Python function to calculate fibonacci numbers")
print(response.content)
```

### Ollama (Local)
```python
from abstractllm import create_llm

# Recommended coding model
llm = create_llm("ollama", model="qwen3-coder:30b")

# Other supported models
llm = create_llm("ollama", model="qwen3:8b")
llm = create_llm("ollama", model="llama3.1:8b")

# Custom endpoint
llm = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")

response = llm.generate("Explain the difference between lists and tuples in Python")
print(response.content)
```

### LMStudio (Local)
```python
from abstractllm import create_llm

# LMStudio with specific model
llm = create_llm("lmstudio", model="qwen/qwen3-coder-30b")

# Custom endpoint
llm = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234")

response = llm.generate("Review this code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)")
print(response.content)
```

### MLX (Apple Silicon)
```python
from abstractllm import create_llm

# MLX with recommended model
llm = create_llm("mlx", model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")

# Alternative models
llm = create_llm("mlx", model="mlx-community/Qwen3-4B")

response = llm.generate("Optimize this JavaScript function for better performance")
print(response.content)
```

### HuggingFace
```python
from abstractllm import create_llm

# Using transformers models
llm = create_llm("huggingface", model="microsoft/DialoGPT-medium")

# Using GGUF models (recommended)
llm = create_llm("huggingface", model="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")

response = llm.generate("Hello! How are you today?")
print(response.content)
```

## 📊 Structured Output

Get type-safe JSON responses using Pydantic models with automatic validation and retry.

### Basic Structured Output

```python
from pydantic import BaseModel
from abstractllm import create_llm

class UserInfo(BaseModel):
    name: str
    age: int
    email: str

# Works with any provider
llm = create_llm("openai", model="gpt-4o-mini")

# Get structured response
user = llm.generate(
    "Extract user info: John Doe, 28 years old, john@example.com",
    response_model=UserInfo
)

print(f"Name: {user.name}")        # Name: John Doe
print(f"Age: {user.age}")          # Age: 28
print(f"Email: {user.email}")      # Email: john@example.com
```

### Automatic Retry on Validation Errors

AbstractLLM automatically retries with detailed error feedback when structured output validation fails:

```python
from pydantic import BaseModel, field_validator
from abstractllm import create_llm

class Product(BaseModel):
    name: str
    price: float
    category: str

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

llm = create_llm("openai", model="gpt-4o-mini")

# This will automatically retry if the LLM returns invalid data
# For example, if LLM returns: {"name": "Laptop", "price": -100, "category": "Electronics"}
# The system will retry with error feedback:
# "IMPORTANT: Your previous response had validation errors:
#  • Field 'price': Price must be positive
#  Please correct these errors and provide a valid JSON response..."

product = llm.generate(
    "Extract product info: Gaming Laptop for $1200 in Electronics category",
    response_model=Product
)

print(f"Product: {product.name} - ${product.price}")
```

### Custom Retry Configuration

Control retry behavior for structured output validation with a simple parameter:

```python
from abstractllm.structured import FeedbackRetry
from pydantic import BaseModel, field_validator
from typing import List

# For complex validation that might need multiple attempts
class StrictValidation(BaseModel):
    items: List[str]
    count: int
    total_value: float

    @field_validator('count')
    @classmethod
    def validate_count(cls, v, info):
        items = info.data.get('items', [])
        if v != len(items):
            raise ValueError(f'Count {v} does not match items length {len(items)}')
        return v

llm = create_llm("anthropic", model="claude-3-5-haiku-latest")

# Custom retry strategy with more attempts
custom_retry = FeedbackRetry(max_attempts=5)  # Try up to 5 times

result = llm.generate(
    "Create a list of 3 programming languages with their count and total rating",
    response_model=StrictValidation,
    retry_strategy=custom_retry  # Pass retry strategy directly!
)

# Or use default behavior (3 attempts)
result = llm.generate(
    "Create a list of 3 programming languages with their count and total rating",
    response_model=StrictValidation  # Uses default 3 attempts
)
```

### Complex Nested Structures

```python
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Contact(BaseModel):
    email: str
    phone: Optional[str] = None

class TeamMember(BaseModel):
    name: str
    role: str
    contact: Contact

class Project(BaseModel):
    title: str
    description: str
    priority: Priority
    team: List[TeamMember]
    estimated_hours: float

# Test with different providers
providers = [
    ("ollama", "qwen3-coder:30b"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
]

project_data = """
Project: AI Research Platform
Description: Build a comprehensive platform for AI research with ML pipeline management
Priority: High priority project
Team:
- Dr. Sarah Chen (Lead Researcher) - sarah@university.edu, 555-0123
- Mike Johnson (Engineer) - mike@company.com, 555-0456
Estimated: 120 hours
"""

for provider, model in providers:
    print(f"\n--- {provider.upper()} | {model} ---")
    llm = create_llm(provider, model=model)

    project = llm.generate(
        f"Extract project information:\n{project_data}",
        response_model=Project
    )

    print(f"Project: {project.title}")
    print(f"Priority: {project.priority}")
    print(f"Team size: {len(project.team)}")
    print(f"Lead: {project.team[0].name} ({project.team[0].role})")
```

### Validation and Retry Control

AbstractLLM provides intelligent retry mechanisms when structured output validation fails, with detailed feedback to help the LLM self-correct.

#### Basic Retry Configuration

```python
from pydantic import BaseModel, field_validator
from abstractllm import create_llm
from abstractllm.structured import StructuredOutputHandler, FeedbackRetry

class StrictUser(BaseModel):
    name: str
    age: int

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v

# Default retry (3 attempts)
llm = create_llm("ollama", model="qwen3-coder:30b")
user = llm.generate(
    "Extract: Alice, 25 years old",
    response_model=StrictUser
)

# Custom retry strategy with more attempts
handler = StructuredOutputHandler(retry_strategy=FeedbackRetry(max_attempts=5))
# Use handler directly for advanced control
```

#### Observing Retry Behavior

```python
from abstractllm.utils.structured_logging import configure_logging, get_logger
import logging

# Configure logging to see retry attempts
configure_logging(
    console_level=logging.INFO,  # See retry attempts in console
    log_dir="logs",              # Save detailed logs to file
    verbatim_enabled=True        # Capture full prompts/responses
)

class ComplexData(BaseModel):
    items: List[str]
    count: int

    @field_validator('count')
    @classmethod
    def validate_count(cls, v, values):
        items = values.get('items', [])
        if v != len(items):
            raise ValueError(f'Count {v} does not match items length {len(items)}')
        return v

# This will log each retry attempt with detailed error feedback
llm = create_llm("lmstudio", model="qwen/qwen3-coder-30b")
result = llm.generate(
    "Create a list of 3 fruits with count",
    response_model=ComplexData
)

# Check logs/verbatim_*.jsonl for complete interaction history
```

#### Custom Retry Strategies

```python
from abstractllm.structured.retry import Retry
from pydantic import ValidationError

class CustomRetry(Retry):
    """Custom retry with exponential backoff and custom error handling."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def should_retry(self, attempt: int, error: Exception) -> bool:
        if attempt >= self.max_attempts:
            return False
        return isinstance(error, (ValidationError, json.JSONDecodeError))

    def prepare_retry_prompt(self, original_prompt: str, error: Exception, attempt: int) -> str:
        delay = self.base_delay * (2 ** (attempt - 1))

        if isinstance(error, ValidationError):
            error_details = self._format_pydantic_errors(error)
            return f"""{original_prompt}

RETRY {attempt}: Your previous response had validation errors:
{error_details}

Please fix these specific issues and provide valid JSON."""

        return f"""{original_prompt}

RETRY {attempt}: Your previous response was not valid JSON.
Please provide a valid JSON object that matches the schema."""

# Use custom retry strategy
handler = StructuredOutputHandler(retry_strategy=CustomRetry(max_attempts=5))
```

#### Real-world Retry Example: Code Analysis

```python
from enum import Enum
from typing import List, Optional

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CodeIssue(BaseModel):
    line_number: int
    severity: Severity
    description: str
    suggestion: str

    @field_validator('line_number')
    @classmethod
    def validate_line_number(cls, v):
        if v < 1:
            raise ValueError('Line number must be positive')
        return v

class CodeAnalysis(BaseModel):
    total_lines: int
    issues: List[CodeIssue]
    overall_score: float

    @field_validator('overall_score')
    @classmethod
    def validate_score(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('Score must be between 0 and 10')
        return v

# Enable detailed logging to observe retry process
configure_logging(
    console_level=logging.WARNING,  # Only show warnings/errors in console
    file_level=logging.DEBUG,       # Full details in file
    log_dir="logs",
    verbatim_enabled=True
)

# Test with challenging code that might require retries
code_to_analyze = '''
def buggy_function(x):
    if x = 5:  # Bug: should be ==
        return x * 2
    return x
'''

llm = create_llm("ollama", model="qwen3-coder:30b")

# This might require retries due to complex validation rules
analysis = llm.generate(
    f"Analyze this Python code for issues:\n{code_to_analyze}",
    response_model=CodeAnalysis
)

print(f"Analysis completed with {len(analysis.issues)} issues found")
print(f"Overall score: {analysis.overall_score}/10")

# Check logs for retry attempts and validation errors
```

#### Monitoring Retry Performance

```python
from abstractllm.utils.structured_logging import capture_session
import time

# Use session capture for detailed monitoring
with capture_session("code_review_session") as session_logger:
    start_time = time.time()

    # Multiple structured operations in one session
    for i, code_snippet in enumerate(code_snippets):
        try:
            analysis = llm.generate(
                f"Analyze code snippet {i+1}:\n{code_snippet}",
                response_model=CodeAnalysis
            )
            session_logger.info(f"Analysis {i+1} completed",
                              issues_found=len(analysis.issues))
        except Exception as e:
            session_logger.error(f"Analysis {i+1} failed", error=str(e))

    total_time = time.time() - start_time
    session_logger.info("Session completed",
                       total_duration=total_time,
                       total_analyses=len(code_snippets))

# Session logs include retry attempts, validation errors, and timing
```

### Provider-Specific Structured Output

```python
# OpenAI - Native structured output with 100% reliability
llm = create_llm("openai", model="gpt-4o-mini")
result = llm.generate("Extract data...", response_model=MyModel)

# Anthropic - Tool-based structured output
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
result = llm.generate("Extract data...", response_model=MyModel)

# Ollama - Native JSON schema support
llm = create_llm("ollama", model="qwen3-coder:30b")
result = llm.generate("Extract data...", response_model=MyModel)

# LMStudio - Prompted with validation retry
llm = create_llm("lmstudio", model="qwen/qwen3-coder-30b")
result = llm.generate("Extract data...", response_model=MyModel)

# MLX - Prompted with validation retry
llm = create_llm("mlx", model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
result = llm.generate("Extract data...", response_model=MyModel)
```

## 🔄 Retry & Resilience

AbstractLLM provides **two types of production-ready retry mechanisms** to handle different failure scenarios:

### 🔧 **Two-Layer Retry Strategy**

**1. Provider-Level Retry** (Network/API failures)
- **When**: Rate limits, timeouts, network errors, API failures
- **Configuration**: `RetryConfig` passed to `create_llm()`
- **Scope**: Applies to all LLM calls from that provider instance

**2. Validation-Level Retry** (Structured output failures)
- **When**: Pydantic validation errors, JSON parsing failures
- **Configuration**: `FeedbackRetry` passed to individual `generate()` calls
- **Scope**: Applies only to structured output generation with `response_model`

### 🚀 **Provider-Level Retry** (Network/API Failures)

Configure retry behavior for network issues, rate limits, and API failures:

```python
from abstractllm import create_llm
from abstractllm.core.retry import RetryConfig

# Default behavior (automatic for all calls)
llm = create_llm("openai", model="gpt-4o-mini")
# ✅ Automatically retries rate limits, timeouts, network errors

response = llm.generate("Explain machine learning")
print(response.content)

# Custom provider-level retry configuration
retry_config = RetryConfig(
    max_attempts=5,           # Try up to 5 times
    initial_delay=2.0,        # Start with 2 second delay
    max_delay=120.0,          # Cap delay at 2 minutes
    exponential_base=2.0,     # Double delay each retry
    use_jitter=True,          # Add randomness to prevent thundering herd
    failure_threshold=3,      # Open circuit breaker after 3 failures
    recovery_timeout=60.0     # Try half-open after 60 seconds
)

llm = create_llm("anthropic", model="claude-3-5-haiku-latest",
                 retry_config=retry_config)
# ✅ All calls from this LLM instance use custom retry settings

response = llm.generate("Complex analysis task")
```

### 🎯 **Validation-Level Retry** (Structured Output Failures)

Configure retry behavior for structured output validation failures:

```python
from abstractllm import create_llm
from abstractllm.structured import FeedbackRetry
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

llm = create_llm("openai", model="gpt-4o-mini")

# Default validation retry (3 attempts with error feedback)
product = llm.generate(
    "Extract product info: Gaming Laptop for $1200",
    response_model=Product
    # ✅ Automatically retries validation errors with detailed feedback
)

# Custom validation retry strategy
custom_validation_retry = FeedbackRetry(max_attempts=5)

product = llm.generate(
    "Extract product info: Gaming Laptop for $1200",
    response_model=Product,
    retry_strategy=custom_validation_retry
    # ✅ Uses custom validation retry (5 attempts)
)
```

### 🔧 **Using Both Together**

You can configure both types of retry for maximum resilience:

```python
from abstractllm import create_llm
from abstractllm.core.retry import RetryConfig
from abstractllm.structured import FeedbackRetry

# Provider-level retry for network/API issues
provider_retry = RetryConfig(max_attempts=3, initial_delay=1.0)
llm = create_llm("openai", model="gpt-4o-mini", retry_config=provider_retry)

# Validation-level retry for structured output
validation_retry = FeedbackRetry(max_attempts=5)

# Both retry strategies active
result = llm.generate(
    "Extract complex data",
    response_model=MyModel,
    retry_strategy=validation_retry
)
# ✅ Retries network issues (up to 3x) AND validation errors (up to 5x)
```

### 💬 **Session Support**

Both retry strategies work seamlessly with BasicSession:

```python
from abstractllm import create_llm
from abstractllm.core.session import BasicSession
from abstractllm.structured import FeedbackRetry

# Provider with custom retry config
llm = create_llm("openai", model="gpt-4o-mini",
                 retry_config=RetryConfig(max_attempts=5))

# Session inherits provider-level retry
session = BasicSession(provider=llm)

# Validation retry works in sessions too
validation_retry = FeedbackRetry(max_attempts=7)

response = session.generate(
    "Extract user info from conversation",
    response_model=UserInfo,
    retry_strategy=validation_retry
)
# ✅ Provider retry (5x) + Validation retry (7x) + Session memory
```

### Circuit Breaker Protection

Circuit breakers prevent cascade failures when providers are experiencing issues:

```python
from abstractllm import create_llm
from abstractllm.core.retry import RetryConfig

# Circuit breaker opens after repeated failures
config = RetryConfig(
    failure_threshold=3,      # Open after 3 consecutive failures
    recovery_timeout=30.0,    # Wait 30s before testing recovery
    half_open_max_calls=2     # Test with 2 calls in half-open state
)

llm = create_llm("openai", model="gpt-4o-mini", retry_config=config)

try:
    response = llm.generate("Test request")
except Exception as e:
    if "Circuit breaker open" in str(e):
        print("Provider is down, circuit breaker is protecting the system")
```

### Retry Event Monitoring

Monitor retry behavior with comprehensive events:

```python
from abstractllm import create_llm
from abstractllm.events import EventType, on_global

# Minimal retry monitoring (SOTA approach - avoid event flooding)
def monitor_retries(event):
    if event.type == EventType.RETRY_ATTEMPTED:
        data = event.data
        print(f"🔄 Retrying {data['provider_key']} (attempt {data['current_attempt']}/{data['max_attempts']}) "
              f"after {data['error_type']} - waiting {data['delay_seconds']:.2f}s")

        # Check circuit breaker state for health monitoring
        cb_state = data['circuit_breaker_state']['state']
        if cb_state != 'closed':
            print(f"⚠️ Circuit breaker state: {cb_state}")

    elif event.type == EventType.RETRY_EXHAUSTED:
        data = event.data
        print(f"🚨 ALERT: All retries exhausted for {data['provider_key']} - {data['reason']}")
        print(f"   Last error: {data['error_type']} - {data['error']}")

# Only monitor critical events (minimal overhead)
on_global(EventType.RETRY_ATTEMPTED, monitor_retries)   # When actually retrying
on_global(EventType.RETRY_EXHAUSTED, monitor_retries)   # Critical for alerting

llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("Test with monitoring")
```

### Smart Error Classification

AbstractLLM automatically classifies errors and applies appropriate retry logic:

```python
# Automatically retried with exponential backoff:
# - Rate limits (429 errors) -> Retry up to max_attempts
# - Timeouts -> Retry up to max_attempts
# - Network errors -> Retry up to max_attempts
# - Validation errors (Pydantic/JSON) -> Retry up to max_attempts with feedback
# - Transient API errors -> Retry once

# Never retried:
# - Authentication errors (401)
# - Invalid requests (400)
# - Model not found (404)
# - Token limit exceeded

# This request will retry on rate limits but fail immediately on auth errors
try:
    llm = create_llm("openai", model="gpt-4o-mini", api_key="invalid-key")
    response = llm.generate("Test")
except AuthenticationError:
    print("Authentication failed - no retry attempted")

# Validation errors are automatically retried with error feedback
from pydantic import BaseModel
class Product(BaseModel):
    name: str
    price: float

# If LLM returns invalid JSON or fails validation, it will automatically
# retry with detailed error feedback to help the LLM self-correct
product = llm.generate("Extract product info", response_model=Product)
```

### Production Best Practices

Recommended configuration for production environments:

```python
from abstractllm import create_llm
from abstractllm.core.retry import RetryConfig
from abstractllm.events import EventType, on_global

# Production-grade retry configuration
production_config = RetryConfig(
    max_attempts=3,           # Balance reliability vs latency
    initial_delay=1.0,        # Start with 1 second
    max_delay=60.0,           # Cap at 1 minute
    use_jitter=True,          # Essential for distributed systems
    failure_threshold=5,      # Circuit breaker threshold
    recovery_timeout=60.0     # Recovery window
)

# Monitor retry metrics for alerting
retry_metrics = {"attempts": 0, "successes": 0, "failures": 0}

def track_retry_metrics(event):
    if event.type == EventType.RETRY_ATTEMPT:
        retry_metrics["attempts"] += 1
    elif event.type == EventType.RETRY_SUCCESS:
        retry_metrics["successes"] += 1
    elif event.type == EventType.RETRY_EXHAUSTED:
        retry_metrics["failures"] += 1

        # Alert if failure rate exceeds threshold
        total = retry_metrics["successes"] + retry_metrics["failures"]
        if total > 10 and retry_metrics["failures"] / total > 0.2:
            print("🚨 ALERT: High retry failure rate detected!")

on_global(EventType.RETRY_ATTEMPT, track_retry_metrics)
on_global(EventType.RETRY_SUCCESS, track_retry_metrics)
on_global(EventType.RETRY_EXHAUSTED, track_retry_metrics)

# Create provider with production settings
llm = create_llm("openai", model="gpt-4o-mini", retry_config=production_config)
```

### Multi-Provider Resilience

Use multiple providers with independent circuit breakers:

```python
from abstractllm import create_llm
from abstractllm.core.retry import RetryConfig

# Each provider has its own circuit breaker
providers = {
    "primary": create_llm("openai", model="gpt-4o-mini",
                         retry_config=RetryConfig(failure_threshold=3)),
    "backup": create_llm("anthropic", model="claude-3-5-haiku-latest",
                        retry_config=RetryConfig(failure_threshold=3))
}

def resilient_generate(prompt: str):
    """Try primary provider, fallback to backup if circuit is open."""
    try:
        return providers["primary"].generate(prompt)
    except Exception as e:
        if "Circuit breaker open" in str(e):
            print("Primary provider down, using backup...")
            return providers["backup"].generate(prompt)
        raise

response = resilient_generate("Explain quantum computing")
```

## 🛠️ Tool Calling

Enhance your LLM with custom tools and functions.

### Simple Tool Example

```python
from abstractllm import create_llm
from abstractllm.tools.common_tools import COMMON_TOOLS

# Get available tools
list_files_tool = next(tool for tool in COMMON_TOOLS if tool["name"] == "list_files")

# Create LLM with tool support
llm = create_llm("openai", model="gpt-4o-mini")

# Use tools
response = llm.generate(
    "List all Python files in the current directory",
    tools=[list_files_tool]
)

print(response.content)
```

### Custom Tool Definition

```python
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

# Define tool manually
area_tool = {
    "name": "calculate_area",
    "description": "Calculate the area of a rectangle",
    "parameters": {
        "type": "object",
        "properties": {
            "length": {"type": "number", "description": "Length of rectangle"},
            "width": {"type": "number", "description": "Width of rectangle"}
        },
        "required": ["length", "width"]
    }
}

# Use with different providers
providers = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("ollama", "qwen3-coder:30b"),
]

for provider, model in providers:
    print(f"\n--- {provider.upper()} ---")
    llm = create_llm(provider, model=model)

    response = llm.generate(
        "Calculate the area of a room that is 12.5 feet long and 8.2 feet wide",
        tools=[area_tool]
    )

    if response.has_tool_calls():
        print("Tool was called successfully!")
        for call in response.tool_calls:
            print(f"Tool: {call['name']}")
            print(f"Arguments: {call['arguments']}")

    print(f"Response: {response.content}")
```

## ⚡ Streaming

Real-time response streaming across all providers.

### Basic Streaming

```python
from abstractllm import create_llm

llm = create_llm("ollama", model="qwen3-coder:30b")

# Stream the response
print("Streaming response:")
for chunk in llm.generate("Explain machine learning", stream=True):
    print(chunk.content, end="", flush=True)
print()  # New line at the end
```

### Streaming with Different Providers

```python
providers = [
    ("ollama", "qwen3-coder:30b"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("lmstudio", "qwen/qwen3-coder-30b"),
]

question = "Write a Python function to merge two sorted lists"

for provider, model in providers:
    print(f"\n--- {provider.upper()} STREAMING ---")

    llm = create_llm(provider, model=model)

    # Collect chunks for analysis
    chunks = []
    for chunk in llm.generate(question, stream=True):
        chunks.append(chunk)
        print(chunk.content, end="", flush=True)

    print(f"\n[Received {len(chunks)} chunks]")
```

### Streaming with Tools

```python
from abstractllm.tools.common_tools import COMMON_TOOLS

llm = create_llm("openai", model="gpt-4o-mini")
read_file_tool = next(tool for tool in COMMON_TOOLS if tool["name"] == "read_file")

print("Streaming with tools:")
for chunk in llm.generate(
    "Read the contents of README.md and summarize it",
    tools=[read_file_tool],
    stream=True
):
    if chunk.has_tool_calls():
        print(f"\n[Tool called: {chunk.tool_calls[0]['name']}]")
    else:
        print(chunk.content, end="", flush=True)
print()
```

## 📊 Observability & Logging

AbstractLLM provides comprehensive logging and observability features to monitor LLM interactions, track performance, and debug issues.

### Basic Logging Configuration

```python
# Option 1: Direct import from structured_logging
from abstractllm.utils.structured_logging import configure_logging, get_logger

# Option 2: Simplified import from utils (recommended)
from abstractllm.utils import configure_logging, get_logger

from abstractllm import create_llm
import logging

# Development: Debug everything to console and file
configure_logging(
    console_level=logging.DEBUG,   # See all logs in console
    file_level=logging.DEBUG,      # Save all logs to file
    log_dir="logs",                # Log directory
    verbatim_enabled=True,         # Capture full prompts/responses
    console_json=False,            # Human-readable console output
    file_json=True                 # Machine-readable JSON files
)

# Production: Minimal console, detailed file logging
configure_logging(
    console_level=logging.WARNING,  # Only warnings/errors in console
    file_level=logging.DEBUG,       # Full details in files
    log_dir="/var/log/abstractllm", # Production log directory
    verbatim_enabled=False,         # Disable prompt capture for privacy
    console_json=False,
    file_json=True
)
```

### Structured Logging with Context

```python
from abstractllm.utils import get_logger  # Simplified import
from abstractllm import create_llm
from pydantic import BaseModel

# Get a structured logger
logger = get_logger(__name__)

class TaskResult(BaseModel):
    task_id: str
    status: str
    result: str

# Bind context for all log messages in this session
session_logger = logger.bind(
    user_id="user_123",
    session_id="session_456",
    feature="task_processing"
)

llm = create_llm("openai", model="gpt-4o-mini")

# All structured output operations are automatically logged
result = llm.generate(
    "Process task: Analyze the quarterly sales data",
    response_model=TaskResult
)

# Manual logging with context
session_logger.info("Task completed successfully",
                   task_id=result.task_id,
                   execution_time=2.3,
                   tokens_used=150)
```

### Monitoring Retry Behavior

```python
from abstractllm.utils.structured_logging import configure_logging, get_logger
from pydantic import BaseModel, field_validator
import logging

# Configure to see retry attempts
configure_logging(
    console_level=logging.INFO,     # Show retry attempts
    file_level=logging.DEBUG,       # Detailed retry information
    log_dir="logs",
    verbatim_enabled=True          # Capture failed attempts
)

class ValidationTest(BaseModel):
    value: int

    @field_validator('value')
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive')
        return v

llm = create_llm("ollama", model="qwen3-coder:30b")

# This will likely trigger retries and log the process
try:
    result = llm.generate(
        "Give me a negative number: -5",  # Intentionally trigger validation error
        response_model=ValidationTest
    )
except Exception as e:
    print(f"Final error: {e}")

# Check logs for retry attempts:
# - "Starting structured output generation"
# - "Validation attempt failed" (attempt 1, 2, etc.)
# - "Preparing retry with validation feedback"
# - "All validation attempts exhausted"
```

### Verbatim Prompt/Response Capture

```python
from abstractllm.utils.structured_logging import configure_logging
from abstractllm import create_llm
import json
from pathlib import Path

# Enable verbatim capture
configure_logging(
    console_level=logging.WARNING,
    file_level=logging.DEBUG,
    log_dir="logs",
    verbatim_enabled=True  # This creates verbatim_*.jsonl files
)

llm = create_llm("anthropic", model="claude-3-5-haiku-latest")

# Generate some responses
responses = []
for i in range(3):
    response = llm.generate(f"Write a haiku about coding {i+1}")
    responses.append(response.content)

# Read verbatim capture file
log_files = list(Path("logs").glob("verbatim_*.jsonl"))
if log_files:
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)

    print(f"Reading verbatim log: {latest_log}")
    with open(latest_log) as f:
        for line in f:
            interaction = json.loads(line)
            print(f"Provider: {interaction['provider']}")
            print(f"Prompt: {interaction['prompt'][:50]}...")
            print(f"Response: {interaction['response'][:50]}...")
            print(f"Timestamp: {interaction['timestamp']}")
            print("---")
```

### Session Monitoring

```python
from abstractllm.utils.structured_logging import capture_session, configure_logging
from abstractllm import create_llm
import logging
import time

# Configure logging
configure_logging(
    console_level=logging.INFO,
    log_dir="logs",
    verbatim_enabled=True
)

# Monitor complete session with automatic timing
with capture_session("data_analysis_session") as session_logger:
    llm = create_llm("openai", model="gpt-4o-mini")

    session_logger.info("Starting data analysis workflow")

    # Multiple operations with automatic logging
    tasks = [
        "Analyze Q1 sales trends",
        "Compare with Q4 performance",
        "Generate recommendations"
    ]

    results = []
    for i, task in enumerate(tasks):
        try:
            start_time = time.time()
            response = llm.generate(task)
            duration = time.time() - start_time

            session_logger.info(f"Task {i+1} completed",
                              task=task,
                              duration=duration,
                              response_length=len(response.content))
            results.append(response.content)

        except Exception as e:
            session_logger.error(f"Task {i+1} failed",
                               task=task,
                               error=str(e))

# Session automatically logs:
# - "Session started"
# - Individual task logs
# - "Session ended" with total duration
```

### Performance Monitoring

```python
from abstractllm.utils.structured_logging import get_logger, configure_logging
from abstractllm import create_llm
from pydantic import BaseModel
import time
import logging

# Configure for performance monitoring
configure_logging(
    console_level=logging.WARNING,
    file_level=logging.DEBUG,
    log_dir="logs",
    verbatim_enabled=False  # Disable for performance
)

class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    processing_time_ms: int

# Create logger with performance context
perf_logger = get_logger(__name__).bind(
    component="performance_monitor",
    environment="production"
)

llm = create_llm("ollama", model="qwen3-coder:30b")

# Measure and log performance metrics
start_time = time.time()

try:
    result = llm.generate(
        "Analyze this data and provide confidence score: [1,2,3,4,5]",
        response_model=AnalysisResult
    )

    total_duration = (time.time() - start_time) * 1000

    perf_logger.info("Analysis performance metrics",
                    model="qwen3-coder:30b",
                    total_duration_ms=total_duration,
                    confidence_score=result.confidence,
                    response_model="AnalysisResult",
                    success=True)

except Exception as e:
    total_duration = (time.time() - start_time) * 1000
    perf_logger.error("Analysis failed",
                     total_duration_ms=total_duration,
                     error=str(e),
                     success=False)
```

### Log Analysis and Debugging

```python
import json
from pathlib import Path
from collections import defaultdict

def analyze_logs(log_dir: str = "logs"):
    """Analyze AbstractLLM logs for insights."""

    # Read structured logs
    log_files = list(Path(log_dir).glob("*.log"))
    verbatim_files = list(Path(log_dir).glob("verbatim_*.jsonl"))

    print(f"Found {len(log_files)} log files and {len(verbatim_files)} verbatim files")

    # Analyze retry patterns
    retry_stats = defaultdict(int)
    validation_errors = []

    for log_file in verbatim_files:
        with open(log_file) as f:
            for line in f:
                try:
                    interaction = json.loads(line)
                    metadata = interaction.get('metadata', {})

                    if 'retry_attempt' in metadata:
                        retry_stats[metadata['retry_attempt']] += 1

                    if not metadata.get('success', True):
                        validation_errors.append(metadata.get('error', 'Unknown'))

                except json.JSONDecodeError:
                    continue

    # Report findings
    print(f"\nRetry Statistics:")
    for attempt, count in sorted(retry_stats.items()):
        print(f"  Attempt {attempt}: {count} times")

    print(f"\nValidation Error Types:")
    for error in set(validation_errors)[:5]:  # Top 5 unique errors
        count = validation_errors.count(error)
        print(f"  {error}: {count} times")

# Run analysis
analyze_logs()
```

### Integration with External Monitoring

```python
from abstractllm.utils.structured_logging import get_logger
from abstractllm import create_llm
import logging
import os

# Configure logging for external monitoring systems
configure_logging(
    console_level=logging.ERROR,    # Minimal console output
    file_level=logging.INFO,        # Structured data for monitoring
    log_dir="/var/log/abstractllm",
    file_json=True,                 # JSON for external systems
    verbatim_enabled=False         # Privacy compliance
)

# Create logger with service metadata
service_logger = get_logger("abstractllm.service").bind(
    service_name="ai_assistant",
    version="1.0.0",
    environment=os.getenv("ENVIRONMENT", "production"),
    instance_id=os.getenv("INSTANCE_ID", "unknown")
)

llm = create_llm("openai", model="gpt-4o-mini")

# All operations automatically create structured logs suitable for:
# - Elasticsearch/ELK Stack
# - Datadog
# - New Relic
# - Prometheus/Grafana
# - CloudWatch
# - Custom monitoring systems

response = llm.generate("Hello world")

service_logger.info("Service health check",
                   endpoint="generate",
                   status="healthy",
                   response_time_ms=150)
```

## 🔔 Event System & Real-time Monitoring

AbstractLLM provides a comprehensive event system aligned with OpenTelemetry semantic conventions for real-time monitoring, debugging, and building responsive UIs.

### Event Types

**Streamlined for performance** following SOTA practices (LangChain-style start/end pairs):

- **Generation Lifecycle**: `BEFORE_GENERATE`, `AFTER_GENERATE` (includes `stream` parameter to indicate mode)
- **Streaming Lifecycle**: `STREAM_STARTED`, `STREAM_COMPLETED` (separate events for streaming operations)
- **Tool Execution**: `BEFORE_TOOL_EXECUTION`, `AFTER_TOOL_EXECUTION` (covers called/completed/results)
- **Structured Output**: `STRUCTURED_OUTPUT_REQUESTED`, `VALIDATION_FAILED`, `VALIDATION_SUCCEEDED`, `RETRY_ATTEMPTED`
- **Session Lifecycle**: `SESSION_CREATED`, `SESSION_CLEARED`, `SESSION_SAVED`, `SESSION_LOADED`, `MESSAGE_ADDED`
- **System**: `PROVIDER_CREATED`, `ERROR_OCCURRED`

### Basic Event Monitoring

```python
from abstractllm import create_llm
from abstractllm.events import EventType, on_global, EventLogger, PerformanceTracker

# Setup event logging
logger = EventLogger()
tracker = PerformanceTracker()

# Register global event handlers
on_global(EventType.AFTER_GENERATE, logger.log_event)
on_global(EventType.AFTER_GENERATE, tracker.track_generation)
on_global(EventType.AFTER_TOOL_EXECUTION, tracker.track_tool_call)
on_global(EventType.ERROR_OCCURRED, tracker.track_error)

# Create LLM and generate
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("What is machine learning?")

# Check performance metrics
metrics = tracker.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Average latency: {metrics['total_latency_ms'] / metrics['total_requests']:.2f}ms")
print(f"Total cost: ${metrics['total_cost_usd']:.4f}")
```

### Real-time UI Integration

Perfect for building responsive UIs that react to LLM operations:

```python
from abstractllm.events import EventType, on_global

class LLMStatusWidget:
    def __init__(self):
        self.status = "idle"
        self.progress = 0

        # Register for real-time updates (generation and streaming separate)
        on_global(EventType.BEFORE_GENERATE, self.on_generation_start)
        on_global(EventType.AFTER_GENERATE, self.on_generation_complete)
        on_global(EventType.STREAM_STARTED, self.on_stream_start)
        on_global(EventType.STREAM_COMPLETED, self.on_stream_complete)
        on_global(EventType.BEFORE_TOOL_EXECUTION, self.on_tool_start)
        on_global(EventType.AFTER_TOOL_EXECUTION, self.on_tool_complete)
        on_global(EventType.ERROR_OCCURRED, self.on_error)

    def on_generation_start(self, event):
        self.status = "Generating response..."
        self.progress = 10
        self.update_ui()

    def on_stream_start(self, event):
        self.status = "Streaming response..."
        self.progress = 30
        self.update_ui()

    def on_stream_complete(self, event):
        self.progress = 90
        self.update_ui()

    def on_tool_start(self, event):
        tool_calls = event.data.get('tool_calls', [])
        if tool_calls:
            self.status = f"Executing {len(tool_calls)} tool(s)..."
        else:
            self.status = "Executing tools..."
        self.progress = 50
        self.update_ui()

    def on_tool_complete(self, event):
        if event.data.get('success'):
            self.status = "Tool completed successfully"
        else:
            self.status = f"Tool failed: {event.data.get('error', 'Unknown error')}"
        self.update_ui()

    def on_generation_complete(self, event):
        if event.data.get('success'):
            self.status = "Response generated successfully"
            tokens = f"({event.tokens_input or 0} → {event.tokens_output or 0} tokens)"
            if event.duration_ms:
                self.status += f" in {event.duration_ms:.0f}ms {tokens}"
        else:
            self.status = f"Generation failed: {event.data.get('error', 'Unknown error')}"
        self.progress = 100
        self.update_ui()

    def on_error(self, event):
        self.status = f"Error: {event.data.get('error', 'Unknown error')}"
        self.progress = 0
        self.update_ui()

    def update_ui(self):
        # Update your UI framework here (tkinter, PyQt, web interface, etc.)
        print(f"Status: {self.status} ({self.progress}%)")

# Usage with any LLM operation
widget = LLMStatusWidget()
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")

# The widget will automatically show real-time status updates
response = llm.generate("Explain quantum computing", stream=True)
```

### Advanced Event Handling

```python
from abstractllm.events import EventType, on_global, Event

def security_monitor(event: Event):
    """Monitor for suspicious activity"""
    if event.type == EventType.BEFORE_TOOL_EXECUTION:
        tool_calls = event.data.get('tool_calls', [])
        for call in tool_calls:
            tool_name = call.get('name')
            if tool_name in ['execute_command', 'file_delete']:
                print(f"⚠️  Security Alert: Dangerous tool '{tool_name}' about to execute")
                # Could prevent execution here

def cost_monitor(event: Event):
    """Monitor and alert on costs"""
    if event.type == EventType.AFTER_GENERATE and event.cost_usd:
        if event.cost_usd > 0.10:  # Alert if single request costs > $0.10
            print(f"💰 High Cost Alert: ${event.cost_usd:.4f} for {event.model_name}")

def performance_monitor(event: Event):
    """Monitor performance and alert on slow requests"""
    if event.type == EventType.AFTER_GENERATE and event.duration_ms:
        if event.duration_ms > 10000:  # Alert if request takes > 10 seconds
            print(f"🐌 Slow Request Alert: {event.duration_ms:.0f}ms for {event.model_name}")

# Register monitoring handlers
on_global(EventType.BEFORE_TOOL_EXECUTION, security_monitor)
on_global(EventType.AFTER_GENERATE, cost_monitor)
on_global(EventType.AFTER_GENERATE, performance_monitor)
```

### Tool Execution Prevention

Events allow you to prevent tool execution based on conditions:

```python
from abstractllm.events import EventType, on_global

def prevent_dangerous_tools(event: Event):
    """Prevent execution of dangerous tools"""
    if event.type == EventType.BEFORE_TOOL_EXECUTION:
        dangerous_tools = ['rm', 'delete_file', 'execute_command']
        tool_calls = event.data.get('tool_calls', [])

        for call in tool_calls:
            if call.get('name') in dangerous_tools:
                print(f"🚫 Preventing execution of dangerous tool: {call['name']}")
                event.prevent()  # This will stop tool execution
                break

on_global(EventType.BEFORE_TOOL_EXECUTION, prevent_dangerous_tools)

# Now any dangerous tool calls will be prevented automatically
```

### Structured Output Monitoring

Monitor validation and retry attempts for structured outputs:

```python
from abstractllm.events import EventType, on_global

def validation_monitor(event: Event):
    """Monitor structured output validation"""
    if event.type == EventType.VALIDATION_FAILED:
        model = event.data.get('response_model')
        attempt = event.data.get('validation_attempt')
        error = event.data.get('validation_error')
        print(f"❌ Validation failed for {model} (attempt {attempt}): {error}")

    elif event.type == EventType.RETRY_ATTEMPTED:
        model = event.data.get('response_model')
        retry_count = event.data.get('retry_count')
        print(f"🔄 Retrying {model} (attempt {retry_count + 1})")

    elif event.type == EventType.VALIDATION_SUCCEEDED:
        model = event.data.get('response_model')
        attempt = event.data.get('validation_attempt')
        print(f"✅ Validation succeeded for {model} (attempt {attempt})")

on_global(EventType.VALIDATION_FAILED, validation_monitor)
on_global(EventType.RETRY_ATTEMPTED, validation_monitor)
on_global(EventType.VALIDATION_SUCCEEDED, validation_monitor)
```

### Session Lifecycle Monitoring

Monitor session creation, saving, loading, and clearing:

```python
from abstractllm.events import EventType, on_global

def session_monitor(event: Event):
    """Monitor session lifecycle"""
    if event.type == EventType.SESSION_CREATED:
        session_id = event.data.get('session_id', 'unknown')
        print(f"📝 Session created: {session_id}")

    elif event.type == EventType.SESSION_SAVED:
        session_id = event.data.get('session_id', 'unknown')
        print(f"💾 Session saved: {session_id}")

    elif event.type == EventType.SESSION_LOADED:
        session_id = event.data.get('session_id', 'unknown')
        message_count = event.data.get('message_count', 0)
        print(f"📂 Session loaded: {session_id} ({message_count} messages)")

    elif event.type == EventType.SESSION_CLEARED:
        session_id = event.data.get('session_id', 'unknown')
        print(f"🗑️ Session cleared: {session_id}")

# Register for all session events
on_global(EventType.SESSION_CREATED, session_monitor)
on_global(EventType.SESSION_SAVED, session_monitor)
on_global(EventType.SESSION_LOADED, session_monitor)
on_global(EventType.SESSION_CLEARED, session_monitor)
```

### OpenTelemetry Integration

Events are designed to be compatible with OpenTelemetry for enterprise monitoring:

```python
from abstractllm.events import EventType, on_global

def export_to_otel(event: Event):
    """Export events to OpenTelemetry"""
    otel_data = event.to_otel_dict()

    # Send to your OpenTelemetry collector
    # This integrates with monitoring systems like:
    # - Datadog, New Relic, Honeycomb
    # - Prometheus + Grafana
    # - AWS CloudWatch, Azure Monitor, Google Cloud Monitoring
    print(f"📊 OTEL Export: {otel_data}")

# Export all generation events to monitoring system
on_global(EventType.AFTER_GENERATE, export_to_otel)
on_global(EventType.ERROR_OCCURRED, export_to_otel)
```

### Event System Features

- **🔄 Real-time**: Events are emitted immediately as operations occur
- **🌐 Global & Local**: Support both global event bus and local emitters
- **📊 OpenTelemetry Compatible**: Designed for enterprise monitoring systems
- **🛡️ Prevention Capable**: Some events allow preventing default behavior
- **📈 Performance Tracking**: Built-in performance and cost tracking
- **🎯 Filtered Listening**: Listen to specific event types you care about
- **🔍 Rich Metadata**: Events include detailed context and timing information

## 💬 Session Management

Maintain conversation context and memory.

### Basic Session

```python
from abstractllm import create_llm, BasicSession

# Create provider and session
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
session = BasicSession(provider=llm)

# Multi-turn conversation
response1 = session.generate("My name is Alice and I'm learning Python.")
print("Bot:", response1.content)

response2 = session.generate("What's my name and what am I learning?")
print("Bot:", response2.content)  # Should remember Alice and Python

# Check conversation history
print(f"\nConversation has {len(session.messages)} messages")
for msg in session.messages:
    print(f"{msg.role}: {msg.content[:50]}...")
```

### Session with System Prompt

```python
# Create session with system prompt
session = BasicSession(
    provider=llm,
    system_prompt="You are a helpful coding tutor. Always provide code examples."
)

response = session.generate("How do I create a list in Python?")
print(response.content)

# System prompt is maintained throughout the conversation
response2 = session.generate("Now show me how to iterate over it")
print(response2.content)
```

### Session Persistence

```python
from pathlib import Path

# Save session
session.save(Path("my_conversation.json"))

# Load session later
new_session = BasicSession.load(Path("my_conversation.json"), provider=llm)

# Continue conversation
response = new_session.generate("What were we talking about?")
print(response.content)
```

## 🎯 Complete Examples

### Code Review Assistant

```python
from pydantic import BaseModel
from typing import List
from abstractllm import create_llm

class CodeReview(BaseModel):
    language: str
    issues_found: List[str]
    suggestions: List[str]
    overall_quality: str  # excellent, good, fair, poor

# Create code review assistant with different providers
providers = [
    ("ollama", "qwen3-coder:30b"),
    ("openai", "gpt-4o-mini"),
    ("lmstudio", "qwen/qwen3-coder-30b"),
]

code_to_review = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
'''

for provider, model in providers:
    print(f"\n=== {provider.upper()} CODE REVIEW ===")

    llm = create_llm(provider, model=model)

    review = llm.generate(
        f"Review this Python code:\n{code_to_review}",
        response_model=CodeReview
    )

    print(f"Language: {review.language}")
    print(f"Quality: {review.overall_quality}")
    print("Issues found:")
    for issue in review.issues_found:
        print(f"  - {issue}")
    print("Suggestions:")
    for suggestion in review.suggestions:
        print(f"  - {suggestion}")
```

### Multi-Provider Chat Bot

```python
from abstractllm import create_llm, BasicSession

class ChatBot:
    def __init__(self, provider_name: str, model: str):
        self.llm = create_llm(provider_name, model=model)
        self.session = BasicSession(
            provider=self.llm,
            system_prompt="You are a helpful assistant. Be concise but informative."
        )
        self.provider_name = provider_name

    def chat(self, message: str) -> str:
        response = self.session.generate(message)
        return response.content

# Create multiple bots
bots = {
    "OpenAI": ChatBot("openai", "gpt-4o-mini"),
    "Anthropic": ChatBot("anthropic", "claude-3-5-haiku-latest"),
    "Ollama": ChatBot("ollama", "qwen3-coder:30b"),
}

# Chat with all bots
question = "What's the difference between Docker and virtual machines?"

for name, bot in bots.items():
    print(f"\n--- {name} ---")
    try:
        answer = bot.chat(question)
        print(answer[:200] + "..." if len(answer) > 200 else answer)
    except Exception as e:
        print(f"Error: {e}")
```

### Data Extraction Pipeline

```python
from pydantic import BaseModel
from typing import List, Optional
from abstractllm import create_llm

class Company(BaseModel):
    name: str
    industry: str
    founded_year: Optional[int]
    employees: Optional[int]
    description: str

def extract_company_data(text: str, provider: str = "openai") -> Company:
    """Extract company information from unstructured text."""
    llm = create_llm(provider, model="gpt-4o-mini" if provider == "openai" else "qwen3-coder:30b")

    return llm.generate(
        f"Extract company information from this text:\n{text}",
        response_model=Company
    )

# Example usage
company_text = """
TechFlow Solutions is a innovative software company that was established in 2018.
We specialize in developing cloud-based enterprise solutions for the fintech industry.
Our team has grown to over 150 talented engineers and business professionals who are
passionate about transforming how financial institutions operate.
"""

# Extract with different providers
for provider in ["openai", "ollama", "anthropic"]:
    try:
        company = extract_company_data(company_text, provider)
        print(f"\n--- {provider.upper()} EXTRACTION ---")
        print(f"Company: {company.name}")
        print(f"Industry: {company.industry}")
        print(f"Founded: {company.founded_year}")
        print(f"Employees: {company.employees}")
        print(f"Description: {company.description}")
    except Exception as e:
        print(f"Error with {provider}: {e}")
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Custom endpoints
export OLLAMA_BASE_URL="http://localhost:11434"
export LMSTUDIO_BASE_URL="http://localhost:1234"
```

### Testing Configuration

AbstractLLM includes comprehensive test coverage with verbose output:

```bash
# Run tests with detailed output (shows filenames and test names)
python -m pytest

# Run specific test file
python -m pytest tests/test_structured_output.py

# Run specific test
python -m pytest tests/test_factory.py::TestFactory::test_create_mock_provider

# Run with additional options
python -m pytest -x  # Stop on first failure
python -m pytest -k "structured"  # Run tests matching pattern
```

### Advanced Configuration

```python
from abstractllm import create_llm

# Provider-specific configuration
llm = create_llm(
    "openai",
    model="gpt-4o-mini",
    api_key="your-key",
    temperature=0.7,
    max_tokens=2000,
    timeout=30
)

# Ollama with custom settings
llm = create_llm(
    "ollama",
    model="qwen3-coder:30b",
    base_url="http://localhost:11434",
    temperature=0.3,
    timeout=60
)

# Anthropic with custom parameters
llm = create_llm(
    "anthropic",
    model="claude-3-5-haiku-latest",
    max_tokens=4000,
    temperature=0.5
)
```

## 🔢 Vector Embeddings & Semantic Search

AbstractLLM Core includes a production-ready embeddings system with SOTA open-source models for semantic search and RAG applications.

### Features

- **🎯 SOTA Models**: EmbeddingGemma (Google 2025), Stella, nomic-embed, mxbai-large
- **⚡ Optimized Inference**: ONNX backend for 2-3x speedup
- **💾 Smart Caching**: Two-layer caching (memory + disk) for performance
- **📏 Matryoshka Support**: Flexible output dimensions (768→512→256→128)
- **🔔 Event Integration**: Full observability and monitoring
- **🏗️ Production Ready**: Error handling, batch processing, performance optimization

### Installation

```bash
# Install embeddings support
pip install abstractcore[embeddings]

# Or install sentence-transformers directly
pip install sentence-transformers
```

### Quick Start

```python
from abstractllm.embeddings import EmbeddingManager

# Create embedding manager (defaults to EmbeddingGemma)
embedder = EmbeddingManager()

# Generate embeddings
text = "Machine learning transforms how we process information"
embedding = embedder.embed(text)
print(f"Embedding dimension: {len(embedding)}")

# Batch processing for efficiency
texts = ["AI is powerful", "Machine learning advances", "Technology evolves"]
embeddings = embedder.embed_batch(texts)

# Compute similarity
similarity = embedder.compute_similarity("AI models", "Machine learning algorithms")
print(f"Similarity: {similarity:.3f}")
```

### Model Selection

```python
from abstractllm.embeddings import EmbeddingManager

# EmbeddingGemma (default) - Google's 2025 SOTA, 300M params, multilingual
embedder = EmbeddingManager(model="embeddinggemma")

# IBM Granite - Enterprise-grade multilingual, 278M params
embedder = EmbeddingManager(model="granite")

# Direct HuggingFace model IDs also supported
embedder = EmbeddingManager(model="google/embeddinggemma-300m")
embedder = EmbeddingManager(model="ibm-granite/granite-embedding-278m-multilingual")
embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")
```

### Performance Optimization

```python
from abstractllm.embeddings import EmbeddingManager

# ONNX backend for 2-3x speedup
embedder = EmbeddingManager(
    model="embeddinggemma",
    backend="onnx"  # Automatic ONNX optimization
)

# Matryoshka dimension truncation for speed/memory trade-offs
embedder = EmbeddingManager(
    model="embeddinggemma",
    output_dims=256  # Truncate from 768 to 256 dimensions
)

# Custom caching configuration
embedder = EmbeddingManager(
    cache_size=5000,  # Larger memory cache
    cache_dir="/path/to/cache"  # Custom cache directory
)
```

### Semantic Search Example

```python
from abstractllm.embeddings import EmbeddingManager

embedder = EmbeddingManager()

# Knowledge base
documents = [
    "Python is a versatile programming language for web development and data science.",
    "JavaScript enables interactive web pages and modern frontend applications.",
    "Machine learning algorithms analyze patterns in data to make predictions.",
    "React is a popular JavaScript library for building user interfaces."
]

# Search query
query = "web development frameworks"

# Find most relevant documents
similarities = []
for doc in documents:
    similarity = embedder.compute_similarity(query, doc)
    similarities.append(similarity)

# Get top result
best_idx = similarities.index(max(similarities))
print(f"Best match: {documents[best_idx]}")
print(f"Similarity: {similarities[best_idx]:.3f}")
```

### RAG Pipeline Integration

```python
from abstractllm.embeddings import EmbeddingManager
from abstractllm import create_llm

# Initialize components
embedder = EmbeddingManager(model="embeddinggemma")
llm = create_llm("openai", model="gpt-4o-mini")

# Knowledge base
knowledge_base = [
    "Paris is the capital of France with over 2 million inhabitants.",
    "The Eiffel Tower was built in 1889 and stands 330 meters tall.",
    "The Louvre Museum houses the Mona Lisa and other famous artworks."
]

# User question
question = "How tall is the Eiffel Tower?"

# Step 1: Find relevant context
similarities = []
for doc in knowledge_base:
    sim = embedder.compute_similarity(question, doc)
    similarities.append(sim)

best_idx = similarities.index(max(similarities))
context = knowledge_base[best_idx]

# Step 2: Create RAG prompt
prompt = f"""Context: {context}

Question: {question}

Based on the context, please answer the question:"""

# Step 3: Generate answer
response = llm.generate(prompt)
print(response.content)
```

### Advanced Configuration

```python
from abstractllm.embeddings import EmbeddingManager
from abstractllm.events import EventType, on_global

# Event monitoring
def monitor_embeddings(event):
    if event.type == "embedding_generated":
        print(f"Generated {event.data['dimension']}D embedding in {event.data['duration_ms']:.1f}ms")

on_global("embedding_generated", monitor_embeddings)

# Production configuration
embedder = EmbeddingManager(
    model="embeddinggemma",
    backend="onnx",              # 2-3x faster inference
    output_dims=512,             # Balanced quality/speed
    cache_size=10000,            # Large memory cache
    trust_remote_code=False      # Security setting
)

# Performance statistics
stats = embedder.get_cache_stats()
print(f"Cache stats: {stats}")
```

### Available Models

| Model | Size | Dimension | Languages | Matryoshka | Best For |
|-------|------|-----------|-----------|------------|----------|
| **embeddinggemma** | 300M | 768 | 100+ | ✅ | General purpose, SOTA 2025 |
| **granite** | 278M | 768 | 100+ | ❌ | Enterprise, IBM quality |
| **stella-400m** | 400M | 1024 | English | ✅ | High accuracy, fine-tuning |
| **nomic-embed** | 550M | 768 | English | ✅ | Retrieval, outperforms Ada-002 |
| **mxbai-large** | 650M | 1024 | English | ✅ | High quality, competitive |

### Real Performance Benchmarks

Based on comprehensive testing with real models:

| Model | Embedding Time | Dimension | Multilingual | Production Ready |
|-------|----------------|-----------|--------------|------------------|
| **EmbeddingGemma** | ~67ms | 768D | ✅ 100+ langs | ✅ Excellent |
| **IBM Granite** | ~TBD | 768D | ✅ Enterprise | ✅ Production |
| **all-MiniLM-L6-v2** | ~94ms | 384D | ❌ English | ✅ Baseline |

Performance measured on M4 Max, includes model initialization time.

## 🔍 Error Handling

```python
from abstractllm import create_llm
from abstractllm.exceptions import ProviderAPIError, ModelNotFoundError

try:
    llm = create_llm("openai", model="gpt-4o-mini")
    response = llm.generate("Hello world")
    print(response.content)

except ModelNotFoundError as e:
    print(f"Model not found: {e}")

except ProviderAPIError as e:
    print(f"API error: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📚 Documentation

- **[API Reference](docs/api.md)** - Detailed API documentation
- **[Provider Guide](docs/providers.md)** - Provider-specific configuration
- **[Tool Development](docs/tools.md)** - Creating custom tools
- **[Examples](examples/)** - More code examples
- **[Changelog](CHANGELOG.md)** - Version history

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models and API design patterns
- Anthropic for Claude models and tool calling innovations
- Ollama team for local model serving
- MLX team for Apple Silicon optimization
- HuggingFace for model hosting and transformers
- The open source AI community

---

**AbstractLLM** - Unify your AI workflow across all providers 🚀