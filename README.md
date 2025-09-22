# AbstractLLM

A unified interface to all LLM providers with essential infrastructure for tool calling, streaming, structured output, and model management.

## ✨ Features

- **🔌 Universal Provider Support**: OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace
- **🛠️ Tool Calling**: Native tool calling with automatic execution
- **📊 Structured Output**: Type-safe JSON responses using Pydantic models
- **⚡ Streaming**: Real-time response streaming across all providers
- **🔄 Session Management**: Conversation memory and context management
- **🎯 Zero Configuration**: Works out of the box with sensible defaults
- **🏗️ Production Ready**: Comprehensive error handling and telemetry

## 🚀 Quick Start

### Installation

```bash
# Install core package
pip install abstractllm

# Install with specific providers
pip install abstractllm[openai,anthropic]  # API providers
pip install abstractllm[ollama,lmstudio]   # Local providers
pip install abstractllm[mlx]               # Apple Silicon
pip install abstractllm[all]               # Everything
```

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
from abstractllm.utils.structured_logging import configure_logging
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
from abstractllm.utils.structured_logging import get_logger
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