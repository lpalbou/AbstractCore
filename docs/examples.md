# Practical Examples

This guide shows real-world use cases for AbstractCore with complete, copy-paste examples. All examples work across any provider - just change the provider name.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Tool Calling Examples](#tool-calling-examples)
- [Tool Call Syntax Rewriting Examples](#tool-call-syntax-rewriting-examples)
- [Structured Output Examples](#structured-output-examples)
- [Streaming Examples](#streaming-examples)
- [Session Management](#session-management)
- [Production Patterns](#production-patterns)
- [Integration Examples](#integration-examples)

## Basic Usage

### Simple Q&A

```python
from abstractcore import create_llm

# Works with any provider
llm = create_llm("openai", model="gpt-4o-mini")  # or "anthropic", "ollama"...

response = llm.generate("What is the difference between Python and JavaScript?")
print(response.content)
```

### Multiple Providers Comparison

```python
from abstractcore import create_llm

providers = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("ollama", "qwen2.5-coder:7b")
]

question = "Explain Python list comprehensions with examples"

for provider_name, model in providers:
    try:
        llm = create_llm(provider_name, model=model)
        response = llm.generate(question)
        print(f"\n--- {provider_name.upper()} ---")
        print(response.content[:200] + "...")
    except Exception as e:
        print(f"{provider_name} failed: {e}")
```

### Provider Fallback

```python
from abstractcore import create_llm

def generate_with_fallback(prompt, **kwargs):
    """Try multiple providers until one works."""
    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-latest"),
        ("ollama", "qwen2.5-coder:7b")
    ]

    for provider_name, model in providers:
        try:
            llm = create_llm(provider_name, model=model)
            return llm.generate(prompt, **kwargs)
        except Exception as e:
            print(f"{provider_name} failed: {e}")
            continue

    raise Exception("All providers failed")

# Usage
response = generate_with_fallback("What is machine learning?")
print(response.content)
```

## Tool Calling Examples

### Weather Tool

```python
from abstractcore import create_llm
import requests

def get_weather(city: str, units: str = "metric") -> str:
    """Get current weather for a city."""
    # In production, use a real weather API
    # This is a simulated implementation
    temperatures = {
        "paris": "22°C, sunny",
        "london": "15°C, cloudy",
        "tokyo": "28°C, humid",
        "new york": "18°C, windy"
    }
    return temperatures.get(city.lower(), f"Weather data not available for {city}")

# Tool definition
weather_tool = {
    "name": "get_weather",
    "description": "Get current weather information for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Name of the city"
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Temperature units"
            }
        },
        "required": ["city"]
    }
}

# Works with any provider that supports tools
llm = create_llm("openai", model="gpt-4o-mini")

response = llm.generate(
    "What's the weather like in Paris and London?",
    tools=[weather_tool]
)

print(response.content)
# Output: The weather in Paris is currently 22°C and sunny, while London is 15°C and cloudy.
```

### Calculator Tool

```python
from abstractcore import create_llm
import math

def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        # In production, use a proper expression parser
        # This is simplified for demo purposes
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def sqrt(number: float) -> str:
    """Calculate square root."""
    try:
        result = math.sqrt(number)
        return f"√{number} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Tool definitions
tools = [
    {
        "name": "calculate",
        "description": "Perform basic mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "sqrt",
        "description": "Calculate square root of a number",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {"type": "number", "description": "Number to calculate square root of"}
            },
            "required": ["number"]
        }
    }
]

llm = create_llm("openai", model="gpt-4o-mini")

response = llm.generate(
    "What is 25 * 4 + 12, and what's the square root of 144?",
    tools=tools
)

print(response.content)
# Output: 25 * 4 + 12 equals 112, and the square root of 144 is 12.
```

### File Operations Tool

```python
from abstractcore import create_llm
from pathlib import Path
import os

def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    try:
        path = Path(directory)
        if not path.exists():
            return f"Directory {directory} does not exist"

        files = []
        for item in path.iterdir():
            if item.is_file():
                files.append(f"📄 {item.name}")
            elif item.is_dir():
                files.append(f"📁 {item.name}/")

        return f"Contents of {directory}:\n" + "\n".join(sorted(files))
    except Exception as e:
        return f"Error listing files: {str(e)}"

def read_file(filename: str) -> str:
    """Read contents of a text file."""
    try:
        path = Path(filename)
        if not path.exists():
            return f"File {filename} does not exist"

        content = path.read_text(encoding='utf-8')
        return f"Contents of {filename}:\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Tool definitions
file_tools = [
    {
        "name": "list_files",
        "description": "List files and directories in a given path",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path to list"}
            }
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a text file",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Path to the file to read"}
            },
            "required": ["filename"]
        }
    }
]

llm = create_llm("anthropic", model="claude-3-5-haiku-latest")

response = llm.generate(
    "List the files in the current directory and read the README.md file if it exists",
    tools=file_tools
)

print(response.content)
```

## Tool Call Syntax Rewriting Examples

> **Real-time tool call format conversion for agentic CLI compatibility**

Tool call syntax rewriting enables AbstractCore to work seamlessly with any agentic CLI by converting tool calls to the expected format in real-time. This happens automatically during generation, including streaming.

> **📋 Related**: [Tool Call Syntax Rewriting Guide](tool-syntax-rewriting.md)

### Codex CLI Integration (Default Format)

```python
from abstractcore import create_llm

# Define tools (standard JSON format)
weather_tool = {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}

# Codex CLI expects qwen3 format (default - no configuration needed)
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
response = llm.generate("What's the weather in Tokyo?", tools=[weather_tool])

print(response.content)
# Output includes: <|tool_call|>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</|tool_call|>
```

### Crush CLI Integration

```python
# Crush CLI expects LLaMA3 format - just specify the format
llm = create_llm("ollama", model="qwen3-coder:30b", tool_call_tags="llama3")
response = llm.generate("Get weather for London", tools=[weather_tool])

print(response.content)
# Output includes: <function_call>{"name": "get_weather", "arguments": {"city": "London"}}</function_call>
```

### Custom CLI Format

```python
# Your custom CLI expects: [TOOL]...JSON...[/TOOL]
llm = create_llm("openai", model="gpt-4o-mini", tool_call_tags="[TOOL],[/TOOL]")
response = llm.generate("Check weather in Paris", tools=[weather_tool])

print(response.content)
# Output includes: [TOOL]{"name": "get_weather", "arguments": {"city": "Paris"}}[/TOOL]
```

### Real-Time Streaming with Tag Rewriting

```python
# Streaming works seamlessly with any format
calculator_tool = {
    "name": "calculate",
    "description": "Perform mathematical calculations",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"]
    }
}

llm = create_llm("ollama", model="qwen3-coder:30b", tool_call_tags="llama3")

print("AI: ", end="", flush=True)
for chunk in llm.generate(
    "Calculate 15 * 23 and explain the result",
    tools=[calculator_tool],
    stream=True
):
    print(chunk.content, end="", flush=True)

    # Tool calls are detected and executed in real-time
    if chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            result = tool_call.execute()
            print(f"\n🛠️ Tool executed: {result}")

print("\n")
# Shows: <function_call>{"name": "calculate", "arguments": {"expression": "15 * 23"}}</function_call>
# With immediate tool execution during streaming
```

### Multiple Tools with Different Formats

```python
# Define multiple tools
tools = [
    {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform calculations",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    },
    {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {"directory": {"type": "string"}},
            "required": ["directory"]
        }
    }
]

# Test with XML format for Gemini CLI
llm = create_llm("anthropic", model="claude-3-5-haiku-latest", tool_call_tags="xml")
response = llm.generate(
    "What's 2+2, weather in NYC, and files in current directory?",
    tools=tools
)

print(response.content)
# All tool calls converted to: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
```

### Session-Based Format Configuration

```python
from abstractcore import BasicSession

# Configure format once for entire session
llm = create_llm("openai", model="gpt-4o-mini", tool_call_tags="llama3")
session = BasicSession(provider=llm)

# All tool calls in this session use LLaMA3 format
session.generate("Calculate 10 * 5", tools=[calculator_tool])
session.generate("What's the weather like?", tools=[weather_tool])
session.generate("List files in documents", tools=[{
    "name": "list_files",
    "description": "List directory contents",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
    }
}])

# All responses contain: <function_call>...JSON...</function_call>
```

### Production Monitoring with Events

```python
from abstractcore.events import EventType, on_global

# Monitor tool usage across different formats
def log_tool_calls(event):
    for call in event.data.get('tool_calls', []):
        print(f"🔧 {call.name} executed for CLI format: {llm.tool_call_tags}")

on_global(EventType.TOOL_COMPLETED, log_tool_calls)

# Test with different formats
for format_name in ["qwen3", "llama3", "xml"]:
    llm = create_llm("ollama", model="qwen3-coder:30b", tool_call_tags=format_name)
    response = llm.generate("Calculate 5 * 5", tools=[calculator_tool])
    print(f"{format_name} format result: {response.content[:100]}...")
```

**Key Benefits**:
- ✅ **Zero Configuration**: Default format works with most CLIs
- ✅ **Real-Time Processing**: No post-processing delays
- ✅ **Streaming Compatible**: Works perfectly with streaming mode
- ✅ **Universal Support**: All providers and models supported
- ✅ **Format Flexibility**: Predefined formats + custom tags

> **📋 Related**: [Tool Call Tag Rewriting Guide](tool-syntax-rewriting.md) | [Unified Streaming Architecture](architecture.md#unified-streaming-architecture)

## Structured Output Examples

### User Profile Extraction

```python
from abstractcore import create_llm
from pydantic import BaseModel, field_validator
from typing import Optional

class UserProfile(BaseModel):
    name: str
    age: int
    email: str
    occupation: Optional[str] = None
    interests: list[str] = []

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v

llm = create_llm("openai", model="gpt-4o-mini")

# Text with user information
user_text = """
Hi, I'm Sarah Johnson, I'm 28 years old and work as a software engineer.
My email is sarah.johnson@techcorp.com. I love hiking, photography, and cooking.
"""

# Extract structured data with automatic validation
user = llm.generate(
    f"Extract user profile from: {user_text}",
    response_model=UserProfile
)

print(f"Name: {user.name}")
print(f"Age: {user.age}")
print(f"Email: {user.email}")
print(f"Occupation: {user.occupation}")
print(f"Interests: {', '.join(user.interests)}")
```

### Product Catalog Extraction

```python
from abstractcore import create_llm
from pydantic import BaseModel, field_validator
from typing import List
from enum import Enum

class ProductCategory(str, Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME = "home"
    SPORTS = "sports"

class Product(BaseModel):
    name: str
    price: float
    category: ProductCategory
    description: str
    in_stock: bool = True

    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

class ProductCatalog(BaseModel):
    products: List[Product]
    total_count: int

    @field_validator('total_count')
    @classmethod
    def validate_count(cls, v, info):
        products = info.data.get('products', [])
        if v != len(products):
            raise ValueError(f'Total count {v} does not match products length {len(products)}')
        return v

llm = create_llm("anthropic", model="claude-3-5-haiku-latest")

catalog_text = """
Our store has these items:
1. Gaming Laptop - $1299.99 - High-performance laptop for gaming and work
2. Wireless Headphones - $199.99 - Noise-cancelling bluetooth headphones
3. Python Programming Book - $49.99 - Complete guide to Python programming
4. Coffee Maker - $89.99 - Automatic drip coffee maker, currently out of stock
"""

catalog = llm.generate(
    f"Extract product catalog from: {catalog_text}",
    response_model=ProductCatalog
)

print(f"Total products: {catalog.total_count}")
for product in catalog.products:
    status = "✅ In Stock" if product.in_stock else "❌ Out of Stock"
    print(f"- {product.name}: ${product.price} ({product.category}) - {status}")
```

### Code Review Analysis

```python
from abstractcore import create_llm
from pydantic import BaseModel
from typing import List
from enum import Enum

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CodeIssue(BaseModel):
    line_number: int
    severity: Severity
    issue_type: str
    description: str
    suggestion: str

class CodeReview(BaseModel):
    language: str
    overall_quality: str
    issues: List[CodeIssue]
    recommendations: List[str]

llm = create_llm("ollama", model="qwen2.5-coder:7b")

code_to_review = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

def process_data(data):
    if data == None:
        return []
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''

review = llm.generate(
    f"Review this Python code for issues:\n{code_to_review}",
    response_model=CodeReview
)

print(f"Language: {review.language}")
print(f"Overall Quality: {review.overall_quality}")
print(f"\nIssues Found ({len(review.issues)}):")
for issue in review.issues:
    print(f"  Line {issue.line_number}: [{issue.severity.upper()}] {issue.issue_type}")
    print(f"    Problem: {issue.description}")
    print(f"    Fix: {issue.suggestion}\n")

print("Recommendations:")
for rec in review.recommendations:
    print(f"  - {rec}")
```

## Streaming Examples

### Basic Streaming (Unified 2025)

```python
# Real-time streaming works identically across ALL providers
from abstractcore import create_llm

llm = create_llm("anthropic", model="claude-3-5-haiku-latest")

print("AI Story Generator: ", end="", flush=True)
for chunk in llm.generate(
    "Write a short story about a programmer who discovers their code is alive",
    stream=True
):
    print(chunk.content, end="", flush=True)
print("\n")
```

### Advanced Streaming with Progress and Performance Tracking

```python
from abstractcore import create_llm
import time

def streaming_with_insights(prompt):
    # Supports any provider: OpenAI, Anthropic, Ollama, MLX
    llm = create_llm("openai", model="gpt-4o-mini")

    print("🤖 Generating response...")

    start_time = time.time()
    chunks = []

    print("Response: ", end="", flush=True)
    for chunk in llm.generate(prompt, stream=True):
        chunks.append(chunk)
        print(chunk.content, end="", flush=True)

        # Optional real-time performance insights
        if len(chunks) % 10 == 0:
            current_time = time.time() - start_time
            chars_generated = sum(len(c.content) for c in chunks)
            print(f"\n📊 Progress: {len(chunks)} chunks, {chars_generated} chars, {current_time:.1f}s")

    # Final performance summary
    total_time = time.time() - start_time
    total_chars = sum(len(chunk.content) for chunk in chunks)

    print(f"\n\n🚀 Streaming Stats:")
    print(f"- Total Chunks: {len(chunks)}")
    print(f"- Total Characters: {total_chars}")
    print(f"- Duration: {total_time:.2f}s")
    print(f"- Speed: {total_chars/total_time:.0f} chars/sec")

# Usage with various prompts
streaming_with_insights("Explain quantum computing in simple terms")
```

### Real-Time Streaming with Tools (Unified Implementation)

```python
from abstractcore import create_llm
from datetime import datetime

def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%H:%M:%S")

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "New York": "Sunny, 22°C",
        "London": "Cloudy, 15°C",
        "Tokyo": "Partly cloudy, 25°C"
    }
    return weather_data.get(city, f"Weather data unavailable for {city}")

time_tool = {
    "name": "get_current_time",
    "description": "Get the current time",
    "parameters": {"type": "object", "properties": {}}
}

weather_tool = {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Name of the city"}
        }
    }
}

# Works identically across providers
llm = create_llm("ollama", model="qwen2.5-coder:7b")

print("🤖 AI Assistant: ", end="", flush=True)
for chunk in llm.generate(
    "What time is it right now? And can you tell me the weather in New York?",
    tools=[time_tool, weather_tool],
    stream=True
):
    # Real-time chunk processing and tool execution
    print(chunk.content, end="", flush=True)

    # Immediate tool call detection and execution
    if chunk.tool_calls:
        for tool_call in chunk.tool_calls:
            result = tool_call.execute()
            print(f"\n🛠️ Tool Result: {result}")

print("\n")  # Newline after streaming

# Features:
# ✅ Real-time tool call detection
# ✅ Immediate mid-stream tool execution
# ✅ Zero buffering overhead
# ✅ Works with OpenAI, Anthropic, Ollama, MLX
# ✅ Consistent behavior across all providers
```

### Performance-Optimized Streaming

```python
from abstractcore import create_llm
import time

def compare_providers(prompt):
    """Compare streaming performance across providers."""
    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-latest"),
        ("ollama", "qwen2.5-coder:7b")
    ]

    for provider, model in providers:
        try:
            llm = create_llm(provider, model=model)

            print(f"\n📊 Testing {provider.upper()} - {model}")
            start_time = time.time()

            chunks = []
            for chunk in llm.generate(prompt, stream=True):
                chunks.append(chunk)
                print(chunk.content, end="", flush=True)

            total_time = time.time() - start_time
            total_chars = sum(len(chunk.content) for chunk in chunks)

            print(f"\n\n🚀 {provider.upper()} Performance:")
            print(f"- Chunks: {len(chunks)}")
            print(f"- Characters: {total_chars}")
            print(f"- Duration: {total_time:.2f}s")
            print(f"- Speed: {total_chars/total_time:.0f} chars/sec")

        except Exception as e:
            print(f"❌ {provider} failed: {e}")

# Compare streaming performance
compare_providers("Write a creative short story about artificial intelligence")
```

**Streaming Features**:
- ⚡ First chunk in <10ms
- 🔧 Unified strategy across ALL providers
- 🛠️ Real-time tool call detection
- 📊 Mid-stream tool execution
- 💨 Zero buffering overhead
- 🚀 Supports: OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace
- 🔒 Robust error handling for malformed responses

## Session Management

### Basic Conversation

```python
from abstractcore import create_llm, BasicSession

llm = create_llm("openai", model="gpt-4o-mini")
session = BasicSession(
    provider=llm,
    system_prompt="You are a helpful coding tutor. Always provide examples."
)

# Multi-turn conversation
print("=== Conversation Start ===")

response1 = session.generate("Hi, I'm learning Python. What are decorators?")
print("User: Hi, I'm learning Python. What are decorators?")
print(f"AI: {response1.content}\n")

response2 = session.generate("Can you show me a practical example?")
print("User: Can you show me a practical example?")
print(f"AI: {response2.content}\n")

response3 = session.generate("What was my first question?")
print("User: What was my first question?")
print(f"AI: {response3.content}\n")

print(f"Total messages in conversation: {len(session.messages)}")
```

### Session Persistence

```python
from abstractcore import create_llm, BasicSession
from pathlib import Path

# Create and use session
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
session = BasicSession(
    provider=llm,
    system_prompt="You are a travel advisor. Help plan trips."
)

# Have a conversation
session.generate("I want to plan a trip to Japan")
session.generate("I'm interested in both modern cities and traditional culture")
session.generate("My budget is around $3000 for 10 days")

# Save session
session_file = Path("travel_planning_session.json")
session.save(session_file)
print(f"Session saved to {session_file}")

# Later: Load session and continue
new_session = BasicSession.load(session_file, provider=llm)
response = new_session.generate("What were we discussing?")
print(f"AI remembers: {response.content}")

# Clean up
session_file.unlink()  # Delete the file
```

### Context Management

```python
from abstractcore import create_llm, BasicSession

def create_coding_assistant():
    """Create a specialized coding assistant session."""
    llm = create_llm("ollama", model="qwen2.5-coder:7b")

    system_prompt = """
    You are an expert Python coding assistant. For each request:
    1. Provide working code examples
    2. Explain the code clearly
    3. Mention potential issues or improvements
    4. Keep responses concise but complete
    """

    return BasicSession(provider=llm, system_prompt=system_prompt)

# Usage
assistant = create_coding_assistant()

# The assistant will remember the context throughout the conversation
assistant.generate("I need a function to validate email addresses")
assistant.generate("Now add logging to that function")
assistant.generate("How would I test this function?")

print(f"Conversation history: {len(assistant.messages)} messages")

# Clear history but keep system prompt
assistant.clear_history()
print(f"After clearing: {len(assistant.messages)} messages")  # Just system prompt remains
```

## Production Patterns

### Retry and Error Handling

```python
from abstractcore import create_llm
from abstractcore.core.retry import RetryConfig
from abstractcore.exceptions import ProviderAPIError, RateLimitError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_production_llm():
    """Create LLM with production-grade retry configuration."""
    retry_config = RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        max_delay=30.0,
        use_jitter=True,
        failure_threshold=5
    )

    return create_llm(
        "openai",
        model="gpt-4o-mini",
        retry_config=retry_config,
        timeout=30
    )

def safe_generate(prompt, **kwargs):
    """Generate with comprehensive error handling."""
    llm = create_production_llm()

    try:
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        response = llm.generate(prompt, **kwargs)
        logger.info(f"Response generated successfully: {len(response.content)} chars")
        return response

    except RateLimitError as e:
        logger.warning(f"Rate limited: {e}")
        raise

    except ProviderAPIError as e:
        logger.error(f"API error: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

# Usage
try:
    response = safe_generate("What is machine learning?")
    print(response.content)
except Exception as e:
    print(f"Generation failed: {e}")
```

### Cost Monitoring

```python
from abstractcore import create_llm
from abstractcore.events import EventType, on_global
from datetime import datetime
import json

class CostMonitor:
    def __init__(self, budget_limit=10.0):
        self.total_cost = 0.0
        self.budget_limit = budget_limit
        self.requests = []

        # Register event handlers
        on_global(EventType.AFTER_GENERATE, self.track_cost)

    def track_cost(self, event):
        """Track costs from generation events."""
        if hasattr(event, 'cost_usd') and event.cost_usd:
            self.total_cost += event.cost_usd
            self.requests.append({
                'timestamp': datetime.now().isoformat(),
                'provider': event.data.get('provider'),
                'model': event.data.get('model'),
                'cost': event.cost_usd,
                'tokens_input': event.tokens_input,
                'tokens_output': event.tokens_output
            })

            print(f"💰 Cost: ${event.cost_usd:.4f} | Total: ${self.total_cost:.4f}")

            if self.total_cost > self.budget_limit:
                print(f"🚨 BUDGET EXCEEDED! ${self.total_cost:.4f} > ${self.budget_limit}")

    def get_report(self):
        """Get cost report."""
        return {
            'total_cost': self.total_cost,
            'budget_limit': self.budget_limit,
            'total_requests': len(self.requests),
            'average_cost': self.total_cost / len(self.requests) if self.requests else 0,
            'requests': self.requests
        }

# Usage
monitor = CostMonitor(budget_limit=1.0)  # $1 budget

llm = create_llm("openai", model="gpt-4o-mini")

# Make some requests
for i in range(3):
    response = llm.generate(f"Tell me a fact about number {i+1}")
    print(f"Fact {i+1}: {response.content[:100]}...\n")

# Get report
report = monitor.get_report()
print(f"\n📊 Final Report:")
print(f"Total cost: ${report['total_cost']:.4f}")
print(f"Requests: {report['total_requests']}")
print(f"Average per request: ${report['average_cost']:.4f}")
```

### Load Balancing

```python
from abstractcore import create_llm
import random
import time
from typing import List, Tuple

class LoadBalancer:
    def __init__(self, providers: List[Tuple[str, str]]):
        """Initialize with list of (provider, model) tuples."""
        self.providers = []
        self.weights = []

        for provider_name, model in providers:
            try:
                llm = create_llm(provider_name, model=model)
                self.providers.append((llm, provider_name, model))
                self.weights.append(1.0)  # Equal weight initially
                print(f"✅ {provider_name} ({model}) ready")
            except Exception as e:
                print(f"❌ {provider_name} ({model}) failed: {e}")

    def generate(self, prompt, **kwargs):
        """Generate using weighted random selection."""
        if not self.providers:
            raise Exception("No providers available")

        # Weighted random selection
        provider_data = random.choices(
            self.providers,
            weights=self.weights,
            k=1
        )[0]

        llm, provider_name, model = provider_data

        try:
            start_time = time.time()
            response = llm.generate(prompt, **kwargs)
            duration = time.time() - start_time

            print(f"✅ {provider_name} responded in {duration:.2f}s")
            return response

        except Exception as e:
            print(f"❌ {provider_name} failed: {e}")
            # Remove failed provider temporarily
            idx = self.providers.index(provider_data)
            self.weights[idx] *= 0.1  # Reduce weight dramatically
            raise

# Usage
balancer = LoadBalancer([
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("ollama", "qwen2.5-coder:7b")
])

# Make requests - they'll be distributed across available providers
for i in range(5):
    try:
        response = balancer.generate(f"Tell me about topic number {i+1}")
        print(f"Response {i+1}: {response.content[:50]}...\n")
    except Exception as e:
        print(f"Request {i+1} failed: {e}\n")
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from abstractcore import create_llm, BasicSession
from typing import Optional
import uuid

app = FastAPI(title="AbstractCore API")

# Global LLM instance
llm = create_llm("openai", model="gpt-4o-mini")

# Store sessions in memory (use Redis in production)
sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get or create session
        if request.session_id and request.session_id in sessions:
            session = sessions[request.session_id]
        else:
            session_id = request.session_id or str(uuid.uuid4())
            session = BasicSession(
                provider=llm,
                system_prompt=request.system_prompt or "You are a helpful assistant."
            )
            sessions[session_id] = session

        # Generate response
        response = session.generate(request.message)

        return ChatResponse(
            response=response.content,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Gradio Web Interface

```python
import gradio as gr
from abstractcore import create_llm, BasicSession
from typing import List, Tuple

class ChatInterface:
    def __init__(self):
        self.llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
        self.session = BasicSession(
            provider=self.llm,
            system_prompt="You are a helpful AI assistant."
        )

    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Handle chat interaction."""
        try:
            response = self.session.generate(message)
            history.append((message, response.content))
            return "", history
        except Exception as e:
            history.append((message, f"Error: {str(e)}"))
            return "", history

    def clear(self) -> Tuple[str, List]:
        """Clear conversation history."""
        self.session.clear_history()
        return "", []

# Create interface
chat_interface = ChatInterface()

with gr.Blocks(title="AbstractCore Chat") as demo:
    gr.Markdown("# 🤖 AbstractCore Chat Interface")

    chatbot = gr.Chatbot(label="Conversation", height=400)
    msg = gr.Textbox(
        label="Message",
        placeholder="Type your message here...",
        lines=2
    )

    with gr.Row():
        submit = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear", variant="secondary")

    # Event handlers
    msg.submit(
        chat_interface.chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    submit.click(
        chat_interface.chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    clear.click(
        chat_interface.clear,
        outputs=[msg, chatbot]
    )

if __name__ == "__main__":
    demo.launch(share=True)
```

### Jupyter Notebook Integration

```python
# Cell 1: Setup
from abstractcore import create_llm
from IPython.display import display, Markdown, HTML
import json

# Create LLM instance
llm = create_llm("openai", model="gpt-4o-mini")

def display_response(response, title="AI Response"):
    """Pretty display for Jupyter notebooks."""
    html = f"""
    <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
        <h4 style="color: #333; margin-top: 0;">{title}</h4>
        <p style="line-height: 1.6;">{response.content}</p>
    </div>
    """
    display(HTML(html))

print("✅ AbstractCore setup complete!")

# Cell 2: Basic Usage
response = llm.generate("Explain quantum computing in simple terms")
display_response(response, "Quantum Computing Explanation")

# Cell 3: Structured Output
from pydantic import BaseModel
from typing import List

class LearningPlan(BaseModel):
    topic: str
    difficulty: str
    estimated_hours: int
    prerequisites: List[str]
    learning_steps: List[str]

plan = llm.generate(
    "Create a learning plan for someone who wants to learn machine learning",
    response_model=LearningPlan
)

# Display as nice table
display(HTML(f"""
<table style="border-collapse: collapse; width: 100%;">
    <tr><td><strong>Topic:</strong></td><td>{plan.topic}</td></tr>
    <tr><td><strong>Difficulty:</strong></td><td>{plan.difficulty}</td></tr>
    <tr><td><strong>Estimated Hours:</strong></td><td>{plan.estimated_hours}</td></tr>
    <tr><td><strong>Prerequisites:</strong></td><td>{', '.join(plan.prerequisites)}</td></tr>
</table>
"""))

display(Markdown("### Learning Steps:"))
for i, step in enumerate(plan.learning_steps, 1):
    display(Markdown(f"{i}. {step}"))
```

### Discord Bot Integration

```python
import discord
from discord.ext import commands
from abstractcore import create_llm, BasicSession
import asyncio

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# LLM setup
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
sessions = {}  # Store user sessions

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='ask')
async def ask(ctx, *, question):
    """Ask the AI a question."""
    user_id = ctx.author.id

    # Get or create session for user
    if user_id not in sessions:
        sessions[user_id] = BasicSession(
            provider=llm,
            system_prompt="You are a helpful Discord bot assistant. Keep responses concise."
        )

    try:
        # Show typing indicator
        async with ctx.typing():
            response = sessions[user_id].generate(question)

        # Discord has a 2000 character limit
        content = response.content
        if len(content) > 2000:
            content = content[:1997] + "..."

        await ctx.reply(content)

    except Exception as e:
        await ctx.reply(f"Sorry, I encountered an error: {str(e)}")

@bot.command(name='clear')
async def clear_session(ctx):
    """Clear your conversation history."""
    user_id = ctx.author.id
    if user_id in sessions:
        sessions[user_id].clear_history()
        await ctx.reply("Your conversation history has been cleared!")
    else:
        await ctx.reply("You don't have an active session to clear.")

@bot.command(name='stats')
async def stats(ctx):
    """Show session statistics."""
    user_id = ctx.author.id
    if user_id in sessions:
        session = sessions[user_id]
        message_count = len(session.messages)
        await ctx.reply(f"Your session has {message_count} messages.")
    else:
        await ctx.reply("You don't have an active session.")

# Run bot (add your Discord bot token)
# bot.run('YOUR_DISCORD_BOT_TOKEN')
```

## Next Steps

These examples show AbstractCore's versatility across different use cases. To continue learning:

1. **Start with basics** - Try the simple Q&A examples
2. **Add tools** - Experiment with the tool calling examples
3. **Structure output** - Use Pydantic models for type-safe responses
4. **Go production** - Implement error handling and monitoring
5. **Build apps** - Use the integration examples as starting points

For more information:
- [Getting Started](getting-started.md) - Basic setup and usage
- [Capabilities](capabilities.md) - What AbstractCore can do
- [Prerequisites](prerequisites.md) - Provider setup and configuration
- [API Reference](api-reference.md) - Complete API documentation

---

**Remember**: All these examples work with any provider - just change the `create_llm()` call to switch between OpenAI, Anthropic, Ollama, MLX, and others!