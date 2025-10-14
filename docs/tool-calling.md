# Tool Calling System

AbstractCore provides a universal tool calling system that works across **all** LLM providers, even those without native tool support. This is one of AbstractCore's most powerful features.

## Table of Contents

- [Quick Start](#quick-start)
- [The @tool Decorator](#the-tool-decorator)
- [Universal Tool Support](#universal-tool-support)
- [Tool Definition](#tool-definition)
- [Tool Execution](#tool-execution)
- [Advanced Patterns](#advanced-patterns)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Tool Syntax Rewriting](#tool-syntax-rewriting)
- [Event System Integration](#event-system-integration)

## Quick Start

The simplest way to create and use tools is with the `@tool` decorator:

```python
from abstractllm import create_llm, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a specified city."""
    # In a real scenario, you'd call an actual weather API
    return f"The weather in {city} is sunny, 72°F"

@tool
def calculate(expression: str) -> float:
    """Perform a mathematical calculation."""
    try:
        result = eval(expression)  # Simplified for demo - don't use eval in production!
        return result
    except Exception:
        return float('nan')

# Works with ANY provider
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "What's the weather in Tokyo and what's 15 * 23?",
    tools=[get_weather, calculate]  # Pass functions directly
)

print(response.content)
# Output: The weather in Tokyo is sunny, 72°F and 15 * 23 = 345.
```

## The @tool Decorator

The `@tool` decorator is the primary way to create tools in AbstractCore. It automatically extracts function metadata and creates proper tool definitions.

### Basic Usage

```python
from abstractllm import tool

@tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a directory matching a pattern."""
    import os
    import fnmatch
    
    try:
        files = [f for f in os.listdir(directory) 
                if fnmatch.fnmatch(f, pattern)]
        return "\n".join(files) if files else "No files found"
    except Exception as e:
        return f"Error: {str(e)}"
```

### Type Annotations

The decorator automatically infers parameter types from type annotations:

```python
@tool
def create_user(name: str, age: int, is_admin: bool = False) -> str:
    """Create a new user with the specified details."""
    user_data = {
        "name": name,
        "age": age,
        "is_admin": is_admin,
        "created_at": "2025-01-14"
    }
    return f"Created user: {user_data}"
```

### Enhanced Metadata

You can provide additional metadata to help the LLM understand when and how to use tools:

```python
@tool
def search_database(query: str, table: str = "users") -> str:
    """Search the database for records matching the query.
    
    This tool should be used when the user asks for specific data
    from the database or wants to find records.
    
    Examples:
    - "Find all users named John" -> query="name=John", table="users"
    - "Search for products under $50" -> query="price<50", table="products"
    """
    # Implementation here
    return f"Searching {table} for: {query}"
```

## Universal Tool Support

AbstractCore's tool system works across **all** providers through two mechanisms:

### 1. Native Tool Support

For providers with native tool APIs (OpenAI, Anthropic):

```python
# OpenAI with native tool support
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("What's the weather?", tools=[get_weather])
```

### 2. Intelligent Prompting

For providers without native tool support (Ollama, MLX, LMStudio):

```python
# Ollama without native tool support - AbstractCore handles this automatically
llm = create_llm("ollama", model="qwen3-coder:30b")
response = llm.generate("What's the weather?", tools=[get_weather])
# AbstractCore automatically adds tool descriptions to the prompt
```

## Tool Definition

Tools are defined using the `ToolDefinition` class, but the `@tool` decorator handles this automatically:

```python
from abstractllm.tools import ToolDefinition

# Manual tool definition (rarely needed)
tool_def = ToolDefinition(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "city": {
            "type": "string",
            "description": "The city name"
        }
    },
    function=get_weather_function
)
```

### Parameter Types

Supported parameter types:

- `string` - Text values
- `integer` - Whole numbers
- `number` - Floating-point numbers
- `boolean` - True/false values
- `array` - Lists of values
- `object` - Complex nested structures

```python
@tool
def complex_tool(
    text: str,
    count: int,
    threshold: float,
    enabled: bool,
    tags: list,
    config: dict
) -> str:
    """Tool with various parameter types."""
    return f"Processed: {text} with {count} items"
```

## Tool Execution

### Execution Flow

1. **LLM generates response** with tool calls
2. **AbstractCore detects** tool calls in the response
3. **Event system emits** `TOOL_STARTED` (preventable)
4. **Tools execute locally** in AbstractCore (not by provider)
5. **Results collected** and error handling applied
6. **Event system emits** `TOOL_COMPLETED` with results
7. **Results integrated** into the final response

### Tool Call Detection

AbstractCore automatically detects tool calls in various formats:

```python
# Different formats automatically detected:
# Qwen3: <|tool_call|>{"name": "get_weather", "arguments": {"city": "Paris"}}</|tool_call|>
# LLaMA3: <function_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</function_call>
# OpenAI: Structured JSON in API response
```

### Local Execution

All tools execute locally in AbstractCore, ensuring:

- **Consistent behavior** across all providers
- **Security control** through event system
- **Error handling** and validation
- **Performance monitoring**

## Advanced Patterns

### Tool Chaining

Tools can call other tools or return data that triggers additional tool calls:

```python
@tool
def get_user_location(user_id: str) -> str:
    """Get the location of a user."""
    # Mock implementation
    locations = {"user123": "Paris", "user456": "Tokyo"}
    return locations.get(user_id, "Unknown")

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

# LLM can chain these tools:
response = llm.generate(
    "What's the weather like for user123?",
    tools=[get_user_location, get_weather]
)
# LLM will first call get_user_location, then get_weather with the result
```

### Conditional Tool Execution

Use the event system to control tool execution:

```python
from abstractllm.events import EventType, on_global

def security_check(event):
    """Prevent execution of dangerous tools."""
    dangerous_tools = ['delete_file', 'system_command', 'send_email']
    
    for tool_call in event.data.get('tool_calls', []):
        if tool_call.name in dangerous_tools:
            print(f"Blocking dangerous tool: {tool_call.name}")
            event.prevent()  # Stop execution

on_global(EventType.TOOL_STARTED, security_check)
```

### Async Tool Support

For tools that need to perform async operations:

```python
import asyncio

@tool
def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    async def async_fetch():
        # Simulate async HTTP request
        await asyncio.sleep(0.1)
        return f"Data from {url}"
    
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(async_fetch())
        return result
    finally:
        loop.close()
```

## Error Handling

### Tool-Level Error Handling

Handle errors within tools:

```python
@tool
def safe_division(a: float, b: float) -> str:
    """Safely divide two numbers."""
    try:
        if b == 0:
            return "Error: Division by zero is not allowed"
        result = a / b
        return f"{a} ÷ {b} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### System-Level Error Handling

AbstractCore provides comprehensive error handling:

```python
from abstractllm.exceptions import ToolExecutionError

try:
    response = llm.generate("Use the broken tool", tools=[broken_tool])
except ToolExecutionError as e:
    print(f"Tool execution failed: {e}")
    print(f"Failed tool: {e.tool_name}")
    print(f"Error details: {e.error_details}")
```

### Validation and Sanitization

Validate tool inputs:

```python
@tool
def create_file(filename: str, content: str) -> str:
    """Create a file with the given content."""
    import os
    import re
    
    # Validate filename
    if not re.match(r'^[a-zA-Z0-9_.-]+$', filename):
        return "Error: Invalid filename. Use only letters, numbers, dots, dashes, and underscores."
    
    # Prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return "Error: Invalid filename. No directory traversal allowed."
    
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"File '{filename}' created successfully"
    except Exception as e:
        return f"Error creating file: {str(e)}"
```

## Performance Optimization

### Tool Registry

Use the tool registry for better performance with many tools:

```python
from abstractllm.tools import ToolRegistry, register_tool

# Register tools globally
register_tool(get_weather)
register_tool(calculate)
register_tool(list_files)

# Use registered tools
registry = ToolRegistry.get_instance()
available_tools = registry.get_all_tools()

response = llm.generate(
    "Help me with weather and calculations",
    tools=available_tools
)
```

### Lazy Loading

Load expensive resources only when needed:

```python
class DatabaseTool:
    def __init__(self):
        self._connection = None
    
    @property
    def connection(self):
        if self._connection is None:
            # Expensive database connection
            self._connection = create_database_connection()
        return self._connection

db_tool = DatabaseTool()

@tool
def query_database(sql: str) -> str:
    """Execute a SQL query."""
    try:
        result = db_tool.connection.execute(sql)
        return str(result)
    except Exception as e:
        return f"Database error: {str(e)}"
```

### Caching Results

Cache expensive tool results:

```python
from functools import lru_cache

@tool
@lru_cache(maxsize=100)
def expensive_calculation(input_data: str) -> str:
    """Perform an expensive calculation with caching."""
    import time
    time.sleep(1)  # Simulate expensive operation
    return f"Result for {input_data}"
```

## Tool Syntax Rewriting

AbstractCore can rewrite tool call formats for compatibility with different agentic CLIs:

```python
# For Codex CLI (Qwen3 format)
response = llm.generate(
    "What's the weather?", 
    tools=[get_weather],
    tool_call_tags="qwen3"
)
# Output: <|tool_call|>{"name": "get_weather", "arguments": {"city": "Paris"}}</|tool_call|>

# For Crush CLI (LLaMA3 format)
response = llm.generate(
    "What's the weather?", 
    tools=[get_weather],
    tool_call_tags="llama3"
)
# Output: <function_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</function_call>

# For Gemini CLI (XML format)
response = llm.generate(
    "What's the weather?", 
    tools=[get_weather],
    tool_call_tags="xml"
)
# Output: <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
```

## Event System Integration

Monitor and control tool execution through events:

### Cost Monitoring

```python
from abstractllm.events import EventType, on_global

def monitor_tool_costs(event):
    """Monitor costs of tool executions."""
    tool_calls = event.data.get('tool_calls', [])
    for tool_call in tool_calls:
        if tool_call.name in ['expensive_api_call', 'premium_service']:
            print(f"Warning: Using expensive tool {tool_call.name}")

on_global(EventType.TOOL_STARTED, monitor_tool_costs)
```

### Performance Tracking

```python
def track_tool_performance(event):
    """Track tool execution performance."""
    duration = event.data.get('duration_ms', 0)
    tool_name = event.data.get('tool_name', 'unknown')
    
    if duration > 5000:  # More than 5 seconds
        print(f"Slow tool execution: {tool_name} took {duration}ms")

on_global(EventType.TOOL_COMPLETED, track_tool_performance)
```

### Security Auditing

```python
def audit_tool_usage(event):
    """Audit all tool usage for security."""
    tool_calls = event.data.get('tool_calls', [])
    for tool_call in tool_calls:
        print(f"Tool used: {tool_call.name} with args: {tool_call.arguments}")
        # Log to security audit system
        security_log(tool_call.name, tool_call.arguments)

on_global(EventType.TOOL_STARTED, audit_tool_usage)
```

## Best Practices

### 1. Clear Documentation

Always provide clear docstrings for your tools:

```python
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient.
    
    Args:
        to: Email address of the recipient
        subject: Subject line of the email
        body: Main content of the email
    
    Returns:
        Success message or error description
    
    Note:
        This tool requires email configuration to be set up.
        Use with caution as it sends actual emails.
    """
    # Implementation here
```

### 2. Input Validation

Always validate and sanitize inputs:

```python
@tool
def process_user_input(user_data: str) -> str:
    """Process user input safely."""
    # Validate input length
    if len(user_data) > 1000:
        return "Error: Input too long (max 1000 characters)"
    
    # Sanitize input
    import html
    safe_data = html.escape(user_data)
    
    # Process safely
    return f"Processed: {safe_data}"
```

### 3. Error Recovery

Provide meaningful error messages and recovery suggestions:

```python
@tool
def connect_to_service(endpoint: str) -> str:
    """Connect to an external service."""
    try:
        # Attempt connection
        result = make_connection(endpoint)
        return f"Connected successfully: {result}"
    except ConnectionError:
        return "Error: Could not connect to service. Please check the endpoint URL and try again."
    except TimeoutError:
        return "Error: Connection timed out. The service may be temporarily unavailable."
    except Exception as e:
        return f"Error: Unexpected error occurred: {str(e)}"
```

### 4. Resource Management

Clean up resources properly:

```python
@tool
def process_large_file(filename: str) -> str:
    """Process a large file efficiently."""
    try:
        with open(filename, 'r') as file:
            # Process file in chunks
            result = ""
            while True:
                chunk = file.read(1024)
                if not chunk:
                    break
                result += process_chunk(chunk)
        return f"Processed file: {filename}"
    except FileNotFoundError:
        return f"Error: File '{filename}' not found"
    except MemoryError:
        return "Error: File too large to process"
```

## Troubleshooting

### Common Issues

1. **Tool not being called**: Check tool description and parameter names
2. **Invalid JSON in tool calls**: Ensure proper error handling in tools
3. **Tools timing out**: Implement proper timeout handling
4. **Memory issues with large tools**: Use streaming or chunking

### Debug Mode

Enable debug mode to see tool execution details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Tool execution details will be logged
response = llm.generate("Use tools", tools=[debug_tool])
```

### Testing Tools

Test tools independently:

```python
# Test tool directly
result = get_weather("Paris")
print(f"Direct call result: {result}")

# Test with LLM
response = llm.generate("What's the weather in Paris?", tools=[get_weather])
print(f"LLM result: {response.content}")
```

## Examples

See the [examples directory](../examples/) for comprehensive tool usage examples:

- [Basic Tool Usage](../examples/tool_usage_basic.py)
- [Advanced Tool Patterns](../examples/tool_usage_advanced.py)
- [Tool Chaining Examples](../examples/progressive/example_3_tool_calling.py)

## Related Documentation

- [API Reference](api-reference.md) - Complete API documentation
- [Tool Syntax Rewriting](tool-syntax-rewriting.md) - Format conversion details
- [Event System](api-reference.md#event-system) - Event-driven tool control
- [Architecture](architecture.md) - System design and tool execution flow
