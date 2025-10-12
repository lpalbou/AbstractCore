# Getting Started with AbstractCore

This guide will get you up and running with AbstractCore in 5 minutes. You'll learn how to install it, make your first LLM call, and explore the key features.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) API keys for cloud providers

## Installation

### Option 1: Start with a Cloud Provider (Recommended)

If you have an API key for OpenAI or Anthropic:

```bash
# For OpenAI
pip install abstractcore[openai]
export OPENAI_API_KEY="your-api-key-here"

# For Anthropic
pip install abstractcore[anthropic]
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Option 2: Start with Local Models (No API keys needed)

For privacy or cost reasons:

```bash
# Install Ollama first
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b-instruct-2507-q4_K_M

# Then install AbstractCore
pip install abstractcore
```

### Option 3: Install Everything

```bash
pip install abstractcore[all]
```

## Your First Program

Create a file called `first_llm.py`:

```python
from abstractllm import create_llm

# Choose your provider (uncomment one):
llm = create_llm("openai", model="gpt-4o-mini")        # Cloud
# llm = create_llm("anthropic", model="claude-3-5-haiku-latest")  # Cloud
# llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")   # Local

# Generate your first response
response = llm.generate("What is the capital of France?")
print(response.content)
```

Run it:

```bash
python first_llm.py
# Output: The capital of France is Paris.
```

**🎉 Congratulations!** You've made your first AbstractCore LLM call.

## Core Concepts (5-Minute Tour)

### 1. Providers and Models

AbstractCore supports multiple providers with the same interface:

```python
from abstractllm import create_llm

# Same interface, different providers
openai_llm = create_llm("openai", model="gpt-4o-mini")
claude_llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
local_llm = create_llm("ollama", model="qwen3-coder:30b")

question = "Explain Python list comprehensions"

# All work the same way
for name, llm in [("OpenAI", openai_llm), ("Claude", claude_llm), ("Ollama", local_llm)]:
    response = llm.generate(question)
    print(f"{name}: {response.content[:50]}...")
```

### 2. Structured Output (Game Changer)

Instead of parsing strings, get typed objects directly:

```python
from pydantic import BaseModel
from abstractllm import create_llm

class MovieReview(BaseModel):
    title: str
    rating: int  # 1-5 stars
    summary: str

llm = create_llm("openai", model="gpt-4o-mini")

# Get structured data automatically
review = llm.generate(
    "Review the movie Inception",
    response_model=MovieReview
)

print(f"Title: {review.title}")
print(f"Rating: {review.rating}/5")
print(f"Summary: {review.summary}")
```

**No more string parsing!** AbstractCore handles JSON validation and retries automatically.

### 3. Tool Calling (LLM with Superpowers)

Let your LLM call functions with the `@tool` decorator:

```python
from abstractllm import create_llm, tool
from abstractllm.tool import ToolDefinition

# Simple way to create tools: Use @tool decorator
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
        return float('nan')  # Return NaN for invalid calculations

# Instantiate the LLM
llm = create_llm("openai", model="gpt-4o-mini")

# Automatically extract tool definitions from decorated functions
response = llm.generate(
    "What's the weather in Tokyo and what's 15 * 23?",
    tools=[get_weather, calculate]  # Pass tool functions directly
)

print(response.content)
# Output: The weather in Tokyo is sunny, 72°F and 15 * 23 = 345.

# Advanced: Manually get tool definitions if needed
weather_tool: ToolDefinition = get_weather.tool_definition
calc_tool: ToolDefinition = calculate.tool_definition

# Inspect tool definitions
print(weather_tool.name)  # "get_weather"
print(weather_tool.description)  # "Get current weather for a specified city."
```

> **🏷️ Advanced Tool Features**:
> - `@tool` automatically generates ToolDefinition
> - Supports complex type hints and docstrings
> - Automatic parameter extraction
> - See [Tool Call Syntax Rewriting](tool-syntax-rewriting.md) for agentic CLI compatibility

### 4. Streaming (Real-Time Responses)

Show responses as they're generated:

```python
from abstractllm import create_llm

llm = create_llm("openai", model="gpt-4o-mini")

print("AI: ", end="", flush=True)
for chunk in llm.generate("Write a haiku about programming", stream=True):
    print(chunk.content, end="", flush=True)
print("\n")
# Output appears word by word in real-time
```

### 5. Conversations with Memory

Maintain context across multiple turns:

```python
from abstractllm import create_llm, BasicSession

llm = create_llm("openai", model="gpt-4o-mini")
session = BasicSession(provider=llm, system_prompt="You are a helpful coding tutor.")

# First exchange
response1 = session.generate("My name is Alex and I'm learning Python.")
print("AI:", response1.content)

# Second exchange - remembers context
response2 = session.generate("What's my name and what am I learning?")
print("AI:", response2.content)
# Output: Your name is Alex and you're learning Python.
```

## Provider Quick Setup

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
```
```python
llm = create_llm("openai", model="gpt-4o-mini")
```

### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
```python
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
```

### Ollama (Local)
```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```
```python
llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
```

### MLX (Apple Silicon)
```bash
pip install abstractcore[mlx]
```
```python
llm = create_llm("mlx", model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
```

## Common Patterns

### Pattern 1: Error Handling

```python
from abstractllm import create_llm
from abstractllm.exceptions import ProviderAPIError, RateLimitError

llm = create_llm("openai", model="gpt-4o-mini")

try:
    response = llm.generate("Hello world")
    print(response.content)
except RateLimitError:
    print("Rate limited - wait a moment")
except ProviderAPIError as e:
    print(f"API error: {e}")
```

### Pattern 2: Provider Fallback

```python
from abstractllm import create_llm

providers = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("ollama", "qwen3-coder:30b")
]

def generate_with_fallback(prompt):
    for provider, model in providers:
        try:
            llm = create_llm(provider, model=model)
            return llm.generate(prompt)
        except Exception as e:
            print(f"{provider} failed: {e}")
            continue
    raise Exception("All providers failed")

response = generate_with_fallback("Hello world")
```

### Pattern 3: Configuration

```python
from abstractllm import create_llm

llm = create_llm(
    "openai",
    model="gpt-4o-mini",
    temperature=0.7,        # Creativity (0-2)
    max_tokens=1000,        # Response length limit
    timeout=30              # Request timeout
)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'openai'"
**Solution**: Install the provider-specific package:
```bash
pip install abstractcore[openai]
```

### Issue: "Authentication Error"
**Solution**: Check your API key is set:
```bash
echo $OPENAI_API_KEY  # Should show your key
```

### Issue: Ollama connection error
**Solution**: Make sure Ollama is running:
```bash
ollama serve  # Start the server
ollama list   # Check available models
```

### Issue: "No model specified"
**Solution**: Always specify a model:
```python
# Wrong
llm = create_llm("openai")  # Missing model

# Correct
llm = create_llm("openai", model="gpt-4o-mini")
```

### Issue: Structured output validation errors
**Solution**: AbstractCore automatically retries, but check your Pydantic model:
```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    age: int

    @field_validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v
```

## Interactive CLI Tool

For quick testing and exploration, AbstractCore includes a basic CLI tool:

**[📋 Complete Internal CLI Guide →](internal-cli.md)**

```bash
# Interactive chat with any provider
python -m abstractllm.utils.cli --provider ollama --model qwen3-coder:30b

# Quick one-off questions
python -m abstractllm.utils.cli --provider openai --model gpt-4o-mini --prompt "What is Python?"

# With streaming mode for real-time responses
python -m abstractllm.utils.cli --provider anthropic --model claude-3-5-haiku-latest --stream

# With web search capabilities
# Try: "search for today's AI news" or "find recent Python tutorials"
```

**Available Commands:**
- `/help` - Show available commands
- `/quit` - Exit the CLI
- `/clear` - Clear conversation history
- `/stream` - Toggle streaming mode
- `/debug` - Toggle debug output
- `/history [n]` - Show conversation history or last n interactions
- `/model <provider:model>` - Switch providers/models
- `/compact` - Compact chat history using fast local model
- `/facts [file]` - Extract facts from conversation (display or save as JSON-LD)
- `/system [prompt]` - Show or change system prompt

**Built-in Tools:**

AbstractCore includes a comprehensive set of tools for file operations, web search, and system commands:

- **`list_files`** - Find and list files by name/path with pattern matching (case-insensitive, glob patterns, recursive)
- **`search_files`** - Search for text patterns INSIDE files using regex (grep-like functionality, multiple output modes)
- **`read_file`** - Read file contents with optional line range selection
- **`write_file`** - Write or append content to files (creates directories automatically)
- **`edit_file`** - Edit files using pattern matching and replacement (supports regex, preview mode)
- **`execute_command`** - Execute shell commands safely with security controls and platform detection
- **`web_search`** - Search the web with DuckDuckGo (no API key required, supports time filtering!)

**For detailed tool documentation and examples, see:** [`abstractllm/tools/common_tools.py`](../abstractllm/tools/common_tools.py)

> **Note**: This CLI is a basic demonstrator. For production applications requiring complex reasoning or advanced agent behaviors, build custom solutions using the AbstractCore framework directly.

## What's Next?

Now that you have the basics:

1. **[Explore Examples](examples.md)** - Real-world use cases and patterns
2. **[Tool Call Syntax Rewriting](tool-syntax-rewriting.md)** - Format conversion for agentic CLI compatibility
3. **[Set Up Server & Agentic CLIs](server.md)** - Universal API server, Codex/Gemini CLI integration
4. **[Provider Setup](prerequisites.md)** - Detailed provider configuration
5. **[Use Internal CLI](internal-cli.md)** - Built-in testing CLI with advanced features
6. **[Understand Capabilities](capabilities.md)** - What AbstractCore can and cannot do
7. **[Read the API Reference](api-reference.md)** - Complete API documentation
8. **[Check Advanced Features](../README.md#core-features)** - Embeddings, events, retry logic

## Getting Help

- **Documentation**: All docs are in the `docs/` folder
- **Examples**: See `docs/examples.md` for copy-paste code
- **Issues**: [GitHub Issues](https://github.com/lpalbou/AbstractCore/issues) for bugs
- **Discussions**: [GitHub Discussions](https://github.com/lpalbou/AbstractCore/discussions) for questions

---

**You're ready to build with AbstractCore!** 🚀

The key insight: AbstractCore gives you **the same simple interface** across all LLM providers, with **production-grade reliability** built-in. Focus on building your application, not managing API differences.