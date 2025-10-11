# AbstractLLM Core: 10 Common Mistakes and How to Fix Them

## Quick Reference: Top 3 Critical Mistakes

1. **🔑 Incorrect Provider Configuration**
   - *Symptom*: Authentication failures, no model response
   - *Quick Fix*: Always set API keys as environment variables

2. **🧩 Mishandling Tool Calls**
   - *Symptom*: Tools not executing, streaming interruptions
   - *Quick Fix*: Use `stream_tools=True` and handle `chunk.tools`

3. **💻 Provider Dependency Confusion**
   - *Symptom*: `ModuleNotFoundError` for providers
   - *Quick Fix*: Install provider-specific packages with `pip install abstractllm[provider]`

## 1. Provider and Configuration Mistakes

### Mistake: Missing or Incorrect API Keys
**You'll See This**:
- `ProviderAPIError: Authentication failed`
- No response from the model
- Cryptic error messages about credentials

**Why This Happens**:
- API keys not set as environment variables
- Whitespace or copying errors in key
- Incorrect key permissions or expired credentials

**Quick Fix**:
```bash
# Set API key correctly
export OPENAI_API_KEY="sk-proj-..."  # No extra spaces!
export ANTHROPIC_API_KEY="sk-ant-..."

# Validate key in AbstractLLM
python -m abstractllm.utils.cli --provider openai --validate-key
```

**How to Avoid**:
- Use environment variables for sensitive credentials
- Store keys in `.env` files (add to `.gitignore`)
- Regularly rotate and update API keys
- Use secret management tools for production

### Mistake: Incorrect Model Selection
**You'll See This**:
- `ValueError: Model not found`
- Unexpected behavior with model capabilities
- Inconsistent responses across providers

**Why This Happens**:
- Using outdated or unsupported model names
- Not checking provider-specific model compatibility
- Misunderstanding model capabilities

**Quick Fix**:
```python
# Always specify exact, current model names
llm = create_llm("openai", model="gpt-4o-mini")
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")

# Check available models
python -m abstractllm.utils.cli --provider ollama --list-models
```

**How to Avoid**:
- Regularly check the [model capabilities documentation](/docs/capabilities.md)
- Use the CLI to list available models
- Design code to handle model fallback gracefully

## 2. Implementation Pattern Mistakes

### Mistake: Incorrect Tool Call Handling
**You'll See This**:
- Tools not executing during generation
- Partial or missing tool call results
- Streaming interruptions

**Why This Happens**:
- Not enabling tool streaming
- Incorrect tool definition
- Complex tool call formats not handled

**Quick Fix**:
```python
def get_weather(city: str):
    return f"Weather in {city}: 72°F, Sunny"

tools = [{
    "name": "get_weather",
    "description": "Get current weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}]

response = llm.generate(
    "What's the weather in Tokyo?",
    tools=tools,
    stream_tools=True  # CRITICAL: Enable tool streaming
)

for chunk in response:
    print(chunk.content, end="")  # Stream content
    if chunk.tools:
        for tool in chunk.tools:
            print(f"Tool executed: {tool.name}")
```

**How to Avoid**:
- Always include `stream_tools=True` for streaming
- Define tools with clear, type-hinted functions
- Handle potential tool execution errors
- Use type validation with Pydantic models

### Mistake: Overlooking Error Handling
**You'll See This**:
- Unhandled exceptions
- Silent failures in tool or generation calls
- Unexpected application crashes

**Why This Happens**:
- Not catching provider-specific exceptions
- Assuming 100% reliability of LLM responses
- No retry or fallback mechanisms

**Quick Fix**:
```python
from abstractllm import create_llm
from abstractllm.exceptions import (
    ProviderAPIError,
    RateLimitError,
    ModelUnavailableError
)

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
        except (ProviderAPIError, RateLimitError) as e:
            print(f"Failed with {provider}: {e}")
            continue
    raise Exception("All providers failed")
```

**How to Avoid**:
- Always use try/except blocks
- Implement provider fallback strategies
- Log and monitor errors systematically
- Design for graceful degradation

## 3. Production and Advanced Usage Mistakes

### Mistake: Memory and Performance Bottlenecks
**You'll See This**:
- High memory consumption
- Slow response times
- Out-of-memory errors during long generations

**Why This Happens**:
- Not managing token limits
- Generating overly long responses
- Inefficient streaming configurations

**Quick Fix**:
```python
# Optimize memory and performance
response = llm.generate(
    "Complex task",
    max_tokens=1000,  # Limit response length
    timeout=30,       # Set reasonable timeout
    low_memory=True,  # Optimize memory usage
    temperature=0.7   # Control creativity/randomness
)

# For very large tasks, use batching
responses = llm.generate_batch(
    prompts=["Task 1", "Task 2", "Task 3"],
    batch_size=16,    # Parallel processing
    max_tokens=500   # Per-task limit
)
```

**How to Avoid**:
- Always set `max_tokens`
- Use `low_memory=True` for memory-intensive tasks
- Implement batching for multiple tasks
- Profile and monitor generation performance

### Mistake: Hardcoding Credentials and Configurations
**You'll See This**:
- Exposed API keys in code
- Inflexible configuration management
- Security vulnerabilities

**Why This Happens**:
- Copying example code directly
- Not understanding configuration best practices
- Lack of environment-based configuration

**Quick Fix**:
```python
# Use environment-based configuration
import os
from abstractllm import create_llm

# Best practice: Load from environment or config file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_MODEL = os.getenv('DEFAULT_LLM_MODEL', 'gpt-4o-mini')

llm = create_llm(
    "openai",
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY
)
```

**How to Avoid**:
- Never hardcode API keys or sensitive data
- Use environment variables
- Implement configuration management libraries
- Follow 12-factor app configuration principles

## Conclusion

Understanding these common mistakes helps developers improve their implementations.

## Resources
- [Documentation](https://abstractllm.ai/docs)
- [Community Discussions](https://github.com/lpalbou/AbstractCore/discussions)

## Support
- Issues: [GitHub Issues](https://github.com/lpalbou/AbstractCore/issues)
- Discussions: [GitHub Discussions](https://github.com/lpalbou/AbstractCore/discussions)

---

Last Updated: 2025-10-11
Version: 2.5.0