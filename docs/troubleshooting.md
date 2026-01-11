# AbstractCore Troubleshooting Guide

Complete troubleshooting guide for AbstractCore core library and server, including common mistakes and how to avoid them.

## Table of Contents

- [Common Mistakes to Avoid](#common-mistakes-to-avoid)
- [Quick Diagnosis](#quick-diagnosis)
- [Installation Issues](#installation-issues)
- [Core Library Issues](#core-library-issues)
- [Server Issues](#server-issues)
- [Provider-Specific Issues](#provider-specific-issues)
- [Performance Issues](#performance-issues)
- [Best Practices](#best-practices)
- [Debug Techniques](#debug-techniques)

---

## Common Mistakes to Avoid

Understanding common pitfalls helps prevent issues before they occur.

### Top 3 Critical Mistakes

1. **🔑 Incorrect Provider Configuration**
   - *Symptom*: Authentication failures, no model response
   - *Quick Fix*: Always set API keys as environment variables
   - See: [Authentication Errors](#issue-authentication-errors)

2. **🧩 Mishandling Tool Calls**
   - *Symptom*: Tools not executing, streaming interruptions
   - *Quick Fix*: Use `@tool` decorator and handle tool calls properly
   - See: [Tool Calls Not Working](#issue-tool-calls-not-working)

3. **💻 Provider Dependency Confusion**
   - *Symptom*: `ModuleNotFoundError` for providers
   - *Quick Fix*: Install provider-specific packages with `pip install abstractcore[provider]`
   - See: [ModuleNotFoundError](#issue-modulenotfounderror)

4. **🖥️ LM Studio Server Not Enabled**
   - *Symptom*: Connection refused, no response from LM Studio
   - *Quick Fix*: Enable "Status: Running" toggle in LM Studio GUI
   - See: [LM Studio Server Not Enabled](#issue-lm-studio-server-not-enabled)

5. **📏 Context Length Too Small (LM Studio/Ollama)**
   - *Symptom*: 400 Bad Request, truncated responses, errors with long inputs
   - *Quick Fix*: Set "Default Context Length" to "Model Maximum" in LM Studio
   - See: [Context Length Too Small](#issue-context-length-too-small-400-bad-request-truncated-responses)

### Common Mistake Patterns

#### Mistake: Missing or Incorrect API Keys

**You'll See:**
- `ProviderAPIError: Authentication failed`
- No response from the model
- Cryptic error messages about credentials

**Why This Happens:**
- API keys not set as environment variables
- Whitespace or copying errors in key
- Incorrect key permissions or expired credentials

**Solution:** See [Authentication Errors](#issue-authentication-errors) for complete fix.

**Prevention:**
- Use environment variables for sensitive credentials
- Store keys in `.env` files (add to `.gitignore`)
- Regularly rotate and update API keys
- Use secret management tools for production

#### Mistake: Incorrect Tool Call Handling

**You'll See:**
- Tools not executing during generation
- Partial or missing tool call results
- Streaming interruptions

**Why This Happens:**
- Not using `@tool` decorator
- Incorrect tool definition format
- Not handling tool responses

**Solution:**
```python
from abstractcore import create_llm, tool

# Use @tool decorator for automatic tool definition
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: sunny, 72°F"

llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "What's the weather in Tokyo?",
    tools=[get_weather]  # Pass decorated function directly
)
```

**Prevention:**
- Always use `@tool` decorator for automatic tool definitions
- Use type hints for all parameters
- Add clear docstrings for tool descriptions
- Handle tool execution errors gracefully
- See: [Tool Calls Not Working](#issue-tool-calls-not-working)

#### Mistake: Overlooking Error Handling

**You'll See:**
- Unhandled exceptions
- Silent failures in tool or generation calls
- Unexpected application crashes

**Why This Happens:**
- Not catching provider-specific exceptions
- Assuming 100% reliability of LLM responses
- No retry or fallback mechanisms

**Solution:**
```python
from abstractcore import create_llm
from abstractcore.exceptions import ProviderAPIError, RateLimitError

providers = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-haiku-4-5"),
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

**Prevention:**
- Always use try/except blocks
- Implement provider fallback strategies
- Log and monitor errors systematically
- Design for graceful degradation

#### Mistake: Memory and Performance Bottlenecks

**You'll See:**
- High memory consumption
- Slow response times
- Out-of-memory errors during long generations

**Why This Happens:**
- Not managing token limits
- Generating overly long responses
- Inefficient streaming configurations

**Solution:**
```python
# Optimize memory and performance
response = llm.generate(
    "Complex task",
    max_tokens=1000,  # Limit response length
    timeout=30,       # Set reasonable timeout
    temperature=0.7   # Control creativity/randomness
)
```

**Prevention:**
- Always set `max_tokens`
- Use streaming for long responses
- Monitor memory usage in production
- See: [Performance Issues](#performance-issues)

#### Mistake: Hardcoding Credentials

**You'll See:**
- Exposed API keys in code
- Inflexible configuration management
- Security vulnerabilities

**Why This Happens:**
- Copying example code directly
- Not understanding configuration best practices
- Lack of environment-based configuration

**Solution:**
```python
import os
from abstractcore import create_llm

# Best practice: Load from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEFAULT_MODEL = os.getenv('DEFAULT_LLM_MODEL', 'gpt-4o-mini')

llm = create_llm(
    "openai",
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY
)
```

**Prevention:**
- Never hardcode API keys or sensitive data
- Use environment variables
- Implement configuration management libraries
- Follow 12-factor app configuration principles

---

## Quick Diagnosis

Run these checks first:

```bash
# Check Python version
python --version  # Should be 3.9+

# Check AbstractCore installation
pip show abstractcore

# Test core library
python -c "from abstractcore import create_llm; print('✓ Core library OK')"

# Test server (if installed)
curl http://localhost:8000/health  # Should return {"status":"healthy"}
```

---

## Installation Issues

### Issue: ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named .abstractcore.
ModuleNotFoundError: No module named 'openai'
```

**Solutions:**
```bash
# Install AbstractCore
pip install abstractcore

# Install with specific provider
pip install abstractcore[openai]
pip install abstractcore[anthropic]
pip install abstractcore[ollama]

# Install the full feature set (pick one)
pip install abstractcore[all-apple]    # macOS/Apple Silicon (includes MLX, excludes vLLM)
pip install abstractcore[all-non-mlx]  # Linux/Windows/Intel Mac (excludes MLX and vLLM)
pip install abstractcore[all-gpu]      # Linux NVIDIA GPU (includes vLLM, excludes MLX)

# Verify installation
pip list | grep abstract
```

### Issue: Dependency Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

**Solutions:**
```bash
# Create clean environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows

# Fresh install
pip install --upgrade pip
pip install abstractcore[all-apple]    # macOS/Apple Silicon
# or: pip install abstractcore[all-non-mlx]  # Linux/Windows/Intel Mac
# or: pip install abstractcore[all-gpu]      # Linux NVIDIA GPU

# If still failing, try one provider at a time
pip install abstractcore[openai]
```

---

## Core Library Issues

### Issue: Authentication Errors

**Symptoms:**
```
Error: OpenAI API key not found
Error: Authentication failed
Error: Invalid API key
```

**Solutions:**

```bash
# Check if API key is set
echo $OPENAI_API_KEY  # Should show your key
echo $ANTHROPIC_API_KEY

# Set API key
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Add to shell profile for persistence
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc

# Verify key format
# OpenAI: starts with "sk-"
# Anthropic: starts with "sk-ant-"

# Test authentication
python -c "from abstractcore import create_llm; llm = create_llm('openai', model='gpt-4o-mini'); print(llm.generate('test').content)"
```

### Issue: Model Not Found

**Symptoms:**
```
Error: Model 'qwen3-coder:30b' not found
Error: Unsupported model
```

**Solutions:**

**For Ollama:**
```bash
# Check available models
ollama list

# Pull missing model
ollama pull qwen3-coder:30b

# Verify Ollama is running
ollama serve
```

**For LMStudio:**
```bash
# Check LMStudio server
curl http://localhost:1234/v1/models

# In LMStudio GUI:
# 1. Go to "Local Server" tab
# 2. Select model from dropdown
# 3. Click "Start Server"
```

**For OpenAI/Anthropic:**
```python
# Use correct model names
llm = create_llm("openai", model="gpt-4o-mini")  # ✓ Correct
llm = create_llm("openai", model="gpt4")  # ✗ Wrong

llm = create_llm("anthropic", model="claude-haiku-4-5")  # ✓ Correct
llm = create_llm("anthropic", model="claude-3")  # ✗ Wrong
```

### Issue: Connection Errors

**Symptoms:**
```
Connection refused
Timeout error
Network error
```

**Solutions:**

**For Ollama:**
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags

# If using custom host
export OLLAMA_HOST="http://localhost:11434"
```

**For LMStudio:**
```bash
# Verify server is running
curl http://localhost:1234/v1/models

# Check port in LMStudio GUI (usually 1234)
```

**For Cloud Providers:**
```bash
# Test network connection
ping api.openai.com
ping api.anthropic.com

# Check proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Disable proxy if needed
unset HTTP_PROXY
unset HTTPS_PROXY
```

### Issue: Tool Calls Not Working

**Symptoms:**
- Tools not being called
- Empty tool responses
- Tool format errors

**Solutions:**

```python
from abstractcore import create_llm, tool

# Ensure @tool decorator is used
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: sunny, 72°F"

# Use tool correctly
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]  # Pass as list
)

# Check if tool was called
if hasattr(response, 'tool_calls') and response.tool_calls:
    print("Tools were called")
```

---

## Server Issues

### Issue: Server Won't Start

**Symptoms:**
```
Address already in use
Port 8000 is already allocated
```

**Solutions:**

```bash
# Check what's using port 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process on port
kill -9 $(lsof -t -i:8000)  # Linux/Mac

# Use different port
uvicorn abstractcore.server.app:app --port 3000
```

### Issue: ABSTRACTCORE_API_KEY Error

**Symptoms:**
```
Error: Missing environment variable: ABSTRACTCORE_API_KEY
```

**Solutions:**

```bash
# Set the required variable
export ABSTRACTCORE_API_KEY="unused"

# For Codex CLI, set ALL three:
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
export ABSTRACTCORE_API_KEY="unused"

# Verify they're set
echo $OPENAI_BASE_URL
echo $OPENAI_API_KEY
echo $ABSTRACTCORE_API_KEY
```

### Issue: Server Running but No Response

**Symptoms:**
- curl hangs
- No response from endpoints
- Timeout errors

**Solutions:**

```bash
# Check server is actually running
curl http://localhost:8000/health

# Check server logs
tail -f logs/abstractcore_*.log

# Enable debug mode
export ABSTRACTCORE_DEBUG=true
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# Test with simple request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]}'
```

### Issue: Models Not Showing

**Symptoms:**
```
curl http://localhost:8000/v1/models returns empty list
```

**Solutions:**

```bash
# Check if providers are configured
curl http://localhost:8000/providers

# Verify provider setup:

# For Ollama
ollama list  # Should show models
ollama serve  # Make sure it's running

# For OpenAI
echo $OPENAI_API_KEY  # Should be set

# For Anthropic
echo $ANTHROPIC_API_KEY  # Should be set

# For LMStudio
curl http://localhost:1234/v1/models  # Should return models
```

### Issue: Tool Calls Not Working with CLI

**Symptoms:**
- Codex/Crush/Gemini CLI not detecting tools
- Tool format errors in streaming

**Solutions:**

```bash
# Set correct tool format for your CLI

# For Codex CLI (qwen3 format - default)
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# For Crush CLI (llama3 format)
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# For Gemini CLI (xml format)
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# Restart server after changing environment variables
pkill -f "abstractcore.server.app"
```

---

## Provider-Specific Issues

### Ollama

**Issue: Ollama not responding**
```bash
# Restart Ollama
pkill ollama
ollama serve

# Check status
curl http://localhost:11434/api/tags

# List models
ollama list

# Pull model if missing
ollama pull qwen3-coder:30b
```

**Issue: Out of memory**
```bash
# Use smaller models
ollama pull gemma3:1b  # Only 1GB
ollama pull qwen3:4b-instruct-2507-q4_K_M  # 4GB

# Check system memory
free -h  # Linux
vm_stat  # macOS

# Close other applications
```

### OpenAI

**Issue: Rate limits**
```bash
# Check your rate limits
# https://platform.openai.com/account/rate-limits

# Implement backoff in code
import time
try:
    response = llm.generate("prompt")
except RateLimitError:
    time.sleep(20)  # Wait before retry
```

**Issue: Billing**
```bash
# Check billing dashboard
# https://platform.openai.com/account/billing

# Verify payment method is added
# Check usage limits aren't exceeded
```

### Anthropic

**Issue: API key format**
```bash
# Anthropic keys start with "sk-ant-"
echo $ANTHROPIC_API_KEY  # Should start with sk-ant-

# Get key from console
# https://console.anthropic.com/
```

### LMStudio

**Issue: Connection refused**
```bash
# Verify LMStudio server is running
# Check LMStudio GUI shows "Server running"

# Test connection
curl http://localhost:1234/v1/models

# Check port number in LMStudio (usually 1234)
```

**Issue: LM Studio Server Not Enabled**
```bash
# CRITICAL: Ensure LM Studio server is enabled in the GUI
# 1. Open LM Studio application
# 2. Look for "Status: Running" toggle switch in the interface
# 3. Make sure the toggle is switched to "ON" (green background, white handle on right)
# 4. If the toggle shows "OFF", click it to enable the server
# 5. Verify the server is running by checking the status indicator

# Test server availability
curl http://localhost:1234/v1/models

# If still failing, check LM Studio logs for any error messages
```

**Issue: Context Length Too Small (400 Bad Request, Truncated Responses)**
```bash
# Problem: LLM returns 400 Bad Request, truncated output, or errors with long inputs
# Root Cause: Insufficient context length configured for the model or server

# Solution 1: Increase Default Context Length (RECOMMENDED)
# This is the most robust way to ensure all models use maximum available context
# 1. Open LM Studio application
# 2. Go to "App Settings" → "General" tab
# 3. Find "Model Defaults" → "Default Context Length"
# 4. Set dropdown to "Model Maximum" (or highest available value like 131072)
# 5. Restart LM Studio server for changes to take effect

# Solution 2: Increase Context Length per Model (Alternative)
# This method applies context length setting to a specific model
# 1. Open LM Studio application
# 2. Go to "My Models" tab
# 3. Select the specific model you are using
# 4. Look for "Context Length" slider/input (usually under "Load" or "Context" tab)
# 5. Adjust slider to maximum value (e.g., 131072 tokens)
# 6. Reload the model for changes to take effect

# Solution 3: Increase Context Length via API Request (Advanced)
# For Ollama, or if you need to override settings for LM Studio via API
# For Ollama:
ollama run <model_name> -c <context_length>
# Example: ollama run llama2 -c 4096

# For LM Studio via API (often handled automatically by AbstractCore):
# Include in request payload:
# {
#   "model": "your-model-name",
#   "prompt": "Your long prompt here...",
#   "options": {
#     "num_ctx": 4096  # Or your desired context length
#   }
# }

# Verification:
# After adjusting, test with a long prompt that previously failed
# Check server logs for any warnings or errors related to context
```

---

## Performance Issues

### Issue: Slow Responses

**Diagnosis:**
```bash
# Time a request
time python -c "from abstractcore import create_llm; llm = create_llm('ollama', model='qwen3:4b-instruct-2507-q4_K_M'); print(llm.generate('test').content)"
```

**Solutions:**

**Use Faster Models:**
```python
# Faster cloud models
llm = create_llm("openai", model="gpt-4o-mini")  # Fast
llm = create_llm("anthropic", model="claude-haiku-4-5")  # Fast

# Faster local models
llm = create_llm("ollama", model="gemma3:1b")  # Very fast
llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")  # Balanced
```

**Enable Streaming:**
```python
# Improves perceived speed
for chunk in llm.generate("Long response", stream=True):
    print(chunk.content, end="", flush=True)
```

**Optimize Parameters:**
```python
response = llm.generate(
    "prompt",
    max_tokens=500,      # Limit length
    temperature=0.3      # Lower = faster
)
```

### Issue: High Memory Usage

**Solutions:**

```bash
# Use smaller models
ollama pull gemma3:1b  # 1GB instead of 30GB

# Close other applications

# For MLX on Mac
# Use 4-bit quantized models
llm = create_llm("mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")
```

---

## Best Practices

Follow these best practices to avoid issues:

### Configuration Management
- Use environment variables for API keys
- Never commit credentials to version control
- Use `.env` files (add to `.gitignore`)
- Implement configuration validation
- Use secret management in production

### Tool Development
- Always use `@tool` decorator
- Add type hints to all parameters
- Write clear docstrings
- Handle edge cases and errors
- Test tools independently first

### Error Handling
- Always use try/except blocks
- Implement provider fallback strategies
- Log errors systematically
- Design for graceful degradation
- Monitor error rates in production

### Performance
- Always set `max_tokens`
- Use streaming for long responses
- Batch similar requests when possible
- Monitor memory usage
- Profile slow operations

### Security
- Validate all user inputs
- Sanitize file paths and commands
- Use least privilege principle
- Regular security audits
- Keep dependencies updated

---

## Debug Techniques

### Enable Debug Logging

**Core Library:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from abstractcore import create_llm
llm = create_llm("openai", model="gpt-4o-mini")
```

**Server:**
```bash
# Enable debug mode
export ABSTRACTCORE_DEBUG=true

# Start with debug logging
uvicorn abstractcore.server.app:app --log-level debug

# Monitor logs
tail -f logs/abstractcore_*.log
```

### Analyze Logs

```bash
# Find errors
grep '"level": "error"' logs/abstractcore_*.log

# Track specific request
grep "req_abc123" logs/abstractcore_*.log

# Monitor latency
cat logs/verbatim_*.jsonl | jq '.metadata.latency_ms'

# Token usage
cat logs/verbatim_*.jsonl | jq '.metadata.tokens | .input + .output' | \
  awk '{sum+=$1} END {print "Total:", sum}'
```

### Test in Isolation

```python
# Test provider directly
from abstractcore import create_llm

try:
    llm = create_llm("openai", model="gpt-4o-mini")
    response = llm.generate("Hello")
    print(f"✓ Success: {response.content}")
except Exception as e:
    print(f"✗ Error: {e}")
```

### Collect Debug Information

```bash
# Create debug report
echo "=== System ===" > debug_report.txt
uname -a >> debug_report.txt
python --version >> debug_report.txt

echo "=== Packages ===" >> debug_report.txt
pip freeze | grep -E "abstract|openai|anthropic" >> debug_report.txt

echo "=== Environment ===" >> debug_report.txt
env | grep -E "ABSTRACT|OPENAI|ANTHROPIC|OLLAMA" >> debug_report.txt

echo "=== Tests ===" >> debug_report.txt
python -c "from abstractcore import create_llm; print('Core library: OK')" >> debug_report.txt 2>&1
curl http://localhost:8000/health >> debug_report.txt 2>&1

cat debug_report.txt
```

---

## Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `ModuleNotFoundError` | Package not installed | `pip install abstractcore` (then add provider extras as needed) |
| `Authentication Error` | Invalid API key | Check API key environment variable |
| `Connection refused` | Service not running | Start Ollama/LMStudio/server |
| `LM Studio connection failed` | LM Studio server not enabled | Enable "Status: Running" toggle in LM Studio GUI |
| `400 Bad Request` (LM Studio) | Context length too small | Increase Default Context Length to "Model Maximum" in LM Studio |
| `Model not found` | Model unavailable | Pull model or check name |
| `Rate limit exceeded` | Too many requests | Wait or upgrade plan |
| `Timeout` | Request took too long | Use smaller model or increase timeout |
| `Out of memory` | Insufficient RAM | Use smaller model |
| `Port already in use` | Another process using port | Kill process or use different port |

---

## Getting Help

If you're still stuck:

1. **Check Documentation:**
   - [Getting Started](getting-started.md) - Core library quick start
   - [Prerequisites](prerequisites.md) - Provider setup
   - [Python API Reference](api-reference.md) - Core library API
   - [Server Guide](server.md) - Server setup
   - [Server API Reference](server.md) - REST API endpoints

2. **Enable Debug Mode:**
   ```bash
   export ABSTRACTCORE_DEBUG=true
   ```

3. **Collect Information:**
   - Error messages
   - Debug logs
   - System information
   - Steps to reproduce

4. **Community Support:**
   - GitHub Issues: [github.com/lpalbou/AbstractCore/issues](https://github.com/lpalbou/AbstractCore/issues)
   - GitHub Discussions: [github.com/lpalbou/AbstractCore/discussions](https://github.com/lpalbou/AbstractCore/discussions)

---

**Remember**: Most issues are configuration-related. Double-check environment variables, API keys, and that services are running before diving deep into debugging.
