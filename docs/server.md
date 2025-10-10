# AbstractCore Server - Universal LLM Gateway

## What It Does

Turn AbstractCore into a **universal API server** that works with ANY LLM provider through simple, clean endpoints. One server, all models, any language.

## Quick Start

```bash
# Install and start
pip install abstractcore[server]
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Now ANY OpenAI client can use ANY provider!
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Use Claude with OpenAI's client!
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Server Endpoints

| Endpoint | Method | What It Does |
|----------|---------|-------------|
| `/` | GET | Server info + quick examples |
| `/chat` | GET/POST | Universal chat endpoint |
| `/models` | GET | List all available models |
| `/providers` | GET | Show provider status |
| `/health` | GET | Health check |
| `/test` | GET | Test all providers quickly |
| `/v1/chat/completions` | POST | OpenAI compatible endpoint |
| `/v1/responses` | POST | OpenAI Responses API (Codex preferred) |
| `/v1/messages` | POST | Anthropic Messages API compatible |
| `/docs` | GET | Interactive API documentation |

## Usage Examples

### 1. URL Chat (No JSON needed!)
```bash
# Just URL parameters - works in any browser
curl "http://localhost:8000/chat?message=Hello%20world"
curl "http://localhost:8000/chat?message=Write%20code&provider=anthropic"

# Streaming responses
curl "http://localhost:8000/chat?message=Tell%20a%20story&stream=true"
```

### 2. POST Chat (Simple JSON)
```bash
# Regular response
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "provider": "anthropic"}'

# Streaming response
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "stream": true}'
```

### 3. OpenAI Compatible
```bash
# Regular response
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming response (OpenAI format)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-5-haiku-latest",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### 4. Test All Providers
```bash
# See which providers are working
curl http://localhost:8000/test

# List models by type
curl "http://localhost:8000/models?type=chat"
curl "http://localhost:8000/models?provider=ollama&type=chat"
```

## Configuration

### Basic Configuration
```bash
# Debug mode (logs every request/response)
export ABSTRACTCORE_DEBUG=true
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Development mode with reload
uvicorn abstractllm.server.app:app --reload --log-level debug

# Custom port
uvicorn abstractllm.server.app:app --port 3000
```

### Environment Variables
```bash
# Set defaults
export ABSTRACTCORE_DEFAULT_PROVIDER=anthropic
export ABSTRACTCORE_DEFAULT_MODEL=claude-3-5-haiku-latest

# Provider API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

## Advanced Parameters

### Tool Call Tag Rewriting

The server supports real-time tool call tag rewriting for agentic CLI compatibility:

```python
# Rewrite tool calls for different CLIs
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}],
    tool_call_tags="qwen3"  # For Codex CLI compatibility
)
```

**Supported Formats:**
- `qwen3` - `<|tool_call|>...JSON...</|tool_call|>` (Codex, OpenAI)
- `llama3` - `<function_call>...JSON...</function_call>` (Crush, Anthropic)
- `xml` - `<tool_call>...JSON...</tool_call>` (Gemini CLI)
- `gemma` - ````tool_code...JSON...```` (Gemma models)
- `codex` - Same as qwen3
- `crush` - Same as llama3
- `gemini` - Same as xml
- `openai` - Same as qwen3
- `anthropic` - Same as llama3

### Tool Execution Control

Control whether the server executes tools automatically or lets the agent handle execution:

```python
# AbstractCore executes tools automatically (default)
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=[{"type": "function", "function": {"name": "get_weather"}}],
    execute_tools=True  # Default: tools are executed
)

# Let the agent handle tool execution (for agentic CLI mode)
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=[{"type": "function", "function": {"name": "get_weather"}}],
    execute_tools=False  # Tools are generated but not executed
)
```

**When to use `execute_tools=False`:**
- **Agentic CLI mode**: When the CLI will handle tool execution
- **API server mode**: When you want to return tool calls for external processing
- **Custom tool handling**: When you have custom tool execution logic

**When to use `execute_tools=True` (default):**
- **Standalone usage**: When you want AbstractCore to handle everything
- **Simple applications**: When you want automatic tool execution
- **Development/testing**: When you want immediate tool results

## Debug Logging

AbstractCore server includes comprehensive debug logging that captures **every request and response** with full payloads.

### Enable Debug Mode

```bash
# Enable debug mode with environment variable
export ABSTRACTCORE_DEBUG=true

# Start server with debug logging
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# Optional: Increase log level for even more detail
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000 --log-level debug
```

### Log Files Created

When debug mode is enabled, AbstractCore creates three types of log files in the `logs/` directory:

#### 1. Structured Logs - `logs/abstractllm_TIMESTAMP.log`
JSON-formatted structured logs with metadata:
```json
{"timestamp": "2025-10-09T15:30:45", "level": "info", "logger": "server", "event": "🚀 AbstractCore Server Starting", "debug_mode": true}
{"timestamp": "2025-10-09T15:30:47", "level": "info", "logger": "anthropic_messages", "event": "🚀 Anthropic Messages API Request", "request_id": "req_abc123", "model": "lmstudio", "user_content": "Write Python code"}
```

#### 2. Request Payloads - `logs/YYYYMMDD-payloads.jsonl`
Complete request payloads (only in debug mode):
```json
{"timestamp": "2025-10-09T15:30:47.123", "request_id": "req_abc123", "endpoint": "/v1/messages", "model": "lmstudio", "max_tokens": 1000, "temperature": 0.7, "messages": [{"role": "user", "content": "Write a Python hello world function"}], "tools": null}
```

#### 3. Verbatim Interactions - `logs/verbatim_TIMESTAMP.jsonl`
Full prompt/response pairs for every generation:
```json
{"timestamp": "2025-10-09T15:30:48.456", "provider": "lmstudio", "model": "qwen3-next-80b", "prompt": "Write a Python hello world function", "response": "def hello_world():\n    print(\"Hello, World!\")\n\nhello_world()", "metadata": {"tokens": {"input": 8, "output": 15}, "latency_ms": 1234}}
```

### Debug Analysis Examples

```bash
# Find all errors in structured logs
grep '"level": "error"' logs/abstractllm_*.log | jq -r '.event + " | " + .error'

# Find requests that failed
grep '"success": false' logs/verbatim_*.jsonl | jq -r '.metadata.error'

# Total tokens used by model
cat logs/verbatim_*.jsonl | jq -r 'select(.provider=="lmstudio") | .metadata.tokens.input + .metadata.tokens.output' | awk '{sum+=$1} END {print "Total tokens:", sum}'

# Average response time by provider
cat logs/verbatim_*.jsonl | jq -r 'select(.provider=="lmstudio") | .metadata.latency_ms' | awk '{sum+=$1; count++} END {print "Average latency:", sum/count "ms"}'
```

## Agentic CLI Compatibility

AbstractCore server provides full compatibility with agentic CLIs like Codex, Gemini CLI, and Crush through OpenAI-compatible endpoints.

### Supported Agentic CLIs

#### Codex (OpenAI)
OpenAI's Codex CLI for AI-powered coding assistance.

**Quick Setup:**
```bash
# Set ALL required environment variables
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
export ABSTRACTCORE_API_KEY="unused"  # CRITICAL: Required by Codex

# Test with Codex
codex --model "ollama/qwen3-coder:30b" "Write a Python function to sort a list"
```

#### Gemini CLI (Google)
Google's Gemini CLI for development workflows.

**Setup:**
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Run Gemini CLI
gemini-cli --model "ollama/qwen3-coder:30b" "Analyze this codebase"
```

#### Crush CLI (Charmbracelet)
Crush provides a beautiful terminal AI experience.

**Setup:**
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

# Run crush
crush --model "ollama/qwen3-coder:30b" "Write a FastAPI server"
```

### Recommended Models for Agentic CLIs

**For Coding Tasks:**
- `ollama/qwen3-coder:30b` - Excellent code generation and analysis
- `ollama/deepseek-coder:33b` - Strong reasoning and debugging
- `lmstudio/qwen3-next-80b` - Best quality for complex tasks

**For General Tasks:**
- `anthropic/claude-3-5-haiku-latest` - Fast and intelligent
- `openai/gpt-4o-mini` - Reliable and cost-effective
- `ollama/llama3:8b` - Local and private

### Example Workflows

```bash
# Code generation with Codex
codex --model "ollama/qwen3-coder:30b" "Create a REST API with authentication"

# Development with Gemini CLI
gemini-cli --model "ollama/qwen3-coder:30b" "What does this project do?"

# Interactive coding with Crush
crush --model "ollama/qwen3-coder:30b"
```

### Troubleshooting Agentic CLIs

#### Enable Debug Logging First
```bash
# 1. Enable debug mode
export ABSTRACTCORE_DEBUG=true

# 2. Restart AbstractCore server
pkill -f "abstractllm.server.app"
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# 3. Try your CLI command
codex --model "lmstudio" "test command"

# 4. Check logs for detailed information
tail -f logs/abstractllm_*.log         # Structured logs
tail -f logs/verbatim_*.jsonl          # Full request/response pairs
cat logs/$(date +%Y%m%d)-payloads.jsonl | jq '.'  # Complete request payloads
```

#### Common Issues

**Missing API Key Error:**
```bash
# Problem: Missing environment variable: ABSTRACTCORE_API_KEY
export ABSTRACTCORE_API_KEY="unused"
echo $ABSTRACTCORE_API_KEY  # Verify it's set
```

**Connection Issues:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Check environment variables
echo $OPENAI_BASE_URL        # Should be http://localhost:8000/v1
echo $OPENAI_API_KEY         # Should be "unused"
echo $ABSTRACTCORE_API_KEY   # Should be "unused"

# Test direct connection
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "test"}]}'
```

**Model Not Found:**
```bash
# List available models
curl http://localhost:8000/v1/models

# Use correct format: provider/model
codex --model "ollama/qwen3-coder:30b"  # Correct
codex --model "qwen3-coder:30b"         # Wrong
```

## Language Examples

### Python
```python
import requests

response = requests.get("http://localhost:8000/chat", params={
    "message": "What is Python?",
    "provider": "openai"
})
print(response.json()["response"])
```

### JavaScript
```javascript
// Regular response
fetch('http://localhost:8000/chat?message=Hello&provider=anthropic')
  .then(r => r.json())
  .then(data => console.log(data.response));

// Streaming response
fetch('http://localhost:8000/chat?message=Tell%20a%20story&stream=true')
  .then(response => {
    const reader = response.body.getReader();
    function readStream() {
      return reader.read().then(({done, value}) => {
        if (done) return;
        const text = new TextDecoder().decode(value);
        console.log(text);
        return readStream();
      });
    }
    return readStream();
  });
```

### curl
```bash
curl "http://localhost:8000/chat?message=Hello&provider=ollama&model=qwen3-coder:30b"
```

## Response Format

All endpoints return consistent, simple responses:

```json
{
  "message": "Hello!",
  "response": "Hello! How can I help you today?",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 12,
    "total_tokens": 20
  }
}
```

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim
RUN pip install abstractcore[server]
EXPOSE 8000
CMD ["uvicorn", "abstractllm.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Railway/Vercel/Fly.io
Just deploy with `pip install abstractcore[server]` and `uvicorn abstractllm.server.app:app`

### Local Development
```bash
uvicorn abstractllm.server.app:app --reload --log-level debug
```

## Why Choose AbstractCore Server?

✅ **Simple** - 10 focused endpoints, not 20+
✅ **Fast** - Clean, lightweight code
✅ **Universal** - Works with all providers
✅ **Compatible** - OpenAI clients work out of the box
✅ **Examples** - Actually working examples
✅ **Reliable** - Production-tested patterns
✅ **Debuggable** - Comprehensive logging with debug mode
✅ **Agentic CLI Ready** - Full Codex/Gemini CLI/Crush compatibility

## Complete Example: LMStudio + Qwen3 Next 80B + Codex Setup

This complete example shows how to set up AbstractCore server with LMStudio and Qwen3 Next 80B to power Codex CLI with a local 80B parameter model.

### Architecture
```
Codex CLI → AbstractCore Server (localhost:8000) → LMStudio (localhost:1234) → Qwen3 Next 80B
```

### Prerequisites
- **Disk Space**: 80GB+ free for model download
- **RAM**: 80GB+ recommended for Qwen3 Next 80B
- **Platform**: macOS, Linux, or Windows
- **Internet**: For initial model download

### Step 1: Install and Configure LMStudio

```bash
# 1. Download LMStudio
# Visit https://lmstudio.ai/ and download for your platform

# 2. Install and launch LMStudio
# Follow the installer instructions for your OS

# 3. Download Qwen3 Next 80B model
# In LMStudio GUI:
#   - Click "Discover" tab
#   - Search: "qwen3-next-80b" or browse Qwen models
#   - Select appropriate quantization (Q4_K_M recommended for 80GB RAM)
#   - Click download (this will take time - ~80GB)

# 4. Start LMStudio local server
# In LMStudio GUI:
#   - Go to "Local Server" tab
#   - Select "qwen3-next-80b" from model dropdown
#   - Adjust settings if needed (context length, temperature)
#   - Click "Start Server"
#   - Verify it shows: "Server running on http://localhost:1234"
```

### Step 2: Install AbstractCore

```bash
# Install AbstractCore with server and LMStudio support
pip install abstractcore[server,lmstudio]

# Verify installation
python -c "import abstractllm; print('AbstractCore installed successfully')"
```

### Step 3: Start AbstractCore Server with Debug Logging

```bash
# Enable debug mode for comprehensive logging
export ABSTRACTCORE_DEBUG=true

# Start AbstractCore server (keep this running in one terminal)
uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000

# You should see:
# INFO: AbstractCore Server Starting | debug_mode=true
# INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Configure Environment for Codex

```bash
# Set ALL required environment variables
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
export ABSTRACTCORE_API_KEY="unused"  # CRITICAL: Required by Codex

# Verify environment variables are set
echo "OPENAI_BASE_URL: $OPENAI_BASE_URL"
echo "OPENAI_API_KEY: $OPENAI_API_KEY"
echo "ABSTRACTCORE_API_KEY: $ABSTRACTCORE_API_KEY"
```

### Step 5: Test Each Component

#### Test 1: LMStudio Direct Connection
```bash
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-next-80b",
    "messages": [{"role": "user", "content": "Hello, can you confirm you are working?"}],
    "max_tokens": 50
  }'

# Expected: JSON response with model's greeting
```

#### Test 2: AbstractCore → LMStudio Connection
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio",
    "messages": [{"role": "user", "content": "Hello from AbstractCore"}],
    "max_tokens": 50
  }'

# Expected: JSON response routed through AbstractCore
```

#### Test 3: Codex → AbstractCore → LMStudio
```bash
# Simple test command
codex --model "lmstudio" "Write a Python hello world function"

# Expected: Codex should generate Python code using Qwen3 Next 80B
```

### Step 6: Verify Setup with Debug Logs

After running the tests, check the logs to ensure everything is working:

```bash
# Check AbstractCore received requests
tail -5 logs/abstractllm_*.log

# Check full request payloads
cat logs/$(date +%Y%m%d)-payloads.jsonl | jq '.endpoint, .model, .messages'

# Check model responses
tail -3 logs/verbatim_*.jsonl | jq '.provider, .model, .response'
```

Expected log entries:
```json
{"level": "info", "event": "🚀 Anthropic Messages API Request", "model": "lmstudio"}
{"level": "info", "event": "🎉 Generation Completed", "provider": "lmstudio"}
```

### Step 7: Real Usage Examples

#### Code Review
```bash
# Review a Python file
codex --model "lmstudio" "Review this code for improvements and security issues" < myfile.py

# Check logs to see the full conversation
tail -f logs/verbatim_*.jsonl | jq '.response'
```

#### Interactive Development
```bash
# Start interactive session
codex --interactive --model "lmstudio"

# Within the session:
> Write a FastAPI endpoint for user registration
> Add input validation with Pydantic
> Include error handling
> Write unit tests for this endpoint
```

#### File Operations
```bash
# Refactor code across multiple files
codex --model "lmstudio" "Refactor this module to use type hints and improve error handling" --files src/

# Debug what files were processed
grep "file_operations" logs/abstractllm_*.log
```

### Troubleshooting

#### LMStudio Connection Issues
```bash
# Check if LMStudio server is running
curl http://localhost:1234/health

# If not responding:
# 1. Restart LMStudio GUI
# 2. Go to "Local Server" → Select model → "Start Server"
# 3. Verify port is 1234 (default)
```

#### AbstractCore Issues
```bash
# Problem: ABSTRACTCORE_API_KEY error
export ABSTRACTCORE_API_KEY="unused"
echo $ABSTRACTCORE_API_KEY  # Verify it's set

# Problem: Model not found errors
curl http://localhost:8000/v1/models | jq '.data[].id'
codex --model "lmstudio"  # Correct
```

#### Codex Issues
```bash
# Check environment
echo $OPENAI_BASE_URL  # Should be http://localhost:8000/v1

# Test direct connection
curl $OPENAI_BASE_URL/models

# Check for typos in environment variables
env | grep -E "(OPENAI|ABSTRACTCORE)"
```

### Performance Tuning
```bash
# In LMStudio GUI:
# 1. Increase context length for longer conversations
# 2. Adjust temperature: 0.1 for code, 0.7 for creative tasks
# 3. Set max tokens based on your needs
# 4. Enable GPU acceleration if available

# Monitor performance via logs
grep "latency_ms" logs/verbatim_*.jsonl | jq '.metadata.latency_ms'
```

### Verification Checklist

Before using in production, verify each component:

- [ ] **LMStudio**: Model loaded, server running on localhost:1234
- [ ] **AbstractCore**: Server running on localhost:8000, debug logs enabled
- [ ] **Environment**: All three variables set (OPENAI_BASE_URL, OPENAI_API_KEY, ABSTRACTCORE_API_KEY)
- [ ] **Connection**: Direct curl tests to both LMStudio and AbstractCore work
- [ ] **Codex**: Basic test command executes and returns sensible output
- [ ] **Logs**: Debug logs capture requests and responses correctly

### Expected Performance

With Qwen3 Next 80B via LMStudio:

- **Response Time**: 2-10 seconds (depends on hardware)
- **Quality**: High-quality code generation and analysis
- **Context**: Up to 32K tokens (configurable in LMStudio)
- **Memory Usage**: ~80GB RAM for model + OS overhead

### Success Indicators

You'll know the setup is working when:

1. **Codex commands execute without errors**
2. **Model responses are coherent and relevant**
3. **Debug logs show successful request/response flow**
4. **LMStudio shows processing activity**
5. **Response times are reasonable for your hardware**

---

**Congratulations!** You now have Codex powered by a local 80B parameter model via AbstractCore, giving you powerful AI coding assistance while keeping your data completely private.

**Debug any issues with:** `tail -f logs/abstractllm_*.log`

## Getting Started

1. **Install**: `pip install abstractcore[server]`
2. **Start**: `uvicorn abstractllm.server.app:app --host 0.0.0.0 --port 8000`
3. **Test**: Visit `http://localhost:8000`
4. **Use**: Point any OpenAI client or agentic CLI to your server

That's it! No complexity, no confusion, just a working universal LLM API server.

---

**AbstractCore Server** - One server, all models, any language. 🚀