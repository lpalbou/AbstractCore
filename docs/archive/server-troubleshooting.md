# AbstractCore Server - Troubleshooting & Debug Guide

## Quick Diagnosis

Before diving deep, run these quick checks:

```bash
# 1. Server running?
curl http://localhost:8000/health

# 2. Models available?
curl http://localhost:8000/v1/models | jq '.data[].id'

# 3. Provider working?
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "test"}]}'
```

## Common Issues & Solutions

### Issue 1: Missing ABSTRACTCORE_API_KEY

**Symptom:**
```
Error: Missing environment variable: ABSTRACTCORE_API_KEY
```

**Solution:**
```bash
export ABSTRACTCORE_API_KEY="unused"
echo $ABSTRACTCORE_API_KEY  # Verify it's set
```

### Issue 2: Connection Refused

**Symptom:**
```
curl: (7) Failed to connect to localhost port 8000: Connection refused
```

**Solution:**
```bash
# Check if server is running
ps aux | grep abstractcore

# Check if port is in use
lsof -i :8000
netstat -an | grep 8000

# Start server if not running
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

### Issue 3: Model Not Found

**Symptom:**
```
Error: Model 'qwen3-coder:30b' not found
```

**Solution:**
```bash
# List available models
curl http://localhost:8000/v1/models | jq '.data[].id'

# Use correct format: provider/model
"model": "ollama/qwen3-coder:30b"  # Correct
"model": "qwen3-coder:30b"         # Wrong

# For local models, ensure they're pulled
ollama list  # Check Ollama models
```

### Issue 4: Tool Calls Not Working

**Symptom:**
Tool calls disappear or aren't formatted correctly in streaming

**Solution:**
```bash
# Set correct tool call tags for your CLI
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3  # For Crush
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml     # For Gemini CLI
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=qwen3   # Default

# Restart server with new configuration
pkill -f "abstractcore.server.app"
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

### Issue 5: Slow Response Times

**Symptom:**
Responses take too long or timeout

**Solution:**
```bash
# Check model size vs available memory
free -h  # Linux
vm_stat  # macOS

# Use smaller model or quantization
"model": "ollama/qwen3-coder:7b"   # Smaller
"model": "ollama/qwen3-coder:30b"  # Larger

# Enable streaming for perceived speed
"stream": true
```

## Debug Logging

### Enable Comprehensive Logging

```bash
# Step 1: Enable debug mode
export ABSTRACTCORE_DEBUG=true

# Step 2: Restart server
pkill -f "abstractcore.server.app"
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# Step 3: Monitor logs in real-time
tail -f logs/abstractcore_*.log
```

### Understanding Log Files

| Log File | Purpose | What to Look For |
|----------|---------|-------------------|
| `logs/abstractcore_TIMESTAMP.log` | Structured events | Errors, warnings, request flow |
| `logs/YYYYMMDD-payloads.jsonl` | Full requests | Exact parameters sent |
| `logs/verbatim_TIMESTAMP.jsonl` | Complete I/O | Full prompts and responses |

### Useful Log Analysis Commands

```bash
# Find all errors
grep '"level": "error"' logs/abstractcore_*.log | jq -r '.event + " | " + .error'

# Track specific request
grep "req_abc123" logs/abstractcore_*.log

# Monitor tool call processing
grep "tool_call" logs/verbatim_*.jsonl | jq '.response'

# Calculate average response time
cat logs/verbatim_*.jsonl | jq '.metadata.latency_ms' | \
  awk '{sum+=$1; count++} END {print "Average:", sum/count "ms"}'

# Find failed requests
grep '"success": false' logs/verbatim_*.jsonl | jq '.metadata.error'

# Token usage by model
cat logs/verbatim_*.jsonl | jq -r 'select(.model=="qwen3-coder:30b") | .metadata.tokens | .input + .output' | \
  awk '{sum+=$1} END {print "Total tokens:", sum}'
```

## Agentic CLI Troubleshooting

### Codex CLI Issues

```bash
# Problem: Codex not connecting
# Solution: Check ALL three variables
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
export ABSTRACTCORE_API_KEY="unused"  # Often forgotten!

# Verify with test command
codex --model "lmstudio" "test"

# Check logs for connection attempts
tail -f logs/abstractcore_*.log | grep "Codex"
```

### Crush CLI Issues

```bash
# Problem: Wrong tool call format
# Solution: Configure for Crush
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false

# Restart server
pkill -f "abstractcore.server.app"
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# Test Crush
crush --model "anthropic/claude-3-5-haiku-latest" "test"
```

### Gemini CLI Issues

```bash
# Problem: XML tool calls not working
# Solution: Configure for Gemini
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=xml
export ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS=false

# Restart and test
pkill -f "abstractcore.server.app"
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
gemini-cli --model "ollama/qwen3-coder:30b" "test"
```

## Provider-Specific Issues

### Ollama

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# List available models
ollama list

# Pull missing model
ollama pull qwen3-coder:30b

# Use correct format in requests
"model": "ollama/qwen3-coder:30b"
```

### LMStudio

```bash
# Verify LMStudio server is running
curl http://localhost:1234/v1/models

# Check LMStudio GUI shows "Server running"
# Default port: 1234

# Use in AbstractCore
"model": "lmstudio"  # Uses whatever model is loaded in LMStudio
```

### OpenAI

```bash
# Verify API key
echo $OPENAI_API_KEY

# Test directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Common issues:
# - Expired key
# - Rate limits
# - Organization restrictions
```

### Anthropic

```bash
# Verify API key
echo $ANTHROPIC_API_KEY

# Check key format
# Should start with "sk-ant-"

# Test with AbstractCore
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-5-haiku-latest",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Performance Diagnostics

### Memory Issues

```bash
# Check system memory
free -h          # Linux
vm_stat | head   # macOS

# Monitor during generation
watch -n 1 'free -h'  # Linux
while true; do vm_stat | head; sleep 1; done  # macOS

# Solutions:
# 1. Use smaller models
# 2. Reduce max_tokens
# 3. Enable low_memory mode
```

### CPU/GPU Utilization

```bash
# Monitor CPU usage
top -p $(pgrep -f abstractcore)  # Linux
top -pid $(pgrep -f abstractcore)  # macOS

# Check if GPU is being used (if available)
nvidia-smi  # NVIDIA GPUs
```

### Network Latency

```bash
# Test network to cloud providers
ping api.openai.com
ping api.anthropic.com

# Measure request time
time curl http://localhost:8000/health

# Check for proxy issues
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

## Emergency Procedures

### Complete Reset

```bash
# 1. Kill all processes
pkill -f "abstractcore"
pkill -f "uvicorn"

# 2. Clear all logs
rm -rf logs/

# 3. Reset environment
unset ABSTRACTCORE_DEBUG
unset ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS
unset ABSTRACTCORE_DEFAULT_EXECUTE_TOOLS

# 4. Fresh start
pip install --upgrade "abstractcore[server]"
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

### Debug Information Collection

When reporting issues, collect this information:

```bash
# Create debug report
echo "=== System Info ===" > debug_report.txt
uname -a >> debug_report.txt
python --version >> debug_report.txt

echo "=== Package Versions ===" >> debug_report.txt
pip freeze | grep -E "abstract|uvicorn|openai" >> debug_report.txt

echo "=== Environment Variables ===" >> debug_report.txt
env | grep -E "ABSTRACT|OPENAI|ANTHROPIC" >> debug_report.txt

echo "=== Server Test ===" >> debug_report.txt
curl http://localhost:8000/health >> debug_report.txt 2>&1
curl http://localhost:8000/v1/models >> debug_report.txt 2>&1

echo "=== Recent Errors ===" >> debug_report.txt
grep '"level": "error"' logs/abstractcore_*.log | tail -20 >> debug_report.txt

echo "Debug report saved to debug_report.txt"
```

## Validation Checklist

Before going to production, ensure:

- [ ] **Server starts without errors**
  ```bash
  uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
  ```

- [ ] **Health check passes**
  ```bash
  curl http://localhost:8000/health
  # Expected: {"status":"healthy"}
  ```

- [ ] **Models are available**
  ```bash
  curl http://localhost:8000/v1/models | jq '.data | length'
  # Expected: > 0
  ```

- [ ] **Basic generation works**
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "YOUR_MODEL", "messages": [{"role": "user", "content": "Hello"}]}'
  ```

- [ ] **Streaming works**
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "YOUR_MODEL", "messages": [{"role": "user", "content": "Count to 5"}], "stream": true}'
  ```

- [ ] **Tool calls work (if using)**
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "YOUR_MODEL",
      "messages": [{"role": "user", "content": "What time is it?"}],
      "tools": [{"type": "function", "function": {"name": "get_time"}}]
    }'
  ```

- [ ] **Logs are being generated**
  ```bash
  ls -la logs/
  # Should see log files if ABSTRACTCORE_DEBUG=true
  ```

## Getting Help

If you're still stuck:

1. **Check existing issues**: [GitHub Issues](https://github.com/abstractcore/core/issues)
2. **Join community**: [Discord Server](https://discord.abstractcore.ai)
3. **Read detailed docs**:
   - [Configuration Guide](server-configuration.md)
   - [Tool Call Tag Rewriting](tool-syntax-rewriting.md)
   - [Unified Streaming Architecture](unified-streaming-architecture.md)

When asking for help, always include:
- Your debug report (see above)
- Exact error messages
- Steps to reproduce
- What you've already tried

---

AbstractCore Server Troubleshooting
