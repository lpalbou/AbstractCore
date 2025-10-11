# AbstractLLM Core Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting strategies for AbstractLLM Core, covering setup, tool calling, performance, integration, and debugging challenges.

## 1. Common Setup Issues

### 1.1 Provider Configuration Problems

#### Symptoms
- Authentication failures
- API key rejection
- Unsupported model selection

#### Diagnosis
```bash
# Check current provider configuration
python -m abstractllm.utils.cli --config show
# Verify API keys and endpoints
cat ~/.abstractllm/config.json
```

#### Solutions

**OpenAI Provider**:
```bash
# Verify and set OpenAI API key
export OPENAI_API_KEY='sk-...'
python -m abstractllm.utils.cli --provider openai --validate-key

# Common API key issues
# - Ensure no whitespace
# - Check key has correct permissions
# - Verify organization access
```

**Local Model Providers**:
```bash
# Validate Ollama/HuggingFace model availability
python -m abstractllm.utils.cli --provider ollama --list-models
python -m abstractllm.utils.cli --provider huggingface --list-models

# Troubleshoot model download
ollama pull qwen3-coder:30b
# OR
huggingface-cli download mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

#### Prevention
- Use environment variables for sensitive credentials
- Regularly update provider configurations
- Check model compatibility matrix in documentation

### 1.2 Installation Dependencies

#### Symptoms
- Package import errors
- Dependency conflicts
- Version incompatibilities

#### Diagnosis
```bash
# Check AbstractLLM and dependency versions
pip freeze | grep abstractllm
pip freeze | grep torch
pip freeze | grep transformers

# Validate installation
python -m pip check abstractllm
```

#### Solutions
```bash
# Create clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Recommended installation
pip install abstractllm[full] --upgrade

# Resolve dependency conflicts
pip install --upgrade pip
pip install abstractllm[full] --no-deps
pip install -r requirements.txt
```

## 2. Tool Calling Issues

### 2.1 Tool Call Format Problems

#### Symptoms
- Tools not executing
- Malformed tool call content
- Streaming interruption

#### Diagnosis
```bash
# Enable verbose tool call logging
export ABSTRACTLLM_DEBUG=true
python -m abstractllm.utils.cli --tools-verbose

# Inspect tool call raw content
python -m abstractllm.utils.debug tool_calls
```

#### Solutions

**Custom Tool Call Tags**:
```python
# Specify exact tool call tags
from abstractllm import ToolCallTags

custom_tags = ToolCallTags(
    start_tag='MYTOOL',
    end_tag='ENDMYTOOL',
    auto_format=False
)

# Use with CLI or programmatically
abstractllm.generate(
    prompt="List files",
    tool_call_tags=custom_tags
)
```

**Debugging Tool Calls**:
```bash
# Validate tool call parsing
python -m abstractllm.utils.cli \
    --model qwen3-coder:30b \
    --prompt "List files in current directory" \
    --tools-debug
```

### 2.2 Streaming Tool Call Issues

#### Symptoms
- Tool calls disappear during streaming
- Partial tool execution
- Inconsistent tool behavior

#### Solution
```python
# Ensure streaming mode compatibility
stream_response = abstractllm.generate(
    prompt="List files",
    stream=True,
    stream_tools=True  # Explicit tool streaming
)

for chunk in stream_response:
    print(chunk.content)  # Safe streaming
    if chunk.tools:
        for tool in chunk.tools:
            print(f"Tool detected: {tool.name}")
```

## 3. Performance Issues

### 3.1 Response Time and Latency

#### Diagnosis
```bash
# Measure generation performance
python -m abstractllm.utils.performance \
    --provider ollama \
    --model qwen3-coder:30b \
    --prompt-length 500 \
    --iterations 10
```

#### Solutions
```python
# Configure timeout and performance parameters
response = abstractllm.generate(
    prompt="Complex task",
    max_tokens=1000,
    timeout=30,  # seconds
    stream=True,
    temperature=0.7
)
```

### 3.2 Memory Management

#### Diagnosis
```bash
# Check memory usage during generation
python -m abstractllm.utils.memory_profile \
    --model mlx-community/GLM-4.5-Air-4bit
```

#### Solutions
```python
# Manage memory-intensive generations
response = abstractllm.generate(
    prompt="Large computation",
    max_tokens=2000,
    low_memory=True,  # Optimize memory
    batch_size=16
)
```

## 4. Server and CLI Integration

### 4.1 OpenAI-Compatible Endpoint

#### Diagnosis
```bash
# Test endpoint compatibility
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }'
```

### 4.2 Agentic CLI Troubleshooting

```bash
# Validate CLI configuration
python -m abstractllm.utils.cli --doctor

# Reset CLI configuration
python -m abstractllm.utils.cli --reset-config
```

## 5. Debugging Techniques

### 5.1 Logging and Verbose Mode

```bash
# Enable comprehensive logging
export ABSTRACTLLM_LOG_LEVEL=DEBUG
export ABSTRACTLLM_TRACE=true

# Log to file for detailed analysis
python -m abstractllm.utils.cli \
    --log-file /tmp/abstractllm_debug.log
```

### 5.2 Performance Profiling

```bash
# Profile model generation
python -m abstractllm.utils.profile \
    --provider ollama \
    --model qwen3-coder:30b \
    --output /tmp/profile_report.json
```

## 6. Error Code Reference

### Common Error Patterns

| Error Code | Description | Solution |
|-----------|-------------|----------|
| `ABTLLM-001` | Provider Authentication Failed | Verify API credentials |
| `ABTLLM-002` | Model Unavailable | Check model compatibility |
| `ABTLLM-003` | Tool Call Parsing Error | Validate tool call format |
| `ABTLLM-004` | Streaming Interruption | Retry with stream=True |
| `ABTLLM-005` | Memory Allocation Error | Use low_memory=True |

## Conclusion

This guide covers the most common challenges with AbstractLLM Core. Always ensure you're using the latest version and consult the [official documentation](https://abstractllm.ai/docs) for the most up-to-date troubleshooting information.

### Need More Help?

- Join our [Discord Community](https://discord.abstractllm.ai)
- File an issue on [GitHub Discussions](https://github.com/abstractllm/core/discussions)
- Email support: `support@abstractllm.ai`

**Last Updated**: 2025-10-11
**AbstractLLM Core Version**: v2.5.0