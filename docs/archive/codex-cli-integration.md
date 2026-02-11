# Codex CLI Integration with AbstractCore

> **Archived (historical)**: This document is kept for reference and may be outdated.
> For current, supported instructions see **[Server](../server.md)** (agentic CLI integration) and **[Troubleshooting](../troubleshooting.md)**.

## Overview

AbstractCore provides seamless integration with Codex CLI, enabling powerful AI-assisted coding across different local and cloud language models. This guide covers everything you need to know about setting up and using Codex CLI with AbstractCore.

## Prerequisites

- Python 3.9+
- AbstractCore server installed: `pip install "abstractcore[server]"`
- Codex CLI (follow Codex installation instructions)
- A supported language model (local or cloud)

## Quick Setup

### 1. Start AbstractCore Server

```bash
# Start server with default configuration
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

### 2. Configure Environment Variables

```bash
# Set these environment variables for Codex CLI
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"
export ABSTRACTCORE_API_KEY="unused"
```

### 3. Choose Your Model

AbstractCore supports multiple models for Codex CLI:

#### Local Models (Recommended)
- `ollama/qwen3-coder:30b` - Best open-source code generation
- `ollama/deepseek-coder:33b` - Strong reasoning capabilities
- `lmstudio/qwen3-next-80b` - High-quality large model

#### Cloud Models
- `openai/gpt-4o-mini` - Reliable and cost-effective
- `anthropic/claude-3-5-haiku-latest` - Fast and intelligent

## Usage Examples

### Basic Code Generation

```bash
# Generate a simple Python function
codex --model "ollama/qwen3-coder:30b" "Write a function to calculate factorial"
```

### Interactive Development

```bash
# Start interactive mode
codex --interactive --model "ollama/qwen3-coder:30b"

# Example interactions:
> Create a FastAPI endpoint for user registration
> Add input validation with Pydantic
> Write unit tests for the endpoint
```

### File Operations

```bash
# Refactor code across multiple files
codex --model "ollama/qwen3-coder:30b" \
    "Refactor this module to use type hints and improve error handling" \
    --files src/
```

## Tool Call Compatibility

AbstractCore automatically handles tool call format conversion:

```bash
# Tool calls work across different models
codex --model "ollama/qwen3-coder:30b" \
    "List files in current directory" \
    --tools list_files_tool
```

### Configuring Tool Call Format

```bash
# Optional: Set default tool call format at server start
export ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS=llama3
uvicorn abstractcore.server.app:app

# Or set per-request
codex --model "ollama/qwen3-coder:30b" \
    --tool_call_tags "xml" \
    "Analyze project structure"
```

## Debugging and Logging

### Enable Debug Mode

```bash
# Start server with comprehensive logging
export ABSTRACTCORE_DEBUG=true
uvicorn abstractcore.server.app:app

# Check logs in real-time
tail -f logs/abstractcore_*.log
```

## Performance Tuning

- Adjust model context length
- Use appropriate temperature settings
- Enable GPU acceleration when possible

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Verify full provider/model name
   - Check available models: `curl http://localhost:8000/v1/models`

2. **Connection Problems**
   - Confirm server is running
   - Check environment variables
   - Verify network connectivity

3. **Tool Call Formatting**
   - Use `--tool_call_tags` to override defaults
   - Verify tool call tags match your CLI's expectations

## Best Practices

- Use models appropriate for your task
- Start with interactive mode for complex tasks
- Leverage debug logging for insights
- Test across different models

## Supported Platforms

- macOS
- Linux
- Windows (with WSL2)

## Security Considerations

- Use `ABSTRACTCORE_API_KEY` for basic access control
- Configure firewall rules
- Use HTTPS in production environments

## Getting Help

- GitHub Issues: [AbstractCore Repository]
- Community Forums: [Link to Forums]
- Email Support: support@abstractcore.ai

---

**Happy Coding with AbstractCore and Codex CLI!** 🚀
