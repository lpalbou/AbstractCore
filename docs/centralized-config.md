# AbstractCore Centralized Configuration

AbstractCore provides a unified configuration system that manages default models, cache directories, logging settings, and other package-wide preferences from a single location.

## Configuration File Location

Configuration is stored in: `~/.abstractcore/config/abstractcore.json`

## Configuration Sections

### Application Defaults

Set default providers and models for specific AbstractCore applications:

```bash
# Set defaults for individual apps
abstractcore --set-app-default summarizer openai gpt-4o-mini
abstractcore --set-app-default cli anthropic claude-3-5-haiku
abstractcore --set-app-default extractor ollama qwen3:4b-instruct
abstractcore --set-app-default intent lmstudio qwen/qwen3-30b-a3b-2507

# View current app defaults
abstractcore --status
```

### Global Defaults

Set fallback defaults when app-specific configurations are not available:

```bash
# Set global fallback model
abstractcore --set-global-default ollama/llama3:8b

# Set specialized defaults
abstractcore --set-chat-model openai/gpt-4o-mini
abstractcore --set-code-model anthropic/claude-3-5-sonnet
```

### Cache Directories

Configure cache locations for different components:

```bash
# Set cache directories
abstractcore --set-default-cache-dir ~/.cache/abstractcore
abstractcore --set-huggingface-cache-dir ~/.cache/huggingface
abstractcore --set-local-models-cache-dir ~/.abstractcore/models
```

**Default cache locations:**
- Default cache: `~/.cache/abstractcore`
- HuggingFace cache: `~/.cache/huggingface`
- Local models: `~/.abstractcore/models`

### Logging Configuration

Control logging behavior across all AbstractCore components:

#### Setting Log Levels

```bash
# Change console logging level (what you see in terminal)
abstractcore --set-console-log-level DEBUG    # Show all messages
abstractcore --set-console-log-level INFO     # Show info and above
abstractcore --set-console-log-level WARNING  # Show warnings and errors only (default)
abstractcore --set-console-log-level ERROR    # Show only errors
abstractcore --set-console-log-level CRITICAL # Show only critical errors
abstractcore --set-console-log-level NONE     # Disable all console logging

# Change file logging level (when file logging is enabled)
abstractcore --set-file-log-level DEBUG
abstractcore --set-file-log-level INFO
abstractcore --set-file-log-level NONE       # Disable all file logging
```

#### File Logging Controls

```bash
# Enable/disable file logging
abstractcore --enable-file-logging      # Start saving logs to files
abstractcore --disable-file-logging     # Stop saving logs to files

# Set log file location
abstractcore --set-log-base-dir ~/.abstractcore/logs
abstractcore --set-log-base-dir /var/log/abstractcore
```

#### Quick Logging Commands

```bash
# Enable debug mode (sets both console and file to DEBUG)
abstractcore --enable-debug-logging

# Disable console output (keeps file logging if enabled)
abstractcore --disable-console-logging

# Check current logging settings
abstractcore --status  # Shows current levels with change commands
```

**Available log levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE

**Log level descriptions:**
- **DEBUG**: Show all messages including detailed diagnostics
- **INFO**: Show informational messages and above
- **WARNING**: Show warnings, errors, and critical messages (default for console)
- **ERROR**: Show only errors and critical messages
- **CRITICAL**: Show only critical errors
- **NONE**: Disable all logging completely

**Default logging settings:**
- Console level: WARNING
- File level: DEBUG
- File logging: Disabled by default
- Log base directory: `~/.abstractcore/logs`

### Vision Configuration

Configure vision fallback for text-only models:

```bash
# Set vision model
abstractcore --set-vision-caption huggingface/Salesforce/blip-image-captioning-base
abstractcore --set-vision-provider huggingface Salesforce/blip-image-captioning-base

# Disable vision fallback
abstractcore --disable-vision
```

### API Keys

Manage API keys for different providers:

```bash
# Set API keys
abstractcore --set-api-key openai sk-your-key-here
abstractcore --set-api-key anthropic your-anthropic-key

# List API key status
abstractcore --list-api-keys
```

### Streaming Configuration

Configure default streaming behavior for CLI:

```bash
# Set streaming behavior
abstractcore --stream on           # Enable streaming by default
abstractcore --stream off          # Disable streaming by default

# Alternative commands
abstractcore --enable-streaming    # Enable streaming by default
abstractcore --disable-streaming   # Disable streaming by default
```

**Note**: Streaming only affects CLI behavior. Apps (summarizer, extractor, judge, intent) don't support streaming because they need complete structured outputs.

## Priority System

AbstractCore uses a clear priority hierarchy for configuration:

1. **Explicit Parameters** (highest priority)
   ```bash
   summarizer document.txt --provider openai --model gpt-4o-mini
   ```

2. **App-Specific Configuration**
   ```bash
   abstractcore --set-app-default summarizer openai gpt-4o-mini
   ```

3. **Global Configuration**
   ```bash
   abstractcore --set-global-default openai/gpt-4o-mini
   ```

4. **Hardcoded Defaults** (lowest priority)
   - Used when no configuration is available
   - Current default: `huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF`

## Debug Mode

The `--debug` parameter overrides configured logging levels and shows detailed diagnostics:

```bash
# Enable debug mode in apps
summarizer document.txt --debug
extractor data.txt --debug

# Debug output shows:
# 🐛 Debug - Configuration details:
#    Provider: huggingface
#    Model: unsloth/Qwen3-4B-Instruct-2507-GGUF
#    Config source: configured defaults
#    Max tokens: 32000
#    ...
```

## Configuration Status

View complete configuration status:

```bash
abstractcore --status
```

This displays:
- Application defaults for each app
- Global fallback settings
- Vision configuration
- Embeddings settings
- API key status
- Cache directories
- Logging configuration
- Configuration file location

## Interactive Configuration

Set up configuration interactively:

```bash
abstractcore --configure
```

This guides you through:
- Default model selection
- Vision fallback setup
- API key configuration

## Example Workflows

### Initial Setup

```bash
# 1. Check current status
abstractcore --status

# 2. Set global fallback
abstractcore --set-global-default ollama/llama3:8b

# 3. Configure specific apps for optimal performance
abstractcore --set-app-default summarizer openai gpt-4o-mini
abstractcore --set-app-default extractor ollama qwen3:4b-instruct
abstractcore --set-app-default judge anthropic claude-3-5-haiku

# 4. Set API keys as needed
abstractcore --set-api-key openai sk-your-key-here

# 5. Configure logging for development
abstractcore --enable-debug-logging
abstractcore --enable-file-logging

# 6. Enable streaming for interactive CLI
abstractcore --stream on

# 7. Verify configuration
abstractcore --status
```

### Development Environment

```bash
# Enable verbose logging for development
abstractcore --set-console-log-level DEBUG
abstractcore --enable-file-logging
abstractcore --set-log-base-dir ./logs

# Use local models to avoid API costs
abstractcore --set-global-default ollama/llama3:8b
abstractcore --set-app-default summarizer ollama qwen3:4b-instruct
```

### Production Environment

```bash
# Use production API services
abstractcore --set-global-default openai/gpt-4o-mini
abstractcore --set-api-key openai $OPENAI_API_KEY

# Set production logging
abstractcore --set-console-log-level WARNING
abstractcore --set-file-log-level INFO
abstractcore --enable-file-logging
abstractcore --set-log-base-dir /var/log/abstractcore
```

## Configuration File Format

The configuration is stored as JSON in `~/.abstractcore/config/abstractcore.json`:

```json
{
  "vision": {
    "strategy": "two_stage",
    "caption_provider": "huggingface",
    "caption_model": "Salesforce/blip-image-captioning-base",
    "fallback_chain": [
      {
        "provider": "huggingface",
        "model": "Salesforce/blip-image-captioning-base"
      }
    ],
    "local_models_path": "~/.abstractcore/models/"
  },
  "defaults": {
    "global_provider": "ollama",
    "global_model": "llama3:8b",
    "chat_model": null,
    "instruct_model": null,
    "code_model": null
  },
  "app_defaults": {
    "cli_provider": "huggingface",
    "cli_model": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    "summarizer_provider": "openai",
    "summarizer_model": "gpt-4o-mini",
    "extractor_provider": "ollama",
    "extractor_model": "qwen3:4b-instruct",
    "judge_provider": "anthropic",
    "judge_model": "claude-3-5-haiku"
  },
  "embeddings": {
    "provider": "huggingface",
    "model": "all-minilm-l6-v2",
    "api_key": null,
    "local_model_path": null
  },
  "api_keys": {
    "openai": null,
    "anthropic": null,
    "google": null,
    "cohere": null,
    "huggingface": null
  },
  "cache": {
    "default_cache_dir": "~/.cache/abstractcore",
    "huggingface_cache_dir": "~/.cache/huggingface",
    "local_models_cache_dir": "~/.abstractcore/models"
  },
  "logging": {
    "console_level": "WARNING",
    "file_level": "DEBUG",
    "log_base_dir": "~/.abstractcore/logs",
    "file_logging_enabled": false,
    "verbatim_enabled": true,
    "console_json": false,
    "file_json": true
  },
  "streaming": {
    "cli_stream_default": false
  },
  "provider_preferences": {}
}
```

## Configuration Parameter Reference

### Vision Section
- **strategy**: Vision fallback strategy (`"two_stage"`, `"disabled"`, `"basic_metadata"`)
- **caption_provider**: Provider for vision model (e.g., `"huggingface"`, `"ollama"`)
- **caption_model**: Vision model name (e.g., `"Salesforce/blip-image-captioning-base"`)
- **fallback_chain**: Array of backup vision models to try if primary fails
- **local_models_path**: Directory for local vision model storage

### Defaults Section (Global Fallbacks)
- **global_provider**: Default provider when app-specific not set (e.g., `"ollama"`)
- **global_model**: Default model when app-specific not set (e.g., `"llama3:8b"`)
- **chat_model**: Specialized model for chat applications (optional)
- **instruct_model**: Specialized model for instruction-following (optional)
- **code_model**: Specialized model for code generation (optional)

### App Defaults Section (Per-Application)
- **cli_provider** / **cli_model**: Default for CLI utility
- **summarizer_provider** / **summarizer_model**: Default for document summarization
- **extractor_provider** / **extractor_model**: Default for entity extraction
- **judge_provider** / **judge_model**: Default for text evaluation

### Embeddings Section
- **provider**: Embeddings provider (`"huggingface"`, `"openai"`, etc.)
- **model**: Embeddings model name (e.g., `"all-minilm-l6-v2"`)
- **api_key**: API key for embeddings provider (if required)
- **local_model_path**: Path to local embeddings model (if applicable)

### API Keys Section
- **openai**: OpenAI API key
- **anthropic**: Anthropic API key
- **google**: Google API key
- **cohere**: Cohere API key
- **huggingface**: HuggingFace API key

### Cache Section
- **default_cache_dir**: General cache directory for AbstractCore (`~/.cache/abstractcore`)
- **huggingface_cache_dir**: HuggingFace models cache (`~/.cache/huggingface`)
- **local_models_cache_dir**: Local models storage (`~/.abstractcore/models`)

### Logging Section
- **console_level**: Console log level (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`, `"NONE"`)
- **file_level**: File log level (same options as console_level)
- **log_base_dir**: Directory for log files (`~/.abstractcore/logs`)
- **file_logging_enabled**: Whether to save logs to files (`true`/`false`)
- **verbatim_enabled**: Whether to capture full prompts/responses (`true`/`false`)
- **console_json**: Use JSON format for console output (`true`/`false`)
- **file_json**: Use JSON format for file output (`true`/`false`)

### Streaming Section
- **cli_stream_default**: Default streaming mode for CLI (`true`/`false`)

### Provider Preferences Section
- **provider_preferences**: Additional provider-specific settings (key-value pairs)

## Common Configuration Tasks

### How to Change Console Log Level

If you see "Console Level: DEBUG" in the status and want to change it:

```bash
# To reduce console output (recommended for normal use)
abstractcore --set-console-log-level WARNING

# To see more information during development
abstractcore --set-console-log-level INFO

# To see all debug information
abstractcore --set-console-log-level DEBUG

# To completely disable console logging
abstractcore --set-console-log-level NONE

# Verify the change
abstractcore --status
```

### How to Enable File Logging

To start saving logs to files:

```bash
# Enable file logging (saves to ~/.abstractcore/logs by default)
abstractcore --enable-file-logging

# Optional: change log directory first
abstractcore --set-log-base-dir /path/to/your/logs
abstractcore --enable-file-logging

# Verify file logging is enabled
abstractcore --status
```

### How to Set Up Debug Mode

For troubleshooting, enable debug mode:

```bash
# Enable debug for both console and file logging
abstractcore --enable-debug-logging

# This is equivalent to:
# abstractcore --set-console-log-level DEBUG
# abstractcore --set-file-log-level DEBUG
# abstractcore --enable-file-logging
```

### How to Completely Disable Logging

To turn off all logging output:

```bash
# Disable console logging completely
abstractcore --set-console-log-level NONE

# Disable file logging completely (if enabled)
abstractcore --set-file-log-level NONE
abstractcore --disable-file-logging

# Note: --debug parameter in apps will still override NONE
# This maintains the priority system: explicit parameters > config defaults
```

## Troubleshooting

### Configuration Not Loading

If apps don't use configured defaults:

1. Check configuration file exists:
   ```bash
   ls -la ~/.abstractcore/config/abstractcore.json
   ```

2. Verify configuration content:
   ```bash
   abstractcore --status
   ```

3. Reset configuration if corrupted:
   ```bash
   rm ~/.abstractcore/config/abstractcore.json
   abstractcore --configure
   ```

### Model Initialization Failures

When models fail to initialize, apps show configuration guidance:

```
[ERROR] Failed to initialize LLM 'openai/gpt-4o-mini': API key not configured

[INFO] Solutions:
   - Set API key: abstractcore --set-api-key openai sk-...
   - Use different provider: summarizer document.txt --provider ollama --model llama3:8b

🔧 Or configure a different default:
   - abstractcore --set-app-default summarizer ollama llama3:8b
   - abstractcore --status
```

### Debug Information

Use `--debug` to see detailed configuration information:

```bash
summarizer document.txt --debug
```

This shows:
- Which configuration source is being used
- Exact provider and model values
- All parameter values
- Configuration file location