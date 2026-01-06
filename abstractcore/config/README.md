# Configuration Module

## Purpose

The `abstractcore/config/` module provides centralized configuration management for AbstractCore, enabling users to:

- Set default models and providers for all applications
- Configure API keys for cloud providers
- Manage vision fallback for text-only models with images
- Control embeddings settings for semantic search
- Configure logging, caching, and timeout behaviors
- Set application-specific defaults (CLI, summarizer, extractor, judge, intent)
- Manage offline-first behavior

Configuration is persistent across sessions via JSON file storage at `~/.abstractcore/config/abstractcore.json`, providing a consistent environment while allowing explicit parameter overrides.

## Architecture Position

**Layer**: Core Infrastructure (Configuration Layer)

**Position in Stack**:
```
User CLI/Applications
        ↓
Configuration Manager (this module)
        ↓
    ┌───┴───┬─────────┬──────────┐
Providers  Media  Processing  Apps
```

**Dependencies**:
- Standard library: `json`, `pathlib`, `argparse`, `dataclasses`
- No AbstractCore internal dependencies (isolated configuration)

**Used By**:
- `abstractcore/cli/main.py` - CLI command interface
- `abstractcore/apps/*` - All applications (summarizer, extractor, judge, intent)
- `abstractcore/providers/*` - Provider initialization with API keys
- `abstractcore/media/auto_handler.py` - Vision fallback configuration
- `abstractcore/utils/cli.py` - Interactive CLI defaults

## Component Structure

The configuration module consists of three core components:

```
abstractcore/config/
├── __init__.py           # Public API exports
├── manager.py            # ConfigurationManager singleton
├── main.py              # CLI commands and configuration logic
└── vision_config.py     # Vision-specific CLI commands (deprecated)
```

### File Descriptions

1. **`manager.py`** (450 lines)
   - Core configuration manager with singleton pattern
   - Dataclass-based configuration structures
   - JSON persistence and loading logic
   - Configuration getter/setter methods

2. **`main.py`** (843 lines)
   - CLI argument definitions and command handlers
   - Interactive configuration wizard
   - Comprehensive status display (`--status`)
   - Vision model download functionality

3. **`vision_config.py`** (491 lines)
   - Legacy vision configuration CLI commands
   - Primarily deprecated in favor of unified config system
   - Provider auto-detection from model names

## Detailed Components

### manager.py - Configuration Manager

**Core Classes**:

#### Configuration Data Classes

```python
@dataclass
class VisionConfig:
    """Vision fallback configuration for text-only models with images."""
    strategy: str = "disabled"  # "two_stage", "disabled", "basic_metadata"
    caption_provider: Optional[str] = None  # e.g., "ollama", "openai"
    caption_model: Optional[str] = None     # e.g., "qwen2.5vl:7b"
    fallback_chain: list = None             # Multiple fallback providers
    local_models_path: Optional[str] = None # Path to downloaded models

@dataclass
class EmbeddingsConfig:
    """Embeddings configuration for semantic search."""
    provider: Optional[str] = "huggingface"
    model: Optional[str] = "all-minilm-l6-v2"

@dataclass
class AppDefaults:
    """Per-application default configurations."""
    cli_provider: Optional[str] = "huggingface"
    cli_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    summarizer_provider: Optional[str] = "huggingface"
    summarizer_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    extractor_provider: Optional[str] = "huggingface"
    extractor_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    judge_provider: Optional[str] = "huggingface"
    judge_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    intent_provider: Optional[str] = "huggingface"
    intent_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"

@dataclass
class DefaultModels:
    """Global default model configurations."""
    global_provider: Optional[str] = None
    global_model: Optional[str] = None
    chat_model: Optional[str] = None  # Specialized for chat
    code_model: Optional[str] = None  # Specialized for coding

@dataclass
class ApiKeysConfig:
    """API keys for cloud providers."""
    openai: Optional[str] = None
    anthropic: Optional[str] = None
    google: Optional[str] = None

@dataclass
class CacheConfig:
    """Cache directory configurations."""
    default_cache_dir: str = "~/.cache/abstractcore"
    huggingface_cache_dir: str = "~/.cache/huggingface"
    local_models_cache_dir: str = "~/.abstractcore/models"
    glyph_cache_dir: str = "~/.abstractcore/glyph_cache"

@dataclass
class LoggingConfig:
    """Logging behavior configuration."""
    console_level: str = "WARNING"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    file_level: str = "DEBUG"
    file_logging_enabled: bool = False
    log_base_dir: Optional[str] = None
    verbatim_enabled: bool = True
    console_json: bool = False
    file_json: bool = True

@dataclass
class TimeoutConfig:
    """HTTP and tool execution timeouts."""
    default_timeout: float = 7200.0  # 2 hours for HTTP requests
    tool_timeout: float = 600.0     # 10 minutes for tool execution

@dataclass
class OfflineConfig:
    """Offline-first behavior configuration."""
    offline_first: bool = True              # Prefer local models
    allow_network: bool = False             # Allow network for API providers
    force_local_files_only: bool = True     # Force HuggingFace local-only
```

#### ConfigurationManager Class

**Singleton Pattern**:
```python
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager
```

**Key Methods**:

- `_load_config()` - Load configuration from JSON or create default
- `_save_config()` - Persist configuration to JSON file
- `set_vision_provider(provider, model)` - Configure vision fallback
- `set_global_default_model(provider_model)` - Set global default
- `set_app_default(app_name, provider, model)` - Set app-specific default
- `get_app_default(app_name)` - Retrieve app defaults
- `set_api_key(provider, key)` - Store API key
- `get_status()` - Return comprehensive configuration status
- `set_default_timeout(timeout)` - Configure HTTP timeout
- `set_tool_timeout(timeout)` - Configure tool execution timeout
- `is_offline_first()` - Check offline-first mode
- `should_force_local_files_only()` - Check HuggingFace local-only enforcement

**Configuration Persistence**:
- Location: `~/.abstractcore/config/abstractcore.json`
- Format: JSON with nested configuration sections
- Auto-creates directory structure on first save
- Graceful fallback to defaults if JSON parsing fails

### main.py - CLI Configuration Interface

**Core Functions**:

#### `add_arguments(parser: argparse.ArgumentParser)`

Defines all CLI configuration arguments organized into groups:

1. **General Configuration**:
   - `--status` - Show current configuration
   - `--configure` - Interactive guided setup
   - `--reset` - Reset to defaults

2. **Model Configuration**:
   - `--set-global-default PROVIDER/MODEL` - Set fallback for all apps
   - `--set-app-default APP PROVIDER MODEL` - Set app-specific model
   - `--set-chat-model PROVIDER/MODEL` - Specialized chat model
   - `--set-code-model PROVIDER/MODEL` - Specialized code model

3. **Authentication**:
   - `--set-api-key PROVIDER KEY` - Configure API keys
   - `--list-api-keys` - Show API key status

4. **Media & Vision Configuration**:
   - `--set-vision-provider PROVIDER MODEL` - Configure vision fallback
   - `--add-vision-fallback PROVIDER MODEL` - Add backup vision provider
   - `--download-vision-model [MODEL]` - Download local vision model
   - `--disable-vision` - Disable vision fallback

5. **Embeddings Configuration**:
   - `--set-embeddings-model MODEL` - Configure embeddings
   - `--set-embeddings-provider PROVIDER` - Set embeddings provider

6. **Storage & Logging**:
   - `--set-default-cache-dir PATH` - Set cache directory
   - `--set-console-log-level LEVEL` - Console logging level
   - `--set-file-log-level LEVEL` - File logging level
   - `--enable-debug-logging` - Enable debug mode
   - `--enable-file-logging` / `--disable-file-logging` - Toggle file logs

7. **Streaming Configuration**:
   - `--stream on/off` - Set streaming behavior
   - `--enable-streaming` / `--disable-streaming` - Toggle streaming

8. **Timeout Configuration**:
   - `--set-default-timeout SECONDS` - HTTP request timeout
   - `--set-tool-timeout SECONDS` - Tool execution timeout

#### `print_status()`

Displays comprehensive configuration status with hierarchical organization:

```
📋 AbstractCore Default Configuration Status
   (Explicit parameters in commands override these defaults)
===========================================================================

┌─ ESSENTIAL CONFIGURATION
│
│  🎯 Application Defaults
│     ✅ CLI          huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF
│     ✅ Summarizer   huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF
│     ...
│
│  🌐 Global Fallback
│     ⚠️  Default      Using built-in default (huggingface/...)
│
│  🔑 Provider Access
│     ✅ Configured    openai, anthropic
│     ⚠️  Missing keys  google
└─

┌─ SECONDARY CONFIGURATION
│
│  👁️  Media Processing
│     ✅ Ready         Smart captioning for text-only models
│     📷 Vision Model  ollama/qwen2.5vl:7b
│
│  🔗 Embeddings
│     ✅ Ready         huggingface/all-minilm-l6-v2
│
│  🌊 Streaming
│     ⚠️ Disabled      Complete response display by default
└─

┌─ ADVANCED CONFIGURATION
│
│  📝 Logging
│     ✅ Console       WARNING
│     ✅ File          DEBUG
│     📊 Summary       Dual logging active
│
│  💾 Storage
│     ✅ Configured    Cache: ~/.cache/abstractcore
│
│  ⏱️  Timeouts
│     ⏱️  HTTP Requests  10m (600s)
│     🔧 Tool Execution 10m (600s)
└─
```

**Status Display Features**:
- Three-tier hierarchy (Essential → Secondary → Advanced)
- Visual status indicators (✅ configured, ⚠️ fallback/missing, ❌ disabled)
- User-friendly descriptions instead of technical terms
- Separated help section with common commands
- Context header explaining default vs. explicit parameters

#### `interactive_configure()`

Interactive wizard for first-time configuration:

```python
def interactive_configure():
    """Interactive configuration setup."""
    # Guides user through:
    # 1. Default model setup
    # 2. Vision fallback configuration
    # 3. API keys setup
```

**User Experience**:
- Prompts for common configuration needs
- Provides sensible suggestions and examples
- Validates inputs before saving
- Shows confirmation messages

#### `download_vision_model(model_name: str)`

Downloads local vision models for offline use:

**Available Models**:
- `blip-base-caption` (990MB) - Salesforce BLIP base
- `blip-large-caption` (1.8GB) - Salesforce BLIP large
- `vit-gpt2` (500MB) - ViT + GPT-2 captioning
- `git-base` (400MB) - Microsoft GIT base

**Download Process**:
1. Validates model availability
2. Auto-installs transformers if missing
3. Downloads from HuggingFace Hub
4. Saves to `~/.abstractcore/models/{model_name}/`
5. Creates download marker file
6. Auto-configures vision fallback

### vision_config.py - Legacy Vision CLI

**Status**: Primarily deprecated in favor of unified configuration in `main.py`.

**Remaining Functions**:
- `handle_vision_commands(args)` - Process vision-specific arguments
- `add_vision_arguments(parser)` - Add vision argument group
- `detect_provider_from_model(model)` - Auto-detect provider from model name

**Legacy Commands** (use `main.py` equivalents instead):
- `--set-vision-caption MODEL` → Use `--set-vision-provider PROVIDER MODEL`
- `--vision-status` → Use `--status`
- `--list-vision` → Documentation provides this information

## Configuration Options

### Complete Configuration Structure

```json
{
  "vision": {
    "strategy": "two_stage",
    "caption_provider": "ollama",
    "caption_model": "qwen2.5vl:7b",
    "fallback_chain": [],
    "local_models_path": null
  },
  "embeddings": {
    "provider": "huggingface",
    "model": "all-minilm-l6-v2"
  },
  "app_defaults": {
    "cli_provider": "lmstudio",
    "cli_model": "qwen/qwen3-next-80b",
    "summarizer_provider": "openai",
    "summarizer_model": "gpt-4o-mini",
    "extractor_provider": "ollama",
    "extractor_model": "qwen3:4b-instruct",
    "judge_provider": "anthropic",
    "judge_model": "claude-3-5-haiku",
    "intent_provider": "huggingface",
    "intent_model": "unsloth/Qwen3-4B-Instruct-2507-GGUF"
  },
  "default_models": {
    "global_provider": "ollama",
    "global_model": "llama3:8b",
    "chat_model": "openai/gpt-4o-mini",
    "code_model": "anthropic/claude-3-5-sonnet"
  },
  "api_keys": {
    "openai": "sk-...",
    "anthropic": "ant_...",
    "google": null
  },
  "cache": {
    "default_cache_dir": "~/.cache/abstractcore",
    "huggingface_cache_dir": "~/.cache/huggingface",
    "local_models_cache_dir": "~/.abstractcore/models",
    "glyph_cache_dir": "~/.abstractcore/glyph_cache"
  },
  "logging": {
    "console_level": "WARNING",
    "file_level": "DEBUG",
    "file_logging_enabled": true,
    "log_base_dir": null,
    "verbatim_enabled": true,
    "console_json": false,
    "file_json": true
  },
  "timeouts": {
    "default_timeout": 7200.0,
    "tool_timeout": 600.0
  },
  "offline": {
    "offline_first": true,
    "allow_network": false,
    "force_local_files_only": true
  }
}
```

## CLI Commands

### Essential Commands

**Check Current Configuration**:
```bash
abstractcore --status
```

**Interactive Setup**:
```bash
abstractcore --configure
```

**Reset to Defaults**:
```bash
abstractcore --reset
```

### Model Configuration

**Set Global Default** (used by all apps unless overridden):
```bash
abstractcore --set-global-default ollama/llama3:8b
```

**Set App-Specific Defaults**:
```bash
# High-quality summarization with OpenAI
abstractcore --set-app-default summarizer openai gpt-4o-mini

# Fast extraction with local model
abstractcore --set-app-default extractor ollama qwen3:4b-instruct

# Powerful coding with Claude
abstractcore --set-app-default cli lmstudio qwen/qwen3-next-80b
```

**Available Apps**:
- `cli` - Interactive CLI (`python -m abstractcore.utils.cli`)
- `summarizer` - Document summarization
- `extractor` - Entity/relationship extraction
- `judge` - Text evaluation and scoring
- `intent` - Intent analysis

**Set Specialized Models** (optional):
```bash
# Chat-optimized model
abstractcore --set-chat-model openai/gpt-4o-mini

# Coding-optimized model
abstractcore --set-code-model anthropic/claude-3-5-sonnet
```

### Authentication

**Configure API Keys**:
```bash
abstractcore --set-api-key openai sk-your-key-here
abstractcore --set-api-key anthropic ant-your-key-here
abstractcore --set-api-key google your-google-key
```

**Check API Key Status**:
```bash
abstractcore --list-api-keys
```

### Vision Configuration

**Set Vision Provider** (for text-only models with images):
```bash
# Local vision model via Ollama
abstractcore --set-vision-provider ollama qwen2.5vl:7b

# Cloud vision via OpenAI
abstractcore --set-vision-provider openai gpt-4o

# Cloud vision via Anthropic
abstractcore --set-vision-provider anthropic claude-3-5-sonnet
```

**Download Local Vision Model**:
```bash
# Default model (blip-base-caption, ~990MB)
abstractcore --download-vision-model

# Specific model
abstractcore --download-vision-model git-base  # Smallest (400MB)
abstractcore --download-vision-model vit-gpt2  # CPU-friendly (500MB)
```

**Add Fallback Vision Providers**:
```bash
abstractcore --add-vision-fallback openai gpt-4o-mini
```

**Disable Vision Fallback**:
```bash
abstractcore --disable-vision
```

### Embeddings Configuration

**Set Embeddings Model**:
```bash
# Local embeddings (HuggingFace)
abstractcore --set-embeddings-model huggingface/all-minilm-l6-v2

# Cloud embeddings (OpenAI)
abstractcore --set-embeddings-model openai/text-embedding-3-small
```

### Logging Configuration

**Enable Debug Logging**:
```bash
abstractcore --enable-debug-logging
```

**Set Console Log Level**:
```bash
abstractcore --set-console-log-level DEBUG    # Verbose
abstractcore --set-console-log-level INFO     # Informational
abstractcore --set-console-log-level WARNING  # Default
abstractcore --set-console-log-level ERROR    # Errors only
abstractcore --set-console-log-level CRITICAL # Critical only
abstractcore --set-console-log-level NONE     # Disable console
```

**Set File Log Level**:
```bash
abstractcore --set-file-log-level DEBUG
```

**Enable/Disable File Logging**:
```bash
abstractcore --enable-file-logging
abstractcore --disable-file-logging
```

**Disable Console Logging**:
```bash
abstractcore --disable-console-logging
```

### Storage Configuration

**Set Cache Directories**:
```bash
abstractcore --set-default-cache-dir /path/to/cache
abstractcore --set-huggingface-cache-dir /path/to/hf-cache
abstractcore --set-local-models-cache-dir /path/to/models
abstractcore --set-log-base-dir /path/to/logs
```

### Timeout Configuration

**Set HTTP Request Timeout**:
```bash
# 2 hours (default)
abstractcore --set-default-timeout 7200

# 5 minutes
abstractcore --set-default-timeout 300

# 10 minutes
abstractcore --set-default-timeout 600

# 30 minutes for slow connections
abstractcore --set-default-timeout 1800
```

**Set Tool Execution Timeout**:
```bash
abstractcore --set-tool-timeout 600  # 10 minutes
```

### Streaming Configuration

**Enable/Disable Streaming**:
```bash
abstractcore --enable-streaming   # Real-time response display
abstractcore --disable-streaming  # Complete response display
```

**Toggle Streaming**:
```bash
abstractcore --stream on
abstractcore --stream off
```

## Usage Patterns

### Programmatic Configuration

**Access Configuration Manager**:
```python
from abstractcore.config import get_config_manager

# Get singleton instance
config_manager = get_config_manager()

# Read configuration
status = config_manager.get_status()
print(status["vision"]["caption_provider"])  # e.g., "ollama"

# Get app defaults
provider, model = config_manager.get_app_default("summarizer")
print(f"Summarizer uses {provider}/{model}")
```

**Modify Configuration**:
```python
from abstractcore.config import get_config_manager

config_manager = get_config_manager()

# Set global default
config_manager.set_global_default_model("ollama/llama3:8b")

# Set app-specific default
config_manager.set_app_default("summarizer", "openai", "gpt-4o-mini")

# Configure vision
config_manager.set_vision_provider("ollama", "qwen2.5vl:7b")

# Set API key
config_manager.set_api_key("openai", "sk-your-key-here")

# Configure timeout
config_manager.set_default_timeout(900)  # 15 minutes
```

**Check Offline Configuration**:
```python
from abstractcore.config import get_config_manager

config_manager = get_config_manager()

# Check offline mode
if config_manager.is_offline_first():
    print("Running in offline-first mode")

# Check if network allowed
if config_manager.is_network_allowed():
    print("Network access permitted for cloud providers")

# Check HuggingFace local-only enforcement
if config_manager.should_force_local_files_only():
    print("HuggingFace will only use cached models")
```

**Access Timeout Configuration**:
```python
from abstractcore.config import get_config_manager

config_manager = get_config_manager()

# Get timeouts
http_timeout = config_manager.get_default_timeout()  # Default: 7200 seconds
tool_timeout = config_manager.get_tool_timeout()     # Default: 600 seconds

# Use in HTTP client
import httpx
client = httpx.Client(timeout=http_timeout)
```

### CLI Configuration

**First-Time Setup**:
```bash
# Run interactive setup
abstractcore --configure

# Or set specific values
abstractcore --set-global-default ollama/llama3:8b
abstractcore --set-vision-provider ollama qwen2.5vl:7b
abstractcore --set-api-key openai sk-your-key-here
```

**Check Configuration**:
```bash
# View comprehensive status
abstractcore --status

# Check API keys
abstractcore --list-api-keys
```

**Per-Application Configuration**:
```bash
# Set different models for different needs
abstractcore --set-app-default cli lmstudio qwen/qwen3-next-80b
abstractcore --set-app-default summarizer openai gpt-4o-mini
abstractcore --set-app-default extractor ollama qwen3:4b-instruct
abstractcore --set-app-default judge anthropic claude-3-5-haiku
```

### Vision Configuration Patterns

**Local Vision Model (Ollama)**:
```bash
# Prerequisite: Pull model in Ollama
ollama pull qwen2.5vl:7b

# Configure AbstractCore
abstractcore --set-vision-provider ollama qwen2.5vl:7b

# Test with text-only model + image
summarizer document-with-images.pdf --provider ollama --model llama3:8b
# Vision model automatically handles images, text model handles text
```

**Cloud Vision Model (OpenAI)**:
```bash
# Set API key
abstractcore --set-api-key openai sk-your-key-here

# Configure vision
abstractcore --set-vision-provider openai gpt-4o

# Use text-only model with images
summarizer document.pdf --provider anthropic --model claude-3-5-haiku
# OpenAI vision processes images, Claude processes text
```

**Downloaded Local Vision Model**:
```bash
# Download lightweight model for offline use
abstractcore --download-vision-model git-base

# Auto-configured, ready to use
summarizer image-document.pdf
```

**Multiple Fallbacks**:
```bash
# Primary vision
abstractcore --set-vision-provider ollama qwen2.5vl:7b

# Fallback if primary unavailable
abstractcore --add-vision-fallback openai gpt-4o-mini
```

## Integration Points

### How Configuration is Used

**Applications (apps/)**:
```python
from abstractcore.apps.app_config_utils import get_app_defaults

def main():
    # Applications read their defaults from config
    provider, model = get_app_defaults("summarizer")

    # Explicit parameters override config
    if args.provider:
        provider = args.provider
    if args.model:
        model = args.model
```

**Providers (providers/)**:
```python
from abstractcore.config import get_config_manager

class OpenAIProvider:
    def __init__(self, api_key: Optional[str] = None):
        # Read API key from config if not provided
        if not api_key:
            config_manager = get_config_manager()
            api_key = config_manager.config.api_keys.openai
```

**Media Processing (media/auto_handler.py)**:
```python
from abstractcore.config import get_config_manager

config_manager = get_config_manager()
vision_config = config_manager.config.vision

# Use vision configuration
if vision_config.strategy == "two_stage":
    caption_provider = vision_config.caption_provider
    caption_model = vision_config.caption_model
```

**CLI Utilities (utils/cli.py)**:
```python
from abstractcore.config import get_config_manager

config_manager = get_config_manager()

# Get default for interactive CLI
provider, model = config_manager.get_app_default("cli")

# Allow override from arguments
if args.provider:
    provider = args.provider
```

### Configuration Priority System

AbstractCore follows a clear priority hierarchy:

**Priority Levels** (highest to lowest):

1. **Explicit Parameters** (highest priority):
   ```python
   llm = create_llm("openai", model="gpt-4o", api_key="sk-...")
   ```
   ```bash
   summarizer doc.pdf --provider openai --model gpt-4o
   ```

2. **App-Specific Configuration**:
   ```bash
   abstractcore --set-app-default summarizer openai gpt-4o-mini
   ```

3. **Global Configuration**:
   ```bash
   abstractcore --set-global-default ollama/llama3:8b
   ```

4. **Built-in Defaults** (lowest priority):
   - Default: `huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF`
   - Ensures AbstractCore always has a working default

**Example Priority Resolution**:
```python
# Config file has:
# - global_default: ollama/llama3:8b
# - app_default[summarizer]: openai/gpt-4o-mini

# Command: summarizer doc.pdf
# Result: Uses openai/gpt-4o-mini (app-specific)

# Command: summarizer doc.pdf --provider anthropic --model claude-3-5-haiku
# Result: Uses anthropic/claude-3-5-haiku (explicit parameters)

# Command: judge essay.md
# Result: Uses ollama/llama3:8b (global default, no judge-specific config)
```

## Best Practices

### DOs

1. **DO** use `--status` frequently to verify configuration:
   ```bash
   abstractcore --status
   ```

2. **DO** set app-specific defaults for different quality/cost needs:
   ```bash
   # Fast extraction with local model
   abstractcore --set-app-default extractor ollama qwen3:4b-instruct

   # High-quality summarization with cloud
   abstractcore --set-app-default summarizer openai gpt-4o-mini
   ```

3. **DO** configure vision fallback for mixed-content documents:
   ```bash
   abstractcore --set-vision-provider ollama qwen2.5vl:7b
   ```

4. **DO** use offline-first configuration for local deployments:
   ```python
   config_manager = get_config_manager()
   assert config_manager.is_offline_first()  # Default behavior
   ```

5. **DO** set appropriate timeouts for your use case:
   ```bash
   # Long-running operations
   abstractcore --set-default-timeout 1800  # 30 minutes

   # Fast API calls
   abstractcore --set-default-timeout 300   # 5 minutes
   ```

6. **DO** use API keys from configuration for cloud providers:
   ```bash
   abstractcore --set-api-key openai sk-your-key-here
   ```

7. **DO** enable file logging for debugging production issues:
   ```bash
   abstractcore --enable-file-logging
   abstractcore --set-file-log-level DEBUG
   ```

8. **DO** use global defaults as fallbacks:
   ```bash
   abstractcore --set-global-default ollama/llama3:8b
   ```

### DON'Ts

1. **DON'T** modify `~/.abstractcore/config/abstractcore.json` directly:
   ```bash
   # ❌ BAD: Direct JSON editing
   vim ~/.abstractcore/config/abstractcore.json

   # ✅ GOOD: Use CLI commands
   abstractcore --set-global-default ollama/llama3:8b
   ```

2. **DON'T** hardcode API keys in application code:
   ```python
   # ❌ BAD: Hardcoded API key
   llm = create_llm("openai", api_key="sk-hardcoded-key")

   # ✅ GOOD: Use configuration
   llm = create_llm("openai")  # Reads from config
   ```

3. **DON'T** assume configuration exists without fallback:
   ```python
   # ❌ BAD: No fallback
   config_manager = get_config_manager()
   provider = config_manager.config.default_models.global_provider
   # Could be None!

   # ✅ GOOD: Provide fallback
   from abstractcore.apps.app_config_utils import get_app_defaults
   provider, model = get_app_defaults("summarizer")
   # Always returns valid defaults
   ```

4. **DON'T** forget to check API key availability for cloud providers:
   ```python
   # ❌ BAD: Assume API key exists
   llm = create_llm("openai")

   # ✅ GOOD: Check first
   from abstractcore.config import get_config_manager
   config = get_config_manager()
   if not config.config.api_keys.openai:
       print("Please set OpenAI API key: abstractcore --set-api-key openai YOUR_KEY")
   ```

5. **DON'T** enable DEBUG console logging in production:
   ```bash
   # ❌ BAD: Verbose console output
   abstractcore --set-console-log-level DEBUG

   # ✅ GOOD: Minimal console, debug to file
   abstractcore --set-console-log-level WARNING
   abstractcore --enable-file-logging
   abstractcore --set-file-log-level DEBUG
   ```

6. **DON'T** use extremely short timeouts:
   ```bash
   # ❌ BAD: May cause premature failures
   abstractcore --set-default-timeout 10  # 10 seconds

   # ✅ GOOD: Reasonable timeout
   abstractcore --set-default-timeout 7200  # 2 hours (default)
   ```

7. **DON'T** set network-dependent configuration in offline environments:
   ```bash
   # ❌ BAD: Cloud provider as global default in air-gapped system
   abstractcore --set-global-default openai/gpt-4o

   # ✅ GOOD: Local provider in offline environment
   abstractcore --set-global-default ollama/llama3:8b
   ```

## Common Pitfalls

### Pitfall 1: Configuration Precedence Confusion

**Problem**: Users expect configuration to override explicit parameters.

**Wrong Expectation**:
```python
# Config has: global_default = "ollama/llama3:8b"
llm = create_llm("openai", model="gpt-4o")
# User expects: Uses ollama/llama3:8b from config
```

**Reality**:
```python
# Explicit parameters ALWAYS override configuration
llm = create_llm("openai", model="gpt-4o")
# Result: Uses openai/gpt-4o (explicit wins)
```

**Solution**: Remember the priority hierarchy:
```
Explicit Parameters > App Config > Global Config > Built-in Defaults
```

### Pitfall 2: Missing API Keys

**Problem**: Configuration shows provider but API key not set.

**Symptom**:
```bash
abstractcore --set-global-default openai/gpt-4o
# Later...
summarizer doc.pdf
# Error: OpenAI API key not found
```

**Solution**: Always set API keys for cloud providers:
```bash
abstractcore --set-api-key openai sk-your-key-here
abstractcore --status  # Verify "OpenAI: ✅ Set"
```

### Pitfall 3: Vision Provider Not Configured

**Problem**: Using text-only model with images without vision fallback.

**Symptom**:
```bash
summarizer document-with-images.pdf --provider ollama --model llama3:8b
# Warning: Model doesn't support images, skipping visual content
```

**Solution**: Configure vision fallback:
```bash
abstractcore --set-vision-provider ollama qwen2.5vl:7b
abstractcore --status  # Verify "Vision: ✅ Ready"
```

### Pitfall 4: Offline-First Conflicts

**Problem**: Trying to use cloud provider in strict offline mode.

**Symptom**:
```python
config_manager = get_config_manager()
config_manager.config.offline.offline_first = True
config_manager.config.offline.allow_network = False

llm = create_llm("openai", model="gpt-4o")
# May fail if network is disabled at system level
```

**Solution**: Enable network for cloud providers:
```python
config_manager.set_allow_network(True)
# Or use local providers only
```

### Pitfall 5: Timeout Too Short

**Problem**: Timeouts set too low for large model operations.

**Symptom**:
```bash
abstractcore --set-default-timeout 30  # 30 seconds
summarizer large-document.pdf --provider ollama --model llama3:8b
# Error: Request timeout after 30s
```

**Solution**: Use appropriate timeouts:
```bash
# Default 10 minutes is reasonable
abstractcore --set-default-timeout 600

# Increase for very large operations
abstractcore --set-default-timeout 1800  # 30 minutes
```

### Pitfall 6: Forgetting App-Specific Overrides

**Problem**: Setting global default but app-specific overrides still apply.

**Symptom**:
```bash
abstractcore --set-global-default ollama/llama3:8b
summarizer doc.pdf
# Still uses openai/gpt-4o-mini (app-specific config)
```

**Solution**: Check status to see effective configuration:
```bash
abstractcore --status
# Shows:
# Summarizer: openai/gpt-4o-mini  (app-specific)
# Global Fallback: ollama/llama3:8b
```

To change app-specific:
```bash
abstractcore --set-app-default summarizer ollama llama3:8b
```

### Pitfall 7: Cache Directory Permissions

**Problem**: Cache directory not writable.

**Symptom**:
```bash
abstractcore --set-default-cache-dir /root/cache
# Later...
# Error: Permission denied writing to cache
```

**Solution**: Use directories with appropriate permissions:
```bash
# Use home directory (default)
abstractcore --set-default-cache-dir ~/.cache/abstractcore

# Or create with proper permissions
sudo mkdir -p /opt/abstractcore-cache
sudo chown $USER /opt/abstractcore-cache
abstractcore --set-default-cache-dir /opt/abstractcore-cache
```

## Testing Strategy

### Unit Testing Configuration Manager

**Test Configuration Persistence**:
```python
import tempfile
from pathlib import Path
from abstractcore.config.manager import ConfigurationManager

def test_config_persistence():
    # Create temporary config directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / ".abstractcore" / "config"

        # Initialize config manager with temporary directory
        manager = ConfigurationManager()
        manager.config_dir = config_dir
        manager.config_file = config_dir / "abstractcore.json"

        # Set configuration
        manager.set_global_default_model("ollama/llama3:8b")
        manager.set_api_key("openai", "test-key")

        # Create new instance (simulates restart)
        manager2 = ConfigurationManager()
        manager2.config_dir = config_dir
        manager2.config_file = config_dir / "abstractcore.json"
        manager2.config = manager2._load_config()

        # Verify persistence
        assert manager2.config.default_models.global_provider == "ollama"
        assert manager2.config.default_models.global_model == "llama3:8b"
        assert manager2.config.api_keys.openai == "test-key"
```

**Test App Defaults**:
```python
from abstractcore.config.manager import ConfigurationManager

def test_app_defaults():
    manager = ConfigurationManager()

    # Set app-specific default
    manager.set_app_default("summarizer", "openai", "gpt-4o-mini")

    # Retrieve default
    provider, model = manager.get_app_default("summarizer")
    assert provider == "openai"
    assert model == "gpt-4o-mini"

    # Unknown app returns fallback
    provider, model = manager.get_app_default("unknown_app")
    assert provider == "huggingface"
    assert model == "unsloth/Qwen3-4B-Instruct-2507-GGUF"
```

**Test Configuration Priority**:
```python
from abstractcore.config.manager import ConfigurationManager
from abstractcore.apps.app_config_utils import get_app_defaults

def test_configuration_priority():
    manager = ConfigurationManager()

    # Set global default
    manager.set_global_default_model("ollama/llama3:8b")

    # Set app-specific default
    manager.set_app_default("summarizer", "openai", "gpt-4o-mini")

    # App-specific should win
    provider, model = get_app_defaults("summarizer")
    assert provider == "openai"
    assert model == "gpt-4o-mini"

    # App without specific config uses global
    manager.set_app_default("extractor", None, None)
    provider, model = get_app_defaults("extractor")
    # Should use global or built-in default
    assert provider in ["ollama", "huggingface"]
```

### Integration Testing CLI Commands

**Test CLI Status Command**:
```python
import subprocess
import json

def test_status_command():
    # Run status command
    result = subprocess.run(
        ["abstractcore", "--status"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "Configuration Status" in result.stdout
    assert "Application Defaults" in result.stdout
```

**Test Configuration Commands**:
```python
import subprocess

def test_set_global_default():
    # Set global default
    result = subprocess.run(
        ["abstractcore", "--set-global-default", "ollama/llama3:8b"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "Set global default to: ollama/llama3:8b" in result.stdout

    # Verify with status
    result = subprocess.run(
        ["abstractcore", "--status"],
        capture_output=True,
        text=True
    )

    assert "ollama/llama3:8b" in result.stdout
```

### Testing Best Practices

1. **Use Temporary Directories**: Don't modify actual configuration during tests
2. **Test Persistence**: Verify configuration survives restarts
3. **Test Defaults**: Ensure fallback defaults work when config unavailable
4. **Test Priority**: Verify configuration hierarchy (explicit > app > global > default)
5. **Test Error Handling**: Invalid inputs should fail gracefully
6. **Test CLI Commands**: Integration tests for all CLI commands
7. **Test Singleton**: Verify ConfigurationManager singleton behavior

## Public API

### Recommended Imports

```python
# Configuration manager (singleton)
from abstractcore.config import get_config_manager

# App defaults utility
from abstractcore.apps.app_config_utils import get_app_defaults
```

### Core API Functions

**Get Configuration Manager**:
```python
from abstractcore.config import get_config_manager

config_manager = get_config_manager()
# Returns singleton ConfigurationManager instance
```

**Get App Defaults** (recommended for applications):
```python
from abstractcore.apps.app_config_utils import get_app_defaults

provider, model = get_app_defaults("summarizer")
# Returns: Tuple[str, str] with provider and model
# Gracefully handles missing configuration with fallbacks
```

### ConfigurationManager Methods

**Configuration Getters**:
- `get_status() -> Dict[str, Any]` - Comprehensive configuration status
- `get_app_default(app_name: str) -> Tuple[str, str]` - Get app provider/model
- `get_default_timeout() -> float` - Get HTTP timeout in seconds
- `get_tool_timeout() -> float` - Get tool timeout in seconds
- `is_offline_first() -> bool` - Check offline-first mode
- `is_network_allowed() -> bool` - Check network access
- `should_force_local_files_only() -> bool` - Check HuggingFace local-only

**Configuration Setters**:
- `set_global_default_model(provider_model: str) -> bool` - Set global default
- `set_app_default(app: str, provider: str, model: str) -> bool` - Set app default
- `set_vision_provider(provider: str, model: str) -> bool` - Configure vision
- `set_api_key(provider: str, key: str) -> bool` - Store API key
- `set_default_timeout(timeout: float) -> bool` - Set HTTP timeout
- `set_tool_timeout(timeout: float) -> bool` - Set tool timeout
- `set_offline_first(enabled: bool) -> bool` - Enable offline mode
- `set_allow_network(enabled: bool) -> bool` - Allow network access

**Example Usage**:
```python
from abstractcore.config import get_config_manager

# Initialize
config_manager = get_config_manager()

# Read configuration
status = config_manager.get_status()
provider, model = config_manager.get_app_default("summarizer")
timeout = config_manager.get_default_timeout()

# Modify configuration
config_manager.set_global_default_model("ollama/llama3:8b")
config_manager.set_app_default("summarizer", "openai", "gpt-4o-mini")
config_manager.set_api_key("openai", "sk-your-key-here")
config_manager.set_default_timeout(900)  # 15 minutes

# Check offline mode
if config_manager.is_offline_first():
    print("Running in offline-first mode")
```

---

**Configuration File Location**: `~/.abstractcore/config/abstractcore.json`

**Documentation**: See `docs/centralized-config.md` for comprehensive usage guide

**Related Modules**:
- `abstractcore/apps/` - Applications that consume configuration
- `abstractcore/providers/` - Providers that read API keys from configuration
- `abstractcore/media/auto_handler.py` - Vision fallback configuration consumer
- `abstractcore/cli/main.py` - CLI entry point for configuration commands

## Related Modules

**Direct dependencies**:
- [`exceptions/`](../exceptions/README.md) - Configuration validation errors
- [`utils/`](../utils/README.md) - Path utilities, validation helpers

**Configuration consumers**:
- [`core/`](../core/README.md) - Factory default settings
- [`providers/`](../providers/README.md) - API keys, base URLs, model defaults
- [`media/`](../media/README.md) - Vision fallback configuration
- [`compression/`](../compression/README.md) - Glyph compression settings
- [`structured/`](../structured/README.md) - Retry strategies
- [`tools/`](../tools/README.md) - Tool execution settings
- [`embeddings/`](../embeddings/README.md) - Embedding model defaults
- [`server/`](../server/README.md) - Server configuration
- [`apps/`](../apps/README.md) - Application defaults

**Data sources**:
- [`assets/`](../assets/README.md) - Model capabilities for defaults
- [`architectures/`](../architectures/README.md) - Architecture detection
