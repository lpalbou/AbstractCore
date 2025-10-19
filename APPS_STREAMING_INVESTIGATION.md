# AbstractCore Streaming & Apps Investigation Report

**Investigation Date**: October 19, 2025  
**Thoroughness Level**: Very Thorough  
**Scope**: Apps directory, streaming architecture, configuration system integration

---

## EXECUTIVE SUMMARY

This investigation explores how streaming is currently used across the AbstractCore ecosystem, with specific focus on:
1. The three main applications (summarizer, extractor, judge)
2. Current streaming parameter handling and configuration
3. Integration with the configuration system
4. Relationship between CLI tools, apps, and structured logging

### Key Findings

- **Streaming is CLI/Server-focused**: Only interactive CLI and HTTP API support streaming
- **Apps don't stream by design**: Summarizer, Extractor, and Judge deliberately don't use streaming
- **Configuration system is ready**: `StreamingConfig` exists with per-app support infrastructure
- **Structured logging is independent**: Separate from streaming, captures generation metadata
- **Apps use configuration defaults**: All three main apps integrate with centralized config for provider/model selection

---

## 1. APPS DIRECTORY STRUCTURE & PURPOSE

### Location
`/Users/albou/projects/abstractcore/abstractcore/apps/`

### Contents
```
apps/
├── __init__.py
├── __main__.py
├── summarizer.py         (CLI app: document summarization)
├── extractor.py          (CLI app: entity & relationship extraction)
├── judge.py              (CLI app: LLM-as-a-judge evaluation)
└── app_config_utils.py   (Shared configuration utilities)
```

### Purpose
Each file is a **standalone CLI application** that can be invoked as:
- `python -m abstractcore.apps.summarizer <file> [options]`
- `python -m abstractcore.apps.extractor <file> [options]`
- `python -m abstractcore.apps.judge <content> [options]`

---

## 2. CURRENT STREAMING USAGE IN APPS

### 2.1 Summarizer (`summarizer.py`)

**Streaming Status**: ❌ **NOT USED** (by design)

**Why**:
- Needs to aggregate complete text across chunks
- Requires post-processing (key points, compression ratio calculation)
- Must return structured `SummaryResult` object with metadata

**Current Flow**:
```
File → Read content → Chunk → LLM.generate(stream=False) → BasicSummarizer
  ↓
Complete response → Parse summary + key points → Format output → Display/Save
```

**Configuration Integration** (Lines 38-52):
```python
def get_app_defaults(app_name: str) -> tuple[str, str]:
    """Get default provider and model for an app."""
    try:
        config_manager = get_config_manager()
        return config_manager.get_app_default(app_name)
    except Exception:
        # Fallback to hardcoded defaults
        return ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF')
```

**Key Parameters**:
- `--provider` / `--model`: Explicit override (line 231-238)
- `--chunk-size`: Processing chunk size (line 224)
- `--max-tokens`: Context window budget (line 241)
- `--max-output-tokens`: Generation budget (line 248)
- `--verbose`: Progress info
- `--debug`: Detailed diagnostics

**Configuration Resolution** (Lines 322-330):
```python
if args.provider and args.model:
    provider, model = args.provider, args.model
    config_source = "explicit parameters"
else:
    provider, model = get_app_defaults('summarizer')
    config_source = "configured defaults"
```

### 2.2 Extractor (`extractor.py`)

**Streaming Status**: ❌ **NOT USED** (by design)

**Why**:
- Needs complete JSON-LD graph output
- Performs iterative refinement (`--iterate N`)
- Must deduplicate entities and verify relationships
- Returns structured knowledge graph, not streaming content

**Current Flow**:
```
File → Read content → Extract once → (Optional: Refine N times)
  ↓
Complete JSON-LD → Format (JSON-LD/Triples/JSON/YAML) → Display/Save
```

**Configuration Integration** (Same pattern as Summarizer):
- Uses `get_app_defaults('extractor')`
- Hardcoded fallback: `('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF')`

**Key Parameters**:
- `--provider` / `--model`: Explicit override (line 261-268)
- `--mode`: Extraction mode fast/balanced/thorough (line 277-287)
- `--iterate N`: Refinement passes (line 290-294)
- `--format`: Output format (line 242-246)
- `--chunk-size`: Processing size (line 254)
- `--no-embeddings`: Disable deduplication (line 271)
- `--max-tokens` / `--max-output-tokens`: Token budgets

### 2.3 Judge (`judge.py`)

**Streaming Status**: ❌ **NOT USED** (by design)

**Why**:
- Needs complete evaluation object with scores
- Must calculate overall_score from individual dimensions
- Requires structured reasoning and feedback
- Returns assessment dictionary with multiple fields

**Current Flow**:
```
Content/File → Read → Initialize BasicJudge
  ↓
LLM.generate(stream=False) → Parse assessment
  ↓
Structured assessment object → Format (JSON/Plain/YAML) → Display/Save
```

**Configuration Integration** (Same pattern):
- Uses `get_app_defaults('judge')`
- Hardcoded fallback

**Key Parameters**:
- `--provider` / `--model`: Explicit override (line 330-337)
- `--criteria`: Evaluation criteria (line 298)
- `--temperature`: Eval temperature (line 340-344)
- `--format`: Output format (line 318-322)
- `--max-tokens` / `--max-output-tokens`: Token budgets
- `--include-criteria`: Include evaluation details

---

## 3. STREAMING PARAMETER HANDLING

### 3.1 Parameter Design Pattern

All three apps follow the same pattern:

```python
# 1. Parse arguments
parser.add_argument('--provider', help='LLM provider')
parser.add_argument('--model', help='LLM model')

# 2. Resolve configuration
if args.provider and args.model:
    provider, model = args.provider, args.model
    config_source = "explicit parameters"
else:
    provider, model = get_app_defaults(app_name)
    config_source = "configured defaults"

# 3. Log source for transparency
print(f"Using {config_source}: {provider}/{model}")

# 4. Create LLM and app
llm = create_llm(provider, model=model, 
                max_tokens=args.max_tokens, 
                max_output_tokens=args.max_output_tokens)
app = BasicSummarizer(llm, ...)
```

### 3.2 Streaming Not Exposed

**Why apps don't have `--stream` flag**:
- Apps are designed for batch processing
- They need complete output for post-processing
- Streaming would require architectural redesign
- No use case for real-time streaming in summarizer/extractor/judge

**Configuration System Comment** (manager.py, Line 96):
```python
@dataclass
class StreamingConfig:
    """Streaming behavior configuration."""
    cli_stream_default: bool = False  # Default streaming mode for CLI
    # Note: Server streaming is per-request, apps don't stream by design
```

---

## 4. CONFIGURATION SYSTEM INTEGRATION

### 4.1 Configuration Hierarchy

```
Default Hardcoded (in apps)
    ↓
Config File (~/.abstractcore/config/abstractcore.json)
    ↓
Explicit CLI Arguments (--provider, --model)
```

### 4.2 AppDefaults Configuration

**File**: `abstractcore/config/manager.py` (lines 37-46)

```python
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
```

### 4.3 StreamingConfig

**File**: `abstractcore/config/manager.py` (lines 93-96)

```python
@dataclass
class StreamingConfig:
    """Streaming behavior configuration."""
    cli_stream_default: bool = False  # Default streaming mode for CLI
    # Note: Server streaming is per-request, apps don't stream by design
```

**Methods** (manager.py, lines 420-445):

```python
def set_streaming_default(self, app_name: str, enabled: bool):
    """Set default streaming behavior for an app."""
    if app_name == "cli":
        self.config.streaming.cli_stream_default = enabled
    else:
        raise ValueError(f"Streaming only supported for CLI (not {app_name})")

def get_streaming_default(self, app_name: str) -> bool:
    """Get default streaming behavior for an app."""
    if app_name == "cli":
        return self.config.streaming.cli_stream_default
    else:
        return False  # Apps don't support streaming by design
```

### 4.4 App Configuration Resolution

All apps use `app_config_utils.py`:

```python
def get_app_defaults(app_name: str) -> tuple[str, str]:
    try:
        config_manager = get_config_manager()
        return config_manager.get_app_default(app_name)
    except Exception:
        hardcoded_defaults = {
            'summarizer': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'extractor': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'judge': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'cli': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
        }
        return hardcoded_defaults.get(app_name, ...)
```

---

## 5. STRUCTURED LOGGING INTEGRATION

### 5.1 Logging Architecture

**File**: `abstractcore/utils/structured_logging.py`

**Features**:
1. **Configuration-driven**: Reads from config system (line 29-72)
2. **Separate console/file levels**: Different verbosity per output
3. **Verbatim capture**: Optional complete prompt/response recording
4. **Structured output**: JSON for machine readability, text for human
5. **Generation logging**: Captures provider, model, tokens, latency (lines 351-403)
6. **Tool call logging**: Captures execution, timing, results (lines 405-436)

### 5.2 Integration with Apps

**Current Usage**:
- Apps don't currently use structured logging
- Could be integrated in processing modules
- Would provide audit trail of generations

**Example Integration** (hypothetical):
```python
from abstractcore.utils.structured_logging import get_logger

logger = get_logger(__name__)

# In app summarize/extract/evaluate methods:
logger.log_generation(
    provider=self.llm.provider_name,
    model=self.llm.model_name,
    prompt=prompt_text,
    response=response_text,
    tokens=token_usage,
    latency_ms=elapsed_ms,
    success=True
)
```

### 5.3 Configuration Sources

**File**: `abstractcore/config/manager.py` (lines 82-90)

```python
@dataclass
class LoggingConfig:
    console_level: str = "WARNING"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    file_level: str = "DEBUG"
    log_base_dir: str = "~/.abstractcore/logs"
    file_logging_enabled: bool = False
    verbatim_enabled: bool = True
    console_json: bool = False
    file_json: bool = True
```

---

## 6. CLI TOOL RELATIONSHIPS

### 6.1 Interactive CLI (`abstractcore/utils/cli.py`)

**Streaming Support**: ✅ **YES** (both flag and runtime toggle)

**Streaming Features**:
- `--stream` flag: Enable streaming mode on startup
- `/stream` command: Toggle at runtime
- Configuration default: `abstractcore --stream on/off`
- Status display: Shows current mode

**Configuration Integration** (lines 1374-1382):
```python
if not args.stream and config_manager:
    try:
        default_streaming = config_manager.get_streaming_default('cli')
        stream_mode = default_streaming
    except Exception:
        stream_mode = False
else:
    stream_mode = args.stream
```

### 6.2 HTTP Server API (`abstractcore/server/app.py`)

**Streaming Support**: ✅ **YES** (per-request parameter)

**Usage**: `POST /v1/chat/completions` with `"stream": true`

**Response Format**: Server-Sent Events (SSE) compatible with OpenAI API

### 6.3 Apps vs. CLI/Server

```
┌─────────────────┐
│   User/API      │
└────────┬────────┘
         │
    ┌────┴─────┐
    │           │
    v           v
┌─────────┐  ┌────────────┐
│   Apps  │  │ CLI/Server │
│ (batch) │  │ (realtime) │
└────┬────┘  └────┬───────┘
     │            │
     v            v
Streaming:     Streaming:
❌ NO          ✅ YES (configurable)
```

---

## 7. STREAMING CONFIGURATION EXAMPLES

### 7.1 CLI Configuration

**Commands**:
```bash
# Enable streaming by default
abstractcore --stream on

# Disable streaming by default
abstractcore --stream off

# Check current setting
abstractcore --status
```

**Config File** (`~/.abstractcore/config/abstractcore.json`):
```json
{
  "streaming": {
    "cli_stream_default": true
  }
}
```

### 7.2 App Configuration

**Current Status**:
```bash
# Set summarizer default
abstractcore --set-app-default summarizer openai gpt-4o-mini

# Check status
abstractcore --status
```

**Apps Don't Have Streaming**:
```bash
# These don't exist (and shouldn't):
# abstractcore.apps.summarizer --stream
# abstractcore.apps.extractor --stream
# abstractcore.apps.judge --stream
```

---

## 8. KEY INTEGRATION POINTS

### 8.1 Provider/LLM Factory

**File**: `abstractcore/core/factory.py`

**Function**: `create_llm(provider, model=..., stream=..., **kwargs)`

**Streaming Support**:
- Providers handle streaming natively
- Each provider's `generate()` method accepts `stream` parameter
- Returns `GenerateResponse` or `Iterator[GenerateResponse]` based on streaming

### 8.2 Session Layer

**File**: `abstractcore/core/session.py`

**Pass-through Architecture**:
```python
def generate(self, prompt, ..., stream=False, ...):
    response = self.provider.generate(
        prompt,
        stream=stream,  # Direct pass-through
        ...
    )
    return response
```

### 8.3 Streaming Processor

**File**: `abstractcore/providers/streaming.py`

**Purpose**:
- Incremental tool call detection
- Tag rewriting support
- Unified streaming strategy
- Format conversion

**Usage**: Internal to provider implementations for tool handling

---

## 9. STRUCTURED LOGGING FEATURES

### 9.1 Core Capabilities

**1. Generation Logging**:
- Provider, model, prompt length, response length
- Token usage (prompt/completion/total)
- Latency in milliseconds
- Success/error status

**2. Tool Call Logging**:
- Tool name, arguments
- Result length, success status
- Execution time
- Error details if failed

**3. Context Binding**:
- Session IDs, request IDs
- User information
- Custom metadata

### 9.2 Configuration Integration

**Three-level Configuration**:

1. **Config File** (`~/.abstractcore/config/abstractcore.json`):
   ```json
   {
     "logging": {
       "console_level": "WARNING",
       "file_level": "DEBUG",
       "log_base_dir": "~/.abstractcore/logs",
       "file_logging_enabled": true,
       "verbatim_enabled": true
     }
   }
   ```

2. **Programmatic** (`configure_logging()`):
   ```python
   from abstractcore.utils.structured_logging import configure_logging
   import logging
   
   configure_logging(
       console_level=logging.DEBUG,
       file_level=logging.DEBUG,
       log_dir="./logs",
       verbatim_enabled=True
   )
   ```

3. **Environment Variables** (via config manager):
   - Configuration manager reads from `~/.abstractcore/config/`

---

## 10. DESIGN RATIONALE: WHY APPS DON'T STREAM

### 10.1 Technical Reasons

| App | Why Not Streaming |
|-----|------------------|
| **Summarizer** | Needs complete text aggregation across chunks; post-processing (key points, compression ratio) |
| **Extractor** | Requires complete JSON-LD output; iterative refinement process; entity deduplication |
| **Judge** | Must calculate scores from complete evaluation; structured output object required |

### 10.2 Architectural Pattern

```
Streaming Use Cases ✅:
  - Interactive chat (real-time feedback)
  - API responses (live updates)
  - Monitoring/logging (immediate capture)

Batch Use Cases ❌:
  - Document processing (needs aggregation)
  - Knowledge extraction (needs structuring)
  - Quality assessment (needs completeness)
```

### 10.3 Configuration System Constraint

**Line 96, manager.py**:
```python
# Note: Server streaming is per-request, apps don't stream by design
```

This is intentional - apps are designed as **batch processors**, not real-time systems.

---

## 11. IMPLEMENTATION OPPORTUNITIES

### 11.1 Enhanced Logging in Apps

```python
# In BasicSummarizer.summarize()
from abstractcore.utils.structured_logging import get_logger

logger = get_logger(__name__)

# Log the complete generation
logger.log_generation(
    provider=llm.provider_name,
    model=llm.model_name,
    prompt=text,
    response=summary_text,
    tokens={"prompt": ..., "completion": ...},
    latency_ms=elapsed_ms,
    success=True
)
```

### 11.2 Configuration-Driven Behavior

**Current**: Hardcoded fallbacks  
**Potential**: Use centralized defaults everywhere

```python
# Current pattern (already done)
provider, model = get_app_defaults('summarizer')

# Could extend to other settings:
config_mgr = get_config_manager()
max_tokens = config_mgr.get_app_setting('summarizer', 'max_tokens', 32000)
chunk_size = config_mgr.get_app_setting('summarizer', 'chunk_size', 8000)
```

### 11.3 Streaming Architecture for Future Apps

If future apps need streaming:
1. Use `session.generate(..., stream=True)`
2. Buffer and process chunks incrementally
3. Emit structured output when complete
4. Integrate with streaming processor for tools

---

## 12. SUMMARY TABLE

| Component | Location | Streaming Support | Config Integration |
|-----------|----------|------------------|-------------------|
| **CLI** | `abstractcore/utils/cli.py` | ✅ Yes (configurable) | ✅ Full |
| **Server API** | `abstractcore/server/app.py` | ✅ Yes (per-request) | ✅ Full |
| **Summarizer** | `abstractcore/apps/summarizer.py` | ❌ No (batch) | ✅ Defaults only |
| **Extractor** | `abstractcore/apps/extractor.py` | ❌ No (batch) | ✅ Defaults only |
| **Judge** | `abstractcore/apps/judge.py` | ❌ No (batch) | ✅ Defaults only |
| **Config System** | `abstractcore/config/manager.py` | ✅ Infrastructure | N/A |
| **Logging** | `abstractcore/utils/structured_logging.py` | N/A (independent) | ✅ Full |
| **Streaming Processor** | `abstractcore/providers/streaming.py` | ✅ Internal | N/A |

---

## 13. FILES INVOLVED

### Core Application Files
- `/Users/albou/projects/abstractcore/abstractcore/apps/summarizer.py` - 429 lines
- `/Users/albou/projects/abstractcore/abstractcore/apps/extractor.py` - 607 lines
- `/Users/albou/projects/abstractcore/abstractcore/apps/judge.py` - 616 lines
- `/Users/albou/projects/abstractcore/abstractcore/apps/app_config_utils.py` - 19 lines

### Configuration System
- `/Users/albou/projects/abstractcore/abstractcore/config/manager.py` - Streaming/App config
- `/Users/albou/projects/abstractcore/abstractcore/config/__init__.py` - Exports

### CLI and Server
- `/Users/albou/projects/abstractcore/abstractcore/utils/cli.py` - Interactive CLI with streaming
- `/Users/albou/projects/abstractcore/abstractcore/cli/main.py` - Configuration CLI
- `/Users/albou/projects/abstractcore/abstractcore/server/app.py` - HTTP API with streaming

### Logging System
- `/Users/albou/projects/abstractcore/abstractcore/utils/structured_logging.py` - Structured logging

### Processing Modules
- `/Users/albou/projects/abstractcore/abstractcore/processing/basic_summarizer.py`
- `/Users/albou/projects/abstractcore/abstractcore/processing/basic_extractor.py`
- `/Users/albou/projects/abstractcore/abstractcore/processing/basic_judge.py`

### Provider/Core Infrastructure
- `/Users/albou/projects/abstractcore/abstractcore/core/interface.py` - LLM interface
- `/Users/albou/projects/abstractcore/abstractcore/core/session.py` - Session layer
- `/Users/albou/projects/abstractcore/abstractcore/providers/streaming.py` - Streaming processor
- `/Users/albou/projects/abstractcore/abstractcore/providers/base.py` - Base provider

---

## 14. INVESTIGATION CONCLUSION

### What We Learned

1. **Streaming is intentionally exclusive**: Only CLI/Server use it; apps deliberately don't
2. **Configuration system is well-designed**: Supports both streaming and app defaults
3. **Structured logging is independent**: Separate from streaming, can be integrated anywhere
4. **Apps follow unified pattern**: All three use same config resolution pattern
5. **Integration is clean**: Apps, CLI, and Server all respect configuration hierarchy

### Current State

- ✅ Streaming works for CLI and Server
- ✅ Apps properly use configuration defaults
- ✅ Structured logging infrastructure ready
- ⚠️ Logging not yet integrated into apps (opportunity)
- ✅ Configuration system ready to expand

### No Issues Found

The current architecture is sound:
- Clear separation of concerns
- Proper configuration hierarchy
- Explicit design decisions documented
- Ready for enhancement or expansion

---

**End of Investigation Report**
