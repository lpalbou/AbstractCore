# AbstractCore Utils Module

## Purpose

The `utils` module provides cross-cutting utilities and foundational services used throughout AbstractCore. These components handle logging, token estimation, message preprocessing, vision-language model calculations, runtime patches, and CLI interfaces. The module serves as the foundation for operational concerns that span multiple architectural layers.

## Architecture Position

**Layer**: Foundation / Infrastructure

**Dependencies**:
- Python standard library (logging, re, json, pathlib, typing)
- Optional: structlog, colorama, tiktoken, PIL (Pillow)
- Internal: `abstractcore.config`, `abstractcore.architectures.detection`

**Used By**:
- **Core Layer**: Factory, session management
- **Providers**: Token calculations, logging
- **Processing**: Token estimation, message formatting
- **Media**: Token calculations for vision models
- **Tools**: Message preprocessing
- **CLI**: Interactive interface

**Design Philosophy**: Lightweight, fail-safe, graceful degradation when optional dependencies are missing.

---

## Components

| File | Purpose | Key Features |
|------|---------|--------------|
| `version.py` | Version constants | Single source of truth for `__version__` |
| `structured_logging.py` | Structured logging | JSON support, verbatim capture, dual levels |
| `token_utils.py` | Token counting | Precise (tiktoken) + fast (heuristic) methods |
| `message_preprocessor.py` | Message parsing | `@filename` syntax for attachments |
| `vlm_token_calculator.py` | Vision tokens | Research-based formulas per provider/model |
| `self_fixes.py` | JSON repair | 4-level repair strategies |
| `cli.py` | Interactive REPL | Chat, tools, file attachments, analytics |

---

## Detailed Components

### 1. version.py
**Usage**: `from abstractcore.utils.version import __version__`
**Note**: Manually sync with `pyproject.toml` during releases

---

### 2. structured_logging.py

**Features**: Dual-level logging (console/file), JSON output, verbatim capture (.jsonl), context binding, colored console

**Quick Start**:
```python
from abstractcore.utils.structured_logging import get_logger, configure_logging
import logging

# Configure once at startup
configure_logging(console_level=logging.WARNING, file_level=logging.DEBUG,
                  log_dir="~/.abstractcore/logs", verbatim_enabled=True)

# Use throughout app
logger = get_logger(__name__).bind(request_id="req_123")
logger.info("Processing started", user_id="123")
logger.log_generation(provider="openai", model="gpt-4o", prompt="...", response="...")
```

**Verbatim Format**: JSON Lines with timestamp, provider, model, full prompt/response, metadata

---

### 3. token_utils.py

**Methods**: AUTO (tiktoken → fast), PRECISE (tiktoken only), FAST (content-aware heuristics)
**Content Types**: Natural (4.0 chars/token), Code (3.5), JSON (3.0), XML (3.2), Markdown (3.8), Mixed (3.7)
**Model Families**: GPT (1.0x), Claude (1.05x), Gemini (0.95x), LLaMA (1.1x), Qwen (1.15x), Unknown (1.2x)

**Quick Start**:
```python
from abstractcore.utils.token_utils import count_tokens, estimate_tokens, count_tokens_precise

tokens = count_tokens(text, model="gpt-4o")              # Auto (tiktoken or fast)
est = estimate_tokens(text, model="claude-3-5-haiku")    # Fast heuristic
precise = count_tokens_precise(text, model="gpt-4o")     # Exact (tiktoken)
batch = TokenUtils.count_tokens_batch(texts, model)      # Optimized batch
```

---

### 4. message_preprocessor.py

**Pattern**: `@filename.ext` (requires extension, no spaces)
**Functions**: `parse_files()`, `has_files()`, `extract_file_paths()`, `get_file_count()`

**Quick Start**:
```python
from abstractcore.utils.message_preprocessor import parse_files, has_files

user_input = "Analyze @screenshot.png and @data.csv"
clean_text, files = parse_files(user_input, verbose=True)
# clean_text: "Analyze  and"
# files: ["screenshot.png", "data.csv"]

if has_files(user_input):
    print("User attached files")
```

**Behavior**: Extracts files, validates existence (optional), cleans text, uses default prompt if empty

---

### 5. vlm_token_calculator.py

**Providers**: OpenAI (tile-based), Anthropic (pixel-based), Google (grid-based)
**Models**: Qwen-VL, LLaMA Vision, Gemma3, GLM-4 (patch-based)
**Features**: Single/batch calculation, Glyph compression analysis, efficiency ratings

**Quick Start**:
```python
from abstractcore.utils.vlm_token_calculator import VLMTokenCalculator, calculate_image_tokens

calculator = VLMTokenCalculator()

# Single image
result = calculator.calculate_tokens_for_image(
    image_path=Path("photo.jpg"), provider="openai", model="gpt-4o", detail_level="high")
# Returns: {'tokens': 765, 'method': 'openai_tile_based', 'tiles': '2x2', ...}

# Batch
batch = calculator.calculate_tokens_for_images(images, provider="anthropic", model="claude-3-5-sonnet")

# Glyph compression
analysis = calculate_glyph_compression_ratio(original_tokens=5000, image_paths=[...], ...)
# Returns: {'compression_ratio': 3.27, 'efficiency_rating': 'very_good', ...}

# Quick estimate
tokens = calculate_image_tokens(Path("img.png"), provider="google", model="gemini-1.5-pro")
```

**Efficiency**: Excellent (>10x), Very Good (>4x), Good (>2x), Marginal (>1x), Poor (≤1x)

---

### 6. self_fixes.py

**Strategies**: (1) Extract from text, (2) Fix formatting (trailing commas, quotes), (3) Repair truncated, (4) Create minimal JSON-LD

**Quick Start**:
```python
from abstractcore.utils.self_fixes import fix_json
import json

malformed = 'Here is {"name": "Alice", "age": 30,} done!'
fixed = fix_json(malformed)  # Returns: '{"name": "Alice", "age": 30}'
data = json.loads(fixed)

# Safe parsing
def safe_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = fix_json(text)
        return json.loads(fixed) if fixed else None
```

---

### 7. cli.py

**Features**: Multi-provider chat, tool execution, `@filename` attachments, session save/load, analytics, command history

**Usage**:
```bash
python -m abstractcore.utils.cli --provider ollama --model qwen3-coder:30b
python -m abstractcore.utils.cli --prompt "Question" --provider openai --model gpt-4o-mini
```

**Commands**: `/session`, `/cache`, `/save` (alias), `/load` (alias), `/facts`, `/judge`, `/intent`, `/history`, `/compact`, `/model`, `/system`, `/stream`, `/debug`, `/status`
**Tools**: `list_files()`, `search_files()`, `read_file()`, `write_file()`, `execute_command()`
**Attachments**: `"Analyze @screenshot.png"` (vision models), `"Read @file.txt"` (text processing)

**Limitations**: Simple demonstrations only; no ReAct patterns, complex chains, or adaptive reasoning. Build custom solutions for production.

---

## Usage Patterns

| Pattern | Key Code | Use Case |
|---------|----------|----------|
| **Production Logging** | `configure_logging(console_level=WARNING, file_level=DEBUG)`<br>`logger = get_logger(__name__).bind(request_id="...")` | Startup config + context binding |
| **Token Budget** | `TokenUtils.count_tokens(text, model, method=PRECISE)`<br>`budget.check_budget(text)` | Track usage against limit |
| **File Attachments** | `clean, files = parse_files(user_input, verbose=True)`<br>`is_vision_model(model)` | Parse @syntax + capability check |
| **Vision Tokens** | `VLMTokenCalculator().calculate_tokens_for_images(...)`<br>`result['total_tokens']` | Estimate image token cost |
| **JSON Repair** | `fixed = fix_json(text)`<br>`data = json.loads(fixed)` | Safe LLM output parsing |

---

## Integration Points

| Layer | Uses | For |
|-------|------|-----|
| **Core** | `count_tokens()`, `get_logger()` | Context management, event tracking |
| **Providers** | `TokenUtils`, `get_logger()` | Input validation, API logging |
| **Media** | `VLMTokenCalculator` | Vision token estimation |
| **Processing** | `parse_files()` | Input parsing |
| **Tools** | `get_logger()` | Execution tracking |

---

## Best Practices

### DO
1. Configure logging early; use structured logging with context binding
2. Use precise tokens for billing; fast for UI updates
3. Validate file attachments before processing
4. Include VLM tokens in vision budgeting
5. Attempt JSON repair before failing
6. Use batch processing for multiple token counts
7. Redact sensitive data in logs

### DON'T
1. Hardcode versions; import from `utils.version`
2. Use precise counting in loops; batch instead
3. Parse files without validation in production
4. Assume tiktoken/structlog availability; use graceful APIs
5. Ignore image token costs; always include VLM calculations
6. Log sensitive data unredacted
7. Use debug mode in production
8. Assume LLM JSON is valid; use repair strategies

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Token inconsistency** | Different methods for budget vs usage | Use same `method` parameter consistently |
| **Missing dependencies** | Import tiktoken/structlog directly | Use graceful wrapper APIs |
| **Relative paths** | Break when cwd changes | Use `Path().resolve()` for absolute |
| **VLM underestimation** | Ignore image tokens | Add text + image tokens |
| **Late logging config** | Configure after first log | Call `configure_logging()` at startup |
| **Repair over-reliance** | Assume `fix_json()` always works | Check if result is `None` |
| **Capability assumptions** | Assume vision support | Call `is_vision_model()` first |

---

## Testing Strategy

**Unit Tests**: Token counting (auto/precise/fast/batch, content detection), Message parsing (@syntax, validation), JSON repair (extract, fix, truncate)

**Integration Tests**: Token counting vs actual provider usage (<10% variance), File attachment flow (parse → LLM with media)

**E2E Tests**: CLI single-prompt mode, CLI tool execution with file operations

---

## Public API

### Recommended Imports

```python
# Version
from abstractcore.utils.version import __version__

# Logging
from abstractcore.utils.structured_logging import (
    get_logger,
    configure_logging,
    capture_session,
    suppress_stdout_stderr
)

# Token Utilities
from abstractcore.utils.token_utils import (
    TokenUtils,
    count_tokens,
    estimate_tokens,
    count_tokens_precise,
    TokenCountMethod,
    ContentType
)

# Message Preprocessing
from abstractcore.utils.message_preprocessor import (
    MessagePreprocessor,
    parse_files,
    has_files
)

# VLM Token Calculator
from abstractcore.utils.vlm_token_calculator import (
    VLMTokenCalculator,
    calculate_image_tokens,
    calculate_glyph_compression_ratio
)

# Self Fixes
from abstractcore.utils.self_fixes import fix_json

# CLI (not typically imported, but available)
# python -m abstractcore.utils.cli
```

### Core Classes

1. **TokenUtils**: Multi-strategy token counting
2. **VLMTokenCalculator**: Vision token estimation
3. **MessagePreprocessor**: File attachment parsing
4. **StructuredLogger**: Enhanced logging with context
5. **SimpleCLI**: Interactive CLI interface

### Convenience Functions

1. **count_tokens()**: Universal token counting
2. **estimate_tokens()**: Fast estimation
3. **count_tokens_precise()**: Accurate counting
4. **parse_files()**: Parse @filename syntax
5. **has_files()**: Check for attachments
6. **calculate_image_tokens()**: Quick image token count
7. **fix_json()**: Repair malformed JSON
8. **get_logger()**: Get structured logger
9. **configure_logging()**: Set up logging system

---

## Summary

The **utils module** provides essential infrastructure for AbstractCore:

- **Version management** for reliable versioning
- **Structured logging** with JSON, verbatim capture, and context binding
- **Token counting** with precise/fast methods and content-aware heuristics
- **Message preprocessing** for @filename attachments
- **VLM token calculation** with research-based formulas
- **Runtime fixes** for malformed LLM outputs
- **CLI interface** for interactive demonstrations

**Design Principles**:
- Lightweight and fail-safe
- Graceful degradation without optional dependencies
- Consistent interfaces across utilities
- Production-ready with comprehensive error handling

**Use Cases**:
- Token budget management
- Production logging and monitoring
- File attachment processing
- Vision token estimation
- Malformed output recovery
- Interactive testing and demos

For detailed implementation examples, see the code samples throughout this documentation.

## Related Modules

**Used by (utility consumers)**:
- [`providers/`](../providers/README.md) - Token estimation, logging, validation
- [`media/`](../media/README.md) - VLM token calculation, image processing
- [`compression/`](../compression/README.md) - Token estimation, analytics, logging
- [`structured/`](../structured/README.md) - Validation utilities, logging
- [`tools/`](../tools/README.md) - Logging, validation
- [`processing/`](../processing/README.md) - Web utilities, logging
- [`config/`](../config/README.md) - Path utilities, validation
- [`server/`](../server/README.md) - Logging, metrics

**Dependencies**:
- [`architectures/`](../architectures/README.md) - Model capabilities for token calculation
- [`events/`](../events/README.md) - Logging integration with events
- [`exceptions/`](../exceptions/README.md) - Validation error handling
