# Architecture Detection Module

## Overview

JSON-driven architecture detection and capability lookup determining **HOW** to communicate (formats, tokens) and **WHAT** models can do (tools, vision, limits).

**Files**: `enums.py` (types), `detection.py` (detection + lookup)
**Assets**: `architecture_formats.json` (message formats), `model_capabilities.json` (per-model capabilities)

---

## Components

### 1. Enums (`enums.py`)

**ToolCallFormat**: NATIVE (API-based), JSON, XML, SPECIAL_TOKEN (`<|tool_call|>`), PYTHONIC, NONE
**ModelType**: CHAT, INSTRUCT, BASE, CODE, VISION, EMBEDDING, UNKNOWN (detected from name patterns)
**ArchitectureFamily** (30+): GPT, Claude, LLaMA (2/3/3.1/3.2/3.3/4), Qwen (2/2.5/2-VL/3/3-MoE/3-Next/3-VL), Mistral/Mixtral/MistralLarge/Codestral, Gemma (1/2/3/3n/Code/Pali), GLM-4/GLM-4-MoE, Phi, DeepSeek, Granite, Seed-OSS, Yi, GENERIC

**Conversion**: `ArchitectureFamily.from_string("qwen2.5")` → `QWEN2_5`

---

### 2. Detection Functions (`detection.py`)

**Core**:
- `detect_architecture(name)` - Pattern matching from JSON (case-insensitive, **most-specific/longest match**, cached) → "llama3_1", "qwen3", "gemma2", "generic"
- `detect_model_type(name)` - Name-based detection → "chat", "instruct", "code", "vision", "base"

**Configuration**:
- `get_architecture_format(arch)` - Returns message/tool formats (prefixes, suffixes, tool_format)
- `get_model_capabilities(name)` - Returns capabilities with fallbacks (Cache → Aliases → Exact → Partial → Defaults)

→ Full schema in [assets/model_capabilities.json](../assets/README.md)

**Helpers**:
- `resolve_model_alias(name, models)` - Converts aliases ("--" → "/"), checks alias field, cached
- `format_messages(messages, arch)` - Applies architecture-specific formatting (prefixes/suffixes per role)

---

### 3. Response Post-processing (`response_postprocessing.py`)

These helpers normalize raw model output into a consistent shape across providers using the same JSON assets:

- **Output wrappers**: Strip leading/trailing wrapper tokens (e.g. GLM `<|begin_of_box|>…<|end_of_box|>`)
- **Harmony transcripts (GPT-OSS)**: Extract final text and capture analysis as `reasoning`
- **Thinking tags**: Extract `<think>...</think>` blocks (when configured) into `reasoning` and remove them from `content`
  - Some models may emit only the closing `</think>` tag (opening provided by the chat template); the extractor handles this variant.

Providers call this from `BaseProvider` so the behavior is consistent for OpenAI-compatible servers, Ollama, and local runtimes.

---

### 3. Convenience Functions

| Function | Returns | Use Case |
|----------|---------|----------|
| `supports_tools(name)` | `bool` | Check tool/function calling support |
| `supports_vision(name)` | `bool` | Check image input support |
| `supports_audio(name)` | `bool` | Check audio input support |
| `supports_embeddings(name)` | `bool` | Check embedding generation support |
| `get_context_limits(name)` | `Dict[str, int]` | Get `max_tokens` and `max_output_tokens` |
| `is_instruct_model(name)` | `bool` | Check if instruction-tuned (contains "instruct", "chat", "assistant", "turbo") |

---

### 4. Vision-Specific Functions

**get_vision_capabilities(name)**: Returns vision metadata (tokenization method, patch size, max tokens, resolutions, etc.). Falls back to generic_vision_model if not found.

**get_glyph_compression_capabilities(name)**: Returns Glyph compression recommendations (compatible, pages/image, DPI based on max_image_tokens)

**check_vision_model_compatibility(name, provider)**: Comprehensive check returning compatible, warnings, recommendations, all capabilities

---

## Usage Patterns

| Pattern | Key Functions | Example |
|---------|---------------|---------|
| **Architecture Detection** | `detect_architecture()`, `get_architecture_format()` | `arch = detect_architecture("qwen3-4b")`<br>`fmt = get_architecture_format(arch)` |
| **Capability Checking** | `supports_tools()`, `supports_vision()`, `get_context_limits()` | `if supports_tools(model): use_tools()` |
| **Message Formatting** | `format_messages(messages, arch)` | Apply arch-specific prefixes/suffixes |
| **Vision Validation** | `check_vision_model_compatibility()`, `get_vision_capabilities()` | Comprehensive checks + warnings/recommendations |
| **Provider Integration** | Detect arch → Get format + caps → Format messages → Check tools | Cache at init, use in generate() |
| **Unknown Models** | `get_model_capabilities()` + check `architecture == "generic"` | Falls back to defaults with warning |

---

## Assets & Caching

**Assets**: `architecture_formats.json` (message/tool formats + patterns), `model_capabilities.json` (per-model specs + defaults)
**Caching**: JSON loaded once, detection results cached, aliases cached → Near-instant repeated calls
**Invalidation**: Not recommended in production; reset caches + call `_load_json_assets()` if needed

---

## Best Practices

1. Use convenience functions (`supports_tools()`) instead of manual capability checks
2. Handle unknown models gracefully (check `architecture == "generic"`)
3. Cache arch results at provider init (`detect_architecture()`, `get_model_capabilities()`)
4. Use type enums for type safety (`ToolCallFormat.NATIVE.value`)
5. Validate vision models before processing (`check_vision_model_compatibility()`)
6. Use vision capabilities for token estimation (get tokenization method + relevant params)

---

## Common Pitfalls

| Pitfall | Wrong | Right |
|---------|-------|-------|
| **Family assumptions** | Assume all LLaMA 3 have tools | Check specific model capabilities |
| **No fallback handling** | `arch_format["tool_prefix"]` → KeyError | Use `.get("tool_prefix", "")` |
| **HF cache format** | Ignore "--" in name | Module auto-converts "--" → "/" |
| **Hardcoded detection** | `if "qwen3" in name.lower()` | Use `detect_architecture(name)` |
| **Skip vision check** | Assume vision support | Call `supports_vision(model)` first |
| **Mix arch/caps** | Use arch to assume capabilities | Use capabilities functions |

---

## Testing Strategy

**Unit**: Architecture detection (LLaMA/Qwen variants), model type detection, unknown model fallback, capability checks
**Integration**: Provider integration with real models, message formatting, HuggingFace cache format
**Vision**: Vision compatibility checks, Glyph compression recommendations, non-vision model handling
**Performance**: Caching validation (second call faster), lookup speed (<100ms)

---

## Summary

The **architectures** module is the intelligence layer that enables AbstractCore to communicate with any LLM family without hardcoding patterns or capabilities. It provides:

- **30+ architecture families** with pattern-based detection
- **JSON-driven configuration** for easy extension
- **Comprehensive capability lookup** with intelligent fallbacks
- **Vision model support** with detailed token estimation capabilities
- **Multi-level caching** for performance
- **Type-safe enums** for tool formats, model types, and architectures
- **Convenience functions** for common capability checks

By centralizing architecture detection and capability lookup, this module ensures consistent behavior across all providers and simplifies adding support for new model families.

## Related Modules

**Direct dependencies**:
- [`assets/`](../assets/README.md) - Model capabilities database (model_capabilities.json)
- [`exceptions/`](../exceptions/README.md) - Error handling for unknown models

**Used by**:
- [`providers/`](../providers/README.md) - Model validation and capability checking
- [`media/`](../media/README.md) - Vision support and format detection
- [`compression/`](../compression/README.md) - Provider optimization profiles
- [`structured/`](../structured/README.md) - Structured output capability detection
- [`tools/`](../tools/README.md) - Tool support detection
- [`config/`](../config/README.md) - Default model selection

**Related systems**:
- [`utils/`](../utils/README.md) - Token estimation utilities
