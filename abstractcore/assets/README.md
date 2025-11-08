# Assets Module

## Purpose

This module contains static resources that are referenced throughout AbstractCore: model capability databases, architecture format definitions, session schemas, OCR fonts for glyph compression, and semantic ontology guides. These assets enable architecture detection, capability discovery, session serialization, and visual text compression.

## Architecture Position

- **Layer**: Foundation (Layer 0)
- **Dependencies**: None (static files only)
- **Used By**:
  - `architectures/` - Uses architecture_formats.json
  - `providers/` - Uses model_capabilities.json
  - `compression/` - Uses OCR fonts (OCRA.ttf, OCRB.ttf, OCRBL.ttf)
  - `core/session.py` - Uses session_schema.json
  - Semantic tools - Uses semantic_models.md

## File Structure

```
assets/
├── architecture_formats.json    (16.6 KB) - Message formatting for 30+ LLM architectures
├── model_capabilities.json      (67.1 KB) - Capability database for 150+ models
├── session_schema.json          (10.6 KB) - JSON schema for session serialization
├── semantic_models.md           (36.8 KB) - Ontology selection guide (Dublin Core, Schema.org)
├── OCRA.ttf                     (15.9 KB) - OCR-A monospace font
├── OCRB.ttf                     (19.4 KB) - OCR-B monospace font
└── OCRBL.ttf                    (21.7 KB) - OCR-B Light monospace font
```

## Assets Overview

### 1. architecture_formats.json

**Purpose**: Define message formatting conventions and special tokens for different LLM architectures.

**Size**: 16,622 bytes (~500 lines)

**Structure**:
```json
{
  "architectures": {
    "llama3": {
      "description": "Meta's LLaMA 3 architecture (April 2024)",
      "message_format": "llama3_header",
      "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
      "system_suffix": "<|eot_id|>",
      "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
      "user_suffix": "<|eot_id|>",
      "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
      "assistant_suffix": "<|eot_id|>",
      "tool_format": "prompted",
      "patterns": ["llama-3.0", "llama3.0", "llama-3-8b", "llama3-70b"]
    },
    ...
  }
}
```

**Supported Architectures** (30+):
- **LLaMA Family**: llama2, llama3, llama3.1, llama3.2, llama3.3, llama4
- **Qwen Family**: qwen2, qwen2.5, qwen3, qwen3-next, qwen3-vl, qwen4-coder
- **Gemma Family**: gemma, gemma2
- **Mistral Family**: mistral, mixtral, ministral
- **DeepSeek**: deepseek-r1, deepseek-v3
- **GLM**: glm-4
- **Phi**: phi3, phi4
- **Others**: command-r, command-r-plus, aya, granite

**Fields**:
- `description` - Human-readable description with release date
- `message_format` - Format type (llama3_header, im_start_end, inst, etc.)
- `system_prefix/suffix` - Special tokens for system messages
- `user_prefix/suffix` - Special tokens for user messages
- `assistant_prefix/suffix` - Special tokens for assistant messages
- `tool_format` - Tool support type (native, prompted, special_token, pythonic)
- `tool_prefix` - Special token for tool calls (optional)
- `patterns` - Model name patterns for detection (regex-style)

**Usage**:
```python
from abstractcore.architectures import detect_architecture

arch = detect_architecture("llama-3-8b-instruct")
# Returns: ModelArchitecture.LLAMA3

# Get format details
with open("assets/architecture_formats.json") as f:
    formats = json.load(f)
    llama3_format = formats["architectures"]["llama3"]
    print(llama3_format["system_prefix"])  # <|start_header_id|>system<|end_header_id|>
```

**When to Update**:
- New model architecture is released
- Existing architecture format is corrected
- New tool calling format is discovered

### 2. model_capabilities.json

**Purpose**: Comprehensive database of model capabilities for all supported providers.

**Size**: 67,073 bytes (~2,000 lines)

**Structure**:
```json
{
  "models": {
    "gpt-4o": {
      "max_output_tokens": 16384,
      "tool_support": "native",
      "structured_output": "native",
      "parallel_tools": true,
      "max_tools": -1,
      "vision_support": true,
      "audio_support": true,
      "video_support": true,
      "image_resolutions": ["variable"],
      "image_tokenization_method": "tile_based",
      "base_image_tokens": 85,
      "tokens_per_tile": 170,
      "tile_size": "512x512",
      "max_image_dimension": 2048,
      "short_side_resize_target": 768,
      "detail_levels": ["low", "high", "auto"],
      "low_detail_tokens": 85,
      "notes": "Multimodal omni model",
      "source": "OpenAI official docs 2025",
      "canonical_name": "gpt-4o",
      "aliases": [],
      "max_tokens": 128000
    },
    ...
  }
}
```

**Coverage** (150+ models):
- **OpenAI**: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, o1, o3-mini
- **Anthropic**: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Google**: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
- **Ollama**: 50+ models (llama, qwen, mistral, gemma, etc.)
- **HuggingFace**: transformers and GGUF models
- **MLX**: Apple Silicon optimized models

**Fields**:

#### Core Capabilities
- `max_tokens` - Context window size (input tokens)
- `max_output_tokens` - Maximum output tokens
- `tool_support` - Tool calling support: "native", "prompted", "none"
- `structured_output` - Structured output support: "native", "prompted", "none"
- `parallel_tools` - Can execute multiple tools in parallel (boolean)
- `max_tools` - Maximum tools per call (-1 = unlimited)

#### Multimodal Capabilities
- `vision_support` - Image input support (boolean)
- `audio_support` - Audio input support (boolean)
- `video_support` - Video input support (boolean)

#### Vision-Specific
- `image_resolutions` - Supported resolutions (e.g., ["variable"], ["low", "high"])
- `image_tokenization_method` - How images are tokenized ("tile_based", "patch_based", etc.)
- `base_image_tokens` - Base token cost per image
- `tokens_per_tile` - Tokens per image tile
- `tile_size` - Tile dimensions (e.g., "512x512")
- `max_image_dimension` - Maximum image dimension
- `short_side_resize_target` - Target for short side resizing
- `detail_levels` - Available detail levels (["low", "high", "auto"])
- `low_detail_tokens` - Token cost for low detail

#### Metadata
- `notes` - Additional notes about the model
- `source` - Where capability info was obtained
- `canonical_name` - Official model name
- `aliases` - Alternative names for the model

**Usage**:
```python
from abstractcore.utils import load_model_capabilities

capabilities = load_model_capabilities()
gpt4o = capabilities["gpt-4o"]

if gpt4o["vision_support"]:
    print(f"Vision enabled. Base tokens: {gpt4o['base_image_tokens']}")

if gpt4o["structured_output"] == "native":
    print("Native structured output supported")
```

**When to Update**:
- New model is released
- Model capabilities change (e.g., vision support added)
- Token limits or pricing updates
- New features like audio/video support

### 3. session_schema.json

**Purpose**: JSON Schema (Draft 07) for validating and serializing AbstractCore conversation sessions.

**Size**: 10,570 bytes (~308 lines)

**Schema Version**: `session-archive/v1`

**Top-Level Structure**:
```json
{
  "schema_version": "session-archive/v1",
  "session": { ... },
  "messages": [ ... ],
  "usage_stats": { ... },
  "events": [ ... ]
}
```

**Session Object**:
```json
{
  "id": "uuid",
  "created_at": "2025-01-01T00:00:00Z",
  "provider": "openai",
  "model": "gpt-4o",
  "model_params": {
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "system_prompt": "You are a helpful assistant",
  "tool_registry": [...],
  "settings": {...},
  "summary": {...},
  "assessment": {...},
  "facts": {...}
}
```

**Key Components**:

#### Session Metadata
- `id` - Unique session identifier (required)
- `created_at` - ISO timestamp (required)
- `provider` - LLM provider name (optional)
- `model` - Model name (optional)
- `model_params` - Generation parameters (optional)
- `system_prompt` - System prompt text (optional)

#### Tool Registry
Declarative tool definitions (not implementations):
```json
{
  "tool_registry": [
    {
      "name": "search_web",
      "description": "Search the web",
      "json_schema": { ... }
    }
  ]
}
```

#### Session Summary
Generated conversation summary:
```json
{
  "summary": {
    "created_at": "2025-01-01T12:00:00Z",
    "preserve_recent": 5,
    "focus": "technical decisions",
    "text": "Summary text...",
    "metrics": {
      "tokens_before": 10000,
      "tokens_after": 2000,
      "compression_ratio": 5.0,
      "gen_time": 1.5
    }
  }
}
```

#### Session Assessment
Quality evaluation:
```json
{
  "assessment": {
    "created_at": "2025-01-01T12:00:00Z",
    "criteria": {
      "clarity": true,
      "coherence": true,
      "relevance": true,
      "completeness": true,
      "actionability": true
    },
    "overall_score": 4.5,
    "judge_summary": "High quality conversation",
    "strengths": ["Clear goals", "Actionable steps"],
    "actionable_feedback": ["Consider adding examples"],
    "reasoning": "Detailed reasoning..."
  }
}
```

#### Extracted Facts
Knowledge graph triples:
```json
{
  "facts": {
    "extracted_at": "2025-01-01T12:00:00Z",
    "simple_triples": [
      ["Python", "is", "programming language"],
      ["AbstractCore", "supports", "OpenAI"]
    ],
    "jsonld": { ... },
    "statistics": {
      "entities_count": 15,
      "relationships_count": 23,
      "extraction_time_ms": 450
    }
  }
}
```

**Messages Array**:
```json
{
  "messages": [
    {
      "id": "msg-uuid",
      "role": "user",
      "timestamp": "2025-01-01T00:00:00Z",
      "content": "Hello",
      "metadata": {
        "name": "user",
        "location": "US"
      }
    },
    {
      "id": "msg-uuid-2",
      "role": "assistant",
      "timestamp": "2025-01-01T00:00:01Z",
      "content": "Hello! How can I help?",
      "metadata": {
        "requested_tool_calls": [
          {
            "call_id": "call-123",
            "name": "search_web",
            "arguments": {"query": "Python"}
          }
        ]
      }
    },
    {
      "id": "msg-uuid-3",
      "role": "tool",
      "timestamp": "2025-01-01T00:00:02Z",
      "content": "Search results...",
      "metadata": {
        "call_id": "call-123",
        "status": "ok",
        "duration_ms": 342.1
      }
    }
  ]
}
```

**Usage Stats**:
```json
{
  "usage_stats": {
    "total_tokens": 15000,
    "by_message_id": {
      "msg-uuid": 100,
      "msg-uuid-2": 50
    },
    "cost_estimate": {
      "currency": "USD",
      "amount": 0.045
    }
  }
}
```

**Events Array**:
```json
{
  "events": [
    {
      "type": "session_created",
      "at": "2025-01-01T00:00:00Z",
      "data": {}
    },
    {
      "type": "compaction_completed",
      "at": "2025-01-01T12:00:00Z",
      "data": {
        "original_messages": 100,
        "compacted_messages": 20,
        "tokens_saved": 8000
      }
    }
  ]
}
```

**Usage**:
```python
import json
from jsonschema import validate

# Load schema
with open("assets/session_schema.json") as f:
    schema = json.load(f)

# Validate session data
session_data = session.to_dict()
validate(instance=session_data, schema=schema)  # Raises if invalid

# Export session
with open("session_export.json", "w") as f:
    json.dump(session_data, f, indent=2)
```

**When to Update**:
- New session features are added (e.g., multi-agent support)
- New message metadata fields
- New event types
- Schema validation requirements change

### 4. semantic_models.md

**Purpose**: Guide for semantic experts on ontology selection and implementation for knowledge representation.

**Size**: 36,835 bytes (~800 lines)

**Content Sections**:
1. **Selected Ontologies** (4 main ontologies)
2. **Entity Type Mapping** (document, conceptual, agent, content entities)
3. **Property Mapping** (document properties, structural relationships)
4. **Relationship Patterns** (citation, semantic, provenance relationships)
5. **JSON-LD Examples** (full implementation examples)

**Core Ontologies**:

| Ontology | Namespace | Adoption | Use Case |
|----------|-----------|----------|----------|
| **Dublin Core Terms** | `dcterms:` | 60-70% | Document metadata, structure |
| **Schema.org** | `schema:` | 35-45% | General entities, relationships |
| **SKOS** | `skos:` | 15-20% | Concept definition, semantics |
| **CiTO** | `cito:` | 15-20% | Scholarly/evidential relationships |

**Standard JSON-LD Context**:
```json
{
  "@context": {
    "dcterms": "http://purl.org/dc/terms/",
    "schema": "https://schema.org/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "cito": "http://purl.org/spar/cito/",
    "ex": "http://example.org/"
  }
}
```

**Entity Mapping Examples**:
- Documents → `dcterms:Text`
- Concepts → `skos:Concept`
- Persons → `schema:Person`
- Organizations → `schema:Organization`
- Claims → `cito:Claim`

**Relationship Examples**:
- `dcterms:isPartOf` - Structural hierarchy
- `dcterms:references` - Citation
- `skos:broader` - Concept hierarchy
- `cito:cites` - Scholarly citation
- `cito:supports` - Evidence support

**Usage**:
```python
# Used by semantic extraction tools
from abstractcore.tools import extract_semantic_triples

triples = extract_semantic_triples(conversation)
# Returns: [(subject, predicate, object), ...]
# Using ontology terms from semantic_models.md

jsonld = convert_to_jsonld(triples)
# Validates against Dublin Core, Schema.org, SKOS, CiTO
```

**When to Update**:
- New ontology adoption trends
- Expanded use cases (e.g., code ontologies, dataset ontologies)
- Updated ontology specifications

### 5. OCR Fonts (OCRA.ttf, OCRB.ttf, OCRBL.ttf)

**Purpose**: Monospace fonts optimized for optical character recognition, used by glyph compression system.

**Sizes**:
- OCRA.ttf: 15,896 bytes
- OCRB.ttf: 19,444 bytes
- OCRBL.ttf (OCR-B Light): 21,728 bytes

**Characteristics**:
- **Monospace**: All characters have identical width
- **High legibility**: Designed for machine and human readability
- **OCR-optimized**: Character shapes minimize recognition errors
- **Compact**: Small file sizes for fast loading

**Why OCR Fonts for Glyph Compression?**
1. **Consistent spacing**: Monospace ensures predictable layout
2. **Token efficiency**: Dense text packing reduces image size
3. **LLM readability**: Vision models trained on OCR-A/B perform better
4. **Standard**: OCR-A and OCR-B are ISO standards (ISO 1073)

**Usage**:
```python
from abstractcore.compression import GlyphProcessor, GlyphConfig

config = GlyphConfig(
    font_path="assets/OCRB.ttf",
    font_size=12,
    char_width=8,
    line_height=16
)

processor = GlyphProcessor(config)
image = processor.text_to_image("Dense text content...")
# Returns PIL Image with text rendered in OCR-B font
```

**Font Selection Guide**:
- **OCRA.ttf**: Most compact, highest density
- **OCRB.ttf**: Best balance of density and readability (recommended)
- **OCRBL.ttf**: Best readability, slightly larger

**When to Update**:
- Never (OCR-A and OCR-B are ISO standards, unchanged since 1973/1980)
- Only update if using alternative fonts (e.g., OCR-A Extended)

## Integration Points

### Architecture Detection
`architectures/detection.py` loads `architecture_formats.json` to:
- Detect model architecture from model name
- Get message format conventions
- Determine tool support type

### Provider Capabilities
`providers/` load `model_capabilities.json` to:
- Validate model names
- Check feature support (vision, tools, structured output)
- Calculate token limits
- Estimate image tokens

### Glyph Compression
`compression/` loads OCR fonts to:
- Render text to images
- Optimize visual-text compression
- Reduce token consumption by 3-4x

### Session Management
`core/session.py` uses `session_schema.json` to:
- Validate session data structure
- Export/import sessions
- Ensure data integrity

### Semantic Tools
`tools/` uses `semantic_models.md` to:
- Extract semantic triples
- Generate JSON-LD
- Validate ontology usage

## Maintenance Guide

### Adding New Model Architecture

1. **Edit architecture_formats.json**:
```json
{
  "architectures": {
    "new_model": {
      "description": "Description with release date",
      "message_format": "format_type",
      "system_prefix": "...",
      "system_suffix": "...",
      "user_prefix": "...",
      "user_suffix": "...",
      "assistant_prefix": "...",
      "assistant_suffix": "...",
      "tool_format": "native|prompted|special_token|none",
      "tool_prefix": "...",
      "patterns": ["pattern1", "pattern2"]
    }
  }
}
```

2. **Test detection**:
```python
from abstractcore.architectures import detect_architecture
arch = detect_architecture("new-model-name")
assert arch == ModelArchitecture.NEW_MODEL
```

### Adding New Model Capabilities

1. **Edit model_capabilities.json**:
```json
{
  "models": {
    "new-model": {
      "max_output_tokens": 4096,
      "tool_support": "native",
      "structured_output": "prompted",
      "parallel_tools": true,
      "max_tools": 10,
      "vision_support": false,
      "audio_support": false,
      "notes": "Description",
      "source": "Official docs",
      "canonical_name": "new-model",
      "aliases": [],
      "max_tokens": 32000
    }
  }
}
```

2. **Test capability lookup**:
```python
from abstractcore.utils import load_model_capabilities
caps = load_model_capabilities()
assert caps["new-model"]["max_tokens"] == 32000
```

### Updating Session Schema

1. **Edit session_schema.json** (follow JSON Schema Draft 07)
2. **Update schema_version** if breaking changes
3. **Add migration logic** for old schema versions
4. **Update session.py** to support new fields

### Adding OCR Fonts

1. **Copy font file** to `assets/`
2. **Update GlyphConfig** to support new font
3. **Test rendering** with all font options
4. **Document characteristics** in this README

## Best Practices

### DO:
✅ **Version control JSON files** - Track changes to capabilities and formats

✅ **Validate JSON structure** - Use JSON Schema validators

✅ **Add source attribution** - Document where capability info came from

✅ **Use canonical names** - Maintain consistency across system

✅ **Test after updates** - Verify detection and capability lookups work

✅ **Document ontology usage** - Follow semantic_models.md guidelines

### DON'T:
❌ **Don't modify OCR fonts** - They are ISO standards

❌ **Don't remove aliases** - Breaks backward compatibility

❌ **Don't guess capabilities** - Use official documentation

❌ **Don't break session schema** - Maintain migration path

❌ **Don't proliferate ontologies** - Stick to the 4 core ontologies

## File Formats

### JSON Files
- **Encoding**: UTF-8
- **Formatting**: 2-space indentation
- **Validation**: JSON Schema Draft 07
- **Line endings**: LF (Unix-style)

### TTF Fonts
- **Format**: TrueType Font
- **Encoding**: Unicode
- **Version**: TrueType 1.0

### Markdown Files
- **Encoding**: UTF-8
- **Format**: GitHub-flavored Markdown
- **Line endings**: LF (Unix-style)

## Testing Strategy

### Testing Architecture Detection
```python
import json
from abstractcore.architectures import detect_architecture

def test_architecture_formats():
    """Test that all architectures can be detected."""
    with open("assets/architecture_formats.json") as f:
        formats = json.load(f)

    for arch_name, arch_data in formats["architectures"].items():
        for pattern in arch_data["patterns"]:
            result = detect_architecture(pattern)
            assert result is not None, f"Failed to detect {pattern}"
```

### Testing Model Capabilities
```python
import json

def test_model_capabilities_structure():
    """Test that model capabilities have required fields."""
    with open("assets/model_capabilities.json") as f:
        capabilities = json.load(f)

    required_fields = ["max_tokens", "tool_support", "structured_output"]

    for model_name, model_data in capabilities["models"].items():
        for field in required_fields:
            assert field in model_data, f"{model_name} missing {field}"
```

### Testing Session Schema Validation
```python
from jsonschema import validate, ValidationError

def test_session_schema():
    """Test session schema validation."""
    with open("assets/session_schema.json") as f:
        schema = json.load(f)

    valid_session = {
        "schema_version": "session-archive/v1",
        "session": {
            "id": "test-session",
            "created_at": "2025-01-01T00:00:00Z"
        },
        "messages": []
    }

    # Should not raise
    validate(instance=valid_session, schema=schema)

    # Invalid session (missing required fields)
    invalid_session = {
        "schema_version": "session-archive/v1",
        "messages": []
    }

    try:
        validate(instance=invalid_session, schema=schema)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected
```

### Testing OCR Fonts
```python
from PIL import Image, ImageDraw, ImageFont
import os

def test_ocr_fonts():
    """Test that OCR fonts can be loaded and rendered."""
    fonts = ["OCRA.ttf", "OCRB.ttf", "OCRBL.ttf"]

    for font_name in fonts:
        font_path = f"assets/{font_name}"
        assert os.path.exists(font_path), f"Font not found: {font_name}"

        # Test loading
        font = ImageFont.truetype(font_path, size=12)

        # Test rendering
        img = Image.new("RGB", (200, 50), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Test OCR", font=font, fill="black")

        # Verify image was created
        assert img.size == (200, 50)
```

## Summary

The assets module provides critical static resources for AbstractCore:

1. **architecture_formats.json**: Enables architecture detection and message formatting for 30+ model families
2. **model_capabilities.json**: Comprehensive capability database for 150+ models across all providers
3. **session_schema.json**: Formal schema for session serialization with summary, assessment, and fact extraction
4. **semantic_models.md**: Ontology selection guide for semantic knowledge representation
5. **OCR fonts**: ISO-standard monospace fonts for glyph compression

**Key Characteristics**:
- **Zero dependencies**: Pure static assets
- **Foundation layer**: Used by all other modules
- **Versioned**: JSON files track schema versions
- **Validated**: JSON Schema ensures data integrity
- **Documented**: Each asset has clear purpose and structure

**Maintenance**:
- Update capabilities when new models are released
- Add architectures when new model families emerge
- Extend session schema as features are added
- Never modify OCR fonts (ISO standards)

**Integration**:
- Architecture detection uses format patterns
- Providers validate capabilities
- Compression uses OCR fonts
- Sessions use JSON schema
- Semantic tools follow ontology guide

## Related Modules

**Used by (consumes capabilities data)**:
- [`architectures/`](../architectures/README.md) - Model detection and capability queries
- [`media/`](../media/README.md) - Vision capability detection and format validation
- [`compression/`](../compression/README.md) - Provider-specific optimization profiles
- [`providers/`](../providers/README.md) - Token limits and feature detection
- [`structured/`](../structured/README.md) - Native structured output detection
- [`tools/`](../tools/README.md) - Tool support capability detection

**Data consumers**:
- [`config/`](../config/README.md) - Default model configuration
- [`server/`](../server/README.md) - API capability reporting
