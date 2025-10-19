# AbstractCore Codebase Structure and Media System Overview

## Executive Summary

AbstractCore is a unified Python library that provides a single interface to interact with multiple Large Language Model (LLM) providers. The codebase follows a layered architecture with a sophisticated media handling system that supports cross-provider multimodal processing. The project currently consists of 80+ Python files organized into 18 functional modules.

**Current Status**: Production-ready with comprehensive media support, provider registry system, and enhanced CLI/server capabilities.

---

## Project Organization (Top-Level)

```
abstractcore/
├── abstractcore/              # Main package
│   ├── apps/                  # High-level applications (extractor, judge)
│   ├── architectures/         # JSON-based model detection and capabilities
│   ├── assets/                # Static assets (images, examples)
│   ├── cli/                   # Command-line interface
│   ├── config/                # Configuration management system
│   ├── core/                  # Core LLM abstractions
│   ├── embeddings/            # Vector embedding support
│   ├── events/                # Event system for observability
│   ├── exceptions/            # Custom exception types
│   ├── media/                 # Media handling system (PRIMARY FOCUS)
│   ├── processing/            # Text processing applications
│   ├── providers/             # Multi-provider implementations
│   ├── server/                # FastAPI REST server
│   ├── structured/            # Structured output support
│   ├── tools/                 # Tool calling and syntax rewriting
│   ├── utils/                 # Utilities (logging, version, etc.)
│   └── __init__.py            # Package entrypoint
├── tests/                     # Comprehensive test suite
│   ├── media_handling/        # Media system tests
│   ├── provider_registry/     # Provider registry tests
│   ├── config/                # Configuration tests
│   ├── integration/           # Integration tests
│   ├── token_terminology/     # Token handling tests
│   └── [50+ test files]       # Feature-specific tests
├── examples/                  # Progressive usage examples
├── docs/                      # Documentation
└── README.md                  # Project overview
```

---

## 1. Core Architecture Layers

### Layer 1: Factory & Provider System
**Location**: `abstractcore/core/factory.py` + `abstractcore/providers/`

The factory system creates LLM instances using a unified interface:
```python
from abstractcore import create_llm
llm = create_llm("openai", model="gpt-4o")
```

**Supported Providers**:
- OpenAI (gpt-4, gpt-4o, gpt-4-turbo, gpt-4o-mini, etc.)
- Anthropic (Claude 3.5 Sonnet, Opus, Haiku)
- Ollama (local models: Qwen, Llama, Mistral, etc.)
- HuggingFace (GGUF models)
- MLX (Apple Silicon optimized)
- LMStudio (local inference)
- Mock (testing)

**Provider Registry** (`abstractcore/providers/registry.py`):
- Centralized provider metadata management
- Automatic model discovery per provider
- 137+ models across 7 providers
- Capability detection and validation

### Layer 2: Message & Interface System
**Location**: `abstractcore/core/interface.py`

Unified interface across all providers:
- Consistent message handling
- Tool calling support (OpenAI, Anthropic, Qwen, Llama formats)
- Streaming and non-streaming generation
- Token management (max_tokens, max_output_tokens, max_input_tokens)

### Layer 3: Session Management
**Location**: `abstractcore/core/session.py`

Persistent conversation management:
- Multi-turn conversation tracking
- Message metadata preservation
- Session analytics (summary, assessment, facts)
- Save/load functionality

### Layer 4: Architecture Detection
**Location**: `abstractcore/architectures/`

JSON-based model detection:
- Automatic architecture format detection (OpenAI, Codex, Qwen, Llama)
- Tool call format translation
- Model capability detection
- Message formatting rules

---

## 2. Media System Architecture (PRIMARY FOCUS)

The media system is a sophisticated multimodal processing framework that handles images, documents, audio, and video across all providers.

### 2.1 Core Media Types & Structures
**Location**: `abstractcore/media/types.py`

**Data Models**:
```python
MediaType:           # IMAGE, DOCUMENT, AUDIO, VIDEO, TEXT
ContentFormat:       # BASE64, URL, FILE_PATH, TEXT, BINARY, AUTO
MediaContent:        # Represents any piece of media with metadata
MultimodalMessage:   # Message containing mixed text + media content
MediaCapabilities:   # Provider/model capabilities registry
MediaProcessingResult: # Processing result with error handling
```

**Key Enums**:
- `MediaType`: 5 supported types
- `ContentFormat`: 6 content representations
- Automatic MIME type detection
- File extension mappings (50+ formats supported)

### 2.2 Base Media Handler Architecture
**Location**: `abstractcore/media/base.py`

**Hierarchy**:
```
BaseMediaHandler (Abstract Base Class)
├── AutoMediaHandler (Automatic processor selection)
├── ImageProcessor (Handles all image formats)
├── TextProcessor (Handles text documents)
├── PDFProcessor (PDF extraction with PyMuPDF4LLM)
└── OfficeProcessor (DOCX, XLSX, PPTX support)

BaseProviderMediaHandler (Provider-specific)
├── OpenAIMediaHandler (OpenAI format)
├── AnthropicMediaHandler (Anthropic format)
└── LocalMediaHandler (Ollama/local models)
```

**Key Responsibilities**:
- File validation and size limits
- Media type detection
- Format support verification
- Telemetry and event emission
- Error handling and fallback

### 2.3 Media Processors
**Location**: `abstractcore/media/processors/`

**ImageProcessor** (`image_processor.py`):
- Handles: JPG, JPEG, PNG, GIF, BMP, TIFF, WebP
- Features:
  - Base64 encoding
  - Automatic resolution optimization
  - Image scaling/resizing
  - Metadata extraction
  - Smart captioning fallback

**TextProcessor** (`text_processor.py`):
- Handles: TXT, MD, CSV, TSV, JSON, YAML, XML, HTML
- Features:
  - UTF-8 encoding/decoding
  - Format-specific parsing
  - Character counting
  - Metadata preservation

**PDFProcessor** (`pdf_processor.py`):
- Handles: PDF documents
- Requires: PyMuPDF4LLM library
- Features:
  - Text extraction
  - Table preservation
  - Layout understanding
  - Multi-page handling
  - Fallback to text processor if library unavailable

**OfficeProcessor** (`office_processor.py`):
- Handles: DOCX, XLSX, PPTX
- Requires: unstructured library
- Features:
  - Structured extraction
  - Formatting preservation
  - Sheet/slide enumeration
  - Fallback to text processor

### 2.4 Media Capability System
**Location**: `abstractcore/media/capabilities.py`

**MediaCapabilities Class**:
Comprehensive media support detection per model:
- Vision support (image input)
- Audio support
- Video support
- Document support
- Max images per message
- Supported formats per media type
- Max file sizes
- Token estimation
- Multimodal message support
- Text embedding preferences (local models)

**Provider-Specific Adjustments**:
- OpenAI: Multiple image formats, 20MB limit, 10 images per message (gpt-4o)
- Anthropic: 20 images/message, 5MB limit
- Ollama/MLX: Text embedding preferred, 10MB limit
- HuggingFace: 15MB limit

**Convenience Functions**:
```python
get_media_capabilities(model, provider)
is_vision_model(model)
is_multimodal_model(model)
get_supported_media_types(model, provider)
supports_images(model, provider)
supports_documents(model, provider)
get_max_images(model, provider)
should_use_text_embedding(model, provider)
```

### 2.5 Automatic Media Handler (Smart Router)
**Location**: `abstractcore/media/auto_handler.py`

**AutoMediaHandler Class**:
- Automatically selects appropriate processor based on file type
- Lazy initialization of processors (only when needed)
- Graceful fallback handling
- Processor availability detection
- Unified interface regardless of file type

**Selection Logic**:
```
File Type Detection
    ↓
IMAGE → ImageProcessor (if PIL available)
TEXT → TextProcessor (always available)
PDF → PDFProcessor (if PyMuPDF4LLM available) → TextProcessor (fallback)
OFFICE → OfficeProcessor (if unstructured available) → TextProcessor (fallback)
OTHER → Placeholder/error handling
```

### 2.6 Vision Fallback System
**Location**: `abstractcore/media/vision_fallback.py`

**Two-Stage Pipeline for Text-Only Models**:
1. Vision model generates image description
2. Text-only model receives description

**Features**:
- Automatic fallback chain (primary provider → fallback chain → local models)
- Local vision model support (BLIP, ViT-GPT2, GIT)
- Unified config system integration
- Error handling and fallback strategies

**Supported Local Models**:
- BLIP (Salesforce)
- ViT-GPT2 (Vision-to-Text)
- GIT (Generative Image-to-Text)
- Generic transformers pipeline

### 2.7 Provider-Specific Handlers
**Location**: `abstractcore/media/handlers/`

**OpenAIMediaHandler**:
- Formats content for OpenAI's vision API
- Base64 encoding, URL references
- Multi-image support handling
- MIME type standardization

**AnthropicMediaHandler**:
- Formats content for Anthropic's multimodal API
- Base64 encoding with media type
- Image stacking for multiple images
- Claude-specific requirements

**LocalMediaHandler**:
- Formats for Ollama/local models
- Base64 encoding
- Fallback text descriptions
- Device-aware optimization

### 2.8 Media Utils
**Location**: `abstractcore/media/utils/`

**ImageScaler** (`image_scaler.py`):
- Resolution optimization
- Aspect ratio preservation
- Format conversion
- Smart downscaling for vision models
- Multiple optimization strategies

---

## 3. Integration Points

### 3.1 Factory Integration
**File**: `abstractcore/core/factory.py`

Media handling integrated at provider creation:
```python
llm = create_llm("openai", model="gpt-4o")
response = llm.generate("What's in this image?", media=["photo.jpg"])
```

### 3.2 Session Integration
**File**: `abstractcore/core/session.py`

Sessions support media in messages:
```python
session.add_message('user', 'Analyze this:', media=['doc.pdf'])
```

### 3.3 Server Integration
**File**: `abstractcore/server/app.py`

FastAPI endpoints for media:
- `/v1/chat/completions` (with media support)
- `/v1/models` (provider listing)
- `/providers` (provider metadata)

### 3.4 Configuration Integration
**File**: `abstractcore/config/manager.py`

Vision configuration:
- Vision strategy (disabled, text_embedding, two_stage)
- Caption provider/model configuration
- Fallback chain management
- Local models path

### 3.5 CLI Integration
**File**: `abstractcore/cli/main.py`

Vision management commands:
- `abstractcore --status` (shows vision config)
- `abstractcore --set-vision-provider` (configure vision)
- `abstractcore --set-vision-caption` (set caption model)
- `abstractcore --download-vision-model` (local model setup)

---

## 4. Testing Infrastructure

### 4.1 Media System Tests
**Location**: `tests/media_handling/test_media_processors.py`

**Test Coverage**:
- ImageProcessor (PNG, JPEG, optimization)
- TextProcessor (CSV, Markdown)
- PDFProcessor (text extraction)
- Unsupported format handling
- File size validation
- MIME type detection

### 4.2 Provider Registry Tests
**Location**: `tests/provider_registry/`

**Test Coverage** (50+ tests):
- Registry core functionality
- Server integration
- Factory integration
- Model discovery
- Backward compatibility
- Error handling

### 4.3 Other Relevant Tests
- `test_provider_connectivity.py`: Provider integration
- `test_streaming_enhancements.py`: Streaming with media
- `test_embeddings_*.py`: Vector embedding tests

---

## 5. Configuration System

### 5.1 Config Manager
**Location**: `abstractcore/config/manager.py`

**Configuration Hierarchy**:
```
AbstractCoreConfig
├── vision: VisionConfig
│   ├── strategy: "disabled" | "text_embedding" | "two_stage"
│   ├── caption_provider: str | None
│   ├── caption_model: str | None
│   ├── fallback_chain: List[Dict]
│   └── local_models_path: str
├── embeddings: EmbeddingsConfig
├── default_models: DefaultModels
│   ├── cli_model: str
│   ├── summarizer: str
│   ├── extractor: str
│   └── judge: str
├── api_keys: ApiKeysConfig
├── cache: CacheConfig
└── logging: LoggingConfig
```

---

## 6. Key Design Patterns

### 6.1 Layered Architecture
```
CLI/Server ← Factory ← Providers ← Media System ← Processors
                              ↓
                      Configuration System
```

### 6.2 Strategy Pattern
- Multiple media processors (strategy selection)
- Multiple vision fallback options
- Multiple provider implementations

### 6.3 Factory Pattern
- `create_llm()` factory
- Provider registry factory
- Processor factory (AutoMediaHandler)

### 6.4 Adapter Pattern
- Provider-specific media handlers (OpenAI, Anthropic, Local)
- Format adapters (file → provider format)

### 6.5 Singleton Pattern
- ConfigurationManager (single instance per process)
- ProviderRegistry (single instance per process)
- EventSystem (single instance per process)

---

## 7. File Type Support Matrix

### Media Types and Extensions

**Images** (IMAGE):
- Common: jpg, jpeg, png, gif, webp
- Advanced: bmp, tiff, ico

**Documents** (DOCUMENT):
- Structured: pdf, docx, xlsx, pptx
- Markup: txt, md, csv, tsv, json, xml, html

**Audio** (AUDIO):
- Formats: mp3, wav, m4a, ogg, flac, aac
- Status: Detected but not processed

**Video** (VIDEO):
- Formats: mp4, avi, mov, mkv, webm, wmv
- Status: Detected but not processed

**Text** (TEXT):
- Formats: txt, md, csv, tsv, json, xml, html, yaml

---

## 8. Recent Enhancements (Task History)

### Task: Centralized Provider Registry (2025-10-16)
- Created `abstractcore/providers/registry.py`
- 137 models discovered across 7 providers
- Replaced hardcoded provider lists
- Enhanced server `/providers` endpoint
- Added comprehensive tests (50 tests, all passing)

### Task: Improved Status Command UI/UX (2025-10-18)
- Redesigned `abstractcore/cli/main.py:print_status()`
- 3-level hierarchical dashboard layout
- User-friendly descriptions
- Consistent status indicators (✅/⚠️/❌)
- Enhanced context clarity

---

## 9. Current Branch Status

**Branch**: `media-handling`

**Modified Files**:
- `abstractcore/cli/main.py`
- `abstractcore/utils/cli.py`
- `abstractcore/utils/structured_logging.py`
- `docs/centralized-config.md`

**Untracked Files** (Investigation docs):
- APPS_STREAMING_ARCHITECTURE_DIAGRAM.txt
- STREAMING_INVESTIGATION.md
- STREAMING_INVESTIGATION_INDEX.md
- Various investigation documents

---

## 10. Key Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 80+ |
| Main Package Modules | 18 |
| Supported Providers | 7 |
| Available Models | 137+ |
| Supported Media Types | 5 |
| Media Processors | 4 |
| Test Files | 50+ |
| Supported File Extensions | 50+ |

---

## 11. Integration Examples

### Example 1: Basic Image Analysis
```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "What's in this image?",
    media=["photo.jpg"]  # Auto-handled by media system
)
```

### Example 2: Document Analysis with Local Model
```python
from abstractcore import create_llm

llm = create_llm("ollama", model="qwen3-coder:30b")
response = llm.generate(
    "Summarize this research paper",
    media=["research.pdf"]  # Auto-converted to text
)
```

### Example 3: Vision Fallback
```python
from abstractcore import create_llm

# Text-only model with vision fallback configured
llm = create_llm("ollama", model="mistral:7b")
response = llm.generate(
    "What's in this image?",
    media=["photo.jpg"]  # Auto-converted via vision fallback
)
```

### Example 4: Session with Media
```python
from abstractcore import BasicSession, create_llm

llm = create_llm("anthropic", model="claude-3.5-sonnet")
session = BasicSession(llm)
session.add_message('user', 'Analyze these documents:', media=['a.pdf', 'b.docx'])
response = session.generate('What's the key insight?')
```

---

## 12. Performance Characteristics

### Processing Time Estimates (per file type)
- Images (with optimization): 50-100ms
- Text documents: 10-50ms
- PDFs (single page): 100-200ms
- Office documents: 200-500ms
- Images (large, 10MB): 500-1000ms

### Memory Usage
- ImageProcessor: ~50-100MB per image (PIL overhead)
- TextProcessor: ~1-2x file size
- PDFProcessor: ~5-10x content size (PyMuPDF overhead)
- OfficeProcessor: ~3-5x file size

---

## 13. Dependencies and Requirements

### Core Media Dependencies
- **PIL/Pillow**: Image processing (optional)
- **PyMuPDF4LLM**: PDF extraction (optional)
- **unstructured**: Office document processing (optional)
- **transformers**: Local vision models (optional)

### Provider Dependencies
- **openai**: OpenAI API
- **anthropic**: Anthropic API
- **ollama**: Ollama client
- **huggingface_hub**: HuggingFace models
- **mlx-lm**: Apple Silicon support
- **pydantic**: Data validation

---

## 14. Extensibility Points

### To Add New Media Type
1. Add to `MediaType` enum in `types.py`
2. Create processor class extending `BaseMediaHandler`
3. Add to `AutoMediaHandler._select_processor()`
4. Add file extensions to `FILE_TYPE_MAPPINGS`
5. Add tests in `tests/media_handling/`

### To Add New Provider Handler
1. Create class extending `BaseProviderMediaHandler`
2. Implement `format_for_provider()` method
3. Add to handlers `__init__.py`
4. Add provider-specific capability adjustments

### To Add New Provider
1. Create provider class extending base provider
2. Register in `providers/registry.py`
3. Add to `create_provider()` factory
4. Add model capabilities to `model_capabilities.json`
5. Add provider tests

---

## 15. Known Limitations and Future Enhancements

### Current Limitations
- Audio/Video types detected but not processed
- Local model vision fallback requires additional setup
- Streaming media support limited
- Parallel media processing not implemented

### Planned Enhancements
- Audio transcription support (Whisper integration)
- Video frame extraction and analysis
- Parallel media processing for efficiency
- Streaming media support
- Advanced image optimization strategies
- Cross-provider media capability negotiation

---

## Conclusion

AbstractCore provides a sophisticated, well-architected system for unified LLM access with comprehensive media handling. The codebase demonstrates strong software engineering practices with clear separation of concerns, extensive testing, and flexible extensibility. The media system is particularly robust, supporting multiple file types across diverse providers with intelligent fallback mechanisms.

The system is production-ready with 137+ models across 7 providers, comprehensive media support for 5 media types and 50+ file formats, and a mature configuration system for flexible deployment.
