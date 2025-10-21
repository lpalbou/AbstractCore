# Changelog

All notable changes to AbstractCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.5] - 2025-10-21

### Fixed

#### Critical Package Distribution Bug
- **Missing Media Subpackages**: Fixed critical package installation bug where media subpackages were not included in distribution
  - **Issue**: `pyproject.toml` only listed `abstractcore.media` parent package but not its subpackages
  - **Impact**: Import `from abstractcore import create_llm` failed with `ModuleNotFoundError: No module named 'abstractcore.media.processors'`
  - **Missing Packages**:
    - `abstractcore.media.processors` (ImageProcessor, PDFProcessor, OfficeProcessor, TextProcessor)
    - `abstractcore.media.handlers` (OpenAIMediaHandler, AnthropicMediaHandler, LocalMediaHandler)
    - `abstractcore.media.utils` (image_scaler utilities)
  - **Solution**: Explicitly added all media subpackages to packages list in `pyproject.toml`
  - **Root Cause**: When explicitly listing packages in pyproject.toml, setuptools does NOT auto-discover subpackages
  - **Workaround for 2.4.4**: Use `from abstractcore.core.factory import create_llm` instead of `from abstractcore import create_llm`
  - **Credit**: Bug discovered and reported during production deployment testing

#### Missing CLI Package
- **Missing abstractcore.cli Module**: Fixed missing `abstractcore.cli` package from distribution
  - **Issue**: CLI entry point `abstractcore` command referenced `abstractcore.cli.main:main` but module was not included in package
  - **Impact**: Configuration CLI commands would fail after installation from PyPI
  - **Solution**: Added `abstractcore.cli` to packages list in `pyproject.toml`

### Added

#### CLI Entry Point Improvements
- **New Entry Points**: Added convenient aliases to clarify CLI purpose and improve user experience
  - `abstractcore-config`: Alias for `abstractcore` command (configuration CLI for settings, API keys, models)
  - `abstractcore-chat`: New entry point for interactive REPL (`abstractcore.utils.cli` → LLM interaction)
  - **Purpose**: Distinguish between configuration CLI (manage settings) and interactive chat CLI (talk to LLMs)
  - **Backwards Compatible**: All existing commands continue to work (`abstractcore`, `python -m abstractcore.utils.cli`)

### Technical

#### Package Configuration
- **Updated packages list** in `pyproject.toml` to include all required modules:
  ```toml
  packages = [
      # ... existing packages ...
      "abstractcore.media",
      "abstractcore.media.processors",  # ✅ Added
      "abstractcore.media.handlers",    # ✅ Added
      "abstractcore.media.utils",       # ✅ Added
      "abstractcore.cli"                # ✅ Added
  ]
  ```
- **Verification**: All 19 packages now properly included in distribution
- **Testing**: Recommended to always test `pip install` from built wheel before PyPI release

### Benefits
- **Installation Works**: Users can now successfully `pip install abstractcore[all]` or `pip install abstractcore[media]`
- **Complete Media System**: All media processing capabilities (images, PDFs, Office docs) now accessible after installation
- **Clear CLI Commands**: Users have obvious entry points for different CLI purposes
- **Production Ready**: Package installation thoroughly tested and verified

### Migration Guide

No migration needed - this is a pure bug fix release. If you experienced installation issues with 2.4.4:

1. **Upgrade**: `pip install --upgrade abstractcore`
2. **Verify**: `python -c "from abstractcore import create_llm; print('✅ Works!')"`
3. **Use new CLI aliases** (optional):
   - `abstractcore-config --status` instead of `abstractcore --status`
   - `abstractcore-chat` instead of `python -m abstractcore.utils.cli`

## [2.4.4] - 2025-10-21

### Added

#### Provider Health Check System
- **NEW `.health()` Method**: Unified health check interface for all providers
  - **Structured Response**: Consistent health status format across all providers
  - **Connectivity Testing**: Uses `list_available_models()` as implicit connectivity test
  - **Smart Timeout Management**: Configurable timeout (default: 5.0s) with automatic restoration
  - **Never Throws**: Errors captured in response structure, never raises exceptions
  - **Rich Information**: Returns status, provider name, model list, model count, error message, and latency
  - **Universal Compatibility**: Works with all provider types (API, local, server-based)
  - **Override-able**: Providers can customize health check logic if needed

#### Health Check Response Structure
```python
{
    "status": bool,              # True if provider is healthy/online
    "provider": str,             # Provider class name (e.g., "OllamaProvider")
    "models": List[str] | None,  # Available models if online, None if offline
    "model_count": int,          # Number of models available (0 if offline)
    "error": str | None,         # Error message if offline, None if healthy
    "latency_ms": float          # Health check duration in milliseconds
}
```

### Fixed

#### HuggingFace Token Counting Consistency
- **Centralized Token Counter**: Fixed HuggingFace provider to use centralized `TokenUtils` for consistency
  - **Problem**: HuggingFace was the only provider using provider-specific `tokenizer.encode()` for token counting
  - **Solution**: Added `_calculate_usage()` method matching MLX provider pattern using `TokenUtils.estimate_tokens()`
  - **Impact**: All local providers now consistently use centralized token counting infrastructure
  - **Benefits**:
    - ✅ Consistency across all providers (MLX, HuggingFace)
    - ✅ Robustness when tokenizer unavailable (GGUF models)
    - ✅ Content-type detection for better accuracy (code vs text vs JSON)
    - ✅ Model-family adjustments (qwen, llama, mistral tokenization patterns)

### Enhanced

#### Token Usage Tracking
- **Comprehensive Token Capture**: All providers consistently capture THREE token metrics
  - **prompt_tokens**: Input/context tokens (system prompt + history + current prompt)
  - **completion_tokens**: Generated/output tokens (model's response)
  - **total_tokens**: Sum of prompt + completion (used for billing/quotas)
  - **API Providers**: OpenAI, Anthropic, Ollama, LMStudio use exact API-provided counts
  - **Local Providers**: MLX, HuggingFace use centralized `TokenUtils` estimation

### Technical

#### Token Counting Implementation
- **Centralized Infrastructure**: Located at `abstractcore/utils/token_utils.py`
  - `TokenUtils.estimate_tokens(text, model)`: Fast estimation with content-type detection
  - `TokenUtils.count_tokens(text, model, method)`: Flexible counting (auto/precise/fast)
  - `TokenUtils.count_tokens_precise(text, model)`: Accurate counting with tiktoken when available
  - Multi-tiered strategy: tiktoken (precise) → provider tokenizer → model-aware heuristics → fast fallback

#### Files Modified
- `abstractcore/providers/base.py`: Added `health()` method (lines 870-965)
- `abstractcore/providers/huggingface_provider.py`:
  - Added `_calculate_usage()` method using centralized TokenUtils (lines 890-902)
  - Updated `_single_generate_transformers()` to use centralized token counting (lines 867-868)

### Benefits
- **Health Monitoring**: Simple interface to check provider connectivity and availability
- **Consistency**: Unified token counting across all providers with same methodology
- **Production Ready**: Built-in timeout management prevents hanging health checks
- **Developer Experience**: Rich health information enables better error handling and monitoring
- **Maintainability**: Single centralized token counter to update/improve

### Migration Guide

#### For Health Check Users
New `.health()` method available on all providers:

```python
from abstractcore.core.factory import create_llm

# Check single provider
provider = create_llm("ollama", model="llama2")
health = provider.health(timeout=3.0)

if health["status"]:
    print(f"✅ {health['provider']} is healthy!")
    print(f"   📦 {health['model_count']} models available")
    print(f"   ⏱️  {health['latency_ms']}ms response time")
else:
    print(f"❌ {health['provider']} is offline")
    print(f"   Error: {health['error']}")
```

#### For Token Counting
No changes required - all existing code continues to work. HuggingFace provider now uses the same centralized token counting infrastructure as other local providers, improving consistency and accuracy.

## [2.4.3] - 2025-10-20

### Major Features

#### OpenAI Responses API Compatibility
- **NEW `/v1/responses` Endpoint**: 100% compatible with OpenAI's Responses API format
  - **input_file Support**: Native support for `{"type": "input_file", "file_url": "..."}` in content arrays
  - **Backward Compatible**: Existing `messages` format continues to work alongside new `input` format
  - **Automatic Format Detection**: Server automatically detects and converts between OpenAI and legacy formats
  - **Streaming Support**: Optional streaming with `"stream": true` for real-time responses (defaults to `false`)
  - **Universal File Processing**: Works with all file types (PDF, DOCX, XLSX, CSV, images) across all providers

#### Enhanced File Attachment System
- **type="file" Support**: New content type alongside `"text"` and `"image_url"` for explicit file attachments
  - **Unified Format**: `{"type": "file", "file_url": {"url": "..."}}` works consistently across all endpoints
  - **Multiple Sources**: Supports HTTP(S) URLs, local file paths, and base64 data URLs
  - **Content-Type Detection**: Intelligent file type detection from headers and URL extensions
  - **Generic Downloader**: Replaces image-only downloader with universal file download supporting 15+ file types

#### Production-Grade PDF Processing
- **Complete Text Extraction**: Full PDF content extraction using PyMuPDF4LLM with formatting preservation
  - **40,000+ Character Support**: Successfully tested with large documents (Berkshire Hathaway annual letter)
  - **LLM-Optimized Output**: Markdown formatting with preserved tables, headers, and structure
  - **Automatic Installation**: Added PyMuPDF4LLM, PyMuPDF, and Pillow to dependencies
  - **Graceful Fallbacks**: Multi-level fallback ensures content extraction even if advanced processing fails

#### Centralized Configuration System
- **Global Configuration Management**: Unified configuration at `~/.abstractcore/config/abstractcore.json`
  - **App-Specific Defaults**: Set different models for CLI, summarizer, extractor, and judge apps
  - **Global Fallbacks**: Configure fallback models when app-specific settings aren't available
  - **API Key Management**: Centralized API key storage for all providers
  - **Cache Configuration**: Configurable cache directories for HuggingFace, local models, and general cache
  - **Logging Control**: Console and file logging levels with enable/disable commands
  - **Streaming Defaults**: Configure default streaming behavior for CLI applications

#### Comprehensive Media Handling System
- **Universal Media API**: Same `media=[]` parameter works across all providers with automatic format conversion
  - **Image Processing**: Automatic resolution optimization for each model's maximum capability (GPT-4o: 4096px, Claude 3.5: 1568px, qwen2.5vl: 3584px)
  - **Document Processing**: Full support for PDF, DOCX, XLSX, PPTX with complete content extraction
  - **Data Files**: CSV, TSV, JSON, XML with intelligent parsing and analysis
  - **Provider-Specific Formatting**: Automatic conversion to OpenAI JSON, Anthropic Messages API, or local text embedding
  - **Error Handling**: Multi-level fallback strategy ensures users always get meaningful results

#### Vision Capabilities and Fallback System
- **Vision Fallback for Text-Only Models**: Transparent two-stage pipeline enables image processing for any model
  - **Automatic Detection**: Identifies when text-only models receive images and activates fallback
  - **One-Command Setup**: `abstractcore --download-vision-model` downloads and configures BLIP vision model
  - **Flexible Configuration**: Supports local models (BLIP, ViT-GPT2, GIT), Ollama, LMStudio, and cloud APIs
  - **Transparent Operation**: Users don't need to change code - system handles vision fallback automatically

### Server Enhancements

#### Enhanced Debug and Logging
- **Command-Line Arguments**: Added `--debug`, `--host`, and `--port` flags for flexible server startup
  - **Debug Mode**: `--debug` enables comprehensive request/response logging with timing metrics
  - **Custom Binding**: `--host` and `--port` allow custom server addresses (default: 127.0.0.1:8000)
  - **Environment Integration**: Follows centralized config patterns with `ABSTRACTCORE_DEBUG` variable

- **Comprehensive Error Reporting**: Enhanced 422 validation error handling with actionable diagnostics
  - **Field-Level Details**: Shows exact field path, validation message, and problematic input
  - **Request Body Capture**: In debug mode, logs full request body for troubleshooting
  - **Structured Logging**: JSON-formatted logs with client IP, timing, and error context
  - **Before vs After**: "422 Unprocessable Entity" now shows detailed field validation errors

#### Media Processing Integration
- **OpenAI Vision API Format**: Full support for `image_url` objects with base64 data URLs and HTTP(S) URLs
- **File Processing Pipeline**: Automatic media extraction, validation, and cleanup with request-specific prefixes
- **Size Limits**: 10MB per file, 32MB total per request with comprehensive validation
- **Cleanup Logic**: Automatic temporary file cleanup for `abstractcore_img_*`, `abstractcore_file_*`, and `abstractcore_b64_*` prefixes
- **Prompt Adaptation**: Intelligent prompt adaptation based on file types to avoid confusion

### Fixed

#### Critical Runtime Issues
- **Time Module Scoping**: Removed redundant local `import time` statements causing "cannot access local variable" errors
  - Fixed in lines 1995-1996 and 2123-2124 of `abstractcore/server/app.py`
  - Now uses global time import consistently throughout server

- **Boolean Syntax**: Corrected JavaScript boolean syntax (`false`/`true`) to Python syntax (`False`/`True`)
  - Fixed in lines 625, 813, 824, 1170, 1181, 1214 across request examples and defaults

- **Streaming Default**: Changed `/v1/responses` endpoint default from `stream=True` to `stream=False`
  - Aligns with OpenAI API standard behavior (streaming opt-in, not opt-out)
  - Line 361 in `OpenAIResponsesRequest` model

#### Swagger UI Integration
- **Payload Input Issue**: Fixed `/v1/responses` endpoint not showing request body in Swagger "Try it out"
  - Replaced raw `Request` parameter with proper FastAPI `Body(...)` annotation
  - Added comprehensive examples for OpenAI format, legacy format, file analysis, and streaming
  - Lines 1148-1220 now properly expose request schema to OpenAPI documentation

#### Media Processing Reliability
- **PDF Download Failures**: Created generic file downloader replacing image-only version
  - Added proper `Accept: */*` headers instead of image-specific headers
  - Comprehensive content-type mapping for PDF, DOCX, XLSX, CSV, and 10+ other types
  - URL extension fallback when content-type header missing
  - Lines 1502-1627 in `abstractcore/server/app.py`

### Enhanced

#### CLI Applications
- **Centralized Configuration Integration**: All CLI apps (summarizer, extractor, judge) now use centralized config
  - Apps respect `abstractcore --set-app-default` configuration
  - Fallback to global defaults when app-specific config not set
  - Enhanced `--debug` mode for all applications

- **Vision Configuration CLI**: New `abstractcore/cli/vision_config.py` for vision fallback setup
  - Interactive configuration wizard
  - Model download commands
  - Status checking and validation

#### Documentation
- **Centralized Configuration**: Created `docs/centralized-config.md` with complete configuration system documentation
  - All available commands with examples
  - Configuration file format and priority system
  - Troubleshooting guide and common tasks

- **Media Handling System**: Comprehensive `docs/media-handling-system.md` with production-tested examples
  - "How It Works Behind the Scenes" section explaining multi-layer architecture
  - Provider-specific formatting documentation (OpenAI JSON, Anthropic Messages API)
  - Real-world CLI usage examples with verified working commands
  - Model compatibility matrix and resolution limits

- **Server Documentation**: Updated `docs/server.md` with `/v1/responses` endpoint details
  - OpenAI Responses API format examples
  - File attachment workflows
  - Streaming configuration
  - Media processing capabilities

### Technical

#### Architecture Improvements
- **Provider Registry Enhancement**: Leverages centralized provider registry for model discovery
  - `/providers` endpoint returns complete provider metadata
  - No hardcoded provider lists - all dynamic discovery
  - Registry version 2.0 indicators in API responses

- **Message Preprocessing**: New `MessagePreprocessor` for `@filename` syntax in CLI
  - Extracts file attachments from text
  - Validates file existence
  - Cleans text for LLM processing

- **Media Type Detection**: Intelligent file type detection and processor selection
  - AutoMediaHandler coordinates specialized processors
  - ImageProcessor, PDFProcessor, OfficeProcessor, TextProcessor
  - Graceful fallback ensures processing never fails completely

#### Test Coverage
- **Media Examples**: Added comprehensive test assets in `tests/media_examples/`
  - PDF reports, Office documents, spreadsheets, presentations
  - CSV/TSV data files with various encodings
  - Image examples with metadata

- **Server Testing**: Enhanced test suite for media processing and OpenAI compatibility
  - Real file processing tests (not mocked)
  - Cross-provider media handling verification
  - Streaming with media attachments

### Breaking Changes
None. All changes maintain full backward compatibility with version 2.4.x.

### Migration Guide

#### For Server Users
The `/v1/responses` endpoint now accepts both OpenAI's `input` format and our legacy `messages` format:

**OpenAI Responses API Format (Recommended):**
```json
{
  "model": "gpt-4o",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Analyze this document"},
        {"type": "input_file", "file_url": "https://example.com/doc.pdf"}
      ]
    }
  ],
  "stream": false
}
```

**Legacy Format (Still Supported):**
```json
{
  "model": "openai/gpt-4",
  "messages": [
    {"role": "user", "content": "Tell me a story"}
  ],
  "stream": false
}
```

**Note**: Streaming is now opt-in (set `"stream": true`) instead of automatic, matching OpenAI's behavior.

#### For Configuration Users
New centralized configuration system available:

```bash
# Set global default model
abstractcore --set-global-default ollama/llama3:8b

# Set app-specific defaults
abstractcore --set-app-default summarizer openai gpt-4o-mini
abstractcore --set-app-default extractor ollama qwen3:4b-instruct

# Configure logging
abstractcore --set-console-log-level WARNING
abstractcore --enable-file-logging

# Check current configuration
abstractcore --status
```

Configuration is stored in `~/.abstractcore/config/abstractcore.json` and respects priority:
1. Explicit parameters (highest priority)
2. App-specific configuration
3. Global configuration
4. Hardcoded defaults (lowest priority)

#### For Media Processing Users
Media processing now supports explicit file types:

**CLI (Using @filename syntax):**
```bash
python -m abstractcore.utils.cli --prompt "Analyze @report.pdf and @chart.png"
```

**Python API:**
```python
response = llm.generate(
    "Analyze these documents",
    media=["report.pdf", "chart.png", "data.xlsx"]
)
```

**Server API (New type="file"):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this file"},
        {"type": "file", "file_url": {"url": "https://example.com/doc.pdf"}}
      ]
    }
  ]
}
```

All formats work identically across all providers with automatic format conversion.

### Dependencies Added
- `pymupdf4llm` (0.0.27): LLM-optimized PDF text extraction
- `pymupdf` (1.26.5): Core PDF processing library
- `pydantic` (2.12.3): Request validation and serialization
- `fastapi`: Enhanced with latest features
- `pillow` (12.0.0): Image processing support

### Benefits
- **Users**: Seamless file attachment across all providers with `@filename` CLI syntax and `media=[]` API
- **Developers**: OpenAI-compatible server endpoints with comprehensive media processing
- **Production**: Robust error handling, detailed logging, and graceful degradation
- **Configuration**: Single source of truth for all package-wide preferences and defaults

## [2.4.3] - 2025-10-19

### Fixed
- **Media System Critical Fixes**: Resolved implementation issues preventing full media processing functionality
  - **PDF Processing**: Fixed `output_format` parameter conflict in `PDFProcessor._create_media_content()` call (line 128) causing "got multiple values for keyword argument" error
  - **Office Document Processing**: Fixed element iteration errors in `OfficeProcessor` by replacing `convert_to_dict()` approach with direct element processing for DOCX, XLSX, and PPTX files
  - **Unstructured Library Integration**: Updated office processor to work correctly with current unstructured library API, eliminating "'NarrativeText' object is not iterable" and "'Table' object is not iterable" errors

### Enhanced
- **Production-Ready Media System**: All file types now working perfectly with comprehensive content extraction
  - **PDF Files**: Full text extraction with formatting preservation using PyMuPDF4LLM
  - **Word Documents**: Complete document analysis with structure preservation (DOCX)
  - **Excel Spreadsheets**: Sheet-by-sheet content extraction with intelligent data analysis (XLSX)
  - **PowerPoint Presentations**: Slide content extraction with comprehensive presentation analysis (PPTX)
  - **CSV/TSV Files**: Intelligent data parsing with quality assessment and recommendations
  - **Images**: Seamless vision model integration with existing test infrastructure

- **Server Debug Support**: Comprehensive debug mode for troubleshooting API issues
  - **Command Line Interface**: Added `--debug`, `--host`, and `--port` arguments to server startup with comprehensive help
  - **Enhanced Error Logging**: Detailed 422 validation error reporting with field-level diagnostics and request body capture
  - **Request/Response Tracking**: Full HTTP request logging with client information, timing metrics, and structured JSON output
  - **Centralized Configuration Integration**: Follows centralized config system patterns with environment variable support
  - **Before vs After**: Uninformative "422 Unprocessable Entity" messages now provide actionable field validation details

### Verified
- **CLI Integration**: Confirmed `@filename` syntax works flawlessly across all file types
  - Tested with real files: PDF reports, Office documents, spreadsheets, presentations, data files, and images
  - Cross-provider compatibility verified with OpenAI, Anthropic, and LMStudio providers
  - All examples documented in `docs/media-handling-system.md` are production-tested and working

### Documentation
- **Comprehensive Media System Documentation**: Completely rewrote `docs/media-handling-system.md` to reflect actual implementation
  - Added detailed "How It Works Behind the Scenes" section explaining the multi-layer architecture
  - Documented provider-specific formatting (OpenAI JSON, Anthropic Messages API, local text embedding)
  - Added real-world CLI usage examples with verified working commands
  - Included cross-provider workflow diagrams and error handling strategies
- **Architecture Documentation**: Updated `docs/architecture.md` with comprehensive media system architecture section
  - Added media processing workflow diagrams and component descriptions
  - Documented graceful fallback strategy and provider-specific formatting
  - Included unified media API documentation and CLI integration details

### Technical
- **Robust Error Handling**: Multi-level fallback strategy ensures users always get meaningful results
  - Advanced processing with specialized libraries (PyMuPDF4LLM, Unstructured)
  - Basic processing fallbacks for text extraction
  - Metadata-only fallbacks when all else fails
  - System never crashes or fails completely
- **Test Infrastructure**: Leveraged existing `tests/vision_examples/` with production-quality test assets
  - 5 high-quality images with comprehensive JSON metadata for validation
  - Real-world testing with actual provider APIs and file processing

### Benefits
- **Users**: Can immediately attach any file type using `@filename` syntax with excellent analysis results
- **Developers**: Universal `media=[]` parameter works identically across all providers
- **Production**: Reliable media processing with comprehensive error handling and graceful degradation
- **CLI**: Simple file attachment workflow that works with all supported file formats

## [2.4.2] - 2025-10-16

### Added
- **Centralized Provider Registry System**: Unified provider discovery and metadata management
  - **Single Source of Truth**: Created `abstractcore/providers/registry.py` with `ProviderRegistry` class for centralized provider management
  - **Package-wide Discovery Function**: `get_all_providers_with_models()` provides unified access to ALL providers with complete metadata
  - **Complete Model Lists**: Fixed truncation issue - now returns all models without "... and X more" truncation
  - **Rich Metadata**: Installation instructions, features, authentication requirements, supported capabilities automatically available
  - **HTTP API Integration**: Server `/providers` endpoint now uses centralized registry (registry_version: "2.0")
  - **Dynamic Discovery**: Automatically discovers providers without hardcoding, eliminating manual synchronization

### Enhanced
- **Factory System**: Simplified `create_llm()` from 70+ line if/elif chain to single registry call while maintaining full backward compatibility
- **Server Endpoints**: Enhanced `/providers` endpoint with comprehensive metadata including model counts, features, and installation instructions
- **Documentation**: Added "Provider Discovery" section to both `llms.txt` and `llms-full.txt` with Python API and HTTP API examples
- **Error Messages**: Improved error messages with dynamic provider lists from registry

### Fixed
- **Manual Provider Synchronization**: Eliminated need to manually update provider lists across factory.py, server/app.py, and documentation
- **Model List Truncation**: Fixed "... and X more" truncation - now returns complete model lists for all providers
- **Provider Metadata Inconsistency**: Centralized all provider information including features, authentication requirements, and installation extras

### Technical
- **Comprehensive Test Suite**: Added 50 tests in `tests/provider_registry/` covering core functionality, server integration, and factory integration
- **Lazy Loading**: Provider classes loaded on-demand for better performance and memory usage
- **Backward Compatibility**: All existing code continues to work unchanged - no breaking changes
- **Extensible Architecture**: Easy to add new providers by registering them in the centralized registry

### Benefits
- **Developers**: Single function to discover all providers programmatically
- **Server Users**: Enhanced `/providers` endpoint with rich metadata
- **Maintainers**: No more manual provider list synchronization across multiple files
- **Documentation**: Always up-to-date provider information in docs

## [2.4.1] - 2025-10-16

### Fixed
- **Critical Package Distribution Fix**: Fixed `ModuleNotFoundError: No module named 'abstractcore.exceptions'` that occurred when installing from PyPI
  - Added missing `abstractcore.exceptions` and `abstractcore.media` packages to the setuptools configuration in `pyproject.toml`
  - This issue was introduced during the refactoring process when these modules were not included in the package distribution list
  - Users can now successfully import `from abstractcore import create_llm` after installing from PyPI
  - Verified fix by building and testing the wheel package with the corrected configuration

## [2.4.0] - 2025-10-15

### Breaking Changes
- **Complete Rebranding**: Comprehensive rename from "AbstractLLM" to "AbstractCore" throughout the entire project
  - **Package Name**: Internal package `abstractllm/` → `abstractcore/` to align with published package name
  - **Product Name**: "AbstractLLM Core" → "AbstractCore" in all documentation and branding
  - **Import statements**: All `from abstractcore import ...` must become `from abstractcore import ...`
  - **Console scripts**: Entry points changed from `abstractllm.apps.*` to `abstractcore.apps.*`
  - **Interface names**: `AbstractLLMInterface` → `AbstractCoreInterface`, `AbstractLLMError` → `AbstractCoreError`
  - **Environment variables**: `ABSTRACTLLM_*` → `ABSTRACTCORE_*` (e.g., `ABSTRACTCORE_ONNX_VERBOSE`)
  - **Cache directories**: `~/.abstractllm/` → `~/.abstractcore/`
  - **Log files**: `abstractllm_*.log` → `abstractcore_*.log`
  - **Module paths**: All absolute imports updated throughout codebase
  - **Impact**: This affects all users - complete migration required from AbstractLLM to AbstractCore branding
  
### Migration Guide
To migrate from 2.3.x to 2.4.0, update all references to AbstractLLM:

**1. Import Statements:**
```python
# Before (2.3.x)
from abstractcore import create_llm
from abstractllm.processing import BasicSummarizer
from abstractllm.embeddings import EmbeddingManager

# After (2.4.0+)
from abstractcore import create_llm
from abstractcore.processing import BasicSummarizer  
from abstractcore.embeddings import EmbeddingManager
```

**2. Interface Names:**
```python
# Before (2.3.x) 
from abstractllm.core.interface import AbstractLLMInterface

# After (2.4.0+)
from abstractcore.core.interface import AbstractCoreInterface
```

**3. Environment Variables:**
```bash
# Before (2.3.x)
export ABSTRACTLLM_ONNX_VERBOSE=1

# After (2.4.0+)
export ABSTRACTCORE_ONNX_VERBOSE=1
```

**4. Console Scripts:**
Console scripts remain the same (both `summarizer` and `abstractcore-summarizer` work), but internal module paths have changed to `abstractcore.apps.*`.

### Technical
- **Directory Structure**: Renamed main package directory from `abstractllm/` to `abstractcore/`
- **Configuration Updates**: Updated `pyproject.toml` with new package names, console scripts, and version paths
- **Build System**: Cleaned and regenerated all build artifacts with correct package structure
- **Documentation**: Updated all code examples, CLI usage, and module references across documentation
- **Examples**: Updated all example files with new import statements
- **Tests**: Updated all test imports and references throughout test suite

## [2.3.9] - 2025-10-25
### Fixed
- **Timeout Handling**: Comprehensive timeout parameter handling across all providers
  - All providers now properly handle `timeout=None` (infinity) as the default
  - **HuggingFace Provider**: Issues warning when non-None timeout is provided (local models don't support timeouts)
  - **MLX Provider**: Issues warning when non-None timeout is provided (local models don't support timeouts)  
  - **Mock Provider**: Accepts timeout parameters for testing without warnings
  - **API Providers** (OpenAI, Anthropic, Ollama, LMStudio): Properly pass timeout to HTTP clients
  - Added `_update_http_client_timeout()` method for providers that need to update client timeouts
- Setting timeout default to None (infinity)

## [2.3.8] - 2025-10-25
### Fixed
- Issue with the version

## [2.3.7] - 2025-10-25

### Fixed
- **Syntax Warning**: Fixed invalid escape sequence `\(` in `common_tools.py` docstring example
- **CLI Enhancement**: Added optional focus parameter to `/compact` command for targeted conversation summarization
  - Usage: `/compact [focus]` where focus can be "technical details", "key decisions", etc.
  - Leverages existing `BasicSummarizer` focus functionality for more precise compaction
  - Maintains backward compatibility (no focus = default behavior)

## [2.3.6] - 2025-10-14

### Added
- **Vector Embeddings**: SOTA open-source models with EmbeddingGemma as default, ONNX optimization, multi-provider support (HuggingFace, Ollama, LMStudio)
- **Processing Applications**: BasicSummarizer, BasicExtractor, BasicJudge with CLI tools and structured output
- **GitHub Pages Website**: Professional documentation site with responsive design and provider showcase
- **Unified Streaming Architecture**: Real-time tool call detection and execution across all providers
- **Memory Management**: Provider unload() methods for resource management in constrained environments
- **Session Management**: Complete serialization with analytics (summary, assessment, facts)
- **CLI Enhancements**: Interactive REPL with tool integration, session persistence, and comprehensive help system

### Fixed
- **Critical Tool Compatibility**: Tools + structured output now work together with sequential execution pattern
- **Ollama Endpoint Selection**: Fixed verbose responses by using correct `/api/chat` endpoint
- **Streaming Tool Execution**: Consistent formatting between streaming and non-streaming modes
- **Architecture Detection**: Corrected Qwen3-Next models and universal tool call parsing
- **Session Serialization**: Fixed parameter consistency and tool result integration
- **Timeout Configuration**: Unified timeout management across all components (default: 5 minutes)
- **Package Dependencies**: Made processing module core dependency, fixed installation extras

### Enhanced
- **Multi-Provider Embedding**: Unified API across HuggingFace, Ollama, LMStudio with caching and optimization
- **Tool Call Syntax Rewriting**: Server-side format conversion for agentic CLI compatibility
- **Documentation**: Consolidated and professional tone, comprehensive tool calling guide
- **Token Management**: Helper methods and validation with provider-specific recommendations
- **Test Coverage**: 346+ tests with real models, comprehensive provider testing

### Technical
- **Event System**: Real-time monitoring and observability with OpenTelemetry compatibility
- **Circuit Breakers**: Netflix Hystrix pattern with exponential backoff retry strategy
- **FastAPI Server**: OpenAI-compatible endpoints with comprehensive parameter support
- **Model Discovery**: Heuristic-based filtering and provider-specific routing

## [2.3.5] - 2025-10-14

### Fixed

#### CRITICAL: Tools + Structured Output Compatibility
- **Problem**: AbstractCore's `tools` and `response_model` parameters were mutually exclusive, preventing users from combining function calling with structured output validation
- **Root Cause**: `StructuredOutputHandler` bypassed normal tool execution flow and tried to validate tool call JSON against Pydantic model
- **Solution**: Implemented sequential execution pattern - tools execute first, then structured output uses results as context
- **Impact**: Enables sophisticated LLM applications requiring both function calling and structured output validation
- **Usage**: `llm.generate(tools=[func], response_model=Model, execute_tools=True)` now works seamlessly
- **Limitation**: Streaming not supported in hybrid mode (clear error message provided)

#### Enhanced BaseProvider Interface
- **Added**: `generate()` method to BaseProvider implementing AbstractCoreInterface
- **Fixed**: Proper delegation from `generate()` to `generate_with_telemetry()` with full parameter passthrough
- **Impact**: Ensures consistent API behavior across all provider implementations

### Technical

#### Implementation Details
- Added `_handle_tools_with_structured_output()` method with sequential execution strategy
- Modified `generate_with_telemetry()` to detect and route hybrid requests appropriately
- Enhanced prompt engineering to inject tool execution results into structured output context
- Maintained full backward compatibility for single-mode usage (tools-only or structured-only)

#### Files Modified
- `abstractcore/providers/base.py`: Added hybrid handling logic and generate() method implementation
- Sequential execution: Tool execution → Context enhancement → Structured output generation
- Clean error handling with descriptive messages for unsupported combinations

#### Test Results
✅ Tools-only mode: Works correctly  
✅ Structured output-only mode: Works correctly  
✅ **NEW**: Hybrid mode (tools + structured output): Now works correctly  
✅ Backward compatibility: All existing functionality preserved  
✅ Error handling: Clear messages for unsupported streaming + hybrid combination

## [2.3.4] - 2025-10-14

### Added

#### State-of-the-Art GitHub Pages Website
- **Professional Website**: Created comprehensive GitHub Pages website at `https://lpalbou.github.io/AbstractCore/`
- **Modern UI/UX**: Responsive design with dark/light theme toggle, smooth animations, and mobile-first approach
- **Interactive Features**: Code block copy functionality, smooth scrolling navigation, and dynamic theme switching
- **Provider Showcase**: Visual display of all supported LLM providers (OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace)
- **SEO Optimization**: Complete sitemap.xml, robots.txt, and meta tags for search engine visibility
- **LLM Integration**: Added `llms.txt` and `llms-full.txt` files for enhanced LLM compatibility and content discovery

#### Comprehensive Tool Calling Documentation
- **New Documentation**: Created `docs/tool-calling.md` with complete coverage of the tool calling system
- **Rich Decorator Examples**: Documented the full capabilities of the `@tool` decorator including metadata injection
- **Architecture-Aware Formatting**: Explained how tool definitions adapt to different model architectures (Qwen, LLaMA, Gemma)
- **Tool Syntax Rewriting**: Integrated comprehensive documentation of Tag Rewriter and Syntax Rewriter systems
- **Real-World Examples**: Showcased actual tools from `common_tools.py` with full metadata and system prompt integration

### Enhanced

#### Documentation Consolidation and Cleanup
- **Professional Tone**: Removed pretentious language, excessive emojis, and marketing hype from all documentation
- **Consolidated Content**: Merged `tool-syntax-rewriting.md` into comprehensive `tool-calling.md` documentation
- **Fixed Cross-References**: Updated all internal links in README.md, docs/README.md, and getting-started.md
- **Consistent Styling**: Standardized documentation format and removed redundant content
- **HTML Documentation**: Created HTML versions of all documentation for the GitHub Pages website

#### Website Architecture
- **Static Site Generation**: Pure HTML/CSS/JavaScript implementation for maximum performance and compatibility
- **Asset Organization**: Structured asset directory with optimized SVG logos and provider icons
- **GitHub Pages Optimization**: Added `.nojekyll` file and proper CNAME configuration for custom domains
- **Documentation Integration**: Seamless integration between website and documentation with consistent navigation

### Technical

#### Files Added
- `index.html`: Main landing page with hero section, features showcase, and provider display
- `assets/css/main.css`: Comprehensive styling with CSS variables for theming and responsive design
- `assets/js/main.js`: Interactive functionality including theme switching and mobile navigation
- `llms.txt`: Concise LLM-friendly project overview with key documentation links
- `llms-full.txt`: Complete documentation content aggregated for LLM consumption
- `docs/tool-calling.html`: HTML version of comprehensive tool calling documentation
- `robots.txt` and `sitemap.xml`: SEO optimization files for search engine discovery

#### Documentation Updates
- Enhanced `docs/tool-calling.md` with complete `@tool` decorator capabilities and real-world examples
- Updated README.md, docs/README.md, and docs/getting-started.md with professional tone and correct links
- Removed redundant `docs/tool-syntax-rewriting.md` after content integration
- Fixed all cross-references and internal navigation links

#### GitHub Pages Deployment
- Created clean `gh-pages` branch with optimized website content
- Implemented proper GitHub Pages configuration with SEO optimization
- Added comprehensive LLM compatibility files for enhanced discoverability
- Structured deployment ready for custom domain configuration

### Impact
- **Enhanced Developer Experience**: Professional website provides clear project overview and easy navigation
- **Improved Documentation Quality**: Consolidated, professional documentation without redundancy or pretentious language
- **Better LLM Integration**: Structured `llms.txt` files enable better LLM understanding and interaction with the project
- **Increased Discoverability**: SEO-optimized website improves project visibility and accessibility
- **Comprehensive Tool Documentation**: Complete coverage of tool calling system with practical examples and architecture details

## [2.3.3] - 2025-10-14

### Fixed

#### ONNX Runtime Warning Suppression
- **Problem**: ONNX Runtime displayed verbose CoreML execution provider warnings on macOS during embedding model initialization
- **Root Cause**: ONNX Runtime logs informational messages about CoreML partitioning and node assignment directly to stderr, bypassing Python's warning system
- **Solution**: Added ONNX Runtime log level configuration in `_suppress_onnx_warnings()` to suppress harmless informational messages
- **Impact**: Cleaner console output during embedding operations while preserving debugging capability via `ABSTRACTLLM_ONNX_VERBOSE=1` environment variable
- **Technical**: Set `onnxruntime.set_default_logger_severity(3)` to suppress warnings that don't affect performance or quality

## [2.3.2] - 2025-10-14

### Fixed

#### Critical Ollama Endpoint Selection Bug
- **Problem**: Ollama provider was generating excessively verbose responses (1000+ characters for simple questions like "What is 2+2?")
- **Root Cause**: Provider incorrectly used `/api/generate` endpoint for all requests, including tool-enabled conversations
- **Solution**: Updated endpoint selection logic to use `/api/chat` by default, following Ollama's API design recommendations
- **Impact**: Reduced response length from 977+ characters to 15 characters for simple queries, eliminated "infinite text" generation issue
- **Technical**: Modified `_generate_internal()` method to use `use_chat_format = tools is not None or messages is not None or True` for proper endpoint routing

#### Session Serialization Parameter Consistency
- **Problem**: Inconsistent parameter naming between `session.add_message()` using `name` and `session.generate()` using `username`
- **Root Cause**: Parameter standardization was incomplete during metadata redesign
- **Solution**: Standardized both methods to use `name` parameter, aligning with `session_schema.json` specification
- **Impact**: Consistent API across session methods, improved developer experience

#### Tool Execution Results in Live Sessions
- **Problem**: Tool execution results were missing from chat history during live CLI sessions but appeared after session reload
- **Root Cause**: Tool results were not being added to session message history during execution
- **Solution**: Modified `_execute_tool_calls()` in CLI to explicitly add `role="tool"` messages with execution metadata
- **Impact**: Tool results now immediately available to assistant during conversation, consistent behavior between live and serialized sessions

#### Common Tools Defensive Programming
- **Problem**: `list_files` and `search_files` tools failed with type errors when `head_limit` parameter was passed as string
- **Root Cause**: LLM-generated tool calls sometimes provided numeric parameters as strings
- **Solution**: Added defensive type conversion with fallback to default values on `ValueError`
- **Impact**: Improved tool reliability and error handling

### Enhanced

#### Comprehensive Session Management System
- **Session Serialization**: Complete session state preservation including provider, model, parameters, system prompt, tool registry, and conversation history
- **Optional Analytics**: Added `generate_summary()`, `generate_assessment()`, and `extract_facts()` methods for session-level insights
- **Versioned Schema**: Implemented `session-archive/v1` format with JSON schema validation in `abstractcore/assets/session_schema.json`
- **CLI Integration**: Added `/save <file> [--summary] [--assessment] [--facts]` and `/load <file>` commands with optional analytics generation
- **Backward Compatibility**: Graceful handling of legacy session formats during load operations

#### Enhanced CLI User Experience
- **Improved Help System**: Comprehensive, aesthetically pleasing help text with detailed command documentation and usage examples
- **Tool Integration**: Added `search_files` tool to CLI with full documentation and status reporting
- **Better Banner**: Informative startup banner with quick commands and available tools overview
- **Parameter Documentation**: Clear documentation of `/save` command options and usage patterns

#### Metadata System Redesign
- **Extensible Metadata**: Moved `name` field into `metadata` dictionary for better extensibility
- **Location Support**: Added `location` property backed by `metadata['location']` for geographical context
- **Property-Based Access**: Clean API with `message.name` and `message.location` properties while maintaining metadata flexibility
- **Backward Compatibility**: Automatic migration of legacy `name` field to `metadata['name']` during deserialization

### Technical

#### Files Modified
- `abstractcore/providers/ollama_provider.py`: Fixed endpoint selection logic to use `/api/chat` by default
- `abstractcore/core/session.py`: Enhanced serialization, standardized parameter naming, added analytics methods
- `abstractcore/core/types.py`: Redesigned metadata system with property-based access
- `abstractcore/utils/cli.py`: Improved help system, added tool integration, enhanced save/load commands
- `abstractcore/tools/common_tools.py`: Added defensive programming for parameter type handling
- `abstractcore/assets/session_schema.json`: Created comprehensive JSON schema for session validation
- `docs/session.md`: New documentation explaining session management and serialization benefits

#### Test Results
✅ Ollama responses now concise (15 chars vs 977+ chars previously)  
✅ Session serialization preserves complete state including analytics  
✅ Tool execution results properly integrated into live chat history  
✅ Parameter consistency across all session methods  
✅ Defensive tool parameter handling prevents type errors  
✅ Backward compatibility maintained for existing session files

## [2.3.0] - 2025-10-12

### Major Changes

#### Server Simplification and Enhancement
- Simplified server implementation in `abstractcore/server/app.py` (reduced from ~4000 to ~1500 lines)
- Removed complex model discovery in favor of direct provider queries
- Added comprehensive endpoint documentation with OpenAI-style descriptions
- Enhanced request/response models with detailed parameter descriptions and examples

#### Multi-Provider Embedding Support
- `EmbeddingManager` now supports three providers: HuggingFace, Ollama, and LMStudio
- Unified embedding API across all providers with automatic format conversion
- Provider-specific caching for isolation and performance
- Backward compatible with existing HuggingFace-only code (default provider)

#### Tool Call Syntax Rewriting
- Added `syntax_rewriter.py` for server-side tool call format conversion
- Supports multiple formats: OpenAI, Codex, Qwen3, LLaMA3, Gemma, XML
- Automatic format detection based on headers, user-agent, and model name
- Enables seamless integration with agentic CLIs (Codex, Crush, Gemini CLI)

#### Model Discovery and Filtering
- Added `/v1/models?type=text-embedding` endpoint for filtering embedding models
- Heuristic-based model type detection (embedding vs text-generation)
- Embedding patterns: "embed", "all-minilm", "bert-", "-bert", "bge-", "gte-", etc.
- Provider-specific model filtering via query parameters

### Server Enhancements

#### API Endpoints
- Enhanced `/v1/embeddings` endpoint with multi-provider support
- Added `type` parameter to `/v1/models` for model type filtering (text-generation/text-embedding)
- Improved `/v1/chat/completions` with comprehensive parameter documentation
- Added `/{provider}/v1/chat/completions` for provider-specific requests
- Enhanced `/v1/responses` endpoint for agentic CLI compatibility
- Updated `/providers` endpoint with detailed provider information

#### Request/Response Models
- Added detailed field descriptions and examples to all Pydantic models
- `EmbeddingRequest`: Comprehensive parameter explanations using OpenAI reference style
- `ChatCompletionRequest`: Enhanced with field-level documentation and examples
- `ChatMessage`: Detailed role and content descriptions with use cases
- Default examples updated to use working models

#### Format Conversion
- Automatic tool call format conversion for different agentic CLIs
- Support for custom tool call tags via `agent_format` parameter
- Configurable tool execution (server-side vs client-side)
- Environment variable configuration for default formats

### Core Library Improvements

#### Embeddings
- Provider parameter added to `EmbeddingManager.__init__()` (default: "huggingface")
- `embed()` and `embed_batch()` methods now delegate to provider-specific implementations
- Ollama provider: Added `embed()` method using `/api/embeddings` endpoint
- LMStudio provider: Added `embed()` method using `/v1/embeddings` endpoint
- Cache naming includes provider for proper isolation

#### Providers
- Enhanced provider base classes with improved error handling
- Better streaming support across all providers
- Consistent timeout handling and retry logic
- Improved tool call detection and parsing

#### Exception Handling
- Added `UnsupportedProviderError` for better error messages
- Enhanced exception types for embedding-specific errors
- Improved error context and debugging information

### Documentation Overhaul

#### Consolidated Documentation
- Merged `common-mistakes.md` into `troubleshooting.md` with cross-references
- Merged `server-api-reference.md` into simplified `server.md` (1006 → 479 lines)
- Created comprehensive `docs/README.md` as navigation hub
- Removed redundant documentation files (8 files consolidated)

#### New Documentation
- Created `tool-syntax-rewriting.md` covering both tag and syntax rewriters
- Enhanced `embeddings.md` with multi-provider support and examples
- Updated `architecture.md` with server architecture and present-tense language
- Improved `getting-started.md` with comprehensive tool documentation

#### Documentation Organization
- Moved `basic-*.md` files to `docs/apps/` subdirectory
- Created `docs/archive/` for superseded documentation
- Added `docs/archive/README.md` explaining archived content
- Updated all cross-references across documentation

#### Documentation Style
- Removed historical/refactoring language ("replaced", "improved", "before/after")
- Converted all documentation to present tense
- Focused on current capabilities and actionable content
- Simplified language for clarity and accessibility

#### Root README Updates
- Added clearer distinction between core library and optional server
- Enhanced documentation section with better organization
- Added "Architecture & Advanced" section
- Improved Quick Links with comprehensive navigation

### Technical Improvements

#### Code Quality
- Removed unused `simple_model_discovery.py` module
- Cleaned up temporary debug files and scripts
- Removed integration.py tool module (functionality moved to providers)
- Better separation of concerns between core and server

#### Testing
- Added comprehensive tests for embedding providers
- Enhanced server endpoint testing
- Improved tool call syntax rewriting tests
- Better test coverage for multi-provider scenarios

### Breaking Changes
None. All changes are backward compatible with version 2.2.x.

### Migration Guide

#### For Embedding Users
If you were using embeddings, no changes needed. The default behavior remains HuggingFace.

To use other providers:
```python
from abstractcore.embeddings import EmbeddingManager

# HuggingFace (default, unchanged)
embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")

# Ollama (new)
embedder = EmbeddingManager(model="granite-embedding:278m", provider="ollama")

# LMStudio (new)
embedder = EmbeddingManager(model="text-embedding-all-minilm-l6-v2-embedding", provider="lmstudio")
```

#### For Server Users
Server API endpoints remain compatible. New features:
- Use `?type=text-embedding` to filter embedding models
- Use `agent_format` parameter for custom tool call formats
- Environment variables for default configuration

#### For Documentation Users
- Use `docs/server.md` instead of `server-api-reference.md`
- Use `docs/troubleshooting.md` for all troubleshooting (includes common mistakes)
- Use `docs/README.md` as navigation hub
- Reference `prerequisites.md` instead of deleted `providers.md`

## [2.2.4] - 2025-10-10

### Fixed
- **ONNX Optimization and Warning Management**: Improved embedding performance and user experience
  - **Smart ONNX Model Selection**: EmbeddingManager now automatically selects optimized `model_O3.onnx` for better performance
  - **Warning Suppression**: Eliminated harmless warnings from PyTorch 2.8+ and sentence-transformers during model loading
  - **Graceful Fallbacks**: Multiple fallback layers ensure reliability (optimized ONNX → basic ONNX → PyTorch)
  - **Performance Improvement**: ONNX optimization provides significant speedup for batch embedding operations
  - **Clean Implementation**: Conservative approach with minimal code changes (40 lines) for maintainability

### Technical
- Added `_suppress_onnx_warnings()` context manager to handle known harmless warnings
- Added `_get_optimal_onnx_model()` function for intelligent ONNX variant selection
- Enhanced `_load_model()` with multi-layer fallback strategy and clear logging
- Zero breaking changes - all improvements are additive with sensible defaults

## [2.2.3] - 2025-10-10

### Fixed
- **Installation Package [all] Extra**: Fixed `pip install abstractcore[all]` to truly install ALL modules
  - **Issue**: The `[all]` extra was missing development dependencies (dev, test, docs)
  - **Solution**: Updated `[all]` extra to include complete dependency set (12 total extras)
  - **Coverage**: Now includes all providers, features, and development tools
    - **All Providers** (6): openai, anthropic, ollama, lmstudio, huggingface, mlx
    - **All Features** (3): embeddings, processing, server
    - **All Development** (3): dev, test, docs
  - **Impact**: Users can now confidently use `abstractcore[all]` for complete installation without missing dependencies

### Technical
- **Comprehensive Installation**: `pip install abstractcore[all]` now installs 12 dependency groups
- **Development Ready**: Includes all testing frameworks (pytest-cov, responses), code tools (black, mypy, ruff), and documentation tools (mkdocs)
- **Verified Configuration**: All referenced extras exist and are properly defined with no circular dependencies

## [2.2.2] - 2025-10-10

### Added
- **LLM-as-a-Judge**: Production-ready objective evaluation with structured assessments
  - **BasicJudge** class for critical assessment with constructive skepticism
  - **Multiple file support** with sequential processing to avoid context overflow
  - **Global assessment synthesis** for multi-file evaluations (appears first, followed by individual file results)
  - **Enhanced assessment structure** with judge summary, source reference, and optional criteria details
  - **9 evaluation criteria**: clarity, simplicity, actionability, soundness, innovation, effectiveness, relevance, completeness, coherence
  - **CLI with simple command**: `judge file1.py file2.py --context="code review"` (console script entry point)
  - **Flexible output formats**: JSON, plain text, YAML with structured scoring (1-5 scale)
  - **Optional global assessment control**: `--exclude-global` flag for original list behavior

### Enhanced
- **Built-in Applications**: BasicJudge added to production-ready application suite
  - **Structured output integration** with Pydantic validation and FeedbackRetry for validation error recovery
  - **Chain-of-thought reasoning** for transparent evaluation with low temperature (0.1) for consistency
  - **Custom criteria support** and reference-based evaluation for specialized assessment needs
  - **Comprehensive error handling** with graceful fallbacks and detailed diagnostics

### Documentation
- **Complete BasicJudge documentation**: Enhanced `docs/basic-judge.md` with API reference, examples, and best practices
  - **Real-world examples**: Code review, documentation assessment, academic writing evaluation, multiple file scenarios
  - **CLI parameter documentation** with practical usage patterns and advanced options
  - **Global assessment examples** showing synthesis of multiple file evaluations
- **Updated README.md**: Added BasicJudge to built-in applications with 30-second examples
- **Internal CLI integration**: Added `/judge` command for conversation quality evaluation with detailed feedback

### Technical
- **Context overflow prevention**: Optimized global assessment prompts to work within model context limits
- **Production-grade architecture**: Proper Pydantic integration, sequential file processing, backward compatibility
- **Console script integration**: Simple `judge` command available after package installation (matches `extractor`, `summarizer`)
- **Full backward compatibility**: All existing functionality preserved, optional features clearly marked

## [2.2.1] - 2025-10-10

### Enhanced
- **Timeout Configuration**: Unified timeout management across all components
  - Updated default HTTP timeout from 180s to 300s (5 minutes) for better reliability with large models
  - All providers now consistently inherit timeout from base configuration
  - Server endpoints updated to use unified 5-minute default
  - Improved handling of large language models (36B+ parameters) that require longer processing time

- **Extractor CLI Improvements**: Enhanced command-line interface for knowledge graph extraction
  - Added `--timeout` parameter with proper validation (30s minimum, 2 hours maximum)
  - Users can now configure timeout for large documents and models: `--timeout 3600` for 60 minutes
  - Improved error messages for timeout validation
  - Better support for processing large documents with resource-intensive models

### Fixed
- **BasicExtractor JSON-LD Consistency**: Resolved structural inconsistencies in knowledge graph output
  - Fixed JSON-LD reference normalization where some providers generated string references instead of proper object format
  - Corrected refinement prompt to match initial extraction format exactly (`@type: "s:Relationship"` vs `@type: "r:provides"`)
  - Added missing `s:name` and `strength` fields in relationship refinement
  - All providers now generate consistent, properly structured JSON-LD output

- **Cross-Provider Compatibility**: Improved extraction reliability across different LLM providers
  - LMStudio models now generate proper JSON-LD object references through automatic normalization
  - Reduced warning noise by converting normalization messages to debug level
  - Enhanced iterative refinement to follow exact same structure rules as initial extraction

### Technical
- **Centralized Timeout Management**: All timeout configuration now emanates from `base.py`
  - Providers inherit timeout via `self._timeout` from BaseProvider class
  - Factory system properly propagates timeout parameters through `**kwargs`
  - No hardcoded timeout values remain in provider implementations
  - Consistent 300-second default across HTTP clients, tool execution, and embeddings

### Documentation
- **Updated Model References**: Modernized documentation to use current recommended models
  - Updated `docs/getting-started.md` to use `qwen3:4b-instruct-2507-q4_K_M` (default) and `qwen3-coder:30b` (premium)
  - Replaced outdated `qwen2.5-coder:7b` references throughout getting started guide
  - Added proper cross-references to reorganized documentation (`server.md`, `internal-cli.md`)
  - Enhanced "What's Next?" section with links to universal API server and CLI documentation

- **Cross-Reference Validation**: Verified all documentation links and anchors
  - Confirmed `docs/prerequisites.md` section anchors match README.md references
  - Validated provider setup links point to correct sections (#openai-setup, #anthropic-setup, etc.)
  - Ensured consistent documentation structure across all guides

## Previous Versions

Previous version history is available in the git commit log.