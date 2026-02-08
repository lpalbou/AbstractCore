# Media Processing Module

## Purpose

The media module provides unified, provider-agnostic multimodal processing capabilities for AbstractCore, enabling seamless handling of images, PDFs, Office documents, text files, and other media types across different LLM providers.

**Core Philosophy**: One interface, multiple media types, automatic provider formatting.

## Quick Reference

### Media Processor Selection

| File Type | Processor | Best For | Requirements |
|-----------|-----------|----------|--------------|
| Images (jpg, png) | ImageProcessor | Vision models, analysis | PIL/Pillow |
| PDF (text-heavy) | PDFProcessor | Text extraction | PyMuPDF4LLM |
| PDF (math/tables) | DirectPDFProcessor | Visual fidelity | pdf2image |
| Office docs | OfficeProcessor | DOCX, XLSX, PPTX | unstructured |
| Text/CSV/JSON | TextProcessor | Data files | Built-in |
| Any file | AutoMediaHandler | Automatic routing | All optional |

### Format Support Matrix

| Format | Extensions | Size Limit | Provider Support |
|--------|-----------|------------|------------------|
| Images | png, jpg, gif, webp, bmp, tiff, ico | 10MB | OpenAI, Anthropic, Ollama (vision models) |
| Documents | pdf, docx, xlsx, pptx, odt, rtf | 10MB | All (text extraction) |
| Text | **90+ extensions** (see below) | 10MB | All (with content detection fallback) |
| Data | csv, tsv, json, jsonl, xml, yaml, toml | 10MB | All (text rendering) |
| Audio | mp3, wav, m4a, ogg, flac, aac | 10MB | Not yet implemented |
| Video | mp4, avi, mov, mkv, webm, wmv | 10MB | Not yet implemented |

**Text File Support (90+ Extensions)**:
- **Programming Languages**: py, js, java, c, cpp, go, rs, rb, php, r, R, sql, jl, lua, dart, swift, kt, scala, etc.
- **Notebooks**: ipynb, rmd, Rmd, qmd
- **Configuration**: yaml, yml, toml, ini, cfg, conf, env, properties
- **Markup**: md, markdown, rst, tex, html, adoc, org
- **Build/Scripts**: sh, bash, zsh, dockerfile, cmake, gradle, makefile
- **Web**: css, scss, sass, less, vue, svelte, jsx, tsx
- **Logs**: log, out, err
- **Unknown text extensions**: Automatically detected via content analysis

### Programmatic Access to Supported Formats

Query supported file extensions programmatically:

```python
from abstractcore.media.types import (
    get_all_supported_extensions,
    get_supported_extensions_by_type,
    MediaType
)

# Get all formats organized by type
all_formats = get_all_supported_extensions()
print(f"Text extensions: {len(all_formats['text'])}")  # 90+
print(f"Image extensions: {len(all_formats['image'])}")  # 9
print(f"Document extensions: {len(all_formats['document'])}")  # 9

# Get formats for specific type
text_extensions = get_supported_extensions_by_type(MediaType.TEXT)
print('r' in text_extensions)  # True - R scripts supported
print('ipynb' in text_extensions)  # True - Jupyter notebooks supported

# Get formats from handler
from abstractcore.media.auto_handler import AutoMediaHandler
handler = AutoMediaHandler()
formats = handler.get_supported_formats()
# Returns: {'text': [...90+ extensions], 'image': [...], 'document': [...]}
```

**Example Script**: See [examples/list_supported_formats.py](../../examples/list_supported_formats.py) for complete demonstration.

### Provider-Specific Features

| Provider | Vision | Max Images | Image Size | Notes |
|----------|--------|-----------|------------|-------|
| OpenAI | ✅ | 10 | 20MB | Detail levels: low/high/auto |
| Anthropic | ✅ | 20 | 5MB | ~1600 tokens per image |
| Ollama | ⚠️ | Varies | 10MB | Model-dependent (llava, qwen2.5vl) |
| LMStudio | ⚠️ | Varies | 10MB | Model-dependent |
| HuggingFace | ✅ | Varies | 10MB | Vision models only |
| MLX | ❌ | - | - | Text-only |

## Common Tasks

- **How do I process an image?** → See [Processing Images](#processing-images)
- **How do I extract text from PDF?** → See [Processing PDFs](#processing-pdfs)
- **How do I handle Office documents?** → See [Processing Office Documents](#processing-office-documents)
- **How do I use vision fallback?** → See [Vision Fallback](#vision-fallback-text-only-models)
- **How do I optimize image size?** → See [Custom Resolution Scaling](#custom-resolution-scaling)
- **How do I process multiple files?** → See [Batch Processing](#batch-processing)
- **What formats are supported?** → See [Format Support Matrix](#format-support-matrix) above
- **How do I handle errors?** → See [Error Handling](#error-handling)

## Architecture Position

**Layer**: Infrastructure layer (between core and providers)

**Dependencies**:
- `abstractcore.core` - Base exceptions, factory patterns
- `abstractcore.architectures` - Model capabilities detection
- `abstractcore.config` - Configuration management
- `abstractcore.compression` - Optional Glyph compression integration
- `abstractcore.events` - Event emission for telemetry
- `abstractcore.utils` - Structured logging

**Used By**:
- `abstractcore.providers.*` - All provider implementations for media handling
- `abstractcore.core.factory` - Media integration in unified LLM interface
- User applications - Direct access to processors for custom workflows

## Component Structure

```
media/
├── base.py                      # Core abstractions (MediaContent, BaseMediaHandler)
├── types.py                     # Type definitions (MediaType, ContentFormat, enums)
├── capabilities.py              # Model capability detection and validation
├── auto_handler.py              # Automatic processor selection
├── vision_fallback.py           # Two-stage vision pipeline for text-only models
│
├── processors/                  # Media type processors
│   ├── image_processor.py       # Image processing with PIL (resize, optimize)
│   ├── pdf_processor.py         # PDF extraction with PyMuPDF4LLM (markdown)
│   ├── direct_pdf_processor.py  # Direct PDF→image conversion (Glyph)
│   ├── glyph_pdf_processor.py   # Glyph-optimized PDF extraction (math/tables)
│   ├── text_processor.py        # Text/CSV/JSON/Markdown processing
│   └── office_processor.py      # Office docs with unstructured library
│
├── handlers/                    # Provider-specific formatters
│   ├── openai_handler.py        # OpenAI API format (image_url)
│   ├── anthropic_handler.py     # Anthropic API format (base64 source)
│   └── local_handler.py         # Local providers (Ollama, MLX, LMStudio)
│
└── utils/                       # Utilities
    └── image_scaler.py          # Model-optimized image scaling
```

## Detailed Components

### Root Files

#### 1. `base.py` - Core Abstractions

**Key Classes**:
- **`MediaContent`**: Core data structure representing processed media
  - `media_type`: IMAGE, DOCUMENT, AUDIO, VIDEO, TEXT
  - `content`: str | bytes (actual content)
  - `content_format`: BASE64, URL, FILE_PATH, TEXT, BINARY
  - `mime_type`: MIME type string
  - `metadata`: Dict with processing info

- **`BaseMediaHandler`**: Abstract base for all processors
  - `process_file(file_path)`: Main entry point with telemetry
  - `process_multiple_files()`: Batch processing
  - `supports_media_type()`: Capability check
  - `_process_internal()`: Abstract method for implementation

- **`BaseProviderMediaHandler`**: Base for provider-specific handlers
  - `format_for_provider()`: Convert MediaContent to provider format
  - `can_handle_media()`: Validate provider compatibility

**Example**:
```python
from abstractcore.media.base import MediaContent
from abstractcore.media.types import MediaType, ContentFormat

# Create media content
content = MediaContent(
    media_type=MediaType.IMAGE,
    content="base64encodeddata...",
    content_format=ContentFormat.BASE64,
    mime_type="image/png",
    metadata={'width': 800, 'height': 600}
)
```

#### 2. `types.py` - Type Definitions

**Enums**:
- **`MediaType`**: IMAGE, DOCUMENT, AUDIO, VIDEO, TEXT
- **`ContentFormat`**: BASE64, URL, FILE_PATH, TEXT, BINARY, AUTO

**Data Classes**:
- **`MediaCapabilities`**: Provider/model capabilities
  - Vision, audio, video, document support flags
  - Format lists, size limits, resolution constraints
  - `supports_media_type()`: Check support
  - `supports_format()`: Check format compatibility

- **`MediaProcessingResult`**: Processing outcome
  - `success`: bool
  - `media_content`: Optional[MediaContent]
  - `error_message`: Optional[str]
  - `processing_time`: float

**Helpers**:
- `detect_media_type(file_path)`: Auto-detect from extension
- `create_media_content(file_path)`: Quick MediaContent creation

#### 3. `capabilities.py` - Model Capability Detection

**Purpose**: Integrates with `model_capabilities.json` to provide comprehensive media capability information.

**Key Function**:
```python
from abstractcore.media.capabilities import get_media_capabilities

caps = get_media_capabilities("gpt-4o", provider="openai")
# Returns: MediaCapabilities with:
#   - vision_support=True
#   - max_images_per_message=10
#   - supported_image_formats=['png', 'jpeg', 'jpg', 'gif', 'webp']
#   - max_image_size_bytes=20MB
```

**Provider-Specific Adjustments**:
- **OpenAI**: 20MB limit, 10 images for GPT-4o
- **Anthropic**: 5MB limit, 20 images for Claude
- **Local (Ollama/MLX)**: 10MB limit, text embedding preferred

#### 4. `auto_handler.py` - Automatic Processor Selection

**Purpose**: Intelligently routes files to appropriate processors based on type and available dependencies.

**Features**:
- Lazy processor initialization
- Dependency checking (PIL, PyMuPDF4LLM, unstructured)
- Glyph compression integration
- Fallback processing

**Example**:
```python
from abstractcore.media.auto_handler import AutoMediaHandler

handler = AutoMediaHandler(
    max_file_size=50*1024*1024,
    enable_glyph_compression=True
)

result = handler.process_file("document.pdf")
if result.success:
    print(result.media_content.content)  # Markdown text
```

**Processor Selection Logic**:
```python
.jpg/.png → ImageProcessor (if PIL available)
.pdf → PDFProcessor (if PyMuPDF4LLM) else TextProcessor
.docx/.xlsx/.pptx → OfficeProcessor (if unstructured) else TextProcessor
.txt/.md/.csv → TextProcessor (always available)
```

#### 5. `vision_fallback.py` - Two-Stage Vision Pipeline

**Purpose**: Enables text-only models to process images using a two-stage pipeline: vision model generates description, text model processes description.

**Configuration**:
```bash
# Download local vision model (990MB BLIP)
abstractcore --download-vision-model

# Configure a provider vision model
abstractcore --set-vision-provider ollama qwen2.5vl:7b

# Use cloud API
abstractcore --set-vision-provider openai gpt-4o
```

**Usage**:
```python
from abstractcore.media.vision_fallback import VisionFallbackHandler

handler = VisionFallbackHandler()
description = handler.create_description(
    "image.jpg",
    user_prompt="What's in this image?"
)
# Description used by text-only model
```

**Fallback Chain**:
1. Primary configured provider/model
2. Fallback chain providers
3. Local vision models (BLIP, ViT-GPT2, GIT)
4. Error if all fail

### Processors

#### 6. `image_processor.py` - Image Processing

**Dependencies**: PIL/Pillow

**Features**:
- Auto-rotation based on EXIF data
- Model-optimized resolution scaling (uses `model_capabilities.json`)
- Format conversion (JPEG, PNG, WebP)
- Quality optimization
- RGBA→RGB conversion with white background

**Example**:
```python
from abstractcore.media.processors import ImageProcessor

processor = ImageProcessor(
    max_resolution=(4096, 4096),
    quality=90,
    auto_rotate=True,
    prefer_max_resolution=True  # Use model's max resolution
)

result = processor.process_file("photo.jpg", model_name="gpt-4o")
# Returns: MediaContent with base64-encoded optimized image
```

**Model-Specific Optimization**:
```python
# Qwen3-VL: Variable resolution, 32x32 pixel grouping
# Qwen2.5-VL: Up to 3584x3584, 28x28 pixel grouping
# Gemma3: Fixed 896x896, 16x16 patches
# Claude: Up to 1568x1568
# GPT-4o: Tile-based, dynamic resolution
```

#### 7. `pdf_processor.py` - PDF Extraction

**Dependencies**: PyMuPDF4LLM, PyMuPDF

**Features**:
- LLM-optimized markdown output
- Table structure preservation
- Image extraction (optional)
- Metadata extraction (title, author, dates)
- Page range selection

**Example**:
```python
from abstractcore.media.processors import PDFProcessor

processor = PDFProcessor(
    extract_images=False,
    preserve_tables=True,
    markdown_output=True
)

result = processor.process_file("document.pdf")
print(result.media_content.content)  # Clean markdown
```

**Output Format**:
```markdown
# Page 1

## Section Title

Regular text content...

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

#### 8. `direct_pdf_processor.py` - Direct PDF→Image Conversion

**Dependencies**: pdf2image, PIL

**Purpose**: Converts PDF pages directly to images for Glyph compression, preserving ALL visual elements (formulas, tables, images).

**Features**:
- Multi-page layouts (2 pages per image)
- Horizontal/vertical layouts
- Configurable DPI
- Cached in Glyph directory

**Example**:
```python
from abstractcore.media.processors.direct_pdf_processor import DirectPDFProcessor

processor = DirectPDFProcessor(
    pages_per_image=2,  # 16 pages → 8 images
    dpi=150,
    layout='horizontal',  # Side-by-side like open book
    gap=20  # pixels between pages
)

result = processor.process_file("research.pdf")
# Returns: First combined image (metadata has all image paths)
```

**Use Case**: Perfect for mathematical papers, technical documents where visual fidelity is critical.

#### 9. `glyph_pdf_processor.py` - Glyph-Optimized PDF Extraction

**Dependencies**: PyMuPDF

**Purpose**: Extracts PDF content while preserving compact mathematical notation and table layouts for optimal Glyph visual compression.

**Features**:
- Mathematical symbol preservation (α, ∑, ∫ kept as-is)
- Table detection and compact formatting
- Whitespace compaction
- Position-aware text block processing

**Example**:
```python
from abstractcore.media.processors.glyph_pdf_processor import GlyphPDFProcessor

processor = GlyphPDFProcessor(
    preserve_math_notation=True,
    preserve_table_layout=True,
    compact_whitespace=True
)

result = processor.process_file("equations.pdf")
# Output: Compact text with symbols preserved
```

#### 10. `text_processor.py` - Text File Processing

**Dependencies**: None (standard library), pandas (optional for CSV)

**Supported Formats**:
- Plain text (.txt)
- Markdown (.md)
- CSV/TSV (.csv, .tsv)
- JSON (.json)
- XML/HTML (.xml, .html)

**Features**:
- Encoding auto-detection (UTF-8, Latin-1, CP1252)
- CSV structure analysis with pandas
- JSON structure extraction
- Markdown header detection

**Example**:
```python
from abstractcore.media.processors import TextProcessor

processor = TextProcessor(
    encoding='utf-8',
    preserve_structure=True
)

# CSV file
result = processor.process_file("data.csv")
# Returns: Structured summary with columns, sample rows

# JSON file
result = processor.process_file("config.json")
# Returns: Pretty-printed JSON with structure info
```

#### 11. `office_processor.py` - Office Document Processing

**Dependencies**: unstructured library

**Supported Formats**:
- Word documents (.docx)
- Excel spreadsheets (.xlsx)
- PowerPoint presentations (.pptx)

**Features**:
- Element-based extraction (Title, Table, Image, Text)
- Table preservation
- Markdown output
- Metadata extraction (author, dates)

**Example**:
```python
from abstractcore.media.processors import OfficeProcessor

processor = OfficeProcessor(
    extract_tables=True,
    markdown_output=True,
    include_metadata=True
)

# Word document
result = processor.process_file("report.docx")

# Excel spreadsheet
result = processor.process_file("data.xlsx")
# Returns: Sheet-by-sheet breakdown

# PowerPoint
result = processor.process_file("slides.pptx")
# Returns: Slide-by-slide content
```

### Handlers

#### 12. `openai_handler.py` - OpenAI Formatter

**Purpose**: Format media for OpenAI API (GPT-4o, GPT-4 Vision)

**Format**:
```python
# Image
{
    "type": "image_url",
    "image_url": {
        "url": "data:image/png;base64,iVBOR...",
        "detail": "high"  # low, high, auto
    }
}

# Text
{
    "type": "text",
    "text": "Content here..."
}
```

**Special Features**:
- Qwen model auto-adjustment (detail level for context limits)
- Token estimation (tile-based for images)
- Detail level configuration

**Example**:
```python
from abstractcore.media.handlers import OpenAIMediaHandler

handler = OpenAIMediaHandler(
    model_capabilities={"vision_support": True},
    model_name="gpt-4o"
)

message = handler.create_multimodal_message(
    text="Analyze this image",
    media_contents=[image_content]
)
```

#### 13. `anthropic_handler.py` - Anthropic Formatter

**Purpose**: Format media for Anthropic API (Claude 3.x, 3.5)

**Format**:
```python
# Image
{
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBOR..."  # Raw base64, no data URL
    }
}

# Text
{
    "type": "text",
    "text": "Content here..."
}
```

**Special Features**:
- Up to 20 images per message
- 5MB per image limit
- Document analysis prompt templates
- Token estimation (~1600 per image)

**Example**:
```python
from abstractcore.media.handlers import AnthropicMediaHandler

handler = AnthropicMediaHandler(
    model_capabilities={"vision_support": True},
    max_images_per_message=20
)

# Specialized document analysis
prompt = handler.create_document_analysis_prompt(
    media_contents=[pdf_content, image_content],
    analysis_type="summary"  # general, summary, extract, qa
)
```

#### 14. `local_handler.py` - Local Provider Formatter

**Purpose**: Format media for local providers (Ollama, MLX, LMStudio)

**Providers**:
- **Ollama**: Simple `{"role": "user", "content": "...", "images": ["base64"]}`
- **MLX**: Apple Silicon optimizations, tensor conversion support
- **LMStudio**: OpenAI-compatible format with limitations

**Special Features**:
- Vision capability detection using `supports_vision()`
- Automatic vision fallback for text-only models
- Text embedding preference for non-vision models
- Intelligent routing (structured vs text-embedded)

**Example**:
```python
from abstractcore.media.handlers import LocalMediaHandler

# Ollama with vision model
handler = LocalMediaHandler(
    provider_name="ollama",
    model_name="llava:13b",
    model_capabilities={"vision_support": True}
)

message = handler.create_multimodal_message(
    text="What's in this image?",
    media_contents=[image_content]
)
# Returns: {"role": "user", "content": "...", "images": ["..."]}

# Ollama with text-only model
handler = LocalMediaHandler(
    provider_name="ollama",
    model_name="llama3.2:3b",  # No vision support
    model_capabilities={"vision_support": False}
)

message = handler.create_multimodal_message(
    text="What's in this image?",
    media_contents=[image_content]
)
# Returns: String with vision fallback description
```

### Utils

#### 15. `image_scaler.py` - Model-Optimized Scaling

**Purpose**: Intelligent image scaling based on model-specific requirements from `model_capabilities.json`.

**Scaling Modes**:
- **FIT**: Scale to fit within bounds, maintain aspect ratio
- **FILL**: Scale to fill bounds, may crop
- **STRETCH**: Stretch to exact size, may distort
- **PAD**: Scale to fit, pad with background
- **CROP_CENTER**: Scale to fill, crop from center

**Model-Specific Strategies**:
```python
from abstractcore.media.utils.image_scaler import scale_image_for_model, ScalingMode

# Qwen3-VL: Variable resolution with patch alignment
scaled = scale_image_for_model(
    image="photo.jpg",
    model_name="qwen3-vl:7b",
    scaling_mode=ScalingMode.FIT
)

# Gemma3: Fixed 896x896 with padding
scaled = scale_image_for_model(
    image="photo.jpg",
    model_name="gemma3-27b-vision",
    scaling_mode=ScalingMode.PAD
)
```

**Optimization Features**:
- Patch size alignment (16x16, 14x14, 28x28)
- Pixel grouping respect
- Adaptive windowing support
- Variable vs fixed resolution handling

## Media Processing Pipeline

```
┌─────────────────┐
│   User Code     │
│  create_llm()   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   AutoMediaHandler          │
│  - Detect media type        │
│  - Check dependencies       │
│  - Select processor         │
│  - Apply Glyph compression? │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Processor                 │
│  - Load file                │
│  - Extract/process          │
│  - Optimize                 │
│  - Create MediaContent      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Provider Handler          │
│  - Check capabilities       │
│  - Format for provider      │
│  - Validate compatibility   │
│  - Estimate tokens          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Provider Implementation   │
│  - Send to API/model        │
│  - Receive response         │
│  - Return to user           │
└─────────────────────────────┘
```

## Provider-Specific Formatting

### OpenAI (GPT-4o, GPT-4 Vision)

**Image Format**:
```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,iVBOR...",
    "detail": "high"
  }
}
```

**Detail Levels**:
- `low`: 85 tokens (faster, less detail)
- `high`: 85 + 170×tiles (slower, more detail)
- `auto`: Model decides (default)

**Qwen Model Optimization**:
- SiliconFlow Qwen3-VL: Auto-adjust to `low` for multiple images
- Context limit: 131,072 tokens
- Per-image cost: 256 tokens (low) vs 24,576+ (high)

### Anthropic (Claude 3.x, 3.5)

**Image Format**:
```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/png",
    "data": "iVBOR..."
  }
}
```

**Key Differences**:
- Raw base64 (no data URL prefix)
- Up to 20 images per message
- 5MB per image limit
- ~1600 tokens per image

### Local Providers

**Ollama**:
```json
{
  "role": "user",
  "content": "What's in this image?",
  "images": ["base64string"]
}
```

**LMStudio** (OpenAI-compatible):
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Analyze this"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

**MLX**:
```json
{
  "type": "image_base64",
  "content": "base64string",
  "mime_type": "image/png",
  "metadata": {...}
}
```

## Usage Patterns

### Processing Images

```python
from abstractcore import create_llm

# Automatic (recommended)
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "What's in this image?",
    media=["photo.jpg"]  # Automatic processing + formatting
)

# Manual processing
from abstractcore.media.processors import ImageProcessor
from abstractcore.media.handlers import OpenAIMediaHandler

processor = ImageProcessor(prefer_max_resolution=True)
result = processor.process_file("photo.jpg", model_name="gpt-4o")

handler = OpenAIMediaHandler(model_name="gpt-4o")
formatted = handler.format_for_provider(result.media_content)

# Use formatted in API call
```

### Processing PDFs

```python
from abstractcore import create_llm

# Automatic text extraction
llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")
response = llm.generate(
    "Summarize this document",
    media=["report.pdf"]  # Markdown extraction
)

# With Glyph compression (visual)
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "Analyze this paper",
    media=["research.pdf"],
    glyph_compression="always"  # Direct PDF→image conversion
)

# Manual PDF processing
from abstractcore.media.processors import PDFProcessor

processor = PDFProcessor(
    markdown_output=True,
    preserve_tables=True
)
result = processor.process_file("document.pdf")
print(result.media_content.content)  # Markdown text
```

### Processing Office Documents

```python
from abstractcore import create_llm

# Automatic
llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")

# Word document
response = llm.generate(
    "Extract action items from this report",
    media=["report.docx"]
)

# Excel spreadsheet
response = llm.generate(
    "Analyze this data",
    media=["sales.xlsx"]
)

# PowerPoint
response = llm.generate(
    "Summarize these slides",
    media=["presentation.pptx"]
)
```

### Batch Processing

```python
from abstractcore.media.auto_handler import AutoMediaHandler

handler = AutoMediaHandler()

files = ["image1.jpg", "doc.pdf", "data.xlsx"]
results = handler.process_multiple_files(files)

for result in results:
    if result.success:
        print(f"✓ {result.metadata['file_name']}")
    else:
        print(f"✗ {result.metadata['file_name']}: {result.error_message}")
```

### Vision Fallback (Text-Only Models)

```python
from abstractcore import create_llm

# Configure vision fallback
# Option 1: Download local model
# $ abstractcore --download-vision-model

# Option 2: Use any provider/model (local or cloud)
# $ abstractcore --set-vision-provider ollama qwen2.5vl:7b
# $ abstractcore --set-vision-provider lmstudio qwen/qwen2.5-vl-7b
# $ abstractcore --set-vision-provider openai gpt-4o

# Use text-only model with images
llm = create_llm("ollama", model="llama3.2:3b")  # No vision support
response = llm.generate(
    "What's in this image?",
    media=["photo.jpg"]  # Vision fallback automatically used
)
# Pipeline: qwen2.5vl:7b generates description → llama3.2:3b processes description
```

### Custom Resolution Scaling

```python
from abstractcore.media.processors import ImageProcessor

processor = ImageProcessor(
    max_resolution=(2048, 2048),
    quality=95,
    resize_mode='fit',  # 'fit', 'crop', 'stretch'
    prefer_max_resolution=False  # Use custom instead of model max
)

result = processor.process_file("highres.jpg")
```

## Integration Points

### 1. Provider Integration

All providers inherit media handling:

```python
# abstractcore/providers/openai_provider.py
from abstractcore.media.handlers import OpenAIMediaHandler

class OpenAIProvider(BaseProvider):
    def __init__(self, ...):
        self.media_handler = OpenAIMediaHandler(
            model_capabilities=self.get_capabilities(),
            model_name=model
        )

    def generate(self, prompt, media=None, ...):
        if media:
            # Process media files
            media_contents = []
            for file in media:
                result = self.media_handler.process_file(file)
                media_contents.append(result.media_content)

            # Format for provider
            message = self.media_handler.create_multimodal_message(
                prompt, media_contents
            )
```

### 2. Compression Integration

AutoMediaHandler integrates with Glyph compression:

```python
# In auto_handler.py
if self._should_apply_compression(file_path, media_type, provider, model, glyph_compression):
    # For PDFs: Use DirectPDFProcessor
    if file_path.suffix == '.pdf':
        processor = DirectPDFProcessor(pages_per_image=2, dpi=150)
        combined_images = processor.get_combined_image_paths(file_path)
        # Return as MediaContent

    # For text: Extract → compress with Glyph
    else:
        text = self._extract_text(file_path)
        orchestrator = self._get_compression_orchestrator()
        compressed = orchestrator.compress_content(text, provider, model)
        # Return as MediaContent
```

### 3. Configuration Integration

Vision fallback uses centralized config:

```python
# In vision_fallback.py
from abstractcore.config import get_config_manager

class VisionFallbackHandler:
    def __init__(self):
        self.config_manager = get_config_manager()
        self.vision_config = self.config_manager.config.vision

    def create_description(self, image_path):
        # Use configured vision model
        if self.vision_config.caption_provider:
            llm = create_llm(
                self.vision_config.caption_provider,
                model=self.vision_config.caption_model
            )
            return llm.generate("Describe this image", media=[image_path])
```

### 4. Capabilities Integration

All components use `model_capabilities.json`:

```python
# In capabilities.py
from abstractcore.architectures import get_model_capabilities

caps = MediaCapabilities.from_model_capabilities("gpt-4o", "openai")
# Returns populated capabilities based on JSON data
```

## Best Practices

### Format Selection

**Use Cases**:
- **Images for vision models**: Always use processors (automatic optimization)
- **PDFs for analysis**: Use text extraction (markdown) for most cases
- **PDFs with formulas/tables**: Use Glyph compression (visual fidelity)
- **Office documents**: Use unstructured library (structure preservation)
- **CSV/JSON**: Use text processor (structure analysis)

**Provider Selection**:
- **OpenAI**: Best for high-quality vision analysis, supports detail levels
- **Anthropic**: Best for document analysis (20 images), strong reasoning
- **Local**: Best for privacy, offline operation, cost savings

### Optimization

**Image Optimization**:
```python
# Always prefer model-specific optimization
processor = ImageProcessor(prefer_max_resolution=True)

# For batch processing, use consistent settings
processor = ImageProcessor(
    max_resolution=(2048, 2048),  # Balance quality/speed
    quality=85,  # Good compression
    auto_rotate=True  # Fix orientation
)
```

**PDF Optimization**:
```python
# For text extraction (most cases)
processor = PDFProcessor(
    markdown_output=True,
    preserve_tables=True,
    extract_images=False  # Usually not needed
)

# For visual fidelity (formulas, diagrams)
handler = AutoMediaHandler(enable_glyph_compression=True)
result = handler.process_file("math.pdf", glyph_compression="always")
```

**Memory Management**:
```python
# Process large batches in chunks
from pathlib import Path

files = list(Path("images/").glob("*.jpg"))
batch_size = 10

handler = AutoMediaHandler()
for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    results = handler.process_multiple_files(batch)
    # Process results
    del results  # Free memory
```

### Error Handling

```python
from abstractcore.media.base import MediaProcessingError, UnsupportedMediaTypeError

handler = AutoMediaHandler()

try:
    result = handler.process_file("document.pdf")
    if not result.success:
        print(f"Processing failed: {result.error_message}")
except UnsupportedMediaTypeError as e:
    print(f"Unsupported media type: {e}")
except MediaProcessingError as e:
    print(f"Processing error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

### Validation

```python
from abstractcore.media.handlers import OpenAIMediaHandler

handler = OpenAIMediaHandler(model_name="gpt-4o")

# Validate before processing
if handler.validate_media_for_model(media_content, "gpt-4o"):
    formatted = handler.format_for_provider(media_content)
else:
    print("Media not compatible with model")

# Check token usage
tokens = handler.estimate_tokens_for_media(media_content)
print(f"Estimated tokens: {tokens}")
```

## Common Pitfalls

### 1. File Size Limits

**Problem**: Exceeding provider limits
```python
# ✗ Bad: May exceed limits
llm.generate("Analyze", media=["huge.pdf"])

# ✓ Good: Check size first
from pathlib import Path
file_size = Path("huge.pdf").stat().st_size
max_size = 20 * 1024 * 1024  # 20MB for OpenAI

if file_size > max_size:
    # Use Glyph compression or chunk the file
    handler = AutoMediaHandler(enable_glyph_compression=True)
    result = handler.process_file("huge.pdf")
```

### 2. Format Compatibility

**Problem**: Using incompatible formats
```python
# ✗ Bad: .ico may not be supported
llm.generate("Analyze", media=["icon.ico"])

# ✓ Good: Convert to supported format
from PIL import Image
img = Image.open("icon.ico")
img.save("icon.png")
llm.generate("Analyze", media=["icon.png"])
```

### 3. Vision Model Detection

**Problem**: Assuming vision support without checking
```python
# ✗ Bad: May fail with text-only model
llm = create_llm("ollama", model="llama3.2:3b")
response = llm.generate("What's in this?", media=["image.jpg"])
# Fails if vision fallback not configured

# ✓ Good: Configure vision fallback
# $ abstractcore --download-vision-model
# Then it works automatically
```

### 4. Detail Level for Context Limits

**Problem**: High detail causing context overflow
```python
# ✗ Bad: Multiple images with high detail on Qwen
for img in ["img1.jpg", "img2.jpg", "img3.jpg"]:
    response = llm.generate("Analyze", media=[img])
# May exceed 131,072 token limit with high detail

# ✓ Good: Use low detail for multiple images
# OpenAI handler automatically adjusts for Qwen models
# Or specify explicitly:
from abstractcore.media.types import MediaContent
content = MediaContent(..., metadata={'detail_level': 'low'})
```

### 5. Office Document Dependencies

**Problem**: Missing unstructured library
```python
# ✗ Bad: Fails without unstructured
handler = AutoMediaHandler()
result = handler.process_file("report.docx")
# Falls back to basic text extraction (loses structure)

# ✓ Good: Install media extras
# $ pip install "abstractcore[media]"
# Includes PIL, PyMuPDF4LLM, unstructured
```

### 6. PDF Processing Choice

**Problem**: Using wrong PDF processor
```python
# ✗ Bad: Text extraction for math paper
processor = PDFProcessor()
result = processor.process_file("quantum_physics.pdf")
# Loses mathematical formulas, diagrams

# ✓ Good: Use direct conversion for visual content
processor = DirectPDFProcessor(pages_per_image=2)
result = processor.process_file("quantum_physics.pdf")
# Preserves all visual elements
```

## Testing Strategy

### Unit Tests

```python
# tests/media/test_image_processor.py
def test_image_processing():
    processor = ImageProcessor()
    result = processor.process_file("test_image.jpg")

    assert result.success
    assert result.media_content.media_type == MediaType.IMAGE
    assert result.media_content.content_format == ContentFormat.BASE64

# tests/media/test_auto_handler.py
def test_automatic_processor_selection():
    handler = AutoMediaHandler()

    # Test image
    result = handler.process_file("test.jpg")
    assert "ImageProcessor" in result.metadata['processor']

    # Test PDF
    result = handler.process_file("test.pdf")
    assert "PDFProcessor" in result.metadata['processor']
```

### Integration Tests

```python
# tests/media/test_provider_integration.py
def test_openai_handler_formatting():
    handler = OpenAIMediaHandler(model_name="gpt-4o")

    # Process image
    processor = ImageProcessor()
    result = processor.process_file("test.jpg")

    # Format for OpenAI
    formatted = handler.format_for_provider(result.media_content)

    assert formatted['type'] == 'image_url'
    assert 'data:image' in formatted['image_url']['url']
    assert 'detail' in formatted['image_url']

def test_anthropic_handler_formatting():
    handler = AnthropicMediaHandler()

    processor = ImageProcessor()
    result = processor.process_file("test.jpg")

    formatted = handler.format_for_provider(result.media_content)

    assert formatted['type'] == 'image'
    assert formatted['source']['type'] == 'base64'
    assert 'data:' not in formatted['source']['data']  # Raw base64
```

### End-to-End Tests

```python
# tests/media/test_e2e_media.py
def test_complete_image_pipeline():
    llm = create_llm("openai", model="gpt-4o")

    response = llm.generate(
        "What's in this image?",
        media=["test_image.jpg"]
    )

    assert response.content
    assert len(response.content) > 0

def test_complete_pdf_pipeline():
    llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")

    response = llm.generate(
        "Summarize this document",
        media=["test.pdf"]
    )

    assert response.content
    assert "summary" in response.content.lower() or len(response.content) > 100
```

### Performance Tests

```python
# tests/media/test_performance.py
import time

def test_image_processing_speed():
    processor = ImageProcessor()

    start = time.time()
    result = processor.process_file("large_image.jpg")
    duration = time.time() - start

    assert result.success
    assert duration < 5.0  # Should process in <5 seconds

def test_batch_processing_efficiency():
    handler = AutoMediaHandler()
    files = [f"test_{i}.jpg" for i in range(10)]

    start = time.time()
    results = handler.process_multiple_files(files)
    duration = time.time() - start

    assert all(r.success for r in results)
    assert duration < 30.0  # Should process 10 images in <30 seconds
```

## Public API

**Recommended Imports**:
```python
# Core types
from abstractcore.media.types import (
    MediaType,
    ContentFormat,
    MediaContent,
    MediaCapabilities,
    MediaProcessingResult,
    detect_media_type,
    create_media_content
)

# Processors
from abstractcore.media.processors import (
    ImageProcessor,
    PDFProcessor,
    TextProcessor,
    OfficeProcessor
)

# Auto handler (most common)
from abstractcore.media.auto_handler import AutoMediaHandler

# Provider handlers
from abstractcore.media.handlers import (
    OpenAIMediaHandler,
    AnthropicMediaHandler,
    LocalMediaHandler
)

# Utilities
from abstractcore.media.utils.image_scaler import (
    scale_image_for_model,
    get_optimal_size_for_model,
    ScalingMode
)

# Capabilities
from abstractcore.media.capabilities import (
    get_media_capabilities,
    is_vision_model,
    supports_images,
    get_max_images
)

# Vision fallback
from abstractcore.media.vision_fallback import (
    VisionFallbackHandler,
    has_vision_capability,
    create_image_description
)

# Exceptions
from abstractcore.media.base import (
    MediaProcessingError,
    UnsupportedMediaTypeError,
    FileSizeExceededError
)
```

**Quick Start**:
```python
# Most users should use create_llm() - media handling is automatic
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o")
response = llm.generate("Analyze this", media=["image.jpg", "doc.pdf"])

# Advanced users can access processors directly
from abstractcore.media.auto_handler import AutoMediaHandler

handler = AutoMediaHandler()
result = handler.process_file("document.pdf")
print(result.media_content.content)
```

---

**Module Statistics**:
- **Total Files**: 15 (5 root + 6 processors + 3 handlers + 1 util)
- **Total Lines**: ~7,500
- **Dependencies**: PIL, PyMuPDF4LLM, PyMuPDF, pdf2image, unstructured, pandas (optional)
- **Supported Formats**: Images (8), Documents (3), Text (8), Office (3)
- **Providers**: OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace

## Related Modules

**Direct dependencies**:
- [`core/`](../core/README.md) - Base media handler abstractions
- [`architectures/`](../architectures/README.md) - Model capability detection
- [`assets/`](../assets/README.md) - Model capabilities database
- [`config/`](../config/README.md) - Vision fallback configuration
- [`compression/`](../compression/README.md) - Glyph compression integration
- [`events/`](../events/README.md) - Media processing events
- [`utils/`](../utils/README.md) - VLM token calculation, logging
- [`exceptions/`](../exceptions/README.md) - Media processing errors

**Used by**:
- [`providers/`](../providers/README.md) - All providers for media formatting
- [`processing/`](../processing/README.md) - Document and media processors
- [`apps/`](../apps/README.md) - Multimodal applications
- [`server/`](../server/README.md) - Media upload endpoints

**Integrates with**:
- [`tools/`](../tools/README.md) - Media from web sources
- [`structured/`](../structured/README.md) - Media metadata validation
