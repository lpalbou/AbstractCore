# Media Handling System

AbstractCore provides a **unified media handling system** with **automatic maximum resolution optimization** that enables consistent file attachment and processing across all providers and models. Upload images, documents, and other media files using the same simple API, and let AbstractCore automatically optimize image resolution for each model's maximum capability.

## Key Benefits

- **Universal API**: Same code works across all providers
- **Maximum Resolution Optimization**: Automatically uses each model's highest supported resolution
- **Automatic Processing**: Intelligent file type detection and optimization
- **Provider Adaptation**: Automatic formatting for each provider's requirements
- **Vision Model Support**: Seamless integration with vision-capable models
- **Document Processing**: Advanced PDF and Office document extraction
- **Error Handling**: Graceful fallbacks and clear error messages

## Quick Start

```python
from abstractcore import create_llm

# Works with any provider - just change the provider name
llm = create_llm("openai", model="gpt-4o", api_key="your-key")
response = llm.generate(
    "What's in this image and document?",
    media=["photo.jpg", "report.pdf"]
)
print(response.content)

# Same code works with Anthropic
llm = create_llm("anthropic", model="claude-3.5-sonnet", api_key="your-key")
response = llm.generate(
    "Analyze these materials",
    media=["chart.png", "data.csv", "presentation.ppt"]
)

# Or with local models
llm = create_llm("ollama", model="qwen2.5vl:7b")
response = llm.generate(
    "Describe this image",
    media=["screenshot.png"]
)
```

## Supported File Types

### Images (Vision Models)
- **Formats**: PNG, JPEG, GIF, WEBP, BMP, TIFF
- **Automatic**: Optimization, resizing, format conversion
- **Features**: EXIF handling, quality optimization for vision models

### Documents
- **Text**: TXT, MD, CSV, TSV, JSON
- **PDF**: Advanced extraction with PyMuPDF4LLM (SOTA 2025)
- **Office**: DOCX, XLSX, PPT (with unstructured library)

### Processing Features
- **Intelligent Detection**: Automatic file type recognition
- **Content Optimization**: Format-specific processing for best LLM results
- **Maximum Resolution Optimization**: Automatically uses each model's highest supported resolution
- **Memory Efficient**: Streaming processing for large files
- **Error Recovery**: Graceful handling of corrupted or unsupported files

## Provider Compatibility

### Vision-Enabled Providers

| Provider | Vision Models | Image Support | Document Support |
|----------|---------------|---------------|------------------|
| **OpenAI** | GPT-4o, GPT-4 Turbo with Vision | ✅ Multi-image | ✅ All formats |
| **Anthropic** | Claude 3.5 Sonnet, Claude 4 series | ✅ Up to 20 images | ✅ All formats |
| **Ollama** | qwen2.5vl:7b, gemma3:4b, llama3.2-vision:11b | ✅ Single image | ✅ All formats |
| **LMStudio** | qwen2.5-vl-7b, gemma-3n-e4b, magistral-small-2509 | ✅ Multiple images | ✅ All formats |

### Text-Only Providers

All providers support document processing even without vision capabilities:

| Provider | Document Processing | Text Extraction |
|----------|-------------------|-----------------|
| **HuggingFace** | ✅ All formats | ✅ Embedded in prompt |
| **MLX** | ✅ All formats | ✅ Embedded in prompt |
| **Any Provider** | ✅ Automatic fallback | ✅ Text extraction |

### ⚠️ Model Compatibility Notes (Updated: 2025-10-17)

Some newer vision models may not be immediately available due to rapid development:

**LMStudio Limitations:**
- `qwen3-vl` models (8B, 30B) - Not yet supported in LMStudio
- Use `qwen2.5-vl-7b` as a proven alternative

**HuggingFace Limitations:**
- `Qwen3-VL` models - Require newer transformers architecture
- Install latest transformers: `pip install --upgrade transformers`
- Or use bleeding edge: `pip install git+https://github.com/huggingface/transformers.git`

**Recommended Stable Models (2025-10-17):**
- **LMStudio**: `qwen/qwen2.5-vl-7b`, `google/gemma-3n-e4b`, `mistralai/magistral-small-2509`
- **Ollama**: `qwen2.5vl:7b`, `gemma3:4b`, `llama3.2-vision:11b`
- **OpenAI**: `gpt-4o`, `gpt-4-turbo-with-vision`
- **Anthropic**: `claude-3.5-sonnet`, `claude-4-series`

## Usage Examples

### Vision Analysis

```python
from abstractcore import create_llm

# Analyze images with any vision model
llm = create_llm("openai", model="gpt-4o")

# Single image analysis
response = llm.generate(
    "What's happening in this image?",
    media=["photo.jpg"]
)

# Multiple images comparison
response = llm.generate(
    "Compare these two charts and explain the trends",
    media=["chart1.png", "chart2.png"]
)

# Mixed media analysis
response = llm.generate(
    "Summarize the report and relate it to what you see in the image",
    media=["financial_report.pdf", "stock_chart.png"]
)
```

### Document Processing

```python
# PDF analysis
response = llm.generate(
    "Summarize the key findings from this research paper",
    media=["research_paper.pdf"]
)

# Office document processing
response = llm.generate(
    "Create a summary of this presentation and spreadsheet",
    media=["quarterly_results.ppt", "financial_data.xlsx"]
)

# CSV data analysis
response = llm.generate(
    "What patterns do you see in this sales data?",
    media=["sales_data.csv"]
)
```

### Cross-Provider Consistency

```python
# Same media processing works across all providers
media_files = ["report.pdf", "chart.png", "data.xlsx"]
prompt = "Analyze these business documents and provide insights"

# OpenAI
openai_llm = create_llm("openai", model="gpt-4o")
openai_response = openai_llm.generate(prompt, media=media_files)

# Anthropic
anthropic_llm = create_llm("anthropic", model="claude-3.5-sonnet")
anthropic_response = anthropic_llm.generate(prompt, media=media_files)

# Local model
ollama_llm = create_llm("ollama", model="qwen2.5vl:7b")
ollama_response = ollama_llm.generate(prompt, media=media_files)

# All three work identically!
```

### Streaming with Media

```python
# Real-time streaming responses with media
llm = create_llm("anthropic", model="claude-3.5-sonnet")

for chunk in llm.generate(
    "Describe this image in detail",
    media=["complex_diagram.png"],
    stream=True
):
    print(chunk.content, end="", flush=True)
```

## Advanced Features

### Maximum Resolution Optimization (NEW)

AbstractCore automatically optimizes image resolution for each model's maximum capability, ensuring the best possible vision results:

```python
from abstractcore import create_llm

# Images are automatically optimized for each model's maximum resolution
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "Analyze this image in detail",
    media=["photo.jpg"]  # Auto-resized to 4096x4096 for GPT-4o
)

# Different model, different optimization
llm = create_llm("ollama", model="qwen2.5vl:7b")
response = llm.generate(
    "What's in this image?",
    media=["photo.jpg"]  # Auto-resized to 3584x3584 for qwen2.5vl
)
```

**Model-Specific Resolution Limits:**
- **GPT-4o**: Up to 4096x4096 pixels
- **Claude 3.5 Sonnet**: Up to 1568x1568 pixels
- **qwen2.5vl:7b**: Up to 3584x3584 pixels
- **gemma3:4b**: Up to 896x896 pixels
- **llama3.2-vision:11b**: Up to 560x560 pixels

**Benefits:**
- **Better Accuracy**: Higher resolution means more detail for the model to analyze
- **Automatic**: No manual configuration required
- **Provider-Aware**: Adapts to each provider's optimal settings
- **Quality Optimization**: Increased JPEG quality (90%) for better compression

### Capability Detection

The system automatically detects model capabilities and adapts accordingly:

```python
from abstractcore.media.capabilities import is_vision_model, supports_images

# Check if a model supports vision
if is_vision_model("gpt-4o"):
    print("This model can process images")

if supports_images("claude-3.5-sonnet"):
    print("This model supports image analysis")

# Current: Automatic fallback for non-vision models
llm = create_llm("openai", model="gpt-4")  # Non-vision model
response = llm.generate(
    "Analyze this image",
    media=["photo.jpg"]  # Currently: Will extract basic metadata instead
)
```

### Vision Fallback System (IMPLEMENTED ✅)

AbstractCore now includes an **automatic vision fallback system** that enables text-only models to process images using a transparent two-stage pipeline:

#### How Vision Fallback Works

When you use a text-only model with images, AbstractCore automatically:

1. **Detects Model Limitations**: Identifies when a text-only model receives an image
2. **Uses Vision Fallback**: Employs a configured vision model to analyze the image
3. **Provides Description**: Passes the image description to the text-only model
4. **Returns Results**: User gets complete image analysis without knowing about the two-stage process

#### Automatic Setup

```python
from abstractcore import create_llm

# Text-only model with image - triggers helpful warnings
llm = create_llm("lmstudio", model="qwen/qwen3-next-80b")  # No vision support
response = llm.generate("What's in this image?", media=["photo.jpg"])

# User sees helpful guidance in logs:
# 🔸 EASIEST: Download BLIP vision model (990MB): abstractcore --download-vision-model
# 🔸 Use existing Ollama model: abstractcore --set-vision-caption qwen2.5vl:7b
# 🔸 Use cloud API: abstractcore --set-vision-provider openai --model gpt-4o
```

#### One-Command Setup

```bash
# Download and configure BLIP model automatically
abstractcore --download-vision-model

# Alternative: Use existing Ollama model
abstractcore --set-vision-caption qwen2.5vl:7b

# Alternative: Use cloud vision API
abstractcore --set-vision-provider openai --model gpt-4o
```

#### Working Example

```python
from abstractcore import create_llm

# After running: abstractcore --set-vision-caption qwen2.5vl:7b

# Text-only model now works seamlessly with images
llm = create_llm("lmstudio", model="qwen/qwen3-next-80b")
response = llm.generate("What's in this image?", media=["whale_photo.jpg"])

print(response.content)
# Output: "The image shows a whale leaping or breaching out of the water.
# This dramatic moment captures the whale in mid-air, often with spray and
# water cascading around its body, highlighting its immense size and power.
# Such behavior, known as 'breaching,' is commonly observed in species like
# humpback whales and is thought to serve purposes such as communication,
# play, or removing parasites..."
```

#### Behind the Scenes

What actually happens (transparent to user):
1. **Stage 1**: `qwen2.5vl:7b` (vision model) analyzes `whale_photo.jpg` → detailed description
2. **Stage 2**: `qwen/qwen3-next-80b` (text-only) processes description + user question → final analysis

#### Configuration Commands

```bash
# Check current status
abstractcore --status

# Download models (automatic setup)
abstractcore --download-vision-model              # BLIP base (990MB)
abstractcore --download-vision-model vit-gpt2     # ViT-GPT2 (500MB, CPU-friendly)
abstractcore --download-vision-model git-base     # GIT base (400MB, smallest)

# Use existing provider models
abstractcore --set-vision-caption qwen2.5vl:7b
abstractcore --set-vision-caption llama3.2-vision:11b

# Cloud APIs
abstractcore --set-vision-provider openai --model gpt-4o
abstractcore --set-vision-provider anthropic --model claude-3.5-sonnet

# Interactive setup
abstractcore --configure

# Advanced: Fallback chains
abstractcore --add-vision-fallback ollama qwen2.5vl:7b
abstractcore --add-vision-fallback openai gpt-4o
```

#### Benefits of Vision Fallback

- **Universal Compatibility**: Any text-only model can now process images
- **Cost Optimization**: Use cheaper text models for reasoning, vision models only for description
- **Transparent Operation**: Users don't need to change their code
- **Flexible Configuration**: Local models, cloud APIs, or hybrid setups
- **Offline-First**: Works without internet after downloading local models
- **Automatic Fallback**: Graceful degradation when vision not configured

#### Supported Vision Models

**Local Models (Downloaded):**
- **BLIP Base**: 990MB, excellent quality, CPU/GPU compatible
- **ViT-GPT2**: 500MB, CPU-friendly, good performance
- **GIT Base**: 400MB, smallest size, basic quality

**Provider Models:**
- **Ollama**: `qwen2.5vl:7b`, `llama3.2-vision:11b`, `gemma3:4b`
- **LMStudio**: `qwen/qwen2.5-vl-7b`, `google/gemma-3n-e4b`
- **OpenAI**: `gpt-4o`, `gpt-4-turbo-with-vision`
- **Anthropic**: `claude-3.5-sonnet`, `claude-4-series`

### Custom Processing Options

```python
# Advanced image processing
from abstractcore.media.processors import ImageProcessor

processor = ImageProcessor(
    optimize_for_vision=True,
    max_dimension=1024,
    quality=85
)

# Advanced PDF processing
from abstractcore.media.processors import PDFProcessor

pdf_processor = PDFProcessor(
    extract_images=True,
    markdown_output=True,
    preserve_tables=True
)
```

### Direct Media Processing

```python
# Process files directly (without LLM)
from abstractcore.media import process_file

# Process any supported file
result = process_file("document.pdf")
if result.success:
    print(f"Content: {result.media_content.content}")
    print(f"Type: {result.media_content.media_type}")
    print(f"Metadata: {result.media_content.metadata}")
```

## Best Practices

### File Size and Limits

```python
# Check model-specific limits
from abstractcore.media.capabilities import get_media_capabilities

caps = get_media_capabilities("gpt-4o")
print(f"Max images per message: {caps.max_images}")
print(f"Supported formats: {caps.supported_formats}")
```

### Error Handling

```python
try:
    response = llm.generate(
        "Analyze this file",
        media=["large_document.pdf"]
    )
except Exception as e:
    print(f"Media processing error: {e}")
    # Fallback to text-only processing
    response = llm.generate("Analyze the uploaded document content")
```

### Performance Tips

```python
# For large documents, consider chunking
from abstractcore.media.processors import PDFProcessor

processor = PDFProcessor(chunk_size=8000)  # Process in chunks

# For multiple images, process in batches
image_files = ["img1.jpg", "img2.jpg", "img3.jpg"]
for batch in [image_files[i:i+3] for i in range(0, len(image_files), 3)]:
    response = llm.generate("Analyze these images", media=batch)
```

## Model-Specific Examples

### OpenAI GPT-4o

```python
# Multi-image analysis with high detail
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "Compare these architectural photos and identify the styles",
    media=["building1.jpg", "building2.jpg", "building3.jpg"]
)
```

### Anthropic Claude 3.5 Sonnet

```python
# Document analysis with specialized prompts
llm = create_llm("anthropic", model="claude-3.5-sonnet")
response = llm.generate(
    "Provide a comprehensive analysis of this research paper",
    media=["academic_paper.pdf"]
)
```

### Local Vision Models

```python
# Ollama with qwen2.5-vl
ollama_llm = create_llm("ollama", model="qwen2.5vl:7b")
response = ollama_llm.generate(
    "What objects do you see in this image?",
    media=["scene.jpg"]
)

# LMStudio with qwen2.5-vl
lmstudio_llm = create_llm("lmstudio", model="qwen/qwen2.5-vl-7b")
response = lmstudio_llm.generate(
    "Describe this chart and its trends",
    media=["business_chart.png"]
)

# Ollama with Llama 3.2 Vision
llama_llm = create_llm("ollama", model="llama3.2-vision:11b")
response = llama_llm.generate(
    "Analyze this document layout",
    media=["document.jpg"]
)
```

## Installation

### Basic Installation

```bash
# Core media handling (images, text, basic documents)
pip install abstractcore[media]
```

### Full Installation

```bash
# All media features including PDF and Office documents
pip install abstractcore[all]

# Individual optional dependencies
pip install pymupdf4llm        # For advanced PDF processing
pip install unstructured       # For Office documents (DOCX, XLSX, PPT)
pip install pillow            # For image processing
pip install pandas            # For CSV/data processing
```

## Troubleshooting

### Common Issues

**Media not processed:**
```python
# Check if media dependencies are installed
try:
    response = llm.generate("Test", media=["test.jpg"])
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install abstractcore[media]")
```

**Vision model not detecting images:**
```python
# Verify model capabilities
from abstractcore.media.capabilities import is_vision_model

if not is_vision_model("your-model"):
    print("This model doesn't support vision")
    print("Try: gpt-4o, claude-3.5-sonnet, qwen2.5vl:7b, or llama3.2-vision:11b")
```

**Large file processing:**
```python
# For large files, check size limits
import os
file_size = os.path.getsize("large_file.pdf")
if file_size > 10 * 1024 * 1024:  # 10MB
    print("File may be too large for some providers")
```

### Validation

```bash
# Test your installation
python validate_media_system.py

# Run comprehensive tests
python -m pytest tests/media_handling/ -v
```

## API Reference

### Core Functions

```python
# Main generation with media
llm.generate(prompt, media=files, **kwargs)

# Direct file processing
from abstractcore.media import process_file
result = process_file(file_path)

# Capability detection
from abstractcore.media.capabilities import (
    is_vision_model,
    supports_images,
    get_media_capabilities
)
```

### Media Types

```python
from abstractcore.media.types import MediaType, ContentFormat

# MediaType.IMAGE, MediaType.DOCUMENT, MediaType.TEXT
# ContentFormat.BASE64, ContentFormat.TEXT, ContentFormat.BINARY
```

### Processors

```python
from abstractcore.media.processors import (
    ImageProcessor,    # Images with PIL
    TextProcessor,     # Text, CSV, JSON with pandas
    PDFProcessor,      # PDFs with PyMuPDF4LLM
    OfficeProcessor    # DOCX, XLSX, PPT with unstructured
)
```

## Next Steps

- **[Getting Started Guide](getting-started.md)** - Complete AbstractCore tutorial
- **[API Reference](api-reference.md)** - Full Python API documentation
- **[Vision Examples](../examples/vision/)** - More vision model examples
- **[Document Processing Examples](../examples/documents/)** - Document analysis examples

---

The media handling system makes AbstractCore truly multimodal while maintaining the same "write once, run everywhere" philosophy. Focus on your application logic while AbstractCore handles the complexity of different provider APIs and media formats.