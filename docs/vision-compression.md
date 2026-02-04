# Vision Compression User Guide

> ⚠️ **EXPERIMENTAL FEATURE**
> Vision compression and glyph compression are currently in experimental status. The API and behavior may change in future releases.
>
> **Vision Model Requirement**: These features **ONLY work with vision-capable models** (e.g., gpt-4o, claude-3-5-sonnet, llama3.2-vision, gemini-1.5-pro).
> Attempting to use compression with non-vision models will raise `UnsupportedFeatureError`.

## Overview

AbstractCore's Vision Compression system transforms long text documents into visual representations for vision-capable models. This can reduce token usage for long inputs, but compression ratios and quality vary significantly by content, model, and configuration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Compression Methods](#compression-methods)
3. [Provider Optimization](#provider-optimization)
4. [Advanced Usage](#advanced-usage)
5. [Analytics and Monitoring](#analytics-and-monitoring)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

## Quick Start

### Installation

```bash
# Glyph compression (Pillow renderer)
pip install "abstractcore[compression]"
```

Optional (experimental): Direct PDF→image conversion requires `pdf2image` and its system dependencies (Poppler).

### Basic Glyph Compression

The simplest way to compress text using vision-based compression:

```python
from abstractcore.compression import GlyphProcessor
from abstractcore.compression.config import GlyphConfig

# Initialize processor
config = GlyphConfig()
config.enabled = True
processor = GlyphProcessor(config=config)

# Compress text
text = "Your long document text here..."
compressed = processor.process_text(
    text,
    provider="openai",
    model="gpt-4o"
)

# Use with LLM
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "Summarize this document",
    media=compressed
)
```

**What to expect**:
- Compression ratio and quality depend heavily on content, model OCR behavior, and rendering settings.
- Treat this as a tuning problem: higher DPI/font sizes typically improve fidelity but increase image tokens.

### PDF Compression

For PDF documents, you can use the (experimental) DirectPDFProcessor:

```python
from abstractcore.media.processors.direct_pdf_processor import DirectPDFProcessor

# Requires: `pip install pdf2image` (+ Poppler installed on your system)
processor = DirectPDFProcessor(pages_per_image=2, dpi=150)
result = processor.process_file("document.pdf")
if not result.success:
    raise RuntimeError(result.error_message)

# MediaContent (base64 PNG) you can pass to a vision-capable model:
media_image = result.media_content
```

## Compression Methods

### 1. Standard Glyph Compression (Recommended)

**Use Case:** General document compression with reliable quality

```python
from abstractcore.compression import GlyphProcessor

processor = GlyphProcessor()
compressed = processor.process_text(
    text,
    provider="openai",
    model="gpt-4o",
    user_preference="auto"  # or "always" to force compression
)
```

**Characteristics:**
- Compression/quality tradeoffs vary by content, model, and rendering settings
- No external infrastructure beyond a vision-capable model
- Latency depends on rendering + model inference

### 2. Optimized Glyph Compression

**Use Case:** Provider-specific optimization for better compression

```python
from abstractcore.compression.optimizer import CompressionOptimizer
from abstractcore.compression import GlyphProcessor

# Get optimized configuration
optimizer = CompressionOptimizer()
config = optimizer.get_optimized_config(
    provider="openai",
    model="gpt-4o",
    aggressive=False  # Set True for more compression
)

# Use optimized config
processor = GlyphProcessor()
compressed = processor.process_text(text, provider="openai", model="gpt-4o")
```

**Characteristics:**
- More aggressive provider-specific rendering defaults (when configured)
- Compression/quality tradeoffs vary by content, model, and rendering settings

### 3. Hybrid Compression (Experimental)

**Use Case:** Maximum compression for large documents

```python
from abstractcore.compression.vision_compressor import HybridCompressionPipeline
from abstractcore import create_llm

# Initialize hybrid pipeline
pipeline = HybridCompressionPipeline(
    vision_provider="ollama",
    vision_model="llama3.2-vision"
)

# Compress with target ratio
result = pipeline.compress(
    text,
    target_ratio=20.0,  # Target 20x compression
    min_quality=0.90    # Minimum 90% quality
)

print(f"Achieved: {result['total_compression_ratio']:.1f}x compression")
print(f"Quality: {result['total_quality_score']:.1%}")

# Use the compressed images with an LLM
llm = create_llm("ollama", model="llama3.2-vision")
response = llm.generate(
    "Summarize this document",
    media=result['media']  # Access the compressed images via result['media']
)
```

**Characteristics:**
- Compression/quality tradeoffs vary by mode and model
- Requires a vision-capable model
- Latency depends on rendering + model inference

## Provider Optimization

### Pre-configured Profiles

AbstractCore includes provider/model rendering profiles as starting points. They are heuristics (not guarantees) and may need tuning for your content and target vision model.

Examples:
- OpenAI: gpt-4o, gpt-4o-mini
- Anthropic: claude-3-5-sonnet, claude-haiku-4-5
- Ollama: llama3.2-vision

### Using Provider Profiles

```python
from abstractcore.compression.optimizer import create_optimized_config
from abstractcore.compression import GlyphProcessor

# Automatic provider optimization
config = create_optimized_config("openai", "gpt-4o")
processor = GlyphProcessor()

# Process with optimized settings
compressed = processor.process_text(
    text,
    provider="openai",
    model="gpt-4o"
)
```

### Custom Profiles

Create custom optimization profiles:

```python
from abstractcore.compression.optimizer import OptimizationProfile

custom_profile = OptimizationProfile(
    provider="custom",
    model="custom-model",
    dpi=72,              # Lower = more compression
    font_size=6,         # Smaller = more compression
    line_height=7,       # Tighter = more compression
    columns=6,           # More = more compression
    margin_x=2,
    margin_y=2,
    target_compression=5.0,
    quality_threshold=0.85,
    notes="Ultra-aggressive compression"
)

# Convert to rendering config
config = custom_profile.to_rendering_config()
```

## Advanced Usage

### Adaptive Compression

Automatically select compression level based on document characteristics:

```python
from abstractcore.compression.vision_compressor import VisionCompressor

compressor = VisionCompressor()

# Adaptive compression to meet targets
result = compressor.adaptive_compress(
    glyph_images=compressed_images,
    original_tokens=25000,
    target_ratio=20.0,   # Aim for 20x
    min_quality=0.85     # But maintain 85% quality
)

print(f"Selected mode: {result.metadata['mode']}")
print(f"Achieved: {result.compression_ratio:.1f}x at {result.quality_score:.1%} quality")
```

### Quality Control

Configure quality thresholds and validation:

```python
from abstractcore.compression.config import GlyphConfig

config = GlyphConfig()
config.quality_threshold = 0.95  # Require 95% quality
config.min_token_threshold = 1000  # Only compress if >1000 tokens
config.target_compression_ratio = 4.0  # Target 4x compression

processor = GlyphProcessor(config=config)

# Quality validation happens automatically
try:
    compressed = processor.process_text(text, provider="openai", model="gpt-4o")
except CompressionQualityError as e:
    print(f"Quality too low: {e.quality_score:.1%}")
    # Fall back to uncompressed text
```

### Caching

Enable caching for repeated compressions:

```python
from abstractcore.compression.cache import CompressionCache

# Cache is enabled by default in GlyphProcessor
processor = GlyphProcessor()

# First compression (slow)
compressed1 = processor.process_text(text, provider="openai", model="gpt-4o")

# Second compression (cached, instant)
compressed2 = processor.process_text(text, provider="openai", model="gpt-4o")

# Check cache statistics
stats = processor.get_compression_stats()
print(f"Cache hits: {stats['cache_stats']['hits']}")
```

## Analytics and Monitoring

### Track Compression Performance

```python
from abstractcore.compression.analytics import get_analytics

analytics = get_analytics()

# Record compression operation
analytics.record_compression(
    provider="openai",
    model="gpt-4o",
    original_tokens=25000,
    compressed_tokens=7500,
    quality_score=0.92,
    processing_time=1.5,
    method="glyph"
)

# Get provider statistics
stats = analytics.get_provider_stats("openai")
print(f"Average compression: {stats['avg_compression_ratio']:.1f}x")
print(f"Average quality: {stats['avg_quality_score']:.1%}")

# Generate report
report = analytics.generate_report()
print(report)
```

### Monitor Trends

```python
# Get compression trends
trends = analytics.get_trends(hours=24)
print(f"Compression trend: {trends['ratio_trend']}")
print(f"Quality trend: {trends['quality_trend']}")

# Get optimization suggestions
suggestions = analytics.get_optimization_suggestions()
for suggestion in suggestions:
    print(f"- {suggestion}")
```

### Benchmark Configurations

```python
from abstractcore.compression.optimizer import CompressionOptimizer

optimizer = CompressionOptimizer()
test_text = "Sample text for benchmarking..."

# Benchmark a profile
profile = optimizer.profiles["openai/gpt-4o"]
results = optimizer.benchmark_profile(profile, test_text)

print(f"Compression: {results['compression_ratio']:.1f}x")
print(f"Quality: {results['quality_score']:.1%}")
print(f"Time: {results['processing_time']:.2f}s")
print(f"Meets target: {results['meets_target']}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Low Compression Ratio

**Problem:** Getting less than 2x compression

**Solutions:**
```python
# Use aggressive mode
config = create_optimized_config(provider, model, aggressive=True)

# Lower quality threshold
config.quality_threshold = 0.85

# Increase columns and reduce font
config.columns = 6
config.font_size = 6
```

#### 2. Quality Too Low

**Problem:** Text becomes unreadable or quality score <90%

**Solutions:**
```python
# Use conservative settings
config.dpi = 96  # Higher DPI
config.font_size = 9  # Larger font
config.columns = 3  # Fewer columns

# Force higher quality threshold
config.quality_threshold = 0.95
```

#### 3. Processing Too Slow

**Problem:** Compression takes >5 seconds

**Solutions:**
```python
# Enable caching
processor = GlyphProcessor()  # Cache enabled by default

# Reduce image count
config.pages_per_image = 3  # More pages per image

# Use simpler renderer
config.auto_crop = False  # Skip auto-cropping
```

#### 4. UnsupportedFeatureError with glyph_compression

**Problem:** `UnsupportedFeatureError: Glyph compression requires a vision-capable model`

**Cause:** Attempting to use `glyph_compression="always"` with a non-vision model

**Solution:**
```python
from abstractcore import create_llm

# WRONG: Non-vision model with forced compression
llm = create_llm("openai", model="gpt-4")  # No vision support
response = llm.generate(
    "Summarize",
    media=["doc.txt"],
    glyph_compression="always"  # Raises UnsupportedFeatureError!
)

# RIGHT: Use a vision-capable model
llm = create_llm("openai", model="gpt-4o")  # Has vision support ✓
response = llm.generate(
    "Summarize",
    media=["doc.txt"],
    glyph_compression="always"  # Works!
)

# ALTERNATIVE: Use auto mode (graceful fallback)
llm = create_llm("openai", model="gpt-4")
response = llm.generate(
    "Summarize",
    media=["doc.txt"]
    # glyph_compression="auto" is default
    # Logs warning and falls back to text processing
)
```

**Vision-Capable Models:**
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4-vision-preview`
- Anthropic: `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- Ollama: `llama3.2-vision`, `llava`, `bakllava`, `moondream`
- Google: `gemini-1.5-pro`, `gemini-1.5-flash`

#### 5. Provider Doesn't Support Vision

**Problem:** Provider rejects compressed images

**Solution:**
```python
# Check provider capabilities first
from abstractcore.media.capabilities import get_model_capabilities

capabilities = get_model_capabilities(provider, model)
if capabilities.get('vision_support'):
    # Safe to use compression
    compressed = processor.process_text(text, provider, model)
else:
    # Use text directly
    response = llm.generate(text)
```

## API Reference

### GlyphProcessor

```python
class GlyphProcessor:
    def __init__(self, config: Optional[GlyphConfig] = None)

    def process_text(
        self,
        content: str,
        provider: str = None,
        model: str = None,
        user_preference: str = "auto"
    ) -> List[MediaContent]

    def can_process(
        self,
        content: str,
        provider: str,
        model: str
    ) -> bool

    def get_compression_stats(self) -> Dict[str, Any]
```

### CompressionOptimizer

```python
class CompressionOptimizer:
    def get_optimized_config(
        self,
        provider: str,
        model: str,
        aggressive: bool = False
    ) -> RenderingConfig

    def analyze_compression_potential(
        self,
        text_length: int,
        provider: str,
        model: str
    ) -> Dict[str, Any]

    def benchmark_profile(
        self,
        profile: OptimizationProfile,
        test_text: str
    ) -> Dict[str, Any]
```

### HybridCompressionPipeline

```python
class HybridCompressionPipeline:
    def __init__(
        self,
        vision_provider: str = "ollama",
        vision_model: str = "llama3.2-vision"
    )

    def compress(
        self,
        text: str,
        target_ratio: float = 30.0,
        min_quality: float = 0.85
    ) -> Dict[str, Any]
    # Returns dict with:
    #   - media: List[MediaContent] - The compressed images to use with LLM
    #   - total_compression_ratio: float - Achieved compression ratio
    #   - total_quality_score: float - Quality score (0-1)
    #   - original_tokens: int - Original token count
    #   - final_tokens: int - Compressed token count
```

### CompressionAnalytics

```python
class CompressionAnalytics:
    def record_compression(...) -> CompressionMetrics
    def get_provider_stats(provider: str) -> Dict[str, Any]
    def get_trends(hours: int = 24) -> Dict[str, Any]
    def get_optimization_suggestions() -> List[str]
    def generate_report() -> str
```

## Best Practices

### 1. Choose the Right Method

- **< 1,000 tokens**: Don't compress (overhead not worth it)
- **1,000-10,000 tokens**: Standard Glyph (2.8-3.5x)
- **10,000-100,000 tokens**: Optimized Glyph (3.5-4.5x)
- **> 100,000 tokens**: Consider hybrid (5-15x)

### 2. Balance Quality and Compression

```python
# For critical documents
config.quality_threshold = 0.95
config.target_compression_ratio = 3.0

# For general use
config.quality_threshold = 0.90
config.target_compression_ratio = 4.0

# For archives
config.quality_threshold = 0.85
config.target_compression_ratio = 5.0
```

### 3. Use Provider-Specific Optimization

Always use the optimized profile for your provider:

```python
# Good
config = create_optimized_config("openai", "gpt-4o")

# Better (for more compression)
config = create_optimized_config("openai", "gpt-4o", aggressive=True)
```

### 4. Monitor and Improve

```python
# Track performance
analytics = get_analytics()

# Review weekly
report = analytics.generate_report()

# Apply suggestions
suggestions = analytics.get_optimization_suggestions()
```

### 5. Handle Failures Gracefully

```python
try:
    compressed = processor.process_text(text, provider, model)
    response = llm.generate(prompt, media=compressed)
except CompressionQualityError:
    # Fall back to uncompressed
    response = llm.generate(prompt + "\n\n" + text)
except Exception as e:
    logger.error(f"Compression failed: {e}")
    # Use alternative approach
```

## Limitations

### Realistic Expectations

- **Maximum practical compression**: 10-15x with good quality
- **Processing overhead**: 1-5 seconds depending on document size
- **Quality tradeoff**: Higher compression = lower quality
- **Provider dependency**: Requires vision-capable models

### Not Suitable For

- Real-time chat (latency too high)
- Short messages (<1000 tokens)
- Mission-critical accuracy requirements
- Providers without vision support

---

*For technical details and research background, see [Vision Compression Reality Report](report/vision-compression-reality.md)*
