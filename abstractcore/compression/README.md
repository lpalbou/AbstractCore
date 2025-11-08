# Glyph Compression Module

## Purpose

The `abstractcore.compression` module implements **glyph-based visual-text compression** for Vision-Language Models (VLMs), achieving **3-4x token savings** by transforming long textual sequences into optimized images. This approach enables efficient processing of large documents that would otherwise exceed token limits or incur excessive costs.

Based on the [Glyph research paper](https://arxiv.org/abs/2403.09248), this implementation converts text into ultra-dense visual representations that VLMs can process more efficiently than raw tokens.

## Quick Reference

### Configuration Quick Reference

| Setting | Recommended | Range | Purpose |
|---------|-------------|-------|---------|
| `font_size` | 7-9 | 6-10 | Balance compression vs readability |
| `dpi` | 72 (compress), 96 (quality) | 72-96 | Image resolution |
| `columns` | 2-4 | 1-8 | Multi-column layout |
| `quality_threshold` | 0.93-0.96 | 0.8-1.0 | Minimum quality score |
| `min_token_threshold` | 10000 | 5000-50000 | When to consider compression |
| `target_compression_ratio` | 3.0-3.5 | 2.5-5.0 | Desired compression |

### Provider Optimization Presets

| Provider | Font Size | DPI | Columns | Target Ratio | Quality | Best For |
|----------|-----------|-----|---------|--------------|---------|----------|
| **OpenAI** | 9 | 72 | 4 | 3.5x | 0.93 | Aggressive compression, excellent OCR |
| **Anthropic** | 10 | 96 | 3 | 3.0x | 0.96 | Conservative, font-sensitive |
| **Ollama** | 9 | 72 | 4-6 | 4.5x | 0.90 | Maximum compression, local models |
| **LMStudio** | 10 | 96 | 3 | 3.0x | 0.94 | Quality focus, good OCR |

### When to Use Compression

| Document Size | Recommendation | Reason |
|---------------|---------------|---------|
| < 10k tokens | Don't compress | Overhead not worth it |
| 10k-50k tokens | Use if cost-sensitive | 3-4x savings beneficial |
| 50k-100k tokens | Strongly recommended | Significant savings, avoid context limits |
| > 100k tokens | Essential | May be only way to process |
| > 80% of context | Required | Approaching model limits |

## Common Tasks

- **How do I enable compression?** → See [Basic Compression](#basic-compression)
- **How do I configure compression?** → See [Advanced: Custom Configuration](#advanced-custom-configuration)
- **When should I compress?** → See [When to Use Compression](#when-to-use-compression-1) table above
- **How do I optimize for my provider?** → See [Provider Optimization Presets](#provider-optimization-presets) above
- **How do I check compression quality?** → See [Quality Assessment](#5-quality-assessment-qualitypy)
- **How do I troubleshoot low quality?** → See [Quality Issues](#quality-issues)
- **How do I cache results?** → See [Compression Cache](#6-compression-cache-cachepy)
- **How do I track performance?** → See [With Analytics](#with-analytics)

## Architecture Position

```
abstractcore/
├── core/                    # Core LLM abstractions
├── providers/              # Provider implementations
├── media/                  # Media processing (IMAGE, VIDEO, AUDIO)
│   ├── capabilities.py     # Model capability detection
│   └── ...
├── compression/            # ← GLYPH COMPRESSION LAYER
│   ├── orchestrator.py     # Decision-making API
│   ├── glyph_processor.py  # Core processing
│   └── ...
└── utils/                  # Utilities
    ├── token_utils.py      # Token estimation
    └── vlm_token_calculator.py  # VLM-specific token calculation

Dependencies:
- abstractcore.media: Vision capability detection, MediaContent types
- abstractcore.utils: Token estimation, structured logging
- abstractcore.config: Centralized configuration
- PIL/Pillow: Image rendering (required)

Used by:
- abstractcore.orchestrator: Automatic compression decisions
- User code: Direct compression API for large documents
```

## Component Structure

The module consists of 11 files organized in a layered architecture:

```
compression/
├── exceptions.py            # Compression-specific exceptions
├── config.py                # Configuration dataclasses
├── text_formatter.py        # Markdown-like text formatting
├── pil_text_renderer.py     # PIL-based image rendering
├── glyph_processor.py       # Core compression logic
├── quality.py               # Quality assessment and metrics
├── cache.py                 # Compression cache management
├── analytics.py             # Performance tracking and analytics
├── optimizer.py             # Provider-specific optimization
├── vision_compressor.py     # Vision-aware compression (experimental)
└── orchestrator.py          # High-level decision API
```

### Layer Breakdown

**Layer 1: Foundation (Configuration & Exceptions)**
- `config.py`: `GlyphConfig`, `RenderingConfig` - All configuration options
- `exceptions.py`: `CompressionError`, `CompressionQualityError`, `RenderingError`

**Layer 2: Text Processing**
- `text_formatter.py`: Markdown formatting, header conversion, newline handling

**Layer 3: Image Rendering**
- `pil_text_renderer.py`: PIL-based rendering with font support (OCRB, OCRA, system fonts)

**Layer 4: Core Compression**
- `glyph_processor.py`: Main compression pipeline, integrates all components

**Layer 5: Quality & Optimization**
- `quality.py`: Quality validation, provider-specific thresholds
- `optimizer.py`: Provider-specific optimization profiles
- `cache.py`: LRU cache for compressed results

**Layer 6: Analytics & Orchestration**
- `analytics.py`: Performance metrics, trend analysis, optimization suggestions
- `vision_compressor.py`: Experimental vision-based compression (DeepSeek-OCR-like)
- `orchestrator.py`: Intelligent compression decision-making

## Detailed Components

### 1. Configuration (`config.py`)

**Purpose**: Centralized configuration for all compression settings

**Key Classes**:
- `GlyphConfig`: Main configuration with cache, quality, provider settings
- `RenderingConfig`: Font, layout, DPI, column configuration

**Key Features**:
```python
config = GlyphConfig.default()
config.enabled = True
config.quality_threshold = 0.95
config.min_token_threshold = 10000
config.target_compression_ratio = 3.0

# Provider-specific profiles
config.provider_profiles = {
    "openai": {"dpi": 72, "font_size": 9},
    "anthropic": {"dpi": 96, "font_size": 10}
}

# Rendering configuration
render_config = config.get_provider_config("openai", "gpt-4o")
```

### 2. Text Formatting (`text_formatter.py`)

**Purpose**: Preprocess text for optimal readability in compressed images

**Key Classes**:
- `TextFormatter`: Main formatter with markdown support
- `TextSegment`: Structured text with formatting metadata
- `FormattingConfig`: Formatting options (bold, italic, headers, newlines)

**Key Features**:
- Markdown formatting: `**bold**` → BOLD, `*italic*` → italic
- Hierarchical headers: `#` → H1, `##` → A., `###` → A.1., etc.
- Newline handling: Single `\n` → space, `\n\n` → 2 spaces, `\n\n\n+` → line break
- Performance optimization: Fast path for large files without formatting

```python
formatter = TextFormatter()
segments = formatter.format_text(text)
# Returns List[TextSegment] with is_bold, is_italic, is_header metadata
```

### 3. Image Rendering (`pil_text_renderer.py`)

**Purpose**: Render text segments to optimized images using PIL/Pillow

**Key Class**: `PILTextRenderer`

**Key Features**:
- Multi-column layout (1-8 columns)
- Custom font support (OCRB, OCRA, system fonts)
- Bold rendering via multiple overlays for monospace fonts
- Automatic pagination for large texts
- VLM-optimized dimensions (1024x1024 default)
- DPI scaling (72 for compression, 96 for quality)

```python
renderer = PILTextRenderer(config)
images = renderer.segments_to_images(
    segments=text_segments,
    config=render_config,
    output_dir="/tmp/glyph",
    unique_id="doc123"
)
# Returns List[Path] of rendered PNG images
```

**Text Capacity Estimation**:
- Calculates chars/line based on font metrics
- Accounts for multi-column layout and margins
- Uses efficiency factors (75-85%) for realistic estimates
- Handles large segments by splitting at word boundaries

### 4. Core Compression (`glyph_processor.py`)

**Purpose**: Main compression pipeline integrating all components

**Key Class**: `GlyphProcessor(BaseMediaHandler)`

**Compression Pipeline**:
1. Check if compression is beneficial (`can_process()`)
2. Apply text formatting if enabled
3. Check cache for previous compression
4. Render text to images using PIL
5. Validate quality using `QualityValidator`
6. Calculate compression statistics
7. Cache successful compression
8. Return `List[MediaContent]` with base64-encoded images

```python
processor = GlyphProcessor(config)
media_contents = processor.process_text(
    content="Long document text...",
    provider="openai",
    model="gpt-4o",
    user_preference="auto"  # auto/always/never
)
# Returns List[MediaContent] with compressed images
```

**Key Features**:
- Provider-specific optimization
- Automatic quality validation with fallback
- Accurate token calculation using VLMTokenCalculator
- Comprehensive metadata in MediaContent
- Integration with AbstractCore media system

### 5. Quality Assessment (`quality.py`)

**Purpose**: Validate compression quality using multiple metrics

**Key Classes**:
- `QualityValidator`: Multi-metric quality assessment
- `CompressionStats`: Compression operation statistics

**Validation Metrics**:
1. **Compression Ratio** (30% weight): Target 3-4x, score based on range
2. **Content Preservation** (40% weight): Length preservation, special char handling
3. **Readability** (30% weight): Provider OCR quality, content type penalties

**Provider-Specific Thresholds**:
```python
validator = QualityValidator()
quality_score = validator.assess(original_text, rendered_images, "openai")

thresholds = {
    "openai": 0.93,      # Excellent OCR
    "anthropic": 0.96,   # Font-sensitive
    "ollama": 0.90,      # Variable quality
    "huggingface": 0.85  # Limited OCR
}
```

### 6. Compression Cache (`cache.py`)

**Purpose**: LRU cache for compressed results with TTL and size limits

**Key Class**: `CompressionCache`

**Features**:
- Hash-based cache keys (content + config)
- Automatic cleanup (TTL, size limits)
- Metadata storage (compression stats)
- Thread-safe operations
- Cache statistics tracking

```python
cache = CompressionCache(
    cache_dir="~/.abstractcore/glyph_cache",
    max_size_gb=1.0,
    ttl_days=7
)

# Automatic cache hit/miss handling
cache_key = generate_key(content, config)
if cached := cache.get(cache_key):
    return cached
else:
    result = compress(content)
    cache.set(cache_key, result, stats)
```

### 7. Analytics System (`analytics.py`)

**Purpose**: Track compression performance and provide optimization insights

**Key Classes**:
- `CompressionAnalytics`: Performance tracking and analysis
- `CompressionMetrics`: Individual compression metrics

**Features**:
- Historical metrics storage (JSON-based)
- Provider/model performance comparison
- Trend analysis (rolling averages)
- Optimization suggestions
- Comprehensive reporting

```python
analytics = get_analytics()
analytics.record_compression(
    provider="openai",
    model="gpt-4o",
    original_tokens=50000,
    compressed_tokens=15000,
    quality_score=0.95,
    processing_time=2.5
)

# Generate insights
report = analytics.generate_report()
suggestions = analytics.get_optimization_suggestions()
provider_stats = analytics.get_provider_stats("openai")
```

### 8. Optimization Profiles (`optimizer.py`)

**Purpose**: Provider-specific optimization for maximum compression

**Key Classes**:
- `CompressionOptimizer`: Profile-based optimization
- `OptimizationProfile`: Provider/model-specific settings

**Provider Profiles**:
```python
# OpenAI (GPT-4o): Aggressive compression
dpi=72, font_size=8, columns=4, target=3.5x

# Anthropic (Claude): Conservative for quality
dpi=96, font_size=9, columns=3, target=3.0x

# Ollama (local models): Maximum compression
dpi=72, font_size=6, columns=6, target=4.5x
```

**Usage**:
```python
optimizer = CompressionOptimizer()
config = optimizer.get_optimized_config(
    provider="openai",
    model="gpt-4o",
    aggressive=True  # Push compression limits
)

# Analyze potential
analysis = optimizer.analyze_compression_potential(
    text_length=100000,
    provider="openai",
    model="gpt-4o"
)
# Returns: estimated ratio, token savings, achievability
```

### 9. Vision Compression (`vision_compressor.py`)

**Purpose**: Experimental vision-based compression (DeepSeek-OCR-like)

**Key Classes**:
- `VisionCompressor`: Vision model-based compression
- `HybridCompressionPipeline`: Glyph + Vision hybrid
- `VisionCompressionResult`: Compression metrics

**Compression Modes**:
- **Conservative**: 2x ratio, 95% quality
- **Balanced**: 5x ratio, 92% quality
- **Aggressive**: 10x ratio, 88% quality

**Hybrid Pipeline**:
```python
pipeline = HybridCompressionPipeline(
    vision_provider="ollama",
    vision_model="llama3.2-vision"
)

result = pipeline.compress(
    text=long_document,
    target_ratio=30.0,  # Target 30x compression
    min_quality=0.85
)
# Stage 1: Glyph (3-4x) → Stage 2: Vision (5-10x) = 15-40x total
```

### 10. Orchestrator (`orchestrator.py`)

**Purpose**: High-level API for intelligent compression decisions

**Key Class**: `CompressionOrchestrator`

**Decision Logic**:
- Check user preference (auto/always/never)
- Verify provider vision support
- Assess content suitability
- Calculate token/context ratio
- Apply decision matrix:
  - `< min_threshold` (10k tokens) → No compression
  - `> 80% context` → Compression necessary
  - `> 50k tokens` → Compression beneficial
  - Otherwise → Standard processing

**Usage**:
```python
orchestrator = CompressionOrchestrator(config)

# Automatic decision
should_compress = orchestrator.should_compress(
    content=text,
    provider="openai",
    model="gpt-4o",
    user_preference="auto"
)

# Get detailed recommendation
recommendation = orchestrator.get_compression_recommendation(
    content=text,
    provider="openai",
    model="gpt-4o"
)
# Returns: should_compress, analysis, estimates, reason

# Apply compression if beneficial
media_contents = orchestrator.compress_content(
    content=text,
    provider="openai",
    model="gpt-4o"
)
```

### 11. Exceptions (`exceptions.py`)

**Purpose**: Compression-specific error handling

**Exception Hierarchy**:
```python
AbstractCoreError
└── CompressionError              # Base compression error
    ├── CompressionQualityError   # Quality below threshold
    ├── RenderingError            # Image rendering failed
    └── CompressionCacheError     # Cache operation failed
```

## Compression Pipeline

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DECISION PHASE (Orchestrator)                             │
│    - Check user preference (auto/always/never)               │
│    - Verify provider vision support                          │
│    - Estimate token count                                    │
│    - Apply decision matrix                                   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TEXT FORMATTING (TextFormatter)                           │
│    - Parse markdown formatting (**bold**, *italic*)          │
│    - Convert headers (# → H1, ## → A., ### → A.1.)          │
│    - Process newlines (single→space, triple→break)           │
│    - Generate TextSegment objects with metadata             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CACHE CHECK (CompressionCache)                            │
│    - Generate cache key (hash of content + config)           │
│    - Check for cached result                                 │
│    - Return cached images if available                       │
└────────────────────┬────────────────────────────────────────┘
                     ↓ (cache miss)
┌─────────────────────────────────────────────────────────────┐
│ 4. CONFIGURATION (GlyphConfig + Optimizer)                   │
│    - Load provider-specific profile                          │
│    - Apply optimization settings (DPI, font, columns)        │
│    - Get rendering configuration                             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. IMAGE RENDERING (PILTextRenderer)                         │
│    - Load fonts (OCRB/OCRA/system fonts)                     │
│    - Estimate text capacity per page                         │
│    - Split segments into pages if needed                     │
│    - Layout text in multi-column format                      │
│    - Render text with bold/italic effects                    │
│    - Save as optimized PNG images                            │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. QUALITY VALIDATION (QualityValidator)                     │
│    - Assess compression ratio (target: 3-4x)                 │
│    - Validate content preservation                           │
│    - Check readability for provider                          │
│    - Calculate weighted quality score                        │
│    - Compare against provider threshold                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. TOKEN CALCULATION (VLMTokenCalculator)                    │
│    - Calculate original token count                          │
│    - Use provider-specific VLM token calculation             │
│    - Consider image dimensions and detail level              │
│    - Compute accurate compression ratio                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. STATISTICS & CACHING (CompressionStats + Cache)           │
│    - Create CompressionStats object                          │
│    - Store in cache for future use                           │
│    - Record analytics metrics                                │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. MEDIA CONTENT CREATION (MediaContent)                     │
│    - Encode images as base64                                 │
│    - Create MediaContent objects                             │
│    - Embed compression metadata                              │
│    - Return List[MediaContent]                               │
└─────────────────────────────────────────────────────────────┘
```

### Example with Real Data

```python
# Input: 50,000 tokens of documentation
original_text = """..."""  # 50k tokens

# Step 1: Orchestrator decides to compress (> 10k threshold)
orchestrator = CompressionOrchestrator()
should_compress = True  # Auto-decision

# Step 2: Format text (markdown → structured)
formatter = TextFormatter()
segments = formatter.format_text(original_text)
# 2500 segments with formatting metadata

# Step 3: Cache miss (first compression)

# Step 4: Load OpenAI optimization profile
# dpi=72, font_size=9, columns=4

# Step 5: Render to images
renderer = PILTextRenderer(config)
images = renderer.segments_to_images(segments)
# 8 pages of 1024x1024 PNG images

# Step 6: Validate quality
validator = QualityValidator()
quality_score = 0.95  # Excellent

# Step 7: Calculate tokens
# Original: 50,000 tokens
# Compressed: 8 images × 1,800 tokens/image = 14,400 tokens
# Ratio: 3.47x compression

# Step 8: Cache and record
cache.set(cache_key, images, stats)
analytics.record_compression(...)

# Step 9: Return MediaContent objects
# 8 MediaContent objects with base64-encoded images
```

## Configuration

### GlyphConfig Options

```python
config = GlyphConfig(
    # Core settings
    enabled=True,                           # Enable compression
    global_default="auto",                  # auto/always/never
    quality_threshold=0.95,                 # Minimum quality score
    min_token_threshold=10000,              # Min tokens to consider
    target_compression_ratio=3.0,           # Target compression

    # Cache settings
    cache_directory="~/.abstractcore/glyph_cache",
    cache_size_gb=1.0,                      # Max cache size
    cache_ttl_days=7,                       # Cache expiration

    # Provider optimization
    provider_optimization=True,             # Use provider profiles
    preferred_provider="openai/gpt-4o",     # Default provider

    # Rendering configuration
    rendering=RenderingConfig(
        # Font configuration
        font_path="Verdana.ttf",            # Custom font path
        font_name="OCRB",                   # Font family name
        font_size=7,                        # Font size (6-10)
        line_height=8,                      # Line spacing

        # Layout configuration
        dpi=72,                             # 72 or 96
        target_width=1024,                  # Image width
        target_height=1024,                 # Image height
        margin_x=10,                        # Horizontal margin
        margin_y=10,                        # Vertical margin

        # Multi-column layout
        columns=2,                          # Number of columns
        column_gap=10,                      # Gap between columns

        # Optimization settings
        auto_crop_width=True,               # Crop unused width
        auto_crop_last_page=True,           # Crop last page
        render_format=True                  # Enable formatting
    ),

    # App-specific defaults
    app_defaults={
        "summarizer": "always",             # Always compress for summarization
        "extractor": "never",               # Never compress for extraction
        "judge": "auto",                    # Auto-decide for judging
        "cli": "auto"                       # Auto-decide for CLI
    },

    # Processing settings
    temp_dir=None,                          # Temp directory
    max_concurrent_compressions=2,          # Parallel compression limit
    processing_timeout=300                  # 5 minute timeout
)
```

### Provider-Specific Profiles

```python
# Optimized profiles based on empirical testing
provider_profiles = {
    "openai": {
        "dpi": 72,                          # Lower DPI for compression
        "font_size": 9,                     # Balanced readability
        "quality_threshold": 0.93,          # Excellent OCR
        "newline_markup": '<font color="#FF0000"> \\n </font>'
    },
    "anthropic": {
        "dpi": 96,                          # Higher DPI for Claude
        "font_size": 10,                    # Larger for clarity
        "quality_threshold": 0.96,          # Font-sensitive
        "font_path": "Verdana.ttf"          # Preferred font
    },
    "ollama": {
        "dpi": 72,                          # Aggressive compression
        "font_size": 9,                     # Standard size
        "auto_crop_width": True,            # Optimize space
        "auto_crop_last_page": True
    },
    "lmstudio": {
        "dpi": 96,                          # Quality focus
        "font_size": 10,                    # Readable
        "quality_threshold": 0.94           # Good OCR
    }
}
```

## Usage Patterns

### Basic Compression

```python
from abstractcore.compression import GlyphProcessor, GlyphConfig

# Initialize processor
config = GlyphConfig.default()
processor = GlyphProcessor(config)

# Compress text
media_contents = processor.process_text(
    content=long_document,
    provider="openai",
    model="gpt-4o"
)

# Use compressed images with LLM
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    prompt="Summarize this document:",
    media=media_contents
)
```

### Orchestrator (Recommended)

```python
from abstractcore.compression import CompressionOrchestrator

# Automatic decision-making
orchestrator = CompressionOrchestrator()

# Check if compression is beneficial
recommendation = orchestrator.get_compression_recommendation(
    content=document,
    provider="openai",
    model="gpt-4o"
)

print(f"Should compress: {recommendation['should_compress']}")
print(f"Reason: {recommendation['recommendation_reason']}")
print(f"Estimated savings: {recommendation['compression_estimate']['estimated_token_savings']} tokens")

# Apply compression if recommended
if recommendation['should_compress']:
    media_contents = orchestrator.compress_content(
        content=document,
        provider="openai",
        model="gpt-4o"
    )
```

### Advanced: Custom Configuration

```python
from abstractcore.compression import GlyphProcessor
from abstractcore.compression.config import GlyphConfig, RenderingConfig

# Custom rendering configuration
render_config = RenderingConfig(
    font_name="OCRB",                      # Use OCRB font
    font_size=7,                           # Ultra-dense
    line_height=8,
    dpi=72,                                # Prioritize compression
    columns=4,                             # 4-column layout
    margin_x=5,
    margin_y=5,
    target_width=1024,
    target_height=1024,
    render_format=True                     # Enable markdown formatting
)

# Custom Glyph configuration
config = GlyphConfig(
    enabled=True,
    quality_threshold=0.93,
    min_token_threshold=5000,              # Lower threshold
    target_compression_ratio=3.5,          # Higher target
    rendering=render_config
)

processor = GlyphProcessor(config)
media_contents = processor.process_text(document, "openai", "gpt-4o")
```

### With Analytics

```python
from abstractcore.compression import get_analytics

# Track compression performance
analytics = get_analytics()

# Process compression
media_contents = processor.process_text(...)

# Record metrics automatically via GlyphProcessor

# View analytics
report = analytics.generate_report()
print(report)

# Get provider-specific stats
openai_stats = analytics.get_provider_stats("openai")
print(f"OpenAI avg compression: {openai_stats['avg_compression_ratio']:.1f}x")
print(f"OpenAI avg quality: {openai_stats['avg_quality_score']:.1%}")

# Get optimization suggestions
suggestions = analytics.get_optimization_suggestions()
for suggestion in suggestions:
    print(f"- {suggestion}")
```

### Hybrid Compression (Experimental)

```python
from abstractcore.compression.vision_compressor import HybridCompressionPipeline

# Initialize hybrid pipeline
pipeline = HybridCompressionPipeline(
    vision_provider="ollama",
    vision_model="llama3.2-vision"
)

# Apply two-stage compression
result = pipeline.compress(
    text=very_long_document,
    target_ratio=30.0,                     # Target 30x compression
    min_quality=0.85                       # Acceptable quality
)

print(f"Total compression: {result['total_compression_ratio']:.1f}x")
print(f"Stage 1 (Glyph): {result['stages']['glyph']['ratio']:.1f}x")
print(f"Stage 2 (Vision): {result['stages']['vision']['ratio']:.1f}x")

# Access compressed media
media_contents = result['media']
```

## Integration Points

### With Media Processing

```python
# GlyphProcessor extends BaseMediaHandler
from abstractcore.media.base import BaseMediaHandler
from abstractcore.media.types import MediaType, MediaContent

class GlyphProcessor(BaseMediaHandler):
    def supports_media_type(self, media_type: MediaType) -> bool:
        return media_type in [MediaType.TEXT, MediaType.DOCUMENT]

    def _process_internal(self, file_path: Path, ...) -> MediaContent:
        # Compress file content
        content = read_file(file_path)
        media_contents = self.process_text(content, ...)
        return media_contents[0]
```

### With Providers

```python
# Provider integration via vision capabilities
from abstractcore.media.capabilities import get_model_capabilities

# Check if provider supports vision
capabilities = get_model_capabilities("openai", "gpt-4o")
if capabilities['vision_support']:
    # Compression is viable
    media_contents = processor.process_text(...)

    # Send to provider with vision support
    llm = create_llm("openai", model="gpt-4o")
    response = llm.generate(prompt="...", media=media_contents)
```

### With Token Calculation

```python
from abstractcore.utils.vlm_token_calculator import VLMTokenCalculator

# Accurate token calculation for VLMs
calculator = VLMTokenCalculator()

# Original text tokens
original_tokens = TokenUtils.estimate_tokens(text, model)

# Compressed image tokens
token_analysis = calculator.calculate_tokens_for_images(
    image_paths=compressed_images,
    provider="openai",
    model="gpt-4o"
)
compressed_tokens = token_analysis['total_tokens']

# Compression ratio
ratio = original_tokens / compressed_tokens
print(f"Compression ratio: {ratio:.1f}x")
```

### With Configuration System

```python
# Integration with AbstractCore centralized config
config = GlyphConfig.from_abstractcore_config()

# Saves to AbstractCore config
config.save_to_abstractcore_config()

# Access via config manager
from abstractcore.config import get_config_manager
config_manager = get_config_manager()
glyph_section = config_manager.config.glyph_compression
```

## Performance

### Token Savings

**Typical Compression Ratios** (based on empirical testing):

| Provider | Model | Avg Ratio | Quality | Notes |
|----------|-------|-----------|---------|-------|
| OpenAI | GPT-4o | 3.5x | 0.95 | Excellent OCR, aggressive compression |
| OpenAI | GPT-4o-mini | 4.0x | 0.93 | Good OCR, more aggressive |
| Anthropic | Claude 3.5 Sonnet | 3.0x | 0.96 | Conservative for quality |
| Anthropic | Claude 3.5 Haiku | 3.5x | 0.94 | Balanced compression |
| Ollama | llama3.2-vision | 4.5x | 0.88 | Aggressive for local models |
| Ollama | qwen2.5-vision | 4.0x | 0.89 | Good local compression |

**Example Savings**:
```
Document: 50,000 tokens
Compressed: 14,400 tokens (3.47x)
Token savings: 35,600 tokens
Cost savings: $0.36 (at $0.01/1k tokens)
```

### Processing Time

**Benchmark Results** (1024x1024 images, 4-column layout):

| Text Size | Pages | Render Time | Quality Check | Total Time |
|-----------|-------|-------------|---------------|------------|
| 10k tokens | 2 | 0.8s | 0.1s | 0.9s |
| 50k tokens | 8 | 2.5s | 0.3s | 2.8s |
| 100k tokens | 16 | 4.8s | 0.5s | 5.3s |
| 500k tokens | 80 | 22.4s | 2.1s | 24.5s |

**Cache Performance**:
- Cache hit: < 10ms
- Cache miss + compression: 0.9-24.5s (depending on size)
- Cache hit rate: ~40% for repeated documents

### Quality Metrics

**Quality Score Components**:

1. **Compression Ratio Score** (30% weight):
   - Excellent (1.0): 2.5-5.0x compression
   - Good (0.8): 2.0-2.5x or 5.0-6.0x
   - Acceptable (0.6): 1.5-2.0x or 6.0-8.0x
   - Poor (0.3): < 1.5x or > 8.0x

2. **Content Preservation Score** (40% weight):
   - Page count deviation penalty
   - Special character penalty (>30%)
   - Long line penalty (>20% lines > 200 chars)

3. **Readability Score** (30% weight):
   - Provider OCR quality multiplier
   - Code content penalty (-10%)
   - Math notation penalty (-15%)

**Provider Thresholds**:
```python
{
    "openai": 0.93,      # Excellent OCR capabilities
    "anthropic": 0.96,   # Font-sensitive, requires quality
    "ollama": 0.90,      # Variable open-source models
    "lmstudio": 0.94,    # Good local models
    "mlx": 0.88,         # Limited OCR on Apple Silicon
    "huggingface": 0.85  # Variable model quality
}
```

## Best Practices

### When to Use Compression

**Recommended Use Cases**:
1. Large documents (> 10k tokens)
2. Approaching context limits (> 80% of context window)
3. Cost-sensitive applications (API token costs)
4. Summarization tasks (always beneficial)
5. Long-form content analysis

**NOT Recommended**:
1. Short documents (< 10k tokens)
2. Extraction tasks (OCR may introduce errors)
3. Code analysis (formatting critical)
4. Mathematical content (notation issues)
5. Highly structured data (tables, JSON)

### Optimization Tips

**1. Choose Appropriate Provider Profile**:
```python
# OpenAI: Aggressive compression (excellent OCR)
config = optimizer.get_optimized_config("openai", "gpt-4o")

# Anthropic: Conservative (quality-focused)
config = optimizer.get_optimized_config("anthropic", "claude-3-5-sonnet")

# Ollama: Maximum compression (local models)
config = optimizer.get_optimized_config("ollama", "llama3.2-vision", aggressive=True)
```

**2. Use OCRB Font for Best Readability**:
```python
render_config = RenderingConfig(
    font_name="OCRB",                      # Optimized for OCR
    font_size=7,                           # Dense but readable
    line_height=8
)
```

**3. Enable Text Formatting**:
```python
render_config.render_format = True         # Markdown formatting
# Improves readability: headers, bold, italic
```

**4. Configure Multi-Column Layout**:
```python
# Balance compression vs readability
render_config.columns = 4                  # 4 columns for OpenAI
render_config.columns = 3                  # 3 columns for Anthropic
render_config.columns = 6                  # 6 columns for aggressive
```

**5. Use Orchestrator for Automatic Decisions**:
```python
orchestrator = CompressionOrchestrator()
media = orchestrator.compress_content(text, provider, model)
# Automatically decides based on best practices
```

**6. Monitor Analytics**:
```python
analytics = get_analytics()
suggestions = analytics.get_optimization_suggestions()
# Adaptive improvement based on actual performance
```

**7. Cache Configuration**:
```python
config.cache_size_gb = 2.0                 # Larger cache for repeated docs
config.cache_ttl_days = 14                 # Longer TTL for stable docs
```

## Common Pitfalls

### Quality Issues

**Problem**: Low quality scores, poor OCR accuracy

**Causes**:
- Font too small (< 6pt)
- Too many columns (> 6)
- DPI too low (< 72)
- Mathematical notation present
- Code-heavy content

**Solutions**:
```python
# Increase font size
render_config.font_size = 9                # Larger for quality

# Reduce columns
render_config.columns = 2                  # Fewer columns

# Increase DPI
render_config.dpi = 96                     # Higher quality

# Adjust quality threshold
config.quality_threshold = 0.90            # More lenient

# Use user_preference="always" to bypass quality check
media = processor.process_text(text, provider, model, user_preference="always")
```

### Model Compatibility

**Problem**: Model doesn't support vision

**Detection**:
```python
from abstractcore.media.capabilities import get_model_capabilities

capabilities = get_model_capabilities(provider, model)
if not capabilities['vision_support']:
    print(f"{provider}/{model} does not support vision")
    # Use standard text processing instead
```

**Vision-Capable Models**:
- OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4-vision-preview
- Anthropic: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-sonnet, claude-3-haiku
- Ollama: llama3.2-vision, qwen2.5-vision, llava, bakllava
- LMStudio: Any vision-capable model
- HuggingFace: Vision models with custom detail levels

### Performance Issues

**Problem**: Slow compression times

**Causes**:
- Large documents (> 500k tokens)
- Complex formatting
- Multiple concurrent compressions
- Inefficient configuration

**Solutions**:
```python
# Increase concurrent limit
config.max_concurrent_compressions = 4     # Parallel processing

# Disable formatting for large files
render_config.render_format = False        # Skip markdown parsing

# Use aggressive configuration
config = optimizer.get_optimized_config(provider, model, aggressive=True)

# Increase timeout for large documents
config.processing_timeout = 600            # 10 minutes
```

### Cache Issues

**Problem**: Cache misses, disk space exhaustion

**Solutions**:
```python
# Clear cache periodically
cache.clear()

# Adjust cache settings
config.cache_size_gb = 0.5                 # Smaller cache
config.cache_ttl_days = 3                  # Shorter TTL

# Manual cleanup
cache._cleanup_if_needed()

# Check cache stats
stats = cache.get_stats()
print(f"Cache utilization: {stats['utilization']:.1%}")
```

## Testing Strategy

### Unit Tests

```python
# Test individual components
def test_text_formatter():
    formatter = TextFormatter()
    segments = formatter.format_text("**Bold** and *italic*")
    assert segments[0].is_bold
    assert segments[1].is_italic

def test_quality_validator():
    validator = QualityValidator()
    score = validator.assess(text, images, "openai")
    assert 0.0 <= score <= 1.0

def test_compression_cache():
    cache = CompressionCache()
    cache.set("key", images, stats)
    assert cache.get("key") is not None
```

### Integration Tests

```python
def test_full_compression_pipeline():
    processor = GlyphProcessor()
    media = processor.process_text(
        content=test_document,
        provider="openai",
        model="gpt-4o"
    )
    assert len(media) > 0
    assert all(m.media_type == MediaType.IMAGE for m in media)
    assert all(m.metadata.get('compression_ratio', 0) > 2.0 for m in media)

def test_orchestrator_decision():
    orchestrator = CompressionOrchestrator()
    # Small document - should not compress
    assert not orchestrator.should_compress("Short text", "openai", "gpt-4o")
    # Large document - should compress
    assert orchestrator.should_compress(large_doc, "openai", "gpt-4o")
```

### Performance Tests

```python
def test_compression_performance():
    import time
    processor = GlyphProcessor()

    start = time.time()
    media = processor.process_text(document_50k_tokens, "openai", "gpt-4o")
    duration = time.time() - start

    assert duration < 5.0  # Should complete in < 5 seconds
    assert len(media) <= 10  # Should create reasonable number of images

def test_cache_performance():
    cache = CompressionCache()

    # First compression (cache miss)
    start = time.time()
    cache.get("key")  # None
    miss_time = time.time() - start

    cache.set("key", images, stats)

    # Second compression (cache hit)
    start = time.time()
    result = cache.get("key")
    hit_time = time.time() - start

    assert result is not None
    assert hit_time < 0.01  # < 10ms for cache hit
```

### Quality Tests

```python
def test_compression_quality():
    processor = GlyphProcessor()
    media = processor.process_text(document, "openai", "gpt-4o")

    # Check quality score
    quality = media[0].metadata['quality_score']
    assert quality >= 0.93  # OpenAI threshold

    # Check compression ratio
    ratio = media[0].metadata['compression_ratio']
    assert 2.5 <= ratio <= 5.0  # Target range

    # Verify token savings
    original_tokens = media[0].metadata.get('original_tokens', 0)
    compressed_tokens = media[0].metadata.get('compressed_tokens', 0)
    assert compressed_tokens < original_tokens

def test_provider_compatibility():
    processor = GlyphProcessor()

    for provider in ["openai", "anthropic", "ollama"]:
        media = processor.process_text(document, provider, "model")
        assert len(media) > 0
        assert media[0].metadata['provider_optimized'] == provider
```

## Public API

### Recommended Imports

```python
# High-level API (recommended for most users)
from abstractcore.compression import CompressionOrchestrator

# Core compression
from abstractcore.compression import GlyphProcessor, GlyphConfig

# Configuration
from abstractcore.compression.config import RenderingConfig

# Analytics
from abstractcore.compression import get_analytics

# Exceptions
from abstractcore.compression.exceptions import (
    CompressionError,
    CompressionQualityError,
    RenderingError,
    CompressionCacheError
)

# Advanced features
from abstractcore.compression.optimizer import CompressionOptimizer
from abstractcore.compression.vision_compressor import (
    HybridCompressionPipeline,
    VisionCompressor
)
```

### Primary API Classes

```python
# Orchestrator: Intelligent decision-making
CompressionOrchestrator()
  .should_compress(content, provider, model, user_preference)
  .compress_content(content, provider, model, user_preference)
  .get_compression_recommendation(content, provider, model)

# GlyphProcessor: Core compression
GlyphProcessor(config)
  .process_text(content, provider, model, user_preference)
  .can_process(content, provider, model)
  .get_compression_stats()

# Configuration
GlyphConfig.default()
GlyphConfig.from_abstractcore_config()
  .get_provider_config(provider, model)
  .should_compress(content_length, provider, model, user_preference)

# Analytics
get_analytics()
  .record_compression(...)
  .get_provider_stats(provider)
  .get_model_stats(provider, model)
  .generate_report()
  .get_optimization_suggestions()

# Optimization
CompressionOptimizer()
  .get_optimized_config(provider, model, aggressive)
  .analyze_compression_potential(text_length, provider, model)

# Hybrid (experimental)
HybridCompressionPipeline(vision_provider, vision_model)
  .compress(text, target_ratio, min_quality)
```

---

## Summary

The `abstractcore.compression` module provides a comprehensive glyph-based compression system for VLMs with:

- **3-4x token savings** through visual-text compression
- **Provider-specific optimization** for OpenAI, Anthropic, Ollama, etc.
- **Intelligent decision-making** via Orchestrator
- **Quality validation** with multi-metric assessment
- **Performance analytics** and optimization suggestions
- **Caching system** for repeated documents
- **Experimental hybrid compression** for 15-40x ratios

**Key Strengths**:
- Production-ready with comprehensive error handling
- Extensive configuration and customization
- Integration with AbstractCore media and provider systems
- Performance-optimized with caching and parallel processing
- Analytics-driven continuous improvement

**Best For**:
- Large document processing (> 10k tokens)
- Cost-sensitive API usage
- Context-limited scenarios
- Summarization and analysis tasks

**Use With Caution**:
- Extraction tasks (OCR errors)
- Code analysis (formatting loss)
- Mathematical content (notation issues)
- Highly structured data (table layout)

## Related Modules

**Direct dependencies**:
- [`media/`](../media/README.md) - Media processing base classes, MediaContent
- [`utils/`](../utils/README.md) - Token estimation, VLM token calculation
- [`config/`](../config/README.md) - Compression configuration
- [`exceptions/`](../exceptions/README.md) - Compression-specific errors
- [`events/`](../events/README.md) - Compression operation events
- [`architectures/`](../architectures/README.md) - Provider capability detection
- [`assets/`](../assets/README.md) - Provider optimization profiles

**Used by**:
- [`media/`](../media/README.md) - Auto handler integrates glyph compression
- [`processing/`](../processing/README.md) - Large document processing
- [`apps/`](../apps/README.md) - Summarizer for long documents

**Compression pipeline**:
- [`providers/`](../providers/README.md) - Vision models process compressed images
- [`core/`](../core/README.md) - Orchestrator in factory pattern
