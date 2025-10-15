# Basic Summarizer

The Basic Summarizer demonstrates how to build sophisticated text processing capabilities on top of AbstractCore using clean, zero-shot structured prompting techniques.

**💡 Recommended Setup**: For best performance, use the free local model `gemma3:1b-it-qat` with Ollama, which provides fast processing (29s), high quality (95% confidence), and zero API costs.

## Overview

The `BasicSummarizer` showcases AbstractCore's core strengths:

- **Structured Output**: Uses Pydantic models for type-safe, validated responses
- **Provider Agnostic**: Works with any LLM provider through AbstractCore's unified interface
- **Built-in Reliability**: Inherits AbstractCore's retry mechanisms and error handling
- **Chunking Support**: Automatically handles long documents using map-reduce approach
- **Event Integration**: All operations emit events for monitoring and debugging

## Quick Start

**Prerequisites**: For local processing, install [Ollama](https://ollama.ai) and download the recommended model:
```bash
# Install Ollama, then download the fast, high-quality model
ollama pull gemma3:1b-it-qat
```

```python
from abstractllm import create_llm
from abstractllm.processing import BasicSummarizer, SummaryStyle, SummaryLength

# Recommended: Fast local model for cost-effective processing
llm = create_llm("ollama", model="gemma3:1b-it-qat")

# Alternative: Cloud provider for highest quality
# llm = create_llm("openai", model="gpt-4o-mini")

# Create summarizer
summarizer = BasicSummarizer(llm)

# Basic usage
result = summarizer.summarize("Your long text here...")
print(result.summary)
print(f"Confidence: {result.confidence:.2f}")
```

## Command Line Interface

The `summarizer` CLI provides direct terminal access for document summarization without any Python programming.

### Quick CLI Usage

```bash
# Simple usage (after pip install abstractcore[all])
summarizer document.pdf

# With specific style and length
summarizer report.txt --style executive --length brief

# Focus on specific aspects
summarizer data.md --focus "technical details" --output summary.txt

# Use different provider
summarizer large.txt --provider openai --model gpt-4o-mini --verbose
```

### CLI Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `file_path` | Any text file | Required | Path to the file to summarize |
| `--style` | `structured`, `narrative`, `objective`, `analytical`, `executive`, `conversational` | `structured` | Summary presentation style |
| `--length` | `brief`, `standard`, `detailed`, `comprehensive` | `standard` | Summary length and depth |
| `--focus` | Any text | None | Specific focus area for summarization |
| `--output` | File path | Console | Output file path (prints to console if not provided) |
| `--chunk-size` | 1000-32000 | 8000 | Chunk size in characters for large documents |
| `--provider` | `openai`, `anthropic`, `ollama`, etc. | `ollama` | LLM provider (requires --model) |
| `--model` | Provider-specific | `gemma3:1b-it-qat` | LLM model (requires --provider) |
| `--verbose` | Flag | False | Show detailed progress information |

### CLI Examples

```bash
# Basic document summarization
summarizer document.pdf
summarizer report.txt --verbose

# Executive summary for business documents
summarizer quarterly_report.pdf --style executive --length brief --output exec_summary.txt

# Technical focus with detailed analysis
summarizer technical_spec.md --focus "implementation details" --style analytical --length detailed

# Large document processing with custom chunking
summarizer large_manual.txt --chunk-size 15000 --verbose

# Using cloud providers for highest quality
summarizer important_doc.pdf --provider openai --model gpt-4o-mini --style executive

# Batch processing with shell scripting
for file in *.pdf; do
    summarizer "$file" --style structured --output "${file%.pdf}_summary.txt"
done
```

### Alternative Usage Methods

```bash
# Method 1: Direct command (recommended after installation)
summarizer document.txt --style executive

# Method 2: Via Python module (always works)
python -m abstractllm.apps.summarizer document.txt --style executive
```

### Supported File Types

The CLI supports most text-based file formats:
- `.txt`, `.md`, `.py`, `.js`, `.html`, `.json`, `.csv`
- Most other text-based files

### Default Model Setup

The CLI uses `gemma3:1b-it-qat` by default for fast, cost-effective processing:

```bash
# Install Ollama: https://ollama.com/
# Download the default model
ollama pull gemma3:1b-it-qat

# Then use directly
summarizer document.txt
```

## Configuration Options

### Summary Styles

Control how the summary is presented:

```python
from abstractllm.processing import SummaryStyle

# Available styles
SummaryStyle.STRUCTURED     # Bullet points, clear sections
SummaryStyle.NARRATIVE      # Flowing, story-like prose
SummaryStyle.OBJECTIVE      # Neutral, factual tone
SummaryStyle.ANALYTICAL     # Critical analysis with insights
SummaryStyle.EXECUTIVE      # Business-focused, action-oriented
SummaryStyle.CONVERSATIONAL # Chat history preservation with context
```

### Summary Lengths

Control the detail level:

```python
from abstractllm.processing import SummaryLength

# Available lengths
SummaryLength.BRIEF         # 2-3 sentences, key point only
SummaryLength.STANDARD      # 1-2 paragraphs, main ideas
SummaryLength.DETAILED      # Multiple paragraphs, comprehensive
SummaryLength.COMPREHENSIVE # Full analysis with context
```

## Advanced Usage

### Focus Areas

Specify what aspect of the text to emphasize:

```python
# Focus on specific aspects
result = summarizer.summarize(
    text,
    focus="business implications",
    style=SummaryStyle.EXECUTIVE,
    length=SummaryLength.DETAILED
)

print(f"Focus alignment: {result.focus_alignment:.2f}")
```

### Different Providers

The same code works with any provider:

```python
# OpenAI
llm_openai = create_llm("openai", model="gpt-4o-mini")
summarizer_openai = BasicSummarizer(llm_openai)

# Anthropic
llm_claude = create_llm("anthropic", model="claude-3-5-haiku-latest")
summarizer_claude = BasicSummarizer(llm_claude)

# Local models
llm_ollama = create_llm("ollama", model="llama3.2")
summarizer_local = BasicSummarizer(llm_ollama)

# All work identically
result = summarizer_openai.summarize(text)
result = summarizer_claude.summarize(text)
result = summarizer_local.summarize(text)
```

### Long Document Processing

Automatically handles documents of any length:

```python
# Works with short documents
short_result = summarizer.summarize(short_article)

# Automatically chunks long documents
long_result = summarizer.summarize(entire_book_text)

# Customize chunking
summarizer = BasicSummarizer(llm, max_chunk_size=6000)
```

## Output Structure

The `SummaryOutput` provides rich, structured information:

```python
result = summarizer.summarize(text)

# Main summary
print(result.summary)

# Key points (3-5 most important)
for point in result.key_points:
    print(f"• {point}")

# Quality metrics
print(f"Confidence: {result.confidence:.2f}")
print(f"Focus alignment: {result.focus_alignment:.2f}")

# Word counts
print(f"Original: {result.word_count_original} words")
print(f"Summary: {result.word_count_summary} words")
print(f"Compression: {result.word_count_original / result.word_count_summary:.1f}x")
```

## Real-World Examples

### Executive Summary

```python
result = summarizer.summarize(
    quarterly_report,
    focus="financial performance and strategic initiatives",
    style=SummaryStyle.EXECUTIVE,
    length=SummaryLength.STANDARD
)

print("Executive Summary:")
print(result.summary)
print("\nKey Action Items:")
for point in result.key_points:
    print(f"• {point}")
```

### Research Paper Analysis

```python
result = summarizer.summarize(
    research_paper,
    focus="methodology and findings",
    style=SummaryStyle.ANALYTICAL,
    length=SummaryLength.DETAILED
)

if result.confidence > 0.8:
    print("High-confidence analysis:")
    print(result.summary)
else:
    print("Consider manual review - confidence low")
```

### Technical Documentation

```python
result = summarizer.summarize(
    technical_docs,
    focus="implementation details and requirements",
    style=SummaryStyle.STRUCTURED,
    length=SummaryLength.COMPREHENSIVE
)

print("Technical Overview:")
print(result.summary)
```

## Event Monitoring

Monitor summarization progress with AbstractCore's event system:

```python
from abstractllm.events import EventType, on_global

def monitor_summarization(event):
    if event.type == EventType.BEFORE_GENERATE:
        print("🔄 Starting summarization...")
    elif event.type == EventType.AFTER_GENERATE:
        print(f"✅ Completed in {event.duration_ms}ms")

on_global(EventType.BEFORE_GENERATE, monitor_summarization)
on_global(EventType.AFTER_GENERATE, monitor_summarization)

result = summarizer.summarize(text)
```

## Error Handling

Built-in reliability through AbstractCore:

```python
from abstractllm.core.retry import RetryConfig

# Configure retry behavior
config = RetryConfig(max_attempts=3, initial_delay=1.0)
llm = create_llm("ollama", model="gemma3:1b-it-qat", retry_config=config)

summarizer = BasicSummarizer(llm)

# Automatic retry on failures
try:
    result = summarizer.summarize(text)
except Exception as e:
    print(f"Summarization failed after retries: {e}")
```

## Performance Considerations

### Document Length Guidelines

- **< 8,000 chars**: Single-pass summarization (fastest)
- **8,000-50,000 chars**: Automatic chunking with minimal overhead
- **> 50,000 chars**: Map-reduce approach, may take longer but handles unlimited size

### Provider Selection

**Recommended for Production:**
- **Ollama gemma3:1b-it-qat**: Fast (29s), high quality (95% confidence), cost-effective local processing
- **Ollama qwen3-coder:30b**: Premium quality (98% confidence), slower (119s), best for critical tasks

**Cloud Alternatives:**
- **OpenAI GPT-4o-mini**: Excellent quality with API costs, good for low-volume
- **Anthropic Claude**: Great for analytical and narrative styles

**Performance Comparison:**
```
Model              Speed    Quality  Cost    Best For
gemma3:1b-it-qat  Fast     High     Free    Production, high-volume
qwen3-coder:30b   Slow     Premium  Free    Critical accuracy
GPT-4o-mini       Medium   High     Paid    Occasional use
Claude-3.5        Medium   High     Paid    Narrative summaries
```

### Cost Optimization

```python
# Free local processing with excellent quality
llm = create_llm("ollama", model="gemma3:1b-it-qat")  # Fast, free, high quality
summarizer = BasicSummarizer(llm)

# Brief summaries for even faster processing
result = summarizer.summarize(
    text,
    length=SummaryLength.BRIEF  # Fastest processing
)

# Cloud option for occasional use
# llm = create_llm("openai", model="gpt-4o-mini")  # vs gpt-4o
```

## Implementation Details

### Chunking Strategy

For long documents:

1. **Smart splitting**: Breaks at sentence boundaries when possible
2. **Overlap**: 200-character overlap between chunks to maintain context
3. **Map-reduce**: Summarizes chunks independently, then combines
4. **Coherence**: Final combination step ensures unified narrative

### Prompt Engineering

The summarizer uses sophisticated prompts that:

- **Adapt to style**: Different instructions for each presentation style
- **Scale with length**: Appropriate guidance for brief vs comprehensive
- **Handle focus**: Specific attention to user-specified focus areas
- **Validate quality**: Self-assessment of confidence and focus alignment

### Quality Assurance

- **Pydantic validation**: Ensures structured output conforms to schema
- **Confidence scoring**: LLM self-assesses summary accuracy
- **Focus alignment**: Measures how well summary addresses specified focus
- **Word counting**: Tracks compression ratios

## Integration Examples

### With AbstractCore Session

```python
from abstractllm import BasicSession

session = BasicSession(llm, system_prompt="You are an expert summarizer")
summarizer = BasicSummarizer(session)

# Maintains conversation context
result1 = summarizer.summarize(doc1, focus="technical aspects")
result2 = summarizer.summarize(doc2, focus="how this relates to the previous document")
```

### Batch Processing

```python
documents = [doc1, doc2, doc3, doc4]
summaries = []

for doc in documents:
    result = summarizer.summarize(
        doc.content,
        focus="key insights",
        style=SummaryStyle.STRUCTURED,
        length=SummaryLength.STANDARD
    )
    summaries.append({
        'title': doc.title,
        'summary': result.summary,
        'key_points': result.key_points,
        'confidence': result.confidence
    })

# Filter high-confidence summaries
high_quality = [s for s in summaries if s['confidence'] > 0.8]
```

## Extending the Summarizer

The Basic Summarizer serves as a foundation for more advanced processing:

```python
class CustomSummarizer(BasicSummarizer):
    def summarize_with_keywords(self, text: str, keywords: List[str]) -> SummaryOutput:
        focus = f"these specific keywords: {', '.join(keywords)}"
        return self.summarize(text, focus=focus, style=SummaryStyle.ANALYTICAL)

    def comparative_summarize(self, texts: List[str]) -> List[SummaryOutput]:
        focus = "comparative analysis and differences"
        return [self.summarize(text, focus=focus) for text in texts]
```

## Best Practices

1. **Choose appropriate length**: Match summary length to use case
2. **Use focus effectively**: Specific focus areas improve relevance
3. **Monitor confidence**: Low confidence may indicate need for manual review
4. **Provider selection**: Match provider capabilities to content type
5. **Batch processing**: Process similar documents together for consistency
6. **Error handling**: Always handle potential failures gracefully

## Conclusion

The Basic Summarizer demonstrates how AbstractCore's infrastructure enables building sophisticated text processing capabilities with minimal complexity. It showcases:

- **Clean API design** with powerful customization options
- **Automatic reliability** through built-in retry and error handling
- **Universal compatibility** across all LLM providers
- **Scalable architecture** handling documents of any size
- **Production readiness** with comprehensive error handling and monitoring

This implementation serves both as a useful tool for real summarization needs and as a reference for building other text processing capabilities on top of AbstractCore.