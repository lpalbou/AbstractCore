# BasicExtractor - Knowledge Graph Extraction

BasicExtractor is a production-ready tool for extracting structured knowledge graphs from text documents. It converts unstructured text into semantic entities and relationships using large language models, outputting clean JSON-LD or RDF triple formats.

## Quick Start

```python
from abstractllm.processing import BasicExtractor

# Initialize with default model (Ollama qwen3:4b-instruct-2507-q4_K_M)
extractor = BasicExtractor()

# Extract knowledge graph
result = extractor.extract("Google created TensorFlow in 2015. Microsoft uses TensorFlow for Azure AI.")

# Result contains entities and relationships in JSON-LD format
entities = [item for item in result['@graph'] if item.get('@id', '').startswith('e:')]
relationships = [item for item in result['@graph'] if item.get('@id', '').startswith('r:')]
```

## Installation & Setup

```bash
# Install AbstractCore with dependencies
pip install abstractcore[all]

# Default model requires Ollama (free, runs locally)
# 1. Install Ollama: https://ollama.com/
# 2. Download model: ollama pull qwen3:4b-instruct-2507-q4_K_M
# 3. Start Ollama service

# Alternative: Use cloud providers
pip install abstractcore[openai,anthropic]
```

### Model Performance Recommendations

**Default Model**: `qwen3:4b-instruct-2507-q4_K_M`
- **Size**: ~4GB model
- **RAM**: ~8GB required
- **Speed**: Good balance of speed and quality
- **Setup**: `ollama pull qwen3:4b-instruct-2507-q4_K_M`

**For Best Performance**:
- **`qwen3-coder:30b`**: Excellent for structured JSON-LD output (requires 32GB RAM)
- **`gpt-oss:120b`**: Highest quality extraction (requires 120GB RAM)

**For Production**: Cloud providers (OpenAI GPT-4o-mini, Claude) offer the most reliable JSON-LD generation.

## Output Formats

BasicExtractor supports three output formats optimized for different use cases:

### 1. JSON-LD Format (Default)

Standard W3C JSON-LD with schema.org vocabulary - ideal for semantic web applications:

```python
result = extractor.extract("Apple acquired Siri in 2010", output_format="jsonld")
```

```json
{
  "@context": {
    "s": "https://schema.org/",
    "e": "http://example.org/entity/",
    "r": "http://example.org/relation/",
    "confidence": "http://example.org/confidence"
  },
  "@graph": [
    {
      "@id": "e:apple",
      "@type": "s:Organization",
      "s:name": "Apple",
      "s:description": "Technology company",
      "confidence": 0.95
    },
    {
      "@id": "r:1",
      "@type": "s:Relationship",
      "s:name": "acquires",
      "s:about": {"@id": "e:apple"},
      "s:object": {"@id": "e:siri"},
      "confidence": 0.90
    }
  ]
}
```

### 2. RDF Triples Format (New)

SUBJECT PREDICATE OBJECT format following semantic web standards - perfect for graph databases:

```python
result = extractor.extract("Apple acquired Siri in 2010", output_format="triples")
```

**Simple output:**
```
Apple acquires Siri
```

**Detailed output (with metadata):**
```json
{
  "format": "triples",
  "simple_triples": ["Apple acquires Siri"],
  "triples": [
    {
      "subject": "e:apple",
      "subject_name": "Apple",
      "predicate": "acquires",
      "object": "e:siri",
      "object_name": "Siri",
      "confidence": 0.90
    }
  ],
  "entities": {...},
  "statistics": {"entities_count": 2, "relationships_count": 1}
}
```

### 3. Minified JSON-LD Format (New)

Compact JSON string without indentation - optimized for storage and transport:

```python
result = extractor.extract("Apple acquired Siri in 2010", output_format="jsonld_minified")
```

```json
{
  "format": "jsonld_minified",
  "data": "{\"@context\":{\"s\":\"https://schema.org/\"},\"@graph\":[...]}",
  "entities_count": 2,
  "relationships_count": 1
}
```

## Python API Reference

### BasicExtractor Class

```python
class BasicExtractor:
    def __init__(
        self,
        llm: Optional[AbstractLLMInterface] = None,
        max_chunk_size: int = 8000
    )

    def extract(
        self,
        text: str,
        domain_focus: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        style: Optional[str] = None,
        length: Optional[str] = None,
        output_format: str = "jsonld"
    ) -> dict
```

### Parameters

- **`text`** (str): Text to extract knowledge from
- **`domain_focus`** (str, optional): Focus area like "technology", "business", "medical"
- **`entity_types`** (List[str], optional): Reserved for future use
- **`style`** (str, optional): Reserved for future use
- **`length`** (str, optional): Extraction depth
  - `"brief"` - 10 entities max (fast)
  - `"standard"` - 15 entities max (balanced)
  - `"detailed"` - 25 entities max (thorough)
  - `"comprehensive"` - 50 entities max (extensive)
- **`output_format`** (str): Output format
  - `"jsonld"` - Standard JSON-LD (default)
  - `"triples"` - RDF SUBJECT PREDICATE OBJECT format
  - `"jsonld_minified"` - Compact JSON string

### Custom LLM Provider

```python
from abstractllm import create_llm
from abstractllm.processing import BasicExtractor

# RECOMMENDED: Use cloud providers for complex JSON-LD extraction
llm = create_llm("openai", model="gpt-4o-mini")  # Best for production
extractor = BasicExtractor(llm)

# OR use Anthropic Claude for high quality
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
extractor = BasicExtractor(llm)

# LOCAL MODELS: Work well for simple extraction, may struggle with complex JSON-LD
llm = create_llm("ollama", model="qwen3-coder:30b")  # Good for code and structured tasks
extractor = BasicExtractor(llm)

# For simple fact extraction with local models, use direct prompting:
facts_prompt = """Extract facts as JSON: [{"entity": "...", "action": "...", "object": "..."}]"""
facts = llm.generate(facts_prompt)  # Works reliably with local models
```

## Command Line Interface

The `extractor` CLI provides direct terminal access for knowledge graph extraction without any Python programming.

### Quick CLI Usage

```bash
# Simple usage (after pip install abstractcore[all])
extractor document.pdf

# With specific format and focus
extractor report.txt --format triples --focus technology

# Extract specific entity types
extractor data.md --entity-types person,organization --output entities.jsonld

# High-quality extraction with iterations
extractor doc.txt --iterate=3 --length=detailed --verbose
```

### Alternative Usage Methods

```bash
# Method 1: Direct command (recommended after installation)
extractor document.txt --format triples

# Method 2: Via Python module (always works)
python -m abstractllm.apps.extractor document.txt --format triples
```

### Basic Usage

```bash
# Extract from file (default: JSON-LD)
extractor document.txt
# OR: python -m abstractllm.apps.extractor document.txt

# Specify output format
extractor document.txt --format triples
# OR: python -m abstractllm.apps.extractor document.txt --format triples

# Save to file
extractor document.txt --output knowledge_graph.jsonld
# OR: python -m abstractllm.apps.extractor document.txt --output knowledge_graph.jsonld
```

### Advanced Options

```bash
# Domain-focused extraction
extractor tech_report.txt --focus technology --length detailed
# OR: python -m abstractllm.apps.extractor tech_report.txt --focus technology --length detailed

# Custom provider and model
extractor document.txt --provider openai --model gpt-4o-mini
# OR: python -m abstractllm.apps.extractor document.txt --provider openai --model gpt-4o-mini

# Minified output for storage
extractor document.txt --format json-ld --minified
# OR: python -m abstractllm.apps.extractor document.txt --format json-ld --minified

# Iterative refinement for quality
extractor document.txt --iterate 3 --verbose
# OR: python -m abstractllm.apps.extractor document.txt --iterate 3 --verbose
```

### CLI Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `file_path` | Any text file | Required | Path to the file to extract from |
| `--focus` | Any text | None | Specific focus area (e.g., "technology", "business") |
| `--style` | `structured`, `focused`, `minimal`, `comprehensive` | `structured` | Extraction style |
| `--length` | `brief`, `standard`, `detailed`, `comprehensive` | `brief` | Extraction depth |
| `--entity-types` | Comma-separated list | All types | Entity types to focus on |
| `--similarity-threshold` | 0.0-1.0 | 0.85 | Similarity threshold for deduplication |
| `--format` | `json-ld`, `triples`, `json`, `yaml` | `json-ld` | Output format |
| `--output` | File path | Console | Output file path |
| `--chunk-size` | 1000-32000 | 6000 | Chunk size in characters |
| `--provider` | `openai`, `anthropic`, `ollama`, etc. | `ollama` | LLM provider |
| `--model` | Provider-specific | `qwen3:4b-instruct-2507-q4_K_M` | LLM model |
| `--no-embeddings` | Flag | False | Disable semantic deduplication |
| `--mode` | `fast`, `balanced`, `thorough` | `balanced` | Extraction mode |
| `--fast` | Flag | False | Legacy flag (use --mode=fast instead) |
| `--iterate` | 1-10 | 1 | Number of refinement iterations |
| `--minified` | Flag | False | Output minified JSON |
| `--verbose` | Flag | False | Show detailed progress |
| `--timeout` | Seconds | 300 | HTTP timeout for LLM requests |

### Entity Types

Available entity types for `--entity-types` parameter:
- `person` - People and individuals
- `organization` - Companies, institutions, groups  
- `location` - Places, cities, countries, addresses
- `concept` - Abstract concepts, ideas, theories
- `event` - Occurrences, meetings, incidents
- `technology` - Software, hardware, technical systems
- `product` - Products, services, offerings
- `date` - Temporal references, dates, times
- `other` - Miscellaneous entities

### Performance Modes

| Mode | Speed | Quality | Description |
|------|-------|---------|-------------|
| `fast` | 2-4x faster | Good | Skip verification, larger chunks, no embeddings |
| `balanced` | Standard | High | Default mode with verification (default) |
| `thorough` | Slower | Highest | Maximum quality with multiple checks |

### Output Format Examples

**Simple triples (no --verbose):**
```bash
python -m abstractllm.apps.extractor doc.txt --format triples
# Output:
# Google creates TensorFlow
# Microsoft uses TensorFlow
# OpenAI develops GPT-4
```

**Detailed triples (with --verbose):**
```bash
python -m abstractllm.apps.extractor doc.txt --format triples --verbose
# Output: JSON with entities, relationships, confidence scores, statistics
```

**Minified JSON-LD:**
```bash
python -m abstractllm.apps.extractor doc.txt --format json-ld --minified
# Output: {"@context":{"s":"https://schema.org/"},"@graph":[...]}
```

## Real-World Examples

### Example 1: Technology Documentation

**Input:**
```
Google's TensorFlow is an open-source machine learning framework.
Microsoft Azure integrates TensorFlow for cloud AI services.
OpenAI's GPT models use transformer architecture developed by Google Research.
```

**Command:**
```bash
python -m abstractllm.apps.extractor tech.txt --focus technology --length detailed --format triples --verbose
```

**Expected Output:**
```json
{
  "format": "triples",
  "simple_triples": [
    "Google creates TensorFlow",
    "Microsoft integrates TensorFlow",
    "OpenAI develops GPT",
    "Google Research develops transformer architecture"
  ],
  "entities": {
    "e:google": {"name": "Google", "type": "s:Organization"},
    "e:tensorflow": {"name": "TensorFlow", "type": "s:SoftwareApplication"},
    "e:microsoft": {"name": "Microsoft", "type": "s:Organization"}
  },
  "statistics": {"entities_count": 6, "relationships_count": 4}
}
```

### Example 2: Business Analysis

**Python API:**
```python
from abstractllm.processing import BasicExtractor

extractor = BasicExtractor()

business_text = """
Amazon acquired Whole Foods for $13.7 billion in 2017.
The acquisition expanded Amazon's grocery delivery capabilities.
Jeff Bezos was CEO of Amazon during this strategic move.
"""

# Extract with business focus
result = extractor.extract(
    business_text,
    domain_focus="business",
    length="standard",
    output_format="jsonld"
)

# Access entities and relationships
entities = [item for item in result['@graph'] if item.get('@id', '').startswith('e:')]
relationships = [item for item in result['@graph'] if item.get('@id', '').startswith('r:')]

print(f"Found {len(entities)} entities and {len(relationships)} relationships")
```

### Example 3: Research Paper Processing

**Command:**
```bash
# Process academic paper with comprehensive extraction
python -m abstractllm.apps.extractor research_paper.pdf \
  --focus "research" \
  --length comprehensive \
  --iterate 2 \
  --format json-ld \
  --output paper_knowledge_graph.jsonld \
  --verbose
```

## Best Practices

### 1. Model Selection

**For Complex JSON-LD Extraction (RECOMMENDED):**
```python
# Best quality for structured knowledge graphs
llm = create_llm("openai", model="gpt-4o-mini")  # $0.001-0.01 per request
extractor = BasicExtractor(llm)

# Alternative: High-quality Claude
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")  # Similar cost
extractor = BasicExtractor(llm)
```

**For Simple Fact Extraction (Local):**
```python
# Works well with qwen3-coder:30b for basic structured output
llm = create_llm("ollama", model="qwen3-coder:30b")  # 18GB, free
# Use simple JSON prompts instead of complex JSON-LD

# Default option
llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")  # 4GB, balanced
```

**Reality Check:**
- ✅ **Cloud models**: Excellent at complex JSON-LD with schema.org vocabulary
- ⚠️ **Local models**: Good for simple facts, struggle with complex structured formats
- 💡 **Best approach**: Use cloud models for production knowledge graphs, local models for simple extraction

### 2. Document Processing

**Small Documents (<8000 chars):**
- Use `length="standard"` or `length="detailed"`
- Single extraction pass is sufficient

**Large Documents (>8000 chars):**
- Automatic chunking with overlap
- Use `--iterate=2` for better coverage
- Consider `length="brief"` to avoid token limits

**Domain-Specific Text:**
- Always use `domain_focus` parameter
- Examples: "technology", "business", "medical", "legal", "academic"

### 3. Output Format Selection

**Choose JSON-LD when:**
- Building semantic web applications
- Need W3C standard compliance
- Integrating with knowledge graph databases
- Require full metadata and context

**Choose Triples when:**
- Building graph databases (Neo4j, etc.)
- Need simple SUBJECT PREDICATE OBJECT format
- Implementing reasoning systems
- Want human-readable relationships

**Choose Minified when:**
- Storage space is limited
- Network transmission efficiency matters
- Building APIs with compact responses

### 4. Quality Optimization

**For Higher Quality:**
```bash
# Use iterative refinement (finds missing entities)
python -m abstractllm.apps.extractor doc.txt --iterate 3 --length detailed

# Use better models
python -m abstractllm.apps.extractor doc.txt --provider openai --model gpt-4o-mini
```

**For Faster Processing:**
```bash
# Use brief extraction with fast models
python -m abstractllm.apps.extractor doc.txt --length brief --fast
```

## Schema & Ontology

BasicExtractor uses standard vocabularies for maximum compatibility:

### Entity Types (schema.org)
- `s:Person` - People by name
- `s:Organization` - Companies, institutions
- `s:SoftwareApplication` - Software, frameworks, tools
- `s:Place` - Locations, venues
- `s:Product` - Products, services
- `s:Event` - Events, meetings
- `sk:Concept` - Abstract concepts, technologies

### Relationship Types
- `creates` - Authorship, development
- `uses` - Utilization, dependency
- `supports` - Support, enablement
- `partOf` - Structural relationships
- `integrates` - Integration, compatibility
- `provides` - Service provision
- `memberOf` - Organizational membership

### Entity Structure
```json
{
  "@id": "e:entity_name",
  "@type": "s:EntityType",
  "s:name": "Human readable name",
  "s:description": "Brief description",
  "confidence": 0.95
}
```

### Relationship Structure
```json
{
  "@id": "r:1",
  "@type": "s:Relationship",
  "s:name": "relationship_type",
  "s:about": {"@id": "e:subject_entity"},
  "s:object": {"@id": "e:object_entity"},
  "s:description": "Relationship description",
  "confidence": 0.90,
  "strength": 0.85
}
```

## JSON Self-Correction

BasicExtractor includes automatic JSON self-correction that attempts to fix malformed LLM responses before giving up:

**Automatic Recovery**:
- Extracts JSON from text with extra content
- Fixes common formatting issues (trailing commas, quote problems)
- Repairs truncated JSON by adding missing braces
- Creates minimal valid structure from partial content

**In Action**:
```
⚠️  JSON parsing failed: Expecting ',' delimiter: line 3 column 45
ℹ️  JSON self-fix: Fixed common formatting issues
✅ JSON self-correction successful! Extracted 3 entities and 2 relationships
```

This significantly improves extraction reliability, especially with local models that may produce slightly malformed JSON.

## Troubleshooting

### Common Issues

**"Failed to initialize default Ollama model"**
```bash
# Install Ollama and download model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama serve
```

**Empty extraction results**
- Try a different model: `--provider openai --model gpt-4o-mini`
- Increase extraction length: `--length detailed`
- Add domain focus: `--focus technology`

**JSON parsing errors**
- Automatic self-correction handles most cases
- If persistent, try a more capable model
- Check model output with `--verbose` flag

**Large file processing slow**
- Use `--fast` flag for speed
- Use `length=brief` for fewer entities
- Consider pre-processing to extract relevant sections

**Poor entity quality**
- Use iterative refinement: `--iterate=2`
- Try more capable models (GPT-4, Claude)
- Add specific domain focus

### Error Messages

**"Chunk size must be at least 1000 characters"**
- Increase `--chunk-size` parameter
- File might be too short for meaningful extraction

**"Iterate cannot exceed 5"**
- Maximum 5 refinement iterations allowed
- Diminishing returns beyond 3 iterations

**"Provider/model required together"**
- Both `--provider` and `--model` must be specified together

## Integration Examples

### Web Application
```python
from flask import Flask, request, jsonify
from abstractllm.processing import BasicExtractor

app = Flask(__name__)
extractor = BasicExtractor()

@app.route('/extract', methods=['POST'])
def extract_knowledge():
    text = request.json.get('text')
    format_type = request.json.get('format', 'jsonld')

    result = extractor.extract(text, output_format=format_type)
    return jsonify(result)
```

### Data Pipeline
```python
import pandas as pd
from abstractllm.processing import BasicExtractor

def process_documents(file_paths):
    extractor = BasicExtractor()
    results = []

    for path in file_paths:
        with open(path, 'r') as f:
            text = f.read()

        kg = extractor.extract(
            text,
            length="standard",
            output_format="triples"
        )

        results.append({
            'file': path,
            'entities': len(kg.get('entities', {})),
            'triples': kg.get('simple_triples', [])
        })

    return pd.DataFrame(results)
```

## Performance Characteristics

### Speed Benchmarks (Approximate)

| Model | Text Length | Extraction Time | Quality |
|-------|-------------|-----------------|---------|
| `qwen3:4b-instruct-2507-q4_K_M` | 1000 chars | 3-7 seconds | Good |
| `qwen3-coder:30b` | 1000 chars | 8-15 seconds | Excellent |
| `gpt-oss:120b` | 1000 chars | 10-20 seconds | Best |
| `gpt-4o-mini` | 1000 chars | 3-8 seconds | Best |
| `claude-3-5-haiku` | 1000 chars | 2-6 seconds | Best |

### Memory Usage

- **BasicExtractor**: ~50MB base memory
- **Local models**: +2-8GB depending on model size
- **Large documents**: Chunking prevents memory issues

BasicExtractor is designed for production use with built-in error handling, retry logic, and efficient processing of documents from small snippets to large reports.