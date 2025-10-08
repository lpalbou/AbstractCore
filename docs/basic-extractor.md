# BasicExtractor - Knowledge Graph Extraction

BasicExtractor is a production-ready tool for extracting structured knowledge graphs from text documents. It converts unstructured text into semantic entities and relationships using large language models, outputting clean JSON-LD or RDF triple formats.

## Quick Start

```python
from abstractllm.processing import BasicExtractor

# Initialize with default model (Ollama gemma3:1b-it-qat)
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
# 2. Download model: ollama pull gemma3:1b-it-qat
# 3. Start Ollama service

# Alternative: Use cloud providers
pip install abstractcore[openai,anthropic]
```

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

The `extractor` CLI processes files and outputs knowledge graphs in various formats:

### Basic Usage

```bash
# Extract from file (default: JSON-LD)
python -m abstractllm.apps.extractor document.txt

# Specify output format
python -m abstractllm.apps.extractor document.txt --format=triples

# Save to file
python -m abstractllm.apps.extractor document.txt --output=knowledge_graph.jsonld
```

### Advanced Options

```bash
# Domain-focused extraction
python -m abstractllm.apps.extractor tech_report.txt --focus=technology --length=detailed

# Custom provider and model
python -m abstractllm.apps.extractor document.txt --provider=openai --model=gpt-4o-mini

# Minified output for storage
python -m abstractllm.apps.extractor document.txt --format=json-ld --minified

# Iterative refinement for quality
python -m abstractllm.apps.extractor document.txt --iterate=3 --verbose
```

### CLI Parameters

| Parameter | Description | Choices/Default |
|-----------|-------------|-----------------|
| `file_path` | Input file path | Required |
| `--format` | Output format | `json-ld` (default), `triples`, `json`, `yaml` |
| `--focus` | Domain focus | e.g., "technology", "business", "medical" |
| `--length` | Extraction depth | `brief` (default), `standard`, `detailed`, `comprehensive` |
| `--style` | Extraction style | `structured` (default), `focused`, `minimal`, `comprehensive` |
| `--entity-types` | Entity types to focus on | Comma-separated list |
| `--output` | Output file path | Console if not provided |
| `--provider` | LLM provider | `ollama`, `openai`, `anthropic`, etc. |
| `--model` | LLM model | Provider-specific model name |
| `--minified` | Compact output | Flag (no indentation) |
| `--iterate` | Refinement iterations | 1-5 (default: 1) |
| `--verbose` | Detailed progress | Flag |

### Output Format Examples

**Simple triples (no --verbose):**
```bash
python -m abstractllm.apps.extractor doc.txt --format=triples
# Output:
# Google creates TensorFlow
# Microsoft uses TensorFlow
# OpenAI develops GPT-4
```

**Detailed triples (with --verbose):**
```bash
python -m abstractllm.apps.extractor doc.txt --format=triples --verbose
# Output: JSON with entities, relationships, confidence scores, statistics
```

**Minified JSON-LD:**
```bash
python -m abstractllm.apps.extractor doc.txt --format=json-ld --minified
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
python -m abstractllm.apps.extractor tech.txt --focus=technology --length=detailed --format=triples --verbose
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
  --focus="research" \
  --length=comprehensive \
  --iterate=2 \
  --format=json-ld \
  --output=paper_knowledge_graph.jsonld \
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

# Lightweight option
llm = create_llm("ollama", model="gemma3:1b-it-qat")  # 1GB, very fast
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
python -m abstractllm.apps.extractor doc.txt --iterate=3 --length=detailed

# Use better models
python -m abstractllm.apps.extractor doc.txt --provider=openai --model=gpt-4o-mini
```

**For Faster Processing:**
```bash
# Use brief extraction with fast models
python -m abstractllm.apps.extractor doc.txt --length=brief --fast
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
ollama pull gemma3:1b-it-qat
ollama serve
```

**Empty extraction results**
- Try a different model: `--provider=openai --model=gpt-4o-mini`
- Increase extraction length: `--length=detailed`
- Add domain focus: `--focus=technology`

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
| `gemma3:1b-it-qat` | 1000 chars | 2-5 seconds | Good |
| `qwen3-coder:7b` | 1000 chars | 5-10 seconds | Better |
| `gpt-4o-mini` | 1000 chars | 3-8 seconds | Best |
| `claude-3-5-haiku` | 1000 chars | 2-6 seconds | Best |

### Memory Usage

- **BasicExtractor**: ~50MB base memory
- **Local models**: +2-8GB depending on model size
- **Large documents**: Chunking prevents memory issues

BasicExtractor is designed for production use with built-in error handling, retry logic, and efficient processing of documents from small snippets to large reports.