# Apps Module

## Purpose and Architecture Position

The Apps Module provides high-level, user-facing command-line applications that wrap AbstractCore's processing capabilities into ready-to-use tools. These applications demonstrate best practices for building CLI interfaces and serve as reference implementations for custom applications.

**Architecture Position**: The Apps Module sits at the highest level of the AbstractCore stack, providing the end-user interface for text processing operations. It bridges the gap between AbstractCore's Python API and shell/terminal usage.

```
┌─────────────────────────────────────────────┐
│          Apps Module (apps/)                │  ← You are here
│   (CLI applications, user interfaces)       │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│       Processing Module (processing/)       │
│  (Specialized NLP operations & workflows)   │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│        Core Layer (core/, providers/)       │
│  (LLM providers, factory, base interfaces)  │
└─────────────────────────────────────────────┘
```

## Component Structure

The Apps Module contains 7 files implementing the CLI framework and applications:

```
apps/
├── __init__.py               # Module exports (31 bytes)
├── __main__.py              # CLI launcher (1,974 bytes)
├── app_config_utils.py      # Configuration utilities (890 bytes)
├── summarizer.py            # Summarization app (14,669 bytes)
├── extractor.py             # Extraction app (23,749 bytes)
├── judge.py                 # Evaluation app (23,169 bytes)
├── intent.py                # Intent analysis app (22,813 bytes)
└── deepsearch.py            # Deep search app (23,191 bytes)
```

### Design Principles

All apps follow consistent patterns:
- **Unified CLI Interface**: Launch via `python -m abstractcore.apps <app>`
- **Argparse Integration**: Rich help text and parameter validation
- **Configuration Management**: Integration with AbstractCore config system
- **File I/O Handling**: Robust file reading with encoding fallbacks
- **Error Reporting**: Clear, actionable error messages
- **Provider Flexibility**: Custom provider/model selection
- **Output Formatting**: Human-readable and machine-readable formats

---

## Detailed Component Documentation

### 1. CLI Launcher (`__main__.py`)

**Purpose**: Entry point for all CLI applications with unified interface.

**File**: `__main__.py` (59 lines)

**Usage**:
```bash
# Show available apps
python -m abstractcore.apps

# Launch specific app
python -m abstractcore.apps summarizer document.txt
python -m abstractcore.apps extractor report.txt
python -m abstractcore.apps judge essay.txt
python -m abstractcore.apps intent "I need help with this"
python -m abstractcore.apps deepsearch "quantum computing advances"

# Get app-specific help
python -m abstractcore.apps summarizer --help
```

**Available Apps**:
- `summarizer` - Document summarization tool
- `extractor` - Entity and relationship extraction tool
- `judge` - Text evaluation and scoring tool
- `intent` - Intent analysis and motivation identification tool
- `deepsearch` - Autonomous web research tool

**Architecture**:
```python
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    app_name = sys.argv[1]

    if app_name == "summarizer":
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove app name
        from .summarizer import main as summarizer_main
        summarizer_main()
    # ... similar for other apps
```

---

### 2. Configuration Utilities (`app_config_utils.py`)

**Purpose**: Shared configuration management for all apps.

**File**: `app_config_utils.py` (19 lines)

**Public API**:

```python
def get_app_defaults(app_name: str) -> tuple[str, str]:
    """Get default provider and model for an app.

    Args:
        app_name: Name of the app ('summarizer', 'extractor', 'judge', 'cli')

    Returns:
        tuple: (provider, model)
    """
```

**Configuration Flow**:
1. Try to load from AbstractCore config system
2. Fallback to hardcoded defaults if config unavailable

**Default Models** (as of 2025-11-06):
```python
{
    'summarizer': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
    'extractor': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
    'judge': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
    'cli': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
}
```

**Customization**:
```bash
# Set app-specific default
abstractcore --set-app-default summarizer openai gpt-4o-mini

# Check current default
abstractcore --get-app-default summarizer
```

---

### 3. Summarizer App (`summarizer.py`)

**Purpose**: Command-line interface for document summarization.

**File**: `summarizer.py` (467 lines)

**Key Features**:
- 6 summarization styles (structured, narrative, objective, analytical, executive, conversational)
- 4 target lengths (brief, standard, detailed, comprehensive)
- Focus-based summarization
- Custom chunking configuration
- Multiple output formats
- Verbose progress reporting

**Usage**:

```bash
# Basic summarization
python -m abstractcore.apps summarizer document.txt

# With style and length
python -m abstractcore.apps summarizer report.pdf --style executive --length brief

# With focus area
python -m abstractcore.apps summarizer data.md --focus "technical details"

# Save to file
python -m abstractcore.apps summarizer large.txt --output summary.txt

# Custom chunking
python -m abstractcore.apps summarizer huge.txt --chunk-size 15000

# Custom provider and model
python -m abstractcore.apps summarizer doc.txt \
  --provider openai \
  --model gpt-4o-mini \
  --max-tokens 32000

# Verbose mode
python -m abstractcore.apps summarizer document.txt --verbose
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | **required** | Path to file to summarize |
| `--style` | string | structured | Summary style |
| `--length` | string | standard | Target length |
| `--focus` | string | null | Specific focus area |
| `--output` | string | null | Output file path |
| `--chunk-size` | int | 8000 | Chunk size (characters) |
| `--provider` | string | null | LLM provider |
| `--model` | string | null | LLM model |
| `--max-tokens` | int | 32000 | Context window size |
| `--max-output-tokens` | int | 8000 | Max output generation |
| `--verbose` | flag | false | Show detailed progress |

**Output Format**:
```
SUMMARY
==================================================
The document discusses the implementation of quantum
error correction techniques...

KEY POINTS
--------------------
1. Quantum error correction is essential for...
2. Surface codes provide the most practical...
3. Recent advances show 10x improvement in...

METADATA
---------------
Confidence Score: 0.95
Focus Alignment: 0.88
Original Words: 15,234
Summary Words: 425
Compression: 97.2%
```

**Supported File Types**:
- Text: `.txt`, `.md`, `.py`, `.js`, `.html`, `.json`, `.csv`
- Documents: `.pdf`, `.docx`
- Any UTF-8 encoded text file

**Implementation Highlights**:

```python
def read_file_content(file_path: str) -> str:
    """Robust file reading with encoding fallbacks"""
    try:
        # UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin1, cp1252, iso-8859-1
        for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # Final fallback: binary read with errors ignored
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')

def format_summary_output(result) -> str:
    """Format summary result for display"""
    output_lines = []
    output_lines.append("SUMMARY")
    output_lines.append("=" * 50)
    output_lines.append(result.summary)
    # ... format key points, metadata
    compression_ratio = (1 - result.word_count_summary / max(result.word_count_original, 1)) * 100
    output_lines.append(f"Compression: {compression_ratio:.1f}%")
    return "\n".join(output_lines)
```

---

### 4. Extractor App (`extractor.py`)

**Purpose**: Command-line interface for knowledge extraction with JSON-LD output.

**File**: `extractor.py` (748 lines)

**Key Features**:
- Multiple output formats (JSON-LD, RDF triples, minified JSON-LD)
- Entity and relationship extraction
- Refinement mode for iterative improvements
- Domain-focused extraction
- Configurable extraction depth
- Pretty-printed JSON output
- Statistics reporting

**Usage**:

```bash
# Basic extraction (default JSON-LD)
python -m abstractcore.apps extractor document.txt

# RDF triples format
python -m abstractcore.apps extractor report.txt --format triples

# Minified JSON-LD
python -m abstractcore.apps extractor data.md --format jsonld-minified

# With domain focus
python -m abstractcore.apps extractor paper.txt --domain "machine learning"

# Extraction depth
python -m abstractcore.apps extractor doc.txt --length comprehensive

# Save to file
python -m abstractcore.apps extractor input.txt --output knowledge.jsonld

# Refinement mode
python -m abstractcore.apps extractor doc.txt \
  --refine previous_extraction.jsonld \
  --length detailed

# Custom provider
python -m abstractcore.apps extractor doc.txt \
  --provider openai \
  --model gpt-4o-mini

# Verbose mode
python -m abstractcore.apps extractor document.txt --verbose
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | **required** | Path to file to extract from |
| `--format` | string | jsonld | Output format |
| `--domain` | string | null | Domain focus area |
| `--length` | string | standard | Extraction depth |
| `--output` | string | null | Output file path |
| `--refine` | string | null | Previous extraction to refine |
| `--chunk-size` | int | 8000 | Chunk size (characters) |
| `--provider` | string | null | LLM provider |
| `--model` | string | null | LLM model |
| `--max-tokens` | int | 32000 | Context window size |
| `--max-output-tokens` | int | 8000 | Max output generation |
| `--verbose` | flag | false | Show detailed progress |

**Output Formats**:

1. **JSON-LD** (default):
```json
{
  "@context": {
    "s": "https://schema.org/",
    "e": "http://example.org/entity/",
    "r": "http://example.org/relation/"
  },
  "@graph": [
    {
      "@id": "e:openai",
      "@type": "s:Organization",
      "s:name": "OpenAI",
      "confidence": 0.95
    },
    {
      "@id": "r:1",
      "@type": "s:Relationship",
      "s:name": "creates",
      "s:about": {"@id": "e:openai"},
      "s:object": {"@id": "e:gpt4"}
    }
  ]
}
```

2. **RDF Triples**:
```
TRIPLE 1
--------
Subject: OpenAI (e:openai)
Predicate: creates
Object: GPT-4 (e:gpt4)
Confidence: 0.95
Description: OpenAI created GPT-4

SIMPLE FORMAT
-------------
OpenAI creates GPT-4
Microsoft develops Copilot
```

3. **Minified JSON-LD**:
```json
{
  "format": "jsonld_minified",
  "data": "{\"@context\":{...}}",
  "entities_count": 5,
  "relationships_count": 8
}
```

**Statistics Output**:
```
EXTRACTION COMPLETE
===================
Entities: 12
Relationships: 18
Processing Time: 23.4s
```

---

### 5. Judge App (`judge.py`)

**Purpose**: Command-line interface for objective quality evaluation.

**File**: `judge.py` (745 lines)

**Key Features**:
- 9 evaluation criteria (clarity, simplicity, actionability, soundness, innovation, effectiveness, relevance, completeness, coherence)
- 1-5 scoring rubric
- Single and multi-file evaluation
- Custom criteria selection
- Focus areas
- Reference comparison
- Detailed criterion explanations
- Global assessment for multiple files

**Usage**:

```bash
# Basic evaluation
python -m abstractcore.apps judge document.txt

# With context
python -m abstractcore.apps judge code.py --context "code review"

# Custom criteria
python -m abstractcore.apps judge essay.txt \
  --criteria clarity,soundness,coherence

# With focus areas
python -m abstractcore.apps judge report.txt \
  --focus "technical accuracy,completeness"

# Reference comparison
python -m abstractcore.apps judge solution.py \
  --reference expected_solution.py

# Multiple files
python -m abstractcore.apps judge file1.py file2.py file3.py \
  --context "codebase review"

# Include detailed criteria explanations
python -m abstractcore.apps judge doc.txt --include-criteria

# Save to file
python -m abstractcore.apps judge document.txt --output assessment.json

# Custom provider
python -m abstractcore.apps judge doc.txt \
  --provider openai \
  --model gpt-4o-mini \
  --temperature 0.1

# Debug mode
python -m abstractcore.apps judge document.txt --debug
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_paths` | string(s) | **required** | Path(s) to file(s) to evaluate |
| `--context` | string | general | Evaluation context |
| `--criteria` | string | all | Comma-separated criteria list |
| `--focus` | string | null | Specific focus areas |
| `--reference` | string | null | Reference file for comparison |
| `--include-criteria` | flag | false | Include detailed criteria explanations |
| `--output` | string | null | Output file path (JSON) |
| `--provider` | string | null | LLM provider |
| `--model` | string | null | LLM model |
| `--temperature` | float | 0.1 | Sampling temperature |
| `--max-tokens` | int | 32000 | Context window size |
| `--max-output-tokens` | int | 8000 | Max output generation |
| `--debug` | flag | false | Show raw LLM responses |

**Evaluation Criteria**:
- `clarity` - Clarity and understandability
- `simplicity` - Appropriate simplicity vs complexity
- `actionability` - Provides actionable insights
- `soundness` - Logical soundness and reasoning
- `innovation` - Creativity and novel thinking
- `effectiveness` - Solves the intended problem
- `relevance` - Relevance to context/requirements
- `completeness` - Comprehensive coverage
- `coherence` - Logical flow and consistency

**Output Format (Single File)**:
```
ASSESSMENT REPORT
=================

Overall Score: 4/5

JUDGE'S SUMMARY
---------------
I was asked to evaluate a technical document in the context
of code review. The code demonstrates strong clarity and
soundness but has room for improvement in completeness.

SOURCE
------
File: document.py (context: code review)

INDIVIDUAL CRITERIA
-------------------
Clarity: 5/5 (Exceptional)
Simplicity: 4/5 (Good)
Soundness: 4/5 (Good)
Completeness: 3/5 (Adequate)

STRENGTHS
---------
- Clear variable naming and function structure
- Well-documented with docstrings
- Logical flow throughout the codebase

WEAKNESSES
----------
- Missing error handling in several functions
- Test coverage is incomplete
- Documentation lacks usage examples

ACTIONABLE FEEDBACK
-------------------
- Add try-except blocks for external API calls
- Increase test coverage to at least 80%
- Add usage examples to README
```

**Output Format (Multiple Files)**:
```
GLOBAL ASSESSMENT
=================

Overall Score: 4/5
Files Evaluated: 3

[Individual file assessments follow...]

GLOBAL SUMMARY
--------------
I conducted a global assessment synthesizing evaluations
of 3 files in the context of codebase review...

COMMON PATTERNS
---------------
Strengths:
- Consistent code style across files
- Good use of type hints

Weaknesses:
- Inconsistent error handling
- Missing integration tests
```

---

### 6. Intent App (`intent.py`)

**Purpose**: Command-line interface for intent analysis and motivation detection.

**File**: `intent.py` (732 lines)

**Key Features**:
- 17 intent types (psychological research-based)
- 3 analysis depths (surface, underlying, comprehensive)
- 4 context types (standalone, conversational, document, interactive)
- Integrated deception analysis
- Focus areas
- File and text input
- JSON output option

**Usage**:

```bash
# Analyze text directly
python -m abstractcore.apps intent "I was wondering if you could help me"

# Analyze from file
python -m abstractcore.apps intent --file message.txt

# With depth
python -m abstractcore.apps intent "I need assistance" --depth comprehensive

# With context type
python -m abstractcore.apps intent --file chat.txt \
  --context conversational

# With focus
python -m abstractcore.apps intent "Help me understand" \
  --focus "emotional drivers"

# JSON output
python -m abstractcore.apps intent "I'm concerned about this" \
  --output analysis.json

# Custom provider
python -m abstractcore.apps intent "I need help" \
  --provider openai \
  --model gpt-4o-mini

# Debug mode
python -m abstractcore.apps intent "Hello" --debug
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | null | Text to analyze (if --file not used) |
| `--file` | string | null | Path to file to analyze |
| `--depth` | string | underlying | Analysis depth |
| `--context` | string | standalone | Context type |
| `--focus` | string | null | Specific focus area |
| `--output` | string | null | Output file path (JSON) |
| `--provider` | string | null | LLM provider |
| `--model` | string | null | LLM model |
| `--max-tokens` | int | 32000 | Context window size |
| `--max-output-tokens` | int | 8000 | Max output generation |
| `--debug` | flag | false | Show raw LLM responses |

**Analysis Depths**:
- `surface` - Obvious, stated intentions
- `underlying` - Hidden motivations and goals (default)
- `comprehensive` - Full psychological analysis

**Context Types**:
- `standalone` - Single message/text (default)
- `conversational` - Part of dialogue
- `document` - Formal document
- `interactive` - Real-time interaction

**Intent Types** (17 total):
- INFORMATION_SEEKING, INFORMATION_SHARING
- PROBLEM_SOLVING, DECISION_MAKING
- PERSUASION, CLARIFICATION
- EMOTIONAL_EXPRESSION, RELATIONSHIP_BUILDING
- INSTRUCTION_GIVING, VALIDATION_SEEKING
- FACE_SAVING, BLAME_DEFLECTION
- POWER_ASSERTION, EMPATHY_SEEKING
- CONFLICT_AVOIDANCE, TRUST_BUILDING
- DECEPTION

**Output Format**:
```
INTENT ANALYSIS REPORT
======================

PRIMARY INTENT
--------------
Type: INFORMATION_SEEKING
Confidence: 0.92
Urgency: 0.75

Description:
The person is seeking information about a specific topic
with genuine curiosity and desire to learn.

Underlying Goal:
To understand the technical details required to make
an informed decision.

Emotional Undertone:
Curious and slightly uncertain, seeking reassurance
through knowledge acquisition.

DECEPTION ANALYSIS
------------------
Likelihood: 0.15 (Low)
Narrative Consistency: 0.88 (High)
Temporal Coherence: 0.92 (High)
Emotional Congruence: 0.85 (High)

Linguistic Markers:
- Direct questioning style
- Honest admission of knowledge gap

Authenticity Evidence:
- Consistent timeline references
- No blame deflection patterns

SECONDARY INTENTS
-----------------
1. VALIDATION_SEEKING (Confidence: 0.68)
   Seeking confirmation that their approach is sound

2. TRUST_BUILDING (Confidence: 0.55)
   Attempting to establish credibility

CONTEXTUAL FACTORS
------------------
- Professional context
- Time-sensitive decision making
- Prior knowledge gaps acknowledged

SUGGESTED RESPONSE APPROACH
----------------------------
Provide detailed, factual information with clear
explanations. Address both the explicit question
and the underlying need for validation. Use
reassuring tone while maintaining accuracy.
```

---

### 7. Deep Search App (`deepsearch.py`)

**Purpose**: Command-line interface for autonomous web research.

**File**: `deepsearch.py` (749 lines)

**Key Features**:
- Multi-stage research pipeline (Planning, Questions, Exploration, Report)
- Parallel web searches
- Configurable search depth
- Focus areas
- Source limits
- Verification mode
- Reflexive refinement
- Full-text extraction
- Multiple output formats

**Usage**:

```bash
# Basic research
python -m abstractcore.apps deepsearch "quantum computing advances"

# With search depth
python -m abstractcore.apps deepsearch "AI in healthcare" \
  --depth comprehensive

# With focus areas
python -m abstractcore.apps deepsearch "sustainable energy" \
  --focus "solar,wind,battery storage"

# Limit sources
python -m abstractcore.apps deepsearch "climate change" \
  --max-sources 20

# Enable verification
python -m abstractcore.apps deepsearch "vaccine efficacy" \
  --verify

# Enable reflexive mode
python -m abstractcore.apps deepsearch "quantum error correction" \
  --reflexive \
  --max-iterations 3

# Full-text extraction
python -m abstractcore.apps deepsearch "protein folding" \
  --full-text

# Save to file
python -m abstractcore.apps deepsearch "neural networks" \
  --output research_report.json

# Output format
python -m abstractcore.apps deepsearch "blockchain" \
  --format narrative

# Custom provider
python -m abstractcore.apps deepsearch "machine learning" \
  --provider openai \
  --model gpt-4o-mini

# Debug mode
python -m abstractcore.apps deepsearch "quantum computing" --debug
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | **required** | Research query/question |
| `--depth` | string | standard | Search depth |
| `--focus` | string | null | Comma-separated focus areas |
| `--max-sources` | int | 15 | Maximum sources to gather |
| `--verify` | flag | false | Enable fact-checking |
| `--reflexive` | flag | false | Enable reflexive refinement |
| `--max-iterations` | int | 2 | Max reflexive iterations |
| `--full-text` | flag | false | Extract full page content |
| `--output` | string | null | Output file path (JSON) |
| `--format` | string | structured | Output format |
| `--provider` | string | null | LLM provider |
| `--model` | string | null | LLM model |
| `--max-tokens` | int | 32000 | Context window size |
| `--max-output-tokens` | int | 8000 | Max output generation |
| `--debug` | flag | false | Show comprehensive debugging |

**Search Depths**:
- `brief` - 3 sub-tasks, ~5 minutes, quick overview
- `standard` - 5 sub-tasks, ~10 minutes, balanced (default)
- `comprehensive` - 8 sub-tasks, ~20 minutes, exhaustive

**Output Formats**:
- `structured` - Full ResearchReport (default)
- `narrative` - Flowing narrative style
- `executive` - Executive summary format

**Output Format**:
```
RESEARCH REPORT
===============

Title: Latest Advances in Quantum Computing

EXECUTIVE SUMMARY
-----------------
Recent developments in quantum computing have shown
significant progress in error correction, with surface
codes demonstrating...

KEY FINDINGS
------------
1. IBM achieved quantum advantage with 127-qubit processor
2. Google's Willow chip reduced error rates by 10x
3. Microsoft invested $1B in quantum computing research

DETAILED ANALYSIS
-----------------
[Comprehensive analysis with citations]

The field of quantum computing has experienced rapid
advancement in 2024-2025, particularly in the areas of...

According to [IBM Quantum Blog], their latest Eagle
processor demonstrates unprecedented stability...

CONCLUSIONS
-----------
The evidence suggests quantum computing is approaching
practical utility for specific applications...

SOURCES (15)
------------
1. IBM Quantum Blog - "127-Qubit Processor Breakthrough"
   URL: https://...
   Relevance: 0.95

2. Nature - "Quantum Error Correction Advances"
   URL: https://...
   Relevance: 0.92

METHODOLOGY
-----------
Research conducted using parallel web searches across
5 themes with 15 specific queries. Sources evaluated
for credibility and recency...

LIMITATIONS
-----------
- Research limited to publicly available sources
- Technical details may require expert validation
- Rapidly evolving field; some findings may be superseded

RESEARCH METADATA
-----------------
Duration: 127.5 seconds
Sources Probed: 32
Sources Selected: 15
Verification: Enabled
Citation Coverage: 87%
```

---

## App Framework Architecture

### Common Patterns

All apps follow a consistent structure:

```python
#!/usr/bin/env python3
"""
App Name - Description

Usage:
    python -m abstractcore.apps.appname <args> [options]

Options:
    [Standard options documentation]
"""

import argparse
import sys
from pathlib import Path

# 1. Import processing module
from ..processing import ProcessorClass

# 2. Import LLM factory
from ..core.factory import create_llm

# 3. Define helper functions
def read_file_content(file_path: str) -> str:
    """Robust file reading"""
    ...

def format_output(result) -> str:
    """Format result for display"""
    ...

# 4. Main function with argparse
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('input', ...)
    parser.add_argument('--option', ...)
    args = parser.parse_args()

    # 5. Initialize processor
    if args.provider and args.model:
        llm = create_llm(args.provider, model=args.model)
        processor = ProcessorClass(llm)
    else:
        processor = ProcessorClass()  # Use default

    # 6. Process input
    result = processor.process(...)

    # 7. Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(format_output(result))
    else:
        print(format_output(result))

if __name__ == "__main__":
    main()
```

### File Reading Strategy

All apps use robust file reading with encoding fallbacks:

```python
def read_file_content(file_path: str) -> str:
    """Read file with encoding fallbacks"""
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Try UTF-8 first (most common)
    try:
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback encodings
        for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path_obj, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Final fallback: binary read with errors ignored
        with open(file_path_obj, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
```

### Error Handling

All apps implement consistent error handling:

```python
try:
    result = processor.process(input_text)
except FileNotFoundError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"Invalid input: {e}", file=sys.stderr)
    sys.exit(2)
except Exception as e:
    if args.debug:
        raise  # Show full traceback in debug mode
    print(f"Processing failed: {e}", file=sys.stderr)
    sys.exit(3)
```

---

## Integration Points

### With Processing Module

Apps depend on:
- Processor classes: `BasicSummarizer`, `BasicExtractor`, `BasicJudge`, `BasicIntentAnalyzer`, `BasicDeepSearch`
- Enum types: `SummaryStyle`, `SummaryLength`, `IntentDepth`, etc.
- Output models: `SummaryOutput`, `Assessment`, `IntentAnalysisOutput`, `ResearchReport`

### With Core Layer

Apps use:
- `create_llm()` - LLM factory for custom provider/model selection
- Configuration system - For default provider/model settings
- Structured logging - For verbose mode output

### With Config System

Apps integrate with configuration:
```bash
# Set app-specific defaults
abstractcore --set-app-default summarizer openai gpt-4o-mini

# Apps automatically use configured defaults
python -m abstractcore.apps summarizer document.txt
```

---

## Best Practices

### 1. Use Appropriate Verbosity

```bash
# Silent mode (errors only)
python -m abstractcore.apps summarizer doc.txt 2>/dev/null

# Normal mode (results only)
python -m abstractcore.apps summarizer doc.txt

# Verbose mode (progress + results)
python -m abstractcore.apps summarizer doc.txt --verbose

# Debug mode (everything)
python -m abstractcore.apps judge doc.txt --debug
```

### 2. Pipe and Redirect

```bash
# Save output
python -m abstractcore.apps summarizer doc.txt > summary.txt

# Process multiple files
for file in documents/*.txt; do
    python -m abstractcore.apps summarizer "$file" --output "${file}.summary"
done

# Chain with other tools
cat document.txt | python -m abstractcore.apps summarizer /dev/stdin
```

### 3. Batch Processing Scripts

```bash
#!/bin/bash
# batch_summarize.sh

FILES="$@"
PROVIDER="openai"
MODEL="gpt-4o-mini"

for file in $FILES; do
    echo "Processing: $file"
    python -m abstractcore.apps summarizer "$file" \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --output "${file}.summary" \
        --verbose
done

echo "Batch processing complete"
```

### 4. Configuration Management

```bash
# Set provider defaults once
abstractcore --set-app-default summarizer openai gpt-4o-mini
abstractcore --set-app-default extractor openai gpt-4o-mini
abstractcore --set-app-default judge openai gpt-4o-mini

# Then use apps without specifying provider
python -m abstractcore.apps summarizer doc.txt
python -m abstractcore.apps extractor doc.txt
python -m abstractcore.apps judge doc.txt
```

### 5. Error Handling in Scripts

```bash
#!/bin/bash
# robust_processing.sh

set -e  # Exit on error

if ! python -m abstractcore.apps summarizer "$1" --output summary.txt; then
    echo "Summarization failed!" >&2
    exit 1
fi

if ! python -m abstractcore.apps judge summary.txt --context "summary quality"; then
    echo "Quality check failed!" >&2
    exit 1
fi

echo "Processing successful"
```

---

## Common Pitfalls

### 1. Missing Provider/Model Pair

**Problem**: Providing `--provider` without `--model` or vice versa.

**Solution**: Always provide both or neither:
```bash
# Wrong
python -m abstractcore.apps summarizer doc.txt --provider openai

# Right
python -m abstractcore.apps summarizer doc.txt --provider openai --model gpt-4o-mini

# Or use default
python -m abstractcore.apps summarizer doc.txt
```

### 2. File Encoding Issues

**Problem**: Non-UTF-8 files fail to read.

**Solution**: Apps handle this automatically with encoding fallbacks. No user action needed.

### 3. Large File Processing

**Problem**: Very large files may exceed context window.

**Solution**: Use chunking parameters:
```bash
python -m abstractcore.apps summarizer huge.txt \
    --chunk-size 6000 \
    --max-tokens 16000
```

### 4. Output Redirection Issues

**Problem**: Progress messages mixed with output.

**Solution**: Apps write results to stdout, progress to stderr:
```bash
# Redirect only results
python -m abstractcore.apps summarizer doc.txt > summary.txt

# Suppress progress (keep results)
python -m abstractcore.apps summarizer doc.txt 2>/dev/null
```

---

## Testing Strategy

### Unit Tests

Test argument parsing and formatting:
```python
import pytest
from abstractcore.apps.summarizer import parse_style, parse_length, format_summary_output

def test_parse_style():
    assert parse_style("executive") == SummaryStyle.EXECUTIVE
    with pytest.raises(ValueError):
        parse_style("invalid_style")

def test_format_output():
    result = SummaryOutput(
        summary="Test summary",
        key_points=["Point 1", "Point 2"],
        confidence=0.95,
        focus_alignment=0.88,
        word_count_original=100,
        word_count_summary=20
    )
    output = format_summary_output(result)
    assert "SUMMARY" in output
    assert "KEY POINTS" in output
    assert "Compression: 80.0%" in output
```

### Integration Tests

Test with actual files:
```python
def test_summarizer_app_with_file(tmp_path):
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document with some content.")

    # Run app
    import subprocess
    result = subprocess.run(
        ["python", "-m", "abstractcore.apps.summarizer", str(test_file)],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "SUMMARY" in result.stdout
    assert "KEY POINTS" in result.stdout
```

### End-to-End Tests

Test complete workflows:
```bash
#!/bin/bash
# test_workflow.sh

set -e

# Create test document
echo "Quantum computing uses quantum bits..." > test_doc.txt

# Test summarization
python -m abstractcore.apps summarizer test_doc.txt --output summary.txt
[ -f summary.txt ] || exit 1

# Test extraction
python -m abstractcore.apps extractor test_doc.txt --output knowledge.jsonld
[ -f knowledge.jsonld ] || exit 1

# Test evaluation
python -m abstractcore.apps judge summary.txt --context "summary quality"

# Cleanup
rm test_doc.txt summary.txt knowledge.jsonld

echo "All tests passed!"
```

---

## Public API Summary

### CLI Commands

```bash
# Launcher
python -m abstractcore.apps <app_name> [args] [options]

# Individual apps
python -m abstractcore.apps summarizer <file> [options]
python -m abstractcore.apps extractor <file> [options]
python -m abstractcore.apps judge <file(s)> [options]
python -m abstractcore.apps intent <text|--file> [options]
python -m abstractcore.apps deepsearch <query> [options]
```

### Common Options

All apps support:
- `--provider <provider>` - Custom LLM provider
- `--model <model>` - Custom LLM model
- `--max-tokens <n>` - Context window size
- `--max-output-tokens <n>` - Max output generation
- `--output <file>` - Save results to file
- `--verbose` - Show detailed progress
- `--debug` - Show debugging information
- `--help` - Show help message

---

## Future Enhancements

Planned improvements:
- Interactive mode with prompts
- Watch mode for file changes
- Batch processing built-in
- Configuration file support (YAML/JSON)
- Progress bars for long operations
- Parallel processing of multiple files
- Plugin system for custom processors
- Web UI alongside CLI

---

For detailed implementation, see individual app files in `/Users/albou/projects/abstractcore/abstractcore/apps/`.

## Related Modules

**Direct dependencies**:
- [`core/`](../core/README.md) - LLM creation and generation
- [`processing/`](../processing/README.md) - Summarizer, extractor, judge processors
- [`structured/`](../structured/README.md) - Application response models
- [`media/`](../media/README.md) - Document and media processing
- [`config/`](../config/README.md) - Application-specific defaults
- [`exceptions/`](../exceptions/README.md) - Application error handling

**May use**:
- [`providers/`](../providers/README.md) - Direct provider access
- [`tools/`](../tools/README.md) - Additional utility tools
- [`embeddings/`](../embeddings/README.md) - Semantic search integration
- [`compression/`](../compression/README.md) - Large document handling

**Integration points**:
- [`server/`](../server/README.md) - API endpoints for apps
- [`utils/`](../utils/README.md) - Logging, validation
