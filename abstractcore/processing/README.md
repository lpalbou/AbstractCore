# Processing Module

## Purpose and Architecture Position

The Processing Module provides a collection of high-performance, production-ready text processing capabilities built on AbstractCore's provider-agnostic foundation. These processors implement sophisticated NLP operations using structured output, automatic chunking, and intelligent retry strategies to deliver reliable, enterprise-grade text analysis.

**Architecture Position**: The Processing Module sits between the Core Layer (providers, LLM abstraction) and the Applications Layer (CLI apps, server endpoints). It transforms raw LLM capabilities into specialized, reusable processing components that can be integrated into any application or workflow.

## Quick Reference

### Processor Selection Guide

| Task | Processor | Default Model | Speed | Use Case |
|------|-----------|---------------|-------|----------|
| **Summarization** | BasicSummarizer | gemma3:1b-it-qat | ⚡⚡⚡⚡ | Documents, articles, chat history |
| **Knowledge Extraction** | BasicExtractor | qwen3:4b-instruct | ⚡⚡⚡ | Entities, relationships, JSON-LD |
| **Quality Evaluation** | BasicJudge | qwen3:4b-instruct | ⚡⚡⚡ | Code reviews, content assessment |
| **Intent Analysis** | BasicIntentAnalyzer | gemma3:1b-it-qat | ⚡⚡⚡⚡ | User messages, support tickets |
| **Research** | BasicDeepSearch | qwen3:4b-instruct | ⚡⚡ | Web research, fact-finding |

### Key Features by Processor

| Processor | Input | Output | Key Features |
|-----------|-------|--------|--------------|
| **BasicSummarizer** | Text | Summary + key points | 6 styles, 4 lengths, focus-based, chunking |
| **BasicExtractor** | Text | JSON-LD graph | Schema.org, 9 entity types, 21 relationships |
| **BasicJudge** | Text | Scored assessment | 9 criteria, 1-5 scale, chain-of-thought |
| **BasicIntentAnalyzer** | Text | Intent analysis | 17 intent types, deception detection, 3 depths |
| **BasicDeepSearch** | Query | Research report | 4-stage pipeline, web search, citations |

### Model Recommendations by Task

| Complexity | Summarizer | Extractor | Judge | Intent | Search |
|------------|-----------|-----------|-------|--------|--------|
| **Fast/Simple** | gemma3:1b | qwen3:4b | qwen3:4b | gemma3:1b | qwen3:4b |
| **Balanced** | qwen3:4b | qwen3:4b | qwen3:4b | qwen3:4b | qwen3:4b |
| **Premium** | qwen3-coder:30b | qwen3-coder:30b | gpt-4o-mini | claude-haiku | gpt-4o-mini |

## Common Tasks

- **How do I summarize a document?** → See [BasicSummarizer](#1-basicsummarizer)
- **How do I extract knowledge?** → See [BasicExtractor](#2-basicextractor)
- **How do I evaluate quality?** → See [BasicJudge](#3-basicjudge)
- **How do I analyze intent?** → See [BasicIntentAnalyzer](#4-basicintentanalyzer)
- **How do I research a topic?** → See [BasicDeepSearch](#5-basicdeepsearch)
- **How do I use custom models?** → See [Pattern 2: Custom LLM Configuration](#pattern-2-custom-llm-configuration)
- **How do I chain processors?** → See [Pattern 3: Pipeline Integration](#pattern-3-pipeline-integration)
- **How do I handle errors?** → See [Pattern 5: Error Handling](#pattern-5-error-handling)

```
┌─────────────────────────────────────────────┐
│         Applications Layer (apps/)          │
│  (CLI interfaces, user-facing tools)        │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│       Processing Module (processing/)       │  ← You are here
│  (Specialized NLP operations & workflows)   │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│        Core Layer (core/, providers/)       │
│  (LLM providers, factory, base interfaces)  │
└─────────────────────────────────────────────┘
```

## Component Structure

The Processing Module contains 5 specialized processor classes:

```
processing/
├── __init__.py                    # Module exports
├── basic_summarizer.py           # Text summarization (584 lines)
├── basic_extractor.py            # Knowledge extraction (1204 lines)
├── basic_judge.py                # Quality evaluation (690 lines)
├── basic_intent.py               # Intent analysis (691 lines)
└── basic_deepsearch.py           # Autonomous research (2600+ lines)
```

### Design Principles

All processors follow consistent design patterns:
- **Provider-agnostic**: Work with any AbstractCore provider (Ollama, OpenAI, Anthropic, HuggingFace, MLX, etc.)
- **Structured output**: Use Pydantic models for type-safe, validated responses
- **Automatic chunking**: Handle long documents through intelligent text splitting
- **Token-aware processing**: Estimate tokens using centralized TokenUtils
- **Retry strategies**: Built-in error handling with FeedbackRetry (3 attempts)
- **Structured logging**: Comprehensive logging via centralized logger
- **Sensible defaults**: Work out-of-the-box with optimal configurations

---

## Detailed Component Documentation

### 1. BasicSummarizer

**Purpose**: High-quality document summarization with configurable styles and lengths.

**File**: `basic_summarizer.py` (584 lines)

**Key Features**:
- 6 presentation styles (structured, narrative, objective, analytical, executive, conversational)
- 4 target lengths (brief, standard, detailed, comprehensive)
- Automatic map-reduce for long documents with overlapping chunks
- Focus-based summarization for specific aspects
- Chat history summarization with context preservation
- Word count tracking and compression ratio calculation

**Public API**:

```python
from abstractcore.processing import BasicSummarizer, SummaryStyle, SummaryLength
from abstractcore import create_llm

# Initialize with default model (gemma3:1b-it-qat)
summarizer = BasicSummarizer()

# Or with custom LLM
llm = create_llm("openai", model="gpt-4o-mini")
summarizer = BasicSummarizer(
    llm=llm,
    max_chunk_size=8000,
    max_tokens=32000,
    max_output_tokens=8000,
    timeout=None
)

# Generate summary
result = summarizer.summarize(
    text="Long document text...",
    focus="business implications",  # Optional focus area
    style=SummaryStyle.EXECUTIVE,
    length=SummaryLength.DETAILED
)

# Access structured output
print(result.summary)
print(f"Confidence: {result.confidence}")
print(f"Key points: {len(result.key_points)}")
print(f"Compression: {(1 - result.word_count_summary/result.word_count_original)*100:.1f}%")

# Specialized: Chat history summarization
result = summarizer.summarize_chat_history(
    messages=[{"role": "user", "content": "..."}, ...],
    preserve_recent=6,  # Keep last 6 messages intact
    focus="key decisions"
)
```

**Output Schema** (`SummaryOutput`):
```python
{
    "summary": str,                      # Main summary text
    "key_points": List[str],             # 3-5 key insights (max 8)
    "confidence": float,                 # 0-1 confidence score
    "focus_alignment": float,            # 0-1 focus adherence
    "word_count_original": int,          # Original text words
    "word_count_summary": int            # Summary words
}
```

**Performance Benchmarks** (from documentation):
- gemma3:1b-it-qat: 29s, 95% confidence, cost-effective
- qwen3-coder:30b: 119s, 98% confidence, premium quality
- GPT-4o-mini: Variable, high cost per request

**Chunking Strategy**:
- Default chunk size: 8000 characters with 200-character overlap
- Sentence boundary breaking for natural splits
- Map-reduce: Individual chunk summaries → combined final summary

→ See [utils/token_utils.py](../utils/README.md#token-counting) for token estimation implementation

---

### 2. BasicExtractor

**Purpose**: Semantic knowledge extraction with structured JSON-LD output following schema.org vocabulary.

**File**: `basic_extractor.py` (1204 lines)

**Key Features**:
- Multiple output formats: JSON-LD, RDF triples, minified JSON-LD
- Schema.org vocabulary compliance
- Generic relationship IDs (r:1, r:2, etc.) with s:name for types
- Entity types: Person, Organization, Event, Goal, Task, Place, Product, SoftwareApplication, Concept
- Relationship types: 21 predefined predicates (is, uses, creates, part_of, requires, etc.)
- Automatic deduplication and dangling reference cleanup
- Refinement capability for iterative improvements
- Graceful fallback with self-healing JSON parsing

**Public API**:

```python
from abstractcore.processing import BasicExtractor
from abstractcore import create_llm

# Initialize with default model (qwen3:4b-instruct-2507-q4_K_M)
extractor = BasicExtractor()

# Or with custom LLM
llm = create_llm("openai", model="gpt-4o-mini")
extractor = BasicExtractor(
    llm=llm,
    max_chunk_size=8000,
    max_tokens=32000,
    max_output_tokens=8000
)

# Extract entities and relationships
result = extractor.extract(
    text="OpenAI created GPT-4 and Microsoft Copilot uses GPT-4",
    domain_focus="AI technology",  # Optional focus
    length="standard",             # brief, standard, detailed, comprehensive
    output_format="jsonld"          # jsonld, triples, jsonld_minified
)

# Access JSON-LD output
print(result["@context"])
print(f"Entities: {len([e for e in result['@graph'] if e['@id'].startswith('e:')])}")
print(f"Relationships: {len([e for e in result['@graph'] if e['@id'].startswith('r:')])}")

# Refine previous extraction
refined = extractor.refine_extraction(
    text="Same text",
    previous_extraction=result,
    domain_focus="organizational relationships",
    length="detailed"
)
```

**Output Formats**:

1. **JSON-LD** (default):
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
      "@id": "e:openai",
      "@type": "s:Organization",
      "s:name": "OpenAI",
      "s:description": "AI company that created GPT-4",
      "confidence": 0.95
    },
    {
      "@id": "r:1",
      "@type": "s:Relationship",
      "s:name": "creates",
      "s:about": {"@id": "e:openai"},
      "s:object": {"@id": "e:gpt4"},
      "s:description": "OpenAI created GPT-4",
      "confidence": 0.95,
      "strength": 0.9
    }
  ]
}
```

2. **RDF Triples**:
```python
{
  "format": "triples",
  "triples": [
    {
      "subject": "e:openai",
      "subject_name": "OpenAI",
      "predicate": "creates",
      "object": "e:gpt4",
      "object_name": "GPT-4",
      "triple_text": "OpenAI creates GPT-4",
      "confidence": 0.95,
      "strength": 0.9
    }
  ],
  "simple_triples": ["OpenAI creates GPT-4", ...],
  "entities": {...},
  "statistics": {
    "entities_count": 4,
    "relationships_count": 3
  }
}
```

3. **Minified JSON-LD**:
```python
{
  "format": "jsonld_minified",
  "data": "{\"@context\":{...}}",  # Minified JSON string
  "entities_count": 4,
  "relationships_count": 3
}
```

**Entity Types**:
- `sk:Concept` - Abstract concepts, technologies
- `s:Person` - People by name
- `s:Organization` - Companies, institutions
- `s:Event` - Events, meetings, conferences
- `s:Goal` - Abstract goals, objectives
- `s:Task` - Abstract tasks, actions
- `s:Place` - Locations
- `s:Product` - Products, services
- `s:SoftwareApplication` - Software, tools, frameworks

**Relationship Types**:
- Identity: is, is_not
- Composition: part_of, constitutes
- Transformation: transforms
- Action: creates, develops, uses, provides, requires
- Interaction: integrates, works_with, compatible_with
- Communication: describes, mentions
- Influence: supports, discourages, enables, disables
- Modeling: models
- Spatiotemporal: occurs_in, occurs_when

**Validation & Self-Healing**:
- Normalizes string references to object references: `"e:id"` → `{"@id": "e:id"}`
- Removes dangling references (relationships to non-existent entities)
- Fuzzy matching by entity name for salvage attempts
- JSON self-correction via `fix_json()` utility
- Fallback to empty graph on catastrophic failures

---

### 3. BasicJudge

**Purpose**: Objective LLM-as-a-judge evaluation with structured assessments and chain-of-thought reasoning.

**File**: `basic_judge.py` (690 lines)

**Key Features**:
- 9 evaluation criteria (clarity, simplicity, actionability, soundness, innovation, effectiveness, relevance, completeness, coherence)
- 1-5 scoring rubric with explicit thresholds
- Chain-of-thought reasoning for transparency
- Configurable criteria focus
- File and multi-file evaluation support
- Global assessment generation for multi-file analysis
- Low temperature (0.1) for consistent scoring
- Constructive skepticism approach

**Public API**:

```python
from abstractcore.processing import BasicJudge, JudgmentCriteria
from abstractcore import create_llm

# Initialize with default model (qwen3:4b-instruct-2507-q4_K_M)
judge = BasicJudge(temperature=0.1, debug=False)

# Or with custom LLM
llm = create_llm("openai", model="gpt-4o-mini", temperature=0.1)
judge = BasicJudge(
    llm=llm,
    temperature=0.1,
    max_tokens=32000,
    max_output_tokens=8000,
    debug=False,
    timeout=None
)

# Evaluate content
result = judge.evaluate(
    content="The code is well-structured and solves the problem elegantly",
    context="code review",
    criteria=JudgmentCriteria(
        is_clear=True,
        is_simple=True,
        is_actionable=True,
        is_sound=True,
        is_innovative=True,
        is_working=True,
        is_relevant=True,
        is_complete=True,
        is_coherent=True
    ),
    focus="technical accuracy, maintainability",  # Optional focus areas
    reference="Expected solution approach...",     # Optional reference
    include_criteria=True                           # Include criteria explanations
)

# Evaluate single file
result = judge.evaluate_files(
    file_paths="document.py",
    context="code review",
    max_file_size=1000000  # 1MB limit
)

# Evaluate multiple files (returns global + individual assessments)
results = judge.evaluate_files(
    file_paths=["file1.py", "file2.py", "file3.py"],
    context="codebase review",
    exclude_global=False  # Include global assessment
)
# Access: results["global"], results["files"]

# Or get only individual assessments
results = judge.evaluate_files(
    file_paths=["file1.py", "file2.py"],
    exclude_global=True  # Skip global assessment
)
# Access: results[0], results[1]
```

**Output Schema** (`Assessment`):
```python
{
    "overall_score": int,              # 1-5 overall assessment
    "judge_summary": str,              # Judge's experiential note
    "source_reference": str,           # What was assessed

    # Individual criterion scores (1-5 or None if not evaluated)
    "clarity_score": Optional[int],
    "simplicity_score": Optional[int],
    "actionability_score": Optional[int],
    "soundness_score": Optional[int],
    "innovation_score": Optional[int],
    "effectiveness_score": Optional[int],
    "relevance_score": Optional[int],
    "completeness_score": Optional[int],
    "coherence_score": Optional[int],

    # Detailed evaluation
    "strengths": List[str],            # Key strengths
    "weaknesses": List[str],           # Areas for improvement
    "actionable_feedback": List[str],  # Specific recommendations

    # Reasoning & metadata
    "reasoning": str,                  # Chain-of-thought analysis
    "evaluation_context": str,         # Evaluation context
    "criteria_used": List[str],        # Criteria names
    "evaluation_criteria_details": Optional[str]  # Detailed explanations
}
```

**Scoring Rubric**:
- **Score 5**: Exceptional - Exceeds expectations in this dimension
- **Score 4**: Good - Meets expectations well with minor room for improvement
- **Score 3**: Adequate - Meets basic expectations but has notable areas for improvement
- **Score 2**: Poor - Falls short of expectations with significant issues
- **Score 1**: Very Poor - Fails to meet basic standards in this dimension

**Evaluation Criteria** (`JudgmentCriteria`):
```python
JudgmentCriteria(
    is_clear=True,       # Clarity and understandability
    is_simple=True,      # Appropriate simplicity vs complexity
    is_actionable=True,  # Actionable insights provided
    is_sound=True,       # Logical soundness and reasoning
    is_innovative=True,  # Creativity and novel thinking
    is_working=True,     # Solves intended problem
    is_relevant=True,    # Relevance to context
    is_complete=True,    # Completeness of coverage
    is_coherent=True     # Logical flow and consistency
)
```

**Multi-File Assessment**:
- Sequential processing to avoid context overflow
- Global assessment synthesizes patterns across files
- Score distribution analysis (5, 4, 3, 2, 1 counts)
- Common strengths/weaknesses identification
- Aggregate recommendations
- Configurable: include or exclude global assessment

---

### 4. BasicIntentAnalyzer

**Purpose**: Identify and analyze intents, motivations, and goals behind text with psychological depth.

**File**: `basic_intent.py` (691 lines)

**Key Features**:
- 17 intent types based on psychological research (information seeking/sharing, problem solving, persuasion, emotional expression, face-saving, deception, trust building, etc.)
- 3 analysis depths (surface, underlying, comprehensive)
- 4 context types (standalone, conversational, document, interactive)
- Integrated deception analysis based on psychological markers
- Multi-layered intent detection (primary + up to 3 secondary)
- Intent complexity scoring and urgency levels
- Conversation history intent analysis by participant
- Response approach suggestions

**Public API**:

```python
from abstractcore.processing import BasicIntentAnalyzer, IntentContext, IntentDepth
from abstractcore import create_llm

# Initialize with default model (gemma3:1b-it-qat)
analyzer = BasicIntentAnalyzer(debug=False)

# Or with custom LLM
llm = create_llm("openai", model="gpt-4o-mini")
analyzer = BasicIntentAnalyzer(
    llm=llm,
    max_chunk_size=8000,
    max_tokens=32000,
    max_output_tokens=8000,
    debug=False
)

# Analyze text intent
result = analyzer.analyze_intent(
    text="I was wondering if you could help me understand...",
    context_type=IntentContext.CONVERSATIONAL,
    depth=IntentDepth.UNDERLYING,
    focus="management concerns"  # Optional focus
)

# Access primary intent
print(f"Intent: {result.primary_intent.intent_type.value}")
print(f"Goal: {result.primary_intent.underlying_goal}")
print(f"Confidence: {result.primary_intent.confidence}")
print(f"Urgency: {result.primary_intent.urgency_level}")

# Deception analysis (always included)
deception = result.primary_intent.deception_analysis
print(f"Deception likelihood: {deception.deception_likelihood}")
print(f"Narrative consistency: {deception.narrative_consistency}")
print(f"Linguistic markers: {deception.linguistic_markers}")

# Secondary intents
for intent in result.secondary_intents:
    print(f"- {intent.intent_type.value}: {intent.description}")

# Analyze conversation by participant
results = analyzer.analyze_conversation_intents(
    messages=[
        {"role": "user", "content": "I need help..."},
        {"role": "assistant", "content": "I can help..."}
    ],
    focus_participant="user",  # Optional: focus on specific participant
    depth=IntentDepth.UNDERLYING
)

# Access by participant
user_intent = results["user"]
assistant_intent = results["assistant"]
```

**Intent Types** (`IntentType` enum):
- `INFORMATION_SEEKING` - Asking questions, requesting data
- `INFORMATION_SHARING` - Providing facts, explanations
- `PROBLEM_SOLVING` - Seeking or offering solutions
- `DECISION_MAKING` - Evaluating options, making choices
- `PERSUASION` - Convincing, influencing opinions
- `CLARIFICATION` - Seeking or providing clarity
- `EMOTIONAL_EXPRESSION` - Expressing feelings, reactions
- `RELATIONSHIP_BUILDING` - Social connection, rapport
- `INSTRUCTION_GIVING` - Teaching, directing actions
- `VALIDATION_SEEKING` - Seeking approval, confirmation
- `FACE_SAVING` - Protecting self-image, avoiding embarrassment
- `BLAME_DEFLECTION` - Redirecting responsibility externally
- `POWER_ASSERTION` - Establishing dominance or authority
- `EMPATHY_SEEKING` - Seeking understanding and emotional support
- `CONFLICT_AVOIDANCE` - Preventing or minimizing confrontation
- `TRUST_BUILDING` - Establishing or maintaining credibility
- `DECEPTION` - Intentional misdirection or false information

**Analysis Depths** (`IntentDepth` enum):
- `SURFACE` - Obvious, stated intentions
- `UNDERLYING` - Hidden motivations and goals
- `COMPREHENSIVE` - Full analysis including subconscious drivers

**Context Types** (`IntentContext` enum):
- `STANDALONE` - Single message/text analysis
- `CONVERSATIONAL` - Part of ongoing dialogue
- `DOCUMENT` - Formal document or article
- `INTERACTIVE` - Real-time interaction context

**Output Schema** (`IntentAnalysisOutput`):
```python
{
    "primary_intent": IdentifiedIntent,       # Most prominent intent
    "secondary_intents": List[IdentifiedIntent],  # Additional intents (max 3)
    "intent_complexity": float,                # 0-1 complexity score
    "contextual_factors": List[str],           # Context elements (max 5)
    "suggested_response_approach": str,        # How to respond
    "overall_confidence": float,               # 0-1 confidence
    "word_count_analyzed": int,                # Words analyzed
    "analysis_depth": IntentDepth,             # Depth used
    "context_type": IntentContext              # Context used
}
```

**Deception Analysis** (`DeceptionIndicators`):
```python
{
    "deception_likelihood": float,           # 0-1 likelihood
    "narrative_consistency": float,          # 0-1 internal consistency
    "linguistic_markers": List[str],         # Specific indicators (max 5)
    "temporal_coherence": float,             # 0-1 timing consistency
    "emotional_congruence": float,           # 0-1 emotion-content alignment
    "deception_evidence": List[str],         # Contradictions, deflections (max 3)
    "authenticity_evidence": List[str]       # Consistency, accountability (max 3)
}
```

**Deception Detection Principles**:
1. Check for internal contradictions
2. Evaluate motivations: "What do they gain by lying?"
3. Assess explanation complexity (overly complex = suspicious)
4. Analyze timeline consistency
5. Look for blame shifting patterns
6. Verify claims that can't be easily checked
7. Assess emotional congruence with content

---

### 5. BasicDeepSearch

**Purpose**: Autonomous research agent with multi-stage pipeline for comprehensive web-based research.

**File**: `basic_deepsearch.py` (2600+ lines)

**Key Features**:
- 4-stage pipeline: Planning, Question Development, Web Exploration, Report Generation
- Parallel web exploration for speed and breadth
- Structured report generation with citations
- Source management with strict limits and deduplication
- Citation validation and enforcement
- Reflexive refinement mode for iterative improvement
- Verification and fact-checking capabilities
- Configurable search depth (brief, standard, comprehensive)
- Full-text extraction support
- DuckDuckGo and Serper.dev integration

**Public API**:

```python
from abstractcore.processing import BasicDeepSearch
from abstractcore import create_llm

# Initialize with default model (qwen3:4b-instruct-2507-q4_K_M)
searcher = BasicDeepSearch(
    full_text_extraction=False,
    reflexive_mode=False,
    max_reflexive_iterations=2,
    max_parallel_searches=5,
    temperature=0.1,
    debug_mode=False
)

# Or with custom LLM
llm = create_llm("openai", model="gpt-4o-mini")
searcher = BasicDeepSearch(
    llm=llm,
    max_tokens=32000,
    max_output_tokens=8000,
    max_parallel_searches=5,
    full_text_extraction=True,  # Extract full page content
    reflexive_mode=True,         # Enable iterative refinement
    max_reflexive_iterations=2,
    temperature=0.1,
    debug_mode=True
)

# Basic research
report = searcher.research(
    query="What are the latest developments in quantum computing?",
    max_sources=15,
    search_depth="standard",
    include_verification=True,
    output_format="structured"
)

# Advanced research with focus areas
report = searcher.research(
    query="Impact of AI on healthcare",
    focus_areas=["medical diagnosis", "drug discovery", "patient care"],
    max_sources=20,
    search_depth="comprehensive",
    include_verification=True,
    output_format="narrative"
)

# Access structured report
print(report.title)
print(report.executive_summary)
for finding in report.key_findings:
    print(f"- {finding}")
print(f"\nSources: {len(report.sources)}")
print(f"Methodology: {report.methodology}")
```

**Search Depths**:
- `brief`: 3 sub-tasks, ~5 minutes, quick overview
- `standard`: 5 sub-tasks, ~10 minutes, balanced coverage (default)
- `comprehensive`: 8 sub-tasks, ~20 minutes, exhaustive analysis

**Output Formats**:
- `structured`: Full ResearchReport with all sections
- `narrative`: Flowing narrative style
- `executive`: Business-focused executive summary

**4-Stage Pipeline**:

1. **Planning** (`_create_research_plan`):
   - Query type detection (person, concept, technology, location, organization)
   - Automatic focus area generation
   - Theme identification and sub-task decomposition
   - Priority assignment (1=essential, 2=important, 3=supplementary)
   - Time estimation

2. **Question Development** (`_develop_search_questions`):
   - Generate 3-5 specific search queries per sub-task
   - Theme-based query clustering
   - Diversity and complementarity checks
   - Query optimization for search engines

3. **Web Exploration** (`_explore_web_sources`):
   - Parallel search execution (configurable concurrency)
   - Source manager with strict limits and deduplication
   - Relevance assessment for each source
   - Optional full-text content extraction
   - URL and title-based deduplication

4. **Report Generation** (`_generate_report`):
   - Structured synthesis of findings
   - Citation integration and validation
   - Key findings extraction
   - Methodology documentation
   - Limitations assessment

**Optional Stages**:
- **Verification** (`_verify_report`): Fact-checking and citation validation
- **Reflexive Refinement** (`_reflexive_refinement`): Iterative gap analysis and improvement

**Source Management** (`SourceManager`):
- Strict max_sources limit enforcement
- URL-based deduplication
- Title-based similarity detection
- Remaining capacity tracking
- Sequential addition with overflow prevention

**Citation Validation** (`CitationValidator`):
```python
{
    "citations_found": int,           # Number of citations detected
    "factual_sentences": int,         # Sentences with factual claims
    "citation_ratio": float,          # Citations per factual sentence
    "is_adequately_cited": bool,      # ≥50% citation coverage
    "uncited_sources": List[str],     # Sources not cited
    "cited_sources": List[str]        # Sources cited
}
```

**Output Schema** (`ResearchReport`):
```python
{
    "title": str,                     # Report title
    "executive_summary": str,         # Brief summary
    "key_findings": List[str],        # Main insights
    "detailed_analysis": str,         # Full analysis
    "conclusions": str,               # Implications
    "sources": List[Dict],            # URLs, titles, scores
    "methodology": str,               # Research approach
    "limitations": str                # Caveats
}
```

**Reflexive Mode**:
- Analyzes report for gaps and weaknesses
- Identifies missing information
- Generates supplementary queries
- Conducts additional research
- Integrates new findings
- Iterates up to `max_reflexive_iterations`

**Debug Mode**:
- Tracks all generated queries
- Records all URLs discovered
- Logs relevance assessments
- Shows accepted/rejected sources
- Prints comprehensive summary

---

## Usage Patterns

### Pattern 1: Quick Start with Defaults

All processors work out-of-the-box with sensible defaults:

```python
from abstractcore.processing import (
    BasicSummarizer, BasicExtractor, BasicJudge,
    BasicIntentAnalyzer, BasicDeepSearch
)

# No configuration needed - uses default models
summarizer = BasicSummarizer()
extractor = BasicExtractor()
judge = BasicJudge()
analyzer = BasicIntentAnalyzer()
searcher = BasicDeepSearch()

# Start using immediately
result = summarizer.summarize("Your text here")
```

**Default Models**:
- Summarizer: `gemma3:1b-it-qat` (fast, cost-effective)
- Extractor: `qwen3:4b-instruct-2507-q4_K_M` (balanced)
- Judge: `qwen3:4b-instruct-2507-q4_K_M` (consistent scoring)
- Intent Analyzer: `gemma3:1b-it-qat` (quick analysis)
- Deep Search: `qwen3:4b-instruct-2507-q4_K_M` (research quality)

### Pattern 2: Custom LLM Configuration

Use any provider with custom models:

```python
from abstractcore import create_llm
from abstractcore.processing import BasicSummarizer

# OpenAI
llm = create_llm("openai", model="gpt-4o-mini", max_tokens=32000)
summarizer = BasicSummarizer(llm, max_chunk_size=15000)

# Anthropic
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
extractor = BasicExtractor(llm)

# HuggingFace GGUF
llm = create_llm("huggingface", model="unsloth/Qwen3-4B-Instruct-2507-GGUF")
judge = BasicJudge(llm, temperature=0.1)

# MLX (Apple Silicon)
llm = create_llm("mlx", model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
analyzer = BasicIntentAnalyzer(llm)

# LMStudio
llm = create_llm("lmstudio", model="qwen3-coder-30b-instruct", base_url="http://localhost:1234/v1")
searcher = BasicDeepSearch(llm)
```

### Pattern 3: Pipeline Integration

Chain processors for complex workflows:

```python
from abstractcore.processing import BasicExtractor, BasicJudge

# Extract knowledge
extractor = BasicExtractor()
knowledge = extractor.extract(document_text, output_format="jsonld")

# Evaluate extraction quality
judge = BasicJudge()
assessment = judge.evaluate(
    content=str(knowledge),
    context="knowledge graph extraction",
    focus="entity completeness, relationship accuracy"
)

if assessment["overall_score"] >= 4:
    # High quality - proceed with refinement
    refined = extractor.refine_extraction(
        text=document_text,
        previous_extraction=knowledge,
        length="detailed"
    )
```

### Pattern 4: Batch Processing

Process multiple documents efficiently:

```python
from abstractcore.processing import BasicSummarizer
from pathlib import Path

summarizer = BasicSummarizer()
results = []

for file_path in Path("documents/").glob("*.txt"):
    with open(file_path, "r") as f:
        text = f.read()

    result = summarizer.summarize(text, length=SummaryLength.BRIEF)
    results.append({
        "file": file_path.name,
        "summary": result.summary,
        "confidence": result.confidence
    })

# Save batch results
import json
with open("summaries.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Pattern 5: Error Handling

All processors include built-in retry strategies:

```python
from abstractcore.processing import BasicSummarizer
from abstractcore.structured.retry import FeedbackRetry

summarizer = BasicSummarizer()

# Default: 3 retry attempts with exponential backoff
# Customizable via retry_strategy attribute
summarizer.retry_strategy = FeedbackRetry(max_attempts=5)

try:
    result = summarizer.summarize(text)
except Exception as e:
    print(f"Processing failed after retries: {e}")
    # Handle gracefully
```

---

## Integration Points

### With Core Layer

Processors depend on:
- `AbstractCoreInterface` - Provider abstraction
- `create_llm()` - LLM factory
- `FeedbackRetry` - Retry strategy
- `TokenUtils` - Token estimation
- Structured logging - `get_logger()`

### With Applications Layer

Applications use processors via:
- Direct instantiation: `BasicSummarizer()`
- CLI wrappers: `apps/summarizer.py`
- Server endpoints: `server/app.py`
- Configuration system: `config/`

### With External Systems

Processors integrate with:
- Web search engines (DuckDuckGo, Serper.dev) via `common_tools`
- File systems for document reading
- JSON/JSON-LD output for downstream systems
- Pydantic models for type safety

---

## Best Practices

### 1. Choose the Right Model

Match model capabilities to task complexity:

| Task | Recommended Models | Rationale |
|------|-------------------|-----------|
| Summarization | gemma3:1b-it-qat, qwen3:4b | Fast, cost-effective, instruction-tuned |
| Extraction | qwen3:4b, qwen3-coder:30b | Structured output quality |
| Judgment | qwen3:4b (temp=0.1), gpt-4o-mini | Consistency over creativity |
| Intent Analysis | gemma3:1b-it-qat, claude-haiku | Speed + psychological reasoning |
| Deep Search | qwen3:4b, gpt-4o-mini, claude-haiku | Research + synthesis |

### 2. Optimize Chunking

For long documents:
```python
# Conservative chunking (safer for smaller context models)
processor = BasicSummarizer(max_chunk_size=6000, max_tokens=16000)

# Aggressive chunking (for large context models)
processor = BasicSummarizer(max_chunk_size=15000, max_tokens=100000)
```

### 3. Use Structured Logging

Processors emit structured logs for observability:
```python
from abstractcore.utils.structured_logging import configure_logging
import logging

# Configure console + file logging
configure_logging(
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    log_dir="logs",
    console_json=False,
    file_json=True
)

# Logs include: text_length, chunk_count, model used, timing, etc.
```

### 4. Handle Timeouts

Set appropriate timeouts for network operations:
```python
# For deep search with web requests
searcher = BasicDeepSearch(timeout=30.0)  # 30 seconds per request

# For local models (no network)
summarizer = BasicSummarizer(timeout=None)  # Unlimited
```

### 5. Leverage Focus Parameters

All processors support focus for targeted analysis:
```python
# Summarization
summarizer.summarize(text, focus="financial implications")

# Extraction
extractor.extract(text, domain_focus="organizational relationships")

# Judgment
judge.evaluate(content, focus="technical accuracy, security")

# Intent analysis
analyzer.analyze_intent(text, focus="emotional drivers")
```

---

## Common Pitfalls

### 1. Context Overflow

**Problem**: Passing text larger than model context window.

**Solution**: Use automatic chunking or preprocess:
```python
# Let processor handle chunking
result = summarizer.summarize(very_long_text)  # Automatic map-reduce

# Or preprocess manually
from abstractcore.utils.token_utils import TokenUtils
tokens = TokenUtils.estimate_tokens(text, model_name="qwen3:4b")
if tokens > 16000:
    # Split into smaller documents
    pass
```

### 2. Temperature Misconfiguration

**Problem**: Using high temperature for tasks requiring consistency.

**Solution**: Use low temperature for deterministic tasks:
```python
# For judgment (consistency critical)
judge = BasicJudge(temperature=0.1)

# For creative summarization (higher temp acceptable)
summarizer = BasicSummarizer()  # Uses default temp from provider
```

### 3. Ignoring Structured Output

**Problem**: Extracting data from text responses instead of using Pydantic models.

**Solution**: Always use the structured output:
```python
# Wrong
response = llm.generate(prompt)
text = response.content
# Parse text manually...

# Right
result = summarizer.summarize(text)
print(result.summary)  # Direct attribute access
print(result.confidence)  # Type-safe
```

### 4. Missing Retry Handling

**Problem**: Not accounting for transient failures.

**Solution**: Trust built-in retry, but handle final failures:
```python
try:
    result = extractor.extract(text)
except Exception as e:
    logger.error(f"Extraction failed after retries: {e}")
    # Implement fallback logic
    result = create_empty_extraction()
```

### 5. Ignoring Confidence Scores

**Problem**: Treating all outputs as equally reliable.

**Solution**: Check confidence and adjust downstream logic:
```python
result = summarizer.summarize(text)

if result.confidence < 0.7:
    logger.warning(f"Low confidence summary: {result.confidence}")
    # Flag for human review or retry with different model

if result.focus_alignment < 0.6:
    logger.warning("Summary doesn't address focus area well")
    # Retry with adjusted prompt or different approach
```

---

## Testing Strategy

### Unit Tests

Test individual processor methods:
```python
import pytest
from abstractcore.processing import BasicSummarizer

def test_summarizer_initialization():
    summarizer = BasicSummarizer()
    assert summarizer is not None
    assert summarizer.llm is not None

def test_summarizer_single_chunk():
    summarizer = BasicSummarizer()
    text = "Short text to summarize."
    result = summarizer.summarize(text)

    assert result.summary
    assert 0 <= result.confidence <= 1
    assert len(result.key_points) >= 3
    assert result.word_count_original > 0
```

### Integration Tests

Test with real models:
```python
from abstractcore import create_llm
from abstractcore.processing import BasicExtractor

def test_extractor_with_real_model():
    llm = create_llm("huggingface", model="unsloth/Qwen3-4B-Instruct-2507-GGUF")
    extractor = BasicExtractor(llm)

    text = "OpenAI created GPT-4 which is used by many developers"
    result = extractor.extract(text)

    # Verify JSON-LD structure
    assert "@context" in result
    assert "@graph" in result

    entities = [e for e in result["@graph"] if e["@id"].startswith("e:")]
    assert len(entities) > 0

def test_chunking_behavior():
    summarizer = BasicSummarizer(max_chunk_size=100)
    long_text = "word " * 1000  # 1000 words

    result = summarizer.summarize(long_text)
    assert result.summary
    # Should have used chunking
```

### Performance Tests

Benchmark processor performance:
```python
import time

def test_summarizer_performance():
    summarizer = BasicSummarizer()
    text = "Sample text " * 500  # Medium-length document

    start = time.time()
    result = summarizer.summarize(text)
    elapsed = time.time() - start

    print(f"Summarization took {elapsed:.2f}s")
    print(f"Confidence: {result.confidence}")

    # Performance assertions
    assert elapsed < 60.0  # Should complete in under 60s
    assert result.confidence > 0.7  # Should be reasonably confident
```

### Regression Tests

Ensure consistent behavior across updates:
```python
def test_extraction_format_stability():
    """Ensure JSON-LD format remains consistent"""
    extractor = BasicExtractor()
    text = "Test company created Test product"
    result = extractor.extract(text)

    # Verify schema hasn't changed
    assert result["@context"]["s"] == "https://schema.org/"
    assert result["@context"]["e"] == "http://example.org/entity/"
    assert result["@context"]["r"] == "http://example.org/relation/"

    # Verify entity structure
    for entity in result["@graph"]:
        if entity["@id"].startswith("e:"):
            assert "@type" in entity
            assert "s:name" in entity
            assert "confidence" in entity
```

---

## Public API Summary

### Exports

```python
from abstractcore.processing import (
    # Summarization
    BasicSummarizer,
    SummaryStyle,
    SummaryLength,
    SummaryOutput,

    # Extraction
    BasicExtractor,

    # Evaluation
    BasicJudge,
    JudgmentCriteria,
    Assessment,
    create_judge,

    # Intent Analysis
    BasicIntentAnalyzer,
    IntentType,
    IntentDepth,
    IntentContext,
    IntentAnalysisOutput,
    IdentifiedIntent,
    DeceptionIndicators,

    # Deep Search
    BasicDeepSearch,
    ResearchReport,
    ResearchPlan,
    ResearchSubTask
)
```

### Processor Classes

| Class | Purpose | Default Model | Key Method |
|-------|---------|---------------|------------|
| `BasicSummarizer` | Document summarization | gemma3:1b-it-qat | `summarize()` |
| `BasicExtractor` | Knowledge extraction | qwen3:4b-instruct | `extract()` |
| `BasicJudge` | Quality evaluation | qwen3:4b-instruct | `evaluate()` |
| `BasicIntentAnalyzer` | Intent analysis | gemma3:1b-it-qat | `analyze_intent()` |
| `BasicDeepSearch` | Autonomous research | qwen3:4b-instruct | `research()` |

### Common Parameters

All processors support:
- `llm`: AbstractCoreInterface (custom LLM instance)
- `max_tokens`: int (context window size)
- `max_output_tokens`: int (max output generation)
- `timeout`: Optional[float] (HTTP timeout)

---

## Performance Considerations

### Model Selection Impact

| Model | Speed | Quality | Memory | Use Case |
|-------|-------|---------|--------|----------|
| gemma3:1b-it-qat | ⚡⚡⚡⚡ | ⭐⭐⭐ | 2GB | Quick summaries, intent analysis |
| qwen3:4b | ⚡⚡⚡ | ⭐⭐⭐⭐ | 4GB | Balanced performance |
| qwen3-coder:30b | ⚡⚡ | ⭐⭐⭐⭐⭐ | 32GB | Premium extraction, deep search |
| gpt-4o-mini | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Cloud | Cloud-based, reliable |
| claude-haiku | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Cloud | Fast cloud, good reasoning |

### Chunking Overhead

| Document Size | Chunking | Overhead | Strategy |
|---------------|----------|----------|----------|
| < 8,000 chars | No | None | Single pass |
| 8,000-24,000 | Yes | ~30% | Map-reduce (2-3 chunks) |
| 24,000-100,000 | Yes | ~60% | Map-reduce (3-12 chunks) |
| > 100,000 | Yes | ~100% | Aggressive chunking |

### Optimization Tips

1. **Batch Similar Tasks**: Reuse processor instances
   ```python
   summarizer = BasicSummarizer()  # Initialize once
   for doc in documents:
       result = summarizer.summarize(doc)  # Reuse
   ```

2. **Adjust Chunk Size**: Larger chunks = fewer LLM calls
   ```python
   # For large context models
   summarizer = BasicSummarizer(max_chunk_size=15000)
   ```

3. **Parallel Processing**: Process independent documents in parallel
   ```python
   from concurrent.futures import ThreadPoolExecutor

   with ThreadPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(summarizer.summarize, documents))
   ```

4. **Use Appropriate Depths**: Don't over-analyze
   ```python
   # Quick intent check
   analyzer.analyze_intent(text, depth=IntentDepth.SURFACE)

   # Full psychological analysis
   analyzer.analyze_intent(text, depth=IntentDepth.COMPREHENSIVE)
   ```

---

## Future Enhancements

Potential improvements under consideration:
- Streaming support for real-time processing
- Caching layer for repeated analyses
- Multi-language support with automatic detection
- Custom model fine-tuning utilities
- Advanced citation management (BibTeX, etc.)
- Visual output formats (diagrams, charts)
- Distributed processing for massive documents
- Interactive refinement UI

---

For detailed implementation examples, see the respective processor files and test suites in `tests/processing/`.

## Related Modules

**Direct dependencies**:
- [`core/`](../core/README.md) - LLM creation and generation
- [`providers/`](../providers/README.md) - Provider implementations
- [`structured/`](../structured/README.md) - Response models for structured outputs
- [`media/`](../media/README.md) - Media processing for multimodal content
- [`tools/`](../tools/README.md) - Web search and utility tools
- [`utils/`](../utils/README.md) - Web utilities, logging, validation
- [`exceptions/`](../exceptions/README.md) - Error handling

**Used by**:
- [`apps/`](../apps/README.md) - High-level application integrations
- [`server/`](../server/README.md) - Processing API endpoints

**Related systems**:
- [`config/`](../config/README.md) - Processing configuration defaults
- [`embeddings/`](../embeddings/README.md) - Document embedding and search
