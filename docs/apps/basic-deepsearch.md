# BasicDeepSearch - Autonomous Research Agent

The BasicDeepSearch application provides autonomous, multi-stage research capabilities that go beyond simple search to deliver comprehensive, synthesized research reports with proper citations.

## Overview

BasicDeepSearch implements a four-stage research pipeline:

1. **Planning**: Decomposes complex research queries into structured sub-tasks with intelligent theme detection
2. **Question Development**: Generates specific, diverse search queries for each sub-task
3. **Web Exploration**: Executes parallel web searches with strict source deduplication and quality filtering
4. **Report Generation**: Synthesizes findings into structured reports with citation validation

## Key Features

### Core Capabilities
- **Autonomous Research**: Minimal human intervention - handles planning, searching, and synthesis
- **Source Deduplication**: Strict URL and title deduplication prevents duplicate content
- **Citation Management**: Automatic citation validation and enhancement ensures proper attribution
- **Parallel Execution**: Multiple web searches run simultaneously (default: 5 parallel threads)
- **Quality Filtering**: Authority indicator detection (Google Scholar, university sites, etc.)
- **Structured Output**: Professional reports in structured, narrative, or executive formats
- **Configurable Depth**: Brief (3 tasks), standard (5 tasks), or comprehensive (8 tasks) research

### Advanced Features
- **Debug Mode**: Comprehensive tracking of queries, URLs, relevance assessments, and source decisions
- **Full-Text Extraction**: Deep content extraction with up to 15,000 characters per source
- **Reflexive Research**: Multi-iteration gap analysis and targeted refinement (independent feature)
- **Intent Analysis**: AI-powered query enhancement based on detected user intent (enhanced mode only)
- **Token Budgeting**: Track and limit token usage for cost control (enhanced mode only)

## Three Research Modes

BasicDeepSearch offers three distinct research modes optimized for different use cases. Each mode adjusts source limits, query generation, and processing strategies.

### Exact Mode Behavior Comparison

| **Parameter** | **Standard Mode** | **Fast Mode** | **Enhanced Mode** |
|--------------|------------------|---------------|-------------------|
| **CLI Flag** | `--research-mode standard` (default) | `--research-mode fast` | `--research-mode enhanced` |
| **Max Sources** | User-configurable (default: 15) | **Hard-coded to 10** | **Hard-coded to 6** |
| **Queries per Sub-task** | 2+ diverse queries | 1 focused query | 1 focused query |
| **Query Generation** | `_develop_search_questions()` | `_develop_search_questions_fast()` | `_develop_search_questions_fast()` |
| **Verification/Fact-checking** | Enabled (default) | **Disabled** | **Disabled** |
| **Intent Analysis** | Disabled | Disabled | **Enabled** (auto) |
| **Iterative Refinement** | Disabled | Disabled | **Enabled** (auto) |
| **Token Budget Tracking** | N/A | N/A | Yes (default: 50,000) |
| **Typical Time** | ~10 minutes | ~7 minutes | ~8 minutes |
| **Best For** | Comprehensive coverage | **Balanced speed/coverage** | AI-optimized quality |

**Note**: User can override auto-enabled features with `--enable-intent-analysis`, `--disable-intent-analysis`, `--enable-iterative-refinement`, `--disable-iterative-refinement` flags.

### Standard Mode (Default)
**Best for**: Comprehensive research with maximum coverage

**Exact Behavior** (lines 430-455 in basic_deepsearch.py):
- **Max Sources**: Uses user-provided `--max-sources` value (default: 15, range: 1-100)
- **Query Generation**: Calls `_develop_search_questions()` which generates `max(2, max_sources // len(sub_tasks))` queries per sub-task
- **Verification**: `include_verification` parameter respected (enabled by default unless `--no-verification` flag)
- **Intent Analysis**: `enable_intent_analysis=None` (not auto-enabled, must use `--enable-intent-analysis` to activate)
- **Iterative Refinement**: `enable_iterative_refinement=None` (not auto-enabled, must use `--enable-iterative-refinement` to activate)
- **Token Budgeting**: Not tracked

**Typical Time**: ~10 minutes for standard depth

**Example**:
```bash
# Standard mode with default settings
deepsearch "quantum computing developments 2024"

# Standard mode with increased sources
deepsearch "AI impact on healthcare" --max-sources 20 --depth comprehensive
```

**Python API**:
```python
from abstractcore.processing import BasicDeepSearch

# Standard mode (default)
searcher = BasicDeepSearch()
report = searcher.research(
    "What are the latest developments in quantum computing?",
    max_sources=15,
    search_depth="standard"
)
```

### Enhanced Mode
**Best for**: AI-optimized research with intelligent query enhancement and refinement

**Exact Behavior** (lines 430-498 in basic_deepsearch.py):
- **Max Sources**: **Force-limited to 6** via `max_sources = min(max_sources, 6)` (ignores user `--max-sources` if higher)
- **Query Generation**: Calls `_develop_search_questions_fast()` which generates **1 query per sub-task**
- **Verification**: **Force-disabled** via `include_verification = False` (ignores `--no-verification` flag)
- **Intent Analysis**: **Auto-enabled** via `enable_intent_analysis = (research_mode == "enhanced")` unless explicitly overridden
- **Iterative Refinement**: **Auto-enabled** via `enable_iterative_refinement = (research_mode == "enhanced")` unless explicitly overridden
- **Token Budgeting**: Tracked via `token_budget` parameter (default: 50,000, range: 1,000-500,000)
- **Refinement Condition**: Only runs if `tokens_used < token_budget * 0.8` (80% threshold)

**Unique Capabilities**:
1. **Intent-Aware Planning** (line 424): Calls `_analyze_intent_and_enhance_query()` to detect intent type and rewrite query
2. **Gap-Aware Refinement** (line 496): Calls `_smart_single_pass_refinement()` to analyze ONE critical gap and enhance report

**Typical Time**: ~8 minutes for standard depth

**Example**:
```bash
# Enhanced mode with intent analysis and refinement
deepsearch "AI safety challenges" --research-mode enhanced

# Enhanced mode with custom token budget
deepsearch "climate solutions" --research-mode enhanced
```

**Python API**:
```python
from abstractcore.processing import BasicDeepSearch

# Enhanced mode with intent analysis and refinement
searcher = BasicDeepSearch(
    research_mode="enhanced",
    token_budget=50000
)
report = searcher.research("What are the current challenges in AI safety?")

# Check token usage
print(f"Tokens used: {searcher.tokens_used}/{searcher.token_budget}")
```

### Fast Mode
**Best for**: Balanced speed and coverage - intermediary between Standard and Enhanced

**Exact Behavior** (lines 430-455 in basic_deepsearch.py):
- **Max Sources**: **Force-limited to 10** via `max_sources = min(max_sources, 10)` (intermediary between Standard's 15 and Enhanced's 6)
- **Query Generation**: Calls `_develop_search_questions_fast()` which generates **1 query per sub-task** (faster than Standard's 2+)
- **Verification**: **Force-disabled** via `include_verification = False` (saves time vs Standard)
- **Intent Analysis**: `enable_intent_analysis=None` (not auto-enabled, must use `--enable-intent-analysis` to activate)
- **Iterative Refinement**: `enable_iterative_refinement=None` (not auto-enabled, must use `--enable-iterative-refinement` to activate)
- **Token Budgeting**: Not tracked
- **Processing**: Streamlined query generation with reasonable source coverage

**Typical Time**: ~7 minutes for standard depth

**Why 10 Sources?**: Provides significantly better coverage than Enhanced mode (6 sources) while being faster than Standard mode (15 sources). This makes Fast mode a practical middle-ground option rather than just a crippled version of Enhanced.

**Example**:
```bash
# Fast mode for quick results
deepsearch "market trends 2025" --research-mode fast --depth brief

# Fast mode for rapid exploration
deepsearch "emerging technologies" --research-mode fast
```

**Python API**:
```python
from abstractcore.processing import BasicDeepSearch

# Fast mode for quick research
searcher = BasicDeepSearch(research_mode="fast")
report = searcher.research(
    "What are the current market trends?",
    search_depth="brief"
)
```

## Reflexive Research Mode (Advanced)

**IMPORTANT**: Reflexive mode is a **separate feature** that can be combined with any research mode (standard, enhanced, or fast).

### What is Reflexive Mode?

Reflexive mode enables **adaptive, self-improving research** through multi-iteration gap analysis and targeted refinement:

1. **Complete Initial Research**: Runs full pipeline (planning → questions → exploration → report)
2. **Gap Analysis**: LLM analyzes methodology & limitations to identify specific information gaps
3. **Targeted Refinement**: Generates focused search queries to address identified gaps (2-3 queries per gap)
4. **Iterative Improvement**: Integrates new findings and regenerates report
5. **Repeat**: Continues until no significant gaps remain or max iterations reached (default: 2)

### Gap Types Identified

The system identifies and addresses these gap types:

- **missing_perspective**: Lack of expert opinions or alternative viewpoints
- **insufficient_data**: Areas needing more quantitative information
- **outdated_info**: When current findings may be superseded
- **alternative_viewpoint**: Missing counterarguments or diverse perspectives
- **technical_detail**: Insufficient technical specifications or implementation details
- **recent_development**: Gaps in coverage of latest news or research

**Importance Threshold**: Only gaps with importance ≥ 6 (out of 10) are addressed.

### Combining Modes

Reflexive mode works independently and can be combined with any research mode:

- **Standard + Reflexive**: Maximum coverage with iterative refinement
- **Enhanced + Reflexive**: Intent analysis + single-pass refinement + multi-iteration gaps
- **Fast + Reflexive**: Quick initial research followed by targeted gap filling

### Examples

```bash
# Standard mode with reflexive improvement
deepsearch "quantum computing timeline" --reflexive

# Enhanced mode with reflexive improvement (most comprehensive)
deepsearch "AI safety research" --research-mode enhanced --reflexive

# Fast mode with reflexive improvement (balanced speed and quality)
deepsearch "market analysis" --research-mode fast --reflexive

# Custom iteration limit
deepsearch "climate solutions" --reflexive --max-reflexive-iterations 3
```

### Python API

```python
from abstractcore.processing import BasicDeepSearch

# Standard mode with reflexive improvement
searcher = BasicDeepSearch(
    reflexive_mode=True,
    max_reflexive_iterations=2
)
report = searcher.research("What are the latest quantum computing breakthroughs?")

# Enhanced mode + reflexive (maximum intelligence)
searcher = BasicDeepSearch(
    research_mode="enhanced",
    reflexive_mode=True,
    max_reflexive_iterations=3
)
report = searcher.research("Complex research topic")

# Fast mode + reflexive (balanced approach)
searcher = BasicDeepSearch(
    research_mode="fast",
    reflexive_mode=True
)
report = searcher.research("Quick but thorough research")
```

### When to Use Reflexive Mode

**Use reflexive mode when**:
- Research topic is complex with potential blind spots
- Initial results may have significant gaps
- You want the system to self-identify and address weaknesses
- Quality is more important than speed

**Skip reflexive mode when**:
- Speed is critical
- Topic is straightforward with clear information
- Budget constraints (more LLM calls = higher cost)
- Initial research is sufficient

## CLI Usage

**Default Provider**: The CLI app uses `ollama/qwen3:4b-instruct-2507-q4_K_M` by default unless configured otherwise via `abstractcore --set-default-model` or specified with `--provider` and `--model` flags.

### Basic Usage

```bash
# Simple research query (standard mode, uses default provider)
deepsearch "What are the latest developments in quantum computing?"

# Research with specific focus areas
deepsearch "AI impact on healthcare" --focus "diagnosis,treatment,ethics"

# Comprehensive research with custom output
deepsearch "sustainable energy 2025" \
  --depth comprehensive \
  --format executive \
  --output report.json

# Use custom provider/model
deepsearch "AI trends 2024" --provider openai --model gpt-4o-mini
```

### Mode Selection

```bash
# Standard mode (default) - comprehensive coverage
deepsearch "quantum computing" --max-sources 20

# Enhanced mode - AI-optimized with intent analysis
deepsearch "AI safety research" --research-mode enhanced

# Fast mode - speed-optimized
deepsearch "market trends" --research-mode fast --depth brief
```

### Advanced Options

```bash
# Custom LLM provider and model
deepsearch "blockchain technology trends" \
  --provider openai \
  --model gpt-4o-mini \
  --max-sources 25 \
  --verbose

# Debug mode for development
deepsearch "research topic" --debug

# Full-text extraction for deeper analysis
deepsearch "technical topic" --full-text

# Reflexive mode with custom iterations
deepsearch "complex topic" \
  --reflexive \
  --max-reflexive-iterations 3

# Manual mode overrides - enable intent analysis in standard mode
deepsearch "AI safety research" \
  --research-mode standard \
  --enable-intent-analysis

# Manual mode overrides - disable refinement in enhanced mode
deepsearch "quick analysis" \
  --research-mode enhanced \
  --disable-iterative-refinement

# Enhanced mode with custom token budget
deepsearch "comprehensive analysis" \
  --research-mode enhanced \
  --token-budget 75000

# Combine modes for maximum capability
deepsearch "comprehensive analysis" \
  --research-mode enhanced \
  --reflexive \
  --full-text \
  --max-sources 20 \
  --verbose
```

## Python API Usage

### Basic Research

```python
from abstractcore.processing import BasicDeepSearch

# Initialize with default settings (Ollama qwen3:4b)
searcher = BasicDeepSearch()

# Conduct research
report = searcher.research("What are the latest developments in quantum computing?")

# Access results
print(f"Title: {report.title}")
print(f"Summary: {report.executive_summary}")
print(f"\nKey Findings:")
for i, finding in enumerate(report.key_findings, 1):
    print(f"{i}. {finding}")
print(f"\nSources: {len(report.sources)}")
for source in report.sources[:5]:
    print(f"- {source['title']} ({source['relevance']:.2f})")
```

### Advanced Configuration

```python
from abstractcore import create_llm
from abstractcore.processing import BasicDeepSearch

# Custom LLM configuration
llm = create_llm(
    "openai",
    model="gpt-4o-mini",
    max_tokens=32000,
    max_output_tokens=8000
)

# Enhanced mode with all features
searcher = BasicDeepSearch(
    llm=llm,
    research_mode="enhanced",        # Intent analysis + refinement
    reflexive_mode=True,              # Multi-iteration improvement
    max_reflexive_iterations=3,
    full_text_extraction=True,        # Deep content extraction
    max_parallel_searches=8,          # More parallelism
    debug_mode=True,                  # Comprehensive logging
    token_budget=75000                # Custom budget
)

# Conduct comprehensive research
report = searcher.research(
    query="What are the current challenges in AI safety research?",
    focus_areas=["alignment", "robustness", "interpretability"],
    max_sources=20,
    search_depth="comprehensive",
    include_verification=True,
    output_format="executive"
)

# Check debug information
if searcher.debug_mode:
    print(f"Total queries: {len(searcher.debug_info['all_queries'])}")
    print(f"URLs found: {len(searcher.debug_info['all_urls_found'])}")
    print(f"Accepted sources: {len(searcher.debug_info['accepted_sources'])}")
    print(f"Rejected sources: {len(searcher.debug_info['rejected_sources'])}")

# Check token usage (enhanced mode only)
if searcher.research_mode == "enhanced":
    print(f"Token usage: {searcher.tokens_used}/{searcher.token_budget}")
```

### Mode Comparison Example

```python
from abstractcore.processing import BasicDeepSearch
import time

query = "Latest developments in quantum computing"

# Test all three modes
modes = ["standard", "enhanced", "fast"]
results = {}

for mode in modes:
    searcher = BasicDeepSearch(research_mode=mode)
    start = time.time()
    report = searcher.research(query, search_depth="brief")
    elapsed = time.time() - start

    results[mode] = {
        "time": elapsed,
        "sources": len(report.sources),
        "findings": len(report.key_findings),
        "report_length": len(report.detailed_analysis)
    }

# Compare results
for mode, metrics in results.items():
    print(f"\n{mode.upper()} MODE:")
    print(f"  Time: {metrics['time']:.1f}s")
    print(f"  Sources: {metrics['sources']}")
    print(f"  Key findings: {metrics['findings']}")
    print(f"  Analysis length: {metrics['report_length']} chars")
```

## Research Depths

Control the thoroughness of research by adjusting the number of sub-tasks:

### Brief (3 sub-tasks, ~5 minutes)
- Quick overview and current state
- Suitable for initial exploration
- 6-10 sources typically

```bash
deepsearch "topic" --depth brief
```

### Standard (5 sub-tasks, ~10 minutes)
- Balanced depth and breadth
- Good for most research needs
- 10-15 sources typically

```bash
deepsearch "topic" --depth standard  # default
```

### Comprehensive (8 sub-tasks, ~20 minutes)
- Deep analysis with multiple perspectives
- Includes stakeholders, economics, technical aspects
- 15-25 sources typically

```bash
deepsearch "topic" --depth comprehensive --max-sources 25
```

## Output Formats

### Structured (Default)
Professional research report format with clear sections:
- Executive Summary
- Key Findings (bullet points)
- Detailed Analysis (3-4 paragraphs)
- Conclusions
- Sources (with URLs and relevance scores)
- Methodology
- Limitations

**Best for**: Academic or professional documentation

```bash
deepsearch "topic" --format structured
```

### Executive
Concise, business-focused format:
- Emphasizes strategic insights and implications
- Actionable information prioritized
- Clear, executive-friendly language
- Highlights trends, opportunities, risks

**Best for**: Business decision-makers

```bash
deepsearch "topic" --format executive
```

### Narrative
Engaging, story-driven format:
- Shows connections between findings
- Logical flow from introduction to conclusion
- Storytelling techniques while maintaining objectivity
- Compelling presentation

**Best for**: Presentations and communication

```bash
deepsearch "topic" --format narrative
```

## Report Structure

All reports include:

- **Title**: Descriptive report title
- **Executive Summary**: 2-3 sentence overview of key insights
- **Key Findings**: Bullet points of main discoveries with citations
- **Detailed Analysis**: Comprehensive synthesis (3-4 paragraphs) with proper citations
- **Conclusions**: Implications and recommendations
- **Sources**: Complete list with URLs and relevance scores (0.0-1.0)
- **Methodology**: Research approach description including mode used
- **Limitations**: Caveats and constraints specific to this research

## Source Management

BasicDeepSearch implements strict source control:

### Deduplication
- **URL Deduplication**: Same URL never processed twice (global across all sub-tasks)
- **Title Deduplication**: Same content from different URLs detected and filtered
- **Early Termination**: Stops processing when source limit reached

### Quality Filtering

**Authority Indicators** (High-priority sources):
- Google Scholar, ORCID, ResearchGate profiles: 0.95 relevance score
- University/institute websites: 0.90 relevance score
- Educational domains (.edu, .ac.*): 0.90 relevance score
- Personal/official websites with name match: 0.95 relevance score

**Relevance Assessment**:
1. **LLM-Based**: Primary assessment using language model
2. **Fallback Heuristic**: Keyword overlap + content quality if LLM fails
3. **Error Detection**: Filters out 404 errors, access restrictions, navigation-only pages

### Source Limits

| Research Mode | Default Max Sources | Adjustable |
|---------------|---------------------|------------|
| Standard      | 15 (parameter default) | ✅ Yes (--max-sources) |
| Enhanced      | 6 (hard-coded) | ❌ No |
| Fast          | 6 (hard-coded) | ❌ No |

## Citation Management

BasicDeepSearch validates and enhances citations automatically:

### Citation Validation
- Detects citation patterns: "according to [Source]", "as reported by [Source]", parenthetical citations
- Counts factual sentences (those with indicators like "research shows", "data indicates", etc.)
- Calculates citation ratio: citations / factual sentences
- **Threshold**: 50% citation ratio required for adequate citation

### Citation Enhancement
If citations are insufficient (<50% ratio):
- Adds source list to end of content
- Enhances key findings with source attribution
- Ensures every claim has proper citation

### Citation Patterns Recognized
```
- according to [Source Name]
- as reported by [Source Name]
- according to Source Name
- as reported by Source Name
- (Source Name)  // parenthetical
```

## Debug Mode

Enable comprehensive tracking for development and troubleshooting:

```bash
deepsearch "topic" --debug
```

```python
searcher = BasicDeepSearch(debug_mode=True)
report = searcher.research("topic")
```

### Debug Information Tracked

**All Queries Generated**:
- Sub-task question
- Search queries for each task
- Total query count

**All URLs Discovered**:
- Organized by search query
- Title and URL for each
- Total URL count

**Relevance Assessments**:
- URL and title
- Relevant/not relevant decision
- Relevance score (0.0-1.0)
- Reason for decision
- Content preview (200 chars)

**Final Source Decisions**:
- **Accepted**: Sources added to report with scores and reasons
- **Rejected**: Sources filtered out with rejection reasons (not relevant, duplicate, limit reached)

### Debug Output

At completion, prints comprehensive summary:
```
=== DEBUG SUMMARY: COMPLETE RESEARCH PROCESS ===

TOTAL QUERIES GENERATED: 15
  Sub-task: What is quantum computing?
    1. "quantum computing overview 2024"
    2. "quantum computer technology explained"
  ...

TOTAL URLS DISCOVERED: 75
  Query: "quantum computing overview 2024" → 5 URLs
    1. Quantum Computing Basics
       🔗 https://example.com/...
  ...

RELEVANCE ASSESSMENTS: 45
  ✅ Quantum Computing: A Gentle Introduction
     Score: 0.85
     Reason: High-authority source with comprehensive content
  ❌ Product Marketing Page
     Score: 0.2
     Reason: Promotional content without informational value
  ...

FINAL SOURCES:
  ✅ Accepted: 15
  ❌ Rejected: 30
```

## Configuration Options

### Research Parameters
- `--focus` / `focus_areas`: Specific areas to emphasize (comma-separated)
- `--max-sources`: Number of sources to gather (1-100, default: 15)
- `--depth` / `search_depth`: Research thoroughness (brief, standard, comprehensive)
- `--no-verification` / `include_verification`: Enable/disable fact-checking
- `--format` / `output_format`: Report style (structured, narrative, executive)

### Mode Parameters
- `--research-mode`: Research mode (standard, enhanced, fast)
- `--reflexive`: Enable reflexive mode
- `--max-reflexive-iterations`: Reflexive iteration limit (default: 2)
- `--token-budget`: Token budget for enhanced mode (default: 50000, range: 1000-500000)
- `--enable-intent-analysis`: Manually enable intent analysis (auto-enabled in enhanced mode)
- `--disable-intent-analysis`: Manually disable intent analysis (overrides mode defaults)
- `--enable-iterative-refinement`: Manually enable iterative refinement (auto-enabled in enhanced mode)
- `--disable-iterative-refinement`: Manually disable iterative refinement (overrides mode defaults)

### Performance Settings
- `--parallel-searches` / `max_parallel_searches`: Concurrent web searches (1-20, default: 5)
- `--timeout`: HTTP request timeout in seconds (default: unlimited)
- `--max-tokens`: LLM context window (default: 32000)
- `--max-output-tokens`: LLM output limit (default: 8000)
- `--temperature`: LLM temperature (default: 0.1 for consistency)

### Advanced Features
- `--full-text`: Enable full-text extraction (15,000 chars vs 8,000)
- `--debug`: Enable comprehensive debug logging
- `--verbose`: Show detailed progress
- `--output`: Save report to file (JSON or markdown)

### Provider Configuration
- `--provider`: LLM provider (openai, anthropic, ollama, lmstudio, mlx, huggingface)
- `--model`: Model name (provider-specific)

## Best Practices

### Query Formulation

**Good Examples** (Specific and focused):
```bash
deepsearch "What are the latest developments in quantum computing for drug discovery?"
deepsearch "How is AI transforming medical diagnosis in 2024-2025?"
deepsearch "What are the main challenges facing renewable energy adoption in Europe?"
```

**Avoid** (Too broad or basic):
```bash
deepsearch "Tell me about AI"  # Too broad
deepsearch "What is quantum computing?"  # Basic definition query
deepsearch "Everything about healthcare"  # Unfocused
```

**Tips**:
- Use specific, focused questions
- Include time constraints when relevant ("2024", "latest", "current")
- Specify domain or context ("in healthcare", "for businesses")
- Avoid overly broad topics

### Focus Areas

Provide 3-5 specific focus areas for complex topics:

```bash
deepsearch "AI in education" \
  --focus "personalized learning,assessment automation,teacher tools,student outcomes,ethical concerns"
```

**Tips**:
- Use domain-specific terminology
- Balance breadth and depth
- Prioritize most important aspects

### Mode Selection Guide

| Scenario | Recommended Mode | Rationale |
|----------|-----------------|-----------|
| Unknown topic, need overview | Fast | Quick exploration |
| Well-defined research question | Standard | Balanced coverage |
| Complex topic needing AI help | Enhanced | Intent analysis helps |
| Topic with potential blind spots | Standard + Reflexive | Self-identifies gaps |
| Maximum comprehensiveness | Enhanced + Reflexive | All features enabled |
| Budget-constrained | Fast | Minimal LLM calls |
| Time-constrained | Fast + Brief depth | Fastest option |
| Production research | Standard/Enhanced | Reliable results |

### Performance Optimization

**Speed Priority**:
```bash
deepsearch "topic" \
  --research-mode fast \
  --depth brief \
  --no-verification \
  --parallel-searches 10
```

**Quality Priority**:
```bash
deepsearch "topic" \
  --research-mode enhanced \
  --reflexive \
  --full-text \
  --depth comprehensive \
  --max-sources 25 \
  --provider openai \
  --model gpt-4o-mini
```

**Balanced Approach**:
```bash
deepsearch "topic" \
  --research-mode standard \
  --depth standard \
  --max-sources 15 \
  --parallel-searches 5
```

### Output Management

**Save to file**:
```bash
# JSON for programmatic use
deepsearch "topic" --output report.json

# Markdown for sharing
deepsearch "topic" --output report.md --format narrative
```

**Process in Python**:
```python
import json

# Research and save
report = searcher.research("topic")
with open("report.json", "w") as f:
    json.dump(report.dict(), f, indent=2)

# Load and analyze
with open("report.json", "r") as f:
    data = json.load(f)
    print(f"Found {len(data['sources'])} sources")
    print(f"Key findings: {len(data['key_findings'])}")
```

## Error Handling

BasicDeepSearch includes robust error handling:

### Network Issues
- **Automatic retries**: Built into HTTP requests
- **Timeout handling**: Configurable timeout (--timeout)
- **Partial results**: Continues with available sources if some fail

### LLM Failures
- **Fallback prompts**: Uses simple queries if structured output fails
- **Graceful degradation**: Creates basic report if advanced generation fails
- **Multiple attempts**: Retry logic with feedback for structured output

### Source Failures
- **Continues execution**: Processes remaining sources if some fail
- **Error detection**: Filters out 404 errors, access restrictions
- **Synthetic findings**: Creates finding from search results if URLs fail

### No Results Scenario
If no sources found, creates special report:
- Explains why research failed (network, search limitations, etc.)
- Recommends manual research alternatives
- Includes attempted methodology

## Limitations

Based on actual implementation:

### Content Access
- Limited to **publicly available web content**
- **No paywalled content** access
- **Search service dependencies** (DuckDuckGo via `ddgs` library)
- **Network connectivity required**

### Temporal Coverage
- May not capture **very recent developments** (search index lag)
- Dependent on search engine freshness
- No real-time data access

### Language & Geography
- Primarily **English-language sources**
- Search results may be **geographically biased**
- Translation not supported

### Quality Assurance
- **Automated fact-checking has limitations**
- Citation detection **based on patterns** (may miss contextual citations)
- Relevance assessment **not perfect** (some false positives/negatives)

### Cost & Performance
- Enhanced mode: **Higher token usage** (intent analysis + refinement)
- Reflexive mode: **Multiple LLM calls** per iteration
- Full-text mode: **Increased processing time**
- Large source limits: **Longer execution time**

## Troubleshooting

### "Failed to initialize default Ollama model"

**Cause**: Ollama not installed or model not downloaded

**Solution**:
```bash
# Option 1: Install Ollama and download model
# 1. Install from https://ollama.com/
# 2. Download model:
ollama pull qwen3:4b-instruct-2507-q4_K_M

# Option 2: Use different provider
deepsearch "topic" --provider openai --model gpt-4o-mini

# Option 3: Python API with custom LLM
from abstractcore import create_llm
from abstractcore.processing import BasicDeepSearch

llm = create_llm("openai", model="gpt-4o-mini")
searcher = BasicDeepSearch(llm=llm)
```

### "No search results found"

**Causes**:
- Internet connectivity issues
- Search service rate limiting
- Query too specific

**Solutions**:
```bash
# Check connectivity
curl -I https://duckduckgo.com

# Try broader query
deepsearch "quantum computing" --depth brief

# Reduce source limit to avoid rate limiting
deepsearch "topic" --max-sources 10

# Install ddgs library
pip install ddgs
```

### "Report generation failed"

**Causes**:
- LLM context overflow
- Insufficient output tokens
- Model capability limitations

**Solutions**:
```bash
# Increase output tokens
deepsearch "topic" --max-output-tokens 16000

# Use more capable model
deepsearch "topic" --provider openai --model gpt-4o-mini

# Reduce source limit
deepsearch "topic" --max-sources 10
```

### "Timeout errors"

**Causes**:
- Slow network
- Many parallel searches
- Large documents

**Solutions**:
```bash
# Increase timeout
deepsearch "topic" --timeout 600

# Reduce parallel searches
deepsearch "topic" --parallel-searches 3

# Use fast mode
deepsearch "topic" --research-mode fast
```

### "Intent analysis failed" (Enhanced mode)

**Cause**: BasicIntentAnalyzer not available

**Solution**:
```bash
# Install required dependencies
pip install abstractcore[all]

# Or disable intent analysis
searcher = BasicDeepSearch(
    research_mode="enhanced",
    enable_intent_analysis=False  # Disable intent analysis
)
```

## Integration Examples

### Research Pipeline

```python
from abstractcore.processing import BasicDeepSearch

# Multi-topic research workflow
topics = [
    "quantum computing applications 2024",
    "AI safety developments 2024",
    "renewable energy innovations 2025"
]

searcher = BasicDeepSearch(research_mode="standard")
reports = []

for topic in topics:
    print(f"\nResearching: {topic}")
    report = searcher.research(
        topic,
        search_depth="standard",
        max_sources=15
    )
    reports.append(report)
    print(f"  ✓ Found {len(report.sources)} sources")
    print(f"  ✓ Generated {len(report.key_findings)} key findings")

# Analyze across reports
all_sources = []
for report in reports:
    all_sources.extend([s['url'] for s in report.sources])

print(f"\nTotal unique sources: {len(set(all_sources))}")
```

### Custom Analysis

```python
from abstractcore.processing import BasicDeepSearch

# Extract specific insights
def extract_trends(report):
    """Extract trend-related findings"""
    trends = []
    keywords = ['trend', 'growing', 'increasing', 'emerging', 'rising']

    for finding in report.key_findings:
        if any(word in finding.lower() for word in keywords):
            trends.append(finding)
    return trends

def extract_challenges(report):
    """Extract challenge-related findings"""
    challenges = []
    keywords = ['challenge', 'issue', 'problem', 'difficulty', 'obstacle']

    for finding in report.key_findings:
        if any(word in finding.lower() for word in keywords):
            challenges.append(finding)
    return challenges

# Conduct research
searcher = BasicDeepSearch(research_mode="enhanced")
report = searcher.research("AI market trends 2025")

# Analyze findings
trends = extract_trends(report)
challenges = extract_challenges(report)

print("Key Trends:")
for trend in trends:
    print(f"- {trend}")

print("\nKey Challenges:")
for challenge in challenges:
    print(f"- {challenge}")
```

### Comparative Research

```python
from abstractcore.processing import BasicDeepSearch
import json

# Compare different research approaches
query = "Impact of AI on healthcare"

# Approach 1: Fast initial scan
fast_searcher = BasicDeepSearch(research_mode="fast")
fast_report = fast_searcher.research(query, search_depth="brief")

# Approach 2: Deep dive on specific aspects
deep_searcher = BasicDeepSearch(research_mode="enhanced", reflexive_mode=True)
deep_report = deep_searcher.research(
    query,
    focus_areas=["diagnosis", "treatment", "patient care"],
    search_depth="comprehensive"
)

# Compare results
comparison = {
    "fast": {
        "sources": len(fast_report.sources),
        "findings": len(fast_report.key_findings),
        "analysis_length": len(fast_report.detailed_analysis)
    },
    "deep": {
        "sources": len(deep_report.sources),
        "findings": len(deep_report.key_findings),
        "analysis_length": len(deep_report.detailed_analysis)
    }
}

print(json.dumps(comparison, indent=2))
```

## Performance Tips

### For Speed
- Use **fast mode** with **brief depth**
- Disable verification (`--no-verification`)
- Increase parallel searches (`--parallel-searches 10`)
- Use local models (Ollama) to avoid network latency
- Reduce source limit (`--max-sources 8`)

### For Quality
- Use **enhanced mode** with **reflexive**
- Enable full-text extraction (`--full-text`)
- Use comprehensive depth (`--depth comprehensive`)
- Increase source limit (`--max-sources 25`)
- Use high-quality models (GPT-4o-mini, Claude Haiku)
- Enable verification (default)

### For Cost
- Use **fast mode** to minimize LLM calls
- Use local models (Ollama) - free
- Reduce source limit (fewer relevance assessments)
- Disable reflexive mode (fewer iterations)
- Use brief depth (fewer sub-tasks = fewer LLM calls)

### Recommended Configurations

**Budget-friendly (Free)**:
```bash
deepsearch "topic" \
  --research-mode fast \
  --provider ollama \
  --model qwen3:4b-instruct-2507-q4_K_M
```

**Production-ready**:
```bash
deepsearch "topic" \
  --research-mode standard \
  --provider openai \
  --model gpt-4o-mini \
  --max-sources 15 \
  --verbose
```

**Research-grade**:
```bash
deepsearch "topic" \
  --research-mode enhanced \
  --reflexive \
  --full-text \
  --provider anthropic \
  --model claude-3-5-haiku-latest \
  --depth comprehensive \
  --max-sources 25 \
  --verbose
```

## See Also

- **[BasicSummarizer](basic-summarizer.md)** - Document summarization
- **[BasicExtractor](basic-extractor.md)** - Knowledge extraction
- **[BasicJudge](basic-judge.md)** - Content evaluation
- **[BasicIntent](basic-intent.md)** - Intent analysis and deception detection
- **[Tool Calling Guide](../tool-calling.md)** - Custom tool integration
- **[Configuration Guide](../centralized-config.md)** - LLM setup
