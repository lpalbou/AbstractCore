# BasicDeepSearch - Autonomous Research Agent

The BasicDeepSearch application provides autonomous, multi-stage research capabilities that go beyond simple search to deliver comprehensive, synthesized research reports with citations.

## Overview

BasicDeepSearch implements a state-of-the-art four-stage research pipeline:

1. **Planning**: Decomposes complex research queries into structured sub-tasks
2. **Question Development**: Generates specific, diverse search queries for each sub-task  
3. **Web Exploration**: Executes parallel web searches and gathers evidence
4. **Report Generation**: Synthesizes findings into structured reports with citations

## Key Features

- **Autonomous Research**: Minimal human intervention required
- **Reflexive Research**: Analyzes limitations and performs targeted refinement searches
- **Parallel Execution**: Multiple web searches run simultaneously for speed
- **Structured Output**: Professional reports with citations and analysis
- **Verification Layer**: Optional fact-checking and validation
- **Multiple Formats**: Structured, narrative, or executive report styles
- **Configurable Depth**: Brief, standard, or comprehensive research modes

## CLI Usage

### Basic Usage

```bash
# Simple research query
deepsearch "What are the latest developments in quantum computing?"

# Research with specific focus areas
deepsearch "AI impact on healthcare" --focus "diagnosis,treatment,ethics"

# Comprehensive research with custom output
deepsearch "sustainable energy 2025" --depth comprehensive --format executive --output report.json
```

### Advanced Options

```bash
# High-volume research with custom LLM
deepsearch "blockchain technology trends" \
  --max-sources 25 \
  --provider openai \
  --model gpt-4o-mini \
  --verbose

# Fast research without verification
deepsearch "current market trends" \
  --depth brief \
  --no-verification \
  --parallel-searches 10

# Reflexive mode - analyzes gaps and refines research automatically
deepsearch "quantum computing breakthroughs" --reflexive
deepsearch "AI safety research" --reflexive --max-reflexive-iterations 3

# Full-text extraction with reflexive improvement
deepsearch "climate change solutions" --full-text --reflexive
```

## Reflexive Research Mode

**Reflexive mode** (`--reflexive`) enables **adaptive, self-improving research** that learns from its own limitations and iteratively refines the results.

### How It Works

1. **Standard Research**: Executes the normal 4-stage pipeline
2. **Gap Analysis**: LLM analyzes the "Methodology & Limitations" section to identify specific information gaps
3. **Targeted Refinement**: Generates focused search queries to address identified gaps
4. **Iterative Improvement**: Repeats until no significant gaps remain or max iterations reached

### Gap Types Identified

- **Missing Perspectives**: Lack of expert opinions or alternative viewpoints
- **Insufficient Data**: Areas where more quantitative information is needed
- **Outdated Information**: When current findings may be superseded by recent developments
- **Technical Details**: Missing technical specifications or implementation details
- **Recent Developments**: Gaps in coverage of latest news or research

### Example Reflexive Analysis

```
Initial Research: "quantum computing timeline"
├── Finds general information about quantum computing progress
├── Limitations: "Limited coverage of recent commercial developments"
└── Reflexive Gap Analysis:
    ├── Gap: "Missing industry expert predictions for 2025-2030"
    ├── Searches: ["quantum computing expert predictions 2025", "industry roadmap quantum timeline"]
    └── Result: Enhanced report with expert opinions and commercial timelines
```

### Configuration

```bash
# Enable reflexive mode with default 2 iterations
deepsearch "AI safety research" --reflexive

# Custom iteration limit
deepsearch "climate solutions" --reflexive --max-reflexive-iterations 3

# Combine with other advanced features
deepsearch "quantum breakthroughs" --reflexive --full-text --max-sources 20
```

## Python API Usage

### Basic Research

```python
from abstractcore.processing import BasicDeepSearch

# Initialize with default settings
searcher = BasicDeepSearch()

# Conduct research
report = searcher.research("What are the latest developments in quantum computing?")

# Access results
print(f"Title: {report.title}")
print(f"Summary: {report.executive_summary}")
print(f"Sources: {len(report.sources)}")
```

### Advanced Configuration

```python
from abstractcore import create_llm
from abstractcore.processing import BasicDeepSearch

# Custom LLM configuration
llm = create_llm("openai", model="gpt-4o-mini", max_tokens=32000)

# Reflexive research configuration
searcher = BasicDeepSearch(
    llm=llm,
    reflexive_mode=True,
    max_reflexive_iterations=3,
    full_text_extraction=True
)

# Conduct reflexive research
report = searcher.research(
    "What are the current challenges in AI safety research?",
    focus_areas=["alignment", "robustness", "interpretability"],
    output_format="structured"
)

print(f"Methodology: {report.methodology}")
print(f"Limitations: {report.limitations}")
print(f"Sources analyzed: {len(report.sources)}")

# Initialize with custom settings
searcher = BasicDeepSearch(
    llm=llm,
    max_parallel_searches=8,
    timeout=600
)

# Comprehensive research
report = searcher.research(
    query="Impact of AI on healthcare",
    focus_areas=["medical diagnosis", "drug discovery", "patient care"],
    max_sources=20,
    search_depth="comprehensive",
    include_verification=True,
    output_format="executive"
)
```

## Research Depths

### Brief (3 sub-tasks, ~5 minutes)
- Quick overview and current state
- Suitable for initial exploration
- 10-15 sources typically

### Standard (5 sub-tasks, ~10 minutes)  
- Balanced depth and breadth
- Good for most research needs
- 15-20 sources typically

### Comprehensive (8 sub-tasks, ~20 minutes)
- Deep analysis with multiple perspectives
- Includes stakeholders, economics, technical aspects
- 20-30 sources typically

## Output Formats

### Structured (Default)
- Professional research report format
- Clear sections: Executive Summary, Key Findings, Analysis, Conclusions
- Ideal for academic or business use

### Executive
- Concise, business-focused format
- Emphasizes strategic insights and implications
- Perfect for decision-makers

### Narrative
- Engaging, story-driven format
- Shows connections between findings
- Great for presentations and communication

## Report Structure

All reports include:

- **Title**: Descriptive report title
- **Executive Summary**: 2-3 sentence overview
- **Key Findings**: Bullet points of main discoveries
- **Detailed Analysis**: Comprehensive synthesis (3-4 paragraphs)
- **Conclusions**: Implications and recommendations
- **Sources**: Complete list with URLs and relevance scores
- **Methodology**: Research approach description
- **Limitations**: Caveats and constraints

## Configuration Options

### Research Parameters
- `focus_areas`: Specific areas to emphasize
- `max_sources`: Number of sources to gather (1-100)
- `search_depth`: Research thoroughness level
- `include_verification`: Enable fact-checking
- `output_format`: Report style

### Performance Settings
- `max_parallel_searches`: Concurrent web searches (1-20)
- `timeout`: HTTP request timeout
- `max_tokens`: LLM context window
- `max_output_tokens`: LLM output limit

## Best Practices

### Query Formulation
- Use specific, focused questions
- Avoid overly broad topics
- Include time constraints when relevant
- Specify domain or context

**Good Examples:**
- "What are the latest developments in quantum computing for drug discovery?"
- "How is AI transforming medical diagnosis in 2024-2025?"
- "What are the main challenges facing renewable energy adoption?"

**Avoid:**
- "Tell me about AI" (too broad)
- "What is quantum computing?" (basic definition)
- "Everything about healthcare" (unfocused)

### Focus Areas
- Provide 3-5 specific focus areas for complex topics
- Use domain-specific terminology
- Balance breadth and depth

**Example:**
```bash
deepsearch "AI in education" --focus "personalized learning,assessment automation,teacher tools,student outcomes,ethical concerns"
```

### Performance Optimization
- Use `brief` depth for quick overviews
- Increase `parallel_searches` for faster execution
- Use cloud providers (OpenAI, Anthropic) for reliability
- Enable `verbose` mode for progress tracking

### Output Management
- Save comprehensive reports to files (`--output report.json`)
- Use markdown format for sharing (`--output report.md`)
- Choose appropriate format for audience

## Error Handling

The system includes robust error handling:

- **Network Issues**: Automatic retries with exponential backoff
- **LLM Failures**: Graceful degradation with fallback responses
- **Parsing Errors**: Fallback to simplified report generation
- **Source Failures**: Continues with available sources

## Limitations

- **Source Quality**: Limited to publicly available web content
- **Real-time Data**: May not capture very recent developments
- **Language**: Primarily English-language sources
- **Verification**: Automated fact-checking has limitations
- **Bias**: Inherits biases from web sources and LLM training

## Integration Examples

### Research Pipeline
```python
# Multi-stage research workflow
topics = [
    "quantum computing applications",
    "AI safety developments", 
    "renewable energy innovations"
]

searcher = BasicDeepSearch()
reports = []

for topic in topics:
    report = searcher.research(
        topic,
        search_depth="standard",
        max_sources=15
    )
    reports.append(report)

# Analyze across reports
all_sources = []
for report in reports:
    all_sources.extend(report.sources)

print(f"Total unique sources: {len(set(s['url'] for s in all_sources))}")
```

### Custom Analysis
```python
# Extract specific insights
def extract_trends(report):
    trends = []
    for finding in report.key_findings:
        if any(word in finding.lower() for word in ['trend', 'growing', 'increasing', 'emerging']):
            trends.append(finding)
    return trends

report = searcher.research("AI market trends 2025")
trends = extract_trends(report)
print("Key trends identified:")
for trend in trends:
    print(f"- {trend}")
```

## Troubleshooting

### Common Issues

**"Failed to initialize default Ollama model"**
- Install Ollama: https://ollama.com/
- Pull model: `ollama pull qwen3:4b-instruct-2507-q4_K_M`
- Or specify custom provider: `--provider openai --model gpt-4o-mini`

**"No search results found"**
- Check internet connectivity
- Try broader search terms
- Reduce `max_sources` if hitting rate limits
- Install `ddgs` for better web search: `pip install ddgs`

**"Report generation failed"**
- Increase `max_output_tokens`
- Use more capable model (e.g., gpt-4o-mini)
- Reduce `max_sources` to avoid context overflow

**"Timeout errors"**
- Increase `--timeout` value
- Reduce `parallel_searches`
- Use faster LLM provider

### Performance Tips

- Use local models (Ollama) for cost-effective research
- Use cloud models (OpenAI, Anthropic) for reliability
- Enable `verbose` mode to monitor progress
- Save reports to files for large research projects
- Use `brief` depth for quick iterations

## See Also

- [BasicSummarizer](basic-summarizer.md) - Document summarization
- [BasicExtractor](basic-extractor.md) - Knowledge extraction  
- [BasicJudge](basic-judge.md) - Content evaluation
- [Tool Calling Guide](../tool-calling.md) - Custom tool integration
- [Configuration Guide](../centralized-config.md) - LLM setup
