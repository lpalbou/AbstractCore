# BasicJudge - LLM-as-a-Judge for Objective Evaluation

BasicJudge is a production-ready LLM-as-a-judge tool for objective evaluation and critical assessment. It provides structured, transparent evaluation with constructive skepticism, delivering clear, simple and actionable feedback using established evaluation practices.

## Quick Start

```python
from abstractcore.processing import BasicJudge

# Initialize with default model (Ollama qwen3:4b-instruct-2507-q4_K_M)
judge = BasicJudge()

# Evaluate content against default criteria
result = judge.evaluate("This code is well-structured and solves the problem elegantly.")

# Access enhanced assessment results
print(f"Judge's Summary: {result['judge_summary']}")
print(f"Source: {result['source_reference']}")
print(f"Overall score: {result['overall_score']}/5")
print(f"Strengths: {result['strengths']}")
print(f"Recommendations: {result['actionable_feedback']}")

# Include detailed criteria explanations (optional)
result_with_criteria = judge.evaluate(
    "Code review content",
    context="code review",
    include_criteria=True  # Add detailed criteria explanations
)
print(f"Criteria Details: {result_with_criteria['evaluation_criteria_details']}")
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
- **Temperature**: 0.1 (low for consistent evaluation)
- **Setup**: `ollama pull qwen3:4b-instruct-2507-q4_K_M`

**For Optimal Evaluation Quality**:
- **`qwen3-coder:30b`**: Good for detailed assessment (requires 32GB RAM)
- **`gpt-oss:120b`**: Highest quality evaluation (requires 120GB RAM)

**For Production**: Cloud providers (OpenAI GPT-4o-mini, Claude) offer the most reliable and consistent evaluation.

## Evaluation Framework

BasicJudge implements LLM-as-a-judge practices with structured assessment and chain-of-thought reasoning.

### Core Evaluation Criteria

The system evaluates content across nine standard quality dimensions:

- **Clarity**: How clear, understandable, and well-explained is the content?
- **Simplicity**: Is it appropriately simple vs unnecessarily complex for its purpose?
- **Actionability**: Does it provide actionable insights, recommendations, or next steps?
- **Soundness**: Is the reasoning logical, well-founded, and free of errors?
- **Innovation**: Does it show creativity, novel thinking, or fresh approaches?
- **Effectiveness**: Does it actually solve the intended problem or achieve its purpose?
- **Relevance**: Is it relevant and appropriate to the context and requirements?
- **Completeness**: Does it address all important aspects comprehensively?
- **Coherence**: Is the flow logical, consistent, and well-structured?

### Scoring System

Uses a **1-5 scale** with clear definitions:
- **Score 5**: Exceptional - Exceeds expectations in this dimension
- **Score 4**: Good - Meets expectations well with minor room for improvement
- **Score 3**: Adequate - Meets basic expectations but has notable areas for improvement
- **Score 2**: Poor - Falls short of expectations with significant issues
- **Score 1**: Very Poor - Fails to meet basic standards in this dimension

### Assessment Structure

Each evaluation returns a structured assessment with:
- **Judge's summary** (experiential note from judge's perspective about the assessment task and key findings)
- **Source reference** (clear indication of what was evaluated)
- **Individual criterion scores** (1-5 for each enabled criterion)
- **Overall score** (calculated average)
- **Strengths** (specific positive aspects identified)
- **Weaknesses** (areas for improvement)
- **Actionable feedback** (specific implementable recommendations)
- **Chain-of-thought reasoning** (transparent evaluation process)
- **Evaluation criteria details** (optional detailed explanation when include_criteria=True)

## Python API Reference

### BasicJudge Class

```python
class BasicJudge:
    def __init__(
        self,
        llm: Optional[AbstractCoreInterface] = None,
        temperature: float = 0.1  # Low temperature for consistent evaluation
    )

    def evaluate(
        self,
        content: str,
        context: Optional[str] = None,
        criteria: Optional[JudgmentCriteria] = None,
        focus: Optional[str] = None,
        reference: Optional[str] = None,
        include_criteria: bool = False
    ) -> dict

    def evaluate_files(
        self,
        file_paths: Union[str, List[str]],
        context: Optional[str] = None,
        criteria: Optional[JudgmentCriteria] = None,
        focus: Optional[str] = None,
        reference: Optional[str] = None,
        include_criteria: bool = False,
        max_file_size: int = 1000000
    ) -> Union[dict, List[dict]]
```

### Parameters

**evaluate() method:**
- **`content`** (str): The content to evaluate
- **`context`** (str, optional): Description of what is being evaluated (e.g., "code review", "documentation assessment")
- **`criteria`** (JudgmentCriteria, optional): Object specifying which standard criteria to use
- **`focus`** (str, optional): Specific areas to focus evaluation on (e.g., "technical accuracy, performance")
- **`reference`** (str, optional): Reference content for comparison-based evaluation
- **`include_criteria`** (bool, optional): Include detailed explanation of evaluation criteria in assessment (default: False)

**evaluate_files() method:**
- **`file_paths`** (str or List[str]): Single file path or list of file paths to evaluate sequentially
- **`context`** (str, optional): Description of evaluation context (default: "file content evaluation")
- **`criteria`** (JudgmentCriteria, optional): Object specifying which standard criteria to use
- **`focus`** (str, optional): Specific areas to focus evaluation on (e.g., "technical accuracy, performance")
- **`reference`** (str, optional): Reference content for comparison-based evaluation
- **`include_criteria`** (bool, optional): Include detailed explanation of evaluation criteria in assessment (default: False)
- **`max_file_size`** (int, optional): Maximum file size in bytes to prevent context overflow (default: 1MB)
- **`exclude_global`** (bool, optional): Skip global assessment for multiple files (default: False)

**Returns:**
- `evaluate()`: Single assessment dictionary
- `evaluate_files()`: Single dict if one file
- `evaluate_files()`: `{"global": global_assessment, "files": [individual_assessments]}` if multiple files (default)
- `evaluate_files()`: `[individual_assessments]` if multiple files and `exclude_global=True`

### JudgmentCriteria Configuration

```python
from abstractcore.processing import JudgmentCriteria

# Enable specific criteria only
criteria = JudgmentCriteria(
    is_clear=True,        # Evaluate clarity
    is_simple=True,       # Evaluate simplicity
    is_actionable=True,   # Evaluate actionability
    is_sound=False,       # Skip soundness evaluation
    is_innovative=False,  # Skip innovation evaluation
    is_working=True,      # Evaluate effectiveness
    is_relevant=True,     # Evaluate relevance
    is_complete=True,     # Evaluate completeness
    is_coherent=True      # Evaluate coherence
)
```

### Custom LLM Provider

```python
from abstractcore import create_llm
from abstractcore.processing import BasicJudge, create_judge

# RECOMMENDED: Use cloud providers for optimal evaluation quality
llm = create_llm("openai", model="gpt-4o-mini", temperature=0.1)
judge = BasicJudge(llm)

# OR use create_judge helper
judge = create_judge("anthropic", model="claude-3-5-haiku-latest", temperature=0.05)

# LOCAL MODELS: Work well for basic evaluation
judge = create_judge("ollama", model="qwen3-coder:30b", temperature=0.1)
```

### Multiple File Evaluation

BasicJudge can evaluate multiple files sequentially to avoid context overflow:

```python
from abstractcore.processing import BasicJudge, JudgmentCriteria

judge = BasicJudge()

# Evaluate single file
result = judge.evaluate_files("document.py", context="code review")
print(f"File assessment: {result['overall_score']}/5")

# Evaluate multiple files sequentially (returns list of assessments)
files = ["src/main.py", "src/utils.py", "tests/test_main.py"]
results = judge.evaluate_files(files, context="code review",
                               criteria=JudgmentCriteria(is_clear=True, is_sound=True))

for i, result in enumerate(results):
    file_name = files[i].split('/')[-1]
    print(f"{file_name}: {result['overall_score']}/5")
    print(f"  Judge Summary: {result['judge_summary']}")
    print(f"  Key Issues: {result['weaknesses']}")

# Configure file size limit (default 1MB)
large_files = ["big_doc.md", "large_code.py"]
try:
    results = judge.evaluate_files(large_files, max_file_size=2000000)  # 2MB limit
except ValueError as e:
    print(f"File too large: {e}")
```

### Global Assessment for Multiple Files

When evaluating multiple files, BasicJudge automatically generates a global assessment that synthesizes all individual evaluations:

```python
from abstractcore.processing import BasicJudge

judge = BasicJudge()

# Evaluate multiple files - returns global + individual assessments
result = judge.evaluate_files(
    ["src/main.py", "src/utils.py", "tests/test_main.py"],
    context="Python code review"
)

# Access global assessment (appears first)
global_assessment = result['global']
print(f"Global Score: {global_assessment['overall_score']}/5")
print(f"Global Summary: {global_assessment['judge_summary']}")

# Access individual file assessments
individual_assessments = result['files']
for assessment in individual_assessments:
    print(f"File: {assessment['source_reference']}")
    print(f"Score: {assessment['overall_score']}/5")

# Optional: Get original format (list of assessments only)
results = judge.evaluate_files(
    ["file1.py", "file2.py"],
    exclude_global=True  # Skip global assessment
)
# Returns: [assessment1, assessment2] (original behavior)
```

**Global Assessment Features:**
- **Synthesis**: Combines patterns across all individual file evaluations
- **Score Distribution**: Shows how many files scored at each level (1-5)
- **Pattern Analysis**: Identifies common strengths and weaknesses
- **Aggregate Scoring**: Provides overall quality assessment
- **Appears First**: Global assessment is shown before individual file results

**CLI Global Assessment:**
```bash
# Default: Includes global assessment
judge file1.py file2.py file3.py --context "code review"

# Skip global assessment (original behavior)
judge file1.py file2.py file3.py --context "code review" --exclude-global
```

## Command Line Interface

The `judge` CLI provides comprehensive evaluation capabilities for files and direct text input.

### Quick CLI Usage

```bash
# Simple usage (after pip install abstractcore[all])
judge "This code is well-structured and efficient."

# Evaluate single file with context
judge document.py --context "code review"

# Multiple files with specific criteria
judge file1.py file2.py file3.py --context "code review" --criteria clarity,soundness

# Custom output format and file
judge proposal.txt --format plain --output assessment.txt
```

### Alternative Usage Methods

```bash
# Method 1: Direct command (recommended after installation)
judge document.txt --context "code review"

# Method 2: Via Python module (always works)
python -m abstractcore.apps.judge document.txt --context "code review"
```

### Basic Usage

```bash
# Simple command (after package installation)
judge "This code is well-structured and efficient."

# Evaluate single file
judge document.py --context "code review"

# Evaluate multiple files sequentially (avoids context overflow)
judge file1.py file2.py file3.py --context "code review"

# Specify output format
judge content.md --format plain

# Save to file
judge proposal.txt --output assessment.json

# Multiple files with wildcard patterns
judge src/*.py --context "Python code review" --format json --output review.json
```

### Advanced Options

```bash
# Focus on specific criteria
judge doc.py --criteria clarity,soundness,effectiveness

# Focus on specific evaluation areas
judge api_docs.md --focus "technical accuracy,examples,error handling"

# Comparison-based evaluation
judge draft.md --reference ideal_solution.md

# Custom provider and model
judge content.txt --provider openai --model gpt-4o-mini

# Include detailed criteria explanations
judge content.txt --include-criteria --format plain

# Verbose output with progress
judge large_doc.md --verbose
```

### CLI Parameters

| Parameter | Description | Choices/Default |
|-----------|-------------|-----------------|
| `content` | Content to evaluate: single text string, single file path, or multiple file paths | Required (one or more arguments) |
| `--context` | Evaluation context description | Free text |
| `--criteria` | Comma-separated standard criteria | clarity, simplicity, actionability, soundness, innovation, effectiveness, relevance, completeness, coherence |
| `--focus` | Specific focus areas for evaluation | Free text (comma-separated) |
| `--reference` | Reference content for comparison | File path or text |
| `--include-criteria` | Include detailed criteria explanations in assessment | Flag |
| `--exclude-global` | Skip global assessment for multiple files | Flag (default: False, global assessment included) |
| `--format` | Output format | `json` (default), `plain`, `yaml` |
| `--output` | Output file path | Console if not provided |
| `--provider` | LLM provider | `ollama`, `openai`, `anthropic`, etc. |
| `--model` | LLM model | Provider-specific model name |
| `--temperature` | Evaluation temperature | 0.0-2.0 (default: 0.1) |
| `--verbose` | Detailed progress | Flag |

### Output Format Examples

**JSON Format (default):**
```bash
python -m abstractcore.apps.judge content.txt --format json
# Output: Structured JSON with scores, feedback, and reasoning
```

**Plain Text Format:**
```bash
python -m abstractcore.apps.judge content.txt --format plain
# Output: Human-readable assessment report
```

**Filtered Criteria:**
```bash
python -m abstractcore.apps.judge code.py --criteria clarity,soundness,effectiveness
# Output: Only evaluates specified criteria
```

**Enhanced Assessment with Criteria Details:**
```bash
python -m abstractcore.apps.judge content.txt --include-criteria --format plain
# Output: Includes judge's summary, source reference, and detailed criteria explanations
```

## Focus Areas: Impact on Evaluation

The `--focus` parameter dramatically changes evaluation outcomes by treating specified areas as **PRIMARY FOCUS AREAS**. Here are real examples showing the impact:

**Key Difference:**
- **Without focus**: Judge evaluates general quality (clarity, coherence) → High score
- **With focus**: Judge prioritizes specified areas → Low score when focus areas are missing

### Real README Evaluation Comparison

**Command:**
```bash
judge README.md --focus "technicalities, architectural diagrams and data flow, explanations of technical choices and comparison with SOTA approaches"
```

**Results:**
- **Overall Score**: 3/5 (down from 5/5 without focus)
- **Judge Summary**: "However, it critically lacks architectural diagrams and technical comparisons to SOTA approaches—core requirements..."
- **Weaknesses**: Directly address focus areas:
  - "No architectural diagrams or data flow visualizations"
  - "Lacks technical comparisons with SOTA approaches like LangChain, LlamaIndex"
  - "No explanation of how tool calling is unified across providers"

**Key Insight**: Focus areas become the **primary evaluation targets**. Even high quality documentation gets lower scores when it lacks the specified focus areas.

> **Fun Fact**: We used our own judge to evaluate our README.md with focus on "architectural diagrams and SOTA comparisons" and got a humbling 3/5 score. Turns out eany documentation can be improved! 😅

### Focus vs Criteria: Understanding the Difference

```bash
# --criteria: HOW to evaluate (evaluation methods)
judge doc.txt --criteria "clarity,soundness,effectiveness"

# --focus: WHAT to focus on (evaluation subjects)  
judge doc.txt --focus "performance benchmarks,security analysis"

# Combined: Evaluate specific areas using specific criteria
judge doc.txt --criteria "clarity,completeness" --focus "API documentation,error handling"
```

**Pro Tip**: Use `--focus` when you want to evaluate **specific content areas**. Use `--criteria` when you want to change **evaluation dimensions**.

## Real-World Examples

### Example 1: Code Review

**Input:**
```python
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
```

**Command:**
```bash
judge "def calculate_total..." --context "code review" --criteria clarity,soundness,effectiveness --format plain
```

**Expected Assessment:**
- **Clarity**: 4/5 - Clear function purpose and implementation
- **Soundness**: 3/5 - Missing error handling for None values
- **Effectiveness**: 4/5 - Solves the problem efficiently
- **Actionable Feedback**: Add input validation, consider using sum() built-in

### Example 2: Documentation Review

**Python API:**
```python
from abstractcore.processing import BasicJudge, JudgmentCriteria

judge = BasicJudge()

doc_content = """
# API Documentation
This API provides user management functionality.
Available endpoints: /users, /users/{id}
"""

# Focus on documentation-specific criteria
criteria = JudgmentCriteria(
    is_clear=True,
    is_complete=True,
    is_actionable=True,
    is_innovative=False,  # Not relevant for docs
    is_working=False      # Not applicable
)

assessment = judge.evaluate(
    content=doc_content,
    context="API documentation review",
    criteria=criteria,
    focus="examples, error handling, API completeness"
)

print(f"Completeness: {assessment['completeness_score']}/5")
print(f"Recommendations: {assessment['actionable_feedback']}")
```

### Example 3: Multiple File Code Review

**Evaluate an entire codebase:**
```bash
# Review all Python files in a project
judge src/*.py tests/*.py \
  --context="Python project review" \
  --criteria clarity,soundness,effectiveness \
  --format json \
  --output project_review.json \
  --verbose
```

**Expected Output:**
- List of assessments for each file
- Individual scores and feedback per file
- Consistent evaluation criteria across all files
- Identification of problematic files requiring attention

**Python API for Multiple Files:**
```python
from abstractcore.processing import BasicJudge
import glob

judge = BasicJudge()

# Get all Python files in project
python_files = glob.glob("src/**/*.py", recursive=True)

# Evaluate all files
results = judge.evaluate_files(
    file_paths=python_files,
    context="code quality review",
    criteria=JudgmentCriteria(is_clear=True, is_sound=True, is_working=True)
)

# Analyze results
problematic_files = [r for r in results if r['overall_score'] < 3]
high_quality_files = [r for r in results if r['overall_score'] >= 4]

print(f"Files needing attention: {len(problematic_files)}")
print(f"High-quality files: {len(high_quality_files)}")
```

### Example 4: Academic Writing Evaluation

**Command:**
```bash
judge research_paper.pdf \
  --context="academic paper review" \
  --criteria clarity,soundness,innovation,completeness \
  --reference conference_guidelines.txt \
  --format json \
  --output review_assessment.json \
  --verbose
```

## Best Practices

### 1. Model Selection for Evaluation

**For Critical Assessments (RECOMMENDED):**
```python
# Best quality for important evaluations
judge = create_judge("openai", model="gpt-4o-mini", temperature=0.1)

# Alternative: High-quality Claude
judge = create_judge("anthropic", model="claude-3-5-haiku-latest", temperature=0.05)
```

**For High-Volume Evaluation (Local):**
```python
# Good balance of quality and speed
judge = create_judge("ollama", model="qwen3-coder:30b", temperature=0.1)

# Fastest option (basic evaluation)
judge = create_judge("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0.1)
```

### 2. Criteria Selection Strategy

**For Code Reviews:**
```python
criteria = JudgmentCriteria(
    is_clear=True,
    is_simple=True,
    is_sound=True,
    is_working=True,
    is_innovative=False  # Usually not the focus
)
```

**For Documentation:**
```python
criteria = JudgmentCriteria(
    is_clear=True,
    is_complete=True,
    is_actionable=True,
    is_relevant=True,
    is_coherent=True,
    is_innovative=False,  # Not typically relevant
    is_sound=False        # Different meaning for docs
)
```

**For Creative Content:**
```python
criteria = JudgmentCriteria(
    is_clear=True,
    is_innovative=True,
    is_coherent=True,
    is_working=False,     # Not applicable
    is_sound=False        # Different context
)
```

### 3. Evaluation Context Guidelines

**Be Specific:**
- "code review for production deployment"
- "user-facing API documentation"
- "academic research proposal"
- "general review"

**Match Context to Criteria:**
- Code reviews: focus on soundness, clarity, effectiveness
- Documentation: focus on completeness, clarity, actionability
- Creative work: focus on innovation, coherence, clarity

### 4. Using Custom Criteria

```python
# Technical documentation
custom_criteria = ["has_examples", "covers_error_cases", "includes_prerequisites"]

# Code evaluation
custom_criteria = ["follows_style_guide", "has_tests", "handles_edge_cases"]

# Business proposals
custom_criteria = ["addresses_costs", "defines_timeline", "identifies_risks"]
```

### 5. Reference-Based Evaluation

```bash
# Compare against ideal solution
judge student_solution.py \
  --reference expert_solution.py \
  --context="programming assignment grading"

# Compare against standards
judge company_policy.md \
  --reference industry_standards.md \
  --context="policy compliance review"
```

## Assessment Interpretation

### Understanding Scores

**5 (Exceptional)**: Content exceeds expectations and demonstrates mastery
**4 (Good)**: Content meets expectations well with minor improvements possible
**3 (Adequate)**: Content meets basic standards but has notable gaps
**2 (Poor)**: Content falls short with significant issues requiring attention
**1 (Very Poor)**: Content fails to meet basic standards

### Actionable Feedback

The judge provides three types of feedback:
- **Strengths**: What works well (build upon these)
- **Weaknesses**: What needs improvement (prioritize addressing these)
- **Actionable Recommendations**: Specific steps to improve (implement these)

### Chain-of-Thought Reasoning

Each assessment includes transparent reasoning showing:
1. How each criterion was evaluated
2. Evidence supporting the scores
3. Calculation of the overall score
4. Justification for feedback and recommendations

## Integration Examples

### Content Management System

```python
from abstractcore.processing import BasicJudge

def evaluate_article(article_content):
    judge = BasicJudge()

    assessment = judge.evaluate(
        content=article_content,
        context="blog article review",
        criteria=JudgmentCriteria(
            is_clear=True,
            is_actionable=True,
            is_relevant=True,
            is_coherent=True
        )
    )

    return {
        'quality_score': assessment['overall_score'],
        'ready_to_publish': assessment['overall_score'] >= 4,
        'improvements_needed': assessment['actionable_feedback']
    }
```

### Code Review Automation

```python
def automated_code_review(code_diff, context="code review"):
    judge = BasicJudge()

    assessment = judge.evaluate(
        content=code_diff,
        context=context,
        focus="code conventions, test coverage, error handling"
    )

    return {
        'approval_recommended': assessment['overall_score'] >= 4,
        'concerns': assessment['weaknesses'],
        'required_changes': assessment['actionable_feedback']
    }
```

### Academic Grading Assistant

```python
def grade_assignment(student_submission, rubric_reference):
    judge = BasicJudge()

    assessment = judge.evaluate(
        content=student_submission,
        context="academic assignment grading",
        reference=rubric_reference,
        criteria=JudgmentCriteria(
            is_clear=True,
            is_sound=True,
            is_complete=True,
            is_coherent=True
        )
    )

    return {
        'grade': assessment['overall_score'],
        'feedback': assessment['actionable_feedback'],
        'strengths': assessment['strengths']
    }
```

## Performance Characteristics

### Speed Benchmarks (Approximate)

| Model | Content Length | Evaluation Time | Quality |
|-------|----------------|-----------------|---------|
| `qwen3:4b-instruct-2507-q4_K_M` | 500 chars | 30-40 seconds | Good |
| `qwen3-coder:30b` | 500 chars | 60-90 seconds | High |
| `gpt-oss:120b` | 500 chars | 90-120 seconds | Optimal |
| `gpt-4o-mini` | 500 chars | 15-30 seconds | Optimal |
| `claude-3-5-haiku` | 500 chars | 10-25 seconds | Optimal |

### Memory Usage

- **BasicJudge**: ~50MB base memory
- **Local models**: +2-8GB depending on model size
- **Structured output**: Adds minimal overhead

### Consistency

The judge uses low temperature (0.1 by default) for consistent evaluation:
- **Same content + same criteria** → Similar scores across runs
- **Different contexts** → Appropriately different assessments
- **Cloud providers** → Highest consistency
- **Local models** → Good consistency with proper temperature

## LLM-as-a-Judge Best Practices

BasicJudge implements 2025 state-of-the-art practices:

### 1. Structured Output
- **JSON format** for easy parsing and integration
- **Consistent schema** across all evaluations
- **Rich metadata** for comprehensive analysis

### 2. Chain-of-Thought Evaluation
- **Step-by-step reasoning** for transparency
- **Evidence-based scoring** with clear justification
- **Explicit calculation** of overall scores

### 3. Low-Temperature Generation
- **Consistent evaluation** across multiple runs
- **Reduced randomness** in scoring decisions
- **Reliable comparative assessments**

### 4. Comprehensive Error Handling
- **Graceful failure** with fallback assessments
- **Retry mechanisms** for transient failures
- **Clear error messages** for debugging

### 5. Configurable Criteria
- **Domain-specific evaluation** with relevant criteria
- **Custom criteria support** for specialized needs
- **Flexible assessment scope** based on context

BasicJudge is designed for production use with built-in error handling, retry logic, and efficient evaluation of content from short snippets to comprehensive documents.

## Timeout Configuration

The judge supports flexible timeout configuration for different evaluation scenarios:

### Default Behavior (Unlimited Timeout)
```bash
# Runs as long as needed - recommended for complex evaluations
python -m abstractcore.apps.judge document.txt
```

### Custom Timeout
```bash
# Set specific timeout (useful for production environments)
python -m abstractcore.apps.judge document.txt --timeout 300   # 5 minutes
python -m abstractcore.apps.judge document.txt --timeout 900   # 15 minutes

# Explicit unlimited timeout
python -m abstractcore.apps.judge document.txt --timeout none
```

### Programmatic Usage
```python
from abstractcore.processing import BasicJudge

# Unlimited timeout (default)
judge = BasicJudge()

# Custom timeout
judge = BasicJudge(timeout=300)  # 5 minutes

# Explicit unlimited timeout
judge = BasicJudge(timeout=None)
```

**When to Use Timeouts:**
- **Production environments**: Set reasonable timeouts (300-900 seconds) to prevent hanging
- **Large documents**: Use unlimited timeout for comprehensive evaluations
- **Multiple files**: Consider longer timeouts when evaluating many files
- **Complex criteria**: Detailed evaluations may need more time

## Troubleshooting

### Common Issues

**"Failed to initialize default Ollama model"**
```bash
# Install Ollama and download model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama serve
```

**Inconsistent evaluation results**
- Use lower temperature: `--temperature=0.05`
- Try a more capable model: `--provider openai --model gpt-4o-mini`
- Ensure context description is specific and consistent

**Low-quality assessments**
- Use more capable models (GPT-4, Claude)
- Provide specific evaluation context
- Focus criteria on relevant dimensions only

**JSON parsing errors**
- Automatic retry handles most cases
- If persistent, try a more capable model
- Check input content with `--verbose` flag

### Error Messages

**"Temperature must be between 0.0 and 2.0"**
- Adjust `--temperature` parameter to valid range
- Recommended: 0.1 for consistency, up to 0.3 for slight variation

**"Provider/model required together"**
- Both `--provider` and `--model` must be specified together

**"Unknown criterion"**
- Check spelling of criteria names
- Use available standard criteria or custom criteria

BasicJudge provides reliable, transparent evaluation suitable for critical assessment across various domains, from code review to content evaluation to academic grading.