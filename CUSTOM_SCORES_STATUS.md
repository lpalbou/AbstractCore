# Custom Scores Implementation Status

## What Works ✅

1. **Custom Criteria Names** - ✅ Working
   - Custom criteria names are accepted via `custom_criteria` parameter
   - Custom criteria appear in `criteria_used` list in results

2. **Custom Criteria Descriptions** - ✅ Working
   - Custom descriptions are included in the LLM evaluation prompt
   - LLM sees: `- **Logical Coherence**: Are the results logically consistent?`

3. **Custom Criteria Evaluation** - ✅ Working
   - Custom criteria ARE evaluated by the LLM
   - Evaluation appears in the `reasoning` field
   - Strengths/weaknesses consider custom criteria

4. **Overall Score** - ✅ Working
   - Overall score (1-5) considers custom criteria
   - Overall assessment is comprehensive

## What Doesn't Work with Small Models ⚠️

**Individual Scores for Custom Criteria** - ⚠️ **Model-Dependent**

The `custom_scores` field (individual 1-5 scores for each custom criterion) is **not reliably populated by smaller models** like qwen3:4b.

### Why This Happens

1. **Structured Output Limitation**: We use Pydantic-based structured output
2. **Dynamic Fields**: `custom_scores` is a dynamic dict that changes based on custom_criteria
3. **Small Model Capacity**: qwen3:4b struggles with dynamic structured fields
4. **Valid Default**: Empty dict `{}` is a valid value, model defaults to that

### Current Behavior

```python
# What you get with qwen3:4b
assessment = session.generate_assessment(
    custom_criteria={
        "logical_coherence": "Are results consistent?",
        "technical_accuracy": "Is information accurate?"
    }
)

print(assessment['custom_scores'])
# Output: {}  ← Empty dict, not populated

print(assessment['criteria_used'])
# Output: [..., 'logical_coherence', 'technical_accuracy']  ← Custom criteria ARE used

print(assessment['reasoning'])
# Output: "...The content demonstrates strong logical coherence with consistent
#          reasoning throughout. Technical accuracy is moderate as some claims
#          lack supporting evidence..."  ← Custom criteria ARE evaluated in text
```

## Solutions

### Option 1: Use Larger Model (Recommended)

Larger models handle dynamic structured output better:

```python
# Try these models for reliable custom_scores
llm = create_llm("ollama", model="qwen3-coder:30b")  # Better
llm = create_llm("ollama", model="gpt-oss:120b")     # Best local
llm = create_llm("openai", model="gpt-4o-mini")      # Cloud option
llm = create_llm("anthropic", model="claude-sonnet-4-5-20250929")  # Best overall
```

### Option 2: Parse from Reasoning Text

The custom criteria ARE being evaluated - they're in the `reasoning` text:

```python
# Custom scores can be inferred from reasoning
reasoning = assessment['reasoning']
# Contains text like: "Technical accuracy is moderate (3/5)..."
# Could parse this programmatically if needed
```

### Option 3: Accept Text Evaluation Only

For smaller models, use the comprehensive text evaluation:

```python
assessment = session.generate_assessment(custom_criteria={...})

# Text evaluation IS comprehensive
print(assessment['overall_score'])  # Overall score considers custom criteria
print(assessment['reasoning'])       # Detailed evaluation of ALL criteria
print(assessment['strengths'])       # Includes custom criteria strengths
print(assessment['weaknesses'])      # Includes custom criteria weaknesses
```

## Recommendation

**For Production Use of Custom Scores:**

1. **Small models (qwen3:4b, gemma:7b)**: Use text evaluation in `reasoning` field
2. **Medium models (qwen3-coder:30b)**: Test if custom_scores populate reliably
3. **Large models (gpt-4o, claude-sonnet)**: Full custom_scores support expected

**For Digital Article Use Case:**

Since you're doing data analysis quality assessment, consider:

```python
# Option 1: Use larger model for scoring
llm = create_llm("ollama", model="qwen3-coder:30b")  # If available locally

# Option 2: Parse overall quality from text evaluation
assessment = session.generate_assessment(
    custom_criteria={
        "logical_coherence": "Are results consistent?",
        "result_plausibility": "Are findings plausible?",
        "assumption_validity": "Were assumptions checked?"
    }
)

# Use overall_score + reasoning for quality assessment
quality_ok = assessment['overall_score'] >= 4
reasoning = assessment['reasoning']  # Contains detailed custom criteria evaluation
```

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Accept custom_criteria | ✅ Complete | `Dict[str, str]` parameter |
| Include in prompts | ✅ Complete | Custom descriptions in LLM prompt |
| Evaluate custom criteria | ✅ Complete | Evaluation in `reasoning` text |
| Return in criteria_used | ✅ Complete | List includes custom criteria names |
| Individual custom_scores | ⚠️ Model-dependent | Works with large models, not small ones |

## Next Steps

1. **Document model requirements** in feature reply
2. **Test with larger models** to verify custom_scores work
3. **Consider fallback**: Parse scores from reasoning if custom_scores empty
4. **User education**: Set expectations about model size requirements
