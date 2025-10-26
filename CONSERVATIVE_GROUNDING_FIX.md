# Conservative Grounding System - Over-Inference Prevention

**Date**: 2025-10-26
**Status**: ✅ Implemented

## Problem Statement

The deep researcher was making **over-inferences** from weak evidence, leading to factually incorrect findings:

### Examples of Over-Inference:
1. **LinkedIn post commenting on Cellosaurus** → LLM inferred "lead developer of Cellosaurus"
2. **LinkedIn post sharing Reactome paper** → LLM inferred "integrated ChatGPT into Reactome"

**Root Cause**: Neither evidence extraction nor synthesis distinguished between:
- Direct statements ("X is Y", "X did Z")
- Indirect statements ("X commented on Y", "X shared Z")
- Citations/references ("X posted about Y")

## Solution Implemented

### **Two-Layer Conservative Grounding**

#### **Layer 1: Conservative Evidence Extraction** (Lines 809-828)

Added **explicit rules** to `_analyze_source_relevance()` prompt:

```python
CRITICAL RULES FOR EXTRACTING FACTS:
- Extract ONLY what is EXPLICITLY stated in the content
- DO NOT infer roles, positions, or actions from indirect evidence
- If content shows person commenting/sharing → say "X commented on/shared Y"
- If content shows person authored/created → say "X authored/created Y"
- If content is person's official profile/bio → state what it says
- Use cautious language ("appears to", "is associated with") if uncertain
- DO NOT assume authorship/creation from discussion/commentary

Examples of CORRECT extraction:
✅ "X shared a LinkedIn post about Cellosaurus" (if they shared)
✅ "X posted about ChatGPT usage in Reactome" (if they posted)
✅ "X is listed as author on paper Y" (if author list shows this)
✅ "X's profile states they work at Z" (if official bio says this)

Examples of WRONG extraction:
❌ "X is lead developer of Cellosaurus" (if they only commented)
❌ "X integrated ChatGPT into Reactome" (if they only shared article)
```

**Impact**:
- Facts extracted from sources are now **accurate descriptions** of what sources say
- No logical leaps during extraction
- Clear distinction between direct vs indirect evidence

#### **Layer 2: Conservative Synthesis** (Lines 1048-1063)

Added **explicit rules** to `_synthesize_with_grounding()` prompt:

```python
CRITICAL REQUIREMENTS FOR CONSERVATIVE GROUNDING:
- Base findings ONLY on what is EXPLICITLY stated in evidence
- DO NOT make inferences beyond what sources explicitly state
- For EACH finding, specify which evidence numbers (e.g., [1, 3, 5]) support it
- Every finding MUST have at least one evidence_id
- If evidence says "X shared Y" → finding should say "X shared Y" (NOT "X created Y")
- If evidence says "X posted about Y" → finding should say "X discussed Y" (NOT "X did Y")
- Use cautious language for indirect evidence ("appears to", "is associated with", "has engaged with")
- Do NOT combine weak evidence to make strong claims

Examples of CONSERVATIVE synthesis:
✅ CORRECT: "X shared LinkedIn posts about Cellosaurus and Reactome, indicating interest in these projects"
❌ WRONG: "X is the lead developer of Cellosaurus" (if evidence only shows sharing/commenting)

✅ CORRECT: "X has engaged with LLM integration in biocuration, as evidenced by posts discussing ChatGPT usage"
❌ WRONG: "X integrated ChatGPT into Reactome" (if evidence only shows discussion)
```

**Impact**:
- Synthesis stays faithful to extracted facts
- No combining weak evidence into strong claims
- Uses cautious language when appropriate

## Benefits

1. **Accurate Findings**: Claims now match what sources actually state
2. **No Over-Inference**: Commenting/sharing ≠ creating/authoring
3. **Cautious Language**: Uses "appears to", "is associated with" for weak evidence
4. **Transparent Reasoning**: Users can verify findings match sources
5. **Research Integrity**: Maintains scientific accuracy

## Evidence Classification

The system now implicitly classifies evidence by strength:

| Evidence Type | Example | Finding Language |
|--------------|---------|------------------|
| **Direct Statement** | "X is CEO of Y" on official bio | "X is CEO of Y" |
| **Authorship** | "X, Y, Z" on paper author list | "X authored paper Z" |
| **Association** | X shared post about Y | "X has engaged with Y", "X discussed Y" |
| **Commentary** | X commented on Y | "X commented on Y" |

## Testing

After implementing, the system should produce findings like:

✅ **Before Fix (Over-Inference)**:
- "He is the lead developer of Cellosaurus"
- "Albou has integrated large language models into Reactome"

✅ **After Fix (Conservative)**:
- "X has shared posts about Cellosaurus and Reactome, indicating interest in these bioinformatics projects"
- "X has engaged with discussions about LLM integration in biocuration"

## Files Modified

1. **`abstractcore/processing/basic_deepresearcherC.py`**:
   - Lines 809-828: Conservative evidence extraction rules
   - Lines 1048-1063: Conservative synthesis rules

## Verification

Run testds2.py with conservative grounding:

```bash
python testds2.py
```

Expected improvements:
- Findings accurately reflect source content
- No claims beyond what sources state
- Cautious language for indirect evidence
- All findings still traceable to sources

## Future Enhancements

1. **Fact Certainty Scoring**: Classify each fact by certainty level (high/medium/low)
2. **Explicit Quote Extraction**: Include direct quotes from sources in evidence
3. **Multi-Source Verification**: Require multiple sources for controversial claims
4. **Inference Validation**: Post-synthesis check to verify no over-inference
5. **Evidence Type Tagging**: Tag facts by type (direct/indirect/inferred)

## Conclusion

The conservative grounding system prevents over-inference by:
- **Extracting only explicit statements** from sources
- **Synthesizing faithfully** without logical leaps
- **Using cautious language** for weak evidence
- **Maintaining traceability** to source material

This ensures research findings are accurate, verifiable, and trustworthy.
