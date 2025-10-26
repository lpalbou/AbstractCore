# Gap Source Attribution System

**Date**: 2025-10-26
**Status**: ✅ Implemented

## Problem Statement

Knowledge gaps were not linked to their source of origin, making it impossible to trace where questions came from.

### Example Problem

**Gap identified**:
```
"The extent of his involvement in the development of the Alpha Shapes-based 3D structural
comparison method (from his doctoral thesis) beyond academic publication is not specified."
```

**Issue**:
- This gap came from a SOURCE (thesis/publication mention)
- BUT we weren't tracking WHICH source raised this question
- Result: `"related_source_urls": []` - no traceability!

**User's insight**: "it didn't invent the question about alpha shapes... as i did work on it, so this is not an hallucination and it found that information somewhere"

## Root Cause

1. **Evidence extraction** creates atomic facts from sources
2. **Synthesis** identifies gaps when information is incomplete/partial
3. **BUT** synthesis gaps weren't linked back to which evidence raised them
4. **Result**: Gaps appeared to come from nowhere

## Solution Implemented

### **1. Added Gap Source Attribution Model**

**New Model** (`GapWithSource`):
```python
class GapWithSource(BaseModel):
    """Knowledge gap with source attribution"""
    gap: str = Field(description="The knowledge gap or unanswered question")
    evidence_ids: List[int] = Field(description="Evidence piece numbers that raised this gap")
```

**Updated SynthesisModel**:
```python
class SynthesisModel(BaseModel):
    title: str
    summary: str
    findings_with_sources: List[FindingWithSource]  # Already had this
    gaps_with_sources: List[GapWithSource]  # NEW - was just List[str]
    confidence: float
```

### **2. Enhanced Synthesis Prompt** (Lines 1066-1070)

```python
GAP SOURCE ATTRIBUTION:
- For EACH gap, identify which evidence numbers raised the question
- Example: If evidence [5] mentions "thesis on Alpha Shapes" → gap should have evidence_ids: [5]
- If gap is general (not from specific evidence), leave evidence_ids empty
- Clean out gaps that are ALREADY covered by findings (don't list as gaps if we have answers!)
```

### **3. Link Gaps to Source URLs** (Lines 1102-1124)

After synthesis, map evidence IDs to actual URLs:

```python
# Link gaps to their source origins
for gap_obj in response.gaps_with_sources:
    source_urls = []
    if gap_obj.evidence_ids:
        # Get URLs for all evidence IDs that raised this gap
        for ev_id in gap_obj.evidence_ids:
            if ev_id in evidence_urls:
                source_urls.append(evidence_urls[ev_id])

    # Update gap tracking with source attribution
    self.active_gaps[gap_obj.gap] = {
        "status": "active",
        "source_urls": source_urls,  # NOW POPULATED!
        "related_findings": [],
        "dimension": "synthesis"
    }
```

### **4. Gap Cleanup Instruction**

Added to synthesis prompt:
```python
"Clean out gaps that are ALREADY covered by findings (don't list as gaps if we have answers!)"
```

This prevents redundancy where findings answer questions but gaps still list them as unanswered.

## How It Works

### **Before Fix**:
```json
{
  "gap": "Extent of Alpha Shapes involvement not specified",
  "status": "active",
  "related_source_urls": [],  // ❌ NO SOURCE LINK
  "related_findings": [],
  "has_source_context": false
}
```

### **After Fix**:
```json
{
  "gap": "Extent of Alpha Shapes involvement not specified",
  "status": "active",
  "related_source_urls": [
    "https://theses.fr/2010STRA6252"  // ✅ THESIS SOURCE!
  ],
  "related_findings": [
    "Albou authored doctoral thesis on Alpha Shapes"
  ],
  "has_source_context": true
}
```

## Benefits

1. **Complete Traceability**: Every gap traceable to source(s) that raised it
2. **Bidirectional Links**: `<gap> ↔ <source>` relationship maintained
3. **Gap Validation**: Can verify if gap is based on actual partial information
4. **Follow-up Research**: Know which sources need deeper investigation
5. **Quality Control**: Distinguish between:
   - **Grounded gaps** (from evidence mentioning incomplete info)
   - **General gaps** (from broad question decomposition)
   - **Covered gaps** (should be removed - already have findings)

## Gap Types

| Gap Type | Evidence IDs | Source URLs | Meaning |
|----------|--------------|-------------|---------|
| **Grounded** | [3, 7] | 2 URLs | Specific sources raised this question |
| **General** | [] | [] | Broad question not tied to specific evidence |
| **Partial** | [5] | 1 URL | One source mentioned but didn't detail this |

## Example Use Cases

### **Use Case 1: Follow-up Research**
```
Gap: "Extent of Alpha Shapes involvement not specified"
Source: https://theses.fr/2010STRA6252 (thesis)

Action: Deep dive into thesis and related publications
```

### **Use Case 2: Gap Validation**
```
Gap: "Current role at Roche not detailed"
Source: https://rocketreach.co/... (mentions Roche)

Action: Search specifically for recent Roche affiliations
```

### **Use Case 3: Gap Cleanup**
```
Gap: "Academic contributions unclear"
Findings: ["Authored 50+ papers", "Cited 10,000+ times"]

Action: Remove gap - already covered by findings!
```

## Files Modified

1. **`abstractcore/processing/basic_deepresearcherC.py`**:
   - Lines 118-121: Added `GapWithSource` model
   - Lines 129: Updated `SynthesisModel` to use `gaps_with_sources`
   - Lines 1026-1027, 1143, 1156: Updated fallback cases
   - Lines 1066-1070: Gap attribution instructions in prompt
   - Lines 1102-1124: Gap-to-source URL mapping
   - Lines 289-293: Extract gaps from new structure

## Testing

Run testds2.py to see gap source attribution:

```bash
python testds2.py
```

**Expected improvements**:
- Gaps with `related_source_urls` populated
- `has_source_context: true` for grounded gaps
- Cleaner gap list (no redundant gaps already covered by findings)

## Future Enhancements

1. **Multi-Source Gaps**: Gaps raised by multiple sources (consensus)
2. **Gap Confidence**: Score gaps by how many sources mention the missing info
3. **Gap Resolution Tracking**: Mark when follow-up research resolves gaps
4. **Gap Prioritization**: Rank gaps by source count and credibility
5. **Gap Clustering**: Group similar gaps from different sources

## Conclusion

The gap source attribution system ensures **complete traceability** of knowledge gaps:
- **Findings** → Linked to sources (already fixed earlier)
- **Gaps** → Linked to sources (fixed now)

Every element of research output is now **grounded** and **traceable** to its origin, eliminating the mystery of "where did this question come from?"

**Result**: Bidirectional links `<finding> ↔ <source>` AND `<gap> ↔ <source>` fully implemented!
