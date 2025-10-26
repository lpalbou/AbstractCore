# TODO: Add Detailed Report to BasicDeepResearcherC

**Date**: 2025-10-26
**Priority**: High
**Estimated Time**: 30-40 minutes

---

## Problem Statement

BasicDeepResearcherC currently generates:
- ✅ Title
- ✅ Executive summary (2-3 sentences)
- ✅ Key findings with source attribution
- ✅ Knowledge gaps with source attribution
- ❌ **NO detailed sectioned report**

**Comparison with Researcher A**:
- Researcher A has `detailed_report` field with organized sections
- Researcher C only has a brief summary - missing depth

---

## Goal

Add comprehensive `detailed_report` field to BasicDeepResearcherC that generates:
- **Sectioned report** with clear headings
- **Detailed analysis** for each section (not just bullet points)
- **Well-organized structure** (Background, Contributions, Current Work, etc.)
- **Conservative grounding** (all statements tied to evidence)

---

## Implementation Plan

### **Step 1: Update `SynthesisModel`** (~5 min)

**File**: `abstractcore/processing/basic_deepresearcherC.py`
**Location**: Around line 124

**Current**:
```python
class SynthesisModel(BaseModel):
    """Final synthesis - SIMPLE SCHEMA"""
    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary")
    findings_with_sources: List[FindingWithSource]
    gaps_with_sources: List[GapWithSource]
    confidence: float = Field(description="Overall confidence")
```

**Add**:
```python
class DetailedSection(BaseModel):
    """Section of detailed report - SIMPLE SCHEMA"""
    heading: str = Field(description="Section heading")
    content: str = Field(description="Detailed content for this section")


class SynthesisModel(BaseModel):
    """Final synthesis - SIMPLE SCHEMA"""
    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary")
    findings_with_sources: List[FindingWithSource]
    detailed_sections: List[DetailedSection] = Field(description="3-5 detailed report sections")  # ← NEW
    gaps_with_sources: List[GapWithSource]
    confidence: float = Field(description="Overall confidence")
```

---

### **Step 2: Update Synthesis Prompt** (~10 min)

**File**: `abstractcore/processing/basic_deepresearcherC.py`
**Location**: Around line 1049 (`_synthesize_with_grounding()` method)

**Current prompt says**:
```
Create a synthesis with:
1. title: Descriptive report title
2. summary: 2-3 sentence executive summary
3. findings_with_sources: 3-7 key findings...
4. gaps_with_sources: Knowledge gaps...
5. confidence: Overall confidence 0-1
```

**Update to**:
```
Create a synthesis with:
1. title: Descriptive report title
2. summary: 2-3 sentence executive summary
3. findings_with_sources: 3-7 key findings, EACH with evidence_ids...
4. detailed_sections: 3-5 detailed sections with heading and content
   - Organize evidence into logical sections (e.g., Background, Key Contributions, Current Work, etc.)
   - Each section should have 2-4 paragraphs of detailed analysis
   - Base all statements on evidence provided
   - Use conservative language (no over-inference)
5. gaps_with_sources: Knowledge gaps...
6. confidence: Overall confidence 0-1

SECTION REQUIREMENTS:
- Create logical sections based on the evidence (not predetermined headings)
- Each section: 150-300 words of detailed analysis
- Ground all statements in evidence (cite evidence numbers if helpful for clarity)
- Use cautious language for indirect evidence
```

**Important**: Add this AFTER the conservative grounding instructions around line 1070.

---

### **Step 3: Update `ResearchOutput` Model** (~3 min)

**File**: `abstractcore/processing/basic_deepresearcherC.py`
**Location**: Around line 133

**Current**:
```python
class ResearchOutput(BaseModel):
    """Final research output"""
    title: str
    summary: str
    key_findings: List[str]
    sources_selected: List[Dict[str, Any]]
    research_metadata: Dict[str, Any]
    knowledge_gaps: List[str]
    confidence_score: float
```

**Add**:
```python
class ResearchOutput(BaseModel):
    """Final research output"""
    title: str
    summary: str
    key_findings: List[str]
    sources_selected: List[Dict[str, Any]]
    detailed_report: Dict[str, Any]  # ← ADD THIS
    research_metadata: Dict[str, Any]
    knowledge_gaps: List[str]
    confidence_score: float
```

---

### **Step 4: Build `detailed_report` in `research()` Method** (~5 min)

**File**: `abstractcore/processing/basic_deepresearcherC.py`
**Location**: Around line 298 (inside `research()` method where `ResearchOutput` is built)

**Current**:
```python
# Build output
output = ResearchOutput(
    title=final_report.title,
    summary=final_report.summary,
    key_findings=key_findings_list,
    sources_selected=[...],
    knowledge_gaps=all_final_gaps,
    confidence_score=final_report.confidence,
    research_metadata={...}
)
```

**Add detailed_report field**:
```python
# Build output
output = ResearchOutput(
    title=final_report.title,
    summary=final_report.summary,
    key_findings=key_findings_list,
    sources_selected=[...],
    detailed_report={  # ← ADD THIS
        "sections": [
            {
                "heading": section.heading,
                "content": section.content
            }
            for section in final_report.detailed_sections
        ]
    },
    knowledge_gaps=all_final_gaps,
    confidence_score=final_report.confidence,
    research_metadata={...}
)
```

---

### **Step 5: Update Fallback Cases** (~5 min)

**File**: `abstractcore/processing/basic_deepresearcherC.py`
**Locations**:
- Line 1017: No evidence fallback
- Line 1139: Text fallback
- Line 1150: Error fallback

**For each fallback `SynthesisModel`, add**:
```python
detailed_sections=[
    DetailedSection(
        heading="Report Status",
        content="Fallback message explaining why detailed report couldn't be generated."
    )
]
```

**Example (no evidence fallback)**:
```python
return SynthesisModel(
    title=f"Research Report: {self.context.query}",
    summary="No reliable information was found...",
    findings_with_sources=[...],
    detailed_sections=[  # ← ADD
        DetailedSection(
            heading="Research Status",
            content="No reliable information was found for this query. This may indicate an unknown topic or very limited online presence. No detailed analysis can be provided without verified sources."
        )
    ],
    gaps_with_sources=[...],
    confidence=0.0
)
```

---

### **Step 6: Update testds2.py to Display Detailed Report** (~5 min)

**File**: `testds2.py`
**Location**: After displaying key findings and gaps

**Add**:
```python
# Display detailed report if present
if result_data.get("detailed_report"):
    print(f"\n{'='*80}")
    print("DETAILED RESEARCH REPORT")
    print(f"{'='*80}")

    for section in result_data["detailed_report"].get("sections", []):
        print(f"\n## {section['heading']}")
        print(f"{section['content']}")
```

**And save to JSON**:
```python
result_data = {
    ...
    "detailed_report": result.detailed_report,  # ← ADD THIS
    ...
}
```

---

## Testing Plan

### **Test 1: Quick Validation** (~3 min)
```bash
python -c "from abstractcore.processing import BasicDeepResearcherC; print('✅ Module loads')"
```

### **Test 2: Simple Query** (~2 min)
```python
from abstractcore import create_llm
from abstractcore.processing import BasicDeepResearcherC

llm = create_llm("lmstudio", model="qwen/qwen3-30b-a3b-2507")
researcher = BasicDeepResearcherC(llm, max_sources=5, max_iterations=3, debug=True)
result = researcher.research("What is quantum computing")

# Check detailed_report exists
assert hasattr(result, 'detailed_report')
assert 'sections' in result.detailed_report
print(f"✅ Sections: {len(result.detailed_report['sections'])}")
```

### **Test 3: Full testds2.py Run** (~3 min)
```bash
python testds2.py
```

**Expected Output**:
```
================================================================================
DETAILED RESEARCH REPORT
================================================================================

## Background and Overview
Laurent-Philippe Albou is a French research scientist with a PhD in Bioinformatics...
[2-4 paragraphs]

## Academic Contributions
His academic work spans multiple domains including structural bioinformatics...
[2-4 paragraphs]

## Current Research and Industry Work
Albou currently engages with large language models...
[2-4 paragraphs]
```

---

## Validation Checklist

- [ ] `DetailedSection` model added
- [ ] `SynthesisModel.detailed_sections` field added
- [ ] Synthesis prompt updated with section requirements
- [ ] `ResearchOutput.detailed_report` field added
- [ ] `detailed_report` built in `research()` method
- [ ] All 3 fallback cases updated
- [ ] testds2.py displays detailed report
- [ ] testds2.py saves detailed_report to JSON
- [ ] Module imports without errors
- [ ] Quick test shows sections generated
- [ ] Full test produces readable detailed report
- [ ] Conservative grounding maintained (no over-inference)
- [ ] All sections have source attribution (via evidence)

---

## Expected Outcome

**Before**:
```json
{
  "summary": "Laurent-Philippe Albou is a French research scientist...",
  "key_findings": ["Finding 1", "Finding 2", ...],
  "detailed_report": null  // ❌ MISSING
}
```

**After**:
```json
{
  "summary": "Laurent-Philippe Albou is a French research scientist...",
  "key_findings": ["Finding 1", "Finding 2", ...],
  "detailed_report": {  // ✅ PRESENT
    "sections": [
      {
        "heading": "Background and Overview",
        "content": "Laurent-Philippe Albou is a French research scientist... [300 words]"
      },
      {
        "heading": "Academic Contributions and Publications",
        "content": "His academic work spans multiple domains... [300 words]"
      },
      {
        "heading": "Current Research and Industry Engagement",
        "content": "Albou currently engages with large language models... [300 words]"
      }
    ]
  }
}
```

---

## Notes

1. **Keep it simple**: Use simple Pydantic models to avoid validation failures
2. **Conservative grounding**: Maintain all existing conservative grounding rules
3. **Flexible sections**: Don't force specific section headings - let LLM organize based on evidence
4. **Reasonable length**: 3-5 sections, 150-300 words each = 450-1500 word report
5. **No hallucination**: Every statement must be grounded in evidence

---

## Estimated Time Breakdown

- Step 1 (SynthesisModel): 5 min
- Step 2 (Prompt update): 10 min
- Step 3 (ResearchOutput): 3 min
- Step 4 (Build detailed_report): 5 min
- Step 5 (Fallbacks): 5 min
- Step 6 (testds2.py): 5 min
- Testing: 8 min

**Total**: ~40 minutes

---

## Future Enhancements (Not in this task)

1. **Link exploration**: Follow embedded links for deeper analysis (separate task)
2. **Citation inline**: Include [1], [2] inline citations in detailed_report content
3. **Markdown formatting**: Generate markdown-formatted sections
4. **PDF export**: Export detailed_report as PDF
5. **Section customization**: Allow user to request specific section types

---

## Ready to Implement?

This plan provides step-by-step instructions to add detailed report generation to BasicDeepResearcherC while maintaining all existing functionality (source attribution, conservative grounding, gap tracking).
