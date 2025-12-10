# Feature Request: Custom Assessment Criteria for generate_assessment()

## Problem

Currently, `BasicSession.generate_assessment()` only supports predefined criteria:
- clarity
- coherence
- relevance
- completeness
- actionability

These work well for general conversation quality assessment, but downstream applications need domain-specific criteria.

### Example: Data Analysis Quality Assessment

Digital Article needs to assess data analysis quality with criteria like:
- `logical_coherence` - Are the results logically consistent?
- `result_plausibility` - Are the findings plausible given the data?
- `assumption_validity` - Were statistical assumptions checked?
- `interpretation_quality` - Is the interpretation appropriate?
- `completeness` - Are limitations acknowledged?

### Current API

```python
# BasicSession.generate_assessment signature (session.py:942)
def generate_assessment(self, criteria: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
```

The `criteria` parameter only enables/disables predefined criteria, it doesn't allow custom criteria names.

## Proposed Enhancement

### Option A: Accept Custom Criteria Definitions

Allow passing criteria with descriptions, not just boolean toggles:

```python
# New signature
def generate_assessment(
    self,
    criteria: Optional[Dict[str, Union[bool, str]]] = None
) -> Dict[str, Any]:
    """
    Args:
        criteria: Either:
            - Dict[str, bool]: Enable/disable predefined criteria
            - Dict[str, str]: Custom criteria with descriptions

    Example:
        # Predefined criteria (current behavior)
        session.generate_assessment({"clarity": True, "coherence": False})

        # Custom criteria (new behavior)
        session.generate_assessment({
            "logical_coherence": "Are the results logically consistent with the input?",
            "result_plausibility": "Are the findings plausible and reasonable?",
            "assumption_validity": "Were underlying assumptions properly checked?"
        })
    """
```

### Option B: JudgmentCriteria Enhancement

Make `JudgmentCriteria` support arbitrary criteria:

```python
# In processing/basic_judge.py

class JudgmentCriteria:
    # Current: Fixed fields
    clarity: bool = True
    coherence: bool = True

    # New: Allow custom criteria
    custom_criteria: Optional[Dict[str, str]] = None  # name -> description
```

### Option C: Separate Domain-Specific Assessment Method

Add a new method for domain-specific assessments:

```python
def generate_domain_assessment(
    self,
    domain: str,  # e.g., "data_analysis", "code_review", "medical_diagnosis"
    criteria: Dict[str, str]  # Custom criteria with descriptions
) -> Dict[str, Any]:
```

## Use Case from Digital Article

Digital Article's `AnalysisCritic` service (analysis_critic.py) attempts:

```python
# Current code (FAILS)
assessment_raw = session.generate_assessment(
    criteria=[
        "logical_coherence",
        "result_plausibility",
        "assumption_validity",
        "interpretation_quality",
        "completeness"
    ],
    include_score=True  # This parameter doesn't exist!
)
```

With enhanced API, it could be:

```python
# With Option A
assessment_raw = session.generate_assessment(
    criteria={
        "logical_coherence": "Are the results logically consistent?",
        "result_plausibility": "Are the findings plausible given the data?",
        "assumption_validity": "Were statistical assumptions checked?",
        "interpretation_quality": "Is the interpretation appropriate?",
        "completeness": "Are limitations acknowledged?"
    }
)
```

## Benefits

1. **Flexibility**: Different domains can define relevant quality criteria
2. **Reusability**: AbstractCore becomes useful for specialized applications
3. **Backward Compatibility**: Existing bool-based criteria still work
4. **Extensibility**: New domains don't require AbstractCore code changes

## Impact

- **Breaking changes**: None (additive feature)
- **Files affected**:
  - `abstractcore/core/session.py` (generate_assessment)
  - `abstractcore/processing/basic_judge.py` (JudgmentCriteria)

## Priority

Medium - Current workaround is to use `BasicJudge.evaluate()` directly with custom context string.

---

**Submitted by**: Digital Article project
**Date**: 2025-12-06
**Related error**: `BasicSession.generate_assessment() got an unexpected keyword argument 'include_score'`
