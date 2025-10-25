# Logging Improvements - DeepSearch V2

## Problem

Warning messages were too cryptic and lacked context. Example:

```
06:16:40 [WARNING] abstractcore.processing.basic_deepsearchv2:       └─ ⚠️  No URLs extracted from search results
06:16:40 [WARNING] abstractcore.processing.basic_deepsearchv2:   └─ ⚠️  No findings in iteration 1
```

**User complaint**: "those warnings are far too cryptic to be useful; which url? search results? what was searched etc..."

## Solution

Added comprehensive context to all warning messages.

---

## Improvements Made

### 1. "No URLs extracted from search results" (Line 533)

**Before**:
```python
logger.warning(f"      └─ ⚠️  No URLs extracted from search results")
```

**After**:
```python
search_preview = search_results[:500] if isinstance(search_results, str) else str(search_results)[:500]
logger.warning(f"      └─ ⚠️  No URLs extracted from search results")
logger.warning(f"         Query: \"{search_query}\"")
logger.warning(f"         Search returned: {search_preview}...")
logger.warning(f"         This may indicate: (1) web_search failed, (2) results have no links, (3) parsing issue")
```

**Now shows**:
- The exact search query used
- What the search actually returned (first 500 chars)
- Possible causes for the issue

---

### 2. "No findings in iteration" (Line 390)

**Before**:
```python
logger.warning(f"  └─ ⚠️  No findings in iteration {iteration}")
```

**After**:
```python
logger.warning(f"  └─ ⚠️  No findings in iteration {iteration}")
logger.warning(f"      Query used: \"{search_query.query}\"")
logger.warning(f"      Expected to find: {search_query.expected_findings}")
logger.warning(f"      Explored {explore_limit} potential sources but none yielded usable content")
logger.warning(f"      This may indicate: (1) query too specific, (2) no web results, (3) all URLs failed to fetch")
```

**Now shows**:
- The specific search query that was tried
- What was expected to be found
- How many sources were attempted (explore_limit)
- Possible causes for the failure

---

### 3. "Gap analysis failed" (Line 850)

**Before**:
```python
logger.warning(f"Gap analysis failed: {e}")
```

**After**:
```python
logger.warning(f"Gap analysis failed: {e}")
logger.warning(f"  Query being analyzed: \"{original_query}\"")
logger.warning(f"  Iteration: {iteration}/{max_iterations}")
logger.warning(f"  Current findings: {len(current_findings)} characters")
logger.warning(f"  Falling back to iteration-based continuation")
```

**Now shows**:
- The query being analyzed for gaps
- Current iteration progress
- How much content has been collected so far
- What fallback strategy will be used

---

### 4. "Query generation failed" (Line 503)

**Before**:
```python
logger.warning(f"Query generation failed, using fallback: {e}")
```

**After**:
```python
logger.warning(f"Query generation failed, using fallback: {e}")
logger.warning(f"  Original query: \"{original_query}\"")
logger.warning(f"  Iteration: {iteration}")
logger.warning(f"  Intent type: {intent.query_type}")
logger.warning(f"  Falling back to simple query formulation")
```

Plus added info log:
```python
logger.info(f"  Using fallback query: \"{fallback_query}\"")
```

**Now shows**:
- The original query that failed to be refined
- Which iteration this occurred in
- The detected intent type
- What fallback query will be used instead

---

### 5. "Intent analyzer failed" (Line 271)

**Before**:
```python
logger.warning(f"Intent analyzer failed, using fallback: {e}")
```

**After**:
```python
logger.warning(f"Intent analyzer failed, using fallback: {e}")
logger.warning(f"  Query: \"{query}\"")
logger.warning(f"  Switching to simple LLM-based intent analysis")
```

**Now shows**:
- The query being analyzed
- What fallback method will be used

---

### 6. "Intent analysis failed, using default" (Line 312)

**Before**:
```python
logger.warning(f"Intent analysis failed, using default: {e}")
```

**After**:
```python
logger.warning(f"Intent analysis failed, using default: {e}")
logger.warning(f"  Query: \"{query}\"")
logger.warning(f"  Both intent analyzer and LLM fallback failed")
logger.warning(f"  Using default 'exploratory' intent")
```

Plus added info log:
```python
logger.info(f"  Defaulting to exploratory research strategy")
```

**Now shows**:
- The query being analyzed
- That both primary and fallback methods failed
- What default strategy will be used

---

## Benefits

### Before
```
⚠️  No findings in iteration 1
```

**User reaction**: "WTF? What was being searched? Why did it fail?"

### After
```
⚠️  No findings in iteration 1
    Query used: "Python async programming best practices"
    Expected to find: detailed guides and examples
    Explored 10 potential sources but none yielded usable content
    This may indicate: (1) query too specific, (2) no web results, (3) all URLs failed to fetch
```

**User reaction**: "Ah, the query was too specific. Let me adjust it."

---

## Summary

**Files modified**: `abstractcore/processing/basic_deepsearchv2.py`

**Lines changed**:
- Line 533-542: URL extraction failure
- Line 390-395: No findings warning
- Line 503-515: Query generation fallback
- Line 271-273: Intent analyzer fallback
- Line 312-327: Intent analysis ultimate fallback
- Line 850-858: Gap analysis failure

**Total improvements**: 6 warning contexts enhanced

**Result**: All warnings now include:
1. ✅ The exact query/data being processed
2. ✅ Current state/iteration/progress
3. ✅ What was expected vs what happened
4. ✅ Possible causes for the issue
5. ✅ What fallback/next action will be taken

**User benefit**: Debugging is now actually possible - users can understand WHY something failed and WHAT to fix.
