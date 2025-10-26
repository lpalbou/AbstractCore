# Strategy B Simplifications - Performance Optimizations

**Date**: 2025-10-26
**Goal**: Reduce Strategy B execution time from 270+ seconds to <90 seconds

---

## Changes Made

### 1. Reduced Retry Attempts (Line 217) ✅
```python
# BEFORE:
self.retry_strategy = FeedbackRetry(max_attempts=3)

# AFTER:
self.retry_strategy = FeedbackRetry(max_attempts=1)  # Reduced from 3 to 1 for speed
```

**Impact**: With native structured outputs (LMStudio), we rarely need retries anyway. This prevents unnecessary retry overhead.
**Savings**: ~10-20 seconds (if any validation errors occur)

### 2. Disabled Intent Analysis (Lines 267-272) ✅
```python
# BEFORE:
logger.info("🧠 Phase 1: Analyzing intent and creating plan")
if self.intent_analyzer:
    intent = self._analyze_query_intent(query)

# AFTER:
logger.info("🧠 Phase 1: Creating research plan")
# Skipping intent analysis - saves 60-80 seconds
# if self.intent_analyzer:
#     intent = self._analyze_query_intent(query)
```

**Impact**: Intent analysis was taking 78 seconds and not providing significant value for research quality.
**Savings**: **~78 seconds** (28% of total time!)

### 3. Simplified Research Planning Prompt (Lines 346-355) ✅
```python
# BEFORE (detailed, verbose):
prompt = f"""Create a comprehensive hierarchical research plan for this question.

Research Question: {query}
Depth Level: {depth}

Create a structured plan with:
1. Clear research goal statement
2. 3-5 main research categories (themes to explore)
3. 8-15 atomic questions (specific, answerable questions)
   - Each question should be self-contained and specific
   - Assign to appropriate category
   - Set priority (1=critical, 2=important, 3=supplementary)
4. Dependencies between questions (which must be answered first)
5. Estimated depth needed

Atomic questions should be:
- Specific and concrete (not broad)
- Independently answerable
- Collectively comprehensive
- Prioritized by importance

Return a structured plan."""

# AFTER (concise, focused):
prompt = f"""Create a research plan for: {query}
Depth: {depth}

Generate:
1. Research goal (one sentence)
2. 3-4 research categories
3. 5-8 specific questions to answer
4. Question dependencies (optional)

Keep questions focused and independently answerable."""
```

**Impact**: Shorter prompt = faster generation, fewer questions = faster execution overall.
**Savings**: **~20-25 seconds** (faster generation + fewer questions to process)

### 4. Simplified Source Quality Assessment (Lines 563-569) ✅
```python
# BEFORE (detailed evaluation):
prompt = f"""Assess the quality of this source for answering: {question.question}

Title: {source.get('title', 'Unknown')}
URL: {source.get('url', 'Unknown')}
Snippet: {source.get('snippet', 'No snippet')}

Evaluate:
1. Credibility score (0-1): Based on domain authority, publication type
2. Recency score (0-1): How current is the information
3. Relevance score (0-1): How relevant to the question
4. Authority indicators (up to 3): Why this source is authoritative
5. Red flags (up to 3): Any quality concerns
6. Should include: Whether to include this source

Return structured assessment."""

# AFTER (simple relevance check):
prompt = f"""Is this source useful for: {question.question}?

Title: {source.get('title', 'Unknown')}
Snippet: {source.get('snippet', 'No snippet')}

Rate relevance (0-1) and say if it should be included."""
```

**Impact**: Was taking 11 seconds per source × 10 sources = 110 seconds total. Now should be ~3-4s per source.
**Savings**: **~70-80 seconds** (42% of total time!)

---

## Expected Performance Improvement

### Before Optimizations:
```
Intent Analysis:        78.35s  (28.9%)
Research Planning:      49.10s  (18.1%)
Source Quality (×10):  114.07s  (42.0%)
Content Analysis:       24.82s  (9.1%)
Query Generation:        5.14s  (1.9%)

TOTAL: ~271 seconds (4.5 minutes)
```

### After Optimizations:
```
Intent Analysis:         0.00s  (DISABLED)         ✅ -78s
Research Planning:      25.00s  (simplified)       ✅ -24s
Source Quality (×10):   35.00s  (simplified×10)    ✅ -79s
Content Analysis:       24.82s  (unchanged)
Query Generation:        5.14s  (unchanged)

TOTAL: ~90 seconds (1.5 minutes)
```

**Overall Improvement**: 271s → 90s (**67% faster!**)

---

## What Wasn't Changed

### Kept (Still Valuable):
1. ✅ Hierarchical planning (simplified but still present)
2. ✅ Atomic question decomposition
3. ✅ Source quality filtering (simplified)
4. ✅ Knowledge graph building
5. ✅ Contradiction detection
6. ✅ Structured final report

### Could Still Optimize (Future):
1. **Knowledge Graph Building**: Currently builds graph but doesn't use it much
2. **Contradiction Detection**: Could be simplified or made optional
3. **Parallel Execution**: Atomic questions still processed sequentially
4. **Content Analysis**: Could be skipped for simple queries

---

## Testing

### Test Command:
```bash
python -c "
from abstractcore import create_llm
from abstractcore.processing import BasicDeepResearcherB
import time

llm = create_llm('lmstudio', model='qwen/qwen3-next-80b')
researcher = BasicDeepResearcherB(llm, max_sources=10, extract_full_content=False)

start = time.time()
result = researcher.research('What is deep learning?', depth='shallow')
duration = time.time() - start

print(f'Duration: {duration:.2f}s')
print(f'Confidence: {result.confidence_score:.2f}')
print(f'Sources: {len(result.sources_selected)}')
"
```

### Expected Results:
- **Before**: 270-350 seconds (4.5-6 minutes)
- **After**: 80-110 seconds (1.3-1.8 minutes)
- **Improvement**: 67-70% faster

---

## Comparison to Strategy A

### Strategy A (Unmodified):
- Duration: 50-60 seconds
- Success Rate: 100%
- Confidence: 0.95-0.96
- Architecture: ReAct + Tree of Thoughts
- Approach: Parallel exploration with fallbacks

### Strategy B (Optimized):
- Duration: 80-110 seconds (estimated)
- Success Rate: TBD (testing now)
- Confidence: TBD
- Architecture: Hierarchical Planning
- Approach: Sequential with simplified prompts

### Why Strategy A is Still Faster:
1. No hierarchical planning step
2. Parallel execution (vs sequential)
3. Simpler prompts throughout
4. Multiple fallback mechanisms

### When to Use Optimized Strategy B:
- When you want more structured planning
- When you prefer hierarchical decomposition
- When you want explicit dependency tracking
- When quality > speed (even with optimizations)

---

## Files Modified

1. **abstractcore/processing/basic_deepresearcherB.py**:
   - Line 217: Reduced retry attempts 3→1
   - Lines 267-272: Disabled intent analysis
   - Lines 346-355: Simplified research planning prompt
   - Lines 563-569: Simplified source quality assessment prompt

---

## Summary

**Key Changes**:
1. ❌ Removed intent analysis (-78s)
2. ✂️ Simplified prompts (-100s total)
3. 🔢 Reduced retries 3→1 (-10-20s)

**Total Savings**: ~190 seconds (67% faster)

**New Duration**: 80-110 seconds (vs 270s before)

**Trade-offs**:
- Lost: Detailed intent analysis, comprehensive quality metrics
- Kept: Hierarchical planning, atomic questions, structured output
- Gained: 3-4x faster execution

---

**Status**: ✅ Optimizations applied, testing in progress
**Next**: Verify performance with qwen3-next-80b
