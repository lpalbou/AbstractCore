# Debug Mode Feature - DeepSearch V2

## Problem

When research finds no usable sources, it's unclear WHY each source was rejected:

```
⚠️  No findings in iteration 1
   Query used: "Laurent-Philippe Albou" site:fr OR site:edu...
   Expected to find: Academic publications, media interviews...
   Explored 10 potential sources but none yielded usable content
```

**User question**: "Can we see ALL the potential sources initially selected?"

## Solution

Added `debug=True` mode that shows:
1. All URLs found from search
2. Detailed rejection reasons for each source
3. Quality scores breakdown
4. Rejection summary statistics

---

## Usage

### Enable Debug Mode

```python
from abstractcore.processing import BasicDeepSearchV2
from abstractcore.core.factory import create_llm

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")

# Enable debug mode
searcher = BasicDeepSearchV2(llm, debug=True)

report = searcher.research("What is Python?")
```

### Command Line (if applicable)

```bash
# Normal mode (minimal logging)
python -m abstractcore.apps.deepsearchv2 "query"

# Debug mode (detailed logging)
python -m abstractcore.apps.deepsearchv2 "query" --debug
```

---

## Debug Output Examples

### 1. All Sources Initially Found

**Normal mode**:
```
├─ Found 10 URLs from search
```

**Debug mode**:
```
├─ Found 10 URLs from search
│
│  🔍 DEBUG: All 10 sources initially found:
│    1. Laurent-Philippe Albou | LinkedIn
│       URL: https://fr.linkedin.com/in/laurent-philippe-albou
│    2. Bioinformatics Expert - USC
│       URL: https://www.usc.edu/people/albou
│    3. Gene Ontology Consortium
│       URL: https://geneontology.org/about/team
│    ... (10 total)
│
```

---

### 2. Fetch Error / Too Short

**Normal mode**:
```
├─ [1/10] Fetching: Laurent-Philippe Albou | LinkedIn
│  └─ ⚠️  Skipped (fetch error or too short)
```

**Debug mode**:
```
├─ [1/10] Fetching: Laurent-Philippe Albou | LinkedIn
│  └─ ⚠️  Skipped (fetch error or too short)
│     🔍 DEBUG: Content preview: ⚠️  Empty Response - Likely Anti-Bot Protection...
│     🔍 DEBUG: Content length: 287 bytes
```

Shows exactly WHY it was skipped (e.g., anti-bot protection, 0 bytes, error message).

---

### 3. No Relevant Content

**Normal mode**:
```
├─ [2/10] Fetching: USC Faculty Directory
│  └─ ⚠️  Skipped (no relevant content)
```

**Debug mode**:
```
├─ [2/10] Fetching: USC Faculty Directory
│  └─ ⚠️  Skipped (no relevant content)
│     🔍 DEBUG: Content extracted but not relevant to query
│     🔍 DEBUG: Original content length: 5,234 bytes
```

Shows the page had content, but it wasn't relevant to the search query.

---

### 4. Quality Score Too Low (Rejected)

**Normal mode**:
```
├─ [3/10] Fetching: Random Blog Post
│  └─ 📊 Quality: 42.3/100 (A:15 R:12 Q:15)
│  └─ ❌ Rejected (score too low: 42.3)
```

**Debug mode**:
```
├─ [3/10] Fetching: Random Blog Post
│  └─ 📊 Quality: 42.3/100 (A:15 R:12 Q:15)
│  └─ ❌ Rejected (score too low: 42.3)
│     🔍 DEBUG: Authority: 15.0/33
│     🔍 DEBUG: Relevance: 12.0/33
│     🔍 DEBUG: Quality: 15.3/33
│     🔍 DEBUG: Reasoning: Low-authority personal blog with thin content
│     🔍 DEBUG: Threshold: 50 (rejected below this)
```

Shows detailed breakdown of:
- Why authority score was low
- Why relevance score was low
- Why quality score was low
- The LLM's reasoning for the low score

---

### 5. Quality Score Good (Accepted)

**Normal mode**:
```
├─ [4/10] Fetching: Gene Ontology Resource Publication
│  └─ 📊 Quality: 87.5/100 (A:29 R:31 Q:28)
│  └─ ✅ Accepted (score: 87.5)
```

**Debug mode**:
```
├─ [4/10] Fetching: Gene Ontology Resource Publication
│  └─ 📊 Quality: 87.5/100 (A:29 R:31 Q:28)
│  └─ ✅ Accepted (score: 87.5)
│     🔍 DEBUG: Reasoning: High-authority academic publication from
                 Nature Genetics, highly relevant to bioinformatics
                 research, well-structured content
```

Shows why it was accepted (high authority, relevance, quality).

---

### 6. Exception During Fetch

**Normal mode**:
```
│  └─ ❌ Failed to fetch https://broken-site.com: Connection timeout
```

**Debug mode**:
```
│  └─ ❌ Failed to fetch https://broken-site.com: Connection timeout
│     🔍 DEBUG: Exception type: TimeoutError
│     🔍 DEBUG: Full error: HTTPConnectionPool timeout after 15s
```

Shows the exception type and full error message.

---

### 7. Rejection Summary

**Normal mode only shows**:
```
└─ ✅ Collected 2 high-quality sources
```

**Debug mode adds**:
```
└─ ✅ Collected 2 high-quality sources

🔍 DEBUG: Rejection Summary
   Total sources evaluated: 10
   Accepted: 2
   Rejected: 8
   Acceptance rate: 20.0%
```

---

## Complete Example Output

```
🔍 ITERATION 1: Broad Discovery
  ├─ 🔍 Search Query: "Laurent-Philippe Albou" site:edu OR site:org
  └─ 🎯 Expected: Academic profiles and publications

      ├─ Found 10 URLs from search
      │
      │  🔍 DEBUG: All 10 sources initially found:
      │    1. Laurent-Philippe Albou - Google Scholar
      │       URL: https://scholar.google.com/citations?user=...
      │    2. USC Bioinformatics Faculty
      │       URL: https://www.usc.edu/faculty/albou
      │    ... (8 more)
      │

      ├─ [1/10] Fetching: Laurent-Philippe Albou - Google Scholar
      │  └─ ⚠️  Skipped (fetch error or too short)
      │     🔍 DEBUG: Content preview: ⚠️  Empty Response - Anti-Bot...
      │     🔍 DEBUG: Content length: 0 bytes

      ├─ [2/10] Fetching: USC Bioinformatics Faculty
      │  └─ 📊 Quality: 78.5/100 (A:27 R:26 Q:25)
      │  └─ ✅ Accepted (score: 78.5)
      │     🔍 DEBUG: Reasoning: Authoritative university source...

      ... (8 more sources evaluated)

      └─ ✅ Collected 2 high-quality sources

      🔍 DEBUG: Rejection Summary
         Total sources evaluated: 10
         Accepted: 2
         Rejected: 8
         Acceptance rate: 20.0%
```

---

## Benefits

### Before Debug Mode
```
⚠️  No findings in iteration 1
   Explored 10 potential sources but none yielded usable content
```

**User reaction**: "What were those 10 sources? Why were they all rejected?"

### With Debug Mode
```
🔍 DEBUG: All 10 sources initially found:
  1. Site A (rejected: anti-bot protection)
  2. Site B (rejected: quality score 35/100 - low authority)
  3. Site C (rejected: no relevant content)
  ...

🔍 DEBUG: Rejection Summary
   Total: 10, Accepted: 0, Rejected: 10
   Main reasons: 5 anti-bot, 3 low quality, 2 not relevant
```

**User reaction**: "Ah, most sources are blocked by anti-bot protection. Let me try a different query or use an API instead."

---

## Implementation Details

### Files Modified

**`abstractcore/processing/basic_deepsearchv2.py`**:

1. **Line 131**: Added `debug: bool = False` parameter to `__init__`
2. **Line 143**: Updated docstring
3. **Line 171**: Stored `self.debug = debug`
4. **Lines 550-557**: Shows all URLs found initially
5. **Lines 578-581**: Debug info for fetch errors
6. **Lines 589-591**: Debug info for no relevant content
7. **Lines 617-626**: Debug info for quality score details
8. **Lines 630-632**: Debug info for exceptions
9. **Lines 637-646**: Rejection summary statistics

### Parameter

```python
def __init__(
    self,
    llm: Optional[AbstractCoreInterface] = None,
    max_tokens: int = 32000,
    max_output_tokens: int = 8000,
    timeout: Optional[float] = None,
    temperature: float = 0.1,
    verbose: bool = True,
    debug: bool = False  # <-- NEW
):
```

---

## Use Cases

### 1. Debugging Query Problems
When searches return no results, debug mode shows exactly which sites were tried and why they failed.

### 2. Understanding Anti-Bot Protection
See which sites have anti-bot protection so you can use their APIs instead.

### 3. Tuning Quality Thresholds
See the quality score distribution to adjust the acceptance threshold (currently 50).

### 4. Improving Queries
See which types of sources are being found and adjust your query to target better sources.

### 5. Identifying Systemic Issues
If ALL sources fail for the same reason (e.g., timeout), it indicates a network/config issue.

---

## Summary

**Before**: Mysterious "no findings" warnings with no details

**After**: Complete transparency into:
- ✅ All sources found
- ✅ Why each was accepted/rejected
- ✅ Detailed quality scores
- ✅ Rejection statistics
- ✅ Error messages

**Usage**: Just add `debug=True` when initializing:
```python
searcher = BasicDeepSearchV2(llm, debug=True)
```

**Result**: Full visibility into the research process for debugging and optimization.
