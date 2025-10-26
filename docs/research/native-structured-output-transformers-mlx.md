# Native Structured Output Research: HuggingFace Transformers & MLX

**Date**: October 26, 2025
**Research Goal**: Investigate native structured output support for HuggingFace Transformers and MLX providers

---

## Executive Summary

Both HuggingFace Transformers and MLX can support **native structured output** through the **Outlines library**, which is the current SOTA solution for constrained generation in 2025. This would be significantly better than the prompted approach currently used.

### Key Findings

| Provider | Native Support Available? | Best Solution | Implementation Complexity |
|----------|---------------------------|---------------|---------------------------|
| **HuggingFace Transformers** | ✅ Yes (via Outlines) | Outlines library | Medium |
| **MLX** | ✅ Yes (via Outlines) | Outlines library | Medium |

**Recommendation**: Implement Outlines integration for both providers. This would provide server-side schema enforcement similar to what we achieved with Ollama, LMStudio, and HuggingFace GGUF models.

---

## Part 1: HuggingFace Transformers

### Current State

HuggingFace Transformers (non-GGUF models) currently use the **prompted approach** for structured outputs:
- Schema is embedded in the prompt
- LLM tries to follow the schema
- Requires validation retry logic
- Success rate varies by model quality

### Native Support: Outlines Library

**Outlines** is the SOTA solution for constrained generation with HuggingFace Transformers in 2025.

#### What is Outlines?

- **Official Integration**: Runs under the hood on HuggingFace's Inference API
- **Constrained Decoding**: Applies bias on logits to force selection of only tokens that conform to constraints
- **Schema Support**: Pydantic models, JSON schemas, regular expressions, context-free grammars
- **GitHub**: `dottxt-ai/outlines` (actively maintained)
- **HuggingFace Integration**: Official recommendation in HuggingFace documentation

#### How It Works

```
User provides JSON schema
  ↓
Outlines converts to constraint grammar
  ↓
During generation, filters logits at each step
  ↓
Only valid tokens can be selected
  ↓
Guaranteed schema-compliant output
```

#### Performance Characteristics

**Advantages:**
- Guarantees schema compliance (no validation errors)
- Eliminates retry logic for validation
- Works with any transformers model
- Reduces total tokens generated (more efficient)

**Tradeoffs:**
- Adds per-token latency (constraint evaluation overhead)
- Requires additional dependency (`pip install outlines`)
- Schema compilation time (one-time cost per schema)

#### Implementation Example

```python
import outlines
from pydantic import BaseModel

# 1. Load model via Outlines
model = outlines.models.transformers(
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    device="auto"
)

# 2. Define schema
class Person(BaseModel):
    name: str
    age: int
    email: str

# 3. Create constrained generator
generator = outlines.generate.json(model, Person)

# 4. Generate with guaranteed schema compliance
result = generator("Extract: John Doe, 30 years old, john@example.com")
# Returns validated Person instance
```

#### Integration into AbstractCore

**Proposed Changes to HuggingFaceProvider:**

```python
# In _generate_transformers() method

if response_model and PYDANTIC_AVAILABLE:
    try:
        import outlines

        # Wrap model with Outlines
        outlines_model = outlines.models.transformers(
            model=self.model_instance,
            tokenizer=self.tokenizer
        )

        # Create constrained generator
        generator = outlines.generate.json(outlines_model, response_model)

        # Generate with schema enforcement
        result = generator(input_text)

        # Return validated instance
        return GenerateResponse(
            content=result.model_dump_json(),
            model=self.model,
            finish_reason="stop",
            validated_object=result
        )
    except ImportError:
        # Fallback to prompted approach
        pass
```

**Complexity**: Medium
- Requires Outlines dependency
- Need to wrap existing model/tokenizer
- Handle both native and fallback paths
- Streaming would need separate handling

---

## Part 2: MLX

### Current State

MLX provider currently uses the **prompted approach** for structured outputs:
- Schema embedded in prompt
- No server-side enforcement
- Relies on validation retry logic

### Native Support: Outlines with MLX Backend

Outlines **officially supports MLX** as of 2025 via the `mlx_lm` backend.

#### MLX-Specific Implementations

Three options available:

1. **Outlines (Official MLX Backend)** ⭐ RECOMMENDED
   - Most maintained and actively developed
   - Official support from Outlines team
   - Full feature parity with transformers backend
   - Installation: `pip install outlines mlx-lm`

2. **llm-structured-output** (Specialized)
   - Purpose-built for MLX with state machine framework
   - Minimizes backtracking (expensive for LLMs)
   - Direct schema steering (not grammar-based)
   - Installation: `pip install llm-structured-output`
   - Less maintained than Outlines

3. **outlines-mlx** (Adapter)
   - Minimalistic adapter for MLX
   - Bridges Outlines to MLX
   - Installation: `pip install outlinesmlx`
   - Smaller community

#### Recommendation: Use Outlines Official MLX Backend

**Reasons:**
- Most mature and well-maintained
- Consistent API with transformers backend
- Official HuggingFace/Outlines integration
- Active development and support

#### Implementation Example

```python
from outlines import models, generate
from pydantic import BaseModel

# 1. Load MLX model via Outlines
model = models.mlxlm("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

# 2. Define schema
class Task(BaseModel):
    title: str
    priority: str
    assignee: str

# 3. Create constrained generator
generator = generate.json(model, Task)

# 4. Generate with guaranteed schema compliance
result = generator("Create task: Fix bug in auth, high priority, assign to Alice")
# Returns validated Task instance
```

#### Integration into AbstractCore

**Proposed Changes to MLXProvider:**

```python
# In _generate_internal() method

if response_model and PYDANTIC_AVAILABLE:
    try:
        import outlines
        from outlines import models, generate

        # Load model via Outlines (caching recommended)
        if not hasattr(self, '_outlines_model'):
            self._outlines_model = models.mlxlm(self.model)

        # Create constrained generator
        generator = generate.json(self._outlines_model, response_model)

        # Generate with schema enforcement
        result = generator(full_prompt)

        # Return validated instance
        return GenerateResponse(
            content=result.model_dump_json(),
            model=self.model,
            finish_reason="stop",
            validated_object=result
        )
    except ImportError:
        # Fallback to prompted approach
        pass
```

**Complexity**: Medium
- Requires Outlines dependency (`pip install outlines`)
- Need separate model loading for Outlines
- Model caching to avoid repeated loading
- Handle both native and fallback paths

---

## Part 3: Comparison with Current Prompted Approach

### Prompted Approach (Current)

**How it works:**
```
Schema → JSON in prompt → LLM generates → Parse JSON → Validate → Retry if invalid
```

**Characteristics:**
- No external dependencies
- Works with any LLM
- Success rate varies by model (60-95%)
- Requires retry logic for validation failures
- Faster initial setup (no constraint compilation)

### Native Approach (Outlines)

**How it works:**
```
Schema → Compile constraints → Filter logits → Guaranteed valid output
```

**Characteristics:**
- Requires Outlines dependency
- 100% schema compliance guaranteed
- No retry logic needed
- Per-token overhead during generation
- One-time schema compilation cost

### Performance Comparison

| Metric | Prompted | Native (Outlines) | Winner |
|--------|----------|-------------------|--------|
| **Setup Time** | Instant | Schema compilation (~100-500ms) | Prompted |
| **Token Latency** | Normal | +10-50ms per token | Prompted |
| **Success Rate** | 60-95% | 100% | Native |
| **Retries Needed** | 0-3 | 0 | Native |
| **Total Time** | Variable | Predictable | Native* |
| **Token Efficiency** | Lower (extra formatting) | Higher (exact schema) | Native |

*Native is faster when accounting for retries on complex schemas

---

## Part 4: Implementation Recommendations

### Option 1: Full Native Support (Recommended)

**Implement Outlines for both Transformers and MLX**

**Pros:**
- 100% schema compliance guaranteed
- Consistent with Ollama/LMStudio/GGUF implementation
- Eliminates validation retry logic
- Better user experience (predictable, reliable)

**Cons:**
- Additional dependency (Outlines)
- Medium implementation complexity
- Per-token latency overhead
- Need to handle fallback for users without Outlines

**Implementation Effort**: 2-3 days
- Add Outlines integration to HuggingFaceProvider (transformers backend)
- Add Outlines integration to MLXProvider
- Update model capabilities registry
- Update StructuredOutputHandler detection logic
- Comprehensive testing (similar to Ollama/LMStudio tests)
- Documentation updates

### Option 2: Hybrid Approach

**Native for simple schemas, prompted for complex**

**Pros:**
- Best of both worlds
- Optimize for common use cases

**Cons:**
- Added complexity in decision logic
- Harder to maintain
- Inconsistent behavior

**Implementation Effort**: 3-4 days

### Option 3: Keep Prompted (Status Quo)

**Continue using prompted approach**

**Pros:**
- No changes needed
- No additional dependencies
- Works with all models

**Cons:**
- Not competitive with Ollama/LMStudio
- Variable success rate
- Requires retry logic
- User experience inconsistency

---

## Part 5: Technical Implementation Plan

### Phase 1: HuggingFace Transformers Native Support

**Files to Modify:**
1. `abstractcore/providers/huggingface_provider.py`
   - Add Outlines integration to `_generate_transformers()`
   - Detect if Outlines is available
   - Wrap model/tokenizer for Outlines
   - Generate with schema enforcement
   - Fallback to prompted if Outlines unavailable

2. `abstractcore/structured/handler.py`
   - Update `_has_native_support()` to detect HuggingFace transformers with Outlines
   ```python
   def _has_native_support(self, provider) -> bool:
       provider_name = provider.__class__.__name__

       # Existing checks
       if provider_name in ['OllamaProvider', 'LMStudioProvider']:
           return True

       # HuggingFace with transformers backend
       if provider_name == 'HuggingFaceProvider':
           if hasattr(provider, 'model_type') and provider.model_type == 'gguf':
               return True  # Already supported

           # Check for Outlines availability
           try:
               import outlines
               return provider.model_type == 'transformers'
           except ImportError:
               return False

       return False
   ```

3. `abstractcore/providers/registry.py`
   - Update HuggingFace provider features to include `"structured_output"` (already done for GGUF)

4. `abstractcore/assets/model_capabilities.json`
   - Mark transformers models as `"structured_output": "native"` when Outlines is available

**Dependencies:**
```toml
# In pyproject.toml
[project.optional-dependencies]
huggingface = [
    "transformers>=4.30.0",
    "torch>=2.0.0",
    "outlines>=0.1.0",  # NEW
]
```

### Phase 2: MLX Native Support

**Files to Modify:**
1. `abstractcore/providers/mlx_provider.py`
   - Add Outlines MLX integration to `_generate_internal()`
   - Cache Outlines model instance
   - Generate with schema enforcement
   - Fallback to prompted if Outlines unavailable

2. `abstractcore/structured/handler.py`
   - Update `_has_native_support()` to detect MLX with Outlines
   ```python
   # MLX provider with Outlines
   if provider_name == 'MLXProvider':
       try:
           import outlines
           return True
       except ImportError:
           return False
   ```

3. `abstractcore/providers/registry.py`
   - Update MLX provider features to include `"structured_output"`

**Dependencies:**
```toml
[project.optional-dependencies]
mlx = [
    "mlx>=0.29.0",
    "mlx-lm>=0.5.0",
    "outlines>=0.1.0",  # NEW
]
```

### Phase 3: Testing

**Create comprehensive tests:**
1. `tests/structured/test_huggingface_transformers_native.py`
   - Test simple, medium, complex schemas
   - Test multiple transformers models
   - Verify 100% success rate
   - Compare performance with prompted

2. `tests/structured/test_mlx_native.py`
   - Test simple, medium, complex schemas
   - Test multiple MLX models
   - Verify 100% success rate
   - Compare performance with prompted

**Expected Results:**
- 100% schema compliance
- 0% validation errors
- 0 retries needed
- Slight increase in generation time (acceptable tradeoff)

### Phase 4: Documentation

**Update documentation:**
1. `docs/structured-output.md`
   - Add HuggingFace Transformers native support section
   - Add MLX native support section
   - Update provider comparison table
   - Add Outlines installation instructions

2. `CHANGELOG.md`
   - Add entries for native structured output support

3. `README.md`
   - Update feature descriptions

---

## Part 6: Estimated Performance Impact

### HuggingFace Transformers with Outlines

**Simple Schema (3 fields):**
- Compilation time: ~150ms (one-time)
- Per-token overhead: +15ms average
- Total for 50 tokens: ~900ms overhead
- But: 0 retries vs 0-1 retries with prompted
- **Net result: Similar or faster**

**Medium Schema (nested, 10 fields):**
- Compilation time: ~300ms (one-time)
- Per-token overhead: +25ms average
- Total for 100 tokens: ~2800ms overhead
- But: 0 retries vs 1-2 retries with prompted (each retry ~5-10s)
- **Net result: Significantly faster**

**Complex Schema (deep nesting, 20+ fields):**
- Compilation time: ~500ms (one-time)
- Per-token overhead: +40ms average
- Total for 200 tokens: ~8500ms overhead
- But: 0 retries vs 2-3 retries with prompted (each retry ~10-20s)
- **Net result: Much faster**

### MLX with Outlines

Similar characteristics to HuggingFace Transformers:
- Slightly higher per-token overhead on Apple Silicon
- But eliminates retries completely
- Net faster for medium-complex schemas

---

## Part 7: Risk Assessment

### Implementation Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Outlines dependency adds installation complexity | Low | Make it optional, provide clear docs |
| Breaking changes in Outlines API | Medium | Pin version, monitor releases |
| Performance regression for simple schemas | Low | Fallback to prompted for simple cases |
| Compatibility issues with specific models | Medium | Comprehensive testing, fallback path |

### Deployment Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Users without Outlines installed | Low | Auto-fallback to prompted |
| Outlines incompatibility with some environments | Medium | Clear documentation, error messages |
| Memory overhead from dual model loading | Medium | Lazy loading, optional feature |

---

## Part 8: Conclusion & Next Steps

### Summary

1. **HuggingFace Transformers**: Native structured output via Outlines is feasible and recommended
2. **MLX**: Native structured output via Outlines is feasible and recommended
3. **Performance**: Net improvement for medium-complex schemas due to elimination of retries
4. **Reliability**: 100% schema compliance vs 60-95% with prompted approach
5. **Implementation**: Medium complexity, 2-3 days effort

### Recommended Action Plan

**Week 1: HuggingFace Transformers**
- Day 1: Implement Outlines integration in HuggingFaceProvider
- Day 2: Update detection logic and registry
- Day 3: Create comprehensive tests
- Day 4: Documentation updates

**Week 2: MLX**
- Day 1: Implement Outlines integration in MLXProvider
- Day 2: Update detection logic and registry
- Day 3: Create comprehensive tests
- Day 4: Documentation updates

**Week 3: Polish & Release**
- Day 1-2: End-to-end testing
- Day 3: Performance benchmarking
- Day 4: Release preparation
- Day 5: Release v2.5.2 or v2.6.0

### Alternative: Quick Validation

**Before full implementation, validate with a prototype:**
1. Create standalone script testing Outlines with one transformers model
2. Create standalone script testing Outlines with one MLX model
3. Measure actual performance vs prompted approach
4. Validate schema compliance (should be 100%)
5. Decision: Proceed with full implementation or stick with prompted

**Prototype Time**: 4-6 hours

---

## Appendix: Example Code Snippets

### Outlines with HuggingFace Transformers

```python
import outlines
from pydantic import BaseModel
from typing import List
from enum import Enum

# Simple schema
class Person(BaseModel):
    name: str
    age: int
    email: str

# Load model
model = outlines.models.transformers("unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")

# Generate
generator = outlines.generate.json(model, Person)
person = generator("Extract: Alice, 28, alice@example.com")
print(person)  # Person(name='Alice', age=28, email='alice@example.com')

# Complex schema
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str
    description: str
    priority: Priority
    assignees: List[str]
    estimated_hours: int

generator = outlines.generate.json(model, Task)
task = generator("Create bug fix task, high priority, assign to Bob and Carol, 4 hours")
# Guaranteed to match Task schema exactly
```

### Outlines with MLX

```python
from outlines import models, generate
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

# Load MLX model
model = models.mlxlm("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

# Generate
generator = generate.json(model, Product)
product = generator("Extract: iPhone 15 Pro, $999, available")
print(product)  # Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

---

**End of Research Report**
