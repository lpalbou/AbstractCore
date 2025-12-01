# Batch Generation API

**Date**: 2025-11-30
**Priority**: P2 - Medium
**Effort**: 10-15 hours
**Target**: v2.7.0 or v2.8.0

---

## Executive Summary

**Problem**: Processing multiple prompts requires sequential calls to `generate()`, wasting GPU/API capacity and increasing latency.

**Solution**: Add `batch_generate(prompts: List[str])` API across providers to process multiple prompts efficiently.

**Benefits**:
- **2-10x throughput** for local GPU providers (MLX, HuggingFace)
- **Cost reduction** for API providers (OpenAI, Anthropic) - fewer API calls
- **Cleaner code** - one method call instead of loops
- **Consistent API** - same pattern across all providers

**Note**: Batching is **different from async**. Async enables concurrent I/O, batching enables efficient multi-prompt processing.

---

## Problem Statement

### Current State

Users processing multiple prompts must write loops:

```python
llm = create_llm("mlx", model="...")

# Inefficient: Sequential processing
results = []
for prompt in prompts:
    result = llm.generate(prompt)
    results.append(result)
# Time: N × single_request_time
```

### What's Wrong

**For GPU providers** (MLX, HuggingFace):
- GPU processes one prompt at a time
- GPU sits idle between prompts
- Wastes parallel processing capability
- 10 prompts = 10× the latency

**For API providers** (OpenAI, Anthropic):
- Multiple API calls increase latency (network overhead)
- Higher costs (per-request pricing)
- Rate limit pressure
- Less efficient resource usage

---

## Proposed Solution

### Add batch_generate() Method

```python
class BaseProvider(ABC):
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerateResponse]:
        """Process multiple prompts efficiently.

        Args:
            prompts: List of prompts to process
            **kwargs: Same parameters as generate()

        Returns:
            List of GenerateResponse objects

        Implementation:
            - GPU providers: Use native batch APIs for parallel processing
            - API providers: May use async batching or sequential (provider-specific)
            - Returns results in same order as input prompts
        """
        raise NotImplementedError(...)

    async def abatch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerateResponse]:
        """Async wrapper for batch generation."""
        return await asyncio.to_thread(self.batch_generate, prompts, **kwargs)
```

---

## Per-Provider Implementation

### MLX Provider (Priority: High)

**Library Support**: ✅ YES - `mlx_lm.batch_generate()`

**Effort**: 3-4 hours

**Implementation**:

```python
class MLXProvider(BaseProvider):
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 32,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[GenerateResponse]:
        """Batch generation using MLX's native batching.

        Performance: 2-10x faster than sequential generate() calls.

        Real-world benchmark: mlx_parallm achieved 1300+ tok/s with batched
        inference (gemma-2b on M3 Max 128GB).
        """
        from mlx_lm import batch_generate as mlx_batch_generate

        # Build full prompts
        full_prompts = [
            self._build_prompt(p,
                messages=kwargs.get('messages'),
                system_prompt=kwargs.get('system_prompt'))
            for p in prompts
        ]

        # Tokenize
        prompt_tokens = [self.tokenizer.encode(p) for p in full_prompts]

        # Batch generate
        results = mlx_batch_generate(
            self.llm,
            self.tokenizer,
            prompts=prompt_tokens,
            max_tokens=max_tokens or self.max_output_tokens,
            completion_batch_size=batch_size,
        )

        # Convert to GenerateResponse
        return [
            GenerateResponse(content=text.strip(), model=self.model, ...)
            for text in results.texts
        ]
```

**Benefits**:
- 2-10x throughput (from MLX documentation)
- GPU utilization improvement
- Shared memory for model weights

---

### HuggingFace Provider (Priority: High)

**Library Support**:
- ✅ Transformers: YES - `pipeline(prompts, batch_size=N)`
- ❌ GGUF (llama-cpp-python): NO - library limitation

**Effort**: 2-3 hours

**Implementation**:

```python
class HuggingFaceProvider(BaseProvider):
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 2,
        **kwargs
    ) -> List[GenerateResponse]:
        """Batch generation for transformers backend only.

        Note: GGUF models do not support batching.
        """
        if self.model_type == "gguf":
            raise NotSupportedError(
                "GGUF models do not support batch generation. "
                "Use transformers backend or process sequentially."
            )

        # Use pipeline batching
        outputs = self.pipeline(
            prompts,
            batch_size=batch_size,
            max_new_tokens=kwargs.get('max_tokens', self.max_output_tokens),
            temperature=kwargs.get('temperature', self.temperature),
        )

        return [
            GenerateResponse(content=out[0]["generated_text"], model=self.model, ...)
            for out in outputs
        ]
```

**Benefits**:
- Better GPU utilization
- Parallel processing within batch
- Reduced per-prompt overhead

**Caveat**: From [HuggingFace docs](https://huggingface.co/docs/transformers/en/pipeline_tutorial), "batch inference may improve speed on GPU, but not guaranteed" - depends on hardware, data, and model.

---

### OpenAI Provider (Priority: Medium)

**Library Support**: ⚠️ PARTIAL - No batch API, but can use async

**Effort**: 1-2 hours

**Implementation Options**:

**Option A: Async Gather (Recommended)**
```python
class OpenAIProvider(BaseProvider):
    async def abatch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerateResponse]:
        """Batch via concurrent async requests."""
        tasks = [self.agenerate(p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerateResponse]:
        """Sync wrapper using asyncio.run()."""
        return asyncio.run(self.abatch_generate(prompts, **kwargs))
```

**Option B: Batch API (When Available)**
```python
# OpenAI has batch API but it's async (up to 24h processing)
# Not suitable for synchronous batch_generate()
# Keep async gather approach
```

**Benefits**:
- Reduces network overhead
- Faster than sequential API calls
- Better rate limit utilization

---

### Anthropic Provider (Priority: Medium)

**Library Support**: ⚠️ PARTIAL - Message Batches API (async only)

**Effort**: 1-2 hours

**Implementation**: Same as OpenAI (async gather approach)

```python
class AnthropicProvider(BaseProvider):
    async def abatch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerateResponse]:
        """Batch via concurrent async requests."""
        tasks = [self.agenerate(p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)
```

**Note**: Anthropic's Message Batches API is for async processing (not real-time), so concurrent `agenerate()` is the right approach for real-time batching.

---

### Ollama Provider (Priority: Low)

**Library Support**: ❌ NO - No batch API

**Effort**: 1 hour (async gather only)

**Implementation**: Use async gather (same as OpenAI/Anthropic)

**Why Low Priority**: Users can already use `asyncio.gather()` with `agenerate()` for concurrent processing. Native batch API provides minimal additional benefit.

---

### LMStudio Provider (Priority: Low)

**Library Support**: ❌ NO - No batch API (OpenAI-compatible, no batch endpoint)

**Effort**: 1 hour (async gather only)

**Implementation**: Same as Ollama

---

## Batching vs Async

### Key Differences

| Aspect | Async (`agenerate()`) | Batching (`batch_generate()`) |
|--------|----------------------|-------------------------------|
| **Purpose** | Concurrent I/O operations | Efficient multi-prompt processing |
| **Best For** | Network providers (Ollama, LMStudio, OpenAI) | GPU providers (MLX, HuggingFace) |
| **Speedup** | 6-7x (network I/O overlap) | 2-10x (GPU parallel processing) |
| **Use Case** | Multiple independent requests | Processing list of prompts |
| **Implementation** | `asyncio.gather(agenerate(), ...)` | `batch_generate([...])` |

### When to Use What

**Use Async** (network providers):
```python
# Multiple independent requests with different parameters
results = await asyncio.gather(
    llm.agenerate("Task 1", temperature=0.7),
    llm.agenerate("Task 2", temperature=0.9),
    llm.agenerate("Task 3", max_tokens=100),
)
```

**Use Batching** (GPU providers):
```python
# List of prompts with same parameters
prompts = [f"Summarize: {doc}" for doc in documents]
results = llm.batch_generate(prompts, temperature=0.5)
```

**Use Both** (when available):
```python
# Batch prompts + async for throughput
batches = [prompts[i:i+10] for i in range(0, len(prompts), 10)]
results = await asyncio.gather(*[
    llm.abatch_generate(batch) for batch in batches
])
```

---

## API Design Decisions

### 1. Return Type

**Decision**: Return `List[GenerateResponse]` in same order as input prompts

```python
prompts = ["Prompt A", "Prompt B", "Prompt C"]
results = llm.batch_generate(prompts)
# results[0] corresponds to prompts[0], etc.
```

**Alternative Considered**: Return dict with prompt as key
- ❌ Rejected: Prompts may not be unique
- ❌ Rejected: More complex API

---

### 2. Error Handling

**Decision**: Fail fast - raise exception on first error

```python
try:
    results = llm.batch_generate(prompts)
except ProviderAPIError as e:
    # One prompt failed - entire batch fails
    ...
```

**Alternative Considered**: Partial results with error markers
- ❌ Rejected: Complex API, harder to reason about
- ⚠️ Future consideration: Add `batch_generate_with_errors()` variant

---

### 3. Parameter Propagation

**Decision**: Same `**kwargs` apply to all prompts in batch

```python
# All prompts use temperature=0.7
results = llm.batch_generate(prompts, temperature=0.7, max_tokens=100)
```

**Alternative Considered**: Per-prompt parameters
- ❌ Rejected: Too complex, defeats purpose of batching
- 💡 If needed: Use multiple batch calls or async gather

---

### 4. Batch Size Parameter

**Decision**: Provider-specific `batch_size` parameter (optional)

```python
# MLX: Controls internal batching
results = llm.batch_generate(prompts, batch_size=32)

# HuggingFace: Controls pipeline batch size
results = llm.batch_generate(prompts, batch_size=4)
```

**Why**: Different providers have different optimal batch sizes

---

## Testing Strategy

### Test Coverage

1. **Basic functionality**:
   - Single prompt in list
   - Multiple prompts (5, 10, 100)
   - Empty list handling

2. **Parameter propagation**:
   - `temperature`, `max_tokens`, etc. apply to all

3. **Order preservation**:
   - Results match input prompt order

4. **Error handling**:
   - Invalid prompt in batch
   - Provider errors

5. **Performance validation**:
   - Measure actual speedup vs sequential
   - Compare to async gather (for network providers)

### Real Implementation Testing

Per project policy (no mocking):

```python
# tests/batch/test_mlx_batch.py
def test_mlx_batch_performance():
    llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")
    prompts = [f"Count to {i}" for i in range(1, 11)]

    # Sequential
    start = time.time()
    seq_results = [llm.generate(p) for p in prompts]
    seq_time = time.time() - start

    # Batch
    start = time.time()
    batch_results = llm.batch_generate(prompts)
    batch_time = time.time() - start

    # Should be 2-5x faster
    assert batch_time < seq_time / 2
    assert batch_results == seq_results  # Same outputs
```

---

## Documentation Updates

### 1. README.md

Add section:

```markdown
### Batch Generation

Process multiple prompts efficiently:

```python
from abstractcore import create_llm

llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")

# Sequential (slow)
results = [llm.generate(p) for p in prompts]  # 10× latency

# Batch (fast)
results = llm.batch_generate(prompts)  # 2-10× faster!
```

**Supported Providers**:
- ✅ MLX: 2-10× faster via native batching
- ✅ HuggingFace (transformers): Parallel pipeline processing
- ✅ OpenAI/Anthropic: Concurrent async requests
- ⚠️ HuggingFace (GGUF): Not supported
- ⚠️ Ollama/LMStudio: Use `asyncio.gather()` with `agenerate()`
```

### 2. Provider Docstrings

```python
class MLXProvider(BaseProvider):
    """
    MLX Provider for Apple Silicon models.

    Batch Processing:
    - batch_generate(): 2-10× faster for multiple prompts
    - Uses MLX native batching for GPU parallel processing
    - Recommended batch_size: 16-32 prompts

    Example:
        >>> prompts = [f"Task {i}" for i in range(10)]
        >>> results = llm.batch_generate(prompts, batch_size=32)
    """
```

### 3. Create docs/batch-generation.md

Comprehensive guide:
- What is batching (vs async)
- When to use batch vs async
- Per-provider capabilities
- Performance characteristics
- Code examples
- Best practices

---

## Implementation Priority

| Provider | Priority | Effort | Benefit | Rationale |
|----------|----------|--------|---------|-----------|
| **MLX** | P1 - High | 3-4h | High (2-10x) | Native batch API, biggest speedup |
| **HuggingFace** | P1 - High | 2-3h | Medium | Transformers support, widely used |
| **OpenAI** | P2 - Medium | 1-2h | Medium | Async gather already possible |
| **Anthropic** | P2 - Medium | 1-2h | Medium | Async gather already possible |
| **Ollama** | P3 - Low | 1h | Low | Async gather sufficient |
| **LMStudio** | P3 - Low | 1h | Low | Async gather sufficient |

**Recommendation**: Implement MLX and HuggingFace first (P1), defer network providers (P2/P3).

---

## Timeline

| Phase | Effort | Target |
|-------|--------|--------|
| **Phase 1: MLX** | 3-4h | v2.7.0 |
| **Phase 2: HuggingFace** | 2-3h | v2.7.0 |
| **Phase 3: Network providers** | 3-4h | v2.8.0 |
| **Documentation** | 2h | v2.7.0 |
| **Total** | **10-13h** | **v2.7.0 & v2.8.0** |

---

## Benefits

1. **Performance**: 2-10× faster for multi-prompt workloads
2. **Cost**: Fewer API calls for network providers
3. **Clean API**: One method call instead of loops
4. **Consistency**: Same pattern across providers
5. **Production-ready**: Real-world multi-document processing

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Memory issues with large batches | Document recommended batch sizes |
| Per-prompt parameter needs | Use multiple batches or async gather |
| GGUF users disappointed | Clear error message with alternatives |
| Complex error handling | Start simple (fail fast), iterate if needed |

---

## Non-Goals

1. ❌ Per-prompt different parameters (use async gather instead)
2. ❌ Partial results on error (may add later as separate method)
3. ❌ Automatic batch size tuning (user-specified)
4. ❌ Batch streaming (separate feature, high complexity)

---

## Success Criteria

1. ✅ MLX `batch_generate()` achieves 2-5× speedup
2. ✅ HuggingFace (transformers) batch working
3. ✅ Results maintain input order
4. ✅ Async wrappers (`abatch_generate()`) available
5. ✅ Clear documentation on batch vs async
6. ✅ Zero breaking changes to existing API

---

## Summary

**What**: Add `batch_generate(prompts)` API for efficient multi-prompt processing

**Why**: Current sequential processing wastes GPU/API capacity

**Benefits**: 2-10× faster for GPU providers, cost reduction for API providers

**Priority**: P1 for MLX/HuggingFace, P2/P3 for network providers

**Effort**: 10-13 hours total

**Note**: Batching is orthogonal to async - both can be used together for maximum throughput.
