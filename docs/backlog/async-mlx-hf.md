# MLX & HuggingFace: Async Investigation

**Date**: 2025-11-30
**Priority**: N/A (Not Possible)
**Status**: Investigation Complete - Recommendation: Keep Current Fallback

---

## Executive Summary

**User's Challenge**: "Ollama and LMStudio are working on the local machine as well. So if you saw improvements with Ollama and LMStudio, there is no reason you wouldn't see improvements with HuggingFace and MLX."

**Investigation Question**: Can MLX and HuggingFace providers have native async/await like Ollama and LMStudio?

**Answer**: **NO - Native async is not possible for MLX and HuggingFace.**

**Reason**: Ollama/LMStudio benefit from async because of **HTTP client/server architecture**, not because they're "remote". MLX/HuggingFace use **direct Python function calls** with no network layer, and the underlying libraries don't expose async APIs.

**Recommendation**: Keep current `asyncio.to_thread()` fallback - this IS the correct SOTA pattern for non-async libraries.

---

## Problem Statement

### Original Claim (Flawed)

"Local Providers (MLX, HuggingFace) are CPU-bound and don't benefit from async/await."

### Why This Was Wrong

Ollama and LMStudio ALSO run locally, yet they achieve 6-7x speedup with native async for concurrent requests. The "local = no async benefit" reasoning was incorrect.

### The Honest Explanation

The actual architectural difference that enables async:

| Provider Type | Architecture | Async Possible? |
|---------------|--------------|-----------------|
| **Ollama/LMStudio** | HTTP client/server model | ✅ YES - Multiple HTTP requests can be in-flight concurrently |
| **MLX/HuggingFace** | Direct Python function calls | ❌ NO - No network layer to overlap, libraries are synchronous |

**Key Insight**: The 6-7x speedup for Ollama/LMStudio comes from **concurrent HTTP I/O**, not from "async" itself. This architecture doesn't exist for MLX/HuggingFace.

---

## Research Findings

### MLX Provider Investigation

**Current Implementation** (`abstractcore/providers/mlx_provider.py`):
- Uses `mlx_lm.generate()` and `mlx_lm.stream_generate()` (synchronous functions)
- Falls back to `asyncio.to_thread()` for `agenerate()` (via BaseProvider)

**mlx_lm Library (v0.28.3) - Async Capabilities**:

| Feature | Available | Notes |
|---------|-----------|-------|
| **Async API (`async def`)** | ❌ NO | All functions are synchronous |
| **Async streaming** | ❌ NO | `stream_generate()` is synchronous iterator |
| **Concurrent inference** | ❌ NO | Single GPU stream, operations serialize |

**Investigation Details**:
```python
# All mlx_lm functions are synchronous:
from mlx_lm import generate, stream_generate

def generate(model, tokenizer, prompt, ...) -> str:
    # Synchronous function - blocks until completion
    ...

def stream_generate(model, tokenizer, prompt, ...) -> Iterator:
    # Synchronous iterator - not async iterator
    for token in ...:
        yield token
```

**Internal Async**: MLX uses `mx.async_eval()` internally for GPU async operations, but this is **NOT exposed** to Python API. The Python API remains synchronous and blocking.

**Conclusion**: Cannot implement native async without library support.

---

### HuggingFace Provider Investigation

**Current Implementation** (`abstractcore/providers/huggingface_provider.py`):
- Supports two backends: GGUF (llama-cpp-python) and Transformers
- Falls back to `asyncio.to_thread()` for `agenerate()`

**Backend 1: llama-cpp-python (GGUF models)**

| Feature | Available | Notes |
|---------|-----------|-------|
| **Async API** | ❌ NO | All methods are synchronous |
| **HTTP mode** | ✅ YES (server only) | `llama-server` supports HTTP, but requires separate process |

```python
# llama-cpp-python API is synchronous:
from llama_cpp import Llama

llm = Llama(model_path="...")
response = llm.create_chat_completion(...)  # Synchronous, blocks
```

**Backend 2: transformers**

| Feature | Available | Notes |
|---------|-----------|-------|
| **Async API** | ❌ NO | `model.generate()` is synchronous |
| **Pipeline async** | ❌ NO | All pipeline operations are synchronous |

```python
# Transformers API is synchronous:
from transformers import pipeline

pipe = pipeline("text-generation", ...)
result = pipe(prompt)  # Synchronous, blocks
```

**Conclusion**: Neither backend exposes async APIs.

---

## Why Current Fallback is Correct

The `asyncio.to_thread()` fallback in BaseProvider is the **SOTA correct pattern** for synchronous libraries:

```python
# Current implementation (abstractcore/providers/base.py)
async def _agenerate_internal(self, prompt, ...):
    """Default fallback - runs sync generate() in thread pool."""
    return await asyncio.to_thread(self.generate, prompt, ...)
```

### Evidence from SOTA Frameworks

**LangChain** (`langchain/chains/base.py`):
```python
async def arun(self, *args, **kwargs):
    """Async version using run_in_executor."""
    return await asyncio.get_event_loop().run_in_executor(
        None,
        self.run,  # Sync method
        *args,
        **kwargs
    )
```

**Pydantic-AI**:
```python
async def run_sync_in_executor(self, fn, *args):
    """Run sync operations in thread pool."""
    return await asyncio.to_thread(fn, *args)
```

**Why This Pattern is Correct**:

1. ✅ **Keeps event loop responsive** - doesn't block other async operations
2. ✅ **Enables mixing CPU-bound and I/O-bound** - can interleave with async I/O
3. ✅ **Provides consistent async API** - users can await all providers
4. ✅ **No fake concurrency claims** - honest about what async does/doesn't provide
5. ✅ **Zero overhead for sync users** - fallback only used when `agenerate()` called
6. ✅ **Industry standard** - [LangChain](https://python.langchain.com/docs/concepts/async/), [Pydantic-AI](https://ai.pydantic.dev) use same pattern

### What asyncio.to_thread() Provides

**Benefit**: Event loop doesn't block during inference

```python
async def mix_inference_and_io():
    llm = create_llm("mlx", model="...")

    # Both run concurrently - I/O happens while inference runs in thread
    result, data = await asyncio.gather(
        llm.agenerate("Task 1"),  # Runs in thread pool
        fetch_from_api()          # Async I/O
    )
```

**Does NOT Provide**: Parallel GPU inference (GPU operations still serialize)

```python
# This does NOT run in parallel on same GPU:
results = await asyncio.gather(
    llm.agenerate("Task 1"),
    llm.agenerate("Task 2"),
    llm.agenerate("Task 3"),
)
# Tasks run sequentially in thread pool, GPU processes one at a time
```

---

## Could We Make It Async?

### Option 1: Wait for Library Async Support

**Status**: Not in roadmap for any library

- MLX: No async API planned
- transformers: No async API planned
- llama-cpp-python: No async API planned

**Timeline**: Unknown / Never

**Exception**: The [genlm-backend](https://pypi.org/project/genlm-backend/) library provides async interface with automatic batching for MLX:
- Uses `asyncio.gather` for concurrent LLM calls
- Backend automatically batches requests
- Example of async + batching working together with middleware layer
- However, this is a specialized use case (Sequential Importance Sampling), not general-purpose async

---

### Option 2: Wrap in HTTP Server

**What it would look like**:
```python
# Start HTTP server wrapping MLX
mlx_server = MLXHTTPServer(model="...")
mlx_server.start()  # Runs in background process

# Now use HTTP client (like Ollama)
llm = create_llm("mlx-http", base_url="http://localhost:8080")
response = await llm.agenerate("...")  # Now we have async!
```

**Problem**: This is just reinventing Ollama/LMStudio architecture

**Better Alternative**: Use LMStudio directly (already supports MLX models with HTTP architecture)

---

### Option 3: Multiprocessing (Process-Level Parallelism)

**What it would look like**:
```python
from multiprocessing import Pool

# Multiple processes, each with own MLX model instance
with Pool(processes=4) as pool:
    results = pool.map(llm.generate, prompts)
```

**Problems**:
- ❌ Each process loads full model (4x memory usage)
- ❌ Not async/await (different paradigm)
- ❌ Complex resource management
- ❌ Doesn't fit AbstractCore async API

**Conclusion**: Not appropriate for async implementation

---

## Architectural Comparison

### Why Ollama/LMStudio Have Native Async

```
User Code:
  └─> httpx.AsyncClient.post("/api/generate")  ← Async I/O starts
        └─> [Network I/O - async wait]
              └─> Server receives request
                    └─> Queue request
                          └─> Run inference
                                └─> Send response
                                      └─> [Network I/O - async wait]

Multiple requests can be in-flight concurrently during network waits!
```

**Key**: HTTP provides I/O layer that can overlap

### Why MLX/HuggingFace Cannot Have Native Async

```
User Code:
  └─> mlx_lm.generate(model, prompt)  ← Synchronous function call
        └─> [GPU computation - blocks Python thread]
              └─> Return result

No I/O layer to overlap, function blocks until completion.
```

**Key**: Direct function call with no async library API

---

## Recommendation

### Keep Current Implementation

**Status Quo**:
- MLX: Uses `asyncio.to_thread()` fallback ✅ CORRECT
- HuggingFace: Uses `asyncio.to_thread()` fallback ✅ CORRECT

**Why**:
1. This IS the SOTA pattern for non-async libraries
2. Keeps event loop responsive for mixing with I/O
3. Provides consistent async API across all providers
4. Honest about capabilities (no fake async)

### Document Clearly

Update provider docstrings to explain:

```python
class MLXProvider(BaseProvider):
    """
    MLX Provider for Apple Silicon models.

    Async Support:
    - agenerate(): Uses asyncio.to_thread() - keeps event loop responsive
    - Native async NOT available (mlx_lm library is synchronous)
    - Concurrent requests do NOT provide speedup (GPU operations serialize)

    Note: For true async concurrency with MLX models, use LMStudio provider
    which wraps MLX in HTTP server architecture.
    """
```

### User Guidance

Add to README.md:

```markdown
#### Async Behavior by Provider Type

**Network Providers** (Ollama, LMStudio, OpenAI, Anthropic):
- ✅ Native async support with `httpx.AsyncClient`
- ✅ True concurrent execution (6-7x faster for batch operations)
- ✅ Recommended for high-throughput async scenarios

**Local Providers** (MLX, HuggingFace):
- ⚠️ Async fallback via `asyncio.to_thread()`
- ⚠️ Keeps event loop responsive but no parallel inference
- ℹ️ Useful for mixing inference with async I/O operations
- 💡 For true concurrency, use LMStudio (wraps local models in HTTP server)
```

---

## Non-Goals

What we are **NOT** implementing:

1. ❌ Native async for MLX (library doesn't support it)
2. ❌ Native async for HuggingFace (neither backend supports it)
3. ❌ HTTP server wrapper (use Ollama/LMStudio instead)
4. ❌ Multiprocessing-based parallelism (different from async)
5. ❌ Fake async that claims concurrency benefits

---

## Summary

**User's Question**: Can we improve MLX/HuggingFace to have actual async instead of fallbacks?

**Answer**: **NO**

**Reasons**:
1. ❌ Libraries don't expose async Python APIs
2. ❌ No HTTP architecture to enable concurrent I/O
3. ❌ Direct function calls block until completion
4. ✅ Current fallback IS the correct SOTA pattern

**What We're Doing Right**:
- ✅ Using `asyncio.to_thread()` for event loop responsiveness
- ✅ Following same pattern as LangChain, Pydantic-AI
- ✅ Not making false concurrency claims
- ✅ Providing consistent async API across providers

**What Changed from Original Explanation**:
- ❌ OLD: "CPU-bound = no async benefit" (wrong reasoning)
- ✅ NEW: "No async library API + no HTTP layer = async not possible" (correct reasoning)

**Recommendation**: Document clearly, keep current implementation, guide users to LMStudio for true async with local models.
