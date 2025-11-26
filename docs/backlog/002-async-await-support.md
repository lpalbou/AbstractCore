# STRATEGIC-001: Async/Await API Support

**Status**: Proposed
**Priority**: P1 - High (Strategic)
**Effort**: Large (80-120 hours)
**Type**: Major Feature / Architecture Enhancement
**Target Version**: 2.6.0 or 3.0.0 (TBD based on breaking changes)

## Executive Summary

Add comprehensive async/await support throughout AbstractCore to enable high-performance concurrent LLM operations. This strategic enhancement positions AbstractCore for modern async-first Python applications while maintaining full backwards compatibility with synchronous APIs.

**Expected Benefits**:
- 3-10x throughput improvement for batch operations
- Better integration with FastAPI, async web frameworks
- Efficient concurrent multi-provider requests
- Non-blocking I/O for improved resource utilization
- Industry-standard async patterns

**Scope**: Complete async variants across entire stack while preserving sync APIs

---

## Problem Statement

### Current Limitations

**Architecture**: AbstractCore is currently 100% synchronous

```python
# Current (synchronous only)
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("Hello")  # Blocks until complete

# Can't do this efficiently:
providers = [create_llm("openai", ...), create_llm("anthropic", ...)]
responses = [p.generate("Hello") for p in providers]  # Sequential, slow
```

**Pain Points**:

1. **Sequential Bottleneck**: Batch operations run sequentially
   - 10 requests × 1s each = 10s total (with async: ~1s total)
   - Multi-provider comparisons are slow
   - No parallelization of independent operations

2. **Resource Underutilization**: Threads for concurrency are inefficient
   - Thread overhead: ~8MB per thread
   - GIL limits true parallelism
   - Context switching overhead

3. **Framework Integration**: Poor fit for async ecosystems
   - FastAPI routes block event loop
   - Can't use with async web frameworks efficiently
   - Incompatible with async middleware patterns

4. **Modern Python**: Async is now standard for I/O operations
   - httpx has excellent AsyncClient support
   - Python 3.11+ optimizations favor async
   - Community expects async APIs

### Real-World Impact

**Scenario 1: Batch Summarization**
```python
# Current (synchronous)
documents = load_documents(1000)  # 1000 documents
summaries = []
for doc in documents:
    summary = llm.generate(f"Summarize: {doc}")  # Blocks
    summaries.append(summary)
# Time: 1000 × 1.5s = 1500s (25 minutes!)

# With async (proposed)
async def summarize_batch(documents):
    tasks = [llm.agenerate(f"Summarize: {doc}") for doc in documents]
    return await asyncio.gather(*tasks, return_exceptions=True)

summaries = await summarize_batch(documents)
# Time: ~150s (2.5 minutes) - 10x faster!
```

**Scenario 2: Multi-Provider Consensus**
```python
# Current (synchronous)
providers = [openai_llm, anthropic_llm, ollama_llm]
responses = [p.generate("Evaluate sentiment") for p in providers]
# Time: 3 × 1s = 3s (sequential)

# With async (proposed)
responses = await asyncio.gather(*[p.agenerate("Evaluate sentiment") for p in providers])
# Time: ~1s (concurrent)
```

---

## Proposed Solution

### Design Principles

1. **Backwards Compatible**: Existing sync API unchanged, async is additive
2. **Consistent Naming**: `async` variants use `a` prefix (`generate` → `agenerate`)
3. **Implementation Flexibility**: Default async uses `asyncio.to_thread()`, providers can optimize
4. **Progressive Enhancement**: Start with core, expand incrementally

### Architecture: Dual API Pattern

```
AbstractCore
├── Synchronous API (current)
│   ├── generate()
│   ├── get_capabilities()
│   └── ...
│
└── Asynchronous API (new)
    ├── agenerate()
    ├── aget_capabilities()
    └── ...
```

**Pattern**: Each sync method has async counterpart with `a` prefix

---

## Implementation Plan

### Phase 1: Core Interface & Base Provider (16-24 hours)

#### 1.1 Update AbstractCoreInterface

```python
# abstractcore/core/interface.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator
import asyncio

class AbstractCoreInterface(ABC):
    """Abstract base class for all LLM providers."""

    # Existing sync method (unchanged)
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response from the LLM (synchronous)."""
        pass

    # New async method
    async def agenerate(self, prompt: str, **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """
        Generate response from the LLM (asynchronous).

        Default implementation uses asyncio.to_thread() to run sync version.
        Providers can override for native async support.

        Args:
            prompt: Input prompt
            **kwargs: Same as generate()

        Returns:
            GenerateResponse or AsyncIterator for streaming
        """
        # Default: Run sync version in thread pool
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    # Existing sync method (unchanged)
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities (synchronous)."""
        pass

    # New async method
    async def aget_capabilities(self) -> List[str]:
        """
        Get list of capabilities (asynchronous).

        Default implementation uses thread pool.
        """
        return await asyncio.to_thread(self.get_capabilities)
```

#### 1.2 Update BaseProvider

```python
# abstractcore/providers/base.py

class BaseProvider(AbstractCoreInterface, ABC):
    """Base provider with telemetry, events, and async support."""

    # Existing sync implementation (unchanged)
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerateResponse:
        pass

    # Async implementation with telemetry
    async def agenerate(self, prompt: str, **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """
        Async generation with full telemetry support.

        Providers should override _agenerate_internal() for native async.
        """
        start_time = time.time()
        response = None
        error = None
        success = True

        try:
            # Call provider-specific async implementation
            response = await self._agenerate_internal(prompt, **kwargs)
            return response

        except Exception as e:
            error = e
            success = False
            raise

        finally:
            # Track generation with telemetry (sync, runs in background)
            self._track_generation(prompt, response, start_time, success, error)

    async def _agenerate_internal(self, prompt: str, **kwargs) -> GenerateResponse:
        """
        Internal async generation method.

        Default: Use thread pool for sync implementation.
        Providers can override for native async (recommended).
        """
        return await asyncio.to_thread(self._generate_internal, prompt, **kwargs)

    @abstractmethod
    def _generate_internal(self, prompt: str, **kwargs) -> GenerateResponse:
        """Internal sync generation (providers implement this)."""
        pass
```

### Phase 2: Provider-Specific Async Implementations (24-32 hours)

Implement native async for HTTP-based providers using `httpx.AsyncClient`.

#### 2.1 OpenAI Provider (Native Async)

```python
# abstractcore/providers/openai_provider.py

class OpenAIProvider(BaseProvider):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)

        # Keep sync client
        self.client = openai.OpenAI(...)

        # Add async client
        self._async_client = None

    @property
    def async_client(self):
        """Lazy-load async client."""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                api_key=self.api_key,
                timeout=self._timeout
            )
        return self._async_client

    async def _agenerate_internal(self, prompt: str, **kwargs) -> GenerateResponse:
        """Native async implementation using AsyncOpenAI."""
        messages = self._prepare_messages(prompt, kwargs)

        # Use native async client
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_output_tokens', self.max_output_tokens),
            stream=kwargs.get('stream', False)
        )

        return self._process_response(response)

    def unload(self):
        """Close both sync and async clients."""
        super().unload()
        if self._async_client:
            # Schedule async close
            asyncio.create_task(self._async_client.close())
```

#### 2.2 Ollama Provider (httpx AsyncClient)

```python
# abstractcore/providers/ollama_provider.py

class OllamaProvider(BaseProvider):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)

        # Keep sync client
        self.client = httpx.Client(...)

        # Add async client (lazy-loaded)
        self._async_client = None

    @property
    def async_client(self):
        """Lazy-load async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._timeout)
            )
        return self._async_client

    async def _agenerate_internal(self, prompt: str, **kwargs) -> GenerateResponse:
        """Native async using httpx.AsyncClient."""
        payload = self._prepare_payload(prompt, kwargs)

        # Use async HTTP client
        response = await self.async_client.post("/api/generate", json=payload)
        response.raise_for_status()

        return self._process_response(response.json())

    def unload(self):
        """Close both clients."""
        super().unload()
        if self._async_client:
            asyncio.create_task(self._async_client.aclose())
```

#### 2.3 Local Providers (MLX, HuggingFace)

```python
# abstractcore/providers/mlx_provider.py

class MLXProvider(BaseProvider):
    """Local inference - async uses thread pool."""

    async def _agenerate_internal(self, prompt: str, **kwargs) -> GenerateResponse:
        """
        Async implementation using thread pool.

        Local inference is CPU-bound, so we use thread pool
        to prevent blocking the event loop.
        """
        return await asyncio.to_thread(self._generate_internal, prompt, **kwargs)

# Note: Same pattern for HuggingFaceProvider
```

### Phase 3: Session Async Support (12-16 hours)

```python
# abstractcore/core/session.py

class BasicSession:
    """Session with async support."""

    def generate(self, prompt: str, **kwargs) -> GenerateResponse:
        """Synchronous generation (existing)."""
        # ... existing code unchanged ...

    async def agenerate(self, prompt: str, **kwargs) -> GenerateResponse:
        """
        Asynchronous generation with conversation management.

        Args:
            prompt: User prompt
            **kwargs: Generation parameters

        Returns:
            GenerateResponse from LLM
        """
        # Pre-processing (sync)
        user_message = self.add_message('user', prompt, **kwargs)

        # Async generation
        response = await self.provider.agenerate(
            prompt=prompt,
            messages=self.get_history(include_system=True),
            system_prompt=self.system_prompt,
            tools=self.tools,
            **kwargs
        )

        # Post-processing (sync)
        assistant_message = self.add_message('assistant', response.content)

        return response

    async def aload_from_file(self, file_path: str) -> 'BasicSession':
        """Async file loading."""
        return await asyncio.to_thread(self.load_from_file, file_path)

    async def asave(self, file_path: str, **kwargs):
        """Async file saving."""
        await asyncio.to_thread(self.save, file_path, **kwargs)
```

### Phase 4: Event System Async Support (8-12 hours)

```python
# abstractcore/events/__init__.py

class AsyncEventEmitter:
    """Async-compatible event emitter."""

    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._async_listeners: Dict[EventType, List[Callable]] = {}

    def on(self, event_type: EventType, handler: Callable):
        """Register sync event handler."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)

    def on_async(self, event_type: EventType, handler: Callable):
        """Register async event handler."""
        if event_type not in self._async_listeners:
            self._async_listeners[event_type] = []
        self._async_listeners[event_type].append(handler)

    async def aemit(self, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None):
        """
        Async event emission.

        Runs sync handlers in thread pool, async handlers natively.
        """
        event = Event(
            type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source
        )

        # Run sync handlers in thread pool
        if event_type in self._listeners:
            tasks = [
                asyncio.to_thread(handler, event)
                for handler in self._listeners[event_type]
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Run async handlers natively
        if event_type in self._async_listeners:
            tasks = [
                handler(event)
                for handler in self._async_listeners[event_type]
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
```

### Phase 5: Tool Execution Async Support (8-12 hours)

```python
# abstractcore/tools/core.py

async def aexecute_tools(
    tool_calls: List[Dict[str, Any]],
    available_tools: List[Callable],
    timeout: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Execute tool calls asynchronously.

    Args:
        tool_calls: List of tool call dicts
        available_tools: Available tool functions
        timeout: Per-tool timeout

    Returns:
        List of tool results
    """
    async def execute_single_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single tool with error handling."""
        try:
            # Find tool function
            tool_func = next(
                (t for t in available_tools if t.__name__ == tool_call['name']),
                None
            )

            if not tool_func:
                return {"error": f"Tool {tool_call['name']} not found"}

            # Check if tool is async
            if asyncio.iscoroutinefunction(tool_func):
                # Native async tool
                result = await asyncio.wait_for(
                    tool_func(**tool_call['arguments']),
                    timeout=timeout
                )
            else:
                # Sync tool - run in thread pool
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool_func, **tool_call['arguments']),
                    timeout=timeout
                )

            return {"result": result}

        except asyncio.TimeoutError:
            return {"error": f"Tool {tool_call['name']} timed out"}
        except Exception as e:
            return {"error": str(e)}

    # Execute all tools concurrently
    results = await asyncio.gather(
        *[execute_single_tool(call) for call in tool_calls],
        return_exceptions=True
    )

    return results
```

### Phase 6: Factory & High-Level APIs (8-12 hours)

```python
# abstractcore/core/factory.py

async def acreate_llm(
    provider: str,
    model: Optional[str] = None,
    **kwargs
) -> AbstractCoreInterface:
    """
    Async factory function for creating LLM providers.

    Returns provider instance ready for async operations.
    For most providers, this is identical to create_llm(),
    but allows for async initialization if needed.

    Args:
        provider: Provider name
        model: Model name
        **kwargs: Provider-specific parameters

    Returns:
        Provider instance with async support

    Example:
        >>> llm = await acreate_llm("openai", model="gpt-4o-mini")
        >>> response = await llm.agenerate("Hello")
    """
    # For now, same as sync (no async initialization needed)
    return create_llm(provider, model, **kwargs)

# Or keep it simple: acreate_llm = create_llm (alias)
```

### Phase 7: Documentation (12-16 hours)

Create comprehensive documentation:

1. **Async Guide** (`docs/async-guide.md`)
   - When to use async vs sync
   - Migration guide from sync to async
   - Best practices and patterns
   - Performance considerations
   - Common pitfalls and solutions

2. **API Reference Updates**
   - Document all async methods
   - Add async examples to existing docs
   - Update architecture diagrams

3. **Examples**
   - Async batch processing
   - Concurrent multi-provider requests
   - FastAPI integration
   - Async streaming responses

### Phase 8: Testing (16-24 hours)

```python
# tests/async/test_async_providers.py

import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_generation():
    """Test basic async generation."""
    llm = create_llm("openai", model="gpt-4o-mini")
    response = await llm.agenerate("Hello")
    assert response.content

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent async requests."""
    llm = create_llm("openai", model="gpt-4o-mini")

    # Execute 10 requests concurrently
    tasks = [llm.agenerate(f"Count to {i}") for i in range(10)]
    responses = await asyncio.gather(*tasks)

    assert len(responses) == 10
    assert all(r.content for r in responses)

@pytest.mark.asyncio
async def test_async_session():
    """Test async session management."""
    llm = create_llm("openai", model="gpt-4o-mini")
    session = BasicSession(provider=llm)

    response1 = await session.agenerate("Hello")
    response2 = await session.agenerate("How are you?")

    assert len(session.messages) == 4  # 2 user + 2 assistant

@pytest.mark.asyncio
async def test_async_streaming():
    """Test async streaming responses."""
    llm = create_llm("openai", model="gpt-4o-mini")

    chunks = []
    async for chunk in await llm.agenerate("Count to 10", stream=True):
        chunks.append(chunk)

    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_async_error_handling():
    """Test async error handling."""
    llm = create_llm("openai", model="invalid-model")

    with pytest.raises(ModelNotFoundError):
        await llm.agenerate("Hello")
```

**Total Estimated Time**: 104-140 hours

---

## Testing & Verification

### Functional Tests

```bash
# Run all async tests
pytest tests/async/ -v -k async

# Test all providers
pytest tests/providers/test_*_provider.py -k async

# Integration tests
pytest tests/async/test_integration.py -v
```

### Performance Benchmarks

```python
# tests/async/test_performance.py

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_batch_performance():
    """Compare sync vs async batch performance."""
    llm = create_llm("openai", model="gpt-4o-mini")

    # Sync baseline
    start = time.time()
    sync_results = [llm.generate(f"Test {i}") for i in range(100)]
    sync_time = time.time() - start

    # Async comparison
    start = time.time()
    async_results = await asyncio.gather(
        *[llm.agenerate(f"Test {i}") for i in range(100)]
    )
    async_time = time.time() - start

    print(f"Sync: {sync_time:.2f}s")
    print(f"Async: {async_time:.2f}s")
    print(f"Speedup: {sync_time / async_time:.2f}x")

    # Verify async is significantly faster
    assert async_time < sync_time * 0.4  # At least 2.5x faster
```

---

## Success Criteria

1. **API Completeness**: All sync methods have async counterparts
2. **Performance**: 3-10x improvement for batch operations
3. **Backwards Compatibility**: Zero breaking changes to sync API
4. **Test Coverage**: >90% coverage for async code paths
5. **Documentation**: Comprehensive async guide and examples
6. **Provider Support**: All 6 providers support async

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking changes | Medium | High | Extensive testing, feature flags |
| Performance regressions | Low | Medium | Comprehensive benchmarking |
| Async complexity | High | Medium | Clear documentation, examples |
| Event loop conflicts | Medium | Medium | Careful event loop management |
| Memory leaks | Low | High | Resource cleanup tests |

---

## Dependencies

**Code Dependencies**:
- httpx >= 0.24.0 (already has AsyncClient)
- openai >= 1.0.0 (already has AsyncOpenAI)
- anthropic >= 0.25.0 (already has AsyncAnthropic)

**Python Version**:
- Minimum: Python 3.9 (current requirement)
- Recommended: Python 3.11+ for async optimizations

---

## Backwards Compatibility

**Breaking Changes**: **NONE**

**Migration**: Optional - users can adopt async at their own pace

**Compatibility Mode**: Sync and async coexist

---

## Version Decision: 2.6.0 vs 3.0.0

**Recommendation**: **2.6.0** (Minor)

**Rationale**:
- Zero breaking changes (async is additive)
- Sync API unchanged
- Follows semver: new features without breaking changes = minor

**Alternative**: 3.0.0 if we want to:
- Signal major architectural shift
- Change default behaviors
- Remove deprecated code
- Major marketing push

---

## Rollout Plan

1. **Alpha**: Internal testing (2-3 weeks)
2. **Beta**: Selected users (2-3 weeks)
3. **RC**: Release candidate (1 week)
4. **GA**: General availability

**Feature Flags**: None needed (async is opt-in)

---

## Follow-up Actions

After implementation:

1. **Async-First Examples**: Create showcase applications
2. **Performance Benchmarks**: Publish detailed benchmarks
3. **Framework Integrations**: FastAPI, Starlette examples
4. **Community Feedback**: Gather usage patterns
5. **Optimization**: Profile and optimize hot paths

---

## References

- httpx Async Client: https://www.python-httpx.org/async/
- Python Async Best Practices: https://docs.python.org/3/library/asyncio.html
- OpenAI Async Client: https://github.com/openai/openai-python#async-usage
- FastAPI Async Patterns: https://fastapi.tiangolo.com/async/

---

## Dependent Features

The following features depend on async support being implemented:
- **MCP Integration** (`008-mcp-integration.md`): MCP SDK is async-native, requires async/await

---

**Document Version**: 1.1
**Created**: 2025-11-25
**Updated**: 2025-11-25 (added MCP dependency)
**Author**: Expert Code Review
**Status**: Ready for Review & Decision
