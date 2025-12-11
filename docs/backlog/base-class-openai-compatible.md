# OpenAI-Compatible Base Class Refactoring

**Priority**: P3 - Low (Code Quality)
**Effort**: 8-12 hours
**Target**: v2.7.0 or later

## Executive Summary

Refactor the OpenAI-compatible provider architecture to reduce code duplication between `OpenAICompatibleProvider`, `VLLMProvider`, and `LMStudioProvider` by introducing a shared base class.

## Current State (v2.6.4)

Three providers share similar OpenAI-compatible HTTP communication code:

| Provider | Lines | OpenAI-Compatible Core | Provider-Specific Features |
|----------|-------|------------------------|---------------------------|
| **OpenAICompatibleProvider** | 764 | 100% | None (pure generic) |
| **LMStudioProvider** | 783 | ~90% | LMStudio-specific model management |
| **VLLMProvider** | 823 | ~70% | Guided decoding, Multi-LoRA, beam search |

**Code Duplication**: ~500-600 lines of identical or near-identical code across the three providers:
- HTTP client setup (httpx sync/async)
- OpenAI-compatible message formatting
- SSE streaming parsing (`data: ` prefix, `[DONE]` marker)
- Response extraction (`choices[0].message.content`)
- Usage tracking (`prompt_tokens`, `completion_tokens`)
- Embeddings API (`/v1/embeddings`)
- Model listing (`/v1/models`)

## Proposed Architecture

### BaseOpenAICompatibleProvider

Create a new base class that handles all OpenAI-compatible communication:

```python
class BaseOpenAICompatibleProvider(BaseProvider):
    """
    Base class for providers using OpenAI-compatible API format.

    Handles:
    - HTTP client management (httpx sync/async)
    - OpenAI message formatting
    - SSE streaming parsing
    - Response extraction
    - Embeddings API
    - Model listing
    """

    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        # ... HTTP client setup

    def _get_headers(self) -> Dict[str, str]:
        """Optional Bearer token authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, messages, stream, **kwargs) -> Dict[str, Any]:
        """Build OpenAI-compatible request payload."""
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": self._get_provider_max_tokens_param(kwargs),
            # ... standard OpenAI params
        }

    def _extract_response(self, result: Dict) -> Tuple[str, str, Dict]:
        """Extract content, finish_reason, usage from OpenAI format."""
        # ...

    def _stream_generate(self, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Standard SSE streaming for OpenAI-compatible APIs."""
        # ...

    async def _async_stream_generate(self, payload: Dict[str, Any]) -> AsyncIterator[GenerateResponse]:
        """Async SSE streaming."""
        # ...

    def embed(self, input_text, **kwargs) -> Dict[str, Any]:
        """OpenAI-compatible embeddings API."""
        # ...

    def list_available_models(self, **kwargs) -> List[str]:
        """GET /v1/models"""
        # ...
```

### Refactored Providers

```python
class OpenAICompatibleProvider(BaseOpenAICompatibleProvider):
    """Generic OpenAI-compatible provider."""

    def __init__(self, model: str = "default", base_url: Optional[str] = None,
                 api_key: Optional[str] = None, **kwargs):
        base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL") or "http://localhost:8080/v1"
        api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        super().__init__(model, base_url, api_key, **kwargs)
        self.provider = "openai-compatible"

class LMStudioProvider(BaseOpenAICompatibleProvider):
    """LMStudio-specific provider."""

    def __init__(self, model: str = "local-model", base_url: Optional[str] = None, **kwargs):
        base_url = base_url or os.getenv("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1"
        super().__init__(model, base_url, api_key=None, **kwargs)
        self.provider = "lmstudio"

class VLLMProvider(BaseOpenAICompatibleProvider):
    """vLLM-specific provider with advanced features."""

    def __init__(self, model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                 base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        base_url = base_url or os.getenv("VLLM_BASE_URL") or "http://localhost:8000/v1"
        api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        super().__init__(model, base_url, api_key, **kwargs)
        self.provider = "vllm"

    def _build_payload(self, messages, stream, **kwargs) -> Dict[str, Any]:
        """Override to add vLLM-specific parameters via extra_body."""
        payload = super()._build_payload(messages, stream, **kwargs)

        # Add vLLM extensions
        extra_body = {}
        if "guided_regex" in kwargs:
            extra_body["guided_regex"] = kwargs["guided_regex"]
        # ... other vLLM params

        if extra_body:
            payload["extra_body"] = extra_body

        return payload

    def load_adapter(self, adapter_name: str, adapter_path: str) -> str:
        """vLLM-specific Multi-LoRA management."""
        # ...
```

## Benefits

1. **Reduced Code Duplication**: ~500-600 lines of shared code moved to base class
2. **Single Source of Truth**: OpenAI-compatible behavior defined once
3. **Easier Maintenance**: Bug fixes in one place benefit all providers
4. **Consistent Behavior**: All providers handle OpenAI format identically
5. **Simpler Provider Creation**: New OpenAI-compatible providers need <100 lines

## Migration Plan

### Phase 1: Create Base Class (~4 hours)
1. Create `BaseOpenAICompatibleProvider` in `abstractcore/providers/base_openai_compatible.py`
2. Extract shared code from `OpenAICompatibleProvider` (the purest implementation)
3. Write comprehensive tests for base class functionality

### Phase 2: Migrate OpenAICompatibleProvider (~1 hour)
1. Refactor to inherit from `BaseOpenAICompatibleProvider`
2. Remove duplicated code
3. Verify all tests still pass

### Phase 3: Migrate LMStudioProvider (~2 hours)
1. Refactor to inherit from `BaseOpenAICompatibleProvider`
2. Remove duplicated code, keep LMStudio-specific model management
3. Verify all tests still pass

### Phase 4: Migrate VLLMProvider (~3 hours)
1. Refactor to inherit from `BaseOpenAICompatibleProvider`
2. Remove duplicated code, keep vLLM-specific features (guided decoding, LoRA, beam search)
3. Override `_build_payload()` to add `extra_body` parameters
4. Verify all tests still pass

### Phase 5: Documentation (~2 hours)
1. Update provider documentation
2. Document base class architecture
3. Create guide for adding new OpenAI-compatible providers

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Breaking Changes** | High | Comprehensive test suite, careful migration |
| **Provider-Specific Behavior Differences** | Medium | Document differences, use override methods |
| **Performance Regression** | Low | Base class should have zero overhead |
| **Increased Complexity** | Low | Better abstraction reduces overall complexity |

## Success Criteria

1. ✅ All 3 providers use `BaseOpenAICompatibleProvider`
2. ✅ ~500-600 lines of code eliminated
3. ✅ All existing tests pass unchanged
4. ✅ Zero breaking changes to public API
5. ✅ Documentation updated

## Alternative Approaches Considered

### Alternative 1: Keep Current Structure
- **Pros**: No migration risk, providers remain independent
- **Cons**: Continued code duplication, maintenance burden
- **Verdict**: Not sustainable long-term

### Alternative 2: Mixin Pattern
- **Pros**: More flexible, can mix multiple behaviors
- **Cons**: More complex, harder to understand inheritance
- **Verdict**: Over-engineered for this use case

### Alternative 3: Composition Over Inheritance
- **Pros**: More flexible, better separation
- **Cons**: Requires major refactoring, changes public API
- **Verdict**: Too invasive for v2.7.0

## Dependencies

None - this is pure internal refactoring

## Future Enhancements

Once base class is established, easier to add:
- OpenAI-compatible retry logic
- OpenAI-compatible rate limiting
- OpenAI-compatible response caching
- Standard error handling across all providers

## References

- Current implementations:
  - `abstractcore/providers/openai_compatible_provider.py`
  - `abstractcore/providers/lmstudio_provider.py`
  - `abstractcore/providers/vllm_provider.py`
- Similar patterns:
  - `BaseProvider` for all providers
  - Media handlers (OpenAIMediaHandler, LocalMediaHandler)
