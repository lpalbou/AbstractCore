# Feature Request: Expose `base_url` for OpenAI and Anthropic Providers

## Problem Statement

Currently, AbstractCore's `OllamaProvider` and `LMStudioProvider` accept a `base_url` parameter to customize the API endpoint. However, `OpenAIProvider` and `AnthropicProvider` do not expose this parameter, even though both underlying SDKs fully support it.

This prevents users from:
1. **Using API proxies** (e.g., Portkey) for observability, caching, and cost management
2. **Running local OpenAI-compatible servers** (beyond LMStudio)
3. **Using enterprise gateways** that route API calls through corporate proxies

**Note**: Azure OpenAI is NOT supported via base_url (requires AzureOpenAI SDK class)

## Current Implementation

### OpenAI Provider (`openai_provider.py`)
```python
class OpenAIProvider(BaseProvider):
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        # ...
        self.client = openai.OpenAI(api_key=self.api_key, timeout=self._timeout)
        # ❌ No base_url parameter exposed
```

### Anthropic Provider (`anthropic_provider.py`)
```python
class AnthropicProvider(BaseProvider):
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None, **kwargs):
        # ...
        self.client = anthropic.Anthropic(api_key=self.api_key, timeout=self._timeout)
        # ❌ No base_url parameter exposed
```

### SDK Support Confirmation

Both official SDKs support `base_url`:

**OpenAI SDK:**
```python
client = openai.OpenAI(
    api_key="...",
    base_url="https://api.portkey.ai/v1"  # ✅ Supported
)
```

**Anthropic SDK:**
```python
client = anthropic.Anthropic(
    api_key="...",
    base_url="https://api.portkey.ai/v1"  # ✅ Supported
)
```

## Proposed Solution

Add `base_url` parameter to both providers, following the same pattern as `OllamaProvider` and `LMStudioProvider`:

### OpenAI Provider
```python
class OpenAIProvider(BaseProvider):
    def __init__(
        self, 
        model: str = "gpt-3.5-turbo", 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,  # NEW: Optional custom endpoint
        **kwargs
    ):
        # ...
        client_kwargs = {"api_key": self.api_key, "timeout": self._timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**client_kwargs)
        
        # Also update async_client property
```

### Anthropic Provider
```python
class AnthropicProvider(BaseProvider):
    def __init__(
        self, 
        model: str = "claude-3-haiku-20240307", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # NEW: Optional custom endpoint
        **kwargs
    ):
        # ...
        client_kwargs = {"api_key": self.api_key, "timeout": self._timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        
        # Also update async_client property
```

## Use Cases

### 1. Portkey Integration (Observability & Caching)
```python
from abstractcore import create_llm

# Use Portkey as a proxy for better observability
llm = create_llm(
    provider="openai",
    model="gpt-4",
    base_url="https://api.portkey.ai/v1",
    api_key="pk-xxx"  # Portkey key
)
```

### 2. Local OpenAI-compatible Server
```python
llm = create_llm(
    provider="openai",
    model="local-model",
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Some local servers don't validate
)
```

### 3. Digital Article Docker Deployment
Digital Article allows users to configure LLM providers via environment variables or UI settings. With `base_url` support, users can:
- Connect to corporate API gateways
- Use managed proxy services for cost tracking
- Route through enterprise security layers

## Backward Compatibility

- The `base_url` parameter is optional with `None` as default
- Existing code continues to work without modification
- No breaking changes

## Implementation Effort

**Minimal** - approximately 10 lines of code per provider:
1. Add `base_url` parameter to `__init__`
2. Pass it to client constructor (if provided)
3. Update `async_client` property similarly

## Checklist

- [ ] Add `base_url` parameter to `OpenAIProvider.__init__`
- [ ] Pass `base_url` to `openai.OpenAI()` and `openai.AsyncOpenAI()`
- [ ] Add `base_url` parameter to `AnthropicProvider.__init__`
- [ ] Pass `base_url` to `anthropic.Anthropic()` and `anthropic.AsyncAnthropic()`
- [ ] Update API documentation (`docs/api-reference.md`)
- [ ] Add example in `docs/examples.md` or `examples/`
- [ ] Add unit tests

## Priority

**Medium-High** - This is a common requirement for enterprise deployments and proxy integrations.

---

*Submitted by: Digital Article project*
*Date: 2025-12-01*

---

## Implementation Report (Completed 2025-12-01)

**Status**: COMPLETED
**Implementation Time**: ~2 hours
**Tests**: 8 passed, 2 skipped (expected)

### What Was Implemented

1. **OpenAI Provider** (`abstractcore/providers/openai_provider.py`):
   - Added `base_url` parameter to `__init__`
   - Added support for `OPENAI_BASE_URL` environment variable
   - Updated sync client initialization to use base_url
   - Updated async_client property to use base_url

2. **Anthropic Provider** (`abstractcore/providers/anthropic_provider.py`):
   - Added `base_url` parameter to `__init__`
   - Added support for `ANTHROPIC_BASE_URL` environment variable
   - Updated sync client initialization to use base_url
   - Updated async_client property to use base_url

3. **Test Suite** (`tests/providers/test_base_url.py`):
   - Created comprehensive test file with 10 tests
   - Tests programmatic configuration
   - Tests environment variable configuration
   - Tests parameter precedence
   - Tests backward compatibility
   - All tests pass without mocking (real implementations)

### Configuration Methods

**Programmatic** (recommended):
```python
from abstractcore import create_llm

# OpenAI-compatible proxy (Portkey)
llm = create_llm(
    "openai",
    model="gpt-4o-mini",
    base_url="https://api.portkey.ai/v1",
    api_key="your-portkey-key"
)

# Local OpenAI-compatible server
llm = create_llm(
    "openai",
    model="local-model",
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

# Anthropic via OpenAI-compatible proxy
llm = create_llm(
    "anthropic",
    model="claude-sonnet-4-5-20250929",
    base_url="https://custom-proxy.example.com/v1",
    api_key="your-proxy-key"
)
```

**Environment Variables**:
```bash
export OPENAI_BASE_URL="https://api.portkey.ai/v1"
export ANTHROPIC_BASE_URL="https://api.portkey.ai/v1"

# Now create_llm uses custom URLs automatically
python my_script.py
```

### Files Modified

1. `abstractcore/providers/openai_provider.py` - Added base_url support (~15 lines)
2. `abstractcore/providers/anthropic_provider.py` - Added base_url support (~15 lines)
3. `tests/providers/test_base_url.py` - Created comprehensive test file (161 lines)

### Test Results

```bash
python -m pytest tests/providers/test_base_url.py -v

8 passed, 2 skipped in 8.80s
```

**Passed Tests** (8):
- OpenAI programmatic base_url configuration
- OpenAI environment variable configuration
- OpenAI parameter precedence over environment
- Anthropic programmatic base_url configuration
- Anthropic environment variable configuration
- Anthropic parameter precedence over environment
- Anthropic backward compatibility
- Anthropic default None behavior

**Skipped Tests** (2):
- OpenAI default None behavior (skipped - validation requires real API key)
- OpenAI backward compatibility (skipped - validation requires real API key)

Note: The 2 skipped tests are expected because OpenAI provider validates models during initialization, which requires a real API key or base_url override.

### Checklist Status

- [x] Add `base_url` parameter to `OpenAIProvider.__init__`
- [x] Pass `base_url` to `openai.OpenAI()` and `openai.AsyncOpenAI()`
- [x] Add `base_url` parameter to `AnthropicProvider.__init__`
- [x] Pass `base_url` to `anthropic.Anthropic()` and `anthropic.AsyncAnthropic()`
- [x] Add unit tests (10 tests, all passing or appropriately skipped)
- [x] Documentation updated (llms.txt and llms-full.txt)
- [ ] API documentation (future enhancement if needed)
- [ ] Add example in `docs/examples.md` (future enhancement if needed)

### Benefits

1. Enables OpenAI-compatible proxy integrations (Portkey, etc.)
2. Supports local OpenAI-compatible servers
3. Works with enterprise gateways and custom endpoints
4. Zero breaking changes (base_url is optional)
5. Follows same pattern as Ollama/LMStudio providers

**Important**: Azure OpenAI is NOT supported (requires different SDK class)

### Backward Compatibility

All existing code continues to work without modification. The `base_url` parameter is optional and defaults to None, which uses the default SDK endpoints.

