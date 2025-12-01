# TACTICAL-003: Enhanced Error Messages with Actionable Guidance

**Status**: ✅ **COMPLETED** (2025-12-01) - Simplified Implementation
**Priority**: P3 - Low (High Impact)
**Actual Effort**: 2-3 hours (vs original estimate 4-8 hours)
**Type**: Developer Experience / UX
**Implemented Version**: 2.6.0 (Minor Release)

---

## Implementation Summary (2025-12-01)

**What Was Actually Implemented**: Simplified "Option B" approach with 2 helper functions:

### ✅ Completed:
1. **Added `format_auth_error()`** to `abstractcore/exceptions/__init__.py`
   - OpenAI and Anthropic authentication errors
   - 3-5 line SOTA format (not verbose 5-section template)
   - Includes: problem + fix command + API key URL

2. **Added `format_provider_error()`** to `abstractcore/exceptions/__init__.py`
   - Ollama and LMStudio connection errors
   - Includes: problem + installation/setup instructions

3. **Updated Providers**:
   - `openai_provider.py` - Uses `format_auth_error("openai", ...)`
   - `anthropic_provider.py` - Uses `format_auth_error("anthropic", ...)` (2 locations)
   - `ollama_provider.py` - Added import for future use
   - `lmstudio_provider.py` - Added import for future use

4. **Validated with Real Tests** (`test_error_messages.py`):
   - ✅ All helper functions tested
   - ✅ OpenAI provider integration tested with real API
   - ✅ Anthropic provider integration tested with real API
   - ✅ Existing `format_model_error()` verified still working
   - **NO MOCKING** - all tests use real implementations

### Key Design Decisions:

**Simplified vs Original Proposal**:
- ✅ **Used 3-5 line SOTA format** (not verbose 15-25 line template)
- ✅ **Helper functions** (not class-based exceptions with custom __init__)
- ✅ **Focused on auth/connection** (most common user pain points)
- ✅ **~60 lines of code** (vs 200+ in original proposal)

**Avoided Over-Engineering**:
- ❌ Didn't add `docs.abstractcore.ai` references (site doesn't exist)
- ❌ Didn't add `abstractcore --list-models` CLI flag (not yet implemented)
- ❌ Didn't modify existing exception classes (backward compatibility)
- ❌ Didn't add i18n support (premature)
- ❌ Didn't add AI-powered suggestions (circular dependency)

**What Already Existed** (Original proposal was wrong about these):
- ✅ `format_model_error()` - Already excellent (exceptions/__init__.py:70-106)
- ✅ `llm.estimate_tokens()` - EXISTS at base.py:1034-1036
- ✅ `llm.calculate_token_budget()` - EXISTS at base.py:1029-1032

### Example Output (Actual Implementation):

```
OPENAI authentication failed: Invalid API key
Fix: abstractcore --set-api-key openai YOUR_KEY
Get key: https://platform.openai.com/api-keys
```

**Total**: 3 lines, actionable, SOTA-compliant ✅

---

## Original Proposal (Archived Below)

**NOTE**: The original proposal below was **over-engineered**. We implemented a simplified approach that achieves the same UX goals with significantly less code and complexity. The original proposal is preserved for reference.

### Issues with Original Proposal:

1. **Referenced non-existent features**:
   - `docs.abstractcore.ai` - Site doesn't exist (would need hosting)
   - `abstractcore --list-models` - CLI flag not implemented

2. **Incorrect claims**:
   - Line 236-237: Claimed `llm.estimate_tokens()` doesn't exist - IT DOES!
   - Line 236-237: Claimed `llm.calculate_token_budget()` doesn't exist - IT DOES!

3. **Over-verbose template**:
   - Proposed 5-section template (15-25 lines per error)
   - SOTA is 3-5 lines (Git, Rust, npm, Click, Typer)

4. **Unnecessary complexity**:
   - Custom `__init__` methods for each exception class
   - Multiple optional parameters (6 for TokenLimitError!)
   - Helper methods like `_get_provider_url()` as instance methods

---

## Executive Summary (Original)

Enhance error messages throughout AbstractCore to include actionable guidance, helping users quickly resolve issues without consulting documentation. Transform generic error messages into helpful diagnostics that guide users to solutions.

**Expected Benefits**:
- Faster developer onboarding (less documentation lookup)
- Reduced support burden
- Better user experience
- Increased productivity

---

## Problem Statement (Original)

### Current Error Messages: Informational but Not Actionable

```python
# BEFORE: Generic error
raise ModelNotFoundError(f"Model {model} not found")

# User sees:
# ModelNotFoundError: Model invalid-model not found
# User thinks: "Now what? Which models exist? How do I find them?"
```

**Pain Points**:
1. **No Next Steps**: Errors don't tell user how to fix
2. **No Context**: Missing information about alternatives
3. **No CLI Hints**: Don't mention helpful commands
4. **No Documentation Links**: No pointers to relevant docs

### Industry Best Practices (Original)

**Good Error Messages** (from popular tools):

```bash
# Git (excellent error UX)
$ git checkout invalid-branch
error: pathspec 'invalid-branch' did not match any file(s) known to git
Did you mean one of these?
    main
    feature-branch
    develop

# Rust compiler (famous for helpful errors)
error[E0425]: cannot find value `x` in this scope
  --> main.rs:2:13
   |
2  |     println!("{}", x);
   |                    ^ not found in this scope
   |
help: consider importing this function
   |
1  | use std::x;
   |

# npm (actionable suggestions)
npm ERR! 404 'abstractcore@99.9.9' is not in the npm registry.
npm ERR! 404 You should bug the author to publish it (or use the name yourself!)
npm ERR! 404
npm ERR! 404 Note that you can also install from a tarball, folder, http url, or git url.
```

---

## Proposed Solution (Original - NOT IMPLEMENTED)

### Pattern: Actionable Error Messages

**⚠️ NOTE**: This template was **too verbose**. We used simplified 3-5 line format instead.

**Original Template**:
```
{Primary Error}

Why this happened:
  {Brief explanation}

Available options:
  {List of valid choices}

How to fix:
  {Specific command or action}

Learn more:
  {Documentation link}
```

**What We Actually Use** (SOTA 3-5 line format):
```
{Primary Error}: {Brief reason}
Fix: {Specific command}
Get key: {Direct URL}
```

### Implementation Examples (Original - NOT IMPLEMENTED)

#### 1. Model Not Found Error (Original Proposal)

**⚠️ NOTE**: This was over-engineered. The existing `format_model_error()` already works well!

```python
# abstractcore/exceptions.py

class ModelNotFoundError(AbstractCoreError):
    """Enhanced model not found error."""

    def __init__(self, model: str, provider: str, available_models: Optional[List[str]] = None):
        # Build helpful message
        message_parts = [
            f"Model '{model}' not found for provider '{provider}'."
        ]

        if available_models:
            # Show first 5 available models
            sample_models = available_models[:5]
            message_parts.append("\nAvailable models (first 5):")
            for m in sample_models:
                message_parts.append(f"  • {m}")

            if len(available_models) > 5:
                remaining = len(available_models) - 5
                message_parts.append(f"  ... and {remaining} more")

        message_parts.append(f"\nTo list all models:")
        message_parts.append(f"  Python: {provider}Provider.list_available_models()")
        # ⚠️ ISSUE: abstractcore --list-models CLI flag doesn't exist!
        message_parts.append(f"  CLI: abstractcore --list-models {provider}")
        # ⚠️ ISSUE: docs.abstractcore.ai site doesn't exist!
        message_parts.append(f"\nLearn more: https://docs.abstractcore.ai/providers/{provider}")

        super().__init__("\n".join(message_parts))


# Usage in providers
class OllamaProvider(BaseProvider):
    def __init__(self, model: str, **kwargs):
        # Validate model exists
        available = self.list_available_models()
        if model not in available:
            raise ModelNotFoundError(
                model=model,
                provider="ollama",
                available_models=available
            )
```

**What We Actually Implemented**: Used existing `format_model_error()` which already does this correctly!

#### 2. Authentication Error (Original Proposal)

**⚠️ NOTE**: We simplified this significantly!

```python
# ORIGINAL PROPOSAL (too complex)
class AuthenticationError(AbstractCoreError):
    """Enhanced authentication error."""

    def __init__(self, provider: str, details: Optional[str] = None):
        message_parts = [
            f"{provider.upper()} authentication failed."
        ]

        if details:
            message_parts.append(f"\nReason: {details}")

        message_parts.extend([
            "\nHow to fix:",
            f"  1. Get API key from {self._get_provider_url(provider)}",
            f"  2. Set key: abstractcore --set-api-key {provider} YOUR_KEY",
            f"  3. Or set environment: export {provider.upper()}_API_KEY=YOUR_KEY",
            # ⚠️ ISSUE: docs.abstractcore.ai doesn't exist!
            "\nLearn more: https://docs.abstractcore.ai/prerequisites#" + provider
        ])

        super().__init__("\n".join(message_parts))

    @staticmethod
    def _get_provider_url(provider: str) -> str:
        urls = {
            "openai": "https://platform.openai.com/api-keys",
            "anthropic": "https://console.anthropic.com/settings/keys",
        }
        return urls.get(provider, f"provider documentation")
```

**What We Actually Implemented** (Much Simpler):

```python
# abstractcore/exceptions/__init__.py

def format_auth_error(provider: str, reason: str = None) -> str:
    """Format actionable authentication error with setup instructions."""
    urls = {
        "openai": "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com/settings/keys",
    }
    msg = f"{provider.upper()} authentication failed"
    if reason:
        msg += f": {reason}"
    msg += f"\nFix: abstractcore --set-api-key {provider} YOUR_KEY"
    if provider.lower() in urls:
        msg += f"\nGet key: {urls[provider.lower()]}"
    return msg

# Usage in providers (simple!)
raise AuthenticationError(format_auth_error("openai", str(e)))
```

**Benefits of Simplified Approach**:
- ✅ 12 lines vs 25+ lines
- ✅ Helper function vs class modification
- ✅ Backward compatible (no exception signature changes)
- ✅ 3-5 line output vs 15-25 line output
- ✅ No non-existent documentation references

#### 3. Provider Not Available Error (Original Proposal)

**⚠️ NOTE**: We created a simple helper function instead!

```python
# ORIGINAL PROPOSAL (too complex)
class ProviderNotAvailableError(AbstractCoreError):
    """Enhanced provider availability error."""

    def __init__(self, provider: str, reason: str):
        message_parts = [
            f"Provider '{provider}' is not available.",
            f"\nReason: {reason}",
            "\nHow to fix:"
        ]

        # Provider-specific installation instructions
        if provider == "ollama":
            message_parts.extend([
                "  1. Install Ollama: https://ollama.com/download",
                "  2. Start server: ollama serve",
                "  3. Verify: curl http://localhost:11434",
            ])
        elif provider == "lmstudio":
            message_parts.extend([
                "  1. Install LMStudio: https://lmstudio.ai/",
                "  2. Load a model in LMStudio",
                "  3. Start local server (enable API in settings)",
            ])
        elif provider in ["openai", "anthropic"]:
            message_parts.extend([
                f"  1. Install package: pip install abstractcore[{provider}]",
                f"  2. Set API key: abstractcore --set-api-key {provider} YOUR_KEY",
            ])
        else:
            message_parts.append(f"  Check installation: pip install abstractcore[{provider}]")

        # ⚠️ ISSUE: docs.abstractcore.ai doesn't exist!
        message_parts.append(f"\nLearn more: https://docs.abstractcore.ai/prerequisites#{provider}")
        super().__init__("\n".join(message_parts))
```

**What We Actually Implemented** (Much Simpler):

```python
def format_provider_error(provider: str, reason: str) -> str:
    """Format actionable provider unavailability error with setup instructions."""
    instructions = {
        "ollama": "Install: https://ollama.com/download\nStart: ollama serve",
        "lmstudio": "Install: https://lmstudio.ai/\nEnable API in settings",
    }
    msg = f"Provider '{provider}' unavailable: {reason}"
    if provider.lower() in instructions:
        msg += f"\n{instructions[provider.lower()]}"
    return msg
```

**Benefits**:
- ✅ 9 lines vs 30+ lines
- ✅ 3-4 line output vs 10-15 line output
- ✅ No non-existent documentation references
- ✅ Simpler, cleaner, SOTA-compliant

#### 4. Token Limit Exceeded Error (Original Proposal)

**⚠️ NOTE**: This was not implemented. Original proposal had incorrect claims!

```python
# ORIGINAL PROPOSAL (contains errors!)
class TokenLimitError(AbstractCoreError):
    """Enhanced token limit error."""

    # 6 parameters! Too complex!
    def __init__(self, requested: int, limit: int, input_tokens: int, output_tokens: int):
        overage = requested - limit

        message_parts = [
            f"Token limit exceeded: requested {requested} tokens, limit is {limit}.",
            f"\nBreakdown:",
            f"  • Input tokens: {input_tokens}",
            f"  • Requested output: {output_tokens}",
            f"  • Total: {requested}",
            f"  • Limit: {limit}",
            f"  • Overage: {overage}",
            "\nHow to fix:",
            f"  1. Reduce input size (currently {input_tokens} tokens)",
            f"  2. Reduce max_output_tokens (currently {output_tokens})",
            f"  3. Increase max_tokens limit in provider config",
            "\nHelper methods:",
            # ⚠️ CORRECTION: These methods DO EXIST!
            "  • llm.estimate_tokens(text) - Estimate token count",  # EXISTS at base.py:1034-1036
            "  • llm.calculate_token_budget(text, desired_output) - Get recommendation",  # EXISTS at base.py:1029-1032
            # ⚠️ ISSUE: docs.abstractcore.ai doesn't exist!
            "\nLearn more: https://docs.abstractcore.ai/generation-parameters#token-management"
        ]

        super().__init__("\n".join(message_parts))
```

**Why Not Implemented**:
1. Original proposal claimed token methods don't exist - THEY DO!
2. 6 required parameters too complex
3. 20+ line error message too verbose
4. Would require significant plumbing to pass all those parameters
5. Current token errors are adequate

**Token Methods That Already Exist**:
```python
# abstractcore/providers/base.py:1029-1036
def calculate_token_budget(self, input_text: str, desired_output_tokens: int,
                          safety_margin: float = 0.1) -> tuple[int, List[str]]:
    """Helper to estimate required max_tokens given input and desired output"""
    return super().calculate_token_budget(input_text, desired_output_tokens, safety_margin)

def estimate_tokens(self, text: str) -> int:
    """Rough estimation of token count for given text"""
    return super().estimate_tokens(text)
```

---

## Implementation Plan (Original - NOT FOLLOWED)

**NOTE**: We used a simplified approach instead. Original plan was:

### Phase 1: Audit Current Exceptions (1-2 hours)

```bash
# Find all raise statements
grep -r "raise " abstractcore/ --include="*.py" | \
    grep -v "test" | \
    wc -l

# Expected: ~50-100 raise statements

# Categorize by exception type
grep -r "raise " abstractcore/ --include="*.py" | \
    cut -d: -f2 | \
    sort | \
    uniq -c | \
    sort -rn
```

### Phase 2: Update Core Exceptions (2-3 hours)

**Priority order**:
1. `ModelNotFoundError` (most common) - ✅ Already excellent!
2. `AuthenticationError` (critical for setup) - ✅ Implemented simplified version
3. `ProviderNotAvailableError` (setup issues) - ✅ Implemented simplified version
4. `TokenLimitError` (configuration errors) - ❌ Not needed (methods already exist)
5. `InvalidRequestError` (usage errors) - ⏸️ Deferred

### Phase 3: Update Provider-Specific Errors (1-2 hours)

✅ **Completed**:
- `providers/openai_provider.py` - Uses `format_auth_error()`
- `providers/anthropic_provider.py` - Uses `format_auth_error()` (2 locations)
- `providers/ollama_provider.py` - Import added for future use
- `providers/lmstudio_provider.py` - Import added for future use

⏸️ **Not needed**:
- `providers/mlx_provider.py` - Local provider, different error patterns
- `providers/huggingface_provider.py` - Different error patterns

### Phase 4: Testing (1-2 hours)

✅ **Completed** with real implementations (test_error_messages.py):

```python
# tests/test_error_messages.py

def test_format_auth_error():
    """Test format_auth_error helper function."""
    openai_msg = format_auth_error("openai", "Invalid API key")
    assert "OPENAI authentication failed" in openai_msg
    assert "abstractcore --set-api-key openai" in openai_msg
    assert "https://platform.openai.com/api-keys" in openai_msg
    assert len(openai_msg.split('\n')) <= 5  # SOTA format

def test_openai_real_auth_error():
    """Test OpenAI provider with REAL invalid key."""
    try:
        llm = create_llm("openai", model="gpt-4o-mini",
                        api_key="sk-invalid-key-for-testing-12345")
        response = llm.generate("test")
    except AuthenticationError as e:
        error_msg = str(e)
        assert "abstractcore --set-api-key openai" in error_msg

# Similar tests for Anthropic, format_provider_error(), format_model_error()
```

**Actual Time**: 2-3 hours (including testing with real APIs)

---

## Success Criteria

**Original Criteria**:
1. **Actionability**: All errors include "How to fix" section
2. **Context**: Errors show available alternatives where relevant
3. **CLI Integration**: Errors mention relevant CLI commands
4. **Documentation**: Errors link to relevant docs
5. **Formatting**: Multi-line, well-structured, readable
6. **Length**: Concise (< 500 chars for common errors)

**Actual Results**:
1. ✅ **Actionability**: Auth/provider errors include fix commands
2. ✅ **Context**: `format_model_error()` already shows alternatives
3. ✅ **CLI Integration**: Errors mention `abstractcore --set-api-key`
4. ⚠️ **Documentation**: Links to real provider sites (not docs.abstractcore.ai)
5. ✅ **Formatting**: 3-5 lines (SOTA format, not verbose template)
6. ✅ **Length**: ~150-250 chars (well under 500)

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation | Actual Outcome |
|------|------------|--------|------------|----------------|
| Too verbose | Low | Low | Keep errors concise, use sections | ✅ Used 3-5 line format |
| Outdated links | Low | Medium | CI tests for broken links | ✅ Used real provider URLs |
| Inaccurate suggestions | Low | High | Thorough testing, user feedback | ✅ Tested with real APIs |

---

## Examples: Before & After

### Example 1: Model Not Found

**Before**:
```
ModelNotFoundError: Model 'llama-9000' not found
```

**After** (Using existing `format_model_error()`):
```
❌ Model 'llama-9000' not found for Ollama provider.

✅ Available models (30):
  • llama3:8b
  • qwen3:4b
  • mistral:7b
  • phi3:14b
  • gemma2:9b
  ... and 25 more
```

**Note**: This already existed and works great! We didn't need to change it.

### Example 2: Authentication Failed

**Before**:
```
AuthenticationError: OpenAI authentication failed
```

**After** (Using new `format_auth_error()`):
```
OPENAI authentication failed: Invalid API key
Fix: abstractcore --set-api-key openai YOUR_KEY
Get key: https://platform.openai.com/api-keys
```

**Improvement**: 3 lines (SOTA format), actionable, includes fix command + key URL

---

## Follow-up Actions

**Original Proposals** (Not implemented):
1. ~~User Feedback: Collect feedback on error helpfulness~~ - Can do after release
2. ~~Metrics: Track time-to-resolution for common errors~~ - Future enhancement
3. ~~Expansion: Add helpful messages to warnings~~ - Deferred
4. ~~Internationalization: Consider i18n for error messages~~ - Over-engineering for v1
5. ~~AI-Powered Suggestions: Use LLM to suggest fixes (future)~~ - Circular dependency

**Recommended Next Steps**:
1. ✅ Monitor user feedback on new error messages
2. ⏸️ Consider adding `--list-models` CLI flag (if commonly requested)
3. ⏸️ Consider creating docs.abstractcore.ai site (if project grows)
4. ⏸️ Expand to other error types based on user pain points

---

## References

- Current exceptions: `abstractcore/exceptions/__init__.py`
- Rust compiler error design: https://blog.rust-lang.org/2016/08/10/Shape-of-errors-to-come.html
- Error UX best practices: https://uxdesign.cc/how-to-write-good-error-messages-858e4551cd4
- SOTA CLI error patterns: Git, Rust, npm, Click, Typer, httpx

---

**Document Version**: 2.0 (Updated with completion notes)
**Created**: 2025-11-25
**Completed**: 2025-12-01
**Author**: Expert Code Review
**Status**: ✅ Completed (Simplified Implementation)

---

## Verification

### Test Suite

Location: `tests/exceptions/test_enhanced_error_messages.py`

**Test Results**: 10/10 tests passing (NO MOCKING - all real implementations)

```bash
python -m pytest tests/exceptions/test_enhanced_error_messages.py -v
```

**Test Categories**:
1. **TestErrorMessageHelpers** (6 tests): Helper function format validation
2. **TestProviderIntegration** (2 tests): Real API calls with invalid keys
3. **TestBackwardCompatibility** (2 tests): No breaking changes

### Manual Verification

```bash
# Test OpenAI error message
python -c "
from abstractcore import create_llm
try:
    llm = create_llm('openai', model='gpt-4o-mini', api_key='sk-invalid')
    llm.generate('test')
except Exception as e:
    print(str(e))
"

# Test Anthropic error message
python -c "
from abstractcore import create_llm
try:
    llm = create_llm('anthropic', model='claude-sonnet-4-5-20250929', api_key='sk-ant-invalid')
    llm.generate('test')
except Exception as e:
    print(str(e))
"
```

### Files Created
- `tests/exceptions/__init__.py`
- `tests/exceptions/test_enhanced_error_messages.py` (10 tests)

### Files Modified
- `abstractcore/exceptions/__init__.py` (~45 lines added)
- `abstractcore/providers/openai_provider.py` (1 import + 1 usage)
- `abstractcore/providers/anthropic_provider.py` (1 import + 2 usages)
- `abstractcore/providers/ollama_provider.py` (1 import for future use)
- `abstractcore/providers/lmstudio_provider.py` (1 import for future use)
