# TACTICAL-003: Enhanced Error Messages with Actionable Guidance

**Status**: Proposed
**Priority**: P3 - Low (High Impact)
**Effort**: Small (4-8 hours)
**Type**: Developer Experience / UX
**Target Version**: 2.6.0 (Minor Release)

## Executive Summary

Enhance error messages throughout AbstractCore to include actionable guidance, helping users quickly resolve issues without consulting documentation. Transform generic error messages into helpful diagnostics that guide users to solutions.

**Expected Benefits**:
- Faster developer onboarding (less documentation lookup)
- Reduced support burden
- Better user experience
- Increased productivity

---

## Problem Statement

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

### Industry Best Practices

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

## Proposed Solution

### Pattern: Actionable Error Messages

**Template**:
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

### Implementation Examples

#### 1. Model Not Found Error

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
        message_parts.append(f"  CLI: abstractcore --list-models {provider}")
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

#### 2. Authentication Error

```python
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

#### 3. Provider Not Available Error

```python
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

        message_parts.append(f"\nLearn more: https://docs.abstractcore.ai/prerequisites#{provider}")
        super().__init__("\n".join(message_parts))
```

#### 4. Token Limit Exceeded Error

```python
class TokenLimitError(AbstractCoreError):
    """Enhanced token limit error."""

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
            "  • llm.estimate_tokens(text) - Estimate token count",
            "  • llm.calculate_token_budget(text, desired_output) - Get recommendation",
            "\nLearn more: https://docs.abstractcore.ai/generation-parameters#token-management"
        ]

        super().__init__("\n".join(message_parts))
```

---

## Implementation Plan

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
1. `ModelNotFoundError` (most common)
2. `AuthenticationError` (critical for setup)
3. `ProviderNotAvailableError` (setup issues)
4. `TokenLimitError` (configuration errors)
5. `InvalidRequestError` (usage errors)

### Phase 3: Update Provider-Specific Errors (1-2 hours)

Update error messages in:
- `providers/openai_provider.py`
- `providers/anthropic_provider.py`
- `providers/ollama_provider.py`
- `providers/lmstudio_provider.py`
- `providers/mlx_provider.py`
- `providers/huggingface_provider.py`

### Phase 4: Testing (1-2 hours)

```python
# tests/errors/test_error_messages.py

def test_model_not_found_error_helpful():
    """Verify ModelNotFoundError provides helpful guidance."""
    try:
        llm = create_llm("ollama", model="invalid-model-xyz")
    except ModelNotFoundError as e:
        error_msg = str(e)

        # Should include these elements
        assert "invalid-model-xyz" in error_msg
        assert "Available models" in error_msg
        assert "abstractcore --list-models" in error_msg
        assert "https://docs.abstractcore.ai" in error_msg

def test_authentication_error_helpful():
    """Verify AuthenticationError provides setup instructions."""
    # ... similar pattern ...

# Visual regression test - capture error output
def test_error_message_format():
    """Verify error messages are well-formatted."""
    try:
        llm = create_llm("openai", model="invalid", api_key="invalid")
    except Exception as e:
        error_msg = str(e)

        # Should be multi-line
        assert "\n" in error_msg

        # Should have clear sections
        assert any(marker in error_msg for marker in ["How to fix:", "Learn more:", "Available"])

        # Should not be too long (< 500 chars for most errors)
        assert len(error_msg) < 1000
```

**Total Estimated Time**: 5-9 hours

---

## Success Criteria

1. **Actionability**: All errors include "How to fix" section
2. **Context**: Errors show available alternatives where relevant
3. **CLI Integration**: Errors mention relevant CLI commands
4. **Documentation**: Errors link to relevant docs
5. **Formatting**: Multi-line, well-structured, readable
6. **Length**: Concise (< 500 chars for common errors)

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Too verbose | Low | Low | Keep errors concise, use sections |
| Outdated links | Low | Medium | CI tests for broken links |
| Inaccurate suggestions | Low | High | Thorough testing, user feedback |

---

## Examples: Before & After

### Example 1: Model Not Found

**Before**:
```
ModelNotFoundError: Model 'llama-9000' not found
```

**After**:
```
Model 'llama-9000' not found for provider 'ollama'.

Available models (first 5):
  • llama3:8b
  • qwen3:4b
  • mistral:7b
  • phi3:14b
  • gemma2:9b
  ... and 45 more

To list all models:
  Python: OllamaProvider.list_available_models()
  CLI: abstractcore --list-models ollama

Learn more: https://docs.abstractcore.ai/providers/ollama
```

### Example 2: Authentication Failed

**Before**:
```
AuthenticationError: OpenAI authentication failed
```

**After**:
```
OPENAI authentication failed.

Reason: Invalid API key

How to fix:
  1. Get API key from https://platform.openai.com/api-keys
  2. Set key: abstractcore --set-api-key openai YOUR_KEY
  3. Or set environment: export OPENAI_API_KEY=YOUR_KEY

Learn more: https://docs.abstractcore.ai/prerequisites#openai
```

---

## Follow-up Actions

1. **User Feedback**: Collect feedback on error helpfulness
2. **Metrics**: Track time-to-resolution for common errors
3. **Expansion**: Add helpful messages to warnings
4. **Internationalization**: Consider i18n for error messages
5. **AI-Powered Suggestions**: Use LLM to suggest fixes (future)

---

## References

- Current exceptions: `abstractcore/exceptions.py`
- Rust compiler error design: https://blog.rust-lang.org/2016/08/10/Shape-of-errors-to-come.html
- Error UX best practices: https://uxdesign.cc/how-to-write-good-error-messages-858e4551cd4

---

**Document Version**: 1.0
**Created**: 2025-11-25
**Author**: Expert Code Review
**Status**: Ready for Implementation
