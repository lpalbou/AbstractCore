# Exceptions Module

## Purpose

This module defines the complete exception hierarchy for AbstractCore, providing a structured error taxonomy that enables precise error handling throughout the framework. All exceptions inherit from a common base class, making it easy to catch AbstractCore-specific errors.

## Architecture Position

- **Layer**: Foundation (Layer 0)
- **Dependencies**: None (pure Python exceptions)
- **Used By**: All modules in AbstractCore (providers, tools, media, config, etc.)

## Exception Hierarchy

```
AbstractCoreError (base)
├── ProviderError
│   ├── ProviderAPIError, AuthenticationError, RateLimitError
│   ├── InvalidRequestError, ModelNotFoundError
├── UnsupportedFeatureError, FileProcessingError
├── ToolExecutionError, SessionError, ConfigurationError
```

## Key Exceptions

| Exception | When | Handling |
|-----------|------|----------|
| **AbstractCoreError** | Base for all AC errors | Catch for any AC error |
| **ProviderError** | Base for provider errors | Catch for provider fallback |
| **ProviderAPIError** | API failures (network, server, timeout) | Retry logic |
| **AuthenticationError** | Invalid/missing credentials | Check API key |
| **RateLimitError** | Rate limit exceeded | Exponential backoff |
| **InvalidRequestError** | Malformed requests | Validate params |
| **ModelNotFoundError** | Invalid model name | Use `format_model_error()` |
| **UnsupportedFeatureError** | Feature not available | Graceful fallback |
| **FileProcessingError** | File/media errors | Check format/dependencies |
| **ToolExecutionError** | Tool call failures | Return error to LLM |
| **SessionError** | Session management issues | Check initialization |
| **ConfigurationError** | Invalid/missing config | Validate settings |

## Helper: format_model_error()

**Purpose**: Creates user-friendly error messages for `ModelNotFoundError`
**Returns**: Formatted message with problem, available models (up to 30), docs links
**Usage**: `error_msg = format_model_error("OpenAI", "gpt-5", available_models)`

## Usage Patterns

| Pattern | Code | Use Case |
|---------|------|----------|
| **Basic Handling** | `except AuthenticationError / ModelNotFoundError / ProviderError` | Specific error types |
| **Catch All AC** | `except AbstractCoreError` | Any AbstractCore error |
| **Retry Logic** | `except RateLimitError: time.sleep(2**attempt)` | Exponential backoff |
| **Feature Fallback** | `except UnsupportedFeatureError: use_fallback()` | Graceful degradation |

## Integration Points

| Module | Raises |
|--------|--------|
| **Providers** | AuthenticationError, RateLimitError, ModelNotFoundError, ProviderAPIError, UnsupportedFeatureError |
| **Media** | FileProcessingError, UnsupportedFeatureError |
| **Tools** | ToolExecutionError, InvalidRequestError |
| **Configuration** | ConfigurationError, AuthenticationError |

## Best Practices

### DO
1. Use specific exceptions for known error types
2. Catch specific exceptions for targeted handling
3. Provide helpful error messages with context
4. Use `format_model_error()` for model not found
5. Let AbstractCore exceptions propagate

### DON'T
1. Catch AbstractCoreError silently (`pass`)
2. Raise generic `Exception` for AC errors
3. Catch overly broad exceptions (`except Exception`)
4. Create new exception types outside this module

## Common Pitfalls

| Pitfall | Wrong | Right |
|---------|-------|-------|
| **No retry logic** | No handling of RateLimitError | Exponential backoff (2**attempt) |
| **Ignore feature support** | Assume all features available | Try/catch UnsupportedFeatureError |
| **Poor error messages** | "Error occurred" | Specific, actionable messages |

## Testing Strategy

**Exception Handling**: Test ModelNotFoundError with formatted message, verify exception hierarchy (isinstance checks)
**Message Formatting**: Test `format_model_error()` with/without available models (check docs link fallback)

## Public API

**Exported**: All exceptions (AbstractCoreError, ProviderError, subtypes), helper `format_model_error()`
**Backward Compat**: `Authentication` alias for `AuthenticationError`, consistent hierarchy + message formats

## Summary

The exceptions module provides a clean, hierarchical error taxonomy that makes error handling predictable and precise. By using specific exception types and the `format_model_error()` helper, AbstractCore enables robust error handling with user-friendly error messages.

**Key Takeaways**:
- All AbstractCore exceptions inherit from `AbstractCoreError`
- Provider errors have a dedicated subtree under `ProviderError`
- Use `format_model_error()` for helpful model not found messages
- Catch specific exceptions for targeted error handling
- Let exceptions propagate rather than silencing them

## Related Modules

**Used by (imports exceptions)**:
- [`core/`](../core/README.md) - Factory pattern and base abstractions raise exceptions
- [`providers/`](../providers/README.md) - All provider implementations for error handling
- [`media/`](../media/README.md) - Media processing error hierarchy
- [`compression/`](../compression/README.md) - Compression-specific exceptions
- [`structured/`](../structured/README.md) - Validation error handling
- [`config/`](../config/README.md) - Configuration validation errors
- [`tools/`](../tools/README.md) - Tool execution error handling
- [`server/`](../server/README.md) - API error responses

**Related infrastructure**:
- [`events/`](../events/README.md) - Error events emitted when exceptions occur
- [`utils/`](../utils/README.md) - Structured logging for exception context
