"""
Custom exceptions for AbstractCore.
"""


class AbstractCoreError(Exception):
    """Base exception for AbstractCore"""
    pass


class ProviderError(AbstractCoreError):
    """Base exception for provider-related errors"""
    pass


class ProviderAPIError(ProviderError):
    """API call to provider failed"""
    pass


class AuthenticationError(ProviderError):
    """Authentication with provider failed"""
    pass


# Alias for backward compatibility with old AbstractCore
Authentication = AuthenticationError


class RateLimitError(ProviderError):
    """Rate limit exceeded"""
    pass


class InvalidRequestError(ProviderError):
    """Invalid request to provider"""
    pass


class UnsupportedFeatureError(AbstractCoreError):
    """Feature not supported by provider"""
    pass


class FileProcessingError(AbstractCoreError):
    """Error processing file or media"""
    pass


class ToolExecutionError(AbstractCoreError):
    """Error executing tool"""
    pass


class SessionError(AbstractCoreError):
    """Error with session management"""
    pass


class ConfigurationError(AbstractCoreError):
    """Invalid configuration"""
    pass


class ModelNotFoundError(ProviderError):
    """Model not found or invalid model name"""
    pass


def format_model_error(provider: str, invalid_model: str, available_models: list) -> str:
    """
    Format a helpful error message for model not found errors.

    Args:
        provider: Provider name (e.g., "OpenAI", "Anthropic")
        invalid_model: The model name that was not found
        available_models: List of available model names

    Returns:
        Formatted error message string
    """
    message = f"❌ Model '{invalid_model}' not found for {provider} provider.\n"

    if available_models:
        message += f"\n✅ Available models ({len(available_models)}):\n"
        for model in available_models[:30]:  # Show max 30
            message += f"  • {model}\n"
        if len(available_models) > 30:
            message += f"  ... and {len(available_models) - 30} more\n"
    else:
        # Show provider documentation when we can't fetch models
        doc_links = {
            "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models",
            "openai": "https://platform.openai.com/docs/models",
            "ollama": "https://ollama.com/library",
            "huggingface": "https://huggingface.co/models",
            "mlx": "https://huggingface.co/mlx-community"
        }

        provider_lower = provider.lower()
        if provider_lower in doc_links:
            message += f"\n📚 See available models: {doc_links[provider_lower]}\n"
        else:
            message += f"\n⚠️  Could not fetch available models for {provider}.\n"

    return message.rstrip()


def format_auth_error(provider: str, reason: str = None) -> str:
    """
    Format actionable authentication error with setup instructions.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        reason: Optional reason for auth failure

    Returns:
        Formatted error message with fix instructions
    """
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


def format_provider_error(provider: str, reason: str) -> str:
    """
    Format actionable provider unavailability error with setup instructions.

    Args:
        provider: Provider name (e.g., "ollama", "lmstudio")
        reason: Reason for unavailability (e.g., "Connection refused")

    Returns:
        Formatted error message with setup instructions
    """
    instructions = {
        "ollama": "Install: https://ollama.com/download\nStart: ollama serve",
        "lmstudio": "Install: https://lmstudio.ai/\nEnable API in settings",
    }
    msg = f"Provider '{provider}' unavailable: {reason}"
    if provider.lower() in instructions:
        msg += f"\n{instructions[provider.lower()]}"
    return msg


# Export all exceptions for easy importing
__all__ = [
    'AbstractCoreError',
    'ProviderError',
    'ProviderAPIError',
    'AuthenticationError',
    'Authentication',  # Backward compatibility alias
    'RateLimitError',
    'InvalidRequestError',
    'UnsupportedFeatureError',
    'FileProcessingError',
    'ToolExecutionError',
    'SessionError',
    'ConfigurationError',
    'ModelNotFoundError',
    'format_model_error',
    'format_auth_error',
    'format_provider_error'
]