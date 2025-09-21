"""
Custom exceptions for AbstractLLM.
"""


class AbstractLLMError(Exception):
    """Base exception for AbstractLLM"""
    pass


class ProviderError(AbstractLLMError):
    """Base exception for provider-related errors"""
    pass


class ProviderAPIError(ProviderError):
    """API call to provider failed"""
    pass


class AuthenticationError(ProviderError):
    """Authentication with provider failed"""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded"""
    pass


class InvalidRequestError(ProviderError):
    """Invalid request to provider"""
    pass


class UnsupportedFeatureError(AbstractLLMError):
    """Feature not supported by provider"""
    pass


class FileProcessingError(AbstractLLMError):
    """Error processing file or media"""
    pass


class ToolExecutionError(AbstractLLMError):
    """Error executing tool"""
    pass


class SessionError(AbstractLLMError):
    """Error with session management"""
    pass


class ConfigurationError(AbstractLLMError):
    """Invalid configuration"""
    pass


class ModelNotFoundError(ProviderError):
    """Model not found or invalid model name"""
    pass