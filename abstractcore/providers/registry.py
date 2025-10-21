"""
Provider Registry - Centralized provider discovery and metadata management.

This module provides a single source of truth for all AbstractCore providers,
eliminating the need for manual synchronization across factory.py, server/app.py,
and __init__.py files.
"""

from typing import List, Dict, Any, Optional, Type, Callable
from dataclasses import dataclass, field
from abc import ABC
import logging
from ..utils.structured_logging import get_logger

logger = get_logger("provider_registry")


@dataclass
class ProviderInfo:
    """Information about a registered provider."""
    name: str
    display_name: str
    provider_class: Type
    description: str
    provider_type: str = "llm"
    default_model: Optional[str] = None
    supported_features: List[str] = field(default_factory=list)
    authentication_required: bool = True
    local_provider: bool = False
    installation_extras: Optional[str] = None
    import_path: str = ""

    def __post_init__(self):
        """Set default values after initialization."""
        if not self.import_path:
            self.import_path = f"..providers.{self.name}_provider"


class ProviderRegistry:
    """
    Centralized registry for all AbstractCore providers.

    This registry serves as the single source of truth for provider discovery,
    metadata, and instantiation across the entire AbstractCore system.
    """

    def __init__(self):
        self._providers: Dict[str, ProviderInfo] = {}
        self._logger = get_logger("ProviderRegistry")
        self._register_all_providers()

    def _register_all_providers(self):
        """Register all available providers with their metadata."""

        # OpenAI Provider
        self.register_provider(ProviderInfo(
            name="openai",
            display_name="OpenAI",
            provider_class=None,  # Will be set during lazy loading
            description="Commercial API with GPT-4, GPT-3.5, and embedding models",
            default_model="gpt-5-nano-2025-08-07",
            supported_features=["chat", "completion", "embeddings", "native_tools", "streaming", "structured_output"],
            authentication_required=True,
            local_provider=False,
            installation_extras="openai",
            import_path="..providers.openai_provider"
        ))

        # Anthropic Provider
        self.register_provider(ProviderInfo(
            name="anthropic",
            display_name="Anthropic",
            provider_class=None,
            description="Commercial API with Claude 3 family models",
            default_model="claude-3-5-haiku-latest",
            supported_features=["chat", "completion", "native_tools", "streaming", "structured_output"],
            authentication_required=True,
            local_provider=False,
            installation_extras="anthropic",
            import_path="..providers.anthropic_provider"
        ))

        # Ollama Provider
        self.register_provider(ProviderInfo(
            name="ollama",
            display_name="Ollama",
            provider_class=None,
            description="Local LLM server for running open-source models",
            default_model="qwen3-coder:30b",
            supported_features=["chat", "completion", "embeddings", "prompted_tools", "streaming"],
            authentication_required=False,
            local_provider=True,
            installation_extras="ollama",
            import_path="..providers.ollama_provider"
        ))

        # LMStudio Provider
        self.register_provider(ProviderInfo(
            name="lmstudio",
            display_name="LMStudio",
            provider_class=None,
            description="Local model development and testing platform",
            default_model="qwen/qwen3-4b-2507",
            supported_features=["chat", "completion", "embeddings", "prompted_tools", "streaming"],
            authentication_required=False,
            local_provider=True,
            installation_extras=None,
            import_path="..providers.lmstudio_provider"
        ))

        # MLX Provider
        self.register_provider(ProviderInfo(
            name="mlx",
            display_name="MLX",
            provider_class=None,
            description="Apple Silicon optimized local inference",
            default_model="mlx-community/Qwen3-4B",
            supported_features=["chat", "completion", "prompted_tools", "streaming", "apple_silicon"],
            authentication_required=False,
            local_provider=True,
            installation_extras="mlx",
            import_path="..providers.mlx_provider"
        ))

        # HuggingFace Provider
        self.register_provider(ProviderInfo(
            name="huggingface",
            display_name="HuggingFace",
            provider_class=None,
            description="Access to HuggingFace models (transformers and embeddings)",
            default_model="Qwen/Qwen3-4B/",
            supported_features=["chat", "completion", "embeddings", "prompted_tools", "local_models"],
            authentication_required=False,  # Optional for public models
            local_provider=True,
            installation_extras="huggingface",
            import_path="..providers.huggingface_provider"
        ))


    def register_provider(self, provider_info: ProviderInfo):
        """Register a provider in the registry."""
        self._providers[provider_info.name] = provider_info
        self._logger.debug(f"Registered provider: {provider_info.name}")

    def get_provider_info(self, provider_name: str) -> Optional[ProviderInfo]:
        """Get information about a specific provider."""
        return self._providers.get(provider_name.lower())

    def list_provider_names(self) -> List[str]:
        """Get list of all registered provider names."""
        return list(self._providers.keys())

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is registered."""
        return provider_name.lower() in self._providers

    def get_provider_class(self, provider_name: str):
        """Get the provider class, loading it lazily if needed."""
        provider_info = self.get_provider_info(provider_name)
        if not provider_info:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Lazy loading of provider class
        if provider_info.provider_class is None:
            provider_info.provider_class = self._load_provider_class(provider_info)

        return provider_info.provider_class

    def _load_provider_class(self, provider_info: ProviderInfo):
        """Dynamically load a provider class."""
        try:
            if provider_info.name == "openai":
                from ..providers.openai_provider import OpenAIProvider
                return OpenAIProvider
            elif provider_info.name == "anthropic":
                from ..providers.anthropic_provider import AnthropicProvider
                return AnthropicProvider
            elif provider_info.name == "ollama":
                from ..providers.ollama_provider import OllamaProvider
                return OllamaProvider
            elif provider_info.name == "lmstudio":
                from ..providers.lmstudio_provider import LMStudioProvider
                return LMStudioProvider
            elif provider_info.name == "mlx":
                from ..providers.mlx_provider import MLXProvider
                return MLXProvider
            elif provider_info.name == "huggingface":
                from ..providers.huggingface_provider import HuggingFaceProvider
                return HuggingFaceProvider
            else:
                raise ImportError(f"No import logic for provider: {provider_info.name}")
        except ImportError as e:
            self._logger.warning(f"Failed to load provider {provider_info.name}: {e}")
            raise ImportError(
                f"{provider_info.display_name} dependencies not installed. "
                f"Install with: pip install abstractcore[{provider_info.installation_extras}]"
            ) from e

    def get_available_models(self, provider_name: str, **kwargs) -> List[str]:
        """
        Get available models for a specific provider.

        Args:
            provider_name: Name of the provider
            **kwargs: Provider-specific parameters (e.g., api_key, base_url)

        Returns:
            List of available model names
        """
        try:
            provider_class = self.get_provider_class(provider_name)

            # Handle providers that need instance for model listing
            if provider_name in ["anthropic", "ollama", "lmstudio"]:
                provider_info = self.get_provider_info(provider_name)
                # Create minimal instance for API access
                instance = provider_class(model=provider_info.default_model, **kwargs)
                return instance.list_available_models(**kwargs)
            else:
                # Handle providers with static method or class method
                try:
                    # First try as static/class method
                    return provider_class.list_available_models(**kwargs)
                except TypeError:
                    # If that fails (method needs 'self'), create temporary instance
                    provider_info = self.get_provider_info(provider_name)
                    instance = provider_class(model=provider_info.default_model, **kwargs)
                    return instance.list_available_models(**kwargs)

        except Exception as e:
            self._logger.debug(f"Failed to get models from provider {provider_name}: {e}")
            return []

    def get_provider_status(self, provider_name: str) -> Dict[str, Any]:
        """
        Get detailed status information for a provider.

        Returns provider information including availability, model count, etc.
        This is used by the server /providers endpoint.
        """
        provider_info = self.get_provider_info(provider_name)
        if not provider_info:
            return {
                "name": provider_name,
                "status": "unknown",
                "error": "Provider not registered"
            }

        try:
            # Try to get models to test availability
            models = self.get_available_models(provider_name)

            return {
                "name": provider_info.name,
                "display_name": provider_info.display_name,
                "type": provider_info.provider_type,
                "model_count": len(models),
                "status": "available" if models else "no_models",
                "description": provider_info.description,
                "local_provider": provider_info.local_provider,
                "authentication_required": provider_info.authentication_required,
                "supported_features": provider_info.supported_features,
                "installation_extras": provider_info.installation_extras,
                "models": models
            }
        except Exception as e:
            return {
                "name": provider_info.name,
                "display_name": provider_info.display_name,
                "type": provider_info.provider_type,
                "model_count": 0,
                "status": "error",
                "description": provider_info.description,
                "error": str(e),
                "local_provider": provider_info.local_provider,
                "authentication_required": provider_info.authentication_required,
                "supported_features": provider_info.supported_features,
                "installation_extras": provider_info.installation_extras
            }

    def get_all_providers_status(self) -> List[Dict[str, Any]]:
        """Get status information for all registered providers."""
        return [
            self.get_provider_status(provider_name)
            for provider_name in self.list_provider_names()
        ]

    def get_providers_with_models(self) -> List[Dict[str, Any]]:
        """Get only providers that have available models."""
        all_providers = self.get_all_providers_status()
        return [
            provider for provider in all_providers
            if provider.get("status") == "available" and provider.get("model_count", 0) > 0
        ]

    def create_provider_instance(self, provider_name: str, model: Optional[str] = None, **kwargs):
        """
        Create a provider instance using the registry.

        This is used by the factory to create provider instances.
        """
        provider_info = self.get_provider_info(provider_name)
        if not provider_info:
            available_providers = ", ".join(self.list_provider_names())
            raise ValueError(f"Unknown provider: {provider_name}. Available providers: {available_providers}")

        provider_class = self.get_provider_class(provider_name)
        model = model or provider_info.default_model

        try:
            return provider_class(model=model, **kwargs)
        except ImportError as e:
            # Re-raise import errors with helpful message
            if provider_info.installation_extras:
                raise ImportError(
                    f"{provider_info.display_name} dependencies not installed. "
                    f"Install with: pip install abstractcore[{provider_info.installation_extras}]"
                ) from e
            else:
                raise ImportError(f"{provider_info.display_name} provider not available") from e


# Global registry instance
_registry = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


# Convenience functions for external use
def list_available_providers() -> List[str]:
    """Get list of all available provider names."""
    return get_provider_registry().list_provider_names()


def get_provider_info(provider_name: str) -> Optional[ProviderInfo]:
    """Get information about a specific provider."""
    return get_provider_registry().get_provider_info(provider_name)


def is_provider_available(provider_name: str) -> bool:
    """Check if a provider is available."""
    return get_provider_registry().is_provider_available(provider_name)


def get_all_providers_with_models() -> List[Dict[str, Any]]:
    """
    Get comprehensive information about all providers with available models.

    This is the main function that should be used throughout AbstractCore
    for provider discovery and information. It replaces the manual provider
    lists in factory.py and server/app.py.

    Returns:
        List of provider dictionaries with comprehensive metadata including:
        - name, display_name, type, description
        - model_count, status, supported_features
        - local_provider, authentication_required
        - installation_extras, sample models
    """
    return get_provider_registry().get_providers_with_models()


def get_all_providers_status() -> List[Dict[str, Any]]:
    """
    Get status information for all registered providers.

    This includes providers that may not have models available,
    useful for debugging and comprehensive provider listing.
    """
    return get_provider_registry().get_all_providers_status()


def create_provider(provider_name: str, model: Optional[str] = None, **kwargs):
    """
    Create a provider instance using the centralized registry.

    This replaces the factory logic and provides better error messages.
    """
    return get_provider_registry().create_provider_instance(provider_name, model, **kwargs)


def get_available_models_for_provider(provider_name: str, **kwargs) -> List[str]:
    """Get available models for a specific provider."""
    return get_provider_registry().get_available_models(provider_name, **kwargs)