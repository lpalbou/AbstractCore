"""
AbstractCore Configuration Module

Provides configuration management and command-line interface for AbstractCore.
"""

from .vision_config import handle_vision_commands, add_vision_arguments
from .manager import get_config_manager


def configure_provider(provider: str, **kwargs) -> None:
    """Configure runtime settings for a provider."""
    get_config_manager().configure_provider(provider, **kwargs)


def get_provider_config(provider: str) -> dict:
    """Get runtime configuration for a provider."""
    return get_config_manager().get_provider_config(provider)


def clear_provider_config(provider: str = None) -> None:
    """Clear runtime provider configuration."""
    get_config_manager().clear_provider_config(provider)


__all__ = [
    'handle_vision_commands',
    'add_vision_arguments',
    'get_config_manager',
    'configure_provider',
    'get_provider_config',
    'clear_provider_config'
]