# LLM provider implementations

from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .ollama_provider import OllamaProvider
from .lmstudio_provider import LMStudioProvider
from .huggingface_provider import HuggingFaceProvider
from .mlx_provider import MLXProvider
from .vllm_provider import VLLMProvider
from .openai_compatible_provider import OpenAICompatibleProvider

# Provider registry for centralized provider discovery and management
from .registry import (
    ProviderRegistry,
    ProviderInfo,
    get_provider_registry,
    list_available_providers,
    get_provider_info,
    is_provider_available,
    get_all_providers_with_models,
    get_all_providers_status,
    create_provider,
    get_available_models_for_provider
)

# Model capability filtering (new system)
from .model_capabilities import (
    ModelInputCapability,
    ModelOutputCapability,
    get_model_input_capabilities,
    get_model_output_capabilities,
    filter_models_by_capabilities,
    get_capability_summary
)

__all__ = [
    # Provider classes
    'BaseProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'OllamaProvider',
    'LMStudioProvider',
    'HuggingFaceProvider',
    'MLXProvider',
    'VLLMProvider',
    'OpenAICompatibleProvider',

    # Provider registry
    'ProviderRegistry',
    'ProviderInfo',
    'get_provider_registry',
    'list_available_providers',
    'get_provider_info',
    'is_provider_available',
    'get_all_providers_with_models',
    'get_all_providers_status',
    'create_provider',
    'get_available_models_for_provider',

    # Model capability filtering (new system)
    'ModelInputCapability',
    'ModelOutputCapability',
    'get_model_input_capabilities',
    'get_model_output_capabilities',
    'filter_models_by_capabilities',
    'get_capability_summary',
]