"""
LLM provider implementations.

Important: keep this package import-light.

Historically this module eagerly imported every provider implementation, which
pulled in heavyweight optional dependencies (e.g. torch, transformers, outlines)
even when users only wanted a lightweight local provider like LMStudio/Ollama.

We now expose providers and registry helpers via lazy attribute loading
(PEP 562: module __getattr__). This keeps `import abstractcore.providers` and
imports of provider submodules fast and import-safe.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Provider base / implementations
    "BaseProvider": ("abstractcore.providers.base", "BaseProvider"),
    "OpenAIProvider": ("abstractcore.providers.openai_provider", "OpenAIProvider"),
    "AnthropicProvider": ("abstractcore.providers.anthropic_provider", "AnthropicProvider"),
    "OllamaProvider": ("abstractcore.providers.ollama_provider", "OllamaProvider"),
    "LMStudioProvider": ("abstractcore.providers.lmstudio_provider", "LMStudioProvider"),
    "HuggingFaceProvider": ("abstractcore.providers.huggingface_provider", "HuggingFaceProvider"),
    "MLXProvider": ("abstractcore.providers.mlx_provider", "MLXProvider"),
    "VLLMProvider": ("abstractcore.providers.vllm_provider", "VLLMProvider"),
    "OpenAICompatibleProvider": ("abstractcore.providers.openai_compatible_provider", "OpenAICompatibleProvider"),
    "OpenRouterProvider": ("abstractcore.providers.openrouter_provider", "OpenRouterProvider"),

    # Provider registry helpers
    "ProviderRegistry": ("abstractcore.providers.registry", "ProviderRegistry"),
    "ProviderInfo": ("abstractcore.providers.registry", "ProviderInfo"),
    "get_provider_registry": ("abstractcore.providers.registry", "get_provider_registry"),
    "list_available_providers": ("abstractcore.providers.registry", "list_available_providers"),
    "get_provider_info": ("abstractcore.providers.registry", "get_provider_info"),
    "is_provider_available": ("abstractcore.providers.registry", "is_provider_available"),
    "get_all_providers_with_models": ("abstractcore.providers.registry", "get_all_providers_with_models"),
    "get_all_providers_status": ("abstractcore.providers.registry", "get_all_providers_status"),
    "create_provider": ("abstractcore.providers.registry", "create_provider"),
    "get_available_models_for_provider": ("abstractcore.providers.registry", "get_available_models_for_provider"),

    # Model capability filtering
    "ModelInputCapability": ("abstractcore.providers.model_capabilities", "ModelInputCapability"),
    "ModelOutputCapability": ("abstractcore.providers.model_capabilities", "ModelOutputCapability"),
    "get_model_input_capabilities": ("abstractcore.providers.model_capabilities", "get_model_input_capabilities"),
    "get_model_output_capabilities": ("abstractcore.providers.model_capabilities", "get_model_output_capabilities"),
    "filter_models_by_capabilities": ("abstractcore.providers.model_capabilities", "filter_models_by_capabilities"),
    "get_capability_summary": ("abstractcore.providers.model_capabilities", "get_capability_summary"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    target = _LAZY_ATTRS.get(name)
    if not target:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    mod = import_module(module_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))


__all__ = list(_LAZY_ATTRS.keys())

