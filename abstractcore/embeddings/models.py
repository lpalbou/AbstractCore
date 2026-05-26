"""
Embedding Model Configurations
=============================

SOTA open-source embedding models with optimized configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class EmbeddingBackend(Enum):
    """Available inference backends for embeddings."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    OPENVINO = "openvino"


@dataclass(frozen=True)
class EmbeddingProviderConfig:
    """Configuration for a supported text embedding provider."""

    provider: str
    label: str
    transport: str
    base_url_configurable: bool = False
    base_url_env_vars: Tuple[str, ...] = ()
    default_base_url: Optional[str] = None
    auth: str = "none"
    requires_local_model_files: bool = False
    server_invocation_strategy: str = "embedding_manager"

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe provider metadata payload for discovery APIs."""
        # Locality is deliberately absent here: endpoint location is derived from
        # the configured route base_url, not from provider identity.
        data: Dict[str, object] = {
            "id": self.provider,
            "provider": self.provider,
            "label": self.label,
            "transport": self.transport,
        }
        if self.base_url_configurable:
            data["base_url_configurable"] = True
        if self.base_url_env_vars:
            data["base_url_env_vars"] = list(self.base_url_env_vars)
        if self.default_base_url:
            data["default_base_url"] = self.default_base_url
        if self.auth != "none":
            data["auth"] = self.auth
        if self.requires_local_model_files:
            data["requires_local_model_files"] = True
        return data


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str
    model_id: str
    dimension: int
    max_sequence_length: int
    supports_matryoshka: bool
    matryoshka_dims: Optional[List[int]]
    description: str
    multilingual: bool = False
    size_mb: Optional[float] = None


EMBEDDING_PROVIDERS: Dict[str, EmbeddingProviderConfig] = {
    "huggingface": EmbeddingProviderConfig(
        provider="huggingface",
        label="HuggingFace",
        transport="python_inprocess",
        requires_local_model_files=True,
    ),
    "lmstudio": EmbeddingProviderConfig(
        provider="lmstudio",
        label="LMStudio",
        transport="openai_compatible_http",
        base_url_configurable=True,
        base_url_env_vars=("LMSTUDIO_BASE_URL",),
        default_base_url="http://localhost:1234/v1",
        server_invocation_strategy="provider_direct",
    ),
    "ollama": EmbeddingProviderConfig(
        provider="ollama",
        label="Ollama",
        transport="ollama_native_http",
        base_url_configurable=True,
        base_url_env_vars=("OLLAMA_BASE_URL", "OLLAMA_HOST"),
        default_base_url="http://localhost:11434",
    ),
    "vllm": EmbeddingProviderConfig(
        provider="vllm",
        label="vLLM",
        transport="openai_compatible_http",
        base_url_configurable=True,
        base_url_env_vars=("VLLM_BASE_URL",),
        default_base_url="http://localhost:8000/v1",
        auth="optional",
        server_invocation_strategy="provider_direct",
    ),
    "openai": EmbeddingProviderConfig(
        provider="openai",
        label="OpenAI",
        transport="openai_http",
        base_url_configurable=True,
        base_url_env_vars=("OPENAI_BASE_URL",),
        default_base_url="https://api.openai.com/v1",
        auth="required",
        server_invocation_strategy="provider_direct",
    ),
    "openai-compatible": EmbeddingProviderConfig(
        provider="openai-compatible",
        label="OpenAI-compatible",
        transport="openai_compatible_http",
        base_url_configurable=True,
        base_url_env_vars=("OPENAI_BASE_URL",),
        default_base_url="http://localhost:1234/v1",
        auth="optional",
        server_invocation_strategy="provider_direct",
    ),
    "openrouter": EmbeddingProviderConfig(
        provider="openrouter",
        label="OpenRouter",
        transport="openai_compatible_http",
        base_url_configurable=True,
        base_url_env_vars=("OPENROUTER_BASE_URL",),
        default_base_url="https://openrouter.ai/api/v1",
        auth="required",
        server_invocation_strategy="provider_direct",
    ),
    "portkey": EmbeddingProviderConfig(
        provider="portkey",
        label="Portkey",
        transport="openai_compatible_http",
        base_url_configurable=True,
        base_url_env_vars=("PORTKEY_BASE_URL",),
        default_base_url="https://api.portkey.ai/v1",
        auth="required",
        server_invocation_strategy="provider_direct",
    ),
}


# Favored HuggingFace Embedding Models
EMBEDDING_MODELS: Dict[str, EmbeddingModelConfig] = {
    "all-minilm-l6-v2": EmbeddingModelConfig(
        name="all-minilm-l6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_sequence_length=256,
        supports_matryoshka=False,
        matryoshka_dims=None,
        description="Lightweight, fast embedding model - perfect for local development and testing (default)",
        multilingual=False,
        size_mb=90
    ),
    "embeddinggemma": EmbeddingModelConfig(
        name="embeddinggemma",
        model_id="google/embeddinggemma-300m",
        dimension=768,
        max_sequence_length=8192,
        supports_matryoshka=True,
        matryoshka_dims=[768, 512, 256, 128],
        description="Google's 2025 SOTA on-device embedding model (300M params)",
        multilingual=True,
        size_mb=300
    ),
    "qwen3-embedding": EmbeddingModelConfig(
        name="qwen3-embedding",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        max_sequence_length=8192,
        supports_matryoshka=False,
        matryoshka_dims=None,
        description="Qwen 0.6B embedding model - efficient multilingual support",
        multilingual=True,
        size_mb=600
    ),
    "granite-30m": EmbeddingModelConfig(
        name="granite-30m",
        model_id="ibm-granite/granite-embedding-30m-english",
        dimension=384,
        max_sequence_length=512,
        supports_matryoshka=False,
        matryoshka_dims=None,
        description="IBM Granite 30M embedding model - English only, ultra-lightweight",
        multilingual=False,
        size_mb=30
    ),
    "granite-107m": EmbeddingModelConfig(
        name="granite-107m",
        model_id="ibm-granite/granite-embedding-107m-multilingual",
        dimension=768,
        max_sequence_length=512,
        supports_matryoshka=False,
        matryoshka_dims=None,
        description="IBM Granite 107M embedding model - multilingual, balanced size",
        multilingual=True,
        size_mb=107
    ),
    "granite-278m": EmbeddingModelConfig(
        name="granite-278m",
        model_id="ibm-granite/granite-embedding-278m-multilingual",
        dimension=768,
        max_sequence_length=512,
        supports_matryoshka=False,
        matryoshka_dims=None,
        description="IBM Granite 278M embedding model - multilingual, high quality",
        multilingual=True,
        size_mb=278
    ),
    "nomic-embed-v1.5": EmbeddingModelConfig(
        name="nomic-embed-v1.5",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        dimension=768,
        max_sequence_length=8192,
        supports_matryoshka=True,
        matryoshka_dims=[768, 512, 256, 128],
        description="Nomic Embed v1.5 - high-quality English embeddings with Matryoshka",
        multilingual=False,
        size_mb=550
    ),
    "nomic-embed-v2-moe": EmbeddingModelConfig(
        name="nomic-embed-v2-moe",
        model_id="nomic-ai/nomic-embed-text-v2-moe",
        dimension=768,
        max_sequence_length=8192,
        supports_matryoshka=True,
        matryoshka_dims=[768, 512, 256, 128],
        description="Nomic Embed v2 MoE - mixture of experts for enhanced performance",
        multilingual=False,
        size_mb=800
    )
}


def _normalize_provider(provider: Optional[str]) -> str:
    return str(provider or "").strip().lower()


def get_provider_config(provider: str) -> EmbeddingProviderConfig:
    """Get configuration for a supported embedding provider."""
    provider_name = _normalize_provider(provider)
    if provider_name not in EMBEDDING_PROVIDERS:
        available = ", ".join(EMBEDDING_PROVIDERS.keys())
        raise ValueError(f"Provider '{provider}' not supported for embeddings. Available: {available}")
    return EMBEDDING_PROVIDERS[provider_name]


def is_provider_supported(provider: str) -> bool:
    """Return whether a provider is supported for text embeddings."""
    return _normalize_provider(provider) in EMBEDDING_PROVIDERS


def list_available_providers() -> List[str]:
    """List all supported text embedding provider IDs."""
    return list(EMBEDDING_PROVIDERS.keys())


def list_provider_details(provider: Optional[str] = None) -> List[Dict[str, object]]:
    """List supported text embedding provider metadata."""
    wanted = _normalize_provider(provider)
    out: List[Dict[str, object]] = []
    for config in EMBEDDING_PROVIDERS.values():
        if wanted and config.provider.lower() != wanted:
            continue
        out.append(config.to_dict())
    return out


def list_providers_requiring_local_model_files() -> List[str]:
    """List embedding providers whose default path loads local model files."""
    return [
        config.provider
        for config in EMBEDDING_PROVIDERS.values()
        if config.requires_local_model_files
    ]


def list_endpoint_backed_providers() -> List[str]:
    """List embedding providers whose transport is an HTTP endpoint."""
    return [
        config.provider
        for config in EMBEDDING_PROVIDERS.values()
        if config.transport != "python_inprocess"
    ]


def list_direct_embedding_providers() -> List[str]:
    """List providers whose embedding API can be called directly by the server."""
    return [
        config.provider
        for config in EMBEDDING_PROVIDERS.values()
        if config.server_invocation_strategy == "provider_direct"
    ]


def get_model_config(model_name: str) -> EmbeddingModelConfig:
    """Get configuration for a specific model.

    Args:
        model_name: Name of the embedding model

    Returns:
        EmbeddingModelConfig for the specified model

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in EMBEDDING_MODELS:
        available = ", ".join(EMBEDDING_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available: {available}")

    return EMBEDDING_MODELS[model_name]


def list_available_models() -> List[str]:
    """List all available embedding models."""
    return list(EMBEDDING_MODELS.keys())


def get_default_model() -> str:
    """Get the default embedding model (all-MiniLM L6-v2) - optimized for speed with perfect clustering."""
    return "all-minilm-l6-v2"
