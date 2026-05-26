"""
Vector Embeddings for AbstractCore
=====================================

Provides efficient text embedding with SOTA open-source models.
Designed for production use with semantic search and RAG capabilities.

Features:
- EmbeddingGemma (Google's 2025 SOTA on-device model)
- ONNX backend for 2-3x faster inference
- Smart caching (memory + disk)
- Matryoshka dimension truncation
- Event system integration
"""

from .manager import EmbeddingManager
from .models import (
    EmbeddingModelConfig,
    EmbeddingProviderConfig,
    get_model_config,
    get_provider_config,
    is_provider_supported,
    list_available_models,
    list_available_providers,
    list_direct_embedding_providers,
    list_endpoint_backed_providers,
    list_provider_details,
    list_providers_requiring_local_model_files,
)

__all__ = [
    "EmbeddingManager",
    "EmbeddingModelConfig",
    "EmbeddingProviderConfig",
    "get_model_config",
    "get_provider_config",
    "is_provider_supported",
    "list_available_models",
    "list_available_providers",
    "list_direct_embedding_providers",
    "list_endpoint_backed_providers",
    "list_provider_details",
    "list_providers_requiring_local_model_files",
]
