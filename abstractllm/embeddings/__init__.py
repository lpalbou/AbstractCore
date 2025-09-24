"""
Vector Embeddings for AbstractLLM Core
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
from .models import EmbeddingModelConfig, get_model_config, list_available_models

__all__ = ["EmbeddingManager", "EmbeddingModelConfig", "get_model_config", "list_available_models"]