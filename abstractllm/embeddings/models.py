"""
Embedding Model Configurations
=============================

SOTA open-source embedding models with optimized configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class EmbeddingBackend(Enum):
    """Available inference backends for embeddings."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    OPENVINO = "openvino"


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