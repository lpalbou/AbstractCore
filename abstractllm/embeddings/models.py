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


# SOTA 2025 embedding models
EMBEDDING_MODELS: Dict[str, EmbeddingModelConfig] = {
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
    "granite": EmbeddingModelConfig(
        name="granite",
        model_id="ibm-granite/granite-embedding-278m-multilingual",
        dimension=768,
        max_sequence_length=512,
        supports_matryoshka=False,
        matryoshka_dims=None,
        description="IBM's enterprise-grade multilingual embedding model (278M params)",
        multilingual=True,
        size_mb=278
    ),
    "stella-400m": EmbeddingModelConfig(
        name="stella-400m",
        model_id="dunzhang/stella_en_400M_v5",
        dimension=1024,
        max_sequence_length=512,
        supports_matryoshka=True,
        matryoshka_dims=[1024, 768, 512, 256],
        description="Excellent accuracy-to-size ratio, fine-tunable",
        multilingual=False,
        size_mb=400
    ),
    "nomic-embed": EmbeddingModelConfig(
        name="nomic-embed",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        dimension=768,
        max_sequence_length=8192,
        supports_matryoshka=True,
        matryoshka_dims=[768, 512, 256, 128],
        description="High-quality English embeddings, outperforms text-embedding-ada-002",
        multilingual=False,
        size_mb=550
    ),
    "mxbai-large": EmbeddingModelConfig(
        name="mxbai-large",
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        dimension=1024,
        max_sequence_length=512,
        supports_matryoshka=True,
        matryoshka_dims=[1024, 768, 512, 256],
        description="Outperforms text-embedding-3-large while being smaller",
        multilingual=False,
        size_mb=650
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
    """Get the default embedding model (EmbeddingGemma)."""
    return "embeddinggemma"