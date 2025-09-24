"""
Core Embedding Manager
=====================

Production-ready embedding generation with SOTA models and efficient serving.
"""

import hashlib
import pickle
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
import time

from .models import (
    EmbeddingModelConfig,
    EmbeddingBackend,
    get_model_config,
    get_default_model
)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Production-ready embedding manager with SOTA models and efficient serving.

    Features:
    - EmbeddingGemma default (Google's 2025 SOTA on-device model)
    - ONNX backend for 2-3x faster inference
    - Smart two-layer caching (memory + disk)
    - Matryoshka dimension truncation
    - Event system integration
    - Batch processing optimization
    """

    def __init__(
        self,
        model: str = None,
        backend: Union[str, EmbeddingBackend] = "auto",
        cache_dir: Optional[Path] = None,
        cache_size: int = 1000,
        output_dims: Optional[int] = None,
        trust_remote_code: bool = False
    ):
        """Initialize the embedding manager.

        Args:
            model: Model name or HuggingFace model ID. Defaults to EmbeddingGemma.
            backend: Inference backend ('auto', 'pytorch', 'onnx', 'openvino')
            cache_dir: Directory for persistent cache. Defaults to ~/.abstractllm/embeddings
            cache_size: Maximum number of embeddings to cache in memory
            output_dims: Output dimensions for Matryoshka truncation (if supported)
            trust_remote_code: Whether to trust remote code (for some models)
        """
        # Model configuration
        if model is None:
            model = get_default_model()

        # Handle both model names and direct HuggingFace IDs
        if model in ["embeddinggemma", "stella-400m", "nomic-embed", "mxbai-large"]:
            self.model_config = get_model_config(model)
            self.model_id = self.model_config.model_id
        else:
            # Direct HuggingFace model ID
            self.model_id = model
            self.model_config = None

        self.backend = EmbeddingBackend(backend) if backend != "auto" else None
        self.cache_dir = cache_dir or Path.home() / ".abstractllm" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_size = cache_size
        self.output_dims = output_dims
        self.trust_remote_code = trust_remote_code

        # Validate Matryoshka dimensions
        if self.output_dims and self.model_config:
            if not self.model_config.supports_matryoshka:
                logger.warning(f"Model {self.model_id} doesn't support Matryoshka. Ignoring output_dims.")
                self.output_dims = None
            elif self.output_dims not in self.model_config.matryoshka_dims:
                logger.warning(f"Dimension {self.output_dims} not in supported dims {self.model_config.matryoshka_dims}")

        # Initialize model
        self.model = None
        self._load_model()

        # Set up persistent cache
        cache_name = self.model_id.replace("/", "_").replace("-", "_")
        if self.output_dims:
            cache_name += f"_dim{self.output_dims}"
        self.cache_file = self.cache_dir / f"{cache_name}_cache.pkl"
        self._persistent_cache = self._load_persistent_cache()

        # Import events if available
        try:
            from ..events import EventType, emit_global
            self.has_events = True
            self.EventType = EventType
            self.emit_global = emit_global

            # Add embedding events if not present
            if not hasattr(EventType, 'EMBEDDING_GENERATED'):
                EventType.EMBEDDING_GENERATED = "embedding_generated"
            if not hasattr(EventType, 'EMBEDDING_CACHED'):
                EventType.EMBEDDING_CACHED = "embedding_cached"

        except ImportError:
            self.has_events = False

    def _load_model(self):
        """Load the embedding model with optimal backend."""
        try:
            import sentence_transformers

            # Determine best backend
            backend = self._select_backend()

            # Load model with backend
            if backend == EmbeddingBackend.ONNX:
                try:
                    self.model = sentence_transformers.SentenceTransformer(
                        self.model_id,
                        backend="onnx",
                        trust_remote_code=self.trust_remote_code
                    )
                    logger.info(f"Loaded {self.model_id} with ONNX backend (optimized)")
                except Exception as e:
                    logger.warning(f"ONNX backend failed: {e}. Falling back to PyTorch.")
                    self.model = sentence_transformers.SentenceTransformer(
                        self.model_id,
                        trust_remote_code=self.trust_remote_code
                    )
            else:
                self.model = sentence_transformers.SentenceTransformer(
                    self.model_id,
                    trust_remote_code=self.trust_remote_code
                )
                logger.info(f"Loaded {self.model_id} with PyTorch backend")

        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding functionality. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_id}: {e}")
            raise

    def _select_backend(self) -> EmbeddingBackend:
        """Select the optimal backend automatically."""
        if self.backend:
            return self.backend

        # Auto-select: prefer ONNX if available
        try:
            import onnxruntime
            return EmbeddingBackend.ONNX
        except ImportError:
            return EmbeddingBackend.PYTORCH

    def _load_persistent_cache(self) -> Dict[str, List[float]]:
        """Load persistent cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.debug(f"Loaded {len(cache)} embeddings from persistent cache")
                return cache
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
        return {}

    def _save_persistent_cache(self):
        """Save persistent cache to disk."""
        try:
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._persistent_cache, f)
            logger.debug(f"Saved {len(self._persistent_cache)} embeddings to persistent cache")
        except Exception as e:
            logger.warning(f"Failed to save persistent cache: {e}")

    def _text_hash(self, text: str) -> str:
        """Generate hash for text caching."""
        content = text + str(self.output_dims) if self.output_dims else text
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event if events are available."""
        if self.has_events:
            try:
                self.emit_global(event_type, data)
            except Exception as e:
                logger.debug(f"Event emission failed: {e}")

    @lru_cache(maxsize=1000)
    def embed(self, text: str) -> List[float]:
        """Embed a single text with caching and optimization.

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding
        """
        start_time = time.time()

        if not text or not text.strip():
            # Return zero vector for empty text
            dim = self.output_dims or self.get_dimension()
            return [0.0] * dim

        text_hash = self._text_hash(text)

        # Check persistent cache first
        if text_hash in self._persistent_cache:
            embedding = self._persistent_cache[text_hash]
            self._emit_event("embedding_cached", {
                "text_length": len(text),
                "cache_hit": True,
                "model": self.model_id,
                "dimension": len(embedding)
            })
            return embedding

        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()

            # Apply Matryoshka truncation if specified
            if self.output_dims and len(embedding) > self.output_dims:
                embedding = embedding[:self.output_dims]

            # Store in persistent cache
            self._persistent_cache[text_hash] = embedding

            # Periodically save cache
            if len(self._persistent_cache) % 10 == 0:
                self._save_persistent_cache()

            # Emit event
            duration_ms = (time.time() - start_time) * 1000
            self._emit_event("embedding_generated", {
                "text_length": len(text),
                "cache_hit": False,
                "model": self.model_id,
                "dimension": len(embedding),
                "duration_ms": duration_ms,
                "backend": self.backend.value if self.backend else "pytorch"
            })

            logger.debug(f"Generated embedding for text (length: {len(text)}, dims: {len(embedding)})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            # Return zero vector as fallback
            dim = self.output_dims or self.get_dimension()
            return [0.0] * dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently using batch processing.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, one for each input text
        """
        if not texts:
            return []

        start_time = time.time()

        # Separate cached and uncached texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                dim = self.output_dims or self.get_dimension()
                cached_embeddings[i] = [0.0] * dim
            else:
                text_hash = self._text_hash(text)
                if text_hash in self._persistent_cache:
                    cached_embeddings[i] = self._persistent_cache[text_hash]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

        # Process uncached texts in batch
        if uncached_texts:
            try:
                batch_embeddings = self.model.encode(
                    uncached_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                # Convert to list and apply Matryoshka truncation
                for i, (text, embedding, idx) in enumerate(zip(uncached_texts, batch_embeddings, uncached_indices)):
                    embedding_list = embedding.tolist()

                    # Apply Matryoshka truncation if specified
                    if self.output_dims and len(embedding_list) > self.output_dims:
                        embedding_list = embedding_list[:self.output_dims]

                    text_hash = self._text_hash(text)
                    self._persistent_cache[text_hash] = embedding_list
                    cached_embeddings[idx] = embedding_list

                logger.debug(f"Generated {len(batch_embeddings)} embeddings in batch")

            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                # Fill with zero vectors as fallback
                dim = self.output_dims or self.get_dimension()
                zero_embedding = [0.0] * dim
                for idx in uncached_indices:
                    cached_embeddings[idx] = zero_embedding

        # Save cache after batch processing
        if uncached_texts:
            self._save_persistent_cache()

        # Emit batch event
        duration_ms = (time.time() - start_time) * 1000
        cache_hits = len(texts) - len(uncached_texts)
        self._emit_event("embedding_batch_generated", {
            "batch_size": len(texts),
            "cache_hits": cache_hits,
            "new_embeddings": len(uncached_texts),
            "model": self.model_id,
            "duration_ms": duration_ms
        })

        # Return embeddings in original order
        return [cached_embeddings[i] for i in range(len(texts))]

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.output_dims:
            return self.output_dims
        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            import numpy as np

            embedding1 = np.array(self.embed(text1))
            embedding2 = np.array(self.embed(text2))

            # Compute cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)

            if norm_product == 0:
                return 0.0

            similarity = dot_product / norm_product
            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            "persistent_cache_size": len(self._persistent_cache),
            "memory_cache_info": self.embed.cache_info()._asdict(),
            "embedding_dimension": self.get_dimension(),
            "model_id": self.model_id,
            "backend": self.backend.value if self.backend else "auto",
            "cache_file": str(self.cache_file),
            "output_dims": self.output_dims
        }

    def clear_cache(self):
        """Clear both memory and persistent caches."""
        self.embed.cache_clear()
        self._persistent_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cleared all embedding caches")

    def __del__(self):
        """Ensure persistent cache is saved when object is destroyed."""
        try:
            self._save_persistent_cache()
        except:
            pass  # Ignore errors during cleanup