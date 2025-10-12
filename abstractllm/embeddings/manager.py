"""
Core Embedding Manager
=====================

Production-ready embedding generation with SOTA models and efficient serving.
"""

import hashlib
import pickle
import logging
import atexit
import sys
import builtins
import warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, TYPE_CHECKING
import time

if TYPE_CHECKING:
    import numpy as np

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None

try:
    from ..events import EventType, emit_global
except ImportError:
    EventType = None
    emit_global = None

from .models import EmbeddingBackend, get_model_config, list_available_models, get_default_model

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_onnx_warnings():
    """Temporarily suppress known harmless ONNX and sentence-transformers warnings."""
    with warnings.catch_warnings():
        # Suppress PyTorch ONNX registration warnings (harmless in PyTorch 2.8+)
        warnings.filterwarnings(
            "ignore",
            message=".*Symbolic function.*already registered.*",
            category=UserWarning,
            module="torch.onnx._internal.registration"
        )

        # Suppress sentence-transformers multiple ONNX file warnings
        warnings.filterwarnings(
            "ignore",
            message=".*Multiple ONNX files found.*defaulting to.*",
            category=UserWarning,
            module="sentence_transformers.models.Transformer"
        )

        # Suppress ONNX Runtime provider warnings (these are system-level logs)
        # Note: CoreML warnings are logged directly to stderr by ONNX Runtime,
        # not through Python's warning system, so they're harder to suppress

        yield


def _get_optimal_onnx_model() -> Optional[str]:
    """Select optimal ONNX model using conservative strategy.

    Returns:
        ONNX model filename or None to use default
    """
    # Conservative strategy: try O3 optimization (good performance, widely supported)
    # If it fails, sentence-transformers will fallback to model.onnx automatically
    return "onnx/model_O3.onnx"


class EmbeddingManager:
    """
    Production-ready embedding manager with multi-provider support and efficient serving.

    Supported Providers:
    - HuggingFace (default): Local sentence-transformers models with ONNX acceleration
    - Ollama: Local embedding models via Ollama API
    - LMStudio: Local embedding models via LMStudio API

    Features:
    - Multi-provider support (HuggingFace, Ollama, LMStudio)
    - Smart two-layer caching (memory + disk) across all providers
    - ONNX backend for 2-3x faster inference (HuggingFace)
    - Matryoshka dimension truncation (when supported)
    - Event system integration
    - Batch processing optimization
    - Unified interface regardless of provider
    """

    def __init__(
        self,
        model: str = None,
        provider: str = "huggingface",
        backend: Union[str, EmbeddingBackend] = "auto",
        cache_dir: Optional[Path] = None,
        cache_size: int = 1000,
        output_dims: Optional[int] = None,
        trust_remote_code: bool = False
    ):
        """Initialize the embedding manager.

        Args:
            model: Model identifier (HuggingFace model ID for HF provider, model name for others).
            provider: Embedding provider ('huggingface', 'ollama', 'lmstudio'). Defaults to 'huggingface'.
            backend: Inference backend for HuggingFace ('auto', 'pytorch', 'onnx', 'openvino')
            cache_dir: Directory for persistent cache. Defaults to ~/.abstractllm/embeddings
            cache_size: Maximum number of embeddings to cache in memory
            output_dims: Output dimensions for Matryoshka truncation (if supported by provider)
            trust_remote_code: Whether to trust remote code (HuggingFace only)
        """
        # Store provider
        self.provider = provider.lower()
        
        # Validate provider
        if self.provider not in ["huggingface", "ollama", "lmstudio"]:
            raise ValueError(f"Unsupported provider: {provider}. Supported: huggingface, ollama, lmstudio")
        
        # Initialize provider-specific attributes
        self.model_config = None
        self._provider_instance = None
        
        # Set up model identifier
        if self.provider == "huggingface":
            # Model configuration - HuggingFace only
            if model is None:
                model = get_default_model()  # Returns alias "all-minilm-l6-v2"

            # Handle model aliases from our favored models config
            if model in list_available_models():
                self.model_config = get_model_config(model)
                self.model_id = self.model_config.model_id
            else:
                # Direct HuggingFace model ID
                self.model_id = model
                self.model_config = None

            self.backend = EmbeddingBackend(backend) if backend != "auto" else None
            self.trust_remote_code = trust_remote_code
            
            # Validate Matryoshka dimensions
            if output_dims and self.model_config:
                if not self.model_config.supports_matryoshka:
                    logger.warning(f"Model {self.model_id} doesn't support Matryoshka. Ignoring output_dims.")
                    output_dims = None
                elif output_dims not in self.model_config.matryoshka_dims:
                    logger.warning(f"Dimension {output_dims} not in supported dims {self.model_config.matryoshka_dims}")
        else:
            # Ollama or LMStudio provider
            if model is None:
                raise ValueError(f"Model name is required for {self.provider} provider")
            
            self.model_id = model
            self.backend = None
            self.trust_remote_code = False
            
            # Create provider instance for delegation
            if self.provider == "ollama":
                from ..providers.ollama_provider import OllamaProvider
                self._provider_instance = OllamaProvider(model=model)
                logger.info(f"Initialized Ollama embedding provider with model: {model}")
            elif self.provider == "lmstudio":
                from ..providers.lmstudio_provider import LMStudioProvider
                self._provider_instance = LMStudioProvider(model=model)
                logger.info(f"Initialized LMStudio embedding provider with model: {model}")

        # Common setup for all providers
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".abstractllm" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_size = cache_size
        self.output_dims = output_dims

        # Initialize model (HuggingFace only)
        self.model = None
        if self.provider == "huggingface":
            self._load_model()

        # Set up persistent cache (include provider in cache name for isolation)
        cache_name = f"{self.provider}_{self.model_id.replace('/', '_').replace('-', '_')}"
        if self.output_dims:
            cache_name += f"_dim{self.output_dims}"
        self.cache_file = self.cache_dir / f"{cache_name}_cache.pkl"
        self._persistent_cache = self._load_persistent_cache()

        # Normalized embeddings cache for performance optimization
        self.normalized_cache_file = self.cache_dir / f"{cache_name}_normalized_cache.pkl"
        self._normalized_cache = self._load_normalized_cache()

        # Register cleanup functions to save cache before Python shutdown
        # Use atexit instead of __del__ for reliable cleanup
        atexit.register(self._safe_save_persistent_cache)
        atexit.register(self._safe_save_normalized_cache)

        # Configure events if available
        if EventType is not None and emit_global is not None:
            self.has_events = True
            self.EventType = EventType
            self.emit_global = emit_global

            # Add embedding events if not present
            if not hasattr(EventType, 'EMBEDDING_GENERATED'):
                EventType.EMBEDDING_GENERATED = "embedding_generated"
            if not hasattr(EventType, 'EMBEDDING_CACHED'):
                EventType.EMBEDDING_CACHED = "embedding_cached"
            if not hasattr(EventType, 'EMBEDDING_BATCH_GENERATED'):
                EventType.EMBEDDING_BATCH_GENERATED = "embedding_batch_generated"
        else:
            self.has_events = False

    def _load_model(self):
        """Load the HuggingFace embedding model with optimal backend and reduced warnings."""
        try:
            if sentence_transformers is None:
                raise ImportError("sentence-transformers is required but not installed")

            # Set HuggingFace cache directory (sentence-transformers uses this automatically)
            import os
            hf_cache_dir = os.path.expanduser("~/.cache/huggingface/")
            os.environ.setdefault("HF_HOME", hf_cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
            os.environ.setdefault("HF_DATASETS_CACHE", hf_cache_dir)

            # Determine best backend
            backend = self._select_backend()

            # Load model with optimal ONNX selection and warning suppression
            with _suppress_onnx_warnings():
                if backend == EmbeddingBackend.ONNX:
                    try:
                        # Try optimized ONNX model first
                        optimal_onnx = _get_optimal_onnx_model()
                        model_kwargs = {"file_name": optimal_onnx} if optimal_onnx else {}

                        self.model = sentence_transformers.SentenceTransformer(
                            self.model_id,
                            backend="onnx",
                            model_kwargs=model_kwargs,
                            trust_remote_code=self.trust_remote_code
                        )
                        onnx_model = optimal_onnx or "model.onnx"
                        logger.info(f"Loaded {self.model_id} with ONNX backend ({onnx_model})")

                    except Exception as e:
                        logger.warning(f"Optimized ONNX failed: {e}. Trying basic ONNX.")
                        try:
                            # Fallback to basic ONNX
                            self.model = sentence_transformers.SentenceTransformer(
                                self.model_id,
                                backend="onnx",
                                trust_remote_code=self.trust_remote_code
                            )
                            logger.info(f"Loaded {self.model_id} with basic ONNX backend")
                        except Exception as e2:
                            logger.warning(f"All ONNX variants failed: {e2}. Falling back to PyTorch.")
                            self.model = sentence_transformers.SentenceTransformer(
                                self.model_id,
                                trust_remote_code=self.trust_remote_code
                            )
                            logger.info(f"Loaded {self.model_id} with PyTorch backend")
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
        """Select the optimal backend automatically with intelligent model compatibility checking."""
        if self.backend:
            return self.backend

        # Check if onnxruntime is available
        try:
            import onnxruntime  # type: ignore
            _ = onnxruntime.__version__
        except ImportError:
            return EmbeddingBackend.PYTORCH

        # Check if this model has good ONNX support before attempting ONNX
        if self._has_onnx_support():
            logger.debug(f"Model {self.model_id} has ONNX support, using ONNX backend")
            return EmbeddingBackend.ONNX
        else:
            logger.debug(f"Model {self.model_id} lacks ONNX support, using PyTorch backend")
            return EmbeddingBackend.PYTORCH

    def _has_onnx_support(self) -> bool:
        """Check if the model has good ONNX support to avoid problematic dynamic export."""
        # Check 1: Does the model have pre-exported ONNX files?
        if self._has_preexported_onnx():
            return True

        # Check 2: Is this a model type known to work well with ONNX export?
        if self._is_onnx_compatible_model():
            return True

        # Default: no ONNX support detected
        return False

    def _has_preexported_onnx(self) -> bool:
        """Check if the model has pre-exported ONNX files in HuggingFace cache."""
        try:
            import os
            from pathlib import Path

            # Get HuggingFace cache directory
            hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

            # Convert model ID to cache directory format (org--model)
            cache_dir_name = f"models--{self.model_id.replace('/', '--')}"
            model_cache_dir = hf_cache_dir / cache_dir_name

            if not model_cache_dir.exists():
                return False

            # Look for ONNX files in snapshots
            for snapshot_dir in model_cache_dir.glob("snapshots/*"):
                if snapshot_dir.is_dir():
                    # Check for common ONNX file patterns
                    onnx_patterns = ["model.onnx", "onnx/model.onnx", "onnx/model_O*.onnx"]
                    for pattern in onnx_patterns:
                        if list(snapshot_dir.glob(pattern)):
                            logger.debug(f"Found pre-exported ONNX files for {self.model_id}")
                            return True

            return False

        except Exception as e:
            logger.debug(f"Error checking for pre-exported ONNX files: {e}")
            return False

    def _is_onnx_compatible_model(self) -> bool:
        """Check if the model type/name is known to work well with ONNX export."""
        # Models known to work well with ONNX (based on sentence-transformers supported models)
        onnx_compatible_patterns = [
            # Popular embedding models with good ONNX support
            "sentence-transformers/all-minilm",
            "sentence-transformers/all-mpnet",
            "sentence-transformers/multi-qa",
            "sentence-transformers/paraphrase",
            "sentence-transformers/distiluse",
            "microsoft/DialoGPT",
            "microsoft/MiniLM",
            # BERT-based models generally work well
            "bert-",
            "distilbert-",
            "roberta-",
            # Some other well-supported models
            "thenlper/gte-",
            "BAAI/bge-",
        ]

        model_lower = self.model_id.lower()

        # Check if model matches any known compatible pattern
        for pattern in onnx_compatible_patterns:
            if pattern.lower() in model_lower:
                logger.debug(f"Model {self.model_id} matches ONNX-compatible pattern: {pattern}")
                return True

        # Models known to have ONNX export issues (avoid dynamic export)
        problematic_patterns = [
            "qwen",  # Qwen models often have ONNX export issues
            "llama",  # LLaMA models usually problematic for embeddings
            "mixtral",
            "deepseek",
            "codellama",
        ]

        for pattern in problematic_patterns:
            if pattern.lower() in model_lower:
                logger.debug(f"Model {self.model_id} matches problematic pattern: {pattern}")
                return False

        # For unknown models, be conservative and use PyTorch
        logger.debug(f"Model {self.model_id} is unknown for ONNX compatibility, using PyTorch")
        return False


    def _load_persistent_cache(self) -> Dict[str, List[float]]:
        """Load persistent cache from disk."""
        try:
            if self.cache_file.exists():
                with builtins.open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.debug(f"Loaded {len(cache)} embeddings from persistent cache")
                return cache
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
        return {}

    def _load_normalized_cache(self) -> Dict[str, List[float]]:
        """Load normalized embeddings cache from disk."""
        try:
            if self.normalized_cache_file.exists():
                with builtins.open(self.normalized_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.debug(f"Loaded {len(cache)} normalized embeddings from cache")
                return cache
        except Exception as e:
            logger.warning(f"Failed to load normalized cache: {e}")
        return {}

    def _save_persistent_cache(self):
        """Save persistent cache to disk."""
        try:
            # Check if cache file attributes exist (may not if initialization failed)
            if not hasattr(self, 'cache_file') or not hasattr(self, '_persistent_cache'):
                return

            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with builtins.open(self.cache_file, 'wb') as f:
                pickle.dump(self._persistent_cache, f)
            logger.debug(f"Saved {len(self._persistent_cache)} embeddings to persistent cache")
        except Exception as e:
            logger.warning(f"Failed to save persistent cache: {e}")

    def _safe_save_persistent_cache(self):
        """Safely save persistent cache, handling shutdown scenarios."""
        try:
            # Check if Python is shutting down
            if sys.meta_path is None:
                return  # Skip saving during shutdown

            self._save_persistent_cache()
        except Exception:
            # Silently ignore errors during shutdown
            pass

    def _save_normalized_cache(self):
        """Save normalized embeddings cache to disk."""
        try:
            # Check if cache file attributes exist (may not if initialization failed)
            if not hasattr(self, 'normalized_cache_file') or not hasattr(self, '_normalized_cache'):
                return

            # Ensure directory exists
            self.normalized_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with builtins.open(self.normalized_cache_file, 'wb') as f:
                pickle.dump(self._normalized_cache, f)
            logger.debug(f"Saved {len(self._normalized_cache)} normalized embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save normalized cache: {e}")

    def _safe_save_normalized_cache(self):
        """Safely save normalized cache, handling shutdown scenarios."""
        try:
            # Check if Python is shutting down
            if sys.meta_path is None:
                return  # Skip saving during shutdown

            self._save_normalized_cache()
        except Exception:
            # Silently ignore errors during shutdown
            pass

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

    def embed_normalized(self, text: str) -> List[float]:
        """Get normalized embedding for text with dedicated caching.

        Normalized embeddings enable faster similarity calculations using simple
        dot products instead of full cosine similarity computation.

        Args:
            text: Text to embed and normalize

        Returns:
            Normalized embedding vector (unit length)
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            dim = self.output_dims or self.get_dimension()
            return [0.0] * dim

        text_hash = self._text_hash(text + "_normalized")

        # Check normalized cache first
        if text_hash in self._normalized_cache:
            return self._normalized_cache[text_hash]

        try:
            import numpy as np

            # Get regular embedding
            embedding = np.array(self.embed(text))

            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm == 0:
                normalized_embedding = embedding.tolist()
            else:
                normalized_embedding = (embedding / norm).tolist()

            # Store in normalized cache
            self._normalized_cache[text_hash] = normalized_embedding

            # Periodically save normalized cache
            if len(self._normalized_cache) % 10 == 0:
                self._save_normalized_cache()

            return normalized_embedding

        except Exception as e:
            logger.error(f"Failed to compute normalized embedding: {e}")
            # Fallback to regular embedding
            return self.embed(text)

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
                "provider": self.provider,
                "dimension": len(embedding)
            })
            return embedding

        try:
            # Generate embedding based on provider
            if self.provider == "huggingface":
                # HuggingFace: Use sentence-transformers model
                embedding = self.model.encode(
                    text,
                    show_progress_bar=False,
                    convert_to_numpy=True
                ).tolist()

                # Apply Matryoshka truncation if specified
                if self.output_dims and len(embedding) > self.output_dims:
                    embedding = embedding[:self.output_dims]
                    
            else:
                # Ollama or LMStudio: Delegate to provider
                result = self._provider_instance.embed(input_text=text)
                
                # Extract embedding from OpenAI-compatible response
                if "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0]["embedding"]
                    
                    # Apply dimension truncation if specified
                    if self.output_dims and len(embedding) > self.output_dims:
                        embedding = embedding[:self.output_dims]
                else:
                    raise ValueError(f"Invalid response from {self.provider} provider")

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
                "provider": self.provider,
                "dimension": len(embedding),
                "duration_ms": duration_ms,
                "backend": self.backend.value if self.backend else self.provider
            })

            logger.debug(f"Generated embedding for text (length: {len(text)}, dims: {len(embedding)}, provider: {self.provider})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text with {self.provider}: {e}")
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
                if self.provider == "huggingface":
                    # HuggingFace: Use sentence-transformers batch encoding
                    batch_embeddings = self.model.encode(
                        uncached_texts,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )

                    # Convert to list and apply Matryoshka truncation
                    for i, (text, embedding, idx) in enumerate(zip(uncached_texts, batch_embeddings, uncached_indices)):
                        embedding_list = embedding.tolist()  # Convert numpy to list

                        # Apply Matryoshka truncation if specified
                        if self.output_dims and len(embedding_list) > self.output_dims:
                            embedding_list = embedding_list[:self.output_dims]

                        text_hash = self._text_hash(text)
                        self._persistent_cache[text_hash] = embedding_list
                        cached_embeddings[idx] = embedding_list

                    logger.debug(f"Generated {len(batch_embeddings)} embeddings in batch (HuggingFace)")
                    
                else:
                    # Ollama or LMStudio: Delegate to provider (supports batch in single call)
                    result = self._provider_instance.embed(input_text=uncached_texts)
                    
                    # Extract embeddings from OpenAI-compatible response
                    if "data" in result:
                        for text, embedding_data, idx in zip(uncached_texts, result["data"], uncached_indices):
                            embedding = embedding_data["embedding"]
                            
                            # Apply dimension truncation if specified
                            if self.output_dims and len(embedding) > self.output_dims:
                                embedding = embedding[:self.output_dims]
                            
                            text_hash = self._text_hash(text)
                            self._persistent_cache[text_hash] = embedding
                            cached_embeddings[idx] = embedding
                        
                        logger.debug(f"Generated {len(result['data'])} embeddings in batch ({self.provider})")
                    else:
                        raise ValueError(f"Invalid batch response from {self.provider} provider")

            except Exception as e:
                logger.error(f"Failed to embed batch with {self.provider}: {e}")
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
            "provider": self.provider,
            "duration_ms": duration_ms
        })

        # Return embeddings in original order
        return [cached_embeddings[i] for i in range(len(texts))]

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.output_dims:
            return self.output_dims
        
        if self.provider == "huggingface":
            return self.model.get_sentence_embedding_dimension()
        else:
            # For Ollama/LMStudio, we need to generate a test embedding to get dimension
            # This is cached, so it's only done once
            test_embedding = self.embed("test")
            return len(test_embedding)

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text for embedding usage calculations.

        This provides a rough estimate based on text length, suitable for usage tracking.
        For embeddings, this is mainly used for billing/quota calculations.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0

        # Simple heuristic: roughly 4 characters per token for most languages
        # This is consistent with OpenAI's embedding token estimation
        estimated_tokens = max(1, len(text) // 4)
        return estimated_tokens

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

    def compute_similarity_direct(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings directly.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            import numpy as np

            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)

            if norm_product == 0:
                return 0.0

            similarity = dot_product / norm_product
            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to compute direct similarity: {e}")
            return 0.0

    def compute_similarities(self, text: str, texts: List[str]) -> List[float]:
        """Compute cosine similarities between one text and a list of texts.

        Args:
            text: Reference text to compare against
            texts: List of texts to compare with the reference text

        Returns:
            List of cosine similarity scores between -1 and 1, one for each input text
        """
        if not texts:
            return []

        try:
            import numpy as np

            # Get embedding for reference text
            ref_embedding = np.array(self.embed(text))

            # Get embeddings for all comparison texts (using batch processing for efficiency)
            comparison_embeddings = self.embed_batch(texts)
            comparison_embeddings = np.array(comparison_embeddings)

            # Compute cosine similarities using vectorized operations
            similarities = []
            for comp_embedding in comparison_embeddings:
                comp_embedding = np.array(comp_embedding)

                # Compute cosine similarity
                dot_product = np.dot(ref_embedding, comp_embedding)
                norm_product = np.linalg.norm(ref_embedding) * np.linalg.norm(comp_embedding)

                if norm_product == 0:
                    similarities.append(0.0)
                else:
                    similarity = dot_product / norm_product
                    similarities.append(float(similarity))

            return similarities

        except Exception as e:
            logger.error(f"Failed to compute batch similarities: {e}")
            # Return zero similarities as fallback
            return [0.0] * len(texts)

    def compute_similarities_matrix(
        self,
        texts_left: List[str],
        texts_right: Optional[List[str]] = None,
        chunk_size: int = 500,
        normalized: bool = True,
        dtype: str = "float32",
        max_memory_gb: float = 4.0
    ) -> "np.ndarray":
        """Compute similarity matrix between two sets of texts using SOTA efficient methods.

        Creates an L×C matrix where L=len(texts_left) and C=len(texts_right or texts_left).
        Uses vectorized operations, chunking, and optional pre-normalization for efficiency.

        Args:
            texts_left: Left set of texts (rows in matrix)
            texts_right: Right set of texts (columns in matrix). If None, uses texts_left (L×L matrix)
            chunk_size: Process matrix in chunks of this many rows to manage memory
            normalized: If True, pre-normalize embeddings for 2x speedup
            dtype: Data type for computations ('float32' or 'float64')
            max_memory_gb: Maximum memory to use before switching to chunked processing

        Returns:
            NumPy array of shape (len(texts_left), len(texts_right or texts_left))
            with cosine similarity values between -1 and 1

        Examples:
            >>> embedder = EmbeddingManager()
            >>>
            >>> # Symmetric matrix (5x5)
            >>> texts = ["AI is amazing", "Machine learning rocks", "Python is great", "Data science", "Deep learning"]
            >>> matrix = embedder.compute_similarities_matrix(texts)
            >>>
            >>> # Asymmetric matrix (3x2)
            >>> queries = ["What is AI?", "How does ML work?", "Python tutorial"]
            >>> docs = ["Artificial intelligence guide", "Machine learning basics"]
            >>> matrix = embedder.compute_similarities_matrix(queries, docs)
        """
        if not texts_left:
            import numpy as np
            return np.array([], dtype=dtype).reshape(0, len(texts_right or []))

        # Use texts_left for both sides if texts_right not provided (symmetric matrix)
        if texts_right is None:
            texts_right = texts_left
            symmetric = True
        else:
            symmetric = False

        if not texts_right:
            import numpy as np
            return np.array([], dtype=dtype).reshape(len(texts_left), 0)

        try:
            import numpy as np

            start_time = time.time()

            # Get embeddings efficiently using batch processing
            logger.debug(f"Computing embeddings for {len(texts_left)}×{len(texts_right)} similarity matrix")

            embeddings_left = self.embed_batch(texts_left)
            if symmetric:
                embeddings_right = embeddings_left
            else:
                embeddings_right = self.embed_batch(texts_right)

            # Convert to numpy arrays with specified dtype
            embeddings_left = np.array(embeddings_left, dtype=dtype)
            embeddings_right = np.array(embeddings_right, dtype=dtype)

            L, C = len(texts_left), len(texts_right)
            D = embeddings_left.shape[1]  # Embedding dimension

            # Estimate memory requirements
            matrix_memory_gb = (L * C * 4) / (1024**3)  # float32 bytes to GB
            embedding_memory_gb = ((L + C) * D * 4) / (1024**3)
            total_memory_gb = matrix_memory_gb + embedding_memory_gb

            logger.debug(f"Estimated memory usage: {total_memory_gb:.2f}GB (matrix: {matrix_memory_gb:.2f}GB)")

            # Pre-normalize embeddings for efficiency (2x speedup)
            if normalized:
                # Compute norms
                norms_left = np.linalg.norm(embeddings_left, axis=1, keepdims=True)
                norms_right = np.linalg.norm(embeddings_right, axis=1, keepdims=True)

                # Avoid division by zero
                norms_left = np.where(norms_left == 0, 1, norms_left)
                norms_right = np.where(norms_right == 0, 1, norms_right)

                # Normalize embeddings
                embeddings_left = embeddings_left / norms_left
                embeddings_right = embeddings_right / norms_right

                logger.debug("Pre-normalized embeddings for efficiency")

            # Choose processing strategy based on memory requirements
            if total_memory_gb <= max_memory_gb and chunk_size >= L:
                # Direct computation - all fits in memory
                if normalized:
                    # Simple dot product after normalization
                    similarities = embeddings_left @ embeddings_right.T
                    # Clamp to valid range to handle floating point precision
                    import numpy as np
                    similarities = np.clip(similarities, -1.0, 1.0)
                else:
                    # Full cosine similarity computation
                    similarities = self._compute_cosine_similarity_matrix(embeddings_left, embeddings_right)

                logger.debug(f"Used direct computation ({total_memory_gb:.2f}GB)")

            else:
                # Chunked computation for memory efficiency
                logger.debug(f"Using chunked processing (chunks of {chunk_size} rows)")
                similarities = self._compute_chunked_similarity_matrix(
                    embeddings_left, embeddings_right, chunk_size, normalized, dtype
                )

            # Emit performance event
            duration_ms = (time.time() - start_time) * 1000
            self._emit_event("similarity_matrix_computed", {
                "matrix_shape": (L, C),
                "symmetric": symmetric,
                "normalized": normalized,
                "chunked": total_memory_gb > max_memory_gb,
                "memory_gb": total_memory_gb,
                "duration_ms": duration_ms,
                "model": self.model_id
            })

            logger.debug(f"Computed {L}×{C} similarity matrix in {duration_ms:.1f}ms")
            return similarities

        except Exception as e:
            logger.error(f"Failed to compute similarity matrix: {e}")
            # Return zero matrix as fallback
            import numpy as np
            return np.zeros((len(texts_left), len(texts_right)), dtype=dtype)

    def _compute_cosine_similarity_matrix(self, embeddings_left: "np.ndarray", embeddings_right: "np.ndarray") -> "np.ndarray":
        """Compute cosine similarity matrix using vectorized operations."""
        import numpy as np

        # Compute dot products (numerator)
        dot_products = embeddings_left @ embeddings_right.T

        # Compute norms
        norms_left = np.linalg.norm(embeddings_left, axis=1, keepdims=True)
        norms_right = np.linalg.norm(embeddings_right, axis=1, keepdims=True)

        # Compute norm products (denominator)
        norm_products = norms_left @ norms_right.T

        # Avoid division by zero
        norm_products = np.where(norm_products == 0, 1, norm_products)

        # Compute cosine similarities
        similarities = dot_products / norm_products

        # Clamp to valid cosine similarity range [-1, 1] to handle floating point precision
        similarities = np.clip(similarities, -1.0, 1.0)

        return similarities

    def _compute_chunked_similarity_matrix(
        self,
        embeddings_left: "np.ndarray",
        embeddings_right: "np.ndarray",
        chunk_size: int,
        normalized: bool,
        dtype: str
    ) -> "np.ndarray":
        """Compute similarity matrix in chunks to manage memory."""
        import numpy as np

        L, C = embeddings_left.shape[0], embeddings_right.shape[0]
        similarities = np.zeros((L, C), dtype=dtype)

        # Process in chunks
        for i in range(0, L, chunk_size):
            end_i = min(i + chunk_size, L)
            chunk_left = embeddings_left[i:end_i]

            if normalized:
                # Simple dot product for normalized embeddings
                chunk_similarities = chunk_left @ embeddings_right.T
                # Clamp to valid range to handle floating point precision
                chunk_similarities = np.clip(chunk_similarities, -1.0, 1.0)
            else:
                # Full cosine similarity computation for chunk
                chunk_similarities = self._compute_cosine_similarity_matrix(chunk_left, embeddings_right)

            similarities[i:end_i] = chunk_similarities

            # Log progress for large matrices
            if L > 1000:
                progress = (end_i / L) * 100
                logger.debug(f"Similarity matrix progress: {progress:.1f}%")

        return similarities

    def find_similar_clusters(
        self,
        texts: List[str],
        threshold: float = 0.8,
        min_cluster_size: int = 2,
        max_memory_gb: float = 4.0
    ) -> List[List[int]]:
        """Find clusters of similar texts using similarity matrix analysis.

        Groups texts that have similarity above the threshold into clusters.
        Useful for identifying duplicate or near-duplicate content, grouping similar
        documents, or finding semantic clusters for organization.

        Args:
            texts: List of texts to cluster
            threshold: Minimum similarity score for texts to be in same cluster (0.0 to 1.0)
            min_cluster_size: Minimum number of texts required to form a cluster
            max_memory_gb: Maximum memory for similarity matrix computation

        Returns:
            List of clusters, where each cluster is a list of text indices

        Examples:
            >>> embedder = EmbeddingManager()
            >>> texts = [
            ...     "Python is great for data science",
            ...     "Machine learning with Python",
            ...     "JavaScript for web development",
            ...     "Data science using Python",
            ...     "Web apps with JavaScript"
            ... ]
            >>> clusters = embedder.find_similar_clusters(texts, threshold=0.7)
            >>> # Result: [[0, 1, 3], [2, 4]] (Python cluster and JavaScript cluster)
        """
        if not texts or len(texts) < min_cluster_size:
            return []

        try:
            import numpy as np

            start_time = time.time()
            logger.debug(f"Finding clusters in {len(texts)} texts with threshold {threshold}")

            # Compute similarity matrix
            similarity_matrix = self.compute_similarities_matrix(
                texts,
                max_memory_gb=max_memory_gb
            )

            # Create adjacency matrix based on threshold
            # Set diagonal to False to avoid self-clustering
            adjacency = similarity_matrix >= threshold
            np.fill_diagonal(adjacency, False)

            # Find connected components (clusters) using simple graph traversal
            visited = set()
            clusters = []

            for i in range(len(texts)):
                if i in visited:
                    continue

                # Start new cluster
                cluster = []
                stack = [i]

                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue

                    visited.add(node)
                    cluster.append(node)

                    # Find all neighbors (similar texts)
                    neighbors = np.where(adjacency[node])[0]
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            stack.append(neighbor)

                # Only keep clusters that meet minimum size
                if len(cluster) >= min_cluster_size:
                    # Sort indices for consistent output
                    cluster.sort()
                    clusters.append(cluster)

            # Sort clusters by size (largest first)
            clusters.sort(key=len, reverse=True)

            # Emit clustering event
            duration_ms = (time.time() - start_time) * 1000
            clustered_texts = sum(len(cluster) for cluster in clusters)

            self._emit_event("clustering_completed", {
                "total_texts": len(texts),
                "clusters_found": len(clusters),
                "clustered_texts": clustered_texts,
                "unclustered_texts": len(texts) - clustered_texts,
                "threshold": threshold,
                "min_cluster_size": min_cluster_size,
                "duration_ms": duration_ms,
                "model": self.model_id
            })

            logger.debug(f"Found {len(clusters)} clusters ({clustered_texts}/{len(texts)} texts clustered) in {duration_ms:.1f}ms")
            return clusters

        except Exception as e:
            logger.error(f"Failed to find clusters: {e}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding caches."""
        return {
            "provider": self.provider,
            "persistent_cache_size": len(self._persistent_cache),
            "normalized_cache_size": len(self._normalized_cache),
            "memory_cache_info": self.embed.cache_info()._asdict(),
            "embedding_dimension": self.get_dimension(),
            "model_id": self.model_id,
            "backend": self.backend.value if self.backend else self.provider,
            "cache_file": str(self.cache_file),
            "normalized_cache_file": str(self.normalized_cache_file),
            "output_dims": self.output_dims
        }

    def clear_cache(self):
        """Clear both memory and persistent caches."""
        self.embed.cache_clear()
        self._persistent_cache.clear()
        self._normalized_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.normalized_cache_file.exists():
            self.normalized_cache_file.unlink()
        logger.info("Cleared all embedding caches")

    def save_caches(self):
        """Explicitly save both caches to disk."""
        self._save_persistent_cache()
        self._save_normalized_cache()