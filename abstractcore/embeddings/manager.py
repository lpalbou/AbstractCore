"""
Core Embedding Manager
=====================

Production-ready embedding generation with SOTA models and efficient serving.
"""

import hashlib
import pickle
import atexit
import gc
import os
import threading
import weakref
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
except Exception:
    sentence_transformers = None

try:
    from ..events import EventType, emit_global
except ImportError:
    EventType = None
    emit_global = None

from .models import EmbeddingBackend, get_model_config, get_default_model, list_available_models, list_available_providers
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


@contextmanager
def _suppress_onnx_warnings():
    """Temporarily suppress known harmless ONNX and sentence-transformers warnings.

    This suppresses the CoreML and node assignment warnings commonly seen on macOS.
    These warnings are informational only and don't impact performance or quality.

    To enable verbose ONNX logging for debugging, set: ABSTRACTCORE_ONNX_VERBOSE=1
    """
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

        # Suppress ONNX Runtime warnings at the source
        try:
            import onnxruntime as ort
            import os

            # Allow users to enable verbose ONNX logging for debugging
            # Set ABSTRACTCORE_ONNX_VERBOSE=1 to see ONNX warnings for debugging
            if os.environ.get("ABSTRACTCORE_ONNX_VERBOSE", "0") != "1":
                # Suppress the CoreML and node assignment warnings you may see on macOS
                # These are harmless informational messages that don't affect performance or quality:
                # - CoreML partitioning warnings: Normal behavior when model ops aren't all CoreML-compatible
                # - Node assignment warnings: ONNX Runtime intelligently assigns ops to best execution provider
                ort.set_default_logger_severity(3)  # Error level - suppresses warnings
        except ImportError:
            pass  # ONNX Runtime not available

        yield


def _hf_download_cache_folder() -> Optional[str]:
    """The hub cache a load that MAY download writes to: huggingface_hub's own
    resolution (HF_HUB_CACHE / HF_HOME / its default), passed per call as
    `cache_folder=` -- the same directory the offline resolver reads."""
    try:
        from huggingface_hub import constants as _hf_constants  # type: ignore

        return str(_hf_constants.HF_HUB_CACHE)
    except Exception:
        return None


def _merge_save_pickle_cache(path: Path, in_memory: Dict[str, Any], added: set, *, label: str) -> bool:
    """Persist a pickled embeddings cache without ever losing on-disk entries.

    Why (incident 2026-09-24): every EmbeddingManager saved its WHOLE in-memory
    cache over the file at interpreter exit, last writer wins. A process that
    loaded nothing (a test run, a second app on the same model) and exited
    later replaced the operator's populated cache with an empty one.

    The rules, in order:
    1. Nothing was added in this process (`added` empty) -> no write at all.
    2. Merge on save: under an advisory lock (POSIX), re-read the file as it
       is NOW and union it with the in-memory entries, so entries another
       process wrote since our load survive.
    3. Never write an empty mapping (an empty write can only lose data).
    4. Atomic publish: unique temp file in the same directory + os.replace,
       so a reader never sees a half-written pickle.
    Returns True when the file was written.
    """
    if not added:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = None
    try:
        try:
            import fcntl  # POSIX only; best-effort elsewhere

            lock_fh = builtins.open(str(path) + ".lock", "a+b")
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            lock_fh = None

        on_disk: Dict[str, Any] = {}
        if path.exists():
            try:
                with builtins.open(path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    on_disk = loaded
            except Exception as e:
                logger.warning(f"Existing {label} cache {path} is unreadable ({e}); writing this process's entries")

        merged: Dict[str, Any] = dict(on_disk)
        merged.update(in_memory)
        if not merged:
            return False

        import tempfile

        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(merged, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        added.clear()
        logger.debug(f"Saved {len(merged)} {label} to {path} ({len(on_disk)} already on disk)")
        return True
    finally:
        if lock_fh is not None:
            try:
                lock_fh.close()
            except Exception:
                pass


def _get_optimal_onnx_model() -> Optional[str]:
    """Select optimal ONNX model using conservative strategy.

    Returns:
        ONNX model filename or None to use default
    """
    # Conservative strategy: try O3 optimization (good performance, widely supported)
    # If it fails, sentence-transformers will fallback to model.onnx automatically
    return "onnx/model_O3.onnx"


# ---------------------------------------------------------------------------
# Process-level residency for in-process embedding models (mission MEM2).
#
# Before 2026-09-25 an EmbeddingManager could NEVER leave memory: `__init__`
# registered two BOUND METHODS with `atexit` (a strong reference for the life
# of the process) and `embed` was a class-level `@lru_cache` whose keys carry
# `self`. A gateway that rebuilt its embedder kept every previous
# SentenceTransformer on MPS, with no listing naming it and no eject reaching
# it. Registration is weak, the cache is per instance, and `unload()` is the
# eject. `resident_embedding_models()` / `eject_embedding_models()` are the
# process-wide truth the memory report and the runtime fold in.
# ---------------------------------------------------------------------------
_EMBEDDING_MANAGERS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _atexit_save_caches(ref: "weakref.ReferenceType") -> None:
    manager = ref()
    if manager is None:
        return
    try:
        manager._safe_save_persistent_cache()
    except Exception:
        pass
    try:
        manager._safe_save_normalized_cache()
    except Exception:
        pass


def registered_embedding_managers() -> List["EmbeddingManager"]:
    return [m for m in list(_EMBEDDING_MANAGERS)]


def resident_embedding_models() -> List[Dict[str, Any]]:
    """Every in-process embedding model whose weights are alive, one row per
    model id (holders = managers, each its own copy), in the residency row
    shape shared with `mlx_residency` / `hf_residency`."""
    by_model: Dict[str, Dict[str, Any]] = {}
    for manager in registered_embedding_managers():
        try:
            claim = manager.get_residency()
        except Exception:
            continue
        if not claim.get("loaded"):
            continue
        name = str(claim.get("model") or "")
        row = by_model.get(name)
        if row is None:
            row = by_model[name] = {
                "lane": "embeddings",
                "backend": "embeddings",
                "model_path": claim.get("model_path"),
                "models": [name],
                "holders": 0,
                "holder_rows": [],
                "weights_bytes": None,
                "cache_bytes": 0,
                "held_bytes": 0,
                "weights_alive": True,
                "shared_weights": False,
                "copies": 0,
                "device": claim.get("device"),
            }
        row["holders"] += 1
        row["copies"] += 1
        row["holder_rows"].append({"id": id(manager), "model": name, "type": type(manager).__name__,
                                   "instance_loaded": True, "device": claim.get("device"),
                                   "weights_bytes": claim.get("weights_bytes")})
        if isinstance(claim.get("weights_bytes"), int):
            row["weights_bytes"] = int(row["weights_bytes"] or 0) + int(claim["weights_bytes"])
        row["held_bytes"] = int(row["weights_bytes"] or 0)
    return list(by_model.values())


def eject_embedding_models(model: Optional[str] = None, *, reason: str = "eject") -> Dict[str, Any]:
    """Unload EVERY manager holding `model` (None: all), collect, and return
    torch's MPS pool. `ok` is True only when nothing of that model remains."""
    from ..providers.hf_residency import release_torch_mps_cache, torch_mps_stats

    model_s = str(model or "").strip().lower()
    before = torch_mps_stats()
    unloaded: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for manager in registered_embedding_managers():
        try:
            claim = manager.get_residency()
        except Exception:
            continue
        if not claim.get("loaded"):
            continue
        name = str(claim.get("model") or "")
        if model_s and name.strip().lower() != model_s:
            continue
        row = {"id": id(manager), "model": name, "type": type(manager).__name__}
        try:
            result = manager.unload()
            if result.get("in_flight"):
                row["error"] = str(result.get("reason"))
                errors.append(row)
                continue
            unloaded.append(row)
        except Exception as exc:  # noqa: BLE001 - one refusing holder must not hide the others
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(row)
    collected = gc.collect()
    released = release_torch_mps_cache()
    after = torch_mps_stats()
    remaining = [r for r in resident_embedding_models() if not model_s or str(r["models"][0]).strip().lower() == model_s]
    residual = remaining[0] if remaining else None
    return {
        "ok": residual is None and not errors,
        "reason": reason,
        "model": model,
        "holders_found": len(unloaded) + len(errors),
        "holders_unloaded": unloaded,
        "holders_refused": errors,
        "gc_collected": collected,
        "cache_cleared": released,
        "before": before,
        "after": after,
        "residual": residual,
        "ts": time.time(),
    }


def embeddings_memory_report() -> Dict[str, Any]:
    rows = resident_embedding_models()
    return {
        "backend": "embeddings",
        "held_bytes": int(sum(int(r.get("held_bytes") or 0) for r in rows)),
        "models": rows,
        "holders": int(sum(int(r.get("holders") or 0) for r in rows)),
        "resident_models": len(rows),
    }


class EmbeddingManager:
    """
    Production-ready embedding manager with multi-provider support and efficient serving.

    Supported Providers:
    - HuggingFace (default): Local sentence-transformers models with ONNX acceleration
    - Ollama: Local embedding models via Ollama API
    - LMStudio: Local embedding models via LMStudio API
    - OpenAI: Cloud embedding models via OpenAI API (text-embedding-3-small, etc.)
    - OpenRouter: Cloud embedding models via OpenRouter gateway
    - Portkey: Cloud embedding models via Portkey AI gateway
    - OpenAI-compatible: Any OpenAI-compatible embedding endpoint

    Features:
    - Multi-provider support (HuggingFace, Ollama, LMStudio, OpenAI, OpenRouter, Portkey, OpenAI-compatible)
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
        provider: str = None,
        backend: Union[str, EmbeddingBackend] = "auto",
        cache_dir: Optional[Path] = None,
        cache_size: int = 1000,
        output_dims: Optional[int] = None,
        trust_remote_code: bool = False,
        strict: Optional[bool] = None,
        provider_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the embedding manager.

        Args:
            model: Model identifier (HuggingFace model ID for HF provider, model name for others).
                  If None, uses configured default from AbstractCore config system.
            provider: Embedding provider ('huggingface', 'ollama', 'lmstudio', 'openai',
                     'openrouter', 'portkey', 'openai-compatible').
                     If None, uses configured default from AbstractCore config system.
            backend: Inference backend for HuggingFace ('auto', 'pytorch', 'onnx', 'openvino')
            cache_dir: Directory for persistent cache. Defaults to ~/.abstractcore/embeddings
            cache_size: Maximum number of embeddings to cache in memory
            output_dims: Output dimensions for Matryoshka truncation (if supported by provider)
            trust_remote_code: Whether to trust remote code (HuggingFace only)
            strict: If true, raise on provider/model failures instead of returning zero vectors.
            provider_kwargs: Optional provider-constructor kwargs for server-backed providers
                (for example ``api_key`` or ``base_url``).
        """
        if strict is None:
            strict_raw = str(os.environ.get("ABSTRACTCORE_EMBEDDINGS_STRICT", "") or "").strip().lower()
            strict = strict_raw in {"1", "true", "yes", "on"}
        self.strict = bool(strict)

        # Load configuration defaults, but ONLY if parameters weren't explicitly provided
        self._load_config_defaults(model, provider)

        # Store provider (after config loading)
        self.provider = self._resolved_provider.lower()
        self.provider_kwargs = dict(provider_kwargs or {})
        if self._resolved_base_url and "base_url" not in self.provider_kwargs:
            self.provider_kwargs["base_url"] = self._resolved_base_url

        # Validate provider
        supported_providers = list_available_providers()
        if self.provider not in supported_providers and not self.provider.startswith("endpoint:"):
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {', '.join(supported_providers)}"
            )

        # Initialize provider-specific attributes
        self.model_config = None
        self._provider_instance = None

        # Set up model identifier
        if self.provider == "huggingface":
            # Model configuration - HuggingFace only
            # Use resolved model (which includes config defaults if not explicitly provided)
            resolved_model = self._resolved_model

            # Handle model aliases from our favored models config
            if resolved_model in list_available_models():
                self.model_config = get_model_config(resolved_model)
                self.model_id = self.model_config.model_id
            else:
                # Direct HuggingFace model ID
                self.model_id = resolved_model
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
            # Server-based provider (Ollama, LMStudio, OpenAI)
            if self._resolved_model is None:
                raise ValueError(f"Model name is required for {self.provider} provider")

            self.model_id = self._resolved_model
            self.backend = None
            self.trust_remote_code = False

            # Create provider instance for delegation
            if self.provider == "ollama":
                from ..providers.ollama_provider import OllamaProvider
                self._provider_instance = OllamaProvider(model=self._resolved_model, **self.provider_kwargs)
                logger.info(f"Initialized Ollama embedding provider with model: {self._resolved_model}")
            elif self.provider == "lmstudio":
                from ..providers.lmstudio_provider import LMStudioProvider
                lmstudio_kwargs = dict(self.provider_kwargs)
                lmstudio_kwargs.setdefault("validate_model", False)
                self._provider_instance = LMStudioProvider(model=self._resolved_model, **lmstudio_kwargs)
                logger.info(f"Initialized LMStudio embedding provider with model: {self._resolved_model}")
            elif self.provider == "openai":
                from ..providers.openai_provider import OpenAIProvider
                self._provider_instance = OpenAIProvider(model=self._resolved_model, **self.provider_kwargs)
                logger.info(f"Initialized OpenAI embedding provider with model: {self._resolved_model}")
            elif self.provider == "openrouter":
                from ..providers.openrouter_provider import OpenRouterProvider
                openrouter_kwargs = dict(self.provider_kwargs)
                openrouter_kwargs.setdefault("validate_model", False)
                self._provider_instance = OpenRouterProvider(model=self._resolved_model, **openrouter_kwargs)
                logger.info(f"Initialized OpenRouter embedding provider with model: {self._resolved_model}")
            elif self.provider == "portkey":
                from ..providers.portkey_provider import PortkeyProvider
                self._provider_instance = PortkeyProvider(model=self._resolved_model, **self.provider_kwargs)
                logger.info(f"Initialized Portkey embedding provider with model: {self._resolved_model}")
            elif self.provider == "openai-compatible":
                from ..providers.openai_compatible_provider import OpenAICompatibleProvider
                compatible_kwargs = dict(self.provider_kwargs)
                compatible_kwargs.setdefault("validate_model", False)
                self._provider_instance = OpenAICompatibleProvider(model=self._resolved_model, **compatible_kwargs)
                logger.info(f"Initialized OpenAI-compatible embedding provider with model: {self._resolved_model}")
            elif self.provider == "vllm":
                from ..providers.vllm_provider import VLLMProvider
                vllm_kwargs = dict(self.provider_kwargs)
                vllm_kwargs.setdefault("validate_model", False)
                self._provider_instance = VLLMProvider(model=self._resolved_model, **vllm_kwargs)
                logger.info(f"Initialized vLLM embedding provider with model: {self._resolved_model}")
            elif self.provider.startswith("endpoint:"):
                from ..providers.registry import create_provider

                endpoint_kwargs = dict(self.provider_kwargs)
                endpoint_kwargs.setdefault("validate_model", False)
                self._provider_instance = create_provider(self._resolved_provider, model=self._resolved_model, **endpoint_kwargs)
                logger.info(
                    f"Initialized endpoint embedding provider {self._resolved_provider} "
                    f"with model: {self._resolved_model}"
                )

        # Common setup for all providers
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".abstractcore" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Register-at-first-write (machine-level data registry, best-effort).
        # ONLY the machine-level default dir registers: a constructor-custom
        # cache dir lives inside its CALLER's data home (e.g. the gateway's
        # data_dir) and rides that container's registry row — self-registering
        # it would both spam rows for ephemeral dirs (live incident 2026-07-13:
        # 372 pytest tmp rows) and collide with the container's own
        # registration under the nesting guard.
        from ..utils.data_registry import ensure_data_home_registered, ensure_core_data_homes
        _default_cache_dir = Path.home() / ".abstractcore" / "embeddings"
        if self.cache_dir.resolve() == _default_cache_dir.resolve():
            ensure_data_home_registered(
                "abstractcore-embeddings-cache",
                path=str(self.cache_dir),
                kind="artifacts",
                owner="abstractcore",
                safe_to_purge=True,
                description=(
                    "Embedding vector cache (pickled, keyed by text hash). Safe to purge: "
                    "vectors recompute on demand."
                ),
            )
        if self.provider == "huggingface":
            # Local embedding models download into the HF hub cache.
            ensure_core_data_homes()
        self.cache_size = cache_size
        self.output_dims = output_dims

        # Initialize model (HuggingFace only)
        self.model = None
        # Serializes the lazy reload after `unload()` and counts the encodes in
        # flight, so an unload waits for them (or refuses) instead of pulling
        # the model from under a running call (`_local_model_in_use`).
        self._load_lock = threading.RLock()
        self._model_cv = threading.Condition(self._load_lock)
        self._inflight_encodes = 0
        if self.provider == "huggingface":
            self._load_model()

        # Set up persistent cache (include provider in cache name for isolation)
        provider_cache_key = self.provider.replace("/", "_").replace("-", "_").replace(":", "_")
        cache_name = f"{provider_cache_key}_{self.model_id.replace('/', '_').replace('-', '_').replace(':', '_')}"
        if self.output_dims:
            cache_name += f"_dim{self.output_dims}"
        self.cache_file = self.cache_dir / f"{cache_name}_cache.pkl"
        self._persistent_cache = self._load_persistent_cache()

        # Normalized embeddings cache for performance optimization
        self.normalized_cache_file = self.cache_dir / f"{cache_name}_normalized_cache.pkl"
        self._normalized_cache = self._load_normalized_cache()

        # Served-model label truth (rogue-embedder-label defense, incident
        # 2026-07-11): the OpenAI-compatible /v1/embeddings response carries a
        # `model` field naming what the server ACTUALLY served — the only
        # server-side label truth in the stack. We record the last-seen served
        # label and cross-check it against the requested model_id, warning ONCE
        # per distinct mismatch. Warn-only by design: this is the SIGNAL layer
        # beneath the store's embedding_pin, which stays THE authority (a home
        # refuses on pin mismatch); a served-label difference is often just
        # formatting variance, so it never refuses on its own. HuggingFace-local
        # has no server label, so this only engages on the served route.
        self.served_model: Optional[str] = None
        self._served_model_mismatch_warned: set = set()

        # Save the caches at interpreter shutdown -- through a WEAK reference.
        # `atexit.register(self._method)` pinned the manager (and its
        # SentenceTransformer on MPS) for the life of the process.
        atexit.register(_atexit_save_caches, weakref.ref(self))
        # Per-instance memo (a class-level `@lru_cache` keyed on `self` was
        # the other pin); `self.embed.cache_info()` / `.cache_clear()` keep working.
        self.embed = lru_cache(maxsize=1000)(self._embed_uncached)
        _EMBEDDING_MANAGERS.add(self)

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

    def _load_config_defaults(self, model: Optional[str], provider: Optional[str]) -> None:
        """Load configuration defaults, but ONLY for parameters not explicitly provided.

        This ensures that direct parameters always take precedence over config defaults.
        """
        try:
            # Import config manager - use lazy import to avoid circular dependencies
            from ..config import get_config_manager
            config_manager = get_config_manager()
            embeddings_config = config_manager.config.embeddings
            embedding_route = config_manager.get_capability_default("embedding", "text")
            route_provider = embedding_route.get("provider") if isinstance(embedding_route, dict) else None
            route_model = embedding_route.get("model") if isinstance(embedding_route, dict) else None
            route_base_url = embedding_route.get("base_url") if isinstance(embedding_route, dict) else None

            # Apply defaults ONLY if not explicitly provided
            if provider is None:
                self._resolved_provider = route_provider or embeddings_config.provider or "huggingface"
                logger.debug(f"Using configured default provider: {self._resolved_provider}")
            else:
                self._resolved_provider = provider
                logger.debug(f"Using explicit provider parameter: {self._resolved_provider}")

            if model is None:
                # Use config default, or fallback to EmbeddingManager's original default
                self._resolved_model = route_model or embeddings_config.model or "all-minilm-l6-v2"
                logger.debug(f"Using configured default model: {self._resolved_model}")
            else:
                self._resolved_model = model
                logger.debug(f"Using explicit model parameter: {self._resolved_model}")

            resolved_provider_key = str(self._resolved_provider or "").strip().lower()
            resolved_model_key = str(self._resolved_model or "").strip()
            route_matches = (
                bool(route_base_url)
                and str(route_provider or "").strip().lower() == resolved_provider_key
                and str(route_model or "").strip() == resolved_model_key
            )
            embeddings_config_matches = (
                bool(embeddings_config.base_url)
                and str(embeddings_config.provider or "").strip().lower() == resolved_provider_key
                and str(embeddings_config.model or "").strip() == resolved_model_key
            )
            if route_matches:
                self._resolved_base_url = route_base_url
            elif embeddings_config_matches:
                self._resolved_base_url = embeddings_config.base_url
            else:
                self._resolved_base_url = None
        except Exception as e:
            # Fallback to hardcoded defaults if config system fails
            logger.debug(f"Config system unavailable, using fallback defaults: {e}")
            self._resolved_provider = provider or "huggingface"
            self._resolved_model = model or "all-minilm-l6-v2"
            self._resolved_base_url = None

    def _load_model(self):
        """Load the HuggingFace embedding model with optimal backend and reduced warnings."""
        try:
            if sentence_transformers is None:
                raise ImportError(
                    "sentence-transformers is required but not installed. "
                    "Install with: pip install \"abstractcore[embeddings]\""
                )

            # No process-wide environment writes (mission EE): this used to
            # `os.environ.setdefault` HF_HOME, TRANSFORMERS_CACHE and
            # HF_DATASETS_CACHE (the last one to the wrong layout) for the
            # WHOLE process and every child it spawns. Where to read or write
            # is now a per-call argument: the resolved snapshot directory, or
            # `cache_folder=` on a load that may download.

            # Offline-first is enforced HERE, per call (mission V): resolve the
            # cached snapshot directory and pass `local_files_only=True`, or
            # fail with a plain "download it first". This load used to stay
            # offline only when the HF provider had already written
            # HF_HUB_OFFLINE=1 into the process; that write is gone.
            source, load_kwargs = self._sentence_transformers_source()

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
                            source,
                            backend="onnx",
                            model_kwargs=model_kwargs,
                            trust_remote_code=self.trust_remote_code,
                            **load_kwargs,
                        )
                        onnx_model = optimal_onnx or "model.onnx"
                        logger.info(f"Loaded {self.model_id} with ONNX backend ({onnx_model})")

                    except Exception as e:
                        logger.warning(f"Optimized ONNX failed: {e}. Trying basic ONNX.")
                        try:
                            # Fallback to basic ONNX
                            self.model = sentence_transformers.SentenceTransformer(
                                source,
                                backend="onnx",
                                trust_remote_code=self.trust_remote_code,
                                **load_kwargs,
                            )
                            logger.info(f"Loaded {self.model_id} with basic ONNX backend")
                        except Exception as e2:
                            logger.warning(f"All ONNX variants failed: {e2}. Falling back to PyTorch.")
                            self.model = sentence_transformers.SentenceTransformer(
                                source,
                                trust_remote_code=self.trust_remote_code,
                                **load_kwargs,
                            )
                            logger.info(f"Loaded {self.model_id} with PyTorch backend")
                else:
                    self.model = sentence_transformers.SentenceTransformer(
                        source,
                        trust_remote_code=self.trust_remote_code,
                        **load_kwargs,
                    )
                    logger.info(f"Loaded {self.model_id} with PyTorch backend")

        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding functionality. "
                "Install with: pip install \"abstractcore[embeddings]\" (recommended) "
                "or: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_id}: {e}")
            raise

    def _ensure_local_model(self) -> Any:
        """Reload the in-process model after `unload()`: the next embedding
        after an eject loads the weights again, transparently. Returns the
        model. Server-backed providers hold nothing here (returns None)."""
        if self.provider != "huggingface":
            return None
        with self._load_lock:
            if self.model is None:
                logger.info(f"embedding model {self.model_id} was unloaded; loading it again for this request")
                self._load_model()
            return self.model

    @contextmanager
    def _local_model_in_use(self):
        """The loaded model, counted as IN USE for the duration of the block:
        `unload()` waits for the count to reach 0 before dropping the model,
        so a running encode never loses its model mid-call (and a call that
        starts after the unload reloads it)."""
        with self._model_cv:
            model = self._ensure_local_model()
            self._inflight_encodes += 1
        try:
            yield model
        finally:
            with self._model_cv:
                self._inflight_encodes -= 1
                self._model_cv.notify_all()

    def _sentence_transformers_source(self) -> "tuple[str, Dict[str, Any]]":
        """`(name_or_path, extra_kwargs)` for `SentenceTransformer(...)`.

        Local-only (`offline.offline_first`, or `force_local_files_only`, both
        default on -- the same rule as the HF provider): the model id resolves
        to its cached snapshot DIRECTORY (`resolve_hf_load_snapshot`: refs/main,
        else the newest usable snapshot), loaded with `local_files_only=True`,
        so nothing asks the Hub -- not even for a `refs/main` a pinned download
        may lack. A bare legacy name (`all-MiniLM-L6-v2`) is also looked up as
        `sentence-transformers/<name>`, as sentence-transformers itself does.
        Not cached -> ModelNotFoundError with the plain "download it first".
        Otherwise the id is passed through and may download on first use.
        """
        try:
            from ..config import get_config_manager

            cfg = get_config_manager()
            local_only = bool(cfg.is_offline_first() or cfg.should_force_local_files_only())
        except Exception:
            local_only = True
        name = str(self.model_id or "").strip()
        if not local_only:
            # May download on first use: say WHERE per call (the hub cache
            # huggingface_hub resolves), never through os.environ.
            return name, {"cache_folder": _hf_download_cache_folder()}
        candidate = Path(name).expanduser()
        if candidate.is_dir():
            return str(candidate), {"local_files_only": True}

        from ..utils.model_cache import hf_hub_cache_dirs, resolve_hf_load_snapshot

        cache_dirs = hf_hub_cache_dirs()
        names = [name] if "/" in name else [f"sentence-transformers/{name}"]
        for repo_id in names:
            snapshot = resolve_hf_load_snapshot(repo_id, cache_dirs=cache_dirs)
            if snapshot is not None:
                logger.debug(f"Embedding model {name} resolved to cached snapshot {snapshot}")
                return str(snapshot), {"local_files_only": True}

        from ..config.manager import hf_download_first_hint
        from ..exceptions import ModelNotFoundError

        looked_in = ", ".join(str(d) for d in cache_dirs) or "no cache directory found"
        raise ModelNotFoundError(
            f"Embedding model {names[0]!r} is not in the local Hugging Face cache (looked in: {looked_in}); "
            + hf_download_first_hint(names[0])
            + "."
        )

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
            from ..utils.model_cache import hf_hub_cache_dirs

            # Convert model ID to cache directory format (org--model)
            cache_dir_name = f"models--{self.model_id.replace('/', '--')}"
            onnx_patterns = ["model.onnx", "onnx/model.onnx", "onnx/model_O*.onnx"]

            # Every hub cache a load reads (HF_HUB_CACHE / HF_HOME / the
            # configured cache dir), never only `~/.cache/huggingface/hub`.
            for hf_cache_dir in hf_hub_cache_dirs():
                model_cache_dir = hf_cache_dir / cache_dir_name
                if not model_cache_dir.exists():
                    continue
                for snapshot_dir in model_cache_dir.glob("snapshots/*"):
                    if snapshot_dir.is_dir():
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

    def _note_added(self, which: str, key: str) -> None:
        """Record that THIS process added `key` to cache `which` (so a save has work to do)."""
        self.__dict__.setdefault(f"_{which}_added", set()).add(key)

    def _save_persistent_cache(self):
        """Save persistent cache to disk (merge-on-save, atomic, never shrinks the file)."""
        try:
            # Check if cache file attributes exist (may not if initialization failed)
            if not hasattr(self, 'cache_file') or not hasattr(self, '_persistent_cache'):
                return
            _merge_save_pickle_cache(
                self.cache_file,
                self._persistent_cache,
                self.__dict__.setdefault("_persistent_added", set()),
                label="embeddings",
            )
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
        """Save normalized embeddings cache to disk (merge-on-save, atomic, never shrinks the file)."""
        try:
            # Check if cache file attributes exist (may not if initialization failed)
            if not hasattr(self, 'normalized_cache_file') or not hasattr(self, '_normalized_cache'):
                return
            _merge_save_pickle_cache(
                self.normalized_cache_file,
                self._normalized_cache,
                self.__dict__.setdefault("_normalized_added", set()),
                label="normalized embeddings",
            )
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

    def _record_served_model(self, result: Any) -> None:
        """Record the server-reported model label from an OpenAI-compatible
        embeddings response and warn (once per distinct mismatch) when it does
        not match the requested model_id.

        Warn-only wire hygiene (the SIGNAL layer): a mismatch here is logged,
        never raised — the store's embedding_pin is the enforcement authority.
        Comparison is case/whitespace-insensitive and tolerant of the common
        `provider/model`, `mlx-community/model`, and `:tag` decorations so we
        only warn on a GENUINE model divergence, not label formatting."""
        try:
            served = result.get("model") if isinstance(result, dict) else None
        except Exception:
            served = None
        served = str(served or "").strip()
        if not served:
            return
        self.served_model = served
        requested = str(self.model_id or "").strip()
        if not requested:
            return

        def _norm_label(label: str) -> str:
            s = label.strip().lower()
            s = s.rsplit("/", 1)[-1]      # drop org/ or provider/ prefix
            s = s.split(":", 1)[0]        # drop :tag / :quant suffix
            return s

        if _norm_label(served) == _norm_label(requested):
            return
        key = (requested, served)
        if key in self._served_model_mismatch_warned:
            return
        self._served_model_mismatch_warned.add(key)
        logger.warning(
            f"#FALLBACK embedding served-model mismatch: requested {requested!r} "
            f"but the server reported serving {served!r}. The vectors were "
            f"produced by the SERVED model; verify the embedding pin/config. "
            f"(warn-only — the store's embedding_pin is the authority)"
        )

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
            self._note_added("normalized", text_hash)

            # Periodically save normalized cache
            if len(self._normalized_cache) % 10 == 0:
                self._save_normalized_cache()

            return normalized_embedding

        except Exception as e:
            logger.error(f"Failed to compute normalized embedding: {e}")
            # Fallback to regular embedding
            return self.embed(text)

    def _embed_uncached(self, text: str) -> List[float]:
        """Embed a single text with caching and optimization (memoized per
        instance as `self.embed`, see `__init__`).

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
                with self._local_model_in_use() as model:
                    embedding = model.encode(
                        text,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    ).tolist()

                # Apply Matryoshka truncation if specified
                if self.output_dims and len(embedding) > self.output_dims:
                    embedding = embedding[:self.output_dims]

            else:
                # Ollama or LMStudio: Delegate to provider
                provider_kwargs = {}
                if self.output_dims:
                    provider_kwargs["dimensions"] = self.output_dims
                result = self._provider_instance.embed(input_text=text, **provider_kwargs)
                self._record_served_model(result)

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
            self._note_added("persistent", text_hash)

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
            if self.strict:
                raise
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
                    with self._local_model_in_use() as model:
                        batch_embeddings = model.encode(
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
                        self._note_added("persistent", text_hash)
                        cached_embeddings[idx] = embedding_list

                    logger.debug(f"Generated {len(batch_embeddings)} embeddings in batch (HuggingFace)")

                else:
                    # Ollama or LMStudio: Delegate to provider (supports batch in single call)
                    provider_kwargs = {}
                    if self.output_dims:
                        provider_kwargs["dimensions"] = self.output_dims
                    result = self._provider_instance.embed(input_text=uncached_texts, **provider_kwargs)
                    self._record_served_model(result)

                    # Extract embeddings from OpenAI-compatible response
                    if "data" in result:
                        for text, embedding_data, idx in zip(uncached_texts, result["data"], uncached_indices):
                            embedding = embedding_data["embedding"]

                            # Apply dimension truncation if specified
                            if self.output_dims and len(embedding) > self.output_dims:
                                embedding = embedding[:self.output_dims]

                            text_hash = self._text_hash(text)
                            self._persistent_cache[text_hash] = embedding
                            self._note_added("persistent", text_hash)
                            cached_embeddings[idx] = embedding

                        logger.debug(f"Generated {len(result['data'])} embeddings in batch ({self.provider})")
                    else:
                        raise ValueError(f"Invalid batch response from {self.provider} provider")

            except Exception as e:
                logger.error(f"Failed to embed batch with {self.provider}: {e}")
                if self.strict:
                    raise
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
            with self._local_model_in_use() as model:
                return model.get_sentence_embedding_dimension()
        else:
            # For Ollama/LMStudio, we need to generate a test embedding to get dimension
            # This is cached, so it's only done once
            test_embedding = self.embed("test")
            return len(test_embedding)

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text for embedding usage calculations.

        This provides accurate estimation using centralized token utilities,
        suitable for usage tracking and billing/quota calculations.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        from ..utils.token_utils import TokenUtils
        # Use the model name if available for more accurate estimation
        model_name = getattr(self, 'model_name', None)
        return TokenUtils.estimate_tokens(text, model_name)

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
            # Server-reported served-model label (served route only; None until
            # the first served embed, and on the HuggingFace-local path). The
            # door's creation probe can record this in the pin's provenance
            # trail so a rogue label is one lookup, not an evening of forensics.
            "served_model": self.served_model,
            "backend": self.backend.value if self.backend else self.provider,
            "cache_file": str(self.cache_file),
            "normalized_cache_file": str(self.normalized_cache_file),
            "output_dims": self.output_dims
        }

    def get_residency(self) -> Dict[str, Any]:
        """Core-owned in-process residency truth for this manager: whether the
        embedding weights are alive, on which device, and their bytes."""
        model = getattr(self, "model", None)
        loaded = model is not None
        claim: Dict[str, Any] = {
            "task": "embedding",
            "provider": str(self.provider),
            "model": str(self.model_id),
            "loaded": loaded,
            "state": "loaded" if loaded else "not_loaded",
            "source": "abstractcore.embeddings",
            "device": None,
            "weights_bytes": None,
            "model_path": None,
        }
        if not loaded:
            return claim
        try:
            from ..providers.hf_residency import module_bytes

            claim["weights_bytes"] = module_bytes(model)
        except Exception:
            pass
        try:
            claim["device"] = str(getattr(model, "device", None) or next(model.parameters()).device)
        except Exception:
            pass
        try:
            claim["model_path"] = str(getattr(model, "model_name_or_path", None) or "") or None
        except Exception:
            pass
        return claim

    def unload(self, *, drain_timeout_s: float = 30.0) -> Dict[str, Any]:
        """Free the in-process embedding model: drop the SentenceTransformer,
        clear the memo, save the persistent caches, collect, and return torch's
        MPS pool to the OS. Idempotent. The manager stays usable: the next
        embedding loads the model again (`_ensure_local_model`).

        Server-backed providers (Ollama, LM Studio, OpenAI-compatible...) hold
        no weights in this process: their client is kept, nothing is dropped,
        and the report says so (`in_process: False`).

        Embeddings running on this manager are waited for (up to
        `drain_timeout_s`); if one is still running then, nothing is freed and
        the report says so (`in_flight`), rather than blaming an unknown
        reference."""
        from ..providers.hf_residency import release_torch_mps_cache, torch_mps_stats

        if self.provider != "huggingface":
            return {
                "unloaded": False,
                "in_process": False,
                "model": str(self.model_id),
                "provider": str(self.provider),
                "loaded": False,
                "freed_weights_bytes": 0,
                "residual_weights_alive": False,
                "residual_weights_bytes": 0,
                "reason": f"{self.provider} serves this embedding model from its own server; nothing is held in this process",
            }

        before = self.get_residency()
        try:
            self._safe_save_persistent_cache()
            self._safe_save_normalized_cache()
        except Exception:
            pass
        weights_ref = None
        with self._model_cv:
            drained = self._model_cv.wait_for(lambda: self._inflight_encodes == 0, timeout=drain_timeout_s)
            if not drained:
                running = int(self._inflight_encodes)
                return {
                    "unloaded": False,
                    "in_process": True,
                    "model": str(self.model_id),
                    "provider": str(self.provider),
                    "loaded": self.model is not None,
                    "in_flight": running,
                    "freed_weights_bytes": 0,
                    "residual_weights_alive": True,
                    "residual_weights_bytes": before.get("weights_bytes"),
                    "reason": (f"{running} embedding call(s) still running on {self.model_id} after "
                               f"{drain_timeout_s}s; nothing was freed; retry the unload when they end"),
                }
            try:
                weights_ref = weakref.ref(self.model) if self.model is not None else None
            except TypeError:
                weights_ref = None
            self.model = None
        try:
            self.embed.cache_clear()
        except Exception:
            pass
        gc.collect()
        released = release_torch_mps_cache()
        after = self.get_residency()
        # An eject never claims success over retained memory: when another
        # reference keeps the SentenceTransformer alive, say so.
        residual_alive = bool(weights_ref is not None and weights_ref() is not None)
        report = {
            "unloaded": bool(before.get("loaded")) and not residual_alive,
            "in_process": True,
            "model": str(self.model_id),
            "provider": str(self.provider),
            "loaded": bool(after.get("loaded")),
            "freed_weights_bytes": None if residual_alive else before.get("weights_bytes"),
            "residual_weights_alive": residual_alive,
            "residual_weights_bytes": before.get("weights_bytes") if residual_alive else 0,
            "torch_mps": torch_mps_stats(),
            "cache_cleared": released,
        }
        if residual_alive:
            report["warnings"] = [
                f"the embedding weights of {self.model_id} are still referenced elsewhere in this process; "
                "nothing was freed for them"
            ]
        logger.info(
            f"embedding model {self.model_id} unloaded ({before.get('weights_bytes')} bytes of weights released, "
            f"mps pool released={released})"
        )
        return report

    def clear_cache(self):
        """Clear both memory and persistent caches."""
        self.embed.cache_clear()
        self._persistent_cache.clear()
        self._normalized_cache.clear()
        self.__dict__.setdefault("_persistent_added", set()).clear()
        self.__dict__.setdefault("_normalized_added", set()).clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.normalized_cache_file.exists():
            self.normalized_cache_file.unlink()
        logger.info("Cleared all embedding caches")

    def save_caches(self):
        """Explicitly save both caches to disk."""
        self._save_persistent_cache()
        self._save_normalized_cache()
