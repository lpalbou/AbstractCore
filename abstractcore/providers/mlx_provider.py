"""
MLX provider implementation for Apple Silicon.
"""

import copy
import json
import time
import uuid
import inspect
import os
import threading
import weakref
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple, Union, Iterator, Type, TYPE_CHECKING

try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

# Try to import Outlines (native structured output for MLX models)
try:
    import outlines

    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False

from .base import BaseProvider, PromptCacheRenderedFragment, ThinkingControlHandling
from .speculation import (
    SpeculationOutcome,
    SpeculationUnavailableError,
    capability_speculation,
    normalize_speculation_request,
    resolve_speculation_request,
    unavailable as speculation_unavailable,
)
from ..architectures.response_postprocessing import (
    TRUNCATED_REASONING_MARKER,
    normalize_assistant_text,
)
from ..core.types import GenerateResponse
from .base import PromptCacheStore
from .generation_progress import PREFILL_PROGRESS_KWARG
from ..exceptions import ProviderAPIError, ModelNotFoundError, format_model_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType

if TYPE_CHECKING:
    from ..media.types import MediaContent


class _SharedMLXModel:
    """ONE loaded text model, and the KV state that only means something with it.

    A provider INSTANCE is a cheap facade: generation defaults, a scoped config, a
    tracer. The expensive parts are the weights and the prompt caches built on them —
    and those used to be per instance. Measured 2026-09-17 on a live gateway: it builds
    one host (runtime → client → provider) per service and rebuilds all of them whenever
    a workflow is published, so one 15 GB model was resident FOUR times (physical
    footprint 113 GB, peak 129 GB on a 128 GB machine, ~85 GB compressed or swapped),
    each publish reloaded it once per service, and each rebuilt provider started with an
    empty prompt-cache store — every session's cache was orphaned by a workflow publish
    or by relaunching a desktop client. Second provider for the same model, measured:
    2.37 → 4.80 GB, `a.llm is b.llm` False, the first provider's cache key not found.

    Sharing the STATE (not the provider object) keeps what must stay per service — the
    runtime stamps per-principal config onto the provider — while a session's cache
    survives for as long as any provider of that model lives in the process.
    """

    def __init__(self, *, key: str, llm: Any, tokenizer: Any, prompt_cache_store: Any) -> None:
        self.key = key
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt_cache_store = prompt_cache_store
        self.hybrid_snapshots: Dict[str, Dict[str, Any]] = {}
        self.hybrid_snapshot_lock = threading.RLock()
        self.delta_feed_warned_keys: set = set()
        # Who is using it. Weak: a provider that is simply dropped (a rebuilt host's old
        # client) must not count as a user forever.
        self.holders: "weakref.WeakSet[Any]" = weakref.WeakSet()


# WEAK values: the providers hold the strong references. When the last provider of a
# model goes away the weights are freed exactly as before (`del llm` still returns the
# memory); while any provider lives, the next one for the same model adopts it.
_SHARED_MLX_MODELS: "weakref.WeakValueDictionary[str, _SharedMLXModel]" = weakref.WeakValueDictionary()
_SHARED_MLX_MODELS_LOCK = threading.RLock()


def _installed_mlx_versions() -> str:
    """`mlx 0.31.2, mlx-lm 0.31.3, mlx-vlm (not installed)` -- for error text.

    A stale MLX stack fails as a ModuleNotFoundError for a module the installed
    version simply does not have yet, so the version triple IS the diagnosis.
    """
    from importlib.metadata import PackageNotFoundError, version

    parts: List[str] = []
    for dist in ("mlx", "mlx-lm", "mlx-vlm"):
        try:
            parts.append(f"{dist} {version(dist)}")
        except PackageNotFoundError:
            parts.append(f"{dist} (not installed)")
        except Exception:
            parts.append(f"{dist} (unknown)")
    return ", ".join(parts)


def _mlx_model_sharing_enabled(kwargs: Optional[Dict[str, Any]] = None) -> bool:
    """On by default. `share_loaded_model=False` or ABSTRACTCORE_MLX_SHARE_MODELS=0 opts out."""
    explicit = (kwargs or {}).get("share_loaded_model")
    if isinstance(explicit, bool):
        return explicit
    raw = str(os.environ.get("ABSTRACTCORE_MLX_SHARE_MODELS", "") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class MLXProvider(BaseProvider):
    """MLX provider for Apple Silicon models with full integration"""

    def __init__(
        self,
        model: str = "mlx-community/Mistral-7B-Instruct-v0.1-4bit",
        structured_output_method: str = "auto",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.provider = "mlx"

        # Register-at-first-write: MLX model loads write into the HF hub cache.
        from ..utils.data_registry import ensure_core_data_homes

        ensure_core_data_homes()

        # Handle timeout parameter for local models
        self._handle_timeout_parameter(kwargs)

        # Structured output method: "auto", "native_outlines", "prompted"
        # auto: Use Outlines if available, otherwise prompted (default)
        # native_outlines: Force Outlines (error if unavailable)
        # prompted: Always use prompted fallback (fastest, still 100% success)
        self.structured_output_method = structured_output_method

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        self.llm = None
        self.tokenizer = None
        self._resolved_model_id: Optional[str] = None
        # Vision add-on state. Set at load from config.json; the encoder itself is
        # built lazily on the first image so a text-only session never imports
        # mlx-vlm. Initialised here so harnesses that build the provider via
        # __new__ (the prompt-cache unit tests) see the attributes.
        self._vision_addon = None
        self._vision_side: Dict[str, Any] = {}
        self._vision_addon_lock = threading.Lock()
        self._vision_usable: bool = False
        self._vision_reason: Optional[str] = None
        self._vision_info: Dict[str, Any] = {}
        # Native-MTP speculative decoding state. The MTP lane is a DIFFERENT
        # runtime, not a flag: mlx-lm strips `mtp.` weights on load and its own
        # speculative path refuses this architecture outright ("requires a
        # trimmable prompt cache"), so acceleration means loading the target
        # through mlx-vlm and pairing it with a separate drafter checkpoint.
        # Resolved once at load; None here means the lane is inert and every
        # generate call behaves exactly as it did before.
        self._speculation_request = normalize_speculation_request(
            kwargs.get("speculation")
        )
        self._speculation_inherits_config = kwargs.get("speculation") is None
        self._speculation_default_supported = False
        self._mtp_drafter = None
        self._mtp_kind: Optional[str] = None
        self._mtp_block_size: Optional[int] = None
        self._mtp_processor = None
        self._mtp_reason: Optional[str] = None
        self._mtp_drafter_id: Optional[str] = None
        # Whether this drafter is known NOT to reproduce the unaccelerated
        # output byte-for-byte. True/None = no known divergence. Surfaced in
        # response metadata so a caller pinning outputs can see it.
        self._mtp_output_preserving: Optional[bool] = None
        self._mtp_prompt_cache_warned: bool = False
        # Set per call by `_apply_per_call_speculation` when a request asks for
        # `off` on a provider whose lane is loaded -- the one per-call change
        # that IS honorable, since skipping the drafter needs no reload.
        self._mtp_call_disabled: bool = False
        self._mtp_call_outcome: Optional[SpeculationOutcome] = None
        # Whether the drafter ran on the most recent generate call. None = no
        # call yet. Read by surfaces that cannot see response metadata (the
        # streaming lane).
        self._mtp_last_used: Optional[bool] = None
        # Set when the lane declined at load time; carries the named reason into
        # every later response's metadata so the diagnosis is not one log line
        # the caller may never have seen.
        self._mtp_outcome_at_load: Optional[SpeculationOutcome] = None
        self._native_qwen4 = None
        self._native_session = None
        self._native_runtime = None
        self._native_owner_id = None
        self._mlx_batching = kwargs.get("mlx_batching", False)
        if not isinstance(self._mlx_batching, bool):
            raise ValueError("mlx_batching must be a bool")
        self._mlx_runtime_options = {
            "max_batch_size": kwargs.get("mlx_max_batch_size", 4),
            "max_queue_size": kwargs.get("mlx_max_queue_size", 32),
            "batch_wait_ms": kwargs.get("mlx_batch_wait_ms", 10),
            "queue_timeout_s": kwargs.get("mlx_queue_timeout_s", 120),
            "output_queue_size": kwargs.get("mlx_output_queue_size", 256),
        }
        # memory_max_gb: None = mlx-vlm's machine-relative budget. It is a cap
        # on each retained prefix snapshot, so it is only ever forwarded when
        # the operator chose a number (see NativeSession.prompt_cache).
        memory_max_gb = kwargs.get("mlx_cache_memory_max_gb")
        if memory_max_gb is not None:
            from .mlx_native_cache import _positive_gb
            memory_max_gb = _positive_gb(memory_max_gb, "mlx_cache_memory_max_gb")
        self._mlx_cache_options = {
            "disk_path": kwargs.get("mlx_cache_disk_path"),
            "disk_max_gb": kwargs.get("mlx_cache_disk_max_gb", 4.0),
            "memory_max_gb": memory_max_gb,
        }
        self._mlx_cache_scope = kwargs.get("mlx_cache_scope", "local")
        if not isinstance(self._mlx_cache_scope, str) or not self._mlx_cache_scope.strip():
            raise ValueError("mlx_cache_scope must be a non-empty string")
        if self._mlx_cache_options["disk_path"] is not None and not self._mlx_batching:
            raise ValueError("mlx_cache_disk_path requires mlx_batching=True")
        self._mtp_generation_lock = threading.Lock()
        self._native_cache_key = None
        self._mlx_ple_offload = kwargs.get("mlx_ple_offload", True)
        if not isinstance(self._mlx_ple_offload, bool):
            raise ValueError("mlx_ple_offload must be a bool")
        # Delta-feed bookkeeping (see _prepare_cache_delta_feed): keys that
        # already warned about unknown-composition warm caches, and the last
        # fragment fed by _prompt_cache_backend_append (consumed by
        # prompt_cache_update to extend the fed-token-id record).
        self._delta_feed_warned_keys: set = set()
        self._pending_append_fragment: Optional[str] = None
        # Exact token ids of the last append. Authoritative over the text stash:
        # a PLANNED bloc fragment (base.prompt_cache_plan_bloc_chain) is a token
        # slice of one rendered conversation and has no standalone text whose
        # re-encoding is guaranteed to reproduce it.
        self._pending_append_fragment_ids: Optional[List[int]] = None
        self._pending_append_precount: int = 0
        # Snapshot/restore lane for UNTRIMMABLE architectures (Gated-DeltaNet
        # hybrids: Qwen3.5/3.6/Ornith, and pure-SSM). A recurrent state cannot
        # be rewound (trim), but it CAN be copied: we keep one deepcopy snapshot
        # per key at the last prefill+reply boundary, keyed by the exact token
        # ids it holds, and restore it when the next full-context prompt extends
        # it — the same forward-only discipline llama.cpp's GGUF lane and
        # mlx_lm's own server (LRUPromptCache) use. Bounded to one snapshot per
        # key (the growing snapshot replaces its predecessor). Guarded because
        # capture stores a live cache object shared with no one else.
        self._hybrid_snapshots: Dict[str, Dict[str, Any]] = {}
        self._hybrid_snapshot_lock = threading.RLock()
        # The stash is instance-level shared state: without the lock, two
        # threads updating DIFFERENT keys could cross-pollinate their
        # fed-token-id records (adversarial find P2-9).
        self._append_stash_lock = threading.RLock()
        self._shared_model: Optional[_SharedMLXModel] = None
        self._share_loaded_model = _mlx_model_sharing_enabled(kwargs)
        self._load_model()

    def supports_concurrent_generation(self) -> bool:
        """The native scheduler, not ordinary thread submission, owns concurrency."""
        return getattr(self, "_native_runtime", None) is not None

    def supports_text_progress_events(self) -> bool:
        """True on every MLX lane: each one can observe the first sampled token.

        The native runtime reports prompt/cached/generation counters per
        snapshot; mlx-vlm and mlx-lm both expose `stream_generate`, whose first
        response with a token IS the end of prefill. See
        `providers/generation_progress.py` for the emitted contract.
        """
        return True

    def supports_generation_cancel(self) -> bool:
        """True on every MLX lane: each decodes token by token in-process.

        The native runtime checks the job's event per token (batched and
        exclusive loops); the mlx-lm / mlx-vlm lanes are driven through their
        `stream_generate` generator whenever an event is supplied, checked per
        sampled token and closed on cancel (mlx-lm additionally aborts prefill
        between chunks). Contract: `providers/generation_cancel.py`.
        """
        return True

    def _bind_native_session(self, session):
        """Bind one execution policy to shared weights, never competing workers."""
        import importlib.metadata
        from pathlib import Path
        self._native_session = session
        self._mtp_generation_lock = session.lock
        batching = getattr(self, "_mlx_batching", False)
        runtime_options = getattr(self, "_mlx_runtime_options", {})
        cache_options = getattr(self, "_mlx_cache_options", {})
        signature = json.dumps({"batching": batching,
                                "runtime": runtime_options,
                                "cache": cache_options}, sort_keys=True, default=str)
        with session.config_lock:
            if session.execution_config is not None and session.execution_config != signature:
                raise ProviderAPIError("Native MLX weights are already resident with different execution/cache settings; unload their providers before changing settings")
            session.execution_config = signature
            if batching:
                if session.runtime is None:
                    from .mlx_native_cache import NativeCacheStore
                    from .mlx_runtime import NativeRuntime
                    head_identity = None
                    if session.drafter is not None:
                        embedded = getattr(self, "_native_qwen4", None) is session
                        head_identity = {
                            "path": (str(Path(self._resolved_model_id).resolve()) + "#mtp") if embedded
                                    else session.drafter_path,
                            "weights_fingerprint": self.prompt_cache_weights_fingerprint() if embedded
                                                   else session.drafter_weights_fingerprint,
                        }
                        if cache_options.get("disk_path") is not None and not head_identity["weights_fingerprint"]:
                            raise ProviderAPIError("Native disk caching requires a verified drafter weights fingerprint")
                    identity = {
                        "model_path": str(Path(self._resolved_model_id).resolve()),
                        "weights_fingerprint": self.prompt_cache_weights_fingerprint(),
                        "model_config": self.prompt_cache_model_config_fingerprint(),
                        "tokenizer": self.prompt_cache_tokenizer_fingerprint(),
                        "mlx": importlib.metadata.version("mlx"),
                        "mlx_vlm": importlib.metadata.version("mlx-vlm"),
                        "head": head_identity,
                        "ple_offload": bool(getattr(session, "ple_offload", False)),
                    }
                    session.cache_store = NativeCacheStore(identity, **cache_options)
                    session.runtime = NativeRuntime(
                        session.model, session.processor, session.drafter,
                        getattr(session, "draft_kind", None) or self._mtp_kind,
                        cache_factory=session.cache_store.manager,
                        on_close=session.cache_store.close, **runtime_options)
                self._native_runtime = session.runtime
                self._native_owner_id = session.runtime.acquire()
            session.holders.add(self)
            from .mlx_native_session import release_native_owner
            self._native_finalizer = weakref.finalize(self, release_native_owner, session, getattr(self, "_native_owner_id", None))

    def _native_request_view(self, cancel_event=None):
        """Request-local facade over shared immutable defaults and runtime lease.

        Explicitly rebind adapter methods: copying a bound method otherwise
        silently routes execution back through the original mutable provider.
        """
        view = copy.copy(self)
        view._native_request_facade = True
        view._native_parent = self  # Keep the real model lease alive through lazy streams.
        view._vision_side = {}
        view._pending_append_fragment = None
        view._pending_append_fragment_ids = None
        view._pending_append_precount = 0
        view._last_output_budget_clamp = None
        view._native_runtime_metadata = {}
        view._native_runtime_stream = None
        # Per-CALL results: the terminal chunk's usage and `prompt_cache` record
        # read these, so a request never starts from another request's values
        # (the scheduler runs requests concurrently, each on its own view).
        view._mtp_last_result = None
        view._mtp_last_apc = None
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            from .mlx_runtime import NativeRuntimeError
            raise NativeRuntimeError("_cancel_event must be a threading.Event", code="invalid_request")
        view._native_cancel_event = cancel_event if cancel_event is not None else threading.Event()
        view.generate_fn = view._mtp_generate_fn
        view.stream_generate_fn = view._mtp_stream_generate_fn
        return view

    def _native_runtime_request(self, text, kwargs):
        import base64
        from pathlib import Path
        from .mlx_runtime import NativeRequest, NativeRuntimeError
        media = []
        for index, part in getattr(self, "_native_media", ()):
            path = getattr(part, "file_path", None)
            raw = Path(path).read_bytes() if path else part.content
            if isinstance(raw, str):
                raw = base64.b64decode(raw, validate=True)
            media.append((index, bytes(raw)))
        enabled = self._mtp_active and not getattr(self, "_mtp_call_disabled", False)
        try:
            # Validate raw values before coercion: bools and fractional counts
            # must not silently become different controls at the native seam.
            return NativeRequest(
                prompt=text, max_tokens=kwargs["max_tokens"],
                temperature=kwargs.get("temperature", 0),
                top_p=kwargs.get("top_p", 1), top_k=kwargs.get("top_k", 0),
                seed=kwargs.get("seed"),
                draft_tokens=(getattr(self, "_mtp_call_block_size", None) or 3) if enabled else 0,
                cache_key=getattr(self, "_native_cache_key", None),
                cache_scope=self._mlx_cache_scope, media=tuple(media),
                sampling=dict(getattr(self, "_native_sampling_kwargs", {})),
                stop=tuple(getattr(self, "_native_stop", ())), owner_id=self._native_owner_id,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise NativeRuntimeError(str(exc), code="invalid_request") from exc

    def _observe_native_runtime_result(self, result):
        self._mtp_last_result = result
        self._native_runtime_metadata = dict(result.metadata or {})
        spec = self._native_runtime_metadata.pop("speculation", {})
        self._mtp_last_used = bool(spec.get("used", False))
        self._mtp_call_stats = dict(spec.get("stats", {}))
        self._native_media_records = list(result.media_records or ())
        self._record_native_media()

    def _publish_native_request_status(self):
        parent = getattr(self, "_native_parent", None)
        if parent is not None:
            # A single immutable snapshot means concurrent responses never read
            # one another's mutable counters. This view is last-COMPLETED only.
            parent._native_last_completed_status = self.speculation_status()

    def _load_or_adopt_shared_model(self, key: str, loader: Any) -> Tuple[Any, Any]:
        """(llm, tokenizer) for `key` — loaded once per process, then adopted.

        Adoption also swaps this instance's prompt-cache store, hybrid snapshots and
        delta-feed bookkeeping for the shared ones: a KV cache is only reusable with the
        weights it was computed on, so the two travel together. The first provider's
        store (its size bound, TTL and eviction hook) becomes the shared one.
        """
        if not getattr(self, "_share_loaded_model", True):
            return loader()
        with _SHARED_MLX_MODELS_LOCK:  # also serializes two 15 GB loads of one model
            shared = _SHARED_MLX_MODELS.get(key)
            if shared is None:
                llm, tokenizer = loader()
                shared = _SharedMLXModel(
                    key=key, llm=llm, tokenizer=tokenizer, prompt_cache_store=self._prompt_cache_store
                )
                _SHARED_MLX_MODELS[key] = shared
            else:
                self.logger.info(
                    f"MLX model already resident in this process; adopting it and its prompt caches: {key}"
                )
            shared.holders.add(self)
            self._shared_model = shared
            self._prompt_cache_store = shared.prompt_cache_store
            self._hybrid_snapshots = shared.hybrid_snapshots
            self._hybrid_snapshot_lock = shared.hybrid_snapshot_lock
            self._delta_feed_warned_keys = shared.delta_feed_warned_keys
            return shared.llm, shared.tokenizer

    def _release_shared_model(self) -> bool:
        """Stop using the shared model. True when OTHER providers still use it — then
        its weights and caches are theirs and must not be cleared by this unload."""
        shared = getattr(self, "_shared_model", None)
        if shared is None:
            return False
        with _SHARED_MLX_MODELS_LOCK:
            shared.holders.discard(self)
            still_used = len(shared.holders) > 0
            self._shared_model = None
            if still_used:
                # Detach: this instance gets private, empty state so nothing it does
                # after unload can touch what the others are using.
                self._prompt_cache_store = PromptCacheStore(
                    max_entries=int(getattr(shared.prompt_cache_store, "_max_entries", 32) or 32),
                    on_evict=self._prompt_cache_store_evicted,
                )
                self._hybrid_snapshots = {}
                self._hybrid_snapshot_lock = threading.RLock()
                self._delta_feed_warned_keys = set()
            else:
                _SHARED_MLX_MODELS.pop(shared.key, None)
            return still_used

    def supports_prompt_cache(self) -> bool:
        """MLX supports KV prompt caches via `mlx_lm.models.cache`."""
        return True

    def get_prompt_cache_capabilities(self):
        if getattr(self, "_mtp_processor", None) is not None:
            from .base import PromptCacheCapabilities
            return PromptCacheCapabilities(
                supported=True, mode="keyed", supports_clear=True, supports_stats=True,
                notes=("Native prefix reuse requires complete messages; optional managed SSD tier when mlx_batching=True. "
                       "Clear invalidates all shared native entries, including clear(key).",),
            )
        return super().get_prompt_cache_capabilities()

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        """MLX KV caches are mutable and can serve as the context source-of-truth."""
        if getattr(self, "_mtp_processor", None) is not None:
            # Native VLM APC requires the full logical history; it is not an
            # append-only hidden conversation store.
            return False
        return True

    def prompt_cache_cache_backend(self) -> str:
        return "mlx"

    def prompt_cache_artifact_format(self) -> str:
        return "abstractcore-mlx-prompt-cache/v1"

    def prompt_cache_engine_fingerprint(self) -> str:
        """mlx_lm owns the KV cache layout — pin its version (0817). A version
        change can alter the safetensors cache serialization, so a reused
        artifact compiled under a different mlx_lm silently injects wrong KV."""
        try:
            import mlx_lm

            version = str(getattr(mlx_lm, "__version__", "") or "").strip()
        except Exception:
            version = ""
        return f"mlx_lm=={version}" if version else "mlx_lm==unknown"

    def prompt_cache_tokenizer_fingerprint(self) -> str:
        """Identity of the LOADED tokenizer's text→ids mapping (0817 axis 2).

        "" while no model is loaded — validation gates abstain on "" and the
        load-time gate re-checks once the tokenizer exists. Never loads the
        model just to fingerprint it.
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return ""
        from .tokenizer_fingerprint import tokenizer_fingerprint_for

        return tokenizer_fingerprint_for(tokenizer)

    def prompt_cache_model_config_fingerprint(self) -> str:
        """Identity of the LOADED model's KV geometry (0817 axis 3).

        Source, strongest first: the loaded model's `args` (mlx_lm ModelArgs —
        built FROM config.json, carries rope/window/position keys), else the
        resolved model directory's config.json. "" while no model is loaded —
        gates abstain on "" and the load-time gate re-checks. Never loads the
        model just to fingerprint it.
        """
        from .model_config_fingerprint import model_config_fingerprint_for

        model_obj = getattr(self, "llm", None)
        if model_obj is not None:
            args = getattr(model_obj, "args", None)
            fingerprint = model_config_fingerprint_for(args)
            if fingerprint:
                return fingerprint
        resolved = str(getattr(self, "_resolved_model_id", "") or "").strip()
        if resolved:
            try:
                from pathlib import Path

                cfg_path = Path(resolved) / "config.json"
                if cfg_path.is_file():
                    raw = cfg_path.read_text(encoding="utf-8", errors="ignore")
                    data = json.loads(raw) if raw.strip() else None
                    if isinstance(data, dict):
                        return model_config_fingerprint_for(data)
            except Exception:
                pass
        return ""

    def prompt_cache_weights_fingerprint(self) -> str:
        """Cheap identity of the LOADED weights (0817 axis 4).

        The resolved model directory names the identity: an HF-cache snapshot
        dir yields the hub commit sha (tier 1, content-addressed upstream);
        any other local dir yields the weight-file-set fingerprint (tier 2).
        "" while unresolved/unloaded — gates abstain and the load-time gate
        re-checks. Never touches full weight bytes.
        """
        resolved = str(getattr(self, "_resolved_model_id", "") or "").strip()
        if not resolved:
            return ""
        from .weights_fingerprint import weights_fingerprint_for_dir

        return weights_fingerprint_for_dir(resolved)

    def prompt_cache_render_fragment(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        prefilled_modules: Optional[List[str]] = None,
    ) -> Optional[PromptCacheRenderedFragment]:
        serialized = self._build_prompt_fragment(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            prefilled_modules=prefilled_modules,
        )
        if not serialized:
            return None
        msg_fmt = (
            str((getattr(self, "architecture_config", {}) or {}).get("message_format") or "")
            .strip()
            .lower()
        )
        model = str(getattr(self, "model", "") or "").strip().lower()
        if msg_fmt == "gemma_turn":
            fmt = "gemma-turn"
        else:
            fmt = "qwen-chatml" if "qwen" in model else "plain-chat"
        return PromptCacheRenderedFragment(
            serialized_prompt=str(serialized),
            serializer_version=f"mlx-prompt-fragment/v1:{fmt}",
            cache_backend="mlx",
            artifact_format=self.prompt_cache_artifact_format(),
            meta={"prompt_format": fmt},
        )

    def prompt_cache_render_bloc_text(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
    ) -> Optional[str]:
        """Exact `generate()` render, for the bloc planner (see base.py).

        `_build_prompt` IS `_build_prompt_fragment` with
        `add_generation_prompt=True`, so this is literally the same renderer the
        live prompt goes through — which is the only reason the planner's
        token-prefix guarantee is worth anything. `include_bos` stays at its
        `generate()` default (True): the planner always renders the CUMULATIVE
        union from position 0, so a BOS appears exactly once, at the head of the
        first bloc, and never inside a later delta.
        """
        if getattr(self, "tokenizer", None) is None:
            return None
        try:
            return self._build_prompt_fragment(
                prompt=str(prompt or ""),
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                add_generation_prompt=bool(add_generation_prompt),
            )
        except Exception:
            return None

    def prompt_cache_encode_bloc_text(self, text: str) -> Optional[List[int]]:
        return self._encode_prompt_token_ids(text)

    def _apply_provider_thinking_kwargs(self, *, enabled, level=None, kwargs: Dict[str, Any]):
        """Map unified thinking control into MLX prompt serialization state.

        mlx-lm local generation takes an already-serialized prompt. For Qwen reasoning
        templates, the robust local disable control is to serialize the assistant
        generation prompt in no-thinking mode (`<think>\n\n</think>\n\n`) before
        generation. Effort levels are enforced the same way the model's own chat
        template does it: the per-level instruction sentence declared in assets as
        `thinking_control.effort_system_lines` is rendered into the system block.
        The actual serialization happens in `_build_prompt_fragment`.
        """
        new_kwargs = dict(kwargs or {})
        surfaces = self._thinking_control_surfaces()
        # Asset-driven: MLX serializes prompts locally, so the robust disable control is the
        # assistant-prefill marker declared as `thinking_control.assistant_prefill_disable`.
        if surfaces.assistant_prefill_disable:
            if enabled is False:
                new_kwargs["_acore_mlx_enable_thinking"] = False
                return new_kwargs, ThinkingControlHandling(
                    handled_enable_disable=True,
                    handled_level=False,
                )
            if enabled is True or level is not None:
                new_kwargs["_acore_mlx_enable_thinking"] = True
                handled_level = False
                # The effort artifact lives in the SYSTEM block. When the caller
                # feeds a prefilled system bloc (KV-mode CachedSession passes
                # prompt_cache_prefilled_modules=("system", ...)), that bloc was
                # serialized WITHOUT the line and the fragment renderer must not
                # reopen the system region — so no artifact can be applied.
                # Claiming handled_level there would report an effort level with
                # no control in the prompt (ADR-0001); decline instead so the
                # base warning ladder reports the honest degradation.
                prefilled = new_kwargs.get("prompt_cache_prefilled_modules")
                if isinstance(prefilled, str):
                    prefilled = [prefilled]
                system_prefilled = isinstance(prefilled, (list, tuple)) and any(
                    str(item or "").strip().lower() == "system" for item in prefilled
                )
                effort_lines = surfaces.effort_system_lines or {}
                if isinstance(level, str) and level in effort_lines and not system_prefilled:
                    # Claimed handled even for a level whose declared line is empty
                    # (Qwen3.8 "medium"): the template renders nothing for it too.
                    new_kwargs["_acore_mlx_reasoning_effort"] = level
                    handled_level = True
                return new_kwargs, ThinkingControlHandling(
                    handled_enable_disable=True,
                    handled_level=handled_level,
                )
        return new_kwargs, ThinkingControlHandling()

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        if getattr(self, "_mtp_processor", None) is not None:
            raise ProviderAPIError(
                f"Native MLX lane for {self._native_family_label()} uses automatic prefix caching: "
                "pass prompt_cache_key and complete messages to generate(); manual "
                "prefill/append/fork is unsupported."
            )
        try:
            from mlx_lm.models.cache import make_prompt_cache
        except Exception:
            return None
        try:
            return make_prompt_cache(self.llm)
        except Exception:
            return None

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        """Deep clone of an MLX prompt cache.

        `copy.deepcopy` FIRST (correctness): `ArraysCache.from_state` returns the
        live per-layer array LIST and `from_state` re-assigns it, so a
        `from_state`-based clone ALIASES the parent's mutable recurrent slots —
        a later live step-write (`cache[i] = …` during generation on a
        Gated-DeltaNet hybrid) then silently corrupts the "clone" (and vice
        versa). This is a latent defect for hybrid module-cache forks
        (`prompt_cache_prepare_modules`) independent of the snapshot lane, found
        by two fable5 adversaries (2026-07-15) and reproduced. `deepcopy` copies
        the arrays (measured 0.2–3 ms on a 4B hybrid; mlx arrays deep-copy
        correctly) and is the same discipline mlx_lm's own `fetch_nearest_cache`
        uses. The `from_state` path stays as a fallback ONLY if deepcopy fails
        (never expected for a real cache).
        """
        if cache_value is None:
            return None

        try:
            # Materialize the source's lazy state FIRST so the deepcopy captures
            # concrete arrays (an independent copy of a lazy graph could still
            # share upstream nodes until evaluated).
            try:
                import mlx.core as mx

                layers = cache_value if isinstance(cache_value, (list, tuple)) else [cache_value]
                states = [getattr(layer, "state", None) for layer in layers]
                mx.eval([s for s in states if s is not None])
            except Exception:
                pass
            return copy.deepcopy(cache_value)
        except Exception:
            pass  # fall through to the legacy from_state clone below

        def _clone_layer(layer: Any) -> Any:
            from_state = getattr(layer.__class__, "from_state", None)
            state_attr: Any = None
            if callable(from_state):
                try:
                    state_attr = getattr(layer, "state", None)
                except Exception:
                    state_attr = None
            if callable(from_state):
                try:
                    state_val = state_attr() if callable(state_attr) else state_attr
                    meta_attr = getattr(layer, "meta_state", None)
                    meta_val = meta_attr() if callable(meta_attr) else meta_attr
                    if state_val is not None:
                        try:
                            sig = inspect.signature(from_state)
                            if len(sig.parameters) == 2:
                                return from_state(state_val, meta_val)
                            if len(sig.parameters) == 1:
                                return from_state(state_val)
                        except Exception:
                            pass

                        # Fallback: try the common 2-arg then 1-arg patterns.
                        try:
                            return from_state(state_val, meta_val)
                        except TypeError:
                            return from_state(state_val)

                    # Some MLX-LM cache layers (notably KVCache) cannot serialize an "empty" state.
                    # Fall back to constructing a new empty instance when state is unavailable.
                    try:
                        empty = layer.__class__()  # type: ignore[call-arg]
                        try:
                            if meta_val is not None and hasattr(empty, "meta_state"):
                                empty.meta_state = meta_val  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        return empty
                    except Exception:
                        return None
                except Exception:
                    return None
            if hasattr(layer, "copy"):
                try:
                    return layer.copy()
                except Exception:
                    return None
            return None

        # MLX-LM prompt caches are typically a list of per-layer KVCache objects.
        if isinstance(cache_value, list):
            cloned: List[Any] = []
            for layer in cache_value:
                c = _clone_layer(layer)
                if c is None:
                    return None
                cloned.append(c)
            return cloned

        if isinstance(cache_value, tuple):
            cloned_layers: List[Any] = []
            for layer in cache_value:
                c = _clone_layer(layer)
                if c is None:
                    return None
                cloned_layers.append(c)
            return tuple(cloned_layers)

        # Fallback: single cache object.
        return _clone_layer(cache_value)

    # ---- Snapshot/restore lane for untrimmable (recurrent) architectures ----

    def _prefill_tokens_into_cache(self, cache_value: Any, token_ids: List[int]) -> bool:
        """Run the model forward over token_ids into cache_value with NO decode.

        Uses mlx_lm's `generate_step(max_tokens=0)`, which executes exactly the
        prefill loop (chunked, cache-mutating) and stops before yielding any
        sampled token — so the cache lands on the EXACT token boundary with no
        reply pollution. This is the clean boundary a recurrent state needs for
        a reusable snapshot (it cannot be reached by trimming after the fact).

        WIRED-LIMIT / STREAM SYNC (2026-08-02, gateway crash-loop). `generate_step`
        leaves an `mx.async_eval` in flight on mlx_lm's dedicated `generation_stream`.
        `stream_generate` — the lane this one replaces — always ran inside
        `wired_limit(model, [generation_stream])`, whose __exit__ SYNCHRONIZES that
        stream before restoring the wired limit; mlx_lm's own docstring warns the
        wired limit "should not be changed during an async eval". Calling
        `generate_step` bare left GPU work pending after return, so the caller's next
        `mx.eval` (and any concurrent generate in another thread flipping the wired
        limit back) hit Metal's `-[_MTLCommandBuffer addCompletedHandler:]` assert →
        SIGABRT. In a single-threaded script that window is invisible; inside the
        gateway, where MLX runs on worker threads, it crash-looped the process on the
        first cached call. Mirror `stream_generate` exactly.
        """
        if cache_value is None or not token_ids:
            return False
        try:
            import mlx.core as mx
            from mlx_lm.generate import generate_step
        except Exception:
            return False
        try:
            from mlx_lm.generate import generation_stream, wired_limit
        except Exception:  # older mlx_lm: fall back to a plain synchronize
            generation_stream = None
            wired_limit = None
        try:
            if wired_limit is not None and generation_stream is not None:
                with wired_limit(self.llm, [generation_stream]):
                    for _ in generate_step(
                        mx.array(token_ids), self.llm, max_tokens=0, prompt_cache=cache_value
                    ):
                        pass  # max_tokens=0 yields nothing; prefill is the side effect
            else:
                for _ in generate_step(
                    mx.array(token_ids), self.llm, max_tokens=0, prompt_cache=cache_value
                ):
                    pass
                try:
                    mx.synchronize()
                except Exception:
                    pass
            try:
                mx.eval([getattr(layer, "state", None) for layer in cache_value])
            except Exception:
                pass
            return True
        except Exception as exc:
            self.logger.debug(f"MLX snapshot prefill failed: {exc}")
            return False

    def prompt_cache_clear(self, key: Optional[str] = None) -> bool:
        """Clear prompt caches AND their hybrid snapshots (avoid stale/leaked
        snapshot state outliving the key it mirrors)."""
        native = getattr(self, "_native_session", None) or getattr(self, "_native_qwen4", None)
        if native:
            runtime = getattr(self, "_native_runtime", None)
            if runtime is not None:
                if key is not None:
                    self.logger.warning("#FALLBACK Native APC clear(key) clears all shared native prefix entries")
                runtime.control(native.cache_store.clear)
                return True
            if not native.lock.acquire(blocking=False):
                raise ProviderAPIError("Cannot clear native prompt cache during generation; close its stream first")
            try:
                if key is not None:
                    self.logger.warning("#FALLBACK Native APC clear(key) clears all shared native prefix entries")
                if native.apc is not None:
                    native.apc.clear()
                return True
            finally:
                native.lock.release()
        result = super().prompt_cache_clear(key)
        with self._hybrid_snapshot_lock:
            if key is None:
                self._hybrid_snapshots.clear()
            else:
                norm = self._normalize_prompt_cache_key(key)
                self._hybrid_snapshots.pop(norm, None)
        return result

    def _ensure_hybrid_snapshot_state(self) -> None:
        """Lazily materialize the snapshot store/lock.

        The delta feed must not assume `__init__` ran — provider instances are
        sometimes built via `__new__` (e.g. unit tests that exercise the pure
        cache logic with fakes). Real instances always have `__init__`, so the
        first-caller race is single-threaded in practice.
        """
        if not hasattr(self, "_hybrid_snapshot_lock"):
            self._hybrid_snapshot_lock = threading.RLock()
        if not hasattr(self, "_hybrid_snapshots"):
            self._hybrid_snapshots = {}

    def _get_hybrid_snapshot(self, key: str) -> Optional[Dict[str, Any]]:
        self._ensure_hybrid_snapshot_state()
        with self._hybrid_snapshot_lock:
            snap = self._hybrid_snapshots.get(key)
            if snap is not None:
                # LRU recency: a restored-from key must not be the next one
                # bound-evicted just because it was stored long ago.
                self._hybrid_snapshots.pop(key, None)
                self._hybrid_snapshots[key] = snap
            return snap

    def _hybrid_snapshot_bound(self) -> int:
        """Max resident snapshots == the prompt-cache store's entry bound.

        A snapshot only ever mirrors a store key (it is captured on the feed
        path of a key that is IN the store), so more snapshots than store
        entries is by definition leaked state. 2026-08-03 leak audit / red-team
        P1: this dict was unbounded — `PromptCacheStore` LRU-evicted keys with
        no callback and each orphaned entry held a full KV-cache deepcopy
        (~1 GB at 30k on Qwen3.5-4B) for the provider's lifetime."""
        store = getattr(self, "_prompt_cache_store", None)
        try:
            bound = int(getattr(store, "_max_entries", 0) or 0)
        except Exception:
            bound = 0
        return bound if bound > 0 else 32

    def _store_hybrid_snapshot(self, key: str, cache_value: Any, token_ids: List[int]) -> None:
        """Keep one snapshot per key (the growing one evicts its predecessor),
        and at most `_hybrid_snapshot_bound()` snapshots overall (LRU)."""
        self._ensure_hybrid_snapshot_state()
        with self._hybrid_snapshot_lock:
            self._hybrid_snapshots.pop(key, None)
            self._hybrid_snapshots[key] = {"cache": cache_value, "ids": list(token_ids)}
            bound = self._hybrid_snapshot_bound()
            while len(self._hybrid_snapshots) > bound:
                # dicts preserve insertion order; get/store re-insert = LRU.
                oldest = next(iter(self._hybrid_snapshots))
                self._hybrid_snapshots.pop(oldest, None)

    def _drop_hybrid_snapshot(self, key: str) -> None:
        self._ensure_hybrid_snapshot_state()
        with self._hybrid_snapshot_lock:
            self._hybrid_snapshots.pop(key, None)

    def _prompt_cache_store_evicted(self, key: str, cache_value: Any) -> None:
        """Store evicted `key` behind our back (LRU capacity / TTL expiry):
        drop its hybrid snapshot too, or the deepcopy outlives the entry it
        mirrors for the provider's lifetime (2026-08-03 leak audit)."""
        _ = cache_value
        try:
            self._drop_hybrid_snapshot(str(key))
        except Exception:
            pass

    @staticmethod
    def _mlx_cache_nbytes(cache_value: Any) -> Optional[int]:
        """Best-effort byte size of an MLX KV cache (sum of state array nbytes).

        None when nothing measurable was found — never guess."""
        if cache_value is None:
            return None
        try:
            total = 0
            found = False
            layers = cache_value if isinstance(cache_value, (list, tuple)) else [cache_value]
            for layer in layers:
                stack: List[Any] = [getattr(layer, "state", None)]
                while stack:
                    node = stack.pop()
                    if node is None:
                        continue
                    if isinstance(node, (list, tuple)):
                        stack.extend(node)
                        continue
                    nbytes = getattr(node, "nbytes", None)
                    if isinstance(nbytes, int):
                        total += nbytes
                        found = True
            return total if found else None
        except Exception:
            return None

    def _prompt_cache_value_bytes(self, cache_value: Any) -> Optional[int]:
        return self._mlx_cache_nbytes(cache_value)

    _NATIVE_APC_COUNTERS = ("exact_stores", "exact_hits", "memory_skips", "rejects", "stores", "hits", "misses")

    def _native_apc_counters(self) -> Optional[Dict[str, int]]:
        """Counters of the direct lane's mlx-vlm prefix cache, or None.

        None when no manager exists yet (the first keyed call builds it and its
        counters start at zero, which is what a None baseline means to
        `_native_apc_delta`) and on the scheduled lane, whose manager is
        worker-owned and must not be read from the caller thread.
        """
        if getattr(self, "_native_runtime", None) is not None:
            return None
        native = getattr(self, "_native_session", None) or getattr(self, "_native_qwen4", None)
        apc = getattr(native, "apc", None) if native is not None else None
        if apc is None:
            return None
        snap = apc.stats_snapshot()
        counters = {name: int(snap.get(name) or 0) for name in self._NATIVE_APC_COUNTERS}
        counters["memory_max_bytes"] = int(snap.get("memory_max_bytes") or 0)
        counters["resident_bytes"] = int(snap.get("resident_bytes") or 0)
        return counters

    def _native_apc_delta(self, before: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        """What THIS call did to the prefix cache: counter deltas plus the budget it ran under."""
        after = self._native_apc_counters()
        if after is None:
            return None
        base = before or {}
        delta = {name: after[name] - int(base.get(name, 0)) for name in self._NATIVE_APC_COUNTERS}
        delta["memory_max_bytes"] = after["memory_max_bytes"]
        delta["resident_bytes"] = after["resident_bytes"]
        return delta

    def _native_apc_telemetry(self, telemetry: Dict[str, Any], native_result: Any) -> None:
        """Fill the native lane's prompt-cache telemetry from MEASURED values only.

        `outcome` uses the mlx-lm lane's vocabulary (`hit_restore` / `cold`) and
        is decided by the runtime's own `cached_tokens`. `apc` carries this
        call's store/skip counters (from `_mtp_last_apc`, captured around the
        native generate call) so a snapshot that did not fit the budget shows
        up in the ledger as what it is. mlx-vlm skips retaining any snapshot
        larger than its memory budget and says nothing; the failure this makes
        loud is "every turn re-prefills 20k tokens and no field explains why".
        """
        cached = int(getattr(native_result, "cached_tokens", 0) or 0)
        prompt_tokens = int(getattr(native_result, "prompt_tokens", 0) or 0)
        telemetry["cached_tokens"] = cached
        telemetry["fed_tokens"] = prompt_tokens - cached
        telemetry["outcome"] = "hit_restore" if cached > 0 else "cold"
        delta = getattr(self, "_mtp_last_apc", None)
        if delta is None:
            return
        telemetry["apc"] = dict(delta)
        stored = delta["exact_stores"] > 0 or delta["stores"] > 0
        reason = None
        if not stored and delta["memory_skips"] > 0:
            reason = (
                f"native_apc_store_skipped: the {prompt_tokens}-token prefix snapshot did not fit "
                f"the prefix-cache memory budget ({delta['memory_max_bytes'] >> 20} MiB), so the "
                "next turn on this key re-prefills the whole conversation. Raise "
                "mlx_cache_memory_max_gb, or unset it for the machine-sized default."
            )
        elif not stored and delta["rejects"] > 0:
            reason = (
                f"native_apc_store_rejected: mlx-vlm refused to snapshot the {prompt_tokens}-token "
                "prefix (see its 'APC exact-cache store rejected' log line); the next turn on "
                "this key re-prefills the whole conversation."
            )
        if reason is None:
            return
        telemetry["degraded_reason"] = f"#FALLBACK {reason}"
        warned = getattr(self, "_native_apc_skip_warned_keys", None)
        if warned is None:
            warned = self._native_apc_skip_warned_keys = set()
        key = telemetry.get("key")
        if key not in warned:
            warned.add(key)
            self.logger.warning("mlx: " + reason)

    def get_prompt_cache_stats(self) -> Dict[str, Any]:
        """Add hybrid-snapshot visibility (count + best-effort bytes) to base stats."""
        native = getattr(self, "_native_session", None) or getattr(self, "_native_qwen4", None)
        if native:
            runtime = getattr(self, "_native_runtime", None)
            if runtime is not None:
                return {"backend": "mlx_vlm_apc", "full_history_required": True,
                        "stats": runtime.control(native.cache_store.stats),
                        "execution": runtime.stats()}
            return {"backend": "mlx_vlm_apc", "full_history_required": True,
                    "stats": native.apc.stats_snapshot() if native.apc is not None else {}}
        stats = super().get_prompt_cache_stats()
        try:
            self._ensure_hybrid_snapshot_state()
            with self._hybrid_snapshot_lock:
                snaps = list(self._hybrid_snapshots.values())
            total = 0
            found = False
            for snap in snaps:
                nbytes = self._mlx_cache_nbytes((snap or {}).get("cache"))
                if isinstance(nbytes, int):
                    total += nbytes
                    found = True
            stats["snapshots"] = {"count": len(snaps), "bytes": total if found else None}
        except Exception:
            pass
        return stats

    # NOTE (2026-08-03): `_capture_hybrid_snapshot` lived here — prefill the ids
    # into a SECOND, fresh cache and store that as the key's boundary. It had
    # zero callers in HEAD and in the tree, and it is now removed rather than
    # wired, for two reasons:
    #   1. It is redundant. `_hybrid_snapshot_feed` takes its deepcopy BEFORE
    #      generation touches `working`, so the boundary it stores is already
    #      clean and reply-free — which was this helper's entire stated purpose.
    #   2. Wiring it would cost a WHOLE EXTRA PROMPT PASS per turn to produce a
    #      boundary the lane already produces for free, and there is no longer a
    #      call site where it could even be correct: untrimmable architectures
    #      now enter the snapshot lane on the architecture check, so they never
    #      finish a turn on the trim lane needing a snapshot bolted on after.
    # Left as a comment because a duplicate snapshot path is exactly the shape a
    # future "fix" would reach for, and it would reintroduce the double prefill.

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        """Token count of a live cache, or None when it cannot be known.

        Empty vs UNCOUNTABLE matters (adversarial find P1-2): pure-SSM
        architectures (mamba/mamba2/rwkv7 → all-ArraysCache) expose neither
        `size()` nor `offset`, so a WARM cache used to read as 0 —
        indistinguishable from cold, silently reviving the double-prefill for
        them. When no layer yields a count, `empty()` is consulted: all-empty
        → genuinely 0; any non-empty (or unknowable) → None, which callers
        must treat as "warm, composition unknowable". (CacheList wrappers —
        falcon-h1, longcat-flash, deepseek-v3.2 — DO expose `size()` in
        mlx_lm ≥0.31 and are countable; their trimmability is decided per
        child layer by `can_trim_prompt_cache`.)
        """
        if cache_value is None:
            return 0
        try:
            if isinstance(cache_value, (list, tuple)):
                counts: List[int] = []
                for layer in cache_value:
                    if hasattr(layer, "size"):
                        try:
                            s = int(layer.size())
                        except Exception:
                            s = None
                        if isinstance(s, int) and s > 0:
                            counts.append(s)
                    if hasattr(layer, "offset"):
                        try:
                            off = int(getattr(layer, "offset", 0))
                        except Exception:
                            off = 0
                        if off > 0:
                            counts.append(off)
                if counts:
                    return max(counts)
                for layer in cache_value:
                    try:
                        if not bool(layer.empty()):
                            return None  # warm but uncountable
                    except Exception:
                        return None  # unknowable — never report cold
                return 0
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Delta generation over warm KV caches (adversarial find B2, 2026-07-12)
    #
    # mlx_lm has NO common-prefix dedup: feeding the full rendered prompt on
    # top of a warm cache prefills the transcript AGAIN on top of its own KV —
    # caching ON cost ~2x caching OFF and duplicated the transcript
    # in-context. This is the HuggingFace provider's delta pattern ported to
    # the token level: track the token ids each cache was fed, LCP the new
    # prompt against them, trim the cache to the shared prefix, feed ONLY the
    # suffix. Warm caches of unknown composition (loaded artifacts, caches
    # born before this fix) keep the legacy full feed with a one-time
    # #FALLBACK warning — never a silent behavior change under a durable-bloc
    # flow, and never a misdescribed cache.
    # ------------------------------------------------------------------

    _FED_TOKEN_IDS_META = "fed_token_ids"

    # Class-level defaults for the append stash. `__init__` sets instance
    # copies; these keep the record path working for providers built through
    # `__new__` (the render/plan-only construction used by tests, which must
    # never load weights).
    _pending_append_fragment: Optional[str] = None
    _pending_append_fragment_ids: Optional[List[int]] = None
    _pending_append_precount: int = 0

    def _encode_prompt_token_ids(self, text: str) -> Optional[List[int]]:
        """Tokenize exactly as mlx_lm's str path would.

        mlx_lm (generate.py) infers `add_special_tokens` from whether the
        prompt STARTS WITH the BOS literal (templates like gemma-turn render
        it into the text; adding it again would double it). The record must
        replicate that inference byte-for-byte or it runs one token long for
        exactly those architectures — an off-by-one that silently skews every
        trim (adversarial find P0-1). Suffix feeds are token lists and bypass
        tokenization entirely, so only this record path needs the mirror.
        """
        text = str(text or "")
        try:
            add_special: Optional[bool] = None
            bos = getattr(self.tokenizer, "bos_token", None)
            if isinstance(bos, str) and bos:
                add_special = not text.startswith(bos)
            if add_special is None:
                ids = self.tokenizer.encode(text)
            else:
                try:
                    ids = self.tokenizer.encode(text, add_special_tokens=add_special)
                except TypeError:
                    # Tokenizer without the kwarg (plain callables in tests,
                    # exotic wrappers): plain encode is then also what
                    # mlx_lm's mx.array(tokenizer.encode(...)) path does.
                    ids = self.tokenizer.encode(text)
        except Exception:
            return None
        if ids is None:
            return None
        try:
            return [int(t) for t in ids]
        except Exception:
            return None

    @staticmethod
    def _token_lcp_len(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _generation_prompt_literals(self) -> List[str]:
        """The exact generation-prompt tail(s) THIS provider's renderer emits.

        DERIVED, never mirrored. `_build_prompt_fragment` with every content input
        empty and `include_bos=False` renders nothing at all except the
        `add_generation_prompt` block, so calling it that way returns the literal
        itself — for whichever of the renderer's branches this model takes, ChatML
        or gemma-turn or the `role:`/`assistant:` fallback alike.

        A hand-copied list of literals was the first attempt and it was wrong: it
        covered two of the renderer's three branches, so every family that falls
        through to `assistant:` — `basic`, `inst`, `openai_chat`, `llama3_header`,
        `special_tokens`, `glm_special_tokens`, `human_assistant`, `harmony`, and
        critically the UNTRIMMABLE HYBRID `granitemoehybrid` that this whole lane
        exists to serve — silently got no holdback and the turn-2 rebuild back.
        Deriving from the renderer cannot drift out of sync with the renderer.

        `enable_thinking` is not known here, so both forms are produced and the
        caller matches longest-first (the thinking-disabled ChatML form extends
        the plain one).
        """
        out: List[str] = []
        for thinking in (False, None):
            try:
                tail = self._build_prompt_fragment(
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                    include_bos=False,
                )
            except Exception:
                continue
            if tail and tail not in out:
                out.append(tail)
        out.sort(key=len, reverse=True)
        return out

    def _generation_prompt_boundary(self, full_prompt: str, full_ids: List[int]) -> Optional[int]:
        """Token POSITION where this call's generation-prompt scaffolding begins.

        That scaffolding is PER-CALL VOLATILE by construction: it asks the model to
        speak now, and the next turn's transcript replaces it with the assistant
        turn that actually happened. A cache boundary containing it is a prefix of
        nothing — which matters only on untrimmable architectures, where a boundary
        is the sole reusable artifact (a recurrent state cannot be rewound to drop
        it after the fact).

        A POSITION, deliberately, not a count. Re-tokenize the prompt without the
        literal and take the token-level LCP against the full prompt: if the seam
        merges, `len(full) - len(head)` and the true divergence position differ by
        one, and using the count as a position would put the boundary one token
        INSIDE the scaffolding — i.e. silently back in the bug. The LCP is the
        divergence position by definition, merge or no merge.

        Returns None when the prompt does not end in this renderer's generation
        prompt, or when it cannot be tokenized: no holdback is applied, which is
        the pre-existing behaviour and never a guess. Costs one extra tokenize, and
        only on a key that has no fed-token record yet (turn 1 of a session).
        """
        text = str(full_prompt or "")
        for tail in self._generation_prompt_literals():
            if not text.endswith(tail):
                continue
            head_ids = self._encode_prompt_token_ids(text[: -len(tail)])
            if not head_ids:
                return None
            return self._token_lcp_len(head_ids, full_ids)
        return None

    def _stable_head_boundary(
        self,
        stable_head: Optional[Callable[[], Optional[str]]],
        full_prompt: str,
        full_ids: List[int],
    ) -> Optional[int]:
        """Token POSITION where this prompt's final turn begins, or None.

        `stable_head` renders the same request minus its final turn through the
        same renderer. It is only trusted when it is a literal text prefix of the
        prompt; the position is the token LCP (never a count — see
        `_generation_prompt_boundary` for why a seam merge makes those differ).
        """
        if not callable(stable_head):
            return None
        try:
            head_text = stable_head()
        except Exception:
            return None
        if not isinstance(head_text, str) or not head_text:
            return None
        if not str(full_prompt or "").startswith(head_text):
            return None
        head_ids = self._encode_prompt_token_ids(head_text)
        if not head_ids:
            return None
        return self._token_lcp_len(head_ids, full_ids)

    def _key_is_forked(self, key: str) -> bool:
        meta = self.prompt_cache_key_meta(key) or {}
        return bool(str(meta.get("forked_from") or "").strip())

    def _warn_seed_diverged(self, key: str, *, shared: int, recorded: int) -> None:
        """A prepared prefix that is not a prefix of the prompt. Loud, once per key.

        This is the failure that hid a dead prefix cache behind healthy-looking
        outcomes (`hit_restore` cached=3, `hit_extend` cached=3). It always means
        the bloc chain and `generate()` rendered the same request differently.
        """
        warn_key = f"seed-diverged:{key}"
        warned = getattr(self, "_delta_feed_warned_keys", None)
        if warned is None:  # instances built via __new__ (pure cache-logic unit tests)
            warned = self._delta_feed_warned_keys = set()
        if warn_key in warned:
            return
        warned.add(warn_key)
        self.logger.warning(
            f"#FALLBACK MLX prompt cache '{key}': the prepared prefix it was forked from is NOT "
            f"a prefix of the prompt generate() built (shared {shared} of {recorded} recorded "
            f"tokens). The prefix cache is unreachable and this call pays a full prefill. The "
            f"bloc chain was rendered differently from generate() — typically a control that "
            f"rewrites the head of the system block (pass the same `thinking=` to "
            f"prompt_cache_prepare_modules that generate() receives)."
        )

    def _cache_is_trimmable(self, cache_value: Any) -> bool:
        """True if mlx_lm can trim this cache (architecture-determined, not
        fill-determined — an EMPTY hybrid cache already reports False). Used to
        route cold untrimmable caches through the snapshot lane from turn 1, so
        the first turn leaves a reusable boundary. Absent predicate → assume
        trimmable (the delta path's own trim attempt is the real gate)."""
        try:
            from mlx_lm.models.cache import can_trim_prompt_cache
        except Exception:
            return True
        try:
            return bool(can_trim_prompt_cache(cache_value))
        except Exception:
            return True

    def _trim_prompt_cache_tokens(self, cache_value: Any, num_tokens: int) -> bool:
        """Trim `num_tokens` from the END of a live cache (best-effort)."""
        if num_tokens <= 0:
            return True
        try:
            from mlx_lm.models.cache import trim_prompt_cache

            try:
                from mlx_lm.models.cache import can_trim_prompt_cache
            except Exception:
                can_trim_prompt_cache = None
            if callable(can_trim_prompt_cache) and not bool(can_trim_prompt_cache(cache_value)):
                return False
            trimmed = trim_prompt_cache(cache_value, int(num_tokens))
            # mlx_lm returns the count actually trimmed; a partial trim would
            # silently corrupt the delta arithmetic, so treat it as failure
            # (callers fall back to a fresh cache).
            if isinstance(trimmed, int) and trimmed < int(num_tokens):
                return False
            return True
        except Exception:
            return False

    def _fed_token_ids_for_key(self, key: str) -> Optional[List[int]]:
        meta = self.prompt_cache_key_meta(key) or {}
        raw = meta.get(self._FED_TOKEN_IDS_META)
        if not isinstance(raw, list) or not raw:
            return None
        try:
            return [int(t) for t in raw]
        except Exception:
            return None

    def _record_fed_token_ids(self, key: str, ids: Optional[List[int]]) -> None:
        if not ids:
            return
        try:
            self.prompt_cache_update_key_meta(
                key, **{self._FED_TOKEN_IDS_META: [int(t) for t in ids]}
            )
        except Exception:
            pass

    @staticmethod
    def _parse_persisted_fed_token_ids(raw: Any) -> Optional[List[int]]:
        """Parse a fed-token-id record out of artifact metadata (0819).

        Safetensors metadata is string-keyed AND string-valued, so a record
        persisted at save arrives back as a JSON string; store-native lists
        are accepted too (round-trips through in-memory paths). Anything that
        does not parse to a non-empty list of ints returns None — an
        unparseable record must never be mistaken for cache truth.
        """
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            try:
                raw = json.loads(text)
            except Exception:
                return None
        if not isinstance(raw, list) or not raw:
            return None
        try:
            return [int(t) for t in raw]
        except Exception:
            return None

    def _prepare_cache_delta_feed(
        self,
        key: str,
        cache_value: Any,
        full_prompt: str,
        *,
        full_context: bool,
        telemetry: Optional[Dict[str, Any]] = None,
        stable_head: Optional[Callable[[], Optional[str]]] = None,
    ) -> tuple[Any, Any, Optional[List[int]]]:
        """Decide what to feed mlx_lm over a (possibly warm) cache.

        `full_context` is the CALLER-SHAPE discriminator and is load-bearing:

        - full_context=True (the caller passed `messages`, i.e. it re-sends
          the WHOLE logical context every call — the runtime transcript lane,
          the population B2's double-prefill actually bites): delta
          discipline. LCP heuristics alone are UNSAFE here-below because a
          delta-style fragment legitimately shares a few chat-template header
          tokens with the recorded head; only the caller shape disambiguates.
        - full_context=False (prompt-only callers: CachedSession KV mode
          sends ONLY the new fragment while history lives in the cache by
          contract; direct `generate(prompt=..., prompt_cache_key=...)`
          callers historically accumulate turns in the cache): APPEND
          semantics — feed as-is on top of the warm cache (that IS the
          correct behavior there), extend the id record, never trim. Running
          LCP arithmetic here would see a tiny shared prefix and trim away
          the whole session context.

        Returns (cache_to_use, prompt_to_feed, ids_to_record_after_feed).
        Full-context lattice: cold → legacy feed + start tracking; warm +
        pure extension → trim generated drift, feed ONLY the suffix ids;
        identical → keep one token; divergence or unknown composition →
        FRESH cache + full feed (one cold prefill is correct for a caller
        that re-sends everything — and never a double one); trim/tokenize
        failure → fresh or bypass. Artifact-backed caches never rebuild AND
        never trim below their recorded prefix — divergence bypasses instead
        (a shared stable bloc key must not be degraded by one divergent call).

        `telemetry` (0819, runtime seam condition): caller-owned out-dict —
        when provided, the decision is recorded into it (`outcome`,
        MEASURED `cached_tokens`/`fed_tokens`, `degraded_reason`). Caller
        ownership keeps concurrent generates race-free (no instance stash).

        `stable_head` (2026-09-17): lazy renderer-derived text of this prompt
        WITHOUT its final turn — consulted only when a forked prefix turns out
        not to be a prefix of the prompt (see `seed_diverged` below).
        """

        def _note(
            outcome: str,
            *,
            cached: Optional[int] = None,
            fed: Optional[int] = None,
            reason: Optional[str] = None,
        ) -> None:
            if telemetry is None:
                return
            telemetry["outcome"] = outcome
            if cached is not None:
                telemetry["cached_tokens"] = int(cached)
            if fed is not None:
                telemetry["fed_tokens"] = int(fed)
            if reason:
                telemetry["degraded_reason"] = f"#FALLBACK {reason}"

        if cache_value is None:
            _note("off")
            return cache_value, full_prompt, None

        # None = warm-but-uncountable (pure-SSM/CacheList architectures) or
        # unknowable — NEVER coerced to 0: reading warm as cold is exactly
        # the double-prefill revival (adversarial find P1-2).
        try:
            raw_count = self._prompt_cache_backend_token_count(cache_value)
        except Exception:
            raw_count = None
        cache_len: Optional[int] = (
            raw_count if isinstance(raw_count, int) and raw_count >= 0 else None
        )

        new_ids = self._encode_prompt_token_ids(full_prompt)
        if not new_ids:
            if full_context and (cache_len is None or cache_len > 0):
                # Cannot tokenize deterministically: never risk the double prefill.
                _note(
                    "bypassed",
                    reason="prompt could not be tokenized deterministically; warm cache bypassed",
                )
                return None, full_prompt, None
            _note("append" if not full_context else "cold", cached=cache_len)
            return cache_value, full_prompt, None

        fed_ids = self._fed_token_ids_for_key(key)

        if not full_context:
            # APPEND semantics (KV-source-of-truth sessions, prompt-only
            # accumulators): the warm cache IS the context; the prompt is the
            # next fragment — exactly the legacy behavior, untouched. LCP
            # arithmetic here would see a tiny shared prefix and trim away
            # the whole session; only the caller SHAPE (messages= present)
            # may select delta discipline, never a content heuristic. The
            # record may only extend while it stays a TRUE token-prefix of
            # the cache — once generated tokens sit between fragments
            # (cache_len > record), an extended record would misdescribe the
            # cache; the old record stays (still a true prefix). Uncountable
            # caches (cache_len None) can never verify prefix-truth: feed
            # legacy, record nothing.
            record = None
            if cache_len is not None:
                if cache_len <= 0:
                    record = new_ids
                elif fed_ids is not None and cache_len == len(fed_ids):
                    record = fed_ids + new_ids
            _note("append", cached=cache_len, fed=len(new_ids))
            return cache_value, full_prompt, record

        cold_empty = cache_len is not None and cache_len <= 0

        def _fresh_full_feed() -> tuple[Any, Any, Optional[List[int]]]:
            # Preserve the entry's own meta (minus the now-stale id record)
            # and TTL: the fresh cache is the same LOGICAL key — wiping
            # binding/provenance fields here broke durable-bloc validation
            # (adversarial find P1-5).
            prior_meta = dict(self.prompt_cache_key_meta(key) or {})
            prior_meta.pop(self._FED_TOKEN_IDS_META, None)
            prior_meta.setdefault("backend", "mlx")
            try:
                prior_ttl = self._prompt_cache_store.ttl_s(key)
            except Exception:
                prior_ttl = None
            fresh = self._prompt_cache_backend_create()
            if fresh is not None:
                try:
                    self._prompt_cache_store.set(key, fresh, meta=prior_meta, ttl_s=prior_ttl)
                except Exception:
                    pass
                return fresh, full_prompt, new_ids
            _note("bypassed", reason="fresh cache creation failed; generated without a cache")
            return None, full_prompt, None

        def _is_artifact_backed() -> bool:
            meta = self.prompt_cache_key_meta(key) or {}
            if not (
                meta.get("loaded_from") or meta.get("binding_id") or meta.get("artifact_sha256")
            ):
                return False
            # A FORKED key is a private COPY of an artifact, not the artifact.
            #
            # `prompt_cache_fork` copies the source meta wholesale, so a session
            # forked from a durable bloc inherited this flag — and every branch it
            # guards is "bypass rather than modify", so from turn 2 onward the
            # session cache was never used AT ALL. Measured 4-turn lattice on both
            # lanes: `hit_extend, bypassed, bypassed, bypassed`, against
            # `hit_restore x4` for an identical fork from a non-artifact bloc. Any
            # benchmark of bloc-ARTIFACT reuse was therefore measuring turn 1 only.
            #
            # The protection is for a SHARED, verified artifact that other callers
            # will read: trimming it to a divergent caller's prefix would leave the
            # next reader a stub. A forked session key has no other readers, and
            # the artifact it was copied from is untouched by anything done here —
            # so the copy must not inherit the protection.
            return not str(meta.get("forked_from") or "").strip()

        def _hybrid_snapshot_feed() -> tuple[Any, Any, Optional[List[int]]]:
            """Snapshot/restore feed for UNTRIMMABLE architectures.

            A recurrent state cannot be trimmed, but it can be COPIED: restore
            the per-key snapshot when its recorded ids are a true prefix of the
            new prompt (forward-only, no rewind — the discipline llama.cpp's
            GGUF lane and mlx_lm's own server use), prefill only the suffix onto
            the restored copy, and re-snapshot the new boundary for the next
            turn. On a cold/divergent turn there is no usable snapshot, so this
            does one full prefill (same cost as the old rebuild-fresh) BUT
            leaves a snapshot behind, so the loop's subsequent warm turns are
            cheap. The boundary is new_ids[:-1] so a single trailing token seeds
            decoding without re-prefilling (generation needs a non-empty seed).
            """
            snap = self._get_hybrid_snapshot(key)
            working: Any = None
            prefix_len = 0
            if snap is not None:
                snap_ids = list(snap.get("ids") or [])
                lcp_snap = self._token_lcp_len(snap_ids, new_ids)
                # A true prefix that still leaves a suffix to feed.
                if snap_ids and lcp_snap == len(snap_ids) and lcp_snap < len(new_ids):
                    restored = self._prompt_cache_backend_clone(snap.get("cache"))
                    if restored is not None:
                        working = restored
                        prefix_len = lcp_snap
            restored_from_snapshot = working is not None

            # LIVE-CACHE SEED (2026-08-03). No snapshot yet, but the live cache
            # may ALREADY be a valid boundary: a session forked from a bloc chain
            # holds exactly its fed-token record, and when that record is a true
            # prefix of this prompt the fork IS the boundary. Reusing it turns the
            # first turn of every forked session from a full cold prefill into a
            # restore. The `cache_len == len(fed_ids)` equality is what makes this
            # safe — it excludes the trim-refusal caller (whose cache holds
            # generated reply tokens PAST the record, unreachable without a rewind)
            # and uncountable caches (cache_len None, composition unverifiable).
            if (
                working is None
                and fed_ids
                and cache_len is not None
                and cache_len == len(fed_ids)
                and self._token_lcp_len(fed_ids, new_ids) == len(fed_ids)
                and len(fed_ids) < len(new_ids)
            ):
                working = cache_value
                prefix_len = len(fed_ids)
            if working is None:
                working = self._prompt_cache_backend_create()
                prefix_len = 0
            if working is None:
                _note("bypassed", reason="fresh cache creation failed; generated without a cache")
                return None, full_prompt, None

            boundary_end = max(len(new_ids) - 1, 0)  # keep ≥1 token to seed decode

            # SNAPSHOT BOUNDARY (2026-08-02). Snapshotting the FULL prompt boundary
            # makes the snapshot unusable for every agent loop whose prompt ENDS in
            # per-call ephemeral bytes — a `[loop] iteration N of M.` tail, a
            # fresh-timestamp `<runtime_metadata>` envelope, the generation prompt.
            # The next turn replaces those bytes with the assistant/tool turn that
            # actually happened, so the stored ids stop being a TRUE PREFIX, the
            # forward-only restore above is refused, and every turn pays a full cold
            # prefill (measured: `rebuilt`, cached=0, on 15 of 16 consecutive calls of
            # a real coding loop). A recurrent state cannot be rewound, so the boundary
            # must be chosen conservatively BEFORE the volatile tail.
            #
            # The PREVIOUS turn's recorded prompt says exactly where that is: whatever
            # two consecutive prompts share is stable transcript, and the first
            # divergence is where this turn's ephemeral tail begins. Snapshotting there
            # costs nothing (the same tokens get prefilled either way — only the clone
            # point moves) and leaves a boundary that lands one turn behind instead of
            # nowhere. `prefix_len` is a floor: a snapshot we just restored from is
            # already known-good, never regress below it.
            #
            # When the previous prompt is a TRUE PREFIX of this one there was no
            # rewritten tail (append-only caller, or turn one off a forked prefix), so
            # nothing is gained by holding back — keep the full boundary.
            stable_end = boundary_end
            seed_diverged = False
            shared = 0
            if fed_ids:
                shared = self._token_lcp_len(fed_ids, new_ids)
                if shared < len(fed_ids):
                    # DIVERGED SEED (2026-09-17). The LCP holdback reads `fed_ids` as
                    # the PREVIOUS PROMPT: what two consecutive prompts share is stable
                    # transcript. That is false for a record that was never a prompt —
                    # a key FORKED from a prepared (system, tools) prefix holds exactly
                    # its record (`cache_len == len(fed_ids)`, nothing generated past
                    # it) and has no snapshot yet. When such a seed is not a prefix of
                    # the prompt, `shared` is not a transcript boundary, it is the
                    # point where two RENDERERS disagree. Measured on the gateway
                    # (Qwen3.8-27B-4bit): the seed omitted the reasoning-effort system
                    # line, `shared` was 3 (`<|im_start|>system\n`), the snapshot was
                    # taken at 3 tokens, and turn 2 reported `hit_restore` cached=3 —
                    # a "hit" that prefilled the whole 5.5k prompt again.
                    seed_diverged = (
                        snap is None and cache_len is not None and cache_len == len(fed_ids)
                    )
                    if seed_diverged:
                        self._warn_seed_diverged(key, shared=shared, recorded=len(fed_ids))
                    else:
                        stable_end = max(prefix_len, min(shared, boundary_end))
            if prefix_len > 0 and not restored_from_snapshot:
                # First snapshot-lane turn on a key seeded from the LIVE cache
                # (a bloc fork). The `shared == len(fed_ids)` test above cannot
                # detect a volatile tail here: the record is a BLOC, and a bloc is
                # trivially a prefix of every prompt, so "no rewritten tail" is
                # unproven rather than true. Snapshot the seed boundary — the only
                # content two observations agree on. Without this the very first
                # snapshot carries this turn's ephemeral tail, the next turn's
                # forward-only restore is refused, and turn 2 rebuilds anyway.
                stable_end = prefix_len
            if not fed_ids:
                # NO record at all — turn 1 on a fresh key. The LCP holdback above
                # has nothing to compare against, so it cannot fire, and the
                # boundary defaults to the FULL prompt. That is wrong for one
                # specific reason we can still name exactly: the generation
                # scaffolding this call appended. `_build_prompt_fragment` ends the
                # prompt with `<|im_start|>assistant\n` and, when thinking is
                # explicitly disabled, `<think>\n\n</think>\n\n` — and the NEXT
                # turn's transcript carries the assistant turn that actually
                # happened there instead. A boundary containing it is a prefix of
                # nothing, so the very next restore is refused.
                #
                # MEASURED (hardware, Qwen3.5-4B-MLX-4bit, 10k agent loop): turn 1
                # snapshotted at len-1, turn 2 rebuilt a full 10,118-token prefill
                # (5.46 s), and only turn 3 — the first turn that HAS a record, so
                # the LCP holdback could fire — restored. The divergence was exactly
                # 4 tokens, and `<think>\n\n</think>\n\n` tokenizes to exactly
                # ['<think>', '\n\n', '</think>', '\n\n'] on that tokenizer.
                #
                # Holding those few tokens back costs a handful of fed tokens on
                # turn 2 and removes a full cold prefill per session.
                gen_at = self._generation_prompt_boundary(full_prompt, new_ids)
                if gen_at is not None:
                    stable_end = max(0, min(stable_end, gen_at))
            if seed_diverged:
                # One observation, like turn 1 — but this caller TOLD us its shape by
                # forking a prefix: stable head, rewritten tail. For that shape the
                # generation-prompt holdback is known-useless (the runtime's final
                # turn carries a fresh-timestamp envelope the next call strips), so
                # hold back the whole final turn instead. Renderer-derived, so it
                # cannot disagree with the prompt; absent → the turn-1 rule.
                at = self._stable_head_boundary(stable_head, full_prompt, new_ids)
                if at is None:
                    at = self._generation_prompt_boundary(full_prompt, new_ids)
                if at is not None:
                    stable_end = max(0, min(stable_end, at))

            head = new_ids[prefix_len:stable_end]
            if head and not self._prefill_tokens_into_cache(working, head):
                # Prefill failed: drop the (now-suspect) snapshot and take the
                # plain fresh full feed — correct, just without snapshot savings.
                self._drop_hybrid_snapshot(key)
                return _fresh_full_feed()

            # Snapshot the clean boundary (deepcopy) BEFORE the tail prefill and
            # before generation mutates `working`. One snapshot per key. The
            # deepcopy is skipped ONLY when a snapshot we actually RESTORED from
            # already describes this boundary. `prefix_len == 0` used to stand in
            # for "nothing was restored"; that is wrong once the live cache can
            # seed the lane (prefix_len > 0 with no snapshot behind it → the
            # boundary would go unrecorded and the next turn would rebuild) and it
            # is also wrong when a STALE, non-restorable snapshot is still present.
            boundary_ids = new_ids[:stable_end]
            if boundary_ids and (stable_end > prefix_len or not restored_from_snapshot):
                snap_copy = self._prompt_cache_backend_clone(working)
                if snap_copy is not None:
                    self._store_hybrid_snapshot(key, snap_copy, boundary_ids)
                else:
                    self._drop_hybrid_snapshot(key)
            elif not boundary_ids:
                self._drop_hybrid_snapshot(key)

            tail = new_ids[stable_end:boundary_end]
            if tail and not self._prefill_tokens_into_cache(working, tail):
                self._drop_hybrid_snapshot(key)
                return _fresh_full_feed()
            to_prefill = head + tail

            # Persist `working` as the live key cache (meta/TTL preserved).
            prior_meta = dict(self.prompt_cache_key_meta(key) or {})
            prior_meta.pop(self._FED_TOKEN_IDS_META, None)
            prior_meta.setdefault("backend", "mlx")
            try:
                prior_ttl = self._prompt_cache_store.ttl_s(key)
            except Exception:
                prior_ttl = None
            try:
                self._prompt_cache_store.set(key, working, meta=prior_meta, ttl_s=prior_ttl)
            except Exception:
                pass

            seed = new_ids[boundary_end:] or new_ids  # trailing token(s) to decode from
            # fed = the tokens actually processed THIS turn (suffix prefilled +
            # the decode seed), mirroring the trim path's suffix accounting;
            # cached = the tokens served from the restored snapshot.
            fed_this_turn = len(to_prefill) + len(seed)
            if prefix_len > 0:
                _note("hit_restore", cached=prefix_len, fed=fed_this_turn)
            elif not fed_ids:
                # TURN ONE ON A FRESH KEY is `cold`, not `rebuilt` (parity,
                # 2026-08-07). There was no prior cache to discard, so `rebuilt`
                # named a failure that did not happen — and it read as a thrashing
                # cache on the FIRST call of every hybrid session. The GGUF lane
                # already reported `cold` here; MLX and transformers now agree.
                _note("cold", cached=0, fed=fed_this_turn)
            elif seed_diverged:
                _note(
                    "rebuilt",
                    cached=0,
                    fed=fed_this_turn,
                    reason=(
                        f"prepared prefix is not a prefix of the prompt (shared {shared} of "
                        f"{len(fed_ids)} recorded tokens); prefix cache unreachable, full prefill"
                    ),
                )
            else:
                _note("rebuilt", cached=0, fed=fed_this_turn)
            return working, seed, new_ids

        # UNTRIMMABLE ARCHITECTURE (hybrid Gated-DeltaNet / SSM): the snapshot
        # lane is the ONLY lane that can reuse anything here, so route to it on
        # the ARCHITECTURE, not on the fill state.
        #
        # This check used to live inside `if cold_empty:` alone, which made it
        # unreachable for the shape the runtime actually uses. A session FORKED
        # from a (system, tools) bloc chain is WARM on turn 1 and its cache
        # exactly matches its fed-token record, so `trim_needed` is 0, the trim
        # below never runs, the architecture is never discovered untrimmable, the
        # turn reports `hit_extend` and stores NO snapshot. Turn 2 is then the
        # first call that needs a trim, it is refused, and the snapshot lane finds
        # nothing to restore: `rebuilt`, cached=0, a full cold prefill. That one
        # rebuild is the whole gap between ~47% and ~96% steady-state reuse, and
        # it governs every locally available model above 4B — they are all Gated
        # DeltaNet hybrids on this lane.
        #
        # Trimmable models are unaffected: `_cache_is_trimmable` is architecture-
        # determined (an empty hybrid cache already reports False), so pure
        # attention keeps the plain cold feed and the trim-based delta path.
        if not _is_artifact_backed() and not self._cache_is_trimmable(cache_value):
            return _hybrid_snapshot_feed()

        if cold_empty:
            _note("cold", cached=0, fed=len(new_ids))
            return cache_value, full_prompt, new_ids

        if fed_ids is None:
            # Warm cache of unknown composition under a full-context caller.
            # LOADED ARTIFACTS are excluded from replacement: destroying a
            # verified durable-bloc cache to save a prefill inverts the
            # feature's whole point — bypass the cache for this call instead
            # (P1-5). Other unknown-composition caches rebuild fresh: correct
            # for a caller that re-sends everything, and it kills the double
            # prefill for pre-fix caches too.
            if _is_artifact_backed():
                if key not in self._delta_feed_warned_keys:
                    self._delta_feed_warned_keys.add(key)
                    self.logger.warning(
                        f"#FALLBACK MLX prompt cache '{key}' is a loaded artifact without a fed-token "
                        f"record; full-context calls bypass it (no prefill savings) rather than "
                        f"destroy the verified artifact. Use prompt-only/KV-session flows for "
                        f"artifact caches."
                    )
                _note(
                    "bypassed",
                    cached=0,
                    fed=len(new_ids),
                    reason="loaded artifact without a fed-token record; bypassed to protect the artifact",
                )
                return None, full_prompt, None
            if key not in self._delta_feed_warned_keys:
                self._delta_feed_warned_keys.add(key)
                self.logger.warning(
                    f"#FALLBACK MLX prompt cache '{key}' was warm with unknown token composition "
                    f"under a full-context call; rebuilt fresh (one cold prefill) and now "
                    f"delta-tracked."
                )
            _note(
                "rebuilt",
                cached=0,
                fed=len(new_ids),
                reason="warm cache of unknown token composition; rebuilt fresh",
            )
            return _fresh_full_feed()

        if cache_len is None:
            # Known record but uncountable cache (recurrent-state layers):
            # trim arithmetic is impossible. Same lattice as trim refusal.
            if _is_artifact_backed():
                _note(
                    "bypassed",
                    fed=len(new_ids),
                    reason="artifact cache state is not countable for this architecture; bypassed",
                )
                return None, full_prompt, None
            if key not in self._delta_feed_warned_keys:
                self._delta_feed_warned_keys.add(key)
                self.logger.warning(
                    f"#FALLBACK MLX prompt cache '{key}': cache state is not countable for this "
                    f"architecture (recurrent/array layers); using the snapshot/restore lane "
                    f"(warm turns reuse a copied boundary, no trim)."
                )
            return _hybrid_snapshot_feed()

        lcp = self._token_lcp_len(fed_ids, new_ids)

        if lcp < len(fed_ids) and _is_artifact_backed():
            # Divergent prompt over a verified artifact (0819): honoring it
            # via trim would DEGRADE the shared stable-key cache down to the
            # tiny shared prefix — the next caller of the bloc would find a
            # stub. No savings exist here anyway (the bloc is not this
            # prompt's head): bypass for this call, artifact stays whole.
            if key not in self._delta_feed_warned_keys:
                self._delta_feed_warned_keys.add(key)
                self.logger.warning(
                    f"#FALLBACK MLX prompt cache '{key}': full-context prompt diverges from the "
                    f"artifact's recorded prefix (shared {lcp} of {len(fed_ids)} tokens); bypassed "
                    f"for this call rather than trimming the shared artifact cache."
                )
            _note(
                "bypassed",
                cached=0,
                fed=len(new_ids),
                reason="prompt diverges from the artifact's recorded prefix; bypassed to protect the shared cache",
            )
            return None, full_prompt, None

        if lcp < len(fed_ids) and cache_len == len(fed_ids) and self._key_is_forked(key):
            # Same diverged seed as the snapshot lane. Trimming makes it CORRECT here
            # (the cache is cut back to the shared tokens), which is exactly why it
            # stayed silent: the turn reports `hit_extend` with a handful of cached
            # tokens while the whole prepared prefix is discarded.
            self._warn_seed_diverged(key, shared=lcp, recorded=len(fed_ids))

        effective_prefix = min(lcp, cache_len)
        identical = effective_prefix >= len(new_ids)
        if identical:
            # Identical (or fully-contained) prompt: keep one token to step generation.
            effective_prefix = len(new_ids) - 1

        trim_needed = cache_len - effective_prefix
        if trim_needed > 0 and not self._trim_prompt_cache_tokens(cache_value, trim_needed):
            # Untrimmable cache type (hybrid ArraysCache layers; sliding
            # windows past their fill point) or partial trim: one honest
            # cold prefill on a fresh cache — loudly, once per key (P1-6).
            if _is_artifact_backed():
                _note(
                    "bypassed",
                    fed=len(new_ids),
                    reason="artifact cache type is not trimmable for this architecture; bypassed",
                )
                return None, full_prompt, None
            if key not in self._delta_feed_warned_keys:
                self._delta_feed_warned_keys.add(key)
                self.logger.warning(
                    f"#FALLBACK MLX prompt cache '{key}': cache type is not trimmable for this "
                    f"architecture (recurrent/hybrid layers); using the snapshot/restore lane "
                    f"(warm turns reuse a copied boundary, no trim)."
                )
            return _hybrid_snapshot_feed()

        suffix = new_ids[effective_prefix:]
        _note("hit_full" if identical else "hit_extend", cached=effective_prefix, fed=len(suffix))
        return cache_value, suffix, new_ids

    def _build_prompt_fragment(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        prefilled_modules: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        include_bos: bool = True,
    ) -> str:
        """Build a prompt fragment intended to be appended to an existing prompt_cache."""

        prefilled = set()
        if prefilled_modules:
            for item in prefilled_modules:
                try:
                    norm = str(item or "").strip().lower()
                except Exception:
                    norm = ""
                if norm:
                    prefilled.add(norm)

        base_system_prompt = system_prompt
        tool_system_prompt = None
        if tools and self.tool_handler.supports_prompted and "tools" not in prefilled:
            include_tool_list = True
            if base_system_prompt and "## Tools (session)" in base_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(
                tools, include_tool_list=include_tool_list
            )
            if tool_prompt:
                tool_system_prompt = tool_prompt

        # CANONICAL WHITESPACE. `PromptCacheModule.normalized()` strips
        # system_prompt before it is fingerprinted AND before it is rendered
        # into a bloc cache, and the gemma-turn and plain branches below strip
        # too — but the ChatML branch used to render the caller's RAW string.
        # A system prompt with any leading/trailing whitespace therefore
        # produced one byte stream through the bloc chain (stripped) and a
        # different one through `generate()` (raw), diverging at the FIRST
        # token of the system text: the entire prefix cache became unreachable
        # on the main lane. Stripping here makes the renderer canonical, so
        # both paths agree by construction. (Not cosmetic: this is the second,
        # independent cause of full-prefix loss, red-team find 2026-08-03.)
        if isinstance(base_system_prompt, str):
            base_system_prompt = base_system_prompt.strip() or None
        if isinstance(tool_system_prompt, str):
            tool_system_prompt = tool_system_prompt.strip() or None

        # ONE system turn (parity with the GGUF/transformers builders): when
        # this fragment renders BOTH the user system prompt and the tool
        # instructions, they share a single system block — chat templates are
        # trained on exactly one system turn, and a second consecutive block
        # is out-of-distribution (degraded tool-calling, live find on
        # Ornith-1.0-35B, 2026-07-15).
        #
        # NOTE `prefilled_modules` (the `"system" not in prefilled` guards) is a
        # legacy escape hatch that CANNOT express a mid-turn continuation: it
        # skips the system block and then reopens a second one for the tools.
        # Bloc chains no longer use it — `BaseProvider.prompt_cache_plan_bloc_chain`
        # cuts one cumulative render instead. See
        # tests/providers/test_mlx_single_system_block_unit.py.
        if base_system_prompt and "system" not in prefilled and tool_system_prompt:
            base_system_prompt = f"{base_system_prompt}\n\n{tool_system_prompt}"
            tool_system_prompt = None

        # Reasoning-effort instruction (Qwen3.8-style): the model's own chat template
        # controls effort by prepending a per-level sentence to the FIRST system block
        # (declared in assets as `thinking_control.effort_system_lines`; an empty value
        # means the level renders no text). Rendered only when this fragment owns the
        # system region — a prefilled system bloc was serialized without the line, and
        # injecting it elsewhere would corrupt the planned cut. NOTE: an explicit level
        # therefore changes the system-block bytes, so bloc plans rendered without it
        # will prefix-miss and recompute (correct, just uncached).
        if isinstance(reasoning_effort, str) and reasoning_effort and "system" not in prefilled:
            effort_lines = self._thinking_control_surfaces().effort_system_lines or {}
            effort_line = effort_lines.get(reasoning_effort)
            if isinstance(effort_line, str) and effort_line.strip():
                effort_line = effort_line.strip()
                if base_system_prompt:
                    base_system_prompt = f"{effort_line}\n\n{base_system_prompt}"
                elif tool_system_prompt:
                    tool_system_prompt = f"{effort_line}\n\n{tool_system_prompt}"
                elif (
                    messages
                    and isinstance(messages[0], dict)
                    and str(messages[0].get("role") or "").strip().lower() == "system"
                ):
                    # The conversation carries its own leading system turn: merge the
                    # line into it. A standalone effort block here would render TWO
                    # consecutive system blocks — the out-of-distribution shape this
                    # renderer's single-system-block rule exists to prevent
                    # (adversarial find 2026-08-19). Non-str content gets the same
                    # json coercion `_as_text` applies at render time.
                    first = dict(messages[0])
                    content = first.get("content")
                    if not isinstance(content, str):
                        try:
                            content = json.dumps(content, ensure_ascii=False)
                        except Exception:
                            content = str(content)
                    first["content"] = f"{effort_line}\n\n{content}"
                    messages = [first, *messages[1:]]
                else:
                    base_system_prompt = effort_line

        def _as_text(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        arch_cfg = (
            getattr(self, "architecture_config", None)
            if isinstance(getattr(self, "architecture_config", None), dict)
            else {}
        )
        msg_fmt = str((arch_cfg or {}).get("message_format") or "").strip().lower()
        # ChatML (`<|im_start|>…<|im_end|>`) is driven by the REGISTRY's
        # message_format ("im_start_end"), NOT a model-name substring. The old
        # `"qwen" in model_name` heuristic mis-rendered every ChatML model whose
        # name lacks "qwen" — notably Ornith (arch qwen3_5_agentic, message_format
        # im_start_end) fell through to the plain `role: content` fallback with
        # ZERO ChatML markers on the live generate path. The name substring is
        # kept only as a fallback for a model whose arch config is missing.
        is_chatml = (msg_fmt == "im_start_end") or ("qwen" in self.model.lower())
        is_gemma_turn = msg_fmt == "gemma_turn"
        parts: List[str] = []

        if is_gemma_turn and include_bos:
            bos = str(getattr(getattr(self, "tokenizer", None), "bos_token", "") or "<bos>")
            if bos:
                parts.append(bos)

        if base_system_prompt and "system" not in prefilled:
            if is_chatml:
                parts.append(f"<|im_start|>system\n{base_system_prompt}<|im_end|>\n")
            elif is_gemma_turn:
                parts.append(f"<|turn>system\n{base_system_prompt.strip()}<turn|>\n")
            else:
                parts.append(f"{base_system_prompt.strip()}\n\n")

        if tool_system_prompt:
            if is_chatml:
                parts.append(f"<|im_start|>system\n{tool_system_prompt}<|im_end|>\n")
            elif is_gemma_turn:
                parts.append(f"<|turn>system\n{tool_system_prompt.strip()}<turn|>\n")
            else:
                parts.append(f"{tool_system_prompt.strip()}\n\n")

        if messages:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "user")
                content = _as_text(msg.get("content"))
                if is_chatml:
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
                elif is_gemma_turn:
                    role_name = (
                        "model" if role.strip().lower() == "assistant" else role.strip().lower()
                    )
                    if role_name in {"system", "user", "model"}:
                        parts.append(f"<|turn>{role_name}\n{content.strip()}<turn|>\n")
                else:
                    parts.append(f"{role}: {content}\n")

        if isinstance(prompt, str) and prompt:
            if is_chatml:
                parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n")
            elif is_gemma_turn:
                parts.append(f"<|turn>user\n{prompt.strip()}<turn|>\n")
            else:
                parts.append(f"user: {prompt}\n")

        if add_generation_prompt:
            if is_chatml:
                parts.append("<|im_start|>assistant\n")
                if enable_thinking is False:
                    parts.append("<think>\n\n</think>\n\n")
            elif is_gemma_turn:
                parts.append("<|turn>model\n")
            else:
                parts.append("assistant:")

        return "".join(parts)

    def _prompt_opened_thinking(self, rendered_prompt: Any) -> bool:
        from ..architectures.response_postprocessing import prompt_opens_thinking

        return prompt_opens_thinking(
            rendered_prompt,
            architecture_format=getattr(self, "architecture_config", None),
            model_capabilities=getattr(self, "model_capabilities", None),
        )

    def _postprocess_generated_text(
        self, text: str, *, thinking_opened_by_prompt: bool = False
    ) -> tuple[str, Optional[str]]:
        cleaned, reasoning = normalize_assistant_text(
            str(text or ""),
            architecture_format=getattr(self, "architecture_config", None),
            model_capabilities=getattr(self, "model_capabilities", None),
            thinking_opened_by_prompt=thinking_opened_by_prompt,
        )
        msg_fmt = (
            str((getattr(self, "architecture_config", {}) or {}).get("message_format") or "")
            .strip()
            .lower()
        )
        if msg_fmt == "gemma_turn":
            stop_candidates = []
            cfg = getattr(self, "architecture_config", None)
            if isinstance(cfg, dict):
                suffix = str(cfg.get("assistant_suffix") or "").strip()
                if suffix:
                    stop_candidates.append(suffix)
            stop_candidates.append("<turn|>")
            for stop in stop_candidates:
                idx = cleaned.find(stop)
                if idx >= 0:
                    cleaned = cleaned[:idx].rstrip()
                    break
        return cleaned, reasoning

    def _prompt_cache_backend_append(
        self,
        cache_value: Any,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> bool:
        if cache_value is None:
            return False

        existing_tokens = self._prompt_cache_backend_token_count(cache_value)

        # PLANNED BLOC CUT (bloc composability, see base.py). When the caller is
        # `prompt_cache_prepare_modules` it has already rendered the WHOLE
        # conversation through this provider's own `generate()` renderer and cut
        # it at successor-independent token boundaries. Those ids are fed
        # verbatim: re-rendering this module standalone is exactly the defect
        # that made a two-bloc chain emit two consecutive `<|im_start|>system`
        # blocks and stranded everything past the first bloc.
        planned = kwargs.get("bloc_token_ids")
        fragment_ids: Optional[List[int]] = None
        if isinstance(planned, (list, tuple)):
            try:
                fragment_ids = [int(t) for t in planned]
            except Exception:
                fragment_ids = None

        fragment: Optional[str] = None
        if fragment_ids is None:
            fragment = self._build_prompt_fragment(
                prompt=str(prompt or ""),
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                add_generation_prompt=bool(add_generation_prompt),
                enable_thinking=kwargs.get("_acore_mlx_enable_thinking"),
                reasoning_effort=kwargs.get("_acore_mlx_reasoning_effort"),
                include_bos=not (isinstance(existing_tokens, int) and int(existing_tokens) > 0),
            )
            fragment_ids = self._encode_prompt_token_ids(fragment) if fragment else None

        # Stash for prompt_cache_update's fed-token-id bookkeeping (delta feed,
        # B2): the base method that calls us knows the KEY; we know the exact
        # tokens that were fed. Locked: prepare_modules (base loop) and
        # prompt_cache_update can race on this instance-level stash from
        # different threads — an unlocked write here cross-pollinates records
        # across keys (adversarial find 2026-07-13). The ID stash is what the
        # record actually uses; the text stash stays for the legacy lane.
        with self._append_stash_lock:
            self._pending_append_fragment = fragment or None
            self._pending_append_fragment_ids = list(fragment_ids) if fragment_ids else None
            self._pending_append_precount = int(existing_tokens or 0)
        if not fragment_ids:
            return True

        # EXACT-BOUNDARY PREFILL (prompt-cache record truth, 2026-08-02).
        #
        # `generate_step(max_tokens=0)` runs the whole prefill (the chunked loop
        # plus the final single-token `_step`) and stops BEFORE yielding, so the
        # cache lands exactly on the fragment boundary with nothing to trim.
        # The legacy lane below samples one token, and mlx_lm feeds that sampled
        # token back into the cache on the next `_step` — hence the `trim(1)`.
        # That trim is a SILENT NO-OP on untrimmable architectures (hybrid
        # linear-attention / SSM: an `ArraysCache` layer reports
        # `is_trimmable() == False`, so mlx_lm returns 0 WITHOUT raising and the
        # `except` below never fires). The cache then sits exactly ONE token
        # ahead of the fragment it claims to hold, `_prompt_cache_append_record_meta`'s
        # precount guard correctly refuses to describe it, the module chain's
        # final prefix cache carries NO `fed_token_ids`, and every session forked
        # from it degrades to "warm cache of unknown token composition; rebuilt
        # fresh" — a full cold prefill on EVERY turn. Same token ids, one fewer
        # forward pass.
        if self._prefill_tokens_into_cache(cache_value, fragment_ids):
            return True

        if fragment is None:
            # A planned bloc cut has no standalone text form (it is a slice of
            # the whole conversation), so the token-sampling fallback below —
            # which takes a STRING — cannot express it. Refusing here keeps the
            # cache honest; `prepare_modules` reports the failure.
            self.logger.warning(
                "#FALLBACK MLX prompt cache: exact-boundary prefill unavailable for a planned "
                "bloc fragment; module preparation cannot proceed on this backend."
            )
            return False

        try:
            from mlx_lm.models.cache import trim_prompt_cache
        except Exception:
            trim_prompt_cache = None

        # Best-effort prefill: MLX-LM generates at least one token; trim it to end exactly at the fragment boundary.
        generated = 0
        try:
            gen = self.stream_generate_fn(
                self.llm,
                self.tokenizer,
                prompt=fragment,
                prompt_cache=cache_value,
                max_tokens=1,
            )
            for _chunk in gen:
                generated += 1
        except TypeError:
            try:
                gen = self.stream_generate_fn(
                    self.llm,
                    self.tokenizer,
                    fragment,
                    prompt_cache=cache_value,
                    max_tokens=1,
                )
                for _chunk in gen:
                    generated += 1
            except Exception:
                return False
        except Exception:
            return False

        if trim_prompt_cache is not None and generated > 0:
            try:
                trim_prompt_cache(cache_value, generated)
            except Exception:
                pass

        return True

    def _prompt_cache_append_record_meta(
        self, prior_meta: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Keep module-chain caches (prepare_modules) delta-capable.

        The backend append just stashed the exact fragment it fed and the
        pre-append token count. When the chain's prior record exactly
        describes the pre-append cache, the new module cache's record is
        `prior + fragment_ids` — so the FINAL prefix key (the one sessions
        fork from) carries a true fed-token-id record and the generate-side
        delta feed can engage instead of falling to fresh-rebuild
        (unknown-composition lane). Any uncertainty → no record (honest
        unknown), never a guess.
        """
        with self._append_stash_lock:
            fragment = self._pending_append_fragment
            stashed_ids = self._pending_append_fragment_ids
            precount = int(self._pending_append_precount or 0)
            self._pending_append_fragment = None
            self._pending_append_fragment_ids = None
            self._pending_append_precount = 0

        prior_ids_raw = (prior_meta or {}).get(self._FED_TOKEN_IDS_META)
        prior_ids: Optional[List[int]] = None
        if isinstance(prior_ids_raw, list) and prior_ids_raw:
            try:
                prior_ids = [int(t) for t in prior_ids_raw]
            except Exception:
                prior_ids = None

        if precount > 0 and (prior_ids is None or len(prior_ids) != precount):
            return None  # unknown or stale head: refuse to describe it
        # The ID stash is authoritative — a planned bloc fragment is a token
        # slice, and re-encoding its text (if it even has one) is not the same
        # operation.
        fragment_ids = list(stashed_ids or [])
        if not fragment_ids and fragment:
            fragment_ids = self._encode_prompt_token_ids(fragment) or []
        if not fragment_ids:
            return {self._FED_TOKEN_IDS_META: list(prior_ids)} if prior_ids else None
        return {self._FED_TOKEN_IDS_META: (prior_ids or []) + fragment_ids}

    def prompt_cache_update(self, key: str, **kwargs) -> bool:
        """Append context into a cache key + keep the fed-token-id record true.

        The delta-feed path (B2) can only trust a warm cache whose fed token
        ids are known. The backend append stashes the exact fragment it fed;
        here (where the KEY is known) the record extends. Caches whose head
        predates tracking stay unrecorded — the generate path then keeps
        legacy behavior for them rather than trusting a partial record.
        """
        with self._append_stash_lock:
            self._pending_append_fragment = None
            self._pending_append_fragment_ids = None
            self._pending_append_precount = 0
            ok = super().prompt_cache_update(key, **kwargs)
            fragment = self._pending_append_fragment
            stashed_ids = self._pending_append_fragment_ids
            precount = int(self._pending_append_precount or 0)
            self._pending_append_fragment = None
            self._pending_append_fragment_ids = None
            self._pending_append_precount = 0
        if not ok or not (fragment or stashed_ids):
            return ok
        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            return ok
        prior_ids = self._fed_token_ids_for_key(normalized)
        if precount > 0 and prior_ids is None:
            # Unknown head: appending a known tail cannot make the record whole.
            return ok
        if prior_ids is not None and precount != len(prior_ids):
            # Generated tokens (or anything else) sit between the record and
            # this fragment: `prior + fragment` would misdescribe a cache that
            # actually holds `prior + gap + fragment` (adversarial find P1-3).
            # The old record stands — still a true prefix, still trimmable-to.
            return ok
        fragment_ids = list(stashed_ids or [])
        if not fragment_ids and fragment:
            fragment_ids = self._encode_prompt_token_ids(fragment) or []
        if fragment_ids:
            self._record_fed_token_ids(normalized, (prior_ids or []) + fragment_ids)
        return ok

    def prompt_cache_set(
        self,
        key: str,
        *,
        make_default: bool = True,
        warm_prompt: Optional[str] = None,
        ttl_s: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """Create/reset a prompt cache for the given key (best-effort)."""
        _ = kwargs
        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            return False
        if not super().prompt_cache_set(normalized, make_default=make_default):
            return False

        try:
            from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        except Exception:
            return False

        cache_obj = make_prompt_cache(self.llm)

        # Best-effort warm: MLX-LM always generates at least 1 token, so we trim it back.
        warmed_ids: Optional[List[int]] = None
        if isinstance(warm_prompt, str) and warm_prompt.strip():
            try:
                gen = self.stream_generate_fn(
                    self.llm,
                    self.tokenizer,
                    prompt=warm_prompt,
                    prompt_cache=cache_obj,
                    max_tokens=1,
                )
                for _ in gen:
                    break
                try:
                    trim_prompt_cache(cache_obj, 1)
                except Exception:
                    pass
                warmed_ids = self._encode_prompt_token_ids(warm_prompt)
            except Exception:
                pass

        try:
            meta: Dict[str, Any] = {"backend": "mlx"}
            if warmed_ids:
                # Fed-token-id record from birth (delta feed, B2) — a fresh
                # empty cache needs no record; the generate path starts one.
                meta[self._FED_TOKEN_IDS_META] = warmed_ids
            self._prompt_cache_store.set(normalized, cache_obj, ttl_s=ttl_s, meta=meta)
        except Exception:
            return False
        return True

    def prompt_cache_save(
        self,
        key: str,
        filename: str,
        *,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Save an MLX KV prompt cache to a `.safetensors` file (model-locked; best-effort)."""
        if getattr(self, "_mtp_processor", None) is not None:
            raise ProviderAPIError("Native MLX automatic prefix caches do not implement manual prompt_cache_save; use the managed SSD tier with mlx_batching=True")
        _ = kwargs
        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            raise ValueError("prompt cache key must be a non-empty string")

        cache_obj = self._prompt_cache_store.get(normalized)
        if cache_obj is None:
            raise ValueError(f"prompt cache key '{normalized}' does not exist")

        try:
            from mlx_lm.models.cache import save_prompt_cache
        except Exception as e:
            raise ImportError(
                'MLX prompt cache saving requires mlx-lm (install: `pip install "abstractcore[mlx]"`).'
            ) from e

        out_meta: Dict[str, Any] = dict(meta or {})
        out_meta.setdefault("format", "abstractcore-prompt-cache/v1")
        out_meta.setdefault("provider", str(getattr(self, "provider", "mlx")))
        out_meta.setdefault("model", str(getattr(self, "model", "")))
        resolved_model_id = str(getattr(self, "_resolved_model_id", "") or "").strip()
        if resolved_model_id:
            out_meta.setdefault("model_resolved_id", resolved_model_id)
        out_meta.setdefault("saved_at", datetime.now().isoformat())
        # 0817 axis 2: record the tokenizer identity whose token-id stream this
        # cache encodes, for EVERY saved artifact (bloc lane passes it in meta;
        # plain keyed saves get it here) — the load gate checks it.
        tokenizer_fp = self.prompt_cache_tokenizer_fingerprint()
        if tokenizer_fp:
            out_meta.setdefault("tokenizer_fingerprint", tokenizer_fp)
        # 0817 axis 3: record the KV-geometry identity (rope/window/position
        # config keys) the cached tensors were computed under.
        model_config_fp = self.prompt_cache_model_config_fingerprint()
        if model_config_fp:
            out_meta.setdefault("model_config_fingerprint", model_config_fp)
        # 0817 axis 4: record the cheap weights identity that computed the
        # cached tensors (checkpoint swaps under the same id must refuse).
        weights_fp = self.prompt_cache_weights_fingerprint()
        if weights_fp:
            out_meta.setdefault("weights_fingerprint", weights_fp)

        try:
            tok = self._prompt_cache_backend_token_count(cache_obj)
            if isinstance(tok, int) and tok >= 0:
                out_meta.setdefault("token_count", tok)
        except Exception:
            pass

        # Persist the fed-token-id record into the artifact (0819, adversary
        # P0-1): the record is the bookkeeping the whole delta lane rides on,
        # and dropping it at this boundary made every artifact-backed cache
        # "warm-unknown" — load cost + a FULL re-prefill under full-context
        # callers (negative value). The freeze invariant guarantees the store
        # record is a TRUE token-prefix of the cache at save time, so it can
        # be trusted verbatim at load.
        record_ids = self._fed_token_ids_for_key(normalized)
        if record_ids:
            out_meta.setdefault(self._FED_TOKEN_IDS_META, record_ids)

        cache_to_save = cache_obj
        if q8:
            # No silent fp fallback (0817 axis 5): a caller that asked for q8
            # gets q8 or a loud error — falling back quietly stores fp bytes
            # under metadata/manifests claiming q8 (the silent-wrong-label
            # class). Hybrid/recurrent architectures carry per-layer cache
            # objects (e.g. mlx_lm ArraysCache for state-space layers) that
            # expose no to_quantized; those stacks cannot store q8.
            try:
                cache_to_save = [layer.to_quantized(group_size=64, bits=8) for layer in cache_obj]
            except AttributeError as e:
                unsupported = sorted(
                    {
                        type(layer).__name__
                        for layer in cache_obj
                        if not callable(getattr(layer, "to_quantized", None))
                    }
                )
                raise ValueError(
                    "q8 prompt-cache storage is not supported for this model's cache stack: "
                    f"layer cache type(s) {unsupported or ['unknown']} expose no to_quantized "
                    "(hybrid/recurrent architectures store per-layer state, not quantizable KV). "
                    "Save with q8=False (fp) for this model."
                ) from e
            out_meta["quantized"] = "q8"

        # mlx_lm saves KV caches via safetensors metadata, which requires string keys + values.
        def _meta_value(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                if isinstance(value, (dict, list, tuple)):
                    return json.dumps(value, ensure_ascii=False)
            except Exception:
                pass
            return str(value)

        out_meta_str: Dict[str, str] = {
            str(k): _meta_value(v) for k, v in out_meta.items() if isinstance(k, str) and k
        }

        save_prompt_cache(str(filename), cache_to_save, metadata=out_meta_str)

        return {
            "supported": True,
            "operation": "save",
            "provider": str(getattr(self, "provider", "mlx")),
            "model": str(getattr(self, "model", "")),
            "key": normalized,
            "filename": str(filename),
            "meta": out_meta_str,
        }

    def prompt_cache_load(
        self,
        filename: str,
        *,
        key: Optional[str] = None,
        make_default: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load an MLX KV prompt cache from a `.safetensors` file (model-locked; best-effort)."""
        if getattr(self, "_mtp_processor", None) is not None:
            raise ProviderAPIError("Native MLX automatic prefix caches do not implement manual prompt_cache_load; use the managed SSD tier with mlx_batching=True")
        _ = kwargs
        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except Exception as e:
            raise ImportError(
                'MLX prompt cache loading requires mlx-lm (install: `pip install "abstractcore[mlx]"`).'
            ) from e

        loaded_cache, meta = load_prompt_cache(str(filename), return_metadata=True)
        meta_dict: Dict[str, Any] = dict(meta or {}) if isinstance(meta, dict) else {}

        required_ids = {
            str(meta_dict.get("model") or "").strip(),
            str(meta_dict.get("model_id") or "").strip(),
            str(meta_dict.get("model_resolved_id") or "").strip(),
        }
        required_ids.discard("")
        current_ids = {str(getattr(self, "model", "") or "").strip()}
        resolved_model_id = str(getattr(self, "_resolved_model_id", "") or "").strip()
        if resolved_model_id:
            current_ids.add(resolved_model_id)
        current_ids.discard("")
        if required_ids and not (required_ids & current_ids):
            raise ValueError(
                "Prompt cache model mismatch: "
                f"cache expects one of {sorted(required_ids)!r}, current provider is {sorted(current_ids)!r}."
            )

        if not required_ids:
            # Best-effort structural check: layer count mismatch is a strong signal of wrong model.
            try:
                expected = self._prompt_cache_backend_create()
                if isinstance(expected, (list, tuple)) and isinstance(loaded_cache, (list, tuple)):
                    if len(expected) != len(loaded_cache):
                        raise ValueError(
                            "Prompt cache appears incompatible with the current model (layer count mismatch)."
                        )
            except Exception:
                pass

        # Tokenizer-identity gate (0817, axis 2): this is the gate that runs
        # WITH the tokenizer available (ensure-time validation abstains when
        # the model is not loaded). The artifact encodes one tokenizer's
        # token-id stream; a tokenizer/chat-template refresh under the same
        # model id makes it silently wrong — refuse loudly, never reload.
        from .tokenizer_fingerprint import check_tokenizer_fingerprint

        stored_tokenizer = str(meta_dict.get("tokenizer_fingerprint") or "").strip()
        current_tokenizer = self.prompt_cache_tokenizer_fingerprint()
        verdict = check_tokenizer_fingerprint(stored_tokenizer, current_tokenizer)
        if verdict == "mismatch":
            raise ValueError(
                "Prompt cache tokenizer mismatch: the artifact encodes the token stream of "
                f"tokenizer '{stored_tokenizer}', but the current tokenizer is '{current_tokenizer}' "
                "(tokenizer.json or chat template changed under the same model id). Reloading it "
                "would inject misaligned KV — recompile the artifact (force_rebuild=True) instead."
            )
        if verdict == "unverified_stored" and current_tokenizer:
            self.logger.warning(
                f"#FALLBACK MLX prompt cache artifact '{filename}' carries no tokenizer "
                f"fingerprint (pre-axis artifact); loading it UNVERIFIED against the current "
                f"tokenizer '{current_tokenizer}'. Re-save to pin it."
            )
        elif verdict == "unverified_current" and stored_tokenizer:
            self.logger.warning(
                f"#FALLBACK MLX prompt cache artifact '{filename}' pins tokenizer "
                f"'{stored_tokenizer}' but the current tokenizer state is unavailable "
                f"(model not loaded); loading UNVERIFIED."
            )

        # Model-config geometry gate (0817, axis 3): the artifact's K/V were
        # computed under one rope/window/position geometry; a config.json edit
        # under the same model id re-defines what they mean. Refuse loudly,
        # never reload.
        from .model_config_fingerprint import check_model_config_fingerprint

        stored_config = str(meta_dict.get("model_config_fingerprint") or "").strip()
        current_config = self.prompt_cache_model_config_fingerprint()
        config_verdict = check_model_config_fingerprint(stored_config, current_config)
        if config_verdict == "mismatch":
            raise ValueError(
                "Prompt cache model-config mismatch: the artifact's KV tensors were computed "
                f"under config geometry '{stored_config}', but the current model config is "
                f"'{current_config}' (rope/window/position keys changed under the same model id). "
                "Reloading it would inject positionally-wrong KV — recompile the artifact "
                "(force_rebuild=True) instead."
            )
        if config_verdict == "unverified_stored" and current_config:
            self.logger.warning(
                f"#FALLBACK MLX prompt cache artifact '{filename}' carries no model-config "
                f"fingerprint (pre-axis artifact); loading it UNVERIFIED against the current "
                f"config '{current_config}'. Re-save to pin it."
            )
        elif config_verdict == "unverified_current" and stored_config:
            self.logger.warning(
                f"#FALLBACK MLX prompt cache artifact '{filename}' pins model-config "
                f"'{stored_config}' but the current config is unavailable "
                f"(model not loaded); loading UNVERIFIED."
            )

        # Weights-identity gate (0817, axis 4): the artifact's tensors were
        # computed BY one set of weights; a checkpoint swap under the same
        # model id leaves every other axis identical. Refuse loudly.
        from .weights_fingerprint import check_weights_fingerprint

        stored_weights = str(meta_dict.get("weights_fingerprint") or "").strip()
        current_weights = self.prompt_cache_weights_fingerprint()
        weights_verdict = check_weights_fingerprint(stored_weights, current_weights)
        if weights_verdict == "mismatch":
            raise ValueError(
                "Prompt cache weights mismatch: the artifact's KV tensors were computed by "
                f"weights '{stored_weights}', but the current weights are '{current_weights}' "
                "(checkpoint swapped under the same model id). Reloading it would inject KV "
                "from weights that are gone — recompile the artifact (force_rebuild=True) instead."
            )
        if weights_verdict == "unverified_stored" and current_weights:
            self.logger.warning(
                f"#FALLBACK MLX prompt cache artifact '{filename}' carries no weights "
                f"fingerprint (pre-axis artifact); loading it UNVERIFIED against the current "
                f"weights '{current_weights}'. Re-save to pin it."
            )
        elif weights_verdict == "unverified_current" and stored_weights:
            self.logger.warning(
                f"#FALLBACK MLX prompt cache artifact '{filename}' pins weights "
                f"'{stored_weights}' but the current weights are unavailable "
                f"(model not loaded); loading UNVERIFIED."
            )

        new_key = key
        normalized = self._normalize_prompt_cache_key(new_key) if new_key is not None else None
        if normalized is None:
            normalized = f"cache:{uuid.uuid4().hex[:12]}"

        store_meta: Dict[str, Any] = {
            "backend": "mlx",
            "loaded_from": str(filename),
        }
        store_meta.update(meta_dict)
        # A loaded artifact IS the artifact, whatever its saved meta happens to
        # say. If the cache was forked before it was saved, `forked_from` rode
        # along into the file — and `_is_artifact_backed` reads that field to tell
        # a private copy from the real thing, so leaving it here would strip this
        # key of the protection it genuinely needs.
        store_meta.pop("forked_from", None)
        live_count: Optional[int] = None
        try:
            tok = self._prompt_cache_backend_token_count(loaded_cache)
            if isinstance(tok, int) and tok >= 0:
                live_count = tok
                store_meta.setdefault("token_count", tok)
        except Exception:
            pass

        # Reconstruct the fed-token-id record persisted at save (0819):
        # safetensors metadata is string-valued, so the record arrives as a
        # JSON string and must become a real int list for the delta lane to
        # read it (`_fed_token_ids_for_key` refuses non-lists). Admission is
        # verified, never assumed: a record LONGER than the loaded cache
        # cannot be a true token-prefix (misdescription — the class this
        # whole lane exists to prevent), so it is dropped loudly and the
        # artifact keeps the protective bypass. A record shorter than the
        # cache is legitimate (the freeze invariant: generated tokens beyond
        # the record stay unrecorded) and the LCP/trim arithmetic handles the
        # tail. Uncountable caches keep the record inert — the delta lattice
        # already bypasses artifact-backed uncountable caches.
        parsed_record = self._parse_persisted_fed_token_ids(
            store_meta.get(self._FED_TOKEN_IDS_META)
        )
        store_meta.pop(self._FED_TOKEN_IDS_META, None)
        if parsed_record is not None:
            if live_count is not None and len(parsed_record) > live_count:
                self.logger.warning(
                    f"#FALLBACK MLX prompt cache artifact '{filename}': persisted fed-token record "
                    f"({len(parsed_record)} ids) is longer than the loaded cache ({live_count} tokens) "
                    f"— record dropped; full-context calls will bypass this cache rather than risk "
                    f"a wrong generation."
                )
            else:
                store_meta[self._FED_TOKEN_IDS_META] = parsed_record

        self._prompt_cache_store.set(normalized, loaded_cache, meta=store_meta)
        if make_default:
            self._default_prompt_cache_key = normalized

        return {
            "supported": True,
            "operation": "load",
            "provider": str(getattr(self, "provider", "mlx")),
            "model": str(getattr(self, "model", "")),
            "key": normalized,
            "filename": str(filename),
            "meta": store_meta,
        }

    # ------------------------------------------------------------------
    # Native MTP (multi-token prediction) speculative decoding
    # ------------------------------------------------------------------

    def _plan_mtp_lane(self, load_target) -> Optional[Dict[str, Any]]:
        """Decide whether this session should run the MTP lane, and with what.

        Returns the resolved plan, or None to leave the ordinary mlx-lm lane
        untouched. Every "no" that follows an explicit request goes through
        `speculation_unavailable`, so it is warned with a named reason (or
        raised, under `require_acceleration=True`) rather than silently dropped.
        """
        request = self._speculation_request
        if request is None or not request.enabled:
            return None

        block = capability_speculation(self.model_capabilities, "mlx") or getattr(self, "_speculation_artifact", None)
        drafter_id = request.drafter or (block or {}).get("drafter")
        if not drafter_id:
            # Deliberately not a guess: the MLX head ships as its own repo whose
            # name we cannot derive from the target's (mlx-community publishes
            # `<model>-MTP-4bit` for some models and nothing for others), and
            # `text_config.mtp_num_hidden_layers` is present even in checkpoints
            # that carry no MTP tensors, so it is not evidence either.
            self._mtp_outcome_at_load = speculation_unavailable(
                request,
                "no_mtp_drafter_for_model",
                f"no MLX MTP drafter is known for '{self.model}'. Pass "
                "speculation={'drafter': '<repo>'} or use a model whose registry "
                "entry lists an mlx drafter",
                logger=self.logger,
            )
            return None

        try:
            import mlx_vlm  # noqa: F401
        except ImportError as exc:
            # Name the INTERPRETER and the real exception. "install mlx-vlm" on
            # its own sent someone chasing a package that was already installed
            # -- in their shell. This repo carries a `.venv` whose console
            # scripts shadow the pyenv ones, and mlx-vlm is present in one and
            # not the other, so the environment is the diagnosis, not the hint.
            import sys as _sys

            self._mtp_outcome_at_load = speculation_unavailable(
                request,
                "mlx_vlm_missing",
                "native MTP on MLX runs through mlx-vlm (mlx-lm strips `mtp.` "
                f"weights on load), but importing it failed under {_sys.executable}: "
                f"{type(exc).__name__}: {exc}. Install it into THAT interpreter: "
                f'"{_sys.executable}" -m pip install "abstractcore[mlx]"  '
                "(if mlx-vlm works in your shell, you are running a different "
                "interpreter than you think -- check `which -a abstractcore-chat`)",
                logger=self.logger,
            )
            return None

        if block is not None:
            spec = self.model_capabilities.get("speculation") or {}
            if isinstance(spec, dict) and spec.get("output_preserving") is False:
                self._mtp_output_preserving = False
        block_size = (
            request.num_draft_tokens
            or (block or {}).get("block_size")
            # The drafter's own config carries a block_size, but it is an
            # architecture/training property, NOT a tuned runtime value --
            # measured, Qwen3.5-4B's drafter declares 4 and 4 is the one setting
            # with no speedup at all (0.98x, against 1.16x at 2). Leaving this
            # None lets mlx-vlm fall back to that config value.
            or None
        )
        if getattr(self, "_speculation_inherits_config", False):
            from .speculation import _local_model_directory
            cached_head = _local_model_directory(str(drafter_id))
            if cached_head is None or not any(cached_head.glob("*.safetensors")):
                self._mtp_outcome_at_load = speculation_unavailable(
                    request, "mtp_head_not_cached",
                    self._companion_missing_message(str(drafter_id)),
                    logger=self.logger,
                )
                return None
            drafter_id = str(cached_head)
        plan = {"drafter": str(drafter_id), "block_size": block_size}
        if getattr(self, "_speculation_inherits_config", False):
            plan["drafter_id"] = request.drafter or (block or {}).get("drafter")
        return plan

    def _companion_missing_message(self, drafter_id: str) -> str:
        """The words a user sees when the model's MTP companion is not on disk."""
        from pathlib import Path as _Path

        model = str(self.model)
        # A model loaded from a local directory has no repo id to re-download by.
        also = (
            ""
            if _Path(model).expanduser().is_dir()
            else f" (downloading the model with `abstractcore models download mlx {model}` or from "
            "the gateway's Models tab fetches it too)"
        )
        return (
            f"MTP acceleration off: companion {drafter_id} (the MTP head for {model}) "
            f"is not downloaded; download it with `abstractcore models download mlx {drafter_id}`"
            f"{also}. The model runs without MTP until then."
        )

    def _native_family_label(self) -> str:
        """`<model> (model_type <t>)` for native-lane error text -- never a guessed family."""
        model_type = getattr(self, "_native_model_type", None)
        if model_type is None:
            session = getattr(self, "_native_qwen4", None)
            model_type = "qwen4_exp" if session is not None else None
        label = str(getattr(self, "model", "") or "this model")
        return f"{label} (model_type {model_type})" if model_type else label

    def _enter_mtp_lane(self, load_target, plan: Dict[str, Any]) -> bool:
        """Load target + drafter through mlx-vlm. True when the lane is live.

        On any failure this returns False and the caller falls back -- to the
        drafter-less mlx-vlm lane for an MTP-preserving checkpoint, else to the
        ordinary mlx-lm load -- so a broken drafter costs a warning rather than
        an unusable provider, unless the caller demanded acceleration, in which
        case `speculation_unavailable` raises.

        `plan["drafter"] is None` is the DRAFTER-LESS native lane (batching, or
        an MTP-preserving checkpoint mlx-lm would corrupt). It has no fallback:
        a failure raises ProviderAPIError naming the model and the reason
        (`plan["why"]`), never a silent mlx-lm load.
        """
        import os
        from contextlib import redirect_stdout, redirect_stderr

        request = self._speculation_request
        drafter_id = plan["drafter"]
        try:
            from .mlx_native_session import load_native_session

            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    # Drafter FIRST, deliberately. It is the cheap, fallible
                    # half -- 256 MB against the target's 15 GB, and the half
                    # that fails (a repo that isn't cached, a kind that won't
                    # resolve). Any failure here returns False and the caller
                    # falls back to the ordinary mlx-lm load, which loads the
                    # target again: loading the target first would spend 15 GB,
                    # discard it, and spend 15 GB more.
                    #
                    # Use the kind the loader RESOLVES from the drafter's
                    # config, never a caller-supplied one -- load_drafter's own
                    # docstring says the resolved value is the dispatch key.
                    session = load_native_session(str(load_target), drafter_id)
                    drafter, kind = session.drafter, session.draft_kind
                    model, processor = session.model, session.processor
        except Exception as exc:
            if drafter_id is None:
                why = plan.get("why") or "this lane loads through mlx-vlm"
                raise ProviderAPIError(
                    f"MLX model '{self.model}' could not be loaded: {why}, and the mlx-vlm "
                    f"load of '{load_target}' raised {type(exc).__name__}: {exc}. It is not "
                    "loaded through mlx-lm instead."
                ) from exc
            if isinstance(exc, ModelNotFoundError):
                message = self._companion_missing_message(str(drafter_id))
            else:
                message = f"could not load MLX MTP drafter '{drafter_id}': {exc}"
            self._mtp_outcome_at_load = speculation_unavailable(
                request,
                "mtp_drafter_load_failed",
                message,
                logger=self.logger,
            )
            return False

        self.llm = model
        self._mtp_processor = processor
        # The provider only ever calls `.encode` on self.tokenizer (4 sites), so
        # handing it the processor's inner tokenizer keeps those byte-identical
        # while mlx-vlm gets the full processor it requires -- it rejects a bare
        # mlx-lm tokenizer with "has no attribute stopping_criteria".
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self._mtp_drafter = drafter
        self._mtp_kind = kind
        self._mtp_drafter_id = plan.get("drafter_id") or drafter_id
        self._mtp_block_size = plan.get("block_size")
        self.generate_fn = self._mtp_generate_fn
        self.stream_generate_fn = self._mtp_stream_generate_fn
        self._bind_native_session(session)
        self.logger.info(
            f"mlx: native VLM lane active (drafter={drafter_id}, kind={kind}, "
            f"block_size={self._mtp_block_size or 'from drafter config'})"
        )
        return True

    def _apply_per_call_speculation(self, value) -> None:
        """Reconcile a per-call `speculation=` with the lane that is loaded.

        Three outcomes, no fourth:
          - `off`  -> honored; this call skips the drafter even if the lane is up.
          - matches the loaded lane -> honored, nothing to do.
          - asks for something the loaded lane is not -> `speculation_is_load_time`
            (warned, or raised under `require_acceleration=True`).
        """
        # Both fields are PER CALL and reset here on every request. Writing a
        # per-call refusal into provider-lifetime state let call 1's rejected
        # drafter name leak into call 2's metadata -- a block whose whole job is
        # to say what actually ran, naming a checkpoint that was never loaded.
        self._mtp_call_disabled = False
        self._mtp_call_outcome = None
        self._mtp_last_used = False
        self._mtp_call_stats = {}
        self._native_sampling_kwargs = {}
        self._native_media = []
        self._native_media_records = []
        self._native_media_report = None
        self._native_media_delivered = set()
        self._mtp_last_result = None
        self._mtp_last_apc = None
        self._native_cache_key = None
        self._mtp_call_block_size = getattr(self, "_mtp_block_size", None)
        default = getattr(self, "_speculation_request", None)
        if getattr(self, "_speculation_inherits_config", False):
            from .speculation import configured_speculation_default
            # Config owns intent; existing session owns readiness. Never reload
            # or mutate another request's defaults when the console changes.
            policy = configured_speculation_default(
                config_file=getattr(self, "_abstractcore_config_file", None),
                capability_defaults=getattr(self, "_abstractcore_capability_defaults", None),
            ) if getattr(self, "_speculation_default_supported", False) else False
            default = normalize_speculation_request(policy if policy is not None else False)
        request = resolve_speculation_request(default, value)
        self._mtp_call_request = request
        if request is None:
            return

        if not request.enabled:
            # Honorable per call: not passing the drafter kwargs is all it takes.
            self._mtp_call_disabled = True
            if self._mtp_active:
                self._mtp_call_outcome = SpeculationOutcome(
                    requested=False,
                    mode="off",
                    used=False,
                    reason="disabled_for_this_call",
                    drafter=getattr(self, "_mtp_drafter_id", None),
                )
            return

        if self._mtp_active:
            if request.num_draft_tokens is not None:
                self._mtp_call_block_size = request.num_draft_tokens
            wanted = request.drafter
            if wanted and wanted != getattr(self, "_mtp_drafter_id", None):
                self._mtp_call_disabled = True
                self._mtp_call_outcome = speculation_unavailable(
                    request,
                    "speculation_is_load_time",
                    f"this provider loaded drafter "
                    f"'{getattr(self, '_mtp_drafter_id', None)}'; switching to "
                    f"'{wanted}' needs a new provider because the drafter is "
                    "bound to the target at load time",
                    logger=self.logger,
                )
            return

        # Lane not loaded: cannot be turned on mid-flight at any price.
        held = getattr(self, "_mtp_outcome_at_load", None)
        if held is not None and held.reason:
            # A constructor-time request already failed for a NAMED reason
            # (missing drafter, missing mlx-vlm). Re-raise under a strict
            # per-call request rather than replacing the better diagnosis.
            if request.require_acceleration:
                raise SpeculationUnavailableError(
                    f"speculation could not be enabled: {held.reason} "
                    "(speculation.require_acceleration=True)",
                    reason=held.reason,
                )
            return

        self._mtp_call_outcome = speculation_unavailable(
            request,
            "speculation_is_load_time",
            "speculation must be requested when the provider is created -- it "
            "selects the runtime that loads the weights. Pass "
            "speculation={'mode': 'native_mtp'} to create_llm(...)",
            logger=self.logger,
        )

    @property
    def _mtp_active(self) -> bool:
        # getattr, not attribute access: the prompt-cache unit harnesses build
        # this provider with `__new__` and never run `__init__` (see the note on
        # the vision-addon fields), so every MTP accessor has to read as "off"
        # on an instance that was never initialised.
        return getattr(self, "_mtp_drafter", None) is not None

    def _mtp_kwargs(self, input_embeddings) -> Dict[str, Any]:
        """Drafter kwargs for one call, or {} when this call must not use it.

        The legacy vision add-on's `input_embeddings` are mlx-lm-shaped and
        unsupported by this adapter. Native Qwen4 images use prepared pixel
        tensors instead and can use MTP after the vision prefill.
        """
        if (
            not self._mtp_active
            or input_embeddings is not None
            or getattr(self, "_mtp_call_disabled", False)
        ):
            # Record the per-call verdict here, the ONE place that decides it,
            # so the streaming lane can report it too. Streaming yields plain
            # content chunks with no usage or metadata, so without this a
            # streamed turn could only guess -- and guessing "not used" while
            # the drafter ran is exactly the kind of quiet lie this contract
            # exists to prevent.
            self._mtp_last_used = False
            return {}
        kw = {
            "draft_model": self._mtp_drafter,
            "draft_kind": getattr(self, "_mtp_kind", None),
        }
        block = getattr(self, "_mtp_call_block_size", getattr(self, "_mtp_block_size", None))
        if block:
            kw["draft_block_size"] = int(block) + 1
        return kw

    def _mtp_stats_snapshot(self):
        draft = getattr(self, "_mtp_drafter", None)
        if draft is None or getattr(self, "_mtp_call_disabled", False):
            return None
        return tuple(getattr(draft, "speculative_total_" + key, 0) for key in ("rounds", "accepted", "drafted"))

    def _mtp_record_execution(self, before):
        if before is None:
            return
        draft = self._mtp_drafter
        now = tuple(getattr(draft, "speculative_total_" + key, 0) for key in ("rounds", "accepted", "drafted"))
        rounds, accepted, drafted = (int(b - a) for a, b in zip(before, now))
        self._mtp_last_used = drafted > 0
        self._mtp_call_stats = {"rounds": rounds, "accepted_tokens": accepted, "drafted_tokens": drafted}
        if drafted:
            self._mtp_call_stats["acceptance_rate"] = accepted / drafted

    def _mtp_generate_fn(self, model, tokenizer, prompt=None, **kwargs):
        """mlx-lm-shaped call site adapter over mlx-vlm's `generate`."""
        text = prompt if prompt is not None else kwargs.pop("prompt", None)
        if getattr(self, "_native_runtime", None) is not None:
            from dataclasses import replace
            from .mlx_runtime import NativeRuntimeError
            handle = self._native_runtime.stream(self._native_runtime_request(text, kwargs),
                cancel_event=getattr(self, "_native_cancel_event", None))
            self._native_runtime_stream = handle
            parts, result = [], None
            if getattr(self, "_native_cancel_event", None) is not None and self._native_cancel_event.is_set():
                handle.close()
            try:
                for result in handle:
                    parts.append(result.text)
                event = getattr(self, "_native_cancel_event", None)
                cancelled = event is not None and event.is_set()
            finally:
                handle.close()
                self._native_runtime_stream = None
            if result is None or result.finish_reason is None:
                raise NativeRuntimeError("Native MLX request ended without a terminal result",
                                         code="cancelled" if cancelled else "backend_error")
            result = replace(result, text="".join(parts))
            self._observe_native_runtime_result(result)
            return result.text
        from mlx_vlm import generate as vlm_generate
        call_kwargs = self._mtp_call_kwargs(kwargs)
        call_kwargs.update(self._native_image_kwargs(text))
        before = self._mtp_stats_snapshot()
        apc_before = self._native_apc_counters()
        try:
            result = vlm_generate(model, self._mtp_processor, text, **call_kwargs)
            self._mtp_last_result = result
            self._mtp_last_apc = self._native_apc_delta(apc_before)
            self._record_native_media()
        finally:
            self._mtp_record_execution(before)
        # mlx-lm's `generate` returns a str; mlx-vlm returns a result object.
        # The call site assigns straight into `response_text`, so normalize here
        # rather than teaching every consumer about a second shape.
        return getattr(result, "text", result)

    def _mtp_stream_generate_fn(self, model, tokenizer, prompt=None, **kwargs):
        text = prompt if prompt is not None else kwargs.pop("prompt", None)
        # Mid-prefill progress (`TextProgressEmitter.prefill_progress`), never
        # forwarded to mlx-vlm or into the native request.
        prefill_progress = kwargs.pop(PREFILL_PROGRESS_KWARG, None)
        if not callable(prefill_progress):
            prefill_progress = None
        if getattr(self, "_native_runtime", None) is not None:
            # Only passed when someone listens: an unobserved call is
            # byte-identical to before.
            observe_kwargs = {}
            if prefill_progress is not None:
                def on_prefill(item, _report=prefill_progress):
                    _report(**item.as_kwargs())
                observe_kwargs["on_prefill_progress"] = on_prefill
            handle = self._native_runtime.stream(self._native_runtime_request(text, kwargs),
                cancel_event=getattr(self, "_native_cancel_event", None), **observe_kwargs)
            self._native_runtime_stream = handle
            if getattr(self, "_native_cancel_event", None) is not None and self._native_cancel_event.is_set():
                handle.close()
            def scheduled():
                from .mlx_runtime import NativeRuntimeError
                result = None
                try:
                    for result in handle:
                        self._observe_native_runtime_result(result)
                        yield result
                    if result is None or result.finish_reason is None:
                        event = getattr(self, "_native_cancel_event", None)
                        cancelled = event is not None and event.is_set()
                        raise NativeRuntimeError("Native MLX stream ended without a terminal result",
                                                 code="cancelled" if cancelled else "backend_error")
                finally:
                    handle.close()
                    self._native_runtime_stream = None
            return scheduled()
        from mlx_vlm import stream_generate as vlm_stream_generate
        call_kwargs = self._mtp_call_kwargs(kwargs)
        call_kwargs.update(self._native_image_kwargs(text))
        before = self._mtp_stats_snapshot()
        apc_before = self._native_apc_counters()
        generator = vlm_stream_generate(model, self._mtp_processor, text, **call_kwargs)
        def observed():
            from .mlx_prefill_observer import observe_prefill

            def _fed(done, total, _report=prefill_progress):
                _report(fed_processed=done, fed_total=total)

            def _results():
                # mlx-vlm's chunked prefill runs inside the FIRST `next()`, on
                # this thread; its "Prefill" bar is observed only while bound.
                while True:
                    with observe_prefill(_fed if prefill_progress is not None else None):
                        try:
                            item = next(generator)
                        except StopIteration:
                            return
                    yield item

            try:
                for result in _results():
                    self._mtp_last_result = result
                    self._record_native_media()
                    self._mtp_record_execution(before)
                    yield result
            finally:
                try:
                    generator.close()
                finally:
                    self._mtp_record_execution(before)
                    self._mtp_last_apc = self._native_apc_delta(apc_before)
        return observed()

    def _native_image_kwargs(self, prompt):
        media = getattr(self, "_native_media", None)
        if not media:
            return {}
        from .mlx_qwen4 import prepare_images
        inputs, self._native_media_records = prepare_images(self.llm, self._mtp_processor, prompt, media)
        return inputs

    def _record_native_media(self):
        report = getattr(self, "_native_media_report", None)
        if report is not None:
            seen = getattr(self, "_native_media_delivered", None)
            if seen is None:
                seen = self._native_media_delivered = set()
            for record in self._native_media_records:
                identity = (record["index"], record["kind"], record["transport"])
                if identity not in seen:
                    report.deliver(**record)
                    seen.add(identity)
            self._native_media_records = []

    def _mtp_call_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Translate the mlx-lm call-site kwargs into mlx-vlm's vocabulary."""
        out = dict(kwargs)
        input_embeddings = out.pop("input_embeddings", None)
        out.pop("verbose", None)
        if input_embeddings is not None:
            # FAIL CLOSED. Enabling the MTP lane swaps generate_fn wholesale, so
            # once speculation is on every call goes through mlx-vlm -- image
            # calls included. `mlx_vlm.generate` absorbs unknown kwargs into
            # **kwargs, so forwarding mlx-lm's `input_embeddings` would not
            # raise: the picture would simply vanish and the model would answer
            # from text as if it had seen it. This lane is text-only in v1, and
            # a refusal the caller can act on beats an answer they cannot trust.
            raise ProviderAPIError(
                "mlx: native MTP speculation is text-only, and this request carries "
                "image/video input. The MTP lane runs through mlx-vlm's speculative "
                "loop, which does not take the vision add-on's precomputed "
                "embeddings, and silently answering without them would be worse "
                "than refusing. Create the provider without speculation "
                "(speculation={'mode': 'off'}) for media requests."
            )
        # The keyed snapshot prompt cache is built from mlx-lm cache objects and
        # its rollback lane is shaped around them. Feeding one into mlx-vlm's
        # speculative loop is exactly the "silent desync" this contract exists
        # to prevent, so the MTP lane declines the warm cache instead. This is
        # the one honest cost of the lane and it is stated in the docs.
        cache = out.pop("prompt_cache", None)
        if cache is not None and not self._mtp_prompt_cache_warned:
            self._mtp_prompt_cache_warned = True
            self.logger.warning(
                "mlx: native MTP lane does not reuse the keyed prompt cache; "
                "prompts are prefilled fresh. Disable speculation to get warm-cache "
                "reuse back."
            )
        out.update(self._mtp_kwargs(input_embeddings))
        if getattr(self, "_mtp_processor", None) is not None:
            out.setdefault("prefill_step_size", 256)
            if getattr(self, "_native_cache_key", None):
                session = getattr(self, "_native_session", None) or getattr(self, "_native_qwen4", None)
                if session is None:
                    raise ProviderAPIError("Native MLX prefix caching requires a loaded native session")
                out["apc_manager"] = session.prompt_cache(
                    memory_max_gb=getattr(self, "_mlx_cache_options", {}).get("memory_max_gb")
                )
                out["apc_tenant"] = json.dumps([getattr(self, "_mlx_cache_scope", "local"), self._native_cache_key])
        return out

    def speculation_status(self) -> Dict[str, Any]:
        """What the speculation lane is doing right now, for display surfaces.

        `last_call_used` is the authoritative per-call fact and is set by
        `_mtp_kwargs`, so it is correct in the streaming lane too.
        """
        completed = getattr(self, "_native_last_completed_status", None)
        if completed is not None and not getattr(self, "_native_request_facade", False):
            return {**completed, "stats": dict(completed.get("stats", {}))}
        return {
            "lane_loaded": self._mtp_active,
            "drafter": getattr(self, "_mtp_drafter_id", None),
            "draft_tokens": getattr(self, "_mtp_block_size", None),
            "effective_draft_tokens": getattr(self, "_mtp_call_block_size", None),
            "stats": dict(getattr(self, "_mtp_call_stats", {})),
            "draft_kind": getattr(self, "_mtp_kind", None),
            "output_preserving": getattr(self, "_mtp_output_preserving", None),
            "last_call_used": getattr(self, "_mtp_last_used", None),
            "unavailable_reason": getattr(
                getattr(self, "_mtp_outcome_at_load", None), "reason", None
            ),
        }

    def _mtp_outcome(self, used_this_call: bool) -> SpeculationOutcome:
        """Response metadata for one call.

        `used` is set ONLY from whether a drafter ran on this call -- never from
        the request or the registry.
        """
        # A per-call verdict is the most specific truth about THIS call, so it
        # wins over both the constructor request and any load-time refusal.
        call_outcome = getattr(self, "_mtp_call_outcome", None)
        if call_outcome is not None:
            return call_outcome
        request = getattr(self, "_mtp_call_request", None) or getattr(self, "_speculation_request", None)
        if request is None or not request.enabled:
            return SpeculationOutcome()
        if used_this_call:
            return SpeculationOutcome(
                requested=True,
                mode=request.mode,
                used=True,
                drafter=getattr(self, "_mtp_drafter_id", None),
                num_draft_tokens=getattr(self, "_mtp_call_block_size", getattr(self, "_mtp_block_size", None)),
                details=(
                    {
                        "runtime": "mlx_vlm",
                        "draft_kind": getattr(self, "_mtp_kind", None),
                        **getattr(self, "_mtp_call_stats", {}),
                    }
                    if getattr(self, "_mtp_output_preserving", None) is not False
                    else {
                        "runtime": "mlx_vlm",
                        "draft_kind": getattr(self, "_mtp_kind", None),
                        **getattr(self, "_mtp_call_stats", {}),
                        "output_preserving": False,
                    }
                ),
            )
        held = getattr(self, "_mtp_outcome_at_load", None)
        if held is not None:
            return held
        return SpeculationOutcome(
            requested=True,
            mode=request.mode,
            used=False,
            reason=getattr(self, "_mtp_reason", None) or "speculation_not_applied",
            drafter=getattr(self, "_mtp_drafter_id", None),
        )

    def _load_model(self):
        """Load MLX model and tokenizer"""
        # Only THESE two imports mean "the MLX provider is not installed". They
        # are deliberately outside the load block below: that block imports
        # mlx_vlm, the Qwen4-Exp MTP drafter and per-architecture modules, and
        # answering any of those with "install mlx-lm" points the caller at a
        # package that is already there while hiding the one that is missing.
        try:
            from mlx_lm import load, generate, stream_generate
            import mlx.core as mx
        except ImportError as e:
            raise ImportError(
                "MLX dependencies not installed. Install with: pip install mlx-lm"
            ) from e

        try:
            import os
            from contextlib import redirect_stdout, redirect_stderr
            from pathlib import Path

            # OFFLINE-FIRST = NO ON-DEMAND DOWNLOAD WHILE LOADING, and it is
            # enforced by CONSTRUCTION below, never by process environment:
            # the model is resolved to a local directory from the LM Studio /
            # Hugging Face caches (`resolve_*`, `_has_weights`), a miss raises
            # ModelNotFoundError before mlx-lm is called, and mlx-lm is only
            # ever handed that directory (its `_download` skips
            # `snapshot_download` for an existing path). The MTP drafter is
            # cache-only the same way (`_plan_mtp_lane`).
            #
            # This block used to `os.environ.setdefault` HF_HUB_OFFLINE /
            # TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE = 1. That never did
            # what it said: `huggingface_hub` snapshots HF_HUB_OFFLINE into
            # `constants.HF_HUB_OFFLINE` at ITS import, `from mlx_lm import
            # load` above has already imported it, and transformers >= 5 asks
            # that same constant -- so the in-process loader saw no change.
            # What it DID do was poison the process for good: every child
            # process inherits `os.environ`, so after the first MLX load each
            # explicit download job (a `snapshot_download` child) died with
            # OfflineModeIsEnabled. Do not reintroduce a process-wide write.

            from ..utils.model_cache import (
                default_hf_hub_cache_dirs,
                default_lmstudio_model_dirs,
                resolve_hf_snapshot_dir,
                resolve_lmstudio_hub_manifest,
                resolve_lmstudio_model_dir,
            )

            # Upstream compatibility: mlx-lm may call `mx.metal.device_info()` which is deprecated in recent MLX.
            # Patch the deprecated entrypoint to the supported API so the warning is fixed (not silenced).
            try:
                if (
                    hasattr(mx, "device_info")
                    and hasattr(mx, "metal")
                    and hasattr(mx.metal, "device_info")
                ):
                    mx.metal.device_info = mx.device_info  # type: ignore[attr-defined]
            except Exception:
                pass

            # Clean model name - remove trailing slashes that cause HuggingFace validation errors
            clean_model_name = self.model.rstrip("/")

            def _has_weights(d: Path) -> bool:
                """Best-effort check to avoid triggering downloads on missing weights."""
                try:
                    if not d.is_dir():
                        return False
                except Exception:
                    return False
                patterns = ("*.safetensors", "*.npz", "*.bin", "*.pt", "*.pth")
                for pat in patterns:
                    try:
                        if any(d.glob(pat)):
                            return True
                    except Exception:
                        continue
                return False

            def _looks_like_gguf_dir(d: Path) -> bool:
                try:
                    if not d.is_dir():
                        return False
                except Exception:
                    return False
                try:
                    return any(p.suffix.lower() == ".gguf" for p in d.iterdir())
                except Exception:
                    return False

            # Resolve to a local directory (cache-only). Do not pass a repo id into mlx-lm,
            # as it can trigger Hub network requests even when cached.
            load_dir: Optional[Path] = None
            explicit_path = Path(clean_model_name).expanduser()
            if explicit_path.is_dir():
                load_dir = explicit_path
            else:
                load_dir = resolve_lmstudio_model_dir(
                    clean_model_name, base_dirs=default_lmstudio_model_dirs()
                )
                if load_dir is None:
                    snap = resolve_hf_snapshot_dir(
                        clean_model_name, cache_dirs=default_hf_hub_cache_dirs()
                    )
                    if snap is not None and _has_weights(snap):
                        load_dir = snap

            if load_dir is None or _looks_like_gguf_dir(load_dir):
                hint_lines: list[str] = []
                if load_dir is not None and _looks_like_gguf_dir(load_dir):
                    hint_lines.append(
                        f"Found GGUF files under '{load_dir}', but the MLX provider cannot load GGUF."
                    )
                    hint_lines.append(
                        "Use `--provider huggingface` (GGUF) or `--provider lmstudio` for GGUF-backed models."
                    )
                else:
                    manifest_path = resolve_lmstudio_hub_manifest(clean_model_name)
                    if manifest_path is not None:
                        try:
                            raw = manifest_path.read_text(encoding="utf-8")
                            manifest = json.loads(raw) if raw.strip() else {}
                            deps = (
                                manifest.get("dependencies") if isinstance(manifest, dict) else None
                            )
                            if isinstance(deps, list) and deps:
                                for dep in deps:
                                    if not isinstance(dep, dict):
                                        continue
                                    for src in dep.get("sources") or []:
                                        if not isinstance(src, dict):
                                            continue
                                        if (
                                            str(src.get("type") or "").strip().lower()
                                            != "huggingface"
                                        ):
                                            continue
                                        user = str(src.get("user") or "").strip()
                                        repo = str(src.get("repo") or "").strip()
                                        if not user or not repo:
                                            continue
                                        repo_id = f"{user}/{repo}"
                                        lm_dir = resolve_lmstudio_model_dir(
                                            repo_id, base_dirs=default_lmstudio_model_dirs()
                                        )
                                        if lm_dir is None:
                                            continue
                                        try:
                                            ggufs = sorted(
                                                [p for p in lm_dir.glob("*.gguf") if p.is_file()]
                                            )
                                        except Exception:
                                            ggufs = []
                                        if ggufs:
                                            hint_lines.append(
                                                f"LM Studio hub entry found for '{clean_model_name}', but it resolves to GGUF files (e.g. '{ggufs[0].name}')."
                                            )
                                            hint_lines.append(
                                                "MLX provider cannot load GGUF; use `--provider huggingface` (GGUF) or `--provider lmstudio`."
                                            )
                                            break
                                    if hint_lines:
                                        break
                        except Exception:
                            pass

                searched_lms = [str(p) for p in default_lmstudio_model_dirs()]
                searched_hf = [str(p) for p in default_hf_hub_cache_dirs()]
                headline = (
                    f"❌ MLX model '{clean_model_name}' not found locally (downloads are disabled)."
                    if load_dir is None
                    else f"❌ MLX provider cannot load '{clean_model_name}' (GGUF detected; downloads are disabled)."
                )
                msg = (
                    f"{headline}\n\n"
                    f"Searched LM Studio caches:\n  - "
                    + "\n  - ".join(searched_lms or ["(none found)"])
                    + "\n\n"
                    f"Searched HuggingFace hub caches:\n  - "
                    + "\n  - ".join(searched_hf or ["(none found)"])
                    + "\n"
                )
                if hint_lines:
                    msg += "\n" + "\n".join(hint_lines) + "\n"
                msg += "\nTip: download explicitly (e.g. with `huggingface-cli download ...`) or pass a local model directory path."
                raise ModelNotFoundError(msg)

            load_target = str(load_dir)
            self._resolved_model_id = load_target
            # The architecture NAME for native-lane error text (the checkpoint's
            # own `model_type`), so a Qwen3.5 model is never told it is "Qwen4".
            try:
                _cfg = json.loads((Path(load_target) / "config.json").read_text(encoding="utf-8"))
                self._native_model_type = str(_cfg.get("model_type") or "") or None if isinstance(_cfg, dict) else None
            except Exception:
                self._native_model_type = None
            from .speculation import mlx_speculation_artifact
            self._speculation_artifact = mlx_speculation_artifact(self.model)
            if getattr(self, "_speculation_inherits_config", False):
                from .speculation import configured_speculation_default, describe_speculation_capabilities
                policy = configured_speculation_default(
                    config_file=getattr(self, "_abstractcore_config_file", None),
                    capability_defaults=getattr(self, "_abstractcore_capability_defaults", None),
                )
                self._speculation_default_supported = describe_speculation_capabilities(self.model, "mlx")["supported"]
                if policy is not None and self._speculation_default_supported:
                    self._speculation_request = normalize_speculation_request(policy)
            from .mlx_qwen4 import is_qwen4_checkpoint, embedded_mtp_keys, load_qwen4_session

            if is_qwen4_checkpoint(load_target):
                request = self._speculation_request
                enable_mtp = bool(request and request.enabled)
                if enable_mtp and request.drafter:
                    raise ValueError("Qwen4-Exp uses its embedded MTP head; omit speculation.drafter")
                if enable_mtp and not embedded_mtp_keys(load_target):
                    self._mtp_outcome_at_load = speculation_unavailable(
                        request, "embedded_mtp_weights_missing",
                        "This Qwen4-Exp checkpoint contains no indexed MTP weights; use an MTP-preserving artifact",
                        logger=self.logger,
                    )
                    enable_mtp = False
                session = load_qwen4_session(load_target, mtp=enable_mtp, ple_offload=self._mlx_ple_offload)
                self._native_qwen4 = session
                self._mtp_generation_lock = session.lock
                self.llm = session.model
                self._mtp_processor = session.processor
                self.tokenizer = getattr(session.processor, "tokenizer", session.processor)
                self._mtp_drafter = session.drafter if enable_mtp else None
                self._mtp_kind = "mtp" if enable_mtp else None
                self._mtp_drafter_id = load_target + "#mtp" if enable_mtp else None
                self._mtp_block_size = (request.num_draft_tokens or 3) if enable_mtp else None
                self.generate_fn = self._mtp_generate_fn
                self.stream_generate_fn = self._mtp_stream_generate_fn
                self._bind_native_session(session)
                self.logger.info(f"mlx: native Qwen4-Exp loaded; embedded MTP={enable_mtp}, PLE mmap={session.ple_offload}")
                return
            # config.json ONLY: no mlx_vlm import, no tensor scan, no network.
            # The transport-actual answer comes from the real add-on load at the
            # first image; this cheap flag only decides whether to try.
            try:
                from .mlx_vision_addon import vision_status

                usable, reason, info = vision_status(load_target)
                self._vision_usable = usable
                self._vision_reason = reason
                self._vision_info = info
                # Say it at LOAD, not at the first image. The extra is opt-in by
                # design, so a sighted checkpoint on a runtime without mlx-vlm is
                # an ordinary, silent, text-only session until someone attaches a
                # picture -- and by then the answer is already wrong. An import
                # check is cheap (no tower load) and turns a surprise into a note.
                import importlib.util

                if usable and importlib.util.find_spec("mlx_vlm") is None:
                    # WARNING, not INFO: mlx-vlm is part of the MLX provider's
                    # dependency set, so its absence is a broken install rather
                    # than a configuration choice, and this checkpoint is about to
                    # silently lose every image it is given.
                    self.logger.warning(
                        f"mlx: {self.model} ships a vision tower but mlx-vlm is "
                        "missing from this interpreter, so images will be dropped. "
                        'This install is incomplete — repair with: pip install '
                        '"abstractcore[mlx]"'
                    )
            except Exception as exc:
                # Keep a NAMED reason. `None` here makes the later drop record
                # skip the reason entirely (see `_try_vision_addon`), so a probe
                # that merely crashed became a request with no diagnosis at all --
                # the one failure mode this contract exists to prevent.
                self._vision_usable = False
                self._vision_reason = "vision_probe_failed"
                self._vision_info = {}
                self.logger.warning(f"mlx: vision capability probe failed: {exc}")

            # Decide the MTP lane BEFORE loading. It is an either/or, never a
            # retrofit: mlx-vlm and mlx-lm each hold their own copy of the
            # weights, and this model is 15 GB at 4-bit, so loading one and then
            # discovering we wanted the other costs a second 15 GB.
            #
            # An MTP-PRESERVING checkpoint (`mtp.` tensors in its weights) NEVER
            # goes to mlx-lm: mlx-lm <= 0.31.3 shifts its already-converted norm
            # weights a second time and it generates garbage (see
            # `mlx_native_session.mtp_weight_keys`). Decided from the local
            # index BEFORE any lane is chosen, so every lane -- drafter
            # present or missing, speculation on/off/inherited, batching or
            # not -- lands on mlx-vlm, or fails loudly.
            from .mlx_native_session import mtp_weight_keys

            try:
                mtp_keys = mtp_weight_keys(load_target)
            except Exception as exc:
                raise ProviderAPIError(
                    f"MLX model '{self.model}': cannot read the weight index of '{load_target}' "
                    f"to tell whether it carries MTP tensors ({type(exc).__name__}: {exc}); "
                    "refusing to guess the loader."
                ) from exc
            self._mtp_preserving_checkpoint = bool(mtp_keys)
            mtp_plan = self._plan_mtp_lane(load_target)
            if mtp_plan is not None and self._enter_mtp_lane(load_target, mtp_plan):
                return
            if self._mlx_batching or mtp_keys:
                why = (
                    f"its weights carry {len(mtp_keys)} MTP tensor(s) (`mtp.` keys), and mlx-lm "
                    "mis-converts such checkpoints (garbage output), so it loads only through mlx-vlm"
                    if mtp_keys
                    else "native MLX batching (mlx_batching=True) runs through mlx-vlm"
                )
                self._enter_mtp_lane(load_target, {"drafter": None, "block_size": None, "why": why})
                if mtp_keys:
                    self.logger.info(
                        f"mlx: {self.model} is MTP-preserving; loaded through mlx-vlm "
                        f"({'with' if self._mtp_active else 'without'} an MTP drafter)"
                    )
                return

            # Silence the "Fetching" progress bar by redirecting stdout/stderr
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    try:
                        self.llm, self.tokenizer = self._load_or_adopt_shared_model(
                            str(load_target), lambda: load(load_target)
                        )
                    except ValueError as e:
                        msg = str(e)
                        low = msg.lower()
                        if "model type" in low and "not supported" in low:
                            model_type = None
                            try:
                                cfg_path = Path(load_target) / "config.json"
                                if cfg_path.is_file():
                                    raw = cfg_path.read_text(encoding="utf-8", errors="ignore")
                                    cfg = json.loads(raw) if raw.strip() else {}
                                    model_type = (
                                        cfg.get("model_type") if isinstance(cfg, dict) else None
                                    )
                            except Exception:
                                model_type = None

                            mlx_lm_version = None
                            try:  # pragma: no cover
                                import mlx_lm  # type: ignore

                                mlx_lm_version = getattr(mlx_lm, "__version__", None)
                            except Exception:
                                mlx_lm_version = None

                            ver_s = f" (mlx-lm {mlx_lm_version})" if mlx_lm_version else ""
                            extra_hint = ""
                            if str(model_type or "").strip().lower() == "gemma4":
                                extra_hint = (
                                    "\n"
                                    "Note:\n"
                                    "  - Gemma 4 MLX models require a newer mlx-lm build (>=0.31.2).\n"
                                    "    If that version is not available on PyPI yet, install mlx-lm from source until it is released.\n"
                                )

                            raise ModelNotFoundError(
                                f"❌ MLX provider cannot load '{clean_model_name}' from '{load_target}'.\n\n"
                                f"Detected model_type={model_type!r}, but the installed mlx-lm does not support it{ver_s}.\n\n"
                                "Try one of:\n"
                                "  - Use provider='huggingface' (transformers) for this local model directory\n"
                                "  - Use provider='lmstudio' if you are running LM Studio's local server\n"
                                "  - Upgrade mlx-lm once a release with this model_type is published on PyPI\n"
                                f"{extra_hint}"
                            ) from e
                        raise

            self.generate_fn = generate
            self.stream_generate_fn = stream_generate
        except ImportError as e:
            # A module the installed MLX stack does not provide -- almost always
            # a version floor that is not met, not an absent provider. The
            # raisers below already name the floor they need (see
            # mlx_qwen4._create_qwen4_session), so KEEP their text and add the
            # one fact they cannot know: what is actually installed here.
            raise ImportError(
                f"{e}\n"
                f"Installed MLX stack: {_installed_mlx_versions()} "
                f"(AbstractCore requires mlx>=0.32.2, mlx-lm>=0.31.3, mlx-vlm>=0.7.1). "
                f"Fix with: pip install -U \"abstractcore[mlx]\""
            ) from e
        except SpeculationUnavailableError:
            # `require_acceleration=True` is an explicit "fail rather than run
            # slow". Re-flattening it into a generic "Failed to load MLX model"
            # below would destroy both the type and the machine-readable reason
            # the caller asked for.
            raise
        except ProviderAPIError:
            # Already names the model and the reason (e.g. an MTP-preserving
            # checkpoint whose mlx-vlm load failed). The generic handler below
            # would re-read "not found" in it as a missing model.
            raise
        except Exception as e:
            # Check if it's a model not found error
            error_str = str(e).lower()
            if (
                "not found" in error_str
                or "does not exist" in error_str
                or "failed to load" in error_str
            ):
                available_models = self.list_available_models()
                error_message = format_model_error("MLX", self.model, available_models)
                raise ModelNotFoundError(error_message)
            raise Exception(f"Failed to load MLX model {self.model}: {str(e)}")

    def load_model(self, model_name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """(Re)load this instance's model after an `unload_model` (idempotent).

        Runs the constructor's own `_load_model()` (weights, tokenizer, native
        session when configured). One instance serves ONE model; the MTP
        drafter is loaded again on the first speculative call, as at start.

        MLX has no idle/TTL unload: `ttl_s`, `keep_alive` (and any other load
        option) are NOT applied, and the response says so in `warnings` and
        `unsupported_options` instead of accepting them silently. `pin` is the
        caller's own residency lock and is not a provider option."""
        unsupported = sorted(k for k, v in kwargs.items() if k != "pin" and v is not None)
        warnings: List[str] = []
        if unsupported:
            timed = [k for k in unsupported if k in ("ttl_s", "keep_alive")]
            other = [k for k in unsupported if k not in ("ttl_s", "keep_alive")]
            if timed:
                warnings.append(
                    f"MLX has no idle/TTL unload: {', '.join(timed)} not applied; "
                    "the model stays resident until it is ejected"
                )
            if other:
                warnings.append(f"MLX load does not support option(s) {', '.join(other)}; not applied")
            self.logger.warning(f"MLX load of {self.model}: {'; '.join(warnings)}")
        extras: Dict[str, Any] = {"warnings": warnings, "unsupported_options": unsupported} if unsupported else {}
        target = str(model_name or self.model or "").strip()
        if target and target != str(self.model):
            raise ValueError(
                f"MLXProvider instance for {self.model!r} cannot load {target!r}: "
                "create a provider for that model instead."
            )
        if self.llm is not None and self.tokenizer is not None:
            return {"supported": True, "operation": "load", "provider": "mlx", "model": self.model,
                    "action": "already_loaded", "source": "abstractcore.provider.mlx", **extras}
        t0 = time.time()
        self._load_model()
        return {"supported": True, "operation": "load", "provider": "mlx", "model": self.model,
                "action": "loaded", "load_s": round(time.time() - t0, 3), "source": "abstractcore.provider.mlx",
                **extras}

    def unload_model(self, model_name: str) -> None:
        # Stop what is running on this instance first (host-cancellable calls:
        # every gateway/runtime call carries its effect's event). A call that
        # does not stop by the deadline makes the unload RAISE, nothing freed.
        self._stop_inflight_before_unload(model_name, refuse_if_running=True)
        runtime = getattr(self, "_native_runtime", None)
        session = getattr(self, "_native_session", None)
        if runtime is not None:
            # release rejects pending/active requests without destroying the lease.
            release_error = None
            with session.config_lock:
                try:
                    runtime.release(self._native_owner_id)
                except Exception as exc:
                    # A stopped worker's flush failure must remain visible, but
                    # must not pin its dead session/weights to this provider.
                    # A live worker (including a close timeout) still owns its
                    # tensors: preserve everything so the same lease can retry.
                    worker = getattr(runtime, "_thread", None)
                    if (getattr(runtime, "_pending_release_owner", None) != self._native_owner_id
                            or worker is None or worker.is_alive()):
                        raise
                    release_error = (str(exc), getattr(exc, "code", "backend_error"))
                    # Keep only scalar failure details: the original exception
                    # traceback can retain the runtime, session and all weights.
                    del worker
                session.holders.discard(self)
                self._native_shared_still_used = bool(list(session.holders))
                if not self._native_shared_still_used:
                    session.runtime = None
                    session.cache_store = None
                    session.execution_config = None
            finalizer = getattr(self, "_native_finalizer", None)
            if finalizer is not None:
                finalizer.detach()
            self._native_runtime = None
            self._native_owner_id = None
            self._native_last_completed_status = None
            self._native_session = None
            self._native_qwen4 = None
            # Drop the outer frame's owners before the common cleanup collects
            # model cycles and clears Metal's allocator. Otherwise weights are
            # freed only after that clear, leaving newly cached allocations.
            del runtime, session
            self._unload_model_unlocked(model_name)
            if release_error is not None:
                from .mlx_runtime import NativeRuntimeError
                message, code = release_error
                raise NativeRuntimeError(message, code=code) from None
            return None
        lock = getattr(self, "_mtp_generation_lock", None) if getattr(self, "_mtp_processor", None) is not None else None
        if lock is not None and not lock.acquire(blocking=False):
            raise ProviderAPIError("Cannot unload native MLX during generation; finish or close its stream first")
        try:
            # Drop only this provider's native references. A sibling provider
            # may still own the shared weights and prefix cache.
            if session is not None:
                session.holders.discard(self)
                self._native_shared_still_used = bool(list(session.holders))
                finalizer = getattr(self, "_native_finalizer", None)
                if finalizer is not None:
                    finalizer()
                self._native_session = None
            self._native_qwen4 = None
            del session
            return self._unload_model_unlocked(model_name)
        finally:
            if lock is not None:
                lock.release()

    def _unload_model_unlocked(self, model_name: str) -> None:
        """
        Unload the MLX model from memory.

        Clears model and tokenizer references and forces garbage collection
        to free GPU/CPU memory immediately. Also drops this instance's session
        caches (prompt-cache store entries AND hybrid KV snapshots): they are
        only useful with the weights resident, and they are the memory hogs.
        """
        import gc

        try:
            # A SHARED model is unloaded for THIS provider only while others still use
            # it: their weights and their sessions' caches are not this call's to clear.
            shared_still_used = self._release_shared_model() or getattr(self, "_native_shared_still_used", False)

            if hasattr(self, "llm") and self.llm is not None:
                # Clear MLX model
                del self.llm
                self.llm = None

            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                # Clear tokenizer
                del self.tokenizer
                self.tokenizer = None

            if hasattr(self, "generate_fn"):
                self.generate_fn = None

            if hasattr(self, "stream_generate_fn"):
                self.stream_generate_fn = None

            # Session caches go with the weights (prompt_cache_clear also drops
            # `_hybrid_snapshots` via this provider's override). After a release that
            # left other users, this clears only the private empty state.
            self._clear_prompt_caches_for_unload()
            if shared_still_used:
                self.logger.info(
                    "MLX model stays resident: other providers in this process still use it."
                )

            # Force garbage collection to free memory immediately
            # The add-on holds the mlx-vlm wrapper, its processor and the tower;
            # `_outlines_model` holds a live reference to self.llm and defeats the
            # unload entirely for any provider that ran a structured request.
            self._vision_addon = None
            self._outlines_model = None
            self._mtp_drafter = None
            self._mtp_processor = None
            self._mtp_last_result = None
            self._native_qwen4 = None
            gc.collect()
            # ALWAYS release MLX's cache of freed buffers, whether or not a
            # sibling still uses the weights. `clear_cache` only returns
            # buffers nobody references any more; it cannot touch a live
            # tensor, so it is harmless for the siblings -- and skipping it
            # is a measured leak: in a hermetic replay of a real gateway
            # session, the sibling that made `shared_still_used` True
            # was itself collected by the `gc.collect()` above, its 15.5 GB
            # of weights + 2.6 GB of prefix cache went into the allocator
            # cache, and the guarded clear never ran: the process kept 21 GB
            # with `mx.get_active_memory()` at 0 and the report at "nothing
            # loaded". The process-level truth lives in `mlx_residency`.
            try:
                from .mlx_residency import clear_mx_cache, mx_memory_stats

                clear_mx_cache()
                if shared_still_used:
                    stats = mx_memory_stats()
                    self.logger.info(
                        "MLX unload of %s: weights stay resident for other holders; "
                        "mlx active=%s bytes after this instance released its references",
                        model_name, stats.get("active_bytes"),
                    )
            except Exception:
                pass
        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, "logger"):
                self.logger.warning(f"Error during unload: {e}")

    def _handle_timeout_parameter(self, kwargs: Dict[str, Any]) -> None:
        """
        Handle timeout parameter for MLX provider.

        Since MLX models run locally on Apple Silicon,
        timeout parameters don't apply. If a non-None timeout is provided,
        issue a warning and treat it as None (infinity).

        Args:
            kwargs: Initialization kwargs that may contain timeout
        """
        timeout_value = kwargs.get("timeout")
        if timeout_value is not None:
            import warnings

            warnings.warn(
                f"MLX provider runs models locally on Apple Silicon and does not support timeout parameters. "
                f"Provided timeout={timeout_value} will be ignored and treated as None (unlimited).",
                UserWarning,
                stacklevel=3,
            )
            # Force timeout to None for local models
            self._timeout = None
        else:
            # Keep None value (unlimited timeout is appropriate for local models)
            self._timeout = None

    def _update_http_client_timeout(self) -> None:
        """
        MLX provider doesn't use HTTP clients for model inference.
        Local models on Apple Silicon don't have timeout constraints.
        """
        # No-op for local models - they don't use HTTP clients
        pass

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    async def agenerate(self, prompt="", messages=None, system_prompt=None,
                        tools=None, media=None, stream=False, **kwargs):
        """Normalize scheduled requests once, through the public sync lane.

        BaseProvider.agenerate applies thinking/routing before calling the
        adapter. Re-entering generate after that loses the original thinking
        choice and can reapply configured defaults. The native scheduler is
        already thread-safe, so hand its sync facade the original request;
        it owns normalization, telemetry, tools and structured output.
        """
        if not self.supports_concurrent_generation():
            return await super().agenerate(
                prompt, messages=messages, system_prompt=system_prompt,
                tools=tools, media=media, stream=stream, **kwargs)
        return await self._agenerate_internal(
            prompt, messages, system_prompt, tools, media, stream, **kwargs)

    async def _agenerate_internal(self, prompt, messages, system_prompt, tools, media, stream, **kwargs):
        """Never advance a blocking native iterator on the asyncio event loop."""
        if not self.supports_concurrent_generation():
            return await super()._agenerate_internal(prompt, messages, system_prompt, tools, media, stream, **kwargs)
        import asyncio
        view = self._native_request_view()
        call = dict(messages=messages, system_prompt=system_prompt, tools=tools, media=media, **kwargs)
        def cancel():
            view._native_cancel_event.set()
            handle = view._native_runtime_stream
            if handle is not None:
                handle.close()
        if not stream:
            task = asyncio.create_task(asyncio.to_thread(view.generate, prompt, stream=False, **call))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5)
                except (Exception, asyncio.CancelledError):
                    pass
                raise
        from ..utils.async_stream import async_stream
        def source():
            # Lazy creation keeps normalization and iterator advancement on
            # the same producer thread, including owner-thread close.
            yield from view.generate(prompt, stream=True, **call)
        return async_stream(source(), on_cancel=cancel)

    async def _annotate_async_stream(self, source):
        # Base's wrapper does not close its nested iterator on client aclose().
        try:
            async for chunk in source:
                self._annotate_output_truncation(chunk)
                yield chunk
        finally:
            if hasattr(source, "aclose"):
                await source.aclose()

    def _resolve_generate_route(self, *, request, output, thinking, kwargs):
        if getattr(self, "_mtp_processor", None) is not None:
            # A directly constructed native provider is already bound to its
            # weights. Global route defaults must not mislabel their execution.
            kwargs = dict(kwargs)
            kwargs.setdefault("_provider", self.provider)
            kwargs.setdefault("_model", self.model)
            if output is not None and (getattr(request, "text", None) or getattr(request, "messages", None)):
                output = [dict(spec) for spec in self._normalize_output_specs(output)]
                for spec in output:
                    if spec.get("modality") == "text" and spec.get("task") in (None, "text_generation"):
                        spec.setdefault("provider", kwargs["_provider"])
                        spec.setdefault("model", kwargs["_model"])
        return super()._resolve_generate_route(request=request, output=output, thinking=thinking, kwargs=kwargs)

    def _structured_output_carries_media(self) -> bool:
        """False: Outlines' MLX adapter accepts a prompt STRING and re-encodes it
        with the plain tokenizer, so an image's expanded placeholder tokens cannot
        survive that path, and the prompted lane returns a validated model with no
        metadata channel either. Refusing loudly is the honest outcome.
        """
        return False

    def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None,
                           media=None, stream=False, response_model=None, **kwargs):
        """Own native runtime state through exhaustion/close of a lazy stream.

        Overlapping native calls fail explicitly instead of resetting another
        request's drafter/cache state. Resolving per-call controls happens only
        after ownership is acquired, not when a lazy iterator is constructed.
        """
        call = dict(messages=messages, system_prompt=system_prompt, tools=tools,
                    media=media, stream=stream, response_model=response_model, **kwargs)
        if getattr(self, "_native_runtime", None) is not None:
            event = call.pop("_cancel_event", None)
            if event is not None and not isinstance(event, threading.Event):
                from .mlx_runtime import NativeRuntimeError
                raise NativeRuntimeError("_cancel_event must be a threading.Event", code="invalid_request")
            view = self if getattr(self, "_native_request_facade", False) else self._native_request_view(event)
            if event is not None:
                view._native_cancel_event = event
            if not stream:
                response = view._generate_internal_unlocked(prompt, **call)
                view._publish_native_request_status()
                return response
            def scheduled_stream():
                iterator = view._generate_internal_unlocked(prompt, **call)
                try:
                    for chunk in iterator:
                        chunk.metadata = dict(chunk.metadata or {})
                        chunk.metadata.update(getattr(view, "_native_runtime_metadata", {}))
                        outcome = view._mtp_outcome(bool(getattr(view, "_mtp_last_used", False)))
                        if outcome.requested or outcome.reason:
                            chunk.metadata["speculation"] = outcome.to_metadata()
                        yield chunk
                finally:
                    if hasattr(iterator, "close"):
                        iterator.close()
                    view._publish_native_request_status()
            return scheduled_stream()
        if getattr(self, "_mtp_processor", None) is None:
            return self._generate_internal_unlocked(prompt, **call)
        lock = getattr(self, "_mtp_generation_lock", None)
        if lock is None:
            lock = self._mtp_generation_lock = threading.Lock()

        def acquire():
            if not lock.acquire(blocking=False):
                raise ProviderAPIError("Native MLX generation is already active; finish or close its stream before another request")

        if not stream:
            acquire()
            try:
                return self._generate_internal_unlocked(prompt, **call)
            finally:
                lock.release()

        def owned_stream():
            acquire()
            iterator = None
            try:
                iterator = self._generate_internal_unlocked(prompt, **call)
                for chunk in iterator:
                    native_report = getattr(self, "_native_media_report", None)
                    if native_report is not None:
                        from ..media.delivery import attach_media_report
                        chunk = attach_media_report(chunk, native_report)
                    outcome = self._mtp_outcome(bool(getattr(self, "_mtp_last_used", False)))
                    if outcome.requested or outcome.reason:
                        chunk.metadata = dict(chunk.metadata or {})
                        chunk.metadata["speculation"] = outcome.to_metadata()
                    yield chunk
            finally:
                try:
                    if iterator is not None and hasattr(iterator, "close"):
                        iterator.close()
                finally:
                    lock.release()

        return owned_stream()

    def _generate_internal_unlocked(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List["MediaContent"]] = None,
        stream: bool = False,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Provider seam. Owns one thing: no return escapes the media report.

        The body lives in `_generate_core`, which may return from any of its
        branches. Every one of those returns funnels through `attach_media_report`
        here, so a NEW branch cannot forget the record — it has no way to return
        past this line. The honesty contract was previously attached at one of six
        return sites, which is exactly why `response_model=` and `stream=True`
        both lost images silently.
        """
        from ..media.delivery import MediaReport, attach_media_report

        # Consume `speculation=` here so a per-call request is never absorbed
        # into **kwargs and dropped. It cannot TURN ON the lane -- entering it
        # decides which runtime loads 15 GB of weights, which happened at
        # construction -- but silence is the one outcome the contract forbids,
        # so an unhonorable per-call request warns (or raises under
        # require_acceleration) exactly like a constructor-time one.
        try:
            self._apply_per_call_speculation(kwargs.pop("speculation", None))
        except (TypeError, ValueError) as exc:
            if getattr(self, "_native_runtime", None) is not None:
                from .mlx_runtime import NativeRuntimeError
                raise NativeRuntimeError(str(exc), code="invalid_controls") from exc
            raise
        if getattr(self, "_mtp_processor", None) is not None:
            self._native_execute_tools = kwargs.get("execute_tools")
            self._prepare_native_sampling(kwargs)
            stop = kwargs.get("stop") or ()
            try:
                self._native_stop = (stop,) if isinstance(stop, str) else tuple(stop)
                if any(not isinstance(item, str) or not item for item in self._native_stop):
                    raise ValueError("stop must contain non-empty strings")
            except (TypeError, ValueError) as exc:
                if getattr(self, "_native_runtime", None) is not None:
                    from .mlx_runtime import NativeRuntimeError
                    raise NativeRuntimeError(str(exc), code="invalid_request") from exc
                raise
            if self._native_stop and not self.supports_concurrent_generation():
                raise ProviderAPIError("Native MLX stop strings require mlx_batching=True")
            if kwargs.get("prompt_cache_key") and messages is None:
                if getattr(self, "_native_runtime", None) is not None:
                    from .mlx_runtime import NativeRuntimeError
                    raise NativeRuntimeError(
                        "Native prefix caching requires messages containing the complete conversation "
                        "(messages=[] for a standalone prompt); hidden-context append fragments are unsupported",
                        code="invalid_request")
                raise ProviderAPIError(
                    f"Native MLX lane for {self._native_family_label()}: prompt_cache_key requires "
                    "messages containing the complete conversation (messages=[] for a standalone "
                    "prompt). Hidden-context append fragments are not supported by the native "
                    "prefix cache."
                )
            if response_model and self.structured_output_method == "native_outlines":
                if getattr(self, "_native_runtime", None) is not None:
                    from .mlx_runtime import NativeRuntimeError
                    raise NativeRuntimeError("Native MLX does not support Outlines' mlx-lm adapter; use structured_output_method='prompted'",
                                             code="invalid_controls")
                raise ProviderAPIError(f"Native MLX lane for {self._native_family_label()} does not support Outlines' mlx-lm adapter; use structured_output_method='prompted'")

        report = MediaReport.for_request(media, provider="mlx", model=self.model)
        out = self._generate_core(
            prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            media=media,
            stream=stream,
            response_model=response_model,
            report=report,
            **kwargs,
        )
        return attach_media_report(out, report)

    def _prepare_native_sampling(self, kwargs):
        """Keep native controls intact; never silently skip target processors.

        mlx-vlm 0.7.1's MTP loop does not apply logit processors beyond the
        first token. Strict acceleration refuses that combination; optional
        acceleration falls back explicitly to the same native target decoder.
        Sampler-side controls (including min_p) remain compatible with MTP.
        """
        names = ("min_p", "repetition_penalty", "presence_penalty", "frequency_penalty",
                 "repetition_context_size", "presence_context_size", "frequency_context_size", "logit_bias")
        self._native_sampling_kwargs = {k: kwargs[k] for k in names if k in kwargs}
        neutral = {"repetition_penalty": (None, 1, 1.0), "presence_penalty": (None, 0, 0.0),
                   "frequency_penalty": (None, 0, 0.0), "logit_bias": (None, {})}
        active = [k for k, values in neutral.items() if kwargs.get(k) not in values]
        if active and self._mtp_active and not getattr(self, "_mtp_call_disabled", False):
            self._mtp_call_disabled = True
            self._mtp_call_outcome = speculation_unavailable(
                self._mtp_call_request, "mtp_logit_processors_unsupported",
                "mlx-vlm MTP cannot honor " + ", ".join(active) +
                "; use speculation=False for native target-only decoding with these controls",
                logger=self.logger,
            )

    def _generate_core(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List["MediaContent"]] = None,
        stream: bool = False,
        response_model: Optional[Type[BaseModel]] = None,
        report: Optional[Any] = None,
        **kwargs,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with MLX and optional Outlines native structured output"""
        from ..media.delivery import MediaReport

        if report is None:  # direct callers (tests) keep working
            report = MediaReport.for_request(media, provider="mlx", model=self.model)

        if not self.llm or not self.tokenizer:
            # Ejected (`unload_model`) and asked again: reload on demand, LOUDLY,
            # like LM Studio / Ollama JIT loading. Before this, a
            # gateway run after an eject "completed" with the answer
            # "Error: MLX model not loaded" and success=true. A failed reload
            # raises (never an error string dressed as an answer).
            self.logger.warning(
                f"MLX model {self.model} is not loaded (ejected); reloading it on demand for this request"
            )
            self.load_model()

        # Phase feedback (prefill vs generation). Popped before
        # `_prepare_generation_kwargs` so it can never be mistaken for a
        # sampling control, and inert unless a host actually subscribed.
        from .generation_progress import PROGRESS_KWARG, TextProgressEmitter

        progress = TextProgressEmitter(
            kwargs.pop(PROGRESS_KWARG, None), provider="mlx", model=self.model,
        )

        # Host cancel (generation_cancel.py). The native-runtime lane already
        # popped it in `_generate_internal` and bound it to the request view
        # (`_native_cancel_event`, checked per token by the scheduler); every
        # other lane receives it here and checks it per sampled token.
        from .generation_cancel import CANCEL_KWARG, as_cancel_event

        cancel_event = as_cancel_event(kwargs.pop(CANCEL_KWARG, None))
        if cancel_event is None and getattr(self, "_native_runtime", None) is not None:
            cancel_event = as_cancel_event(getattr(self, "_native_cancel_event", None))

        prompt_cache_prefilled_modules = kwargs.pop("prompt_cache_prefilled_modules", None)
        if isinstance(prompt_cache_prefilled_modules, tuple):
            prompt_cache_prefilled_modules = list(prompt_cache_prefilled_modules)
        if isinstance(prompt_cache_prefilled_modules, str):
            prompt_cache_prefilled_modules = [prompt_cache_prefilled_modules]
        if not isinstance(prompt_cache_prefilled_modules, list):
            prompt_cache_prefilled_modules = None
        mlx_enable_thinking = kwargs.get("_acore_mlx_enable_thinking")
        mlx_reasoning_effort = kwargs.get("_acore_mlx_reasoning_effort")

        # Native structured output via Outlines (if configured and available)
        should_use_outlines = (
            response_model
            and PYDANTIC_AVAILABLE
            and not stream
            and getattr(self, "_mtp_processor", None) is None
            and self.structured_output_method != "prompted"  # Skip if explicitly prompted
        )

        if should_use_outlines:
            # Check if Outlines is required but unavailable
            if self.structured_output_method == "native_outlines" and not OUTLINES_AVAILABLE:
                return GenerateResponse(
                    content="Error: structured_output_method='native_outlines' requires Outlines library. Install with: pip install \"abstractcore[mlx]\"",
                    model=self.model,
                    finish_reason="error",
                )

            # Try Outlines if available (auto or native_outlines mode)
            if OUTLINES_AVAILABLE:
                try:
                    # Cache Outlines MLX model wrapper to avoid re-initialization
                    if not hasattr(self, "_outlines_model") or self._outlines_model is None:
                        self.logger.debug(
                            "Creating Outlines MLX model wrapper for native structured output"
                        )
                        self._outlines_model = outlines.from_mlxlm(self.llm, self.tokenizer)

                    # Build full prompt (same as normal generation, thinking controls
                    # included — this lane used to drop them entirely, so a structured
                    # request claimed a level/off with no artifact in the Outlines
                    # prompt; adversarial find V2-F1, 2026-08-20)
                    processed_prompt = prompt
                    full_prompt = self._build_prompt(
                        processed_prompt,
                        messages,
                        system_prompt,
                        tools,
                        prefilled_modules=prompt_cache_prefilled_modules,
                        enable_thinking=(
                            mlx_enable_thinking if isinstance(mlx_enable_thinking, bool) else None
                        ),
                        reasoning_effort=(
                            mlx_reasoning_effort if isinstance(mlx_reasoning_effort, str) else None
                        ),
                    )

                    # Create constrained generator with JSON schema
                    self.logger.debug(
                        f"Using Outlines native structured output for {response_model.__name__}"
                    )
                    # Output cap: after the boundary rename callers pass
                    # max_output_tokens (max_tokens here is the CONTEXT
                    # WINDOW, never the output cap — adversarial find
                    # 2026-07-13: structured truncation-retry bumps never
                    # reached this lane).
                    outlines_max_out = (
                        kwargs.get("max_output_tokens")
                        or kwargs.get("max_tokens")
                        or self.max_output_tokens
                        or 512
                    )
                    # Sampling controls ride along (Outlines forwards extra kwargs to
                    # `mlx_lm.generate`). This lane passed none, so once it started
                    # returning it would have ignored temperature/top_p/top_k/seed
                    # without a word — greedy decoding whatever the caller asked for.
                    outlines_gen_kwargs = self._prepare_generation_kwargs(**kwargs)
                    outlines_sampler = self._build_mlx_sampler(
                        outlines_gen_kwargs.get("temperature", self.temperature),
                        outlines_gen_kwargs.get("top_p", 0.9),
                        outlines_gen_kwargs.get("top_k"),
                    )
                    outlines_seed = outlines_gen_kwargs.get("seed")
                    if outlines_seed is not None:
                        import mlx.core as mx

                        mx.random.seed(outlines_seed)
                    outlines_started = time.time()
                    generator = self._outlines_model(
                        full_prompt,
                        outlines.json_schema(response_model),
                        max_tokens=int(outlines_max_out),
                        **({"sampler": outlines_sampler} if outlines_sampler is not None else {}),
                    )

                    # Validate and return
                    # Outlines 1.x returns the constrained JSON as a STRING.
                    # `model_validate(str)` always raises, so this lane used to run a
                    # full constrained generation, discard the correct result at debug
                    # level and generate again on the prompted lane (2026-09-17).
                    validated_obj = (
                        response_model.model_validate_json(generator)
                        if isinstance(generator, (str, bytes, bytearray))
                        else response_model.model_validate(generator)
                    )

                    # `GenerateResponse` has no `validated_object` field. Passing one raised
                    # TypeError right here, AFTER a successful constrained generation, and
                    # the handler below filed it as "Outlines generation failed" — so this
                    # lane never once returned: every structured call generated twice. The
                    # structured handler validates `content` itself; nothing reads the object.
                    content = validated_obj.model_dump_json()
                    usage = self._calculate_usage(full_prompt, content)
                    return GenerateResponse(
                        content=content,
                        model=self.model,
                        finish_reason="stop",
                        usage=usage,
                        gen_time=round((time.time() - outlines_started) * 1000, 1),
                    )
                except Exception as e:
                    # If native_outlines was explicitly requested, don't fall back
                    if self.structured_output_method == "native_outlines":
                        return GenerateResponse(
                            content=f"Error: Outlines native structured output failed: {str(e)}",
                            model=self.model,
                            finish_reason="error",
                        )
                    # Otherwise fall back to prompted approach — LOUDLY. At debug level this
                    # hid a lane that failed on every call for the cost of a whole generation.
                    self.logger.warning(
                        f"#FALLBACK Outlines native structured output failed "
                        f"({type(e).__name__}: {e}); generating again on the prompted lane."
                    )
                    # Continue with normal generation below

        # Handle media content first if present
        processed_prompt = prompt
        media_enrichment = None
        # Image parts this transport could not carry (delegated-sight honesty,
        # 2026-08-21). This provider generates from a TEXT prompt: a structured
        # multimodal message is reduced to its text part below, and every
        # failure path here continues without the media. Both were silent, so
        # `generate(media=[an image])` returned an ordinary success from a
        # model that never saw it — and `analyze_media` stamped the resulting
        # "there is no image in this conversation" as an observation, complete
        # with provenance (measured on a live run, 2026-08-21). Dropping is
        # still the behavior; claiming it did not happen is what stops here.
        dropped_media: List[str] = report.dropped
        input_embeddings = None
        vision_ids = None
        native_images = []
        if getattr(self, "_mtp_processor", None) is not None and media:
            native_images = [(i, part) for i, part in enumerate(media) if self._is_image_part(part)]
            self._native_media = native_images
            self._native_media_report = report
            report.note_images(len(native_images))
            media = [part for part in media if not self._is_image_part(part)]
            if any(str(getattr(getattr(part, "media_type", None), "value", "")) in ("video", "audio") for part in media):
                raise ProviderAPIError(f"Native MLX lane for {self._native_family_label()} accepts image frames, not audio/video containers; supply extracted images")
        if media:
            # Native sight first. Fills `report.delivered` and returns embeddings;
            # on ANY failure it records a named reason and returns None, and we
            # fall through to the delegated-sight/caption path below unchanged.
            input_embeddings, vision_ids, processed_prompt = self._try_vision_addon(
                prompt, media, report, messages, system_prompt, tools, kwargs
            )
        if media and input_embeddings is None:
            try:
                from ..media.handlers import LocalMediaHandler

                media_handler = LocalMediaHandler(
                    "mlx", self.model_capabilities, model_name=self.model
                )

                # Create multimodal message combining text and media
                multimodal_message = media_handler.create_multimodal_message(prompt, media)
                media_enrichment = getattr(media_handler, "media_enrichment", None)

                # For MLX (local provider), we get text-embedded content
                if isinstance(multimodal_message, str):
                    processed_prompt = multimodal_message
                else:
                    # If we get a structured message, extract the content
                    if isinstance(multimodal_message, dict) and "content" in multimodal_message:
                        if isinstance(multimodal_message["content"], list):
                            # Find text content in the structured message
                            text_content = ""
                            for item in multimodal_message["content"]:
                                if item.get("type") == "text":
                                    text_content = item.get("text", "")
                                    break
                            processed_prompt = text_content or prompt
                            report.drop_parts(
                                str(item.get("type"))
                                for item in multimodal_message["content"]
                                if isinstance(item, dict) and item.get("type") != "text"
                            )
                        else:
                            processed_prompt = str(multimodal_message["content"])
                # Non-image media (documents, PDFs) legitimately reach the model as
                # TEXT on this lane. Record that positively: without it the report
                # stays empty, and an empty report from a reporting provider reads
                # as "a code path forgot", so a successful document request was
                # being judged `not_delivered`.
                if not report.delivered and not report.dropped:
                    embedded = processed_prompt if isinstance(processed_prompt, str) else ""
                    if embedded and embedded != prompt:
                        for i, mc in enumerate(media or []):
                            if self._is_image_part(mc):
                                # An image that reached the text-embedded path was
                                # NOT shown to the model. Claiming delivery here is
                                # the exact inversion this module exists to stop.
                                report.drop("vision_encode_failed")
                                continue
                            report.deliver(
                                index=i,
                                kind=str(
                                    getattr(getattr(mc, "media_type", None), "value", "document")
                                ),
                                content=getattr(mc, "content", b""),
                                tokens=max(2, len(embedded) // 4),
                                transport="text_embedded",
                            )
            except ImportError:
                self.logger.warning(
                    'Media processing not available. Install with: pip install "abstractcore[media]"'
                )
                report.drop("media_processing_unavailable")
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                report.drop("media_processing_failed", detail=str(e))
            if dropped_media:
                from ..media.delivery import remedy_for

                reasons = ", ".join(sorted(set(dropped_media)))
                fix = remedy_for(dropped_media)
                self.logger.warning(
                    f"mlx: {len(dropped_media)} media part(s) were NOT sent to the "
                    f"model ({reasons}). The answer is text-only; callers that need "
                    "sight must read response.metadata['media_delivered'] (absent "
                    "here) or ['media_dropped']."
                    + (f" To fix: {fix}." if fix else "")
                )

            # Tell the MODEL, not just the caller. `media_dropped` is a
            # machine-readable record the model never sees, so a blind model kept
            # answering "Yes, I can see it!" and inventing the picture. See
            # `blind_notice` for the two measured runs this comes from.
            from ..media.delivery import blind_notice

            notice = blind_notice(report)
            if notice:
                base = processed_prompt if isinstance(processed_prompt, str) else prompt
                processed_prompt = f"{notice}\n\n{base}"

        # Build full prompt with tool support
        if native_images:
            processed_prompt = "<|vision_start|><|image_pad|><|vision_end|>" * len(native_images) + "\n" + processed_prompt
        full_prompt = self._build_prompt(
            processed_prompt,
            messages,
            system_prompt,
            tools,
            prefilled_modules=prompt_cache_prefilled_modules,
            enable_thinking=mlx_enable_thinking if isinstance(mlx_enable_thinking, bool) else None,
            reasoning_effort=(
                mlx_reasoning_effort if isinstance(mlx_reasoning_effort, str) else None
            ),
        )

        # MLX generation parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        temperature = generation_kwargs.get("temperature", self.temperature)
        top_p = generation_kwargs.get("top_p", 0.9)
        top_k = generation_kwargs.get("top_k")
        seed_value = generation_kwargs.get("seed")
        prompt_cache = None
        prompt_to_feed: Any = full_prompt
        fed_ids_to_record: Optional[List[int]] = None
        cache_telemetry: Optional[Dict[str, Any]] = None
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if getattr(self, "_mtp_processor", None) is not None and prompt_cache_key:
            # Native VLM APC owns recurrent/QSA snapshots and receives the FULL
            # prompt. Never run mlx-lm's suffix/delta discipline on these states.
            self._native_cache_key = str(prompt_cache_key)
            cache_telemetry = {"mode": "key", "key": prompt_cache_key, "backend": "mlx_vlm_apc"}
            prompt_cache_key = None
        # A turn whose prompt is fed as EMBEDDINGS must never enter key-mode delta
        # feed. `_prepare_cache_delta_feed` matches by longest-common-prefix over
        # TOKEN IDS, and an image turn's ids contain N identical placeholder tokens
        # — so the same question with a DIFFERENT image of the same pixel size
        # produces byte-identical ids, matches perfectly, and the cache serves KV
        # built from the previous image. No exception, wrong answer. Nulling the
        # local leaves the stored cache and its fed-id record untouched and
        # mutually consistent. See backlog 0843.
        if input_embeddings is not None:
            prompt_cache_key = None
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            cache_key = prompt_cache_key.strip()
            prompt_cache = self._prompt_cache_store.get(cache_key)
            if prompt_cache is None:
                self.prompt_cache_set(cache_key, make_default=False)
                prompt_cache = self._prompt_cache_store.get(cache_key)
            # Telemetry struct (0819, runtime seam condition): the ledger
            # must be able to explain 90s-vs-2s turns — mode/key, the
            # decision outcome, MEASURED cached/fed token counts, binding
            # identity when artifact-bound, degraded reason when degraded.
            cache_telemetry = {"mode": "key", "key": cache_key}
            # Delta feed over warm caches (B2): trim to the shared token
            # prefix and feed only the suffix — never re-prefill the whole
            # transcript on top of its own KV. Callers that pass `messages`
            # re-send the whole logical context (delta discipline applies);
            # prompt-only callers (CachedSession KV mode) append by contract.
            # `messages=[]` IS full-context ("empty so far" — key-mode turn
            # one); only `messages=None` means prompt-only (P2-8).

            def _stable_head() -> Optional[str]:
                # This request minus its FINAL turn, through the same renderer that
                # built `full_prompt` (the final turn is `prompt` when one is given,
                # else the last message).
                if isinstance(processed_prompt, str) and processed_prompt.strip():
                    head_messages = messages
                elif messages:
                    head_messages = list(messages)[:-1]
                else:
                    return None
                return self._build_prompt_fragment(
                    prompt="",
                    messages=head_messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    add_generation_prompt=False,
                    prefilled_modules=prompt_cache_prefilled_modules,
                    enable_thinking=(
                        mlx_enable_thinking if isinstance(mlx_enable_thinking, bool) else None
                    ),
                    reasoning_effort=(
                        mlx_reasoning_effort if isinstance(mlx_reasoning_effort, str) else None
                    ),
                )

            prompt_cache, prompt_to_feed, fed_ids_to_record = self._prepare_cache_delta_feed(
                cache_key,
                prompt_cache,
                full_prompt,
                full_context=messages is not None,
                telemetry=cache_telemetry,
                stable_head=_stable_head,
            )
            try:
                key_meta = self.prompt_cache_key_meta(cache_key) or {}
                for meta_field in ("bloc_sha256", "artifact_sha256", "binding_id"):
                    value = key_meta.get(meta_field)
                    if isinstance(value, str) and value:
                        cache_telemetry[meta_field] = value
            except Exception:
                pass

        # The encoder returned processor-expanded ids; they must be what the
        # decoder consumes on BOTH branches, since generate_step requires
        # len(prompt) == len(input_embeddings).
        if vision_ids is not None:
            prompt_to_feed = vision_ids

        # Install the multi-scale visual residual for the duration of this
        # generation, and always remove it afterwards -- the wrappers hold THIS
        # request's features, so outliving the request would feed one image's
        # detail into a later, different prompt.
        from contextlib import ExitStack

        from .mlx_vision_addon import deepstack_layers

        _stack = ExitStack()
        _side = getattr(self, "_vision_side", None) or {}
        if input_embeddings is not None and _side.get("deepstack_visual_embeds") is not None:
            _stack.enter_context(
                deepstack_layers(
                    self.llm,
                    _side.get("deepstack_visual_embeds"),
                    _side.get("visual_pos_masks"),
                    int(_side.get("seq_len") or 0),
                )
            )

        # PREFILL STARTS HERE. Announce it before the GPU work, with whatever
        # the cache lane already measured: the key-mode lane knows cached/fed
        # exactly, the native APC lane only learns them once prefill has run,
        # so there we pay one tokenizer encode to report an honest total rather
        # than a plausible zero. Both are skipped entirely when nobody listens.
        if progress.active:
            cached_hint = fed_hint = total_hint = None
            if isinstance(cache_telemetry, dict):
                cached_hint = cache_telemetry.get("cached_tokens")
                fed_hint = cache_telemetry.get("fed_tokens")
            if fed_hint is None and not isinstance(prompt_to_feed, str):
                try:
                    fed_hint = len(prompt_to_feed)
                except TypeError:
                    fed_hint = None
            if cached_hint is not None and fed_hint is not None:
                total_hint = int(cached_hint) + int(fed_hint)
            elif isinstance(full_prompt, str) and full_prompt:
                encoded = self._encode_prompt_token_ids(full_prompt)
                total_hint = len(encoded) if encoded is not None else None
            progress.prefill(
                prompt_tokens=total_hint, cached_tokens=cached_hint, fed_tokens=fed_hint,
            )

        try:
            if stream:
                if (
                    fed_ids_to_record
                    and isinstance(prompt_cache_key, str)
                    and prompt_cache_key.strip()
                ):
                    # Recorded eagerly: the stream feeds lazily, but the ids are
                    # deterministic and a mid-stream failure self-heals at the
                    # next call through the min(lcp, cache_len) guard.
                    self._record_fed_token_ids(prompt_cache_key.strip(), fed_ids_to_record)
                _streamed = self._stream_generate_with_tools(
                    prompt_to_feed,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    tools,
                    kwargs.get("tool_call_tags"),
                    seed_value,
                    prompt_cache,
                    input_embeddings=input_embeddings,
                    progress=progress,
                    cancel_event=cancel_event,
                    usage_prompt=full_prompt,
                    prompt_cache_telemetry=cache_telemetry,
                )

                # The chat template may END the prompt inside an opened thinking
                # block (Qwen3.x with thinking on renders `<think>\n`): the model
                # then writes reasoning first and only the CLOSING tag. Say so
                # on a leading empty chunk, so the stream shows that reasoning
                # as it is generated instead of holding it until `</think>`.
                from ..architectures.response_postprocessing import THINKING_OPENED_BY_PROMPT

                _opened = self._prompt_opened_thinking(full_prompt)

                # The streamed generator is consumed AFTER this function returns,
                # so the residual has to stay installed until it is exhausted --
                # closing here would remove it before the prompt pass runs.
                def _guarded_stream(_inner=_streamed, _st=_stack, _opened=_opened):
                    try:
                        if _opened:
                            yield GenerateResponse(
                                content="", model=self.model,
                                metadata={THINKING_OPENED_BY_PROMPT: True},
                            )
                        for _chunk in _inner:
                            yield _chunk
                    finally:
                        try:
                            _inner.close()
                        finally:
                            _st.close()

                return _guarded_stream()
            else:
                response = self._single_generate(
                    prompt_to_feed,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    seed_value,
                    prompt_cache,
                    usage_prompt=full_prompt,
                    input_embeddings=input_embeddings,
                    progress=progress,
                    cancel_event=cancel_event,
                )
                if (
                    fed_ids_to_record
                    and isinstance(prompt_cache_key, str)
                    and prompt_cache_key.strip()
                ):
                    if response.finish_reason != "error":
                        # Deliberate: the record holds FED ids only, not the
                        # reply the model just generated — re-tokenized reply
                        # text is not guaranteed token-identical to the
                        # sampled ids, so the next call trims the generated
                        # tokens and re-prefills the reply as suffix (small,
                        # bounded cost; never a correctness risk). Do not
                        # "optimize" by extending the record from reply text.
                        self._record_fed_token_ids(prompt_cache_key.strip(), fed_ids_to_record)
                if cache_telemetry is not None:
                    # Both lanes: the streamed lane attaches the same record to
                    # its terminal chunk (`_stream_generate`), so a runtime that
                    # streams records exactly what a non-streamed call does.
                    response.metadata = dict(response.metadata or {})
                    response.metadata["prompt_cache"] = self._final_prompt_cache_telemetry(
                        cache_telemetry
                    )
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    response.metadata = merge_enrichment_metadata(
                        response.metadata, media_enrichment
                    )
                # Speculation telemetry rides on every response where the caller
                # asked for it, whether or not it worked -- a request that
                # quietly did nothing is the failure mode the contract targets.
                # `used` is computed from what this call actually ran: the MTP
                # lane declines image requests, so sight silently disables it.
                spec_outcome = self._mtp_outcome(bool(getattr(self, "_mtp_last_used", False)))
                if spec_outcome.requested or spec_outcome.reason:
                    # `reason` alone is enough to report: a per-call
                    # `speculation={'mode':'off'}` on an accelerated provider
                    # requested nothing, yet "the lane exists and was skipped
                    # for this call" is exactly what the caller needs to see.
                    response.metadata = dict(response.metadata or {})
                    response.metadata["speculation"] = spec_outcome.to_metadata()

                # Handle tool execution for prompted models
                if tools and self.tool_handler.supports_prompted and response.content:
                    execution_kwargs = ({"execute_tools_param": getattr(self, "_native_execute_tools", None)}
                                        if getattr(self, "_mtp_processor", None) is not None else {})
                    response = self._handle_prompted_tool_execution(response, tools, **execution_kwargs)

                return response

        except Exception as e:
            _stack.close()
            if self.supports_concurrent_generation():
                raise
            from ..exceptions import GenerationCancelledError

            if isinstance(e, GenerationCancelledError):
                # A host cancel is never an "Error: ..." answer with
                # finish_reason=error: the caller must see the cancel itself
                # (observed live: the runtime recorded the stopped call as
                # COMPLETED with this error text as its content).
                raise
            return GenerateResponse(
                content=f"Error: {str(e)}", model=self.model, finish_reason="error"
            )
        finally:
            # The sync path is done with the model by now; the streaming path
            # closes its own stack when the generator is exhausted (above).
            if not stream:
                _stack.close()

    def _build_prompt(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        prefilled_modules: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Build prompt for MLX model with tool support."""
        return self._build_prompt_fragment(
            prompt=str(prompt or ""),
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=True,
            prefilled_modules=prefilled_modules,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )

    def _build_mlx_sampler(
        self, temperature: float, top_p: float, top_k: Optional[int] = None
    ) -> Optional[Any]:
        """Create an mlx-lm sampler from AbstractCore generation parameters."""
        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception:
            return None
        try:
            temp_value = float(temperature)
        except Exception:
            temp_value = 0.0
        try:
            top_p_value = float(top_p)
        except Exception:
            top_p_value = 0.0
        try:
            top_k_value = int(top_k) if top_k is not None else 0
        except Exception:
            top_k_value = 0
        return make_sampler(
            temp=max(0.0, temp_value),
            top_p=max(0.0, top_p_value),
            top_k=max(0, top_k_value),
        )

    def _try_vision_addon(self, prompt, media, report, messages, system_prompt, tools, kwargs):
        """Try native sight. Returns ``(embeddings, ids, processed_prompt)``.

        Never raises: every failure records a named reason on the report and
        returns ``(None, None, prompt)`` so the caller falls through to the
        existing delegated-sight path, which produces a caption with provenance.
        A hard exception here would throw away a subsystem this codebase already
        built, to produce a worse outcome.
        """
        from pathlib import Path

        from ..media.types import MediaType
        from ..media.delivery import (
            MLX_VLM_NOT_INSTALLED,
            VISION_ENCODE_FAILED,
            VISION_FAMILY_UNSUPPORTED,
            VISION_MULTI_IMAGE_UNSUPPORTED,
        )

        # Select by CONTENT, not by file extension: `detect_media_type` classifies
        # from the suffix and its table omits real image formats, so a genuine
        # JPEG named `.jfif` was classified DOCUMENT, skipped this lane entirely,
        # and got embedded as text -- while `analyze_media`'s decode gate, which
        # sniffs content, let it through. See `_is_image_part`.
        self._vision_side = {}
        images = [mc for mc in (media or []) if self._is_image_part(mc)]
        # Recorded before any refusal below, so the report can tell "no image was
        # asked for" apart from "an image was asked for and none landed" -- the
        # difference between a normal text turn and a blind one.
        report.note_images(len(images))
        if not images:
            return None, None, prompt  # documents/PDFs keep the existing path
        # The capability registry is the source of truth for what the MODEL can
        # do. A checkpoint the registry does not declare sighted is not served
        # here, even if a vision tower is present on disk -- the registry is where
        # that claim belongs, and disagreeing with it silently would create a
        # second source of truth. The on-disk and runtime checks below answer a
        # different question: whether this TRANSPORT can carry the pixels.
        if not (self.model_capabilities or {}).get("vision_support", False):
            report.drop(
                "vision_not_declared",
                detail=f"{self.model} is not declared vision-capable in the "
                "model capability registry",
            )
            return None, None, prompt
        if not getattr(self, "_vision_usable", False):
            reason = getattr(self, "_vision_reason", None)
            if reason:
                report.drop(reason, detail=str(getattr(self, "_vision_info", {}).get("model_type")))
            return None, None, prompt
        if len(images) > 1:
            report.drop(
                VISION_MULTI_IMAGE_UNSUPPORTED,
                detail=f"{len(images)} images; this lane carries one",
            )
            return None, None, prompt

        try:
            from .mlx_vision_addon import MLXVisionAddOn, VisionAddOnUnavailable

            try:
                from mlx_lm.utils import does_model_support_input_embeddings

                if not does_model_support_input_embeddings(self.llm):
                    report.drop(
                        VISION_FAMILY_UNSUPPORTED,
                        detail="loaded mlx-lm model does not accept input_embeddings",
                    )
                    return None, None, prompt
            except ImportError:
                pass

            # Built once per loaded checkpoint, under a lock: the gateway serves
            # requests from worker threads, and two concurrent first-image calls
            # would otherwise each load a vision tower and race to publish it.
            addon = getattr(self, "_vision_addon", None)
            if addon is None:
                with self._vision_addon_lock:
                    addon = self._vision_addon
                    if addon is None:
                        addon = MLXVisionAddOn(
                            self._resolved_model_id,
                            self.tokenizer,
                            self._vision_info,
                            self.llm,
                        )
                        self._vision_addon = addon

            processed = f"{addon.placeholder}{prompt}"
            # Render with the SAME thinking controls the text path uses. The
            # encoder tokenizes this string and its ids are what the decoder
            # consumes, so rendering it differently would silently disable
            # thinking control on the vision lane -- which measurably changed the
            # answer (a model that reasons freely can talk itself out of a
            # correct reading, and on a small budget spends the whole allowance
            # thinking and returns empty content).
            mlx_enable_thinking = kwargs.get("_acore_mlx_enable_thinking")
            mlx_reasoning_effort = kwargs.get("_acore_mlx_reasoning_effort")
            full = self._build_prompt(
                processed,
                messages,
                system_prompt,
                tools,
                enable_thinking=(
                    mlx_enable_thinking if isinstance(mlx_enable_thinking, bool) else None
                ),
                reasoning_effort=(
                    mlx_reasoning_effort if isinstance(mlx_reasoning_effort, str) else None
                ),
            )
            paths = [self._materialize_image(mc) for mc in images]
            # Hash the bytes actually handed to the encoder. Hashing a parallel
            # field lets the record attest to an image the model never saw.
            encoded_bytes = Path(paths[0]).read_bytes()
            ids, embeds, n_tokens, fidelity, side = addon.compute_embeddings(self.llm, full, paths)
        except Exception as exc:
            from ..media.delivery import remedy_for

            reason = getattr(exc, "reason", None) or VISION_ENCODE_FAILED
            report.drop(reason, detail=str(exc))
            fix = remedy_for([reason])
            self.logger.warning(
                f"mlx vision add-on unavailable ({reason}): {exc}"
                + (f" — to fix: {fix}" if fix else "")
            )
            return None, None, prompt

        report.deliver(
            index=0,
            kind="image",
            content=encoded_bytes,
            tokens=n_tokens,
            transport="mlx_vision_addon",
            # ADR 0001: annotate best-effort behaviour rather than absorbing it.
            # Only M-RoPE families lose positional fidelity here; `fidelity` adds
            # any encoder side channel input_embeddings could not carry.
            fidelity=(
                (("rope_1d_substituted",) if self._vision_info.get("uses_mrope") else ())
                + tuple(fidelity)
            ),
        )
        self._vision_side = side
        return embeds, ids, processed

    @staticmethod
    def _is_image_part(mc) -> bool:
        """Is this media part an image, judged by its bytes?

        The declared `media_type` is honoured when it already says IMAGE. When it
        does not, the bytes are sniffed, because extension-based classification
        misses several real image formats (.jfif, .jpe, .heic, .heif, .avif,
        .jp2, .pjpeg) and would silently route them away from the vision lane
        while `analyze_media`'s own decode gate -- which sniffs content -- lets
        them through.
        """
        from pathlib import Path as _Path

        from ..media.types import MediaType

        if getattr(mc, "media_type", None) is MediaType.IMAGE:
            return True
        if str(getattr(mc, "mime_type", "") or "").startswith("image/"):
            return True
        try:
            import base64 as _b64
            import io

            from PIL import Image

            # The FILE first. When a part was misclassified as a document its
            # `content` holds EXTRACTED TEXT, not the original bytes, so sniffing
            # content would miss exactly the case this exists to catch.
            path = getattr(mc, "file_path", None)
            if path and _Path(str(path)).is_file():
                Image.open(str(path)).verify()
                return True
            raw = getattr(mc, "content", None)
            if isinstance(raw, str):
                raw = _b64.b64decode(raw, validate=False)
            if not isinstance(raw, (bytes, bytearray)) or not raw:
                return False
            Image.open(io.BytesIO(bytes(raw))).verify()
            return True
        except Exception:
            return False

    @staticmethod
    def _materialize_image(mc) -> str:
        """A filesystem path the vision processor can open."""
        import base64 as _b64
        import tempfile
        from pathlib import Path as _Path

        path = getattr(mc, "file_path", None)
        if path and _Path(str(path)).is_file():
            return str(path)
        raw = getattr(mc, "content", None)
        data = _b64.b64decode(raw) if isinstance(raw, str) else raw
        suffix = ".png"
        mime = str(getattr(mc, "mime_type", "") or "")
        if "jpeg" in mime or "jpg" in mime:
            suffix = ".jpg"
        fd = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        fd.write(data)
        fd.close()
        return fd.name

    def _single_generate(
        self,
        prompt: Any,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
        usage_prompt: Optional[str] = None,
        *,
        input_embeddings: Optional[Any] = None,
        progress: Optional[Any] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> GenerateResponse:
        """Generate single response.

        `cancel_event` (host Stop, generation_cancel.py): when supplied the call
        is ALSO driven through `_observed_generate`, which checks it per sampled
        token, closes the generator and raises `GenerationCancelledError`.

        `prompt` may be a rendered string OR a token-id list (the delta-feed
        suffix over a warm cache — mlx_lm accepts both). `usage_prompt` is the
        full logical prompt for usage accounting, so a suffix feed does not
        under-report prompt tokens.

        `progress` is a `TextProgressEmitter`. When one is subscribed the call
        is driven through `stream_generate_fn` instead of `generate_fn` so the
        first sampled token — the end of prefill — is observable; the joined
        text and every downstream count are identical, because mlx-lm's
        `generate` IS `"".join(r.text for r in stream_generate(...))` and the
        native adapter's two lanes consume the same runtime stream.
        """

        # Handle seed parameter (MLX supports seed via mx.random.seed)
        if seed is not None and getattr(self, "_native_runtime", None) is None:
            import mlx.core as mx

            mx.random.seed(seed)
            self.logger.debug(f"Set MLX random seed to {seed} for deterministic generation")

        # Track generation time
        start_time = time.time()
        if getattr(self, "_mtp_processor", None) is not None:
            sampler_kwargs = {"temperature": temperature, "top_p": top_p, "top_k": top_k or 0, "seed": seed}
            sampler_kwargs.update(getattr(self, "_native_sampling_kwargs", {}))
        else:
            sampler = self._build_mlx_sampler(temperature, top_p, top_k)
            sampler_kwargs = {"sampler": sampler} if sampler is not None else {}

        # Try different MLX API signatures
        try:
            # Try new mlx-lm API
            # Only pass the kwarg when we actually have embeddings, so a
            # text-only call is byte-identical to the previous call site.
            embed_kwargs = (
                {"input_embeddings": input_embeddings} if input_embeddings is not None else {}
            )
            if (progress is not None and getattr(progress, "active", False)) or cancel_event is not None:
                if progress is None:
                    from .generation_progress import TextProgressEmitter

                    progress = TextProgressEmitter(None, provider="mlx", model=self.model)
                response_text = self._observed_generate(
                    progress,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    prompt_cache=prompt_cache,
                    sampler_kwargs=sampler_kwargs,
                    embed_kwargs=embed_kwargs,
                    cancel_event=cancel_event,
                )
            else:
                response_text = self.generate_fn(
                    self.llm,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                    prompt_cache=prompt_cache,
                    **sampler_kwargs,
                    **embed_kwargs,
                )
        except TypeError:
            if input_embeddings is not None or getattr(self, "_mtp_processor", None) is not None:
                # The legacy-signature retry below drops prompt_cache AND the
                # embeddings and re-runs text-only, and the bare `except` under it
                # substitutes a canned sentence. Either would answer from text
                # while the lane had already claimed sight. Fail closed instead.
                raise
            try:
                # Try older API without parameters
                response_text = self.generate_fn(self.llm, self.tokenizer, prompt)
            except:
                # Fallback to basic response
                response_text = (
                    str(usage_prompt or prompt)
                    + " I am an AI assistant powered by MLX on Apple Silicon."
                )

        gen_time = round((time.time() - start_time) * 1000, 1)

        raw_text = response_text.strip()
        generated, reasoning = self._postprocess_generated_text(
            raw_text, thinking_opened_by_prompt=self._prompt_opened_thinking(usage_prompt)
        )
        metadata = {"reasoning": reasoning} if reasoning else None

        native_result = getattr(self, "_mtp_last_result", None) if getattr(self, "_mtp_processor", None) is not None else None
        if native_result is not None:
            metadata = dict(metadata or {})
            metadata.update(self._native_result_metadata(native_result))
        usage, finish_reason = self._reply_usage_and_finish(
            raw_text=raw_text,
            reasoning=reasoning,
            prompt=prompt,
            usage_prompt=usage_prompt,
            input_embeddings=input_embeddings,
            max_tokens=max_tokens,
            native_result=native_result,
        )

        return GenerateResponse(
            content=generated,
            model=self.model,
            finish_reason=finish_reason,
            usage=usage,
            gen_time=gen_time,
            metadata=metadata,
        )

    @staticmethod
    def _progress_counts(response: Any) -> Dict[str, Any]:
        """Per-snapshot counters, whatever lane produced the response object.

        mlx-lm's `GenerationResponse`, mlx-vlm's result and the native
        runtime's `NativeResult` all spell these the same way; only
        `cached_tokens` is native-only, and a lane that cannot measure it
        must report nothing rather than a plausible zero.
        """

        return {
            "generated_tokens": getattr(response, "generation_tokens", None),
            "prompt_tokens": getattr(response, "prompt_tokens", None),
            "cached_tokens": getattr(response, "cached_tokens", None),
            "tokens_per_second": getattr(response, "generation_tps", None),
            "prompt_tokens_per_second": getattr(response, "prompt_tps", None),
        }

    def _prefill_observation_kwargs(
        self,
        progress: Any,
        cancel_event: Optional[threading.Event] = None,
        make_cancelled: Optional[Callable[[int, int], BaseException]] = None,
    ) -> Dict[str, Any]:
        """`stream_generate_fn` kwargs that observe the prompt pass chunk by chunk.

        Mid-prefill progress per lane (see `generation_progress.prefill_progress`):

        * mlx-lm: its own `prompt_progress_callback(processed, total)`, called
          after every `prefill_step_size` chunk over the tokens this call
          feeds. The same hook aborts the prefill on a host cancel.
        * native runtime / in-process mlx-vlm (`_mtp_stream_generate_fn`): the
          emitter's `prefill_progress` rides the provider-internal
          `PREFILL_PROGRESS_KWARG`; the adapter hands it to the runtime's
          `on_prefill_progress` or to the mlx-vlm prefill-bar observer.

        Empty when nobody listens and nothing can cancel, so an unobserved call
        is byte-identical to before.
        """
        report = None
        if progress is not None and getattr(progress, "active", False):
            report = getattr(progress, "prefill_progress", None)
        mlx_lm_lane = (
            getattr(self, "_mtp_processor", None) is None
            and getattr(self, "_native_runtime", None) is None
        )
        if not mlx_lm_lane:
            return {PREFILL_PROGRESS_KWARG: report} if report is not None else {}
        if report is None and cancel_event is None:
            return {}

        def _on_prompt_progress(processed: int, total: int) -> None:
            if cancel_event is not None and cancel_event.is_set() and make_cancelled is not None:
                raise make_cancelled(processed, total)
            if report is not None:
                report(fed_processed=processed, fed_total=total)

        return {"prompt_progress_callback": _on_prompt_progress}

    def _observed_generate(
        self,
        progress: Any,
        *,
        prompt: Any,
        max_tokens: int,
        prompt_cache: Optional[Any],
        sampler_kwargs: Dict[str, Any],
        embed_kwargs: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """`generate_fn` equivalent that reports the prefill→generation boundary.

        The FIRST response carrying a token ends prefill: everything before it
        was prompt processing, and its arrival is the ttft this call will
        report. Afterwards the emitter rate-limits, so a 45 tok/s decode still
        costs the host ~2 records per second, never one per token.

        HOST CANCEL (generation_cancel.py): `cancel_event` is checked before
        every sampled token is accepted; when set, the generator is closed
        (mlx-lm/mlx-vlm stop decoding; the native runtime's handle closes its
        job) and `GenerationCancelledError` is raised with the partial count.
        On the plain mlx-lm lane the event also aborts PREFILL between
        `prefill_step_size` chunks through mlx-lm's `prompt_progress_callback`.
        """
        from .generation_cancel import cancelled_error

        def _cancelled(where: str, generated: Optional[int], partial: str):
            return cancelled_error(provider="mlx", model=self.model, where=where,
                                   generated_tokens=generated, partial_text=partial)

        if cancel_event is not None and cancel_event.is_set():
            raise _cancelled("before decoding started", 0, "")

        extra_kwargs = self._prefill_observation_kwargs(
            progress,
            cancel_event,
            lambda processed, total: _cancelled(f"during prefill ({processed}/{total} prompt tokens)", 0, ""),
        )

        generator = self.stream_generate_fn(
            self.llm,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            **sampler_kwargs,
            **embed_kwargs,
            **extra_kwargs,
        )
        parts: List[str] = []
        counts: Dict[str, Any] = {}
        finish_reason = None
        try:
            for response in generator:
                if cancel_event is not None and cancel_event.is_set():
                    finish_reason = "cancelled"
                    raise _cancelled("while decoding", counts.get("generated_tokens"), "".join(parts))
                parts.append(str(getattr(response, "text", "") or ""))
                counts = self._progress_counts(response)
                finish_reason = getattr(response, "finish_reason", None) or finish_reason
                if counts.get("generated_tokens") is None:
                    # A lane that does not count for us still has tokens: the
                    # arrival of a text segment is the observable event.
                    counts["generated_tokens"] = len(parts)
                progress.generation(**counts)
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                close()
            progress.complete(finish_reason=finish_reason, **counts)
        return "".join(parts)

    def _native_result_metadata(self, native_result: Any) -> Dict[str, Any]:
        """Performance + runtime metadata a native (mlx-vlm / native runtime) result reports."""
        out: Dict[str, Any] = {
            "performance": {
                "prompt_tokens_per_second": native_result.prompt_tps,
                "generation_tokens_per_second": native_result.generation_tps,
                "peak_memory_gb": native_result.peak_memory,
            }
        }
        out.update(getattr(self, "_native_runtime_metadata", {}))
        return out

    def _reply_usage_and_finish(
        self,
        *,
        raw_text: str,
        reasoning: Optional[str],
        prompt: Any,
        usage_prompt: Optional[str],
        input_embeddings: Optional[Any],
        max_tokens: int,
        native_result: Optional[Any],
    ) -> Tuple[Dict[str, int], str]:
        """Usage and finish_reason of one finished reply, identical on BOTH lanes.

        The sync lane (`_single_generate`) and the streamed lane's terminal
        chunk (`_stream_generate`) call this with the same inputs, so a
        streamed record accounts exactly like a non-streamed one.

        `raw_text` is what the model EMITTED (stripped), before thinking is
        split off; `usage_prompt` the full logical prompt (a delta feed passes
        only the suffix as `prompt`).
        """
        usage_text = (
            usage_prompt
            if isinstance(usage_prompt, str)
            else (prompt if isinstance(prompt, str) else "")
        )
        usage = self._calculate_usage(usage_text, raw_text)

        if native_result is not None:
            usage["input_tokens"] = usage["prompt_tokens"] = int(native_result.prompt_tokens)
            usage["cached_input_tokens"] = int(native_result.cached_tokens)

        # Count what the model EMITTED, not what survived post-processing.
        # `generated` is the text left after the thinking block is stripped, so a
        # response whose whole budget went to reasoning reported `output_tokens: 0`
        # -- a caller watching for runaway reasoning saw a free call.
        n_out = self._count_tokens(raw_text)
        if native_result is not None:
            n_out = int(native_result.generation_tokens)
        if n_out is not None:
            usage["output_tokens"] = n_out
            usage["completion_tokens"] = n_out

        # The vision lane feeds expanded TOKEN IDS, and their length is the exact
        # prompt size including the image's thousands of placeholder tokens. The
        # text estimate below cannot see those -- an 11,844-token image was being
        # reported as a 56-token prompt, which silently wrecks the context meter
        # and every downstream cost figure.
        if input_embeddings is not None and not isinstance(prompt, str):
            try:
                n_in = int(len(prompt))
            except Exception:
                n_in = 0
            if n_in > 0:
                usage["input_tokens"] = n_in
                usage["prompt_tokens"] = n_in
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        # A response cut off at `max_tokens` was indistinguishable from one the
        # model chose to end. That is the difference between "the model answered"
        # and "the model was interrupted mid-thought", and callers retry on one
        # and not the other.
        finish_reason = "stop"
        if n_out is not None and isinstance(max_tokens, int) and max_tokens > 0:
            if n_out >= max_tokens:
                finish_reason = "length"
        if reasoning and reasoning.endswith(TRUNCATED_REASONING_MARKER):
            # An unterminated thinking block is proof of truncation even when the
            # token count is off by the tokenizer's accounting of special tokens.
            finish_reason = "length"
        if native_result is not None and getattr(native_result, "finish_reason", None):
            finish_reason = native_result.finish_reason
        return usage, finish_reason

    def _final_prompt_cache_telemetry(self, cache_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """The `metadata["prompt_cache"]` record of a FINISHED call, same on both lanes.

        Key mode (mlx-lm) decides everything before decoding
        (`_prepare_cache_delta_feed`); the native APC lanes only learn
        cached/fed counts and their store/skip counters once the generation
        has run, so this must be called after the last token (sync: after
        `_single_generate`; stream: on the terminal chunk).
        """
        native_result = getattr(self, "_mtp_last_result", None)
        if getattr(self, "_mtp_processor", None) is not None and native_result is not None:
            self._native_apc_telemetry(cache_telemetry, native_result)
        return dict(cache_telemetry)

    def _count_tokens(self, text: str) -> Optional[int]:
        """Exact token count via the loaded tokenizer, or None if it cannot say."""
        if not text:
            return 0
        try:
            return int(len(self.tokenizer.encode(text)))
        except Exception:
            return None

    def _calculate_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Calculate token usage using centralized token utilities."""
        from ..utils.token_utils import TokenUtils

        input_tokens = TokenUtils.estimate_tokens(prompt, self.model)
        output_tokens = TokenUtils.estimate_tokens(response, self.model)
        total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # Keep legacy keys for backward compatibility
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        }

    def _stream_generate(
        self,
        prompt: Any,  # rendered string OR delta-feed token ids
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
        tool_call_tags: Optional[str] = None,
        seed: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
        *,
        input_embeddings: Optional[Any] = None,
        progress: Optional[Any] = None,
        cancel_event: Optional[threading.Event] = None,
        usage_prompt: Optional[str] = None,
        prompt_cache_telemetry: Optional[Dict[str, Any]] = None,
    ) -> Iterator[GenerateResponse]:
        """Generate real streaming response using MLX stream_generate with tool tag rewriting support

        `progress` reports generation snapshots (the caller already emitted the
        prefill event); `cancel_event` is checked per sampled token and stops
        the stream with `GenerationCancelledError` (generation_cancel.py).

        TERMINAL CHUNK (sync/stream parity): the stream ends with exactly one
        chunk carrying `finish_reason` and `usage`, and — when the call ran
        with a prompt-cache key (`prompt_cache_telemetry`) — the same
        `metadata["prompt_cache"]` record the non-streamed lane returns. The
        native runtime's own final result is that chunk; the mlx-lm and
        in-process mlx-vlm lanes, whose per-token responses carry no
        accounting, get an empty-content terminal chunk computed with the sync
        lane's `_reply_usage_and_finish` from the text they streamed. It is
        built only after the source generator is exhausted: the native APC
        lanes learn their cached/fed counts and store counters at that point.
        A failed or cancelled stream has no terminal chunk (sync: no response).
        """
        from ..exceptions import GenerationCancelledError
        from .generation_cancel import cancelled_error

        stream_finish_reason = None
        stream_counts: Dict[str, Any] = {}
        source_gen = None
        emitted_parts: List[str] = []
        native_runtime_lane = getattr(self, "_native_runtime", None) is not None
        try:
            # Handle seed parameter (MLX supports seed via mx.random.seed)
            if seed is not None and getattr(self, "_native_runtime", None) is None:
                import mlx.core as mx

                mx.random.seed(seed)
                self.logger.debug(
                    f"Set MLX random seed to {seed} for deterministic streaming generation"
                )

            # Initialize tool tag rewriter if needed
            rewriter = None
            buffer = ""
            if tool_call_tags:
                try:
                    from ..tools.tag_rewriter import create_tag_rewriter

                    rewriter = create_tag_rewriter(tool_call_tags)
                except ImportError:
                    pass

            # Use MLX's native streaming with minimal parameters
            if getattr(self, "_mtp_processor", None) is not None:
                sampler_kwargs = {"temperature": temperature, "top_p": top_p, "top_k": top_k or 0, "seed": seed}
                sampler_kwargs.update(getattr(self, "_native_sampling_kwargs", {}))
            else:
                sampler = self._build_mlx_sampler(temperature, top_p, top_k)
                sampler_kwargs = {"sampler": sampler} if sampler is not None else {}
            # Only pass the kwarg when we have embeddings, so a text-only stream
            # is byte-identical to the previous call site.
            embed_kwargs = (
                {"input_embeddings": input_embeddings} if input_embeddings is not None else {}
            )
            if cancel_event is not None and cancel_event.is_set():
                raise cancelled_error(provider="mlx", model=self.model,
                                      where="before decoding started", generated_tokens=0)
            source_gen = self.stream_generate_fn(
                self.llm,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                **sampler_kwargs,
                **embed_kwargs,
                **self._prefill_observation_kwargs(progress),
            )
            for response in source_gen:
                if cancel_event is not None and cancel_event.is_set():
                    stream_finish_reason = "cancelled"
                    raise cancelled_error(provider="mlx", model=self.model, where="while streaming",
                                          generated_tokens=stream_counts.get("generated_tokens"))
                if progress is not None:
                    stream_counts = self._progress_counts(response)
                    stream_finish_reason = getattr(response, "finish_reason", None) or stream_finish_reason
                    if stream_counts.get("generated_tokens") is not None:
                        progress.generation(**stream_counts)
                # Each response has a .text attribute with the new token(s)
                content = response.text
                emitted_parts.append(str(content or ""))

                # Apply tool tag rewriting if enabled
                if rewriter and content:
                    rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                    content = rewritten_content

                usage = None
                metadata = None
                finish_reason = None
                if getattr(self, "_native_runtime", None) is not None:
                    metadata = dict(response.metadata or {})
                    metadata.pop("speculation", None)  # Canonical provider outcome is attached above.
                    finish_reason = response.finish_reason
                    if finish_reason:
                        usage = {"input_tokens": response.prompt_tokens, "prompt_tokens": response.prompt_tokens,
                                 "output_tokens": response.generation_tokens, "completion_tokens": response.generation_tokens,
                                 "cached_input_tokens": response.cached_tokens,
                                 "total_tokens": response.prompt_tokens + response.generation_tokens}
                        if prompt_cache_telemetry is not None:
                            # The final result was observed (`_observe_native_runtime_result`)
                            # before it was yielded, so the APC counts are final here.
                            metadata["prompt_cache"] = self._final_prompt_cache_telemetry(
                                prompt_cache_telemetry
                            )
                yield GenerateResponse(
                    content=content,
                    model=self.model,
                    finish_reason=finish_reason,
                    usage=usage,
                    metadata=metadata,
                    raw_response=response,
                )

            if not native_runtime_lane:
                # mlx-lm / in-process mlx-vlm: the source is exhausted (its own
                # `finally` has recorded the APC counters), so the reply is final.
                raw_text = "".join(emitted_parts).strip()
                _, reasoning = self._postprocess_generated_text(
                    raw_text, thinking_opened_by_prompt=self._prompt_opened_thinking(usage_prompt)
                )
                native_result = (
                    getattr(self, "_mtp_last_result", None)
                    if getattr(self, "_mtp_processor", None) is not None
                    else None
                )
                terminal_meta: Dict[str, Any] = {}
                if native_result is not None:
                    terminal_meta.update(self._native_result_metadata(native_result))
                usage, finish_reason = self._reply_usage_and_finish(
                    raw_text=raw_text,
                    reasoning=reasoning,
                    prompt=prompt,
                    usage_prompt=usage_prompt,
                    input_embeddings=input_embeddings,
                    max_tokens=max_tokens,
                    native_result=native_result,
                )
                if prompt_cache_telemetry is not None:
                    terminal_meta["prompt_cache"] = self._final_prompt_cache_telemetry(
                        prompt_cache_telemetry
                    )
                yield GenerateResponse(
                    content="",
                    model=self.model,
                    finish_reason=finish_reason,
                    usage=usage,
                    metadata=terminal_meta or None,
                )

        except GenerationCancelledError:
            # A host cancel is never rendered as an "Error: ..." content chunk.
            raise
        except Exception as e:
            if self.supports_concurrent_generation():
                raise
            yield GenerateResponse(
                content=f"Error: {str(e)}", model=self.model, finish_reason="error"
            )
        finally:
            _close = getattr(source_gen, "close", None)
            if callable(_close):
                _close()
            if progress is not None:
                progress.complete(finish_reason=stream_finish_reason, **stream_counts)

    def get_capabilities(self) -> List[str]:
        """Get MLX capabilities"""
        return ["streaming", "chat"]

    def _est_weights_bytes(self) -> Optional[int]:
        """Best-effort in-memory weight size of the loaded MLX model (bytes)."""
        try:
            from mlx.utils import tree_flatten

            arrays = [v for _, v in tree_flatten(self.llm.parameters())]
            drafter = getattr(self, "_mtp_drafter", None)
            if drafter is not None:
                arrays.extend(v for _, v in tree_flatten(drafter.parameters()))
            # Shared target/drafter embeddings must not be counted twice.
            return int(sum(v.nbytes for v in {id(a): a for a in arrays}.values()))
        except Exception:
            return None

    def get_model_residency(
        self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Return Core-owned in-process residency truth for the loaded MLX provider.

        Carries `est_weights_bytes` (best-effort, absent when unknowable) so
        every consumer of the claim — the base `list_loaded_models` record,
        the server's managed residency rows, the runtime's local rows — sees
        the per-model memory footprint without a second probe."""
        _ = kwargs
        task_s = str(task or "text_generation").strip() or "text_generation"
        model_s = str(model or self.model or "").strip()
        loaded = self.llm is not None and self.tokenizer is not None
        claim: Dict[str, Any] = {
            "task": task_s,
            "provider": "mlx",
            "model": model_s,
            "provider_residency_verified": True,
            "provider_resident": loaded,
            "loaded": loaded,
            "state": "loaded" if loaded else "not_loaded",
            "source": "abstractcore.provider.mlx",
        }
        # PROCESS truth on top of instance truth. The weights are shared
        # between provider instances (`_SharedMLXModel` / `NativeSession`) and
        # stay in Metal memory while ANY instance holds them. An instance that
        # answered "not_loaded" after its own unload while a sibling (a boot-
        # time summarizer, an override client, an old runtime) still held the
        # model is how a gateway came to say "No models loaded" over 92 GB of
        # live MLX buffers. So: `loaded`/`resident` mean "these
        # weights are in this process's memory"; `provider_state` says whether
        # THIS instance holds them or only others do.
        try:
            from .mlx_residency import process_residency_for

            row = process_residency_for(model_s or None, getattr(self, "_resolved_model_id", None))
        except Exception:
            row = None
        if loaded:
            est_weights = self._est_weights_bytes()
            if est_weights is not None:
                claim["est_weights_bytes"] = est_weights
            claim["provider_state"] = "loaded"
            if row is not None:
                claim["process_holders"] = int(row.get("holders") or 0)
                claim["held_bytes"] = int(row.get("held_bytes") or 0)
                claim["cache_bytes"] = int(row.get("cache_bytes") or 0)
                claim["process_lane"] = row.get("lane")
        elif row is not None:
            claim.update({
                "provider_resident": True,
                "loaded": True,
                "state": "loaded",
                "provider_state": "resident_via_other_holders",
                "process_holders": int(row.get("holders") or 0),
                "held_bytes": int(row.get("held_bytes") or 0),
                "cache_bytes": int(row.get("cache_bytes") or 0),
                "process_lane": row.get("lane"),
            })
            if isinstance(row.get("weights_bytes"), int):
                claim["est_weights_bytes"] = int(row["weights_bytes"])
        else:
            claim["provider_state"] = "not_loaded"
        return claim

    def validate_config(self) -> bool:
        """Validate MLX model is loaded"""
        return self.llm is not None and self.tokenizer is not None

    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for MLX generation"""
        # For MLX, max_tokens is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _stream_generate_with_tools(
        self,
        full_prompt: Any,  # rendered string OR delta-feed token ids
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_call_tags: Optional[str] = None,
        seed: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
        *,
        input_embeddings: Optional[Any] = None,
        progress: Optional[Any] = None,
        cancel_event: Optional[threading.Event] = None,
        usage_prompt: Optional[str] = None,
        prompt_cache_telemetry: Optional[Dict[str, Any]] = None,
    ) -> Iterator[GenerateResponse]:
        """Stream generate with tool execution at the end

        The terminal chunk (`finish_reason`, `usage`, `metadata["prompt_cache"]`;
        see `_stream_generate`) is held back and yielded LAST on every lane, after
        any provider-side tool-execution text, so "the last chunk carries the
        accounting" holds whatever the lane.

        (`progress`/`cancel_event`: see `_stream_generate`. `_generate_core`
        passed `progress=` here before this signature accepted it, so every
        `stream=True` call on this lane raised TypeError; now fixed.)
        """
        collected_content = ""
        terminal = None
        scheduled = self.supports_concurrent_generation()
        source = self._stream_generate(
            full_prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            tool_call_tags,
            seed,
            prompt_cache,
            input_embeddings=input_embeddings,
            progress=progress,
            cancel_event=cancel_event,
            usage_prompt=usage_prompt,
            prompt_cache_telemetry=prompt_cache_telemetry,
        )
        try:
            for chunk in source:
                collected_content += chunk.content or ""
                if chunk.finish_reason is not None:
                    # A terminal describes the whole public response, including
                    # any explicitly requested provider-side tool execution.
                    terminal = copy.copy(chunk)
                    terminal.content = ""
                    if chunk.content or chunk.tool_calls:
                        partial = copy.copy(chunk)
                        partial.finish_reason = None
                        partial.usage = None
                        yield partial
                else:
                    yield chunk

            if tools and self.tool_handler.supports_prompted and collected_content:
                complete_response = GenerateResponse(
                    content=collected_content, model=self.model, finish_reason="stop"
                )
                execution_kwargs = ({"execute_tools_param": getattr(self, "_native_execute_tools", None)}
                                    if getattr(self, "_mtp_processor", None) is not None else {})
                final_response = self._handle_prompted_tool_execution(complete_response, tools, **execution_kwargs)
                if final_response.content != collected_content:
                    if scheduled:
                        # Base strips tool-call markup before appending results;
                        # slicing by the ORIGINAL text length truncates results.
                        cleaned = self.tool_handler.parse_response(collected_content, mode="prompted").content or ""
                        if not final_response.content.startswith(cleaned):
                            raise ProviderAPIError("Tool execution returned a response without its parsed content prefix")
                        result_text = final_response.content[len(cleaned):]
                    else:
                        result_text = final_response.content[len(collected_content):]
                    if result_text:
                        yield GenerateResponse(content=result_text, model=self.model,
                                               finish_reason=None if (scheduled or terminal is not None) else "stop")
            if terminal is not None:
                yield terminal
        finally:
            if hasattr(source, "close"):
                source.close()

    @classmethod
    def list_available_models(cls, **kwargs) -> List[str]:
        """
        List available MLX models from local caches.

        This scans:
        - every HuggingFace hub cache (`utils.model_cache.hf_hub_cache_dirs`)
        - the LM Studio store (~/.lmstudio/models)

        and keeps the repos `mlx_model_rules.is_mlx_model` classifies as MLX.
        That is the SAME call `HuggingFaceProvider.list_available_models` uses
        to exclude them, so the two lists are complements: no repo can land in
        both dropdowns, and none can fall out of both.

        Args:
            **kwargs: Optional parameters including:
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        from pathlib import Path
        from .model_capabilities import filter_models_by_capabilities
        from .mlx_model_rules import has_local_weights, is_mlx_model

        try:
            model_set = set()

            # Every hub cache a load reads (HF_HUB_CACHE / HF_HOME / the
            # configured cache dir), never only `~/.cache/huggingface/hub`.
            from ..utils.model_cache import hf_hub_cache_dirs

            for hf_cache in hf_hub_cache_dirs():
                for item in hf_cache.iterdir():
                    if item.is_dir() and item.name.startswith("models--"):
                        # Convert models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit to mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
                        model_name = item.name.replace("models--", "").replace("--", "/")

                        # A cache entry is not a model: a repo that was only
                        # RESOLVED leaves `refs/` and no weights, and `_load_model`
                        # refuses it. Offering it puts a model in the picker that
                        # cannot answer a single turn.
                        if is_mlx_model(model_name, local_path=item) and has_local_weights(item):
                            model_set.add(model_name)

            lmstudio_models = Path.home() / ".lmstudio" / "models"
            if lmstudio_models.exists():
                # LM Studio stores models under: ~/.lmstudio/models/<org>/<model>/*
                for org_dir in lmstudio_models.iterdir():
                    if not org_dir.is_dir():
                        continue
                    for model_dir in org_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        model_name = f"{org_dir.name}/{model_dir.name}"
                        if is_mlx_model(model_name, local_path=model_dir):
                            model_set.add(model_name)

            models = sorted(model_set)

            # Apply new capability filtering if provided
            input_capabilities = kwargs.get("input_capabilities")
            output_capabilities = kwargs.get("output_capabilities")
            capability_routes = kwargs.get("capability_routes")

            if input_capabilities or output_capabilities or capability_routes:
                models = filter_models_by_capabilities(
                    models,
                    input_capabilities=input_capabilities,
                    output_capabilities=output_capabilities,
                    capability_routes=capability_routes,
                )

            return models

        except Exception:
            return []
