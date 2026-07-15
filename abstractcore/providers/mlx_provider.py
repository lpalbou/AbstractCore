"""
MLX provider implementation for Apple Silicon.
"""

import copy
import json
import time
import uuid
import inspect
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Iterator, Type, TYPE_CHECKING

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
from ..architectures.response_postprocessing import normalize_assistant_text
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError, format_model_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType

if TYPE_CHECKING:
    from ..media.types import MediaContent


class MLXProvider(BaseProvider):
    """MLX provider for Apple Silicon models with full integration"""

    def __init__(self, model: str = "mlx-community/Mistral-7B-Instruct-v0.1-4bit",
                 structured_output_method: str = "auto", **kwargs):
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
        # Delta-feed bookkeeping (see _prepare_cache_delta_feed): keys that
        # already warned about unknown-composition warm caches, and the last
        # fragment fed by _prompt_cache_backend_append (consumed by
        # prompt_cache_update to extend the fed-token-id record).
        self._delta_feed_warned_keys: set = set()
        self._pending_append_fragment: Optional[str] = None
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
        self._load_model()

    def supports_prompt_cache(self) -> bool:
        """MLX supports KV prompt caches via `mlx_lm.models.cache`."""
        return True

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        """MLX KV caches are mutable and can serve as the context source-of-truth."""
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
        msg_fmt = str((getattr(self, "architecture_config", {}) or {}).get("message_format") or "").strip().lower()
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

    def _apply_provider_thinking_kwargs(self, *, enabled, level=None, kwargs: Dict[str, Any]):
        """Map unified thinking control into MLX prompt serialization state.

        mlx-lm local generation takes an already-serialized prompt. For Qwen reasoning
        templates, the robust local disable control is to serialize the assistant
        generation prompt in no-thinking mode (`<think>\n\n</think>\n\n`) before
        generation. The actual serialization happens in `_build_prompt_fragment`.
        """
        new_kwargs = dict(kwargs or {})
        # Asset-driven: MLX serializes prompts locally, so the robust disable control is the
        # assistant-prefill marker declared as `thinking_control.assistant_prefill_disable`.
        if self._thinking_control_surfaces().assistant_prefill_disable:
            if enabled is False:
                new_kwargs["_acore_mlx_enable_thinking"] = False
                return new_kwargs, ThinkingControlHandling(
                    handled_enable_disable=True,
                    handled_level=False,
                )
            if enabled is True or level is not None:
                new_kwargs["_acore_mlx_enable_thinking"] = True
                return new_kwargs, ThinkingControlHandling(
                    handled_enable_disable=True,
                    handled_level=False,
                )
        return new_kwargs, ThinkingControlHandling()

    def _prompt_cache_backend_create(self) -> Optional[Any]:
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
        """
        if cache_value is None or not token_ids:
            return False
        try:
            import mlx.core as mx
            from mlx_lm.generate import generate_step
        except Exception:
            return False
        try:
            for _ in generate_step(
                mx.array(token_ids), self.llm, max_tokens=0, prompt_cache=cache_value
            ):
                pass  # max_tokens=0 yields nothing; prefill is the side effect
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
            return self._hybrid_snapshots.get(key)

    def _store_hybrid_snapshot(self, key: str, cache_value: Any, token_ids: List[int]) -> None:
        """Keep one snapshot per key (the growing one evicts its predecessor)."""
        self._ensure_hybrid_snapshot_state()
        with self._hybrid_snapshot_lock:
            self._hybrid_snapshots[key] = {"cache": cache_value, "ids": list(token_ids)}

    def _drop_hybrid_snapshot(self, key: str) -> None:
        self._ensure_hybrid_snapshot_state()
        with self._hybrid_snapshot_lock:
            self._hybrid_snapshots.pop(key, None)

    def _capture_hybrid_snapshot(self, key: str, new_ids: List[int]) -> None:
        """Prefill new_ids into a FRESH cache and store it as this key's
        snapshot boundary, so the NEXT full-context call that extends new_ids
        restores it and feeds only the suffix.

        A fresh dedicated prefill (rather than reusing the just-generated cache)
        is what gives a clean, reply-free boundary — the generated cache holds
        new_ids + the sampled reply, and an untrimmable recurrent state cannot
        be rewound to drop the reply. The prefill costs one prompt pass, but it
        is amortized across every subsequent warm turn in the loop; without it
        the architecture pays that pass EVERY turn.
        """
        if not new_ids:
            return
        fresh = self._prompt_cache_backend_create()
        if fresh is None:
            return
        if not self._prefill_tokens_into_cache(fresh, new_ids):
            return
        self._store_hybrid_snapshot(key, fresh, new_ids)

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
                        return None      # unknowable — never report cold
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
            self.prompt_cache_update_key_meta(key, **{self._FED_TOKEN_IDS_META: [int(t) for t in ids]})
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
        cache_len: Optional[int] = raw_count if isinstance(raw_count, int) and raw_count >= 0 else None

        new_ids = self._encode_prompt_token_ids(full_prompt)
        if not new_ids:
            if full_context and (cache_len is None or cache_len > 0):
                # Cannot tokenize deterministically: never risk the double prefill.
                _note("bypassed", reason="prompt could not be tokenized deterministically; warm cache bypassed")
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
            return bool(meta.get("loaded_from") or meta.get("binding_id") or meta.get("artifact_sha256"))

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
            if working is None:
                working = self._prompt_cache_backend_create()
                prefix_len = 0
            if working is None:
                _note("bypassed", reason="fresh cache creation failed; generated without a cache")
                return None, full_prompt, None

            boundary_end = max(len(new_ids) - 1, 0)  # keep ≥1 token to seed decode
            to_prefill = new_ids[prefix_len:boundary_end]
            if to_prefill and not self._prefill_tokens_into_cache(working, to_prefill):
                # Prefill failed: drop the (now-suspect) snapshot and take the
                # plain fresh full feed — correct, just without snapshot savings.
                self._drop_hybrid_snapshot(key)
                return _fresh_full_feed()

            # Snapshot the clean boundary (deepcopy) BEFORE generation mutates
            # `working` with the seed token + reply. One snapshot per key.
            boundary_ids = new_ids[:boundary_end]
            snap_copy = self._prompt_cache_backend_clone(working)
            if snap_copy is not None and boundary_ids:
                self._store_hybrid_snapshot(key, snap_copy, boundary_ids)
            else:
                self._drop_hybrid_snapshot(key)

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
            else:
                _note("rebuilt", cached=0, fed=fed_this_turn)
            return working, seed, new_ids

        # Cold EMPTY cache: if the architecture is untrimmable (hybrid/SSM —
        # empty already reports not-trimmable), take the snapshot lane NOW so
        # turn 1 leaves a reusable boundary for turn 2. Trimmable models keep
        # the plain cold feed (their delta path handles warm reuse by trimming).
        if cold_empty:
            if not _is_artifact_backed() and not self._cache_is_trimmable(cache_value):
                return _hybrid_snapshot_feed()
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
                    "bypassed", cached=0, fed=len(new_ids),
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
                "rebuilt", cached=0, fed=len(new_ids),
                reason="warm cache of unknown token composition; rebuilt fresh",
            )
            return _fresh_full_feed()

        if cache_len is None:
            # Known record but uncountable cache (recurrent-state layers):
            # trim arithmetic is impossible. Same lattice as trim refusal.
            if _is_artifact_backed():
                _note(
                    "bypassed", fed=len(new_ids),
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
                "bypassed", cached=0, fed=len(new_ids),
                reason="prompt diverges from the artifact's recorded prefix; bypassed to protect the shared cache",
            )
            return None, full_prompt, None

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
                    "bypassed", fed=len(new_ids),
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
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if tool_prompt:
                tool_system_prompt = tool_prompt

        # ONE system turn (parity with the GGUF/transformers builders): when
        # this fragment renders BOTH the user system prompt and the tool
        # instructions, they share a single system block — chat templates are
        # trained on exactly one system turn, and a second consecutive block
        # is out-of-distribution (degraded tool-calling, live find on
        # Ornith-1.0-35B, 2026-07-15). When the system module is already
        # prefilled in the KV cache, its block is closed and cannot be
        # reopened, so the tool prompt still enters as its own block below —
        # module-chain appends carry system and tools in separate calls, so
        # their rendered bytes are unchanged by this merge.
        if base_system_prompt and "system" not in prefilled and tool_system_prompt:
            base_system_prompt = f"{base_system_prompt}\n\n{tool_system_prompt}"
            tool_system_prompt = None

        def _as_text(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        arch_cfg = getattr(self, "architecture_config", None) if isinstance(getattr(self, "architecture_config", None), dict) else {}
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
                    role_name = "model" if role.strip().lower() == "assistant" else role.strip().lower()
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

    def _postprocess_generated_text(self, text: str) -> tuple[str, Optional[str]]:
        cleaned, reasoning = normalize_assistant_text(
            str(text or ""),
            architecture_format=getattr(self, "architecture_config", None),
            model_capabilities=getattr(self, "model_capabilities", None),
        )
        msg_fmt = str((getattr(self, "architecture_config", {}) or {}).get("message_format") or "").strip().lower()
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
        _ = kwargs
        if cache_value is None:
            return False

        existing_tokens = self._prompt_cache_backend_token_count(cache_value)
        fragment = self._build_prompt_fragment(
            prompt=str(prompt or ""),
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=bool(add_generation_prompt),
            enable_thinking=kwargs.get("_acore_mlx_enable_thinking"),
            include_bos=not (isinstance(existing_tokens, int) and int(existing_tokens) > 0),
        )
        # Stash for prompt_cache_update's fed-token-id bookkeeping (delta feed,
        # B2): the base method that calls us knows the KEY; we know the exact
        # fragment bytes that were fed. Locked: prepare_modules (base loop)
        # and prompt_cache_update can race on this instance-level stash from
        # different threads — an unlocked write here cross-pollinates records
        # across keys (adversarial find 2026-07-13).
        with self._append_stash_lock:
            self._pending_append_fragment = fragment or None
            self._pending_append_precount = int(existing_tokens or 0)
        if not fragment:
            return True

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
            precount = int(self._pending_append_precount or 0)
            self._pending_append_fragment = None
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
        if not fragment:
            return {self._FED_TOKEN_IDS_META: list(prior_ids)} if prior_ids else None
        fragment_ids = self._encode_prompt_token_ids(fragment) or []
        if not fragment_ids:
            return None
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
            self._pending_append_precount = 0
            ok = super().prompt_cache_update(key, **kwargs)
            fragment = self._pending_append_fragment
            precount = int(self._pending_append_precount or 0)
            self._pending_append_fragment = None
            self._pending_append_precount = 0
        if not ok or not fragment:
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
                "MLX prompt cache saving requires mlx-lm (install: `pip install \"abstractcore[mlx]\"`)."
            ) from e

        out_meta: Dict[str, Any] = dict(meta or {})
        out_meta.setdefault("format", "abstractcore-prompt-cache/v1")
        out_meta.setdefault("provider", str(getattr(self, "provider", "mlx")))
        out_meta.setdefault("model", str(getattr(self, "model", "")))
        resolved_model_id = str(getattr(self, "_resolved_model_id", "") or "").strip()
        if resolved_model_id:
            out_meta.setdefault("model_resolved_id", resolved_model_id)
        out_meta.setdefault("saved_at", datetime.now().isoformat())

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
            try:
                cache_to_save = [layer.to_quantized(group_size=64, bits=8) for layer in cache_obj]
                out_meta["quantized"] = "q8"
            except Exception:
                # Best-effort: fall back to full precision.
                cache_to_save = cache_obj

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

        out_meta_str: Dict[str, str] = {str(k): _meta_value(v) for k, v in out_meta.items() if isinstance(k, str) and k}

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
        _ = kwargs
        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except Exception as e:
            raise ImportError(
                "MLX prompt cache loading requires mlx-lm (install: `pip install \"abstractcore[mlx]\"`)."
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

        new_key = key
        normalized = self._normalize_prompt_cache_key(new_key) if new_key is not None else None
        if normalized is None:
            normalized = f"cache:{uuid.uuid4().hex[:12]}"

        store_meta: Dict[str, Any] = {
            "backend": "mlx",
            "loaded_from": str(filename),
        }
        store_meta.update(meta_dict)
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
        parsed_record = self._parse_persisted_fed_token_ids(store_meta.get(self._FED_TOKEN_IDS_META))
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

    def _load_model(self):
        """Load MLX model and tokenizer"""
        try:
            from mlx_lm import load, generate, stream_generate
            import mlx.core as mx
            import os
            from contextlib import redirect_stdout, redirect_stderr
            from pathlib import Path

            # Respect AbstractCore's offline-first defaults: never download model files on-demand.
            try:
                from ..config.manager import get_config_manager

                _cfg = get_config_manager()
                if _cfg.is_offline_first():
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
            except Exception:
                pass

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
                if hasattr(mx, "device_info") and hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
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
                load_dir = resolve_lmstudio_model_dir(clean_model_name, base_dirs=default_lmstudio_model_dirs())
                if load_dir is None:
                    snap = resolve_hf_snapshot_dir(clean_model_name, cache_dirs=default_hf_hub_cache_dirs())
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
                            deps = manifest.get("dependencies") if isinstance(manifest, dict) else None
                            if isinstance(deps, list) and deps:
                                for dep in deps:
                                    if not isinstance(dep, dict):
                                        continue
                                    for src in dep.get("sources") or []:
                                        if not isinstance(src, dict):
                                            continue
                                        if str(src.get("type") or "").strip().lower() != "huggingface":
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
                                            ggufs = sorted([p for p in lm_dir.glob("*.gguf") if p.is_file()])
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

            # Silence the "Fetching" progress bar by redirecting stdout/stderr
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    try:
                        self.llm, self.tokenizer = load(load_target)
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
                                    model_type = cfg.get("model_type") if isinstance(cfg, dict) else None
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
        except ImportError:
            raise ImportError("MLX dependencies not installed. Install with: pip install mlx-lm")
        except Exception as e:
            # Check if it's a model not found error
            error_str = str(e).lower()
            if "not found" in error_str or "does not exist" in error_str or "failed to load" in error_str:
                available_models = self.list_available_models()
                error_message = format_model_error("MLX", self.model, available_models)
                raise ModelNotFoundError(error_message)
            raise Exception(f"Failed to load MLX model {self.model}: {str(e)}")

    def unload_model(self, model_name: str) -> None:
        """
        Unload the MLX model from memory.

        Clears model and tokenizer references and forces garbage collection
        to free GPU/CPU memory immediately.
        """
        import gc
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                # Clear MLX model
                del self.llm
                self.llm = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # Clear tokenizer
                del self.tokenizer
                self.tokenizer = None

            if hasattr(self, 'generate_fn'):
                self.generate_fn = None

            if hasattr(self, 'stream_generate_fn'):
                self.stream_generate_fn = None

            # Force garbage collection to free memory immediately
            gc.collect()
        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
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
        timeout_value = kwargs.get('timeout')
        if timeout_value is not None:
            import warnings
            warnings.warn(
                f"MLX provider runs models locally on Apple Silicon and does not support timeout parameters. "
                f"Provided timeout={timeout_value} will be ignored and treated as None (unlimited).",
                UserWarning,
                stacklevel=3
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

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with MLX and optional Outlines native structured output"""

        if not self.llm or not self.tokenizer:
            return GenerateResponse(
                content="Error: MLX model not loaded",
                model=self.model,
                finish_reason="error"
            )

        prompt_cache_prefilled_modules = kwargs.pop("prompt_cache_prefilled_modules", None)
        if isinstance(prompt_cache_prefilled_modules, tuple):
            prompt_cache_prefilled_modules = list(prompt_cache_prefilled_modules)
        if isinstance(prompt_cache_prefilled_modules, str):
            prompt_cache_prefilled_modules = [prompt_cache_prefilled_modules]
        if not isinstance(prompt_cache_prefilled_modules, list):
            prompt_cache_prefilled_modules = None
        mlx_enable_thinking = kwargs.get("_acore_mlx_enable_thinking")

        # Native structured output via Outlines (if configured and available)
        should_use_outlines = (
            response_model and
            PYDANTIC_AVAILABLE and
            not stream and
            self.structured_output_method != "prompted"  # Skip if explicitly prompted
        )

        if should_use_outlines:
            # Check if Outlines is required but unavailable
            if self.structured_output_method == "native_outlines" and not OUTLINES_AVAILABLE:
                return GenerateResponse(
                    content="Error: structured_output_method='native_outlines' requires Outlines library. Install with: pip install \"abstractcore[mlx]\"",
                    model=self.model,
                    finish_reason="error"
                )

            # Try Outlines if available (auto or native_outlines mode)
            if OUTLINES_AVAILABLE:
                try:
                    # Cache Outlines MLX model wrapper to avoid re-initialization
                    if not hasattr(self, '_outlines_model') or self._outlines_model is None:
                        self.logger.debug("Creating Outlines MLX model wrapper for native structured output")
                        self._outlines_model = outlines.from_mlxlm(self.llm, self.tokenizer)

                    # Build full prompt (same as normal generation)
                    processed_prompt = prompt
                    full_prompt = self._build_prompt(processed_prompt, messages, system_prompt, tools)

                    # Create constrained generator with JSON schema
                    self.logger.debug(f"Using Outlines native structured output for {response_model.__name__}")
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
                    generator = self._outlines_model(
                        full_prompt,
                        outlines.json_schema(response_model),
                        max_tokens=int(outlines_max_out),
                    )

                    # Validate and return
                    validated_obj = response_model.model_validate(generator)

                    return GenerateResponse(
                        content=validated_obj.model_dump_json(),
                        model=self.model,
                        finish_reason="stop",
                        validated_object=validated_obj
                    )
                except Exception as e:
                    # If native_outlines was explicitly requested, don't fall back
                    if self.structured_output_method == "native_outlines":
                        return GenerateResponse(
                            content=f"Error: Outlines native structured output failed: {str(e)}",
                            model=self.model,
                            finish_reason="error"
                        )
                    # Otherwise fall back to prompted approach
                    self.logger.debug(f"Outlines generation failed, falling back to prompted: {e}")
                    # Continue with normal generation below

        # Handle media content first if present
        processed_prompt = prompt
        media_enrichment = None
        if media:
            try:
                from ..media.handlers import LocalMediaHandler
                media_handler = LocalMediaHandler("mlx", self.model_capabilities, model_name=self.model)

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
                        else:
                            processed_prompt = str(multimodal_message["content"])
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")

        # Build full prompt with tool support
        full_prompt = self._build_prompt(
            processed_prompt,
            messages,
            system_prompt,
            tools,
            prefilled_modules=prompt_cache_prefilled_modules,
            enable_thinking=mlx_enable_thinking if isinstance(mlx_enable_thinking, bool) else None,
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
            prompt_cache, prompt_to_feed, fed_ids_to_record = self._prepare_cache_delta_feed(
                cache_key, prompt_cache, full_prompt,
                full_context=messages is not None,
                telemetry=cache_telemetry,
            )
            try:
                key_meta = self.prompt_cache_key_meta(cache_key) or {}
                for meta_field in ("bloc_sha256", "artifact_sha256", "binding_id"):
                    value = key_meta.get(meta_field)
                    if isinstance(value, str) and value:
                        cache_telemetry[meta_field] = value
            except Exception:
                pass

        try:
            if stream:
                if fed_ids_to_record and isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
                    # Recorded eagerly: the stream feeds lazily, but the ids are
                    # deterministic and a mid-stream failure self-heals at the
                    # next call through the min(lcp, cache_len) guard.
                    self._record_fed_token_ids(prompt_cache_key.strip(), fed_ids_to_record)
                return self._stream_generate_with_tools(
                    prompt_to_feed,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    tools,
                    kwargs.get('tool_call_tags'),
                    seed_value,
                    prompt_cache,
                )
            else:
                response = self._single_generate(
                    prompt_to_feed, max_tokens, temperature, top_p, top_k, seed_value, prompt_cache,
                    usage_prompt=full_prompt,
                )
                if fed_ids_to_record and isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
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
                    # Sync lane only, deliberately: the runtime's durable
                    # llm_call lane forces stream=False, and that ledger is
                    # the consumer this struct exists for.
                    response.metadata = dict(response.metadata or {})
                    response.metadata["prompt_cache"] = dict(cache_telemetry)
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

                # Handle tool execution for prompted models
                if tools and self.tool_handler.supports_prompted and response.content:
                    response = self._handle_prompted_tool_execution(response, tools)

                return response

        except Exception as e:
            return GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _build_prompt(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        prefilled_modules: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
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
        )

    def _build_mlx_sampler(self, temperature: float, top_p: float, top_k: Optional[int] = None) -> Optional[Any]:
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
    ) -> GenerateResponse:
        """Generate single response.

        `prompt` may be a rendered string OR a token-id list (the delta-feed
        suffix over a warm cache — mlx_lm accepts both). `usage_prompt` is the
        full logical prompt for usage accounting, so a suffix feed does not
        under-report prompt tokens.
        """

        # Handle seed parameter (MLX supports seed via mx.random.seed)
        if seed is not None:
            import mlx.core as mx
            mx.random.seed(seed)
            self.logger.debug(f"Set MLX random seed to {seed} for deterministic generation")

        # Track generation time
        start_time = time.time()
        sampler = self._build_mlx_sampler(temperature, top_p, top_k)
        sampler_kwargs = {"sampler": sampler} if sampler is not None else {}

        # Try different MLX API signatures
        try:
            # Try new mlx-lm API
            response_text = self.generate_fn(
                self.llm,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                prompt_cache=prompt_cache,
                **sampler_kwargs,
            )
        except TypeError:
            try:
                # Try older API without parameters
                response_text = self.generate_fn(
                    self.llm,
                    self.tokenizer,
                    prompt
                )
            except:
                # Fallback to basic response
                response_text = str(usage_prompt or prompt) + " I am an AI assistant powered by MLX on Apple Silicon."

        gen_time = round((time.time() - start_time) * 1000, 1)

        generated, reasoning = self._postprocess_generated_text(response_text.strip())
        metadata = {"reasoning": reasoning} if reasoning else None

        usage_text = usage_prompt if isinstance(usage_prompt, str) else (prompt if isinstance(prompt, str) else "")
        return GenerateResponse(
            content=generated,
            model=self.model,
            finish_reason="stop",
            usage=self._calculate_usage(usage_text, generated),
            gen_time=gen_time,
            metadata=metadata,
        )

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
            "completion_tokens": output_tokens
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
    ) -> Iterator[GenerateResponse]:
        """Generate real streaming response using MLX stream_generate with tool tag rewriting support"""
        try:
            # Handle seed parameter (MLX supports seed via mx.random.seed)
            if seed is not None:
                import mlx.core as mx
                mx.random.seed(seed)
                self.logger.debug(f"Set MLX random seed to {seed} for deterministic streaming generation")

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
            sampler = self._build_mlx_sampler(temperature, top_p, top_k)
            sampler_kwargs = {"sampler": sampler} if sampler is not None else {}
            for response in self.stream_generate_fn(
                self.llm,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                **sampler_kwargs,
            ):
                # Each response has a .text attribute with the new token(s)
                content = response.text

                # Apply tool tag rewriting if enabled
                if rewriter and content:
                    rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                    content = rewritten_content

                yield GenerateResponse(
                    content=content,
                    model=self.model,
                    finish_reason=None,  # MLX doesn't provide finish reason in stream
                    raw_response=response
                )

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def get_capabilities(self) -> List[str]:
        """Get MLX capabilities"""
        return ["streaming", "chat"]

    def get_model_residency(self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Return Core-owned in-process residency truth for the loaded MLX provider."""
        _ = kwargs
        task_s = str(task or "text_generation").strip() or "text_generation"
        model_s = str(model or self.model or "").strip()
        loaded = self.llm is not None and self.tokenizer is not None
        return {
            "task": task_s,
            "provider": "mlx",
            "model": model_s,
            "provider_residency_verified": True,
            "provider_resident": loaded,
            "loaded": loaded,
            "state": "loaded" if loaded else "not_loaded",
            "source": "abstractcore.provider.mlx",
        }

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
    ) -> Iterator[GenerateResponse]:
        """Stream generate with tool execution at the end"""
        collected_content = ""

        # Stream the response content
        for chunk in self._stream_generate(
            full_prompt, max_tokens, temperature, top_p, top_k, tool_call_tags, seed, prompt_cache
        ):
            collected_content += chunk.content or ""
            yield chunk

        # Handle tool execution if we have tools and content
        if tools and self.tool_handler.supports_prompted and collected_content:
            # Create complete response for tool processing
            complete_response = GenerateResponse(
                content=collected_content,
                model=self.model,
                finish_reason="stop"
            )

            # Handle tool execution using base method
            final_response = self._handle_prompted_tool_execution(complete_response, tools)

            # If tools were executed, yield the tool results as final chunk
            if final_response.content != collected_content:
                tool_results_content = final_response.content[len(collected_content):]
                yield GenerateResponse(
                    content=tool_results_content,
                    model=self.model,
                    finish_reason="stop"
                )

    @classmethod
    def list_available_models(cls, **kwargs) -> List[str]:
        """
        List available MLX models from local caches.

        This includes:
        - HuggingFace hub cache (~/.cache/huggingface/hub) for any repo containing "mlx"
        - LM Studio cache (~/.lmstudio/models) for any org/model containing "mlx"

        Args:
            **kwargs: Optional parameters including:
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        from pathlib import Path
        from .model_capabilities import filter_models_by_capabilities

        try:
            model_set = set()

            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_cache.exists():
                for item in hf_cache.iterdir():
                    if item.is_dir() and item.name.startswith("models--"):
                        # Convert models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit to mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
                        model_name = item.name.replace("models--", "").replace("--", "/")

                        # Include ANY model with "mlx" in the name (case-insensitive)
                        # This captures: mlx-community/*, */mlx-*, *-mlx-*, etc.
                        if "mlx" in model_name.lower():
                            model_set.add(model_name)

            lmstudio_models = Path.home() / ".lmstudio" / "models"
            if lmstudio_models.exists():
                # LM Studio stores models under: ~/.lmstudio/models/<org>/<model>/*
                for org_dir in lmstudio_models.iterdir():
                    if not org_dir.is_dir():
                        continue
                    # These org folders are MLX by design (model names may not include "mlx")
                    include_all_in_org = org_dir.name.lower() in {"mlx-community", "lmstudio-community"}
                    for model_dir in org_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        model_name = f"{org_dir.name}/{model_dir.name}"
                        if include_all_in_org or "mlx" in model_name.lower():
                            model_set.add(model_name)

            models = sorted(model_set)

            # Apply new capability filtering if provided
            input_capabilities = kwargs.get('input_capabilities')
            output_capabilities = kwargs.get('output_capabilities')
            capability_routes = kwargs.get('capability_routes')

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
