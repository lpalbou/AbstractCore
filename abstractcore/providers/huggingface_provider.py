"""
HuggingFace provider implementation with GGUF support.
Supports both transformers models and GGUF models via llama-cpp-python.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import copy
import hashlib
import json
import platform
import sys
import threading
import time
import uuid
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Union, Iterator, Type, TYPE_CHECKING

# Import config manager to respect offline-first settings
from ..config.manager import get_config_manager

# Get config instance and set offline environment variables if needed
_config = get_config_manager()

# Did the CALLER ask for offline, or did we? `offline_first` defaults to True,
# so the three variables below are almost always ours. That distinction is
# load-bearing exactly once — see `_resolve_bnb_mps_fused_kernel`, which may
# lift OUR flag for a single accelerator-kernel resolution but must never
# override an offline setting the user made deliberately.
_USER_SET_HF_OFFLINE = {
    name: os.environ.get(name)
    for name in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
}

if _config.is_offline_first():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


TRANSFORMERS_AVAILABLE = _module_available("transformers")
LLAMACPP_AVAILABLE = _module_available("llama_cpp")
OUTLINES_AVAILABLE = _module_available("outlines")

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider, PromptCacheCapabilities, PromptCacheRenderedFragment, ThinkingControlHandling
from ..core.types import GenerateResponse
from ..core import degeneration as _degeneration
from ..exceptions import (
    InvalidRequestError,
    ModelArtifactMismatchError,
    ModelNotFoundError,
    format_model_error,
)
from ..tools import UniversalToolHandler, execute_tools, merge_tools_into_system
from ..events import EventType

if TYPE_CHECKING:
    import torch
    from ..media.types import MediaContent


_MPS_GENERATION_LOCK = threading.Lock()
_AUTO_GROWING_LLAMA_RAM_CACHE_CLS = None

_ARTIFACT_LOGGER = None


def _artifact_logger():
    """Module-scoped logger for artifact-resolution decisions (ADR 0009).

    Deliberately NOT `self.logger`: the resolution helpers run on providers built
    with `__new__` (no instance logger), and a resolution choice that cannot be
    logged would be a silent choice. Resolved lazily to keep import cheap.
    """
    global _ARTIFACT_LOGGER
    if _ARTIFACT_LOGGER is None:
        import logging

        _ARTIFACT_LOGGER = logging.getLogger("abstractcore.providers.huggingface")
    return _ARTIFACT_LOGGER


# ---------------------------------------------------------------------------
# torch-MPS short-query SDPA corruption (pytorch#163597) — DETECT AND WARN ONLY
# ---------------------------------------------------------------------------
# MEASURED on torch 2.10.0, macOS 26.3, Apple M5 Max (2026-08-05; artifacts in
# untracked/prompt-cache-bench/results/MPS_INVESTIGATION_RESULTS.md):
# scaled_dot_product_attention on the MPS backend returned NON-DETERMINISTIC and
# numerically WRONG results for q_len <= 8 over kv_len >= 1024 in float16/bfloat16
# (32 identical calls -> 32 distinct results, error 6.35x the output's magnitude
# vs a CPU float64 reference). q_len == 1 is every decode step, so on 2.10 every
# token past ~1024 tokens of context read corrupted attention.
#
# RE-MEASURED on torch 2.13.0 at the same shapes: 1 distinct result / 16 calls,
# rel err 0.004 (bf16 rounding) — the defect is GONE
# (untracked/prompt-cache-bench/cleanremeasure/verify_torch213_kernel.json).
# torch >= 2.11.0 measured clean; <= 2.9 was never measured and is not claimed.
#
# The in-process attention workaround that used to live here has been REMOVED:
# it had no torch version gate, so it fired on every MPS load regardless of
# torch, and it defeated the fused decode kernel (materialising a [B,H,q,kv]
# score matrix plus a repeat_interleave of K/V on every decode step). All that
# remains is a one-time warning on the exact affected configuration.
_MPS_SDPA_DEFECT_WARNED = False


def _warn_if_mps_sdpa_defective(device, model) -> bool:
    """Warn once if this torch build carries the MPS SDPA defect (2.10.x only)."""
    global _MPS_SDPA_DEFECT_WARNED
    if _MPS_SDPA_DEFECT_WARNED or str(device) != "mps":
        return False
    try:
        import torch  # type: ignore
        ver = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
        dtype = next(model.parameters()).dtype
    except Exception:
        return False
    # Version gate is STRICT: 2.10.x only. 2.10.0 measured broken, >= 2.11.0
    # measured clean, <= 2.9 never measured (so no claim, and no slow path).
    # Deliberately NOT narrowed by head_dim: the fault was characterised at
    # head_dim 128, and this is only a warning about a known-bad torch build —
    # under-warning costs silently wrong generations, over-warning costs a line
    # of stderr on a torch release the user should be off anyway.
    if ver != (2, 10) or dtype not in (torch.float16, torch.bfloat16):
        return False
    _MPS_SDPA_DEFECT_WARNED = True
    import warnings
    warnings.warn(
        f"abstractcore: torch {torch.__version__} on the MPS backend returns "
        "non-deterministic, numerically wrong scaled_dot_product_attention for "
        "short queries over a long KV cache (pytorch#163597) — i.e. every token "
        "generated past ~1024 tokens of context in float16/bfloat16. UPGRADE to "
        "torch >= 2.11 (measured clean); as a stopgap set "
        "ABSTRACTCORE_TRANSFORMERS_ATTN_IMPL=eager, which is slower but exact."
    )
    return True


# The bitsandbytes fused Metal 4-bit kernel, and why abstractcore probes for it.
#
# `bitsandbytes.backends.mps.ops._get_kernel()` resolves
# `kernels-community/bitsandbytes-mps` lazily on the first `Linear4bit` forward.
# It swallows EVERY failure and latches `_kernel_load_failed = True` for the
# lifetime of the process, after which every 4-bit op falls back to
# `dequantize -> F.linear`. The fallback is numerically fine — the model loads
# and answers correctly — it is just about x4 slower to decode, with no warning
# anywhere. That is a silent degradation, which ADR 0001/0009 forbid, and it
# killed three measurement cells on 2026-08-06 as the `kernels` package moved
# between versions underneath a running sweep.
#
# bitsandbytes is third-party, so the probe lives here: run the SAME resolution
# it would run, at model-load time, and say so out loud when it fails.
_BNB_MPS_FUSED_KERNEL_REPO = "kernels-community/bitsandbytes-mps"

# Serialises the brief offline-flag lift below. The lift mutates process-wide
# state, so no two loads may overlap inside it.
_BNB_KERNEL_RESOLVE_LOCK = threading.Lock()

# The lift is attempted at most ONCE per process. On a genuinely disconnected
# machine it costs 7.2 s before huggingface_hub's own connect timeout gives up
# (measured against a black-hole endpoint; bounded, and the flags are restored)
# — acceptable once against a permanent x4 decode penalty, not acceptable on
# every subsequent model load. bitsandbytes' own latch cannot serve as this
# memo: it is only set when ITS `_get_kernel()` runs, and a failed lift never
# reaches that call.
_BNB_KERNEL_LIFT_ATTEMPTED = False


def _resolve_bnb_mps_fused_kernel():
    """Resolve the fused Metal 4-bit kernel, lifting OUR offline flag if needed.

    THE PRODUCT BUG THIS FIXES (measured 2026-08-06, in-process, one variable):
    `offline_first` (default True) sets `HF_HUB_OFFLINE=1` at the top of this
    module, and `kernels.get_kernel` verifies the kernel repo's publisher over
    the Hub API — a check with no offline path. Through the product path the
    resolution therefore fails in 0.008 s, bitsandbytes latches the failure for
    the life of the process, and every `Linear4bit` forward silently falls back
    to `dequantize -> F.linear` at about x4 the cost. A benchmark harness that
    happened to import bitsandbytes BEFORE abstractcore got the fused kernel and
    kept it (`kernels.get_kernel` memoises), which is how a warm NF4 figure of
    0.0681 s came to be published against a product-path reality of 0.2696 s.

    `offline_first` exists to keep model WEIGHTS off the network. It was never
    meant to disable an accelerator. So: try the shipped offline path first; if
    that fails, retry ONCE with our own flag lifted, then put it back.

    Clearing the environment variable is NOT sufficient and that is not an
    oversight — `huggingface_hub` snapshots `HF_HUB_OFFLINE` into
    `constants.HF_HUB_OFFLINE` at ITS import, so the constant must be patched
    too (env-only: still fails in 0.000 s; env + constant: succeeds in 0.608 s).
    Both are restored in `finally`, whatever happens.

    Never lifts a flag the USER set: `_USER_SET_HF_OFFLINE` records the values
    that existed before this module touched them, and a user-set offline flag is
    left exactly as found — those callers get the warning instead.

    The lifted window must cover BITSANDBYTES' OWN `_get_kernel()`, not just
    ours, and that is not belt-and-braces. `kernels.get_kernel` memoises the
    build it returns but re-runs `_check_trust_remote_code` on EVERY call, so a
    resolution we complete and then hand back to a re-armed offline flag buys
    bitsandbytes nothing — measured: our call succeeds via the lift, bitsandbytes'
    next call still returns None and latches. Priming its module-global `_kernel`
    inside the window is what makes the fused path reachable, and once that
    global is set bitsandbytes never calls `get_kernel` again.

    Ordering is load-bearing too: bitsandbytes' `_get_kernel()` is NEVER called
    before a plain `get_kernel` has succeeded, because its `except` arm latches
    `_kernel_load_failed = True` permanently and every later attempt — lifted or
    not — short-circuits on that latch.

    Returns (kernel_or_None, how) where `how` is 'offline', 'network-lift',
    'declined-user-offline' or 'failed'.
    """
    try:
        from kernels import get_kernel  # type: ignore
        from bitsandbytes.backends.mps import ops as bnb_ops  # type: ignore
    except Exception:
        return None, "failed"

    def _prime_bnb():
        """Populate bitsandbytes' module-global `_kernel`. Only ever called
        after a plain `get_kernel` has already succeeded under the same
        conditions, so its latching `except` arm cannot be reached."""
        try:
            return bnb_ops._get_kernel()
        except Exception:  # noqa: BLE001 - bitsandbytes swallows internally
            return None

    # Already resolved earlier in this process (a second provider instance, a
    # second model): bitsandbytes holds the kernel in a module global and will
    # never call `get_kernel` again, so neither should we.
    existing = getattr(bnb_ops, "_kernel", None)
    if existing is not None and not getattr(bnb_ops, "_kernel_load_failed", False):
        return existing, "already-resolved"

    try:
        get_kernel(_BNB_MPS_FUSED_KERNEL_REPO, version=1)
        primed = _prime_bnb()
        if primed is not None:
            return primed, "offline"
    except Exception:
        pass

    # The user asked for offline explicitly — respect it and do not retry.
    if any(str(v or "").strip() not in ("", "0")
           for v in _USER_SET_HF_OFFLINE.values()):
        return None, "declined-user-offline"

    if not _config.is_offline_first():
        return None, "failed"  # nothing of ours to lift; the failure is real

    global _BNB_KERNEL_LIFT_ATTEMPTED
    names = ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
    with _BNB_KERNEL_RESOLVE_LOCK:
        if _BNB_KERNEL_LIFT_ATTEMPTED:
            return None, "failed"  # already tried this process; do not re-stall
        _BNB_KERNEL_LIFT_ATTEMPTED = True
        saved_env = {n: os.environ.get(n) for n in names}
        hub_constants = None
        saved_constant = None
        try:
            import huggingface_hub.constants as hub_constants  # type: ignore

            saved_constant = getattr(hub_constants, "HF_HUB_OFFLINE", None)
        except Exception:
            hub_constants = None
        try:
            for n in names:
                os.environ.pop(n, None)
            if hub_constants is not None:
                hub_constants.HF_HUB_OFFLINE = False
            get_kernel(_BNB_MPS_FUSED_KERNEL_REPO, version=1)
            primed = _prime_bnb()
            return (primed, "network-lift") if primed is not None else (None, "failed")
        except Exception:
            return None, "failed"
        finally:
            for n, v in saved_env.items():
                if v is None:
                    os.environ.pop(n, None)
                else:
                    os.environ[n] = v
            if hub_constants is not None and saved_constant is not None:
                hub_constants.HF_HUB_OFFLINE = saved_constant


def _probe_bnb_mps_fused_kernel() -> Dict[str, Any]:
    """Resolve the bitsandbytes fused Metal 4-bit kernel and report the outcome.

    Returns a record; never raises. `available` is True only when the kernel
    object is live AND bitsandbytes has not already latched its failure flag
    from an earlier attempt in this process.
    """
    record: Dict[str, Any] = {
        "available": False,
        "reason": None,
        "remedy": None,
        "error": None,
        "kernels_version": None,
        "bitsandbytes_version": None,
        "macos_major": None,
        "latched_failed": None,
        "resolved_via": None,
        "bnb_own_call_returned_kernel": None,
        "user_set_offline_flags": {k: v for k, v in _USER_SET_HF_OFFLINE.items()
                                   if v is not None},
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "offline_flags_set": [
            f"{k}={os.environ[k]}" for k in
            ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
            if str(os.environ.get(k) or "").strip() not in ("", "0")
        ],
    }
    kernels_pin_note = (
        "pin `kernels==0.14.1` — kernels <=0.13 resolves the wrong Hub repo type "
        "(the model repo, which carries no torch213 Metal build) and kernels "
        ">=0.15 breaks transformers model imports"
    )
    try:
        import platform

        mac = platform.mac_ver()[0]
        record["macos_major"] = int(mac.split(".")[0]) if mac else 0
    except Exception:
        record["macos_major"] = None
    for pkg, field_name in (("kernels", "kernels_version"), ("bitsandbytes", "bitsandbytes_version")):
        try:
            import importlib.metadata as _md

            record[field_name] = _md.version(pkg)
        except Exception:
            record[field_name] = None

    try:
        from bitsandbytes.backends.mps import ops as _bnb_mps_ops  # type: ignore
    except Exception as e:
        record["reason"] = "bitsandbytes MPS backend not importable"
        record["error"] = f"{type(e).__name__}: {e}"
        record["remedy"] = "install a bitsandbytes build with the MPS backend"
        return record

    # macOS < 26 pre-sets the latch: bitsandbytes never attempts the hub
    # kernel there, so this is a platform limit, not a resolution failure.
    if isinstance(record["macos_major"], int) and record["macos_major"] < 26:
        record["latched_failed"] = True
        record["reason"] = (
            f"macOS {record['macos_major']} is below 26; bitsandbytes does not "
            "attempt the Metal hub kernel on this OS"
        )
        record["remedy"] = (
            "no software fix — the fused Metal kernel needs macOS 26+; use the "
            "MLX or GGUF lane for fast 4-bit on this OS"
        )
        return record

    # Run the real resolution, lifting our own offline flag if that is what is
    # in the way (see `_resolve_bnb_mps_fused_kernel`). `kernels.get_kernel`
    # memoises, so this is the work the first Linear4bit forward would have
    # done, moved to load time — not extra work.
    kernel, how = _resolve_bnb_mps_fused_kernel()
    record["resolved_via"] = how
    if kernel is None:
        try:
            from kernels import get_kernel  # type: ignore

            get_kernel(_BNB_MPS_FUSED_KERNEL_REPO, version=1)
            e = RuntimeError("resolution failed under the offline flag")
        except Exception as exc:  # noqa: BLE001 - captured only to name it
            e = exc
        record["reason"] = f"{_BNB_MPS_FUSED_KERNEL_REPO} did not resolve"
        record["error"] = f"{type(e).__name__}: {e}"
        # `kernels` verifies the publisher over the Hub API and has no offline
        # path, so abstractcore's own offline-first env (HF_HUB_OFFLINE=1, set
        # at import) fails the check even when the kernel is already in the
        # local cache. Isolated A/B on 2026-08-06: HF_HUB_OFFLINE=1 alone, with
        # nothing else changed, turns a LOADED kernel into None and latches it.
        if "trust status" in str(e) and record["offline_flags_set"]:
            flags = ", ".join(record["offline_flags_set"])
            record["reason"] = (
                f"{_BNB_MPS_FUSED_KERNEL_REPO} could not be trust-verified because "
                f"offline mode ({flags}) blocks the Hub publisher check `kernels` "
                "requires (the kernel itself is cached locally; the check is not). "
                "Measured 2026-08-06: HF_HUB_OFFLINE=1 alone, and "
                "TRANSFORMERS_OFFLINE=1 alone, each turn a loading kernel into None"
            )
            record["remedy"] = (
                "this is abstractcore's own offline-first env, not a packaging "
                "fault: the Hub publisher check has no offline path in `kernels`. "
                "Either allow that one Hub call at load (clear the offline flags "
                "for the load), or accept the fallback and use the MLX/GGUF lane "
                "for fast 4-bit"
            )
        else:
            record["remedy"] = kernels_pin_note
        return record

    # `_resolve_bnb_mps_fused_kernel` already primed bitsandbytes' own global
    # inside whatever window was needed. Re-read it here through bitsandbytes'
    # OWN accessor — the code path every `Linear4bit` forward takes — because
    # that, not our resolution, is what proves the fix reaches the product.
    try:
        bnb_kernel = _bnb_mps_ops._get_kernel()
    except Exception:  # noqa: BLE001 - bitsandbytes never raises here, but
        bnb_kernel = None  # a future version must not break a model load
    record["bnb_own_call_returned_kernel"] = bnb_kernel is not None

    # An earlier attempt in this process may already have latched the failure —
    # in which case the fused path stays dead regardless of what resolves now.
    latched = bool(getattr(_bnb_mps_ops, "_kernel_load_failed", False))
    record["latched_failed"] = latched
    if latched or bnb_kernel is None:
        record["reason"] = (
            "the kernel resolves for us, but bitsandbytes' own `_get_kernel()` "
            f"still returns {bnb_kernel!r} (latched={latched}) — the fused path "
            "is dead for this process's lifetime"
        )
        record["remedy"] = (
            "restart the process: bitsandbytes latches its failure permanently "
            "and exposes no way to clear it, so a first 4-bit load that failed "
            "cannot be rescued by a later one"
        )
        return record

    record["available"] = True
    return record


# We no longer download models - cache-only approach
# huggingface_hub not required for basic operation


def _get_local_model_path(model_name: str) -> Optional[str]:
    """Get local cache path for a HuggingFace model if it exists."""
    # Use centralized configuration for cache directory
    config = _config
    hf_cache_dir = Path(config.config.cache.huggingface_cache_dir).expanduser()

    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = hf_cache_dir / "hub" / model_cache_name / "snapshots"

    if model_cache_path.exists():
        snapshot_dirs = [d for d in model_cache_path.iterdir() if d.is_dir()]
        if snapshot_dirs:
            # Deterministic pick, matching `_find_gguf_in_cache`: `iterdir()` order is
            # filesystem-dependent, so the previous `[0]` could hand back a different
            # cached revision run to run. Same repository either way (never a
            # substitution), but the choice must at least be reproducible.
            return str(max(snapshot_dirs, key=lambda d: d.stat().st_mtime))
    return None


@dataclass
class _GGUFPromptCacheValue:
    cache: Any
    capacity_bytes: int
    system_prompt_parts: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    add_generation_prompt: bool = False
    prompt_text: str = ""
    prompt_tokens: tuple[int, ...] = field(default_factory=tuple)
    # The PREVIOUS generate turn's full prompt ids, for the snapshot-boundary
    # holdback only (MLX `_FED_TOKEN_IDS_META` / transformers
    # `_TRANSFORMERS_FED_IDS_META` parity). Deliberately NOT `prompt_tokens`:
    # that field is the durable-bloc prefix that
    # `_gguf_compose_cached_prompt_tokens` treats as source-of-truth and
    # CONCATENATES a suffix onto, so overwriting it per turn would compose
    # previous-prompt + new-prompt into a garbage prompt.
    fed_prompt_tokens: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class _TransformersPromptCacheValue:
    """Best-effort cache state for HuggingFace transformers KV reuse.

    `cache` is expected to be a `transformers.cache_utils.Cache` (typically `DynamicCache`).
    `prompt_tokens` tracks the token ids that have been prefetched into `cache` so the provider
    can build attention masks and compute delta lengths.
    """

    cache: Any
    prompt_tokens: tuple[int, ...] = field(default_factory=tuple)
    system_prompt_parts: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    add_generation_prompt: bool = False


_TRANSFORMERS_TENSOR_LIST_CACHE_ATTRS = (
    "key_cache",
    "value_cache",
    "conv_states",
    "recurrent_states",
    "ssm_states",
    "conv_states_q",
    "conv_states_k",
    "conv_states_v",
)
_TRANSFORMERS_JSON_CACHE_ATTRS = (
    "layer_types",
    "transformer_layers",
    "last_linear_layer",
)


class HuggingFaceProvider(BaseProvider):
    """HuggingFace provider with dual support for transformers and GGUF models"""

    @staticmethod
    def _resolve_requested_device(device: Optional[str]) -> Optional[str]:
        """Resolve the requested device from explicit arg or env override.

        Supported env var: ABSTRACTCORE_HF_DEVICE=cpu|mps|cuda|auto
        """
        if isinstance(device, str) and device.strip():
            val = device.strip().lower()
            return None if val == "auto" else val

        env_device = os.environ.get("ABSTRACTCORE_HF_DEVICE")
        if isinstance(env_device, str) and env_device.strip():
            val = env_device.strip().lower()
            if val in {"auto", "cpu", "mps", "cuda"}:
                return None if val == "auto" else val
        return None

    def __init__(self, model: str = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                 device: Optional[str] = None,
                 n_gpu_layers: Optional[int] = None,
                 structured_output_method: str = "auto",
                 **kwargs):

        # Handle legacy context_size parameter with deprecation warning
        context_size = kwargs.pop("context_size", None)
        if context_size is not None:
            import warnings
            warnings.warn(
                "The 'context_size' parameter is deprecated. Use 'max_tokens' instead. "
                "context_size will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = context_size

        # Explicit artifact selector (ADR 0009). `model_type=` is how a caller declares
        # WHICH artifact of an ambiguous handle they want: "gguf" or "transformers".
        # It is deliberately the ONLY such switch — the model handle is otherwise
        # authoritative, and nothing else may override it.
        requested_model_type = self._coerce_requested_model_type(kwargs.pop("model_type", None))

        user_provided_max_tokens = "max_tokens" in kwargs and kwargs.get("max_tokens") is not None

        super().__init__(model, **kwargs)
        self.provider = "huggingface"
        self._user_provided_max_tokens = bool(user_provided_max_tokens)

        # Register-at-first-write: HF model loads write into the HF hub cache.
        from ..utils.data_registry import ensure_core_data_homes
        ensure_core_data_homes()

        # Handle timeout parameter for local models
        self._handle_timeout_parameter(kwargs)

        # Structured output method: "auto", "native_outlines", "prompted"
        # auto: Use Outlines if available (for transformers), otherwise prompted (default)
        # native_outlines: Force Outlines (error if unavailable)
        # prompted: Always use prompted fallback (fastest for transformers, still 100% success)
        # Note: GGUF models always use llama-cpp-python native support regardless of this setting
        self.structured_output_method = structured_output_method

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Store provider-specific configuration
        self.n_gpu_layers = n_gpu_layers
        self.model_type = None  # Will be "transformers" or "gguf"
        self.device = self._resolve_requested_device(device)

        # Store transformers-specific parameters
        self.transformers_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['trust_remote_code', 'torch_dtype', 'device_map', 'load_in_8bit', 'load_in_4bit', 'attn_implementation', 'quantization_config']
        }
        # `load_in_4bit` / `load_in_8bit` were REMOVED from
        # `from_pretrained` in transformers 5.9 — `quantization_config` is the
        # only supported route — so forwarding them verbatim raises on a current
        # stack. Translate them (with their `bnb_4bit_*` / `llm_int8_*`
        # companions) into a BitsAndBytesConfig, and let an explicit
        # `quantization_config=` win. Purely additive: a caller that passes
        # neither is unaffected.
        self._transformers_quantization_request = self._build_transformers_quantization_config(kwargs)
        if self._transformers_quantization_request is not None:
            self.transformers_kwargs['quantization_config'] = self._transformers_quantization_request
        self.transformers_kwargs.pop('load_in_4bit', None)
        self.transformers_kwargs.pop('load_in_8bit', None)

        # Store device preference for custom models
        self.preferred_device = kwargs.get('device_map', 'auto')

        # Model instances
        self.tokenizer = None
        self.processor = None  # For vision models
        self.model_instance = None
        self.pipeline = None
        self.llm = None  # For GGUF models
        self._gguf_prompt_cache_lock = threading.Lock()
        self._gguf_prompt_cache_default_capacity_bytes = self._coerce_gguf_prompt_cache_capacity_bytes(
            kwargs.get("prompt_cache_capacity_bytes", None)
        )
        self._gguf_prompt_cache_pending_capacity_bytes: Optional[int] = None

        # Artifact selection (ADR 0009: a named handle is honoured, or the call fails).
        #
        # An explicit `model_type=` is the caller's own declaration and wins. Otherwise
        # the artifact class comes from the handle itself, and a handle that does not
        # name a GGUF is NEVER promoted to one.
        #
        # This site previously did the opposite: any handle with a local LM Studio Hub
        # manifest was promoted to GGUF whenever *some* GGUF could be resolved from the
        # caches. Because a Hub manifest's `baseModel` dependency routinely points at a
        # DIFFERENT repository (`Qwen/Qwen3.6-27B` -> `lmstudio-community/Qwen3.6-27B-GGUF`),
        # asking for a repo's bf16 transformers weights silently returned someone else's
        # 4-bit conversion, on llama.cpp, with no warning — and measurements taken through
        # it were attributed to the model the caller named.
        if requested_model_type is not None:
            is_gguf = requested_model_type == "gguf"
            if not is_gguf and self._is_gguf_model(model):
                raise ModelArtifactMismatchError(
                    f"model_type='transformers' was requested, but the handle '{model}' names a "
                    "GGUF artifact, which transformers cannot load.\n\n"
                    "Drop model_type= to load it as GGUF, or pass a transformers repository id "
                    "(or a local snapshot directory) instead."
                )
        else:
            is_gguf = self._is_gguf_model(model)
            if not is_gguf:
                self._reject_silent_gguf_substitution(model)

        if is_gguf:
            if not LLAMACPP_AVAILABLE:
                raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            self.model_type = "gguf"
            self._setup_device_gguf()
            self._load_gguf_model()
        else:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not installed. Install with: pip install transformers torch")
            self.model_type = "transformers"
            self._setup_device_transformers()
            self._load_transformers_model()

    # Artifact classes this provider can load. `model_type=` selects between them
    # when the handle alone is ambiguous; there is no third option and no other flag.
    _ARTIFACT_TYPES = ("transformers", "gguf")

    @staticmethod
    def _coerce_requested_model_type(value: Any) -> Optional[str]:
        """Validate an explicit `model_type=` artifact declaration."""
        if value is None:
            return None
        val = str(value).strip().lower()
        if val in HuggingFaceProvider._ARTIFACT_TYPES:
            return val
        raise InvalidRequestError(
            f"HuggingFaceProvider: model_type={value!r} is not a valid artifact selector. "
            f"Use one of {list(HuggingFaceProvider._ARTIFACT_TYPES)}, or omit it and let the "
            f"model handle decide."
        )

    def _reject_silent_gguf_substitution(self, model: str) -> None:
        """Fail loudly when this handle would have been swapped for a cached GGUF.

        ADR 0009. `model` reaching here does NOT name a GGUF (no `.gguf` path, no
        on-disk GGUF directory, no `gguf` token in the id), so the caller asked for
        this repository's transformers weights. If an LM Studio Hub manifest can
        nonetheless route the name to a GGUF, that GGUF is a *different artifact* —
        a separately quantized conversion, usually built from a different
        repository, executed by llama.cpp rather than transformers. Returning it
        under the requested name is the substitution this method exists to stop.

        Only fires when the named artifact CANNOT be loaded as named. ADR 0009 is a
        rule against *substitution*, not against coexistence: a handle whose own
        transformers weights are sitting in the cache is going to load those weights,
        so there is nothing to substitute and nothing to refuse. Checking the Hub
        manifest alone is not sufficient evidence of a substitution — `Qwen/Qwen3.6-27B`
        has both a complete bf16 snapshot and a manifest pointing at
        `lmstudio-community/Qwen3.6-27B-GGUF`, and refusing it blocked a load that
        would have been correct.

        Silent on the common path: costs one manifest `is_file()` probe for handles
        that have no Hub manifest, which is the overwhelming majority.
        """
        try:
            from ..utils.model_cache import resolve_lmstudio_hub_manifest

            manifest = resolve_lmstudio_hub_manifest(model)
            substitute = self._find_gguf_in_cache(model) if manifest is not None else None
        except ModelArtifactMismatchError:
            raise
        except Exception:
            # A broken/unreadable manifest means we cannot prove a substitution would
            # occur. Proceed as transformers — the honest reading of the handle.
            return

        if not substitute:
            return

        # The decisive question, and the only one that separates substitution from
        # coexistence: can this handle be loaded as the artifact class it names? If
        # yes, that is what happens next and the GGUF is irrelevant.
        if self._transformers_weights_present(model):
            return

        raise ModelArtifactMismatchError(
            f"Refusing to load a different model artifact than the one requested.\n"
            f"\n"
            f"  Requested : {model!r}  (HuggingFace transformers weights — NOT present locally)\n"
            f"  Found     : {substitute!r}\n"
            f"              (GGUF, reached via the LM Studio Hub manifest at {manifest})\n"
            f"\n"
            f"These are not interchangeable. A GGUF is a separately quantized conversion — "
            f"typically built from a different repository and usually 4-bit — and it runs on "
            f"llama.cpp instead of transformers. Loading it under the name {model!r} would "
            f"attribute its speed, memory profile and output quality to the model you named.\n"
            f"\n"
            f"Ask for exactly one of them:\n"
            f"  - the GGUF                 : pass the .gguf file path, or the GGUF repository "
            f"id, or model_type='gguf'\n"
            f"  - the transformers weights : pass model_type='transformers' (which reports "
            f"plainly if they are not present locally)\n"
        )

    @staticmethod
    def _transformers_weights_present(model: str) -> bool:
        """True when `model` names transformers weights that are on this disk.

        Cache-only and network-free, so it cannot turn a local decision into a Hub
        round trip. Requires a config AND at least one weight shard: a snapshot
        directory holding only a tokenizer (what a bare `tokenizer_config.json`
        download leaves behind) is not loadable and must not read as present.
        """
        try:
            from pathlib import Path

            from ..utils.model_cache import resolve_hf_snapshot_dir

            candidate = Path(str(model)).expanduser()
            snapshot = candidate if candidate.is_dir() else resolve_hf_snapshot_dir(model)
            if snapshot is None or not snapshot.is_dir():
                return False
            if not (snapshot / "config.json").is_file():
                return False
            return any(
                any(snapshot.glob(pattern))
                for pattern in ("*.safetensors", "*.bin", "*.pt", "*.msgpack")
            )
        except Exception:
            # Cannot prove presence -> fall through to the caller's refusal, which
            # names both artifacts and asks the caller to pick. Never silently load.
            return False

    def _apply_provider_thinking_kwargs(
        self,
        *,
        enabled: Optional[bool],
        level: Optional[str],
        kwargs: Dict[str, Any],
        request_shape: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], ThinkingControlHandling]:
        new_kwargs = dict(kwargs or {})
        model_type = str(getattr(self, "model_type", "") or "").strip().lower()
        # Asset-driven: the tokenizer chat-template boolean switch (`enable_thinking`-style)
        # is declared as `thinking_control.template_kwarg` in the registries.
        surfaces = self._thinking_control_surfaces()

        # Effort artifacts live in the SYSTEM region. When the caller feeds a prefilled
        # system bloc, the renderers must not reopen it, so no level artifact can be
        # applied — decline the claim so the base ladder warns (same rule as MLX).
        prefilled = new_kwargs.get("prompt_cache_prefilled_modules")
        if isinstance(prefilled, str):
            prefilled = [prefilled]
        system_prefilled = isinstance(prefilled, (list, tuple)) and any(
            str(item or "").strip().lower() == "system" for item in prefilled
        )

        if model_type == "transformers" and surfaces.template_kwarg:
            if enabled is False:
                new_kwargs["_acore_hf_transformers_enable_thinking"] = False
                return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=False)
            if enabled is True or level is not None:
                new_kwargs["_acore_hf_transformers_enable_thinking"] = True
                handled_level = False
                effort_lines = surfaces.effort_system_lines or {}
                if (
                    surfaces.effort_template_kwarg
                    and isinstance(level, str)
                    and level
                    and not system_prefilled
                ):
                    declared = self._model_reasoning_levels()
                    # Uncached lane: the REAL chat template consumes the kwarg
                    # (verified on transformers 5.8.0: extra apply_chat_template
                    # kwargs reach template.render). Cached lane renders by hand
                    # and can only inject a level declared in the asset
                    # effort_system_lines map — so in cache mode the claim is
                    # additionally gated on that map.
                    cache_key = new_kwargs.get("prompt_cache_key")
                    cached_mode = isinstance(cache_key, str) and bool(cache_key.strip())
                    level_renderable = (not cached_mode) or (level in effort_lines)
                    if (not declared or level in declared) and level_renderable:
                        new_kwargs["_acore_hf_transformers_reasoning_effort"] = level
                        handled_level = True
                return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=handled_level)

        # (helper for the transformers/vision lanes lives below: _hf_thinking_template_kwargs)
        # llama-cpp-python (GGUF) does not currently expose chat-template kwargs such as
        # `enable_thinking` / `thinking_budget`. For exact local control-plane renders we can still
        # place Qwen's no-thinking marker at the assistant generation boundary ourselves.
        #
        # TODO(upstream): Add an explicit `chat_template_kwargs` (or at least `enable_thinking`)
        # parameter to llama-cpp-python's `Llama.create_chat_completion()` and forward it into the
        # chat handler / Jinja template renderer. Once available, map our unified `thinking=...`
        # directly to `enable_thinking` instead of relying on the `<think>\n\n</think>\n\n` marker.
        # See: `docs/backlog/planned/2026-03-30_llama-cpp-python_expose_chat_template_kwargs.md`.
        if model_type == "gguf" and surfaces.assistant_prefill_disable:
            # Requests that cannot ride the local-render lane: native structured output
            # (response_format) and multimodal payloads go through llama-cpp-python's
            # create_chat_completion, whose Jinja formatter renders a trailing assistant
            # marker as a CLOSED turn — the disable marker measurably does not work
            # there (live find 2026-08-19: model thought 113-1359 chars behind an
            # "off" claim on three models). Decline every claim for those shapes so
            # the base ladder warns honestly. Text-shaped requests are served by the
            # control-plane local render, where both controls are real.
            shape = request_shape if isinstance(request_shape, dict) else {}
            lane_blocked = bool(shape.get("has_response_model")) or bool(shape.get("has_media"))
            # Lane-predicate PARITY (adversarial find V2-F2, 2026-08-20): the claim
            # must be computed with the same tests the control-plane gate applies at
            # generate time, or a claim-then-fallback path reaches
            # create_chat_completion where neither control has a transport.
            if os.environ.get("ABSTRACTCORE_GGUF_CONTROL_PLANE", "1").strip().lower() in {"0", "false", "no", "off"}:
                lane_blocked = True
            # Template-less GGUF conversions (no embedded template, llama.cpp guesses
            # e.g. "llama-2") have no control-plane render — same parity rule.
            if not self._gguf_prompt_cache_supports_local_control_plane():
                lane_blocked = True
            shape_messages = shape.get("messages")
            if isinstance(shape_messages, list) and shape_messages:
                # Same criteria as _gguf_control_plane_can_stream, applied to the raw
                # caller messages (whose roles/content _gguf_build_chat_messages copies
                # verbatim): exotic roles (tool/function) or content-parts payloads
                # fall back to create_chat_completion.
                for msg in shape_messages:
                    if not isinstance(msg, dict):
                        lane_blocked = True
                        break
                    role = str(msg.get("role") or "").strip().lower()
                    if role not in {"system", "user", "assistant"}:
                        lane_blocked = True
                        break
                    content = msg.get("content")
                    if content is not None and not isinstance(content, str):
                        lane_blocked = True
                        break
            if lane_blocked:
                return kwargs, ThinkingControlHandling()
            chat_format = (
                self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
            )
            if enabled is False:
                new_kwargs["_acore_gguf_enable_thinking"] = False
                return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=False)
            if enabled is True or level is not None:
                handled_level = False
                effort_lines = surfaces.effort_system_lines or {}
                # chatml renders the asset line by hand; the embedded-template format
                # renders the model's own template with the declared kwarg. llama-3
                # format has no effort surface — never claim there.
                level_renderable = (
                    chat_format == "llama-cpp-chat-template" and bool(surfaces.effort_template_kwarg)
                ) or (chat_format not in {"llama-3", "llama-cpp-chat-template"} and level in effort_lines)
                if (
                    isinstance(level, str)
                    and level
                    and level_renderable
                    and not system_prefilled
                ):
                    declared = self._model_reasoning_levels()
                    if not declared or level in declared:
                        new_kwargs["_acore_gguf_reasoning_effort"] = level
                        handled_level = True
                return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=handled_level)
        return kwargs, ThinkingControlHandling()

    def _hf_thinking_template_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Chat-template render kwargs for the transformers/vision lanes.

        Reads the request stashes set by `_apply_provider_thinking_kwargs` and maps
        the effort level to the asset-declared template variable. Callers pass the
        result into `apply_chat_template`; processors that reject extra kwargs get
        a warned retry without them (never a silent drop).
        """
        out: Dict[str, Any] = {}
        et = kwargs.get("_acore_hf_transformers_enable_thinking")
        if isinstance(et, bool):
            out["enable_thinking"] = et
        effort = kwargs.get("_acore_hf_transformers_reasoning_effort")
        if isinstance(effort, str) and effort:
            surfaces = self._thinking_control_surfaces()
            if surfaces.effort_template_kwarg:
                out[surfaces.effort_template_kwarg] = effort
        return out

    def unload_model(self, model_name: str) -> None:
        """
        Unload the model from memory.

        For GGUF models, calls llm.close() to free llama.cpp resources.
        For transformers models, clears model and tokenizer references.
        """
        import gc
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                # Try to properly close the Llama object (GGUF models)
                if hasattr(self.llm, 'close'):
                    self.llm.close()
                # Clear the reference
                self.llm = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                self.tokenizer = None

            if hasattr(self, 'processor') and self.processor is not None:
                self.processor = None

            if hasattr(self, 'model_instance') and self.model_instance is not None:
                self.model_instance = None

            if hasattr(self, 'pipeline') and self.pipeline is not None:
                self.pipeline = None

            # Hybrid boundary snapshots are the LARGEST tensors this provider
            # holds (deepcopied KV caches, up to the snapshot bound of them) —
            # an unload that strands them frees almost nothing (ADVERSARY
            # FINDING 2; same defect family as the 2026-08-03 leak audit).
            # The lane-routing flag must go with the model it describes, or a
            # different model loaded onto this instance would be mis-routed.
            self._ensure_transformers_snapshot_state()
            with self._transformers_snapshot_lock:
                self._transformers_snapshots.clear()
            for attr in ("_transformers_snapshot_lane_flag",
                         "_transformers_logits_to_keep_supported",
                         "_transformers_prefill_step_cached"):
                if hasattr(self, attr):
                    delattr(self, attr)

            # Force garbage collection to free memory immediately
            gc.collect()
            self._transformers_release_device_pool()
        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

    def _coerce_gguf_prompt_cache_capacity_bytes(self, value: Any) -> int:
        try:
            cap = int(value)
        except Exception:
            cap = 0
        # `0` means "auto": the cache backend may resize to fit large prompts.
        return cap if cap > 0 else 0

    def _gguf_prompt_cache_chat_format(self) -> str:
        llm = getattr(self, "llm", None)
        chat_format = str(getattr(llm, "chat_format", "") or "").strip().lower()
        if chat_format:
            return chat_format
        model_lower = str(getattr(self, "model", "") or "").lower()
        if "qwen" in model_lower or "coder" in model_lower:
            return "chatml-function-calling"
        if "llama-3" in model_lower or "llama3" in model_lower:
            return "llama-3"
        return ""

    def _gguf_architecture_message_format(self) -> str:
        cfg = getattr(self, "architecture_config", None)
        if not isinstance(cfg, dict):
            return ""
        return str(cfg.get("message_format") or "").strip().lower()

    def _gguf_prompt_cache_control_plane_chat_format(self) -> str:
        chat_format = self._gguf_prompt_cache_chat_format()
        aliases = {
            # Plain "chatml" is llama-cpp-python's byte-exact template guess
            # for ChatML GGUFs; the exact renderer (and its dedicated
            # tokenization branch) serves it identically to the
            # function-calling variant.
            "chatml": "chatml",
            "chatml-function-calling": "chatml-function-calling",
            "llama-3": "llama-3",
            "llama3": "llama-3",
        }
        if chat_format in aliases:
            return aliases[chat_format]
        if chat_format.startswith("chat_template"):
            metadata = getattr(getattr(self, "llm", None), "metadata", {})
            template = str(metadata.get("tokenizer.chat_template") or "") if isinstance(metadata, dict) else ""
            if not template:
                return ""
            if self._gguf_architecture_message_format() == "gemma_turn":
                return "llama-cpp-chat-template"
            # 0821: a GGUF whose embedded Jinja template IS ChatML (both turn
            # markers present) is served by the SAME embedded-template lane the
            # Gemma-4 branch proved live — the model's OWN template renders the
            # prompt (fidelity to its training format incl. think-block
            # handling that the plain-ChatML renderer would drop), and the
            # control plane owns both render and generate so byte-consistency
            # is by construction. Detection is by template CONTENT, never by
            # model name or llama.cpp's guessed format id (the gap: Ornith 1.0
            # GGUFs are Qwen3.5 post-trains whose name lacks "qwen" and whose
            # chat_format reports the generic "chat_template.default").
            # Renderability is PROVEN by a cached probe render, not assumed —
            # a template the formatter cannot render falls back to "keyed",
            # exactly the pre-0821 behavior (fail-safe, never a wrong cache).
            if "<|im_start|>" in template and "<|im_end|>" in template:
                if self._gguf_embedded_template_probe_renders(template):
                    return "llama-cpp-chat-template"
        return ""

    def _gguf_embedded_template_probe_renders(self, template: str) -> bool:
        """True when the embedded Jinja template renders a minimal ChatML
        conversation through the control-plane renderer (0821 guard).

        Jinja templates can refuse at render time (raise_exception branches,
        required variables); claiming the control plane for one of those would
        crash every cached turn. The probe renders once per template identity
        per provider instance and requires the output to carry the user
        content inside ChatML turn markers.
        """
        probe_cache = getattr(self, "_gguf_embedded_template_probe_cache", None)
        if probe_cache is None:
            probe_cache = {}
            self._gguf_embedded_template_probe_cache = probe_cache
        key = hashlib.sha256(template.encode("utf-8", errors="replace")).hexdigest()[:16]
        cached = probe_cache.get(key)
        if cached is not None:
            return bool(cached)
        ok = False
        try:
            rendered = self._gguf_render_llama_cpp_chat_template_prompt(
                messages=[
                    {"role": "system", "content": "PROBE-SYSTEM"},
                    {"role": "user", "content": "PROBE-USER"},
                    {"role": "assistant", "content": "PROBE-ASSISTANT"},
                    {"role": "user", "content": "PROBE-FOLLOWUP"},
                ],
                add_generation_prompt=True,
            )
            ok = self._gguf_probe_render_is_chatml_shaped(rendered)
        except Exception:
            ok = False
        probe_cache[key] = ok
        return ok

    @staticmethod
    def _gguf_probe_render_is_chatml_shaped(rendered: str) -> bool:
        """True when the probe render is genuinely ChatML-SHAPED, not merely
        marker-mentioning (adversary F2, 2026-07-19: a llama-2-wire template
        whose preamble MENTIONS the markers, or one that ChatML-wraps only the
        system turn, must not be admitted). Requirements: the last USER turn's
        content sits directly inside an <|im_start|>...<|im_end|> pair, and a
        ChatML generation prompt (<|im_start|>assistant) follows it. A false
        negative falls back to "keyed" — the safe direction.
        """
        text = str(rendered or "")
        if not text or "PROBE-USER" not in text:
            return False
        followup = text.rfind("PROBE-FOLLOWUP")
        if followup < 0:
            return False
        open_before = text.rfind("<|im_start|>", 0, followup)
        if open_before < 0:
            return False
        # The nearest structural marker before the content must be its OPEN
        # (no close between open and content = content is inside the turn).
        if text.rfind("<|im_end|>", open_before, followup) >= 0:
            return False
        close_after = text.find("<|im_end|>", followup)
        if close_after < 0:
            return False
        return text.find("<|im_start|>assistant", close_after) >= 0

    def _gguf_prompt_cache_supports_local_control_plane(self) -> bool:
        if getattr(self, "model_type", None) != "gguf":
            return False
        return bool(self._gguf_prompt_cache_control_plane_chat_format())

    def _transformers_prompt_cache_supported(self) -> bool:
        if getattr(self, "model_type", None) != "transformers":
            return False
        if not TRANSFORMERS_AVAILABLE:
            return False
        if getattr(self, "tokenizer", None) is None or getattr(self, "model_instance", None) is None:
            return False
        if not hasattr(self.model_instance, "generate"):
            return False
        # Avoid claiming prompt-cache support for vision/custom models that do not follow the
        # decoder-only chat caching semantics.
        if getattr(self, "pipeline", None) is None:
            return False
        return True

    def supports_prompt_cache(self) -> bool:
        """Return True if this provider can retain an in-process prompt cache keyed by `prompt_cache_key`."""
        model_type = getattr(self, "model_type", None)
        if model_type == "gguf":
            return True
        return self._transformers_prompt_cache_supported()

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        """Return True when this provider can treat the prompt cache as the context source-of-truth."""
        return self._transformers_prompt_cache_supported()

    def prompt_cache_artifact_extension(self) -> str:
        if getattr(self, "model_type", None) == "gguf":
            return ".npz"
        return ".safetensors"

    def prompt_cache_cache_backend(self) -> str:
        if getattr(self, "model_type", None) == "transformers":
            return "hf-transformers"
        if getattr(self, "model_type", None) == "gguf":
            return "hf-gguf"
        return "huggingface"

    def prompt_cache_artifact_format(self) -> str:
        if getattr(self, "model_type", None) == "transformers":
            return "abstractcore-transformers-prompt-cache/v1"
        if getattr(self, "model_type", None) == "gguf":
            return "abstractcore-gguf-prompt-cache/v1"
        return f"abstractcore-{self.prompt_cache_cache_backend()}-prompt-cache/v1"

    def prompt_cache_engine_fingerprint(self) -> str:
        """Pin the KV-serialization engine version (0817). transformers owns
        the `DynamicCache`/`past_key_values` layout; llama.cpp (via
        llama-cpp-python) owns the GGUF cache. A version change can alter the
        serialized layout, so a reused artifact from a different engine version
        silently injects wrong KV."""
        model_type = getattr(self, "model_type", None)
        if model_type == "gguf":
            try:
                import llama_cpp

                version = str(getattr(llama_cpp, "__version__", "") or "").strip()
            except Exception:
                version = ""
            return f"llama_cpp=={version}" if version else "llama_cpp==unknown"
        try:
            import transformers

            version = str(getattr(transformers, "__version__", "") or "").strip()
        except Exception:
            version = ""
        return f"transformers=={version}" if version else "transformers==unknown"

    def prompt_cache_tokenizer_fingerprint(self) -> str:
        """Identity of the LOADED tokenizer's text→ids mapping (0817 axis 2).

        GGUF deliberately returns "": its tokenizer travels INSIDE the model
        file, so a tokenizer refresh cannot happen without the weights file
        changing — the weights-identity axis owns that signal. transformers
        models fingerprint the loaded AutoTokenizer; "" while unloaded (the
        gates abstain on "" and re-check when the tokenizer exists).
        """
        if getattr(self, "model_type", None) == "gguf":
            return ""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return ""
        from .tokenizer_fingerprint import tokenizer_fingerprint_for

        return tokenizer_fingerprint_for(tokenizer)

    def prompt_cache_model_config_fingerprint(self) -> str:
        """Identity of the LOADED model's KV geometry (0817 axis 3).

        GGUF deliberately returns "": its config travels INSIDE the model
        file, so a geometry edit cannot happen without the weights file
        changing — the weights-identity axis owns that signal. transformers
        models fingerprint the loaded model's `config` (PretrainedConfig);
        "" while unloaded (the gates abstain on "" and re-check when the
        model exists).
        """
        if getattr(self, "model_type", None) == "gguf":
            return ""
        model_obj = getattr(self, "model_instance", None)
        config = getattr(model_obj, "config", None) if model_obj is not None else None
        if config is None:
            return ""
        from .model_config_fingerprint import model_config_fingerprint_for

        return model_config_fingerprint_for(config)

    def prompt_cache_weights_fingerprint(self) -> str:
        """Cheap identity of the LOADED weights (0817 axis 4).

        GGUF does NOT abstain here — this is exactly the axis the tokenizer
        and config axes deferred to (their state travels INSIDE the weights
        file): the loaded file's size + header-slice digest moves on any
        re-quant/re-pack. transformers models prefer the hub commit sha the
        loaded config carries (`_commit_hash`, tier 1), else the resolved
        local snapshot/model directory (tier 1/2). "" while unloaded — gates
        abstain and re-check at load time.
        """
        from .weights_fingerprint import (
            weights_fingerprint_for_dir,
            weights_fingerprint_for_file,
            weights_fingerprint_for_revision,
        )

        if getattr(self, "model_type", None) == "gguf":
            llama_obj = getattr(self, "llm", None)
            model_path = getattr(llama_obj, "model_path", None) if llama_obj is not None else None
            if not model_path:
                return ""
            return weights_fingerprint_for_file(model_path)

        model_obj = getattr(self, "model_instance", None)
        config = getattr(model_obj, "config", None) if model_obj is not None else None
        if config is not None:
            revision = weights_fingerprint_for_revision(getattr(config, "_commit_hash", None))
            if revision:
                return revision
            name_or_path = str(getattr(config, "_name_or_path", "") or "").strip()
            if name_or_path:
                from_dir = weights_fingerprint_for_dir(name_or_path)
                if from_dir:
                    return from_dir
        if model_obj is None:
            return ""
        local = _get_local_model_path(str(getattr(self, "model", "") or ""))
        if local:
            return weights_fingerprint_for_dir(local)
        return ""

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
        model_type = str(getattr(self, "model_type", "") or "").strip().lower()
        if model_type == "transformers":
            serialized = self._transformers_build_prompt_fragment(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                prefilled_modules=prefilled_modules,
            )
            if not serialized:
                return None
            architecture = str(getattr(self, "architecture", "") or "generic").strip().lower()
            template = str(getattr(getattr(self, "tokenizer", None), "chat_template", "") or "")
            if not template:
                try:
                    template = json.dumps(getattr(self, "architecture_config", {}) or {}, sort_keys=True)
                except Exception:
                    template = ""
            template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12] if template else "default"
            return PromptCacheRenderedFragment(
                serialized_prompt=str(serialized),
                serializer_version=f"hf-transformers-prompt-fragment/v1:{architecture}:{template_hash}",
                cache_backend="hf-transformers",
                artifact_format=self.prompt_cache_artifact_format(),
                meta={
                    "architecture": architecture,
                    "template_hash": template_hash,
                    "cache_implementation": "dynamic",
                    "cache_position_strategy": self._transformers_cache_position_strategy(),
                },
            )

        if model_type == "gguf":
            if not self._gguf_prompt_cache_supports_local_control_plane():
                return None
            chat_format = self._gguf_prompt_cache_control_plane_chat_format()
            chat_messages = self._gguf_build_chat_messages(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                user_message_content=prompt if isinstance(prompt, str) and prompt else None,
            )
            prompt_text, prompt_tokens = self._gguf_render_prompt_tokens(
                messages=chat_messages,
                add_generation_prompt=bool(add_generation_prompt),
            )
            serialized = json.dumps(
                {
                    "chat_format": chat_format,
                    "prompt_text": prompt_text,
                    "prompt_tokens": [int(tok) for tok in prompt_tokens],
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            token_bytes = json.dumps([int(tok) for tok in prompt_tokens], separators=(",", ":")).encode("utf-8")
            metadata = getattr(getattr(self, "llm", None), "metadata", {})
            template = str(metadata.get("tokenizer.chat_template") or "") if isinstance(metadata, dict) else ""
            serializer_version = f"hf-gguf-prompt-fragment/v1:{chat_format}"
            if chat_format == "llama-cpp-chat-template":
                template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12] if template else "missing"
                serializer_version = f"hf-gguf-prompt-fragment/v1:{chat_format}:{template_hash}"
            return PromptCacheRenderedFragment(
                serialized_prompt=serialized,
                serializer_version=serializer_version,
                cache_backend="hf-gguf",
                artifact_format=self.prompt_cache_artifact_format(),
                meta={
                    "chat_format": chat_format,
                    "exact_prompt_renderer": chat_format,
                    "prompt_tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
                },
            )

        return None

    def prompt_cache_render_bloc_text(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = False,
    ) -> Optional[str]:
        """Exact `generate()` render, for the bloc planner (see base.py).

        PARITY (2026-08-07). Before this, only the MLX lane implemented this
        hook, so only MLX got a token-exact bloc plan; both HuggingFace lanes
        returned the base default (None) and fell back to the legacy per-module
        append path with NO seam verification. Same abstraction, two very
        different guarantees — measured: MLX planned 2 blocs, transformers and
        GGUF planned none.

        Both lanes delegate to the SAME renderer their live prompt goes through
        (`_transformers_build_prompt_fragment` / `_gguf_render_prompt_text`),
        which is the only reason the planner's token-prefix guarantee holds.
        """
        model_type = str(getattr(self, "model_type", "") or "").strip().lower()

        if model_type == "transformers":
            if getattr(self, "tokenizer", None) is None:
                return None
            try:
                return self._transformers_build_prompt_fragment(
                    prompt=str(prompt or ""),
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    add_generation_prompt=bool(add_generation_prompt),
                )
            except Exception:
                return None

        if model_type == "gguf":
            # Gated on the exact renderer, deliberately. Without it the text
            # this returns would not be the text `generate()` sends, and every
            # bloc boundary derived from it would be a lie.
            if getattr(self, "llm", None) is None:
                return None
            if not self._gguf_prompt_cache_supports_local_control_plane():
                return None
            try:
                chat_messages = self._gguf_build_chat_messages(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    user_message_content=prompt if isinstance(prompt, str) and prompt else None,
                )
                return self._gguf_render_prompt_text(
                    messages=chat_messages,
                    add_generation_prompt=bool(add_generation_prompt),
                )
            except Exception:
                return None

        return None

    def prompt_cache_encode_bloc_text(self, text: str) -> Optional[List[int]]:
        """Tokenize rendered bloc text with the SAME call the live path uses.

        The base default (`tokenizer.encode`) is wrong for both lanes: the GGUF
        lane has no `self.tokenizer` at all (llama.cpp owns the vocabulary), and
        the transformers lane needs the live path's BOS policy or every boundary
        shifts by one token.
        """
        if not isinstance(text, str):
            return None
        model_type = str(getattr(self, "model_type", "") or "").strip().lower()

        if model_type == "transformers":
            if getattr(self, "tokenizer", None) is None:
                return None
            try:
                return list(self._transformers_tokenize_fragment(text, add_bos_if_empty=True))
            except Exception:
                return None

        if model_type == "gguf":
            if getattr(self, "llm", None) is None:
                return None
            try:
                return [int(t) for t in self._gguf_tokenize_rendered_prompt(text)]
            except Exception:
                return None

        return super().prompt_cache_encode_bloc_text(text)

    def get_prompt_cache_capabilities(self) -> PromptCacheCapabilities:
        if not self.supports_prompt_cache():
            return PromptCacheCapabilities()

        if getattr(self, "model_type", None) == "transformers":
            return PromptCacheCapabilities(
                supported=True,
                mode="local_control_plane",
                supports_set=True,
                supports_clear=True,
                supports_update=True,
                supports_fork=True,
                supports_prepare_modules=True,
                supports_stats=True,
                supports_save=True,
                supports_load=True,
                supports_ttl=True,
                notes=(
                    "Transformers prompt caching uses cross-call KV reuse (past_key_values / Cache).",
                    "Supports KV source-of-truth mode (delta-only prompts) via CachedSession.",
                    "cache_implementation=dynamic",
                ),
            )

        if self._gguf_prompt_cache_supports_local_control_plane():
            chat_format = self._gguf_prompt_cache_control_plane_chat_format()
            return PromptCacheCapabilities(
                supported=True,
                mode="local_control_plane",
                supports_set=True,
                supports_clear=True,
                supports_update=True,
                supports_fork=True,
                supports_prepare_modules=True,
                supports_stats=True,
                supports_save=True,
                supports_load=True,
                supports_ttl=True,
                notes=(
                    "GGUF prompt caching uses llama.cpp state snapshots plus keyed prefix reuse.",
                    f"exact_prompt_renderer={chat_format}",
                ),
            )

        chat_format = self._gguf_prompt_cache_chat_format() or "unknown"
        return PromptCacheCapabilities(
            supported=True,
            mode="keyed",
            supports_set=True,
            supports_clear=True,
            supports_update=False,
            supports_fork=False,
            supports_prepare_modules=False,
            supports_stats=True,
            supports_save=True,
            supports_load=True,
            supports_ttl=True,
            notes=(
                "GGUF prompt caching supports keyed cache selection for this model.",
                "Local control-plane parity currently requires an exact cached prompt renderer.",
                f"supported_renderers=chatml-function-calling,llama-3,gemma_turn/chat_template "
                f"current_chat_format={chat_format}",
            ),
        )

    def get_prompt_cache_stats(self) -> Dict[str, Any]:
        """Return prompt cache stats, including GGUF cache sizing (best-effort)."""
        stats = super().get_prompt_cache_stats()

        if str(getattr(self, "model_type", "") or "").strip().lower() != "gguf":
            return stats

        keys = stats.get("keys") if isinstance(stats, dict) else None
        if not isinstance(keys, list):
            return stats

        per_key: Dict[str, Any] = {}
        for key in keys:
            key_s = str(key)
            try:
                cache_value = self._prompt_cache_store.get(key_s)
            except Exception:
                continue

            state = self._gguf_prompt_cache_state(cache_value)
            if state is None:
                continue

            cache_obj = self._gguf_prompt_cache_unwrap(state)
            if cache_obj is None:
                continue

            cap_bytes = None
            try:
                cap_bytes = int(getattr(cache_obj, "capacity_bytes", None) or state.capacity_bytes)
            except Exception:
                try:
                    cap_bytes = int(state.capacity_bytes)
                except Exception:
                    cap_bytes = None

            cache_state = getattr(cache_obj, "cache_state", None)
            cache_entries: Optional[int] = None
            total_state_bytes: Optional[int] = None
            max_state_bytes: Optional[int] = None
            if hasattr(cache_state, "items"):
                total = 0
                max_b = 0
                count = 0
                try:
                    for _k, llama_state in cache_state.items():
                        count += 1
                        try:
                            size = int(getattr(llama_state, "llama_state_size", 0) or 0)
                        except Exception:
                            size = 0
                        if size > 0:
                            total += size
                            if size > max_b:
                                max_b = size
                except Exception:
                    count = 0
                    total = 0
                    max_b = 0
                cache_entries = int(count)
                total_state_bytes = int(total) if total > 0 else None
                max_state_bytes = int(max_b) if max_b > 0 else None

            per_key[key_s] = {
                "capacity_bytes": cap_bytes,
                "cache_state_entries": cache_entries,
                "cache_state_total_bytes": total_state_bytes,
                "cache_state_max_bytes": max_state_bytes,
                "prompt_tokens": int(len(state.prompt_tokens or ())),
                "prompt_text_chars": int(len(state.prompt_text or "")),
            }

        stats["gguf"] = {
            "control_plane_chat_format": (
                self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
            ),
            "keys": per_key,
        }
        return stats

    def _gguf_prompt_cache_export_state(self, cache_value: Any) -> Dict[str, Any]:
        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            return {}
        cap_val = getattr(state.cache, "capacity_bytes", state.capacity_bytes)
        try:
            cap_i = int(cap_val)
        except Exception:
            cap_i = int(state.capacity_bytes)
        return {
            "capacity_bytes": cap_i,
            "system_prompt_parts": copy.deepcopy(state.system_prompt_parts),
            "messages": copy.deepcopy(state.messages),
            "tools": copy.deepcopy(state.tools),
            "add_generation_prompt": bool(state.add_generation_prompt),
            "prompt_text": str(state.prompt_text or ""),
            "prompt_tokens": [int(tok) for tok in state.prompt_tokens],
        }

    def _gguf_prompt_cache_import_state(self, cache_obj: Any, meta: Optional[Dict[str, Any]] = None) -> _GGUFPromptCacheValue:
        cap = getattr(cache_obj, "capacity_bytes", None)
        state = _GGUFPromptCacheValue(
            cache=cache_obj,
            capacity_bytes=self._coerce_gguf_prompt_cache_capacity_bytes(cap),
        )
        payload = dict(meta or {})
        raw_parts = payload.get("system_prompt_parts")
        if isinstance(raw_parts, list):
            state.system_prompt_parts = [str(part) for part in raw_parts if isinstance(part, str) and part]
        raw_messages = payload.get("messages")
        if isinstance(raw_messages, list):
            state.messages = [copy.deepcopy(msg) for msg in raw_messages if isinstance(msg, dict)]
        raw_tools = payload.get("tools")
        if isinstance(raw_tools, list):
            state.tools = [copy.deepcopy(tool) for tool in raw_tools if isinstance(tool, dict)]
        state.add_generation_prompt = bool(payload.get("add_generation_prompt"))
        if isinstance(payload.get("prompt_text"), str):
            state.prompt_text = str(payload.get("prompt_text") or "")
        raw_tokens = payload.get("prompt_tokens")
        if isinstance(raw_tokens, list):
            toks: List[int] = []
            for tok in raw_tokens:
                try:
                    toks.append(int(tok))
                except Exception:
                    continue
            state.prompt_tokens = tuple(toks)
        if not state.prompt_tokens:
            state.prompt_tokens = self._gguf_prompt_cache_longest_prefix_tokens(cache_obj)
        return state

    def _gguf_prompt_cache_state(self, cache_value: Any) -> Optional[_GGUFPromptCacheValue]:
        if isinstance(cache_value, _GGUFPromptCacheValue):
            return cache_value
        cache_obj = self._gguf_prompt_cache_unwrap(cache_value)
        if cache_obj is None:
            return None
        return self._gguf_prompt_cache_import_state(cache_obj, None)

    def _gguf_prompt_cache_unwrap(self, cache_value: Any) -> Optional[Any]:
        if isinstance(cache_value, _GGUFPromptCacheValue):
            return cache_value.cache
        if cache_value is None:
            return None
        try:
            from llama_cpp.llama_cache import LlamaRAMCache
        except Exception:
            return None
        return cache_value if isinstance(cache_value, LlamaRAMCache) else None

    def _gguf_prompt_cache_longest_prefix_tokens(self, cache_obj: Any) -> tuple[int, ...]:
        state_map = getattr(cache_obj, "cache_state", None)
        if not hasattr(state_map, "keys"):
            return ()
        best: tuple[int, ...] = ()
        for key in state_map.keys():
            try:
                normalized = tuple(int(tok) for tok in key)
            except Exception:
                continue
            if len(normalized) > len(best):
                best = normalized
        return best

    def _gguf_clone_llama_state(self, state: Any) -> Optional[Any]:
        try:
            import numpy as np
            from llama_cpp.llama import LlamaState
        except Exception:
            return None
        try:
            return LlamaState(
                input_ids=np.asarray(getattr(state, "input_ids"), dtype=np.intc).copy(),
                scores=np.asarray(getattr(state, "scores"), dtype=np.single).copy(),
                n_tokens=int(getattr(state, "n_tokens", 0) or 0),
                llama_state=bytes(getattr(state, "llama_state", b"")),
                llama_state_size=int(getattr(state, "llama_state_size", 0) or 0),
                seed=int(getattr(state, "seed", 0) or 0),
            )
        except Exception:
            return None

    def _gguf_clone_llama_cache(self, cache_obj: Any, *, capacity_bytes: int) -> Optional[Any]:
        try:
            from llama_cpp.llama_cache import LlamaRAMCache
        except Exception:
            return None

        try:
            cap_i = int(capacity_bytes)
        except Exception:
            cap_i = 0
        if cap_i < 0:
            cap_i = 0

        # Preserve the concrete cache implementation (auto-growing vs fixed-capacity).
        cache_cls = cache_obj.__class__ if isinstance(cache_obj, LlamaRAMCache) else LlamaRAMCache
        try:
            cloned = cache_cls(capacity_bytes=int(cap_i))
        except Exception:
            cloned = LlamaRAMCache(capacity_bytes=int(cap_i))
        state_map = getattr(cache_obj, "cache_state", None)
        if not hasattr(state_map, "items"):
            return cloned
        for key, state in state_map.items():
            cloned_state = self._gguf_clone_llama_state(state)
            if cloned_state is None:
                return None
            try:
                cloned[tuple(int(tok) for tok in key)] = cloned_state
            except Exception:
                return None
        return cloned

    def _gguf_build_chat_messages(
        self,
        *,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        user_message_content: Any = None,
    ) -> List[Dict[str, Any]]:
        chat_messages: List[Dict[str, Any]] = []

        # ONE system turn, always (shared placement policy): chat templates
        # (ChatML/Qwen, Gemma, Llama-3) are trained on a single system block, so
        # a second consecutive system message is out-of-distribution and degrades
        # tool-calling (live find on Ornith-1.0-35B GGUF, 2026-07-15). The merged
        # system stays in the same stable prefix position before every turn, so
        # message-append KV reuse is unchanged; only the system-only ->
        # system+tools module boundary loses reuse (one-time re-prefill).
        base_system = system_prompt if isinstance(system_prompt, str) else None
        merged_system = (
            merge_tools_into_system(self.tool_handler, base_system, tools) if tools else base_system
        )
        if isinstance(merged_system, str) and merged_system:
            chat_messages.append({"role": "system", "content": merged_system})

        if isinstance(messages, list) and messages:
            chat_messages.extend(copy.deepcopy(messages))

        if user_message_content is not None:
            # Allow "messages-only" calls (prompt="") without appending an empty user turn.
            if isinstance(user_message_content, str):
                if user_message_content.strip():
                    chat_messages.append({"role": "user", "content": user_message_content})
            else:
                chat_messages.append({"role": "user", "content": user_message_content})

        return chat_messages

    def _gguf_prompt_cache_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _gguf_normalize_tool_call_arguments_for_template(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse tool-call ``arguments`` from JSON string to dict for the
        llama.cpp chat-template lane.

        Impedance mismatch this bridges: the OpenAI/llama-cpp wire contract
        carries ``tool_calls[].function.arguments`` as a JSON STRING (that is
        exactly what ``create_chat_completion`` emits and what AbstractCore
        returns), but embedded GGUF chat templates are written against the
        HuggingFace ``apply_chat_template`` convention, where ``arguments`` is a
        DICT — Qwen-Agent / Hermes / Ornith-family templates iterate it with
        ``arguments|items`` (or ``.items()``), and Jinja's ``items`` filter
        raises ``TypeError: Can only get item pairs from a mapping`` on a
        string. The crash surfaces only on the SECOND turn of a tool-using
        loop, once the assistant tool-call is replayed through the template
        (found live on Ornith-1.0-35B GGUF, 2026-07-15).

        This is a general fix for every dict-expecting template, not a
        per-model patch: a JSON-string ``arguments`` is parsed to its object;
        anything already a dict, or not valid JSON, or not an object, is left
        untouched (a template that genuinely wants a string still gets one).
        The control-plane renderer is unaffected — it uses AbstractCore's own
        renderers, which already accept both shapes.
        """
        if not isinstance(messages, list):
            return messages

        def _coerce(container: Any) -> None:
            fn = container.get("function") if isinstance(container.get("function"), dict) else None
            target = fn if fn is not None else container
            raw = target.get("arguments")
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except Exception:
                    return  # not JSON — leave the string as-is
                if isinstance(parsed, dict):
                    target["arguments"] = parsed

        for message in messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        _coerce(tc)
            fc = message.get("function_call")
            if isinstance(fc, dict):
                _coerce({"function": fc})
        return messages

    def _gguf_prompt_cache_tool_call_text(self, tool_call: Any) -> str:
        if not isinstance(tool_call, dict):
            return ""
        fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(fn.get("name") or tool_call.get("name") or "").strip()
        if not name:
            return ""
        raw_args = fn.get("arguments")
        if isinstance(raw_args, (dict, list)):
            try:
                args_text = json.dumps(raw_args, ensure_ascii=False)
            except Exception:
                args_text = str(raw_args)
        elif raw_args is None:
            args_text = ""
        else:
            args_text = str(raw_args)
        return f"functions.{name}:\n{args_text}"

    def _gguf_render_chatml_prompt(
        self,
        *,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        # Reasoning-effort instruction (asset surface `thinking_control.
        # effort_system_lines`): merged into the FIRST system message — the same
        # placement the model's own template uses — or emitted as the leading
        # system block when the conversation has none. Injection lives HERE, not
        # upstream in _gguf_build_chat_messages, so the embedded-template branch
        # (which renders the sentence itself from the kwarg) can never see it
        # twice (adversarial design review 2026-08-19).
        effort_line = ""
        if isinstance(reasoning_effort, str) and reasoning_effort:
            effort_lines = self._thinking_control_surfaces().effort_system_lines or {}
            candidate = effort_lines.get(reasoning_effort)
            if isinstance(candidate, str) and candidate.strip():
                effort_line = candidate.strip()
        if effort_line:
            first = messages[0] if messages and isinstance(messages[0], dict) else None
            if first is not None and str(first.get("role") or "").strip().lower() == "system":
                merged = dict(first)
                merged["content"] = (
                    f"{effort_line}\n\n{self._gguf_prompt_cache_message_text(merged.get('content'))}"
                )
                messages = [merged, *list(messages)[1:]]
            else:
                messages = [{"role": "system", "content": effort_line}, *list(messages)]

        parts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role not in {"system", "user", "assistant"}:
                continue
            parts.append(f"<|im_start|>{role}\n")
            if role in {"system", "user"}:
                parts.append(self._gguf_prompt_cache_message_text(message.get("content")))
                parts.append("<|im_end|>\n")
                continue

            content = self._gguf_prompt_cache_message_text(message.get("content"))
            if content:
                parts.append(content)
                parts.append("<|im_end|>\n")

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                for tool_call in tool_calls:
                    rendered = self._gguf_prompt_cache_tool_call_text(tool_call)
                    if rendered:
                        parts.append(rendered)
                parts.append("<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            parts.append(self._thinking_disable_prefill(enable_thinking))
        return "".join(parts)

    def _gguf_render_llama3_prompt(
        self,
        *,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        _ = enable_thinking
        role_map = {
            "system": "<|start_header_id|>system<|end_header_id|>\n\n",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        }
        sep = "<|eot_id|>"
        parts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            prefix = role_map.get(role)
            if not prefix:
                continue
            content = self._gguf_prompt_cache_message_text(message.get("content"))
            if content:
                parts.append(prefix)
                parts.append(content)
                parts.append(sep)
            else:
                parts.append(prefix)
        if add_generation_prompt:
            parts.append(role_map["assistant"])
        return "".join(parts)

    def _gguf_model_token_text(self, token_id: int) -> str:
        try:
            token_text = self.llm._model.token_get_text(int(token_id))
        except Exception:
            return ""
        if isinstance(token_text, bytes):
            return token_text.decode("utf-8", "ignore")
        return str(token_text or "")

    def _gguf_template_bos_text(self) -> str:
        try:
            return self._gguf_model_token_text(int(self.llm.token_bos()))
        except Exception:
            return ""

    def _gguf_strip_leading_template_bos(self, prompt_text: str) -> str:
        text = str(prompt_text or "")
        bos = self._gguf_template_bos_text()
        if bos and text.startswith(bos):
            return text[len(bos) :]
        return text

    def _gguf_render_llama_cpp_chat_template_prompt(
        self,
        *,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        metadata = getattr(getattr(self, "llm", None), "metadata", {})
        template = str(metadata.get("tokenizer.chat_template") or "") if isinstance(metadata, dict) else ""
        if not template:
            raise ValueError("GGUF chat-template renderer requires tokenizer.chat_template metadata.")
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        except Exception as e:
            raise ValueError("GGUF chat-template renderer requires llama-cpp-python Jinja2ChatFormatter.") from e

        bos = self._gguf_template_bos_text()
        try:
            eos = self._gguf_model_token_text(int(self.llm.token_eos()))
        except Exception:
            eos = ""
        formatter = Jinja2ChatFormatter(
            template,
            eos_token=eos or "<turn|>",
            bos_token=bos or "",
            add_generation_prompt=bool(add_generation_prompt),
        )
        render_kwargs: Dict[str, Any] = {}
        if isinstance(enable_thinking, bool):
            render_kwargs["enable_thinking"] = enable_thinking
        if isinstance(reasoning_effort, str) and reasoning_effort:
            # The model's OWN embedded template consumes the declared kwarg
            # (Jinja2ChatFormatter forwards **kwargs into template.render —
            # verified on llama-cpp-python 0.3.35), so the effort sentence,
            # its placement, and the think scaffold are byte-true by construction.
            effort_kwarg = self._thinking_control_surfaces().effort_template_kwarg
            if effort_kwarg:
                render_kwargs[effort_kwarg] = reasoning_effort
        response = formatter(
            messages=[copy.deepcopy(m) for m in messages if isinstance(m, dict)],
            tools=None,
            tool_choice=None,
            **render_kwargs,
        )
        return str(getattr(response, "prompt", "") or "")

    def _gguf_tokenize_completion_prompt(self, prompt_text: str) -> List[int]:
        if getattr(self, "llm", None) is None:
            return []
        bos_token_id = int(self.llm.token_bos())
        cls_token_id = int(self.llm._model.token_cls())
        bos_tokens: List[int] = [cls_token_id if cls_token_id != -1 else bos_token_id]

        if not self.llm._model.add_bos_token() or bos_tokens[:1] == [-1]:
            bos_tokens = []

        prefix_tokens = (
            self.llm.tokenize(
                prompt_text.encode("utf-8"),
                add_bos=False,
                special=True,
            )
            if prompt_text != ""
            else []
        )
        # For modular prompt-cache prefixes we intentionally omit the terminal EOS that
        # llama.cpp adds for string prompts. Keeping EOS in the stored prefix would make
        # the key non-extendable (`prefix + eos` is not a prefix of `prefix + delta + eos`).
        return list(bos_tokens + prefix_tokens)

    def _gguf_render_prompt_text(
        self,
        *,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """The exact prompt STRING this lane sends for these chat messages.

        Split out of `_gguf_render_prompt_tokens` so the bloc planner and the
        live generate path share ONE renderer and ONE tokenizer (below). The
        planner's guarantee — every bloc boundary is a true token prefix of the
        prompt `generate()` would build — is only worth something if both sides
        are literally the same code.
        """
        chat_format = self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
        if chat_format == "llama-3":
            return self._gguf_render_llama3_prompt(
                messages=messages,
                add_generation_prompt=bool(add_generation_prompt),
                enable_thinking=enable_thinking,
            )
        if chat_format == "llama-cpp-chat-template":
            return self._gguf_render_llama_cpp_chat_template_prompt(
                messages=messages,
                add_generation_prompt=bool(add_generation_prompt),
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        return self._gguf_render_chatml_prompt(
            messages=messages,
            add_generation_prompt=bool(add_generation_prompt),
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )

    def _gguf_tokenize_rendered_prompt(self, prompt_text: str) -> tuple[int, ...]:
        """Tokenize a rendered prompt exactly as the live path tokenizes it.

        BOS handling is per-chat-format and is NOT incidental: getting it wrong
        shifts every position by one and silently invalidates every cached
        prefix. Kept in one place for that reason.
        """
        chat_format = self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
        if chat_format == "llama-cpp-chat-template":
            return tuple(
                int(tok)
                for tok in self.llm.tokenize(
                    prompt_text.encode("utf-8"),
                    add_bos=False,
                    special=True,
                )
            )
        if chat_format == "chatml":
            return tuple(
                int(tok)
                for tok in self.llm.tokenize(
                    prompt_text.encode("utf-8"),
                    add_bos=True,
                    special=True,
                )
            )
        return tuple(int(tok) for tok in self._gguf_tokenize_completion_prompt(prompt_text))

    def _gguf_render_prompt_tokens(
        self,
        *,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> tuple[str, tuple[int, ...]]:
        prompt_text = self._gguf_render_prompt_text(
            messages=messages,
            add_generation_prompt=bool(add_generation_prompt),
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        return prompt_text, self._gguf_tokenize_rendered_prompt(prompt_text)

    def _gguf_tokenize_prompt_suffix(self, prompt_text: str) -> tuple[int, ...]:
        if getattr(self, "llm", None) is None or not prompt_text:
            return ()
        suffix_text = self._gguf_strip_leading_template_bos(str(prompt_text or ""))
        try:
            return tuple(
                int(tok)
                for tok in self.llm.tokenize(
                    suffix_text.encode("utf-8"),
                    add_bos=False,
                    special=True,
                )
            )
        except Exception:
            return ()

    def _gguf_compose_cached_prompt_tokens(
        self,
        *,
        cache_state: Optional[_GGUFPromptCacheValue],
        live_prompt_text: str,
        live_prompt_tokens: tuple[int, ...],
    ) -> tuple[str, tuple[int, ...], Dict[str, Any]]:
        """Return the prompt tokens to evaluate for a cache-bound GGUF request.

        Durable memory-bloc requests pass only the live suffix at request time. The loaded
        prompt cache must therefore be treated as the prefix source-of-truth rather than as a
        best-effort cache object attached to a suffix-only prompt.
        """
        meta: Dict[str, Any] = {
            "prompt_cache_prefix_source": "live_prompt",
            "prompt_cache_composed": False,
            "prompt_cache_prefix_token_count": 0,
            "prompt_cache_live_token_count": int(len(live_prompt_tokens)),
            "prompt_cache_prompt_token_count": int(len(live_prompt_tokens)),
        }
        if cache_state is None:
            return live_prompt_text, live_prompt_tokens, meta

        prefix_tokens = tuple(int(tok) for tok in (cache_state.prompt_tokens or ()))
        if not prefix_tokens:
            return live_prompt_text, live_prompt_tokens, meta

        meta["prompt_cache_prefix_token_count"] = int(len(prefix_tokens))
        if live_prompt_tokens[: len(prefix_tokens)] == prefix_tokens:
            meta["prompt_cache_prefix_source"] = "live_prompt"
            return live_prompt_text, live_prompt_tokens, meta

        live_suffix_text = self._gguf_strip_leading_template_bos(live_prompt_text)
        suffix_tokens = self._gguf_tokenize_prompt_suffix(live_suffix_text)
        if not suffix_tokens and live_suffix_text:
            meta["prompt_cache_error"] = "failed_to_tokenize_live_suffix"
            return live_prompt_text, live_prompt_tokens, meta

        composed_prompt_text = f"{str(cache_state.prompt_text or '')}{str(live_suffix_text or '')}"
        composed_tokens = tuple(int(tok) for tok in (prefix_tokens + suffix_tokens))
        meta.update(
            {
                "prompt_cache_prefix_source": "loaded_cache",
                "prompt_cache_composed": True,
                "prompt_cache_suffix_token_count": int(len(suffix_tokens)),
                "prompt_cache_prompt_token_count": int(len(composed_tokens)),
            }
        )
        return composed_prompt_text, composed_tokens, meta

    @staticmethod
    def _gguf_state_held_tokens(key_tokens: tuple[int, ...], state: Any) -> tuple[int, ...]:
        """The tokens a saved llama state actually HOLDS (reader ground truth).

        The map KEY is an index hint, never the truth: fallback-lane writers
        (llama.cpp's own cache save after ``create_chat_completion``) save
        states keyed by prompt+completion whose last sampled token was never
        eval'd, so the state holds ``len(key) - 1`` tokens. A reader that
        trusts the key length skips eval'ing that token and serves a KV with
        one mid-prompt token missing — every later position shifted, wrong
        output, zero errors (the silently-wrong-cache class; adversary F1,
        2026-07-19, reproduced). The state's own ``n_tokens``/``input_ids``
        are the truth; a state whose held tokens disagree with its key prefix
        is foreign/corrupt and is refused entirely.
        """
        n_tokens = getattr(state, "n_tokens", None)
        if not isinstance(n_tokens, int) or n_tokens < 0:
            return tuple(key_tokens)
        n_tokens = min(n_tokens, len(key_tokens))
        input_ids = getattr(state, "input_ids", None)
        if input_ids is not None:
            try:
                raw = input_ids[:n_tokens]
                held = tuple(int(tok) for tok in (raw.tolist() if hasattr(raw, "tolist") else list(raw)))
                if held != tuple(key_tokens[:n_tokens]):
                    return ()
                return held
            except Exception:
                pass
        return tuple(key_tokens[:n_tokens])

    def _gguf_prompt_cache_prefix_state(self, cache_obj: Any, prompt_tokens: tuple[int, ...]) -> tuple[int, Optional[Any]]:
        state_map = getattr(cache_obj, "cache_state", None)
        if not hasattr(state_map, "items") or not prompt_tokens:
            return 0, None

        llm = getattr(self, "llm", None)
        longest_prefix_fn = getattr(llm, "longest_token_prefix", None)
        if not callable(longest_prefix_fn):
            longest_prefix_fn = getattr(getattr(llm, "__class__", None), "longest_token_prefix", None)
        if not callable(longest_prefix_fn):
            try:
                from llama_cpp.llama import Llama as _Llama  # type: ignore

                longest_prefix_fn = getattr(_Llama, "longest_token_prefix", None)
            except Exception:
                longest_prefix_fn = None

        best_len = 0
        best_state = None
        for key, state in state_map.items():
            try:
                normalized = tuple(int(tok) for tok in key)
            except Exception:
                continue
            # Reuse exactly what the STATE holds, never what the key claims
            # (see _gguf_state_held_tokens — the F1 off-by-one class).
            held = self._gguf_state_held_tokens(normalized, state)
            if not held:
                continue
            try:
                prefix_len = int(longest_prefix_fn(held, prompt_tokens)) if callable(longest_prefix_fn) else 0
            except Exception:
                prefix_len = 0
            if prefix_len != len(held):
                continue
            if len(held) > best_len:
                best_len = len(held)
                best_state = state
        return best_len, best_state

    def _gguf_live_context_prefix_len(self, prompt_tokens: tuple[int, ...]) -> int:
        """Tokens of `prompt_tokens` already RESIDENT in llama.cpp's context.

        llama.cpp keeps the previous call's KV in the context, and that resident
        prefix is the cheapest reuse available anywhere on this lane: `Llama.eval`
        drops everything past `n_tokens` by itself, so reusing it costs one
        in-place `kv_cache_seq_rm` and no copy at all — against `load_state`, which
        restores a multi-GB snapshot.

        Mirrors `Llama.generate`'s own scan, including its `tokens[:-1]` bound: at
        least one prompt token must be evaluated or there are no logits to sample
        from.
        """
        llm = getattr(self, "llm", None)
        if llm is None or not prompt_tokens:
            return 0
        try:
            n_tokens = int(getattr(llm, "n_tokens", 0) or 0)
        except Exception:
            return 0
        if n_tokens <= 0:
            return 0
        input_ids = getattr(llm, "_input_ids", None)
        if input_ids is None:
            return 0
        try:
            resident = list(input_ids[:n_tokens])
        except Exception:
            return 0
        shared = 0
        for a, b in zip(resident, prompt_tokens[:-1]):
            if int(a) != int(b):
                break
            shared += 1
        return shared

    @staticmethod
    def _gguf_token_lcp(a: Sequence[int], b: Sequence[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and int(a[i]) == int(b[i]):
            i += 1
        return i

    def _gguf_generation_prompt_boundary(
        self,
        *,
        messages: List[Dict[str, Any]],
        prompt_tokens: tuple[int, ...],
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Optional[int]:
        """Token POSITION where this call's generation-prompt scaffolding begins.

        That scaffolding is PER-CALL VOLATILE by construction: it asks the model
        to speak now, and the next turn's transcript replaces it with the
        assistant turn that actually happened. A snapshot boundary containing it
        is a prefix of nothing — which matters only where a boundary is the sole
        reusable artifact (Gated-DeltaNet / recurrent layers, where llama.cpp
        refuses a partial `kv_cache_seq_rm` so the tail cannot be dropped after
        the fact).

        DERIVED FROM THE RENDERER, never from a literal list: the position is
        obtained by re-rendering THIS call's messages through the SAME
        `_gguf_render_prompt_tokens` with `add_generation_prompt=False` and
        taking the token-level LCP. A hardcoded tail list has to be maintained
        per template family and was previously inert for 8 of 10 of them; this
        asks the template itself, so a new family works with no code change.

        A POSITION, deliberately, not a count: if the seam merges into one token,
        `len(full) - len(head)` and the true divergence position differ by one and
        a count would put the boundary one token INSIDE the scaffolding. The LCP
        is the divergence position by definition, merge or no merge — and it is a
        prefix of `prompt_tokens` by construction, so a template that does more
        than append can only cost reuse, never correctness.

        Returns None when it cannot be computed (no renderer, render failure,
        empty head, or a head that shares nothing): no holdback is applied, which
        is the pre-existing behaviour and never a guess.
        """
        if not prompt_tokens:
            return None
        try:
            _head_text, head_ids = self._gguf_render_prompt_tokens(
                messages=messages,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        except Exception:
            return None
        if not head_ids:
            return None
        shared = self._gguf_token_lcp(head_ids, prompt_tokens)
        if shared <= 0 or shared >= len(prompt_tokens):
            # `shared >= len(prompt_tokens)` means the renderer emitted no
            # generation prompt at all — nothing volatile to hold back.
            return None
        return int(shared)

    def _gguf_snapshot_boundary(
        self,
        prompt_tokens: tuple[int, ...],
        *,
        prefix_len: int,
        prev_prompt_tokens: Sequence[int] = (),
        generation_boundary: Optional[int] = None,
    ) -> int:
        """Where to take this turn's state snapshot (ported 1:1 from the MLX
        `_hybrid_snapshot_feed` / transformers `_transformers_snapshot_feed`
        lattice).

        - `boundary_end = len - 1` — at least one prompt token must always be
          left to feed, so the snapshot can never produce a zero-token
          evaluation (see `_gguf_prefill_prompt_cache`).
        - previous-prompt LCP holdback: when the previous turn's prompt is NOT a
          prefix of this one its tail was rewritten; whatever the two prompts
          share is stable transcript and the first divergence is where this
          turn's ephemeral tail begins.
        - renderer-derived generation-scaffolding holdback on a recordless key
          (turn 1), where the LCP has nothing to compare against.
        - `prefix_len` is a FLOOR: never regress below a boundary already
          restored from.
        """
        boundary_end = max(len(prompt_tokens) - 1, 0)
        stable_end = boundary_end
        if prev_prompt_tokens:
            shared = self._gguf_token_lcp(prev_prompt_tokens, prompt_tokens)
            if shared < len(prev_prompt_tokens):
                stable_end = min(shared, boundary_end)
        elif isinstance(generation_boundary, int) and generation_boundary > 0:
            stable_end = min(stable_end, int(generation_boundary))
        return max(int(prefix_len), int(stable_end))

    def _gguf_snapshot_bound(self) -> int:
        """Hard cap on states kept in ONE key's llama cache (LRU beyond it).

        A `LlamaState` is the whole context serialized — measured 53.7 MB for a
        4B hybrid at n_ctx=512 and multi-GB at benchmark context sizes — so
        keeping one per turn of a growing loop is a leak with a nice name.
        `LlamaRAMCache` already evicts on a byte budget, but the auto-growing
        variant raises that budget to fit the largest state, so the byte bound
        alone is not a bound on COUNT. Default 2 (the boundary in use plus one
        predecessor, so an A/B alternation still hits), env-overridable via
        `ABSTRACTCORE_GGUF_SNAPSHOT_BOUND`."""
        bound = 2
        try:
            raw = os.environ.get("ABSTRACTCORE_GGUF_SNAPSHOT_BOUND")
            if raw is not None and str(raw).strip():
                bound = max(1, int(str(raw).strip()))
        except Exception:
            bound = 2
        return bound

    def _gguf_prune_snapshots(
        self,
        cache_obj: Any,
        keep_key: tuple[int, ...],
        *,
        protect: Sequence[int] = (),
    ) -> None:
        """Keep one boundary per key (the growing one supersedes its
        predecessor) and at most `_gguf_snapshot_bound()` states overall.

        A stored key that is a STRICT PREFIX of `keep_key` is dominated by it:
        every restore `keep_key` can serve, the shorter one can serve too, and
        worse. `protect` is the durable-bloc prefix the key was built from
        (`_GGUFPromptCacheValue.prompt_tokens`), which `prompt_cache_save`
        requires to exist verbatim — never evicted here."""
        state_map = getattr(cache_obj, "cache_state", None)
        if not hasattr(state_map, "items") or not hasattr(state_map, "pop"):
            return
        keep = tuple(int(t) for t in keep_key)
        guarded = {keep}
        if protect:
            guarded.add(tuple(int(t) for t in protect))
        try:
            keys = [tuple(int(t) for t in k) for k in list(state_map.keys())]
        except Exception:
            return
        for k in keys:
            if k in guarded:
                continue
            if len(k) < len(keep) and keep[: len(k)] == k:
                try:
                    state_map.pop(k, None)
                except Exception:
                    pass
        bound = self._gguf_snapshot_bound()
        try:
            while len(state_map) > bound:
                dropped = False
                for k in list(state_map.keys()):
                    if tuple(int(t) for t in k) in guarded:
                        continue
                    state_map.pop(k, None)  # OrderedDict order == LRU (oldest first)
                    dropped = True
                    break
                if not dropped:
                    break
        except Exception:
            pass

    def _gguf_reuse_live_context(self, prefix_len: int) -> bool:
        """Keep `prefix_len` resident tokens and drop the rest, in place.

        `Llama.eval` already begins with `kv_cache_seq_rm(-1, n_tokens, -1)`, so
        setting `n_tokens` is sufficient for the append itself. The explicit probe
        mirrors `Llama.generate`, which checks the return value before committing:
        a backend without partial KV removal must fall back to a full reset rather
        than silently evaluate against stale KV.
        """
        llm = getattr(self, "llm", None)
        if llm is None or prefix_len <= 0:
            return False
        ctx = getattr(llm, "_ctx", None)
        rm = getattr(ctx, "kv_cache_seq_rm", None)
        if not callable(rm):
            return False
        try:
            if not rm(-1, int(prefix_len), -1):
                return False
            llm.n_tokens = int(prefix_len)
        except Exception:
            return False
        return True

    def _gguf_prefill_prompt_cache(
        self,
        cache_obj: Any,
        prompt_tokens: tuple[int, ...],
        *,
        save_state: bool = True,
        save_state_on_live_reuse: bool = True,
        set_cache: bool = True,
        snapshot_at_boundary: bool = False,
        prev_prompt_tokens: Sequence[int] = (),
        generation_boundary: Optional[int] = None,
        protect_snapshot_key: Sequence[int] = (),
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Bring llama.cpp's context to `prompt_tokens`, reusing whatever is cheapest.

        LIVE CONTEXT FIRST (2026-08-03). This method used to open with an
        unconditional `llm.reset()`, which erased `n_tokens`/`_input_ids` — exactly
        the state llama.cpp's own prefix reuse reads. The consequence was that
        passing a `prompt_cache_key` made the GGUF lane DRAMATICALLY SLOWER than
        passing none: measured on `Qwen3-4B-Instruct-2507-GGUF`, same model, same
        process, same growing prompts, the only difference being the key —

            turn 2  with key: 10,016 tokens prefilled / 14.97 s
            turn 2  no key:       24 tokens prefilled /  0.31 s

        — a 48x regression, because with no key the engine's `Llama.generate`
        trimmed its resident context to the 9,993-token shared prefix and evaluated
        only the 24 grown tokens. `load_state` fired ZERO times across the whole
        run, while `save_state` was paid on EVERY call at 1.41 GB a snapshot: the
        machinery that replaced the engine's reuse delivered none of its own.

        The policy below is llama.cpp's, not a new one — `Llama._create_completion`
        loads a snapshot only `if cache_prefix_len > eval_prefix_len`, i.e. only
        when the stored state beats what is already resident. Snapshots become the
        fallback for a genuinely cold or divergent turn instead of the default.

        TWO INVARIANTS THIS METHOD NOW OWNS (2026-08-05):

        1. **At least one prompt token is ALWAYS evaluated.** `llama.cpp` writes
           no logits when nothing is decoded (`Llama.eval`'s explicit `else: pass`
           with `logits_all=False`), and `LlamaState` restores KV / `input_ids` /
           `n_tokens` but NOT the context's output-logits buffer. A restore that
           covers the whole prompt therefore left `remaining == []`, no forward
           pass ran, and the sampler read the PREVIOUS call's last decoded
           position: an identical resend replayed the tail of the previous answer
           (measured to token-id precision: first token 304 warm vs 785 cold on
           `unsloth/Qwen3-4B-Instruct-2507-GGUF`), or re-sampled EOS and raised
           `EmptyCompletionError` when the previous call had ended on EOS. The
           sibling live-context path already carried this bound and documents it
           (`_gguf_live_context_prefix_len`, `prompt_tokens[:-1]`); it is
           llama.cpp's own bound in `Llama.generate`.

        2. **`snapshot_at_boundary=True` stores the state BEFORE this turn's
           volatile tail**, not at the full prompt. Storing at the full prompt is
           why every grown turn re-prefilled everything on Gated-DeltaNet models:
           a chat render puts `<|im_end|>\\n<|im_start|>assistant\\n` AFTER the
           user content, so turn *i*'s full prompt stops being a prefix of turn
           *i+1*'s, the forward-only restore is refused, and — because llama.cpp
           REFUSES a partial `kv_cache_seq_rm` on recurrent state — there is no
           trim to fall back on either. Dense models hid it (live trimming works
           there, and their numbers are unchanged by this). Same defect, same
           repair, as the MLX and transformers lanes.
        """
        llm = getattr(self, "llm", None)
        if llm is None:
            return False
        if not prompt_tokens:
            # Nothing to evaluate means nothing to sample from — invariant 1.
            return False

        def _note(**fields: Any) -> None:
            if telemetry is not None:
                telemetry.update(fields)

        n_prompt = len(prompt_tokens)
        feed_floor = n_prompt - 1  # max reusable prefix; ≥1 token must be fed

        live_prefix = self._gguf_live_context_prefix_len(prompt_tokens)  # already ≤ feed_floor
        state_prefix_len, prefix_state = self._gguf_prompt_cache_prefix_state(cache_obj, prompt_tokens)

        reused_live = False
        if live_prefix > 0 and live_prefix >= state_prefix_len:
            reused_live = self._gguf_reuse_live_context(live_prefix)

        restored_key_len: Optional[int] = None
        if reused_live:
            prefix_len = live_prefix
        else:
            try:
                llm.reset()
            except Exception:
                return False
            prefix_len = 0
            if prefix_state is not None and state_prefix_len > 0:
                try:
                    llm.load_state(prefix_state)
                    prefix_len = state_prefix_len
                    restored_key_len = state_prefix_len
                except Exception:
                    try:
                        llm.reset()
                    except Exception:
                        pass
                    prefix_len = 0
                    restored_key_len = None
                if prefix_len > feed_floor:
                    # ZERO-FEED GUARD (invariant 1). A state covering the WHOLE
                    # prompt — what every pre-2026-08-05 turn stored, and what a
                    # durable bloc still stores — would leave nothing to feed.
                    # Roll the restored context back one token so the final prompt
                    # token is re-evaluated and the sampled position has logits
                    # produced by THIS prompt. `_gguf_reuse_live_context` is the
                    # CHECKED rollback (it verifies `kv_cache_seq_rm` rather than
                    # assuming it): recurrent architectures refuse it, and there
                    # the only honest answer is a full re-prefill.
                    if feed_floor <= 0 or not self._gguf_reuse_live_context(feed_floor):
                        try:
                            llm.reset()
                        except Exception:
                            return False
                        prefix_len = 0
                        restored_key_len = None
                    else:
                        prefix_len = feed_floor
                        restored_key_len = None  # live boundary ≠ the stored key

        # Where the snapshot goes. Legacy/durable callers keep the full-prompt
        # boundary they depend on (`prompt_cache_save` looks the key up verbatim);
        # only the generate lane opts into the volatile-tail holdback.
        if snapshot_at_boundary:
            stable_end = self._gguf_snapshot_boundary(
                prompt_tokens,
                prefix_len=prefix_len,
                prev_prompt_tokens=prev_prompt_tokens,
                generation_boundary=generation_boundary,
            )
        else:
            stable_end = n_prompt

        head = list(prompt_tokens[prefix_len:stable_end])
        tail = list(prompt_tokens[stable_end:])
        try:
            if head:
                llm.eval(head)
            # Snapshot the clean boundary BEFORE the volatile tail is evaluated
            # and before generation mutates the context.
            #
            # `save_state_on_live_reuse=False` is the OPPORTUNISTIC case (the
            # generate lane): when the resident context already carried this turn,
            # it now holds this prompt plus the reply, so the next turn of the same
            # session gets a LONGER live prefix than any snapshot could offer, and
            # `save_state` — measured at 1.41 GB per call — buys nothing. Callers
            # for which persisting IS the point (`prompt_cache_update`, building a
            # durable bloc) keep the default and always snapshot.
            boundary_key = tuple(prompt_tokens[:stable_end])
            if (
                save_state
                and cache_obj is not None  # keyless control-plane rides with no store:
                # a None deref HERE lands after the full prefill AND a ~GB save_state
                # were already paid, and the except below turns it into a whole-request
                # failure (adversarial design review 2026-08-19)
                and boundary_key
                and (save_state_on_live_reuse or not reused_live)
                and restored_key_len != stable_end  # already stored at this boundary
            ):
                saved_state = llm.save_state()
                cloned_state = self._gguf_clone_llama_state(saved_state)
                cache_obj[boundary_key] = cloned_state if cloned_state is not None else saved_state
                if snapshot_at_boundary:
                    self._gguf_prune_snapshots(cache_obj, boundary_key, protect=protect_snapshot_key)
            if tail:
                llm.eval(tail)
            if set_cache and hasattr(llm, "set_cache"):
                llm.set_cache(cache_obj)
        except Exception:
            return False

        fed = len(head) + len(tail)
        if reused_live:
            outcome = "hit_extend"
        elif prefix_len > 0:
            outcome = "hit_restore"
        else:
            outcome = "cold" if state_prefix_len <= 0 else "rebuilt"
        _note(
            backend="gguf",
            outcome=outcome,
            cached_tokens=int(prefix_len),
            fed_tokens=int(fed),
            prompt_tokens=int(n_prompt),
            snapshot_boundary=int(stable_end),
        )
        return True

    def _transformers_prompt_cache_state(self, cache_value: Any) -> Optional[_TransformersPromptCacheValue]:
        return cache_value if isinstance(cache_value, _TransformersPromptCacheValue) else None

    def _transformers_cache_device(self) -> Optional["torch.device"]:
        try:
            import torch  # type: ignore
        except Exception:
            return None

        model = getattr(self, "model_instance", None)
        if model is not None:
            try:
                param = next(model.parameters(), None)
                if param is not None:
                    return param.device
            except Exception:
                pass
        dev = str(getattr(self, "device", "") or "").strip().lower()
        if dev in {"cuda", "mps", "cpu"}:
            try:
                return torch.device(dev)
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")

    def _transformers_cache_device_str(self) -> str:
        dev = self._transformers_cache_device()
        if dev is None:
            return "cpu"
        s = str(dev)
        if s.startswith("cuda"):
            return "cuda"
        if s.startswith("mps"):
            return "mps"
        return "cpu"

    def _transformers_empty_native_cache(self) -> Any:
        """Return an empty cache object matching the loaded transformers model, when known."""
        model = getattr(self, "model_instance", None)
        config = getattr(model, "config", None)

        # Some modern architectures ship custom cache classes that must be constructed from
        # their text config. Let the model create the cache on first prefill when we cannot
        # confidently instantiate the right class.
        model_type = str(getattr(config, "model_type", "") or "").strip().lower()
        cache_model_type = model_type[:-5] if model_type.endswith("_text") else model_type
        custom_cache_classes = {
            "qwen3_5": "Qwen3_5DynamicCache",
            "qwen3_5_moe": "Qwen3_5MoeDynamicCache",
            "qwen3_6": "Qwen3_6DynamicCache",
            "qwen3_next": "Qwen3NextDynamicCache",
            "jamba": "HybridMambaAttentionDynamicCache",
            "zamba": "ZambaHybridDynamicCache",
            "zamba2": "Zamba2HybridDynamicCache",
            "mamba": "MambaCache",
            "mamba2": "Mamba2Cache",
            "falcon_mamba": "FalconMambaCache",
        }
        if cache_model_type in custom_cache_classes:
            try:
                module = importlib.import_module(f"transformers.models.{cache_model_type}.modeling_{cache_model_type}")
                cache_cls = getattr(module, custom_cache_classes[cache_model_type], None)
                text_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
                if cache_cls is not None:
                    return self._transformers_construct_cache(cache_cls, text_config)
            except Exception:
                return None

        try:
            from transformers.cache_utils import DynamicCache  # type: ignore
        except Exception:
            return None
        try:
            return DynamicCache(config=config)
        except Exception:
            return DynamicCache()

    def _transformers_construct_cache(self, cache_cls: Any, config: Any) -> Any:
        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore[assignment]

        device = self._transformers_cache_device()
        dtype = None
        model = getattr(self, "model_instance", None)
        if model is not None:
            try:
                param = next(model.parameters(), None)
                dtype = getattr(param, "dtype", None)
            except Exception:
                dtype = None
        if dtype is None and torch is not None:
            dtype = getattr(torch, "float16", None)

        trials = [
            ((), {"config": config}),
            ((config,), {}),
            ((), {"config": config, "batch_size": 1, "dtype": dtype, "device": device}),
            ((config,), {"batch_size": 1, "dtype": dtype, "device": device}),
            ((), {"config": config, "max_batch_size": 1, "dtype": dtype, "device": device}),
            ((config,), {"max_batch_size": 1, "dtype": dtype, "device": device}),
            ((config, 1), {"dtype": dtype, "device": device}),
        ]
        for args, kwargs in trials:
            cleaned = {k: v for k, v in kwargs.items() if v is not None}
            try:
                return cache_cls(*args, **cleaned)
            except TypeError:
                continue
        return cache_cls()

    def _transformers_cache_class_meta(self, cache: Any) -> Dict[str, str]:
        if cache is None:
            return {"cache_class": "", "cache_module": ""}
        cls = cache.__class__
        return {
            "cache_class": str(getattr(cls, "__name__", "") or ""),
            "cache_module": str(getattr(cls, "__module__", "") or ""),
        }

    def _transformers_cache_position_strategy(self) -> str:
        cfg = getattr(self, "architecture_config", None)
        if isinstance(cfg, dict):
            strategy = str(cfg.get("prompt_cache_position_strategy") or "").strip().lower()
            if strategy:
                return strategy
        return "cache_seq_length"

    @staticmethod
    def _decode_transformers_cache_json_attrs(meta: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw = json.loads(str(meta.get("cache_json_attrs") or "{}"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _transformers_dynamic_layer_lengths(self, cache: Any, prompt_len: int) -> List[int]:
        lengths: List[int] = []
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, list):
            return lengths
        for layer in layers:
            length = 0
            try:
                if hasattr(layer, "cumulative_length"):
                    length = int(getattr(layer, "cumulative_length") or 0)
            except Exception:
                length = 0
            if length <= 0:
                try:
                    length = int(layer.get_seq_length())
                except Exception:
                    length = 0
            if length <= 0 and bool(getattr(layer, "is_initialized", False)):
                length = int(prompt_len)
            lengths.append(max(0, int(length)))
        return lengths

    def _restore_transformers_dynamic_layer_lengths(
        self,
        cache: Any,
        *,
        prompt_len: int,
        layer_lengths: Optional[List[Any]] = None,
    ) -> None:
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, list):
            return
        for idx, layer in enumerate(layers):
            if not hasattr(layer, "cumulative_length"):
                continue
            if not bool(getattr(layer, "is_initialized", False)):
                continue
            length = 0
            if isinstance(layer_lengths, list) and idx < len(layer_lengths):
                try:
                    length = int(layer_lengths[idx] or 0)
                except Exception:
                    length = 0
            if length <= 0:
                length = int(prompt_len)
            if length > 0:
                try:
                    setattr(layer, "cumulative_length", int(length))
                except Exception:
                    pass

    def _transformers_instantiate_cache_from_meta(self, meta: Dict[str, Any]) -> Any:
        module_name = str(meta.get("cache_module") or "").strip()
        class_name = str(meta.get("cache_class") or "").strip()
        if module_name and class_name and module_name.startswith("transformers."):
            try:
                module = importlib.import_module(module_name)
                cache_cls = getattr(module, class_name, None)
                if cache_cls is not None:
                    model = getattr(self, "model_instance", None)
                    config = getattr(model, "config", None)
                    text_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
                    try:
                        return self._transformers_construct_cache(cache_cls, text_config)
                    except Exception:
                        pass
            except Exception:
                pass
        return self._transformers_empty_native_cache()

    def _transformers_cache_has_serializable_tensor_state(self, cache: Any) -> bool:
        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore[assignment]
        for attr in _TRANSFORMERS_TENSOR_LIST_CACHE_ATTRS:
            value = getattr(cache, attr, None)
            if isinstance(value, list):
                return True
            if torch is not None and isinstance(value, torch.Tensor):
                return True
        return False

    def _transformers_clone_cache(self, cache: Any) -> Any:
        if cache is None:
            return None
        try:
            return copy.deepcopy(cache)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Snapshot/restore lane for UNTRIMMABLE transformers architectures
    # (Gated-DeltaNet / linear-attention hybrids: Qwen3.5/3.6, Ornith, …).
    #
    # `Cache.crop` is an explicit no-op on linear-attention layers, so
    # `_transformers_crop_cache` REFUSES these architectures and every warm
    # full-context call previously rebuilt fresh — correct output, zero
    # savings (measured x0.96 vs no-cache at 10k). The MLX provider solved
    # the identical problem with a snapshot-before-decode policy, ported
    # here 1:1: keep ONE deepcopy per key taken at a clean boundary BEFORE
    # generation mutates the cache; restore FORWARD-ONLY when the stored
    # boundary ids are a TRUE PREFIX of the new prompt; feed only the
    # suffix; never roll anything back. Bounded (leak audit 2026-08-03):
    # one snapshot per key, hard LRU bound overall, dropped on store
    # eviction (`_prompt_cache_store_evicted`) and `prompt_cache_clear`.
    # ------------------------------------------------------------------

    _TRANSFORMERS_FED_IDS_META = "fed_token_ids"

    def _ensure_transformers_snapshot_state(self) -> None:
        """Lazily materialize the snapshot store/lock (instances are sometimes
        built via `__new__` in unit tests that exercise pure cache logic)."""
        if not hasattr(self, "_transformers_snapshot_lock"):
            self._transformers_snapshot_lock = threading.RLock()
        if not hasattr(self, "_transformers_snapshots"):
            self._transformers_snapshots = {}

    def _get_transformers_snapshot(self, key: str) -> Optional[Dict[str, Any]]:
        self._ensure_transformers_snapshot_state()
        with self._transformers_snapshot_lock:
            snap = self._transformers_snapshots.get(key)
            if snap is not None:
                # LRU recency: a restored-from key must not be the next one
                # bound-evicted just because it was stored long ago.
                self._transformers_snapshots.pop(key, None)
                self._transformers_snapshots[key] = snap
            return snap

    def _transformers_snapshot_bound(self) -> int:
        """Hard cap on resident snapshots (LRU beyond it).

        A hybrid Cache deepcopy is BIG. Qwen3.5-4B bf16 from config: 8
        full-attention layers x 4 kv-heads x 256 head-dim x 2 (K+V) x 2 B =
        32 KiB/token -> ~0.31 GiB at 10k + ~0.05-0.1 GiB linear/conv states
        ~= 0.36-0.41 GiB per snapshot (~1.0 GiB at 30k; ~2 GiB per 30k
        snapshot on a 27B) — so unlike the MLX lane (bound = store entries,
        32) the default here is 4, overridable via
        `ABSTRACTCORE_TRANSFORMERS_SNAPSHOT_BOUND` and never above the
        store's own entry bound (a snapshot only ever mirrors a store key;
        more snapshots than entries is by definition leaked state)."""
        bound = 4
        try:
            raw = os.environ.get("ABSTRACTCORE_TRANSFORMERS_SNAPSHOT_BOUND")
            if raw is not None and str(raw).strip():
                bound = max(1, int(str(raw).strip()))
        except Exception:
            bound = 4
        store = getattr(self, "_prompt_cache_store", None)
        try:
            store_bound = int(getattr(store, "_max_entries", 0) or 0)
        except Exception:
            store_bound = 0
        if store_bound > 0:
            bound = min(bound, store_bound)
        return bound

    def _store_transformers_snapshot(self, key: str, cache: Any, token_ids: List[int]) -> None:
        """Keep one snapshot per key (the growing one evicts its predecessor),
        and at most `_transformers_snapshot_bound()` snapshots overall (LRU).
        Dropped deepcopies land in the MPS pool; the threshold-guarded
        `_transformers_maybe_release_device_pool` on the generate path caps
        the ratchet (never an unconditional empty_cache on a hot path)."""
        self._ensure_transformers_snapshot_state()
        with self._transformers_snapshot_lock:
            self._transformers_snapshots.pop(key, None)
            self._transformers_snapshots[key] = {"cache": cache, "ids": list(token_ids)}
            bound = self._transformers_snapshot_bound()
            while len(self._transformers_snapshots) > bound:
                # dicts preserve insertion order; get/store re-insert = LRU.
                oldest = next(iter(self._transformers_snapshots))
                self._transformers_snapshots.pop(oldest, None)

    def _drop_transformers_snapshot(self, key: str) -> None:
        self._ensure_transformers_snapshot_state()
        with self._transformers_snapshot_lock:
            self._transformers_snapshots.pop(key, None)

    @staticmethod
    def _transformers_token_lcp(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _transformers_snapshot_lane_required(self, state: "_TransformersPromptCacheValue") -> bool:
        """Architecture routing for the snapshot lane, decided ONCE per loaded
        model: True when the cache has layers whose state cannot be rolled
        back (`_transformers_uncroppable_layers` — the same predicate the
        crop-refusal guard uses, so lane routing and refusal can never
        disagree). Pure-attention models return False and take the existing
        crop/delta path untouched. Undecidable (a cache with no layers yet)
        returns False WITHOUT caching, so a lazily-built hybrid cache is
        re-examined once its layers exist."""
        cached = getattr(self, "_transformers_snapshot_lane_flag", None)
        if isinstance(cached, bool):
            return cached
        probe = getattr(state, "cache", None)
        layers = getattr(probe, "layers", None)
        if not (isinstance(layers, (list, tuple)) and len(layers) > 0):
            fallback = self._transformers_empty_native_cache()
            f_layers = getattr(fallback, "layers", None)
            if isinstance(f_layers, (list, tuple)) and len(f_layers) > 0:
                probe = fallback
            else:
                return False  # undecidable now — do not cache the answer
        required = bool(self._transformers_uncroppable_layers(probe))

        # HYBRID + UNSAFE TRANSFORMERS -> REFUSE TO CACHE, LOUDLY.
        #
        # This is the only place that knows both facts at once: that the model is
        # a linear-attention hybrid, and which transformers is installed. On
        # versions below the floor the warm path returns confidently wrong text
        # (see `_HYBRID_CACHE_MIN_TRANSFORMERS`). Refusing costs a re-prefill;
        # not refusing costs the user a wrong answer they cannot detect. ADR 0001
        # allows a degradation only if it is announced, so this announces on
        # `warnings.warn` — `logger.warning` is dead here (root logger is ERROR,
        # every `abstractcore.*` logger NOTSET, record never created).
        if required:
            ok, ver = self._hybrid_cache_transformers_version_ok()
            if not ok:
                if not getattr(self, "_hybrid_cache_version_warned", False):
                    self._hybrid_cache_version_warned = True
                    floor = ".".join(str(p) for p in self._HYBRID_CACHE_MIN_TRANSFORMERS)
                    warnings.warn(
                        f"#FALLBACK prompt cache DISABLED for {self.model}: it is a "
                        f"linear-attention hybrid and transformers {ver} is below the "
                        f"{floor} floor required for a correct warm path. On affected "
                        f"versions a cached call returns fluent but WRONG answers with "
                        f"no error — measured warm recall 0/5 on planted facts. Answers "
                        f"stay correct; prefill is not reused. Upgrade transformers to "
                        f">= {floor} to re-enable caching for this model.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                self._transformers_snapshot_lane_flag = False
                self._transformers_hybrid_cache_blocked = True
                return False

        self._transformers_snapshot_lane_flag = required
        return required

    def _transformers_generation_prompt_literals(self) -> List[str]:
        """The exact generation-prompt tail(s) THIS provider's renderer emits.

        DERIVED, never mirrored (the MLX lane's `<think>\\n\\n</think>\\n\\n`
        lesson): `_transformers_build_prompt_fragment` with every content
        input empty renders nothing except the `add_generation_prompt`
        block, so calling it that way returns the literal itself for
        whichever renderer branch this model takes. `enable_thinking` is not
        known here, so both forms are produced and the caller matches
        longest-first (the thinking-disabled form extends the plain one)."""
        out: List[str] = []
        for thinking in (False, None):
            try:
                tail = self._transformers_build_prompt_fragment(
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            except Exception:
                continue
            if tail and tail not in out:
                out.append(tail)
        out.sort(key=len, reverse=True)
        return out

    def _transformers_generation_prompt_boundary(
        self, full_text: str, full_ids: List[int]
    ) -> Optional[int]:
        """Token POSITION where this call's generation-prompt scaffolding
        begins — per-call volatile by construction (the next turn's
        transcript replaces it with the assistant turn that actually
        happened), so a snapshot boundary must stop BEFORE it.

        A POSITION, deliberately, not a count: re-tokenize the prompt
        without the literal and take the token-level LCP against the full
        prompt's ids — if the seam merges into one token, a count would put
        the boundary one token INSIDE the scaffolding. Returns None when the
        prompt does not end in this renderer's generation prompt (no
        holdback is applied — never a guess)."""
        text = str(full_text or "")
        for tail in self._transformers_generation_prompt_literals():
            if not text.endswith(tail):
                continue
            head_ids = self._transformers_tokenize_fragment(
                text[: -len(tail)], add_bos_if_empty=True
            )
            if not head_ids:
                return None
            return self._transformers_token_lcp(head_ids, full_ids)
        return None

    def _transformers_snapshot_feed(
        self,
        key: str,
        state: "_TransformersPromptCacheValue",
        full_text: str,
        new_ids: List[int],
        telemetry: Dict[str, Any],
    ) -> List[int]:
        """Snapshot/restore feed for untrimmable architectures (full-context
        callers only). Decides what the cache should contain BEFORE
        `generate()` runs, mutates `state` accordingly, and returns the
        delta ids `generate()` must feed. On return, `state.prompt_tokens`
        describes exactly the tokens resident in `state.cache`.

        Lattice (ported from the MLX `_hybrid_snapshot_feed`):
        - live-cache seed: the cache holds exactly its recorded prompt ids
          and they are a true prefix of the new prompt (a key prefilled via
          `prompt_cache_update`, never generated into) → use it in place;
        - snapshot restore: the per-key snapshot's ids are a TRUE PREFIX of
          the new prompt with a suffix left to feed → deepcopy-restore,
          forward-only, never a rollback;
        - otherwise: one honest cold prefill on a fresh cache (release the
          old one first — leak audit 2026-08-03) that still leaves a
          snapshot behind, so the loop's next warm turn is cheap.

        The snapshot boundary is chosen conservatively BEFORE this call's
        volatile tail: previous-prompt LCP holdback when a fed-token record
        exists, renderer-derived generation-scaffolding holdback on a
        recordless key (turn 1). The deepcopy happens at that boundary,
        before the tail prefill and before generation mutates the cache."""
        telemetry.setdefault("lane", "snapshot")
        meta = dict(self._prompt_cache_store.meta(key) or {})
        raw_prev = meta.get(self._TRANSFORMERS_FED_IDS_META)
        prev_ids: List[int] = []
        if isinstance(raw_prev, (list, tuple)):
            try:
                prev_ids = [int(t) for t in raw_prev]
            except Exception:
                prev_ids = []

        live_ids = [int(t) for t in state.prompt_tokens]
        live_len = len(live_ids)
        prefix_len = 0
        restored = False
        live_seed = False

        # PHYSICAL length check (MLX parity: `cache_len == len(fed_ids)`,
        # mlx_provider.py live-seed). Bookkeeping alone is not trustworthy:
        # generate() can raise after mutating the cache, leaving phantom KV
        # past the record (ADVERSARY FINDING 1). On hybrids get_seq_length
        # reads the full-attention layers, which count real tokens; an
        # uncountable cache (no layer reports) can never verify — skip the
        # seed and let restore/rebuild handle it.
        phys_lens = (
            self._transformers_layer_seq_lengths(state.cache)
            if state.cache is not None else []
        )
        phys_len = max(phys_lens) if phys_lens else None

        if (
            live_len
            and state.cache is not None
            and phys_len == live_len
            and live_len < len(new_ids)
            and self._transformers_token_lcp(live_ids, new_ids) == live_len
        ):
            # LIVE-CACHE SEED: no generated drift — the cache IS a valid
            # boundary already; forward-only extension, no copy needed.
            prefix_len = live_len
            live_seed = True
        else:
            snap = self._get_transformers_snapshot(key)
            if snap is not None:
                snap_ids = list(snap.get("ids") or [])
                lcp_snap = self._transformers_token_lcp(snap_ids, new_ids)
                if snap_ids and lcp_snap == len(snap_ids) and lcp_snap < len(new_ids):
                    # Drop the stale live cache FIRST so the MPS pool can hand
                    # its buffers straight to the clone (never old + snapshot
                    # + clone resident at once). If the clone then fails, the
                    # not-restored branch below rebuilds fresh — same result
                    # the old cache was headed for anyway.
                    state.cache = None
                    state.prompt_tokens = ()
                    clone = self._transformers_clone_cache(snap.get("cache"))
                    if clone is not None:
                        state.cache = clone
                        state.prompt_tokens = tuple(snap_ids)
                        prefix_len = lcp_snap
                        restored = True
            if not restored:
                if live_len:
                    # Divergence with nothing restorable: rebuild = RELEASE
                    # point for the old full cache (existing discipline).
                    state.cache = None
                    self._transformers_release_device_pool()
                    state.cache = self._transformers_empty_native_cache()
                    state.prompt_tokens = ()
                elif state.cache is None:
                    state.cache = self._transformers_empty_native_cache()
                prefix_len = 0

        boundary_end = max(len(new_ids) - 1, 0)  # keep ≥1 token to seed decode
        stable_end = boundary_end
        if prev_ids:
            shared = self._transformers_token_lcp(prev_ids, new_ids)
            if shared < len(prev_ids):
                # The previous prompt's tail was rewritten: whatever the two
                # prompts share is stable transcript; the first divergence is
                # where THIS turn's ephemeral tail begins. `prefix_len` is a
                # floor — never regress below a boundary we restored from.
                stable_end = max(prefix_len, min(shared, boundary_end))
        if prefix_len > 0 and not restored:
            # First snapshot-lane turn seeded from the LIVE cache: the seed
            # is the only content two observations agree on.
            stable_end = prefix_len
        if not prev_ids:
            gen_at = self._transformers_generation_prompt_boundary(full_text, new_ids)
            if gen_at is not None:
                stable_end = max(0, min(stable_end, gen_at))

        def _rebuild_full_feed(reason: str) -> List[int]:
            self._drop_transformers_snapshot(key)
            state.cache = None
            self._transformers_release_device_pool()
            state.cache = self._transformers_empty_native_cache()
            state.prompt_tokens = ()
            meta[self._TRANSFORMERS_FED_IDS_META] = list(new_ids)
            try:
                self._prompt_cache_store.set(key, state, meta=meta)
            except Exception:
                pass
            telemetry.update(
                {"outcome": "rebuilt", "cached_tokens": 0, "fed_tokens": len(new_ids),
                 "degraded_reason": f"#FALLBACK {reason}"}
            )
            return list(new_ids)

        head = list(new_ids[prefix_len:stable_end])
        if head and not self._transformers_prefill_cache(state, head):
            return _rebuild_full_feed("snapshot-lane head prefill failed; rebuilt fresh")

        # Snapshot the clean boundary (deepcopy) BEFORE generation mutates the
        # cache. Skipped only when a snapshot we actually RESTORED from already
        # describes this exact boundary. The `len == stable_end` invariant
        # guards against ever storing ids that misdescribe the cache.
        boundary_ids = list(new_ids[:stable_end])
        if boundary_ids and len(state.prompt_tokens) == stable_end and (
            stable_end > prefix_len or not restored
        ):
            snap_copy = self._transformers_clone_cache(state.cache)
            if snap_copy is not None:
                self._store_transformers_snapshot(key, snap_copy, boundary_ids)
            else:
                self._drop_transformers_snapshot(key)
        elif not boundary_ids:
            self._drop_transformers_snapshot(key)

        meta[self._TRANSFORMERS_FED_IDS_META] = list(new_ids)
        try:
            self._prompt_cache_store.set(key, state, meta=meta)
        except Exception:
            pass

        delta = list(new_ids[stable_end:])
        fed_this_turn = len(head) + len(delta)
        if prefix_len > 0:
            telemetry.update(
                {"outcome": "hit_restore", "cached_tokens": int(prefix_len),
                 "fed_tokens": int(fed_this_turn)}
            )
        elif not prev_ids and not live_len:
            # TURN ONE ON A FRESH KEY is `cold`, not `rebuilt` (parity,
            # 2026-08-07) — nothing existed to discard. Matches the MLX snapshot
            # lane and the GGUF lane, which already said `cold` here.
            telemetry.update(
                {"outcome": "cold", "cached_tokens": 0, "fed_tokens": int(fed_this_turn)}
            )
        else:
            telemetry.update(
                {"outcome": "rebuilt", "cached_tokens": 0, "fed_tokens": int(fed_this_turn)}
            )
        return delta

    def _transformers_arch_prefix_suffix(self, role: str) -> tuple[str, str]:
        cfg = getattr(self, "architecture_config", None)
        if not isinstance(cfg, dict):
            cfg = {}
        r = str(role or "").strip().lower()
        if r == "system":
            return str(cfg.get("system_prefix") or ""), str(cfg.get("system_suffix") or "")
        if r == "user":
            return str(cfg.get("user_prefix") or ""), str(cfg.get("user_suffix") or "")
        if r == "assistant":
            return str(cfg.get("assistant_prefix") or ""), str(cfg.get("assistant_suffix") or "")
        # Fallback: simple conversational format.
        return "", "\n"

    def _thinking_disable_prefill(self, enable_thinking: Optional[bool]) -> str:
        """Assistant-generation-prompt prefill that disables thinking, or ``""``.

        The marker is whatever the model's registry entry DECLARES as
        `thinking_control.assistant_prefill_disable` — never a hardcoded
        architecture allow-list. Every renderer in this provider that emits a
        generation prompt by hand (rather than through the tokenizer's chat
        template) must ask here, because a hand-rendered prompt gets no
        `enable_thinking` template variable and would otherwise leave a
        reasoning model free to open a `<think>` block.

        Why an allow-list was wrong (2026-08-05, measured on
        `deepreinforce-ai/Ornith-1.0-9B` bf16/MPS): the set
        `{"qwen3", "qwen3_5", "qwen3_6"}` excluded `qwen3_5_agentic` (Ornith,
        Qwen-AgentWorld, Agents-A1), so on the KEYED-CACHE lane — the only lane
        that uses the hand renderer — the prompt ended at a bare
        `<|im_start|>assistant\\n`, the model's first generated token was
        `<think>` (id 248068), and `strip_thinking_tags` turned that
        unterminated block into EMPTY visible content: every cached arm
        returned "" with finish_reason=stop and no exception, while the
        uncached arm (chat template + `enable_thinking=False`) was healthy.
        Reading the declared surface makes a new model family work by registry
        update alone, which is the contract the typed surfaces exist for."""
        if enable_thinking is not False:
            return ""
        try:
            return self._thinking_control_surfaces().assistant_prefill_disable or ""
        except Exception:
            return ""

    def _transformers_render_message(self, role: str, content: str, *, close: bool = True) -> str:
        prefix, suffix = self._transformers_arch_prefix_suffix(role)
        if prefix or suffix:
            out = f"{prefix}{content}"
            if close:
                out += suffix
            return out
        label = str(role or "user").strip().capitalize() or "User"
        out = f"{label}: {content}\n"
        return out if close else out.rstrip("\n")

    def _transformers_build_prompt_fragment(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        add_generation_prompt: bool = False,
        prefilled_modules: Optional[Union[List[str], tuple[str, ...]]] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Build a prompt fragment that can be appended to an existing KV cache."""

        prefilled: set[str] = set()
        if prefilled_modules:
            for item in prefilled_modules:
                try:
                    norm = str(item or "").strip().lower()
                except Exception:
                    norm = ""
                if norm:
                    prefilled.add(norm)

        def _as_text(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        parts: List[str] = []

        base_system_prompt = str(system_prompt or "").strip() if system_prompt is not None else ""
        tool_prompt = ""
        if tools is not None and getattr(self, "tool_handler", None) is not None:
            if getattr(self.tool_handler, "supports_prompted", False) and "tools" not in prefilled:
                include_tool_list = True
                if base_system_prompt and "## Tools (session)" in base_system_prompt:
                    include_tool_list = False
                try:
                    tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
                except Exception:
                    tool_prompt = ""
                tool_prompt = str(tool_prompt or "").strip()

        # Reasoning-effort instruction (Qwen3.8-style): the model's own chat template
        # controls effort by prepending a per-level sentence to the FIRST system block
        # (asset surface `thinking_control.effort_system_lines`; empty value = level
        # renders no text). Only when this fragment owns the system region — a
        # prefilled system bloc was serialized without the line. Merged into a
        # leading system message inside `messages` rather than opening a second
        # consecutive system block (same rule as the MLX renderer).
        effort_line = ""
        if isinstance(reasoning_effort, str) and reasoning_effort and "system" not in prefilled:
            effort_lines = self._thinking_control_surfaces().effort_system_lines or {}
            candidate = effort_lines.get(reasoning_effort)
            if isinstance(candidate, str) and candidate.strip():
                effort_line = candidate.strip()

        if base_system_prompt and "system" not in prefilled:
            # ONE system turn (parity with _gguf_build_chat_messages and the
            # uncached _build_input_text_transformers, which already merged):
            # tools join the user's system message instead of opening a second
            # consecutive system block — chat templates are trained on exactly
            # one system turn. When the system module is already prefilled in
            # the KV cache its block is closed and cannot be reopened, so the
            # tool prompt below still enters as its own block (module-chain
            # appends carry system and tools in separate calls; their rendered
            # bytes are unchanged by this merge).
            system_block = base_system_prompt
            if tool_prompt:
                system_block = f"{base_system_prompt}\n\n{tool_prompt}"
                tool_prompt = ""
            if effort_line:
                system_block = f"{effort_line}\n\n{system_block}"
                effort_line = ""
            parts.append(self._transformers_render_message("system", system_block, close=True))

        if tool_prompt:
            if effort_line:
                tool_prompt = f"{effort_line}\n\n{tool_prompt}"
                effort_line = ""
            parts.append(self._transformers_render_message("system", tool_prompt, close=True))

        if effort_line and messages and isinstance(messages[0], dict) and (
            str(messages[0].get("role") or "").strip().lower() == "system"
        ):
            first = dict(messages[0])
            first["content"] = f"{effort_line}\n\n{_as_text(first.get('content'))}"
            messages = [first, *list(messages)[1:]]
            effort_line = ""

        if effort_line:
            parts.append(self._transformers_render_message("system", effort_line, close=True))
            effort_line = ""

        if messages:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "user").strip().lower() or "user"
                if role in {"tool", "function"}:
                    role = "assistant"
                content = _as_text(msg.get("content"))
                parts.append(self._transformers_render_message(role, content, close=True))

        if isinstance(prompt, str) and prompt:
            parts.append(self._transformers_render_message("user", str(prompt), close=True))

        if add_generation_prompt:
            parts.append(self._transformers_render_message("assistant", "", close=False))
            parts.append(self._thinking_disable_prefill(enable_thinking))

        return "".join(parts)

    def _transformers_tokenize_fragment(self, fragment: str, *, add_bos_if_empty: bool) -> List[int]:
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            return []

        text = str(fragment or "")
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except Exception:
            try:
                ids = list(tok(text, add_special_tokens=False)["input_ids"])
            except Exception:
                return []

        out = [int(i) for i in ids] if ids else []
        if add_bos_if_empty:
            bos = getattr(tok, "bos_token_id", None)
            add_bos = getattr(tok, "add_bos_token", None)
            if isinstance(add_bos, bool) and not add_bos:
                return out
            try:
                bos_i = int(bos) if bos is not None else None
            except Exception:
                bos_i = None
            if bos_i is not None and bos_i >= 0:
                if not out or out[0] != bos_i:
                    out.insert(0, bos_i)
        return out

    @staticmethod
    def _quantization_config_is_bnb_4bit(cfg: Dict[str, Any]) -> bool:
        """Does this quantization config mean bitsandbytes 4-bit?"""
        if not cfg:
            return False
        try:
            if cfg.get("load_in_4bit"):
                return True
            method = str(cfg.get("quant_method") or "").lower()
            if method in ("bitsandbytes", "bnb", "bitsandbytes_4bit"):
                return bool(cfg.get("load_in_4bit")) or int(cfg.get("bits") or 0) == 4
        except Exception:
            return False
        return False

    def _prepare_bnb_mps_fused_kernel(self, quantization_config: Dict[str, Any]) -> None:
        """Resolve the fused Metal kernel BEFORE the weights are loaded.

        Timing is the whole point and it was learned the hard way: loading a
        bnb-quantized checkpoint QUANTIZES as it loads, which calls
        `bitsandbytes...ops._get_kernel()` — under the offline flag — and latches
        `_kernel_load_failed = True` before any post-load hook can run. A probe
        placed after `from_pretrained` therefore always finds a dead latch and
        can only report it. Measured exactly that way before this hook existed:
        `probe=failed bnb_kernel_live=False latched=True linear4bit=248`.

        So the resolution has to happen here, before the first quantize op.
        Cheap and inert otherwise: it returns immediately unless the target is
        bitsandbytes 4-bit on MPS.
        """
        try:
            if str(getattr(self, "device", "") or "").strip().lower() != "mps":
                return
            requested = getattr(self, "_transformers_quantization_request", None)
            as_dict: Dict[str, Any] = {}
            if requested is not None:
                as_dict = requested if isinstance(requested, dict) else (
                    requested.to_dict() if hasattr(requested, "to_dict") else {})
            if not (self._quantization_config_is_bnb_4bit(as_dict)
                    or self._quantization_config_is_bnb_4bit(quantization_config or {})):
                return
            kernel, how = _resolve_bnb_mps_fused_kernel()
            self._bnb_mps_fused_kernel_preload = {
                "resolved": kernel is not None, "resolved_via": how}
        except Exception:  # noqa: BLE001 - must never break a model load
            pass

    def _transformers_count_bnb_4bit_modules(self) -> int:
        """Number of live `bitsandbytes` `Linear4bit` modules in the loaded model."""
        model = getattr(self, "model_instance", None)
        if model is None or "bitsandbytes" not in sys.modules:
            return 0
        try:
            return sum(1 for _, m in model.named_modules() if type(m).__name__ == "Linear4bit")
        except Exception:
            return 0

    def _warn_if_bnb_mps_fused_kernel_missing(self) -> bool:
        """Warn ONCE per provider instance when a 4-bit model on MPS has no fused kernel.

        bitsandbytes degrades silently here (see `_probe_bnb_mps_fused_kernel`):
        the model loads, answers correctly, and decodes about x4 slower with no
        signal of any kind. ADR 0001/0009 forbid silent degradation, so state it.

        Never raises: an unusable probe leaves the load untouched.
        """
        try:
            if str(getattr(self, "device", "") or "").strip().lower() != "mps":
                return False
            n_4bit = self._transformers_count_bnb_4bit_modules()
            if n_4bit <= 0:
                return False
            record = _probe_bnb_mps_fused_kernel()
            record["linear4bit_modules"] = n_4bit
            self._bnb_mps_fused_kernel_probe = record
            if record.get("available"):
                return False
            if getattr(self, "_bnb_mps_fused_kernel_warned", False):
                return False
            self._bnb_mps_fused_kernel_warned = True
            cause = record.get("reason") or "unknown"
            error = record.get("error")
            remedy = record.get("remedy") or "unknown"
            self.logger.warning(
                f"#FALLBACK bitsandbytes 4-bit on MPS: the fused Metal kernel "
                f"({_BNB_MPS_FUSED_KERNEL_REPO}) is NOT available, so all "
                f"{n_4bit} Linear4bit module(s) run the dequantize -> F.linear "
                f"fallback on every forward. Output stays correct; decode is about "
                f"x4 SLOWER (measured on this stack: warm identical resend 0.0681s "
                f"fused vs 0.2696s fallback = x3.96; per-forward A/B x4.65-x4.79). "
                f"bitsandbytes swallows this failure and latches it for the whole "
                f"process, so nothing else reports it. "
                f"Cause: {cause}"
                + (f" [{error}]" if error else "")
                + f". Fix: {remedy}. "
                f"Versions seen: kernels={record.get('kernels_version')}, "
                f"bitsandbytes={record.get('bitsandbytes_version')}, "
                f"macOS major={record.get('macos_major')}."
            )
            return True
        except Exception:
            return False

    def _transformers_release_device_pool(self) -> None:
        """Return freed device memory to the OS after a cache RELEASE point.

        torch's MPS allocator pools every freed buffer for reuse and never
        returns it to the OS on its own; on unified memory the pool counts
        against process footprint. Before 2026-08-03 the ONLY call site of
        `torch.mps.empty_cache()` was the vision lane, so the text/cached
        lanes' release paths (key cleared, LRU-evicted, hybrid rebuild
        replacing a full cache) parked every dropped KV cache in the pool
        forever — the 164.5 GB peak class of failure. Release points only:
        NEVER on the per-token or per-call hot path (empty_cache walks the
        allocator; a performance decrease is unacceptable)."""
        if str(getattr(self, "device", "") or "").strip().lower() != "mps":
            return
        try:
            import torch  # type: ignore

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    if hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    torch.mps.empty_cache()
        except Exception:
            pass

    def _prompt_cache_store_evicted(self, key: str, cache_value: Any) -> None:
        """Store LRU/TTL-evicted `key`: its `_TransformersPromptCacheValue`
        just lost its last durable reference. Drop the key's hybrid snapshot
        too (a deepcopy must never outlive the entry it mirrors — 2026-08-03
        leak audit), then return the freed KV tensors' device memory to the
        OS instead of the MPS pool. Fires only on implicit evictions
        (≥ max_entries concurrent keys / TTL) — not on the per-call
        store.set of a resident key."""
        try:
            self._drop_transformers_snapshot(str(key))
        except Exception:
            pass
        if self._transformers_prompt_cache_state(cache_value) is not None:
            self._transformers_release_device_pool()
            return
        # GGUF: the key's boundary snapshots are `LlamaState`s holding the whole
        # serialized context (multi-GB at benchmark context sizes). They die with
        # the cache object, but only once the last reference does — drop them here
        # so an LRU/TTL eviction actually returns the memory at eviction time.
        gguf_cache = self._gguf_prompt_cache_unwrap(cache_value)
        if gguf_cache is not None:
            state_map = getattr(gguf_cache, "cache_state", None)
            if hasattr(state_map, "clear"):
                try:
                    state_map.clear()
                except Exception:
                    pass

    def _transformers_pool_release_threshold_bytes(self) -> int:
        """Pooled-but-unused MPS bytes above which the text lanes release the
        pool between calls. Env `ABSTRACTCORE_MPS_POOL_RELEASE_GB` (float GiB;
        <= 0 disables), default 4 GiB."""
        cached = getattr(self, "_transformers_pool_release_threshold", None)
        if isinstance(cached, int):
            return cached
        gib = 4.0
        try:
            raw = os.environ.get("ABSTRACTCORE_MPS_POOL_RELEASE_GB")
            if raw is not None and str(raw).strip():
                gib = float(raw)
        except Exception:
            gib = 4.0
        threshold = int(gib * 1073741824) if gib > 0 else 0
        self._transformers_pool_release_threshold = threshold
        return threshold

    def _transformers_maybe_release_device_pool(self) -> None:
        """Threshold-guarded pool release for the text-lane HOT paths (cached
        and uncached generate).

        The 164.5 GB incident record (`hf_bf16_30000.json`) shows the blowup
        happened during floor + A_nocache + B_cold — arms C/D never ran — so
        release points on keyed-cache paths alone cannot protect the lane that
        actually died: the UNCACHED pipeline path has no cache to release,
        only per-forward transients that the MPS allocator pools forever.

        Design constraint: a performance decrease is unacceptable, so the pool
        is NEVER dropped while it is doing its job. This reads two allocator
        counters (cheap) and calls `empty_cache` only when the pool holds more
        than `_transformers_pool_release_threshold_bytes()` of freed-but-
        retained memory — i.e. only in the pathological regime. Healthy small-
        call loops never cross the threshold and keep full pool reuse. If the
        counters are unavailable, do NOTHING (an unconditional empty_cache on
        the hot path is exactly the perf hazard this guard exists to avoid).

        MEASURED AND DELIBERATELY NOT TUNED (FINISHER, 2026-08-06). A hot-path
        exemption — skip the release on decode-shaped cached calls, hold them to
        a raised pooled-bytes bound — was implemented and A/B'd in-process on one
        resident model, one knob, both orders
        (`results/bench_b/finisher_pool_ab_v2_ornith9b_30000.json`). It was
        REVERTED on its own evidence: the dispatch it removes costs 3.8 ms of a
        124-164 ms warm call (~2-3%, not the 13% reported), the arm-level
        difference is swamped by a same-arm drift of 0.124 s -> 0.164 s across
        the cell, and retained pool rose (max slack 7.482 GiB exempt vs 6.634
        GiB shipped, parked at ~7.4 vs ~4.2 GiB). The discriminator is also
        caller-shape dependent: the same logical warm call feeds 1 token through
        the `messages=` full-context lane and 35 through the `prompt=` append
        lane, so the exemption would apply to one spelling and silently not the
        other. 3.8 ms is not worth reintroducing the ratchet this guard exists
        to bound."""
        if str(getattr(self, "device", "") or "").strip().lower() != "mps":
            return
        threshold = self._transformers_pool_release_threshold_bytes()
        if threshold <= 0:
            return
        try:
            import torch  # type: ignore

            if not (hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache")):
                return
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                return
            driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
            current_fn = getattr(torch.mps, "current_allocated_memory", None)
            if not callable(driver_fn) or not callable(current_fn):
                return
            pooled = int(driver_fn()) - int(current_fn())
            if pooled >= threshold:
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                torch.mps.empty_cache()
        except Exception:
            pass

    def _transformers_pool_guard_stride(self) -> int:
        """Forwards between pooled-bytes checks inside a live `generate()` call.

        Env `ABSTRACTCORE_MPS_POOL_GUARD_STRIDE`, default 8; <= 0 DISABLES the
        in-call guard, matching the `<= 0 disables` convention of
        `ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP` and
        `ABSTRACTCORE_MPS_POOL_RELEASE_GB`. The switch exists so the fix can be
        A/B'd against its own absence on one resident model rather than by
        editing the source between arms — that is how
        `oom/results/consecutive_*_12k.json` was produced."""
        cached = getattr(self, "_transformers_pool_guard_stride_cached", None)
        if isinstance(cached, int):
            return cached
        stride = 8
        try:
            raw = os.environ.get("ABSTRACTCORE_MPS_POOL_GUARD_STRIDE")
            if raw is not None and str(raw).strip():
                stride = int(str(raw).strip())
        except Exception:
            stride = 8
        self._transformers_pool_guard_stride_cached = stride
        return stride

    @contextlib.contextmanager
    def _transformers_decode_pool_guard(self):
        """Keep the MPS allocator pool bounded DURING a single `generate()` call.

        WHY THIS EXISTS (2026-08-07, measured). `_transformers_maybe_release_
        device_pool` is CALL-scoped: every text-lane call site invokes it in a
        `finally`, i.e. after `generate()` has already returned. The ratchet it
        is meant to bound is STEP-scoped. `DynamicCache` grows by `torch.cat` on
        every decode step, so step *t* asks the allocator for a buffer sized for
        *t* tokens and frees the *t-1* one. The sizes never repeat and never
        shrink, which is the worst possible input to a caching allocator that
        only reuses blocks it already holds and never returns them to the OS on
        its own.

        MEASURED, Qwen3.5-4B bf16 on MPS, 12,718-token UNCACHED prompt, one
        process, one model load (`oom/results/decode_budget_v1.json`):

            max_output_tokens=512   ->  driver  10.26 GiB, pool slack   1.89 GiB
            max_output_tokens=4096  ->  driver 113.26 GiB, pool slack 104.78 GiB

        In the second arm `current_allocated_memory` was 8.48 GiB: the
        computation needed ~8.5 GiB and the allocator was sitting on 104.8 GiB
        of FREED buffers. Host free+inactive fell from 111 GB to 18 GB inside
        one call and a watchdog had to stop it. Eight times the decode budget
        cost a hundred times the memory, because the growth is not steady: the
        driver sits flat for hundreds of steps and then jumps in 1-2 GiB
        heap-sized increments as the requested sizes outgrow every heap the
        allocator already owns (`oom/results/decode_mechanism.json`). Nothing
        inside `generate()` ever yields to the call-scoped guard, so one long
        call ratchets without limit — and a call that never returns never
        releases at all, which is the shape of the two gateway deaths in
        `product/results/hf_gateway_death.json`.

        This does NOT change the policy: same two counters, same
        `_transformers_pool_release_threshold_bytes()` bound, still a no-op
        while the pool is doing its job, still never an unconditional
        `empty_cache` on the hot path. It only gives the existing policy
        somewhere to run while a call is still in flight, through a read-only
        forward pre-hook. Nothing about generation semantics is touched — the
        hook inspects no tensors, returns nothing, and swallows its own errors.
        Stride-limited (`ABSTRACTCORE_MPS_POOL_GUARD_STRIDE`, default 8
        forwards) so two counter reads do not land on literally every token.

        Yields a stats dict so callers and tests can assert the guard actually
        ran rather than assuming it: {"checks", "releases", "seconds",
        "peak_pooled_bytes"}.
        """
        stats: Dict[str, Any] = {"checks": 0, "releases": 0, "seconds": 0.0,
                                 "peak_pooled_bytes": 0, "enabled": False}
        model = getattr(self, "model_instance", None)
        try:
            threshold = self._transformers_pool_release_threshold_bytes()
        except Exception:
            threshold = 0
        if (str(getattr(self, "device", "") or "").strip().lower() != "mps"
                or threshold <= 0
                or model is None
                or not hasattr(model, "register_forward_pre_hook")):
            yield stats
            return
        try:
            import torch  # type: ignore

            if not (hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache")):
                yield stats
                return
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                yield stats
                return
            driver_fn = getattr(torch.mps, "driver_allocated_memory", None)
            current_fn = getattr(torch.mps, "current_allocated_memory", None)
            if not callable(driver_fn) or not callable(current_fn):
                yield stats
                return
        except Exception:
            yield stats
            return

        stride = self._transformers_pool_guard_stride()
        if stride <= 0:  # operator kill-switch / A-B control arm
            yield stats
            return
        stats["enabled"] = True
        seen = {"n": 0}

        def _check() -> None:
            seen["n"] += 1
            if seen["n"] % stride:
                return
            try:
                pooled = int(driver_fn()) - int(current_fn())
                if pooled > stats["peak_pooled_bytes"]:
                    stats["peak_pooled_bytes"] = pooled
                stats["checks"] += 1
                if pooled >= threshold:
                    t0 = time.time()
                    if hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    torch.mps.empty_cache()
                    stats["seconds"] += time.time() - t0
                    stats["releases"] += 1
            except Exception:
                pass

        handle = None
        try:
            try:
                handle = model.register_forward_pre_hook(
                    lambda _m, _a, _k: _check(), with_kwargs=True)
            except TypeError:
                handle = model.register_forward_pre_hook(lambda _m, _a: _check())
        except Exception:
            handle = None
        try:
            yield stats
        finally:
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass

    def _transformers_crop_cache(self, state: _TransformersPromptCacheValue, keep_tokens: int) -> bool:
        """Crop the live KV cache back to its first `keep_tokens` tokens.

        REFUSES on any architecture with layers whose state cannot be rolled
        back. Returns False; the caller rebuilds fresh (one honest cold prefill).

        WHY THIS IS A REFUSAL AND NOT AN ACCEPTED APPROXIMATION (2026-08-03).
        `transformers.cache_utils` implements `crop` as an explicit `pass` on the
        linear-attention layer classes ("We don't crop the linear attention cache,
        so simply do nothing here"), while `Cache.crop` loops every layer. On a
        hybrid the result is not an approximation, it is an INTERNALLY INCONSISTENT
        cache: the full-attention layers roll back, the linear layers keep the
        removed tokens' recurrent state, and the two halves then disagree about
        what context they hold. Measured on `Qwen/Qwen3.5-4B` (24 linear / 8 full
        of 32 layers): 24 of 32 layers BIT-IDENTICAL after a crop at 10k, and
        reproduced at 30k. Behaviourally it does not degrade gracefully — cropped-
        warm arms returned EMPTY completions (`finish_reason='stop'`, no content)
        across retries until the circuit breaker tripped.

        This code previously accepted that case with a warning, on the theory that
        "the long attention prefix is reused exactly and the residual linear state
        is an approximation". The empty-output measurement refutes the theory, and
        a warning is not a substitute for a correct cache.

        WHY THE OLD VERIFY MISSED IT. It read `cache.get_seq_length()`, and
        `Cache.get_seq_length` deliberately skips to the first attention layer for
        exactly these hybrids (`cache_utils.py`: "For alternating attention/linear
        attention caches, `get_seq_length` needs to use attention layer idx"). It
        therefore sampled a layer that DID crop and always passed. The verify below
        checks EVERY layer that can report a length instead of trusting one.
        """
        cache = getattr(state, "cache", None)
        if cache is None:
            return False
        crop = getattr(cache, "crop", None)
        if not callable(crop):
            return False

        # Refuse BEFORE mutating. `Cache.crop` rolls back the attention layers even
        # when it no-ops the linear ones, so a post-hoc refusal would hand the
        # caller back the very inconsistent cache this guard exists to prevent.
        uncroppable = self._transformers_uncroppable_layers(cache)
        if uncroppable:
            if not getattr(self, "_transformers_linear_crop_warned", False):
                self._transformers_linear_crop_warned = True
                shown = ", ".join(f"[{i}]{n}" for i, n in uncroppable[:4])
                self.logger.warning(
                    f"#FALLBACK transformers prompt cache: this model has "
                    f"{len(uncroppable)} layer(s) whose state cannot be rolled back "
                    f"({shown}{' …' if len(uncroppable) > 4 else ''}). `crop` is a no-op on "
                    f"them while attention layers DO roll back, which leaves an inconsistent "
                    f"cache that produces empty completions. Cropping is refused for this "
                    f"architecture; warm calls rebuild fresh (one cold prefill, correct "
                    f"output, no prefill savings)."
                )
            return False

        try:
            crop(int(keep_tokens))
        except Exception:
            return False

        # Post-crop verify, PER LAYER. `cache.get_seq_length()` alone cannot detect
        # a partial rollback — that is precisely how the hybrid case survived.
        try:
            for length in self._transformers_layer_seq_lengths(cache):
                if length > int(keep_tokens):
                    return False
        except Exception:
            pass  # unverifiable: keep legacy accept (raise-only contract)
        return True

    @staticmethod
    def _transformers_layer_seq_lengths(cache: Any) -> List[int]:
        """Every per-layer sequence length this cache can report.

        Layers that do not track a length at all (linear-attention/recurrent)
        contribute nothing — they are unverifiable by construction, which is why
        they are refused up front rather than checked here.
        """
        out: List[int] = []
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, (list, tuple)):
            get_len = getattr(cache, "get_seq_length", None)
            if callable(get_len):
                try:
                    out.append(int(get_len()))
                except Exception:
                    pass
            return out
        for layer in layers:
            fn = getattr(layer, "get_seq_length", None)
            if not callable(fn):
                continue
            try:
                out.append(int(fn()))
            except Exception:
                continue
        return out

    # Minimum transformers version whose HYBRID (linear-attention) warm path is
    # trustworthy on this provider. Below this, a warm bloc-chain/snapshot call on
    # a Gated-DeltaNet model returns FLUENT, CONFIDENT, WRONG text — not an error,
    # not an empty string. Measured on Qwen3.5-4B, stock sdpa, three gate cells:
    #
    #   transformers 5.6.0  ->  warm recall 0/5 at every planted-fact depth
    #                           cold "designated KESTREL-9" / warm "designated as Loop C"
    #                           cold "3.7 microradians"     / warm "the log does not contain
    #                                                              any record of segment M-14"
    #   transformers 5.9.0  ->  PASS
    #
    # Deterministic, zero errors, nothing in the payload signals failure — the
    # worst shape a cache defect can take. `pyproject.toml` pins
    # `transformers>=4.57.1,<6.0.0`, which ADMITS 5.6.0, and the shim that once
    # masked this (`abstractcore_sdpa_mps_safe`) no longer exists in the tree. So a
    # spec-compliant install plus a default-on cache plus a hybrid model produced
    # silently wrong answers with no guard anywhere. This is that guard.
    _HYBRID_CACHE_MIN_TRANSFORMERS = (5, 9, 0)

    @classmethod
    def _hybrid_cache_transformers_version_ok(cls) -> tuple[bool, str]:
        """(ok, version_string) for the installed transformers on the hybrid path."""
        try:
            import transformers  # type: ignore

            raw = str(getattr(transformers, "__version__", "") or "")
            parts = []
            for chunk in raw.split(".")[:3]:
                digits = "".join(ch for ch in chunk if ch.isdigit())
                parts.append(int(digits) if digits else 0)
            while len(parts) < 3:
                parts.append(0)
            return (tuple(parts) >= cls._HYBRID_CACHE_MIN_TRANSFORMERS, raw)
        except Exception:  # noqa: BLE001
            # Cannot prove the version is safe -> treat as unsafe. A cache that
            # might return wrong answers must fail closed.
            return (False, "unknown")

    @staticmethod
    def _transformers_uncroppable_layers(cache: Any) -> List[tuple]:
        """`(index, class_name)` for every layer whose `crop` cannot roll state back.

        Identified by TYPE where transformers exposes the type
        (`LinearAttentionCacheLayerMixin`, the base of all three linear-attention
        layer classes — including `LinearAttentionAndFullAttentionLayer`, whose MRO
        puts the no-op `crop` ahead of its own attention half). The name heuristic
        is only a fallback for versions that do not export the mixin: a substring
        match is not a contract, and this predicate now decides correctness rather
        than the wording of a warning.
        """
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, (list, tuple)):
            return []
        mixin = None
        try:
            from transformers.cache_utils import LinearAttentionCacheLayerMixin  # type: ignore

            mixin = LinearAttentionCacheLayerMixin
        except Exception:
            mixin = None

        out: List[tuple] = []
        for idx, layer in enumerate(layers):
            name = type(layer).__name__
            if mixin is not None and isinstance(layer, mixin):
                out.append((idx, name))
                continue
            lowered = name.lower()
            if "linear" in lowered or "mamba" in lowered or "conv" in lowered or "recurrent" in lowered:
                out.append((idx, name))
                continue
            # A layer that cannot report a length cannot be verified after a crop
            # either; treat it as uncroppable rather than trust it silently.
            if mixin is None and not callable(getattr(layer, "get_seq_length", None)):
                out.append((idx, name))
        return out

    def _transformers_forward_supports_logits_to_keep(self) -> bool:
        """True when the loaded model's forward() accepts `logits_to_keep`.

        Mirrors transformers' own `GenerationMixin._supports_logits_to_keep`;
        computed once per loaded model (signature inspection is not free and
        the prefill path can run per turn)."""
        cached = getattr(self, "_transformers_logits_to_keep_supported", None)
        if isinstance(cached, bool):
            return cached
        supported = False
        model = getattr(self, "model_instance", None)
        try:
            probe = getattr(model, "_supports_logits_to_keep", None)
            if callable(probe):
                supported = bool(probe())
            elif model is not None:
                import inspect

                supported = "logits_to_keep" in set(
                    inspect.signature(model.forward).parameters.keys()
                )
        except Exception:
            supported = False
        self._transformers_logits_to_keep_supported = supported
        return supported

    def _transformers_prefill_step(self) -> int:
        """Prefill chunk size (tokens per forward) for the transformers lanes.

        A one-shot long prefill materializes an attention-score transient of
        [heads, L, L] float32 whenever torch's SDPA falls back to its math
        path (MPS does at long L — resolved implementation `sdpa`, verified
        by probe): at 30k on Qwen3-4B (32 q-heads) that is ONE 107.15 GiB
        MTLBuffer, which Metal refuses outright — measured twice as
        `Failed to allocate private MTLBuffer for size 115054126208`
        (= 32 x 29981^2 x 4 exactly). Chunked at 2048 the largest transient
        is [32, 2048, 30k] fp32 ≈ 7.9 GiB. Env override
        `ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP`; <= 0 disables chunking."""
        cached = getattr(self, "_transformers_prefill_step_cached", None)
        if isinstance(cached, int):
            return cached
        step = 2048
        try:
            raw = os.environ.get("ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP")
            if raw is not None and str(raw).strip():
                step = int(str(raw).strip())
        except Exception:
            step = 2048
        self._transformers_prefill_step_cached = step
        return step

    def _transformers_prefill_cache(self, state: _TransformersPromptCacheValue, token_ids: List[int]) -> bool:
        if not token_ids:
            return True
        if getattr(self, "model_instance", None) is None:
            return False
        try:
            import torch  # type: ignore
        except Exception:
            return False

        device = self._transformers_cache_device() or torch.device("cpu")
        use_mps_lock = str(device).startswith("mps") or str(getattr(self, "device", "") or "").strip().lower() == "mps"
        step = self._transformers_prefill_step()
        if step <= 0:
            step = len(token_ids)

        # Chunked: never materialize an [heads, L, L] score transient for the
        # whole prompt at once (see _transformers_prefill_step). Each chunk's
        # forward extends the SAME cache, so the resulting KV is identical.
        for start in range(0, len(token_ids), step):
            chunk = token_ids[start:start + step]
            past_len = len(state.prompt_tokens)
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            attention_mask = torch.ones((1, past_len + len(chunk)), dtype=torch.long, device=device)

            kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if state.cache is not None:
                kwargs["past_key_values"] = state.cache
            # Only the KV cache is kept from this forward — the logits are never
            # read. Without `logits_to_keep=1` a raw forward materializes
            # [1, seq, vocab] float32 logits: at 30k on Qwen3.5-4B (vocab 248320)
            # that is a ~27.8 GiB transient PER PREFILL, which the MPS allocator
            # then pools forever (2026-08-03 leak audit). `generate()` already
            # passes 1 on every model that supports it (generation/utils.py:2527);
            # this forward must match.
            if self._transformers_forward_supports_logits_to_keep():
                kwargs["logits_to_keep"] = 1

            try:
                with torch.inference_mode():
                    if use_mps_lock:
                        with _MPS_GENERATION_LOCK:
                            outputs = self.model_instance(**kwargs)
                    else:
                        outputs = self.model_instance(**kwargs)
            except Exception:
                return False

            new_cache = getattr(outputs, "past_key_values", None)
            if new_cache is not None:
                state.cache = new_cache
            state.prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens + tuple(chunk)))
        return True

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        if not self.supports_prompt_cache():
            return None

        model_type = getattr(self, "model_type", None)
        if model_type == "transformers":
            # Start with the provider-native cache type. For hybrid architectures such as Qwen3.5,
            # the first prefill call may be the only reliable way to obtain the right cache object.
            return _TransformersPromptCacheValue(cache=self._transformers_empty_native_cache())

        try:
            from llama_cpp.llama_cache import LlamaRAMCache
        except Exception:
            return None

        cap = getattr(self, "_gguf_prompt_cache_pending_capacity_bytes", None)
        cap = self._coerce_gguf_prompt_cache_capacity_bytes(
            cap if cap is not None else getattr(self, "_gguf_prompt_cache_default_capacity_bytes", None)
        )
        if cap > 0:
            cache_obj = LlamaRAMCache(capacity_bytes=int(cap))
            cap_effective = int(getattr(cache_obj, "capacity_bytes", cap) or cap)
        else:
            # Default is "auto": do not silently disable caching for large prompts.
            # When a single LlamaState exceeds a fixed capacity, llama-cpp-python evicts it
            # immediately (the cache stays empty). Auto-grow ensures at least the most recent
            # prefix KV snapshot can be retained for subsequent turns.
            global _AUTO_GROWING_LLAMA_RAM_CACHE_CLS
            cache_cls = _AUTO_GROWING_LLAMA_RAM_CACHE_CLS
            if cache_cls is None:
                class _AutoGrowingLlamaRAMCache(LlamaRAMCache):  # type: ignore[misc]
                    def __setitem__(self, key, value):  # type: ignore[override]
                        try:
                            state_size = int(getattr(value, "llama_state_size", 0) or 0)
                        except Exception:
                            state_size = 0
                        if state_size > 0:
                            try:
                                cap_now = int(getattr(self, "capacity_bytes", 0) or 0)
                            except Exception:
                                cap_now = 0
                            if state_size > cap_now:
                                # Grow just enough to retain this state (the base class eviction policy
                                # will still drop older entries as needed).
                                self.capacity_bytes = int(state_size)
                        return super().__setitem__(key, value)

                cache_cls = _AutoGrowingLlamaRAMCache
                _AUTO_GROWING_LLAMA_RAM_CACHE_CLS = cache_cls

            cache_obj = cache_cls(capacity_bytes=0)
            cap_effective = int(getattr(cache_obj, "capacity_bytes", 0) or 0)

        return _GGUFPromptCacheValue(
            cache=cache_obj,
            capacity_bytes=cap_effective,
        )

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        transformers_state = self._transformers_prompt_cache_state(cache_value)
        if transformers_state is not None:
            cloned_cache = self._transformers_clone_cache(transformers_state.cache)
            if transformers_state.cache is not None and cloned_cache is None:
                return None

            return _TransformersPromptCacheValue(
                cache=cloned_cache,
                prompt_tokens=tuple(int(tok) for tok in transformers_state.prompt_tokens),
                system_prompt_parts=copy.deepcopy(transformers_state.system_prompt_parts),
                messages=copy.deepcopy(transformers_state.messages),
                tools=copy.deepcopy(transformers_state.tools),
                add_generation_prompt=bool(transformers_state.add_generation_prompt),
            )

        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            return None
        cloned_cache = self._gguf_clone_llama_cache(state.cache, capacity_bytes=state.capacity_bytes)
        if cloned_cache is None:
            return None
        return _GGUFPromptCacheValue(
            cache=cloned_cache,
            capacity_bytes=int(state.capacity_bytes),
            system_prompt_parts=copy.deepcopy(state.system_prompt_parts),
            messages=copy.deepcopy(state.messages),
            tools=copy.deepcopy(state.tools),
            add_generation_prompt=bool(state.add_generation_prompt),
            prompt_text=str(state.prompt_text or ""),
            prompt_tokens=tuple(int(tok) for tok in state.prompt_tokens),
            fed_prompt_tokens=tuple(int(tok) for tok in state.fed_prompt_tokens),
        )

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
        hf_transformers_enable_thinking = kwargs.get("_acore_hf_transformers_enable_thinking")
        hf_transformers_enable_thinking = (
            hf_transformers_enable_thinking if isinstance(hf_transformers_enable_thinking, bool) else None
        )
        hf_transformers_reasoning_effort = kwargs.get("_acore_hf_transformers_reasoning_effort")
        hf_transformers_reasoning_effort = (
            hf_transformers_reasoning_effort if isinstance(hf_transformers_reasoning_effort, str) else None
        )
        gguf_enable_thinking = kwargs.get("_acore_gguf_enable_thinking")
        gguf_enable_thinking = gguf_enable_thinking if isinstance(gguf_enable_thinking, bool) else None

        # PLANNED BLOC CUT (bloc composability, see base.py — MLX has honoured
        # this since the bloc work landed; both HuggingFace lanes did not).
        #
        # WHAT THIS FIXES, MEASURED 2026-08-07 on Qwen3-4B-Instruct-2507 (bf16,
        # MPS), a 702-token system bloc + a 661-token tools bloc:
        #
        #   before  chain [system, tools] cold build ...... 2068 tokens prefilled
        #           (the system bloc is prefilled, then THROWN AWAY and the whole
        #            system+tools text re-prefilled, because `tools is not None`
        #            forced `needs_rebuild`)
        #           edit ONE tool description ............. 1367 tokens re-prefilled
        #           (the unchanged system bloc bought nothing)
        #
        # So the tools bloc cost 52% MORE to build than one merged bloc and saved
        # nothing on the edit it exists for. The plan was correct all along
        # (`boundaries=[702, 1363]`, `collapsed=False`) — this lane simply never
        # read it. Feeding the planned fragment verbatim is what makes the system
        # bloc's KV survive a tools change.
        planned = kwargs.get("bloc_token_ids")
        planned_ids: Optional[List[int]] = None
        if isinstance(planned, (list, tuple)):
            try:
                planned_ids = [int(tok) for tok in planned]
            except Exception:
                planned_ids = None
            if not planned_ids:
                planned_ids = None
        planned_text = kwargs.get("bloc_stable_text")
        planned_text = planned_text if isinstance(planned_text, str) else None

        transformers_state = self._transformers_prompt_cache_state(cache_value)
        if transformers_state is not None:
            prev_add_generation_prompt = bool(transformers_state.add_generation_prompt)
            prev_prompt_tokens = tuple(int(tok) for tok in (transformers_state.prompt_tokens or ()))

            # Mutate state first; we may rebuild or append depending on what changed.
            if system_prompt is not None:
                text = str(system_prompt or "").strip()
                if text:
                    transformers_state.system_prompt_parts.append(text)
            if tools is not None:
                transformers_state.tools = [copy.deepcopy(tool) for tool in tools if isinstance(tool, dict)] or None

            delta_messages: List[Dict[str, Any]] = []
            if isinstance(messages, list) and messages:
                for msg in messages:
                    if isinstance(msg, dict):
                        copied = copy.deepcopy(msg)
                        delta_messages.append(copied)
                        transformers_state.messages.append(copied)
            if isinstance(prompt, str) and prompt:
                user_msg = {"role": "user", "content": prompt}
                delta_messages.append(copy.deepcopy(user_msg))
                transformers_state.messages.append(user_msg)

            new_add_generation_prompt = bool(add_generation_prompt)
            transformers_state.add_generation_prompt = new_add_generation_prompt

            if planned_ids is not None:
                # The planner already rendered the WHOLE conversation through this
                # lane's own renderer and cut it at successor-independent token
                # boundaries. Feed the cut verbatim on top of the warm cache; the
                # logical state above is still updated so a later UNPLANNED append
                # (a live `generate`) re-renders the correct cumulative text, and
                # the generate path composes against `prompt_tokens` by LCP, so
                # the closing tag the plan deliberately holds back arrives with
                # the next turn's suffix.
                return self._transformers_prefill_cache(transformers_state, list(planned_ids))

            needs_rebuild = bool(system_prompt is not None or tools is not None or not prev_prompt_tokens)
            # Changing add_generation_prompt from True -> False is a structural edit; rebuild.
            if prev_add_generation_prompt and not new_add_generation_prompt:
                needs_rebuild = True
            # Appending after a generation prompt is ambiguous; rebuild for safety.
            if prev_add_generation_prompt and (delta_messages or system_prompt is not None or tools is not None):
                needs_rebuild = True

            if needs_rebuild:
                system_text = "\n\n".join(
                    part for part in transformers_state.system_prompt_parts if isinstance(part, str) and part
                )
                full_text = self._transformers_build_prompt_fragment(
                    prompt="",
                    messages=transformers_state.messages,
                    system_prompt=system_text or None,
                    tools=transformers_state.tools,
                    add_generation_prompt=transformers_state.add_generation_prompt,
                    enable_thinking=hf_transformers_enable_thinking,
                    reasoning_effort=hf_transformers_reasoning_effort,
                )

                token_ids = self._transformers_tokenize_fragment(full_text, add_bos_if_empty=True)
                transformers_state.cache = self._transformers_empty_native_cache()
                transformers_state.prompt_tokens = ()
                return self._transformers_prefill_cache(transformers_state, token_ids)

            # Incremental: append-only messages or toggling add_generation_prompt to True.
            delta_add_gen = bool(new_add_generation_prompt and not prev_add_generation_prompt)
            # NOTE: no reasoning_effort here — delta fragments never own the system
            # region, and the effort line belongs to the first system block only
            # (rendered by the full-rebuild branch above).
            delta_text = self._transformers_build_prompt_fragment(
                prompt="",
                messages=delta_messages,
                system_prompt=None,
                tools=None,
                add_generation_prompt=delta_add_gen,
                enable_thinking=hf_transformers_enable_thinking,
            )
            token_ids = self._transformers_tokenize_fragment(delta_text, add_bos_if_empty=False)
            return self._transformers_prefill_cache(transformers_state, token_ids)

        state = self._gguf_prompt_cache_state(cache_value)
        if state is None or getattr(self, "llm", None) is None:
            return False

        prev_add_generation_prompt = bool(state.add_generation_prompt)
        prev_prompt_text = str(state.prompt_text or "")
        prev_prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens or ()))

        if system_prompt is not None:
            text = str(system_prompt or "").strip()
            if text:
                state.system_prompt_parts.append(text)
        if tools is not None:
            state.tools = [copy.deepcopy(tool) for tool in tools if isinstance(tool, dict)] or None
        delta_messages: List[Dict[str, Any]] = []
        if isinstance(messages, list) and messages:
            for msg in messages:
                if isinstance(msg, dict):
                    copied = copy.deepcopy(msg)
                    delta_messages.append(copied)
                    state.messages.append(copied)
        if isinstance(prompt, str) and prompt:
            user_msg = {"role": "user", "content": prompt}
            delta_messages.append(copy.deepcopy(user_msg))
            state.messages.append(user_msg)
        new_add_generation_prompt = bool(add_generation_prompt)
        state.add_generation_prompt = new_add_generation_prompt

        if not self._gguf_prompt_cache_supports_local_control_plane():
            # Keyed-only GGUF caches still keep the in-process cache object, but they do not
            # advertise modular update/fork support to higher layers.
            return False

        if planned_ids is not None:
            # PLANNED BLOC CUT, GGUF lane. Same contract as the transformers
            # branch above: feed the planner's cut instead of re-rendering this
            # module, so the predecessor bloc's KV survives.
            #
            # `prompt_text` must stay EXACTLY the text of `prompt_tokens` —
            # `_gguf_compose_cached_prompt_tokens` concatenates it with a live
            # suffix to form the prompt actually sent, so a text/token pair that
            # disagree by even one token corrupts every warm turn after it.
            #
            # The planner's stable text is NOT that text in general: the seam
            # backoff drops the last token(s) the tokenizer could still merge, so
            # the text runs a token or two past the ids (measured on this lane:
            # `tokenize(stable_text)` = 283 against `boundary` = 282). llama.cpp's
            # own detokenizer IS the exact inverse — verified here by re-encoding,
            # never assumed — so the text is derived from the ids that were fed.
            cumulative_ids = tuple(int(tok) for tok in (prev_prompt_tokens + tuple(planned_ids)))
            verified_text: Optional[str] = None
            candidates: List[str] = []
            try:
                decoded = self.llm.detokenize(list(cumulative_ids), special=True)
                candidates.append(decoded.decode("utf-8", errors="strict"))
            except Exception:
                pass
            if planned_text is not None:
                candidates.append(planned_text)
            for candidate in candidates:
                try:
                    text_ids = tuple(int(tok) for tok in self._gguf_tokenize_rendered_prompt(candidate))
                except Exception:
                    continue
                if text_ids == cumulative_ids:
                    verified_text = candidate
                    break
            if verified_text is not None:
                with getattr(self, "_gguf_prompt_cache_lock", _MPS_GENERATION_LOCK):
                    ok = self._gguf_prefill_prompt_cache(state.cache, cumulative_ids)
                if not ok:
                    return False
                state.prompt_text = verified_text
                state.prompt_tokens = cumulative_ids
                return True
            message = (
                "#FALLBACK prompt-cache bloc: GGUF could not verify the planned fragment's text "
                "against its token ids; rebuilding this bloc from the cumulative render instead "
                "(correct, but the predecessor bloc's prefill is not reused)."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self.logger.warning(message)

        # Fast path: append-only updates (no system/tools changes) can reuse the serialized prompt.
        can_incremental = (
            system_prompt is None
            and tools is None
            and prev_prompt_tokens
            and (not prev_add_generation_prompt)
            and (not new_add_generation_prompt)
        )
        if can_incremental and delta_messages:
            # Reject non-append semantics: new system messages belong in the prefix, not the tail.
            has_system_delta = any(
                str(m.get("role") or "").strip().lower() == "system" for m in delta_messages if isinstance(m, dict)
            )
            if not has_system_delta:
                chat_format = self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
                if chat_format not in {"chatml", "chatml-function-calling", "llama-3"}:
                    delta_text = ""
                elif chat_format == "llama-3":
                    delta_text = self._gguf_render_llama3_prompt(
                        messages=delta_messages,
                        add_generation_prompt=False,
                        enable_thinking=gguf_enable_thinking,
                    )
                else:
                    delta_text = self._gguf_render_chatml_prompt(
                        messages=delta_messages,
                        add_generation_prompt=False,
                        enable_thinking=gguf_enable_thinking,
                    )

                try:
                    delta_tokens = (
                        tuple(
                            int(tok)
                            for tok in self.llm.tokenize(
                                delta_text.encode("utf-8"),
                                add_bos=False,
                                special=True,
                            )
                        )
                        if delta_text
                        else ()
                    )
                except Exception:
                    delta_tokens = ()

                if delta_tokens:
                    prompt_tokens = tuple(int(tok) for tok in (prev_prompt_tokens + delta_tokens))
                    with getattr(self, "_gguf_prompt_cache_lock", _MPS_GENERATION_LOCK):
                        ok = self._gguf_prefill_prompt_cache(state.cache, prompt_tokens)
                    if not ok:
                        return False
                    state.prompt_text = prev_prompt_text + delta_text
                    state.prompt_tokens = prompt_tokens
                    return True

        system_text = "\n\n".join(part for part in state.system_prompt_parts if isinstance(part, str) and part)
        chat_messages = self._gguf_build_chat_messages(
            system_prompt=system_text or None,
            messages=state.messages,
            tools=state.tools,
            user_message_content=None,
        )
        prompt_text, prompt_tokens = self._gguf_render_prompt_tokens(
            messages=chat_messages,
            add_generation_prompt=state.add_generation_prompt,
            enable_thinking=gguf_enable_thinking,
        )

        with getattr(self, "_gguf_prompt_cache_lock", _MPS_GENERATION_LOCK):
            ok = self._gguf_prefill_prompt_cache(state.cache, prompt_tokens)
        if not ok:
            return False

        state.prompt_text = prompt_text
        state.prompt_tokens = tuple(int(tok) for tok in prompt_tokens)
        return True

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        transformers_state = self._transformers_prompt_cache_state(cache_value)
        if transformers_state is not None:
            if transformers_state.prompt_tokens:
                return len(transformers_state.prompt_tokens)
            cache_obj = transformers_state.cache
            try:
                tok = cache_obj.get_seq_length() if cache_obj is not None else None
            except Exception:
                tok = None
            if isinstance(tok, int) and tok >= 0:
                return tok
            return None

        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            return None
        longest = self._gguf_prompt_cache_longest_prefix_tokens(state.cache)
        if state.prompt_tokens:
            return max(len(state.prompt_tokens), len(longest))
        return len(longest)

    def prompt_cache_set(
        self,
        key: str,
        *,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        capacity_bytes: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Create/reset a prompt cache for the given key (best-effort)."""
        _ = kwargs
        if not self.supports_prompt_cache():
            return False

        if getattr(self, "model_type", None) != "gguf":
            ok = super().prompt_cache_set(key, make_default=make_default)
            if not ok:
                return False
            normalized = self._normalize_prompt_cache_key(key)
            if normalized is None:
                return False
            cache_value = self._prompt_cache_store.get(normalized)
            state = self._transformers_prompt_cache_state(cache_value)
            if state is None:
                return False
            try:
                self._prompt_cache_store.set(
                    normalized,
                    state,
                    ttl_s=ttl_s,
                    meta={"backend": "transformers"},
                )
            except Exception:
                return False
            return True

        self._gguf_prompt_cache_pending_capacity_bytes = self._coerce_gguf_prompt_cache_capacity_bytes(capacity_bytes)
        try:
            ok = super().prompt_cache_set(key, make_default=make_default)
        finally:
            self._gguf_prompt_cache_pending_capacity_bytes = None
        if not ok:
            return False

        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            return False
        cache_value = self._prompt_cache_store.get(normalized)
        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            return False
        try:
            self._prompt_cache_store.set(
                normalized,
                state,
                ttl_s=ttl_s,
                meta={
                    "backend": "llama_cpp",
                    "capacity_bytes": int(getattr(state.cache, "capacity_bytes", state.capacity_bytes) or state.capacity_bytes),
                },
            )
        except Exception:
            return False

        try:
            if getattr(self, "llm", None) is not None and hasattr(self.llm, "set_cache"):
                self.llm.set_cache(state.cache)
        except Exception:
            pass

        return True

    def prompt_cache_clear(self, key: Optional[str] = None) -> bool:
        """Clear prompt caches AND their hybrid snapshots (a snapshot must not
        outlive the key it mirrors); on the transformers lane also return the
        freed KV tensors' device memory to the OS (release point, not hot
        path)."""
        cleared = super().prompt_cache_clear(key)
        self._ensure_transformers_snapshot_state()
        with self._transformers_snapshot_lock:
            if key is None:
                self._transformers_snapshots.clear()
            else:
                norm = self._normalize_prompt_cache_key(key)
                if norm:
                    self._transformers_snapshots.pop(norm, None)
        if getattr(self, "model_type", None) == "transformers":
            self._transformers_release_device_pool()
        llm = getattr(self, "llm", None)
        try:
            if llm is not None and hasattr(llm, "set_cache"):
                llm.set_cache(None)
        except Exception:
            pass
        # llama.cpp can still reuse in-process KV state via prefix matching even when no cache
        # object is configured. When clearing *all* caches, reset the runtime context as well so
        # "cache cleared" is observable in long-running processes (CLI/REPL).
        if key is None and str(getattr(self, "model_type", "") or "").strip().lower() == "gguf":
            try:
                if llm is not None and hasattr(llm, "reset"):
                    llm.reset()
            except Exception:
                pass
        return cleared

    def prompt_cache_save(self, key: str, filename: str, **kwargs: Any) -> Dict[str, Any]:
        """Save a GGUF llama.cpp prompt cache snapshot to disk (best-effort).

        This persists the provider-side cache metadata plus the *single* longest-prefix llama.cpp
        state snapshot for the key. It is sufficient to warm-start large chats without rebuilding
        the prefix, but it does not attempt to persist every intermediate prefix in the RAM cache.
        """
        incoming_meta = kwargs.get("meta") if isinstance(kwargs.get("meta"), dict) else {}

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

        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        if getattr(self, "model_type", None) != "gguf":
            try:
                import torch  # type: ignore
                from safetensors.torch import save_file  # type: ignore
                from transformers.cache_utils import DynamicCache  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Transformers prompt cache saving requires `torch`, `transformers`, and `safetensors`."
                ) from e

            normalized = self._normalize_prompt_cache_key(key)
            if normalized is None:
                raise ValueError("prompt cache key must be a non-empty string")

            cache_value = self._prompt_cache_store.get(normalized)
            state = self._transformers_prompt_cache_state(cache_value)
            if state is None:
                raise ValueError(f"prompt cache key '{normalized}' does not exist")
            tensors: Dict[str, torch.Tensor] = {}
            prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens or ()))
            tensors["prompt_tokens"] = torch.tensor(prompt_tokens, dtype=torch.int32, device="cpu")

            cache = state.cache
            if cache is None:
                raise ValueError("prompt cache key does not reference a concrete transformers cache object")
            cache_meta = self._transformers_cache_class_meta(cache)
            cache_schema = ""
            list_lengths: Dict[str, int] = {}
            json_attrs: Dict[str, Any] = {}

            if isinstance(cache, DynamicCache) and getattr(cache, "layers", None) is not None:
                cache_schema = "dynamic-cache-layers/v1"
                layers = getattr(cache, "layers", []) or []
                prompt_len = len(prompt_tokens)
                json_attrs["cache_layer_classes"] = [
                    f"{layer.__class__.__module__}.{layer.__class__.__name__}"
                    for layer in layers
                ]
                layer_lengths = self._transformers_dynamic_layer_lengths(cache, prompt_len)
                if layer_lengths:
                    json_attrs["cache_layer_sequence_lengths"] = layer_lengths
                json_attrs["cache_position_strategy"] = self._transformers_cache_position_strategy()
                for idx, layer in enumerate(layers):
                    if not bool(getattr(layer, "is_initialized", False)):
                        continue
                    keys = getattr(layer, "keys", None)
                    values = getattr(layer, "values", None)
                    if keys is None or values is None:
                        continue
                    tensors[f"layer_{idx}_keys"] = keys.detach().to("cpu").contiguous()
                    tensors[f"layer_{idx}_values"] = values.detach().to("cpu").contiguous()
            elif self._transformers_cache_has_serializable_tensor_state(cache):
                cache_schema = "tensor-list-cache/v1"
                for attr in _TRANSFORMERS_TENSOR_LIST_CACHE_ATTRS:
                    values = getattr(cache, attr, None)
                    if not isinstance(values, list):
                        if isinstance(values, torch.Tensor):
                            tensors[f"cache__{attr}"] = values.detach().to("cpu").contiguous()
                            list_lengths[attr] = -1
                        continue
                    list_lengths[attr] = len(values)
                    for idx, value in enumerate(values):
                        if value is None or not isinstance(value, torch.Tensor):
                            continue
                        tensors[f"cache__{attr}__{idx}"] = value.detach().to("cpu").contiguous()
                for attr in _TRANSFORMERS_JSON_CACHE_ATTRS:
                    value = getattr(cache, attr, None)
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        json_attrs[attr] = value
                    elif isinstance(value, (list, tuple)):
                        json_attrs[attr] = list(value)
            else:
                raise ValueError(
                    f"prompt cache object type {cache.__class__.__module__}.{cache.__class__.__name__} "
                    "is not serializable by the transformers prompt cache artifact writer"
                )

            meta: Dict[str, str] = {
                "format": "abstractcore-transformers-prompt-cache/v1",
                "provider": str(getattr(self, "provider", "huggingface")),
                "model": str(getattr(self, "model", "")),
                "saved_at": datetime.now().isoformat(),
                "token_count": str(len(prompt_tokens)),
                "cache_implementation": "dynamic",
                "cache_schema": cache_schema,
                "cache_class": cache_meta.get("cache_class", ""),
                "cache_module": cache_meta.get("cache_module", ""),
            }
            if list_lengths:
                meta["cache_list_lengths"] = json.dumps(list_lengths, ensure_ascii=False, separators=(",", ":"))
            if json_attrs:
                meta["cache_json_attrs"] = json.dumps(json_attrs, ensure_ascii=False, separators=(",", ":"))
            for mk, mv in dict(incoming_meta or {}).items():
                if isinstance(mk, str) and mk:
                    meta[mk] = _meta_value(mv)

            save_file(tensors, str(filename), metadata=meta)

            return {
                "supported": True,
                "operation": "save",
                "provider": str(getattr(self, "provider", "huggingface")),
                "model": str(getattr(self, "model", "")),
                "key": normalized,
                "filename": str(filename),
                "meta": meta,
            }

        try:
            import numpy as np
            from llama_cpp.llama import LlamaState
        except Exception as e:
            raise ImportError("GGUF prompt cache saving requires `llama-cpp-python` and `numpy`.") from e

        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            raise ValueError("prompt cache key must be a non-empty string")

        cache_value = self._prompt_cache_store.get(normalized)
        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            raise ValueError(f"prompt cache key '{normalized}' does not exist")

        cache_obj = self._gguf_prompt_cache_unwrap(state)
        if cache_obj is None:
            raise ValueError("prompt cache key does not reference a llama.cpp cache object")

        prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens or self._gguf_prompt_cache_longest_prefix_tokens(cache_obj)))
        if not prompt_tokens:
            raise ValueError("prompt cache has no stored prefix tokens to save")

        # Ensure a concrete state exists for the stored prefix tokens.
        state_map = getattr(cache_obj, "cache_state", None)
        llama_state = None
        if hasattr(state_map, "get"):
            llama_state = state_map.get(prompt_tokens)
        if llama_state is None:
            with getattr(self, "_gguf_prompt_cache_lock", _MPS_GENERATION_LOCK):
                if not self._gguf_prefill_prompt_cache(cache_obj, prompt_tokens):
                    raise RuntimeError("failed to prefill prompt cache prior to saving")
            state_map = getattr(cache_obj, "cache_state", None)
            if hasattr(state_map, "get"):
                llama_state = state_map.get(prompt_tokens)

        if not isinstance(llama_state, LlamaState):
            raise RuntimeError("could not retrieve a llama.cpp state snapshot for the prompt cache key")

        exported_state = self._gguf_prompt_cache_export_state(state)
        meta: Dict[str, Any] = {
            "format": "abstractcore-gguf-prompt-cache/v1",
            "provider": str(getattr(self, "provider", "huggingface")),
            "model": str(getattr(self, "model", "")),
            "saved_at": datetime.now().isoformat(),
            "cache_state": exported_state,
        }
        meta.update(dict(incoming_meta or {}))

        with open(str(filename), "wb") as f:
            np.savez_compressed(
                f,
                meta_json=np.array(json.dumps(meta, ensure_ascii=False), dtype=np.str_),
                prompt_tokens=np.asarray(prompt_tokens, dtype=np.intc),
                input_ids=np.asarray(getattr(llama_state, "input_ids"), dtype=np.intc),
                scores=np.asarray(getattr(llama_state, "scores"), dtype=np.single),
                n_tokens=np.asarray(int(getattr(llama_state, "n_tokens", 0) or 0), dtype=np.int64),
                llama_state=np.frombuffer(bytes(getattr(llama_state, "llama_state", b"")), dtype=np.uint8),
                llama_state_size=np.asarray(int(getattr(llama_state, "llama_state_size", 0) or 0), dtype=np.int64),
                seed=np.asarray(int(getattr(llama_state, "seed", 0) or 0), dtype=np.int64),
            )

        return {
            "supported": True,
            "operation": "save",
            "provider": str(getattr(self, "provider", "huggingface")),
            "model": str(getattr(self, "model", "")),
            "key": normalized,
            "filename": str(filename),
            "meta": meta,
        }

    def prompt_cache_load(
        self,
        filename: str,
        *,
        key: Optional[str] = None,
        make_default: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load a GGUF llama.cpp prompt cache snapshot from disk (best-effort)."""
        _ = kwargs
        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        if getattr(self, "model_type", None) != "gguf":
            try:
                import torch  # type: ignore
                from safetensors import safe_open  # type: ignore
                from safetensors.torch import load_file  # type: ignore
                from transformers.cache_utils import DynamicCache, DynamicLayer  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Transformers prompt cache loading requires `torch`, `transformers`, and `safetensors`."
                ) from e

            device_str = self._transformers_cache_device_str()
            meta: Dict[str, Any] = {}
            try:
                with safe_open(str(filename), framework="pt", device="cpu") as f:
                    raw_meta = f.metadata() or {}
                    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
            except Exception:
                meta = {}

            fmt = meta.get("format")
            accepted_formats = {
                "abstractcore-transformers-prompt-cache/v1",
                "abstractcore-hf-transformers-prompt-cache/v1",
            }
            if fmt and str(fmt) not in accepted_formats:
                raise ValueError(f"Unsupported transformers prompt cache format: {fmt}")

            required_model = meta.get("model")
            current_model = str(getattr(self, "model", "") or "")
            if isinstance(required_model, str) and required_model.strip() and required_model.strip() != current_model:
                raise ValueError(
                    f"Prompt cache model mismatch: cache expects '{required_model.strip()}', current model is '{current_model}'."
                )

            tensors = load_file(str(filename), device=device_str)
            prompt_tok_tensor = tensors.get("prompt_tokens")
            if prompt_tok_tensor is None:
                raise ValueError("Invalid transformers prompt cache file (missing prompt_tokens)")
            prompt_tokens = tuple(int(tok) for tok in prompt_tok_tensor.to("cpu").tolist())

            cache_schema = str(meta.get("cache_schema") or "").strip()
            json_attrs = self._decode_transformers_cache_json_attrs(meta)
            if cache_schema == "tensor-list-cache/v1":
                cache = self._transformers_instantiate_cache_from_meta(meta)
                if cache is None:
                    raise ValueError("Could not instantiate transformers cache object from artifact metadata.")

                try:
                    list_lengths_raw = json.loads(str(meta.get("cache_list_lengths") or "{}"))
                    list_lengths = list_lengths_raw if isinstance(list_lengths_raw, dict) else {}
                except Exception:
                    list_lengths = {}
                for attr in _TRANSFORMERS_TENSOR_LIST_CACHE_ATTRS:
                    raw_len = list_lengths.get(attr)
                    try:
                        length = int(raw_len)
                    except Exception:
                        current = getattr(cache, attr, None)
                        length = len(current) if isinstance(current, list) else 0
                    if length == -1:
                        tensor = tensors.get(f"cache__{attr}")
                        if tensor is not None:
                            setattr(cache, attr, tensor)
                        continue
                    values: List[Any] = [None for _ in range(max(0, length))]
                    for idx in range(len(values)):
                        tensor = tensors.get(f"cache__{attr}__{idx}")
                        if tensor is not None:
                            values[idx] = tensor
                    if values or hasattr(cache, attr):
                        setattr(cache, attr, values)

                for attr in _TRANSFORMERS_JSON_CACHE_ATTRS:
                    if attr in json_attrs and hasattr(cache, attr):
                        try:
                            setattr(cache, attr, json_attrs[attr])
                        except Exception:
                            pass
            elif cache_schema in {"", "dynamic-cache-layers/v1"}:
                layer_indices: List[int] = []
                for name in tensors.keys():
                    if name.startswith("layer_") and name.endswith("_keys"):
                        try:
                            layer_indices.append(int(name.split("_", 2)[1]))
                        except Exception:
                            continue
                max_idx = max(layer_indices) if layer_indices else -1

                cache = self._transformers_instantiate_cache_from_meta(meta)
                if cache is None or not isinstance(cache, DynamicCache):
                    model = getattr(self, "model_instance", None)
                    config = getattr(model, "config", None)
                    try:
                        cache = DynamicCache(config=config)
                    except Exception:
                        cache = DynamicCache()
                if max_idx >= 0:
                    layers = getattr(cache, "layers", None)
                    if not isinstance(layers, list):
                        cache.layers = []
                        layers = cache.layers
                    while len(layers) <= max_idx:
                        layers.append(DynamicLayer())
                    for idx in range(max_idx + 1):
                        keys = tensors.get(f"layer_{idx}_keys")
                        values = tensors.get(f"layer_{idx}_values")
                        if keys is None or values is None:
                            continue
                        layer = layers[idx]
                        layer.keys = keys
                        layer.values = values
                        layer.is_initialized = True
                    layer_lengths = json_attrs.get("cache_layer_sequence_lengths")
                    self._restore_transformers_dynamic_layer_lengths(
                        cache,
                        prompt_len=len(prompt_tokens),
                        layer_lengths=layer_lengths if isinstance(layer_lengths, list) else None,
                    )
            else:
                raise ValueError(f"Unsupported transformers prompt cache schema: {cache_schema}")

            imported_state = _TransformersPromptCacheValue(cache=cache, prompt_tokens=prompt_tokens)

            normalized = self._normalize_prompt_cache_key(key) if key is not None else None
            if normalized is None:
                normalized = f"cache:{uuid.uuid4().hex[:12]}"

            store_meta: Dict[str, Any] = {
                "backend": "transformers",
                "loaded_from": str(filename),
            }
            store_meta.update(meta)
            try:
                store_meta.setdefault("token_count", len(prompt_tokens))
            except Exception:
                pass

            self._prompt_cache_store.set(normalized, imported_state, meta=store_meta)
            if make_default:
                self._default_prompt_cache_key = normalized

            return {
                "supported": True,
                "operation": "load",
                "provider": str(getattr(self, "provider", "huggingface")),
                "model": str(getattr(self, "model", "")),
                "key": normalized,
                "filename": str(filename),
                "meta": store_meta,
            }

        try:
            import numpy as np
            from llama_cpp.llama_cache import LlamaRAMCache
            from llama_cpp.llama import LlamaState
        except Exception as e:
            raise ImportError("GGUF prompt cache loading requires `llama-cpp-python` and `numpy`.") from e

        with np.load(str(filename), allow_pickle=False) as data:
            meta_json = data.get("meta_json")
            meta_raw = str(meta_json.tolist()) if meta_json is not None else ""
            try:
                meta: Dict[str, Any] = json.loads(meta_raw) if meta_raw else {}
            except Exception:
                meta = {}

            fmt = meta.get("format")
            accepted_formats = {
                "abstractcore-gguf-prompt-cache/v1",
                "abstractcore-hf-gguf-prompt-cache/v1",
            }
            if fmt and str(fmt) not in accepted_formats:
                raise ValueError(f"Unsupported GGUF prompt cache format: {fmt}")

            required_model = meta.get("model") if isinstance(meta, dict) else None
            current_model = str(getattr(self, "model", "") or "")
            if isinstance(required_model, str) and required_model.strip() and required_model.strip() != current_model:
                raise ValueError(
                    f"Prompt cache model mismatch: cache expects '{required_model.strip()}', current model is '{current_model}'."
                )

            prompt_tokens_arr = data.get("prompt_tokens")
            if prompt_tokens_arr is None:
                raise ValueError("Invalid GGUF prompt cache file (missing prompt_tokens)")
            prompt_tokens = tuple(int(tok) for tok in prompt_tokens_arr.tolist())

            input_ids = data.get("input_ids")
            scores = data.get("scores")
            n_tokens = int(data.get("n_tokens").tolist()) if data.get("n_tokens") is not None else 0
            llama_state_u8 = data.get("llama_state")
            llama_state_size = int(data.get("llama_state_size").tolist()) if data.get("llama_state_size") is not None else 0
            seed = int(data.get("seed").tolist()) if data.get("seed") is not None else 0

        if not prompt_tokens or input_ids is None or scores is None or llama_state_u8 is None:
            raise ValueError("Invalid GGUF prompt cache file (missing required arrays)")

        llama_state = LlamaState(
            input_ids=np.asarray(input_ids, dtype=np.intc).copy(),
            scores=np.asarray(scores, dtype=np.single).copy(),
            n_tokens=int(n_tokens),
            llama_state=bytes(np.asarray(llama_state_u8, dtype=np.uint8).tobytes()),
            llama_state_size=int(llama_state_size),
            seed=int(seed),
        )

        cache_state = meta.get("cache_state") if isinstance(meta, dict) else None
        cap = None
        if isinstance(cache_state, dict):
            cap = cache_state.get("capacity_bytes")
        cap_i = self._coerce_gguf_prompt_cache_capacity_bytes(cap)
        try:
            state_size_i = int(getattr(llama_state, "llama_state_size", 0) or 0)
        except Exception:
            state_size_i = 0

        # If the saved state is larger than the declared capacity, fall back to the auto-growing
        # cache so we don't evict the single snapshot during load.
        if cap_i > 0 and state_size_i > 0 and state_size_i <= cap_i:
            cache_obj = LlamaRAMCache(capacity_bytes=int(cap_i))
        else:
            global _AUTO_GROWING_LLAMA_RAM_CACHE_CLS
            cache_cls = _AUTO_GROWING_LLAMA_RAM_CACHE_CLS
            if cache_cls is None:
                class _AutoGrowingLlamaRAMCache(LlamaRAMCache):  # type: ignore[misc]
                    def __setitem__(self, key, value):  # type: ignore[override]
                        try:
                            state_size = int(getattr(value, "llama_state_size", 0) or 0)
                        except Exception:
                            state_size = 0
                        if state_size > 0:
                            try:
                                cap_now = int(getattr(self, "capacity_bytes", 0) or 0)
                            except Exception:
                                cap_now = 0
                            if state_size > cap_now:
                                self.capacity_bytes = int(state_size)
                        return super().__setitem__(key, value)

                cache_cls = _AutoGrowingLlamaRAMCache
                _AUTO_GROWING_LLAMA_RAM_CACHE_CLS = cache_cls
            cache_obj = cache_cls(capacity_bytes=0)
        cache_obj[prompt_tokens] = llama_state

        imported_state = self._gguf_prompt_cache_import_state(cache_obj, cache_state if isinstance(cache_state, dict) else None)
        # Ensure the loaded state knows the saved prompt_tokens.
        imported_state.prompt_tokens = prompt_tokens

        normalized = self._normalize_prompt_cache_key(key) if key is not None else None
        if normalized is None:
            normalized = f"cache:{uuid.uuid4().hex[:12]}"

        store_meta: Dict[str, Any] = {"backend": "llama_cpp", "loaded_from": str(filename)}
        if isinstance(meta, dict):
            store_meta.update({k: v for k, v in meta.items() if k != "cache_state"})
        try:
            store_meta.setdefault("token_count", len(prompt_tokens))
        except Exception:
            pass

        self._prompt_cache_store.set(normalized, imported_state, meta=store_meta)
        if make_default:
            self._default_prompt_cache_key = normalized

        try:
            if getattr(self, "llm", None) is not None and hasattr(self.llm, "set_cache"):
                self.llm.set_cache(cache_obj)
        except Exception:
            pass

        return {
            "supported": True,
            "operation": "load",
            "provider": str(getattr(self, "provider", "huggingface")),
            "model": str(getattr(self, "model", "")),
            "key": normalized,
            "filename": str(filename),
            "meta": store_meta,
        }

    def _is_gguf_model(self, model: str) -> bool:
        """Detect if the model is a GGUF model"""
        # Check if it's a .gguf file path
        if model.endswith('.gguf'):
            return True

        # Local filesystem path (a .gguf FILE, or a DIRECTORY containing .gguf
        # files). A user pointing --model at an on-disk model — the LM Studio
        # layout ~/.lmstudio/models/org/Model[/ | :quant] is the common one —
        # must be recognized as GGUF even when the name carries no "gguf" token.
        # A trailing ":quant" selector is stripped before the path test.
        try:
            head = model.split(":", 1)[0] if ":" in model else model
            p = Path(head).expanduser()
            if p.is_file() and p.suffix.lower() == '.gguf':
                return True
            if p.is_dir() and any(p.glob("*.gguf")):
                return True
        except (OSError, ValueError):
            pass

        # Check if it's a HF repo with GGUF in the name (various formats)
        model_lower = model.lower()
        if 'gguf' in model_lower:
            # Handle formats like:
            # - "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
            # - "unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF" (cache format)
            # - "repo/model-GGUF"
            return True

        return False

    def _is_vision_model(self, model: str) -> bool:
        """Detect if the model is a vision model that requires special handling"""
        model_lower = model.lower()

        # Known vision models that require AutoModelForImageTextToText
        vision_models = [
            'glyph',           # zai-org/Glyph
            'glm-4.1v',        # GLM-4.1V variants
            'glm4v',           # GLM4V architecture
            'qwen-vl',         # Qwen-VL models
            'qwen2-vl',        # Qwen2-VL models
            'qwen2.5-vl',      # Qwen2.5-VL models
            'llava',           # LLaVA models
            'instructblip',    # InstructBLIP models
            'blip2',           # BLIP2 models
            'flamingo',        # Flamingo models
        ]

        return any(vision_keyword in model_lower for vision_keyword in vision_models)

    def _setup_device_transformers(self):
        """Setup device for transformers models (best-effort).

        We validate explicit device requests even when Transformers isn't available,
        since Torch availability (MPS/CUDA) may still matter for downstream behavior.
        """
        try:
            import torch  # type: ignore
        except Exception:
            self.device = "cpu"
            return

        requested = str(self.device or "").strip().lower() if isinstance(self.device, str) else ""
        if requested and requested != "auto":
            # Respect explicit user/env request, but fall back safely if unavailable.
            if requested == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and not torch.backends.mps.is_available():
                    self.logger.warning(
                        "HuggingFaceProvider requested device=mps but MPS is not available. "
                        "This usually means the process cannot see Metal devices (sandboxed execution). "
                        "Falling back to CPU. To silence this, set ABSTRACTCORE_HF_DEVICE=cpu."
                    )
                    self.device = "cpu"
                else:
                    self.device = "mps"
                    # Enable MPS fallback for unsupported ops (notably some vision pipelines).
                    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            elif requested == "cuda":
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.logger.warning(
                        "HuggingFaceProvider requested device=cuda but CUDA is not available; falling back to CPU."
                    )
                    self.device = "cpu"
            else:
                self.device = "cpu"
            return

        if not TRANSFORMERS_AVAILABLE:
            # Without transformers, default to CPU for safety.
            self.device = "cpu"
            return

        # Auto device selection.
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Apple Silicon: MPS built but unavailable is usually a sandbox / Metal visibility issue.
        try:
            import platform

            if (
                self.device == "cpu"
                and platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and not torch.backends.mps.is_available()
            ):
                self.logger.warning(
                    "PyTorch was built with MPS support, but MPS is not available. "
                    "This often indicates the process cannot access Metal devices (sandboxed execution). "
                    "Run outside the sandbox or force CPU via ABSTRACTCORE_HF_DEVICE=cpu."
                )
        except Exception:
            pass

    def _setup_device_gguf(self):
        """Setup device for GGUF models"""
        if self.n_gpu_layers is not None:
            return

        requested = str(self.device or "").strip().lower() if isinstance(self.device, str) else ""
        if requested == "cpu":
            self.n_gpu_layers = 0
            return

        is_metal_platform = platform.system().lower() == "darwin" and platform.machine().lower() == "arm64"
        wants_metal_request = requested == "mps" or (not requested and is_metal_platform)

        # Safety guard: on macOS, importing PyTorch/transformers in-process can hard-crash
        # llama.cpp when using Metal offload. Avoid SIGABRT by forcing CPU in that scenario,
        # unless the user explicitly opts into the unsafe path.
        llama_cpp_preimported_for_metal = False
        if "llama_cpp" in sys.modules:
            try:
                import llama_cpp  # type: ignore

                llama_cpp_preimported_for_metal = bool(
                    getattr(llama_cpp, "__abstractcore_preimported_for_metal", False)
                )
            except Exception:
                llama_cpp_preimported_for_metal = False

        if (
            wants_metal_request
            and os.environ.get("ABSTRACTCORE_GGUF_METAL_UNSAFE", "").strip().lower() not in {"1", "true", "yes"}
            and ("torch" in sys.modules or "transformers" in sys.modules)
            and not llama_cpp_preimported_for_metal
        ):
            import warnings

            poisoner = "PyTorch" if "torch" in sys.modules else "transformers"
            detail = (
                f"GGUF Metal offload disabled because {poisoner} is already imported in this "
                f"process (llama-cpp-python Metal offload can SIGABRT). EVERY layer of "
                f"{self.model} will run on CPU, which is typically 5-20x slower than Metal "
                f"and changes memory behaviour. "
                f"To get Metal: build a provider (any `create_llm`) BEFORE importing "
                f"torch/transformers/mlx_lm in this process — abstractcore pre-imports "
                f"llama.cpp for Metal at that point and marks it safe — or run the GGUF "
                f"work in a fresh process, or set ABSTRACTCORE_GGUF_METAL_UNSAFE=1 to "
                f"force Metal offload."
            )
            # The previous wording said "ensure `llama_cpp` is imported before PyTorch",
            # which does not work and was measured not working: this gate keys on
            # `llama_cpp.__abstractcore_preimported_for_metal`, a marker set ONLY by
            # `registry._preimport_llama_cpp_for_macos()`. A caller who imports
            # llama_cpp by hand, in any order, still lands here. A remedy that cannot
            # work is worse than no remedy — it sends the user away satisfied.
            # A bare `warnings.warn` is not enough signal for a whole-model CPU
            # downgrade: warnings are deduplicated per location, are filtered out
            # entirely under `-W ignore`, and leave nothing on the object for
            # telemetry to record. `import mlx_lm` and `abstractcore.embeddings` both
            # pull in transformers, so an ordinary process reaches this branch without
            # the caller ever naming torch — and then silently benchmarks CPU numbers
            # as if they were Metal. Emit it the way every other degradation in this
            # provider is emitted, and record it where a caller can read it back.
            self.logger.warning(f"#FALLBACK {detail}")
            self.gguf_metal_disabled_reason = detail
            warnings.warn(detail, RuntimeWarning, stacklevel=3)
            self.n_gpu_layers = 0
            return

        # Prefer GPU offload when available. Use llama.cpp's own capability probe so we
        # don't need to import PyTorch.
        supports_gpu_offload = False
        try:
            import llama_cpp  # type: ignore

            probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
            supports_gpu_offload = bool(probe() if callable(probe) else probe)
        except Exception:
            supports_gpu_offload = False

        wants_metal = requested == "mps" or (not requested and is_metal_platform and supports_gpu_offload)
        wants_cuda = requested == "cuda" or (not requested and not is_metal_platform and supports_gpu_offload)

        self.n_gpu_layers = int(-1 if (wants_metal or wants_cuda) else 0)

    def _transformers_config_kwargs(self) -> Dict[str, Any]:
        kwargs = {k: v for k, v in self.transformers_kwargs.items() if k in ["trust_remote_code"]}
        if _config.should_force_local_files_only():
            kwargs["local_files_only"] = True
        return kwargs

    @staticmethod
    def _build_transformers_quantization_config(kwargs: Dict[str, Any]) -> Any:
        """Caller-requested transformers quantization, as a config object.

        Two accepted spellings, both first-class:

          create_llm("huggingface", model=..., quantization_config=BitsAndBytesConfig(...))
          create_llm("huggingface", model=..., load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)

        The second is the legacy bitsandbytes spelling. transformers 5.9 REMOVED
        `load_in_4bit`/`load_in_8bit` from `from_pretrained` (modeling_utils
        accepts `quantization_config` only), so passing them through verbatim
        raises. They are translated here instead of being dropped, so callers
        written against older transformers keep working.

        Returns None when the caller asked for no quantization — in which case
        nothing about the load changes.
        """
        explicit = kwargs.get("quantization_config")
        if explicit is not None:
            return explicit

        want_4bit = bool(kwargs.get("load_in_4bit"))
        want_8bit = bool(kwargs.get("load_in_8bit"))
        if not (want_4bit or want_8bit):
            return None
        if want_4bit and want_8bit:
            raise ValueError(
                "load_in_4bit and load_in_8bit are mutually exclusive; pass one, "
                "or build a transformers BitsAndBytesConfig and pass it as "
                "quantization_config=."
            )

        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception as exc:  # pragma: no cover - transformers is a hard dep here
            raise ImportError(
                "load_in_4bit/load_in_8bit need transformers' BitsAndBytesConfig, "
                f"which could not be imported: {exc}"
            ) from exc
        if not _module_available("bitsandbytes"):
            raise ImportError(
                "load_in_4bit/load_in_8bit require the `bitsandbytes` package, which "
                "is not installed. Install it for this platform, or use an "
                "unquantized Transformers model, an MLX model with the MLX provider, "
                "or a GGUF model with the GGUF path."
            )

        cfg: Dict[str, Any] = {"load_in_4bit": want_4bit, "load_in_8bit": want_8bit}
        for name in ("bnb_4bit_quant_type", "bnb_4bit_use_double_quant",
                     "bnb_4bit_quant_storage", "llm_int8_threshold",
                     "llm_int8_skip_modules", "llm_int8_enable_fp32_cpu_offload",
                     "llm_int8_has_fp16_weight"):
            if name in kwargs:
                cfg[name] = kwargs[name]
        compute_dtype = kwargs.get("bnb_4bit_compute_dtype")
        if isinstance(compute_dtype, str):
            import torch  # type: ignore
            resolved = getattr(torch, compute_dtype, None)
            if not isinstance(resolved, torch.dtype):
                raise ValueError(
                    f"bnb_4bit_compute_dtype={compute_dtype!r} is not a torch dtype name")
            compute_dtype = resolved
        if compute_dtype is not None:
            cfg["bnb_4bit_compute_dtype"] = compute_dtype
        return BitsAndBytesConfig(**cfg)

    @staticmethod
    def _extract_quantization_config(config: Any) -> Dict[str, Any]:
        quantization_config = getattr(config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            return dict(quantization_config)
        if quantization_config is not None and hasattr(quantization_config, "to_dict"):
            try:
                data = quantization_config.to_dict()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        if hasattr(config, "to_dict"):
            try:
                data = config.to_dict()
                quantization_config = data.get("quantization_config")
                if isinstance(quantization_config, dict):
                    return dict(quantization_config)
            except Exception:
                pass
        return {}

    @staticmethod
    def _quantization_method_from_config(quantization_config: Dict[str, Any]) -> str:
        candidates = [
            quantization_config.get("quant_method"),
            quantization_config.get("format"),
            quantization_config.get("mode"),
            quantization_config.get("load_in_4bit"),
            quantization_config.get("load_in_8bit"),
        ]
        return " ".join(str(v).strip().lower() for v in candidates if v is not None)

    def _validate_transformers_quantization_runtime(self, quantization_config: Dict[str, Any]) -> None:
        if not quantization_config:
            return

        method = self._quantization_method_from_config(quantization_config)
        model_label = str(getattr(self, "model", ""))
        model_lower = model_label.lower()

        if "compressed-tensors" in method or method.startswith("pack-quantized"):
            if not _module_available("compressed_tensors"):
                raise ImportError(
                    "HuggingFace transformers model "
                    f"{model_label!r} uses compressed-tensors quantization, but the "
                    "`compressed-tensors` package is not installed. Install it explicitly for this "
                    "model family, or use an unquantized Transformers model, an MLX model with the "
                    "MLX provider, or a GGUF model with the GGUF path."
                )
            return

        if "awq" in method:
            if not (
                _module_available("autoawq")
                or _module_available("awq")
                or _module_available("gptqmodel")
            ):
                raise ImportError(
                    "HuggingFace transformers model "
                    f"{model_label!r} uses AWQ quantization, but no supported AWQ runtime is "
                    "installed. Install the quantization runtime intentionally for this platform, "
                    "or choose an unquantized Transformers model, MLX model, or GGUF model."
                )
            return

        if "gptq" in method:
            if not (
                _module_available("gptqmodel")
                or _module_available("auto_gptq")
                or _module_available("optimum")
            ):
                raise ImportError(
                    "HuggingFace transformers model "
                    f"{model_label!r} uses GPTQ quantization, but no supported GPTQ runtime is "
                    "installed. Install a compatible GPTQ runtime intentionally for this platform, "
                    "or choose an unquantized Transformers model, MLX model, or GGUF model."
            )
            return

        if "fp8" in method:
            try:
                import torch  # type: ignore
                from transformers.utils import is_torch_xpu_available  # type: ignore

                supports_fp8_runtime = bool(torch.cuda.is_available() or is_torch_xpu_available())
            except Exception:
                supports_fp8_runtime = False

            if not supports_fp8_runtime:
                raise ImportError(
                    "HuggingFace transformers model "
                    f"{model_label!r} uses FP8 quantization. Transformers FP8 execution requires "
                    "a CUDA/XPU runtime; this local HuggingFace provider instance cannot validate "
                    "the model on CPU/MPS. Use an unquantized Transformers model, an MLX model, a "
                    "GGUF model, or run this FP8 checkpoint on a supported CUDA/XPU stack."
                )
            return

        # MLX quantized repos expose safetensors and a compact config such as
        # {"bits": 4, "group_size": 64, "mode": "affine"}, but they are not
        # standard HuggingFace Transformers quantized checkpoints.
        if (
            "mlx" in model_lower
            or (
                quantization_config.get("mode") == "affine"
                and "bits" in quantization_config
                and "group_size" in quantization_config
                and "quant_method" not in quantization_config
            )
        ):
            raise ImportError(
                "HuggingFace transformers model "
                f"{model_label!r} looks like an MLX-format quantized checkpoint. Use "
                "`create_llm('mlx', model=...)` for MLX models, or choose a Transformers-native "
                "checkpoint for the HuggingFace provider."
            )

    @staticmethod
    def _unpack_transformers_load_result(result: Any) -> tuple[Any, Dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], result[1]
        return result, {}

    @staticmethod
    def _generation_config_defaults(config: Any) -> Dict[str, Any]:
        """Extract safe sampling defaults from a Transformers GenerationConfig-like object."""
        out: Dict[str, Any] = {}
        for key in ("temperature", "top_p", "top_k"):
            value = getattr(config, key, None)
            if value is None:
                continue
            try:
                if key == "top_k":
                    value_i = int(value)
                    if value_i <= 0:
                        continue
                    out[key] = value_i
                else:
                    out[key] = float(value)
            except Exception:
                continue
        return out

    def _apply_loaded_generation_config_defaults(self) -> None:
        """Apply defaults published by the loaded HF model when present.

        The model repo's `generation_config.json` is more specific than an
        architecture family default, but explicit caller values still win.
        """
        model = getattr(self, "model_instance", None)
        config = getattr(model, "generation_config", None)
        params = self._generation_config_defaults(config)
        if params:
            self._apply_generation_parameter_defaults(params)

    def _validate_transformers_weight_load(
        self,
        loading_info: Dict[str, Any],
        quantization_config: Dict[str, Any],
    ) -> None:
        if not quantization_config or not loading_info:
            return

        missing = list(loading_info.get("missing_keys") or [])
        unexpected = list(loading_info.get("unexpected_keys") or [])
        mismatched = list(loading_info.get("mismatched_keys") or [])
        errors = list(loading_info.get("error_msgs") or [])
        if not (missing or unexpected or mismatched or errors):
            return

        def sample(values: List[Any]) -> str:
            shown = [str(v) for v in values[:8]]
            suffix = "" if len(values) <= 8 else f", ... +{len(values) - 8} more"
            return ", ".join(shown) + suffix

        details = []
        if missing:
            details.append(f"missing_keys={sample(missing)}")
        if unexpected:
            details.append(f"unexpected_keys={sample(unexpected)}")
        if mismatched:
            details.append(f"mismatched_keys={sample(mismatched)}")
        if errors:
            details.append(f"errors={sample(errors)}")

        raise RuntimeError(
            "HuggingFace transformers model "
            f"{getattr(self, 'model', '')!r} did not load its quantized weights cleanly. "
            "This is a model/runtime compatibility issue, not a generation or cache issue. "
            + "; ".join(details)
        )

    def _load_transformers_model(self):
        """Load standard HuggingFace transformers model"""
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoConfig,
                AutoModel,
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
            )
        except Exception as e:
            raise ImportError("Transformers + PyTorch are required for HuggingFace (transformers) models.") from e

        try:
            # Check if this is a vision model that requires special handling
            if self._is_vision_model(self.model):
                return self._load_vision_model()

            quantization_config: Dict[str, Any] = {}
            config = None
            try:
                config = AutoConfig.from_pretrained(self.model, **self._transformers_config_kwargs())
            except Exception:
                config = None
            if config is not None:
                quantization_config = self._extract_quantization_config(config)
                self._validate_transformers_quantization_runtime(quantization_config)

            # Load tokenizer with transformers-specific parameters
            tokenizer_kwargs = self._transformers_config_kwargs()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, **tokenizer_kwargs)

            # Load model with all transformers-specific parameters
            # Try AutoModelForCausalLM first, fall back to AutoModel for custom models
            model_kwargs = self.transformers_kwargs.copy()

            # An explicit caller choice always wins; ABSTRACTCORE_TRANSFORMERS_ATTN_IMPL
            # overrides the transformers default (use "eager" as the stopgap for
            # the torch-2.10 MPS SDPA defect — see _warn_if_mps_sdpa_defective).
            if 'attn_implementation' not in model_kwargs:
                forced = os.environ.get("ABSTRACTCORE_TRANSFORMERS_ATTN_IMPL")
                if isinstance(forced, str) and forced.strip():
                    model_kwargs['attn_implementation'] = forced.strip()

            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                model_kwargs['local_files_only'] = True
            if quantization_config:
                model_kwargs["output_loading_info"] = True

            # Caller-requested quantization (quantization_config=, or the legacy
            # load_in_4bit/8bit spelling translated in __init__). Validated on
            # the same runtime rules as a checkpoint that carries its own config,
            # and pinned to the target device at load time: a bnb-quantized
            # module cannot be moved afterwards, and the post-load `.to(device)`
            # below is skipped only when `device_map` is present.
            if getattr(self, "_transformers_quantization_request", None) is not None:
                requested = self._transformers_quantization_request
                as_dict = requested if isinstance(requested, dict) else (
                    requested.to_dict() if hasattr(requested, "to_dict") else {})
                self._validate_transformers_quantization_runtime(dict(as_dict or {}))
                if self.device in ("cuda", "mps") and "device_map" not in model_kwargs:
                    model_kwargs["device_map"] = {"": self.device}
                    self.transformers_kwargs["device_map"] = model_kwargs["device_map"]

            def _load_with(kwargs: Dict[str, Any]):
                try:
                    obj = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)
                except ValueError as e:
                    if "Unrecognized configuration class" in str(e) or "glm4v" in str(e).lower():
                        # Fall back to AutoModel for custom models like DeepSeek-OCR
                        obj = AutoModel.from_pretrained(self.model, **kwargs)
                    else:
                        raise
                inst, info = self._unpack_transformers_load_result(obj)
                self._validate_transformers_weight_load(info, quantization_config)
                return inst

            # MUST run before the first weight is quantized: the quantize op
            # itself resolves (and latches) the fused kernel. See
            # `_prepare_bnb_mps_fused_kernel`.
            self._prepare_bnb_mps_fused_kernel(quantization_config)

            self.model_instance = _load_with(model_kwargs)

            # Move to device (only if not using device_map)
            if self.device in ["cuda", "mps"] and 'device_map' not in self.transformers_kwargs:
                self.model_instance = self.model_instance.to(self.device)
            _warn_if_mps_sdpa_defective(self.device, self.model_instance)
            # 4-bit on MPS silently loses its fused kernel and decodes ~x4
            # slower. Probe at load, warn once, never raise.
            self._warn_if_bnb_mps_fused_kernel_missing()
            self._apply_loaded_generation_config_defaults()

            # Create pipeline - handle custom models that don't support text-generation
            device_arg = 0 if self.device == "cuda" else -1
            if self.device == "mps":
                device_arg = -1

            try:
                # Don't pass device argument if using device_map (accelerate)
                if 'device_map' in self.transformers_kwargs:
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model_instance,
                        tokenizer=self.tokenizer
                    )
                else:
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model_instance,
                        tokenizer=self.tokenizer,
                        device=device_arg
                    )
            except ValueError as e:
                if "not supported for text-generation" in str(e) or "accelerate" in str(e):
                    # For custom models like DeepSeek-OCR, skip pipeline creation
                    # We'll handle generation directly through the model
                    self.pipeline = None
                else:
                    raise

        except Exception as e:
            error_str = str(e).lower()
            if ('not found' in error_str or 'does not exist' in error_str or
                'not a valid model identifier' in error_str):
                available_models = self.list_available_models()
                error_message = format_model_error("HuggingFace", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise RuntimeError(f"Failed to load HuggingFace model {self.model}: {str(e)}")

    def _load_vision_model(self):
        """Load vision model using AutoModelForImageTextToText and AutoProcessor"""
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

            # Suppress progress bars during model loading unless in debug mode
            import os
            from transformers.utils import logging as transformers_logging

            if not self.debug:
                # Disable transformers progress bars
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
                transformers_logging.set_verbosity_error()
                # Disable tqdm progress bars
                os.environ['DISABLE_TQDM'] = '1'

            # Load processor for vision models (handles both text and images)
            processor_kwargs = {k: v for k, v in self.transformers_kwargs.items() 
                              if k in ['trust_remote_code']}
            # Enable trust_remote_code for custom architectures like GLM4V
            processor_kwargs['trust_remote_code'] = True
            # Set use_fast=True to avoid the slow processor warning
            processor_kwargs['use_fast'] = True
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                processor_kwargs['local_files_only'] = True

            # Use local cache path if offline mode is enabled and model is cached
            model_path = self.model
            if _config.should_force_local_files_only():
                local_path = _get_local_model_path(self.model)
                if local_path:
                    model_path = local_path
                    processor_kwargs.pop('local_files_only', None)  # Remove since we're using local path
                    self.logger.debug(f"Loading processor from local cache: {local_path}")

            self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

            # Load vision model using AutoModelForImageTextToText with trust_remote_code
            vision_kwargs = self.transformers_kwargs.copy()
            vision_kwargs['trust_remote_code'] = True
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                vision_kwargs['local_files_only'] = True

            # Safer defaults on GPU backends: float16 unless caller provided torch_dtype.
            try:
                if self.device in {"mps", "cuda"} and "torch_dtype" not in vision_kwargs:
                    import torch as _torch

                    vision_kwargs["torch_dtype"] = _torch.float16
            except Exception:
                pass

            # Use local cache path if offline mode is enabled and model is cached
            model_path = self.model
            if _config.should_force_local_files_only():
                local_path = _get_local_model_path(self.model)
                if local_path:
                    model_path = local_path
                    vision_kwargs.pop('local_files_only', None)  # Remove since we're using local path
                    self.logger.debug(f"Loading model from local cache: {local_path}")

            self.model_instance = AutoModelForImageTextToText.from_pretrained(model_path, **vision_kwargs)
            self._apply_loaded_generation_config_defaults()

            # Restore logging levels if they were suppressed
            if not self.debug:
                # Restore transformers logging
                transformers_logging.set_verbosity_warning()
                # Remove tqdm suppression
                if 'DISABLE_TQDM' in os.environ:
                    del os.environ['DISABLE_TQDM']

            # Move to device (only if not using device_map)
            if self.device in ["cuda", "mps"] and 'device_map' not in self.transformers_kwargs:
                self.model_instance = self.model_instance.to(self.device)

            try:
                self.model_instance.eval()
            except Exception:
                pass

            # For vision models, we don't use the standard pipeline
            self.pipeline = None

            self.logger.info(f"Successfully loaded vision model {self.model} using AutoModelForImageTextToText")

        except Exception as e:
            error_str = str(e).lower()

            # Check for transformers version issues
            if 'glm4v' in error_str and 'does not recognize this architecture' in error_str:
                import transformers
                current_version = transformers.__version__
                raise RuntimeError(
                    f"GLM4V architecture requires transformers>=4.57.1, but you have {current_version}. "
                    f"Please upgrade: pip install transformers>=4.57.1"
                )
            elif ('not found' in error_str or 'does not exist' in error_str or
                'not a valid model identifier' in error_str):
                available_models = self.list_available_models()
                error_message = format_model_error("HuggingFace", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise RuntimeError(f"Failed to load HuggingFace vision model {self.model}: {str(e)}")

    def _find_gguf_in_cache(
        self,
        model_name: str,
        *,
        _seen: Optional[set[str]] = None,
        _selector: Optional[str] = None,
    ) -> Optional[str]:
        """Find GGUF model in local caches (HuggingFace hub / LM Studio; cache-only, no downloading).

        Pure locator: it answers "where does this GGUF live", and callers decide
        whether asking was legitimate. The ADR 0009 gate lives at the construction
        site (`_reject_silent_gguf_substitution`), so probing a Hub alias here stays
        allowed.
        """

        if _seen is None:
            _seen = set()
        key = str(model_name or "").strip()
        if not key or key in _seen:
            return None
        _seen.add(key)

        # An explicit ":quant" selector belongs to the ORIGINAL request and must survive
        # alias/manifest recursion — otherwise `repo:Q8_0` resolved through a manifest and
        # landed on the preferred-quant default under a name the caller had disambiguated.
        selector_source = str(model_name or "").strip()
        explicit_selector = _selector
        if explicit_selector is None and ":" in selector_source:
            explicit_selector = selector_source.split(":", 1)[1].strip().strip("/") or None

        def _announce_quant_pick(picked: Path, candidates: list[Path]) -> str:
            # A repository id underdetermines WHICH quantization to load, so a default
            # pick is legitimate resolution rather than substitution (ADR 0009) — but
            # ADR 0001 still forbids it being invisible.
            if len(candidates) > 1:
                _artifact_logger().warning(
                    "huggingface: %r names a GGUF repository holding %d quantizations; "
                    "loading %r by default. Append ':<quant>' to the handle to choose "
                    "explicitly.",
                    model_name, len(candidates), picked.name,
                )
            return str(picked)

        def _pick_preferred_gguf(gguf_files: list[Path]) -> Optional[str]:
            if not gguf_files:
                return None
            gguf_files = sorted(gguf_files, key=lambda p: p.name)

            if explicit_selector:
                selector_upper = explicit_selector.upper()
                # Ordered and deterministic: exact filename, then filename substring,
                # then path substring. The previous single OR-pass let a loose *path*
                # match on an early file beat an exact *filename* match on a later one.
                for matches in (
                    lambda p: selector_upper == p.name.upper(),
                    lambda p: selector_upper in p.name.upper(),
                    lambda p: selector_upper in str(p).upper(),
                ):
                    for gguf_file in gguf_files:
                        if matches(gguf_file):
                            return str(gguf_file)

                # ADR 0009: an explicit quant selector that matches nothing is a request
                # we cannot satisfy. Falling through to `preferred_quants` here handed
                # back a DIFFERENT quantization than the caller spelled out, silently.
                raise ModelArtifactMismatchError(
                    f"No GGUF matching the selector ':{explicit_selector}' for "
                    f"{model_name!r}.\n"
                    f"\n"
                    f"  Requested : ':{explicit_selector}'\n"
                    f"  Available : {', '.join(p.name for p in gguf_files)}\n"
                    f"\n"
                    f"Re-request with one of the available files, or drop the ':...' "
                    f"selector to accept this provider's default quantization pick."
                )

            preferred_quants = ['Q4_K_M', 'Q5_K_M', 'Q4_0', 'Q4_1', 'Q5_0', 'Q8_0']
            for quant in preferred_quants:
                for gguf_file in gguf_files:
                    if quant in gguf_file.name.upper():
                        return _announce_quant_pick(gguf_file, gguf_files)
            return _announce_quant_pick(gguf_files[0], gguf_files)

        def _to_repo_id(raw: str) -> Optional[str]:
            s = str(raw or "").strip()
            if not s:
                return None
            if ":" in s:
                s = s.split(":", 1)[0].strip()
            if not s:
                return None
            if s.startswith("models--"):
                parts = s.replace("models--", "").split("--", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    return f"{parts[0]}/{parts[1]}"
            if "--" in s and "/" not in s:
                parts = s.split("--", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    return f"{parts[0]}/{parts[1]}"
            if "/" in s:
                return s.strip().strip("/")
            return None

        # Direct filesystem path FIRST: a .gguf FILE, or a DIRECTORY containing
        # .gguf files (LM Studio stores models at real on-disk paths, and users
        # naturally pass that path — e.g. ~/.lmstudio/models/org/Model-GGUF).
        # Without this, an absolute/relative path is mis-parsed as a repo id by
        # the cache logic below and reported "not found" even though the file is
        # right there. A trailing ":selector" (quant hint) is honored when the
        # head is a real path (never eaten from a non-path hub id / Windows
        # drive letter). Resolved before any cache lookup so a path always wins.
        try:
            path_head = key
            if ":" in key:
                head = key.split(":", 1)[0]
                if head and (Path(head).expanduser().is_file() or Path(head).expanduser().is_dir()):
                    path_head = head
            candidate = Path(path_head).expanduser()
            if candidate.is_file() and candidate.suffix.lower() == ".gguf":
                return str(candidate)
            if candidate.is_dir():
                picked = _pick_preferred_gguf(list(candidate.glob("*.gguf")))
                if picked:
                    return picked
        except ModelArtifactMismatchError:
            # An unsatisfiable explicit selector is an answer, not a lookup failure.
            raise
        except (OSError, ValueError):
            # A non-path string (NUL bytes, over-long) — fall through to cache
            # resolution, never crash the lookup.
            pass

        # Normalize model name to cache format
        # Convert "unsloth/model" or "unsloth--model" to "models--unsloth--model"
        cache_name = self._normalize_to_cache_format(model_name)

        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_dir = cache_base / cache_name

        if not model_cache_dir.exists():
            model_cache_dir = None

        # Look for GGUF files in HuggingFace snapshots
        if model_cache_dir is not None:
            snapshots_dir = model_cache_dir / "snapshots"
            if snapshots_dir.exists():
                # Find the latest snapshot (most recent directory)
                try:
                    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshot_dirs:
                        # Use the most recent snapshot
                        latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
                        if len(snapshot_dirs) > 1:
                            # Same repository either way, so not a substitution — but the
                            # caller named no revision and we picked one, so say which.
                            _artifact_logger().warning(
                                "huggingface: %r has %d cached snapshots (revisions); using the "
                                "most recently modified one (%s).",
                                model_name, len(snapshot_dirs), latest_snapshot.name,
                            )

                        # Look for GGUF files in the snapshot
                        gguf_files = list(latest_snapshot.glob("*.gguf"))
                        picked = _pick_preferred_gguf(gguf_files)
                        if picked:
                            return picked

                except ModelArtifactMismatchError:
                    raise
                except Exception:
                    pass

        # Fallback: LM Studio model cache (~/.lmstudio/models) often stores GGUF files directly.
        try:
            from ..utils.model_cache import default_lmstudio_model_dirs, resolve_lmstudio_model_dir

            repo_id = _to_repo_id(model_name)
            lm_dir = resolve_lmstudio_model_dir(repo_id, base_dirs=default_lmstudio_model_dirs()) if repo_id else None
            if lm_dir is not None and lm_dir.is_dir():
                gguf_files = list(lm_dir.glob("*.gguf"))
                picked = _pick_preferred_gguf(gguf_files)
                if picked:
                    return picked
        except ModelArtifactMismatchError:
            raise
        except Exception:
            pass

        # LM Studio Hub alias support: resolve org/model manifest to its GGUF dependency.
        try:
            from ..utils.model_cache import resolve_lmstudio_hub_manifest

            repo_id = _to_repo_id(model_name)
            if repo_id:
                manifest_path = resolve_lmstudio_hub_manifest(repo_id)
            else:
                manifest_path = None
            if manifest_path is not None:
                try:
                    raw = manifest_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    raw = ""
                manifest = json.loads(raw) if raw.strip() else {}
                deps = manifest.get("dependencies") if isinstance(manifest, dict) else None
                if isinstance(deps, list):
                    candidates: list[str] = []
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
                            if user and repo:
                                candidates.append(f"{user}/{repo}")
                        for mk in dep.get("modelKeys") or []:
                            if isinstance(mk, str) and mk.strip():
                                candidates.append(mk.strip())

                    seen: set[str] = set()
                    ordered: list[str] = []
                    for c in candidates:
                        c2 = str(c or "").strip().strip("/")
                        if not c2 or c2 in seen:
                            continue
                        seen.add(c2)
                        ordered.append(c2)

                    for cand in ordered:
                        resolved = self._find_gguf_in_cache(
                            cand, _seen=_seen, _selector=explicit_selector
                        )
                        if resolved:
                            if str(cand).strip().strip("/").lower() != str(repo_id or "").strip().strip("/").lower():
                                # Legitimate only because the caller already asked for a
                                # GGUF (ADR 0009 gate ran at construction). Still a
                                # cross-repository hop, so it is stated out loud.
                                _artifact_logger().warning(
                                    "huggingface: LM Studio Hub manifest for %r resolves to a "
                                    "different repository %r; loading %r.",
                                    repo_id, cand, resolved,
                                )
                            return resolved
        except ModelArtifactMismatchError:
            raise
        except Exception:
            pass

        return None

    def _normalize_to_cache_format(self, model_name: str) -> str:
        """Convert model name to HuggingFace cache directory format"""
        # Remove any ":filename" suffix
        if ':' in model_name:
            model_name = model_name.split(':', 1)[0]

        # Handle different input formats:
        if model_name.startswith('models--'):
            # Already in cache format
            return model_name
        elif '/' in model_name:
            # Standard format: "unsloth/model" -> "models--unsloth--model"
            return f"models--{model_name.replace('/', '--')}"
        elif '--' in model_name and not model_name.startswith('models--'):
            # Cache format without prefix: "unsloth--model" -> "models--unsloth--model"
            return f"models--{model_name}"
        else:
            # Single name, assume it's just the model part
            return f"models--{model_name}"

    def _load_gguf_model(self):
        """Load GGUF model using llama-cpp-python (cache-only, no downloading)"""
        import os
        try:
            llama_cls = globals().get("Llama")
            if llama_cls is None:
                from llama_cpp import Llama as llama_cls  # type: ignore

            # llama-cpp-python 0.3.x can throw an "Exception ignored in: __del__" when model
            # initialization fails early (missing `sampler` attribute). Patch defensively to
            # keep failures clean and actionable.
            try:  # pragma: no cover
                import llama_cpp._internals as _llama_internals  # type: ignore

                if hasattr(_llama_internals, "LlamaModel") and not hasattr(_llama_internals.LlamaModel, "sampler"):
                    setattr(_llama_internals.LlamaModel, "sampler", None)
            except Exception:
                pass

            model_path = None

            # First, try as a direct file path
            if Path(self.model).exists() and self.model.endswith('.gguf'):
                model_path = self.model
            else:
                # Try to find in HuggingFace cache
                model_path = self._find_gguf_in_cache(self.model)

            if not model_path:
                # Model not found in cache - provide graceful fallback
                self._handle_gguf_not_found()
                return

            # Verify file exists and is accessible
            if not Path(model_path).exists():
                raise FileNotFoundError(f"GGUF model file not found: {model_path}")

            gguf_arch: str | None = None
            try:
                from ..utils.model_cache import read_gguf_architecture

                gguf_arch = read_gguf_architecture(Path(model_path))
            except Exception:
                gguf_arch = None

            model_lower = self.model.lower()

            if "mtp" in model_lower:
                self.logger.warning(
                    "Loading an MTP GGUF through llama-cpp-python. The model can be used as a regular GGUF, "
                    "but current public llama-cpp-python bindings do not expose native MTP acceleration in-process. "
                    "Use an external llama.cpp server/runtime with native MTP support if you need the speedup."
                )

            # Determine chat format for function calling
            chat_format = None
            if 'qwen' in model_lower or 'coder' in model_lower:
                # Qwen models often support function calling
                chat_format = "chatml-function-calling"
            elif 'functionary' in model_lower:
                chat_format = "functionary-v2"

            # IMPORTANT (macOS): when loading GGUFs from LM Studio's cache, llama-cpp-python can
            # segfault with `use_mmap=True` (even on supported architectures). Disable mmap for
            # LM Studio paths to keep loads stable.
            use_mmap = True
            try:
                import platform

                if platform.system().lower() == "darwin":
                    from ..utils.model_cache import default_lmstudio_model_dirs

                    model_real = Path(model_path).resolve()
                    for base in default_lmstudio_model_dirs():
                        try:
                            base_real = base.resolve()
                        except Exception:
                            base_real = base
                        try:
                            if model_real.is_relative_to(base_real):
                                use_mmap = False
                                break
                        except Exception:
                            # Python <3.9 fallback (or odd path types).
                            if str(model_real).startswith(str(base_real) + os.sep):
                                use_mmap = False
                                break
            except Exception:
                pass

            # Initialize llama-cpp-python with stderr redirected to our logger.
            #
            # `self.max_tokens` is AbstractCore's unified "context window budget" and defaults
            # to the model's `max_tokens` from `assets/model_capabilities.json`.
            #
            # For GGUF/llama.cpp we must allocate a concrete KV cache (`n_ctx`). When callers do
            # not pass `max_tokens=...` explicitly and the advertised context is too large for
            # the local machine, we retry with smaller windows (best-effort) instead of using a
            # hidden env var.
            requested_n_ctx = self.max_tokens if self.max_tokens is not None else 16384
            try:
                requested_n_ctx_i = int(requested_n_ctx)
            except Exception:
                requested_n_ctx_i = 16384
            if requested_n_ctx_i <= 0:
                requested_n_ctx_i = 16384

            if getattr(self, "_user_provided_max_tokens", False):
                candidate_ctxs = [requested_n_ctx_i]
            else:
                candidate_ctxs = [requested_n_ctx_i]
                for fallback in (131072, 65536, 32768, 16384, 8192, 4096):
                    if fallback < requested_n_ctx_i:
                        candidate_ctxs.append(int(fallback))

            last_error: Exception | None = None
            chosen_n_ctx: int | None = None
            for n_ctx_i in candidate_ctxs:
                llama_kwargs = {
                    "model_path": model_path,
                    "n_ctx": int(n_ctx_i),
                    "n_gpu_layers": self.n_gpu_layers,
                    "chat_format": chat_format,
                    "verbose": self.debug,  # Use debug flag for verbose output
                    "n_threads": os.cpu_count() // 2 if os.cpu_count() else 4,
                    # Additional performance settings
                    "n_batch": 512,
                    "use_mmap": use_mmap,
                    "use_mlock": False,
                }

                try:
                    self.llm = llama_cls(**llama_kwargs)
                    chosen_n_ctx = int(n_ctx_i)
                    break
                except Exception as e:
                    # Common on macOS: Metal backend unavailable for the current process. Retry on CPU.
                    if isinstance(self.n_gpu_layers, int) and self.n_gpu_layers != 0:
                        try:
                            self.logger.warning(
                                f"GGUF load failed with n_gpu_layers={self.n_gpu_layers}; retrying with CPU (n_gpu_layers=0). Error: {e}"
                            )
                            llama_kwargs_cpu = llama_kwargs.copy()
                            llama_kwargs_cpu["n_gpu_layers"] = 0
                            # Avoid any GPU KV offload in the retry.
                            llama_kwargs_cpu["offload_kqv"] = False
                            self.llm = llama_cls(**llama_kwargs_cpu)
                            self.n_gpu_layers = 0
                            chosen_n_ctx = int(n_ctx_i)
                            break
                        except Exception as e_cpu:
                            e = e_cpu

                    last_error = e

                    # If caller explicitly requested a context window, fail fast with a clear message.
                    if getattr(self, "_user_provided_max_tokens", False):
                        raise RuntimeError(
                            f"Failed to load GGUF model {self.model} with n_ctx={n_ctx_i}. "
                            "Try lowering max_tokens=... when constructing HuggingFaceProvider(). "
                            f"Underlying error: {e}"
                        ) from e

                    # Best-effort retry with smaller context windows for local stability.
                    is_last = (n_ctx_i == candidate_ctxs[-1])
                    if not is_last:
                        self.logger.warning(
                            f"GGUF load failed for {self.model} with n_ctx={n_ctx_i}; retrying with a smaller context window. Error: {e}"
                        )
                        continue

                    # No more fallbacks available.
                    if gguf_arch == "qwen35moe":
                        raise RuntimeError(
                            "This GGUF uses architecture 'qwen35moe', which is not supported by the "
                            "llama.cpp version bundled with your installed llama-cpp-python. "
                            "LM Studio's Metal llama.cpp backend can load it; use `--provider lmstudio` "
                            "for this model, or upgrade to a llama-cpp-python build that includes "
                            "qwen35moe support."
                        ) from e
                    if gguf_arch:
                        raise RuntimeError(
                            f"Failed to load GGUF model (architecture: {gguf_arch}): {e}"
                        ) from e
                    raise

            if self.llm is None or chosen_n_ctx is None:
                raise RuntimeError(f"Failed to load GGUF model {self.model}: {last_error}")

            # Keep AbstractCore's token budget in sync with the actual llama.cpp context window.
            #
            # `model_capabilities.json` stores advertised limits, but GGUF loads require a
            # concrete `n_ctx` allocation. After successful load (including fallbacks),
            # treat `self.max_tokens` as the runtime context window budget.
            self.max_tokens = int(chosen_n_ctx)

            # Ensure output reservation never exceeds the runtime context window.
            #
            # Many models (e.g. Qwen3.5) advertise very large output limits, but GGUF runtime
            # context windows are often smaller locally. Keep invariants consistent to avoid
            # negative/invalid derived `max_input_tokens` and provider-side errors.
            try:
                if (
                    isinstance(self.max_output_tokens, int)
                    and int(self.max_output_tokens) > int(self.max_tokens)
                ):
                    self.logger.warning(
                        (
                            f"Clamping max_output_tokens for {self.model}: configured "
                            f"max_output_tokens={self.max_output_tokens} exceeds GGUF n_ctx={self.max_tokens}."
                        )
                    )
                    self.max_output_tokens = int(self.max_tokens)
            except Exception:
                pass

        except ModelArtifactMismatchError:
            # ADR 0009 Enforcement: an artifact-fidelity refusal must reach the caller
            # as itself. Wrapping it in RuntimeError here made the contract type
            # uncatchable on the constructor path — callers that handle
            # ModelArtifactMismatchError saw a generic load failure instead, which is
            # the same information loss the ADR exists to prevent.
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model {self.model}: {str(e)}")

    def _handle_gguf_not_found(self):
        """Handle GGUF model not found with graceful fallback like other providers"""
        # Suggest the correct repo format
        suggested_repo = self._suggest_correct_repo_format(self.model)

        # List any similar models in cache
        similar_models = self._find_similar_gguf_models()

        error_parts = [
            f"❌ GGUF model '{self.model}' not found in local caches (HuggingFace hub / LM Studio).",
            "",
            "💡 To download this model, run:",
            f"   huggingface-cli download {suggested_repo}",
            "",
            "🔍 Suggested formats:",
            f"   • Correct: '{suggested_repo}'",
            f"   • Your input: '{self.model}'",
        ]

        if similar_models:
            error_parts.extend([
                "",
                "📂 Similar GGUF models found in cache:",
            ])
            for model in similar_models[:5]:  # Show max 5
                error_parts.append(f"   • {model}")

        error_parts.extend([
            "",
            "📖 For more info: https://huggingface.co/docs/hub/en/gguf",
            "🔧 AbstractCore only uses cached models - we never download automatically."
        ])

        error_message = "\n".join(error_parts)
        raise ModelNotFoundError(error_message)

    def _suggest_correct_repo_format(self, model_name: str) -> str:
        """Suggest the correct repository format"""
        # Handle various input formats and suggest the standard format
        if model_name.startswith('models--'):
            # "models--unsloth--model" -> "unsloth/model"
            parts = model_name.replace('models--', '').split('--', 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"

        elif '--' in model_name and not '/' in model_name:
            # "unsloth--model" -> "unsloth/model"
            parts = model_name.split('--', 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"

        # Return as-is if already in correct format or unknown format
        return model_name

    def _find_similar_gguf_models(self) -> List[str]:
        """Find similar GGUF models in cache"""
        similar: set[str] = set()

        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_base.exists():
            try:
                for cache_dir in cache_base.iterdir():
                    if cache_dir.is_dir() and 'gguf' in cache_dir.name.lower():
                        if cache_dir.name.startswith('models--'):
                            repo_name = cache_dir.name.replace('models--', '').replace('--', '/', 1)
                            similar.add(repo_name)
            except Exception:
                pass

        # Also include GGUF models stored in LM Studio's model cache.
        try:
            from ..utils.model_cache import default_lmstudio_model_dirs

            for base in default_lmstudio_model_dirs():
                try:
                    for org_dir in base.iterdir():
                        if not org_dir.is_dir():
                            continue
                        for model_dir in org_dir.iterdir():
                            if not model_dir.is_dir():
                                continue
                            try:
                                if any(p.suffix.lower() == ".gguf" for p in model_dir.iterdir()):
                                    similar.add(f"{org_dir.name}/{model_dir.name}")
                            except Exception:
                                continue
                except Exception:
                    continue
        except Exception:
            pass

        return sorted(similar)

    def _handle_timeout_parameter(self, kwargs: Dict[str, Any]) -> None:
        """
        Handle timeout parameter for HuggingFace provider.

        Since HuggingFace models run locally (both transformers and GGUF),
        timeout parameters don't apply. If a non-None timeout is provided,
        issue a warning and treat it as None (infinity).

        Args:
            kwargs: Initialization kwargs that may contain timeout
        """
        timeout_value = kwargs.get('timeout')
        if timeout_value is not None:
            import warnings
            warnings.warn(
                f"HuggingFace provider runs models locally and does not support timeout parameters. "
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
        HuggingFace provider doesn't use HTTP clients for model inference.
        Local models (transformers and GGUF) don't have timeout constraints.
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
        """Generate response using appropriate backend"""

        if self.model_type == "gguf":
            return self._generate_gguf(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
        else:
            return self._generate_transformers(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)

    def _generate_transformers(self,
                               prompt: str,
                               messages: Optional[List[Dict[str, str]]] = None,
                               system_prompt: Optional[str] = None,
                               tools: Optional[List[Dict[str, Any]]] = None,
                               media: Optional[List['MediaContent']] = None,
                               stream: bool = False,
                               response_model: Optional[Type[BaseModel]] = None,
                               **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using transformers backend with optional Outlines native structured output"""

        if not self.pipeline:
            # Handle vision models that use processor instead of pipeline
            if self.processor and hasattr(self.model_instance, 'generate'):
                return self._generate_vision_model(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
            # Handle custom models like DeepSeek-OCR that don't support standard pipelines
            elif hasattr(self.model_instance, 'infer'):
                return self._generate_custom_model(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
            else:
                return GenerateResponse(
                    content="Error: Transformers model not loaded or doesn't support generation",
                    model=self.model,
                    finish_reason="error"
                )

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
                    content="Error: structured_output_method='native_outlines' requires Outlines library. Install with: pip install \"abstractcore[huggingface]\"",
                    model=self.model,
                    finish_reason="error"
                )

            # Try Outlines if available (auto or native_outlines mode)
            if OUTLINES_AVAILABLE:
                try:
                    import outlines  # type: ignore

                    # Cache Outlines model wrapper to avoid re-initialization
                    if not hasattr(self, '_outlines_model') or self._outlines_model is None:
                        self.logger.debug("Creating Outlines model wrapper for native structured output")
                        self._outlines_model = outlines.from_transformers(
                            self.model_instance,
                            self.tokenizer
                        )

                    # Build input text (same as normal generation, thinking controls included —
                    # this lane used to drop them entirely, adversarial find 2026-08-19)
                    _ol_enable_thinking = kwargs.get("_acore_hf_transformers_enable_thinking")
                    _ol_reasoning_effort = kwargs.get("_acore_hf_transformers_reasoning_effort")
                    input_text = self._build_input_text_transformers(
                        prompt,
                        messages,
                        system_prompt,
                        tools,
                        enable_thinking=_ol_enable_thinking if isinstance(_ol_enable_thinking, bool) else None,
                        reasoning_effort=_ol_reasoning_effort if isinstance(_ol_reasoning_effort, str) else None,
                    )

                    generation_kwargs = self._prepare_generation_kwargs(**kwargs)
                    max_new_tokens = self._get_provider_max_tokens_param(generation_kwargs)

                    # Create constrained generator with JSON schema
                    self.logger.debug(f"Using Outlines native structured output for {response_model.__name__}")
                    generator = self._outlines_model(
                        input_text,
                        outlines.json_schema(response_model),
                        max_tokens=max_new_tokens,
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

        # Build input text with tool and media support
        # Handle media content first if present
        media_enrichment = None
        if media:
            try:
                from ..media.handlers import LocalMediaHandler
                media_handler = LocalMediaHandler("huggingface", self.model_capabilities, model_name=self.model)

                # Create multimodal message combining text and media
                multimodal_message = media_handler.create_multimodal_message(prompt, media)
                media_enrichment = getattr(media_handler, "media_enrichment", None)

                # For local providers, we get text-embedded content
                if isinstance(multimodal_message, str):
                    prompt = multimodal_message
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
                            prompt = text_content or prompt
                        else:
                            prompt = str(multimodal_message["content"])
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")

        # Generation parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_new_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        temperature = generation_kwargs.get("temperature", self.temperature)
        top_p = generation_kwargs.get("top_p", 0.9)
        top_k = generation_kwargs.get("top_k")
        seed_value = generation_kwargs.get("seed")
        hf_transformers_enable_thinking = kwargs.get("_acore_hf_transformers_enable_thinking")
        hf_transformers_reasoning_effort = kwargs.get("_acore_hf_transformers_reasoning_effort")

        prompt_cache_key = kwargs.get("prompt_cache_key")
        prefilled_modules = kwargs.get("prompt_cache_prefilled_modules")
        if (
            isinstance(prompt_cache_key, str)
            and prompt_cache_key.strip()
            and self._transformers_prompt_cache_supported()
        ):
            try:
                cached = self._single_generate_transformers_cached(
                    prompt=str(prompt or ""),
                    prompt_cache_key=prompt_cache_key.strip(),
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    prefilled_modules=prefilled_modules,
                    max_new_tokens=max_new_tokens,
                    temperature=float(temperature) if temperature is not None else None,
                    top_p=float(top_p) if top_p is not None else 0.9,
                    top_k=int(top_k) if top_k is not None else None,
                    seed=seed_value,
                    enable_thinking=hf_transformers_enable_thinking if isinstance(hf_transformers_enable_thinking, bool) else None,
                    reasoning_effort=hf_transformers_reasoning_effort,
                )
            except Exception as e:
                return GenerateResponse(
                    content=f"Error generating response with prompt cache: {str(e)}",
                    model=self.model,
                    finish_reason="error",
                )

            if stream:
                def _stream_cached() -> Iterator[GenerateResponse]:
                    # Simulated streaming: yield word chunks, then run tool execution if requested.
                    content = cached.content or ""
                    tool_call_tags = kwargs.get("tool_call_tags")
                    if tool_call_tags and content:
                        try:
                            from ..tools.tag_rewriter import create_tag_rewriter
                            rewriter = create_tag_rewriter(tool_call_tags)
                            content = rewriter.rewrite_text(content)
                        except Exception:
                            pass

                    words = content.split()
                    collected = ""
                    if not words:
                        yield GenerateResponse(content="", model=self.model, finish_reason="stop")
                        return
                    for i, word in enumerate(words):
                        chunk_content = word + (" " if i < len(words) - 1 else "")
                        collected += chunk_content
                        yield GenerateResponse(
                            content=chunk_content,
                            model=self.model,
                            finish_reason="stop" if i == len(words) - 1 else None,
                        )

                    # Tool execution (prompted) happens after streaming in this provider.
                    if tools and getattr(self.tool_handler, "supports_prompted", False) and collected:
                        complete = GenerateResponse(
                            content=collected,
                            model=self.model,
                            finish_reason="stop",
                        )
                        final = self._handle_prompted_tool_execution(complete, tools)
                        if final.content and final.content != collected:
                            suffix = final.content[len(collected):]
                            if suffix:
                                yield GenerateResponse(
                                    content=suffix,
                                    model=self.model,
                                    finish_reason="stop",
                                )

                return _stream_cached()

            response = cached
            if media_enrichment:
                from ..media.enrichment import merge_enrichment_metadata

                response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

            # Handle tool execution for prompted models
            if tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools)

            return response

        input_text = self._build_input_text_transformers(
            prompt,
            messages,
            system_prompt,
            tools,
            enable_thinking=hf_transformers_enable_thinking if isinstance(hf_transformers_enable_thinking, bool) else None,
            reasoning_effort=hf_transformers_reasoning_effort if isinstance(hf_transformers_reasoning_effort, str) else None,
        )

        try:
            if stream:
                return self._stream_generate_transformers_with_tools(input_text, max_new_tokens, temperature, top_p, top_k, tools, kwargs.get('tool_call_tags'), seed_value)
            else:
                response = self._single_generate_transformers(input_text, max_new_tokens, temperature, top_p, top_k, seed_value)
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

                # Handle tool execution for prompted models
                if tools and self.tool_handler.supports_prompted and response.content:
                    response = self._handle_prompted_tool_execution(response, tools)

                return response

        except Exception as e:
            return GenerateResponse(
                content=f"Error generating response: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _generate_custom_model(self,
                              prompt: str,
                              messages: Optional[List[Dict[str, str]]] = None,
                              system_prompt: Optional[str] = None,
                              tools: Optional[List[Dict[str, Any]]] = None,
                              media: Optional[List['MediaContent']] = None,
                              stream: bool = False,
                              response_model: Optional[Type[BaseModel]] = None,
                              **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using custom model methods (e.g., DeepSeek-OCR's infer method)"""

        import time
        import tempfile
        import os
        start_time = time.time()

        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore[assignment]

        try:
            # Handle media content for vision models like DeepSeek-OCR
            if media and len(media) > 0:
                # Use the first image for OCR
                media_item = media[0]

                # DeepSeek-OCR expects image file path
                if hasattr(media_item, 'file_path') and media_item.file_path:
                    image_file = str(media_item.file_path)
                else:
                    # If no file path, save media content to temp file
                    from PIL import Image

                    if hasattr(media_item, 'content') and media_item.content:
                        # Handle base64 content
                        if media_item.content_format == 'BASE64':
                            import base64
                            image_data = base64.b64decode(media_item.content)
                            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_file.write(image_data)
                            temp_file.close()
                            image_file = temp_file.name
                        else:
                            return GenerateResponse(
                                content="Error: Unsupported media format for DeepSeek-OCR",
                                model=self.model,
                                finish_reason="error"
                            )
                    else:
                        return GenerateResponse(
                            content="Error: No valid image content found",
                            model=self.model,
                            finish_reason="error"
                        )

                # Use DeepSeek-OCR's infer method
                try:
                    # Create temporary output directory for DeepSeek-OCR
                    temp_output_dir = tempfile.mkdtemp()

                    # Patch DeepSeek-OCR for MPS/CPU compatibility if needed
                    if torch is not None and (
                        self.device == "mps"
                        or (self.device is None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                    ):
                        self._patch_deepseek_for_mps()

                    result = self.model_instance.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_file,
                        output_path=temp_output_dir,  # DeepSeek-OCR requires output path
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=False
                    )

                    # Clean up temp output directory
                    import shutil
                    shutil.rmtree(temp_output_dir, ignore_errors=True)

                    # Clean up temp file if created
                    if 'temp_file' in locals() and os.path.exists(image_file):
                        os.unlink(image_file)

                    # Calculate generation time
                    gen_time = (time.time() - start_time) * 1000

                    return GenerateResponse(
                        content=result if isinstance(result, str) else str(result),
                        model=self.model,
                        finish_reason="stop",
                        input_tokens=len(prompt.split()),  # Rough estimate
                        output_tokens=len(str(result).split()) if result else 0,
                        gen_time=gen_time
                    )

                except Exception as e:
                    return GenerateResponse(
                        content=f"Error during DeepSeek-OCR inference: {str(e)}",
                        model=self.model,
                        finish_reason="error"
                    )
            else:
                return GenerateResponse(
                    content="Error: DeepSeek-OCR requires image input",
                    model=self.model,
                    finish_reason="error"
                )

        except Exception as e:
            return GenerateResponse(
                content=f"Error in custom model generation: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _generate_vision_model(self,
                              prompt: str,
                              messages: Optional[List[Dict[str, str]]] = None,
                              system_prompt: Optional[str] = None,
                              tools: Optional[List[Dict[str, Any]]] = None,
                              media: Optional[List['MediaContent']] = None,
                              stream: bool = False,
                              response_model: Optional[Type[BaseModel]] = None,
                              **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using vision model (Glyph, GLM-4.1V, etc.)"""

        import time
        start_time = time.time()

        # Import torch safely
        try:
            import torch
        except ImportError:
            return GenerateResponse(
                content="Error: PyTorch not available for vision model generation",
                model=self.model,
                finish_reason="error",
                gen_time=0.0
            )

        try:
            # Server/gateway sometimes call providers with prompt="" + messages=[...] + media=[...].
            # For multimodal models, the user text and the media must live in the SAME user turn.
            # Best-effort: if prompt is empty, lift the last user message text into the prompt and
            # remove that message from the history to avoid duplication.
            prompt_text = prompt
            messages_for_context = list(messages) if isinstance(messages, list) else None
            if (not isinstance(prompt_text, str) or not prompt_text.strip()) and media and messages_for_context:
                for i in range(len(messages_for_context) - 1, -1, -1):
                    msg = messages_for_context[i] or {}
                    role = str(msg.get("role", "") or "").strip().lower()
                    if role != "user":
                        continue
                    content = msg.get("content", "")
                    lifted = None
                    if isinstance(content, str) and content.strip():
                        lifted = content.strip()
                    elif isinstance(content, list):
                        # OpenAI-style list content: [{"type":"text","text":"..."}, ...]
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            if str(item.get("type", "") or "").strip().lower() == "text":
                                text_val = item.get("text")
                                if isinstance(text_val, str) and text_val.strip():
                                    lifted = text_val.strip()
                                    break
                    if lifted:
                        prompt_text = lifted
                        del messages_for_context[i]
                    break

            # Build messages for vision model
            chat_messages = []

            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})

            if messages_for_context:
                chat_messages.extend(messages_for_context)

            # Build user message with media content
            user_content = []

            # Add text content
            if isinstance(prompt_text, str) and prompt_text.strip():
                user_content.append({"type": "text", "text": prompt_text.strip()})

            # Add media content (images, video)
            has_video = False
            try:
                from ..media.types import MediaType, ContentFormat
            except Exception:
                MediaType = None  # type: ignore[assignment]
                ContentFormat = None  # type: ignore[assignment]

            if media:
                for media_item in media:
                    media_type = getattr(media_item, "media_type", None)

                    # Text markers (e.g. provenance / policy annotations) should be preserved for the model.
                    if MediaType is not None and media_type == MediaType.TEXT:
                        txt = getattr(media_item, "content", None)
                        if isinstance(txt, str) and txt.strip():
                            user_content.append({"type": "text", "text": txt.strip()})
                        continue

                    # Video inputs
                    if MediaType is not None and media_type == MediaType.VIDEO:
                        has_video = True
                        # The actual video content is provided to the processor via `videos=...`;
                        # the chat template only needs a `<video>` placeholder token.
                        user_content.append({"type": "video"})
                        continue

                    # Image inputs
                    if MediaType is None or media_type == MediaType.IMAGE:
                        if getattr(media_item, "file_path", None):
                            user_content.append({"type": "image", "url": str(media_item.file_path)})
                            continue

                        content = getattr(media_item, "content", None)
                        if not content:
                            continue

                        content_format = getattr(media_item, "content_format", None)
                        is_base64 = False
                        if ContentFormat is not None and content_format == ContentFormat.BASE64:
                            is_base64 = True
                        elif isinstance(content_format, str) and content_format.strip().lower() == "base64":
                            is_base64 = True

                        if is_base64:
                            mime_type = getattr(media_item, "mime_type", "image/png")
                            data_url = f"data:{mime_type};base64,{content}"
                            user_content.append({"type": "image", "url": data_url})

            # Add user message
            chat_messages.append({
                "role": "user",
                "content": user_content
            })

            # Process messages using the processor.
            #
            # Some multimodal processors (e.g. LlavaNextVideoProcessor) return a *string*
            # from apply_chat_template; for those we must call the processor separately
            # with explicit images/videos tensors and keep video frame counts bounded.
            if has_video:
                # Resolve max frames for video sampling (keep small to avoid huge context).
                max_frames_raw = kwargs.get("video_max_frames", None)
                if max_frames_raw is None:
                    try:
                        from ..config.manager import get_config_manager

                        cfg_video = getattr(get_config_manager().config, "video", None)
                        max_frames_raw = getattr(cfg_video, "max_frames_native", None) if cfg_video is not None else None
                        if max_frames_raw is None:
                            max_frames_raw = getattr(cfg_video, "max_frames", None) if cfg_video is not None else None
                    except Exception:
                        max_frames_raw = 3
                try:
                    max_video_frames = max(1, int(max_frames_raw))
                except Exception:
                    max_video_frames = 3

                sampling_strategy_raw = kwargs.get("video_sampling_strategy", None)
                if sampling_strategy_raw is None:
                    try:
                        from ..config.manager import get_config_manager

                        sampling_strategy_raw = getattr(get_config_manager().config, "video", None).sampling_strategy  # type: ignore[union-attr]
                    except Exception:
                        sampling_strategy_raw = "uniform"
                sampling_strategy = str(sampling_strategy_raw or "uniform").strip().lower()
                if sampling_strategy not in {"uniform", "keyframes"}:
                    sampling_strategy = "uniform"

                max_frame_side_raw = kwargs.get("video_max_frame_side", None)
                if max_frame_side_raw is None:
                    try:
                        from ..config.manager import get_config_manager

                        max_frame_side_raw = getattr(get_config_manager().config, "video", None).max_frame_side  # type: ignore[union-attr]
                    except Exception:
                        max_frame_side_raw = 1024
                try:
                    max_frame_side = int(max_frame_side_raw) if max_frame_side_raw is not None else None
                except Exception:
                    max_frame_side = 1024
                if isinstance(max_frame_side, int) and max_frame_side <= 0:
                    max_frame_side = None

                # Build multimodal-typed messages for chat_template renderers that expect list content.
                # NOTE: Many HF native-video VLMs are brittle in multi-turn mode if prior turns
                # referenced media but we only retained text history (no `<video>` placeholders).
                # This can cause follow-ups like "and this one?" to over-weight the previous
                # text-only answer and ignore the newly attached video.
                #
                # To make follow-ups robust, collapse prior USER/ASSISTANT turns into a single
                # text block inside the current user message, and keep exactly one `<video>`
                # placeholder (the current attachment) in the chat template input.
                history_lines = []
                if messages_for_context:
                    for msg in messages_for_context:
                        role = str(msg.get("role", "user") or "").strip().lower()
                        # Include mid-stream system turns (e.g. compaction summaries or
                        # context hints) — dropping them silently loses instructions.
                        if role not in {"user", "assistant", "system"}:
                            continue
                        content = msg.get("content", "")
                        text = ""
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            # OpenAI-style list content: [{"type":"text","text":"..."}, ...]
                            for item in content:
                                if not isinstance(item, dict):
                                    continue
                                if str(item.get("type", "") or "").strip().lower() != "text":
                                    continue
                                v = item.get("text")
                                if isinstance(v, str) and v.strip():
                                    text = v
                                    break
                        else:
                            text = str(content)

                        text = str(text or "").strip()
                        if not text:
                            continue
                        prefix = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}[role]
                        history_lines.append(f"{prefix}: {text}")

                if history_lines:
                    history_block = "Prior chat context (text-only):\n" + "\n".join(history_lines) + "\n\n"
                    # Cap to avoid pathological prompt growth; keep the most recent tail.
                    if len(history_block) > 8_000:
                        history_block = "Prior chat context (text-only; truncated):\n…\n" + history_block[-7_800:]
                    user_content = [{"type": "text", "text": history_block}] + list(user_content)

                mm_messages = []
                if system_prompt:
                    mm_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
                mm_messages.append({"role": "user", "content": user_content})

                mm_template_kwargs = self._hf_thinking_template_kwargs(kwargs)
                try:
                    prompt_text = self.processor.apply_chat_template(
                        mm_messages, add_generation_prompt=True, **mm_template_kwargs
                    )
                except TypeError:
                    if mm_template_kwargs:
                        warnings.warn(
                            f"vision processor rejected chat-template kwargs {sorted(mm_template_kwargs)}; "
                            "rendering without them — thinking controls were NOT applied to this request.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    prompt_text = self.processor.apply_chat_template(mm_messages, add_generation_prompt=True)

                # Prepare explicit video inputs for the processor.
                #
                # Prefer ffmpeg-sampled frames (our own extraction) over relying on torchvision/torchcodec
                # decoding inside Transformers, which can vary by platform/codec support (notably for .mov).
                video_paths = []
                image_inputs = []
                for media_item in (media or []):
                    if MediaType is not None and getattr(media_item, "media_type", None) == MediaType.VIDEO:
                        video_path = getattr(media_item, "file_path", None) or getattr(media_item, "content", None)
                        if not isinstance(video_path, str) or not video_path.strip():
                            raise ValueError("Video MediaContent must provide file_path for HuggingFace video models.")
                        video_paths.append(video_path)
                    elif MediaType is not None and getattr(media_item, "media_type", None) == MediaType.IMAGE:
                        fp = getattr(media_item, "file_path", None)
                        if isinstance(fp, str) and fp.strip():
                            try:
                                from PIL import Image as PILImage
                            except ImportError as e:
                                raise RuntimeError(f"PIL is required for HuggingFace image inputs: {e}")
                            image_inputs.append(PILImage.open(fp).convert("RGB"))

                processor_call: Dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
                if image_inputs:
                    processor_call["images"] = image_inputs if len(image_inputs) > 1 else image_inputs[0]
                if video_paths:
                    # Try ffmpeg frame sampling first.
                    video_frame_inputs = []
                    temp_dirs = []
                    try:
                        from pathlib import Path
                        import tempfile

                        from ..media.utils.video_frames import extract_video_frames
                        from PIL import Image as PILImage

                        for vp in video_paths:
                            out_dir = Path(tempfile.mkdtemp(prefix="abstractcore_hf_video_frames_"))
                            temp_dirs.append(out_dir)
                            frames, _timestamps_s = extract_video_frames(
                                Path(vp),
                                max_frames=max_video_frames,
                                frame_format="jpg",
                                sampling_strategy=sampling_strategy,
                                max_side=max_frame_side,
                                output_dir=out_dir,
                            )
                            if not frames:
                                raise RuntimeError("No frames extracted")
                            video_frame_inputs.append([PILImage.open(p).convert("RGB") for p in frames])

                        # Single video -> pass list[PIL]; multiple videos -> list[list[PIL]]
                        processor_call["videos"] = (
                            video_frame_inputs[0]
                            if len(video_frame_inputs) == 1
                            else video_frame_inputs
                        )
                    except Exception:
                        # If anything goes wrong with ffmpeg sampling, fall back to transformers decode.
                        processor_call["videos"] = video_paths if len(video_paths) > 1 else video_paths[0]
                        processor_call["videos_kwargs"] = {"do_sample_frames": True, "num_frames": max_video_frames}
                    finally:
                        # Cleanup extracted frames directories (frames are already loaded into memory as PIL).
                        for d in temp_dirs:
                            try:
                                import shutil

                                shutil.rmtree(d, ignore_errors=True)
                            except Exception:
                                pass

                inputs = self.processor(**processor_call)
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.model_instance.device)
            else:
                mm_template_kwargs = self._hf_thinking_template_kwargs(kwargs)
                try:
                    templated = self.processor.apply_chat_template(
                        chat_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                        **mm_template_kwargs,
                    )
                except TypeError:
                    if mm_template_kwargs:
                        warnings.warn(
                            f"vision processor rejected chat-template kwargs {sorted(mm_template_kwargs)}; "
                            "rendering without them — thinking controls were NOT applied to this request.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    templated = self.processor.apply_chat_template(
                        chat_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                if isinstance(templated, str):
                    # Processor returned a prompt string; fall back to explicit processor call.
                    image_inputs = []
                    for media_item in (media or []):
                        if MediaType is not None and getattr(media_item, "media_type", None) == MediaType.IMAGE:
                            fp = getattr(media_item, "file_path", None)
                            if isinstance(fp, str) and fp.strip():
                                try:
                                    from PIL import Image as PILImage
                                except ImportError as e:
                                    raise RuntimeError(f"PIL is required for HuggingFace image inputs: {e}")
                                image_inputs.append(PILImage.open(fp).convert("RGB"))

                    processor_call: Dict[str, Any] = {"text": templated, "return_tensors": "pt"}
                    if image_inputs:
                        processor_call["images"] = image_inputs if len(image_inputs) > 1 else image_inputs[0]
                    inputs = self.processor(**processor_call)
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.model_instance.device)
                else:
                    inputs = templated.to(self.model_instance.device)

            temperature_value = kwargs.get("temperature", self.temperature)
            # For HF multimodal video models, default to greedy decoding unless the caller explicitly
            # provided a temperature. This avoids premature EOS producing unusably short answers.
            if has_video and ("temperature" in kwargs) and kwargs.get("temperature") is None:
                temperature_value = 0.0
            if temperature_value is None:
                temperature_value = self.temperature

            max_new_tokens_raw = kwargs.get("max_output_tokens", None)
            if max_new_tokens_raw is None:
                max_new_tokens_raw = kwargs.get("max_tokens", None)
            if max_new_tokens_raw is None:
                max_new_tokens_raw = self.max_output_tokens or 512
            try:
                max_new_tokens_value = max(1, int(max_new_tokens_raw))
            except Exception:
                max_new_tokens_value = int(self.max_output_tokens or 512)

            do_sample = True
            try:
                if temperature_value is None or float(temperature_value) <= 0:
                    do_sample = False
                    temperature_value = 0.0
            except Exception:
                do_sample = True

            generation_kwargs = {
                "max_new_tokens": max_new_tokens_value,
                "do_sample": do_sample,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature_value
                top_p_value = kwargs.get("top_p", getattr(self, "top_p", None))
                if top_p_value is not None:
                    generation_kwargs["top_p"] = top_p_value
                top_k_value = kwargs.get("top_k", getattr(self, "top_k", None))
                if top_k_value is not None:
                    generation_kwargs["top_k"] = int(top_k_value)

            # Add seed if provided
            seed_value = self._normalize_seed(kwargs.get("seed", self.seed))
            if seed_value is not None:
                torch.manual_seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)

            # Runaway-loop detection. `generate()` is a closed loop, so a
            # StoppingCriteria is the only place a detector can observe the
            # stream. Never a cap: it stops on evidence of a verbatim repeating
            # cycle and returns everything produced before it.
            _rep_detector = _degeneration.attach_to_generation_kwargs(generation_kwargs)

            # Generate response
            generated_ids = None
            try:
                with torch.inference_mode():
                    use_mps_lock = str(getattr(self, "device", "") or "").strip().lower() == "mps"
                    if use_mps_lock:
                        with _MPS_GENERATION_LOCK:
                            generated_ids = self.model_instance.generate(**inputs, **generation_kwargs)
                    else:
                        generated_ids = self.model_instance.generate(**inputs, **generation_kwargs)
            except RuntimeError as e:
                if str(getattr(self, "device", "") or "").strip().lower() == "mps":
                    raise RuntimeError(
                        "HuggingFaceProvider vision/video generation failed on MPS. "
                        "If this persists, force CPU via ABSTRACTCORE_HF_DEVICE=cpu."
                    ) from e
                raise
            finally:
                # Best-effort: keep MPS memory pressure low between calls.
                try:
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            if hasattr(torch.mps, "synchronize"):
                                torch.mps.synchronize()
                            torch.mps.empty_cache()
                except Exception:
                    pass
                try:
                    import gc

                    gc.collect()
                except Exception:
                    pass

            # Decode response
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )

            # Calculate generation time
            gen_time = (time.time() - start_time) * 1000

            # Calculate token usage
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = len(generated_ids[0]) - input_tokens

            # A degenerate stop must be distinguishable from a natural one and
            # from budget exhaustion — collapsing all three into "stop" is the
            # information loss this detector exists to remove.
            if _rep_detector is not None and _rep_detector.tripped:
                _rep_detector.warn(self.model)

            response = GenerateResponse(
                content=output_text.strip(),
                model=self.model,
                finish_reason=(
                    _rep_detector.finish_reason("stop")
                    if _rep_detector is not None else "stop"
                ),
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens
                },
                gen_time=gen_time
            )
            if _rep_detector is not None and _rep_detector.tripped:
                # The caller needs the cycle length, the repeat count and where
                # it started — enough to tell a real loop from a false stop.
                md = dict(getattr(response, "metadata", None) or {})
                md.update(_rep_detector.metadata())
                response.metadata = md
            if stream:
                def _single_chunk_stream() -> Iterator[GenerateResponse]:
                    yield response
                return _single_chunk_stream()
            return response

        except Exception as e:
            gen_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0.0
            error_resp = GenerateResponse(
                content=f"Error in vision model generation: {str(e)}",
                model=self.model,
                finish_reason="error",
                gen_time=gen_time
            )
            if stream:
                def _error_stream() -> Iterator[GenerateResponse]:
                    yield error_resp
                return _error_stream()
            return error_resp

    def _patch_deepseek_for_mps(self):
        """Patch DeepSeek-OCR model to work with MPS instead of CUDA"""
        import types

        def patched_infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
            """Patched infer method that uses MPS instead of CUDA"""
            import torch

            # Determine the best available device
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            # Call the original infer method but patch tensor.cuda() calls
            original_cuda = torch.Tensor.cuda

            def patched_cuda(tensor, device=None, non_blocking=False, **kwargs):
                """Redirect .cuda() calls to the appropriate device"""
                if device == 'mps' or (device is None and torch.backends.mps.is_available()):
                    return tensor.to('mps', non_blocking=non_blocking)
                elif torch.cuda.is_available():
                    return original_cuda(tensor, device, non_blocking, **kwargs)
                else:
                    return tensor.to('cpu', non_blocking=non_blocking)

            # Temporarily patch the cuda method
            torch.Tensor.cuda = patched_cuda

            try:
                # Move model to the appropriate device first
                self.to(device)

                # Call original infer with device patching
                return self._original_infer(tokenizer, prompt, image_file, output_path, base_size, image_size, crop_mode, test_compress, save_results, eval_mode)
            finally:
                # Restore original cuda method
                torch.Tensor.cuda = original_cuda

        # Only patch if not already patched
        if not hasattr(self.model_instance, '_original_infer'):
            self.model_instance._original_infer = self.model_instance.infer
            self.model_instance.infer = types.MethodType(patched_infer, self.model_instance)

    def _generate_gguf(self,
                       prompt: str,
                       messages: Optional[List[Dict[str, str]]] = None,
                       system_prompt: Optional[str] = None,
                       tools: Optional[List[Dict[str, Any]]] = None,
                       media: Optional[List['MediaContent']] = None,
                       stream: bool = False,
                       response_model: Optional[Type[BaseModel]] = None,
                       **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using GGUF backend with llama-cpp-python"""

        if not self.llm:
            return GenerateResponse(
                content="Error: GGUF model not loaded",
                model=self.model,
                finish_reason="error"
            )

        # Handle media content for the user message - use proper vision format for GGUF models
        media_enrichment = None
        if media:
            try:
                from ..architectures.detection import supports_vision

                # Check if this model supports vision natively
                if supports_vision(self.model):
                    # Use HuggingFace multimodal format for vision-capable GGUF models
                    user_message_content = []

                    # Add text content
                    user_message_content.append({"type": "text", "text": prompt})

                    # Add media content
                    for media_item in media:
                        if hasattr(media_item, 'file_path') and media_item.file_path:
                            # Use file:// URL format as specified in HuggingFace docs
                            file_path = str(media_item.file_path)
                            if not file_path.startswith('file://'):
                                file_path = f"file://{file_path}"
                            user_message_content.append({
                                "type": "image",
                                "image": file_path
                            })
                        elif hasattr(media_item, 'content') and media_item.content:
                            # For base64 or other content, we might need to save to temp file
                            import tempfile
                            import base64
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                                if isinstance(media_item.content, str) and media_item.content.startswith('data:'):
                                    # Handle base64 data URLs
                                    header, data = media_item.content.split(',', 1)
                                    decoded_data = base64.b64decode(data)
                                    tmp_file.write(decoded_data)
                                else:
                                    tmp_file.write(media_item.content)
                                tmp_file.flush()
                                user_message_content.append({
                                    "type": "image",
                                    "image": f"file://{tmp_file.name}"
                                })
                else:
                    # Fallback to text-based media handling for non-vision models
                    from ..media.handlers import LocalMediaHandler
                    media_handler = LocalMediaHandler("huggingface", self.model_capabilities, model_name=self.model)
                    multimodal_message = media_handler.create_multimodal_message(prompt, media)
                    media_enrichment = getattr(media_handler, "media_enrichment", None)
                    user_message_content = multimodal_message if isinstance(multimodal_message, str) else prompt

            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                user_message_content = prompt
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                user_message_content = prompt
        else:
            user_message_content = prompt

        chat_messages = self._gguf_build_chat_messages(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            user_message_content=user_message_content,
        )
        gguf_enable_thinking = kwargs.get("_acore_gguf_enable_thinking")
        gguf_enable_thinking = gguf_enable_thinking if isinstance(gguf_enable_thinking, bool) else None

        # Prompt caching (GGUF/llama.cpp): best-effort per-key cache selection.
        cache_obj = None
        cache_state = None
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            key = prompt_cache_key.strip()
            cache_value = self._prompt_cache_store.get(key)
            if cache_value is None:
                self.prompt_cache_set(key, make_default=False)
                cache_value = self._prompt_cache_store.get(key)
            cache_state = self._gguf_prompt_cache_state(cache_value)
            cache_obj = self._gguf_prompt_cache_unwrap(cache_value)
            try:
                if cache_obj is not None and hasattr(self.llm, "set_cache"):
                    self.llm.set_cache(cache_obj)
            except Exception:
                pass
        else:
            # Disable cache for this request when no key is provided.
            try:
                if hasattr(self.llm, "set_cache"):
                    self.llm.set_cache(None)
            except Exception:
                pass

        # Prepare parameters using unified system
        unified_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(unified_kwargs)

        generation_kwargs = {
            "messages": chat_messages,
            "max_tokens": max_output_tokens,  # This is max_output_tokens for llama-cpp
            "temperature": unified_kwargs.get("temperature", self.temperature),
            "top_p": unified_kwargs.get("top_p", 0.9),
            "stream": stream
        }

        # Add seed if provided (GGUF/llama-cpp supports seed)
        seed_value = unified_kwargs.get("seed")
        if seed_value is not None:
            generation_kwargs["seed"] = seed_value

        # Add native structured output support (llama-cpp-python format)
        # llama-cpp-python supports native structured outputs using the response_format parameter
        # This provides server-side guaranteed schema compliance
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            generation_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        # Handle tools - both native and prompted support
        has_native_tools = False
        if tools:
            # Check if model supports native tools - but fall back to prompted for now
            # TODO: Re-enable native tools once parameter default handling is fixed
            if False and self.llm.chat_format in ["chatml-function-calling", "functionary-v2"]:
                # Use unified tool handler for consistent formatting
                openai_tools = self.tool_handler.prepare_tools_for_native(tools)
                generation_kwargs["tools"] = openai_tools

                # Debug: Print what we're sending to the model
                print(f"DEBUG: Sending tools to HuggingFace model (unified handler):")
                import json
                print(json.dumps(openai_tools, indent=2))

                # Don't use auto for streaming (limitation of llama-cpp-python)
                if not stream:
                    generation_kwargs["tool_choice"] = "auto"
                has_native_tools = True

        try:
            # GGUF local control-plane generation: use cached state snapshots + `llm.generate(reset=False)`
            # to avoid llama-cpp-python's `create_chat_completion()` resetting and re-evaluating the full prompt.
            #
            # ALSO the lane for KEYLESS thinking-controlled requests: the fallback lane's
            # trailing-assistant no-think marker renders as a CLOSED turn through
            # create_chat_completion and measurably does not disable thinking (live find
            # 2026-08-19: 113-1359 reasoning chars behind an "off" claim on three models),
            # and effort levels have no transport there at all. The control-plane render
            # owns the bytes, so both controls are real — route thinking-controlled text
            # requests here even with no cache key (cache_obj=None: prefill runs without
            # snapshot stores).
            gguf_reasoning_effort = kwargs.get("_acore_gguf_reasoning_effort")
            gguf_reasoning_effort = gguf_reasoning_effort if isinstance(gguf_reasoning_effort, str) else None
            thinking_controlled = gguf_enable_thinking is False or gguf_reasoning_effort is not None
            control_plane_enabled = (
                (cache_obj is not None or thinking_controlled)
                and self._gguf_prompt_cache_supports_local_control_plane()
                and os.environ.get("ABSTRACTCORE_GGUF_CONTROL_PLANE", "1").strip().lower() not in {"0", "false", "no", "off"}
                and response_model is None
                and not has_native_tools
                and self._gguf_control_plane_can_stream(chat_messages)
            )
            if control_plane_enabled:
                return self._gguf_control_plane_generate(
                    chat_messages=chat_messages,
                    cache_obj=cache_obj,
                    max_output_tokens=int(max_output_tokens),
                    temperature=float(generation_kwargs.get("temperature") or 0.2),
                    top_p=float(generation_kwargs.get("top_p") or 0.95),
                    top_k=int(generation_kwargs.get("top_k", 40) or 40),
                    min_p=float(kwargs.get("min_p", 0.05) or 0.05),
                    typical_p=float(kwargs.get("typical_p", 1.0) or 1.0),
                    repeat_penalty=float(kwargs.get("repeat_penalty", 1.1) or 1.1),
                    presence_penalty=float(kwargs.get("presence_penalty", 0.0) or 0.0),
                    frequency_penalty=float(kwargs.get("frequency_penalty", 0.0) or 0.0),
                    tfs_z=float(kwargs.get("tfs_z", 1.0) or 1.0),
                    mirostat_mode=int(kwargs.get("mirostat_mode", 0) or 0),
                    mirostat_tau=float(kwargs.get("mirostat_tau", 5.0) or 5.0),
                    mirostat_eta=float(kwargs.get("mirostat_eta", 0.1) or 0.1),
                    seed=seed_value,
                    stream=bool(stream),
                    enable_thinking=gguf_enable_thinking,
                    reasoning_effort=gguf_reasoning_effort,
                    cache_state=cache_state,
                    cache_key=(
                        prompt_cache_key.strip()
                        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip()
                        else None
                    ),
                )

            marker = self._thinking_disable_prefill(gguf_enable_thinking)
            if marker:
                fallback_messages = copy.deepcopy(chat_messages)
                if not (
                    fallback_messages
                    and isinstance(fallback_messages[-1], dict)
                    and str(fallback_messages[-1].get("role") or "").strip().lower() == "assistant"
                    and str(fallback_messages[-1].get("content") or "") == marker
                ):
                    fallback_messages.append({"role": "assistant", "content": marker})
                generation_kwargs["messages"] = fallback_messages

            # Fallback lane (create_chat_completion → the model's embedded Jinja
            # chat template): bridge the wire-string vs template-dict `arguments`
            # convention so replayed tool-call history does not crash
            # dict-expecting templates (`arguments|items`). The control-plane
            # lane above already returned; this only touches the delegated path.
            _fallback_messages = generation_kwargs.get("messages")
            if isinstance(_fallback_messages, list):
                generation_kwargs["messages"] = self._gguf_normalize_tool_call_arguments_for_template(
                    _fallback_messages
                )

            if stream:
                return self._stream_generate_gguf_with_tools(generation_kwargs, tools, has_native_tools, kwargs.get('tool_call_tags'))
            else:
                response = self._single_generate_gguf(generation_kwargs)
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

                # Handle tool execution for both native and prompted responses
                if tools and (response.has_tool_calls() or
                             (self.tool_handler.supports_prompted and response.content)):
                    response = self._handle_tool_execution_gguf(response, tools, has_native_tools)

                return response

        except Exception as e:
            error_message = str(e)
            if stream:
                # Return error as a generator
                def error_generator():
                    yield GenerateResponse(
                        content=f"Error: {error_message}",
                        model=self.model,
                        finish_reason="error"
                    )
                return error_generator()
            else:
                return GenerateResponse(
                    content=f"Error: {error_message}",
                    model=self.model,
                    finish_reason="error"
                )

    def _single_generate_gguf(self, kwargs: Dict[str, Any]) -> GenerateResponse:
        """Generate single response using GGUF"""
        response = self.llm.create_chat_completion(**kwargs)

        choice = response['choices'][0]
        message = choice['message']

        # Extract tool calls if present
        tool_calls = None
        if 'tool_calls' in message:
            tool_calls = []
            for tc in message['tool_calls']:
                tool_calls.append({
                    "id": tc.get('id'),
                    "type": tc.get('type', 'function'),
                    "name": tc['function']['name'],
                    "arguments": tc['function']['arguments']
                })

        # Extract usage (normalized + legacy key spellings, cross-provider parity)
        usage = None
        if 'usage' in response:
            _pt = response['usage'].get('prompt_tokens', 0)
            _ct = response['usage'].get('completion_tokens', 0)
            usage = {
                "input_tokens": _pt,
                "output_tokens": _ct,
                "total_tokens": response['usage'].get('total_tokens', 0),
                "prompt_tokens": _pt,
                "completion_tokens": _ct,
            }

        # Fix HTML escaping in llama-cpp-python responses
        content = message.get('content', '')
        if content:
            import html
            content = html.unescape(content)

        return GenerateResponse(
            content=content,
            raw_response=response,
            model=self.model,
            finish_reason=choice.get('finish_reason', 'stop'),
            usage=usage,
            tool_calls=tool_calls
        )

    def _gguf_control_plane_stop_strings(self) -> List[str]:
        """Return stop strings for GGUF local control-plane generation."""
        chat_format = self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
        fmt = str(chat_format or "").strip().lower()
        if fmt == "llama-3":
            return ["<|eot_id|>"]
        if fmt == "llama-cpp-chat-template":
            stops: List[str] = []
            try:
                eos = self._gguf_model_token_text(int(self.llm.token_eos()))
                if eos:
                    stops.append(eos)
            except Exception:
                pass
            cfg = getattr(self, "architecture_config", None)
            if isinstance(cfg, dict):
                suffix = str(cfg.get("assistant_suffix") or "").strip()
                if suffix and suffix not in stops:
                    stops.append(suffix)
            return stops or ["<turn|>"]
        # ChatML and chatml-function-calling.
        return ["<|im_end|>"]

    def _gguf_control_plane_can_stream(self, chat_messages: List[Dict[str, Any]]) -> bool:
        """Return True when control-plane streaming can safely handle the message payloads."""
        # Control-plane renderer/tokenizer only supports text content (strings / JSON-serializable).
        for msg in chat_messages or []:
            if not isinstance(msg, dict):
                return False
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"system", "user", "assistant"}:
                return False
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                continue
            # For now, fall back to llama-cpp-python's chat completion for multimodal payloads.
            return False
        return True

    def _gguf_control_plane_stream_generate(
        self,
        *,
        chat_messages: List[Dict[str, Any]],
        cache_obj: Any,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        typical_p: float,
        repeat_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        tfs_z: float,
        mirostat_mode: int,
        mirostat_tau: float,
        mirostat_eta: float,
        seed: Optional[int],
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        cache_state: Optional[_GGUFPromptCacheValue] = None,
        cache_key: Optional[str] = None,
    ) -> Iterator[GenerateResponse]:
        """Generate GGUF text by prefilling cached KV state and sampling from it.

        This bypasses llama-cpp-python's `create_chat_completion()` so we can benefit from
        cached state snapshots even when llama.cpp does not support incremental KV trimming.
        Also the lane that serves KEYLESS thinking-controlled requests (cache_obj=None):
        it is the only GGUF path whose render owns the control artifacts.
        """
        llm = getattr(self, "llm", None)
        if llm is None:
            yield GenerateResponse(
                content="Error: GGUF model not loaded",
                model=self.model,
                finish_reason="error",
            )
            return

        stop_strs = [s for s in (self._gguf_control_plane_stop_strings() or []) if isinstance(s, str) and s]
        flush_threshold = 160

        # Render inside the generator body must DEGRADE, not raise: a template
        # that refuses a conversation shape (e.g. the Ornith template's
        # raise_exception on a mid-history system message) used to surface as
        # a raw ValueError at the consumer's first next() on the streaming
        # lane, while the fallback lane degraded gracefully (adversary F3,
        # 2026-07-19). Mirror the non-stream error shape.
        try:
            prompt_text, prompt_tokens = self._gguf_render_prompt_tokens(
                messages=chat_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {e}",
                model=self.model,
                finish_reason="error",
            )
            return
        prompt_text, prompt_tokens, prompt_cache_meta = self._gguf_compose_cached_prompt_tokens(
            cache_state=cache_state,
            live_prompt_text=prompt_text,
            live_prompt_tokens=prompt_tokens,
        )

        # Bring the context to `prompt_tokens`, preferring llama.cpp's RESIDENT KV
        # over a stored snapshot (see `_gguf_prefill_prompt_cache`). In a
        # growing-prefix loop the resident context is always the better source: it
        # already holds the previous turn's prompt AND its reply, so turn N+1 shares
        # a longer prefix with it than with any snapshot taken before that reply
        # existed.
        #
        # `save_state_on_live_reuse=False` therefore stops the snapshot being paid
        # for on exactly the turns it cannot help. The earlier note here claimed
        # `save_state=True` was what restored reuse on this lane; hardware refuted
        # that — across a 4-turn growing session `load_state` fired ZERO times while
        # `save_state` was paid EVERY call at 1.41 GB. The snapshot remains for the
        # cold/divergent turn, where it is the only thing that can help.
        #
        # set_cache stays False: the low-level `llm.generate([], reset=False)` below
        # does not consult `llm.cache`, so attaching it would be inert.
        #
        # `snapshot_at_boundary=True` is the AGENT-LOOP repair: store the state
        # before this turn's volatile tail so turn i's boundary is still a true
        # prefix of turn i+1's prompt. The boundary comes from the previous turn's
        # recorded prompt (LCP holdback) once there is one, and from the renderer
        # itself on turn 1 — never from a hardcoded tail literal.
        #
        # `ABSTRACTCORE_GGUF_BOUNDARY_SNAPSHOT=0` restores the pre-2026-08-05
        # full-prompt snapshot policy. It exists so the repair stays MEASURABLE
        # in one process against the policy it replaced (the A/B that produced
        # the numbers above), and as an escape hatch — same shape as the existing
        # `ABSTRACTCORE_GGUF_CONTROL_PLANE` switch. It does NOT disable the
        # zero-feed guard, which is a correctness invariant and not a policy.
        boundary_enabled = os.environ.get(
            "ABSTRACTCORE_GGUF_BOUNDARY_SNAPSHOT", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        prev_prompt_tokens: tuple[int, ...] = ()
        if cache_state is not None:
            prev_prompt_tokens = tuple(int(t) for t in (cache_state.fed_prompt_tokens or ()))
        generation_boundary: Optional[int] = None
        if boundary_enabled and not prev_prompt_tokens and not prompt_cache_meta.get("prompt_cache_composed"):
            # Composed prompts (durable-bloc prefix + live suffix) are excluded:
            # their head is not what `chat_messages` renders, so the re-render
            # would diverge early and the LCP would understate the boundary.
            generation_boundary = self._gguf_generation_prompt_boundary(
                messages=chat_messages,
                prompt_tokens=prompt_tokens,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        # Same STRUCTURE as the MLX and transformers lanes (parity, 2026-08-07):
        # mode/key identify the cache, outcome/cached_tokens/fed_tokens describe
        # the call. This lane already reported the decision fields; it did not
        # say WHICH cache it was reporting on, so a reader with more than one
        # session in flight could not attribute a row to a key.
        cache_telemetry: Dict[str, Any] = {"mode": "key", "key": cache_key} if cache_key else {}
        ok = self._gguf_prefill_prompt_cache(
            cache_obj,
            prompt_tokens,
            save_state=cache_obj is not None,
            save_state_on_live_reuse=False,
            set_cache=False,
            snapshot_at_boundary=boundary_enabled,
            prev_prompt_tokens=prev_prompt_tokens,
            generation_boundary=generation_boundary,
            protect_snapshot_key=(cache_state.prompt_tokens if cache_state is not None else ()),
            telemetry=cache_telemetry,
        )
        if not ok:
            yield GenerateResponse(
                content="Error: failed to prefill GGUF prompt cache",
                model=self.model,
                finish_reason="error",
            )
            return
        if cache_state is not None:
            cache_state.fed_prompt_tokens = tuple(int(t) for t in prompt_tokens)

        # Best-effort determinism.
        if seed is not None:
            try:
                llm.set_seed(int(seed))
            except Exception:
                pass

        # Prefer stop detection by token id because special tokens (e.g. `<|im_end|>`) often
        # detokenize to empty bytes.
        stop_token_seqs: List[tuple[int, ...]] = []
        for s in stop_strs:
            try:
                toks = llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)
                seq = tuple(int(t) for t in toks)
                if seq:
                    stop_token_seqs.append(seq)
            except Exception:
                continue

        max_stop_seq_len = max((len(seq) for seq in stop_token_seqs), default=0)
        recent_tokens: List[int] = []

        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")()
        pending = ""
        output_tokens = 0
        finish_reason = "stop"

        try:
            for tok in llm.generate(
                [],
                top_k=int(top_k),
                top_p=float(top_p),
                min_p=float(min_p),
                typical_p=float(typical_p),
                temp=float(temperature),
                repeat_penalty=float(repeat_penalty),
                frequency_penalty=float(frequency_penalty),
                presence_penalty=float(presence_penalty),
                tfs_z=float(tfs_z),
                mirostat_mode=int(mirostat_mode),
                mirostat_tau=float(mirostat_tau),
                mirostat_eta=float(mirostat_eta),
                reset=False,
            ):
                tok_i = int(tok)

                # Stop token detection (token-id based).
                if stop_token_seqs:
                    recent_tokens.append(tok_i)
                    if max_stop_seq_len and len(recent_tokens) > max_stop_seq_len:
                        recent_tokens = recent_tokens[-max_stop_seq_len:]
                    if any(
                        len(seq) <= len(recent_tokens) and tuple(recent_tokens[-len(seq) :]) == seq
                        for seq in stop_token_seqs
                    ):
                        finish_reason = "stop"
                        break

                output_tokens += 1
                if isinstance(max_output_tokens, int) and max_output_tokens > 0 and output_tokens > int(max_output_tokens):
                    finish_reason = "length"
                    break

                try:
                    token_bytes = llm.detokenize([tok_i])
                except Exception:
                    token_bytes = b""

                if token_bytes:
                    pending += decoder.decode(token_bytes)

                if len(pending) > flush_threshold:
                    yield GenerateResponse(content=pending, model=self.model)
                    pending = ""

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error",
            )
            return

        # Flush decoder and any remaining buffered content.
        try:
            pending += decoder.decode(b"", final=True) or ""
        except Exception:
            pass

        if pending:
            yield GenerateResponse(content=pending, model=self.model)

        completion_tokens = int(output_tokens)
        if finish_reason == "length" and isinstance(max_output_tokens, int) and max_output_tokens > 0:
            completion_tokens = int(max_output_tokens)

        usage = {
            "input_tokens": int(len(prompt_tokens)),
            "output_tokens": completion_tokens,
            "total_tokens": int(len(prompt_tokens) + completion_tokens),
            "prompt_tokens": int(len(prompt_tokens)),
            "completion_tokens": completion_tokens,
        }

        yield GenerateResponse(
            content="",
            model=self.model,
            finish_reason=finish_reason,
            usage=usage,
            metadata={
                "_acore_backend": "gguf_control_plane",
                # MEASURED reuse for this call (MLX-lane parity: same key names,
                # same meaning). `fed_tokens` is the number of prompt tokens this
                # call actually pushed through a forward pass — the ground truth
                # the estimate in `usage.input_tokens` cannot give.
                "prompt_cache": dict(cache_telemetry),
                **prompt_cache_meta,
            },
        )

    def _gguf_control_plane_generate(
        self,
        *,
        chat_messages: List[Dict[str, Any]],
        cache_obj: Any,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        typical_p: float,
        repeat_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        tfs_z: float,
        mirostat_mode: int,
        mirostat_tau: float,
        mirostat_eta: float,
        seed: Optional[int],
        stream: bool,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        cache_state: Optional[_GGUFPromptCacheValue] = None,
        cache_key: Optional[str] = None,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        if stream:
            return self._gguf_control_plane_stream_generate(
                chat_messages=chat_messages,
                cache_obj=cache_obj,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                seed=seed,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                cache_state=cache_state,
                cache_key=cache_key,
            )

        collected = ""
        last: Optional[GenerateResponse] = None
        for chunk in self._gguf_control_plane_stream_generate(
            chat_messages=chat_messages,
            cache_obj=cache_obj,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            seed=seed,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            cache_state=cache_state,
            cache_key=cache_key,
        ):
            last = chunk
            if isinstance(chunk.content, str) and chunk.content:
                collected += chunk.content

        return GenerateResponse(
            content=collected,
            model=self.model,
            finish_reason=getattr(last, "finish_reason", None) if last is not None else "stop",
            usage=getattr(last, "usage", None) if last is not None else None,
            metadata=getattr(last, "metadata", None) if last is not None else None,
        )

    def _stream_generate_gguf(self, kwargs: Dict[str, Any], tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Stream response using GGUF with tool tag rewriting support"""
        stream = self.llm.create_chat_completion(**kwargs)

        current_tool_call = None
        accumulated_arguments = ""

        # Initialize tool tag rewriter if needed
        rewriter = None
        buffer = ""
        if tool_call_tags:
            try:
                from ..tools.tag_rewriter import create_tag_rewriter
                rewriter = create_tag_rewriter(tool_call_tags)
            except ImportError:
                pass

        for chunk in stream:
            if 'choices' not in chunk or not chunk['choices']:
                continue

            choice = chunk['choices'][0]
            delta = choice.get('delta', {})

            # Handle text content
            if 'content' in delta and delta['content']:
                # Fix HTML escaping in streaming content
                content = delta['content']
                if content:
                    import html
                    content = html.unescape(content)

                    # Apply tool tag rewriting if enabled
                    if rewriter:
                        rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                        content = rewritten_content

                yield GenerateResponse(
                    content=content,
                    model=self.model,
                    finish_reason=choice.get('finish_reason')
                )

            # Handle tool calls
            if 'tool_calls' in delta:
                for tc in delta['tool_calls']:
                    if 'function' in tc:
                        if tc.get('id'):  # New tool call
                            if current_tool_call and accumulated_arguments:
                                # Yield the previous tool call
                                current_tool_call['arguments'] = accumulated_arguments
                                yield GenerateResponse(
                                    content="",
                                    model=self.model,
                                    tool_calls=[current_tool_call]
                                )

                            # Start new tool call
                            current_tool_call = {
                                "id": tc.get('id'),
                                "type": tc.get('type', 'function'),
                                "name": tc['function'].get('name'),
                                "arguments": ""
                            }
                            accumulated_arguments = tc['function'].get('arguments', '')
                        else:
                            # Accumulate arguments
                            if current_tool_call:
                                accumulated_arguments += tc['function'].get('arguments', '')

            # Handle finish reason
            if choice.get('finish_reason'):
                # Yield any pending tool call
                if current_tool_call and accumulated_arguments:
                    current_tool_call['arguments'] = accumulated_arguments
                    yield GenerateResponse(
                        content="",
                        model=self.model,
                        finish_reason=choice['finish_reason'],
                        tool_calls=[current_tool_call]
                    )
                else:
                    yield GenerateResponse(
                        content="",
                        model=self.model,
                        finish_reason=choice['finish_reason']
                    )

    def _single_generate_transformers_cached(
        self,
        *,
        prompt: str,
        prompt_cache_key: str,
        messages: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        tools: Optional[List[Any]],
        prefilled_modules: Any,
        max_new_tokens: int,
        temperature: Optional[float],
        top_p: float,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> GenerateResponse:
        """Generate a single response using a transformers KV cache keyed by `prompt_cache_key`."""

        if not isinstance(prompt_cache_key, str) or not prompt_cache_key.strip():
            raise ValueError("prompt_cache_key must be a non-empty string")

        if not self._transformers_prompt_cache_supported():
            raise ValueError("Transformers prompt caching is not available for this model/provider instance.")

        try:
            import torch  # type: ignore
        except Exception as e:
            raise ImportError("Transformers prompt caching requires `torch`.") from e

        if getattr(self, "model_instance", None) is None or getattr(self, "tokenizer", None) is None:
            raise RuntimeError("Transformers model/tokenizer not loaded")

        key = prompt_cache_key.strip()
        cache_value = self._prompt_cache_store.get(key)
        if cache_value is None:
            self.prompt_cache_set(key, make_default=False)
            cache_value = self._prompt_cache_store.get(key)

        state = self._transformers_prompt_cache_state(cache_value)
        if state is None:
            raise RuntimeError("prompt cache key does not reference a transformers cache state")

        # Caller-shape discriminator (parity with the MLX delta lane):
        # `messages is not None` = FULL-CONTEXT caller re-sending the whole
        # logical transcript every call (the runtime/ReAct shape). Warm calls
        # must LCP against the recorded tokens, crop the cache to the shared
        # prefix, and feed ONLY the suffix — the pre-fix behavior ignored
        # `messages` on warm calls entirely, so the model answered LAST
        # call's question over a stale context (wrong content, no savings).
        # Prompt-only callers keep the append lane below, unchanged.
        full_context = messages is not None

        # Seed for determinism (best-effort).
        if seed is not None:
            try:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            except Exception:
                pass

        start_time = time.time()

        # Surfaces as response.metadata["prompt_cache"] with the MLX lane's
        # vocabulary and the MLX lane's STRUCTURE: mode/key identify the cache,
        # outcome/cached_tokens/fed_tokens describe what this call actually did,
        # degraded_reason explains any fallback. Seeded here (not per-lane) so
        # every lane through this method reports — crop, snapshot and append
        # alike. An empty struct would omit the key from metadata entirely,
        # which is what made the crop lane invisible before 2026-08-07.
        cache_telemetry: Dict[str, Any] = {"mode": "key", "key": key}

        if full_context:
            full_text = self._transformers_build_prompt_fragment(
                prompt=str(prompt or ""),
                messages=messages,
                system_prompt=system_prompt,
                tools=tools if isinstance(tools, list) else None,
                add_generation_prompt=True,
                prefilled_modules=prefilled_modules,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
            new_ids = self._transformers_tokenize_fragment(full_text, add_bos_if_empty=True)
            if not new_ids:
                raise RuntimeError("Transformers cached generation could not tokenize the prompt")

            def _lcp(a, b) -> int:
                n = min(len(a), len(b))
                i = 0
                while i < n and a[i] == b[i]:
                    i += 1
                return i

            if self._transformers_snapshot_lane_required(state):
                # UNTRIMMABLE ARCHITECTURE (linear-attention/Gated-DeltaNet
                # hybrids): crop is refused by construction, so route on the
                # ARCHITECTURE — from turn 1, so the first turn already
                # leaves a reusable boundary snapshot (the MLX lane's
                # lesson: entering only on trim refusal costs the loop one
                # extra full prefill on turn 2).
                delta_ids = self._transformers_snapshot_feed(
                    key, state, full_text, new_ids, cache_telemetry
                )
            else:
                # CROP LANE TELEMETRY (parity, 2026-08-07). This lane is the one
                # every pure-attention model takes — i.e. most models — and until
                # now it was the ONLY local cache lane that reported nothing:
                # `cache_telemetry` stayed empty, so `metadata["prompt_cache"]`
                # was omitted entirely and a user could not tell a warm call from
                # a cold one. Measured before the fix on Qwen3-4B-Instruct-2507
                # bf16/MPS: warm calls ran 1.2s against 3.8s uncached (the cache
                # was working perfectly) with zero evidence of it on the wire.
                # Vocabulary is MLX's, exactly: cold / hit_full / hit_extend /
                # rebuilt, with `cached_tokens` + `fed_tokens` measured, and
                # `degraded_reason` carrying the `#FALLBACK` label on rebuild.
                cache_len_before = len(state.prompt_tokens)
                prefix = _lcp(state.prompt_tokens, new_ids)
                identical = prefix >= len(new_ids)
                if identical:
                    # Identical resend: keep one token to step generation.
                    prefix = len(new_ids) - 1
                crop_refused = False
                if prefix < len(state.prompt_tokens):
                    # Divergence or stale generated tokens: crop back to the LCP;
                    # non-croppable caches rebuild fresh (one cold
                    # prefill — never a stale-context answer, never a double one).
                    if prefix > 0 and self._transformers_crop_cache(state, prefix):
                        state.prompt_tokens = tuple(state.prompt_tokens[:prefix])
                    else:
                        crop_refused = True
                        # Crop refused (sliding-window past fill, no-op-crop
                        # hybrids caught by the verify, or prefix 0): one honest
                        # cold prefill — loudly, once per key (labeled-degradation
                        # policy; the MLX lane's analogous branches already warn).
                        warned = getattr(self, "_transformers_rebuild_warned_keys", None)
                        if warned is None:
                            warned = set()
                            self._transformers_rebuild_warned_keys = warned
                        if key not in warned:
                            warned.add(key)
                            self.logger.warning(
                                f"#FALLBACK transformers prompt cache '{key}': cache cannot be cropped "
                                f"for this architecture (sliding-window past fill, or non-croppable "
                                f"layers); rebuilding fresh per warm call — no prefill savings."
                            )
                        # Rebuild = RELEASE point for the old full cache (~1 GB at
                        # 30k). Drop it BEFORE building fresh so its buffers are
                        # dead at empty_cache time; otherwise every warm rep parks
                        # another full KV in the MPS pool (2026-08-03 leak audit).
                        # Cost: this branch already pays a full cold prefill —
                        # empty_cache here is noise, not hot-path overhead.
                        state.cache = None
                        self._transformers_release_device_pool()
                        state.cache = self._transformers_empty_native_cache()
                        state.prompt_tokens = ()
                        prefix = 0
                delta_ids = list(new_ids[prefix:])

                cache_telemetry.setdefault("lane", "crop")
                if crop_refused:
                    cache_telemetry.update({
                        "outcome": "rebuilt",
                        "cached_tokens": 0,
                        "fed_tokens": len(delta_ids),
                        "degraded_reason": (
                            "#FALLBACK cache cannot be cropped for this architecture "
                            "(sliding-window past fill, or non-croppable layers); "
                            "rebuilt fresh — no prefill savings this call"
                        ),
                    })
                elif cache_len_before <= 0:
                    cache_telemetry.update({
                        "outcome": "cold",
                        "cached_tokens": 0,
                        "fed_tokens": len(delta_ids),
                    })
                else:
                    cache_telemetry.update({
                        "outcome": "hit_full" if identical else "hit_extend",
                        "cached_tokens": int(prefix),
                        "fed_tokens": len(delta_ids),
                    })

            # Keep the update-lane bookkeeping coherent for mixed callers.
            state.system_prompt_parts = [str(system_prompt)] if isinstance(system_prompt, str) and system_prompt.strip() else []
            state.messages = [copy.deepcopy(m) for m in messages if isinstance(m, dict)]
            if isinstance(prompt, str) and prompt:
                state.messages.append({"role": "user", "content": prompt})
            state.add_generation_prompt = True
        else:
            # Best-effort first-call prefill when callers pass system/tools alongside the key.
            if not state.prompt_tokens and (system_prompt is not None or tools):
                tools_for_cache = None
                if isinstance(tools, list) and tools and all(isinstance(t, dict) for t in tools):
                    tools_for_cache = tools  # type: ignore[assignment]
                self.prompt_cache_update(
                    key,
                    system_prompt=system_prompt,
                    tools=tools_for_cache,  # type: ignore[arg-type]
                    messages=None,
                    add_generation_prompt=False,
                )
                cache_value = self._prompt_cache_store.get(key)
                state = self._transformers_prompt_cache_state(cache_value) or state

            # Delta-only fragment: user message + assistant generation prefix.
            delta_text = self._transformers_build_prompt_fragment(
                prompt=str(prompt or ""),
                messages=None,
                system_prompt=None,
                tools=None,
                add_generation_prompt=True,
                prefilled_modules=prefilled_modules,
                enable_thinking=enable_thinking,
            )
            delta_ids = self._transformers_tokenize_fragment(delta_text, add_bos_if_empty=not bool(state.prompt_tokens))

            # APPEND LANE (prompt-only caller: CachedSession KV mode). The cache
            # IS the context by contract, so there is nothing to reconcile — but
            # it still has to report, or KV-mode sessions stay unobservable.
            # MLX names this outcome `append`; so do we.
            cache_telemetry.setdefault("lane", "append")
            cache_telemetry.update({
                "outcome": "append" if state.prompt_tokens else "cold",
                "cached_tokens": len(state.prompt_tokens),
                "fed_tokens": len(delta_ids),
            })

        if not delta_ids:
            # Nothing left to feed. This returns EMPTY content, so it must say so
            # rather than look like a normal empty reply (ADR 0001: no silent
            # no-ops) — the telemetry names the outcome and the reason.
            cache_telemetry.update({
                "outcome": "noop",
                "cached_tokens": len(state.prompt_tokens),
                "fed_tokens": 0,
                "degraded_reason": (
                    "#FALLBACK prompt produced no tokens to feed beyond the cached "
                    "prefix; returned empty content without generating"
                ),
            })
            return GenerateResponse(
                content="",
                model=self.model,
                finish_reason="stop",
                usage={
                    "input_tokens": len(state.prompt_tokens),
                    "output_tokens": 0,
                    "total_tokens": len(state.prompt_tokens),
                    "prompt_tokens": len(state.prompt_tokens),
                    "completion_tokens": 0,
                },
                gen_time=round((time.time() - start_time) * 1000, 1),
                metadata={"prompt_cache": dict(cache_telemetry)},
            )

        # LONG-DELTA GUARD: `generate()` forwards its whole input in ONE pass,
        # which at 30k asserts inside Metal (see _transformers_prefill_step).
        # Feed all but the last token through the chunked prefill and let
        # `generate()` start from a single-token seed — same KV, same next
        # token, bounded transients. Small deltas keep the one-pass path
        # byte-identically.
        chunk_step = self._transformers_prefill_step()
        if chunk_step > 0 and len(delta_ids) > chunk_step:
            if not self._transformers_prefill_cache(state, list(delta_ids[:-1])):
                raise RuntimeError(
                    "Transformers cached generation failed: chunked prefill of the "
                    f"{len(delta_ids) - 1}-token delta failed"
                )
            delta_ids = [delta_ids[-1]]

        device = self._transformers_cache_device() or torch.device("cpu")
        past_len = len(state.prompt_tokens)
        input_ids = torch.tensor([delta_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, past_len + len(delta_ids)), dtype=torch.long, device=device)

        do_sample = True
        temp_val: float = float(temperature) if temperature is not None else float(getattr(self, "temperature", 0.7) or 0.7)
        try:
            if temp_val <= 0:
                do_sample = False
                temp_val = 0.0
        except Exception:
            do_sample = True

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        try:
            pad_i = int(pad_token_id) if pad_token_id is not None else None
        except Exception:
            pad_i = None
        try:
            eos_i = int(eos_token_id) if eos_token_id is not None else None
        except Exception:
            eos_i = None
        if pad_i is None and eos_i is not None:
            pad_i = eos_i

        generate_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "use_cache": True,
            "return_dict_in_generate": True,
            "pad_token_id": pad_i,
        }
        if do_sample:
            generate_kwargs["temperature"] = temp_val
            generate_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                generate_kwargs["top_k"] = int(top_k)
        if eos_i is not None:
            generate_kwargs["eos_token_id"] = eos_i

        if state.cache is None:
            raise RuntimeError("prompt cache key has no concrete transformers cache state")
        # Prefer updating the existing provider-native cache object in-place for speed.
        generate_kwargs["past_key_values"] = state.cache

        output = None
        try:
            # The pool ratchet is per DECODE STEP, so the release policy has to
            # be able to run inside this call, not only after it returns.
            with torch.inference_mode(), self._transformers_decode_pool_guard():
                use_mps_lock = str(device).startswith("mps") or str(getattr(self, "device", "") or "").strip().lower() == "mps"
                if use_mps_lock:
                    with _MPS_GENERATION_LOCK:
                        output = self.model_instance.generate(**generate_kwargs)
                else:
                    output = self.model_instance.generate(**generate_kwargs)
        except Exception as e:
            # generate() can RAISE AFTER mutating the cache in place (MPS OOM
            # mid-decode): `state.prompt_tokens` then no longer describes the
            # physical KV, and every later call would silently misattend over
            # phantom tokens — and the snapshot lane would store a poisoned
            # boundary (ADVERSARY FINDING 1, adversary_snapshot_poison_test).
            # Reset the key's live state before re-raising; the key's next
            # call rebuilds — or restores from the still-clean pre-decode
            # snapshot, which is exactly what the snapshot is for.
            state.cache = None
            state.prompt_tokens = ()
            self._transformers_release_device_pool()
            raise RuntimeError(f"Transformers cached generation failed: {e}") from e
        finally:
            # Threshold-guarded (no-op below the pooled-bytes bound): caps the
            # MPS pool ratchet from decode cat-churn without touching healthy
            # pool reuse. See _transformers_maybe_release_device_pool.
            self._transformers_maybe_release_device_pool()

        sequences = getattr(output, "sequences", None)
        if sequences is None:
            # Some generate paths return a raw tensor; treat it like sequences.
            sequences = output

        seq0 = sequences[0].tolist() if hasattr(sequences, "__getitem__") else []
        gen_ids = [int(tok) for tok in seq0[len(delta_ids):]] if seq0 else []

        decoded = ""
        try:
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids else ""
        except Exception:
            decoded = ""

        # Update KV cache + token tracker.
        new_cache = getattr(output, "past_key_values", None)
        if new_cache is not None:
            state.cache = new_cache
        state.prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens + tuple(delta_ids) + tuple(gen_ids)))
        state.add_generation_prompt = False

        # Full-context callers already recorded the prompt into state.messages
        # in their branch above — appending again here duplicated the user turn
        # in the rebuild-lane bookkeeping (adversarial find 2026-07-13: a later
        # prompt_cache_update re-rendered the cache with the question twice).
        if prompt and not full_context:
            try:
                state.messages.append({"role": "user", "content": str(prompt)})
            except Exception:
                pass
        if decoded:
            try:
                state.messages.append({"role": "assistant", "content": decoded})
            except Exception:
                pass

        try:
            meta = self._prompt_cache_store.meta(key) or {}
            meta = dict(meta)
            meta["token_count"] = len(state.prompt_tokens)
            self._prompt_cache_store.set(key, state, meta=meta)
        except Exception:
            pass

        gen_time = round((time.time() - start_time) * 1000, 1)
        usage = {
            "prompt_tokens": past_len + len(delta_ids),
            "completion_tokens": len(gen_ids),
            "total_tokens": past_len + len(delta_ids) + len(gen_ids),
            "input_tokens": past_len + len(delta_ids),
            "output_tokens": len(gen_ids),
        }

        return GenerateResponse(
            content=decoded,
            model=self.model,
            finish_reason="stop",
            usage=usage,
            gen_time=gen_time,
            metadata={"prompt_cache": dict(cache_telemetry)} if cache_telemetry else None,
        )

    def _transformers_generate_uncached_chunked(
        self,
        input_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int] = None,
    ) -> Optional[GenerateResponse]:
        """Uncached generation for prompts too long to one-shot prefill.

        The pipeline path forwards the whole prompt in ONE pass; on MPS the
        SDPA math fallback then materializes an [heads, L, L] float32 score
        transient — at 30k on Qwen3-4B that is a single 107.15 GiB MTLBuffer
        and Metal aborts the PROCESS (measured twice; see
        `_transformers_prefill_step`). This path prefills all but the last
        token in chunks over a throwaway cache, generates from the one-token
        seed, and discards the cache: same tokens, bounded transients.

        Returns None when not applicable (short prompt, chunking disabled,
        or setup failure) — the caller then uses the pipeline exactly as
        before. Once the prompt is known to be one-shot-hostile, failures
        RAISE instead of falling back: the fallback would abort the process,
        not just the call."""
        step = self._transformers_prefill_step()
        if step <= 0:
            return None
        if getattr(self, "model_instance", None) is None or getattr(self, "tokenizer", None) is None:
            return None
        try:
            import torch  # type: ignore
        except Exception:
            return None
        try:
            prompt_ids = self.tokenizer(str(input_text or ""))["input_ids"]
        except Exception:
            return None
        if not isinstance(prompt_ids, list) or len(prompt_ids) <= step:
            return None

        start_time = time.time()
        output = None
        pool_guard_stats: Dict[str, Any] = {}
        state = _TransformersPromptCacheValue(cache=self._transformers_empty_native_cache())
        try:
            if not self._transformers_prefill_cache(state, [int(t) for t in prompt_ids[:-1]]):
                raise RuntimeError(
                    f"chunked prefill of the {len(prompt_ids) - 1}-token prompt failed"
                )

            device = self._transformers_cache_device() or torch.device("cpu")
            seed_ids = [int(prompt_ids[-1])]
            input_ids = torch.tensor([seed_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device)

            do_sample = True
            try:
                if float(temperature) <= 0:
                    do_sample = False
            except Exception:
                pass
            eos_i = getattr(self.tokenizer, "eos_token_id", None)
            generate_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "use_cache": True,
                "return_dict_in_generate": True,
                "pad_token_id": eos_i,
            }
            if do_sample:
                generate_kwargs["temperature"] = float(temperature)
                generate_kwargs["top_p"] = float(top_p)
                if top_k is not None:
                    generate_kwargs["top_k"] = int(top_k)
            if eos_i is not None:
                generate_kwargs["eos_token_id"] = eos_i
            if state.cache is not None:
                generate_kwargs["past_key_values"] = state.cache

            use_mps_lock = str(device).startswith("mps") or str(getattr(self, "device", "") or "").strip().lower() == "mps"
            # This path bounds its PREFILL by chunking and then decodes with
            # whatever budget the caller resolved — up to the model's registry
            # `max_output_tokens`. Without a step-scoped release that asymmetry
            # is what consumed the host: 12.7k prompt + 4096 decode = 113 GiB
            # driver / 104.8 GiB of it pooled and dead. See
            # `_transformers_decode_pool_guard`.
            with torch.inference_mode(), self._transformers_decode_pool_guard() as _pool_stats:
                if use_mps_lock:
                    with _MPS_GENERATION_LOCK:
                        output = self.model_instance.generate(**generate_kwargs)
                else:
                    output = self.model_instance.generate(**generate_kwargs)
            pool_guard_stats = dict(_pool_stats)

            sequences = getattr(output, "sequences", None)
            if sequences is None:
                sequences = output
            seq0 = sequences[0].tolist() if hasattr(sequences, "__getitem__") else []
            gen_ids = [int(tok) for tok in seq0[len(seed_ids):]] if seq0 else []
            try:
                response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids else ""
            except Exception:
                response_text = ""
        finally:
            # The throwaway cache is a full-prompt KV (~1 GB at 30k):
            # drop the reference, then let the threshold-guarded release
            # decide whether the pool needs trimming (never unconditional
            # on a hot path).
            #
            # EVERY reference to this call's KV must die BEFORE the release
            # measures the pool, not after (2026-08-07). The release compares
            # `driver - current`, so anything still holding the decoded cache is
            # charged to `current` and UNDER-REPORTS the pool by exactly that
            # much. Three separate references held it: `output` (which carries
            # `past_key_values` because `return_dict_in_generate=True`),
            # `sequences`, and `generate_kwargs["past_key_values"]` — clearing
            # `state.cache` alone left the other three alive.
            #
            # This is not a rounding error, it is the difference between firing
            # and not firing. Measured across 10 consecutive uncached calls with
            # 1024-token budgets at 10-13.6k (`oom/results/consecutive_BEFORE_12k.json`):
            # resting pool slack climbed 2.426 -> 4.427 GiB and STOPPED there,
            # parked just under the 4 GiB bound once the ~0.46 GiB of live KV was
            # subtracted, so the release never ran and the driver ratcheted
            # 10.26 -> 12.26 GiB with no recovery.
            state.cache = None
            output = None
            sequences = None
            try:
                generate_kwargs.clear()
            except Exception:
                pass
            self._transformers_maybe_release_device_pool()

        return GenerateResponse(
            content=response_text,
            model=self.model,
            finish_reason="stop",
            usage=self._calculate_usage(input_text, response_text),
            gen_time=round((time.time() - start_time) * 1000, 1),
            metadata={"mps_pool_guard": pool_guard_stats} if pool_guard_stats.get("enabled") else None,
        )

    def _single_generate_transformers(self, input_text: str, max_new_tokens: int,
                                     temperature: float, top_p: float, top_k: Optional[int] = None,
                                     seed: Optional[int] = None) -> GenerateResponse:
        """Generate single response using transformers (original implementation)"""
        try:
            # Set seed for deterministic generation if provided
            if seed is not None:
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except ImportError:
                    pass  # Skip seeding if torch not available

            # Track generation time
            start_time = time.time()

            do_sample = True
            try:
                temperature_value = float(temperature)
                if temperature_value <= 0:
                    do_sample = False
            except Exception:
                temperature_value = temperature

            pipeline_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "do_sample": bool(do_sample),
                "truncation": True,
                "return_full_text": False,
            }
            if do_sample:
                pipeline_kwargs["temperature"] = temperature_value
                pipeline_kwargs["top_p"] = top_p
                if top_k is not None:
                    pipeline_kwargs["top_k"] = int(top_k)

            # Prompts too long to one-shot prefill take the chunked manual
            # path (None = not applicable → pipeline below, exactly as
            # before). On MPS a one-shot 30k prefill aborts the PROCESS
            # (Metal assert), so this must run before the pipeline.
            chunked_resp = self._transformers_generate_uncached_chunked(
                input_text, max_new_tokens, temperature, top_p, top_k
            )
            if chunked_resp is not None:
                return chunked_resp

            try:
                # Same step-scoped guard as the chunked path. A short prompt
                # still decodes with the caller's full budget, so this lane can
                # reach a long context too — just from the other end.
                with self._transformers_decode_pool_guard():
                    outputs = self.pipeline(input_text, **pipeline_kwargs)
            finally:
                # The UNCACHED text lane is where the 164.5 GB incident
                # actually peaked (arms floor/A/B; C/D never ran). Threshold-
                # guarded: no-op unless the MPS pool retains > bound of freed
                # memory. See _transformers_maybe_release_device_pool.
                self._transformers_maybe_release_device_pool()

            gen_time = round((time.time() - start_time) * 1000, 1)

            if outputs and len(outputs) > 0:
                response_text = outputs[0]['generated_text'].strip()

                # Calculate token usage using centralized utilities
                usage = self._calculate_usage(input_text, response_text)

                return GenerateResponse(
                    content=response_text,
                    model=self.model,
                    finish_reason="stop",
                    usage=usage,
                    gen_time=gen_time
                )
            else:
                return GenerateResponse(
                    content="",
                    model=self.model,
                    finish_reason="stop",
                    gen_time=gen_time
                )

        except Exception as e:
            gen_time = round((time.time() - start_time) * 1000, 1) if 'start_time' in locals() else 0.0
            return GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error",
                gen_time=gen_time
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

    def _stream_generate_transformers(self, input_text: str, max_new_tokens: int,
                                     temperature: float, top_p: float, top_k: Optional[int] = None,
                                     tool_call_tags: Optional[str] = None, seed: Optional[int] = None) -> Iterator[GenerateResponse]:
        """Stream response using transformers (simulated, original implementation) with tool tag rewriting support"""
        try:
            # HuggingFace doesn't have native streaming, so we simulate it
            full_response = self._single_generate_transformers(input_text, max_new_tokens, temperature, top_p, top_k, seed)

            if full_response.content:
                # Apply tool tag rewriting if enabled
                content = full_response.content
                if tool_call_tags:
                    try:
                        from ..tools.tag_rewriter import create_tag_rewriter
                        rewriter = create_tag_rewriter(tool_call_tags)
                        content = rewriter.rewrite_text(content)
                    except ImportError:
                        pass

                words = content.split()
                for i, word in enumerate(words):
                    chunk_content = word + (" " if i < len(words) - 1 else "")
                    yield GenerateResponse(
                        content=chunk_content,
                        model=self.model,
                        finish_reason="stop" if i == len(words) - 1 else None
                    )
            else:
                yield GenerateResponse(
                    content="",
                    model=self.model,
                    finish_reason="stop"
                )

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _build_input_text_transformers(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Build input text for transformers model with tool support"""

        # Add tools to system prompt if provided (one shared placement policy).
        final_system_prompt = merge_tools_into_system(self.tool_handler, system_prompt, tools)

        # Check if model has chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use chat template if available
            chat_messages = []

            if final_system_prompt:
                chat_messages.append({"role": "system", "content": final_system_prompt})

            if messages:
                chat_messages.extend(messages)

            chat_messages.append({"role": "user", "content": prompt})

            try:
                template_kwargs: Dict[str, Any] = {}
                if isinstance(enable_thinking, bool):
                    template_kwargs["enable_thinking"] = bool(enable_thinking)
                if isinstance(reasoning_effort, str) and reasoning_effort:
                    # The model's own template consumes this (declared in assets as
                    # thinking_control.effort_template_kwarg); transformers forwards
                    # extra apply_chat_template kwargs into template.render.
                    surfaces = self._thinking_control_surfaces()
                    if surfaces.effort_template_kwarg:
                        template_kwargs[surfaces.effort_template_kwarg] = reasoning_effort
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            except Exception as e:
                # Fallback if chat template fails. Never silent when a thinking
                # control was riding on the render: the plain-format fallback
                # carries no effort artifact and only a crude disable prefill.
                if template_kwargs:
                    warnings.warn(
                        f"chat template render failed ({e}); falling back to plain format — "
                        f"thinking controls {sorted(template_kwargs)} were NOT applied.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # Build simple conversational format. Use `final_system_prompt` (with
        # the tool block merged in) — the prior fallback used the raw
        # `system_prompt` and SILENTLY DROPPED the tool declaration on any
        # template-less model (S4 find, 2026-07-15).
        text_parts = []

        if final_system_prompt:
            text_parts.append(f"System: {final_system_prompt}\n")

        if messages:
            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                text_parts.append(f"{role}: {content}\n")

        text_parts.append(f"User: {prompt}\n")
        text_parts.append("Assistant:")
        text_parts.append(self._thinking_disable_prefill(enable_thinking))

        return "".join(text_parts)

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        capabilities = ["chat", "streaming"]

        if self.model_type == "gguf":
            capabilities.append("gguf")
            if self.llm and self.llm.chat_format:
                capabilities.append("tools")
        else:
            # Check for specific model capabilities
            model_lower = self.model.lower()

            if "gpt2" in model_lower or "dialogpt" in model_lower:
                capabilities.append("dialogue")

            if "codegen" in model_lower or "starcoder" in model_lower or "coder" in model_lower:
                capabilities.append("code")

        return capabilities

    def get_model_residency(self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Return Core-owned in-process residency truth for the loaded HuggingFace provider."""
        _ = kwargs
        task_s = str(task or "text_generation").strip() or "text_generation"
        model_s = str(model or self.model or "").strip()
        loaded = any(
            value is not None
            for value in (
                getattr(self, "llm", None),
                getattr(self, "model_instance", None),
                getattr(self, "pipeline", None),
            )
        )
        return {
            "task": task_s,
            "provider": "huggingface",
            "model": model_s,
            "provider_residency_verified": True,
            "provider_resident": loaded,
            "loaded": loaded,
            "state": "loaded" if loaded else "not_loaded",
            "source": "abstractcore.provider.huggingface",
        }

    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if self.model_type == "gguf":
            return self.llm is not None
        else:
            return self.pipeline is not None


    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter appropriate for the model type"""
        max_output_tokens = kwargs.get("max_output_tokens", self.max_output_tokens)

        if self.model_type == "gguf":
            # For GGUF models, this is the generation limit
            return max_output_tokens
        else:
            # For transformers, this is max_new_tokens
            return max_output_tokens


    def _stream_generate_transformers_with_tools(self, input_text: str, max_new_tokens: int,
                                               temperature: float, top_p: float, top_k: Optional[int] = None,
                                               tools: Optional[List[Dict[str, Any]]] = None,
                                               tool_call_tags: Optional[str] = None, seed: Optional[int] = None) -> Iterator[GenerateResponse]:
        """Stream generate with tool execution at the end"""
        collected_content = ""

        # Stream the response content
        for chunk in self._stream_generate_transformers(input_text, max_new_tokens, temperature, top_p, top_k, tool_call_tags, seed):
            collected_content += chunk.content
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

    def _handle_tool_execution_gguf(self, response: GenerateResponse, tools: List[Dict[str, Any]], has_native_tools: bool) -> GenerateResponse:
        """Handle tool execution for GGUF responses - both native and prompted"""
        if has_native_tools and response.has_tool_calls():
            # Handle native tool calls using base method
            tool_calls = self._convert_native_tool_calls_to_standard(response.tool_calls)
            return self._execute_tools_with_events(response, tool_calls)
        elif self.tool_handler.supports_prompted and response.content:
            # Handle prompted tool calls using base method
            return self._handle_prompted_tool_execution(response, tools)

        return response

    def _stream_generate_gguf_with_tools(self, generation_kwargs: Dict[str, Any],
                                       tools: Optional[List[Dict[str, Any]]] = None,
                                       has_native_tools: bool = False,
                                       tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Stream generate GGUF with tool execution at the end"""
        collected_content = ""
        collected_tool_calls = []

        # Stream the response content
        for chunk in self._stream_generate_gguf(generation_kwargs, tool_call_tags):
            collected_content += chunk.content
            if chunk.tool_calls:
                collected_tool_calls.extend(chunk.tool_calls)
            yield chunk

        # Handle tool execution if we have tools and content/calls
        if tools and (collected_tool_calls or
                     (self.tool_handler.supports_prompted and collected_content)):
            # Create complete response for tool processing
            complete_response = GenerateResponse(
                content=collected_content,
                model=self.model,
                finish_reason="stop",
                tool_calls=collected_tool_calls
            )

            # Handle tool execution using simplified method
            final_response = self._handle_tool_execution_gguf(complete_response, tools, has_native_tools)

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
        List available HuggingFace models from local cache (excluding MLX models).

        Args:
            **kwargs: Optional parameters including:
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            from .model_capabilities import filter_models_by_capabilities

            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if not hf_cache.exists():
                return []

            models = []
            for item in hf_cache.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Convert models--microsoft--DialoGPT-medium to microsoft/DialoGPT-medium
                    model_name = item.name.replace("models--", "").replace("--", "/")

                    # CRITICAL: Exclude MLX models from HuggingFace list
                    # Any model with "mlx" in the name should be classified as MLX, not HuggingFace
                    if "mlx" not in model_name.lower():
                        models.append(model_name)

            models = sorted(models)

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
