"""Provider-agnostic speculative-decoding (MTP) request contract.

Speculative decoding only ever changes HOW fast tokens arrive, never WHAT the
caller asked for, so the whole contract here is about honesty rather than
behaviour: a request either runs on a lane that can accelerate it, or the caller
is told -- by name -- why it did not. The failure this module exists to prevent
is a `speculation={...}` that quietly evaporates into ordinary autoregressive
generation while the caller believes it bought a speedup.

Two facts shape the design, both measured on real artifacts rather than inferred
from model names (see the `speculation` blocks in `model_capabilities.json`):

- The MTP head is a property of the ARTIFACT, not the model family. The same
  Qwen3.8-27B ships its head embedded in the GGUF but split into a separate
  256 MB drafter repo for MLX, and `mtp_num_hidden_layers` appears in MLX repo
  configs whose weights contain no MTP tensors at all. So capability lookup is
  keyed on (model, runtime), and a config key alone is never evidence.
- Whether a runtime can EXECUTE the head is independent of whether the weights
  are present. llama.cpp implements the graph and llama-cpp-python cannot reach
  its driver, so the identical GGUF accelerates under `llama-server` and does
  nothing in-process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

# v1 deliberately ships two modes. `draft_model` (a separate small model drafting
# for a big one) is NOT here: for the Qwen3.5/3.8 hybrids it is not merely
# unimplemented but impossible -- mlx-lm raises "Speculative decoding requires a
# trimmable prompt cache (got {'ArraysCache'})" because the linear-attention
# layers keep a cache that cannot be rewound. Adding the mode would mean adding a
# mode that always fails on the very models MTP is for.
SPECULATION_MODES = ("native_mtp", "off")

_REQUEST_KEYS = frozenset(
    {"mode", "drafter", "num_draft_tokens", "require_acceleration"}
)


class SpeculationUnavailableError(RuntimeError):
    """Raised when `require_acceleration=True` and the lane cannot honor it.

    Carries the same machine-readable `reason` slug that a non-strict request
    would have been warned with, so callers can branch on the cause instead of
    parsing prose.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class SpeculationRequest:
    """A normalized `speculation=` request."""

    mode: str = "off"
    drafter: Optional[str] = None
    num_draft_tokens: Optional[int] = None
    require_acceleration: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class SpeculationOutcome:
    """What actually happened, for response metadata.

    `used` is the load-bearing field and it means exactly one thing: a drafter
    ran during this call. It must never be set from a request, a capability
    entry, or an import check -- only from the lane that did the work.
    """

    requested: bool = False
    mode: str = "off"
    used: bool = False
    reason: Optional[str] = None
    drafter: Optional[str] = None
    num_draft_tokens: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        # `details` is provider-supplied free-form telemetry, so it goes in
        # FIRST and the contract fields overwrite it. The other order let a
        # provider hand back details={"used": True} and have the serialized
        # metadata claim acceleration that never ran -- the single lie this
        # whole module exists to make impossible.
        out: Dict[str, Any] = dict(self.details) if self.details else {}
        out.update(
            {
                "requested": self.requested,
                "mode": self.mode,
                "used": self.used,
            }
        )
        # Optional fields are still authoritative over `details`: absent means
        # absent, so a stale detail key must not resurrect them either.
        out.pop("reason", None)
        out.pop("drafter", None)
        out.pop("num_draft_tokens", None)
        if self.reason:
            out["reason"] = self.reason
        if self.drafter:
            out["drafter"] = self.drafter
        if self.num_draft_tokens is not None:
            out["num_draft_tokens"] = self.num_draft_tokens
        return out


def normalize_speculation_request(value: Any) -> Optional[SpeculationRequest]:
    """Coerce a caller's `speculation=` into a `SpeculationRequest`.

    Returns None when nothing was asked for, so callers can distinguish "no
    opinion" from an explicit `speculation=False` (which pins the lane off and
    should suppress any capability-driven default).

    Raises ValueError on anything malformed. A typo'd key here would otherwise
    become a silent no-op -- the precise failure the caller is trying to avoid.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return SpeculationRequest(mode="native_mtp" if value else "off")
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in ("", "none", "off", "false"):
            return SpeculationRequest(mode="off")
        if mode in ("mtp", "native_mtp"):
            return SpeculationRequest(mode="native_mtp")
        raise ValueError(
            f"Unknown speculation mode {value!r}. Supported: {list(SPECULATION_MODES)}"
        )
    if not isinstance(value, Mapping):
        raise ValueError(
            "speculation must be a dict, bool, or str; got " f"{type(value).__name__}"
        )

    unknown = set(value) - _REQUEST_KEYS
    if unknown:
        raise ValueError(
            f"Unknown speculation keys: {sorted(unknown)}. "
            f"Supported: {sorted(_REQUEST_KEYS)}"
        )

    raw_mode = value.get("mode", "native_mtp")
    mode = str(raw_mode).strip().lower()
    if mode in ("mtp",):
        mode = "native_mtp"
    if mode not in SPECULATION_MODES:
        raise ValueError(
            f"Unknown speculation mode {raw_mode!r}. Supported: {list(SPECULATION_MODES)}"
        )

    n = value.get("num_draft_tokens")
    if n is not None:
        if isinstance(n, bool) or not (
            isinstance(n, int) or (isinstance(n, str) and n.strip().isdigit())
        ):
            raise ValueError(f"num_draft_tokens must be an int, got {n!r}")
        n = int(n)
        if n < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {n}")

    drafter = value.get("drafter")
    if drafter is not None and (not isinstance(drafter, str) or not drafter.strip()):
        raise ValueError("speculation.drafter must be a non-empty string when given")

    required = value.get("require_acceleration", False)
    if not isinstance(required, bool):
        raise ValueError("speculation.require_acceleration must be a bool")

    return SpeculationRequest(
        mode=mode,
        drafter=str(drafter).strip() if drafter else None,
        num_draft_tokens=n,
        require_acceleration=required,
    )


def normalize_speculation_value(value: Any) -> Any:
    """Validate a JSON control without filling omitted, inheritable fields.

    Shared by configuration, Runtime and HTTP adapters. None means inherit;
    False means off. Never use truthiness when transporting this value.
    """
    request = normalize_speculation_request(value)
    if request is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return False if not request.enabled else {"mode": request.mode}
    out = dict(value)
    if "mode" in out:
        out["mode"] = request.mode
    if "num_draft_tokens" in out:
        out["num_draft_tokens"] = request.num_draft_tokens
    if "drafter" in out:
        out["drafter"] = request.drafter
    return out


def resolve_speculation_request(
    default: Optional[SpeculationRequest], override: Any
) -> Optional[SpeculationRequest]:
    """Resolve per-call controls without mutating provider-lifetime defaults.

    Omitted fields inherit the loaded session's defaults. Explicit off always
    wins. This is backend-neutral: runtimes translate the resulting draft-token
    count to their own verifier width (which may include a seed/bonus token).
    """
    request = normalize_speculation_request(override)
    if request is None:
        return default
    if not request.enabled or default is None:
        return request
    explicit_required = isinstance(override, Mapping) and "require_acceleration" in override
    return SpeculationRequest(
        mode=request.mode,
        drafter=request.drafter or default.drafter,
        num_draft_tokens=request.num_draft_tokens or default.num_draft_tokens,
        require_acceleration=(request.require_acceleration if explicit_required else default.require_acceleration),
    )


def capability_speculation(
    model_capabilities: Any, runtime: str
) -> Optional[Dict[str, Any]]:
    """Return the `speculation.runtimes[<runtime>]` block for a model, or None.

    `native_mtp` must be truthy for the block to count: an entry can describe a
    runtime while recording that the head is absent, and that is a "no".
    """
    caps = model_capabilities if isinstance(model_capabilities, Mapping) else {}
    spec = caps.get("speculation")
    if not isinstance(spec, Mapping) or not spec.get("native_mtp"):
        return None
    runtimes = spec.get("runtimes")
    if not isinstance(runtimes, Mapping):
        return None
    block = runtimes.get(runtime)
    if not isinstance(block, Mapping):
        return None
    out = dict(block)
    out.setdefault("mtp_layers", spec.get("mtp_layers"))
    return out


def configured_speculation_default(*, config_file: Any = None, capability_defaults: Any = None) -> Any:
    """Read the current Core-owned policy, without creating config or loading weights.

    Key the tiny read cache by the file stamp: a console change takes effect on
    the next request, without replacing a shared provider or holding GPU locks.
    """
    from ..config.manager import resolve_config_file
    if capability_defaults is not None:
        from ..config.capability_defaults import capability_default_speculation
        return capability_default_speculation(capability_defaults)
    path = resolve_config_file(config_file=config_file)
    try:
        stamp = path.stat()
    except FileNotFoundError:
        from ..config.capability_defaults import RECOMMENDED_CAPABILITY_DEFAULT_ROUTES
        return normalize_speculation_value(RECOMMENDED_CAPABILITY_DEFAULT_ROUTES["input.text"].options.get("speculation"))
    return normalize_speculation_value(_read_configured_speculation(str(path), stamp.st_mtime_ns, stamp.st_ctime_ns, stamp.st_size))


@lru_cache(maxsize=16)
def _read_configured_speculation(path: str, mtime: int, ctime: int, size: int) -> Any:
    import json
    from pathlib import Path
    from ..config.capability_defaults import capability_default_speculation
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return capability_default_speculation(document.get("capability_defaults", {}).get("routes", {}))


def _local_model_directory(model: str):
    from pathlib import Path
    from ..utils.model_cache import resolve_hf_snapshot_dir, resolve_lmstudio_model_dir
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    return resolve_hf_snapshot_dir(model) or resolve_lmstudio_model_dir(model)


def mlx_speculation_artifact(model: str) -> Optional[Dict[str, Any]]:
    """Registry evidence using an exact HF cache repo identity, never a midfix guess."""
    from pathlib import Path
    from ..architectures.detection import get_model_capabilities
    path = Path(model).expanduser()
    identity = model
    if path.is_dir() and path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        parts = path.parent.parent.name.removeprefix("models--").split("--")
        if len(parts) == 2 and all(parts):
            identity = "/".join(parts)
    return capability_speculation(get_model_capabilities(identity), "mlx")


def describe_speculation_capabilities(model: str, provider: str, instance: Any = None) -> Dict[str, Any]:
    """Describe artifact + adapter + instance facts; never download or load a model.

    Unknown readiness (None) is different from an unsupported adapter (False).
    Static registry evidence is not an assertion that a loaded head exists.
    """
    import importlib.util
    from ..architectures.detection import get_model_capabilities
    caps = getattr(instance, "model_capabilities", None) if instance is not None else None
    caps = caps if isinstance(caps, Mapping) else get_model_capabilities(model)
    block = (capability_speculation(caps, "mlx") or mlx_speculation_artifact(model)) if provider == "mlx" else None
    active = instance is not None and getattr(instance, "_mtp_active", False) is True
    supported = provider == "mlx" and bool(block or active)
    reason = None if supported else "native_mtp_backend_unavailable" if provider != "mlx" else "mtp_artifact_unverified"
    head_present = None
    local = _local_model_directory(model) if provider == "mlx" else None
    if local is not None:
        from .mlx_qwen4 import is_qwen4_checkpoint, embedded_mtp_keys
        if is_qwen4_checkpoint(str(local)):
            head_present = bool(embedded_mtp_keys(str(local)))
            supported = head_present
            reason = None if supported else "embedded_mtp_weights_missing"
        elif block and block.get("drafter"):
            head = _local_model_directory(str(block["drafter"]))
            head_present = bool(head and any(head.glob("*.safetensors")))
    if supported and importlib.util.find_spec("mlx_vlm") is None:
        supported, reason = False, "mlx_vlm_missing"
    ready = bool(active) if instance is not None else None
    if not supported:
        ready = False
    elif not active:
        reason = "mtp_head_not_cached" if head_present is False else "model_not_loaded" if instance is None else "speculation_is_load_time"
    default = configured_speculation_default(
        config_file=getattr(instance, "_abstractcore_config_file", None),
        capability_defaults=getattr(instance, "_abstractcore_capability_defaults", None),
    )
    effective_default = normalize_speculation_value(default) if supported else False
    if instance is not None and getattr(instance, "_speculation_inherits_config", True) is False:
        request = getattr(instance, "_speculation_request", None)
        if request is not None:
            effective_default = False if not request.enabled else {
                "mode": request.mode,
                "num_draft_tokens": request.num_draft_tokens or getattr(instance, "_mtp_block_size", None),
                "require_acceleration": request.require_acceleration,
            }
    return {
        "supported": supported,
        "ready": ready,
        "reason": reason,
        # Choices implemented by the native MLX verifier, not a universal model
        # vocabulary. Other adapters must declare their own choices when added.
        "supported_depths": [2, 3, 4, 5] if supported else [],
        "default": normalize_speculation_value(default),
        "effective_default": effective_default,
        "requires_reload": (not active) if supported and instance is not None else None,
        "head_present": True if active else head_present,
    }


def get_execution_capabilities(model_name: str, *, provider: str, instance: Any = None) -> Dict[str, Any]:
    """Public library/HTTP discovery contract for one selected execution route."""
    concurrent = None
    concurrency_reason = None
    if instance is not None:
        try:
            probe = getattr(instance, "supports_concurrent_generation", None)
            answer = probe() if callable(probe) else False
            concurrent = answer if isinstance(answer, bool) else None
            if concurrent is None:
                concurrency_reason = "invalid_capability_probe"
        except Exception:
            concurrency_reason = "capability_probe_failed"
    return {
        "version": 1,
        "provider": provider,
        "model": model_name,
        "speculation": describe_speculation_capabilities(model_name, provider, instance),
        "concurrency": {"supported": concurrent, "source": "loaded_instance" if instance is not None else "unknown", "reason": concurrency_reason},
    }


def prepare_provider_speculation(instance: Any, kwargs: Dict[str, Any]) -> Optional[SpeculationOutcome]:
    """Validate every provider's explicit request; unsupported must not evaporate.

    MLX negotiates its head and per-call restrictions in its adapter. Other
    providers currently have no native-MTP execution adapter; off is always
    safe, while a best-effort refusal is attached to the individual response.
    """
    value = kwargs.get("speculation")
    if value is None:
        value = getattr(instance, "_speculation_constructor_default", None)
    request = normalize_speculation_request(value)
    if request is None:
        return None
    if getattr(instance, "provider", None) == "mlx":
        return None
    kwargs.pop("speculation", None)
    if not request.enabled:
        return SpeculationOutcome(reason="disabled_for_this_call")
    return unavailable(
        request, "native_mtp_backend_unavailable",
        f"The {getattr(instance, 'provider', None) or type(instance).__name__} provider has no native MTP adapter",
        logger=getattr(instance, "logger", None),
    )


def attach_speculation_outcome(response: Any, outcome: Optional[SpeculationOutcome]) -> Any:
    if outcome is not None and response is not None and hasattr(response, "metadata"):
        response.metadata = {**(response.metadata or {}), "speculation": outcome.to_metadata()}
    return response


def speculation_outcome_stream(source: Any, outcome: Optional[SpeculationOutcome]):
    try:
        for chunk in source:
            yield attach_speculation_outcome(chunk, outcome)
    finally:
        if hasattr(source, "close"):
            source.close()


async def speculation_outcome_async_stream(source: Any, outcome: Optional[SpeculationOutcome]):
    try:
        async for chunk in source:
            yield attach_speculation_outcome(chunk, outcome)
    finally:
        if hasattr(source, "aclose"):
            await source.aclose()


def unavailable(
    request: Optional[SpeculationRequest],
    reason: str,
    message: str,
    *,
    logger: Any = None,
) -> SpeculationOutcome:
    """Record that a requested speculation could not run.

    Warns (never debug: the caller asked for something and did not get it), and
    raises when the caller said `require_acceleration=True`. Returns an outcome
    with `used=False` so the response metadata carries the reason even in the
    non-strict case.
    """
    if request is None or not request.enabled:
        return SpeculationOutcome()

    if request.require_acceleration:
        raise SpeculationUnavailableError(
            f"{message} (speculation.require_acceleration=True)", reason=reason
        )
    if logger is not None:
        logger.warning(f"speculation: {message} [{reason}]")
    return SpeculationOutcome(
        requested=True,
        mode=request.mode,
        used=False,
        reason=reason,
        drafter=request.drafter,
        num_draft_tokens=request.num_draft_tokens,
        # Carry the prose, not just the slug. The full diagnosis previously
        # existed only as a log line, so any surface that showed the outcome
        # (the CLI's /speculation view) could report "mlx_vlm_missing" and
        # nothing about WHICH interpreter was missing it.
        details={"message": message},
    )
