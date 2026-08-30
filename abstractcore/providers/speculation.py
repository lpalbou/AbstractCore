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
        try:
            n = int(n)
        except (TypeError, ValueError):
            raise ValueError(f"num_draft_tokens must be an int, got {n!r}")
        if n < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {n}")

    drafter = value.get("drafter")
    if drafter is not None and not str(drafter).strip():
        raise ValueError("speculation.drafter must be a non-empty string when given")

    return SpeculationRequest(
        mode=mode,
        drafter=str(drafter).strip() if drafter else None,
        num_draft_tokens=n,
        require_acceleration=bool(value.get("require_acceleration", False)),
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
    )
