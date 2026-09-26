"""One door to every in-process model residency backend: MLX
(`mlx_residency`), the HuggingFace provider (`hf_residency`: transformers on
torch + GGUF on llama.cpp) and in-process embeddings
(`abstractcore.embeddings.manager`).

Hosts (the AbstractCore server, the runtime facade, the gateway) use it to

- list every model whose weights are alive in THIS process, whoever holds
  them (`resident_rows`), in the shared row shape (`backend`, `lane`,
  `models`, `holders`, `weights_bytes`, `cache_bytes`, `held_bytes`,
  `weights_alive`, `shared_weights`);
- eject a model from every holder in the process, then return the backend's
  allocator cache to the OS (`eject`), with a report that never pretends.

A backend whose module was never imported holds nothing, so it is skipped
without importing it: a listing never pulls MLX, torch or
sentence-transformers into a process that did not load them.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

# Backend names as they appear in rows (`row["backend"]`).
BACKENDS = ("mlx", "huggingface", "embeddings")

_EMBEDDINGS_MODULE = "abstractcore.embeddings.manager"


def backend_for(provider: Optional[str], task: Optional[str] = None) -> Optional[str]:
    """The in-process backend serving `provider` for `task`, or None when the
    provider's weights live in another process (Ollama, LM Studio, a cloud API).
    Embeddings are one backend whatever their local provider."""
    task_s = str(task or "").strip().lower()
    if task_s in ("embedding", "embeddings"):
        return "embeddings"
    provider_s = str(provider or "").strip().lower()
    if provider_s in ("mlx", "huggingface"):
        return provider_s
    return None


def _backend_loaded(backend: str) -> bool:
    if backend == "mlx":
        return "mlx.core" in sys.modules
    if backend == "huggingface":
        return "abstractcore.providers.huggingface_provider" in sys.modules
    if backend == "embeddings":
        return _EMBEDDINGS_MODULE in sys.modules
    raise ValueError(f"unknown in-process residency backend {backend!r}; expected one of {BACKENDS}")


def _rows_for(backend: str) -> List[Dict[str, Any]]:
    if backend == "mlx":
        from .mlx_residency import resident_models

        rows = resident_models()
    elif backend == "huggingface":
        from .hf_residency import resident_models

        rows = resident_models()
    else:
        rows = sys.modules[_EMBEDDINGS_MODULE].resident_embedding_models()
    out = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("weights_alive", True):
            continue
        item = {k: v for k, v in row.items() if k != "holder_rows"}
        item.setdefault("backend", backend)
        out.append(item)
    return out


def resident_rows(backend: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every model whose weights are alive in this process (optionally for one
    backend). A backend that fails to report raises: a listing must not show
    "nothing resident" because a probe broke."""
    backends = [backend] if backend else list(BACKENDS)
    rows: List[Dict[str, Any]] = []
    for name in backends:
        if not _backend_loaded(name):
            continue
        rows.extend(_rows_for(name))
    return rows


def row_names(row: Dict[str, Any]) -> List[str]:
    return [str(n) for n in (row.get("models") or []) if str(n).strip()]


def eject(backend: str, model: Optional[str], *, reason: str = "eject") -> Dict[str, Any]:
    """Unload EVERY holder of `model` (None: every model of that backend) in
    this process, collect, and return the backend's allocator cache. The
    report carries `ok` (True only when nothing of the model remains),
    `holders_found`, `holders_unloaded`, `holders_refused`, `residual`.
    A backend that was never imported holds nothing: `ok` with 0 holders."""
    if not _backend_loaded(backend):
        return {"ok": True, "backend": backend, "model": model, "reason": reason, "holders_found": 0,
                "holders_unloaded": [], "holders_refused": [], "residual": None,
                "note": f"{backend} is not loaded in this process; nothing to eject"}
    if backend == "mlx":
        from .mlx_residency import eject_model

        report = eject_model(model, reason=reason)
    elif backend == "huggingface":
        from .hf_residency import eject_model

        report = eject_model(model, reason=reason)
    else:
        report = sys.modules[_EMBEDDINGS_MODULE].eject_embedding_models(model, reason=reason)
    report = dict(report)
    report.setdefault("backend", backend)
    return report


# ---------------------------------------------------------------------------
# Who still wants a model.
#
# An eject is process-wide: it unloads EVERY holder. Deciding "nobody uses it
# any more" from one client's pool is therefore wrong in a process that hosts
# several clients (one per gateway user, per-entity runtimes, the AbstractCore
# server's managed runtimes). Every such owner registers as a CLAIMANT and
# answers `residency_claims()`: the (provider, model) pairs it has pooled,
# locked, or is building right now. `eject_unclaimed` checks every claimant
# with the SAME matcher the eject uses (case-insensitive, path-aware) and
# ejects under one process lock, so no claim can appear between the check and
# the eject. Owners take `residency_lock()` when they register a new claim.
# ---------------------------------------------------------------------------
import threading as _threading
import weakref as _weakref

_RESIDENCY_LOCK = _threading.RLock()
_CLAIMANTS: "_weakref.WeakSet[Any]" = _weakref.WeakSet()


def residency_lock() -> "_threading.RLock":
    """The process lock under which claims are checked and ejects run. Hold it
    while registering a claim (e.g. inserting into a pool / marking a build)."""
    return _RESIDENCY_LOCK


def register_claimant(owner: Any) -> None:
    """Register an object whose `residency_claims()` yields dicts
    `{"provider", "model", "locked": bool, "kind": str, "path"?: str, "owner"?: str}`.
    Held weakly: a collected owner claims nothing."""
    if not callable(getattr(owner, "residency_claims", None)):
        raise TypeError(f"{type(owner).__name__} has no residency_claims(); it cannot claim models")
    _CLAIMANTS.add(owner)


def unregister_claimant(owner: Any) -> None:
    _CLAIMANTS.discard(owner)


def model_matches(claim_model: Optional[str], claim_path: Optional[str], model: Optional[str],
                  path: Optional[str] = None) -> bool:
    """The eject's own matcher (`mlx_residency._matches`): case-insensitive
    names, resolved paths, and HF hub-cache directories -- symmetric, so a
    claim spelled differently from the request still matches."""
    from .mlx_residency import _matches

    if not (model or path):
        return False
    if _matches({"models": [claim_model] if claim_model else [], "model_path": claim_path or ""}, model, path):
        return True
    # symmetric: the request as the row, the claim as the query (a request
    # given as a local / hub-cache path is that row's path)
    req_path = path or (model if model and str(model).startswith(("/", "~", ".")) else "")
    return bool(claim_model) and _matches({"models": [model] if model else [], "model_path": req_path or ""},
                                          claim_model, claim_path)


def claims_for(provider: Optional[str], model: Optional[str], *, path: Optional[str] = None,
               task: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every registered claim on (provider, model), across all claimants. A
    claimant whose `residency_claims()` raises is itself reported as a claim
    (`kind: "claimant_error"`): an unreadable owner never reads as "unused"."""
    backend = backend_for(provider, task)
    out: List[Dict[str, Any]] = []
    with _RESIDENCY_LOCK:
        for owner in list(_CLAIMANTS):
            try:
                claims = list(owner.residency_claims())
            except Exception as exc:  # noqa: BLE001
                out.append({"provider": provider, "model": model, "locked": True, "kind": "claimant_error",
                            "owner": type(owner).__name__, "error": f"{type(exc).__name__}: {exc}"})
                continue
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                if backend_for(claim.get("provider"), claim.get("task")) != backend:
                    continue
                if model_matches(claim.get("model"), claim.get("path"), model, path):
                    out.append(dict(claim))
    return out


def eject_unclaimed(provider: str, model: str, *, task: Optional[str] = None, path: Optional[str] = None,
                    reason: str = "eject") -> Optional[Dict[str, Any]]:
    """Eject `model` process-wide unless a registered owner still pools, locks
    or is building it. Check and eject run under one process lock. None when
    the provider's weights live in another process. A skipped eject returns
    `{"ok": True, "skipped": True, "claims": [...], "reason": ...}`."""
    backend = backend_for(provider, task)
    if backend is None:
        return None
    with _RESIDENCY_LOCK:
        claims = claims_for(provider, model, path=path, task=task)
        if claims:
            owners = sorted({f"{c.get('owner') or c.get('kind')}{' (locked)' if c.get('locked') else ''}" for c in claims})
            return {"ok": True, "skipped": True, "backend": backend, "model": model, "claims": claims,
                    "locked": any(bool(c.get("locked")) for c in claims),
                    "reason": f"{model} is still in use by {', '.join(owners)}; not ejected"}
        return eject(backend, model, reason=reason)
