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


def resident_provider(provider: Optional[str], model: Optional[str], *, task: Optional[str] = None) -> Optional[Any]:
    """An alive provider INSTANCE of THIS process already holding `model`, to reuse instead of building
    another full copy, or None.

    Only the `huggingface` backend lends (GGUF on llama.cpp, transformers): its instances do not share
    weights, so a second instance of a resident model is a second full copy (llama.cpp: another context
    and compute buffers; on a 16 GB GPU the second one fails to allocate), and one instance is safe to
    share across threads (see below). MLX instances already share their weights (`mlx_residency`), and embeddings are not
    chat providers: None for those.

    Sharing is safe because a HuggingFace instance runs every generation on its own worker thread,
    one at a time (`HuggingFaceProvider._generation_worker`; a stream is produced there into a queue).
    An instance the server built for ONE request (`_abstractcore_request_scoped`: it may be unloaded
    when that request ends, `unload_after`) is never lent.

    A holder is the first alive registered instance whose model name or path matches (`model_matches`,
    the eject's matcher). The caller must not unload a lent instance: it belongs to its holder (a
    runtime pool, a server registry)."""
    if backend_for(provider, task) != "huggingface" or not model or not _backend_loaded("huggingface"):
        return None
    from .hf_residency import _holder_alive, registered_providers

    for holder in registered_providers():
        try:
            if not _holder_alive(holder) or getattr(holder, "_abstractcore_request_scoped", False):
                continue
            path = getattr(getattr(holder, "llm", None), "model_path", None)
            if model_matches(getattr(holder, "model", None), str(path) if path else None, model):
                return holder
        except Exception:  # noqa: BLE001 - an unreadable holder is skipped, never lent
            continue
    return None


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
            # Holders in words (ADR 0043): a claimant's `owner_label` ("the
            # runtime of account default/bob"), never its id.
            owners = sorted({f"{c.get('owner_label') or 'another owner in this process'}"
                             f"{' (locked)' if c.get('locked') else ''}" for c in claims})
            return {"ok": True, "skipped": True, "backend": backend, "model": model, "claims": claims,
                    "locked": any(bool(c.get("locked")) for c in claims),
                    "reason": f"{model} is still in use by {', '.join(owners)}; not ejected"}
        return eject(backend, model, reason=reason)


# ---------------------------------------------------------------------------
# The residency failsafe (core ADR 0010): a load checks weights + KV against
# the free memory, evicts resident models that are not locked (least recently
# used first, one at a time), and otherwise refuses with a typed error.
# ---------------------------------------------------------------------------
_GB = 1000 ** 3
_LOAD_LOCK = _threading.Lock()


def load_lock() -> "_threading.Lock":
    """One in-process model load at a time (ADR 0010): fit check + allocation."""
    return _LOAD_LOCK


def _release_torch_device_cache() -> None:
    """Freed torch tensors stay in its CUDA cache, which the free-memory probe
    counts as used: return them before measuring."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _fmt_gb(n: Optional[int]) -> str:
    return "unknown" if n is None else f"{n / _GB:.1f} GB"


def _hf_holder_state(holder: Any) -> Dict[str, Any]:
    from .hf_residency import _holder_row

    row = _holder_row(holder)
    inflight = 0
    last_used = getattr(holder, "_residency_registered_at", None)
    try:
        registry = holder._inflight_generations()
        inflight = int(registry.active())
        if getattr(registry, "last_used_at", None):
            last_used = max(float(last_used or 0.0), float(registry.last_used_at))
    except Exception:  # noqa: BLE001 - a holder without the registry is never "in use" by it
        pass
    # A slot pool with a generation running or a request waiting is in use (never evicted).
    inflight += int(row.get("active") or 0) + int(row.get("queued") or 0)
    return {"held_bytes": int(row.get("held_bytes") or 0), "inflight": inflight, "last_used_at": last_used,
            "weights_bytes": int(row.get("weights_bytes") or 0), "weights_id": row.get("weights_id"),
            "scheduler_id": row.get("scheduler_id")}


def resident_model_states() -> List[Dict[str, Any]]:
    """Every in-process HuggingFace model resident now, with what the failsafe
    decides on: `held_bytes`, `last_used_at` (newest over its instances),
    `in_use` (a generation running on any instance), `locked_by` (the readable
    owners whose claim locks it). MLX and embeddings are not evicted by the
    failsafe yet (ADR 0010: adopted lane by lane)."""
    if not _backend_loaded("huggingface"):
        return []
    from .hf_residency import _holder_alive, registered_providers

    by_model: Dict[str, Dict[str, Any]] = {}
    counted_weights: set = set()
    counted_pools: set = set()
    for holder in registered_providers():
        try:
            if not _holder_alive(holder):
                continue
            name = str(getattr(holder, "model", "") or "").strip()
            if not name:
                continue
            st = _hf_holder_state(holder)
        except Exception:  # noqa: BLE001
            continue
        cur = by_model.setdefault(name, {"backend": "huggingface", "provider": "huggingface", "model": name,
                                         "held_bytes": 0, "last_used_at": None, "in_use": False, "holders": 0})
        cur["holders"] += 1
        # Weights shared by several holders (gguf_shared) are held ONCE: what evicting frees.
        wid = st.get("weights_id")
        pid = st.get("scheduler_id")
        if pid is not None and pid in counted_pools:
            cur["in_use"] = cur["in_use"] or st["inflight"] > 0
            if st["last_used_at"] is not None:
                cur["last_used_at"] = max(float(cur["last_used_at"] or 0.0), float(st["last_used_at"]))
            continue  # the pool's contexts and weights are counted with its first holder
        if pid is not None:
            counted_pools.add(pid)
        if wid is not None and wid in counted_weights:
            cur["held_bytes"] += max(0, int(st["held_bytes"]) - int(st.get("weights_bytes") or 0))
        else:
            cur["held_bytes"] += st["held_bytes"]
            if wid is not None:
                counted_weights.add(wid)
        cur["in_use"] = cur["in_use"] or st["inflight"] > 0
        if st["last_used_at"] is not None:
            cur["last_used_at"] = max(float(cur["last_used_at"] or 0.0), float(st["last_used_at"]))
    for state in by_model.values():
        claims = claims_for("huggingface", state["model"])
        state["locked_by"] = sorted({str(c.get("owner_label") or "another owner in this process")
                                     for c in claims if c.get("locked")})
    return list(by_model.values())


def make_room(needed_bytes: int, free_bytes: Any, *, provider: Optional[str] = None, model: Optional[str] = None,
              reason: str = "make_room") -> Dict[str, Any]:
    """Evict resident models that are neither locked nor in use, least recently
    used first, one at a time, until `free_bytes()` reaches `needed_bytes` or
    none is evictable. Never evicts `model` itself. Returns `{"fits", "free_bytes",
    "evicted": [{model, freed_bytes}], "kept": [readable reasons]}`."""
    evicted: List[Dict[str, Any]] = []
    with _RESIDENCY_LOCK:
        free = int(free_bytes())
        if free < int(needed_bytes):
            # PLAN FIRST (OVH 8080, 2026-10-29): evicting the unlocked model and then
            # refusing anyway left the operator with neither model. Evict only when the
            # evictable models can cover the gap; otherwise refuse with everything kept.
            evictable = [s for s in resident_model_states()
                         if not (model and model_matches(s["model"], None, model))
                         and not s["locked_by"] and not s["in_use"]]
            if free + sum(int(s["held_bytes"] or 0) for s in evictable) < int(needed_bytes):
                kept = [f"{s['model']} (locked by {', '.join(s['locked_by'])})" if s["locked_by"]
                        else (f"{s['model']} (in use)" if s["in_use"] else f"{s['model']} (not enough to free)")
                        for s in resident_model_states() if not (model and model_matches(s["model"], None, model))]
                return {"fits": False, "free_bytes": free, "evicted": [], "kept": kept,
                        "unaccounted": True}
        while free < int(needed_bytes):
            states = [s for s in resident_model_states()
                      if not (model and model_matches(s["model"], None, model))]
            candidates = [s for s in states if not s["locked_by"] and not s["in_use"]]
            if not candidates:
                break
            victim = min(candidates, key=lambda s: float(s["last_used_at"] or 0.0))
            # Re-checked right before the eject: a generation that started since is
            # never cancelled for room (the next candidate is tried instead).
            again = next((s for s in resident_model_states() if s["model"] == victim["model"]), None)
            if again is None or again["in_use"] or again["locked_by"]:
                if again is None:
                    continue
                states_skip = victim["model"]
                candidates = [c for c in candidates if c["model"] != states_skip]
                if not candidates:
                    break
                continue
            report = eject(victim["backend"], victim["model"], reason=reason)
            _release_torch_device_cache()
            after = int(free_bytes())
            evicted.append({"model": victim["model"], "freed_bytes": max(0, after - free),
                            "ok": bool(report.get("ok", True))})
            if after <= free and not report.get("ok", True):
                break  # it did not free anything: stop rather than loop
            free = after
        kept = []
        for s in resident_model_states():
            if model and model_matches(s["model"], None, model):
                continue
            if s["locked_by"]:
                kept.append(f"{s['model']} (locked by {', '.join(s['locked_by'])})")
            elif s["in_use"]:
                kept.append(f"{s['model']} (in use)")
    return {"fits": free >= int(needed_bytes), "free_bytes": free, "evicted": evicted, "kept": kept}


def smaller_suggestions(model: str, budget_bytes: int, *, provider: str = "huggingface",
                        limit: int = 3, reserve_bytes: int = 1 << 30) -> List[Dict[str, Any]]:
    """Catalog artifacts of the same model (or its family) on `provider` whose
    download fits `budget_bytes` with room for a KV cache, largest first."""
    try:
        from ..config.model_catalog import load_seed

        rows = list(load_seed().get("rows") or [])
    except Exception:  # noqa: BLE001
        return []
    name = str(model or "").lower()
    same = [r for r in rows if any(str(a.get("artifact") or "").lower() == name for a in r.get("artifacts") or [])]
    if not same:
        same = [r for r in rows if str(r.get("id") or "").lower() and str(r.get("id")).lower() in name.replace("_", "-")]
    families = {str(r.get("family") or "") for r in same if r.get("family")}
    if not families:
        # A fine-tune or re-quant the catalog does not list ("mradermacher/Gemma-4-14B-
        # A4B-Heretic-GGUF"): its family by name ("gemma4" in "gemma-4-14b-...").
        import re as _re

        flat = _re.sub(r"[^a-z0-9]", "", name)
        families = {str(r.get("family")) for r in rows
                    if r.get("family") and len(_re.sub(r"[^a-z0-9]", "", str(r.get("family")))) >= 5
                    and _re.sub(r"[^a-z0-9]", "", str(r.get("family")).lower()) in flat}
    pool = same + [r for r in rows if r not in same and str(r.get("family") or "") in families]
    out: List[Dict[str, Any]] = []
    for r in pool:
        for a in r.get("artifacts") or []:
            size = a.get("download_bytes")
            if str(a.get("provider") or "") != provider or not isinstance(size, int):
                continue
            if str(a.get("artifact") or "").lower() == name:
                continue
            # The fit check's own arithmetic: weights + a small-context KV (~15%) + the reserve.
            if size * 1.15 + int(reserve_bytes) > budget_bytes:
                continue
            out.append({"model": r.get("display_name") or r.get("id"), "artifact": a.get("artifact"),
                        "quant": a.get("quant"), "download_bytes": size})
    out.sort(key=lambda x: -int(x["download_bytes"]))
    seen, uniq = set(), []
    for item in out:
        if item["artifact"] in seen:
            continue
        seen.add(item["artifact"])
        uniq.append(item)
    return uniq[:limit]


def ensure_fit(model: str, needed_bytes: int, free_bytes: Any, *, provider: str = "huggingface",
               device: str = "GPU", reserve_bytes: int = 1 << 30) -> Dict[str, Any]:
    """THE fit check (ADR 0010). Evicts unlocked LRU models when needed; raises
    `ModelDoesNotFitError` with the sentence, numbers and smaller suggestions when
    the load still cannot fit. Returns the make-room report when it fits."""
    report = make_room(needed_bytes, free_bytes, provider=provider, model=model, reason=f"room for {model}")
    for ev in report["evicted"]:
        import logging

        logging.getLogger("abstractcore.residency").warning(
            f"evicted {ev['model']} (least recently used, not locked) to make room for {model}")
    if report["fits"]:
        return report
    refuse_fit(model, needed_bytes, int(report["free_bytes"]), device=device, kept=report["kept"],
               evicted=report["evicted"], provider=provider, reserve_bytes=reserve_bytes)
    return report  # pragma: no cover - refuse_fit raises


def refuse_fit(model: str, needed_bytes: int, free: int, *, device: str, kept: List[str],
               evicted: Optional[List[Dict[str, Any]]] = None, provider: str = "huggingface",
               reserve_bytes: int = 1 << 30) -> None:
    """Raise the typed refusal (ADR 0010) with its sentence and suggestions."""
    from ..exceptions import ModelDoesNotFitError

    evicted = list(evicted or [])
    suggestions = smaller_suggestions(model, free, provider=provider, reserve_bytes=reserve_bytes)
    sentence = f"{model} needs {_fmt_gb(needed_bytes)} but only {_fmt_gb(free)} is free on the {device}"
    if kept:
        sentence += f" (kept: {'; '.join(kept)})"
    sentence += ". Choose a smaller model"
    if suggestions:
        sentence += ", for example " + ", ".join(
            f"{s['artifact']} ({_fmt_gb(s['download_bytes'])})" for s in suggestions)
    if any("locked" in k for k in kept):
        sentence += ", or unlock a kept model on the Models page"
    sentence += "."
    raise ModelDoesNotFitError(sentence, model=model, needed_bytes=int(needed_bytes), free_bytes=free,
                               device=device, kept=kept, evicted=evicted, suggestions=suggestions)
