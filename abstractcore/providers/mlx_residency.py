"""Process-level MLX residency: what THIS process holds in Metal memory, and who.

Why this exists (live incident, 2026-09-25): a gateway showed "No models loaded"
while `mx.get_active_memory()` was 92 GB. Every MLX provider INSTANCE answered
truthfully for itself (`self.llm is None` after its own `unload_model`), but the
weights are SHARED between instances (`_SharedMLXModel`, `NativeSession`) and
survive as long as ANY instance holds them -- a chat summarizer built at boot, a
client created for a per-request override, an old runtime after a bundle
reload, another principal's service. None of those is reachable from the eject
route, so the eject freed nothing and the report said nothing was loaded.

This module is the process-wide truth every consumer can ask:

- `resident_models()`  -- every MLX model whose weights are alive in this
  process, with its holders and the bytes it pins (weights, drafter, prefix
  caches, prompt-cache stores, hybrid snapshots).
- `process_residency_for(model, path)` -- the row for one model (or None).
- `eject_model(model)` -- unload EVERY holder of that model, then
  `gc.collect()` + `mx.clear_cache()`, and report what is still held.
- `mlx_memory_report()` -- MLX active / cache / peak plus the rows above.

Policy (ADR-0026: no silent caps, no silent leaks): an eject drops the model's
prefix caches (APC) and prompt-cache stores together with the weights -- a KV
cache is only meaningful with the weights it was computed on -- and the eject
result REPORTS the bytes freed and any residual. Nothing here evicts on its own.
"""
from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("abstractcore.providers.mlx")

_EJECT_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# MLX memory counters (None when MLX is not importable / not Metal)
# ---------------------------------------------------------------------------
def mx_memory_stats() -> Dict[str, Optional[int]]:
    """`active` (live buffers), `cache` (freed buffers MLX keeps), `peak`."""
    out: Dict[str, Optional[int]] = {"active_bytes": None, "cache_bytes": None, "peak_bytes": None}
    try:
        import mlx.core as mx  # type: ignore

        metal = getattr(mx, "metal", None)
        for name, key in (("get_active_memory", "active_bytes"), ("get_cache_memory", "cache_bytes"),
                          ("get_peak_memory", "peak_bytes")):
            fn = getattr(mx, name, None) or getattr(metal, name, None)
            if callable(fn):
                try:
                    out[key] = int(fn())
                except Exception:
                    pass
    except Exception:
        pass
    return out


def clear_mx_cache() -> bool:
    """Release MLX's cache of freed buffers back to the system. True when it ran."""
    try:
        import mlx.core as mx  # type: ignore

        fn = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if callable(fn):
            fn()
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Byte accounting helpers
# ---------------------------------------------------------------------------
def _is_array_like(obj: Any) -> bool:
    try:
        import mlx.core as mx  # type: ignore

        if isinstance(obj, mx.array):
            return True
    except Exception:
        pass
    nbytes = getattr(obj, "nbytes", None)
    return isinstance(nbytes, int) and not isinstance(nbytes, bool) and not isinstance(obj, (dict, list, tuple, set))


def params_bytes(module: Any) -> Optional[int]:
    """Bytes of a module's parameters (unique arrays), None when unknowable."""
    if module is None:
        return None
    try:
        from mlx.utils import tree_flatten  # type: ignore

        arrays = [v for _, v in tree_flatten(module.parameters())]
        return int(sum(int(v.nbytes) for v in {id(a): a for a in arrays}.values()))
    except Exception:
        pass
    # Fakes/tests: a bare object with nbytes counts as one array.
    nbytes = getattr(module, "nbytes", None)
    if isinstance(nbytes, int) and not isinstance(nbytes, bool):
        return int(nbytes)
    return None


def reachable_array_bytes(root: Any, *, max_depth: int = 6, max_nodes: int = 100_000) -> int:
    """Sum of unique array bytes reachable from `root` through python containers,
    attribute dicts and `nn.Module.parameters()`. Bounded; never raises."""
    if root is None:
        return 0
    seen: set = set()
    arrays: Dict[int, int] = {}
    frontier: List[Tuple[Any, int]] = [(root, 0)]
    n = 0
    while frontier and n < max_nodes:
        obj, depth = frontier.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        n += 1
        try:
            if _is_array_like(obj):
                arrays[id(obj)] = int(getattr(obj, "nbytes", 0) or 0)
                continue
            if depth >= max_depth or isinstance(obj, (str, bytes, int, float, bool, type(None))):
                continue
            children: List[Any] = []
            if hasattr(obj, "parameters") and callable(getattr(obj, "parameters")) and hasattr(obj, "children"):
                try:
                    from mlx.utils import tree_flatten  # type: ignore

                    children.extend(v for _, v in tree_flatten(obj.parameters()))
                except Exception:
                    pass
            if isinstance(obj, dict):
                children.extend(obj.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                children.extend(obj)
            elif hasattr(obj, "__dict__"):
                children.extend(vars(obj).values())
            for slot in getattr(type(obj), "__slots__", ()) or ():
                try:
                    children.append(getattr(obj, slot))
                except Exception:
                    pass
            for child in children:
                if not isinstance(child, (str, bytes, int, float, bool, type(None))):
                    frontier.append((child, depth + 1))
        except Exception:
            continue
    return int(sum(arrays.values()))


def apc_bytes(apc: Any) -> int:
    """Bytes a mlx-vlm APCManager pins: its own accounting when it has one
    (`stats_snapshot()['resident_bytes'] + ['exact_resident_bytes']`), else a
    bounded reachability walk."""
    if apc is None:
        return 0
    try:
        snap = apc.stats_snapshot()
        if isinstance(snap, dict):
            total = 0
            known = False
            for key in ("resident_bytes", "exact_resident_bytes"):
                value = snap.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    total += value
                    known = True
            if known:
                return int(total)
    except Exception:
        pass
    return reachable_array_bytes(apc)


def prompt_cache_store_bytes(store: Any) -> int:
    if store is None:
        return 0
    return reachable_array_bytes(store)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
def _native_session_entries() -> Iterable[Tuple[str, str, Any]]:
    """(lane, model_path, session) for every live native session."""
    # Registries live in modules; a module that was never imported holds
    # nothing, and importing it here would pull mlx-vlm into a process that
    # merely asked for a memory report (MEM2 discipline: reports never import
    # a backend).
    try:
        ns = sys.modules.get("abstractcore.providers.mlx_native_session")
        for key, session in (list(ns._SESSIONS.items()) if ns is not None else []):
            yield "mlx_vlm", str(key[0]) if isinstance(key, tuple) and key else str(key), session
    except Exception:
        pass
    try:
        q4 = sys.modules.get("abstractcore.providers.mlx_qwen4")
        for key, session in (list(q4._SESSIONS.items()) if q4 is not None else []):
            yield "mlx_vlm_qwen4", str(key[0]) if isinstance(key, tuple) and key else str(key), session
    except Exception:
        pass


def _shared_model_entries() -> Iterable[Tuple[str, str, Any]]:
    try:
        mp = sys.modules.get("abstractcore.providers.mlx_provider")
        for key, shared in (list(mp._SHARED_MLX_MODELS.items()) if mp is not None else []):
            yield "mlx_lm", str(key), shared
    except Exception:
        pass


def _holder_row(holder: Any) -> Dict[str, Any]:
    return {
        "id": id(holder),
        "model": getattr(holder, "model", None),
        "instance_loaded": getattr(holder, "llm", None) is not None,
        "type": type(holder).__name__,
    }


def resident_models() -> List[Dict[str, Any]]:
    """Every MLX model whose weights are alive in this process, with holders and bytes.

    A session/shared entry whose holders are all gone but which is still alive
    (a finalizer that has not run yet, a cycle awaiting gc) is reported too, with
    `holders: 0` -- it is still memory.
    """
    rows: List[Dict[str, Any]] = []
    for lane, model_path, session in _native_session_entries():
        holders = [h for h in list(getattr(session, "holders", ()) or ())]
        model = getattr(session, "model", None)
        drafter = getattr(session, "drafter", None)
        weights = params_bytes(model)
        drafter_b = params_bytes(drafter)
        apc_b = apc_bytes(getattr(session, "apc", None))
        cache_store = getattr(session, "cache_store", None)
        store_b = reachable_array_bytes(cache_store) if cache_store is not None else 0
        row = {
            "lane": lane,
            "model_path": model_path,
            "models": sorted({str(getattr(h, "model", "") or "") for h in holders if getattr(h, "model", None)}),
            "holders": len(holders),
            "holder_rows": [_holder_row(h) for h in holders],
            "weights_bytes": weights,
            "drafter_bytes": drafter_b,
            "apc_bytes": apc_b,
            "prompt_cache_bytes": store_b,
            "hybrid_snapshot_bytes": 0,
            "runtime_present": getattr(session, "runtime", None) is not None,
            "weights_alive": model is not None,
        }
        row["cache_bytes"] = int(apc_b + store_b)
        row["held_bytes"] = int((weights or 0) + (drafter_b or 0) + row["cache_bytes"])
        rows.append(row)
    for lane, model_path, shared in _shared_model_entries():
        holders = [h for h in list(getattr(shared, "holders", ()) or ())]
        weights = params_bytes(getattr(shared, "llm", None))
        store_b = prompt_cache_store_bytes(getattr(shared, "prompt_cache_store", None))
        snaps = getattr(shared, "hybrid_snapshots", None)
        snaps_b = reachable_array_bytes(snaps) if snaps else 0
        row = {
            "lane": lane,
            "model_path": model_path,
            "models": sorted({str(getattr(h, "model", "") or "") for h in holders if getattr(h, "model", None)}),
            "holders": len(holders),
            "holder_rows": [_holder_row(h) for h in holders],
            "weights_bytes": weights,
            "drafter_bytes": None,
            "apc_bytes": 0,
            "prompt_cache_bytes": store_b,
            "hybrid_snapshot_bytes": snaps_b,
            "runtime_present": False,
            "weights_alive": getattr(shared, "llm", None) is not None,
        }
        row["cache_bytes"] = int(store_b + snaps_b)
        row["held_bytes"] = int((weights or 0) + row["cache_bytes"])
        rows.append(row)
    return rows


def _matches(row: Dict[str, Any], model: Optional[str], path: Optional[str]) -> bool:
    if model is None and path is None:
        return True
    names = {str(m).strip().lower() for m in row.get("models") or []}
    model_s = str(model or "").strip().lower()
    path_s = str(path or "").strip()
    if model_s and model_s in names:
        return True
    row_path = str(row.get("model_path") or "")
    if path_s:
        try:
            if Path(path_s).resolve() == Path(row_path).resolve():
                return True
        except Exception:
            if path_s == row_path:
                return True
    if model_s:
        try:
            if Path(model_s).resolve() == Path(row_path).resolve():
                return True
        except Exception:
            pass
        # `Jundot/Qwen3.8-27B` vs `.../models--Jundot--Qwen3.8-27B/snapshots/<sha>`
        hub_dir = "models--" + model_s.replace("/", "--")
        if hub_dir in row_path.lower():
            return True
    return False


def process_residency_for(model: Optional[str], path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The residency row for `model` (name or path) when this process holds its
    weights, else None. Several lanes of one model (mlx-lm AND mlx-vlm) are
    folded into one row with summed bytes and holders."""
    matched = [row for row in resident_models() if _matches(row, model, path) and row.get("weights_alive")]
    if not matched:
        return None
    if len(matched) == 1:
        return dict(matched[0])
    out = dict(matched[0])
    out["lane"] = "+".join(sorted({str(r["lane"]) for r in matched}))
    out["holders"] = sum(int(r["holders"]) for r in matched)
    out["holder_rows"] = [h for r in matched for h in r["holder_rows"]]
    for key in ("weights_bytes", "drafter_bytes"):
        vals = [r.get(key) for r in matched if isinstance(r.get(key), int)]
        out[key] = int(sum(vals)) if vals else None
    for key in ("apc_bytes", "prompt_cache_bytes", "hybrid_snapshot_bytes", "cache_bytes", "held_bytes"):
        out[key] = int(sum(int(r.get(key) or 0) for r in matched))
    out["lanes"] = len(matched)
    return out


def mlx_memory_report() -> Dict[str, Any]:
    """MLX counters + resident rows: what the host/tray/console should show."""
    stats = mx_memory_stats()
    rows = resident_models()
    active = stats.get("active_bytes")
    cache = stats.get("cache_bytes")
    held = None
    if isinstance(active, int) or isinstance(cache, int):
        held = int(active or 0) + int(cache or 0)
    return {
        "backend": "mlx",
        "active_bytes": active,
        "cache_bytes": cache,
        "peak_bytes": stats.get("peak_bytes"),
        "held_bytes": held,
        "models": rows,
        "holders": int(sum(int(r.get("holders") or 0) for r in rows)),
        "resident_models": int(sum(1 for r in rows if r.get("weights_alive"))),
    }


# ---------------------------------------------------------------------------
# Eject: every holder, then collect + clear
# ---------------------------------------------------------------------------
def _holders_for(model: Optional[str], path: Optional[str]) -> List[Any]:
    holders: Dict[int, Any] = {}
    for lane, model_path, session in _native_session_entries():
        row = {"models": [getattr(h, "model", None) for h in list(session.holders)], "model_path": model_path}
        if _matches(row, model, path) or (model is None and path is None):
            for h in list(session.holders):
                holders[id(h)] = h
    for lane, model_path, shared in _shared_model_entries():
        row = {"models": [getattr(h, "model", None) for h in list(shared.holders)], "model_path": model_path}
        if _matches(row, model, path) or (model is None and path is None):
            for h in list(shared.holders):
                holders[id(h)] = h
    return list(holders.values())


def eject_model(model: Optional[str] = None, *, path: Optional[str] = None, reason: str = "eject") -> Dict[str, Any]:
    """Unload EVERY provider instance holding `model` in this process (None: all
    MLX models), then `gc.collect()` and `mx.clear_cache()`.

    Returns a report: holders found / unloaded / refused (with their errors),
    MLX counters before and after, and `residual` -- the row still resident for
    that model after the eject, if any (a holder mid-generation refuses; a
    reference this module cannot see keeps the weights). An eject NEVER pretends:
    `ok` is True only when the model is no longer resident.
    """
    with _EJECT_LOCK:
        before = mx_memory_stats()
        before_row = process_residency_for(model, path) if (model or path) else None
        holders = _holders_for(model, path)
        unloaded: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for holder in holders:
            row = _holder_row(holder)
            try:
                holder.unload_model(str(getattr(holder, "model", "") or model or ""))
                unloaded.append(row)
            except Exception as exc:  # noqa: BLE001 - one refusing holder must not hide the others
                row["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(row)
        del holders
        collected = gc.collect()
        cleared = clear_mx_cache()
        after = mx_memory_stats()
        residual = process_residency_for(model, path) if (model or path) else None
        if model is None and path is None:
            remaining = [r for r in resident_models() if r.get("weights_alive")]
            residual = remaining[0] if remaining else None
        ok = residual is None and not errors
        report = {
            "ok": ok,
            "reason": reason,
            "model": model,
            "path": path,
            "holders_found": len(unloaded) + len(errors),
            "holders_unloaded": unloaded,
            "holders_refused": errors,
            "gc_collected": collected,
            "cache_cleared": cleared,
            "before": {**before, "row": before_row},
            "after": after,
            "freed_bytes": (int(before.get("active_bytes") or 0) + int(before.get("cache_bytes") or 0))
            - (int(after.get("active_bytes") or 0) + int(after.get("cache_bytes") or 0))
            if isinstance(before.get("active_bytes"), int) and isinstance(after.get("active_bytes"), int) else None,
            "residual": residual,
            "ts": time.time(),
        }
        if residual is not None:
            logger.warning(
                "mlx eject of %s left %s byte(s) resident (%d holder(s) still alive: %s)",
                model or path or "all", residual.get("held_bytes"), int(residual.get("holders") or 0),
                [h.get("type") for h in residual.get("holder_rows") or []],
            )
        else:
            logger.info(
                "mlx eject of %s: %d holder(s) unloaded, cache cleared=%s, active %s -> %s bytes",
                model or path or "all", len(unloaded), cleared, before.get("active_bytes"), after.get("active_bytes"),
            )
        return report
