"""Process-level HuggingFace residency: what THIS process holds through the
`huggingface` provider (transformers on torch / GGUF on llama.cpp), and who.

Why this exists: `mlx_residency` closed the
"No models loaded over 86 GB" lie for MLX. The HuggingFace provider had the
SAME hole with a different mechanism: its instances do not share weights, so a
boot-time summarizer, a per-request override client or an old runtime that
built its own `HuggingFaceProvider` for the same model holds a SECOND FULL
COPY (transformers: another set of MPS tensors; GGUF: another llama.cpp
context + compute buffers over the shared mmap). The runtime pool's eject
freed only ITS instance and reported `unloaded: true, not_loaded` while the
other copy stayed resident, unlisted and unreachable (measured: 251 MB of
SmolLM2 weights and ~1 GB of llama.cpp buffers surviving a "successful" eject).

This module is the process-wide truth for that provider, in the same shape as
`mlx_residency` so every consumer (runtime listing, gateway console, tray)
can fold it in:

- `resident_models()` -- every model whose weights are alive in this process
  through a HuggingFace provider instance, with holders (each a full copy) and
  the bytes it pins (weights, KV/prompt caches, hybrid snapshots, llama.cpp
  context state).
- `process_residency_for(model)` -- the row for one model (or None).
- `eject_model(model)` -- unload EVERY holder, then `gc.collect()` and return
  torch's MPS allocator pool to the OS (`torch.mps.empty_cache()`), and report
  what is still held. `ok` is True only when nothing of that model remains.
- `hf_memory_report()` -- torch MPS allocator figures (this process), the
  llama.cpp bytes the rows account for, and the rows.

Never imports torch, transformers or llama_cpp on its own: it reads them from
`sys.modules` only. Importing torch here would (a) cost hundreds of MB in a
process that never loads a transformers model and (b) break the GGUF Metal
safety guard that depends on llama_cpp being imported before torch.
"""
from __future__ import annotations

import gc
import logging
import sys
import threading
import time
import weakref
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("abstractcore.providers.huggingface")

_EJECT_LOCK = threading.RLock()

# Every HuggingFaceProvider built in this process (weak: a collected instance
# leaves on its own). Registered by HuggingFaceProvider.__init__.
_HF_PROVIDERS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def register_provider(provider: Any) -> None:
    try:
        _HF_PROVIDERS.add(provider)
    except TypeError:
        pass


def registered_providers() -> List[Any]:
    return [p for p in list(_HF_PROVIDERS)]


# ---------------------------------------------------------------------------
# Allocator figures (None when the backend is not even imported)
# ---------------------------------------------------------------------------
def torch_mps_stats() -> Dict[str, Optional[int]]:
    """torch's MPS allocator for THIS process: `allocated_bytes` (live tensors)
    and `driver_bytes` (what the Metal driver holds for torch, pool included).
    None when torch is not imported or MPS is unavailable."""
    out: Dict[str, Optional[int]] = {"allocated_bytes": None, "driver_bytes": None}
    torch = sys.modules.get("torch")
    if torch is None:
        return out
    try:
        mps = getattr(torch, "mps", None)
        backends = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or backends is None or not backends.is_available():
            return out
        fn = getattr(mps, "current_allocated_memory", None)
        if callable(fn):
            out["allocated_bytes"] = int(fn())
        fn = getattr(mps, "driver_allocated_memory", None)
        if callable(fn):
            out["driver_bytes"] = int(fn())
    except Exception:
        pass
    return out


def release_torch_mps_cache() -> bool:
    """Return torch's MPS pool to the OS. True when it ran."""
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        mps = getattr(torch, "mps", None)
        backends = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or backends is None or not backends.is_available():
            return False
        sync = getattr(mps, "synchronize", None)
        if callable(sync):
            sync()
        fn = getattr(mps, "empty_cache", None)
        if callable(fn):
            fn()
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------
def _torch_tensor_type() -> Optional[type]:
    torch = sys.modules.get("torch")
    return getattr(torch, "Tensor", None) if torch is not None else None


def _storage_key_and_bytes(tensor: Any) -> Tuple[Any, int]:
    try:
        storage = tensor.untyped_storage()
        return (storage.data_ptr(), str(tensor.device)), int(storage.nbytes())
    except Exception:
        try:
            return id(tensor), int(tensor.numel()) * int(tensor.element_size())
        except Exception:
            return id(tensor), 0


def module_bytes(module: Any) -> Optional[int]:
    """Bytes of a torch module's parameters + buffers (unique storages), or None.
    Fakes/tests: an object exposing `nbytes` counts as one array; an object
    exposing `parameters()` yielding items with `numel()`/`element_size()` is
    summed."""
    if module is None:
        return None
    try:
        params = getattr(module, "parameters", None)
        if callable(params):
            seen: Dict[Any, int] = {}
            items = list(params())
            buffers = getattr(module, "buffers", None)
            if callable(buffers):
                items.extend(list(buffers()))
            for item in items:
                key, nbytes = _storage_key_and_bytes(item)
                seen[key] = nbytes
            return int(sum(seen.values()))
    except Exception:
        pass
    nbytes = getattr(module, "nbytes", None)
    if isinstance(nbytes, int) and not isinstance(nbytes, bool):
        return int(nbytes)
    return None


def reachable_tensor_bytes(root: Any, *, max_depth: int = 10, max_nodes: int = 100_000) -> int:
    """Unique torch-tensor storage bytes reachable from `root` through python
    containers and attribute dicts (KV caches, snapshots). Bounded; never raises."""
    if root is None:
        return 0
    tensor_type = _torch_tensor_type()
    seen: set = set()
    storages: Dict[Any, int] = {}
    frontier: List[Tuple[Any, int]] = [(root, 0)]
    n = 0
    while frontier and n < max_nodes:
        obj, depth = frontier.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        n += 1
        try:
            if tensor_type is not None and isinstance(obj, tensor_type):
                key, nbytes = _storage_key_and_bytes(obj)
                storages[key] = nbytes
                continue
            nbytes = getattr(obj, "nbytes", None)
            if tensor_type is None and isinstance(nbytes, int) and not isinstance(nbytes, bool) \
                    and not isinstance(obj, (dict, list, tuple, set)):
                storages[id(obj)] = int(nbytes)  # fakes in tests
                continue
            if depth >= max_depth or isinstance(obj, (str, bytes, int, float, bool, type(None))):
                continue
            children: List[Any] = []
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
    return int(sum(storages.values()))


def llama_state_bytes(root: Any, *, max_depth: int = 10, max_nodes: int = 50_000) -> int:
    """Bytes of llama.cpp `LlamaState` snapshots (serialized KV + logits)
    reachable from `root` (a prompt-cache store, a LlamaRAMCache...)."""
    if root is None:
        return 0
    seen: set = set()
    total = 0
    frontier: List[Tuple[Any, int]] = [(root, 0)]
    n = 0
    while frontier and n < max_nodes:
        obj, depth = frontier.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        n += 1
        try:
            if type(obj).__name__ == "LlamaState":
                size = getattr(obj, "llama_state_size", None)
                if not isinstance(size, int):
                    size = len(getattr(obj, "llama_state", b"") or b"")
                total += int(size or 0)
                for attr in ("scores", "input_ids"):
                    arr = getattr(obj, attr, None)
                    nbytes = getattr(arr, "nbytes", None)
                    if isinstance(nbytes, int):
                        total += int(nbytes)
                continue
            if depth >= max_depth or isinstance(obj, (str, bytes, int, float, bool, type(None))):
                continue
            children: List[Any] = []
            if isinstance(obj, dict):
                children.extend(obj.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                children.extend(obj)
            elif hasattr(obj, "__dict__"):
                children.extend(vars(obj).values())
            for child in children:
                if not isinstance(child, (str, bytes, int, float, bool, type(None))):
                    frontier.append((child, depth + 1))
        except Exception:
            continue
    return int(total)


def llama_kv_alloc_bytes(metadata: Any, n_ctx: Optional[int], *, bytes_per_elem: int = 2) -> Optional[int]:
    """The KV cache llama.cpp ALLOCATES for a context of `n_ctx` tokens, from
    the GGUF metadata (`<arch>.block_count`, `.attention.head_count_kv`,
    `.attention.key_length` / `embedding_length` / `head_count`), f16 K and V.
    This is what the process holds from the moment the engine is built --
    `llama_state_get_size` only measures the tokens in use (measured:
    0.05 GB of state beside a 4.7 GB allocation). None when the
    metadata does not describe the geometry. An estimate: it is labelled so."""
    try:
        if not isinstance(metadata, dict) or not n_ctx:
            return None
        arch = str(metadata.get("general.architecture") or "").strip()
        if not arch:
            return None

        def _int_key(*keys: str) -> Optional[int]:
            for key in keys:
                value = metadata.get(key)
                if value is None:
                    continue
                try:
                    return int(str(value).strip())
                except Exception:
                    continue
            return None

        n_layer = _int_key(f"{arch}.block_count")
        n_head = _int_key(f"{arch}.attention.head_count")
        n_head_kv = _int_key(f"{arch}.attention.head_count_kv") or n_head
        head_dim = _int_key(f"{arch}.attention.key_length")
        if head_dim is None:
            n_embd = _int_key(f"{arch}.embedding_length")
            head_dim = (n_embd // n_head) if (n_embd and n_head) else None
        if not (n_layer and n_head_kv and head_dim):
            return None
        return int(n_ctx) * int(n_layer) * 2 * int(n_head_kv) * int(head_dim) * int(bytes_per_elem)
    except Exception:
        return None


def llama_bytes(llm: Any) -> Dict[str, Optional[int]]:
    """A `llama_cpp.Llama`'s weights (`llama_model_size`: bytes of the loaded
    tensors -- file-backed when memory-mapped), its KV ALLOCATION for `n_ctx`
    (`kv_alloc_bytes`, estimated from the GGUF geometry) and the context state
    in use (`llama_state_get_size`)."""
    out: Dict[str, Optional[int]] = {"weights_bytes": None, "state_bytes": None, "kv_alloc_bytes": None,
                                     "n_ctx": None, "mmap": None}
    if llm is None:
        return out
    llama_cpp = sys.modules.get("llama_cpp")
    try:
        params = getattr(llm, "model_params", None)
        if params is not None and hasattr(params, "use_mmap"):
            out["mmap"] = bool(params.use_mmap)
    except Exception:
        pass
    try:
        n_ctx = getattr(llm, "n_ctx", None)
        out["n_ctx"] = int(n_ctx()) if callable(n_ctx) else None
    except Exception:
        pass
    try:
        out["kv_alloc_bytes"] = llama_kv_alloc_bytes(getattr(llm, "metadata", None), out["n_ctx"])
    except Exception:
        pass
    if llama_cpp is None:
        # Fakes/tests: a bare object with nbytes counts as weights.
        nbytes = getattr(llm, "nbytes", None)
        if isinstance(nbytes, int) and not isinstance(nbytes, bool):
            out["weights_bytes"] = int(nbytes)
        return out
    try:
        model = getattr(getattr(llm, "_model", None), "model", None)
        fn = getattr(llama_cpp, "llama_model_size", None)
        if model is not None and callable(fn):
            out["weights_bytes"] = int(fn(model))
    except Exception:
        pass
    try:
        ctx = getattr(getattr(llm, "_ctx", None), "ctx", None)
        fn = getattr(llama_cpp, "llama_state_get_size", None)
        if ctx is not None and callable(fn):
            out["state_bytes"] = int(fn(ctx))
    except Exception:
        pass
    if out["weights_bytes"] is None:
        nbytes = getattr(llm, "nbytes", None)
        if isinstance(nbytes, int) and not isinstance(nbytes, bool):
            out["weights_bytes"] = int(nbytes)
    return out


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def _holder_alive(holder: Any) -> bool:
    return any(getattr(holder, attr, None) is not None for attr in ("llm", "model_instance", "pipeline"))


def _holder_row(holder: Any) -> Dict[str, Any]:
    lane = "gguf" if getattr(holder, "llm", None) is not None else "transformers"
    row: Dict[str, Any] = {
        "id": id(holder),
        "model": getattr(holder, "model", None),
        "instance_loaded": _holder_alive(holder),
        "type": type(holder).__name__,
        "lane": lane,
        "device": getattr(holder, "device", None),
    }
    weights: Optional[int] = None
    cache = 0
    path: Optional[str] = None
    if lane == "gguf":
        info = llama_bytes(getattr(holder, "llm", None))
        weights = info.get("weights_bytes")
        row["state_bytes"] = info.get("state_bytes")
        row["kv_alloc_bytes"] = info.get("kv_alloc_bytes")
        row["n_ctx"] = info.get("n_ctx")
        row["mmap"] = info.get("mmap")
        # What the engine holds: the KV allocation for n_ctx (estimated from
        # the geometry) when known, else the state in use (a floor).
        kv = info.get("kv_alloc_bytes")
        # The KV allocation is COMPUTED from the GGUF geometry assuming f16 K/V
        # (llama.cpp's default cache type), not read from the engine: say so.
        row["kv_bytes_estimated"] = isinstance(kv, int)
        cache += int(kv if isinstance(kv, int) else (info.get("state_bytes") or 0))
        cache += llama_state_bytes(getattr(holder, "_prompt_cache_store", None))
        path = getattr(getattr(holder, "llm", None), "model_path", None)
    else:
        model_obj = getattr(holder, "model_instance", None)
        if model_obj is None:
            model_obj = getattr(getattr(holder, "pipeline", None), "model", None)
        weights = module_bytes(model_obj)
        snaps = getattr(holder, "_transformers_snapshots", None)
        cache += reachable_tensor_bytes(snaps) if snaps else 0
        cache += reachable_tensor_bytes(getattr(holder, "_prompt_cache_store", None))
        source = getattr(holder, "_transformers_source", None)
        path = getattr(source, "model", None) if source is not None else None
    row["weights_bytes"] = weights
    row["cache_bytes"] = int(cache)
    row["held_bytes"] = int((weights or 0) + cache)
    row["model_path"] = str(path) if path else None
    return row


def resident_models() -> List[Dict[str, Any]]:
    """Every model whose weights are alive in this process through a
    HuggingFace provider instance, ONE ROW PER MODEL NAME. Unlike MLX, holders
    do not share weights: `holders` == full copies, and `weights_bytes` is the
    SUM over the copies (what the process actually pins)."""
    by_model: Dict[str, Dict[str, Any]] = {}
    for holder in registered_providers():
        try:
            if not _holder_alive(holder):
                continue
            hrow = _holder_row(holder)
        except Exception:
            continue
        name = str(hrow.get("model") or "").strip()
        row = by_model.get(name)
        if row is None:
            row = by_model[name] = {
                "lane": hrow["lane"],
                "backend": "huggingface",
                "model_path": hrow.get("model_path"),
                "models": [name] if name else [],
                "holders": 0,
                "holder_rows": [],
                "weights_bytes": None,
                "drafter_bytes": None,
                "apc_bytes": 0,
                "prompt_cache_bytes": 0,
                "hybrid_snapshot_bytes": 0,
                "cache_bytes": 0,
                "held_bytes": 0,
                "runtime_present": False,
                "weights_alive": True,
                "shared_weights": False,
                "copies": 0,
                "mmap": hrow.get("mmap"),
            }
        row["holders"] += 1
        row["copies"] += 1
        row["holder_rows"].append(hrow)
        if hrow.get("kv_bytes_estimated"):
            # `cache_bytes` / `held_bytes` include an f16 KV estimate.
            row["kv_bytes_estimated"] = True
        if row["lane"] != hrow["lane"]:
            row["lane"] = "transformers+gguf"
        if isinstance(hrow.get("weights_bytes"), int):
            row["weights_bytes"] = int(row["weights_bytes"] or 0) + int(hrow["weights_bytes"])
        row["cache_bytes"] += int(hrow.get("cache_bytes") or 0)
        row["prompt_cache_bytes"] += int(hrow.get("cache_bytes") or 0)
        row["held_bytes"] = int((row["weights_bytes"] or 0) + row["cache_bytes"])
        if row.get("model_path") is None and hrow.get("model_path"):
            row["model_path"] = hrow["model_path"]
    return list(by_model.values())


def _matches(row: Dict[str, Any], model: Optional[str], path: Optional[str]) -> bool:
    from .mlx_residency import _matches as mlx_matches

    return mlx_matches(row, model, path)


def process_residency_for(model: Optional[str], path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    matched = [row for row in resident_models() if _matches(row, model, path) and row.get("weights_alive")]
    if not matched:
        return None
    if len(matched) == 1:
        return dict(matched[0])
    out = dict(matched[0])
    out["holders"] = sum(int(r["holders"]) for r in matched)
    out["copies"] = sum(int(r.get("copies") or 0) for r in matched)
    out["holder_rows"] = [h for r in matched for h in r["holder_rows"]]
    vals = [r.get("weights_bytes") for r in matched if isinstance(r.get("weights_bytes"), int)]
    out["weights_bytes"] = int(sum(vals)) if vals else None
    for key in ("cache_bytes", "prompt_cache_bytes", "held_bytes"):
        out[key] = int(sum(int(r.get(key) or 0) for r in matched))
    return out


def hf_memory_report() -> Dict[str, Any]:
    rows = resident_models()
    mps = torch_mps_stats()
    llama_total = 0
    for row in rows:
        for h in row.get("holder_rows") or []:
            if h.get("lane") == "gguf":
                llama_total += int(h.get("held_bytes") or 0)
    return {
        "backend": "huggingface",
        "torch_mps_allocated_bytes": mps.get("allocated_bytes"),
        "torch_mps_driver_bytes": mps.get("driver_bytes"),
        "llama_cpp_bytes": int(llama_total),
        "held_bytes": int(sum(int(r.get("held_bytes") or 0) for r in rows)),
        "models": rows,
        "holders": int(sum(int(r.get("holders") or 0) for r in rows)),
        "resident_models": int(sum(1 for r in rows if r.get("weights_alive"))),
    }


# ---------------------------------------------------------------------------
# Eject: every holder, then collect + return the MPS pool
# ---------------------------------------------------------------------------
def _holders_for(model: Optional[str], path: Optional[str]) -> List[Any]:
    holders: Dict[int, Any] = {}
    for holder in registered_providers():
        if not _holder_alive(holder):
            continue
        row = {"models": [getattr(holder, "model", None)], "model_path": ""}
        try:
            hpath = _holder_row(holder).get("model_path") or ""
            row["model_path"] = hpath
        except Exception:
            pass
        if (model is None and path is None) or _matches(row, model, path):
            holders[id(holder)] = holder
    return list(holders.values())


def eject_model(model: Optional[str] = None, *, path: Optional[str] = None, reason: str = "eject") -> Dict[str, Any]:
    """Unload EVERY HuggingFace provider instance holding `model` (None: all),
    then `gc.collect()` and `torch.mps.empty_cache()`. The report names the
    holders unloaded / refused, the allocator before/after, and `residual`:
    the row still resident afterwards, if any. `ok` never pretends."""
    with _EJECT_LOCK:
        before = torch_mps_stats()
        before_row = process_residency_for(model, path) if (model or path) else None
        holders = _holders_for(model, path)
        unloaded: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for holder in holders:
            row = {"id": id(holder), "model": getattr(holder, "model", None), "type": type(holder).__name__,
                   "lane": "gguf" if getattr(holder, "llm", None) is not None else "transformers"}
            try:
                holder.unload_model(str(getattr(holder, "model", "") or model or ""))
                if _holder_alive(holder):
                    raise RuntimeError("unload_model returned but the instance still reports the model loaded")
                unloaded.append(row)
            except Exception as exc:  # noqa: BLE001 - one refusing holder must not hide the others
                row["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(row)
        del holders
        collected = gc.collect()
        released = release_torch_mps_cache()
        after = torch_mps_stats()
        residual = process_residency_for(model, path) if (model or path) else None
        if model is None and path is None:
            remaining = [r for r in resident_models() if r.get("weights_alive")]
            residual = remaining[0] if remaining else None
        ok = residual is None and not errors
        freed = None
        if isinstance(before.get("driver_bytes"), int) and isinstance(after.get("driver_bytes"), int):
            freed = int(before["driver_bytes"]) - int(after["driver_bytes"])
        report = {
            "ok": ok,
            "reason": reason,
            "model": model,
            "path": path,
            "holders_found": len(unloaded) + len(errors),
            "holders_unloaded": unloaded,
            "holders_refused": errors,
            "gc_collected": collected,
            "cache_cleared": released,
            "before": {**before, "row": before_row},
            "after": after,
            "freed_bytes": freed,
            "residual": residual,
            "ts": time.time(),
        }
        if residual is not None:
            logger.warning(
                "huggingface eject of %s left %s byte(s) resident (%d holder(s) still alive: %s)",
                model or path or "all", residual.get("held_bytes"), int(residual.get("holders") or 0),
                [h.get("lane") for h in residual.get("holder_rows") or []],
            )
        else:
            logger.info(
                "huggingface eject of %s: %d holder(s) unloaded, mps pool released=%s, driver %s -> %s bytes",
                model or path or "all", len(unloaded), released, before.get("driver_bytes"), after.get("driver_bytes"),
            )
        return report
