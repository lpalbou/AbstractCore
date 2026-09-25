"""One door to every in-process model residency backend (mission M2,
2026-09-25): MLX (`mlx_residency`), the HuggingFace provider (`hf_residency`:
transformers on torch + GGUF on llama.cpp) and in-process embeddings
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
