"""Analytical context-fit estimation (no weights are ever loaded).

`estimate_context_fit()` answers "how large a context could this model
plausibly sustain on this host, and what would it cost in KV memory" from
cheap geometry sources only:

- HF transformers / MLX: the snapshot's `config.json` (local hub cache only);
- GGUF: the file header (`utils.model_cache.read_gguf_geometry`);
- Ollama: `POST {base}/api/show` model_info (short timeout, best-effort);
- LM Studio: no geometry source — unknown.

A previously recorded calibration entry (utils.context_calibration) beats any
estimate: it is a measured settle point on this hardware. Estimates are
labeled as such; unknown stays unknown (ADR 0008) and this function NEVER
raises.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reserve subtracted from every budget basis: the host still needs headroom
# for compute buffers, fragmentation, and the OS itself. Deliberately SMALL —
# the old flat 80%-of-total fraction wrongly reported models as not fitting on
# machines whose real ceiling (e.g. a raised iogpu.wired_limit_mb) was far
# higher. The notes always state basis + reserve so the budget is auditable.
_BUDGET_RESERVE_FLOOR_BYTES = 2 * 1024**3  # 2 GiB
_BUDGET_RESERVE_FRACTION = 0.05
# Fallback basis when no real device ceiling is observable (no wired-limit
# sysctl, no Metal working-set info): a stated fraction, labeled as fallback.
_FALLBACK_CEILING_FRACTION = 0.75
# KV cache assumed f16: 2 bytes per element.
_KV_BYTES_PER_ELEMENT = 2
_OLLAMA_SHOW_TIMEOUT_S = 3.0

_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".gguf", ".npz")


def _default_ollama_base_url() -> str:
    base = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://localhost:11434"
    ).strip().rstrip("/")
    if base and "://" not in base:
        base = f"http://{base}"
    return base


def _resolve_snapshot_dir(model: str) -> Optional[Path]:
    try:
        candidate = Path(model).expanduser()
        if candidate.is_dir():
            return candidate
    except Exception:
        pass
    try:
        from .model_cache import resolve_hf_snapshot_dir

        return resolve_hf_snapshot_dir(model)
    except Exception:
        return None


def _config_geometry_from_dir(snapshot_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        import json

        config_path = snapshot_dir / "config.json"
        if not config_path.is_file():
            return None
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        from ..providers.model_config_fingerprint import model_geometry_for

        return model_geometry_for(config)
    except Exception:
        return None


def _snapshot_weight_bytes(snapshot_dir: Path) -> Optional[int]:
    """Sum of weight-file sizes directly under a snapshot dir (cheap only)."""
    try:
        total = 0
        for entry in snapshot_dir.iterdir():
            if entry.suffix.lower() in _WEIGHT_FILE_SUFFIXES:
                try:
                    total += int(entry.stat().st_size)
                except Exception:
                    continue
        return total or None
    except Exception:
        return None


def _find_gguf_file(model: str) -> Optional[Path]:
    """Best-effort local GGUF file for `model` (direct path or hub cache)."""
    raw = str(model or "").strip()
    try:
        direct = Path(raw).expanduser()
        if direct.is_file() and direct.suffix.lower() == ".gguf":
            return direct
    except Exception:
        pass
    repo_id, selector = raw, ""
    if ":" in raw and "/" in raw:
        repo_id, selector = raw.split(":", 1)
    snapshot_dir = _resolve_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return None
    try:
        ggufs = sorted(
            (p for p in snapshot_dir.rglob("*.gguf") if p.is_file()),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
    except Exception:
        return None
    if selector:
        sel = selector.strip().lower()
        matching = [p for p in ggufs if sel in p.name.lower()]
        if matching:
            return matching[0]
    return ggufs[0] if ggufs else None


def _geometry_from_gguf(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """(normalized geometry, kv_bytes_per_token) from a GGUF header."""
    try:
        from .model_cache import read_gguf_geometry

        raw = read_gguf_geometry(path)
    except Exception:
        raw = None
    if not isinstance(raw, dict):
        return None, None
    n_layers = raw.get("block_count")
    n_heads = raw.get("head_count")
    n_kv_heads = raw.get("head_count_kv") or n_heads
    hidden = raw.get("embedding_length")
    k_len = raw.get("key_length")
    v_len = raw.get("value_length")
    head_dim = k_len
    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads or None
    geometry = {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "max_position_embeddings": raw.get("context_length"),
    }
    if all(v is None for v in geometry.values()):
        return None, None
    kv = None
    if n_layers and n_kv_heads and head_dim:
        k_dim = k_len if isinstance(k_len, int) and k_len > 0 else head_dim
        v_dim = v_len if isinstance(v_len, int) and v_len > 0 else head_dim
        kv = int(n_layers) * int(n_kv_heads) * (int(k_dim) + int(v_dim)) * _KV_BYTES_PER_ELEMENT
    return geometry, kv


def _geometry_from_ollama(model: str, base_url: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        import httpx

        base = str(base_url or "").strip().rstrip("/") or _default_ollama_base_url()
        resp = httpx.post(
            f"{base}/api/show", json={"model": model}, timeout=_OLLAMA_SHOW_TIMEOUT_S
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None
    model_info = data.get("model_info") if isinstance(data, dict) else None
    if not isinstance(model_info, dict):
        return None

    def _by_suffix(suffix: str) -> Optional[int]:
        for key, value in model_info.items():
            if str(key).endswith(suffix):
                try:
                    out = int(value)
                except (TypeError, ValueError):
                    continue
                if out > 0:
                    return out
        return None

    n_layers = _by_suffix(".block_count")
    n_heads = _by_suffix(".attention.head_count")
    n_kv_heads = _by_suffix(".attention.head_count_kv") or n_heads
    hidden = _by_suffix(".embedding_length")
    head_dim = _by_suffix(".attention.key_length")
    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads or None
    geometry = {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "max_position_embeddings": _by_suffix(".context_length"),
    }
    if all(v is None for v in geometry.values()):
        return None
    return geometry


def _kv_bytes_per_token(geometry: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(geometry, dict):
        return None
    n_layers = geometry.get("n_layers")
    n_kv_heads = geometry.get("n_kv_heads")
    head_dim = geometry.get("head_dim")
    if not (n_layers and n_kv_heads and head_dim):
        return None
    # K + V, f16: 2 tensors * n_layers * n_kv_heads * head_dim * 2 bytes.
    return 2 * int(n_layers) * int(n_kv_heads) * int(head_dim) * _KV_BYTES_PER_ELEMENT


def _memory_view() -> Dict[str, Optional[int]]:
    try:
        from .memory import get_memory_snapshot

        snap = get_memory_snapshot()
        ram = snap.get("ram") or {}
        device = snap.get("device") or {}
        return {
            "ram_total_bytes": ram.get("total_bytes"),
            "ram_available_bytes": ram.get("available_bytes"),
            "device_backend": device.get("backend"),
            "device_total_bytes": device.get("total_bytes"),
            "device_allocated_bytes": device.get("allocated_bytes"),
            "device_free_bytes": device.get("free_bytes"),
        }
    except Exception:
        return {
            "ram_total_bytes": None,
            "ram_available_bytes": None,
            "device_backend": None,
            "device_total_bytes": None,
            "device_allocated_bytes": None,
            "device_free_bytes": None,
        }


# The Metal ceiling probes are SHARED with the memory snapshot
# (`utils.memory` owns them: `device.wired_limit_bytes` reports the same
# ceiling this budget uses). Bound as module attributes so tests can pin
# them per-module without touching the real host.
from .memory import (  # noqa: E402 - deliberate late import, module ordering above
    metal_recommended_working_set_bytes as _metal_recommended_working_set_bytes,
    metal_wired_limit_bytes as _metal_wired_limit_bytes,
)


def _memory_budget(memory_full: Dict[str, Optional[int]], notes: List[str]) -> Optional[int]:
    """Memory budget (bytes) available for weights + KV, or None when unknown.

    Basis — a REAL ceiling, not a blanket fraction (first known wins):
    - Metal/MPS: `sysctl iogpu.wired_limit_mb` when set (> 0), else Metal's
      `max_recommended_working_set_size` (mlx device_info), else a stated
      fallback of 75% of device_total / total RAM; currently-allocated device
      memory is deducted.
    - CUDA: `torch.cuda.mem_get_info()` free bytes (already excludes
      allocations — nothing further deducted).
    - No device backend: available RAM.

    A small reserve — max(2 GiB, 5% of the basis) — is subtracted; the notes
    state basis and reserve so the number is auditable. ADVISORY ONLY: no load
    path in core, runtime, or gateway gates on this budget.
    """
    backend = memory_full.get("device_backend")
    device_total = memory_full.get("device_total_bytes")
    device_alloc = memory_full.get("device_allocated_bytes")
    device_alloc = int(device_alloc) if isinstance(device_alloc, int) else 0
    ram_total = memory_full.get("ram_total_bytes")
    ram_available = memory_full.get("ram_available_bytes")

    basis: Optional[int] = None
    basis_label = ""
    deduct_allocated = False
    if backend in ("metal", "mps"):
        deduct_allocated = True
        wired = _metal_wired_limit_bytes()
        if wired is not None:
            basis, basis_label = wired, "iogpu.wired_limit_mb sysctl"
        else:
            recommended = _metal_recommended_working_set_bytes()
            if recommended is not None:
                basis, basis_label = recommended, "Metal max_recommended_working_set_size"
            elif isinstance(device_total, int) and device_total > 0:
                basis = int(_FALLBACK_CEILING_FRACTION * device_total)
                basis_label = (
                    f"fallback {int(_FALLBACK_CEILING_FRACTION * 100)}% of device_total "
                    "(no iogpu.wired_limit_mb sysctl or Metal working-set info)"
                )
            elif isinstance(ram_total, int) and ram_total > 0:
                basis = int(_FALLBACK_CEILING_FRACTION * ram_total)
                basis_label = (
                    f"fallback {int(_FALLBACK_CEILING_FRACTION * 100)}% of total RAM "
                    "(no iogpu.wired_limit_mb sysctl or Metal working-set info)"
                )
    elif backend == "cuda":
        free = memory_full.get("device_free_bytes")
        if isinstance(free, int) and free > 0:
            basis, basis_label = free, "cuda mem_get_info free bytes"
    if basis is None and isinstance(ram_available, int) and ram_available > 0:
        basis, basis_label = ram_available, "available RAM (no device ceiling observable)"
        deduct_allocated = False
    if basis is None or basis <= 0:
        return None
    reserve = max(_BUDGET_RESERVE_FLOOR_BYTES, int(_BUDGET_RESERVE_FRACTION * basis))
    budget = basis - reserve - (device_alloc if deduct_allocated else 0)
    note = f"budget basis: {basis_label} = {int(basis)} bytes; reserve max(2 GiB, 5%) = {reserve} bytes"
    if deduct_allocated:
        note += f"; allocated device memory {device_alloc} bytes deducted"
    notes.append(note)
    if budget <= 0:
        notes.append("budget is non-positive after reserve and allocation deductions")
        return None
    return int(budget)


def _public_memory(memory_full: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    """The `memory` block every result path emits — one key set, always."""
    return {
        "ram_available_bytes": memory_full.get("ram_available_bytes"),
        "device_total_bytes": memory_full.get("device_total_bytes"),
        "device_allocated_bytes": memory_full.get("device_allocated_bytes"),
    }


def _calibration_hit(
    provider: str,
    model: str,
    memory: Dict[str, Optional[int]],
    extra_candidates: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        from .context_calibration import lookup_context_calibration

        candidates: List[str] = []
        # The extra candidates carry the RESOLVED artifact id the ladder
        # records under (gguf_calibration_model_id of the resolved file) —
        # a hub id like `org/Repo-GGUF:q4_k_m` never matches the store
        # without it. Resolved ids first: they are the recording key.
        for candidate in list(extra_candidates or []) + [model, Path(model).name, model.split(":", 1)[0]]:
            candidate = str(candidate or "").strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        for model_id in candidates:
            hit = lookup_context_calibration(
                provider,
                model_id,
                memory.get("device_total_bytes"),
                memory.get("ram_total_bytes"),
            )
            if isinstance(hit, dict):
                return hit
        return None
    except Exception:
        return None


def estimate_context_fit(
    provider: str, model: str, context_length: Optional[int] = None, base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Estimate how much context `provider/model` can sustain on this host.

    Pure observation + arithmetic: geometry comes from configs/headers/server
    metadata (never from loading weights), memory from `get_memory_snapshot`,
    and a measured calibration entry (same provider/model/hardware key) wins
    over any estimate. Unknown members are omitted; `confidence` is one of
    "calibrated" | "estimated" | "unknown". NEVER raises.
    """
    try:
        return _estimate_context_fit(provider, model, context_length, base_url)
    except Exception as exc:  # pragma: no cover - belt and braces
        return {
            "ok": True,
            "provider": str(provider or "").strip().lower(),
            "model": str(model or "").strip(),
            "confidence": "unknown",
            "predicted_max_context": None,
            "fits_weights": None,
            "fits_requested_context": None,
            # Same key set as the normal path's memory block.
            "memory": _public_memory(_memory_view()),
            "notes": [f"estimation failed internally: {exc}"],
        }


def _estimate_context_fit(
    provider: str, model: str, context_length: Optional[int], base_url: Optional[str]
) -> Dict[str, Any]:
    provider_s = str(provider or "").strip().lower()
    model_s = str(model or "").strip()
    notes: List[str] = []
    memory_full = _memory_view()
    memory = _public_memory(memory_full)

    out: Dict[str, Any] = {
        "ok": True,
        "provider": provider_s,
        "model": model_s,
        "memory": memory,
        "notes": notes,
    }
    requested: Optional[int] = None
    if context_length is not None:
        try:
            requested = int(context_length)
        except (TypeError, ValueError):
            requested = None
    if requested is not None and requested > 0:
        out["requested_context_length"] = requested
    else:
        requested = None

    # -- Geometry (source depends on the provider lane; never loads weights) --
    geometry: Optional[Dict[str, Any]] = None
    kv_per_token: Optional[int] = None
    est_weights: Optional[int] = None
    # Resolved artifact ids the calibration store may have recorded under
    # (the ladder keys GGUF settles by gguf_calibration_model_id).
    calibration_extra: List[str] = []

    if provider_s == "huggingface":
        gguf_path = _find_gguf_file(model_s)
        if gguf_path is not None:
            try:
                from .context_calibration import gguf_calibration_model_id

                calibration_extra.append(gguf_calibration_model_id(gguf_path))
            except Exception:
                pass
            geometry, kv_per_token = _geometry_from_gguf(gguf_path)
            try:
                est_weights = int(gguf_path.stat().st_size)
            except Exception:
                est_weights = None
        else:
            snapshot_dir = _resolve_snapshot_dir(model_s)
            if snapshot_dir is not None:
                geometry = _config_geometry_from_dir(snapshot_dir)
                est_weights = _snapshot_weight_bytes(snapshot_dir)
        if geometry is None:
            notes.append("no local config.json or GGUF header found for this model")
    elif provider_s == "mlx":
        snapshot_dir = _resolve_snapshot_dir(model_s)
        if snapshot_dir is not None:
            geometry = _config_geometry_from_dir(snapshot_dir)
            est_weights = _snapshot_weight_bytes(snapshot_dir)
        if geometry is None:
            notes.append("no local snapshot config.json found for this model")
    elif provider_s == "ollama":
        geometry = _geometry_from_ollama(model_s, base_url)
        if geometry is None:
            notes.append("ollama /api/show returned no usable model geometry")
    elif provider_s == "lmstudio":
        notes.append("lmstudio exposes no model geometry source; geometry unknown")
    else:
        notes.append(f"provider '{provider_s}' has no local geometry source")

    if kv_per_token is None:
        kv_per_token = _kv_bytes_per_token(geometry)

    if geometry is not None:
        out["geometry"] = geometry
    if kv_per_token is not None:
        out["kv_bytes_per_token"] = kv_per_token
        notes.append("kv math assumes an f16 KV cache (2 bytes/element)")
    if est_weights is not None:
        out["est_weights_bytes"] = est_weights
    if requested is not None and kv_per_token is not None:
        out["est_kv_bytes"] = int(requested) * int(kv_per_token)

    model_max: Optional[int] = None
    if isinstance(geometry, dict):
        mpe = geometry.get("max_position_embeddings")
        if isinstance(mpe, int) and mpe > 0:
            model_max = mpe

    # -- Budget (a real ceiling, not a blanket fraction) + fit verdicts --
    # `fits_weights` / `fits_requested_context` are additive tri-state fields:
    # true/false when computable, null when a component is unknown. ADVISORY
    # ONLY — no load path anywhere gates on them.
    budget = _memory_budget(memory_full, notes)
    if budget is not None:
        out["budget_bytes"] = budget
    # ALREADY-LOADED weights must not be counted twice: the budget already
    # deducted `device_allocated_bytes`, and a model resident in THIS process
    # sits inside that figure — subtracting est_weights again would report a
    # running model as not fitting (the exact defect class the real-ceiling
    # budget removed). Heuristic (this utility cannot query provider
    # residency without constructing providers): the weights count as already
    # allocated when the process allocation covers them AND they account for
    # the bulk (>= 50%) of it — a small model must not read as "resident"
    # merely because some OTHER allocation dwarfs it.
    device_alloc_raw = memory_full.get("device_allocated_bytes")
    weights_resident = bool(
        est_weights is not None
        and int(est_weights) > 0
        and isinstance(device_alloc_raw, int)
        and not isinstance(device_alloc_raw, bool)
        and int(est_weights) <= device_alloc_raw <= 2 * int(est_weights)
    )
    if weights_resident:
        notes.append(
            "weights appear already resident (device allocation covers est_weights_bytes and is "
            "mostly them) — counted once inside the allocation, not deducted from the budget again"
        )
    fits_weights: Optional[bool] = None
    if budget is not None and est_weights is not None:
        fits_weights = True if weights_resident else int(est_weights) <= budget
    out["fits_weights"] = fits_weights
    fits_requested: Optional[bool] = None
    if budget is not None and est_weights is not None and kv_per_token is not None and requested is not None:
        requested_kv = int(requested) * int(kv_per_token)
        fits_requested = (
            requested_kv <= budget if weights_resident else (int(est_weights) + requested_kv) <= budget
        )
    out["fits_requested_context"] = fits_requested

    # -- Calibration beats estimation (measured settle on this hardware) --
    calibration = _calibration_hit(provider_s, model_s, memory_full, calibration_extra)
    if calibration is not None:
        settled = calibration.get("settled_context")
        if isinstance(settled, int) and settled > 0:
            out["confidence"] = "calibrated"
            out["calibrated_context_length"] = settled
            out["predicted_max_context"] = settled
            notes.append(
                "calibrated: a previous load on this hardware settled at this context"
            )
            return out

    # -- Pure estimate from KV math and the budget --
    if kv_per_token is not None:
        if budget is not None:
            if fits_weights is False:
                # Weights alone exceed the budget: no context fits BESIDE them.
                out["confidence"] = "estimated"
                out["predicted_max_context"] = None
                notes.append("est_weights_bytes exceeds the budget; no context fits beside the weights")
                return out
            if weights_resident:
                # The weights already live inside the deducted allocation:
                # the whole remaining budget is context headroom.
                predicted = max(0, budget // int(kv_per_token))
                notes.append(
                    "predicted_max_context = budget // kv_bytes_per_token "
                    "(weights already resident — counted once)"
                )
            elif est_weights is not None:
                predicted = max(0, (budget - int(est_weights)) // int(kv_per_token))
                notes.append(
                    "predicted_max_context = (budget - est_weights_bytes) // kv_bytes_per_token "
                    "— the context that fits beside the weights"
                )
            else:
                predicted = max(0, budget // int(kv_per_token))
                notes.append(
                    "weight size unknown; predicted_max_context = budget // kv_bytes_per_token "
                    "(weights NOT deducted)"
                )
            if model_max is not None:
                predicted = min(predicted, model_max)
            out["confidence"] = "estimated"
            out["predicted_max_context"] = int(predicted)
            notes.append("estimated from geometry, not measured — a real load must still probe")
            return out
        notes.append("no memory budget observable; cannot estimate")

    out["confidence"] = "unknown"
    out["predicted_max_context"] = None
    return out
