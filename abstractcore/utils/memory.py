"""Host memory visibility helpers.

`get_memory_snapshot()` answers "how much memory is there / in use, right now"
for capacity decisions (e.g. whether another local model fits). Pure
observation: every probe is individually guarded, unknown values stay None,
and the function never raises — visibility must not be able to break a caller.

This module also owns the Apple-silicon accelerator ceilings the context
estimator budgets against (`metal_wired_limit_bytes`,
`metal_recommended_working_set_bytes`) and the cross-process accelerator-heap
probe (`ioreg` IOAccelerator PerformanceStatistics). The distinction matters:

- `device.allocated_bytes` is THIS PROCESS's accelerator memory (mlx active
  memory) — truthful per-process, blind to every other process.
- `device.host_in_use_bytes` is ACCELERATOR-HEAP memory across processes:
  buffers the Metal driver allocated, wherever they were allocated from. An
  MLX model — in this process or in an MLX-engine server such as LM Studio —
  shows up here.

What `host_in_use_bytes` does NOT see, and must never be presented as seeing:
**memory-mapped GGUF/llama.cpp weights**. llama.cpp mmaps the `.gguf` and wraps
those pages with `newBufferWithBytesNoCopy`, so they are file-backed and never
become driver-allocated accelerator memory — a fully offloaded 90 GB GGUF moves
this counter by ~0. Such weights are visible instead as process RSS
(`process.rss_bytes`) and as the model's own reported weight size
(`est_weights_bytes`). It is therefore NOT the host's total memory use, and a
resident-model total legitimately exceeds it. Use `ram` for the system picture.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional


def _ram_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "total_bytes": None,
        "available_bytes": None,
        "used_bytes": None,
        "percent": None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["total_bytes"] = int(vm.total)
        out["available_bytes"] = int(vm.available)
        out["used_bytes"] = int(vm.used)
        out["percent"] = float(vm.percent)
    except Exception:
        pass
    return out


def darwin_phys_footprint_bytes(pid: Optional[int] = None) -> Optional[int]:
    """This process's PHYSICAL FOOTPRINT on macOS (`proc_pid_rusage`,
    `ri_phys_footprint`): the number Activity Monitor's "Memory" column and
    `footprint`/`vmmap --summary` report. It INCLUDES Metal buffers (MLX
    weights and KV caches), which RSS does NOT -- a gateway holding 88 GB of
    MLX memory had an RSS of 6 GB (2026-09-25). None off macOS or on failure."""
    try:
        import ctypes
        import os
        import platform

        if platform.system() != "Darwin":
            return None
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        rusage_info_v2 = 2
        # struct rusage_info_v2: uint8 ri_uuid[16] then 18 uint64 fields;
        # ri_phys_footprint is the 8th uint64 (offset 16 + 7 * 8).
        buf = ctypes.create_string_buffer(16 + 18 * 8)
        rc = libproc.proc_pid_rusage(ctypes.c_int(int(pid or os.getpid())), ctypes.c_int(rusage_info_v2), buf)
        if rc != 0:
            return None
        value = int.from_bytes(buf.raw[72:80], "little", signed=False)
        return value if value > 0 else None
    except Exception:
        return None


def _process_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {"rss_bytes": None, "footprint_bytes": None}
    try:
        import psutil

        out["rss_bytes"] = int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    out["footprint_bytes"] = darwin_phys_footprint_bytes()
    return out


def metal_wired_limit_bytes() -> Optional[int]:
    """`sysctl iogpu.wired_limit_mb` in bytes when set and > 0, else None.

    On Apple Silicon this sysctl is the wired-memory ceiling the OS actually
    enforces for GPU allocations; operators raise it precisely so larger
    models fit, so when it is set it IS the budget basis. Shared by the
    memory snapshot (`device.wired_limit_bytes`) and the context estimator's
    budget — one probe, one truth."""
    try:
        import platform
        import subprocess

        if platform.system() != "Darwin":
            return None
        proc = subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0:
            return None
        mb = int(str(proc.stdout).strip())
        if mb <= 0:
            return None
        return mb * 1024 * 1024
    except Exception:
        return None


def metal_recommended_working_set_bytes() -> Optional[int]:
    """Metal's own `max_recommended_working_set_size` via mlx device_info."""
    try:
        import mlx.core as mx

        device_info = getattr(mx, "device_info", None) or getattr(
            getattr(mx, "metal", None), "device_info", None
        )
        if not callable(device_info):
            return None
        value = dict(device_info()).get("max_recommended_working_set_size")
        out = int(value)
        return out if out > 0 else None
    except Exception:
        return None


# `ioreg -r -c IOAccelerator -l` parsing (same discipline as the gateway's
# GPU-utilization reader): PerformanceStatistics is a `{ "key"=value, ... }`
# dict printed on one line; on Apple silicon the "In use system memory"
# statistic is driver-allocated accelerator-heap memory currently in use across
# processes, in bytes. It counts allocator-backed buffers only — memory-mapped
# GGUF weights (`newBufferWithBytesNoCopy` over an mmap) are absent from it.
_IOREG_PERF_STATS_RE = re.compile(r'"PerformanceStatistics"\s*=\s*\{(.*?)\}', re.DOTALL)
_IOREG_KV_RE = re.compile(r'"([^"]+)"\s*=\s*([^,}]+)')
_IOREG_IN_USE_KEY = "In use system memory"


def parse_ioreg_accelerator_in_use_bytes(text: str) -> Optional[int]:
    """Sum of `"In use system memory"` across IOAccelerator
    PerformanceStatistics blocks in raw `ioreg` output, or None when absent.

    This is accelerator-HEAP memory across processes, not whole-system memory
    use: it tracks driver-allocated buffers, so memory-mapped GGUF weights are
    excluded (see the module docstring).

    The key must match EXACTLY — `"In use system memory (driver)"` is a
    different (driver-only) statistic and is deliberately not counted."""
    try:
        total = 0
        found = False
        for block in _IOREG_PERF_STATS_RE.findall(str(text or "")):
            for key, raw_value in _IOREG_KV_RE.findall(block):
                if key != _IOREG_IN_USE_KEY:
                    continue
                try:
                    value = int(str(raw_value).strip())
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    total += value
                    found = True
        return total if found else None
    except Exception:
        return None


def _ioreg_accelerator_in_use_bytes(timeout_s: float = 1.0) -> Optional[int]:
    """Host-wide accelerator memory in use (bytes) via `ioreg`, or None."""
    try:
        import platform
        import shutil
        import subprocess

        if platform.system() != "Darwin":
            return None
        exe = shutil.which("ioreg") or "/usr/sbin/ioreg"
        proc = subprocess.run(
            [exe, "-r", "-c", "IOAccelerator", "-l"],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
        if proc.returncode != 0:
            return None
        return parse_ioreg_accelerator_in_use_bytes(proc.stdout or "")
    except Exception:
        return None


def _device_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "backend": None,
        "allocated_bytes": None,
        "total_bytes": None,
        "free_bytes": None,
        # Metal-only extras (cross-process accelerator heap; stay None on other
        # backends — CUDA's total/free already ARE device-wide truth):
        "host_in_use_bytes": None,
        "wired_limit_bytes": None,
        # MLX allocator truth for THIS process (metal only): live buffers,
        # freed-but-cached buffers, high-water mark, and their sum -- what the
        # process actually pins in unified memory. `allocated_bytes` stays the
        # live figure for compatibility; `mlx_held_bytes` is the one to show.
        "mlx_active_bytes": None,
        "mlx_cache_bytes": None,
        "mlx_peak_bytes": None,
        "mlx_held_bytes": None,
    }

    # Apple Silicon via MLX (unified memory; Metal exposes no free-bytes query).
    try:
        import mlx.core as mx

        metal = getattr(mx, "metal", None)
        is_available = getattr(metal, "is_available", None)
        if callable(is_available) and not is_available():
            raise RuntimeError("Metal unavailable")
        # `mx.get_active_memory` is the current spelling; older mlx keeps it
        # under `mx.metal.get_active_memory`.
        get_active = getattr(mx, "get_active_memory", None) or getattr(metal, "get_active_memory", None)
        if callable(get_active):
            out["backend"] = "metal"
            try:
                # PROCESS-LOCAL truth: mlx active memory of THIS process only.
                out["allocated_bytes"] = int(get_active())
                out["mlx_active_bytes"] = out["allocated_bytes"]
            except Exception:
                pass
            for name, key in (("get_cache_memory", "mlx_cache_bytes"), ("get_peak_memory", "mlx_peak_bytes")):
                fn = getattr(mx, name, None) or getattr(metal, name, None)
                if callable(fn):
                    try:
                        out[key] = int(fn())
                    except Exception:
                        pass
            if isinstance(out["mlx_active_bytes"], int) or isinstance(out["mlx_cache_bytes"], int):
                out["mlx_held_bytes"] = int(out["mlx_active_bytes"] or 0) + int(out["mlx_cache_bytes"] or 0)
            try:
                device_info = getattr(mx, "device_info", None) or getattr(metal, "device_info", None)
                if callable(device_info):
                    total = dict(device_info()).get("memory_size")
                    if total is not None:
                        out["total_bytes"] = int(total)
            except Exception:
                pass
            # CROSS-PROCESS truth: driver-allocated accelerator-heap memory in
            # use across ALL processes (ioreg IOAccelerator
            # PerformanceStatistics). An MLX model resident in an MLX-engine
            # server (LM Studio) shows up here, not in `allocated_bytes`.
            # Memory-mapped GGUF weights do NOT appear here at all — they are
            # file-backed no-copy buffers; see the module docstring.
            out["host_in_use_bytes"] = _ioreg_accelerator_in_use_bytes()
            # The enforced accelerator ceiling: the wired-limit sysctl when
            # set, else Metal's recommended working set. None when neither is
            # observable.
            out["wired_limit_bytes"] = (
                metal_wired_limit_bytes() or metal_recommended_working_set_bytes()
            )
            return out
    except Exception:
        pass

    # NVIDIA CUDA via torch.
    try:
        import torch

        if torch.cuda.is_available():
            out["backend"] = "cuda"
            try:
                out["allocated_bytes"] = int(torch.cuda.memory_allocated())
            except Exception:
                pass
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                out["free_bytes"] = int(free_bytes)
                out["total_bytes"] = int(total_bytes)
            except Exception:
                pass
            return out
    except Exception:
        pass

    # Apple MPS via torch (only reached when MLX is absent).
    try:
        import torch

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            out["backend"] = "mps"
            try:
                out["allocated_bytes"] = int(torch.mps.current_allocated_memory())
            except Exception:
                pass
            return out
    except Exception:
        pass

    return out


def _host_snapshot() -> Dict[str, Any]:
    try:
        from .hostinfo import get_host_identity

        return get_host_identity()
    except Exception:
        return {"host_id": None, "host_name": None, "kind": "local"}


def prompt_cache_store_bytes(stats: Any) -> Optional[int]:
    """Total bytes a provider's prompt-cache store holds, from its
    `get_prompt_cache_stats()` payload; None when unknown.

    Sums the per-key `bytes` figures where known (base-store `meta_by_key`
    rows) plus the MLX hybrid `snapshots.bytes` total. An empty store (empty
    `keys`, no snapshots) is a KNOWN 0; a store whose keys carry no byte
    figures is unknown → None. Never raises."""
    try:
        if not isinstance(stats, dict):
            return None
        total = 0
        known = False
        keys = stats.get("keys")
        meta_by_key = stats.get("meta_by_key")
        meta_by_key = meta_by_key if isinstance(meta_by_key, dict) else {}
        if isinstance(keys, list):
            for key in keys:
                row = meta_by_key.get(str(key))
                value = row.get("bytes") if isinstance(row, dict) else None
                if isinstance(value, int) and not isinstance(value, bool):
                    total += value
                    known = True
        snapshots = stats.get("snapshots")
        if isinstance(snapshots, dict):
            value = snapshots.get("bytes")
            if isinstance(value, int) and not isinstance(value, bool):
                total += value
                known = True
        if known:
            return total
        # An empty store is a known zero, not an unknown.
        snapshot_count = snapshots.get("count") if isinstance(snapshots, dict) else None
        if isinstance(keys, list) and not keys and not snapshot_count:
            return 0
        return None
    except Exception:
        return None


def get_memory_snapshot() -> Dict[str, Any]:
    """Best-effort host memory snapshot (system RAM, this process, device backend).

    Shape:
        {"ts": <unix float>,
         "ram": {"total_bytes", "available_bytes", "used_bytes", "percent"},
         "process": {"rss_bytes",
                     "footprint_bytes"},           # macOS phys_footprint: INCLUDES Metal/MLX buffers (RSS does not)
         "device": {"backend": "metal"|"cuda"|"mps"|None,
                    "allocated_bytes": int|None,   # THIS process (metal/mps: mlx/torch active memory)
                    "total_bytes": int|None,
                    "free_bytes": int|None,
                    "host_in_use_bytes": int|None,  # metal: cross-process accelerator HEAP (ioreg);
                                                    #   excludes memory-mapped GGUF weights
                    "wired_limit_bytes": int|None,  # metal: enforced ceiling (sysctl, else Metal working set)
                    "mlx_active_bytes": int|None,   # metal: MLX live buffers (this process)
                    "mlx_cache_bytes": int|None,    # metal: MLX freed-but-cached buffers (this process)
                    "mlx_peak_bytes": int|None,
                    "mlx_held_bytes": int|None},    # active + cache: what the process pins
         "held": {"backend": "mlx", "active_bytes", "cache_bytes", "peak_bytes", "held_bytes",
                  "resident_models": int, "holders": int,
                  "models": [{"lane", "model_path", "models", "holders", "weights_bytes",
                              "drafter_bytes", "apc_bytes", "prompt_cache_bytes",
                              "hybrid_snapshot_bytes", "cache_bytes", "held_bytes", ...}]}
                 | None,                            # None when MLX is absent
         "host": {"host_id", "host_name", "kind"}}

    `held` is the PROCESS-level residency truth (`abstractcore.providers.
    mlx_residency`): every MLX model whose weights are alive in this process
    and who holds them -- including holders no runtime pool can reach. A UI
    that lists "no models loaded" while `held.held_bytes` > 0 is lying; the
    tray/console/gateway read this block so they cannot.

    Missing/unknowable values are None; this function never raises.
    """
    try:
        ts = float(time.time())
    except Exception:
        ts = 0.0
    return {
        "ts": ts,
        "ram": _ram_snapshot(),
        "process": _process_snapshot(),
        "device": _device_snapshot(),
        "held": _held_snapshot(),
        "host": _host_snapshot(),
    }


def _held_snapshot() -> Optional[Dict[str, Any]]:
    """Process-level MLX residency (see `get_memory_snapshot`). None without MLX."""
    try:
        import mlx.core  # noqa: F401  (no MLX: nothing can be held by it)
    except Exception:
        return None
    try:
        from ..providers.mlx_residency import mlx_memory_report

        report = mlx_memory_report()
        # `holder_rows` carry object ids for diagnostics; keep the snapshot JSON-light.
        for row in report.get("models") or []:
            row.pop("holder_rows", None)
        return report
    except Exception as exc:  # noqa: BLE001 - visibility must never break a caller
        return {"backend": "mlx", "error": f"{type(exc).__name__}: {exc}"}
