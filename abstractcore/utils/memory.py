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
import sys
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


def _device_snapshot_backend() -> Dict[str, Any]:
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


def _torch_cuda_reserved_bytes() -> Optional[int]:
    """torch's CUDA reserved bytes over every visible GPU, or None when torch
    is not imported, has no CUDA, or has not initialized it (reading never
    initializes CUDA)."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    try:
        if cuda is None or not cuda.is_available() or not cuda.is_initialized():
            return None
        return int(sum(int(cuda.memory_reserved(i)) for i in range(int(cuda.device_count()))))
    except Exception:
        return None


def _device_snapshot() -> Dict[str, Any]:
    """The backend figure (`_device_snapshot_backend`) plus what EVERY
    in-process allocator pins, so the tray/console can show ONE truthful
    process figure (mission MEM2, 2026-09-25):

    - `torch_mps_allocated_bytes` / `torch_mps_driver_bytes`: torch's MPS
      allocator (transformers text models, embeddings, voice/vision on torch).
      The driver figure INCLUDES torch's freed-but-pooled buffers -- what the
      process still pins after a model is dropped without `empty_cache()`.
    - `llama_cpp_bytes`: weights + context state of every live llama.cpp
      engine (GGUF), from the residency rows. Memory-mapped weights are
      file-backed (see the module docstring) but still count against what the
      process holds resident.
    - `metal_process_allocated_bytes`: the Metal device's allocation for THIS
      process (`torch.mps.driver_allocated_memory()` = `MTLDevice.
      currentAllocatedSize`). MLX, llama.cpp and torch all allocate from that
      one device, so this counter is the UNION. Verified 2026-09-25 (M5 Max,
      torch 2.x + MLX + llama-cpp-python 0.3.35, one process): a 2 GiB MLX
      array moved it by exactly 2,147,483,648 bytes (MLX's freed-but-cached
      buffers stay inside it until `mx.clear_cache()`); a 1 GiB torch tensor
      by 1 GiB more; a Qwen3.5-4B Q4_K_M GGUF on Metal (n_ctx 8192) by
      3.59 GB, all of it returned on close. Known only while torch is
      imported (reading it never imports torch).
    - `torch_cuda_reserved_bytes`: on CUDA, what torch's caching allocator
      has reserved on every visible GPU for this process
      (`torch.cuda.memory_reserved`), freed-but-cached blocks included. Read
      only when torch is imported and CUDA is already initialized.
    - `process_held_bytes`: the accelerator memory this process holds, as far
      as the figures AbstractCore can read go. `process_held_basis` says
      which:
        * `metal_device_counter`: the Metal device counter above (a
          measurement covering MLX, torch and llama.cpp);
        * `cuda_device_counter[+llama_cpp_bytes(estimated)]`: torch's CUDA
          reserved bytes, plus the llama.cpp figure when a GGUF engine is
          live (llama.cpp allocates on CUDA outside torch's allocator);
        * `sum:<fields>`: MLX held (measured: active + cache) +
          `llama_cpp_bytes`, marked `(estimated)` when non-zero (weights plus
          an f16 KV ESTIMATE from the GGUF geometry, which overstates hybrid
          models: 3.80 GB estimated vs 3.59 GB measured above).
      Other native libraries that allocate GPU memory on their own
      (whisper.cpp, CoreML, onnxruntime...) are counted only by a device
      counter (Metal); in a `sum:` or CUDA figure they are NOT included.
      CPU-side heap (tokenizers, Python objects) is never in it --
      `process.rss_bytes` / the footprint cover that. The per-backend
      figures are ATTRIBUTIONS of this total, never added on top of it.

    Backends are read from `sys.modules` only: a report must never import
    torch (hundreds of MB, and it breaks the GGUF Metal import-order guard).
    """
    out = _device_snapshot_backend()
    out.setdefault("torch_mps_allocated_bytes", None)
    out.setdefault("torch_mps_driver_bytes", None)
    out.setdefault("llama_cpp_bytes", None)
    out.setdefault("metal_process_allocated_bytes", None)
    out.setdefault("process_held_bytes", None)
    out.setdefault("process_held_basis", None)
    out.setdefault("torch_cuda_reserved_bytes", None)
    try:
        from ..providers.hf_residency import hf_memory_report

        hf = hf_memory_report()
        out["torch_mps_allocated_bytes"] = hf.get("torch_mps_allocated_bytes")
        out["torch_mps_driver_bytes"] = hf.get("torch_mps_driver_bytes")
        out["llama_cpp_bytes"] = int(hf.get("llama_cpp_bytes") or 0)
    except Exception:
        pass
    driver = out.get("torch_mps_driver_bytes")
    if isinstance(driver, int) and not isinstance(driver, bool):
        # torch is imported: its driver counter is the device's allocation
        # for this process, MLX and llama.cpp buffers included.
        out["metal_process_allocated_bytes"] = int(driver)
        out["process_held_bytes"] = int(driver)
        out["process_held_basis"] = "metal_device_counter"
        return out
    llama = out.get("llama_cpp_bytes")
    llama_i = int(llama) if isinstance(llama, int) and not isinstance(llama, bool) else 0
    cuda_reserved = _torch_cuda_reserved_bytes()
    if cuda_reserved is not None:
        out["torch_cuda_reserved_bytes"] = cuda_reserved
        out["process_held_bytes"] = int(cuda_reserved) + llama_i
        out["process_held_basis"] = "cuda_device_counter" + ("+llama_cpp_bytes(estimated)" if llama_i else "")
        return out
    total = 0
    parts = []
    for key in ("mlx_held_bytes", "llama_cpp_bytes"):
        value = out.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
            parts.append(f"{key}(estimated)" if key == "llama_cpp_bytes" and value else key)
    if not parts and isinstance(out.get("allocated_bytes"), int):
        total, parts = int(out["allocated_bytes"]), ["allocated_bytes"]
    out["process_held_bytes"] = int(total) if parts else None
    # Which figures were summed; `llama_cpp_bytes` carries an f16 KV estimate.
    out["process_held_basis"] = ("sum:" + "+".join(parts)) if parts else None
    return out


def _resident_snapshot() -> Dict[str, Any]:
    """Every in-process model whose weights are alive, whoever holds them,
    across backends: MLX (`mlx_residency`), the HuggingFace provider
    (`hf_residency`: transformers + GGUF), in-process embeddings. Rows share
    one shape (`lane`, `models`, `holders`, `weights_bytes`, `cache_bytes`,
    `held_bytes`, `weights_alive`, `backend`). Never raises; a failing backend
    is reported as `error` in its block, never dropped silently."""
    backends: Dict[str, Any] = {}
    rows: list = []
    try:
        import mlx.core  # noqa: F401
    except Exception:
        backends["mlx"] = None
    else:
        try:
            from ..providers.mlx_residency import mlx_memory_report

            report = mlx_memory_report()
            for row in report.get("models") or []:
                row.pop("holder_rows", None)
                row.setdefault("backend", "mlx")
            backends["mlx"] = report
        except Exception as exc:  # noqa: BLE001
            backends["mlx"] = {"backend": "mlx", "error": f"{type(exc).__name__}: {exc}"}
    try:
        from ..providers.hf_residency import hf_memory_report

        report = hf_memory_report()
        for row in report.get("models") or []:
            row.pop("holder_rows", None)
        backends["huggingface"] = report
    except Exception as exc:  # noqa: BLE001
        backends["huggingface"] = {"backend": "huggingface", "error": f"{type(exc).__name__}: {exc}"}
    # The manager module imports sentence-transformers (and torch) eagerly;
    # a process that never built an EmbeddingManager cannot hold one, so read
    # the module only when it is already imported -- never import it here.
    manager_mod = sys.modules.get("abstractcore.embeddings.manager")
    if manager_mod is None:
        backends["embeddings"] = None
    else:
        try:
            report = manager_mod.embeddings_memory_report()
            for row in report.get("models") or []:
                row.pop("holder_rows", None)
            backends["embeddings"] = report
        except Exception as exc:  # noqa: BLE001
            backends["embeddings"] = {"backend": "embeddings", "error": f"{type(exc).__name__}: {exc}"}
    total = 0
    for report in backends.values():
        if isinstance(report, dict):
            rows.extend(r for r in (report.get("models") or []) if isinstance(r, dict))
            value = report.get("held_bytes")
            if isinstance(value, int) and not isinstance(value, bool):
                total += value
    return {
        "backends": backends,
        "models": rows,
        "resident_models": len(rows),
        "holders": int(sum(int(r.get("holders") or 0) for r in rows)),
        "total_held_bytes": int(total),
    }


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
                    "mlx_held_bytes": int|None,     # active + cache: what MLX pins
                    "torch_mps_allocated_bytes": int|None,  # torch live tensors on MPS (this process)
                    "torch_mps_driver_bytes": int|None,     # torch MPS driver incl. its pool (this process)
                    "llama_cpp_bytes": int|None,    # GGUF weights + KV allocation (est.) of live llama.cpp engines
                    "metal_process_allocated_bytes": int|None,  # the Metal device's allocation for this process (torch present)
                    "process_held_basis": str|None,  # "metal_device_counter" or "sum:<the fields summed>"
                    "process_held_bytes": int|None},  # accelerator memory this process pins: the Metal device counter (measured) when torch is imported, else MLX held + llama.cpp (KV estimated)
         "resident": {"backends": {"mlx": <held block>|None,
                                   "huggingface": {"backend", "held_bytes", "models", "holders",
                                                   "torch_mps_allocated_bytes", "torch_mps_driver_bytes",
                                                   "llama_cpp_bytes", ...},
                                   "embeddings": {"backend", "held_bytes", "models", "holders", ...}},
                      "models": [every row across backends, each with "backend"],
                      "resident_models": int, "holders": int,
                      "total_held_bytes": int},     # what the process pins through model holders
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
        "resident": _resident_snapshot(),
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
