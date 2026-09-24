"""What kind of machine is this, for the purpose of running local models.

`host_profile()` is contract A (`host_profile_v1`) of the models & engines
surfaces: one dict that every console (core CLI/TUI/web, gateway CLI/TUI/web)
reads to pre-select sensible defaults and to judge "does this model fit".

THE CEILING IS THE SAME CEILING THE CONTEXT ESTIMATOR USES. The Apple-silicon
probes (`iogpu.wired_limit_mb`, Metal's recommended working set) live in
`utils.memory`, and the order in which they win is the order
`utils.context_estimate._memory_budget` applies: wired limit, then Metal's
recommended working set, then a stated 75% fallback. A model browser that
disagreed with the context estimator about how much memory the machine has
would tell an operator a model fits and then fail to load it.

UNKNOWN STAYS UNKNOWN. Every probe is guarded; a value that cannot be observed
is `null`, never a guess, and this function never raises.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "HOST_PROFILE_SCHEMA",
    "host_profile",
    "normalize_os",
    "normalize_arch",
    "default_model_store_paths",
    "disk_free_bytes",
    "utc_now_iso",
]

HOST_PROFILE_SCHEMA = "host_profile_v1"

# Fallback basis when no real accelerator ceiling is observable. Kept equal to
# `context_estimate._FALLBACK_CEILING_FRACTION` on purpose (one policy).
_FALLBACK_CEILING_FRACTION = 0.75
_CACHE_TTL_S = 5.0
_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"at": 0.0, "value": None}


def utc_now_iso() -> str:
    """ISO-8601 UTC with a `Z` suffix -- the `generated_at` of every payload."""

    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_os(value: Optional[str] = None) -> str:
    raw = str(value or platform.system()).strip().lower()
    if raw.startswith("win"):
        return "windows"
    if raw in {"darwin", "macos", "mac", "osx"}:
        return "darwin"
    if raw.startswith("linux"):
        return "linux"
    return raw or "unknown"


def normalize_arch(value: Optional[str] = None) -> str:
    raw = str(value or platform.machine()).strip().lower()
    if raw in {"arm64", "aarch64", "armv8", "arm64e"}:
        return "arm64"
    if raw in {"x86_64", "amd64", "x64", "i686-64"}:
        return "x86_64"
    return raw or "unknown"


def _run(argv: List[str], timeout: float = 3.0) -> Optional[str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or ""


def _ram_total_and_available() -> Tuple[Optional[int], Optional[int]]:
    try:
        from .memory import _ram_snapshot

        ram = _ram_snapshot()
        total = ram.get("total_bytes")
        available = ram.get("available_bytes")
        if isinstance(total, int):
            return total, available if isinstance(available, int) else None
    except Exception:
        pass
    # psutil missing: the OS still knows the total.
    if normalize_os() == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        try:
            return int(str(out).strip()), None
        except Exception:
            return None, None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(size), None
    except Exception:
        return None, None


def _nvidia_gpus() -> List[Dict[str, Any]]:
    """`nvidia-smi` rows `{name, total_bytes, free_bytes}`; [] when absent."""

    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = _run([exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"], timeout=5.0)
    rows: List[Dict[str, Any]] = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append(
                {
                    "name": parts[0],
                    "total_bytes": int(float(parts[1])) * 1024 * 1024,
                    "free_bytes": int(float(parts[2])) * 1024 * 1024,
                }
            )
        except Exception:
            continue
    return rows


def _rocm_present() -> bool:
    return bool(shutil.which("rocm-smi") or shutil.which("amd-smi"))


def _apple_chip_name() -> Optional[str]:
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    name = str(out or "").strip()
    return name or None


def _torch_cuda() -> Optional[Dict[str, Any]]:
    """Fallback CUDA view through the memory snapshot (torch), only when
    nvidia-smi is unavailable. Importing torch is slow, so it is the last
    resort, never the first probe."""

    try:
        from .memory import _device_snapshot

        device = _device_snapshot()
    except Exception:
        return None
    if device.get("backend") != "cuda":
        return None
    return {"name": None, "total_bytes": device.get("total_bytes"), "free_bytes": device.get("free_bytes")}


def _home() -> Path:
    try:
        return Path.home()
    except Exception:
        return Path(os.path.expanduser("~"))


def _display_path(path: Path) -> str:
    try:
        rel = path.expanduser().resolve().relative_to(_home().resolve())
        return "~/" + rel.as_posix() if str(rel) != "." else "~"
    except Exception:
        return str(path)


def default_model_store_paths(os_name: Optional[str] = None) -> Dict[str, Path]:
    """Where each engine stores weights by default on this host.

    Reuses `utils.model_cache` (HF + LM Studio candidates, env overrides
    included) and falls back to each engine's documented default when no
    directory exists yet -- disk space for a FIRST download is exactly when
    the directory does not exist.
    """

    os_id = normalize_os(os_name)
    home = _home()
    out: Dict[str, Path] = {}

    hf: Optional[Path] = None
    try:
        from .model_cache import default_hf_hub_cache_dirs

        dirs = default_hf_hub_cache_dirs()
        hf = dirs[0] if dirs else None
    except Exception:
        hf = None
    if hf is None:
        env = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
        if env:
            hf = Path(env)
        elif os.getenv("HF_HOME"):
            hf = Path(str(os.getenv("HF_HOME"))) / "hub"
        else:
            hf = home / ".cache" / "huggingface" / "hub"
    out["hf_cache"] = hf

    lm: Optional[Path] = None
    try:
        from .model_cache import default_lmstudio_model_dirs

        dirs = default_lmstudio_model_dirs()
        lm = dirs[0] if dirs else None
    except Exception:
        lm = None
    out["lmstudio"] = lm or (home / ".lmstudio" / "models")

    ollama_env = str(os.getenv("OLLAMA_MODELS") or "").strip()
    if ollama_env:
        ollama = Path(ollama_env)
    elif os_id == "linux" and Path("/usr/share/ollama/.ollama/models").is_dir():
        # The official Linux installer runs the daemon as the `ollama` user.
        ollama = Path("/usr/share/ollama/.ollama/models")
    else:
        ollama = home / ".ollama" / "models"
    out["ollama"] = ollama
    return out


def disk_free_bytes(path: Path) -> Optional[int]:
    """Free bytes on the filesystem that holds `path` (or its nearest existing
    ancestor -- a store that does not exist yet will be created there)."""

    try:
        probe = Path(path).expanduser()
        while not probe.exists():
            parent = probe.parent
            if parent == probe:
                break
            probe = parent
        return int(shutil.disk_usage(str(probe)).free)
    except Exception:
        return None


def _disk_block(os_name: str) -> Dict[str, Dict[str, Any]]:
    block: Dict[str, Dict[str, Any]] = {}
    for key, path in default_model_store_paths(os_name).items():
        p = Path(path).expanduser()
        block[key] = {
            "path": _display_path(p),
            "abs_path": str(p),
            "exists": p.is_dir(),
            "free_bytes": disk_free_bytes(p),
        }
    return block


def _metal_ceiling() -> Tuple[Optional[int], Optional[str]]:
    try:
        from .memory import metal_recommended_working_set_bytes, metal_wired_limit_bytes
    except Exception:
        return None, None
    wired = metal_wired_limit_bytes()
    if wired:
        return int(wired), "metal_wired_limit"
    recommended = metal_recommended_working_set_bytes()
    if recommended:
        return int(recommended), "metal_recommended"
    return None, None


def _build_profile() -> Dict[str, Any]:
    os_id = normalize_os()
    arch = normalize_arch()
    ram_total, ram_available = _ram_total_and_available()
    notes: List[str] = []

    accelerator = "none"
    gpu_name: Optional[str] = None
    gpu_count = 0
    unified = False
    vram: Optional[int] = None
    vram_free: Optional[int] = None
    ceiling: Optional[int] = None
    ceiling_source: Optional[str] = None

    if os_id == "darwin" and arch == "arm64":
        accelerator = "metal"
        unified = True
        gpu_name = _apple_chip_name()
        gpu_count = 1
        ceiling, ceiling_source = _metal_ceiling()
    else:
        gpus = _nvidia_gpus()
        if not gpus:
            cuda = _torch_cuda()
            gpus = [cuda] if cuda else []
        if gpus:
            accelerator = "cuda"
            gpu_count = len(gpus)
            names = sorted({str(g.get("name")) for g in gpus if g.get("name")})
            gpu_name = names[0] if len(names) == 1 else (", ".join(names) if names else None)
            totals = [g.get("total_bytes") for g in gpus if isinstance(g.get("total_bytes"), int)]
            frees = [g.get("free_bytes") for g in gpus if isinstance(g.get("free_bytes"), int)]
            vram = sum(totals) if totals else None
            vram_free = sum(frees) if frees else None
            if vram:
                ceiling, ceiling_source = vram, "cuda_total"
            if gpu_count > 1:
                notes.append(f"{gpu_count} GPUs: vram_bytes is the sum; a model split across GPUs pays some overhead")
        elif _rocm_present():
            accelerator = "rocm"
            notes.append("ROCm tools found; VRAM is not measured (no portable query), ceiling falls back to RAM")

    if ceiling is None and isinstance(ram_total, int) and ram_total > 0:
        ceiling = int(_FALLBACK_CEILING_FRACTION * ram_total)
        ceiling_source = "ram_75pct"
        if accelerator == "metal":
            notes.append(
                "no iogpu.wired_limit_mb sysctl and no mlx to read Metal's working set; "
                "ceiling is 75% of unified memory"
            )

    # FREE NOW: what could be allocated without evicting anything. On unified
    # memory and CPU that is available RAM (capped at the ceiling: the GPU
    # cannot use more than its ceiling even when RAM is free). On CUDA it is
    # free VRAM.
    free_now: Optional[int] = None
    if accelerator == "cuda":
        free_now = vram_free
    elif isinstance(ram_available, int):
        free_now = min(ram_available, ceiling) if isinstance(ceiling, int) else ram_available

    return {
        "schema": HOST_PROFILE_SCHEMA,
        "os": os_id,
        "arch": arch,
        "accelerator": accelerator,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "unified_memory": unified,
        "ram_bytes": ram_total,
        "vram_bytes": vram,
        "ceiling_bytes": ceiling,
        "ceiling_source": ceiling_source,
        "free_now_bytes": free_now,
        "disk": _disk_block(os_id),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "notes": notes,
        "generated_at": utc_now_iso(),
    }


def _build_light_profile() -> Dict[str, Any]:
    """os / arch / Apple-silicon accelerator / RAM only: no GPU tool, no torch,
    no mlx. What the recommended-model TIER needs (it reads accelerator ==
    metal and unified memory), cheap enough for import-time config seeding."""

    os_id = normalize_os()
    arch = normalize_arch()
    ram_total, _available = _ram_total_and_available()
    metal = os_id == "darwin" and arch == "arm64"
    return {
        "schema": HOST_PROFILE_SCHEMA,
        "os": os_id,
        "arch": arch,
        # Only Apple silicon is detected in the light reading; CUDA/ROCm need
        # the full probe and read as "none" here (the tier treats every
        # non-metal host alike).
        "accelerator": "metal" if metal else "none",
        "unified_memory": metal,
        "ram_bytes": ram_total,
        "light": True,
        "generated_at": utc_now_iso(),
    }


def host_profile(
    *,
    refresh: bool = False,
    builder: Optional[Callable[[], Dict[str, Any]]] = None,
    light: bool = False,
) -> Dict[str, Any]:
    """Contract A: the `host_profile_v1` dict. Never raises.

    Cached for a few seconds so one catalog payload (dozens of fit verdicts)
    measures the machine once; `refresh=True` forces a new reading.

    `light=True` returns the cached full profile when there is one, otherwise
    a light reading (`_build_light_profile`: no GPU tools, no torch, no mlx)
    that is never cached. Importing AbstractCore seeds a fresh config store,
    and that must not load an engine.
    """

    now = time.monotonic()
    if light and builder is None and not refresh:
        with _cache_lock:
            cached = _cache.get("value")
            if cached is not None and now - float(_cache.get("at") or 0.0) < _CACHE_TTL_S:
                return dict(cached)
        try:
            return _build_light_profile()
        except Exception:  # pragma: no cover - defensive: a profile never raises
            return {"schema": HOST_PROFILE_SCHEMA, "os": normalize_os(), "arch": normalize_arch(),
                    "accelerator": "none", "ram_bytes": None, "light": True}
    with _cache_lock:
        cached = _cache.get("value")
        if not refresh and builder is None and cached is not None and now - float(_cache.get("at") or 0.0) < _CACHE_TTL_S:
            return dict(cached)
    try:
        value = (builder or _build_profile)()
    except Exception as exc:  # pragma: no cover - defensive: a profile never raises
        value = {
            "schema": HOST_PROFILE_SCHEMA,
            "os": normalize_os(),
            "arch": normalize_arch(),
            "accelerator": "none",
            "gpu_name": None,
            "gpu_count": 0,
            "unified_memory": False,
            "ram_bytes": None,
            "vram_bytes": None,
            "ceiling_bytes": None,
            "ceiling_source": None,
            "free_now_bytes": None,
            "disk": {},
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "notes": [f"host profile failed: {exc}"],
            "generated_at": utc_now_iso(),
        }
    if builder is None:
        with _cache_lock:
            _cache["value"] = dict(value)
            _cache["at"] = now
    return dict(value)
