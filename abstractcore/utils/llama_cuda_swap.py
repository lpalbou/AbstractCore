"""Give a plain ``pip install "abstractcore[gpu]"`` the CUDA build of llama.cpp on an NVIDIA Linux host.

The requirement (operator, 2026-11-10): the ``gpu`` install must work fully on a GPU machine, GGUF models
included. PyPI carries ``llama-cpp-python`` as a source build only, so pip resolves abetlen's prebuilt
*processor* wheel (``llama_supports_gpu_offload()`` is False) and every GGUF model runs on the CPU
although PyTorch sees the card. The prebuilt CUDA builds live on abetlen's own index
(https://abetlen.github.io/llama-cpp-python/whl/<build>/), which a pip extra cannot name, and they are
400 to 600 MB each, above PyPI's file limit, so they cannot be re-hosted there either.

The one-line installers already swap the CUDA build in after the install (``scripts/install.sh``,
the framework installer). This module does the same swap from inside the package, once per
environment, the first time AbstractCore is about to import ``llama_cpp`` on a host where it would
otherwise run GGUF on the processor:

1. Linux x86_64 only (the CUDA pages carry ``manylinux_2_35_x86_64`` wheels alone), opt-out with
   ``ABSTRACTCORE_LLAMA_CUDA_SWAP=0``.
2. The installed ``llama_cpp`` is a CUDA build already (``lib/libggml-cuda.so``, the way the installers
   leave it): nothing to do, no pip call. This is the common case on a host set up by the one-liners.
3. PyTorch's CUDA major is read from ``torch/version.py`` on disk (torch is never imported here) and the
   NVIDIA driver from ``nvidia-smi``: CUDA 13 wants driver 580 or newer (build ``cu130``), CUDA 12 wants
   525 or newer (``cu125``). Without a CUDA torch, a driver, or with one too old, the processor build
   stays and the reason is recorded.
4. The swap is the installers' command: ``pip install --no-index --no-deps --force-reinstall
   --find-links <index>/<build>/llama-cpp-python/ llama-cpp-python==<installed version>`` (``uv pip``
   with ``--reinstall-package`` when the environment has no pip, as uv venvs do). Same version, same
   files the installer would have written. One loud log line before and after.
5. The outcome is written next to the package (``site-packages/llama_cpp/.abstractcore-cuda-swap.json``)
   so a failed swap (offline host, read-only site-packages) is not retried at every model load; a
   reinstall of llama-cpp-python removes the marker with the folder and the check runs again.

`ensure_llama_cpp_cuda_build()` returns a dict (``outcome``, ``reason``, ``build``, ``version``) that the
HuggingFace provider records in its runtime info, so a user whose GGUF model still runs on the processor
reads why in the first answer. When ``llama_cpp`` was already imported in this process (the processor
library is loaded), the swap still happens on disk and the outcome says the next process gets the GPU.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

INDEX = "https://abetlen.github.io/llama-cpp-python/whl"
# CUDA major of the installed torch -> (llama.cpp build folder, minimum NVIDIA driver major).
BUILDS: Dict[int, tuple] = {13: ("cu130", 580), 12: ("cu125", 525)}
ENV_OPT_OUT = "ABSTRACTCORE_LLAMA_CUDA_SWAP"
MARKER_NAME = ".abstractcore-cuda-swap.json"
SWAP_TIMEOUT_S = 900

_result: Optional[Dict[str, Any]] = None


def _package_dir(name: str) -> Optional[Path]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if not locations and spec.origin:
        locations = [str(Path(spec.origin).parent)]
    return Path(locations[0]) if locations else None


def llama_cpp_is_cuda_build() -> bool:
    """True when the installed llama-cpp-python ships its CUDA backend. Never imports llama_cpp."""
    pkg = _package_dir("llama_cpp")
    if pkg is None:
        return False
    lib = pkg / "lib"
    return lib.is_dir() and any(lib.glob("libggml-cuda.so*"))


def torch_cuda_major() -> Optional[int]:
    """The CUDA major of the installed torch wheel, read from ``torch/version.py`` (``cuda = '13.0'``);
    None without torch or for a processor/ROCm build. torch is never imported."""
    pkg = _package_dir("torch")
    if pkg is None:
        return None
    try:
        text = (pkg / "version.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"^cuda\s*=\s*['\"](\d+)\.(\d+)['\"]", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def nvidia_driver_major() -> Optional[int]:
    """The NVIDIA driver's major version from ``nvidia-smi``; None when no driver answers."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = (out.stdout or "").strip().splitlines()
    if not first:
        return None
    head = first[0].strip().split(".")[0]
    return int(head) if head.isdigit() else None


def installed_version() -> Optional[str]:
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        return None


def marker_path() -> Optional[Path]:
    pkg = _package_dir("llama_cpp")
    return (pkg / MARKER_NAME) if pkg is not None else None


def _read_marker() -> Optional[Dict[str, Any]]:
    path = marker_path()
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_marker(data: Dict[str, Any]) -> None:
    path = marker_path()
    if path is None:
        return
    try:
        path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        logger.debug("could not write %s: %s", path, exc)


def swap_command(build: str, version: str) -> Optional[List[str]]:
    """The installers' swap for this interpreter: pip when the environment has it, else ``uv pip``
    (uv venvs carry no pip). None when neither is available."""
    links = f"{INDEX}/{build}/llama-cpp-python/"
    requirement = f"llama-cpp-python=={version}"
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall",
                "--find-links", links, requirement]
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, "--no-index", "--no-deps",
                "--reinstall-package", "llama-cpp-python", "--find-links", links, requirement]
    return None


def run_swap(command: List[str]) -> subprocess.CompletedProcess:
    """Run the swap command (separate so tests replace it)."""
    return subprocess.run(command, capture_output=True, text=True, timeout=SWAP_TIMEOUT_S)


def _outcome(outcome: str, reason: str = "", **extra: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {"outcome": outcome, "reason": reason}
    data.update(extra)
    return data


def ensure_llama_cpp_cuda_build() -> Dict[str, Any]:
    """Make sure llama.cpp is its CUDA build on an NVIDIA Linux host before AbstractCore imports it.

    Returns the outcome (cached for the process):
    ``cuda-build-present`` (nothing done), ``swapped`` (the CUDA build is now installed),
    ``swapped-restart-needed`` (installed, but this process already loaded the processor build),
    ``failed`` (the swap ran and did not deliver; ``reason`` says why; not retried until
    llama-cpp-python is reinstalled), ``skipped`` (not applicable here; ``reason`` says why)."""
    global _result
    if _result is not None:
        return dict(_result)
    _result = _decide_and_swap()
    return dict(_result)


def _decide_and_swap() -> Dict[str, Any]:
    if not sys.platform.startswith("linux"):
        return _outcome("skipped", "not Linux")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return _outcome("skipped", f"no prebuilt CUDA llama.cpp for {platform.machine()} (x86_64 only)")
    if os.environ.get(ENV_OPT_OUT, "").strip() == "0":
        return _outcome("skipped", f"{ENV_OPT_OUT}=0")
    version = installed_version()
    if version is None or _package_dir("llama_cpp") is None:
        return _outcome("skipped", "llama-cpp-python is not installed")
    if llama_cpp_is_cuda_build():
        return _outcome("cuda-build-present", "", version=version)
    marker = _read_marker()
    if marker and marker.get("version") == version and marker.get("outcome") in {"failed", "swapped"}:
        # "swapped" with a processor build on disk means someone reinstalled the CPU wheel over it
        # without removing the marker (pip keeps unknown files in the folder): run again.
        if marker.get("outcome") == "failed":
            return dict(marker)
    major = torch_cuda_major()
    if major is None:
        return _outcome("skipped", "PyTorch has no CUDA build here", version=version)
    if major not in BUILDS:
        return _outcome("skipped", f"no prebuilt llama.cpp build for CUDA {major}", version=version)
    build, floor = BUILDS[major]
    driver = nvidia_driver_major()
    if driver is None:
        return _outcome("skipped", "no NVIDIA driver answers (nvidia-smi)", version=version, build=build)
    if driver < floor:
        return _outcome("skipped", f"PyTorch uses CUDA {major}, which needs NVIDIA driver {floor} or newer "
                        f"(this one is {driver})", version=version, build=build)
    command = swap_command(build, version)
    if command is None:
        data = _outcome("failed", "neither pip nor uv is available to install the CUDA build",
                        version=version, build=build)
        _write_marker(data)
        return data
    already_loaded = "llama_cpp" in sys.modules
    logger.warning(
        "AbstractCore: llama-cpp-python %s here is the processor build and this machine has an NVIDIA card "
        "(CUDA %s torch, driver %s): installing its CUDA build (%s) from %s so GGUF models run on the GPU; "
        "one-time, a few hundred MB. Set %s=0 to keep the processor build.",
        version, major, driver, build, INDEX, ENV_OPT_OUT,
    )
    started = time.time()
    try:
        proc = run_swap(command)
    except (OSError, subprocess.SubprocessError) as exc:
        data = _outcome("failed", f"the swap command did not run: {exc}", version=version, build=build,
                        command=" ".join(command))
        _write_marker(data)
        logger.warning("AbstractCore: llama.cpp CUDA build not installed: %s", data["reason"])
        return data
    tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-6:]
    if proc.returncode != 0 or not llama_cpp_is_cuda_build():
        why = (f"exit {proc.returncode}" if proc.returncode != 0 else
               "the command succeeded but the installed files are not a CUDA build")
        data = _outcome("failed", f"{why}: {' | '.join(tail)[:600]}", version=version, build=build,
                        command=" ".join(command), seconds=round(time.time() - started, 1))
        _write_marker(data)
        logger.warning("AbstractCore: llama.cpp CUDA build not installed (%s); GGUF models stay on the processor. "
                       "The one-line installer does this step too: see docs/installation.md.", data["reason"])
        return data
    data = _outcome("swapped-restart-needed" if already_loaded else "swapped", "", version=version, build=build,
                    command=" ".join(command), seconds=round(time.time() - started, 1))
    _write_marker(data)
    if already_loaded:
        logger.warning("AbstractCore: llama.cpp CUDA build %s installed in %.0f s; this process had already loaded "
                       "the processor build, the next one runs GGUF models on the GPU.", build, data["seconds"])
    else:
        logger.warning("AbstractCore: llama.cpp CUDA build %s installed in %.0f s; GGUF models run on the GPU.",
                       build, data["seconds"])
    return data
