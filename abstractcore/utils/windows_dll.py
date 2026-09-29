"""Let llama.cpp's CUDA build find the cuBLAS / cudart libraries that PyTorch's NVIDIA wheels ship.

Windows and Linux (the module keeps its first, Windows-only name).

llama-cpp-python's prebuilt Windows CUDA wheels (abetlen's index, installed by install.ps1 for
the gpu setting) do not bundle cuBLAS or the CUDA runtime. Its loader
(`llama_cpp._ctypes_extensions.load_shared_library`) only adds its own folder and
``%CUDA_PATH%\\bin`` to the DLL search, then loads ``llama.dll`` with ``winmode=0`` -- the
legacy search order, where dependent DLLs resolve through ``PATH``. PyTorch's CUDA wheel carries
the same CUDA major's ``cublas64_XX.dll`` / ``cudart64_XX.dll`` in ``<torch>\\lib``; install.ps1
pairs the stacks (torch cu130 with llama cu130, torch cu126 with llama cu125), so pointing the
loader at that folder is enough on a machine without the CUDA toolkit (backlog 0988).

`prepare_llama_cpp_import()` is called before AbstractCore's first ``import llama_cpp``. It is a
no-op off Windows, when torch is absent, or when torch has no ``lib`` folder, and it never imports
torch (``importlib.util.find_spec`` only). The folder goes into both ``os.add_dll_directory`` (for
loads that use the safe search flags) and the front of ``PATH`` (for llama.cpp's ``winmode=0``
load), the same two steps llama-cpp-python takes for its own folder.

Linux (backlog 0989): llama-cpp-python's manylinux CUDA wheels (``cu124``/``cu125``/``cu130``/
``cu132``) link ``libggml-cuda.so`` against ``libcudart.so.N`` / ``libcublas.so.N`` without bundling
them, and the NVIDIA wheels PyTorch depends on unpack those to ``site-packages/nvidia/<lib>/lib``
(CUDA 12) or ``site-packages/nvidia/cu13/lib`` (CUDA 13), which the dynamic loader does not search.
Measured on a Linux + NVIDIA gpu install: ``import llama_cpp`` failed with "libcudart.so.12: cannot
open shared object file" unless torch had been imported first (torch preloads its copies). So when
the installed llama.cpp is a CUDA build (its ``lib`` folder has ``libggml-cuda.so``), the CUDA
runtime, cuBLASLt and cuBLAS files found there are preloaded with ``RTLD_GLOBAL`` -- the way torch
does it -- so the loader resolves llama.cpp's dependencies to them. Every major found is preloaded
(their sonames differ and the symbols are versioned); torch is still never imported.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional

_prepared: Optional[List[str]] = None
_handles: list = []


def torch_lib_dir() -> Optional[Path]:
    """``<torch package>/lib`` when torch is installed and has that folder, else None.
    Never imports torch."""

    try:
        spec = importlib.util.find_spec("torch")
    except Exception:
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if not locations and spec.origin:
        locations = [str(Path(spec.origin).parent)]
    for location in locations:
        lib = Path(location) / "lib"
        if lib.is_dir():
            return lib
    return None


# The CUDA libraries llama.cpp's Linux CUDA build links, in the order they are preloaded
# (cuBLAS needs cuBLASLt, both need the runtime).
LINUX_CUDA_LIBRARY_STEMS = ("libcudart.so.", "libcublasLt.so.", "libcublas.so.")


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
    """True when the installed llama-cpp-python ships its CUDA backend (``lib/libggml-cuda.so*``).
    Never imports llama_cpp."""

    pkg = _package_dir("llama_cpp")
    if pkg is None:
        return False
    lib = pkg / "lib"
    return lib.is_dir() and any(lib.glob("libggml-cuda.so*"))


def linux_nvidia_cuda_libraries() -> List[Path]:
    """The CUDA runtime / cuBLASLt / cuBLAS files of the installed NVIDIA wheels, in load order.

    Looks in every ``site-packages/nvidia/<name>/lib`` folder (CUDA 12 wheels: ``cuda_runtime``,
    ``cublas``; CUDA 13 wheels: ``cu13``) for files named exactly ``<stem><major>``."""

    try:
        spec = importlib.util.find_spec("nvidia")
    except Exception:
        return []
    if spec is None:
        return []
    folders: List[Path] = []
    for root in spec.submodule_search_locations or []:
        try:
            children = sorted(Path(root).iterdir())
        except OSError:
            continue
        folders.extend(child / "lib" for child in children if (child / "lib").is_dir())
    out: List[Path] = []
    for stem in LINUX_CUDA_LIBRARY_STEMS:
        for folder in folders:
            for candidate in sorted(folder.glob(stem + "*")):
                if candidate.name[len(stem):].isdigit() and candidate.is_file():
                    out.append(candidate)
    return out


def prepare_llama_cpp_import() -> List[str]:
    """Make llama.cpp's CUDA build find PyTorch's CUDA libraries before it loads.

    Windows: adds ``<torch>\\lib`` to the DLL search and ``PATH``; returns the folders added.
    Linux: when llama.cpp is a CUDA build, preloads the NVIDIA wheels' CUDA runtime and cuBLAS
    (``RTLD_GLOBAL``); returns the files preloaded. Empty elsewhere, without torch/NVIDIA wheels,
    or for a CPU/Metal/Vulkan llama.cpp. Idempotent: the work is done once per process."""

    global _prepared
    if _prepared is not None:
        return list(_prepared)
    added: List[str] = []
    if sys.platform == "win32":
        lib = torch_lib_dir()
        if lib is not None:
            folder = str(lib)
            add = getattr(os, "add_dll_directory", None)
            if callable(add):
                try:
                    # Keep the handle alive: closing it removes the folder again.
                    _handles.append(add(folder))
                except OSError:
                    pass
            entries = os.environ.get("PATH", "").split(os.pathsep)
            if folder not in entries:
                os.environ["PATH"] = folder + (os.pathsep + os.environ["PATH"] if os.environ.get("PATH") else "")
            added.append(folder)
    elif sys.platform.startswith("linux") and llama_cpp_is_cuda_build():
        for library in linux_nvidia_cuda_libraries():
            try:
                _handles.append(ctypes.CDLL(str(library), mode=getattr(ctypes, "RTLD_GLOBAL", 0)))
                added.append(str(library))
            except OSError:
                # A library that does not load (wrong driver, damaged file) is left to
                # llama.cpp's own import, which then reports the real error.
                continue
    _prepared = added
    return list(added)
