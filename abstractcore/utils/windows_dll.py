"""Windows: let llama.cpp's CUDA build find the cuBLAS / cudart DLLs that PyTorch ships.

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
"""

from __future__ import annotations

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


def prepare_llama_cpp_import() -> List[str]:
    """Add PyTorch's CUDA DLL folder to the Windows DLL search before llama.cpp loads.

    Returns the folders added (empty off Windows or without torch). Idempotent: the work is
    done once per process."""

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
    _prepared = added
    return list(added)
