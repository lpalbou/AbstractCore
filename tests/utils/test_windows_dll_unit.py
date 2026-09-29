"""llama.cpp's CUDA wheel finds torch's cuBLAS/cudart: Windows (backlog 0988) and Linux (0989).

The platform is simulated (`sys.platform = "win32"`, a fake `os.add_dll_directory`, a fake torch
package on disk found through `importlib.util.find_spec`), so these run on any OS. torch is never
imported.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from abstractcore.utils import windows_dll


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(windows_dll, "_prepared", None)
    monkeypatch.setattr(windows_dll, "_handles", [])


def _fake_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_lib: bool = True) -> Path:
    pkg = tmp_path / "site-packages" / "torch"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('torch must not be imported')\n")
    if with_lib:
        (pkg / "lib").mkdir()
    spec = importlib.util.spec_from_file_location("torch", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: spec if name == "torch" else real(name, *a, **k))
    return pkg / "lib"


def _simulate_windows(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    added: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", lambda p: added.append(p) or object(), raising=False)
    monkeypatch.setenv("PATH", "C:\\Windows\\system32")
    return added


def test_windows_with_torch_lib_adds_it_to_the_dll_search_and_path(tmp_path, monkeypatch) -> None:
    lib = _fake_torch(tmp_path, monkeypatch)
    added = _simulate_windows(monkeypatch)

    assert windows_dll.prepare_llama_cpp_import() == [str(lib)]
    assert added == [str(lib)]
    assert os.environ["PATH"].split(os.pathsep)[0] == str(lib)
    assert "torch" not in sys.modules or not str(getattr(sys.modules["torch"], "__file__", "")).startswith(str(tmp_path))
    # Idempotent: a second call adds nothing more.
    assert windows_dll.prepare_llama_cpp_import() == [str(lib)]
    assert added == [str(lib)]
    assert os.environ["PATH"].split(os.pathsep).count(str(lib)) == 1


def test_windows_without_torch_adds_nothing(monkeypatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "torch" else real(name, *a, **k))
    added = _simulate_windows(monkeypatch)

    assert windows_dll.prepare_llama_cpp_import() == []
    assert added == [] and os.environ["PATH"] == "C:\\Windows\\system32"


def test_windows_torch_without_lib_folder_adds_nothing(tmp_path, monkeypatch) -> None:
    _fake_torch(tmp_path, monkeypatch, with_lib=False)
    added = _simulate_windows(monkeypatch)

    assert windows_dll.prepare_llama_cpp_import() == []
    assert added == []


def test_macos_is_a_no_op(tmp_path, monkeypatch) -> None:
    _fake_torch(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    before = os.environ.get("PATH")

    assert windows_dll.prepare_llama_cpp_import() == []
    assert os.environ.get("PATH") == before


def _fake_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, llama_cuda: bool, nvidia: dict) -> Path:
    """A fake site-packages with llama_cpp (CUDA build or not) and `nvidia/<name>/lib/<files>`."""

    site = tmp_path / "site-packages"
    llama = site / "llama_cpp"
    (llama / "lib").mkdir(parents=True)
    (llama / "__init__.py").write_text("raise RuntimeError('llama_cpp must not be imported')\n")
    (llama / "lib" / "libllama.so").write_bytes(b"")
    if llama_cuda:
        (llama / "lib" / "libggml-cuda.so").write_bytes(b"")
    root = site / "nvidia"
    for name, files in nvidia.items():
        (root / name / "lib").mkdir(parents=True)
        for f in files:
            (root / name / "lib" / f).write_bytes(b"")
    llama_spec = importlib.util.spec_from_file_location(
        "llama_cpp", llama / "__init__.py", submodule_search_locations=[str(llama)]
    )
    nvidia_spec = importlib.util.spec_from_loader("nvidia", loader=None, is_package=True)
    nvidia_spec.submodule_search_locations = [str(root)]
    real = importlib.util.find_spec
    specs = {"llama_cpp": llama_spec, "nvidia": nvidia_spec if nvidia else None}
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: specs[name] if name in specs else real(name, *a, **k)
    )
    return root


def _simulate_linux(monkeypatch: pytest.MonkeyPatch) -> list:
    import ctypes

    loads: list = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda name, mode=0, *a, **k: loads.append((name, mode)) or object())
    return loads


CU12 = {"cuda_runtime": ["libcudart.so.12"], "cublas": ["libcublas.so.12", "libcublasLt.so.12", "libnvblas.so.12"]}
CU13 = {"cu13": ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13", "libcudart_static.a", "libnvrtc.so.13"]}


def test_linux_cuda_llama_preloads_the_nvidia_wheels_cuda_libraries_in_dependency_order(tmp_path, monkeypatch) -> None:
    # Measured on a Linux + NVIDIA gpu install (backlog 0989): `import llama_cpp` of the cu125
    # wheel failed with "libcudart.so.12: cannot open shared object file" unless torch had been
    # imported first; preloading these files made it load and offload every layer to the GPU.
    import ctypes

    root = _fake_site(tmp_path, monkeypatch, llama_cuda=True, nvidia={**CU12, **CU13})
    loads = _simulate_linux(monkeypatch)
    before = os.environ.get("PATH")

    expected = [
        str(root / "cu13" / "lib" / "libcudart.so.13"),
        str(root / "cuda_runtime" / "lib" / "libcudart.so.12"),
        str(root / "cu13" / "lib" / "libcublasLt.so.13"),
        str(root / "cublas" / "lib" / "libcublasLt.so.12"),
        str(root / "cu13" / "lib" / "libcublas.so.13"),
        str(root / "cublas" / "lib" / "libcublas.so.12"),
    ]
    assert windows_dll.prepare_llama_cpp_import() == expected
    assert loads == [(p, ctypes.RTLD_GLOBAL) for p in expected]
    assert os.environ.get("PATH") == before
    # Idempotent.
    assert windows_dll.prepare_llama_cpp_import() == expected
    assert len(loads) == len(expected)


def test_linux_cpu_llama_preloads_nothing(tmp_path, monkeypatch) -> None:
    _fake_site(tmp_path, monkeypatch, llama_cuda=False, nvidia=CU12)
    loads = _simulate_linux(monkeypatch)

    assert windows_dll.prepare_llama_cpp_import() == []
    assert loads == []


def test_linux_cuda_llama_without_nvidia_wheels_preloads_nothing(tmp_path, monkeypatch) -> None:
    _fake_site(tmp_path, monkeypatch, llama_cuda=True, nvidia={})
    loads = _simulate_linux(monkeypatch)

    assert windows_dll.prepare_llama_cpp_import() == []
    assert loads == []


def test_linux_library_that_does_not_load_is_skipped(tmp_path, monkeypatch) -> None:
    import ctypes

    root = _fake_site(tmp_path, monkeypatch, llama_cuda=True, nvidia=CU12)
    monkeypatch.setattr(sys, "platform", "linux")
    bad = str(root / "cuda_runtime" / "lib" / "libcudart.so.12")

    def cdll(name, mode=0, *a, **k):
        if name == bad:
            raise OSError("broken")
        return object()

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    assert windows_dll.prepare_llama_cpp_import() == [
        str(root / "cublas" / "lib" / "libcublasLt.so.12"),
        str(root / "cublas" / "lib" / "libcublas.so.12"),
    ]


def test_gguf_load_prepares_the_dll_search_before_llama_cpp(monkeypatch) -> None:
    # The provider calls the helper on the GGUF branch, before `_setup_device_gguf` imports
    # llama_cpp for the first time.
    import inspect

    from abstractcore.providers import huggingface_provider as hf

    source = inspect.getsource(hf.HuggingFaceProvider)
    branch = source.split('self.model_type = "gguf"', 1)[1]
    assert branch.index("prepare_llama_cpp_import()") < branch.index("self._setup_device_gguf()")
