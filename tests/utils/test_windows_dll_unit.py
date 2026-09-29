"""Windows: llama.cpp's CUDA wheel finds torch's cuBLAS/cudart (backlog 0988).

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


def test_off_windows_is_a_no_op(tmp_path, monkeypatch) -> None:
    _fake_torch(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    before = os.environ.get("PATH")

    assert windows_dll.prepare_llama_cpp_import() == []
    assert os.environ.get("PATH") == before


def test_gguf_load_prepares_the_dll_search_before_llama_cpp(monkeypatch) -> None:
    # The provider calls the helper on the GGUF branch, before `_setup_device_gguf` imports
    # llama_cpp for the first time.
    import inspect

    from abstractcore.providers import huggingface_provider as hf

    source = inspect.getsource(hf.HuggingFaceProvider)
    branch = source.split('self.model_type = "gguf"', 1)[1]
    assert branch.index("prepare_llama_cpp_import()") < branch.index("self._setup_device_gguf()")
