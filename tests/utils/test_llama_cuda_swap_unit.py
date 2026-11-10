"""`pip install "abstractcore[gpu]"` on a Linux NVIDIA host gets llama.cpp's CUDA build (utils/llama_cuda_swap).

Everything is simulated: a fake llama_cpp package on disk (processor or CUDA build by the presence of
lib/libggml-cuda.so), a fake torch/version.py, a fake nvidia-smi answer and a fake swap runner that
either installs the CUDA file or fails. Neither torch nor llama_cpp is imported; pip never runs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from abstractcore.utils import llama_cuda_swap as swap
from abstractcore.utils import windows_dll

# The real swap_command, captured before the autouse fixture replaces it with a fake.
_REAL_SWAP_COMMAND = swap.swap_command


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(swap, "_result", None)
    monkeypatch.setattr(windows_dll, "_prepared", None)
    monkeypatch.setattr(windows_dll, "_handles", [])
    monkeypatch.delenv(swap.ENV_OPT_OUT, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(swap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(swap, "installed_version", lambda: "0.3.36")
    monkeypatch.setattr(swap, "nvidia_driver_major", lambda: 595)
    monkeypatch.setattr(swap, "swap_command", lambda build, version: ["fake-pip", build, version])


def _fake_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cuda_build: bool, torch_cuda: str | None):
    site = tmp_path / "site-packages"
    llama = site / "llama_cpp"
    (llama / "lib").mkdir(parents=True)
    (llama / "__init__.py").write_text("raise RuntimeError('llama_cpp must not be imported')\n")
    if cuda_build:
        (llama / "lib" / "libggml-cuda.so").write_bytes(b"")
    specs = {"llama_cpp": importlib.util.spec_from_file_location("llama_cpp", llama / "__init__.py",
                                                                    submodule_search_locations=[str(llama)])}
    if torch_cuda is not None:
        torch = site / "torch"
        torch.mkdir()
        (torch / "__init__.py").write_text("raise RuntimeError('torch must not be imported')\n")
        (torch / "version.py").write_text(f"__version__ = '2.13.0+cu130'\ndebug = False\ncuda = {torch_cuda!r}\n")
        specs["torch"] = importlib.util.spec_from_file_location("torch", torch / "__init__.py",
                                                                submodule_search_locations=[str(torch)])
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: specs[name] if name in specs else (None if name == "torch" else real(name, *a, **k)))
    return llama


class _Runner:
    """The fake swap: records the command and installs (or not) the CUDA file."""

    def __init__(self, llama: Path, *, succeed: bool = True, returncode: int = 0):
        self.llama, self.succeed, self.returncode, self.calls = llama, succeed, returncode, []

    def __call__(self, command):
        self.calls.append(list(command))
        if self.succeed:
            (self.llama / "lib" / "libggml-cuda.so").write_bytes(b"")

        class P:
            returncode = self.returncode
            stdout = "Successfully installed llama-cpp-python-0.3.36\n" if self.succeed else ""
            stderr = "" if self.succeed else "ERROR: No matching distribution found for llama-cpp-python==0.3.36\n"
        return P()


def _marker(llama: Path) -> dict:
    return json.loads((llama / swap.MARKER_NAME).read_text())


def test_processor_build_on_a_cuda_13_host_is_swapped_once_with_the_cu130_build(tmp_path, monkeypatch):
    llama = _fake_packages(tmp_path, monkeypatch, cuda_build=False, torch_cuda="13.0")
    runner = _Runner(llama)
    monkeypatch.setattr(swap, "run_swap", runner)
    out = swap.ensure_llama_cpp_cuda_build()
    assert out["outcome"] == "swapped" and out["build"] == "cu130" and out["version"] == "0.3.36"
    assert runner.calls == [["fake-pip", "cu130", "0.3.36"]]
    assert _marker(llama)["outcome"] == "swapped"
    # Second process: the CUDA build is in place, nothing runs.
    monkeypatch.setattr(swap, "_result", None)
    again = swap.ensure_llama_cpp_cuda_build()
    assert again["outcome"] == "cuda-build-present" and runner.calls == [["fake-pip", "cu130", "0.3.36"]]


def test_cuda_12_torch_gets_the_cu125_build(tmp_path, monkeypatch):
    llama = _fake_packages(tmp_path, monkeypatch, cuda_build=False, torch_cuda="12.8")
    runner = _Runner(llama)
    monkeypatch.setattr(swap, "run_swap", runner)
    assert swap.ensure_llama_cpp_cuda_build()["build"] == "cu125"
    assert runner.calls == [["fake-pip", "cu125", "0.3.36"]]


def test_a_host_the_installer_set_up_never_runs_the_swap(tmp_path, monkeypatch):
    # The CUDA build is detected as such (lib/libggml-cuda.so): no pip call, no marker.
    llama = _fake_packages(tmp_path, monkeypatch, cuda_build=True, torch_cuda="13.0")
    runner = _Runner(llama)
    monkeypatch.setattr(swap, "run_swap", runner)
    out = swap.ensure_llama_cpp_cuda_build()
    assert out["outcome"] == "cuda-build-present"
    assert runner.calls == [] and not (llama / swap.MARKER_NAME).exists()


@pytest.mark.parametrize("why, setup", [
    ("opt-out", dict(env="0", torch="13.0", driver=595, machine="x86_64", platform="linux")),
    ("no CUDA torch", dict(env=None, torch=None, driver=595, machine="x86_64", platform="linux")),
    ("processor torch", dict(env=None, torch="", driver=595, machine="x86_64", platform="linux")),
    ("driver too old", dict(env=None, torch="13.0", driver=550, machine="x86_64", platform="linux")),
    ("no driver", dict(env=None, torch="13.0", driver=None, machine="x86_64", platform="linux")),
    ("aarch64", dict(env=None, torch="13.0", driver=595, machine="aarch64", platform="linux")),
    ("macOS", dict(env=None, torch="13.0", driver=595, machine="x86_64", platform="darwin")),
    ("Windows", dict(env=None, torch="13.0", driver=595, machine="x86_64", platform="win32")),
])
def test_no_swap_when_it_does_not_apply_and_the_reason_is_recorded(tmp_path, monkeypatch, why, setup):
    torch = setup["torch"]
    llama = _fake_packages(tmp_path, monkeypatch, cuda_build=False, torch_cuda=torch)
    if torch == "":  # a processor build of torch: version.py has `cuda = None`
        (tmp_path / "site-packages" / "torch" / "version.py").write_text("__version__ = '2.13.0+cpu'\ncuda = None\n")
    if setup["env"] is not None:
        monkeypatch.setenv(swap.ENV_OPT_OUT, setup["env"])
    monkeypatch.setattr(swap, "nvidia_driver_major", lambda: setup["driver"])
    monkeypatch.setattr(swap.platform, "machine", lambda: setup["machine"])
    monkeypatch.setattr(sys, "platform", setup["platform"])
    runner = _Runner(llama)
    monkeypatch.setattr(swap, "run_swap", runner)
    out = swap.ensure_llama_cpp_cuda_build()
    assert out["outcome"] == "skipped", (why, out)
    assert out["reason"], why
    assert runner.calls == [], why
    if why == "driver too old":
        assert "580" in out["reason"] and "550" in out["reason"]


def test_a_failed_swap_is_recorded_once_and_not_retried(tmp_path, monkeypatch):
    llama = _fake_packages(tmp_path, monkeypatch, cuda_build=False, torch_cuda="13.0")
    runner = _Runner(llama, succeed=False, returncode=1)
    monkeypatch.setattr(swap, "run_swap", runner)
    out = swap.ensure_llama_cpp_cuda_build()
    assert out["outcome"] == "failed" and "exit 1" in out["reason"] and "No matching distribution" in out["reason"]
    assert _marker(llama)["outcome"] == "failed"
    monkeypatch.setattr(swap, "_result", None)
    again = swap.ensure_llama_cpp_cuda_build()
    assert again["outcome"] == "failed" and len(runner.calls) == 1


def test_a_swap_that_succeeds_without_delivering_a_cuda_build_is_a_failure(tmp_path, monkeypatch):
    llama = _fake_packages(tmp_path, monkeypatch, cuda_build=False, torch_cuda="13.0")
    runner = _Runner(llama, succeed=False, returncode=0)
    monkeypatch.setattr(swap, "run_swap", runner)
    out = swap.ensure_llama_cpp_cuda_build()
    assert out["outcome"] == "failed" and "not a CUDA build" in out["reason"]


def test_when_llama_cpp_is_already_imported_the_outcome_says_restart(tmp_path, monkeypatch):
    llama = _fake_packages(tmp_path, monkeypatch, cuda_build=False, torch_cuda="13.0")
    monkeypatch.setattr(swap, "run_swap", _Runner(llama))
    monkeypatch.setitem(sys.modules, "llama_cpp", type(sys)("llama_cpp"))
    assert swap.ensure_llama_cpp_cuda_build()["outcome"] == "swapped-restart-needed"


def test_swap_command_uses_pip_when_present_else_uv(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object() if name == "pip" else real(name, *a, **k))
    cmd = _REAL_SWAP_COMMAND("cu130", "0.3.36")
    assert cmd[:3] == [sys.executable, "-m", "pip"] and "--no-index" in cmd and "--force-reinstall" in cmd
    assert cmd[-1] == "llama-cpp-python==0.3.36" and f"{swap.INDEX}/cu130/llama-cpp-python/" in cmd
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "pip" else real(name, *a, **k))
    monkeypatch.setattr(swap.shutil, "which", lambda name: "/usr/local/bin/uv" if name == "uv" else None)
    cmd = _REAL_SWAP_COMMAND("cu125", "0.3.36")
    assert cmd[:3] == ["/usr/local/bin/uv", "pip", "install"] and "--reinstall-package" in cmd and "--python" in cmd
    monkeypatch.setattr(swap.shutil, "which", lambda name: None)
    assert _REAL_SWAP_COMMAND("cu125", "0.3.36") is None


def test_torch_cuda_major_reads_version_py_without_importing_torch(tmp_path, monkeypatch):
    _fake_packages(tmp_path, monkeypatch, cuda_build=False, torch_cuda="13.0")
    assert swap.torch_cuda_major() == 13
    (tmp_path / "site-packages" / "torch" / "version.py").write_text("cuda = '12.8'\n")
    assert swap.torch_cuda_major() == 12
    (tmp_path / "site-packages" / "torch" / "version.py").write_text("cuda = None\n")
    assert swap.torch_cuda_major() is None


def test_prepare_llama_cpp_import_runs_the_swap_before_preloading_on_linux(tmp_path, monkeypatch):
    # The hook: windows_dll.prepare_llama_cpp_import() is what every `import llama_cpp` in AbstractCore
    # goes through; on Linux it swaps first, then preloads the CUDA libraries for a CUDA build.
    llama = _fake_packages(tmp_path, monkeypatch, cuda_build=False, torch_cuda="13.0")
    runner = _Runner(llama)
    monkeypatch.setattr(swap, "run_swap", runner)
    monkeypatch.setattr(windows_dll, "linux_nvidia_cuda_libraries", lambda: [])
    monkeypatch.setattr(windows_dll, "llama_cpp_is_cuda_build", swap.llama_cpp_is_cuda_build)
    windows_dll.prepare_llama_cpp_import()
    assert runner.calls == [["fake-pip", "cu130", "0.3.36"]]
    assert swap.ensure_llama_cpp_cuda_build()["outcome"] == "swapped"
