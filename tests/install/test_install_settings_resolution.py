"""Fresh-environment proofs for the three install settings (network; opt-in).

Run with ``ABSTRACTCORE_INSTALL_TESTS=1 pytest tests/install -q``. Needs ``uv`` on PATH and
network access to PyPI; it builds this checkout's wheel once and:

- installs the light setting with plain ``pip install <wheel>`` into a fresh venv and runs
  ``light_install_smoke.py`` there: every remote provider is constructed and answers a local
  fake server (no keys), and tools / media / server / plugins import;
- resolves light for macOS arm64, manylinux x86_64 and Windows (Python 3.9 and 3.12);
- resolves ``apple`` for macOS 14 arm64 and ``gpu`` for manylinux_2_35 x86_64 with
  ``uv pip compile --python-platform`` (no install);
- resolves ``gpu`` for Windows x86_64 with wheels only (pure-Python sdists allowed) on each
  PyTorch build install.ps1 picks (cu130, cu126, cpu), without vLLM, llama-cpp-python or
  stable-diffusion-cpp-python (backlog 0988);
- resolves the deprecated aliases that released packages pin (runtime 0.7.x light,
  ``all-apple``, ``all-gpu``) to exactly the packages their setting resolves to.

Set ABSTRACTCORE_INSTALL_FIND_LINKS to extra wheel directories (os.pathsep-separated) to prove
the install against sibling wheels not yet on PyPI. Nothing here downloads a model. The venv lives under pytest's tmp_path and is deleted with it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SMOKE = Path(__file__).with_name("light_install_smoke.py")

pytestmark = pytest.mark.skipif(
    os.environ.get("ABSTRACTCORE_INSTALL_TESTS") != "1",
    reason="network install proofs; set ABSTRACTCORE_INSTALL_TESTS=1",
)

MAC = "aarch64-apple-darwin"
LINUX = "x86_64-manylinux_2_28"
LINUX_GPU = "x86_64-manylinux_2_35"  # mlx-gen (via abstractvision[all-gpu]) needs glibc 2.35
WINDOWS = "x86_64-pc-windows-msvc"


def _find_links() -> list[str]:
    extra = os.environ.get("ABSTRACTCORE_INSTALL_FIND_LINKS", "")
    return [d for d in extra.split(os.pathsep) if d]


def _uv() -> str:
    uv = shutil.which("uv")
    assert uv, "uv is required for the install proofs (https://docs.astral.sh/uv/)"
    return uv


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN"))}
    env["MACOSX_DEPLOYMENT_TARGET"] = "14.0"
    return env


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    subprocess.run([_uv(), "build", "--wheel", "-o", str(out), str(ROOT)], check=True, capture_output=True,
                   env=_env())
    [built] = sorted(out.glob("abstractcore-*.whl"))
    return built


def _version(wheel: Path) -> str:
    return wheel.name.split("-")[1]


def _compile(wheel: Path, requirements: list[str], *, platform: str, python: str,
             extra: tuple[str, ...] = ()) -> dict[str, str]:
    proc = subprocess.run(
        [_uv(), "pip", "compile", "--quiet", "--no-header", "--find-links", str(wheel.parent),
         *[arg for d in _find_links() for arg in ("--find-links", d)],
         "--python-platform", platform, "--python-version", python, *extra, "-"],
        input="\n".join(requirements), text=True, capture_output=True, env=_env(),
    )
    assert proc.returncode == 0, f"{requirements} does not resolve for {platform} py{python}:\n{proc.stderr[-3000:]}"
    pins = {}
    for line in proc.stdout.splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, _, version = line.strip().partition("==")
            pins[name.lower()] = version.split()[0]
    return pins


def test_light_installs_with_plain_pip_and_runs_every_remote_provider(wheel, tmp_path) -> None:
    env_dir = tmp_path / "light"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    links = [arg for d in _find_links() for arg in ("--find-links", d)]
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", *links, str(wheel)], check=True, env=_env())
    home = tmp_path / "home"
    home.mkdir()
    env = _env()
    env.update({"HOME": str(home), "HF_HUB_OFFLINE": "1"})
    proc = subprocess.run([str(python), str(SMOKE)], capture_output=True, text=True, env=env, cwd=tmp_path,
                          timeout=600)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("LIGHT_OK"), proc.stdout[-4000:] + proc.stderr[-4000:]


@pytest.mark.parametrize("platform", [MAC, LINUX, WINDOWS])
@pytest.mark.parametrize("python", ["3.9", "3.12"])
def test_light_resolves_everywhere(wheel, platform, python) -> None:
    pins = _compile(wheel, [f"abstractcore=={_version(wheel)}"], platform=platform, python=python)
    assert {"openai", "anthropic", "requests", "pypdf", "fastapi", "abstractvoice"} <= set(pins)
    assert not {"torch", "transformers", "mlx", "mlx-lm", "vllm"} & set(pins)


def test_apple_resolves_on_macos_arm64(wheel) -> None:
    pins = _compile(wheel, [f"abstractcore[apple]=={_version(wheel)}"], platform=MAC, python="3.12")
    assert {"mlx", "mlx-lm", "mlx-vlm", "torch", "transformers", "sentence-transformers", "playwright",
            "openai", "anthropic"} <= set(pins)
    assert "vllm" not in pins


def test_gpu_resolves_on_manylinux(wheel) -> None:
    pins = _compile(wheel, [f"abstractcore[gpu]=={_version(wheel)}"], platform=LINUX_GPU, python="3.12")
    assert {"vllm", "torch", "transformers", "sentence-transformers", "playwright", "openai", "anthropic"} <= set(pins)
    assert "mlx-lm" not in pins


# Pure-Python packages that publish only an sdist (they build anywhere without a compiler).
PURE_PYTHON_SDISTS = ("antlr4-python3-runtime", "encodec", "langdetect", "transformers-stream-generator")


@pytest.mark.parametrize("python, torch_backend", [("3.12", "cu130"), ("3.12", "cu126"), ("3.13", "cu130"),
                                                   ("3.11", "cpu")])
def test_gpu_resolves_on_windows_with_wheels_only(wheel, python, torch_backend) -> None:
    # Backlog 0988: vLLM (Linux only) and llama-cpp-python / stable-diffusion-cpp-python (source
    # builds on PyPI) are marked out on Windows; install.ps1 adds llama.cpp's prebuilt wheel.
    only_wheels = ("--only-binary", ":all:", *[a for p in PURE_PYTHON_SDISTS for a in ("--no-binary", p)],
                   "--torch-backend", torch_backend)
    pins = _compile(wheel, [f"abstractcore[gpu]=={_version(wheel)}"], platform=WINDOWS, python=python,
                    extra=only_wheels)
    assert {"torch", "transformers", "sentence-transformers", "diffusers", "faster-whisper", "playwright"} <= set(pins)
    assert not {"vllm", "llama-cpp-python", "stable-diffusion-cpp-python", "mlx", "mlx-gen"} & set(pins)
    assert pins["torch"].endswith(f"+{torch_backend}"), pins["torch"]


@pytest.mark.parametrize(
    "alias, setting, platform",
    [
        ("remote,tools,vision,voice,audio,music", None, LINUX),  # AbstractRuntime 0.7.x light pin
        ("all-apple", "apple", MAC),
        ("all-gpu", "gpu", LINUX_GPU),
    ],
)
def test_released_alias_pins_resolve_to_their_setting(wheel, alias, setting, platform) -> None:
    version = _version(wheel)
    target = f"abstractcore[{setting}]=={version}" if setting else f"abstractcore=={version}"
    via_alias = _compile(wheel, [f"abstractcore[{alias}]=={version}"], platform=platform, python="3.12")
    via_setting = _compile(wheel, [target], platform=platform, python="3.12")
    assert via_alias == via_setting
