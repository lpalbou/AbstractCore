"""The standalone one-line installers (scripts/install.sh, scripts/install.ps1).

They are served at https://abstractframework.ai/core/install.sh and .../install.ps1, so these
checks keep them in step with the package: the default pin is this release, they install only the
three install settings (light = no extra, `apple`, `gpu`), the commands they guard are exactly
`[project.scripts]`, the llama.cpp handling targets a real requirement, and the documented
one-liners are the exact URLs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "scripts" / "install.sh"
PS1 = ROOT / "scripts" / "install.ps1"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

SH_LINE = "curl -LsSf https://abstractframework.ai/core/install.sh | sh"
PS1_LINE = 'powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/core/install.ps1 | iex"'
PIP_LINES = ("pip install abstractcore", 'pip install "abstractcore[apple]"', 'pip install "abstractcore[gpu]"')


def _sh() -> str:
    return SH.read_text(encoding="utf-8")


def _ps1() -> str:
    return PS1.read_text(encoding="utf-8")


def _version() -> str:
    text = (ROOT / "abstractcore" / "utils" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "abstractcore/utils/version.py has no __version__"
    return m.group(1)


def _one(pattern: str, text: str, what: str) -> str:
    m = re.search(pattern, text, re.MULTILINE)
    assert m, f"{what} not found"
    return m.group(1)


def _name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0].lower()


def test_both_scripts_exist():
    assert SH.is_file()
    assert PS1.is_file()


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX sh")
def test_install_sh_parses_and_prints_help():
    subprocess.run(["sh", "-n", str(SH)], check=True)
    out = subprocess.run(["sh", str(SH), "--help"], check=True, capture_output=True, text=True).stdout
    for opt in ("--profile auto|light|apple|gpu", "--pin VERSION|latest", "--from PATH", "--print"):
        assert opt in out


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="no PowerShell")
def test_install_ps1_parses():
    cmd = (
        "$e = $null; [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{PS1}', [ref]$null, [ref]$e) | Out-Null; $e.Count"
    )
    out = subprocess.run(["pwsh", "-NoProfile", "-Command", cmd], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "0"


def test_default_pin_is_this_release():
    assert _one(r'^AC_PIN_DEFAULT="([^"]+)"', _sh(), "AC_PIN_DEFAULT in install.sh") == _version()
    assert _one(r"^\$AcPinDefault = '([^']+)'", _ps1(), "$AcPinDefault in install.ps1") == _version()


def test_profiles_install_only_the_three_settings():
    sh, ps1 = _sh(), _ps1()
    extras = PYPROJECT["project"]["optional-dependencies"]
    # The profile -> extra mapping each script installs; light is the base package, no extra.
    assert dict(re.findall(r"(apple|gpu)\) EXTRAS=(\w+)", sh)) == {"apple": "apple", "gpu": "gpu"}
    assert '*) EXTRAS="" ;;' in sh
    assert "$extras = if ($profileName -eq 'gpu') { 'gpu' } else { '' }" in ps1
    # Every abstractcore[...] the scripts or their headers name is `apple` or `gpu`, and exists.
    for text, name, expected in ((sh, "install.sh", {"apple", "gpu"}), (ps1, "install.ps1", {"gpu"})):
        named = set()
        for group in re.findall(r"abstractcore\[([a-z0-9,-]+)\]", text):
            named.update(group.split(","))
        assert named == expected, f"{name} names abstractcore[...] extras {sorted(named)}"
        assert named <= set(extras)
    # Both scripts offer the same profiles (apple is refused on Windows, with a sentence).
    for p in ("auto", "light", "apple", "gpu"):
        assert f"'{p}'" in ps1, f"install.ps1 does not handle profile {p}"
        assert re.search(rf"^\s*{p}[|)]", sh, re.MULTILINE), f"install.sh does not handle profile {p}"


def test_commands_are_exactly_the_project_scripts():
    scripts = set(PYPROJECT["project"]["scripts"])
    sh_names = _one(r'^AC_COMMANDS="([^"]+)"', _sh(), "AC_COMMANDS in install.sh").split()
    ps1_names = re.findall(r"'([^']+)'", _one(r"^\$AcCommands = @\((.+)\)$", _ps1(), "$AcCommands in install.ps1"))
    assert len(sh_names) == len(set(sh_names))
    assert set(sh_names) == scripts
    assert set(ps1_names) == scripts


def test_prebuilt_wheel_handling_targets_real_requirements():
    sh = _sh()
    extras = PYPROJECT["project"]["optional-dependencies"]
    # llama.cpp comes from upstream's prebuilt wheel: it must be a requirement of both settings
    # (else the --with pin adds a package the setting does not carry), and never be built.
    for setting in ("apple", "gpu"):
        assert "llama-cpp-python" in {_name(r) for r in extras[setting]}, setting
    pin = _one(r'^AC_LLAMA_CPU_PIN="([^"]+)"', sh, "AC_LLAMA_CPU_PIN")
    metal = _one(r'^AC_LLAMA_METAL_PIN="([^"]+)"', sh, "AC_LLAMA_METAL_PIN")
    for value in (pin, metal):
        assert re.fullmatch(r"\d+\.\d+\.\d+", value)
    assert "--no-build-package" in sh and "--overrides uv-overrides.txt" in sh
    assert "for _p in $AC_NO_WHEEL llama-cpp-python; do" in sh
    assert _one(r'^AC_NO_WHEEL="([^"]+)"', sh, "AC_NO_WHEEL").split() == ["aec-audio-processing", "stable-diffusion-cpp-python"]


def test_scripts_and_docs_name_the_exact_one_liners():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    for doc, name in ((readme, "README.md"), (install, "docs/installation.md")):
        for line in (SH_LINE, PS1_LINE) + PIP_LINES:
            assert line in doc, f"{name} lacks: {line}"
    assert "## One-line install" in install
    assert SH_LINE in _sh()
    assert PS1_LINE in _ps1()
    assert "docs/installation.md#one-line-install" in _sh()
    assert "docs/installation.md#one-line-install" in _ps1()


def test_nvidia_gets_llama_cpp_cuda_build_checked_the_way_abstractcore_loads_it():
    sh, ps1 = _sh(), _ps1()
    install = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    # The CUDA build follows PyTorch's CUDA major (install.sh) / the driver's stack (install.ps1).
    assert _one(r'^AC_LLAMA_CUDA13="([^"]+)"', sh, "AC_LLAMA_CUDA13") == "cu130"
    assert _one(r'^AC_LLAMA_CUDA12="([^"]+)"', sh, "AC_LLAMA_CUDA12") == "cu125"
    assert "$llamaVariants = @('cu130', 'cu132', 'cpu')" in ps1
    assert "$llamaVariants = @('cu125', 'cpu')" in ps1
    # Same llama.cpp version on both scripts (a swap is a same-version reinstall, which is what
    # keeps the CUDA build through `uv tool upgrade` and re-runs).
    assert _one(r"^\$AcLlamaPin = '([^']+)'", ps1, "$AcLlamaPin") == _one(r'^AC_LLAMA_CPU_PIN="([^"]+)"', sh, "AC_LLAMA_CPU_PIN")
    # The swap: that build's page only, llama-cpp-python alone, reinstalled.
    assert '--find-links "$AC_LLAMA_INDEX/$1/llama-cpp-python/" --no-deps --reinstall-package llama-cpp-python' in sh
    assert "'--no-index', '--find-links', \"$AcLlamaIndex/$V/llama-cpp-python/\", '--no-deps', '--reinstall-package', 'llama-cpp-python'" in ps1
    # Kept only when it loads the way AbstractCore loads it (the Linux CUDA build needs AbstractCore's
    # preload of PyTorch's CUDA libraries) and reports GPU offload.
    from abstractcore.utils.windows_dll import prepare_llama_cpp_import  # the seam both scripts call

    assert callable(prepare_llama_cpp_import)
    for text in (sh, ps1):
        assert "from abstractcore.utils.windows_dll import prepare_llama_cpp_import" in text
        assert "llama_supports_gpu_offload()" in text
    assert "llama.cpp's CUDA build" in install and "llama_supports_gpu_offload()" in install
