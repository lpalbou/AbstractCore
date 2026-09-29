"""Packaging invariants that sit under the three install settings (tests/test_install_settings.py)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
BASE: list[str] = PROJECT["dependencies"]
EXTRAS: dict[str, list[str]] = PROJECT["optional-dependencies"]


def _name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0].lower()


def test_every_abstractvoice_pin_carries_the_engine_runtime_floor() -> None:
    # The route engine check imports `abstractvoice.engine_runtime`: the constant and the pins move together.
    from abstractcore.config.route_engines import ABSTRACTVOICE_ENGINE_RUNTIME_FLOOR as voice_floor

    assert f"abstractvoice>={voice_floor}" in BASE
    assert f"abstractvoice[all-apple]>={voice_floor}" in EXTRAS["apple"]
    assert f"abstractvoice[all-gpu]>={voice_floor}" in EXTRAS["gpu"]


def test_mlx_vlm_ships_with_mlx_lm_in_apple() -> None:
    # A gateway with mlx-lm but no mlx-vlm accepts images and silently drops them.
    apple = EXTRAS["apple"]
    for dep in ("mlx>=0.32.2,<1.0.0", "mlx-lm>=0.31.3,<1.0.0", "mlx-vlm>=0.7.1,<0.8.0", "outlines>=0.1.0"):
        assert dep in apple
    assert "vllm>=0.6.0,<1.0.0" in EXTRAS["gpu"]


def test_apple_keeps_the_pins_that_avoid_backtracking_with_the_plugin_apple_stacks() -> None:
    apple = EXTRAS["apple"]
    for dep in ("transformers>=5.3.0,<6.0.0", "torch>=2.7.1,<3.0.0", "llama-cpp-python>=0.3.23,<1.0.0",
                "accelerate>=1.0.0", "numpy>=2.1.0,<3.0.0", "Pillow>=12.1.1,<13.0.0"):
        assert dep in apple


def test_pymupdf_family_is_never_in_light_apple_or_gpu() -> None:
    # PyMuPDF-family licensing is an explicit opt-in; pypdf is the permissive default in light.
    assert "pypdf>=6.0.0,<7.0.0" in BASE
    for requirements in (BASE, EXTRAS["apple"], EXTRAS["gpu"]):
        assert not any("pymupdf" in r.lower() for r in requirements)


def test_numpy_2_is_allowed_wherever_numpy_is_pinned() -> None:
    # abstractvision[all-gpu] (mlx-gen) and mlx-vlm require numpy>=2; a <2 cap makes a setting unsatisfiable.
    for key in ("apple", "gpu"):
        numpy_lines = [r for r in EXTRAS[key] if _name(r) == "numpy"]
        assert numpy_lines, f"{key}: expected an explicit numpy requirement"
        for req in numpy_lines:
            assert "<2" not in req.split(";")[0], f"{key}: numpy capped below 2: {req!r}"
            assert "<3.0.0" in req, f"{key}: numpy lost its <3.0.0 upper bound: {req!r}"


def test_music_and_3d_plugins_are_gated_to_the_python_versions_they_support() -> None:
    # abstractmusic and abstract3d require Python 3.10+; light still installs on 3.9.
    for package in ("abstractmusic", "abstract3d"):
        [req] = [r for r in BASE if _name(r) == package]
        assert "python_version >= '3.10'" in req


def test_server_docker_image_installs_the_exact_light_release_wheel() -> None:
    text = (ROOT / "docker" / "abstractcore-server" / "Dockerfile").read_text(encoding="utf-8")
    assert "https://pypi.org/pypi/abstractcore/" in text
    assert "ABSTRACTCORE_WHEEL_URL" in text
    assert '"abstractcore @ ${ABSTRACTCORE_WHEEL_URL}"' in text
