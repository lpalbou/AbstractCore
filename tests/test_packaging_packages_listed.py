"""Every importable package under abstractcore/ ships in the wheel.

`[tool.setuptools].packages` is an explicit list: a new subpackage that is not
added to it builds fine from a checkout and is missing from the published
distribution (ImportError on install).
"""
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_every_subpackage_is_listed_in_setuptools_packages() -> None:
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"])
    on_disk = {
        ".".join(init.parent.relative_to(ROOT).parts)
        for init in (ROOT / "abstractcore").rglob("__init__.py")
        if "__pycache__" not in init.parts
    }
    missing = sorted(on_disk - listed)
    assert not missing, f"add these to [tool.setuptools].packages in pyproject.toml, or they are not published: {missing}"
