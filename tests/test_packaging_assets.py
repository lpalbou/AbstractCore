from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # Python 3.9-3.10


ROOT = Path(__file__).resolve().parents[1]


def _abstractcore_package_data() -> set[str]:
    with (ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)

    return set(pyproject["tool"]["setuptools"]["package-data"]["abstractcore"])


def test_package_data_includes_runtime_assets():
    package_data = _abstractcore_package_data()

    assert "assets/*.json" in package_data
    assert "assets/*.jsonld" in package_data
    assert "assets/*.md" in package_data
    assert "assets/*.ttf" in package_data

    assets_dir = ROOT / "abstractcore" / "assets"
    for filename in ("OCRA.ttf", "OCRB.ttf", "OCRBL.ttf", "semantic_models.md"):
        assert (assets_dir / filename).is_file()


def test_pep561_marker_exists_and_is_packaged():
    package_data = _abstractcore_package_data()

    assert "py.typed" in package_data
    assert (ROOT / "abstractcore" / "py.typed").is_file()
