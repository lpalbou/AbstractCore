from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.tools.common_tools import skim_folders


pytestmark = pytest.mark.basic


def test_skim_folders_returns_tree_and_notable_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "README.md").write_text("hello\n", encoding="utf-8")

    docs = root / "docs"
    (docs / "adr").mkdir(parents=True)
    (docs / "architecture.md").write_text("x\n", encoding="utf-8")
    (docs / "adr" / "0019-testing-strategy-and-levels.md").write_text("x\n", encoding="utf-8")

    data = root / "data"
    data.mkdir()
    (data / "records.json").write_text("{\"ok\": true}\n", encoding="utf-8")

    out = skim_folders(paths=[str(root)], max_depth=3)

    assert f"Folder: {root}" in out
    assert "./ (" in out
    assert "docs/ (" in out
    assert "docs/adr/ (" in out
    assert "Notable files:" in out
    assert "- README.md" in out
    assert "- docs/architecture.md" in out
    assert "- docs/adr/0019-testing-strategy-and-levels.md" in out


def test_skim_folders_respects_abstractignore(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / ".abstractignore").write_text("secret/\n", encoding="utf-8")

    secret = root / "secret"
    secret.mkdir()
    (secret / "README.md").write_text("top secret\n", encoding="utf-8")

    out = skim_folders(paths=[str(root)], max_depth=2)
    assert "secret/" not in out

