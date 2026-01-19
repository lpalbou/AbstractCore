from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.tools.common_tools import edit_file


pytestmark = pytest.mark.basic


def test_edit_file_relative_path_error_includes_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "dir.md"
    p.mkdir()

    out = edit_file(file_path="dir.md", pattern="x", replacement="y")
    assert f"❌ Path is not a file: {p}" in out
    assert "Input: dir.md" in out

