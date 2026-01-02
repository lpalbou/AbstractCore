from __future__ import annotations

from pathlib import Path

from abstractcore.tools.common_tools import list_files


def test_list_files_includes_directories(tmp_path: Path) -> None:
    """`list_files` should list directories (not only files).

    This is important for agent workflows that create empty directories (e.g. `mkdir -p`)
    and then need to confirm the directory exists before writing files into it.
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "readme.md").write_text("hello", encoding="utf-8")

    out = list_files(str(tmp_path), pattern="*", recursive=False, include_hidden=False, head_limit=None)

    assert "docs/" in out
    assert "readme.md" in out


def test_list_files_excludes_hidden_entries_by_default(tmp_path: Path) -> None:
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / ".hidden_file.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")

    out = list_files(str(tmp_path), pattern="*", recursive=False, include_hidden=False, head_limit=None)

    assert ".hidden_dir" not in out
    assert ".hidden_file.txt" not in out
    assert "visible.txt" in out


def test_list_files_can_include_hidden_entries(tmp_path: Path) -> None:
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / ".hidden_file.txt").write_text("secret", encoding="utf-8")

    out = list_files(str(tmp_path), pattern="*", recursive=False, include_hidden=True, head_limit=None)

    assert ".hidden_dir" in out
    assert ".hidden_file.txt" in out


