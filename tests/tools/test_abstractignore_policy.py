from __future__ import annotations

from pathlib import Path

from abstractcore.tools.common_tools import analyze_code, edit_file, list_files, read_file, search_files, write_file


def test_search_files_ignores_runtime_store_dot_d_by_default(tmp_path: Path) -> None:
    # Project file that should be searched
    src = tmp_path / "src"
    src.mkdir()
    good = src / "main.py"
    good.write_text("import pygame\n# pygame usage\n", encoding="utf-8")

    # Simulated runtime store directory (AbstractCode/Runtime file store convention)
    store = tmp_path / "rt-80bc.d"
    store.mkdir()
    (store / "run_123.json").write_text('{"content": "pygame should not be matched"}\n', encoding="utf-8")

    out = search_files("pygame", path=str(tmp_path), file_pattern="*.py|*.json", output_mode="files_with_matches", head_limit=None)
    assert str(good) in out
    assert "rt-80bc.d" not in out
    assert "run_123.json" not in out


def test_list_files_recursive_ignores_runtime_store_dot_d_by_default(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    store = tmp_path / "state.d"
    store.mkdir()
    (store / "run_abc.json").write_text("x\n", encoding="utf-8")

    out = list_files(directory_path=str(tmp_path), pattern="*", recursive=True, include_hidden=True, head_limit=None)
    assert "a.txt" in out
    assert "state.d" not in out


def test_abstractignore_excludes_paths_for_search_and_write(tmp_path: Path) -> None:
    # Ignore the directory
    (tmp_path / ".abstractignore").write_text("ignored/\n", encoding="utf-8")

    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "note.txt").write_text("pygame\n", encoding="utf-8")

    out = search_files("pygame", path=str(tmp_path), file_pattern="*.txt", output_mode="files_with_matches", head_limit=None)
    assert "ignored" not in out

    w = write_file(file_path=str(ignored / "new.txt"), content="x\n")
    assert "Refused" in w


def test_read_file_refuses_when_ignored(tmp_path: Path) -> None:
    (tmp_path / ".abstractignore").write_text("secret.txt\n", encoding="utf-8")
    p = tmp_path / "secret.txt"
    p.write_text("top secret\n", encoding="utf-8")

    out = read_file(file_path=str(p))
    assert "ignored by .abstractignore policy" in out


def test_analyze_and_edit_refuse_when_ignored(tmp_path: Path) -> None:
    (tmp_path / ".abstractignore").write_text("secret/\n", encoding="utf-8")
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    p = secret_dir / "a.py"
    p.write_text("print('x')\n", encoding="utf-8")

    out = analyze_code(file_path=str(p))
    assert "ignored by .abstractignore policy" in out

    out2 = edit_file(file_path=str(p), pattern="x", replacement="y")
    assert "ignored by .abstractignore policy" in out2


