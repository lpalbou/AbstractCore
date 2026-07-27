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


def test_list_files_empty_result_names_matching_hidden_entries(tmp_path: Path) -> None:
    """When the pattern matches ONLY hidden entries, the empty-result message
    must say so — an agent probing for '.env' must not conclude it is absent
    (adversary F1: the streaming rewrite lost the old disambiguator)."""
    (tmp_path / "visible.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")

    out = list_files(str(tmp_path), pattern=".env*", recursive=False, include_hidden=False, head_limit=None)

    assert "matching hidden entries exist" in out
    assert "include_hidden=True" in out


def test_list_files_hidden_hint_fires_for_pruned_hidden_dirs_recursive(tmp_path: Path) -> None:
    """Recursive walks prune hidden dirs before the per-entry loop; the hint
    must still fire when the pattern matches a pruned hidden directory."""
    (tmp_path / ".secrets").mkdir()
    (tmp_path / ".secrets" / "key.pem").write_text("k", encoding="utf-8")
    (tmp_path / "visible.py").write_text("x = 1", encoding="utf-8")

    out = list_files(str(tmp_path), pattern=".secrets*", recursive=True, include_hidden=False, head_limit=None)

    assert "matching hidden entries exist" in out


def test_list_files_composition_counts_files_only_not_directories(tmp_path: Path) -> None:
    """Directories with dots in their names must not appear in the extension
    composition — 'pkg.v0.data/' is not a .data file (adversary F2)."""
    for i in range(6):
        (tmp_path / f"mod_{i}.py").write_text("x = 1", encoding="utf-8")
    for i in range(3):
        (tmp_path / f"pkg.v{i}.data").mkdir()

    # head_limit below the match count forces truncation, which emits the
    # composition line even on a small (fully scanned) tree.
    out = list_files(str(tmp_path), pattern="*", recursive=False, include_hidden=False, head_limit=5)

    assert "Composition (total): 6 .py" in out
    assert ".data" not in out.split("Composition", 1)[1]


def test_list_files_collect_cap_boundary_exact_vs_many(tmp_path: Path) -> None:
    """At head_limit + look-ahead budget (10 + 500 = 510) exactly, the stream
    exhausts and counts stay exact; one entry more flips to the honest
    'many' label (adversary: boundary was unpinned)."""
    exact_dir = tmp_path / "exact"
    exact_dir.mkdir()
    for i in range(510):
        (exact_dir / f"f_{i:04d}.txt").write_text("x", encoding="utf-8")
    out_exact = list_files(str(exact_dir), pattern="*", recursive=False, include_hidden=False, head_limit=10)
    assert "showing 10 of 510 entries" in out_exact

    many_dir = tmp_path / "many"
    many_dir.mkdir()
    for i in range(511):
        (many_dir / f"f_{i:04d}.txt").write_text("x", encoding="utf-8")
    out_many = list_files(str(many_dir), pattern="*", recursive=False, include_hidden=False, head_limit=10)
    assert "showing 10 of many entries" in out_many




