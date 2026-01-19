from __future__ import annotations

from abstractcore.tools.common_tools import list_files, read_file, search_files


def test_list_files_empty_directory_reports_exists_but_empty(tmp_path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    out = list_files(directory_path=str(empty_dir), pattern="*")
    assert out == f"Directory '{empty_dir}' exists but is empty"


def test_list_files_hidden_only_directory_reports_hidden_entries(tmp_path) -> None:
    hidden_dir = tmp_path / "hidden_only"
    hidden_dir.mkdir()
    (hidden_dir / ".secret").write_text("x\n", encoding="utf-8")

    out = list_files(directory_path=str(hidden_dir), pattern="*")
    assert out == f"Directory '{hidden_dir}' exists but contains only hidden entries (use include_hidden=True)"


def test_read_file_inclusive_single_line_range_returns_line_number(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        start_line=2,
        end_line=2,
    )

    assert out.startswith(f"File: {path} (1 lines)\n\n")
    assert "\n2: two\n" in out + "\n"


def test_read_file_inclusive_two_line_range_returns_both_lines(tmp_path) -> None:
    path = tmp_path / "demo2.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        start_line=2,
        end_line=3,
    )
    assert "\n2: b\n3: c\n" in out + "\n"


def test_read_file_preserves_trailing_spaces(tmp_path) -> None:
    path = tmp_path / "spaces.txt"
    path.write_text("a  \nb\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        start_line=1,
        end_line=1,
    )
    assert "1: a  " in out


def test_search_files_content_mode_line_prefix_is_line_number(tmp_path) -> None:
    path = tmp_path / "code.py"
    path.write_text("print('x')\n# TODO: fix\n# TODO: later\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.py", head_limit=None)
    assert f"\n📄 {path}:\n" in out
    assert "    2: # TODO: fix" in out
    assert "    3: # TODO: later" in out


def test_search_files_head_limit_is_per_file_not_global(tmp_path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("TODO a1\nTODO a2\nTODO a3\n", encoding="utf-8")
    b.write_text("TODO b1\nTODO b2\nTODO b3\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.txt", head_limit=2, max_hits=None)

    assert f"\n📄 {a}:\n" in out
    assert f"\n📄 {b}:\n" in out
    assert "    1: TODO a1" in out
    assert "    2: TODO a2" in out
    assert "TODO a3" not in out
    assert "    1: TODO b1" in out
    assert "    2: TODO b2" in out
    assert "TODO b3" not in out


def test_search_files_max_hits_limits_number_of_files(tmp_path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    a.write_text("TODO a\n", encoding="utf-8")
    b.write_text("TODO b\n", encoding="utf-8")
    c.write_text("TODO c\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.txt", head_limit=1, max_hits=2)
    assert out.count("\n📄 ") == 2
    assert sum(int(str(p) in out) for p in (a, b, c)) == 2


def test_read_file_entire_file_small_returns_all_lines(tmp_path) -> None:
    path = tmp_path / "small.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    out = read_file(file_path=str(path))
    assert out.startswith(f"File: {path} (3 lines)\n\n")
    assert "\n1: one\n2: two\n3: three\n" in out + "\n"


def test_read_file_entire_file_refuses_when_over_line_limit(tmp_path) -> None:
    path = tmp_path / "many-lines.txt"
    path.write_text("\n".join(["x"] * 2001) + "\n", encoding="utf-8")

    out = read_file(file_path=str(path))
    assert out.startswith(f"Refused: File '{path}' is too large to read entirely")
    assert "> 2000 lines" in out
    assert "Next step:" in out


def test_read_file_entire_file_does_not_refuse_based_on_bytes_only(tmp_path) -> None:
    path = tmp_path / "large-bytes.txt"
    path.write_text("a" * 100_001, encoding="utf-8")

    out = read_file(file_path=str(path))
    assert out.startswith(f"File: {path} (1 lines)\n\n1: a")


def test_read_file_range_refuses_when_requested_lines_over_limit(tmp_path) -> None:
    path = tmp_path / "range-too-large.txt"
    path.write_text("\n".join([str(i) for i in range(1, 3001)]) + "\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        start_line=1,
        end_line=2001,
    )

    assert out.startswith("Refused: Requested range would return 2001 lines")
    assert "> 2000 lines" in out
