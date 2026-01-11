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

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.py", output_mode="content", head_limit=None)
    assert f"\n📄 {path}:\n" in out
    assert "    2: # TODO: fix" in out
    assert "    3: # TODO: later" in out


def test_search_files_context_mode_defaults_to_five_lines_and_marks_match(tmp_path) -> None:
    path = tmp_path / "code.py"
    lines = [f"line {i}" for i in range(1, 21)]
    lines[9] = "line 10: TODO match"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.py", output_mode="context", head_limit=None)
    assert "Search context for pattern 'TODO'" in out
    assert "(±5 lines)" in out
    assert f"\n📄 {path}:\n" in out
    assert "    5: line 5" in out
    assert "  > 10: line 10: TODO match" in out
    assert "    15: line 15" in out


def test_search_files_context_mode_respects_head_limit_matches(tmp_path) -> None:
    path = tmp_path / "code.py"
    lines = [f"line {i}" for i in range(1, 41)]
    lines[9] = "line 10: TODO match"
    lines[29] = "line 30: TODO match"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.py", output_mode="context", head_limit=1)
    assert "  > 10: line 10: TODO match" in out
    assert "  > 30: line 30: TODO match" not in out
    assert "... (showing context for first 1 matches)" in out


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
