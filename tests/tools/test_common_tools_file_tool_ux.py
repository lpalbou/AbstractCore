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
        should_read_entire_file=False,
        start_line_one_indexed=2,
        end_line_one_indexed_inclusive=2,
    )

    assert out.startswith(f"File: {path} (1 lines)\n\n")
    assert "\n2: two\n" in out + "\n"


def test_read_file_inclusive_two_line_range_returns_both_lines(tmp_path) -> None:
    path = tmp_path / "demo2.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        should_read_entire_file=False,
        start_line_one_indexed=2,
        end_line_one_indexed_inclusive=3,
    )
    assert "\n2: b\n3: c\n" in out + "\n"


def test_read_file_preserves_trailing_spaces(tmp_path) -> None:
    path = tmp_path / "spaces.txt"
    path.write_text("a  \nb\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        should_read_entire_file=False,
        start_line_one_indexed=1,
        end_line_one_indexed_inclusive=1,
    )
    assert "1: a  " in out


def test_search_files_content_mode_line_prefix_is_line_number(tmp_path) -> None:
    path = tmp_path / "code.py"
    path.write_text("print('x')\n# TODO: fix\n# TODO: later\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.py", output_mode="content", head_limit=None)
    assert f"\n📄 {path}:\n" in out
    assert "    2: # TODO: fix" in out
    assert "    3: # TODO: later" in out
