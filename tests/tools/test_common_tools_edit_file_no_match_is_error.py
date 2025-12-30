from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_no_match_returns_error_prefix(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("hello\nworld\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="missing", replacement="x", use_regex=False)
    assert isinstance(out, str)
    assert out.startswith("❌"), out


def test_edit_file_no_regex_match_returns_error_prefix(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("hello\nworld\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern=r"missing\\s+pattern", replacement="x", use_regex=True)
    assert isinstance(out, str)
    assert out.startswith("❌"), out

