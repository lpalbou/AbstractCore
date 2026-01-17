from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_identical_pattern_and_replacement_returns_error(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("hello\nworld\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="hello", replacement="hello", use_regex=False)
    assert isinstance(out, str)
    assert out.startswith("❌"), out
    assert "identical" in out.lower(), out

    assert path.read_text(encoding="utf-8") == "hello\nworld\n"

