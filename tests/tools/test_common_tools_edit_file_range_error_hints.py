from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_missing_pattern_with_line_range_includes_hint(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="b\n",
        replacement="B\n",
        start_line=1,
        end_line=1,
        use_regex=False,
    )

    assert isinstance(out, str)
    assert out.startswith("❌ No occurrences of"), out
    assert "Hint:" in out
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"

