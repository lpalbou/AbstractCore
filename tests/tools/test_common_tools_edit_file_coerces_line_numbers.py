from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_coerces_string_line_numbers(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="two",
        replacement="TWO",
        start_line="2",
        end_line="2",
        use_regex=False,
    )

    assert isinstance(out, str)
    assert not out.startswith("❌ Error editing file:"), out
    assert path.read_text(encoding="utf-8") == "one\nTWO\nthree\n"

