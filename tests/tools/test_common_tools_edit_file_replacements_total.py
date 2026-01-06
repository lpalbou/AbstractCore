from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_reports_total_matches_and_remaining_when_limited(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("x\nx\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="x",
        replacement="y",
        max_replacements=1,
    )

    first_line = out.splitlines()[0] if out else ""
    assert "replacements=1/2" in first_line
    assert "Note: 1 more match" in out

