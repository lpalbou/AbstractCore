from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_appends_post_edit_excerpt_single_hunk(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="b\n",
        replacement="b\nX\n",
        use_regex=False,
    )

    assert "Post-edit excerpt (to avoid an extra read_file):" in out
    assert f"File: {path.absolute()} (lines 1-4)" in out
    assert "1: a" in out
    assert "2: b" in out
    assert "3: X" in out
    assert "4: c" in out


def test_edit_file_merges_nearby_hunk_excerpts(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    lines = [f"l{i:02d}" for i in range(1, 36)]
    lines[9] = "EDITME"  # line 10
    lines[24] = "EDITME"  # line 25
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="EDITME",
        replacement="EDITED",
        max_replacements=-1,
    )

    hunk_headers = [line for line in out.splitlines() if line.startswith("@@")]
    assert len(hunk_headers) == 2, out
    assert "Post-edit excerpt (to avoid an extra read_file):" in out
    assert out.count("File: ") == 1, out
    assert f"File: {path.absolute()} (lines 6-29)" in out
    assert "10: EDITED" in out
    assert "25: EDITED" in out
