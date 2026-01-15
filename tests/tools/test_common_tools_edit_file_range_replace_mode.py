from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_can_replace_line_range_without_pattern(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        start_line=2,
        end_line=3,
        replacement="B\nC\n",
    )

    assert out.startswith("Edited "), out
    assert "replacements=1/1" in out, out
    assert "Post-edit excerpt (to avoid an extra read_file):" in out
    assert path.read_text(encoding="utf-8") == "a\nB\nC\nd\n"


def test_edit_file_range_replace_requires_both_start_and_end(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        start_line=2,
        replacement="X\n",
    )

    assert out.startswith("❌ Invalid range replace:"), out
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"


def test_edit_file_schema_keeps_pattern_required_for_guidance() -> None:
    tool_def = edit_file._tool_definition.to_dict()
    required = tool_def.get("required_args")
    assert isinstance(required, list) and "pattern" in required

