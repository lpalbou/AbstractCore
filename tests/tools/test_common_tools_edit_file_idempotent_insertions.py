from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_suffix_insertion_is_idempotent_when_already_present(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\nb\nX\nc\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="b\n",
        replacement="b\nX\n",
        use_regex=False,
    )

    assert isinstance(out, str)
    assert out.startswith("No changes applied"), out
    assert path.read_text(encoding="utf-8") == "a\nb\nX\nc\n"


def test_edit_file_prefix_insertion_is_idempotent_when_already_present(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\n# comment\nb\nc\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="b\n",
        replacement="# comment\nb\n",
        use_regex=False,
    )

    assert isinstance(out, str)
    assert out.startswith("No changes applied"), out
    assert path.read_text(encoding="utf-8") == "a\n# comment\nb\nc\n"

