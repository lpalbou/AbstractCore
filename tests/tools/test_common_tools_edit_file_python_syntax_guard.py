from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def test_edit_file_refuses_to_write_python_syntax_breakage(tmp_path) -> None:
    path = tmp_path / "demo.py"
    path.write_text(
        "def f():\n"
        "    if True:\n"
        "        pass\n"
        "    elif False:\n"
        "        pass\n",
        encoding="utf-8",
    )

    out = edit_file(
        file_path=str(path),
        start_line=3,
        end_line=3,
        replacement="        pass        elif False:",  # invalid Python (two statements on one line)
    )

    assert out.startswith("❌ Refused:"), out
    assert "python syntax error" in out.lower()
    assert path.read_text(encoding="utf-8").startswith("def f():\n")
