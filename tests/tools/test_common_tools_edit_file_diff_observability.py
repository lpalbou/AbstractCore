from __future__ import annotations

import re

from abstractcore.tools.common_tools import edit_file


def test_edit_file_diff_includes_context_and_old_new_line_numbers(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="b\n",
        replacement="b\nX\n",
        use_regex=False,
    )

    assert out.startswith("Edited "), out
    assert "@@" in out

    # Expect 1-line context before/after (b and c) and an insertion line (X).
    assert re.search(r"(?m)^\\s+2\\s+2\\s+\\| b$", out), out
    assert re.search(r"(?m)^\\+\\s+3\\s+\\| X$", out), out
    assert re.search(r"(?m)^\\s+3\\s+4\\s+\\| c$", out), out

