"""The analyze_code next-step hint is agent-steering text: it must teach
edit_file's REAL contract, not a wrong calling pattern.

Live incident (operator trace, 2026-07-26, abstractcode-tui): the outline's old
hint said "then edit_file(start_line/end_line) for a bounded edit", so a model
dutifully passed line params it did not need — 0-based and stale — got refused,
and burned a turn. These tests pin the corrected teaching:

- default edit_file mode = short UNIQUE pattern, NO line params;
- start_line/end_line are 1-based scope limiters for disambiguation
  (or bound a range replace with pattern="");
- outline line numbers go stale after edits (re-run / re-read discipline).

Semantic asserts on purpose (never full-string equality): wording may be
polished, the CLAIMS may not regress.
"""

from __future__ import annotations

from pathlib import Path

from abstractcore.tools.code_analysis import ANALYZE_CODE_NEXT_STEP_HINT
from abstractcore.tools.common_tools import analyze_code


def _hint_line(output: str) -> str:
    """The next-step hint renders as line 2 of every outline (after the header)."""
    lines = output.splitlines()
    assert len(lines) >= 2, f"outline too short to carry a hint: {output!r}"
    return lines[1]


def test_next_step_hint_teaches_unique_pattern_and_staleness() -> None:
    hint = ANALYZE_CODE_NEXT_STEP_HINT
    # Mode teaching: unique pattern is the default; line params are optional.
    assert "UNIQUE pattern" in hint
    assert "no line params" in hint
    # Line params' real jobs: disambiguation and range replace.
    assert "disambiguate" in hint
    assert 'pattern=""' in hint
    # Base + staleness discipline (the trace's two failure axes).
    assert "1-based" in hint
    assert "stale" in hint
    # The reader must be told HOW to refresh, not just that numbers drift.
    assert "re-run analyze_code" in hint or "re-read" in hint


def test_next_step_hint_never_regresses_to_line_param_teaching() -> None:
    # The exact old teaching that steered a model into a bad call: presenting
    # edit_file(start_line/end_line) as THE calling convention.
    assert "edit_file(start_line/end_line)" not in ANALYZE_CODE_NEXT_STEP_HINT


def test_deep_lane_and_engine_lane_render_the_same_hint(tmp_path: Path) -> None:
    # Deep lane (python, bespoke analyzer in common_tools.py).
    py = tmp_path / "demo.py"
    py.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    # Engine lane (rust, declarative spec in code_analysis.py).
    rs = tmp_path / "demo.rs"
    rs.write_text("fn main() {\n    println!(\"hi\");\n}\n", encoding="utf-8")

    hint_py = _hint_line(analyze_code(file_path=str(py)))
    hint_rs = _hint_line(analyze_code(file_path=str(rs)))

    # One constant, no drifting second copy (the two lanes used to carry
    # byte-duplicated strings — that duplication is how hint fixes rot).
    assert hint_py == ANALYZE_CODE_NEXT_STEP_HINT
    assert hint_rs == ANALYZE_CODE_NEXT_STEP_HINT


def test_outline_line_numbers_are_one_based(tmp_path: Path) -> None:
    # The hint promises "1-based": pin the outline's numbering to file reality
    # so the promise can never silently drift from what is emitted.
    path = tmp_path / "numbered.py"
    path.write_text(
        "import os\n"        # line 1
        "\n"                 # line 2
        "def alpha():\n"     # line 3
        "    return 1\n"     # line 4
        "\n"                 # line 5
        "class Beta:\n"      # line 6
        "    pass\n",        # line 7
        encoding="utf-8",
    )

    out = analyze_code(file_path=str(path))
    assert "3-4: alpha()" in out
    assert "Beta (lines 6-7)" in out
