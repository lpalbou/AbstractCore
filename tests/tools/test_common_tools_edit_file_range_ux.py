"""Range-parameter UX for edit_file (live-trace driven).

Pins the policy adopted after a live agent trace wasted a model turn on
`start_line=0, end_line=0` (0-based habit) against a 112-line file:

- validation is BATCHED (one refusal names every invalid parameter),
- unambiguous off-by-convention values are tolerated with a visible note
  (start_line 0 -> 1; end_line past EOF -> last line; end_line -1 = EOF),
- ambiguous values (end_line 0) are refused with teaching,
- a scoped miss probes the WHOLE file and reports matches outside the range
  (stale line numbers) definitively instead of speculatively,
- a scoped success discloses the scope it searched and any matches outside it,
- the ordinary un-scoped path is byte-identical to the pre-policy render.
"""

from __future__ import annotations

from abstractcore.tools.common_tools import edit_file


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _numbered_file(tmp_path, total: int = 12, name: str = "demo.txt"):
    return _write(tmp_path, name, "".join(f"line {i}\n" for i in range(1, total + 1)))


def test_batched_validation_names_every_invalid_param(tmp_path) -> None:
    path = _numbered_file(tmp_path)
    before = path.read_text(encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="line 5", replacement="X", start_line=-3, end_line=0)

    assert out.startswith("❌ Invalid line range"), out
    # ONE refusal must teach BOTH problems (each retry costs a full model turn).
    assert "start_line -3" in out, out
    assert "end_line 0 is ambiguous" in out, out
    assert "Omit start_line/end_line to search the whole file" in out, out
    assert "12 lines" in out, out
    assert path.read_text(encoding="utf-8") == before


def test_trace_shape_zero_zero_refuses_end_and_discloses_start_clamp(tmp_path) -> None:
    # The live trace: a 0-based caller sent start_line=0 AND end_line=0.
    path = _numbered_file(tmp_path)

    out = edit_file(file_path=str(path), pattern="line 5", replacement="X", start_line=0, end_line=0)

    assert out.startswith("❌ Invalid line range"), out
    assert "end_line 0 is ambiguous" in out, out
    # start_line 0 is tolerated; the refusal must still disclose that reading.
    assert "start_line 0 is treated as line 1" in out, out
    # No derived cross-param bullet comparing against the already-refused end value.
    assert "greater than end_line" not in out, out


def test_start_line_zero_clamps_with_visible_note(tmp_path) -> None:
    path = _numbered_file(tmp_path)

    out = edit_file(file_path=str(path), pattern="line 5", replacement="LINE 5", start_line=0, end_line=12)

    assert out.startswith("Edited "), out
    assert "Note (line range): start_line 0 is treated as line 1" in out, out
    assert "line 5" not in path.read_text(encoding="utf-8").splitlines()[4]
    assert "LINE 5" in path.read_text(encoding="utf-8")


def test_end_line_past_eof_clamps_with_visible_note(tmp_path) -> None:
    path = _numbered_file(tmp_path)

    out = edit_file(file_path=str(path), pattern="line 5", replacement="LINE 5", start_line=1, end_line=999)

    assert out.startswith("Edited "), out
    assert "end_line 999 exceeds the file and is treated as 12 (end of file)" in out, out
    assert "LINE 5" in path.read_text(encoding="utf-8")


def test_end_line_minus_one_is_eof_sentinel(tmp_path) -> None:
    path = _numbered_file(tmp_path)

    out = edit_file(file_path=str(path), pattern="line 10", replacement="LINE 10", start_line=8, end_line=-1)

    assert out.startswith("Edited "), out
    assert "(searched lines 8-12)" in out, out
    assert "LINE 10" in path.read_text(encoding="utf-8")


def test_scoped_miss_reports_matches_outside_range(tmp_path) -> None:
    # The stale-scope class: line numbers drifted, the pattern exists — just not
    # inside the scope. The error must be definitive, not "may exist outside".
    path = _numbered_file(tmp_path)
    before = path.read_text(encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="line 10", replacement="X", start_line=1, end_line=3)

    assert out.startswith("❌ No occurrences of"), out
    assert "not found in lines 1-3" in out, out
    assert "1 match(es) exist outside this range (first at line 10)" in out, out
    assert "may be stale" in out, out
    assert path.read_text(encoding="utf-8") == before


def test_scoped_regex_miss_reports_matches_outside_range(tmp_path) -> None:
    path = _numbered_file(tmp_path)

    out = edit_file(
        file_path=str(path),
        pattern=r"line 1[01]",
        replacement="X",
        use_regex=True,
        start_line=1,
        end_line=3,
    )

    assert out.startswith("❌ No matches found for regex pattern"), out
    assert "2 match(es) exist outside this range (first at line 10)" in out, out
    assert "may be stale" in out, out


def test_scoped_miss_when_pattern_nowhere_is_definitive(tmp_path) -> None:
    path = _numbered_file(tmp_path)

    out = edit_file(file_path=str(path), pattern="zzz-not-here", replacement="X", start_line=1, end_line=3)

    assert out.startswith("❌ No occurrences of"), out
    assert "not found anywhere in the file" in out, out
    assert "fix the pattern rather than the line range" in out, out


def test_scoped_success_discloses_scope_and_outside_matches(tmp_path) -> None:
    # Scoped uniqueness is evaluated in scope only; a success must disclose the
    # scope and the matches it never considered.
    path = _write(
        tmp_path,
        "dup.txt",
        "line 1\nconst DUP = 1;\nline 3\nline 4\nline 5\nline 6\nline 7\nconst DUP = 1;\n",
    )

    out = edit_file(
        file_path=str(path),
        pattern="const DUP = 1;",
        replacement="const DUP = 2;",
        start_line=1,
        end_line=3,
    )

    assert out.startswith("Edited "), out
    assert "(searched lines 1-3)" in out.splitlines()[0], out
    assert "1 match(es) outside this range were NOT considered" in out, out
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[1] == "const DUP = 2;"
    assert text.splitlines()[7] == "const DUP = 1;"  # outside scope: untouched


def test_reversed_range_is_refused_with_teaching(tmp_path) -> None:
    path = _numbered_file(tmp_path)
    before = path.read_text(encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="line 5", replacement="X", start_line=9, end_line=2)

    assert out.startswith("❌ Invalid line range"), out
    assert "start_line (9) is greater than end_line (2)" in out, out
    assert "Omit start_line/end_line" in out, out
    assert path.read_text(encoding="utf-8") == before


def test_schema_teaches_one_based_line_params() -> None:
    # The docstring never reaches the model (the schema builder emits type/default
    # only), so the 1-based teaching must live in the exported parameter schema —
    # this is the channel the live-trace model was missing.
    params = edit_file._tool_definition.parameters
    assert "1-based" in params["start_line"].get("description", ""), params["start_line"]
    assert "1-based" in params["end_line"].get("description", ""), params["end_line"]
    assert "-1" in params["end_line"].get("description", ""), params["end_line"]


def test_unscoped_edit_render_is_byte_identical_to_pre_policy_render(tmp_path) -> None:
    # The overwhelmingly common path (no line params) must not change AT ALL.
    # This expected string was captured from the pre-policy implementation.
    path = _write(tmp_path, "plain.txt", "alpha\nbravo\ncharlie\n")

    out = edit_file(file_path=str(path), pattern="bravo", replacement="BRAVO")

    expected = (
        "Edited <PATH> (+1 -1) replacements=1/1\n"
        "@@ -1,3 +1,3 @@\n"
        " 1 1 | alpha\n"
        "-2   | bravo\n"
        "+  2 | BRAVO\n"
        " 3 3 | charlie\n"
        "\n"
        "Post-edit excerpt (to avoid an extra read_file):\n"
        "File: <PATH> (lines 1-3)\n"
        "\n"
        "1: alpha\n"
        "2: BRAVO\n"
        "3: charlie"
    )
    assert out.replace(str(path), "<PATH>") == expected
