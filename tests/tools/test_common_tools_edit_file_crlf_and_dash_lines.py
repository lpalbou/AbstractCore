"""edit_file 0216 follow-ups: CRLF preservation + '-- '-prefixed deletion-line parsing.

Adversarial findings (2026-07-08):
- edit_file read files with universal newlines, so CRLF files were silently rewritten
  to LF on every edit (whole-file corruption for CRLF codebases).
- A unified-diff DELETION of a line whose content starts with '-- ' (SQL/Lua comments,
  CLI flag docs) produces a diff line '--- <text>' that was misparsed as a new file
  header, refusing an otherwise valid patch.
"""

import pytest

from abstractcore.tools.common_tools import edit_file


def _write_bytes(path, data: bytes) -> None:
    path.write_bytes(data)


# ---------------------------------------------------------------------------
# CRLF preservation
# ---------------------------------------------------------------------------

def test_find_replace_preserves_crlf(tmp_path):
    p = tmp_path / "crlf.txt"
    _write_bytes(p, b"alpha\r\nbeta\r\ngamma\r\n")

    result = edit_file(str(p), "beta", "BETA")

    assert "\u274c" not in result  # no error marker
    data = p.read_bytes()
    assert data == b"alpha\r\nBETA\r\ngamma\r\n"


def test_unified_diff_preserves_crlf(tmp_path):
    p = tmp_path / "crlf_diff.txt"
    _write_bytes(p, b"one\r\ntwo\r\nthree\r\n")

    patch = (
        "--- a/crlf_diff.txt\n"
        "+++ b/crlf_diff.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        " three\n"
    )
    result = edit_file(str(p), patch)

    assert "\u274c" not in result, result
    assert p.read_bytes() == b"one\r\nTWO\r\nthree\r\n"


def test_range_replace_preserves_crlf(tmp_path):
    p = tmp_path / "crlf_range.txt"
    _write_bytes(p, b"l1\r\nl2\r\nl3\r\n")

    result = edit_file(str(p), "", "L2", start_line=2, end_line=2)

    assert "\u274c" not in result, result
    assert p.read_bytes() == b"l1\r\nL2\r\nl3\r\n"


def test_lf_file_stays_lf(tmp_path):
    p = tmp_path / "lf.txt"
    _write_bytes(p, b"alpha\nbeta\n")

    result = edit_file(str(p), "beta", "BETA")

    assert "\u274c" not in result
    assert p.read_bytes() == b"alpha\nBETA\n"


def test_mixed_endings_normalize_to_dominant_with_note(tmp_path):
    p = tmp_path / "mixed.txt"
    # CRLF-dominant file with one stray LF line.
    _write_bytes(p, b"a\r\nb\nc\r\nd\r\n")

    result = edit_file(str(p), "b", "B")

    assert "\u274c" not in result
    assert "mixed line endings" in result
    assert p.read_bytes() == b"a\r\nB\r\nc\r\nd\r\n"


def test_crlf_preview_does_not_write(tmp_path):
    p = tmp_path / "crlf_preview.txt"
    original = b"x\r\ny\r\n"
    _write_bytes(p, original)

    result = edit_file(str(p), "y", "Y", preview_only=True)

    assert "Preview" in result
    assert p.read_bytes() == original


def test_crlf_pattern_matches_crlf_file(tmp_path):
    # Models sometimes echo CRLF from read output into the pattern.
    p = tmp_path / "crlf_pat.txt"
    _write_bytes(p, b"first\r\nsecond\r\nthird\r\n")

    result = edit_file(str(p), "first\r\nsecond", "first\r\nSECOND")

    assert "\u274c" not in result, result
    assert p.read_bytes() == b"first\r\nSECOND\r\nthird\r\n"


def test_python_crlf_file_syntax_guard_still_works(tmp_path):
    # The pre-write parse guard must still refuse syntax-breaking edits on CRLF files.
    p = tmp_path / "code.py"
    _write_bytes(p, b"def f():\r\n    return 1\r\n")

    result = edit_file(str(p), "return 1", "return (")

    assert "Refused" in result and "syntax error" in result
    assert p.read_bytes() == b"def f():\r\n    return 1\r\n"


# ---------------------------------------------------------------------------
# '-- '-prefixed deletion lines in unified diffs
# ---------------------------------------------------------------------------

def test_diff_deletes_sql_comment_line(tmp_path):
    p = tmp_path / "query.sql"
    p.write_text("SELECT 1;\n-- old comment\nSELECT 2;\n", encoding="utf-8")

    patch = (
        "--- a/query.sql\n"
        "+++ b/query.sql\n"
        "@@ -1,3 +1,2 @@\n"
        " SELECT 1;\n"
        "--- old comment\n"
        " SELECT 2;\n"
    )
    result = edit_file(str(p), patch)

    assert "\u274c" not in result, result
    assert p.read_text(encoding="utf-8") == "SELECT 1;\nSELECT 2;\n"


def test_diff_replaces_lua_comment_line(tmp_path):
    p = tmp_path / "conf.lua"
    p.write_text("x = 1\n-- disable feature\ny = 2\n", encoding="utf-8")

    patch = (
        "--- a/conf.lua\n"
        "+++ b/conf.lua\n"
        "@@ -1,3 +1,3 @@\n"
        " x = 1\n"
        "--- disable feature\n"
        "+-- enable feature\n"
        " y = 2\n"
    )
    result = edit_file(str(p), patch)

    assert "\u274c" not in result, result
    assert p.read_text(encoding="utf-8") == "x = 1\n-- enable feature\ny = 2\n"


def test_multi_file_diff_still_refused(tmp_path):
    # A genuine second-file header (after the hunk's old side is fully consumed)
    # must still be detected and refused: this tool applies single-file patches.
    p = tmp_path / "a.txt"
    p.write_text("one\ntwo\n", encoding="utf-8")

    patch = (
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n"
        "--- a/b.txt\n"
        "+++ b/b.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+y\n"
    )
    result = edit_file(str(p), patch)

    assert "multiple files" in result
    assert p.read_text(encoding="utf-8") == "one\ntwo\n"


def test_diff_deleting_line_of_dashes(tmp_path):
    # Markdown horizontal rule '---' deleted -> diff line '----'
    p = tmp_path / "doc.md"
    p.write_text("title\n---\nbody\n", encoding="utf-8")

    patch = (
        "--- a/doc.md\n"
        "+++ b/doc.md\n"
        "@@ -1,3 +1,2 @@\n"
        " title\n"
        "----\n"
        " body\n"
    )
    result = edit_file(str(p), patch)

    assert "\u274c" not in result, result
    assert p.read_text(encoding="utf-8") == "title\nbody\n"
