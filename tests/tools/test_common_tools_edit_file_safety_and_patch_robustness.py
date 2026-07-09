"""Tests for edit_file safety-by-default and patch robustness (backlog item 0216).

Covers:
1. max_replacements default = exactly one unique match; ambiguity fails loudly.
2. Context-anchored unified-diff application (header line numbers are hints).
3. JSON/YAML pre-write parse-refuse alongside the preserved Python ast guard.
4. ADR-0026 marking on bounded previews/notices.
"""
from __future__ import annotations

import json
import sys

import pytest

from abstractcore.tools.common_tools import edit_file


# ---------------------------------------------------------------------------
# 1. Safe-by-default replacements
# ---------------------------------------------------------------------------

def test_default_ambiguous_literal_pattern_fails_and_names_count(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    original = "x = 1\nx = 2\nx = 3\n"
    path.write_text(original, encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="x = ", replacement="y = ")

    assert out.startswith("❌ Ambiguous pattern:"), out
    assert "3 matches" in out, out
    assert "at line(s) 1, 2, 3" in out, out
    assert "unique" in out, out
    assert "max_replacements=-1" in out, out
    assert path.read_text(encoding="utf-8") == original

    # Preview mode must report the same ambiguity (no misleading "would apply").
    out_preview = edit_file(file_path=str(path), pattern="x = ", replacement="y = ", preview_only=True)
    assert out_preview.startswith("❌ Ambiguous pattern:"), out_preview
    assert path.read_text(encoding="utf-8") == original


def test_default_single_match_replaces(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="b", replacement="B")

    assert out.startswith("Edited "), out
    assert "replacements=1/1" in out.splitlines()[0], out
    assert path.read_text(encoding="utf-8") == "a\nB\nc\n"


def test_explicit_all_occurrences_replaces_all(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("x = 1\nx = 2\nx = 3\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="x = ", replacement="y = ", max_replacements=-1)

    assert out.startswith("Edited "), out
    assert "replacements=3/3" in out.splitlines()[0], out
    assert path.read_text(encoding="utf-8") == "y = 1\ny = 2\ny = 3\n"


def test_explicit_zero_means_all_occurrences(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("x\nx\nx\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="x", replacement="y", max_replacements=0)

    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8") == "y\ny\ny\n"


def test_explicit_limit_replaces_first_match_and_notes_remaining(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("x\nx\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="x", replacement="y", max_replacements=1)

    assert "replacements=1/2" in out.splitlines()[0], out
    assert "1 more match" in out, out
    assert path.read_text(encoding="utf-8") == "y\nx\n"


def test_string_all_opt_in_still_works(tmp_path) -> None:
    # Pre-039 robustness: numeric fields may arrive as strings; must not regress
    # once centralized coercion lands (it would deliver a real int).
    path = tmp_path / "demo.txt"
    path.write_text("x\nx\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="x", replacement="y", max_replacements="-1")

    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8") == "y\ny\n"


def test_boolean_max_replacements_is_rejected(tmp_path) -> None:
    # bool is an int subclass; False would otherwise silently mean "replace all".
    path = tmp_path / "demo.txt"
    original = "x\nx\n"
    path.write_text(original, encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="x", replacement="y", max_replacements=False)

    assert out.startswith("❌ Invalid max_replacements"), out
    assert path.read_text(encoding="utf-8") == original


def test_default_ambiguous_regex_pattern_fails(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    original = "x = 1\nx = 2\nx = 3\n"
    path.write_text(original, encoding="utf-8")

    out = edit_file(file_path=str(path), pattern=r"x = \d", replacement="y = 0", use_regex=True)

    assert out.startswith("❌ Ambiguous pattern:"), out
    assert "3 matches" in out, out
    assert "regex pattern" in out, out
    assert path.read_text(encoding="utf-8") == original


def test_default_ambiguous_flexible_whitespace_match_fails(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    original = "def a():\n    return 1\ndef b():\n    return 1\n"
    path.write_text(original, encoding="utf-8")

    # Two-space indent forces the flexible-whitespace path (exact match fails),
    # which matches BOTH `return 1` lines.
    out = edit_file(file_path=str(path), pattern="  return 1", replacement="  return 2")

    assert out.startswith("❌ Ambiguous pattern:"), out
    assert "2 matches" in out, out
    assert "at line(s) 2, 4" in out, out
    assert path.read_text(encoding="utf-8") == original


def test_line_range_scoping_disambiguates(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("x = 1\nx = 2\nx = 3\n", encoding="utf-8")

    out = edit_file(
        file_path=str(path),
        pattern="x = ",
        replacement="y = ",
        start_line=2,
        end_line=2,
    )

    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8") == "x = 1\ny = 2\nx = 3\n"


# ---------------------------------------------------------------------------
# 2. Context-anchored unified-diff application
# ---------------------------------------------------------------------------

def _numbered_file(tmp_path, name: str = "t.txt", n: int = 30):
    path = tmp_path / name
    lines = [f"L{i:02d}" for i in range(1, n + 1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, lines


def test_unified_diff_applies_with_drifted_header_line_numbers(tmp_path) -> None:
    path, _ = _numbered_file(tmp_path)

    # True positions: L10 block starts at line 10 (header says 13, off by -3);
    # L21 block starts at line 21 (header says 18, off by +3).
    patch = (
        "--- a/t.txt\n"
        "+++ b/t.txt\n"
        "@@ -13,3 +13,3 @@\n"
        " L10\n"
        "-L11\n"
        "+CHANGED1\n"
        " L12\n"
        "@@ -18,3 +18,3 @@\n"
        " L21\n"
        "-L22\n"
        "+CHANGED2\n"
        " L23\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("Edited "), out
    assert "Note (patch anchoring):" in out, out
    assert "offset -3" in out, out
    assert "offset +3" in out, out
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[10] == "CHANGED1"
    assert lines[21] == "CHANGED2"
    assert lines[9] == "L10" and lines[11] == "L12"
    assert lines[20] == "L21" and lines[22] == "L23"


def test_unified_diff_exact_header_still_applies_without_notes(tmp_path) -> None:
    path, _ = _numbered_file(tmp_path)

    patch = (
        "--- a/t.txt\n"
        "+++ b/t.txt\n"
        "@@ -10,3 +10,3 @@\n"
        " L10\n"
        "-L11\n"
        "+CHANGED\n"
        " L12\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("Edited "), out
    assert "Note (patch anchoring):" not in out, out
    assert path.read_text(encoding="utf-8").splitlines()[10] == "CHANGED"


def test_unified_diff_context_not_found_reports_reason(tmp_path) -> None:
    path, _ = _numbered_file(tmp_path)
    original = path.read_text(encoding="utf-8")

    patch = (
        "--- a/t.txt\n"
        "+++ b/t.txt\n"
        "@@ -10,3 +10,3 @@\n"
        " NOPE1\n"
        "-NOPE2\n"
        "+X\n"
        " NOPE3\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("❌ Error: Patch did not apply cleanly:"), out
    assert "context not found" in out, out
    assert "'NOPE1'" in out, out
    assert "line 10" in out, out
    assert path.read_text(encoding="utf-8") == original


def test_unified_diff_ambiguous_context_resolved_by_header(tmp_path) -> None:
    path = tmp_path / "t.txt"
    path.write_text("A\nB\nC\nx1\nA\nB\nC\nx2\n", encoding="utf-8")

    patch = (
        "--- a/t.txt\n"
        "+++ b/t.txt\n"
        "@@ -5,3 +5,3 @@\n"
        " A\n"
        "-B\n"
        "+BB\n"
        " C\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("Edited "), out
    # Strict header positioning picked the SECOND occurrence (line 5).
    assert path.read_text(encoding="utf-8") == "A\nB\nC\nx1\nA\nBB\nC\nx2\n"


def test_unified_diff_ambiguous_context_without_matching_header_fails(tmp_path) -> None:
    path = tmp_path / "t.txt"
    original = "A\nB\nC\nx1\nA\nB\nC\nx2\n"
    path.write_text(original, encoding="utf-8")

    patch = (
        "--- a/t.txt\n"
        "+++ b/t.txt\n"
        "@@ -3,3 +3,3 @@\n"
        " A\n"
        "-B\n"
        "+BB\n"
        " C\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("❌ Error: Patch did not apply cleanly:"), out
    assert "ambiguous" in out, out
    assert "lines 1, 5" in out, out
    assert path.read_text(encoding="utf-8") == original


def test_unified_diff_whitespace_flexible_context(tmp_path) -> None:
    path = tmp_path / "f.txt"
    path.write_text(
        "def foo():\n    return 1\n\ndef bar():\n    return 2\n",
        encoding="utf-8",
    )

    # Patch context/removal lines use 2-space indent; the file uses 4 spaces.
    patch = (
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -4,2 +4,2 @@\n"
        " def bar():\n"
        "-  return 2\n"
        "+  return 99\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("Edited "), out
    assert "whitespace" in out, out
    # Context lines keep the FILE's content; the added line is the patch's.
    assert path.read_text(encoding="utf-8") == (
        "def foo():\n    return 1\n\ndef bar():\n  return 99\n"
    )


def test_unified_diff_out_of_order_hunks_fail(tmp_path) -> None:
    path, _ = _numbered_file(tmp_path)
    original = path.read_text(encoding="utf-8")

    patch = (
        "--- a/t.txt\n"
        "+++ b/t.txt\n"
        "@@ -20,3 +20,3 @@\n"
        " L20\n"
        "-L21\n"
        "+X\n"
        " L22\n"
        "@@ -5,3 +5,3 @@\n"
        " L05\n"
        "-L06\n"
        "+Y\n"
        " L07\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("❌ Error: Patch did not apply cleanly:"), out
    assert "earlier in the file" in out, out
    assert path.read_text(encoding="utf-8") == original


def test_unified_diff_blank_context_line_without_prefix(tmp_path) -> None:
    path = tmp_path / "b.txt"
    path.write_text("A\n\nB\n", encoding="utf-8")

    # The blank context line lost its ' ' prefix (common transport behavior).
    patch = (
        "--- a/b.txt\n"
        "+++ b/b.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " A\n"
        "\n"
        "-B\n"
        "+C\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8") == "A\n\nC\n"


def test_unified_diff_pure_insertion_follows_after_line_convention(tmp_path) -> None:
    # `@@ -N,0 +M,k @@` means "insert AFTER line N" (diff -U0 convention).
    path = tmp_path / "ins.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    patch = (
        "--- a/ins.txt\n"
        "+++ b/ins.txt\n"
        "@@ -2,0 +3,1 @@\n"
        "+X\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8") == "a\nb\nX\nc\n"


def test_unified_diff_pure_insertion_prepend_and_append(tmp_path) -> None:
    path = tmp_path / "ins.txt"
    path.write_text("a\nb\n", encoding="utf-8")

    prepend = "@@ -0,0 +1,1 @@\n+TOP\n"
    out = edit_file(file_path=str(path), pattern=prepend)
    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8") == "TOP\na\nb\n"

    append = "@@ -3,0 +4,1 @@\n+END\n"
    out = edit_file(file_path=str(path), pattern=append)
    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8") == "TOP\na\nb\nEND\n"


def test_apply_unified_diff_helper_preserves_crlf_text() -> None:
    # Helper-level contract (parity with _flexible_whitespace_match): CRLF input
    # text stays CRLF. Note: edit_file itself reads files with universal
    # newlines, so whole-file newline style is normalized before this helper
    # runs — a pre-existing edit_file behavior shared by all edit modes.
    from abstractcore.tools.common_tools import _apply_unified_diff, _parse_unified_diff

    patch = (
        "@@ -1,3 +1,3 @@\n"
        " a\n"
        "-b\n"
        "+B\n"
        " c\n"
    )
    _, hunks, err = _parse_unified_diff(patch)
    assert err is None
    updated, apply_err, _notes = _apply_unified_diff("a\r\nb\r\nc\r\n", hunks)
    assert apply_err is None
    assert updated == "a\r\nB\r\nc\r\n"


def test_unified_diff_python_syntax_guard_still_refuses(tmp_path) -> None:
    path = tmp_path / "demo.py"
    original = "def f():\n    return 1\n"
    path.write_text(original, encoding="utf-8")

    patch = (
        "--- a/demo.py\n"
        "+++ b/demo.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return (\n"
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("❌ Refused:"), out
    assert "python syntax error" in out.lower(), out
    assert path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# 3. JSON/YAML pre-write parse-refuse
# ---------------------------------------------------------------------------

def test_json_edit_introducing_syntax_error_is_refused(tmp_path) -> None:
    path = tmp_path / "cfg.json"
    original = '{\n  "a": 1,\n  "b": 2\n}\n'
    path.write_text(original, encoding="utf-8")

    out = edit_file(file_path=str(path), pattern='"b": 2', replacement='"b": 2,')

    assert out.startswith("❌ Refused:"), out
    assert "JSON syntax error" in out, out
    assert path.read_text(encoding="utf-8") == original


def test_json_valid_edit_applies(tmp_path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text('{\n  "a": 1,\n  "b": 2\n}\n', encoding="utf-8")

    out = edit_file(file_path=str(path), pattern='"b": 2', replacement='"b": 3')

    assert out.startswith("Edited "), out
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 3}


def test_json_edit_on_already_invalid_file_is_not_refused(tmp_path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text('{"a": 1,}\n', encoding="utf-8")  # invalid before the edit

    out = edit_file(file_path=str(path), pattern='"a": 1', replacement='"a": 2')

    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8") == '{"a": 2,}\n'


def test_json_syntax_guard_applies_in_unified_diff_mode(tmp_path) -> None:
    path = tmp_path / "cfg.json"
    original = '{\n  "a": 1,\n  "b": 2\n}\n'
    path.write_text(original, encoding="utf-8")

    patch = (
        "--- a/cfg.json\n"
        "+++ b/cfg.json\n"
        "@@ -2,2 +2,2 @@\n"
        '   "a": 1,\n'
        '-  "b": 2\n'
        '+  "b": 2,\n'
    )

    out = edit_file(file_path=str(path), pattern=patch)

    assert out.startswith("❌ Refused:"), out
    assert "JSON syntax error" in out, out
    assert path.read_text(encoding="utf-8") == original


def test_yaml_edit_introducing_syntax_error_is_refused(tmp_path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    original = "name: test\nitems:\n  - one\n  - two\n"
    path.write_text(original, encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="items:", replacement="items: [")

    assert out.startswith("❌ Refused:"), out
    assert "YAML syntax error" in out, out
    assert path.read_text(encoding="utf-8") == original


def test_yaml_valid_edit_applies(tmp_path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "cfg.yml"
    path.write_text("name: test\nitems:\n  - one\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="name: test", replacement="name: prod")

    assert out.startswith("Edited "), out
    assert path.read_text(encoding="utf-8").startswith("name: prod\n")


def test_yaml_validation_skipped_gracefully_without_pyyaml(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("name: test\nitems:\n  - one\n", encoding="utf-8")

    # Simulate pyyaml being absent: `import yaml` raises ImportError.
    monkeypatch.setitem(sys.modules, "yaml", None)

    out = edit_file(file_path=str(path), pattern="items:", replacement="items: [")

    # No YAML validation available -> the edit proceeds (graceful skip).
    assert out.startswith("Edited "), out
    assert "items: [" in path.read_text(encoding="utf-8")


def test_python_ast_guard_still_refuses_breakage(tmp_path) -> None:
    path = tmp_path / "demo.py"
    original = "def f():\n    return 1\n"
    path.write_text(original, encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="    return 1", replacement="    return (")

    assert out.startswith("❌ Refused:"), out
    assert "python syntax error" in out.lower(), out
    assert path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# 4. ADR-0026: bounded previews are marked
# ---------------------------------------------------------------------------

def test_noop_error_pattern_preview_marks_truncation(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("hello\n", encoding="utf-8")
    long_text = "x" * 250

    out = edit_file(file_path=str(path), pattern=long_text, replacement=long_text)

    assert out.startswith("❌ Error:"), out
    assert "(truncated; 250 chars total)" in out, out


def test_no_match_diagnostics_mark_truncated_lines(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("prefix specialtoken " + "z" * 260 + "\n", encoding="utf-8")

    out = edit_file(file_path=str(path), pattern="specialtoken notpresent", replacement="x")

    assert out.startswith("❌ No occurrences"), out
    assert "… (truncated)" in out, out


def test_post_edit_excerpt_omission_is_disclosed(tmp_path) -> None:
    path = tmp_path / "big.txt"
    lines = [f"l{i}" for i in range(1, 301)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    replacement = "\n".join(f"n{i}" for i in range(1, 251)) + "\n"

    out = edit_file(
        file_path=str(path),
        pattern="",
        start_line=1,
        end_line=250,
        replacement=replacement,
    )

    assert out.startswith("Edited "), out
    assert "Post-edit excerpt omitted" in out, out
    assert "Post-edit excerpt (to avoid an extra read_file):" not in out, out
    assert path.read_text(encoding="utf-8").startswith("n1\n")
