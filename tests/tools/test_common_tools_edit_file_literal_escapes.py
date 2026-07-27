"""edit_file exact-match-first escape handling (audit item 0829).

Regression for the confirmed defect: `edit_file` unconditionally rewrote literal
`\\n`/`\\t`/`\\r` in BOTH pattern and replacement into real control characters
before matching, so a caller inserting a literal escape into SOURCE CODE (e.g.
`sep = "\\n"`) got a real line break written — silent corruption on unguarded
file types and an unfixable retry loop on guarded ones.

The fix: pattern/replacement are kept VERBATIM; escape-normalization survives only
as a LABELED fallback on the PATTERN (for weak models that over-escape), and the
replacement is never rewritten. Regex mode is never normalized.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.tools.common_tools import edit_file


pytestmark = pytest.mark.basic


def _edit(tmp_path: Path, name: str, content: str, *args, **kwargs) -> tuple[str, str]:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    out = edit_file(str(p), *args, **kwargs)
    return out, p.read_text(encoding="utf-8")


def test_replacement_literal_backslash_n_is_written_verbatim_js(tmp_path: Path) -> None:
    """The headline case: writing a literal backslash-n escape into JS source. It must be
    written as the two characters backslash-n, NOT a real newline."""
    out, content = _edit(tmp_path, "demo.js", 'const sep = "OLD";\n', 'const sep = "OLD";', 'const sep = "\\n";')
    assert out.splitlines()[0].startswith("Edited")
    assert 'const sep = "\\n";' in content  # literal backslash-n preserved
    assert 'const sep = "\n";' not in content.replace('";\n', '";X")')  # not a real newline in the literal


def test_replacement_literal_backslash_t_written_verbatim(tmp_path: Path) -> None:
    out, content = _edit(tmp_path, "conf.txt", "sep=OLD\n", "sep=OLD", "sep=\\t")
    assert "sep=\\t" in content  # literal backslash-t, not a real tab
    assert "\t" not in content


def test_matching_source_containing_a_literal_escape(tmp_path: Path) -> None:
    """A pattern containing a literal backslash-n matches source that contains it, and the
    replacement (another literal escape) is written verbatim."""
    out, content = _edit(tmp_path, "split.py", 'x = re.split("\\n", s)\n', 're.split("\\n", s)', 're.split("\\t", s)')
    assert out.splitlines()[0].startswith("Edited")
    assert 're.split("\\t", s)' in content


def test_over_escaped_pattern_falls_back_and_labels(tmp_path: Path) -> None:
    """A pattern over-escaped as literal backslash-n (matching content with a REAL newline)
    still matches via the labeled fallback, and the output discloses the unescape."""
    out, content = _edit(tmp_path, "multi.txt", "foo\nbar\nbaz\n", "foo\\nbar", "MERGED")
    assert out.splitlines()[0].startswith("Edited")
    assert "MERGED" in content and "foo" not in content
    assert "escape handling" in out.lower()  # the fallback is disclosed, not silent


def test_raw_pattern_wins_over_fallback_no_spurious_note(tmp_path: Path) -> None:
    """When the raw pattern matches, there is no fallback and no escape note."""
    out, content = _edit(tmp_path, "plain.txt", "hello world\n", "hello", "goodbye")
    assert content == "goodbye world\n"
    assert "escape handling" not in out.lower()


def test_regex_replacement_template_newline_not_pre_corrupted(tmp_path: Path) -> None:
    """Regex mode must not be escape-normalized: a \\n in the re.sub template is regex
    newline semantics, applied by re.sub, not rewritten before compile."""
    out, content = _edit(tmp_path, "re.py", "a=1\n", r"a=(\d+)", r"a=\1\n# added", use_regex=True)
    assert content == "a=1\n# added\n"


def test_regex_pattern_with_escapes_matches(tmp_path: Path) -> None:
    """A regex pattern using \\s/\\d etc. is compiled as-is (not unescaped to literals)."""
    out, content = _edit(tmp_path, "code.py", "x   =   5\n", r"x\s+=\s+\d+", "x = 9", use_regex=True)
    assert content == "x = 9\n"


def test_range_replace_replacement_verbatim(tmp_path: Path) -> None:
    """Range-replace must also write the replacement verbatim (no escape rewrite)."""
    out, content = _edit(
        tmp_path, "r.txt", "line1\nline2\nline3\n", "", "sep = \"\\n\"",
        start_line=2, end_line=2,
    )
    assert 'sep = "\\n"' in content  # literal backslash-n, not a real newline


def test_regex_pattern_matches_literal_backslash_n_in_content(tmp_path: Path) -> None:
    """DISCRIMINATING regex test (review 2026-07-25): a regex `a\\nb` (escaped backslash +
    n) must match a LITERAL backslash-n in file content. Under the old up-front normalization
    the pattern's `\\n` was rewritten to a real newline before re.compile, so it matched a
    newline instead — this test fails on the old code, passes on the fix."""
    # File contains the 4 chars: a, backslash, n, b (a literal escape in source).
    out, content = _edit(tmp_path, "lit.py", 's = "a\\nb"\n', r"a\\nb", "MATCHED", use_regex=True)
    assert out.splitlines()[0].startswith("Edited")
    assert 's = "MATCHED"' in content


def test_regex_replacement_template_emits_literal_backslash_n(tmp_path: Path) -> None:
    """DISCRIMINATING regex test: a re.sub template `\\1\\\\nEND` must emit a LITERAL
    backslash-n. Under the old normalization the template's `\\n` was rewritten to a real
    newline before re.sub, emitting a newline instead — fails on old, passes on the fix."""
    # .txt (not .py): a literal backslash-n is valid text but INVALID Python source, so a
    # .py file would (correctly) hit the parse guard — the template-emission property is
    # about re.sub, not Python validity.
    out, content = _edit(tmp_path, "tpl.txt", "id=7\n", r"id=(\d+)", r"id=\1\\nEND", use_regex=True)
    assert out.splitlines()[0].startswith("Edited")
    assert "id=7\\nEND" in content  # literal backslash-n, not a real newline
    assert "id=7\nEND" not in content  # NOT a real newline between 7 and END
