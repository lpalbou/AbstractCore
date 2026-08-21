"""`output_mode` accepts the near-miss spellings models actually send.

Two live sessions (2026-08-21) refused on `output_mode='lines'` and
`output_mode='context_lines'`, both meaning the DEFAULT mode. The second is
this tool's own doing: `when_to_use` listed "context_lines=N ... output_mode=
files_with_matches|count" in one comma-separated run and never named
`content`, so the model bound the neighbouring PARAMETER as a mode VALUE — and
sent `context_lines: 5` alongside it, which is the tell.

The synonym set is CLOSED. `count_files` is genuinely ambiguous between two
modes and must keep refusing: a catch-all would trade a loud refusal for a
silent wrong mode, which is the trade this whole investigation is about.
"""

from __future__ import annotations

import pytest

from abstractcore.tools.common_tools import search_files


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("spelling", ["lines", "line", "context", "context_lines", "matches", "text"])
def test_line_output_synonyms_resolve_to_content(tree, spelling):
    out = search_files(pattern="def", path=str(tree), output_mode=spelling)
    assert out.startswith("Search results for pattern"), out[:120]
    assert "alpha" in out and "beta" in out


@pytest.mark.parametrize("spelling", ["files", "paths", "filenames", "files_only"])
def test_path_output_synonyms_resolve_to_files_with_matches(tree, spelling):
    out = search_files(pattern="def", path=str(tree), output_mode=spelling)
    assert out.startswith("Files matching pattern"), out[:120]


def test_the_canonical_values_are_untouched(tree):
    assert search_files(pattern="def", path=str(tree), output_mode="content").startswith(
        "Search results for pattern"
    )
    assert search_files(pattern="def", path=str(tree), output_mode="files_with_matches").startswith(
        "Files matching pattern"
    )
    assert "Match count" in search_files(pattern="def", path=str(tree), output_mode="count") or \
        search_files(pattern="def", path=str(tree), output_mode="count").strip()


@pytest.mark.parametrize("spelling", ["bogus", "count_files", "everything"])
def test_the_set_stays_closed(tree, spelling):
    """An ambiguous or unknown spelling must still refuse, loudly and by name."""
    out = search_files(pattern="def", path=str(tree), output_mode=spelling)
    assert out.startswith("Error: output_mode must be one of"), out[:120]
    assert repr(spelling).strip("'") in out


def test_the_model_facing_hint_names_the_default_mode():
    """The hint that caused the collision must name `content` as a mode."""
    hint = str(getattr(search_files.tool_definition, "when_to_use", "") or "")
    assert "output_mode=content|files_with_matches|count" in hint, hint
