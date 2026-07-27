from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.tools.common_tools import (
    skim_files,
    _SKIM_HEADING_RE,
    _SKIM_HR_RE,
    _SKIM_LIST_RE,
    _SKIM_CODE_DECL_RE,
    _SKIM_SENTENCE_END_RE,
)


pytestmark = pytest.mark.basic


def test_skim_structure_patterns_are_live_regression() -> None:
    """Regression for audit item 0827: the structure-detection regexes were raw strings
    with DOUBLED backslashes (r"\\\\s"), matching a literal backslash instead of the
    whitespace class — heading/list/def/sentence detection was entirely dead while the
    tool still "worked" via bookend sampling. This isolation test would have caught it;
    it exists so a future edit cannot silently re-break the patterns."""
    # Headings (ATX)
    assert _SKIM_HEADING_RE.match("## Title")
    assert _SKIM_HEADING_RE.match("###### H6")
    assert not _SKIM_HEADING_RE.match("just a paragraph")
    assert not _SKIM_HEADING_RE.match("#nospace")  # ATX requires whitespace after #
    # Horizontal rule / setext underline
    assert _SKIM_HR_RE.match("---")
    assert _SKIM_HR_RE.match("=====")
    # Lists / checkboxes
    assert _SKIM_LIST_RE.match("- item")
    assert _SKIM_LIST_RE.match("* item")
    assert _SKIM_LIST_RE.match("1. item")
    assert _SKIM_LIST_RE.match("[ ] task")
    assert _SKIM_LIST_RE.match("[x] done")
    assert not _SKIM_LIST_RE.match("-nospace")
    # Code-ish declarations
    assert _SKIM_CODE_DECL_RE.match("def foo():")
    assert _SKIM_CODE_DECL_RE.match("class Bar:")
    assert not _SKIM_CODE_DECL_RE.match("defoo")
    # First-sentence boundary
    m = _SKIM_SENTENCE_END_RE.search("Hello world. Second sentence.")
    assert m is not None and "Hello world. Second sentence."[: m.end(1)].strip() == "Hello world."


def test_skim_files_prioritizes_midfile_heading(tmp_path: Path) -> None:
    """End-to-end: a heading deep in a long body is surfaced because _is_heading_line
    (now live) prioritizes it in the middle sample — not merely by bookend luck."""
    body = ["Filler paragraph sentence number %d here." % i for i in range(1, 60)]
    body[40] = "## Deep Section Marker"
    body[41] = "The first line under the deep heading. Trailing sentence."
    p = tmp_path / "long.md"
    p.write_text("\n".join(body) + "\n", encoding="utf-8")

    out = skim_files(paths=[str(p)], target_percent=6.0, head_lines=5, tail_lines=5)
    assert "## Deep Section Marker" in out
    # Discriminating assertion (review 2026-07-25): the follow-up line is pulled ONLY by the
    # heading rule (heading_followup). With the old dead regexes it is absent — so this line
    # is what proves the fix, not the heading's presence (which interval sampling can hit by luck).
    assert "first line under the deep heading" in out


def test_skim_files_returns_line_numbered_excerpts_and_gaps(tmp_path: Path) -> None:
    p = tmp_path / "demo.md"
    p.write_text(
        "\n".join(
            [
                "# Title",
                "",
                "Intro first sentence. Second sentence should not appear.",
                "",
                "## Section One",
                "This section describes X in detail. More details follow.",
                "- Bullet one: important point",
                "- Bullet two: another point",
                "",
                "## Section Two",
                "Another paragraph starts here. Additional sentence.",
                "END.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out = skim_files(paths=[str(p)], target_percent=8.0)
    assert isinstance(out, str)
    assert str(p) in out
    assert "1: # Title" in out
    assert "5: ## Section One" in out
    assert "12: END." in out
    assert "skipped" not in out.lower()


def test_skim_files_heading_includes_followup_sentence(tmp_path: Path) -> None:
    p = tmp_path / "demo2.md"
    p.write_text("# Title\n\nThis is the intro sentence. Second sentence.\n\n## Next\nBody.\n", encoding="utf-8")

    # Force a tiny head window so the follow-up line must be pulled in by the heading rule.
    out = skim_files(paths=[str(p)], head_lines=1, tail_lines=0, target_percent=1.0)
    assert "1: # Title" in out
    assert "3: This is the intro sentence." in out
    assert "skipped" not in out.lower()


def test_skim_files_minimum_20_lines_when_possible(tmp_path: Path) -> None:
    p = tmp_path / "many_lines.md"
    p.write_text("\n".join([f"Line {i}. Second sentence." for i in range(1, 81)]) + "\n", encoding="utf-8")

    out = skim_files(paths=[str(p)], target_percent=1.0)
    excerpt_lines = [line for line in out.splitlines() if line.lstrip().split(":", 1)[0].isdigit()]
    assert len(excerpt_lines) >= 20


def test_skim_files_accepts_multiple_paths_and_legacy_separators(tmp_path: Path) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# A\n\nIntro.\n", encoding="utf-8")
    b.write_text("# B\n\nIntro.\n", encoding="utf-8")

    out = skim_files(paths=[str(a), str(b)])
    assert str(a) in out
    assert str(b) in out
    assert "\n\n---\n\n" in out

    out2 = skim_files(paths=f"{a},{b}")
    assert str(a) in out2
    assert str(b) in out2


def test_skim_files_refuses_when_ignored(tmp_path: Path) -> None:
    (tmp_path / ".abstractignore").write_text("secret.txt\n", encoding="utf-8")
    p = tmp_path / "secret.txt"
    p.write_text("top secret\n", encoding="utf-8")

    out = skim_files(paths=[str(p)])
    assert "ignored by .abstractignore policy" in out


def test_skim_files_refuses_binary(tmp_path: Path) -> None:
    p = tmp_path / "bin.dat"
    p.write_bytes(b"\xff\xfe\x00\x00")

    out = skim_files(paths=[str(p)])
    assert "appears to be binary" in out.lower()


def test_skim_files_relative_path_shows_input_and_resolved_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "rel.md"
    p.write_text("# Title\n\nIntro.\n", encoding="utf-8")

    out = skim_files(paths=["rel.md"], target_percent=8.0)
    assert str(p) in out  # resolved absolute path
    assert "Input: rel.md" in out
