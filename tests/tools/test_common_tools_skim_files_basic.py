from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.tools.common_tools import skim_files


pytestmark = pytest.mark.basic


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
    assert "… skipped " in out


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
