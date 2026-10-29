"""Changelog fragments (scripts/changelog.py): parse, assemble at release, and the PR check."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("changelog_script", ROOT / "scripts" / "changelog.py")
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)


def test_every_fragment_in_the_repository_is_well_formed() -> None:
    for f in cl.fragment_files():
        cl.parse(f.read_text(encoding="utf-8"), name=f.name)


def test_parse_refuses_what_a_release_could_not_place() -> None:
    with pytest.raises(ValueError, match="unknown section"):
        cl.parse("### Improved\n\n- x\n")
    with pytest.raises(ValueError, match="before the first"):
        cl.parse("- x\n### Fixed\n- y\n")
    with pytest.raises(ValueError, match="no entries"):
        cl.parse("### Fixed\n\n")


def test_assemble_groups_sections_keeps_older_releases_and_removes_fragments(tmp_path: Path) -> None:
    d = tmp_path / "changelog.d"
    d.mkdir()
    (d / "README.md").write_text("# not a fragment\n")
    (d / "b.md").write_text("### Fixed\n\n- **b fixed**\n")
    (d / "a.md").write_text("### Fixed\n\n- **a fixed**\n\n### Added\n\n- **a added**\n")
    ch = tmp_path / "CHANGELOG.md"
    ch.write_text("# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n\n### Added\n\n- old\n")
    used = cl.assemble("0.2.0", "2026-10-29", changelog=ch, directory=d)
    text = ch.read_text()
    assert [p.name for p in used] == ["a.md", "b.md"] and not (d / "a.md").exists() and (d / "README.md").exists()
    assert text.index("## [Unreleased]") < text.index("## [0.2.0] - 2026-10-29") < text.index("## [0.1.0]")
    new = text[text.index("## [0.2.0]"):text.index("## [0.1.0]")]
    assert new.index("### Added") < new.index("### Fixed")  # Keep a Changelog order
    assert new.count("### Fixed") == 1 and "- **a fixed**\n- **b fixed**" in new
    assert "- old" in text


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@example.invalid", "HOME": str(repo), "PATH": "/usr/bin:/bin"})


def test_check_needs_a_fragment_when_src_changes(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = tmp_path / "repo"
    src = cl.SOURCE_PREFIXES[0].rstrip("/")
    (repo / src).mkdir(parents=True)
    (repo / "changelog.d").mkdir()
    (repo / src / "a.py").write_text("x = 1\n")
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-q", "-b", "pr")
    monkeypatch.setattr(cl, "ROOT", repo)
    (repo / src / "a.py").write_text("x = 2\n")
    _git(repo, "commit", "-qam", "change src")
    assert cl.check("main") == 1
    assert "adds no changelog fragment" in capsys.readouterr().out
    (repo / "changelog.d" / "change.md").write_text("### Changed\n\n- **x is 2**\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fragment")
    assert cl.check("main") == 0
    (repo / "changelog.d" / "bad.md").write_text("### Improved\n\n- x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "bad fragment")
    with pytest.raises(ValueError):
        cl.check("main")
