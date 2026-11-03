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


def _release_repo(tmp_path: Path, monkeypatch, *, pending: bool) -> Path:
    """A repository at 1.0.0 (with one fragment still in changelog.d/ when `pending`), on a `release`
    branch off `main`."""
    repo = tmp_path / "repo"
    (repo / "abstractcore" / "utils").mkdir(parents=True)
    (repo / "changelog.d").mkdir()
    (repo / "changelog.d" / "README.md").write_text("# not a fragment\n")
    if pending:
        (repo / "changelog.d" / "fix.md").write_text("### Fixed\n\n- **a fix**\n")
    (repo / cl.VERSION_FILE).write_text('__version__ = "1.0.0"\n')
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n\n### Added\n\n- old\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-q", "-b", "release")
    monkeypatch.setattr(cl, "ROOT", repo)
    return repo


def _bump(repo: Path, version: str, section: bool) -> None:
    (repo / cl.VERSION_FILE).write_text(f'__version__ = "{version}"\n')
    if section:
        text = (repo / "CHANGELOG.md").read_text()
        (repo / "CHANGELOG.md").write_text(text.replace(
            "## [Unreleased]\n", f"## [Unreleased]\n\n## [{version}] - 2026-11-03\n\n### Fixed\n\n- **a fix**\n", 1))


def test_a_release_pr_with_every_fragment_assembled_passes(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _release_repo(tmp_path, monkeypatch, pending=False)  # every fragment already assembled
    _bump(repo, "1.1.0", section=True)
    _git(repo, "commit", "-qam", "release 1.1.0")
    assert cl.check("main") == 0
    assert "release 1.1.0" in capsys.readouterr().out


def test_a_release_pr_that_leaves_a_fragment_behind_fails_naming_it(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _release_repo(tmp_path, monkeypatch, pending=True)
    _bump(repo, "1.1.0", section=True)
    _git(repo, "commit", "-qam", "release 1.1.0, fragment left")
    assert cl.check("main") == 1
    assert "Release 1.1.0: changelog.d/ still holds 1 fragment(s) (fix.md)" in capsys.readouterr().out


@pytest.mark.parametrize("version,section", [("1.0.0", True), ("1.1.0", False)])
def test_without_both_a_version_bump_and_its_section_it_is_an_ordinary_pr(tmp_path, monkeypatch, capsys, version, section):
    """A CHANGELOG section without a bump, or a bump without its section, is not a release: a source change
    still needs a fragment, and none is added here."""
    repo = _release_repo(tmp_path, monkeypatch, pending=False)
    _bump(repo, version, section=section)
    if version == "1.0.0":
        (repo / "abstractcore" / "a.py").write_text("x = 1\n")
        _git(repo, "add", "-A")
    _git(repo, "commit", "-qam", "not a release")
    assert cl.check("main") == 1
    out = capsys.readouterr().out
    assert "adds no changelog fragment" in out or "CHANGELOG.md was edited" in out


def test_a_release_pr_whose_section_has_no_entries_fails_naming_the_version(tmp_path: Path, monkeypatch, capsys) -> None:
    """Fragments deleted, the header written alone: a release with nothing in it is refused at PR time."""
    repo = _release_repo(tmp_path, monkeypatch, pending=False)
    (repo / cl.VERSION_FILE).write_text('__version__ = "1.1.0"\n')
    text = (repo / "CHANGELOG.md").read_text()
    (repo / "CHANGELOG.md").write_text(text.replace("## [Unreleased]\n", "## [Unreleased]\n\n## [1.1.0] - 2026-11-03\n\n", 1))
    _git(repo, "commit", "-qam", "release 1.1.0, empty section")
    assert cl.check("main") == 1
    assert "Release 1.1.0: its CHANGELOG.md section `## [1.1.0] - <date>` has no entries" in capsys.readouterr().out


def test_a_fragment_in_a_subfolder_or_with_an_upper_case_suffix_is_seen(tmp_path: Path, monkeypatch, capsys) -> None:
    """check and assemble read the same list: `changelog.d/**/*.md`, any case, README excluded."""
    repo = _release_repo(tmp_path, monkeypatch, pending=False)
    (repo / "changelog.d" / "late").mkdir()
    (repo / "changelog.d" / "late" / "fix.md").write_text("### Fixed\n\n- **a late fix**\n")
    (repo / "changelog.d" / "LOUD.MD").write_text("### Added\n\n- **a loud addition**\n")
    (repo / "changelog.d" / "late" / "README.md").write_text("# not a fragment\n")
    assert [p.relative_to(repo / "changelog.d").as_posix() for p in cl.fragment_files(repo / "changelog.d")] == [
        "LOUD.MD", "late/fix.md"]
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "leftovers")
    _bump(repo, "1.1.0", section=True)
    _git(repo, "commit", "-qam", "release 1.1.0")
    assert cl.check("main") == 1
    assert "still holds 2 fragment(s) (LOUD.MD, late/fix.md)" in capsys.readouterr().out
