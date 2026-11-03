#!/usr/bin/env python3
"""Changelog fragments: each pull request adds `changelog.d/<pr-or-slug>.md`; a release assembles them.

Why: every PR used to edit `CHANGELOG.md`'s `[Unreleased]` section, so every merge of main into a
branch conflicted there (and merges stacked duplicate `### Fixed` headings). A fragment is a new
file, which never conflicts.

A fragment is Keep a Changelog markdown: one or more `### Added` / `### Changed` / `### Deprecated` /
`### Removed` / `### Fixed` / `### Security` headings, each followed by its entries.

    python scripts/changelog.py check [--base origin/main]   # CI: a PR touching abstractcore/ adds a fragment,
        # unless it is the release PR: the version source bumped to X.Y.Z, CHANGELOG.md gaining its dated
        # `## [X.Y.Z] - <date>` section, and no fragment left in changelog.d/ (all assembled)
    python scripts/changelog.py preview                      # the assembled section, printed
    python scripts/changelog.py assemble X.Y.Z [--date YYYY-MM-DD]
        # release step: writes `## [X.Y.Z] - <date>` under `## [Unreleased]` from every fragment,
        # in section order, then deletes the fragments (commit both with the version bump)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
FRAGMENTS = ROOT / "changelog.d"
CHANGELOG = ROOT / "CHANGELOG.md"
SECTIONS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
SOURCE_PREFIXES = ("abstractcore/",)
UNRELEASED = "## [Unreleased]"
VERSION_FILE = "abstractcore/utils/version.py"
_VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.M)


def _is_fragment_path(path: str) -> bool:
    """`changelog.d/**/<name>.md`, any case, except a README."""
    low = path.replace("\\", "/").lower()
    return low.startswith("changelog.d/") and low.endswith(".md") and not low.endswith("readme.md")


def fragment_files(directory: Path = FRAGMENTS) -> List[Path]:
    """Every fragment under `directory`, in subfolders too and whatever the case of `.md` (a
    `fix.MD` or `sub/fix.md` left behind must be seen by `check` and `assemble`)."""
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".md"
                  and p.name.lower() != "readme.md")


def parse(text: str, *, name: str = "fragment") -> List[Tuple[str, str]]:
    """[(section, body)] in the order written; raises ValueError on text outside a known section."""
    out: List[Tuple[str, str]] = []
    section = None
    lines: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^###\s+(.+?)\s*$", line)
        if m:
            if section is not None:
                out.append((section, "\n".join(lines).strip("\n")))
            title = m.group(1).strip().title()
            if title not in SECTIONS:
                raise ValueError(f"{name}: unknown section {m.group(1)!r} (use one of: {', '.join(SECTIONS)})")
            section, lines = title, []
            continue
        if section is None:
            if line.strip():
                raise ValueError(f"{name}: text before the first '### <Section>' heading: {line.strip()[:60]!r}")
            continue
        lines.append(line)
    if section is not None:
        out.append((section, "\n".join(lines).strip("\n")))
    if not out or not any(body.strip() for _, body in out):
        raise ValueError(f"{name}: no entries (write '### Fixed' and a '- **What changed** ...' line)")
    return out


def assemble_text(files: List[Path]) -> str:
    grouped: Dict[str, List[str]] = {s: [] for s in SECTIONS}
    for f in files:
        for section, body in parse(f.read_text(encoding="utf-8"), name=f.name):
            if body.strip():
                grouped[section].append(body.strip("\n"))
    parts = []
    for section in SECTIONS:
        if grouped[section]:
            parts.append(f"### {section}\n\n" + "\n".join(grouped[section]))
    return "\n\n".join(parts)


def assemble(version: str, date: str, *, changelog: Path = CHANGELOG, directory: Path = FRAGMENTS) -> List[Path]:
    files = fragment_files(directory)
    if not files:
        raise SystemExit("no fragments in changelog.d/: nothing to release")
    body = assemble_text(files)
    text = changelog.read_text(encoding="utf-8")
    if UNRELEASED not in text:
        raise SystemExit(f"{changelog.name} has no '{UNRELEASED}' heading")
    head, rest = text.split(UNRELEASED, 1)
    # Whatever still sits under [Unreleased] (hand-written before fragments) is kept, above them.
    nxt = re.search(r"^## \[", rest, flags=re.M)
    leftover, tail = (rest[: nxt.start()], rest[nxt.start():]) if nxt else (rest, "")
    leftover = leftover.strip("\n")
    section = f"## [{version}] - {date}\n\n" + ((leftover + "\n\n") if leftover else "") + body + "\n\n"
    changelog.write_text(f"{head}{UNRELEASED}\n\n{section}{tail}", encoding="utf-8")
    for f in files:
        f.unlink()
    return files


def changed_files(base: str) -> Tuple[List[str], List[str]]:
    """(every changed path, paths ADDED) between the merge base with `base` and HEAD."""
    def _git(*args: str) -> List[str]:
        out = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        return [x for x in out.splitlines() if x.strip()]

    mb = _git("merge-base", base, "HEAD")[0]
    return _git("diff", "--name-only", mb, "HEAD"), _git("diff", "--name-only", "--diff-filter=A", mb, "HEAD")


def _git_out(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _version_in(text: str) -> str | None:
    m = _VERSION_RE.search(text or "")
    return m.group(1) if m else None


def _release_section_re(version: str) -> re.Pattern:
    return re.compile(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$", re.M)


def release_version(base: str, changed: List[str]) -> str | None:
    """The version this pull request releases, or None when it is not a release.

    A release bumps `VERSION_FILE` and adds `CHANGELOG.md`'s dated section for that version
    (`## [X.Y.Z] - YYYY-MM-DD`, absent at the merge base). Whether its fragments were all assembled
    is the caller's check, so a release that left one behind is refused by name."""
    if VERSION_FILE not in changed or "CHANGELOG.md" not in changed:
        return None
    mb = _git_out("merge-base", base, "HEAD").strip()
    try:
        before = _version_in(_git_out("show", f"{mb}:{VERSION_FILE}"))
    except subprocess.CalledProcessError:
        before = None
    after = _version_in((ROOT / VERSION_FILE).read_text(encoding="utf-8"))
    if not after or after == before:
        return None
    section = _release_section_re(after)
    try:
        had_section = bool(section.search(_git_out("show", f"{mb}:CHANGELOG.md")))
    except subprocess.CalledProcessError:
        had_section = False
    if had_section or not section.search((ROOT / "CHANGELOG.md").read_text(encoding="utf-8")):
        return None
    return after


def release_section_entries(version: str) -> List[str]:
    """The `- ` entries of CHANGELOG.md's `## [version] - <date>` section (up to the next `## [`)."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    m = _release_section_re(version).search(text)
    if not m:
        return []
    nxt = re.search(r"^## \[", text[m.end():], flags=re.M)
    body = text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]
    return [line for line in body.splitlines() if line.lstrip().startswith("- ")]


def check(base: str) -> int:
    changed, added = changed_files(base)
    version = release_version(base, changed)
    if version is not None:
        fragments_dir = ROOT / "changelog.d"
        left = [p.relative_to(fragments_dir).as_posix() for p in fragment_files(fragments_dir)]
        if left:
            print(f"Release {version}: changelog.d/ still holds {len(left)} fragment(s) ({', '.join(left[:8])}"
                  f"{' ...' if len(left) > 8 else ''}). `python scripts/changelog.py assemble {version}` writes them "
                  "into CHANGELOG.md and removes them; remove the ones already assembled.")
            return 1
        if not release_section_entries(version):
            print(f"Release {version}: its CHANGELOG.md section `## [{version}] - <date>` has no entries "
                  "(the fragments were removed without being assembled?): `python scripts/changelog.py assemble "
                  f"{version}` writes them.")
            return 1
        print(f"changelog: release {version}, CHANGELOG.md has its section and every fragment is assembled")
        return 0
    for f in [p for p in added if _is_fragment_path(p)]:
        parse((ROOT / f).read_text(encoding="utf-8"), name=f)  # a malformed fragment fails here
    src = [p for p in changed if p.startswith(SOURCE_PREFIXES)]
    frags = [p for p in added if p.lower().startswith("changelog.d/") and not p.lower().endswith("readme.md")]
    if "CHANGELOG.md" in changed and src and not frags:
        print("CHANGELOG.md was edited: add the entry as changelog.d/<pr-or-slug>.md instead (see changelog.d/README.md)")
        return 1
    if src and not frags:
        print(f"This pull request changes {SOURCE_PREFIXES[0]} but adds no changelog fragment: add changelog.d/<pr-or-slug>.md "
              "(see changelog.d/README.md).\n  changed: " + ", ".join(src[:8]) + (" ..." if len(src) > 8 else ""))
        return 1
    print(f"changelog: {len(frags)} fragment(s) for {len(src)} changed source file(s)")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--base", default="origin/main")
    sub.add_parser("preview")
    a = sub.add_parser("assemble")
    a.add_argument("version")
    a.add_argument("--date", default=_dt.date.today().isoformat())
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return check(args.base)
    if args.cmd == "preview":
        files = fragment_files()
        print(assemble_text(files) if files else "(no fragments)")
        return 0
    used = assemble(args.version, args.date)
    print(f"CHANGELOG.md: [{args.version}] - {args.date} from {len(used)} fragment(s); fragments removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
