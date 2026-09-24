"""Refresh the llms-full canonical appendix; --check fails on stale content.

Keep the curated handbook before the appendix intact. Source links are rebased
to the repository root, and fenced examples are preserved byte-for-byte.
"""
from __future__ import annotations

import argparse
import ast
import posixpath
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "## Appendix A) Inlined canonical docs snapshot"
INDEX = "docs/README.md"
# Folders under docs/ that are not user documentation (planning, history,
# engineering notes, ADRs); their pages never enter llms-full.txt.
NON_USER_DIRS = ("adr", "archive", "backlog", "known_bugs", "reports", "research")


def indexed_sources() -> tuple:
    """README.md, then every docs page the docs index links to, in index order.

    The index (`docs/README.md`) is the single list of user pages: a page it
    links is inlined, a page it does not link is not. Every top-level
    `docs/*.md` page must be linked, so a new page cannot be left out of
    llms-full.txt silently.
    """
    index = (ROOT / INDEX).read_text(encoding="utf-8")
    sources = ["README.md"]
    for dest in re.findall(r"\]\(([^\s)#]+\.md)(?:#[^\s)]*)?\)", index):
        path = posixpath.normpath(posixpath.join("docs", dest))
        if not path.startswith("docs/") or path == INDEX:
            continue
        if path.split("/")[1] in NON_USER_DIRS:
            continue
        if not (ROOT / path).is_file():
            raise ValueError("%s links to a missing page: %s" % (INDEX, dest))
        if path not in sources:
            sources.append(path)
    unlisted = sorted(
        "docs/" + page.name for page in (ROOT / "docs").glob("*.md")
        if page.name != "README.md" and "docs/" + page.name not in sources
    )
    if unlisted:
        raise ValueError("%s does not link these pages: %s" % (INDEX, ", ".join(unlisted)))
    return tuple(sources)


def rebase_links(body: str, source: str) -> str:
    parent = posixpath.dirname(source)

    def replace(match):
        dest = match.group(2)
        if re.match(r"[A-Za-z][\w+.-]*:", dest) or dest.startswith("/"):
            return match.group(0)
        if dest.startswith("#"):
            dest = source + dest
        else:
            dest = posixpath.normpath(posixpath.join(parent, dest))
        return match.group(1) + dest + match.group(3)

    lines = []
    fence_char = None
    fence_length = 0
    for line in body.splitlines(keepends=True):
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            token = fence.group(1)
            if fence_char is None:
                fence_char, fence_length = token[0], len(token)
            elif token[0] == fence_char and len(token) >= fence_length:
                fence_char = None
            lines.append(line)
        elif fence_char is not None:
            lines.append(line)
        else:
            lines.append(re.sub(r"(\]\()([^\s)]+)(\))", replace, line))
    return "".join(lines)


def render(current: str) -> str:
    if current.count(MARKER) != 1:
        raise ValueError("Expected exactly one canonical appendix marker")
    tree = ast.parse((ROOT / "abstractcore/utils/version.py").read_text())
    version = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__version__"
                           for target in node.targets))
    preamble = current.split(MARKER, 1)[0].rstrip()
    preamble = re.sub(r"^Package version: .+$", "Package version: " + version,
                      preamble, flags=re.MULTILINE)
    sections = [preamble, MARKER,
                "The linked source pages are canonical. Fenced examples retain their source formatting."]
    for source in indexed_sources():
        body = (ROOT / source).read_text(encoding="utf-8")
        sections.extend(["---", "### Inlined: `" + source + "`",
                         rebase_links(body, source).rstrip()])
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "llms-full.txt"
    current = target.read_text(encoding="utf-8")
    expected = render(current)
    count = len(indexed_sources())
    if args.check:
        if current != expected:
            raise SystemExit("llms-full.txt is stale; run python scripts/update_llms.py")
        print("llms-full.txt matches all %d canonical source pages" % count)
    else:
        target.write_text(expected, encoding="utf-8")
        print("Updated llms-full.txt from %d canonical source pages" % count)


if __name__ == "__main__":
    main()
