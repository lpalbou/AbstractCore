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
SOURCES = (
    "README.md", "docs/getting-started.md", "docs/prerequisites.md",
    "docs/api.md", "docs/session.md", "docs/async-guide.md",
    "docs/tool-calling.md", "docs/tool-syntax-rewriting.md",
    "docs/structured-output.md", "docs/media-handling-system.md",
    "docs/vision-capabilities.md", "docs/embeddings.md",
    "docs/centralized-config.md", "docs/server.md", "docs/endpoint.md",
    "docs/troubleshooting.md", "docs/faq.md", "docs/architecture.md",
    "docs/native-mlx-runtime.md", "docs/native-mlx-benchmarks.md",
    "docs/speculative-decoding.md", "docs/generation-cancel.md",
    "docs/examples.md", "docs/mcp.md",
    "docs/structured-logging.md", "docs/api-reference.md",
)


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
    for source in SOURCES:
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
    if args.check:
        if current != expected:
            raise SystemExit("llms-full.txt is stale; run python scripts/update_llms.py")
        print("llms-full.txt matches all %d canonical source pages" % len(SOURCES))
    else:
        target.write_text(expected, encoding="utf-8")
        print("Updated llms-full.txt from %d canonical source pages" % len(SOURCES))


if __name__ == "__main__":
    main()
