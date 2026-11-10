#!/usr/bin/env python3
"""Sync AbstractCore's user documentation into the abstractcore.ai (gh-pages) site.

The site mirrors the user documentation only: docs/*.md, docs/apps/*.md and docs/assets/,
the root project files, llms.txt / llms-full.txt and the agent skill. Internal folders
(docs/backlog, docs/adr, docs/reports, docs/research, docs/archive, docs/known_bugs) are not
mirrored, matching the repository's context7.json. A rendered page that links to any file the
site does not mirror points at that file on GitHub at the released tag; a link whose target does
not exist at that tag stops the sync.

    python3 scripts/sync_docs.py --source <export of the tag> --repo <abstractcore checkout> --ref v2.35.0 --copy-md
"""
from __future__ import annotations

import argparse
import html
import posixpath
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import markdown
import markdown.extensions.toc as md_toc
from markdown.extensions.toc import TocExtension


def github_slugify(value: str, separator: str = "-") -> str:
    # Heading ids as GitHub makes them, so anchors written for GitHub (`page.md#some--heading`) resolve here too.
    value = re.sub(r"[^\w\- ]", "", value.strip().lower())
    return value.replace(" ", separator)


def github_unique(id: str, ids: "set[str]") -> str:
    # GitHub numbers repeated headings `name-1`, `name-2` (Python-Markdown would use `name_1`).
    base, n = id, 0
    while id in ids or not id:
        n += 1
        id = f"{base}-{n}"
    ids.add(id)
    return id


md_toc.unique = github_unique


GITHUB = "https://github.com/lpalbou/AbstractCore"

MENU_ITEMS_JS = (
    "[{'text': 'Features', 'href': '/docs/capabilities.html'}, "
    "{'text': 'Quick Start', 'href': '/docs/getting-started.html'}, "
    "{'text': 'Documentation', 'href': '/#docs'}, "
    "{'text': 'Examples', 'href': '/docs/examples.html'}, "
    "{'text': 'GitHub', 'href': 'https://github.com/lpalbou/abstractcore', 'target': '_blank', 'icon': 'github'}, "
    "{'text': 'PyPI', 'href': 'https://pypi.org/project/abstractcore/', 'target': '_blank', 'icon': 'pypi'}]"
)

ROOT_FILES = [
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "ACKNOWLEDGEMENTS.md",
    "LICENSE",
    "pyproject.toml",
    "llms.txt",
    "llms-full.txt",
]
# Root files that get a rendered page of their own.
ROOT_HTML = {
    "CHANGELOG.md": "docs/changelog.html",
    "SECURITY.md": "docs/security.html",
}
# Files the site serves as they are (linked from llms.txt).
EXTRA_COPIES = [
    "skills/abstractcore/SKILL.md",
    "skills/abstractcore/references/integrations.md",
    "skills/abstractcore/references/operations.md",
    "examples/README.md",
]

DOC_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title} - AbstractCore</title>
    <meta name="description" content="{meta_description}">
    <link rel="canonical" href="https://abstractcore.ai/{page_path}">
    <link rel="icon" type="image/x-icon" href="{root}favicon.ico" sizes="any">
    <link rel="apple-touch-icon" href="{root}apple-touch-icon.png">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{root}assets/css/main.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">

    <!-- Navbar Component -->
    <script src="{root}assets/js/navbar-component.js"></script>
</head>
<body>
    <!-- Navigation -->
    <div class="navbar-placeholder"></div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            createNavbar({{
                basePath: '',
                menuItems: {menu_items}
            }});
        }});
    </script>

    <!-- Main Content -->
    <main style="padding-top: 5rem;">
        <div class="container" style="max-width: 1100px;">
            <div class="doc-header" style="margin-bottom: 3rem;">
                <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">{doc_title}</h1>
                <p style="font-size: 1.25rem; color: var(--text-secondary);">{doc_description}</p>
            </div>

            {toc_block}

            <div class="doc-content">

{doc_body}
            </div>

            <footer class="doc-footer">
                <a href="{root}index.html">AbstractCore</a> {version} &middot;
                <a href="{root}docs/README.html">Documentation index</a> &middot;
                <a href="{root}llms.txt">llms.txt</a> &middot;
                Part of <a href="https://abstractframework.ai/">AbstractFramework</a>
            </footer>
        </div>
    </main>

    <!-- Scripts -->
    {mermaid_scripts}
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <script src="{root}assets/js/main.js"></script>
</body>
</html>
"""


TOC_BLOCK_TEMPLATE = """<div style="background: var(--background-secondary); padding: 2rem; border-radius: 0.75rem; margin-bottom: 3rem;"><h2 style="margin: 0 0 1rem 0;">Table of Contents</h2>{links}</div>"""
TOC_LINK_TEMPLATE = """<a href="#{hid}" style="display: block; padding: 0.4rem 0; color: var(--primary-color);">{label}</a>"""


HREF_RE = re.compile(r'(href|src)="([^"]+)"')
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\r?\n(.*?)\r?\n```", re.DOTALL)


@dataclass(frozen=True)
class RenderedDoc:
    title: str
    description_text: str
    body_html: str
    toc_links_html: str
    has_mermaid: bool


class LinkResolver:
    """Maps a link found in a source page to its address on the site."""

    def __init__(self, rendered: set[str], repo: Path | None, ref: str) -> None:
        self.rendered = rendered  # repo-relative .md paths that get an .html page
        self.repo = repo
        self.ref = ref
        self.missing: list[str] = []
        self._kinds: dict[str, str | None] = {}

    def _kind_at_ref(self, path: str) -> str | None:
        if path not in self._kinds:
            kind = None
            if self.repo is not None:
                proc = subprocess.run(
                    ["git", "-C", str(self.repo), "cat-file", "-t", f"{self.ref}:{path}"],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    kind = proc.stdout.strip()
            self._kinds[path] = kind
        return self._kinds[path]

    def resolve(self, href: str, src_path: str, out_dir: str) -> str:
        if href.startswith(("#", "mailto:", "tel:", "data:")):
            return href
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path:
            return href
        target = posixpath.normpath(posixpath.join(posixpath.dirname(src_path), parts.path))
        if target.startswith("../"):
            return href
        if target in self.rendered:
            new = target[: -len(".md")] + ".html"
        elif target in ROOT_HTML:
            new = ROOT_HTML[target]
        elif target.startswith("docs/assets/"):
            new = target
        else:
            kind = self._kind_at_ref(target)
            if kind is None:
                self.missing.append(f"{src_path}: {href} -> {target} (not at {self.ref})")
                return href
            word = "tree" if kind == "tree" else "blob"
            return urlunsplit(("https", "github.com", f"/lpalbou/AbstractCore/{word}/{self.ref}/{target}", "", parts.fragment))
        rel = posixpath.relpath(new, out_dir)
        return urlunsplit(("", "", rel, parts.query, parts.fragment))


def strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value)


def truncate_meta(text: str, limit: int = 160) -> str:
    text = " ".join(text.split()).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return f"{cut}…"


def extract_title_and_split(markdown_text: str, fallback_title: str) -> tuple[str, str, str]:
    lines = markdown_text.splitlines()
    title = None
    for idx, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            lines = lines[idx + 1 :]
            break
    if not title:
        title = fallback_title

    while lines and not lines[0].strip():
        lines.pop(0)

    # First paragraph becomes the doc header description; it is removed from body,
    # unless it is a block (fence, table, list, heading) rather than prose.
    description_lines: list[str] = []
    if lines and not lines[0].lstrip().startswith(("```", "|", "#", "- ", "* ", ">", "<")):
        while lines and lines[0].strip():
            description_lines.append(lines.pop(0))

    while lines and not lines[0].strip():
        lines.pop(0)

    description_md = "\n".join(description_lines).strip()
    body_md = ("\n".join(lines).rstrip() + "\n") if lines else ""
    return title, description_md, body_md


def rewrite_links(html_text: str, resolver: LinkResolver, src_path: str, out_dir: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        attr, href = match.group(1), match.group(2)
        new = resolver.resolve(html.unescape(href), src_path, out_dir)
        return f'{attr}="{html.escape(new)}"'

    return HREF_RE.sub(_repl, html_text)


def replace_mermaid_fences(markdown_text: str) -> tuple[str, bool]:
    has_mermaid = False

    def _repl(match: re.Match[str]) -> str:
        nonlocal has_mermaid
        has_mermaid = True
        content = match.group(1).rstrip("\r\n")
        return f"\n\n<div class=\"mermaid\">{html.escape(content)}</div>\n\n"

    return MERMAID_FENCE_RE.sub(_repl, markdown_text), has_mermaid


def wrap_code_blocks(html_text: str) -> str:
    html_text = html_text.replace("<pre><code", '<div class="code-block"><pre><code')
    html_text = html_text.replace("</code></pre>", "</code></pre></div>")
    return html_text


def wrap_tables(html_text: str) -> str:
    # Wide tables scroll inside their own box instead of widening the page on a phone.
    html_text = html_text.replace("<table>", '<div class="table-scroll"><table>')
    return html_text.replace("</table>", "</table></div>")


def toc_links_from_tokens(toc_tokens: list[dict]) -> str:
    links: list[str] = []

    def _walk(tokens: list[dict]) -> None:
        for tok in tokens:
            if tok.get("level") == 2:
                hid = tok.get("id") or ""
                label = strip_tags(tok.get("html") or tok.get("name") or "")
                if hid and label:
                    links.append(TOC_LINK_TEMPLATE.format(hid=html.escape(hid), label=html.escape(label)))
            children = tok.get("children") or []
            if children:
                _walk(children)

    _walk(toc_tokens)
    return "\n".join(links)


def render_markdown_doc(markdown_text: str, fallback_title: str, resolver: LinkResolver, src_path: str, out_dir: str) -> RenderedDoc:
    title, description_md, body_md = extract_title_and_split(markdown_text, fallback_title=fallback_title)

    description_text = ""
    if description_md:
        description_html = markdown.markdown(description_md, extensions=["fenced_code", "tables"])
        description_text = " ".join(strip_tags(description_html).split()).strip()

    body_md, has_mermaid = replace_mermaid_fences(body_md)

    md = markdown.Markdown(extensions=["fenced_code", "tables", TocExtension(permalink=False, slugify=github_slugify)])
    body_html = md.convert(body_md)
    body_html = rewrite_links(body_html, resolver, src_path, out_dir)
    body_html = wrap_code_blocks(body_html)
    body_html = wrap_tables(body_html)

    toc_links_html = toc_links_from_tokens(getattr(md, "toc_tokens", []))
    return RenderedDoc(
        title=title,
        description_text=description_text,
        body_html=body_html,
        toc_links_html=toc_links_html,
        has_mermaid=has_mermaid or ('class="mermaid"' in body_html),
    )


def render_html_page(doc: RenderedDoc, out_path: str, version: str) -> str:
    out_dir = posixpath.dirname(out_path)
    root = posixpath.relpath(".", out_dir) + "/"
    toc_block = ""
    if doc.toc_links_html:
        toc_block = TOC_BLOCK_TEMPLATE.format(links=doc.toc_links_html)

    mermaid_scripts = ""
    if doc.has_mermaid:
        mermaid_scripts = (
            '<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>\n'
            '<script>mermaid.initialize({ startOnLoad: true, theme: "dark" });</script>'
        )

    html_text = DOC_TEMPLATE.format(
        page_title=html.escape(doc.title),
        meta_description=html.escape(truncate_meta(doc.description_text)),
        page_path=out_path,
        root=root,
        menu_items=MENU_ITEMS_JS,
        doc_title=html.escape(doc.title),
        doc_description=html.escape(doc.description_text),
        toc_block=toc_block,
        doc_body=doc.body_html.strip() + "\n",
        mermaid_scripts=mermaid_scripts,
        version=html.escape(version),
    )
    return "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n"


def copy_markdown_assets(source_repo: Path, site_root: Path) -> None:
    for rel in ROOT_FILES + EXTRA_COPIES:
        src = source_repo / rel
        if not src.exists():
            raise SystemExit(f"Missing in source: {rel}")
        dst = site_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    src_docs = source_repo / "docs"
    dst_docs = site_root / "docs"
    for md_file in sorted(list(src_docs.glob("*.md")) + list(src_docs.glob("apps/*.md"))):
        out = dst_docs / md_file.relative_to(src_docs)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, out)

    assets_dir = src_docs / "assets"
    if assets_dir.exists():
        for asset in sorted(assets_dir.rglob("*")):
            if asset.is_dir():
                continue
            out = dst_docs / asset.relative_to(src_docs)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset, out)


def read_version(source_repo: Path) -> str:
    text = (source_repo / "llms.txt").read_text(encoding="utf-8")
    match = re.search(r"^Package version: (\S+)", text, re.M)
    return match.group(1) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync AbstractCore user docs into the abstractcore.ai gh-pages site.")
    parser.add_argument("--source", type=Path, default=Path("../abstractcore"), help="Path to the abstractcore sources (a checkout or an export of the release tag)")
    parser.add_argument("--site", type=Path, default=Path("."), help="Path to the gh-pages checkout")
    parser.add_argument("--repo", type=Path, default=None, help="An abstractcore git checkout holding --ref, used to check links to files the site does not mirror")
    parser.add_argument("--ref", default="main", help="Git ref those links point at on GitHub (the release tag)")
    parser.add_argument("--copy-md", action="store_true", help="Copy the markdown docs and root files into the site")
    args = parser.parse_args()

    source_repo = args.source.resolve()
    site_root = args.site.resolve()
    src_docs = source_repo / "docs"
    if not src_docs.exists():
        raise SystemExit(f"Source docs folder not found: {src_docs}")

    if args.copy_md:
        copy_markdown_assets(source_repo=source_repo, site_root=site_root)

    version = read_version(source_repo)
    pages = sorted(list(src_docs.glob("*.md")) + list(src_docs.glob("apps/*.md")))
    rendered = {p.relative_to(source_repo).as_posix() for p in pages}
    resolver = LinkResolver(rendered=rendered, repo=args.repo.resolve() if args.repo else None, ref=args.ref)

    jobs = [(p.relative_to(source_repo).as_posix(), p.relative_to(source_repo).with_suffix(".html").as_posix()) for p in pages]
    jobs += [(src, out) for src, out in ROOT_HTML.items() if (source_repo / src).exists()]

    for src_rel, out_rel in jobs:
        md_path = source_repo / src_rel
        fallback_title = md_path.stem.replace("-", " ").replace("_", " ").title()
        rendered_doc = render_markdown_doc(
            md_path.read_text(encoding="utf-8"),
            fallback_title=fallback_title,
            resolver=resolver,
            src_path=src_rel,
            out_dir=posixpath.dirname(out_rel),
        )
        out_path = site_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_html_page(rendered_doc, out_rel, version), encoding="utf-8")

    print(f"Generated {len(jobs)} pages into {site_root / 'docs'} (AbstractCore {version}, links at {args.ref})")
    print(f"Date: {date.today().isoformat()}")
    if resolver.missing:
        print("Links whose target does not exist:", file=sys.stderr)
        for line in resolver.missing:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
