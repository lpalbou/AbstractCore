#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import markdown
from markdown.extensions.toc import TocExtension


MENU_ITEMS_JS = (
    "[{'text': 'Features', 'href': '/docs/capabilities.html'}, "
    "{'text': 'Quick Start', 'href': '/docs/getting-started.html'}, "
    "{'text': 'Documentation', 'href': '/#docs'}, "
    "{'text': 'Examples', 'href': '/docs/examples.html'}, "
    "{'text': 'GitHub', 'href': 'https://github.com/lpalbou/abstractcore', 'target': '_blank', 'icon': 'github'}, "
    "{'text': 'PyPI', 'href': 'https://pypi.org/project/abstractcore/', 'target': '_blank', 'icon': 'pypi'}]"
)


DOC_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title} - AbstractCore</title>
    <meta name="description" content="{meta_description}">
    <link rel="icon" type="image/svg+xml" href="../assets/logo.svg">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="../assets/css/main.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">

    <!-- Navbar Component -->
    <script src="../assets/js/navbar-component.js"></script>
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
        </div>
    </main>

    <!-- Scripts -->
    {mermaid_scripts}
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <script src="../assets/js/main.js"></script>
</body>
</html>
"""


TOC_BLOCK_TEMPLATE = """<div style="background: var(--background-secondary); padding: 2rem; border-radius: 0.75rem; margin-bottom: 3rem;"><h2 style="margin: 0 0 1rem 0;">Table of Contents</h2>{links}</div>"""
TOC_LINK_TEMPLATE = """<a href="#{hid}" style="display: block; padding: 0.4rem 0; color: var(--primary-color);">{label}</a>"""


HREF_RE = re.compile(r'href="([^"]+)"')
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\r?\n(.*?)\r?\n```", re.DOTALL)


@dataclass(frozen=True)
class RenderedDoc:
    title: str
    description_text: str
    body_html: str
    toc_links_html: str
    has_mermaid: bool


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

    # Remove leading blank lines after title.
    while lines and not lines[0].strip():
        lines.pop(0)

    # First paragraph becomes the doc header description; it is removed from body.
    description_lines: list[str] = []
    while lines and lines[0].strip():
        description_lines.append(lines.pop(0))

    # Remove blank lines after description.
    while lines and not lines[0].strip():
        lines.pop(0)

    description_md = "\n".join(description_lines).strip()
    body_md = ("\n".join(lines).rstrip() + "\n") if lines else ""
    return title, description_md, body_md


def rewrite_href(href: str) -> str:
    if href.startswith(("#", "mailto:", "tel:")):
        return href
    if href.startswith(("http://", "https://")):
        return href

    parts = urlsplit(href)
    path = parts.path
    if path.startswith("../"):
        return href

    # Only rewrite same-folder markdown links (docs/*.md -> docs/*.html).
    if path.endswith(".md") and ("/" not in path or path.startswith("./")):
        path = path[: -len(".md")] + ".html"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    return href


def rewrite_links(html_text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        href = match.group(1)
        return f'href="{html.escape(rewrite_href(html.unescape(href)))}"'

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
    # Prism expects language-* on <code>; add the wrapper used by the site for copy button styling.
    html_text = html_text.replace("<pre><code", '<div class="code-block"><pre><code')
    html_text = html_text.replace("</code></pre>", "</code></pre></div>")
    return html_text


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


def render_markdown_doc(markdown_text: str, fallback_title: str) -> RenderedDoc:
    title, description_md, body_md = extract_title_and_split(markdown_text, fallback_title=fallback_title)

    description_text = ""
    if description_md:
        description_html = markdown.markdown(description_md, extensions=["fenced_code", "tables"])
        description_text = " ".join(strip_tags(description_html).split()).strip()

    body_md, has_mermaid = replace_mermaid_fences(body_md)

    md = markdown.Markdown(extensions=["fenced_code", "tables", TocExtension(permalink=False)])
    body_html = md.convert(body_md)
    body_html = rewrite_links(body_html)
    body_html = wrap_code_blocks(body_html)

    toc_links_html = toc_links_from_tokens(getattr(md, "toc_tokens", []))
    return RenderedDoc(
        title=title,
        description_text=description_text,
        body_html=body_html,
        toc_links_html=toc_links_html,
        has_mermaid=has_mermaid or ('class="mermaid"' in body_html),
    )


def render_html_page(doc: RenderedDoc) -> str:
    toc_block = ""
    if doc.toc_links_html:
        toc_block = TOC_BLOCK_TEMPLATE.format(links=doc.toc_links_html)

    mermaid_scripts = ""
    if doc.has_mermaid:
        mermaid_scripts = (
            '<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>\n'
            '<script>mermaid.initialize({ startOnLoad: true, theme: "dark" });</script>'
        )

    doc_description = html.escape(doc.description_text)
    meta_description = html.escape(truncate_meta(doc.description_text))

    return DOC_TEMPLATE.format(
        page_title=html.escape(doc.title),
        meta_description=meta_description,
        menu_items=MENU_ITEMS_JS,
        doc_title=html.escape(doc.title),
        doc_description=doc_description,
        toc_block=toc_block,
        doc_body=doc.body_html.strip() + "\n",
        mermaid_scripts=mermaid_scripts,
    )


def copy_markdown_assets(source_repo: Path, site_root: Path) -> None:
    # Copy a small set of repo-root files referenced by llms.txt.
    root_files = [
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "ACKNOWLEDGEMENTS.md",
        "LICENSE",
        "pyproject.toml",
    ]
    for rel in root_files:
        src = source_repo / rel
        dst = site_root / rel
        if src.exists():
            shutil.copy2(src, dst)

    # Copy docs markdown (agents prefer markdown; llms.txt references these paths).
    src_docs = source_repo / "docs"
    dst_docs = site_root / "docs"
    dst_docs.mkdir(parents=True, exist_ok=True)

    # 1) Markdown files (including nested docs/apps, docs/reports, docs/archive, …).
    for md_file in sorted(src_docs.rglob("*.md")):
        rel = md_file.relative_to(src_docs)
        out = dst_docs / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, out)

    # 2) Doc assets referenced by markdown (images/plots/CSVs under docs/assets).
    assets_dir = src_docs / "assets"
    if assets_dir.exists():
        for asset in sorted(assets_dir.rglob("*")):
            if asset.is_dir():
                continue
            rel = asset.relative_to(src_docs)
            out = dst_docs / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset, out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync AbstractCore markdown docs into AbstractCore gh-pages HTML.")
    parser.add_argument("--source", type=Path, default=Path("../abstractcore"), help="Path to the abstractcore repo")
    parser.add_argument("--site", type=Path, default=Path("."), help="Path to the gh-pages repo root")
    parser.add_argument("--copy-md", action="store_true", help="Copy markdown docs and key root files into the site")
    args = parser.parse_args()

    source_repo = args.source.resolve()
    site_root = args.site.resolve()

    src_docs = source_repo / "docs"
    out_docs = site_root / "docs"
    if not src_docs.exists():
        raise SystemExit(f"Source docs folder not found: {src_docs}")
    out_docs.mkdir(parents=True, exist_ok=True)

    if args.copy_md:
        copy_markdown_assets(source_repo=source_repo, site_root=site_root)

    generated = 0
    for md_path in sorted(src_docs.glob("*.md")):
        fallback_title = md_path.stem.replace("-", " ").replace("_", " ").title()
        rendered = render_markdown_doc(md_path.read_text(encoding="utf-8"), fallback_title=fallback_title)
        html_out = render_html_page(rendered)
        out_path = out_docs / f"{md_path.stem}.html"
        out_path.write_text(html_out, encoding="utf-8")
        generated += 1

    print(f"Generated {generated} docs pages into {out_docs}")
    print(f"Date: {date.today().isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
