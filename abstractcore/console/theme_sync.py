"""Sync the AbstractCore web console's themes from the abstractuic ui-kit.

The console (`abstractcore.console.web`) is HTML served from Python with no
npm build, so it cannot import ``@abstractframework/ui-kit`` at runtime. The
kit's ``theme.ts`` (THEME_SPECS) and ``theme.css`` (base ``:root`` tokens and
per-theme token blocks) are therefore GENERATED into ``themes.py`` at sync
time, exactly the way abstractgateway vendors them for its own console
(``abstractgateway/console_theme_sync.py``). A drift test
(``tests/test_console_web.py``) regenerates from the kit source and fails
loud on any divergence, so the copy can never rot silently.

Run: ``python -m abstractcore.console.theme_sync`` from a checkout where the
abstractuic repository sits in an ancestor directory (the framework
monorepo layout), or set ``ABSTRACTUIC_SRC`` to the ``ui-kit/src`` directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GENERATED_MODULE = "themes.py"

_SPEC_RE = re.compile(
    r"\{\s*id:\s*\"(?P<id>[a-z0-9-]+)\",\s*label:\s*\"(?P<label>[^\"]+)\",\s*"
    r"group:\s*\"(?P<group>dark|light)\",\s*swatches:\s*\[(?P<swatches>[^\]]*)\]",
)


def locate_kit_src(start: Optional[Path] = None) -> Optional[Path]:
    """Return the kit's ``ui-kit/src`` directory, or None.

    ``ABSTRACTUIC_SRC`` wins. Otherwise every ancestor of this file (or of
    ``start``) is checked for ``abstractuic/ui-kit/src``: that finds the kit
    from the monorepo checkout and from git worktrees nested below it."""
    env = str(os.getenv("ABSTRACTUIC_SRC") or "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    here = (start if start is not None else Path(__file__)).resolve()
    for parent in here.parents:
        candidate = parent / "abstractuic" / "ui-kit" / "src"
        if (candidate / "theme.css").is_file() and (candidate / "theme.ts").is_file():
            return candidate
    return None


def parse_theme_specs(theme_ts_text: str) -> List[Dict[str, Any]]:
    """THEME_SPECS entries (id/label/group/swatches) from theme.ts.

    Deliberately narrow: an empty result means the kit reshaped the file and
    the caller raises instead of guessing."""
    specs: List[Dict[str, Any]] = []
    for m in _SPEC_RE.finditer(theme_ts_text):
        swatches = [s.strip().strip('"') for s in m.group("swatches").split(",") if s.strip()]
        specs.append({"id": m.group("id"), "label": m.group("label"), "group": m.group("group"), "swatches": swatches})
    return specs


def parse_root_block(theme_css_text: str) -> str:
    """The kit's FIRST top-level ``:root { ... }`` block (the base dark tokens), verbatim."""
    m = re.search(r"(?m)^:root \{\n", theme_css_text)
    if not m:
        raise ValueError("kit theme.css has no top-level ':root {' block; update abstractcore.console.theme_sync")
    end = theme_css_text.find("\n}", m.end())
    if end < 0:
        raise ValueError("unterminated :root block in kit theme.css")
    block = theme_css_text[m.start() : end + 2]
    if "{" in block[m.end() - m.start() :]:
        raise ValueError("kit :root block is not flat CSS; update abstractcore.console.theme_sync")
    return block


def parse_theme_css_blocks(theme_css_text: str) -> List[Tuple[str, str]]:
    """Every block whose selector names ``:root.theme-*``, verbatim, in source order.

    Kit theme blocks are FLAT css; a nested ``{`` means the format changed and
    the sync must be revisited, so that raises."""
    blocks: List[Tuple[str, str]] = []
    idx = 0
    text = theme_css_text
    while True:
        brace = text.find("{", idx)
        if brace < 0:
            break
        sel_start = max(text.rfind("}", 0, brace), text.rfind("*/", 0, brace))
        selector = text[sel_start + 1 if sel_start >= 0 else 0 : brace].strip().lstrip("/").strip()
        end = text.find("\n}", brace)
        if end < 0:
            break
        body = text[brace + 1 : end]
        if ":root.theme-" in selector:
            if "{" in body:
                raise ValueError(f"kit theme block for {selector!r} is not flat CSS; update abstractcore.console.theme_sync")
            blocks.append((selector, f"{selector} {{{body}\n}}"))
        idx = end + 2
    return blocks


def generate_themes_module(kit_src: Path) -> str:
    """Render the generated ``themes.py`` content from the kit source."""
    theme_ts = (kit_src / "theme.ts").read_text(encoding="utf-8")
    theme_css = (kit_src / "theme.css").read_text(encoding="utf-8")

    specs = parse_theme_specs(theme_ts)
    if len(specs) < 10:
        raise ValueError(
            f"parsed only {len(specs)} THEME_SPECS from {kit_src / 'theme.ts'}; "
            "the kit file shape changed; update abstractcore.console.theme_sync"
        )
    root_css = parse_root_block(theme_css)
    blocks = parse_theme_css_blocks(theme_css)
    if not blocks:
        raise ValueError(f"no :root.theme-* blocks parsed from {kit_src / 'theme.css'}")

    block_ids = set()
    for selector, _ in blocks:
        block_ids.update(re.findall(r":root\.theme-([a-z0-9-]+)", selector))
    # "dark" is the kit's :root default: the ONLY spec allowed without a class block.
    missing = [s["id"] for s in specs if s["id"] != "dark" and s["id"] not in block_ids]
    if missing:
        raise ValueError(f"kit theme.css has no blocks for spec ids {missing}; refusing to generate a lying list")

    light_ids = [s["id"] for s in specs if s["group"] == "light"]
    css_text = "\n\n".join(block for _, block in blocks)
    provenance = {
        "theme_ts_sha256": hashlib.sha256(theme_ts.encode("utf-8")).hexdigest(),
        "theme_css_sha256": hashlib.sha256(theme_css.encode("utf-8")).hexdigest(),
    }
    return (
        '"""GENERATED by abstractcore.console.theme_sync -- DO NOT EDIT.\n'
        "\n"
        "Source of truth: abstractuic ui-kit/src/theme.ts (THEME_SPECS) and\n"
        "theme.css (base :root tokens + per-theme token blocks), copied verbatim\n"
        "so the AbstractCore web console offers exactly the framework's themes\n"
        "(same mechanism as abstractgateway's console_theme_sync). Regenerate:\n"
        "    python -m abstractcore.console.theme_sync\n"
        "The drift test in tests/test_console_web.py fails when this is stale.\n"
        '"""\n'
        "\n"
        "# fmt: off\n"
        f"KIT_THEME_SPECS = {json.dumps(specs, ensure_ascii=False, indent=4)}\n"
        "\n"
        f"KIT_LIGHT_THEME_IDS = {json.dumps(light_ids, ensure_ascii=False)}\n"
        "\n"
        f"KIT_SOURCE_PROVENANCE = {json.dumps(provenance, ensure_ascii=False, indent=4)}\n"
        "\n"
        f"KIT_ROOT_CSS = {root_css!r}\n"
        "\n"
        f"KIT_THEME_CSS = {css_text!r}\n"
        "# fmt: on\n"
    )


def sync(target_dir: Optional[Path] = None) -> Path:
    """Write the generated module beside this one and return its path."""
    kit_src = locate_kit_src()
    if kit_src is None:
        raise FileNotFoundError(
            "abstractuic kit source not found; set ABSTRACTUIC_SRC to the ui-kit/src "
            "directory or run from the framework monorepo checkout"
        )
    out_dir = target_dir if target_dir is not None else Path(__file__).resolve().parent
    out_path = out_dir / GENERATED_MODULE
    out_path.write_text(generate_themes_module(kit_src), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print(f"wrote {sync()}")
