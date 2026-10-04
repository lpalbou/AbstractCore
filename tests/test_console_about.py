"""AbstractCore console About (round 5): the kit's compact About card
(ui-kit 0.7.0 `AfAbout`), rendered by the server with
`abstractcore.utils.identity.about_card_html` and styled by the vendored kit
``af-about`` CSS. Content rule: name + version, framework + gateway versions,
six links, ONE licence line, NO package list."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import pytest

from abstractcore.console import theme_sync, web
from abstractcore.console.themes import KIT_ABOUT_CSS
from abstractcore.utils.identity import about_card_html, about_version_facts, app_identity, framework_identity


def _page() -> str:
    return web.render_console_html()


def test_page_has_an_about_button_and_a_modal_dialog() -> None:
    html = _page()
    assert 'id="acc-about-open"' in html and 'aria-label="About AbstractCore"' in html and 'aria-haspopup="dialog"' in html
    m = re.search(r'<dialog id="acc-about" class="af-appearance af-about acc-about" aria-labelledby="acc-about-title">(.*?)</dialog>', html, re.S)
    assert m, "no About dialog in the console page"
    card = m.group(1)
    assert 'id="acc-about-title">AbstractCore <span class="af-about-card__version">' in card
    assert 'data-action="close-about"' in card
    assert "aboutDialog.showModal()" in html


def test_about_card_content_rule() -> None:
    card = web.console_about_card_html()
    ident = app_identity("abstractcore", "1")
    fw = framework_identity()
    assert card.count("<dt>") == 2 and "<dt>AbstractFramework</dt>" in card and "<dt>AbstractGateway</dt>" in card
    order = re.findall(r'data-link="([a-z]+)"', card)
    assert order == ["website", "source", "docs", "issues", "feedback", "contact"]
    for href in (ident.website, ident.repo, ident.docs, ident.issues, ident.feedback):
        assert f'href="{href}"' in card
    assert card.count('href="mailto:') == 1 and f'href="mailto:{fw.contact_email}"' in card
    assert f'<p class="af-about-card__legal">{fw.copyright}</p>' in card
    # NO package list: none of the other framework packages' names or rows.
    for pkg in ("abstractruntime", "abstractvoice", "abstractagent", "Gateway package"):
        assert pkg not in card, pkg


def test_versions_are_the_installed_ones_or_said_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    v = web.console_about_versions()
    assert v["core"] == metadata.version("abstractcore")

    def fake_version(dist: str) -> str:
        if dist == "abstractcore":
            return "9.9.9"
        raise metadata.PackageNotFoundError(dist)

    import abstractcore.utils.identity as identity

    monkeypatch.setattr(identity, "installed_version", fake_version)
    v = web.console_about_versions()
    assert v == {"core": "9.9.9", "framework": None, "framework_note": "not installed on this host", "gateway": None, "gateway_note": "not installed on this host"}
    card = web.console_about_card_html()
    assert card.count("<dd>not installed on this host</dd>") == 2


def test_version_facts_defaults_match_the_kit() -> None:
    assert about_version_facts(None, None) == [("AbstractFramework", "not reported"), ("AbstractGateway", "not connected")]
    assert about_version_facts("0.9.6", "0.12.0") == [("AbstractFramework", "0.9.6"), ("AbstractGateway", "0.12.0")]


def test_about_css_is_vendored_from_the_kit() -> None:
    assert KIT_ABOUT_CSS.startswith("/* af-about:begin") and KIT_ABOUT_CSS.rstrip().endswith("/* af-about:end */")
    assert KIT_ABOUT_CSS in _page()


def test_card_markup_equals_the_kit_component() -> None:
    """Parity with the kit's `AfAbout` (renderToStaticMarkup over the built
    kit dist): same structure, classes, links and text."""
    kit_src = theme_sync.locate_kit_src()
    if kit_src is None:
        pytest.skip("abstractuic kit source not present (non-monorepo checkout)")
    dist = kit_src.parent / "dist" / "index.js"
    node = shutil.which("node")
    if node is None or not dist.is_file():
        pytest.skip("node or the built kit dist is missing")
    react_dir = kit_src.parent.parent / "node_modules"
    script = f"""
import {{ createRequire }} from "node:module";
const require = createRequire({json.dumps(str(react_dir) + "/")});
const React = require("react"); const {{ renderToStaticMarkup }} = require("react-dom/server");
const kit = await import({json.dumps(dist.as_uri())});
const html = renderToStaticMarkup(React.createElement(kit.AfAbout, {{ identity: kit.appIdentity("abstractcore", "2.0.0"), versions: {{ framework: "0.9.6", gateway: null, gatewayNote: "not installed on this host" }} }}));
console.log(html);
"""
    out = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    ours = about_card_html(app_identity("abstractcore", "2.0.0"), "0.9.6", None, gateway_note="not installed on this host")
    assert ours == out.stdout.strip()
