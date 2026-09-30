"""Core web console Email tab (DESIGN 2026-09-30 §1-§3, §6, §12): markup rules and a
Playwright run against a served console.

The browser tests start the real server app (with the console's stubbed contract routes, as
the smoke test does) on a free port >= 18120 under a scratch HOME, and serve `/acore/email*`
from fake account records through `page.route`: no mail server, no provider key, no network.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from abstractcore.console import theme_sync
from abstractcore.console.themes import KIT_FORM_CSS, KIT_SWITCH_CSS
from abstractcore.console.web import _EMAIL_HTML, render_console_html

pytestmark = pytest.mark.basic

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


# ------------------------------------------------------------------ markup rules (no browser)


def test_email_tab_has_the_design_order_and_no_save_per_section() -> None:
    html = _EMAIL_HTML
    order = [html.index(f'data-acc="{name}"') for name in ("email-identity-card", "email-mailbox-card", "email-agent-card", "email-advanced")]
    assert order == sorted(order)
    # One inline Save (the Email address field); everything else applies at once.
    assert re.findall(r">Save<", html) == [">Save<"]
    for gone in ("Save and test", "Save policy", "Save limits", "Save notifications", "Turn on", "Turn off", "Registered address", "optional", "example.com\"", "smtp.example.com", "imap.example.com"):
        assert gone not in html, gone
    assert 'placeholder=' not in html.split('data-acc="email-advanced"')[0]  # no placeholders in the forms
    # Tabs Google / Microsoft / Other, ONE Connect, the connected state's Test + Disconnect.
    assert [m for m in re.findall(r'role="tab"[^>]*>([^<]+)<', html)] == ["Google", "Microsoft", "Other"]
    assert html.count('data-acc-action="email-connect"') == 1
    assert "Disconnect this mailbox? Your agents lose email until you connect again. Policy and limits are kept." in html
    # The CLI line never shows a password on the command line.
    assert "--password-stdin" in html and "--password &lt;" not in html and "--password <" not in html


def test_email_tab_switches_use_the_kit_markup_labelled_by_the_feature() -> None:
    for label, action in (("Agent email tools", "agent-tools-switch"), ("Use this mailbox", "mailbox-switch")):
        m = re.search(r'<button type="button" role="switch" class="af-switch af-switch--row" aria-checked="false" data-acc-action="' + action + r'"[^>]*>(.*?)</button>', _EMAIL_HTML, re.S)
        assert m, action
        assert f'<span class="af-switch__label">{label}</span>' in m.group(1)


def test_kit_form_and_switch_css_are_generated_and_served() -> None:
    assert KIT_FORM_CSS.startswith("/* af-form:begin") and KIT_FORM_CSS.rstrip().endswith("/* af-form:end */")
    assert KIT_SWITCH_CSS.startswith("/* af-switch:begin")
    page = render_console_html()
    assert KIT_FORM_CSS in page and KIT_SWITCH_CSS in page


def test_theme_sync_refuses_a_kit_without_the_form_block() -> None:
    kit_src = theme_sync.locate_kit_src()
    if kit_src is None:
        pytest.skip("abstractuic kit source not present (non-monorepo checkout)")
    css = (kit_src / "theme.css").read_text(encoding="utf-8")
    assert theme_sync.parse_form_css(css) == KIT_FORM_CSS
    stripped = css.replace("/* af-form:end */", "/* gone */")
    with pytest.raises(ValueError, match="af-form"):
        theme_sync.parse_form_css(stripped)
    with pytest.raises(ValueError, match="af-tabs__tab"):
        theme_sync.parse_form_css(css.replace(".af-tabs__tab", ".af-tabz"))


def test_console_sources_pass_the_kits_verb_toggle_guard() -> None:
    """The kit's `findVerbToggleLabels` over the console source (ui-kit dist, when built)."""
    kit_src = theme_sync.locate_kit_src()
    node = shutil.which("node")
    lint = kit_src.parent / "dist" / "state_toggle_lint.js" if kit_src else None
    if not (node and lint and lint.is_file()):
        pytest.skip("needs node and a built abstractuic ui-kit (dist/state_toggle_lint.js)")
    web = REPO_ROOT / "abstractcore" / "console" / "web.py"
    script = (
        f"import {{ findVerbToggleLabels }} from {json.dumps(lint.as_uri())};"
        f"import {{ readFileSync }} from 'node:fs';"
        f"const hits = findVerbToggleLabels(readFileSync({json.dumps(str(web))}, 'utf8'));"
        "console.log(JSON.stringify(hits));"
    )
    out = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []
    # The guard is live: a verb pair fails it.
    bad = subprocess.run(
        [node, "--input-type=module", "-e", f"import {{ findVerbToggleLabels }} from {json.dumps(lint.as_uri())};"
         "console.log(findVerbToggleLabels('label = on ? \"Turn off\" : \"Turn on\"').length)"],
        capture_output=True, text=True, check=True,
    )
    assert int(bad.stdout.strip()) == 1


# ------------------------------------------------------------------ the browser run

_SERVER = """
import sys, uvicorn
sys.path.insert(0, {tests_dir!r})
from abstractcore.server.app import app
from console_web_fixtures import build_stub_router
app.include_router(build_stub_router())
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="error")
"""

TOKEN = "email-tab-e2e"
LIMITS = {"per_hour": 20, "per_day": 100, "used_last_hour": 0, "used_last_day": 3}
EMPTY = {
    "schema": "email_settings_v1", "configured": False, "enabled": False,
    "agent_tools": {"enabled": False, "active": False, "reason": "no mailbox is connected"},
    "address": "", "display_name": "", "username": "", "auth_kind": "", "imap": None, "smtp": None, "oauth": None,
    "secret_storage": "", "secret_warning": "", "policy": {"mode": "allowlist", "entries": [], "default": True},
    "limits": LIMITS, "registered_address": "me@fastmail.com", "registered_address_stored": "me@fastmail.com",
    "status": {"last_test": "", "last_ok": "", "last_error": None}, "notices": [],
    "oauth_providers": [{"id": "google", "available": True, "reason": None}, {"id": "microsoft", "available": False, "reason": "No built-in Microsoft sign-in client in this version."}],
}
CONNECTED = {
    **EMPTY, "configured": True, "enabled": True, "address": "me@fastmail.com", "username": "me@fastmail.com", "auth_kind": "password",
    "secret_storage": "key-file", "agent_tools": {"enabled": False, "active": False, "reason": "off (your choice; default)"},
    "imap": {"host": "imap.fastmail.com", "port": 993, "security": "ssl", "folder": "INBOX", "ca_file": ""},
    "smtp": {"host": "smtp.fastmail.com", "port": 465, "security": "ssl", "ca_file": ""},
    "policy": {"mode": "allowlist", "entries": ["me@fastmail.com"], "default": False},
    "status": {"last_test": "", "last_ok": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(), "last_error": None},
}
FOUND = {"address": "me@fastmail.com", "domain": "fastmail.com", "found": True, "source": "known", "provider": None,
         "imap": {"host": "imap.fastmail.com", "port": 993, "security": "ssl"}, "smtp": {"host": "smtp.fastmail.com", "port": 465, "security": "ssl"},
         "username": "me@fastmail.com", "tried": []}
NOT_FOUND = {**FOUND, "address": "me@small-isp.net", "domain": "small-isp.net", "found": False, "source": None, "imap": None, "smtp": None}


def _free_port() -> int:
    for _ in range(200):
        port = random.randint(18120, 18999)
        if port in (18793, 18794):
            continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port >= 18120")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    pw = pytest.importorskip("playwright.sync_api")
    home = tmp_path_factory.mktemp("email-tab-home")
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not (k.startswith("ABSTRACT") or k.endswith(("_KEY", "_TOKEN")) or "PASSWORD" in k)}
    env.update(
        HOME=str(home), ABSTRACTCORE_AUTH_TOKEN=TOKEN, ABSTRACTCORE_SERVER_DISABLE_CENTRALIZED_CONFIG="1",
        ABSTRACTFRAMEWORK_DATA_REGISTRY=str(home / "registry.json"), HF_HUB_OFFLINE="1",
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]),
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER.format(tests_dir=str(TESTS_DIR), port=port)],
        env=env, cwd=str(home), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as res:
                    if res.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                pytest.fail(f"server did not come up on {base}: {(proc.stdout.read() if proc.poll() is not None else '')[-2000:]}")
            time.sleep(0.3)
        with pw.sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # the browser binary is not installed
                pytest.skip(f"playwright chromium unavailable: {exc}")
            yield base, browser
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


class FakeEmail:
    """`/acore/email*` from fake records; every request is recorded."""

    def __init__(self, doc, discover=FOUND, fail=None):
        self.doc = json.loads(json.dumps(doc))
        if self.doc.get("configured"):  # "checked 2 min ago", relative to now
            self.doc["status"]["last_ok"] = (datetime.now(timezone.utc) - timedelta(minutes=2, seconds=5)).isoformat()
        self.discover = discover
        self.fail = fail or {}  # "METHOD /path" -> (status, error body)
        self.calls = []

    def __call__(self, route):
        req = route.request
        path = "/" + req.url.split("/", 3)[3].split("?")[0]
        body = json.loads(req.post_data) if req.post_data else None
        self.calls.append((req.method, path, body))
        key = f"{req.method} {path}"
        if key in self.fail:
            status, err = self.fail[key]
            return route.fulfill(status=status, content_type="application/json", body=json.dumps({"ok": False, "error": err}))
        if path.endswith("/discover"):
            return route.fulfill(status=200, content_type="application/json", body=json.dumps(self.discover))
        if key == "PUT /acore/email/agent-tools":
            self.doc["agent_tools"] = {**self.doc["agent_tools"], "enabled": body["enabled"]}
        elif key == "PUT /acore/email/folder":
            self.doc["imap"]["folder"] = body["folder"] or "INBOX"
        elif key == "PUT /acore/email/registered-address":
            self.doc["registered_address_stored"] = body["address"]
        elif key == "PUT /acore/email/limits":
            self.doc["limits"] = {**self.doc["limits"], "per_hour": body["per_hour"], "per_day": body["per_day"]}
        return route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok": True, **self.doc}))


def _open(served, fake, width=1440, height=900, touch=False):
    base, browser = served
    ctx = browser.new_context(viewport={"width": width, "height": height}, has_touch=touch)
    page = ctx.new_page()
    page.route("**/acore/email**", fake)
    page.goto(f"{base}/console", wait_until="domcontentloaded")
    page.evaluate(f"sessionStorage.setItem('abstractcore_console_token', '{TOKEN}')")
    page.goto(f"{base}/console", wait_until="domcontentloaded")
    page.click("#acc-tab-button-email")
    page.wait_for_selector("#acc-tab-email:not([hidden])")
    page.wait_for_function("document.querySelector('[data-acc=\"email-identity-card\"] input').value !== ''")
    return ctx, page


# The kit's checkLabelScale rule (ui-kit label_scale.ts): labels, switch labels and field
# captions compute to <= 15 px and weight <= 600.
LABEL_SCALE_JS = """(root) => {
  const sel = 'label, .af-switch__label, .af-form__label, .af-field-caption, [data-af-caption]';
  const w = (v) => v === 'bold' || v === 'bolder' ? 700 : (v === 'normal' || !v ? 400 : parseFloat(v));
  return Array.from(root.querySelectorAll(sel)).filter((el) => el.getClientRects().length).map((el) => {
    const cs = getComputedStyle(el);
    return { text: el.textContent.trim().slice(0, 40), size: parseFloat(cs.fontSize), weight: w(cs.fontWeight) };
  }).filter((h) => h.size > 15.01 || h.weight > 600);
}"""


def test_type_scale_and_layout_at_desktop_and_phone(served) -> None:
    fake = FakeEmail(EMPTY, discover=NOT_FOUND)
    fake.doc["registered_address_stored"] = "me@small-isp.net"
    ctx, page = _open(served, fake)
    try:
        page.click("#acc-email-tab-other")
        page.wait_for_selector('[data-acc="email-servers"][open]')  # discovery failed -> Server settings open by themselves
        page.evaluate("document.querySelector('[data-acc=\"email-advanced\"]').open = true")
        root = page.locator("#acc-email")
        assert page.evaluate(LABEL_SCALE_JS, root.element_handle()) == []
        assert root.bounding_box()["width"] <= 720.5
        # Labels above their fields; port + security side by side on desktop.
        geo = page.evaluate("""() => {
          const r = (s) => document.querySelector(s).getBoundingClientRect();
          return { label: r('label[for="acc-email-imap-port"]'), port: r('#acc-email-imap-port'), sec: r('#acc-email-imap-security'),
                   card: r('[data-acc="email-mailbox-card"]'), widest: Math.max(...Array.from(document.querySelectorAll('#acc-email input, #acc-email select')).filter((e) => e.getClientRects().length).map((e) => e.getBoundingClientRect().right)) };
        }""")
        assert geo["label"]["bottom"] <= geo["port"]["top"] + 0.5
        assert abs(geo["port"]["top"] - geo["sec"]["top"]) < 1 and geo["sec"]["left"] > geo["port"]["right"]
        assert geo["widest"] <= geo["card"]["right"] + 0.5
        assert "Couldn't find the mail servers for small-isp.net" in page.inner_text('[data-acc="email-servers-reason"]')
        # Phone: one column, the page gutter <= 16 px, flat sections (no card border).
        page.set_viewport_size({"width": 390, "height": 844})
        geo = page.evaluate("""() => {
          const r = (s) => document.querySelector(s).getBoundingClientRect();
          const card = document.querySelector('[data-acc="email-mailbox-card"]');
          return { port: r('#acc-email-imap-port'), sec: r('#acc-email-imap-security'), left: r('#acc-email').left,
                   border: getComputedStyle(card).borderLeftWidth, docW: document.documentElement.scrollWidth };
        }""")
        assert geo["sec"]["top"] > geo["port"]["bottom"]
        assert geo["left"] <= 16.5 and geo["border"] == "0px" and geo["docW"] <= 390
        assert page.evaluate(LABEL_SCALE_JS, root.element_handle()) == []
    finally:
        ctx.close()


def test_not_connected_other_tab_discovers_and_connect_names_the_failing_step(served) -> None:
    fake = FakeEmail(EMPTY, fail={"PUT /acore/email": (422, {"code": "email_auth_failed", "cause": "The IMAP server rejected the user name or password.", "fix": "Use an app password.", "details": {"protocol": "imap", "host": "imap.fastmail.com", "port": 993}})})
    ctx, page = _open(served, fake)
    try:
        # Default tab: servers found for a provider other than Google/Microsoft -> Other.
        page.wait_for_selector('#acc-email-tab-other[aria-selected="true"]')
        page.wait_for_function("document.querySelector('[data-acc=\"email-servers-summary\"]').textContent.includes('993')")
        assert page.text_content('[data-acc="email-servers-summary"]') == "imap.fastmail.com · 993 · SSL  ·  smtp.fastmail.com · 465 · SSL"
        assert page.input_value("#acc-email-address") == "me@fastmail.com"  # prefilled from the Email address
        assert page.get_attribute('[data-acc="email-servers"]', "open") is None  # folded when discovery works
        page.fill("#acc-email-password", "app-password")
        page.click('[data-acc-action="email-connect"]')
        err = page.locator('[data-acc="email-connect-error"]')
        err.wait_for(state="visible")
        assert err.inner_text().startswith("Sign-in refused by imap.fastmail.com — check the password")
        method, path, body = [c for c in fake.calls if c[0] == "PUT" and c[1] == "/acore/email"][-1]
        assert body["address"] == "me@fastmail.com" and body["imap"]["host"] == "imap.fastmail.com" and body["test"] is True
        # The agent-tools switch is unavailable with the reason while no mailbox is connected.
        sw = page.locator('[data-acc="email-agent-tools"]')
        assert sw.get_attribute("aria-disabled") == "true"
        assert page.inner_text("#acc-email-agent-tools-reason") == "Connect a mailbox first."
        sw.click(force=True)  # aria-disabled: focusable, but a click changes nothing
        assert not [c for c in fake.calls if c[1] == "/acore/email/agent-tools"]
        # Microsoft has no built-in client here: its button is unavailable with the reason.
        page.click("#acc-email-tab-microsoft")
        assert page.get_attribute('[data-acc="email-oauth-start"]', "aria-disabled") == "true"
        assert "No built-in Microsoft sign-in client" in page.inner_text('[data-acc="email-oauth-reason"]')
    finally:
        ctx.close()


def test_connected_state_switches_apply_at_once_and_revert_on_failure(served) -> None:
    fake = FakeEmail(CONNECTED)
    ctx, page = _open(served, fake)
    try:
        status = page.locator('[data-acc="email-status"]')
        status.wait_for(state="visible")
        assert status.inner_text() == "Connected as me@fastmail.com · Password · checked 2 min ago", status.inner_text()
        assert page.locator('[data-acc="email-setup"]').is_hidden()
        visible_buttons = page.eval_on_selector_all('[data-acc="email-mailbox-card"] button', "bs => bs.filter((b) => b.getClientRects().length).map((b) => b.textContent.trim())")
        assert visible_buttons == ["Test", "Disconnect"]
        sw = page.locator('[data-acc="email-agent-tools"]')
        assert sw.get_attribute("aria-checked") == "false" and sw.get_attribute("aria-disabled") is None
        sw.click()
        page.wait_for_selector('[data-acc="email-agent-tools"][aria-checked="true"]:not([aria-busy])')
        assert ("PUT", "/acore/email/agent-tools", {"enabled": True}) in fake.calls
        assert page.inner_text('#acc-email [data-acc="message"]') == "Agent email tools are on."
        fake.fail["PUT /acore/email/agent-tools"] = (500, {"code": "email_error", "cause": "disk full", "fix": ""})
        sw.click()
        page.wait_for_function("document.querySelector('#acc-email [data-acc=\"message\"]').textContent.includes('stays as it was')")
        assert sw.get_attribute("aria-checked") == "true"  # reverted
        # Advanced: limits and folder save on change, "Saved" inline; no Save buttons.
        page.evaluate("document.querySelector('[data-acc=\"email-advanced\"]').open = true")
        page.fill("#acc-email-per-hour", "5")
        page.press("#acc-email-per-hour", "Tab")
        page.wait_for_function("document.querySelector('[data-acc=\"email-limits-saved\"]').textContent === 'Saved'")
        assert ("PUT", "/acore/email/limits", {"per_hour": 5, "per_day": 100}) in fake.calls
        page.fill("#acc-email-folder", "Archive")
        page.press("#acc-email-folder", "Tab")
        page.wait_for_function("document.querySelector('[data-acc=\"email-folder-saved\"]').textContent === 'Saved'")
        assert ("PUT", "/acore/email/folder", {"folder": "Archive"}) in fake.calls
        # Email address: the one inline Save, which turns into "Saved".
        page.fill("#acc-email-registered", "me@work.example.org")
        page.click('[data-acc-action="identity-save"]')
        page.wait_for_function("document.querySelector('[data-acc=\"email-identity-save\"]').textContent === 'Saved'")
        assert ("PUT", "/acore/email/registered-address", {"address": "me@work.example.org"}) in fake.calls
        # Disconnect asks inline first.
        page.click('[data-acc-action="email-disconnect"]')
        assert page.locator('[data-acc="email-disconnect-confirm"]').is_visible()
        assert not [c for c in fake.calls if c[0] == "DELETE"]
    finally:
        ctx.close()


def test_list_panels_collapse_and_remember(served) -> None:
    base, browser = served
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True)
    page = ctx.new_page()
    try:
        page.goto(f"{base}/console", wait_until="domcontentloaded")
        page.evaluate(f"sessionStorage.setItem('abstractcore_console_token', '{TOKEN}')")
        page.goto(f"{base}/console", wait_until="domcontentloaded")
        page.click("#acc-tab-button-catalog")
        btn = page.locator('[data-acc-section="catalog"]')
        assert btn.get_attribute("aria-expanded") == "true"
        assert btn.bounding_box()["height"] >= 44
        btn.click()
        assert btn.get_attribute("aria-expanded") == "false"
        assert page.locator('[data-acc="catalog-table"]').is_hidden()
        page.reload(wait_until="domcontentloaded")
        page.click("#acc-tab-button-catalog")
        assert page.get_attribute('[data-acc-section="catalog"]', "aria-expanded") == "false"
        assert page.locator('[data-acc="catalog-table"]').is_hidden()
    finally:
        ctx.close()
