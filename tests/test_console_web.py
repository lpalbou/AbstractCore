"""AbstractCore web console (/console): page, fragments, theme drift, routes.

The console only renders contract payloads; the backend routes are stubbed
with `console_web_fixtures.build_stub_router()` (same shapes as the shared
contracts), so these tests do not depend on the backend workstream.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from abstractcore.console import theme_sync
from abstractcore.console.themes import KIT_LIGHT_THEME_IDS, KIT_RESPONSIVE_CSS, KIT_ROOT_CSS, KIT_THEME_CSS, KIT_THEME_SPECS
from abstractcore.console.web import FRAGMENT_KINDS, fragment, render_console_html
from abstractcore.server.console_routes import router as console_router

sys.path.insert(0, str(Path(__file__).parent))
from console_web_fixtures import CATALOG, build_stub_router  # noqa: E402

pytestmark = pytest.mark.basic

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def _scripts(html: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html, flags=re.S)


def _node_check(source: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for JavaScript syntax checking")
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        result = subprocess.run([node, "--check", path], capture_output=True, text=True, check=False)
    finally:
        os.unlink(path)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------- page HTML


def test_page_has_the_four_tabs_with_contract_ids() -> None:
    html = render_console_html()
    for tab, label in (("overview", "Overview"), ("catalog", "Models"), ("engines", "Engines"), ("providers", "Providers")):
        assert f'id="acc-tab-button-{tab}"' in html
        assert f'id="acc-tab-{tab}"' in html
        assert f">{label}</button>" in html
    assert "<title>AbstractCore Console</title>" in html
    assert '"apiBase": "/acore"' in html


def test_page_title_and_api_base_are_escaped_and_configurable() -> None:
    html = render_console_html(api_base="/core/acore/", title="Box <1>")
    assert "<title>Box &lt;1&gt;</title>" in html
    assert '"apiBase": "/core/acore"' in html
    assert "Box <1>" not in html.split("<script>")[0]


def test_models_fragment_has_search_filters_tables_and_jobs() -> None:
    html = fragment("models")["html"]
    for role in ("q", "engine", "modality", "fits", "installed-only", "hub", "catalog-body", "installed-body", "jobs", "view-cli", "host-line"):
        assert f'data-acc="{role}"' in html, role
    assert "Fits this machine" in html and "Installed only" in html
    for header in ("Provider", "Artifact", "Quant", "Download size", "Weights", "Fit", "Actions"):
        assert f"<th>{header}</th>" in html
    # "Fits this machine" is pre-selected (fewest clicks, principle 3).
    assert 'data-acc="fits" checked' in html


def test_engines_fragment_has_status_table_and_refresh() -> None:
    html = fragment("engines")["html"]
    for header in ("Engine", "Supported", "Installed", "Version", "Running", "Reachable", "Base URL", "Models", "Actions"):
        assert f"<th>{header}</th>" in html
    assert 'data-acc-action="refresh"' in html
    assert "abstractcore engines status --probe" in html


def test_shared_vocabulary_labels_are_exactly_the_contract() -> None:
    js = fragment("models")["js"]
    for status, label in (("installed", "installed"), ("absent", "not downloaded"), ("unknown", "unknown"), ("not_applicable", "remote")):
        assert f'{status}: {{ label: "{label}"' in js
    for verdict, label in (("fits", "fits"), ("tight", "tight"), ("too_large", "too large"), ("partial_offload", "partial offload"), ("needs_gpu_limit", "needs GPU limit"), ("unknown", "unknown")):
        assert f'{verdict}: {{ label: "{label}"' in js
    for verb in (">Download</button>", ">Delete</button>", ">Install</button>", ">Open download page</button>"):
        assert verb in js, verb
    # Install confirmation names the host and shows the exact argv.
    assert "Runs on host <strong>" in js and 'data-acc="install-argv"' in js


def test_fragment_contract_shape_and_single_namespace() -> None:
    for kind in FRAGMENT_KINDS:
        frag = fragment(kind)
        assert set(frag) == {"html", "js", "css"}
        assert f'data-acc-kind="{kind}"' in frag["html"]
        assert frag["css"].count(".acc-root") > 10
    js = fragment("models")["js"]
    assert js == fragment("engines")["js"]
    assert "window.AbstractCoreConsole = Object.freeze(" in js
    assert re.search(r"^\(function \(\) \{", js) and js.rstrip().endswith("})();")
    # Only one global: every other assignment to window.* is forbidden.
    assert re.findall(r"window\.([A-Za-z_$][\w$]*)\s*=(?!=)", js) == ["AbstractCoreConsole"]
    # Mount contract: injected request + 1.5 s job polling through it.
    assert "const POLL_MS = 1500;" in js
    assert 'ctx.request("GET", `${ctx.apiBase}/jobs/${encodeURIComponent(id)}`)' in js
    assert "function mount(kind, rootEl, options)" in js
    with pytest.raises(ValueError):
        fragment("overview")


def test_every_action_carries_a_cli_equivalent() -> None:
    js = fragment("models")["js"]
    assert "job.cli_equivalent" in js
    for cli in ("models download", "models delete", "engines install", "engines open", "models search", "models catalog"):
        assert cli in js, cli


def test_default_request_prompts_for_the_token_on_401_and_keeps_it_in_session_storage() -> None:
    js = fragment("models")["js"]
    assert "res.status === 401" in js and "promptToken(" in js
    assert 'const TOKEN_KEY = "abstractcore_console_token";' in js
    assert 'storeSet("session", TOKEN_KEY' in js
    assert "Authorization = `Bearer ${token}`" in js


def test_templates_embedded_in_js_cannot_close_the_script_tag() -> None:
    js = fragment("models")["js"]
    assert "</script" not in js.lower()
    assert "<\\/div>" in js


def test_page_inline_javascript_parses() -> None:
    scripts = _scripts(render_console_html())
    assert len(scripts) == 2
    _node_check("\n".join(scripts))


def test_fragment_javascript_parses_standalone() -> None:
    _node_check(fragment("engines")["js"])


# -------------------------------------------------------------------- theme


def test_theme_module_matches_the_kit_source_exactly() -> None:
    """Drift pin: regenerate from the abstractuic checkout and compare byte-for-byte."""
    kit_src = theme_sync.locate_kit_src()
    if kit_src is None:
        pytest.skip("abstractuic kit source not present (non-monorepo checkout)")
    regenerated = theme_sync.generate_themes_module(kit_src)
    current = (Path(theme_sync.__file__).parent / theme_sync.GENERATED_MODULE).read_text(encoding="utf-8")
    assert current == regenerated, (
        "abstractcore/console/themes.py is STALE vs the abstractuic kit; run "
        "`python -m abstractcore.console.theme_sync`"
    )


def test_every_kit_theme_is_offered_and_styled() -> None:
    ids = [s["id"] for s in KIT_THEME_SPECS]
    assert len(ids) >= 20 and len(set(ids)) == len(ids)
    css_ids = set(re.findall(r":root\.theme-([a-z0-9-]+)", KIT_THEME_CSS))
    assert [i for i in ids if i != "dark" and i not in css_ids] == []
    assert "light" in KIT_LIGHT_THEME_IDS
    assert KIT_ROOT_CSS.startswith(":root {") and "--bg-primary:" in KIT_ROOT_CSS

    html = render_console_html()
    assert KIT_ROOT_CSS in html and KIT_THEME_CSS in html
    for spec in KIT_THEME_SPECS:
        assert json.dumps(spec["label"]) in html
    # light/dark follows the OS by default, with a toggle.
    assert "prefers-color-scheme: light" in html and 'id="acc-theme-toggle"' in html


def test_the_responsive_token_layer_is_carried_and_served() -> None:
    """ui-kit >= 0.3.0 responsive tokens (--tap-min, --vh-full, --safe-*, the
    pointer:coarse floors) reach the standalone page by name; component rules
    and theme blocks do not (the console styles its own markup)."""
    for token in ("--tap-min", "--vh-full", "--safe-top", "--gutter", "--font-size-input", "text-size-adjust"):
        assert token in KIT_RESPONSIVE_CSS, token
    assert ".theme-" not in KIT_RESPONSIVE_CSS and ".af-" not in KIT_RESPONSIVE_CSS
    html = render_console_html()
    assert KIT_RESPONSIVE_CSS in html
    assert "viewport-fit=cover" in html
    assert "maximum-scale" not in html and "user-scalable" not in html
    # WebKit ignores min-height on native selects: on touch they drop the native
    # look (the kit 0.3.2 rule, mirrored because the kit component CSS is not loaded).
    coarse = html[html.index("@media (pointer: coarse)"):]
    assert 'select:not([multiple]):where(:not([size]), [size="1"]), .acc-root select:not([multiple]):where(:not([size]), [size="1"]) { appearance: none;' in coarse
    assert "background-image: var(--af-select-chevron," in coarse
    # ...but never through the embeddable fragment, whose CSS must stay scoped to .acc-root.
    assert "select:not([multiple])" not in fragment("models")["css"]


def test_every_css_variable_the_page_uses_is_declared_or_has_a_fallback() -> None:
    """An undefined `var(--x)` without a fallback is silently `unset` (a 44 px touch
    rule would do nothing): every token the standalone page reads comes from
    KIT_ROOT_CSS / KIT_RESPONSIVE_CSS / the theme blocks, or carries a fallback."""
    html = render_console_html()
    css = re.sub(r"/\*.*?\*/", "", "\n".join(re.findall(r"<style\b[^>]*>(.*?)</style>", html, flags=re.S)), flags=re.S)
    declared = set(re.findall(r"(--[A-Za-z0-9_-]+)\s*:", css))
    assert {"--tap-min", "--vh-full", "--safe-top", "--font-size-input"} <= declared
    undefined = sorted({name for name, sep in re.findall(r"var\(\s*(--[A-Za-z0-9_-]+)\s*([,)])", css)
                        if sep == ")" and name not in declared})
    assert undefined == [], f"CSS variables used without a declaration or fallback: {undefined}"


def _page_css() -> str:
    html = render_console_html()
    return re.sub(r"/\*.*?\*/", "", "\n".join(re.findall(r"<style\b[^>]*>(.*?)</style>", html, flags=re.S)), flags=re.S)


def test_every_media_query_uses_a_named_breakpoint() -> None:
    """DESIGN breakpoints only: 480 / 768 / 1024 / 1440 (+ 500 px tall)."""
    queries = " ".join(re.findall(r"@media\s*([^{]+)\{", _page_css()))
    widths = re.findall(r"\((max|min)-width:\s*([0-9.]+px)\)", queries)
    assert widths, "no width queries found (the style extraction broke)"
    allowed = {"max": {"479.98px", "767.98px", "1023.98px", "1439.98px"}, "min": {"480px", "768px", "1024px", "1440px"}}
    stray = sorted({f"{k}-width: {v}" for k, v in widths if v not in allowed[k]})
    assert stray == [], f"app-local breakpoints: {stray}"
    assert set(re.findall(r"\(max-height:\s*([0-9.]+px)\)", queries)) <= {"500px"}


def _coarse_blocks(css: str) -> str:
    out, i = [], 0
    while True:
        i = css.find("@media (pointer: coarse) {", i)
        if i < 0:
            return "\n".join(out)
        depth, j = 0, css.index("{", i)
        while True:
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            if depth == 0:
                break
            j += 1
        out.append(css[i : j + 1])
        i = j


def test_touch_floors_are_present() -> None:
    """44 px targets and 16 px inputs on coarse pointers (DESIGN §3.2/§2.1)."""
    coarse = _coarse_blocks(_page_css())
    assert ".acc-root .acc-btn, .acc-root .acc-toolbar label, .acc-root details summary { min-height: var(--tap-min, 44px); }" in coarse
    assert re.search(r"\.acc-root select \{[^}]*font-size: var\(--font-size-input, 16px\)[^}]*min-height: var\(--tap-min, 44px\)", coarse)
    assert re.search(r"\.acc-tabs button \{ min-height: var\(--tap-min, 44px\); \}", coarse)


def test_cards_and_tables_never_widen_the_page() -> None:
    """Overview key/value cards (Engines chips) shrink with their pane; on phones
    the tables scroll inside their wrapper with one-line, ellipsized artifact ids."""
    css = _page_css()
    assert "dl.acc-kv { display: grid; grid-template-columns: minmax(0, max-content) minmax(0, 1fr);" in css
    assert re.search(r"dl\.acc-kv dd \.acc-badge \{[^}]*max-width: 100%", css)
    phone = css[css.index("@media (max-width: 767.98px) {\n  .acc-topbar"):]
    assert ".acc-main .acc-table-scroll > table.acc-table { min-width: 640px; }" in phone
    assert re.search(r"\.acc-main code\.acc-artifact-id \{[^}]*white-space: nowrap;[^}]*text-overflow: ellipsis", phone)
    assert 'class="acc-artifact-id" title="${esc(r.artifact)}"' in render_console_html()


def test_theme_sync_refuses_a_kit_without_the_responsive_layer() -> None:
    with pytest.raises(ValueError):
        theme_sync.parse_responsive_token_css(":root {\n  --x: 1;\n}\n\nhtml {\n  text-size-adjust: 100%;\n}\n")


def test_theme_sync_refuses_a_reshaped_kit(tmp_path: Path) -> None:
    (tmp_path / "theme.ts").write_text("export const THEME_SPECS = [];", encoding="utf-8")
    (tmp_path / "theme.css").write_text(":root {\n  --x: 1;\n}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        theme_sync.generate_themes_module(tmp_path)


# ------------------------------------------------------------------- routes


def _stub_app() -> tuple[FastAPI, object]:
    app = FastAPI()
    app.include_router(console_router)
    stub = build_stub_router()
    app.include_router(stub)
    return app, stub


def test_console_routes_serve_page_and_fragments_with_stubbed_contract_endpoints() -> None:
    app, stub = _stub_app()
    client = TestClient(app)

    page = client.get("/console")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert 'id="acc-tab-catalog"' in page.text

    for kind in FRAGMENT_KINDS:
        res = client.get(f"/console/fragment/{kind}")
        assert res.status_code == 200
        body = res.json()
        assert body["kind"] == kind and set(body) == {"kind", "html", "js", "css"}
        assert body == {"kind": kind, **fragment(kind)}
    assert client.get("/console/fragment/overview").status_code == 404

    # The endpoints the page calls answer the contract shapes.
    assert client.get("/acore/host/profile").json()["schema"] == "host_profile_v1"
    assert client.get("/acore/engines?probe=1").json()["schema"] == "engines_status_v1"
    assert client.get("/acore/models/catalog?q=qwen&fits=1").json()["rows"][0]["id"] == "qwen3-8b"
    assert client.get("/acore/models/installed").json()["schema"] == "models_installed_v1"
    job = client.post("/acore/models/download", json={"provider": "ollama", "artifact": "qwen3:8b", "dry_run": False}).json()
    assert job["schema"] == "host_job_v1" and job["status"] == "running"
    assert client.get(f"/acore/jobs/{job['job_id']}").json()["percent"] == 50.0
    assert client.get(f"/acore/jobs/{job['job_id']}").json()["status"] == "completed"
    assert client.get("/acore/jobs/nope").status_code == 404


def test_catalog_fixture_covers_every_vocabulary_state() -> None:
    arts = [a for r in CATALOG["rows"] for a in r["artifacts"]]
    assert {a["presence"]["status"] for a in arts} >= {"installed", "absent", "not_applicable"}
    assert {a["fit"]["verdict"] for a in arts} >= {"fits", "too_large", "unknown"}


def test_real_server_app_mounts_console_outside_the_auth_boundary(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "console-test-token")
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    from abstractcore.server.app import app

    client = TestClient(app)
    page = client.get("/console")
    assert page.status_code == 200 and "AbstractCoreConsole.boot(" in page.text
    assert client.get("/console/fragment/models").status_code == 200
    # Everything else stays behind the bearer token.
    assert client.get("/acore/auth/validate").status_code == 401
    assert client.get("/console-not-exempt").status_code == 401
    ok = client.get("/acore/auth/validate", headers={"Authorization": "Bearer console-test-token"})
    assert ok.status_code == 200 and ok.json()["authenticated"] is True


# ------------------------------------------------ subprocess smoke (no browser)

_SMOKE_SERVER = """
import sys, uvicorn
sys.path.insert(0, {tests_dir!r})
from abstractcore.server.app import app
from console_web_fixtures import build_stub_router
app.include_router(build_stub_router())
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="error")
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, token: str | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


@pytest.mark.slow
def test_served_console_smoke_in_a_subprocess(tmp_path: Path) -> None:
    port = _free_port()
    env = dict(os.environ)
    env.update(
        ABSTRACTCORE_AUTH_TOKEN="smoke-token",
        ABSTRACTCORE_SERVER_DISABLE_CENTRALIZED_CONFIG="1",
        ABSTRACTFRAMEWORK_DATA_REGISTRY=str(tmp_path / "registry.json"),
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    env.pop("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", None)
    proc = subprocess.Popen(
        [sys.executable, "-c", _SMOKE_SERVER.format(tests_dir=str(TESTS_DIR), port=port)],
        env=env, cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                if _get(f"{base}/health")[0] == 200:
                    break
            except OSError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                out = proc.stdout.read() if proc.poll() is not None else ""
                pytest.fail(f"server did not come up on {base}: {out[-2000:]}")
            time.sleep(0.3)

        status, page = _get(f"{base}/console")
        assert status == 200 and 'id="acc-tab-button-catalog"' in page
        status, frag = _get(f"{base}/console/fragment/models")
        assert status == 200 and json.loads(frag)["kind"] == "models"
        # Data routes need the token the page prompts for.
        assert _get(f"{base}/acore/host/profile")[0] == 401
        status, profile = _get(f"{base}/acore/host/profile", token="smoke-token")
        assert status == 200 and json.loads(profile)["schema"] == "host_profile_v1"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
