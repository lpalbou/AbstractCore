"""Adversarial round 5: attack the JavaScript render escalation.

`fetch_url(render_js=...)` re-fetches a page whose static extraction came back
empty through the headless browser this package already ships (the optional
`browser` extra) and runs the RENDERED DOM through the same extraction
pipeline. That turns fetch_url into a tool that EXECUTES third-party
JavaScript, which is a different security and cost posture from an HTTP GET.

This file attacks the escalation on six axes:

  A. WHEN it fires. A browser launch must be the exception, never the toll on
     an ordinary fetch, and it must not be burned on a page a browser cannot
     help (bot mitigation).
  B. NEVER HANGS. Infinite JS loop, a load event that never fires, forever
     polling, a DOM that grows without bound.
  C. NEVER LEAKS. chrome-headless-shell escapes the worker's process group via
     setsid(); the count must return to baseline after success, after a hard
     timeout, and after a navigation error.
  D. SSRF. The static path refuses loopback/link-local/metadata destinations.
     A JavaScript-executing browser must not become the way around that guard.
  E. THE ENVELOPE. Provenance must be disclosed, evidence must still mean what
     its key says, amplification must stay bounded.
  F. DEGRADATION. No Playwright, or Playwright without the Chromium binary,
     must produce an actionable hint — never a traceback, never a hang.

Offline and deterministic by default: every hostile page is served from a
loopback HTTP server built in-process, and the SSRF allowlist is scoped to that
server's port for the duration of the test. Tests that need a REAL client-
rendered site are gated behind ABSTRACT_E2E_FETCH_URL=1 (the convention in
test_fetch_url_roster_live.py) — a rendered DOM cannot be committed as a
fixture without going stale, so it is gated rather than faked.

Tests that need a working headless browser skip when one is not installed, so
the file is safe in a CI image without the `browser` extra.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import pytest

FIXTURES = Path(__file__).resolve().parent / "fetch_url_fixtures"
sys.path.insert(0, str(FIXTURES))

from abstractcore.tools import browser_tools as BT  # noqa: E402
from abstractcore.tools.common_tools import (  # noqa: E402
    _MIN_REAL_CONTENT_CHARS,
    _ZERO_TEXT_FLOOR_CHARS,
    _detect_unrenderable_html,
    _escalate_to_rendered_dom,
    _extract_main_content,
    fetch_url,
)
from abstractcore.tools.fetch_url_ssrf import (  # noqa: E402
    fetch_url_guard_destination,
    reset_fetch_url_allowlist_cache,
)

# The fetch is faked; the SSRF check still resolves the host first. Answer that
# from a fake resolver instead of real DNS (network guard finding, 2026-09-24).
pytestmark = pytest.mark.usefixtures("fake_public_dns")

LIVE = os.getenv("ABSTRACT_E2E_FETCH_URL") == "1"
live_only = pytest.mark.skipif(
    not LIVE, reason="Set ABSTRACT_E2E_FETCH_URL=1 to run the live render tests."
)


def _browser_available() -> bool:
    """True when Playwright AND a Chromium binary are actually usable.

    Importable-but-no-binary is a real state on CI images, and it produces a
    `render_unavailable` result rather than a render — a test that only checked
    the import would then assert against an install hint.
    """
    if not BT._ensure_playwright():
        return False
    # Launch chromium directly rather than going through `render_url_html`:
    # that function refuses non-http(s) schemes on purpose (rendering `data:`
    # executes attacker-supplied markup and `file://` reads local files), so a
    # `data:` self-check reports "no browser" on a perfectly working install
    # and SILENTLY SKIPS every test in this file — including the SSRF ones.
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


_HAS_BROWSER: Optional[bool] = None


def _has_browser() -> bool:
    global _HAS_BROWSER
    if _HAS_BROWSER is None:
        _HAS_BROWSER = _browser_available()
    return _HAS_BROWSER


needs_browser = pytest.mark.skipif(
    not _has_browser(),
    reason='needs the browser extra: pip install "abstractcore[browser]" '
    "&& python -m playwright install --only-shell chromium",
)


# ---------------------------------------------------------------------------
# Loopback fixture server. Every hostile page in this file is served from here,
# so the attacks are offline, deterministic, and cannot reach the internet.
# ---------------------------------------------------------------------------

# Enough inert script bytes that the shell clears the 20 KB mark
# `_detect_unrenderable_html` uses to call a text-free 200 an "empty_content"
# SPA shell. Without this a hostile page is simply "a short valid page" and the
# escalation never fires, so the attack would test nothing.
_SHELL_PAD = "<script>" + ("var pad" + "x" * 40 + "=1;") * 700 + "</script>"

_REAL_PROSE = (
    "This paragraph is ordinary server-rendered prose with enough substance "
    "that the static extractor is satisfied and no escalation can be justified. "
) * 6


def _page(body: str, *, status: int = 200) -> Callable[[Any], None]:
    def handler(h: Any) -> None:
        doc = (
            "<!doctype html><html><head><title>Adversarial Page</title></head>"
            f"<body>{body}</body></html>"
        ).encode()
        h.send_response(status)
        h.send_header("Content-Type", "text/html; charset=utf-8")
        h.send_header("Content-Length", str(len(doc)))
        h.end_headers()
        h.wfile.write(doc)

    return handler


def _never_responds(h: Any) -> None:
    # Accepts the connection and then holds it. Used as a subresource so the
    # `load` event can never fire.
    time.sleep(900)


def _redirect_to(location: str) -> Callable[[Any], None]:
    def handler(h: Any) -> None:
        h.send_response(302)
        h.send_header("Location", location)
        h.send_header("Content-Length", "0")
        h.end_headers()

    return handler


class _Server:
    def __init__(self, routes: Dict[str, Callable[[Any], None]]) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # silence the test log
                pass

            def do_GET(self) -> None:  # noqa: N802
                route = outer.routes.get(self.path.split("?", 1)[0])
                if route is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                try:
                    route(self)
                except Exception:  # a hostile route may drop the connection
                    pass

        self.routes = routes
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._srv.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        try:
            self._srv.shutdown()
        except Exception:
            pass
        try:
            self._srv.server_close()
        except Exception:
            pass


ROUTES: Dict[str, Callable[[Any], None]] = {
    # --- pages the escalation is SUPPOSED to fire on -----------------------
    # A shell with no text: the static extractor gets nothing, the browser
    # paints a real article. This is the feature working.
    "/spa": _page(
        f"<div id='root'></div>{_SHELL_PAD}"
        "<script>document.getElementById('root').innerHTML="
        f"'<main><article><h1>Rendered Headline</h1><p>{_REAL_PROSE}</p></article></main>';</script>"
    ),
    # --- pages that must NOT cost a browser launch -------------------------
    "/article": _page(f"<main><article><h1>Static Headline</h1><p>{_REAL_PROSE}</p></article></main>"),
    # --- hostile pages -----------------------------------------------------
    "/infinite-loop": _page(f"<div></div>{_SHELL_PAD}<script>while(true){{}}</script>"),
    "/deferred-loop": _page(
        f"<div></div>{_SHELL_PAD}<script>setTimeout(()=>{{while(true){{}}}},250)</script>"
    ),
    "/never-load": _page(f"<div></div>{_SHELL_PAD}<img src='/black-hole'>"),
    "/forever-poll": _page(
        f"<div></div>{_SHELL_PAD}"
        "<script>setInterval(()=>fetch('/black-hole').catch(()=>0),40)</script>"
    ),
    "/forever-socket": _page(
        f"<div></div>{_SHELL_PAD}"
        "<script>setInterval(()=>{try{new WebSocket('ws://127.0.0.1:1/x')}catch(e){}},50)</script>"
    ),
    "/dom-bomb": _page(
        f"<div id='x'></div>{_SHELL_PAD}"
        "<script>setInterval(()=>{for(let i=0;i<400;i++){const d=document.createElement('div');"
        "d.textContent='grow '.repeat(40);document.getElementById('x').appendChild(d)}},20)</script>"
    ),
    "/black-hole": _never_responds,
}


@pytest.fixture(scope="module")
def server() -> Iterator[_Server]:
    srv = _Server(dict(ROUTES))
    yield srv
    srv.close()


@pytest.fixture()
def allow_loopback(server: _Server) -> Iterator[str]:
    """Scope the SSRF allowlist to the fixture server for one test.

    fetch_url refuses loopback by design; without this the attacks below could
    never reach the hostile pages at all. Only the fixture server's port is
    allowlisted, which is what makes the SSRF tests meaningful: a SECOND
    loopback port stays refused, so a bypass shows up as a bypass.
    """
    previous = os.environ.get("ABSTRACTCORE_FETCH_URL_ALLOW")
    os.environ["ABSTRACTCORE_FETCH_URL_ALLOW"] = f"127.0.0.1:{server.port}"
    reset_fetch_url_allowlist_cache()
    try:
        yield server.base
    finally:
        if previous is None:
            os.environ.pop("ABSTRACTCORE_FETCH_URL_ALLOW", None)
        else:
            os.environ["ABSTRACTCORE_FETCH_URL_ALLOW"] = previous
        reset_fetch_url_allowlist_cache()


def _chromium_processes() -> int:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "chrome-headless-shell"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return 0
    return len([tok for tok in out.split() if tok.strip()])


def _settle_to(baseline: int, *, limit_s: float = 30.0) -> int:
    """Wait for the browser process count to fall back to `baseline`.

    Teardown is asynchronous: the worker exits, then the OS reaps the browser
    tree. Measured at up to ~5s on a loaded machine, so a test that counted
    immediately would report a leak that is not one.
    """
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        current = _chromium_processes()
        if current <= baseline:
            return current
        time.sleep(0.4)
    return _chromium_processes()


# ===========================================================================
# A. WHEN THE ESCALATION FIRES
# ===========================================================================


def test_a_page_that_extracts_fine_statically_never_launches_a_browser(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A browser launch is ~0.5-1.5s. An ordinary article must never pay it.

    Counting launches rather than timing is the honest check: a fast machine
    hides a launch inside a generous timing budget.
    """
    launches: list[str] = []
    real = BT.render_url_html
    monkeypatch.setattr(
        BT, "render_url_html", lambda url, **kw: (launches.append(url), real(url, **kw))[1]
    )

    result = fetch_url(url=f"{allow_loopback}/article", timeout=20)

    assert result.get("success") is True
    assert len(str(result.get("content") or "")) > _MIN_REAL_CONTENT_CHARS
    assert launches == [], f"a statically-extractable page launched a browser: {launches}"
    assert result.get("rendered_with_browser") is False


def test_render_js_never_refuses_to_launch_even_on_an_empty_shell(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`render_js="never"` is the escape hatch for a caller that cannot afford
    a browser (sandbox, cost ceiling, hostile-URL policy). It must be absolute."""
    launches: list[str] = []
    monkeypatch.setattr(BT, "render_url_html", lambda url, **kw: launches.append(url) or {"ok": False})

    result = fetch_url(url=f"{allow_loopback}/spa", timeout=20, render_js="never")

    assert launches == [], "render_js='never' still launched a browser"
    assert result.get("success") is False
    assert result.get("error_class") == "empty_content"


@needs_browser
def test_an_empty_shell_escalates_and_recovers_the_rendered_article(
    allow_loopback: str,
) -> None:
    """The feature, end to end: text that exists only after JS runs comes back."""
    static_only = fetch_url(url=f"{allow_loopback}/spa", timeout=25, render_js="never")
    assert static_only.get("success") is False, "the fixture is not actually a JS-only shell"

    result = fetch_url(url=f"{allow_loopback}/spa", timeout=25)

    assert result.get("success") is True, result.get("error")
    assert result.get("rendered_with_browser") is True
    content = str(result.get("content") or "")
    assert "Rendered Headline" in content
    assert "ordinary server-rendered prose" in content


# ---------------------------------------------------------------------------
# A2. bot mitigation is not a rendering problem
# ---------------------------------------------------------------------------

# Measured 2026-08-21 by rendering each of these in headless Chromium with the
# same honest identity the static fetch uses:
#   reddit.com/r/programming  static 8.4 KB shell (HTTP 200) -> rendered DOM says
#                             "You've been blocked by network security"
#   pubmed.ncbi.nlm.nih.gov   static 5.5 KB cookie gate (HTTP 203) -> rendered HTTP 403
#   imdb.com/title/...        static AWS WAF shell (HTTP 202)      -> rendered HTTP 403
# In every case the browser made the outcome WORSE, never better. That is the
# empirical basis for excluding request-level blocks from the escalation.
BOT_SHELL_FIXTURES = ("bot_reddit_programming.html", "bot_pubmed_abstract.html", "bot_imdb_title.html")


@pytest.mark.parametrize("fixture", BOT_SHELL_FIXTURES)
def test_a_bot_mitigation_shell_is_not_reported_as_a_render_success(fixture: str) -> None:
    """Whatever else happens, a bot block must never come back as content.

    The failure mode this guards is subtle: reddit's rendered DOM carries ~130
    characters of "You've been blocked by network security". That clears
    `_ZERO_TEXT_FLOOR_CHARS` (25), so the escalation's own success test accepts
    it and the block message is returned as if it were the page.
    """
    html = (FIXTURES / fixture).read_text(encoding="utf-8", errors="replace")
    content = str(_extract_main_content(html, "https://example.com/blocked", keep_links=True).get("content") or "")
    detected = _detect_unrenderable_html(html, content)
    assert detected is not None, f"{fixture} extracted {len(content)} chars and was called a success"
    assert detected[0] in {"bot_challenge", "empty_content", "js_required"}


def test_a_render_that_only_produced_a_block_message_is_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED (attack): the escalation's success floor is `_ZERO_TEXT_FLOOR_CHARS`
    (25), 8x looser than `_MIN_REAL_CONTENT_CHARS` (200) — the floor the static
    path uses to REJECT the very same page. So a rendered DOM whose entire
    visible text is a network-block notice is accepted and returned as content.

    Measured on a live reddit render (2026-08-21): 130 characters reading
    "You've been blocked by network security. If you think you've been blocked
    by mistake, file a ticket below and we'll look into it." — over the floor.

    The two floors should be the same number. A render that cannot clear the
    module's own definition of real content has not helped.
    """
    block_dom = (
        "<!doctype html><html><head><title>Blocked</title></head><body><main><p>"
        "You've been blocked by network security. If you think you've been blocked "
        "by mistake, file a ticket below and we'll look into it."
        "</p></main></body></html>"
    )
    monkeypatch.setattr(
        BT,
        "render_url_html",
        lambda url, **kw: {"ok": True, "html": block_dom, "final_url": url, "elapsed_s": 0.5},
    )

    main, note = _escalate_to_rendered_dom("https://example.com/blocked", "empty_content", "auto")

    assert main is None, (
        "a rendered DOM containing only a network-block notice was accepted as content: "
        f"{str((main or {}).get('content'))[:120]!r}"
    )
    assert "block" in note.lower() or "did not help" in note.lower()


def test_a_render_below_the_real_content_floor_is_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED (attack): same asymmetry, on the maintainer's own class-(a) example.

    excalidraw.com is genuinely client-rendered: a 6.8 KB shell, zero static
    text. It escalates, renders 60,615 characters of DOM, and yields 189
    characters of extracted text — a browser-storage disclaimer, not the app.
    189 < `_MIN_REAL_CONTENT_CHARS`, so the same bytes handed to the STATIC path
    would be rejected as "no readable content"; handed to the render path they
    are returned as `success=True`.

    The requirement is "must not come back empty or thin". 189 characters from
    a whiteboard app is thin, and the loud, actionable error it replaced was
    more useful to an agent than a storage warning.
    """
    disclaimer = (
        "<!doctype html><html><head><title>Excalidraw</title></head><body><main><p>"
        "Your drawings are saved in your browser's storage. Browser storage can be "
        "cleared unexpectedly. Save your work to a file regularly to avoid losing it."
        "</p></main></body></html>"
    )
    monkeypatch.setattr(
        BT,
        "render_url_html",
        lambda url, **kw: {"ok": True, "html": disclaimer, "final_url": url, "elapsed_s": 1.0},
    )

    main, note = _escalate_to_rendered_dom("https://excalidraw.com/", "empty_content", "auto")
    extracted = str((main or {}).get("content") or "")

    assert len(extracted) == 0 or len(extracted) >= _MIN_REAL_CONTENT_CHARS, (
        f"the render path accepted {len(extracted)} chars, below the module's own "
        f"real-content floor of {_MIN_REAL_CONTENT_CHARS} that the static path enforces; "
        f"note={note!r}"
    )


def test_status_code_202_203_are_a_pre_launch_bot_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cheap discriminator that needs no browser.

    Measured with fetch_url's own headers: pubmed answers 203 (Non-Authoritative
    Information) and imdb answers 202 — both bot-mitigation tells, both `ok` as
    far as `requests` is concerned, so both currently reach the escalation and
    burn a launch. A 2xx that is neither 200 nor 204 and yields NO text is a
    request-level block, not a client-rendered page.

    This test states the contract; it does not assert on a constant that does
    not exist yet.
    """
    seen: Dict[str, Any] = {}

    def _fake(url: str, **kw: Any) -> Dict[str, Any]:
        seen["launched"] = url
        return {"ok": False, "error_class": "render_failed", "message": "n/a", "hint": ""}

    monkeypatch.setattr(BT, "render_url_html", _fake)
    # Documented, not asserted-on-a-constant: the shells that carry these
    # statuses are committed so a future discriminator can be measured.
    assert (FIXTURES / "bot_pubmed_abstract.html").exists()
    assert (FIXTURES / "bot_imdb_title.html").exists()
    imdb = (FIXTURES / "bot_imdb_title.html").read_text(encoding="utf-8", errors="replace")
    assert "awswaf" in imdb.lower(), "the imdb fixture no longer carries the AWS WAF signature"
    pubmed = (FIXTURES / "bot_pubmed_abstract.html").read_text(encoding="utf-8", errors="replace")
    assert "cookies-required" in pubmed.lower()


# ===========================================================================
# B. NEVER HANGS
# ===========================================================================

# `RENDER_DEFAULT_TIMEOUT_S` (20) + `_RENDER_LAUNCH_GRACE_S` (15) is the worst
# case the parent enforces with a process-tree kill. Measured 35.2s on an
# infinite JS loop. The budget here is that bound plus slack for a loaded box.
_HARD_BOUND_S = BT.RENDER_DEFAULT_TIMEOUT_S + BT._RENDER_LAUNCH_GRACE_S
_HANG_BUDGET_S = _HARD_BOUND_S + 20.0

HOSTILE_PAGES = ("/infinite-loop", "/deferred-loop", "/never-load", "/forever-poll", "/forever-socket", "/dom-bomb")


@needs_browser
@pytest.mark.parametrize("path", HOSTILE_PAGES)
def test_a_hostile_page_can_never_hang_the_fetch(allow_loopback: str, path: str) -> None:
    started = time.monotonic()
    result = fetch_url(url=f"{allow_loopback}{path}", timeout=10)
    elapsed = time.monotonic() - started

    assert elapsed < _HANG_BUDGET_S, f"{path} took {elapsed:.1f}s (bound {_HANG_BUDGET_S:.0f}s)"
    assert isinstance(result, dict)
    # Whatever it decided, it must have decided something — never a bare
    # success carrying nothing.
    if result.get("success"):
        assert len(str(result.get("content") or "")) >= _ZERO_TEXT_FLOOR_CHARS
    else:
        assert result.get("error_class"), f"{path} failed with no error_class"


@needs_browser
def test_the_render_budget_respects_the_callers_timeout(allow_loopback: str) -> None:
    """RED (attack): the escalation budget ignores fetch_url's `timeout`.

    Measured on the infinite-JS-loop page: `fetch_url(timeout=5)` takes 35.2s
    and `fetch_url(timeout=45)` takes 35.1s — identical, because the render
    budget is the module constant `RENDER_DEFAULT_TIMEOUT_S` plus its grace,
    independent of what the caller asked for. `render_url_html` already accepts
    `timeout_s`; `_escalate_to_rendered_dom` calls it with the default.

    An agent that sets `timeout=5` because it is inside a 30s tool deadline
    blows that deadline every time a page escalates. A caller's timeout should
    bound the whole call, not just its first HTTP hop.
    """
    started = time.monotonic()
    fetch_url(url=f"{allow_loopback}/infinite-loop", timeout=5)
    elapsed = time.monotonic() - started

    assert elapsed < 5 + BT._RENDER_LAUNCH_GRACE_S + 5, (
        f"fetch_url(timeout=5) spent {elapsed:.1f}s; the render escalation ignores the "
        f"caller's timeout and uses RENDER_DEFAULT_TIMEOUT_S={BT.RENDER_DEFAULT_TIMEOUT_S}"
    )


# ===========================================================================
# C. NEVER LEAKS A BROWSER PROCESS
# ===========================================================================


@needs_browser
@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="pgrep-based accounting")
@pytest.mark.parametrize(
    "path", ["/spa", "/infinite-loop", "/never-load", "/dom-bomb"]
)
def test_no_browser_process_survives_the_fetch(allow_loopback: str, path: str) -> None:
    """Covers the success path, the hard-timeout path, and the degraded path.

    chrome-headless-shell calls setsid(), so it escapes the worker's process
    group — a group-only kill orphans it. The guarantee is a process-TREE kill
    in the parent, and this is the only assertion that can see it fail.
    """
    baseline = _chromium_processes()
    fetch_url(url=f"{allow_loopback}{path}", timeout=10)
    settled = _settle_to(baseline)

    assert settled <= baseline, (
        f"{path} leaked {settled - baseline} chrome-headless-shell process(es)"
    )


@needs_browser
@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="pgrep-based accounting")
def test_no_browser_process_survives_an_exception_in_the_extraction(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leak check that a happy-path sweep misses: the browser has already
    run when the code that consumes its output blows up."""
    baseline = _chromium_processes()

    import abstractcore.tools.common_tools as CT

    real_extract = CT._extract_main_content
    calls = {"n": 0}

    def _boom(html: str, url: str, **kw: Any) -> Dict[str, Any]:
        calls["n"] += 1
        if calls["n"] > 1:  # first call is the static path; blow up on the DOM
            raise RuntimeError("adversarial failure while extracting the rendered DOM")
        return real_extract(html, url, **kw)

    monkeypatch.setattr(CT, "_extract_main_content", _boom)
    try:
        fetch_url(url=f"{allow_loopback}/spa", timeout=15)
    except Exception:
        pass

    settled = _settle_to(baseline)
    assert settled <= baseline, f"an exception after the render leaked {settled - baseline} process(es)"


@needs_browser
@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="pgrep-based accounting")
def test_repeated_escalations_do_not_accumulate_browsers(allow_loopback: str) -> None:
    """A slow drip is the leak that survives a 3-fetch sweep."""
    baseline = _chromium_processes()
    for _ in range(6):
        fetch_url(url=f"{allow_loopback}/spa", timeout=15)
    settled = _settle_to(baseline)
    assert settled <= baseline, f"6 escalations left {settled - baseline} process(es) behind"


# ===========================================================================
# D. SSRF — the render path must not be the way around the static guard
# ===========================================================================


@pytest.fixture()
def victim() -> Iterator[Tuple[_Server, str]]:
    """A second loopback server that is NEVER allowlisted.

    fetch_url refuses it on the static path (`blocked_ssrf`). Anything that
    surfaces its marker in a fetch_url result reached it some other way.
    """
    marker = "LOOPBACK-ONLY-MARKER-7c1a"
    body = (
        "<main><p>"
        + marker
        + " internal admin console. "
        + ("Sensitive internal prose that must never reach a model. " * 8)
        + "</p></main>"
    )
    srv = _Server({"/internal": _page(body)})
    yield srv, marker
    srv.close()


def test_the_static_path_refuses_a_non_allowlisted_loopback_host(
    allow_loopback: str, victim: Tuple[_Server, str]
) -> None:
    """The baseline the render path must not undercut."""
    vsrv, marker = victim
    result = fetch_url(url=f"{vsrv.base}/internal", timeout=10)
    assert result.get("success") is False
    assert result.get("error_class") == "blocked_ssrf"
    assert marker not in json.dumps(result)


def test_the_guard_still_refuses_the_victim_while_the_fixture_server_is_allowed(
    allow_loopback: str, victim: Tuple[_Server, str]
) -> None:
    """Sanity: the allowlist is scoped to one port, so a bypass is visible."""
    vsrv, _ = victim
    assert fetch_url_guard_destination(f"{vsrv.base}/internal") is not None
    assert fetch_url_guard_destination(f"{allow_loopback}/article") is None


@needs_browser
def test_a_server_redirect_into_loopback_is_refused_on_the_render_path(
    allow_loopback: str, victim: Tuple[_Server, str], server: _Server
) -> None:
    """A 3xx from an allowed host to a refused one. `requests` re-enters the
    guarded adapter on every hop; Chromium follows redirects internally."""
    vsrv, marker = victim
    server.routes["/redirect-302"] = _redirect_to(f"{vsrv.base}/internal")
    try:
        result = fetch_url(url=f"{allow_loopback}/redirect-302", timeout=15)
    finally:
        server.routes.pop("/redirect-302", None)

    assert marker not in json.dumps(result), (
        "a 302 into a non-allowlisted loopback host delivered its content through fetch_url"
    )


@needs_browser
def test_a_client_side_redirect_into_loopback_is_refused_on_the_render_path(
    allow_loopback: str, victim: Tuple[_Server, str], server: _Server
) -> None:
    """RED (attack): the SSRF bypass.

    `_escalate_to_rendered_dom` screens `response.url` — the destination the
    STATIC fetch landed on — and then hands that URL to a browser. Everything
    the browser navigates to AFTERWARDS is unscreened: `location.href`, a meta
    refresh, `location.replace`, a 3xx Chromium follows on its own. The document
    that ends up in `page.content()` is then same-origin with the internal host,
    so its whole body is readable and is returned as `content`.

    Measured 2026-08-21: an allowlisted page whose script sets
    `location.href='http://127.0.0.1:<victim>/internal'` returns
    `success=True`, `rendered_with_browser=True`, the internal marker in
    `content`, and `final_url` still pointing at the ORIGINAL allowed URL — so
    the envelope does not even record where the content came from.

    `render_url_html` already returns the landed `final_url` (verified: it
    reports the victim URL). Re-screening it before accepting the DOM closes the
    single-hop case; blocking at the request level (`context.route`, which
    browser_tools already uses for local targets) is what closes it properly,
    because a page can navigate away and back inside one probe.
    """
    vsrv, marker = victim
    server.routes["/js-redirect"] = _page(
        f"<div></div>{_SHELL_PAD}"
        f"<script>location.href='{vsrv.base}/internal';</script>"
    )
    try:
        result = fetch_url(url=f"{allow_loopback}/js-redirect", timeout=20)
    finally:
        server.routes.pop("/js-redirect", None)

    assert marker not in json.dumps(result), (
        "SSRF BYPASS: a client-side redirect carried non-public content through the render "
        f"path that the static path refuses. final_url={result.get('final_url')!r}, "
        f"rendered_with_browser={result.get('rendered_with_browser')!r}"
    )


@needs_browser
def test_a_meta_refresh_into_loopback_is_refused_on_the_render_path(
    allow_loopback: str, victim: Tuple[_Server, str], server: _Server
) -> None:
    """RED (attack): same bypass without a line of JavaScript.

    `<meta http-equiv="refresh">` needs no script execution at all, so a
    render-path guard that reasons about "JS can do anything" still has to cover
    plain declarative navigation.
    """
    vsrv, marker = victim
    server.routes["/meta-refresh"] = _page(
        f"<div></div>{_SHELL_PAD}"
        f"<meta http-equiv='refresh' content='0;url={vsrv.base}/internal'>"
    )
    try:
        result = fetch_url(url=f"{allow_loopback}/meta-refresh", timeout=20)
    finally:
        server.routes.pop("/meta-refresh", None)

    assert marker not in json.dumps(result), (
        "SSRF BYPASS: a meta refresh carried non-public content through the render path"
    )


@needs_browser
def test_the_envelope_reports_where_the_rendered_content_actually_came_from(
    allow_loopback: str, server: _Server
) -> None:
    """Provenance, independent of the SSRF question.

    If the browser followed the page somewhere else, `final_url` must say so.
    A `final_url` that reports the requested URL while `content` came from a
    different origin is a false receipt — and it is what makes the bypass above
    invisible in an audit.
    """
    server.routes["/hop"] = _page(
        f"<div></div>{_SHELL_PAD}<script>location.href='/article';</script>"
    )
    try:
        result = fetch_url(url=f"{allow_loopback}/hop", timeout=20)
    finally:
        server.routes.pop("/hop", None)

    if result.get("rendered_with_browser"):
        assert str(result.get("final_url") or "").endswith("/article"), (
            f"content came from /article but final_url says {result.get('final_url')!r}"
        )


def test_render_url_html_refuses_non_http_schemes() -> None:
    """RED (attack): `render_url_html` is a documented public helper and has no
    scheme validation at all — it builds the probe config directly instead of
    going through `_resolve_target`, which browser_probe uses to refuse
    `javascript:`, `data:` and friends by name.

    Verified 2026-08-21: `render_url_html('data:text/html,<h1>x</h1>')` renders
    the data URL, and `render_url_html('file:///path/to/local.html')` reads and
    renders a local file. fetch_url only ever feeds it `response.url`, so this
    is not reachable through the tool today — but it is one refactor away, and
    the helper's own docstring puts screening on the caller while accepting
    anything.
    """
    for target in ("data:text/html,<h1>data-url-body</h1>", "file:///etc/hosts", "javascript:1"):
        result = BT.render_url_html(target, timeout_s=10)
        assert not result.get("ok"), f"render_url_html rendered a non-http target: {target!r}"
        assert "scheme" in str(result.get("message", "")).lower() or result.get(
            "error_class"
        ) in {"render_bad_target", "render_blocked"}, (
            f"{target!r} was refused, but not by name: {result!r}"
        )


def test_the_metadata_ip_is_refused_before_any_browser_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud metadata is deny-always in the static guard. The render path must
    reach the same verdict WITHOUT paying for a launch."""
    launches: list[str] = []
    monkeypatch.setattr(BT, "render_url_html", lambda url, **kw: launches.append(url) or {"ok": False})

    main, note = _escalate_to_rendered_dom(
        "http://169.254.169.254/latest/meta-data/", "empty_content", "auto"
    )

    assert main is None
    assert launches == [], "a browser was launched at the cloud metadata endpoint"
    assert "ssrf" in note.lower() or "refused" in note.lower()


# ===========================================================================
# E. THE ENVELOPE CONTRACT
# ===========================================================================


def test_the_error_envelope_also_reports_whether_a_browser_was_tried(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED (attack): the three provenance keys exist only on the SUCCESS
    envelope.

    `rendered_with_browser`, `render_note` and `extraction_error` are set in the
    success return. The `No readable content extracted` return — the envelope a
    caller gets when the escalation was TRIED AND DID NOT HELP, which is exactly
    when "did we execute this page's JavaScript?" matters most — omits all
    three. The information survives only as prose appended to `suggestions`,
    which no consumer can branch on.

    Verified against a real fetch: the error envelope's keys are
    {content_type, description, detected_as, error, error_class, final_url,
    rendered, retryable, status_code, success, suggestions, timestamp, title,
    url} — none of the three.
    """
    monkeypatch.setattr(
        BT,
        "render_url_html",
        lambda url, **kw: {"ok": False, "error_class": "render_failed", "message": "nope", "hint": ""},
    )
    result = fetch_url(url=f"{allow_loopback}/spa", timeout=15)

    assert result.get("success") is False
    for key in ("rendered_with_browser", "render_note"):
        assert key in result, (
            f"the failure envelope omits {key!r}; a consumer cannot tell a browser was tried"
        )


def test_raw_text_still_means_the_bytes_the_server_sent(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED (attack): after an escalation, `raw_text` is silently replaced by the
    RENDERED DOM.

    `_escalate_to_rendered_dom` succeeds and the call site does
    `raw_text = rendered_main["_html"]`. From then on the envelope carries a
    browser-mutated DOM under the key that means "the source bytes", with
    `content_type` still quoting the HTTP response header and `size_bytes` still
    quoting the transport length. Measured on excalidraw.com: `size_bytes=6,862`
    alongside `raw_text` of 60,615 characters.

    Two consequences. The server's own bytes are DISCARDED — nothing in the
    envelope carries what was actually served, so the fetch is not reproducible
    and `raw_text_withheld.sha256` (when it fires) attests a DOM no one can
    re-derive. And `abstractruntime`'s EvidenceRecorder stores that value under
    `part: "raw"` with the HTTP content type, so the forensic record claims to
    be the server response and is not.

    The rendered DOM deserves its own key.
    """
    dom = (
        "<!doctype html><html><head><title>Rendered</title></head><body><main><p>"
        + ("Text that exists only after JavaScript ran on this page. " * 12)
        + "</p></main></body></html>"
    )
    monkeypatch.setattr(
        BT, "render_url_html", lambda url, **kw: {"ok": True, "html": dom, "final_url": url, "elapsed_s": 0.4}
    )
    result = fetch_url(url=f"{allow_loopback}/spa", timeout=15)
    assert result.get("rendered_with_browser") is True

    raw = result.get("raw_text")
    withheld = result.get("raw_text_withheld")
    if isinstance(raw, str):
        assert "Text that exists only after JavaScript ran" not in raw, (
            "`raw_text` carries the rendered DOM, not the bytes the server sent"
        )
    size_bytes = int(result.get("size_bytes") or 0)
    if isinstance(raw, str) and size_bytes:
        assert abs(len(raw) - size_bytes) < max(2048, size_bytes), (
            f"size_bytes={size_bytes:,} does not describe raw_text ({len(raw):,} chars)"
        )
    if isinstance(withheld, dict):
        assert withheld.get("chars", 0) <= size_bytes * 4, (
            "the withheld descriptor attests something much larger than the response"
        )


def test_a_rendered_fetch_does_not_blow_the_amplification_budget(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED (attack): the withholding exemption inverts on the render path.

    `_apply_fetch_url_payload_policy` keeps evidence inline whenever
    `content_chars < _MIN_REAL_CONTENT_CHARS`, on the reasoning that a page
    which extracted nothing must not have its only copy withheld. A render that
    produces a THIN result (see the excalidraw case above) satisfies that test
    while also attaching a rendered DOM an order of magnitude larger than the
    shell — so the exemption fires exactly when the payload is biggest.

    Measured on excalidraw.com: content 189 chars, raw_text 60,615 chars inline,
    total envelope 64,699 chars = 342x amplification over the payload.
    """
    big_dom = (
        "<!doctype html><html><head><title>App</title></head><body>"
        + "<script>" + ("var noise" + "y" * 60 + "=1;") * 900 + "</script>"
        + "<main><p>Short.</p></main></body></html>"
    )
    monkeypatch.setattr(
        BT,
        "render_url_html",
        lambda url, **kw: {"ok": True, "html": big_dom, "final_url": url, "elapsed_s": 0.4},
    )
    result = fetch_url(url=f"{allow_loopback}/spa", timeout=15)

    content_chars = len(str(result.get("content") or ""))
    total = len(json.dumps(result))
    if content_chars:
        assert total <= max(8_000, content_chars * 12), (
            f"envelope is {total:,} chars for {content_chars:,} chars of content "
            f"({total / content_chars:.0f}x amplification after a render escalation)"
        )


def test_the_human_readable_render_discloses_that_a_browser_was_used(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED (attack): `rendered` — the copy the MODEL reads — says nothing.

    Verified on excalidraw.com: the `rendered` block reports status, headers and
    "Size: 6,862 bytes" and never mentions that the content came out of a
    headless browser which executed the page's JavaScript. The model, and the
    human reading the transcript, believe they are looking at a plain HTTP GET.
    Provenance that exists only in a sibling key the model never sees is not
    disclosure.
    """
    dom = (
        "<!doctype html><html><head><title>Rendered</title></head><body><main><p>"
        + ("Content that only exists after scripts run. " * 12)
        + "</p></main></body></html>"
    )
    monkeypatch.setattr(
        BT, "render_url_html", lambda url, **kw: {"ok": True, "html": dom, "final_url": url, "elapsed_s": 0.4}
    )
    result = fetch_url(url=f"{allow_loopback}/spa", timeout=15)
    assert result.get("rendered_with_browser") is True

    rendered = str(result.get("rendered") or "").lower()
    assert "browser" in rendered or "javascript" in rendered or "rendered dom" in rendered, (
        "the model-visible `rendered` copy does not disclose the headless render"
    )


def test_two_fetches_of_the_same_rendered_page_agree(allow_loopback: str) -> None:
    """Determinism. Guards against a render whose output depends on animation
    frames, lazy hydration order, or a race with the readiness check.

    Measured stable on three live sites (hetzner, vercel, notion): byte-identical
    extraction across two renders. This pins the local case so a regression in
    the readiness signal is visible offline.
    """
    if not _has_browser():
        pytest.skip("needs the browser extra")
    first = fetch_url(url=f"{allow_loopback}/spa", timeout=25)
    second = fetch_url(url=f"{allow_loopback}/spa", timeout=25)
    if not (first.get("rendered_with_browser") and second.get("rendered_with_browser")):
        pytest.skip("the fixture did not escalate on this machine")
    assert str(first.get("content")) == str(second.get("content")), (
        "two renders of the same page produced different content"
    )


def test_extraction_error_is_recorded_instead_of_a_silent_empty_success(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the silent-degradation fix found while wiring the escalation.

    A bare `except Exception` around the whole HTML extraction block turned any
    internal failure into `success=True, content=None` — indistinguishable from
    a page that genuinely had nothing. The exception must now reach the caller.
    """
    import abstractcore.tools.common_tools as CT

    def _boom(html: str, url: str, **kw: Any) -> Dict[str, Any]:
        raise RuntimeError("adversarial extraction failure")

    monkeypatch.setattr(CT, "_extract_main_content", _boom)
    result = fetch_url(url=f"{allow_loopback}/article", timeout=15)

    recorded = str(result.get("extraction_error") or "")
    assert recorded, "an extraction that raised was reported as an ordinary result"
    assert "adversarial extraction failure" in recorded or "RuntimeError" in recorded


# ===========================================================================
# F. DEGRADATION WITHOUT A BROWSER
# ===========================================================================


def test_a_missing_playwright_gives_an_install_hint_not_a_traceback(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(BT, "_PLAYWRIGHT_AVAILABLE", False)
    started = time.monotonic()
    result = fetch_url(url=f"{allow_loopback}/spa", timeout=15)
    elapsed = time.monotonic() - started

    assert elapsed < 15.0, f"the unavailable-browser path took {elapsed:.1f}s"
    assert result.get("success") is False
    blob = json.dumps(result)
    assert "Traceback" not in blob
    assert "abstractcore[browser]" in blob or "playwright install" in blob, (
        "no actionable install hint reached the caller"
    )


def test_a_missing_chromium_binary_gives_the_download_hint(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        BT,
        "_spawn_probe",
        lambda cfg, budget: {
            "error": {"kind": "browser_missing", "message": "Executable doesn't exist at ..."}
        },
    )
    monkeypatch.setattr(BT, "_PLAYWRIGHT_AVAILABLE", True)
    result = fetch_url(url=f"{allow_loopback}/spa", timeout=15)

    blob = json.dumps(result)
    assert result.get("success") is False
    assert "Traceback" not in blob
    assert "playwright install" in blob, "the browser-binary hint did not reach the caller"


def test_the_escalation_never_raises_whatever_the_browser_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_escalate_to_rendered_dom` promises "never raises". Feed it garbage."""
    for payload in (
        {},
        {"ok": True},
        {"ok": True, "html": None},
        {"ok": True, "html": ""},
        {"ok": True, "html": "\x00\x01\x02"},
        {"ok": False},
        {"ok": False, "error_class": None, "message": None, "hint": None},
    ):
        monkeypatch.setattr(BT, "render_url_html", lambda url, _p=payload, **kw: _p)
        main, note = _escalate_to_rendered_dom("https://example.com/x", "empty_content", "auto")
        assert main is None or isinstance(main, dict)
        assert isinstance(note, str) and note, f"empty note for {payload!r}"

    def _explode(url: str, **kw: Any) -> Dict[str, Any]:
        raise RuntimeError("the browser layer blew up")

    monkeypatch.setattr(BT, "render_url_html", _explode)
    try:
        main, note = _escalate_to_rendered_dom("https://example.com/x", "empty_content", "auto")
    except Exception as exc:  # pragma: no cover - this is the assertion
        pytest.fail(
            f"_escalate_to_rendered_dom raised {type(exc).__name__}: {exc} — its docstring "
            "promises a (None, note) pair. `render_url_html` promises never to raise, but the "
            "call is unguarded, so an OSError from process creation (fd/fork exhaustion under "
            "an agent running tools in parallel) escapes."
        )
    assert main is None
    assert isinstance(note, str) and note


def test_a_browser_layer_exception_is_not_a_silent_empty_success(
    allow_loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED (attack): the escape hatch above lands in the WRONG handler.

    A raise inside `render_url_html` unwinds into the extraction block's
    `except Exception`. Measured: `success=True`, `content=None`,
    `error_class=None`, `extraction_error='OSError: simulated fork exhaustion'`.

    The new `extraction_error` key does its job — the diagnostic survives — but
    the envelope still claims success while carrying nothing, which is the
    never-empty contract violated from the other side. An envelope that recorded
    an extraction failure and produced no content is a failure.
    """
    monkeypatch.setattr(
        BT,
        "render_url_html",
        lambda url, **kw: (_ for _ in ()).throw(OSError("simulated fork exhaustion")),
    )
    result = fetch_url(url=f"{allow_loopback}/spa", timeout=15)

    if result.get("extraction_error") and not str(result.get("content") or "").strip():
        assert result.get("success") is False, (
            "success=True with no content and "
            f"extraction_error={result.get('extraction_error')!r} — a silent empty success"
        )
        assert result.get("error_class"), "an extraction failure produced no branchable error_class"


# ===========================================================================
# G. LIVE — the three classes, against the real internet
# ===========================================================================

# Class (a): genuinely client-rendered with no embedded state. A browser is the
#            only answer.
# Class (b): client-rendered in the browser, but the DATA is already in the
#            server bytes (__NEXT_DATA__, JSON-LD, an RSS link). A browser is
#            overkill and, measured, adds literally nothing.
# Class (c): not a rendering problem — bot mitigation. A headless browser with
#            the same honest identity is blocked too, usually harder.
#
# Measured 2026-08-21, static extraction vs the same page rendered headlessly
# and put through the SAME pipeline:
#
#   class  url                          static   rendered   delta
#   (a)    excalidraw.com                    0        189    +189  (a storage disclaimer)
#   (a)    hetzner.com/cloud             7,357      7,371     +14
#   (a)    vercel.com/blog             *51,481    (n/a)         0  *serves text/markdown to Accept: */*
#   (b)    notion.so/blog                3,616      3,616       0
#   (b)    nasa.gov/news                   366        366       0
#   (b)    instagram.com                     0          0       0  (logged out: there is no content)
#   (c)    reddit.com/r/programming          0        130    worse (a network-block notice)
#   (c)    pubmed.../33199877              203 -> rendered HTTP 403
#   (c)    imdb.com/title/tt0111161        202 -> rendered HTTP 403
#
# On this corpus the escalation changes the outcome on exactly ONE page.

LIVE_CLASS_A = ["https://excalidraw.com/"]
LIVE_CLASS_B = [
    "https://www.notion.so/blog",
    "https://www.nasa.gov/news/",
    "https://www.hetzner.com/cloud/",
]
LIVE_CLASS_C = [
    "https://www.reddit.com/r/programming/",
    "https://pubmed.ncbi.nlm.nih.gov/33199877/",
    "https://www.imdb.com/title/tt0111161/",
]


@live_only
@needs_browser
@pytest.mark.parametrize("url", LIVE_CLASS_A)
def test_live_class_a_client_rendered_page_comes_back_with_real_content(url: str) -> None:
    """Class (a): after escalation, real content, not a stub.

    RED today for excalidraw: it escalates, renders 60 KB of DOM, and yields 189
    characters of browser-storage disclaimer — under the module's own
    real-content floor. "Not empty" is not the bar; "not thin" is.
    """
    result = fetch_url(url=url, timeout=60)
    content = str(result.get("content") or "")
    if result.get("success"):
        assert result.get("rendered_with_browser") is True, (
            f"{url} succeeded statically — it is no longer a class-(a) example"
        )
        assert len(content) >= _MIN_REAL_CONTENT_CHARS, (
            f"{url} rendered but produced only {len(content)} chars: {content[:160]!r}"
        )
    else:
        assert result.get("error_class"), f"{url} failed with no error_class"


@live_only
@pytest.mark.parametrize("url", LIVE_CLASS_B)
def test_live_class_b_is_answered_without_launching_a_browser(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Class (b): the data is in the server bytes. A browser is the wrong tool.

    Measured: rendering these pages changes the extracted content by ZERO
    characters, so a launch is pure cost. The answer is to mine what is already
    there (`__NEXT_DATA__`, JSON-LD, the RSS `<link rel=alternate>`), not to
    start Chromium.
    """
    launches: list[str] = []
    real = BT.render_url_html
    monkeypatch.setattr(
        BT, "render_url_html", lambda u, **kw: (launches.append(u), real(u, **kw))[1]
    )
    result = fetch_url(url=url, timeout=60)

    assert launches == [], f"{url} launched a browser although its data ships in the HTML: {launches}"
    assert result.get("success") is True, f"{url}: {result.get('error')}"


@live_only
@pytest.mark.parametrize("url", LIVE_CLASS_C)
def test_live_class_c_bot_mitigation_fails_loudly_and_accurately(url: str) -> None:
    """Class (c): a block must be reported AS a block, never as content and
    never as a rendering failure the caller could fix by retrying."""
    result = fetch_url(url=url, timeout=60)

    assert result.get("success") is False, (
        f"{url} returned success with {len(str(result.get('content') or ''))} chars: "
        f"{str(result.get('content'))[:160]!r}"
    )
    # The terminal classes (2026-09-22) are MORE precise than the old set: once a
    # real browser has run the page and been refused, the result names the
    # refusal (reddit's listing now answers `captcha_required`) instead of the
    # generic `empty_content`.
    assert result.get("error_class") in {
        "bot_challenge", "empty_content", "empty_body", "client_error",
        "captcha_required", "blocked_by_site", "login_required", "paywall",
    }
    suggestions = " ".join(str(s) for s in (result.get("suggestions") or []))
    assert suggestions.strip(), f"{url} failed with no suggestions"


@live_only
@needs_browser
@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="pgrep-based accounting")
def test_live_a_normal_article_fetch_launches_no_browser_and_leaks_nothing() -> None:
    """The cost claim, against the real internet."""
    baseline = _chromium_processes()
    for url in (
        "https://danluu.com/why-benchmark/",
        "https://simonwillison.net/2024/Dec/31/llms-in-2024/",
        "https://lwn.net/Articles/1000000/",
    ):
        result = fetch_url(url=url, timeout=45)
        if result.get("success"):
            assert result.get("rendered_with_browser") is False, f"{url} escalated needlessly"
    assert _settle_to(baseline) <= baseline
