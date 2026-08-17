"""Browser render-probe tool (Playwright-backed, optional `browser` extra).

`browser_probe` answers ONE question reliably: "does this page/HTML actually
render?" — the verification gap behind blank-page incidents (an agent writes
an HTML/JS app, `read_file` shows plausible source, and nobody notices the
page renders empty; OpenAI's codex hit the same class, issue #14755).

Design (research report untracked/browser_harnessing_best_practices.md,
operator-approved dm:core--laurent#21):

- PLAYWRIGHT + HEADLESS CHROMIUM SHELL: the 2026 cross-OS default (one API
  on macOS/Windows/Linux; auto-waiting kills manual-wait flakiness). The
  dependency is an OPTIONAL extra (`pip install "abstractcore[browser]"`)
  plus a browser-binary step (`python -m playwright install --only-shell
  chromium`) — never a base-install cost; missing pieces produce actionable
  install hints, never tracebacks.
- SUBPROCESS ISOLATION, HARD DEADLINE: the whole probe runs in a worker
  subprocess killed at `timeout_s` + fixed grace. This is the only real
  "never hangs" guarantee — a page whose main thread is blocked (infinite
  JS loop) makes in-process CDP calls like `evaluate` hang with NO timeout
  parameter, and Playwright's sync API objects cannot be closed from
  another thread. Process-group kill also guarantees "never leaks a
  headless browser" and sidesteps the sync-API-inside-asyncio-loop refusal.
- READINESS = CONTENT SIGNALS, never `networkidle`/sleep: `networkidle` is
  REFUSED with teaching (background polling/websockets make it flaky-by-
  design); non-blank means visible text OR visual elements (canvas/svg/img/
  video — a WebGL game has zero innerText and is not blank). A navigation
  timeout degrades to a readyState snapshot of what DID render (the codex
  #14755 fallback) instead of a hard fail; the requested checks decide.
- ONE WALL-CLOCK BUDGET: navigation and every check share `timeout_s`
  (clamped 1..120); console + uncaught-exception listeners attach BEFORE
  navigation; each probe uses a fresh browser, closed in the child and
  killed with it.

Security posture (risk facts: mutating=False, remote_write_capable=True —
same class as fetch_url, NEVER read-only-safe): navigating to a
model-controlled URL executes that page's JavaScript headlessly, and page JS
can send requests anywhere. LOCAL file targets therefore BLOCK all network
requests by default (`allow_network=False`) — a just-generated page must not
phone home (the classic `img src` exfiltration vector); blocked requests are
reported. Screenshots are written only to a fresh temp directory (never a
model-controlled path), so there is no local-write surface.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import unquote, urlparse

from .core import tool

# Tri-state import probe (fetch_url's _ensure_requests pattern): None =
# unknown, then cached. Tests flip this directly to simulate absence.
_PLAYWRIGHT_AVAILABLE: Optional[bool] = None

_PIP_HINT = 'pip install "abstractcore[browser]"'
_BROWSER_HINT = "python -m playwright install --only-shell chromium"
_LINUX_HINT = "python -m playwright install --with-deps chromium"
_OFFLINE_HINT = (
    "offline/firewalled hosts: set PLAYWRIGHT_DOWNLOAD_HOST to an internal mirror, "
    "or PLAYWRIGHT_BROWSERS_PATH to a pre-provisioned browser location"
)

_VALID_WAIT_UNTIL = ("load", "domcontentloaded", "commit")

# Hard-kill grace on top of timeout_s: browser launch + teardown + child
# interpreter import. The probe's own waits are bounded by timeout_s; the
# grace only absorbs fixed process overhead.
_LAUNCH_GRACE_S = 12.0

_MAX_CONSOLE_ENTRIES = 15
_MAX_CONSOLE_CHARS = 300
_MAX_BLOCKED_SHOWN = 8
_TEXT_EXCERPT_CHARS = 240

# JS predicates (content-signal readiness). innerText reflects VISIBILITY
# (display:none text excluded) — the right predicate vs textContent; the
# element fallback keeps canvas/svg/img/video-only pages honest.
_NONBLANK_JS = """
() => {
  const b = document.body;
  if (!b) return false;
  if (((b.innerText) || '').trim().length > 0) return true;
  return document.querySelectorAll('canvas, svg, img, video, embed, object').length > 0;
}
"""
_TEXT_INCLUDES_JS = "t => (((document.body && document.body.innerText) || '')).includes(t)"


def _cors_signature(text: str) -> bool:
    """True when the text is Chromium's file://-origin CORS error (P0).

    Two signals: the `origin 'null'` blocked-request phrasing, and Chromium's
    exact file-scheme sentence. The latter is the FULL phrase 'only supported
    for protocol schemes' (a bare 'protocol schemes' substring is page-
    spoofable — a hostile page could log it to force the note, P2-D-b)."""
    low = str(text).lower()
    if "origin 'null'" in low and ("cors" in low or "err_failed" in low):
        return True
    if "only supported for protocol schemes" in low:
        return True
    return False


# One screenshot dir per PROCESS, created on first use (P2-3: mkdtemp-per-call
# leaked a fresh temp dir every probe across a long-lived agent session). Files
# within it carry unique names so a later probe never clobbers a path already
# handed to analyze_media.
_SCREENSHOT_DIR: Optional[str] = None
_SCREENSHOT_DIR_LOCK = threading.Lock()


def _shared_screenshot_dir() -> str:
    # Locked (P2-F): agent hosts run tools on worker threads; a check-then-set
    # race would mkdtemp twice and leak the loser's dir.
    global _SCREENSHOT_DIR
    with _SCREENSHOT_DIR_LOCK:
        if _SCREENSHOT_DIR is None or not os.path.isdir(_SCREENSHOT_DIR):
            _SCREENSHOT_DIR = tempfile.mkdtemp(prefix="abstractcore_browser_probe_")
        return _SCREENSHOT_DIR


def _ensure_playwright() -> bool:
    global _PLAYWRIGHT_AVAILABLE
    if _PLAYWRIGHT_AVAILABLE is not None:
        return _PLAYWRIGHT_AVAILABLE
    try:
        import playwright.sync_api  # noqa: F401

        _PLAYWRIGHT_AVAILABLE = True
    except Exception:
        _PLAYWRIGHT_AVAILABLE = False
    return _PLAYWRIGHT_AVAILABLE


def _resolve_target(target: str) -> Dict[str, Any]:
    """Resolve target into {url, is_local} or {error}.

    Accepted: http(s):// URLs, file:// URLs, and local file paths (the
    write_file → browser_probe flow). Other schemes are refused by name —
    a tight contract beats silently coercing e.g. javascript: targets.
    """
    t = (target or "").strip()
    if not t:
        return {"error": "target is empty — pass an http(s):// URL or a local HTML file path"}
    lowered = t.lower()
    if lowered.startswith(("http://", "https://")):
        return {"url": t, "is_local": False}
    if lowered.startswith("file://"):
        p = Path(unquote(urlparse(t).path))
        if not p.exists():
            return {"error": f"Target file not found: {p} (from file:// URL)"}
        if p.is_dir():
            return {"error": f"Target is a directory, not a page: {p}"}
        return {"url": p.resolve().as_uri(), "is_local": True}
    if "://" in lowered or lowered.startswith(("javascript:", "data:", "about:")):
        scheme = t.split(":", 1)[0]
        return {
            "error": (
                f"Unsupported target scheme '{scheme}:' — browser_probe accepts http(s):// URLs, "
                "file:// URLs, and local file paths only"
            )
        }
    p = Path(t).expanduser()
    if not p.exists():
        return {
            "error": (
                f"Target file not found: {t} — pass an http(s):// URL or an existing local "
                "HTML file path (write the file first, then probe it)"
            )
        }
    if p.is_dir():
        return {"error": f"Target is a directory, not a page: {t} (point at a specific .html file)"}
    return {"url": p.resolve().as_uri(), "is_local": True}


def _parse_viewport(viewport: str) -> Any:
    try:
        w_s, h_s = str(viewport).lower().replace(" ", "").split("x", 1)
        w, h = int(w_s), int(h_s)
    except Exception:
        return None
    if not (64 <= w <= 4096 and 64 <= h <= 4096):
        return None
    return {"width": w, "height": h}


# --------------------------------------------------------------------------
# Child-process side: the actual Playwright probe. Runs isolated so the
# parent can enforce the wall-clock guarantee with a process-group kill.
# --------------------------------------------------------------------------

def _run_probe(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the probe per config; returns a JSON-serializable result.

    Never raises for page-level problems (dead server, blank page, console
    errors) — those are REPORT content. Raises only for infrastructure
    failures the caller maps to install hints (import/launch).
    """
    from playwright.sync_api import Error as PWError  # type: ignore
    from playwright.sync_api import TimeoutError as PWTimeout  # type: ignore
    from playwright.sync_api import sync_playwright  # type: ignore

    url: str = cfg["url"]
    is_local: bool = cfg["is_local"]
    timeout_s: float = cfg["timeout_s"]
    proc_started = time.monotonic()
    # `deadline` is set AFTER launch (below): timeout_s is the page-behavior
    # budget the docstring promises (navigation + checks), NOT browser cold
    # start. Launch is absorbed by the parent's launch grace + hard kill, so
    # a slow cold start under parallel load never eats the page budget.
    deadline = proc_started + timeout_s  # provisional; reset post-launch

    def remaining_ms(floor_ms: float = 250.0) -> float:
        return max(floor_ms, (deadline - time.monotonic()) * 1000.0)

    result: Dict[str, Any] = {
        "url": url,
        "final_url": None,  # where navigation actually LANDED (redirect disclosure)
        "is_local": is_local,
        "checks": [],
        "console_errors": [],
        "page_errors": [],
        "blocked_requests": [],
        "nav": {"timed_out": False, "elapsed_s": None, "error": None},
        "http_status": None,
        "ready_state": None,
        "title": None,
        "visible_text_len": None,
        "text_excerpt": None,
        "visual_elements": None,
        "frame_count": None,  # >1 ⇒ iframes present (nonblank sees TOP frame only)
        "screenshot": None,
        # Set at capture time on the UNTRUNCATED console text — the stored
        # error is clipped to _MAX_CONSOLE_CHARS, which can cut "origin 'null'"
        # out for long script paths, so re-detecting from stored text would
        # false-negative (P2-D-a). The report prefers this flag.
        "local_cors_detected": False,
        # The requested checks + wait_until + allow_network, echoed so the
        # report can qualify a check-less PASS, disclose iframe scope, name the
        # real wait_until in the readyState-fallback line, and never claim the
        # network was blocked when allow_network let requests out (P1-B).
        "request": {
            "require_nonblank": bool(cfg["require_nonblank"]),
            "expect_selector": cfg["expect_selector"],
            "expect_text": cfg["expect_text"],
            "wait_until": cfg["wait_until"],
            "allow_network": bool(cfg["allow_network"]),
        },
    }

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except PWError as e:
            msg = str(e)
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                return {"error": {"kind": "browser_missing", "message": msg[:400]}}
            return {"error": {"kind": "launch_failed", "message": msg[:400]}}
        # Start the page-behavior budget clock now that the browser is up.
        deadline = time.monotonic() + timeout_s
        try:
            context = browser.new_context(viewport=cfg["viewport"])
            page = context.new_page()

            console_errors: List[str] = result["console_errors"]
            page_errors: List[str] = result["page_errors"]
            blocked: List[str] = result["blocked_requests"]

            def _on_console(msg: Any) -> None:
                try:
                    if msg.type == "error":
                        text = str(msg.text)
                        # Detect the file:// CORS signature on the FULL text
                        # (before truncation) so a long script path can't clip
                        # the signal out (P2-D-a).
                        if is_local and _cors_signature(text):
                            result["local_cors_detected"] = True
                        if len(console_errors) < _MAX_CONSOLE_ENTRIES:
                            loc = msg.location or {}
                            where = f" (at {loc.get('url', '?')}:{loc.get('lineNumber', '?')})"
                            entry = text[:_MAX_CONSOLE_CHARS] + ("…[truncated]" if len(text) > _MAX_CONSOLE_CHARS else "")
                            console_errors.append(entry + where)
                except Exception:
                    pass

            def _on_pageerror(err: Any) -> None:
                try:
                    if len(page_errors) < _MAX_CONSOLE_ENTRIES:
                        text = str(err)
                        page_errors.append(
                            text[:_MAX_CONSOLE_CHARS] + ("…[truncated]" if len(text) > _MAX_CONSOLE_CHARS else "")
                        )
                except Exception:
                    pass

            # Listeners BEFORE navigation — a page can render yet be broken;
            # errors fired during load are the tell.
            page.on("console", _on_console)
            page.on("pageerror", _on_pageerror)

            if is_local and not cfg["allow_network"]:
                # Local pages must not phone home: allow only file/data/blob/
                # about; abort (and report) everything else.
                def _route(route: Any) -> None:
                    try:
                        r_url = route.request.url
                        if r_url.startswith(("file://", "data:", "about:", "blob:")):
                            route.continue_()
                        else:
                            if len(blocked) < 50:
                                blocked.append(r_url)
                            route.abort("blockedbyclient")
                    except Exception:
                        try:
                            route.continue_()
                        except Exception:
                            pass

                context.route("**/*", _route)

            nav_started = time.monotonic()
            response = None
            try:
                response = page.goto(url, wait_until=cfg["wait_until"], timeout=remaining_ms())
            except PWTimeout:
                # readyState fallback (codex #14755): report what DID render;
                # the requested checks decide the verdict.
                result["nav"]["timed_out"] = True
                # P2-B: if NOTHING committed (still about:blank), the server
                # accepted the connection but never sent a response. The
                # checks have nothing to inspect and the snapshot evaluate()
                # would HANG (no CDP timeout while a navigation is pending),
                # eating budget+grace into a hard-kill misdiagnosed as an
                # "infinite JS loop". Report honestly and skip them.
                try:
                    committed = page.url
                except Exception:
                    committed = ""
                if not committed or committed == "about:blank":
                    result["nav"]["no_commit"] = True
                    result["nav"]["elapsed_s"] = round(time.monotonic() - nav_started, 2)
                    return result
            except PWError as e:
                # Dead server / DNS failure / refused connection: a verdict,
                # not an exception — the agent probing its own server needs
                # FAIL + reason. Strip Playwright's multi-line "Call log"
                # trailer (noise in a one-line report).
                nav_err = str(e).split("\nCall log", 1)[0].strip()
                result["nav"]["error"] = nav_err[:400]
                result["checks"].append(
                    {"name": "navigation", "ok": False, "detail": nav_err[:300], "elapsed_s": round(time.monotonic() - nav_started, 2)}
                )
                return result
            result["nav"]["elapsed_s"] = round(time.monotonic() - nav_started, 2)
            if response is not None:
                result["http_status"] = response.status

            if cfg["require_nonblank"]:
                t0 = time.monotonic()
                try:
                    page.wait_for_function(_NONBLANK_JS, timeout=remaining_ms())
                    result["checks"].append(
                        {"name": "nonblank", "ok": True, "detail": "visible content present", "elapsed_s": round(time.monotonic() - t0, 2)}
                    )
                except PWTimeout:
                    result["checks"].append(
                        {
                            "name": "nonblank",
                            "ok": False,
                            "detail": "page stayed blank within budget (no visible text, no canvas/svg/img/video)",
                            "elapsed_s": round(time.monotonic() - t0, 2),
                        }
                    )
                except PWError as e:
                    result["checks"].append(
                        {"name": "nonblank", "ok": False, "detail": f"check failed: {str(e)[:200]}", "elapsed_s": round(time.monotonic() - t0, 2)}
                    )

            if cfg["expect_selector"]:
                t0 = time.monotonic()
                sel = cfg["expect_selector"]
                try:
                    page.wait_for_selector(sel, state="visible", timeout=remaining_ms())
                    result["checks"].append(
                        {"name": f"selector {sel!r} visible", "ok": True, "detail": "", "elapsed_s": round(time.monotonic() - t0, 2)}
                    )
                except PWTimeout:
                    result["checks"].append(
                        {"name": f"selector {sel!r} visible", "ok": False, "detail": "not visible within budget", "elapsed_s": round(time.monotonic() - t0, 2)}
                    )
                except PWError as e:
                    result["checks"].append(
                        {"name": f"selector {sel!r} visible", "ok": False, "detail": f"invalid selector or check failed: {str(e)[:200]}", "elapsed_s": round(time.monotonic() - t0, 2)}
                    )

            if cfg["expect_text"]:
                t0 = time.monotonic()
                txt = cfg["expect_text"]
                shown = txt if len(txt) <= 60 else txt[:60] + "…"
                try:
                    page.wait_for_function(_TEXT_INCLUDES_JS, arg=txt, timeout=remaining_ms())
                    result["checks"].append(
                        {"name": f"text {shown!r} present", "ok": True, "detail": "", "elapsed_s": round(time.monotonic() - t0, 2)}
                    )
                except PWTimeout:
                    result["checks"].append(
                        {
                            "name": f"text {shown!r} present",
                            "ok": False,
                            "detail": "not found in visible DOM text within budget",
                            "elapsed_s": round(time.monotonic() - t0, 2),
                        }
                    )
                except PWError as e:
                    result["checks"].append(
                        {"name": f"text {shown!r} present", "ok": False, "detail": f"check failed: {str(e)[:200]}", "elapsed_s": round(time.monotonic() - t0, 2)}
                    )

            # Snapshot (each field independent — a crashed frame must not
            # void the rest of the report). On a blocked main thread these
            # CDP calls hang; the parent's process kill is the guarantee.
            try:
                result["ready_state"] = page.evaluate("() => document.readyState")
            except Exception:
                pass
            try:
                result["title"] = page.title()
            except Exception:
                pass
            try:
                text = page.evaluate("() => ((document.body && document.body.innerText) || '')")
                result["visible_text_len"] = len(text)
                excerpt = " ".join(text.split())[:_TEXT_EXCERPT_CHARS]
                result["text_excerpt"] = excerpt
            except Exception:
                pass
            try:
                result["visual_elements"] = page.evaluate(
                    "() => document.querySelectorAll('canvas, svg, img, video, embed, object').length"
                )
            except Exception:
                pass
            try:
                # >1 frame ⇒ the page hosts iframe(s); the nonblank predicate
                # and all checks see the TOP frame only, so a blank verdict on
                # an iframe-hosted app must disclose that scope.
                result["frame_count"] = len(page.frames)
            except Exception:
                pass
            try:
                # Captured HERE (after the checks) so a LATE commit / client-side
                # redirect during the waits is reflected — capturing right after
                # goto() gave 'about:blank' for a not-yet-committed page (P1-A).
                # Chromium auto-follows 3xx, so this discloses where navigation
                # actually landed. page.url is a cached property (no CDP hang).
                result["final_url"] = page.url
            except Exception:
                pass

            if cfg.get("screenshot_dir"):
                try:
                    shot = Path(cfg["screenshot_dir"]) / (cfg.get("screenshot_name") or "probe.png")
                    page.screenshot(path=str(shot), full_page=False)
                    result["screenshot"] = {"path": str(shot), "bytes": shot.stat().st_size}
                except Exception as e:
                    result["screenshot"] = {"error": str(e)[:200]}
        finally:
            try:
                browser.close()
            except Exception:
                pass

    result["total_s"] = round(time.monotonic() - proc_started, 2)
    return result


def _subprocess_main() -> None:
    """Child entrypoint: JSON config on stdin → JSON result on stdout."""
    cfg = json.loads(sys.stdin.read())
    try:
        out = _run_probe(cfg)
    except Exception as e:  # infrastructure failure — parent renders hints
        out = {"error": {"kind": "probe_crashed", "message": f"{type(e).__name__}: {str(e)[:400]}"}}
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Parent side: validation, subprocess management, report formatting.
# --------------------------------------------------------------------------

# Process-tree kill lives in the shared module (execute_command needs the same
# leak-proof kill for its shell-child grandchildren, runtime c5004). Aliased to
# the historical names so this module's call sites + tests are unchanged.
from .process_tree import descendant_pids as _descendant_pids  # noqa: E402
from .process_tree import hard_kill_tree as _hard_kill_tree  # noqa: E402


def _spawn_probe(cfg: Dict[str, Any], hard_budget_s: float) -> Dict[str, Any]:
    """Run _run_probe in a worker subprocess with a hard process-TREE kill."""
    cmd = [sys.executable, "-c", "from abstractcore.tools.browser_tools import _subprocess_main; _subprocess_main()"]
    popen_kwargs: Dict[str, Any] = dict(
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True  # own group + a stable ancestry root to walk
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(input=json.dumps(cfg), timeout=hard_budget_s)
    except subprocess.TimeoutExpired:
        # The only real "never hangs / never leaks" guarantee: kill the whole
        # process TREE — the browser escapes the worker's group via setsid(),
        # so a group-only kill would orphan chrome-headless-shell.
        _hard_kill_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return {
            "error": {
                "kind": "hard_timeout",
                "message": (
                    f"probe exceeded its wall-clock budget ({hard_budget_s:.0f}s incl. launch grace) and was killed — "
                    "possible causes: the page's main thread is blocked (e.g. an infinite JS loop), the browser could "
                    "not start in time, or the host is under heavy load. Retry with a larger timeout_s to distinguish"
                ),
            }
        }
    if proc.returncode != 0:
        tail = (stderr or "")[-500:]
        if "Executable doesn't exist" in (stderr or "") or "playwright install" in (stderr or ""):
            return {"error": {"kind": "browser_missing", "message": tail}}
        return {"error": {"kind": "probe_crashed", "message": f"worker exited {proc.returncode}: {tail}"}}
    try:
        return json.loads(stdout or "")
    except Exception:
        return {"error": {"kind": "probe_crashed", "message": f"worker returned non-JSON output: {(stdout or '')[:200]!r}; stderr: {(stderr or '')[-300:]}"}}


def _install_message(kind: str, detail: str = "") -> str:
    if kind == "package_missing":
        return (
            "❌ Missing dependency: `playwright`\n"
            "browser_probe renders pages in a headless browser via Playwright.\n"
            f"Install (2 steps): {_PIP_HINT}\n"
            f"then download the headless browser: {_BROWSER_HINT}\n"
            f"(Linux also needs system libs: {_LINUX_HINT}; {_OFFLINE_HINT})"
        )
    return (
        "❌ Browser binary missing: Playwright is installed but its Chromium headless shell is not.\n"
        f"Download it with: {_BROWSER_HINT}\n"
        f"(Linux also needs system libs: {_LINUX_HINT}; {_OFFLINE_HINT})"
        + (f"\nDetail: {detail}" if detail else "")
    )


def _urls_equivalent(a: str, b: str) -> bool:
    """True if two URLs differ only by browser NORMALIZATION, not a redirect (P1-A).

    Chromium adds a trailing slash to a bare host, percent-encodes spaces, and
    lowercases scheme/host — none of which is a real redirect. Comparing raw
    strings fired 'Redirected to:' on the tool's own `http://127.0.0.1:8000`
    example. Compare scheme+netloc (lowercased), unquoted path (empty→'/'), and
    query; ignore the fragment."""
    try:
        pa, pb = urlparse(a), urlparse(b)
    except Exception:
        return a == b

    def _norm(p: Any) -> tuple:
        return (p.scheme.lower(), p.netloc.lower(), unquote(p.path) or "/", p.query)

    return _norm(pa) == _norm(pb)


def _local_file_cors_blocked(is_local: bool, console_errors: List[str], page_errors: List[str]) -> bool:
    """Text-based fallback for the file://-origin CORS signature (P0).

    Chromium blocks `<script type="module">` and `fetch()` from a file://
    origin ('null'), so a modern ES-module/Vite app renders EMPTY as a local
    file though it works fine over http. The child sets a capture-time flag on
    the UNTRUNCATED error; this text scan is the belt for pre-flag results."""
    if not is_local:
        return False
    return any(_cors_signature(e) for e in list(console_errors) + list(page_errors))


def _format_report(res: Dict[str, Any], *, timeout_s: float) -> str:
    checks: List[Dict[str, Any]] = res.get("checks") or []
    console_errors: List[str] = res.get("console_errors") or []
    page_errors: List[str] = res.get("page_errors") or []
    blocked: List[str] = res.get("blocked_requests") or []
    nav = res.get("nav") or {}
    status = res.get("http_status")
    req = res.get("request") or {}
    is_local = bool(res.get("is_local"))

    failures = [c for c in checks if not c.get("ok")]
    http_fail = status is not None and int(status) >= 400
    no_commit = bool(nav.get("no_commit"))
    ok = not failures and not http_fail and not nav.get("error") and not no_commit

    # P1-2: a PASS with NO content assertion (require_nonblank off + no
    # expect_selector/expect_text) proves only that navigation succeeded — it
    # must NOT read as "the page renders". Qualify it so an LLM keying on the
    # first line cannot over-claim.
    no_content_assertion = not checks and not req.get("require_nonblank") and not req.get("expect_selector") and not req.get("expect_text")

    # P0: file:// CORS blank — the page didn't render as a local file because
    # Chromium blocked its modules/fetch, NOT because the agent's code is broken.
    # Prefer the child's capture-time flag (set on untruncated text); fall back
    # to a text scan for pre-flag results.
    local_cors = bool(res.get("local_cors_detected")) or _local_file_cors_blocked(is_local, console_errors, page_errors)
    nonblank_failed = any(c.get("name") == "nonblank" and not c.get("ok") for c in checks)
    text_failed = any(str(c.get("name") or "").startswith("text ") and not c.get("ok") for c in checks)
    visual_elements = res.get("visual_elements")
    has_visual_elements = isinstance(visual_elements, int) and visual_elements > 0

    lines: List[str] = []
    verdict = "PASS" if ok else "FAIL"
    reason = ""
    if not ok:
        if nav.get("error"):
            reason = " — navigation failed"
        elif no_commit:
            reason = " — no response committed within budget (server accepted the connection but sent no page)"
        elif http_fail:
            reason = f" — HTTP {status}"
        elif local_cors and nonblank_failed:
            reason = " — blank as a local file (likely file:// CORS, not your code — see note)"
        elif failures:
            reason = f" — {len(failures)} check(s) failed"
    elif no_content_assertion:
        # Verdict is PASS but nothing about rendering was asserted. P2-A: if
        # navigation itself did not complete, say so — never a bare "renders".
        if nav.get("timed_out"):
            reason = " (navigation incomplete — no content assertions requested)"
        else:
            reason = " (navigation only — no content assertions requested)"
    lines.append(f"Browser probe: {verdict}{reason}")

    kind = "local file" if is_local else "url"
    net_note = ""
    if is_local:
        if req.get("allow_network"):
            # P1-B: no route was installed — do NOT claim the sandbox was in
            # force (that would hide a real phone-home in the exact posture the
            # guard is named for).
            net_note = "; network ALLOWED (allow_network=true — outbound requests were NOT blocked)"
        elif blocked:
            shown = ", ".join(blocked[:_MAX_BLOCKED_SHOWN])
            more = f" (+{len(blocked) - _MAX_BLOCKED_SHOWN} more)" if len(blocked) > _MAX_BLOCKED_SHOWN else ""
            net_note = f"; network BLOCKED — {len(blocked)} outbound request(s) refused: {shown}{more}"
        else:
            net_note = "; network blocked (no outbound attempts)"
    lines.append(f"Target: {res.get('url')} ({kind}{net_note})")

    # P1-4: disclose where navigation actually LANDED — a silent 3xx to a
    # login/setup/error page renders + PASSes on the wrong page otherwise.
    # P1-A: 'about:blank' is a not-yet-committed page (no redirect), and only
    # a difference beyond browser URL-normalization counts.
    final_url = res.get("final_url")
    if (
        final_url
        and not is_local
        and str(final_url) != "about:blank"
        and not _urls_equivalent(str(final_url), str(res.get("url") or ""))
    ):
        lines.append(f"Redirected to: {final_url} (the requested URL did not serve the page directly)")

    if status is not None:
        lines.append(f"HTTP status: {status}")
    if nav.get("error"):
        lines.append(f"Navigation error: {nav['error']}")
    elif no_commit:
        # P2-B: connection accepted, no response committed — distinct from a
        # partial render (readyState fallback) and from a dead server (error).
        lines.append("Navigation: connection accepted but no response committed within budget (server never sent a page)")
    elif nav.get("timed_out"):
        # P2-1: name the ACTUAL wait_until, never a hardcoded 'load'.
        wu = req.get("wait_until") or "load"
        lines.append(
            f"Navigation: '{wu}' event did not fire within budget — proceeded with the DOM as-is (readyState fallback)"
        )
    elif nav.get("elapsed_s") is not None:
        lines.append(f"Navigation: {nav['elapsed_s']}s")

    meta_bits = []
    if res.get("ready_state"):
        meta_bits.append(f"readyState: {res['ready_state']}")
    if res.get("title"):
        meta_bits.append(f"title: {res['title']!r}")
    if meta_bits:
        lines.append(" | ".join(meta_bits))

    if res.get("visible_text_len") is not None:
        excerpt = res.get("text_excerpt") or ""
        excerpt_part = f' — "{excerpt}"' if excerpt else ""
        lines.append(f"Visible text: {res['visible_text_len']:,} chars{excerpt_part}")
    if res.get("visual_elements") is not None:
        lines.append(f"Visual elements (canvas/svg/img/video/embed): {res['visual_elements']}")

    if checks:
        lines.append("Checks:")
        for c in checks:
            mark = "✓" if c.get("ok") else "✗"
            detail = f" — {c['detail']}" if c.get("detail") else ""
            elapsed = f" ({c['elapsed_s']}s)" if c.get("elapsed_s") is not None else ""
            lines.append(f"  {mark} {c['name']}{detail}{elapsed}")

    if text_failed and has_visual_elements and not nonblank_failed and not local_cors:
        lines.append(
            "\nNote: `expect_text` checks visible DOM text only; it does not read pixels drawn inside "
            "canvas/WebGL/images. This page rendered nonblank and includes visual elements, so if the "
            "expected content is drawn rather than DOM text, switch to `expect_selector` or screenshot/"
            "visual verification instead of retrying the same text probe."
        )

    # P0: the blank was almost certainly a file:// limitation, not the code.
    if local_cors and nonblank_failed:
        lines.append(
            "\nNote: this local file:// origin blocked its ES modules and/or fetch() (CORS — see console errors above). "
            "If the app uses <script type=\"module\"> or fetch, it will render EMPTY as a file but fine over http. "
            "Serve it (e.g. `python -m http.server`) and re-probe the http:// URL."
        )
    # P1-1: nonblank/all checks see the TOP frame only — disclose when the
    # page hosts iframes so an iframe-app blank isn't misread as broken.
    frame_count = res.get("frame_count")
    if nonblank_failed and not local_cors and isinstance(frame_count, int) and frame_count > 1:
        lines.append(
            f"\nNote: this page hosts {frame_count - 1} iframe(s); the non-blank check and selector/text checks "
            "inspect the TOP frame only. If your UI renders inside an iframe, this blank verdict may be wrong — "
            "probe the framed document's URL directly."
        )

    total_errs = len(console_errors) + len(page_errors)
    if total_errs:
        lines.append(
            f"Console: {len(console_errors)} error(s), {len(page_errors)} uncaught exception(s) — a page can render and still be broken:"
        )
        for e in page_errors[:5]:
            lines.append(f"  [uncaught] {e}")
        for e in console_errors[:5]:
            lines.append(f"  [error] {e}")
        if total_errs > 10:
            lines.append(f"  … {total_errs - 10} more not shown")
    else:
        lines.append("Console: clean (no errors, no uncaught exceptions)")

    shot = res.get("screenshot")
    if isinstance(shot, dict):
        if shot.get("path"):
            lines.append(
                f"Screenshot: {shot['path']} ({shot.get('bytes', 0):,} bytes) — pass to analyze_media for a visual check"
            )
        elif shot.get("error"):
            lines.append(f"Screenshot failed: {shot['error']}")

    if res.get("total_s") is not None:
        lines.append(f"Timing: {res['total_s']}s total (budget {timeout_s:.0f}s)")
    return "\n".join(lines)


@tool(
    description="Render a URL or local HTML file in a headless browser and verify it actually displays (non-blank, selector/text present); reports console errors.",
    when_to_use="Use after writing/serving a web page to verify it truly renders (blank-page bugs pass read_file review). Needs browser extra. capture_screenshot declares the shot as media, so a vision-capable model sees it (else use analyze_media).",
    # "executor" (agent seam, commons c4932): opts into the ReAct verifier's
    # declaration-driven execution preference — verification probes PREFER
    # artifact-EXECUTING tools over read-only review (the R-Type lesson: only
    # execution catches what an LLM-read review blesses — the blank-page class
    # this tool exists for). Misdeclaration fails safe both directions.
    tags=["write", "remote_write", "browser", "executor"],
    examples=[
        {
            "description": "Verify a just-written game page renders non-blank",
            "arguments": {"target": "snake/index.html"},
        },
        {
            "description": "Verify a served app shows its board element",
            "arguments": {"target": "http://127.0.0.1:8000", "expect_selector": "#board"},
        },
        {
            "description": "Check text appears and capture a screenshot for visual review",
            "arguments": {"target": "docs/report.html", "expect_text": "Q3 Revenue", "capture_screenshot": True},
        },
    ],
)
def browser_probe(
    target: str,
    require_nonblank: bool = True,
    expect_selector: Optional[str] = None,
    expect_text: Optional[str] = None,
    timeout_s: float = 20.0,
    wait_until: str = "load",
    allow_network: bool = False,
    capture_screenshot: bool = False,
    viewport: str = "1280x720",
) -> Union[str, Dict[str, Any]]:
    """Render a page in a headless browser and verify it displays correctly.

    Args:
        target: http(s):// URL, file:// URL, or local HTML file path. NOTE: a
            local file:// origin CANNOT load ES modules (`<script
            type="module">`) or `fetch()` (browser CORS) — a modern
            module/fetch app renders EMPTY as a file though it works over
            http. Serve those (e.g. `python -m http.server`) and probe the
            http:// URL; the report flags this case when it detects it.
        require_nonblank: Fail if no visible text AND no visual elements
            (canvas/svg/img/video) appear within the budget (default: True).
            Sees the TOP frame only — iframe-hosted content is not inspected.
        expect_selector: Optional CSS selector that must become VISIBLE.
        expect_text: Optional text that must appear in the page's visible DOM
            text. It does NOT read pixels drawn inside canvas/WebGL/images, so
            canvas-heavy pages may need selector- or screenshot-based
            verification instead.
        timeout_s: Total wall-clock budget shared by navigation and all
            checks (default 20, clamped 1..120). The probe never hangs: the
            worker is killed at budget + fixed launch grace.
        wait_until: Navigation readiness event — load | domcontentloaded |
            commit. 'networkidle' is refused (flaky by design: background
            polling/websockets keep the network busy independent of the UI).
        allow_network: For LOCAL file targets only — allow outbound network
            requests (default False: a generated page must not phone home;
            blocked requests are reported). http(s) targets always use the
            network.
        capture_screenshot: Save a viewport PNG to a fresh temp dir and
            report its path (compose with analyze_media for a visual pass).
        viewport: Browser viewport as 'WIDTHxHEIGHT' (default '1280x720').

    Returns:
        A PASS/FAIL report: navigation outcome, HTTP status, readyState,
        title, visible-text stats, per-check results, console errors,
        blocked network attempts (local targets), screenshot path, timing.
    """
    if not _ensure_playwright():
        return _install_message("package_missing")

    resolved = _resolve_target(target)
    if "error" in resolved:
        return f"❌ {resolved['error']}"

    wait_choice = str(wait_until or "load").strip().lower()
    if wait_choice == "networkidle":
        return (
            "❌ wait_until='networkidle' is refused: modern pages keep the network busy (polling, "
            "websockets, analytics) independent of the UI, so networkidle is flaky by design. "
            "Use 'load' (default) or 'domcontentloaded' plus require_nonblank/expect_selector — "
            "content signals, not network silence."
        )
    if wait_choice not in _VALID_WAIT_UNTIL:
        return f"❌ Invalid wait_until '{wait_until}' — valid: {', '.join(_VALID_WAIT_UNTIL)}"

    try:
        budget = float(timeout_s)
    except Exception:
        budget = 20.0
    budget = max(1.0, min(120.0, budget))

    vp = _parse_viewport(viewport)
    if vp is None:
        return f"❌ Invalid viewport '{viewport}' — use 'WIDTHxHEIGHT' between 64x64 and 4096x4096 (e.g. '1280x720')"

    screenshot_dir: Optional[str] = None
    screenshot_name: Optional[str] = None
    if capture_screenshot:
        # One per-process dir (never a model-controlled path — no local-write
        # surface beyond the probe's own artifact), unique file per call.
        screenshot_dir = _shared_screenshot_dir()
        screenshot_name = f"probe_{uuid.uuid4().hex[:12]}.png"

    cfg: Dict[str, Any] = {
        "url": resolved["url"],
        "is_local": bool(resolved["is_local"]),
        "require_nonblank": bool(require_nonblank),
        "expect_selector": expect_selector or None,
        "expect_text": expect_text or None,
        "timeout_s": budget,
        "wait_until": wait_choice,
        "allow_network": bool(allow_network),
        "viewport": vp,
        "screenshot_dir": screenshot_dir,
        "screenshot_name": screenshot_name,
    }

    res = _spawn_probe(cfg, hard_budget_s=budget + _LAUNCH_GRACE_S)

    err = res.get("error")
    if err:
        kind = err.get("kind")
        if kind == "browser_missing":
            return _install_message("browser_missing", err.get("message", ""))
        if kind == "hard_timeout":
            return f"Browser probe: FAIL — {err.get('message')}"
        return f"❌ Browser probe could not run ({kind}): {err.get('message')}"

    report = _format_report(res, timeout_s=budget)

    # Declared-media sight lane (commons c3969 shape A, the camera-tools pattern): when a
    # screenshot was captured, DECLARE it as a `media` output so a host that supports the
    # sight lane (agent adapters) attaches it to the NEXT model call — a vision-capable main
    # model then sees the render NATIVELY, with no separate vision-fallback config. The report
    # rides `rendered` (identical text), so non-sight-lane hosts show the same string. Opt-in
    # by construction: the dict is returned ONLY when the caller asked for a screenshot AND one
    # was actually written. analyze_media over the fallback route remains the path when the
    # main model has no encoders (audit item 0836 / analyze_media escalation, 2026-07-25).
    shot = res.get("screenshot")
    if isinstance(shot, dict) and shot.get("path"):
        return {"rendered": report, "media": [shot["path"]]}
    return report
