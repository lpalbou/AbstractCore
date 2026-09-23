"""fetch_url's escalation ladder (Mission D, 2026-09-22).

What the operator hit, in their own UI, on pages a human opens fine:
  * reddit threads   -> "No readable content extracted (empty_content)" + "render_empty"
  * aiuntethered.com -> "HTTP Error 403: Forbidden" (Cloudflare "Just a moment...")
  * economist.com    -> "HTTP Error 403" with nothing readable said about the article

The ladder these tests pin, all with STUBBED transports (no network, no browser):
  1. static fetch, honest UA               (unchanged; the fast path)
  2. site adapter, when one covers the URL (reddit -> the thread's own Atom feed)
  3. real-browser render                   (on a JS shell, or a 403/429/451 whose
                                            body is a challenge — never a policy
                                            refusal that explains itself)
  4. the site's own public feed            (title + standfirst, flagged degraded)
  5. an honest, CLASSED failure            (captcha_required / paywall /
                                            login_required / blocked_by_site)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractcore.tools import browser_tools as BT
from abstractcore.tools import common_tools as ct
from abstractcore.tools import fetch_url_ssrf as ssrf

ct._ensure_requests()

ARTICLE_HTML = (
    "<html><head><title>A Real Article</title></head><body><article><h1>A Real Article</h1>"
    + "".join(
        f"<p>Paragraph {i}: the substance of the article that a human reads in their browser, "
        "long enough to clear every real-content floor the extractor applies.</p>"
        for i in range(8)
    )
    + "</article></body></html>"
)
CF_CHALLENGE_403 = (
    "<!DOCTYPE html><html><head><title>Just a moment...</title></head><body>"
    "<script src='https://challenges.cloudflare.com/turnstile/v0/api.js'></script>"
    "<noscript>Enable JavaScript and cookies to continue</noscript></body></html>"
)
POLICY_403 = (
    "<html><body><h1>Access restricted</h1><p>This service is not available in your region "
    "because of licensing restrictions that apply to our content distribution agreements. "
    "Please contact our support team if you believe this is an error in our records.</p></body></html>"
)
JS_SHELL_200 = (
    "<html><head><title>Reddit</title></head><body><div id='root'></div>"
    "<script src='/app.js'></script></body></html>"
)
CAPTCHA_DOM = (
    "<html><body><h1>Prove your humanity</h1><p>We're committed to safety and security. "
    "But not for bots. Complete the challenge below and let us know you're a real person.</p>"
    "<footer>Reddit, Inc. All rights reserved. User Agreement Privacy Policy Content Policy Help "
    "and a long footer that easily clears the two-hundred-character real-content floor.</footer></body></html>"
)
PAYWALL_DOM = (
    "<html><body><article><h1>Big Story</h1><p>The opening line of the big story, which every "
    "visitor can read before the wall comes down across the rest of the text.</p>"
    "<div class='wall'><p>This article is for subscribers. Subscribe to continue reading, or "
    "log in if you are already a subscriber to our award-winning journalism.</p></div>"
    "</article></body></html>"
)
REDDIT_URL = "https://www.reddit.com/r/samharris/comments/1twnyyj/no_ai_is_not_conscious/"
REDDIT_ATOM = (
    '<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
    "<entry><author><name>/u/window-sil</name></author>"
    '<content type="html">&lt;div class=&quot;md&quot;&gt;&lt;p&gt;The submission body text, '
    "which argues the thesis at length.&lt;/p&gt;&lt;/div&gt; &lt;a href=&quot;https://example.org/story&quot;&gt;[link]&lt;/a&gt;"
    '</content><id>t3_1twnyyj</id><link href="' + REDDIT_URL + '" />'
    "<updated>2026-06-04T13:38:37+00:00</updated><title>No, AI Is Not Conscious</title></entry>"
    + "".join(
        f"<entry><author><name>/u/commenter{i}</name></author>"
        f'<content type="html">&lt;div class=&quot;md&quot;&gt;&lt;p&gt;Top-level comment number {i}, '
        f"with an actual opinion in it.&lt;/p&gt;&lt;/div&gt;</content><id>t1_c{i}</id>"
        f'<link href="{REDDIT_URL}c{i}/" /><updated>2026-06-05T10:00:00+00:00</updated>'
        f"<title>/u/commenter{i} on No, AI Is Not Conscious</title></entry>"
        for i in range(5)
    )
    + "</feed>"
)
ECON_URL = "https://www.economist.com/by-invitation/2026/08/20/humanity-has-the-debate"
ECON_RSS = (
    '<?xml version="1.0" encoding="UTF-8"?><rss><channel>'
    "<item><title><![CDATA[Some other piece]]></title><description><![CDATA[Wrong standfirst]]></description>"
    "<link>https://www.economist.com/by-invitation/2026/08/19/other</link></item>"
    "<item><title><![CDATA[Humanity has the debate about AI consciousness backwards]]></title>"
    "<description><![CDATA[We don’t care for others because they’re conscious, argues a writer]]></description>"
    f"<link>{ECON_URL}</link></item></channel></rss>"
)


class _Resp:
    def __init__(self, url: str, status: int, body: str, ctype: str, headers: Optional[dict] = None):
        self.url = url
        self.status_code = status
        self.ok = 200 <= status < 400
        self.reason = "OK" if self.ok else "Forbidden"
        self.headers = {"content-type": ctype, **(headers or {})}
        self.content = body.encode("utf-8")
        self.text = body
        self.encoding = "utf-8"

    def iter_content(self, chunk_size: int = 16384):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


class _CM:
    def __init__(self, r: _Resp):
        self._r = r

    def __enter__(self):
        return self._r

    def __exit__(self, *exc):
        return False


def _install_transport(monkeypatch, routes: Dict[str, Tuple[int, str, str, dict]]) -> List[str]:
    """Route every requests.Session call by exact URL; unknown URLs 404.

    Returns the list of URLs requested, so a test can assert what was (not) tried.
    """
    seen: List[str] = []

    def _answer(url: str) -> _Resp:
        seen.append(url)
        status, body, ctype, headers = routes.get(url, (404, "not found", "text/plain", {}))
        return _Resp(url, status, body, ctype, headers)

    class _Session:
        def __init__(self) -> None:
            self.headers: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def mount(self, *a, **k):
            return None

        def request(self, method: str = "GET", url: str = "", **kw):
            return _CM(_answer(url))

        def get(self, url: str, **kw):
            return _answer(url)

    monkeypatch.setattr(ct.requests, "Session", _Session)
    # Hermetic: no DNS. The SSRF guard has its own suite.
    monkeypatch.setattr(ct, "fetch_url_guard_destination", lambda u: None)
    monkeypatch.setattr(ssrf, "fetch_url_guard_destination", lambda u: None)
    # No retry sleeps.
    monkeypatch.setattr(ct, "_FETCH_URL_RETRY_BASE_DELAY_S", 0.0)
    return seen


def _install_render(monkeypatch, html: str, *, status: int = 200, text: str = "", ok: bool = True) -> List[dict]:
    calls: List[dict] = []

    def _fake(url: str, **kw: Any) -> dict:
        calls.append({"url": url, **kw})
        if not ok:
            return {"ok": False, "error_class": "render_empty", "message": "stub", "hint": ""}
        return {
            "ok": True, "html": html, "text": text, "final_url": url, "status": status,
            "elapsed_s": 0.1, "browser_mode": "persistent:chromium",
        }

    monkeypatch.setattr(BT, "render_url_html", _fake)
    return calls


# --------------------------------------------------------------------------
# rung 3 on a blocked status: challenge 403 -> real browser
# --------------------------------------------------------------------------
def test_a_cloudflare_challenge_403_is_cleared_by_the_real_browser_render(monkeypatch) -> None:
    url = "https://blog.example.com/post/"
    _install_transport(monkeypatch, {url: (403, CF_CHALLENGE_403, "text/html", {"cf-mitigated": "challenge"})})
    calls = _install_render(monkeypatch, ARTICLE_HTML)

    r = ct.fetch_url(url, include_full_content=False, keep_links=False)

    assert r["success"] is True, r.get("rendered")
    assert "substance of the article" in r["content"]
    assert r["rendered_with_browser"] is True
    assert r["static_status_blocked"] == 403
    assert len(calls) == 1 and calls[0].get("thorough") is True, "the escalation must use the full-browser profile"


def test_a_403_that_explains_itself_never_launches_a_browser(monkeypatch) -> None:
    """A policy/geo refusal is not a challenge: a browser changes nothing and
    costs seconds. The cheap path must stay cheap."""
    url = "https://news.example.com/a/"
    _install_transport(monkeypatch, {url: (403, POLICY_403, "text/html", {})})
    calls = _install_render(monkeypatch, ARTICLE_HTML)

    r = ct.fetch_url(url, include_full_content=False, keep_links=False)

    assert r["success"] is False
    assert r["status_code"] == 403
    assert calls == [], "a self-explaining 403 paid for a browser launch"


def test_render_js_never_is_honoured_on_a_blocked_status(monkeypatch) -> None:
    url = "https://blog.example.com/post/"
    _install_transport(monkeypatch, {url: (403, CF_CHALLENGE_403, "text/html", {"cf-mitigated": "challenge"})})
    calls = _install_render(monkeypatch, ARTICLE_HTML)

    r = ct.fetch_url(url, render_js="never")

    assert calls == []
    assert r["success"] is False


# --------------------------------------------------------------------------
# rung 3 on a 200: JS shell -> render
# --------------------------------------------------------------------------
def test_a_js_shell_is_rendered_with_the_full_browser_profile(monkeypatch) -> None:
    url = "https://app.example.com/page"
    _install_transport(monkeypatch, {url: (200, JS_SHELL_200, "text/html", {})})
    calls = _install_render(monkeypatch, ARTICLE_HTML)

    r = ct.fetch_url(url, include_full_content=False, keep_links=False)

    assert r["success"] is True
    assert "substance of the article" in r["content"]
    assert calls and calls[0].get("thorough") is True


def test_shadow_dom_text_is_used_when_the_serialised_dom_is_a_scaffold(monkeypatch) -> None:
    """Web-component pages keep the article in shadow roots: page.content()
    carries none of it; innerText does."""
    url = "https://app.example.com/wc"
    _install_transport(monkeypatch, {url: (200, JS_SHELL_200, "text/html", {})})
    visible = "\n".join(f"Shadow paragraph {i} with real reading material in it." for i in range(12))
    _install_render(monkeypatch, "<html><body><my-app></my-app></body></html>", text=visible)

    r = ct.fetch_url(url, include_full_content=False, keep_links=False)

    assert r["success"] is True
    assert "Shadow paragraph 7" in r["content"]


# --------------------------------------------------------------------------
# rung 2: site adapter (reddit)
# --------------------------------------------------------------------------
def test_adapter_selection_is_narrow() -> None:
    assert ct._select_site_adapter(REDDIT_URL)["name"] == "reddit-thread-atom"
    assert ct._select_site_adapter("https://old.reddit.com/r/x/comments/abc/t/")["name"] == "reddit-thread-atom"
    # A subreddit listing is not a thread; another host is not reddit.
    assert ct._select_site_adapter("https://www.reddit.com/r/samharris/") is None
    assert ct._select_site_adapter("https://notreddit.com/r/x/comments/abc/t/") is None


def test_a_reddit_thread_is_read_from_its_own_atom_feed_before_any_browser(monkeypatch) -> None:
    feed = REDDIT_URL.rstrip("/") + "/.rss"
    seen = _install_transport(
        monkeypatch,
        {
            REDDIT_URL: (200, JS_SHELL_200, "text/html", {}),
            feed: (200, REDDIT_ATOM, "application/atom+xml; charset=UTF-8", {}),
        },
    )
    calls = _install_render(monkeypatch, ARTICLE_HTML)

    r = ct.fetch_url(REDDIT_URL, include_full_content=False, keep_links=False)

    assert r["success"] is True, r.get("rendered")
    assert r["adapter_used"] == "reddit-thread-atom"
    assert r["adapter_source_url"] == feed
    assert r["rendered_with_browser"] is False, "an adapter is not a browser render"
    assert r["title"] == "No, AI Is Not Conscious"
    assert "The submission body text" in r["content"]
    assert "link: https://example.org/story" in r["content"], "a link post must keep its target"
    assert "Top-level comment number 3" in r["content"]
    assert "/u/commenter3" in r["content"]
    assert calls == [], "the cheaper adapter must win before a browser launch"
    assert feed in seen


def test_a_rate_limited_adapter_says_so(monkeypatch) -> None:
    feed = REDDIT_URL.rstrip("/") + "/.rss"
    _install_transport(
        monkeypatch,
        {REDDIT_URL: (200, JS_SHELL_200, "text/html", {}), feed: (429, "", "text/plain", {})},
    )
    _install_render(monkeypatch, CAPTCHA_DOM)

    r = ct.fetch_url(REDDIT_URL, include_full_content=False, keep_links=False)

    assert r["success"] is False
    assert any("rate-limited" in s for s in r["suggestions"]), r["suggestions"]


# --------------------------------------------------------------------------
# rung 5: honest classing after a real browser was refused
# --------------------------------------------------------------------------
def test_a_captcha_wall_is_classed_captcha_required_not_empty_content(monkeypatch) -> None:
    url = "https://forum.example.com/t/1"
    _install_transport(monkeypatch, {url: (200, JS_SHELL_200, "text/html", {})})
    _install_render(monkeypatch, CAPTCHA_DOM)

    r = ct.fetch_url(url, include_full_content=False, keep_links=False)

    assert r["success"] is False, "a CAPTCHA page was returned as content"
    assert r["error_class"] == "captcha_required"
    assert r["retryable"] is False
    assert any("out of scope by design" in s for s in r["suggestions"])


def test_a_paywall_is_named_paywall(monkeypatch) -> None:
    url = "https://paper.example.com/story"
    _install_transport(monkeypatch, {url: (200, JS_SHELL_200, "text/html", {})})
    _install_render(monkeypatch, PAYWALL_DOM)

    r = ct.fetch_url(url, include_full_content=False, keep_links=False)

    assert r["success"] is False
    assert r["error_class"] == "paywall"


def test_an_uncleared_interstitial_is_never_returned_as_the_article(monkeypatch) -> None:
    """medium.com, measured: the render snapshot caught Cloudflare mid-way and
    'Verification successful. Waiting for medium.com to respond' came back as
    a 254-char success."""
    url = "https://medium.com/@someone/post-123"
    _install_transport(monkeypatch, {url: (403, CF_CHALLENGE_403, "text/html", {"cf-mitigated": "challenge"})})
    _install_render(
        monkeypatch,
        "<html><body><h1>medium.com</h1><h2>Performing security verification</h2><p>This website uses a "
        "security service to protect against malicious bots. This page is displayed while the website "
        "verifies you are not a bot.</p><h2>Verification successful. Waiting for medium.com to respond</h2></body></html>",
    )

    r = ct.fetch_url(url, include_full_content=False, keep_links=False)

    assert r["success"] is False or r.get("degraded"), r.get("content")
    assert r["error_class"] == "blocked_by_site"


# --------------------------------------------------------------------------
# rung 4: the site's own feed when the article is refused to everyone
# --------------------------------------------------------------------------
def test_a_hard_block_recovers_title_and_standfirst_from_the_sites_own_feed(monkeypatch) -> None:
    _install_transport(
        monkeypatch,
        {
            ECON_URL: (403, CF_CHALLENGE_403, "text/html", {"cf-mitigated": "challenge"}),
            "https://www.economist.com/by-invitation/rss.xml": (200, ECON_RSS, "text/xml", {}),
        },
    )
    _install_render(monkeypatch, "<html><body></body></html>", status=403)

    r = ct.fetch_url(ECON_URL, include_full_content=False, keep_links=False)

    assert r["success"] is True and r["degraded"] is True
    assert r["error_class"] == "blocked_by_site"
    assert r["title"] == "Humanity has the debate about AI consciousness backwards"
    assert "care for others" in r["content"]
    assert "Wrong standfirst" not in r["content"], "a title was paired with another item's text"
    assert "public feed" in r["degraded_reason"]
    assert r["recovered_from_feed"].endswith("/by-invitation/rss.xml")


def test_a_hard_block_with_no_feed_is_a_classed_failure_not_a_bare_403(monkeypatch) -> None:
    _install_transport(
        monkeypatch, {ECON_URL: (403, CF_CHALLENGE_403, "text/html", {"cf-mitigated": "challenge"})}
    )
    _install_render(monkeypatch, "<html><body></body></html>", status=403)

    r = ct.fetch_url(ECON_URL, include_full_content=False, keep_links=False)

    assert r["success"] is False
    assert r["error_class"] == "blocked_by_site"
    assert r["rendered_with_browser"] is True
    assert "every client tried" in r["rendered"]


def test_feed_entry_matching_never_pairs_a_title_with_another_items_link() -> None:
    entry = ct._feed_entry_for_url(ECON_RSS, ECON_URL)
    assert entry == {
        "title": "Humanity has the debate about AI consciousness backwards",
        "summary": "We don’t care for others because they’re conscious, argues a writer",
    }
    assert ct._feed_entry_for_url(ECON_RSS, "https://www.economist.com/nope") is None


# --------------------------------------------------------------------------
# browser_tools: the render worker
# --------------------------------------------------------------------------
def test_the_render_worker_runs_from_a_neutral_cwd(monkeypatch) -> None:
    """`python -c` puts cwd first on sys.path; the monorepo root contains a
    folder named `abstractcore`, which shadowed the package and killed every
    render with an ImportError ("worker exited 1")."""
    import tempfile

    seen: Dict[str, Any] = {}

    class _Proc:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return ('{"html": "<html></html>"}', "")

    def _popen(cmd, **kw):
        seen.update(kw)
        return _Proc()

    monkeypatch.setattr(BT.subprocess, "Popen", _popen)
    BT._spawn_probe({"url": "https://example.com"}, 5.0)
    assert seen.get("cwd") == tempfile.gettempdir()


def test_render_empty_says_which_empty_it_was(monkeypatch) -> None:
    monkeypatch.setattr(BT, "_ensure_playwright", lambda: True)
    monkeypatch.setattr(
        BT, "_spawn_probe",
        lambda cfg, budget: {"html": None, "nav": {"timed_out": True, "no_commit": True}, "browser_mode": "persistent:chromium"},
    )
    r = BT.render_url_html("https://example.com/x", timeout_s=5, thorough=True)
    assert r["ok"] is False and r["error_class"] == "render_empty"
    assert "never committed" in r["message"]
    assert r["message"] != "the rendered page produced no DOM"


def test_thorough_render_asks_for_the_full_browser_and_a_tool_owned_profile(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(BT, "_ensure_playwright", lambda: True)
    monkeypatch.setattr(BT, "browser_profile_dir", lambda: str(tmp_path / "profile"))
    captured: Dict[str, Any] = {}

    def _spawn(cfg, budget):
        captured.update(cfg)
        return {"html": "<html><body>x</body></html>", "text": "x", "final_url": cfg["url"], "http_status": 200}

    monkeypatch.setattr(BT, "_spawn_probe", _spawn)
    r = BT.render_url_html("https://example.com/x", timeout_s=5, thorough=True)
    prof = captured["browser_profile"]
    assert prof["channel"] == "chromium"
    assert prof["user_data_dir"] == str(tmp_path / "profile")
    assert (tmp_path / "profile").is_dir(), "the parent must create the profile dir"
    assert captured["capture_text"] is True and captured["challenge_wait_ms"] > 0
    assert r["ok"] is True and r["text"] == "x"

    captured.clear()
    BT.render_url_html("https://example.com/x", timeout_s=5)
    assert captured["browser_profile"] is None, "the default render must stay the plain headless shell"


def test_the_default_profile_lives_under_the_tools_own_state_dir() -> None:
    from pathlib import Path

    assert BT.browser_profile_dir() == str(Path.home() / ".abstractcore" / "browser-profile")


def test_skim_url_points_at_fetch_url_on_a_challenge_page(monkeypatch) -> None:
    url = "https://blog.example.com/post/"
    _install_transport(monkeypatch, {url: (403, CF_CHALLENGE_403, "text/html", {"cf-mitigated": "challenge"})})
    out = ct.skim_url(url)
    assert "403" in out
    assert "fetch_url(url) escalates to a real browser" in out


def test_skim_url_adds_no_hint_to_a_plain_refusal(monkeypatch) -> None:
    url = "https://news.example.com/a/"
    _install_transport(monkeypatch, {url: (403, POLICY_403, "text/html", {})})
    assert "escalates to a real browser" not in ct.skim_url(url)


def test_escalated_content_reaches_the_model_visible_rendered_text(monkeypatch) -> None:
    """`rendered` is what most consumers show the model. It used to be composed
    from the server's shell before any escalation, so a rendered/adapter page
    showed 'Title: Reddit' and nothing else while the article sat in `content`."""
    feed = REDDIT_URL.rstrip("/") + "/.rss"
    _install_transport(
        monkeypatch,
        {REDDIT_URL: (200, JS_SHELL_200, "text/html", {}), feed: (200, REDDIT_ATOM, "application/atom+xml", {})},
    )
    r = ct.fetch_url(REDDIT_URL, include_full_content=False, keep_links=False)
    assert "The submission body text" in r["rendered"]

    url = "https://app.example.com/page"
    _install_transport(monkeypatch, {url: (200, JS_SHELL_200, "text/html", {})})
    _install_render(monkeypatch, ARTICLE_HTML)
    r = ct.fetch_url(url, include_full_content=False, keep_links=False)
    assert "substance of the article" in r["rendered"]


def test_a_self_posts_link_back_to_itself_is_dropped(monkeypatch) -> None:
    feed = REDDIT_URL.rstrip("/") + "/.rss"
    self_post = REDDIT_ATOM.replace("https://example.org/story", REDDIT_URL)
    _install_transport(monkeypatch, {feed: (200, self_post, "application/atom+xml", {})})
    out = ct._reddit_thread_via_atom(REDDIT_URL)
    assert out and "The submission body text" in out["content"]
    assert "link: " not in out["content"]
