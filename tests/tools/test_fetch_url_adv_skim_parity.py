"""Adversarial round 5: pin `skim_url` and `fetch_url` to ONE extraction.

`skim_url` used to re-implement the extraction pipeline inline — it called
`_prune_html_soup_for_text` / `_select_html_main_container` / `_html_to_markdown`
directly instead of `_extract_main_content`. It therefore shared the helpers but
not the FUNCTION'S OWN decisions, and a skim silently disagreed with a fetch of
the same URL. The measured cost on the committed fixtures: `bbc_tech_hub`
extracted 8,296 characters through fetch_url and 6,777 through the skim path,
because the inline copy never consulted the link-dominant override and dropped
every headline URL from a hub page — the one thing a skim of a hub page exists
to give you.

The unification landed. This file is what keeps it landed, plus the half of
"same source" that the unification does NOT address: skim_url reads at most
`max_bytes` (default 200,000) and then extracts from those TRUNCATED bytes, so
for a large page the two tools still run the same pipeline over different
documents.

Offline and deterministic: the committed fixtures are served from a loopback
HTTP server, so both tools take their real network code path with no network.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

FIXTURES = Path(__file__).resolve().parent / "fetch_url_fixtures"
sys.path.insert(0, str(FIXTURES))

from abstractcore.tools.common_tools import (  # noqa: E402
    _extract_main_content,
    fetch_url,
    skim_url,
)
from abstractcore.tools.fetch_url_ssrf import (  # noqa: E402
    reset_fetch_url_allowlist_cache,
)

# Every committed HTML fixture that carries real extractable content. The
# empty/challenge shells are excluded here on purpose — their parity is pinned
# in test_fetch_url_adv_js_render.py, where "both return nothing" is the point.
PARITY_FIXTURES = [
    "arxiv_abs.html",
    "bbc_tech_hub.html",
    "budgyapp.html",
    "discourse_thread.html",
    "do_pricing.html",
    "github_issue.html",
    "github_readme.html",
    "hn_thread.html",
    "ja_wikipedia.html",
    "js_hetzner_cloud.html",
    "js_notion_blog.html",
    "mdn_http_status.html",
    "newsletter.html",
    "nextbigfuture.html",
    "pydocs_json.html",
    "sqlite_datefunc.html",
    "supabase_pricing.html",
    "techstartups.html",
    "techxplore.html",
    "wikipedia_bert.html",
]

# skim_url's default byte cap. Ten of the fixtures above are larger than this.
# The OLD skim_url download default. No longer a default — kept as the value
# these tests pass EXPLICITLY, because truncation is now opt-in and its
# disclosure still has to be correct when a caller asks for a cheap peek.
SKIM_EXPLICIT_SMALL_MAX_BYTES = 200_000
SKIM_DEFAULT_MAX_BYTES = SKIM_EXPLICIT_SMALL_MAX_BYTES
# Big enough that no fixture is truncated, so a difference is a PIPELINE
# difference rather than a source difference.
UNTRUNCATED = 4_000_000


class _FixtureServer:
    def __init__(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                name = self.path.lstrip("/").split("?", 1)[0]
                path = FIXTURES / name
                if not path.is_file() or ".." in name:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except Exception:
                    pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._srv.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}/"
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        try:
            self._srv.shutdown()
            self._srv.server_close()
        except Exception:
            pass


@pytest.fixture(scope="module")
def fixture_server() -> Iterator[_FixtureServer]:
    srv = _FixtureServer()
    previous = os.environ.get("ABSTRACTCORE_FETCH_URL_ALLOW")
    os.environ["ABSTRACTCORE_FETCH_URL_ALLOW"] = f"127.0.0.1:{srv.port}"
    reset_fetch_url_allowlist_cache()
    try:
        yield srv
    finally:
        if previous is None:
            os.environ.pop("ABSTRACTCORE_FETCH_URL_ALLOW", None)
        else:
            os.environ["ABSTRACTCORE_FETCH_URL_ALLOW"] = previous
        reset_fetch_url_allowlist_cache()
        srv.close()


_MD_LINK = re.compile(r"\[[^\]]*\]\((https?://[^)]+)\)")


_PREVIEW_MARKER = "📄 Preview:\n"


def _skim_preview(report: str) -> str:
    """The markdown preview block out of a skim report.

    The report is a human-readable envelope: status lines, title/description,
    an optional headings outline, then `📄 Preview:` and the markdown. The
    trailing "Next: use fetch_url(...)" hint and preview_text's own
    "… (truncated)" marker are not part of the extraction.
    """
    _, _, body = report.partition(_PREVIEW_MARKER)
    if not body:
        return ""
    body = body.split("\nNext: use fetch_url", 1)[0]
    return body.strip()


def _preview_head(preview: str, limit: int = 600) -> str:
    """The part of a preview that must appear verbatim in the fetch content."""
    head = preview.split("… (truncated)", 1)[0].split("…", 1)[0].strip()
    return head[:limit]


# ---------------------------------------------------------------------------
# 1. one pipeline
# ---------------------------------------------------------------------------


def test_skim_url_calls_the_shared_extractor() -> None:
    """Structural pin. The inline copy is the thing that must not come back.

    Asserting on the source is blunt, but it is the only check that fails on the
    day someone re-inlines the pipeline "just for the preview" — a behavioural
    test would pass again as soon as the copy happened to agree.
    """
    import inspect

    source = inspect.getsource(skim_url)
    assert "_extract_main_content(" in source, "skim_url no longer routes through the shared extractor"
    for reimplemented in ("_prune_html_soup_for_text(", "_select_html_main_container(", "_html_to_markdown("):
        assert reimplemented not in source, (
            f"skim_url calls {reimplemented} directly again — that is the inline pipeline copy "
            "whose decisions diverged from _extract_main_content"
        )


@pytest.mark.parametrize("fixture", PARITY_FIXTURES)
def test_skim_preview_is_a_prefix_of_the_fetch_extraction(
    fixture: str, fixture_server: _FixtureServer
) -> None:
    """The documented difference is TRUNCATION and nothing else.

    A skim preview is `preview_text(markdown, max_chars=...)` over the same
    markdown a fetch returns, so its opening must appear verbatim in the fetch's
    `content`. Anything else means the two tools disagreed about what the page
    says — which is the defect the unification removed.
    """
    url = fixture_server.base + fixture
    fetched = str(
        fetch_url(url=url, timeout=30, keep_links=False, render_js="never").get("content") or ""
    )
    report = skim_url(url=url, timeout=30, max_bytes=UNTRUNCATED, max_preview_chars=4000)
    preview = _skim_preview(report)

    assert fetched, f"{fixture}: the fetch extracted nothing, so parity is untestable"
    assert preview, f"{fixture}: the skim produced no preview"

    probe = _preview_head(preview)
    assert probe and probe in fetched, (
        f"{fixture}: the skim preview is not a prefix of the fetch content.\n"
        f"  skim : {probe[:180]!r}\n"
        f"  fetch: {fetched[:180]!r}"
    )


@pytest.mark.parametrize("fixture", PARITY_FIXTURES)
def test_skim_headings_all_appear_in_the_fetch_extraction(
    fixture: str, fixture_server: _FixtureServer
) -> None:
    """The outline must describe the same document.

    Headings are harvested from the same markdown the preview is cut from, so a
    heading the fetch has never heard of means the two extractions diverged.
    """
    url = fixture_server.base + fixture
    fetched = str(
        fetch_url(url=url, timeout=30, keep_links=False, render_js="never").get("content") or ""
    )
    report = skim_url(url=url, timeout=30, max_bytes=UNTRUNCATED, max_preview_chars=1200)

    headings: List[str] = [
        line[2:].split(":", 1)[1].strip()
        for line in report.splitlines()
        if line.startswith("- H") and ":" in line
    ]
    # The heading harvest runs `_strip_markdown_inline` over each `#` line, so
    # `## \`json\` — JSON encoder` is reported as `json — JSON encoder`. Compare
    # with the same inline syntax removed, or the test fails on formatting
    # rather than on the two tools disagreeing about the document.
    def _flat(text: str) -> str:
        return " ".join(text.replace("`", "").replace("*", "").replace("…", "").split()).lower()

    normalized = _flat(fetched)
    for heading in headings:
        text = _flat(heading)
        if len(text) < 8:
            continue
        assert text[:60] in normalized, (
            f"{fixture}: skim reports heading {heading!r} that the fetch extraction does not contain"
        )


# ---------------------------------------------------------------------------
# 2. the hub-page regression the unification exists for
# ---------------------------------------------------------------------------


def test_a_hub_page_skim_keeps_the_urls(fixture_server: _FixtureServer) -> None:
    """The concrete defect: a hub-page skim used to return bare headlines.

    `bbc_tech_hub` is 40+ links wrapped in headings. The inline copy passed
    `keep_links=False` straight through and never saw the link-dominant
    override, so a reader got forty titles and no way to open any of them.
    """
    report = skim_url(
        url=fixture_server.base + "bbc_tech_hub.html",
        timeout=30,
        max_bytes=UNTRUNCATED,
        max_preview_chars=8000,
    )
    urls = _MD_LINK.findall(report)
    assert len(urls) >= 8, (
        f"a hub-page skim carried only {len(urls)} link target(s); the link-dominant "
        "override is not reaching the skim path"
    )


def test_the_link_dominant_override_is_what_produces_them() -> None:
    """Guard the mechanism, not just the symptom.

    If the links ever came back for some unrelated reason, this fails and says
    so, which keeps the test above honest.
    """
    html = (FIXTURES / "bbc_tech_hub.html").read_text(encoding="utf-8", errors="replace")
    main = _extract_main_content(html, "https://www.bbc.com/technology", keep_links=False)
    assert main.get("link_dominant") is True, "bbc_tech_hub is no longer a link-dominant container"
    assert len(_MD_LINK.findall(str(main.get("content") or ""))) >= 8


def test_an_article_page_skim_is_not_flooded_with_links(fixture_server: _FixtureServer) -> None:
    """The false-positive side. `keep_links=False` must still mean something on
    a page that is prose, or the fix has simply traded one defect for another."""
    report = skim_url(
        url=fixture_server.base + "techstartups.html",
        timeout=30,
        max_bytes=UNTRUNCATED,
        max_preview_chars=4000,
    )
    body = _skim_preview(report)
    assert len(_MD_LINK.findall(body)) <= 3, (
        "an article skim with keep_links=False came back full of link targets"
    )


# ---------------------------------------------------------------------------
# 3. the half of "same source" that is still open
# ---------------------------------------------------------------------------

# Measured 2026-08-21: `_extract_main_content` over the full fixture vs over its
# first 200,000 bytes (skim's default cap):
#   ja_wikipedia         27,225 -> 2,163   92.1% lost
#   github_issue            919 ->   166   81.9% lost
#   bbc_tech_hub          8,377 -> 2,022   75.9% lost
#   wikipedia_bert       36,600 -> 22,808  37.7% lost
#   supabase_pricing     14,850 -> 12,010  19.1% lost
TRUNCATION_SENSITIVE = ["ja_wikipedia.html", "bbc_tech_hub.html", "wikipedia_bert.html"]


@pytest.mark.parametrize("fixture", TRUNCATION_SENSITIVE)
def test_a_truncated_skim_says_its_extraction_is_partial(
    fixture: str, fixture_server: _FixtureServer
) -> None:
    """A truncated skim must disclose a partial EXTRACTION, not just a partial DOWNLOAD.

    Originally written against the default `max_bytes=200_000`, which cut the
    response mid-document and ran the shared pipeline over the fragment: the
    report said "Downloaded: 200,000 bytes (partial; limit 200,000)" — true, and
    easy to read as "the tail of a long page was trimmed". It was not.
    `bbc_tech_hub` lost 76% of its extracted content and `ja_wikipedia` 92%, and
    the headings outline silently shrank to match, so nothing looked short.

    That default is GONE: the download bound is now a memory safety net
    (`SKIM_URL_MAX_DOWNLOAD_BYTES`, the same 10 MB fetch_url uses) and the
    skimming is done by `max_preview_chars` on the EXTRACTED text, where taking a
    fraction of the content actually means something. Measured before the change:
    truncating those pages saved 0.00-0.07s of download and cost up to 92% of
    the article.

    So the requirement moved rather than disappeared. Truncation still happens
    when a caller asks for it explicitly — a cheap peek is a legitimate use — and
    it must still say that the EXTRACTION, not merely the transfer, was cut. That
    is what this pins now.
    """
    url = fixture_server.base + fixture
    # Explicit small cap: the only way truncation happens now.
    report = skim_url(url=url, timeout=30, max_bytes=SKIM_DEFAULT_MAX_BYTES)
    full = str(
        fetch_url(url=url, timeout=30, keep_links=False, render_js="never").get("content") or ""
    )
    truncated_source = (FIXTURES / fixture).read_bytes()[:SKIM_DEFAULT_MAX_BYTES]
    partial = str(
        _extract_main_content(
            truncated_source.decode("utf-8", errors="replace"), url, keep_links=False
        ).get("content")
        or ""
    )
    lost = 1 - (len(partial) / max(1, len(full)))
    assert lost > 0.10, f"{fixture} is no longer truncation-sensitive; re-pick the fixture"

    low = report.lower()
    assert "partial" in low, f"{fixture}: a truncated skim did not disclose truncation at all"

    # The existing disclosure is on the DOWNLOAD line only:
    #   "Downloaded: 200,000 bytes (partial; limit 200,000)"
    # Nothing tells the reader that the preview and the headings outline were
    # built from a cut-off document. Look for a claim about the EXTRACTION that
    # is not just the Content-Type header echoing the word "content".
    download_line = next((ln for ln in report.splitlines() if ln.startswith("Downloaded:")), "")
    elsewhere = "\n".join(
        ln for ln in report.splitlines()
        if not ln.startswith(("Downloaded:", "Content-Type:", "URL:", "Final URL:", "Status:", "Detected-As:"))
    ).lower()
    assert any(
        phrase in elsewhere
        for phrase in ("partial", "incomplete", "cut off", "truncated source", "extraction is")
    ), (
        f"{fixture}: the only truncation disclosure is {download_line!r}, which describes the "
        f"DOWNLOAD. {lost:.0%} of the extractable content is missing from the preview and the "
        "headings outline, and nothing in the report says so — use fetch_url or raise max_bytes."
    )


def test_skim_and_fetch_agree_when_the_source_is_not_truncated(
    fixture_server: _FixtureServer,
) -> None:
    """The control for the test above: raise the cap and the disagreement goes
    away, which localises it in the byte cap rather than the pipeline."""
    url = fixture_server.base + "bbc_tech_hub.html"
    full = str(
        fetch_url(url=url, timeout=30, keep_links=False, render_js="never").get("content") or ""
    )
    generous = _skim_preview(
        skim_url(url=url, timeout=30, max_bytes=UNTRUNCATED, max_preview_chars=12000)
    )
    head = _preview_head(generous, limit=800)
    assert head and head in full


# ---------------------------------------------------------------------------
# 4. skim_url is a URL fetcher too
# ---------------------------------------------------------------------------


def test_skim_url_refuses_a_non_public_destination() -> None:
    """RED (attack): `skim_url` has no SSRF guard.

    It is the sibling model-callable URL fetcher — same "point this at a URL"
    surface, same model-controlled argument — and it carries fetch_url's base64
    screen, so the parity was clearly intended. But it builds a bare
    `requests.Session()` with no `SSRFGuardAdapter` and never calls
    `fetch_url_guard_destination`.

    Verified 2026-08-21: an internal page that `fetch_url` refuses with
    `blocked_ssrf` is fetched and previewed by `skim_url`, marker and all. The
    guidance in `fetch_url`'s own docstring — prefer skim_url first — points
    the model at the unguarded one.
    """
    marker = "SKIM-LOOPBACK-MARKER-4d2"
    body = (
        "<!doctype html><html><head><title>internal</title></head><body><main><p>"
        + marker
        + " internal admin console. "
        + ("Sensitive internal prose that must never reach a model. " * 8)
        + "</p></main></body></html>"
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/internal"
    try:
        previous = os.environ.pop("ABSTRACTCORE_FETCH_URL_ALLOW", None)
        reset_fetch_url_allowlist_cache()
        try:
            blocked = fetch_url(url=url, timeout=10)
            assert blocked.get("error_class") == "blocked_ssrf", (
                "the control failed: fetch_url did not refuse the loopback host"
            )
            report = skim_url(url=url, timeout=10)
            assert marker not in report, (
                "skim_url fetched a non-public destination that fetch_url refuses"
            )
        finally:
            if previous is not None:
                os.environ["ABSTRACTCORE_FETCH_URL_ALLOW"] = previous
            reset_fetch_url_allowlist_cache()
    finally:
        srv.shutdown()
        srv.server_close()
