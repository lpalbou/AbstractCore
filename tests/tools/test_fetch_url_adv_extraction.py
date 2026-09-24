"""Adversarial regressions for fetch_url extraction / structuration.

Round 1 (2026-08-21). Each test below pins ONE defect found by scoring the
expanded fixture corpus (see `fetch_url_fixtures/README.md`) against the
shipping extractor. Every test either carries a <30-line minimal HTML repro or
asserts one precise string/metric on a committed real-page fixture. Nothing
here is xfail-gated: a red test is a defect that is still live.

Grouping:
  A. markdown fidelity   — code blocks, tables, inline runs
  B. content destruction — blocks deleted by the boilerplate blacklist
  C. boilerplate leaks   — chrome the blacklist does not catch
  D. envelope cost       — the same text shipped under several keys
  E. documents (PDF)     — truncation, ligatures, triplication
  F. unrenderable pages  — SPA shells must fail loudly (regression guard)

Deterministic + offline: committed fixtures only, no network.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fetch_url_fixtures"
sys.path.insert(0, str(FIXTURES))

from harness import (  # noqa: E402
    GOLD,
    doc_keys,
    fixture_bytes,
    envelope_efficiency,
    fixture_html,
    offline_envelope,
    score,
    special_keys,
    structure_score,
    url_keys,
)

from abstractcore.tools.common_tools import (  # noqa: E402
    _detect_unrenderable_html,
    _extract_main_content,
)

# The fetch is faked; the SSRF check still resolves the host first. Answer that
# from a fake resolver instead of real DNS (network guard finding, 2026-09-24).
pytestmark = pytest.mark.usefixtures("fake_public_dns")

# The maintainer bar for how much a caller should pay, in characters of JSON,
# per character of real extracted content. Today `raw_text` (the entire raw
# HTML, <script>/<style> included) dominates the envelope; see group D.
MAX_ENVELOPE_AMPLIFICATION = 3.0
# Below this much content the ratio is dominated by fixed envelope overhead, so
# the bar becomes an absolute char budget instead of a multiplier.
_AMPLIFICATION_FLOOR_CHARS = 2000
_SMALL_PAGE_ENVELOPE_BUDGET_CHARS = 8000

_PAD = "<p>" + ("Real article prose that carries the actual meaning of this page. " * 14) + "</p>"


def _structure_backend_available() -> bool:
    from importlib.util import find_spec

    return find_spec("pymupdf4llm") is not None


def _content(body_html: str, *, url: str = "https://example.com/a") -> str:
    """Extract `content` from a minimal page whose <article> holds `body_html`."""
    html = f"<html><head><title>T</title></head><body><article>{body_html}{_PAD}</article></body></html>"
    return str(_extract_main_content(html, url, keep_links=True).get("content") or "")


def _fixture_content(key: str) -> str:
    main = _extract_main_content(fixture_html(key), GOLD[key]["url"], keep_links=True)
    return str(main.get("content") or "")


def _lines_containing(content: str, needle: str):
    return [ln for ln in content.splitlines() if needle in ln]


# ---------------------------------------------------------------------------
# A. markdown fidelity
# ---------------------------------------------------------------------------


def test_syntax_highlighted_code_block_is_not_shredded():
    """A <pre> whose tokens are wrapped in highlight <span>s must stay one line
    per source line — today every token becomes its own line inside the fence.

    Live impact: docs.python.org, GitHub READMEs, MDN samples — every code
    example in the corpus is unusable.
    """
    content = _content(
        '<h1>Doc</h1><div class="highlight"><pre>'
        '<span class="kn">import</span> <span class="nn">json</span>\n'
        '<span class="n">json</span><span class="o">.</span><span class="n">dumps</span>'
        '<span class="p">([</span><span class="mi">1</span><span class="p">])</span>\n'
        "</pre></div>"
    )
    assert "import json" in content, f"tokens split apart:\n{content[:400]}"
    assert "json.dumps([1])" in content, f"tokens split apart:\n{content[:400]}"


def test_inline_spans_in_a_div_do_not_each_become_a_paragraph():
    """Sentence fragments wrapped in sibling <span>s (styled-components sites:
    BBC, DigitalOcean, arXiv labs) must join into one line, not one block each."""
    content = _content("<div><span>Version</span> <span>3.14</span> <span>released today.</span></div>")
    assert "Version 3.14 released today." in content, f"inline run fragmented:\n{content[-300:]}"


def test_table_inside_a_definition_list_keeps_its_table_structure():
    """A <table> nested in a <dd> is flattened into prose today, and adjacent
    cells are glued together with no separator ('JSONPython number (real)float').

    Live impact: every conversion/parameter table in the Python docs.
    """
    content = _content(
        "<h1>Doc</h1><dl><dt>class json.JSONDecoder</dt>"
        "<dd><p>Performs the following translations:</p>"
        "<table><tr><th>JSON</th><th>Python</th></tr>"
        "<tr><td>number (real)</td><td>float</td></tr></table></dd></dl>"
    )
    rows = _lines_containing(content, "number (real)")
    assert rows, "table cell text vanished entirely"
    assert "|" in rows[0], f"table flattened into prose: {rows[0]!r}"
    assert "JSONPython" not in content, "adjacent table cells were concatenated without a separator"


def test_table_inside_a_list_item_keeps_its_cells_separated():
    """Same defect one level up: <table> inside <li>."""
    content = _content(
        "<h1>Doc</h1><ul><li><p>Conversions:</p>"
        "<table><tr><th>JSON</th><th>Python</th></tr>"
        "<tr><td>number (real)</td><td>float</td></tr></table></li></ul>"
    )
    assert "JSONPython" not in content, "adjacent table cells were concatenated without a separator"
    rows = _lines_containing(content, "number (real)")
    assert rows and "|" in rows[0], f"table flattened into prose: {content[-300:]!r}"


def test_pydocs_code_examples_survive_intact():
    """Fixture proof of the two defects above on docs.python.org/3/library/json.html."""
    content = _fixture_content("pydocs_json")
    assert "json.dumps([1, 2, 3, {'4': 5, '6': 7}], separators=(',', ':'))" in content
    assert "raise TypeError(f'Cannot serialize object of {type(obj)}')" in content
    assert "json.loads('1.1', parse_float=decimal.Decimal)" in content


def test_pydocs_conversion_table_survives_as_a_markdown_table():
    content = _fixture_content("pydocs_json")
    rows = _lines_containing(content, "number (real)")
    assert rows, "the JSON->Python conversion table was dropped"
    assert "|" in rows[0], f"conversion table flattened into prose: {rows[0][:160]!r}"


@pytest.mark.parametrize(
    "gold_key", [k for k in url_keys() if GOLD[k].get("structural_facts")]
)
def test_structural_facts_survive_with_their_structure(gold_key: str):
    """Facts that live in a table cell / code block / list item / link label must
    still be inside that structure in `content` (harness.structure_score)."""
    st = structure_score(_fixture_content(gold_key), gold_key)
    assert st["structure_score"] == 1.0, (
        f"{gold_key}: structure_score={st['structure_score']}; "
        f"flattened={st['flattened']} missing={st['missing']}"
    )


# ---------------------------------------------------------------------------
# B. content destruction (the boilerplate blacklist eats real content)
# ---------------------------------------------------------------------------


def test_discussion_comments_are_not_pruned_as_boilerplate():
    """`_prune_html_container_for_readability` blacklists the class keywords
    'comment'/'comments'. On a discussion page the comments ARE the content, so
    the whole thread is silently deleted.

    Live impact: Hacker News, Discourse, GitHub issue threads, Reddit, forums.
    """
    content = _content(
        "<h1>Codex bug thread</h1>"
        '<div class="comment-tree">'
        '<div class="comment"><p>Wow, that whole thread is borderline incoherent.</p></div>'
        '<div class="comment"><p>Our read / write cache ratio was less than 5 percent.</p></div>'
        "</div>"
    )
    assert "borderline incoherent" in content, f"comment tree deleted:\n{content[:400]}"
    assert "read / write cache ratio" in content


def test_table_cell_whose_class_merely_contains_comment_is_not_deleted():
    """The blacklist is a substring match on class names, so arXiv's
    `<td class="tablecell comments mathjax">15 pages, 5 figures</td>` is
    decomposed and the row is emitted with an empty cell."""
    content = _content(
        "<h1>Attention Is All You Need</h1>"
        '<div class="metatable"><table>'
        '<tr><td class="tablecell label">Comments:</td>'
        '<td class="tablecell comments mathjax">15 pages, 5 figures</td></tr>'
        '<tr><td class="tablecell label">Subjects:</td>'
        '<td class="tablecell subjects">Computation and Language (cs.CL)</td></tr>'
        "</table></div>"
    )
    assert "15 pages, 5 figures" in content, f"cell deleted by class-substring prune:\n{content[:400]}"


def test_hn_thread_keeps_its_comments():
    """Fixture proof: the HN item page has ~6.2k chars of real text (26
    comments); `content` currently carries ~1.7k chars and NOT one comment."""
    content = _fixture_content("hn_thread")
    s = score(content, "hn_thread")
    assert s["recall"] >= 0.90, f"HN recall {s['recall']}; missed {s['missed']}"


def test_arxiv_metadata_table_keeps_the_comments_row():
    content = _fixture_content("arxiv_abs")
    assert "15 pages, 5 figures" in content


# ---------------------------------------------------------------------------
# C. boilerplate leaks (chrome the blacklist does not catch)
# ---------------------------------------------------------------------------


def test_article_footer_feedback_cta_does_not_leak():
    """`<div class="article-footer__inner">` is not a <footer> tag and 'footer'
    is not a blacklisted class keyword, so MDN's feedback CTA lands in content."""
    content = _content(
        '<h1>Doc</h1><div class="article-footer__inner"><h3>Help improve MDN</h3>'
        "<p>Was this page helpful to you?</p><button>Yes</button><button>No</button>"
        '<a href="/c">Report a problem with this content</a></div>'
    )
    assert "Help improve MDN" not in content
    assert "Was this page helpful to you?" not in content
    assert "Report a problem with this content" not in content


def test_trailing_last_updated_stamp_does_not_leak():
    """sqlite.org closes every page with a build stamp in a plain <p><small><i>."""
    content = _content(
        '<h1>Doc</h1><div class="fancy"><p><small><i>'
        "This page was last updated on 2026-07-11 15:07:16Z</i></small></p></div>"
    )
    assert "This page was last updated on" not in content


def test_toggle_rail_of_third_party_widgets_does_not_leak():
    """arXiv's arXivLabs rail: dozens of `X Toggle` / `(What is X?)` switches."""
    content = _content(
        '<h1>Attention Is All You Need</h1><div class="labstabs"><div class="tab labs-display-bib">'
        '<div class="lab-row"><div class="lab-switch"><label class="switch">'
        '<span class="is-sr-only">Bibliographic Explorer Toggle</span></label></div>'
        '<div>Bibliographic Explorer (<a href="https://info.arxiv.org/labs/showcase.html">'
        "What is the Explorer?</a>)</div></div></div></div>"
    )
    assert "Bibliographic Explorer" not in content
    assert "What is the Explorer?" not in content


def test_link_only_site_footer_without_semantic_hints_does_not_leak():
    """Styled-components sites (DigitalOcean) ship hashed class names only, so a
    keyword blacklist can never see the footer. A link-density / trailing-block
    heuristic is the only thing that can."""
    content = _content(
        '<h1>Doc</h1><div class="Sectionstyles__StyledSectionInner-sc-4l5hhw-1 bMaVH">'
        '<div class="Containerstyles-sc-11hjsrs-0 jIQAEC"><span>&copy;</span><span>2026</span>'
        "<span>ExampleCorp, LLC.</span>"
        '<a href="/sitemap">Sitemap</a><a href="/about">About</a><a href="/careers">Careers</a>'
        '<a href="/press">Press</a><a href="/legal">Legal</a><a href="/privacy">Privacy Policy</a>'
        '<a href="/security">Security</a></div></div>'
    )
    assert "Sitemap" not in content
    assert "Privacy Policy" not in content


def test_site_nav_pipe_bar_does_not_leak_on_table_layout_pages():
    """Hacker News' `<span class="pagetop">` nav has no nav/role semantics, so
    the whole `new | past | comments | ask | show | jobs | submit` bar and the
    `Guidelines | FAQ | ... | Apply to YC | Contact` footer land in content."""
    content = _fixture_content("hn_thread")
    s = score(content, "hn_thread")
    assert s["junk_ratio"] == 0.0, f"chrome leaked into content: {s['junk_hit']}"


@pytest.mark.parametrize("gold_key", ["mdn_http_status", "do_pricing", "sqlite_datefunc", "arxiv_abs"])
def test_known_leaky_fixtures_carry_zero_junk(gold_key: str):
    s = score(_fixture_content(gold_key), gold_key)
    assert s["junk_ratio"] == 0.0, f"{gold_key}: boilerplate leaked: {s['junk_hit']}"


# ---------------------------------------------------------------------------
# D. envelope cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gold_key", url_keys() + doc_keys())
def test_no_payload_field_duplicates_the_primary_content(gold_key: str):
    """D5: `normalized_text` (and, for PDFs, `raw_text`) is the SAME text as
    `content` under another key. A caller pays for it twice or three times."""
    env = envelope_efficiency(offline_envelope(gold_key))
    assert not env["duplicate_fields"], (
        f"{gold_key}: {env['duplicate_payload_chars']} chars duplicated: "
        f"{env['duplicate_fields']}"
    )


@pytest.mark.parametrize("gold_key", url_keys() + doc_keys())
def test_envelope_amplification_is_bounded(gold_key: str):
    """Total JSON chars per char of real content. Today the raw HTML rides
    inline in `raw_text`, so a 2.4kB README costs 337kB of envelope."""
    env = envelope_efficiency(offline_envelope(gold_key))
    assert env["amplification"] is not None, f"{gold_key}: no content at all"
    if env["content_chars"] < _AMPLIFICATION_FLOOR_CHARS:
        # A ratio is meaningless against a few hundred chars of content: the
        # envelope's fixed overhead (status line, headers, rendered preamble)
        # dominates and is not duplication. Hold small pages to an ABSOLUTE
        # budget instead, so the assertion measures what it claims to.
        assert env["total_json_chars"] <= _SMALL_PAGE_ENVELOPE_BUDGET_CHARS, (
            f"{gold_key}: {env['total_json_chars']} chars of envelope for "
            f"{env['content_chars']} chars of content; payload fields: {env['payload_fields']}"
        )
        return
    assert env["amplification"] <= MAX_ENVELOPE_AMPLIFICATION, (
        f"{gold_key}: {env['amplification']}x amplification "
        f"(content={env['content_chars']}, total={env['total_json_chars']}); "
        f"payload fields: {env['payload_fields']}"
    )


def test_raw_html_is_not_shipped_inline_in_the_envelope():
    """`raw_text` currently carries the entire response body, <script>/<style>
    included. Evidence belongs in an artifact, not in the tool result."""
    env = offline_envelope("techstartups")
    raw = str(env.get("raw_text") or "")
    assert "<script" not in raw.lower(), (
        f"raw_text ships {len(raw)} chars of raw HTML (script/style included) inline"
    )


# ---------------------------------------------------------------------------
# E. documents (PDF)
# ---------------------------------------------------------------------------


def test_pdf_content_covers_the_whole_document_at_default_settings():
    """Regression guard (GREEN). Round 1 reported this as red — that was MY bug:
    the harness rebuilt the envelope with route_pdf_bytes' own default
    (include_full_content=False) instead of fetch_url's (True). At fetch_url's
    real defaults all 15 pages come through. Keep it that way."""
    env = offline_envelope("arxiv_paper_pdf")
    content = str(env.get("content") or "")
    s = score(content, "arxiv_paper_pdf")
    assert s["recall"] >= 0.90, (
        f"PDF recall {s['recall']:.2f} ({len(content)} chars extracted); missed {s['missed']}"
    )


def test_pdf_content_carries_page_anchors_so_a_model_can_cite_a_page():
    """D9. A model asked for "what does page 12 say" needs page boundaries in
    the text. The pymupdf/pypdf path emits `# Page N`; whichever backend `auto`
    ends up choosing must keep them — pymupdf4llm currently does NOT, so this
    guard fires if the D7 fix is applied naively."""
    content = str(offline_envelope("arxiv_paper_pdf").get("content") or "")
    markers = re.findall(r"(?m)^#{1,6}\s+Page\s+(\d+)\s*$", content)
    assert markers, "no page anchors in extracted PDF text — a model cannot cite a page"
    pages = [int(m) for m in markers]
    assert pages[:3] == [1, 2, 3], f"page anchors are not sequential from 1: {pages[:6]}"
    assert len(pages) >= 15, f"only {len(pages)} page anchors for a 15-page document"


# D7 was originally written to demand that `auto` reach the markdown-producing
# backend whenever it was importable. The maintainer's licence policy overrides
# that: PyMuPDF, pymupdf4llm and pymupdf-layout are AGPL-3.0/commercial, only
# `pypdf` (BSD-3-Clause) ships in the default install profiles, and
# docs/backlog/completed/0805 keeps it that way on purpose. Measured on this
# fixture, `pymupdf` buys nothing over `pypdf` anyway (49,096 vs 49,049 chars,
# 15 page anchors each, 0 tables either way) — so `auto` has no reason to reach
# for an AGPL backend on its own. The requirement therefore has TWO halves, and
# both are pinned below: auto stays permissive, and structure stays reachable on
# an explicit, deliberate opt-in.
_PERMISSIVE_PDF_BACKENDS = {"pypdf"}


def test_pdf_auto_backend_stays_on_a_permissively_licensed_extractor():
    """`auto` must not silently route a document through an AGPL backend."""
    from abstractcore.media.pdf_routing import route_pdf_bytes

    route = route_pdf_bytes(
        fixture_bytes("arxiv_paper_pdf"),
        source_url=GOLD["arxiv_paper_pdf"]["url"],
        include_full_content=True,
        preferred_backend="auto",
    )
    backend = str(route.get("text_backend") or "")
    assert backend in _PERMISSIVE_PDF_BACKENDS, (
        f"'auto' chose {backend!r}; the default document path must stay on a "
        f"permissively licensed backend ({sorted(_PERMISSIVE_PDF_BACKENDS)})"
    )
    # Permissive does not mean degraded: the text and the page anchors are all
    # there, which is what a model needs to read and cite the document.
    text = str(route.get("normalized_text") or "")
    assert len(text) > 40_000, f"auto extracted only {len(text)} chars of a 15-page paper"
    assert len(re.findall(r"(?m)^#{1,6}\s*Page \d+", text)) >= 15


@pytest.mark.skipif(
    not _structure_backend_available(),
    reason="pymupdf4llm is not importable; the structure-preserving backend cannot be exercised",
)
def test_pdf_structure_backend_recovers_tables_on_explicit_opt_in():
    """Asking for the AGPL backend BY NAME must actually recover the structure.

    This is the escape hatch that makes the permissive default acceptable: a
    caller who has accepted the licence gets tables, and — because the page
    anchors were carried across the backend switch — keeps citability too.
    """
    from abstractcore.media.pdf_routing import route_pdf_bytes

    route = route_pdf_bytes(
        fixture_bytes("arxiv_paper_pdf"),
        source_url=GOLD["arxiv_paper_pdf"]["url"],
        include_full_content=True,
        preferred_backend="pymupdf4llm",
    )
    text = str(route.get("normalized_text") or "")
    table_rows = len(re.findall(r"(?m)^\s*\|.*\|", text))
    assert table_rows > 0, (
        f"the explicit structure-preserving backend produced no table rows "
        f"(backend={route.get('text_backend')!r})"
    )
    assert len(re.findall(r"(?m)^#{1,6}\s*Page \d+", text)) >= 15, (
        "page anchors were lost when the structure-preserving backend was used"
    )


def test_pdf_text_normalizes_typographic_ligatures():
    """pypdf emits U+FB01 for 'fi'; a reader searching for 'fixed-length vector'
    finds nothing."""
    content = str(offline_envelope("arxiv_paper_pdf").get("content") or "")
    assert "ﬁ" not in content, "U+FB01 (fi) ligature left un-normalized in PDF content"
    assert "fixed-length vector" in content


def test_pdf_envelope_does_not_ship_the_same_text_three_times():
    env = offline_envelope("arxiv_paper_pdf")
    content = str(env.get("content") or "")
    assert str(env.get("raw_text") or "") != content, "raw_text is byte-identical to content"
    assert str(env.get("normalized_text") or "") != content, "normalized_text is byte-identical to content"


# ---------------------------------------------------------------------------
# F. unrenderable pages (regression guard — these pass today)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gold_key", special_keys())
def test_js_only_shell_is_reported_as_unrenderable(gold_key: str):
    """A JS-only SPA shell must produce an actionable error, never a silent
    empty success."""
    entry = GOLD[gold_key]
    html = fixture_html(gold_key)
    main = _extract_main_content(html, entry["url"], keep_links=True)
    verdict = _detect_unrenderable_html(html, str(main.get("content") or ""))
    assert verdict is not None, f"{gold_key}: SPA shell was accepted as real content"
    err_class, suggestions = verdict
    assert err_class == entry["expect_error_class"], f"{gold_key}: got {err_class}"
    assert suggestions, "an unrenderable verdict must carry actionable suggestions"


@pytest.mark.parametrize("gold_key", special_keys())
def test_js_only_shell_does_not_leak_its_noscript_text_as_content(gold_key: str):
    main = _extract_main_content(fixture_html(gold_key), GOLD[gold_key]["url"], keep_links=True)
    s = score(str(main.get("content") or ""), gold_key)
    assert s["junk_ratio"] == 0.0, f"{gold_key}: {s['junk_hit']}"


# ---------------------------------------------------------------------------
# G. harness fidelity — the rebuild must not drift from the shipping tool
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    """Enough of a requests.Response for fetch_url's streaming read path."""

    def __init__(self, body: bytes, url: str, content_type: str) -> None:
        self.status_code = 200
        self.ok = True
        self.reason = "OK"
        self.url = url
        self.headers = {"content-type": content_type}
        self._body = body
        self.text = body.decode("utf-8", errors="replace")

    def iter_content(self, chunk_size: int = 16384):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class _FakeResponseCM:
    def __init__(self, response: _FakeHTTPResponse) -> None:
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, *exc):
        return False


def _fake_session_factory(body: bytes, url: str, content_type: str):
    class _FakeSession:
        def __init__(self) -> None:
            self.headers: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def mount(self, *args, **kwargs):
            return None

        def request(self, **kwargs):
            return _FakeResponseCM(_FakeHTTPResponse(body, url, content_type))

    return _FakeSession


@pytest.mark.parametrize("gold_key", ["techstartups", "github_readme", "hn_thread"])
def test_offline_envelope_matches_the_real_fetch_url_envelope(monkeypatch, gold_key: str):
    """The harness rebuilds fetch_url's envelope offline so envelope cost is
    measurable in CI. That rebuild is only worth anything if it stays identical
    to what the tool actually returns.

    This has already bitten twice: the rebuild used route_pdf_bytes' default
    instead of fetch_url's, and it missed `_apply_fetch_url_payload_policy`
    entirely. Both made the scoreboard describe a tool that does not ship.
    """
    from abstractcore.tools import common_tools as ct

    ct._ensure_requests()
    url = GOLD[gold_key]["url"]
    body = fixture_bytes(gold_key)
    monkeypatch.setattr(
        ct.requests, "Session", _fake_session_factory(body, url, "text/html; charset=utf-8")
    )

    real = ct.fetch_url(url=url, timeout=5)
    rebuilt = offline_envelope(gold_key)

    assert real.get("success") is True, f"fake transport did not produce a success: {real!r}"
    assert set(rebuilt) == set(real), (
        f"{gold_key}: offline_envelope has drifted from fetch_url.\n"
        f"  missing from rebuild: {sorted(set(real) - set(rebuilt))}\n"
        f"  extra in rebuild:     {sorted(set(rebuilt) - set(real))}"
    )
    # The payload-bearing fields must agree in KIND, not just in name: a rebuild
    # that inlines what the tool withholds measures the wrong envelope.
    for field in ("content", "raw_text", "normalized_text"):
        assert type(rebuilt[field]) is type(real[field]), (
            f"{gold_key}: {field} is {type(rebuilt[field]).__name__} in the rebuild but "
            f"{type(real[field]).__name__} in the real result"
        )
    assert rebuilt["content"] == real["content"], f"{gold_key}: rebuilt content differs"


# ---------------------------------------------------------------------------
# H. round 3 — do the generic fixes hold on real bytes they have not seen?
# ---------------------------------------------------------------------------


def test_discourse_thread_keeps_every_reply():
    """Round-2 prediction under test: "Discourse remains broken even though HN is
    green". On real bytes the comment tree SURVIVES — 21/21 facts across the OP
    and the replies. Kept as a regression guard for the generic fix."""
    s = score(_fixture_content("discourse_thread"), "discourse_thread")
    assert s["recall"] >= 0.90, f"discourse recall {s['recall']}; missed {s['missed']}"


def test_crawler_linkback_rail_inside_the_first_post_does_not_leak():
    """Discourse nests a suggested-topics rail INSIDE the first post's body
    (`<div class="crawler-linkback-list">` under `#post_1`), so four unrelated
    topic headlines ride into `content`. Same shape as the techstartups
    Read-Next rail (D2), a class name no blacklist keyword covers."""
    content = _content(
        '<h1>PEP 703 acceptance</h1><div class="topic-body crawler-post" id="post_1">'
        "<p>The Steering Council has decided to accept PEP 703.</p>"
        '<div class="crawler-linkback-list">'
        '<a href="/t/free-threading-trove-classifier/1">Free-Threading Trove Classifier?</a>'
        '<a href="/t/pep-803/2">PEP 803: Stable ABI for Free-Threaded Builds</a>'
        "</div></div>"
    )
    assert "Free-Threading Trove Classifier?" not in content
    assert "PEP 803: Stable ABI for Free-Threaded Builds" not in content


def test_utility_class_cta_banner_does_not_leak():
    """Supabase's bottom CTA is a Tailwind-utility-classed anchor
    (`class="relative inline-flex items-center justify-center cursor-pointer"`).
    Like the DigitalOcean footer, there is no semantic class to key on — only a
    structural signal (trailing block, link-only, no prose) can catch it."""
    content = _content(
        "<h1>Pricing</h1>"
        '<div class="bg-background grid grid-cols-12 items-center gap-4 border-t py-32">'
        '<div class="flex items-center justify-center gap-2 col-span-12 mt-4">'
        '<a class="relative inline-flex items-center justify-center cursor-pointer space-x-2" '
        'href="/dashboard"><span class="truncate">Start your project</span></a>'
        '<a class="relative inline-flex items-center justify-center cursor-pointer space-x-2" '
        'href="/contact"><span class="truncate">Request a demo</span></a>'
        "</div></div>"
    )
    assert "Request a demo" not in content


def test_comment_form_chrome_does_not_leak():
    """budgyapp's article tail carries `Continue Reading / Click to comment /
    Cancel reply`. This was invisible until round 3 because the fixture's junk
    list contained a string that did not occur in the fixture at all."""
    s = score(_fixture_content("budgyapp"), "budgyapp")
    assert s["junk_ratio"] == 0.0, f"comment-form chrome leaked: {s['junk_hit']}"


def test_issue_page_content_says_which_issue_it_is():
    """A GitHub issue extracts to a clean 556-char body with perfect code fences
    and zero junk — but `content` alone never says WHICH issue, WHO opened it, or
    WHEN. A model handed this has an orphan comment. (`title` does carry it, so
    the information is recoverable; `content` is what gets summarized.)"""
    content = _fixture_content("github_issue")
    assert "#2690" in content or "Issue #2690" in content, "issue number missing from content"
    assert "anatolyborodin" in content, "issue author missing from content"
    assert "Jul 24, 2015" in content, "issue date missing from content"
