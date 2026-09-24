"""Adversarial round 4: attack the FIXES, not the old defects.

Every fix landed in rounds 1-3 is a new heuristic, and every heuristic has a
false-positive side. This file is the other half of the bar: content that the
NEW passes destroy, cost the new passes add, and behaviour at their boundaries.

Groups:
  A. the consecutive-duplicate-line collapse   (`_normalize_markdown`, ~L9246)
  B. the inline-emphasis rendering
  C. link retention on link-heavy containers   (guards, green)
  D. the withheld-evidence cap boundary        (`_apply_fetch_url_payload_policy`)
  E. cost + exception safety of the new DOM passes

Offline and deterministic. The timing budgets in group E are deliberately
generous (5-10x current measurements) so they catch a pathology, not a slow CI box.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fetch_url_fixtures"
sys.path.insert(0, str(FIXTURES))

from harness import GOLD, fixture_html  # noqa: E402

from abstractcore.tools.common_tools import (  # noqa: E402
    FETCH_URL_MAX_INLINE_EVIDENCE_CHARS,
    _apply_fetch_url_payload_policy,
    _extract_main_content,
)

_PROSE = "<p>" + ("Real sentence carrying the meaning of this page. " * 20) + "</p>"


def _content(body_html: str, *, url: str = "https://example.com/x") -> str:
    html = f"<html><head><title>T</title></head><body><article>{body_html}{_PROSE}</article></body></html>"
    return str(_extract_main_content(html, url, keep_links=True).get("content") or "")


# ---------------------------------------------------------------------------
# A. the consecutive-duplicate-line collapse eats legitimate repetition
# ---------------------------------------------------------------------------


def test_identical_table_rows_all_survive():
    """`_normalize_markdown` exempts table rows from the dedup key check --
    `if not stripped.startswith("|")` -- but an EARLIER branch,
    `if prev_line == stripped: continue`, runs first and has no such exemption.
    So byte-identical `| 0 | 0 |` rows are dropped before the exemption is
    consulted.

    A matrix of zeros, an SLA table of "0 incidents", a compatibility grid full
    of "Yes" -- all silently truncated to one row.
    """
    content = _content(
        "<h1>Incidents</h1><table><tr><th>sev1</th><th>sev2</th></tr>"
        "<tr><td>0</td><td>0</td></tr>"
        "<tr><td>0</td><td>0</td></tr>"
        "<tr><td>0</td><td>0</td></tr></table>"
    )
    rows = [ln for ln in content.splitlines() if ln.strip() == "| 0 | 0 |"]
    assert len(rows) == 3, f"3 identical data rows collapsed to {len(rows)}:\n{content[:400]}"


def test_labelled_table_rows_are_not_affected():
    """Control: the same table keeps every row once the rows differ. Confirms
    the defect above is the exact-line branch, not table rendering."""
    content = _content(
        "<h1>Incidents</h1><table><tr><th>quarter</th><th>sev1</th><th>sev2</th></tr>"
        "<tr><td>Q1</td><td>0</td><td>0</td></tr>"
        "<tr><td>Q2</td><td>0</td><td>0</td></tr>"
        "<tr><td>Q3</td><td>0</td><td>0</td></tr></table>"
    )
    assert content.count("| 0 | 0 |") == 3


def test_repeated_list_items_all_survive():
    """A changelog with three "No changes." entries becomes one entry."""
    content = _content(
        "<h1>Changelog</h1><ul>"
        "<li>No changes.</li><li>No changes.</li><li>No changes.</li></ul>"
    )
    items = [ln for ln in content.splitlines() if ln.strip() == "- No changes."]
    assert len(items) == 3, f"3 list items collapsed to {len(items)}:\n{content[:300]}"


def test_repeated_paragraphs_all_survive():
    """A refrain is real content: three <p>Nevermore.</p> are three lines."""
    content = _content("<h1>Poem</h1><p>Nevermore.</p><p>Nevermore.</p><p>Nevermore.</p>")
    assert content.count("Nevermore.") == 3, f"refrain collapsed:\n{content[:300]}"


def test_visually_duplicated_timestamp_is_still_collapsed():
    """The other half of the bar: the defect the dedup pass exists for (a mobile
    copy and a desktop copy of the same card) must STAY fixed. Green today.

    Uses BLOCK elements, which is the real BBC shape. Sibling inline <span>s with
    no whitespace between them render fused ("2 days ago2 days ago") — but a
    browser does exactly the same, so that is faithful rendering, not a defect,
    and the line-based dedup correctly cannot see it."""
    content = _content(
        '<div class="card"><h2>Headline one</h2>'
        '<div class="mobile">2 days ago</div><div class="desktop">2 days ago</div></div>'
    )
    assert content.count("2 days ago") == 1, f"visual duplicate leaked twice:\n{content[:300]}"


# ---------------------------------------------------------------------------
# B. inline emphasis vs literal asterisks
# ---------------------------------------------------------------------------


def test_literal_asterisks_inside_emphasis_are_not_corrupted():
    """`<em>*args</em>` renders as `**args*` -- the literal leading asterisk
    fuses with the emphasis marker into malformed markdown that reads as
    bold-"args"-italic. Python docs, C docs and shell docs are full of `*args`,
    `**kwargs`, `*.py` and `*ptr` outside <code>.

    NOTE: `pydocs_json` does NOT cover this -- that page wraps every such token
    in <code>, where backticks protect it. The corpus had a hole here.
    """
    content = _content("<p>Pass <em>*args</em> and <em>**kwargs</em> through.</p>")
    assert "**args*" not in content, f"literal asterisk corrupted by emphasis markers: {content[:200]}"
    assert "*args" in content and "**kwargs" in content


def test_code_wrapped_asterisks_survive():
    """Control, green: inside <code> the backticks protect the asterisks."""
    content = _content("<p>Pass <code>**kwargs</code> through.</p>")
    assert "`**kwargs`" in content


def test_heading_emphasis_is_still_stripped():
    """The fix this group attacks must stay fixed: `## **Bold**` -> `## Bold`."""
    content = _content("<h2><strong>Breaking News</strong></h2>")
    assert "## Breaking News" in content, content[:200]


# ---------------------------------------------------------------------------
# C. link retention (guards -- green today)
# ---------------------------------------------------------------------------


def test_navigation_link_rail_is_still_dropped():
    links = "".join(
        f'<li><a href="https://example.com/a{i}">Related story number {i}</a></li>' for i in range(60)
    )
    content = _content(f"<h1>Article</h1><nav><ul>{links}</ul></nav>")
    assert "Related story number 7" not in content
    assert len(content) < 3000, f"nav rail bloated content to {len(content)} chars"


def test_curated_in_body_link_list_is_kept():
    """The false-positive side: a link list INSIDE the article body is content
    (a docs index, a further-reading section) and must survive with its URLs."""
    links = "".join(
        f'<li><a href="https://example.com/a{i}">Further reading number {i}</a></li>' for i in range(20)
    )
    content = _content(f"<h1>Article</h1><h2>Further reading</h2><ul>{links}</ul>")
    assert content.count("](http") >= 20, "in-body link list lost its URLs"


# ---------------------------------------------------------------------------
# D. the withheld-evidence cap boundary
# ---------------------------------------------------------------------------


def _policy(content: str, evidence_len: int):
    return _apply_fetch_url_payload_policy(
        {
            "content": content,
            "content_type": "text/html",
            "raw_text": "y" * evidence_len,
            "normalized_text": "z" * evidence_len,
        }
    )


@pytest.mark.parametrize(
    "evidence_len,withheld",
    [
        (FETCH_URL_MAX_INLINE_EVIDENCE_CHARS - 1, False),
        (FETCH_URL_MAX_INLINE_EVIDENCE_CHARS, False),
        (FETCH_URL_MAX_INLINE_EVIDENCE_CHARS + 1, True),
    ],
)
def test_withholding_boundary_is_inclusive_and_exact(evidence_len: int, withheld: bool):
    """Green guard: the cap is `len(value) <= cap` stays inline. Pins the edge so
    an off-by-one cannot drift in unnoticed."""
    result = _policy("x" * 5000, evidence_len)
    assert (result["raw_text"] is None) is withheld
    assert ("raw_text_withheld" in result) is withheld


def test_evidence_is_kept_when_content_is_empty():
    """Green guard: the 'never withhold the only copy' rule."""
    result = _policy("", 50_000)
    assert isinstance(result["raw_text"], str) and result["raw_text"], (
        "the only copy of the text was withheld"
    )


def test_evidence_is_kept_when_content_is_below_the_real_content_floor():
    """The gap in that rule: it triggers only on `content_chars <= 0`. A page
    whose extraction produced a few junk characters -- below the module's own
    `_MIN_REAL_CONTENT_CHARS = 200` floor -- has its 50KB of real source text
    withheld in favour of 4 chars of nothing.

    The guard should use the same real-content floor the rest of the file does.
    """
    from abstractcore.tools.common_tools import _MIN_REAL_CONTENT_CHARS

    result = _policy("tiny", 50_000)
    assert _MIN_REAL_CONTENT_CHARS > 4, "premise stale"
    assert isinstance(result["raw_text"], str) and result["raw_text"], (
        "content was 4 chars (below the real-content floor) yet 50,000 chars of source "
        "evidence were withheld — the run keeps neither a usable payload nor the bytes"
    )


# ---------------------------------------------------------------------------
# E. cost and exception safety of the new DOM passes
# ---------------------------------------------------------------------------

BIG_FIXTURES = ["ja_wikipedia", "wikipedia_bert", "do_pricing", "bbc_tech_hub", "supabase_pricing"]
# Measured 2026-08-21: worst real fixture is wikipedia_bert at ~0.6s (487KB) on
# a developer machine; shared CI runners have taken 3.1s. 8.0s absorbs runner
# noise while still catching a pathology (a quadratic pass costs minutes).
MAX_EXTRACTION_SECONDS = 8.0


@pytest.mark.parametrize("gold_key", BIG_FIXTURES)
def test_extraction_of_a_large_real_page_is_bounded(gold_key: str):
    html = fixture_html(gold_key)
    start = time.perf_counter()
    _extract_main_content(html, GOLD[gold_key]["url"], keep_links=True)
    elapsed = time.perf_counter() - start
    assert elapsed < MAX_EXTRACTION_SECONDS, (
        f"{gold_key} ({len(html)//1024}KB) took {elapsed:.2f}s"
    )


def test_deeply_nested_markup_does_not_blow_up_the_cost():
    """Nesting cost is superlinear: depth 1000 ~1.5s, 2000 ~5.7s, 4000 ~20s,
    10000 ~90s -- on a page of only 108KB. `fetch_url`'s timeout covers the HTTP
    fetch, not extraction, so a badly-generated (or hostile) page hangs the
    agent well past any deadline. HEAD was quadratic too; the new passes roughly
    double the constant."""
    html = "<html><body>" + "<div>" * 2000 + _PROSE + "</div>" * 2000 + "</body></html>"
    start = time.perf_counter()
    _extract_main_content(html, "https://example.com/x", keep_links=True)
    elapsed = time.perf_counter() - start
    # ~0.25s on a developer machine; a shared CI runner (Python 3.9) took 2.05s
    # on 2026-09-24. 5.0s absorbs runner noise and still catches the old
    # quadratic cost (~5.7s on a developer machine, far more on a runner).
    assert elapsed < 5.0, f"2000 nested divs ({len(html)//1024}KB) took {elapsed:.2f}s"


MALFORMED = {
    "unclosed_tags": "<html><body><article><div><p>Text that never closes" + _PROSE + "<div><span><b>more",
    "minified_one_line": "<html><body><article>" + ("<span>x</span>" * 20000) + _PROSE + "</article></body></html>",
    "mismatched_fences": "<html><body><article><pre>```python\nprint(1)</pre><p>text ``` unclosed</p>"
    + _PROSE
    + "</article></body></html>",
    "nul_and_control_chars": "<html><body><article><p>a\x00b\x07c\x1fd</p>" + _PROSE + "</article></body></html>",
    "broken_entities": "<html><body><article><p>&amp&#xZZ;&#999999999;&lt</p>" + _PROSE + "</article></body></html>",
    "orphan_table_cell": "<html><body><article><table><td>orphan</td></table>" + _PROSE + "</article></body></html>",
    "empty_body": "<html><body></body></html>",
    "attribute_bomb": "<html><body><article><div "
    + " ".join(f'data-x{i}="v"' for i in range(5000))
    + ">"
    + _PROSE
    + "</div></article></body></html>",
    "no_html_at_all": "just a bare sentence with no markup whatsoever",
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_adversarial_markup_degrades_instead_of_raising(name: str):
    """House style: every DOM walk is wrapped so a malformed page degrades. A new
    pass that raises on adversarial markup turns a bad page into a tool crash.
    Green today across all nine shapes — keep it that way."""
    result = _extract_main_content(MALFORMED[name], "https://example.com/x", keep_links=True)
    assert isinstance(result, dict)
    assert set(result) >= {"title", "content", "text"}
