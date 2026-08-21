"""Adversarial: does the fetch_url payload policy break its DOWNSTREAM consumers?

Round 4 (2026-08-21). `fetch_url` now withholds `raw_text`/`normalized_text`
above FETCH_URL_MAX_INLINE_EVIDENCE_CHARS, setting them to None and leaving a
`*_withheld` descriptor. Three consumers were written against the OLD contract,
where those two keys WERE the text.

The failure is size-dependent, which is what makes it dangerous: a page whose
normalized text is under the cap still works, so hand-built test fixtures with
short strings stay green while every real article silently yields nothing.

In-repo consumer pinned here: `abstractcore.processing.basic_deepsearch`.
Out-of-repo consumer (reported, not testable from here):
`abstractruntime/src/abstractruntime/evidence/recorder.py`.

Offline and deterministic: the committed fixtures rebuilt through
`harness.offline_envelope`, which applies the same payload policy the tool does.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fetch_url_fixtures"
sys.path.insert(0, str(FIXTURES))

from harness import GOLD, offline_envelope  # noqa: E402

import abstractcore.processing.basic_deepsearch as basic_deepsearch  # noqa: E402
from abstractcore.tools.common_tools import (  # noqa: E402
    FETCH_URL_MAX_INLINE_EVIDENCE_CHARS,
)

# Fixtures whose normalized text is comfortably OVER the withholding cap, i.e.
# every ordinary article.
BIG_FIXTURES = ["techstartups", "wikipedia_bert", "do_pricing", "ja_wikipedia"]


class _DummyLLM:
    provider = "dummy"
    model = "dummy"


def _searcher():
    return basic_deepsearch.BasicDeepSearch(llm=_DummyLLM())


# ---------------------------------------------------------------------------
# the withheld descriptor itself must be trustworthy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gold_key", BIG_FIXTURES)
def test_withheld_descriptor_is_a_verifiable_receipt(gold_key: str):
    """An auditor's only remaining handle on the withheld bytes. It has to carry
    enough to prove WHAT was withheld, and its digest has to be right — a wrong
    sha256 is worse than none, because it looks checkable."""
    env = offline_envelope(gold_key)
    for field in ("raw_text", "normalized_text"):
        assert env[field] is None, f"{gold_key}: {field} was not withheld; test premise is stale"
        descriptor = env.get(f"{field}_withheld")
        assert isinstance(descriptor, dict), f"{gold_key}: no {field}_withheld descriptor"
        assert set(descriptor) >= {"chars", "bytes", "sha256", "reason"}, descriptor
        assert descriptor["chars"] > FETCH_URL_MAX_INLINE_EVIDENCE_CHARS
        assert len(str(descriptor["sha256"])) == 64


def test_withheld_sha256_matches_the_text_that_was_withheld():
    """Recompute the digest from the source the tool withheld and compare."""
    from abstractcore.tools.common_tools import _normalize_text_for_evidence

    key = "techstartups"
    # Decode from BYTES, not read_text(): text mode collapses CRLF to LF and the
    # digest is over what the transport actually delivered.
    html = (FIXTURES / GOLD[key]["fixture"]).read_bytes().decode("utf-8", errors="replace")
    env = offline_envelope(key)

    expected_raw = hashlib.sha256(html.encode("utf-8", errors="replace")).hexdigest()
    assert env["raw_text_withheld"]["sha256"] == expected_raw, (
        "raw_text_withheld's digest does not match the bytes it claims to describe"
    )

    normalized = _normalize_text_for_evidence(
        raw_text=html, content_type_header="text/html; charset=utf-8", url=GOLD[key]["url"]
    )
    expected_norm = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
    assert env["normalized_text_withheld"]["sha256"] == expected_norm


# ---------------------------------------------------------------------------
# basic_deepsearch — the in-repo consumer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gold_key", BIG_FIXTURES)
def test_deepsearch_full_text_extraction_survives_withheld_evidence(gold_key: str):
    """`_extract_full_text_from_fetch_output` (basic_deepsearch.py:1545) reads
    `normalized_text or raw_text` and only then falls back to `rendered`. Both
    are now None for any real article, so it drops from the full article to a
    ~4KB rendered preview — or to nothing."""
    env = offline_envelope(gold_key)
    content = str(env.get("content") or "")
    got = _searcher()._extract_full_text_from_fetch_output(dict(env))
    assert len(got) >= 0.5 * len(content), (
        f"{gold_key}: deepsearch recovered {len(got)} chars from a page whose `content` is "
        f"{len(content)} chars (raw_text/normalized_text are withheld)"
    )


@pytest.mark.parametrize("gold_key", BIG_FIXTURES)
def test_deepsearch_structured_parse_keeps_the_full_text(gold_key: str):
    """`_parse_fetch_url_output` (basic_deepsearch.py:1412-1427) sets
    `_full_text`/`text_preview` from `normalized_text`, else `raw_text`. With
    both withheld it sets neither, so downstream relevance scoring has no text."""
    env = offline_envelope(gold_key)
    structured = _searcher()._parse_fetch_url_output(dict(env))
    assert structured.get("_full_text"), (
        f"{gold_key}: _parse_fetch_url_output produced no _full_text; keys={sorted(structured)}"
    )
    assert structured.get("text_preview"), f"{gold_key}: no text_preview"


@pytest.mark.parametrize("gold_key", BIG_FIXTURES)
def test_deepsearch_does_not_discard_a_page_it_successfully_fetched(
    monkeypatch: pytest.MonkeyPatch, gold_key: str
):
    """The sharpest one. basic_deepsearch.py:800 computes
    `extracted_text = normalized_text or raw_text` and, when empty, logs
    "Skipping URL because fetch returned no extracted text" and CONTINUES.

    So a successful fetch of a 29KB article is discarded as empty. Search
    returns nothing and nothing in the logs says the text was in `content` all
    along.
    """
    env = offline_envelope(gold_key)
    url = GOLD[gold_key]["url"]

    monkeypatch.setattr(
        basic_deepsearch,
        "web_search",
        lambda *a, **k: json.dumps(
            {
                "success": True,
                "results": [{"rank": 1, "title": "Example", "url": url, "snippet": "example"}],
            }
        ),
    )
    monkeypatch.setattr(basic_deepsearch, "fetch_url", lambda *a, **k: dict(env))

    findings = _searcher()._execute_search(
        "task-1", "example query", basic_deepsearch.SourceManager(max_sources=5), set()
    )
    assert findings, (
        f"{gold_key}: deepsearch dropped a successful fetch of a "
        f"{len(str(env.get('content') or ''))}-char page as 'no extracted text'"
    )


def test_deepsearch_still_skips_a_fetch_that_really_has_no_text(
    monkeypatch: pytest.MonkeyPatch,
):
    """The other half of the bar, and a guard on the fix above: when there is
    genuinely no text — no content either — the URL must still be skipped.
    Mirrors test_basic_deepsearch_skips_rendered_only_fetch_success so a fix
    cannot buy the tests above by accepting empty pages.
    """
    monkeypatch.setattr(
        basic_deepsearch,
        "web_search",
        lambda *a, **k: json.dumps(
            {
                "success": True,
                "results": [
                    {"rank": 1, "title": "Example", "url": "https://example.com/doc", "snippet": "x"}
                ],
            }
        ),
    )
    monkeypatch.setattr(
        basic_deepsearch,
        "fetch_url",
        lambda *a, **k: {
            "success": True,
            "rendered": "metadata only",
            "content": None,
            "raw_text": None,
            "normalized_text": None,
        },
    )
    findings = _searcher()._execute_search(
        "task-2", "example query", basic_deepsearch.SourceManager(max_sources=5), set()
    )
    assert findings == []


def test_small_pages_hide_the_regression():
    """Documents WHY this was not caught: under the cap nothing is withheld, so
    a short fixture behaves exactly as before. Every hand-built dict in the
    existing contract tests is short."""
    env = offline_envelope("github_issue")
    assert env["normalized_text"] is not None, "premise stale: this fixture is no longer small"
    assert _searcher()._extract_full_text_from_fetch_output(dict(env)).strip(), (
        "even the small-page path is broken"
    )
