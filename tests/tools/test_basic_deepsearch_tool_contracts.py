from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.basic


class _DummyLLM:
    provider = "dummy"
    model = "dummy"


def test_basic_deepsearch_extracts_titles_from_structured_search_json() -> None:
    from abstractcore.processing.basic_deepsearch import BasicDeepSearch

    searcher = BasicDeepSearch(llm=_DummyLLM())
    payload = json.dumps(
        {
            "success": True,
            "results": [
                {"rank": 1, "title": "Example One", "url": "https://example.com/one", "snippet": "first"},
                {"rank": 2, "title": "Example Two", "url": "https://example.com/two", "snippet": "second"},
            ],
        }
    )

    urls = searcher._extract_urls_from_search(payload)

    assert urls == [
        ("https://example.com/one", "Example One"),
        ("https://example.com/two", "Example Two"),
    ]


def test_basic_deepsearch_preserves_search_snippets_in_candidates() -> None:
    from abstractcore.processing.basic_deepsearch import BasicDeepSearch

    searcher = BasicDeepSearch(llm=_DummyLLM())
    payload = json.dumps(
        {
            "success": True,
            "results": [
                {
                    "rank": 1,
                    "title": "Example One",
                    "url": "https://example.com/one",
                    "snippet": "This snippet contains enough detail to guide fetch decisions.",
                }
            ],
        }
    )

    candidates = searcher._extract_search_candidates(payload)

    assert candidates == [
        {
            "url": "https://example.com/one",
            "title": "Example One",
            "snippet": "This snippet contains enough detail to guide fetch decisions.",
            "rank": 1,
        }
    ]


def test_basic_deepsearch_parses_fetch_url_dict_output() -> None:
    from abstractcore.processing.basic_deepsearch import BasicDeepSearch

    searcher = BasicDeepSearch(llm=_DummyLLM(), full_text_extraction=True)
    structured = searcher._parse_fetch_url_output(
        {
            "rendered": "📰 Title: Example Doc\n📝 Description: Example Summary",
            "normalized_text": "Paragraph one.\n\nParagraph two.",
        }
    )

    assert structured is not None
    assert structured["title"] == "Example Doc"
    assert structured["description"] == "Example Summary"
    assert "Paragraph one." in structured["_full_text"]


def test_basic_deepsearch_keeps_wider_structured_preview_for_non_full_text() -> None:
    from abstractcore.processing.basic_deepsearch import BasicDeepSearch

    searcher = BasicDeepSearch(llm=_DummyLLM(), full_text_extraction=False)
    long_text = "Alpha Beta Gamma. " * 300
    structured = searcher._parse_fetch_url_output({"normalized_text": long_text})

    assert structured is not None
    assert len(structured["text_preview"]) > 2000
    assert len(structured["text_preview"]) <= 2400


def test_basic_deepsearch_does_not_synthesize_failed_search_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.processing.basic_deepsearch as basic_deepsearch

    searcher = basic_deepsearch.BasicDeepSearch(llm=_DummyLLM())
    monkeypatch.setattr(
        basic_deepsearch,
        "web_search",
        lambda *args, **kwargs: json.dumps(
            {
                "success": False,
                "status_hint": "error",
                "error": "backend unavailable",
                "results": [],
            }
        ),
    )

    findings = searcher._execute_search(
        "task-1",
        "example query",
        basic_deepsearch.SourceManager(max_sources=5),
        set(),
    )

    assert findings == []


def test_basic_deepsearch_merges_search_snippet_into_relevant_content() -> None:
    from abstractcore.processing.basic_deepsearch import BasicDeepSearch

    searcher = BasicDeepSearch(llm=_DummyLLM())
    merged = searcher._merge_search_snippet("**Content:** Detailed page body", "Short useful search snippet")

    assert merged is not None
    assert merged.startswith("**Search Snippet:** Short useful search snippet")


def test_basic_deepsearch_skips_rendered_only_fetch_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.processing.basic_deepsearch as basic_deepsearch

    searcher = basic_deepsearch.BasicDeepSearch(llm=_DummyLLM())
    monkeypatch.setattr(
        basic_deepsearch,
        "web_search",
        lambda *args, **kwargs: json.dumps(
            {
                "success": True,
                "results": [
                    {"rank": 1, "title": "Example", "url": "https://example.com/doc", "snippet": "example"},
                ],
            }
        ),
    )
    monkeypatch.setattr(
        basic_deepsearch,
        "fetch_url",
        lambda *args, **kwargs: {
            "success": True,
            "rendered": "metadata only",
            "raw_text": None,
            "normalized_text": None,
        },
    )

    findings = searcher._execute_search(
        "task-2",
        "example query",
        basic_deepsearch.SourceManager(max_sources=5),
        set(),
    )

    assert findings == []
