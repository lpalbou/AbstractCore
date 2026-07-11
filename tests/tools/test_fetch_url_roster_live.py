"""Live roster: fetch_url across a diverse class spread of real URLs.

Env-gated (ABSTRACT_E2E_FETCH_URL=1) because the network is nondeterministic.
The roster is chosen to span the CLASSES fetch_url must handle, not a set of
easy pages: encyclopedia (EN + non-English), framework docs, MDN, a JSON API,
an academic abstract, a link-aggregator front page, an independent blog, and a
news site. The contract asserted for every entry: EITHER success with a
non-trivial `content` field, OR a structured actionable error (error_class +
suggestions) — never a silent empty success.
"""
from __future__ import annotations

import os

import pytest

from abstractcore.tools.common_tools import fetch_url

ROSTER = [
    ("wikipedia_en", "https://en.wikipedia.org/wiki/Recursive_self-improvement"),
    ("wikipedia_de", "https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz"),
    ("python_docs", "https://docs.python.org/3/library/asyncio-task.html"),
    ("mdn_docs", "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After"),
    ("github_json_api", "https://api.github.com/repos/python/cpython"),
    ("arxiv_abstract", "https://arxiv.org/abs/1706.03762"),
    ("hacker_news", "https://news.ycombinator.com/"),
    ("independent_blog", "https://simonwillison.net/2023/Dec/31/ai-in-2023/"),
    ("python_blog", "https://blog.python.org/2024/05/python-3130-beta-1-released.html"),
    ("news_fr", "https://www.lemonde.fr/pixels/article/2024/05/13/intelligence-artificielle-openai_6233070_4408996.html"),
]

_ACTIONABLE_ERROR_CLASSES = {
    "bot_challenge", "rate_limited", "auth_required", "not_found",
    "gone", "server_error", "client_error", "js_required", "empty_content",
}


@pytest.mark.skipif(
    os.getenv("ABSTRACT_E2E_FETCH_URL") != "1",
    reason="Set ABSTRACT_E2E_FETCH_URL=1 to run the live roster.",
)
@pytest.mark.parametrize("name,url", ROSTER, ids=[n for n, _ in ROSTER])
def test_roster_url_rich_or_actionable(name: str, url: str) -> None:
    result = fetch_url(url=url, timeout=45)
    if result.get("success"):
        detected = result.get("detected_as")
        content = str(result.get("content") or "")
        # Rich success: real content, or a structured non-HTML payload (JSON/XML).
        assert content.strip() or detected in {"json", "xml"}, (
            f"{name}: success but empty content (silent-empty forbidden)"
        )
        if detected == "html":
            assert len(content) > 400, f"{name}: HTML content too thin ({len(content)} chars)"
    else:
        # Failure MUST be actionable: classified + at least one suggestion.
        assert result.get("error_class") in _ACTIONABLE_ERROR_CLASSES, (
            f"{name}: unclassified error {result.get('error_class')!r}"
        )
        assert result.get("suggestions"), f"{name}: error carries no suggestions"
