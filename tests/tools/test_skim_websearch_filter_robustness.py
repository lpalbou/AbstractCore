"""Filter-path robustness tests for skim_websearch (and the web_search HTML fallback extractor).

Live incident pinned here: a query about "Game Boy" returned 15 results with
counts {"fetched":15,"matched":0,"returned":0} because the ddgs backend
delivered snippet bodies with words fused at highlight-tag boundaries
("TheGameBoyhas foursoundchannels") — the multi-word required term
"game boy" is not an exact substring of the fused text. All backend payloads
here are synthetic (web_search is monkeypatched); no network involved.
"""

from __future__ import annotations

import json
import types

import pytest

pytestmark = pytest.mark.basic


def _patch_search(monkeypatch: pytest.MonkeyPatch, results: list[dict[str, object]]) -> None:
    import abstractcore.tools.common_tools as common_tools

    sample = {
        "success": True,
        "status_hint": "ok",
        "degraded": False,
        "backend_used": "ddgs.text",
        "query": "synthetic",
        "results": results,
    }
    monkeypatch.setattr(common_tools, "web_search", lambda *args, **kwargs: json.dumps(sample))


def _skim(**kwargs: object) -> dict[str, object]:
    import abstractcore.tools.common_tools as common_tools

    return json.loads(common_tools.skim_websearch(**kwargs))  # type: ignore[arg-type]


def test_fused_snippet_matches_via_whitespace_insensitive_fallback_with_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The exact fusion shape from the live incident (ddgs highlight-tag stripping).
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "Audio - Pan Docs",
                "url": "https://gbdev.io/pandocs/Audio.html",
                "snippet": "TheGameBoyhas foursoundchannels: twopulsechannels, a wave channel and a noise channel",
            },
            {
                "rank": 2,
                "title": "Unrelated",
                "url": "https://example.com/other",
                "snippet": "Nothing about handheld consoles here",
            },
        ],
    )

    data = _skim(
        query="Pan Docs Game Boy audio",
        required_terms=["Game Boy"],
        match="all",
        require_in="snippet",
        num_results=5,
    )

    assert data["counts"] == {"fetched": 2, "matched": 1, "returned": 1}
    assert [r["url"] for r in data["results"]] == ["https://gbdev.io/pandocs/Audio.html"]
    assert "whitespace-insensitive" in str(data.get("note") or "")
    assert "hint" not in data


def test_clean_snippet_matches_via_primary_path_with_no_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "The Game Boy APU",
                "url": "https://example.com/apu",
                "snippet": "The Game Boy APU contains 4 audio channels.",
            }
        ],
    )

    data = _skim(query="gb audio", required_terms=["Game Boy"], require_in="snippet")

    assert data["counts"] == {"fetched": 1, "matched": 1, "returned": 1}
    assert "note" not in data
    assert "hint" not in data


def test_nbsp_in_snippet_matches_via_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # &nbsp; survives some extraction paths as \xa0; a plain-space term must
    # still be able to match (whitespace corruption class, not incident-only).
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "Handheld audio",
                "url": "https://example.com/nbsp",
                "snippet": "The Game\u00a0Boy has four sound channels.",
            }
        ],
    )

    data = _skim(query="gb audio", required_terms=["Game Boy"], require_in="snippet")

    assert data["counts"]["matched"] == 1
    assert "whitespace-insensitive" in str(data.get("note") or "")


def test_whitespace_free_terms_never_take_the_elided_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Eliding the haystack fuses "java script" into "javascript"; a
    # whitespace-free term like "ascript" would then match across the word
    # boundary. The fallback is restricted to whitespace-bearing terms, so
    # this must stay a non-match.
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "Languages",
                "url": "https://example.com/lang",
                "snippet": "A comparison of java script dialects",
            }
        ],
    )

    data = _skim(query="languages", required_terms=["ascript"], require_in="snippet")

    assert data["counts"]["matched"] == 0


def test_elided_fallback_does_not_match_across_field_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Title ends with "game", snippet starts with (whitespace +) "boy". The
    # primary space-joined haystack reads "game  boy" (double space) so the
    # exact-substring check fails; eliding the WHOLE haystack would fuse it
    # into "...gameboy..." and manufacture a phantom cross-field phrase hit.
    # Fields are elided separately, so this must stay a non-match.
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "The greatest game",
                "url": "https://example.com/x",
                "snippet": " boy adventures in a castle",
            }
        ],
    )

    data = _skim(query="games", required_terms=["Game Boy"], require_in="title_snippet")

    assert data["counts"]["matched"] == 0


def test_match_all_mixes_primary_and_fallback_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    # One term matches exactly, the other only via elision: match='all' must
    # succeed and the result counts as a whitespace-insensitive match.
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "Pan Docs",
                "url": "https://example.com/pd",
                "snippet": "TheGameBoyhas four sound channels and a frame sequencer",
            }
        ],
    )

    data = _skim(
        query="gb audio",
        required_terms=["Game Boy", "frame sequencer"],
        match="all",
        require_in="snippet",
    )

    assert data["counts"]["matched"] == 1
    assert "whitespace-insensitive" in str(data.get("note") or "")


def test_zero_match_hint_reports_empty_backend_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(
        monkeypatch,
        [
            {"rank": i, "title": f"Game Boy page {i}", "url": f"https://example.com/{i}", "snippet": ""}
            for i in range(1, 6)
        ],
    )

    data = _skim(query="gb audio", required_terms=["Game Boy"], require_in="snippet")

    assert data["counts"] == {"fetched": 5, "matched": 0, "returned": 0}
    hint = str(data.get("hint") or "")
    assert "5 of 5 fetched results have empty snippets" in hint
    assert "require_in='title_snippet'" in hint


def test_zero_match_hint_points_at_wider_scope_when_terms_live_in_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "Game Boy APU deep dive",
                "url": "https://example.com/a",
                "snippet": "Four channels of audio.",
            },
            {
                "rank": 2,
                "title": "Game Boy sound registers",
                "url": "https://example.com/b",
                "snippet": "NR52 controls power.",
            },
        ],
    )

    data = _skim(query="gb audio", required_terms=["Game Boy"], require_in="snippet")

    assert data["counts"]["matched"] == 0
    hint = str(data.get("hint") or "")
    assert "2 result(s) match in the wider 'title_snippet' scope" in hint
    assert "Retry with require_in='title_snippet'" in hint


def test_zero_match_hint_keeps_generic_guidance_when_terms_are_truly_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "Gardening tips",
                "url": "https://example.com/garden",
                "snippet": "How to grow tomatoes",
            }
        ],
    )

    data = _skim(query="gardening", required_terms=["Game Boy"], require_in="snippet")

    assert data["counts"]["matched"] == 0
    assert data["hint"] == "No matches. Try fewer required_terms or match='any'."


def test_counts_are_truthful_and_num_results_is_honored_after_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_search(
        monkeypatch,
        [
            {
                "rank": i,
                "title": f"Game Boy article {i}",
                "url": f"https://example.com/{i}",
                "snippet": "Game Boy audio hardware notes",
            }
            for i in range(1, 11)
        ],
    )

    data = _skim(query="gb", required_terms=["Game Boy"], require_in="snippet", num_results=3)

    assert data["counts"] == {"fetched": 10, "matched": 10, "returned": 3}
    assert len(data["results"]) == 3


def test_required_terms_accepts_json_encoded_string_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tool-call transports often deliver list arguments as their JSON source
    # text; the literal brackets/quotes must not become part of the term.
    _patch_search(
        monkeypatch,
        [
            {
                "rank": 1,
                "title": "APU",
                "url": "https://example.com/apu",
                "snippet": "The Game Boy APU has four channels; latency is low.",
            }
        ],
    )

    data = _skim(query="gb", required_terms='["Game Boy", "latency"]', match="all", require_in="snippet")

    assert data["filter"]["required_terms"] == ["game boy", "latency"]
    assert data["counts"]["matched"] == 1

    data_single = _skim(query="gb", required_terms='"Game Boy"', require_in="snippet")
    assert data_single["filter"]["required_terms"] == ["game boy"]
    assert data_single["counts"]["matched"] == 1


def test_web_search_html_fallback_does_not_fuse_words_at_tag_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractcore.tools.common_tools as common_tools

    if not common_tools._ensure_requests():
        pytest.skip('requests not available; install with: pip install "abstractcore[tools]"')

    page = (
        "<html><body>"
        '<a class="result__a" href="https://example.com/gb">The <b>Game</b><b>Boy</b> APU</a>'
        '<a class="result__snippet">The<b>Game</b><b>Boy</b>has four<b>sound</b>channels &amp; a noise channel</a>'
        "</body></html>"
    )

    class _FakeResp:
        text = page

        def raise_for_status(self) -> None:
            return None

    # Force the duckduckgo.html fallback (pretend ddgs is not importable).
    monkeypatch.setattr(common_tools, "_import_ddgs_class", lambda: (None, "ddgs not installed (test)"))
    monkeypatch.setattr(common_tools, "requests", types.SimpleNamespace(get=lambda *a, **k: _FakeResp()))

    data = json.loads(common_tools.web_search("game boy audio", num_results=3))

    assert data["backend_used"] == "duckduckgo.html"
    result = data["results"][0]
    # Tag boundaries become spaces (then collapse); entities still unescape.
    assert result["title"] == "The Game Boy APU"
    assert result["snippet"] == "The Game Boy has four sound channels & a noise channel"
    assert "GameBoyhas" not in result["snippet"]
