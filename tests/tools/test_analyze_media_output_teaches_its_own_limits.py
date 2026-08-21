"""The observation says what it is: one bounded reading, and how to get more.

`analyze_media` returns prose that reads like a description of the image. It is
not: it is ONE model's 3-4 sentence reading, biased by whatever question was
asked. A detail nobody asked about is simply absent, and nothing in the text
tells the caller that — so a caller treats an incomplete reading as complete.

The metadata strings cannot carry it (200/240-char caps, already spent on
routing and on the "don't use this if the image is already in front of you"
rule), so the disclosure rides the OUTPUT, where it is free and lands at the
moment it matters. The line differs by case because the useful next step
differs: focus a blind reading, or re-ask a focused one.
"""

from __future__ import annotations

import pytest

from abstractcore.tools.common_tools import analyze_media


class _Handler:
    """Stands in for VisionFallbackHandler: a fixed caption, no network."""

    def __init__(self, *a, **k):
        pass

    def create_description_via_route(self, provider, model, path, user_prompt=None):
        return "A card with text on it.", {
            "backend": {"kind": "llm", "provider": provider, "model": model, "source": "session_route"},
            "strategy": "session_route",
        }


@pytest.fixture()
def image(tmp_path):
    from PIL import Image

    p = tmp_path / "card.png"
    Image.new("RGB", (24, 12), (10, 20, 30)).save(p)
    return str(p)


@pytest.fixture()
def routed(monkeypatch):
    import abstractcore.media.vision_fallback as vf

    monkeypatch.setattr(vf, "VisionFallbackHandler", _Handler)
    monkeypatch.setattr(
        "abstractcore.tools.common_tools._analyze_media_route_declares_vision",
        lambda p, m: True,
    )
    return {"provider": "endpoint:test", "model": "sees-1"}


def test_a_focused_reading_says_it_was_focused(image, routed):
    out = analyze_media(file_path=image, question="what token is on the card?", _session_route=routed)
    assert "(observed by endpoint:test/sees-1)" in out, "provenance is unchanged"
    assert "one bounded reading, focused on your question" in out
    assert "call analyze_media again with a different question" in out


def test_an_unfocused_reading_points_at_the_question_parameter(image, routed):
    out = analyze_media(file_path=image, _session_route=routed)
    assert "(observed by endpoint:test/sees-1)" in out
    assert "unfocused" in out
    assert "`question=`" in out, "the blind case must name the parameter that fixes it"


def test_a_refusal_carries_no_reading_footer(image, routed, tmp_path):
    """The line describes an observation; an error has none to describe."""
    out = analyze_media(file_path=str(tmp_path / "absent.png"), question="x", _session_route=routed)
    assert out.startswith("Error:")
    assert "bounded reading" not in out


def test_when_to_use_states_the_case_for_NOT_calling_it():
    """Live evidence 2026-08-21: a run whose image rode every LLM call still
    spent a nested vision call on analyze_media. The metadata never said not to."""
    hint = str(analyze_media.tool_definition.when_to_use or "")
    assert "NOT for an image already attached to this call" in hint
    assert "one nested vision call" in hint
