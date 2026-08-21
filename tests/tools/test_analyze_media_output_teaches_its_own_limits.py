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


def test_the_footer_echoes_the_question_that_shaped_the_reading(image, routed):
    """Generic advice is unusable: the caller decides its next move from what
    this reading was actually pointed at, so the question comes back verbatim."""
    out = analyze_media(file_path=image, question="what token is on the card?", _session_route=routed)
    assert "(observed by endpoint:test/sees-1)" in out, "provenance is unchanged"
    assert 'asked: "what token is on the card?"' in out
    assert "card.png" in out, "and which file it read"


def test_the_next_call_is_written_out_not_described(image, routed):
    """A hint the caller has to translate into a call is half a hint — and it
    echoes the PATH THIS CALL USED, so it is re-callable even when a host
    rewrote file_path (a materialized attachment copy). Naming the pretty
    basename there would suggest a call that resolves to nothing."""
    out = analyze_media(file_path=image, question="what token?", _session_route=routed)
    assert f'analyze_media(file_path="{image}", question="<what you need>")' in out


def test_core_never_offers_attaching(image, routed):
    """Whether these bytes CAN be attached is a session fact core lacks.
    `open_attachment` resolves session attachments only, so offering it for a
    plain disk file would be exactly the dead-end advice this work removed.
    The HOST appends that call when it is real (runtime test covers it)."""
    out = analyze_media(file_path=image, question="what token?", _session_route=routed)
    assert "open_attachment" not in out
    assert "attach the image" not in out.lower()


def test_a_long_question_is_bounded_in_the_echo(image, routed):
    """The echo is the caller's own text and could be arbitrarily long."""
    q = "x" * 400
    out = analyze_media(file_path=image, question=q, _session_route=routed)
    assert q not in out, "the full 400-char question must not ride the footer"
    assert "x" * 99 + "…" in out, "it is cut at 100 with the cut marked"


def test_an_unfocused_reading_points_at_the_question_parameter(image, routed):
    out = analyze_media(file_path=image, _session_route=routed)
    assert "(observed by endpoint:test/sees-1)" in out
    assert "no question asked" in out
    assert f'analyze_media(file_path="{image}", question="<what you need>")' in out


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
    assert "One nested vision call per use" in hint
