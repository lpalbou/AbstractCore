"""Delegated sight refuses a caption the route never actually saw.

Measured 2026-08-21 on a live run (session acode-54be34bd76af, route
mlx/mlx-community/Qwen3.8-27B-4bit): `analyze_media` returned success with

    "No image or visual content is present in this conversation... The prompt
     arrived without an attached file, screenshot, or embedded picture...
     (observed by mlx/mlx-community/Qwen3.8-27B-4bit)"

— a confident, provenance-stamped description of nothing, which is exactly
what the tool's own decode gate says it exists to prevent. The gate proves the
FILE is an image; it cannot prove the ROUTE carried it. The MLX transport
generates from a text prompt and reduced the structured multimodal message to
its text part, silently.

The provider now records the drop structurally (`metadata["media_dropped"]`)
and the sight path refuses on it. Deliberately NOT prose-matching the model's
answer: "the error-substring class stays banned".
"""

from __future__ import annotations

import pytest

from abstractcore.media.vision_fallback import VisionFallbackHandler, VisionGenerationError


class _Resp:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}


class _LLM:
    def __init__(self, metadata):
        self._metadata = metadata

    def generate(self, prompt, media=None, **kwargs):
        return _Resp("a description of something", self._metadata)


def _handler_with(monkeypatch, metadata):
    import abstractcore

    monkeypatch.setattr(abstractcore, "create_llm", lambda *a, **k: _LLM(metadata), raising=False)
    return VisionFallbackHandler()


def test_a_dropped_image_is_refused_not_captioned(monkeypatch, tmp_path):
    handler = _handler_with(monkeypatch, {"media_dropped": ["image_url"]})
    with pytest.raises(VisionGenerationError) as e:
        handler.create_description_via_route("mlx", "some-model", str(tmp_path / "x.png"))
    msg = str(e.value)
    assert "did not transport the image" in msg
    assert "mlx/some-model" in msg, "the refusal must name the route that could not see"


def test_a_delivered_image_still_captions(monkeypatch, tmp_path):
    handler = _handler_with(monkeypatch, {})
    description, trace = handler.create_description_via_route(
        "openai", "gpt-vision", str(tmp_path / "x.png")
    )
    assert description.strip()
    assert trace["strategy"] == "session_route"
