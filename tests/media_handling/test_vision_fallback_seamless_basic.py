from __future__ import annotations

import pytest


@pytest.mark.basic
def test_vision_fallback_caption_prompt_includes_user_request() -> None:
    from abstractcore.media.vision_fallback import VisionFallbackHandler

    handler = VisionFallbackHandler()

    prompt = handler._build_caption_prompt(user_prompt="What's written on the sign?")
    assert "User request (for context): What's written on the sign?" in prompt
    assert "readable text" in prompt.lower()
    assert "Return only the description text." in prompt

    prompt_no_user = handler._build_caption_prompt(user_prompt=None)
    assert "User request (for context):" not in prompt_no_user


@pytest.mark.basic
def test_vision_fallback_extracts_caption_from_jsonish() -> None:
    from abstractcore.media.vision_fallback import VisionFallbackHandler

    handler = VisionFallbackHandler()

    assert handler._extract_caption_text('{"description": "A red square."}') == "A red square."
    assert handler._extract_caption_text('{"caption": "A blue circle."}') == "A blue circle."
    assert handler._extract_caption_text("Just plain text.") == "Just plain text."


@pytest.mark.basic
def test_text_only_image_fallback_preserves_user_request(monkeypatch, sample_media_files) -> None:
    from abstractcore.media.handlers import LocalMediaHandler
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType
    from abstractcore.media.vision_fallback import VisionFallbackHandler

    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description",
        lambda self, image_path, user_prompt=None: "A solid red square on a plain background.",
    )

    handler = LocalMediaHandler("openrouter", {"vision_support": False}, model_name="qwen/qwen3-next-80b")

    image = MediaContent(
        media_type=MediaType.IMAGE,
        content=str(sample_media_files["png"]),
        content_format=ContentFormat.FILE_PATH,
        mime_type="image/png",
        file_path=str(sample_media_files["png"]),
        metadata={"file_name": sample_media_files["png"].name},
    )

    user_request = "What is the dominant color in this image?"
    message = handler.create_multimodal_message(user_request, [image])

    assert isinstance(message, str)
    assert "Visual context from attached image(s)" in message
    assert "Image 1" in message
    assert "A solid red square on a plain background." in message
    assert "Now answer the user's request:" in message
    assert message.strip().endswith(user_request)

