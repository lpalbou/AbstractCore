import base64

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider


def test_lmstudio_fallback_injects_user_message_when_missing(monkeypatch) -> None:
    """LM Studio templates can hard-fail when no user message exists (jinja: no user query).

    Ensure OpenAICompatibleProvider (LMStudioProvider) always sends at least one user message.
    """

    # Avoid any dependency on a running LMStudio server during provider init.
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)

    provider = LMStudioProvider(model="qwen/qwen3.5-9b", base_url="http://localhost:1234/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider._generate_internal(prompt="", system_prompt="You are a helpful assistant.", stream=False)

    messages = captured["payload"]["messages"]
    assert any(isinstance(m, dict) and m.get("role") == "user" for m in messages)


def test_lmstudio_file_path_media_dict_is_encoded_as_image_url(monkeypatch, tmp_path) -> None:
    """Artifact-backed browser uploads arrive as file_path dicts after Gateway resolution."""

    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="qwen/qwen3.5-9b", base_url="http://localhost:1234/v1")

    png_path = tmp_path / "content.png"
    png_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate(
        prompt="Describe this picture.",
        media=[
            {
                "file_path": str(png_path),
                "content_type": "image/png",
                "artifact_id": "art_test",
                "filename": "content.png",
            }
        ],
    )

    content = captured["payload"]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe this picture."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_compatible_file_path_media_dict_is_encoded_as_image_url(monkeypatch, tmp_path) -> None:
    """Gateway endpoint profiles route through the generic OpenAI-compatible provider."""

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)
    provider = OpenAICompatibleProvider(model="qwen/qwen3.5-9b", base_url="https://example.test/v1")

    png_path = tmp_path / "content.png"
    png_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate(
        prompt="Describe this picture.",
        media=[
            {
                "file_path": str(png_path),
                "content_type": "image/png",
                "artifact_id": "art_test",
                "filename": "content.png",
            }
        ],
    )

    content = captured["payload"]["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe this picture."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
