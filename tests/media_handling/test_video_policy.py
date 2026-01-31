from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.basic
def test_native_only_rejects_video_when_model_not_video_capable(tmp_path: Path) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.exceptions import UnsupportedFeatureError
    from abstractcore.providers.base import BaseProvider

    class DummyProvider(BaseProvider):
        def __init__(self):
            super().__init__(model="qwen/qwen3-next-80b")
            # Gate native video support to HuggingFace only (v0).
            self.provider = "huggingface"

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop", metadata={})

        def get_capabilities(self):
            return []

        def unload_model(self, model_name: str) -> None:
            return None

        def list_available_models(self, **kwargs):
            return []

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")  # VideoProcessor only needs a file ref for v0.

    provider = DummyProvider()

    with pytest.raises(UnsupportedFeatureError):
        provider.generate("What is in this video?", media=[str(video_path)], stream=False, video_policy="native_only")


@pytest.mark.basic
def test_native_only_allows_video_when_model_capabilities_claim_support(tmp_path: Path) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.base import BaseProvider

    class DummyProvider(BaseProvider):
        def __init__(self):
            super().__init__(model="qwen/qwen3-next-80b")
            self.provider = "huggingface"
            # Simulate a video-capable model.
            self.model_capabilities["video_support"] = True
            self.last_media = None

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
            self.last_media = media
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop", metadata={})

        def get_capabilities(self):
            return []

        def unload_model(self, model_name: str) -> None:
            return None

        def list_available_models(self, **kwargs):
            return []

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    provider = DummyProvider()
    resp = provider.generate("What is in this video?", media=[str(video_path)], stream=False, video_policy="native_only")
    assert resp.content == "ok"
    assert isinstance(provider.last_media, list)
    assert provider.last_media, "Expected native video handling to preserve media"
    kinds = [getattr(item, "media_type", None).value for item in provider.last_media]
    assert "text" in kinds, "Expected a text marker describing native-video context"
    assert "video" in kinds, "Expected the video to be passed through for provider-native handling"
    marker = next(item for item in provider.last_media if getattr(getattr(item, "media_type", None), "value", None) == "text")
    marker_text = str(getattr(marker, "content", "") or "").lower()
    assert "answer the user's question about this video" in marker_text
    assert "do not mention" in marker_text


@pytest.mark.basic
def test_frames_caption_replaces_video_with_images(monkeypatch, tmp_path: Path) -> None:
    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.base import BaseProvider

    try:
        from PIL import Image as PILImage
    except ImportError:
        pytest.skip("PIL not installed")

    class DummyProvider(BaseProvider):
        def __init__(self):
            super().__init__(model="gpt-4o")  # vision-capable, but not native video input in v0
            self.provider = "openai"
            self.last_media = None

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
            self.last_media = media
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop", metadata={})

        def get_capabilities(self):
            return []

        def unload_model(self, model_name: str) -> None:
            return None

        def list_available_models(self, **kwargs):
            return []

    def fake_extract_video_frames(
        video_path: Path,
        *,
        max_frames: int = 3,
        frame_format: str = "jpg",
        sampling_strategy: str = "uniform",
        max_side=None,
        output_dir=None,
    ):
        out_dir = Path(output_dir) if output_dir is not None else tmp_path
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_path = out_dir / f"frame_01.{frame_format}"
        img = PILImage.new("RGB", (16, 16), color="red")
        img.save(frame_path)
        return [frame_path], [0.0]

    monkeypatch.setattr(
        "abstractcore.media.utils.video_frames.extract_video_frames",
        fake_extract_video_frames,
    )

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    provider = DummyProvider()
    resp = provider.generate("What is in this video?", media=[str(video_path)], stream=False, video_policy="frames_caption")
    assert resp.content == "ok"
    assert isinstance(provider.last_media, list)
    assert provider.last_media, "Expected frames-caption fallback to produce media"
    kinds = [getattr(item, "media_type", None).value for item in provider.last_media]
    assert "text" in kinds, "Expected a text marker describing frame provenance"
    assert "image" in kinds, "Expected extracted frames to be passed as images"
    marker = next(item for item in provider.last_media if getattr(getattr(item, "media_type", None), "value", None) == "text")
    marker_text = str(getattr(marker, "content", "") or "").lower()
    assert "do not mention frames" in marker_text
    assert "timestamps_s" not in marker_text


@pytest.mark.basic
def test_native_video_defaults_use_max_frames_native(monkeypatch, tmp_path: Path) -> None:
    from abstractcore.config.manager import AbstractCoreConfig
    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.base import BaseProvider

    class DummyCfgMgr:
        def __init__(self):
            self.config = AbstractCoreConfig.default()
            # Make the distinction obvious.
            self.config.video.max_frames = 3
            self.config.video.max_frames_native = 8

    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", lambda: DummyCfgMgr())

    class DummyProvider(BaseProvider):
        def __init__(self):
            super().__init__(model="llava-next-video-7b-hf")
            self.provider = "huggingface"
            self.model_capabilities["video_support"] = True
            self.last_kwargs = None

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
            self.last_kwargs = dict(kwargs)
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop", metadata={})

        def get_capabilities(self):
            return []

        def unload_model(self, model_name: str) -> None:
            return None

        def list_available_models(self, **kwargs):
            return []

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    provider = DummyProvider()
    _ = provider.generate("What is this video?", media=[str(video_path)], stream=False, video_policy="native_only")
    assert provider.last_kwargs is not None
    assert provider.last_kwargs.get("video_max_frames") == 8
