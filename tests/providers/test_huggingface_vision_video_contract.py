from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest


@pytest.mark.basic
def test_hf_vision_video_merges_last_user_text_when_prompt_empty(monkeypatch) -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")

    from abstractcore.core.types import GenerateResponse
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    class FakeInputs(dict):
        def to(self, device: Any):
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(eos_token_id=2)
            self.last_template_messages: List[Dict[str, Any]] = []

        def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
            self.last_template_messages = messages
            return "PROMPT"

        def __call__(self, **kwargs) -> FakeInputs:
            # Minimal transformer-like inputs
            input_ids = torch.tensor([[1, 2, 3, 4]])
            return FakeInputs({"input_ids": input_ids})

        def decode(self, tokens, skip_special_tokens: bool = True) -> str:
            return "ok"

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"
            self.last_generate_kwargs: Dict[str, Any] = {}

        def generate(self, **kwargs):
            self.last_generate_kwargs = dict(kwargs)
            # Return [input_ids + 2 output tokens]
            input_ids = kwargs.get("input_ids")
            in_len = int(getattr(input_ids, "shape", [1, 4])[1])
            return torch.tensor([list(range(in_len + 2))])

        def to(self, device: str):
            self.device = device
            return self

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = "llava-next-video-7b-hf"
    provider.temperature = 0.7
    provider.max_output_tokens = 1024
    provider.seed = None
    provider.logger = SimpleNamespace(warning=lambda *a, **k: None)
    provider.processor = FakeProcessor()
    provider.model_instance = FakeModel()

    media = [
        MediaContent(
            media_type=MediaType.VIDEO,
            content="/tmp/video.mp4",
            content_format=ContentFormat.FILE_PATH,
            mime_type="video/mp4",
            file_path="/tmp/video.mp4",
            metadata={"file_name": "video.mp4"},
        )
    ]

    messages = [
        {"role": "assistant", "content": "previous"},
        {"role": "user", "content": "Describe the attached video."},
    ]

    resp = provider._generate_vision_model(
        prompt="",
        messages=messages,
        media=media,
        stream=False,
        video_max_frames=2,
    )
    assert isinstance(resp, GenerateResponse)

    templated = provider.processor.last_template_messages
    assert templated, "Expected apply_chat_template to be called"
    last = templated[-1]
    assert last.get("role") == "user"
    content = last.get("content")
    assert isinstance(content, list)
    assert any(item.get("type") == "text" and item.get("text") == "Describe the attached video." for item in content)
    assert any(item.get("type") == "video" for item in content)


@pytest.mark.basic
def test_hf_vision_video_followup_collapses_prior_turns_into_user_message() -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")

    from abstractcore.core.types import GenerateResponse
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    class FakeInputs(dict):
        def to(self, device: Any):
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(eos_token_id=2)
            self.last_template_messages: List[Dict[str, Any]] = []

        def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
            self.last_template_messages = messages
            return "PROMPT"

        def __call__(self, **kwargs) -> FakeInputs:
            input_ids = torch.tensor([[1, 2, 3, 4]])
            return FakeInputs({"input_ids": input_ids})

        def decode(self, tokens, skip_special_tokens: bool = True) -> str:
            return "ok"

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def generate(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            in_len = int(getattr(input_ids, "shape", [1, 4])[1])
            return torch.tensor([list(range(in_len + 2))])

        def to(self, device: str):
            self.device = device
            return self

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = "llava-next-video-7b-hf"
    provider.temperature = 0.7
    provider.max_output_tokens = 1024
    provider.seed = None
    provider.logger = SimpleNamespace(warning=lambda *a, **k: None)
    provider.processor = FakeProcessor()
    provider.model_instance = FakeModel()

    media = [
        MediaContent(
            media_type=MediaType.VIDEO,
            content="/tmp/video.mp4",
            content_format=ContentFormat.FILE_PATH,
            mime_type="video/mp4",
            file_path="/tmp/video.mp4",
            metadata={"file_name": "video.mp4"},
        )
    ]

    # Simulate a follow-up: prior Q/A about video A, then "and this one?" with video B.
    # The native-video path must not represent prior turns as separate messages (we lost `<video>`
    # placeholders in history); instead it should collapse them into text inside the current user turn.
    messages = [
        {"role": "user", "content": "what is that video about ?"},
        {"role": "assistant", "content": "The video shows a squirrel."},
        {"role": "user", "content": "and this one ?"},
    ]

    resp = provider._generate_vision_model(
        prompt="",
        messages=messages,
        media=media,
        stream=False,
        video_max_frames=1,
    )
    assert isinstance(resp, GenerateResponse)

    templated = provider.processor.last_template_messages
    assert templated, "Expected apply_chat_template to be called"
    assert all(m.get("role") != "assistant" for m in templated), "Expected collapsed history (no assistant turns)"

    last = templated[-1]
    assert last.get("role") == "user"
    content = last.get("content")
    assert isinstance(content, list)
    history_texts = [str(item.get("text", "")) for item in content if item.get("type") == "text"]
    joined = "\n".join(history_texts)
    assert "Prior chat context (text-only)" in joined
    assert "USER: what is that video about ?" in joined
    assert "ASSISTANT: The video shows a squirrel." in joined
    assert any(item.get("type") == "video" for item in content)


@pytest.mark.basic
def test_hf_vision_video_includes_text_media_markers_in_template() -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")

    from abstractcore.core.types import GenerateResponse
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    class FakeInputs(dict):
        def to(self, device: Any):
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(eos_token_id=2)
            self.last_template_messages: List[Dict[str, Any]] = []

        def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
            self.last_template_messages = messages
            return "PROMPT"

        def __call__(self, **kwargs) -> FakeInputs:
            input_ids = torch.tensor([[1, 2, 3, 4]])
            return FakeInputs({"input_ids": input_ids})

        def decode(self, tokens, skip_special_tokens: bool = True) -> str:
            return "ok"

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def generate(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            in_len = int(getattr(input_ids, "shape", [1, 4])[1])
            return torch.tensor([list(range(in_len + 2))])

        def to(self, device: str):
            self.device = device
            return self

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = "llava-next-video-7b-hf"
    provider.temperature = 0.7
    provider.max_output_tokens = 1024
    provider.seed = None
    provider.logger = SimpleNamespace(warning=lambda *a, **k: None)
    provider.processor = FakeProcessor()
    provider.model_instance = FakeModel()

    marker = MediaContent(
        media_type=MediaType.TEXT,
        content="Video 1 (sample.mp4) — answer about this video only.",
        content_format=ContentFormat.TEXT,
        mime_type="text/plain",
        file_path=None,
        metadata={},
    )
    video = MediaContent(
        media_type=MediaType.VIDEO,
        content="/tmp/video.mp4",
        content_format=ContentFormat.FILE_PATH,
        mime_type="video/mp4",
        file_path="/tmp/video.mp4",
        metadata={"file_name": "video.mp4"},
    )

    resp = provider._generate_vision_model(
        prompt="Describe.",
        messages=None,
        media=[marker, video],
        stream=False,
        video_max_frames=1,
    )
    assert isinstance(resp, GenerateResponse)

    templated = provider.processor.last_template_messages
    assert templated, "Expected apply_chat_template to be called"
    last = templated[-1]
    assert last.get("role") == "user"
    content = last.get("content")
    assert isinstance(content, list)
    assert any(item.get("type") == "text" and "answer about this video only" in str(item.get("text", "")) for item in content)
    assert any(item.get("type") == "video" for item in content)


@pytest.mark.basic
def test_hf_vision_video_stream_true_returns_iterator() -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")

    from abstractcore.core.types import GenerateResponse
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    class FakeInputs(dict):
        def to(self, device: Any):
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(eos_token_id=2)

        def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
            return "PROMPT"

        def __call__(self, **kwargs) -> FakeInputs:
            input_ids = torch.tensor([[1, 2, 3]])
            return FakeInputs({"input_ids": input_ids})

        def decode(self, tokens, skip_special_tokens: bool = True) -> str:
            return "ok"

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def generate(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            in_len = int(getattr(input_ids, "shape", [1, 3])[1])
            return torch.tensor([list(range(in_len + 1))])

        def to(self, device: str):
            self.device = device
            return self

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = "llava-next-video-7b-hf"
    provider.temperature = 0.7
    provider.max_output_tokens = 1024
    provider.seed = None
    provider.logger = SimpleNamespace(warning=lambda *a, **k: None)
    provider.processor = FakeProcessor()
    provider.model_instance = FakeModel()

    media = [
        MediaContent(
            media_type=MediaType.VIDEO,
            content="/tmp/video.mp4",
            content_format=ContentFormat.FILE_PATH,
            mime_type="video/mp4",
            file_path="/tmp/video.mp4",
        )
    ]

    resp_stream = provider._generate_vision_model(
        prompt="Describe the video.",
        messages=None,
        media=media,
        stream=True,
        max_output_tokens=7,
        video_max_frames=1,
    )
    assert not isinstance(resp_stream, GenerateResponse)
    chunks = list(resp_stream)
    assert len(chunks) == 1
    assert isinstance(chunks[0], GenerateResponse)
    assert chunks[0].content == "ok"


@pytest.mark.basic
def test_hf_vision_video_respects_max_output_tokens_param() -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")

    from abstractcore.media.types import ContentFormat, MediaContent, MediaType
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    class FakeInputs(dict):
        def to(self, device: Any):
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(eos_token_id=2)

        def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
            return "PROMPT"

        def __call__(self, **kwargs) -> FakeInputs:
            input_ids = torch.tensor([[1, 2]])
            return FakeInputs({"input_ids": input_ids})

        def decode(self, tokens, skip_special_tokens: bool = True) -> str:
            return "ok"

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"
            self.last_generate_kwargs: Dict[str, Any] = {}

        def generate(self, **kwargs):
            self.last_generate_kwargs = dict(kwargs)
            input_ids = kwargs.get("input_ids")
            in_len = int(getattr(input_ids, "shape", [1, 2])[1])
            return torch.tensor([list(range(in_len + 1))])

        def to(self, device: str):
            self.device = device
            return self

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = "llava-next-video-7b-hf"
    provider.temperature = 0.7
    provider.max_output_tokens = 1024
    provider.seed = None
    provider.logger = SimpleNamespace(warning=lambda *a, **k: None)
    provider.processor = FakeProcessor()
    provider.model_instance = FakeModel()

    media = [
        MediaContent(
            media_type=MediaType.VIDEO,
            content="/tmp/video.mp4",
            content_format=ContentFormat.FILE_PATH,
            mime_type="video/mp4",
            file_path="/tmp/video.mp4",
        )
    ]

    _ = provider._generate_vision_model(
        prompt="Describe.",
        messages=None,
        media=media,
        stream=False,
        max_output_tokens=7,
        video_max_frames=1,
    )

    assert provider.model_instance.last_generate_kwargs.get("max_new_tokens") == 7


@pytest.mark.basic
def test_hf_vision_video_prefers_ffmpeg_sampled_frames_when_available(monkeypatch, tmp_path) -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")

    try:
        from PIL import Image as PILImage
    except Exception:
        pytest.skip("PIL not installed")

    from abstractcore.core.types import GenerateResponse
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    frame_path = tmp_path / "frame_01.jpg"
    PILImage.new("RGB", (16, 16), color="red").save(frame_path)

    def fake_extract_video_frames(
        video_path,
        *,
        max_frames: int = 3,
        frame_format: str = "jpg",
        sampling_strategy: str = "uniform",
        max_side=None,
        output_dir=None,
    ):
        return [frame_path], [0.0]

    monkeypatch.setattr(
        "abstractcore.media.utils.video_frames.extract_video_frames",
        fake_extract_video_frames,
    )

    class FakeInputs(dict):
        def to(self, device: Any):
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(eos_token_id=2)
            self.last_call_kwargs: Dict[str, Any] = {}

        def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
            return "PROMPT"

        def __call__(self, **kwargs) -> FakeInputs:
            self.last_call_kwargs = dict(kwargs)
            input_ids = torch.tensor([[1, 2]])
            return FakeInputs({"input_ids": input_ids})

        def decode(self, tokens, skip_special_tokens: bool = True) -> str:
            return "ok"

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def generate(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            in_len = int(getattr(input_ids, "shape", [1, 2])[1])
            return torch.tensor([list(range(in_len + 1))])

        def to(self, device: str):
            self.device = device
            return self

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = "llava-next-video-7b-hf"
    provider.temperature = 0.7
    provider.max_output_tokens = 1024
    provider.seed = None
    provider.logger = SimpleNamespace(warning=lambda *a, **k: None)
    provider.processor = FakeProcessor()
    provider.model_instance = FakeModel()

    video_path = tmp_path / "sample.mov"
    video_path.write_bytes(b"not-a-real-video")

    media = [
        MediaContent(
            media_type=MediaType.VIDEO,
            content=str(video_path),
            content_format=ContentFormat.FILE_PATH,
            mime_type="video/quicktime",
            file_path=str(video_path),
        )
    ]

    resp = provider._generate_vision_model(
        prompt="Describe the video.",
        messages=None,
        media=media,
        stream=False,
        video_max_frames=1,
    )
    assert isinstance(resp, GenerateResponse)

    videos_arg = provider.processor.last_call_kwargs.get("videos")
    assert isinstance(videos_arg, list)
    assert videos_arg, "Expected at least one PIL frame"
    assert hasattr(videos_arg[0], "size"), "Expected PIL-like image objects passed to processor"
