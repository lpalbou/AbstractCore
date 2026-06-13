from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from abstractcore.server.capability_generation import ServerVisionFacade


class _Request:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


@dataclass
class _Asset:
    data: bytes = b"payload"
    mime_type: str = "application/octet-stream"


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def generate_image(self, request: Any) -> _Asset:
        self.calls.append(("generate_image", request))
        return _Asset(data=b"image")

    def generate_image_with_progress(
        self,
        request: Any,
        *,
        progress_callback: Optional[Any] = None,
    ) -> _Asset:
        self.calls.append(("generate_image_with_progress", request))
        if callable(progress_callback):
            progress_callback(2, 4)
        hook = getattr(request, "extra", {}).get("on_progress")
        if callable(hook):
            hook({"step": 2, "total_steps": 4})
        return _Asset(data=b"image")

    def generate_video(self, request: Any) -> _Asset:
        self.calls.append(("generate_video", request))
        return _Asset(data=b"video", mime_type="video/mp4")

    def generate_video_with_progress(
        self,
        request: Any,
        *,
        progress_callback: Optional[Any] = None,
    ) -> _Asset:
        self.calls.append(("generate_video_with_progress", request))
        if callable(progress_callback):
            progress_callback(1, 3)
        hook = getattr(request, "extra", {}).get("on_progress")
        if callable(hook):
            hook({"step": 1, "total_steps": 3})
        return _Asset(data=b"video", mime_type="video/mp4")

    def image_to_video(self, request: Any) -> _Asset:
        self.calls.append(("image_to_video", request))
        return _Asset(data=b"i2v", mime_type="video/mp4")

    def upscale_image(self, request: Any) -> _Asset:
        self.calls.append(("upscale_image", request))
        return _Asset(data=b"upscale")


def _facade(*, video: bool = True, image_to_video: bool = True, upscale: bool = True) -> ServerVisionFacade:
    return ServerVisionFacade(
        backend=_Backend(),
        call_lock=None,
        image_generation_request_cls=_Request,
        image_edit_request_cls=_Request,
        video_generation_request_cls=_Request if video else None,
        image_to_video_request_cls=_Request if image_to_video else None,
        image_upscale_request_cls=_Request if upscale else None,
        backend_id="test-backend",
    )


def test_server_vision_facade_support_flags_reflect_backend_request_contract() -> None:
    facade = _facade(video=False, image_to_video=False, upscale=False)

    assert facade.supports_video_generation is False
    assert facade.supports_image_to_video is False
    assert facade.supports_image_upscale is False


def test_server_vision_facade_plan_batch_seeds_preserves_explicit_and_incremental_rules() -> None:
    assert ServerVisionFacade.plan_batch_seeds(count=2, seeds=[11, 12]) == [11, 12]
    assert ServerVisionFacade.plan_batch_seeds(count=3, seed=40) == [40, 41, 42]
    assert ServerVisionFacade.plan_batch_seeds(count=1, seed=None) == [None]


def test_server_vision_facade_prefers_progress_override_when_hooks_are_present() -> None:
    events: list[dict[str, Any]] = []
    ticks: list[tuple[int, Optional[int]]] = []
    facade = _facade()

    asset = facade.generate_video_asset(
        "orbit camera",
        steps=3,
        on_progress=lambda event: events.append(dict(event)),
        progress_callback=lambda step, total=None: ticks.append((step, total)),
    )

    assert asset.data == b"video"
    assert ticks == [(1, 3)]
    assert events == [{"step": 1, "total_steps": 3}]
    assert facade._backend.calls[0][0] == "generate_video_with_progress"
