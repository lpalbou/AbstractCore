"""Server-local provider used to route capability outputs through generate().

The HTTP server still owns HTTP validation and OpenAI-compatible response
shapes. This module gives those routes a small provider host so image, voice,
and future media behavior can reuse BaseProvider's unified output dispatcher.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from ..core.types import GenerateResponse
from ..providers.base import BaseProvider


class ServerCapabilityProvider(BaseProvider):
    """Minimal provider that delegates non-text outputs to capability facades."""

    def __init__(
        self,
        *,
        model: str = "abstractcore-server-capabilities",
        vision_facade: Optional[Any] = None,
        voice_facade: Optional[Any] = None,
        audio_facade: Optional[Any] = None,
        **config: Any,
    ) -> None:
        super().__init__(model=model, **config)
        self.provider = "abstractcore-server"
        self._server_vision_facade = vision_facade
        self._server_voice_facade = voice_facade
        self._server_audio_facade = audio_facade

    @property
    def vision(self) -> Any:
        if self._server_vision_facade is not None:
            return self._server_vision_facade
        return super().vision

    @property
    def voice(self) -> Any:
        if self._server_voice_facade is not None:
            return self._server_voice_facade
        return super().voice

    @property
    def audio(self) -> Any:
        if self._server_audio_facade is not None:
            return self._server_audio_facade
        return super().audio

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[list[dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        media: Optional[list[Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenerateResponse:
        _ = messages, system_prompt, tools, media, stream, kwargs
        return GenerateResponse(content=str(prompt or ""), model=self.model)

    def get_capabilities(self) -> list[str]:
        return []

    def unload_model(self, model_name: str) -> None:
        _ = model_name
        return None

    def list_available_models(self, **kwargs: Any) -> list[str]:
        _ = kwargs
        return [self.model]


class ServerVisionFacade:
    """Adapt existing server vision backends to AbstractCore VisionCapability."""

    def __init__(
        self,
        *,
        backend: Any,
        call_lock: Optional[threading.Lock],
        image_generation_request_cls: Any,
        image_edit_request_cls: Any,
        backend_id: str,
    ) -> None:
        self._backend = backend
        self._call_lock = call_lock or threading.Lock()
        self._image_generation_request_cls = image_generation_request_cls
        self._image_edit_request_cls = image_edit_request_cls
        self.backend_id = backend_id

    def t2i(self, prompt: str, **kwargs: Any) -> Any:
        req = self._image_generation_request_cls(
            prompt=str(prompt or ""),
            negative_prompt=kwargs.get("negative_prompt"),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            steps=kwargs.get("steps"),
            guidance_scale=kwargs.get("guidance_scale"),
            seed=kwargs.get("seed"),
            extra=kwargs.get("extra"),
        )
        with self._call_lock:
            asset = self._backend.generate_image(req)
        return bytes(getattr(asset, "data", b""))

    def i2i(self, prompt: str, image: Any, *, mask: Any = None, **kwargs: Any) -> Any:
        req = self._image_edit_request_cls(
            prompt=str(prompt or ""),
            image=image,
            mask=mask,
            negative_prompt=kwargs.get("negative_prompt"),
            seed=kwargs.get("seed"),
            steps=kwargs.get("steps"),
            guidance_scale=kwargs.get("guidance_scale"),
            extra=kwargs.get("extra"),
        )
        with self._call_lock:
            asset = self._backend.edit_image(req)
        return bytes(getattr(asset, "data", b""))


def create_capability_generation_core(
    *,
    model: str = "abstractcore-server-capabilities",
    vision_facade: Optional[Any] = None,
    voice_facade: Optional[Any] = None,
    audio_facade: Optional[Any] = None,
    **config: Any,
) -> ServerCapabilityProvider:
    return ServerCapabilityProvider(
        model=model,
        vision_facade=vision_facade,
        voice_facade=voice_facade,
        audio_facade=audio_facade,
        **config,
    )


__all__ = [
    "ServerCapabilityProvider",
    "ServerVisionFacade",
    "create_capability_generation_core",
]
