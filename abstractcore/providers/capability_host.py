"""A provider with NO text model: it only hosts the capability plugins (voice, audio, vision,
music) for media-only requests.

Why (AbstractFramework 0.7.0 end-to-end, light install on a 128 GB Mac): every voice request
went through a provider built for the host's DEFAULT TEXT model (`generate(..., output=voice)`
runs on the text provider's capability plugins), so a text default that could not be built
(MLX without MLX installed) made text-to-speech fail with "MLX dependencies not installed",
and a buildable one loaded tens of GB of text weights just to speak. A media-only request
(image, video, voice, music, transcription) never needs the text model: hosts run it on this
provider instead (AbstractRuntime's local clients do).

The capability plugins are resolved from the same config as any provider (`**config`: the
execution host's core config file and capability defaults), so a media request behaves exactly
as on a text provider. Asking this provider for TEXT raises a typed error that says so; it never
answers with an empty or echoed completion.
"""

from __future__ import annotations

from typing import Any, List

from ..exceptions import InvalidRequestError
from .base import BaseProvider

CAPABILITY_HOST_PROVIDER = "abstractcore-capabilities"
CAPABILITY_HOST_MODEL = "media-only"


class TextGenerationUnavailable(InvalidRequestError):
    """Text generation was asked of the media-only capability host (not retryable)."""


class CapabilityHostProvider(BaseProvider):
    """Runs media-only outputs through the capability plugins; refuses text generation."""

    def __init__(self, model: str = CAPABILITY_HOST_MODEL, **config: Any) -> None:
        super().__init__(model=model, **config)
        self.provider = CAPABILITY_HOST_PROVIDER

    def _generate_internal(self, prompt: str, *args: Any, **kwargs: Any):  # type: ignore[override]
        raise TextGenerationUnavailable(
            "This request needs a text model, and the media-only capability host has none "
            "(it serves image, video, voice, music and transcription outputs only). "
            "Send the request with a text provider/model, or set the text default "
            "(abstractcore config set-default output.text --provider <provider> --model <model>)."
        )

    def get_capabilities(self) -> List[str]:
        return []

    def unload_model(self, model_name: str) -> None:
        _ = model_name
        return None

    def list_available_models(self, **kwargs: Any) -> List[str]:
        _ = kwargs
        return []


def create_capability_host(**config: Any) -> CapabilityHostProvider:
    """The media-only capability host for `config` (the same kwargs `create_llm` takes)."""

    return CapabilityHostProvider(**config)


__all__ = [
    "CAPABILITY_HOST_MODEL",
    "CAPABILITY_HOST_PROVIDER",
    "CapabilityHostProvider",
    "TextGenerationUnavailable",
    "create_capability_host",
]
