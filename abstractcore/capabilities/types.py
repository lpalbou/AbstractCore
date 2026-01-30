from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core.types import GenerateResponse


ArtifactRef = Dict[str, Any]  # expects {"$artifact": "...", ...}
BytesOrArtifactRef = Union[bytes, ArtifactRef]


def is_artifact_ref(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("$artifact"), str) and bool(value.get("$artifact"))


@runtime_checkable
class ArtifactStoreLike(Protocol):
    """Duck-typed artifact store interface (framework-mode durability).

    This intentionally mirrors AbstractRuntime's `ArtifactStore` shape, but is
    defined here to avoid introducing a hard dependency on `abstractruntime`.
    """

    def store(
        self,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        artifact_id: Optional[str] = None,
    ) -> Any: ...

    def load(self, artifact_id: str) -> Any: ...


class VoiceCapability(Protocol):
    backend_id: str

    def tts(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        format: str = "wav",
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BytesOrArtifactRef: ...

    def stt(
        self,
        audio: Union[bytes, ArtifactRef, str],
        *,
        language: Optional[str] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str: ...


class AudioCapability(Protocol):
    backend_id: str

    def transcribe(
        self,
        audio: Union[bytes, ArtifactRef, str],
        *,
        language: Optional[str] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str: ...


class VisionCapability(Protocol):
    backend_id: str

    def t2i(
        self,
        prompt: str,
        *,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...

    def i2i(
        self,
        prompt: str,
        image: Union[bytes, ArtifactRef, str],
        *,
        mask: Optional[Union[bytes, ArtifactRef, str]] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...

    def t2v(
        self,
        prompt: str,
        *,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...

    def i2v(
        self,
        image: Union[bytes, ArtifactRef, str],
        *,
        prompt: Optional[str] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...


@dataclass(frozen=True)
class GenerateWithOutputsResult:
    response: "GenerateResponse"
    outputs: Dict[str, Any]

