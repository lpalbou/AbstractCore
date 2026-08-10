from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable, TYPE_CHECKING

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


@dataclass(frozen=True)
class CapabilityProviderInfo:
    provider_id: str
    display_name: str
    capability: str
    tasks: List[str] = field(default_factory=list)
    local: bool = False
    remote: bool = False
    status: str = "unknown"
    backend_id: Optional[str] = None
    installed: Optional[bool] = None
    configured: Optional[bool] = None
    reachable: Optional[bool] = None
    selected: Optional[bool] = None
    install_hint: Optional[str] = None
    config_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "capability": self.capability,
            "tasks": list(self.tasks or []),
            "local": bool(self.local),
            "remote": bool(self.remote),
            "status": self.status,
        }
        for key in (
            "backend_id",
            "installed",
            "configured",
            "reachable",
            "selected",
            "install_hint",
            "config_hint",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


@dataclass(frozen=True)
class CapabilityModelInfo:
    model_id: str
    provider_id: str
    capability: str
    tasks: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    local: bool = False
    remote: bool = False
    status: str = "unknown"
    backend_id: Optional[str] = None
    routed_model: Optional[str] = None
    formats: List[str] = field(default_factory=list)
    source: Optional[str] = None
    recommended: Optional[bool] = None
    license: Optional[str] = None
    commercial_allowed: Optional[bool] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "capability": self.capability,
            "tasks": list(self.tasks or []),
            "modalities": list(self.modalities or []),
            "local": bool(self.local),
            "remote": bool(self.remote),
            "status": self.status,
        }
        for key in (
            "backend_id",
            "routed_model",
            "source",
            "recommended",
            "license",
            "commercial_allowed",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.formats:
            out["formats"] = list(self.formats)
        if self.raw_metadata:
            out["raw_metadata"] = dict(self.raw_metadata)
        return out


@dataclass(frozen=True)
class CapabilityOperationInfo:
    operation_id: str
    capability: str
    task: str
    input_modalities: List[str] = field(default_factory=list)
    output_modalities: List[str] = field(default_factory=list)
    parameter_schema: Optional[Dict[str, Any]] = None
    required_parameters: List[str] = field(default_factory=list)
    artifact_output: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "operation_id": self.operation_id,
            "capability": self.capability,
            "task": self.task,
            "input_modalities": list(self.input_modalities or []),
            "output_modalities": list(self.output_modalities or []),
        }
        if self.parameter_schema is not None:
            out["parameter_schema"] = dict(self.parameter_schema)
        if self.required_parameters:
            out["required_parameters"] = list(self.required_parameters)
        if self.artifact_output is not None:
            out["artifact_output"] = bool(self.artifact_output)
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


CapabilityArtifactRef = ArtifactRef


@dataclass(frozen=True)
class CapabilityInvokeResult:
    content: Optional[Any] = None
    artifacts: List[CapabilityArtifactRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "content": self.content,
            "artifacts": [dict(a) for a in self.artifacts],
        }
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


@dataclass(frozen=True)
class CoreTextResult:
    content: str
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"content": self.content}
        if self.model is not None:
            out["model"] = self.model
        if self.usage is not None:
            out["usage"] = dict(self.usage)
        if self.finish_reason is not None:
            out["finish_reason"] = self.finish_reason
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out


@runtime_checkable
class CoreTextGenerationService(Protocol):
    def generate_text(
        self,
        prompt: str = "",
        *,
        messages: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Union[bool, str]] = None,
        purpose: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CoreTextResult: ...

    def generate_structured(
        self,
        prompt: str = "",
        *,
        response_model: Optional[Any] = None,
        json_schema: Optional[Mapping[str, Any]] = None,
        messages: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        purpose: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any: ...


@runtime_checkable
class CapabilityHostContext(Protocol):
    @property
    def text(self) -> CoreTextGenerationService: ...

    def service(self, name: str) -> Any: ...


class VoiceCapability(Protocol):
    backend_id: str

    def load_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]: ...

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]: ...

    def unload_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def list_profiles(self, *, kind: str = "tts") -> List[Dict[str, Any]]: ...

    def available_providers(self) -> Dict[str, Any]: ...

    def list_models(self, *, kind: str = "tts", provider: Optional[str] = None) -> List[str]: ...

    def list_tts_models(self, provider: Optional[str] = None) -> List[str]: ...

    def list_stt_models(self, provider: Optional[str] = None) -> List[str]: ...

    def list_cloning_models(self, provider: Optional[str] = None) -> List[str]: ...

    def voice_catalog(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        providers_only: bool = False,
    ) -> Dict[str, Any]: ...

    def tts(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        format: str = "wav",
        profile: Optional[str] = None,
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
        quality_preset: Optional[str] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...

    def tts_stream(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        format: str = "wav",
        profile: Optional[str] = None,
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
        quality_preset: Optional[str] = None,
        **kwargs: Any,
    ) -> Any: ...

    def stt(
        self,
        audio: Union[bytes, ArtifactRef, str],
        *,
        language: Optional[str] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    def clone(
        self,
        audio: Union[bytes, Dict[str, Any], str],
        *,
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        consent: Optional[str] = None,
        validate: Optional[bool] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any: ...


class AudioCapability(Protocol):
    backend_id: str

    def load_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]: ...

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]: ...

    def unload_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

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

    def load_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]: ...

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]: ...

    def unload_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def list_provider_models(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]: ...

    def list_provider_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

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

    def t2i_batch(
        self,
        prompt: str,
        *,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        count: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> List[BytesOrArtifactRef]: ...

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

    def i2i_batch(
        self,
        prompt: str,
        image: Union[bytes, ArtifactRef, str],
        *,
        mask: Optional[Union[bytes, ArtifactRef, str]] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        count: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> List[BytesOrArtifactRef]: ...

    def upscale_image(
        self,
        image: Union[bytes, ArtifactRef, str],
        *,
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

    def t2v_batch(
        self,
        prompt: str,
        *,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        count: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> List[BytesOrArtifactRef]: ...

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

    def i2v_batch(
        self,
        image: Union[bytes, ArtifactRef, str],
        *,
        prompt: Optional[str] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        count: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> List[BytesOrArtifactRef]: ...


class MusicCapability(Protocol):
    """Deterministic music generation capability (text-to-music).

    This is intentionally modeled like `VisionCapability` / `VoiceCapability`:
    - returns either raw bytes (library mode) or an ArtifactRef (framework mode)
    - backend selection/resolution happens via `abstractcore.capabilities.registry`
    """

    backend_id: str

    def available_providers(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]: ...

    def list_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    def list_provider_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    def list_operations(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]: ...

    # Every known provider with whether/why it is usable (abstractmusic >= 0.1.14).
    # The registry facade treats it as optional-by-duck-typing so older or
    # third-party backends keep working; absence raises a clean
    # CapabilityUnavailableError naming the minimum version.
    def provider_details(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]: ...

    def t2m(
        self,
        prompt: str,
        *,
        lyrics: Optional[str] = None,
        format: str = "wav",
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...

    def generate(
        self,
        prompt: str,
        *,
        task: Optional[str] = None,
        lyrics: Optional[str] = None,
        format: str = "wav",
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...


class Scene3dCapability(Protocol):
    """Deterministic 3D scene/object generation capability.

    The first release focuses on single-object mesh generation with a primary
    binary artifact such as GLB. Capability metadata may reference preview
    renders, secondary exports, and bundle paths when available.
    """

    backend_id: str

    def available_providers(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]: ...

    def list_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    def list_provider_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    def list_operations(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]: ...

    def load_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]: ...

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]: ...

    def unload_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def t23d(
        self,
        prompt: str,
        *,
        format: str = "glb",
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...

    def i23d(
        self,
        image: Union[bytes, ArtifactRef, str],
        *,
        prompt: Optional[str] = None,
        format: str = "glb",
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...

    def generate(
        self,
        prompt: str = "",
        *,
        task: Optional[str] = None,
        image: Optional[Union[bytes, ArtifactRef, str]] = None,
        format: str = "glb",
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BytesOrArtifactRef: ...


# --- camera (drafted by seat: camera; owner review: core — commons c3168) ---
class CameraCapability(Protocol):
    """Real-camera piloting capability (abstractcamera plugin).

    Minimal-required by ruling (commons c3168): backend_id plus the
    operations below; discovery methods (available_providers/list_models/
    list_operations) stay optional-by-duck-typing — the facade tolerates
    their absence. Operations return JSON-SAFE dicts (results may land in
    runtime ledgers); capture content rides base64 (`data_b64`) or
    `{"$artifact": ...}` refs, never raw bytes. `preview_frame` returns
    JPEG bytes as the RETURN VALUE (the documented exception, mirroring
    the vision-plugin convention for image payloads) or an artifact ref.

    Addressing: `camera_id` is a DISCOVERY id (list_cameras), accepted only
    by open(); `camera` is the DEVICE UID of a live camera (open/status),
    None meaning the active one. Failures raise with user-actionable text.
    """

    backend_id: str

    def list_cameras(self) -> List[Dict[str, Any]]: ...

    def open(self, camera_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]: ...

    def close(self, camera: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]: ...

    def close_all(self) -> Dict[str, Any]: ...

    def status(self, camera: Optional[str] = None) -> Dict[str, Any]: ...

    def capture_photo(
        self,
        camera: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        include_bytes: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def capture_video(
        self,
        duration_s: float,
        camera: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        include_bytes: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def stop_recording(
        self,
        camera: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def preview_frame(
        self,
        camera: Optional[str] = None,
        *,
        artifact_store: Optional[ArtifactStoreLike] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[bytes, ArtifactRef]: ...

    def start_detection(
        self,
        camera: Optional[str] = None,
        *,
        target: str = "motion",
        action: str = "photo",
        sensitivity: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def stop_detection(self, camera: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]: ...

    def detection_events(
        self,
        camera: Optional[str] = None,
        *,
        since_id: int = 0,
        kinds: Optional[List[str]] = None,
        limit: int = 100,
        include_thumbnails: bool = False,
    ) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class GenerateWithOutputsResult:
    response: "GenerateResponse"
    outputs: Dict[str, Any]
