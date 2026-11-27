"""Server-local provider used to route capability outputs through generate().

The HTTP server still owns HTTP validation and OpenAI-compatible response
shapes. This module gives those routes a small provider host so image, voice,
and future media behavior can reuse BaseProvider's unified output dispatcher.
"""

from __future__ import annotations

import random
import threading
from typing import Any, Callable, Optional

from abstractvision.backends.base_backend import REQUEST_OPTION_FIELDS
from abstractvision.errors import CapabilityNotSupportedError

from ..core.types import GenerateResponse
from ..providers.base import BaseProvider

# Request fields a caller sets per call besides the generation parameters, as AbstractVision's
# request types declare them: the cancel event, the memory mode and load-time quantization, and on
# video requests the soundtrack options too (REQUEST_OPTION_FIELDS).
_IMAGE_REQUEST_OPTIONS = ("cancel_event", "memory_mode", "quantize")
_VIDEO_REQUEST_OPTIONS = ("cancel_event", *REQUEST_OPTION_FIELDS)


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

    def _seed_host_voice_credentials(self) -> None:
        """Never self-seed the saved OpenAI key: the server hands credentials
        over per request, only to requests allowed to spend them
        (`audio_endpoints._voice_core_for_request`)."""
        return None

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
        video_generation_request_cls: Optional[Any] = None,
        image_to_video_request_cls: Optional[Any] = None,
        image_upscale_request_cls: Optional[Any] = None,
        backend_id: str,
        takes_request_options: bool = True,
    ) -> None:
        self._backend = backend
        self._call_lock = call_lock or threading.Lock()
        self._image_generation_request_cls = image_generation_request_cls
        self._image_edit_request_cls = image_edit_request_cls
        self._video_generation_request_cls = video_generation_request_cls
        self._image_to_video_request_cls = image_to_video_request_cls
        self._image_upscale_request_cls = image_upscale_request_cls
        self.backend_id = backend_id
        # False for a backend whose request types carry no memory mode, quantization, soundtrack
        # or cancel event (the server's OpenAI-compatible proxy): a caller that sets one is refused.
        self._takes_request_options = takes_request_options

    def _request_options(self, kwargs: dict[str, Any], names: tuple) -> dict[str, Any]:
        """The per-call request options set in `kwargs`, as request fields.

        `quantize="none"` asks for no load-time quantization (over a route default) and is sent as
        unset. On a backend that takes none of them, a set option is refused; the cancel event is
        then left out (the run is cancelled between items).
        """
        options: dict[str, Any] = {}
        for name in names:
            value = kwargs.get(name)
            if name == "quantize" and value == "none":
                continue
            if value is None:
                continue
            if not self._takes_request_options:
                if name == "cancel_event":
                    continue
                raise CapabilityNotSupportedError(
                    f"The selected provider does not take `{name}`; send backend-specific values in `extra`.",
                    field=name,
                    supported=(),
                )
            options[name] = value
        return options

    @staticmethod
    def plan_batch_seeds(
        *,
        count: int,
        seed: Optional[int] = None,
        seeds: Optional[Any] = None,
    ) -> list[Optional[int]]:
        if seeds is not None:
            if not isinstance(seeds, (list, tuple)):
                raise ValueError("Batch generation seeds must be a list of integers.")
            planned = [int(value) for value in seeds]
            if not planned:
                raise ValueError("Batch generation seeds cannot be empty.")
            if int(count) != len(planned):
                raise ValueError(
                    f"Batch generation count ({int(count)}) must match the number of explicit seeds ({len(planned)})."
                )
            return planned
        if int(count) <= 0:
            raise ValueError("Batch generation count must be >= 1.")
        if int(count) == 1:
            return [int(seed)] if seed is not None else [None]
        if seed is not None:
            base_seed = int(seed)
            return [base_seed + index for index in range(int(count))]
        rng = random.SystemRandom()
        return [int(rng.randrange(0, 1_000_000_000)) for _ in range(int(count))]

    @staticmethod
    def _backend_overrides_progress_method(backend: Any, method_name: str) -> bool:
        fn = getattr(backend, method_name, None)
        if not callable(fn):
            return False
        for cls in type(backend).__mro__:
            if method_name not in getattr(cls, "__dict__", {}):
                continue
            module = str(getattr(cls, "__module__", "") or "")
            qualname = str(getattr(cls, "__qualname__", "") or "")
            if module.endswith(".base_backend") and qualname == "VisionBackend":
                return False
            return True
        return False

    @staticmethod
    def _request_has_progress_hooks(req: Any) -> bool:
        extra = getattr(req, "extra", None)
        if not isinstance(extra, dict):
            return False
        for key in ("on_progress", "progress_event_callback", "progress_callback"):
            if callable(extra.get(key)):
                return True
        return False

    def _call_backend(
        self,
        *,
        standard_method: str,
        progress_method: str,
        req: Any,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> Any:
        use_progress_method = self._backend_overrides_progress_method(self._backend, progress_method) and (
            callable(progress_callback) or self._request_has_progress_hooks(req)
        )
        with self._call_lock:
            if use_progress_method:
                fn = getattr(self._backend, progress_method)
                return fn(req, progress_callback=progress_callback)
            fn = getattr(self._backend, standard_method)
            return fn(req)

    def list_provider_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        _ = provider
        method = getattr(self._backend, "list_provider_adapters", None)
        if not callable(method):
            return []
        out = method(model=model, task=task)
        return list(out or [])

    @property
    def supports_video_generation(self) -> bool:
        return self._video_generation_request_cls is not None

    @property
    def supports_image_to_video(self) -> bool:
        return self._image_to_video_request_cls is not None

    @property
    def supports_image_upscale(self) -> bool:
        return self._image_upscale_request_cls is not None

    def build_image_generation_request(self, prompt: str, **kwargs: Any) -> Any:
        return self._image_generation_request_cls(
            prompt=str(prompt or ""),
            negative_prompt=kwargs.get("negative_prompt"),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            steps=kwargs.get("steps"),
            guidance_scale=kwargs.get("guidance_scale"),
            seed=kwargs.get("seed"),
            lora_adapters=kwargs.get("lora_adapters") or (),
            extra=self._extra_with_image_params(kwargs),
            **self._request_options(kwargs, _IMAGE_REQUEST_OPTIONS),
        )

    def generate_image_asset(
        self,
        prompt: str,
        *,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        **kwargs: Any,
    ) -> Any:
        req = self.build_image_generation_request(prompt, **kwargs)
        return self._call_backend(
            standard_method="generate_image",
            progress_method="generate_image_with_progress",
            req=req,
            progress_callback=progress_callback,
        )

    def t2i(self, prompt: str, **kwargs: Any) -> Any:
        asset = self.generate_image_asset(prompt, **kwargs)
        return bytes(getattr(asset, "data", b""))

    def build_image_edit_request(self, prompt: str, image: Any, *, mask: Any = None, **kwargs: Any) -> Any:
        return self._image_edit_request_cls(
            prompt=str(prompt or ""),
            image=image,
            mask=mask,
            negative_prompt=kwargs.get("negative_prompt"),
            seed=kwargs.get("seed"),
            steps=kwargs.get("steps"),
            guidance_scale=kwargs.get("guidance_scale"),
            lora_adapters=kwargs.get("lora_adapters") or (),
            extra=self._extra_with_image_params(kwargs),
            **self._request_options(kwargs, _IMAGE_REQUEST_OPTIONS),
        )

    def edit_image_asset(
        self,
        prompt: str,
        image: Any,
        *,
        mask: Any = None,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        **kwargs: Any,
    ) -> Any:
        req = self.build_image_edit_request(prompt, image, mask=mask, **kwargs)
        return self._call_backend(
            standard_method="edit_image",
            progress_method="edit_image_with_progress",
            req=req,
            progress_callback=progress_callback,
        )

    def i2i(self, prompt: str, image: Any, *, mask: Any = None, **kwargs: Any) -> Any:
        asset = self.edit_image_asset(prompt, image, mask=mask, **kwargs)
        return bytes(getattr(asset, "data", b""))

    def build_image_upscale_request(self, image: Any, **kwargs: Any) -> Any:
        if self._image_upscale_request_cls is None:
            raise AttributeError("The selected AbstractVision backend does not expose ImageUpscaleRequest.")
        return self._image_upscale_request_cls(
            image=image,
            resolution=kwargs.get("resolution"),
            scale=kwargs.get("scale"),
            seed=kwargs.get("seed"),
            softness=kwargs.get("softness"),
            vae_tiling=kwargs.get("vae_tiling"),
            extra=self._extra_with_image_params(kwargs),
            # As sent: on upscale, "none" (like "0", "false", ...) is the backend's own spelling of
            # an unquantized load, not "unset".
            quantize=kwargs.get("quantize"),
            **self._request_options(kwargs, ("memory_mode", "cancel_event")),
        )

    def upscale_image_asset(
        self,
        image: Any,
        *,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        **kwargs: Any,
    ) -> Any:
        req = self.build_image_upscale_request(image, **kwargs)
        return self._call_backend(
            standard_method="upscale_image",
            progress_method="upscale_image_with_progress",
            req=req,
            progress_callback=progress_callback,
        )

    def upscale_image(self, image: Any, **kwargs: Any) -> Any:
        asset = self.upscale_image_asset(image, **kwargs)
        return bytes(getattr(asset, "data", b""))

    @staticmethod
    def _extra_with_params(kwargs: dict[str, Any], *, request_keys: set[str]) -> dict[str, Any]:
        """`extra` for a request: other keyword parameters, then the caller's `extra` over them."""
        merged: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in request_keys or key in {"on_progress", "progress_event_callback", "progress_callback"}:
                continue
            if value is not None:
                merged[str(key)] = value
        extra = kwargs.get("extra")
        if isinstance(extra, dict):
            merged.update(extra)
        for key in ("on_progress", "progress_event_callback", "progress_callback"):
            callback = kwargs.get(key)
            if callback is not None:
                merged[key] = callback
        return merged

    @classmethod
    def _extra_with_image_params(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        return cls._extra_with_params(
            kwargs,
            request_keys={
                "prompt",
                "image",
                "mask",
                "negative_prompt",
                "width",
                "height",
                "resolution",
                "scale",
                "seed",
                "steps",
                "guidance_scale",
                "guidance_2",
                "lora_adapters",
                "softness",
                *_IMAGE_REQUEST_OPTIONS,
                "vae_tiling",
                "extra",
                "provider",
                "model",
                "artifact_store",
                "run_id",
                "tags",
            },
        )

    @classmethod
    def _extra_with_video_params(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        return cls._extra_with_params(
            kwargs,
            request_keys={
                "prompt",
                "image",
                "negative_prompt",
                "width",
                "height",
                "fps",
                "num_frames",
                "seed",
                "steps",
                "guidance_scale",
                "guidance_2",
                "flow_shift",
                "lora_adapters",
                *_VIDEO_REQUEST_OPTIONS,
                "extra",
                "provider",
                "model",
                "artifact_store",
                "run_id",
                "tags",
            },
        )

    def build_video_generation_request(self, prompt: str, **kwargs: Any) -> Any:
        if self._video_generation_request_cls is None:
            raise AttributeError("The selected AbstractVision backend does not expose VideoGenerationRequest.")
        return self._video_generation_request_cls(
            prompt=str(prompt or ""),
            negative_prompt=kwargs.get("negative_prompt"),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            fps=kwargs.get("fps"),
            num_frames=kwargs.get("num_frames"),
            seed=kwargs.get("seed"),
            steps=kwargs.get("steps"),
            guidance_scale=kwargs.get("guidance_scale"),
            guidance_2=kwargs.get("guidance_2"),
            flow_shift=kwargs.get("flow_shift"),
            lora_adapters=kwargs.get("lora_adapters") or (),
            extra=self._extra_with_video_params(kwargs),
            **self._request_options(kwargs, _VIDEO_REQUEST_OPTIONS),
        )

    def generate_video_asset(
        self,
        prompt: str,
        *,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        **kwargs: Any,
    ) -> Any:
        req = self.build_video_generation_request(prompt, **kwargs)
        return self._call_backend(
            standard_method="generate_video",
            progress_method="generate_video_with_progress",
            req=req,
            progress_callback=progress_callback,
        )

    def t2v(self, prompt: str, **kwargs: Any) -> Any:
        asset = self.generate_video_asset(prompt, **kwargs)
        return bytes(getattr(asset, "data", b""))

    def build_image_to_video_request(self, image: Any, **kwargs: Any) -> Any:
        if self._image_to_video_request_cls is None:
            raise AttributeError("The selected AbstractVision backend does not expose ImageToVideoRequest.")
        return self._image_to_video_request_cls(
            image=image,
            prompt=kwargs.get("prompt"),
            negative_prompt=kwargs.get("negative_prompt"),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            fps=kwargs.get("fps"),
            num_frames=kwargs.get("num_frames"),
            seed=kwargs.get("seed"),
            steps=kwargs.get("steps"),
            guidance_scale=kwargs.get("guidance_scale"),
            guidance_2=kwargs.get("guidance_2"),
            flow_shift=kwargs.get("flow_shift"),
            lora_adapters=kwargs.get("lora_adapters") or (),
            extra=self._extra_with_video_params(kwargs),
            **self._request_options(kwargs, _VIDEO_REQUEST_OPTIONS),
        )

    def image_to_video_asset(
        self,
        image: Any,
        *,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        **kwargs: Any,
    ) -> Any:
        req = self.build_image_to_video_request(image, **kwargs)
        return self._call_backend(
            standard_method="image_to_video",
            progress_method="image_to_video_with_progress",
            req=req,
            progress_callback=progress_callback,
        )

    def i2v(self, image: Any, **kwargs: Any) -> Any:
        asset = self.image_to_video_asset(image, **kwargs)
        return bytes(getattr(asset, "data", b""))

    def t2i_batch(self, prompt: str, **kwargs: Any) -> Any:
        planned_seeds = self.plan_batch_seeds(
            count=int(kwargs.pop("count", 1) or 1),
            seed=kwargs.get("seed"),
            seeds=kwargs.pop("seeds", None),
        )
        out = []
        for planned_seed in planned_seeds:
            call_kwargs = dict(kwargs)
            call_kwargs["seed"] = planned_seed
            out.append(self.t2i(prompt, **call_kwargs))
        return out

    def i2i_batch(self, prompt: str, image: Any, *, mask: Any = None, **kwargs: Any) -> Any:
        planned_seeds = self.plan_batch_seeds(
            count=int(kwargs.pop("count", 1) or 1),
            seed=kwargs.get("seed"),
            seeds=kwargs.pop("seeds", None),
        )
        out = []
        for planned_seed in planned_seeds:
            call_kwargs = dict(kwargs)
            call_kwargs["seed"] = planned_seed
            out.append(self.i2i(prompt, image, mask=mask, **call_kwargs))
        return out

    def t2v_batch(self, prompt: str, **kwargs: Any) -> Any:
        planned_seeds = self.plan_batch_seeds(
            count=int(kwargs.pop("count", 1) or 1),
            seed=kwargs.get("seed"),
            seeds=kwargs.pop("seeds", None),
        )
        out = []
        for planned_seed in planned_seeds:
            call_kwargs = dict(kwargs)
            call_kwargs["seed"] = planned_seed
            out.append(self.t2v(prompt, **call_kwargs))
        return out

    def i2v_batch(self, image: Any, **kwargs: Any) -> Any:
        planned_seeds = self.plan_batch_seeds(
            count=int(kwargs.pop("count", 1) or 1),
            seed=kwargs.get("seed"),
            seeds=kwargs.pop("seeds", None),
        )
        out = []
        for planned_seed in planned_seeds:
            call_kwargs = dict(kwargs)
            call_kwargs["seed"] = planned_seed
            out.append(self.i2v(image, **call_kwargs))
        return out


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
