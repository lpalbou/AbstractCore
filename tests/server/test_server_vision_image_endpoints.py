import os
import base64
import time
from dataclasses import dataclass
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from abstractcore.server.app import app

_PNG_BYTES = b"\x89PNG\r\n\x1a\nabstractcore-test-png"
_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42abstractcore-test-mp4"


@pytest.fixture(autouse=True)
def clean_vision_state(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("ABSTRACTCORE_VISION_", "ABSTRACTVISION_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    from abstractcore.server import vision_endpoints

    with vision_endpoints._BACKEND_CACHE_LOCK:
        vision_endpoints._BACKEND_CACHE.clear()
    with vision_endpoints._ACTIVE_LOCK:
        vision_endpoints._ACTIVE_MODEL_ID = None
        vision_endpoints._ACTIVE_BACKEND_KIND = None
        vision_endpoints._ACTIVE_BACKEND = None
        vision_endpoints._ACTIVE_CALL_LOCK = None
        vision_endpoints._ACTIVE_LOADED_AT_S = None
    with vision_endpoints._JOBS_LOCK:
        vision_endpoints._JOBS.clear()


@pytest.fixture()
def client():
    return TestClient(app)


def test_images_generations_without_model_uses_configured_openai_compatible_default(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("ABSTRACTVISION_MODEL_ID", "remote-image-model")
    monkeypatch.setenv("OPENAI_API_KEY", "vision-key")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    resp = client.post("/v1/images/generations", json={"prompt": "hello", "width": 64, "height": 64, "response_format": "b64_json"})

    assert resp.status_code == 200
    data = resp.json()
    assert base64.b64decode(data["data"][0]["b64_json"]) == _PNG_BYTES
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "https://images.example/v1/images/generations"
    assert call["headers"]["Authorization"] == "Bearer vision-key"
    assert call["json"]["model"] == "remote-image-model"
    assert call["json"]["size"] == "64x64"


def test_images_edits_without_model_uses_configured_openai_compatible_default(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("ABSTRACTVISION_MODEL_ID", "remote-image-model")
    monkeypatch.setenv("OPENAI_API_KEY", "vision-key")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    files = {"image": ("image.png", b"\x89PNG\r\n\x1a\nabc", "image/png")}
    resp = client.post("/v1/images/edits", data={"prompt": "edit"}, files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert base64.b64decode(data["data"][0]["b64_json"]) == _PNG_BYTES
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "https://images.example/v1/images/edits"
    assert call["headers"]["Authorization"] == "Bearer vision-key"
    assert call["data"]["model"] == "remote-image-model"
    assert call["data"]["prompt"] == "edit"


def test_images_generations_without_model_returns_501_when_unconfigured(client, monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_VISION_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_MODEL_ID", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL", raising=False)

    resp = client.post("/v1/images/generations", json={"prompt": "hello", "response_format": "b64_json"})

    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "not configured" in data["error"]["message"]


def test_images_generations_returns_501_when_sdcpp_unconfigured(client, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "sdcpp")
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL", raising=False)
    resp = client.post("/v1/images/generations", json={"prompt": "hello", "response_format": "b64_json"})
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "not configured for sdcpp mode" in data["error"]["message"]


def test_images_edits_returns_501_when_sdcpp_unconfigured(client, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "sdcpp")
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL", raising=False)
    files = {"image": ("image.png", b"\x89PNG\r\n\x1a\nabc", "image/png")}
    resp = client.post("/v1/images/edits", data={"prompt": "edit"}, files=files)
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "not configured for sdcpp mode" in data["error"]["message"]


def test_images_generations_rejects_chat_model_id(client, monkeypatch):
    """Ensure AbstractCore-style chat model ids don't get misrouted as image models."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_BACKEND", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_MODEL_ID", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL", raising=False)

    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "hello", "model": "openai/gpt-4o-mini", "response_format": "b64_json"},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data
    assert "not supported by `/v1/images/*`" in data["error"]["message"]


def test_openai_compatible_generation_uses_size_not_width_height(monkeypatch):
    from abstractcore.server import vision_endpoints

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    width, height, extra = vision_endpoints._image_generation_request_parts(
        {
            "model": "openai-compatible/gpt-image-2",
            "prompt": "hello",
            "size": "256x256",
            "width": 256,
            "height": 256,
        }
    )

    assert width is None
    assert height is None
    assert extra["size"] == "256x256"


class _FakeProxyResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        content: Optional[bytes] = None,
        content_type: str = "image/png",
    ):
        self._payload = payload
        self.status_code = status_code
        self.content = content if content is not None else b""
        self.headers = {"content-type": content_type}
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeProxyClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers=None, json=None, data=None, files=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json, "data": data, "files": files})
        content = _MP4_BYTES if "/videos/" in str(url) else _PNG_BYTES
        return _FakeProxyResponse(
            {"data": [{"b64_json": base64.b64encode(content).decode("ascii")}]},
            content_type="video/mp4" if content == _MP4_BYTES else "image/png",
        )

    def get(self, url):
        self.calls.append({"url": url, "method": "GET"})
        return _FakeProxyResponse({}, content=_PNG_BYTES)


def test_openai_compatible_generation_proxy_success(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    resp = client.post(
        "/v1/images/generations",
        json={
            "prompt": "a red square",
            "model": "openai-compatible/gpt-image-2",
            "width": 256,
            "height": 256,
            "response_format": "b64_json",
            "seed": 1234,
            "steps": 20,
            "guidance_scale": 7.5,
            "negative_prompt": "blur",
            "quality": "standard",
            "extra": {"safety_checker": False},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert base64.b64decode(data["data"][0]["b64_json"]) == _PNG_BYTES
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "https://images.example/v1/images/generations"
    assert call["headers"]["Authorization"] == "Bearer provider-key"
    assert call["json"]["model"] == "gpt-image-2"
    assert call["json"]["size"] == "256x256"
    assert call["json"]["quality"] == "standard"
    assert call["json"]["safety_checker"] is False
    assert "seed" not in call["json"]
    assert "steps" not in call["json"]
    assert "guidance_scale" not in call["json"]
    assert "negative_prompt" not in call["json"]
    assert "response_format" not in call["json"]
    assert "width" not in call["json"]
    assert "height" not in call["json"]


def test_provider_scoped_images_generation_prefixes_plain_model(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    resp = client.post(
        "/openai-compatible/v1/images/generations",
        json={
            "prompt": "a red square",
            "model": "gpt-image-2",
            "width": 256,
            "height": 256,
            "response_format": "b64_json",
        },
    )

    assert resp.status_code == 200
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "https://images.example/v1/images/generations"
    assert call["headers"]["Authorization"] == "Bearer provider-key"
    assert call["json"]["model"] == "gpt-image-2"
    assert call["json"]["size"] == "256x256"


def test_images_generations_accepts_request_base_url_override_and_provider_key(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    resp = client.post(
        "/v1/images/generations",
        headers={"X-AbstractCore-Provider-API-Key": "request-image-key"},
        json={
            "prompt": "a red square",
            "provider": "openai-compatible",
            "base_url": "http://127.0.0.1:5000/v1",
            "width": 128,
            "height": 128,
            "response_format": "b64_json",
        },
    )

    assert resp.status_code == 200
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "http://127.0.0.1:5000/v1/images/generations"
    assert call["headers"]["Authorization"] == "Bearer request-image-key"
    assert call["json"]["model"] == "default"
    assert call["json"]["size"] == "128x128"


def test_provider_scoped_openai_images_generation_defaults_to_openai_api(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    resp = client.post(
        "/openai/v1/images/generations",
        json={"prompt": "a red square", "model": "gpt-image-1", "response_format": "b64_json"},
    )

    assert resp.status_code == 200
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "https://api.openai.com/v1/images/generations"
    assert call["headers"]["Authorization"] == "Bearer openai-key"
    assert call["json"]["model"] == "gpt-image-1"


def test_openai_compatible_generation_proxy_allows_backend_specific_extra(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://images.example/v1")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    resp = client.post(
        "/v1/images/generations",
        json={
            "prompt": "a red square",
            "model": "openai-compatible/custom-image-model",
            "width": 256,
            "height": 256,
            "response_format": "b64_json",
            "extra": {"seed": 1234, "steps": 20, "guidance_scale": 7.5},
        },
    )

    assert resp.status_code == 200
    call = _FakeProxyClient.calls[0]
    assert call["json"]["seed"] == 1234
    assert call["json"]["steps"] == 20
    assert call["json"]["guidance_scale"] == 7.5


def test_openai_compatible_video_generation_proxy_success(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    resp = client.post(
        "/v1/videos/generations",
        json={
            "prompt": "a red square slowly rotating",
            "model": "openai-compatible/custom-video-model",
            "width": 1280,
            "height": 704,
            "fps": 24,
            "num_frames": 41,
            "steps": 10,
            "guidance_scale": 5.0,
            "guidance_2": 3.0,
            "response_format": "b64_json",
            "extra": {"max_sequence_length": 256},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert base64.b64decode(data["data"][0]["b64_json"]) == _MP4_BYTES
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "https://images.example/v1/videos/generations"
    assert call["headers"]["Authorization"] == "Bearer provider-key"
    assert call["json"]["model"] == "custom-video-model"
    assert call["json"]["prompt"] == "a red square slowly rotating"
    assert call["json"]["width"] == 1280
    assert call["json"]["height"] == 704
    assert call["json"]["fps"] == 24
    assert call["json"]["num_frames"] == 41
    assert call["json"]["guidance_2"] == 3.0
    assert call["json"]["max_sequence_length"] == 256


def test_openai_compatible_image_to_video_proxy_success(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    files = {"image": ("image.png", _PNG_BYTES, "image/png")}
    resp = client.post(
        "/v1/videos/edits",
        headers={"X-AbstractCore-Provider-API-Key": "request-video-key"},
        data={
            "prompt": "slow camera push-in",
            "provider": "openai-compatible",
            "base_url": "http://127.0.0.1:5000/v1",
            "width": "1280",
            "height": "704",
            "fps": "24",
            "num_frames": "41",
            "guidance_2": "3.5",
            "max_sequence_length": "256",
        },
        files=files,
    )

    assert resp.status_code == 200
    data = resp.json()
    assert base64.b64decode(data["data"][0]["b64_json"]) == _MP4_BYTES
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "http://127.0.0.1:5000/v1/videos/edits"
    assert call["headers"]["Authorization"] == "Bearer request-video-key"
    assert call["data"]["model"] == "default"
    assert call["data"]["prompt"] == "slow camera push-in"
    assert call["data"]["width"] == "1280"
    assert call["data"]["height"] == "704"
    assert call["data"]["guidance_2"] == "3.5"
    assert call["data"]["fps"] == "24"
    assert call["data"]["num_frames"] == "41"
    assert call["data"]["max_sequence_length"] == "256"
    assert "image" in call["files"]


def test_images_generation_schema_uses_width_height_not_size(client):
    schema = client.get("/openapi.json").json()
    body_schema = schema["components"]["schemas"]["ImageGenerationBody"]
    props = body_schema["properties"]

    assert "width" in props
    assert "height" in props
    assert "size" in props
    assert "base_url" in props
    assert body_schema.get("additionalProperties") is False
    assert "additionalProp1" not in str(body_schema)


def test_openai_compatible_edit_proxy_success_with_abstractvision_env(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("ABSTRACTVISION_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "vision-key")
    monkeypatch.setenv("ABSTRACTVISION_MODEL_ID", "remote-image-model")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    files = {"image": ("image.png", _PNG_BYTES, "image/png")}
    resp = client.post("/v1/images/edits", data={"prompt": "make it watercolor"}, files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert base64.b64decode(data["data"][0]["b64_json"]) == _PNG_BYTES
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "https://images.example/v1/images/edits"
    assert call["headers"]["Authorization"] == "Bearer vision-key"
    assert call["data"]["model"] == "remote-image-model"
    assert call["data"]["prompt"] == "make it watercolor"
    assert "response_format" not in call["data"]
    assert "image" in call["files"]


def test_images_edits_accepts_provider_and_request_base_url_override(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    files = {"image": ("image.png", _PNG_BYTES, "image/png")}
    resp = client.post(
        "/v1/images/edits",
        headers={"X-AbstractCore-Provider-API-Key": "request-image-key"},
        data={
            "prompt": "make it watercolor",
            "provider": "openai-compatible",
            "base_url": "http://127.0.0.1:5000/v1",
            "size": "256x256",
        },
        files=files,
    )

    assert resp.status_code == 200
    call = _FakeProxyClient.calls[0]
    assert call["url"] == "http://127.0.0.1:5000/v1/images/edits"
    assert call["headers"]["Authorization"] == "Bearer request-image-key"
    assert call["data"]["model"] == "default"
    assert call["data"]["size"] == "256x256"


def test_openai_compatible_generation_with_abstractvision_env_strips_provider_prefix(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("OPENAI_BASE_URL", "https://images.example/v1")
    monkeypatch.setattr(vision_endpoints.httpx, "Client", _FakeProxyClient)

    resp = client.post(
        "/v1/images/generations",
        json={
            "prompt": "a red square",
            "model": "openai-compatible/gpt-image-2",
            "width": 256,
            "height": 256,
            "response_format": "b64_json",
        },
    )

    assert resp.status_code == 200
    call = _FakeProxyClient.calls[0]
    assert call["json"]["model"] == "gpt-image-2"
    assert call["json"]["size"] == "256x256"
    assert "width" not in call["json"]
    assert "height" not in call["json"]


@dataclass
class _FakeGeneratedAsset:
    data: bytes = _PNG_BYTES
    mime_type: str = "image/png"


class _FakeImageGenerationRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeImageEditRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeVideoGenerationRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeImageToVideoRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeImageUpscaleRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeDiffusersConfig:
    instances: list["_FakeDiffusersConfig"] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.instances.append(self)


class _FakeDiffusersBackend:
    requests: list[Any] = []

    def __init__(self, *, config):
        self.config = config

    def generate_image(self, request):
        self.requests.append(request)
        return _FakeGeneratedAsset()

    def generate_image_with_progress(self, request, progress_callback=None):
        self.requests.append(request)
        if callable(request.extra.get("on_progress")):
            class _Event:
                phase = "denoise"
                step = 2
                total_steps = 4
                progress = 0.5
                step_progress = 0.5

            request.extra["on_progress"](_Event())
        if progress_callback is not None:
            progress_callback(2, 4)
        return _FakeGeneratedAsset()

    def edit_image(self, request):
        self.requests.append(request)
        return _FakeGeneratedAsset()

    def edit_image_with_progress(self, request, progress_callback=None):
        self.requests.append(request)
        if callable(request.extra.get("on_progress")):
            class _Event:
                phase = "denoise"
                step = 3
                total_steps = 6
                progress = 0.5
                step_progress = 0.5

            request.extra["on_progress"](_Event())
        if progress_callback is not None:
            progress_callback(3, 6)
        return _FakeGeneratedAsset()

    def upscale_image(self, request):
        self.requests.append(request)
        return _FakeGeneratedAsset()

    def upscale_image_with_progress(self, request, progress_callback=None):
        self.requests.append(request)
        if callable(request.extra.get("on_progress")):
            class _Event:
                phase = "denoise"
                step = 1
                total_steps = 1
                progress = 1.0
                step_progress = 1.0
                task = "image_upscale"

            request.extra["on_progress"](_Event())
        if progress_callback is not None:
            progress_callback(1, 1)
        return _FakeGeneratedAsset()

    def generate_video(self, request):
        self.requests.append(request)
        return _FakeGeneratedAsset(data=_MP4_BYTES, mime_type="video/mp4")

    def generate_video_with_progress(self, request, progress_callback=None):
        self.requests.append(request)
        if callable(request.extra.get("on_progress")):
            class _Event:
                phase = "generate"
                frame = 3
                total_frames = 5
                step = 2
                total_steps = 4
                progress = 0.5
                step_progress = 0.5
                frame_progress = 0.6

            request.extra["on_progress"](_Event())
        if progress_callback is not None:
            progress_callback(2, 4)
        return _FakeGeneratedAsset(data=_MP4_BYTES, mime_type="video/mp4")

    def image_to_video(self, request):
        self.requests.append(request)
        return _FakeGeneratedAsset(data=_MP4_BYTES, mime_type="video/mp4")

    def image_to_video_with_progress(self, request, progress_callback=None):
        self.requests.append(request)
        if progress_callback is not None:
            progress_callback(2, 4)
        return _FakeGeneratedAsset(data=_MP4_BYTES, mime_type="video/mp4")


def test_diffusers_default_provider_model_uses_configured_diffusers_model(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeDiffusersConfig.instances = []
    _FakeDiffusersBackend.requests = []

    def fake_import_abstractvision():
        return (
            object,
            object,
            _FakeDiffusersConfig,
            _FakeDiffusersBackend,
            object,
            object,
            object,
            object,
            RuntimeError,
            (_FakeImageGenerationRequest, _FakeImageEditRequest),
        )

    monkeypatch.setattr(vision_endpoints, "_import_abstractvision", fake_import_abstractvision)
    monkeypatch.setenv("ABSTRACTCORE_VISION_MODEL_ID", "example/local-image-model")

    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a small red square", "model": "diffusers/default", "width": 64, "height": 64, "steps": 2},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert base64.b64decode(data["data"][0]["b64_json"]) == _PNG_BYTES
    cfg = _FakeDiffusersConfig.instances[0]
    assert cfg.model_id == "example/local-image-model"
    assert cfg.device == "auto"
    assert cfg.allow_download is False
    req = _FakeDiffusersBackend.requests[0]
    assert req.prompt == "a small red square"
    assert req.width == 64
    assert req.height == 64
    assert req.steps == 2


def test_mflux_provider_model_uses_mflux_backend_without_diffusers_prefix(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeDiffusersConfig.instances = []
    _FakeDiffusersBackend.requests = []

    def fake_import_abstractvision():
        return (
            object,
            object,
            object,
            object,
            _FakeDiffusersConfig,
            _FakeDiffusersBackend,
            object,
            object,
            RuntimeError,
            (_FakeImageGenerationRequest, _FakeImageEditRequest),
        )

    monkeypatch.setattr(vision_endpoints, "_import_abstractvision", fake_import_abstractvision)

    resp = client.post(
        "/v1/images/generations",
        json={
            "prompt": "a small red square",
            "provider": "mflux",
            "model": "AbstractFramework/flux.2-klein-9b-4bit",
            "width": 64,
            "height": 64,
            "steps": 2,
        },
    )

    assert resp.status_code == 200
    cfg = _FakeDiffusersConfig.instances[0]
    assert cfg.model == "AbstractFramework/flux.2-klein-9b-4bit"
    assert not str(cfg.model).startswith("diffusers/")
    assert _FakeDiffusersBackend.requests[0].prompt == "a small red square"


def test_mflux_provider_video_generation_uses_exact_model_and_video_request(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeDiffusersConfig.instances = []
    _FakeDiffusersBackend.requests = []

    def fake_import_abstractvision():
        return (
            object,
            object,
            object,
            object,
            _FakeDiffusersConfig,
            _FakeDiffusersBackend,
            object,
            object,
            RuntimeError,
            (_FakeImageGenerationRequest, _FakeImageEditRequest, _FakeVideoGenerationRequest, _FakeImageToVideoRequest),
        )

    monkeypatch.setattr(vision_endpoints, "_import_abstractvision", fake_import_abstractvision)

    resp = client.post(
        "/v1/videos/generations",
        json={
            "prompt": "a small red square moving",
            "provider": "mlx-gen",
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "width": 1280,
            "height": 704,
            "fps": 24,
            "num_frames": 41,
            "steps": 10,
            "guidance_2": 3.25,
            "extra": {"max_sequence_length": 256},
        },
    )

    assert resp.status_code == 200
    assert base64.b64decode(resp.json()["data"][0]["b64_json"]) == _MP4_BYTES
    cfg = _FakeDiffusersConfig.instances[0]
    assert cfg.model == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    req = _FakeDiffusersBackend.requests[0]
    assert req.prompt == "a small red square moving"
    assert req.width == 1280
    assert req.height == 704
    assert req.fps == 24
    assert req.num_frames == 41
    assert req.guidance_2 == 3.25
    assert req.extra["max_sequence_length"] == 256


def test_video_generation_job_records_progress_event(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeDiffusersConfig.instances = []
    _FakeDiffusersBackend.requests = []

    def fake_import_abstractvision():
        return (
            object,
            object,
            object,
            object,
            _FakeDiffusersConfig,
            _FakeDiffusersBackend,
            object,
            object,
            RuntimeError,
            (_FakeImageGenerationRequest, _FakeImageEditRequest, _FakeVideoGenerationRequest, _FakeImageToVideoRequest),
        )

    monkeypatch.setattr(vision_endpoints, "_import_abstractvision", fake_import_abstractvision)

    resp = client.post(
        "/v1/vision/jobs/videos/generations",
        json={
            "prompt": "a small red square moving",
            "provider": "mlx-gen",
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "num_frames": 5,
            "steps": 4,
            "guidance_2": 2.75,
        },
    )

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    data = {}
    for _ in range(50):
        poll = client.get(f"/v1/vision/jobs/{job_id}")
        assert poll.status_code == 200
        data = poll.json()
        if data["state"] == "succeeded":
            break
        time.sleep(0.02)

    assert data["state"] == "succeeded"
    assert base64.b64decode(data["result"]["data"][0]["b64_json"]) == _MP4_BYTES
    progress = data["progress"]
    assert progress["step"] == 2
    assert progress["total_steps"] == 4
    assert progress["last_event"]["frame"] == 3
    assert progress["last_event"]["total_frames"] == 5
    assert progress["last_event"]["step"] == 2
    assert progress["last_event"]["total_steps"] == 4
    assert progress["last_event"]["progress"] == 0.5
    assert progress["last_event"]["step_progress"] == 0.5
    assert progress["last_event"]["frame_progress"] == 0.6
    assert _FakeDiffusersBackend.requests[0].guidance_2 == 2.75


def test_image_generation_job_records_progress_event(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeDiffusersConfig.instances = []
    _FakeDiffusersBackend.requests = []

    def fake_import_abstractvision():
        return (
            object,
            object,
            object,
            object,
            _FakeDiffusersConfig,
            _FakeDiffusersBackend,
            object,
            object,
            RuntimeError,
            (_FakeImageGenerationRequest, _FakeImageEditRequest, _FakeVideoGenerationRequest, _FakeImageToVideoRequest),
        )

    monkeypatch.setattr(vision_endpoints, "_import_abstractvision", fake_import_abstractvision)

    resp = client.post(
        "/v1/vision/jobs/images/generations",
        json={
            "prompt": "a small red square",
            "provider": "mlx-gen",
            "model": "AbstractFramework/flux.2-klein-9b-8bit",
            "steps": 4,
        },
    )

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    data = {}
    for _ in range(50):
        poll = client.get(f"/v1/vision/jobs/{job_id}")
        assert poll.status_code == 200
        data = poll.json()
        if data["state"] == "succeeded":
            break
        time.sleep(0.02)

    assert data["state"] == "succeeded"
    assert base64.b64decode(data["result"]["data"][0]["b64_json"]) == _PNG_BYTES
    progress = data["progress"]
    assert progress["step"] == 2
    assert progress["total_steps"] == 4
    assert progress["progress"] == 0.5
    assert progress["last_event"]["phase"] == "denoise"
    assert progress["last_event"]["step_progress"] == 0.5


def test_image_upscale_job_records_progress_event(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeDiffusersConfig.instances = []
    _FakeDiffusersBackend.requests = []

    def fake_import_abstractvision():
        return (
            object,
            object,
            object,
            object,
            _FakeDiffusersConfig,
            _FakeDiffusersBackend,
            object,
            object,
            RuntimeError,
            (
                _FakeImageGenerationRequest,
                _FakeImageEditRequest,
                _FakeVideoGenerationRequest,
                _FakeImageToVideoRequest,
                _FakeImageUpscaleRequest,
            ),
        )

    monkeypatch.setattr(vision_endpoints, "_import_abstractvision", fake_import_abstractvision)

    files = {"image": ("image.png", b"\x89PNG\r\n\x1a\nabc", "image/png")}
    resp = client.post(
        "/v1/vision/jobs/images/upscale",
        data={
            "provider": "mlx-gen",
            "model": "AbstractFramework/seedvr2-3b-8bit",
            "scale": "2x",
            "softness": "0.25",
            "quantize": "8",
        },
        files=files,
    )

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    data = {}
    for _ in range(50):
        poll = client.get(f"/v1/vision/jobs/{job_id}")
        assert poll.status_code == 200
        data = poll.json()
        if data["state"] == "succeeded":
            break
        time.sleep(0.02)

    assert data["state"] == "succeeded"
    assert base64.b64decode(data["result"]["data"][0]["b64_json"]) == _PNG_BYTES
    req = _FakeDiffusersBackend.requests[0]
    assert isinstance(req, _FakeImageUpscaleRequest)
    assert req.scale == "2x"
    assert req.softness == 0.25
    assert req.quantize == 8
    progress = data["progress"]
    assert progress["step"] == 1
    assert progress["total_steps"] == 1
    assert progress["last_event"]["task"] == "image_upscale"
    assert progress["last_event"]["step_progress"] == 1.0


def test_images_upscale_route_forwards_exact_q4_seedvr2_model(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeDiffusersConfig.instances = []
    _FakeDiffusersBackend.requests = []

    def fake_import_abstractvision():
        return (
            object,
            object,
            object,
            object,
            _FakeDiffusersConfig,
            _FakeDiffusersBackend,
            object,
            object,
            RuntimeError,
            (
                _FakeImageGenerationRequest,
                _FakeImageEditRequest,
                _FakeVideoGenerationRequest,
                _FakeImageToVideoRequest,
                _FakeImageUpscaleRequest,
            ),
        )

    monkeypatch.setattr(vision_endpoints, "_import_abstractvision", fake_import_abstractvision)

    files = {"image": ("image.png", b"\x89PNG\r\n\x1a\nabc", "image/png")}
    resp = client.post(
        "/v1/images/upscale",
        data={
            "provider": "mlx-gen",
            "model": "AbstractFramework/seedvr2-7b-4bit",
            "scale": "2x",
            "softness": "0.25",
        },
        files=files,
    )

    assert resp.status_code == 200
    assert base64.b64decode(resp.json()["data"][0]["b64_json"]) == _PNG_BYTES
    assert _FakeDiffusersConfig.instances[-1].model == "AbstractFramework/seedvr2-7b-4bit"
    req = _FakeDiffusersBackend.requests[0]
    assert isinstance(req, _FakeImageUpscaleRequest)
    assert req.scale == "2x"
    assert req.softness == 0.25


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("scale", "-1", "scale must be positive"),
        ("resolution", "large", "resolution must be a positive integer"),
        ("softness", "2", "softness must be <="),
        ("seed", "abc", "seed must be an integer"),
        ("quantize", "abc", "quantize must be an integer"),
        ("quantize", "7", "quantize must be one of"),
        ("vae_tiling", "maybe", "vae_tiling must be a boolean"),
    ],
)
@pytest.mark.parametrize("path", ["/v1/images/upscale", "/v1/vision/jobs/images/upscale"])
def test_image_upscale_routes_reject_malformed_form_fields(client, path, field, value, detail):
    data = {
        "provider": "mlx-gen",
        "model": "AbstractFramework/seedvr2-3b-8bit",
        "scale": "2x",
        "softness": "0.25",
        "seed": "1234",
        "quantize": "8",
        "vae_tiling": "false",
    }
    data[field] = value
    files = {"image": ("image.png", b"\x89PNG\r\n\x1a\nabc", "image/png")}

    resp = client.post(path, data=data, files=files)

    assert resp.status_code == 400
    assert detail in resp.text


def test_images_edits_forwards_reference_images_to_abstractvision(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeDiffusersConfig.instances = []
    _FakeDiffusersBackend.requests = []

    def fake_import_abstractvision():
        return (
            object,
            object,
            object,
            object,
            _FakeDiffusersConfig,
            _FakeDiffusersBackend,
            object,
            object,
            RuntimeError,
            (_FakeImageGenerationRequest, _FakeImageEditRequest, _FakeVideoGenerationRequest, _FakeImageToVideoRequest),
        )

    monkeypatch.setattr(vision_endpoints, "_import_abstractvision", fake_import_abstractvision)

    files = [
        ("image", ("source.png", b"source-image", "image/png")),
        ("reference_images", ("style.png", b"style-reference", "image/png")),
        ("reference_images", ("layout.png", b"layout-reference", "image/png")),
    ]
    resp = client.post(
        "/v1/images/edits",
        data={
            "prompt": "compose with references",
            "provider": "mlx-gen",
            "model": "AbstractFramework/flux.2-klein-9b-8bit",
        },
        files=files,
    )

    assert resp.status_code == 200
    assert base64.b64decode(resp.json()["data"][0]["b64_json"]) == _PNG_BYTES
    req = _FakeDiffusersBackend.requests[0]
    assert req.prompt == "compose with references"
    assert req.extra["reference_images"] == [b"style-reference", b"layout-reference"]


def test_diffusers_default_provider_model_requires_configured_model(client):
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a small red square", "model": "diffusers/default", "width": 64, "height": 64, "steps": 2},
    )

    assert resp.status_code == 501
    data = resp.json()
    assert "Diffusers mode" in data["error"]["message"]
    assert "ABSTRACTCORE_VISION_MODEL_ID" in data["error"]["message"]


def test_server_default_provider_model_is_rejected(client):
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a small red square", "model": "server/default", "width": 64, "height": 64, "steps": 2},
    )

    assert resp.status_code == 400
    data = resp.json()
    assert "Omit `model`" in data["error"]["message"]


def test_removed_local_abstractvision_alias_is_rejected(client):
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a small red square", "model": "local/abstractvision", "width": 64, "height": 64},
    )

    assert resp.status_code == 400
    data = resp.json()
    assert "diffusers/default" in data["error"]["message"]
