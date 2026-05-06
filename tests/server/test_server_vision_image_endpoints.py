import os
import base64
from dataclasses import dataclass
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from abstractcore.server.app import app

_PNG_BYTES = b"\x89PNG\r\n\x1a\nabstractcore-test-png"


@pytest.fixture(autouse=True)
def clean_vision_state(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("ABSTRACTCORE_VISION_", "ABSTRACTVISION_")):
            monkeypatch.delenv(key, raising=False)

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
    monkeypatch.setenv("ABSTRACTVISION_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("ABSTRACTVISION_MODEL_ID", "remote-image-model")
    monkeypatch.setenv("ABSTRACTVISION_API_KEY", "vision-key")
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
    monkeypatch.setenv("ABSTRACTVISION_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("ABSTRACTVISION_MODEL_ID", "remote-image-model")
    monkeypatch.setenv("ABSTRACTVISION_API_KEY", "vision-key")
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
    monkeypatch.delenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", raising=False)
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
    monkeypatch.delenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", raising=False)
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

    monkeypatch.setenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", "https://api.openai.com/v1")

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
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200, content: Optional[bytes] = None):
        self._payload = payload
        self.status_code = status_code
        self.content = content if content is not None else b""
        self.headers = {"content-type": "image/png"}
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
        return _FakeProxyResponse({"data": [{"b64_json": base64.b64encode(_PNG_BYTES).decode("ascii")}]})

    def get(self, url):
        self.calls.append({"url": url, "method": "GET"})
        return _FakeProxyResponse({}, content=_PNG_BYTES)


def test_openai_compatible_generation_proxy_success(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("ABSTRACTCORE_VISION_UPSTREAM_API_KEY", "provider-key")
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


def test_openai_compatible_generation_proxy_allows_backend_specific_extra(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", "https://images.example/v1")
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


def test_images_generation_schema_uses_width_height_not_size(client):
    schema = client.get("/openapi.json").json()
    body_schema = schema["components"]["schemas"]["ImageGenerationBody"]
    props = body_schema["properties"]

    assert "width" in props
    assert "height" in props
    assert "size" not in props
    assert body_schema.get("additionalProperties") is False
    assert "additionalProp1" not in str(body_schema)


def test_openai_compatible_edit_proxy_success_with_abstractvision_env(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("ABSTRACTVISION_BACKEND", "openai")
    monkeypatch.setenv("ABSTRACTVISION_BASE_URL", "https://images.example/v1")
    monkeypatch.setenv("ABSTRACTVISION_API_KEY", "vision-key")
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


def test_openai_compatible_generation_with_abstractvision_env_strips_provider_prefix(client, monkeypatch):
    from abstractcore.server import vision_endpoints

    _FakeProxyClient.calls = []
    monkeypatch.setenv("ABSTRACTVISION_BASE_URL", "https://images.example/v1")
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

    def edit_image(self, request):
        self.requests.append(request)
        return _FakeGeneratedAsset()


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
