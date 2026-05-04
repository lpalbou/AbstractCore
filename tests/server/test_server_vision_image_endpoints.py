import os

import pytest
from fastapi.testclient import TestClient

from abstractcore.server.app import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_images_generations_returns_501_when_unconfigured(client, monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_BACKEND", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_MODEL_ID", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL", raising=False)
    resp = client.post("/v1/images/generations", json={"prompt": "hello", "response_format": "b64_json"})
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "Vision image endpoints are not configured" in data["error"]["message"]


def test_images_edits_returns_501_when_unconfigured(client, monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_BACKEND", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_MODEL_ID", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL", raising=False)
    files = {"image": ("image.png", b"\x89PNG\r\n\x1a\nabc", "image/png")}
    resp = client.post("/v1/images/edits", data={"prompt": "edit"}, files=files)
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "Vision image endpoints are not configured" in data["error"]["message"]


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


def test_images_generations_falls_back_when_chat_model_id_is_passed(client, monkeypatch):
    """Ensure AbstractCore-style chat model ids don't get misrouted as Diffusers model ids."""
    monkeypatch.delenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_BACKEND", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_MODEL_ID", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL", raising=False)

    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "hello", "model": "openai/gpt-4o-mini", "response_format": "b64_json"},
    )
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "Vision image endpoints are not configured" in data["error"]["message"]


def test_openai_compatible_generation_uses_size_not_width_height(monkeypatch):
    from abstractcore.server import vision_endpoints

    monkeypatch.setenv("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", "https://api.openai.com/v1")

    width, height, extra = vision_endpoints._image_generation_request_parts(
        {
            "model": "openai/dall-e-2",
            "prompt": "hello",
            "size": "256x256",
            "width": 256,
            "height": 256,
        }
    )

    assert width is None
    assert height is None
    assert extra["size"] == "256x256"
