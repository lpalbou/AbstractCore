from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.basic


def test_server_capability_defaults_routes_persist_to_core_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)

    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    with client:
        created = client.put(
            "/v1/config/capability-defaults/output/text",
            json={
                "provider": "lmstudio",
                "model": "qwen/qwen3.6-35b-a3b",
                "base_url": "http://127.0.0.1:1234/v1",
                "options": {"profile": "local"},
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        route = next(item for item in body["routes"] if item["key"] == "output.text")
        assert route["kind"] == "output"
        assert route["provider"] == "lmstudio"
        assert route["model"] == "qwen/qwen3.6-35b-a3b"
        assert route["base_url"] == "http://127.0.0.1:1234/v1"
        assert route["options"] == {"profile": "local"}
        assert route["source"] == "abstractcore.capability_defaults"

        listed = client.get("/v1/config/capability-defaults")
        assert listed.status_code == 200, listed.text
        assert any(item["key"] == "output.text" and item["configured"] for item in listed.json()["routes"])

        cleared = client.delete("/v1/config/capability-defaults/output/text")
        assert cleared.status_code == 200, cleared.text
        cleared_route = next(item for item in cleared.json()["routes"] if item["key"] == "output.text")
        assert cleared_route["configured"] is False


def test_server_task_capability_defaults_routes_persist_to_core_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)

    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    with client:
        created = client.put(
            "/v1/config/capability-defaults/output/image/image_to_image",
            json={
                "provider": "mlx-gen",
                "model": "AbstractFramework/qwen-image-edit-2511-4bit",
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        route = next(item for item in body["routes"] if item["key"] == "output.image.image_to_image")
        assert route["kind"] == "output"
        assert route["modality"] == "image"
        assert route["task"] == "image_to_image"
        assert route["provider"] == "mlx-gen"
        assert route["model"] == "AbstractFramework/qwen-image-edit-2511-4bit"
        assert route["source"] == "abstractcore.capability_defaults"

        listed = client.get("/v1/config/capability-defaults")
        assert listed.status_code == 200, listed.text
        routes = {item["key"]: item for item in listed.json()["routes"]}
        assert routes["output.image.image_to_image"]["configured"] is True
        assert routes["output.image"]["configured"] is False

        cleared = client.delete("/v1/config/capability-defaults/output/image/image_to_image")
        assert cleared.status_code == 200, cleared.text
        cleared_route = next(item for item in cleared.json()["routes"] if item["key"] == "output.image.image_to_image")
        assert cleared_route["configured"] is False
