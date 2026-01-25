from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_server_prompt_cache_control_plane_requires_base_url() -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    r = client.post("/acore/prompt_cache/set", json={"key": "k1"})
    assert r.status_code == 200
    body = r.json()
    assert body["supported"] is False
    assert "base_url" in body.get("error", "")

    r = client.get("/acore/prompt_cache/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["supported"] is False

