from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_server_prompt_cache_control_plane_requires_base_url_or_provider_model() -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    r = client.post("/acore/prompt_cache/set", json={"key": "k1"})
    assert r.status_code == 200
    body = r.json()
    assert body["supported"] is False
    assert "base_url or provider+model" in body.get("error", "")

    r = client.get("/acore/prompt_cache/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["supported"] is False
    assert "base_url or provider+model" in body.get("error", "")


def test_server_bloc_control_plane_uses_local_store_without_base_url(tmp_path) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._SERVER_BLOC_STORE = server_app.FileBlocStore(root_dir=tmp_path / "gateway-blocs")
    client = TestClient(server_app.app)

    upsert = client.post(
        "/acore/blocs/upsert_text",
        json={"path": "/tmp/orbit.txt", "content": "Orbit notes"},
    )
    assert upsert.status_code == 200
    sha256 = upsert.json()["record"]["sha256"]

    record = client.get(f"/acore/blocs/record?sha256={sha256}")
    assert record.status_code == 200
    assert record.json()["record"]["sha256"] == sha256


def test_server_bloc_proxy_strips_v1_and_forwards_query_and_auth(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    seen = {}

    class _Resp:
        status_code = 200
        text = "ok"

        def json(self):
            return {"ok": True, "operation": "kv_manifest"}

    class _Client:
        def __init__(self, *args, **kwargs):
            seen["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None, params=None):
            seen["url"] = url
            seen["headers"] = dict(headers or {})
            seen["params"] = dict(params or {})
            return _Resp()

    monkeypatch.setattr(server_app.httpx, "Client", _Client)

    client = TestClient(server_app.app)
    resp = client.get(
        "/acore/blocs/kv/manifest",
        params={
            "base_url": "http://127.0.0.1:8001/v1",
            "sha256": "abc",
            "bloc_id": 7,
            "artifact_path": "/tmp/orbit.safetensors",
        },
        headers={"X-AbstractCore-Provider-API-Key": "provider-secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert seen["url"] == "http://127.0.0.1:8001/acore/blocs/kv/manifest"
    assert seen["headers"]["Authorization"] == "Bearer provider-secret"
    assert seen["params"] == {
        "sha256": "abc",
        "bloc_id": 7,
        "artifact_path": "/tmp/orbit.safetensors",
    }


def test_server_bloc_proxy_propagates_upstream_http_status(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    class _Resp:
        status_code = 404
        text = "not found"

        def json(self):
            return {"ok": False, "operation": "record", "error": "bloc not found"}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None, params=None):
            _ = (url, headers, params)
            return _Resp()

    monkeypatch.setattr(server_app.httpx, "Client", _Client)

    client = TestClient(server_app.app)
    resp = client.get("/acore/blocs/record", params={"base_url": "http://127.0.0.1:8001/v1", "sha256": "missing"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "bloc not found"
