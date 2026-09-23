"""`/acore/host|engines|models|jobs` over HTTP (contracts A-F), auth on and off.

The routes are thin: payloads are the config-layer payloads. What is tested
here is the HTTP policy -- who may mutate the host, what a refusal looks like
-- plus one end-to-end download job through a fake Ollama.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from abstractcore.server import app as server_app
from tests.models_engines_fakes import FakeOllama, isolate_host, make_hf_repo, ollama_tag

TOKEN = "acore-test-token"


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


@pytest.fixture
def client(host, monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_BIND_HOST", "127.0.0.1")
    return TestClient(server_app.app)


@pytest.fixture
def authed(host, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", TOKEN)
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_BIND_HOST", "127.0.0.1")
    return TestClient(server_app.app)


def _wait_job(client, job_id, headers=None, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/acore/jobs/{job_id}", headers=headers or {}).json()
        if job.get("status") not in ("queued", "running"):
            return job
        time.sleep(0.05)
    return job


def test_reads_return_the_contract_payloads(client):
    assert client.get("/acore/host/profile").json()["schema"] == "host_profile_v1"
    engines = client.get("/acore/engines").json()
    assert engines["schema"] == "engines_status_v1" and engines["install_allowed"] is True
    assert client.get("/acore/engines/ollama").json()["id"] == "ollama"
    assert client.get("/acore/engines/nope").status_code == 404
    catalog = client.get("/acore/models/catalog", params={"q": "qwen3 8b", "engine": "ollama", "fits": "true"}).json()
    assert catalog["schema"] == "model_catalog_v1" and catalog["query"]["engine"] == "ollama"
    installed = client.get("/acore/models/installed", params={"provider": "huggingface"}).json()
    assert installed["schema"] == "models_installed_v1"
    jobs = client.get("/acore/jobs").json()
    assert jobs["schema"] == "host_jobs_v1" and isinstance(jobs["jobs"], list)


def test_download_job_end_to_end_through_a_fake_ollama(client, monkeypatch):
    with FakeOllama([]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        job = client.post("/acore/models/download", json={"provider": "ollama", "artifact": "qwen3:8b"}).json()
        assert job["schema"] == "host_job_v1" and job["kind"] == "download"
        done = _wait_job(client, job["job_id"])
    assert done["status"] == "completed" and done["percent"] == 100.0
    listed = client.get("/acore/jobs", params={"kind": "download", "status": "completed"}).json()["jobs"]
    assert any(j["job_id"] == job["job_id"] for j in listed)
    assert all(j["kind"] == "download" and j["status"] == "completed" for j in listed)


def test_dry_runs_come_back_finished(client):
    job = client.post("/acore/models/download", json={"provider": "ollama", "artifact": "qwen3:8b", "dry_run": True}).json()
    assert job["status"] == "completed" and job["result"]["status"] == "planned"
    install = client.post("/acore/engines/ollama/install", json={"dry_run": True}).json()
    assert install["status"] == "completed" and install["command"]


def test_delete_refusals_are_structured(client, host, monkeypatch):
    with FakeOllama([ollama_tag("qwen3:8b", 10, "8B", "Q4_K_M")], loaded=["qwen3:8b"]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        r = client.post("/acore/models/delete", json={"provider": "ollama", "artifact": "qwen3:8b"})
        assert r.status_code == 409
        body = r.json()
        assert body["status"] == "refused" and body["delete_blockers"] == ["loaded"]
        assert body["error"]["message"]
        forced = client.post("/acore/models/delete", json={"provider": "ollama", "artifact": "qwen3:8b", "force": True}).json()
        assert _wait_job(client, forced["job_id"])["status"] == "completed"
    missing = client.post("/acore/models/delete", json={"provider": "huggingface", "artifact": "no/such"})
    assert missing.status_code == 404 and missing.json()["status"] == "not_found"


def test_delete_hf_repo(client, host):
    folder = make_hf_repo(host["hf"], "unsloth/Qwen3-8B-GGUF", {"Qwen3-8B-Q4_K_M.gguf": b"q" * 30})
    job = client.post("/acore/models/delete", json={"provider": "huggingface", "artifact": "unsloth/Qwen3-8B-GGUF"}).json()
    assert _wait_job(client, job["job_id"])["status"] == "completed"
    assert not folder.exists()


def test_engine_install_is_403_off_loopback_and_409_when_busy(client, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_BIND_HOST", "0.0.0.0")
    r = client.post("/acore/engines/ollama/install", json={})
    assert r.status_code == 403 and r.json()["reason"] == "not_allowed"
    assert client.get("/acore/engines").json()["install_allowed"] is False

    monkeypatch.setenv("ABSTRACTCORE_SERVER_BIND_HOST", "127.0.0.1")
    from abstractcore.config import host_jobs

    gate = __import__("threading").Event()
    host_jobs.default_registry().start(kind="engine_install", key="engine_install", runner=lambda ctx: gate.wait(5) and {"ok": True}, join=False)
    try:
        busy = client.post("/acore/engines/ollama/install", json={"force": True})
        assert busy.status_code == 409 and busy.json()["status"] == "busy"
    finally:
        gate.set()
    assert client.post("/acore/engines/nope/install", json={}).status_code == 404


def test_cancel_route(client):
    from abstractcore.config import host_jobs

    registry = host_jobs.default_registry()

    def runner(ctx):
        while not ctx.control.is_cancelled():
            time.sleep(0.02)
        raise host_jobs.JobCancelled()

    job = registry.start(kind="download", key="x", runner=runner)
    r = client.post(f"/acore/jobs/{job['job_id']}/cancel")
    assert r.status_code == 200
    assert _wait_job(client, job["job_id"])["status"] == "cancelled"
    assert client.post("/acore/jobs/dl_nope/cancel").status_code == 404
    assert client.get("/acore/jobs/dl_nope").status_code == 404


def test_with_auth_enabled_every_route_needs_the_token(authed):
    assert authed.get("/acore/engines").status_code == 401
    assert authed.post("/acore/models/download", json={"provider": "ollama", "artifact": "x:1b", "dry_run": True}).status_code == 401
    headers = {"Authorization": f"Bearer {TOKEN}"}
    assert authed.get("/acore/engines", headers=headers).status_code == 200
    ok = authed.post("/acore/models/download", json={"provider": "ollama", "artifact": "x:1b", "dry_run": True}, headers=headers)
    assert ok.status_code == 200 and ok.json()["status"] == "completed"


def test_a_provider_key_alone_cannot_mutate_the_host(host, monkeypatch):
    """No server token + no ALLOW_UNAUTHENTICATED: the middleware lets a request
    through on an upstream provider key -- which must not delete files here."""

    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    client = TestClient(server_app.app)
    headers = {"X-AbstractCore-Provider-API-Key": "sk-upstream-provider-key"}
    r = client.post("/acore/models/delete", json={"provider": "huggingface", "artifact": "a/b"}, headers=headers)
    assert r.status_code == 403
    r = client.post("/acore/engines/ollama/install", json={"dry_run": True}, headers=headers)
    assert r.status_code == 403
