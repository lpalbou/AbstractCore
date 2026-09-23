"""Contract F over the CLI: `--json` smoke through real subprocesses.

Each test runs `python -m abstractcore.config.main ...` in an isolated host
(HOME, caches, config, PATH in tmp_path), exactly as the terminal console's
CLI transport does: stdout is parsed, exit codes are 0 / 1 / 2 (refused).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from tests.models_engines_fakes import FakeOllama, isolate_host, make_hf_repo, ollama_tag


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


def _cli(*args: str, timeout: float = 60.0):
    proc = subprocess.run(
        [sys.executable, "-m", "abstractcore.config.main", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _json(stdout: str):
    return json.loads(stdout)


def test_host_profile_json(host):
    code, out, err = _cli("host", "profile", "--json")
    assert code == 0, err
    assert _json(out)["schema"] == "host_profile_v1"


def test_engines_status_json(host):
    code, out, err = _cli("engines", "status", "--json")
    assert code == 0, err
    payload = _json(out)
    assert payload["schema"] == "engines_status_v1"
    assert {e["id"] for e in payload["engines"]} == {"ollama", "lmstudio", "mlx", "llamacpp", "vllm", "huggingface"}


def test_models_list_json_sees_the_hf_cache(host):
    make_hf_repo(host["hf"], "unsloth/Qwen3-8B-GGUF", {"Qwen3-8B-Q4_K_M.gguf": b"q" * 30})
    code, out, err = _cli("models", "list", "--json")
    assert code == 0, err
    payload = _json(out)
    assert payload["schema"] == "models_installed_v1"
    assert any(r["artifact"] == "unsloth/Qwen3-8B-GGUF" and r["size_bytes"] == 30 for r in payload["rows"])


def test_models_catalog_and_search_accept_engine_and_fits(host):
    code, out, err = _cli("models", "catalog", "--engine", "ollama", "--fits", "--json")
    assert code == 0, err
    payload = _json(out)
    assert payload["schema"] == "model_catalog_v1"
    assert payload["query"]["engine"] == "ollama" and payload["query"]["fits"] is True
    assert all(a["provider"] == "ollama" for r in payload["rows"] for a in r["artifacts"])
    code, out, _ = _cli("models", "search", "qwen3", "8b", "--json")
    # argparse: a second positional is an error -> the query is ONE argument
    assert code != 0
    code, out, err = _cli("models", "search", "qwen3 8b", "--engine", "mlx", "--json")
    assert code == 0, err
    assert any(r["id"] == "qwen3-8b" for r in _json(out)["rows"])


def test_engine_install_refusals_exit_2_with_a_reason(host, monkeypatch):
    code, out, _ = _cli("engines", "install", "vllm", "--yes", "--json")
    if code == 2:  # every host but linux+cuda
        body = _json(out)
        assert body["status"] == "refused" and body["reason"] == "unsupported"
    monkeypatch.setenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL", "0")
    start = time.monotonic()
    code, out, _ = _cli("engines", "install", "ollama", "--yes", "--json")
    assert code == 2
    assert time.monotonic() - start < 10
    body = _json(out)
    assert body["reason"] == "not_allowed" and body["install"]["argv"]
    monkeypatch.delenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL")
    code, out, _ = _cli("engines", "install", "ollama", "--json")  # no --yes in machine mode
    assert code == 2 and _json(out)["reason"] == "not_confirmed"


def test_engine_install_dry_run_prints_the_job_with_the_exact_command(host):
    code, out, err = _cli("engines", "install", "ollama", "--yes", "--dry-run", "--json")
    assert code == 0, err
    job = _json(out)
    assert job["schema"] == "host_job_v1" and job["kind"] == "engine_install"
    assert job["result"]["status"] in {"planned", "already_installed"}
    assert job["command"]


def test_engines_open_prints_the_download_url(host):
    code, out, _ = _cli("engines", "open", "lmstudio", "--no-browser", "--json")
    assert code == 0
    assert _json(out) == {"engine": "lmstudio", "url": "https://lmstudio.ai/download", "opened": False}


def test_models_delete_refusals_and_success(host):
    folder = make_hf_repo(host["hf"], "unsloth/Qwen3-8B-GGUF", {"Qwen3-8B-Q4_K_M.gguf": b"q" * 30})
    code, out, _ = _cli("models", "delete", "huggingface", "no/such-repo", "--yes", "--json")
    assert code == 1 and _json(out)["status"] == "not_found"

    code, out, _ = _cli("models", "delete", "huggingface", "unsloth/Qwen3-8B-GGUF", "--json")
    assert code == 2 and _json(out)["reason"] == "not_confirmed"
    assert folder.exists()

    code, out, _ = _cli("models", "delete", "mlx", "unsloth/Qwen3-8B-GGUF", "--yes", "--json")
    body = _json(out)
    assert code == 2 and body["delete_blockers"] == ["shared_cache:mlx,huggingface"]

    code, out, err = _cli("models", "delete", "huggingface", "unsloth/Qwen3-8B-GGUF", "--yes", "--json")
    assert code == 0, err
    job = _json(out)
    assert job["schema"] == "host_job_v1" and job["kind"] == "delete" and job["status"] == "completed"
    assert not folder.exists()

    code, out, _ = _cli("models", "jobs", "--json")
    jobs = _json(out)["jobs"]
    assert any(j["job_id"] == job["job_id"] for j in jobs)
    code, out, _ = _cli("models", "jobs", job["job_id"], "--json")
    assert code == 0 and _json(out)["status"] == "completed"
    code, out, _ = _cli("models", "jobs", "--kind", "download", "--json")
    assert all(j["kind"] == "download" for j in _json(out)["jobs"])
    code, _, _ = _cli("models", "cancel", "dl_doesnotexist", "--json")
    assert code == 1


def test_models_download_json_streams_ndjson_jobs(host, monkeypatch):
    with FakeOllama([]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        code, out, err = _cli("models", "download", "ollama", "qwen3:8b", "--json")
    assert code == 0, err
    lines = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(lines) >= 2
    assert all(line["schema"] == "host_job_v1" for line in lines)
    assert all(line["job_id"] == lines[0]["job_id"] for line in lines)
    final = lines[-1]
    assert final["status"] == "completed" and final["percent"] == 100.0
    assert final["total_bytes"] == 1000
    # Legacy one-document keys ride on the final line.
    assert final["ok"] is True and final["results"][0]["status"] == "completed"


def test_models_download_json_dry_run_is_one_final_line(host):
    code, out, err = _cli("models", "download", "ollama", "qwen3:8b", "--dry-run", "--json")
    assert code == 0, err
    lines = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0]["results"][0]["status"] == "planned"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_sigterm_cancels_the_download_and_the_last_line_says_cancelled(host, monkeypatch):
    with FakeOllama([]) as fake:
        fake.pull_chunks = 400
        fake.pull_delay = 0.05
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        proc = subprocess.Popen(
            [sys.executable, "-m", "abstractcore.config.main", "models", "download", "ollama", "qwen3:8b", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ),
        )
        seen = []
        assert proc.stdout is not None
        deadline = time.time() + 20
        for line in proc.stdout:
            seen.append(json.loads(line))
            if seen[-1].get("downloaded_bytes") or time.time() > deadline:
                break
        proc.send_signal(signal.SIGTERM)
        rest, _err = proc.communicate(timeout=20)
    final = json.loads(rest.strip().splitlines()[-1])
    assert final["status"] == "cancelled", final
    assert proc.returncode == 1
    assert final["percent"] is None or final["percent"] < 100


def test_models_download_detach_returns_a_pollable_job(host, monkeypatch):
    with FakeOllama([]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        code, out, err = _cli("models", "download", "ollama", "qwen3:8b", "--detach", "--json")
        assert code == 0, err
        queued = _json(out)
        assert queued["schema"] == "host_job_v1" and queued["status"] == "queued"
        deadline = time.time() + 30
        status = None
        while time.time() < deadline:
            code, out, _ = _cli("models", "jobs", queued["job_id"], "--json")
            if code == 0:
                status = _json(out)["status"]
                if status not in ("queued", "running"):
                    break
            time.sleep(0.3)
    assert status == "completed"
