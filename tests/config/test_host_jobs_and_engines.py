"""Contract E (host jobs) and contract B (engines: detection, plans, install).

No real installer ever runs: install plans are asserted as data, and the one
test that executes an install swaps the plan for a harmless Python child.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest

from abstractcore.config import engines, host_jobs
from abstractcore.config import model_materializer as mm
from tests.models_engines_fakes import FakeOllama, install_fake_tool, isolate_host, synthetic_host

_JOB_KEYS = {
    "schema", "job_id", "kind", "status", "provider", "artifact", "engine", "percent", "downloaded_bytes",
    "total_bytes", "message", "log_tail", "command", "dry_run", "started_at", "finished_at", "error",
    "joined", "cli_equivalent",
}


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# Registry semantics
# ---------------------------------------------------------------------------


def test_job_dict_has_every_contract_field(host):
    reg = host_jobs.HostJobRegistry()
    job = reg.start(kind="download", key="k", runner=lambda ctx: {"ok": True, "message": "done"}, run_inline=True)
    assert _JOB_KEYS <= set(job)
    assert job["schema"] == "host_job_v1"
    assert job["status"] == "completed" and job["percent"] == 100.0
    assert job["job_id"].startswith("dl_")
    # Gateway model_downloads.py aliases.
    assert job["job"] == job["job_id"] and job["events"] == job["log_tail"]


def test_a_second_request_for_the_same_key_joins_the_running_job(host):
    reg = host_jobs.HostJobRegistry()
    gate = threading.Event()
    runs = []

    def runner(ctx):
        runs.append(1)
        gate.wait(5)
        return {"ok": True}

    first = reg.start(kind="download", key="download:ollama/qwen3:8b", runner=runner)
    second = reg.start(kind="download", key="download:ollama/qwen3:8b", runner=runner)
    assert second["job_id"] == first["job_id"]
    assert second["joined"] == 1
    gate.set()
    assert reg.wait(first["job_id"], 5)["status"] == "completed"
    assert len(runs) == 1
    # The slot is free again: a retry starts real work.
    third = reg.start(kind="download", key="download:ollama/qwen3:8b", runner=lambda ctx: {"ok": True}, run_inline=True)
    assert third["job_id"] != first["job_id"]


def test_join_false_refuses_a_concurrent_job(host):
    reg = host_jobs.HostJobRegistry()
    gate = threading.Event()
    reg.start(kind="engine_install", key="engine_install", runner=lambda ctx: gate.wait(5) and {"ok": True}, join=False)
    with pytest.raises(host_jobs.JobBusy) as busy:
        reg.start(kind="engine_install", key="engine_install", runner=lambda ctx: {"ok": True}, join=False)
    assert busy.value.job["kind"] == "engine_install"
    gate.set()


def test_cancel_terminates_the_subprocess_and_reports_cancelled(host):
    reg = host_jobs.HostJobRegistry()
    marker = host["config"] / "child-started"
    script = f"import pathlib,time; pathlib.Path({str(marker)!r}).write_text('x'); print('working', flush=True); time.sleep(60)"

    def runner(ctx):
        outcome = mm._run_streaming([sys.executable, "-c", script], "engine", "test", ctx.progress)
        return outcome.to_dict()

    job = reg.start(kind="engine_install", key="engine_install", runner=runner, join=False)
    deadline = time.time() + 10
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert marker.exists()
    reg.cancel(job["job_id"])
    done = reg.wait(job["job_id"], 15)
    assert done["status"] == "cancelled", done
    assert "cancelled" in done["message"]


def test_failures_keep_the_tools_own_words(host):
    reg = host_jobs.HostJobRegistry()

    def runner(ctx):
        return mm._run_streaming([sys.executable, "-c", "print('E: no such formula'); raise SystemExit(3)"], "engine", "x", ctx.progress).to_dict()

    done = reg.start(kind="engine_install", key="engine_install", runner=runner, run_inline=True)
    assert done["status"] == "failed"
    assert "E: no such formula" in done["log_tail"]
    assert "exited 3" in done["error"]


def test_retention_is_bounded_and_never_drops_a_running_job(host):
    reg = host_jobs.HostJobRegistry(max_jobs=5)
    gate = threading.Event()
    running = reg.start(kind="download", key="long", runner=lambda ctx: gate.wait(5) and {"ok": True})
    for i in range(12):
        reg.start(kind="download", key=f"k{i}", runner=lambda ctx: {"ok": True}, run_inline=True)
    jobs = reg.list()
    assert len(jobs) <= 5
    assert running["job_id"] in {j["job_id"] for j in jobs}
    gate.set()


def test_persisted_snapshots_and_cancel_markers_work_across_processes(host, tmp_path):
    jobs_dir = tmp_path / "jobs"
    owner = host_jobs.HostJobRegistry(persist_dir=jobs_dir)
    started = threading.Event()

    def runner(ctx):
        started.set()
        while not ctx.control.is_cancelled():
            time.sleep(0.05)
        raise host_jobs.JobCancelled()

    job = owner.start(kind="download", key="x", runner=runner)
    assert started.wait(5)
    # "Another process": read the file, drop a cancel marker.
    snap = host_jobs.read_persisted_job(job["job_id"], jobs_dir)
    assert snap["status"] == "running"
    assert host_jobs.request_cancel(job["job_id"], jobs_dir)["message"] == "cancel requested"
    done = owner.wait(job["job_id"], 10)
    assert done["status"] == "cancelled"
    assert host_jobs.read_persisted_job(job["job_id"], jobs_dir)["status"] == "cancelled"
    assert not (jobs_dir / f"{job['job_id']}.cancel").exists()


def test_a_running_snapshot_whose_owner_died_reads_as_failed(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "dl_dead.json").write_text(
        json.dumps({"schema": "host_job_v1", "job_id": "dl_dead", "status": "running", "pid": 999999, "started_at": "x"})
    )
    snap = host_jobs.read_persisted_jobs(jobs_dir)[0]
    assert snap["status"] == "failed" and "owner process exited" in snap["error"]


def test_download_job_streams_real_byte_progress_from_ollama(host, monkeypatch):
    with FakeOllama([]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        reg = host_jobs.HostJobRegistry()
        job = host_jobs.start_download_job("ollama", "qwen3:8b", registry=reg, run_inline=True)
    assert job["status"] == "completed", job
    assert job["total_bytes"] == 1000 and job["downloaded_bytes"] == 1000
    assert job["percent"] == 100.0
    assert job["cli_equivalent"] == "abstractcore models download ollama qwen3:8b"
    assert job["result"]["status"] == "completed"


def test_download_job_cancel_stops_the_ollama_stream(host, monkeypatch):
    with FakeOllama([]) as fake:
        fake.pull_chunks = 200
        fake.pull_delay = 0.05
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        reg = host_jobs.HostJobRegistry()
        job = host_jobs.start_download_job("ollama", "qwen3:8b", registry=reg)
        deadline = time.time() + 10
        while (reg.get(job["job_id"]) or {}).get("downloaded_bytes") in (None, 0) and time.time() < deadline:
            time.sleep(0.05)
        reg.cancel(job["job_id"])
        done = reg.wait(job["job_id"], 10)
    assert done["status"] == "cancelled", done
    assert done["percent"] < 100


def test_dry_run_download_job_spends_nothing(host, monkeypatch):
    monkeypatch.setitem(mm._DOWNLOADERS, "ollama", lambda *a: pytest.fail("dry run downloaded"))
    job = host_jobs.start_download_job("ollama", "qwen3:8b", dry_run=True, registry=host_jobs.HostJobRegistry(), run_inline=True)
    assert job["status"] == "completed" and job["dry_run"] is True
    assert job["result"]["status"] == "planned"
    assert job["percent"] is None


# ---------------------------------------------------------------------------
# Engines: plans are FIXED argv from an allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine,os_name,tools,method,argv_head",
    [
        ("ollama", "darwin", {"brew": True}, "brew", ["brew", "install", "ollama"]),
        ("ollama", "darwin", {"brew": False}, "script", ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]),
        ("ollama", "linux", {}, "script", ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]),
        ("ollama", "windows", {"winget": True}, "winget", ["winget", "install", "--id", "Ollama.Ollama"]),
        ("ollama", "windows", {"winget": False}, "script", ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "irm https://ollama.com/install.ps1 | iex"]),
        ("lmstudio", "darwin", {"brew": True}, "download_page", ["bash", "-c", "curl -fsSL https://lmstudio.ai/install.sh | bash"]),
        ("lmstudio", "windows", {"winget": True}, "download_page", ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "irm https://lmstudio.ai/install.ps1 | iex"]),
    ],
)
def test_install_plans_come_from_the_allowlist(engine, os_name, tools, method, argv_head):
    plan = engines.engine_install_plan(engine, os_name, "arm64" if os_name == "darwin" else "x86_64", tools=tools)
    assert plan["available"] is True
    assert plan["method"] == method
    assert plan["argv"][: len(argv_head)] == argv_head
    assert plan["requires_confirmation"] is True
    assert plan["notes"]


def test_linux_ollama_plan_says_it_needs_sudo():
    plan = engines.engine_install_plan("ollama", "linux", "x86_64", tools={})
    assert plan["requires_admin"] is True and "sudo" in plan["notes"].lower()


def test_lmstudio_is_a_download_page_plus_optional_headless_bootstrap():
    plan = engines.engine_install_plan("lmstudio", "darwin", "arm64", tools={"brew": True})
    assert plan["url"] == "https://lmstudio.ai/download"
    assert plan["alternatives"][0]["argv"] == ["brew", "install", "--cask", "lm-studio"]
    intel = engines.engine_install_plan("lmstudio", "darwin", "x86_64", tools={})
    assert intel["available"] is False and "Apple Silicon" in intel["notes"]


def test_pip_engines_install_into_this_interpreter():
    pip = engines.engine_install_plan("mlx", "darwin", "arm64", prefer_uv=False)
    assert pip["argv"] == [sys.executable, "-m", "pip", "install", "mlx-lm"]
    uv = engines.engine_install_plan("mlx", "darwin", "arm64", prefer_uv=True)
    assert uv["argv"] == ["uv", "pip", "install", "--python", sys.executable, "mlx-lm"]
    llama = engines.engine_install_plan("llamacpp", "darwin", "arm64", tools={"brew": True}, prefer_uv=False)
    assert llama["argv"][-1] == "llama-cpp-python"
    assert llama["alternatives"][0]["argv"] == ["brew", "install", "llama.cpp"]
    win = engines.engine_install_plan("llamacpp", "windows", "x86_64", tools={"winget": True}, prefer_uv=False)
    assert win["alternatives"][0]["argv"][:4] == ["winget", "install", "--id", "ggml.llamacpp"]


def test_mlx_and_vllm_are_refused_where_they_cannot_run():
    assert engines.engine_install_plan("mlx", "linux", "x86_64")["available"] is False
    assert engines.engine_install_plan("vllm", "darwin", "arm64")["available"] is False
    assert engines.engine_install_plan("vllm", "linux", "x86_64", accelerator="none")["available"] is False
    cuda = engines.engine_install_plan("vllm", "linux", "x86_64", accelerator="cuda", prefer_uv=True)
    assert cuda["available"] is True and "--torch-backend=auto" in cuda["argv"]


def test_unknown_engine_is_a_key_error():
    with pytest.raises(KeyError):
        engines.engine_install_plan("rm -rf /", "linux", "x86_64")


# ---------------------------------------------------------------------------
# Engines: detection and inventory
# ---------------------------------------------------------------------------

_ENGINE_KEYS = {
    "id", "name", "kind", "supported_on_host", "unsupported_reason", "installed", "version", "install_location",
    "running", "base_url", "reachable", "models_count", "install", "docs_url",
}


def test_inventory_rows_have_every_contract_field(host):
    payload = engines.engine_inventory(probe=False, host=synthetic_host("cpu16"))
    assert payload["schema"] == "engines_status_v1"
    ids = [e["id"] for e in payload["engines"]]
    assert ids == ["ollama", "lmstudio", "mlx", "llamacpp", "vllm", "huggingface"]
    for row in payload["engines"]:
        assert _ENGINE_KEYS <= set(row), row["id"]
        assert row["running"] is None  # not probed
    vllm = next(e for e in payload["engines"] if e["id"] == "vllm")
    assert vllm["kind"] == "remote_only" and vllm["supported_on_host"] is False


def test_ollama_detection_reads_the_cli_version_and_probe_reads_the_server(host, monkeypatch):
    install_fake_tool(host["fakebin"], "ollama", "print('ollama version is 0.20.2')")
    with FakeOllama([{"name": "a:1b"}, {"name": "b:2b"}]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        row = engines.engine_status("ollama", probe=True, host=synthetic_host("cpu16"))
    assert row["installed"] is True and row["version"] == "0.20.2"
    assert row["install_location"] == str(host["fakebin"] / "ollama")
    assert row["running"] is True and row["reachable"] is True and row["models_count"] == 2
    assert row["base_url"] == fake.url


def test_absent_engine_is_not_installed_with_an_install_plan(host):
    row = engines.engine_status("ollama", probe=True, host=synthetic_host("cpu16"))
    assert row["installed"] is False and row["running"] is False
    assert row["install"]["available"] is True


def test_lms_cli_version_is_reported(host, monkeypatch):
    install_fake_tool(host["fakebin"], "lms", "print('lms is LM Studio CLI')\nprint('CLI commit: 71bd99c')")
    monkeypatch.setenv("ABSTRACTCORE_LMS_CLI", str(host["fakebin"] / "lms"))
    row = engines.engine_status("lmstudio", host=synthetic_host("cpu16"))
    assert row["installed"] is True
    assert row["cli_version"] == "cli-71bd99c"


# ---------------------------------------------------------------------------
# Engines: install policy and execution
# ---------------------------------------------------------------------------


def test_install_is_refused_when_not_allowed_but_dry_run_still_shows_the_plan(host, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL", "0")
    with pytest.raises(engines.EngineInstallRefused) as refused:
        engines.engine_install("ollama", registry=host_jobs.HostJobRegistry())
    assert refused.value.reason == "not_allowed"
    job = engines.engine_install("ollama", dry_run=True, registry=host_jobs.HostJobRegistry(), run_inline=True)
    assert job["status"] == "completed" and job["result"]["status"] == "planned"
    assert job["command"] and job["cli_equivalent"].endswith("--dry-run")


def test_unsupported_engine_install_is_refused(host, monkeypatch):
    monkeypatch.setattr(engines, "_support", lambda *a: (False, "not here"))
    with pytest.raises(engines.EngineInstallRefused) as refused:
        engines.engine_install("mlx", registry=host_jobs.HostJobRegistry())
    assert refused.value.reason == "unsupported"


def test_install_runs_the_plan_argv_as_a_streamed_job(host, monkeypatch):
    fake_plan = {
        "available": True,
        "method": "pip",
        "argv": [sys.executable, "-c", "print('Collecting mlx-lm'); print('Successfully installed mlx-lm')"],
        "url": None,
        "requires_confirmation": True,
        "requires_admin": False,
        "estimated_bytes": None,
        "notes": "test",
        "alternatives": [],
    }
    monkeypatch.setattr(engines, "engine_install_plan", lambda *a, **k: dict(fake_plan))
    monkeypatch.setattr(engines, "_support", lambda *a: (True, None))
    reg = host_jobs.HostJobRegistry()
    job = engines.engine_install("mlx", force=True, registry=reg, run_inline=True)
    assert job["kind"] == "engine_install" and job["engine"] == "mlx"
    assert job["status"] == "completed", job
    assert "Successfully installed mlx-lm" in job["log_tail"]
    assert job["command"] == fake_plan["argv"]


def test_an_installed_engine_is_not_reinstalled_without_force(host, monkeypatch):
    install_fake_tool(host["fakebin"], "ollama", "print('ollama version is 0.20.2')")
    monkeypatch.setattr(mm, "_run_streaming", lambda *a, **k: pytest.fail("reinstalled"))
    job = engines.engine_install("ollama", registry=host_jobs.HostJobRegistry(), run_inline=True)
    assert job["result"]["status"] == "already_installed"


def test_loopback_policy_for_the_server(monkeypatch):
    from abstractcore.server import host_routes

    monkeypatch.delenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL", raising=False)
    for bind, allowed in [("127.0.0.1", True), ("localhost", True), ("::1", True), ("0.0.0.0", False), ("192.168.1.4", False), ("", False)]:
        monkeypatch.setenv("ABSTRACTCORE_SERVER_BIND_HOST", bind)
        assert host_routes.server_allows_engine_install() is allowed, bind
    monkeypatch.setenv("ABSTRACTCORE_SERVER_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL", "1")
    assert host_routes.server_allows_engine_install() is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_cancel_kills_the_whole_process_tree_not_just_the_child(host):
    """An installer (brew, pip) spawns children; cancel must stop them too."""

    reg = host_jobs.HostJobRegistry()
    pidfile = host["config"] / "grandchild.pid"
    script = (
        "import subprocess, sys, time, pathlib\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))\n"
        "print('spawned', flush=True)\n"
        "time.sleep(60)\n"
    )

    def runner(ctx):
        return mm._run_streaming([sys.executable, "-c", script], "engine", "tree", ctx.progress).to_dict()

    job = reg.start(kind="engine_install", key="engine_install", runner=runner, join=False)
    deadline = time.time() + 10
    while not pidfile.exists() and time.time() < deadline:
        time.sleep(0.05)
    grandchild = int(pidfile.read_text())
    reg.cancel(job["job_id"])
    assert reg.wait(job["job_id"], 15)["status"] == "cancelled"
    deadline = time.time() + 10
    alive = True
    while time.time() < deadline:
        try:
            os.kill(grandchild, 0)
            # A zombie still answers kill(0); reap check via waitpid is not ours to do.
            with open(f"/proc/{grandchild}/status") as fh:  # linux
                if "zombie" in fh.read().lower():
                    alive = False
                    break
        except ProcessLookupError:
            alive = False
            break
        except FileNotFoundError:
            # macOS: no /proc; use ps to tell zombies apart
            out = __import__("subprocess").run(["ps", "-o", "stat=", "-p", str(grandchild)], capture_output=True, text=True).stdout.strip()
            if not out or out.startswith("Z"):
                alive = False
                break
        time.sleep(0.1)
    assert not alive, "the grandchild survived the cancel"
