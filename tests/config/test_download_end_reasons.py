"""Mission KK (2026-09-24): a download that stops on its own is never "Cancelled".

"Cancelled" follows only an explicit cancel REQUEST, and the job records who
made it (`cancelled_by`: console / api / cli / other_process, plus the account
when known). Everything else -- a dropped connection, a Hub error, a full
disk, the owning process restarting, a failed MTP companion -- is `failed`,
with one plain sentence (`ended_reason`) saying what happened and what a retry
reuses. Also: a copied Hugging Face cache whose snapshot links are missing or
dangling is never reported installed.

Offline: fake runners, fake handlers, tmp caches. No network, no live port.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from abstractcore.config import host_jobs
from abstractcore.config import model_materializer as mm
from tests.models_engines_fakes import isolate_host

DROP = "httpx.RemoteProtocolError: peer closed connection without sending complete message body (received 0 bytes, expected 65011712)"


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


def _progress(ctx, done, total):
    ctx.progress({"status": "downloading", "message": "x", "downloaded_bytes": done, "total_bytes": total, "phase": "downloading"})


# ---------------------------------------------------------------------------
# The wrong mapping: nothing but a request makes a job "cancelled"
# ---------------------------------------------------------------------------


def test_a_network_drop_is_failed_with_a_plain_reason_never_cancelled(host):
    reg = host_jobs.HostJobRegistry()

    def runner(ctx):
        _progress(ctx, 200_000_000, 266_000_000)
        return {"ok": False, "status": "failed", "message": DROP}

    job = reg.start(kind="download", key="k", runner=runner, provider="mlx", artifact="a/b", run_inline=True)
    assert job["status"] == "failed" and job["state"] == "failed"
    assert job["cancelled_by"] is None and job["cancel_requested"] is False
    assert job["ended_reason"].startswith("The connection to Hugging Face dropped after 200 MB of 266 MB.")
    assert "the files that finished are kept, and the file that was in progress starts over" in job["ended_reason"]
    assert job["error"] == DROP  # the verbatim error is kept for "Show details"


def test_a_companion_that_fails_mid_download_is_failed_and_says_the_model_is_here(host, monkeypatch):
    ref, companion = "org/Model-4bit", "org/Model-MTP-4bit"

    def probe(pid, artifact):
        status = mm.PRESENCE_INSTALLED if artifact == ref else mm.PRESENCE_ABSENT
        return mm.ModelPresence(pid, artifact, status, evidence="test", location="/x" if status == mm.PRESENCE_INSTALLED else None)

    monkeypatch.setattr(mm, "_probe_huggingface", probe)
    monkeypatch.setattr(mm, "_hf_file_plan", lambda repo, patterns, token: ([{"name": "model.safetensors", "size": 266_000_000, "etag": "e"}], "rev", ""))
    fetched = []

    def handler(artifact, emit, base_url):
        fetched.append(artifact)
        emit(mm.DownloadProgress(status=mm.DownloadStatus.DOWNLOADING, message="m", downloaded_bytes=200_000_000, total_bytes=266_000_000))
        return mm.DownloadOutcome("huggingface", artifact, False, "failed", message=DROP)

    reg = host_jobs.HostJobRegistry()
    job = reg.start(
        kind="download",
        key="k",
        provider="mlx",
        artifact=ref,
        run_inline=True,
        runner=lambda ctx: mm._download_with_companions("mlx", ref, [companion], handler, ctx.progress, base_url=None, dry_run=False, expected_bytes=None).to_dict(),
    )
    assert fetched == [companion]  # the model part was already installed
    assert job["status"] == "failed" and job["cancelled_by"] is None
    assert job["ended_reason"].startswith("The model itself is already on this computer; the small add-on that makes it faster")
    assert "The connection to Hugging Face dropped after 200 MB of 266 MB." in job["ended_reason"]


def test_the_owner_restarting_mid_download_is_failed_never_cancelled(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "dl_dead.json").write_text(json.dumps({
        "schema": "host_job_v1", "job_id": "dl_dead", "status": "running", "state": "downloading", "pid": 999999,
        "provider": "mlx", "downloaded_bytes": 105_000_000, "total_bytes": 275_000_000, "started_at": "x",
        "files": [{"name": "model.safetensors", "state": "downloading"}, {"name": "config.json", "state": "done"}],
    }))
    snap = host_jobs.read_persisted_job("dl_dead", jobs_dir)
    assert snap["status"] == "failed" and snap["state"] == "failed"
    assert snap.get("cancelled_by") is None
    assert "stopped after 105 MB of 275 MB because the program running it" in snap["ended_reason"]
    assert "Nobody cancelled it." in snap["ended_reason"]
    assert [f["state"] for f in snap["files"]] == ["failed", "done"]


def test_a_user_cancel_is_cancelled_and_says_who(host):
    reg = host_jobs.HostJobRegistry(tick_s=0.02)
    started = []

    def runner(ctx):
        _progress(ctx, 105_000_000, 275_000_000)
        started.append(1)
        ctx.control.event.wait(10)
        ctx.control.check()
        return {"ok": True}

    job = reg.start(kind="download", key="k", runner=runner, provider="mlx", artifact="a/b")
    while not started:
        time.sleep(0.01)
    reg.cancel(job["job_id"], by="console", user="admin")
    reg.cancel(job["job_id"], by="api")  # a second request never rewrites who asked first
    done = reg.wait(job["job_id"], 10)
    assert done["status"] == "cancelled"
    assert (done["cancelled_by"], done["cancelled_by_user"]) == ("console", "admin")
    assert done["ended_reason"].startswith("Cancelled in the console by admin at ")
    assert "after 105 MB of 275 MB" in done["ended_reason"]


def test_a_cancel_marker_from_the_cli_says_the_command_line(host, tmp_path):
    jobs_dir = tmp_path / "jobs"
    owner = host_jobs.HostJobRegistry(persist_dir=jobs_dir, tick_s=0.02)
    job = owner.start(kind="download", key="k", runner=lambda ctx: (ctx.control.event.wait(10), ctx.control.check(), {"ok": True})[-1], provider="ollama", artifact="q")
    assert host_jobs.request_cancel(job["job_id"], jobs_dir, by="cli")["message"] == "cancel requested"
    done = owner.wait(job["job_id"], 10)
    assert done["status"] == "cancelled" and done["cancelled_by"] == "cli"
    assert done["ended_reason"].startswith("Cancelled from the command line")
    assert "Ollama keeps what it already fetched" in done["ended_reason"]


@pytest.mark.parametrize(
    "error,start",
    [
        (DROP, "The connection to Hugging Face dropped"),
        ("httpx.ReadTimeout: The read operation timed out", "Hugging Face stopped answering"),
        ("httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known", "This computer could not reach Hugging Face"),
        ("LocalEntryNotFoundError: An error happened while trying to locate the files on the Hub", "This computer could not reach Hugging Face"),
        ("HfHubHTTPError: 503 Server Error: Service Unavailable for url: https://huggingface.co/x", "Hugging Face had a problem on its side"),
        ("HfHubHTTPError: 416 Client Error: Range Not Satisfiable for url: x", "Hugging Face refused to continue the unfinished file"),
        ("RepositoryNotFoundError: 404 Client Error. Repository Not Found for url: x", "Hugging Face says this model"),
        ("GatedRepoError: 401 Client Error. Cannot access gated repo", "Hugging Face refused access to this model"),
        ("OSError: [Errno 28] No space left on device", "The disk is full"),
        ("ValueError: something odd", "The download stopped after 1.0 MB with an error: ValueError: something odd."),
    ],
)
def test_failure_reasons_are_plain_and_specific(error, start):
    reason = host_jobs.failure_reason(error, {"provider": "mlx", "downloaded_bytes": 1_000_000})
    assert reason.startswith(start), reason
    assert "HF_TOKEN" not in reason and "export " not in reason


def test_a_traceback_line_number_is_not_an_http_status():
    output = 'Traceback (most recent call last):\n  File "_base.py", line 401, in __get_result\n    raise self._exception\n' + DROP
    reason = host_jobs.failure_reason("companion failed: " + DROP, {"provider": "mlx"}, output=output)
    assert reason.startswith("The connection to Hugging Face dropped"), reason


def test_the_hf_download_maps_a_network_error_to_failed_with_a_matching_instruction(host, monkeypatch):
    pytest.importorskip("huggingface_hub", reason="the Hugging Face download path needs huggingface_hub (not in the [test] extra)")
    monkeypatch.setattr(mm, "_hf_file_plan", lambda repo, patterns, token: ([{"name": "model.safetensors", "size": 10, "etag": "e1"}], "0" * 40, ""))
    monkeypatch.setattr(mm, "_hf_transfer_subprocess", lambda kwargs, control, watcher, env=None: (False, DROP, "Traceback...\n" + DROP))
    reg = host_jobs.HostJobRegistry()
    job = reg.start(kind="download", key="k", provider="huggingface", artifact="org/m", run_inline=True,
                    runner=lambda ctx: mm._download_huggingface("org/m", ctx.progress, None).to_dict())
    assert job["status"] == "failed" and job["cancelled_by"] is None
    instruction = job["result"]["instruction"]
    assert instruction.startswith("The connection to Hugging Face dropped")
    assert "gated" not in instruction and "HF_TOKEN" not in instruction


def test_the_hf_download_cancelled_message_is_honest_about_what_is_kept(host, monkeypatch):
    pytest.importorskip("huggingface_hub", reason="the Hugging Face download path needs huggingface_hub (not in the [test] extra)")
    monkeypatch.setattr(mm, "_hf_file_plan", lambda repo, patterns, token: ([{"name": "model.safetensors", "size": 10, "etag": "e1"}], "0" * 40, ""))

    def transfer(kwargs, control, watcher, env=None):
        control.cancel()
        return False, "killed", ""

    monkeypatch.setattr(mm, "_hf_transfer_subprocess", transfer)
    reg = host_jobs.HostJobRegistry()
    job = reg.start(kind="download", key="k", provider="huggingface", artifact="org/m", run_inline=True,
                    runner=lambda ctx: mm._download_huggingface("org/m", ctx.progress, None).to_dict())
    assert job["status"] == "cancelled"
    assert "the next download resumes" not in job["result"]["message"]
    assert "the file that was in progress starts over" in job["result"]["message"]


# ---------------------------------------------------------------------------
# The download child never outlives its owner (a restart stops the bytes)
# ---------------------------------------------------------------------------


def test_the_hf_child_exits_when_its_owner_dies(tmp_path):
    stub = tmp_path / "stub"
    (stub / "huggingface_hub").mkdir(parents=True)
    (stub / "huggingface_hub" / "__init__.py").write_text("import time\ndef snapshot_download(**kw):\n    time.sleep(60)\n")
    pidfile = tmp_path / "child.pid"
    owner = (
        "import json, os, subprocess, sys\n"
        f"from abstractcore.config.model_materializer import _HF_CHILD\n"
        "p = subprocess.Popen([sys.executable, '-c', _HF_CHILD, json.dumps({}), str(os.getpid())], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "os._exit(0)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(stub), os.getcwd()]))
    subprocess.run([sys.executable, "-c", owner], env=env, check=True, timeout=30)
    child = int(pidfile.read_text())
    deadline = time.time() + 10
    alive = True
    while time.time() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.2)
    if alive:
        os.kill(child, 9)
    assert not alive, "the download child kept running after its owner exited"
