"""The download PROGRESS contract (`host_job_v1` state/bytes/speed/ETA/files/stall).

The operator's verdict that started this: a download showed a bare
"downloading" pill for minutes, "no way to know if it was working before it
was finished". Every source must map its native progress into the same
contract, a silent transfer must say it is stalled, and cancel must stop the
bytes and leave nothing that later reads as installed.

Nothing here touches the network or a real engine: sources are local fake
servers / fake CLIs replaying the real tools' output formats.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from abstractcore.config import host_jobs
from abstractcore.config import model_materializer as mm
from abstractcore.download import DownloadProgress, DownloadStatus
from tests.models_engines_fakes import FakeOllama, isolate_host


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


def _wait(pred, timeout=10.0, step=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


# ---------------------------------------------------------------------------
# The registry: speed, ETA, message, stall, terminal numbers
# ---------------------------------------------------------------------------


def test_contract_fields_are_present_on_every_job(host):
    reg = host_jobs.HostJobRegistry()
    job = reg.start(kind="download", key="k", runner=lambda ctx: {"ok": True, "status": "completed"}, run_inline=True)
    for key in ("state", "bytes_done", "bytes_total", "size_unknown", "percent", "bytes_per_second", "eta_s",
                "started_at", "updated_at", "message", "files", "error", "detail", "transitions", "stall_after_s"):
        assert key in job, key
    assert job["state"] == "done"
    assert [t["state"] for t in job["transitions"]] == ["queued", "resolving", "done"]


def test_speed_eta_and_the_plain_message_come_from_the_bytes(host):
    reg = host_jobs.HostJobRegistry(tick_s=0.05, speed_window_s=1.0)
    gate = threading.Event()

    def runner(ctx):
        files = [{"name": "config.json", "bytes_done": 100, "bytes_total": 100, "state": "done"},
                 {"name": "model.safetensors", "bytes_done": 0, "bytes_total": 10_000_000, "state": "downloading"}]
        for i in range(1, 21):
            files[1]["bytes_done"] = i * 100_000
            ctx.progress(DownloadProgress(status=DownloadStatus.DOWNLOADING, message="raw tool line", downloaded_bytes=100 + i * 100_000,
                                          total_bytes=10_000_100, files=[dict(f) for f in files], current_file="model.safetensors"))
            time.sleep(0.05)  # 2 MB/s
        gate.wait(5)
        return {"ok": True, "status": "completed"}

    job = reg.start(kind="download", key="k", runner=runner)
    assert _wait(lambda: (reg.get(job["job_id"])["bytes_done"] or 0) >= 1_500_000)
    snap = reg.get(job["job_id"])
    assert snap["state"] == "downloading"
    assert 1_000_000 < snap["bytes_per_second"] < 3_500_000, snap["bytes_per_second"]
    assert snap["eta_s"] and 2 <= snap["eta_s"] <= 10
    assert snap["message"].startswith("Downloading model.safetensors (2 of 2) · "), snap["message"]
    assert " MB/s" in snap["message"] and "left" in snap["message"]
    assert snap["detail"] == "raw tool line", "the provider's own words are kept verbatim"
    assert snap["files"][1]["name"] == "model.safetensors"
    # Bytes stop: the speed DECAYS instead of freezing at its last value.
    assert _wait(lambda: reg.get(job["job_id"])["bytes_per_second"] == 0.0, timeout=5)
    gate.set()
    done = reg.wait(job["job_id"], 5)
    assert done["state"] == "done" and done["bytes_done"] == done["bytes_total"] == 10_000_100
    assert done["eta_s"] == 0 and done["bytes_per_second"] is None
    assert all(f["state"] == "done" for f in done["files"])
    assert done["message"].startswith("Downloaded 10 MB in ")


def test_a_silent_download_turns_stalled_says_so_and_resumes_by_itself(host, caplog):
    reg = host_jobs.HostJobRegistry(tick_s=0.05, stall_after_s=0.4)
    phase = {"go": threading.Event(), "end": threading.Event()}

    def runner(ctx):
        ctx.progress(DownloadProgress(status=DownloadStatus.DOWNLOADING, message="x", downloaded_bytes=1000, total_bytes=5000))
        phase["go"].wait(5)  # silence: no bytes
        ctx.progress(DownloadProgress(status=DownloadStatus.DOWNLOADING, message="x", downloaded_bytes=2000, total_bytes=5000))
        phase["end"].wait(5)
        return {"ok": True, "status": "completed"}

    caplog.set_level("WARNING", logger="abstractcore.host_jobs")
    job = reg.start(kind="download", key="k", runner=runner)
    jid = job["job_id"]
    assert _wait(lambda: reg.get(jid)["state"] == "stalled", timeout=5)
    snap = reg.get(jid)
    assert snap["status"] == "running", "a stall is not a failure"
    assert snap["message"].startswith("Stalled: no data for "), snap["message"]
    assert "1.0 KB of 5.0 KB" in snap["message"]
    assert snap["stalled_for_s"] is not None and snap["eta_s"] is None
    assert any("stalled: no bytes received" in line for line in snap["log_tail"])
    assert any("stalled" in rec.getMessage() for rec in caplog.records), "the stall is LOGGED"
    phase["go"].set()
    assert _wait(lambda: reg.get(jid)["state"] == "downloading", timeout=5)
    assert any(t["state"] == "downloading" and t["why"].startswith("resumed") for t in reg.get(jid)["transitions"])
    phase["end"].set()
    assert reg.wait(jid, 5)["state"] == "done"
    states = [e["state"] for e in reg.events(jid)]
    assert "stalled" in states and states.index("stalled") < len(states) - 1 - states[::-1].index("downloading")


def test_verifying_and_installing_never_count_as_a_stall(host):
    reg = host_jobs.HostJobRegistry(tick_s=0.05, stall_after_s=0.2)
    gate = threading.Event()

    def runner(ctx):
        ctx.progress(DownloadProgress(status=DownloadStatus.VERIFYING, message="verifying sha256 digest", phase="verifying"))
        gate.wait(5)
        return {"ok": True, "status": "completed"}

    job = reg.start(kind="download", key="k", runner=runner)
    time.sleep(0.6)
    assert reg.get(job["job_id"])["state"] == "verifying"
    gate.set()
    reg.wait(job["job_id"], 5)


def test_unknown_size_is_flagged_with_a_reason(host):
    reg = host_jobs.HostJobRegistry(tick_s=0.05)
    gate = threading.Event()

    def runner(ctx):
        ctx.progress(DownloadProgress(status=DownloadStatus.DOWNLOADING, message="m", downloaded_bytes=3_000_000,
                                      size_unknown=True, size_note="LM Studio reports no progress"))
        gate.wait(5)
        return {"ok": True, "status": "completed"}

    job = reg.start(kind="download", key="k", runner=runner)
    assert _wait(lambda: reg.get(job["job_id"])["bytes_done"] == 3_000_000)
    snap = reg.get(job["job_id"])
    assert snap["size_unknown"] is True and snap["bytes_total"] is None and snap["percent"] is None
    assert "3.0 MB so far" in snap["message"]
    gate.set()
    reg.wait(job["job_id"], 5)


def test_progress_events_are_never_capped_and_are_persisted(host, tmp_path):
    reg = host_jobs.HostJobRegistry(persist_dir=tmp_path / "jobs")

    def runner(ctx):
        for i in range(1, 301):
            ctx.progress(DownloadProgress(status=DownloadStatus.DOWNLOADING, message="m", downloaded_bytes=i, total_bytes=300))
        return {"ok": True, "status": "completed"}

    job = reg.start(kind="download", key="k", runner=runner, run_inline=True)
    events = reg.events(job["job_id"])
    assert len(events) >= 300, "ADR-0026: progress is never rate-limited by count"
    lines = (tmp_path / "jobs" / f"{job['job_id']}.events.jsonl").read_text().splitlines()
    assert len(lines) == len(events) and json.loads(lines[-1])["state"] == "done"


def test_a_listener_sees_every_change(host):
    reg = host_jobs.HostJobRegistry()
    seen = []
    remove = reg.add_listener(lambda snap: seen.append((snap["state"], snap["bytes_done"])))

    def runner(ctx):
        ctx.progress({"message": "m", "downloaded_bytes": 5, "total_bytes": 10, "status": "downloading"})
        return {"ok": True, "status": "completed"}

    reg.start(kind="download", key="k", runner=runner, run_inline=True)
    remove()
    assert ("downloading", 5) in seen and seen[-1][0] == "done"


# ---------------------------------------------------------------------------
# Ollama: /api/pull per-layer NDJSON -> one artifact-wide progress
# ---------------------------------------------------------------------------

# The shape `ollama pull` streams (0.20.x): manifest, then each layer with its
# digest/total/completed (layers interleave), verify, write, success.
_OLLAMA_PULL = [
    {"status": "pulling manifest"},
    {"status": "pulling 3f2b", "digest": "sha256:3f2b1c", "total": 1000},
    {"status": "pulling 3f2b", "digest": "sha256:3f2b1c", "total": 1000, "completed": 400},
    {"status": "pulling 77aa", "digest": "sha256:77aa99", "total": 200, "completed": 0},
    {"status": "pulling 3f2b", "digest": "sha256:3f2b1c", "total": 1000, "completed": 1000},
    {"status": "pulling 77aa", "digest": "sha256:77aa99", "total": 200, "completed": 200},
    {"status": "verifying sha256 digest"},
    {"status": "writing manifest"},
    {"status": "success"},
]


def test_ollama_layers_sum_into_one_bar_that_never_goes_back():
    progress = mm._OllamaPullProgress()
    updates = [u for u in (progress.feed(e) for e in _OLLAMA_PULL) if u is not None]
    phases = [u.phase for u in updates]
    assert phases[0] == "resolving" and "verifying" in phases and phases[-1] == "installing"
    byte_updates = [u for u in updates if u.downloaded_bytes is not None and u.phase == "downloading"]
    done = [u.downloaded_bytes for u in byte_updates]
    assert done == sorted(done), f"the bar went backwards across layers: {done}"
    assert byte_updates[-1].downloaded_bytes == 1200 and byte_updates[-1].total_bytes == 1200
    assert [f["name"] for f in byte_updates[-1].files] == ["layer 3f2b1c", "layer 77aa99"]
    assert byte_updates[1].current_file == "layer 3f2b1c"
    assert progress.succeeded


def test_an_ollama_stream_without_success_is_a_failure_not_a_pull(host, monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def close(self):
            pass

        def __iter__(self):
            yield b'{"status":"pulling manifest"}\n'
            yield b'{"status":"pulling 3f","digest":"sha256:3f","total":10,"completed":4}\n'

    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))
    monkeypatch.setattr(mm.urllib.request, "urlopen", lambda *a, **k: _Resp())
    out = mm.download("ollama", "tiny:1b")
    assert out.ok is False and "ended without `success`" in out.message


def test_ollama_job_reports_bytes_through_the_contract_and_cancel_closes_a_quiet_stream(host, monkeypatch):
    with FakeOllama([]) as fake:
        # A QUIET stream: 3 s between lines. Only closing the socket on cancel
        # can stop it within a second; a flag checked per line cannot.
        fake.pull_chunks = 5
        fake.pull_delay = 3.0
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        reg = host_jobs.HostJobRegistry(tick_s=0.1)
        job = host_jobs.start_download_job("ollama", "qwen3:8b", registry=reg)
        jid = job["job_id"]
        assert _wait(lambda: (reg.get(jid)["bytes_done"] or 0) > 0)
        snap = reg.get(jid)
        assert snap["state"] == "downloading" and snap["bytes_total"] == 1000
        assert snap["files"] and snap["files"][0]["name"] == "layer 3f2b"
        t0 = time.time()
        reg.cancel(jid)
        done = reg.wait(jid, 10)
        took = time.time() - t0
    assert done["state"] == "cancelled" and done["status"] == "cancelled"
    assert took < 1.5, f"cancel took {took:.2f}s"


# ---------------------------------------------------------------------------
# LM Studio: `lms get` draws a \r progress bar; parse it
# ---------------------------------------------------------------------------


def _lms_frame(ratio: float, done: str, total: str, speed: str, eta: str) -> str:
    """One redraw exactly as `lms` ProgressBar.refresh() writes it (lms 0.4.x bundle)."""

    blocks = "█" * int(ratio * 22)
    text = f"{done.rjust(9)} / {total.rjust(9)} | {speed.rjust(9)}/s | ETA {eta}"
    return f"\x1b[?25l\x1b[s\r⠋ [{blocks}] {ratio * 100:.2f}% | {text}          \x1b[u"


def test_the_lms_bar_is_parsed_into_bytes_total_and_percent():
    parser = mm._LmsGetProgress()
    frame = mm._ANSI_RE.sub("", _lms_frame(0.2563, "1.23 GB", "4.80 GB", "38.00 MB", "01:34")).strip()
    update = parser.feed(frame)
    assert update is not None and update.phase == "downloading"
    assert update.downloaded_bytes == 1_230_000_000 and update.total_bytes == 4_800_000_000
    assert update.percent == pytest.approx(25.63)
    assert parser.feed("Searching for models with the term qwen") is None
    assert parser.line("Finalizing download...").phase == "installing"


def _install_fake_lms(bin_dir: Path, marker: Path, frames: int = 40, delay: float = 0.05) -> Path:
    """A `lms get` that behaves like the real one: bar on stdout, SIGINT asks Y/N."""

    script = bin_dir / "lms"
    script.write_text(
        f"""#!{sys.executable}
import signal, sys, time
frames = {frames}
def on_int(*_):
    sys.stdout.write("\\nContinue to download in the background? (Y/N): "); sys.stdout.flush()
    answer = sys.stdin.readline().strip().upper()
    open({str(marker)!r}, "w").write(answer)
    if answer == "N":
        print("W Download canceled.", flush=True); sys.exit(1)
    print("I Download will continue in the background.", flush=True); sys.exit(1)
signal.signal(signal.SIGINT, on_int)
print("Searching for models with the term qwen/qwen3.5-9b@4bit", flush=True)
total = 4.8
for i in range(1, frames + 1):
    r = i / frames
    blocks = chr(0x2588) * int(r * 22)
    done = f"{{total * r:.2f}} GB"
    sys.stdout.write(f"\\x1b[?25l\\x1b[s\\r\\u280b [{{blocks}}] {{r * 100:.2f}}% | {{done:>9}} /   4.80 GB |  38.00 MB/s | ETA 00:10          \\x1b[u")
    sys.stdout.flush()
    time.sleep({delay})
print("\\nFinalizing download...", flush=True)
print("Download completed.", flush=True)
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_lms_get_job_reports_real_bytes(host, monkeypatch, tmp_path):
    lms = _install_fake_lms(tmp_path / "fakebin", tmp_path / "answer", frames=30, delay=0.03)
    monkeypatch.setattr(mm, "_lms_cli", lambda: str(lms))
    answers = iter([mm.PRESENCE_ABSENT])
    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, next(answers, mm.PRESENCE_INSTALLED)))
    reg = host_jobs.HostJobRegistry(tick_s=0.05)
    done = host_jobs.start_download_job("lmstudio", "qwen/qwen3.5-9b@4bit", registry=reg, run_inline=True)
    assert done["state"] == "done", done
    events = reg.events(done["job_id"])
    moving = [e for e in events if e["state"] == "downloading" and e["bytes_total"] == 4_800_000_000]
    assert len(moving) >= 5, events
    assert moving[0]["bytes_done"] < moving[-1]["bytes_done"]
    assert "installing" in [e["state"] for e in events], "Finalizing download... is the install phase"


def test_cancelling_lms_get_answers_its_prompt_so_lm_studio_stops_too(host, monkeypatch, tmp_path):
    marker = tmp_path / "answer"
    lms = _install_fake_lms(tmp_path / "fakebin", marker, frames=2000, delay=0.05)
    monkeypatch.setattr(mm, "_lms_cli", lambda: str(lms))
    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))
    reg = host_jobs.HostJobRegistry(tick_s=0.05)
    job = host_jobs.start_download_job("lmstudio", "qwen/qwen3.5-9b@4bit", registry=reg)
    assert _wait(lambda: (reg.get(job["job_id"])["bytes_done"] or 0) > 0)
    t0 = time.time()
    reg.cancel(job["job_id"])
    done = reg.wait(job["job_id"], 10)
    assert done["state"] == "cancelled"
    assert time.time() - t0 < 2.0
    assert marker.read_text() == "N", "the cancel must answer lms's 'continue in background?' with N"


def test_lmstudio_without_a_bar_reports_bytes_on_disk(host, monkeypatch, tmp_path):
    """No bar parsed -> the honest number is what lands in LM Studio's library."""

    root = tmp_path / "lms-models"
    root.mkdir()
    monkeypatch.setattr(mm, "_lmstudio_models_root", lambda: root)
    seen = []
    watch = mm._LmStudioDiskWatch(mm._LmsGetProgress(), seen.append, expected_bytes=None)
    (root / "pub").mkdir()
    (root / "pub" / "model.gguf").write_bytes(b"x" * 2_000_000)
    thread = threading.Thread(target=watch.run, daemon=True)
    thread.start()
    assert _wait(lambda: bool(seen), timeout=5)
    watch.stop.set()
    assert seen[-1].downloaded_bytes == 2_000_000
    assert seen[-1].message == "LM Studio reports no progress; 2.0 MB on disk so far"
    assert seen[-1].size_unknown is True


# ---------------------------------------------------------------------------
# Supertonic: per-file streaming, stall-visible, cancel removes the partial
# ---------------------------------------------------------------------------


class _SlowHub:
    """A fake HF `resolve` endpoint: HEAD sizes, GET bodies, optional pause."""

    def __init__(self, files, *, chunk=4096, delay=0.0, pause_after=None, pause_s=0.0):
        self.files = files
        self.chunk, self.delay = chunk, delay
        self.pause_after, self.pause_s = pause_after, pause_s
        hub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _name(self):
                return self.path.split("/resolve/", 1)[1].split("/", 1)[1]

            def do_HEAD(self):
                body = hub.files.get(self._name())
                self.send_response(200 if body is not None else 404)
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()

            def do_GET(self):
                body = hub.files.get(self._name())
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                sent = 0
                paused = False
                try:
                    while sent < len(body):
                        self.wfile.write(body[sent: sent + hub.chunk])
                        self.wfile.flush()
                        sent += hub.chunk
                        if hub.pause_after is not None and not paused and sent >= hub.pause_after:
                            paused = True
                            time.sleep(hub.pause_s)
                        if hub.delay:
                            time.sleep(hub.delay)
                except Exception:
                    return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()


@pytest.fixture
def fake_supertonic(host, monkeypatch, tmp_path):
    """abstractvoice's Supertonic runtime, pointed at a scratch cache and 3 files."""

    runtime = pytest.importorskip("abstractvoice.supertonic.runtime")
    cache = tmp_path / "supertonic"
    monkeypatch.setenv("ABSTRACTVOICE_SUPERTONIC_CACHE_DIR", str(cache))
    files = [Path("onnx/a.onnx"), Path("onnx/b.onnx"), Path("voice_styles/F1.json")]
    monkeypatch.setattr(runtime, "_REQUIRED_FILES", files)
    return SimpleNamespace(runtime=runtime, cache=cache, files=files)


def test_supertonic_streams_per_file_bytes_from_the_first_second(fake_supertonic, monkeypatch):
    bodies = {"onnx/a.onnx": b"a" * 300_000, "onnx/b.onnx": b"b" * 500_000, "voice_styles/F1.json": b"{}" * 10}
    with _SlowHub(bodies, chunk=50_000, delay=0.01) as hub:
        monkeypatch.setenv("HF_ENDPOINT", hub.url)
        reg = host_jobs.HostJobRegistry(tick_s=0.05)
        done = host_jobs.start_download_job("supertonic", "supertonic-3", registry=reg, run_inline=True)
    assert done["state"] == "done", done
    assert done["bytes_total"] == 800_020 and done["bytes_done"] == 800_020
    events = reg.events(done["job_id"])
    first = next(e for e in events if e["state"] == "downloading")
    assert first["bytes_done"] == 0 and first["bytes_total"] == 800_020, "the total is known before the first byte"
    assert any(e["current_file"] == "onnx/b.onnx" and 0 < e["bytes_done"] < 800_020 for e in events)
    assert {f["name"]: f["state"] for f in done["files"]} == {"onnx/a.onnx": "done", "onnx/b.onnx": "done", "voice_styles/F1.json": "done"}
    assert fake_supertonic.runtime.is_supertonic_cached(fake_supertonic.cache)


def test_supertonic_cancel_stops_within_a_second_and_leaves_no_partial(fake_supertonic, monkeypatch):
    bodies = {"onnx/a.onnx": b"a" * 50_000, "onnx/b.onnx": b"b" * 5_000_000, "voice_styles/F1.json": b"{}"}
    with _SlowHub(bodies, chunk=10_000, delay=0.02) as hub:
        monkeypatch.setenv("HF_ENDPOINT", hub.url)
        reg = host_jobs.HostJobRegistry(tick_s=0.05)
        job = host_jobs.start_download_job("supertonic", "supertonic-3", registry=reg)
        assert _wait(lambda: (reg.get(job["job_id"])["bytes_done"] or 0) > 200_000)
        t0 = time.time()
        reg.cancel(job["job_id"])
        done = reg.wait(job["job_id"], 10)
        took = time.time() - t0
    assert done["state"] == "cancelled" and took < 1.0, took
    b = fake_supertonic.cache / "onnx"
    assert not (b / "b.onnx").exists()
    assert not list(b.glob(".*abstractcore-partial")), "the partial file must be removed"
    assert (b / "a.onnx").exists(), "files already fetched are kept"
    assert mm.probe("supertonic", "supertonic-3").status == mm.PRESENCE_ABSENT


def test_supertonic_quiet_server_is_reported_stalled_then_finishes(fake_supertonic, monkeypatch):
    bodies = {"onnx/a.onnx": b"a" * 200_000, "onnx/b.onnx": b"b" * 10, "voice_styles/F1.json": b"{}"}
    with _SlowHub(bodies, chunk=20_000, pause_after=60_000, pause_s=1.2) as hub:
        monkeypatch.setenv("HF_ENDPOINT", hub.url)
        reg = host_jobs.HostJobRegistry(tick_s=0.05, stall_after_s=0.5)
        done = host_jobs.start_download_job("supertonic", "supertonic-3", registry=reg, run_inline=True)
    assert done["state"] == "done"
    states = [t["state"] for t in done["transitions"]]
    assert "stalled" in states and states[states.index("stalled") + 1] == "downloading", states


# ---------------------------------------------------------------------------
# Hugging Face: per-file bytes from blobs/, marker, orphan temp files
# ---------------------------------------------------------------------------


def test_hf_watcher_reads_per_file_bytes_across_hub_temp_names(tmp_path):
    repo = tmp_path / "models--org--repo"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    plan = [{"name": "config.json", "size": 10, "etag": "e1"}, {"name": "model.safetensors", "size": 1000, "etag": "e2"},
            {"name": "tokenizer.json", "size": 50, "etag": "e3"}]
    (blobs / "e1").write_bytes(b"x" * 10)                          # whole
    (blobs / "e2.1a2b3c4d.incomplete").write_bytes(b"x" * 400)     # huggingface_hub 1.x temp name
    (blobs / "e3.incomplete").write_bytes(b"x" * 5)                # older hubs' resumable name
    seen = []
    watcher = mm._HfBlobWatcher(repo, plan, seen.append, "org/repo")
    watcher.push()
    update = seen[-1]
    assert update.downloaded_bytes == 415 and update.total_bytes == 1060
    assert {f["name"]: f["state"] for f in update.files} == {"config.json": "done", "model.safetensors": "downloading", "tokenizer.json": "downloading"}
    assert update.current_file == "model.safetensors"
    # A killed 1.x child leaves its uuid temp file: removed; the resumable one stays.
    assert mm._hf_drop_orphan_partials(blobs, plan) == 1
    assert not (blobs / "e2.1a2b3c4d.incomplete").exists() and (blobs / "e3.incomplete").exists()


def test_a_download_marker_keeps_a_between_files_cancel_from_reading_installed(host):
    hf = host["hf"]
    repo = hf / "models--org--repo"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / "rev1").mkdir(parents=True)
    (repo / "blobs" / "e1").write_bytes(b"x" * 10)
    os.symlink("../../blobs/e1", repo / "snapshots" / "rev1" / "model-00001-of-00002.safetensors")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev1")
    plan = [{"name": "model-00001-of-00002.safetensors", "size": 10, "etag": "e1"},
            {"name": "model-00002-of-00002.safetensors", "size": 99, "etag": "e2"}]
    marker = mm._hf_write_marker(repo, "org/repo", "rev1", plan)
    # No `.incomplete` anywhere -- the cancel landed between two shards, and
    # the snapshot holds one WHOLE weight file: without the marker it reads
    # "installed".
    miss = mm.probe("huggingface", "org/repo")
    assert miss.status == mm.PRESENCE_ABSENT and "partially downloaded" in (miss.detail or ""), miss
    # Someone else finished the job: a marker whose files are all whole is stale.
    (repo / "blobs" / "e2").write_bytes(b"y" * 99)
    assert mm._hf_marker_unfinished(repo) == 0
    assert mm.probe("huggingface", "org/repo").status == mm.PRESENCE_INSTALLED
    marker.unlink()


def test_hf_job_cancel_kills_the_child_and_cleans_its_temp_files(host, monkeypatch, tmp_path):
    """The child process is the real transfer path; here it is a stand-in that
    writes a growing 1.x-style temp file, like `snapshot_download` does."""

    hf = host["hf"]
    plan = [{"name": "model.safetensors", "size": 50_000_000, "etag": "e9"}]
    monkeypatch.setattr(mm, "_hf_file_plan", lambda repo, patterns, token: (plan, "rev1", ""))
    monkeypatch.setattr(mm, "_hf_download_cache_dir", lambda: hf)
    child = (
        "import sys, time, pathlib\n"
        f"p = pathlib.Path({str(hf / 'models--org--big' / 'blobs')!r}); p.mkdir(parents=True, exist_ok=True)\n"
        "f = open(p / 'e9.0badc0de.incomplete', 'wb')\n"
        "while True:\n    f.write(b'x' * 100000); f.flush(); time.sleep(0.02)\n"
    )
    monkeypatch.setattr(mm, "_HF_CHILD", child)
    fake_hub = SimpleNamespace(snapshot_download=lambda **kw: pytest.fail("a job must not download in-process"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    reg = host_jobs.HostJobRegistry(tick_s=0.05)
    job = host_jobs.start_download_job("huggingface", "org/big", registry=reg)
    assert _wait(lambda: (reg.get(job["job_id"])["bytes_done"] or 0) > 500_000)
    snap = reg.get(job["job_id"])
    assert snap["current_file"] == "model.safetensors" and snap["bytes_total"] == 50_000_000
    t0 = time.time()
    reg.cancel(job["job_id"])
    done = reg.wait(job["job_id"], 10)
    assert done["state"] == "cancelled" and time.time() - t0 < 1.5
    assert not list((hf / "models--org--big" / "blobs").glob("*.incomplete")), "orphan temp file removed"
    assert (hf / "models--org--big" / mm._HF_MARKER).exists(), "the marker stays: the repo is NOT installed"
    assert mm.probe("huggingface", "org/big").status != mm.PRESENCE_INSTALLED
