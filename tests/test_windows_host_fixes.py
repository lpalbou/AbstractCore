"""Windows host behavior: console output, process trees and LM Studio's model folder."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from abstractcore.utils.stdio import safe_stdio


def test_safe_stdio_replaces_what_a_cp1252_stream_cannot_encode(monkeypatch) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    safe_stdio()
    print("\U0001f4cb status → ok │", file=sys.stdout)
    sys.stdout.flush()
    assert raw.getvalue().decode("cp1252").strip() == "? status ? ok ?"


def test_safe_stdio_leaves_a_utf8_stream_as_it_is(monkeypatch) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    safe_stdio()
    assert sys.stdout.errors == "strict"
    print("→", file=sys.stdout)
    sys.stdout.flush()
    assert raw.getvalue().decode("utf-8").strip() == "→"


def test_safe_stdio_tolerates_streams_without_reconfigure(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", None)
    safe_stdio()


@pytest.mark.parametrize(
    "module",
    ["abstractcore.config.main", "abstractcore.utils.cli", "abstractcore.endpoint.app", "abstractcore.apps.summarizer",
     "abstractcore.apps.extractor", "abstractcore.apps.judge", "abstractcore.apps.intent", "abstractcore.apps.deepsearch"],
)
def test_every_console_entry_point_makes_stdio_safe_first(module) -> None:
    import importlib.util
    import inspect

    spec = importlib.util.find_spec(module)
    source = Path(spec.origin).read_text(encoding="utf-8")
    body = source[source.index("\ndef main("):]
    assert "safe_stdio()" in body.split("\ndef ", 2)[1], f"{module}.main does not call safe_stdio()"


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


@pytest.mark.skipif(os.name != "nt", reason="Windows process tree")
def test_hard_kill_tree_ends_a_windows_workers_children() -> None:
    from abstractcore.tools.process_tree import hard_kill_tree

    code = ("import subprocess, sys, time; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print(p.pid, flush=True); time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    grandchild = int(proc.stdout.readline())
    assert _alive(proc.pid) and _alive(grandchild)
    hard_kill_tree(proc)
    deadline = time.time() + 10
    while time.time() < deadline and (_alive(proc.pid) or _alive(grandchild)):
        time.sleep(0.2)
    assert not _alive(proc.pid) and not _alive(grandchild)


@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
def test_lm_studio_models_in_the_home_folder_are_found_on_every_os(tmp_path, monkeypatch, system) -> None:
    from abstractcore.capabilities import vision_catalog

    models = tmp_path / ".lmstudio" / "models"
    models.mkdir(parents=True)
    monkeypatch.setattr(vision_catalog.platform, "system", lambda: system)
    monkeypatch.setattr(vision_catalog.Path, "home", classmethod(lambda cls: tmp_path))
    for key in ("LMSTUDIO_MODELS_DIR", "LMSTUDIO_MODEL_DIR", "LM_STUDIO_MODELS_DIR", "LOCALAPPDATA", "APPDATA"):
        monkeypatch.delenv(key, raising=False)
    assert models in vision_catalog._default_lmstudio_model_dirs()


def _copy_cache(tmp_path: Path) -> tuple[Path, list]:
    """A Hugging Face cache without links (Windows without Developer Mode): huggingface_hub moved
    each finished file from blobs/ into snapshots/<rev>/ and left no blob behind."""
    repo = tmp_path / "models--org--tiny"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    (snap / "config.json").write_bytes(b"{}" * 10)
    (snap / "model.safetensors").write_bytes(b"\0" * 300)
    plan = [{"name": "config.json", "size": 20, "etag": "e1"}, {"name": "model.safetensors", "size": 300, "etag": "e2"}]
    return repo, plan


def test_a_download_into_a_cache_without_links_is_whole(tmp_path) -> None:
    from abstractcore.config import model_materializer as mm

    repo, plan = _copy_cache(tmp_path)
    watcher = mm._HfBlobWatcher(repo, plan, lambda *_a, **_k: None, "org/tiny")
    assert watcher.missing() == []
    assert watcher.bytes_done() == 320


def test_a_snapshot_copy_of_the_wrong_size_is_not_whole(tmp_path) -> None:
    from abstractcore.config import model_materializer as mm

    repo, plan = _copy_cache(tmp_path)
    (repo / "snapshots" / "abc123" / "model.safetensors").write_bytes(b"\0" * 10)
    watcher = mm._HfBlobWatcher(repo, plan, lambda *_a, **_k: None, "org/tiny")
    assert watcher.missing() == ["model.safetensors"]


def test_the_download_marker_counts_snapshot_copies_as_whole(tmp_path) -> None:
    import json
    from abstractcore.config import model_materializer as mm

    repo, plan = _copy_cache(tmp_path)
    (repo / mm._HF_MARKER).write_text(json.dumps({"files": plan}), encoding="utf-8")
    assert mm._hf_marker_unfinished(repo) == 0


def test_a_blob_layout_is_read_as_before(tmp_path) -> None:
    from abstractcore.config import model_materializer as mm

    repo = tmp_path / "models--org--tiny"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / "abc123").mkdir(parents=True)
    (repo / "blobs" / "e1").write_bytes(b"x" * 20)
    plan = [{"name": "config.json", "size": 20, "etag": "e1"}, {"name": "model.safetensors", "size": 300, "etag": "e2"}]
    watcher = mm._HfBlobWatcher(repo, plan, lambda *_a, **_k: None, "org/tiny")
    assert watcher.missing() == ["model.safetensors"]


def _owner_watch_header() -> str:
    from abstractcore.config.model_materializer import _HF_CHILD

    head, sep, _ = _HF_CHILD.partition("from huggingface_hub import snapshot_download")
    assert sep, "the download child no longer has the expected shape"
    return head


def test_the_download_child_keeps_running_while_its_owner_lives() -> None:
    # On Windows sys.executable in a venv is a launcher: the child's parent pid is the
    # launcher, not the owner. The child must not take that for a dead owner.
    code = _owner_watch_header() + "time.sleep(3.5)\nprint('alive', flush=True)\n"
    out = subprocess.run(
        [sys.executable, "-c", code, "{}", str(os.getpid())],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, (out.returncode, out.stderr)
    assert out.stdout.strip() == "alive"


def test_the_download_child_stops_when_its_owner_exits() -> None:
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
    code = _owner_watch_header() + "time.sleep(30)\nprint('outlived', flush=True)\n"
    started = time.monotonic()
    out = subprocess.run(
        [sys.executable, "-c", code, "{}", str(owner.pid)],
        capture_output=True, text=True, timeout=60,
    )
    owner.wait(timeout=30)
    assert out.returncode == 75, (out.returncode, out.stdout, out.stderr)
    assert time.monotonic() - started < 20
