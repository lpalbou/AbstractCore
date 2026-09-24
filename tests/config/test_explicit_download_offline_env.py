"""An explicit Hugging Face download always reaches the Hub.

`offline_first` means "never download on-demand while LOADING a model". It
never makes the process offline, so an HF offline flag written into
`os.environ` in-process after start (a provider import, a library's load-time
override) must not reach the download child. Only a flag the OPERATOR set
before start stops a download -- explicitly, with the variable named.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

from abstractcore.config import host_jobs
from abstractcore.config import manager
from abstractcore.config import model_materializer as mm
from tests.models_engines_fakes import isolate_host

_NAMES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


def _operator(monkeypatch, **values):
    snapshot = {name: values.get(name) for name in _NAMES}
    monkeypatch.setattr(manager, "_OPERATOR_HF_OFFLINE_ENV", snapshot)


def test_in_process_flags_are_dropped_and_reported(monkeypatch):
    _operator(monkeypatch)
    env, dropped = manager.explicit_download_hf_env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "KEEP": "me"})
    assert env == {"KEEP": "me"}
    assert dropped == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_operator_flags_are_kept_verbatim(monkeypatch):
    _operator(monkeypatch, HF_HUB_OFFLINE="0", HF_DATASETS_OFFLINE="1")
    env, dropped = manager.explicit_download_hf_env({"HF_HUB_OFFLINE": "1"})
    assert env == {"HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "1"}
    assert dropped == {"HF_HUB_OFFLINE": "1"}
    assert manager.operator_forces_hf_offline() is None  # "0" is not offline; datasets flag is not the Hub


@pytest.mark.parametrize("name,value", [("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "yes"), ("HF_HUB_OFFLINE", "TRUE")])
def test_operator_forcing_matches_huggingface_hub_truthiness(monkeypatch, name, value):
    _operator(monkeypatch, **{name: value})
    assert manager.operator_forces_hf_offline() == name


# The stand-in for `snapshot_download`: refuses exactly like huggingface_hub when
# the environment it was started with says offline, downloads otherwise.
_CHILD = r"""
import os, sys, pathlib, json
blobs = pathlib.Path(%(blobs)r)
started = pathlib.Path(%(started)r)
started.write_text(json.dumps({k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}))
if any((os.environ.get(k) or "").upper() in ("1", "ON", "YES", "TRUE") for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
    print("huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach https://huggingface.co: offline mode is enabled.")
    sys.exit(1)
blobs.mkdir(parents=True, exist_ok=True)
(blobs / "e1").write_bytes(b"x" * 4096)
print("ABSTRACTCORE_RESOLVED=" + str(blobs.parent / "snapshots" / "rev1"), flush=True)
"""


def _fake_hub_job(host, monkeypatch, tmp_path):
    hf = host["hf"]
    plan = [{"name": "model.safetensors", "size": 4096, "etag": "e1"}]
    monkeypatch.setattr(mm, "_hf_file_plan", lambda repo, patterns, token: (plan, "rev1", ""))
    monkeypatch.setattr(mm, "_hf_download_cache_dir", lambda: hf)
    started = tmp_path / "child-started.json"
    child = _CHILD % {"blobs": str(hf / "models--org--small" / "blobs"), "started": str(started)}
    monkeypatch.setattr(mm, "_HF_CHILD", child)
    fake_hub = SimpleNamespace(snapshot_download=lambda **kw: pytest.fail("a job must not download in-process"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    reg = host_jobs.HostJobRegistry(tick_s=0.05)
    job = host_jobs.start_download_job("huggingface", "org/small", registry=reg)
    return reg.wait(job["job_id"], 20), started


def test_a_download_after_an_in_process_offline_write_still_downloads(host, monkeypatch, tmp_path):
    """The defect: something in the process set HF_HUB_OFFLINE=1 after start
    (the MLX load did, the HF provider's import does); the job must not care."""

    _operator(monkeypatch)
    for name in _NAMES:
        monkeypatch.setenv(name, "1")
    done, started = _fake_hub_job(host, monkeypatch, tmp_path)
    assert done["status"] == "completed", done.get("error") or done.get("log_tail")
    import json

    assert json.loads(started.read_text()) == {name: None for name in _NAMES}, "the child saw an offline flag"
    notes = [line for line in done["log_tail"] if line.startswith("explicit download:")]
    assert notes and "HF_HUB_OFFLINE=1" in notes[0] and "not passed" in notes[0], done["log_tail"]


def test_a_clean_process_logs_that_no_offline_flag_applies(host, monkeypatch, tmp_path):
    _operator(monkeypatch)
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    done, _ = _fake_hub_job(host, monkeypatch, tmp_path)
    assert done["status"] == "completed", done.get("error")
    assert "explicit download: no Hub offline flag in the download environment" in done["log_tail"]


def test_an_operator_offline_flag_refuses_the_download_by_name(host, monkeypatch, tmp_path):
    _operator(monkeypatch, HF_HUB_OFFLINE="1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    t0 = time.time()
    done, started = _fake_hub_job(host, monkeypatch, tmp_path)
    assert done["status"] == "failed"
    assert "HF_HUB_OFFLINE=1 was set in the environment before this process started" in (done.get("error") or "")
    assert not started.exists(), "no download process is started when the operator said offline"
    assert time.time() - t0 < 10


def test_a_detached_job_child_gets_the_operator_flags_not_in_process_ones(host, monkeypatch):
    """A detached job is a fresh process: it would read an inherited in-process
    flag as the operator's and refuse. It must get the operator's values."""

    _operator(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    seen = {}

    class _Proc:
        pid = 4242

    def _popen(argv, **kwargs):
        seen["env"] = kwargs.get("env")
        return _Proc()

    monkeypatch.setattr(host_jobs.subprocess, "Popen", _popen)
    host_jobs.spawn_detached({"kind": "download", "provider": "huggingface", "artifact": "org/small"})
    assert seen["env"] is not None and "HF_HUB_OFFLINE" not in seen["env"]
