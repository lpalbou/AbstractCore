"""Offline-first is scoped to the MLX LOAD call, never to the process.

`offline_first` (default True) means "loading a model never downloads it
on-demand". The MLX provider used to express that by writing HF_HUB_OFFLINE /
TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE = 1 into `os.environ` -- inert for
the loader itself (huggingface_hub snapshots the flag at its import, which
`from mlx_lm import load` has already done) and fatal for everything after it:
every child process inherits `os.environ`, so each later explicit download job
died with OfflineModeIsEnabled.

These tests pin both halves:
  * a load leaves the three variables exactly as the operator set them;
  * a load of a model that is not (fully) in the cache still never reaches the
    network -- the original intent, now enforced by cache-only resolution.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.models_engines_fakes import isolate_host

pytest.importorskip("mlx_lm")

_NAMES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


@pytest.fixture
def offline_first(tmp_path, monkeypatch):
    """An isolated host, offline_first ON, and no HF offline variable set."""

    host = isolate_host(tmp_path, monkeypatch)
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    import abstractcore.config.manager as manager

    stub = SimpleNamespace(is_offline_first=lambda: True, get_config_file=lambda: None)
    monkeypatch.setattr(manager, "get_config_manager", lambda *a, **k: stub)
    return host


@pytest.fixture
def no_network(monkeypatch):
    """Record (and refuse) every outbound connection and every hub fetch."""

    attempts: list = []

    def _refuse(kind):
        def _fn(*args, **kwargs):
            attempts.append((kind, args[:2]))
            raise OSError(f"network refused by test ({kind})")

        return _fn

    monkeypatch.setattr(socket.socket, "connect", _refuse("socket.connect"))
    monkeypatch.setattr(socket, "create_connection", _refuse("socket.create_connection"))
    import huggingface_hub
    import mlx_lm
    import mlx_lm.utils

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _refuse("huggingface_hub.snapshot_download"))
    monkeypatch.setattr(mlx_lm.utils, "snapshot_download", _refuse("mlx_lm.utils.snapshot_download"))
    monkeypatch.setattr(mlx_lm, "load", _refuse("mlx_lm.load"))
    return attempts


def _fake_mlx_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"")
    return root


def _load_with_stubbed_mlx_lm(monkeypatch, model: str) -> list:
    import mlx_lm

    loaded: list = []
    monkeypatch.setattr(mlx_lm, "load", lambda path, *a, **k: (loaded.append(path) or (object(), object())))
    from abstractcore.providers.mlx_provider import MLXProvider

    MLXProvider(model=model)
    return loaded


def test_a_load_leaves_no_hf_offline_variable_in_the_process(offline_first, monkeypatch, tmp_path):
    model_dir = _fake_mlx_dir(tmp_path / "models" / "tiny")
    loaded = _load_with_stubbed_mlx_lm(monkeypatch, str(model_dir))
    assert loaded == [str(model_dir)], "the loader was handed the resolved local directory"
    leaked = {name: os.environ[name] for name in _NAMES if name in os.environ}
    assert leaked == {}, f"an MLX load wrote process-wide offline flags: {leaked}"


def test_a_load_keeps_exactly_what_the_operator_set(offline_first, monkeypatch, tmp_path):
    # The operator chose HF_HUB_OFFLINE=0 before start: it must survive as-is,
    # and nothing may be ADDED next to it.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    model_dir = _fake_mlx_dir(tmp_path / "models" / "tiny")
    _load_with_stubbed_mlx_lm(monkeypatch, str(model_dir))
    assert {name: os.environ.get(name) for name in _NAMES} == {
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": None,
        "HF_DATASETS_OFFLINE": None,
    }


def test_a_later_download_child_would_not_inherit_an_offline_flag(offline_first, monkeypatch, tmp_path):
    """The defect end to end, in-process: load, then build a download child's env."""

    model_dir = _fake_mlx_dir(tmp_path / "models" / "tiny")
    _load_with_stubbed_mlx_lm(monkeypatch, str(model_dir))
    child_env = dict(os.environ)  # what `subprocess.Popen(env=None)` hands a child
    assert not any(name in child_env for name in _NAMES)


@pytest.mark.parametrize("cached", ["absent", "config_only"])
def test_loading_an_uncached_model_never_reaches_the_network(offline_first, no_network, cached):
    """Offline-first's real intent: a miss is ModelNotFoundError, not a download."""

    from abstractcore.exceptions import ModelNotFoundError
    from abstractcore.providers.mlx_provider import MLXProvider

    repo = "mlx-community/not-in-this-cache-4bit"
    if cached == "config_only":
        # A partial snapshot: config present, weights missing -- the case where a
        # loader handed the repo id would "helpfully" fetch the rest.
        snap = offline_first["hf"] / "models--mlx-community--not-in-this-cache-4bit" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")
        refs = snap.parent.parent / "refs"
        refs.mkdir()
        (refs / "main").write_text("abc123", encoding="utf-8")

    with pytest.raises(ModelNotFoundError):
        MLXProvider(model=repo)
    assert no_network == [], f"a load under offline_first attempted a download: {no_network}"
    assert not any(name in os.environ for name in _NAMES)
