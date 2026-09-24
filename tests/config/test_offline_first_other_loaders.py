"""Offline-first per call in the loaders outside the HF provider (mission V, 2026-09-24).

`EmbeddingManager` loaded `SentenceTransformer(model_id)` with no
`local_files_only`: it stayed offline only when the HF provider had already
written HF_HUB_OFFLINE=1 into the process at import -- a write mission U
removed. Pins:
- offline-first ON: the id resolves to its cached snapshot DIRECTORY (even
  without `refs/main`, the state a pinned download used to leave) and the load
  gets `local_files_only=True`; a bare legacy name is looked up as
  `sentence-transformers/<name>`;
- not cached: ModelNotFoundError with the plain "download it first" wording,
  and sentence-transformers is never called;
- offline-first OFF (and force_local_files_only off): the id passes through,
  with the hub cache as `cache_folder=` (mission EE: no os.environ writes).

`abstractcore --download-vision-model` (vision_config and config/main.py) and
the config wizard's "download embeddings now" are EXPLICIT downloads:
`offline_first` never applies to them, so their `from_pretrained` calls carry
no `local_files_only`; only an operator-set Hub-offline flag stops them, by name.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import abstractcore.config as core_config
from abstractcore.config import manager as cfg_manager
from abstractcore.embeddings import manager as emb_manager
from abstractcore.exceptions import ModelNotFoundError

SHA = "c" * 40


def _put_snapshot(cache: Path, repo_id: str, files: Dict[str, bytes]) -> Path:
    repo_dir = cache / ("models--" + repo_id.replace("/", "--"))
    snap = repo_dir / "snapshots" / SHA
    for name, data in files.items():
        blob = repo_dir / "blobs" / hashlib.sha256(data).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        link = snap / name
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, link.parent), link)
    return snap  # deliberately NO refs/main


ST_FILES = {"config.json": b"{}", "modules.json": b"[]", "model.safetensors": b"w" * 8}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return cache


@pytest.fixture
def st_calls(monkeypatch):
    calls: List[Dict[str, Any]] = []

    class _FakeST:
        def __init__(self, name_or_path, **kwargs):
            calls.append({"source": name_or_path, **kwargs})

    monkeypatch.setattr(emb_manager, "sentence_transformers", SimpleNamespace(SentenceTransformer=_FakeST))
    return calls


def _config(monkeypatch, *, offline_first: bool, force_local: bool) -> None:
    fake = SimpleNamespace(is_offline_first=lambda: offline_first, should_force_local_files_only=lambda: force_local)
    monkeypatch.setattr(core_config, "get_config_manager", lambda: fake)


def _manager(model: str, tmp_path: Path):
    return emb_manager.EmbeddingManager(model=model, provider="huggingface", backend="pytorch", cache_dir=tmp_path / "emb-cache")


def test_offline_first_loads_the_cached_snapshot_directory_local_only(hub, st_calls, monkeypatch, tmp_path):
    snap = _put_snapshot(hub, "org/tiny-emb", ST_FILES)
    _config(monkeypatch, offline_first=True, force_local=False)
    _manager("org/tiny-emb", tmp_path)
    assert len(st_calls) == 1
    assert st_calls[0]["source"] == str(snap)
    assert st_calls[0]["local_files_only"] is True


def test_a_bare_legacy_name_resolves_under_the_sentence_transformers_org(hub, st_calls, monkeypatch, tmp_path):
    snap = _put_snapshot(hub, "sentence-transformers/tiny-legacy", ST_FILES)
    _config(monkeypatch, offline_first=True, force_local=True)
    _manager("tiny-legacy", tmp_path)
    assert st_calls[0]["source"] == str(snap) and st_calls[0]["local_files_only"] is True


def test_an_uncached_model_fails_plainly_and_never_reaches_sentence_transformers(hub, st_calls, monkeypatch, tmp_path):
    _config(monkeypatch, offline_first=True, force_local=False)
    with pytest.raises(ModelNotFoundError) as info:
        _manager("org/not-here", tmp_path)
    message = str(info.value)
    assert "download it first" in message and "abstractcore models download huggingface org/not-here" in message
    assert st_calls == []


def test_with_offline_first_off_the_id_passes_through(hub, st_calls, monkeypatch, tmp_path):
    _config(monkeypatch, offline_first=False, force_local=False)
    _manager("org/anything", tmp_path)
    # The download location is a per-call argument (mission EE), never os.environ.
    hf_constants = pytest.importorskip(
        "huggingface_hub.constants", reason="the expected cache folder is huggingface_hub's own (not in the CI [test] extra)"
    )

    assert st_calls == [
        {"source": "org/anything", "trust_remote_code": False, "cache_folder": str(hf_constants.HF_HUB_CACHE)}
    ]


def test_the_embedding_and_provider_hints_share_one_wording():
    assert "abstractcore models download huggingface org/x" in cfg_manager.hf_download_first_hint("org/x")


# ---------------------------------------------------------------------------
# Explicit downloads: never local_files_only; operator-set offline refuses by name
# ---------------------------------------------------------------------------


def _operator_env(monkeypatch, **values):
    snapshot = {name: None for name in cfg_manager.HF_OFFLINE_ENV_NAMES}
    snapshot.update(values)
    monkeypatch.setattr(cfg_manager, "_OPERATOR_HF_OFFLINE_ENV", snapshot)


def test_operator_offline_refusal_names_the_variable(monkeypatch):
    _operator_env(monkeypatch, HF_HUB_OFFLINE="1")
    refusal = cfg_manager.operator_hf_offline_refusal("org/m")
    assert refusal and "HF_HUB_OFFLINE=1" in refusal and "org/m" in refusal
    _operator_env(monkeypatch)
    assert cfg_manager.operator_hf_offline_refusal("org/m") is None


@pytest.fixture
def fake_transformers(monkeypatch):
    transformers = pytest.importorskip("transformers")
    calls: List[Dict[str, Any]] = []

    class _Fake:
        def __init__(self, cls):
            self.cls = cls

        def from_pretrained(self, name, **kwargs):
            calls.append({"cls": self.cls, "name": name, **kwargs})
            return SimpleNamespace(save_pretrained=lambda *_a, **_k: None)

    for cls in ("AutoProcessor", "AutoModel", "GitProcessor", "GitForCausalLM"):
        monkeypatch.setattr(transformers, cls, _Fake(cls), raising=False)
    return calls


@pytest.fixture(autouse=True)
def _no_network_no_real_home(monkeypatch, tmp_path):
    """Nothing here may reach the Hub or the operator's home -- not even a
    MUTANT of the code under test (a mutation run that removed the refusal
    once downloaded 1.3 GB into ~/.abstractcore/models and rewrote the
    operator's vision config: HOME and the config dir are now scratch, and
    every outbound connect fails)."""
    import socket

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ABSTRACTCORE_CONFIG_DIR", str(tmp_path / "abstractcore-config"))
    monkeypatch.setenv("ABSTRACTFRAMEWORK_DATA_REGISTRY", str(tmp_path / "data_registry.json"))

    def _refuse(self, addr):
        raise OSError(f"test network guard: connect to {addr} refused")

    monkeypatch.setattr(socket.socket, "connect", _refuse)


def _vision_handler(tmp_path):
    saved = []
    handler = SimpleNamespace(
        config=SimpleNamespace(local_models_path=str(tmp_path / "vision-models"), strategy="disabled"),
        _save_config=lambda cfg: saved.append(cfg),
    )
    return handler, saved


def test_vision_download_reaches_the_hub_without_local_files_only(monkeypatch, tmp_path, fake_transformers):
    from abstractcore.config import vision_config

    _operator_env(monkeypatch)
    handler, _ = _vision_handler(tmp_path)
    vision_config.handle_download_vision_model(handler, "git-base")
    assert [c["cls"] for c in fake_transformers] == ["AutoProcessor", "AutoModel"]
    assert all("local_files_only" not in c for c in fake_transformers)


def test_vision_download_refuses_by_name_when_the_operator_set_offline(monkeypatch, tmp_path, capsys, fake_transformers):
    from abstractcore.config import vision_config

    _operator_env(monkeypatch, TRANSFORMERS_OFFLINE="1")
    handler, _ = _vision_handler(tmp_path)
    vision_config.handle_download_vision_model(handler, "git-base")
    assert fake_transformers == []
    assert "TRANSFORMERS_OFFLINE=1" in capsys.readouterr().out


@pytest.fixture
def config_main(monkeypatch):
    from abstractcore.config import main as config_main

    configured: List[tuple] = []
    fake_cm = SimpleNamespace(set_vision_provider=lambda provider, model: configured.append((provider, model)) or True)
    monkeypatch.setattr(config_main, "get_config_manager", lambda: fake_cm)
    monkeypatch.setattr(config_main, "CONFIG_AVAILABLE", True, raising=False)
    config_main._configured = configured  # type: ignore[attr-defined]
    return config_main


def test_config_main_vision_download_reaches_the_hub_without_local_files_only(monkeypatch, capsys, fake_transformers, config_main):
    _operator_env(monkeypatch)
    assert config_main.download_vision_model("git-base") is True
    assert [c["cls"] for c in fake_transformers] == ["GitProcessor", "GitForCausalLM"]
    assert all("local_files_only" not in c for c in fake_transformers)


def test_config_main_vision_download_refuses_by_name_when_the_operator_set_offline(monkeypatch, capsys, fake_transformers, config_main):
    _operator_env(monkeypatch, HF_HUB_OFFLINE="yes")
    assert config_main.download_vision_model("git-base") is False
    assert "HF_HUB_OFFLINE=yes" in capsys.readouterr().out
    assert fake_transformers == [] and config_main._configured == []
