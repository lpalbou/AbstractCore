"""Embeddings manager: safe persistent caches, no process-wide env writes,
per-call cache locations (mission EE, 2026-09-24).

Incident: every EmbeddingManager rewrote its WHOLE in-memory cache over the
on-disk pickle at interpreter exit, last writer wins -- a test process holding
an empty cache emptied the operator's populated one.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from abstractcore.embeddings import manager as manager_module
from abstractcore.embeddings.manager import EmbeddingManager

pytestmark = pytest.mark.usefixtures("fake_embedding_snapshots")

DEFAULT_REPO = "sentence-transformers/all-MiniLM-L6-v2"


def _mock_st(dim: int = 8):
    st = MagicMock()
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim
    model.encode.side_effect = lambda text, **_k: (
        np.full((len(text), dim), 0.5) if isinstance(text, list) else np.full(dim, float(len(text)))
    )
    st.SentenceTransformer.return_value = model
    return st


def _read(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- item 2: saves


def test_save_skipped_when_nothing_was_added(tmp_path):
    """The incident path: a manager that embedded nothing must not touch the file."""
    with patch.object(manager_module, "sentence_transformers", _mock_st()):
        writer = EmbeddingManager(cache_dir=tmp_path)
        for text in ("alpha", "beta", "gamma"):
            writer.embed(text)
        writer._safe_save_persistent_cache()
        before = _read(writer.cache_file)
        assert len(before) == 3

        # A second "process" that starts, embeds nothing, and exits.
        idle = EmbeddingManager(cache_dir=tmp_path)
        idle._persistent_cache.clear()  # even an EMPTY in-memory cache...
        idle._safe_save_persistent_cache()  # ...is the atexit hook
        idle._safe_save_normalized_cache()

    assert _read(writer.cache_file) == before
    assert not writer.normalized_cache_file.exists()


def test_merge_on_save_keeps_entries_written_by_another_instance(tmp_path):
    """A (loaded while the file was empty) adds one entry; B meanwhile saved two.
    A's save must keep B's two -- last writer no longer wins."""
    with patch.object(manager_module, "sentence_transformers", _mock_st()):
        a = EmbeddingManager(cache_dir=tmp_path)  # loads nothing: file absent
        b = EmbeddingManager(cache_dir=tmp_path)
        b.embed("from-b-1")
        b.embed("from-b-2")
        b._save_persistent_cache()

        a.embed("from-a")
        a._save_persistent_cache()

    on_disk = _read(a.cache_file)
    assert {a._text_hash("from-b-1"), a._text_hash("from-b-2"), a._text_hash("from-a")} <= set(on_disk)


def test_normalized_cache_merges_too(tmp_path):
    with patch.object(manager_module, "sentence_transformers", _mock_st()):
        a = EmbeddingManager(cache_dir=tmp_path)
        b = EmbeddingManager(cache_dir=tmp_path)
        b.embed_normalized("nb")
        b._save_normalized_cache()
        a.embed_normalized("na")
        a._save_normalized_cache()
    assert len(_read(a.normalized_cache_file)) == 2


def test_never_writes_an_empty_mapping(tmp_path):
    path = tmp_path / "x_cache.pkl"
    assert manager_module._merge_save_pickle_cache(path, {}, {"k"}, label="t") is False
    assert not path.exists()


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "x_cache.pkl"
    path.write_bytes(pickle.dumps({"old": [1.0]}))
    added = {"new"}
    assert manager_module._merge_save_pickle_cache(path, {"new": [2.0]}, added, label="t") is True
    assert _read(path) == {"old": [1.0], "new": [2.0]}
    assert added == set()  # a second exit-time save has nothing to do
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")) == []


_CHILD = textwrap.dedent(
    """
    import sys
    from unittest.mock import MagicMock
    import numpy as np
    import abstractcore.embeddings.manager as m

    st = MagicMock()
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 8
    model.encode.side_effect = lambda text, **k: np.full(8, float(len(text)))
    st.SentenceTransformer.return_value = model
    m.sentence_transformers = st

    mgr = m.EmbeddingManager(cache_dir=sys.argv[1])
    if sys.argv[2:] == ["--idle-until-stdin"]:
        print("loaded", flush=True)  # the cache is read NOW (the file may still be empty)
        sys.stdin.readline()          # ...another process writes meanwhile...
    else:
        for text in sys.argv[2:]:
            mgr.embed(text)
    # no explicit save: the interpreter-exit (atexit) hook is what is under test
    """
)


def _run_child(cache_dir: Path, *texts: str) -> None:
    subprocess.run(
        [sys.executable, "-c", _CHILD, str(cache_dir), *texts],
        check=True,
        env=dict(os.environ),
        timeout=300,
    )


def test_two_processes_exit_hooks_never_lose_entries(tmp_path):
    """Real interpreter exits: a populating process, then an idle one, then another writer."""
    cache_dir = tmp_path / "emb"
    _run_child(cache_dir, "one", "two", "three")
    files = list(cache_dir.glob("*_cache.pkl"))
    cache_file = next(p for p in files if "normalized" not in p.name)
    assert len(_read(cache_file)) == 3

    _run_child(cache_dir)  # starts, embeds nothing, exits
    assert len(_read(cache_file)) == 3

    _run_child(cache_dir, "four")
    assert len(_read(cache_file)) == 4


def test_idle_process_that_loaded_early_does_not_empty_the_cache_at_exit(tmp_path):
    """The exact 2026-09-24 incident: process A loads while the cache is EMPTY and
    stays up; process B populates the cache and exits; A exits last. Before the
    fix A's exit hook wrote its empty cache over B's entries."""
    cache_dir = tmp_path / "emb"
    env = dict(os.environ)
    idle = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(cache_dir), "--idle-until-stdin"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
    )
    try:
        assert idle.stdout.readline().strip() == "loaded"
        _run_child(cache_dir, "one", "two", "three")
        cache_file = next(p for p in cache_dir.glob("*_cache.pkl") if "normalized" not in p.name)
        assert len(_read(cache_file)) == 3
    finally:
        idle.communicate("exit\n", timeout=300)
    assert idle.returncode == 0
    assert len(_read(cache_file)) == 3


# ------------------------------------------------------ item 3: no env writes


def test_constructing_a_manager_leaves_os_environ_unchanged(monkeypatch, tmp_path):
    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        monkeypatch.delenv(key, raising=False)
    before = dict(os.environ)
    with patch.object(manager_module, "sentence_transformers", _mock_st()):
        EmbeddingManager(cache_dir=tmp_path)
    assert dict(os.environ) == before


def test_offline_load_passes_the_snapshot_dir_and_local_files_only(fake_embedding_snapshots, tmp_path):
    st = _mock_st()
    with patch.object(manager_module, "sentence_transformers", st):
        EmbeddingManager(cache_dir=tmp_path, backend="pytorch")
    args, kwargs = st.SentenceTransformer.call_args
    assert args[0] == str(fake_embedding_snapshots[DEFAULT_REPO])
    assert kwargs["local_files_only"] is True
    assert "cache_folder" not in kwargs


def test_download_allowed_load_names_the_cache_folder_per_call(monkeypatch, tmp_path):
    from abstractcore.config import get_config_manager

    constants = pytest.importorskip(
        "huggingface_hub.constants", reason="the expected cache folder is huggingface_hub's own (not in the CI [test] extra)"
    )

    cfg = get_config_manager()
    monkeypatch.setattr(cfg.config.offline, "offline_first", False)
    monkeypatch.setattr(cfg.config.offline, "force_local_files_only", False)
    st = _mock_st()
    with patch.object(manager_module, "sentence_transformers", st):
        EmbeddingManager(cache_dir=tmp_path, backend="pytorch")
    args, kwargs = st.SentenceTransformer.call_args
    assert args[0] == DEFAULT_REPO
    assert kwargs["cache_folder"] == str(constants.HF_HUB_CACHE)
    assert "local_files_only" not in kwargs


def test_explicit_local_directory_and_legacy_bare_name(fake_embedding_snapshots, tmp_path):
    local = tmp_path / "my-model"
    local.mkdir()
    st = _mock_st()
    with patch.object(manager_module, "sentence_transformers", st):
        EmbeddingManager(model=str(local), cache_dir=tmp_path / "c1", backend="pytorch")
        assert st.SentenceTransformer.call_args[0][0] == str(local)
        EmbeddingManager(model="all-MiniLM-L6-v2", cache_dir=tmp_path / "c2", backend="pytorch")
        assert st.SentenceTransformer.call_args[0][0] == str(fake_embedding_snapshots[DEFAULT_REPO])


def test_uncached_model_refuses_with_download_hint(tmp_path):
    from abstractcore.exceptions import ModelNotFoundError

    with patch.object(manager_module, "sentence_transformers", _mock_st()):
        with pytest.raises(ModelNotFoundError, match="download it first"):
            EmbeddingManager(model="acme/not-downloaded", cache_dir=tmp_path)


# ------------------------------------------- item 4: HF_HUB_CACHE is honoured


def test_preexported_onnx_found_in_a_relocated_hub_cache(monkeypatch, tmp_path, make_hf_snapshot):
    hub = tmp_path / "relocated-hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    make_hf_snapshot(hub, DEFAULT_REPO, files=("config.json", "onnx/model.onnx"))
    with patch.object(manager_module, "sentence_transformers", _mock_st()):
        mgr = EmbeddingManager(cache_dir=tmp_path / "c", backend="pytorch")
    assert mgr._has_preexported_onnx() is True
