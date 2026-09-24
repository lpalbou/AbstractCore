"""`refs/main` after a pinned download, and `abstractcore models repair-refs`
(mission V, 2026-09-24).

AbstractCore's downloader pins the listed commit (`snapshot_download(revision=
<sha>)`), and huggingface_hub writes no `refs/main` for a revision that already
is a hash. The snapshot is then complete on disk yet invisible BY NAME to every
offline loader (transformers `local_files_only`, `mlx_lm.load(id)`, vLLM):
"couldn't find them in the cached files".

Pins:
- a complete pinned download leaves `refs/main` = the downloaded sha;
- an existing `refs/main` naming another commit is never overwritten (and the
  completion message says which commit loading by name resolves);
- an incomplete download writes no ref;
- `repair_hf_refs` writes the ref only for a repo with NO ref and exactly ONE
  complete snapshot; ambiguous / interrupted / README-only / missing-shard /
  dangling-ref repos are reported and left alone; `--dry-run` writes nothing;
- quarantined caches are not scanned.
No network: the hub is a fake that writes the real cache layout.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest

from abstractcore.config import model_materializer as mm
from tests.models_engines_fakes import isolate_host

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


def _repo(cache: Path, repo_id: str) -> Path:
    return cache / ("models--" + repo_id.replace("/", "--"))


def _put_snapshot(repo_dir: Path, sha: str, files: Dict[str, bytes]) -> Path:
    """The hub's layout: content in blobs/<etag>, relative symlinks in snapshots/<sha>/."""
    snap = repo_dir / "snapshots" / sha
    for name, data in files.items():
        etag = hashlib.sha256(data).hexdigest()
        blob = repo_dir / "blobs" / etag
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        link = snap / name
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, link.parent), link)
    return snap


MODEL_FILES = {"config.json": b'{"model_type": "llama"}', "model.safetensors": b"w" * 64}


# ---------------------------------------------------------------------------
# ensure_hf_main_ref
# ---------------------------------------------------------------------------


def test_the_ref_is_written_when_missing_and_holds_just_the_sha(tmp_path):
    repo = _repo(tmp_path, "org/m")
    _put_snapshot(repo, SHA_A, MODEL_FILES)
    state, note = mm.ensure_hf_main_ref(repo, SHA_A)
    assert state == "written" and SHA_A[:12] in note
    assert (repo / "refs" / "main").read_text() == SHA_A  # the hub's format: no newline
    assert mm.ensure_hf_main_ref(repo, SHA_A) == ("present", "")


def test_an_existing_ref_to_another_commit_is_never_overwritten(tmp_path):
    repo = _repo(tmp_path, "org/m")
    _put_snapshot(repo, SHA_A, MODEL_FILES)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(SHA_B)
    state, note = mm.ensure_hf_main_ref(repo, SHA_A)
    assert state == "kept_other"
    assert (repo / "refs" / "main").read_text() == SHA_B
    assert "org/m" in note and SHA_B[:12] in note


@pytest.mark.parametrize("sha", [None, "", "main", "v1.0", "A" * 40, "a" * 39])
def test_a_revision_that_is_not_a_commit_hash_writes_nothing(tmp_path, sha):
    repo = _repo(tmp_path, "org/m")
    _put_snapshot(repo, SHA_A, MODEL_FILES)
    assert mm.ensure_hf_main_ref(repo, sha)[0] == "skipped"
    assert not (repo / "refs").exists()


def test_no_ref_is_written_for_a_snapshot_that_is_not_on_disk(tmp_path):
    repo = _repo(tmp_path, "org/m")
    _put_snapshot(repo, SHA_A, MODEL_FILES)
    assert mm.ensure_hf_main_ref(repo, SHA_B)[0] == "skipped"
    assert not (repo / "refs" / "main").exists()


def test_a_ref_created_by_someone_else_mid_write_survives(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "org/m")
    _put_snapshot(repo, SHA_A, MODEL_FILES)
    real_open = open
    raced = {"done": False}

    def racing_open(path, mode="r", *args, **kwargs):
        if mode == "x" and not raced["done"]:
            raced["done"] = True
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(SHA_B)  # another writer wins the race
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", racing_open)
    state, _ = mm.ensure_hf_main_ref(repo, SHA_A)
    assert state == "kept_other"
    assert (repo / "refs" / "main").read_text() == SHA_B


# ---------------------------------------------------------------------------
# _download_huggingface (in-process, fake hub writing the real layout)
# ---------------------------------------------------------------------------


def _fake_pinned_hub(cache: Path, files: Dict[str, bytes], *, drop: Optional[str] = None):
    calls: List[dict] = []

    def snapshot_download(**kw):
        calls.append(kw)
        repo_dir = _repo(Path(kw["cache_dir"]), kw["repo_id"])
        present = {k: v for k, v in files.items() if k != drop}
        return str(_put_snapshot(repo_dir, kw["revision"], present))  # like the hub: NO refs/main for a sha

    return SimpleNamespace(snapshot_download=snapshot_download), calls


def _plan(files: Dict[str, bytes]):
    return [{"name": n, "size": len(d), "etag": hashlib.sha256(d).hexdigest()} for n, d in files.items()]


def test_a_complete_pinned_download_leaves_refs_main(host, monkeypatch):
    hf = host["hf"]
    hub, calls = _fake_pinned_hub(hf, MODEL_FILES)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(mm, "_hf_file_plan", lambda repo, patterns, token: (_plan(MODEL_FILES), SHA_A, ""))
    monkeypatch.setattr(mm, "_hf_download_cache_dir", lambda: hf)
    monkeypatch.setattr(mm, "_job_control", lambda: None)

    out = mm._download_huggingface("org/tiny", lambda _p: None, None)

    assert out.ok, out
    assert calls and calls[0]["revision"] == SHA_A
    assert (_repo(hf, "org/tiny") / "refs" / "main").read_text() == SHA_A
    assert "refs/main" in out.message


def test_a_download_keeps_an_existing_ref_and_says_which_commit_loads(host, monkeypatch):
    hf = host["hf"]
    repo = _repo(hf, "org/tiny")
    _put_snapshot(repo, SHA_B, MODEL_FILES)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(SHA_B)
    hub, _ = _fake_pinned_hub(hf, {**MODEL_FILES, "config.json": b'{"model_type": "llama", "v": 2}'})
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    new_files = {**MODEL_FILES, "config.json": b'{"model_type": "llama", "v": 2}'}
    monkeypatch.setattr(mm, "_hf_file_plan", lambda repo, patterns, token: (_plan(new_files), SHA_A, ""))
    monkeypatch.setattr(mm, "_hf_download_cache_dir", lambda: hf)
    monkeypatch.setattr(mm, "_job_control", lambda: None)

    out = mm._download_huggingface("org/tiny", lambda _p: None, None)

    assert out.ok, out
    assert (repo / "refs" / "main").read_text() == SHA_B
    assert "left unchanged" in out.message and SHA_B[:12] in out.message


def test_an_incomplete_pinned_download_writes_no_ref(host, monkeypatch):
    hf = host["hf"]
    hub, _ = _fake_pinned_hub(hf, MODEL_FILES, drop="model.safetensors")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(mm, "_hf_file_plan", lambda repo, patterns, token: (_plan(MODEL_FILES), SHA_A, ""))
    monkeypatch.setattr(mm, "_hf_download_cache_dir", lambda: hf)
    monkeypatch.setattr(mm, "_job_control", lambda: None)

    out = mm._download_huggingface("org/tiny", lambda _p: None, None)

    assert not out.ok
    assert not (_repo(hf, "org/tiny") / "refs" / "main").exists()


# ---------------------------------------------------------------------------
# repair_hf_refs / `abstractcore models repair-refs`
# ---------------------------------------------------------------------------


def _by_repo(payload) -> Dict[str, dict]:
    return {r["repo_id"]: r for r in payload["rows"]}


def _fixture_cache(cache: Path) -> None:
    _put_snapshot(_repo(cache, "org/fixable"), SHA_A, MODEL_FILES)
    _put_snapshot(_repo(cache, "org/two"), SHA_A, MODEL_FILES)
    _put_snapshot(_repo(cache, "org/two"), SHA_B, {**MODEL_FILES, "config.json": b"{}"})
    interrupted = _repo(cache, "org/interrupted")
    _put_snapshot(interrupted, SHA_A, MODEL_FILES)
    (interrupted / "blobs" / "deadbeef.incomplete").write_bytes(b"x")
    _put_snapshot(_repo(cache, "org/readme-only"), SHA_A, {"README.md": b"# card"})
    index = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
    _put_snapshot(
        _repo(cache, "org/missing-shard"),
        SHA_A,
        {"config.json": b"{}", "model.safetensors.index.json": index.encode(), "model-00001-of-00002.safetensors": b"1"},
    )
    dangling = _repo(cache, "org/dangling")
    _put_snapshot(dangling, SHA_A, MODEL_FILES)
    (dangling / "refs").mkdir()
    (dangling / "refs" / "main").write_text(SHA_B)
    healthy = _repo(cache, "org/healthy")
    _put_snapshot(healthy, SHA_A, MODEL_FILES)
    (healthy / "refs").mkdir()
    (healthy / "refs" / "main").write_text(SHA_A)


def test_repair_dry_run_lists_and_writes_nothing(tmp_path):
    cache = tmp_path / "hub"
    _fixture_cache(cache)
    payload = mm.repair_hf_refs(apply=False, cache_dirs=[cache])
    rows = _by_repo(payload)
    assert rows["org/fixable"]["status"] == "repairable" and rows["org/fixable"]["snapshot"] == SHA_A
    assert rows["org/two"]["status"] == "ambiguous"
    assert rows["org/interrupted"]["status"] == "no_complete_snapshot"
    assert rows["org/readme-only"]["status"] == "no_complete_snapshot"
    assert rows["org/missing-shard"]["status"] == "no_complete_snapshot"
    assert "model-00002-of-00002.safetensors" in rows["org/missing-shard"]["reason"]
    assert rows["org/dangling"]["status"] == "dangling_ref"
    assert "org/healthy" not in rows and payload["counts"]["ok"] == 1
    assert not any((cache / d / "refs" / "main").exists() for d in ("models--org--fixable", "models--org--two"))


def test_repair_apply_writes_only_the_unambiguous_ref(tmp_path):
    cache = tmp_path / "hub"
    _fixture_cache(cache)
    payload = mm.repair_hf_refs(apply=True, cache_dirs=[cache])
    rows = _by_repo(payload)
    assert rows["org/fixable"]["status"] == "repaired"
    assert (_repo(cache, "org/fixable") / "refs" / "main").read_text() == SHA_A
    for untouched in ("org/two", "org/interrupted", "org/readme-only", "org/missing-shard"):
        assert not (_repo(cache, untouched) / "refs" / "main").exists(), untouched
    assert (_repo(cache, "org/dangling") / "refs" / "main").read_text() == SHA_B
    # Idempotent: a second pass finds it healthy.
    again = mm.repair_hf_refs(apply=True, cache_dirs=[cache])
    assert "org/fixable" not in _by_repo(again) and again["counts"]["ok"] == 2


def test_quarantined_caches_are_not_scanned(tmp_path):
    cache = tmp_path / "runtime" / "model-quarantine" / "q1" / "hf-hub"
    _put_snapshot(_repo(cache, "org/fixable"), SHA_A, MODEL_FILES)
    payload = mm.repair_hf_refs(apply=True, cache_dirs=[cache])
    assert payload["rows"] == [] and payload["cache_dirs"] == []
    assert not (_repo(cache, "org/fixable") / "refs").exists()


def test_the_cli_verb_dry_run_and_apply(tmp_path, capsys):
    import argparse

    from abstractcore.config.models_engines_cli import add_models_subparsers

    cache = tmp_path / "hub"
    _fixture_cache(cache)
    parser = argparse.ArgumentParser()
    add_models_subparsers(parser.add_subparsers(dest="cmd"))

    args = parser.parse_args(["repair-refs", "--dry-run", "--cache-dir", str(cache), "--json"])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["apply"] is False and _by_repo(payload)["org/fixable"]["status"] == "repairable"
    assert not (_repo(cache, "org/fixable") / "refs" / "main").exists()

    args = parser.parse_args(["repair-refs", "--cache-dir", str(cache)])
    assert args.func(args) == 0
    text = capsys.readouterr().out
    assert "repaired (1)" in text and "org/fixable" in text and "ambiguous (1)" in text
    assert (_repo(cache, "org/fixable") / "refs" / "main").read_text() == SHA_A
