"""Shared fixtures for the embeddings tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# The HuggingFace repos the mocked EmbeddingManager tests load: the default
# (`all-minilm-l6-v2` -> sentence-transformers/all-MiniLM-L6-v2) and the
# `embeddinggemma` alias.
FAKE_EMBEDDING_REPOS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/embeddinggemma-300m",
)


def make_fake_hf_snapshot(hub_cache: Path, repo_id: str, *, sha: str = "0" * 40, files=("config.json",)) -> Path:
    """A cached snapshot laid out exactly like huggingface_hub writes one."""
    model_dir = Path(hub_cache) / ("models--" + repo_id.replace("/", "--"))
    snapshot = model_dir / "snapshots" / sha
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    (model_dir / "refs").mkdir(parents=True, exist_ok=True)
    (model_dir / "refs" / "main").write_text(sha, encoding="utf-8")
    return snapshot


@pytest.fixture
def fake_embedding_snapshots(tmp_path, monkeypatch):
    """Cache the embedding repos the mocked tests load, in a scratch hub cache.

    Loading is offline-first (mission V): the manager resolves the repo id to
    its cached snapshot DIRECTORY and never asks the Hub, so a test that mocks
    `sentence_transformers` must still have the model "downloaded". Returns
    {repo_id: snapshot_dir}.
    """
    hub = tmp_path / "hf-hub-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    return {repo: make_fake_hf_snapshot(hub, repo) for repo in FAKE_EMBEDDING_REPOS}


@pytest.fixture
def make_hf_snapshot():
    """`make_fake_hf_snapshot` for test modules (tests/embeddings is not a package)."""
    return make_fake_hf_snapshot
