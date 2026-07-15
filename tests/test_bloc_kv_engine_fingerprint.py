"""0817 first axis: KV-artifact engine-identity gate.

The silently-wrong-cache class: a saved KV artifact is only valid under the
engine (mlx_lm / transformers / llama.cpp) + version that produced it — the
serialized cache layout is engine-specific. Before this axis, an engine
upgrade left the old artifact loadable with NO error, injecting stale-layout
KV. These pins hold the gate:

- The compiled manifest RECORDS the engine fingerprint.
- Re-ensuring under a DIFFERENT engine fingerprint REFUSES the stale artifact
  and recompiles (never a silent reload).
- Re-ensuring under the SAME fingerprint reuses (no false invalidation).
- A pre-0817 artifact (empty fingerprint) is reused UNVERIFIED with a
  #FALLBACK — not refused (no corpus-wide invalidation), not silent.
- The fingerprint is NOT part of binding_id (backfill safety: adding it there
  would reject every pre-0817 manifest via a recomputed-binding mismatch).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from abstractcore.core.bloc_kv import (
    BlocKVArtifactManifest,
    _compute_binding_id,
    ensure_bloc_kv_artifact,
    read_bloc_kv_manifest,
)
from abstractcore.core.file_blocs import FileBlocStore

from tests.test_bloc_kv import _StubPersistentMLXProvider, _upsert_record


class _EngineStub(_StubPersistentMLXProvider):
    """Stub whose engine fingerprint is settable, to drive the 0817 gate."""

    def __init__(self, *args, engine: str = "mlx_lm==1.0.0", **kwargs):
        super().__init__(*args, **kwargs)
        self._engine = engine

    def prompt_cache_engine_fingerprint(self) -> str:
        return self._engine


def test_manifest_records_engine_fingerprint(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="a" * 64, path_name="doc.txt", content="hello world\n")
    provider = _EngineStub(model="qwen3-test", engine="mlx_lm==0.28.3")

    result = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert result.manifest.engine_fingerprint == "mlx_lm==0.28.3"

    # Round-trips through the on-disk manifest.
    reread = read_bloc_kv_manifest(store=store, record=record, provider=provider, model="qwen3-test")
    assert reread is not None
    assert reread.engine_fingerprint == "mlx_lm==0.28.3"


def test_same_engine_reuses_artifact(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="b" * 64, path_name="doc.txt", content="hello world\n")
    provider = _EngineStub(model="qwen3-test", engine="mlx_lm==0.28.3")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert first.compiled is True

    # Same engine → no false invalidation.
    second = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert second.compiled is False
    assert second.manifest.engine_fingerprint == "mlx_lm==0.28.3"


def test_engine_mismatch_refuses_and_recompiles(tmp_path: Path, caplog) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="c" * 64, path_name="doc.txt", content="hello world\n")

    old = _EngineStub(model="qwen3-test", engine="mlx_lm==0.28.3")
    first = ensure_bloc_kv_artifact(provider=old, store=store, record=record)
    assert first.compiled is True

    # An mlx_lm upgrade: the recorded layout can no longer be trusted.
    new = _EngineStub(model="qwen3-test", engine="mlx_lm==0.30.0")
    second = ensure_bloc_kv_artifact(provider=new, store=store, record=record)
    assert second.compiled is True, "stale-engine artifact must be REFUSED and recompiled"
    assert second.manifest.engine_fingerprint == "mlx_lm==0.30.0"

    # And now reusing under the new engine is stable.
    third = ensure_bloc_kv_artifact(provider=new, store=store, record=record)
    assert third.compiled is False


def test_pre_0817_artifact_reused_unverified_with_fallback(tmp_path: Path, caplog) -> None:
    """An empty recorded fingerprint (pre-0817) is reused, NOT refused — but
    the reuse is labeled #FALLBACK, never silent."""
    import logging

    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="d" * 64, path_name="doc.txt", content="hello world\n")

    # Compile with a provider that reports NO engine (simulates a pre-0817 artifact).
    legacy = _EngineStub(model="qwen3-test", engine="")
    first = ensure_bloc_kv_artifact(provider=legacy, store=store, record=record)
    assert first.compiled is True
    assert first.manifest.engine_fingerprint == ""

    # Reuse under a provider that DOES report an engine: accepted (no corpus
    # invalidation) but warned.
    current = _EngineStub(model="qwen3-test", engine="mlx_lm==0.30.0")
    with caplog.at_level(logging.WARNING):
        second = ensure_bloc_kv_artifact(provider=current, store=store, record=record)
    assert second.compiled is False, "pre-0817 artifact must be REUSED, not refused"
    assert any("#FALLBACK" in r.getMessage() and "no engine fingerprint" in r.getMessage() for r in caplog.records)


def test_engine_fingerprint_is_not_part_of_binding_id() -> None:
    """Backfill safety: binding_id must be identical with and without the new
    field, or every pre-0817 manifest's recomputed binding would mismatch and
    reject the existing corpus."""
    base = {
        "version": 1,
        "provider": "mlx",
        "model": "qwen3-test",
        "model_resolved_id": "/resolved/qwen3-test",
        "cache_backend": "mlx",
        "artifact_format": "abstractcore-mlx-prompt-cache/v1",
        "bloc_sha256": "a" * 64,
        "bloc_id": None,
        "content_sha256": "b" * 64,
        "path_in_prompt": "/x/doc.txt",
        "recipe_id": "attached_file_box",
        "recipe_version": 1,
        "rendered_recipe_sha256": "c" * 64,
        "renderer_version": 1,
        "serializer_version": "mlx-prompt-fragment/v1:qwen-chatml",
        "artifact_filename": "abc.safetensors",
        "artifact_sha256": "d" * 64,
        "quantization": "fp",
    }
    without = _compute_binding_id(dict(base), include_binding=False)
    with_engine = _compute_binding_id({**base, "engine_fingerprint": "mlx_lm==0.30.0"}, include_binding=False)
    assert without == with_engine


def test_from_dict_defaults_missing_engine_to_empty() -> None:
    """A pre-0817 manifest dict (no engine_fingerprint key) loads with ''."""
    data = {
        "version": 1,
        "provider": "mlx",
        "model": "qwen3-test",
        "model_resolved_id": "/resolved/qwen3-test",
        "cache_backend": "mlx",
        "artifact_format": "abstractcore-mlx-prompt-cache/v1",
        "bloc_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "path_in_prompt": "/x/doc.txt",
        "recipe_id": "attached_file_box",
        "recipe_version": 1,
        "rendered_recipe_sha256": "c" * 64,
        "renderer_version": 1,
        "serializer_version": "mlx-prompt-fragment/v1:qwen-chatml",
        "artifact_filename": "abc.safetensors",
        "artifact_sha256": "d" * 64,
        "quantization": "fp",
        "created_at": "2026-07-01T00:00:00+00:00",
        "token_count": 10,
        "binding_id": "unused-in-this-test",
    }
    manifest = BlocKVArtifactManifest.from_dict(data)
    assert manifest.engine_fingerprint == ""
    assert manifest.to_dict()["engine_fingerprint"] == ""


def test_default_provider_engine_fingerprint_is_empty() -> None:
    """BaseProvider default is '' (engine not version-pinnable); MLX/HF override."""
    from abstractcore.providers.base import BaseProvider

    stub = _StubPersistentMLXProvider(model="qwen3-test")
    # The stub does not override the base hook.
    assert BaseProvider.prompt_cache_engine_fingerprint(stub) == ""
