"""0817 axis 4: KV-artifact weights-identity gate (bloc lane).

A checkpoint swap under the same model id (force-pushed revision,
re-quantized GGUF, edited shards) leaves text/tokenizer/config traces all
identical while the weights that computed the cached tensors are gone
(runtime c1734). These pins hold the gate:

- The compiled manifest RECORDS the weights fingerprint.
- Re-ensuring under a DIFFERENT fingerprint REFUSES and recompiles.
- Re-ensuring under the SAME fingerprint reuses (no false invalidation).
- A pre-axis artifact (empty recorded value) is reused UNVERIFIED with a
  #FALLBACK when the current weights are known.
- An unavailable CURRENT state abstains — reuse without noise; the provider
  load-time gate re-checks.
- The fingerprint is NOT part of binding_id (backfill safety).
"""

from __future__ import annotations

import logging
from pathlib import Path

from abstractcore.core.bloc_kv import (
    BlocKVArtifactManifest,
    _compute_binding_id,
    ensure_bloc_kv_artifact,
    read_bloc_kv_manifest,
)
from abstractcore.core.file_blocs import FileBlocStore

from tests.test_bloc_kv import _StubPersistentMLXProvider, _upsert_record


class _WeightsStub(_StubPersistentMLXProvider):
    """Stub whose weights fingerprint is settable, to drive the axis-4 gate."""

    def __init__(self, *args, weights_fp: str = "weights-revision:aaaa", **kwargs):
        super().__init__(*args, **kwargs)
        self._weights_fp = weights_fp

    def prompt_cache_weights_fingerprint(self) -> str:
        return self._weights_fp


def test_manifest_records_weights_fingerprint(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="1" * 64, path_name="doc.txt", content="hello world\n")
    provider = _WeightsStub(model="qwen3-test", weights_fp="weights-revision:v1aa")

    result = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert result.manifest.weights_fingerprint == "weights-revision:v1aa"

    reread = read_bloc_kv_manifest(store=store, record=record, provider=provider, model="qwen3-test")
    assert reread is not None
    assert reread.weights_fingerprint == "weights-revision:v1aa"


def test_same_weights_reuse_artifact(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="2" * 64, path_name="doc.txt", content="hello world\n")
    provider = _WeightsStub(model="qwen3-test", weights_fp="weights-revision:v1aa")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert first.compiled is True
    second = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert second.compiled is False
    assert second.manifest.weights_fingerprint == "weights-revision:v1aa"


def test_weights_mismatch_refuses_and_recompiles(tmp_path: Path, caplog) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="3" * 64, path_name="doc.txt", content="hello world\n")

    old = _WeightsStub(model="qwen3-test", weights_fp="weights-revision:v1aa")
    first = ensure_bloc_kv_artifact(provider=old, store=store, record=record)
    assert first.compiled is True

    # A checkpoint swap under the same model id.
    new = _WeightsStub(model="qwen3-test", weights_fp="weights-revision:v2bb")
    with caplog.at_level(logging.WARNING):
        second = ensure_bloc_kv_artifact(provider=new, store=store, record=record)
    assert second.compiled is True, "swapped-weights artifact must be REFUSED and recompiled"
    assert second.manifest.weights_fingerprint == "weights-revision:v2bb"
    assert any(
        "#FALLBACK" in r.getMessage() and "weights" in r.getMessage() and "recompiling" in r.getMessage()
        for r in caplog.records
    )

    third = ensure_bloc_kv_artifact(provider=new, store=store, record=record)
    assert third.compiled is False


def test_pre_axis_artifact_reused_unverified_with_fallback(tmp_path: Path, caplog) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="4" * 64, path_name="doc.txt", content="hello world\n")

    legacy = _WeightsStub(model="qwen3-test", weights_fp="")
    first = ensure_bloc_kv_artifact(provider=legacy, store=store, record=record)
    assert first.compiled is True
    assert first.manifest.weights_fingerprint == ""

    current = _WeightsStub(model="qwen3-test", weights_fp="weights-revision:v9ff")
    with caplog.at_level(logging.WARNING):
        second = ensure_bloc_kv_artifact(provider=current, store=store, record=record)
    assert second.compiled is False, "pre-axis artifact must be REUSED, not refused"
    assert any(
        "#FALLBACK" in r.getMessage() and "no weights fingerprint" in r.getMessage() for r in caplog.records
    )


def test_unloaded_weights_abstain_without_noise(tmp_path: Path, caplog) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="5" * 64, path_name="doc.txt", content="hello world\n")

    pinned = _WeightsStub(model="qwen3-test", weights_fp="weights-revision:v1aa")
    first = ensure_bloc_kv_artifact(provider=pinned, store=store, record=record)
    assert first.compiled is True

    unloaded = _WeightsStub(model="qwen3-test", weights_fp="")
    with caplog.at_level(logging.WARNING):
        second = ensure_bloc_kv_artifact(provider=unloaded, store=store, record=record)
    assert second.compiled is False
    assert not any("weights" in r.getMessage().lower() for r in caplog.records)


def test_weights_fingerprint_is_not_part_of_binding_id() -> None:
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
    with_weights = _compute_binding_id(
        {**base, "weights_fingerprint": "weights-revision:v1aa"}, include_binding=False
    )
    assert without == with_weights


def test_from_dict_defaults_missing_weights_to_empty() -> None:
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
    assert manifest.weights_fingerprint == ""
    assert manifest.to_dict()["weights_fingerprint"] == ""


def test_default_provider_weights_fingerprint_is_empty() -> None:
    from abstractcore.providers.base import BaseProvider

    stub = _StubPersistentMLXProvider(model="qwen3-test")
    assert BaseProvider.prompt_cache_weights_fingerprint(stub) == ""
