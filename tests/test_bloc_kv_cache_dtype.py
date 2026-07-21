"""0817 axis 5: KV-artifact cache-dtype gate (bloc lane).

Before this axis the bloc lane hardcoded quantization="fp" and REJECTED any
other value at validation — q8-quantized KV artifacts (the storage win MLX
already implements at the provider layer) could not exist in the corpus.
These pins hold the axis:

- Default requests ("fp") keep the whole pre-axis corpus valid (no false
  invalidation).
- quantization="q8" compiles a quantized artifact, records the dtype in the
  manifest, and REUSES under subsequent q8 requests.
- A dtype mismatch (stored vs requested) recompiles AT THE REQUESTED dtype
  with a labeled #FALLBACK — the request is authoritative.
- A stored dtype UNKNOWN to this build refuses and recompiles (never guess a
  tensor layout written by a different build).
- q8 against a provider whose save does not declare a `q8` parameter raises
  loudly (a silent fp-under-q8-label write is the exact class this axis
  kills).
- An unknown requested dtype raises up front.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from abstractcore.core.bloc_kv import (
    _compute_binding_id,
    ensure_bloc_kv_artifact,
    load_bloc_kv_artifact,
)
from abstractcore.core.file_blocs import FileBlocStore

from tests.test_bloc_kv import _StubPersistentMLXProvider, _upsert_record

import pytest


def test_default_fp_request_compiles_and_reuses(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="1" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert first.compiled is True
    assert first.manifest.quantization == "fp"
    second = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert second.compiled is False


def test_q8_request_compiles_quantized_and_reuses_under_q8(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="2" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record, quantization="q8")
    assert first.compiled is True
    assert first.manifest.quantization == "q8"
    # The provider was actually asked to quantize (stub records it in the payload).
    payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    assert payload.get("quantization") == "q8"

    second = ensure_bloc_kv_artifact(provider=provider, store=store, record=record, quantization="q8")
    assert second.compiled is False, "same-dtype request must reuse"
    assert second.manifest.quantization == "q8"


def test_dtype_mismatch_recompiles_at_requested_dtype(tmp_path: Path, caplog) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="3" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert first.compiled is True and first.manifest.quantization == "fp"

    with caplog.at_level(logging.WARNING):
        as_q8 = ensure_bloc_kv_artifact(provider=provider, store=store, record=record, quantization="q8")
    assert as_q8.compiled is True, "dtype mismatch must recompile"
    assert as_q8.manifest.quantization == "q8"
    assert any(
        "#FALLBACK" in r.getMessage() and "recompiling at the requested cache dtype" in r.getMessage()
        for r in caplog.records
    )

    back_to_fp = ensure_bloc_kv_artifact(provider=provider, store=store, record=record, quantization="fp")
    assert back_to_fp.compiled is True
    assert back_to_fp.manifest.quantization == "fp"


def test_unknown_stored_dtype_refuses_and_recompiles(tmp_path: Path, caplog) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="4" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert first.compiled is True

    # Simulate an artifact written by a DIFFERENT build: a dtype this build
    # does not know, with a consistent binding (so the dtype gate — not the
    # binding check — is what fires).
    manifest_path = first.manifest_path
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["quantization"] = "q4"
    data["binding_id"] = _compute_binding_id(dict(data), include_binding=False)
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        second = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert second.compiled is True, "unknown stored dtype must be refused and recompiled"
    assert second.manifest.quantization == "fp"
    assert any(
        "#FALLBACK" in r.getMessage() and "unknown cache" in r.getMessage() for r in caplog.records
    )


def test_q8_without_provider_support_raises(tmp_path: Path) -> None:
    class _NoQ8Stub(_StubPersistentMLXProvider):
        def prompt_cache_save(self, key, filename, **kwargs):  # no explicit q8 param
            kwargs.pop("q8", None)
            return super().prompt_cache_save(key, filename, **kwargs)

    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="5" * 64, path_name="doc.txt", content="hello world\n")
    provider = _NoQ8Stub(model="qwen3-test")

    with pytest.raises(ValueError, match="does not support q8"):
        ensure_bloc_kv_artifact(provider=provider, store=store, record=record, quantization="q8")


def test_unknown_requested_dtype_raises(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="6" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    with pytest.raises(ValueError, match="Unknown bloc KV cache dtype"):
        ensure_bloc_kv_artifact(provider=provider, store=store, record=record, quantization="q4")


def test_load_threads_quantization_through(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="7" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    result = load_bloc_kv_artifact(
        provider=provider, store=store, record=record, key="cache:q8-test", quantization="q8"
    )
    assert result.manifest.quantization == "q8"
    assert result.loaded is True


def test_pre_axis_manifest_missing_quantization_reads_as_fp(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)
    record = _upsert_record(store, tmp_path, sha="8" * 64, path_name="doc.txt", content="hello world\n")
    provider = _StubPersistentMLXProvider(model="qwen3-test")

    first = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert first.compiled is True

    # A pre-axis manifest never recorded quantization; simulate by clearing
    # the field (binding kept consistent) — the corpus must stay valid under
    # default requests.
    manifest_path = first.manifest_path
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["quantization"] = None
    data["binding_id"] = _compute_binding_id(dict(data), include_binding=False)
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    second = ensure_bloc_kv_artifact(provider=provider, store=store, record=record)
    assert second.compiled is False, "pre-axis manifest must remain valid under default fp requests"
