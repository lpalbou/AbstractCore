from __future__ import annotations

import json
from pathlib import Path

from abstractcore.core.file_blocs import FileBlocStore, heuristic_summary_and_keywords


def test_file_bloc_store_upsert_and_list(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)

    file_meta = {
        "path": str(tmp_path / "doc.md"),
        "media_type": "text",
        "size_bytes": 123,
        "mtime_ns": 456,
        "sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "format": "text/markdown",
        "content_length": 11,
        "estimated_tokens": 3,
    }

    rec = store.upsert(file_meta=file_meta, content="hello world", relpath_base=tmp_path)
    assert rec.sha256 == "a" * 64
    assert rec.filename == "doc.md"
    assert rec.relpath == "doc.md"

    got = store.get("a" * 64)
    assert got is not None
    assert got.sha256 == rec.sha256

    items = store.list()
    assert len(items) == 1
    assert items[0].sha256 == rec.sha256

    assert store.content_path(rec.sha256).exists()
    assert store.meta_path(rec.sha256).exists()

    kv = store.kv_cache_path(rec.sha256, provider="huggingface", model="dummy-model")
    manifest = store.kv_cache_manifest_path(rec.sha256, provider="huggingface", model="dummy-model")
    assert str(kv).endswith(".safetensors")
    assert str(manifest).endswith(".manifest.json")
    assert store.has_kv_cache(rec.sha256, provider="huggingface", model="dummy-model") is False
    assert store.has_kv_cache_manifest(rec.sha256, provider="huggingface", model="dummy-model") is False


def test_file_bloc_store_stable_bloc_ids_are_monotonic_and_durable(tmp_path: Path) -> None:
    store = FileBlocStore(root_dir=tmp_path)

    def _meta(*, sha: str, name: str) -> dict:
        return {
            "path": str(tmp_path / name),
            "media_type": "text",
            "size_bytes": 1,
            "mtime_ns": 1,
            "sha256": sha,
            "content_sha256": sha,
            "format": "text/plain",
            "content_length": 1,
            "estimated_tokens": 1,
        }

    sha_a = "a" * 64
    sha_b = "b" * 64
    sha_c = "c" * 64

    for i, sha in enumerate([sha_a, sha_b, sha_c], 1):
        _ = store.upsert(file_meta=_meta(sha=sha, name=f"{i}.txt"), content="x", relpath_base=tmp_path)
        mp = store.meta_path(sha)
        data = json.loads(mp.read_text(encoding="utf-8"))
        data.pop("bloc_id", None)
        data["created_at"] = float(i)
        data["updated_at"] = float(i)
        mp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    assigned = store.ensure_bloc_ids()
    assert assigned == 3
    assert store.bloc_id(sha_a) == 1
    assert store.bloc_id(sha_b) == 2
    assert store.bloc_id(sha_c) == 3
    assert store.sha_for_bloc_id(2) == sha_b
    assert store.get_by_bloc_id(3) is not None

    sha_d = "d" * 64
    rec_d = store.upsert(file_meta=_meta(sha=sha_d, name="d.txt"), content="y", relpath_base=tmp_path)
    assert rec_d.bloc_id == 4
    assert store.sha_for_bloc_id(4) == sha_d

    rec_b2 = store.upsert(file_meta=_meta(sha=sha_b, name="2.txt"), content="x", relpath_base=tmp_path)
    assert rec_b2.bloc_id == 2


def test_heuristic_summary_and_keywords() -> None:
    summary, kw = heuristic_summary_and_keywords(
        "Title line\n\nThis is a test document about Prompt Caching and KV reuse.\nPrompt caching improves TTFT.\n"
    )
    assert summary == "Title line"
    assert isinstance(kw, list)
    assert "the" not in kw
