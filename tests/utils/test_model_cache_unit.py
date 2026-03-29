import os
import struct
from pathlib import Path

from abstractcore.utils.model_cache import (
    read_gguf_architecture,
    resolve_hf_snapshot_dir,
    resolve_lmstudio_model_dir,
)


def test_resolve_hf_snapshot_dir_prefers_refs_main(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    repo = cache / "models--org--name"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")

    resolved = resolve_hf_snapshot_dir("org/name", cache_dirs=[cache])
    assert resolved == snap


def test_resolve_hf_snapshot_dir_falls_back_to_latest_snapshot(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    repo = cache / "models--org--name"
    old = repo / "snapshots" / "old"
    new = repo / "snapshots" / "new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)

    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    resolved = resolve_hf_snapshot_dir("org/name", cache_dirs=[cache])
    assert resolved == new


def test_resolve_lmstudio_model_dir_case_insensitive(tmp_path: Path) -> None:
    base = tmp_path / "models"
    target = base / "Qwen" / "Qwen3.5-4B-MLX-4bit"
    target.mkdir(parents=True)

    resolved = resolve_lmstudio_model_dir("qwen/qwen3.5-4b-mlx-4bit", base_dirs=[base])
    assert resolved is not None
    assert resolved.samefile(target)


def test_read_gguf_architecture_reads_general_architecture(tmp_path: Path) -> None:
    p = tmp_path / "tiny.gguf"
    key = b"general.architecture"
    val = b"qwen35moe"

    payload = b"".join(
        [
            b"GGUF",
            struct.pack("<I", 3),  # version
            struct.pack("<Q", 0),  # tensor_count
            struct.pack("<Q", 1),  # kv_count
            struct.pack("<Q", len(key)),
            key,
            struct.pack("<I", 8),  # GGUF_TYPE_STRING
            struct.pack("<Q", len(val)),
            val,
        ]
    )
    p.write_bytes(payload)

    assert read_gguf_architecture(p) == "qwen35moe"
