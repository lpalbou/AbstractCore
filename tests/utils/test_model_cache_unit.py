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


def _snapshot(cache: Path, repo: str, sha: str, files: dict, *, mtime: float) -> Path:
    snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / sha
    snap.mkdir(parents=True)
    for name, text in files.items():
        (snap / name).write_text(text)
    os.utime(snap, (mtime, mtime))
    return snap


def test_resolve_hf_load_snapshot_without_refs_main_skips_a_newer_readme_only_snapshot(tmp_path: Path) -> None:
    """AbstractCore's pinned downloads leave no refs/main; a later README-only
    fetch must not shadow the loadable snapshot (mission U)."""
    from abstractcore.utils.model_cache import resolve_hf_load_snapshot

    cache = tmp_path / "hub"
    good = _snapshot(cache, "org/name", "a" * 40, {"config.json": "{}", "model.safetensors": ""}, mtime=1_000)
    _snapshot(cache, "org/name", "b" * 40, {"README.md": "# card"}, mtime=2_000)

    assert resolve_hf_load_snapshot("org/name", cache_dirs=[cache]) == good


def test_resolve_hf_load_snapshot_honours_refs_main_and_misses_cleanly(tmp_path: Path) -> None:
    from abstractcore.utils.model_cache import resolve_hf_load_snapshot

    cache = tmp_path / "hub"
    _snapshot(cache, "org/name", "a" * 40, {"config.json": "{}"}, mtime=2_000)
    pinned = _snapshot(cache, "org/name", "c" * 40, {"config.json": "{}"}, mtime=1_000)
    (cache / "models--org--name" / "refs").mkdir()
    (cache / "models--org--name" / "refs" / "main").write_text("c" * 40)

    assert resolve_hf_load_snapshot("org/name", cache_dirs=[cache]) == pinned
    assert resolve_hf_load_snapshot("org/absent", cache_dirs=[cache]) is None


def test_describe_hf_snapshot_kinds(tmp_path: Path) -> None:
    import json

    from abstractcore.utils.model_cache import describe_hf_snapshot

    model = _snapshot(tmp_path, "o/m", "1" * 40, {"config.json": "{}", "model.safetensors": "", "tokenizer.json": "{}"}, mtime=1)
    adapter = _snapshot(
        tmp_path, "o/a", "2" * 40,
        {"adapter_config.json": json.dumps({"base_model_name_or_path": "o/m"}), "adapter_model.safetensors": ""}, mtime=1,
    )
    half = _snapshot(tmp_path, "o/h", "3" * 40, {"config.json": "{}"}, mtime=1)
    readme = _snapshot(tmp_path, "o/r", "4" * 40, {"README.md": "x"}, mtime=1)

    assert describe_hf_snapshot(model)["kind"] == "model"
    assert describe_hf_snapshot(model)["has_tokenizer"] is True
    assert describe_hf_snapshot(adapter)["kind"] == "adapter"
    assert describe_hf_snapshot(adapter)["base_model"] == "o/m"
    assert describe_hf_snapshot(half)["kind"] == "config_only"
    assert describe_hf_snapshot(readme) == {"kind": "unknown", "files": ["README.md"], "base_model": None, "has_tokenizer": False}
