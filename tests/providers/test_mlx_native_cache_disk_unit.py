"""Real upstream disk-header round trips, without MLX arrays/model execution.

These optional backend tests exercise actual safetensors metadata and upstream
DiskBlockStore index reconstruction. They do not claim KV tensor correctness;
the parent live validation separately checks native warm/restored inference.
"""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from abstractcore.providers.mlx_native_cache import NativeCacheStore


@pytest.fixture
def disk_api(monkeypatch):
    apc = pytest.importorskip("mlx_vlm.apc")
    safetensors = pytest.importorskip("safetensors")

    def no_tensor_io(*args, **kwargs):
        pytest.fail("Disk metadata roundtrip must not allocate/evaluate MLX tensors")

    monkeypatch.setattr(apc.mx, "load", no_tensor_io)
    monkeypatch.setattr(apc.mx, "eval", no_tensor_io)
    return apc, safetensors


def _header(safetensors, path, metadata):
    # The Rust writer produces a valid safetensors header without any tensors.
    safetensors.serialize_file({}, str(path), metadata=metadata)


def test_real_disk_index_roundtrip_and_prefix_semantic_salt(tmp_path, disk_api):
    apc, safetensors = disk_api
    namespace = "native-metadata-roundtrip"
    directory = tmp_path / namespace
    directory.mkdir(mode=0o700)
    exact = directory / ("exact_" + "a" * 32 + ".safetensors")
    _header(safetensors, exact, {
        "layout": "exact_cache_v1", "cache_hash": "991", "extra_hash": "42",
        "token_ids": "11,12,13,14", "num_entries": "0", "prefix_trimmable": "1",
    })
    shard = directory / ("shard_" + "b" * 32 + ".safetensors")
    _header(safetensors, shard, {
        "layout": "layer_major_v1", "block_hashes": "901,902", "block_size": "2",
        "num_layers": "0",
        "b0_meta": json.dumps({"parent_hash": 0, "extra_hash": 42, "token_ids": [11, 12]}),
        "b1_meta": json.dumps({"parent_hash": 901, "extra_hash": 42, "token_ids": [13, 14]}),
    })
    before = exact.read_bytes(), shard.read_bytes()
    for _ in range(2):
        store = apc.DiskBlockStore(tmp_path, namespace=namespace, max_bytes=1 << 20)
        try:
            assert store.num_blocks_indexed == 2
            assert store.num_exact_indexed == 1
            assert store.has(901) and store.has(902)
            assert store.find_exact_prefix([11, 12, 13, 14, 15], extra_hash=42) == (991, 4)
            assert store.find_exact_prefix([11, 12, 13, 14, 15], extra_hash=99) is None
            _, metadata, _ = store._open_shard_header(shard)
            decoded = store._decode_block_metadata(metadata, 1)
            assert decoded == {"parent_hash": 901, "extra_hash": "42", "token_ids": "13,14"}
            assert store.disk_bytes == exact.stat().st_size + shard.stat().st_size
            store.flush()
        finally:
            store.close()
    assert before == (exact.read_bytes(), shard.read_bytes())


def test_real_disk_eviction_respects_canonical_files_and_namespace(tmp_path, disk_api):
    apc, safetensors = disk_api
    directory = tmp_path / "bounded"
    directory.mkdir(mode=0o700)
    cached = directory / ("exact_" + "c" * 32 + ".safetensors")
    _header(safetensors, cached, {"cache_hash": "7", "layout": "exact_cache_v1"})
    marker = directory / "abstractcore-identity.json"
    marker.write_text('{"owned":true}')
    other = tmp_path / "another-model.safetensors"
    _header(safetensors, other, {"other": "model"})
    store = apc.DiskBlockStore(tmp_path, namespace="bounded", max_bytes=1)
    try:
        assert store.num_exact_indexed == 1
        assert store._maybe_evict() == 1
        assert store.disk_bytes == 0 and store.num_exact_indexed == 0
        assert not cached.exists()
        assert marker.read_text() == '{"owned":true}' and other.is_file()
    finally:
        store.close()


def test_real_disk_corrupt_header_is_rejected_as_cache_miss(tmp_path, disk_api):
    apc, _ = disk_api
    directory = tmp_path / "corrupt"
    directory.mkdir(mode=0o700)
    corrupted = directory / ("exact_" + "d" * 32 + ".safetensors")
    corrupted.write_bytes(b"not a valid safetensors header")
    store = apc.DiskBlockStore(tmp_path, namespace="corrupt", max_bytes=1 << 20)
    try:
        assert store.num_exact_indexed == 0
        assert store.find_exact_prefix([1, 2, 3], extra_hash=0) is None
        assert not corrupted.exists()
    finally:
        store.close()


def test_native_namespace_lock_survives_clear_and_releases_on_close(tmp_path, disk_api):
    first = NativeCacheStore({"weights_fingerprint": "same"}, disk_path=tmp_path)
    second = NativeCacheStore({"weights_fingerprint": "same"}, disk_path=tmp_path)
    try:
        first._prepare_directory()
        with pytest.raises(RuntimeError, match="already owned"):
            second._prepare_directory()
        first.clear()
        with pytest.raises(RuntimeError, match="already owned"):
            second._prepare_directory()
        first.close()
        assert second._prepare_directory() == first.directory
    finally:
        first.close()
        second.close()


def test_native_namespace_lock_refuses_second_process(tmp_path):
    first = NativeCacheStore({"weights_fingerprint": "same"}, disk_path=tmp_path)
    first._prepare_directory()
    program = """
import sys
from abstractcore.providers.mlx_native_cache import NativeCacheStore
store = NativeCacheStore({'weights_fingerprint': 'same'}, disk_path=sys.argv[1])
try:
    store._prepare_directory()
except RuntimeError:
    print('locked')
else:
    print('acquired')
finally:
    store.close()
"""

    def child():
        return subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()

    try:
        assert child() == "locked"
        first.close()
        assert child() == "acquired"
    finally:
        first.close()


def test_native_close_failure_does_not_leak_process_lock(tmp_path):
    first = NativeCacheStore({"weights_fingerprint": "same"}, disk_path=tmp_path)
    second = NativeCacheStore({"weights_fingerprint": "same"}, disk_path=tmp_path)
    first._prepare_directory()

    def fail():
        raise OSError("disk flush failed")

    first._manager = SimpleNamespace(disk=SimpleNamespace(flush=fail), close=lambda: None, clear=lambda: None)
    try:
        with pytest.raises(OSError, match="disk flush failed"):
            first.close()
        assert first._closed
        assert first._lock_fd is None
        assert second._prepare_directory() == first.directory
    finally:
        first.close()
        second.close()


def test_native_clear_does_not_delete_non_safetensors_same_stem(tmp_path, disk_api):
    store = NativeCacheStore({"weights_fingerprint": "same"}, disk_path=tmp_path)
    try:
        store._prepare_directory()
        note = store.directory / ("exact_" + "e" * 32 + ".txt")
        note.write_text("operator note, not a cache tensor")
        assert store.clear()["disk_files_removed"] == 0
        assert note.read_text() == "operator note, not a cache tensor"
    finally:
        store.close()
