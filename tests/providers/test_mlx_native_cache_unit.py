"""CPU-only tests of native cache identity, bounds and deletion authority."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from abstractcore.providers.mlx_native_cache import NativeCacheStore


@pytest.fixture
def fake_apc(monkeypatch):
    class Disk:
        def __init__(self, root, namespace, max_bytes, **kwargs):
            self.dir = Path(root) / namespace
            self.max_bytes = max_bytes
            self.flushed = self.closed = False

        def flush(self):
            self.flushed = True

        def close(self):
            self.closed = True

        @staticmethod
        def _is_canonical_store_file(path):
            return path.suffix == '.safetensors' and path.name.startswith('exact_')

    class Manager:
        def __init__(self, disk=None, **kwargs):
            self.disk = disk
            self.options = kwargs
            self.closed = self.cleared = False

        def close(self):
            self.closed = True
            if self.disk:
                self.disk.close()

        def clear(self):
            self.cleared = True

        def stats_snapshot(self):
            return {'hits': 2}

    monkeypatch.setitem(sys.modules, 'mlx_vlm.apc', SimpleNamespace(APCManager=Manager, DiskBlockStore=Disk))
    return Manager


def request(key='test', depth=0):
    return SimpleNamespace(cache_key=key, draft_tokens=depth)


def test_no_key_means_no_apc_or_disk_writes(tmp_path, fake_apc):
    store = NativeCacheStore({'weights_fingerprint':'a'}, disk_path=tmp_path/'new')
    assert store.manager(request(None)) is None
    assert not (tmp_path/'new').exists()


def test_one_manager_bounds_all_depths(tmp_path, fake_apc):
    store = NativeCacheStore({'weights_fingerprint':'a'}, disk_path=tmp_path)
    a = store.manager(request(depth=1))
    assert store.manager(request(depth=5)) is a
    assert a.disk.max_bytes == 4 << 30
    # Absent an operator value the snapshot budget is mlx-vlm's machine-sized
    # default: the budget caps EACH retained snapshot and a too-small constant
    # silently disables reuse for long conversations.
    assert 'memory_max_gb' not in a.options['overrides']
    assert store.memory_max_gb is None and store.stats()['memory_max_gb'] is None
    assert not store.directory.stat().st_mode & 0o077


def test_explicit_memory_budget_is_forwarded(fake_apc):
    store = NativeCacheStore({'weights_fingerprint':'a'}, memory_max_gb=3)
    overrides = store.manager(request()).options['overrides']
    assert overrides['memory_max_gb'] == 3.0
    # Mission A3: this lane now runs the session lane's measured shape instead of
    # silently inheriting mlx-vlm's 2048 x 2 (one snapshot per call, 8 prompts).
    from abstractcore.providers.mlx_native_session import (
        CHECKPOINT_ENTRIES, CHECKPOINT_GUARD_TOKENS, CHECKPOINT_INTERVAL_TOKENS)
    assert overrides['checkpoint_entries'] == CHECKPOINT_ENTRIES
    assert overrides['checkpoint_interval_tokens'] == CHECKPOINT_INTERVAL_TOKENS
    assert overrides['checkpoint_guard_tokens'] == CHECKPOINT_GUARD_TOKENS
    assert store.stats()['memory_max_gb'] == 3.0


def test_full_short_fingerprint_and_immutable_identity(tmp_path, fake_apc):
    identity = {'weights_fingerprint':'one', 'resolved_model':'/very-long-path/'*100}
    a = NativeCacheStore(identity, disk_path=tmp_path)
    identity['weights_fingerprint'] = 'two'
    b = NativeCacheStore(identity, disk_path=tmp_path)
    assert a.namespace != b.namespace
    assert len(a.namespace) < 128
    assert a.identity['weights_fingerprint'] == 'one'
    assert a.namespace == NativeCacheStore(a.identity, disk_path=tmp_path).namespace


def test_disk_requires_weights_identity(tmp_path):
    with pytest.raises(ValueError, match='weights_fingerprint'):
        NativeCacheStore({}, disk_path=tmp_path)


@pytest.mark.parametrize('value', [0, -1, True, '4', float('nan'), float('inf')])
@pytest.mark.parametrize('name', ['disk_max_gb','memory_max_gb'])
def test_budgets_reject_invalid(name, value):
    with pytest.raises(ValueError):
        NativeCacheStore({}, **{name:value})


def test_clear_flushes_and_removes_only_owned_cache_files(tmp_path, fake_apc):
    store = NativeCacheStore({'weights_fingerprint':'a'}, disk_path=tmp_path)
    manager = store.manager(request())
    cached = store.directory/'exact_a.safetensors'
    cached.write_bytes(b'cache')
    unrelated = store.directory/'notes.txt'
    unrelated.write_text('keep')
    other = tmp_path/'another-model.safetensors'
    other.write_bytes(b'keep')
    assert store.clear()['disk_files_removed'] == 1
    assert manager.closed and manager.cleared and manager.disk.flushed
    assert unrelated.read_text() == 'keep' and other.read_bytes() == b'keep'
    assert not cached.exists()
    assert store.manager(request()) is not manager


def test_foreign_namespace_and_symlink_are_refused(tmp_path, fake_apc):
    store = NativeCacheStore({'weights_fingerprint':'a'}, disk_path=tmp_path)
    store.directory.mkdir()
    with pytest.raises(ValueError, match='ownership marker'):
        store.manager(request())
    store.directory.rmdir()
    target = tmp_path/'foreign'
    target.mkdir()
    store.directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        store.manager(request())


def test_close_preserves_disk_and_reopens_same_namespace(tmp_path, fake_apc):
    store = NativeCacheStore({'weights_fingerprint':'a'}, disk_path=tmp_path)
    manager = store.manager(request())
    cached = store.directory/'exact_a.safetensors'
    cached.write_bytes(b'cache')
    store.close()
    store.close()
    assert manager.closed and cached.exists()
    with pytest.raises(RuntimeError, match='closed'):
        store.manager(request())
    other = NativeCacheStore({'weights_fingerprint':'a'}, disk_path=tmp_path)
    assert other.manager(request()).disk.dir == store.directory


def test_clear_refuses_changed_identity_marker(tmp_path, fake_apc):
    store = NativeCacheStore({'weights_fingerprint':'a'}, disk_path=tmp_path)
    store.manager(request())
    marker = store.directory/'abstractcore-identity.json'
    marker.write_text(json.dumps({'foreign':True}))
    with pytest.raises(ValueError, match='mismatch'):
        store.clear()
