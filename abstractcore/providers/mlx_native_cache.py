"""Bounded, engine-private prefix storage for native MLX VLM sessions.

This is automatic full-history prefix reuse, not the public durable
``abstractcore-mlx-prompt-cache/v1`` append-cache format. The execution worker
owns this object; all methods which touch APC run on that worker.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import fcntl
from pathlib import Path
from typing import Any, Mapping


def _positive_gb(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


class NativeCacheStore:
    """One RAM/disk budget per native runtime; disk persistence is opt-in.

    ``identity`` must describe the actual target and processor, not just its
    repository alias. The provider supplies its existing artifact fingerprints.
    Per-request scope, key, image and execution-layout salts belong to the
    runtime's APC semantic key. Neither a cache key nor scope is authentication.
    """

    def __init__(self, identity: Mapping[str, Any], *, disk_path=None,
                 disk_max_gb: float = 4.0, memory_max_gb: float | None = None):
        if not isinstance(identity, Mapping):
            raise ValueError("Native cache identity must be a mapping")
        self.disk_max_gb = _positive_gb(disk_max_gb, "mlx_cache_disk_max_gb")
        # None = mlx-vlm's machine-relative budget (min(8 GiB, working set / 10)).
        # A number is a hard cap on EACH retained snapshot, not only the total:
        # mlx-vlm skips retaining any snapshot larger than it, so a cap below a
        # long conversation's snapshot size silently disables reuse entirely.
        self.memory_max_gb = None if memory_max_gb is None else _positive_gb(memory_max_gb, "mlx_cache_memory_max_gb")
        self.identity = json.loads(json.dumps(dict(identity), sort_keys=True))
        self.identity["cache_schema"] = "abstractcore-native-vlm-apc/v1"
        self.identity["engine_versions"] = {}
        for name in ("mlx", "mlx-vlm"):
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = "unavailable"
            self.identity["engine_versions"][name] = version
        self._encoded_identity = json.dumps(self.identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(self._encoded_identity.encode()).hexdigest()
        # Short enough that upstream's 128-character sanitization cannot drop
        # the fingerprint (long Hugging Face paths otherwise can do so).
        self.namespace = "acore-mlx-apc-v1-" + digest
        self.root = Path(disk_path).expanduser().resolve() if disk_path is not None else None
        if self.root is not None and not str(self.identity.get("weights_fingerprint") or "").strip():
            raise ValueError("Native disk caching requires a verified weights_fingerprint")
        self.directory = self.root / self.namespace if self.root else None
        self._manager = None
        self._closed = False
        self._lock_fd = None

    def _prepare_directory(self) -> Path:
        assert self.directory is not None and self.root is not None
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self.directory
        if directory.is_symlink():
            raise ValueError("Native cache namespace must not be a symlink")
        existed = directory.exists()
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.resolve().parent != self.root:
            raise ValueError("Native cache namespace escaped its configured root")
        marker = directory / "abstractcore-identity.json"
        if existed:
            if not marker.is_file() or marker.is_symlink():
                raise ValueError("Refusing an existing native cache namespace without its ownership marker")
            if marker.read_text() != self._encoded_identity:
                raise ValueError("Native cache identity marker mismatch")
        else:
            # The namespace is newly owned by this instance. Atomic exclusive
            # creation does not overwrite unrelated files or follow symlinks.
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(self._encoded_identity)
        if directory.stat().st_mode & 0o077:
            raise ValueError("Native cache namespace must be private (mode 0700)")
        if self._lock_fd is None:
            fd = os.open(directory / "abstractcore-owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException as exc:
                os.close(fd)
                raise RuntimeError("Native cache namespace is already owned by another live runtime; choose a different disk path or close its owner") from exc
            self._lock_fd = fd
        return directory

    def _release_lock(self):
        if self._lock_fd is not None:
            fd, self._lock_fd = self._lock_fd, None
            os.close(fd)

    def manager(self, request):
        if self._closed:
            raise RuntimeError("Native cache store is closed")
        if not getattr(request, "cache_key", None):
            return None
        if self._manager is not None:
            return self._manager
        from mlx_vlm.apc import APCManager, DiskBlockStore

        disk = None
        try:
            if self.root is not None:
                self._prepare_directory()
                disk = DiskBlockStore(
                    self.root, namespace=self.namespace, max_bytes=int(self.disk_max_gb * (1 << 30)),
                    num_workers=1, overrides={"disk_queue_max_gb": min(0.25, self.disk_max_gb / 4),
                                              "disk_shard_max_blocks": 32},
                )
            # Same pool shape and the same "only override what the operator
            # set" rule as NativeSession.prompt_cache(); see its docstring.
            overrides = {}
            if self.memory_max_gb is not None:
                overrides["memory_max_gb"] = self.memory_max_gb
            # This lane used to inherit mlx-vlm's
            # 2048 x 2 while the session lane ran 256 x 4, so the two managers
            # spent the same budget on different things and neither kept a
            # conversation's lineage across a tool loop. One shape now, sized and
            # justified in `NativeSession.prompt_cache`: one snapshot per call,
            # `CHECKPOINT_ENTRIES` distinct prompts restorable.
            from .mlx_native_session import (
                CHECKPOINT_ENTRIES,
                CHECKPOINT_GUARD_TOKENS,
                CHECKPOINT_INTERVAL_TOKENS,
            )

            for key, env, measured in (
                ("checkpoint_interval_tokens", "APC_CHECKPOINT_INTERVAL_TOKENS", CHECKPOINT_INTERVAL_TOKENS),
                ("checkpoint_entries", "APC_CHECKPOINT_ENTRIES", CHECKPOINT_ENTRIES),
                ("checkpoint_guard_tokens", "APC_CHECKPOINT_GUARD_TOKENS", CHECKPOINT_GUARD_TOKENS),
            ):
                if env not in os.environ:
                    overrides[key] = measured
            self._manager = APCManager(
                num_blocks=256, block_size=256, disk=disk, overrides=overrides,
            )
        except BaseException:
            if disk is not None:
                disk.close()
            self._release_lock()
            raise
        return self._manager

    def _dispose(self):
        manager, self._manager = self._manager, None
        if manager is not None:
            disk = getattr(manager, "disk", None)
            try:
                if disk is not None:
                    disk.flush()
            finally:
                try:
                    manager.close()
                finally:
                    manager.clear()

    def clear(self):
        """Invalidate this whole runtime namespace, both RAM and persisted KV.

        Caller must own the runtime control barrier. This is intentionally an
        all-keys operation; the provider reports that scope to its caller.
        Only upstream canonical cache files within our identity-marked namespace
        are removed. Model files and other cache namespaces are never targets.
        """
        if self._closed:
            raise RuntimeError("Native cache store is closed")
        self._dispose()
        removed = 0
        if self.directory is not None and self.directory.exists():
            directory = self._prepare_directory()
            from mlx_vlm.apc import DiskBlockStore
            for entry in directory.iterdir():
                if entry.is_symlink():
                    raise ValueError("Refusing symlink inside native cache namespace")
                if entry.is_file() and entry.suffix == ".safetensors" and DiskBlockStore._is_canonical_store_file(entry):
                    entry.unlink()
                    removed += 1
        return {"scope": "all_runtime_keys", "disk_files_removed": removed}

    def stats(self):
        return {"backend": "mlx_vlm_apc", "full_history_required": True,
                "disk_enabled": self.root is not None, "namespace": self.namespace,
                "memory_max_gb": self.memory_max_gb, "disk_max_gb": self.disk_max_gb,
                "stats": self._manager.stats_snapshot() if self._manager is not None else {}}

    def close(self):
        if not self._closed:
            try:
                self._dispose()
            finally:
                self._release_lock()
                self._closed = True
