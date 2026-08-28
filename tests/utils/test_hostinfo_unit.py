"""Unit tests for `abstractcore.utils.hostinfo` and the memory-snapshot host block."""

from __future__ import annotations

import hashlib

from abstractcore.utils import hostinfo as hostinfo_mod
from abstractcore.utils.hostinfo import get_host_identity
from abstractcore.utils.memory import get_memory_snapshot


def test_host_identity_shape_and_stability() -> None:
    first = get_host_identity()
    second = get_host_identity()

    assert set(first.keys()) == {"host_id", "host_name", "kind"}
    assert first == second  # stable across calls
    assert first["kind"] == "local"
    assert isinstance(first["host_name"], str) and first["host_name"]
    assert isinstance(first["host_id"], str)
    assert len(first["host_id"]) == 12
    int(first["host_id"], 16)  # 12-hex


def test_host_identity_is_hash_of_hostname(monkeypatch) -> None:
    monkeypatch.setattr(hostinfo_mod, "_CACHED_IDENTITY", None)
    monkeypatch.setattr(hostinfo_mod, "_hostname", lambda: "test-host-9")

    identity = get_host_identity()

    assert identity["host_name"] == "test-host-9"
    assert identity["host_id"] == hashlib.sha256(b"test-host-9").hexdigest()[:12]


def test_host_identity_result_is_a_copy(monkeypatch) -> None:
    monkeypatch.setattr(hostinfo_mod, "_CACHED_IDENTITY", None)
    first = get_host_identity()
    first["host_id"] = "tampered"

    assert get_host_identity()["host_id"] != "tampered"


def test_host_identity_never_raises_without_hostname(monkeypatch) -> None:
    import socket
    import platform

    monkeypatch.setattr(hostinfo_mod, "_CACHED_IDENTITY", None)

    def _boom() -> str:
        raise RuntimeError("no hostname")

    monkeypatch.setattr(socket, "gethostname", _boom)
    monkeypatch.setattr(platform, "node", _boom)

    identity = get_host_identity()
    assert identity["host_name"] == "unknown-host"
    assert len(identity["host_id"]) == 12

    # Do not leave the fallback cached for other tests in this process.
    hostinfo_mod._CACHED_IDENTITY = None


def test_memory_snapshot_carries_host_block() -> None:
    snap = get_memory_snapshot()

    assert snap["host"] == get_host_identity()
