"""Host identity for residency/memory records.

`get_host_identity()` names the machine a record was observed on so a future
multi-machine aggregator can merge records without schema breakage. Kept
deliberately dumb: a stable 12-hex hash of the hostname plus the hostname
itself, `kind: "local"` for records produced by this process. Never raises.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

_CACHED_IDENTITY: Optional[Dict[str, str]] = None


def _hostname() -> str:
    try:
        import socket

        name = str(socket.gethostname() or "").strip()
        if name:
            return name
    except Exception:
        pass
    try:
        import platform

        name = str(platform.node() or "").strip()
        if name:
            return name
    except Exception:
        pass
    return "unknown-host"


def get_host_identity() -> Dict[str, str]:
    """Stable identity of THIS host.

    Shape: {"host_id": <12-hex sha256 of hostname>, "host_name": <hostname>,
    "kind": "local"}. Stable across calls within a process (cached) and across
    processes on the same host (pure function of the hostname). Never raises.
    """
    global _CACHED_IDENTITY
    if _CACHED_IDENTITY is not None:
        return dict(_CACHED_IDENTITY)
    name = _hostname()
    host_id = hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest()[:12]
    _CACHED_IDENTITY = {"host_id": host_id, "host_name": name, "kind": "local"}
    return dict(_CACHED_IDENTITY)
