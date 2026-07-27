"""Shared httpx timeout construction (read-idle bound separated from the total).

FACE 2 (runtime routing c5004, confirmed c5041): a stalled LLM STREAM held a
worker up to the full timeout because httpx's `read` timeout — the max wait for
the NEXT chunk of the response body, i.e. the no-progress bound — was set to
the same value as the total budget (DEFAULT_LLM_TIMEOUT_S=7200s). Separating
the read-idle bound lets a no-progress stream abort at the socket.

One helper so every httpx-based provider (openai_compatible + its subclasses,
ollama, openai-native) builds the timeout the same way — no per-provider copy.
"""

from __future__ import annotations

from typing import Optional

import httpx


def build_read_idle_timeout(total: Optional[float], read_idle: Optional[float]) -> httpx.Timeout:
    """httpx.Timeout with `read` = the read-idle bound, others = the total.

    - `total`: the connect/write/pool budget (None = unlimited). Non-positive → None.
    - `read_idle`: the max seconds to wait for the NEXT response chunk (None or
      non-positive = unset → `read` falls back to `total`, i.e. BYTE-UNCHANGED
      from the prior single-value httpx.Client(timeout=total)).

    When read_idle is set, a no-progress stream aborts at `read` while a
    connecting/uploading request keeps the (possibly larger) total budget.
    """
    if total is not None and total <= 0:
        total = None
    if read_idle is not None and read_idle <= 0:
        read_idle = None
    read = read_idle if read_idle is not None else total
    # Every field explicit (each None = unlimited for that operation).
    return httpx.Timeout(connect=total, read=read, write=total, pool=total)


__all__ = ["build_read_idle_timeout"]
