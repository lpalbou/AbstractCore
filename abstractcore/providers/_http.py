"""Shared httpx timeout construction (read-idle bound separated from the total).

FACE 2 (runtime routing c5004, confirmed c5041): a stalled LLM STREAM held a
worker up to the full timeout because httpx's `read` timeout — the max wait for
the NEXT chunk of the response body, i.e. the no-progress bound — was set to
the same value as the total budget (DEFAULT_LLM_TIMEOUT_S=7200s). Separating
the read-idle bound lets a no-progress stream abort at the socket.

FACE 2 REGRESSION (2026-08-02, LM Studio tool-call aborts — CONFIRMED): the
read-idle bound was applied at CLIENT construction, so it also governed
NON-STREAMING requests. For `stream: false` the server sends nothing until the
whole generation is finished, so httpx's `read` timeout degenerates into
TIME-TO-FIRST-BYTE — i.e. a hard cap on the ENTIRE generation, not an idle gap.
A 300s read-idle therefore silently capped every non-streaming call at 300s:
LM Studio logged `Client disconnected. Stopping generation...` at exactly
+300.0s and dropped the half-emitted tool call
(`Failed to parse tool call: Unexpected end of content`), while the runtime's
authoritative 7200s budget (ADR-0014) was never reached. That is precisely the
silent low timeout ADR-0027 §1/§2 forbids.

The bound is therefore STREAM-ONLY and must be opted into per request:
`build_read_idle_timeout(total, read_idle, streaming=True)` at `client.stream()`
call sites. The default (`streaming=False`) puts the authoritative total on
`read`, so a long non-streaming generation runs to completion.

One helper so every httpx-based provider (openai_compatible + its subclasses,
ollama, openai-native) builds the timeout the same way — no per-provider copy.
"""

from __future__ import annotations

from typing import Optional

import httpx


# #[WARNING:TIMEOUT] — ADR-0027 §4 (tagged timeout site), ADR-0027 §2 (no low
# defaults on correctness-critical LLM paths), ADR-0014 (abstractruntime owns
# the authoritative per-effect budget; nothing below it may cap lower silently).
def build_read_idle_timeout(
    total: Optional[float],
    read_idle: Optional[float],
    *,
    streaming: bool = False,
) -> httpx.Timeout:
    """httpx.Timeout with `read` = the read-idle bound, others = the total.

    - `total`: the connect/write/pool budget (None = unlimited). Non-positive → None.
      This is the runtime's authoritative per-effect budget (ADR-0014), normally
      7200s for local providers.
    - `read_idle`: the max seconds to wait for the NEXT response chunk (None or
      non-positive = unset → `read` falls back to `total`).
    - `streaming`: whether the response body arrives INCREMENTALLY. The read-idle
      bound is only meaningful when it does.

    Why `streaming` gates it (#[WARNING:TIMEOUT], ADR-0027 §2): httpx's `read`
    timeout is "max wait for the next chunk". On a streaming response that is a
    genuine no-progress bound. On a NON-streaming response the first chunk only
    arrives once generation has fully finished, so `read` becomes a cap on the
    whole generation — a 300s read-idle silently killed 5-minute-plus local tool
    calls (2026-08-02 LM Studio incident). Non-streaming requests therefore get
    the authoritative `total` on `read`; only streaming call sites opt in.
    """
    if total is not None and total <= 0:
        total = None
    if read_idle is not None and read_idle <= 0:
        read_idle = None
    read = read_idle if (streaming and read_idle is not None) else total
    # Every field explicit (each None = unlimited for that operation).
    return httpx.Timeout(connect=total, read=read, write=total, pool=total)


__all__ = ["build_read_idle_timeout"]
