"""read_idle_timeout_s: the STREAM-ONLY no-progress socket bound.

FACE 2 (runtime routing c5004, confirmed c5041): a stalled LLM STREAM held a
worker for up to the full timeout (DEFAULT_LLM_TIMEOUT_S=7200s) because httpx's
`read` timeout — the max wait for the NEXT response chunk, i.e. the no-progress
bound — was set to the same 7200s total. `read_idle_timeout_s` separates the
read-idle bound from the total budget so a no-progress stream aborts at the
socket.

FACE 2 REGRESSION (2026-08-02, CONFIRMED): the bound was installed on the
shared httpx CLIENT, so it also governed NON-STREAMING requests. On a
`stream: false` call the body only starts arriving once generation has fully
finished, so httpx's `read` is TIME-TO-FIRST-BYTE — a cap on the whole
generation. The runtime's 300s default therefore killed every local tool call
longer than 5 minutes (LM Studio: `Client disconnected. Stopping generation...`
at exactly +300.0s, tool call dropped mid-write), below the authoritative 7200s
budget of ADR-0014 and in breach of ADR-0027 §2.

The bound is now STREAM-ONLY: the client carries read == total, and streaming
call sites opt in per request with `streaming=True`.
"""

from __future__ import annotations

from abstractcore.providers._http import build_read_idle_timeout
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

_BASE = "http://127.0.0.1:59999/v1"


def _provider(**kw) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        model="default", base_url=_BASE, api_key="x", validate_model_on_init=False, **kw
    )


def test_read_idle_unset_is_byte_unchanged() -> None:
    # No read_idle → the single total value on all four fields, identical to
    # the prior httpx.Client(timeout=total). Every existing consumer unchanged.
    p = _provider(timeout=7200.0)
    assert p._read_idle_timeout is None
    t = p.client.timeout
    assert t.read == 7200.0 and t.connect == 7200.0 and t.write == 7200.0 and t.pool == 7200.0


def test_client_never_caps_nonstreaming_reads_at_the_read_idle_bound() -> None:
    # REGRESSION GUARD (2026-08-02 LM Studio incident): the shared client is used
    # for non-streaming POSTs, where httpx `read` == time-to-first-byte == the
    # whole generation. It must carry the AUTHORITATIVE total, never the
    # read-idle bound, or long tool calls are silently aborted mid-generation.
    p = _provider(timeout=7200.0, read_idle_timeout_s=300.0)
    assert p._read_idle_timeout == 300.0
    t = p.client.timeout
    assert t.read == 7200.0, "client-level read must be the total, not the read-idle bound"
    assert t.connect == 7200.0 and t.write == 7200.0 and t.pool == 7200.0


def test_read_idle_separates_the_read_bound_from_the_total_when_streaming() -> None:
    # The face-2 protection survives: streaming call sites opt in per request.
    p = _provider(timeout=7200.0, read_idle_timeout_s=120.0)
    t = p._httpx_timeout(streaming=True)
    # A stalled stream aborts at 120s (read); connect/write/pool keep the total.
    assert t.read == 120.0
    assert t.connect == 7200.0 and t.write == 7200.0 and t.pool == 7200.0


def test_read_idle_with_unlimited_total() -> None:
    # The FACE 2 core case: no total cap (None = unlimited) but a read-idle
    # bound — a no-progress stream still aborts, an unbounded-but-progressing
    # generation still completes.
    p = _provider(timeout=None, read_idle_timeout_s=120.0)
    t = p._httpx_timeout(streaming=True)
    assert t.read == 120.0
    assert t.connect is None and t.write is None and t.pool is None
    # Non-streaming stays unlimited on every field.
    assert p.client.timeout.read is None


def test_async_client_is_also_nonstreaming_safe() -> None:
    p = _provider(timeout=7200.0, read_idle_timeout_s=90.0)
    assert p.async_client.timeout.read == 7200.0
    assert p.async_client.timeout.connect == 7200.0
    assert p._httpx_timeout(streaming=True).read == 90.0


def test_nonpositive_read_idle_is_ignored() -> None:
    # A non-positive read-idle is not a "0s read-idle" (which would abort every
    # request instantly) — it means "unset", falling back to the total.
    p = _provider(timeout=7200.0, read_idle_timeout_s=0)
    assert p._read_idle_timeout is None
    assert p.client.timeout.read == 7200.0
    assert p._httpx_timeout(streaming=True).read == 7200.0


def test_update_http_client_timeout_keeps_total_on_read() -> None:
    p = _provider(timeout=600.0, read_idle_timeout_s=120.0)
    p._timeout = 1200.0
    p._update_http_client_timeout()
    t = p.client.timeout
    assert t.read == 1200.0  # non-streaming: authoritative total, not the bound
    assert t.connect == 1200.0  # new total applied to connect/write/pool
    assert p._httpx_timeout(streaming=True).read == 120.0  # read-idle preserved


def test_helper_default_is_nonstreaming_safe() -> None:
    # The shared helper defaults to the SAFE shape so a new client-construction
    # call site cannot reintroduce the silent non-streaming cap by omission.
    assert build_read_idle_timeout(7200.0, 300.0).read == 7200.0
    assert build_read_idle_timeout(7200.0, 300.0, streaming=True).read == 300.0
