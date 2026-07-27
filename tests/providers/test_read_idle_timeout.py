"""read_idle_timeout_s: the no-progress socket bound, distinct from the total.

FACE 2 (runtime routing c5004, confirmed c5041): a stalled LLM STREAM held a
worker for up to the full timeout (DEFAULT_LLM_TIMEOUT_S=7200s) because httpx's
`read` timeout — the max wait for the NEXT response chunk, i.e. the no-progress
bound — was set to the same 7200s total. `read_idle_timeout_s` separates the
read-idle bound from the total budget so a no-progress stream aborts at the
socket. None (default) = byte-unchanged (read == total, the prior behavior).
"""

from __future__ import annotations

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


def test_read_idle_separates_the_read_bound_from_the_total() -> None:
    p = _provider(timeout=7200.0, read_idle_timeout_s=120.0)
    assert p._read_idle_timeout == 120.0
    t = p.client.timeout
    # A stalled stream aborts at 120s (read); connect/write/pool keep the total.
    assert t.read == 120.0
    assert t.connect == 7200.0 and t.write == 7200.0 and t.pool == 7200.0


def test_read_idle_with_unlimited_total() -> None:
    # The FACE 2 core case: no total cap (None = unlimited) but a read-idle
    # bound — a no-progress stream still aborts, an unbounded-but-progressing
    # generation still completes.
    p = _provider(timeout=None, read_idle_timeout_s=120.0)
    t = p.client.timeout
    assert t.read == 120.0
    assert t.connect is None and t.write is None and t.pool is None


def test_read_idle_applies_to_async_client() -> None:
    p = _provider(timeout=7200.0, read_idle_timeout_s=90.0)
    assert p.async_client.timeout.read == 90.0
    assert p.async_client.timeout.connect == 7200.0


def test_nonpositive_read_idle_is_ignored() -> None:
    # A non-positive read-idle is not a "0s read-idle" (which would abort every
    # request instantly) — it means "unset", falling back to the total.
    p = _provider(timeout=7200.0, read_idle_timeout_s=0)
    assert p._read_idle_timeout is None
    assert p.client.timeout.read == 7200.0


def test_update_http_client_timeout_preserves_read_idle() -> None:
    p = _provider(timeout=600.0, read_idle_timeout_s=120.0)
    p._timeout = 1200.0
    p._update_http_client_timeout()
    t = p.client.timeout
    assert t.read == 120.0  # read-idle preserved
    assert t.connect == 1200.0  # new total applied to connect/write/pool
