"""Wall-clock budgets in the AbstractCore tests: reported by default, enforced on request.

Operator rule (2026-11-03, "CI/CD fail should be the exception, not the normal path"; the gateway's
`tests/timing.py` since 2026-10-27): a test never fails because a shared CI runner or a busy Mac is
slow. A test keeps its STRUCTURAL assertion unconditionally: the call was cancelled, an event has
not fired yet, a thread is still running, or the elapsed time is below the natural duration of the
thing the code must not wait for (a child's `sleep 40`). Its latency budget goes through
`timing_budget`:

- by default it prints one `[timing] <what>: <actual> vs limit <limit>` line and never fails;
- with `ABSTRACTCORE_TIMING_STRICT=1` in the environment it asserts the budget (a quiet machine,
  measuring a latency change on purpose).

`patient(timeout_s)` is for a wait whose expiry fails the test (`Event.wait`, `join`, a registry
`wait`): x5 unless strict, so a slow machine waits longer instead of failing.
"""
from __future__ import annotations

import os
import sys

STRICT_ENV = "ABSTRACTCORE_TIMING_STRICT"
PATIENCE = 5.0


def timing_strict() -> bool:
    return str(os.environ.get(STRICT_ENV) or "").strip() == "1"


def _fmt(value: float, unit: str) -> str:
    if unit == "ms":
        return f"{value:.1f} ms"
    if unit == "s":
        return f"{value * 1000:.1f} ms" if value < 1.0 else f"{value:.2f} s"
    return f"{value:g} {unit}"


def timing_budget(actual: float, limit: float, what: str, *, unit: str = "s", inclusive: bool = False) -> bool:
    """`actual < limit` (`<=` when `inclusive`): asserted under ABSTRACTCORE_TIMING_STRICT=1, otherwise
    reported on stderr. Returns whether the budget held."""
    ok = actual <= limit if inclusive else actual < limit
    line = f"[timing] {what}: {_fmt(actual, unit)} vs limit {_fmt(limit, unit)}{'' if ok else ' (OVER)'}"
    if timing_strict():
        assert ok, line
    print(line, file=sys.stderr, flush=True)
    return ok


def patient(timeout_s: float) -> float:
    """A wait bound that only a hang should reach: x5 unless ABSTRACTCORE_TIMING_STRICT=1."""
    return float(timeout_s) if timing_strict() else float(timeout_s) * PATIENCE
