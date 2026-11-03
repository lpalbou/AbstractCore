### Changed

- **The test suite never fails on a slow machine** (`tests/timing.py`, `ABSTRACTCORE_TIMING_STRICT`). Every
  elapsed-time assertion in `tests/` was audited. A "did not block" check became an event or state check:
  - the slot pool's dropped holder: the blocked work is still running when `gc.collect()` returns;
  - the cancelled slot request: at most one token after the cancel;
  - `execute_command`'s drains: below the child's own `sleep 30/40`, the wait the code must not do;
  - the retry that fails fast: below the 30 s backoff.

  Latency budgets (cancel promptness, ReDoS guards, streaming overheads, live-provider response times) go through
  `timing_budget`: one `[timing]` line on stderr by default, asserted only with `ABSTRACTCORE_TIMING_STRICT=1`,
  mirroring the gateway's `tests/timing.py`. The session's terminal summary lists every measure ("timing budgets",
  over-budget first), so the numbers are in every CI log. A blowup guard also passes `ceiling=`: a generous bound
  asserted always (a ReDoS regression is never a slow green). Test-only.
