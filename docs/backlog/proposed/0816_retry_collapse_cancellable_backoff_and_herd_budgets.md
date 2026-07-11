# Retry collapse — cancellable backoff, Retry-After honoring, shared endpoint damping (plan item 12 / core C3)

## Metadata
- Created: 2026-07-10
- Status: IMPLEMENTED (built 2026-07-10 on laurent's c398 approval; both reviewer
  asks were discharged pre-build — runtime c260, agency 0015/082837Z). Shipped as
  designed incl. the c365 rename (`endpoint_damping`); 16 tests in
  `tests/core/test_retry_collapse_c3.py`; full suite 1850/0. Uncommitted per the
  standing rule.
- Completed: 2026-07-10

## ADR status
- Governing ADRs: `docs/adr/0001-engineering-guardrails-and-no-silent-degradation.md`
- ADR impact: None expected (all changes opt-in or default-preserving; any behavior
  change is loud and documented)

## Context
Entity-topology consensus plan, item 12 (core C3): "Retry collapse: cancellable
backoff + jittered budgets so N runtimes cannot thundering-herd a hiccuping LLM
endpoint." Baseline handed over by the agency seat (agency-parity README, 0217a
findings, corrected per the Critic-4 audit 2026-07-09):

- **Two stacked retry loops**: the runtime's `RetryPolicy` (3 attempts) wraps
  abstractcore's per-provider `RetryManager` (up to 3 attempts on
  timeout/network/rate-limit) — a hung provider can bind a tick worker for up to
  9 × 7200s ≈ 18h with uninterruptible sleeps and no cancel preemption.
- Runtime-side mitigations already shipped (2026-07-09): deterministic 4xx fail
  once; truncation/structured-repair exhaustion non-retryable.
- Named-open items that land HERE: collapse the double stack (single-attempt
  provider retry config when constructed for runtime use), cancellable backoff.

Code facts verified 2026-07-10 (this repo, `abstractcore/core/retry.py` +
`abstractcore/providers/base.py`); independently re-verified by the agency seat
2026-07-10 (0015 file thread, 082837Z — all four facts confirmed from their bench):
1. `RetryManager.execute_with_retry` waits via `time.sleep(delay)` — the backoff
   is uninterruptible; nothing can cancel a provider mid-backoff
   (`abstractcore/core/retry.py:352`).
2. Full jitter EXISTS and is correct (`RetryConfig.get_delay`: AWS full-jitter,
   capped) — the "jittered" half of item 12 is per-attempt jitter, already
   shipped; what is missing is CROSS-INSTANCE coordination.
3. Circuit breakers are per provider INSTANCE: `self.retry_manager =
   RetryManager(...)` per provider object, keyed `class:model` (base.py:594-601).
   N provider instances pointing at ONE endpoint each hold an independent
   breaker — the fleet never shares failure state, so N runtimes discover a down
   endpoint N times, in parallel, with retries. This is the thundering-herd
   mechanism at the core layer.
4. No `Retry-After` honoring anywhere: 429s are retried on our own schedule even
   when the server names the wait it wants (verified: zero matches for
   retry[-_]after in abstractcore/).

## Design (four pieces, all core-side; the boundary pin below is load-bearing)

### 1. Cancellable backoff
`execute_with_retry(..., cancel_event: Optional[threading.Event] = None)` (also
threaded through `generate_with_telemetry` as a kwarg or construction param).
Backoff sleeps in bounded slices (≤1s) checking the event — the runtime's
interruptible-sleep lesson (loop-stop, 2026-07-09) applied at the provider
layer. On cancellation: raise the last error immediately, message labeled
`[retry cancelled by host]` — never a silent success/absorb. Default None
preserves today's behavior exactly.

### 2. Retry-After honoring (429/503)
`RateLimitError` (and `ProviderAPIError` where the server names a wait) gains an
optional `retry_after_s` extracted at `_raise_for_status` sites from the
`Retry-After` header / JSON body when present. `RetryManager` uses
`min(max(retry_after_s, computed_jitter), max_delay)` for that attempt — the
server's own signal beats our guess; the cap stays ours. Absent header =
unchanged behavior.

### 2b. status_code at native wrap sites (FOLDED INTO C3 — placement per
agency's 0015 ask; a rider on the same error-construction surfaces)
The native-OpenAI/Anthropic/remote-client error wrap sites still stringify the
HTTP status code away on some paths (openai native STREAMING is the named
case), which starves the runtime's status-code-first retry classifier — the
only 4xx gate once the double stack collapses (piece 4, consumer pin 2). Since
piece 2 already reworks error construction at these sites, C3 attaches
`status_code` to every typed provider error raised from a native/remote wrap
site. Acceptance: no provider error path constructs a typed exception from an
HTTP failure without carrying its status code.

### 3. Shared endpoint damping (per-process)
An opt-in process-wide registry of circuit breakers + retry budgets keyed by
`(base_url, model)` (relocation-stable: never handle/address-derived keys —
C1 pin; base_url here IS the endpoint identity, which is correct for damping).
- Shared breaker: N provider instances on one endpoint see ONE failure state —
  first instance to trip the breaker stops the other N-1 from burning retries.
- Retry budget: token-bucket per endpoint key (e.g. max concurrent
  retry-waiters); a request that cannot get a retry token fails fast with the
  last error labeled `[retry budget exhausted for endpoint]` instead of joining
  the herd.
- OPT-IN at construction (`endpoint_damping=True` or a registry handle; renamed
  from `shared_retry_domain` per the c365 renaming wave — "domain" named neither
  the mechanism nor the key, `endpoint_damping` says what it does) so
  library users' single-instance behavior never changes; the runtime factory /
  gateway host turn it on for fleet use.

### 4. Double-stack collapse (construction preset)
The stack collapse itself is the RUNTIME factory's call (their retry policy is
the outer loop and owns attempts). Core's half: a documented, named construction
preset — `retry_config=RetryConfig.single_attempt()` (max_attempts=1, breaker
still active) — so "provider retries OFF, runtime owns attempts" is one
readable line instead of folklore numbers. Timeout classification and the
Harmony carve-out (`ProviderAPIError` transient) are unaffected: with
max_attempts=1 the OUTER RetryPolicy still resamples them per ITS classifier
(the runtime carve-out already exists).

CONSUMER-CONFIRMED (runtime seat, commons c260, 2026-07-10): the preset is
exactly what `create_local_runtime` will consume (their
`RetryPolicy(llm_max_attempts=3, tool_max_attempts=1)` stays the one place
attempts and backoff live). Two pins from the consuming seat are DESIGN
CONSTRAINTS here, to be test-pinned at build time:
1. The Harmony carve-out SURVIVES the collapse: single-attempt inner + outer
   resample must still absorb the 400-signature race (the C2 harness pins the
   signature -> retryable mapping; add one collapse-mode test proving the
   preset + outer-resample path end to end).
2. Deterministic-4xx discipline: with inner retries gone, the runtime's
   fail-once classification is the ONLY gate against burning attempts — core
   must keep raising precisely typed errors (InvalidRequestError vs
   ProviderAPIError) so that gate keeps working; no double-classification
   remains anywhere.
Runtime also endorsed the C3/item-11 boundary pin from the R3/R4 seat: the
admission hook consumes budget-exhaustion fail-fasts as ordinary loud
failures, never a queue.

## The boundary pin (for the item-11 party — prevents a phase-4 collision)
Core C3 provides PER-PROCESS primitives only: cancellable waits, server-signal
honoring, shared damping within one process. CROSS-PROCESS fairness (N entity
loop processes + the gateway all dialing one endpoint) is item 11's admission
controller (runtime R3/R4 + gateway GW-D/GW-E + agency P3 + core C4): admission
decides WHO may call; C3 decides how a call that failed BEHAVES. Item 11's
honesty pin already says priority only governs traffic routed through the
shared admission point — the same honesty applies here: C3's shared damping
reaches only instances inside one process. Neither item should assume the other
covers its half; this section is the seam statement both build against.

## Non-goals
- No default behavior change for existing library users (everything opt-in or
  absent-input-preserving).
- No cross-process coordination in core (no lock files, no shared sockets —
  that is admission's lane).
- No queueing/prioritization (admission's lane); C3 fails fast when budgets are
  exhausted, loudly.

## Validation sketch
- Unit: cancel-during-backoff latency bound (cancel observed ≤1.2s into a 60s
  backoff); Retry-After=7 honored (waits ~7s not jitter); absent header
  unchanged; shared-domain breaker trips once for N instances (N-1 fail fast);
  budget exhaustion fails fast with the labeled message; `single_attempt()`
  preset makes exactly 1 HTTP call per generate.
- Herd proof (the live proof this item owes the 0014 ledger): N=8 concurrent
  generates against a stub endpoint returning 503s — WITHOUT shared damping:
  8×3 requests; WITH: first trip stops the rest (assert total requests bounded);
  wall-clock bounded by cancellable backoff.
