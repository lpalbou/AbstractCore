"""Live phase feedback for TEXT generation: prefill vs generation.

A caller that renders "Thinking…" while a 6k-token prompt is prefilled has no
way to tell a slow prefill from a slow decode, or a stuck call from a working
one. Providers that can honestly report the boundary (MLX: the native runtime's
per-snapshot counters and mlx-lm's `stream_generate` responses) call a
`progress_callback` with plain JSON-safe dicts, and the host (AbstractRuntime)
turns them into durable ledger events.

Contract (documented in docs/native-mlx-runtime.md and docs/prompt-caching.md):

    {
      "kind": "llm",                    # discriminator: never a media event
      "phase": "prefill"|"generate"|"complete",
      "provider": "mlx", "model": "...",
      "prompt_tokens": int,             # total prompt tokens for this call
      "cached_tokens": int,             # prompt tokens served from KV cache
      "fed_tokens": int,                # prompt tokens actually prefilled
      "generated_tokens": int,
      "elapsed_s": float,               # since the call entered the provider
      "ttft_s": float,                  # set from the first-token event on
      "tokens_per_second": float,       # generation tok/s
      "prompt_tokens_per_second": float,
      "prefill_processed_tokens": int,  # prefill events only: prompt tokens
                                        # already in the KV cache (restored +
                                        # fed so far), cumulative, monotone
      "prefill_tokens_per_second": float,  # prefill events only: rate over the
                                        # tokens NEWLY processed since the
                                        # first mid-prefill observation
      "event_index": int,               # 0-based, monotone within one call
      "final": bool,                    # true exactly once, on "complete"
    }

Only keys with a known value are present: a provider that cannot measure
`cached_tokens` omits it rather than reporting a plausible zero.

MID-PREFILL PROGRESS (`prefill_progress`). A lane that can observe its prompt
pass chunk by chunk (mlx-lm's `prompt_progress_callback`, mlx-vlm's chunked
prefill loop, the native runtime's batched `PromptProcessingBatch`, the
transformers 2,048-token chunk loop, llama.cpp's `n_batch` slices) reports
cumulative processed tokens; the emitter turns them into further
`phase: "prefill"` events at the same 0.5 s cadence, each carrying
`prefill_processed_tokens` next to `prompt_tokens`. Tokens restored from a
cache count as processed immediately (the start event carries them when the
lane already knows them). A lane that CANNOT observe its prefill (every HTTP
provider; an unchunked one-shot prefill) never calls it, and its events carry
no `prefill_processed_tokens` at all — a consumer then shows the total only.
No prefill event is ever emitted after the first token.

COST. Not a per-token stream: the emitter enforces a minimum wall-clock
interval between generation events (`min_interval_s`, default 0.5 s), so a
host persisting them writes about two records per second of decode, for the
WHOLE call. There is NO count-based cap of any kind. Progress events are never
truncated: a cap of 64 per call shipped in the first version, went dark after
~60 s of a multi-minute decode and read as a hang (operator, 2026-09-23) — the
same silent-truncation class ADR-0026 forbids. Do not reintroduce one.

A provider that has no true signal must emit nothing. There is no synthetic
phase in this module.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

PHASE_PREFILL = "prefill"
PHASE_GENERATE = "generate"
PHASE_COMPLETE = "complete"

EVENT_KIND = "llm"

DEFAULT_MIN_INTERVAL_S = 0.5

_ENV_MIN_INTERVAL = "ABSTRACTCORE_PROGRESS_MIN_INTERVAL_S"

_TEXT_PROGRESS_KWARGS = ("on_progress", "progress_callback", "progress_event_callback")

#: The single canonical kwarg name the provider seam sees. Host aliases are
#: normalized onto it at the BaseProvider boundary so no provider has to know
#: about three spellings and no strict SDK ever receives one.
PROGRESS_KWARG = "_text_progress_callback"

#: Provider-INTERNAL kwarg carrying `TextProgressEmitter.prefill_progress` from
#: a provider's generation loop into its own lane adapters (MLX: the native
#: runtime / mlx-vlm stream adapters). Never reaches a backend or an SDK.
PREFILL_PROGRESS_KWARG = "_acore_prefill_progress"


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default



def pop_text_progress_callback(kwargs: Dict[str, Any]) -> Optional[Callable[[Dict[str, Any]], None]]:
    """Remove every host progress-callback alias from ``kwargs``; return the first callable.

    Called at the provider boundary for TEXT requests: an unknown callable must
    never reach a strict provider SDK's request payload, and a provider that
    cannot report phases must not receive one at all.
    """

    found: Optional[Callable[[Dict[str, Any]], None]] = None
    for key in _TEXT_PROGRESS_KWARGS:
        value = kwargs.pop(key, None)
        if found is None and callable(value):
            found = value
    return found


def _positive_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _positive_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return number if number >= 0 else None


class TextProgressEmitter:
    """Rate-limited phase reporter for one text generation call (rate-limited by time only; never capped by count)."""

    def __init__(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        min_interval_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._callback = callback if callable(callback) else None
        self._provider = str(provider) if provider else None
        self._model = str(model) if model else None
        self._min_interval_s = (
            _float_env(_ENV_MIN_INTERVAL, DEFAULT_MIN_INTERVAL_S)
            if min_interval_s is None
            else float(min_interval_s)
        )
        self._clock = clock
        self._started_at = clock()
        self._last_emit_at: Optional[float] = None
        self._emitted = 0
        self._ttft_s: Optional[float] = None
        self._first_token_seen = False
        self._done = False
        #: Sticky prompt-side facts: the native lane only learns `cached_tokens`
        #: once prefill has run, and a later event must not un-report them.
        self._prompt_tokens: Optional[int] = None
        self._cached_tokens: Optional[int] = None
        self._fed_tokens: Optional[int] = None
        self._prompt_tps: Optional[float] = None
        #: Mid-prefill progress: the newest observed cumulative count, the one
        #: last EMITTED (monotone), and the first observation (clock, count)
        #: the prefill rate is measured from.
        self._prefill_observed: Optional[int] = None
        self._prefill_emitted: Optional[int] = None
        self._prefill_rate_base: Optional[tuple] = None

    # -- introspection -----------------------------------------------------
    @property
    def active(self) -> bool:
        return self._callback is not None

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def ttft_s(self) -> Optional[float]:
        return self._ttft_s

    # -- emission ----------------------------------------------------------
    def _remember(
        self,
        *,
        prompt_tokens: Any = None,
        cached_tokens: Any = None,
        fed_tokens: Any = None,
        prompt_tokens_per_second: Any = None,
    ) -> None:
        # PROMPT SIZE ONLY EVER GROWS. On a warm-cache delta feed the backend
        # reports the tokens IT processed (mlx-vlm answered `prompt_tokens: 1`
        # for a 779-token turn whose first 690 were restored and whose fed
        # suffix it had already consumed), and taking that at face value made
        # the line read "Generating · 1 prompt token" one event after "Prefill
        # · 779". The prefill number is computed from the whole rendered
        # prompt, so a smaller later claim is a narrower measurement, not news
        # — and the prompt rate that came with it describes that same narrower
        # slice, so it is rejected together with the count.
        value = _positive_int(prompt_tokens)
        authoritative = bool(value) and value >= (self._prompt_tokens or 0)
        if authoritative:
            self._prompt_tokens = value
        value = _positive_int(cached_tokens)
        if value is not None:
            self._cached_tokens = value
        value = _positive_int(fed_tokens)
        if value is not None:
            self._fed_tokens = value
        rate = _positive_float(prompt_tokens_per_second)
        if rate and (authoritative or prompt_tokens is None):
            self._prompt_tps = rate
        # `fed` is derivable whenever the other two are known; deriving it here
        # keeps every consumer from re-implementing the subtraction.
        if self._fed_tokens is None and self._prompt_tokens is not None and self._cached_tokens is not None:
            self._fed_tokens = max(0, self._prompt_tokens - self._cached_tokens)

    def _send(self, phase: str, extra: Dict[str, Any]) -> bool:
        if self._callback is None:
            return False
        now = self._clock()
        event: Dict[str, Any] = {
            "kind": EVENT_KIND,
            "phase": phase,
            "event_index": self._emitted,
            "elapsed_s": round(max(0.0, now - self._started_at), 3),
        }
        if self._provider:
            event["provider"] = self._provider
        if self._model:
            event["model"] = self._model
        if self._prompt_tokens is not None:
            event["prompt_tokens"] = self._prompt_tokens
        if self._cached_tokens is not None:
            event["cached_tokens"] = self._cached_tokens
        if self._fed_tokens is not None:
            event["fed_tokens"] = self._fed_tokens
        if self._prompt_tps is not None:
            event["prompt_tokens_per_second"] = round(self._prompt_tps, 2)
        if self._ttft_s is not None:
            event["ttft_s"] = round(self._ttft_s, 3)
        event.update(extra)
        self._emitted += 1
        self._last_emit_at = now
        try:
            self._callback(event)
        except Exception:
            # Observability must never break a generation. A host whose
            # callback raises simply stops getting phase feedback.
            self._callback = None
            return False
        return True

    def prefill(
        self,
        *,
        prompt_tokens: Any = None,
        cached_tokens: Any = None,
        fed_tokens: Any = None,
    ) -> bool:
        """Announce that prompt processing has started. Always emitted."""

        if self._callback is None or self._done or self._emitted:
            return False
        self._remember(prompt_tokens=prompt_tokens, cached_tokens=cached_tokens, fed_tokens=fed_tokens)
        extra: Dict[str, Any] = {"generated_tokens": 0}
        # Restored-from-cache tokens are processed the moment prefill starts.
        # Only a POSITIVE restored count is reported here: "0 / 5,642" frozen
        # through a prefill the lane cannot observe would read as a stall.
        if self._prompt_tokens and self._cached_tokens:
            processed = min(self._cached_tokens, self._prompt_tokens)
            self._prefill_observed = self._prefill_emitted = processed
            extra["prefill_processed_tokens"] = processed
        return self._send(PHASE_PREFILL, extra)

    def prefill_progress(
        self,
        *,
        processed_tokens: Any = None,
        fed_processed: Any = None,
        fed_total: Any = None,
        prompt_tokens: Any = None,
        cached_tokens: Any = None,
        force: bool = False,
    ) -> bool:
        """Report how much of the prompt is already in the KV cache.

        Two spellings, because lanes see different axes:

        * ``processed_tokens`` — cumulative position on the WHOLE prompt
          (restored prefix included): the native batch lane, transformers,
          llama.cpp.
        * ``fed_processed`` / ``fed_total`` — progress over the tokens THIS
          call feeds (mlx-lm's ``prompt_progress_callback(processed, total)``,
          mlx-vlm's prefill bar). They are placed on the prompt axis by adding
          the restored count; when that count is not known yet it is
          ``prompt_tokens - fed_total``. When neither is known the report is
          dropped — progress on an unknown axis is not a measurement.

        Emits a ``prefill`` event carrying ``prefill_processed_tokens`` at the
        same wall-clock cadence as generation events (never count-capped). The
        first report that reveals a restored prefix is emitted at once; any
        report after the first token is ignored.
        """

        if self._callback is None or self._done or self._first_token_seen:
            return False
        self._remember(prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
        processed = _positive_int(processed_tokens)
        if processed is None:
            done = _positive_int(fed_processed)
            if done is None:
                return False
            total_fed = _positive_int(fed_total)
            base = self._cached_tokens
            if base is None and total_fed is not None and self._prompt_tokens is not None:
                if self._prompt_tokens < total_fed:
                    return False  # the two counts disagree: say nothing
                base = self._prompt_tokens - total_fed
                self._cached_tokens = base
                self._fed_tokens = total_fed
            if base is None:
                return False
            processed = base + done
        if not self._prompt_tokens:
            return False  # "x / ?" is not a progress report
        processed = min(processed, self._prompt_tokens)
        now = self._clock()
        if self._prefill_rate_base is None:
            self._prefill_rate_base = (now, processed)
        if self._prefill_observed is None or processed > self._prefill_observed:
            self._prefill_observed = processed
        processed = self._prefill_observed
        if self._prefill_emitted is not None and processed <= self._prefill_emitted:
            return False
        reveals_restored = self._prefill_emitted is None and processed > 0 and bool(self._cached_tokens)
        if not (force or reveals_restored):
            if self._last_emit_at is not None and (now - self._last_emit_at) < self._min_interval_s:
                return False
        extra: Dict[str, Any] = {"generated_tokens": 0, "prefill_processed_tokens": processed}
        base_at, base_count = self._prefill_rate_base
        if processed > base_count and now > base_at:
            extra["prefill_tokens_per_second"] = round((processed - base_count) / (now - base_at), 2)
        self._prefill_emitted = processed
        return self._send(PHASE_PREFILL, extra)

    def generation(
        self,
        *,
        generated_tokens: Any,
        prompt_tokens: Any = None,
        cached_tokens: Any = None,
        fed_tokens: Any = None,
        tokens_per_second: Any = None,
        prompt_tokens_per_second: Any = None,
        force: bool = False,
    ) -> bool:
        """Report decode progress. The FIRST call marks ttft and always emits."""

        if self._callback is None or self._done:
            return False
        count = _positive_int(generated_tokens) or 0
        self._remember(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            fed_tokens=fed_tokens,
            prompt_tokens_per_second=prompt_tokens_per_second,
        )
        first = False
        if not self._first_token_seen and count > 0:
            self._first_token_seen = True
            self._ttft_s = max(0.0, self._clock() - self._started_at)
            first = True
        if not (first or force):
            if self._last_emit_at is not None and (self._clock() - self._last_emit_at) < self._min_interval_s:
                return False
        extra: Dict[str, Any] = {"generated_tokens": count}
        # A single token has no rate. mlx-vlm's first response divides one token
        # by a near-zero interval and reports 16,609 tok/s — a number a UI would
        # print verbatim. Report nothing until there is a second token.
        if count > 1:
            rate = _positive_float(tokens_per_second)
            if rate:
                extra["tokens_per_second"] = round(rate, 2)
            elif self._ttft_s is not None:
                decode_s = max(1e-6, self._clock() - self._started_at - self._ttft_s)
                extra["tokens_per_second"] = round((count - 1) / decode_s, 2)
        if first:
            extra["first_token"] = True
        return self._send(PHASE_GENERATE, extra)

    def complete(
        self,
        *,
        generated_tokens: Any = None,
        prompt_tokens: Any = None,
        cached_tokens: Any = None,
        fed_tokens: Any = None,
        tokens_per_second: Any = None,
        prompt_tokens_per_second: Any = None,
        finish_reason: Any = None,
    ) -> bool:
        """Terminal event. Always emitted exactly once."""

        if self._callback is None or self._done:
            return False
        self._remember(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            fed_tokens=fed_tokens,
            prompt_tokens_per_second=prompt_tokens_per_second,
        )
        count = _positive_int(generated_tokens) or 0
        extra: Dict[str, Any] = {"generated_tokens": count, "final": True}
        rate = _positive_float(tokens_per_second)
        if rate and count > 1:
            extra["tokens_per_second"] = round(rate, 2)
        if isinstance(finish_reason, str) and finish_reason.strip():
            extra["finish_reason"] = finish_reason.strip()
        sent = self._send(PHASE_COMPLETE, extra)
        self._done = True
        return sent
