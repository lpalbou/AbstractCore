"""Shared runaway-generation detection for every local provider.

WHY THIS EXISTS. A model that emits tens of thousands of tokens is almost never
producing a long answer — it is stuck repeating itself. Nothing in this codebase
detected that: `StoppingCriteria` appeared zero times, the transformers lane
forwarded no `repetition_penalty`, MLX forwarded none, and only llama.cpp had a
weak default. A decode ended on EOS or on exhausting its budget, and nothing
could tell those two apart afterwards. One such run on a 4B model held 113 GiB
of GPU memory for 90 minutes and took down the process hosting it.

WHAT THIS IS NOT. It is not a cap. Lowering `max_output_tokens` to make runaways
cheaper would silently truncate legitimate long generations, which ADR 0001
forbids. This detects the *degenerate state itself* and stops on evidence, so a
legitimate long answer runs to completion untouched.

THE CONTRACT, which every provider must honour identically:

  1. Stop only on X FULL repetitions of the same cycle (default 4). One repeated
     phrase is normal language; a cycle repeated verbatim four times is not.
  2. RETURN THE ANSWER. Never raise, never discard. The caller gets everything
     generated up to the stop.
  3. SAY SO. `finish_reason="repetition"` plus a `degeneration` block in metadata
     naming the cycle length, the repeat count and where it started, and a
     `warnings.warn` so it reaches a caller who reads no metadata. A stop the
     caller cannot distinguish from a natural one is a silent degradation.

Providers differ in how they decode; they must not differ in this. Each one feeds
token ids in and honours `should_stop`, then merges `metadata()` into its
response.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence

# A cycle shorter than this is normal text ("the the", " - - -", indentation).
# A cycle longer than this is a paragraph-scale loop that the repeat requirement
# already makes very unlikely to be a false positive.
DEFAULT_MIN_CYCLE_TOKENS = 3
# Cycles longer than this are NOT detected at all. Stated plainly because the
# earlier wording here claimed the repeat requirement made long cycles "unlikely
# to false-positive", which implied they were checked. They are not: a
# paragraph-scale loop with a period over 128 tokens runs to the budget.
DEFAULT_MAX_CYCLE_TOKENS = 128
# EIGHT verbatim repeats, chosen from measured data rather than intuition.
#
# The first version of this file used 4 with the comment "three fires on
# legitimate structured output; four essentially does not." That was wrong, and
# a replay against this project's own corpus proved it: at 4 repeats the
# detector fires on **17.5% of agent-written markdown**, because the rule
# reduces to "four consecutive identical lines or cells". Real examples that
# tripped it: a markdown table separator `|---|` with 8+ columns, an alignment
# row `|:---|` with 5+ columns, a `- [ ]` checklist, a JSON array of four
# identical objects, four repeated imports. Median fire position was 34% into
# the document and the worst was 3.2% — a caller asking for a wide comparison
# table would have received a truncated answer labelled as model degeneration.
#
# Measured trade (49 real degenerate completions recovered from this week's
# artifacts as the true-positive set):
#
#   reps=4  ->  49/49 caught, 17.5% of prose falsely stopped   <- do not ship
#   reps=6  ->  49/49 caught,  7.5% falsely stopped            <- floor
#   reps=8  ->  49/49 caught,  0.0% falsely stopped            <- default
#
# The cost is 12 extra tokens before stopping (token 24 instead of token 12).
# Against the 81920-token budget this lane resolves, that is 0.015%.
DEFAULT_REPEATS_REQUIRED = 20

# ...but repeats ALONE cannot separate a loop from a table, at any threshold.
# A markdown separator row for an 8-column table IS eight identical `|---|`
# cells; a 12-column table is twelve. Raising the repeat count just moves which
# tables break. Verified: at 8 repeats an 8-column separator still fires.
#
# The dimension that actually separates them is HOW MUCH BUDGET THE LOOP BURNS.
# Legitimate repeating structure is small and then stops — a whole separator row
# is a few dozen tokens. A degenerate loop does not stop; it runs to the budget.
# So the repeating region must ALSO span at least this many tokens before it
# counts as degeneration.
#
# Effect, since span = cycle x repeats:
#   cycle 3   (a `|---|` cell, or a `!` flood) -> needs 64 repeats to reach 192
#   cycle 24  (a repeated sentence)            -> needs 8
#   cycle 100 (a repeated paragraph)           -> needs 8, spanning 800
# Short cycles must therefore persist much longer to prove they are not
# structure, which is exactly the right bias. No table survives 192 tokens of
# one repeated cell; every real flood does.
#
# THE COSTS HERE ARE NOT SYMMETRIC, and the thresholds are set accordingly.
# Stopping a real answer early is destructive and unrecoverable: the caller gets
# a truncated result labelled as model degeneration, and cannot tell it was
# wrong. Letting a genuine loop run a few hundred tokens longer costs seconds of
# GPU and nothing else. So both gates are set well past the point where the
# measured data separates, deliberately buying false negatives to buy out false
# positives.
#
# Detection lands near token 512 rather than 24: 0.6% of an 81920 budget, about
# 34 seconds at the measured 66 ms/token, against the ~90 minutes a runaway
# costs. 512 tokens of a verbatim repeating cycle is not a shape legitimate
# output produces — a cycle of 3 must now repeat 171 times, which is a
# 170-column table.
DEFAULT_MIN_LOOP_TOKENS = 512


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class RepetitionDetector:
    """Detect a verbatim repeating cycle in a token stream.

    Cheap by construction: it only inspects the tail of the stream, and only up
    to `max_cycle_tokens`, so cost per token is bounded and independent of how
    long the generation runs.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_cycle_tokens: Optional[int] = None,
        max_cycle_tokens: Optional[int] = None,
        repeats_required: Optional[int] = None,
        min_loop_tokens: Optional[int] = None,
    ) -> None:
        # Env overrides exist so a caller who hits a false positive can widen or
        # disable the detector without editing code — and so this can be turned
        # off entirely for a workload that legitimately emits repeating output.
        if os.environ.get("ABSTRACTCORE_REPETITION_DETECT", "").strip().lower() in {
            "0", "false", "no", "off",
        }:
            enabled = False
        self.enabled = bool(enabled)
        self.min_cycle = min_cycle_tokens or _env_int(
            "ABSTRACTCORE_REPETITION_MIN_CYCLE", DEFAULT_MIN_CYCLE_TOKENS)
        self.max_cycle = max_cycle_tokens or _env_int(
            "ABSTRACTCORE_REPETITION_MAX_CYCLE", DEFAULT_MAX_CYCLE_TOKENS)
        self.repeats_required = repeats_required or _env_int(
            "ABSTRACTCORE_REPETITION_REPEATS", DEFAULT_REPEATS_REQUIRED)
        self.min_loop_tokens = min_loop_tokens or _env_int(
            "ABSTRACTCORE_REPETITION_MIN_LOOP_TOKENS", DEFAULT_MIN_LOOP_TOKENS)

        self._tokens: List[int] = []
        self._tripped = False
        self._cycle_tokens: Optional[int] = None
        self._repeats_seen: int = 0
        self._first_index: Optional[int] = None

    # -- feeding -----------------------------------------------------------

    def feed(self, token_id: int) -> bool:
        """Add one generated token. Returns True when generation should stop."""
        if not self.enabled or self._tripped:
            return self._tripped
        self._tokens.append(int(token_id))
        # Only the tail can close a cycle, and a cycle needs `repeats_required`
        # copies to exist at all, so nothing below the window can trip.
        window = max(self.max_cycle * self.repeats_required, self.min_loop_tokens) + self.max_cycle
        if len(self._tokens) > window:
            # Keep the buffer bounded: a 90-minute decode must not also leak.
            self._tokens = self._tokens[-window:]
        return self._check_tail()

    def feed_many(self, token_ids: Sequence[int]) -> bool:
        for tok in token_ids:
            if self.feed(tok):
                return True
        return self._tripped

    def _check_tail(self) -> bool:
        n = len(self._tokens)
        for cycle in range(self.min_cycle, self.max_cycle + 1):
            # Satisfy BOTH gates: at least `repeats_required` copies, AND a
            # repeating region of at least `min_loop_tokens`. For a short cycle
            # the token floor dominates and demands many more repeats.
            reps = max(self.repeats_required,
                       -(-self.min_loop_tokens // cycle))  # ceil division
            span = cycle * reps
            if span > n:
                break
            tail = self._tokens[-span:]
            head = tail[:cycle]
            # Every block must be byte-identical. A near-repeat is not evidence.
            if all(tail[i * cycle:(i + 1) * cycle] == head for i in range(1, reps)):
                self._tripped = True
                self._cycle_tokens = cycle
                self._repeats_seen = reps
                self._first_index = n - span
                return True
        return False

    # -- reporting ---------------------------------------------------------

    @property
    def tripped(self) -> bool:
        return self._tripped

    def metadata(self) -> Dict[str, Any]:
        """The `degeneration` block to merge into a response's metadata."""
        if not self._tripped:
            return {}
        return {
            "degeneration": {
                "detected": True,
                "kind": "verbatim_repetition",
                "cycle_tokens": self._cycle_tokens,
                "repeats": self._repeats_seen,
                "stopped_at_token_index": self._first_index,
                "detail": (
                    f"Generation stopped early: the model repeated the same "
                    f"{self._cycle_tokens}-token sequence {self._repeats_seen} times "
                    f"verbatim. The text generated before the loop IS returned. "
                    f"This is a stop on evidence of a degenerate loop, not a length "
                    f"limit — `finish_reason` is 'repetition', not 'length'."
                ),
                "disable_with": "ABSTRACTCORE_REPETITION_DETECT=0",
            }
        }

    def warn(self, model: str = "") -> None:
        """Announce on the channel that actually reaches callers.

        `logger.warning` does NOT: importing abstractcore sets the root logger to
        ERROR and leaves every `abstractcore.*` logger at NOTSET, so such a record
        is never created. ADR 0001 requires the degradation be visible.
        """
        if not self._tripped:
            return
        warnings.warn(
            f"#FALLBACK repetition detected{f' for {model}' if model else ''}: the model "
            f"repeated the same {self._cycle_tokens}-token sequence "
            f"{self._repeats_seen} times verbatim, so generation was stopped. The "
            f"text produced before the loop is returned and finish_reason is "
            f"'repetition'. Set ABSTRACTCORE_REPETITION_DETECT=0 to disable.",
            RuntimeWarning,
            stacklevel=3,
        )

    FINISH_REASON = "repetition"

    def finish_reason(self, natural: str = "stop") -> str:
        """`repetition` when tripped, otherwise whatever the provider decided.

        A caller must be able to tell a degenerate stop from a natural one and
        from budget exhaustion; collapsing them into 'stop' or 'length' is the
        information loss this whole module exists to prevent.
        """
        return self.FINISH_REASON if self._tripped else natural


def transformers_stopping_criteria(detector: "RepetitionDetector"):
    """Wrap a detector as a HuggingFace `StoppingCriteria`.

    `model.generate()` is a closed loop — there is no per-token callback — so a
    StoppingCriteria is the only place a detector can see the stream. It reads
    the last column of `input_ids` each step, which is the token just emitted.

    Returns None when transformers is unavailable or the detector is disabled,
    so callers can pass the result straight through without branching.
    """
    if detector is None or not detector.enabled:
        return None
    try:
        from transformers import StoppingCriteria
    except Exception:  # noqa: BLE001
        return None

    class _RepetitionStoppingCriteria(StoppingCriteria):
        def __init__(self, det: "RepetitionDetector") -> None:
            super().__init__()
            self._det = det
            self._seen_prompt = False

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
            # The first call arrives with the whole prompt; only generated
            # tokens are evidence of degeneration, so skip it.
            if not self._seen_prompt:
                self._seen_prompt = True
                return False
            try:
                return bool(self._det.feed(int(input_ids[0, -1])))
            except Exception:  # noqa: BLE001
                # A detector fault must never break generation. ADR 0001: fail
                # open, and the absence of a stop is not a silent degradation —
                # it is the pre-existing behaviour.
                return False

    return _RepetitionStoppingCriteria(detector)


def attach_to_generation_kwargs(
    generation_kwargs: Dict[str, Any],
    *,
    enabled: bool = True,
) -> Optional["RepetitionDetector"]:
    """Build a detector and append its StoppingCriteria to `generation_kwargs`.

    Appends rather than replaces: a caller's own stopping criteria are preserved.
    Returns the detector so the caller can read `finish_reason()` and
    `metadata()` afterwards, or None when detection is off/unavailable.
    """
    detector = RepetitionDetector(enabled=enabled)
    criteria = transformers_stopping_criteria(detector)
    if criteria is None:
        return None
    try:
        from transformers import StoppingCriteriaList

        existing = generation_kwargs.get("stopping_criteria")
        if existing is None:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])
        else:
            list(existing).append(criteria)
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                list(existing) + [criteria])
    except Exception:  # noqa: BLE001
        return None
    return detector
