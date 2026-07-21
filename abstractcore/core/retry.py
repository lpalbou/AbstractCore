"""
Production-ready retry strategies for AbstractCore.

Implements SOTA exponential backoff with jitter and circuit breaker patterns
based on 2025 best practices from AWS Architecture Blog, Tenacity principles,
and production LLM system requirements.
"""

import time
import random
import threading
from typing import Type, Optional, Set, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)

# Cancellable backoff waits in bounded slices so a host's cancel/stop signal is
# observed within ~1s even mid-backoff (the interruptible-sleep discipline).
_BACKOFF_SLICE_S = 1.0


class RetryCancelledError(Exception):
    """Raised when a host cancels a retry sequence mid-backoff.

    Carries the underlying provider error as `last_error` so callers still see
    WHY retries were running. Deliberately not a ProviderError subclass: a
    cancel is a host decision, not a provider failure class, and it must never
    be re-classified as retryable by an outer retry layer.
    """

    def __init__(self, message: str, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.last_error = last_error


class RetryableErrorType(Enum):
    """Types of errors that can be retried."""
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior following SOTA best practices."""

    # Basic retry settings
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0     # seconds
    exponential_base: float = 2.0

    # Jitter type - using full jitter as AWS recommends
    use_jitter: bool = True

    # Circuit breaker settings
    failure_threshold: int = 5  # failures before opening circuit
    recovery_timeout: float = 60.0  # seconds before trying half-open
    half_open_max_calls: int = 3  # calls to test in half-open state

    # Total wall-clock budget across the WHOLE retry sequence (seconds).
    # None (default) preserves historical behavior: attempts × per-attempt
    # timeout can stack (3 × a 600s config default = 30 minutes on a wedged
    # endpoint — the 2026-07-21 entity-visit incident). When set, no RETRY
    # starts once the elapsed wall clock exceeds the budget: the first
    # attempt always runs (the budget bounds retries, never legitimate long
    # generations), and exhaustion raises the last error loudly with the
    # budget named. Interactive lanes (entity visits, UIs) should set this
    # to their user-facing patience window.
    max_total_wall_clock_s: Optional[float] = None

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt with exponential backoff and full jitter.

        Uses full jitter strategy as recommended by AWS:
        delay = random(0, min(cap, base * 2^attempt))

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds with jitter applied
        """
        # Calculate exponential backoff
        exponential_delay = self.initial_delay * (self.exponential_base ** (attempt - 1))

        # Cap the delay
        capped_delay = min(exponential_delay, self.max_delay)

        if self.use_jitter:
            # Full jitter: random between 0 and capped_delay
            return random.uniform(0, capped_delay)
        else:
            return capped_delay

    @classmethod
    def single_attempt(cls, **overrides) -> "RetryConfig":
        """Preset for hosts whose OUTER retry layer owns attempts (double-stack collapse).

        `create_llm(..., retry_config=RetryConfig.single_attempt())` makes the provider
        perform exactly ONE attempt per call — no inner resamples, no backoff sleeps —
        while the circuit breaker stays active (fleet failure state is still tracked).
        The consuming host's own retry policy (e.g. a runtime EffectPolicy) then owns
        attempt counts and backoff in one place. Entity-topology plan item 12 (C3),
        consumer-confirmed shape (runtime factory).
        """
        params: Dict[str, Any] = {"max_attempts": 1}
        params.update(overrides)
        return cls(**params)


class CircuitBreaker:
    """
    Circuit breaker implementation preventing cascade failures.

    Based on Netflix Hystrix and production patterns from legacy code.
    Follows the 3-state pattern: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    """

    def __init__(self, config: RetryConfig):
        """Initialize circuit breaker with configuration."""
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

    def record_success(self):
        """Record successful call and potentially close circuit."""
        if self.state == CircuitState.HALF_OPEN:
            # Half-open test succeeded, close circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            logger.info("Circuit breaker closed after successful recovery")
        elif self.state == CircuitState.CLOSED:
            # Decay failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record failed call and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

        elif self.state == CircuitState.HALF_OPEN:
            # Half-open test failed, reopen circuit
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened after failure in half-open state")

    def can_execute(self) -> bool:
        """Check if execution is allowed by circuit breaker."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def get_state_info(self) -> Dict[str, Any]:
        """Get circuit breaker state information for events/logging."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "half_open_calls": self.half_open_calls,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class RetryManager:
    """
    Central retry manager with smart error classification and circuit breakers.

    Implements production-ready retry patterns following SOTA best practices:
    - Exponential backoff with full jitter (AWS recommended)
    - Circuit breaker pattern for cascade failure prevention
    - Smart error classification (retry vs non-retry errors)
    - Comprehensive event emission for observability
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry manager with configuration."""
        self.config = config or RetryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        # Optional per-endpoint damping domain (core/endpoint_damping.py, opt-in):
        # when set, its SHARED breaker replaces the per-instance one and backoff
        # waits require a retry slot from its budget (fail fast when exhausted).
        self.damping_domain = None

        # Define retryable vs non-retryable error types
        self.retryable_errors = {
            "RateLimitError",
            "ProviderAPIError",
            "TimeoutError",
            "ConnectionError",
            "HTTPError",
            "ValidationError",
            "JSONDecodeError"
        }

        self.non_retryable_errors = {
            "AuthenticationError",
            "InvalidRequestError",
            "ModelNotFoundError",
            "UnsupportedFeatureError",
            "ConfigurationError"
        }

    def get_circuit_breaker(self, key: str) -> CircuitBreaker:
        """Get or create circuit breaker for a provider/model key."""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = CircuitBreaker(self.config)
        return self.circuit_breakers[key]

    def classify_error(self, error: Exception) -> RetryableErrorType:
        """
        Classify error type for appropriate retry strategy.

        Based on SOTA practices for LLM API error handling:
        - Rate limits: Always retry with backoff
        - Timeouts/Network: Retry with backoff
        - API errors: Retry once for transient issues
        - Auth/Invalid: Never retry
        """
        error_type_name = type(error).__name__
        error_str = str(error).lower()

        # Check explicit error types first
        if error_type_name in self.non_retryable_errors:
            return RetryableErrorType.UNKNOWN  # Will not be retried

        if "rate limit" in error_str or "429" in error_str or error_type_name == "RateLimitError":
            return RetryableErrorType.RATE_LIMIT
        elif "timeout" in error_str or "timed out" in error_str:
            return RetryableErrorType.TIMEOUT
        elif "network" in error_str or "connection" in error_str:
            return RetryableErrorType.NETWORK
        elif error_type_name == "ValidationError" or error_type_name == "JSONDecodeError":
            return RetryableErrorType.VALIDATION_ERROR
        elif "validation" in error_str or "invalid json" in error_str or "json" in error_str:
            return RetryableErrorType.VALIDATION_ERROR
        elif error_type_name in self.retryable_errors:
            return RetryableErrorType.API_ERROR
        else:
            return RetryableErrorType.UNKNOWN

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if error should be retried based on type and attempt count.

        Implements smart retry logic:
        - Rate limits: Retry up to max_attempts with longer delays
        - Timeouts/Network: Retry up to max_attempts
        - Validation errors: Retry up to max_attempts with feedback
        - API errors: Retry once for transient issues
        - Others: No retry
        """
        error_type = self.classify_error(error)

        if attempt >= self.config.max_attempts:
            return False

        if error_type in [RetryableErrorType.RATE_LIMIT, RetryableErrorType.TIMEOUT, RetryableErrorType.NETWORK]:
            return True
        elif error_type == RetryableErrorType.VALIDATION_ERROR:
            return True  # Retry validation errors up to max_attempts
        elif error_type == RetryableErrorType.API_ERROR:
            return attempt < 2  # Retry once for API errors
        else:
            return False  # No retry for unknown/non-retryable errors

    def execute_with_retry(self, func, *args, provider_key: str = "default",
                           cancel_event: Optional[threading.Event] = None, **kwargs):
        """
        Execute function with retry logic and circuit breaker protection.

        Args:
            func: Function to execute
            provider_key: Key for circuit breaker (e.g., "openai:gpt-4")
            cancel_event: Optional host cancel signal. Backoff waits are sliced
                (~1s) and check it; when set mid-backoff the sequence raises
                RetryCancelledError immediately (labeled, never a silent absorb).
                Default None preserves historical uninterruptible behavior... in
                semantics only: the wait itself is now sliced either way, which is
                behavior-identical for callers without a cancel signal.
            *args, **kwargs: Arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            RetryCancelledError: If cancel_event is set during backoff
            Last exception: If all retries fail
        """
        damping = self.damping_domain
        # With damping on, the ENDPOINT's shared breaker is the failure state —
        # the first instance to trip it stops every other instance on the same
        # endpoint (per-instance breakers are the herd mechanism this replaces).
        circuit_breaker = damping.breaker if damping is not None else self.get_circuit_breaker(provider_key)
        last_error = None

        # Check circuit breaker before starting
        if not circuit_breaker.can_execute():
            from ..exceptions import ProviderAPIError
            scope = f"endpoint {damping.label}" if damping is not None else provider_key
            raise ProviderAPIError(f"Circuit breaker open for {scope}")

        # Handle edge case of zero max attempts
        if self.config.max_attempts <= 0:
            from ..exceptions import ProviderAPIError
            raise ProviderAPIError(f"Max attempts is {self.config.max_attempts}, cannot execute")

        sequence_started = time.monotonic()

        for attempt in range(1, self.config.max_attempts + 1):
            # Wall-clock budget (0817-adjacent, 2026-07-21 wedge incident):
            # per-attempt timeouts STACK across retries — 3 attempts against a
            # wedged endpoint at a 600s client timeout is 30 minutes of wall
            # clock with the caller stuck on "Thinking…". When a budget is
            # configured, retries stop the moment it is exceeded; the first
            # attempt is never budget-gated (legitimate long generations are
            # not retries).
            budget = self.config.max_total_wall_clock_s
            if (
                budget is not None
                and attempt > 1
                and (time.monotonic() - sequence_started) >= float(budget)
            ):
                logger.warning(
                    f"#FALLBACK retry wall-clock budget exhausted for {provider_key}: "
                    f"{time.monotonic() - sequence_started:.1f}s elapsed >= {float(budget):.1f}s "
                    f"budget before attempt {attempt}; raising last error instead of retrying."
                )
                self._emit_retry_event("RETRY_EXHAUSTED", {
                    "provider_key": provider_key,
                    "attempt": attempt - 1,
                    "error_type": self.classify_error(last_error).value if last_error is not None else "unknown",
                    "error": str(last_error),
                    "reason": "wall_clock_budget_exhausted",
                    "budget_seconds": float(budget),
                    "circuit_breaker_state": circuit_breaker.get_state_info(),
                })
                break
            # A cancel observed between attempts (e.g. set while the previous attempt
            # was in flight) stops the sequence before burning another call.
            if cancel_event is not None and cancel_event.is_set() and attempt > 1:
                raise RetryCancelledError(
                    f"[retry cancelled by host] {provider_key}: retry sequence cancelled "
                    f"before attempt {attempt}; last error: {last_error}",
                    last_error=last_error,
                )
            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record success in circuit breaker
                circuit_breaker.record_success()

                # Success after retry - no event needed (success is implicit)
                # SOTA approach: Only emit critical events (exhausted) and retry attempts

                return result

            except Exception as e:
                last_error = e
                error_type = self.classify_error(e)

                # Record failure in circuit breaker — TRANSIENT classes only.
                # Non-retryable failures (auth/invalid-request = caller bugs)
                # say nothing about endpoint health; counting them let one
                # misconfigured caller open a SHARED breaker under endpoint
                # damping and block every healthy instance on that endpoint
                # (adversarial find, 2026-07-13).
                if error_type is not RetryableErrorType.UNKNOWN:
                    circuit_breaker.record_failure()

                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.debug(f"Not retrying {error_type.value} error after attempt {attempt}: {e}")

                    # Emit retry exhausted event (critical for alerting)
                    self._emit_retry_event("RETRY_EXHAUSTED", {
                        "provider_key": provider_key,
                        "attempt": attempt,
                        "error_type": error_type.value,
                        "error": str(e),
                        "reason": "non_retryable_error",
                        "circuit_breaker_state": circuit_breaker.get_state_info()
                    })
                    break

                # This is the last attempt
                if attempt >= self.config.max_attempts:
                    logger.warning(f"All {self.config.max_attempts} attempts failed for {provider_key}")

                    # Emit retry exhausted event (critical for alerting)
                    self._emit_retry_event("RETRY_EXHAUSTED", {
                        "provider_key": provider_key,
                        "attempt": attempt,
                        "error_type": error_type.value,
                        "error": str(e),
                        "reason": "max_attempts_reached",
                        "circuit_breaker_state": circuit_breaker.get_state_info()
                    })
                    break

                # Calculate delay and emit retry event (minimal - only when we're actually retrying)
                delay = self.config.get_delay(attempt)

                # Retry-After honoring: when the server named its own wait (429/503),
                # that signal beats our jitter guess — capped by OUR max_delay so a
                # hostile/buggy header can never park a worker for hours.
                retry_after = getattr(e, "retry_after_s", None)
                if isinstance(retry_after, (int, float)) and retry_after >= 0:
                    delay = min(max(float(retry_after), delay), self.config.max_delay)

                logger.info(f"Retrying {provider_key} after {error_type.value} error (attempt {attempt}/{self.config.max_attempts}). "
                           f"Waiting {delay:.2f}s...")

                # Emit retry attempted event (minimal approach - includes all needed context)
                self._emit_retry_event("RETRY_ATTEMPTED", {
                    "provider_key": provider_key,
                    "current_attempt": attempt,
                    "max_attempts": self.config.max_attempts,
                    "error_type": error_type.value,
                    "delay_seconds": delay,
                    "circuit_breaker_state": circuit_breaker.get_state_info()
                })

                # Retry budget (damping only): a slot is required to WAIT for this
                # endpoint. Exhausted budget = fail fast with the labeled last error —
                # never queue (queueing would hide the pressure admission should see).
                if damping is not None:
                    if not damping.try_acquire_retry_slot():
                        raise self._label_budget_exhausted(e, damping)
                    try:
                        self._wait_cancellable(delay, cancel_event)
                    finally:
                        damping.release_retry_slot()
                else:
                    # Wait before retry — in bounded slices so a host cancel signal is
                    # observed within ~1s instead of after the full (up to max_delay) sleep.
                    self._wait_cancellable(delay, cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    raise RetryCancelledError(
                        f"[retry cancelled by host] {provider_key}: retry sequence cancelled "
                        f"during backoff after attempt {attempt}; last error: {e}",
                        last_error=e,
                    )

        # All retries exhausted, raise the last error
        raise last_error

    @staticmethod
    def _label_budget_exhausted(error: Exception, damping) -> Exception:
        """Return the fail-fast error for an exhausted endpoint retry budget.

        Keeps the error's TYPE (so status-code-first classifiers see the same class)
        and carries status_code/retry_after_s through for ProviderError subclasses;
        anything unexpected falls back to the original error with a logged label.
        """
        from ..exceptions import ProviderError

        label = f"[retry budget exhausted for endpoint {damping.label}]"
        if isinstance(error, ProviderError):
            try:
                return type(error)(
                    f"{error} {label}",
                    status_code=getattr(error, "status_code", None),
                    retry_after_s=getattr(error, "retry_after_s", None),
                )
            except Exception:
                pass
        logger.warning("endpoint damping: %s (original error type %s)", label, type(error).__name__)
        return error

    @staticmethod
    def _wait_cancellable(delay: float, cancel_event: Optional[threading.Event]) -> None:
        """Backoff wait: ≤1s cancellable slices with a cancel signal; one plain
        sleep without (byte-identical legacy behavior for non-opting callers)."""
        if delay <= 0:
            return
        if cancel_event is None:
            time.sleep(delay)
            return
        deadline = time.monotonic() + delay
        while not cancel_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            # Event.wait returns True the moment the event is set — no oversleep.
            if cancel_event.wait(timeout=min(_BACKOFF_SLICE_S, remaining)):
                return

    def _emit_retry_event(self, event_type: str, data: Dict[str, Any]):
        """Emit retry-related events for observability."""
        try:
            from ..events import emit_global, EventType

            # Map our retry events to the minimal event types (SOTA approach)
            if event_type == "RETRY_ATTEMPTED":
                emit_global(EventType.RETRY_ATTEMPTED, data, source="RetryManager")
            elif event_type == "RETRY_EXHAUSTED":
                emit_global(EventType.RETRY_EXHAUSTED, data, source="RetryManager")
        except Exception as e:
            # Don't let event emission failures affect retry logic
            logger.debug(f"Failed to emit retry event: {e}")


# Global retry manager instance for convenience
default_retry_manager = RetryManager()