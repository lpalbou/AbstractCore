"""
Production-ready retry strategies for AbstractLLM Core.

Implements SOTA exponential backoff with jitter and circuit breaker patterns
based on 2025 best practices from AWS Architecture Blog, Tenacity principles,
and production LLM system requirements.
"""

import time
import random
import logging
from typing import Type, Optional, Set, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


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

    def execute_with_retry(self, func, *args, provider_key: str = "default", **kwargs):
        """
        Execute function with retry logic and circuit breaker protection.

        Args:
            func: Function to execute
            provider_key: Key for circuit breaker (e.g., "openai:gpt-4")
            *args, **kwargs: Arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Last exception: If all retries fail
        """
        circuit_breaker = self.get_circuit_breaker(provider_key)
        last_error = None

        # Check circuit breaker before starting
        if not circuit_breaker.can_execute():
            from ..exceptions import ProviderAPIError
            raise ProviderAPIError(f"Circuit breaker open for {provider_key}")

        # Handle edge case of zero max attempts
        if self.config.max_attempts <= 0:
            from ..exceptions import ProviderAPIError
            raise ProviderAPIError(f"Max attempts is {self.config.max_attempts}, cannot execute")

        for attempt in range(1, self.config.max_attempts + 1):
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

                # Record failure in circuit breaker
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

                # Wait before retry
                time.sleep(delay)

        # All retries exhausted, raise the last error
        raise last_error

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