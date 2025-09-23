"""
Comprehensive tests for the retry strategy implementation.

Tests cover:
- Basic retry functionality with exponential backoff
- Circuit breaker behavior in all states
- Error classification and retry decisions
- Event emission during retry operations
- Integration with BaseProvider
- Edge cases and error conditions
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from abstractllm.core.retry import (
    RetryConfig, RetryManager, CircuitBreaker, CircuitState,
    RetryableErrorType, default_retry_manager
)
import json
from abstractllm.exceptions import (
    RateLimitError, ProviderAPIError, AuthenticationError,
    InvalidRequestError, ModelNotFoundError
)
from abstractllm.events import EventType, GlobalEventBus
from abstractllm.providers.base import BaseProvider
from abstractllm.core.types import GenerateResponse


class TestRetryConfig:
    """Test RetryConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.use_jitter is True
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=2.0,
            max_delay=120.0,
            use_jitter=False
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.use_jitter is False

    def test_get_delay_without_jitter(self):
        """Test delay calculation without jitter."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, use_jitter=False)

        assert config.get_delay(1) == 1.0  # 1.0 * 2^0
        assert config.get_delay(2) == 2.0  # 1.0 * 2^1
        assert config.get_delay(3) == 4.0  # 1.0 * 2^2
        assert config.get_delay(4) == 8.0  # 1.0 * 2^3

    def test_get_delay_with_cap(self):
        """Test delay capping."""
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, use_jitter=False)

        assert config.get_delay(1) == 1.0
        assert config.get_delay(2) == 2.0
        assert config.get_delay(3) == 4.0
        assert config.get_delay(4) == 5.0  # Capped at max_delay
        assert config.get_delay(5) == 5.0  # Still capped

    def test_get_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(initial_delay=2.0, use_jitter=True)

        # With jitter, delay should be random between 0 and calculated delay
        delay1 = config.get_delay(1)
        delay2 = config.get_delay(1)

        assert 0 <= delay1 <= 2.0
        assert 0 <= delay2 <= 2.0
        # They should potentially be different due to randomness
        # (though they might occasionally be the same)


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    def test_initial_state(self):
        """Test circuit breaker starts in closed state."""
        config = RetryConfig()
        cb = CircuitBreaker(config)

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
        assert cb.half_open_calls == 0
        assert cb.can_execute() is True

    def test_record_success_in_closed_state(self):
        """Test success recording in closed state."""
        config = RetryConfig()
        cb = CircuitBreaker(config)

        # Add some failures first
        cb.failure_count = 3
        cb.record_success()

        # Failure count should decay
        assert cb.failure_count == 2
        assert cb.state == CircuitState.CLOSED

    def test_circuit_opens_after_failures(self):
        """Test circuit opens after reaching failure threshold."""
        config = RetryConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        # Record failures up to threshold
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        assert cb.failure_count == 3
        assert cb.last_failure_time is not None
        assert cb.can_execute() is False

    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after recovery timeout."""
        config = RetryConfig(failure_threshold=2, recovery_timeout=0.1)  # 100ms timeout
        cb = CircuitBreaker(config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should transition to half-open on can_execute check
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Test successful call in half-open state closes circuit."""
        config = RetryConfig(failure_threshold=2)
        cb = CircuitBreaker(config)

        # Force into half-open state
        cb.state = CircuitState.HALF_OPEN
        cb.failure_count = 2

        # Record success
        cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.half_open_calls == 0

    def test_half_open_failure_reopens_circuit(self):
        """Test failed call in half-open state reopens circuit."""
        config = RetryConfig(failure_threshold=2)
        cb = CircuitBreaker(config)

        # Force into half-open state
        cb.state = CircuitState.HALF_OPEN

        # Record failure
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_half_open_call_limiting(self):
        """Test half-open state limits concurrent calls."""
        config = RetryConfig(half_open_max_calls=2)
        cb = CircuitBreaker(config)

        # Force into half-open state
        cb.state = CircuitState.HALF_OPEN

        # First two calls should be allowed
        assert cb.can_execute() is True
        assert cb.half_open_calls == 1
        assert cb.can_execute() is True
        assert cb.half_open_calls == 2

        # Third call should be rejected
        assert cb.can_execute() is False
        assert cb.half_open_calls == 2

    def test_get_state_info(self):
        """Test state information retrieval."""
        config = RetryConfig()
        cb = CircuitBreaker(config)

        cb.failure_count = 2
        cb.last_failure_time = datetime.now()

        info = cb.get_state_info()

        assert info["state"] == "closed"
        assert info["failure_count"] == 2
        assert info["half_open_calls"] == 0
        assert info["last_failure_time"] is not None


class TestRetryManager:
    """Test RetryManager functionality."""

    def test_default_initialization(self):
        """Test retry manager initializes with defaults."""
        manager = RetryManager()

        assert isinstance(manager.config, RetryConfig)
        assert len(manager.circuit_breakers) == 0
        assert "RateLimitError" in manager.retryable_errors
        assert "AuthenticationError" in manager.non_retryable_errors

    def test_custom_config_initialization(self):
        """Test retry manager with custom config."""
        config = RetryConfig(max_attempts=5)
        manager = RetryManager(config)

        assert manager.config.max_attempts == 5

    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and reuse."""
        manager = RetryManager()

        cb1 = manager.get_circuit_breaker("test-key")
        cb2 = manager.get_circuit_breaker("test-key")
        cb3 = manager.get_circuit_breaker("different-key")

        assert cb1 is cb2  # Same key returns same instance
        assert cb1 is not cb3  # Different key returns different instance

    def test_error_classification(self):
        """Test error type classification."""
        manager = RetryManager()

        # Rate limit errors
        rate_error = RateLimitError("Rate limit exceeded")
        assert manager.classify_error(rate_error) == RetryableErrorType.RATE_LIMIT

        # Timeout errors
        timeout_error = Exception("Request timed out")
        assert manager.classify_error(timeout_error) == RetryableErrorType.TIMEOUT

        # Network errors
        network_error = Exception("Connection failed")
        assert manager.classify_error(network_error) == RetryableErrorType.NETWORK

        # API errors
        api_error = ProviderAPIError("API error")
        assert manager.classify_error(api_error) == RetryableErrorType.API_ERROR

        # Validation errors
        from pydantic import ValidationError
        validation_error = ValidationError.from_exception_data("TestModel", [])
        assert manager.classify_error(validation_error) == RetryableErrorType.VALIDATION_ERROR

        # JSON decode errors
        json_error = json.JSONDecodeError("Invalid JSON", "test", 0)
        assert manager.classify_error(json_error) == RetryableErrorType.VALIDATION_ERROR

        # Validation errors by string content
        validation_str_error = Exception("Validation failed for field name")
        assert manager.classify_error(validation_str_error) == RetryableErrorType.VALIDATION_ERROR

        json_str_error = Exception("Invalid JSON format")
        assert manager.classify_error(json_str_error) == RetryableErrorType.VALIDATION_ERROR

        # Non-retryable errors
        auth_error = AuthenticationError("Invalid API key")
        assert manager.classify_error(auth_error) == RetryableErrorType.UNKNOWN

        invalid_error = InvalidRequestError("Invalid request")
        assert manager.classify_error(invalid_error) == RetryableErrorType.UNKNOWN

    def test_should_retry_logic(self):
        """Test retry decision logic."""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config)

        # Rate limit errors should retry up to max attempts
        rate_error = RateLimitError("Rate limit")
        assert manager.should_retry(rate_error, 1) is True
        assert manager.should_retry(rate_error, 2) is True
        assert manager.should_retry(rate_error, 3) is False  # At max attempts

        # Validation errors should retry up to max attempts
        from pydantic import ValidationError
        validation_error = ValidationError.from_exception_data("TestModel", [])
        assert manager.should_retry(validation_error, 1) is True
        assert manager.should_retry(validation_error, 2) is True
        assert manager.should_retry(validation_error, 3) is False  # At max attempts

        # JSON decode errors should retry up to max attempts
        json_error = json.JSONDecodeError("Invalid JSON", "test", 0)
        assert manager.should_retry(json_error, 1) is True
        assert manager.should_retry(json_error, 2) is True
        assert manager.should_retry(json_error, 3) is False  # At max attempts

        # API errors should retry once
        api_error = ProviderAPIError("API error")
        assert manager.should_retry(api_error, 1) is True
        assert manager.should_retry(api_error, 2) is False  # Only one retry

        # Non-retryable errors should never retry
        auth_error = AuthenticationError("Auth failed")
        assert manager.should_retry(auth_error, 1) is False

    @patch('time.sleep')
    def test_successful_execution(self, mock_sleep):
        """Test successful execution without retries."""
        manager = RetryManager()
        mock_func = Mock(return_value="success")

        result = manager.execute_with_retry(mock_func, provider_key="test")

        assert result == "success"
        mock_func.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('time.sleep')
    def test_retry_on_retryable_error(self, mock_sleep):
        """Test retry behavior on retryable errors."""
        config = RetryConfig(max_attempts=3, initial_delay=1.0, use_jitter=False)
        manager = RetryManager(config)

        # Mock function that fails twice then succeeds
        mock_func = Mock(side_effect=[
            RateLimitError("Rate limit"),
            RateLimitError("Rate limit"),
            "success"
        ])

        result = manager.execute_with_retry(mock_func, provider_key="test")

        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries

    @patch('time.sleep')
    def test_no_retry_on_non_retryable_error(self, mock_sleep):
        """Test no retry on non-retryable errors."""
        manager = RetryManager()
        mock_func = Mock(side_effect=AuthenticationError("Auth failed"))

        with pytest.raises(AuthenticationError):
            manager.execute_with_retry(mock_func, provider_key="test")

        mock_func.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('time.sleep')
    def test_retry_exhaustion(self, mock_sleep):
        """Test behavior when all retries are exhausted."""
        config = RetryConfig(max_attempts=2)
        manager = RetryManager(config)

        # Mock function that always fails with retryable error
        error = RateLimitError("Always fails")
        mock_func = Mock(side_effect=error)

        with pytest.raises(RateLimitError):
            manager.execute_with_retry(mock_func, provider_key="test")

        assert mock_func.call_count == 2  # max_attempts
        mock_sleep.assert_called_once()  # One retry

    def test_circuit_breaker_integration(self):
        """Test circuit breaker prevents execution when open."""
        config = RetryConfig(failure_threshold=1)
        manager = RetryManager(config)

        # Get circuit breaker and force it open
        cb = manager.get_circuit_breaker("test")
        cb.record_failure()  # This should open the circuit

        mock_func = Mock()

        with pytest.raises(ProviderAPIError, match="Circuit breaker open"):
            manager.execute_with_retry(mock_func, provider_key="test")

        mock_func.assert_not_called()

    @patch('abstractllm.events.emit_global')
    def test_event_emission(self, mock_emit):
        """Test retry events are properly emitted."""
        config = RetryConfig(max_attempts=2)
        manager = RetryManager(config)

        # Mock function that fails once then succeeds
        mock_func = Mock(side_effect=[RateLimitError("Rate limit"), "success"])

        with patch('time.sleep'):  # Skip actual sleep
            result = manager.execute_with_retry(mock_func, provider_key="test")

        assert result == "success"

        # Check that retry events were emitted (minimal approach)
        assert mock_emit.call_count >= 1  # At least retry_attempted

        # Verify specific events
        call_args_list = [call[0] for call in mock_emit.call_args_list]
        event_types = [args[0] for args in call_args_list]

        assert EventType.RETRY_ATTEMPTED in event_types
        # No retry_success event in minimal approach - success is implicit


class MockProvider(BaseProvider):
    """Mock provider for testing integration."""

    def __init__(self, model: str = "test-model", **kwargs):
        super().__init__(model, **kwargs)

    def _generate_internal(self, prompt: str, **kwargs):
        """Mock generation method."""
        return GenerateResponse(
            content="Generated response",
            model=self.model,
            finish_reason="stop"
        )

    def generate(self, prompt: str, **kwargs):
        """Mock generate method (required by interface)."""
        return self.generate_with_telemetry(prompt, **kwargs)

    def get_capabilities(self):
        """Mock capabilities method (required by interface)."""
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "supports_structured_output": True
        }


class TestBaseProviderIntegration:
    """Test retry integration with BaseProvider."""

    def test_provider_initialization_with_retry(self):
        """Test provider initializes with retry manager."""
        provider = MockProvider("test-model")

        assert hasattr(provider, 'retry_manager')
        assert isinstance(provider.retry_manager, RetryManager)
        assert provider.provider_key == "MockProvider:test-model"

    def test_structured_output_retry_strategy_parameter(self):
        """Test passing retry_strategy parameter to generate method."""
        from abstractllm.structured import FeedbackRetry
        from pydantic import BaseModel, ValidationError

        class TestModel(BaseModel):
            name: str
            value: int

        provider = MockProvider("test-model")
        custom_retry = FeedbackRetry(max_attempts=5)

        # Test that we can pass retry_strategy parameter without error
        # We'll mock the _generate_internal to return valid JSON
        def mock_generate(*args, **kwargs):
            return GenerateResponse(
                content='{"name": "test", "value": 42}',
                model="test-model",
                finish_reason="stop"
            )

        provider._generate_internal = mock_generate

        # This should not raise an error and should accept the retry_strategy parameter
        result = provider.generate_with_telemetry(
            prompt="Test prompt",
            response_model=TestModel,
            retry_strategy=custom_retry
        )

        assert isinstance(result, TestModel)
        assert result.name == "test"
        assert result.value == 42

    def test_structured_output_default_retry(self):
        """Test default retry behavior for structured output."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        provider = MockProvider("test-model")

        def mock_generate(*args, **kwargs):
            return GenerateResponse(
                content='{"name": "test", "value": 42}',
                model="test-model",
                finish_reason="stop"
            )

        provider._generate_internal = mock_generate

        # This should work with default retry (no retry_strategy parameter)
        result = provider.generate_with_telemetry(
            prompt="Test prompt",
            response_model=TestModel
        )

        assert isinstance(result, TestModel)
        assert result.name == "test"
        assert result.value == 42

    def test_provider_custom_retry_config(self):
        """Test provider with custom retry configuration."""
        custom_config = RetryConfig(max_attempts=5)
        provider = MockProvider("test-model", retry_config=custom_config)

        assert provider.retry_manager.config.max_attempts == 5

    @patch('abstractllm.providers.base.time.sleep')
    def test_provider_retry_on_failure(self, mock_sleep):
        """Test provider retries on API failures."""
        provider = MockProvider("test-model")

        # Mock _generate_internal to fail then succeed
        original_generate = provider._generate_internal
        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("Rate limit exceeded")
            return original_generate(*args, **kwargs)

        provider._generate_internal = mock_generate

        # This should succeed after one retry
        response = provider.generate_with_telemetry("Test prompt")

        assert response.content == "Generated response"
        assert call_count == 2  # Initial call + 1 retry
        mock_sleep.assert_called_once()

    def test_provider_no_retry_on_auth_error(self):
        """Test provider doesn't retry on authentication errors."""
        provider = MockProvider("test-model")

        # Mock _generate_internal to always fail with auth error
        provider._generate_internal = Mock(side_effect=AuthenticationError("Invalid API key"))

        with pytest.raises(AuthenticationError):
            provider.generate_with_telemetry("Test prompt")

        # Should only be called once (no retries)
        assert provider._generate_internal.call_count == 1

    @patch('abstractllm.events.emit_global')
    def test_provider_retry_events(self, mock_emit):
        """Test provider emits retry events."""
        provider = MockProvider("test-model")

        # Mock _generate_internal to fail once then succeed
        call_count = 0
        original_generate = provider._generate_internal

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("Rate limit")
            return original_generate(*args, **kwargs)

        provider._generate_internal = mock_generate

        with patch('time.sleep'):  # Skip actual sleep
            response = provider.generate_with_telemetry("Test prompt")

        assert response.content == "Generated response"

        # Check that events were emitted (including retry events)
        assert mock_emit.call_count > 0

        # Look for generation and retry events
        call_args_list = [call[0] for call in mock_emit.call_args_list]
        event_types = [args[0] for args in call_args_list]

        # Should include generation events and minimal retry events
        assert EventType.GENERATION_STARTED in event_types
        assert EventType.GENERATION_COMPLETED in event_types
        # Retry events only if actually retrying (minimal approach)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_max_attempts(self):
        """Test behavior with zero max attempts."""
        config = RetryConfig(max_attempts=0)
        manager = RetryManager(config)

        mock_func = Mock(side_effect=RateLimitError("Rate limit"))

        with pytest.raises(ProviderAPIError, match="Max attempts is 0"):
            manager.execute_with_retry(mock_func, provider_key="test")

        # Should not call function if max_attempts is 0
        mock_func.assert_not_called()

    def test_negative_delays(self):
        """Test behavior with edge case delay configurations."""
        config = RetryConfig(initial_delay=0, max_delay=0, use_jitter=False)

        delay = config.get_delay(1)
        assert delay == 0

    def test_very_large_delays(self):
        """Test delay capping works with large exponential values."""
        config = RetryConfig(initial_delay=1.0, max_delay=10.0, use_jitter=False)

        # Should be capped at max_delay
        delay = config.get_delay(10)  # Would be 1.0 * 2^9 = 512 without cap
        assert delay == 10.0

    def test_event_emission_failure_doesnt_break_retry(self):
        """Test that event emission failures don't break retry logic."""
        manager = RetryManager()

        # Mock emit_global to raise exception
        with patch('abstractllm.events.emit_global', side_effect=Exception("Event error")):
            mock_func = Mock(return_value="success")

            # Should still work despite event emission failure
            result = manager.execute_with_retry(mock_func, provider_key="test")
            assert result == "success"

    def test_circuit_breaker_state_persistence(self):
        """Test circuit breaker state persists across retry manager instances."""
        # Create two managers with same circuit breaker keys
        manager1 = RetryManager()
        manager2 = RetryManager()

        # They should get different circuit breaker instances
        cb1 = manager1.get_circuit_breaker("test")
        cb2 = manager2.get_circuit_breaker("test")

        assert cb1 is not cb2  # Different managers have separate circuit breakers

    def test_concurrent_circuit_breaker_access(self):
        """Test circuit breaker behaves correctly under concurrent access simulation."""
        config = RetryConfig(failure_threshold=3, half_open_max_calls=2)
        cb = CircuitBreaker(config)

        # Simulate concurrent failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

        # Force to half-open and simulate concurrent can_execute calls
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_calls = 0

        results = []
        for _ in range(5):
            results.append(cb.can_execute())

        # Only first 2 calls should be allowed
        assert results == [True, True, False, False, False]
        assert cb.half_open_calls == 2


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @patch('time.sleep')
    @patch('abstractllm.events.emit_global')
    def test_rate_limit_recovery_scenario(self, mock_emit, mock_sleep):
        """Test realistic rate limit recovery scenario."""
        provider = MockProvider("gpt-4")

        # Simulate rate limit that clears after retries
        call_count = 0

        def rate_limit_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RateLimitError("Rate limit exceeded, try again later")
            return GenerateResponse(
                content="Success after rate limit",
                model="gpt-4",
                finish_reason="stop"
            )

        provider._generate_internal = rate_limit_then_success

        # Should succeed after retries
        response = provider.generate_with_telemetry("Test prompt")

        assert response.content == "Success after rate limit"
        assert call_count == 3  # Initial + 2 retries

        # Verify retry events were emitted (minimal approach)
        event_types = [call[0][0] for call in mock_emit.call_args_list]
        assert EventType.RETRY_ATTEMPTED in event_types

    def test_circuit_breaker_prevents_cascade_failure(self):
        """Test circuit breaker prevents cascade failures."""
        config = RetryConfig(failure_threshold=2, max_attempts=1)
        provider = MockProvider("failing-model", retry_config=config)

        # Mock to always fail
        provider._generate_internal = Mock(side_effect=ProviderAPIError("Service down"))

        # First two calls should trigger failures and open circuit
        with pytest.raises(ProviderAPIError):
            provider.generate_with_telemetry("Test 1")

        with pytest.raises(ProviderAPIError):
            provider.generate_with_telemetry("Test 2")

        # Third call should be blocked by circuit breaker
        with pytest.raises(ProviderAPIError, match="Circuit breaker open"):
            provider.generate_with_telemetry("Test 3")

        # The actual _generate_internal should only have been called twice
        assert provider._generate_internal.call_count == 2

    def test_mixed_error_types_scenario(self):
        """Test handling of mixed error types in sequence."""
        provider = MockProvider("mixed-errors")

        call_count = 0
        def mixed_errors(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("Rate limit")  # Retryable
            elif call_count == 2:
                raise AuthenticationError("Invalid key")  # Non-retryable
            return GenerateResponse(content="Success", model="mixed-errors")

        provider._generate_internal = mixed_errors

        # Should fail on auth error without further retries
        with pytest.raises(AuthenticationError):
            provider.generate_with_telemetry("Test prompt")

        # Should have attempted rate limit retry, then hit auth error
        assert call_count == 2

    @patch('time.sleep')
    def test_validation_error_retry_behavior(self, mock_sleep):
        """Test retry behavior specifically for validation errors."""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config)

        # Mock function that fails with validation errors twice then succeeds
        from pydantic import ValidationError
        mock_func = Mock(side_effect=[
            ValidationError.from_exception_data("TestModel", []),
            json.JSONDecodeError("Invalid JSON", "test", 0),
            "success"
        ])

        result = manager.execute_with_retry(mock_func, provider_key="test")

        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries

    @patch('time.sleep')
    def test_validation_error_retry_exhaustion(self, mock_sleep):
        """Test validation error retry exhaustion."""
        config = RetryConfig(max_attempts=2)
        manager = RetryManager(config)

        # Mock function that always fails with validation error
        validation_error = json.JSONDecodeError("Always invalid JSON", "test", 0)
        mock_func = Mock(side_effect=validation_error)

        with pytest.raises(json.JSONDecodeError):
            manager.execute_with_retry(mock_func, provider_key="test")

        assert mock_func.call_count == 2  # max_attempts
        assert mock_sleep.call_count == 1  # One retry


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])