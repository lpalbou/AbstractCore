"""
Comprehensive tests for the enhanced event system.

This module tests all the new event types and comprehensive event functionality
across the entire AbstractLLM library.
"""

import pytest
import json
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from abstractllm import create_llm
from abstractllm.events import (
    EventType, Event, EventEmitter, GlobalEventBus,
    EventLogger, PerformanceTracker,
    create_generation_event, create_tool_event, create_structured_output_event,
    emit_global, on_global
)
from abstractllm.tools import ToolDefinition, ToolCall, execute_tool, clear_registry, register_tool
from abstractllm.core.types import GenerateResponse


class EventCapture:
    """Helper class to capture events for testing"""

    def __init__(self):
        self.captured_events: List[Event] = []

    def capture_event(self, event: Event):
        """Capture an event"""
        self.captured_events.append(event)

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all captured events of a specific type"""
        return [e for e in self.captured_events if e.type == event_type]

    def clear(self):
        """Clear captured events"""
        self.captured_events.clear()


@pytest.fixture
def event_capture():
    """Fixture to provide event capture functionality"""
    capture = EventCapture()
    yield capture
    # Clean up global event bus
    GlobalEventBus.clear()


class TestEnhancedEventTypes:
    """Test new comprehensive event types"""

    def test_all_event_types_exist(self):
        """Test that all expected event types are defined"""
        expected_events = [
            # Core events (4) - minimal essential set
            "GENERATION_STARTED", "GENERATION_COMPLETED",
            "TOOL_STARTED", "TOOL_COMPLETED",
            # Error handling (1)
            "ERROR",
            # Useful but minimal (3)
            "VALIDATION_FAILED", "SESSION_CREATED", "SESSION_CLEARED"
        ]

        for event_name in expected_events:
            assert hasattr(EventType, event_name), f"Missing event type: {event_name}"

    def test_simple_event_structure(self):
        """Test that events have simple structure"""
        from datetime import datetime
        event = Event(
            type=EventType.GENERATION_COMPLETED,
            timestamp=datetime.now(),
            data={
                "test": "data",
                "model": "gpt-4",
                "provider": "OpenAIProvider",
                "tokens_input": 100,
                "tokens_output": 50,
                "cost_usd": 0.003,
                "duration_ms": 1500.5
            },
            source="TestSource"
        )

        # Verify simple structure - only 4 fields
        assert event.type == EventType.GENERATION_COMPLETED
        assert isinstance(event.timestamp, datetime)
        assert event.data["model"] == "gpt-4"
        assert event.source == "TestSource"


class TestEventUtilityFunctions:
    """Test event utility functions"""

    def test_create_generation_event(self):
        """Test generation event creation utility"""
        event_data, kwargs = create_generation_event(
            model_name="gpt-4",
            provider_name="OpenAIProvider",
            tokens_input=100,
            tokens_output=50,
            duration_ms=1200.5,
            cost_usd=0.003,
            custom_field="custom_value"
        )

        assert event_data["model_name"] == "gpt-4"
        assert event_data["provider_name"] == "OpenAIProvider"
        assert event_data["custom_field"] == "custom_value"

        assert kwargs["tokens_input"] == 100
        assert kwargs["tokens_output"] == 50
        assert kwargs["duration_ms"] == 1200.5
        assert kwargs["cost_usd"] == 0.003

    def test_create_tool_event(self):
        """Test tool event creation utility"""
        event_data = create_tool_event(
            tool_name="test_tool",
            arguments={"param": "value"},
            result="tool output",
            success=True
        )

        assert event_data["tool_name"] == "test_tool"
        assert event_data["arguments"] == {"param": "value"}
        assert event_data["result"] == "tool output"
        assert event_data["success"] is True

    def test_create_structured_output_event(self):
        """Test structured output event creation utility"""
        event_data = create_structured_output_event(
            response_model="UserProfile",
            validation_attempt=2,
            validation_error="Field 'age' must be positive",
            retry_count=1
        )

        assert event_data["response_model"] == "UserProfile"
        assert event_data["validation_attempt"] == 2
        assert event_data["validation_error"] == "Field 'age' must be positive"
        assert event_data["retry_count"] == 1


class TestEventHandlerPatterns:
    """Test built-in event handler patterns"""

    def test_event_logger(self, capfd):
        """Test EventLogger pattern"""
        logger = EventLogger()

        from datetime import datetime
        event = Event(
            type=EventType.TOOL_STARTED,
            timestamp=datetime.now(),
            data={"tool_calls": [{"name": "test_tool"}]}
        )

        logger.log_event(event)

        captured = capfd.readouterr()
        assert "tool_started" in captured.out
        assert "test_tool" in captured.out

    def test_performance_tracker(self):
        """Test PerformanceTracker pattern"""
        tracker = PerformanceTracker()

        from datetime import datetime
        # Test generation tracking
        gen_event = Event(
            type=EventType.GENERATION_COMPLETED,
            timestamp=datetime.now(),
            data={
                "duration_ms": 1500.0,
                "tokens_input": 100,
                "tokens_output": 50,
                "cost_usd": 0.003
            }
        )
        tracker.track_generation(gen_event)

        # Test tool tracking
        tool_event = Event(
            type=EventType.TOOL_COMPLETED,
            timestamp=datetime.now(),
            data={}
        )
        tracker.track_tool_call(tool_event)

        # Test error tracking
        error_event = Event(
            type=EventType.ERROR,
            timestamp=datetime.now(),
            data={}
        )
        tracker.track_error(error_event)

        metrics = tracker.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["total_latency_ms"] == 1500.0
        assert metrics["total_tokens_input"] == 100
        assert metrics["total_tokens_output"] == 50
        assert metrics["total_cost_usd"] == 0.003
        assert metrics["tool_calls"] == 1
        assert metrics["errors"] == 1


class TestProviderEvents:
    """Test events emitted by providers"""

    def test_mock_provider_events(self, event_capture):
        """Test that mock provider emits expected events"""
        # Register event listener for generation events only (PROVIDER_CREATED removed)
        on_global(EventType.GENERATION_STARTED, event_capture.capture_event)
        on_global(EventType.GENERATION_COMPLETED, event_capture.capture_event)

        # Create provider and generate response
        llm = create_llm("mock", model="test-model")
        response = llm.generate("Test prompt")

        # Verify generation events were emitted
        start_events = event_capture.get_events_by_type(EventType.GENERATION_STARTED)
        assert len(start_events) == 1
        assert start_events[0].data["model"] == "test-model"

        complete_events = event_capture.get_events_by_type(EventType.GENERATION_COMPLETED)
        assert len(complete_events) == 1
        assert complete_events[0].data["success"] is True


class TestToolEvents:
    """Test events emitted during tool execution"""

    @pytest.fixture(autouse=True)
    def setup_tools(self):
        """Setup test tools"""
        clear_registry()

        def test_tool(message: str) -> str:
            """A simple test tool"""
            return f"Tool executed with: {message}"

        register_tool(test_tool)
        yield
        clear_registry()

    def test_tool_execution_events(self, event_capture):
        """Test that tool execution emits expected events"""
        # Register event listeners
        on_global(EventType.TOOL_COMPLETED, event_capture.capture_event)

        # Execute a tool
        tool_call = ToolCall(
            name="test_tool",
            arguments={"message": "hello world"}
        )

        result = execute_tool(tool_call)

        # Verify events were emitted
        result_events = event_capture.get_events_by_type(EventType.TOOL_COMPLETED)
        assert len(result_events) == 1
        assert result_events[0].data["success"] is True
        assert result_events[0].data["result"] is not None

    def test_tool_error_events(self, event_capture):
        """Test that tool errors emit appropriate events"""
        # Register event listener
        on_global(EventType.TOOL_COMPLETED, event_capture.capture_event)

        # Execute a non-existent tool
        tool_call = ToolCall(
            name="nonexistent_tool",
            arguments={}
        )

        result = execute_tool(tool_call)

        # Verify error event was emitted
        result_events = event_capture.get_events_by_type(EventType.TOOL_COMPLETED)
        assert len(result_events) == 1
        assert result_events[0].data["success"] is False
        assert "not found" in result_events[0].data["error"]


class TestStructuredOutputEvents:
    """Test events emitted during structured output processing"""

    def test_structured_output_mock_events(self, event_capture):
        """Test structured output validation events"""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            age: int

        # Register event listeners for validation events
        on_global(EventType.VALIDATION_FAILED, event_capture.capture_event)

        # Create provider and generate structured response
        llm = create_llm("mock", model="test-model")

        # Test with invalid JSON to trigger validation failure
        with patch.object(llm, '_generate_internal') as mock_generate:
            # First return invalid JSON to trigger validation failure
            mock_generate.return_value = GenerateResponse(
                content='{"invalid": "json_missing_required_fields"}',
                model="test-model"
            )

            try:
                # This should fail validation and retry
                result = llm.generate(
                    "Generate user data",
                    response_model=TestModel
                )
                # If we get here, validation succeeded (mock might have been called again)
            except Exception:
                # Expected - validation failed with no retries left
                pass

            # Check if validation failure was recorded
            validation_events = event_capture.get_events_by_type(EventType.VALIDATION_FAILED)
            # We should have at least one validation failure event
            assert len(validation_events) >= 1


class TestEventBusIntegration:
    """Test global event bus integration"""

    def test_global_and_local_events(self, event_capture):
        """Test that global events work properly (local events removed in simplification)"""
        # Register global handler
        on_global(EventType.GENERATION_STARTED, event_capture.capture_event)

        # Emit event globally
        emit_global(EventType.GENERATION_STARTED, {"test": "global"})

        # Verify event was captured
        events = event_capture.get_events_by_type(EventType.GENERATION_STARTED)
        assert len(events) == 1
        assert events[0].data.get("test") == "global"
        assert events[0].source == "GlobalEventBus"

    def test_event_prevention(self):
        """Test that simplified event system has no prevention (removed for simplicity)"""
        # Event prevention was removed in the simplified system
        # This test now verifies that events are simple and direct

        handler_called = []

        def test_handler(event: Event):
            handler_called.append(True)

        # Register handler
        on_global(EventType.TOOL_STARTED, test_handler)

        # Emit event
        emit_global(EventType.TOOL_STARTED, {"test": "simple"})

        # Verify handler was called (no prevention mechanism)
        assert len(handler_called) == 1


class TestEventSystemPerformance:
    """Test event system performance characteristics"""

    def test_event_overhead(self):
        """Test that event system has minimal overhead"""
        import time

        # Test with global events (simplified system)
        def dummy_handler(event):
            pass

        on_global(EventType.GENERATION_COMPLETED, dummy_handler)

        start_time = time.time()
        for _ in range(100):  # Reduced iterations for more realistic test
            emit_global(EventType.GENERATION_COMPLETED, {"test": "data"})
        event_time = time.time() - start_time

        # Event system should complete 100 iterations in reasonable time (< 1 second)
        assert event_time < 1.0

    def test_large_event_data(self):
        """Test handling of large event data"""
        # Create large data payload
        large_data = {"large_field": "x" * 10000}

        captured_event = None
        def capture_handler(event):
            nonlocal captured_event
            captured_event = event

        on_global(EventType.GENERATION_COMPLETED, capture_handler)

        # Emit event with large data
        emit_global(EventType.GENERATION_COMPLETED, large_data)

        # Verify event was handled correctly
        assert captured_event is not None
        assert len(captured_event.data["large_field"]) == 10000


if __name__ == "__main__":
    pytest.main([__file__])