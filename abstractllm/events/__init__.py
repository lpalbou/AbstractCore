"""
Event system for AbstractLLM - OpenTelemetry compatible.

This module provides a comprehensive event system for tracking LLM operations,
including generation, tool calls, structured output, and performance metrics.
Events are designed to be compatible with OpenTelemetry semantic conventions.

Key features:
- Standardized event types aligned with OpenTelemetry
- Support for structured data and metadata
- Global and local event emission
- Built-in performance tracking
- Tool execution monitoring
- Structured output validation tracking
- Error and warning handling
"""

from typing import Dict, Any, List, Callable, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class EventType(Enum):
    """Minimal event system - clean, simple, efficient"""

    # Core events (4) - matches LangChain pattern
    GENERATION_STARTED = "generation_started"      # Unified for streaming and non-streaming
    GENERATION_COMPLETED = "generation_completed"  # Includes all metrics
    TOOL_STARTED = "tool_started"                  # Before tool execution
    TOOL_COMPLETED = "tool_completed"              # After tool execution

    # Error handling (1)
    ERROR = "error"                                # Any error

    # Retry and resilience events (2) - SOTA minimal approach
    RETRY_ATTEMPTED = "retry_attempted"            # When retry process starts (includes attempt count)
    RETRY_EXHAUSTED = "retry_exhausted"            # When all retries fail (critical for alerting)

    # Useful but minimal (3)
    VALIDATION_FAILED = "validation_failed"        # For retry logic only
    SESSION_CREATED = "session_created"            # Track new sessions
    SESSION_CLEARED = "session_cleared"            # Track cleanup


@dataclass
class Event:
    """Simple event structure - minimal and efficient"""
    type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: Optional[str] = None


class EventEmitter:
    """Mixin for classes that emit events"""

    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}

    def on(self, event_type: EventType, handler: Callable):
        """
        Register an event handler.

        Args:
            event_type: Type of event to listen for
            handler: Function to call when event occurs
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)

    def off(self, event_type: EventType, handler: Callable):
        """
        Unregister an event handler.

        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if event_type in self._listeners:
            self._listeners[event_type].remove(handler)

    def emit(self, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None, **kwargs) -> Event:
        """
        Emit an event to all registered handlers.

        Args:
            event_type: Type of event
            data: Event data
            source: Source of the event
            **kwargs: Additional event attributes (model_name, tokens, etc.)

        Returns:
            The event object (which may have been prevented by handlers)
        """
        # Filter kwargs to only include valid Event fields
        try:
            valid_fields = set(Event.__dataclass_fields__.keys())
        except AttributeError:
            # Fallback for older Python versions
            valid_fields = {'trace_id', 'span_id', 'request_id', 'duration_ms', 'model_name',
                          'provider_name', 'tokens_input', 'tokens_output', 'cost_usd', 'metadata'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        event = Event(
            type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source or self.__class__.__name__,
            **filtered_kwargs
        )

        if event_type in self._listeners:
            for handler in self._listeners[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    # Log error but don't stop event propagation
                    print(f"Error in event handler: {e}")

        return event

    def emit_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Emit an error event.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        self.emit(
            EventType.ERROR,
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context or {}
            }
        )


class GlobalEventBus:
    """
    Global event bus for system-wide events.
    Singleton pattern.
    """
    _instance = None
    _listeners: Dict[EventType, List[Callable]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def on(cls, event_type: EventType, handler: Callable):
        """Register a global event handler"""
        if event_type not in cls._listeners:
            cls._listeners[event_type] = []
        cls._listeners[event_type].append(handler)

    @classmethod
    def off(cls, event_type: EventType, handler: Callable):
        """Unregister a global event handler"""
        if event_type in cls._listeners and handler in cls._listeners[event_type]:
            cls._listeners[event_type].remove(handler)

    @classmethod
    def emit(cls, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None, **kwargs):
        """Emit a global event"""
        # Filter kwargs to only include valid Event fields
        try:
            valid_fields = set(Event.__dataclass_fields__.keys())
        except AttributeError:
            # Fallback for older Python versions
            valid_fields = {'trace_id', 'span_id', 'request_id', 'duration_ms', 'model_name',
                          'provider_name', 'tokens_input', 'tokens_output', 'cost_usd', 'metadata'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        event = Event(
            type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source or "GlobalEventBus",
            **filtered_kwargs
        )

        if event_type in cls._listeners:
            for handler in cls._listeners[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in global event handler: {e}")

    @classmethod
    def clear(cls):
        """Clear all global event handlers"""
        cls._listeners.clear()


# Convenience functions
def on_global(event_type: EventType, handler: Callable):
    """Register a global event handler"""
    GlobalEventBus.on(event_type, handler)


def emit_global(event_type: EventType, data: Dict[str, Any], source: Optional[str] = None, **kwargs):
    """Emit a global event"""
    GlobalEventBus.emit(event_type, data, source, **kwargs)


# Event utility functions for common patterns
def create_generation_event(model_name: str, provider_name: str,
                          tokens_input: int = None, tokens_output: int = None,
                          duration_ms: float = None, cost_usd: float = None,
                          **data) -> Dict[str, Any]:
    """Create standardized generation event data"""
    event_data = {
        "model_name": model_name,
        "provider_name": provider_name,
        **data
    }

    # Include attributes for event emission
    kwargs = {
        "model_name": model_name,
        "provider_name": provider_name,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "duration_ms": duration_ms,
        "cost_usd": cost_usd
    }

    return event_data, kwargs


def create_tool_event(tool_name: str, arguments: Dict[str, Any],
                     result: Any = None, success: bool = True,
                     error: str = None, **data) -> Dict[str, Any]:
    """Create standardized tool event data"""
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "result": str(result)[:500] if result else None,
        "success": success,
        "error": error,
        **data
    }


def create_structured_output_event(response_model: str,
                                  validation_attempt: int = None,
                                  validation_error: str = None,
                                  retry_count: int = None, **data) -> Dict[str, Any]:
    """Create standardized structured output event data"""
    return {
        "response_model": response_model,
        "validation_attempt": validation_attempt,
        "validation_error": validation_error,
        "retry_count": retry_count,
        **data
    }


# Common event handler patterns
class EventLogger:
    """Basic event logger for debugging and monitoring"""

    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level

    def log_event(self, event: Event):
        """Log event details"""
        print(f"[{event.timestamp}] {event.type.value} from {event.source}: {event.data}")


class PerformanceTracker:
    """Track performance metrics from events"""

    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_latency_ms": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_cost_usd": 0.0,
            "tool_calls": 0,
            "errors": 0
        }

    def track_generation(self, event: Event):
        """Track generation performance"""
        if event.type == EventType.GENERATION_COMPLETED:
            self.metrics["total_requests"] += 1
            data = event.data
            if data.get("duration_ms"):
                self.metrics["total_latency_ms"] += data["duration_ms"]
            if data.get("tokens_input"):
                self.metrics["total_tokens_input"] += data["tokens_input"]
            if data.get("tokens_output"):
                self.metrics["total_tokens_output"] += data["tokens_output"]
            if data.get("cost_usd"):
                self.metrics["total_cost_usd"] += data["cost_usd"]

    def track_tool_call(self, event: Event):
        """Track tool call metrics"""
        if event.type == EventType.TOOL_COMPLETED:
            self.metrics["tool_calls"] += 1

    def track_error(self, event: Event):
        """Track error metrics"""
        if event.type == EventType.ERROR:
            self.metrics["errors"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()