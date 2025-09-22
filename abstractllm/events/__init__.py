"""
Event system for AbstractLLM.
"""

from typing import Dict, Any, List, Callable, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class EventType(Enum):
    """Types of events in the system"""
    BEFORE_GENERATE = "before_generate"
    AFTER_GENERATE = "after_generate"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    BEFORE_TOOL_EXECUTION = "before_tool_execution"
    AFTER_TOOL_EXECUTION = "after_tool_execution"
    ERROR_OCCURRED = "error_occurred"
    SESSION_CREATED = "session_created"
    SESSION_CLEARED = "session_cleared"
    MESSAGE_ADDED = "message_added"
    PROVIDER_CREATED = "provider_created"
    STREAM_STARTED = "stream_started"
    STREAM_CHUNK = "stream_chunk"
    STREAM_COMPLETED = "stream_completed"


@dataclass
class Event:
    """Represents an event in the system"""
    type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    prevented: bool = False

    def prevent(self):
        """Prevent the event's default behavior"""
        self.prevented = True


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

    def emit(self, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None) -> Event:
        """
        Emit an event to all registered handlers.

        Args:
            event_type: Type of event
            data: Event data
            source: Source of the event

        Returns:
            The event object (which may have been prevented by handlers)
        """
        event = Event(
            type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source or self.__class__.__name__
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
            EventType.ERROR_OCCURRED,
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
    def emit(cls, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None):
        """Emit a global event"""
        event = Event(
            type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source or "GlobalEventBus"
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


def emit_global(event_type: EventType, data: Dict[str, Any], source: Optional[str] = None):
    """Emit a global event"""
    GlobalEventBus.emit(event_type, data, source)