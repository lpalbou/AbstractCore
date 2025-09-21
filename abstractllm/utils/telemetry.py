"""
Telemetry system for tracking usage and performance.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class TelemetryData:
    """Data structure for telemetry events"""
    event_type: str
    timestamp: datetime
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata
        }


class Telemetry:
    """
    Telemetry system for tracking LLM usage.
    Supports verbatim capture for debugging.
    """

    def __init__(self, enabled: bool = True, verbatim: bool = False,
                 output_path: Optional[Path] = None):
        """
        Initialize telemetry system.

        Args:
            enabled: Whether telemetry is enabled
            verbatim: Whether to capture full prompts/responses
            output_path: Path to write telemetry data
        """
        self.enabled = enabled
        self.verbatim = verbatim
        # Convert string to Path if needed
        if output_path:
            self.output_path = Path(output_path) if isinstance(output_path, str) else output_path
        else:
            self.output_path = Path.home() / ".abstractllm" / "telemetry.jsonl"
        self.events: List[TelemetryData] = []

        # Ensure output directory exists
        if self.enabled:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def track_generation(self,
                        provider: str,
                        model: str,
                        prompt: Optional[str] = None,
                        response: Optional[str] = None,
                        tokens: Optional[Dict[str, int]] = None,
                        latency_ms: Optional[float] = None,
                        success: bool = True,
                        error: Optional[str] = None):
        """
        Track a generation event.

        Args:
            provider: Provider name
            model: Model name
            prompt: Input prompt (if verbatim enabled)
            response: Generated response (if verbatim enabled)
            tokens: Token usage dictionary
            latency_ms: Response latency in milliseconds
            success: Whether generation succeeded
            error: Error message if failed
        """
        if not self.enabled:
            return

        metadata = {}
        if self.verbatim:
            if prompt:
                metadata["prompt"] = prompt
            if response:
                metadata["response"] = response

        data = TelemetryData(
            event_type="generation",
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            prompt_tokens=tokens.get("prompt_tokens") if tokens else None,
            completion_tokens=tokens.get("completion_tokens") if tokens else None,
            total_tokens=tokens.get("total_tokens") if tokens else None,
            latency_ms=latency_ms,
            success=success,
            error=error,
            metadata=metadata
        )

        self.events.append(data)
        self._write_event(data)

    def track_tool_call(self,
                       tool_name: str,
                       arguments: Optional[Dict[str, Any]] = None,
                       result: Optional[Any] = None,
                       success: bool = True,
                       error: Optional[str] = None):
        """
        Track a tool call event.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments (if verbatim enabled)
            result: Tool result (if verbatim enabled)
            success: Whether tool call succeeded
            error: Error message if failed
        """
        if not self.enabled:
            return

        metadata = {"tool_name": tool_name}
        if self.verbatim:
            if arguments:
                metadata["arguments"] = arguments
            if result:
                metadata["result"] = str(result)

        data = TelemetryData(
            event_type="tool_call",
            timestamp=datetime.now(),
            success=success,
            error=error,
            metadata=metadata
        )

        self.events.append(data)
        self._write_event(data)

    def track_session(self,
                     action: str,
                     session_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Track session events.

        Args:
            action: Session action (created, cleared, saved, loaded)
            session_id: Session identifier
            metadata: Additional metadata
        """
        if not self.enabled:
            return

        event_metadata = {"action": action}
        if session_id:
            event_metadata["session_id"] = session_id
        if metadata:
            event_metadata.update(metadata)

        data = TelemetryData(
            event_type="session",
            timestamp=datetime.now(),
            metadata=event_metadata
        )

        self.events.append(data)
        self._write_event(data)

    def _write_event(self, data: TelemetryData):
        """Write event to file"""
        try:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(data.to_dict()) + "\n")
        except Exception as e:
            # Silently fail to not interrupt operations
            pass

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of telemetry data.

        Returns:
            Summary statistics
        """
        if not self.events:
            return {"total_events": 0}

        total_tokens = sum(e.total_tokens or 0 for e in self.events
                          if e.event_type == "generation")
        total_generations = sum(1 for e in self.events
                               if e.event_type == "generation")
        total_tool_calls = sum(1 for e in self.events
                              if e.event_type == "tool_call")
        success_rate = sum(1 for e in self.events if e.success) / len(self.events)

        avg_latency = None
        latencies = [e.latency_ms for e in self.events
                    if e.latency_ms is not None]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)

        return {
            "total_events": len(self.events),
            "total_generations": total_generations,
            "total_tool_calls": total_tool_calls,
            "total_tokens": total_tokens,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency
        }

    def clear(self):
        """Clear in-memory events"""
        self.events.clear()

    def disable(self):
        """Disable telemetry"""
        self.enabled = False

    def enable(self):
        """Enable telemetry"""
        self.enabled = True


# Global telemetry instance
_global_telemetry = None


def get_telemetry() -> Telemetry:
    """Get global telemetry instance"""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = Telemetry()
    return _global_telemetry


def setup_telemetry(enabled: bool = True, verbatim: bool = False,
                   output_path: Optional[Path] = None):
    """Setup global telemetry"""
    global _global_telemetry
    _global_telemetry = Telemetry(enabled, verbatim, output_path)