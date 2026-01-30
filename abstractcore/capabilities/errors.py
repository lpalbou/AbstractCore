from __future__ import annotations

from typing import Any, Dict, List, Optional


class CapabilityError(RuntimeError):
    """Base class for capability-related errors."""


class CapabilityUnavailableError(CapabilityError):
    """Raised when a requested capability or backend is not available.

    This error is intended to be actionable: it should include install and/or
    configuration hints so callers can fix the issue quickly.
    """

    def __init__(
        self,
        *,
        capability: str,
        reason: str,
        install_hint: Optional[str] = None,
        config_hint: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.capability = str(capability or "").strip() or "unknown"
        self.reason = str(reason or "").strip() or "Capability unavailable"
        self.install_hint = str(install_hint).strip() if isinstance(install_hint, str) and install_hint.strip() else None
        self.config_hint = str(config_hint).strip() if isinstance(config_hint, str) and config_hint.strip() else None
        self.alternatives = list(alternatives) if isinstance(alternatives, list) else None
        self.details = dict(details) if isinstance(details, dict) else None

        lines = [f"{self.capability}: {self.reason}"]
        if self.install_hint:
            lines.append(f"Install: {self.install_hint}")
        if self.config_hint:
            lines.append(f"Configure: {self.config_hint}")
        if self.alternatives:
            lines.append(f"Alternatives: {', '.join([str(a) for a in self.alternatives])}")
        super().__init__("\n".join(lines))

