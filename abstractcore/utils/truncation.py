"""Truncation utilities (explicit + searchable).

Policy authority: ADR-0026 (docs/adr/0026-truncation-policy-and-contract.md).

All lossy truncation must:
- be explicit in the returned text (marker),
- and be searchable in code via `#[WARNING:TRUNCATION]`.
"""

from __future__ import annotations

from typing import Any


def preview_text(value: Any, *, max_chars: int, marker: str = "… (truncated)") -> str:
    """Return `value` as a bounded preview with an explicit truncation marker."""
    s = str(value or "")
    if max_chars <= 0:
        #[WARNING:TRUNCATION] bounded preview requested with max_chars<=0
        return ""
    max_chars_i = int(max_chars)
    if len(s) <= max_chars_i:
        return s
    #[WARNING:TRUNCATION] bounded preview (logs/telemetry/UI)
    keep = max(0, max_chars_i - len(marker))
    if keep <= 0:
        return marker[:max_chars_i].rstrip()
    return s[:keep].rstrip() + marker

