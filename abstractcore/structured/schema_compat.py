"""Backend schema-compatibility support for structured outputs.

Some backends validate the JSON schema attached to a structured-output request
against STRICT rules (OpenAI strict mode and subscription relays in front of
it: every object must declare `properties`, `required` listing every key, and
`additionalProperties: false`). A schema that violates those rules — most
notably a free-form dict `{"type": "object"}` without `properties`, which is
not expressible under strict rules at all — is refused with a deterministic
4xx before any generation happens.

That refusal is a property of (backend, model, schema), not of the request
content: retrying the identical call can never succeed, but the SAME schema
can still be satisfied through the prompted structured-output lane (schema in
the prompt + JSON extraction + validation), which every provider supports.

This module provides the two pieces the structured handler needs to make that
class survivable for every provider and every schema, with no
provider-specific special cases:

- ``is_schema_rejection_error``: a conservative, evidence-based classifier for
  "the backend refused the schema itself" (never matches auth, rate-limit,
  context-length, or other unrelated 4xx errors).
- ``SchemaRejectionRegistry``: a process-lifetime memo of rejection decisions
  keyed by (provider class, base_url, model, schema fingerprint) so every
  later call with the same schema skips the doomed native attempt instead of
  re-hitting the 4xx each cycle.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "is_schema_rejection_error",
    "schema_rejection_registry",
    "SchemaRejectionRegistry",
]


# 4xx statuses that are NEVER schema rejections regardless of message text:
# authentication (401/403/407), request timeout (408), conflict/too-early
# during model load (409/425), and rate limiting (429).
_EXCLUDED_4XX = frozenset({401, 403, 407, 408, 409, 425, 429})

# Evidence-based signatures observed from strict-schema validators
# (OpenAI strict mode, subscription relays, and OpenAI-compatible servers).
# All matching is done lowercase.
_SCHEMA_REJECTION_SIGNATURES = (
    # OpenAI error code for strict-mode schema refusals (param text.format.schema).
    "invalid_json_schema",
    # OpenAI responses-API parameter path reported in refusals.
    "text.format.schema",
    # "Invalid schema for response_format 'X'..." (OpenAI) and compatible servers.
    "invalid schema",
    # Strict-mode complaints about additionalProperties (observed live:
    # "The subscription backend requires every object schema to set
    #  `additionalProperties` to `false`").
    "additionalproperties",
    "object schema",
    # OpenAI strict validator: "'required' is required to be supplied and to
    # be an array including every key in properties."
    "'required' is required",
    # Servers that refuse the structured-output request parameter outright
    # ("response_format is not supported...", "json_schema mode requires...").
    "response_format",
    "json_schema",
)

# Message dialects our providers use when the exception object carries no
# status_code attribute: "... API error (422): ..." / "Error code: 400 - ...".
_STATUS_IN_MESSAGE_RE = re.compile(r"(?:api error \((\d{3})\)|error code:? (\d{3}))")


def _extract_status(exc: BaseException) -> Optional[int]:
    """HTTP status for the error: attribute first, message dialect fallback."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, bool):
        status = None
    if isinstance(status, int):
        return status
    match = _STATUS_IN_MESSAGE_RE.search(str(exc or "").lower())
    if match:
        try:
            return int(match.group(1) or match.group(2))
        except (TypeError, ValueError):
            return None
    return None


def is_schema_rejection_error(exc: BaseException) -> bool:
    """True when a provider error means "the backend refused the JSON schema".

    Conservative by design — this classifier gates a silent lane switch, so a
    false positive would hide real request errors behind the prompted lane:

    - The error must carry a 4xx HTTP status (attribute, or our providers'
      "API error (NNN)" / "Error code: NNN" message dialects) that is not an
      auth/timeout/conflict/rate-limit status.
    - AND the message must match a known strict-schema-validator signature.

    Auth failures, context-length 400s, rate limits, and 5xx server errors
    never match; they keep their existing fatal/retryable semantics.

    NOTE: this is only meaningful for errors raised from a structured-output
    request (`response_model` present). Within that lane no tools are attached
    (the hybrid tools path strips them before the structured pass), so a
    schema complaint can only be about the response schema.
    """
    if exc is None:  # defensive: callers pass whatever they caught
        return False

    status = _extract_status(exc)
    if status is None or not (400 <= status < 500) or status in _EXCLUDED_4XX:
        return False

    message = str(exc or "").lower()
    if not message:
        return False
    return any(sig in message for sig in _SCHEMA_REJECTION_SIGNATURES)


def _schema_fingerprint(response_model: Any) -> str:
    """Stable fingerprint of the schema the provider would put on the wire."""
    try:
        schema = response_model.model_json_schema()
        canonical = json.dumps(schema, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        # Degenerate models still get a stable identity (per-class decision).
        canonical = f"{getattr(response_model, '__module__', '?')}.{getattr(response_model, '__qualname__', repr(response_model))}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SchemaRejectionRegistry:
    """Process-lifetime memo of native-schema rejections.

    Keyed by (provider class, base_url, model, schema fingerprint): the same
    model name behind two endpoints must not share a decision, and a backend
    that rejects one schema may accept another. Bounded FIFO eviction keeps
    the registry small in long sessions.
    """

    def __init__(self, max_entries: int = 512) -> None:
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._entries: Dict[Tuple[str, str, str, str], str] = {}

    @staticmethod
    def _key(provider: Any, response_model: Any) -> Tuple[str, str, str, str]:
        return (
            provider.__class__.__name__,
            str(getattr(provider, "base_url", "") or ""),
            str(getattr(provider, "model", "") or ""),
            _schema_fingerprint(response_model),
        )

    def mark_rejected(self, provider: Any, response_model: Any, error: str) -> None:
        key = self._key(provider, response_model)
        with self._lock:
            self._entries[key] = str(error or "")[:500]
            while len(self._entries) > self._max_entries:
                self._entries.pop(next(iter(self._entries)))

    def rejection_reason(self, provider: Any, response_model: Any) -> Optional[str]:
        """The recorded rejection error, or None when native was never refused."""
        key = self._key(provider, response_model)
        with self._lock:
            return self._entries.get(key)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# One registry per process ("per session"): StructuredOutputHandler instances
# are constructed per call, so the decision must outlive them.
schema_rejection_registry = SchemaRejectionRegistry()
