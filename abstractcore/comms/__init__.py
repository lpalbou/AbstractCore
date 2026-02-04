"""Framework-native communications helpers (non-LLM, deterministic)."""

from .email_digests import render_email_digest_text, send_email_digest

__all__ = [
    "render_email_digest_text",
    "send_email_digest",
]

