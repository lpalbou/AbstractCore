"""Music backend selector helpers.

These helpers are intentionally dependency-light so they can be reused by:
- local library-mode generation (`llm.generate(..., output={"modality": "music", ...})`)
- AbstractCore Server audio routes (`/v1/audio/music`)

The goal is simple: normalize user-facing backend/provider selectors into the
concrete `backend_id` values registered by AbstractMusic (or compatible plugins).
"""

from __future__ import annotations

from typing import Any, Optional


MUSIC_BACKEND_ALIAS_MAP = {
    # Remote hosted backends
    "abstractmusic:acemusic": "abstractmusic:acemusic",
    "acemusic": "abstractmusic:acemusic",
    "ace-music": "abstractmusic:acemusic",
    "acemusic-api": "abstractmusic:acemusic",
    "ace-music-api": "abstractmusic:acemusic",
    "aceapi": "abstractmusic:acemusic",
    "remote": "abstractmusic:acemusic",
    "api": "abstractmusic:acemusic",
    "abstractmusic:elevenlabs-music": "abstractmusic:elevenlabs-music",
    "elevenlabs-music": "abstractmusic:elevenlabs-music",
    "elevenlabs": "abstractmusic:elevenlabs-music",
    # Local backends
    "abstractmusic:acestep-diffusers": "abstractmusic:acestep-diffusers",
    "acestep": "abstractmusic:acestep-diffusers",
    "ace-step": "abstractmusic:acestep-diffusers",
    "ace": "abstractmusic:acestep-diffusers",
    "acestep-diffusers": "abstractmusic:acestep-diffusers",
    "ace-step-diffusers": "abstractmusic:acestep-diffusers",
    "abstractmusic:acestep-v15": "abstractmusic:acestep-v15",
    "acestep-v15": "abstractmusic:acestep-v15",
    "ace-step-v15": "abstractmusic:acestep-v15",
    "abstractmusic:stable-audio": "abstractmusic:stable-audio",
    "stable-audio": "abstractmusic:stable-audio",
    "stableaudio": "abstractmusic:stable-audio",
    "stability-ai": "abstractmusic:stable-audio",
    "stabilityai": "abstractmusic:stable-audio",
    "abstractmusic:stable-audio-3": "abstractmusic:stable-audio-3",
    "stable-audio-3": "abstractmusic:stable-audio-3",
    "stableaudio-3": "abstractmusic:stable-audio-3",
    "stable-audio3": "abstractmusic:stable-audio-3",
    "abstractmusic:diffusers": "abstractmusic:diffusers",
    "diffusers": "abstractmusic:diffusers",
}

MUSIC_BACKEND_ALIASES = set(MUSIC_BACKEND_ALIAS_MAP)


def _selector_text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip().lower().replace("_", "-")
    return None


def resolve_music_backend_id(*values: Any, allow_unknown: bool = False) -> Optional[str]:
    """Resolve user/backend selectors into a concrete plugin backend_id.

    Values are checked in order. Returns `None` when nothing is provided.
    """

    for value in values:
        text = _selector_text(value)
        if not text:
            continue
        if text in MUSIC_BACKEND_ALIAS_MAP:
            return MUSIC_BACKEND_ALIAS_MAP[text]
        if allow_unknown:
            return text
    return None


def resolve_music_provider_hint(*values: Any) -> Optional[str]:
    """Return a provider/catalog hint that is *not* a backend selector."""

    for value in values:
        text = _selector_text(value)
        if text and text not in MUSIC_BACKEND_ALIASES:
            return text
    return None


__all__ = [
    "MUSIC_BACKEND_ALIAS_MAP",
    "MUSIC_BACKEND_ALIASES",
    "resolve_music_backend_id",
    "resolve_music_provider_hint",
]

