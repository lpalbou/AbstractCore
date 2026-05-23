"""Music backend selector helpers.

These helpers are intentionally dependency-light so they can be reused by:
- local library-mode generation (`llm.generate(..., output={"modality": "music", ...})`)
- AbstractCore Server audio routes (`/v1/audio/music`)

Music exposes one selector space only: the backend name (via `provider`) or the
full registered plugin backend id.
"""

from __future__ import annotations

from typing import Any, Optional


MUSIC_BACKEND_IDS = {
    "abstractmusic:acemusic",
    "abstractmusic:elevenlabs-music",
    "abstractmusic:acestep",
    "abstractmusic:stable-audio",
    "abstractmusic:stable-audio-3",
    "abstractmusic:diffusers",
}

MUSIC_BACKEND_NAMES = {item.split(":", 1)[-1] for item in MUSIC_BACKEND_IDS}


def _selector_text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip().lower().replace("_", "-")
    return None


def resolve_music_backend_id(*values: Any, allow_unknown: bool = False) -> Optional[str]:
    """Resolve exact music backend selectors into a concrete plugin backend_id."""

    for value in values:
        text = _selector_text(value)
        if not text:
            continue
        if text in MUSIC_BACKEND_IDS:
            return text
        if text in MUSIC_BACKEND_NAMES:
            return f"abstractmusic:{text}"
        if allow_unknown:
            return text
    return None

__all__ = [
    "MUSIC_BACKEND_IDS",
    "MUSIC_BACKEND_NAMES",
    "resolve_music_backend_id",
]
