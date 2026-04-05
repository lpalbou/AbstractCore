"""
File attachment "boxes" for prompt caching.

Goal
----
Turn `@file` attachments into deterministic text "boxes" that can be appended once to a
session transcript and (when supported) appended once to a provider KV/prefix cache.

This enables fast iteration on large contexts because:
- The file content becomes part of the conversation history (so it persists across turns).
- KV / prefix caches can reuse the evaluated prefix after the file has been appended once.

Notes
-----
- This module is intentionally conservative: it only extracts *text* from TEXT/DOCUMENT
  media types. Other modalities (images/audio/video) should remain `media=` inputs.
- Extraction uses AbstractCore's `AutoMediaHandler` to reuse existing processors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


def _sha256_file_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FileBox:
    """A deterministic extracted-text snapshot of a file attachment."""

    path: str
    media_type: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    content: str
    content_sha256: str
    format: Optional[str] = None
    content_length: int = 0
    estimated_tokens: Optional[int] = None

    def to_meta(self) -> Dict[str, Any]:
        """Metadata suitable for storing in Message.metadata (does not include content)."""
        return {
            "path": self.path,
            "media_type": self.media_type,
            "size_bytes": int(self.size_bytes),
            "mtime_ns": int(self.mtime_ns),
            "sha256": self.sha256,
            "content_sha256": self.content_sha256,
            "format": self.format,
            "content_length": int(self.content_length),
            "estimated_tokens": int(self.estimated_tokens) if isinstance(self.estimated_tokens, int) else None,
        }


def extract_file_box(
    file_path: Union[str, Path],
    *,
    max_file_size: Optional[int] = None,
    format_output: str = "structured",
) -> FileBox:
    """Extract a text-only FileBox from a local file path.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file type is not TEXT/DOCUMENT or extraction fails.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(str(path))

    resolved = path.resolve()
    stat = resolved.stat()
    size_bytes = int(stat.st_size)
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))

    # Detect media type before extraction so callers can decide whether to fall back to media=.
    try:
        from ..media.types import MediaType, detect_media_type
    except Exception as e:  # pragma: no cover
        raise ValueError("Media pipeline is unavailable; cannot extract file boxes.") from e

    media_type = detect_media_type(resolved)
    if media_type not in {MediaType.TEXT, MediaType.DOCUMENT}:
        raise ValueError(f"Unsupported file box media type: {media_type.value}")

    sha256 = _sha256_file_bytes(resolved)

    try:
        from ..media.auto_handler import AutoMediaHandler
    except Exception as e:  # pragma: no cover
        raise ValueError("AutoMediaHandler is unavailable; cannot extract file boxes.") from e

    handler_kwargs: Dict[str, Any] = {"enable_events": False}
    if max_file_size is not None:
        handler_kwargs["max_file_size"] = int(max_file_size)
    handler = AutoMediaHandler(**handler_kwargs)

    result = handler.process_file(resolved, format_output=str(format_output or "structured"))
    if not getattr(result, "success", False) or getattr(result, "media_content", None) is None:
        msg = str(getattr(result, "error_message", None) or "unknown extraction error")
        raise ValueError(f"Failed to extract file box for {resolved}: {msg}")

    content_val = getattr(result.media_content, "content", "")
    if isinstance(content_val, str):
        content = content_val
    else:
        # Best-effort stringify to keep the box JSON-serializable.
        try:
            import json

            content = json.dumps(content_val, ensure_ascii=False)
        except Exception:
            content = str(content_val)

    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    content_length = len(content)
    estimated_tokens: Optional[int]
    try:
        from ..utils.token_utils import estimate_tokens

        estimated_tokens = int(estimate_tokens(content))
    except Exception:
        estimated_tokens = None
    fmt = None
    try:
        fmt = getattr(result.media_content, "format", None)
    except Exception:
        fmt = None
    if fmt is not None:
        fmt = str(fmt)

    return FileBox(
        path=str(resolved),
        media_type=str(media_type.value),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        sha256=sha256,
        content=content,
        content_sha256=content_sha256,
        format=fmt,
        content_length=content_length,
        estimated_tokens=estimated_tokens,
    )


def render_file_box_message(file_box: FileBox) -> str:
    """Render a FileBox into a stable prompt snippet."""
    path = str(file_box.path)
    # Escape quotes to avoid breaking the attribute boundary.
    path_attr = path.replace('"', '\\"')
    content = str(file_box.content or "")
    if not content.endswith("\n"):
        content += "\n"
    return f'<attached_file path="{path_attr}">\n{content}</attached_file>'


def attached_file_dedupe_key(meta: Dict[str, Any]) -> Optional[Tuple[str, int, int]]:
    """Return a stable dedupe key for an attached file meta dict (path, size, mtime_ns)."""
    if not isinstance(meta, dict):
        return None
    path = meta.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    size = meta.get("size_bytes")
    mtime_ns = meta.get("mtime_ns")
    try:
        size_i = int(size)
        mtime_i = int(mtime_ns)
    except Exception:
        return None
    return path.strip(), size_i, mtime_i
