"""
Video processor for AbstractCore media handling.

v0 goals:
- Treat video as a first-class media type (MediaType.VIDEO) in the media pipeline.
- Keep processing lightweight and dependency-free by default (store as a file ref).

Higher-level semantic handling (native video models, frame sampling, captioning)
is handled by policy and capability layers (see planned video policy backlog).
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from ..base import BaseMediaHandler, MediaProcessingError
from ..types import ContentFormat, MediaCapabilities, MediaContent, MediaType


class VideoProcessor(BaseMediaHandler):
    """Lightweight video processor that stores a video file reference."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.capabilities = MediaCapabilities(
            vision_support=False,
            audio_support=False,
            video_support=True,
            document_support=False,
            max_file_size=self.max_file_size,
        )

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        if media_type != MediaType.VIDEO:
            raise MediaProcessingError(f"VideoProcessor only handles video, got {media_type}")

        mime_type, _enc = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size if file_path.exists() else None,
            "processor": self.__class__.__name__,
        }
        metadata.update(kwargs.get("metadata", {}) if isinstance(kwargs.get("metadata"), dict) else {})

        return MediaContent(
            media_type=MediaType.VIDEO,
            content=str(file_path),
            content_format=ContentFormat.FILE_PATH,
            mime_type=mime_type,
            file_path=str(file_path),
            metadata=metadata,
        )

