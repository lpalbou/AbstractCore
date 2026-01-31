from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.basic
def test_video_processor_returns_file_ref(tmp_path: Path) -> None:
    from abstractcore.media.processors import VideoProcessor
    from abstractcore.media.types import ContentFormat, MediaType

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")

    processor = VideoProcessor()
    result = processor.process_file(video_path)

    assert result.success
    assert result.media_content.media_type == MediaType.VIDEO
    assert result.media_content.content_format == ContentFormat.FILE_PATH
    assert result.media_content.file_path == str(video_path)
    assert result.media_content.metadata.get("file_name") == "sample.mp4"


@pytest.mark.basic
def test_auto_media_handler_routes_video_to_video_processor(tmp_path: Path) -> None:
    from abstractcore.media import AutoMediaHandler
    from abstractcore.media.types import MediaType

    video_path = tmp_path / "clip.mov"
    video_path.write_bytes(b"fake")

    handler = AutoMediaHandler()
    result = handler.process_file(video_path)

    assert result.success
    assert result.media_content.media_type == MediaType.VIDEO
    assert result.media_content.file_path == str(video_path)

