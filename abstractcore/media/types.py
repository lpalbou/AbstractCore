"""
Core media types and models for AbstractCore multimodal support.

This module defines the fundamental data structures for handling various media types
across different LLM providers, following AbstractCore's unified interface patterns.
"""

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator


class MediaType(Enum):
    """Supported media types for multimodal processing."""
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


class ContentFormat(Enum):
    """Different ways media content can be represented."""
    BASE64 = "base64"
    URL = "url"
    FILE_PATH = "file_path"
    TEXT = "text"
    BINARY = "binary"
    AUTO = "auto"


@dataclass
class MediaContent:
    """
    Represents a piece of media content with metadata.

    This is the core data structure for all media handling in AbstractCore.
    It provides a unified way to represent different types of content regardless
    of the underlying provider.
    """
    media_type: MediaType
    content: Union[str, bytes]
    content_format: ContentFormat
    mime_type: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize the MediaContent after initialization."""
        # Auto-detect MIME type if not provided and we have a file path
        if self.mime_type == "auto" and self.file_path:
            detected_mime, _ = mimetypes.guess_type(self.file_path)
            self.mime_type = detected_mime or "application/octet-stream"

        # Ensure content format matches content type
        if self.content_format == ContentFormat.BASE64 and isinstance(self.content, bytes):
            self.content = base64.b64encode(self.content).decode('utf-8')
        elif self.content_format == ContentFormat.TEXT and isinstance(self.content, bytes):
            self.content = self.content.decode('utf-8')


class MultimodalMessage(BaseModel):
    """
    A message that can contain both text and media content.

    This follows the pattern of modern multimodal APIs where a single message
    can contain multiple content elements of different types.
    """
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Mixed content list containing text strings and media objects"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('role')
    def validate_role(cls, v):
        valid_roles = {'user', 'assistant', 'system', 'tool'}
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return v

    def add_text(self, text: str) -> None:
        """Add text content to the message."""
        self.content.append(text)

    def add_media(self, media: MediaContent) -> None:
        """Add media content to the message."""
        media_dict = {
            "type": "media",
            "media_type": media.media_type.value,
            "content": media.content,
            "content_format": media.content_format.value,
            "mime_type": media.mime_type,
            "metadata": media.metadata
        }
        if media.file_path:
            media_dict["file_path"] = media.file_path

        self.content.append(media_dict)

    def has_media(self) -> bool:
        """Check if this message contains any media content."""
        return any(
            isinstance(item, dict) and item.get("type") == "media"
            for item in self.content
        )

    def get_text_content(self) -> str:
        """Extract all text content from the message."""
        text_parts = [
            item for item in self.content
            if isinstance(item, str)
        ]
        return " ".join(text_parts)

    def get_media_content(self) -> List[Dict[str, Any]]:
        """Extract all media content from the message."""
        return [
            item for item in self.content
            if isinstance(item, dict) and item.get("type") == "media"
        ]


@dataclass
class MediaCapabilities:
    """
    Represents what media capabilities a provider/model supports.

    This is used for intelligent routing and validation of media content
    based on the target provider and model capabilities.
    """
    vision_support: bool = False
    audio_support: bool = False
    video_support: bool = False
    document_support: bool = False

    # Image-specific capabilities
    max_image_resolution: Optional[str] = None
    supported_image_formats: List[str] = field(default_factory=lambda: ["jpg", "png"])

    # Document-specific capabilities
    supported_document_formats: List[str] = field(default_factory=lambda: ["pdf", "txt"])

    # Audio/Video capabilities
    max_audio_duration: Optional[int] = None  # in seconds
    max_video_duration: Optional[int] = None  # in seconds

    # Provider-specific limits
    max_file_size: Optional[int] = None  # in bytes
    max_concurrent_media: int = 1

    def supports_media_type(self, media_type: MediaType) -> bool:
        """Check if this provider supports the given media type."""
        support_map = {
            MediaType.IMAGE: self.vision_support,
            MediaType.AUDIO: self.audio_support,
            MediaType.VIDEO: self.video_support,
            MediaType.DOCUMENT: self.document_support,
            MediaType.TEXT: True  # All providers support text
        }
        return support_map.get(media_type, False)

    def supports_format(self, media_type: MediaType, format_ext: str) -> bool:
        """Check if this provider supports the specific format."""
        format_ext = format_ext.lower().lstrip('.')

        if media_type == MediaType.IMAGE:
            return format_ext in self.supported_image_formats
        elif media_type == MediaType.DOCUMENT:
            return format_ext in self.supported_document_formats
        elif media_type in [MediaType.AUDIO, MediaType.VIDEO]:
            # For now, assume basic support if the media type is supported
            return self.supports_media_type(media_type)
        else:
            return True


class MediaProcessingResult(BaseModel):
    """
    Result of processing a media file.

    Contains the processed content and metadata about the processing operation.
    """
    success: bool
    media_content: Optional[MediaContent] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Check if processing failed."""
        return not self.success


# File extension mappings for quick media type detection
FILE_TYPE_MAPPINGS = {
    # Images
    'jpg': MediaType.IMAGE, 'jpeg': MediaType.IMAGE, 'png': MediaType.IMAGE,
    'gif': MediaType.IMAGE, 'bmp': MediaType.IMAGE, 'tif': MediaType.IMAGE,
    'tiff': MediaType.IMAGE, 'webp': MediaType.IMAGE, 'ico': MediaType.IMAGE,

    # Documents
    'pdf': MediaType.DOCUMENT, 'doc': MediaType.DOCUMENT, 'docx': MediaType.DOCUMENT,
    'xls': MediaType.DOCUMENT, 'xlsx': MediaType.DOCUMENT, 'ppt': MediaType.DOCUMENT,
    'pptx': MediaType.DOCUMENT, 'odt': MediaType.DOCUMENT, 'rtf': MediaType.DOCUMENT,

    # Text formats
    'txt': MediaType.TEXT, 'md': MediaType.TEXT, 'csv': MediaType.TEXT,
    'tsv': MediaType.TEXT, 'json': MediaType.TEXT, 'xml': MediaType.TEXT,
    'html': MediaType.TEXT, 'htm': MediaType.TEXT,

    # Audio
    'mp3': MediaType.AUDIO, 'wav': MediaType.AUDIO, 'm4a': MediaType.AUDIO,
    'ogg': MediaType.AUDIO, 'flac': MediaType.AUDIO, 'aac': MediaType.AUDIO,

    # Video
    'mp4': MediaType.VIDEO, 'avi': MediaType.VIDEO, 'mov': MediaType.VIDEO,
    'mkv': MediaType.VIDEO, 'webm': MediaType.VIDEO, 'wmv': MediaType.VIDEO,
}


def detect_media_type(file_path: Union[str, Path]) -> MediaType:
    """
    Detect the media type of a file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        MediaType enum value
    """
    path = Path(file_path)
    extension = path.suffix.lower().lstrip('.')

    return FILE_TYPE_MAPPINGS.get(extension, MediaType.DOCUMENT)


def create_media_content(
    file_path: Union[str, Path],
    content_format: ContentFormat = ContentFormat.AUTO,
    mime_type: str = "auto"
) -> MediaContent:
    """
    Create a MediaContent object from a file path.

    Args:
        file_path: Path to the media file
        content_format: How to represent the content
        mime_type: MIME type of the content (auto-detected if "auto")

    Returns:
        MediaContent object
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    media_type = detect_media_type(path)

    # Auto-select content format based on media type
    if content_format == ContentFormat.AUTO:
        if media_type in [MediaType.IMAGE, MediaType.AUDIO, MediaType.VIDEO]:
            content_format = ContentFormat.BASE64
        else:
            content_format = ContentFormat.TEXT

    # Read and encode content based on format
    if content_format == ContentFormat.BASE64:
        with open(path, 'rb') as f:
            content = base64.b64encode(f.read()).decode('utf-8')
    elif content_format == ContentFormat.TEXT:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    elif content_format == ContentFormat.FILE_PATH:
        content = str(path)
    else:
        with open(path, 'rb') as f:
            content = f.read()

    return MediaContent(
        media_type=media_type,
        content=content,
        content_format=content_format,
        mime_type=mime_type,
        file_path=str(path),
        metadata={
            'file_size': path.stat().st_size,
            'file_name': path.name,
            'file_extension': path.suffix
        }
    )