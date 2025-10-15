"""
Media handling for different providers.
"""

import base64
from pathlib import Path
from typing import Union, Dict, Any, Optional
from enum import Enum


class MediaType(Enum):
    """Supported media types"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class MediaHandler:
    """Base class for media handling"""

    @staticmethod
    def encode_image(image_path: Union[str, Path]) -> str:
        """
        Encode an image file to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def format_for_openai(image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Format image for OpenAI API.

        Args:
            image_path: Path to the image

        Returns:
            Formatted content for OpenAI
        """
        base64_image = MediaHandler.encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

    @staticmethod
    def format_for_anthropic(image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Format image for Anthropic API.

        Args:
            image_path: Path to the image

        Returns:
            Formatted content for Anthropic
        """
        base64_image = MediaHandler.encode_image(image_path)

        # Detect image type
        path = Path(image_path)
        media_type = "image/jpeg"
        if path.suffix.lower() == ".png":
            media_type = "image/png"
        elif path.suffix.lower() == ".gif":
            media_type = "image/gif"
        elif path.suffix.lower() == ".webp":
            media_type = "image/webp"

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_image
            }
        }

    @staticmethod
    def format_for_provider(image_path: Union[str, Path], provider: str) -> Optional[Dict[str, Any]]:
        """
        Format media for a specific provider.

        Args:
            image_path: Path to the media file
            provider: Provider name

        Returns:
            Formatted content or None if not supported
        """
        provider_lower = provider.lower()

        if provider_lower == "openai":
            return MediaHandler.format_for_openai(image_path)
        elif provider_lower == "anthropic":
            return MediaHandler.format_for_anthropic(image_path)
        else:
            # Local providers typically don't support images directly
            return None

    @staticmethod
    def is_image_file(path: Union[str, Path]) -> bool:
        """
        Check if a file is an image.

        Args:
            path: Path to check

        Returns:
            True if the file is an image
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.ico', '.tiff'}
        return Path(path).suffix.lower() in image_extensions

    @staticmethod
    def get_media_type(path: Union[str, Path]) -> MediaType:
        """
        Determine the media type of a file.

        Args:
            path: Path to the file

        Returns:
            MediaType enum value
        """
        path = Path(path)
        extension = path.suffix.lower()

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        audio_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md'}

        if extension in image_extensions:
            return MediaType.IMAGE
        elif extension in audio_extensions:
            return MediaType.AUDIO
        elif extension in video_extensions:
            return MediaType.VIDEO
        elif extension in document_extensions:
            return MediaType.DOCUMENT
        else:
            return MediaType.DOCUMENT  # Default to document