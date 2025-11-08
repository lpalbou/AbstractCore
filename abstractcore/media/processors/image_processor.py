"""
Image processor for vision model support.

This module provides comprehensive image processing capabilities using PIL,
optimized for vision model inputs across different providers.
"""

from __future__ import annotations  # PEP 563 - deferred type hint evaluation for optional dependencies

import base64
import io
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

try:
    from PIL import Image, ImageOps, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageOps = None
    ExifTags = None

from ..base import BaseMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat
from ..utils.image_scaler import get_scaler, ScalingMode


class ImageProcessor(BaseMediaHandler):
    """
    Image processor using PIL for vision model support.

    Handles image loading, preprocessing, format conversion, and optimization
    for various vision models across different providers.
    """

    def __init__(self, **kwargs):
        """
        Initialize the image processor.

        Args:
            **kwargs: Configuration parameters including:
                - max_resolution: Maximum image resolution (width, height)
                - quality: JPEG quality (1-100)
                - auto_rotate: Whether to auto-rotate based on EXIF
                - resize_mode: How to resize ('fit', 'crop', 'stretch')
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow is required for image processing. "
                "Install with: pip install \"abstractcore[media]\""
            )

        super().__init__(**kwargs)

        # Image processing configuration - Use maximum resolution for best quality
        self.max_resolution = kwargs.get('max_resolution', (4096, 4096))  # Increased default for better quality
        self.quality = kwargs.get('quality', 90)  # Increased quality for better results
        self.auto_rotate = kwargs.get('auto_rotate', True)
        self.resize_mode = kwargs.get('resize_mode', 'fit')  # 'fit', 'crop', 'stretch'
        self.prefer_max_resolution = kwargs.get('prefer_max_resolution', True)  # Always use max when possible

        # Set capabilities for image processing
        from ..types import MediaCapabilities
        self.capabilities = MediaCapabilities(
            vision_support=True,
            audio_support=False,
            video_support=False,
            document_support=False,
            supported_image_formats=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'webp'],
            max_file_size=self.max_file_size
        )

        self.logger.debug(
            f"Initialized ImageProcessor with max_resolution={self.max_resolution}, "
            f"quality={self.quality}, auto_rotate={self.auto_rotate}, prefer_max_resolution={self.prefer_max_resolution}"
        )

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Process an image file and return optimized content for vision models.

        Args:
            file_path: Path to the image file
            media_type: Detected media type (should be IMAGE)
            **kwargs: Additional processing parameters:
                - target_format: Target format ('png', 'jpeg', 'webp')
                - max_resolution: Override default max resolution
                - quality: Override default quality
                - auto_rotate: Override default auto rotation

        Returns:
            MediaContent with base64-encoded optimized image

        Raises:
            MediaProcessingError: If image processing fails
        """
        if media_type != MediaType.IMAGE:
            raise MediaProcessingError(f"ImageProcessor only handles images, got {media_type}")

        try:
            # Override defaults with kwargs
            # Preserve original format unless explicitly specified
            original_format = file_path.suffix.lower().lstrip('.')
            if original_format == 'jpg':
                original_format = 'jpeg'
            target_format = kwargs.get('target_format', original_format if original_format in ['png', 'jpeg', 'webp', 'gif'] else 'jpeg')
            model_name = kwargs.get('model_name', None)

            # Use model-specific maximum resolution if available
            if model_name and self.prefer_max_resolution:
                max_resolution = self._get_model_max_resolution(model_name)
                self.logger.debug(f"Using model-specific max resolution for {model_name}: {max_resolution}")
            else:
                max_resolution = kwargs.get('max_resolution', self.max_resolution)

            quality = kwargs.get('quality', self.quality)
            auto_rotate = kwargs.get('auto_rotate', self.auto_rotate)

            # Load and process the image
            with Image.open(file_path) as img:
                # Auto-rotate based on EXIF data
                if auto_rotate:
                    img = self._auto_rotate_image(img)

                # Convert to RGB if necessary (for JPEG output)
                if target_format.lower() in ['jpeg', 'jpg'] and img.mode in ['RGBA', 'P', 'LA']:
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background

                # Resize if needed
                if max_resolution and self._needs_resize(img.size, max_resolution):
                    img = self._resize_image(img, max_resolution)

                # Optimize the image
                img = self._optimize_image(img)

                # Convert to base64
                base64_content = self._image_to_base64(img, target_format, quality)

                # Determine MIME type
                mime_type = self._get_mime_type(target_format)

                # Create metadata
                metadata = {
                    'original_format': file_path.suffix.lower().lstrip('.'),
                    'target_format': target_format,
                    'original_size': img.size if hasattr(img, 'size') else None,
                    'final_size': img.size,
                    'color_mode': img.mode,
                    'quality': quality if target_format.lower() in ['jpeg', 'jpg'] else None,
                    'auto_rotated': auto_rotate,
                    'optimized': True
                }

                # Add EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    metadata['has_exif'] = True
                    # Extract useful EXIF data
                    exif_data = self._extract_useful_exif(img)
                    if exif_data:
                        metadata['exif'] = exif_data

                return self._create_media_content(
                    content=base64_content,
                    file_path=file_path,
                    media_type=MediaType.IMAGE,
                    content_format=ContentFormat.BASE64,
                    mime_type=mime_type,
                    **metadata
                )

        except Exception as e:
            raise MediaProcessingError(f"Failed to process image {file_path}: {str(e)}") from e

    def _auto_rotate_image(self, img: Image.Image) -> Image.Image:
        """
        Auto-rotate image based on EXIF orientation data.

        Args:
            img: PIL Image object

        Returns:
            Rotated image
        """
        try:
            return ImageOps.exif_transpose(img)
        except Exception:
            # If auto-rotation fails, return original image
            return img

    def _get_model_max_resolution(self, model_name: Optional[str] = None) -> Tuple[int, int]:
        """
        Get maximum resolution for a specific model or return default high resolution.

        Args:
            model_name: Name of the model to check capabilities for

        Returns:
            Maximum resolution tuple (width, height)
        """
        if not model_name or not self.prefer_max_resolution:
            return self.max_resolution

        try:
            from ..capabilities import get_media_capabilities
            caps = get_media_capabilities(model_name)

            if hasattr(caps, 'image_resolutions') and caps.image_resolutions:
                resolution_str = caps.image_resolutions
                if isinstance(resolution_str, list) and resolution_str:
                    resolution_str = resolution_str[0]
                else:
                    resolution_str = str(resolution_str)

                # Parse resolution strings like "3584x3584", "56x56 to 3584x3584", "variable"
                if "to" in resolution_str:
                    # Extract maximum from range like "56x56 to 3584x3584"
                    max_part = resolution_str.split("to")[-1].strip()
                    if "x" in max_part:
                        width, height = map(int, max_part.split("x"))
                        return (width, height)
                elif "x" in resolution_str and "variable" not in resolution_str.lower():
                    # Parse direct resolution like "896x896"
                    width, height = map(int, resolution_str.split("x"))
                    return (width, height)
                elif "variable" in resolution_str.lower():
                    # For variable resolution models, use a high default
                    return (4096, 4096)

        except Exception as e:
            self.logger.debug(f"Could not get model-specific resolution for {model_name}: {e}")

        # Fallback to default high resolution
        return self.max_resolution

    def _needs_resize(self, current_size: Tuple[int, int], max_resolution: Tuple[int, int]) -> bool:
        """
        Check if image needs resizing.

        Args:
            current_size: Current image size (width, height)
            max_resolution: Maximum allowed resolution (width, height)

        Returns:
            True if resizing is needed
        """
        return current_size[0] > max_resolution[0] or current_size[1] > max_resolution[1]

    def _resize_image(self, img: Image.Image, max_resolution: Tuple[int, int]) -> Image.Image:
        """
        Resize image according to the specified mode.

        Args:
            img: PIL Image object
            max_resolution: Maximum allowed resolution (width, height)

        Returns:
            Resized image
        """
        if self.resize_mode == 'fit':
            # Maintain aspect ratio, fit within bounds
            img.thumbnail(max_resolution, Image.Resampling.LANCZOS)
            return img
        elif self.resize_mode == 'crop':
            # Maintain aspect ratio, crop to exact size
            return ImageOps.fit(img, max_resolution, Image.Resampling.LANCZOS)
        elif self.resize_mode == 'stretch':
            # Stretch to exact size (may distort)
            return img.resize(max_resolution, Image.Resampling.LANCZOS)
        else:
            # Default to fit
            img.thumbnail(max_resolution, Image.Resampling.LANCZOS)
            return img

    def _optimize_image(self, img: Image.Image) -> Image.Image:
        """
        Apply optimization to the image.

        Args:
            img: PIL Image object

        Returns:
            Optimized image
        """
        # For now, just return the image as-is
        # Future optimizations could include:
        # - Color palette optimization
        # - Compression-specific optimizations
        # - Noise reduction
        return img

    def _image_to_base64(self, img: Image.Image, format: str, quality: int) -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            img: PIL Image object
            format: Target format ('jpeg', 'png', 'webp')
            quality: Quality setting (for JPEG/WebP)

        Returns:
            Base64-encoded image string
        """
        buffer = io.BytesIO()

        # Set format-specific options
        save_kwargs = {}
        if format.lower() in ['jpeg', 'jpg']:
            format = 'JPEG'
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        elif format.lower() == 'png':
            format = 'PNG'
            save_kwargs['optimize'] = True
        elif format.lower() == 'webp':
            format = 'WebP'
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True

        # Save image to buffer
        img.save(buffer, format=format, **save_kwargs)
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _get_mime_type(self, format: str) -> str:
        """
        Get MIME type for the given format.

        Args:
            format: Image format

        Returns:
            MIME type string
        """
        mime_map = {
            'jpeg': 'image/jpeg',
            'jpg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp',
            'tiff': 'image/tiff',
            'tif': 'image/tiff'
        }
        return mime_map.get(format.lower(), 'image/jpeg')

    def _extract_useful_exif(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Extract useful EXIF data from image.

        Args:
            img: PIL Image object

        Returns:
            Dictionary of useful EXIF data or None
        """
        try:
            exif = img._getexif()
            if not exif:
                return None

            useful_data = {}

            # Map of useful EXIF tags
            useful_tags = {
                'DateTime': 'datetime',
                'DateTimeOriginal': 'datetime_original',
                'Make': 'camera_make',
                'Model': 'camera_model',
                'Software': 'software',
                'Orientation': 'orientation',
                'XResolution': 'x_resolution',
                'YResolution': 'y_resolution',
                'ResolutionUnit': 'resolution_unit'
            }

            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in useful_tags:
                    useful_data[useful_tags[tag]] = value

            return useful_data if useful_data else None

        except Exception:
            return None

    def get_image_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive information about an image without full processing.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with image information
        """
        file_path = Path(file_path)

        try:
            with Image.open(file_path) as img:
                info = {
                    'filename': file_path.name,
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.size[0],
                    'height': img.size[1],
                    'file_size': file_path.stat().st_size,
                    'has_transparency': img.mode in ['RGBA', 'LA', 'P'] and 'transparency' in img.info
                }

                # Add EXIF info if available
                exif_data = self._extract_useful_exif(img)
                if exif_data:
                    info['exif'] = exif_data

                return info

        except Exception as e:
            return {
                'filename': file_path.name,
                'error': str(e),
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            }

    def create_thumbnail(self, file_path: Union[str, Path], size: Tuple[int, int] = (128, 128)) -> str:
        """
        Create a thumbnail of the image.

        Args:
            file_path: Path to the image file
            size: Thumbnail size (width, height)

        Returns:
            Base64-encoded thumbnail
        """
        file_path = Path(file_path)

        try:
            with Image.open(file_path) as img:
                # Auto-rotate if needed
                if self.auto_rotate:
                    img = self._auto_rotate_image(img)

                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)

                # Convert to base64
                return self._image_to_base64(img, 'jpeg', 75)

        except Exception as e:
            raise MediaProcessingError(f"Failed to create thumbnail for {file_path}: {str(e)}") from e

    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get information about the image processor capabilities.

        Returns:
            Dictionary with processor information
        """
        return {
            'processor_type': 'ImageProcessor',
            'supported_formats': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
            'capabilities': {
                'max_resolution': self.max_resolution,
                'quality': self.quality,
                'auto_rotate': self.auto_rotate,
                'resize_mode': self.resize_mode,
                'base64_output': True,
                'exif_handling': True,
                'thumbnail_creation': True,
                'model_optimized_scaling': True
            },
            'dependencies': {
                'PIL': PIL_AVAILABLE
            }
        }

    def process_for_model(self, file_path: Union[str, Path], model_name: str, **kwargs) -> MediaContent:
        """
        Process image optimally for a specific vision model.

        Args:
            file_path: Path to the image file
            model_name: Name of the target vision model
            **kwargs: Additional processing parameters:
                - scaling_mode: ScalingMode for image scaling
                - target_format: Target format ('png', 'jpeg', 'webp')
                - quality: Image quality (1-100)
                - auto_rotate: Whether to auto-rotate based on EXIF

        Returns:
            MediaContent optimized for the specified model

        Raises:
            MediaProcessingError: If processing fails
        """
        file_path = Path(file_path)

        try:
            # Get scaling mode from kwargs or use default
            scaling_mode = kwargs.get('scaling_mode', ScalingMode.FIT)
            if isinstance(scaling_mode, str):
                scaling_mode = ScalingMode(scaling_mode)

            # Override other defaults with kwargs
            target_format = kwargs.get('target_format', 'jpeg')
            quality = kwargs.get('quality', self.quality)
            auto_rotate = kwargs.get('auto_rotate', self.auto_rotate)

            # Load the image
            with Image.open(file_path) as img:
                # Auto-rotate based on EXIF data
                if auto_rotate:
                    img = self._auto_rotate_image(img)

                # Get model-optimized scaler
                scaler = get_scaler()

                # Scale image for the specific model
                img = scaler.scale_for_model(img, model_name, scaling_mode)

                # Convert to RGB if necessary (for JPEG output)
                if target_format.lower() in ['jpeg', 'jpg'] and img.mode in ['RGBA', 'P', 'LA']:
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background

                # Optimize the image
                img = self._optimize_image(img)

                # Convert to base64
                base64_content = self._image_to_base64(img, target_format, quality)

                # Determine MIME type
                mime_type = self._get_mime_type(target_format)

                # Get optimal resolution for metadata
                optimal_size = scaler.get_optimal_resolution(model_name, img.size)

                # Create metadata
                metadata = {
                    'original_format': file_path.suffix.lower().lstrip('.'),
                    'target_format': target_format,
                    'final_size': img.size,
                    'optimal_size_for_model': optimal_size,
                    'target_model': model_name,
                    'scaling_mode': scaling_mode.value,
                    'color_mode': img.mode,
                    'quality': quality if target_format.lower() in ['jpeg', 'jpg'] else None,
                    'auto_rotated': auto_rotate,
                    'model_optimized': True
                }

                # Add EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    metadata['has_exif'] = True
                    # Extract useful EXIF data
                    exif_data = self._extract_useful_exif(img)
                    if exif_data:
                        metadata['exif'] = exif_data

                return self._create_media_content(
                    content=base64_content,
                    file_path=file_path,
                    media_type=MediaType.IMAGE,
                    content_format=ContentFormat.BASE64,
                    mime_type=mime_type,
                    **metadata
                )

        except Exception as e:
            raise MediaProcessingError(f"Failed to process image {file_path} for model {model_name}: {str(e)}") from e

    def get_optimal_size_for_model(self, model_name: str, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get optimal image size for a specific model without processing the image.

        Args:
            model_name: Name of the target vision model
            original_size: Original image size (width, height)

        Returns:
            Optimal target size (width, height) for the model
        """
        scaler = get_scaler()
        return scaler.get_optimal_resolution(model_name, original_size)

    def supports_model(self, model_name: str) -> bool:
        """
        Check if the processor supports optimizations for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            True if model-specific optimizations are available
        """
        try:
            # Test if we can get capabilities for this model
            scaler = get_scaler()
            scaler._get_model_capabilities(model_name)
            return True
        except Exception:
            return False