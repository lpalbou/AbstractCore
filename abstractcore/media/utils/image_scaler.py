"""
Image scaling utility for AbstractCore media handling.

Provides intelligent image scaling based on model-specific requirements
and capabilities for vision models.
"""

from typing import Tuple, Optional, Union, Dict, Any
from enum import Enum
from pathlib import Path

from PIL import Image, ImageOps

from ..base import MediaProcessingError
from ...utils.structured_logging import get_logger


class ScalingMode(Enum):
    """Image scaling modes."""
    FIT = "fit"          # Scale to fit within target size, maintaining aspect ratio
    FILL = "fill"        # Scale to fill target size, may crop, maintaining aspect ratio
    STRETCH = "stretch"  # Stretch to exact target size, may distort aspect ratio
    PAD = "pad"          # Scale to fit and pad with background to exact target size
    CROP_CENTER = "crop_center"  # Scale to fill and crop from center


class ModelOptimizedScaler:
    """
    Intelligent image scaler that optimizes images for specific vision models.

    Uses model capability information to determine optimal scaling strategies.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def get_optimal_resolution(self, model_name: str, original_size: Tuple[int, int],
                             model_capabilities: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
        """
        Get optimal resolution for a specific model.

        Args:
            model_name: Name of the model
            original_size: Original image size (width, height)
            model_capabilities: Model capability information

        Returns:
            Optimal target size (width, height)
        """
        if model_capabilities is None:
            model_capabilities = self._get_model_capabilities(model_name)

        max_resolution = model_capabilities.get("max_image_resolution", "variable")
        image_patch_size = model_capabilities.get("image_patch_size", 16)
        adaptive_windowing = model_capabilities.get("adaptive_windowing", False)

        # Handle different resolution strategies
        if max_resolution == "variable":
            return self._optimize_variable_resolution(original_size, image_patch_size)
        elif max_resolution == "896x896":
            return (896, 896)
        elif max_resolution == "3584x3584":
            return self._optimize_large_resolution(original_size, (3584, 3584), image_patch_size)
        elif "x" in str(max_resolution):
            # Parse specific resolution like "1568x1568"
            w, h = map(int, str(max_resolution).split("x"))
            return (w, h)
        else:
            # Default fallback
            return self._optimize_variable_resolution(original_size, image_patch_size)

    def _optimize_variable_resolution(self, original_size: Tuple[int, int],
                                    patch_size: int = 16) -> Tuple[int, int]:
        """Optimize for variable resolution models like Qwen3-VL."""
        width, height = original_size

        # For variable resolution, aim for reasonable size that's efficient
        # while maintaining good quality
        max_dimension = 1024  # Good balance for most use cases

        # Scale down if too large
        if max(width, height) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
        else:
            new_width, new_height = width, height

        # Round to nearest patch size multiple for efficiency
        new_width = ((new_width + patch_size - 1) // patch_size) * patch_size
        new_height = ((new_height + patch_size - 1) // patch_size) * patch_size

        return (new_width, new_height)

    def _optimize_large_resolution(self, original_size: Tuple[int, int],
                                 max_size: Tuple[int, int],
                                 patch_size: int = 14) -> Tuple[int, int]:
        """Optimize for large resolution models like Qwen2.5-VL."""
        width, height = original_size
        max_width, max_height = max_size

        # Scale to fit within max size while maintaining aspect ratio
        scale = min(max_width / width, max_height / height)

        if scale < 1:  # Only scale down, never up
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = width, height

        # Round to nearest patch size multiple
        new_width = ((new_width + patch_size - 1) // patch_size) * patch_size
        new_height = ((new_height + patch_size - 1) // patch_size) * patch_size

        return (new_width, new_height)

    def scale_image(self, image: Image.Image, target_size: Tuple[int, int],
                   mode: ScalingMode = ScalingMode.FIT,
                   background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Scale image to target size using specified mode.

        Args:
            image: PIL Image to scale
            target_size: Target size (width, height)
            mode: Scaling mode
            background_color: Background color for padding (RGB)

        Returns:
            Scaled PIL Image
        """
        target_width, target_height = target_size

        if mode == ScalingMode.FIT:
            # Scale to fit within target size, maintaining aspect ratio
            image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            return image

        elif mode == ScalingMode.FILL:
            # Scale to fill target size, may crop
            return ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)

        elif mode == ScalingMode.STRETCH:
            # Stretch to exact target size
            return image.resize(target_size, Image.Resampling.LANCZOS)

        elif mode == ScalingMode.PAD:
            # Scale to fit and pad to exact size
            image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

            # Create new image with background color
            new_image = Image.new('RGB', target_size, background_color)

            # Paste scaled image centered
            paste_x = (target_width - image.width) // 2
            paste_y = (target_height - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))

            return new_image

        elif mode == ScalingMode.CROP_CENTER:
            # Scale to fill and crop from center
            return ImageOps.fit(image, target_size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        else:
            raise MediaProcessingError(f"Unknown scaling mode: {mode}")

    def scale_for_model(self, image: Image.Image, model_name: str,
                       scaling_mode: ScalingMode = ScalingMode.FIT,
                       model_capabilities: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Scale image optimally for a specific model.

        Args:
            image: PIL Image to scale
            model_name: Name of the target model
            scaling_mode: How to scale the image
            model_capabilities: Model capability information

        Returns:
            Optimally scaled PIL Image for the model
        """
        original_size = image.size
        target_size = self.get_optimal_resolution(model_name, original_size, model_capabilities)

        self.logger.debug(f"Scaling image for {model_name}: {original_size} -> {target_size}")

        # For fixed resolution models, always use PAD mode to maintain exact size
        if model_capabilities and model_capabilities.get("max_image_resolution") == "896x896":
            scaling_mode = ScalingMode.PAD

        return self.scale_image(image, target_size, scaling_mode)

    def _get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """
        Get model capabilities from the capabilities JSON.

        Args:
            model_name: Name of the model

        Returns:
            Model capabilities dictionary
        """
        try:
            from ..capabilities import get_media_capabilities
            return get_media_capabilities(model_name).__dict__
        except ImportError:
            # Fallback capability detection
            return self._fallback_model_capabilities(model_name)

    def _fallback_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Fallback capability detection when capabilities module not available."""
        model_lower = model_name.lower()

        # Gemma models - fixed 896x896
        if any(gem in model_lower for gem in ["gemma3", "gemma-3n"]):
            return {
                "max_image_resolution": "896x896",
                "image_patch_size": 16,
                "adaptive_windowing": True
            }

        # Qwen2.5-VL models - up to 3584x3584
        elif "qwen2.5" in model_lower and "vl" in model_lower:
            return {
                "max_image_resolution": "3584x3584",
                "image_patch_size": 14,
                "pixel_grouping": "28x28"
            }

        # Qwen3-VL models - variable resolution
        elif "qwen3" in model_lower and "vl" in model_lower:
            return {
                "max_image_resolution": "variable",
                "image_patch_size": 16,
                "pixel_grouping": "32x32"
            }

        # Claude models - up to 1568x1568
        elif "claude" in model_lower:
            return {
                "max_image_resolution": "1568x1568",
                "image_patch_size": 14
            }

        # Default fallback
        else:
            return {
                "max_image_resolution": "variable",
                "image_patch_size": 16
            }


# Convenience functions for easy usage
_scaler_instance = None

def get_scaler() -> ModelOptimizedScaler:
    """Get shared scaler instance."""
    global _scaler_instance
    if _scaler_instance is None:
        _scaler_instance = ModelOptimizedScaler()
    return _scaler_instance

def scale_image_for_model(image: Union[Image.Image, str, Path],
                         model_name: str,
                         scaling_mode: ScalingMode = ScalingMode.FIT) -> Image.Image:
    """
    Convenience function to scale an image for a specific model.

    Args:
        image: PIL Image, or path to image file
        model_name: Name of the target model
        scaling_mode: How to scale the image

    Returns:
        Optimally scaled PIL Image
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    scaler = get_scaler()
    return scaler.scale_for_model(image, model_name, scaling_mode)

def get_optimal_size_for_model(model_name: str, original_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Get optimal image size for a specific model.

    Args:
        model_name: Name of the target model
        original_size: Original image size (width, height)

    Returns:
        Optimal target size (width, height)
    """
    scaler = get_scaler()
    return scaler.get_optimal_resolution(model_name, original_size)