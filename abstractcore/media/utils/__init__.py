"""
Utility modules for AbstractCore media handling.

Provides reusable utilities for media processing, including image scaling
optimized for different vision models.
"""

from .image_scaler import (
    ModelOptimizedScaler,
    ScalingMode,
    get_scaler,
    scale_image_for_model,
    get_optimal_size_for_model
)

__all__ = [
    'ModelOptimizedScaler',
    'ScalingMode',
    'get_scaler',
    'scale_image_for_model',
    'get_optimal_size_for_model'
]