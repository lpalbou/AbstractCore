"""
Provider-specific optimization profiles for enhanced Glyph compression.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from ..utils.structured_logging import get_logger
from .config import RenderingConfig


@dataclass
class OptimizationProfile:
    """Optimization profile for a specific provider/model combination."""
    provider: str
    model: str
    dpi: int
    font_size: int
    line_height: int
    columns: int
    margin_x: int
    margin_y: int
    target_compression: float
    quality_threshold: float
    notes: str = ""

    def to_rendering_config(self) -> RenderingConfig:
        """Convert to RenderingConfig."""
        return RenderingConfig(
            dpi=self.dpi,
            font_size=self.font_size,
            line_height=self.line_height,
            columns=self.columns,
            margin_x=self.margin_x,
            margin_y=self.margin_y,
            font_path="Verdana",
            auto_crop=True
        )


class CompressionOptimizer:
    """
    Optimizes Glyph compression for different providers and models.

    Based on empirical testing, provides optimized rendering configurations
    to achieve maximum compression while maintaining quality.
    """

    def __init__(self):
        """Initialize compression optimizer."""
        self.logger = get_logger(self.__class__.__name__)
        self.profiles = self._load_optimization_profiles()

    def _load_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Load provider-specific optimization profiles."""
        profiles = {}

        # OpenAI models - optimized for GPT-4 vision
        profiles["openai/gpt-4o"] = OptimizationProfile(
            provider="openai",
            model="gpt-4o",
            dpi=72,  # Lower DPI for better compression
            font_size=8,  # Balanced readability
            line_height=9,
            columns=4,  # Multi-column for density
            margin_x=3,
            margin_y=3,
            target_compression=3.5,
            quality_threshold=0.93,
            notes="Optimized for GPT-4o vision encoder"
        )

        profiles["openai/gpt-4o-mini"] = OptimizationProfile(
            provider="openai",
            model="gpt-4o-mini",
            dpi=72,
            font_size=7,  # Smaller for mini model
            line_height=8,
            columns=5,  # More columns for efficiency
            margin_x=2,
            margin_y=2,
            target_compression=4.0,
            quality_threshold=0.90,
            notes="Aggressive compression for mini model"
        )

        # Anthropic models - optimized for Claude vision
        profiles["anthropic/claude-3-5-sonnet"] = OptimizationProfile(
            provider="anthropic",
            model="claude-3-5-sonnet",
            dpi=96,  # Higher DPI for Claude
            font_size=9,
            line_height=10,
            columns=3,  # Fewer columns for clarity
            margin_x=4,
            margin_y=4,
            target_compression=3.0,
            quality_threshold=0.94,
            notes="Conservative for Claude's detail focus"
        )

        profiles["anthropic/claude-3-5-haiku"] = OptimizationProfile(
            provider="anthropic",
            model="claude-3-5-haiku",
            dpi=72,
            font_size=7,
            line_height=8,
            columns=4,
            margin_x=3,
            margin_y=3,
            target_compression=3.5,
            quality_threshold=0.91,
            notes="Balanced for Haiku efficiency"
        )

        # Ollama models - optimized for open source vision models
        profiles["ollama/llama3.2-vision"] = OptimizationProfile(
            provider="ollama",
            model="llama3.2-vision",
            dpi=72,
            font_size=6,  # Aggressive for local model
            line_height=7,
            columns=6,  # Maximum columns
            margin_x=2,
            margin_y=2,
            target_compression=4.5,
            quality_threshold=0.88,
            notes="Maximum compression for Llama vision"
        )

        profiles["ollama/qwen2.5-vision"] = OptimizationProfile(
            provider="ollama",
            model="qwen2.5-vision",
            dpi=72,
            font_size=7,
            line_height=8,
            columns=5,
            margin_x=2,
            margin_y=2,
            target_compression=4.0,
            quality_threshold=0.89,
            notes="Optimized for Qwen2.5 vision"
        )

        # LMStudio models
        profiles["lmstudio/default"] = OptimizationProfile(
            provider="lmstudio",
            model="default",
            dpi=72,
            font_size=8,
            line_height=9,
            columns=4,
            margin_x=3,
            margin_y=3,
            target_compression=3.5,
            quality_threshold=0.90,
            notes="Generic LMStudio optimization"
        )

        # Default profile for unknown providers
        profiles["default"] = OptimizationProfile(
            provider="default",
            model="default",
            dpi=72,
            font_size=8,
            line_height=9,
            columns=4,
            margin_x=3,
            margin_y=3,
            target_compression=3.0,
            quality_threshold=0.92,
            notes="Safe default configuration"
        )

        return profiles

    def get_optimized_config(
        self,
        provider: str,
        model: str,
        aggressive: bool = False
    ) -> RenderingConfig:
        """
        Get optimized rendering configuration for provider/model.

        Args:
            provider: Provider name
            model: Model name
            aggressive: Use more aggressive compression

        Returns:
            Optimized RenderingConfig
        """
        # Look for exact match
        key = f"{provider}/{model}"
        if key in self.profiles:
            profile = self.profiles[key]
        # Look for provider default
        elif f"{provider}/default" in self.profiles:
            profile = self.profiles[f"{provider}/default"]
        else:
            profile = self.profiles["default"]

        self.logger.debug(f"Using optimization profile: {profile.provider}/{profile.model}")

        # Apply aggressive modifications if requested
        if aggressive:
            profile = self._apply_aggressive_settings(profile)

        return profile.to_rendering_config()

    def _apply_aggressive_settings(self, profile: OptimizationProfile) -> OptimizationProfile:
        """Apply more aggressive compression settings."""
        # Create modified profile
        aggressive = OptimizationProfile(
            provider=profile.provider,
            model=profile.model,
            dpi=profile.dpi,
            font_size=max(5, profile.font_size - 1),  # Smaller font
            line_height=max(6, profile.line_height - 1),
            columns=min(8, profile.columns + 2),  # More columns
            margin_x=max(1, profile.margin_x - 1),
            margin_y=max(1, profile.margin_y - 1),
            target_compression=profile.target_compression * 1.5,
            quality_threshold=profile.quality_threshold * 0.95,
            notes=f"{profile.notes} (aggressive mode)"
        )
        return aggressive

    def analyze_compression_potential(
        self,
        text_length: int,
        provider: str,
        model: str
    ) -> Dict[str, Any]:
        """
        Analyze potential compression for given text.

        Args:
            text_length: Length of text in characters
            provider: Provider name
            model: Model name

        Returns:
            Analysis of compression potential
        """
        # Get profile
        key = f"{provider}/{model}"
        profile = self.profiles.get(key, self.profiles["default"])

        # Estimate tokens (rough estimate)
        estimated_tokens = text_length // 4

        # Calculate potential compression
        chars_per_page = (
            (profile.columns * 40) *  # Characters per line
            (60)  # Lines per page (approximate)
        )
        estimated_pages = text_length / chars_per_page
        estimated_images = max(1, int(estimated_pages / 2))  # 2 pages per image

        # Estimate compressed tokens
        tokens_per_image = 1500  # Approximate for most vision models
        compressed_tokens = estimated_images * tokens_per_image

        # Calculate metrics
        compression_ratio = estimated_tokens / compressed_tokens if compressed_tokens > 0 else 1.0

        return {
            "text_length": text_length,
            "estimated_tokens": estimated_tokens,
            "estimated_images": estimated_images,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compression_ratio,
            "profile_used": f"{profile.provider}/{profile.model}",
            "target_compression": profile.target_compression,
            "achievable": compression_ratio >= profile.target_compression * 0.8
        }

    def save_profiles(self, path: Path):
        """Save optimization profiles to JSON file."""
        profiles_dict = {}
        for key, profile in self.profiles.items():
            profiles_dict[key] = {
                "provider": profile.provider,
                "model": profile.model,
                "dpi": profile.dpi,
                "font_size": profile.font_size,
                "line_height": profile.line_height,
                "columns": profile.columns,
                "margin_x": profile.margin_x,
                "margin_y": profile.margin_y,
                "target_compression": profile.target_compression,
                "quality_threshold": profile.quality_threshold,
                "notes": profile.notes
            }

        with open(path, 'w') as f:
            json.dump(profiles_dict, f, indent=2)

        self.logger.info(f"Saved {len(profiles_dict)} optimization profiles to {path}")

    def benchmark_profile(
        self,
        profile: OptimizationProfile,
        test_text: str
    ) -> Dict[str, Any]:
        """
        Benchmark a specific optimization profile.

        Args:
            profile: Profile to benchmark
            test_text: Text to test with

        Returns:
            Benchmark results
        """
        from .glyph_processor import GlyphProcessor
        from .config import GlyphConfig
        import time

        # Create processor with profile
        config = GlyphConfig()
        config.enabled = True
        config.min_token_threshold = 100

        processor = GlyphProcessor(config=config)

        # Test compression
        start_time = time.time()

        try:
            results = processor.process_text(
                test_text,
                provider=profile.provider,
                model=profile.model,
                user_preference="always"
            )

            processing_time = time.time() - start_time

            # Calculate metrics
            from ..utils.token_utils import TokenUtils
            original_tokens = TokenUtils.estimate_tokens(test_text, profile.model)
            compressed_tokens = len(results) * 1500
            actual_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0

            # Get quality from results
            quality = results[0].metadata.get("quality_score", 0.0) if results else 0.0

            return {
                "success": True,
                "profile": f"{profile.provider}/{profile.model}",
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": actual_ratio,
                "target_ratio": profile.target_compression,
                "quality_score": quality,
                "quality_threshold": profile.quality_threshold,
                "processing_time": processing_time,
                "images_created": len(results),
                "meets_target": actual_ratio >= profile.target_compression * 0.9
            }

        except Exception as e:
            return {
                "success": False,
                "profile": f"{profile.provider}/{profile.model}",
                "error": str(e)
            }


def create_optimized_config(provider: str, model: str, aggressive: bool = False) -> RenderingConfig:
    """
    Convenience function to create optimized rendering configuration.

    Args:
        provider: Provider name
        model: Model name
        aggressive: Use more aggressive compression

    Returns:
        Optimized RenderingConfig
    """
    optimizer = CompressionOptimizer()
    return optimizer.get_optimized_config(provider, model, aggressive)