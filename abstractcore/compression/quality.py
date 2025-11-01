"""
Quality validation and assessment for Glyph compression.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class CompressionStats:
    """Statistics for compression operation."""
    
    compression_ratio: float
    quality_score: float
    token_savings: int
    processing_time: float
    provider_optimized: str
    original_tokens: int
    compressed_tokens: int
    cost_savings: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'compression_ratio': self.compression_ratio,
            'quality_score': self.quality_score,
            'token_savings': self.token_savings,
            'processing_time': self.processing_time,
            'provider_optimized': self.provider_optimized,
            'original_tokens': self.original_tokens,
            'compressed_tokens': self.compressed_tokens,
            'cost_savings': self.cost_savings
        }


class QualityValidator:
    """Validates compression quality using multiple metrics."""

    def __init__(self):
        """Initialize quality validator."""
        from ..utils.structured_logging import get_logger
        self.logger = get_logger(self.__class__.__name__)
        self.validation_methods = [
            self._validate_compression_ratio,
            self._validate_content_preservation,
            self._validate_readability
        ]
    
    def assess(self, original_content: str, rendered_images: List[Path], 
               provider: str = None) -> float:
        """
        Assess compression quality using multiple metrics.
        
        Args:
            original_content: Original text content
            rendered_images: List of rendered image paths
            provider: Provider name for provider-specific assessment
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        scores = []
        
        # Basic compression ratio validation
        ratio_score = self._validate_compression_ratio(original_content, rendered_images)
        scores.append(ratio_score)
        
        # Content preservation assessment
        preservation_score = self._validate_content_preservation(original_content, rendered_images)
        scores.append(preservation_score)
        
        # Readability assessment
        readability_score = self._validate_readability(original_content, rendered_images, provider)
        scores.append(readability_score)
        
        # Weighted average (can be tuned based on empirical results)
        weights = [0.3, 0.4, 0.3]  # ratio, preservation, readability
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return min(1.0, max(0.0, weighted_score))
    
    def _validate_compression_ratio(self, original_content: str, rendered_images: List[Path]) -> float:
        """Validate compression ratio is within expected range."""
        from ..utils.token_utils import TokenUtils
        
        # Estimate original tokens
        original_tokens = TokenUtils.estimate_tokens(original_content)
        
        # Estimate compressed tokens (rough approximation)
        # Calculate accurate token count using proper VLM token calculation
        from ..utils.vlm_token_calculator import VLMTokenCalculator
        from pathlib import Path
        
        calculator = VLMTokenCalculator()
        try:
            # Get provider from context or use default
            provider_name = getattr(self, '_provider', 'openai')
            model_name = getattr(self, '_model', '')
            
            # Calculate tokens for all rendered images
            if rendered_images and len(rendered_images) > 0:
                # Assume rendered_images contains file paths or can be converted to paths
                image_paths = []
                for img in rendered_images:
                    if hasattr(img, 'path') and img.path:
                        image_paths.append(Path(img.path))
                    elif isinstance(img, (str, Path)):
                        image_paths.append(Path(img))
                
                if image_paths:
                    token_analysis = calculator.calculate_tokens_for_images(
                        image_paths=image_paths,
                        provider=provider_name,
                        model=model_name
                    )
                    compressed_tokens = token_analysis['total_tokens']
                    self.logger.debug(f"Calculated {compressed_tokens} tokens for {len(image_paths)} images using {provider_name}")
                else:
                    # Fallback to provider-specific base estimation
                    base_tokens = calculator.PROVIDER_CONFIGS.get(provider_name, {}).get('base_tokens', 512)
                    compressed_tokens = len(rendered_images) * base_tokens
                    self.logger.warning(f"Using fallback estimation: {compressed_tokens} tokens for {len(rendered_images)} images")
            else:
                compressed_tokens = 0
                
        except Exception as e:
            # Fallback to conservative estimate if calculation fails
            self.logger.warning(f"VLM token calculation failed, using fallback: {e}")
            compressed_tokens = len(rendered_images) * 1500  # Conservative fallback
        
        if original_tokens == 0:
            return 0.0
        
        compression_ratio = original_tokens / compressed_tokens
        
        # Score based on compression ratio
        # Target: 3-4x compression
        if 2.5 <= compression_ratio <= 5.0:
            return 1.0  # Excellent compression
        elif 2.0 <= compression_ratio < 2.5 or 5.0 < compression_ratio <= 6.0:
            return 0.8  # Good compression
        elif 1.5 <= compression_ratio < 2.0 or 6.0 < compression_ratio <= 8.0:
            return 0.6  # Acceptable compression
        else:
            return 0.3  # Poor compression ratio
    
    def _validate_content_preservation(self, original_content: str, rendered_images: List[Path]) -> float:
        """Validate that content is preserved in rendering."""
        # Basic heuristics for content preservation
        score = 1.0
        
        # Check for content length preservation
        # Images should roughly correspond to content length
        content_length = len(original_content)
        expected_images = max(1, content_length // 5000)  # ~5000 chars per image
        actual_images = len(rendered_images)
        
        # Penalize significant deviations
        if actual_images < expected_images * 0.5 or actual_images > expected_images * 2:
            score *= 0.8
        
        # Check for special characters and formatting
        special_chars = sum(1 for c in original_content if not c.isalnum() and not c.isspace())
        if special_chars > len(original_content) * 0.3:  # >30% special chars
            score *= 0.9  # Slight penalty for complex formatting
        
        # Check for very long lines (may cause rendering issues)
        lines = original_content.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 200)
        if long_lines > len(lines) * 0.2:  # >20% long lines
            score *= 0.95
        
        return score
    
    def _validate_readability(self, original_content: str, rendered_images: List[Path], 
                            provider: str = None) -> float:
        """Validate readability for the target provider."""
        score = 1.0
        
        # Provider-specific readability assessment
        if provider == "openai":
            # GPT-4o has excellent OCR, can handle dense text
            score = 1.0
        elif provider == "anthropic":
            # Claude is font-sensitive, prefers clear rendering
            score = 0.95
        elif provider and "qwen" in provider.lower():
            # Qwen models similar to Glyph's base model
            score = 1.0
        elif provider and "llava" in provider.lower():
            # LLaVA models have limited OCR
            score = 0.8
        else:
            # Conservative score for unknown providers
            score = 0.85
        
        # Adjust based on content characteristics
        # Check for code content (harder to read in images)
        code_indicators = ['def ', 'class ', 'import ', 'function', '{', '}', '#!/']
        code_score = sum(1 for indicator in code_indicators if indicator in original_content)
        if code_score > 5:
            score *= 0.9  # Slight penalty for code content
        
        # Check for mathematical notation (challenging for OCR)
        math_indicators = ['∑', '∫', '∂', '√', '±', '≤', '≥', '≠', '∞']
        math_score = sum(1 for indicator in math_indicators if indicator in original_content)
        if math_score > 0:
            score *= 0.85  # Penalty for mathematical notation
        
        return score
    
    def get_provider_threshold(self, provider: str) -> float:
        """Get quality threshold for specific provider."""
        thresholds = {
            "openai": 0.93,
            "anthropic": 0.96,
            "ollama": 0.90,
            "lmstudio": 0.94,
            "mlx": 0.88,
            "huggingface": 0.85
        }
        
        return thresholds.get(provider, 0.90)  # Default threshold

