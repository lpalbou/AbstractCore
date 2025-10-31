"""
Compression orchestrator for intelligent decision-making.
"""

from typing import Optional, Union, Dict, Any, List
from pathlib import Path

from ..utils.token_utils import TokenUtils
from ..utils.structured_logging import get_logger
from .config import GlyphConfig
# Import GlyphProcessor lazily to avoid circular imports
from .exceptions import CompressionError


class CompressionOrchestrator:
    """Intelligent decision-making for when and how to apply Glyph compression."""
    
    def __init__(self, config: Optional[GlyphConfig] = None):
        """
        Initialize compression orchestrator.
        
        Args:
            config: Glyph configuration
        """
        self.config = config or GlyphConfig.from_abstractcore_config()
        self.logger = get_logger(self.__class__.__name__)
        self.glyph_processor = None  # Lazy initialization
        
        self.logger.debug("CompressionOrchestrator initialized")
    
    def _get_glyph_processor(self):
        """Get or create Glyph processor instance."""
        if self.glyph_processor is None:
            # Lazy import to avoid circular dependency
            from .glyph_processor import GlyphProcessor
            self.glyph_processor = GlyphProcessor(self.config)
        return self.glyph_processor
    
    def should_compress(
        self, 
        content: Union[str, Path],
        provider: str,
        model: str,
        user_preference: str = "auto"
    ) -> bool:
        """
        Intelligent compression decision based on multiple factors.
        
        Args:
            content: Text content or file path
            provider: Provider name
            model: Model name
            user_preference: User compression preference (auto, always, never)
            
        Returns:
            True if compression should be applied
        """
        # Check user preference first
        if user_preference == "never":
            return False
        elif user_preference == "always":
            return self._can_compress(content, provider, model)
        
        # Auto-decision logic
        try:
            # Get content as string
            if isinstance(content, Path):
                with open(content, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            else:
                text_content = content
            
            # Basic feasibility checks
            if not self._can_compress(text_content, provider, model):
                return False
            
            # Token-based decision
            token_count = TokenUtils.estimate_tokens(text_content, model)
            model_context = self._get_model_context_window(provider, model)
            
            # Decision matrix based on Glyph research
            if token_count < self.config.min_token_threshold:
                return False  # Too small to benefit
            elif token_count > model_context * 0.8:
                return True  # Necessary if approaching limits
            elif token_count > 50000:
                return True  # Beneficial for large texts
            else:
                return False  # Standard processing sufficient
                
        except Exception as e:
            self.logger.warning(f"Compression decision failed, defaulting to False: {e}")
            return False
    
    def _can_compress(self, content: Union[str, Path], provider: str, model: str) -> bool:
        """Check if compression is technically feasible."""
        try:
            # Check if compression is enabled
            if not self.config.enabled:
                return False
            
            # Check provider vision support
            if not self._supports_vision(provider, model):
                return False
            
            # Check content type suitability
            if isinstance(content, Path):
                if not self._is_compressible_file(content):
                    return False
            else:
                if not self._is_compressible_text(content):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _supports_vision(self, provider: str, model: str) -> bool:
        """Check if provider/model supports vision."""
        try:
            from ..media.capabilities import get_model_capabilities
            capabilities = get_model_capabilities(provider, model)
            return capabilities.get('vision_support', False)
        except Exception:
            # Conservative approach for unknown providers
            vision_providers = ['openai', 'anthropic', 'ollama', 'lmstudio']
            return any(vp in provider.lower() for vp in vision_providers)
    
    def _get_model_context_window(self, provider: str, model: str) -> int:
        """Get model context window size."""
        try:
            from ..media.capabilities import get_model_capabilities
            capabilities = get_model_capabilities(provider, model)
            return capabilities.get('context_window', 32000)
        except Exception:
            # Default context windows by provider
            defaults = {
                'openai': 128000,
                'anthropic': 200000,
                'ollama': 32000,
                'lmstudio': 32000,
                'mlx': 32000,
                'huggingface': 32000
            }
            return defaults.get(provider, 32000)
    
    def _is_compressible_file(self, file_path: Path) -> bool:
        """Check if file type is suitable for compression."""
        if not file_path.exists():
            return False
        
        # Check file extension
        compressible_extensions = {'.txt', '.md', '.csv', '.tsv', '.json', '.yaml', '.yml'}
        if file_path.suffix.lower() not in compressible_extensions:
            return False
        
        # Check file size (avoid very large files)
        try:
            file_size = file_path.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                return False
        except Exception:
            return False
        
        return True
    
    def _is_compressible_text(self, text: str) -> bool:
        """Check if text content is suitable for compression."""
        if not text or len(text.strip()) < 100:
            return False
        
        # Check for problematic content types
        # Mathematical notation (challenging for OCR)
        math_indicators = ['∑', '∫', '∂', '√', '±', '≤', '≥', '≠', '∞']
        math_count = sum(1 for indicator in math_indicators if indicator in text)
        if math_count > len(text) * 0.01:  # >1% mathematical symbols
            return False
        
        # Very dense special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars > len(text) * 0.5:  # >50% special characters
            return False
        
        return True
    
    def compress_content(
        self,
        content: Union[str, Path],
        provider: str,
        model: str,
        user_preference: str = "auto"
    ) -> Optional[List[Any]]:
        """
        Compress content if beneficial.
        
        Args:
            content: Text content or file path
            provider: Provider name
            model: Model name
            user_preference: User compression preference
            
        Returns:
            List of MediaContent objects if compressed, None if not compressed
        """
        try:
            # Check if compression should be applied
            if not self.should_compress(content, provider, model, user_preference):
                return None
            
            # Get content as string
            if isinstance(content, Path):
                with open(content, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            else:
                text_content = content
            
            # Apply compression
            processor = self._get_glyph_processor()
            compressed_content = processor.process_text(text_content, provider, model, user_preference)
            
            self.logger.info(f"Content compressed successfully: {len(compressed_content)} images")
            return compressed_content
            
        except Exception as e:
            self.logger.error(f"Content compression failed: {e}")
            raise CompressionError(f"Compression failed: {e}") from e
    
    def get_compression_recommendation(
        self,
        content: Union[str, Path],
        provider: str,
        model: str
    ) -> Dict[str, Any]:
        """
        Get detailed compression recommendation.
        
        Args:
            content: Text content or file path
            provider: Provider name
            model: Model name
            
        Returns:
            Dictionary with recommendation details
        """
        try:
            # Get content as string
            if isinstance(content, Path):
                with open(content, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                content_source = "file"
            else:
                text_content = content
                content_source = "text"
            
            # Analyze content
            token_count = TokenUtils.estimate_tokens(text_content, model)
            model_context = self._get_model_context_window(provider, model)
            supports_vision = self._supports_vision(provider, model)
            is_compressible = self._is_compressible_text(text_content)
            
            # Calculate potential benefits
            estimated_compression_ratio = self._estimate_compression_ratio(text_content, provider)
            estimated_savings = token_count - (token_count / estimated_compression_ratio)
            
            # Make recommendation
            should_compress = self.should_compress(text_content, provider, model, "auto")
            
            recommendation = {
                'should_compress': should_compress,
                'content_analysis': {
                    'source': content_source,
                    'length_chars': len(text_content),
                    'estimated_tokens': token_count,
                    'is_compressible': is_compressible
                },
                'provider_analysis': {
                    'provider': provider,
                    'model': model,
                    'supports_vision': supports_vision,
                    'context_window': model_context,
                    'utilization': token_count / model_context if model_context > 0 else 0
                },
                'compression_estimate': {
                    'estimated_ratio': estimated_compression_ratio,
                    'estimated_token_savings': int(estimated_savings),
                    'estimated_cost_savings': self._estimate_cost_savings(estimated_savings, provider)
                },
                'recommendation_reason': self._get_recommendation_reason(
                    should_compress, token_count, model_context, supports_vision, is_compressible
                )
            }
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to generate compression recommendation: {e}")
            return {
                'should_compress': False,
                'error': str(e),
                'recommendation_reason': f"Analysis failed: {e}"
            }
    
    def _estimate_compression_ratio(self, text: str, provider: str) -> float:
        """Estimate compression ratio based on content and provider."""
        base_ratio = 3.0  # Default from Glyph research
        
        # Adjust based on content type
        if self._is_code_content(text):
            base_ratio *= 0.8  # Code compresses less well
        elif self._is_prose_content(text):
            base_ratio *= 1.1  # Prose compresses better
        
        # Adjust based on provider OCR quality
        provider_multipliers = {
            'openai': 1.1,      # Excellent OCR
            'anthropic': 1.0,   # Good OCR
            'ollama': 0.9,      # Variable OCR
            'lmstudio': 0.9,    # Variable OCR
            'mlx': 0.8,         # Limited OCR
            'huggingface': 0.8  # Variable OCR
        }
        
        multiplier = provider_multipliers.get(provider, 0.9)
        return base_ratio * multiplier
    
    def _is_code_content(self, text: str) -> bool:
        """Check if content appears to be code."""
        code_indicators = ['def ', 'class ', 'import ', 'function', '{', '}', '#!/', 'var ', 'const ']
        return sum(1 for indicator in code_indicators if indicator in text) > 3
    
    def _is_prose_content(self, text: str) -> bool:
        """Check if content appears to be prose."""
        # Simple heuristic: high ratio of common words
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        word_count = len(text.split())
        common_count = sum(1 for word in text.lower().split() if word in common_words)
        return word_count > 0 and (common_count / word_count) > 0.1
    
    def _estimate_cost_savings(self, token_savings: float, provider: str) -> float:
        """Estimate cost savings from token reduction."""
        # Rough cost estimates per 1K tokens (as of 2024)
        cost_per_1k = {
            'openai': 0.01,     # GPT-4o pricing
            'anthropic': 0.015, # Claude pricing
            'ollama': 0.0,      # Local models
            'lmstudio': 0.0,    # Local models
            'mlx': 0.0,         # Local models
            'huggingface': 0.002 # API pricing
        }
        
        rate = cost_per_1k.get(provider, 0.01)
        return (token_savings / 1000) * rate
    
    def _get_recommendation_reason(
        self, 
        should_compress: bool,
        token_count: int,
        model_context: int,
        supports_vision: bool,
        is_compressible: bool
    ) -> str:
        """Get human-readable recommendation reason."""
        if not should_compress:
            if not supports_vision:
                return "Provider does not support vision processing"
            elif not is_compressible:
                return "Content type not suitable for visual compression"
            elif token_count < self.config.min_token_threshold:
                return f"Content too small ({token_count} tokens < {self.config.min_token_threshold} threshold)"
            else:
                return "Standard processing is sufficient for this content size"
        else:
            if token_count > model_context * 0.8:
                return f"Compression necessary - approaching context limit ({token_count}/{model_context} tokens)"
            elif token_count > 50000:
                return f"Compression beneficial - large content ({token_count} tokens) will benefit from 3-4x reduction"
            else:
                return "Compression recommended based on content analysis"

