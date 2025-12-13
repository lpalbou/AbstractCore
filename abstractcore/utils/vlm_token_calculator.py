"""
VLM Token Calculator - Research-Based Accurate Token Estimation

This module provides state-of-the-art token calculations for images processed by different VLM providers,
integrating with AbstractCore's detection system and model capabilities database.

Key Research Insights Integrated:
- OpenAI GPT-4V: 85 base + 170 tokens per 512x512 tile (with resizing logic)
- Anthropic Claude: ~(width * height) / 750 with 1600 token cap
- Google Gemini: 258 tokens for small images, 258 per 768x768 tile for large
- Qwen-VL models: Variable patch sizes (14px, 16px) with adaptive resolution
- LLaMA Vision: 14px patches with specific resolution tiers
- Local models: Architecture-specific optimizations

References:
- Image Tokenization for Visual Models research
- Glyph Visual Text Compression framework
- OpenAI, Anthropic, Google official documentation
- Recent VLM architecture papers (2024-2025)
"""

import math
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import logging

from PIL import Image

from ..utils.structured_logging import get_logger
from ..architectures.detection import get_model_capabilities, detect_architecture

logger = get_logger(__name__)


class VLMTokenCalculator:
    """
    Research-based VLM token calculator that integrates with AbstractCore's
    model detection and capabilities system for maximum accuracy.
    """
    
    # Provider-specific base configurations (enhanced with research data)
    PROVIDER_CONFIGS = {
        'openai': {
            'base_tokens': 85,
            'tokens_per_tile': 170,
            'tile_size': 512,
            'max_dimension': 2048,
            'short_side_target': 768,  # From research: resized to 768px short side
            'detail_levels': {
                'low': 85,
                'high': 'calculated',
                'auto': 'calculated'
            }
        },
        'anthropic': {
            'base_formula': 'pixel_area',  # (width * height) / 750
            'pixel_divisor': 750,
            'max_dimension': 1568,
            'token_cap': 1600,
            'min_dimension_warning': 200
        },
        'google': {
            'small_image_threshold': 384,  # Both dimensions <= 384
            'small_image_tokens': 258,
            'tile_size': 768,
            'tokens_per_tile': 258
        },
        'ollama': {
            'base_tokens': 256,
            'scaling_factor': 0.5,
            'max_dimension': 1024
        },
        'lmstudio': {
            'base_tokens': 512,
            'scaling_factor': 0.7,
            'max_dimension': 1024
        }
    }
    
    # Model-specific overrides based on research and model capabilities
    MODEL_SPECIFIC_CONFIGS = {
        # Qwen VL models (from research: 28x28 pixel patches for 2.5, 32x32 for 3.0)
        'qwen2.5-vl': {
            'patch_size': 14,
            'pixel_grouping': '28x28',
            'max_image_tokens': 16384,
            'adaptive_resolution': True,
            'resolution_range': (56, 3584)
        },
        'qwen3-vl': {
            'patch_size': 16,
            'pixel_grouping': '32x32',
            'max_image_tokens': 24576,
            'adaptive_resolution': True,
            'resolution_range': (64, 4096)
        },
        # LLaMA Vision models (from research: 14px patches, specific resolutions)
        'llama3.2-vision': {
            'patch_size': 14,
            'supported_resolutions': [(560, 560), (1120, 560), (560, 1120), (1120, 1120)],
            'max_image_tokens': 6400,
            'base_tokens': 256
        },
        # Gemma Vision models (from research: SigLIP encoder, 896x896 fixed)
        'gemma3': {
            'fixed_resolution': (896, 896),
            'vision_encoder': 'SigLIP-400M',
            'tokens_per_image': 256,
            'adaptive_windowing': True
        },
        # GLM models (Glyph research base model)
        'glm-4': {
            'optimized_for_glyph': True,
            'text_image_processing': True,
            'base_tokens': 512
        }
    }
    
    def __init__(self):
        """Initialize the VLM token calculator."""
        self.logger = get_logger(self.__class__.__name__)
    
    def calculate_tokens_for_image(self, 
                                 image_path: Optional[Path] = None,
                                 width: Optional[int] = None, 
                                 height: Optional[int] = None,
                                 provider: str = 'openai',
                                 model: str = '',
                                 detail_level: str = 'auto') -> Dict[str, Any]:
        """
        Calculate accurate token count using model capabilities and research-based formulas.
        
        Args:
            image_path: Path to image file
            width: Image width in pixels
            height: Image height in pixels
            provider: VLM provider
            model: Specific model name
            detail_level: Detail level for applicable models
            
        Returns:
            Dictionary with token count and calculation details
        """
        # Get image dimensions
        if image_path and image_path.exists():
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                self.logger.debug(f"Loaded image dimensions: {width}x{height} from {image_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load image {image_path}: {e}")
                if width is None or height is None:
                    raise ValueError(f"Cannot determine image dimensions: {e}")
        
        if width is None or height is None:
            raise ValueError("Must provide either image_path or both width and height")
        
        # Get model capabilities from AbstractCore's detection system
        model_caps = get_model_capabilities(model) if model else {}
        architecture = detect_architecture(model) if model else provider.lower()
        
        # Determine calculation method based on model capabilities and research
        calculation_result = self._calculate_with_model_awareness(
            width, height, provider, model, architecture, model_caps, detail_level
        )
        
        # Add metadata about the calculation
        calculation_result.update({
            'image_dimensions': f"{width}x{height}",
            'total_pixels': width * height,
            'provider': provider,
            'model': model,
            'architecture': architecture,
            'calculation_timestamp': self._get_timestamp()
        })
        
        return calculation_result
    
    def _calculate_with_model_awareness(self, width: int, height: int, provider: str, 
                                      model: str, architecture: str, model_caps: Dict[str, Any],
                                      detail_level: str) -> Dict[str, Any]:
        """Calculate tokens using model-specific knowledge and research insights."""
        
        # Check for model-specific configurations first
        model_config = self._get_model_specific_config(model, architecture)
        if model_config:
            return self._calculate_with_model_config(width, height, model_config, model_caps)
        
        # Fall back to provider-specific calculations
        provider_lower = provider.lower()
        
        if provider_lower == 'openai':
            return self._calculate_openai_tokens_enhanced(width, height, model, detail_level, model_caps)
        elif provider_lower == 'anthropic':
            return self._calculate_anthropic_tokens_enhanced(width, height, model, model_caps)
        elif provider_lower == 'google':
            return self._calculate_google_tokens(width, height, model, model_caps)
        elif provider_lower in ['ollama', 'lmstudio']:
            return self._calculate_local_tokens_enhanced(width, height, provider_lower, model, model_caps)
        else:
            self.logger.warning(f"Unknown provider '{provider}', using research-based estimation")
            return self._calculate_research_based_fallback(width, height, model_caps)
    
    def _get_model_specific_config(self, model: str, architecture: str) -> Optional[Dict[str, Any]]:
        """Get model-specific configuration based on research data."""
        model_lower = model.lower()
        
        # Check exact model matches first
        for model_key, config in self.MODEL_SPECIFIC_CONFIGS.items():
            if model_key in model_lower:
                return config
        
        # Check architecture-based matches
        if 'qwen' in model_lower and 'vl' in model_lower:
            if '2.5' in model_lower:
                return self.MODEL_SPECIFIC_CONFIGS.get('qwen2.5-vl')
            elif '3' in model_lower:
                return self.MODEL_SPECIFIC_CONFIGS.get('qwen3-vl')
        
        if 'llama' in model_lower and 'vision' in model_lower:
            return self.MODEL_SPECIFIC_CONFIGS.get('llama3.2-vision')
        
        if 'gemma' in model_lower and ('vision' in model_lower or '3' in model_lower):
            return self.MODEL_SPECIFIC_CONFIGS.get('gemma3')
        
        if 'glm' in model_lower:
            return self.MODEL_SPECIFIC_CONFIGS.get('glm-4')
        
        return None
    
    def _calculate_with_model_config(self, width: int, height: int, 
                                   model_config: Dict[str, Any], 
                                   model_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate tokens using model-specific configuration."""
        
        # Handle fixed resolution models (like Gemma3)
        if 'fixed_resolution' in model_config:
            target_width, target_height = model_config['fixed_resolution']
            tokens = model_config.get('tokens_per_image', 256)
            
            return {
                'tokens': tokens,
                'method': 'fixed_resolution',
                'target_resolution': f"{target_width}x{target_height}",
                'vision_encoder': model_config.get('vision_encoder', 'unknown'),
                'adaptive_windowing': model_config.get('adaptive_windowing', False)
            }
        
        # Handle patch-based models (Qwen-VL, LLaMA Vision)
        if 'patch_size' in model_config:
            return self._calculate_patch_based_tokens(width, height, model_config, model_caps)
        
        # Handle supported resolution models (LLaMA Vision)
        if 'supported_resolutions' in model_config:
            return self._calculate_resolution_tier_tokens(width, height, model_config, model_caps)
        
        # Fallback to base tokens
        tokens = model_config.get('base_tokens', 512)
        return {
            'tokens': tokens,
            'method': 'model_specific_base',
            'config_used': model_config
        }
    
    def _calculate_patch_based_tokens(self, width: int, height: int, 
                                    model_config: Dict[str, Any],
                                    model_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate tokens for patch-based models like Qwen-VL."""
        patch_size = model_config['patch_size']
        max_tokens = model_config.get('max_image_tokens', 16384)
        
        # Handle adaptive resolution
        if model_config.get('adaptive_resolution'):
            min_res, max_res = model_config.get('resolution_range', (56, 3584))
            # Resize if outside supported range
            if max(width, height) > max_res:
                scale = max_res / max(width, height)
                width = int(width * scale)
                height = int(height * scale)
            elif min(width, height) < min_res:
                scale = min_res / min(width, height)
                width = int(width * scale)
                height = int(height * scale)
        
        # Calculate patches
        patches_width = math.ceil(width / patch_size)
        patches_height = math.ceil(height / patch_size)
        total_patches = patches_width * patches_height
        
        # Apply token limit
        tokens = min(total_patches, max_tokens)
        
        return {
            'tokens': tokens,
            'method': 'patch_based',
            'patch_size': patch_size,
            'patches': f"{patches_width}x{patches_height}",
            'total_patches': total_patches,
            'max_tokens': max_tokens,
            'pixel_grouping': model_config.get('pixel_grouping', f"{patch_size}x{patch_size}"),
            'resized_to': f"{width}x{height}" if model_config.get('adaptive_resolution') else None
        }
    
    def _calculate_resolution_tier_tokens(self, width: int, height: int,
                                        model_config: Dict[str, Any],
                                        model_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate tokens for models with specific supported resolutions."""
        supported_resolutions = model_config['supported_resolutions']
        max_tokens = model_config.get('max_image_tokens', 6400)
        base_tokens = model_config.get('base_tokens', 256)
        
        # Find the best matching resolution
        best_resolution = None
        min_scale_factor = float('inf')
        
        for res_width, res_height in supported_resolutions:
            scale_w = res_width / width
            scale_h = res_height / height
            scale_factor = max(scale_w, scale_h)  # Scale to fit
            
            if scale_factor < min_scale_factor:
                min_scale_factor = scale_factor
                best_resolution = (res_width, res_height)
        
        # Calculate tokens based on resolution tier
        if best_resolution:
            res_area = best_resolution[0] * best_resolution[1]
            base_area = 560 * 560  # Base resolution area
            tokens = int(base_tokens * (res_area / base_area))
            tokens = min(tokens, max_tokens)
        else:
            tokens = base_tokens
        
        return {
            'tokens': tokens,
            'method': 'resolution_tier',
            'best_resolution': f"{best_resolution[0]}x{best_resolution[1]}" if best_resolution else None,
            'scale_factor': min_scale_factor,
            'max_tokens': max_tokens,
            'supported_resolutions': supported_resolutions
        }
    
    def _calculate_openai_tokens_enhanced(self, width: int, height: int, model: str, 
                                        detail_level: str, model_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced OpenAI token calculation with research-based improvements."""
        config = self.PROVIDER_CONFIGS['openai']
        
        if detail_level == 'low':
            return {
                'tokens': config['detail_levels']['low'],
                'method': 'openai_low_detail',
                'detail_level': 'low'
            }
        
        # Step 1: Resize to fit 2048x2048 square (preserving aspect ratio)
        max_dim = config['max_dimension']
        if width > max_dim or height > max_dim:
            scale = min(max_dim / width, max_dim / height)
            width = int(width * scale)
            height = int(height * scale)
        
        # Step 2: Resize so shortest side is 768px (from research)
        short_side_target = config['short_side_target']
        if min(width, height) != short_side_target:
            scale = short_side_target / min(width, height)
            width = int(width * scale)
            height = int(height * scale)
        
        # Step 3: Calculate tiles
        tile_size = config['tile_size']
        tiles_width = math.ceil(width / tile_size)
        tiles_height = math.ceil(height / tile_size)
        total_tiles = tiles_width * tiles_height
        
        # Step 4: Apply formula
        base_tokens = config['base_tokens']
        tile_tokens = config['tokens_per_tile']
        total_tokens = base_tokens + (total_tiles * tile_tokens)
        
        return {
            'tokens': total_tokens,
            'method': 'openai_tile_based',
            'detail_level': detail_level,
            'resized_to': f"{width}x{height}",
            'tiles': f"{tiles_width}x{tiles_height}",
            'total_tiles': total_tiles,
            'base_tokens': base_tokens,
            'tile_tokens': tile_tokens
        }
    
    def _calculate_anthropic_tokens_enhanced(self, width: int, height: int, model: str,
                                           model_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Anthropic token calculation based on research formula."""
        config = self.PROVIDER_CONFIGS['anthropic']
        
        # Resize if exceeds limits
        max_dim = config['max_dimension']
        original_size = f"{width}x{height}"
        
        if width > max_dim or height > max_dim:
            scale = min(max_dim / width, max_dim / height)
            width = int(width * scale)
            height = int(height * scale)
        
        # Apply Anthropic's formula: (width * height) / 750
        pixel_area = width * height
        calculated_tokens = pixel_area / config['pixel_divisor']
        
        # Apply token cap
        tokens = min(int(calculated_tokens), config['token_cap'])
        
        # Check for small image warning
        min_dim_warning = None
        if min(width, height) < config['min_dimension_warning']:
            min_dim_warning = f"Image dimension below {config['min_dimension_warning']}px may degrade performance"
        
        return {
            'tokens': tokens,
            'method': 'anthropic_pixel_area',
            'formula': f"({width} * {height}) / {config['pixel_divisor']}",
            'calculated_tokens': calculated_tokens,
            'token_cap': config['token_cap'],
            'resized_from': original_size if f"{width}x{height}" != original_size else None,
            'warning': min_dim_warning
        }
    
    def _calculate_google_tokens(self, width: int, height: int, model: str,
                               model_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate tokens for Google Gemini models using hybrid approach."""
        config = self.PROVIDER_CONFIGS['google']
        
        # Check if it's a small image
        threshold = config['small_image_threshold']
        if width <= threshold and height <= threshold:
            return {
                'tokens': config['small_image_tokens'],
                'method': 'google_small_image',
                'threshold': f"{threshold}x{threshold}",
                'classification': 'small'
            }
        
        # Large image: calculate tiles
        tile_size = config['tile_size']
        tiles_width = math.ceil(width / tile_size)
        tiles_height = math.ceil(height / tile_size)
        total_tiles = tiles_width * tiles_height
        
        tokens = total_tiles * config['tokens_per_tile']
        
        return {
            'tokens': tokens,
            'method': 'google_tiled',
            'classification': 'large',
            'tile_size': f"{tile_size}x{tile_size}",
            'tiles': f"{tiles_width}x{tiles_height}",
            'total_tiles': total_tiles,
            'tokens_per_tile': config['tokens_per_tile']
        }
    
    def _calculate_local_tokens_enhanced(self, width: int, height: int, provider: str,
                                       model: str, model_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced local model token calculation with model capabilities integration."""
        config = self.PROVIDER_CONFIGS[provider]
        base_tokens = config['base_tokens']
        
        # Get model-specific information from capabilities
        vision_support = model_caps.get('vision_support', False)
        max_image_tokens = model_caps.get('max_image_tokens', base_tokens)
        image_patch_size = model_caps.get('image_patch_size', 16)
        
        # Use patch-based calculation if patch size is available
        if image_patch_size and vision_support:
            patches_width = math.ceil(width / image_patch_size)
            patches_height = math.ceil(height / image_patch_size)
            total_patches = patches_width * patches_height
            
            # Scale by efficiency factor
            tokens = int(total_patches * config['scaling_factor'])
            tokens = min(tokens, max_image_tokens)
            
            return {
                'tokens': tokens,
                'method': f'{provider}_patch_based',
                'patch_size': image_patch_size,
                'patches': f"{patches_width}x{patches_height}",
                'scaling_factor': config['scaling_factor'],
                'max_tokens': max_image_tokens,
                'vision_support': vision_support
            }
        
        # Fallback to area-based calculation
        standard_pixels = 512 * 512
        actual_pixels = width * height
        scaling_factor = math.sqrt(actual_pixels / standard_pixels)
        
        tokens = int(base_tokens * scaling_factor * config['scaling_factor'])
        
        return {
            'tokens': tokens,
            'method': f'{provider}_area_based',
            'base_tokens': base_tokens,
            'scaling_factor': config['scaling_factor'],
            'pixel_scaling': scaling_factor
        }
    
    def _calculate_research_based_fallback(self, width: int, height: int,
                                         model_caps: Dict[str, Any]) -> Dict[str, Any]:
        """Research-based fallback calculation for unknown models."""
        # Use Vision Transformer patch-based approach as fallback
        patch_size = model_caps.get('image_patch_size', 16)  # Default ViT patch size
        
        patches_width = math.ceil(width / patch_size)
        patches_height = math.ceil(height / patch_size)
        total_patches = patches_width * patches_height
        
        # Conservative token estimate
        tokens = min(total_patches, 2048)  # Cap at reasonable limit
        
        return {
            'tokens': tokens,
            'method': 'research_based_fallback',
            'patch_size': patch_size,
            'patches': f"{patches_width}x{patches_height}",
            'note': 'Using Vision Transformer patch-based estimation'
        }
    
    def calculate_tokens_for_images(self, 
                                  image_paths: List[Path],
                                  provider: str = 'openai',
                                  model: str = '',
                                  detail_level: str = 'auto') -> Dict[str, Any]:
        """Calculate tokens for multiple images with detailed breakdown."""
        results = {
            'total_tokens': 0,
            'image_count': len(image_paths),
            'per_image_results': [],
            'average_tokens_per_image': 0,
            'provider': provider,
            'model': model,
            'calculation_summary': {}
        }
        
        method_counts = {}
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.calculate_tokens_for_image(
                    image_path=image_path,
                    provider=provider,
                    model=model,
                    detail_level=detail_level
                )
                
                results['per_image_results'].append(result)
                results['total_tokens'] += result['tokens']
                
                # Track calculation methods
                method = result.get('method', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
                
            except Exception as e:
                self.logger.error(f"Failed to calculate tokens for {image_path}: {e}")
                fallback_result = {
                    'tokens': 512,  # Conservative fallback
                    'method': 'error_fallback',
                    'error': str(e),
                    'image_path': str(image_path)
                }
                results['per_image_results'].append(fallback_result)
                results['total_tokens'] += 512
        
        if results['image_count'] > 0:
            results['average_tokens_per_image'] = results['total_tokens'] / results['image_count']
        
        results['calculation_summary'] = {
            'methods_used': method_counts,
            'primary_method': max(method_counts.items(), key=lambda x: x[1])[0] if method_counts else 'none'
        }
        
        return results
    
    def get_compression_ratio(self, 
                            original_text_tokens: int,
                            image_paths: List[Path],
                            provider: str = 'openai',
                            model: str = '') -> Dict[str, Any]:
        """Calculate accurate compression ratio with enhanced analysis."""
        image_analysis = self.calculate_tokens_for_images(
            image_paths=image_paths,
            provider=provider,
            model=model
        )
        
        compressed_tokens = image_analysis['total_tokens']
        compression_ratio = original_text_tokens / compressed_tokens if compressed_tokens > 0 else 0
        
        return {
            'original_tokens': original_text_tokens,
            'compressed_tokens': compressed_tokens,
            'compression_ratio': compression_ratio,
            'images_created': len(image_paths),
            'average_tokens_per_image': image_analysis['average_tokens_per_image'],
            'provider': provider,
            'model': model,
            'calculation_methods': image_analysis['calculation_summary'],
            'per_image_breakdown': image_analysis['per_image_results'],
            'token_savings': original_text_tokens - compressed_tokens,
            'efficiency_analysis': self._analyze_efficiency(compression_ratio, provider, model)
        }
    
    def _analyze_efficiency(self, ratio: float, provider: str, model: str) -> Dict[str, Any]:
        """Analyze compression efficiency and provide insights."""
        if ratio > 10:
            efficiency = "excellent"
            insight = "Exceptional compression achieved, ideal for long-context processing"
        elif ratio > 4:
            efficiency = "very_good"
            insight = "Strong compression ratio, significant token savings"
        elif ratio > 2:
            efficiency = "good"
            insight = "Moderate compression, suitable for most use cases"
        elif ratio > 1:
            efficiency = "marginal"
            insight = "Limited compression benefit, consider alternative approaches"
        else:
            efficiency = "poor"
            insight = "No compression benefit, text processing may be more efficient"
        
        return {
            'efficiency_rating': efficiency,
            'insight': insight,
            'compression_ratio': ratio,
            'recommended_use': ratio > 1.5
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for calculation metadata."""
        from datetime import datetime
        return datetime.now().isoformat()


# Convenience functions for backward compatibility
def calculate_image_tokens(image_path: Path, provider: str = 'openai', model: str = '') -> int:
    """Calculate tokens for a single image."""
    calculator = VLMTokenCalculator()
    result = calculator.calculate_tokens_for_image(image_path=image_path, provider=provider, model=model)
    return result['tokens']


def calculate_glyph_compression_ratio(original_tokens: int, 
                                    image_paths: List[Path], 
                                    provider: str = 'openai',
                                    model: str = '') -> Dict[str, Any]:
    """Calculate accurate Glyph compression ratio."""
    calculator = VLMTokenCalculator()
    return calculator.get_compression_ratio(original_tokens, image_paths, provider, model)