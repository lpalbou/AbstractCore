"""
Glyph compression configuration classes.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class RenderingConfig:
    """Configuration for text rendering to images."""
    
    # Font configuration (EXTREME density optimization)
    font_path: str = "Verdana.ttf"
    font_name: Optional[str] = "OCRB"  # Default to OCRB for optimal readability
    font_size: int = 7  # Minimum readable font size
    line_height: int = 8  # Ultra-tight line spacing
    
    # Layout configuration (VLM-optimized defaults)
    dpi: int = 72  # 72 for higher compression, 96 for better quality
    target_width: Optional[int] = None  # Target image width in pixels (default: 1024 for VLMs)
    target_height: Optional[int] = None  # Target image height in pixels (default: 768 for VLMs)
    margin_x: int = 10  # Generous margins for better readability
    margin_y: int = 10  # Generous margins for better readability
    page_width: int = 595  # A4 width in points (used when target dimensions not set)
    page_height: int = 842  # A4 height in points (used when target dimensions not set)
    
    # Optimization settings
    auto_crop_width: bool = True
    auto_crop_last_page: bool = True
    newline_markup: str = '<font color="#FF0000"> \\n </font>'
    
    # Multi-column layout (optimized for readability)
    columns: int = 2  # 2-column layout for optimal balance
    column_gap: int = 10  # Optimal gap between columns
    
    # Text formatting options
    render_format: bool = True  # Enable markdown-like formatting
    
    def copy(self) -> 'RenderingConfig':
        """Create a copy of this configuration."""
        return RenderingConfig(
            font_path=self.font_path,
            font_name=self.font_name,
            font_size=self.font_size,
            line_height=self.line_height,
            dpi=self.dpi,
            target_width=self.target_width,
            target_height=self.target_height,
            margin_x=self.margin_x,
            margin_y=self.margin_y,
            page_width=self.page_width,
            page_height=self.page_height,
            auto_crop_width=self.auto_crop_width,
            auto_crop_last_page=self.auto_crop_last_page,
            newline_markup=self.newline_markup,
            columns=self.columns,
            column_gap=self.column_gap,
            render_format=self.render_format
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'font_path': self.font_path,
            'font_name': self.font_name,
            'font_size': self.font_size,
            'line_height': self.line_height,
            'dpi': self.dpi,
            'target_width': self.target_width,
            'target_height': self.target_height,
            'margin_x': self.margin_x,
            'margin_y': self.margin_y,
            'page_width': self.page_width,
            'page_height': self.page_height,
            'auto_crop_width': self.auto_crop_width,
            'auto_crop_last_page': self.auto_crop_last_page,
            'newline_markup': self.newline_markup,
            'columns': self.columns,
            'column_gap': self.column_gap,
            'render_format': self.render_format
        }


@dataclass
class GlyphConfig:
    """Main configuration for Glyph compression."""
    
    # Core settings
    enabled: bool = True
    global_default: str = "auto"  # auto, always, never
    quality_threshold: float = 0.95
    min_token_threshold: int = 10000  # Minimum tokens to consider compression
    target_compression_ratio: float = 3.0
    
    # Cache settings
    cache_directory: str = field(default_factory=lambda: str(Path.home() / ".abstractcore" / "glyph_cache"))
    cache_size_gb: float = 1.0
    cache_ttl_days: int = 7
    
    # Provider optimization
    provider_optimization: bool = True
    preferred_provider: str = "openai/gpt-4o"
    
    # Rendering configuration
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    
    # App-specific defaults
    app_defaults: Dict[str, str] = field(default_factory=dict)
    
    # Provider-specific profiles
    provider_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Processing settings
    temp_dir: Optional[str] = None
    max_concurrent_compressions: int = 2
    processing_timeout: int = 300  # 5 minutes
    
    @classmethod
    def default(cls) -> 'GlyphConfig':
        """Create default configuration."""
        config = cls()
        
        # Set up default provider profiles based on Glyph research
        config.provider_profiles = {
            "openai": {
                "dpi": 72,
                "font_size": 9,
                "quality_threshold": 0.93,
                "newline_markup": '<font color="#FF0000"> \\n </font>'
            },
            "anthropic": {
                "dpi": 96,
                "font_size": 10,
                "quality_threshold": 0.96,
                "font_path": "Verdana.ttf"
            },
            "ollama": {
                "dpi": 72,
                "font_size": 9,
                "auto_crop_width": True,
                "auto_crop_last_page": True
            },
            "lmstudio": {
                "dpi": 96,
                "font_size": 10,
                "quality_threshold": 0.94
            }
        }
        
        # Set up default app preferences
        config.app_defaults = {
            "summarizer": "always",
            "extractor": "never", 
            "judge": "auto",
            "cli": "auto"
        }
        
        return config
    
    @classmethod
    def from_abstractcore_config(cls) -> 'GlyphConfig':
        """Load Glyph config from AbstractCore's centralized configuration."""
        try:
            # Try to load from AbstractCore config system
            from ..config import get_config_manager
            config_manager = get_config_manager()
            
            # Check if glyph_compression section exists
            if hasattr(config_manager.config, 'glyph_compression'):
                glyph_section = config_manager.config.glyph_compression
                config = cls.default()
                
                # Update with user settings
                config.enabled = getattr(glyph_section, 'enabled', config.enabled)
                config.global_default = getattr(glyph_section, 'global_default', config.global_default)
                config.quality_threshold = getattr(glyph_section, 'quality_threshold', config.quality_threshold)
                config.cache_directory = getattr(glyph_section, 'cache_directory', config.cache_directory)
                config.preferred_provider = getattr(glyph_section, 'preferred_provider', config.preferred_provider)
                
                # Update app defaults
                if hasattr(glyph_section, 'app_defaults'):
                    config.app_defaults.update(glyph_section.app_defaults)
                
                # Update provider profiles
                if hasattr(glyph_section, 'provider_profiles'):
                    for provider, profile in glyph_section.provider_profiles.items():
                        if provider in config.provider_profiles:
                            config.provider_profiles[provider].update(profile)
                        else:
                            config.provider_profiles[provider] = profile
                
                return config
            else:
                # No glyph config section, return defaults
                return cls.default()
                
        except (ImportError, AttributeError):
            # Fallback to default if config system not available
            return cls.default()
    
    def save_to_abstractcore_config(self):
        """Save Glyph config to AbstractCore's centralized configuration."""
        try:
            # Try to save to AbstractCore config system
            from ..config import get_config_manager
            config_manager = get_config_manager()
            
            # Create glyph_compression section
            glyph_config = {
                'enabled': self.enabled,
                'global_default': self.global_default,
                'quality_threshold': self.quality_threshold,
                'cache_directory': self.cache_directory,
                'preferred_provider': self.preferred_provider,
                'app_defaults': self.app_defaults,
                'provider_profiles': self.provider_profiles
            }
            
            # Save configuration
            config_manager.set_glyph_compression(glyph_config)
            config_manager.save()
            
        except (ImportError, AttributeError):
            # Silently fail if config system not available
            # Could implement file-based fallback here if needed
            pass
    
    def get_provider_config(self, provider: str, model: str = None) -> RenderingConfig:
        """Get provider-specific rendering configuration."""
        base_config = self.rendering.copy()
        
        # Apply provider-specific settings
        if provider in self.provider_profiles:
            profile = self.provider_profiles[provider]
            
            # Update rendering config with provider settings
            if 'dpi' in profile:
                base_config.dpi = profile['dpi']
            if 'font_size' in profile:
                base_config.font_size = profile['font_size']
            if 'font_path' in profile:
                base_config.font_path = profile['font_path']
            if 'newline_markup' in profile:
                base_config.newline_markup = profile['newline_markup']
            if 'auto_crop_width' in profile:
                base_config.auto_crop_width = profile['auto_crop_width']
            if 'auto_crop_last_page' in profile:
                base_config.auto_crop_last_page = profile['auto_crop_last_page']
        
        return base_config
    
    def should_compress(self, content_length: int, provider: str, model: str, user_preference: str = None) -> bool:
        """Determine if content should be compressed."""
        # Check user preference first
        preference = user_preference or self.global_default
        
        if preference == "never":
            return False
        elif preference == "always":
            return True
        
        # Auto-decision logic
        if content_length < self.min_token_threshold:
            return False  # Too small to benefit
        
        # Check if provider supports vision
        try:
            from ..media.capabilities import get_model_capabilities
            capabilities = get_model_capabilities(provider, model)
            if not capabilities.get('vision_support', False):
                return False  # Provider doesn't support vision
        except:
            # Conservative approach if capabilities unknown
            return False
        
        return True  # Beneficial for large content with vision support
