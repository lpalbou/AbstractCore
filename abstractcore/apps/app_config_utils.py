"""
Shared configuration utilities for AbstractCore apps.
"""

def get_app_defaults(app_name: str) -> tuple[str, str]:
    """Get default provider and model for an app."""
    try:
        from ..config import get_config_manager
        config_manager = get_config_manager()
        return config_manager.get_app_default(app_name)
    except (ImportError, Exception):
        # Fallback to hardcoded defaults if config unavailable
        hardcoded_defaults = {
            'summarizer': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'extractor': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'judge': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
            'cli': ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'),
        }
        return hardcoded_defaults.get(app_name, ('huggingface', 'unsloth/Qwen3-4B-Instruct-2507-GGUF'))