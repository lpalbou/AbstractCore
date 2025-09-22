"""
JSON-based architecture detection system.

Loads architecture formats and model capabilities from JSON assets
to determine how to communicate with different models.
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Cache for loaded JSON data
_architecture_formats: Optional[Dict[str, Any]] = None
_model_capabilities: Optional[Dict[str, Any]] = None


def _load_json_assets():
    """Load architecture formats and model capabilities from JSON files."""
    global _architecture_formats, _model_capabilities

    if _architecture_formats is not None and _model_capabilities is not None:
        return

    # Get the assets directory path
    current_dir = Path(__file__).parent.parent
    assets_dir = current_dir / "assets"

    # Load architecture formats
    arch_file = assets_dir / "architecture_formats.json"
    if arch_file.exists():
        with open(arch_file, 'r', encoding='utf-8') as f:
            _architecture_formats = json.load(f)
    else:
        logger.warning(f"Architecture formats file not found: {arch_file}")
        _architecture_formats = {"architectures": {}}

    # Load model capabilities
    cap_file = assets_dir / "model_capabilities.json"
    if cap_file.exists():
        with open(cap_file, 'r', encoding='utf-8') as f:
            _model_capabilities = json.load(f)
    else:
        logger.warning(f"Model capabilities file not found: {cap_file}")
        _model_capabilities = {"models": {}, "default_capabilities": {}}


def detect_architecture(model_name: str) -> str:
    """
    Detect model architecture from model name using JSON patterns.

    Args:
        model_name: Name of the model

    Returns:
        Architecture name (e.g., 'qwen', 'llama', 'openai')
    """
    _load_json_assets()

    if not _architecture_formats or "architectures" not in _architecture_formats:
        return "generic"

    model_lower = model_name.lower()

    # Check each architecture's patterns
    for arch_name, arch_config in _architecture_formats["architectures"].items():
        patterns = arch_config.get("patterns", [])
        for pattern in patterns:
            if pattern.lower() in model_lower:
                logger.debug(f"Detected architecture '{arch_name}' for model '{model_name}' (pattern: '{pattern}')")
                return arch_name

    # Fallback to generic
    logger.debug(f"No specific architecture detected for '{model_name}', using generic")
    return "generic"


def get_architecture_format(architecture: str) -> Dict[str, Any]:
    """
    Get architecture format configuration.

    Args:
        architecture: Architecture name

    Returns:
        Architecture format configuration
    """
    _load_json_assets()

    if not _architecture_formats or "architectures" not in _architecture_formats:
        return {}

    return _architecture_formats["architectures"].get(architecture, {})


def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get model capabilities from JSON configuration.

    Args:
        model_name: Full model name

    Returns:
        Model capabilities dictionary
    """
    _load_json_assets()

    if not _model_capabilities:
        return {}

    models = _model_capabilities.get("models", {})

    # First try exact match
    if model_name in models:
        capabilities = models[model_name].copy()
        # Add architecture if not present
        if "architecture" not in capabilities:
            capabilities["architecture"] = detect_architecture(model_name)
        return capabilities

    # Try partial matches for common model naming patterns
    model_lower = model_name.lower()
    for model_key, capabilities in models.items():
        if model_key.lower() in model_lower or model_lower in model_key.lower():
            result = capabilities.copy()
            if "architecture" not in result:
                result["architecture"] = detect_architecture(model_name)
            logger.debug(f"Using capabilities from '{model_key}' for '{model_name}'")
            return result

    # Fallback to default capabilities based on architecture
    architecture = detect_architecture(model_name)
    default_caps = _model_capabilities.get("default_capabilities", {}).copy()
    default_caps["architecture"] = architecture

    # Enhance defaults based on architecture
    arch_format = get_architecture_format(architecture)
    if arch_format.get("tool_format") == "native":
        default_caps["tool_support"] = "native"
    elif arch_format.get("tool_format") in ["special_token", "json", "xml", "pythonic"]:
        default_caps["tool_support"] = "prompted"
    else:
        default_caps["tool_support"] = "none"

    logger.debug(f"Using default capabilities for '{model_name}' (architecture: {architecture})")
    return default_caps


def format_messages(messages: List[Dict[str, str]], architecture: str) -> str:
    """
    Format messages according to architecture specifications.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        architecture: Architecture name

    Returns:
        Formatted message string
    """
    arch_format = get_architecture_format(architecture)
    if not arch_format:
        # Generic fallback
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    formatted_parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            prefix = arch_format.get("system_prefix", "")
            suffix = arch_format.get("system_suffix", "")
        elif role == "user":
            prefix = arch_format.get("user_prefix", "")
            suffix = arch_format.get("user_suffix", "")
        elif role == "assistant":
            prefix = arch_format.get("assistant_prefix", "")
            suffix = arch_format.get("assistant_suffix", "")
        else:
            prefix = suffix = ""

        formatted_parts.append(f"{prefix}{content}{suffix}")

    return "".join(formatted_parts)


# Convenience functions
def supports_tools(model_name: str) -> bool:
    """Check if model supports tools."""
    capabilities = get_model_capabilities(model_name)
    return capabilities.get("tool_support", "none") != "none"


def supports_vision(model_name: str) -> bool:
    """Check if model supports vision."""
    capabilities = get_model_capabilities(model_name)
    return capabilities.get("vision_support", False)


def supports_audio(model_name: str) -> bool:
    """Check if model supports audio."""
    capabilities = get_model_capabilities(model_name)
    return capabilities.get("audio_support", False)


def supports_embeddings(model_name: str) -> bool:
    """Check if model supports embeddings."""
    capabilities = get_model_capabilities(model_name)
    return capabilities.get("embedding_support", False)


def get_context_limits(model_name: str) -> Dict[str, int]:
    """Get context and output token limits."""
    capabilities = get_model_capabilities(model_name)
    return {
        "context_length": capabilities.get("context_length", 4096),
        "max_output_tokens": capabilities.get("max_output_tokens", 2048)
    }


def is_instruct_model(model_name: str) -> bool:
    """Check if model is instruction-tuned."""
    model_lower = model_name.lower()
    instruct_indicators = ["instruct", "chat", "assistant", "turbo"]
    return any(indicator in model_lower for indicator in instruct_indicators)


def detect_model_type(model_name: str) -> str:
    """
    Detect high-level model type.

    Returns:
        Model type like 'chat', 'instruct', 'base', 'code', 'vision'
    """
    model_lower = model_name.lower()

    if any(x in model_lower for x in ["chat", "turbo", "assistant"]):
        return "chat"
    elif "instruct" in model_lower:
        return "instruct"
    elif any(x in model_lower for x in ["code", "coder", "starcoder", "codegen"]):
        return "code"
    elif any(x in model_lower for x in ["vision", "vl", "multimodal"]):
        return "vision"
    else:
        return "base"