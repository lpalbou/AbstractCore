"""
JSON-based architecture detection system.

Loads architecture formats and model capabilities from JSON assets
to determine how to communicate with different models.
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)

# Cache for loaded JSON data
_architecture_formats: Optional[Dict[str, Any]] = None
_model_capabilities: Optional[Dict[str, Any]] = None

# Cache for resolved model names and architectures to reduce redundant logging
_resolved_aliases_cache: Dict[str, str] = {}
_detected_architectures_cache: Dict[str, str] = {}
# Cache to avoid repeating default-capabilities warnings for the same unknown model.
_default_capabilities_warning_cache: set[str] = set()


# Some callers pass provider/model as a single string (e.g. "lmstudio/qwen/qwen3-next-80b").
# For capability lookup we want the underlying model id, not the provider prefix.
_KNOWN_PROVIDER_PREFIXES = {
    "anthropic",
    "azure",
    "bedrock",
    "fireworks",
    "gemini",
    "google",
    "groq",
    "huggingface",
    "lmstudio",
    "local",
    "mlx",
    "nvidia",
    "ollama",
    "openai",
    "openai-compatible",
    "together",
    "vllm",
}


def _strip_provider_prefix(model_name: str) -> str:
    s = str(model_name or "").strip()
    if not s or "/" not in s:
        return s
    head, rest = s.split("/", 1)
    if head.strip().lower() in _KNOWN_PROVIDER_PREFIXES and rest.strip():
        return rest.strip()
    return s


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
    # Check cache first to avoid redundant logging
    if model_name in _detected_architectures_cache:
        return _detected_architectures_cache[model_name]
    
    _load_json_assets()

    if not _architecture_formats or "architectures" not in _architecture_formats:
        _detected_architectures_cache[model_name] = "generic"
        return "generic"

    # Normalize model names for better pattern matching:
    # - HuggingFace cache names use `--` as `/` separators (models--org--name).
    # - Claude versions sometimes appear as `claude-3-5-sonnet` (normalize to `claude-3.5-sonnet`).
    model_lower = model_name.lower().replace("--", "/")
    import re
    model_lower = re.sub(r'(claude-\d+)-(\d+)(?=-|$)', r'\1.\2', model_lower)

    # Choose the most specific matching architecture.
    # Many architectures include broad patterns (e.g. "gpt") that can accidentally
    # match more specific models (e.g. "gpt-oss"). We resolve this by selecting the
    # longest matching pattern across all architectures.
    best_arch = "generic"
    best_pattern = ""

    for arch_name, arch_config in _architecture_formats["architectures"].items():
        patterns = arch_config.get("patterns", [])
        for pattern in patterns:
            pat = str(pattern).lower()
            if not pat:
                continue
            if pat in model_lower and len(pat) > len(best_pattern):
                best_arch = arch_name
                best_pattern = pat

    if best_arch != "generic":
        logger.debug(
            f"Detected architecture '{best_arch}' for model '{model_name}' (pattern: '{best_pattern}')"
        )
        _detected_architectures_cache[model_name] = best_arch
        return best_arch

    # Fallback to generic
    logger.debug(f"No specific architecture detected for '{model_name}', using generic")
    _detected_architectures_cache[model_name] = "generic"
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


def resolve_model_alias(model_name: str, models: Dict[str, Any]) -> str:
    """
    Resolve a model name to its canonical name by checking aliases.

    Automatically converts "--" to "/" for HuggingFace cache format compatibility.
    Normalizes Claude version numbers (e.g., "claude-3-5-sonnet" -> "claude-3.5-sonnet").

    Args:
        model_name: Model name that might be an alias
        models: Models dictionary from capabilities JSON

    Returns:
        Canonical model name
    """
    # Check cache first to avoid redundant logging
    if model_name in _resolved_aliases_cache:
        return _resolved_aliases_cache[model_name]

    # First check if it's already a canonical name
    if model_name in models:
        _resolved_aliases_cache[model_name] = model_name
        return model_name

    # Normalize model name
    normalized_model_name = model_name

    # Convert "--" to "/" for HuggingFace cache format compatibility
    normalized_model_name = normalized_model_name.replace("--", "/")

    # Normalize Claude version numbers: convert "-X-Y-" to "-X.Y-" or "-X-Y" to "-X.Y"
    # Examples:
    #   "claude-3-5-sonnet" -> "claude-3.5-sonnet"
    #   "claude-4-1-opus" -> "claude-4.1-opus"
    #   "claude-3-5-sonnet-20241022" -> "claude-3.5-sonnet-20241022"
    import re
    normalized_model_name = re.sub(r'(claude-\d+)-(\d+)(?=-|$)', r'\1.\2', normalized_model_name)

    if normalized_model_name != model_name:
        logger.debug(f"Normalized model name '{model_name}' to '{normalized_model_name}'")

    # Also support "provider/model" strings by stripping known provider prefixes.
    stripped_model_name = _strip_provider_prefix(model_name)
    stripped_normalized_name = _strip_provider_prefix(normalized_model_name)

    def _tail(name: str) -> str:
        s = str(name or "").strip()
        if not s or "/" not in s:
            return s
        return s.split("/")[-1].strip()

    def _candidates(*names: str) -> List[str]:
        out: List[str] = []
        for n in names:
            s = str(n or "").strip()
            if not s:
                continue
            out.append(s)
            t = _tail(s)
            if t and t != s:
                out.append(t)
        # Deduplicate while preserving order
        uniq: List[str] = []
        seen: set[str] = set()
        for s in out:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        return uniq

    # Check if any normalized/stripped name is a canonical name.
    for candidate in _candidates(normalized_model_name, stripped_normalized_name, stripped_model_name):
        if candidate in models:
            _resolved_aliases_cache[model_name] = candidate
            return candidate

    # Check if it's an alias of any model (try both original and normalized).
    # Some JSON entries intentionally share aliases (e.g. base + variant). Prefer the
    # most specific canonical model name deterministically.
    alias_matches: List[str] = []
    for canonical_name, model_info in models.items():
        aliases = model_info.get("aliases", [])
        if not isinstance(aliases, list) or not aliases:
            continue
        candidates = _candidates(model_name, normalized_model_name, stripped_model_name, stripped_normalized_name)
        alias_set = {str(a).strip().lower() for a in aliases if isinstance(a, str) and str(a).strip()}
        cand_set = {str(c).strip().lower() for c in candidates if isinstance(c, str) and str(c).strip()}
        if alias_set and cand_set and alias_set.intersection(cand_set):
            alias_matches.append(canonical_name)

    if alias_matches:
        best = max(alias_matches, key=lambda n: (len(str(n)), str(n)))
        logger.debug(f"Resolved alias '{model_name}' to canonical name '{best}'")
        _resolved_aliases_cache[model_name] = best
        return best

    # Return normalized name if no alias found
    fallback = stripped_normalized_name or normalized_model_name
    fallback_tail = _tail(fallback)
    if fallback_tail:
        fallback = fallback_tail
    _resolved_aliases_cache[model_name] = fallback
    return fallback


def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get model capabilities from JSON configuration with alias support.

    Args:
        model_name: Full model name (can be an alias)

    Returns:
        Model capabilities dictionary
    """
    _load_json_assets()

    if not _model_capabilities:
        return {}

    models = _model_capabilities.get("models", {})

    # Step 1: Resolve aliases to canonical names
    canonical_name = resolve_model_alias(model_name, models)

    # Step 2: Try exact match with canonical name
    if canonical_name in models:
        capabilities = models[canonical_name].copy()
        # Remove alias-specific fields from capabilities
        capabilities.pop("canonical_name", None)
        capabilities.pop("aliases", None)
        # Add architecture if not present
        if "architecture" not in capabilities:
            capabilities["architecture"] = detect_architecture(canonical_name)
        return capabilities

    # Step 3: Try partial matches for common model naming patterns
    # Use canonical_name (which has been normalized) for better matching
    canonical_lower = canonical_name.lower()
    candidates_name_in_key: List[tuple[int, int, str]] = []
    candidates_key_in_name: List[tuple[int, str]] = []
    for model_key in models.keys():
        if not isinstance(model_key, str) or not model_key.strip():
            continue
        key_lower = model_key.lower()

        # Prefer a close "superstring" match where the canonical name is missing a suffix.
        # Example: "qwen3-next-80b" -> "qwen3-next-80b-a3b"
        if canonical_lower and canonical_lower in key_lower:
            extra = max(0, len(key_lower) - len(canonical_lower))
            candidates_name_in_key.append((extra, len(key_lower), model_key))
            continue

        # Otherwise, prefer the most specific substring match (e.g. provider/model prefixes).
        if key_lower in canonical_lower:
            candidates_key_in_name.append((len(key_lower), model_key))

    best_key: Optional[str] = None
    best_mode: Optional[str] = None
    if candidates_name_in_key:
        candidates_name_in_key.sort(key=lambda x: (x[0], -x[1]))
        best_key = candidates_name_in_key[0][2]
        best_mode = "name_in_key"
    elif candidates_key_in_name:
        best_key = max(candidates_key_in_name, key=lambda x: x[0])[1]
        best_mode = "key_in_name"

    if best_key is not None:
        capabilities = models.get(best_key)
        if isinstance(capabilities, dict):
            result = capabilities.copy()
            # Remove alias-specific fields
            result.pop("canonical_name", None)
            result.pop("aliases", None)
            if "architecture" not in result:
                result["architecture"] = detect_architecture(model_name)
            logger.debug(f"Using capabilities from '{best_key}' for '{model_name}' (partial match: {best_mode})")
            return result

    # Step 4: Fallback to default capabilities based on architecture
    architecture = detect_architecture(model_name)
    default_caps = _model_capabilities.get("default_capabilities", {}).copy()
    default_caps["architecture"] = architecture

    # Enhance defaults based on architecture.
    #
    # NOTE: `architecture_formats.json.tool_format` describes the *prompted transcript syntax*
    # for tool calls (e.g. XML-wrapped, <|tool_call|> blocks, etc). Some architectures/models
    # also support *native tool APIs* (provider-level `tools` payloads) even when their prompted
    # transcript format is non-native. For those cases, architectures can set an explicit
    # `default_tool_support` to avoid relying on tool_format heuristics.
    arch_format = get_architecture_format(architecture)

    explicit_support = str(arch_format.get("default_tool_support") or "").strip().lower()
    if explicit_support in {"native", "prompted", "none"}:
        default_caps["tool_support"] = explicit_support
    elif arch_format.get("tool_format") == "native":
        default_caps["tool_support"] = "native"
    elif arch_format.get("tool_format") in ["special_token", "json", "xml", "pythonic", "glm_xml"]:
        default_caps["tool_support"] = "prompted"
    else:
        default_caps["tool_support"] = "none"

    # Propagate architecture-level output wrappers into default capabilities.
    wrappers = arch_format.get("output_wrappers")
    if isinstance(wrappers, dict) and wrappers:
        default_caps["output_wrappers"] = dict(wrappers)

    logger.debug(f"Using default capabilities for '{model_name}' (architecture: {architecture})")

    # Emit a one-time warning for unknown models to keep model_capabilities.json up to date.
    try:
        raw_name = str(model_name).strip()
    except Exception:
        raw_name = ""

    if raw_name and raw_name not in _default_capabilities_warning_cache:
        _default_capabilities_warning_cache.add(raw_name)
        logger.warning(
            "Model not found in model_capabilities.json; falling back to architecture defaults",
            model_name=raw_name,
            detected_architecture=architecture,
            default_tool_support=default_caps.get("tool_support"),
            next_steps=(
                "Add this model (or an alias) to abstractcore/abstractcore/assets/model_capabilities.json "
                "or email contact@abstractcore.ai with the exact model id and provider."
            ),
        )
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
        "max_tokens": capabilities.get("max_tokens", 16384),  # 16K total context window
        "max_output_tokens": capabilities.get("max_output_tokens", 4096)  # 4K output tokens
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


def get_vision_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get vision-specific capabilities for a model with fallback to generic vision model.
    
    Args:
        model_name: Model name to get vision capabilities for
        
    Returns:
        Dictionary with vision capabilities, using generic fallback if model not found
    """
    from ..utils.structured_logging import get_logger
    logger = get_logger(__name__)
    
    # Get model capabilities
    capabilities = get_model_capabilities(model_name)
    
    # Check if model has vision support
    if not capabilities.get('vision_support', False):
        logger.warning(f"Model '{model_name}' does not have vision support")
        return {}
    
    # Extract vision-specific fields
    vision_fields = [
        'image_resolutions', 'max_image_resolution', 'image_patch_size', 
        'max_image_tokens', 'image_tokenization_method', 'adaptive_resolution',
        'vision_encoder', 'pixel_grouping', 'supported_resolutions',
        'base_tokens_per_resolution', 'fixed_resolution', 'tokens_per_tile',
        'tile_size', 'base_image_tokens', 'pixel_divisor', 'token_cap'
    ]
    
    vision_capabilities = {}
    for field in vision_fields:
        if field in capabilities:
            vision_capabilities[field] = capabilities[field]
    
    # If we have minimal vision capabilities, use generic fallback
    if not vision_capabilities or len(vision_capabilities) < 3:
        logger.warning(
            f"Model '{model_name}' has limited vision metadata, using generic vision model fallback",
            model=model_name,
            found_fields=list(vision_capabilities.keys())
        )
        
        # Get generic vision model capabilities
        _load_json_assets()
        if _model_capabilities and "generic_vision_model" in _model_capabilities:
            generic_caps = _model_capabilities["generic_vision_model"]
            for field in vision_fields:
                if field in generic_caps:
                    vision_capabilities[field] = generic_caps[field]
    
    return vision_capabilities


def get_glyph_compression_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get capabilities relevant for Glyph compression with intelligent fallbacks.
    
    Args:
        model_name: Model name to get Glyph capabilities for
        
    Returns:
        Dictionary with Glyph-relevant capabilities and recommendations
    """
    from ..utils.structured_logging import get_logger
    logger = get_logger(__name__)
    
    capabilities = get_model_capabilities(model_name)
    
    # Check if model supports vision (required for Glyph)
    if not capabilities.get('vision_support', False):
        logger.error(
            f"Model '{model_name}' does not support vision, cannot use Glyph compression",
            model=model_name
        )
        return {
            'glyph_compatible': False,
            'reason': 'no_vision_support'
        }
    
    # Get vision capabilities
    vision_caps = get_vision_capabilities(model_name)
    
    # Determine Glyph compatibility and optimal settings
    glyph_caps = {
        'glyph_compatible': True,
        'model_name': model_name,
        'vision_support': True
    }
    
    # Add vision-specific fields for token calculation
    glyph_caps.update(vision_caps)
    
    # Determine optimal compression settings based on model capabilities
    max_image_tokens = vision_caps.get('max_image_tokens', 2048)
    image_patch_size = vision_caps.get('image_patch_size', 16)
    
    # Recommend compression parameters
    if max_image_tokens >= 16000:
        glyph_caps['recommended_pages_per_image'] = 2
        glyph_caps['recommended_dpi'] = 150
    elif max_image_tokens >= 8000:
        glyph_caps['recommended_pages_per_image'] = 1
        glyph_caps['recommended_dpi'] = 120
    else:
        glyph_caps['recommended_pages_per_image'] = 1
        glyph_caps['recommended_dpi'] = 100
    
    # Check for Glyph-optimized models
    if capabilities.get('optimized_for_glyph', False):
        glyph_caps['glyph_optimized'] = True
        logger.info(f"Model '{model_name}' is optimized for Glyph compression")
    
    return glyph_caps


def check_vision_model_compatibility(model_name: str, provider: str = None) -> Dict[str, Any]:
    """
    Comprehensive check for vision model compatibility with detailed recommendations.
    
    Args:
        model_name: Model name to check
        provider: Provider name (optional, for provider-specific checks)
        
    Returns:
        Dictionary with compatibility status and recommendations
    """
    from ..utils.structured_logging import get_logger
    logger = get_logger(__name__)
    
    result = {
        'model_name': model_name,
        'provider': provider,
        'compatible': False,
        'vision_support': False,
        'glyph_compatible': False,
        'warnings': [],
        'recommendations': [],
        'capabilities': {}
    }
    
    # Get model capabilities
    capabilities = get_model_capabilities(model_name)
    
    # Check if this is an unknown model (architecture is 'generic' means it wasn't found in database)
    is_unknown_model = capabilities.get('architecture') == 'generic' and not capabilities.get('vision_support', False)
    
    if is_unknown_model:
        result['warnings'].append(f"Model '{model_name}' not found in capabilities database")
        result['recommendations'].append("Add model specifications to model_capabilities.json")
        result['recommendations'].append("Using generic vision model fallback for VLM calculations")
        
        # Use generic fallback - assume vision support for unknown models
        _load_json_assets()
        if _model_capabilities and "generic_vision_model" in _model_capabilities:
            generic_caps = _model_capabilities["generic_vision_model"].copy()
            result['compatible'] = True
            result['vision_support'] = True
            result['capabilities'] = generic_caps
            
            # Also get vision capabilities using the generic model
            vision_caps = generic_caps.copy()
            result['vision_capabilities'] = vision_caps
            
            # Check Glyph compatibility with generic model
            glyph_caps = {
                'glyph_compatible': True,
                'model_name': model_name,
                'vision_support': True,
                'recommended_pages_per_image': 1,
                'recommended_dpi': 100
            }
            glyph_caps.update(vision_caps)
            result['glyph_compatible'] = True
            result['glyph_capabilities'] = glyph_caps
            
            logger.warning(f"Using generic vision model fallback for unknown model '{model_name}'")
        
        return result
    
    # Check vision support
    vision_support = capabilities.get('vision_support', False)
    result['vision_support'] = vision_support
    result['capabilities'] = capabilities
    
    if not vision_support:
        result['warnings'].append(f"Model '{model_name}' does not support vision")
        result['recommendations'].append("Use a vision-capable model for image processing")
        return result
    
    result['compatible'] = True
    
    # Get vision-specific capabilities
    vision_caps = get_vision_capabilities(model_name)
    result['vision_capabilities'] = vision_caps
    
    # Check Glyph compatibility
    glyph_caps = get_glyph_compression_capabilities(model_name)
    result['glyph_compatible'] = glyph_caps.get('glyph_compatible', False)
    result['glyph_capabilities'] = glyph_caps
    
    # Add specific recommendations based on capabilities
    if not vision_caps.get('image_patch_size'):
        result['warnings'].append("No image_patch_size specified, using generic fallback")
        result['recommendations'].append("Add image_patch_size to model capabilities for better accuracy")
    
    if not vision_caps.get('max_image_tokens'):
        result['warnings'].append("No max_image_tokens specified")
        result['recommendations'].append("Add max_image_tokens to model capabilities")
    
    return result
