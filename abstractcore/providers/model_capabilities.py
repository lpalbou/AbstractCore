"""
Model capability definitions for input and output filtering.

This module provides clear enums for filtering models based on what types of
input they can process and what types of output they can generate.

Key Concepts:
- Input Capabilities: What data types can the model accept and analyze?
- Output Capabilities: What data types can the model generate?

Examples:
    >>> from abstractcore.providers.model_capabilities import ModelInputCapability, ModelOutputCapability
    >>> from abstractcore.providers import OllamaProvider
    >>> 
    >>> # Get models that can analyze images
    >>> vision_models = OllamaProvider.list_available_models(
    ...     input_capabilities=[ModelInputCapability.IMAGE]
    ... )
    >>> 
    >>> # Get embedding models
    >>> embedding_models = OllamaProvider.list_available_models(
    ...     output_capabilities=[ModelOutputCapability.EMBEDDINGS]
    ... )
    >>> 
    >>> # Get vision models that generate text (most common case)
    >>> vision_text_models = OllamaProvider.list_available_models(
    ...     input_capabilities=[ModelInputCapability.TEXT, ModelInputCapability.IMAGE],
    ...     output_capabilities=[ModelOutputCapability.TEXT]
    ... )
"""

from enum import Enum
from typing import List, Set, Optional, Dict, Any
from ..architectures.detection import get_model_capabilities


class ModelInputCapability(Enum):
    """
    Enumeration of input data types that models can process and analyze.
    
    These capabilities define what types of input data a model can accept
    and understand. Most multimodal models support TEXT plus one or more
    additional input types.
    
    Values:
        TEXT: Model can process text input (all models support this)
        IMAGE: Model can analyze and understand images (vision models)
        AUDIO: Model can process and analyze audio input
        VIDEO: Model can analyze video content
    
    Examples:
        >>> # Text-only model
        >>> text_only = [ModelInputCapability.TEXT]
        >>> 
        >>> # Vision model (supports both text and images)
        >>> vision_model = [ModelInputCapability.TEXT, ModelInputCapability.IMAGE]
        >>> 
        >>> # Audio model (supports both text and audio)
        >>> audio_model = [ModelInputCapability.TEXT, ModelInputCapability.AUDIO]
    """
    
    TEXT = "text"
    """Model can process and understand text input (supported by all models)"""
    
    IMAGE = "image"
    """Model can analyze and understand image input (vision models)"""
    
    AUDIO = "audio"
    """Model can process and analyze audio input"""
    
    VIDEO = "video"
    """Model can analyze and understand video input"""


class ModelOutputCapability(Enum):
    """
    Enumeration of output data types that models can generate.
    
    These capabilities define what types of output a model can produce.
    Currently, AbstractCore supports text generation and embedding generation.
    
    Values:
        TEXT: Model generates text responses (most common)
        EMBEDDINGS: Model generates vector embeddings (embedding models)
    
    Examples:
        >>> # Regular chat/completion model
        >>> text_model = [ModelOutputCapability.TEXT]
        >>> 
        >>> # Embedding model
        >>> embedding_model = [ModelOutputCapability.EMBEDDINGS]
    
    Note:
        Future versions may include IMAGE, AUDIO, VIDEO for generative models.
    """
    
    TEXT = "text"
    """Model generates text responses (chat, completion, etc.)"""
    
    EMBEDDINGS = "embeddings"
    """Model generates vector embeddings for semantic search/similarity"""


def get_model_input_capabilities(model_name: str) -> List[ModelInputCapability]:
    """
    Determine what input capabilities a model supports.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        List of input capabilities the model supports
        
    Examples:
        >>> caps = get_model_input_capabilities("gpt-4-vision-preview")
        >>> print(caps)
        [<ModelInputCapability.TEXT: 'text'>, <ModelInputCapability.IMAGE: 'image'>]
        
        >>> caps = get_model_input_capabilities("gpt-4")
        >>> print(caps)
        [<ModelInputCapability.TEXT: 'text'>]
    """
    try:
        capabilities = get_model_capabilities(model_name)
    except Exception:
        # If we can't get capabilities, assume text-only
        return [ModelInputCapability.TEXT]
    
    input_caps = [ModelInputCapability.TEXT]  # All models support text
    
    if capabilities.get("vision_support", False):
        input_caps.append(ModelInputCapability.IMAGE)
    
    if capabilities.get("audio_support", False):
        input_caps.append(ModelInputCapability.AUDIO)
    
    if capabilities.get("video_support", False):
        input_caps.append(ModelInputCapability.VIDEO)
    
    return input_caps


def get_model_output_capabilities(model_name: str) -> List[ModelOutputCapability]:
    """
    Determine what output capabilities a model supports.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        List of output capabilities the model supports
        
    Examples:
        >>> caps = get_model_output_capabilities("gpt-4")
        >>> print(caps)
        [<ModelOutputCapability.TEXT: 'text'>]
        
        >>> caps = get_model_output_capabilities("text-embedding-3-small")
        >>> print(caps)
        [<ModelOutputCapability.EMBEDDINGS: 'embeddings'>]
    """
    try:
        capabilities = get_model_capabilities(model_name)
    except Exception:
        # If we can't get capabilities, assume text generation
        return [ModelOutputCapability.TEXT]
    
    # Check if it's explicitly marked as an embedding model
    if capabilities.get("model_type") == "embedding":
        return [ModelOutputCapability.EMBEDDINGS]
    
    # Check for embedding model name patterns
    model_lower = model_name.lower()
    embedding_patterns = [
        "embedding", "embed", "embeddings", 
        "text-embedding", "sentence-transformer",
        "all-minilm", "nomic-embed", "granite-embedding",
        "qwen3-embedding", "embeddinggemma"
    ]
    
    if any(pattern in model_lower for pattern in embedding_patterns):
        return [ModelOutputCapability.EMBEDDINGS]
    
    # Default to text generation
    return [ModelOutputCapability.TEXT]


def model_matches_input_capabilities(
    model_name: str, 
    required_capabilities: List[ModelInputCapability]
) -> bool:
    """
    Check if a model supports all required input capabilities.
    
    Args:
        model_name: Name of the model to check
        required_capabilities: List of required input capabilities
        
    Returns:
        True if model supports all required capabilities, False otherwise
        
    Examples:
        >>> # Check if model supports both text and image input
        >>> required = [ModelInputCapability.TEXT, ModelInputCapability.IMAGE]
        >>> model_matches_input_capabilities("gpt-4-vision-preview", required)
        True
        
        >>> model_matches_input_capabilities("gpt-4", required)
        False
    """
    if not required_capabilities:
        return True
    
    model_caps = get_model_input_capabilities(model_name)
    model_caps_set = set(model_caps)
    required_set = set(required_capabilities)
    
    return required_set.issubset(model_caps_set)


def model_matches_output_capabilities(
    model_name: str, 
    required_capabilities: List[ModelOutputCapability]
) -> bool:
    """
    Check if a model supports all required output capabilities.
    
    Args:
        model_name: Name of the model to check
        required_capabilities: List of required output capabilities
        
    Returns:
        True if model supports all required capabilities, False otherwise
        
    Examples:
        >>> # Check if model generates text
        >>> required = [ModelOutputCapability.TEXT]
        >>> model_matches_output_capabilities("gpt-4", required)
        True
        
        >>> # Check if model generates embeddings
        >>> required = [ModelOutputCapability.EMBEDDINGS]
        >>> model_matches_output_capabilities("text-embedding-3-small", required)
        True
        >>> model_matches_output_capabilities("gpt-4", required)
        False
    """
    if not required_capabilities:
        return True
    
    model_caps = get_model_output_capabilities(model_name)
    model_caps_set = set(model_caps)
    required_set = set(required_capabilities)
    
    return required_set.issubset(model_caps_set)


def filter_models_by_capabilities(
    models: List[str],
    input_capabilities: Optional[List[ModelInputCapability]] = None,
    output_capabilities: Optional[List[ModelOutputCapability]] = None
) -> List[str]:
    """
    Filter a list of models based on input and output capability requirements.
    
    Args:
        models: List of model names to filter
        input_capabilities: Required input capabilities (None = no filtering)
        output_capabilities: Required output capabilities (None = no filtering)
        
    Returns:
        Filtered list of model names that match all requirements
        
    Examples:
        >>> models = ["gpt-4", "gpt-4-vision-preview", "text-embedding-3-small"]
        >>> 
        >>> # Get vision models
        >>> vision_models = filter_models_by_capabilities(
        ...     models, 
        ...     input_capabilities=[ModelInputCapability.IMAGE]
        ... )
        >>> print(vision_models)
        ['gpt-4-vision-preview']
        >>> 
        >>> # Get embedding models
        >>> embedding_models = filter_models_by_capabilities(
        ...     models,
        ...     output_capabilities=[ModelOutputCapability.EMBEDDINGS]
        ... )
        >>> print(embedding_models)
        ['text-embedding-3-small']
        >>> 
        >>> # Get text generation models
        >>> text_models = filter_models_by_capabilities(
        ...     models,
        ...     output_capabilities=[ModelOutputCapability.TEXT]
        ... )
        >>> print(text_models)
        ['gpt-4', 'gpt-4-vision-preview']
    """
    filtered_models = []
    
    for model_name in models:
        try:
            # Check input capabilities
            if input_capabilities and not model_matches_input_capabilities(model_name, input_capabilities):
                continue
                
            # Check output capabilities
            if output_capabilities and not model_matches_output_capabilities(model_name, output_capabilities):
                continue
                
            filtered_models.append(model_name)
        except Exception:
            # If we can't get capabilities, skip this model
            # (it likely doesn't have an entry in model_capabilities.json)
            continue
    
    return filtered_models


def get_capability_summary(model_name: str) -> Dict[str, Any]:
    """
    Get a comprehensive summary of a model's input and output capabilities.
    
    Args:
        model_name: Name of the model to analyze
        
    Returns:
        Dictionary containing input and output capabilities
        
    Examples:
        >>> summary = get_capability_summary("gpt-4-vision-preview")
        >>> print(summary)
        {
            'model_name': 'gpt-4-vision-preview',
            'input_capabilities': ['text', 'image'],
            'output_capabilities': ['text'],
            'is_multimodal': True,
            'is_embedding_model': False
        }
    """
    input_caps = get_model_input_capabilities(model_name)
    output_caps = get_model_output_capabilities(model_name)
    
    return {
        'model_name': model_name,
        'input_capabilities': [cap.value for cap in input_caps],
        'output_capabilities': [cap.value for cap in output_caps],
        'is_multimodal': len(input_caps) > 1,
        'is_embedding_model': ModelOutputCapability.EMBEDDINGS in output_caps
    }
