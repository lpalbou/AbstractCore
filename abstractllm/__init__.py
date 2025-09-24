"""
AbstractLLM - Unified interface to all LLM providers with essential infrastructure.
"""

__version__ = "2.1.0"

from .core.factory import create_llm
from .core.session import BasicSession
from .core.types import GenerateResponse, Message
from .core.enums import ModelParameter, ModelCapability, MessageRole
from .exceptions import ModelNotFoundError, ProviderAPIError, AuthenticationError

# Embeddings module (optional import)
try:
    from .embeddings import EmbeddingManager
    _has_embeddings = True
except ImportError:
    _has_embeddings = False

__all__ = [
    'create_llm',
    'BasicSession',
    'GenerateResponse',
    'Message',
    'ModelParameter',
    'ModelCapability',
    'MessageRole',
    'ModelNotFoundError',
    'ProviderAPIError',
    'AuthenticationError'
]

if _has_embeddings:
    __all__.append('EmbeddingManager')