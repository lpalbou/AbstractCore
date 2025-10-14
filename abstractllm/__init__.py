# -*- coding: utf-8 -*-
"""
AbstractLLM - Unified interface to all LLM providers with essential infrastructure.

Key Features:
• Multi-provider support (OpenAI, Anthropic, Ollama, HuggingFace, MLX, LMStudio)
• Unified token parameter vocabulary across all providers
• Production-ready retry strategies and circuit breakers
• Tool calling and structured output support
• Vector embeddings with SOTA models
• Event system for observability

Quick Start:
    from abstractllm import create_llm

    # Unified token management across all providers
    llm = create_llm(
        provider="openai",
        model="gpt-4o",
        max_tokens=8000,        # Total budget
        max_output_tokens=2000  # Reserve for output
    )

    response = llm.generate("Hello, world!")
    print(response.content)
"""

from .utils.version import __version__

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

# Processing module (core functionality)
from .processing import BasicSummarizer, SummaryStyle, SummaryLength, BasicExtractor
_has_processing = True

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

# Processing is core functionality
__all__.extend(['BasicSummarizer', 'SummaryStyle', 'SummaryLength', 'BasicExtractor'])