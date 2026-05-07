"""
AbstractCore - Core abstractions and interfaces

This module provides the fundamental building blocks for AbstractCore:
- Factory functions for creating LLM providers
- Session management for conversation tracking
- Type definitions and interfaces
- Provider abstractions
"""

from .factory import create_llm
from .session import BasicSession
from .cached_session import CachedSession
from .types import GenerateResponse, Message
from .multimodal_generation import (
    GeneratedItem,
    GeneratedResource,
    GenerationIssue,
    MultimodalGenerateResponse,
)
from .output_specs import (
    GenerationOutputSpec,
    is_output_request,
    normalize_output_spec,
    normalize_output_specs,
    output_has_generated_media,
    output_requires_non_chat_dispatch,
    strip_runtime_output_metadata,
)
from .enums import ModelParameter, ModelCapability, MessageRole
from .interface import AbstractCoreInterface

__all__ = [
    'create_llm',
    'BasicSession',
    'CachedSession',
    'GenerateResponse',
    'GenerationOutputSpec',
    'GeneratedItem',
    'GeneratedResource',
    'GenerationIssue',
    'MultimodalGenerateResponse',
    'is_output_request',
    'normalize_output_spec',
    'normalize_output_specs',
    'output_has_generated_media',
    'output_requires_non_chat_dispatch',
    'strip_runtime_output_metadata',
    'Message',
    'ModelParameter',
    'ModelCapability',
    'MessageRole',
    'AbstractCoreInterface'
]
