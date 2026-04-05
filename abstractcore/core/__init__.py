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
from .enums import ModelParameter, ModelCapability, MessageRole
from .interface import AbstractCoreInterface

__all__ = [
    'create_llm',
    'BasicSession',
    'CachedSession',
    'GenerateResponse',
    'Message',
    'ModelParameter',
    'ModelCapability',
    'MessageRole',
    'AbstractCoreInterface'
]
