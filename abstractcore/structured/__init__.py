"""
Structured output support for AbstractCore.

This module provides structured output capabilities using Pydantic models
with automatic validation and retry mechanisms.
"""

from .retry import Retry, FeedbackRetry
from .handler import StructuredOutputHandler
from .schema_compat import (
    SchemaRejectionRegistry,
    is_schema_rejection_error,
    schema_rejection_registry,
)

__all__ = [
    "Retry",
    "FeedbackRetry",
    "StructuredOutputHandler",
    "SchemaRejectionRegistry",
    "is_schema_rejection_error",
    "schema_rejection_registry",
]