"""
AbstractCore Processing Module

Basic text processing capabilities built on top of AbstractCore,
demonstrating how to leverage the core infrastructure for real-world tasks.
"""

from .basic_summarizer import BasicSummarizer, SummaryStyle, SummaryLength
from .basic_extractor import BasicExtractor, EntityType, RelationType, Entity, Relationship, ExtractionOutput

__all__ = [
    'BasicSummarizer', 'SummaryStyle', 'SummaryLength',
    'BasicExtractor', 'EntityType', 'RelationType', 'Entity', 'Relationship', 'ExtractionOutput'
]