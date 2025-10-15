"""
AbstractCore Processing Module

Basic text processing capabilities built on top of AbstractCore,
demonstrating how to leverage the core infrastructure for real-world tasks.
"""

from .basic_summarizer import BasicSummarizer, SummaryStyle, SummaryLength
from .basic_extractor import BasicExtractor
from .basic_judge import BasicJudge, JudgmentCriteria, Assessment, create_judge

__all__ = [
    'BasicSummarizer', 'SummaryStyle', 'SummaryLength',
    'BasicExtractor',
    'BasicJudge', 'JudgmentCriteria', 'Assessment', 'create_judge'
]