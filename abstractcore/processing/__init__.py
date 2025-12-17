"""
AbstractCore Processing Module

Basic text processing capabilities built on top of AbstractCore,
demonstrating how to leverage the core infrastructure for real-world tasks.
"""

from .basic_summarizer import BasicSummarizer, SummaryStyle, SummaryLength, CompressionMode
from .basic_extractor import BasicExtractor
from .basic_judge import BasicJudge, JudgmentCriteria, Assessment, create_judge
from .basic_deepsearch import BasicDeepSearch, ResearchReport, ResearchFinding, ResearchPlan, ResearchSubTask
from .basic_intent import BasicIntentAnalyzer, IntentType, IntentDepth, IntentContext, IdentifiedIntent, IntentAnalysisOutput

__all__ = [
    'BasicSummarizer', 'SummaryStyle', 'SummaryLength', 'CompressionMode',
    'BasicExtractor',
    'BasicJudge', 'JudgmentCriteria', 'Assessment', 'create_judge',
    'BasicDeepSearch', 'ResearchReport', 'ResearchFinding', 'ResearchPlan', 'ResearchSubTask',
    'BasicIntentAnalyzer', 'IntentType', 'IntentDepth', 'IntentContext', 'IdentifiedIntent', 'IntentAnalysisOutput'
]