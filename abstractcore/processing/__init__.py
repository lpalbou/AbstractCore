"""
AbstractCore Processing Module

Basic text processing capabilities built on top of AbstractCore,
demonstrating how to leverage the core infrastructure for real-world tasks.
"""

from .basic_summarizer import BasicSummarizer, SummaryStyle, SummaryLength
from .basic_extractor import BasicExtractor
from .basic_judge import BasicJudge, JudgmentCriteria, Assessment, create_judge
from .basic_deepsearch import BasicDeepSearch, ResearchReport, ResearchFinding, ResearchPlan, ResearchSubTask
from .basic_deepsearchv2 import BasicDeepSearchV2
from .basic_intent import BasicIntentAnalyzer, IntentType, IntentDepth, IntentContext, IdentifiedIntent, IntentAnalysisOutput
from .basic_deepresearcherA import BasicDeepResearcherA
from .basic_deepresearcherB import BasicDeepResearcherB

__all__ = [
    'BasicSummarizer', 'SummaryStyle', 'SummaryLength',
    'BasicExtractor',
    'BasicJudge', 'JudgmentCriteria', 'Assessment', 'create_judge',
    'BasicDeepSearch', 'ResearchReport', 'ResearchFinding', 'ResearchPlan', 'ResearchSubTask',
    'BasicDeepSearchV2',
    'BasicIntentAnalyzer', 'IntentType', 'IntentDepth', 'IntentContext', 'IdentifiedIntent', 'IntentAnalysisOutput',
    'BasicDeepResearcherA', 'BasicDeepResearcherB'
]