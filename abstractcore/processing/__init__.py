"""
AbstractCore Processing Module

Basic text processing capabilities built on top of AbstractCore,
demonstrating how to leverage the core infrastructure for real-world tasks.
"""

from __future__ import annotations

# Keep this package import-safe for minimal installs.
# Some processing apps pull optional deps (e.g. DeepSearch uses built-in web tools).
from importlib import import_module
from typing import Any

from .basic_summarizer import BasicSummarizer, SummaryStyle, SummaryLength, CompressionMode
from .basic_extractor import BasicExtractor
from .basic_judge import BasicJudge, JudgmentCriteria, Assessment, create_judge
from .basic_intent import (
    BasicIntentAnalyzer,
    IntentType,
    IntentDepth,
    IntentContext,
    IdentifiedIntent,
    IntentAnalysisOutput,
)


def __getattr__(name: str) -> Any:
    lazy = {
        "BasicDeepSearch",
        "ResearchReport",
        "ResearchFinding",
        "ResearchPlan",
        "ResearchSubTask",
    }
    if name in lazy:
        mod = import_module("abstractcore.processing.basic_deepsearch")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BasicSummarizer', 'SummaryStyle', 'SummaryLength', 'CompressionMode',
    'BasicExtractor',
    'BasicJudge', 'JudgmentCriteria', 'Assessment', 'create_judge',
    'BasicDeepSearch', 'ResearchReport', 'ResearchFinding', 'ResearchPlan', 'ResearchSubTask',
    'BasicIntentAnalyzer', 'IntentType', 'IntentDepth', 'IntentContext', 'IdentifiedIntent', 'IntentAnalysisOutput'
]
