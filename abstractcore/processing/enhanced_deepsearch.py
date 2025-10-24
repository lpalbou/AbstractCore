"""
Enhanced Deep Search - Clean improvements to BasicDeepSearch

Implements three key enhancements with minimal code:
1. Simple Iterative Refinement (20 lines)
2. Basic Token Budget Management (15 lines) 
3. Intent-Aware Planning (25 lines)

Total enhancement: ~60 lines of clean, focused code
"""

import time
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from .basic_deepsearch import BasicDeepSearch, ResearchReport, ResearchPlan
from .basic_intent import BasicIntentAnalyzer, IntentContext, IntentDepth
from ..core.interface import AbstractCoreInterface
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimpleTokenTracker:
    """Simple token budget management (15 lines of code)"""
    max_budget: int = 50000
    used_tokens: int = 0
    
    def track_usage(self, tokens: int) -> bool:
        """Track token usage and return True if budget allows continuation"""
        self.used_tokens += tokens
        return self.used_tokens < self.max_budget
        
    def get_remaining_budget(self) -> int:
        return max(0, self.max_budget - self.used_tokens)
        
    def can_continue(self) -> bool:
        """Simple check: can we afford at least one more LLM call?"""
        return self.get_remaining_budget() > 1000  # Reserve 1k tokens minimum


class EnhancedDeepSearch(BasicDeepSearch):
    """
    Enhanced Deep Search with three clean improvements:
    
    1. **Simple Iterative Refinement**: 3-step loop that identifies gaps and refines research
    2. **Basic Token Budget Management**: Prevents runaway costs with simple tracking
    3. **Intent-Aware Planning**: Uses BasicIntentAnalyzer to understand user intent
    
    Usage:
        # Basic usage (same as BasicDeepSearch)
        searcher = EnhancedDeepSearch()
        report = searcher.research("How do transformers work?")
        
        # With budget control
        searcher = EnhancedDeepSearch(token_budget=30000)
        report = searcher.research("Complex research query", enable_iterative_refinement=True)
        
        # With intent analysis
        searcher = EnhancedDeepSearch(enable_intent_analysis=True)
        report = searcher.research("I need help understanding quantum computing")
    """
    
    def __init__(
        self,
        llm: Optional[AbstractCoreInterface] = None,
        token_budget: int = 50000,
        enable_iterative_refinement: bool = True,
        enable_intent_analysis: bool = True,
        max_refinement_iterations: int = 2,
        **kwargs
    ):
        """Initialize Enhanced Deep Search with clean improvements
        
        Args:
            llm: AbstractCore LLM instance
            token_budget: Maximum tokens to use (default 50000)
            enable_iterative_refinement: Enable simple iterative refinement (default True)
            enable_intent_analysis: Enable intent-aware planning (default True)
            max_refinement_iterations: Max refinement cycles (default 2)
            **kwargs: Additional arguments passed to BasicDeepSearch
        """
        super().__init__(llm=llm, **kwargs)
        
        # Enhancement 1: Token Budget Management
        self.token_tracker = SimpleTokenTracker(token_budget)
        
        # Enhancement 2: Iterative Refinement
        self.enable_iterative_refinement = enable_iterative_refinement
        self.max_refinement_iterations = max_refinement_iterations
        
        # Enhancement 3: Intent Analysis
        self.enable_intent_analysis = enable_intent_analysis
        self.intent_analyzer = None
        if enable_intent_analysis:
            try:
                self.intent_analyzer = BasicIntentAnalyzer(llm=self.llm)
                logger.info("✅ Intent analysis enabled")
            except Exception as e:
                logger.warning(f"Intent analysis disabled due to error: {e}")
                self.enable_intent_analysis = False
        
        # Wrap LLM generate method to track token usage
        self._wrap_llm_for_token_tracking()
        
        logger.info(f"🚀 Enhanced DeepSearch initialized with budget: {token_budget} tokens")
    
    def _wrap_llm_for_token_tracking(self):
        """Wrap LLM generate method to automatically track token usage"""
        original_generate = self.llm.generate
        
        def tracked_generate(*args, **kwargs):
            if not self.token_tracker.can_continue():
                raise RuntimeError(f"Token budget exceeded ({self.token_tracker.used_tokens}/{self.token_tracker.max_budget})")
            
            result = original_generate(*args, **kwargs)
            
            # Estimate tokens (rough approximation: 4 chars = 1 token)
            result_text = result.content if hasattr(result, 'content') else str(result)
            estimated_tokens = len(result_text) // 4
            self.token_tracker.track_usage(estimated_tokens)
            
            logger.debug(f"Token usage: +{estimated_tokens} (total: {self.token_tracker.used_tokens}/{self.token_tracker.max_budget})")
            return result
            
        self.llm.generate = tracked_generate
    
    def research(
        self,
        query: str,
        enable_iterative_refinement: Optional[bool] = None,
        **kwargs
    ) -> Union[ResearchReport, Dict[str, Any]]:
        """Enhanced research with iterative refinement and intent analysis"""
        
        # Use instance setting if not overridden
        if enable_iterative_refinement is None:
            enable_iterative_refinement = self.enable_iterative_refinement
        
        logger.info(f"🔍 Enhanced research starting: {query}")
        logger.info(f"🎯 Budget: {self.token_tracker.max_budget} tokens")
        logger.info(f"🔄 Iterative refinement: {'enabled' if enable_iterative_refinement else 'disabled'}")
        logger.info(f"🧠 Intent analysis: {'enabled' if self.enable_intent_analysis else 'disabled'}")
        
        start_time = time.time()
        
        try:
            # Enhancement 3: Intent-Aware Planning (25 lines)
            if self.enable_intent_analysis:
                enhanced_query = self._analyze_intent_and_enhance_query(query)
                if enhanced_query != query:
                    logger.info(f"🧠 Query enhanced based on intent analysis")
                    query = enhanced_query
            
            # Get initial research report using parent method
            initial_report = super().research(query, **kwargs)
            
            # Enhancement 2: Simple Iterative Refinement (20 lines)
            if enable_iterative_refinement and self.token_tracker.can_continue():
                final_report = self._simple_iterative_refinement(query, initial_report)
            else:
                final_report = initial_report
            
            # Add enhancement metadata
            if hasattr(final_report, 'methodology'):
                enhancements = []
                if self.enable_intent_analysis:
                    enhancements.append("intent-aware planning")
                if enable_iterative_refinement:
                    enhancements.append("iterative refinement")
                enhancements.append(f"token budget management ({self.token_tracker.used_tokens}/{self.token_tracker.max_budget} tokens used)")
                
                final_report.methodology += f" Enhanced with: {', '.join(enhancements)}."
            
            elapsed_time = time.time() - start_time
            logger.info(f"✨ Enhanced research completed in {elapsed_time:.1f} seconds")
            logger.info(f"📊 Token usage: {self.token_tracker.used_tokens}/{self.token_tracker.max_budget}")
            
            return final_report
            
        except Exception as e:
            if "Token budget exceeded" in str(e):
                logger.warning(f"⚠️ Research stopped due to token budget limit")
                # Return partial results if available
                return self._create_budget_exceeded_report(query)
            else:
                raise
    
    def _analyze_intent_and_enhance_query(self, query: str) -> str:
        """Enhancement 3: Intent-Aware Planning (25 lines of code)"""
        try:
            # Analyze user intent
            intent_analysis = self.intent_analyzer.analyze_intent(
                query,
                context_type=IntentContext.STANDALONE,
                depth=IntentDepth.UNDERLYING
            )
            
            primary_intent = intent_analysis.primary_intent
            logger.info(f"🧠 Detected intent: {primary_intent.intent_type.value} (confidence: {primary_intent.confidence:.2f})")
            
            # Enhance query based on intent
            if primary_intent.intent_type.value in ["information_seeking", "problem_solving"]:
                enhancement_prompt = f"""
                The user asked: "{query}"
                
                Their underlying goal is: {primary_intent.underlying_goal}
                Intent type: {primary_intent.intent_type.value}
                
                Rewrite this as a more specific, research-focused query that addresses their underlying goal.
                If the original query is already well-formed, return it unchanged.
                
                Respond with just the enhanced query, no explanation.
                """
                
                response = self.llm.generate(enhancement_prompt)
                enhanced_query = response.content if hasattr(response, 'content') else str(response)
                enhanced_query = enhanced_query.strip()
                
                # Clean up the response (remove quotes if present)
                if enhanced_query.startswith('"') and enhanced_query.endswith('"'):
                    enhanced_query = enhanced_query[1:-1]
                
                return enhanced_query
            
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
        
        return query  # Return original query if enhancement fails
    
    def _simple_iterative_refinement(self, original_query: str, initial_report: ResearchReport) -> ResearchReport:
        """Enhancement 1: Simple Iterative Refinement (20 lines of code)"""
        current_report = initial_report
        
        for iteration in range(self.max_refinement_iterations):
            if not self.token_tracker.can_continue():
                logger.info(f"🔄 Stopping refinement due to token budget")
                break
                
            logger.info(f"🔄 Refinement iteration {iteration + 1}/{self.max_refinement_iterations}")
            
            # Simple gap analysis
            gap_analysis_prompt = f"""
            Original question: {original_query}
            Current research summary: {current_report.executive_summary}
            Current key findings: {current_report.key_findings}
            
            What important information is still missing to fully answer the original question?
            If the research is complete, respond with "COMPLETE".
            If gaps exist, provide ONE specific search query to address the most important gap.
            
            Respond with just the search query or "COMPLETE", no explanation.
            """
            
            response = self.llm.generate(gap_analysis_prompt)
            gap_response = response.content if hasattr(response, 'content') else str(response)
            gap_response = gap_response.strip()
            
            if gap_response.upper() == "COMPLETE" or not gap_response:
                logger.info("✅ Research deemed complete by gap analysis")
                break
            
            logger.info(f"🎯 Identified gap, searching: {gap_response}")
            
            # Execute targeted search for the gap
            try:
                # Use the existing web search infrastructure
                from ..tools.common_tools import web_search, fetch_url
                
                search_results = web_search(gap_response, num_results=3)
                
                if search_results and "Error searching internet" not in search_results:
                    # Extract URLs and get content from first result
                    urls = self._extract_urls_from_search(search_results)
                    if urls:
                        url, title = urls[0]  # Take first URL
                        content = fetch_url(url, timeout=15)
                        
                        if content and "Error" not in content:
                            # Update report with new information
                            update_prompt = f"""
                            Original research summary: {current_report.executive_summary}
                            
                            New information found: {content[:1000]}...
                            
                            Update the research summary to incorporate this new information.
                            Keep it concise (max 3 sentences).
                            """
                            
                            response = self.llm.generate(update_prompt)
                            updated_summary = response.content if hasattr(response, 'content') else str(response)
                            updated_summary = updated_summary.strip()
                            current_report.executive_summary = updated_summary
                            
                            logger.info(f"✅ Report updated with new information from {title}")
                        
            except Exception as e:
                logger.warning(f"Gap search failed: {e}")
                continue
        
        return current_report
    
    def _create_budget_exceeded_report(self, query: str) -> ResearchReport:
        """Create a report when token budget is exceeded"""
        from .basic_deepsearch import ResearchReport
        
        return ResearchReport(
            title=f"Partial Research Report: {query} (Budget Exceeded)",
            executive_summary="Research was terminated due to token budget limits. This represents partial findings only.",
            key_findings=[
                "Token budget exceeded during research process",
                "Partial results may be incomplete",
                "Consider increasing token budget for comprehensive research"
            ],
            detailed_analysis="Research process was automatically terminated when the allocated token budget was exceeded. This helps prevent runaway costs while still providing partial insights.",
            conclusions="Increase token budget or simplify query for complete research.",
            sources=[],
            methodology=f"Enhanced deep search with token budget management. Budget exceeded at {self.token_tracker.used_tokens} tokens.",
            limitations="Research incomplete due to token budget constraints. Results may not be comprehensive."
        )
