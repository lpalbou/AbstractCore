"""
BasicDeepSearchV2 - Simple, Clean, and Effective Deep Research

A simplified deep research implementation that follows SOTA best practices:
- Intent-driven query understanding
- Iterative refinement with gap analysis
- Sequential learning (each search builds on previous findings)
- Quality-focused source selection
- Transparent verbose logging at every step

Key Improvements over V1:
- Simpler architecture (~500 lines vs 2000+)
- Smarter iterative approach (vs parallel blind searching)
- Better source quality assessment
- Clear reasoning transparency
- Easier to maintain and extend

Architecture:
    Stage 1: Intent & Query Understanding
    Stage 2: Iterative Research Loop (3-5 iterations)
    Stage 3: Source Quality Assessment
    Stage 4: Synthesis & Report Generation
"""

import json
import time
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from ..core.interface import AbstractCoreInterface
from ..core.factory import create_llm
from ..utils.structured_logging import get_logger
from ..tools.common_tools import web_search, fetch_url

# Try to import intent analyzer (optional dependency)
try:
    from .basic_intent import BasicIntentAnalyzer, IntentContext, IntentDepth
    INTENT_ANALYZER_AVAILABLE = True
except ImportError:
    INTENT_ANALYZER_AVAILABLE = False

logger = get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class QueryIntent(BaseModel):
    """User query intent analysis"""
    query_type: str = Field(description="Type of query: factual, exploratory, comparative, analytical")
    key_entities: List[str] = Field(description="Key entities and concepts in the query")
    underlying_goal: str = Field(description="What the user ultimately wants to learn")
    suggested_strategy: str = Field(description="Recommended research strategy")
    estimated_depth: str = Field(description="Estimated research depth needed: quick, standard, deep")


class SearchQuery(BaseModel):
    """Generated search query with reasoning"""
    query: str = Field(description="The actual search query to execute")
    reasoning: str = Field(description="Why this query was chosen")
    expected_findings: str = Field(description="What we expect to find")


class SourceQuality(BaseModel):
    """Source quality assessment"""
    url: str
    title: str
    authority_score: float = Field(description="Authority rating 0-40", ge=0, le=40)
    relevance_score: float = Field(description="Relevance rating 0-40", ge=0, le=40)
    quality_score: float = Field(description="Content quality rating 0-20", ge=0, le=20)
    total_score: float = Field(description="Total score 0-100", ge=0, le=100)
    reasoning: str = Field(description="Explanation of the scoring")


class ResearchFinding(BaseModel):
    """Single research finding with metadata"""
    source_url: str
    source_title: str
    content: str
    quality_assessment: SourceQuality
    timestamp: str
    iteration: int = Field(description="Which research iteration this came from")


class KnowledgeGap(BaseModel):
    """Identified gap in current knowledge"""
    gap_description: str = Field(description="What information is missing")
    importance: str = Field(description="How critical this gap is: critical, important, nice-to-have")
    next_query: str = Field(description="Suggested search query to fill this gap")


class ResearchReport(BaseModel):
    """Final research report"""
    query: str
    executive_summary: str
    key_findings: List[str] = Field(description="Main findings with citations")
    detailed_analysis: str
    sources: List[Dict[str, Any]]
    confidence_level: float = Field(description="Overall confidence in findings 0-1", ge=0, le=1)
    knowledge_gaps: List[str] = Field(description="Identified gaps that remain")
    research_path: List[str] = Field(description="Log of research steps taken")
    metadata: Dict[str, Any]


# ============================================================================
# BasicDeepSearchV2 - Main Class
# ============================================================================

class BasicDeepSearchV2:
    """
    Simple and effective deep research system with transparent reasoning.

    Example:
        >>> from abstractcore.processing import BasicDeepSearchV2
        >>> searcher = BasicDeepSearchV2()
        >>> report = searcher.research("What are the latest developments in LLMs in 2024?")
        >>> print(report.executive_summary)
    """

    def __init__(
        self,
        llm: Optional[AbstractCoreInterface] = None,
        max_tokens: int = 32000,
        max_output_tokens: int = 8000,
        timeout: Optional[float] = None,
        temperature: float = 0.1,
        verbose: bool = True,
        debug: bool = False
    ):
        """
        Initialize BasicDeepSearchV2

        Args:
            llm: AbstractCore LLM instance. If None, uses default Ollama model
            max_tokens: Maximum context tokens (default: 32000)
            max_output_tokens: Maximum output tokens (default: 8000)
            timeout: HTTP timeout in seconds (default: None for unlimited)
            temperature: LLM temperature for consistency (default: 0.1)
            verbose: Enable detailed logging (default: True)
            debug: Enable debug mode showing all sources and rejection reasons (default: False)
        """
        # Initialize LLM
        if llm is None:
            try:
                self.llm = create_llm(
                    "ollama",
                    model="qwen3:4b-instruct-2507-q4_K_M",
                    max_tokens=max_tokens,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default Ollama model: {e}\n\n"
                    "💡 Please provide a custom LLM instance:\n"
                    "   from abstractcore import create_llm\n"
                    "   from abstractcore.processing import BasicDeepSearchV2\n"
                    "   \n"
                    "   llm = create_llm('openai', model='gpt-4o-mini')\n"
                    "   searcher = BasicDeepSearchV2(llm)\n"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm

        self.verbose = verbose
        self.debug = debug
        self.research_path = []  # Track all steps taken

        # Initialize intent analyzer if available
        self.intent_analyzer = None
        if INTENT_ANALYZER_AVAILABLE:
            try:
                self.intent_analyzer = BasicIntentAnalyzer(llm=self.llm)
                logger.info("✅ Intent analyzer enabled")
            except Exception as e:
                logger.warning(f"Intent analyzer disabled: {e}")

        if self.verbose:
            logger.info(f"🤖 Initialized BasicDeepSearchV2 with {self.llm.provider}/{self.llm.model}")

    def research(
        self,
        query: str,
        max_iterations: int = 3,
        max_sources: int = 10,
        min_confidence: float = 0.7
    ) -> ResearchReport:
        """
        Conduct deep research on a query with iterative refinement.

        Args:
            query: The research question
            max_iterations: Maximum research iterations (default: 3)
            max_sources: Maximum sources to collect (default: 10)
            min_confidence: Minimum confidence to stop early (default: 0.7)

        Returns:
            ResearchReport with findings, sources, and metadata
        """
        logger.info("="*80)
        logger.info(f"🔬 Starting Deep Research V2")
        logger.info(f"📋 Query: {query}")
        logger.info("="*80)

        start_time = time.time()
        self.research_path = []

        # Stage 1: Intent & Query Understanding
        logger.info("\n🎯 STAGE 1: Understanding Query Intent...")
        intent = self._analyze_intent(query)
        self._log_intent(intent)

        # Stage 2: Iterative Research Loop
        logger.info("\n🔄 STAGE 2: Iterative Research Loop...")
        findings = self._iterative_research_loop(
            query=query,
            intent=intent,
            max_iterations=max_iterations,
            max_sources=max_sources,
            min_confidence=min_confidence
        )

        # Stage 3: Synthesis & Report Generation
        logger.info("\n📝 STAGE 3: Synthesizing Research Report...")
        report = self._generate_report(
            query=query,
            intent=intent,
            findings=findings,
            elapsed_time=time.time() - start_time
        )

        elapsed = time.time() - start_time
        logger.info(f"\n✨ Research completed in {elapsed:.1f}s")
        logger.info(f"📊 Sources found: {len(report.sources)}")
        logger.info(f"📈 Confidence level: {report.confidence_level:.2f}")
        logger.info("="*80)

        return report

    # ========================================================================
    # Stage 1: Intent Analysis
    # ========================================================================

    def _analyze_intent(self, query: str) -> QueryIntent:
        """Analyze query intent to guide research strategy"""

        # If intent analyzer is available, use it for detailed analysis
        if self.intent_analyzer:
            try:
                analysis = self.intent_analyzer.analyze_intent(
                    text=query,
                    context_type=IntentContext.STANDALONE,
                    depth=IntentDepth.UNDERLYING
                )

                # Convert to QueryIntent
                intent = QueryIntent(
                    query_type=analysis.primary_intent.intent_type.value,
                    key_entities=analysis.contextual_factors[:5],
                    underlying_goal=analysis.primary_intent.underlying_goal,
                    suggested_strategy=analysis.suggested_response_approach,
                    estimated_depth="standard"
                )

                self.research_path.append(f"Intent analysis: {intent.query_type} query")
                return intent

            except Exception as e:
                logger.warning(f"Intent analyzer failed, using fallback: {e}")
                logger.warning(f"  Query: \"{query}\"")
                logger.warning(f"  Switching to simple LLM-based intent analysis")

        # Fallback: simple LLM-based intent analysis
        prompt = f"""Analyze this research query and provide a structured response.

Query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
    "query_type": "factual|exploratory|comparative|analytical",
    "key_entities": ["entity1", "entity2", "entity3"],
    "underlying_goal": "what the user wants to learn",
    "suggested_strategy": "recommended research approach",
    "estimated_depth": "quick|standard|deep"
}}

Guidelines:
- factual: seeks specific facts or data
- exploratory: wants to understand a topic broadly
- comparative: comparing multiple things
- analytical: seeking analysis or evaluation

Be concise and specific."""

        try:
            response = self.llm.generate(prompt, temperature=0.1)
            response_text = self._extract_text(response)

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                intent_data = json.loads(response_text[json_start:json_end])
                intent = QueryIntent(**intent_data)
                self.research_path.append(f"Intent analysis: {intent.query_type} query")
                return intent

        except Exception as e:
            logger.warning(f"Intent analysis failed, using default: {e}")
            logger.warning(f"  Query: \"{query}\"")
            logger.warning(f"  Both intent analyzer and LLM fallback failed")
            logger.warning(f"  Using default 'exploratory' intent")

        # Ultimate fallback
        intent = QueryIntent(
            query_type="exploratory",
            key_entities=[],
            underlying_goal="General research on the query topic",
            suggested_strategy="Broad search followed by focused investigation",
            estimated_depth="standard"
        )

        logger.info(f"  Defaulting to exploratory research strategy")
        self.research_path.append("Intent analysis: default exploratory (all methods failed)")
        return intent

    def _log_intent(self, intent: QueryIntent):
        """Log intent analysis results"""
        logger.info(f"  ├─ Query Type: {intent.query_type.upper()}")
        if intent.key_entities:
            logger.info(f"  ├─ Key Entities: {', '.join(intent.key_entities[:3])}")
        logger.info(f"  ├─ Goal: {intent.underlying_goal}")
        logger.info(f"  ├─ Strategy: {intent.suggested_strategy}")
        logger.info(f"  └─ Estimated Depth: {intent.estimated_depth}")

    # ========================================================================
    # Stage 2: Iterative Research Loop
    # ========================================================================

    def _iterative_research_loop(
        self,
        query: str,
        intent: QueryIntent,
        max_iterations: int,
        max_sources: int,
        min_confidence: float
    ) -> List[ResearchFinding]:
        """
        Iterative research loop with gap analysis and adaptive querying.
        Each iteration builds on previous findings.
        """

        all_findings = []
        current_knowledge = ""
        previous_gaps = []  # Track knowledge gaps from previous iteration

        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n🔄 ITERATION {iteration}/{max_iterations}")

            # Check if we have enough sources
            if len(all_findings) >= max_sources:
                logger.info(f"  └─ ✅ Reached source limit ({max_sources}), stopping")
                self.research_path.append(f"Iteration {iteration}: Stopped (source limit reached)")
                break

            # Generate search query based on current knowledge and gaps
            search_query = self._generate_search_query(
                original_query=query,
                intent=intent,
                current_knowledge=current_knowledge,
                knowledge_gaps=previous_gaps,
                iteration=iteration
            )

            logger.info(f"  ├─ 🔍 Search Query: \"{search_query.query}\"")
            logger.info(f"  ├─ 💭 Reasoning: {search_query.reasoning}")
            logger.info(f"  └─ 🎯 Expected: {search_query.expected_findings}")

            self.research_path.append(f"Iteration {iteration}: {search_query.query}")

            # Execute search and fetch content
            # Always explore at least 10 URLs to find the best sources
            # (we'll filter by quality and keep only the top ones)
            explore_limit = 10  # Explore more sources than we keep
            findings = self._search_and_fetch(
                search_query=search_query.query,
                original_query=query,
                iteration=iteration,
                max_results=explore_limit
            )

            if not findings:
                logger.warning(f"  └─ ⚠️  No findings in iteration {iteration}")
                logger.warning(f"      Query used: \"{search_query.query}\"")
                logger.warning(f"      Expected to find: {search_query.expected_findings}")
                logger.warning(f"      Explored {explore_limit} potential sources but none yielded usable content")
                logger.warning(f"      This may indicate: (1) query too specific, (2) no web results, (3) all URLs failed to fetch")
                self.research_path.append(f"Iteration {iteration}: No findings from '{search_query.query}'")
                continue

            # Add findings and keep only the best ones if we exceed max_sources
            all_findings.extend(findings)

            # If we exceed max_sources, keep only the highest quality sources
            if len(all_findings) > max_sources:
                logger.info(f"  ├─ 📊 Sorting {len(all_findings)} sources by quality...")
                all_findings = sorted(all_findings, key=lambda f: f.quality_assessment.total_score, reverse=True)
                all_findings = all_findings[:max_sources]
                logger.info(f"  └─ ✅ Kept top {len(all_findings)} sources (quality >= {all_findings[-1].quality_assessment.total_score:.1f})")
            else:
                logger.info(f"  └─ ✅ Found {len(findings)} sources (total: {len(all_findings)})")

            # Update current knowledge summary
            current_knowledge = self._summarize_findings(all_findings)

            # Analyze knowledge gaps (decide if we should continue)
            if iteration < max_iterations:
                should_continue, gap_analysis, knowledge_gaps = self._should_continue_research(
                    original_query=query,
                    current_findings=current_knowledge,
                    iteration=iteration,
                    max_iterations=max_iterations
                )

                if not should_continue:
                    logger.info(f"  └─ ✅ Research complete: {gap_analysis}")
                    self.research_path.append(f"Iteration {iteration}: Research complete")
                    break
                else:
                    logger.info(f"  └─ 🔄 Continuing research: {gap_analysis}")
                    if knowledge_gaps:
                        logger.info(f"      └─ 🎯 Identified gaps: {', '.join(knowledge_gaps[:2])}")
                        previous_gaps = knowledge_gaps  # Store for next iteration

        return all_findings

    def _generate_search_query(
        self,
        original_query: str,
        intent: QueryIntent,
        current_knowledge: str,
        knowledge_gaps: List[str],
        iteration: int
    ) -> SearchQuery:
        """Generate targeted search query based on current knowledge state and identified gaps"""

        if iteration == 1:
            # First iteration: broad query based on intent
            prompt = f"""Generate an effective web search query for this research question.

Research Question: "{original_query}"
Query Type: {intent.query_type}
Goal: {intent.underlying_goal}

Generate a specific, focused search query that will find authoritative sources.
Include relevant year markers if asking about recent developments.

Respond with JSON:
{{
    "query": "the search query",
    "reasoning": "why this query will be effective",
    "expected_findings": "what sources we expect to find"
}}"""

        else:
            # Subsequent iterations: fill knowledge gaps explicitly
            gaps_text = "\n".join(f"- {gap}" for gap in knowledge_gaps[:3]) if knowledge_gaps else "No explicit gaps identified yet"

            prompt = f"""Generate a follow-up search query based on current research progress and identified knowledge gaps.

Original Question: "{original_query}"

Current Knowledge Summary:
{current_knowledge[:800]}

Identified Knowledge Gaps:
{gaps_text}

Generate a targeted search query to fill the most important gap. Focus on finding specific missing information.

Respond with JSON:
{{
    "query": "the search query",
    "reasoning": "what knowledge gap this addresses",
    "expected_findings": "what specific information we need"
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.2)
            response_text = self._extract_text(response)

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                query_data = json.loads(response_text[json_start:json_end])

                # Normalize expected_findings to string if it's a list
                if isinstance(query_data.get('expected_findings'), list):
                    query_data['expected_findings'] = ', '.join(query_data['expected_findings'])

                return SearchQuery(**query_data)

        except Exception as e:
            logger.warning(f"Query generation failed, using fallback: {e}")
            logger.warning(f"  Original query: \"{original_query}\"")
            logger.warning(f"  Iteration: {iteration}")
            logger.warning(f"  Intent type: {intent.query_type}")
            logger.warning(f"  Falling back to simple query formulation")

        # Fallback: simple query
        if iteration == 1:
            fallback_query = original_query
        else:
            fallback_query = f"{original_query} details"

        logger.info(f"  Using fallback query: \"{fallback_query}\"")
        return SearchQuery(
            query=fallback_query,
            reasoning="Fallback query generation (LLM unavailable)",
            expected_findings="General information about the topic"
        )

    def _search_and_fetch(
        self,
        search_query: str,
        original_query: str,
        iteration: int,
        max_results: int
    ) -> List[ResearchFinding]:
        """Execute search and fetch content from top results"""

        findings = []

        try:
            # Execute web search
            search_results = web_search(search_query, num_results=max_results)

            # Extract URLs
            urls = self._extract_urls_from_search(search_results)
            logger.info(f"      ├─ Found {len(urls)} URLs from search")

            # Debug mode: Show all URLs found
            if self.debug and urls:
                logger.info(f"      │")
                logger.info(f"      │  🔍 DEBUG: All {len(urls)} sources initially found:")
                for debug_idx, (debug_url, debug_title) in enumerate(urls, 1):
                    logger.info(f"      │    {debug_idx}. {debug_title}")
                    logger.info(f"      │       URL: {debug_url}")
                logger.info(f"      │")

            if not urls:
                # Show what we got instead of URLs
                search_preview = search_results[:500] if isinstance(search_results, str) else str(search_results)[:500]
                logger.warning(f"      └─ ⚠️  No URLs extracted from search results")
                logger.warning(f"         Query: \"{search_query}\"")
                logger.warning(f"         Search returned: {search_preview}...")
                logger.warning(f"         This may indicate: (1) web_search failed, (2) results have no links, (3) parsing issue")
                return findings

            # Fetch and assess each URL
            for idx, (url, title) in enumerate(urls[:max_results], 1):
                try:
                    logger.info(f"      ├─ [{idx}/{len(urls)}] Fetching: {title}")

                    # Fetch content
                    content = fetch_url(url, timeout=15)

                    if "Error" in content or len(content) < 100:
                        logger.info(f"      │  └─ ⚠️  Skipped (fetch error or too short)")
                        if self.debug:
                            error_preview = content[:200] if content else "No content"
                            logger.info(f"      │     🔍 DEBUG: Content preview: {error_preview}...")
                            logger.info(f"      │     🔍 DEBUG: Content length: {len(content)} bytes")
                        continue

                    # Extract relevant content
                    relevant_content = self._extract_relevant_content(content, original_query)

                    if not relevant_content:
                        logger.info(f"      │  └─ ⚠️  Skipped (no relevant content)")
                        if self.debug:
                            logger.info(f"      │     🔍 DEBUG: Content extracted but not relevant to query")
                            logger.info(f"      │     🔍 DEBUG: Original content length: {len(content)} bytes")
                        continue

                    # Assess source quality
                    quality = self._assess_source_quality(
                        url=url,
                        title=title,
                        content=relevant_content,
                        query=original_query
                    )

                    logger.info(f"      │  └─ 📊 Quality: {quality.total_score:.1f}/100 "
                               f"(A:{quality.authority_score:.0f} R:{quality.relevance_score:.0f} Q:{quality.quality_score:.0f})")

                    # Only keep high-quality sources (score >= 50)
                    if quality.total_score >= 50:
                        finding = ResearchFinding(
                            source_url=url,
                            source_title=title,
                            content=relevant_content,
                            quality_assessment=quality,
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                            iteration=iteration
                        )
                        findings.append(finding)
                        logger.info(f"      │  └─ ✅ Accepted (score: {quality.total_score:.1f})")
                        if self.debug:
                            logger.info(f"      │     🔍 DEBUG: Reasoning: {quality.reasoning}")
                    else:
                        logger.info(f"      │  └─ ❌ Rejected (score too low: {quality.total_score:.1f})")
                        if self.debug:
                            logger.info(f"      │     🔍 DEBUG: Authority: {quality.authority_score:.1f}/33")
                            logger.info(f"      │     🔍 DEBUG: Relevance: {quality.relevance_score:.1f}/33")
                            logger.info(f"      │     🔍 DEBUG: Quality: {quality.quality_score:.1f}/33")
                            logger.info(f"      │     🔍 DEBUG: Reasoning: {quality.reasoning}")
                            logger.info(f"      │     🔍 DEBUG: Threshold: 50 (rejected below this)")

                except Exception as e:
                    logger.warning(f"      │  └─ ❌ Failed to fetch {url}: {e}")
                    if self.debug:
                        logger.warning(f"      │     🔍 DEBUG: Exception type: {type(e).__name__}")
                        logger.warning(f"      │     🔍 DEBUG: Full error: {str(e)}")
                    continue

            logger.info(f"      └─ ✅ Collected {len(findings)} high-quality sources")

            # Debug mode: Show rejection summary
            if self.debug and urls:
                rejected_count = len(urls[:max_results]) - len(findings)
                if rejected_count > 0:
                    logger.info(f"")
                    logger.info(f"      🔍 DEBUG: Rejection Summary")
                    logger.info(f"         Total sources evaluated: {len(urls[:max_results])}")
                    logger.info(f"         Accepted: {len(findings)}")
                    logger.info(f"         Rejected: {rejected_count}")
                    logger.info(f"         Acceptance rate: {(len(findings) / len(urls[:max_results]) * 100):.1f}%")

        except Exception as e:
            logger.error(f"      └─ ❌ Search failed: {e}")

        return findings

    def _extract_relevant_content(self, content: str, query: str) -> str:
        """Extract relevant portions of content"""

        # Simple extraction: get text preview from fetch_url structured output
        lines = content.split('\n')
        text_lines = []
        in_text_section = False

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith('📄 Text Content Preview:'):
                in_text_section = True
                continue
            elif in_text_section and line_stripped.startswith(('📊', '🔗', '📋', '📰')):
                break
            elif in_text_section and line_stripped:
                text_lines.append(line_stripped)

        if text_lines:
            # Return extracted text (increased limit from 1000 to 5000 chars)
            extracted = ' '.join(text_lines)

            # Detect JavaScript-heavy pages with minimal content
            js_indicators = ['Loading...', 'JavaScript required', 'enable JavaScript',
                           'The system can\'t perform the operation now']
            if len(extracted) < 600 and any(indicator in extracted for indicator in js_indicators):
                logger.debug("Detected JavaScript-heavy page with minimal content")
                return None  # Skip this source

            return extracted[:5000] if len(extracted) > 5000 else extracted

        # Fallback: return larger content chunk
        return content[:5000] if len(content) > 5000 else content

    # ========================================================================
    # Stage 3: Source Quality Assessment
    # ========================================================================

    def _assess_source_quality(
        self,
        url: str,
        title: str,
        content: str,
        query: str
    ) -> SourceQuality:
        """
        Assess source quality with a simple scoring system:
        - Authority (0-40): Official sites, academic sources, reputable organizations
        - Relevance (0-40): How well content matches query
        - Quality (0-20): Information density and presentation
        """

        # Authority score (0-40)
        authority_score, authority_reason = self._score_authority(url, title)

        # Relevance score (0-40)
        relevance_score, relevance_reason = self._score_relevance(content, query)

        # Quality score (0-20)
        quality_score, quality_reason = self._score_quality(content)

        # Total score
        total_score = authority_score + relevance_score + quality_score

        reasoning = f"Authority: {authority_reason} | Relevance: {relevance_reason} | Quality: {quality_reason}"

        return SourceQuality(
            url=url,
            title=title,
            authority_score=authority_score,
            relevance_score=relevance_score,
            quality_score=quality_score,
            total_score=total_score,
            reasoning=reasoning
        )

    def _score_authority(self, url: str, title: str) -> tuple[float, str]:
        """Score source authority (0-40)"""

        url_lower = url.lower()
        title_lower = title.lower()

        # High authority domains (40 points)
        high_authority = [
            '.edu', '.gov', 'arxiv.org', 'ieee.org',
            'acm.org', 'nature.com', 'science.org', 'cell.com',
            'orcid.org', 'dblp.org', 'semanticscholar.org', 'researchgate.net'
        ]

        for domain in high_authority:
            if domain in url_lower:
                return 40.0, f"High authority domain: {domain}"

        # Medium authority (30 points)
        medium_authority = [
            'github.com', 'microsoft.com', 'google.com', 'openai.com',
            'anthropic.com', 'stanford.edu', 'mit.edu', 'berkeley.edu',
            'wikidata.org', 'wikipedia.org'
        ]

        for domain in medium_authority:
            if domain in url_lower:
                return 30.0, f"Medium authority: {domain}"

        # Reputable news/tech sites (20 points)
        reputable_sites = [
            'techcrunch', 'wired', 'reuters', 'bloomberg', 'nytimes',
            'wsj.com', 'theguardian', 'bbc.com', 'technologyreview.com'
        ]

        # JavaScript-heavy sites that don't work well (10 points penalty)
        # Note: These sites may load but extract poorly due to JavaScript
        js_heavy_sites = [
            'scholar.google', 'linkedin.com'
        ]

        for site in reputable_sites:
            if site in url_lower:
                return 20.0, f"Reputable source: {site}"

        # Default score
        return 10.0, "General web source"

    def _score_relevance(self, content: str, query: str) -> tuple[float, str]:
        """Score content relevance to query (0-40)"""

        content_lower = content.lower()
        query_words = [w.lower().strip('.,!?;:"()[]{}') for w in query.split() if len(w) > 2]

        if not query_words:
            return 20.0, "No query words to match"

        # Count keyword matches
        matches = sum(1 for word in query_words if word in content_lower)
        match_ratio = matches / len(query_words)

        # Score based on match ratio
        if match_ratio >= 0.8:
            score = 40.0
            reason = f"Excellent match ({matches}/{len(query_words)} keywords)"
        elif match_ratio >= 0.6:
            score = 32.0
            reason = f"Good match ({matches}/{len(query_words)} keywords)"
        elif match_ratio >= 0.4:
            score = 24.0
            reason = f"Moderate match ({matches}/{len(query_words)} keywords)"
        elif match_ratio >= 0.2:
            score = 16.0
            reason = f"Weak match ({matches}/{len(query_words)} keywords)"
        else:
            score = 8.0
            reason = f"Poor match ({matches}/{len(query_words)} keywords)"

        return score, reason

    def _score_quality(self, content: str) -> tuple[float, str]:
        """Score content quality (0-20)"""

        words = content.split()
        word_count = len(words)

        # Check for informational content markers
        info_markers = ['according', 'research', 'study', 'found', 'shows', 'data', 'analysis']
        marker_count = sum(1 for marker in info_markers if marker in content.lower())

        # Score based on length and informational content
        if word_count >= 200 and marker_count >= 3:
            return 20.0, f"High quality ({word_count} words, {marker_count} info markers)"
        elif word_count >= 100 and marker_count >= 2:
            return 15.0, f"Good quality ({word_count} words, {marker_count} info markers)"
        elif word_count >= 50:
            return 10.0, f"Moderate quality ({word_count} words)"
        else:
            return 5.0, f"Low quality ({word_count} words)"

    # ========================================================================
    # Gap Analysis & Continuation Logic
    # ========================================================================

    def _summarize_findings(self, findings: List[ResearchFinding]) -> str:
        """Create a summary of current findings"""

        if not findings:
            return "No findings yet."

        # Sort by quality score
        sorted_findings = sorted(findings, key=lambda f: f.quality_assessment.total_score, reverse=True)

        # Create summary from top findings
        summary_parts = []
        for idx, finding in enumerate(sorted_findings[:5], 1):
            snippet = finding.content[:200] + "..." if len(finding.content) > 200 else finding.content
            summary_parts.append(f"{idx}. [{finding.source_title}]: {snippet}")

        return "\n".join(summary_parts)

    def _should_continue_research(
        self,
        original_query: str,
        current_findings: str,
        iteration: int,
        max_iterations: int
    ) -> tuple[bool, str, List[str]]:
        """Decide if research should continue based on gap analysis

        Returns:
            (should_continue, reasoning, identified_gaps)
        """

        prompt = f"""Evaluate if this research is complete or needs more investigation.

Original Question: "{original_query}"

Current Findings Summary:
{current_findings}

Research Progress: Iteration {iteration}/{max_iterations}

Is this sufficient to answer the question comprehensively?
Consider:
- Do we have authoritative sources?
- Are the key aspects covered?
- What specific information is still missing?

Respond with JSON:
{{
    "is_complete": true/false,
    "reasoning": "brief explanation",
    "knowledge_gaps": ["specific gap 1", "specific gap 2", ...]
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.1)
            response_text = self._extract_text(response)

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
                is_complete = result.get('is_complete', False)
                reasoning = result.get('reasoning', 'No reasoning provided')
                gaps = result.get('knowledge_gaps', [])

                return not is_complete, reasoning, gaps

        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            logger.warning(f"  Query being analyzed: \"{original_query}\"")
            logger.warning(f"  Iteration: {iteration}/{max_iterations}")
            logger.warning(f"  Current findings: {len(current_findings)} characters")
            logger.warning(f"  Falling back to iteration-based continuation")

        # Fallback: continue if we have iterations left
        if iteration < max_iterations:
            return True, "Continuing to gather more sources (gap analysis unavailable)", []
        else:
            return False, "Reached maximum iterations", []

    # ========================================================================
    # Stage 4: Report Generation
    # ========================================================================

    def _generate_report(
        self,
        query: str,
        intent: QueryIntent,
        findings: List[ResearchFinding],
        elapsed_time: float
    ) -> ResearchReport:
        """Generate final research report"""

        if not findings:
            return self._create_no_findings_report(query, elapsed_time)

        # Sort findings by quality
        sorted_findings = sorted(findings, key=lambda f: f.quality_assessment.total_score, reverse=True)

        # Prepare findings summary for LLM
        findings_text = self._format_findings_for_report(sorted_findings)

        # Generate report
        prompt = f"""Generate a comprehensive research report based on these findings.

Research Question: "{query}"
Query Type: {intent.query_type}
Research Goal: {intent.underlying_goal}

Findings from {len(findings)} sources:
{findings_text}

Create a structured report with:
1. Executive summary (2-3 sentences)
2. Key findings (3-5 bullet points with citations)
3. Detailed analysis (2-3 paragraphs synthesizing the information)
4. Confidence assessment (0-1 scale)
5. Any remaining knowledge gaps

Respond with JSON:
{{
    "executive_summary": "concise summary",
    "key_findings": ["finding 1 with citation", "finding 2 with citation", ...],
    "detailed_analysis": "comprehensive analysis with citations",
    "confidence_level": 0.0-1.0,
    "knowledge_gaps": ["gap 1", "gap 2", ...]
}}

IMPORTANT: Always cite sources using [Source Title] format."""

        try:
            response = self.llm.generate(prompt, temperature=0.3)
            response_text = self._extract_text(response)

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                report_data = json.loads(response_text[json_start:json_end])

                # Create sources list
                sources = [
                    {
                        "title": f.source_title,
                        "url": f.source_url,
                        "quality_score": f.quality_assessment.total_score,
                        "iteration": f.iteration
                    }
                    for f in sorted_findings
                ]

                return ResearchReport(
                    query=query,
                    executive_summary=report_data.get('executive_summary', ''),
                    key_findings=report_data.get('key_findings', []),
                    detailed_analysis=report_data.get('detailed_analysis', ''),
                    sources=sources,
                    confidence_level=float(report_data.get('confidence_level', 0.5)),
                    knowledge_gaps=report_data.get('knowledge_gaps', []),
                    research_path=self.research_path,
                    metadata={
                        'elapsed_time': elapsed_time,
                        'total_sources': len(findings),
                        'avg_quality_score': sum(f.quality_assessment.total_score for f in findings) / len(findings),
                        'intent_type': intent.query_type
                    }
                )

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return self._create_fallback_report(query, sorted_findings, elapsed_time)

    def _format_findings_for_report(self, findings: List[ResearchFinding]) -> str:
        """Format findings for report generation"""

        formatted = []
        for idx, finding in enumerate(findings[:10], 1):  # Top 10 sources
            formatted.append(
                f"\n{idx}. [{finding.source_title}] (Quality: {finding.quality_assessment.total_score:.0f}/100)\n"
                f"   URL: {finding.source_url}\n"
                f"   Content: {finding.content[:400]}..."
            )

        return "\n".join(formatted)

    def _create_no_findings_report(self, query: str, elapsed_time: float) -> ResearchReport:
        """Create report when no findings available"""

        return ResearchReport(
            query=query,
            executive_summary="Research could not be completed due to inability to access web sources.",
            key_findings=[
                "No web sources could be accessed for this research query",
                "Manual research using alternative sources is recommended"
            ],
            detailed_analysis="The automated research process was unable to gather information from web sources.",
            sources=[],
            confidence_level=0.0,
            knowledge_gaps=["Complete information about the query topic"],
            research_path=self.research_path,
            metadata={'elapsed_time': elapsed_time, 'total_sources': 0}
        )

    def _create_fallback_report(
        self,
        query: str,
        findings: List[ResearchFinding],
        elapsed_time: float
    ) -> ResearchReport:
        """Create fallback report when LLM generation fails"""

        sources = [
            {
                "title": f.source_title,
                "url": f.source_url,
                "quality_score": f.quality_assessment.total_score
            }
            for f in findings
        ]

        return ResearchReport(
            query=query,
            executive_summary=f"Research gathered {len(findings)} sources about: {query}",
            key_findings=[f"Found information from {f.source_title}" for f in findings[:5]],
            detailed_analysis="Research completed but detailed synthesis unavailable. Review sources for details.",
            sources=sources,
            confidence_level=0.5,
            knowledge_gaps=[],
            research_path=self.research_path,
            metadata={'elapsed_time': elapsed_time, 'total_sources': len(findings)}
        )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _extract_text(self, response) -> str:
        """Extract text from LLM response (handle different response types)"""
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)

    def _extract_urls_from_search(self, search_results: str) -> List[tuple]:
        """Extract URLs and titles from search results"""
        urls = []
        lines = search_results.split('\n')

        current_title = ""
        for line in lines:
            line = line.strip()

            # Look for numbered results
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                current_title = line[2:].strip()

            # Look for URLs
            elif line.startswith('🔗'):
                url = line.replace('🔗', '').strip()
                if url.startswith('http'):
                    urls.append((url, current_title or "Web Result"))

        return urls
