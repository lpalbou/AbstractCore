"""
Basic Deep Researcher B - Hierarchical Planning with Progressive Refinement

This implementation follows a structured planning-first approach with emphasis on
information quality and progressive knowledge building.

Architecture:
1. Hierarchical Planning: Structured decomposition into atomic questions
2. Quality-Focused Retrieval: Source credibility scoring and filtering
3. Content Extraction: Full-text analysis of selected sources
4. Progressive Refinement: Iterative knowledge graph building
5. Contradiction Detection: Identifying and resolving conflicting information

Key Features:
- Detailed upfront planning with dependency tracking
- Source quality scoring (credibility, recency, authority)
- Semantic deduplication of information
- Contradiction detection and resolution
- Knowledge graph construction for coherence
"""

import json
import time
import re
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.interface import AbstractCoreInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry
from ..utils.structured_logging import get_logger
from ..tools.common_tools import web_search, fetch_url
from .basic_summarizer import BasicSummarizer, SummaryStyle, SummaryLength
from .basic_intent import BasicIntentAnalyzer, IntentContext, IntentDepth

logger = get_logger(__name__)


# ==================== Data Models ====================

@dataclass
class AtomicQuestion:
    """Represents an atomic research question"""
    id: str
    question: str
    category: str
    priority: int  # 1=critical, 2=important, 3=supplementary
    dependencies: List[str] = field(default_factory=list)  # IDs of questions that must be answered first
    status: str = "pending"  # pending, researching, completed
    findings: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    id: str
    concept: str
    facts: List[str]
    sources: List[str]
    confidence: float
    contradictions: List[str] = field(default_factory=list)
    related_nodes: List[str] = field(default_factory=list)


class ResearchPlanModel(BaseModel):
    """Hierarchical research plan"""
    research_goal: str = Field(description="Clear research goal statement")
    categories: List[Dict[str, Any]] = Field(description="3-5 research categories")
    atomic_questions: List[Dict[str, Any]] = Field(description="8-15 atomic questions")
    dependencies: Dict[str, List[str]] = Field(description="Question dependencies")
    estimated_depth: str = Field(description="shallow, medium, or deep")


class QueriesModel(BaseModel):
    """Search queries for an atomic question"""
    primary_query: str = Field(description="Main search query")
    alternative_queries: List[str] = Field(description="2 alternative formulations", min_items=2, max_items=2)


class SourceQualityModel(BaseModel):
    """Source quality assessment"""
    credibility_score: float = Field(description="Credibility 0-1", ge=0, le=1)
    recency_score: float = Field(description="Recency 0-1", ge=0, le=1)
    relevance_score: float = Field(description="Relevance 0-1", ge=0, le=1)
    authority_indicators: List[str] = Field(description="Authority indicators", max_items=3)
    red_flags: List[str] = Field(description="Quality concerns", max_items=3)
    should_include: bool = Field(description="Whether to include source")


class ContentAnalysisModel(BaseModel):
    """Analysis of extracted content"""
    key_facts: List[str] = Field(description="3-7 key facts", min_items=3, max_items=7)
    claims_needing_verification: List[str] = Field(description="Claims to verify", max_items=3)
    confidence_level: float = Field(description="Confidence 0-1", ge=0, le=1)
    contradicts_existing: bool = Field(description="Contradicts existing knowledge")
    contradiction_details: Optional[str] = Field(description="Details of contradiction if any")


class ContradictionResolutionModel(BaseModel):
    """Resolution of contradictory information"""
    contradiction_summary: str = Field(description="Summary of contradiction")
    resolution: str = Field(description="How to resolve it")
    more_credible_source: str = Field(description="Which source is more credible")
    requires_additional_research: bool = Field(description="Need more research")


class FinalReportModel(BaseModel):
    """Final research report"""
    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Executive summary 2-3 paragraphs")
    main_findings: List[Dict[str, str]] = Field(description="5-10 findings with evidence")
    detailed_sections: List[Dict[str, Any]] = Field(description="Detailed sections")
    methodology: str = Field(description="Research methodology description")
    confidence_assessment: float = Field(description="Overall confidence 0-1", ge=0, le=1)
    limitations: List[str] = Field(description="Research limitations", max_items=3)


class ResearchOutput(BaseModel):
    """Final output structure"""
    title: str
    summary: str
    key_findings: List[str]
    sources_probed: List[Dict[str, Any]]
    sources_selected: List[Dict[str, Any]]
    detailed_report: Dict[str, Any]
    confidence_score: float
    research_metadata: Dict[str, Any]


# ==================== Main Class ====================

class BasicDeepResearcherB:
    """
    Deep Researcher using Hierarchical Planning with Progressive Refinement

    This implementation emphasizes:
    - Structured planning before execution
    - Quality over quantity in source selection
    - Deep content analysis and extraction
    - Contradiction detection and resolution
    - Progressive knowledge graph building

    Example:
        >>> from abstractcore import create_llm
        >>> from abstractcore.processing import BasicDeepResearcherB
        >>>
        >>> llm = create_llm("openai", model="gpt-4o-mini")
        >>> researcher = BasicDeepResearcherB(llm)
        >>>
        >>> result = researcher.research("How does the EU AI Act compare to China's AI regulations?")
        >>> print(json.dumps(result.dict(), indent=2))
    """

    def __init__(
        self,
        llm: Optional[AbstractCoreInterface] = None,
        max_tokens: int = 32000,
        max_output_tokens: int = 8000,
        timeout: Optional[float] = None,
        max_sources: int = 20,
        quality_threshold: float = 0.3,  # Lowered from 0.7 to prevent all sources being rejected
        extract_full_content: bool = True,
        search_provider: str = "duckduckgo",
        search_api_key: Optional[str] = None,
        temperature: float = 0.1,
        debug: bool = False
    ):
        """
        Initialize the Deep Researcher with Hierarchical Planning

        Args:
            llm: LLM instance (defaults to Ollama if None)
            max_tokens: Maximum context tokens
            max_output_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            max_sources: Maximum sources to collect
            quality_threshold: Minimum quality score for sources (0-1)
            extract_full_content: Whether to extract full page content
            search_provider: "duckduckgo" (default, free) or "serper"
            search_api_key: API key for search provider if needed
            temperature: LLM temperature (0.1 for consistency)
            debug: Enable debug mode
        """
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
                    f"❌ Failed to initialize default model: {e}\n\n"
                    "💡 Please provide a custom LLM instance or install Ollama:\n"
                    "   from abstractcore import create_llm\n"
                    "   llm = create_llm('openai', model='gpt-4o-mini')\n"
                    "   researcher = BasicDeepResearcherB(llm)"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm

        self.max_sources = max_sources
        self.quality_threshold = quality_threshold
        # Safety check: if threshold too high and no sources pass, warn user
        self._warn_no_sources = True
        self.extract_full_content = extract_full_content
        self.search_provider = search_provider
        self.search_api_key = search_api_key
        self.temperature = temperature
        self.debug = debug
        self.retry_strategy = FeedbackRetry(max_attempts=1)  # Reduced from 3 to 1 for speed

        # Initialize helpers
        try:
            self.summarizer = BasicSummarizer(llm=self.llm)
            self.intent_analyzer = BasicIntentAnalyzer(llm=self.llm)
        except Exception as e:
            logger.warning(f"Helper tools initialization failed: {e}")
            self.summarizer = None
            self.intent_analyzer = None

        # Tracking
        self.atomic_questions: Dict[str, AtomicQuestion] = {}
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.sources_probed: List[Dict[str, Any]] = []
        self.sources_selected: List[Dict[str, Any]] = []
        self.seen_urls: Set[str] = set()
        self.content_cache: Dict[str, str] = {}

        logger.info(f"🤖 Initialized BasicDeepResearcherB with {self.llm.provider}/{self.llm.model}")
        logger.info(f"🎯 Strategy: Hierarchical Planning | Quality threshold: {quality_threshold}")

    def research(
        self,
        query: str,
        focus_areas: Optional[List[str]] = None,
        depth: str = "medium"
    ) -> ResearchOutput:
        """
        Conduct deep research using hierarchical planning

        Args:
            query: Research question
            focus_areas: Optional specific areas to focus on
            depth: "shallow", "medium", or "deep"

        Returns:
            ResearchOutput: Structured research report
        """
        start_time = time.time()
        logger.info(f"🔬 Starting research: {query}")

        # Reset tracking
        self.atomic_questions = {}
        self.knowledge_graph = {}
        self.sources_probed = []
        self.sources_selected = []
        self.seen_urls = set()
        self.content_cache = {}

        # Phase 1: Create research plan (skip intent analysis for speed)
        logger.info("🧠 Phase 1: Creating research plan")
        # Skipping intent analysis - saves 60-80 seconds
        # if self.intent_analyzer:
        #     intent = self._analyze_query_intent(query)
        research_plan = self._create_hierarchical_plan(query, focus_areas, depth)

        # Phase 2: Execute plan with progressive refinement
        logger.info("🔍 Phase 2: Executing research plan")
        self._execute_research_plan(research_plan)

        # Phase 3: Build knowledge graph
        logger.info("🕸️  Phase 3: Building knowledge graph")
        self._build_knowledge_graph()

        # Phase 4: Detect and resolve contradictions
        logger.info("⚖️  Phase 4: Detecting and resolving contradictions")
        self._resolve_contradictions()

        # Phase 5: Generate final report
        logger.info("📝 Phase 5: Generating final report")
        final_report = self._generate_final_report(query, research_plan)

        duration = time.time() - start_time

        # Build output
        # Extract findings - handle different dict key formats
        key_findings = []
        for item in final_report.main_findings:
            if isinstance(item, dict):
                # Try different possible keys
                finding = item.get('finding') or item.get('key_finding') or item.get('content') or str(next(iter(item.values())))
                key_findings.append(finding)
            else:
                key_findings.append(str(item))

        output = ResearchOutput(
            title=final_report.title,
            summary=final_report.executive_summary,
            key_findings=key_findings,
            sources_probed=self.sources_probed,
            sources_selected=self.sources_selected,
            detailed_report={
                "main_findings": final_report.main_findings,
                "sections": final_report.detailed_sections,
                "methodology": final_report.methodology,
                "confidence_assessment": final_report.confidence_assessment,
                "limitations": final_report.limitations
            },
            confidence_score=final_report.confidence_assessment,
            research_metadata={
                "strategy": "hierarchical_planning",
                "duration_seconds": round(duration, 2),
                "atomic_questions": len(self.atomic_questions),
                "knowledge_nodes": len(self.knowledge_graph),
                "sources_probed": len(self.sources_probed),
                "sources_selected": len(self.sources_selected),
                "quality_threshold": self.quality_threshold,
                "model_used": f"{self.llm.provider}/{self.llm.model}"
            }
        )

        logger.info(f"✅ Research completed in {duration:.1f}s | {len(self.sources_selected)} sources")
        return output

    def _analyze_query_intent(self, query: str) -> Any:
        """Analyze query intent to guide research strategy"""
        if not self.intent_analyzer:
            return None

        try:
            intent = self.intent_analyzer.analyze_intent(
                query,
                context_type=IntentContext.STANDALONE,
                depth=IntentDepth.UNDERLYING
            )
            logger.info(f"📊 Intent: {intent.primary_intent.intent_type.value} (confidence: {intent.overall_confidence:.2f})")
            return intent
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            return None

    def _create_hierarchical_plan(self, query: str, focus_areas: Optional[List[str]], depth: str) -> ResearchPlanModel:
        """Create structured hierarchical research plan (simplified for speed)"""
        focus_context = ""
        if focus_areas:
            focus_context = f"\n\nFocus Areas: {', '.join(focus_areas)}"

        # Simplified prompt - reduced from detailed instructions to concise format
        prompt = f"""Create a research plan for: {query}
Depth: {depth}{focus_context}

Generate:
1. Research goal (one sentence)
2. 3-4 research categories
3. 5-8 specific questions to answer
4. Question dependencies (optional)

Keep questions focused and independently answerable."""

        response = self.llm.generate(prompt, response_model=ResearchPlanModel, retry_strategy=self.retry_strategy)

        if isinstance(response, ResearchPlanModel):
            plan = response
        elif hasattr(response, 'structured_output'):
            plan = response.structured_output
        else:
            raise ValueError("Failed to create research plan")

        # Create atomic question objects
        for q_data in plan.atomic_questions:
            q_id = q_data.get('id', f"q_{len(self.atomic_questions)}")
            question = AtomicQuestion(
                id=q_id,
                question=q_data.get('question', ''),
                category=q_data.get('category', 'general'),
                priority=q_data.get('priority', 2),
                dependencies=plan.dependencies.get(q_id, [])
            )
            self.atomic_questions[q_id] = question

        logger.info(f"📋 Created plan with {len(self.atomic_questions)} atomic questions")
        return plan

    def _execute_research_plan(self, plan: ResearchPlanModel):
        """Execute research plan in dependency order"""
        # Topological sort for dependency order
        completed = set()
        remaining = set(self.atomic_questions.keys())

        while remaining:
            # Find questions with satisfied dependencies
            ready = []
            for q_id in remaining:
                question = self.atomic_questions[q_id]
                if all(dep in completed for dep in question.dependencies):
                    ready.append(q_id)

            if not ready:
                # No questions ready - break circular dependencies
                logger.warning("Circular dependencies detected, processing remaining questions")
                ready = list(remaining)

            # Process ready questions by priority
            ready.sort(key=lambda qid: self.atomic_questions[qid].priority)

            for q_id in ready[:3]:  # Process up to 3 in parallel
                if len(self.sources_selected) >= self.max_sources:
                    logger.info("📊 Source limit reached")
                    return

                self._research_atomic_question(q_id)
                completed.add(q_id)
                remaining.remove(q_id)

    def _research_atomic_question(self, q_id: str):
        """Research a single atomic question"""
        question = self.atomic_questions[q_id]
        question.status = "researching"

        # Generate queries
        queries = self._generate_queries_for_question(question)

        # Search
        sources = []
        for query in [queries.primary_query] + queries.alternative_queries:
            search_results = self._search(query)
            sources.extend(search_results)

            if len(sources) >= 10:  # Limit per question
                break

        # Assess quality
        high_quality_sources = self._assess_source_quality(sources, question)

        # CRITICAL: If no sources pass quality check, include top sources anyway!
        if len(high_quality_sources) == 0 and len(sources) > 0:
            logger.warning(f"⚠️  No sources passed quality threshold ({self.quality_threshold}), including top 3 anyway to prevent hallucination!")
            high_quality_sources = sources[:3]
            # Mark as lower quality
            for src in high_quality_sources:
                src['quality_score'] = 0.5
                src['note'] = 'Included despite low quality score to prevent hallucination'

        # Extract and analyze content
        for source in high_quality_sources[:5]:  # Top 5 per question
            if len(self.sources_selected) >= self.max_sources:
                break

            content_analysis = self._extract_and_analyze_content(source, question)
            if content_analysis:
                source['content_analysis'] = content_analysis
                question.findings.append(source)
                self.sources_selected.append(source)

                # Update confidence
                question.confidence = min(1.0, question.confidence + 0.2)

        question.status = "completed"
        logger.info(f"✅ Completed: {question.question} ({len(question.findings)} sources)")

    def _generate_queries_for_question(self, question: AtomicQuestion) -> QueriesModel:
        """Generate search queries for atomic question"""
        prompt = f"""Generate search queries for this specific research question:

Question: {question.question}
Category: {question.category}
Priority: {question.priority}

Create:
1. One primary search query (most likely to find good answers)
2. Two alternative query formulations (different angles)

Queries should be:
- Specific and targeted
- Likely to retrieve authoritative sources
- Varied enough to maximize coverage

Return structured queries."""

        response = self.llm.generate(prompt, response_model=QueriesModel, retry_strategy=self.retry_strategy)

        if isinstance(response, QueriesModel):
            return response
        elif hasattr(response, 'structured_output'):
            return response.structured_output
        else:
            # Fallback
            return QueriesModel(
                primary_query=question.question,
                alternative_queries=[question.question, question.question]
            )

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Execute web search"""
        sources = []

        try:
            if self.search_provider == "serper" and self.search_api_key:
                search_result = self._search_serper(query)
            else:
                search_result = web_search(query)

            # Parse results
            sources = self._parse_search_results(search_result, query)

        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")

        return sources

    def _search_serper(self, query: str) -> str:
        """Search using Serper.dev"""
        import requests

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.search_api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": 5}

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        results = [f"🔍 Search results for: '{query}'\n"]

        for i, result in enumerate(data.get('organic', []), 1):
            title = result.get('title', 'No title')
            link = result.get('link', '')
            snippet = result.get('snippet', '')

            results.append(f"\n{i}. {title}")
            results.append(f"   🔗 {link}")
            results.append(f"   📄 {snippet}")

        return "\n".join(results)

    def _parse_search_results(self, search_output: str, query: str) -> List[Dict[str, Any]]:
        """Parse search output into structured sources"""
        sources = []

        url_pattern = r'🔗\s+(https?://[^\s]+)'
        title_pattern = r'\d+\.\s+([^\n]+?)(?:\s+🔗|$)'
        snippet_pattern = r'📄\s+([^\n]+)'

        urls = re.findall(url_pattern, search_output)
        titles = re.findall(title_pattern, search_output)
        snippets = re.findall(snippet_pattern, search_output)

        for i, url in enumerate(urls):
            if url in self.seen_urls:
                continue

            self.seen_urls.add(url)

            source = {
                "url": url,
                "title": titles[i] if i < len(titles) else "Unknown",
                "snippet": snippets[i] if i < len(snippets) else "",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

            sources.append(source)
            self.sources_probed.append(source.copy())

        return sources

    def _assess_source_quality(self, sources: List[Dict[str, Any]], question: AtomicQuestion) -> List[Dict[str, Any]]:
        """Assess source quality and filter"""
        high_quality = []

        for source in sources:
            # Simplified prompt for speed (reduced from detailed quality assessment)
            prompt = f"""Is this source useful for: {question.question}?

Title: {source.get('title', 'Unknown')}
Snippet: {source.get('snippet', 'No snippet')}

Rate relevance (0-1) and say if it should be included."""

            try:
                response = self.llm.generate(prompt, response_model=SourceQualityModel, retry_strategy=self.retry_strategy)

                if isinstance(response, SourceQualityModel):
                    assessment = response
                elif hasattr(response, 'structured_output'):
                    assessment = response.structured_output
                else:
                    continue

                # Calculate overall quality
                quality_score = (
                    assessment.credibility_score * 0.4 +
                    assessment.recency_score * 0.2 +
                    assessment.relevance_score * 0.4
                )

                if assessment.should_include and quality_score >= self.quality_threshold:
                    source['quality_score'] = quality_score
                    source['credibility'] = assessment.credibility_score
                    source['recency'] = assessment.recency_score
                    source['relevance'] = assessment.relevance_score
                    source['authority_indicators'] = assessment.authority_indicators

                    high_quality.append(source)

            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")

        # Sort by quality
        high_quality.sort(key=lambda s: s.get('quality_score', 0), reverse=True)
        return high_quality

    def _extract_and_analyze_content(self, source: Dict[str, Any], question: AtomicQuestion) -> Optional[Dict]:
        """Extract and analyze full content from source"""
        url = source.get('url', '')

        if not self.extract_full_content:
            return None

        # Check cache
        if url in self.content_cache:
            content = self.content_cache[url]
        else:
            try:
                # Fetch full content
                fetch_result = fetch_url(url, render_js=False)

                # Extract text (simplified)
                content = self._extract_text_from_fetch(fetch_result)
                self.content_cache[url] = content

                if not content or len(content) < 100:
                    return None

            except Exception as e:
                logger.warning(f"Content extraction failed for {url}: {e}")
                return None

        # Analyze content
        prompt = f"""Analyze this content for answering: {question.question}

Content (excerpt):
{content[:2000]}

Extract:
1. 3-7 key facts relevant to the question
2. Claims that need verification (up to 3)
3. Confidence level (0-1) in the content
4. Whether this contradicts existing knowledge
5. Details of contradiction if any

Return structured analysis."""

        try:
            response = self.llm.generate(prompt, response_model=ContentAnalysisModel, retry_strategy=self.retry_strategy)

            if isinstance(response, ContentAnalysisModel):
                return {
                    'key_facts': response.key_facts,
                    'claims_to_verify': response.claims_needing_verification,
                    'confidence': response.confidence_level,
                    'contradicts': response.contradicts_existing,
                    'contradiction_details': response.contradiction_details
                }
            elif hasattr(response, 'structured_output'):
                analysis = response.structured_output
                return {
                    'key_facts': analysis.key_facts,
                    'claims_to_verify': analysis.claims_needing_verification,
                    'confidence': analysis.confidence_level,
                    'contradicts': analysis.contradicts_existing,
                    'contradiction_details': analysis.contradiction_details
                }

        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")

        return None

    def _extract_text_from_fetch(self, fetch_result: str) -> str:
        """Extract clean text from fetch_url result"""
        # Simple extraction - look for text content markers
        lines = fetch_result.split('\n')
        text_lines = []

        in_content = False
        for line in lines:
            if '📄 Text Content' in line or '📄 Content' in line:
                in_content = True
                continue
            if in_content and not line.startswith('📊'):
                text_lines.append(line)

        return '\n'.join(text_lines)

    def _build_knowledge_graph(self):
        """Build knowledge graph from findings"""
        # Group findings by concept
        concepts = defaultdict(list)

        for question in self.atomic_questions.values():
            for finding in question.findings:
                if 'content_analysis' in finding:
                    analysis = finding['content_analysis']
                    for fact in analysis.get('key_facts', []):
                        # Simple concept extraction (first few words)
                        concept_key = ' '.join(fact.split()[:3]).lower()
                        concepts[concept_key].append({
                            'fact': fact,
                            'source': finding.get('url', ''),
                            'confidence': analysis.get('confidence', 0.5)
                        })

        # Create knowledge nodes
        for i, (concept_key, items) in enumerate(concepts.items()):
            node_id = f"node_{i}"
            facts = [item['fact'] for item in items]
            sources = list(set(item['source'] for item in items))
            avg_confidence = sum(item['confidence'] for item in items) / len(items)

            node = KnowledgeNode(
                id=node_id,
                concept=concept_key,
                facts=facts,
                sources=sources,
                confidence=avg_confidence
            )

            self.knowledge_graph[node_id] = node

        logger.info(f"🕸️  Built knowledge graph with {len(self.knowledge_graph)} nodes")

    def _resolve_contradictions(self):
        """Detect and resolve contradictions"""
        contradictions_found = 0

        for question in self.atomic_questions.values():
            for finding in question.findings:
                if 'content_analysis' in finding:
                    analysis = finding['content_analysis']
                    if analysis.get('contradicts', False):
                        contradictions_found += 1
                        # Log contradiction for report
                        logger.warning(f"⚠️  Contradiction detected: {analysis.get('contradiction_details', 'Unknown')}")

        logger.info(f"⚖️  Detected {contradictions_found} contradictions")

    def _generate_final_report(self, original_query: str, plan: ResearchPlanModel) -> FinalReportModel:
        """Generate comprehensive final report"""
        # Aggregate all findings
        all_facts = []
        all_sources = []

        for question in self.atomic_questions.values():
            for finding in question.findings:
                all_sources.append(f"- {finding.get('title', 'Unknown')} (Quality: {finding.get('quality_score', 0):.2f})")

                if 'content_analysis' in finding:
                    all_facts.extend(finding['content_analysis'].get('key_facts', []))

        sources_summary = "\n".join(all_sources[:15])
        facts_summary = "\n".join([f"- {fact}" for fact in all_facts[:30]])

        prompt = f"""Generate a comprehensive research report.

Original Question: {original_query}
Research Goal: {plan.research_goal}

Sources Analyzed ({len(self.sources_selected)} high-quality sources):
{sources_summary}

Key Facts Discovered:
{facts_summary}

Create a detailed report with:
1. Engaging title
2. Executive summary (2-3 paragraphs)
3. 5-10 main findings with supporting evidence
4. 3-5 detailed sections with analysis
5. Methodology description
6. Overall confidence assessment (0-1)
7. 0-3 limitations

Ensure findings are well-supported by evidence."""

        response = self.llm.generate(prompt, response_model=FinalReportModel, retry_strategy=self.retry_strategy)

        if isinstance(response, FinalReportModel):
            return response
        elif hasattr(response, 'structured_output'):
            return response.structured_output
        else:
            raise ValueError("Failed to generate report")
