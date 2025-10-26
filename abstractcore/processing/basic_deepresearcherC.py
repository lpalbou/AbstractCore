"""
Basic Deep Researcher C - Adaptive ReAct Web Research Agent

This implementation follows the ReAct (Reasoning and Acting) paradigm with adaptive
planning and proper grounding to prevent hallucinations.

Architecture:
1. Understanding Phase: Analyze query and build initial context through exploratory searches
2. Adaptive Planning: Create dynamic research plan that evolves based on findings
3. ReAct Loop: Iterative Reason → Act → Observe → Adapt cycles
4. Content Fetching: Actual web page analysis (not just snippets)
5. Grounding System: Every claim tied to verified sources

Key Features:
- Adaptive planning that evolves based on discoveries
- True ReAct implementation with explicit reasoning traces
- Full content extraction from web pages
- Anti-hallucination through mandatory source grounding
- Breadth-first exploration with depth-first refinement
- Handles "no information found" gracefully
"""

import json
import time
import re
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.interface import AbstractCoreInterface
from ..core.factory import create_llm
from ..utils.structured_logging import get_logger
from ..tools.common_tools import web_search, fetch_url

logger = get_logger(__name__)


# ==================== Data Models ====================

@dataclass
class ResearchContext:
    """Current understanding of the research query"""
    query: str
    key_concepts: List[str] = field(default_factory=list)
    research_dimensions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    query_type: str = "exploratory"  # factual, analytical, exploratory, comparative
    confidence: float = 0.0


@dataclass
class ResearchTask:
    """Single research task in the adaptive plan"""
    id: str
    dimension: str
    queries: List[str]
    priority: float
    status: str = "pending"  # pending, active, completed, abandoned
    findings: List[Dict[str, Any]] = field(default_factory=list)
    urls_explored: Set[str] = field(default_factory=set)
    confidence: float = 0.0


@dataclass
class ReActStep:
    """Single step in the ReAct loop"""
    iteration: int
    reasoning: str
    action: str
    queries: List[str]
    urls_fetched: List[str]
    observation: str
    adaptation: str
    new_gaps: List[str]
    timestamp: str


@dataclass
class SourceEvidence:
    """Evidence extracted from a source with grounding"""
    url: str
    title: str
    excerpt: str  # Actual text from source
    relevance_score: float
    credibility_score: float
    key_facts: List[str]
    timestamp: str


# Simple Pydantic models with minimal constraints
class UnderstandingModel(BaseModel):
    """Understanding of the query - SIMPLE SCHEMA"""
    concepts: List[str] = Field(description="Key concepts")
    dimensions: List[str] = Field(description="Research dimensions")
    gaps: List[str] = Field(description="Knowledge gaps")


class SearchQueriesModel(BaseModel):
    """Search queries - SIMPLE SCHEMA"""
    queries: List[str] = Field(description="Search queries")


class SourceRelevanceModel(BaseModel):
    """Source relevance assessment - SIMPLE SCHEMA"""
    is_relevant: bool = Field(description="Is source relevant")
    relevance_score: float = Field(description="0-1 relevance")
    credibility_score: float = Field(description="0-1 credibility")
    facts: List[str] = Field(description="Key facts")


class FindingWithSource(BaseModel):
    """Finding with source attribution - SIMPLE SCHEMA"""
    finding: str = Field(description="The key finding statement")
    evidence_ids: List[int] = Field(description="Evidence piece numbers that support this finding")


class GapWithSource(BaseModel):
    """Knowledge gap with source attribution - SIMPLE SCHEMA"""
    gap: str = Field(description="The knowledge gap or unanswered question")
    evidence_ids: List[int] = Field(description="Evidence piece numbers that raised this gap (optional, can be empty)")


class SynthesisModel(BaseModel):
    """Final synthesis - SIMPLE SCHEMA"""
    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary")
    findings_with_sources: List[FindingWithSource] = Field(description="Key findings with source attribution")
    gaps_with_sources: List[GapWithSource] = Field(description="Knowledge gaps with source attribution")
    confidence: float = Field(description="Overall confidence")


class ResearchOutput(BaseModel):
    """Final research output"""
    title: str
    summary: str
    key_findings: List[str]
    sources_selected: List[Dict[str, Any]]
    research_metadata: Dict[str, Any]
    knowledge_gaps: List[str]
    confidence_score: float


# ==================== Main Class ====================

class BasicDeepResearcherC:
    """
    Deep Researcher using Adaptive ReAct Strategy

    This implementation emphasizes:
    - Adaptive planning that evolves based on findings
    - True ReAct loops with explicit reasoning
    - Full web content fetching (not just snippets)
    - Mandatory source grounding to prevent hallucination
    - Graceful handling of "no information found"

    Example:
        >>> from abstractcore import create_llm
        >>> from abstractcore.processing import BasicDeepResearcherC
        >>>
        >>> llm = create_llm("lmstudio", model="qwen/qwen3-30b-a3b-2507")
        >>> researcher = BasicDeepResearcherC(llm, max_sources=30)
        >>>
        >>> result = researcher.research("Laurent-Philippe Albou")
        >>> print(json.dumps(result.dict(), indent=2))
    """

    def __init__(
        self,
        llm: Optional[AbstractCoreInterface] = None,
        max_sources: int = 30,
        max_urls_to_probe: int = 60,
        max_iterations: int = 25,
        fetch_timeout: int = 10,
        enable_breadth: bool = True,
        enable_depth: bool = True,
        grounding_threshold: float = 0.7,
        temperature: float = 0.2,
        debug: bool = False
    ):
        """
        Initialize the Adaptive ReAct Deep Researcher

        Args:
            llm: LLM instance (defaults to LMStudio if None)
            max_sources: Target number of high-quality sources to include
            max_urls_to_probe: Maximum URLs to explore (filters to max_sources)
            max_iterations: Maximum ReAct loop iterations
            fetch_timeout: Timeout for fetching individual URLs (seconds)
            enable_breadth: Explore multiple research dimensions
            enable_depth: Deep dive on promising leads
            grounding_threshold: Minimum relevance score for inclusion (0-1)
            temperature: LLM temperature for research
            debug: Enable detailed execution traces
        """
        if llm is None:
            try:
                self.llm = create_llm(
                    "lmstudio",
                    model="qwen/qwen3-30b-a3b-2507",
                    temperature=temperature,
                    timeout=120
                )
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default model: {e}\n\n"
                    "💡 Please provide a custom LLM instance:\n"
                    "   from abstractcore import create_llm\n"
                    "   llm = create_llm('lmstudio', model='qwen/qwen3-30b-a3b-2507')\n"
                    "   researcher = BasicDeepResearcherC(llm)"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm

        self.max_sources = max_sources
        self.max_urls_to_probe = max_urls_to_probe
        self.max_iterations = max_iterations
        self.fetch_timeout = fetch_timeout
        self.enable_breadth = enable_breadth
        self.enable_depth = enable_depth
        self.grounding_threshold = grounding_threshold
        self.temperature = temperature
        self.debug = debug

        # State tracking
        self.context: Optional[ResearchContext] = None
        self.tasks: List[ResearchTask] = []
        self.react_steps: List[ReActStep] = []
        self.evidence: List[SourceEvidence] = []
        self.seen_urls: Set[str] = set()
        self.active_gaps: Dict[str, Dict[str, Any]] = {}  # gap_text -> {status, source_urls, related_findings}
        self.finding_to_source: Dict[str, str] = {}  # finding_text -> source_url (for traceability)

        logger.info(f"🤖 Initialized BasicDeepResearcherC with {self.llm.provider}/{self.llm.model}")
        logger.info(f"🎯 Strategy: Adaptive ReAct | Max iterations: {max_iterations} | Max sources: {max_sources}")

    def research(
        self,
        query: str,
        focus_areas: Optional[List[str]] = None
    ) -> ResearchOutput:
        """
        Conduct deep research using Adaptive ReAct strategy

        Args:
            query: Research question or topic
            focus_areas: Optional specific areas to focus on

        Returns:
            ResearchOutput: Comprehensive research report with grounded findings
        """
        start_time = time.time()
        logger.info(f"🔬 Starting adaptive ReAct research: {query}")

        # Reset state
        self.context = None
        self.tasks = []
        self.react_steps = []
        self.evidence = []
        self.seen_urls = set()
        self.active_gaps = {}
        self.finding_to_source = {}

        # Phase 1: Understand the query
        logger.info("🧠 Phase 1: Understanding query...")
        self.context = self._understand_query(query, focus_areas)

        # Phase 2: Build adaptive plan
        logger.info("📋 Phase 2: Building adaptive research plan...")
        self._build_adaptive_plan(self.context)

        # Phase 3: ReAct loop
        logger.info("🔄 Phase 3: Executing ReAct loop...")
        self._react_loop()

        # Phase 4: Synthesize with grounding
        logger.info("📝 Phase 4: Synthesizing grounded report...")
        final_report = self._synthesize_with_grounding()

        duration = time.time() - start_time

        # Collect only unresolved gaps for final report
        unresolved_gaps = [
            gap_text for gap_text, gap_data in self.active_gaps.items()
            if gap_data["status"] == "active"
        ]

        # Extract gap text from gaps_with_sources
        synthesis_gaps = [g.gap for g in final_report.gaps_with_sources]

        # Merge synthesis gaps with unresolved active gaps (already have source attribution in active_gaps)
        all_final_gaps = list(set(unresolved_gaps + synthesis_gaps))

        # Extract findings from findings_with_sources structure
        key_findings_list = [f.finding for f in final_report.findings_with_sources]

        # Build output
        output = ResearchOutput(
            title=final_report.title,
            summary=final_report.summary,
            key_findings=key_findings_list,
            sources_selected=[
                {
                    "url": ev.url,
                    "title": ev.title,
                    "relevance": ev.relevance_score,
                    "credibility": ev.credibility_score,
                    "excerpt": ev.excerpt[:200] + "..." if len(ev.excerpt) > 200 else ev.excerpt
                }
                for ev in self.evidence
            ],
            knowledge_gaps=all_final_gaps,  # Only unresolved gaps
            confidence_score=final_report.confidence,
            research_metadata={
                "strategy": "adaptive_react",
                "duration_seconds": round(duration, 2),
                "react_iterations": len(self.react_steps),
                "research_tasks": len(self.tasks),
                "evidence_pieces": len(self.evidence),
                "urls_explored": len(self.seen_urls),
                "gaps_resolved": sum(1 for gap_data in self.active_gaps.values() if gap_data["status"] == "resolved"),
                "gaps_remaining": len(unresolved_gaps),
                "model_used": f"{self.llm.provider}/{self.llm.model}"
            }
        )

        logger.info(f"✅ Research completed in {duration:.1f}s | {len(self.evidence)} sources | {final_report.confidence:.2f} confidence")
        return output

    def _understand_query(self, query: str, focus_areas: Optional[List[str]]) -> ResearchContext:
        """
        Phase 1: Understand the query through LLM analysis and exploratory searches
        """
        # Step 1: LLM analysis of query
        focus_context = ""
        if focus_areas:
            focus_context = f"\n\nFocus particularly on: {', '.join(focus_areas)}"

        prompt = f"""Analyze this research query and extract key information:

Query: {query}{focus_context}

Provide:
1. Key concepts/entities to research
2. Specific research dimensions to explore
3. Initial knowledge gaps to address

Return ONLY concepts, dimensions, and gaps lists. Keep each item concise (1-3 words)."""

        try:
            # Try structured output first
            response = self.llm.generate(prompt, response_model=UnderstandingModel)
            if isinstance(response, UnderstandingModel):
                understanding = response
            elif hasattr(response, 'content'):
                # Fallback: parse from text
                understanding = self._parse_understanding(response.content)
            else:
                understanding = UnderstandingModel(
                    concepts=[query],
                    dimensions=["general information"],
                    gaps=["basic facts"]
                )
        except Exception as e:
            logger.warning(f"Structured understanding failed, using fallback: {e}")
            understanding = UnderstandingModel(
                concepts=[query],
                dimensions=["general information", "background", "recent developments"],
                gaps=["basic facts", "detailed information"]
            )

        # Step 2: Exploratory searches to validate
        logger.info("🔍 Conducting exploratory searches...")
        exploratory_queries = [
            f"{query}",
            f"{query} overview",
            f"{query} information"
        ]

        search_results_found = False
        for q in exploratory_queries[:2]:  # Just 2 exploratory searches
            try:
                search_result = web_search(q, num_results=5)
                if search_result and "⚠️ Limited results" not in search_result:
                    search_results_found = True
                    break
            except Exception as e:
                logger.warning(f"Exploratory search failed: {e}")

        # Build context
        context = ResearchContext(
            query=query,
            key_concepts=understanding.concepts[:5],  # Limit to 5
            research_dimensions=understanding.dimensions[:8] if self.enable_breadth else understanding.dimensions[:3],
            knowledge_gaps=understanding.gaps[:5],
            query_type="exploratory" if search_results_found else "unknown",
            confidence=0.5 if search_results_found else 0.1
        )

        if self.debug:
            logger.debug(f"Context: {context}")

        return context

    def _parse_understanding(self, text: str) -> UnderstandingModel:
        """Parse understanding from text fallback"""
        # Simple text parsing
        concepts = []
        dimensions = []
        gaps = []

        # Extract lists from text
        if "concepts:" in text.lower():
            concept_section = text.lower().split("concepts:")[1].split("\n\n")[0]
            concepts = [c.strip().strip("-•").strip() for c in concept_section.split("\n") if c.strip()]

        if "dimensions:" in text.lower():
            dimension_section = text.lower().split("dimensions:")[1].split("\n\n")[0]
            dimensions = [d.strip().strip("-•").strip() for d in dimension_section.split("\n") if d.strip()]

        if "gaps:" in text.lower():
            gap_section = text.lower().split("gaps:")[1].split("\n\n")[0]
            gaps = [g.strip().strip("-•").strip() for g in gap_section.split("\n") if g.strip()]

        return UnderstandingModel(
            concepts=concepts[:5] or ["general information"],
            dimensions=dimensions[:8] or ["overview"],
            gaps=gaps[:5] or ["details"]
        )

    def _build_adaptive_plan(self, context: ResearchContext):
        """
        Phase 2: Build adaptive research plan from context
        """
        # Create research tasks for each dimension
        for idx, dimension in enumerate(context.research_dimensions):
            task = ResearchTask(
                id=f"task_{idx}",
                dimension=dimension,
                queries=[],
                priority=1.0 - (idx * 0.1)  # Decreasing priority
            )
            self.tasks.append(task)

        logger.info(f"📋 Created {len(self.tasks)} research tasks")
        if self.debug:
            for task in self.tasks:
                logger.debug(f"  - {task.dimension} (priority: {task.priority:.2f})")

    def _react_loop(self):
        """
        Phase 3: Execute ReAct (Reason-Act-Observe-Adapt) loop with phased strategy

        PHASE 1 (iter 1-3, sources < 10): EXPLORATION - Cast wide net, find diverse sources
        PHASE 2 (iter 3-5, sources >= 10): DEEPENING - Use findings to go deeper
        PHASE 3 (iter 5+): VALIDATION - Critically evaluate gaps

        Continues until we reach max_sources OR all tasks completed OR max_iterations
        """
        iteration = 0
        max_total_iterations = self.max_iterations * 2  # Allow more iterations to reach source target
        min_sources_threshold = min(10, self.max_sources)  # Critical threshold

        while iteration < max_total_iterations:
            iteration += 1

            # Determine research phase
            phase = self._determine_research_phase(iteration, len(self.evidence))

            # Check if we have enough sources - but be AGGRESSIVE if below threshold
            if len(self.evidence) >= self.max_sources:
                logger.info(f"✅ Reached target of {self.max_sources} sources")
                break

            if len(self.evidence) < min_sources_threshold:
                logger.info(f"⚡ EXPLORATION MODE: {len(self.evidence)}/{min_sources_threshold} sources - seeking more angles")

            logger.info(f"🔄 ReAct iteration {iteration} ({phase}) | Sources: {len(self.evidence)}/{self.max_sources}")

            # REASON: Analyze current state and decide next action
            reasoning = self._reason_about_state(iteration)

            # ACT: Execute searches and fetch content
            active_task = self._select_next_task()
            if not active_task:
                # If no tasks but sources < threshold, generate exploratory queries
                if len(self.evidence) < min_sources_threshold:
                    logger.info(f"⚡ No tasks remain but only {len(self.evidence)} sources - exploring new angles")
                    active_task = self._generate_exploratory_task(iteration, phase)
                else:
                    logger.info("✅ All tasks completed or no promising tasks remain")
                    break

            queries = self._generate_queries_for_task(active_task, phase=phase, iteration=iteration)
            active_task.queries = queries
            active_task.status = "active"

            urls_fetched, new_evidence = self._act_search_and_fetch(queries, active_task)

            # OBSERVE: Process findings and validate
            observation = self._observe_findings(new_evidence)

            # ADAPT: Update plan based on observations
            adaptation, new_gaps = self._adapt_plan(active_task, new_evidence)

            # Record step
            step = ReActStep(
                iteration=iteration,
                reasoning=reasoning,
                action=f"Researching: {active_task.dimension}",
                queries=queries,
                urls_fetched=urls_fetched,
                observation=observation,
                adaptation=adaptation,
                new_gaps=new_gaps,
                timestamp=datetime.now().isoformat()
            )
            self.react_steps.append(step)

            active_task.status = "completed"

            # Check convergence (80% tasks done AND decent source count)
            if self._check_convergence() and len(self.evidence) >= self.max_sources * 0.5:
                logger.info(f"✅ Research converged at iteration {iteration}")
                break

    def _reason_about_state(self, iteration: int) -> str:
        """REASON: Analyze current state"""
        completed = sum(1 for t in self.tasks if t.status == "completed")
        pending = sum(1 for t in self.tasks if t.status == "pending")

        reasoning = f"Iteration {iteration + 1}: {completed} tasks completed, {pending} pending. "
        reasoning += f"Found {len(self.evidence)} evidence pieces so far."

        if self.debug:
            logger.debug(f"REASON: {reasoning}")

        return reasoning

    def _determine_research_phase(self, iteration: int, source_count: int) -> str:
        """
        Determine current research phase based on iteration and source count

        EXPLORATION (iter 1-3 OR sources < 10): Cast wide net
        DEEPENING (iter 3-5 AND sources >= 10): Go deeper on findings
        VALIDATION (iter 5+): Critically evaluate gaps
        """
        min_threshold = min(10, self.max_sources)

        if source_count < min_threshold or iteration <= 3:
            return "EXPLORATION"
        elif iteration <= 5:
            return "DEEPENING"
        else:
            return "VALIDATION"

    def _select_next_task(self) -> Optional[ResearchTask]:
        """Select next task to execute based on priority"""
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        if not pending_tasks:
            return None

        # Sort by priority
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        return pending_tasks[0]

    def _generate_exploratory_task(self, iteration: int, phase: str) -> ResearchTask:
        """
        Generate a new exploratory task when existing tasks are exhausted but source count is low

        This helps find unexplored angles and diverse sources
        """
        # Analyze what we've covered so far
        covered_topics = set()
        for task in self.tasks:
            covered_topics.add(task.dimension.lower())

        for ev in self.evidence:
            for fact in ev.key_facts:
                covered_topics.update(fact.lower().split()[:5])

        # Create exploratory task
        task = ResearchTask(
            id=f"exploratory_{iteration}",
            dimension=f"unexplored angles and perspectives",
            queries=[],
            priority=0.9  # High priority to reach source threshold
        )

        logger.info(f"📐 Generated exploratory task to find new angles (iteration {iteration})")
        return task

    def _get_phase_instructions(self, phase: str, current_sources: int, target_sources: int) -> str:
        """Get phase-specific query generation instructions"""

        if phase == "EXPLORATION":
            return f"""PRIORITY: Find DIVERSE sources quickly! ({current_sources}/{target_sources} sources so far)

Your queries should:
1. CAST A WIDE NET - explore different angles and perspectives
2. Seek VARIETY - academic, news, technical docs, expert opinions
3. Find AUTHORITATIVE sources - credible, well-cited content
4. Avoid repetition - try NEW search angles if previous ones didn't yield sources

NOTE: We need at least 10 sources minimum. Be AGGRESSIVE in finding diverse, credible sources."""

        elif phase == "DEEPENING":
            return f"""PRIORITY: DEEPEN understanding using multi-hop reasoning ({current_sources}/{target_sources} sources)

Your queries should:
1. BUILD ON existing findings - use discoveries to find related topics
2. Follow CITATION CHAINS - "who cites this?", "what does this reference?"
3. Explore IMPLICATIONS - "how does this apply to X?", "what does this enable?"
4. Find SPECIALIZED sources - deep dives into specific aspects

NOTE: We have enough sources, now go DEEPER into what we've found."""

        elif phase == "VALIDATION":
            return f"""PRIORITY: VALIDATE gaps and VERIFY findings ({current_sources}/{target_sources} sources)

Your queries should:
1. CRITICALLY EVALUATE gaps - are they real questions or based on false assumptions?
2. CROSS-REFERENCE findings - do multiple sources agree?
3. Test gap VALIDITY - "does X actually exist?", "is this question answerable?"
4. Seek CONFLICTING evidence - find counter-arguments or corrections

NOTE: Some gaps may not be real questions. Validate whether they're worth pursuing."""

        return ""

    def _generate_queries_for_task(self, task: ResearchTask, phase: str = "EXPLORATION", iteration: int = 1) -> List[str]:
        """
        Generate search queries for a task using phase-aware strategy

        EXPLORATION: Broad, diverse queries to find many sources
        DEEPENING: Focused queries building on existing findings
        VALIDATION: Critical queries to evaluate gap validity
        """

        # Collect ONLY ACTIVE (unresolved) knowledge gaps WITH their source URLs
        active_gaps_with_sources = [
            (gap_text, gap_data["source_urls"], gap_data["related_findings"])
            for gap_text, gap_data in self.active_gaps.items()
            if gap_data["status"] == "active"
        ]

        # Collect key findings so far WITH their source URLs for refinement
        existing_findings_with_sources = []
        for evidence in self.evidence[:10]:  # Top 10 findings
            if evidence.key_facts:
                for fact in evidence.key_facts[:1]:  # Top fact from each source
                    existing_findings_with_sources.append((fact, evidence.url))

        # Build phase-specific context and instructions
        phase_instructions = self._get_phase_instructions(phase, len(self.evidence), self.max_sources)

        # Build context sections based on phase priority
        gaps_context = ""
        findings_context = ""

        if phase == "EXPLORATION":
            # EXPLORATION: Priority is diverse angles, gaps secondary
            if task.dimension and "unexplored" not in task.dimension:
                gaps_context = f"\n\nCurrent focus: {task.dimension}"
            if active_gaps_with_sources:
                top_gaps = active_gaps_with_sources[:3]  # Just top 3 to maintain diversity
                gaps_context += f"\n\nAreas needing coverage:\n" + "\n".join(f"- {gap[0]}" for gap in top_gaps)

        elif phase == "DEEPENING":
            # DEEPENING: Priority is multi-hop reasoning from findings
            # CRITICAL: Include source URLs to start from!
            if existing_findings_with_sources:
                findings_context = f"\n\nDeepen understanding by building on these findings:\n"
                for finding, source_url in existing_findings_with_sources[:3]:
                    findings_context += f"- {finding[:100]}\n  SOURCE: {source_url}\n  ACTION: Use this source as starting point for deeper exploration\n"

            if active_gaps_with_sources:
                gaps_context = f"\n\nRemaining questions to address:\n"
                for gap_text, source_urls, related_findings in active_gaps_with_sources[:3]:
                    gaps_context += f"- {gap_text}\n"
                    if source_urls:
                        gaps_context += f"  RELATED SOURCES: {', '.join(source_urls[:2])}\n"

        elif phase == "VALIDATION":
            # VALIDATION: Priority is critical evaluation of gaps
            # CRITICAL: Include source URLs for verification!
            if active_gaps_with_sources:
                gaps_context = f"\n\nCritically evaluate these gaps (are they real questions or false assumptions?):\n"
                for gap_text, source_urls, related_findings in active_gaps_with_sources[:5]:
                    gaps_context += f"- {gap_text}\n"
                    if source_urls:
                        gaps_context += f"  START FROM SOURCES: {', '.join(source_urls[:2])}\n"
                        gaps_context += f"  VERIFY: Can these sources answer this? Or is the gap invalid?\n"
                    else:
                        gaps_context += f"  NO SOURCES YET: Search to validate if this gap is real\n"

            if existing_findings_with_sources:
                findings_context = f"\n\nVerify and cross-reference these findings:\n"
                for finding, source_url in existing_findings_with_sources[:2]:
                    findings_context += f"- {finding[:80]}\n  FROM: {source_url}\n  VERIFY: Find conflicting or confirming evidence\n"

        prompt = f"""Generate 2-3 specific search queries for this research task:

Query context: {self.context.query}
Research dimension: {task.dimension}{gaps_context}{findings_context}

CURRENT RESEARCH PHASE: {phase}
{phase_instructions}

Return ONLY a list of query strings, one per line."""

        try:
            response = self.llm.generate(prompt, response_model=SearchQueriesModel)
            if isinstance(response, SearchQueriesModel):
                return response.queries[:3]
            elif hasattr(response, 'content'):
                # Parse from text
                lines = [l.strip() for l in response.content.split("\n") if l.strip() and not l.strip().startswith("#")]
                return lines[:3]
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")

        # Fallback: construct simple queries
        return [
            f"{self.context.query} {task.dimension}",
            f"{task.dimension} {self.context.query}"
        ]

    def _act_search_and_fetch(self, queries: List[str], task: ResearchTask) -> tuple[List[str], List[SourceEvidence]]:
        """ACT: Execute searches and fetch content"""
        urls_fetched = []
        new_evidence = []

        # Calculate how many sources we still need
        sources_needed = self.max_sources - len(self.evidence)
        urls_to_fetch_per_query = max(5, min(10, sources_needed // len(queries)))

        for query in queries:
            # Search
            try:
                # Request more results to have a better pool
                search_result = web_search(query, num_results=10)
                urls = self._extract_urls_from_search(search_result)

                # Fetch and analyze promising URLs - be more aggressive
                urls_fetched_this_query = 0
                for url in urls:
                    if url in self.seen_urls:
                        continue
                    if urls_fetched_this_query >= urls_to_fetch_per_query:
                        break
                    if len(self.evidence) >= self.max_sources:
                        return urls_fetched, new_evidence

                    self.seen_urls.add(url)
                    urls_fetched.append(url)
                    urls_fetched_this_query += 1

                    # Fetch actual content
                    evidence = self._fetch_and_analyze(url, task)
                    if evidence:
                        new_evidence.append(evidence)
                        self.evidence.append(evidence)

                        logger.info(f"📊 Progress: {len(self.evidence)}/{self.max_sources} sources collected")

                        if len(self.evidence) >= self.max_sources:
                            return urls_fetched, new_evidence

            except Exception as e:
                logger.warning(f"Search/fetch failed for '{query}': {e}")

        return urls_fetched, new_evidence

    def _extract_urls_from_search(self, search_result: str) -> List[str]:
        """Extract URLs from search result text"""
        urls = []

        # Find URLs marked with 🔗
        url_pattern = r'🔗\s+(https?://[^\s]+)'
        matches = re.findall(url_pattern, search_result)
        urls.extend(matches)

        return urls[:10]  # Limit to 10 URLs per search

    def _fetch_and_analyze(self, url: str, task: ResearchTask) -> Optional[SourceEvidence]:
        """Fetch URL content and analyze for relevance"""
        try:
            # Fetch content
            content = fetch_url(url, timeout=self.fetch_timeout)

            if not content or "Error" in content[:100]:
                return None

            # Extract title
            title = "Unknown"
            if "<title>" in content.lower():
                title_match = re.search(r'<title>(.*?)</title>', content, re.I | re.S)
                if title_match:
                    title = title_match.group(1).strip()[:200]

            # Analyze relevance
            prompt = f"""Analyze this web content for relevance to the research query.

Research query: {self.context.query}
Research dimension: {task.dimension}

Content excerpt:
{content[:3000]}

Assess:
1. is_relevant: Is this content relevant to our research?
2. relevance_score: 0-1 how relevant
3. credibility_score: 0-1 how credible/authoritative
4. facts: List 2-3 key facts IF relevant, empty list otherwise

CRITICAL RULES FOR EXTRACTING FACTS:
- Extract ONLY what is EXPLICITLY stated in the content
- DO NOT infer roles, positions, or actions from indirect evidence
- If content shows person commenting/sharing → say "X commented on/shared Y"
- If content shows person authored/created → say "X authored/created Y"
- If content is person's official profile/bio → state what it says
- Use cautious language ("appears to", "is associated with") if uncertain
- DO NOT assume authorship/creation from discussion/commentary

Examples of CORRECT extraction:
✅ "X shared a LinkedIn post about Cellosaurus" (if they shared)
✅ "X posted about ChatGPT usage in Reactome" (if they posted)
✅ "X is listed as author on paper Y" (if author list shows this)
✅ "X's profile states they work at Z" (if official bio says this)

Examples of WRONG extraction:
❌ "X is lead developer of Cellosaurus" (if they only commented)
❌ "X integrated ChatGPT into Reactome" (if they only shared article)

Return assessment."""

            response = self.llm.generate(prompt, response_model=SourceRelevanceModel)

            if isinstance(response, SourceRelevanceModel):
                assessment = response
            elif hasattr(response, 'content'):
                # Fallback parsing
                assessment = SourceRelevanceModel(
                    is_relevant="relevant" in response.content.lower() or "yes" in response.content.lower(),
                    relevance_score=0.5,
                    credibility_score=0.5,
                    facts=[]
                )
            else:
                return None

            # Only keep if meets threshold
            if not assessment.is_relevant or assessment.relevance_score < self.grounding_threshold:
                return None

            # Extract relevant excerpt
            excerpt = content[:1000]

            evidence = SourceEvidence(
                url=url,
                title=title,
                excerpt=excerpt,
                relevance_score=assessment.relevance_score,
                credibility_score=assessment.credibility_score,
                key_facts=assessment.facts,
                timestamp=datetime.now().isoformat()
            )

            # CRITICAL: Track finding->source mapping for traceability
            for fact in assessment.facts:
                self.finding_to_source[fact] = url

            logger.info(f"✅ Evidence: {title[:50]}... (rel: {assessment.relevance_score:.2f})")

            return evidence

        except Exception as e:
            logger.warning(f"Failed to fetch/analyze {url}: {e}")
            return None

    def _observe_findings(self, new_evidence: List[SourceEvidence]) -> str:
        """OBSERVE: Summarize new findings"""
        if not new_evidence:
            return "No relevant evidence found for this dimension."

        observation = f"Found {len(new_evidence)} relevant sources. "
        total_facts = sum(len(e.key_facts) for e in new_evidence)
        observation += f"Extracted {total_facts} key facts."

        if self.debug:
            logger.debug(f"OBSERVE: {observation}")

        return observation

    def _adapt_plan(self, task: ResearchTask, new_evidence: List[SourceEvidence]) -> tuple[str, List[str]]:
        """ADAPT: Update plan based on observations and identify remaining gaps"""
        task.findings = [{"evidence": e} for e in new_evidence]
        task.confidence = sum(e.relevance_score for e in new_evidence) / max(len(new_evidence), 1)

        adaptation = f"Task '{task.dimension}' completed with {len(new_evidence)} sources (confidence: {task.confidence:.2f})."

        # Check if new evidence resolves any existing gaps
        self._check_gap_resolution(new_evidence, task)

        # Identify new gaps based on what we learned
        new_gaps = []

        if not new_evidence:
            # No evidence found - could mean:
            # 1. Information doesn't exist online
            # 2. Need better search queries
            # 3. Topic is legitimately unknown/unverifiable
            gap_text = f"No verifiable information found for: {task.dimension}"
            new_gaps.append(gap_text)
            self.active_gaps[gap_text] = {
                "status": "active",
                "source_urls": [],  # No sources led to this gap
                "related_findings": [],
                "dimension": task.dimension
            }

        elif task.confidence < 0.6:
            # Low confidence - need more investigation
            # Extract what's missing from the evidence we do have
            found_topics = set()
            source_urls = []
            related_findings = []

            for ev in new_evidence:
                source_urls.append(ev.url)
                for fact in ev.key_facts:
                    # Simple keyword extraction
                    found_topics.update(fact.lower().split()[:5])
                    related_findings.append(fact)

            gap_text = f"Insufficient detail on: {task.dimension} (needs refinement)"
            new_gaps.append(gap_text)
            self.active_gaps[gap_text] = {
                "status": "active",
                "source_urls": source_urls,  # Sources that partially covered this
                "related_findings": related_findings[:3],  # Top 3 related findings
                "dimension": task.dimension
            }

        elif task.confidence < 0.8:
            # Medium confidence - we have something but could be better
            # This is actually OK - not adding to gaps
            # But we might want to refine if enable_depth is True
            if self.enable_depth and len(self.evidence) < self.max_sources * 0.7:
                source_urls = [ev.url for ev in new_evidence]
                related_findings = []
                for ev in new_evidence:
                    related_findings.extend(ev.key_facts[:2])

                gap_text = f"Could deepen understanding of: {task.dimension}"
                new_gaps.append(gap_text)
                self.active_gaps[gap_text] = {
                    "status": "active",
                    "source_urls": source_urls,  # Sources to build upon
                    "related_findings": related_findings[:3],
                    "dimension": task.dimension
                }

        # If we found good evidence (confidence >= 0.8), no new gaps for this dimension
        # The iteration will naturally continue to other dimensions or refinement

        if self.debug:
            logger.debug(f"ADAPT: {adaptation} | New gaps: {len(new_gaps)}")

        return adaptation, new_gaps

    def _check_gap_resolution(self, new_evidence: List[SourceEvidence], task: ResearchTask):
        """Check if new evidence resolves any active gaps"""
        if not new_evidence or not self.active_gaps:
            return

        # Extract key topics from new evidence
        evidence_topics = set()
        for ev in new_evidence:
            # Extract from title
            evidence_topics.update(ev.title.lower().split())
            # Extract from facts
            for fact in ev.key_facts:
                evidence_topics.update(fact.lower().split())

        # Check each active gap
        gaps_to_resolve = []
        for gap_text, gap_data in list(self.active_gaps.items()):
            if gap_data["status"] != "active":
                continue

            # Simple keyword matching to see if gap is addressed
            gap_keywords = set(gap_text.lower().split())

            # If the current task dimension is mentioned in the gap
            # AND we found evidence with good confidence, mark as resolved
            if task.dimension.lower() in gap_text.lower():
                if task.confidence >= 0.7:  # Good evidence found
                    gaps_to_resolve.append(gap_text)
                    if self.debug:
                        logger.debug(f"📋 Gap RESOLVED: {gap_text[:80]}...")

        # Mark gaps as resolved (update status field)
        for gap in gaps_to_resolve:
            self.active_gaps[gap]["status"] = "resolved"

    def _check_convergence(self) -> bool:
        """Check if research has converged"""
        completed = sum(1 for t in self.tasks if t.status == "completed")
        return completed >= len(self.tasks) * 0.8  # 80% completion

    def _synthesize_with_grounding(self) -> SynthesisModel:
        """
        Phase 4: Synthesize findings with mandatory grounding and source attribution
        """
        if not self.evidence:
            logger.warning("⚠️ No evidence found - cannot synthesize grounded report")
            return SynthesisModel(
                title=f"Research Report: {self.context.query}",
                summary="No reliable information was found for this query. This may indicate an unknown topic or very limited online presence.",
                findings_with_sources=[
                    FindingWithSource(
                        finding="No verifiable information found",
                        evidence_ids=[]
                    )
                ],
                gaps_with_sources=[GapWithSource(gap=f"Complete information gap for: {self.context.query}", evidence_ids=[])],
                confidence=0.0
            )

        # Build evidence summary with NUMBERED sources
        evidence_summary = []
        evidence_urls = {}  # Map evidence ID to URL
        for i, ev in enumerate(self.evidence[:15], 1):  # Top 15 sources
            evidence_urls[i] = ev.url
            evidence_summary.append(
                f"[{i}] URL: {ev.url}\n"
                f"    Facts: {'; '.join(ev.key_facts[:3])}"
            )

        evidence_text = "\n".join(evidence_summary)

        prompt = f"""Synthesize a research report based ONLY on the verified evidence below.

Research query: {self.context.query}

VERIFIED EVIDENCE (numbered):
{evidence_text}

Create a synthesis with:
1. title: Descriptive report title
2. summary: 2-3 sentence executive summary
3. findings_with_sources: 3-7 key findings, EACH with evidence_ids listing which evidence numbers ([1], [2], etc.) support it
4. gaps_with_sources: Knowledge gaps, EACH with evidence_ids of sources that raised the question (can be empty if gap is general)
5. confidence: Overall confidence 0-1

CRITICAL REQUIREMENTS FOR CONSERVATIVE GROUNDING:
- Base findings ONLY on what is EXPLICITLY stated in evidence
- DO NOT make inferences beyond what sources explicitly state
- For EACH finding, specify which evidence numbers (e.g., [1, 3, 5]) support it
- Every finding MUST have at least one evidence_id
- If evidence says "X shared Y" → finding should say "X shared Y" (NOT "X created Y")
- If evidence says "X posted about Y" → finding should say "X discussed Y" (NOT "X did Y")
- Use cautious language for indirect evidence ("appears to", "is associated with", "has engaged with")
- Do NOT combine weak evidence to make strong claims

GAP SOURCE ATTRIBUTION:
- For EACH gap, identify which evidence numbers raised the question
- Example: If evidence [5] mentions "thesis on Alpha Shapes" → gap should have evidence_ids: [5]
- If gap is general (not from specific evidence), leave evidence_ids empty
- Clean out gaps that are ALREADY covered by findings (don't list as gaps if we have answers!)

Examples of CONSERVATIVE synthesis:
✅ CORRECT: "X shared LinkedIn posts about Cellosaurus and Reactome, indicating interest in these projects"
❌ WRONG: "X is the lead developer of Cellosaurus" (if evidence only shows sharing/commenting)

✅ CORRECT: "X has engaged with LLM integration in biocuration, as evidenced by posts discussing ChatGPT usage"
❌ WRONG: "X integrated ChatGPT into Reactome" (if evidence only shows discussion)"""

        response = self.llm.generate(prompt, response_model=SynthesisModel)

        if isinstance(response, SynthesisModel):
            # Rebuild finding_to_source map with synthesized findings
            self.finding_to_source = {}  # Clear old atomic findings
            for finding_obj in response.findings_with_sources:
                # Map finding to the FIRST supporting source URL
                if finding_obj.evidence_ids:
                    # Get the first valid evidence ID
                    for ev_id in finding_obj.evidence_ids:
                        if 1 <= ev_id <= len(self.evidence):
                            source_url = self.evidence[ev_id - 1].url
                            self.finding_to_source[finding_obj.finding] = source_url
                            logger.debug(f"✅ Mapped finding to source: {source_url}")
                            break
                    else:
                        # No valid evidence ID
                        self.finding_to_source[finding_obj.finding] = "UNKNOWN"
                        logger.warning(f"⚠️ Finding has no valid evidence IDs: {finding_obj.finding[:80]}")
                else:
                    self.finding_to_source[finding_obj.finding] = "UNKNOWN"
                    logger.warning(f"⚠️ Finding has no evidence IDs: {finding_obj.finding[:80]}")

            # Link gaps to their source origins
            for gap_obj in response.gaps_with_sources:
                source_urls = []
                if gap_obj.evidence_ids:
                    # Get URLs for all evidence IDs that raised this gap
                    for ev_id in gap_obj.evidence_ids:
                        if ev_id in evidence_urls:
                            source_urls.append(evidence_urls[ev_id])
                        elif 1 <= ev_id <= len(self.evidence):
                            source_urls.append(self.evidence[ev_id - 1].url)

                # Update or create gap tracking with source attribution
                self.active_gaps[gap_obj.gap] = {
                    "status": "active",
                    "source_urls": source_urls,
                    "related_findings": [],
                    "dimension": "synthesis"
                }

                if source_urls:
                    logger.debug(f"✅ Linked gap to {len(source_urls)} source(s): {gap_obj.gap[:60]}...")
                else:
                    logger.debug(f"⚠️ Gap has no source attribution: {gap_obj.gap[:60]}...")

            return response
        elif hasattr(response, 'content'):
            # Fallback: create simple synthesis with atomic findings
            fallback_findings = []
            for i, e in enumerate(self.evidence[:5]):
                if e.key_facts:
                    finding = e.key_facts[0]
                    fallback_findings.append(FindingWithSource(
                        finding=finding,
                        evidence_ids=[i + 1]
                    ))
                    self.finding_to_source[finding] = e.url

            return SynthesisModel(
                title=f"Research Report: {self.context.query}",
                summary=f"Based on {len(self.evidence)} verified sources.",
                findings_with_sources=fallback_findings,
                gaps_with_sources=[GapWithSource(gap="Some details may require further research", evidence_ids=[])],
                confidence=sum(e.relevance_score for e in self.evidence) / len(self.evidence)
            )
        else:
            return SynthesisModel(
                title=f"Research Report: {self.context.query}",
                summary="Synthesis generation failed.",
                findings_with_sources=[
                    FindingWithSource(
                        finding="Unable to generate structured synthesis",
                        evidence_ids=[]
                    )
                ],
                gaps_with_sources=[GapWithSource(gap="Synthesis error", evidence_ids=[])],
                confidence=0.5
            )
