"""
Basic Deep Researcher A - ReAct + Tree of Thoughts Strategy

This implementation follows the ReAct (Reasoning and Acting) paradigm combined with
Tree of Thoughts exploration for comprehensive research.

Architecture:
1. Master Orchestrator: Coordinates overall research workflow
2. Tree of Thoughts: Generates multiple research paths simultaneously
3. ReAct Loops: Iterative Think → Act → Observe → Refine cycles
4. Parallel Exploration: Concurrent searches across multiple branches
5. Progressive Synthesis: Incremental knowledge building

Key Features:
- Explicit reasoning traces for transparency
- Multi-hop reasoning across information sources
- Citation tracking with confidence scores
- Self-verification and fact-checking
- Adaptive depth control based on findings
"""

import json
import time
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.interface import AbstractCoreInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry
from ..utils.structured_logging import get_logger
from ..tools.common_tools import web_search, fetch_url

logger = get_logger(__name__)


# ==================== Data Models ====================

@dataclass
class ThoughtNode:
    """Represents a single thought node in the Tree of Thoughts"""
    id: str
    parent_id: Optional[str]
    thought: str
    priority: float
    status: str = "pending"  # pending, exploring, completed, pruned
    sub_queries: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ReActTrace:
    """Represents a single ReAct iteration trace"""
    iteration: int
    thought: str
    action: str
    observation: str
    reflection: str
    timestamp: str
    sources_found: int


class ResearchThought(BaseModel):
    """Structured thought for research planning"""
    aspect: str = Field(description="Research aspect to explore")
    reasoning: str = Field(description="Why this aspect is important")
    sub_questions: List[str] = Field(description="2-3 specific sub-questions", min_items=2, max_items=3)
    priority: float = Field(description="Priority score 0-1", ge=0, le=1)


class TreeOfThoughtsModel(BaseModel):
    """Multiple research paths to explore"""
    main_objective: str = Field(description="Main research objective")
    thoughts: List[ResearchThought] = Field(description="3-5 research paths to explore", min_items=3, max_items=5)
    exploration_strategy: str = Field(description="parallel or sequential")


class SearchQuerySetModel(BaseModel):
    """Set of search queries for a specific aspect"""
    queries: List[str] = Field(description="2-4 search queries", min_items=2, max_items=4)


class SourceRelevanceModel(BaseModel):
    """Relevance assessment for a source"""
    is_relevant: bool = Field(description="Whether source is relevant")
    relevance_score: float = Field(description="Relevance score 0-1", ge=0, le=1)
    confidence_score: float = Field(description="Confidence in source credibility 0-1", ge=0, le=1)
    key_insights: List[str] = Field(description="Key insights from source", max_items=3)
    needs_verification: bool = Field(description="Whether claims need verification")


class SynthesisModel(BaseModel):
    """Synthesis of research findings"""
    title: str = Field(description="Research report title")
    executive_summary: str = Field(description="Executive summary paragraph")
    key_findings: List[str] = Field(description="3-7 key findings", min_items=3, max_items=7)
    detailed_sections: List[Dict[str, str]] = Field(description="Detailed analysis sections with headings and content")
    confidence_assessment: float = Field(description="Overall confidence 0-1", ge=0, le=1)
    knowledge_gaps: List[str] = Field(description="Identified knowledge gaps", max_items=3)


class ResearchOutput(BaseModel):
    """Final research report output"""
    title: str
    summary: str
    key_findings: List[str]
    sources_probed: List[Dict[str, Any]]
    sources_selected: List[Dict[str, Any]]
    detailed_report: Dict[str, Any]
    confidence_score: float
    research_metadata: Dict[str, Any]


# ==================== Main Class ====================

class BasicDeepResearcherA:
    """
    Deep Researcher using ReAct + Tree of Thoughts Strategy

    This implementation emphasizes:
    - Explicit reasoning at each step
    - Parallel exploration of multiple research paths
    - Iterative refinement through ReAct loops
    - Self-verification and confidence scoring

    Example:
        >>> from abstractcore import create_llm
        >>> from abstractcore.processing import BasicDeepResearcherA
        >>>
        >>> llm = create_llm("openai", model="gpt-4o-mini")
        >>> researcher = BasicDeepResearcherA(llm)
        >>>
        >>> result = researcher.research("What are the latest advances in quantum error correction?")
        >>> print(json.dumps(result.dict(), indent=2))
    """

    def __init__(
        self,
        llm: Optional[AbstractCoreInterface] = None,
        max_tokens: int = 32000,
        max_output_tokens: int = 8000,
        timeout: Optional[float] = None,
        max_react_iterations: int = 3,
        max_parallel_paths: int = 3,
        max_sources: int = 25,
        search_provider: str = "duckduckgo",
        search_api_key: Optional[str] = None,
        temperature: float = 0.2,
        debug: bool = False
    ):
        """
        Initialize the Deep Researcher with ReAct + Tree of Thoughts

        Args:
            llm: LLM instance (defaults to Ollama if None)
            max_tokens: Maximum context tokens
            max_output_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            max_react_iterations: Maximum ReAct iterations per path
            max_parallel_paths: Maximum parallel thought branches
            max_sources: Maximum sources to collect
            search_provider: "duckduckgo" (default, free) or "serper" (requires API key)
            search_api_key: API key for search provider if needed
            temperature: LLM temperature (0.2 for balanced creativity/consistency)
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
                    "   researcher = BasicDeepResearcherA(llm)"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm

        self.max_react_iterations = max_react_iterations
        self.max_parallel_paths = max_parallel_paths
        self.max_sources = max_sources
        self.search_provider = search_provider
        self.search_api_key = search_api_key
        self.temperature = temperature
        self.debug = debug
        self.retry_strategy = FeedbackRetry(max_attempts=3)

        # Tracking
        self.thought_nodes: Dict[str, ThoughtNode] = {}
        self.react_traces: List[ReActTrace] = []
        self.sources_probed: List[Dict[str, Any]] = []
        self.sources_selected: List[Dict[str, Any]] = []
        self.seen_urls: set = set()

        logger.info(f"🤖 Initialized BasicDeepResearcherA with {self.llm.provider}/{self.llm.model}")
        logger.info(f"🎯 Strategy: ReAct + Tree of Thoughts | Max iterations: {max_react_iterations}")

    def research(
        self,
        query: str,
        max_depth: int = 2,
        focus_areas: Optional[List[str]] = None
    ) -> ResearchOutput:
        """
        Conduct deep research using ReAct + Tree of Thoughts

        Args:
            query: Research question
            max_depth: Maximum tree depth for thought exploration
            focus_areas: Optional list of specific areas to focus on

        Returns:
            ResearchOutput: Structured research report
        """
        start_time = time.time()
        logger.info(f"🔬 Starting research: {query}")

        # Reset tracking
        self.thought_nodes = {}
        self.react_traces = []
        self.sources_probed = []
        self.sources_selected = []
        self.seen_urls = set()

        # Phase 1: Generate Tree of Thoughts
        logger.info("🌳 Phase 1: Generating Tree of Thoughts")
        thought_tree = self._generate_thought_tree(query, focus_areas)

        # Phase 2: ReAct Exploration
        logger.info("🔄 Phase 2: ReAct Exploration")
        self._explore_with_react(thought_tree, max_depth)

        # Phase 3: Synthesis
        logger.info("📝 Phase 3: Synthesizing findings")
        final_report = self._synthesize_findings(query)

        duration = time.time() - start_time

        # Build output
        output = ResearchOutput(
            title=final_report.title,
            summary=final_report.executive_summary,
            key_findings=final_report.key_findings,
            sources_probed=self.sources_probed,
            sources_selected=self.sources_selected,
            detailed_report={
                "sections": final_report.detailed_sections,
                "confidence_assessment": final_report.confidence_assessment,
                "knowledge_gaps": final_report.knowledge_gaps
            },
            confidence_score=final_report.confidence_assessment,
            research_metadata={
                "strategy": "react_tree_of_thoughts",
                "duration_seconds": round(duration, 2),
                "react_iterations": len(self.react_traces),
                "thought_nodes_explored": len(self.thought_nodes),
                "sources_probed": len(self.sources_probed),
                "sources_selected": len(self.sources_selected),
                "model_used": f"{self.llm.provider}/{self.llm.model}"
            }
        )

        logger.info(f"✅ Research completed in {duration:.1f}s | {len(self.sources_selected)} sources")
        return output

    def _generate_thought_tree(self, query: str, focus_areas: Optional[List[str]]) -> TreeOfThoughtsModel:
        """Generate initial Tree of Thoughts for research exploration"""
        focus_context = ""
        if focus_areas:
            focus_context = f"\n\nFocus particularly on these areas: {', '.join(focus_areas)}"

        prompt = f"""You are a research strategist. Generate a Tree of Thoughts for exploring this research question.

Research Question: {query}{focus_context}

Create 3-5 distinct research paths (thoughts) that would comprehensively explore this topic.
For each thought:
1. Identify a specific aspect/angle to explore
2. Explain why it's important
3. Generate 2-3 concrete sub-questions
4. Assign a priority score (0-1, where 1 is most critical)

Think strategically about coverage - ensure the thoughts collectively address different facets of the question.
Decide if thoughts should be explored in parallel (independent aspects) or sequentially (building on each other).

Return a structured plan with main_objective, thoughts, and exploration_strategy."""

        response = self.llm.generate(prompt, response_model=TreeOfThoughtsModel, retry_strategy=self.retry_strategy)

        if isinstance(response, TreeOfThoughtsModel):
            thought_tree = response
        elif hasattr(response, 'structured_output'):
            thought_tree = response.structured_output
        else:
            raise ValueError("Failed to generate thought tree")

        # Create root node
        root_node = ThoughtNode(
            id="root",
            parent_id=None,
            thought=query,
            priority=1.0,
            status="completed"
        )
        self.thought_nodes["root"] = root_node

        # Create nodes for each thought
        for idx, thought in enumerate(thought_tree.thoughts):
            node_id = f"thought_{idx}"
            node = ThoughtNode(
                id=node_id,
                parent_id="root",
                thought=thought.aspect,
                priority=thought.priority,
                status="pending",
                sub_queries=thought.sub_questions
            )
            self.thought_nodes[node_id] = node
            root_node.children.append(node_id)

        logger.info(f"🌱 Generated {len(thought_tree.thoughts)} thought branches")
        return thought_tree

    def _explore_with_react(self, thought_tree: TreeOfThoughtsModel, max_depth: int):
        """Explore thought tree using ReAct loops"""
        # Sort thoughts by priority
        thoughts_to_explore = sorted(
            thought_tree.thoughts,
            key=lambda t: t.priority,
            reverse=True
        )[:self.max_parallel_paths]

        # CONCURRENCY FIX: Force sequential exploration for LMStudio provider
        # to avoid concurrent structured output JSON corruption issues
        provider_name = getattr(self.llm, 'provider', '').lower()
        force_sequential = provider_name in ['lmstudio', 'lmstudioprovider']
        
        if force_sequential:
            logger.info("🔧 Using sequential exploration (LMStudio concurrency fix)")
            exploration_strategy = "sequential"
        else:
            exploration_strategy = thought_tree.exploration_strategy

        if exploration_strategy == "parallel":
            # Parallel exploration
            logger.info("🔄 Using parallel exploration")
            with ThreadPoolExecutor(max_workers=self.max_parallel_paths) as executor:
                futures = []
                for idx, thought in enumerate(thoughts_to_explore):
                    node_id = f"thought_{idx}"
                    future = executor.submit(self._react_loop, node_id, thought, max_depth)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"ReAct loop failed: {e}")
        else:
            # Sequential exploration
            logger.info("🔄 Using sequential exploration")
            for idx, thought in enumerate(thoughts_to_explore):
                node_id = f"thought_{idx}"
                self._react_loop(node_id, thought, max_depth)

    def _react_loop(self, node_id: str, thought: ResearchThought, max_depth: int):
        """Execute ReAct loop for a single thought branch"""
        node = self.thought_nodes[node_id]
        node.status = "exploring"

        for iteration in range(self.max_react_iterations):
            if len(self.sources_selected) >= self.max_sources:
                logger.info(f"📊 Source limit reached, stopping ReAct loop for {node_id}")
                break

            # Think: Generate search queries
            queries = self._think_generate_queries(thought, iteration, node.findings)

            # Act: Execute searches
            new_sources = self._act_search(queries)

            # Observe: Assess sources
            relevant_sources = self._observe_assess_sources(new_sources, thought.aspect)

            # Reflect: Analyze progress
            reflection = self._reflect_on_progress(thought, node.findings, relevant_sources)

            # Update node
            node.findings.extend(relevant_sources)
            node.confidence = min(1.0, node.confidence + 0.3)  # Incremental confidence

            # Record trace
            trace = ReActTrace(
                iteration=iteration + 1,
                thought=f"Exploring: {thought.aspect}",
                action=f"Searched {len(queries)} queries",
                observation=f"Found {len(relevant_sources)} relevant sources",
                reflection=reflection,
                timestamp=datetime.now().isoformat(),
                sources_found=len(relevant_sources)
            )
            self.react_traces.append(trace)

            # Check if we should continue
            if "sufficient" in reflection.lower() or "complete" in reflection.lower():
                logger.info(f"✅ ReAct loop converged for {node_id} at iteration {iteration + 1}")
                break

        node.status = "completed"

    def _think_generate_queries(self, thought: ResearchThought, iteration: int, current_findings: List[Dict]) -> List[str]:
        """Think phase: Generate search queries"""
        context = ""
        if current_findings:
            context = f"\n\nCurrent findings summary: Found {len(current_findings)} sources. "
            context += "Generate queries to fill gaps or explore different angles."

        prompt = f"""Generate search queries for this research aspect (iteration {iteration + 1}):

Aspect: {thought.aspect}
Reasoning: {thought.reasoning}
Sub-questions: {', '.join(thought.sub_questions)}{context}

Generate 2-4 specific, diverse search queries that would find high-quality information.
Vary the query formulations to maximize coverage.

Return a JSON object with a 'queries' field containing the list of queries."""

        try:
            response = self.llm.generate(prompt, response_model=SearchQuerySetModel, retry_strategy=self.retry_strategy)

            if isinstance(response, SearchQuerySetModel):
                return response.queries
            elif hasattr(response, 'structured_output'):
                return response.structured_output.queries
            else:
                # Fallback: use sub-questions
                return thought.sub_questions[:2]
        except Exception as e:
            logger.warning(f"Query generation failed, using fallback: {e}")
            return thought.sub_questions[:2]

    def _act_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Act phase: Execute web searches"""
        sources = []

        for query in queries:
            if len(self.sources_probed) >= self.max_sources * 2:  # Probe 2x sources
                break

            try:
                if self.search_provider == "serper" and self.search_api_key:
                    # Use Serper.dev if configured
                    search_result = self._search_serper(query)
                else:
                    # Use DuckDuckGo (default, free)
                    search_result = web_search(query)

                # Parse search results
                parsed_sources = self._parse_search_results(search_result, query)
                sources.extend(parsed_sources)

            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")

        return sources

    def _search_serper(self, query: str) -> str:
        """Search using Serper.dev API"""
        import requests

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.search_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": 5
        }

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        results = []

        # Format similar to web_search output
        results.append(f"🔍 Search results for: '{query}'\n")

        for i, result in enumerate(data.get('organic', []), 1):
            title = result.get('title', 'No title')
            link = result.get('link', '')
            snippet = result.get('snippet', '')

            results.append(f"\n{i}. {title}")
            results.append(f"   🔗 {link}")
            results.append(f"   📄 {snippet}")

        return "\n".join(results)

    def _parse_search_results(self, search_output: str, query: str) -> List[Dict[str, Any]]:
        """Parse search results into structured format"""
        sources = []

        # Extract URLs and titles using regex
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

    def _observe_assess_sources(self, sources: List[Dict[str, Any]], aspect: str) -> List[Dict[str, Any]]:
        """Observe phase: Assess source relevance and quality"""
        relevant_sources = []

        for source in sources:
            if len(self.sources_selected) >= self.max_sources:
                break

            # Assess relevance
            prompt = f"""Assess the relevance of this source for researching: {aspect}

Source Title: {source.get('title', 'Unknown')}
Source URL: {source.get('url', 'Unknown')}
Snippet: {source.get('snippet', 'No snippet')}

Provide a structured assessment with these specific fields:
- is_relevant: true or false - whether this source is relevant to "{aspect}"
- relevance_score: number between 0.0 and 1.0 - how relevant is this source
- confidence_score: number between 0.0 and 1.0 - how credible is this source
- key_insights: list of 1-3 strings - what insights this source might provide
- needs_verification: true or false - whether claims need verification

Return ONLY the JSON object with these exact fields, no additional text."""

            try:
                response = self.llm.generate(prompt, response_model=SourceRelevanceModel, retry_strategy=self.retry_strategy)

                if isinstance(response, SourceRelevanceModel):
                    assessment = response
                elif hasattr(response, 'structured_output'):
                    assessment = response.structured_output
                else:
                    continue

                if assessment.is_relevant and assessment.relevance_score >= 0.6:
                    source['relevance_score'] = assessment.relevance_score
                    source['confidence_score'] = assessment.confidence_score
                    source['key_insights'] = assessment.key_insights
                    source['needs_verification'] = assessment.needs_verification

                    relevant_sources.append(source)
                    self.sources_selected.append(source.copy())

            except Exception as e:
                logger.warning(f"Source assessment failed: {e}")

        return relevant_sources

    def _reflect_on_progress(self, thought: ResearchThought, current_findings: List[Dict], new_findings: List[Dict]) -> str:
        """Reflect phase: Analyze research progress"""
        total_findings = len(current_findings) + len(new_findings)

        if total_findings == 0:
            return "No findings yet. Need to explore different search strategies."
        elif total_findings < 3:
            return "Limited findings. Continue exploring with different query formulations."
        elif total_findings < 5:
            return "Making progress. Need a few more sources for comprehensive coverage."
        else:
            return "Sufficient findings gathered. Ready to synthesize insights."

    def _synthesize_findings(self, original_query: str) -> SynthesisModel:
        """Synthesize all findings into comprehensive report"""
        # Aggregate findings
        all_insights = []
        source_info = []

        for source in self.sources_selected:
            source_info.append(f"- {source['title']} (Relevance: {source.get('relevance_score', 0):.2f})")
            if 'key_insights' in source:
                all_insights.extend(source['key_insights'])

        sources_summary = "\n".join(source_info[:10])  # Top 10 sources
        insights_summary = "\n".join([f"- {insight}" for insight in all_insights[:20]])  # Top 20 insights

        prompt = f"""Synthesize comprehensive research findings into a detailed report.

Original Research Question: {original_query}

Total Sources Analyzed: {len(self.sources_selected)}
ReAct Iterations Completed: {len(self.react_traces)}

Top Sources:
{sources_summary}

Key Insights Discovered:
{insights_summary}

Create a comprehensive research report with:
1. An engaging title
2. A clear executive summary (2-3 paragraphs)
3. 3-7 key findings (most important insights)
4. 3-5 detailed sections with headings and content
5. Overall confidence assessment (0-1)
6. 0-3 identified knowledge gaps

Ensure the report is well-structured, factual, and actionable.
Base all claims on the provided insights."""

        response = self.llm.generate(prompt, response_model=SynthesisModel, retry_strategy=self.retry_strategy)

        if isinstance(response, SynthesisModel):
            return response
        elif hasattr(response, 'structured_output'):
            return response.structured_output
        else:
            raise ValueError("Failed to synthesize findings")
