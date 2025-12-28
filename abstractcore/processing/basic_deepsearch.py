"""
Basic Deep Search - Autonomous research agent with multi-stage pipeline

Features:
- Four-stage pipeline: Planning, Question Development, Web Exploration, Report Generation
- Parallel web exploration for speed and breadth
- Structured report generation with citations
- Verification and fact-checking capabilities
- Configurable search depth and focus areas
"""

import json
import logging
import asyncio
import time
import re
import hashlib
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from ..core.interface import AbstractCoreInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry
from ..utils.structured_logging import get_logger
from ..tools.common_tools import web_search, fetch_url

logger = get_logger(__name__)


class SourceManager:
    """Manages source collection with strict limits and deduplication"""
    
    def __init__(self, max_sources: int):
        self.max_sources = max_sources
        self.collected_sources = []
        self.source_urls = set()  # For deduplication
        self.source_titles = set()  # Additional deduplication by title
    
    def add_source(self, source: Dict[str, Any]) -> bool:
        """Add source if under limit and not duplicate"""
        if len(self.collected_sources) >= self.max_sources:
            logger.debug(f"Source limit reached ({self.max_sources}), skipping: {source.get('url', 'unknown')}")
            return False
        
        url = source.get('url', '')
        title = source.get('title', '').lower().strip()
        
        # Check for URL duplication
        if url and url in self.source_urls:
            logger.debug(f"Duplicate URL skipped: {url}")
            return False
        
        # Check for title duplication (similar content from different URLs)
        if title and title in self.source_titles:
            logger.debug(f"Duplicate title skipped: {title}")
            return False
        
        self.collected_sources.append(source)
        if url:
            self.source_urls.add(url)
        if title:
            self.source_titles.add(title)
        
        logger.debug(f"Source added ({len(self.collected_sources)}/{self.max_sources}): {title or url}")
        return True
    
    def get_remaining_capacity(self) -> int:
        return max(0, self.max_sources - len(self.collected_sources))
    
    def get_sources(self) -> List[Dict[str, Any]]:
        return self.collected_sources.copy()
    
    def is_full(self) -> bool:
        return len(self.collected_sources) >= self.max_sources


class CitationValidator:
    """Validates and enforces citations in generated content"""
    
    @staticmethod
    def validate_citations(text: str, sources: List[Dict]) -> Dict[str, Any]:
        """Check if text contains proper citations for claims"""
        if not text or not sources:
            return {
                'citations_found': 0,
                'factual_sentences': 0,
                'citation_ratio': 0.0,
                'is_adequately_cited': False,
                'missing_citations': []
            }
        
        source_names = [s.get('title', '').strip() for s in sources if s.get('title')]
        
        # Count citation patterns (case-insensitive)
        citation_patterns = [
            r'according to \[([^\]]+)\]',
            r'as reported by \[([^\]]+)\]',
            r'according to ([^,.]+)',
            r'as reported by ([^,.]+)',
            r'\(([^)]+)\)',  # Parenthetical citations
        ]
        
        citations_found = 0
        cited_sources = set()
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations_found += len(matches)
            for match in matches:
                cited_sources.add(match.strip().lower())
        
        # Estimate factual claims (sentences with factual indicators)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        factual_indicators = [
            'show', 'found', 'research', 'study', 'data', 'report', 'analysis', 
            'indicates', 'reveals', 'demonstrates', 'confirms', 'suggests',
            'according', 'published', 'announced', 'released', 'stated'
        ]
        
        factual_sentences = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in factual_indicators):
                factual_sentences.append(sentence)
        
        citation_ratio = citations_found / max(len(factual_sentences), 1)
        
        # Check which sources are not cited
        uncited_sources = []
        for source in source_names:
            source_lower = source.lower()
            if not any(source_lower in cited.lower() for cited in cited_sources):
                uncited_sources.append(source)
        
        return {
            'citations_found': citations_found,
            'factual_sentences': len(factual_sentences),
            'citation_ratio': citation_ratio,
            'is_adequately_cited': citation_ratio >= 0.5,  # 50% threshold
            'uncited_sources': uncited_sources,
            'cited_sources': list(cited_sources)
        }
    
    @staticmethod
    def enhance_text_with_citations(text: str, sources: List[Dict]) -> str:
        """Enhance text by adding missing citations where appropriate"""
        if not sources:
            return text
        
        # Simple enhancement: add source list at the end if no citations found
        validation = CitationValidator.validate_citations(text, sources)
        
        if validation['citations_found'] == 0 and sources:
            source_list = "\n\nSources:\n" + "\n".join([
                f"- {s.get('title', 'Unknown')}: {s.get('url', 'No URL')}"
                for s in sources[:5]  # Limit to top 5 sources
            ])
            return text + source_list
        
        return text


# Pydantic models for structured output
class ResearchSubTaskModel(BaseModel):
    """Pydantic model for research sub-task (for structured output)"""
    id: str = Field(description="Unique task identifier")
    question: str = Field(description="Specific research question")
    theme: str = Field(description="Research theme this task addresses")
    priority: int = Field(description="Priority level (1=essential, 2=important, 3=supplementary)", ge=1, le=3)

class ResearchThemeModel(BaseModel):
    """Pydantic model for research theme (for structured output)"""
    name: str = Field(description="Theme name")
    questions: List[str] = Field(description="3 specific research questions for this theme", min_items=3, max_items=3)
    priority: int = Field(description="Theme priority (1=essential, 2=important, 3=supplementary)", ge=1, le=3)

class ResearchPlanModel(BaseModel):
    """Pydantic model for research plan (for structured output)"""
    research_objective: str = Field(description="Clear research objective")
    themes: List[ResearchThemeModel] = Field(description="Research themes with questions")
    search_strategy: str = Field(description="Search strategy", default="parallel")
    estimated_time_minutes: int = Field(description="Estimated time in minutes", gt=0)


class SearchQueriesModel(BaseModel):
    """Pydantic model for search queries (for structured output)"""
    queries: List[str] = Field(description="List of specific search queries", min_items=1, max_items=5)


@dataclass
class ResearchSubTask:
    """Represents a single research sub-task"""
    id: str
    question: str
    focus_area: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    search_queries: List[str] = None
    findings: List[Dict[str, Any]] = None
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class ResearchPlan:
    """Represents the overall research plan"""
    original_query: str
    research_objective: str
    sub_tasks: List[ResearchSubTask]
    estimated_time_minutes: int
    focus_areas: List[str]
    search_strategy: str = "breadth_first"  # breadth_first, depth_first, parallel


class ResearchFinding(BaseModel):
    """Structured representation of a research finding"""
    source_url: str = Field(description="URL of the source")
    title: str = Field(description="Title of the source")
    content: str = Field(description="Relevant content excerpt")
    relevance_score: float = Field(description="Relevance score 0-1")
    timestamp: str = Field(description="When this was found")
    sub_task_id: str = Field(description="Which sub-task this relates to")


class ResearchReport(BaseModel):
    """Structured research report"""
    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Brief executive summary")
    key_findings: List[str] = Field(description="List of key findings")
    detailed_analysis: str = Field(description="Detailed analysis section")
    conclusions: str = Field(description="Conclusions and implications")
    sources: List[Dict[str, Any]] = Field(description="List of sources with URLs, titles, and relevance scores")
    methodology: str = Field(description="Research methodology used")
    limitations: str = Field(description="Research limitations and caveats")


class BasicDeepSearch:
    """
    Basic Deep Search implementation following the four-stage pipeline:
    1. Planning: Decompose query into structured research plan
    2. Question Development: Generate specific search queries
    3. Web Exploration: Execute searches and gather evidence
    4. Report Generation: Synthesize findings into structured report

    Key features:
    - Autonomous multi-step research workflow
    - Parallel web exploration for speed
    - Structured output with citations
    - Configurable search depth and focus
    - Verification and fact-checking

    Examples:
        >>> searcher = BasicDeepSearch()
        
        # Basic research query
        >>> report = searcher.research("What are the latest developments in quantum computing?")
        
        # Research with specific focus
        >>> report = searcher.research(
        ...     "Impact of AI on healthcare", 
        ...     focus_areas=["medical diagnosis", "drug discovery", "patient care"]
        ... )
        
        # Deep research with custom parameters
        >>> report = searcher.research(
        ...     "Sustainable energy solutions 2025",
        ...     max_sources=20,
        ...     search_depth="comprehensive",
        ...     include_verification=True
        ... )
    """

    def __init__(
        self,
        llm: Optional[AbstractCoreInterface] = None,
        max_tokens: int = 32000,
        max_output_tokens: int = 8000,
        timeout: Optional[float] = None,
        max_parallel_searches: int = 5,
        full_text_extraction: bool = False,
        reflexive_mode: bool = False,
        max_reflexive_iterations: int = 2,
        temperature: float = 0.1,  # Low temperature for consistency
        debug_mode: bool = False
    ):
        """Initialize the deep search system
        
        Args:
            llm: AbstractCore instance (any provider). If None, uses default Ollama model
            max_tokens: Maximum total tokens for LLM context (default 32000)
            max_output_tokens: Maximum tokens for LLM output generation (default 8000)
            timeout: HTTP request timeout in seconds. None for unlimited timeout (default None)
            max_parallel_searches: Maximum number of parallel web searches (default 5)
            full_text_extraction: Whether to extract full text content from pages (default False)
            reflexive_mode: Whether to enable reflexive research that analyzes gaps and refines (default False)
            max_reflexive_iterations: Maximum number of reflexive refinement cycles (default 2)
            temperature: LLM temperature for consistency (default 0.1 for deterministic outputs)
            debug_mode: Enable comprehensive debug logging (default False)
        """
        if llm is None:
            try:
                self.llm = create_llm(
                    "ollama", 
                    model="qwen3:4b-instruct-2507-q4_K_M", 
                    max_tokens=max_tokens, 
                    max_output_tokens=max_output_tokens, 
                    temperature=temperature,  # Use consistent low temperature
                    timeout=timeout
                )
                
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default Ollama model 'qwen3:4b-instruct-2507-q4_K_M': {e}\n\n"
                    "💡 To use the default model, please:\n"
                    "   1. Install Ollama from: https://ollama.com/\n"
                    "   2. Download the model: ollama pull qwen3:4b-instruct-2507-q4_K_M\n"
                    "   3. Start Ollama service\n\n"
                    "⚡ For best deep search performance, consider these models:\n"
                    "   - qwen3-coder:30b (excellent for research and analysis, requires 32GB RAM)\n"
                    "   - gpt-4o-mini (cloud-based, fast and reliable)\n"
                    "   - claude-3-5-haiku (cloud-based, excellent reasoning)\n\n"
                    "🔧 Alternatively, provide a custom LLM instance:\n"
                    "   from abstractcore import create_llm\n"
                    "   from abstractcore.processing import BasicDeepSearch\n"
                    "   \n"
                    "   llm = create_llm('openai', model='gpt-4o-mini', max_tokens=32000, max_output_tokens=8000)\n"
                    "   searcher = BasicDeepSearch(llm)"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm

        self.max_parallel_searches = max_parallel_searches
        self.full_text_extraction = full_text_extraction
        self.reflexive_mode = reflexive_mode
        self.max_reflexive_iterations = max_reflexive_iterations
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.retry_strategy = FeedbackRetry(max_attempts=3)
        print(f"🤖 Initialized LLM: {self.llm.provider} {self.llm.model}")
        
        # Debug tracking
        if self.debug_mode:
            self.debug_info = {
                'all_queries': [],
                'all_urls_found': [],
                'relevance_assessments': [],
                'accepted_sources': [],
                'rejected_sources': []
            }

    def research(
        self,
        query: str,
        focus_areas: Optional[List[str]] = None,
        max_sources: int = 15,
        search_depth: str = "standard",  # brief, standard, comprehensive
        include_verification: bool = True,
        output_format: str = "structured"  # structured, narrative, executive
    ) -> Union[ResearchReport, Dict[str, Any]]:
        """
        Conduct autonomous deep research on a given query
        
        Args:
            query: The research question or topic
            focus_areas: Specific areas to focus on (optional)
            max_sources: Maximum number of sources to gather (default 15)
            search_depth: Research depth - brief, standard, comprehensive (default standard)
            include_verification: Whether to include fact-checking (default True)
            output_format: Output format - structured, narrative, executive (default structured)
            
        Returns:
            ResearchReport object or dictionary with research findings
        """
        logger.info(f"🔍 Starting deep search research: {query}")
        start_time = time.time()
        
        try:
            # Initialize source manager with strict limits
            source_manager = SourceManager(max_sources)
            logger.info(f"🎯 Initialized source manager with limit: {max_sources}")
            
            # Stage 1: Planning
            logger.info("📋 Stage 1: Planning research approach...")
            research_plan = self._create_research_plan(query, focus_areas, search_depth)
            
            # Stage 2: Question Development
            logger.info("❓ Stage 2: Developing search questions...")
            self._develop_search_questions(research_plan, max_sources)
            
            # Debug: Show all generated queries
            if self.debug_mode:
                print("\n" + "="*80)
                print("🔍 DEBUG: ALL GENERATED SEARCH QUERIES")
                print("="*80)
                for i, sub_task in enumerate(research_plan.sub_tasks, 1):
                    print(f"\n📋 Sub-task {i}: {sub_task.question}")
                    print(f"🎯 Focus: {sub_task.focus_area}")
                    print(f"🔍 Queries ({len(sub_task.search_queries)}):")
                    for j, query in enumerate(sub_task.search_queries, 1):
                        print(f"   {j}. \"{query}\"")
                        self.debug_info['all_queries'].append({
                            'sub_task_id': sub_task.id,
                            'sub_task_question': sub_task.question,
                            'query': query
                        })
                print("="*80)
            
            # Stage 3: Web Exploration
            logger.info("🌐 Stage 3: Exploring web sources...")
            findings = self._explore_web_sources(research_plan, source_manager)
            
            # Stage 4: Report Generation
            logger.info("📝 Stage 4: Generating research report...")
            report = self._generate_report(research_plan, findings, output_format)
            
            # Optional: Verification
            if include_verification:
                logger.info("✅ Stage 5: Verifying findings...")
                report = self._verify_report(report, findings)
            
            # Stage 6: Reflexive improvement (if enabled)
            if self.reflexive_mode:
                logger.info("🔄 Stage 6: Reflexive analysis and refinement...")
                report = self._reflexive_refinement(report, research_plan, findings)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✨ Deep search completed in {elapsed_time:.1f} seconds")
            
            # Debug: Show comprehensive summary
            if self.debug_mode:
                self._print_debug_summary()
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Deep search failed: {e}")
            raise

    def _create_research_plan(
        self, 
        query: str, 
        focus_areas: Optional[List[str]], 
        search_depth: str
    ) -> ResearchPlan:
        """Stage 1: Create a structured research plan"""
        
        # Detect query type and get appropriate focus areas
        query_type = self._detect_query_type(query)
        if not focus_areas:
            focus_areas = self._get_focus_areas_by_type(query_type)
        
        # Determine number of sub-tasks based on search depth
        depth_config = {
            "brief": {"sub_tasks": 3, "time_estimate": 5},
            "standard": {"sub_tasks": 5, "time_estimate": 10},
            "comprehensive": {"sub_tasks": 8, "time_estimate": 20}
        }
        config = depth_config.get(search_depth, depth_config["standard"])
        
        planning_prompt = f"""
You are an expert research strategist. Analyze the following research query and create a comprehensive research plan.

RESEARCH QUERY: "{query}"
SEARCH DEPTH: {search_depth}
TARGET SUB-TASKS: {config['sub_tasks']}

Your task is to intelligently decompose this query into focused research themes and specific sub-questions.

INSTRUCTIONS:
1. First, identify the key themes/dimensions that need to be researched to fully answer this query
2. For each theme, generate 3 specific, actionable research questions
3. Prioritize the themes based on their importance to answering the original query
4. Ensure the questions are diverse and cover different aspects of the topic

RESEARCH OBJECTIVE: Write a clear 1-2 sentence objective that captures what we're trying to learn.

THEMES & SUB-QUESTIONS:
For each theme you identify, provide:
- Theme name (e.g., "Technical Overview", "Historical Context", "Current Applications")  
- 3 specific research questions for that theme
- Priority level (1=essential, 2=important, 3=supplementary)

REQUIREMENTS:
- Generate exactly {config['sub_tasks']} total sub-questions across all themes
- Each question should be specific enough to guide targeted web searches
- Questions should be complementary, not overlapping
- Adapt the themes naturally to the query - don't force artificial categories
- For person queries: focus on biography, work, impact, affiliations, recent activities
- For concept queries: focus on definition, applications, development, current state, implications
- For technology queries: focus on how it works, use cases, advantages/limitations, market adoption
- For location queries: focus on geography, culture, economy, politics, current events
- For organization queries: focus on mission, leadership, products/services, market position, recent news

The themes should emerge naturally from understanding what someone would want to know about this specific topic.
"""

        try:
            # Use structured output for reliable JSON parsing
            plan_model = self.llm.generate(
                planning_prompt, 
                temperature=0.3,
                response_model=ResearchPlanModel
            )
            
            # Convert Pydantic model to dataclass objects
            sub_tasks = []
            focus_areas = []
            task_counter = 1
            
            for theme_model in plan_model.themes:
                focus_areas.append(theme_model.name)
                
                # Create sub-tasks from theme questions
                for question in theme_model.questions:
                    sub_task = ResearchSubTask(
                        id=f"task_{task_counter}",
                        question=question,
                        focus_area=theme_model.name,
                        priority=theme_model.priority,
                        search_queries=[],
                        findings=[],
                        status="pending"
                    )
                    sub_tasks.append(sub_task)
                    task_counter += 1
            
            research_plan = ResearchPlan(
                original_query=query,
                research_objective=plan_model.research_objective,
                sub_tasks=sub_tasks,
                estimated_time_minutes=plan_model.estimated_time_minutes,
                focus_areas=focus_areas,
                search_strategy=plan_model.search_strategy
            )
            
            logger.info(f"📋 Created research plan with {len(sub_tasks)} sub-tasks")
            return research_plan
            
        except Exception as e:
            logger.error(f"Failed to parse research plan: {e}")
            # Fallback to simple plan
            return self._create_fallback_plan(query, focus_areas, search_depth)

    def _create_fallback_plan(self, query: str, focus_areas: Optional[List[str]], search_depth: str) -> ResearchPlan:
        """Create a simple fallback research plan if JSON parsing fails"""
        
        # Simple sub-tasks based on common research patterns
        sub_tasks = [
            ResearchSubTask("task_1", f"What is {query}? Provide definitions and overview", "overview", 1),
            ResearchSubTask("task_2", f"What are the current developments in {query}?", "current_state", 1),
            ResearchSubTask("task_3", f"What are the key challenges or issues with {query}?", "challenges", 2),
            ResearchSubTask("task_4", f"What are future trends and predictions for {query}?", "future", 2),
        ]
        
        if search_depth == "comprehensive":
            sub_tasks.extend([
                ResearchSubTask("task_5", f"Who are the key players or experts in {query}?", "stakeholders", 2),
                ResearchSubTask("task_6", f"What are the economic or business implications of {query}?", "economics", 3),
                ResearchSubTask("task_7", f"What are the technical or scientific aspects of {query}?", "technical", 3),
            ])
        
        return ResearchPlan(
            original_query=query,
            research_objective=f"Comprehensive research on {query}",
            sub_tasks=sub_tasks,
            estimated_time_minutes=10,
            focus_areas=focus_areas or ["overview", "current_state", "challenges", "future"],
            search_strategy="parallel"
        )

    def _develop_search_questions(self, research_plan: ResearchPlan, max_sources: int) -> None:
        """Stage 2: Develop specific search queries for each sub-task"""
        
        queries_per_task = max(2, max_sources // len(research_plan.sub_tasks))
        
        for sub_task in research_plan.sub_tasks:
            query_prompt = f"""
Generate {queries_per_task} specific, diverse search queries for this research question:

RESEARCH QUESTION: {sub_task.question}
FOCUS AREA: {sub_task.focus_area}
ORIGINAL QUERY: {research_plan.original_query}

Create search queries that:
1. Use different keywords and phrasings
2. Target different types of sources (news, academic, industry, etc.)
3. Include recent time-sensitive queries where relevant (2024, 2025)
4. Are specific enough to find relevant information
5. Avoid generic terms that might return irrelevant results

Examples of good search queries for quantum computing:
- "quantum computing hardware advances 2024"
- "quantum computer error correction breakthrough 2024"
- "IBM Google quantum computing progress 2024"
- "quantum computing industry applications 2024"
- "quantum supremacy achievements 2024"

Avoid generic terms like "qubit" alone (which returns lab instruments) - be specific about quantum computing context.
"""

            try:
                # Use structured output for reliable parsing
                queries_model = self.llm.generate(
                    query_prompt, 
                    temperature=0.5,
                    response_model=SearchQueriesModel
                )
                
                sub_task.search_queries = queries_model.queries[:queries_per_task]
                logger.info(f"📝 Generated queries for {sub_task.id}: {sub_task.search_queries}")
                
            except Exception as e:
                logger.warning(f"Failed to parse queries for {sub_task.id}, using fallback")
                # Improved fallback queries with better specificity
                base_topic = research_plan.original_query.replace("What are the latest developments in ", "").replace("?", "")
                sub_task.search_queries = [
                    f"{base_topic} {sub_task.focus_area} 2024",
                    f"{base_topic} advances {sub_task.focus_area} 2024",
                    f"{base_topic} research {sub_task.focus_area} latest"
                ][:queries_per_task]
                logger.info(f"📝 Using fallback queries for {sub_task.id}: {sub_task.search_queries}")

    def _explore_web_sources(self, research_plan: ResearchPlan, source_manager: SourceManager) -> List[ResearchFinding]:
        """Stage 3: Execute web searches and gather evidence"""
        
        all_findings = []
        
        # Collect all search queries with their sub-task context
        search_tasks = []
        for sub_task in research_plan.sub_tasks:
            for query in sub_task.search_queries:
                search_tasks.append((sub_task.id, query, sub_task.priority))
        
        # Sort by priority (1=high priority first)
        search_tasks.sort(key=lambda x: x[2])
        
        # Global URL deduplication across all sub-tasks
        processed_urls = set()
        
        if self.debug_mode:
            print(f"\n🔍 DEBUG: Starting web exploration with {len(search_tasks)} search tasks")
            print(f"🎯 Source limit: {source_manager.max_sources}")
            for i, (sub_task_id, query, priority) in enumerate(search_tasks, 1):
                print(f"   {i}. [{sub_task_id}] \"{query}\" (Priority: {priority})")
        
        # Execute searches in parallel with source limit management
        with ThreadPoolExecutor(max_workers=self.max_parallel_searches) as executor:
            # Submit search tasks
            future_to_task = {}
            for sub_task_id, query, priority in search_tasks:
                # Check if we still have capacity
                if source_manager.is_full():
                    logger.info(f"🎯 Source limit reached ({source_manager.max_sources}), stopping search submission")
                    break
                    
                future = executor.submit(self._execute_search, sub_task_id, query, source_manager, processed_urls)
                future_to_task[future] = (sub_task_id, query)
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                sub_task_id, query = future_to_task[future]
                try:
                    findings = future.result()
                    all_findings.extend(findings)
                    logger.debug(f"Completed search for {sub_task_id}: {query} - {len(findings)} findings")
                except Exception as e:
                    logger.warning(f"Search failed for {sub_task_id} '{query}': {e}")
                
                # Early termination if source limit reached
                if source_manager.is_full():
                    logger.info(f"🎯 Source limit reached ({source_manager.max_sources}), stopping early")
                    break
        
        # Update sub-tasks with their findings
        findings_by_task = {}
        for finding in all_findings:
            task_id = finding.sub_task_id
            if task_id not in findings_by_task:
                findings_by_task[task_id] = []
            findings_by_task[task_id].append(finding)
        
        for sub_task in research_plan.sub_tasks:
            sub_task.findings = findings_by_task.get(sub_task.id, [])
            sub_task.status = "completed" if sub_task.findings else "failed"
        
        logger.info(f"🌐 Gathered {len(all_findings)} findings from web exploration")
        return all_findings

    def _execute_search(self, sub_task_id: str, query: str, source_manager: SourceManager, processed_urls: set) -> List[ResearchFinding]:
        """Execute a single web search and extract findings"""
        
        findings = []
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Perform web search
            logger.info(f"🔍 Executing search for: {query}")
            search_results = web_search(query, num_results=5)
            logger.debug(f"📄 Search results length: {len(search_results)}")
            logger.debug(f"📄 Search results preview: {search_results[:500]}")
            
            # Parse search results to extract URLs and content
            urls = self._extract_urls_from_search(search_results)
            logger.info(f"🔗 Extracted {len(urls)} URLs from search results")
            
            # Deduplicate URLs globally across all sub-tasks
            original_count = len(urls)
            urls = [(url, title) for url, title in urls if url not in processed_urls]
            deduplicated_count = len(urls)
            
            # Add new URLs to processed set
            for url, title in urls:
                processed_urls.add(url)
            
            if self.debug_mode and original_count > deduplicated_count:
                print(f"\n🔄 DEBUG: URL Deduplication for query \"{query}\":")
                print(f"   📊 Original URLs: {original_count}")
                print(f"   📊 After deduplication: {deduplicated_count}")
                print(f"   📊 Duplicates removed: {original_count - deduplicated_count}")
            
            # Debug: Show all URLs found for this query
            if self.debug_mode:
                print(f"\n🔍 DEBUG: URLs found for query \"{query}\":")
                for i, (url, title) in enumerate(urls, 1):
                    print(f"   {i}. {title}")
                    print(f"      🔗 {url}")
                    self.debug_info['all_urls_found'].append({
                        'query': query,
                        'sub_task_id': sub_task_id,
                        'url': url,
                        'title': title
                    })
            
            if not urls:
                logger.warning(f"⚠️ No URLs found in search results for query: {query}")
                logger.debug(f"Full search results: {search_results}")
                # Try to create a synthetic finding from the search results if they contain useful information
                if len(search_results) > 100 and "Error searching internet" not in search_results:
                    synthetic_finding = ResearchFinding(
                        source_url="https://duckduckgo.com/?q=" + query.replace(" ", "+"),
                        title=f"Search results for: {query}",
                        content=search_results[:500] + "...",
                        relevance_score=0.3,
                        timestamp=timestamp,
                        sub_task_id=sub_task_id
                    )
                    findings.append(synthetic_finding)
                    logger.info(f"✅ Created synthetic finding from search results")
                return findings
            
            # Fetch content from promising URLs with source manager control
            for i, (url, title) in enumerate(urls):
                # Check source manager capacity before processing
                if source_manager.is_full():
                    logger.info(f"🎯 Source limit reached, stopping URL processing for query: {query}")
                    break
                    
                try:
                    logger.debug(f"🌐 Fetching content from URL {i+1}: {url}")
                    content = fetch_url(url, timeout=15, include_full_content=self.full_text_extraction)
                    
                    if "Error" in content or len(content) < 100:
                        logger.debug(f"⚠️ Skipping URL due to fetch error or short content: {url}")
                        continue
                    
                    # Extract relevant content using structured parsing or LLM
                    if self.full_text_extraction:
                        # For full text mode, use custom fetch with more content
                        relevant_content = self._extract_relevant_content_full_text(content, query, url)
                    else:
                        # Standard mode with structured parsing
                        relevant_content = self._extract_relevant_content(content, query)
                    
                    if relevant_content:
                        # Use LLM to assess content relevance and quality
                        quality_assessment = self._assess_content_relevance(relevant_content, query, title)
                        
                        # Debug: Show relevance assessment details
                        if self.debug_mode:
                            print(f"\n🧠 DEBUG: Relevance Assessment for {title}")
                            print(f"   🔗 URL: {url}")
                            print(f"   📊 Relevant: {quality_assessment['is_relevant']}")
                            print(f"   📈 Score: {quality_assessment['relevance_score']:.2f}")
                            print(f"   💭 Reason: {quality_assessment['reason']}")
                            print(f"   📝 Content preview: {relevant_content[:200]}...")
                            
                            self.debug_info['relevance_assessments'].append({
                                'url': url,
                                'title': title,
                                'query': query,
                                'is_relevant': quality_assessment['is_relevant'],
                                'relevance_score': quality_assessment['relevance_score'],
                                'reason': quality_assessment['reason'],
                                'content_preview': relevant_content[:200]
                            })
                        
                        if quality_assessment['is_relevant']:
                            # Create source for manager validation
                            source_data = {
                                'url': url,
                                'title': title,
                                'content': relevant_content,
                                'relevance_score': quality_assessment['relevance_score'],
                                'timestamp': timestamp,
                                'sub_task_id': sub_task_id
                            }
                            
                            # Try to add to source manager (handles deduplication and limits)
                            if source_manager.add_source(source_data):
                                finding = ResearchFinding(
                                    source_url=url,
                                    title=title,
                                    content=relevant_content,
                                    relevance_score=quality_assessment['relevance_score'],
                                    timestamp=timestamp,
                                    sub_task_id=sub_task_id
                                )
                                findings.append(finding)
                                logger.info(f"✅ Added relevant finding from {url} (score: {quality_assessment['relevance_score']:.2f}) ({len(source_manager.get_sources())}/{source_manager.max_sources})")
                                
                                if self.debug_mode:
                                    self.debug_info['accepted_sources'].append({
                                        'url': url,
                                        'title': title,
                                        'relevance_score': quality_assessment['relevance_score'],
                                        'reason': 'Accepted by source manager'
                                    })
                            else:
                                logger.debug(f"🎯 Source not added (duplicate or limit reached): {url}")
                                if self.debug_mode:
                                    self.debug_info['rejected_sources'].append({
                                        'url': url,
                                        'title': title,
                                        'relevance_score': quality_assessment['relevance_score'],
                                        'reason': 'Duplicate or source limit reached'
                                    })
                        else:
                            logger.info(f"🚫 Content filtered out from {url}: {quality_assessment['reason']}")
                            if self.debug_mode:
                                self.debug_info['rejected_sources'].append({
                                    'url': url,
                                    'title': title,
                                    'relevance_score': quality_assessment['relevance_score'],
                                    'reason': f"Not relevant: {quality_assessment['reason']}"
                                })
                    else:
                        logger.debug(f"⚠️ No relevant content extracted from {url}")
                        if self.debug_mode:
                            self.debug_info['rejected_sources'].append({
                                'url': url,
                                'title': title,
                                'relevance_score': 0.0,
                                'reason': 'No relevant content could be extracted'
                            })
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Search execution failed for '{query}': {e}")
        
        logger.info(f"📊 Search completed for '{query}': {len(findings)} findings")
        return findings

    def _check_authority_indicators(self, title: str, content: str, query: str) -> Dict[str, Any]:
        """Check for high-authority source indicators that should be prioritized"""
        
        title_lower = title.lower()
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Extract potential person name from query
        query_words = query_lower.split()
        potential_names = [word for word in query_words if word.istitle() or len(word) > 3]
        
        # High-authority indicators
        authority_indicators = [
            # Academic/Professional profiles
            ('google scholar', 0.95, 'Official Google Scholar profile'),
            ('orcid', 0.95, 'Official ORCID researcher profile'),
            ('researchgate', 0.90, 'ResearchGate academic profile'),
            ('linkedin', 0.85, 'Professional LinkedIn profile'),
            ('academia.edu', 0.85, 'Academia.edu academic profile'),
            
            # Institutional websites
            ('university', 0.90, 'University/academic institution'),
            ('institute', 0.90, 'Research institute'),
            ('laboratory', 0.85, 'Research laboratory'),
            ('.edu', 0.90, 'Educational institution domain'),
            ('.ac.', 0.90, 'Academic institution domain'),
            
            # Personal/official websites
            ('personal website', 0.95, 'Personal/official website'),
            ('official site', 0.95, 'Official website'),
        ]
        
        # Check for personal name match in title/content
        name_match_score = 0.0
        if potential_names:
            for name in potential_names:
                if name in title_lower or name in content_lower:
                    name_match_score = 0.8
                    break
        
        # Check authority indicators
        for indicator, base_score, reason in authority_indicators:
            if indicator in title_lower or indicator in content_lower:
                final_score = min(1.0, base_score + name_match_score * 0.2)
                return {
                    'is_high_authority': True,
                    'authority_score': final_score,
                    'reason': reason + (f' with name match' if name_match_score > 0 else '')
                }
        
        # Check for personal domain (e.g., lpalbou.info)
        if any(name in title_lower for name in potential_names if len(name) > 3):
            return {
                'is_high_authority': True,
                'authority_score': 0.95,
                'reason': 'Personal domain/website matching query subject'
            }
        
        return {
            'is_high_authority': False,
            'authority_score': 0.0,
            'reason': 'No high-authority indicators found'
        }

    def _assess_content_relevance(self, content: str, query: str, title: str) -> Dict[str, Any]:
        """Use LLM to quickly assess if content is relevant to the research query"""
        
        # First check for high-authority sources that should be prioritized
        authority_indicators = self._check_authority_indicators(title, content, query)
        if authority_indicators['is_high_authority']:
            return {
                'is_relevant': True,
                'relevance_score': authority_indicators['authority_score'],
                'reason': f"High-authority source: {authority_indicators['reason']}"
            }
        
        # Limit content for efficient assessment
        assessment_content = content[:1500] + "..." if len(content) > 1500 else content
        
        assessment_prompt = f"""
Assess if this content contains meaningful information related to the research query.

RESEARCH QUERY: {query}
SOURCE TITLE: {title}

CONTENT:
{assessment_content}

Respond with ONLY a JSON object in this exact format:
{{
    "is_relevant": true/false,
    "relevance_score": 0.0-1.0,
    "reason": "brief explanation"
}}

CRITICAL: Mark as RELEVANT (true) if the content:
- Contains ANY substantive information that could help answer or relate to the query
- Provides facts, data, explanations, or details about the query topic
- Is from authoritative sources (official websites, academic profiles, institutional pages)
- Has meaningful textual content beyond navigation elements
- Shows biographical, professional, or academic information when querying about a person

Mark as NOT RELEVANT (false) ONLY if the content:
- Is completely unrelated to the query topic (different person, concept, etc.)
- Contains ONLY navigation menus, headers, footers, or structural elements
- Shows clear error messages, access restrictions, or "page not found" 
- Is purely promotional/advertising without ANY informational value
- Discusses entirely different subjects with no connection to the query

BE GENEROUS with relevance assessment - when in doubt, mark as relevant.
"""

        try:
            response = self.llm.generate(assessment_prompt, temperature=0.1)
            
            # Extract text from response (handle different response types)
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse JSON from response
            import json
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                assessment = json.loads(json_text)
                
                # Validate and normalize
                if 'is_relevant' in assessment and 'relevance_score' in assessment:
                    assessment['relevance_score'] = max(0.0, min(1.0, float(assessment['relevance_score'])))
                    assessment['reason'] = assessment.get('reason', 'No reason provided')
                    return assessment
            
            # Fallback if JSON parsing fails
            logger.debug(f"Content relevance assessment JSON parsing failed, using fallback")
            return self._fallback_relevance_assessment(content, query)
            
        except Exception as e:
            logger.debug(f"Content relevance assessment failed: {e}")
            return self._fallback_relevance_assessment(content, query)
    
    def _fallback_relevance_assessment(self, content: str, query: str) -> Dict[str, Any]:
        """Fallback relevance assessment using general content quality heuristics"""
        
        content_lower = content.lower()
        words = content.split()
        word_count = len(words)
        
        # Check for obvious error/empty content indicators
        error_indicators = [
            'page not found', '404 error', '403 error', '500 error',
            'access denied', 'login required', 'sign in required',
            'javascript required', 'cookies required', 'enable javascript',
            'subscribe to continue', 'sign up to read', 'premium content',
            'page does not exist', 'content not available'
        ]
        
        has_errors = any(indicator in content_lower for indicator in error_indicators)
        
        # Check for navigation-heavy content (low information density)
        navigation_indicators = ['home', 'about', 'contact', 'menu', 'navigation', 'footer', 'header']
        nav_count = sum(1 for indicator in navigation_indicators if indicator in content_lower)
        nav_ratio = nav_count / max(word_count, 1)
        
        # Basic content quality assessment
        if has_errors:
            return {
                'is_relevant': False,
                'relevance_score': 0.0,
                'reason': 'Contains error messages or access restrictions'
            }
        
        if word_count < 10:
            return {
                'is_relevant': False,
                'relevance_score': 0.0,
                'reason': f'Too little content ({word_count} words)'
            }
        
        if nav_ratio > 0.3:  # More than 30% navigation terms
            return {
                'is_relevant': False,
                'relevance_score': 0.2,
                'reason': 'Content appears to be mostly navigation elements'
            }
        
        # If content passes basic quality checks, calculate relevance
        query_words = [word.lower().strip('.,!?;:"()[]{}') for word in query.split() if len(word) > 2]
        
        if not query_words:
            # If query has no meaningful words, accept content based on quality
            relevance_score = 0.7 if word_count >= 50 else 0.5
            return {
                'is_relevant': True,
                'relevance_score': relevance_score,
                'reason': f'Query has no key terms, accepting based on content quality ({word_count} words)'
            }
        
        # Calculate keyword overlap
        matches = sum(1 for word in query_words if word in content_lower)
        keyword_relevance = matches / len(query_words)
        
        # Content length bonus (longer content more likely to be informative)
        length_bonus = min(0.3, word_count / 200)  # Up to 0.3 bonus for 200+ words
        
        final_relevance = keyword_relevance + length_bonus
        is_relevant = final_relevance >= 0.4  # Require meaningful keyword overlap, don't accept long irrelevant content
        
        return {
            'is_relevant': is_relevant,
            'relevance_score': min(1.0, final_relevance),
            'reason': f'{matches}/{len(query_words)} keywords, {word_count} words, score: {final_relevance:.2f}'
        }

    def _print_debug_summary(self):
        """Print comprehensive debug summary"""
        print("\n" + "="*80)
        print("🔍 DEBUG SUMMARY: COMPLETE RESEARCH PROCESS")
        print("="*80)
        
        # Query summary
        print(f"\n📋 TOTAL QUERIES GENERATED: {len(self.debug_info['all_queries'])}")
        query_by_subtask = {}
        for q in self.debug_info['all_queries']:
            subtask = q['sub_task_question']
            if subtask not in query_by_subtask:
                query_by_subtask[subtask] = []
            query_by_subtask[subtask].append(q['query'])
        
        for subtask, queries in query_by_subtask.items():
            print(f"\n🎯 {subtask}")
            for i, query in enumerate(queries, 1):
                print(f"   {i}. \"{query}\"")
        
        # URL summary
        print(f"\n🔗 TOTAL URLS DISCOVERED: {len(self.debug_info['all_urls_found'])}")
        urls_by_query = {}
        for url_info in self.debug_info['all_urls_found']:
            query = url_info['query']
            if query not in urls_by_query:
                urls_by_query[query] = []
            urls_by_query[query].append((url_info['title'], url_info['url']))
        
        for query, urls in urls_by_query.items():
            print(f"\n🔍 Query: \"{query}\" → {len(urls)} URLs")
            for i, (title, url) in enumerate(urls, 1):
                print(f"   {i}. {title}")
                print(f"      🔗 {url}")
        
        # Relevance assessment summary
        print(f"\n🧠 RELEVANCE ASSESSMENTS: {len(self.debug_info['relevance_assessments'])}")
        relevant_count = sum(1 for a in self.debug_info['relevance_assessments'] if a['is_relevant'])
        irrelevant_count = len(self.debug_info['relevance_assessments']) - relevant_count
        
        print(f"   ✅ Relevant: {relevant_count}")
        print(f"   ❌ Not Relevant: {irrelevant_count}")
        
        if self.debug_info['relevance_assessments']:
            print(f"\n📊 DETAILED ASSESSMENTS:")
            for i, assessment in enumerate(self.debug_info['relevance_assessments'], 1):
                status = "✅" if assessment['is_relevant'] else "❌"
                print(f"\n   {i}. {status} {assessment['title']}")
                print(f"      🔗 {assessment['url']}")
                print(f"      📈 Score: {assessment['relevance_score']:.2f}")
                print(f"      💭 Reason: {assessment['reason']}")
                print(f"      📝 Preview: {assessment['content_preview']}...")
        
        # Final source summary
        print(f"\n📚 FINAL SOURCES:")
        print(f"   ✅ Accepted: {len(self.debug_info['accepted_sources'])}")
        print(f"   ❌ Rejected: {len(self.debug_info['rejected_sources'])}")
        
        if self.debug_info['accepted_sources']:
            print(f"\n✅ ACCEPTED SOURCES:")
            for i, source in enumerate(self.debug_info['accepted_sources'], 1):
                print(f"   {i}. {source['title']} (Score: {source['relevance_score']:.2f})")
                print(f"      🔗 {source['url']}")
                print(f"      ✅ {source['reason']}")
        
        if self.debug_info['rejected_sources']:
            print(f"\n❌ REJECTED SOURCES:")
            for i, source in enumerate(self.debug_info['rejected_sources'], 1):
                print(f"   {i}. {source['title']} (Score: {source['relevance_score']:.2f})")
                print(f"      🔗 {source['url']}")
                print(f"      ❌ {source['reason']}")
        
        print("\n" + "="*80)
        print("🔍 END DEBUG SUMMARY")
        print("="*80)

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query to generate appropriate research plan"""
        query_lower = query.lower()
        
        # Person indicators
        person_indicators = [
            'who is', 'biography of', 'background of', 'profile of',
            'researcher', 'scientist', 'professor', 'dr.', 'phd'
        ]
        
        # Concept/idea indicators  
        concept_indicators = [
            'what is', 'explain', 'definition of', 'concept of', 'theory of',
            'how does', 'why does', 'principle of', 'mechanism of'
        ]
        
        # Location/country indicators
        location_indicators = [
            'country', 'city', 'region', 'geography of', 'history of',
            'economy of', 'politics of', 'culture of'
        ]
        
        # Technology/product indicators
        technology_indicators = [
            'technology', 'software', 'algorithm', 'method', 'technique',
            'system', 'platform', 'tool', 'framework'
        ]
        
        # Company/organization indicators
        organization_indicators = [
            'company', 'organization', 'institution', 'startup', 'business',
            'corporation', 'agency', 'foundation'
        ]
        
        # Check for patterns
        if any(indicator in query_lower for indicator in person_indicators):
            return "person"
        elif any(indicator in query_lower for indicator in concept_indicators):
            return "concept"
        elif any(indicator in query_lower for indicator in location_indicators):
            return "location"
        elif any(indicator in query_lower for indicator in technology_indicators):
            return "technology"
        elif any(indicator in query_lower for indicator in organization_indicators):
            return "organization"
        else:
            # Default based on query structure
            words = query_lower.split()
            if len(words) <= 3 and any(word.istitle() for word in query.split()):
                return "person"  # Likely a name
            else:
                return "concept"  # General topic

    def _get_focus_areas_by_type(self, query_type: str) -> List[str]:
        """Get appropriate focus areas based on query type"""
        focus_areas_map = {
            "person": [
                "Professional Biography", "Academic Output", "Industry Impact", 
                "Public Presence", "Professional Affiliations"
            ],
            "concept": [
                "Definition & Overview", "Historical Development", "Key Applications",
                "Current Research", "Future Implications"
            ],
            "location": [
                "Geography & Demographics", "History & Culture", "Economy & Politics",
                "Current Events", "International Relations"
            ],
            "technology": [
                "Technical Overview", "Development History", "Current Applications",
                "Market Analysis", "Future Trends"
            ],
            "organization": [
                "Company Overview", "Business Model", "Leadership & History",
                "Market Position", "Recent Developments"
            ]
        }
        
        return focus_areas_map.get(query_type, focus_areas_map["concept"])

    def _extract_urls_from_search(self, search_results: str) -> List[tuple]:
        """Extract URLs and titles from search results"""
        urls = []
        lines = search_results.split('\n')
        
        current_title = ""
        for line in lines:
            line = line.strip()
            
            # Look for numbered results (1., 2., etc.)
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                current_title = line[2:].strip()
            
            # Look for URLs with link emoji
            elif line.startswith('🔗'):
                url = line.replace('🔗', '').strip()
                if url.startswith('http'):
                    urls.append((url, current_title or "Web Result"))
            
            # Also look for direct URLs in the text (fallback)
            elif 'http' in line and ('://' in line):
                import re
                url_matches = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', line)
                for url in url_matches:
                    # Clean up URL (remove trailing punctuation)
                    url = url.rstrip('.,;:!?)')
                    if url not in [u[0] for u in urls]:  # Avoid duplicates
                        title = current_title or f"Web Result from {url.split('/')[2]}"
                        urls.append((url, title))
        
        # If no URLs found, try a more aggressive search
        if not urls:
            import re
            all_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', search_results)
            for url in all_urls:
                url = url.rstrip('.,;:!?)')
                title = f"Web Result from {url.split('/')[2] if '/' in url else 'Unknown'}"
                urls.append((url, title))
        
        logger.debug(f"🔗 URL extraction found {len(urls)} URLs: {[u[0] for u in urls[:3]]}")
        return urls

    def _extract_relevant_content(self, content: str, query: str) -> str:
        """Extract relevant content from fetched web page using structured parsing"""
        
        # First, try to parse the structured output from fetch_url
        structured_content = self._parse_fetch_url_output(content)
        
        if structured_content:
            # Use structured data for more efficient extraction
            return self._extract_from_structured_content(structured_content, query)
        else:
            # Fallback to LLM-based extraction for unstructured content
            return self._extract_with_llm(content, query)
    
    def _parse_fetch_url_output(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse structured output from fetch_url tool"""
        try:
            # Look for the structured sections in fetch_url output
            if "📄 Content Analysis:" not in content:
                return None
            
            structured = {}
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Extract title
                if line.startswith('📰 Title:'):
                    structured['title'] = line.replace('📰 Title:', '').strip()
                
                # Extract description
                elif line.startswith('📝 Description:'):
                    structured['description'] = line.replace('📝 Description:', '').strip()
                
                # Extract headings
                elif line.startswith('📋 Headings'):
                    headings = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith('•'):
                        heading = lines[j].strip().replace('• ', '')
                        headings.append(heading)
                        j += 1
                    structured['headings'] = headings
                
                # Extract text content preview
                elif line.startswith('📄 Text Content Preview:'):
                    # Collect multiple lines of text content
                    text_lines = []
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith(('📊', '📄', '🔗', '📋', '📰', '📝')):
                        if lines[j].strip():  # Skip empty lines
                            text_lines.append(lines[j].strip())
                        j += 1
                    if text_lines:
                        structured['text_preview'] = ' '.join(text_lines)
            
            # Store raw content for full text extraction if needed
            if self.full_text_extraction:
                structured['_raw_content'] = content
            
            return structured if structured else None
            
        except Exception as e:
            logger.debug(f"Failed to parse fetch_url output: {e}")
            return None
    
    def _extract_from_structured_content(self, structured: Dict[str, Any], query: str) -> str:
        """Extract relevant information from structured content"""
        
        # Build content summary from structured data
        content_parts = []
        
        # Add title if relevant
        title = structured.get('title', '')
        if title and any(word.lower() in title.lower() for word in query.split()):
            content_parts.append(f"**Title:** {title}")
        
        # Add description if available
        description = structured.get('description', '')
        if description:
            content_parts.append(f"**Summary:** {description}")
        
        # Add relevant headings
        headings = structured.get('headings', [])
        relevant_headings = []
        query_words = [word.lower() for word in query.split()]
        
        for heading in headings[:10]:  # Limit to first 10 headings
            if any(word in heading.lower() for word in query_words):
                relevant_headings.append(heading)
        
        if relevant_headings:
            content_parts.append(f"**Key Sections:** {'; '.join(relevant_headings[:5])}")
        
        # Add text preview (longer or full text based on mode)
        text_preview = structured.get('text_preview', '')
        if text_preview:
            if self.full_text_extraction:
                # In full text mode, try to get more content from fetch_url
                full_text = self._extract_full_text_from_fetch_output(structured.get('_raw_content', ''))
                if full_text and len(full_text) > len(text_preview):
                    content_parts.append(f"**Full Content:** {full_text}")
                else:
                    content_parts.append(f"**Content:** {text_preview}")
            else:
                # Standard mode: use longer preview (up to 1000 chars)
                preview = text_preview[:1000] + "..." if len(text_preview) > 1000 else text_preview
                content_parts.append(f"**Content:** {preview}")
        
        if not content_parts:
            return None
        
        # Combine and validate relevance
        combined_content = '\n'.join(content_parts)
        
        # Quick relevance check - if query words appear in the content
        query_words_lower = [word.lower() for word in query.split() if len(word) > 2]
        content_lower = combined_content.lower()
        
        relevance_score = sum(1 for word in query_words_lower if word in content_lower) / len(query_words_lower)
        
        if relevance_score < 0.2:  # Less than 20% of query words found
            return None
        
        return combined_content
    
    def _extract_full_text_from_fetch_output(self, raw_content: str) -> str:
        """Extract full clean text content from fetch_url output"""
        if not raw_content or "📄 Text Content Preview:" not in raw_content:
            return ""
        
        try:
            # Find the text content section
            lines = raw_content.split('\n')
            text_lines = []
            in_text_section = False
            
            for line in lines:
                line_stripped = line.strip()
                
                # Start collecting after "Text Content Preview:"
                if line_stripped.startswith('📄 Text Content Preview:'):
                    in_text_section = True
                    continue
                
                # Stop at next section or metadata
                elif in_text_section and line_stripped.startswith(('📊', '📄', '🔗', '📋', '📰', '📝', '⏰', '✅')):
                    break
                
                # Collect text lines
                elif in_text_section and line_stripped:
                    # Skip obvious metadata or navigation
                    if not any(skip in line_stripped.lower() for skip in [
                        'total text length:', 'characters', 'download image', 
                        'press inquiries', 'contact:', 'email:', 'phone:', 
                        'breadcrumb', 'navigation', 'menu', 'footer'
                    ]):
                        text_lines.append(line_stripped)
            
            if text_lines:
                full_text = ' '.join(text_lines)
                # Clean up excessive whitespace
                full_text = ' '.join(full_text.split())
                return full_text
            
            return ""
            
        except Exception as e:
            logger.debug(f"Failed to extract full text: {e}")
            return ""
    
    def _extract_relevant_content_full_text(self, content: str, query: str, url: str) -> str:
        """Extract relevant content using full text mode with custom processing"""
        
        # First try structured parsing
        structured_content = self._parse_fetch_url_output(content)
        
        if structured_content:
            # Get the full text if available
            full_text = self._extract_full_text_from_fetch_output(content)
            
            if full_text and len(full_text) > 200:
                # Use LLM to extract relevant parts from the full text
                llm_result = self._extract_with_llm_full_text(full_text, query)
                if llm_result:
                    return llm_result
            
            # Always try structured extraction as fallback
            structured_result = self._extract_from_structured_content(structured_content, query)
            if structured_result:
                return structured_result
        
        # Final fallback to standard LLM extraction
        return self._extract_with_llm(content, query)
    
    def _extract_with_llm_full_text(self, full_text: str, query: str) -> str:
        """Extract relevant content from full text using LLM"""
        
        # Limit content length for LLM processing (but allow more than standard mode)
        max_length = 15000  # 3x more than standard mode
        if len(full_text) > max_length:
            # Try to truncate at sentence boundary
            truncated = full_text[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length - 500:  # If period is reasonably close to end
                full_text = truncated[:last_period + 1]
            else:
                full_text = truncated + "..."
        
        extraction_prompt = f"""
Extract the most relevant and comprehensive information from this full text content for the research query.

RESEARCH QUERY: {query}

FULL TEXT CONTENT:
{full_text}

Extract 3-5 key points that directly answer or relate to the research query. 
Focus on:
- Specific facts, data, statistics, and recent developments
- Technical details and performance metrics
- Key findings and authoritative statements
- Recent breakthroughs or announcements
- Comparative information and benchmarks

Format as a comprehensive summary (max 800 words) with the most important information.
Include specific details like numbers, dates, company names, and technical specifications.
If the content is not relevant to the query, respond with "NOT_RELEVANT".
"""

        try:
            response = self.llm.generate(extraction_prompt, temperature=0.2)
            
            # Extract text from response (handle different response object types)
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            extracted = response_text.strip()
            
            if extracted == "NOT_RELEVANT" or len(extracted) < 100:
                return None
                
            return extracted
            
        except Exception as e:
            logger.debug(f"Full text extraction failed: {e}")
            return None
    
    def _extract_with_llm(self, content: str, query: str) -> str:
        """Fallback LLM-based extraction for unstructured content"""
        
        # Limit content length for processing
        if len(content) > 8000:
            content = content[:8000] + "..."
        
        extraction_prompt = f"""
Extract the most relevant information from this content for the research query.

RESEARCH QUERY: {query}

CONTENT:
{content}

Extract 2-3 key points that directly answer or relate to the research query. 
Focus on facts, data, recent developments, and authoritative statements.
Ignore navigation, ads, and irrelevant content.

Format as a concise summary (max 300 words) with the most important information.
If the content is not relevant to the query, respond with "NOT_RELEVANT".
"""

        try:
            response = self.llm.generate(extraction_prompt, temperature=0.2)
            
            # Extract text from response (handle different response object types)
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            extracted = response_text.strip()
            
            if extracted == "NOT_RELEVANT" or len(extracted) < 50:
                return None
                
            return extracted
            
        except Exception as e:
            logger.debug(f"Content extraction failed: {e}")
            return None

    def _generate_report(
        self, 
        research_plan: ResearchPlan, 
        findings: List[ResearchFinding], 
        output_format: str
    ) -> ResearchReport:
        """Stage 4: Generate structured research report"""
        
        # Check if we have any findings
        if not findings:
            logger.warning("⚠️ No findings available for report generation")
            return self._create_no_findings_report(research_plan)
        
        # Organize findings by sub-task
        findings_by_task = {}
        for finding in findings:
            task_id = finding.sub_task_id
            if task_id not in findings_by_task:
                findings_by_task[task_id] = []
            findings_by_task[task_id].append(finding)
        
        # Collect research context for specific methodology/limitations
        search_queries_used = []
        for sub_task in research_plan.sub_tasks:
            if sub_task.search_queries:
                search_queries_used.extend(sub_task.search_queries)
        
        successful_extractions = len([f for f in findings if f.content and len(f.content.strip()) > 50])
        total_sources_attempted = len(findings)
        failed_extractions = total_sources_attempted - successful_extractions
        
        research_context = {
            'total_sources_found': total_sources_attempted,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'search_queries_used': search_queries_used,
            'extraction_method': 'full_text' if self.full_text_extraction else 'preview',
            'focus_areas': research_plan.focus_areas,
            'sub_tasks_count': len(research_plan.sub_tasks)
        }
        
        # Prepare findings summary for the LLM
        findings_summary = []
        total_findings_count = 0
        
        for sub_task in research_plan.sub_tasks:
            task_findings = findings_by_task.get(sub_task.id, [])
            if task_findings:
                findings_summary.append(f"\n## {sub_task.question}")
                for finding in task_findings:
                    findings_summary.append(f"- {finding.content}")
                    findings_summary.append(f"  Source: {finding.title} ({finding.source_url})")
                    total_findings_count += 1
        
        findings_text = "\n".join(findings_summary)
        
        if not findings_text.strip():
            logger.warning("⚠️ No usable findings content for report generation")
            return self._create_no_findings_report(research_plan)
        
        logger.info(f"📝 Generating report from {total_findings_count} findings across {len(findings_by_task)} sub-tasks")
        
        # Generate report based on format
        if output_format == "executive":
            report_prompt = self._get_executive_report_prompt(research_plan, findings_text, research_context)
        elif output_format == "narrative":
            report_prompt = self._get_narrative_report_prompt(research_plan, findings_text, research_context)
        else:  # structured
            report_prompt = self._get_structured_report_prompt(research_plan, findings_text, research_context)
        
        try:
            response = self.llm.generate(report_prompt, temperature=0.3)
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            # Extract text from response (handle different response object types)
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            elif hasattr(response, 'content'):
                response_text = response.content.strip()
            else:
                response_text = str(response).strip()
            
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                logger.debug(f"📄 Extracted JSON: {json_text[:200]}...")
                report_data = json.loads(json_text)
            else:
                logger.warning("⚠️ No JSON found in LLM response, using fallback")
                raise json.JSONDecodeError("No JSON found", response_text, 0)
            
            # Create sources list
            sources = []
            for finding in findings:
                source_entry = {
                    "title": finding.title,
                    "url": finding.source_url,
                    "relevance": finding.relevance_score
                }
                if source_entry not in sources:
                    sources.append(source_entry)
            
            # Validate and enhance citations in the generated content
            detailed_analysis = report_data.get("detailed_analysis", "")
            key_findings = report_data.get("key_findings", [])
            
            # Validate citations in detailed analysis
            citation_validation = CitationValidator.validate_citations(detailed_analysis, sources)
            logger.info(f"📊 Citation validation: {citation_validation['citations_found']} citations found, "
                       f"{citation_validation['citation_ratio']:.2f} ratio, "
                       f"adequately cited: {citation_validation['is_adequately_cited']}")
            
            # Enhance content if citations are insufficient
            if not citation_validation['is_adequately_cited']:
                logger.warning("⚠️ Insufficient citations detected, enhancing content")
                detailed_analysis = CitationValidator.enhance_text_with_citations(detailed_analysis, sources)
                
                # Also enhance key findings if they lack citations
                enhanced_findings = []
                for finding in key_findings:
                    if isinstance(finding, str):
                        finding_validation = CitationValidator.validate_citations(finding, sources)
                        if finding_validation['citations_found'] == 0:
                            enhanced_finding = CitationValidator.enhance_text_with_citations(finding, sources[:2])  # Limit to top 2 sources
                            enhanced_findings.append(enhanced_finding)
                        else:
                            enhanced_findings.append(finding)
                    else:
                        enhanced_findings.append(finding)
                key_findings = enhanced_findings
            
            # Ensure all fields are properly formatted for Pydantic validation
            def ensure_string(value, default=""):
                """Convert list or other types to string"""
                if isinstance(value, list):
                    return " ".join(str(item) for item in value)
                elif value is None:
                    return default
                else:
                    return str(value)
            
            def ensure_list(value, default=None):
                """Ensure value is a list"""
                if default is None:
                    default = []
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    return [value] if value else default
                else:
                    return default
            
            report = ResearchReport(
                title=ensure_string(report_data.get("title"), f"Research Report: {research_plan.original_query}"),
                executive_summary=ensure_string(report_data.get("executive_summary"), ""),
                key_findings=ensure_list(key_findings, []),
                detailed_analysis=ensure_string(detailed_analysis, ""),
                conclusions=ensure_string(report_data.get("conclusions"), ""),
                sources=sources,
                methodology=ensure_string(report_data.get("methodology"), "Web-based research using multi-stage pipeline"),
                limitations=ensure_string(report_data.get("limitations"), "Limited to publicly available web sources")
            )
            
            return report
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse report: {e}")
            # Return fallback report
            return self._create_fallback_report(research_plan, findings)

    def _create_no_findings_report(self, research_plan: ResearchPlan) -> ResearchReport:
        """Create a report when no findings are available"""
        
        return ResearchReport(
            title=f"Research Report: {research_plan.original_query} (No Sources Found)",
            executive_summary="Research could not be completed due to inability to access web sources. This may be due to network connectivity issues, search service limitations, or content access restrictions.",
            key_findings=[
                "No web sources could be accessed for this research query",
                "Search functionality may be limited due to network or service issues",
                "Manual research using alternative sources is recommended"
            ],
            detailed_analysis="The automated research process was unable to gather information from web sources for this query. This could be due to several factors: network connectivity issues preventing access to search services, search API limitations or rate limiting, content access restrictions, or the specific query terms not yielding accessible results. To complete this research, manual investigation using academic databases, library resources, or direct access to authoritative sources would be recommended.",
            conclusions="Automated web-based research could not be completed for this query. Alternative research methods should be employed to gather the required information.",
            sources=[],
            methodology="Attempted web-based research using multi-stage pipeline with parallel search execution. Search services were inaccessible or returned no usable results.",
            limitations="Complete inability to access web sources severely limits the scope and reliability of this research. No factual claims can be made without source verification."
        )

    def _get_structured_report_prompt(self, research_plan: ResearchPlan, findings_text: str, research_context: Dict[str, Any] = None) -> str:
        """Get prompt for structured report generation"""
        
        # Build research context information
        context_info = ""
        if research_context:
            total_sources = research_context.get('total_sources_found', 0)
            successful_extractions = research_context.get('successful_extractions', 0)
            failed_extractions = research_context.get('failed_extractions', 0)
            search_queries_used = research_context.get('search_queries_used', [])
            extraction_method = research_context.get('extraction_method', 'standard')
            
            context_info = f"""
RESEARCH PROCESS CONTEXT:
- Total sources discovered: {total_sources}
- Successfully analyzed: {successful_extractions}
- Failed to access: {failed_extractions}
- Extraction method: {extraction_method}
- Search queries executed: {len(search_queries_used)}
- Key search terms: {', '.join(search_queries_used[:5]) if search_queries_used else 'None'}
"""

        return f"""
Generate a comprehensive research report based on the findings below. 

IMPORTANT: Respond with ONLY valid JSON, no additional text before or after.

RESEARCH OBJECTIVE: {research_plan.research_objective}
ORIGINAL QUERY: {research_plan.original_query}
{context_info}
RESEARCH FINDINGS:
{findings_text}

Create a structured research report with this EXACT JSON format:
{{
    "title": "Descriptive report title about {research_plan.original_query}",
    "executive_summary": "2-3 sentence summary of key insights from the research findings",
    "key_findings": [
        "Key finding 1 with specific details and citation (according to Source Name)",
        "Key finding 2 with specific details and citation (according to Source Name)",
        "Key finding 3 with specific details and citation (according to Source Name)"
    ],
    "detailed_analysis": "Comprehensive analysis section (3-4 paragraphs) that synthesizes the findings, identifies patterns, and provides context. MUST include citations like 'according to [Source Name]' or 'as reported by [Source Name]' for every claim and fact",
    "conclusions": "Clear conclusions and implications (2-3 paragraphs) based on the evidence gathered",
    "methodology": "Detailed description of the specific research approach used, including: search strategies employed, number of sources analyzed, types of sources accessed, any challenges encountered, and verification methods applied",
    "limitations": "Specific limitations encountered during THIS research, including: sources that were inaccessible, information gaps identified, potential biases in available sources, temporal constraints, and areas requiring further investigation"
}}

CRITICAL REQUIREMENTS:
- Respond with ONLY the JSON object, no other text
- Base all content strictly on the provided findings
- ALWAYS include proper citations for every claim using source titles or "according to [Source]"
- Include specific facts, data, and examples from the sources WITH citations
- Use proper JSON formatting with escaped quotes if needed
- Do not include markdown formatting or code blocks
- Every key finding and analysis point MUST reference its source
"""

    def _get_executive_report_prompt(self, research_plan: ResearchPlan, findings_text: str, research_context: Dict[str, Any] = None) -> str:
        """Get prompt for executive report generation"""
        
        # Build research context information
        context_info = ""
        if research_context:
            total_sources = research_context.get('total_sources_found', 0)
            successful_extractions = research_context.get('successful_extractions', 0)
            failed_extractions = research_context.get('failed_extractions', 0)
            search_queries_used = research_context.get('search_queries_used', [])
            extraction_method = research_context.get('extraction_method', 'standard')
            
            context_info = f"""
RESEARCH PROCESS CONTEXT:
- Total sources discovered: {total_sources}
- Successfully analyzed: {successful_extractions}
- Failed to access: {failed_extractions}
- Extraction method: {extraction_method}
- Search queries executed: {len(search_queries_used)}
- Key search terms: {', '.join(search_queries_used[:5]) if search_queries_used else 'None'}
"""

        return f"""
Generate a concise executive research report based on the findings below.

RESEARCH OBJECTIVE: {research_plan.research_objective}
ORIGINAL QUERY: {research_plan.original_query}
{context_info}
RESEARCH FINDINGS:
{findings_text}

Create an executive-style report with the following JSON format:
{{
    "title": "Executive Brief: [Topic]",
    "executive_summary": "3-4 sentence executive summary highlighting the most critical insights and implications",
    "key_findings": [
        "3-5 bullet points with the most important findings",
        "Focus on actionable insights and strategic implications"
    ],
    "detailed_analysis": "2-3 paragraphs of focused analysis on the most critical aspects",
    "conclusions": "Clear, actionable conclusions and recommendations",
    "methodology": "Concise description of research approach: sources analyzed, search methods used, and verification applied",
    "limitations": "Specific limitations from this research: inaccessible sources, information gaps, or areas needing further study"
}}

Guidelines:
- Focus on strategic insights and business implications
- Prioritize actionable information
- Keep language clear and executive-friendly
- Highlight trends, opportunities, and risks
- Be concise but comprehensive
- ALWAYS cite sources for every claim using "according to [Source]" or similar
- Include proper attribution for all facts and data points
"""

    def _get_narrative_report_prompt(self, research_plan: ResearchPlan, findings_text: str, research_context: Dict[str, Any] = None) -> str:
        """Get prompt for narrative report generation"""
        
        # Build research context information
        context_info = ""
        if research_context:
            total_sources = research_context.get('total_sources_found', 0)
            successful_extractions = research_context.get('successful_extractions', 0)
            failed_extractions = research_context.get('failed_extractions', 0)
            search_queries_used = research_context.get('search_queries_used', [])
            extraction_method = research_context.get('extraction_method', 'standard')
            
            context_info = f"""
RESEARCH PROCESS CONTEXT:
- Total sources discovered: {total_sources}
- Successfully analyzed: {successful_extractions}
- Failed to access: {failed_extractions}
- Extraction method: {extraction_method}
- Search queries executed: {len(search_queries_used)}
- Key search terms: {', '.join(search_queries_used[:5]) if search_queries_used else 'None'}
"""

        return f"""
Generate a narrative research report based on the findings below.

RESEARCH OBJECTIVE: {research_plan.research_objective}
ORIGINAL QUERY: {research_plan.original_query}
{context_info}
RESEARCH FINDINGS:
{findings_text}

Create a narrative-style report with the following JSON format:
{{
    "title": "Research Report: [Topic]",
    "executive_summary": "Engaging summary that tells the story of what was discovered",
    "key_findings": [
        "Key discoveries presented as narrative points",
        "Each finding should tell part of the overall story"
    ],
    "detailed_analysis": "Comprehensive narrative analysis (4-5 paragraphs) that weaves together the findings into a coherent story, showing how different aspects connect and build upon each other",
    "conclusions": "Narrative conclusions that bring the story together and point toward future implications",
    "methodology": "Narrative account of the research journey: what sources were explored, how information was gathered, challenges faced, and methods used to verify findings",
    "limitations": "Honest reflection on what this specific research couldn't uncover: missing perspectives, inaccessible information, temporal constraints, and areas requiring deeper investigation"
}}

Guidelines:
- Write in an engaging, narrative style
- Show connections and relationships between findings
- Use storytelling techniques to make the report compelling
- Maintain objectivity while being engaging
- Create a logical flow from introduction to conclusion
- ALWAYS include proper citations throughout the narrative using source titles
- Attribute all facts, quotes, and data to their specific sources
"""

    def _create_fallback_report(self, research_plan: ResearchPlan, findings: List[ResearchFinding]) -> ResearchReport:
        """Create a simple fallback report if JSON parsing fails"""
        
        # Extract key information from findings
        key_findings = []
        sources = []
        
        for finding in findings[:10]:  # Limit to top 10 findings
            key_findings.append(finding.content[:200] + "..." if len(finding.content) > 200 else finding.content)
            sources.append({
                "title": finding.title,
                "url": finding.source_url,
                "relevance": finding.relevance_score
            })
        
        return ResearchReport(
            title=f"Research Report: {research_plan.original_query}",
            executive_summary=f"Research conducted on {research_plan.original_query} with {len(findings)} sources analyzed.",
            key_findings=key_findings,
            detailed_analysis="Detailed analysis could not be generated due to processing error. Please refer to key findings and sources.",
            conclusions="Further analysis recommended based on the gathered sources.",
            sources=sources,
            methodology="Web-based research using multi-stage pipeline",
            limitations="Limited to publicly available web sources. Report generation encountered technical issues."
        )

    def _verify_report(self, report: ResearchReport, findings: List[ResearchFinding]) -> ResearchReport:
        """Stage 5: Verify report accuracy and add fact-checking"""
        
        verification_prompt = f"""
Review this research report for accuracy and consistency with the source findings.

REPORT TITLE: {report.title}
EXECUTIVE SUMMARY: {report.executive_summary}
KEY FINDINGS: {report.key_findings}

SOURCE FINDINGS:
{[f"- {f.content[:200]}..." for f in findings[:10]]}

Identify any potential issues:
1. Claims not supported by the source findings
2. Overgeneralizations or unsupported conclusions
3. Missing important caveats or limitations
4. Factual inconsistencies

Provide verification results as JSON:
{{
    "verification_status": "verified|needs_review|issues_found",
    "issues_identified": ["list of specific issues if any"],
    "confidence_score": 0.85,
    "recommendations": ["suggestions for improvement"]
}}
"""

        try:
            response = self.llm.generate(verification_prompt, temperature=0.2)
            
            # Extract text from response (handle different response object types)
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            verification = json.loads(response_text)
            
            # Add verification metadata to report
            if hasattr(report, 'metadata'):
                report.metadata = {}
            
            # Update limitations if issues were found
            if verification.get("verification_status") == "issues_found":
                issues = verification.get("issues_identified", [])
                additional_limitations = f" Verification identified potential issues: {'; '.join(issues)}"
                report.limitations += additional_limitations
            
            logger.info(f"✅ Report verification completed: {verification.get('verification_status', 'unknown')}")
            
        except Exception as e:
            logger.warning(f"Report verification failed: {e}")
            report.limitations += " Report verification could not be completed."
        
        return report

    def _reflexive_refinement(
        self, 
        initial_report: ResearchReport, 
        research_plan: ResearchPlan, 
        existing_findings: List[ResearchFinding]
    ) -> ResearchReport:
        """Stage 6: Reflexive analysis and iterative improvement"""
        
        current_report = initial_report
        current_findings = existing_findings.copy()
        
        for iteration in range(self.max_reflexive_iterations):
            logger.info(f"🔄 Reflexive iteration {iteration + 1}/{self.max_reflexive_iterations}")
            
            # Analyze gaps and limitations
            gaps = self._analyze_research_gaps(current_report, research_plan)
            
            if not gaps or len(gaps) == 0:
                logger.info("✅ No significant gaps identified - reflexive analysis complete")
                break
                
            logger.info(f"🎯 Identified {len(gaps)} research gaps to address")
            
            # Execute targeted searches for gaps
            new_findings = self._execute_gap_searches(gaps, research_plan.original_query)
            
            if new_findings:
                logger.info(f"📚 Found {len(new_findings)} additional sources")
                current_findings.extend(new_findings)
                
                # Regenerate report with enhanced findings
                current_report = self._generate_report(research_plan, current_findings, "structured")
                
                # Update methodology to reflect reflexive process
                current_report.methodology += f" Enhanced through {iteration + 1} reflexive analysis cycle(s) addressing identified gaps."
            else:
                logger.info("⚠️ No additional sources found for identified gaps")
                break
        
        return current_report

    def _analyze_research_gaps(self, report: ResearchReport, research_plan: ResearchPlan) -> List[Dict[str, Any]]:
        """Analyze methodology and limitations to identify actionable research gaps"""
        
        gap_analysis_prompt = f"""
Analyze this research report to identify SPECIFIC, ACTIONABLE information gaps that could be addressed with targeted web searches.

ORIGINAL QUERY: {research_plan.original_query}

CURRENT METHODOLOGY: {report.methodology}

CURRENT LIMITATIONS: {report.limitations}

CURRENT KEY FINDINGS: {report.key_findings}

FOCUS AREAS EXPLORED: {research_plan.focus_areas}

Identify gaps that are:
1. SPECIFIC enough to generate targeted search queries
2. IMPORTANT for answering the original query
3. FEASIBLE to find through web search
4. NOT already covered in current findings

For each gap, provide:
- gap_type: "missing_perspective", "insufficient_data", "outdated_info", "alternative_viewpoint", "technical_detail", "recent_development"
- description: What specific information is missing
- importance: 1-10 scale (10 = critical for answering original query)
- search_strategy: Specific approach to find this information
- target_queries: 2-3 specific search queries to address this gap

Return ONLY a JSON array of gaps (max 5 most important):
[
  {{
    "gap_type": "missing_perspective",
    "description": "Lack of industry expert opinions on quantum computing timeline",
    "importance": 8,
    "search_strategy": "Search for expert interviews, industry reports, and analyst predictions",
    "target_queries": ["quantum computing expert predictions 2024", "industry analysis quantum timeline", "quantum computing roadmap enterprise"]
  }}
]

CRITICAL: Return ONLY the JSON array, no other text.
"""
        
        try:
            response = self.llm.generate(gap_analysis_prompt)
            
            # Extract text from response (handle GenerateResponse objects)
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Extract JSON from response
            import json
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                gaps_data = json.loads(json_match.group())
                
                # Filter gaps by importance (only keep high-importance ones)
                important_gaps = [gap for gap in gaps_data if gap.get('importance', 0) >= 6]
                
                logger.info(f"🔍 Gap analysis identified {len(important_gaps)} high-priority gaps")
                return important_gaps
            else:
                logger.warning("No valid JSON found in gap analysis response")
                return []
                
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return []

    def _execute_gap_searches(self, gaps: List[Dict[str, Any]], original_query: str) -> List[ResearchFinding]:
        """Execute targeted searches to address identified gaps"""
        
        new_findings = []
        
        for gap in gaps:
            gap_type = gap.get('gap_type', 'unknown')
            description = gap.get('description', '')
            target_queries = gap.get('target_queries', [])
            
            logger.info(f"🎯 Addressing gap: {gap_type} - {description}")
            
            # Execute searches for this gap
            for query in target_queries[:2]:  # Limit to 2 queries per gap
                try:
                    logger.info(f"🔍 Gap search: {query}")
                    
                    # Use existing search infrastructure
                    gap_findings = self._execute_search(query, f"gap_{gap_type}")
                    
                    if gap_findings:
                        # Mark findings as gap-addressing
                        for finding in gap_findings:
                            finding.sub_task_id = f"reflexive_gap_{gap_type}"
                            
                        new_findings.extend(gap_findings)
                        logger.info(f"✅ Found {len(gap_findings)} sources for gap: {description}")
                    
                except Exception as e:
                    logger.warning(f"Gap search failed for '{query}': {e}")
                    continue
        
        return new_findings
