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
        max_parallel_searches: int = 5
    ):
        """Initialize the deep search system
        
        Args:
            llm: AbstractCore instance (any provider). If None, uses default Ollama model
            max_tokens: Maximum total tokens for LLM context (default 32000)
            max_output_tokens: Maximum tokens for LLM output generation (default 8000)
            timeout: HTTP request timeout in seconds. None for unlimited timeout (default None)
            max_parallel_searches: Maximum number of parallel web searches (default 5)
        """
        if llm is None:
            try:
                self.llm = create_llm(
                    "ollama", 
                    model="qwen3:4b-instruct-2507-q4_K_M", 
                    max_tokens=max_tokens, 
                    max_output_tokens=max_output_tokens, 
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
        self.retry_strategy = FeedbackRetry(max_attempts=3)

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
            # Stage 1: Planning
            logger.info("📋 Stage 1: Planning research approach...")
            research_plan = self._create_research_plan(query, focus_areas, search_depth)
            
            # Stage 2: Question Development
            logger.info("❓ Stage 2: Developing search questions...")
            self._develop_search_questions(research_plan, max_sources)
            
            # Stage 3: Web Exploration
            logger.info("🌐 Stage 3: Exploring web sources...")
            findings = self._explore_web_sources(research_plan)
            
            # Stage 4: Report Generation
            logger.info("📝 Stage 4: Generating research report...")
            report = self._generate_report(research_plan, findings, output_format)
            
            # Optional: Verification
            if include_verification:
                logger.info("✅ Stage 5: Verifying findings...")
                report = self._verify_report(report, findings)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✨ Deep search completed in {elapsed_time:.1f} seconds")
            
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
        
        # Determine number of sub-tasks based on search depth
        depth_config = {
            "brief": {"sub_tasks": 3, "time_estimate": 5},
            "standard": {"sub_tasks": 5, "time_estimate": 10},
            "comprehensive": {"sub_tasks": 8, "time_estimate": 20}
        }
        config = depth_config.get(search_depth, depth_config["standard"])
        
        planning_prompt = f"""
You are a research planning expert. Create a structured research plan for the following query.

RESEARCH QUERY: {query}
FOCUS AREAS: {focus_areas or "Not specified - determine automatically"}
SEARCH DEPTH: {search_depth}
TARGET SUB-TASKS: {config['sub_tasks']}

Create a research plan with the following structure:

1. RESEARCH OBJECTIVE: Clear, specific objective (1-2 sentences)
2. SUB-TASKS: Break down into {config['sub_tasks']} specific research sub-tasks
3. FOCUS AREAS: Identify 3-5 key focus areas if not provided
4. SEARCH STRATEGY: Choose breadth_first, depth_first, or parallel

For each sub-task, provide:
- A clear, specific research question
- The focus area it addresses
- Priority level (1=high, 2=medium, 3=low)

Format your response as JSON with this structure:
{{
    "research_objective": "Clear research objective",
    "sub_tasks": [
        {{
            "id": "task_1",
            "question": "Specific research question",
            "focus_area": "Focus area name",
            "priority": 1
        }}
    ],
    "focus_areas": ["area1", "area2", "area3"],
    "search_strategy": "parallel",
    "estimated_time_minutes": {config['time_estimate']}
}}

Ensure questions are specific, actionable, and cover different aspects of the research topic.
"""

        try:
            response = self.llm.generate(planning_prompt, temperature=0.3)
            plan_data = json.loads(response.content)
            
            # Create ResearchSubTask objects
            sub_tasks = []
            for task_data in plan_data["sub_tasks"]:
                sub_task = ResearchSubTask(
                    id=task_data["id"],
                    question=task_data["question"],
                    focus_area=task_data["focus_area"],
                    priority=task_data.get("priority", 2),
                    search_queries=[],
                    findings=[],
                    status="pending"
                )
                sub_tasks.append(sub_task)
            
            research_plan = ResearchPlan(
                original_query=query,
                research_objective=plan_data["research_objective"],
                sub_tasks=sub_tasks,
                estimated_time_minutes=plan_data.get("estimated_time_minutes", config['time_estimate']),
                focus_areas=plan_data["focus_areas"],
                search_strategy=plan_data.get("search_strategy", "parallel")
            )
            
            logger.info(f"📋 Created research plan with {len(sub_tasks)} sub-tasks")
            return research_plan
            
        except (json.JSONDecodeError, KeyError) as e:
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

Format as a JSON list of strings:
["query 1", "query 2", "query 3"]

Examples of good search queries for quantum computing:
- "quantum computing hardware advances 2024"
- "quantum computer error correction breakthrough 2024"
- "IBM Google quantum computing progress 2024"
- "quantum computing industry applications 2024"
- "quantum supremacy achievements 2024"

Avoid generic terms like "qubit" alone (which returns lab instruments) - be specific about quantum computing context.
"""

            try:
                response = self.llm.generate(query_prompt, temperature=0.5)
                queries = json.loads(response.content)
                sub_task.search_queries = queries[:queries_per_task]
                logger.info(f"📝 Generated queries for {sub_task.id}: {sub_task.search_queries}")
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse queries for {sub_task.id}, using fallback")
                # Improved fallback queries with better specificity
                base_topic = research_plan.original_query.replace("What are the latest developments in ", "").replace("?", "")
                sub_task.search_queries = [
                    f"{base_topic} {sub_task.focus_area} 2024",
                    f"{base_topic} advances {sub_task.focus_area} 2024",
                    f"{base_topic} research {sub_task.focus_area} latest"
                ][:queries_per_task]
                logger.info(f"📝 Using fallback queries for {sub_task.id}: {sub_task.search_queries}")

    def _explore_web_sources(self, research_plan: ResearchPlan) -> List[ResearchFinding]:
        """Stage 3: Execute web searches and gather evidence"""
        
        all_findings = []
        
        # Collect all search queries with their sub-task context
        search_tasks = []
        for sub_task in research_plan.sub_tasks:
            for query in sub_task.search_queries:
                search_tasks.append((sub_task.id, query, sub_task.priority))
        
        # Sort by priority (1=high priority first)
        search_tasks.sort(key=lambda x: x[2])
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel_searches) as executor:
            # Submit search tasks
            future_to_task = {}
            for sub_task_id, query, priority in search_tasks:
                future = executor.submit(self._execute_search, sub_task_id, query)
                future_to_task[future] = (sub_task_id, query)
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                sub_task_id, query = future_to_task[future]
                try:
                    findings = future.result()
                    all_findings.extend(findings)
                    logger.debug(f"Completed search for {sub_task_id}: {query}")
                except Exception as e:
                    logger.warning(f"Search failed for {sub_task_id} '{query}': {e}")
        
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

    def _execute_search(self, sub_task_id: str, query: str) -> List[ResearchFinding]:
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
            
            # Fetch content from promising URLs
            for i, (url, title) in enumerate(urls[:3]):  # Limit to top 3 results per query
                try:
                    logger.debug(f"🌐 Fetching content from URL {i+1}: {url}")
                    content = fetch_url(url, timeout=15)
                    
                    if "Error" in content or len(content) < 100:
                        logger.debug(f"⚠️ Skipping URL due to fetch error or short content: {url}")
                        continue
                    
                    # Extract relevant content using LLM
                    relevant_content = self._extract_relevant_content(content, query)
                    
                    if relevant_content:
                        finding = ResearchFinding(
                            source_url=url,
                            title=title,
                            content=relevant_content,
                            relevance_score=0.8,  # Could be improved with semantic scoring
                            timestamp=timestamp,
                            sub_task_id=sub_task_id
                        )
                        findings.append(finding)
                        logger.info(f"✅ Created finding from {url}")
                    else:
                        logger.debug(f"⚠️ No relevant content extracted from {url}")
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Search execution failed for '{query}': {e}")
        
        logger.info(f"📊 Search completed for '{query}': {len(findings)} findings")
        return findings

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
        """Extract relevant content from fetched web page"""
        
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
            extracted = response.content.strip()
            
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
            report_prompt = self._get_executive_report_prompt(research_plan, findings_text)
        elif output_format == "narrative":
            report_prompt = self._get_narrative_report_prompt(research_plan, findings_text)
        else:  # structured
            report_prompt = self._get_structured_report_prompt(research_plan, findings_text)
        
        try:
            response = self.llm.generate(report_prompt, temperature=0.3)
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            response_text = response.content.strip()
            
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
            
            report = ResearchReport(
                title=report_data.get("title", f"Research Report: {research_plan.original_query}"),
                executive_summary=report_data.get("executive_summary", ""),
                key_findings=report_data.get("key_findings", []),
                detailed_analysis=report_data.get("detailed_analysis", ""),
                conclusions=report_data.get("conclusions", ""),
                sources=sources,
                methodology=report_data.get("methodology", "Web-based research using multi-stage pipeline"),
                limitations=report_data.get("limitations", "Limited to publicly available web sources")
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

    def _get_structured_report_prompt(self, research_plan: ResearchPlan, findings_text: str) -> str:
        """Get prompt for structured report generation"""
        
        return f"""
Generate a comprehensive research report based on the findings below. 

IMPORTANT: Respond with ONLY valid JSON, no additional text before or after.

RESEARCH OBJECTIVE: {research_plan.research_objective}
ORIGINAL QUERY: {research_plan.original_query}

RESEARCH FINDINGS:
{findings_text}

Create a structured research report with this EXACT JSON format:
{{
    "title": "Descriptive report title about {research_plan.original_query}",
    "executive_summary": "2-3 sentence summary of key insights from the research findings",
    "key_findings": [
        "Key finding 1 with specific details from sources",
        "Key finding 2 with specific details from sources",
        "Key finding 3 with specific details from sources"
    ],
    "detailed_analysis": "Comprehensive analysis section (3-4 paragraphs) that synthesizes the findings, identifies patterns, and provides context based on the research sources",
    "conclusions": "Clear conclusions and implications (2-3 paragraphs) based on the evidence gathered",
    "methodology": "Web-based research using multi-stage pipeline with parallel search execution",
    "limitations": "Research limitations and caveats including source accessibility and scope"
}}

CRITICAL REQUIREMENTS:
- Respond with ONLY the JSON object, no other text
- Base all content strictly on the provided findings
- Include specific facts, data, and examples from the sources
- Use proper JSON formatting with escaped quotes if needed
- Do not include markdown formatting or code blocks
"""

    def _get_executive_report_prompt(self, research_plan: ResearchPlan, findings_text: str) -> str:
        """Get prompt for executive report generation"""
        
        return f"""
Generate a concise executive research report based on the findings below.

RESEARCH OBJECTIVE: {research_plan.research_objective}
ORIGINAL QUERY: {research_plan.original_query}

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
    "methodology": "Brief methodology note",
    "limitations": "Key limitations to consider"
}}

Guidelines:
- Focus on strategic insights and business implications
- Prioritize actionable information
- Keep language clear and executive-friendly
- Highlight trends, opportunities, and risks
- Be concise but comprehensive
"""

    def _get_narrative_report_prompt(self, research_plan: ResearchPlan, findings_text: str) -> str:
        """Get prompt for narrative report generation"""
        
        return f"""
Generate a narrative research report based on the findings below.

RESEARCH OBJECTIVE: {research_plan.research_objective}
ORIGINAL QUERY: {research_plan.original_query}

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
    "methodology": "Story of how the research was conducted",
    "limitations": "Honest discussion of what the research couldn't uncover"
}}

Guidelines:
- Write in an engaging, narrative style
- Show connections and relationships between findings
- Use storytelling techniques to make the report compelling
- Maintain objectivity while being engaging
- Create a logical flow from introduction to conclusion
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
            verification = json.loads(response.content)
            
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
