"""
Basic Summarizer - Clean, powerful summarization built on AbstractCore

Demonstrates how to use AbstractCore's infrastructure to create sophisticated
text processing capabilities with minimal complexity.
"""

from enum import Enum
import json
import re
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

from ..core.interface import AbstractCoreInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


class SummaryStyle(Enum):
    """Summary presentation styles"""
    STRUCTURED = "structured"      # Bullet points, clear sections
    NARRATIVE = "narrative"        # Flowing, story-like prose
    OBJECTIVE = "objective"        # Neutral, factual tone
    ANALYTICAL = "analytical"      # Critical analysis with insights
    EXECUTIVE = "executive"        # Business-focused, action-oriented
    CONVERSATIONAL = "conversational"  # Chat history preservation with context


class SummaryLength(Enum):
    """Target summary lengths"""
    BRIEF = "brief"               # 2-3 sentences, key point only
    STANDARD = "standard"         # 1-2 paragraphs, main ideas
    DETAILED = "detailed"         # Multiple paragraphs, comprehensive
    COMPREHENSIVE = "comprehensive"  # Full analysis with context


class CompressionMode(Enum):
    """Compression aggressiveness for chat history summarization.

    Controls how aggressively the summarizer compresses conversation history:
    - LIGHT: Keep most information, only remove redundancy
    - STANDARD: Balanced compression, main points and context
    - HEAVY: Aggressive compression, only critical information
    """
    LIGHT = "light"
    STANDARD = "standard"
    HEAVY = "heavy"


# Compression mode-specific instructions for summarization prompts
COMPRESSION_INSTRUCTIONS = {
    CompressionMode.LIGHT: (
        "Preserve most details from this conversation while removing only redundancy. "
        "Keep: all key decisions and outcomes, important context and background, "
        "specific details/names/numbers/technical terms, all tool calls and results, "
        "error messages and resolutions. Remove only: repetitive greetings, duplicate information."
    ),
    CompressionMode.STANDARD: (
        "Summarize with balanced compression, keeping main points and essential context. "
        "Keep: key decisions and rationale, important outcomes, critical context for ongoing work, "
        "unresolved items and pending tasks. Remove: intermediate reasoning steps, "
        "exploratory tangents, detailed tool outputs (keep only key findings)."
    ),
    CompressionMode.HEAVY: (
        "Extract only the most critical information. Keep ONLY: final decisions made, "
        "critical outcomes (success/failure), essential context to continue work, "
        "blocking issues and hard dependencies. Remove: all exploratory discussion, "
        "all intermediate steps, all detailed outputs, all background explanations."
    ),
}


class LLMSummaryOutput(BaseModel):
    """LLM-generated summary output (without word counts)"""
    summary: str = Field(description="The main summary text")
    key_points: List[str] = Field(description="3-5 most important points", max_length=8)
    confidence: float = Field(description="Confidence in summary accuracy (0-1)", ge=0, le=1)
    focus_alignment: float = Field(description="How well the summary addresses the specified focus (0-1)", ge=0, le=1)


class SummaryOutput(BaseModel):
    """Complete summary output with computed word counts"""
    summary: str = Field(description="The main summary text")
    key_points: List[str] = Field(description="3-5 most important points", max_length=8)
    confidence: float = Field(description="Confidence in summary accuracy (0-1)", ge=0, le=1)
    focus_alignment: float = Field(description="How well the summary addresses the specified focus (0-1)", ge=0, le=1)
    word_count_original: int = Field(description="Word count of original text (computed client-side)")
    word_count_summary: int = Field(description="Word count of generated summary (computed client-side)")


class BasicSummarizer:
    """
    Basic Summarizer using zero-shot structured prompting

    Demonstrates AbstractCore best practices:
    - Structured output with Pydantic validation
    - Clean parameter design
    - Automatic chunking for long documents
    - Provider-agnostic implementation
    - Built-in retry and error handling (inherited from AbstractCore)

    Optimized defaults (no setup required):
        summarizer = BasicSummarizer()  # Uses gemma3:1b-it-qat, 32k context, 8k chunks

    Custom setup for different needs:
        llm = create_llm("openai", model="gpt-4o-mini", max_tokens=32000)
        summarizer = BasicSummarizer(llm, max_chunk_size=15000)

    Performance benchmarks:
    - gemma3:1b-it-qat: 29s, 95% confidence, cost-effective (instruction-tuned & quantized)
    - qwen3-coder:30b: 119s, 98% confidence, premium quality
    - GPT-4o-mini: Variable, high cost per request
    """

    def __init__(
        self, 
        llm: Optional[AbstractCoreInterface] = None, 
        max_chunk_size: int = 8000,
        max_tokens: int = -1,
        max_output_tokens: int = -1,
        timeout: Optional[float] = None,
        retry_strategy: Optional[FeedbackRetry] = None,
    ):
        """
        Initialize the summarizer

        Args:
            llm: AbstractCore instance (any provider). If None, attempts to create ollama gemma3:1b-it-qat
            max_chunk_size: Maximum characters per chunk for long documents (default 8000)
            max_tokens: Maximum total tokens for LLM context (default -1 = AUTO).
                       - Use -1 (AUTO): Automatically uses model's context window capability
                       - Use specific value: Hard limit for deployment constraint (GPU/RAM limits)
                       Example: max_tokens=16000 limits to 16K even if model supports 128K
            max_output_tokens: Maximum tokens for LLM output generation (default -1 = AUTO).
                              - Use -1 (AUTO): Automatically uses model's output capability
                              - Use specific value: Hard limit for output tokens
            timeout: HTTP request timeout in seconds. None for unlimited timeout (default None)
            retry_strategy: Custom retry strategy for structured output. If None, uses default (3 attempts)
        """
        if llm is None:
            try:
                # Default to gemma3:1b-it-qat with configurable token limits
                # Only pass token limits if not using AUTO mode (-1)
                llm_kwargs = {'timeout': timeout} if timeout is not None else {}
                if max_tokens != -1:
                    llm_kwargs['max_tokens'] = max_tokens
                if max_output_tokens != -1:
                    llm_kwargs['max_output_tokens'] = max_output_tokens
                self.llm = create_llm("ollama", model="gemma3:1b-it-qat", **llm_kwargs)
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default Ollama model 'gemma3:1b-it-qat': {e}\n\n"
                    "💡 To use the default model, please:\n"
                    "   1. Install Ollama from: https://ollama.com/\n"
                    "   2. Download the model: ollama pull gemma3:1b-it-qat\n"
                    "   3. Start Ollama service\n\n"
                    "🔧 Alternatively, provide a custom LLM instance:\n"
                    "   from abstractcore import create_llm\n"
                    "   from abstractcore.processing import BasicSummarizer\n"
                    "   \n"
                    "   # Using OpenAI\n"
                    "   llm = create_llm('openai', model='gpt-4o-mini')\n"
                    "   summarizer = BasicSummarizer(llm)\n"
                    "   \n"
                    "   # Using Anthropic\n"
                    "   llm = create_llm('anthropic', model='claude-3-5-haiku-latest')\n"
                    "   summarizer = BasicSummarizer(llm)\n"
                    "   \n"
                    "   # Using different Ollama model\n"
                    "   llm = create_llm('ollama', model='llama3.2:3b')\n"
                    "   summarizer = BasicSummarizer(llm)"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm
        self.max_chunk_size = max_chunk_size
        # Store token budgets. -1 means AUTO (use model's capability).
        # In AbstractCore, `max_tokens` is the total (input + output) context budget.
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens

        # Default retry strategy with 3 attempts (callers may override for latency-sensitive UX).
        self.retry_strategy = retry_strategy or FeedbackRetry(max_attempts=3)

    def summarize(
        self,
        text: str,
        focus: Optional[str] = None,
        style: SummaryStyle = SummaryStyle.STRUCTURED,
        length: SummaryLength = SummaryLength.STANDARD,
    ) -> SummaryOutput:
        """
        Generate a structured summary of the text

        Args:
            text: Text to summarize
            focus: Optional specific aspect to focus on (e.g., "financial implications", "technical details")
            style: Summary presentation style
            length: Target summary length

        Returns:
            SummaryOutput: Structured summary with metadata

        Example:
            >>> from abstractcore import create_llm
            >>> from abstractcore.processing import BasicSummarizer, SummaryStyle, SummaryLength
            >>>
            >>> llm = create_llm("openai", model="gpt-4o-mini")
            >>> summarizer = BasicSummarizer(llm)
            >>>
            >>> result = summarizer.summarize(
            ...     long_text,
            ...     focus="business implications",
            ...     style=SummaryStyle.EXECUTIVE,
            ...     length=SummaryLength.DETAILED
            ... )
            >>> print(result.summary)
            >>> print(f"Confidence: {result.confidence:.2f}")
        """
        # Handle long documents through chunking
        # Use token-aware chunking for better accuracy
        if self._should_chunk_by_tokens(text):
            logger.info("Using chunked summarization for long document", 
                       text_length=len(text), 
                       style=style.value, 
                       length=length.value)
            return self._summarize_long_document(text, focus, style, length)
        else:
            logger.info("Using single-chunk summarization", 
                       text_length=len(text), 
                       style=style.value, 
                       length=length.value)
            return self._summarize_single_chunk(text, focus, style, length)

    def _summarize_single_chunk(
        self,
        text: str,
        focus: Optional[str],
        style: SummaryStyle,
        length: SummaryLength
    ) -> SummaryOutput:
        """Summarize a single chunk of text"""

        # Build the prompt based on parameters
        prompt = self._build_prompt(text, focus, style, length)

        llm_result: Optional[LLMSummaryOutput] = None
        try:
            # Use AbstractCore's structured output with retry strategy (no word counts in LLM response)
            response = self.llm.generate(prompt, response_model=LLMSummaryOutput, retry_strategy=self.retry_strategy)
            llm_result = self._extract_summary_structured_output(response, context="summary")
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                "Structured summary output failed; falling back to marker format",
                error_type=type(e).__name__,
                error=str(e),
            )
            llm_result = self._summarize_fallback(text=text, focus=focus, style=style, length=length)

        # Compute word counts ourselves (reliable, client-side calculation)
        actual_original_words = len(text.split())
        actual_summary_words = len((llm_result.summary if llm_result else "").split())

        # Create complete result with computed word counts
        return SummaryOutput(
            summary=(llm_result.summary if llm_result else ""),
            key_points=(llm_result.key_points if llm_result else []),
            confidence=(llm_result.confidence if llm_result else 0.5),
            focus_alignment=(llm_result.focus_alignment if llm_result else 0.5),
            word_count_original=actual_original_words,
            word_count_summary=actual_summary_words
        )

    def _summarize_long_document(
        self,
        text: str,
        focus: Optional[str],
        style: SummaryStyle,
        length: SummaryLength
    ) -> SummaryOutput:
        """
        Handle long documents using map-reduce approach

        1. Split into chunks with overlap
        2. Summarize each chunk
        3. Combine chunk summaries into final summary
        """

        # Split text into overlapping chunks
        chunks = self._split_text_into_chunks(text)
        
        logger.debug("Split document into chunks", 
                    chunk_count=len(chunks), 
                    avg_chunk_size=sum(len(c) for c in chunks) // len(chunks))

        if len(chunks) == 1:
            return self._summarize_single_chunk(chunks[0], focus, style, length)

        # Step 1: Summarize each chunk (Map phase)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = self._build_chunk_prompt(chunk, focus, i + 1, len(chunks))

            # Use a simplified output model for chunks
            class ChunkSummary(BaseModel):
                summary: str
                key_points: List[str] = Field(max_length=5)

            try:
                response = self.llm.generate(chunk_prompt, response_model=ChunkSummary, retry_strategy=self.retry_strategy)
                if isinstance(response, ChunkSummary):
                    # When structured output succeeds, response is the ChunkSummary object directly
                    chunk_summaries.append(response)
                elif hasattr(response, 'structured_output') and response.structured_output:
                    # Fallback: check for structured_output attribute
                    chunk_summaries.append(response.structured_output)
                else:
                    raise ValueError(f"Unexpected chunk response type: {type(response)}")
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                # If chunk processing fails, create a minimal placeholder (do not fail the whole summary).
                logger.warning(
                    "Chunk processing failed, creating fallback",
                    chunk_number=i + 1,
                    total_chunks=len(chunks),
                    error_type=type(e).__name__,
                    error=str(e),
                )
                chunk_summaries.append(
                    ChunkSummary(
                        summary=f"Section {i+1} content summary unavailable",
                        key_points=["Content processing failed"],
                    )
                )

        # Step 2: Combine chunk summaries (Reduce phase)
        combined_text = "\n\n".join([
            f"Section {i+1}:\nSummary: {cs.summary}\nKey Points: {', '.join(cs.key_points)}"
            for i, cs in enumerate(chunk_summaries)
        ])

        # Generate final summary from combined summaries
        final_prompt = self._build_final_combination_prompt(combined_text, focus, style, length, len(text))

        llm_result: Optional[LLMSummaryOutput] = None
        try:
            response = self.llm.generate(final_prompt, response_model=LLMSummaryOutput, retry_strategy=self.retry_strategy)
            llm_result = self._extract_summary_structured_output(response, context="final_summary")
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                "Structured final summary output failed; falling back to marker format",
                error_type=type(e).__name__,
                error=str(e),
            )
            llm_result = self._summarize_fallback(text=combined_text, focus=focus, style=style, length=length)

        # Compute word counts ourselves (reliable, client-side calculation)
        actual_original_words = len(text.split())
        actual_summary_words = len((llm_result.summary if llm_result else "").split())

        # Create complete result with computed word counts
        return SummaryOutput(
            summary=(llm_result.summary if llm_result else ""),
            key_points=(llm_result.key_points if llm_result else []),
            confidence=(llm_result.confidence if llm_result else 0.5),
            focus_alignment=(llm_result.focus_alignment if llm_result else 0.5),
            word_count_original=actual_original_words,
            word_count_summary=actual_summary_words
        )

    def _should_chunk_by_tokens(self, text: str) -> bool:
        """
        Determine if text should be chunked based on token count.
        
        Token budget logic:
        - max_tokens = -1 (AUTO): Uses model's full context window capability
        - max_tokens = N: Hard limit (deployment constraint for GPU/RAM)
        
        This ensures we don't exceed GPU memory constraints even when the model
        theoretically supports larger contexts.
        
        Uses centralized TokenUtils for accurate token estimation.
        Falls back to character count if model information unavailable.
        """
        from ..utils.token_utils import TokenUtils
        
        # Get model name from LLM if available
        model_name = None
        if self.llm and hasattr(self.llm, 'model'):
            model_name = self.llm.model
            
        # Estimate tokens using centralized utility. If estimation fails for any reason,
        # fall back to character chunking (conservative).
        try:
            estimated_tokens = TokenUtils.estimate_tokens(text, model_name)
        except Exception:
            return len(text) > self.max_chunk_size
        
        # Determine the effective token budget
        # Get provider's capabilities
        provider_max_input = getattr(self.llm, "max_input_tokens", None) if self.llm else None
        if provider_max_input is None:
            provider_total = getattr(self.llm, "max_tokens", None) if self.llm else None
            provider_output = getattr(self.llm, "max_output_tokens", None) if self.llm else None
            if provider_total is not None and provider_output is not None:
                try:
                    provider_max_input = int(provider_total) - int(provider_output)
                except Exception:
                    provider_max_input = None
        
        # Determine effective max_input_tokens based on configuration
        if self.max_tokens == -1:
            # AUTO mode: Use model's capability
            if provider_max_input is not None:
                max_input_tokens = provider_max_input
            else:
                # Fallback to safe default if model info unavailable
                max_input_tokens = 24000  # Conservative default
        else:
            # User-specified limit (deployment constraint)
            user_max_output = self.max_output_tokens if self.max_output_tokens != -1 else 8000
            user_max_input = self.max_tokens - user_max_output
            
            if provider_max_input is not None:
                # Respect BOTH user limit AND model capability (take minimum)
                max_input_tokens = min(provider_max_input, user_max_input)
            else:
                # No model info, use user limit
                max_input_tokens = user_max_input

        # Reserve prompt/formatting overhead (structured output schemas + instructions).
        # Keep the historical safety floor (8000) for small-context models.
        try:
            token_limit = max(8000, int(max_input_tokens) - 1200)
        except Exception:
            token_limit = 8000

        logger.debug(
            "Chunking decision",
            estimated_tokens=estimated_tokens,
            token_limit=token_limit,
            max_tokens_config=self.max_tokens,
            is_auto_mode=(self.max_tokens == -1),
            will_chunk=(estimated_tokens > token_limit)
        )

        return estimated_tokens > token_limit

    def _extract_summary_structured_output(self, response: object, *, context: str) -> LLMSummaryOutput:
        """Extract structured summary output from AbstractCore responses."""
        if isinstance(response, LLMSummaryOutput):
            return response
        if hasattr(response, "structured_output") and getattr(response, "structured_output"):
            return response.structured_output

        error_msg = f"Failed to generate structured {context} output. Response type: {type(response)}"
        if hasattr(response, "content") and getattr(response, "content"):
            try:
                error_msg += f", Content: {str(response.content)[:200]}..."
            except Exception:
                pass
        if hasattr(response, "structured_output"):
            try:
                error_msg += f", Structured output: {getattr(response, 'structured_output')}"
            except Exception:
                pass
        raise ValueError(error_msg)

    def _summarize_fallback(
        self,
        *,
        text: str,
        focus: Optional[str],
        style: SummaryStyle,
        length: SummaryLength,
    ) -> LLMSummaryOutput:
        """Best-effort summary when structured output cannot be produced reliably."""
        prompt = self._build_fallback_prompt(text=text, focus=focus, style=style, length=length)
        response = self.llm.generate(prompt)
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        summary, key_points, confidence, focus_alignment = self._parse_fallback_response(str(content))
        return LLMSummaryOutput(
            summary=summary,
            key_points=key_points[:8],
            confidence=confidence,
            focus_alignment=focus_alignment,
        )

    def _build_fallback_prompt(
        self,
        *,
        text: str,
        focus: Optional[str],
        style: SummaryStyle,
        length: SummaryLength,
    ) -> str:
        """Build a non-JSON prompt that is easy to parse deterministically."""
        style_instructions = {
            SummaryStyle.STRUCTURED: "Present the summary in a clear, organized format with distinct sections or bullet points.",
            SummaryStyle.NARRATIVE: "Write the summary as a flowing narrative that tells the story of the content.",
            SummaryStyle.OBJECTIVE: "Maintain a neutral, factual tone without opinions or interpretations.",
            SummaryStyle.ANALYTICAL: "Provide critical analysis with insights, implications, and deeper understanding.",
            SummaryStyle.EXECUTIVE: "Focus on actionable insights, business implications, and key decisions.",
            SummaryStyle.CONVERSATIONAL: "Preserve conversational context, key decisions, ongoing topics, and user intent. Focus on information needed for conversation continuity.",
        }

        length_instructions = {
            SummaryLength.BRIEF: "Keep the summary very concise - 2-3 sentences covering only the most essential points.",
            SummaryLength.STANDARD: "Provide a balanced summary of 1-2 paragraphs covering the main ideas.",
            SummaryLength.DETAILED: "Create a comprehensive summary with multiple paragraphs covering all important aspects.",
            SummaryLength.COMPREHENSIVE: "Provide an extensive analysis covering all significant points, context, and implications.",
        }

        focus_instruction = ""
        if focus:
            focus_instruction = f"\nPay special attention to: {focus}\n"

        return f"""Analyze the following text and produce a summary.

{style_instructions[style]}
{length_instructions[length]}{focus_instruction}

Text to summarize:
{text}

Return your answer in this EXACT plain-text format (no JSON, no code blocks):

SUMMARY:
<the main summary text>

KEY POINTS:
- <point 1>
- <point 2>
- <point 3>

CONFIDENCE: <0-1>
FOCUS_ALIGNMENT: <0-1>
"""

    @staticmethod
    def _parse_fallback_response(content: str) -> Tuple[str, List[str], float, float]:
        """Parse marker-format fallback summaries into structured fields."""
        text = (content or "").strip()
        if not text:
            return "", [], 0.5, 0.5

        def _parse_score(label_re: str, default: float) -> float:
            m = re.search(rf"(?im)^{label_re}\s*:\s*(.+?)\s*$", text)
            if not m:
                return default
            raw = m.group(1).strip()
            try:
                if raw.endswith("%"):
                    val = float(raw[:-1].strip()) / 100.0
                else:
                    val = float(raw)
            except Exception:
                return default
            return max(0.0, min(1.0, val))

        summary = ""
        m_summary = re.search(r"(?is)summary\s*:\s*(.*?)\n\s*key\s*points\s*:", text)
        if m_summary:
            summary = m_summary.group(1).strip()
        else:
            # Best-effort: take the first paragraph.
            summary = text.split("\n\n", 1)[0].strip()

        key_points: List[str] = []
        m_kp = re.search(
            r"(?is)key\s*points\s*:\s*(.*?)(?:\n\s*confidence\s*:|\n\s*focus[_ ]alignment\s*:|\Z)",
            text,
        )
        if m_kp:
            block = m_kp.group(1)
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(("-", "•", "*")):
                    line = line.lstrip("-•*").strip()
                if line:
                    key_points.append(line)
        if not key_points:
            # Fallback: try to extract bullet-like lines anywhere.
            for line in text.splitlines():
                line = line.strip()
                if line.startswith(("-", "•", "*")):
                    cleaned = line.lstrip("-•*").strip()
                    if cleaned:
                        key_points.append(cleaned)
        key_points = key_points[:8]

        confidence = _parse_score("confidence", 0.6)
        focus_alignment = _parse_score(r"focus[_ ]alignment", 0.6)

        return summary, key_points, confidence, focus_alignment

    def _split_text_into_chunks(self, text: str, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.max_chunk_size

            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = text.rfind('. ', start + self.max_chunk_size - 500, end)
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 2  # Include the period and space

            chunks.append(text[start:end].strip())

            # Move start position with overlap
            if end >= len(text):
                break
            start = max(start + self.max_chunk_size - overlap, end - overlap)

        return chunks

    def _build_prompt(
        self,
        text: str,
        focus: Optional[str],
        style: SummaryStyle,
        length: SummaryLength
    ) -> str:
        """Build the main summarization prompt"""

        # Style instructions
        style_instructions = {
            SummaryStyle.STRUCTURED: "Present the summary in a clear, organized format with distinct sections or bullet points.",
            SummaryStyle.NARRATIVE: "Write the summary as a flowing narrative that tells the story of the content.",
            SummaryStyle.OBJECTIVE: "Maintain a neutral, factual tone without opinions or interpretations.",
            SummaryStyle.ANALYTICAL: "Provide critical analysis with insights, implications, and deeper understanding.",
            SummaryStyle.EXECUTIVE: "Focus on actionable insights, business implications, and key decisions.",
            SummaryStyle.CONVERSATIONAL: "Preserve conversational context, key decisions, ongoing topics, and user intent. Focus on information needed for conversation continuity."
        }

        # Length instructions
        length_instructions = {
            SummaryLength.BRIEF: "Keep the summary very concise - 2-3 sentences covering only the most essential points.",
            SummaryLength.STANDARD: "Provide a balanced summary of 1-2 paragraphs covering the main ideas.",
            SummaryLength.DETAILED: "Create a comprehensive summary with multiple paragraphs covering all important aspects.",
            SummaryLength.COMPREHENSIVE: "Provide an extensive analysis covering all significant points, context, and implications."
        }

        # Build focus instruction
        focus_instruction = ""
        if focus:
            focus_instruction = f"\nPay special attention to: {focus}\nEnsure the summary addresses this focus area and rate how well it does so."

        prompt = f"""Analyze the following text and create a structured summary.

{style_instructions[style]}
{length_instructions[length]}{focus_instruction}

Text to summarize:
{text}

Requirements:
- Extract 3-5 key points that capture the most important information
- Provide a confidence score (0-1) for the accuracy of your summary
- Estimate the focus alignment (0-1) - how well the summary addresses any specified focus
- Be precise, factual, and avoid hallucination

Generate a comprehensive structured summary following the specified style and length guidelines."""

        return prompt

    def _build_chunk_prompt(self, chunk: str, focus: Optional[str], chunk_num: int, total_chunks: int) -> str:
        """Build prompt for individual chunk processing"""

        focus_instruction = f" You have been asked to focus especially on {focus}." if focus else ""


        return f"""Summarize this section of a larger document (Part {chunk_num} of {total_chunks}).

{focus_instruction}

Text section:
{chunk}

Create a concise summary capturing:
- Main points from this section
- Key information relevant to the overall document
- Important details that shouldn't be lost

Keep the summary focused but comprehensive enough to be combined with other sections later."""

    def _build_final_combination_prompt(
        self,
        combined_summaries: str,
        focus: Optional[str],
        style: SummaryStyle,
        length: SummaryLength,
        original_length: int
    ) -> str:
        """Build prompt for combining chunk summaries into final summary"""

        style_instructions = {
            SummaryStyle.STRUCTURED: "Present the final summary in a clear, organized format.",
            SummaryStyle.NARRATIVE: "Weave the information into a flowing narrative.",
            SummaryStyle.OBJECTIVE: "Maintain a neutral, factual tone.",
            SummaryStyle.ANALYTICAL: "Provide analytical insights and implications.",
            SummaryStyle.EXECUTIVE: "Focus on actionable insights and business implications.",
            SummaryStyle.CONVERSATIONAL: "Preserve conversational context and key information for conversation continuity."
        }

        length_instructions = {
            SummaryLength.BRIEF: "Create a very concise final summary (2-3 sentences).",
            SummaryLength.STANDARD: "Create a balanced final summary (1-2 paragraphs).",
            SummaryLength.DETAILED: "Create a comprehensive final summary (multiple paragraphs).",
            SummaryLength.COMPREHENSIVE: "Create an extensive final analysis covering all aspects."
        }

        focus_instruction = ""
        if focus:
            focus_instruction = f" You have been asked to focus especially on {focus}. Ensure the final summary strongly addresses this focus area." if focus_instruction else ""

        return f"""Combine these section summaries into a cohesive final summary of the complete document.

{style_instructions[style]}
{length_instructions[length]}{focus_instruction}

Section summaries:
{combined_summaries}

Requirements:
- Synthesize information from all sections into a coherent whole
- Eliminate redundancy while preserving important details
- Extract the most significant key points across all sections
- The original document had approximately {original_length} characters
- Provide confidence and focus alignment scores

Create a unified summary that represents the entire document effectively."""

    def summarize_chat_history(
        self,
        messages: List[dict],
        preserve_recent: int = 6,
        focus: Optional[str] = None,
        compression_mode: CompressionMode = CompressionMode.STANDARD
    ) -> SummaryOutput:
        """
        Specialized method for chat history summarization following SOTA 2025 practices

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            preserve_recent: Number of recent messages to keep intact (default 6)
            focus: Optional focus for summarization (e.g., "key decisions", "technical solutions")
            compression_mode: How aggressively to compress (LIGHT, STANDARD, HEAVY)

        Returns:
            SummaryOutput: Structured summary optimized for chat history context

        SOTA Best Practices Implemented:
        - Preserves conversational context and flow
        - Focuses on decisions, solutions, and ongoing topics
        - Maintains user intent and assistant responses
        - Optimized for chat continuation rather than standalone summary

        Compression Modes:
        - LIGHT: Keep most information, only remove redundancy
        - STANDARD: Balanced compression, main points and context
        - HEAVY: Aggressive compression, only critical information
        """
        # Build focus with compression instructions
        compression_instruction = COMPRESSION_INSTRUCTIONS.get(
            compression_mode,
            COMPRESSION_INSTRUCTIONS[CompressionMode.STANDARD]
        )

        # Combine user focus with compression instruction
        if focus:
            effective_focus = f"{compression_instruction} Focus especially on: {focus}"
        else:
            effective_focus = compression_instruction

        # Map compression mode to summary length for appropriate output size
        length_map = {
            CompressionMode.LIGHT: SummaryLength.DETAILED,
            CompressionMode.STANDARD: SummaryLength.STANDARD,
            CompressionMode.HEAVY: SummaryLength.BRIEF,
        }
        target_length = length_map.get(compression_mode, SummaryLength.STANDARD)

        logger.debug("Chat history summarization with compression mode",
                    message_count=len(messages),
                    preserve_recent=preserve_recent,
                    compression_mode=compression_mode.value,
                    target_length=target_length.value)

        if len(messages) <= preserve_recent:
            # If short enough, just summarize normally
            logger.debug("Chat history is short, using standard summarization",
                        message_count=len(messages),
                        preserve_recent=preserve_recent)
            chat_text = self._format_chat_messages_to_text(messages)
            return self.summarize(
                chat_text,
                focus=effective_focus,
                style=SummaryStyle.CONVERSATIONAL,
                length=target_length
            )

        # Split into older messages (to summarize) and recent messages (to preserve)
        older_messages = messages[:-preserve_recent]
        recent_messages = messages[-preserve_recent:]

        logger.debug("Splitting chat history for summarization",
                    total_messages=len(messages),
                    older_messages=len(older_messages),
                    recent_messages=len(recent_messages))

        # Summarize older messages with conversational focus and compression mode
        older_text = self._format_chat_messages_to_text(older_messages)
        older_summary = self.summarize(
            older_text,
            focus=effective_focus,
            style=SummaryStyle.CONVERSATIONAL,
            length=target_length
        )

        # The summary should ONLY contain the older messages summary
        # Recent messages will be preserved as separate messages in the session
        summary_only = older_summary.summary

        # Calculate metrics using accurate word counts
        original_text = self._format_chat_messages_to_text(messages)
        actual_original_words = len(original_text.split())
        actual_summary_words = len(summary_only.split())

        return SummaryOutput(
            summary=summary_only,
            key_points=older_summary.key_points,
            confidence=older_summary.confidence,
            focus_alignment=older_summary.focus_alignment,
            word_count_original=actual_original_words,
            word_count_summary=actual_summary_words
        )

    def _format_chat_messages_to_text(self, messages: List[dict]) -> str:
        """Format chat messages to readable text for summarization"""
        formatted_lines = []

        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '').strip()

            if not content:
                continue

            if role == 'system':
                formatted_lines.append(f"[SYSTEM]: {content}")
            elif role == 'user':
                formatted_lines.append(f"[USER]: {content}")
            elif role == 'assistant':
                formatted_lines.append(f"[ASSISTANT]: {content}")
            else:
                formatted_lines.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(formatted_lines)
