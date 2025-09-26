"""
Basic Summarizer - Clean, powerful summarization built on AbstractCore

Demonstrates how to use AbstractCore's infrastructure to create sophisticated
text processing capabilities with minimal complexity.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from ..core.interface import AbstractLLMInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry


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
        summarizer = BasicSummarizer()  # Uses gemma3:1b-it-qat, 16k context, 8k chunks

    Custom setup for different needs:
        llm = create_llm("openai", model="gpt-4o-mini", max_tokens=32000)
        summarizer = BasicSummarizer(llm, max_chunk_size=15000)

    Performance benchmarks:
    - gemma3:1b-it-qat: 29s, 95% confidence, cost-effective (instruction-tuned & quantized)
    - qwen3-coder:30b: 119s, 98% confidence, premium quality
    - GPT-4o-mini: Variable, high cost per request
    """

    def __init__(self, llm: Optional[AbstractLLMInterface] = None, max_chunk_size: int = 8000):
        """
        Initialize the summarizer

        Args:
            llm: AbstractLLM instance (any provider). If None, attempts to create ollama gemma3:1b-it-qat with 16k context
            max_chunk_size: Maximum characters per chunk for long documents (default 8k)
        """
        if llm is None:
            try:
                # Default to gemma3:1b-it-qat with 16k context window
                self.llm = create_llm("ollama", model="gemma3:1b-it-qat", max_tokens=16000)
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default Ollama model 'gemma3:1b-it-qat': {e}\n\n"
                    "💡 To use the default model, please:\n"
                    "   1. Install Ollama from: https://ollama.com/\n"
                    "   2. Download the model: ollama pull gemma3:1b-it-qat\n"
                    "   3. Start Ollama service\n\n"
                    "🔧 Alternatively, provide a custom LLM instance:\n"
                    "   from abstractllm import create_llm\n"
                    "   from abstractllm.processing import BasicSummarizer\n"
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

        # Default retry strategy with 3 attempts
        self.retry_strategy = FeedbackRetry(max_attempts=3)

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
            >>> from abstractllm import create_llm
            >>> from abstractllm.processing import BasicSummarizer, SummaryStyle, SummaryLength
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
        if len(text) > self.max_chunk_size:
            return self._summarize_long_document(text, focus, style, length)
        else:
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

        # Use AbstractCore's structured output with retry strategy (no word counts in LLM response)
        response = self.llm.generate(prompt, response_model=LLMSummaryOutput, retry_strategy=self.retry_strategy)

        # Extract the structured output
        llm_result = None
        if isinstance(response, LLMSummaryOutput):
            # When structured output succeeds, response is the LLMSummaryOutput object directly
            llm_result = response
        elif hasattr(response, 'structured_output') and response.structured_output:
            # Fallback: check for structured_output attribute
            llm_result = response.structured_output
        else:
            # Debug information for troubleshooting
            error_msg = f"Failed to generate structured summary output. Response type: {type(response)}"
            if hasattr(response, 'content'):
                error_msg += f", Content: {response.content[:200]}..."
            if hasattr(response, 'structured_output'):
                error_msg += f", Structured output: {response.structured_output}"
            raise ValueError(error_msg)

        # Compute word counts ourselves (reliable, client-side calculation)
        actual_original_words = len(text.split())
        actual_summary_words = len(llm_result.summary.split())

        # Create complete result with computed word counts
        return SummaryOutput(
            summary=llm_result.summary,
            key_points=llm_result.key_points,
            confidence=llm_result.confidence,
            focus_alignment=llm_result.focus_alignment,
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

            response = self.llm.generate(chunk_prompt, response_model=ChunkSummary, retry_strategy=self.retry_strategy)
            if isinstance(response, ChunkSummary):
                # When structured output succeeds, response is the ChunkSummary object directly
                chunk_summaries.append(response)
            elif hasattr(response, 'structured_output') and response.structured_output:
                # Fallback: check for structured_output attribute
                chunk_summaries.append(response.structured_output)
            else:
                # If chunk processing fails, create a fallback summary
                print(f"Warning: Chunk {i+1} processing failed, creating fallback")
                chunk_summaries.append(ChunkSummary(
                    summary=f"Section {i+1} content summary unavailable",
                    key_points=["Content processing failed"]
                ))

        # Step 2: Combine chunk summaries (Reduce phase)
        combined_text = "\n\n".join([
            f"Section {i+1}:\nSummary: {cs.summary}\nKey Points: {', '.join(cs.key_points)}"
            for i, cs in enumerate(chunk_summaries)
        ])

        # Generate final summary from combined summaries
        final_prompt = self._build_final_combination_prompt(combined_text, focus, style, length, len(text))

        response = self.llm.generate(final_prompt, response_model=LLMSummaryOutput, retry_strategy=self.retry_strategy)

        # Extract the structured output
        llm_result = None
        if isinstance(response, LLMSummaryOutput):
            # When structured output succeeds, response is the LLMSummaryOutput object directly
            llm_result = response
        elif hasattr(response, 'structured_output') and response.structured_output:
            # Fallback: check for structured_output attribute
            llm_result = response.structured_output
        else:
            # Debug information for troubleshooting
            error_msg = f"Failed to generate final structured summary output. Response type: {type(response)}"
            if hasattr(response, 'content'):
                error_msg += f", Content: {response.content[:200]}..."
            if hasattr(response, 'structured_output'):
                error_msg += f", Structured output: {response.structured_output}"
            raise ValueError(error_msg)

        # Compute word counts ourselves (reliable, client-side calculation)
        actual_original_words = len(text.split())
        actual_summary_words = len(llm_result.summary.split())

        # Create complete result with computed word counts
        return SummaryOutput(
            summary=llm_result.summary,
            key_points=llm_result.key_points,
            confidence=llm_result.confidence,
            focus_alignment=llm_result.focus_alignment,
            word_count_original=actual_original_words,
            word_count_summary=actual_summary_words
        )

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

        focus_instruction = f"\nFocus area: {focus}" if focus else ""

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
            focus_instruction = f"\nSpecial focus: {focus}\nEnsure the final summary strongly addresses this focus area."

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
        focus: Optional[str] = None
    ) -> SummaryOutput:
        """
        Specialized method for chat history summarization following SOTA 2025 practices

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            preserve_recent: Number of recent messages to keep intact (default 6)
            focus: Optional focus for summarization (e.g., "key decisions", "technical solutions")

        Returns:
            SummaryOutput: Structured summary optimized for chat history context

        SOTA Best Practices Implemented:
        - Preserves conversational context and flow
        - Focuses on decisions, solutions, and ongoing topics
        - Maintains user intent and assistant responses
        - Optimized for chat continuation rather than standalone summary
        """
        if len(messages) <= preserve_recent:
            # If short enough, just summarize normally
            chat_text = self._format_chat_messages_to_text(messages)
            return self.summarize(
                chat_text,
                focus=focus or "conversational context and key information",
                style=SummaryStyle.CONVERSATIONAL,
                length=SummaryLength.STANDARD
            )

        # Split into older messages (to summarize) and recent messages (to preserve)
        older_messages = messages[:-preserve_recent]
        recent_messages = messages[-preserve_recent:]

        # Summarize older messages with conversational focus
        older_text = self._format_chat_messages_to_text(older_messages)
        older_summary = self.summarize(
            older_text,
            focus=focus or "key decisions, solutions, and ongoing context",
            style=SummaryStyle.CONVERSATIONAL,
            length=SummaryLength.DETAILED
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