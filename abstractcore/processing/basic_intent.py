"""
Basic Intent Analyzer - Identify and analyze intents behind text

Demonstrates how to use AbstractCore's infrastructure to create sophisticated
intent analysis capabilities with minimal complexity.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ..core.interface import AbstractCoreInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


class IntentType(Enum):
    """Primary intent categories based on psychological research"""
    INFORMATION_SEEKING = "information_seeking"      # Asking questions, requesting data
    INFORMATION_SHARING = "information_sharing"      # Providing facts, explanations
    PROBLEM_SOLVING = "problem_solving"              # Seeking or offering solutions
    DECISION_MAKING = "decision_making"              # Evaluating options, making choices
    PERSUASION = "persuasion"                        # Convincing, influencing opinions
    CLARIFICATION = "clarification"                  # Seeking or providing clarity
    EMOTIONAL_EXPRESSION = "emotional_expression"    # Expressing feelings, reactions
    RELATIONSHIP_BUILDING = "relationship_building"  # Social connection, rapport
    INSTRUCTION_GIVING = "instruction_giving"        # Teaching, directing actions
    VALIDATION_SEEKING = "validation_seeking"        # Seeking approval, confirmation
    # New intent types from psychological research
    FACE_SAVING = "face_saving"                      # Protecting self-image, avoiding embarrassment
    BLAME_DEFLECTION = "blame_deflection"            # Redirecting responsibility to external factors
    POWER_ASSERTION = "power_assertion"              # Establishing dominance or authority
    EMPATHY_SEEKING = "empathy_seeking"              # Seeking understanding and emotional support
    CONFLICT_AVOIDANCE = "conflict_avoidance"        # Preventing or minimizing confrontation
    TRUST_BUILDING = "trust_building"                # Establishing or maintaining credibility
    DECEPTION = "deception"                          # Intentional misdirection or false information


class IntentDepth(Enum):
    """Depth of intent analysis"""
    SURFACE = "surface"           # Obvious, stated intentions
    UNDERLYING = "underlying"     # Hidden motivations and goals
    COMPREHENSIVE = "comprehensive"  # Full analysis including subconscious drivers


class IntentContext(Enum):
    """Context type for intent analysis"""
    STANDALONE = "standalone"     # Single message/text analysis
    CONVERSATIONAL = "conversational"  # Part of ongoing dialogue
    DOCUMENT = "document"         # Formal document or article
    INTERACTIVE = "interactive"   # Real-time interaction context


class DeceptionIndicators(BaseModel):
    """Deception analysis indicators based on psychological research"""
    deception_likelihood: float = Field(description="Likelihood of deceptive intent (0-1)", ge=0, le=1)
    narrative_consistency: float = Field(description="Internal consistency of the narrative (0-1)", ge=0, le=1)
    linguistic_markers: List[str] = Field(description="Specific linguistic indicators of potential deception", max_length=5)
    temporal_coherence: float = Field(description="Logical flow and timing consistency (0-1)", ge=0, le=1)
    emotional_congruence: float = Field(description="Alignment between stated emotions and content (0-1)", ge=0, le=1)
    deception_evidence: List[str] = Field(description="Evidence indicating potential deception (contradictions, deflection, inconsistencies)", max_length=3)
    authenticity_evidence: List[str] = Field(description="Evidence indicating authenticity (consistency, accountability, directness)", max_length=3)


class IdentifiedIntent(BaseModel):
    """Single identified intent with details"""
    intent_type: IntentType = Field(description="Primary intent category")
    confidence: float = Field(description="Confidence in this intent identification (0-1)", ge=0, le=1)
    description: str = Field(description="Human-readable description of the intent")
    underlying_goal: str = Field(description="What the person ultimately wants to achieve")
    emotional_undertone: str = Field(description="Emotional context or undertone")
    urgency_level: float = Field(description="How urgent or pressing this intent is (0-1)", ge=0, le=1)
    deception_analysis: DeceptionIndicators = Field(description="Deception evaluation based on psychological markers - always included in intent analysis")


class LLMIntentOutput(BaseModel):
    """LLM-generated intent analysis output"""
    primary_intent: IdentifiedIntent = Field(description="The most prominent intent")
    secondary_intents: List[IdentifiedIntent] = Field(description="Additional intents present", max_length=3)
    intent_complexity: float = Field(description="How complex/layered the intentions are (0-1)", ge=0, le=1)
    contextual_factors: List[str] = Field(description="Important contextual elements affecting intent", max_length=5)
    suggested_response_approach: str = Field(description="How one should respond to these intents")
    overall_confidence: float = Field(description="Overall confidence in the analysis (0-1)", ge=0, le=1)


class IntentAnalysisOutput(BaseModel):
    """Complete intent analysis output with computed metadata"""
    primary_intent: IdentifiedIntent = Field(description="The most prominent intent")
    secondary_intents: List[IdentifiedIntent] = Field(description="Additional intents present", max_length=3)
    intent_complexity: float = Field(description="How complex/layered the intentions are (0-1)", ge=0, le=1)
    contextual_factors: List[str] = Field(description="Important contextual elements affecting intent", max_length=5)
    suggested_response_approach: str = Field(description="How one should respond to these intents")
    overall_confidence: float = Field(description="Overall confidence in the analysis (0-1)", ge=0, le=1)
    word_count_analyzed: int = Field(description="Word count of analyzed text (computed client-side)")
    analysis_depth: IntentDepth = Field(description="Depth of analysis performed")
    context_type: IntentContext = Field(description="Context type used for analysis")


class BasicIntentAnalyzer:
    """
    Basic Intent Analyzer using zero-shot structured prompting

    Demonstrates AbstractCore best practices:
    - Structured output with Pydantic validation
    - Clean parameter design
    - Automatic chunking for long documents
    - Provider-agnostic implementation
    - Built-in retry and error handling (inherited from AbstractCore)

    Optimized defaults (no setup required):
        analyzer = BasicIntentAnalyzer()  # Uses gemma3:1b-it-qat, 32k context, 8k chunks

    Custom setup for different needs:
        llm = create_llm("openai", model="gpt-4o-mini", max_tokens=32000)
        analyzer = BasicIntentAnalyzer(llm, max_chunk_size=15000)

    Performance considerations:
    - gemma3:1b-it-qat: Fast, cost-effective for basic intent analysis
    - qwen3-coder:30b: Premium quality for complex psychological analysis
    - GPT-4o-mini: Excellent for nuanced intent understanding
    """

    def __init__(
        self, 
        llm: Optional[AbstractCoreInterface] = None, 
        max_chunk_size: int = 8000,
        max_tokens: int = 32000,
        max_output_tokens: int = 8000,
        timeout: Optional[float] = None,
        debug: bool = False
    ):
        """
        Initialize the intent analyzer

        Args:
            llm: AbstractCore instance (any provider). If None, attempts to create ollama gemma3:1b-it-qat
            max_chunk_size: Maximum characters per chunk for long documents (default 8000)
            max_tokens: Maximum total tokens for LLM context (default 32000)
            max_output_tokens: Maximum tokens for LLM output generation (default 8000)
            timeout: HTTP request timeout in seconds. None for unlimited timeout (default None)
            debug: Enable debug output including raw LLM responses (default False)
        """
        if llm is None:
            try:
                # Default to gemma3:1b-it-qat with configurable token limits
                self.llm = create_llm("ollama", model="gemma3:1b-it-qat", max_tokens=max_tokens, max_output_tokens=max_output_tokens, timeout=timeout)
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default Ollama model 'gemma3:1b-it-qat': {e}\n\n"
                    "💡 To use the default model, please:\n"
                    "   1. Install Ollama from: https://ollama.com/\n"
                    "   2. Download the model: ollama pull gemma3:1b-it-qat\n"
                    "   3. Start Ollama service\n\n"
                    "🔧 Alternatively, provide a custom LLM instance:\n"
                    "   from abstractcore import create_llm\n"
                    "   from abstractcore.processing import BasicIntentAnalyzer\n"
                    "   \n"
                    "   # Using OpenAI\n"
                    "   llm = create_llm('openai', model='gpt-4o-mini')\n"
                    "   analyzer = BasicIntentAnalyzer(llm)\n"
                    "   \n"
                    "   # Using Anthropic\n"
                    "   llm = create_llm('anthropic', model='claude-3-5-haiku-latest')\n"
                    "   analyzer = BasicIntentAnalyzer(llm)\n"
                    "   \n"
                    "   # Using different Ollama model\n"
                    "   llm = create_llm('ollama', model='llama3.2:3b')\n"
                    "   analyzer = BasicIntentAnalyzer(llm)"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm
        self.max_chunk_size = max_chunk_size
        self.debug = debug

        # Default retry strategy with 3 attempts
        self.retry_strategy = FeedbackRetry(max_attempts=3)

    def analyze_intent(
        self,
        text: str,
        context_type: IntentContext = IntentContext.STANDALONE,
        depth: IntentDepth = IntentDepth.UNDERLYING,
        focus: Optional[str] = None,
    ) -> IntentAnalysisOutput:
        """
        Analyze the intents behind the given text

        Args:
            text: Text to analyze for intents
            context_type: Type of context for the analysis
            depth: Depth of intent analysis to perform
            focus: Optional specific aspect to focus on (e.g., "business motivations", "emotional drivers")

        Returns:
            IntentAnalysisOutput: Structured intent analysis with metadata including deception assessment

        Example:
            >>> from abstractcore import create_llm
            >>> from abstractcore.processing import BasicIntentAnalyzer, IntentContext, IntentDepth
            >>>
            >>> llm = create_llm("openai", model="gpt-4o-mini")
            >>> analyzer = BasicIntentAnalyzer(llm)
            >>>
            >>> result = analyzer.analyze_intent(
            ...     "I was wondering if you could help me understand how to improve our team's productivity?",
            ...     context_type=IntentContext.CONVERSATIONAL,
            ...     depth=IntentDepth.UNDERLYING,
            ...     focus="management concerns"
            ... )
            >>> print(result.primary_intent.intent_type)
            >>> print(f"Confidence: {result.overall_confidence:.2f}")
        """
        # Handle long documents through chunking
        if self._should_chunk_by_tokens(text):
            logger.info("Using chunked intent analysis for long document", 
                       text_length=len(text), 
                       context_type=context_type.value, 
                       depth=depth.value)
            return self._analyze_long_document(text, context_type, depth, focus)
        else:
            logger.info("Using single-chunk intent analysis", 
                       text_length=len(text), 
                       context_type=context_type.value, 
                       depth=depth.value)
            return self._analyze_single_chunk(text, context_type, depth, focus)

    def _analyze_single_chunk(
        self,
        text: str,
        context_type: IntentContext,
        depth: IntentDepth,
        focus: Optional[str]
    ) -> IntentAnalysisOutput:
        """Analyze intent for a single chunk of text"""

        # Build the prompt based on parameters (deception analysis always included)
        prompt = self._build_prompt(text, context_type, depth, focus)

        if self.debug:
            print(f"\n🔧 DEBUG: Prompt sent to LLM:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        # Use AbstractCore's structured output with retry strategy
        try:
            response = self.llm.generate(prompt, response_model=LLMIntentOutput, retry_strategy=self.retry_strategy)
        except Exception as e:
            if self.debug:
                print(f"\n❌ DEBUG: LLM generation failed with error: {e}")
                # Try to get the raw response if available
                try:
                    raw_response = self.llm.generate(prompt)
                    print(f"\n🔧 DEBUG: Raw LLM response (without structured output):")
                    print("=" * 80)
                    print(raw_response.content if hasattr(raw_response, 'content') else str(raw_response))
                    print("=" * 80)
                except Exception as raw_e:
                    print(f"❌ DEBUG: Could not get raw response: {raw_e}")
            raise

        # Extract the structured output
        llm_result = None
        if isinstance(response, LLMIntentOutput):
            # When structured output succeeds, response is the LLMIntentOutput object directly
            llm_result = response
        elif hasattr(response, 'structured_output') and response.structured_output:
            # Fallback: check for structured_output attribute
            llm_result = response.structured_output
        else:
            # Debug information for troubleshooting
            error_msg = f"Failed to generate structured intent analysis output. Response type: {type(response)}"
            if hasattr(response, 'content'):
                error_msg += f", Content: {response.content[:200]}..."
            if hasattr(response, 'structured_output'):
                error_msg += f", Structured output: {response.structured_output}"
            raise ValueError(error_msg)

        # Compute word count (reliable, client-side calculation)
        actual_word_count = len(text.split())

        # Create complete result with computed metadata
        return IntentAnalysisOutput(
            primary_intent=llm_result.primary_intent,
            secondary_intents=llm_result.secondary_intents,
            intent_complexity=llm_result.intent_complexity,
            contextual_factors=llm_result.contextual_factors,
            suggested_response_approach=llm_result.suggested_response_approach,
            overall_confidence=llm_result.overall_confidence,
            word_count_analyzed=actual_word_count,
            analysis_depth=depth,
            context_type=context_type
        )

    def _analyze_long_document(
        self,
        text: str,
        context_type: IntentContext,
        depth: IntentDepth,
        focus: Optional[str]
    ) -> IntentAnalysisOutput:
        """
        Handle long documents using map-reduce approach

        1. Split into chunks with overlap
        2. Analyze intent for each chunk
        3. Combine chunk analyses into final intent analysis
        """

        # Split text into overlapping chunks
        chunks = self._split_text_into_chunks(text)
        
        logger.debug("Split document into chunks", 
                    chunk_count=len(chunks), 
                    avg_chunk_size=sum(len(c) for c in chunks) // len(chunks))

        if len(chunks) == 1:
            return self._analyze_single_chunk(chunks[0], context_type, depth, focus)

        # Step 1: Analyze each chunk (Map phase)
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = self._build_chunk_prompt(chunk, context_type, focus, i + 1, len(chunks))

            # Use a simplified output model for chunks
            class ChunkIntentAnalysis(BaseModel):
                primary_intent_type: str
                intent_description: str
                underlying_goal: str
                confidence: float = Field(ge=0, le=1)

            response = self.llm.generate(chunk_prompt, response_model=ChunkIntentAnalysis, retry_strategy=self.retry_strategy)
            if isinstance(response, ChunkIntentAnalysis):
                # When structured output succeeds, response is the ChunkIntentAnalysis object directly
                chunk_analyses.append(response)
            elif hasattr(response, 'structured_output') and response.structured_output:
                # Fallback: check for structured_output attribute
                chunk_analyses.append(response.structured_output)
            else:
                # If chunk processing fails, create a fallback analysis
                logger.warning("Chunk intent analysis failed, creating fallback", 
                             chunk_number=i+1, 
                             total_chunks=len(chunks))
                chunk_analyses.append(ChunkIntentAnalysis(
                    primary_intent_type="information_sharing",
                    intent_description=f"Section {i+1} intent analysis unavailable",
                    underlying_goal="Content processing failed",
                    confidence=0.1
                ))

        # Step 2: Combine chunk analyses (Reduce phase)
        combined_analysis = "\n\n".join([
            f"Section {i+1}:\nIntent Type: {ca.primary_intent_type}\nDescription: {ca.intent_description}\nUnderlying Goal: {ca.underlying_goal}\nConfidence: {ca.confidence:.2f}"
            for i, ca in enumerate(chunk_analyses)
        ])

        # Generate final intent analysis from combined analyses
        final_prompt = self._build_final_combination_prompt(combined_analysis, context_type, depth, focus, len(text))

        response = self.llm.generate(final_prompt, response_model=LLMIntentOutput, retry_strategy=self.retry_strategy)

        # Extract the structured output
        llm_result = None
        if isinstance(response, LLMIntentOutput):
            # When structured output succeeds, response is the LLMIntentOutput object directly
            llm_result = response
        elif hasattr(response, 'structured_output') and response.structured_output:
            # Fallback: check for structured_output attribute
            llm_result = response.structured_output
        else:
            # Debug information for troubleshooting
            error_msg = f"Failed to generate final structured intent analysis output. Response type: {type(response)}"
            if hasattr(response, 'content'):
                error_msg += f", Content: {response.content[:200]}..."
            if hasattr(response, 'structured_output'):
                error_msg += f", Structured output: {response.structured_output}"
            raise ValueError(error_msg)

        # Compute word count (reliable, client-side calculation)
        actual_word_count = len(text.split())

        # Create complete result with computed metadata
        return IntentAnalysisOutput(
            primary_intent=llm_result.primary_intent,
            secondary_intents=llm_result.secondary_intents,
            intent_complexity=llm_result.intent_complexity,
            contextual_factors=llm_result.contextual_factors,
            suggested_response_approach=llm_result.suggested_response_approach,
            overall_confidence=llm_result.overall_confidence,
            word_count_analyzed=actual_word_count,
            analysis_depth=depth,
            context_type=context_type
        )

    def _should_chunk_by_tokens(self, text: str) -> bool:
        """
        Determine if text should be chunked based on token count.
        
        Uses centralized TokenUtils for accurate token estimation.
        Falls back to character count if model information unavailable.
        """
        from ..utils.token_utils import TokenUtils
        
        # Get model name from LLM if available
        model_name = None
        if self.llm and hasattr(self.llm, 'model'):
            model_name = self.llm.model
            
        # Estimate tokens using centralized utility
        estimated_tokens = TokenUtils.estimate_tokens(text, model_name)
        
        # Use a conservative token limit (leaving room for prompt overhead)
        # Most models have 32k+ context nowadays, so 8k tokens for input text is safe
        token_limit = 8000
        
        if estimated_tokens > token_limit:
            return True
            
        # Fallback to character-based check for very long texts
        return len(text) > self.max_chunk_size

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
        context_type: IntentContext,
        depth: IntentDepth,
        focus: Optional[str]
    ) -> str:
        """Build the main intent analysis prompt"""

        # Context instructions
        context_instructions = {
            IntentContext.STANDALONE: "Analyze this text as an independent piece of communication.",
            IntentContext.CONVERSATIONAL: "Analyze this text as part of an ongoing conversation or dialogue.",
            IntentContext.DOCUMENT: "Analyze this text as part of a formal document or structured content.",
            IntentContext.INTERACTIVE: "Analyze this text as part of a real-time interactive communication."
        }

        # Depth instructions
        depth_instructions = {
            IntentDepth.SURFACE: "Focus on the obvious, explicitly stated intentions and goals.",
            IntentDepth.UNDERLYING: "Look beyond surface statements to identify hidden motivations, implicit goals, and underlying psychological drivers.",
            IntentDepth.COMPREHENSIVE: "Provide a complete analysis including conscious intentions, subconscious motivations, emotional drivers, and potential unstated goals."
        }

        # Build focus instruction
        focus_instruction = ""
        if focus:
            focus_instruction = f"\nPay special attention to: {focus}\nEnsure the analysis addresses this focus area thoroughly."

        # Deception analysis is always integrated into intent analysis
        deception_instruction = """

DECEPTION ANALYSIS: Always evaluate authenticity with healthy skepticism.

Key principles:
1. Check for contradictions within the conversation
2. Ask: "What does this person gain by lying here?"
3. Be suspicious of overly complex explanations for simple failures
4. Confident, detailed stories can mask deception - don't trust them blindly
5. Look for what consequences they're trying to avoid

Red flags:
- Timeline inconsistencies or contradictions
- Elaborate technical excuses for basic mistakes
- Shifting blame to multiple external factors
- Claims that can't be easily verified

For deception analysis, assess:
- Likelihood of deception (0-1 scale)
- What motive exists for lying in this context
- Whether the explanation is unnecessarily complex
- Evidence for/against authenticity

Note: If someone admits past deception to a third party, that admission itself is likely honest."""

        prompt = f"""Analyze the following text to identify and understand the intents, motivations, and goals behind the communication.

{context_instructions[context_type]}
{depth_instructions[depth]}{focus_instruction}{deception_instruction}

Text to analyze:
{text}

Your task is to identify:
1. PRIMARY INTENT: The main purpose or goal behind this communication
2. SECONDARY INTENTS: Additional intentions that may be present (up to 3)
3. UNDERLYING GOALS: What the person ultimately wants to achieve
4. EMOTIONAL UNDERTONES: The emotional context affecting the communication
5. CONTEXTUAL FACTORS: Important situational elements that influence the intents
6. RESPONSE APPROACH: How someone should respond to address these intents effectively

For each intent, consider:
- What type of intent it is (information seeking, problem solving, persuasion, face-saving, deception, etc.)
- How confident you are in identifying this intent
- What emotional undertones are present
- How urgent or pressing this intent seems to be
- What the person's underlying goal really is

Requirements:
- Be precise and avoid speculation beyond what the text supports
- Consider both explicit and implicit intentions
- Evaluate the complexity and layering of multiple intents
- Provide confidence scores for your assessments
- Focus on actionable insights for responding appropriately
- When deception analysis is requested, provide evidence-based psychological assessment

Generate a comprehensive structured analysis of the intents behind this communication.

CRITICAL JSON FORMAT REQUIREMENTS:
- Respond with ONLY valid JSON - no other text before or after
- Use double quotes for all strings and keys

- All field names must match exactly: primary_intent, secondary_intents, intent_complexity, etc.
- Arrays must use square brackets [], objects must use curly braces {{}}
- No trailing commas, no comments, pure JSON only"""

        return prompt

    def _build_chunk_prompt(self, chunk: str, context_type: IntentContext, focus: Optional[str], chunk_num: int, total_chunks: int) -> str:
        """Build prompt for individual chunk processing"""

        context_instruction = f"This is part of a {context_type.value} communication."
        focus_instruction = f" Focus especially on {focus}." if focus else ""

        return f"""Analyze the intents in this section of a larger text (Part {chunk_num} of {total_chunks}).

{context_instruction}{focus_instruction}

Text section:
{chunk}

Identify:
- The primary intent type in this section
- A brief description of what the person wants
- The underlying goal they're trying to achieve
- Your confidence in this assessment

Keep the analysis focused on this section while considering it's part of a larger communication."""

    def _build_final_combination_prompt(
        self,
        combined_analyses: str,
        context_type: IntentContext,
        depth: IntentDepth,
        focus: Optional[str],
        original_length: int
    ) -> str:
        """Build prompt for combining chunk analyses into final intent analysis"""

        context_instructions = {
            IntentContext.STANDALONE: "Synthesize the intents from this independent communication.",
            IntentContext.CONVERSATIONAL: "Combine the intents from this conversational exchange.",
            IntentContext.DOCUMENT: "Analyze the overall intents from this formal document.",
            IntentContext.INTERACTIVE: "Synthesize the intents from this interactive communication."
        }

        depth_instructions = {
            IntentDepth.SURFACE: "Focus on the most obvious intentions across all sections.",
            IntentDepth.UNDERLYING: "Identify the deeper motivations and hidden goals throughout the text.",
            IntentDepth.COMPREHENSIVE: "Provide a complete psychological analysis of all conscious and subconscious intentions."
        }

        focus_instruction = ""
        if focus:
            focus_instruction = f" Pay special attention to {focus} throughout the analysis."

        return f"""Combine these section analyses into a comprehensive intent analysis of the complete communication.

{context_instructions[context_type]}
{depth_instructions[depth]}{focus_instruction}

Section analyses:
{combined_analyses}

Requirements:
- Synthesize information from all sections into a coherent understanding
- Identify the overarching primary intent and up to 3 secondary intents
- Determine the complexity and layering of intentions throughout
- Consider how different sections contribute to the overall goals
- The original text had approximately {original_length} characters
- Provide confidence scores and contextual factors
- Suggest an appropriate response approach

Create a unified intent analysis that captures the complete communication's purposes and motivations.

CRITICAL JSON FORMAT REQUIREMENTS:
- Respond with ONLY valid JSON - no other text before or after
- Use double quotes for all strings and keys

- All field names must match exactly: primary_intent, secondary_intents, intent_complexity, etc.
- Arrays must use square brackets [], objects must use curly braces {{}}
- No trailing commas, no comments, pure JSON only"""

    def analyze_conversation_intents(
        self,
        messages: List[dict],
        focus_participant: Optional[str] = None,
        depth: IntentDepth = IntentDepth.UNDERLYING
    ) -> Dict[str, IntentAnalysisOutput]:
        """
        Specialized method for analyzing intents in conversation history

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            focus_participant: Optional role to focus analysis on (e.g., "user", "assistant")
            depth: Depth of intent analysis to perform

        Returns:
            Dict mapping participant roles to their intent analyses (including deception assessment)

        Example:
            >>> analyzer = BasicIntentAnalyzer()
            >>> messages = [
            ...     {"role": "user", "content": "I'm having trouble with my code..."},
            ...     {"role": "assistant", "content": "I'd be happy to help..."},
            ...     {"role": "user", "content": "Actually, never mind, I figured it out."}
            ... ]
            >>> results = analyzer.analyze_conversation_intents(messages, focus_participant="user")
        """
        # Group messages by participant
        participant_messages = {}
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '').strip()
            
            if not content:
                continue
                
            if role not in participant_messages:
                participant_messages[role] = []
            participant_messages[role].append(content)

        # Analyze intents for each participant (or just the focused one)
        results = {}
        
        participants_to_analyze = [focus_participant] if focus_participant else list(participant_messages.keys())
        
        for role in participants_to_analyze:
            if role not in participant_messages:
                continue
                
            # Combine all messages from this participant
            combined_text = "\n\n".join(participant_messages[role])
            
            logger.debug("Analyzing conversation intents for participant", 
                        participant=role,
                        message_count=len(participant_messages[role]),
                        text_length=len(combined_text))
            
            # Analyze with conversational context (deception analysis always included)
            analysis = self.analyze_intent(
                combined_text,
                context_type=IntentContext.CONVERSATIONAL,
                depth=depth,
                focus=f"{role} intentions and goals in this conversation"
            )
            
            results[role] = analysis

        return results
