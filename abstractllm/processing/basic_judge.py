"""
Basic Judge - Production-ready LLM-as-a-judge for objective evaluation

Features:
- Structured JSON assessment with clear scoring rubrics
- Chain-of-thought reasoning for transparent evaluation
- Configurable evaluation criteria with default standards
- Critical assessment with constructive skepticism
- Clear, simple and actionable feedback
"""

from typing import Optional, List, Dict, Any, Union
import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field

from ..core.interface import AbstractLLMInterface
from ..core.factory import create_llm
from ..structured.retry import FeedbackRetry

logger = logging.getLogger(__name__)


class JudgmentCriteria(BaseModel):
    """Default evaluation criteria for the judge"""
    is_clear: bool = Field(True, description="Evaluate clarity and understandability")
    is_simple: bool = Field(True, description="Evaluate appropriate simplicity vs complexity")
    is_actionable: bool = Field(True, description="Evaluate if it provides actionable insights")
    is_sound: bool = Field(True, description="Evaluate logical soundness and reasoning")
    is_innovative: bool = Field(True, description="Evaluate creativity and novel thinking")
    is_working: bool = Field(True, description="Evaluate if it solves the intended problem")
    is_relevant: bool = Field(True, description="Evaluate relevance to context/question")
    is_complete: bool = Field(True, description="Evaluate completeness of coverage")
    is_coherent: bool = Field(True, description="Evaluate logical flow and consistency")


class Assessment(BaseModel):
    """Structured assessment result"""
    overall_score: int = Field(..., description="Overall assessment score (1-5)")

    # Judge's Summary (new)
    judge_summary: str = Field(..., description="Judge's experiential note summarizing the assessment task and key findings")
    source_reference: str = Field(..., description="Reference to what was assessed (file, content type, context)")

    # Individual criterion scores
    clarity_score: Optional[int] = Field(None, description="Clarity score (1-5)")
    simplicity_score: Optional[int] = Field(None, description="Simplicity score (1-5)")
    actionability_score: Optional[int] = Field(None, description="Actionability score (1-5)")
    soundness_score: Optional[int] = Field(None, description="Soundness score (1-5)")
    innovation_score: Optional[int] = Field(None, description="Innovation score (1-5)")
    effectiveness_score: Optional[int] = Field(None, description="Working/effectiveness score (1-5)")
    relevance_score: Optional[int] = Field(None, description="Relevance score (1-5)")
    completeness_score: Optional[int] = Field(None, description="Completeness score (1-5)")
    coherence_score: Optional[int] = Field(None, description="Coherence score (1-5)")

    # Detailed evaluation
    strengths: List[str] = Field(default_factory=list, description="Key strengths identified")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    actionable_feedback: List[str] = Field(default_factory=list, description="Specific actionable recommendations")

    # Reasoning
    reasoning: str = Field(..., description="Chain-of-thought reasoning for the assessment")

    # Metadata
    evaluation_context: str = Field(..., description="What was being evaluated")
    criteria_used: List[str] = Field(default_factory=list, description="Criteria used for evaluation")

    # Optional detailed criteria explanations (new)
    evaluation_criteria_details: Optional[str] = Field(None, description="Detailed explanation of evaluation criteria used")


class BasicJudge:
    """
    Basic Judge for objective LLM-as-a-judge evaluation

    Key features:
    - Structured JSON assessment with 1-5 scoring rubric
    - Chain-of-thought reasoning for transparency
    - Default evaluation criteria covering key quality dimensions
    - Support for custom criteria and evaluation contexts
    - Critical assessment with constructive skepticism
    - Clear, simple and actionable feedback
    - Multiple file evaluation with sequential processing to avoid context overflow

    Examples:
        >>> judge = BasicJudge()

        # Evaluate content against default criteria
        >>> result = judge.evaluate("This code is well-structured and solves the problem elegantly.")
        >>> print(f"Overall score: {result['overall_score']}/5")

        # Evaluate single file
        >>> result = judge.evaluate_files("document.py", context="code review")
        >>> print(f"File assessment: {result['overall_score']}/5")

        # Evaluate multiple files sequentially (returns list of assessments)
        >>> results = judge.evaluate_files(["file1.py", "file2.py", "file3.py"], context="code review")
        >>> for i, result in enumerate(results):
        ...     print(f"File {i+1}: {result['overall_score']}/5")

        # Evaluate with custom criteria
        >>> result = judge.evaluate(
        ...     content="The API documentation explains endpoints clearly.",
        ...     context="API documentation review",
        ...     custom_criteria=["has_examples", "covers_error_cases"]
        ... )

        # Evaluate with specific criteria focus
        >>> criteria = JudgmentCriteria(is_innovative=False, is_working=False)  # Focus on clarity/simplicity
        >>> result = judge.evaluate("Text content", criteria=criteria)
    """

    def __init__(
        self,
        llm: Optional[AbstractLLMInterface] = None,
        temperature: float = 0.1  # Low temperature for consistent evaluation
    ):
        """Initialize the judge"""
        if llm is None:
            try:
                # Use low temperature for consistent evaluation
                self.llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M",
                                    max_tokens=32000, max_output_tokens=4000, temperature=temperature)
            except Exception as e:
                error_msg = (
                    f"❌ Failed to initialize default Ollama model 'qwen3:4b-instruct-2507-q4_K_M': {e}\n\n"
                    "💡 To use the default model, please:\n"
                    "   1. Install Ollama from: https://ollama.com/\n"
                    "   2. Download the model: ollama pull qwen3:4b-instruct-2507-q4_K_M\n"
                    "   3. Start Ollama service\n\n"
                    "⚡ For best evaluation quality, consider these models:\n"
                    "   - qwen3-coder:30b (excellent for detailed assessment, requires 32GB RAM)\n"
                    "   - gpt-oss:120b (highest quality evaluation, requires 120GB RAM)\n\n"
                    "🔧 Alternatively, provide a custom LLM instance:\n"
                    "   from abstractllm import create_llm\n"
                    "   from abstractllm.processing import BasicJudge\n"
                    "   \n"
                    "   llm = create_llm('openai', model='gpt-4o-mini', temperature=0.1)\n"
                    "   judge = BasicJudge(llm)"
                )
                raise RuntimeError(error_msg) from e
        else:
            self.llm = llm

        self.retry_strategy = FeedbackRetry(max_attempts=3)

    def evaluate(
        self,
        content: str,
        context: Optional[str] = None,
        criteria: Optional[JudgmentCriteria] = None,
        custom_criteria: Optional[List[str]] = None,
        reference: Optional[str] = None,
        include_criteria: bool = False
    ) -> dict:
        """
        Evaluate content against specified criteria

        Args:
            content: The content to evaluate
            context: Optional context describing what is being evaluated
            criteria: JudgmentCriteria object specifying which standard criteria to use
            custom_criteria: List of additional custom criteria to evaluate
            reference: Optional reference/expected output for comparison
            include_criteria: Include detailed explanation of evaluation criteria in assessment

        Returns:
            dict: Structured assessment result
        """

        # Set default criteria if none provided
        if criteria is None:
            criteria = JudgmentCriteria()

        # Set default context if none provided
        if context is None:
            context = "general content evaluation"

        logger.info(f"Starting evaluation: {context}")

        # Build the evaluation prompt
        prompt = self._build_evaluation_prompt(content, context, criteria, custom_criteria, reference, include_criteria)

        # Generate structured assessment
        try:
            result = self.llm.generate(
                prompt,
                response_model=Assessment,
                retry_strategy=self.retry_strategy
            )

            # Convert to dict and add metadata
            assessment_dict = result.dict() if hasattr(result, 'dict') else result

            # Log results
            overall_score = assessment_dict.get('overall_score', 0)
            logger.info(f"Evaluation completed: {overall_score}/5 overall score")

            return assessment_dict

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return basic failure assessment
            return {
                "overall_score": 1,
                "judge_summary": f"I was asked to evaluate content in the context of '{context}', but encountered a technical error that prevented completion of the assessment.",
                "source_reference": f"Content evaluation in context: {context}",
                "reasoning": f"Evaluation failed due to technical error: {str(e)}",
                "evaluation_context": context,
                "criteria_used": [],
                "strengths": [],
                "weaknesses": ["Technical evaluation failure"],
                "actionable_feedback": ["Please retry the evaluation or check the input content"],
                "evaluation_criteria_details": None
            }

    def evaluate_files(
        self,
        file_paths: Union[str, List[str]],
        context: Optional[str] = None,
        criteria: Optional[JudgmentCriteria] = None,
        custom_criteria: Optional[List[str]] = None,
        reference: Optional[str] = None,
        include_criteria: bool = False,
        max_file_size: int = 1000000,  # 1MB default limit per file
        exclude_global: bool = False  # Include global assessment by default
    ) -> Union[dict, List[dict]]:
        """
        Evaluate content from one or multiple files sequentially to avoid context overflow

        Args:
            file_paths: Single file path or list of file paths to evaluate
            context: Optional context describing what is being evaluated
            criteria: JudgmentCriteria object specifying which standard criteria to use
            custom_criteria: List of additional custom criteria to evaluate
            reference: Optional reference/expected output for comparison
            include_criteria: Include detailed explanation of evaluation criteria in assessment
            max_file_size: Maximum file size in bytes (default 1MB to avoid context overflow)
            exclude_global: If True, skip global assessment for multiple files (default False)

        Returns:
            dict: Single assessment if one file provided
            dict: {"global": global_assessment, "files": [assessments]} if multiple files and exclude_global=False
            List[dict]: List of assessments if multiple files and exclude_global=True

        Raises:
            FileNotFoundError: If any file doesn't exist
            ValueError: If file is too large or can't be read
        """

        # Handle single file path
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        logger.info(f"Starting evaluation of {len(file_paths)} file(s)")

        # Set default context if none provided
        if context is None:
            context = "file content evaluation"

        assessments = []

        for i, file_path_str in enumerate(file_paths):
            file_path = Path(file_path_str)

            # Validate file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")

            # Check file size to avoid context overflow
            file_size = file_path.stat().st_size
            if file_size > max_file_size:
                raise ValueError(
                    f"File {file_path.name} is too large ({file_size:,} bytes). "
                    f"Maximum allowed size is {max_file_size:,} bytes. "
                    f"Consider splitting the file or increasing max_file_size parameter."
                )

            # Read file content
            try:
                # Try UTF-8 first
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback to other encodings
                for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all text encodings fail, try binary read and decode with errors ignored
                    try:
                        with open(file_path, 'rb') as f:
                            file_content = f.read()
                            content = file_content.decode('utf-8', errors='ignore')
                    except Exception as e:
                        raise ValueError(f"Cannot read file {file_path.name}: {e}")

            if not content.strip():
                logger.warning(f"File {file_path.name} is empty or contains no readable content")
                # Create minimal assessment for empty file
                assessment = {
                    "overall_score": 1,
                    "judge_summary": f"I was asked to evaluate file '{file_path.name}' but found it to be empty or containing no readable content.",
                    "source_reference": f"File: {file_path.name} (context: {context})",
                    "reasoning": "Cannot evaluate empty or unreadable file content",
                    "evaluation_context": context,
                    "criteria_used": [],
                    "strengths": [],
                    "weaknesses": ["File is empty or unreadable"],
                    "actionable_feedback": ["Ensure file contains valid content and is properly encoded"],
                    "evaluation_criteria_details": None
                }
                assessments.append(assessment)
                continue

            # Create file-specific context
            file_context = f"{context} (file: {file_path.name})"

            logger.info(f"Evaluating file {i+1}/{len(file_paths)}: {file_path.name} ({len(content):,} characters)")

            # Evaluate the file content
            assessment = self.evaluate(
                content=content,
                context=file_context,
                criteria=criteria,
                custom_criteria=custom_criteria,
                reference=reference,
                include_criteria=include_criteria
            )

            # Update source reference to include file name
            assessment['source_reference'] = f"File: {file_path.name} (context: {context})"

            assessments.append(assessment)

        logger.info(f"Completed evaluation of {len(file_paths)} file(s)")

        # Return single assessment if only one file
        if len(assessments) == 1:
            return assessments[0]

        # Multiple files: handle global assessment
        if exclude_global:
            # Return list of assessments (original behavior)
            return assessments
        else:
            # Generate global assessment and return structured result
            logger.info("Generating global assessment from individual file evaluations...")
            global_assessment = self._generate_global_assessment(
                assessments, context, criteria, custom_criteria, include_criteria
            )

            return {
                "global": global_assessment,
                "files": assessments
            }

    def _generate_global_assessment(
        self,
        individual_assessments: List[dict],
        context: str,
        criteria: Optional[JudgmentCriteria],
        custom_criteria: Optional[List[str]],
        include_criteria: bool
    ) -> dict:
        """
        Generate a global assessment from multiple individual file assessments

        Args:
            individual_assessments: List of individual file assessment dictionaries
            context: The evaluation context
            criteria: JudgmentCriteria used for individual evaluations
            custom_criteria: Custom criteria used for individual evaluations
            include_criteria: Whether to include detailed criteria explanations

        Returns:
            dict: Global assessment summarizing all individual assessments
        """

        # Build summary content from individual assessments
        file_summaries = []
        total_files = len(individual_assessments)

        # Calculate aggregate statistics
        overall_scores = [a.get('overall_score', 0) for a in individual_assessments]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        score_distribution = {
            5: sum(1 for s in overall_scores if s >= 4.5),
            4: sum(1 for s in overall_scores if 3.5 <= s < 4.5),
            3: sum(1 for s in overall_scores if 2.5 <= s < 3.5),
            2: sum(1 for s in overall_scores if 1.5 <= s < 2.5),
            1: sum(1 for s in overall_scores if s < 1.5)
        }

        # Collect all strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        all_feedback = []

        for i, assessment in enumerate(individual_assessments):
            file_ref = assessment.get('source_reference', f'File {i+1}')
            score = assessment.get('overall_score', 0)

            # Keep file summaries very concise to avoid context overflow
            file_name = file_ref.split(":")[-1].strip() if ":" in file_ref else f"File {i+1}"
            file_summaries.append(f"{file_name}: {score}/5")

            # Collect feedback for pattern analysis
            strengths = assessment.get('strengths', [])
            weaknesses = assessment.get('weaknesses', [])
            feedback = assessment.get('actionable_feedback', [])

            all_strengths.extend(strengths)
            all_weaknesses.extend(weaknesses)
            all_feedback.extend(feedback)

        # Extract most common patterns (limit to avoid context overflow)
        common_strengths = list(set(all_strengths))[:3]  # Top 3 unique strengths
        common_weaknesses = list(set(all_weaknesses))[:3]  # Top 3 unique weaknesses
        key_recommendations = list(set(all_feedback))[:3]  # Top 3 unique recommendations

        # Create concise content for global evaluation
        global_content = f"""GLOBAL ASSESSMENT REQUEST: Evaluate {total_files} files in "{context}" context.

FILE SCORES: {', '.join(file_summaries)}
AVERAGE: {avg_score:.1f}/5 (Excellent:{score_distribution[5]}, Good:{score_distribution[4]}, Adequate:{score_distribution[3]}, Poor:{score_distribution[2]}, Very Poor:{score_distribution[1]})

KEY PATTERNS:
Strengths: {'; '.join(common_strengths) if common_strengths else 'Various individual strengths'}
Weaknesses: {'; '.join(common_weaknesses) if common_weaknesses else 'Various individual weaknesses'}
Recommendations: {'; '.join(key_recommendations) if key_recommendations else 'See individual assessments'}

Provide a comprehensive global assessment of overall quality and recommendations."""

        # Generate global assessment using the judge
        try:
            global_assessment = self.evaluate(
                content=global_content,
                context=f"global assessment summary for {total_files} files ({context})",
                criteria=criteria,
                custom_criteria=custom_criteria,
                include_criteria=include_criteria
            )

            # Update the source reference to indicate this is a global assessment
            global_assessment['source_reference'] = f"Global assessment of {total_files} files (context: {context})"

            # Enhance the judge summary to indicate this is a global assessment
            original_summary = global_assessment.get('judge_summary', '')
            global_assessment['judge_summary'] = f"I conducted a global assessment synthesizing evaluations of {total_files} files. {original_summary}"

            return global_assessment

        except Exception as e:
            logger.error(f"Global assessment generation failed: {e}")
            # Return fallback global assessment
            return {
                "overall_score": int(round(avg_score)),
                "judge_summary": f"I was asked to provide a global assessment of {total_files} files in the context of '{context}', but encountered a technical error. Based on individual scores, the average quality appears to be {avg_score:.1f}/5.",
                "source_reference": f"Global assessment of {total_files} files (context: {context})",
                "reasoning": f"Global assessment failed due to technical error: {str(e)}. Fallback assessment based on {total_files} individual file scores with average {avg_score:.1f}/5.",
                "evaluation_context": f"global assessment summary for {total_files} files ({context})",
                "criteria_used": [],
                "strengths": list(set(all_strengths))[:5] if all_strengths else [],
                "weaknesses": list(set(all_weaknesses))[:5] if all_weaknesses else [],
                "actionable_feedback": ["Review individual file assessments for detailed recommendations", "Address common patterns identified across files"],
                "evaluation_criteria_details": None
            }

    def _build_evaluation_prompt(
        self,
        content: str,
        context: str,
        criteria: JudgmentCriteria,
        custom_criteria: Optional[List[str]],
        reference: Optional[str],
        include_criteria: bool = False
    ) -> str:
        """Build the evaluation prompt with chain-of-thought reasoning"""

        # Build active criteria list
        active_criteria = []
        criteria_descriptions = []

        if criteria.is_clear:
            active_criteria.append("clarity")
            criteria_descriptions.append("- **Clarity**: How clear, understandable, and well-explained is the content?")

        if criteria.is_simple:
            active_criteria.append("simplicity")
            criteria_descriptions.append("- **Simplicity**: Is it appropriately simple vs unnecessarily complex for its purpose?")

        if criteria.is_actionable:
            active_criteria.append("actionability")
            criteria_descriptions.append("- **Actionability**: Does it provide actionable insights, recommendations, or next steps?")

        if criteria.is_sound:
            active_criteria.append("soundness")
            criteria_descriptions.append("- **Soundness**: Is the reasoning logical, well-founded, and free of errors?")

        if criteria.is_innovative:
            active_criteria.append("innovation")
            criteria_descriptions.append("- **Innovation**: Does it show creativity, novel thinking, or fresh approaches?")

        if criteria.is_working:
            active_criteria.append("effectiveness")
            criteria_descriptions.append("- **Effectiveness**: Does it actually solve the intended problem or achieve its purpose?")

        if criteria.is_relevant:
            active_criteria.append("relevance")
            criteria_descriptions.append("- **Relevance**: Is it relevant and appropriate to the context and requirements?")

        if criteria.is_complete:
            active_criteria.append("completeness")
            criteria_descriptions.append("- **Completeness**: Does it address all important aspects comprehensively?")

        if criteria.is_coherent:
            active_criteria.append("coherence")
            criteria_descriptions.append("- **Coherence**: Is the flow logical, consistent, and well-structured?")

        # Add custom criteria
        if custom_criteria:
            for custom in custom_criteria:
                active_criteria.append(custom)
                criteria_descriptions.append(f"- **{custom.title()}**: Custom evaluation criterion")

        criteria_text = "\n".join(criteria_descriptions)

        # Build reference section if provided
        reference_section = ""
        if reference:
            reference_section = f"""
REFERENCE FOR COMPARISON:
{reference}

When evaluating, consider how the content compares to this reference in terms of quality and approach.
"""

        # Build criteria details section if requested
        criteria_details_section = ""
        if include_criteria:
            criteria_details_section = f"""

DETAILED CRITERIA EXPLANATIONS:
{criteria_text}

These criteria form the foundation of this assessment. Each criterion is evaluated independently before calculating the overall score."""

        # Determine source reference
        source_ref = f"Content evaluation in context: {context}"
        if len(content) > 50:
            content_preview = content[:50] + "..."
            source_ref = f"Content: '{content_preview}' (context: {context})"
        else:
            source_ref = f"Content: '{content}' (context: {context})"

        # Build the evaluation prompt
        prompt = f"""You are an expert evaluator conducting a critical assessment with constructive skepticism. Your role is to provide objective, fair, and actionable evaluation.

EVALUATION CONTEXT: {context}

CONTENT TO EVALUATE:
{content}
{reference_section}
EVALUATION CRITERIA:
{criteria_text}{criteria_details_section}

SCORING RUBRIC (1-5 scale):
- **Score 5**: Exceptional - Exceeds expectations in this dimension
- **Score 4**: Good - Meets expectations well with minor room for improvement
- **Score 3**: Adequate - Meets basic expectations but has notable areas for improvement
- **Score 2**: Poor - Falls short of expectations with significant issues
- **Score 1**: Very Poor - Fails to meet basic standards in this dimension

EVALUATION PROCESS:
1. **STEP 1**: Carefully analyze the content for each active criterion
2. **STEP 2**: Identify specific strengths and weaknesses
3. **STEP 3**: Provide actionable recommendations for improvement
4. **STEP 4**: Assign scores based on the rubric (be fair but appropriately critical)
5. **STEP 5**: Calculate overall score as average of individual scores

CRITICAL ASSESSMENT PRINCIPLES:
- Be objective and evidence-based in your evaluation
- Apply constructive skepticism - question assumptions and look for gaps
- Focus on providing clear, simple, and actionable feedback
- Balance recognition of strengths with honest identification of weaknesses
- Ensure recommendations are specific and implementable

RESPONSE FORMAT:
Provide your assessment as a structured JSON response with the following format:

{{
    "overall_score": <1-5 integer>,
    "judge_summary": "A brief experiential note from your perspective as the judge: what you were asked to evaluate, what you found, and your key assessment insights (2-3 sentences)",
    "source_reference": "{source_ref}",
    "clarity_score": <1-5 integer or null if not evaluated>,
    "simplicity_score": <1-5 integer or null if not evaluated>,
    "actionability_score": <1-5 integer or null if not evaluated>,
    "soundness_score": <1-5 integer or null if not evaluated>,
    "innovation_score": <1-5 integer or null if not evaluated>,
    "effectiveness_score": <1-5 integer or null if not evaluated>,
    "relevance_score": <1-5 integer or null if not evaluated>,
    "completeness_score": <1-5 integer or null if not evaluated>,
    "coherence_score": <1-5 integer or null if not evaluated>,
    "strengths": ["list of specific strengths identified"],
    "weaknesses": ["list of specific areas for improvement"],
    "actionable_feedback": ["list of specific actionable recommendations"],
    "reasoning": "Your step-by-step chain-of-thought analysis and justification for the scores",
    "evaluation_context": "{context}",
    "criteria_used": {json.dumps(active_criteria)},
    "evaluation_criteria_details": {'"Detailed explanation of the evaluation criteria and their meaning in this assessment context"' if include_criteria else 'null'}
}}

Begin your evaluation now."""

        return prompt


def create_judge(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs
) -> BasicJudge:
    """
    Create a BasicJudge instance with specified provider and model

    Args:
        provider: LLM provider (e.g., "ollama", "openai", "anthropic")
        model: Model name
        temperature: Temperature for evaluation (default 0.1 for consistency)
        **kwargs: Additional arguments passed to create_llm

    Returns:
        BasicJudge instance
    """
    if provider and model:
        llm = create_llm(provider, model=model, temperature=temperature, **kwargs)
        return BasicJudge(llm)
    else:
        return BasicJudge()