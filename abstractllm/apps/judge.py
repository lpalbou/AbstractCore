#!/usr/bin/env python3
"""
AbstractLLM Basic Judge CLI Application

Usage:
    python -m abstractllm.apps.judge <file_path_or_text> [file2] [file3] ... [options]

Options:
    --context <context>             Evaluation context description (e.g., "code review", "documentation assessment")
    --criteria <criteria>           Comma-separated criteria to evaluate (clarity,simplicity,actionability,soundness,innovation,effectiveness,relevance,completeness,coherence)
    --focus <focus>                  Specific focus areas for evaluation (e.g., "technical accuracy,performance,security")
    --reference <file_or_text>      Reference content for comparison-based evaluation
    --format <format>               Output format (json, yaml, plain, default: json)
    --output <output>               Output file path (optional, prints to console if not provided)
    --provider <provider>           LLM provider (requires --model)
    --model <model>                 LLM model (requires --provider)
    --temperature <temp>            Temperature for evaluation (default: 0.1 for consistency)
    --max-tokens <tokens>           Maximum total tokens for LLM context (default: 32000)
    --max-output-tokens <tokens>    Maximum tokens for LLM output generation (default: 8000)
    --verbose                       Show detailed progress information
    --debug                         Show raw LLM responses and detailed debugging information
    --include-criteria              Include detailed explanation of evaluation criteria in assessment
    --timeout <seconds>             HTTP timeout for LLM providers (default: 300)
    --help                          Show this help message

Examples:
    # Single file or text
    python -m abstractllm.apps.judge "This code is well-structured and solves the problem efficiently."
    python -m abstractllm.apps.judge document.py --context "code review" --criteria clarity,soundness,effectiveness

    # Multiple files (evaluated sequentially to avoid context overflow)
    python -m abstractllm.apps.judge file1.py file2.py file3.py --context "code review" --output assessments.json
    python -m abstractllm.apps.judge *.py --context "Python code review" --format plain
    python -m abstractllm.apps.judge docs/*.md --context "documentation review" --criteria clarity,completeness

    # Other options
    python -m abstractllm.apps.judge proposal.md --focus "technical accuracy,completeness,examples" --output assessment.json
    python -m abstractllm.apps.judge content.txt --reference ideal_solution.txt --format plain --verbose
    python -m abstractllm.apps.judge text.md --provider openai --model gpt-4o-mini --temperature 0.05
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, List

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..processing.basic_judge import BasicJudge, JudgmentCriteria, create_judge


def read_content(content_or_path: str) -> str:
    """
    Read content from file path or return as direct text

    Args:
        content_or_path: Either a file path or direct text content

    Returns:
        Content as string

    Raises:
        Exception: If file cannot be read
    """
    # Check if it's a file path
    try:
        file_path = Path(content_or_path)
        if file_path.exists() and file_path.is_file():
            # Try to read as text file
            try:
                # Try UTF-8 first
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Fallback to other encodings
                for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue

                # If all text encodings fail, try binary read and decode
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        return content.decode('utf-8', errors='ignore')
                except Exception as e:
                    raise Exception(f"Cannot read file {content_or_path}: {e}")
        else:
            # Treat as direct text content
            return content_or_path
    except Exception:
        # If path checking fails, treat as direct text
        return content_or_path


def parse_criteria_list(criteria_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated criteria string"""
    if not criteria_str:
        return None
    return [c.strip().lower() for c in criteria_str.split(',')]


def build_judgment_criteria(criteria_list: Optional[List[str]]) -> JudgmentCriteria:
    """Build JudgmentCriteria object from criteria list"""
    if not criteria_list:
        return JudgmentCriteria()  # All criteria enabled by default

    # Start with all criteria disabled
    criteria_kwargs = {
        'is_clear': False,
        'is_simple': False,
        'is_actionable': False,
        'is_sound': False,
        'is_innovative': False,
        'is_working': False,
        'is_relevant': False,
        'is_complete': False,
        'is_coherent': False
    }

    # Enable specified criteria
    criteria_mapping = {
        'clarity': 'is_clear',
        'simplicity': 'is_simple',
        'actionability': 'is_actionable',
        'soundness': 'is_sound',
        'innovation': 'is_innovative',
        'effectiveness': 'is_working',
        'relevance': 'is_relevant',
        'completeness': 'is_complete',
        'coherence': 'is_coherent'
    }

    for criterion in criteria_list:
        if criterion in criteria_mapping:
            criteria_kwargs[criteria_mapping[criterion]] = True
        else:
            # Show warning for unknown criteria but continue
            print(f"Warning: Unknown criterion '{criterion}'. Available criteria: {', '.join(criteria_mapping.keys())}")

    return JudgmentCriteria(**criteria_kwargs)


def format_assessment_plain(assessment: dict) -> str:
    """Format assessment as human-readable plain text"""
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("LLM JUDGE ASSESSMENT")
    lines.append("=" * 60)
    lines.append("")

    # Judge's Summary (first thing shown)
    judge_summary = assessment.get('judge_summary', '')
    if judge_summary:
        lines.append("📝 Judge's Summary:")
        lines.append("-" * 18)
        lines.append(judge_summary)
        lines.append("")

    # Source Reference
    source_ref = assessment.get('source_reference', '')
    if source_ref:
        lines.append(f"📄 Source: {source_ref}")
        lines.append("")

    # Context and overall score
    lines.append(f"Context: {assessment.get('evaluation_context', 'General evaluation')}")
    lines.append(f"Overall Score: {assessment.get('overall_score', 0)}/5")
    lines.append("")

    # Individual scores
    score_fields = [
        ('clarity_score', 'Clarity'),
        ('simplicity_score', 'Simplicity'),
        ('actionability_score', 'Actionability'),
        ('soundness_score', 'Soundness'),
        ('innovation_score', 'Innovation'),
        ('effectiveness_score', 'Effectiveness'),
        ('relevance_score', 'Relevance'),
        ('completeness_score', 'Completeness'),
        ('coherence_score', 'Coherence')
    ]

    lines.append("Individual Scores:")
    lines.append("-" * 20)
    for field, label in score_fields:
        score = assessment.get(field)
        if score is not None:
            lines.append(f"{label:15}: {score}/5")
    lines.append("")

    # Strengths
    strengths = assessment.get('strengths', [])
    if strengths:
        lines.append("Strengths:")
        lines.append("-" * 10)
        for strength in strengths:
            lines.append(f"• {strength}")
        lines.append("")

    # Weaknesses
    weaknesses = assessment.get('weaknesses', [])
    if weaknesses:
        lines.append("Areas for Improvement:")
        lines.append("-" * 25)
        for weakness in weaknesses:
            lines.append(f"• {weakness}")
        lines.append("")

    # Actionable feedback
    feedback = assessment.get('actionable_feedback', [])
    if feedback:
        lines.append("Actionable Recommendations:")
        lines.append("-" * 28)
        for item in feedback:
            lines.append(f"• {item}")
        lines.append("")

    # Evaluation Criteria Details (if included)
    criteria_details = assessment.get('evaluation_criteria_details')
    if criteria_details:
        lines.append("Evaluation Criteria Details:")
        lines.append("-" * 29)
        lines.append(criteria_details)
        lines.append("")

    # Reasoning
    reasoning = assessment.get('reasoning', '')
    if reasoning:
        lines.append("Evaluation Reasoning:")
        lines.append("-" * 21)
        lines.append(reasoning)
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="AbstractLLM Basic Judge - LLM-as-a-judge for objective evaluation (Default: qwen3:4b-instruct-2507-q4_K_M)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file or text
  python -m abstractllm.apps.judge "This code is well-structured."
  python -m abstractllm.apps.judge document.py --context "code review" --criteria clarity,soundness
  python -m abstractllm.apps.judge proposal.md --focus "technical accuracy,examples" --output assessment.json

  # Multiple files (evaluated sequentially)
  python -m abstractllm.apps.judge file1.py file2.py file3.py --context "code review" --format json
  python -m abstractllm.apps.judge docs/*.md --context "documentation review" --format plain

  # Other options
  python -m abstractllm.apps.judge content.txt --reference ideal.txt --format plain --verbose
  python -m abstractllm.apps.judge text.md --provider openai --model gpt-4o-mini

Available criteria:
  clarity, simplicity, actionability, soundness, innovation, effectiveness,
  relevance, completeness, coherence

Output formats:
  - json: Structured JSON format (default)
  - yaml: YAML format
  - plain: Human-readable text format

Default model setup:
  - Requires Ollama: https://ollama.com/
  - Download model: ollama pull qwen3:4b-instruct-2507-q4_K_M
  - For best evaluation: qwen3-coder:30b or gpt-oss:120b
        """
    )

    parser.add_argument(
        'content',
        nargs='+',
        help='Content to evaluate: single text string, single file path, or multiple file paths'
    )

    parser.add_argument(
        '--context',
        help='Evaluation context description (e.g., "code review", "documentation assessment")'
    )

    parser.add_argument(
        '--criteria',
        help='Comma-separated criteria to evaluate (clarity,simplicity,actionability,soundness,innovation,effectiveness,relevance,completeness,coherence)'
    )

    parser.add_argument(
        '--focus',
        help='Specific focus areas for evaluation (e.g., "technical accuracy,performance,security")'
    )

    parser.add_argument(
        '--reference',
        help='Reference content for comparison-based evaluation (file path or direct text)'
    )

    # Build format choices based on available dependencies
    format_choices = ['json', 'plain']
    if YAML_AVAILABLE:
        format_choices.append('yaml')

    parser.add_argument(
        '--format',
        choices=format_choices,
        default='json',
        help='Output format: json (structured), plain (human-readable)' + (', yaml' if YAML_AVAILABLE else ' - install PyYAML for YAML support')
    )

    parser.add_argument(
        '--output',
        help='Output file path (prints to console if not provided)'
    )

    parser.add_argument(
        '--provider',
        help='LLM provider (requires --model)'
    )

    parser.add_argument(
        '--model',
        help='LLM model (requires --provider)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature for evaluation (default: 0.1 for consistency)'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=32000,
        help='Maximum total tokens for LLM context (default: 32000)'
    )

    parser.add_argument(
        '--max-output-tokens',
        type=int,
        default=8000,
        help='Maximum tokens for LLM output generation (default: 8000)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show raw LLM responses and detailed debugging information'
    )

    parser.add_argument(
        '--include-criteria',
        action='store_true',
        help='Include detailed explanation of evaluation criteria in assessment'
    )

    parser.add_argument(
        '--exclude-global',
        action='store_true',
        help='Skip global assessment for multiple files (default: False, global assessment included)'
    )

    parser.add_argument(
        '--timeout',
        type=float,
        default=300.0,
        help='HTTP request timeout in seconds for LLM providers (default: 300)'
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Validate temperature
        if not 0.0 <= args.temperature <= 2.0:
            print("Error: Temperature must be between 0.0 and 2.0")
            sys.exit(1)

        # Validate timeout
        if args.timeout < 30.0:
            print("Error: Timeout must be at least 30 seconds")
            sys.exit(1)

        # Validate provider/model pair
        if args.provider and not args.model:
            print("Error: --model is required when --provider is specified")
            sys.exit(1)

        if args.model and not args.provider:
            print("Error: --provider is required when --model is specified")
            sys.exit(1)

        # Import Path at the top of this section to avoid scoping issues
        from pathlib import Path

        # Determine if we have multiple files or single content
        if len(args.content) == 1:
            # Single argument - could be text or file
            single_input = args.content[0]

            # Check if it looks like a file path (and exists)
            is_file = Path(single_input).exists() and Path(single_input).is_file()

            if is_file and args.verbose:
                print(f"Evaluating single file: {single_input}")
            elif not is_file and args.verbose:
                print(f"Evaluating direct text content")

            # Use evaluate_files for consistency (handles single file too)
            file_or_text = [single_input]
        else:
            # Multiple arguments - assume all are file paths
            if args.verbose:
                print(f"Evaluating {len(args.content)} files: {', '.join(args.content)}")
            file_or_text = args.content

        # Read reference if provided
        reference = None
        if args.reference:
            if args.verbose:
                print(f"Reading reference: {args.reference}")
            reference = read_content(args.reference)

        # Parse criteria
        criteria_list = parse_criteria_list(args.criteria)
        focus = args.focus
        judgment_criteria = build_judgment_criteria(criteria_list)

        # Initialize judge
        if args.provider and args.model:
            if args.verbose:
                print(f"Initializing BasicJudge ({args.provider}, {args.model}, temperature={args.temperature}, {args.max_tokens} token context, {args.max_output_tokens} output tokens)...")

            judge = create_judge(
                provider=args.provider,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_output_tokens=args.max_output_tokens,
                debug=args.debug,
                timeout=args.timeout
            )
        else:
            if args.verbose:
                print(f"Initializing BasicJudge (ollama, qwen3:4b-instruct-2507-q4_K_M, temperature={args.temperature}, {args.max_tokens} token context, {args.max_output_tokens} output tokens)...")

            try:
                judge = BasicJudge(
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_output_tokens=args.max_output_tokens,
                    debug=args.debug
                )
            except RuntimeError as e:
                print(f"\n{e}")
                print("\n🚀 Quick alternatives to get started:")
                print("   - Use --provider and --model to specify an available provider")
                print("   - Example: judge 'content' --provider openai --model gpt-4o-mini")
                sys.exit(1)

        # Perform evaluation
        if args.verbose:
            context = args.context or "file content evaluation"
            print(f"Evaluation context: {context}")

        start_time = time.time()

        # Handle both single content and multiple files
        if len(file_or_text) == 1:
            # Try to determine if it's a file or text content
            single_input = file_or_text[0]
            from pathlib import Path

            if Path(single_input).exists() and Path(single_input).is_file():
                # It's a file - use evaluate_files
                assessment = judge.evaluate_files(
                    file_paths=single_input,
                    context=args.context,
                    criteria=judgment_criteria,
                    focus=focus,
                    reference=reference,
                    include_criteria=args.include_criteria,
                    exclude_global=args.exclude_global
                )
            else:
                # It's text content - use evaluate
                assessment = judge.evaluate(
                    content=single_input,
                    context=args.context,
                    criteria=judgment_criteria,
                    focus=focus,
                    reference=reference,
                    include_criteria=args.include_criteria
                )
        else:
            # Multiple files - use evaluate_files
            assessment = judge.evaluate_files(
                file_paths=file_or_text,
                context=args.context,
                criteria=judgment_criteria,
                focus=focus,
                reference=reference,
                include_criteria=args.include_criteria,
                exclude_global=args.exclude_global
            )

        end_time = time.time()

        if args.verbose:
            duration = end_time - start_time
            if isinstance(assessment, list):
                # Original format: list of assessments (exclude_global=True)
                print(f"\nEvaluation completed in {duration:.2f} seconds")
                print(f"Evaluated {len(assessment)} files:")
                for i, result in enumerate(assessment):
                    overall_score = result.get('overall_score', 0)
                    source = result.get('source_reference', f'File {i+1}')
                    print(f"  {source}: {overall_score}/5")
            elif isinstance(assessment, dict) and 'global' in assessment and 'files' in assessment:
                # New format: global + files (exclude_global=False)
                print(f"\nEvaluation completed in {duration:.2f} seconds")
                global_score = assessment['global'].get('overall_score', 0)
                file_count = len(assessment['files'])
                print(f"Global assessment: {global_score}/5")
                print(f"Individual files ({file_count}):")
                for result in assessment['files']:
                    overall_score = result.get('overall_score', 0)
                    source = result.get('source_reference', f'File')
                    print(f"  {source}: {overall_score}/5")
            else:
                # Single assessment
                overall_score = assessment.get('overall_score', 0)
                print(f"\nEvaluation completed in {duration:.2f} seconds")
                print(f"Overall assessment: {overall_score}/5")

        # Format output
        if args.format == 'json':
            formatted_output = json.dumps(assessment, indent=2, ensure_ascii=False)
        elif args.format == 'plain':
            if isinstance(assessment, list):
                # Original format: list of assessments (exclude_global=True)
                output_parts = []
                for i, single_assessment in enumerate(assessment):
                    if i > 0:
                        output_parts.append("\n" + "=" * 80 + "\n")
                    output_parts.append(format_assessment_plain(single_assessment))
                formatted_output = "\n".join(output_parts)
            elif isinstance(assessment, dict) and 'global' in assessment and 'files' in assessment:
                # New format: global + files (exclude_global=False) - Global assessment first
                output_parts = []

                # Add global assessment first
                output_parts.append("🌍 GLOBAL ASSESSMENT")
                output_parts.append("=" * 80)
                output_parts.append(format_assessment_plain(assessment['global']))

                # Add individual file assessments
                output_parts.append("\n" + "📁 INDIVIDUAL FILE ASSESSMENTS")
                output_parts.append("=" * 80)
                for i, single_assessment in enumerate(assessment['files']):
                    if i > 0:
                        output_parts.append("\n" + "-" * 40 + "\n")
                    output_parts.append(format_assessment_plain(single_assessment))

                formatted_output = "\n".join(output_parts)
            else:
                # Single assessment
                formatted_output = format_assessment_plain(assessment)
        elif args.format == 'yaml':
            if not YAML_AVAILABLE:
                print("Error: PyYAML is required for YAML output format. Install with: pip install PyYAML")
                sys.exit(1)
            formatted_output = yaml.dump(assessment, default_flow_style=False, indent=2, sort_keys=False)
        else:
            formatted_output = json.dumps(assessment, indent=2, ensure_ascii=False)

        # Output result
        if args.output:
            # Write to file
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            if args.verbose:
                print(f"Assessment saved to: {output_path}")
        else:
            # Print to console
            print(formatted_output)

    except KeyboardInterrupt:
        print("\nEvaluation cancelled by user")
        sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()