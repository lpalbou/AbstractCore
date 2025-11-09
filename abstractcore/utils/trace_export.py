"""
Trace export utilities for interaction observability.

Provides functions to export LLM interaction traces to various formats
for debugging, analysis, and compliance purposes.
"""

import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime


def export_traces(
    traces: Union[Dict[str, Any], List[Dict[str, Any]]],
    format: str = 'jsonl',
    file_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Export interaction traces to file or return formatted string.

    Args:
        traces: Single trace dict or list of trace dicts
        format: Output format - 'jsonl', 'json', or 'markdown'
        file_path: Optional file path to write to. If None, returns string.

    Returns:
        Formatted trace data as string

    Raises:
        ValueError: If format is not supported

    Examples:
        >>> # Export single trace to JSONL
        >>> trace = llm.get_traces(trace_id="...")
        >>> export_traces(trace, format='jsonl', file_path='trace.jsonl')

        >>> # Export multiple traces to JSON
        >>> traces = llm.get_traces(last_n=10)
        >>> export_traces(traces, format='json', file_path='traces.json')

        >>> # Get markdown report as string
        >>> report = export_traces(traces, format='markdown')
        >>> print(report)
    """
    # Normalize to list
    if isinstance(traces, dict):
        traces = [traces]

    # Validate format
    if format not in ['jsonl', 'json', 'markdown']:
        raise ValueError(f"Unsupported format: {format}. Use 'jsonl', 'json', or 'markdown'")

    # Format based on type
    if format == 'jsonl':
        content = _format_as_jsonl(traces)
    elif format == 'json':
        content = _format_as_json(traces)
    elif format == 'markdown':
        content = _format_as_markdown(traces)

    # Write to file if path provided
    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    return content


def _format_as_jsonl(traces: List[Dict[str, Any]]) -> str:
    """Format traces as JSON Lines (one JSON object per line)."""
    lines = [json.dumps(trace, ensure_ascii=False) for trace in traces]
    return '\n'.join(lines)


def _format_as_json(traces: List[Dict[str, Any]]) -> str:
    """Format traces as pretty-printed JSON array."""
    return json.dumps(traces, indent=2, ensure_ascii=False)


def _format_as_markdown(traces: List[Dict[str, Any]]) -> str:
    """Format traces as human-readable markdown report."""
    lines = ["# LLM Interaction Trace Report", ""]
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Total Interactions:** {len(traces)}")
    lines.append("")

    for i, trace in enumerate(traces, 1):
        lines.append(f"## Interaction {i}: {trace.get('trace_id', 'unknown')}")
        lines.append("")

        # Metadata section
        lines.append("### Metadata")
        lines.append(f"- **Timestamp:** {trace.get('timestamp', 'N/A')}")
        lines.append(f"- **Provider:** {trace.get('provider', 'N/A')}")
        lines.append(f"- **Model:** {trace.get('model', 'N/A')}")

        # Custom metadata
        metadata = trace.get('metadata', {})
        if metadata:
            for key, value in metadata.items():
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        lines.append("")

        # Input section
        lines.append("### Input")

        system_prompt = trace.get('system_prompt')
        if system_prompt:
            lines.append(f"**System Prompt:**")
            lines.append(f"```")
            lines.append(system_prompt)
            lines.append(f"```")
            lines.append("")

        prompt = trace.get('prompt', '')
        lines.append(f"**User Prompt:**")
        lines.append(f"```")
        lines.append(prompt)
        lines.append(f"```")
        lines.append("")

        # Parameters
        parameters = trace.get('parameters', {})
        if parameters:
            lines.append("**Parameters:**")
            for key, value in parameters.items():
                if value is not None:
                    lines.append(f"- `{key}`: {value}")
            lines.append("")

        # Tools
        tools = trace.get('tools')
        if tools:
            lines.append(f"**Tools Available:** {len(tools)} tools")
            lines.append("")

        # Response section
        response = trace.get('response', {})
        lines.append("### Response")

        content = response.get('content', '')
        if content:
            lines.append(f"**Content:**")
            lines.append(f"```")
            lines.append(content)
            lines.append(f"```")
            lines.append("")

        # Tool calls
        tool_calls = response.get('tool_calls')
        if tool_calls:
            lines.append(f"**Tool Calls:** {len(tool_calls)}")
            for call in tool_calls:
                lines.append(f"- `{call.get('name', 'unknown')}({call.get('arguments', {})})`")
            lines.append("")

        # Usage metrics
        usage = response.get('usage', {})
        if usage:
            lines.append("**Usage:**")
            input_tokens = usage.get('input_tokens') or usage.get('prompt_tokens', 0)
            output_tokens = usage.get('output_tokens') or usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', input_tokens + output_tokens)

            lines.append(f"- Input tokens: {input_tokens}")
            lines.append(f"- Output tokens: {output_tokens}")
            lines.append(f"- Total tokens: {total_tokens}")

            visual_tokens = usage.get('visual_tokens')
            if visual_tokens:
                lines.append(f"- Visual tokens: {visual_tokens}")
            lines.append("")

        # Performance
        gen_time = response.get('generation_time_ms')
        finish_reason = response.get('finish_reason')
        if gen_time or finish_reason:
            lines.append("**Performance:**")
            if gen_time:
                lines.append(f"- Generation time: {gen_time:.2f}ms")
            if finish_reason:
                lines.append(f"- Finish reason: {finish_reason}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return '\n'.join(lines)


def summarize_traces(traces: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Generate summary statistics for traces.

    Args:
        traces: Single trace dict or list of trace dicts

    Returns:
        Dictionary with summary statistics:
            - total_interactions: Number of traces
            - total_tokens: Sum of all tokens used
            - total_time_ms: Sum of generation times
            - avg_tokens_per_interaction: Average tokens
            - avg_time_ms: Average generation time
            - providers: Set of providers used
            - models: Set of models used
            - date_range: First and last timestamps

    Example:
        >>> traces = session.get_interaction_history()
        >>> summary = summarize_traces(traces)
        >>> print(f"Total interactions: {summary['total_interactions']}")
        >>> print(f"Total tokens: {summary['total_tokens']}")
        >>> print(f"Average time: {summary['avg_time_ms']:.2f}ms")
    """
    # Normalize to list
    if isinstance(traces, dict):
        traces = [traces]

    if not traces:
        return {
            'total_interactions': 0,
            'total_tokens': 0,
            'total_time_ms': 0,
            'avg_tokens_per_interaction': 0,
            'avg_time_ms': 0,
            'providers': set(),
            'models': set(),
            'date_range': None
        }

    total_tokens = 0
    total_time_ms = 0
    providers = set()
    models = set()
    timestamps = []

    for trace in traces:
        # Extract usage
        response = trace.get('response', {})
        usage = response.get('usage', {})
        if usage:
            total_tokens += usage.get('total_tokens', 0)

        # Extract timing
        gen_time = response.get('generation_time_ms')
        if gen_time:
            total_time_ms += gen_time

        # Extract metadata
        provider = trace.get('provider')
        if provider:
            providers.add(provider)

        model = trace.get('model')
        if model:
            models.add(model)

        timestamp = trace.get('timestamp')
        if timestamp:
            timestamps.append(timestamp)

    num_traces = len(traces)
    avg_tokens = total_tokens / num_traces if num_traces > 0 else 0
    avg_time = total_time_ms / num_traces if num_traces > 0 else 0

    date_range = None
    if timestamps:
        timestamps_sorted = sorted(timestamps)
        date_range = {
            'first': timestamps_sorted[0],
            'last': timestamps_sorted[-1]
        }

    return {
        'total_interactions': num_traces,
        'total_tokens': total_tokens,
        'total_time_ms': total_time_ms,
        'avg_tokens_per_interaction': avg_tokens,
        'avg_time_ms': avg_time,
        'providers': list(providers),
        'models': list(models),
        'date_range': date_range
    }
