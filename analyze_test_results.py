#!/usr/bin/env python3
"""
Test Results Analysis Tool

Analyzes comprehensive test results to identify patterns, bottlenecks,
and opportunities for improvement.

Usage:
    python analyze_test_results.py [--output report.md]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
from datetime import datetime


def load_all_test_results(results_dir: str = "test_results") -> List[Dict]:
    """Load all test result files"""
    results_path = Path(results_dir)
    all_results = []

    for filepath in results_path.glob("session_*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
            all_results.append(data)

    return all_results


def analyze_strategy_performance(sessions: List[Dict]) -> Dict[str, Any]:
    """Analyze performance by strategy"""
    strategy_stats = defaultdict(lambda: {
        'total_tests': 0,
        'successful_tests': 0,
        'failed_tests': 0,
        'durations': [],
        'confidences': [],
        'sources_selected': [],
        'key_findings': [],
        'errors': []
    })

    for session in sessions:
        for result in session['test_results']:
            strategy = result['strategy']
            stats = strategy_stats[strategy]

            stats['total_tests'] += 1

            if result['success']:
                stats['successful_tests'] += 1
                stats['durations'].append(result['duration_seconds'])
                stats['confidences'].append(result['confidence_score'])
                stats['sources_selected'].append(result['sources_selected'])
                stats['key_findings'].append(result['key_findings_count'])
            else:
                stats['failed_tests'] += 1
                stats['errors'].append({
                    'error': result.get('error_message', 'Unknown'),
                    'model': f"{result['model_provider']}/{result['model_name']}",
                    'query': result['query_id']
                })

    # Calculate averages
    for strategy, stats in strategy_stats.items():
        if stats['durations']:
            stats['avg_duration'] = sum(stats['durations']) / len(stats['durations'])
            stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
            stats['avg_sources'] = sum(stats['sources_selected']) / len(stats['sources_selected'])
            stats['avg_findings'] = sum(stats['key_findings']) / len(stats['key_findings'])
        else:
            stats['avg_duration'] = 0
            stats['avg_confidence'] = 0
            stats['avg_sources'] = 0
            stats['avg_findings'] = 0

        stats['success_rate'] = stats['successful_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0

    return dict(strategy_stats)


def analyze_model_performance(sessions: List[Dict]) -> Dict[str, Any]:
    """Analyze performance by model"""
    model_stats = defaultdict(lambda: {
        'total_tests': 0,
        'successful_tests': 0,
        'strategy_performance': defaultdict(lambda: {'success': 0, 'total': 0}),
        'query_categories': defaultdict(lambda: {'success': 0, 'total': 0}),
        'errors': []
    })

    for session in sessions:
        model_key = f"{session['model_provider']}/{session['model_name']}"

        for result in session['test_results']:
            stats = model_stats[model_key]
            stats['total_tests'] += 1

            if result['success']:
                stats['successful_tests'] += 1
                stats['strategy_performance'][result['strategy']]['success'] += 1
                stats['query_categories'][result['query_category']]['success'] += 1
            else:
                stats['errors'].append({
                    'error': result.get('error_message', 'Unknown'),
                    'strategy': result['strategy'],
                    'query': result['query_id']
                })

            stats['strategy_performance'][result['strategy']]['total'] += 1
            stats['query_categories'][result['query_category']]['total'] += 1

    # Calculate success rates
    for model, stats in model_stats.items():
        stats['success_rate'] = stats['successful_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0

        # Strategy-specific success rates
        for strategy, perf in stats['strategy_performance'].items():
            perf['success_rate'] = perf['success'] / perf['total'] if perf['total'] > 0 else 0

        # Category-specific success rates
        for category, perf in stats['query_categories'].items():
            perf['success_rate'] = perf['success'] / perf['total'] if perf['total'] > 0 else 0

    return dict(model_stats)


def analyze_error_patterns(sessions: List[Dict]) -> Dict[str, Any]:
    """Analyze common error patterns"""
    error_patterns = defaultdict(lambda: {
        'count': 0,
        'affected_models': set(),
        'affected_strategies': set(),
        'sample_traces': []
    })

    for session in sessions:
        for result in session['test_results']:
            if not result['success'] and result.get('error_message'):
                error_msg = result['error_message']

                # Categorize error
                if 'validation' in error_msg.lower() or 'field required' in error_msg.lower():
                    category = 'Structured Output Validation Error'
                elif 'timeout' in error_msg.lower():
                    category = 'Timeout Error'
                elif 'connection' in error_msg.lower():
                    category = 'Connection Error'
                else:
                    category = 'Other Error'

                pattern = error_patterns[category]
                pattern['count'] += 1
                pattern['affected_models'].add(f"{result['model_provider']}/{result['model_name']}")
                pattern['affected_strategies'].add(result['strategy'])

                if len(pattern['sample_traces']) < 3:
                    pattern['sample_traces'].append({
                        'model': f"{result['model_provider']}/{result['model_name']}",
                        'strategy': result['strategy'],
                        'query': result['query_id'],
                        'error': error_msg
                    })

    # Convert sets to lists for JSON serialization
    for pattern in error_patterns.values():
        pattern['affected_models'] = list(pattern['affected_models'])
        pattern['affected_strategies'] = list(pattern['affected_strategies'])

    return dict(error_patterns)


def identify_improvement_opportunities(
    strategy_stats: Dict,
    model_stats: Dict,
    error_patterns: Dict
) -> List[str]:
    """Identify specific improvement opportunities"""
    opportunities = []

    # Check Strategy B validation issues
    if 'B' in strategy_stats:
        b_stats = strategy_stats['B']
        if b_stats['success_rate'] < 0.5:
            opportunities.append(
                "**Critical**: Strategy B has low success rate. Consider simplifying structured output models."
            )

    # Check for timeout issues
    if 'Timeout Error' in error_patterns:
        opportunities.append(
            f"**Performance**: {error_patterns['Timeout Error']['count']} timeout errors detected. "
            "Consider optimizing search queries or increasing timeouts."
        )

    # Check validation errors
    if 'Structured Output Validation Error' in error_patterns:
        count = error_patterns['Structured Output Validation Error']['count']
        strategies = error_patterns['Structured Output Validation Error']['affected_strategies']
        opportunities.append(
            f"**Robustness**: {count} structured output validation errors in {strategies}. "
            "Implement fallback parsing and simpler Pydantic models."
        )

    # Check model-specific issues
    for model, stats in model_stats.items():
        if stats['success_rate'] < 0.5:
            opportunities.append(
                f"**Model-Specific**: {model} has low success rate ({stats['success_rate']:.1%}). "
                "May need model-specific prompt optimization."
            )

    return opportunities


def generate_markdown_report(
    strategy_stats: Dict,
    model_stats: Dict,
    error_patterns: Dict,
    opportunities: List[str],
    output_path: str = "analysis_report.md"
):
    """Generate comprehensive markdown report"""
    report = []

    report.append("# Deep Researcher Test Results Analysis")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")

    # Executive Summary
    report.append("## Executive Summary\n")

    total_tests = sum(s['total_tests'] for s in strategy_stats.values())
    total_success = sum(s['successful_tests'] for s in strategy_stats.values())

    report.append(f"- **Total Tests**: {total_tests}")
    report.append(f"- **Overall Success Rate**: {total_success/total_tests*100:.1f}%")
    report.append(f"- **Strategies Tested**: {', '.join(strategy_stats.keys())}")
    report.append(f"- **Models Tested**: {len(model_stats)}\n")

    # Strategy Performance
    report.append("## Strategy Performance Comparison\n")

    for strategy, stats in sorted(strategy_stats.items()):
        report.append(f"### Strategy {strategy}\n")
        report.append(f"- **Success Rate**: {stats['success_rate']*100:.1f}% ({stats['successful_tests']}/{stats['total_tests']})")
        if stats['avg_duration'] > 0:
            report.append(f"- **Average Duration**: {stats['avg_duration']:.1f}s")
            report.append(f"- **Average Confidence**: {stats['avg_confidence']:.2f}")
            report.append(f"- **Average Sources Selected**: {stats['avg_sources']:.1f}")
            report.append(f"- **Average Key Findings**: {stats['avg_findings']:.1f}")

        if stats['errors']:
            report.append(f"- **Failures**: {stats['failed_tests']}")
            report.append("\n**Common Errors**:")
            error_summary = defaultdict(int)
            for error in stats['errors'][:5]:  # Top 5 errors
                error_summary[error['error'][:50]] += 1
            for error_type, count in error_summary.items():
                report.append(f"  - {error_type}... ({count}x)")

        report.append("")

    # Model Performance
    report.append("## Model Performance Analysis\n")

    # Sort models by success rate
    sorted_models = sorted(
        model_stats.items(),
        key=lambda x: x[1]['success_rate'],
        reverse=True
    )

    for model, stats in sorted_models:
        report.append(f"### {model}\n")
        report.append(f"- **Overall Success Rate**: {stats['success_rate']*100:.1f}%")

        # Strategy-specific performance
        report.append("- **Strategy Performance**:")
        for strategy, perf in stats['strategy_performance'].items():
            report.append(f"  - Strategy {strategy}: {perf['success_rate']*100:.1f}% ({perf['success']}/{perf['total']})")

        # Category performance
        if stats['query_categories']:
            report.append("- **Query Category Performance**:")
            for category, perf in stats['query_categories'].items():
                report.append(f"  - {category}: {perf['success_rate']*100:.1f}%")

        report.append("")

    # Error Analysis
    report.append("## Error Pattern Analysis\n")

    if error_patterns:
        for error_type, pattern in sorted(error_patterns.items(), key=lambda x: x[1]['count'], reverse=True):
            report.append(f"### {error_type}\n")
            report.append(f"- **Occurrences**: {pattern['count']}")
            report.append(f"- **Affected Models**: {', '.join(pattern['affected_models'])}")
            report.append(f"- **Affected Strategies**: {', '.join(pattern['affected_strategies'])}")

            if pattern['sample_traces']:
                report.append("\n**Sample Cases**:")
                for sample in pattern['sample_traces']:
                    report.append(f"- Model: {sample['model']}, Strategy: {sample['strategy']}, Query: {sample['query']}")
                    report.append(f"  Error: `{sample['error'][:100]}...`")

            report.append("")
    else:
        report.append("No errors detected.\n")

    # Improvement Opportunities
    report.append("## Improvement Opportunities\n")

    if opportunities:
        for i, opportunity in enumerate(opportunities, 1):
            report.append(f"{i}. {opportunity}\n")
    else:
        report.append("No specific improvement opportunities identified.\n")

    # Recommendations
    report.append("## Strategic Recommendations\n")

    # Determine best strategy
    best_strategy = max(strategy_stats.items(), key=lambda x: x[1]['success_rate'])
    report.append(f"### Primary Recommendation\n")
    report.append(f"**Use Strategy {best_strategy[0]}** as the primary deep research implementation.")
    report.append(f"- Success rate: {best_strategy[1]['success_rate']*100:.1f}%")
    if best_strategy[1]['avg_confidence'] > 0:
        report.append(f"- Average confidence: {best_strategy[1]['avg_confidence']:.2f}")
        report.append(f"- Average duration: {best_strategy[1]['avg_duration']:.1f}s\n")

    # Model recommendations
    if sorted_models:
        best_model = sorted_models[0]
        report.append(f"### Best Performing Model\n")
        report.append(f"**{best_model[0]}** with {best_model[1]['success_rate']*100:.1f}% success rate\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"📊 Analysis report generated: {output_path}")


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="Analyze deep researcher test results")
    parser.add_argument('--output', default='analysis_report.md', help='Output markdown file')
    parser.add_argument('--results-dir', default='test_results', help='Test results directory')
    args = parser.parse_args()

    print("="*80)
    print(" "*25 + "TEST RESULTS ANALYSIS")
    print("="*80)

    # Load results
    print(f"\n📂 Loading results from: {args.results_dir}")
    sessions = load_all_test_results(args.results_dir)

    if not sessions:
        print("❌ No test results found!")
        return

    print(f"✅ Loaded {len(sessions)} test sessions")

    # Analyze
    print("\n🔍 Analyzing strategy performance...")
    strategy_stats = analyze_strategy_performance(sessions)

    print("🔍 Analyzing model performance...")
    model_stats = analyze_model_performance(sessions)

    print("🔍 Analyzing error patterns...")
    error_patterns = analyze_error_patterns(sessions)

    print("💡 Identifying improvement opportunities...")
    opportunities = identify_improvement_opportunities(
        strategy_stats,
        model_stats,
        error_patterns
    )

    # Print quick summary
    print("\n" + "="*80)
    print(" "*30 + "QUICK SUMMARY")
    print("="*80)

    for strategy, stats in strategy_stats.items():
        print(f"\n Strategy {strategy}:")
        print(f"   Success Rate: {stats['success_rate']*100:.1f}%")
        if stats['avg_duration'] > 0:
            print(f"   Avg Duration: {stats['avg_duration']:.1f}s")
            print(f"   Avg Confidence: {stats['avg_confidence']:.2f}")

    if opportunities:
        print(f"\n💡 Top Improvement Opportunities:")
        for i, opp in enumerate(opportunities[:3], 1):
            # Extract just the bold part
            main_point = opp.split(':')[0].replace('*', '')
            print(f"   {i}. {main_point}")

    # Generate report
    print(f"\n📝 Generating detailed report...")
    generate_markdown_report(
        strategy_stats,
        model_stats,
        error_patterns,
        opportunities,
        args.output
    )

    print("\n" + "="*80)
    print(f"✅ Analysis complete! Report saved to: {args.output}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
