#!/usr/bin/env python3
"""
Comprehensive Test Runner for Deep Researcher Strategies

This script runs both strategies (A and B) across multiple models and collects
detailed performance metrics for analysis and improvement.

Usage:
    python run_comprehensive_tests.py [--models MODEL1,MODEL2] [--strategies A,B] [--quick]
"""

import argparse
import sys
import json
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tests.deepresearcher.comprehensive_test_framework import ComprehensiveTestFramework


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run comprehensive deep researcher tests")

    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of model indices to test (e.g., "0,1,2" or "all")',
        default='all'
    )

    parser.add_argument(
        '--strategies',
        type=str,
        help='Comma-separated list of strategies to test ("A", "B", or "all")',
        default='all'
    )

    parser.add_argument(
        '--questions',
        type=str,
        help='Comma-separated list of question IDs to test (or "all")',
        default='all'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test (one question per strategy per model)'
    )

    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Only run baseline (ollama qwen3:4b) for comparison'
    )

    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt (for automated testing)'
    )

    return parser.parse_args()


def main():
    """Main test runner"""
    args = parse_args()

    print("="*100)
    print(" "*30 + "COMPREHENSIVE DEEP RESEARCHER TESTING")
    print("="*100)

    # Initialize framework
    framework = ComprehensiveTestFramework()

    # Load test configuration
    with open('tests/deepresearcher/test_questions.json', 'r') as f:
        config = json.load(f)

    # Determine which models to test
    model_configs = config['model_configurations']

    if args.baseline_only:
        # Only test baseline model
        models_to_test = [model_configs[0]]  # First is always baseline
        print("\n📌 Running baseline-only mode (Ollama qwen3:4b)")
    elif args.models == 'all':
        models_to_test = model_configs
    else:
        indices = [int(i) for i in args.models.split(',')]
        models_to_test = [model_configs[i] for i in indices if i < len(model_configs)]

    # Determine which strategies to test
    if args.strategies == 'all':
        strategies_to_test = ['A', 'B']
    else:
        strategies_to_test = [s.strip().upper() for s in args.strategies.split(',')]

    # Determine which questions to test
    test_questions = config['test_questions']
    if args.quick:
        # Quick mode: only test simple question
        question_ids = ['simple_ml_basics']
        print("\n⚡ Quick mode: testing only simple question")
    elif args.questions == 'all':
        question_ids = None  # Test all
    else:
        question_ids = [qid.strip() for qid in args.questions.split(',')]

    print(f"\n📊 Test Configuration:")
    print(f"   Models: {len(models_to_test)}")
    print(f"   Strategies: {strategies_to_test}")
    print(f"   Questions: {len(question_ids) if question_ids else len(test_questions)}")
    print(f"   Total tests: {len(models_to_test) * len(strategies_to_test) * (len(question_ids) if question_ids else len(test_questions))}")

    if not args.no_confirm:
        input("\n⏸️  Press Enter to start testing (or Ctrl+C to cancel)...")
    else:
        print("\n▶️  Starting tests immediately (no-confirm mode)...")

    # Run tests for each model
    all_sessions = []

    for i, model_config in enumerate(models_to_test, 1):
        print(f"\n{'='*100}")
        print(f"  MODEL {i}/{len(models_to_test)}: {model_config['provider']}/{model_config['model']}")
        print(f"  {model_config['description']}")
        print(f"{'='*100}")

        try:
            session = framework.run_model_test_session(
                model_config,
                strategies=strategies_to_test,
                question_ids=question_ids
            )
            all_sessions.append(session)

        except Exception as e:
            print(f"\n❌ Model {model_config['provider']}/{model_config['model']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print final summary
    print("\n" + "="*100)
    print(" "*35 + "FINAL RESULTS SUMMARY")
    print("="*100)

    framework.print_summary_table()

    # Generate analysis summary
    print("\n📈 Analysis Summary:")
    print("─"*100)

    total_tests = sum(s.total_tests for s in all_sessions)
    total_success = sum(s.successful_tests for s in all_sessions)
    total_failed = sum(s.failed_tests for s in all_sessions)

    print(f"\nOverall Statistics:")
    print(f"  Total tests run: {total_tests}")
    print(f"  Successful: {total_success} ({total_success/total_tests*100:.1f}%)")
    print(f"  Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")

    # Strategy comparison
    strategy_a_results = []
    strategy_b_results = []

    for session in all_sessions:
        for result in session.test_results:
            if result.strategy == 'A':
                strategy_a_results.append(result)
            else:
                strategy_b_results.append(result)

    print(f"\nStrategy A (ReAct + Tree of Thoughts):")
    a_success = [r for r in strategy_a_results if r.success]
    if a_success:
        print(f"  Success rate: {len(a_success)}/{len(strategy_a_results)} ({len(a_success)/len(strategy_a_results)*100:.1f}%)")
        print(f"  Average duration: {sum(r.duration_seconds for r in a_success) / len(a_success):.1f}s")
        print(f"  Average confidence: {sum(r.confidence_score for r in a_success) / len(a_success):.2f}")
    else:
        print(f"  No successful tests")

    print(f"\nStrategy B (Hierarchical Planning):")
    b_success = [r for r in strategy_b_results if r.success]
    if b_success:
        print(f"  Success rate: {len(b_success)}/{len(strategy_b_results)} ({len(b_success)/len(strategy_b_results)*100:.1f}%)")
        print(f"  Average duration: {sum(r.duration_seconds for r in b_success) / len(b_success):.1f}s")
        print(f"  Average confidence: {sum(r.confidence_score for r in b_success) / len(b_success):.2f}")
    else:
        print(f"  No successful tests")

    # Model performance comparison
    print(f"\nModel Performance Ranking:")
    model_stats = {}
    for session in all_sessions:
        model_key = f"{session.model_provider}/{session.model_name}"
        success_rate = session.successful_tests / session.total_tests if session.total_tests > 0 else 0
        model_stats[model_key] = {
            'success_rate': success_rate,
            'avg_confidence': session.average_confidence,
            'avg_duration': session.average_duration
        }

    # Sort by success rate, then confidence
    ranked_models = sorted(
        model_stats.items(),
        key=lambda x: (x[1]['success_rate'], x[1]['avg_confidence']),
        reverse=True
    )

    for i, (model, stats) in enumerate(ranked_models, 1):
        print(f"  {i}. {model}")
        print(f"     Success: {stats['success_rate']*100:.1f}% | Confidence: {stats['avg_confidence']:.2f} | Duration: {stats['avg_duration']:.1f}s")

    print("\n" + "="*100)
    print(f"💾 All results saved to: test_results/")
    print(f"📊 Next step: Run 'python analyze_test_results.py' for detailed analysis")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
