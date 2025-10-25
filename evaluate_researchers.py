#!/usr/bin/env python3
"""
Evaluation script to compare BasicDeepResearcherA and BasicDeepResearcherB

This script runs both strategies on a test question and provides detailed
comparison to determine which approach works best.

Usage:
    python evaluate_researchers.py
"""

import json
import time
from abstractcore import create_llm
from abstractcore.processing.basic_deepresearcherA import BasicDeepResearcherA
from abstractcore.processing.basic_deepresearcherB import BasicDeepResearcherB


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def evaluate_strategy(researcher, strategy_name: str, query: str):
    """Evaluate a single strategy"""
    print(f"\n🔬 Testing {strategy_name}...")
    print(f"   Query: {query}")

    start_time = time.time()
    try:
        if strategy_name == "Strategy A":
            result = researcher.research(query, max_depth=1)
        else:
            result = researcher.research(query, depth="medium")

        duration = time.time() - start_time

        # Collect metrics
        metrics = {
            "success": True,
            "duration": duration,
            "sources_probed": len(result.sources_probed),
            "sources_selected": len(result.sources_selected),
            "key_findings": len(result.key_findings),
            "confidence": result.confidence_score,
            "metadata": result.research_metadata,
            "title": result.title,
            "summary": result.summary[:200] + "..." if len(result.summary) > 200 else result.summary
        }

        # Print results
        print(f"\n✅ {strategy_name} completed in {duration:.1f}s")
        print(f"\n📝 Title: {metrics['title']}")
        print(f"\n📄 Summary:\n   {metrics['summary']}")
        print(f"\n📊 Metrics:")
        print(f"   • Duration: {duration:.1f}s")
        print(f"   • Sources probed: {metrics['sources_probed']}")
        print(f"   • Sources selected: {metrics['sources_selected']}")
        print(f"   • Key findings: {metrics['key_findings']}")
        print(f"   • Confidence score: {metrics['confidence']:.2f}")

        if strategy_name == "Strategy A":
            print(f"   • ReAct iterations: {metrics['metadata']['react_iterations']}")
            print(f"   • Thought nodes explored: {metrics['metadata']['thought_nodes_explored']}")
        else:
            print(f"   • Atomic questions: {metrics['metadata']['atomic_questions']}")
            print(f"   • Knowledge nodes: {metrics['metadata']['knowledge_nodes']}")

        print(f"\n🔑 Key Findings:")
        for i, finding in enumerate(result.key_findings[:5], 1):
            print(f"   {i}. {finding}")

        return metrics

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ {strategy_name} failed after {duration:.1f}s")
        print(f"   Error: {str(e)}")
        return {
            "success": False,
            "duration": duration,
            "error": str(e)
        }


def compare_results(metrics_a: dict, metrics_b: dict):
    """Compare results from both strategies"""
    print_section("COMPARATIVE ANALYSIS")

    if not metrics_a["success"] or not metrics_b["success"]:
        print("\n⚠️  Cannot compare - one or both strategies failed")
        if metrics_a["success"]:
            print("   Winner by default: Strategy A (ReAct + Tree of Thoughts)")
            return "A"
        elif metrics_b["success"]:
            print("   Winner by default: Strategy B (Hierarchical Planning)")
            return "B"
        else:
            print("   Both strategies failed")
            return None

    # Calculate scores
    scores_a = 0
    scores_b = 0

    print("\n📊 Head-to-Head Comparison:\n")

    # Speed
    print(f"⏱️  Speed:")
    print(f"   Strategy A: {metrics_a['duration']:.1f}s")
    print(f"   Strategy B: {metrics_b['duration']:.1f}s")
    if metrics_a['duration'] < metrics_b['duration']:
        print(f"   Winner: Strategy A ({metrics_b['duration'] - metrics_a['duration']:.1f}s faster)")
        scores_a += 1
    else:
        print(f"   Winner: Strategy B ({metrics_a['duration'] - metrics_b['duration']:.1f}s faster)")
        scores_b += 1

    # Source quality (selected / probed ratio)
    ratio_a = metrics_a['sources_selected'] / max(metrics_a['sources_probed'], 1)
    ratio_b = metrics_b['sources_selected'] / max(metrics_b['sources_probed'], 1)

    print(f"\n🎯 Source Selection Quality (selected/probed):")
    print(f"   Strategy A: {metrics_a['sources_selected']}/{metrics_a['sources_probed']} = {ratio_a:.2%}")
    print(f"   Strategy B: {metrics_b['sources_selected']}/{metrics_b['sources_probed']} = {ratio_b:.2%}")
    if ratio_a > ratio_b:
        print(f"   Winner: Strategy A (more selective)")
        scores_a += 1
    else:
        print(f"   Winner: Strategy B (more selective)")
        scores_b += 1

    # Number of findings
    print(f"\n📋 Number of Key Findings:")
    print(f"   Strategy A: {metrics_a['key_findings']}")
    print(f"   Strategy B: {metrics_b['key_findings']}")
    if metrics_a['key_findings'] > metrics_b['key_findings']:
        print(f"   Winner: Strategy A (more findings)")
        scores_a += 1
    elif metrics_b['key_findings'] > metrics_a['key_findings']:
        print(f"   Winner: Strategy B (more findings)")
        scores_b += 1
    else:
        print(f"   Tie")

    # Confidence
    print(f"\n🎓 Confidence Score:")
    print(f"   Strategy A: {metrics_a['confidence']:.2f}")
    print(f"   Strategy B: {metrics_b['confidence']:.2f}")
    if metrics_a['confidence'] > metrics_b['confidence']:
        print(f"   Winner: Strategy A")
        scores_a += 1
    else:
        print(f"   Winner: Strategy B")
        scores_b += 1

    # Overall winner
    print(f"\n{'='*80}")
    print(f"🏆 OVERALL WINNER: ", end="")
    if scores_a > scores_b:
        print(f"Strategy A (ReAct + Tree of Thoughts)")
        print(f"   Score: {scores_a} vs {scores_b}")
        winner = "A"
    elif scores_b > scores_a:
        print(f"Strategy B (Hierarchical Planning)")
        print(f"   Score: {scores_b} vs {scores_a}")
        winner = "B"
    else:
        print(f"TIE")
        print(f"   Score: {scores_a} vs {scores_b}")
        winner = "TIE"

    # Recommendations
    print(f"\n💡 Recommendations:")
    print(f"\n   Strategy A (ReAct + Tree of Thoughts) is best for:")
    print(f"   • Exploratory research requiring multiple angles")
    print(f"   • Questions benefiting from parallel exploration")
    print(f"   • Time-sensitive queries (potentially faster)")
    print(f"   • Breadth-first coverage")

    print(f"\n   Strategy B (Hierarchical Planning) is best for:")
    print(f"   • Structured, methodical research")
    print(f"   • Questions requiring deep analysis")
    print(f"   • High-quality source prioritization")
    print(f"   • Depth-first investigation with dependency tracking")

    return winner


def main():
    """Main evaluation function"""
    print_section("DEEP RESEARCHER STRATEGY EVALUATION")

    # Test question
    test_query = "What are the latest advances in quantum error correction?"

    print(f"\n📝 Test Query: {test_query}")
    print(f"\nThis query tests:")
    print(f"  • Technical/scientific understanding")
    print(f"  • Ability to find recent information")
    print(f"  • Source quality assessment")
    print(f"  • Synthesis of complex information")

    # Initialize LLM
    print(f"\n🤖 Initializing LLM...")
    try:
        llm = create_llm(
            "ollama",
            model="qwen3:4b-instruct-2507-q4_K_M",
            timeout=120
        )
        print(f"   ✅ LLM initialized: ollama/qwen3:4b-instruct-2507-q4_K_M")
    except Exception as e:
        print(f"   ❌ Failed to initialize LLM: {e}")
        print(f"\n💡 Please ensure Ollama is running with the model:")
        print(f"   ollama pull qwen3:4b-instruct-2507-q4_K_M")
        return

    # Initialize researchers
    print(f"\n🔧 Initializing researchers...")
    researcher_a = BasicDeepResearcherA(
        llm=llm,
        max_react_iterations=2,
        max_parallel_paths=2,
        max_sources=15
    )
    researcher_b = BasicDeepResearcherB(
        llm=llm,
        max_sources=15,
        quality_threshold=0.65,
        extract_full_content=False  # Disabled for faster testing
    )
    print(f"   ✅ Both researchers initialized")

    # Evaluate Strategy A
    print_section("STRATEGY A: ReAct + Tree of Thoughts")
    metrics_a = evaluate_strategy(researcher_a, "Strategy A", test_query)

    # Evaluate Strategy B
    print_section("STRATEGY B: Hierarchical Planning")
    metrics_b = evaluate_strategy(researcher_b, "Strategy B", test_query)

    # Compare
    winner = compare_results(metrics_a, metrics_b)

    # Save results
    results = {
        "test_query": test_query,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy_a": metrics_a,
        "strategy_b": metrics_b,
        "winner": winner
    }

    output_file = "researcher_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"💾 Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
