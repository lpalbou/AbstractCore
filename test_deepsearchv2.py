"""
Test script for BasicDeepSearchV2

Tests the new deep search implementation with a query about LLMs
and evaluates the quality using LLM-as-judge approach.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abstractcore.processing.basic_deepsearchv2 import BasicDeepSearchV2
from abstractcore.core.factory import create_llm


def test_deepsearch_v2():
    """Test BasicDeepSearchV2 with a known topic"""

    print("=" * 80)
    print("TESTING BASICDEEPSEARCHV2")
    print("=" * 80)

    # Create searcher with default LLM
    print("\n🔧 Initializing searcher...")
    try:
        searcher = BasicDeepSearchV2(verbose=True)
    except Exception as e:
        print(f"❌ Failed to initialize searcher: {e}")
        print("\n💡 Trying with alternative configuration...")
        try:
            llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
            searcher = BasicDeepSearchV2(llm=llm, verbose=True)
        except Exception as e2:
            print(f"❌ Alternative configuration also failed: {e2}")
            return None

    # Test query about LLMs (a subject we can evaluate well)
    test_query = "What are the major developments in large language models in 2024?"

    print(f"\n📋 Test Query: {test_query}")
    print("\n" + "=" * 80)

    # Run research
    try:
        report = searcher.research(
            query=test_query,
            max_iterations=3,
            max_sources=10,
            min_confidence=0.7
        )

        print("\n" + "=" * 80)
        print("📊 RESEARCH REPORT")
        print("=" * 80)

        print(f"\n📝 Executive Summary:")
        print(f"{report.executive_summary}")

        print(f"\n🔑 Key Findings ({len(report.key_findings)}):")
        for idx, finding in enumerate(report.key_findings, 1):
            print(f"{idx}. {finding}")

        print(f"\n📖 Detailed Analysis:")
        print(f"{report.detailed_analysis}")

        print(f"\n📚 Sources ({len(report.sources)}):")
        for idx, source in enumerate(report.sources[:5], 1):  # Show top 5
            print(f"{idx}. {source['title']}")
            print(f"   URL: {source['url']}")
            print(f"   Quality: {source.get('quality_score', 'N/A')}/100")

        if len(report.sources) > 5:
            print(f"   ... and {len(report.sources) - 5} more sources")

        print(f"\n📈 Metadata:")
        print(f"  ├─ Confidence Level: {report.confidence_level:.2f}")
        print(f"  ├─ Total Sources: {report.metadata.get('total_sources', 0)}")
        print(f"  ├─ Avg Quality Score: {report.metadata.get('avg_quality_score', 0):.1f}/100")
        print(f"  ├─ Research Time: {report.metadata.get('elapsed_time', 0):.1f}s")
        print(f"  └─ Intent Type: {report.metadata.get('intent_type', 'unknown')}")

        if report.knowledge_gaps:
            print(f"\n⚠️  Knowledge Gaps:")
            for gap in report.knowledge_gaps:
                print(f"  • {gap}")

        print(f"\n🛤️  Research Path ({len(report.research_path)} steps):")
        for step in report.research_path:
            print(f"  • {step}")

        print("\n" + "=" * 80)

        return report

    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_report(report):
    """Evaluate report quality using LLM-as-judge"""

    if not report:
        print("\n❌ No report to evaluate")
        return

    print("\n" + "=" * 80)
    print("🎯 LLM-AS-JUDGE EVALUATION")
    print("=" * 80)

    # Create judge LLM
    try:
        judge_llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
    except Exception as e:
        print(f"❌ Failed to create judge LLM: {e}")
        return

    # Evaluation prompt
    eval_prompt = f"""You are evaluating the quality of a research report about large language models in 2024.

RESEARCH QUERY: "{report.query}"

EXECUTIVE SUMMARY:
{report.executive_summary}

KEY FINDINGS:
{chr(10).join(f"{i}. {f}" for i, f in enumerate(report.key_findings, 1))}

DETAILED ANALYSIS:
{report.detailed_analysis}

NUMBER OF SOURCES: {len(report.sources)}
CONFIDENCE LEVEL: {report.confidence_level}

Evaluate this report on the following criteria (0-10 scale):

1. COMPLETENESS: Does it address the main aspects of LLM developments in 2024?
2. ACCURACY: Are the claims consistent with known facts about 2024 LLM developments?
3. SOURCE QUALITY: Are the sources authoritative and relevant?
4. CLARITY: Is the information well-organized and easy to understand?
5. CITATIONS: Are claims properly attributed to sources?

Respond with JSON:
{{
    "completeness_score": 0-10,
    "accuracy_score": 0-10,
    "source_quality_score": 0-10,
    "clarity_score": 0-10,
    "citations_score": 0-10,
    "overall_score": 0-10,
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "verdict": "brief overall assessment"
}}"""

    try:
        print("\n🤔 Evaluating report quality...")
        response = judge_llm.generate(eval_prompt, temperature=0.1)

        # Extract response text
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)

        # Parse JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            eval_result = json.loads(response_text[json_start:json_end])

            print(f"\n📊 EVALUATION SCORES:")
            print(f"  ├─ Completeness: {eval_result.get('completeness_score', 'N/A')}/10")
            print(f"  ├─ Accuracy: {eval_result.get('accuracy_score', 'N/A')}/10")
            print(f"  ├─ Source Quality: {eval_result.get('source_quality_score', 'N/A')}/10")
            print(f"  ├─ Clarity: {eval_result.get('clarity_score', 'N/A')}/10")
            print(f"  ├─ Citations: {eval_result.get('citations_score', 'N/A')}/10")
            print(f"  └─ OVERALL: {eval_result.get('overall_score', 'N/A')}/10")

            print(f"\n✅ STRENGTHS:")
            for strength in eval_result.get('strengths', []):
                print(f"  • {strength}")

            print(f"\n⚠️  WEAKNESSES:")
            for weakness in eval_result.get('weaknesses', []):
                print(f"  • {weakness}")

            print(f"\n📝 VERDICT:")
            print(f"{eval_result.get('verdict', 'No verdict provided')}")

            overall = eval_result.get('overall_score', 0)
            if overall >= 8:
                print(f"\n🏆 EXCELLENT PERFORMANCE (>= 8/10)")
            elif overall >= 6:
                print(f"\n✅ GOOD PERFORMANCE (>= 6/10)")
            elif overall >= 4:
                print(f"\n⚠️  NEEDS IMPROVEMENT (>= 4/10)")
            else:
                print(f"\n❌ POOR PERFORMANCE (< 4/10)")

        else:
            print(f"\n⚠️  Could not parse evaluation JSON")
            print(f"Raw response: {response_text}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""

    # Run test
    report = test_deepsearch_v2()

    # Evaluate if successful
    if report:
        evaluate_report(report)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
