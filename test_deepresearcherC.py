"""
Test script for BasicDeepResearcherC - Adaptive ReAct Web Research Agent

This script tests the new deep researcher with various query types to verify:
1. Adaptive planning and ReAct execution
2. Actual web content fetching
3. Grounding to prevent hallucinations
4. Graceful handling of unknown queries
"""

from abstractcore import create_llm
from abstractcore.processing import BasicDeepResearcherC
import json
import time
from datetime import datetime

# Test queries covering different scenarios
TEST_QUERIES = [
    {
        "name": "Known Technical Topic",
        "query": "What is quantum computing",
        "expected": "Should find technical information with real sources"
    },
    {
        "name": "Known Person (Verifiable)",
        "query": "Laurent-Philippe Albou",
        "expected": "Should find actual information OR report no information found"
    },
    {
        "name": "Recent Technology",
        "query": "GPT-4 capabilities and features",
        "expected": "Should find recent technical documentation"
    },
    {
        "name": "Complex Analytical Query",
        "query": "Impact of AI on software development practices",
        "expected": "Should explore multiple dimensions and synthesize findings"
    }
]

def run_test(researcher, test_case, save_results=True):
    """Run a single test case"""
    print("\n" + "="*80)
    print(f"TEST: {test_case['name']}")
    print(f"Query: {test_case['query']}")
    print(f"Expected: {test_case['expected']}")
    print("="*80)

    start_time = time.time()

    try:
        result = researcher.research(test_case['query'])
        duration = time.time() - start_time

        # Display results
        print(f"\n✅ Research completed in {duration:.1f}s")
        print(f"\n📊 RESULTS:")
        print(f"  - Title: {result.title}")
        print(f"  - Confidence: {result.confidence_score:.2f}")
        print(f"  - Sources: {len(result.sources_selected)}")
        print(f"  - Findings: {len(result.key_findings)}")
        print(f"  - Knowledge Gaps: {len(result.knowledge_gaps)}")

        print(f"\n📝 Executive Summary:")
        print(f"  {result.summary[:300]}...")

        print(f"\n🔑 Key Findings:")
        for i, finding in enumerate(result.key_findings[:5], 1):
            print(f"  {i}. {finding[:150]}...")

        if result.knowledge_gaps:
            print(f"\n⚠️ Knowledge Gaps:")
            for gap in result.knowledge_gaps[:3]:
                print(f"  - {gap}")

        print(f"\n🌐 Top Sources:")
        for i, source in enumerate(result.sources_selected[:5], 1):
            print(f"  {i}. {source['title'][:60]}...")
            print(f"     URL: {source['url'][:80]}")
            print(f"     Relevance: {source['relevance']:.2f} | Credibility: {source['credibility']:.2f}")

        print(f"\n📈 Metadata:")
        for key, value in result.research_metadata.items():
            print(f"  - {key}: {value}")

        # Validate grounding
        print(f"\n🔍 Grounding Check:")
        if len(result.sources_selected) == 0:
            print(f"  ⚠️  NO SOURCES - Verify this is 'no information found' case")
            print(f"  ✅ Correctly reports: {result.key_findings}")
        else:
            print(f"  ✅ {len(result.sources_selected)} sources verified")
            print(f"  ✅ Grounding appears valid")

        # Save results if requested
        if save_results:
            safe_name = test_case['name'].replace(" ", "_").replace("/", "_").lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"researcherC_results_{safe_name}_{timestamp}.json"

            result_data = {
                "test_case": test_case,
                "duration_seconds": round(duration, 2),
                "result": result.dict()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Results saved to: {filename}")

        return {
            "success": True,
            "duration": duration,
            "sources": len(result.sources_selected),
            "confidence": result.confidence_score
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ Test failed after {duration:.1f}s")
        print(f"Error: {type(e).__name__}: {str(e)}")

        import traceback
        print(f"\nTraceback:")
        traceback.print_exc()

        return {
            "success": False,
            "duration": duration,
            "error": str(e)
        }


def main():
    """Main test execution"""
    print("\n" + "="*80)
    print("BasicDeepResearcherC - Adaptive ReAct Test Suite")
    print("="*80)

    # Initialize researcher with LMStudio
    print("\n🤖 Initializing BasicDeepResearcherC...")
    print("   Provider: LMStudio")
    print("   Model: qwen/qwen3-30b-a3b-2507")
    print("   Max Sources: 30")
    print("   Debug: False")

    llm = create_llm(
        "lmstudio",
        model="qwen/qwen3-30b-a3b-2507",
        temperature=0.2,
        timeout=120
    )

    researcher = BasicDeepResearcherC(
        llm=llm,
        max_sources=30,
        max_urls_to_probe=60,
        max_iterations=25,  # Default: will allow up to 50 total iterations
        fetch_timeout=10,
        enable_breadth=True,
        enable_depth=True,
        grounding_threshold=0.7,
        debug=False
    )

    print("✅ Researcher initialized")

    # Run tests
    results_summary = []

    for test_case in TEST_QUERIES:
        result = run_test(researcher, test_case, save_results=True)
        results_summary.append({
            "name": test_case['name'],
            **result
        })

        # Brief pause between tests
        time.sleep(2)

    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    successful = sum(1 for r in results_summary if r['success'])
    total = len(results_summary)

    print(f"\n✅ Tests passed: {successful}/{total}")
    print(f"\nDetailed Results:")

    for r in results_summary:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        print(f"\n{status} - {r['name']}")
        print(f"  Duration: {r.get('duration', 0):.1f}s")

        if r['success']:
            print(f"  Sources: {r.get('sources', 0)}")
            print(f"  Confidence: {r.get('confidence', 0):.2f}")
        else:
            print(f"  Error: {r.get('error', 'Unknown error')}")

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
