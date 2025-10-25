"""
Test suite to compare BasicDeepResearcherA and BasicDeepResearcherB strategies

This test suite evaluates both research strategies on various types of questions:
1. Technical/Scientific questions
2. Multi-hop comparative questions
3. Current events/trending topics

Evaluation criteria:
- Answer quality and comprehensiveness
- Source diversity and credibility
- Speed and efficiency
- Handling of complex queries
- Confidence scoring accuracy
"""

import pytest
import json
import time
from abstractcore import create_llm
from abstractcore.processing.basic_deepresearcherA import BasicDeepResearcherA
from abstractcore.processing.basic_deepresearcherB import BasicDeepResearcherB


# Test questions covering different scenarios
TEST_QUESTIONS = {
    "technical": {
        "query": "What are the latest advances in quantum error correction?",
        "expected_themes": ["quantum computing", "error correction", "qubits", "fault tolerance"],
        "min_sources": 5,
        "min_confidence": 0.6
    },
    "comparative": {
        "query": "How does the EU AI Act compare to China's AI regulations?",
        "expected_themes": ["EU AI Act", "China regulations", "comparison", "governance"],
        "min_sources": 8,
        "min_confidence": 0.65
    },
    "current_events": {
        "query": "What are the implications of recent developments in large language models?",
        "expected_themes": ["language models", "AI", "implications", "recent"],
        "min_sources": 6,
        "min_confidence": 0.6
    }
}


@pytest.fixture(scope="module")
def llm_instance():
    """Create LLM instance for testing"""
    try:
        # Try to use a good model for testing
        llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", timeout=120)
        return llm
    except Exception as e:
        pytest.skip(f"Could not initialize LLM: {e}")


@pytest.fixture(scope="module")
def researcher_a(llm_instance):
    """Create BasicDeepResearcherA instance"""
    return BasicDeepResearcherA(
        llm=llm_instance,
        max_react_iterations=2,  # Reduced for testing
        max_parallel_paths=2,
        max_sources=10,
        debug=True
    )


@pytest.fixture(scope="module")
def researcher_b(llm_instance):
    """Create BasicDeepResearcherB instance"""
    return BasicDeepResearcherB(
        llm=llm_instance,
        max_sources=10,
        quality_threshold=0.6,
        extract_full_content=False,  # Disabled for speed
        debug=True
    )


class TestStrategyComparison:
    """Compare both research strategies"""

    def test_strategy_a_technical_question(self, researcher_a):
        """Test Strategy A on technical question"""
        test_case = TEST_QUESTIONS["technical"]

        start_time = time.time()
        result = researcher_a.research(test_case["query"], max_depth=1)
        duration = time.time() - start_time

        # Validate result structure
        assert result.title
        assert result.summary
        assert len(result.key_findings) >= 3
        assert len(result.sources_selected) >= test_case["min_sources"]
        assert result.confidence_score >= test_case["min_confidence"]

        # Check metadata
        metadata = result.research_metadata
        assert metadata["strategy"] == "react_tree_of_thoughts"
        assert metadata["duration_seconds"] > 0
        assert metadata["react_iterations"] > 0
        assert metadata["thought_nodes_explored"] > 0

        print(f"\n✅ Strategy A - Technical Question:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Sources: {len(result.sources_selected)}")
        print(f"   Confidence: {result.confidence_score:.2f}")
        print(f"   Key Findings: {len(result.key_findings)}")

    def test_strategy_b_technical_question(self, researcher_b):
        """Test Strategy B on technical question"""
        test_case = TEST_QUESTIONS["technical"]

        start_time = time.time()
        result = researcher_b.research(test_case["query"], depth="medium")
        duration = time.time() - start_time

        # Validate result structure
        assert result.title
        assert result.summary
        assert len(result.key_findings) >= 3
        assert len(result.sources_selected) >= test_case["min_sources"]
        assert result.confidence_score >= test_case["min_confidence"]

        # Check metadata
        metadata = result.research_metadata
        assert metadata["strategy"] == "hierarchical_planning"
        assert metadata["duration_seconds"] > 0
        assert metadata["atomic_questions"] > 0

        print(f"\n✅ Strategy B - Technical Question:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Sources: {len(result.sources_selected)}")
        print(f"   Confidence: {result.confidence_score:.2f}")
        print(f"   Key Findings: {len(result.key_findings)}")

    def test_strategy_a_comparative_question(self, researcher_a):
        """Test Strategy A on comparative question"""
        test_case = TEST_QUESTIONS["comparative"]

        start_time = time.time()
        result = researcher_a.research(test_case["query"], max_depth=1)
        duration = time.time() - start_time

        assert result.title
        assert len(result.sources_selected) >= test_case["min_sources"]

        print(f"\n✅ Strategy A - Comparative Question:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Sources: {len(result.sources_selected)}")
        print(f"   Confidence: {result.confidence_score:.2f}")

    def test_strategy_b_comparative_question(self, researcher_b):
        """Test Strategy B on comparative question"""
        test_case = TEST_QUESTIONS["comparative"]

        start_time = time.time()
        result = researcher_b.research(test_case["query"], depth="medium")
        duration = time.time() - start_time

        assert result.title
        assert len(result.sources_selected) >= test_case["min_sources"]

        print(f"\n✅ Strategy B - Comparative Question:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Sources: {len(result.sources_selected)}")
        print(f"   Confidence: {result.confidence_score:.2f}")


class TestOutputFormat:
    """Test output format compliance"""

    def test_strategy_a_output_format(self, researcher_a):
        """Verify Strategy A output format"""
        result = researcher_a.research("What is machine learning?", max_depth=1)

        # Check required fields
        assert hasattr(result, 'title')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'key_findings')
        assert hasattr(result, 'sources_probed')
        assert hasattr(result, 'sources_selected')
        assert hasattr(result, 'detailed_report')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'research_metadata')

        # Validate JSON serialization
        json_str = json.dumps(result.dict(), indent=2)
        assert json_str
        assert len(json_str) > 100

        # Validate sources format
        for source in result.sources_selected:
            assert 'url' in source
            assert 'title' in source

    def test_strategy_b_output_format(self, researcher_b):
        """Verify Strategy B output format"""
        result = researcher_b.research("What is machine learning?", depth="shallow")

        # Check required fields
        assert hasattr(result, 'title')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'key_findings')
        assert hasattr(result, 'sources_probed')
        assert hasattr(result, 'sources_selected')
        assert hasattr(result, 'detailed_report')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'research_metadata')

        # Validate JSON serialization
        json_str = json.dumps(result.dict(), indent=2)
        assert json_str
        assert len(json_str) > 100


class TestPerformanceMetrics:
    """Compare performance metrics"""

    def test_strategy_comparison_simple_query(self, researcher_a, researcher_b):
        """Compare both strategies on a simple query"""
        query = "What is deep learning?"

        # Strategy A
        start_a = time.time()
        result_a = researcher_a.research(query, max_depth=1)
        duration_a = time.time() - start_a

        # Strategy B
        start_b = time.time()
        result_b = researcher_b.research(query, depth="shallow")
        duration_b = time.time() - start_b

        # Print comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"\n📊 Strategy A (ReAct + Tree of Thoughts):")
        print(f"   Duration: {duration_a:.1f}s")
        print(f"   Sources probed: {len(result_a.sources_probed)}")
        print(f"   Sources selected: {len(result_a.sources_selected)}")
        print(f"   Key findings: {len(result_a.key_findings)}")
        print(f"   Confidence: {result_a.confidence_score:.2f}")
        print(f"   ReAct iterations: {result_a.research_metadata['react_iterations']}")
        print(f"   Thought nodes: {result_a.research_metadata['thought_nodes_explored']}")

        print(f"\n📊 Strategy B (Hierarchical Planning):")
        print(f"   Duration: {duration_b:.1f}s")
        print(f"   Sources probed: {len(result_b.sources_probed)}")
        print(f"   Sources selected: {len(result_b.sources_selected)}")
        print(f"   Key findings: {len(result_b.key_findings)}")
        print(f"   Confidence: {result_b.confidence_score:.2f}")
        print(f"   Atomic questions: {result_b.research_metadata['atomic_questions']}")
        print(f"   Knowledge nodes: {result_b.research_metadata['knowledge_nodes']}")

        print(f"\n🏆 Comparison:")
        print(f"   Speed winner: {'A' if duration_a < duration_b else 'B'} ({abs(duration_a - duration_b):.1f}s faster)")
        print(f"   More sources: {'A' if len(result_a.sources_selected) > len(result_b.sources_selected) else 'B'}")
        print(f"   Higher confidence: {'A' if result_a.confidence_score > result_b.confidence_score else 'B'}")

        # Both should produce valid results
        assert result_a.confidence_score > 0
        assert result_b.confidence_score > 0


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation comparing both strategies

    This function is meant to be run manually for detailed comparison
    """
    print("="*80)
    print("COMPREHENSIVE DEEP RESEARCHER EVALUATION")
    print("="*80)

    # Initialize
    try:
        llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", timeout=120)
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return

    researcher_a = BasicDeepResearcherA(llm=llm, max_sources=15, max_react_iterations=2)
    researcher_b = BasicDeepResearcherB(llm=llm, max_sources=15, extract_full_content=False)

    evaluation_results = []

    for test_name, test_case in TEST_QUESTIONS.items():
        print(f"\n{'='*80}")
        print(f"TEST: {test_name.upper()}")
        print(f"Query: {test_case['query']}")
        print(f"{'='*80}")

        # Test Strategy A
        print("\n🔬 Running Strategy A (ReAct + Tree of Thoughts)...")
        start = time.time()
        try:
            result_a = researcher_a.research(test_case['query'], max_depth=1)
            duration_a = time.time() - start

            score_a = {
                'duration': duration_a,
                'sources_selected': len(result_a.sources_selected),
                'key_findings': len(result_a.key_findings),
                'confidence': result_a.confidence_score,
                'success': True
            }
        except Exception as e:
            print(f"❌ Strategy A failed: {e}")
            score_a = {'success': False, 'error': str(e)}

        # Test Strategy B
        print("\n🔬 Running Strategy B (Hierarchical Planning)...")
        start = time.time()
        try:
            result_b = researcher_b.research(test_case['query'], depth="medium")
            duration_b = time.time() - start

            score_b = {
                'duration': duration_b,
                'sources_selected': len(result_b.sources_selected),
                'key_findings': len(result_b.key_findings),
                'confidence': result_b.confidence_score,
                'success': True
            }
        except Exception as e:
            print(f"❌ Strategy B failed: {e}")
            score_b = {'success': False, 'error': str(e)}

        # Compare
        evaluation_results.append({
            'test': test_name,
            'query': test_case['query'],
            'strategy_a': score_a,
            'strategy_b': score_b
        })

        if score_a['success'] and score_b['success']:
            print(f"\n📊 Comparison:")
            print(f"   Duration: A={duration_a:.1f}s vs B={duration_b:.1f}s")
            print(f"   Sources: A={score_a['sources_selected']} vs B={score_b['sources_selected']}")
            print(f"   Findings: A={score_a['key_findings']} vs B={score_b['key_findings']}")
            print(f"   Confidence: A={score_a['confidence']:.2f} vs B={score_b['confidence']:.2f}")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*80}")

    a_wins = sum(1 for r in evaluation_results if r['strategy_a'].get('success') and
                 (not r['strategy_b'].get('success') or
                  r['strategy_a']['confidence'] > r['strategy_b']['confidence']))
    b_wins = sum(1 for r in evaluation_results if r['strategy_b'].get('success') and
                 (not r['strategy_a'].get('success') or
                  r['strategy_b']['confidence'] > r['strategy_a']['confidence']))

    print(f"\n🏆 Overall Winner: {'Strategy A' if a_wins > b_wins else 'Strategy B' if b_wins > a_wins else 'TIE'}")
    print(f"   Strategy A wins: {a_wins}/{len(evaluation_results)}")
    print(f"   Strategy B wins: {b_wins}/{len(evaluation_results)}")

    return evaluation_results


if __name__ == "__main__":
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation()

    # Save results
    if results:
        with open("strategy_comparison_results.json", "w") as f:
            json.dumps(results, f, indent=2)
        print(f"\n💾 Results saved to strategy_comparison_results.json")
