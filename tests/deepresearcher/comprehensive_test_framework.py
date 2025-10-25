"""
Comprehensive Test Framework for Deep Researcher Strategies

This framework provides:
1. Multi-model testing support
2. Structured result storage
3. Performance metrics collection
4. Error tracking and analysis
5. Comparative analysis tools
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import traceback

from abstractcore import create_llm
from abstractcore.processing.basic_deepresearcherA import BasicDeepResearcherA
from abstractcore.processing.basic_deepresearcherB import BasicDeepResearcherB


# ==================== Data Models ====================

@dataclass
class TestResult:
    """Single test result"""
    test_id: str
    timestamp: str
    model_provider: str
    model_name: str
    strategy: str
    query_id: str
    query: str
    query_category: str
    query_difficulty: str

    # Results
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None
    error_trace: Optional[str] = None

    # Metrics
    sources_probed: int = 0
    sources_selected: int = 0
    key_findings_count: int = 0
    confidence_score: float = 0.0

    # Quality metrics
    title: Optional[str] = None
    summary_length: int = 0
    has_structured_report: bool = False

    # Strategy-specific metrics
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)

    # Resource usage
    estimated_tokens_used: int = 0

    # Output sample
    output_sample: Optional[Dict[str, Any]] = None


@dataclass
class ModelTestSession:
    """Test session for a specific model"""
    session_id: str
    timestamp: str
    model_provider: str
    model_name: str
    model_description: str
    test_results: List[TestResult] = field(default_factory=list)

    # Session-level statistics
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    average_duration: float = 0.0
    average_confidence: float = 0.0


# ==================== Test Framework ====================

class ComprehensiveTestFramework:
    """
    Comprehensive test framework for evaluating deep researcher strategies
    """

    def __init__(
        self,
        test_questions_path: str = "tests/deepresearcher/test_questions.json",
        results_dir: str = "test_results"
    ):
        """
        Initialize test framework

        Args:
            test_questions_path: Path to test questions JSON
            results_dir: Directory to store results
        """
        self.test_questions_path = test_questions_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Load test questions
        with open(test_questions_path, 'r') as f:
            data = json.load(f)
            self.test_questions = data['test_questions']
            self.model_configs = data['model_configurations']
            self.test_parameters = data['test_parameters']

        # Storage for results
        self.sessions: Dict[str, ModelTestSession] = {}

        print(f"📋 Loaded {len(self.test_questions)} test questions")
        print(f"🔧 Configured for {len(self.model_configs)} models")

    def create_llm_instance(self, provider: str, model: str, timeout: int = 300):
        """Create LLM instance with error handling"""
        try:
            llm = create_llm(
                provider,
                model=model,
                timeout=timeout,
                temperature=0.1  # Low for consistency
            )
            print(f"✅ Created {provider}/{model} instance")
            return llm
        except Exception as e:
            print(f"❌ Failed to create {provider}/{model}: {e}")
            return None

    def test_strategy_a(
        self,
        llm: Any,
        query: str,
        test_config: Dict[str, Any]
    ) -> Tuple[bool, Optional[Any], Optional[str], Optional[str]]:
        """
        Test Strategy A with detailed error capture

        Returns:
            (success, result, error_message, error_trace)
        """
        try:
            researcher = BasicDeepResearcherA(
                llm=llm,
                max_react_iterations=test_config.get('max_react_iterations', 2),
                max_parallel_paths=2,
                max_sources=test_config.get('max_sources', 15),
                debug=False
            )

            result = researcher.research(query, max_depth=1)
            return True, result, None, None

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            return False, None, error_msg, error_trace

    def test_strategy_b(
        self,
        llm: Any,
        query: str,
        test_config: Dict[str, Any]
    ) -> Tuple[bool, Optional[Any], Optional[str], Optional[str]]:
        """
        Test Strategy B with detailed error capture

        Returns:
            (success, result, error_message, error_trace)
        """
        try:
            researcher = BasicDeepResearcherB(
                llm=llm,
                max_sources=test_config.get('max_sources', 15),
                quality_threshold=0.6,
                extract_full_content=False,  # Disabled for speed
                debug=False
            )

            result = researcher.research(query, depth="medium")
            return True, result, None, None

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            return False, None, error_msg, error_trace

    def run_single_test(
        self,
        llm: Any,
        model_provider: str,
        model_name: str,
        strategy: str,
        question: Dict[str, Any],
        test_config: Dict[str, Any]
    ) -> TestResult:
        """Run a single test and collect metrics"""

        test_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        print(f"\n🔬 Test {strategy.upper()} on: {question['query'][:50]}...")

        start_time = time.time()

        # Execute test
        if strategy == "A":
            success, result, error_msg, error_trace = self.test_strategy_a(
                llm, question['query'], test_config
            )
        else:
            success, result, error_msg, error_trace = self.test_strategy_b(
                llm, question['query'], test_config
            )

        duration = time.time() - start_time

        # Collect metrics
        test_result = TestResult(
            test_id=test_id,
            timestamp=timestamp,
            model_provider=model_provider,
            model_name=model_name,
            strategy=strategy,
            query_id=question['id'],
            query=question['query'],
            query_category=question['category'],
            query_difficulty=question['difficulty'],
            success=success,
            duration_seconds=duration,
            error_message=error_msg,
            error_trace=error_trace
        )

        # Extract metrics if successful
        if success and result:
            test_result.sources_probed = len(result.sources_probed)
            test_result.sources_selected = len(result.sources_selected)
            test_result.key_findings_count = len(result.key_findings)
            test_result.confidence_score = result.confidence_score
            test_result.title = result.title
            test_result.summary_length = len(result.summary)
            test_result.has_structured_report = bool(result.detailed_report)
            test_result.strategy_metadata = result.research_metadata

            # Store output sample
            test_result.output_sample = {
                'title': result.title,
                'summary': result.summary[:200] + "..." if len(result.summary) > 200 else result.summary,
                'key_findings': result.key_findings[:3],
                'confidence': result.confidence_score
            }

            print(f"   ✅ Success! Duration: {duration:.1f}s | Confidence: {result.confidence_score:.2f}")
        else:
            print(f"   ❌ Failed after {duration:.1f}s: {error_msg}")

        return test_result

    def run_model_test_session(
        self,
        model_config: Dict[str, Any],
        strategies: List[str] = ["A", "B"],
        question_ids: Optional[List[str]] = None
    ) -> ModelTestSession:
        """
        Run full test session for a specific model

        Args:
            model_config: Model configuration dict
            strategies: List of strategies to test (default: both)
            question_ids: Specific questions to test (default: all)
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        provider = model_config['provider']
        model = model_config['model']

        print(f"\n{'='*80}")
        print(f"  TEST SESSION: {provider}/{model}")
        print(f"{'='*80}")

        session = ModelTestSession(
            session_id=session_id,
            timestamp=timestamp,
            model_provider=provider,
            model_name=model,
            model_description=model_config.get('description', '')
        )

        # Create LLM instance
        llm = self.create_llm_instance(
            provider,
            model,
            timeout=self.test_parameters.get('timeout', 300)
        )

        if not llm:
            print(f"❌ Could not create LLM instance for {provider}/{model}")
            return session

        # Filter questions if needed
        questions_to_test = self.test_questions
        if question_ids:
            questions_to_test = [q for q in self.test_questions if q['id'] in question_ids]

        # Test each strategy
        for strategy in strategies:
            print(f"\n📊 Testing Strategy {strategy}")
            print(f"{'─'*80}")

            for question in questions_to_test:
                test_config = {
                    'max_sources': self.test_parameters.get('max_sources', [15])[0],
                    'max_react_iterations': self.test_parameters.get('max_react_iterations', [2])[0]
                }

                result = self.run_single_test(
                    llm,
                    provider,
                    model,
                    strategy,
                    question,
                    test_config
                )

                session.test_results.append(result)
                session.total_tests += 1

                if result.success:
                    session.successful_tests += 1
                else:
                    session.failed_tests += 1

        # Calculate session statistics
        if session.successful_tests > 0:
            successful_results = [r for r in session.test_results if r.success]
            session.average_duration = sum(r.duration_seconds for r in successful_results) / len(successful_results)
            session.average_confidence = sum(r.confidence_score for r in successful_results) / len(successful_results)

        # Store session
        self.sessions[session_id] = session

        # Save results
        self.save_session(session)

        print(f"\n{'='*80}")
        print(f"  SESSION SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {session.total_tests}")
        print(f"Successful: {session.successful_tests}")
        print(f"Failed: {session.failed_tests}")
        print(f"Success rate: {session.successful_tests/session.total_tests*100:.1f}%")
        if session.successful_tests > 0:
            print(f"Average duration: {session.average_duration:.1f}s")
            print(f"Average confidence: {session.average_confidence:.2f}")

        return session

    def save_session(self, session: ModelTestSession):
        """Save session results to JSON"""
        filename = f"session_{session.model_provider}_{session.model_name.replace('/', '_')}_{session.timestamp.replace(':', '-')}.json"
        filepath = self.results_dir / filename

        # Convert to dict
        session_dict = asdict(session)

        with open(filepath, 'w') as f:
            json.dump(session_dict, f, indent=2)

        print(f"\n💾 Results saved to: {filepath}")

    def load_all_sessions(self) -> List[ModelTestSession]:
        """Load all saved test sessions"""
        sessions = []

        for filepath in self.results_dir.glob("session_*.json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Convert back to dataclass (simplified)
                sessions.append(data)

        return sessions

    def print_summary_table(self):
        """Print summary table of all sessions"""
        if not self.sessions:
            print("No test sessions to summarize")
            return

        print(f"\n{'='*100}")
        print(f"  TEST RESULTS SUMMARY")
        print(f"{'='*100}")
        print(f"{'Model':<40} {'Strategy':<10} {'Success':<10} {'Avg Time':<12} {'Avg Conf':<10}")
        print(f"{'─'*100}")

        for session in self.sessions.values():
            for strategy in ['A', 'B']:
                strategy_results = [r for r in session.test_results if r.strategy == strategy]
                if not strategy_results:
                    continue

                successful = [r for r in strategy_results if r.success]
                success_rate = f"{len(successful)}/{len(strategy_results)}"

                avg_time = "N/A"
                avg_conf = "N/A"
                if successful:
                    avg_time = f"{sum(r.duration_seconds for r in successful) / len(successful):.1f}s"
                    avg_conf = f"{sum(r.confidence_score for r in successful) / len(successful):.2f}"

                model_display = f"{session.model_provider}/{session.model_name}"
                print(f"{model_display:<40} {strategy:<10} {success_rate:<10} {avg_time:<12} {avg_conf:<10}")

        print(f"{'='*100}\n")


# ==================== Convenience Functions ====================

def quick_test(provider: str, model: str, strategy: str = "A", question_id: str = "simple_ml_basics"):
    """Quick test of a single configuration"""
    framework = ComprehensiveTestFramework()

    model_config = {
        'provider': provider,
        'model': model,
        'description': f'Quick test of {provider}/{model}'
    }

    session = framework.run_model_test_session(
        model_config,
        strategies=[strategy],
        question_ids=[question_id]
    )

    return session


if __name__ == "__main__":
    # Example usage
    framework = ComprehensiveTestFramework()

    # Quick test with Ollama
    print("Running quick test with Ollama model...")
    session = quick_test("ollama", "qwen3:4b-instruct-2507-q4_K_M", "A", "simple_ml_basics")

    framework.print_summary_table()
