"""
Comprehensive tests for custom assessment criteria functionality.

Tests the new custom_criteria parameter in BasicSession.generate_assessment()
and BasicJudge.evaluate() methods.

NO MOCKING - All tests use real Ollama provider implementation.
"""

import pytest
from abstractcore import create_llm
from abstractcore.core.session import BasicSession
from abstractcore.processing import BasicJudge


@pytest.fixture
def ollama_llm():
    """Create Ollama LLM for testing (real implementation)"""
    try:
        llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
        # Verify Ollama is available
        response = llm.generate("test", max_output_tokens=5)
        return llm
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")


class TestCustomCriteriaBasicJudge:
    """Test custom_criteria parameter in BasicJudge.evaluate()"""

    def test_custom_criteria_only(self, ollama_llm):
        """Test evaluation with custom criteria only (no predefined criteria)"""
        judge = BasicJudge(ollama_llm)

        content = "The machine learning model achieved 95% accuracy on the test set."
        custom_criteria = {
            "technical_accuracy": "Is the technical information accurate and precise?",
            "statistical_validity": "Are the statistics presented in a valid way?"
        }

        result = judge.evaluate(
            content=content,
            context="technical validation",
            custom_criteria=custom_criteria
        )

        # Verify result structure
        assert "overall_score" in result
        assert "judge_summary" in result
        assert "criteria_used" in result
        assert result["overall_score"] >= 1
        assert result["overall_score"] <= 5

        # Verify custom criteria were used
        criteria_used = result.get("criteria_used", [])
        assert "technical_accuracy" in criteria_used
        assert "statistical_validity" in criteria_used

        # NEW: Verify custom scores are returned
        assert "custom_scores" in result
        custom_scores = result.get("custom_scores")
        assert custom_scores is not None
        assert "technical_accuracy" in custom_scores
        assert "statistical_validity" in custom_scores
        assert 1 <= custom_scores["technical_accuracy"] <= 5
        assert 1 <= custom_scores["statistical_validity"] <= 5

    def test_mixed_predefined_and_custom_criteria(self, ollama_llm):
        """Test evaluation with both predefined and custom criteria"""
        judge = BasicJudge(ollama_llm)

        from abstractcore.processing.basic_judge import JudgmentCriteria

        content = "Our quarterly revenue increased by 23%, driven primarily by strong SaaS growth."

        # Predefined criteria (enable only clarity)
        predefined = JudgmentCriteria(
            is_clear=True,
            is_simple=False,
            is_actionable=False,
            is_sound=False,
            is_innovative=False,
            is_working=False,
            is_relevant=False,
            is_complete=False,
            is_coherent=False
        )

        # Custom criteria for financial analysis
        custom_criteria = {
            "financial_accuracy": "Are the financial figures plausible and well-presented?",
            "business_context": "Is appropriate business context provided?"
        }

        result = judge.evaluate(
            content=content,
            context="financial report review",
            criteria=predefined,
            custom_criteria=custom_criteria
        )

        # Verify result
        assert "overall_score" in result
        assert result["overall_score"] >= 1
        assert result["overall_score"] <= 5

        # Verify both types of criteria were used
        criteria_used = result.get("criteria_used", [])
        assert "clarity" in criteria_used  # predefined
        assert "financial_accuracy" in criteria_used  # custom
        assert "business_context" in criteria_used  # custom

    def test_custom_criteria_with_underscores(self, ollama_llm):
        """Test that criteria names with underscores are formatted correctly"""
        judge = BasicJudge(ollama_llm)

        content = "The API endpoint returns user data in JSON format."
        custom_criteria = {
            "api_design_quality": "Is the API design following best practices?",
            "data_format_clarity": "Is the data format clearly specified?"
        }

        result = judge.evaluate(
            content=content,
            context="API documentation review",
            custom_criteria=custom_criteria
        )

        # Verify custom criteria with underscores were processed
        criteria_used = result.get("criteria_used", [])
        assert "api_design_quality" in criteria_used
        assert "data_format_clarity" in criteria_used


class TestCustomCriteriaBasicSession:
    """Test custom_criteria parameter in BasicSession.generate_assessment()"""

    def test_session_assessment_custom_criteria_only(self, ollama_llm):
        """Test session assessment with custom criteria only"""
        session = BasicSession(provider=ollama_llm)

        # Simulate a data analysis conversation
        session.generate("What is the mean of [1, 2, 3, 4, 5]?")
        session.generate("What is the standard deviation?")

        # Digital Article use case: data analysis quality assessment
        custom_criteria = {
            "logical_coherence": "Are the results logically consistent?",
            "result_plausibility": "Are the findings plausible given the data?",
            "assumption_validity": "Were statistical assumptions properly checked?",
            "interpretation_quality": "Is the interpretation appropriate and sound?",
            "completeness": "Are limitations and caveats acknowledged?"
        }

        assessment = session.generate_assessment(
            custom_criteria=custom_criteria
        )

        # Verify assessment structure
        assert "overall_score" in assessment
        assert "custom_criteria" in assessment
        assert assessment["custom_criteria"] == custom_criteria
        assert "judge_summary" in assessment
        assert "created_at" in assessment

        # Verify score is valid
        assert assessment["overall_score"] >= 1
        assert assessment["overall_score"] <= 5

    def test_session_assessment_mixed_criteria(self, ollama_llm):
        """Test session assessment with both predefined and custom criteria"""
        session = BasicSession(provider=ollama_llm)

        # Simple conversation
        session.generate("Explain what a JWT token is.")

        # Mixed criteria: standard + custom
        assessment = session.generate_assessment(
            criteria={
                "clarity": True,
                "completeness": True,
                "coherence": False
            },
            custom_criteria={
                "technical_depth": "Does the explanation show appropriate technical depth?",
                "security_awareness": "Are security considerations mentioned?"
            }
        )

        # Verify assessment structure
        assert "overall_score" in assessment
        assert "criteria" in assessment
        assert "custom_criteria" in assessment
        assert assessment["criteria"]["clarity"] is True
        assert assessment["criteria"]["completeness"] is True
        assert "technical_depth" in assessment["custom_criteria"]
        assert "security_awareness" in assessment["custom_criteria"]

    def test_session_assessment_backward_compatibility(self, ollama_llm):
        """Test that existing code still works (no custom_criteria)"""
        session = BasicSession(provider=ollama_llm)

        session.generate("What is Python?")

        # Old API: only predefined criteria (backward compatible)
        assessment = session.generate_assessment()

        # Should work without custom_criteria
        assert "overall_score" in assessment
        assert "criteria" in assessment
        assert "custom_criteria" in assessment
        assert assessment["custom_criteria"] is None  # Not provided

    def test_session_assessment_empty_custom_criteria(self, ollama_llm):
        """Test that empty custom criteria dict works correctly"""
        session = BasicSession(provider=ollama_llm)

        session.generate("Hello world")

        # Empty custom criteria should be handled gracefully
        assessment = session.generate_assessment(
            custom_criteria={}
        )

        assert "overall_score" in assessment
        assert assessment["custom_criteria"] == {}


class TestCustomCriteriaDomainExamples:
    """Test real-world domain-specific use cases"""

    def test_code_review_criteria(self, ollama_llm):
        """Test code review domain criteria"""
        judge = BasicJudge(ollama_llm)

        code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price']
    return total
"""

        custom_criteria = {
            "code_quality": "Is the code clean and well-structured?",
            "error_handling": "Are edge cases and errors handled appropriately?",
            "performance": "Is the implementation efficient?",
            "maintainability": "Is the code easy to understand and maintain?"
        }

        result = judge.evaluate(
            content=code,
            context="Python code review",
            custom_criteria=custom_criteria
        )

        assert result["overall_score"] >= 1
        assert result["overall_score"] <= 5
        criteria_used = result.get("criteria_used", [])
        assert "code_quality" in criteria_used
        assert "error_handling" in criteria_used

    def test_medical_diagnosis_criteria(self, ollama_llm):
        """Test medical/safety domain criteria"""
        judge = BasicJudge(ollama_llm)

        content = "Patient presents with fever and cough. Diagnosis: Common cold. Treatment: Rest and fluids."

        custom_criteria = {
            "diagnostic_reasoning": "Is the diagnostic reasoning sound and evidence-based?",
            "safety_considerations": "Are patient safety considerations addressed?",
            "treatment_appropriateness": "Is the treatment plan appropriate for the diagnosis?"
        }

        result = judge.evaluate(
            content=content,
            context="medical case review",
            custom_criteria=custom_criteria
        )

        assert result["overall_score"] >= 1
        criteria_used = result.get("criteria_used", [])
        assert "diagnostic_reasoning" in criteria_used
        assert "safety_considerations" in criteria_used

    def test_educational_content_criteria(self, ollama_llm):
        """Test educational domain criteria"""
        judge = BasicJudge(ollama_llm)

        content = "Photosynthesis is how plants make food using sunlight, water, and carbon dioxide."

        custom_criteria = {
            "age_appropriateness": "Is the content appropriate for the target age group?",
            "pedagogical_quality": "Does it follow good teaching principles?",
            "engagement_potential": "Is the content engaging and interesting?"
        }

        result = judge.evaluate(
            content=content,
            context="elementary science lesson",
            custom_criteria=custom_criteria
        )

        assert result["overall_score"] >= 1
        criteria_used = result.get("criteria_used", [])
        assert "age_appropriateness" in criteria_used
        assert "pedagogical_quality" in criteria_used
