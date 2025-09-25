"""
Tests for BasicSummarizer - Real implementation testing (no mocking)

Following AbstractCore's testing philosophy of testing real implementations
with real models and real content.
"""

import pytest
from abstractllm import create_llm
from abstractllm.processing import BasicSummarizer, SummaryStyle, SummaryLength


class TestBasicSummarizer:
    """Test BasicSummarizer with real LLM and real content"""

    @pytest.fixture
    def llm(self):
        """Create a real LLM instance for testing"""
        return create_llm("openai", model="gpt-4o-mini")

    @pytest.fixture
    def summarizer(self, llm):
        """Create BasicSummarizer instance"""
        return BasicSummarizer(llm)

    @pytest.fixture
    def readme_content(self):
        """Load the actual README.md content for testing"""
        import os
        readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    def test_basic_summarization(self, summarizer, readme_content):
        """Test basic summarization functionality"""
        result = summarizer.summarize(readme_content)

        # Basic structure validation
        assert result.summary is not None
        assert len(result.summary) > 0
        assert isinstance(result.key_points, list)
        assert len(result.key_points) >= 3
        assert len(result.key_points) <= 8
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.focus_alignment <= 1.0
        assert result.word_count_original > 0
        assert result.word_count_summary > 0

        print(f"\n{'='*60}")
        print("BASIC SUMMARIZATION TEST")
        print(f"{'='*60}")
        print(f"Summary:\n{result.summary}")
        print(f"\nKey Points:")
        for i, point in enumerate(result.key_points, 1):
            print(f"{i}. {point}")
        print(f"\nMetrics:")
        print(f"- Confidence: {result.confidence:.2f}")
        print(f"- Focus alignment: {result.focus_alignment:.2f}")
        print(f"- Compression ratio: {result.word_count_original / result.word_count_summary:.1f}x")

        # Content validation - check for key AbstractCore concepts
        summary_lower = result.summary.lower()
        key_points_text = " ".join(result.key_points).lower()
        combined_text = f"{summary_lower} {key_points_text}"

        # Essential AbstractCore concepts that should be preserved
        essential_concepts = [
            "abstractcore",  # The main name
            "llm",          # Core concept
            "provider",     # Key architecture element
        ]

        preserved_concepts = []
        for concept in essential_concepts:
            if concept in combined_text:
                preserved_concepts.append(concept)

        print(f"- Preserved essential concepts: {preserved_concepts}")

        # At least 2 out of 3 essential concepts should be preserved
        assert len(preserved_concepts) >= 2, f"Too few essential concepts preserved: {preserved_concepts}"

    def test_executive_style_summarization(self, summarizer, readme_content):
        """Test executive style summarization with business focus"""
        result = summarizer.summarize(
            readme_content,
            focus="business value and practical applications",
            style=SummaryStyle.EXECUTIVE,
            length=SummaryLength.DETAILED
        )

        print(f"\n{'='*60}")
        print("EXECUTIVE STYLE TEST")
        print(f"{'='*60}")
        print(f"Summary:\n{result.summary}")
        print(f"\nKey Points:")
        for i, point in enumerate(result.key_points, 1):
            print(f"{i}. {point}")
        print(f"\nMetrics:")
        print(f"- Confidence: {result.confidence:.2f}")
        print(f"- Focus alignment: {result.focus_alignment:.2f}")

        # Validate structure
        assert result.summary is not None
        assert len(result.key_points) >= 3
        assert result.confidence > 0.0

        # Executive summaries should have good focus alignment when focus is specified
        assert result.focus_alignment > 0.5, f"Poor focus alignment: {result.focus_alignment}"

        # Should contain business/practical terminology
        summary_lower = result.summary.lower()
        business_terms = ["api", "application", "development", "production", "infrastructure"]
        found_terms = [term for term in business_terms if term in summary_lower]

        print(f"- Business terms found: {found_terms}")
        assert len(found_terms) >= 2, f"Executive summary lacks business focus: {found_terms}"

    def test_structured_style_brief(self, summarizer, readme_content):
        """Test structured style with brief length"""
        result = summarizer.summarize(
            readme_content,
            style=SummaryStyle.STRUCTURED,
            length=SummaryLength.BRIEF
        )

        print(f"\n{'='*60}")
        print("STRUCTURED BRIEF TEST")
        print(f"{'='*60}")
        print(f"Summary:\n{result.summary}")
        print(f"\nKey Points:")
        for i, point in enumerate(result.key_points, 1):
            print(f"{i}. {point}")
        print(f"\nMetrics:")
        print(f"- Confidence: {result.confidence:.2f}")
        print(f"- Word count: {result.word_count_summary}")

        # Brief summaries should be concise (allow some flexibility for LLM variance)
        assert len(result.summary.split()) < 120, f"Brief summary too long: {len(result.summary.split())} words"
        assert result.word_count_summary < 120, f"Brief summary word count too high: {result.word_count_summary}"

        # Should still capture key points
        assert len(result.key_points) >= 3

    def test_analytical_style_with_focus(self, summarizer, readme_content):
        """Test analytical style with specific technical focus"""
        result = summarizer.summarize(
            readme_content,
            focus="architecture and technical implementation details",
            style=SummaryStyle.ANALYTICAL,
            length=SummaryLength.STANDARD
        )

        print(f"\n{'='*60}")
        print("ANALYTICAL TECHNICAL FOCUS TEST")
        print(f"{'='*60}")
        print(f"Summary:\n{result.summary}")
        print(f"\nKey Points:")
        for i, point in enumerate(result.key_points, 1):
            print(f"{i}. {point}")
        print(f"\nMetrics:")
        print(f"- Confidence: {result.confidence:.2f}")
        print(f"- Focus alignment: {result.focus_alignment:.2f}")

        # Good focus alignment expected with specific technical focus
        assert result.focus_alignment > 0.6, f"Poor technical focus alignment: {result.focus_alignment}"

        # Should contain technical terminology
        summary_lower = result.summary.lower()
        technical_terms = ["architecture", "implementation", "interface", "provider", "system"]
        found_terms = [term for term in technical_terms if term in summary_lower]

        print(f"- Technical terms found: {found_terms}")
        assert len(found_terms) >= 2, f"Analytical summary lacks technical depth: {found_terms}"

    def test_comprehensive_length(self, summarizer, readme_content):
        """Test comprehensive length produces detailed output"""
        result = summarizer.summarize(
            readme_content,
            length=SummaryLength.COMPREHENSIVE
        )

        print(f"\n{'='*60}")
        print("COMPREHENSIVE LENGTH TEST")
        print(f"{'='*60}")
        print(f"Summary length: {len(result.summary)} characters")
        print(f"Word count: {result.word_count_summary}")
        print(f"Key points count: {len(result.key_points)}")
        print(f"Compression ratio: {result.word_count_original / result.word_count_summary:.1f}x")

        # Comprehensive should be detailed (relative to brief)
        assert len(result.summary) > 300, f"Comprehensive summary too short: {len(result.summary)} chars"
        assert result.word_count_summary > 80, f"Comprehensive word count too low: {result.word_count_summary}"

        # Should have maximum key points
        assert len(result.key_points) >= 5, f"Comprehensive summary has too few key points: {len(result.key_points)}"

    def test_short_text_processing(self, summarizer):
        """Test processing of short text (no chunking needed)"""
        short_text = """
        AbstractCore is a unified interface to all LLM providers. It provides production-grade
        reliability with built-in retry mechanisms, structured output validation, and
        comprehensive event monitoring. The framework supports OpenAI, Anthropic, Ollama,
        and other providers through a single API.
        """

        result = summarizer.summarize(short_text)

        print(f"\n{'='*60}")
        print("SHORT TEXT TEST")
        print(f"{'='*60}")
        print(f"Original: {short_text.strip()}")
        print(f"\nSummary: {result.summary}")
        print(f"Key Points: {result.key_points}")
        print(f"Confidence: {result.confidence:.2f}")

        assert result.summary is not None
        assert len(result.key_points) >= 3
        assert result.confidence > 0.0

        # Should preserve key concepts from short text
        summary_lower = result.summary.lower()
        assert "abstractcore" in summary_lower or "abstract" in summary_lower
        assert "llm" in summary_lower or "provider" in summary_lower

    def test_chunking_behavior(self, summarizer):
        """Test that chunking works for very long documents"""
        # Create a long text by repeating README content
        import os
        readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
        with open(readme_path, 'r', encoding='utf-8') as f:
            base_content = f.read()

        # Create text longer than max_chunk_size (default 8000 chars)
        long_text = base_content * 3  # Should trigger chunking

        result = summarizer.summarize(long_text, length=SummaryLength.STANDARD)

        print(f"\n{'='*60}")
        print("CHUNKING TEST")
        print(f"{'='*60}")
        print(f"Original length: {len(long_text)} characters")
        print(f"Summary: {result.summary}")
        print(f"Confidence: {result.confidence:.2f}")

        assert result.summary is not None
        assert len(result.key_points) >= 3
        assert result.confidence > 0.0

        # The summary should still be coherent despite chunking
        summary_words = len(result.summary.split())
        assert 50 <= summary_words <= 300, f"Chunked summary unexpected length: {summary_words} words"

    def test_error_handling(self, summarizer):
        """Test error handling with edge cases"""
        # Test with very short text
        very_short = "Hello world."
        result = summarizer.summarize(very_short)
        assert result.summary is not None

        # Test with empty-ish text (should handle gracefully)
        minimal_text = "A. B. C."
        result = summarizer.summarize(minimal_text)
        assert result.summary is not None
        print(f"\nMinimal text summary: {result.summary}")

    def test_focus_effectiveness(self, summarizer, readme_content):
        """Test that focus parameter actually changes summary content"""
        # Summarize with different focuses
        tech_result = summarizer.summarize(
            readme_content,
            focus="technical architecture and implementation",
            style=SummaryStyle.OBJECTIVE
        )

        business_result = summarizer.summarize(
            readme_content,
            focus="business value and use cases",
            style=SummaryStyle.OBJECTIVE
        )

        print(f"\n{'='*60}")
        print("FOCUS EFFECTIVENESS TEST")
        print(f"{'='*60}")
        print(f"Technical focus summary:\n{tech_result.summary}")
        print(f"\nBusiness focus summary:\n{business_result.summary}")
        print(f"\nTech focus alignment: {tech_result.focus_alignment:.2f}")
        print(f"Business focus alignment: {business_result.focus_alignment:.2f}")

        # Both should have decent focus alignment
        assert tech_result.focus_alignment > 0.5
        assert business_result.focus_alignment > 0.5

        # Summaries should be different (not identical)
        assert tech_result.summary != business_result.summary, "Different focuses produced identical summaries"

        # Technical focus should contain more technical terms
        tech_lower = tech_result.summary.lower()
        business_lower = business_result.summary.lower()

        technical_indicators = sum(1 for term in ["architecture", "implementation", "interface", "system"]
                                 if term in tech_lower)
        business_indicators = sum(1 for term in ["business", "application", "use", "value"]
                                if term in business_lower)

        print(f"Technical indicators in tech summary: {technical_indicators}")
        print(f"Business indicators in business summary: {business_indicators}")

    def test_different_lengths_produce_different_outputs(self, summarizer, readme_content):
        """Test that different length settings produce appropriately sized outputs"""
        brief_result = summarizer.summarize(readme_content, length=SummaryLength.BRIEF)
        standard_result = summarizer.summarize(readme_content, length=SummaryLength.STANDARD)
        detailed_result = summarizer.summarize(readme_content, length=SummaryLength.DETAILED)

        print(f"\n{'='*60}")
        print("LENGTH COMPARISON TEST")
        print(f"{'='*60}")
        print(f"Brief ({brief_result.word_count_summary} words): {brief_result.summary[:100]}...")
        print(f"Standard ({standard_result.word_count_summary} words): {standard_result.summary[:100]}...")
        print(f"Detailed ({detailed_result.word_count_summary} words): {detailed_result.summary[:100]}...")

        # Length progression should be logical
        assert brief_result.word_count_summary <= standard_result.word_count_summary
        assert standard_result.word_count_summary <= detailed_result.word_count_summary

        print(f"Length progression: {brief_result.word_count_summary} → {standard_result.word_count_summary} → {detailed_result.word_count_summary}")


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])