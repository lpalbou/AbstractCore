from __future__ import annotations

from abstractcore.processing import BasicSummarizer


class _DummyLLM:
    def __init__(self, *, model: str, max_tokens: int, max_output_tokens: int, max_input_tokens: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens


def test_should_not_chunk_when_within_budget_even_if_longer_than_chunk_size() -> None:
    llm = _DummyLLM(model="qwen3", max_tokens=100_000, max_output_tokens=8_000, max_input_tokens=92_000)
    summarizer = BasicSummarizer(llm, max_chunk_size=8_000, max_tokens=100_000, max_output_tokens=8_000)

    # > 8k chars, but tiny token estimate compared to the available budget.
    text = "x" * 9_001
    assert summarizer._should_chunk_by_tokens(text) is False


def test_should_chunk_when_over_token_budget() -> None:
    llm = _DummyLLM(model="qwen3", max_tokens=10_000, max_output_tokens=2_000, max_input_tokens=8_000)
    summarizer = BasicSummarizer(llm, max_chunk_size=8_000, max_tokens=10_000, max_output_tokens=2_000)

    # Large enough to exceed the token_limit (~8k, given the safety floor) with heuristic estimation.
    text = ("a " * 40_000).strip()
    assert summarizer._should_chunk_by_tokens(text) is True

