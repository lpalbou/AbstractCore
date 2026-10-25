"""MLX text-lane usage counts the prompt with the LOADED tokenizer, not a char estimate.

Live find 2026-10-25: a Qwen3.6 prompt of 4,899 real tokens was reported as 7,755
input tokens (`TokenUtils.estimate_tokens`) while output tokens were already exact.
"""

from __future__ import annotations

from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.utils.token_utils import TokenUtils


class _WordTokenizer:
    """One token per whitespace-separated word; BOS literal "<s>" like real tokenizers."""

    bos_token = "<s>"

    def encode(self, text, add_special_tokens=False):
        ids = [len(w) for w in str(text).split()]
        if add_special_tokens:
            ids = [1] + ids
        return ids


def _provider() -> MLXProvider:
    p = MLXProvider.__new__(MLXProvider)
    p.model = "qwen/qwen3.6-35b-a3b"
    p.tokenizer = _WordTokenizer()
    return p


PROMPT = "<|im_start|>user\n" + " ".join(["abcdefghijklmnopqrstuvwxyz0123456789"] * 50) + "<|im_end|>"


def test_calculate_usage_counts_the_prompt_with_the_tokenizer() -> None:
    p = _provider()
    usage = p._calculate_usage(PROMPT, "one two three")
    exact_in = len(p._encode_prompt_token_ids(PROMPT))
    assert usage["input_tokens"] == exact_in == usage["prompt_tokens"]
    # The estimate is far off for this prompt: the test would pass by accident otherwise.
    assert TokenUtils.estimate_tokens(PROMPT, p.model) != exact_in
    assert usage["output_tokens"] == 3 == usage["completion_tokens"]
    assert usage["total_tokens"] == exact_in + 3


def test_calculate_usage_follows_mlx_lm_bos_inference() -> None:
    p = _provider()
    # A prompt that already starts with the BOS literal is encoded WITHOUT adding another.
    assert p._calculate_usage("<s> a b", "")["input_tokens"] == 3
    assert p._calculate_usage("a b", "")["input_tokens"] == 3  # BOS added: [1, a, b]


def test_reply_usage_reports_the_exact_prompt_count() -> None:
    p = _provider()
    usage, finish = p._reply_usage_and_finish(
        raw_text="x y",
        reasoning=None,
        prompt=PROMPT,
        usage_prompt=None,
        input_embeddings=None,
        max_tokens=100,
        native_result=None,
    )
    assert usage["input_tokens"] == len(p._encode_prompt_token_ids(PROMPT))
    assert usage["output_tokens"] == 2
    assert finish == "stop"


def test_calculate_usage_falls_back_to_the_estimate_without_a_tokenizer() -> None:
    p = MLXProvider.__new__(MLXProvider)
    p.model = "qwen/qwen3.6-35b-a3b"
    p.tokenizer = None
    usage = p._calculate_usage(PROMPT, "one two")
    assert usage["input_tokens"] == TokenUtils.estimate_tokens(PROMPT, p.model)
    assert usage["output_tokens"] == TokenUtils.estimate_tokens("one two", p.model)
