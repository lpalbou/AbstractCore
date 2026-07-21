"""0817 axis 2: tokenizer/chat-template fingerprint computation + verdicts.

The fingerprint identifies the tokenizer STATE that maps text to token ids
(the thing a KV artifact actually encodes), not the rendered text. These pins
hold the computation across the tokenizer shapes in this codebase and the
shared three-way verdict every gate consumes.
"""

from __future__ import annotations

from abstractcore.providers.tokenizer_fingerprint import (
    check_tokenizer_fingerprint,
    tokenizer_fingerprint_for,
)


class _FastBackend:
    def __init__(self, state: str) -> None:
        self._state = state

    def to_str(self) -> str:
        return self._state


class _FastTokenizer:
    """Shape of a transformers fast tokenizer."""

    def __init__(self, state: str = "state-v1", template: str = "tmpl-v1", eos: int = 2) -> None:
        self.backend_tokenizer = _FastBackend(state)
        self.chat_template = template
        self.bos_token_id = 1
        self.eos_token_id = eos
        self.pad_token_id = None
        self.unk_token_id = 0


class _WrappedTokenizer:
    """Shape of mlx_lm's TokenizerWrapper (HF tokenizer rides `_tokenizer`)."""

    def __init__(self, inner: _FastTokenizer) -> None:
        self._tokenizer = inner


class _SlowTokenizer:
    """No fast backend: vocab-summary tier."""

    def __init__(self, vocab_size: int = 10, template: str = "tmpl-v1") -> None:
        self._vocab = {f"tok{i}": i for i in range(vocab_size)}
        self.chat_template = template
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = None
        self.unk_token_id = 0

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return {"<extra>": 9999}


def test_fast_tokenizer_fingerprint_is_stable_and_tiered() -> None:
    a = tokenizer_fingerprint_for(_FastTokenizer())
    b = tokenizer_fingerprint_for(_FastTokenizer())
    assert a == b
    assert a.startswith("tokenizer-full:sha256:")


def test_state_template_and_special_changes_all_change_the_fingerprint() -> None:
    base = tokenizer_fingerprint_for(_FastTokenizer())
    assert tokenizer_fingerprint_for(_FastTokenizer(state="state-v2")) != base
    assert tokenizer_fingerprint_for(_FastTokenizer(template="tmpl-v2")) != base
    assert tokenizer_fingerprint_for(_FastTokenizer(eos=99)) != base


def test_mlx_wrapper_shape_matches_inner_tokenizer() -> None:
    inner = _FastTokenizer()
    assert tokenizer_fingerprint_for(_WrappedTokenizer(inner)) == tokenizer_fingerprint_for(inner)


def test_dict_chat_template_is_supported() -> None:
    tok = _FastTokenizer()
    tok.chat_template = {"default": "tmpl-a", "tool_use": "tmpl-b"}
    a = tokenizer_fingerprint_for(tok)
    tok2 = _FastTokenizer()
    tok2.chat_template = {"default": "tmpl-a", "tool_use": "tmpl-CHANGED"}
    assert a.startswith("tokenizer-full:sha256:")
    assert tokenizer_fingerprint_for(tok2) != a


def test_slow_tokenizer_uses_vocab_tier() -> None:
    a = tokenizer_fingerprint_for(_SlowTokenizer())
    assert a.startswith("tokenizer-vocab:sha256:")
    assert tokenizer_fingerprint_for(_SlowTokenizer(vocab_size=11)) != a
    assert tokenizer_fingerprint_for(_SlowTokenizer(template="tmpl-v2")) != a


def test_no_tokenizer_state_returns_empty() -> None:
    assert tokenizer_fingerprint_for(None) == ""
    assert tokenizer_fingerprint_for(object()) == ""


def test_check_verdicts_cover_all_four_cases() -> None:
    assert check_tokenizer_fingerprint("", "") == "unverified_stored"
    assert check_tokenizer_fingerprint("", "tokenizer-full:sha256:abc") == "unverified_stored"
    assert check_tokenizer_fingerprint("tokenizer-full:sha256:abc", "") == "unverified_current"
    assert check_tokenizer_fingerprint("tokenizer-full:sha256:abc", "tokenizer-full:sha256:abc") == "ok"
    assert check_tokenizer_fingerprint("tokenizer-full:sha256:abc", "tokenizer-full:sha256:def") == "mismatch"


def test_tier_difference_reads_as_mismatch_fail_safe() -> None:
    # A full-tier recording compared against a vocab-tier current state (or
    # vice versa) compares different subjects; the tier prefix forces a
    # mismatch, which recompiles once and records the stronger tier.
    full = tokenizer_fingerprint_for(_FastTokenizer())
    vocab = tokenizer_fingerprint_for(_SlowTokenizer())
    assert check_tokenizer_fingerprint(full, vocab) == "mismatch"
