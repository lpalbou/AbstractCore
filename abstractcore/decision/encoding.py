"""Token packing of a decision request.

The model reads one sequence per question, each continuing from the shared
state:

    <state> state tokens | <q> instruction <opt> option </opt> ... <opt> option </opt> <decide>

Five rarely used Qwen special tokens serve as the delimiters (the adapter was
trained with them, so no vocabulary row is added). Caller text can never forge
a delimiter: every ``<|name|>`` in user text is rewritten to ``<¦name¦>`` before
tokenisation, and user text is tokenised without special tokens.

The pointer head reads the hidden state of each option's ``</opt>`` token and
of the question's ``<decide>`` token, so the encoder records those offsets.
Questions never see each other: a question row is the state followed by that
question only, which every backend runs as an ordinary causal sequence (with
the state computed once and reused through its cache).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence

# (state, question, option, end-of-option, decide)
DELIMITERS = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")

# Serving limits (tokens). A state may hold up to 65,536 tokens and every
# question row (state + its branch) up to 8,192 tokens more.
MAX_STATE_TOKENS = 65_536
MAX_ROW_TOKENS = MAX_STATE_TOKENS + 8_192

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def escape_special_tokens(text: str) -> str:
    """``<|name|>`` -> ``<¦name¦>`` so caller text cannot carry control tokens."""
    return _SPECIAL_RE.sub(r"<¦\1¦>", text)


class DecisionTokenizer(Protocol):
    """What the encoder needs from a tokenizer."""

    pad_id: int

    def encode_text(self, text: str) -> List[int]:
        """Token ids of plain text, no special tokens added or parsed."""

    def delimiter_ids(self) -> List[int]:
        """Ids of DELIMITERS, in order."""


class ContextOverflow(ValueError):
    """A request does not fit the serving context. ``state_tokens``/``max_state``
    are set when the state itself is what overflows."""

    def __init__(self, message: str, state_tokens: int | None = None, max_state: int | None = None):
        super().__init__(message)
        self.state_tokens = state_tokens
        self.max_state = max_state


@dataclass
class EncodedQuestion:
    ids: List[int]                   # the branch tokens (<q> ... <decide>)
    decide: int                      # offset of <decide> within ``ids``
    option_ends: List[int]           # offset of each option's </opt> within ``ids``

    @property
    def n_options(self) -> int:
        return len(self.option_ends)


@dataclass
class EncodedRequest:
    state_ids: List[int]             # <state> + state tokens (possibly truncated)
    questions: List[EncodedQuestion] = field(default_factory=list)
    state_tokens: int = 0            # the whole state's token count, <state> included
    state_truncated: bool = False

    @property
    def total_tokens(self) -> int:
        return len(self.state_ids) + sum(len(q.ids) for q in self.questions)

    def row(self, index: int) -> List[int]:
        """Question ``index`` as one causal row: state tokens then its branch."""
        return self.state_ids + self.questions[index].ids


def encode(
    tokenizer: DecisionTokenizer,
    record: Dict[str, Any],
    *,
    max_state: int = MAX_STATE_TOKENS,
    max_row: int = MAX_ROW_TOKENS,
    truncate: bool = False,
) -> EncodedRequest:
    """Pack ``record`` ({"state": str, "questions": [{"instr": str, "options": [str]}]}).

    A state over ``max_state`` tokens raises ContextOverflow, or is cut to its first
    ``max_state`` tokens when ``truncate`` is set (the result says so). A question
    whose row exceeds ``max_row`` always raises.
    """
    state_tok, q_tok, opt_tok, end_tok, decide_tok = tokenizer.delimiter_ids()
    state_tokens = tokenizer.encode_text(escape_special_tokens(record["state"]))
    state_count = len(state_tokens) + 1
    if state_count > max_state:
        if not truncate:
            raise ContextOverflow(
                f"state is {state_count:,} tokens, over the {max_state:,}-token limit "
                "(the <state> token included): shorten the document or split it across requests",
                state_tokens=state_count,
                max_state=max_state,
            )
        state_tokens = state_tokens[: max_state - 1]
    state_ids = [state_tok] + state_tokens
    encoded = EncodedRequest(state_ids=state_ids, state_tokens=state_count, state_truncated=state_count > max_state)

    for question in record["questions"]:
        ids = [q_tok] + tokenizer.encode_text(escape_special_tokens(question["instr"]))
        option_ends: List[int] = []
        for option in question["options"]:
            ids += [opt_tok] + tokenizer.encode_text(escape_special_tokens(option)) + [end_tok]
            option_ends.append(len(ids) - 1)
        ids.append(decide_tok)
        if len(state_ids) + len(ids) > max_row:
            raise ContextOverflow(
                f"a question row is {len(state_ids) + len(ids):,} tokens with a {len(state_ids):,}-token state, "
                f"over the {max_row:,}-token row limit: shorten the question or its options"
            )
        encoded.questions.append(EncodedQuestion(ids=ids, decide=len(ids) - 1, option_ends=option_ends))
    return encoded


def rows_per_pass(rows: Sequence[Sequence[int]], prefix_len: int = 0, budget: int = 16_384) -> int:
    """How many question rows one forward pass takes: as many as fit ``budget`` tokens
    counting the cached state each row carries, at least one. Rows are independent,
    so the split never changes an answer; it bounds the memory of one pass."""
    longest = max(len(r) for r in rows) if rows else 0
    return max(1, budget // max(1, prefix_len + longest))


__all__ = [
    "DELIMITERS",
    "MAX_STATE_TOKENS",
    "MAX_ROW_TOKENS",
    "escape_special_tokens",
    "DecisionTokenizer",
    "ContextOverflow",
    "EncodedQuestion",
    "EncodedRequest",
    "encode",
    "rows_per_pass",
]
