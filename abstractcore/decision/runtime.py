"""The decision runtime: one loaded decision model answering typed questions.

``DecisionModel`` binds a tokenizer, a backend (transformers / mlx-lm /
llama.cpp) and the pointer head, and exposes ``decide(state, questions)``.
The state prefix (its computed cache) is kept across calls so repeated
questions about the same document only pay for the question tokens.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .encoding import MAX_ROW_TOKENS, MAX_STATE_TOKENS, ContextOverflow, EncodedRequest, encode
from .render import answers_token_count, to_answers, to_record
from .types import DecisionRequest, DecisionResponse, DecisionUsage


class DecisionModel:
    def __init__(
        self,
        backend: Any,
        tokenizer: Any,
        *,
        model_id: str,
        provider: Optional[str] = None,
        prefix_cache_size: int = 4,
        prefix_cache_max_tokens: int = MAX_STATE_TOKENS,
        max_state_tokens: int = MAX_STATE_TOKENS,
        max_row_tokens: int = MAX_ROW_TOKENS,
        truncate_states: bool = False,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.provider = provider
        self.max_state_tokens = int(max_state_tokens)
        self.max_row_tokens = int(max_row_tokens)
        self.truncate_states = bool(truncate_states)
        self._prefix_cache_size = max(0, int(prefix_cache_size))
        self._prefix_cache_max_tokens = int(prefix_cache_max_tokens)
        self._prefixes: "OrderedDict[tuple, Any]" = OrderedDict()
        self._lock = threading.RLock()
        self.prefix_hits = 0
        self.prefix_misses = 0

    # --- public ---------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", type(self.backend).__name__)

    @property
    def precision(self) -> Optional[str]:
        return getattr(self.backend, "precision", None)

    def encode(self, request: DecisionRequest, truncate: Optional[bool] = None):
        record, meta = to_record(request)
        encoded = encode(
            self.tokenizer, record,
            max_state=self.max_state_tokens, max_row=self.max_row_tokens,
            truncate=self.truncate_states if truncate is None else bool(truncate),
        )
        return encoded, meta

    def decide(
        self,
        state: Any = None,
        questions: Optional[Dict[str, Any]] = None,
        *,
        request: Optional[DecisionRequest] = None,
        truncate: Optional[bool] = None,
    ) -> DecisionResponse:
        """Answer ``questions`` about ``state``. ``questions`` accepts typed questions,
        System One dicts or the Python shorthand (see ``types.coerce_question``)."""
        if request is None:
            if questions is None:
                raise ValueError("decide() needs questions")
            request = DecisionRequest(state=state, questions=questions)
        encoded, meta = self.encode(request, truncate=truncate)
        kinds = [m["type"] for m in meta]
        with self._lock:
            started = time.perf_counter()
            probs, hit = self._probabilities(encoded, kinds)
            latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        answers = to_answers(probs, meta)
        usage = DecisionUsage(
            input_tokens=encoded.total_tokens,
            output_tokens=answers_token_count(self.tokenizer.encode_text, answers),
            state_tokens=encoded.state_tokens,
            state_tokens_used=len(encoded.state_ids),
            prefix_cache_hit=hit,
        )
        return DecisionResponse(
            model=request.model or self.model_id,
            answers=answers,
            usage=usage,
            latency_ms=latency_ms,
            truncated=encoded.state_truncated,
            provider=self.provider,
            backend=self.backend_name,
            precision=self.precision,
        )

    def probabilities(self, request: DecisionRequest) -> List[np.ndarray]:
        """Raw per-question probability vectors (evaluation and parity checks)."""
        encoded, meta = self.encode(request)
        with self._lock:
            probs, _ = self._probabilities(encoded, [m["type"] for m in meta])
        return probs

    def clear_prefix_cache(self) -> None:
        with self._lock:
            for handle in self._prefixes.values():
                self.backend.release(handle)
            self._prefixes.clear()

    def unload(self) -> None:
        self.clear_prefix_cache()
        self.backend.unload()

    # --- internals ------------------------------------------------------------

    def _probabilities(self, encoded: EncodedRequest, kinds: Sequence[str]):
        key = tuple(encoded.state_ids)
        handle = self._prefixes.get(key)
        hit = handle is not None
        if hit:
            self._prefixes.move_to_end(key)
            self.prefix_hits += 1
        else:
            self.prefix_misses += 1
            handle = self.backend.prefix(encoded.state_ids)
            if self._prefix_cache_size and len(key) <= self._prefix_cache_max_tokens:
                self._prefixes[key] = handle
                while len(self._prefixes) > self._prefix_cache_size:
                    _, old = self._prefixes.popitem(last=False)
                    self.backend.release(old)
        logits = self.backend.question_logits(handle, encoded.questions, kinds)
        if not hit and key not in self._prefixes:
            self.backend.release(handle)
        probs = [_softmax(z) for z in logits]
        return probs, hit


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


__all__ = ["DecisionModel", "ContextOverflow"]
