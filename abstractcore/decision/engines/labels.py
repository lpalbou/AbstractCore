"""Label-logit decision engine (Lev, OpenJev-style checkpoints).

The question is rendered as a chat prompt that lists the options under short
labels (``A``, ``B``, ...). One forward pass of the backbone gives the
next-token logits at the end of the prompt; the logits of the label tokens are
the option scores, divided by the calibration temperature and softmaxed. Lev
reads a choice in two option orders and averages them, and reads a noul as a
0..8 rating whose expectation (over 8) is the probability of "yes".

The backbone is any of the local backends that can return last-token logits:
transformers (``HFLogitsBackend``), mlx-lm (``MLXLogitsBackend``) or llama.cpp
(``GGUFLogitsBackend``, plain logits, no decision-aware llama.cpp needed).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..kinds import LEV_N_RATINGS, LabelKindSpec
from ..render import answers_token_count, to_answers, to_record
from ..types import DecisionRequest, DecisionResponse, DecisionUsage


def _sort_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sort_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sort_keys(v) for v in value]
    return value


class LabelLogitDecisionModel:
    def __init__(self, backend: Any, tokenizer: Any, spec: LabelKindSpec, *, model_id: str, provider: Optional[str] = None):
        """``backend``: exposes ``last_token_logits(ids, candidate_ids) -> np.ndarray`` (one logit per
        candidate), ``name``, ``precision``, ``unload()``. ``tokenizer``: a transformers tokenizer
        (or the llama tokenizer adapter) used to turn the rendered prompt into ids."""
        import jinja2

        self.backend = backend
        self.tokenizer = tokenizer
        self.spec = spec
        self.model_id = model_id
        self.provider = provider
        self.kind = spec.kind
        env = jinja2.Environment(autoescape=False, keep_trailing_newline=True)
        env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
        self._template = env.from_string(spec.template)
        # label tokens: only codes that tokenize to one token are usable (llama.cpp does the same)
        self.labels: List[int] = []
        self.label_texts: List[str] = []
        for code in spec.label_codes:
            ids = self._encode(code)
            if len(ids) == 1:
                self.labels.append(int(ids[0])); self.label_texts.append(code)
            if len(self.labels) >= spec.max_options:
                break
        if len(self.labels) < 2:
            raise ValueError("the tokenizer has no single-token option labels; not a label-logit decision model")

    # --- identity --------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", type(self.backend).__name__)

    @property
    def precision(self) -> Optional[str]:
        return getattr(self.backend, "precision", None)

    # --- prompt --------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, "encode_prompt"):
            return self.tokenizer.encode_prompt(text)
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def render(self, state: Any, qid: str, qtype: str, instructions: Any, options: List[Dict[str, Any]]) -> str:
        if self.spec.sort_json_keys:
            state, instructions = _sort_keys(state), _sort_keys(instructions)
        return self._template.render(id=qid, type=qtype, instructions=instructions, state=state, options=options, images=[])

    def _options(self, q: Any) -> List[Dict[str, Any]]:
        if q.type == "choice":
            return [{"key": k, "description": v} for k, v in q.criteria.items()]
        if q.type == "score":
            return [{"key": str(i), "description": level} for i, level in enumerate(q.criteria)]
        crit = q.criteria or {}
        pair = [{"key": "false", "description": crit.get("false")}, {"key": "true", "description": crit.get("true")}]
        return pair[::-1] if self.spec.noul_mode == "true_first" else pair

    def n_outputs(self, q: Any) -> int:
        if self.spec.kind == "lev" and q.type == "noul":
            return LEV_N_RATINGS
        return len(self._options(q))

    # --- scoring --------------------------------------------------------------------

    def _prompts(self, state: Any, qid: str, q: Any):
        """(token ids, n_outputs) for every prompt variant of one question."""
        options = self._options(q)
        n = self.n_outputs(q)
        if n > len(self.labels):
            raise ValueError(f"too many options ({n}); this model supports at most {len(self.labels)}")
        variants = self.spec.choice_variants if (q.type == "choice" and len(options) > 1) else 1
        prompts = []
        for variant in range(variants):
            ordered = options if variant == 0 else options[::-1]
            labelled = [dict(o, label=self.label_texts[i]) for i, o in enumerate(ordered)]
            prompts.append(self._encode(self.render(state, qid, q.type, q.instructions, labelled)))
        return prompts, n, variants, len(options)

    @staticmethod
    def shared_prefix_length(sequences: Sequence[Sequence[int]]) -> int:
        """Length of the token prefix every sequence shares (the system text and the state)."""
        if not sequences:
            return 0
        first = sequences[0]
        n = min(len(s) for s in sequences)
        for i in range(n):
            t = first[i]
            if any(s[i] != t for s in sequences[1:]):
                return i
        return n

    def probabilities(self, request: DecisionRequest) -> List[np.ndarray]:
        """One probability vector per question. Every prompt of the request (all questions, all option
        orders) shares its leading tokens up to the first question-specific token; that prefix runs
        once through the backbone and is reused for every suffix (the state is read once)."""
        plans = [(qid, q) + self._prompts(request.state, qid, q) for qid, q in request.questions.items()]
        all_prompts = [ids for _, _, prompts, _, _, _ in plans for ids in prompts]
        shared = self.shared_prefix_length(all_prompts)
        # keep at least one token for every suffix (a prompt identical to another would otherwise be empty)
        shared = min(shared, min(len(p) for p in all_prompts) - 1)
        can_reuse = shared > 0 and hasattr(self.backend, "prefix") and len(all_prompts) > 1
        if can_reuse:
            handle, shared = self._prefix_handle(all_prompts[0][:shared])
            n_labels = max(n for _, _, _, n, _, _ in plans)
            logits_rows = self.backend.suffix_logits(handle, [ids[shared:] for ids in all_prompts], self.labels[:n_labels])
        else:
            logits_rows = [np.asarray(self.backend.last_token_logits(ids, self.labels[: max(n for _, _, _, n, _, _ in plans)])) for ids in all_prompts]
        self.last_prefix_tokens = shared if can_reuse else 0
        out: List[np.ndarray] = []
        cursor = 0
        for qid, q, prompts, n, variants, n_options in plans:
            temperature = self.spec.temperature(q.type, n_options)
            probs = np.zeros(n, dtype=np.float64)
            for variant in range(variants):
                scores = np.asarray(logits_rows[cursor][:n], dtype=np.float64); cursor += 1
                z = (scores - scores.max()) / temperature
                p = np.exp(z); p /= p.sum()
                probs += (p if variant == 0 else p[::-1]) / variants
            p = probs.astype(np.float32)
            if q.type == "noul":
                if self.spec.noul_mode == "rating":
                    expected = float(sum(p[i] * i / (len(p) - 1) for i in range(len(p))))
                    p = np.asarray([1.0 - expected, expected], dtype=np.float32)
                elif self.spec.noul_mode == "true_first":
                    p = p[::-1].copy()   # back to [false, true]
            out.append(p)
        return out

    def _prefix_handle(self, prefix_ids: List[int]):
        """-> (handle, length). A small LRU of computed prefixes: an exact hit, else a cached prefix
        that is a leading part of the wanted one (at least half of it: the state of an earlier request
        about the same document, now followed by a different question), else a fresh computation."""
        key = tuple(prefix_ids)
        cache = self.__dict__.setdefault("_prefix_cache", {})
        if key in cache:
            return cache[key], len(key)
        for cached_key, handle in cache.items():
            n = len(cached_key)
            if 2 * n >= len(key) and key[:n] == cached_key:
                return handle, n
        handle = self.backend.prefix(prefix_ids)
        cache[key] = handle
        while len(cache) > 2:
            old_key = next(iter(cache))
            if old_key == key:
                break
            self.backend.release(cache.pop(old_key))
        return handle, len(key)

    def question_probabilities(self, state: Any, qid: str, q: Any) -> np.ndarray:
        """Probabilities of one question alone (used by tests); the label order of the kind."""
        return self.probabilities(DecisionRequest(state=state, questions={qid: q}))[0]

    def decide(self, state: Any = None, questions: Optional[Dict[str, Any]] = None, *, request: Optional[DecisionRequest] = None, truncate: Optional[bool] = None) -> DecisionResponse:
        if request is None:
            if questions is None:
                raise ValueError("decide() needs questions")
            request = DecisionRequest(state=state, questions=questions)
        _, meta = to_record(request)
        started = time.perf_counter()
        probs = self.probabilities(request)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        answers = to_answers(probs, meta)
        shared = int(getattr(self, "last_prefix_tokens", 0))
        return DecisionResponse(model=request.model or self.model_id, answers=answers,
                                usage=DecisionUsage(output_tokens=answers_token_count(self._encode, answers), state_tokens=shared, state_tokens_used=shared),
                                latency_ms=latency_ms, provider=self.provider, backend=self.backend_name, precision=self.precision)

    def unload(self) -> None:
        self.backend.unload()


__all__ = ["LabelLogitDecisionModel"]
