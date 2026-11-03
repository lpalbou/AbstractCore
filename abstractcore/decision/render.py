"""Rendering of a decision request into the text the model sees, and of the
model's probabilities into typed answers.

The two directions are deliberately small and deterministic:

- ``to_record`` flattens a request into one state string and, per question, an
  instruction string and the option strings. Objects and arrays are rendered as
  labelled text (field names kept as labels) so structured states stay readable.
- ``to_answers`` turns one probability vector per question into the typed
  answer: argmax + confidence (choice), P(true) (noul), expected level +
  confidence (score).

Confidence formulas follow TypeSafe's public reference adapter so answers are
comparable with hosted System One models: choice confidence is the top
probability rescaled from uniform (0) to certainty (1); score confidence is
one minus the expected absolute distance to the mode, normalised by the spread
of a uniform distribution.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

from .types import ChoiceAnswer, DecisionAnswer, DecisionRequest, JSONContent, NoulAnswer, ScoreAnswer


def render(value: JSONContent, indent: int = 0) -> str:
    """Flatten a JSON value into text. Dict keys become labels, lists become bullets."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:\n{render(item, indent + 1)}")
            else:
                lines.append(f"{pad}{key}: {render(item)}")
        return "\n".join(lines)
    return str(value)


def option_text(name: str, description: JSONContent) -> str:
    """An option as the model reads it: ``name`` or ``name: description``."""
    if description is None or description == "":
        return name
    return f"{name}: {render(description)}"


def question_keys(qtype: str, criteria: Any) -> List[str]:
    """The keys a question's probabilities are reported under, in option order."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(request: DecisionRequest) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """-> (record, meta). ``record`` = {"state": str, "questions": [{"instr", "options"}]} for the
    encoder; ``meta`` carries per question what is needed to map probabilities back."""
    questions, meta = [], []
    for name, question in request.questions.items():
        info: Dict[str, Any] = {"id": name, "type": question.type, "keys": question_keys(question.type, question.criteria)}
        if question.type == "noul":
            criteria = question.criteria or {}
            options = [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
        elif question.type == "choice":
            options = [option_text(key, desc) for key, desc in question.criteria.items()]
        else:
            options = [render(level) for level in question.criteria]
            info["legend"] = dict(zip(info["keys"], options))
        questions.append({"instr": render(question.instructions), "options": options})
        meta.append(info)
    return {"state": render(request.state), "questions": questions}, meta


def _normalize(p: Sequence[float]) -> List[float]:
    total = float(sum(p))
    if total <= 0:
        return [1.0 / len(p)] * len(p)
    return [float(x) / total for x in p]


def choice_confidence(p: Sequence[float]) -> float:
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty; 1 for a single option."""
    k = len(p)
    if k == 1:
        return 1.0
    return (max(_normalize(p)) - 1.0 / k) / (1.0 - 1.0 / k)


def score_confidence(p: Sequence[float]) -> float:
    """max(0, 1 - E|level - mode| / D) with D the mean absolute deviation of a uniform
    distribution over the levels: 1 when all mass sits on one level, 0 at uniform."""
    n = len(p)
    if n == 1:
        return 1.0
    probs = _normalize(p)
    mode = max(range(n), key=probs.__getitem__)
    spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    expected_distance = sum(pi * abs(i - mode) for i, pi in enumerate(probs))
    return max(0.0, 1.0 - expected_distance / spread)


def round_prob(x: float) -> float:
    """Four decimals: a rounded 255-option distribution still sums to 1 within 0.02."""
    return round(float(x), 4)


def to_answers(probs: Sequence[Sequence[float]], meta: Sequence[Dict[str, Any]]) -> Dict[str, DecisionAnswer]:
    out: Dict[str, DecisionAnswer] = {}
    for p, info in zip(probs, meta):
        p = [float(x) for x in p]
        if info["type"] == "noul":
            out[info["id"]] = NoulAnswer(noul=round_prob(p[1]))
        elif info["type"] == "choice":
            best = max(range(len(p)), key=p.__getitem__)
            out[info["id"]] = ChoiceAnswer(
                choice=info["keys"][best],
                confidence=round_prob(choice_confidence(p)),
                probabilities={k: round_prob(v) for k, v in zip(info["keys"], p)},
            )
        else:
            expected = sum(i * pi for i, pi in enumerate(p))
            out[info["id"]] = ScoreAnswer(
                score=round_prob(expected),
                confidence=round_prob(score_confidence(p)),
                legend=info["legend"],
                probabilities={str(i): round_prob(v) for i, v in enumerate(p)},
            )
    return out


def answers_token_count(encode_text, answers: Dict[str, DecisionAnswer]) -> int:
    """Billing-style ``output_tokens``: the tokens of the serialised answers. Nothing is generated."""
    payload = json.dumps({name: answer.model_dump() for name, answer in answers.items()})
    return len(encode_text(payload))


__all__ = [
    "render",
    "option_text",
    "question_keys",
    "to_record",
    "choice_confidence",
    "score_confidence",
    "round_prob",
    "to_answers",
    "answers_token_count",
]
