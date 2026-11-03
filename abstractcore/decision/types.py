"""Typed questions and answers of a decision model.

A decision model reads one document (the *state*) and answers typed questions
about it with a calibrated probability distribution over the options the caller
supplies. It never generates text. Three question types exist:

- ``choice``: one of 1..255 named options, each with an optional description.
  The answer is the most likely option, its probability distribution, and a
  confidence derived from that distribution.
- ``noul``: a yes/no statement. The answer is the probability that the statement
  is true (``noul`` comes from the TypeSafe System One API this mirrors).
- ``score``: 1..255 ordered levels, lowest first. The answer is the expected
  level index, the per-level distribution, the legend, and a confidence.

The request and answer shapes are wire-compatible with the public System One
API (``POST /v1/decisions``) so a request written for that API is a request
for AbstractCore's ``/v1/decisions`` endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, Dict[str, Any], List[Any], int, float, bool, None]

MAX_OPTIONS = 255


class Noul(BaseModel):
    """A yes/no question: the answer is P(the statement in ``instructions`` is true)."""

    type: Literal["noul"] = "noul"
    instructions: JSONContent = None
    # Optional descriptions of what "true" and "false" mean for this question.
    criteria: Optional[Dict[str, JSONContent]] = None


class Choice(BaseModel):
    """A categorical question over named options (``criteria`` maps option name to an optional description)."""

    type: Literal["choice"] = "choice"
    instructions: JSONContent = None
    criteria: Dict[str, JSONContent]

    @model_validator(mode="after")
    def _check_options(self) -> "Choice":
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"a choice question needs 1..{MAX_OPTIONS} options, got {len(self.criteria)}")
        return self


class Score(BaseModel):
    """An ordinal question over ordered levels (``criteria`` lists the levels, lowest first)."""

    type: Literal["score"] = "score"
    instructions: JSONContent = None
    criteria: List[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


def coerce_question(value: Any) -> Question:
    """Accept a typed question, a System One dict, or AbstractCore shorthand.

    Shorthand forms (convenient from Python, never on the wire):

    - ``"Is the customer angry?"`` -> ``Noul(instructions=...)``
    - ``["billing", "shipping"]`` -> ``Choice`` with those options and no descriptions
    - ``{"instructions": ..., "options": [...] | {...}}`` -> ``Choice``
    - ``{"instructions": ..., "levels": [...]}`` -> ``Score``
    """
    if isinstance(value, (Noul, Choice, Score)):
        return value
    if isinstance(value, str):
        return Noul(instructions=value)
    if isinstance(value, (list, tuple)):
        return Choice(criteria={str(option): None for option in value})
    if isinstance(value, dict):
        data = dict(value)
        if "type" not in data:
            if "levels" in data:
                data = {"type": "score", "instructions": data.get("instructions"), "criteria": list(data["levels"])}
            elif "options" in data:
                options = data["options"]
                criteria = {str(o): None for o in options} if isinstance(options, (list, tuple)) else dict(options)
                data = {"type": "choice", "instructions": data.get("instructions"), "criteria": criteria}
            elif "criteria" in data:
                criteria = data["criteria"]
                data["type"] = "score" if isinstance(criteria, list) else "choice"
            else:
                data["type"] = "noul"
        qtype = data.get("type")
        if qtype == "noul":
            return Noul.model_validate(data)
        if qtype == "choice":
            return Choice.model_validate(data)
        if qtype == "score":
            return Score.model_validate(data)
        raise ValueError(f"unknown question type {qtype!r}; expected noul, choice or score")
    raise ValueError(f"cannot interpret {type(value).__name__} as a decision question")


class DecisionRequest(BaseModel):
    """One state and the named questions to answer about it (System One request shape)."""

    state: JSONContent
    questions: Dict[str, Question] = Field(min_length=1)
    model: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("questions"), dict):
            data = dict(data)
            data["questions"] = {str(k): coerce_question(v) for k, v in data["questions"].items()}
        return data


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float  # probability that the statement is true


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: Dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float  # expected level index
    confidence: float
    legend: Dict[str, str]
    probabilities: Dict[str, float]


DecisionAnswer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class DecisionUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    state_tokens: int = 0          # tokens of the state as sent (delimiter included)
    state_tokens_used: int = 0     # tokens the model read (less only when truncated)
    prefix_cache_hit: bool = False


class DecisionResponse(BaseModel):
    """Answers keyed by question name, plus usage and timing (System One response shape)."""

    model: str
    answers: Dict[str, DecisionAnswer]
    usage: DecisionUsage = Field(default_factory=DecisionUsage)
    latency_ms: float = 0.0
    truncated: bool = False
    # AbstractCore additions (not on the System One wire): where the answer came from.
    provider: Optional[str] = None
    backend: Optional[str] = None
    precision: Optional[str] = None

    def __getitem__(self, name: str) -> DecisionAnswer:
        return self.answers[name]

    def to_systemone(self) -> Dict[str, Any]:
        """The exact System One response body."""
        model = self.model
        if self.provider and not model.startswith(f"{self.provider}/"):
            model = f"{self.provider}/{model}"
        body = {
            "model": model,
            "answers": {name: answer.model_dump() for name, answer in self.answers.items()},
            "usage": {"input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens},
            "latency_ms": self.latency_ms,
        }
        if self.truncated:
            body["truncated"] = True
            body["usage"]["state_tokens"] = self.usage.state_tokens
            body["usage"]["state_tokens_used"] = self.usage.state_tokens_used
        return body


__all__ = [
    "JSONContent",
    "MAX_OPTIONS",
    "Noul",
    "Choice",
    "Score",
    "Question",
    "coerce_question",
    "DecisionRequest",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "DecisionAnswer",
    "DecisionUsage",
    "DecisionResponse",
]
