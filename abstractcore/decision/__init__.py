"""Decision models: text in, calibrated decisions out.

A decision model (Kev, Jev-like) reads a document and answers typed questions
about it — ``choice`` over named options, ``noul`` (yes/no probability),
``score`` over ordered levels — with calibrated probabilities, in one forward
pass and without generating text. In AbstractCore it is the ``decision``
output modality: text input, decision output.

    from abstractcore import create_llm
    llm = create_llm("mlx", model="AbstractFramework/kev-4b-mlx-4bit")
    result = llm.generate(ticket_text, output="decision", questions={
        "department": {"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": None, "shipping": None, "returns": None}},
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    })
    result.decisions["department"].choice        # "returns"
    result.decisions["escalate"].noul            # 0.93

Or without a provider object: ``load_decision_model("mlx", "...")`` returns a
``DecisionModel`` whose ``decide(state, questions)`` gives a ``DecisionResponse``
in the public System One shape (``POST /v1/decisions``).
"""

from .encoding import DELIMITERS, MAX_ROW_TOKENS, MAX_STATE_TOKENS, ContextOverflow
from .head import PointerHead
from .loader import (
    DecisionModelInfo,
    describe_decision_model,
    is_decision_model,
    load_decision_model,
)
from .pool import decision_provider
from .runtime import DecisionModel
from .types import (
    Choice,
    ChoiceAnswer,
    DecisionAnswer,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    coerce_question,
)

__all__ = [
    "DELIMITERS",
    "MAX_ROW_TOKENS",
    "MAX_STATE_TOKENS",
    "ContextOverflow",
    "PointerHead",
    "DecisionModel",
    "DecisionModelInfo",
    "describe_decision_model",
    "is_decision_model",
    "load_decision_model",
    "decision_provider",
    "Choice",
    "ChoiceAnswer",
    "DecisionAnswer",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionUsage",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "coerce_question",
]
