from __future__ import annotations

from typing import Any, List, Optional, Protocol, Sequence

import numpy as np

from ..encoding import EncodedQuestion


class PrefixHandle(Protocol):
    """An opaque computed state prefix a backend can reuse for more questions."""

    state_ids: List[int]


class DecisionBackend(Protocol):
    name: str            # "hf" | "mlx" | "gguf"
    precision: str       # e.g. "bf16", "fp32", "4bit", "Q4_K_M"
    model_path: str

    def prefix(self, state_ids: Sequence[int]) -> Any:
        """Run the state tokens once; return a handle ``question_logits`` accepts."""

    def question_logits(self, prefix: Any, questions: Sequence[EncodedQuestion]) -> List[np.ndarray]:
        """Calibrated logits (one [K] vector per question) for branches continuing ``prefix``.
        The prefix must stay reusable afterwards."""

    def release(self, prefix: Any) -> None:
        """Free whatever ``prefix`` holds (a no-op where the handle owns nothing)."""

    def unload(self) -> None:
        """Free the backbone."""
