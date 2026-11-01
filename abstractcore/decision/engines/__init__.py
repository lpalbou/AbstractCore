"""Decision engines: everything that can answer a ``DecisionRequest``.

The request/answer types are shared by every engine; what differs is how the
probabilities are produced:

- ``PointerHeadModel`` (``runtime.DecisionModel``): a local backbone plus a pointer
  head (the Kev family) on transformers / mlx-lm / llama.cpp.
- ``LabelLogitModel``: a local chat model whose answer is read from the next-token
  logits of one label token per option (Lev; OpenJev-style checkpoints).
- ``GGUFDecisionGraphModel``: any llama.cpp decision GGUF, driven by the metadata
  and the ``systemone`` chat template the file carries (kev, lev, openjev, laya).
- ``RemoteSystemOneModel``: a hosted System One endpoint (TypeSafe's Jev, a
  compatible host, or another AbstractCore server).

All implement ``DecisionEngine``: ``decide(state, questions, *, request=None,
truncate=None) -> DecisionResponse`` plus ``unload()`` and the identity fields.
"""

from .base import DecisionEngine
from .remote import RemoteSystemOneModel

__all__ = ["DecisionEngine", "RemoteSystemOneModel"]
