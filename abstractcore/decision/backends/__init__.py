"""Decision backends: each turns the encoded rows of a request into one logit
vector per question. The pointer head runs in numpy on every backend; the
backbone runs on transformers (``hf``), mlx-lm (``mlx``) or llama.cpp (``gguf``).
"""

from .base import DecisionBackend, PrefixHandle

__all__ = ["DecisionBackend", "PrefixHandle"]
