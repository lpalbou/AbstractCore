"""Tokenizer + chat-template identity for KV-artifact validity (backlog 0817).

A saved KV artifact encodes the TOKEN-ID stream a text produced under one
specific tokenizer + chat template. A `tokenizer.json` / `chat_template`
refresh under the SAME model id changes the text-to-ids mapping, so a reused
artifact holds KV for positions the current tokenizer would never produce —
wrong output with no error anywhere (the silently-wrong-cache class). The
existing `rendered_recipe_sha256` hashes rendered TEXT and cannot see this;
the fingerprint here identifies the tokenizer STATE, which fully determines
text-to-ids.

Fingerprint subject, strongest available first:
1. The fast (Rust) tokenizer's complete serialized state (`to_str()` —
   vocab, merges, normalizers, added tokens: everything that maps text to
   ids), plus the chat template text (it lives OUTSIDE that state, in
   tokenizer_config, and is the most frequently refreshed piece), plus the
   special-token ids.
2. Without a fast backend: vocab size + sorted added/special tokens + chat
   template — weaker, still catches template refreshes and vocab growth;
   the tier is encoded in the fingerprint prefix so equal values always
   compare the same subjects.
3. No tokenizer available: "" (unverifiable — the validator must treat it
   as "cannot verify", never as "matches").
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

__all__ = ["tokenizer_fingerprint_for", "check_tokenizer_fingerprint"]


def _chat_template_text(tokenizer: Any) -> str:
    for source in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
        if source is None:
            continue
        template = getattr(source, "chat_template", None)
        if isinstance(template, str) and template.strip():
            return template
        # Multi-template models expose a dict of named templates.
        if isinstance(template, dict) and template:
            try:
                return "\x00".join(f"{k}\x01{v}" for k, v in sorted(template.items()))
            except Exception:
                continue
    return ""


def _special_token_ids(tokenizer: Any) -> str:
    parts = []
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value: Any = None
        for source in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
            if source is None:
                continue
            value = getattr(source, name, None)
            if value is not None:
                break
        parts.append(f"{name}={value}")
    return "|".join(parts)


def _fast_backend_state(tokenizer: Any) -> Optional[str]:
    """The complete serialized state of a HF fast tokenizer, when present.

    Duck-typed across the shapes in this codebase: a plain transformers fast
    tokenizer (`backend_tokenizer.to_str()`), mlx_lm's TokenizerWrapper (the
    HF tokenizer rides `_tokenizer`), and a raw `tokenizers.Tokenizer`
    (`to_str()` directly).
    """
    candidates = (
        getattr(tokenizer, "backend_tokenizer", None),
        getattr(getattr(tokenizer, "_tokenizer", None), "backend_tokenizer", None),
        tokenizer if not hasattr(tokenizer, "backend_tokenizer") else None,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        to_str = getattr(candidate, "to_str", None)
        if not callable(to_str):
            continue
        try:
            state = to_str()
        except Exception:
            continue
        if isinstance(state, str) and state:
            return state
    return None


def _vocab_summary(tokenizer: Any) -> Optional[str]:
    """Weaker tier for slow tokenizers: size + sorted added/special tokens."""
    for source in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
        if source is None:
            continue
        get_vocab = getattr(source, "get_vocab", None)
        if not callable(get_vocab):
            continue
        try:
            vocab = get_vocab()
        except Exception:
            continue
        if not isinstance(vocab, dict) or not vocab:
            continue
        added = ""
        get_added = getattr(source, "get_added_vocab", None)
        if callable(get_added):
            try:
                added_vocab = get_added()
                if isinstance(added_vocab, dict):
                    added = ",".join(f"{k}:{v}" for k, v in sorted(added_vocab.items()))
            except Exception:
                added = ""
        return f"vocab_size={len(vocab)}|added={added}"
    return None


def tokenizer_fingerprint_for(tokenizer: Any) -> str:
    """Return a stable fingerprint of a tokenizer's text-to-ids identity.

    "" when no usable tokenizer state is reachable (validators must treat
    that as unverifiable, never as a match).
    """
    if tokenizer is None:
        return ""
    template = _chat_template_text(tokenizer)
    specials = _special_token_ids(tokenizer)

    state = _fast_backend_state(tokenizer)
    if state is not None:
        tier = "full"
        subject = state
    else:
        subject = _vocab_summary(tokenizer) or ""
        if not subject:
            return ""
        tier = "vocab"

    digest = hashlib.sha256()
    for part in (subject, template, specials):
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return f"tokenizer-{tier}:sha256:{digest.hexdigest()[:24]}"


def check_tokenizer_fingerprint(stored: Any, current: Any) -> str:
    """Three-way verdict shared by every gate that consumes the fingerprint.

    - "mismatch": both known and different — the artifact's token stream can
      no longer be trusted; refuse (recompile or raise), never reload.
    - "unverified_stored": the artifact predates this axis (no recorded
      fingerprint) — reuse is allowed but must be LABELED, never silent.
    - "unverified_current": the current tokenizer state is unavailable
      (e.g. model not loaded at validation time) — comparison abstains;
      a later gate that has the tokenizer must re-check.
    - "ok": both known and equal.

    Fingerprints of DIFFERENT tiers (full vs vocab) for the same tokenizer
    are not comparable subject-for-subject; the tier prefix makes them
    unequal, which reads as a mismatch — the fail-safe direction (recompile
    once, record the stronger tier).
    """
    stored_text = str(stored or "").strip()
    current_text = str(current or "").strip()
    if not stored_text:
        return "unverified_stored"
    if not current_text:
        return "unverified_current"
    return "ok" if stored_text == current_text else "mismatch"
