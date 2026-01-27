"""
Response post-processing helpers driven by architecture formats and model capabilities.

These utilities normalize model output across providers (local runtimes, OpenAI-compatible
servers, etc.) based on `assets/architecture_formats.json` and `assets/model_capabilities.json`.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple


def _coerce_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s or None


def strip_output_wrappers(
    text: str,
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
) -> str:
    """Strip known model-specific wrapper tokens around assistant output.

    Some model/server combinations emit wrapper tokens like:
      <|begin_of_box|> ... <|end_of_box|>
    We remove these only when they appear as leading/trailing wrappers (not when
    embedded mid-text).
    """
    if not isinstance(text, str) or not text:
        return text

    # Architecture defaults first, model-specific overrides last.
    start_token: Optional[str] = None
    end_token: Optional[str] = None
    for src in (architecture_format, model_capabilities):
        if not isinstance(src, Mapping):
            continue
        wrappers = src.get("output_wrappers")
        if not isinstance(wrappers, Mapping):
            continue
        start = _coerce_str(wrappers.get("start"))
        end = _coerce_str(wrappers.get("end"))
        if start is not None:
            start_token = start
        if end is not None:
            end_token = end

    if start_token is None and end_token is None:
        return text

    out = text
    if start_token:
        out = re.sub(r"^\s*" + re.escape(start_token) + r"\s*", "", out, count=1)
    if end_token:
        out = re.sub(r"\s*" + re.escape(end_token) + r"\s*$", "", out, count=1)

    return out


def split_harmony_response_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort split of OpenAI Harmony-style transcripts into (final, reasoning).

    Expected shape (common in GPT-OSS):
      <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...<|end|>
    """
    if not isinstance(text, str) or not text:
        return None, None

    final_marker = "<|channel|>final"
    msg_marker = "<|message|>"
    end_marker = "<|end|>"
    start_marker = "<|start|>"

    idx_final = text.rfind(final_marker)

    # Extract analysis reasoning if present (even if final is truncated/missing).
    reasoning_text: Optional[str] = None
    idx_analysis = text.find("<|channel|>analysis")
    if idx_analysis != -1:
        idx_analysis_msg = text.find(msg_marker, idx_analysis)
        if idx_analysis_msg != -1:
            a_start = idx_analysis_msg + len(msg_marker)
            # Prefer explicit end marker; otherwise stop at final marker if present; otherwise consume remainder.
            a_end = text.find(end_marker, a_start)
            if a_end == -1 and idx_final != -1:
                a_end = idx_final
            if a_end == -1:
                a_end = len(text)
            reasoning_raw = text[a_start:a_end]
            reasoning_text = reasoning_raw.strip() if reasoning_raw.strip() else None

    if idx_final == -1:
        return None, reasoning_text

    idx_msg = text.find(msg_marker, idx_final)
    start = (idx_msg + len(msg_marker)) if idx_msg != -1 else (idx_final + len(final_marker))
    final_raw = text[start:]

    # Cut off any trailing transcript tokens.
    cut_points = []
    for marker in (end_marker, start_marker):
        pos = final_raw.find(marker)
        if pos != -1:
            cut_points.append(pos)
    if cut_points:
        final_raw = final_raw[: min(cut_points)]
    final_text = final_raw.strip()

    return final_text, reasoning_text


def should_extract_harmony_final(
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return True when this model is expected to emit Harmony transcripts."""
    msg_fmt = ""
    resp_fmt = ""
    try:
        msg_fmt = str((architecture_format or {}).get("message_format") or "").strip().lower()
    except Exception:
        msg_fmt = ""
    try:
        resp_fmt = str((model_capabilities or {}).get("response_format") or "").strip().lower()
    except Exception:
        resp_fmt = ""
    return msg_fmt == "harmony" or resp_fmt == "harmony"


def maybe_extract_harmony_final_text(
    text: str,
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """If the model emits Harmony transcripts, return (clean_text, reasoning)."""
    if not isinstance(text, str) or not text:
        return text, None

    if not should_extract_harmony_final(
        architecture_format=architecture_format,
        model_capabilities=model_capabilities,
    ):
        return text, None

    final_text, reasoning = split_harmony_response_text(text)

    if final_text is None:
        # If we only got analysis (e.g., truncated before final), strip the wrapper tokens
        # so the caller doesn't see raw Harmony markup.
        if isinstance(reasoning, str) and reasoning.strip() and text.lstrip().startswith("<|channel|>analysis"):
            return reasoning.strip(), reasoning.strip()
        return text, reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None

    return final_text, reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None

