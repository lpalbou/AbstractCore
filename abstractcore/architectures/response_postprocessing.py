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


def _get_thinking_tags(
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
) -> Optional[Tuple[str, str]]:
    """Return (start_tag, end_tag) for inline thinking tags when configured."""
    tags: Any = None
    for src in (architecture_format, model_capabilities):
        if not isinstance(src, Mapping):
            continue
        value = src.get("thinking_tags")
        if value is not None:
            tags = value
    if not isinstance(tags, (list, tuple)) or len(tags) != 2:
        return None
    start = _coerce_str(tags[0])
    end = _coerce_str(tags[1])
    if start is None or end is None:
        return None
    return start, end


def strip_thinking_tags(
    text: str,
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Strip inline thinking tags and return (clean_text, reasoning).

    Some models emit reasoning as tagged blocks inside the assistant content, e.g.:
      <think> ... </think>
    When configured via assets, we extract the tagged reasoning and remove it from
    the visible content. This keeps downstream transcripts clean while preserving
    reasoning in metadata.
    """
    if not isinstance(text, str) or not text:
        return text, None

    tags = _get_thinking_tags(
        architecture_format=architecture_format,
        model_capabilities=model_capabilities,
    )
    if tags is None:
        return text, None

    start_tag, end_tag = tags
    # Non-greedy across newlines; allow multiple blocks.
    pattern = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        # Some models (notably Qwen3 Thinking variants) may emit ONLY the closing tag `</think>`
        # with the opening tag provided by the chat template (i.e., not present in decoded text).
        # In that case, treat everything before the first end tag as reasoning.
        if end_tag in text and start_tag not in text:
            before, after = text.split(end_tag, 1)
            reasoning_only = before.strip() or None
            cleaned = after
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            return cleaned, reasoning_only
        return text, None

    extracted: list[str] = []
    for m in matches:
        chunk = (m.group(1) or "").strip()
        if chunk:
            extracted.append(chunk)

    cleaned = pattern.sub("", text)
    # Tidy up: collapse multiple blank lines created by removal.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    reasoning = "\n\n".join(extracted).strip() if extracted else None
    return cleaned, reasoning or None


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


def extract_reasoning_from_message(
    message: Mapping[str, Any],
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Extract reasoning from a provider message dict when present.

    Supported keys:
    - `reasoning` (OpenAI-compatible reasoning outputs)
    - `thinking`  (Ollama thinking outputs)
    - `thinking_output_field` from assets (e.g., `reasoning_content` for some GLM models)
    """
    if not isinstance(message, Mapping):
        return None

    for key in ("reasoning", "thinking"):
        v = message.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    thinking_output_field: Optional[str] = None
    for src in (architecture_format, model_capabilities):
        if not isinstance(src, Mapping):
            continue
        field = _coerce_str(src.get("thinking_output_field"))
        if field is not None:
            thinking_output_field = field

    if thinking_output_field:
        v = message.get(thinking_output_field)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def normalize_assistant_text(
    text: str,
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Normalize provider output into (clean_text, reasoning).

    Order:
    1) Strip wrapper tokens (e.g., GLM box wrappers)
    2) Extract Harmony final (GPT-OSS) into final text + reasoning
    3) Extract inline <think>...</think> blocks when configured
    """
    if not isinstance(text, str) or not text:
        return text, None

    cleaned = strip_output_wrappers(
        text,
        architecture_format=architecture_format,
        model_capabilities=model_capabilities,
    )
    cleaned, reasoning_harmony = maybe_extract_harmony_final_text(
        cleaned,
        architecture_format=architecture_format,
        model_capabilities=model_capabilities,
    )
    cleaned, reasoning_tags = strip_thinking_tags(
        cleaned,
        architecture_format=architecture_format,
        model_capabilities=model_capabilities,
    )

    parts = [r for r in (reasoning_harmony, reasoning_tags) if isinstance(r, str) and r.strip()]
    reasoning: Optional[str] = None
    if parts:
        reasoning = "\n\n".join(parts).strip() or None
    return cleaned, reasoning
