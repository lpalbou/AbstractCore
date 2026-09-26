"""
Response post-processing helpers driven by architecture formats and model capabilities.

These utilities normalize model output across providers (local runtimes, OpenAI-compatible
servers, etc.) based on `assets/architecture_formats.json` and `assets/model_capabilities.json`.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple

from ..utils.structured_logging import get_logger

_logger = get_logger(__name__)

# Marker appended to reasoning captured from an unterminated thinking block
# (e.g. the stream/response was truncated before the closing tag arrived).
TRUNCATED_REASONING_MARKER = " (...)"

# Stream-chunk metadata flag: the provider's rendered prompt ENDED with the
# thinking start tag (the chat template opened the block), so the first
# generated tokens are reasoning. BaseProvider consumes it (never forwarded).
THINKING_OPENED_BY_PROMPT = "_thinking_opened_by_prompt"


def prompt_opens_thinking(
    prompt: Any,
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when a rendered prompt ends inside an opened thinking block."""
    if not isinstance(prompt, str) or not prompt:
        return False
    tags = _get_thinking_tags(architecture_format=architecture_format, model_capabilities=model_capabilities)
    if tags is None:
        return False
    start_tag, end_tag = tags
    tail = prompt.rstrip()
    return tail.endswith(start_tag)


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
    opened_by_prompt: bool = False,
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
    if opened_by_prompt:
        # The rendered prompt ended inside an opened block (`prompt_opens_thinking`):
        # the reply BEGINS as reasoning. Restore the opening the template wrote
        # (dropping a start tag the model repeated), so a truncated reply is a
        # truncated block and a later block cannot pair with the first closing tag.
        body = text.lstrip()
        if body.startswith(start_tag):
            body = body[len(start_tag):]
        text = start_tag + body
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
        if start_tag in text and end_tag not in text:
            # Unterminated thinking block (e.g. finish_reason=length truncated the response
            # before the closing tag): auto-close it and capture the block as reasoning
            # instead of leaking it into visible content.
            before, after = text.split(start_tag, 1)
            truncated = after.strip()
            cleaned = re.sub(r"\n{3,}", "\n\n", before).strip()
            if truncated:
                _logger.warning(
                    f"#TRUNCATION: unterminated thinking block auto-closed; captured {len(truncated)} chars as reasoning"
                )
                return cleaned, truncated + TRUNCATED_REASONING_MARKER
            return cleaned, None
        return text, None

    extracted: list[str] = []
    for m in matches:
        chunk = (m.group(1) or "").strip()
        if chunk:
            extracted.append(chunk)

    cleaned = pattern.sub("", text)

    # A trailing unterminated block can follow complete pairs; auto-close it as well.
    if start_tag in cleaned and end_tag not in cleaned.split(start_tag, 1)[1]:
        before, after = cleaned.split(start_tag, 1)
        truncated = after.strip()
        cleaned = before
        if truncated:
            _logger.warning(
                f"#TRUNCATION: unterminated thinking block auto-closed; captured {len(truncated)} chars as reasoning"
            )
            extracted.append(truncated + TRUNCATED_REASONING_MARKER)

    # Tidy up: collapse multiple blank lines created by removal.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    reasoning = "\n\n".join(extracted).strip() if extracted else None
    return cleaned, reasoning or None


class IncrementalThinkingTagStripper:
    """Incrementally strip inline thinking tags from a streamed assistant response.

    This mirrors `strip_thinking_tags()` semantics as closely as possible while handling
    tag boundaries that may be split across stream chunks.

    Notes:
    - Only strips when a complete start/end pair is observed.
    - Also supports the "closing-only" case (end tag appears, start tag absent) by
      treating everything before the first end tag as reasoning (the start tag may
      have been injected by the chat template). This requires buffering until a tag
      is seen; when `assume_visible_start=True` (e.g. thinking was effectively
      disabled for the request), the stripper starts in the "visible" state and
      streams content immediately instead of withholding it for closing-only capture.
    - If a start tag is observed but the end tag never appears (truncated stream),
      the block is auto-closed and captured as reasoning with a truncation marker,
      matching `strip_thinking_tags()` behavior (#TRUNCATION labeled).
    """

    def __init__(self, *, start_tag: str, end_tag: str, assume_visible_start: bool = False) -> None:
        if not isinstance(start_tag, str) or not start_tag:
            raise ValueError("start_tag must be a non-empty string")
        if not isinstance(end_tag, str) or not end_tag:
            raise ValueError("end_tag must be a non-empty string")

        self._start_tag = start_tag
        self._end_tag = end_tag

        self._buffer = ""
        self._state: str = "visible" if assume_visible_start else "searching"  # searching|visible|thinking
        self._current_reasoning_parts: list[str] = []
        self._reasoning_blocks: list[str] = []
        # Reasoning identified since the last `take_reasoning_delta()`: streamed
        # to the caller as it is generated, not only as the final aggregate.
        self._pending_delta: list[str] = []
        # The ANSWER starts without the blank lines models put after the closing
        # tag, as the non-streamed `strip_thinking_tags` (which strips the whole
        # text) returns it. Only the answer's start: once visible text has been
        # emitted, whitespace after a later block is part of the answer.
        self._strip_leading_visible = False
        self._emitted_visible = False
        # Once a block was removed, the non-streamed split also collapses blank
        # line runs (3+ newlines -> 2), including a run that spans the removed
        # block; `_trailing_newlines` counts the run the emitted text ends with.
        self._seen_block = False
        self._trailing_newlines = 0
        # Prompt-opened block: a model that writes the start tag again anyway
        # must not leave the literal tag in the reasoning.
        self._drop_leading_start_tag = False

    def open_thinking(self) -> None:
        """The prompt already opened the thinking block (the chat template ended
        with the start tag): what the model generates first IS reasoning.

        Without this, a response that only carries the CLOSING tag is withheld
        until that tag arrives (the "closing-only" case below), so a model that
        thinks for minutes streams nothing for minutes. Only valid before any
        text was processed."""
        if self._state == "searching" and not self._buffer:
            self._state = "thinking"
            self._seen_block = True
            self._current_reasoning_parts = []
            self._drop_leading_start_tag = True

    def take_reasoning_delta(self) -> str:
        """Reasoning text identified since the previous call (may be empty)."""
        delta = "".join(self._pending_delta)
        self._pending_delta = []
        return delta

    def _add_reasoning(self, text: str) -> None:
        if text:
            self._current_reasoning_parts.append(text)
            self._pending_delta.append(text)

    def _add_visible(self, out_parts: list, text: str) -> None:
        if self._strip_leading_visible:
            text = text.lstrip()
            if text:
                self._strip_leading_visible = False
        if not text:
            return
        if self._seen_block:
            lead = len(text) - len(text.lstrip("\n"))
            allowed = max(0, 2 - self._trailing_newlines)
            if lead > allowed:
                text = text[lead - allowed :]
            text = re.sub(r"\n{3,}", "\n\n", text)
            if not text:
                return
        out_parts.append(text)
        body = text.rstrip("\n")
        tail_newlines = len(text) - len(body)
        self._trailing_newlines = tail_newlines if body else self._trailing_newlines + tail_newlines
        if text.strip():
            self._emitted_visible = True

    @staticmethod
    def _suffix_prefix_len(haystack: str, needle: str) -> int:
        """Return length of longest suffix of `haystack` that is a prefix of `needle`."""
        if not haystack or not needle:
            return 0
        max_len = min(len(haystack), max(0, len(needle) - 1))
        for n in range(max_len, 0, -1):
            if needle.startswith(haystack[-n:]):
                return n
        return 0

    def _finalize_current_reasoning(self) -> None:
        raw = "".join(self._current_reasoning_parts)
        self._current_reasoning_parts = []
        chunk = raw.strip()
        if chunk:
            self._reasoning_blocks.append(chunk)

    def process(self, text: str) -> str:
        """Process a streamed text chunk and return visible content."""
        if not isinstance(text, str) or not text:
            return ""

        self._buffer += text
        out_parts: list[str] = []

        while self._buffer:
            if self._state == "searching":
                start_idx = self._buffer.find(self._start_tag)
                end_idx = self._buffer.find(self._end_tag)
                if start_idx == -1 and end_idx == -1:
                    # Ambiguous prefix: we might be in the "closing-only" mode where the
                    # start tag is injected by the template. Buffer until we see a tag.
                    break

                candidates: list[tuple[int, str]] = []
                if start_idx != -1:
                    candidates.append((start_idx, "start"))
                if end_idx != -1:
                    candidates.append((end_idx, "end"))
                idx, kind = min(candidates)

                if kind == "start":
                    prefix = self._buffer[:idx]
                    if prefix:
                        self._add_visible(out_parts, prefix)
                    self._buffer = self._buffer[idx + len(self._start_tag) :]
                    self._state = "thinking"
                    self._seen_block = True
                    self._current_reasoning_parts = []
                    continue

                # end tag before any explicit start tag -> closing-only case
                reasoning_prefix = self._buffer[:idx]
                self._seen_block = True
                self._current_reasoning_parts = []
                self._add_reasoning(reasoning_prefix)
                self._buffer = self._buffer[idx + len(self._end_tag) :]
                self._finalize_current_reasoning()
                self._state = "visible"
                self._strip_leading_visible = not self._emitted_visible
                continue

            if self._state == "visible":
                start_idx = self._buffer.find(self._start_tag)
                if start_idx == -1:
                    keep = self._suffix_prefix_len(self._buffer, self._start_tag)
                    if keep:
                        self._add_visible(out_parts, self._buffer[:-keep])
                        self._buffer = self._buffer[-keep:]
                    else:
                        self._add_visible(out_parts, self._buffer)
                        self._buffer = ""
                    break

                prefix = self._buffer[:start_idx]
                if prefix:
                    self._add_visible(out_parts, prefix)
                self._buffer = self._buffer[start_idx + len(self._start_tag) :]
                self._state = "thinking"
                self._seen_block = True
                self._current_reasoning_parts = []
                continue

            if self._state == "thinking":
                if self._drop_leading_start_tag:
                    head = self._buffer.lstrip()
                    if not head or (len(head) < len(self._start_tag) and self._start_tag.startswith(head)):
                        break  # undecided: whitespace, or the start of a repeated start tag
                    self._drop_leading_start_tag = False
                    if head.startswith(self._start_tag):
                        self._buffer = head[len(self._start_tag) :]
                        continue
                end_idx = self._buffer.find(self._end_tag)
                if end_idx == -1:
                    keep = self._suffix_prefix_len(self._buffer, self._end_tag)
                    if keep:
                        self._add_reasoning(self._buffer[:-keep])
                        self._buffer = self._buffer[-keep:]
                    else:
                        self._add_reasoning(self._buffer)
                        self._buffer = ""
                    break

                self._add_reasoning(self._buffer[:end_idx])
                self._buffer = self._buffer[end_idx + len(self._end_tag) :]
                self._finalize_current_reasoning()
                self._state = "visible"
                self._strip_leading_visible = not self._emitted_visible
                continue

            break

        return "".join(out_parts)

    def finalize(self) -> tuple[str, Optional[str]]:
        """Return (visible_tail, reasoning) after the stream ends."""
        visible_tail = ""

        if self._state == "thinking":
            # Unterminated thinking block (stream ended before the closing tag): auto-close
            # and capture the block as reasoning instead of leaking it into visible content.
            truncated = ("".join(self._current_reasoning_parts) + self._buffer).strip()
            self._current_reasoning_parts = []
            self._buffer = ""
            if truncated:
                _logger.warning(
                    f"#TRUNCATION: unterminated thinking block auto-closed; captured {len(truncated)} chars as reasoning"
                )
                self._reasoning_blocks.append(truncated + TRUNCATED_REASONING_MARKER)
        else:
            # searching/visible: emit any remaining buffered visible content.
            parts: list[str] = []
            self._add_visible(parts, self._buffer)
            self._buffer = ""
            visible_tail = "".join(parts)

        reasoning = "\n\n".join(self._reasoning_blocks).strip() if self._reasoning_blocks else None
        return visible_tail, reasoning or None


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

    # Cut off any trailing transcript tokens. `<|return|>` / `<|call|>` end a
    # message too: backends normally stop on them without emitting them (they
    # are EOS ids in GPT-OSS's generation config), but a backend that does
    # emit them must not put them in the answer.
    cut_points = []
    for marker in (end_marker, start_marker, "<|return|>", "<|call|>"):
        pos = final_raw.find(marker)
        if pos != -1:
            cut_points.append(pos)
    if cut_points:
        final_raw = final_raw[: min(cut_points)]
    final_text = final_raw.strip()

    return final_text, reasoning_text


def maybe_create_incremental_thinking_tag_stripper(
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
    assume_visible_start: bool = False,
) -> Optional[IncrementalThinkingTagStripper]:
    """Return an incremental thinking-tag stripper when configured via assets.

    Set ``assume_visible_start=True`` when thinking is effectively disabled for the
    request: the stripper then streams visible content immediately instead of buffering
    for the "closing-only" reasoning-first case (which cannot occur with thinking off).
    """
    tags = _get_thinking_tags(
        architecture_format=architecture_format,
        model_capabilities=model_capabilities,
    )
    if tags is None:
        return None
    start_tag, end_tag = tags
    try:
        return IncrementalThinkingTagStripper(
            start_tag=start_tag,
            end_tag=end_tag,
            assume_visible_start=assume_visible_start,
        )
    except Exception:
        return None


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


def _harmony_without_final(text: str, reasoning: Optional[str]) -> Tuple[str, Optional[str]]:
    """A Harmony transcript with no `final` message: what the answer is, and the reasoning.

    - Not Harmony at all (no framing): the text, unchanged.
    - A tool message (`to=<recipient>`): the transcript minus its `analysis`
      messages, so the tool-call parser still sees the call; the analysis is
      reasoning.
    - Otherwise (typically cut by the output budget before `final`): the
      analysis is REASONING, never the answer. The answer is only what the
      model addressed to the user outside `analysis` (a recipient-less
      `commentary` preamble, bare text) -- usually nothing. An `analysis`
      message left open is marked truncated like an unterminated think block.
      The streamed path (`providers.streaming.IncrementalHarmonySplitter`)
      produces the same text and reasoning for the same transcript.
    """
    if "<|channel|>" not in text and "<|start|>" not in text:
        return text, (reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None)

    if re.search(r"\bto=\S", text):
        without_analysis = re.sub(
            r"(?:<\|start\|>[^<]*)?<\|channel\|>\s*analysis\b.*?(?:<\|end\|>|$)",
            "",
            text,
            flags=re.DOTALL,
        )
        return without_analysis, (reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None)

    from ..providers.streaming import IncrementalHarmonySplitter

    splitter = IncrementalHarmonySplitter()
    events = splitter.feed(text) + splitter.finish()
    content = "".join(v for k, v in events if k == "content").strip()
    thought = "".join(v for k, v in events if k == "reasoning").strip()
    return content, (thought or None)


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
        return _harmony_without_final(text, reasoning)

    return final_text, reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None


def extract_reasoning_from_message(
    message: Mapping[str, Any],
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
    strip: bool = True,
) -> Optional[str]:
    """Extract reasoning from a provider message dict when present.

    Supported keys:
    - `reasoning` (OpenAI-compatible reasoning outputs)
    - `reasoning_content` (some OpenAI-compatible servers)
    - `thinking`  (Ollama thinking outputs)
    - `thinking_output_field` from assets (e.g., `reasoning_content` for some GLM models)

    `strip=False` preserves the value verbatim (whitespace included). Streaming DELTAS
    must be extracted verbatim: stripping each fragment corrupts word boundaries when
    the fragments are later joined into the complete reasoning text.
    """
    if not isinstance(message, Mapping):
        return None

    def _result(v: str) -> Optional[str]:
        if not v.strip():
            return None
        return v.strip() if strip else v

    for key in ("reasoning", "reasoning_content", "thinking"):
        v = message.get(key)
        if isinstance(v, str) and v.strip():
            return _result(v)

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
            return _result(v)

    return None


def normalize_assistant_text(
    text: str,
    *,
    architecture_format: Optional[Mapping[str, Any]] = None,
    model_capabilities: Optional[Mapping[str, Any]] = None,
    thinking_opened_by_prompt: bool = False,
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
        opened_by_prompt=thinking_opened_by_prompt,
    )

    parts = [r for r in (reasoning_harmony, reasoning_tags) if isinstance(r, str) and r.strip()]
    reasoning: Optional[str] = None
    if parts:
        reasoning = "\n\n".join(parts).strip() or None
    return cleaned, reasoning
