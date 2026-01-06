"""Relaxed JSON/Python-literal parsing helpers.

Some models emit tool-call JSON that is "almost JSON" (e.g., Python booleans,
single quotes) or includes unescaped control characters (notably literal
newlines inside string values). These helpers provide a single, robust way to
parse dict-like payloads across AbstractCore.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape literal control chars that appear inside quoted strings.

    This turns invalid JSON like:
        {"content":"line1
        line2"}
    into valid JSON:
        {"content":"line1\\nline2"}

    Works for both single- and double-quoted strings (for Python-literal fallbacks).
    """
    if not text:
        return text

    out: list[str] = []
    in_string = False
    quote = ""
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == quote:
                out.append(ch)
                in_string = False
                quote = ""
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue

        if ch in ("'", '"'):
            in_string = True
            quote = ch
            out.append(ch)
            continue

        out.append(ch)

    return "".join(out)


def loads_dict_like(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON-ish or Python-literal dict safely."""
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        fixed = _escape_control_chars_in_strings(text)
        if fixed != text:
            try:
                value = json.loads(fixed)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    except Exception:
        pass

    candidate = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
    candidate = re.sub(r"\bfalse\b", "False", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bnull\b", "None", candidate, flags=re.IGNORECASE)
    candidate = _escape_control_chars_in_strings(candidate)
    try:
        value = ast.literal_eval(candidate)
    except Exception:
        return None

    if not isinstance(value, dict):
        return None
    return {str(k): v for k, v in value.items()}

