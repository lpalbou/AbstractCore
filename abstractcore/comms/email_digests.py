from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_sections(sections: Any) -> List[Dict[str, Any]]:
    if sections is None:
        return []
    if isinstance(sections, dict):
        return [dict(sections)]
    if isinstance(sections, (list, tuple)):
        out: list[Dict[str, Any]] = []
        for s in sections:
            if isinstance(s, dict):
                out.append(dict(s))
            elif isinstance(s, str) and s.strip():
                out.append({"title": "", "items": [s.strip()]})
        return out
    if isinstance(sections, str) and sections.strip():
        return [{"title": "", "items": [sections.strip()]}]
    return []


def _iter_items(items: Any) -> Iterable[str]:
    if items is None:
        return []
    if isinstance(items, str):
        t = items.strip()
        return [t] if t else []
    if isinstance(items, dict):
        t = _as_str(items.get("text") or items.get("title") or items.get("value") or "")
        return [t] if t else []
    if isinstance(items, (list, tuple)):
        out: list[str] = []
        for it in items:
            if isinstance(it, str):
                t = it.strip()
                if t:
                    out.append(t)
                continue
            if isinstance(it, dict):
                t2 = _as_str(it.get("text") or it.get("title") or it.get("value") or "")
                if t2:
                    out.append(t2)
                continue
            t3 = _as_str(it)
            if t3:
                out.append(t3)
        return out
    t = _as_str(items)
    return [t] if t else []


def render_email_digest_text(
    *,
    title: str,
    sections: Any,
    intro: str = "",
    footer: str = "",
    max_items_per_section: int = 50,
) -> str:
    """Render a deterministic plaintext digest body.

    Notes:
    - This function is deterministic: it does not include timestamps.
    - Ordering is preserved as provided by the caller.
    """
    title_s = _as_str(title)
    intro_s = _as_str(intro)
    footer_s = _as_str(footer)
    max_items = max(0, int(max_items_per_section))

    lines: list[str] = []
    if title_s:
        lines.append(title_s)
        lines.append("")
    if intro_s:
        lines.append(intro_s)
        lines.append("")

    for sec in _coerce_sections(sections):
        sec_title = _as_str(sec.get("title") or "")
        if sec_title:
            lines.append(f"## {sec_title}")
        items = list(_iter_items(sec.get("items")))
        if max_items and len(items) > max_items:
            items = items[:max_items] + ["… (truncated)"]
        for it in items:
            if it:
                lines.append(f"- {it}")
        lines.append("")

    if footer_s:
        lines.append(footer_s)
        lines.append("")

    body = "\n".join(lines).rstrip() + "\n"
    return body


def send_email_digest(
    *,
    account: Optional[str] = None,
    to: Any,
    subject: str,
    title: str,
    sections: Any,
    intro: str = "",
    footer: str = "",
    max_items_per_section: int = 50,
) -> Dict[str, Any]:
    """Send a digest email using the configured email tool account."""
    from abstractcore.tools.comms_tools import send_email

    body_text = render_email_digest_text(
        title=title,
        sections=sections,
        intro=intro,
        footer=footer,
        max_items_per_section=max_items_per_section,
    )

    return send_email(
        account=account,
        to=to,
        subject=subject,
        body_text=body_text,
    )
