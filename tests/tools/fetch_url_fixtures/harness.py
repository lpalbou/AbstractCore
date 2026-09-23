"""Mechanical scoreboard for fetch_url extraction quality.

Three deterministic, offline, LLM-free metrics over committed fixtures:

1. ``score(content, key)``      — fact recall / junk ratio (the original bar).
2. ``structure_score(content, key)`` — do facts that live in a TABLE CELL, a
   CODE BLOCK, a LIST ITEM or a LINK LABEL survive *with* their structure?
3. ``envelope_efficiency(result)``   — how many characters does the whole
   fetch_url envelope cost per character of real content, and how much of that
   is the SAME text repeated under another key?

Gold checklists are curated by adversarial agents from each page's RAW bytes,
NEVER from the tool's own output. Whitespace/case/dash-normalized substring
matching — no fuzzy similarity, no LLM judgement.

Bar (maintainer: "on par or beyond the gold reference"): fact_recall >= 0.90
AND junk_ratio == 0.0 for each URL.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

FIXTURE_DIR = Path(__file__).resolve().parent
GOLD = json.loads((FIXTURE_DIR / "gold_facts.json").read_text(encoding="utf-8"))


_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_MD_LINK_PAIR = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
# *em* / **strong** / ***both*** -> inner text (underscores left alone: `__init__`)
_MD_EMPHASIS = re.compile(r"\*{1,3}([^*\n]+?)\*{1,3}")


def _norm(text: str) -> str:
    """Case-fold + collapse whitespace + unify dash variants + strip markdown
    MARKUP (link syntax, inline-code backticks, bold/italic runs) so matching
    compares CONTENT, not incidental rendering. A fact wrapped in `[label](url)`,
    `` `code` `` or `**bold**` is still present — the markup is a feature, not a
    missing fact. Structure checks (``structure_score``) look at the raw lines,
    so this normalization never hides a flattened table/code block/list."""
    t = str(text or "")
    t = _MD_LINK.sub(r"\1", t)  # [label](url) -> label
    t = t.replace("`", "")  # `inline code` / ``` fences -> plain text
    t = _MD_EMPHASIS.sub(r"\1", t)  # *em* / **strong** / ***both*** -> plain
    t = t.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", t).strip().lower()


def score(content: str, gold_key: str) -> Dict[str, Any]:
    """Score one extracted-content string against a gold entry."""
    gold = GOLD[gold_key]
    hay = _norm(content)
    facts: List[str] = gold["facts"]
    junk: List[str] = gold["junk"]

    matched_facts = [f for f in facts if _norm(f) in hay]
    missed_facts = [f for f in facts if _norm(f) not in hay]
    matched_junk = [j for j in junk if _norm(j) in hay]

    recall = len(matched_facts) / max(1, len(facts))
    junk_ratio = len(matched_junk) / max(1, len(junk))
    return {
        "url_key": gold_key,
        "chars": len(content or ""),
        "recall": round(recall, 3),
        "facts_matched": len(matched_facts),
        "facts_total": len(facts),
        "missed": missed_facts,
        "junk_ratio": round(junk_ratio, 3),
        "junk_hit": matched_junk,
        "passes": recall >= 0.90 and junk_ratio == 0.0,
    }


# ---------------------------------------------------------------------------
# metric 2 — structure preservation
# ---------------------------------------------------------------------------

_LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(?:```|~~~)")


def _fenced_line_indexes(lines: List[str]) -> set:
    """Indexes of lines that sit INSIDE a ``` / ~~~ fenced code block."""
    inside = False
    out = set()
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            inside = not inside
            continue
        if inside:
            out.add(i)
    return out


def _link_labels(content: str) -> List[str]:
    return [_norm(m.group(1)) for m in _MD_LINK_PAIR.finditer(str(content or ""))]


def _kept_with_structure(content: str, fact: str, kind: str) -> bool:
    """True if `fact` survives in `content` WITH its original structure."""
    needle = _norm(fact)
    if not needle:
        return False
    lines = str(content or "").splitlines()

    if kind == "link":
        # the fact must be (inside) a markdown link label, i.e. the URL survived
        return any(needle in label for label in _link_labels(content))

    fenced = _fenced_line_indexes(lines) if kind == "code" else set()
    for i, line in enumerate(lines):
        if needle not in _norm(line):
            continue
        if kind == "table":
            # a markdown table row: pipe-delimited cells
            return "|" in line
        if kind == "code":
            # inside a fence, or an indented code block, or an inline `code` span
            if i in fenced:
                return True
            if line.startswith("    ") or line.startswith("\t"):
                return True
            if "`" in line:
                return True
            continue
        if kind == "list":
            if _LIST_LINE.match(line):
                return True
            continue
    return False


def structure_score(content: str, gold_key: str) -> Dict[str, Any]:
    """Fraction of the gold `structural_facts` that survive WITH their structure.

    A code fact must still sit in a fenced/indented code block, a table fact on
    a `|` row, a list fact on a `-`/`1.` line, a link fact inside `[label](url)`.
    A fact that survives only as flat prose counts as LOST — that is exactly the
    information the flat `get_text()` path destroys.
    """
    entry = GOLD[gold_key]
    structural = entry.get("structural_facts") or []
    hay = _norm(content)
    kept, lost, absent = [], [], []
    for item in structural:
        fact, kind = item["fact"], item["kind"]
        if _norm(fact) not in hay:
            absent.append(fact)  # not extracted at all — also not structured
            continue
        if _kept_with_structure(content, fact, kind):
            kept.append(fact)
        else:
            lost.append(f"{fact}  [{kind} flattened]")
    total = len(structural)
    return {
        "url_key": gold_key,
        "structural_total": total,
        "structural_kept": len(kept),
        "structure_score": round(len(kept) / total, 3) if total else None,
        "flattened": lost,
        "missing": absent,
    }


# ---------------------------------------------------------------------------
# metric 3 — envelope efficiency (token cost of the tool result)
# ---------------------------------------------------------------------------

_SHINGLE_K = 8
# below this, a string field is metadata (url, reason, detected_as), not payload
_PAYLOAD_MIN_CHARS = 200
# `title`/`description` are declared first-class metadata in the fetch_url
# contract, not evidence mirrors of the page. A meta description is lifted from
# the opening prose, so it is ~100% "contained" in `content` by construction —
# counting it as duplicated payload is a category error, and on a short page it
# is large enough relative to `content` to slip past the mirror-size rule below.
_CONTRACT_METADATA_FIELDS = {"content", "title", "description"}


def _shingles(text: str, k: int = _SHINGLE_K) -> set:
    words = _norm(text).split()
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _containment(field_text: str, content_text: str) -> float:
    """Fraction of `field_text`'s 8-gram shingles that also occur in `content_text`.

    Cheap, deterministic, order-insensitive. Short fields (< k words) fall back
    to a plain normalized-substring test.
    """
    a = _shingles(field_text)
    if not a:
        n = _norm(field_text)
        return 1.0 if n and n in _norm(content_text) else 0.0
    b = _shingles(content_text)
    if not b:
        return 0.0
    return len(a & b) / len(a)


def envelope_efficiency(
    result: Dict[str, Any], *, redundancy_threshold: float = 0.85
) -> Dict[str, Any]:
    """Measure the char cost of a full fetch_url result dict.

    Returns
      content_chars           len(result["content"])
      total_json_chars        len(json.dumps(result))  — what a caller pays
      amplification           total_json_chars / content_chars
      duplicate_payload_chars sum of len() of every OTHER string field whose
                              text is >= `redundancy_threshold` redundant with
                              `content` (8-gram containment)
      duplicate_fields        [{field, chars, containment}, ...] worst first
      wasted_ratio            duplicate_payload_chars / total_json_chars
    """
    content = str(result.get("content") or "")
    content_chars = len(content)
    total_json_chars = len(json.dumps(result, ensure_ascii=False, default=str))

    payload: List[Dict[str, Any]] = []
    for key, value in result.items():
        if key in _CONTRACT_METADATA_FIELDS or not isinstance(value, str):
            continue
        if len(value) < _PAYLOAD_MIN_CHARS:
            continue  # url/reason/detected_as etc. are not payload
        payload.append(
            {
                "field": key,
                "chars": len(value),
                "containment": round(_containment(value, content), 3),
            }
        )
    payload.sort(key=lambda d: -d["chars"])
    # A field is a MIRROR of the content only if it is both highly redundant and
    # of comparable size. `description` is a ~200-char meta blurb lifted from the
    # opening prose: 100% redundant by containment, but it is first-class
    # metadata a consumer asks for by name, not a second copy of the page.
    # Requiring >=25% of content length keeps the metric on actual duplication.
    dupes = [
        d
        for d in payload
        if d["containment"] >= redundancy_threshold
        and content_chars > 0
        and d["chars"] >= 0.25 * content_chars
    ]
    duplicate_payload_chars = sum(d["chars"] for d in dupes)

    return {
        "content_chars": content_chars,
        "total_json_chars": total_json_chars,
        "amplification": round(total_json_chars / content_chars, 2) if content_chars else None,
        "duplicate_payload_chars": duplicate_payload_chars,
        "duplicate_fields": dupes,
        "payload_fields": payload,
        "wasted_ratio": (
            round(duplicate_payload_chars / total_json_chars, 3) if total_json_chars else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def fixture_path(gold_key: str) -> Path:
    return FIXTURE_DIR / GOLD[gold_key]["fixture"]


def fixture_html(gold_key: str) -> str:
    return fixture_path(gold_key).read_text(encoding="utf-8", errors="replace")


def fixture_bytes(gold_key: str) -> bytes:
    return fixture_path(gold_key).read_bytes()


def _kind(gold_key: str) -> str:
    return str(GOLD[gold_key].get("kind") or "html")


def all_keys() -> List[str]:
    return [k for k in GOLD if not k.startswith("_")]


def url_keys() -> List[str]:
    """Content-bearing HTML fixtures — the ones held to the recall/junk bar."""
    return [k for k in all_keys() if _kind(k) == "html"]


def doc_keys() -> List[str]:
    """Non-HTML documents (PDF) — same gold schema, different extraction path."""
    return [k for k in all_keys() if _kind(k) == "pdf"]


def special_keys() -> List[str]:
    """Fixtures that must NOT yield content (SPA shells, challenge pages)."""
    return [k for k in all_keys() if _kind(k) not in {"html", "pdf"}]


def offline_envelope(gold_key: str, *, keep_links: bool = True) -> Dict[str, Any]:
    """Rebuild the exact fetch_url result envelope for a fixture, offline.

    Mirrors the field-by-field construction in
    ``abstractcore.tools.common_tools.fetch_url``, with the HTTP round-trip
    replaced by the committed fixture bytes, so ``envelope_efficiency`` can be
    measured deterministically in CI with no network.
    """
    from abstractcore.tools.common_tools import (  # noqa: E402
        _extract_main_content,
        _normalize_text_for_evidence,
        _parse_content_by_type,
    )

    raw_bytes = fixture_bytes(gold_key)
    url = GOLD[gold_key]["url"]
    kind = _kind(gold_key)

    if kind == "pdf":
        from abstractcore.media.pdf_routing import route_pdf_bytes  # noqa: E402

        # `include_full_content=True` mirrors fetch_url's OWN default
        # (common_tools.fetch_url signature, `include_full_content: bool = True`),
        # not route_pdf_bytes' default of False. Getting this wrong makes the
        # rebuilt envelope look truncated when the shipping tool is not.
        route = route_pdf_bytes(
            raw_bytes, source_url=url, include_full_content=True, preferred_backend="auto"
        )
        raw_text: Optional[str] = str(route.get("raw_text") or "") or None
        normalized_text: Optional[str] = str(route.get("normalized_text") or "") or raw_text
        content_text: Optional[str] = normalized_text
        title: Optional[str] = str(route.get("title") or "") or None
        description: Optional[str] = None
        parsed = str(route.get("rendered") or "")
        content_type = "application/pdf"
        detected = "pdf"
        # a PDF has no link-density notion; the tool reports False, not None
        link_dominant = False
    else:
        html = raw_bytes.decode("utf-8", errors="replace")
        content_type = "text/html; charset=utf-8"
        detected = "html"
        raw_text = html
        normalized_text = _normalize_text_for_evidence(
            raw_text=html, content_type_header=content_type, url=url
        )
        main = _extract_main_content(html, url, keep_links=keep_links)
        title = str(main.get("title") or "") or None
        description = str(main.get("description") or "") or None
        content_text = str(main.get("content") or "") or None
        link_dominant = bool(main.get("link_dominant"))
        parsed = _parse_content_by_type(raw_bytes, content_type, url, keep_links=keep_links)

    rendered = "\n".join(
        [
            "🌐 URL Fetch Results",
            f"📍 URL: {url}",
            "⏰ Timestamp: 2026-08-21T00:00:00Z",
            "✅ Status: 200 OK",
            f"📊 Content-Type: {content_type}",
            f"📏 Size: {len(raw_bytes):,} bytes",
            f"🧭 Detected-As: {detected}",
            "\n📄 Content Analysis:",
            parsed,
        ]
    )
    result: Dict[str, Any] = {
        "success": True,
        "error": None,
        "url": url,
        "final_url": url,
        "timestamp": "2026-08-21T00:00:00Z",
        "status_code": 200,
        "attempts": 1,
        "reason": "OK",
        "content_type": content_type,
        "detected_as": detected,
        "text_available": bool(content_text and content_text.strip()),
        "size_bytes": len(raw_bytes),
        "title": title,
        "description": description,
        "content": content_text,
        "content_chars": len(str(content_text or "")),
        # fetch_url reports WHY a link-dominant listing page kept its URLs even
        # when the caller passed keep_links=False. The rebuild must carry it or
        # the drift detector fires on a key the tool legitimately added.
        "link_dominant": link_dominant,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        # RENDER-ESCALATION PROVENANCE (2026-08-21). fetch_url may re-fetch a
        # page whose static extraction came back empty through a headless
        # browser and extract from the RENDERED DOM instead. Every fixture here
        # is replayed from committed bytes with no browser in the loop, so all
        # three take their static-path values. They are carried anyway because
        # the drift detector compares KEY SETS: a rebuild that omits them
        # reports the tool as having drifted when it has not.
        #   rendered_with_browser — did this `content` come out of a headless render
        #   render_note           — why it did or did not escalate (None on the static path)
        #   extraction_error      — the exception that made extraction degrade, if any.
        #                           Exists because a bare `except Exception` used to
        #                           turn any internal failure into success=True /
        #                           content=None, indistinguishable from an empty page.
        # The post-JavaScript DOM. Always None on this offline rebuild: the
        # fixtures are the SERVER's bytes, so no render ever happens here.
        "rendered_dom": None,
        "rendered_with_browser": False,
        # Site adapter (a machine-readable view the site publishes, e.g. reddit's
        # thread Atom feed). None on this offline rebuild: fixtures are served
        # HTML, and an adapter only runs after the static extraction failed.
        "adapter_used": None,
        "adapter_source_url": None,
        "render_note": None,
        "extraction_error": None,
        "rendered": rendered,
    }
    # CRITICAL for fidelity: fetch_url applies a payload policy to the result
    # dict as its LAST step before returning (it withholds oversized evidence
    # mirrors and leaves a `*_withheld` descriptor). A rebuild that skips it
    # measures a tool that does not ship. Guarded so the harness still runs
    # against a tree where the symbol does not exist.
    # `test_offline_envelope_matches_the_real_fetch_url_envelope` pins that this
    # rebuild has not drifted from the tool.
    try:
        from abstractcore.tools.common_tools import (  # noqa: E402
            _apply_fetch_url_payload_policy,
        )
    except ImportError:
        return result
    return _apply_fetch_url_payload_policy(result)


if __name__ == "__main__":  # baseline / manual run
    import sys

    sys.path.insert(0, str(FIXTURE_DIR.parents[2]))
    from abstractcore.tools.common_tools import _extract_main_content

    print("== extraction quality (primary `content`) ==")
    print(f"{'url':<18} {'chars':>8} {'recall':>7} {'facts':>8} {'junk':>6} {'struct':>7} {'pass':>5}")
    print("-" * 72)
    all_pass = True
    for key in url_keys():
        html = fixture_html(key)
        url = GOLD[key]["url"]
        # Score the SHIPPING primary `content` field (structure-preserving markdown).
        main = _extract_main_content(html, url, keep_links=True)
        content = str(main.get("content") or "")
        s = score(content, key)
        st = structure_score(content, key)
        all_pass = all_pass and s["passes"]
        struct = "-" if st["structure_score"] is None else f"{st['structure_score']:.2f}"
        print(
            f"{key:<18} {s['chars']:>8} {s['recall']:>7} "
            f"{s['facts_matched']:>3}/{s['facts_total']:<4} {s['junk_ratio']:>6} {struct:>7} {str(s['passes']):>5}"
        )
        if s["missed"]:
            print(f"    missed: {s['missed']}")
        if s["junk_hit"]:
            print(f"    junk:   {s['junk_hit']}")
        if st["flattened"] or st["missing"]:
            print(f"    struct: flattened={st['flattened']} missing={st['missing']}")
    print("-" * 72)
    print("ALL PASS" if all_pass else "NOT ALL PASSING")

    print()
    print("== envelope efficiency (whole fetch_url result dict) ==")
    print(f"{'url':<18} {'content':>9} {'total':>9} {'amp':>7} {'dup_chars':>10} {'wasted':>7}  duplicate fields")
    print("-" * 100)
    for key in url_keys() + doc_keys():
        try:
            env = envelope_efficiency(offline_envelope(key))
        except Exception as exc:  # pragma: no cover - manual tool
            print(f"{key:<18} ERROR {exc}")
            continue
        amp = "-" if env["amplification"] is None else f"{env['amplification']:.2f}x"
        fields = (
            ", ".join(f"{d['field']}({d['chars']}@{d['containment']})" for d in env["payload_fields"])
            or "-"
        )
        print(
            f"{key:<18} {env['content_chars']:>9} {env['total_json_chars']:>9} {amp:>7} "
            f"{env['duplicate_payload_chars']:>10} {env['wasted_ratio']:>7}  {fields}"
        )
