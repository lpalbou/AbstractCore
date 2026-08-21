"""Adversarial regressions for the WEB SEARCH path (web_search / skim_websearch).

Round 2 (2026-08-21). Pins W1-W4 from the 2026-08-21 production transcript.

Offline and deterministic: every test replaces BOTH backends —
`common_tools._import_ddgs_class` (the ddgs primary) and `common_tools.requests`
(the duckduckgo.html fallback) — so nothing here touches the network. The fake
ddgs deliberately raises `KeyError(timelimit)` on any value outside
{h,d,w,m,y}, which is exactly what ddgs 9.x does and exactly what killed the
primary backend in production.

Groups:
  W1  require_in synonyms are silently coerced with no warning
  W2  time_range aliases are passed through and kill the primary backend
  W3  fused / space-starved result text is neither repaired nor flagged
  W4  skim_websearch's schema teaches the model no allowed values
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

import abstractcore.tools.common_tools as ct

FIXTURES = Path(__file__).resolve().parent / "fetch_url_fixtures"
FUSED_FIXTURE = json.loads((FIXTURES / "websearch_fused_rows.json").read_text(encoding="utf-8"))
PAYLOADS: List[Dict[str, Any]] = FUSED_FIXTURE["payloads"]

# ddgs accepts exactly these; anything else raises KeyError inside the library.
DDGS_TIMELIMITS = {"h", "d", "w", "m", "y"}


# ---------------------------------------------------------------------------
# offline backends
# ---------------------------------------------------------------------------


class _FakeResponse:
    status_code = 200
    text = (
        '<a class="result__a" href="https://fallback.example/one">Fallback result one</a>'
        '<a class="result__snippet">A clean fallback snippet with real spaces.</a>'
    )

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None


class _FakeRequests:
    """Stands in for the duckduckgo.html fallback backend."""

    def __init__(self, calls: List[Dict[str, Any]]) -> None:
        self._calls = calls

    def get(self, url, params=None, headers=None, timeout=None):  # noqa: ANN001
        self._calls.append({"url": url, "params": dict(params or {})})
        return _FakeResponse()


def _install_backends(monkeypatch, rows: List[Dict[str, Any]]):
    """Replace both search backends. Returns a dict recording what they saw."""
    seen: Dict[str, Any] = {"ddgs_calls": [], "html_calls": []}

    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(
            self,
            query=None,
            keywords=None,
            max_results=10,
            region=None,
            safesearch=None,
            timelimit=None,
        ):  # noqa: ANN001
            seen["ddgs_calls"].append({"timelimit": timelimit, "max_results": max_results})
            if timelimit is not None and timelimit not in DDGS_TIMELIMITS:
                # Verbatim ddgs 9.x behaviour: an unknown timelimit is a KeyError.
                raise KeyError(timelimit)
            return [
                {"title": r["title"], "href": r["url"], "body": r["snippet"]} for r in rows
            ]

    monkeypatch.setattr(ct, "_import_ddgs_class", lambda: (_FakeDDGS, "ddgs"))
    monkeypatch.setattr(ct, "requests", _FakeRequests(seen["html_calls"]))
    monkeypatch.setattr(ct, "REQUESTS_AVAILABLE", True)
    return seen


CLEAN_ROWS = [
    {
        "title": "Top Tech News Today, August 20, 2026",
        "url": "https://techstartups.com/a",
        "snippet": "Google is deepening its bet on custom AI silicon in the latest tech news round-up.",
    },
    {
        "title": "Reuters AI News",
        "url": "https://reuters.com/b",
        "snippet": "Explore the latest artificial intelligence news with Reuters.",
    },
]


def _payload_rows(payload_id: str) -> List[Dict[str, Any]]:
    for p in PAYLOADS:
        if p["id"] == payload_id:
            return p["rows"]
    raise KeyError(payload_id)


# ---------------------------------------------------------------------------
# W1 — require_in synonyms
# ---------------------------------------------------------------------------

# What a model plausibly sends -> the scope it obviously means.
REQUIRE_IN_SYNONYMS = [
    ("snippet", "snippet"),
    ("title", "title"),
    ("title_snippet", "title_snippet"),
    ("all", "all"),
    ("body", "snippet"),
    ("text", "snippet"),
    ("Snippet", "snippet"),
    ("title,body", "title_snippet"),
    ("title,snippet", "title_snippet"),
    ("title+snippet", "title_snippet"),
    ("both", "title_snippet"),
    ("any", "all"),
    ("everything", "all"),
]


@pytest.mark.parametrize("sent,expected", REQUIRE_IN_SYNONYMS, ids=[s for s, _ in REQUIRE_IN_SYNONYMS])
def test_require_in_synonyms_resolve_to_the_scope_the_model_meant(monkeypatch, sent, expected):
    """W1a. Production sent `require_in="title,body"` — a perfectly reasonable
    guess — and skim_websearch (L4721-4723) silently rewrote it to "snippet",
    then echoed `"require_in":"snippet"` back, so the transcript contradicts the
    tool call. Precedent for the fix: test_search_files_output_mode_synonyms.py.
    """
    _install_backends(monkeypatch, CLEAN_ROWS)
    out = json.loads(
        ct.skim_websearch(query="q", required_terms=["silicon"], require_in=sent, num_results=3)
    )
    assert out["filter"]["require_in"] == expected, (
        f"require_in={sent!r} resolved to {out['filter']['require_in']!r}, expected {expected!r}"
    )


def test_require_in_synonym_actually_widens_the_match(monkeypatch):
    """W1a, functionally: "title,body" must let a title-only hit through.

    Asserting the echoed scope alone would be satisfiable by relabelling; this
    asserts the filter really behaves like title_snippet.
    """
    rows = [
        {
            "title": "Marvell and Google sign a custom silicon deal",
            "url": "https://example.com/x",
            "snippet": "A short summary with none of the interesting words in it.",
        }
    ]
    _install_backends(monkeypatch, rows)
    out = json.loads(
        ct.skim_websearch(query="q", required_terms=["marvell"], require_in="title,body", num_results=3)
    )
    assert out["counts"]["matched"] == 1, (
        "a title-only hit was dropped: require_in='title,body' was treated as snippet-only "
        f"(counts={out['counts']}, filter={out['filter']})"
    )


@pytest.mark.parametrize("sent", ["bogus_scope", "headline", "url_only"])
def test_unmappable_require_in_is_reported_not_silently_swapped(monkeypatch, sent):
    """W1b. When a value genuinely cannot be mapped, the substitution must be
    VISIBLE: the payload has to name both the value sent and the value used, so
    the model can correct itself instead of silently searching the wrong field.
    """
    _install_backends(monkeypatch, CLEAN_ROWS)
    out = json.loads(
        ct.skim_websearch(query="q", required_terms=["silicon"], require_in=sent, num_results=3)
    )
    surfaced = " ".join(
        [json.dumps(out.get("warnings") or []), json.dumps(out.get("limitations") or []), str(out.get("hint") or "")]
    ).lower()
    assert sent.lower() in surfaced, (
        f"require_in={sent!r} was coerced to {out['filter']['require_in']!r} with no mention of "
        f"the original value anywhere in warnings/limitations/hint: {out.get('warnings')} "
        f"{out.get('limitations')}"
    )
    assert out["filter"]["require_in"] in {"snippet", "title", "title_snippet", "all"}


# ---------------------------------------------------------------------------
# W2 — time_range aliases
# ---------------------------------------------------------------------------

TIME_RANGE_ALIASES = [
    ("h", "h"),
    ("d", "d"),
    ("w", "w"),
    ("m", "m"),
    ("y", "y"),
    ("24h", "h"),
    ("7d", "w"),
    ("30d", "m"),
    ("1y", "y"),
    ("hour", "h"),
    ("past hour", "h"),
    ("1h", "h"),
    ("day", "d"),
    ("today", "d"),
    ("1d", "d"),
    ("24 hours", "d"),
    ("past day", "d"),
    ("week", "w"),
    ("7 days", "w"),
    ("past week", "w"),
    ("month", "m"),
    ("30 days", "m"),
    ("year", "y"),
    ("past year", "y"),
    ("D", "d"),
    (" day ", "d"),
]


@pytest.mark.parametrize("sent,expected", TIME_RANGE_ALIASES, ids=[repr(s) for s, _ in TIME_RANGE_ALIASES])
def test_time_range_aliases_never_reach_the_backend_unnormalized(monkeypatch, sent, expected):
    """W2. `_normalize_time_range` (L4321) maps only 24h/7d/30d/1y and PASSES
    EVERYTHING ELSE THROUGH. In production `time_range="day"` reached
    `ddgs.text(timelimit="day")`, raised KeyError('day'), and the whole primary
    backend was abandoned for the weaker HTML scraper:

        "backend_attempts":[{"name":"ddgs.text","success":false,"error":"'day'"},
                            {"name":"duckduckgo.html","success":true,...}]

    One unlucky word silently halved result quality. A filter value must never
    cost the primary backend.
    """
    seen = _install_backends(monkeypatch, CLEAN_ROWS)
    payload = json.loads(ct.web_search(query="q", num_results=3, time_range=sent))

    sent_limits = [c["timelimit"] for c in seen["ddgs_calls"]]
    assert sent_limits, "the ddgs backend was never called"
    assert sent_limits[0] == expected, (
        f"time_range={sent!r} was sent to the backend as {sent_limits[0]!r}, expected {expected!r}"
    )
    assert payload.get("backend_used") == "ddgs.text", (
        f"time_range={sent!r} knocked the run onto the fallback backend "
        f"({payload.get('backend_used')}); attempts={payload.get('backend_attempts')}"
    )


@pytest.mark.parametrize("sent", ["banana", "fortnight", "since tuesday", "2026"])
def test_unmappable_time_range_is_dropped_with_a_warning_not_passed_through(monkeypatch, sent):
    """W2b. A value that cannot be mapped must be DROPPED before the backend
    call and reported — never forwarded to raise inside the library."""
    seen = _install_backends(monkeypatch, CLEAN_ROWS)
    payload = json.loads(ct.web_search(query="q", num_results=3, time_range=sent))

    sent_limits = [c["timelimit"] for c in seen["ddgs_calls"]]
    assert all(v is None or v in DDGS_TIMELIMITS for v in sent_limits), (
        f"unmappable time_range={sent!r} was forwarded to the backend as {sent_limits!r}"
    )
    assert payload.get("backend_used") == "ddgs.text", (
        f"unmappable time_range={sent!r} cost the primary backend "
        f"(used {payload.get('backend_used')})"
    )
    surfaced = " ".join(
        [json.dumps(payload.get("warnings") or []), json.dumps(payload.get("limitations") or [])]
    ).lower()
    assert sent.strip().lower() in surfaced or "time_range" in surfaced, (
        f"time_range={sent!r} was ignored with no warning: warnings={payload.get('warnings')} "
        f"limitations={payload.get('limitations')}"
    )


def test_skim_websearch_inherits_the_time_range_normalization(monkeypatch):
    """The production call came through skim_websearch, not web_search."""
    seen = _install_backends(monkeypatch, CLEAN_ROWS)
    out = json.loads(ct.skim_websearch(query="q", num_results=3, time_range="day"))
    assert [c["timelimit"] for c in seen["ddgs_calls"]] == ["d"]
    assert out.get("backend_used") == "ddgs.text", out.get("backend_attempts")


# ---------------------------------------------------------------------------
# W3 — fused / space-starved result text
# ---------------------------------------------------------------------------

_STRIP = ".,:;|!?()[]\"'`·—–"
# Common English words that show up glued INSIDE a fused token. A long token
# that contains two of them at a non-zero offset is not one word.
_GLUE = (
    "the", "and", "news", "latest", "today", "with", "for", "you", "that",
    "this", "from", "about", "more", "here", "breaking", "world", "tech",
)


def _tokens(text: str) -> List[str]:
    return [t.strip(_STRIP) for t in str(text or "").split()]


def _case_transitions(token: str) -> int:
    return sum(
        1
        for a, b in zip(token, token[1:])
        if (a.islower() and b.isupper())
        or (a.isalpha() and b.isdigit())
        or (a.isdigit() and b.isalpha())
    )


def row_looks_fused(text: str) -> bool:
    """Reference detector for space-starved backend text.

    Improved over the first prototype (which scored 9/13 true positives and
    4/19 false positives on the committed fixture) by (a) requiring internal
    glue words before flagging a long all-alphabetic token, and (b) adding an
    alpha->digit rule that catches the very common `August21,2026` shape.
    Measured on `websearch_fused_rows.json`: 13/13 true positives, 1/19 false
    positives (a CamelCase-identifier row) — see
    `test_reference_fusion_detector_quality_is_pinned`.
    """
    for token in _tokens(text):
        if len(token) < 12:
            continue
        if _case_transitions(token) >= 2:
            return True
        if re.search(r"[A-Za-z]\d", token):
            return True
        if token.isalpha():
            low = token.lower()
            hits = sum(1 for g in _GLUE if low.find(g, 1) > 0)
            if hits >= 2:
                return True
            if len(token) >= 13 and hits >= 1:
                return True
    return False


def payload_looks_fused(rows: List[Dict[str, Any]]) -> bool:
    """Fusion is a property of the BACKEND, not of one row: require at least two
    flagged rows AND at least 30% of the payload. A single long real word, or a
    single CamelCase-heavy result, can never trip it."""
    if not rows:
        return False
    flagged = sum(1 for r in rows if row_looks_fused(f"{r.get('title','')} {r.get('snippet','')}"))
    return flagged >= 2 and (flagged / len(rows)) >= 0.30


def test_reference_fusion_detector_quality_is_pinned():
    """The detector is the reference the production fix is graded against, so
    its measured quality is itself a pinned fact. If someone improves it, these
    numbers may only move in the right direction."""
    tp = fn = fp = tn = 0
    for payload in PAYLOADS:
        for row in payload["rows"]:
            flagged = row_looks_fused(f"{row['title']} {row['snippet']}")
            if row["fused"]:
                tp, fn = (tp + 1, fn) if flagged else (tp, fn + 1)
            else:
                fp, tn = (fp + 1, tn) if flagged else (fp, tn + 1)
    assert tp + fn == 13 and fp + tn == 19, "fixture labels changed; re-derive the numbers"
    assert tp >= 13, f"row recall regressed: {tp}/13"
    assert fp <= 1, f"row false positives regressed: {fp}/19"

    for payload in PAYLOADS:
        got = payload_looks_fused(payload["rows"])
        assert got == payload["expect_fused_payload"], (
            f"{payload['id']}: payload gate said {got}, expected "
            f"{payload['expect_fused_payload']} — {payload['notes']}"
        )


@pytest.mark.parametrize(
    "payload_id",
    [p["id"] for p in PAYLOADS if p["expect_fused_payload"]],
)
def test_fused_backend_text_is_repaired_or_flagged(monkeypatch, payload_id):
    """W3. The primary backend returns rows with words fused at stripped-<b>
    boundaries ("technewsstories today,August21,2026"). Production shipped them
    with `degraded:false` and no warning at all, so the model read damaged prose
    as if it were clean.

    Either outcome is acceptable — repair the text, or tell the caller it is
    damaged — so this asserts the OUTCOME, not a mechanism.
    """
    rows = _payload_rows(payload_id)
    _install_backends(monkeypatch, rows)
    out = json.loads(ct.skim_websearch(query="q", num_results=len(rows)))

    returned = [{"title": r.get("title", ""), "snippet": r.get("snippet", "")} for r in out["results"]]
    repaired = not payload_looks_fused(returned)
    signalled = bool(out.get("degraded")) or "fused_result_text" in [
        str(x) for x in (out.get("limitations") or [])
    ]
    assert repaired or signalled, (
        f"{payload_id}: fused rows returned verbatim with degraded={out.get('degraded')}, "
        f"warnings={out.get('warnings')}, limitations={out.get('limitations')} — the caller has "
        f"no way to know the snippet text is damaged"
    )


@pytest.mark.parametrize(
    "payload_id",
    [p["id"] for p in PAYLOADS if not p["expect_fused_payload"]],
)
def test_clean_backend_text_is_not_falsely_flagged(monkeypatch, payload_id):
    """The other half of the bar: the duckduckgo.html payload, the mostly-clean
    ddgs payload, and a page full of long legitimate words must NOT be reported
    as degraded."""
    rows = _payload_rows(payload_id)
    _install_backends(monkeypatch, rows)
    out = json.loads(ct.skim_websearch(query="q", num_results=len(rows)))
    assert "fused_result_text" not in [str(x) for x in (out.get("limitations") or [])], (
        f"{payload_id}: clean payload flagged as fused"
    )


def test_fused_text_does_not_silently_shrink_required_terms_matching(monkeypatch):
    """W3 downstream. `required_terms` is exact-substring, so "tech news" cannot
    match "technewsstories".

    NOTE (correction to the round-1 framing): `_match_item`'s whitespace-elided
    fallback already rescues MULTI-WORD terms, and it sets `note`. What is still
    missing is the machine-readable signal — `degraded`/`limitations` stay clean,
    so a programmatic consumer cannot tell. This test pins the recovery AND the
    signal together.
    """
    rows = _payload_rows("transcript_1")
    _install_backends(monkeypatch, rows)
    out = json.loads(
        ct.skim_websearch(query="q", required_terms=["tech news"], require_in="title_snippet", num_results=8)
    )
    assert out["counts"]["matched"] >= 1, (
        f"'tech news' matched nothing against fused rows: {out['counts']}"
    )
    signalled = bool(out.get("degraded")) or "fused_result_text" in [
        str(x) for x in (out.get("limitations") or [])
    ]
    assert signalled, (
        "matches were only recovered through whitespace-elision, but the payload reports "
        f"degraded={out.get('degraded')} limitations={out.get('limitations')} — a programmatic "
        "consumer cannot see that the matched text is damaged"
    )


# ---------------------------------------------------------------------------
# W4 — the schema teaches the model nothing
# ---------------------------------------------------------------------------

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

ENUMERATED_ARGS = {
    "require_in": ["snippet", "title", "title_snippet", "all"],
    "match": ["any", "all"],
    "safe_search": ["strict", "moderate", "off"],
    "time_range": ["h", "d", "w", "m", "y"],
}


def _tool_definition(name: str):
    from abstractcore.tools.inventory import _INVENTORY_MODULES, _scan_module_tool_definitions

    for module_path in _INVENTORY_MODULES:
        for candidate in _scan_module_tool_definitions(module_path):
            if candidate.name == name:
                return candidate
    raise AssertionError(f"{name} not found in any scanned module")


def _model_visible_surface(name: str) -> tuple[str, Dict[str, Any]]:
    """Everything about the tool that actually reaches a model, both lanes."""
    from abstractcore.tools.handler import UniversalToolHandler

    tool_def = _tool_definition(name)
    handler = UniversalToolHandler(MODEL)
    prompted = handler.format_tools_prompt([tool_def])
    native_tools = handler.prepare_tools_for_native([tool_def])
    native = json.dumps(native_tools, ensure_ascii=False)
    properties = (
        native_tools[0].get("function", {}).get("parameters", {}).get("properties", {})
        if native_tools
        else {}
    )
    return f"{prompted}\n{native}", properties


def _teaches_values(surface: str, properties: Dict[str, Any], arg: str, allowed: List[str]) -> List[str]:
    """Allowed values for `arg` that the model is NEVER shown.

    A bare substring test is worthless here — every single-letter time_range
    value ("h", "d", "w") occurs by accident in ordinary prose. So a value only
    counts as taught if it is either (a) in a JSON-schema `enum` for that
    argument, or (b) a delimited token inside a 320-char window that starts at a
    mention of the argument name.
    """
    enum = properties.get(arg, {}).get("enum") if isinstance(properties.get(arg), dict) else None
    if isinstance(enum, list):
        present = {str(v).strip().lower() for v in enum}
        return [v for v in allowed if v.lower() not in present]

    windows = "".join(
        surface[m.start() : m.start() + 320] for m in re.finditer(re.escape(arg), surface)
    )
    missing = []
    for value in allowed:
        token = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])")
        if not token.search(windows):
            missing.append(value)
    return missing


@pytest.mark.parametrize("arg,allowed", sorted(ENUMERATED_ARGS.items()))
def test_skim_websearch_schema_enumerates_its_allowed_values(arg, allowed):
    """W4. skim_websearch's entire docstring is one line — "Return a smaller,
    filtered subset of `web_search` results." — with no Args block, so neither
    the prompted block nor the native JSON schema carries a single allowed value
    for require_in / match / safe_search / time_range. W1 and W2 are both
    downstream of this: the model guessed "title,body" and "day" because nothing
    told it otherwise.

    Asserted against the union of BOTH lanes so either fix works (a JSON-schema
    `enum`, or an Args block that reaches the rendered description).
    """
    surface, properties = _model_visible_surface("skim_websearch")
    assert arg in surface, f"{arg} is not even named in the model-visible tool surface"
    missing = _teaches_values(surface, properties, arg, allowed)
    assert not missing, (
        f"skim_websearch teaches the model nothing about {arg}: {missing} never appear in the "
        f"prompted block or the native schema"
    )


def test_web_search_schema_enumerates_its_time_range_values():
    """web_search DOES document time_range in its docstring Args block — but the
    docstring is not what gets rendered, so the model never sees it either."""
    surface, properties = _model_visible_surface("web_search")
    missing = _teaches_values(surface, properties, "time_range", ["h", "d", "w", "m", "y"])
    assert not missing, (
        f"web_search's documented time_range values {missing} never reach the model: the Args "
        f"block lives in the docstring, which neither lane renders"
    )


def test_skim_websearch_docstring_documents_its_arguments():
    """The root cause in one assertion."""
    doc = ct.skim_websearch.__doc__ or ""
    assert "Args:" in doc, (
        "skim_websearch has no Args section; its whole docstring is: " + repr(doc.strip()[:200])
    )
    for arg in ENUMERATED_ARGS:
        assert arg in doc, f"{arg} is undocumented in skim_websearch's docstring"
