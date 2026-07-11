"""Mechanical fact-recall harness for fetch_url extraction quality.

Scores an extracted-content string against a per-URL gold checklist curated by
adversarial subagents from raw-byte fetches (NEVER from the tool's own output).
Whitespace- and case-normalized substring matching — no fuzzy similarity, no
LLM judgement. Offline and deterministic: runs against committed HTML fixtures.

Bar (maintainer: "on par or beyond the gold reference"): fact_recall >= 0.90
AND junk_ratio == 0.0 for each URL.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

FIXTURE_DIR = Path(__file__).resolve().parent
GOLD = json.loads((FIXTURE_DIR / "gold_facts.json").read_text(encoding="utf-8"))


_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")


def _norm(text: str) -> str:
    """Case-fold + collapse whitespace + unify dash variants + reduce markdown
    links to their label, so matching survives line-splitting, en/em-dash vs
    hyphen rendering, and inline links (a fact wrapped in [label](url) is still
    present — the link markup is a feature, not a missing fact)."""
    t = str(text or "")
    t = _MD_LINK.sub(r"\1", t)  # [label](url) -> label
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
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


def fixture_html(gold_key: str) -> str:
    return (FIXTURE_DIR / GOLD[gold_key]["fixture"]).read_text(encoding="utf-8", errors="replace")


def url_keys() -> List[str]:
    return [k for k in GOLD if not k.startswith("_")]


if __name__ == "__main__":  # baseline / manual run
    import sys

    sys.path.insert(0, str(FIXTURE_DIR.parents[2]))
    from abstractcore.tools.common_tools import _extract_main_content

    print(f"{'url':<15} {'chars':>7} {'recall':>7} {'facts':>7} {'junk':>6} {'pass':>5}")
    print("-" * 60)
    all_pass = True
    for key in url_keys():
        html = fixture_html(key)
        url = GOLD[key]["url"]
        # Score the SHIPPING primary `content` field (structure-preserving markdown).
        main = _extract_main_content(html, url, keep_links=True)
        s = score(str(main.get("content") or ""), key)
        all_pass = all_pass and s["passes"]
        print(f"{key:<15} {s['chars']:>7} {s['recall']:>7} "
              f"{s['facts_matched']:>3}/{s['facts_total']:<3} {s['junk_ratio']:>6} {str(s['passes']):>5}")
        if s["missed"]:
            print(f"    missed: {s['missed']}")
        if s["junk_hit"]:
            print(f"    junk:   {s['junk_hit']}")
    print("-" * 60)
    print("ALL PASS" if all_pass else "NOT ALL PASSING")
