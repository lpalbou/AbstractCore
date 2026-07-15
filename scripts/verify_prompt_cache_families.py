#!/usr/bin/env python3
"""Live prompt-cache verification harness (MLX + GGUF lanes).

Purpose: empirically verify the claims in docs/prompt-caching.md for a given
model — does the in-process cache engage, what outcome does the delta lane
report (hit_extend / rebuilt / bypassed), is warm-turn output still CORRECT
(fact recall under a warm cache catches stale-context/wrong-trim bugs), and
what does the warm turn cost relative to the cold one.

Shape: a ReAct-like growing-prefix conversation — the case that matters for
agent loops. Three turns against ONE provider instance and ONE cache key:

  turn 1  cold   system + q1
  turn 2  warm   system + q1 + a1 + q2            (pure growing prefix)
  turn 3  warm   ... + q3 = fact-recall question   (answer must contain the
                                                    fact planted in the SYSTEM
                                                    prompt at turn 1)

Usage:
  python scripts/verify_prompt_cache_families.py --provider mlx \
      --model mlx-community/Qwen3-4B-Instruct-2507-4bit --json out.json
  python scripts/verify_prompt_cache_families.py --provider huggingface \
      --model /path/to/model-Q4_K_M.gguf

Never call this from tests; it loads real models. It is an operator/live tool.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))  # local working tree, never site-packages

FACT = "BLUE-HERON-42"

SYSTEM_PROMPT = (
    "You are a precise assistant inside an agent loop. Follow instructions "
    "exactly and keep answers short.\n\n"
    "Operational context (stable across the session):\n"
    "- The project codename is "
    + FACT
    + ". If asked for the project codename, answer with it verbatim.\n"
    "- You have no tools available in this session; answer directly.\n"
    "- Style: terse, factual, no preamble.\n\n"
    + "Reference notes (padding to make the shared prefix realistic):\n"
    + "\n".join(
        f"- note {i:02d}: the pipeline stage '{name}' runs at priority {i} and "
        "must not be reordered."
        for i, name in enumerate(
            [
                "ingest", "normalize", "dedupe", "embed", "index", "rank",
                "digest", "commit", "audit", "archive", "replicate", "expire",
            ]
        )
    )
)

TURNS = [
    "In one short sentence, what is the first pipeline stage and its priority?",
    "In one short sentence, which stage runs at priority 5?",
    "What is the project codename? Answer with the codename only.",
]


class _WarnCatcher(logging.Handler):
    """Collect abstractcore #FALLBACK / cache warnings emitted during the run."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(k in msg for k in ("FALLBACK", "cache", "trim", "prompt_cache")):
            self.lines.append(msg)


def _usage_tokens(usage: Optional[Dict[str, Any]], *keys: str) -> Optional[int]:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        val = usage.get(key)
        if isinstance(val, int):
            return val
    return None


def run(provider: str, model: str, max_tokens: int, out_tokens: int) -> Dict[str, Any]:
    from abstractcore import create_llm

    catcher = _WarnCatcher()
    logging.getLogger().addHandler(catcher)
    for name in ("abstractcore", "abstractcore.providers"):
        logging.getLogger(name).addHandler(catcher)

    t_load0 = time.monotonic()
    llm = create_llm(provider, model=model, max_tokens=max_tokens)
    load_s = time.monotonic() - t_load0

    key = f"verify-{int(time.time())}"
    messages: List[Dict[str, str]] = []
    turns_report: List[Dict[str, Any]] = []
    fact_ok: Optional[bool] = None

    for idx, question in enumerate(TURNS):
        messages.append({"role": "user", "content": question})
        t0 = time.monotonic()
        resp = llm.generate(
            messages=list(messages),
            system_prompt=SYSTEM_PROMPT,
            prompt_cache_key=key,
            max_output_tokens=out_tokens,
            temperature=0.0,
        )
        wall_s = time.monotonic() - t0
        content = str(getattr(resp, "content", "") or "")
        messages.append({"role": "assistant", "content": content})

        meta = getattr(resp, "metadata", None) or {}
        cache_tel = meta.get("prompt_cache") if isinstance(meta, dict) else None
        usage = getattr(resp, "usage", None)
        turn = {
            "turn": idx + 1,
            "wall_s": round(wall_s, 3),
            "input_tokens": _usage_tokens(usage, "input_tokens", "prompt_tokens"),
            "output_tokens": _usage_tokens(usage, "output_tokens", "completion_tokens"),
            "prompt_cache": cache_tel,
            "content_head": content[:160],
        }
        turns_report.append(turn)
        if idx == 2:
            fact_ok = FACT.lower() in content.lower()

    # Provider-side cache stats (GGUF exposes control-plane mode here).
    stats: Dict[str, Any] = {}
    try:
        raw_stats = llm.get_prompt_cache_stats()
        if isinstance(raw_stats, dict):
            gguf = raw_stats.get("gguf")
            if isinstance(gguf, dict):
                stats["gguf_control_plane_chat_format"] = gguf.get(
                    "control_plane_chat_format"
                )
                keys = gguf.get("keys")
                if isinstance(keys, dict):
                    stats["gguf_keys"] = {
                        k: {kk: vv for kk, vv in v.items() if kk != "fed_token_ids"}
                        if isinstance(v, dict)
                        else v
                        for k, v in keys.items()
                    }
            for k in ("mode", "enabled", "entries"):
                if k in raw_stats:
                    stats[k] = raw_stats[k]
    except Exception as exc:  # honest reporting beats a crash
        stats["stats_error"] = f"{type(exc).__name__}: {exc}"

    report = {
        "provider": provider,
        "model": model,
        "load_s": round(load_s, 2),
        "turns": turns_report,
        "fact_recall_ok": fact_ok,
        "warnings": catcher.lines,
        "provider_stats": stats,
    }

    cold = turns_report[0]["wall_s"]
    warm = turns_report[1]["wall_s"]
    report["warm_over_cold"] = round(warm / cold, 3) if cold else None
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", required=True, choices=["mlx", "huggingface"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="context window to request (GGUF n_ctx)")
    ap.add_argument("--out-tokens", type=int, default=384,
                    help="output budget; thinking models (Qwen3.5/3.6, Ornith) "
                         "spend reasoning tokens first, so keep this generous")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write full JSON report to this path")
    args = ap.parse_args()

    report = run(args.provider, args.model, args.max_tokens, args.out_tokens)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Verdict line for quick scanning.
    t2 = report["turns"][1] if len(report["turns"]) > 1 else {}
    tel = t2.get("prompt_cache") or {}
    print(
        f"\nVERDICT model={args.model} turn2_outcome={tel.get('outcome', 'n/a')} "
        f"cached={tel.get('cached_tokens', 'n/a')} fed={tel.get('fed_tokens', 'n/a')} "
        f"warm/cold={report.get('warm_over_cold')} fact_ok={report.get('fact_recall_ok')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
