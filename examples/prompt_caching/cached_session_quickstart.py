#!/usr/bin/env python3
"""
CachedSession quickstart (prompt caching + file attachments).

What this demonstrates
- Turning prompt caching on/off with `prompt_cache_strategy`.
- Attaching local files once (`attach_files`) so subsequent turns reuse a cached prefix.
- Inspecting provider prompt-cache stats (best-effort; depends on provider capabilities).

Run
  python examples/prompt_caching/cached_session_quickstart.py --provider mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit --cache on
  python examples/prompt_caching/cached_session_quickstart.py --provider mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit --cache off
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from abstractcore import CachedSession, create_llm


def _print_json(title: str, payload: Any) -> None:
    print(f"\n== {title} ==")
    try:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        print(payload)


def _write_tmp_file(root: Path, name: str, content: str) -> Path:
    p = (root / name).resolve()
    p.write_text(content, encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mlx", choices=["mlx", "huggingface"])
    ap.add_argument("--model", required=True, help="Model id (or local path for MLX).")
    ap.add_argument("--cache", default="on", choices=["on", "off"])
    ap.add_argument("--max-output-tokens", type=int, default=400)
    args = ap.parse_args()

    llm_kwargs: Dict[str, Any] = {"model": args.model, "max_output_tokens": int(args.max_output_tokens)}
    if args.provider == "huggingface":
        # Keep it runnable everywhere (CPU is slow but avoids GPU requirements).
        llm_kwargs.setdefault("device", "cpu")
    llm = create_llm(args.provider, **llm_kwargs)

    strategy = "auto" if args.cache == "on" else "off"
    session = CachedSession(
        provider=llm,
        system_prompt="You are a concise assistant. Use the attached files as ground truth.",
        prompt_cache_strategy=strategy,
    )
    print("=" * 88)
    print(f"provider={args.provider} model={args.model}")
    print(f"prompt_cache_strategy={strategy} prompt_cache_mode={session.prompt_cache_mode} key={session.prompt_cache_key}")
    _print_json("prompt_cache_capabilities", llm.get_prompt_cache_capabilities().to_dict())

    # Create two small files to attach (so this demo is self-contained).
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = _write_tmp_file(
            root,
            "A.md",
            "A:\n- The ship is called Caldera.\n- The captain is Idris.\n- The core event is a sensor anomaly in ring 3.\n",
        )
        b = _write_tmp_file(
            root,
            "B.md",
            "B:\n- The ship is called Caldera.\n- The captain is Mara.\n- The core event is a coolant leak in bay 7.\n",
        )

        t0 = time.perf_counter()
        attach = session.attach_files([a, b])
        t1 = time.perf_counter()
        _print_json("attach_files_result", attach)
        print(f"(attach wall time: {(t1 - t0) * 1000.0:.1f}ms)")

        q1 = "What is the key difference between A and B? Answer in 2 bullet points."
        q2 = "Same question, but be even shorter."

        t2 = time.perf_counter()
        r1 = session.generate(q1)
        t3 = time.perf_counter()
        r2 = session.generate(q2)
        t4 = time.perf_counter()

        print("\n== Q1 ==")
        print(q1)
        print("\n== A1 ==")
        print(str(getattr(r1, 'content', '') or '').strip())
        print(f"(wall time: {(t3 - t2):.2f}s)")

        print("\n== Q2 ==")
        print(q2)
        print("\n== A2 ==")
        print(str(getattr(r2, 'content', '') or '').strip())
        print(f"(wall time: {(t4 - t3):.2f}s)")

        _print_json("prompt_cache_stats", llm.get_prompt_cache_stats())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

