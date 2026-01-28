#!/usr/bin/env python3
"""
MLX concurrency benchmark (continuous batching).

This script loads a single MLX model (via `mlx-lm`) and measures how total
throughput changes as you increase the number of *concurrent* generations
served by continuous batching.

Outputs:
- Realtime START / DONE logs per query (optional)
- CSV summary + per-query CSV
- PNG plot (if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _prepared_queries(count: int) -> list[str]:
    topics = [
        "LRU cache",
        "binary search",
        "SQL injection prevention",
        "asyncio task scheduling",
        "unit tests with pytest",
        "typing.Protocol",
        "REST pagination",
        "rate limiting algorithms",
        "retry with exponential backoff",
        "JSON schema validation",
        "Docker multi-stage build",
        "Kubernetes readiness probe",
        "Git rebase strategy",
        "CI pipeline optimization",
        "PostgreSQL index design",
        "Redis cache invalidation",
        "event sourcing",
        "CQRS pattern",
        "idempotency keys",
        "webhook signature verification",
        "OpenAPI spec generation",
        "JWT pitfalls",
        "OAuth PKCE flow",
        "CORS misconfigurations",
        "XSS defense",
        "CSRF defense",
        "RBAC design",
        "feature flags",
        "database migrations",
        "observability metrics",
        "structured logging",
        "distributed tracing",
        "load testing plan",
        "profiling Python",
        "memory leak hunting",
        "vector search",
        "RAG evaluation",
        "prompt caching",
        "KV cache tradeoffs",
        "continuous batching",
        "streaming responses",
        "backpressure handling",
        "circuit breaker",
        "bulkheads pattern",
        "latency SLOs",
        "tail latency",
        "p95 vs p99",
        "canary deploys",
        "blue-green deploys",
        "chaos testing",
    ]

    total = max(1, int(count))
    width = len(str(total))

    prompts: list[str] = []
    for i in range(1, total + 1):
        base = topics[(i - 1) % len(topics)]
        variant = (i - 1) // len(topics)
        if variant <= 0:
            topic = base
        else:
            other = topics[((i * 7) + (variant * 13)) % len(topics)]
            topic = f"{base} / {other} (variant {variant + 1})"

        prompts.append(
            "\n".join(
                [
                    f"Query {i:0{width}d}/{total} — Topic: {topic}",
                    "Write a detailed, practical engineering note with:",
                    "1) A short overview (2-4 paragraphs)",
                    "2) A concrete example in Python (include code)",
                    "3) 8-12 bullet-point gotchas and best practices",
                    "4) A short checklist at the end",
                    "",
                    "Be thorough and keep going until you hit the output token limit.",
                ]
            )
        )
    return prompts


def _safe_preview(text: str, max_chars: int = 60) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    xs = sorted(values)
    k = (len(xs) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    d = k - f
    return float(xs[f] * (1.0 - d) + xs[c] * d)


def _ascii_bar(value: float, max_value: float, width: int = 30) -> str:
    if max_value <= 0:
        return ""
    frac = max(0.0, min(1.0, value / max_value))
    n = int(round(frac * width))
    return "█" * n + " " * (width - n)


def _resolve_model_load_target(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        raise ValueError("--model is required")

    p = Path(raw).expanduser()
    if p.is_dir():
        return str(p)

    lmstudio = Path.home() / ".lmstudio" / "models" / raw
    if lmstudio.is_dir():
        return str(lmstudio)

    # Otherwise assume a Hugging Face repo id (may download if not cached).
    return raw


def _load_prompts_from_file(path: Path) -> list[str]:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Prompts file not found: {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "prompts" in data:
            data = data["prompts"]
        if not isinstance(data, list) or not data:
            raise ValueError(f"Invalid prompts JSON (expected non-empty list): {p}")
        prompts: list[str] = []
        for i, item in enumerate(data, start=1):
            if not isinstance(item, str):
                raise ValueError(f"Invalid prompt at index {i} in {p} (expected string)")
            prompts.append(item)
        return prompts

    raise ValueError(f"Unsupported prompts file type: {p} (expected .json)")


def _parse_concurrency_levels(raw: Optional[list[str]]) -> Optional[list[int]]:
    if not raw:
        return None
    out: list[int] = []
    for item in raw:
        for part in str(item or "").split(","):
            s = part.strip()
            if not s:
                continue
            out.append(int(s))
    return out


def _load_mlx_lm(load_target: str):
    # Suppress Transformers advisory noise (mlx-lm also sets this on import).
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from mlx_lm import load  # type: ignore
    except Exception as e:
        raise RuntimeError(f"mlx-lm is not installed or failed to import: {e}") from e

    # Silence "Fetching" progress bar / noisy logs during load.
    import os as _os
    from contextlib import redirect_stderr, redirect_stdout

    with open(_os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            llm, tokenizer = load(load_target)
    return llm, tokenizer


def _build_prompt_tokens(tokenizer: Any, *, user_prompt: str, system_prompt: Optional[str]) -> list[int]:
    if getattr(tokenizer, "has_chat_template", False) and hasattr(tokenizer, "apply_chat_template"):
        messages: list[dict[str, str]] = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": str(user_prompt or "")})
        toks = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return list(toks)

    # Fallback for tokenizers without chat templates.
    text = ""
    if isinstance(system_prompt, str) and system_prompt.strip():
        text += system_prompt.strip() + "\n\n"
    text += f"user: {user_prompt}\nassistant:"
    try:
        return list(tokenizer.encode(text, add_special_tokens=True))
    except TypeError:
        return list(tokenizer.encode(text))


@dataclass(frozen=True)
class PreparedQuery:
    query_id: int
    prompt: str
    prompt_preview: str
    prompt_tokens: list[int]


@dataclass(frozen=True)
class QueryResult:
    concurrency: int
    query_id: int
    prompt_preview: str
    started_at_s: float
    finished_at_s: float
    duration_s: float
    ttft_s: Optional[float]
    prompt_tokens: int
    output_tokens: int
    tokens_per_s: float
    decode_tokens: int
    decode_s: float
    decode_tokens_per_s: float
    prefill_prompt_tokens_per_s: float
    finish_reason: str


@dataclass(frozen=True)
class RunSummary:
    concurrency: int
    queries: int
    wall_s: float
    total_output_tokens: int
    throughput_tokens_per_s: float
    avg_query_tokens_per_s: float
    median_query_tokens_per_s: float
    p90_query_tokens_per_s: float
    avg_ttft_s: float
    median_ttft_s: float
    p90_ttft_s: float
    avg_decode_tokens_per_s: float
    median_decode_tokens_per_s: float
    p90_decode_tokens_per_s: float


def _prepare_queries(tokenizer: Any, prompts: list[str], *, system_prompt: Optional[str]) -> list[PreparedQuery]:
    out: list[PreparedQuery] = []
    for i, p in enumerate(prompts, start=1):
        tokens = _build_prompt_tokens(tokenizer, user_prompt=p, system_prompt=system_prompt)
        out.append(
            PreparedQuery(
                query_id=i,
                prompt=str(p or ""),
                prompt_preview=_safe_preview(p, max_chars=70),
                prompt_tokens=tokens,
            )
        )
    return out


def run_concurrency_level(
    *,
    llm: Any,
    tokenizer: Any,
    prepared: list[PreparedQuery],
    concurrency: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    prefill_step_size: int,
    verbose: bool,
    progress_interval_s: float,
) -> tuple[RunSummary, list[QueryResult]]:
    try:
        import mlx.core as mx  # type: ignore
        from mlx_lm.generate import BatchGenerator, make_sampler  # type: ignore
    except Exception as e:
        raise RuntimeError(f"MLX runtime imports failed (mlx / mlx-lm): {e}") from e

    # Upstream compatibility: mlx-lm may still call `mx.metal.device_info()` which is deprecated in recent MLX.
    # Patch the deprecated entrypoint to the supported API so the warning is fixed (not silenced).
    try:
        if hasattr(mx, "device_info") and hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
            mx.metal.device_info = mx.device_info  # type: ignore[attr-defined]
    except Exception:
        pass

    if not prepared:
        raise ValueError("No prompts provided")

    # `concurrency` is the configured scheduler capacity (completion_batch_size).
    # It can be larger than the number of total requests (underfilled run).
    concurrency = max(1, int(concurrency))
    max_output_tokens = max(1, int(max_output_tokens))

    # Important: set prefill_batch_size=1 so the scheduler can fill *exact* concurrency.
    # (BatchGenerator only inserts when capacity >= prefill_batch_size.)
    prefill_batch_size = 1

    stop_tokens = set(getattr(tokenizer, "eos_token_ids", set()) or set())
    sampler = make_sampler(temp=float(temperature), top_p=float(top_p))

    generator = BatchGenerator(
        llm,
        stop_tokens=stop_tokens,
        completion_batch_size=int(concurrency),
        prefill_batch_size=int(prefill_batch_size),
        prefill_step_size=int(prefill_step_size),
    )

    # uid -> state
    active: dict[int, dict[str, Any]] = {}
    pending_idx = 0

    started_times: list[float] = []
    finished_times: list[float] = []
    results: list[QueryResult] = []

    total_out_tokens = 0
    done = 0
    total = len(prepared)
    last_progress = time.perf_counter()

    def _print(line: str) -> None:
        print(line, flush=True)

    def _insert_more() -> None:
        nonlocal pending_idx
        slots = concurrency - len(active)
        if slots <= 0 or pending_idx >= total:
            return
        take = min(slots, total - pending_idx)
        batch = prepared[pending_idx : pending_idx + take]
        pending_idx += take

        prompts = [q.prompt_tokens for q in batch]
        max_tokens = [max_output_tokens] * len(batch)
        samplers = [sampler] * len(batch)

        uids = generator.insert(prompts, max_tokens=max_tokens, caches=None, samplers=samplers)
        now = time.perf_counter()
        for uid, q in zip(uids, batch):
            active[int(uid)] = {
                "query_id": q.query_id,
                "preview": q.prompt_preview,
                "prompt_tokens": len(q.prompt_tokens),
                "started_at": now,
                "first_token_at": None,
                "out_tokens": 0,
            }
            started_times.append(now)
            if verbose:
                _print(f"[START]  q={q.query_id:02d} conc={concurrency:>2} :: {q.prompt_preview}")

    try:
        _insert_more()
        while done < total:
            # If active is empty (shouldn't happen), try inserting more.
            if not active:
                _insert_more()
                if not active:
                    break

            # One scheduler step: yields one token per active request.
            responses = generator.next()

            now = time.perf_counter()
            for r in responses:
                uid = int(getattr(r, "uid"))
                st = active.get(uid)
                if st is None:
                    continue

                finish_reason = getattr(r, "finish_reason", None)
                token = getattr(r, "token", None)

                # Time-to-first-token (TTFT): first token observed for this request (including EOS/stop token).
                if isinstance(token, int) and st["first_token_at"] is None:
                    st["first_token_at"] = now
                    if verbose:
                        _print(f"[TTFT]   q={st['query_id']:02d} ttft={(now - st['started_at']):6.2f}s")

                # Count output tokens (exclude EOS/stop token).
                if finish_reason != "stop" and isinstance(token, int):
                    st["out_tokens"] += 1

                if finish_reason is not None:
                    finished_times.append(now)
                    started = float(st["started_at"])
                    duration = max(0.0, now - started)
                    prompt_tokens = int(st.get("prompt_tokens") or 0)
                    out_tokens = int(st["out_tokens"])
                    tps = (out_tokens / duration) if duration > 0 else 0.0
                    ttft = (
                        (float(st["first_token_at"]) - started)
                        if isinstance(st["first_token_at"], (int, float))
                        else None
                    )

                    # "Decode" metrics (roughly: time after first token is observed).
                    # We attribute the first output token to the prefill/TTFT phase for clarity.
                    decode_s = max(0.0, duration - (float(ttft) if isinstance(ttft, float) else duration))
                    decode_tokens = max(out_tokens - 1, 0) if out_tokens > 0 else 0
                    decode_tps = (decode_tokens / decode_s) if decode_s > 0 else 0.0

                    prefill_prompt_tps = (prompt_tokens / float(ttft)) if isinstance(ttft, float) and ttft > 0 else 0.0

                    results.append(
                        QueryResult(
                            concurrency=int(concurrency),
                            query_id=int(st["query_id"]),
                            prompt_preview=str(st["preview"]),
                            started_at_s=started,
                            finished_at_s=float(now),
                            duration_s=float(duration),
                            ttft_s=float(ttft) if isinstance(ttft, (int, float)) else None,
                            prompt_tokens=int(prompt_tokens),
                            output_tokens=out_tokens,
                            tokens_per_s=float(tps),
                            decode_tokens=int(decode_tokens),
                            decode_s=float(decode_s),
                            decode_tokens_per_s=float(decode_tps),
                            prefill_prompt_tokens_per_s=float(prefill_prompt_tps),
                            finish_reason=str(finish_reason),
                        )
                    )

                    total_out_tokens += out_tokens
                    done += 1
                    if verbose:
                        ttft_part = f" ttft={ttft:5.2f}s" if isinstance(ttft, float) else ""
                        _print(
                            f"[DONE]   q={st['query_id']:02d} dur={duration:6.2f}s out={out_tokens:5d} "
                            f"tps={tps:8.2f}{ttft_part} finish={finish_reason}"
                        )
                    active.pop(uid, None)

            _insert_more()

            if progress_interval_s > 0 and (time.perf_counter() - last_progress) >= progress_interval_s:
                last_progress = time.perf_counter()
                elapsed = (time.perf_counter() - min(started_times)) if started_times else 0.0
                throughput_so_far = (total_out_tokens / elapsed) if elapsed > 0 else 0.0
                _print(
                    f"[PROGRESS] conc={concurrency:>2} done={done:>2}/{total} "
                    f"elapsed={elapsed:6.2f}s throughput_so_far={throughput_so_far:8.2f} tok/s"
                )
    finally:
        try:
            generator.close()
        except Exception:
            pass

    wall = (max(finished_times) - min(started_times)) if started_times and finished_times else 0.0
    throughput = (total_out_tokens / wall) if wall > 0 else 0.0
    per_q_tps = [r.tokens_per_s for r in results]
    ttfts = [r.ttft_s for r in results if isinstance(r.ttft_s, (int, float))]
    decode_tps = [r.decode_tokens_per_s for r in results]

    summary = RunSummary(
        concurrency=int(concurrency),
        queries=len(prepared),
        wall_s=float(wall),
        total_output_tokens=int(total_out_tokens),
        throughput_tokens_per_s=float(throughput),
        avg_query_tokens_per_s=float(statistics.mean(per_q_tps) if per_q_tps else 0.0),
        median_query_tokens_per_s=float(statistics.median(per_q_tps) if per_q_tps else 0.0),
        p90_query_tokens_per_s=float(_percentile(per_q_tps, 90.0) if per_q_tps else 0.0),
        avg_ttft_s=float(statistics.mean(ttfts) if ttfts else 0.0),
        median_ttft_s=float(statistics.median(ttfts) if ttfts else 0.0),
        p90_ttft_s=float(_percentile(ttfts, 90.0) if ttfts else 0.0),
        avg_decode_tokens_per_s=float(statistics.mean(decode_tps) if decode_tps else 0.0),
        median_decode_tokens_per_s=float(statistics.median(decode_tps) if decode_tps else 0.0),
        p90_decode_tokens_per_s=float(_percentile(decode_tps, 90.0) if decode_tps else 0.0),
    )
    return summary, results


def _write_csv(
    *,
    out_dir: Path,
    run_summaries: list[RunSummary],
    per_query_results: list[QueryResult],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_ts()
    summary_path = out_dir / f"mlx_concurrency_summary_{ts}.csv"
    per_query_path = out_dir / f"mlx_concurrency_per_query_{ts}.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "concurrency",
                "queries",
                "wall_s",
                "total_output_tokens",
                "throughput_tokens_per_s",
                "avg_query_tokens_per_s",
                "median_query_tokens_per_s",
                "p90_query_tokens_per_s",
                "avg_ttft_s",
                "median_ttft_s",
                "p90_ttft_s",
                "avg_decode_tokens_per_s",
                "median_decode_tokens_per_s",
                "p90_decode_tokens_per_s",
            ]
        )
        for s in run_summaries:
            w.writerow(
                [
                    s.concurrency,
                    s.queries,
                    f"{s.wall_s:.6f}",
                    s.total_output_tokens,
                    f"{s.throughput_tokens_per_s:.6f}",
                    f"{s.avg_query_tokens_per_s:.6f}",
                    f"{s.median_query_tokens_per_s:.6f}",
                    f"{s.p90_query_tokens_per_s:.6f}",
                    f"{s.avg_ttft_s:.6f}",
                    f"{s.median_ttft_s:.6f}",
                    f"{s.p90_ttft_s:.6f}",
                    f"{s.avg_decode_tokens_per_s:.6f}",
                    f"{s.median_decode_tokens_per_s:.6f}",
                    f"{s.p90_decode_tokens_per_s:.6f}",
                ]
            )

    with per_query_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "concurrency",
                "query_id",
                "prompt_preview",
                "prompt_tokens",
                "duration_s",
                "ttft_s",
                "output_tokens",
                "tokens_per_s",
                "decode_tokens",
                "decode_s",
                "decode_tokens_per_s",
                "prefill_prompt_tokens_per_s",
                "finish_reason",
            ]
        )
        for r in per_query_results:
            w.writerow(
                [
                    r.concurrency,
                    r.query_id,
                    r.prompt_preview,
                    r.prompt_tokens,
                    f"{r.duration_s:.6f}",
                    "" if r.ttft_s is None else f"{r.ttft_s:.6f}",
                    r.output_tokens,
                    f"{r.tokens_per_s:.6f}",
                    r.decode_tokens,
                    f"{r.decode_s:.6f}",
                    f"{r.decode_tokens_per_s:.6f}",
                    f"{r.prefill_prompt_tokens_per_s:.6f}",
                    r.finish_reason,
                ]
            )

    return summary_path, per_query_path


def _maybe_plot(
    *,
    out_dir: Path,
    run_summaries: list[RunSummary],
    per_query_results: list[QueryResult],
    title: str,
) -> Optional[Path]:
    if not run_summaries:
        return None

    try:
        mpl_cache = out_dir / "mpl_cache"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    conc_levels = [s.concurrency for s in run_summaries]
    throughput = [s.throughput_tokens_per_s for s in run_summaries]
    avg_q = [s.avg_query_tokens_per_s for s in run_summaries]

    xs: list[int] = []
    ys: list[float] = []
    for r in per_query_results:
        xs.append(int(r.concurrency))
        ys.append(float(r.tokens_per_s))

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title("Per-query output tokens/sec (all queries) + mean")
    ax1.scatter(xs, ys, s=10, alpha=0.35, linewidths=0)
    ax1.plot(conc_levels, avg_q, marker="o", color="tab:orange", label="Mean per-query tok/s")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Output tok/s")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title("Overall throughput (sum output tokens / wall time)")
    ax2.plot(conc_levels, throughput, marker="o", color="tab:blue", label="Throughput tok/s")
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("Throughput tok/s")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="best")

    out_path = out_dir / f"mlx_concurrency_plot_{_now_ts()}.png"
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    with contextlib.suppress(Exception):
        plt.close(fig)
    return out_path


def _print_run_table(run_summaries: list[RunSummary]) -> None:
    if not run_summaries:
        return
    max_thr = max((s.throughput_tokens_per_s for s in run_summaries), default=0.0)
    print("\nResults (overall throughput):")
    print("  conc | wall(s) | total_out_tok | throughput(tok/s) | avg_query(tok/s) | avg_ttft(s) | avg_decode(tok/s)")
    print("  -----+---------+---------------+------------------+-----------------+------------+-----------------")
    for s in run_summaries:
        bar = _ascii_bar(s.throughput_tokens_per_s, max_thr, width=24)
        print(
            f"  {s.concurrency:>4} | {s.wall_s:>7.2f} | {s.total_output_tokens:>13} | "
            f"{s.throughput_tokens_per_s:>16.2f} | {s.avg_query_tokens_per_s:>15.2f} | "
            f"{s.avg_ttft_s:>10.2f} | {s.avg_decode_tokens_per_s:>15.2f}  {bar}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLX concurrency via continuous batching (mlx-lm).")
    parser.add_argument("--model", required=True, help="MLX model id or local path (supports LM Studio cache ids)")
    parser.add_argument("--queries", type=int, default=50, help="How many queries to run (default: 50)")
    parser.add_argument(
        "--prompts-file",
        default="",
        help="Optional JSON file containing a stable prompt set (list[str] or {prompts:[...]}).",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency for single run (ignored with --autorun)")
    parser.add_argument(
        "--concurrency-levels",
        nargs="+",
        default=None,
        help="Custom concurrency sweep, e.g. --concurrency-levels 1 5 10 20 40 (commas also allowed).",
    )
    parser.add_argument("--autorun", action="store_true", help="Run sweep at 1,5,10,20,30,40,50")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Max output tokens per query")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p nucleus sampling")
    parser.add_argument("--prefill-step-size", type=int, default=2048, help="Prefill step size (mlx-lm)")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt applied to all queries")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle query order")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Seed for --shuffle (reproducible order)")
    parser.add_argument("--out-dir", default=str(Path("test_results") / "mlx_concurrency"), help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print per-query START/TTFT/DONE lines")
    parser.add_argument("--progress-interval", type=float, default=0.0, help="Print progress every N seconds")
    args = parser.parse_args()

    total_queries = max(1, int(args.queries))
    if isinstance(args.prompts_file, str) and args.prompts_file.strip():
        all_prompts = _load_prompts_from_file(Path(args.prompts_file.strip()))
        if total_queries > len(all_prompts):
            raise ValueError(
                f"--queries={total_queries} exceeds prompts-file size ({len(all_prompts)}): {args.prompts_file}"
            )
        prompts = list(all_prompts[:total_queries])
    else:
        prompts = _prepared_queries(total_queries)

    if args.shuffle:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(prompts)

    load_target = _resolve_model_load_target(args.model)
    llm, tokenizer = _load_mlx_lm(load_target)
    prepared = _prepare_queries(tokenizer, prompts, system_prompt=args.system_prompt)

    total_requests = len(prepared)

    requested_levels = _parse_concurrency_levels(args.concurrency_levels)
    if requested_levels:
        seen: set[int] = set()
        conc_levels: list[int] = []
        for c0 in requested_levels:
            if not isinstance(c0, int):
                continue
            if c0 < 1:
                continue
            c = int(c0)
            if c not in seen:
                seen.add(c)
                conc_levels.append(c)
        if not conc_levels:
            conc_levels = [max(1, int(args.concurrency))]
        verbose = bool(args.verbose)
    elif args.autorun:
        conc_levels = [1, 5, 10, 20, 30, 40, 50]
        conc_levels = [c for c in conc_levels if c <= total_requests]
        verbose = bool(args.verbose)
    else:
        conc_levels = [max(1, int(args.concurrency))]
        # Single-run mode: default to per-query START/TTFT/DONE logs.
        verbose = True

    max_requested = max(conc_levels) if conc_levels else 0
    if max_requested > total_requests:
        print(
            f"[WARN] max requested concurrency={max_requested} but queries={total_requests}; "
            f"runs will be underfilled (at most {total_requests} requests active).",
            flush=True,
        )

    run_summaries: list[RunSummary] = []
    per_query_results: list[QueryResult] = []

    for c in conc_levels:
        print(f"\n=== Running concurrency={c} queries={len(prepared)} max_out={args.max_output_tokens} ===", flush=True)
        summary, per_q = run_concurrency_level(
            llm=llm,
            tokenizer=tokenizer,
            prepared=prepared,
            concurrency=c,
            max_output_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            prefill_step_size=int(args.prefill_step_size),
            verbose=verbose,
            progress_interval_s=float(args.progress_interval),
        )
        run_summaries.append(summary)
        per_query_results.extend(per_q)

        print(
            f"=== Summary conc={c} wall={summary.wall_s:.2f}s total_out={summary.total_output_tokens} "
            f"throughput={summary.throughput_tokens_per_s:.2f} tok/s avg_query={summary.avg_query_tokens_per_s:.2f} tok/s "
            f"avg_ttft={summary.avg_ttft_s:.2f}s p90_ttft={summary.p90_ttft_s:.2f}s "
            f"avg_decode={summary.avg_decode_tokens_per_s:.2f} tok/s p90_decode={summary.p90_decode_tokens_per_s:.2f} tok/s ===",
            flush=True,
        )

    _print_run_table(run_summaries)

    out_dir = Path(args.out_dir)
    summary_csv, per_query_csv = _write_csv(out_dir=out_dir, run_summaries=run_summaries, per_query_results=per_query_results)
    plot_path = _maybe_plot(
        out_dir=out_dir,
        run_summaries=run_summaries,
        per_query_results=per_query_results,
        title=f"MLX Concurrency Benchmark — model={args.model} max_out={args.max_output_tokens}",
    )

    print("\nArtifacts:")
    print(f"- {summary_csv}")
    print(f"- {per_query_csv}")
    print(f"- {plot_path}" if plot_path is not None else "- Plot skipped (matplotlib unavailable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
