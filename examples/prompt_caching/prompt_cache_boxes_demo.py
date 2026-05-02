"""
Prompt cache "box caching" demo.

This is a local/dev script intended to be run on a machine with local models installed.

Run:
  python examples/prompt_caching/prompt_cache_boxes_demo.py

It demonstrates:
- CachedSession (MLX KV delta-only mode when available)
- CachedSession KV mode for HuggingFace transformers (cross-call KV reuse)
- Module-based prefix caches (system/tools) via prompt_cache_prepare_modules()
- GGUF prompt-cache control plane warmup (HuggingFaceProvider GGUF)
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

from abstractcore import CachedSession, create_llm


def _print_json(title: str, payload) -> None:
    print(f"\n== {title} ==")
    try:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        print(payload)


def demo_mlx_cached_session(model: str) -> None:
    print("\n" + "=" * 80)
    print(f"MLX CachedSession demo  model={model}")
    llm = create_llm("mlx", model=model, max_output_tokens=128)

    session = CachedSession(
        provider=llm,
        system_prompt="You are a concise assistant. Reply with exactly one short line.",
        tools=[],
        prompt_cache_strategy="auto",
    )
    print(f"CachedSession.prompt_cache_mode={session.prompt_cache_mode} key={session.prompt_cache_key}")
    _print_json("prompt_cache_capabilities", llm.get_prompt_cache_capabilities().to_dict())

    t0 = time.time()
    r1 = session.generate("Say: OK", max_output_tokens=8)
    t1 = time.time()
    r2 = session.generate("Say: OK2", max_output_tokens=8)
    t2 = time.time()

    print(f"r1: {getattr(r1, 'content', '')!r}  ({(t1 - t0):.2f}s)")
    print(f"r2: {getattr(r2, 'content', '')!r}  ({(t2 - t1):.2f}s)")
    _print_json("prompt_cache_stats", llm.get_prompt_cache_stats())


def demo_hf_transformers_cached_session(model: str) -> None:
    print("\n" + "=" * 80)
    print(f"HuggingFace transformers CachedSession demo  model={model}")
    llm = create_llm("huggingface", model=model, max_output_tokens=128, device="cpu")

    session = CachedSession(
        provider=llm,
        system_prompt="You are a concise assistant. Reply with exactly one short line.",
        tools=[],
        prompt_cache_strategy="auto",
    )
    print(f"CachedSession.prompt_cache_mode={session.prompt_cache_mode} key={session.prompt_cache_key}")
    _print_json("prompt_cache_capabilities", llm.get_prompt_cache_capabilities().to_dict())

    t0 = time.time()
    r1 = session.generate("Say: OK", max_output_tokens=8)
    t1 = time.time()
    r2 = session.generate("Say: OK2", max_output_tokens=8)
    t2 = time.time()

    print(f"r1: {getattr(r1, 'content', '')!r}  ({(t1 - t0):.2f}s)")
    print(f"r2: {getattr(r2, 'content', '')!r}  ({(t2 - t1):.2f}s)")
    _print_json("prompt_cache_stats", llm.get_prompt_cache_stats())


def demo_gguf_module_prefix(model: str) -> None:
    print("\n" + "=" * 80)
    print(f"GGUF module prefix demo  model={model}")
    llm = create_llm("huggingface", model=model, max_output_tokens=96, max_tokens=8192)
    _print_json("prompt_cache_capabilities", llm.get_prompt_cache_capabilities().to_dict())

    ns = f"demo:{uuid.uuid4().hex[:8]}"
    prepared = llm.prompt_cache_prepare_modules(
        namespace=ns,
        modules=[
            {"module_id": "system", "system_prompt": "You are a helpful assistant."},
            {"module_id": "tools", "tools": [{"name": "noop", "description": "No-op", "parameters": {}}]},
        ],
    )
    _print_json("prepared_modules", prepared)

    prefix_key = prepared.get("final_cache_key")
    if not isinstance(prefix_key, str) or not prefix_key.strip():
        raise RuntimeError("prepare_modules did not return a final_cache_key")

    session_key = f"sess:{uuid.uuid4().hex[:8]}"
    llm.prompt_cache_fork(prefix_key, session_key, make_default=True)
    llm.prompt_cache_update(session_key, messages=[{"role": "user", "content": "Say: OK"}])
    _print_json("prompt_cache_stats", llm.get_prompt_cache_stats())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-model", default="lmstudio-community/LFM2.5-1.2B-Instruct-MLX-8bit")
    ap.add_argument("--hf-model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--gguf-model", default="lmstudio-community/Llama-3.2-1B-Instruct-GGUF")
    ap.add_argument("--skip-mlx", action="store_true")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--skip-gguf", action="store_true")
    args = ap.parse_args()

    if not args.skip_mlx:
        try:
            demo_mlx_cached_session(args.mlx_model)
        except Exception as e:
            print(f"MLX demo skipped/failed: {e}")

    if not args.skip_hf:
        try:
            demo_hf_transformers_cached_session(args.hf_model)
        except Exception as e:
            print(f"HuggingFace transformers demo skipped/failed: {e}")

    if not args.skip_gguf:
        try:
            demo_gguf_module_prefix(args.gguf_model)
        except Exception as e:
            print(f"GGUF demo skipped/failed: {e}")


if __name__ == "__main__":
    main()
