"""
Reasoning/thinking control demo across local providers.

This is a local/dev script intended to be run on a machine with local servers/models:
- LM Studio (OpenAI-compatible server)
- Ollama

It exercises `thinking=` (unified reasoning control) and prints any warnings + responses.
"""

from __future__ import annotations

import argparse
import warnings

from abstractcore import create_llm


def _run_one(provider: str, model: str, *, base_url: str | None = None, thinking=None) -> None:
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    llm = create_llm(provider, model=model, max_output_tokens=96, **kwargs)

    print("\n" + "=" * 80)
    print(f"provider={provider} model={model} thinking={thinking!r}")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        resp = llm.generate("Reply with exactly: OK", thinking=thinking)
        for w in rec:
            print(f"warning: {w.category.__name__}: {w.message}")

    print(f"content: {getattr(resp, 'content', '')!r}")
    reasoning = getattr(resp, "reasoning", None)
    if reasoning:
        print(f"reasoning: {reasoning!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmstudio-model", default="openai/gpt-oss-20b")
    ap.add_argument("--lmstudio-base-url", default="http://localhost:1234/v1")
    ap.add_argument("--ollama-model", default="gpt-oss:20b")
    ap.add_argument("--skip-lmstudio", action="store_true")
    ap.add_argument("--skip-ollama", action="store_true")
    ap.add_argument("--thinking", default="minimal", help="e.g. minimal|low|medium|high|xhigh|off|on")
    args = ap.parse_args()

    thinking = args.thinking
    if thinking in {"none", "off"}:
        thinking = "off"
    elif thinking in {"on", "true"}:
        thinking = "on"

    if not args.skip_lmstudio:
        try:
            _run_one("lmstudio", args.lmstudio_model, base_url=args.lmstudio_base_url, thinking=thinking)
        except Exception as e:
            print(f"LM Studio demo skipped/failed: {e}")

    if not args.skip_ollama:
        try:
            _run_one("ollama", args.ollama_model, thinking=thinking)
        except Exception as e:
            print(f"Ollama demo skipped/failed: {e}")


if __name__ == "__main__":
    main()

