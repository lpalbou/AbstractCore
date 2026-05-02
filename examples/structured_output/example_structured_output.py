#!/usr/bin/env python3
"""
Structured output (schema-bound JSON) in AbstractCore.

What this demonstrates
- Asking an LLM to return data that matches a Pydantic schema.
- Validating that output and (optionally) retrying if it is invalid.
- Keeping the example provider-agnostic: the same code works with Ollama, LM Studio,
  and hosted providers (given the right credentials).

Run
  python examples/structured_output/example_structured_output.py --provider ollama --model llama3.2:3b
  python examples/structured_output/example_structured_output.py --provider lmstudio --model qwen3.5-4b@q4_k_m --base-url http://localhost:1234/v1
"""

from __future__ import annotations

import argparse
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from abstractcore import create_llm
from abstractcore.structured.retry import FeedbackRetry


class User(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=125)
    email: str


class Task(BaseModel):
    title: str
    priority: Literal["high", "medium", "low"]
    estimated_hours: Optional[float] = Field(default=None, ge=0)
    completed: bool = False


class TaskList(BaseModel):
    tasks: List[Task]


def _demo_user(llm) -> None:
    prompt = "Extract user info from: John Smith, 25 years old, john.smith@email.com"
    user = llm.generate(
        prompt,
        response_model=User,
        retry_strategy=FeedbackRetry(max_attempts=3),
        temperature=0.0,
    )
    print("user:", user.model_dump())


def _demo_task_list(llm) -> None:
    prompt = (
        "Turn these into a task list:\n"
        "- Review quarterly financial reports (urgent, ~3 hours)\n"
        "- Book a dentist appointment (not urgent)\n"
        "- Write a 1-page project status update (medium priority)\n"
        "\n"
        "Use priority one of: high, medium, low."
    )
    tasks = llm.generate(
        prompt,
        response_model=TaskList,
        retry_strategy=FeedbackRetry(max_attempts=3),
        temperature=0.0,
    )
    print("tasks:")
    for t in tasks.tasks:
        print(f"- {t.priority}: {t.title} (hours={t.estimated_hours}, completed={t.completed})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama", help="e.g. ollama|lmstudio|openai|anthropic|huggingface|mlx")
    ap.add_argument("--model", default="llama3.2:3b", help="Provider model id")
    ap.add_argument("--base-url", default=None, help="Optional base URL (lmstudio/openai-compatible).")
    args = ap.parse_args()

    llm_kwargs = {"model": args.model}
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url

    llm = create_llm(args.provider, **llm_kwargs)

    print("=" * 80)
    print(f"provider={args.provider} model={args.model}")
    print("Structured output demos")
    print("=" * 80)

    _demo_user(llm)
    print("-" * 80)
    _demo_task_list(llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
