#!/usr/bin/env python3
"""
Retry logging + observability demo.

This module used to execute a long demo at import time (which is unsafe under pytest).
It is now split into:
- `main(log_dir=...)`: runnable demo script
- `test_retry_observability_demo(...)`: opt-in pytest wrapper
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def main(*, log_dir: Path) -> None:
    import logging
    import shutil
    from pydantic import BaseModel, field_validator

    from abstractcore import create_llm
    from abstractcore.utils import configure_logging, capture_session
    from abstractcore.structured import StructuredOutputHandler

    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        log_dir=str(log_dir),
        verbatim_enabled=True,
        console_json=False,
        file_json=True,
    )

    class StrictNumber(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def validate_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("Value must be positive integer")
            return v

    # Demo intentionally triggers retries (validation failure) and writes logs.
    llm = create_llm(
        "ollama",
        model="qwen3:4b-instruct",
        base_url="http://localhost:11434",
        timeout=10.0,
    )

    handler = StructuredOutputHandler()
    with capture_session("retry_observability_demo"):
        try:
            handler.generate_structured(
                provider=llm,
                prompt="Give me a negative number like -5",
                response_model=StrictNumber,
            )
        except Exception:
            # Expected: schema validation should fail after retries.
            pass


def test_retry_observability_demo(tmp_path: Path) -> None:
    """Opt-in wrapper to run the retry/observability demo under pytest."""
    if os.getenv("ABSTRACTCORE_RUN_RETRY_OBSERVABILITY_TESTS") != "1":
        pytest.skip("Retry observability demo is opt-in; set ABSTRACTCORE_RUN_RETRY_OBSERVABILITY_TESTS=1")

    if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
        pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")

    try:
        main(log_dir=tmp_path / "retry_observability_logs")
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("connection", "refused", "timeout", "operation not permitted")):
            pytest.skip(f"Ollama not reachable in this environment: {e}")
        raise


if __name__ == "__main__":
    main(log_dir=Path("test_logs"))
