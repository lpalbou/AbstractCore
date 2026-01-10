import os
import warnings
from typing import Any, Dict, Tuple

import pytest

from abstractcore import BasicSession, create_llm


def _normalize_one_token(text: str) -> str:
    s = (text or "").strip().lower()
    if not s:
        return ""
    token = s.split()[0].strip()
    return token.strip("`\"'.,:;[](){}")


def _run_two_turn_session(*, provider: str, model: str, params: Dict[str, Any]) -> Tuple[str, str]:
    llm = create_llm(provider, model=model, **params)
    session = BasicSession(provider=llm, temperature=params.get("temperature"), seed=params.get("seed"))

    prompt1 = "Pick exactly one option: good, bad, neutral. Reply with the single word only."
    prompt2 = "Repeat the same single word you chose previously. Reply with the single word only."

    r1 = session.generate(prompt1)
    r2 = session.generate(prompt2)

    return _normalize_one_token(r1.content or ""), _normalize_one_token(r2.content or "")


@pytest.mark.integration
@pytest.mark.parametrize(
    "provider,model,params,expect_seed_supported",
    [
        (
            "lmstudio",
            os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507"),
            {
                "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                "temperature": 0.0,
                "seed": int(os.getenv("DETERMINISM_SEED", "42")),
            },
            True,
        ),
        (
            "ollama",
            os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"),
            {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "temperature": 0.0,
                "seed": int(os.getenv("DETERMINISM_SEED", "42")),
            },
            True,
        ),
        (
            "openai",
            os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.0,
                "seed": int(os.getenv("DETERMINISM_SEED", "42")),
            },
            True,
        ),
        (
            "anthropic",
            os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "temperature": 0.0,
                "seed": int(os.getenv("DETERMINISM_SEED", "42")),
            },
            False,
        ),
    ],
)
def test_seed_temperature_two_turn_determinism(provider: str, model: str, params: Dict[str, Any], expect_seed_supported: bool):
    if provider in {"ollama", "lmstudio"} and os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
        pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
    if provider in {"openai", "anthropic"} and os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
        pytest.skip("Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)")
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    # Best-effort detection for providers/models that ignore seed/temperature.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out1 = _run_two_turn_session(provider=provider, model=model, params=params)
            out2 = _run_two_turn_session(provider=provider, model=model, params=params)
        except Exception as e:
            msg = str(e).lower()
            if any(
                keyword in msg
                for keyword in (
                    "connection error",
                    "connecterror",
                    "operation not permitted",
                    "connection refused",
                    "timeout",
                    "not running",
                    "nodename nor servname provided",
                )
            ):
                pytest.skip(f"{provider} not reachable in this environment: {e}")
            raise

    unsupported_msgs = [
        str(w.message).lower()
        for w in caught
        if "seed parameter" in str(w.message).lower() and "not supported" in str(w.message).lower()
        or "temperature parameter" in str(w.message).lower() and "not supported" in str(w.message).lower()
    ]

    if not expect_seed_supported or unsupported_msgs:
        # Contract: warn and continue (do not assert determinism when seed isn't honored).
        assert unsupported_msgs or not expect_seed_supported
        return

    assert out1 == out2, f"Expected deterministic two-turn outputs, got {out1!r} vs {out2!r}"
