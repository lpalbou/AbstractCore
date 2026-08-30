"""Live proof that native MTP speculation actually accelerates the MLX lane.

Gated behind the usual MLX weight gate plus an MTP-specific one, because this
lane needs BOTH a 15 GB target and its separate drafter checkpoint, which a
plain MLX developer will not have cached:

    ABSTRACTCORE_RUN_MLX_TESTS=1 ABSTRACTCORE_RUN_MTP_TESTS=1 pytest \
        tests/providers/test_mtp_mlx_live.py

Measured on an M5 Max (128 GB) on 2026-08-29, Qwen3.8-27B 4-bit, batch 1:
32.2 -> 53.6 tok/s on a code prompt, with byte-identical output. The speed
assertion below is a LOOSE FLOOR (1.15x), not that ratio: acceptance rate is
prompt-dependent (measured 0.48 on prose vs 1.00 on a repetitive table), and a
test that pins a number will fail on the next machine for no good reason.

The output-equality assertion is the load-bearing one. Speculative decoding is
only ever a speed trade -- the verifier accepts a drafted token exactly when the
target model would have produced it -- so a divergence here means the lane is
returning something the unaccelerated model would not have said.
"""

import os
import time

import pytest

TARGET = os.getenv("ABSTRACTCORE_MTP_TEST_MODEL", "mlx-community/Qwen3.8-27B-4bit")

pytestmark = pytest.mark.slow

# Deterministic, and long enough that the drafter's acceptance rate dominates
# the fixed per-call overhead.
PROMPT = (
    "Write a complete Python class `LRUCache` with get/put in O(1) using a dict "
    "and a doubly linked list. Include docstrings and a short usage example."
)
MAX_TOKENS = 400
MIN_SPEEDUP = 1.15


def _gate():
    if os.getenv("ABSTRACTCORE_RUN_MLX_TESTS", "0") != "1":
        pytest.skip("Set ABSTRACTCORE_RUN_MLX_TESTS=1 to run MLX weight tests")
    if os.getenv("ABSTRACTCORE_RUN_MTP_TESTS", "0") != "1":
        pytest.skip("Set ABSTRACTCORE_RUN_MTP_TESTS=1 to run native-MTP tests")


def _run(llm, **kwargs):
    """One warmed, timed generation. Returns (text, tok/s, metadata).

    `kwargs` are forwarded to both the warmup and the timed call so the graph
    that gets warmed is the graph that gets measured.
    """
    llm.generate("hi", max_tokens=8, thinking=False, **kwargs)  # warm the graph
    t0 = time.time()
    r = llm.generate(
        PROMPT, max_tokens=MAX_TOKENS, temperature=0.0, thinking=False, **kwargs
    )
    elapsed = time.time() - t0
    n = (r.usage or {}).get("output_tokens") or 0
    assert n > 0, "no tokens generated"
    return r.content, n / elapsed, (r.metadata or {}).get("speculation")


@pytest.mark.integration
def test_native_mtp_is_faster_and_says_so_without_changing_the_answer():
    """The control is the SAME provider with the drafter turned off per call.

    An earlier version of this test compared against a separate, non-speculating
    provider -- and that is a two-variable experiment, because enabling
    speculation also switches the MLX lane from mlx-lm to mlx-vlm. Those two
    libraries do not always agree: on Qwen3.5-4B-4bit, greedy decoding of the
    same prompt gives different text under each, with the drafter uninvolved.
    Comparing across them would attribute a library difference to MTP.

    `speculation={"mode": "off"}` per call keeps mlx-vlm and drops only the
    drafter, which isolates exactly one variable -- and that is the comparison
    where MTP must be byte-identical, because speculative decoding accepts a
    drafted token only when the target model would have emitted it anyway.
    """
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=TARGET, speculation={"mode": "native_mtp"})

    control_text, control_tps, control_meta = _run(llm, speculation={"mode": "off"})
    assert control_meta is not None and control_meta["used"] is False, (
        f"the drafter-off control must report used=False; got {control_meta}"
    )

    mtp_text, mtp_tps, mtp_meta = _run(llm)
    assert mtp_meta is not None, "speculation was requested but not reported"
    assert mtp_meta["used"] is True, f"drafter did not run: {mtp_meta}"
    assert mtp_meta["mode"] == "native_mtp"
    assert mtp_meta["draft_kind"] == "mtp"
    assert mtp_meta["drafter"], "no drafter recorded"

    assert mtp_text == control_text, (
        "native MTP changed the output against its OWN runtime's baseline. "
        "Verification accepts a drafted token only when the target model would "
        "have produced it, so a greedy prompt must come back byte-identical.\n"
        f"--- drafter off ---\n{control_text[:400]}\n--- mtp ---\n{mtp_text[:400]}"
    )
    assert mtp_tps > control_tps * MIN_SPEEDUP, (
        f"native MTP was not meaningfully faster: {control_tps:.1f} -> {mtp_tps:.1f} "
        f"tok/s (needed >{MIN_SPEEDUP}x). Check the drafter matches the target."
    )


@pytest.mark.integration
def test_unrequested_speculation_is_never_claimed():
    _gate()
    from abstractcore import create_llm

    llm = create_llm("mlx", model=TARGET)
    _, _, meta = _run(llm)
    assert meta is None, f"unrequested speculation metadata appeared: {meta}"


@pytest.mark.integration
def test_require_acceleration_raises_when_no_drafter_exists():
    """A model with no published drafter must refuse, not quietly run slow."""
    _gate()
    from abstractcore import create_llm
    from abstractcore.providers.speculation import SpeculationUnavailableError

    small = os.getenv(
        "ABSTRACTCORE_MTP_TEST_NO_DRAFTER_MODEL",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
    )
    with pytest.raises(SpeculationUnavailableError) as excinfo:
        create_llm(
            "mlx",
            model=small,
            speculation={"mode": "native_mtp", "require_acceleration": True},
        )
    assert excinfo.value.reason == "no_mtp_drafter_for_model"


@pytest.mark.integration
def test_unhonorable_request_degrades_but_reports_the_reason():
    _gate()
    from abstractcore import create_llm

    small = os.getenv(
        "ABSTRACTCORE_MTP_TEST_NO_DRAFTER_MODEL",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
    )
    llm = create_llm("mlx", model=small, speculation={"mode": "native_mtp"})
    r = llm.generate("Say hello.", max_tokens=20)

    assert r.content.strip(), "degraded lane must still answer"
    spec = (r.metadata or {}).get("speculation")
    assert spec is not None, "a request that could not be honored must still be reported"
    assert spec["requested"] is True
    assert spec["used"] is False
    assert spec["reason"] == "no_mtp_drafter_for_model"
