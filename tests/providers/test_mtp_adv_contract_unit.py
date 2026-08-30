"""Adversarial scoreboard for the speculation request/response contract.

The whole value of `speculation=` is a promise about HONESTY, not about speed:
the caller cannot see whether a drafter ran, so every one of these tests asks the
same question from a different angle -- *can the lane claim acceleration it did
not deliver?*

Deliberately mechanical: no timings, no model loads, no network. Every assertion
is on a decision, a reason slug, or an exception type.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from abstractcore.providers.speculation import (
    SPECULATION_MODES,
    SpeculationOutcome,
    SpeculationRequest,
    SpeculationUnavailableError,
    capability_speculation,
    normalize_speculation_request,
    unavailable,
)


class _RecordingLogger:
    """Captures warnings without touching global logging state."""

    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.infos: List[str] = []

    def warning(self, msg: str, *a: Any, **k: Any) -> None:
        self.warnings.append(str(msg))

    def info(self, msg: str, *a: Any, **k: Any) -> None:
        self.infos.append(str(msg))

    def debug(self, msg: str, *a: Any, **k: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# The load-bearing invariant: `used` means "a drafter ran", full stop.
# ---------------------------------------------------------------------------


def test_serialized_used_flag_always_equals_the_object_truth():
    """`to_metadata()` must not be able to disagree with the outcome object.

    ATTACK: `SpeculationOutcome.details` is merged into the metadata dict, and
    the merge happens LAST. A `details` payload carrying the contract's own key
    names therefore rewrites them on the way out -- so an outcome that records
    `used=False, reason='mlx_vlm_missing'` can serialize as `used=True`. Nothing
    in the lane has to be malicious for this to bite: any future `details` key
    that happens to collide silently corrupts the one field callers trust.

    The class docstring already states the rule ("`used` ... must never be set
    from a request, a capability entry, or an import check -- only from the lane
    that did the work"); this test holds the serializer to it.
    """
    poisoned = SpeculationOutcome(
        requested=True,
        mode="native_mtp",
        used=False,
        reason="mlx_vlm_missing",
        details={
            "used": True,
            "requested": False,
            "mode": "off",
            "reason": "everything is fine",
        },
    )
    meta = poisoned.to_metadata()

    assert meta["used"] is poisoned.used, (
        "details[] overrode the `used` flag: the response metadata claims "
        f"used={meta['used']!r} while the outcome records used={poisoned.used!r}. "
        "Merge `details` FIRST, or drop reserved keys from it, so the contract "
        "fields always win."
    )
    assert meta["requested"] is poisoned.requested
    assert meta["mode"] == poisoned.mode
    assert meta.get("reason") == poisoned.reason


def test_unavailable_never_reports_used_true():
    """Every "could not accelerate" path must serialize used=False."""
    request = SpeculationRequest(mode="native_mtp")
    for reason in (
        "mlx_vlm_missing",
        "no_mtp_drafter_for_model",
        "mtp_drafter_load_failed",
        "speculation_not_applied",
    ):
        outcome = unavailable(request, reason, f"cannot: {reason}", logger=None)
        assert outcome.used is False
        assert outcome.to_metadata()["used"] is False
        assert outcome.requested is True
        assert outcome.to_metadata()["reason"] == reason


def test_default_outcome_is_not_a_claim_of_acceleration():
    """A bare outcome (nothing requested) must not read as `used`."""
    meta = SpeculationOutcome().to_metadata()
    assert meta["used"] is False
    assert meta["requested"] is False
    assert meta["mode"] == "off"


# ---------------------------------------------------------------------------
# Warn vs raise: never silently degrade.
# ---------------------------------------------------------------------------


def test_unhonorable_request_warns_with_a_named_reason():
    logger = _RecordingLogger()
    outcome = unavailable(
        SpeculationRequest(mode="native_mtp"),
        "mlx_vlm_missing",
        "native MTP on MLX runs through mlx-vlm",
        logger=logger,
    )
    assert len(logger.warnings) == 1, "a dropped speculation request must WARN"
    assert "mlx_vlm_missing" in logger.warnings[0], (
        "the warning must carry the machine-readable reason slug, not just prose"
    )
    assert outcome.reason == "mlx_vlm_missing"


def test_require_acceleration_raises_instead_of_warning():
    logger = _RecordingLogger()
    with pytest.raises(SpeculationUnavailableError) as excinfo:
        unavailable(
            SpeculationRequest(mode="native_mtp", require_acceleration=True),
            "mlx_vlm_missing",
            "native MTP on MLX runs through mlx-vlm",
            logger=logger,
        )
    assert excinfo.value.reason == "mlx_vlm_missing", (
        "the exception must expose the same slug callers would have been warned "
        "with, so they can branch on the cause instead of parsing prose"
    )
    assert logger.warnings == [], (
        "strict mode must RAISE, not warn-and-raise: a warning here would make "
        "the failure look survivable in logs"
    )


def test_require_acceleration_is_inert_when_speculation_is_off():
    """`require_acceleration` must not raise on a lane nobody asked to speed up."""
    for request in (
        None,
        SpeculationRequest(mode="off", require_acceleration=True),
    ):
        outcome = unavailable(request, "whatever", "message", logger=None)
        assert outcome.used is False
        assert outcome.requested is False


# ---------------------------------------------------------------------------
# Request normalization: a typo must never become a silent no-op.
# ---------------------------------------------------------------------------


def test_none_and_explicit_off_are_distinguishable():
    """`None` = no opinion; `False` = pin it off. They must not collapse."""
    assert normalize_speculation_request(None) is None
    off = normalize_speculation_request(False)
    assert off is not None and off.enabled is False


@pytest.mark.parametrize(
    "value",
    [
        {"nmode": "native_mtp"},  # typo'd key
        {"mode": "draft_model"},  # a mode v1 does not implement
        {"mode": "native_mtp", "num_draft_tokens": 0},
        {"mode": "native_mtp", "num_draft_tokens": -3},
        {"mode": "native_mtp", "num_draft_tokens": "three"},
        {"mode": "native_mtp", "drafter": "   "},
        "turbo",
        42,
        object(),
    ],
)
def test_malformed_requests_raise_rather_than_silently_disabling(value):
    with pytest.raises(ValueError):
        normalize_speculation_request(value)


def test_accepted_request_shapes_round_trip():
    assert normalize_speculation_request(True).mode == "native_mtp"
    assert normalize_speculation_request("mtp").mode == "native_mtp"
    assert normalize_speculation_request("off").mode == "off"

    request = normalize_speculation_request(
        {
            "mode": "native_mtp",
            "drafter": "mlx-community/Qwen3.8-27B-MTP-4bit",
            "num_draft_tokens": 3,
            "require_acceleration": True,
        }
    )
    assert request.mode == "native_mtp"
    assert request.drafter == "mlx-community/Qwen3.8-27B-MTP-4bit"
    assert request.num_draft_tokens == 3
    assert request.require_acceleration is True
    assert request.enabled is True


def test_modes_are_exactly_the_two_v1_ships():
    """Guards against a `draft_model` mode reappearing.

    mlx-lm cannot run classic draft-model speculation for this architecture at
    all -- it raises "Speculative decoding requires a trimmable prompt cache
    (got {'ArraysCache'})" because the Qwen3.5/3.8 hybrids keep a linear-attention
    cache that cannot be rewound. A mode that always fails is worse than absent.
    """
    assert set(SPECULATION_MODES) == {"native_mtp", "off"}


# ---------------------------------------------------------------------------
# Capability lookup must demand evidence, not a plausible-looking block.
# ---------------------------------------------------------------------------


def test_capability_block_requires_native_mtp_to_be_true():
    """A runtime entry without `native_mtp: true` is a "no", not a "maybe"."""
    caps: Dict[str, Any] = {
        "speculation": {
            "native_mtp": False,
            "runtimes": {"mlx": {"mode": "drafter_repo", "drafter": "some/repo"}},
        }
    }
    assert capability_speculation(caps, "mlx") is None


def test_capability_lookup_is_per_runtime():
    """A GGUF-only entry must not hand the MLX lane a drafter it does not have.

    The two Qwen3.6 `*-MTP-GGUF` entries are exactly this case: the head exists
    for llama.cpp and LM Studio, and mlx-community publishes no drafter for
    them. Leaking a llama_cpp block to the mlx lane would invent one.
    """
    caps = {
        "speculation": {
            "native_mtp": True,
            "mtp_layers": 1,
            "runtimes": {
                "llama_cpp": {"mode": "embedded"},
                "lmstudio": {"mode": "load_flag"},
            },
        }
    }
    assert capability_speculation(caps, "llama_cpp") is not None
    assert capability_speculation(caps, "mlx") is None, (
        "no MLX drafter is published for the Qwen3.6 MTP GGUFs -- the lookup "
        "must return None rather than falling back to another runtime's block"
    )


def test_capability_lookup_survives_junk_input():
    for caps in (None, {}, {"speculation": "yes"}, {"speculation": {"native_mtp": True}}):
        assert capability_speculation(caps, "mlx") is None
