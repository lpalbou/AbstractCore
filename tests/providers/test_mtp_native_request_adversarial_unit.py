"""Independent adversarial tests for the native embedded-MTP public contract.

No model, MLX runtime, server, or network is used.  In particular, these tests
must fail if a requested depth is silently truncated or only reported rather
than being sent to the actual decoding lane.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.speculation import (
    SpeculationRequest,
    SpeculationUnavailableError,
    normalize_speculation_request,
    resolve_speculation_request,
)


@pytest.mark.parametrize("depth", [True, False, 2.9, float("nan"), float("inf"), [], {}])
def test_depth_never_silently_changes_type_or_value(depth):
    with pytest.raises(ValueError):
        normalize_speculation_request({"num_draft_tokens": depth})


@pytest.mark.parametrize("strict", ["false", "true", 0, 1, [], {}])
def test_strict_mode_does_not_use_python_truthiness(strict):
    with pytest.raises(ValueError):
        normalize_speculation_request({"require_acceleration": strict})


@pytest.mark.parametrize("drafter", [3, [], ["publisher/checkpoint"], {}, {"repo": "x"}])
def test_drafter_identifier_is_not_arbitrary_object_stringification(drafter):
    with pytest.raises(ValueError):
        normalize_speculation_request({"drafter": drafter})


@pytest.mark.parametrize("depth", [1, 3, 5, "5"])
def test_valid_positive_depth_is_preserved(depth):
    request = normalize_speculation_request(
        {"mode": "native_mtp", "num_draft_tokens": depth, "require_acceleration": True}
    )
    assert request.num_draft_tokens == int(depth)
    assert request.require_acceleration is True


def test_shared_resolver_inherits_only_omitted_fields_and_preserves_explicit_off():
    default = SpeculationRequest(mode="native_mtp", drafter="repo/head", num_draft_tokens=3, require_acceleration=True)
    effective = resolve_speculation_request(default, {"num_draft_tokens": 5})
    assert effective == SpeculationRequest(mode="native_mtp", drafter="repo/head", num_draft_tokens=5, require_acceleration=True)
    assert resolve_speculation_request(default, {"require_acceleration": False}).require_acceleration is False
    assert resolve_speculation_request(default, False).enabled is False
    assert resolve_speculation_request(default, None) is default
    assert default.num_draft_tokens == 3


def _provider():
    provider = MLXProvider.__new__(MLXProvider)
    provider.logger = Mock()
    provider.model = "publisher/Qwen3.8-Flash-Next-Q4-mtp"
    provider.model_capabilities = {}
    provider._speculation_request = SpeculationRequest(
        mode="native_mtp", num_draft_tokens=3, require_acceleration=True
    )
    provider._mtp_drafter = SimpleNamespace()
    provider._mtp_kind = "qwen4_exp_mtp"
    provider._mtp_processor = object()
    provider._mtp_drafter_id = provider.model
    provider._mtp_block_size = 3
    provider._mtp_last_used = None
    provider._mtp_outcome_at_load = None
    provider._mtp_call_outcome = None
    provider._mtp_call_disabled = False
    provider._mtp_reason = None
    return provider


def _requested_depth(provider, request):
    provider._apply_per_call_speculation(request)
    return provider._mtp_kwargs(None).get("draft_block_size")


def test_depth_is_a_per_call_override_and_never_latches_into_the_default():
    provider = _provider()
    assert _requested_depth(provider, None) == 4
    assert _requested_depth(provider, {"num_draft_tokens": 5}) == 6
    assert provider._mtp_block_size == 3, "per-call depth overwrote the loaded default"
    assert _requested_depth(provider, {"mode": "off"}) is None
    assert _requested_depth(provider, None) == 4
    assert _requested_depth(provider, {"num_draft_tokens": 1}) == 2
    assert _requested_depth(provider, None) == 4


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_qwen4_verifier_width_includes_one_seed_in_addition_to_requested_drafts(depth):
    provider = _provider()
    provider._native_qwen4 = SimpleNamespace()
    assert _requested_depth(provider, {"num_draft_tokens": depth}) == depth + 1
    assert provider.speculation_status()["effective_draft_tokens"] == depth


def test_execution_telemetry_uses_counter_delta_not_lifetime_totals():
    provider = _provider()
    provider._apply_per_call_speculation({"num_draft_tokens": 5})
    draft = provider._mtp_drafter
    draft.speculative_total_rounds = 10
    draft.speculative_total_accepted = 20
    draft.speculative_total_drafted = 40
    before = provider._mtp_stats_snapshot()
    provider._mtp_record_execution(before)
    assert provider.speculation_status()["last_call_used"] is False
    draft.speculative_total_rounds += 2
    draft.speculative_total_accepted += 4
    draft.speculative_total_drafted += 10
    provider._mtp_record_execution(before)
    status = provider.speculation_status()
    assert status["last_call_used"] is True
    assert status["stats"] == {"rounds": 2, "accepted_tokens": 4, "drafted_tokens": 10, "acceptance_rate": 0.4}
    metadata = provider._mtp_outcome(status["last_call_used"]).to_metadata()
    assert metadata["used"] is True
    assert metadata["num_draft_tokens"] == 5
    provider._apply_per_call_speculation(False)
    assert provider._mtp_stats_snapshot() is None
    assert provider.speculation_status()["last_call_used"] is False


def test_preparing_drafter_kwargs_is_not_evidence_the_drafter_ran():
    provider = _provider()
    provider._apply_per_call_speculation(None)
    kwargs = provider._mtp_kwargs(None)
    assert kwargs["draft_model"] is provider._mtp_drafter  # Non-vacuity.
    assert provider.speculation_status()["last_call_used"] is not True, (
        "constructing kwargs claimed actual execution before the generator ran"
    )


def test_each_call_starts_without_a_stale_used_claim():
    provider = _provider()
    provider._mtp_last_used = True  # A previous call really drafted.
    provider._apply_per_call_speculation({"num_draft_tokens": 5})
    assert provider.speculation_status()["last_call_used"] is not True


def test_invalid_per_call_depth_fails_before_activating_drafter():
    provider = _provider()
    with pytest.raises(ValueError):
        provider._apply_per_call_speculation({"num_draft_tokens": 2.7})
    assert provider._mtp_block_size == 3


def test_a_strict_drafter_switch_cannot_be_faked_by_depth_only():
    provider = _provider()
    with pytest.raises(SpeculationUnavailableError):
        provider._apply_per_call_speculation(
            {
                "drafter": "someone/other-model",
                "num_draft_tokens": 5,
                "require_acceleration": True,
            }
        )
    assert provider._mtp_drafter_id == provider.model
    assert provider._mtp_block_size == 3


def test_disabling_one_call_does_not_erase_the_loaded_head():
    provider = _provider()
    head = provider._mtp_drafter
    provider._apply_per_call_speculation(False)
    assert provider._mtp_kwargs(None) == {}
    assert provider._mtp_drafter is head
    provider._apply_per_call_speculation({"num_draft_tokens": 5})
    assert provider._mtp_kwargs(None)["draft_model"] is head
    assert provider._mtp_kwargs(None)["draft_block_size"] == 6


def test_native_decoder_typeerror_is_not_retried_without_request_constraints():
    provider = _provider()
    provider.llm = object()
    provider.tokenizer = SimpleNamespace(encode=lambda text: list(text))
    provider._build_mlx_sampler = Mock(return_value=None)
    provider._postprocess_generated_text = lambda text, **_: (text, None)
    provider._calculate_usage = lambda prompt, text: {
        "input_tokens": len(prompt), "output_tokens": len(text)
    }
    original = TypeError("native cache transaction rejected the tensor layout")
    provider.generate_fn = Mock(side_effect=original)

    with pytest.raises(TypeError) as raised:
        provider._single_generate("hi", max_tokens=8, temperature=0.0, top_p=1.0)

    assert raised.value is original
    assert provider.generate_fn.call_count == 1, (
        "a decoding error retried with no max_tokens/sampler and could synthesize an answer"
    )
