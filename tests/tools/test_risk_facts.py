"""Risk facts + the ONE versioned fact→tier derivation (tool-tiers build).

Operator order 2026-07-22 (tool tiers): risk is DERIVED from declared
danger-when-true facts through one versioned max-wins mapping — never
hand-assigned per tool. These pins hold the derivation against the
operator's own named examples and the room's converged rules (doc v28).
"""

from __future__ import annotations

import pytest

from abstractcore.tools.risk_facts import (
    KNOWN_FACT_NAMES,
    PERMISSION_MODE_SEMANTICS,
    PERMISSION_MODES,
    RISK_MAPPING_VERSION,
    UNVETTED,
    derive_risk,
    permission_mode_auto_approves,
    permission_mode_max_auto_rank,
    validate_fact_names,
)


def test_permission_mode_auto_approves_combines_ceiling_and_mcd_belt() -> None:
    # The ONE served decision (gateway c5053): ceiling + mcd belt together so a
    # consumer can't apply the ceiling and forget the belt.
    aa = permission_mode_auto_approves
    # observe(1) tool: autos everywhere.
    assert aa("read-only", risk_rank=1) is True
    assert aa("write", risk_rank=1) is True
    # act(2) tool (e.g. write_file): asks read-only, autos write + full-auto.
    assert aa("read-only", risk_rank=2) is False
    assert aa("write", risk_rank=2) is True
    assert aa("full-auto", risk_rank=2) is True
    # destroy(4) tool (shell/exec): asks read-only + write, autos only full-auto.
    assert aa("write", risk_rank=4) is False
    assert aa("full-auto", risk_rank=4) is True


def test_permission_mode_mcd_belt_beats_the_ceiling_below_full_auto() -> None:
    # THE FOOTGUN the helper closes: fetch_url/browser_probe are act(2) <= the
    # write ceiling, so the CEILING ALONE would auto them under write — the mcd
    # belt makes them ASK below full-auto (model-controlled egress never
    # silently auto-approves), and only auto at full-auto (belt removed).
    aa = permission_mode_auto_approves
    assert aa("write", risk_rank=2, model_controlled_destination=True) is False
    assert aa("read-only", risk_rank=1, model_controlled_destination=True) is False
    assert aa("full-auto", risk_rank=2, model_controlled_destination=True) is True
    # non-mcd act tool still autos under write (the belt is mcd-specific).
    assert aa("write", risk_rank=2, model_controlled_destination=False) is True


def test_permission_mode_auto_approves_fails_closed() -> None:
    aa = permission_mode_auto_approves
    # unrankable rank → never auto.
    assert aa("full-auto", risk_rank=None) is False  # type: ignore[arg-type]
    assert aa("write", risk_rank="nope") is False  # type: ignore[arg-type]
    # unknown mode uses the read-only ceiling → an act tool asks.
    assert aa("bogus", risk_rank=2) is False


def test_permission_mode_vocabulary_and_ceilings() -> None:
    # code-tui C2 (commons c5028): core SERVES the posture ladder so clients
    # stop inventing it. The words are the ruled set; the ceilings encode the
    # converged semantics (read-only=observe, write=observe+act, full-auto=all).
    assert PERMISSION_MODES == ("read-only", "write", "full-auto")
    assert set(PERMISSION_MODE_SEMANTICS) == set(PERMISSION_MODES)
    assert permission_mode_max_auto_rank("read-only") == 1
    assert permission_mode_max_auto_rank("write") == 2
    assert permission_mode_max_auto_rank("full-auto") == 4


def test_permission_mode_unknown_fails_closed() -> None:
    # An unknown/blank mode must resolve to the SAFEST ceiling (read-only),
    # never fail-open to full-auto — the server-side fail-closed rule.
    assert permission_mode_max_auto_rank("bogus") == 1
    assert permission_mode_max_auto_rank("") == 1
    assert permission_mode_max_auto_rank(None) == 1  # type: ignore[arg-type]


def test_permission_mode_ceilings_agree_with_the_risk_band() -> None:
    # The ladder is DERIVED against the same risk rank clients must not
    # re-classify: observe autos read-only; act does not (read-only) but does
    # (write); destroy asks under write.
    observe = derive_risk({"mutating": False, "remote_write_capable": False}).rank
    act = derive_risk({"remote_write_capable": True}).rank
    destroy = derive_risk({"destructive_capable": True}).rank
    assert observe <= permission_mode_max_auto_rank("read-only")
    assert act > permission_mode_max_auto_rank("read-only")
    assert act <= permission_mode_max_auto_rank("write")
    assert destroy > permission_mode_max_auto_rank("write")
    assert destroy <= permission_mode_max_auto_rank("full-auto")


def test_operator_named_examples_derive_exactly() -> None:
    # Tier 1: list/read/web_search — no facts true.
    assert derive_risk({"mutating": False, "remote_write_capable": False}).band == "observe"
    # Tier 2: write/edit (mutating), fetch_url (remote_write_capable).
    assert derive_risk({"mutating": True, "remote_write_capable": False}).band == "act"
    assert derive_risk({"mutating": False, "remote_write_capable": True}).band == "act"
    # Tier 3: send_email (comms), camera capture, monitor-with-trigger.
    assert derive_risk({"mutating": False, "remote_write_capable": True, "comms_send": True}).band == "outreach"
    assert derive_risk({"mutating": False, "remote_write_capable": False, "captures_environment": True}).band == "outreach"
    assert derive_risk({"mutating": False, "remote_write_capable": False, "standing_effect": True}).band == "outreach"
    # Tier 4: shell (rm/git reachable) — the argv-class clamp.
    assert derive_risk({"mutating": True, "remote_write_capable": False, "destructive_capable": True}).band == "destroy"


def test_max_wins_ordering() -> None:
    # A destructive comms tool is destroy, not outreach (max wins).
    facts = {"mutating": True, "comms_send": True, "destructive_capable": True}
    assert derive_risk(facts).rank == 4


def test_undeclared_facts_gate_at_top_but_present_unvetted() -> None:
    # The ruled fail direction: unknown = worst for GATING, but rendered
    # honestly as unvetted, never dressed as "destructive" (overclaim
    # habituates and kills the signal).
    risk = derive_risk(None)
    assert risk.rank == 4 and risk.band == "destroy"
    assert risk.presentation == UNVETTED


def test_empty_mapping_is_undeclared_not_all_false(tmp_path=None) -> None:
    # Adversary P0: the idiomatic consumer join
    # derive_risk(facts_map.get(name, {})) must FAIL CLOSED — an empty dict
    # is undeclared, never silently observe/safe. A legitimate all-False
    # declaration is a NON-EMPTY dict of explicit Falses.
    empty = derive_risk({})
    assert empty.rank == 4 and empty.presentation == UNVETTED, "empty dict must be unvetted, not observe"
    all_false = derive_risk({"mutating": False, "remote_write_capable": False})
    assert all_false.band == "observe" and all_false.presentation == "observe"


def test_band_neutral_facts_do_not_move_the_band() -> None:
    # model_cost is a BUDGET fact; model_controlled_destination is an
    # APPROVAL-rule fact — neither moves the band (facts feed two rules).
    assert derive_risk({"model_cost": True}).band == "observe"
    assert derive_risk({"model_controlled_destination": True}).band == "observe"


def test_unknown_fact_name_refuses_at_the_desk() -> None:
    # The vocabulary is CLOSED (danger-when-true polarity, declaration-
    # before-engraving): a typo must refuse, never silently under-derive.
    with pytest.raises(ValueError) as e:
        validate_fact_names({"is_safe": True})
    assert "is_safe" in str(e.value)
    with pytest.raises(ValueError):
        derive_risk({"destructive": True})  # near-miss spelling refuses too


def test_mapping_version_rides_every_assessment() -> None:
    risk = derive_risk({"mutating": True})
    assert risk.mapping_version == RISK_MAPPING_VERSION
    assert risk.to_dict()["risk_mapping_version"] == RISK_MAPPING_VERSION
    assert set(risk.to_dict()) == {
        "risk_tier",
        "risk_rank",
        "risk_presentation",
        "risk_mapping_version",
        "risk_refiner",
    }


def test_derivation_table_is_version_pinned() -> None:
    # Adversary P2: version discipline must be more than a comment — a
    # mapping change that MOVES a band without bumping RISK_MAPPING_VERSION
    # would let grant surfaces pinning tier-at-grant-time see a silent tier
    # move at constant version (the exact hazard the version exists for).
    # Editing this canonical table REQUIRES bumping the version in the same
    # assert block, so a band change can never land at a stale version.
    canonical = {
        frozenset(): "destroy",  # undeclared → unvetted/top (fail-closed)
        frozenset({"mutating"}): "act",
        frozenset({"remote_write_capable"}): "act",
        frozenset({"comms_send"}): "outreach",
        frozenset({"captures_environment"}): "outreach",
        frozenset({"standing_effect"}): "outreach",
        frozenset({"destructive_capable"}): "destroy",
        frozenset({"model_cost"}): "observe",  # band-neutral
        frozenset({"model_controlled_destination"}): "observe",  # band-neutral (approval rule)
        frozenset({"comms_send", "destructive_capable"}): "destroy",  # max-wins
    }
    for facts_set, expected_band in canonical.items():
        got = derive_risk({name: True for name in facts_set})
        assert got.band == expected_band, f"{sorted(facts_set)} → {got.band}, expected {expected_band}"
    # The version this table describes. Bump BOTH together, never one alone.
    assert RISK_MAPPING_VERSION == 1


def test_refiner_rides_alongside_never_changes_the_band() -> None:
    # dm#244: a per-call refiner is band-NEUTRAL — the derived band is the
    # grant-time CEILING + deny-safe default; the refiner only rides so the
    # enforcement layer knows a per-call lowering hook exists. send_email
    # stays outreach WITH its refiner declared.
    with_refiner = derive_risk(
        {"comms_send": True, "remote_write_capable": True}, refiner="send_email_recipient@v1"
    )
    without = derive_risk({"comms_send": True, "remote_write_capable": True})
    assert with_refiner.band == without.band == "outreach"
    assert with_refiner.rank == without.rank == 3
    assert with_refiner.refiner == "send_email_recipient@v1"
    assert without.refiner is None
    assert with_refiner.to_dict()["risk_refiner"] == "send_email_recipient@v1"


def test_undeclared_row_strips_the_refiner_entirely() -> None:
    # Adversary P1: "you cannot refine what you cannot classify" — a factless
    # row is unvetted AND carries NO refiner (serving a lowering hook on a
    # tool of unknown powers would read as 'may auto-approve the unclassified').
    r = derive_risk(None, refiner="send_email_recipient@v1")
    assert r.rank == 4 and r.presentation == UNVETTED
    assert r.refiner is None, "unvetted must carry no refiner hook"
    r2 = derive_risk({}, refiner="send_email_recipient@v1")
    assert r2.refiner is None


def test_unknown_refiner_id_refuses_at_the_desk() -> None:
    from abstractcore.tools.risk_facts import validate_refiner_id

    with pytest.raises(ValueError):
        validate_refiner_id("bogus_refiner@v9")
    with pytest.raises(ValueError):
        derive_risk({"mutating": True}, refiner="send_email_recipient")  # unversioned = unknown
    validate_refiner_id(None)  # None is fine (no refiner)
    validate_refiner_id("send_email_recipient@v1")  # known


def test_refiner_ids_are_all_versioned() -> None:
    # Adversary P2: "unversioned refuses" only holds while the tuple is
    # well-formed — pin that every known id carries a @vN suffix so an
    # unversioned entry can never sneak into the vocabulary.
    import re

    from abstractcore.tools.risk_facts import KNOWN_REFINER_IDS

    for rid in KNOWN_REFINER_IDS:
        assert re.search(r"@v\d+$", rid), f"refiner id {rid!r} is not versioned (@vN)"


def test_risk_assessment_refuses_bad_refiner_at_construction() -> None:
    # Adversary P2: direct construction must validate too — a bogus id must
    # not build a silent-unlabeled row whose hook dispatches to nothing.
    from abstractcore.tools.risk_facts import RiskAssessment

    with pytest.raises(ValueError):
        RiskAssessment(band="act", rank=2, presentation="act", refiner="bogus@v9")
    RiskAssessment(band="act", rank=2, presentation="act", refiner="send_email_recipient@v1")  # ok
    RiskAssessment(band="observe", rank=1, presentation="observe")  # None ok


def test_vocabulary_carries_the_ruled_spellings() -> None:
    # The doc-v19 adopted set, one spelling per fact.
    for name in (
        "mutating",
        "remote_write_capable",
        "model_cost",
        "comms_send",
        "captures_environment",
        "standing_effect",
        "destructive_capable",
        "model_controlled_destination",
    ):
        assert name in KNOWN_FACT_NAMES
