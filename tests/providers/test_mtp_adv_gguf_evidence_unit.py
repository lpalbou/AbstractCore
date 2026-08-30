"""MTP detection must read the GGUF, never the filename.

The heuristic this file exists to keep dead was `if "mtp" in model_lower:` in the
HuggingFace GGUF lane. It is wrong in BOTH directions, and both directions occur
on this machine:

  * MISS  -- `Qwen3.8-27B-Q4_K_M.gguf` carries `qwen35.nextn_predict_layers=1`
             and `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`,
             with no "mtp" anywhere in the filename.
  * FIRE  -- unsloth ships the Qwen3.6 heads in repos named `*-MTP-GGUF`, so a
             path can shout MTP while the file carries no nextn anything. The
             local `Qwen3.6-27B-Q4_K_M.gguf` has exactly zero nextn tensors.

So the fixtures below deliberately decouple name from content: a file named like
the innocent one that IS MTP, and a file named like the loud one that IS NOT.

Hermetic: the GGUFs are synthesized (under 1 KB each), no weights, no network.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from abstractcore.utils.model_cache import (
    read_gguf_architecture,
    read_gguf_mtp_layers,
)

_BUILDER_PATH = Path(__file__).parent / "fixtures" / "mtp_adv_gguf_builder.py"
_spec = importlib.util.spec_from_file_location("mtp_adv_gguf_builder", _BUILDER_PATH)
assert _spec and _spec.loader
gguf_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gguf_builder)


REAL_GGUF = Path(
    os.path.expanduser(
        "~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
    )
)


# ---------------------------------------------------------------------------
# The two traps.
# ---------------------------------------------------------------------------


def test_mtp_head_is_found_in_a_file_whose_name_never_says_mtp(tmp_path):
    evidenced, _ = gguf_builder.build_trap_pair(tmp_path)

    assert "mtp" not in evidenced.name.lower(), "fixture must not leak the answer"
    assert read_gguf_mtp_layers(evidenced) == 1, (
        f"'{evidenced.name}' carries nextn metadata and the four nextn tensors, "
        "but the detector missed it. A filename-based check fails exactly here: "
        "this is the shape of the real local Qwen3.8-27B GGUF."
    )


def test_a_file_named_mtp_without_the_head_is_not_reported_as_mtp(tmp_path):
    _, decoy = gguf_builder.build_trap_pair(tmp_path)

    assert "mtp" in decoy.name.lower(), "fixture must actually bait the heuristic"
    assert not read_gguf_mtp_layers(decoy), (
        f"'{decoy.name}' has no nextn metadata and no nextn tensors, yet the "
        "detector called it MTP. It is reading the name, not the file. This is "
        "the shape of the real local Qwen3.6 GGUFs (0 nextn tensors)."
    )


def test_zero_predict_layers_is_not_a_head(tmp_path):
    """`nextn_predict_layers = 0` states the head is absent."""
    path = gguf_builder.build_gguf(
        tmp_path / "explicitly-zero.gguf",
        nextn=True,
        nextn_predict_layers=0,
    )
    assert not read_gguf_mtp_layers(path)


@pytest.mark.parametrize("architecture", ["qwen35", "qwen35moe", "deepseek2", "glm4_moe"])
def test_detection_is_architecture_agnostic(tmp_path, architecture):
    """The metadata key is `<arch>.nextn_predict_layers`, and llama.cpp implements
    the MTP graph for a dozen-plus architectures. Matching must be by suffix, not
    by an enumerated list that silently goes stale.
    """
    path = gguf_builder.build_gguf(
        tmp_path / f"{architecture}-model.gguf",
        architecture=architecture,
        nextn=True,
    )
    assert read_gguf_architecture(path) == architecture
    assert read_gguf_mtp_layers(path) == 1


# ---------------------------------------------------------------------------
# It must never crash the load path.
# ---------------------------------------------------------------------------


def test_non_gguf_and_broken_inputs_return_none_quietly(tmp_path):
    """This runs before a 17 GB load; it must degrade, not raise."""
    not_gguf = tmp_path / "readme.txt"
    not_gguf.write_bytes(b"this is not a GGUF file at all")

    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x01")

    empty = tmp_path / "empty.gguf"
    empty.write_bytes(b"")

    for path in (not_gguf, truncated, empty, tmp_path / "does-not-exist.gguf"):
        assert read_gguf_mtp_layers(path) is None, f"{path.name} should read as None"


def test_detection_reads_only_the_header(tmp_path):
    """A header-only read is what makes this safe to call before loading."""
    evidenced, _ = gguf_builder.build_trap_pair(tmp_path)
    assert evidenced.stat().st_size < 8192, (
        "fixture sanity: these files are tiny, so a detector that needed the "
        "whole file would still pass -- the real guarantee is asserted against "
        "the 16 GB file in test_real_local_gguf_is_detected_without_loading_it"
    )


# ---------------------------------------------------------------------------
# Against the real 16 GB artifact -- header only, never loaded.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_GGUF.exists(), reason="local Qwen3.8-27B GGUF not present")
def test_real_local_gguf_is_detected_without_loading_it():
    """Ground truth: 16.8 GB on disk, 866 tensors, 4 of them nextn.

    Reads the header only -- no llama_cpp, no weights in memory.
    """
    assert "mtp" not in REAL_GGUF.name.lower(), (
        "the whole point: the real MTP-carrying file has no 'mtp' in its name"
    )
    assert read_gguf_architecture(REAL_GGUF) == "qwen35"
    assert read_gguf_mtp_layers(REAL_GGUF) == 1


# ---------------------------------------------------------------------------
# The stale-warning defect: the filename heuristic must stay dead.
# ---------------------------------------------------------------------------


def _gguf_lane_source() -> str:
    import abstractcore.providers.huggingface_provider as hf

    return Path(hf.__file__).read_text(encoding="utf-8")


def test_filename_mtp_heuristic_is_gone_from_the_gguf_lane():
    source = _gguf_lane_source()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if '"mtp" in model_lower' in line or "'mtp' in model_lower" in line
        if not line.strip().startswith("#")
    ]
    assert not offenders, (
        "the filename heuristic is back in huggingface_provider.py:\n  "
        + "\n  ".join(offenders)
    )


def test_the_warning_names_both_concrete_escapes():
    """A warning that says "use a runtime with MTP support" is not actionable.

    Both escapes below were measured on this architecture (1.32x-2.59x), so the
    message must name them precisely enough to copy.
    """
    source = _gguf_lane_source()
    for needle in ("--spec-type", "draft-mtp", "--speculative-draft-mtp"):
        assert needle in source, (
            f"the GGUF MTP warning does not mention {needle!r}; the user is told "
            "they cannot have the speedup without being told how to get it"
        )


def test_the_warning_states_the_real_reason():
    """"Public bindings do not expose it" was vague and wrong-ish.

    The graphs ARE in the shipped libllama; what is missing is the driver, which
    lives in libllama-common and is not in the wheel. Naming the actual blocker
    is what stops someone burning a day looking for a flag.
    """
    source = _gguf_lane_source()
    assert "libllama-common" in source, (
        "the GGUF MTP warning should name libllama-common as the missing piece "
        "rather than vaguely blaming 'public bindings'"
    )


# ---------------------------------------------------------------------------
# Opt-in hardening (not a defect today -- no local artifact triggers it).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("ABSTRACTCORE_MTP_STRICT_EVIDENCE") != "1",
    reason="opt-in hardening: set ABSTRACTCORE_MTP_STRICT_EVIDENCE=1",
)
def test_metadata_key_alone_could_be_corroborated_by_tensors(tmp_path):
    """Detection is metadata-only, so a stripped head would read as present.

    This is the exact shape of the MLX `mtp_num_hidden_layers` false positive,
    one format over: a converter that drops the nextn tensors but keeps the KV
    pair would make AbstractCore promise a speedup llama.cpp cannot deliver,
    since llama.cpp gates on the TENSORS being there.

    No GGUF on this machine exhibits the mismatch, so this is hardening rather
    than a live bug -- hence opt-in.
    """
    half_evidence = gguf_builder.build_gguf(
        tmp_path / "stripped-head.gguf",
        nextn=True,
        include_nextn_metadata=True,
        include_nextn_tensors=False,
    )
    assert not read_gguf_mtp_layers(half_evidence), (
        "a GGUF declaring nextn_predict_layers but carrying no nextn tensors "
        "was reported as MTP-capable on metadata alone"
    )
