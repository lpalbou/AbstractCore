"""`speculation` is a wire-contract field: it must not ride a fuzzy name match.

Capability lookup falls back to substring matching, and a MIDFIX hit is family
STYLE inference, not identity -- "Skywork-o1-Open-Llama-3.1-8B" catches registry
key "o1" without being an o1. Soft fields (vision, context length) survive that
by design; fields that change what gets CONSTRUCTED must not.

`speculation` changes engine construction: it names a drafter repo to download
and a second model to hold in memory. Inheriting it through a midfix match means
a random community fine-tune whose name happens to contain "qwen3.8-27b" tries to
pair itself with a drafter trained for different weights. So it belongs in
`_WIRE_CONTRACT_CAPABILITY_FIELDS`, and this file fails if that is ever forgotten
-- structurally AND behaviourally, because the constant could be right while the
guard that consumes it regresses.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

import abstractcore.architectures.detection as detection
from abstractcore.architectures.detection import (
    _WIRE_CONTRACT_CAPABILITY_FIELDS,
    get_model_capabilities,
)
from abstractcore.providers.speculation import capability_speculation

CAPABILITY_KEY = "speculation"


# ---------------------------------------------------------------------------
# Structural: the field is registered.
# ---------------------------------------------------------------------------


def test_speculation_is_registered_as_a_wire_contract_field():
    assert CAPABILITY_KEY in _WIRE_CONTRACT_CAPABILITY_FIELDS, (
        "`speculation` must be listed in _WIRE_CONTRACT_CAPABILITY_FIELDS "
        "(abstractcore/architectures/detection.py). It selects a drafter "
        "checkpoint, so a midfix fuzzy match would pair unrelated weights with "
        "a drafter and either crash the load or silently draft garbage."
    )


# ---------------------------------------------------------------------------
# Behavioural: the guard actually drops it on a midfix match.
# ---------------------------------------------------------------------------


def _registry_keys_with_speculation() -> Dict[str, Any]:
    detection._load_json_assets()
    models = (detection._model_capabilities or {}).get("models", {}) or {}
    return {
        key: value
        for key, value in models.items()
        if isinstance(value, dict) and CAPABILITY_KEY in value
    }


def test_registry_has_at_least_one_speculation_entry():
    """Otherwise the behavioural tests below would vacuously pass."""
    assert _registry_keys_with_speculation(), (
        "no model_capabilities.json entry carries a `speculation` block, so the "
        "inheritance tests cannot prove anything"
    )


@pytest.mark.parametrize("wrapper", ["frankenmerge-of-{}-v2", "TeamX-{}-distill"])
def test_midfix_match_does_not_inherit_speculation(wrapper):
    """A name that merely CONTAINS a speculating key must not inherit the block."""
    for key in _registry_keys_with_speculation():
        impostor = wrapper.format(key)
        caps = get_model_capabilities(impostor) or {}
        assert CAPABILITY_KEY not in caps, (
            f"'{impostor}' inherited `speculation` from registry key '{key}' "
            "through a midfix substring match. That is family inference, not "
            "identity: this model's weights were never checked for an MTP head."
        )
        # The match itself is meant to still happen (soft fields are fine) --
        # we are asserting the wire-contract stripping, not a lookup miss.
        assert caps, f"expected '{impostor}' to still resolve soft capabilities"


def test_exact_and_prefix_aligned_names_keep_speculation():
    """The guard must not be so blunt that the real model loses its block."""
    for key in _registry_keys_with_speculation():
        caps = get_model_capabilities(key) or {}
        assert CAPABILITY_KEY in caps, (
            f"exact registry key '{key}' lost its own `speculation` block"
        )


def test_the_guard_is_generic_not_hardcoded_to_this_key():
    """Inject a synthetic entry: the mechanism, not the JSON, must do the work.

    If someone re-implements the guard with an explicit allow-list of field
    names this still passes, but if the guard is narrowed to only the two
    original fields it fails here even when the JSON happens to look right.
    """
    detection._load_json_assets()
    models = (detection._model_capabilities or {}).get("models", {}) or {}
    synthetic_key = "advsynth-mtp-model-9000"
    assert synthetic_key not in models

    models[synthetic_key] = {
        "context_length": 4096,
        "max_output_tokens": 1024,
        CAPABILITY_KEY: {
            "native_mtp": True,
            "runtimes": {"mlx": {"mode": "drafter_repo", "drafter": "fake/drafter"}},
        },
    }
    try:
        detection._resolved_aliases_cache.clear()
        exact = get_model_capabilities(synthetic_key) or {}
        assert CAPABILITY_KEY in exact, "exact match must keep the field"

        midfix = get_model_capabilities(f"prefix-{synthetic_key}-suffix") or {}
        assert CAPABILITY_KEY not in midfix, (
            "the wire-contract guard did not strip `speculation` on a midfix "
            "match of a synthetic registry entry"
        )
    finally:
        models.pop(synthetic_key, None)
        detection._resolved_aliases_cache.clear()


# ---------------------------------------------------------------------------
# Evidence discipline: no guessed drafters, no config-key false positives.
# ---------------------------------------------------------------------------


def test_no_speculation_entry_points_a_drafter_at_the_target_itself():
    """`mlx-community/Qwen3.8-27B-4bit` carries NO MTP tensors of its own.

    Its 2180 safetensors keys contain zero mtp/nextn weights; the head ships
    separately as the 31-tensor `...-MTP-4bit` repo (`fc`, `pre_fc_norm_*`,
    one transformer block). A `drafter` field naming the target would therefore
    be a self-reference that loads 15 GB twice and drafts nothing.
    """
    for key, entry in _registry_keys_with_speculation().items():
        runtimes = entry[CAPABILITY_KEY].get("runtimes", {}) or {}
        mlx_block = runtimes.get("mlx")
        if not isinstance(mlx_block, dict):
            continue
        if mlx_block.get("mode") == "embedded":
            assert not mlx_block.get("drafter"), (
                f"'{key}' embeds its head; a separate drafter would duplicate "
                "the target or substitute a different checkpoint"
            )
            continue
        assert mlx_block.get("mode") == "drafter_repo", (
            f"'{key}' declares an unrecognized MLX MTP mode"
        )
        drafter = str(mlx_block.get("drafter", ""))
        assert drafter, f"'{key}' declares an mlx runtime with no drafter repo"
        assert drafter.lower() != key.lower()

        # Test the PROPERTY, not the spelling. This assertion used to require
        # "mtp" in the repo name, which is the same filename-heuristic mistake
        # we deleted from the GGUF lane: Gemma 4's drafter is published as
        # `...-it-qat-assistant-4bit` and is a perfectly real separate
        # checkpoint. What actually matters is that the drafter is not the
        # target itself -- a self-reference loads the weights twice and drafts
        # nothing.
        aliases = {str(a).lower() for a in (entry.get("aliases") or [])}
        aliases.add(str(entry.get("canonical_name", "")).lower())
        assert drafter.lower() not in aliases, (
            f"'{key}' names drafter '{drafter}', which is one of the entry's own "
            "aliases -- that is the target, not a distinct drafter checkpoint."
        )


def test_mlx_drafters_are_only_claimed_where_one_is_published():
    """Require deliberately verified sidecar or embedded MLX head evidence."""
    # Expanded 2026-08-29 as this test instructs -- deliberately, after fetching
    # each repo's config.json and checking it reports `model_type:
    # "qwen3_5_mtp"` (the discriminator a plain target never carries) AND a
    # `text_config.hidden_size` equal to its target's. Repo existence alone was
    # not accepted as evidence.
    #   Qwen3.5-4B-MTP-4bit       block_size 4, hidden 2560   (end-to-end tested)
    #   Qwen3.5-9B-MTP-4bit       block_size 3, hidden 4096
    #   Qwen3.6-27B-MTP-4bit      block_size 3, hidden 5120
    #   Qwen3.6-35B-A3B-MTP-4bit  block_size 3, hidden 2048
    #   Qwen3.8-27B-MTP-4bit      block_size 3, hidden 5120   (end-to-end tested)
    known_published_mlx_drafters = {
        # Gemma 4's is an ASSISTANT drafter, not an in-weights MTP head; it is
        # routed through the same mlx-vlm "mtp" loop and is registered with
        # output_preserving:false because it measurably is not.
        "mlx-community/gemma-4-26b-a4b-it-qat-assistant-4bit",
        "mlx-community/qwen3.5-4b-mtp-4bit",
        "mlx-community/qwen3.5-9b-mtp-4bit",
        "mlx-community/qwen3.6-27b-mtp-4bit",
        "mlx-community/qwen3.6-35b-a3b-mtp-4bit",
        "mlx-community/qwen3.8-27b-mtp-4bit",
    }
    # This checkpoint carries 76 actual indexed mtp.* tensors. Its native
    # loader validates their presence and strictly loads the head; a config
    # flag or a generic Qwen4 family name is not sufficient evidence.
    known_embedded_mlx_artifacts = {
        "qwen3.8-flash-next-oq4e-mtp": (
            "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp",
            "2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8",
        ),
    }
    for key, entry in _registry_keys_with_speculation().items():
        runtimes = entry[CAPABILITY_KEY].get("runtimes", {}) or {}
        mlx_block = runtimes.get("mlx")
        if not isinstance(mlx_block, dict):
            continue
        if mlx_block.get("mode") == "embedded":
            assert key in known_embedded_mlx_artifacts, (
                f"'{key}' claims an embedded MLX head without verified tensor "
                "evidence; verify the artifact before extending this allow-list"
            )
            repo, revision = known_embedded_mlx_artifacts[key]
            assert repo in (entry.get("aliases") or [])
            source = entry[CAPABILITY_KEY].get("source", "")
            assert repo in source and revision in source
            assert "76 embedded mtp tensors" in source
            continue
        drafter = str(mlx_block.get("drafter", "")).lower()
        assert drafter in known_published_mlx_drafters, (
            f"'{key}' claims MLX drafter '{drafter}', which is not a drafter "
            "repo anyone has verified exists. Verify the repo resolves and add "
            "it to this allow-list deliberately, or drop the mlx runtime."
        )


def test_every_speculation_block_cites_its_evidence():
    """The spec forbids guessing, so each block must say how it was verified."""
    for key, entry in _registry_keys_with_speculation().items():
        block = entry[CAPABILITY_KEY]
        source = str(block.get("source", "")).strip()
        assert len(source) > 40, (
            f"'{key}' declares `speculation` with no substantive `source` "
            "field. Every claim must name the artifact evidence behind it."
        )


def test_mtp_config_key_alone_is_never_treated_as_evidence():
    """The `mtp_num_hidden_layers` false positive from the investigation.

    MEASURED, on this machine's HF cache: BOTH `mlx-community/Qwen3.8-27B-4bit`
    (2180 tensors, zero mtp weights) and `mlx-community/Qwen3.8-27B-MTP-4bit`
    (31 tensors, the actual head) report `text_config.mtp_num_hidden_layers = 1`.
    A probe keyed on that config field calls the plain 15 GB target MTP-capable
    on its own weights and then fails at generation time -- or worse, silently
    falls back while the metadata says it accelerated.

    The discriminator is `model_type == "qwen3_5_mtp"` plus `block_size`, which
    only the drafter repo carries.
    """
    target_like_config = {
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1},
    }
    caps_from_config_probe = {
        CAPABILITY_KEY: {
            "native_mtp": bool(
                target_like_config["text_config"].get("mtp_num_hidden_layers")
            ),
            "runtimes": {"mlx": {"mode": "drafter_repo"}},
        }
    }
    block = capability_speculation(caps_from_config_probe, "mlx")
    assert not (block or {}).get("drafter"), (
        "a config-key-derived capability produced an MLX runtime with no "
        "drafter repo. `mtp_num_hidden_layers` is present in checkpoints that "
        "carry no MTP tensors, so it must never be the thing that turns the "
        "lane on."
    )


def test_no_code_path_derives_mtp_capability_from_the_config_key():
    """Source-level guard: nothing may branch on `mtp_num_hidden_layers`.

    Cheap, mechanical, and it catches the false positive being reintroduced in
    a helper nobody thought to test.
    """
    import ast
    import pathlib

    root = pathlib.Path(detection.__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "mtp_num_hidden_layers" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        # Docstrings are where we WANT this trap documented, so exclude exactly
        # those nodes -- but keep every other string constant, because the real
        # attack looks like `config.get("mtp_num_hidden_layers")`.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                body = getattr(node, "body", None)
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))

        for node in ast.walk(tree):
            if id(node) in docstrings:
                continue
            hit = (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "mtp_num_hidden_layers" in node.value
            ) or (
                isinstance(node, ast.Name) and "mtp_num_hidden_layers" in node.id
            ) or (
                isinstance(node, ast.Attribute)
                and "mtp_num_hidden_layers" in node.attr
            )
            if hit:
                lineno = getattr(node, "lineno", 0)
                line = text.splitlines()[lineno - 1].strip() if lineno else ""
                offenders.append(f"{path}:{lineno}: {line}")
    assert not offenders, (
        "executable code reads `mtp_num_hidden_layers`; that key is present in "
        "checkpoints with no MTP weights and is a known false positive:\n  "
        + "\n  ".join(offenders)
    )
