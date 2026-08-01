"""THE ONE TABLE: generation-task vocabulary -> capability default route key.

AbstractCore owns the only store of per-modality provider/model defaults, so it
owns the only table that says which route key holds a given request's default.
`abstractcore.core.generate_contract` and AbstractRuntime's LLM client both
delegate here; each used to carry its own copy and the copies had drifted.
"""

from __future__ import annotations

import pytest

from abstractcore.config.capability_defaults import (
    CAPABILITY_ROUTE_TASKS,
    _OUTPUT_ROUTE_TABLE,
    capability_default_specs_dict,
    capability_route_key_for_output,
    capability_route_keys_for_output,
    normalize_task,
)


@pytest.mark.parametrize(
    ("modality", "task", "expected"),
    [
        # Text generation is the text cell; ANY unlabelled text task lands there.
        ("text", "", "output.text"),
        ("text", "text_generation", "output.text"),
        ("text", "something_else", "output.text"),
        # Image: bare/generation/aliases split by task, all under output.image.
        ("image", "", "output.image.text_to_image"),
        ("image", "image_generation", "output.image.text_to_image"),
        ("image", "text_to_image", "output.image.text_to_image"),
        ("image", "t2i", "output.image.text_to_image"),
        ("image", "image_edit", "output.image.image_to_image"),
        ("image", "image_to_image", "output.image.image_to_image"),
        ("image", "image_upscale", "output.image.image_upscale"),
        ("image", "upscale_image", "output.image.image_upscale"),
        # Video.
        ("video", "", "output.video.text_to_video"),
        ("video", "text_to_video", "output.video.text_to_video"),
        ("video", "image_to_video", "output.video.image_to_video"),
        # Voice/music/sound resolve at the modality cell (no persistable subtask).
        ("voice", "", "output.voice"),
        ("voice", "tts", "output.voice"),
        ("voice", "text_to_speech", "output.voice"),
        # Voice cloning SYNTHESISES voice: the reference audio is conditioning,
        # not a second modality, so it is the same output.voice route as tts.
        ("voice", "voice_clone", "output.voice"),
        ("voice", "clone", "output.voice"),
        ("music", "", "output.music"),
        ("music", "music_generation", "output.music"),
        ("music", "text_to_audio", "output.sound"),
        ("sound", "", "output.sound"),
        # 3D scenes.
        ("scene3d", "", "output.scene3d.text_to_scene3d"),
        ("scene3d", "image_to_scene3d", "output.scene3d.image_to_scene3d"),
        # Case/dash normalization is the table's job, not the caller's.
        ("IMAGE", "Image-Upscale", "output.image.image_upscale"),
    ],
)
def test_output_route_key_for_each_task_in_the_vocabulary(modality, task, expected) -> None:
    assert capability_route_key_for_output(modality, task) == expected


def test_speech_to_text_routes_to_the_input_voice_route_not_a_text_default() -> None:
    """STT provisions the AUDIO side, so its default is an INPUT route.

    Claiming `output.text` for a transcription would hand the STT call the
    operator's chat model. The canonical spec shape emitted by the VisualFlow
    transcribe node is `{modality: text, task: transcription}`; a caller that
    labels it `modality: voice` must land on the same STT route.
    """

    assert capability_route_key_for_output("text", "transcription") is None
    assert capability_route_key_for_output("voice", "stt") == "input.voice"
    assert capability_route_key_for_output("voice", "transcription") == "input.voice"
    # ...and TTS on the same modality stays on the OUTPUT side.
    assert capability_route_key_for_output("voice", "tts") == "output.voice"


@pytest.mark.parametrize(
    ("modality", "task", "expected"),
    [
        ("image", "", "output.image.image_to_image"),
        ("image", "image_generation", "output.image.image_to_image"),
        ("video", "", "output.video.image_to_video"),
        ("scene3d", "", "output.scene3d.image_to_scene3d"),
    ],
)
def test_a_source_image_selects_the_edit_variant_of_a_bare_request(modality, task, expected) -> None:
    """"Generate" with a source image attached means edit/i2v/i23d."""

    assert capability_route_key_for_output(modality, task, has_source_image=True) == expected
    # An explicitly-named task is not re-interpreted by the attachment.
    assert capability_route_key_for_output("image", "image_upscale", has_source_image=True) == (
        "output.image.image_upscale"
    )


def test_unroutable_specs_return_none() -> None:
    assert capability_route_key_for_output("", "") is None
    assert capability_route_key_for_output("bogus", "whatever") is None
    assert capability_route_key_for_output(None, None) is None
    assert capability_route_keys_for_output("bogus", "x") == (None, None)


def test_broad_fallback_is_the_modality_cell_and_never_repeats_the_exact_key() -> None:
    assert capability_route_keys_for_output("image", "image_upscale") == (
        "output.image.image_upscale",
        "output.image",
    )
    # A key that already IS the modality cell has no second lookup to do.
    assert capability_route_keys_for_output("voice", "tts") == ("output.voice", None)
    assert capability_route_keys_for_output("music", "") == ("output.music", None)


def test_every_route_key_the_table_emits_is_one_the_store_can_actually_hold() -> None:
    """The regression that motivated the single table.

    AbstractRuntime's copy minted `output.voice.tts`, `input.voice.stt`,
    `output.music.text_to_music` and `output.sound.text_to_sound`. None of those
    task names are in `CAPABILITY_ROUTE_TASKS`, so `set_capability_default`
    could never persist them -- every one silently fell through to the broad
    modality key. A table that emits keys the store cannot hold is a table that
    lies about where a default lives.
    """

    known_keys = set(capability_default_specs_dict())
    probes = [
        ("text", ""),
        ("text", "text_generation"),
        ("image", ""),
        ("image", "image_generation"),
        ("image", "image_edit"),
        ("image", "image_upscale"),
        ("video", ""),
        ("video", "text_to_video"),
        ("video", "image_to_video"),
        ("voice", "tts"),
        ("voice", "stt"),
        ("music", "music_generation"),
        ("music", "text_to_audio"),
        ("sound", ""),
        ("scene3d", "text_to_scene3d"),
        ("scene3d", "image_to_scene3d"),
    ]
    for modality, task in probes:
        for has_source in (False, True):
            key = capability_route_key_for_output(modality, task, has_source_image=has_source)
            if key is None:
                continue
            assert key in known_keys, f"{modality}/{task} emitted unknown route key {key!r}"
            parts = key.split(".")
            if len(parts) == 3:
                # A task suffix is only legal for the tasks the store persists.
                assert parts[2] in CAPABILITY_ROUTE_TASKS
                assert normalize_task(parts[2]) == parts[2]


def test_every_generation_task_in_the_output_vocabulary_has_a_route() -> None:
    """No generation task may fall through the table.

    `OUTPUT_TASK_MODALITIES` is the complete set of (task -> modality) pairs
    `_infer_output_specs` can produce, so a task missing from the table is a
    request whose default silently does not exist: the call falls back to the
    plugin's own env-or-openai default, which is the dm#28 429 incident.
    `voice_clone` -- the task assigned to `output="voice"` plus reference audio
    -- was missing from the first cut of this table and this is its guard.

    `transcription` is the one deliberate None: it is provisioned by the
    `input.voice` INPUT route, not by an output route.
    """

    from abstractcore.core.output_specs import OUTPUT_TASK_MODALITIES

    missing = [
        f"{modality}/{task}"
        for task, modality in sorted(OUTPUT_TASK_MODALITIES.items())
        if task != "transcription" and not capability_route_key_for_output(modality, task)
    ]
    assert not missing, f"generation tasks with no capability default route: {missing}"


def test_voice_clone_keeps_the_operator_configured_voice_route_end_to_end() -> None:
    """The regression at the layer that actually executes.

    `generate(output="voice", media=[audio])` normalizes to
    `{modality: voice, task: voice_clone}`; it must still receive the
    configured `output.voice` provider/model, exactly as a bare TTS does.
    """

    from abstractcore.core.generate_contract import normalize_generate_request, resolve_generate_route

    routes = {"output.voice": {"provider": "supertonic", "model": "supertonic-3"}}
    request = normalize_generate_request(prompt="say this in my voice", messages=None, media=None)
    resolved = resolve_generate_route(
        request=request, output={"modality": "voice", "task": "voice_clone"}, scoped_routes=routes
    )
    spec = resolved.output_specs[0]
    assert (spec.get("provider"), spec.get("model")) == ("supertonic", "supertonic-3")


def test_generate_contract_delegates_to_the_same_table() -> None:
    """Core's execution path must not re-derive the mapping."""

    from abstractcore.core.generate_contract import _output_route_key, normalize_generate_request

    request = normalize_generate_request(prompt="a cat", messages=None, media=None)
    for modality, task in (("image", "image_generation"), ("video", "text_to_video"), ("voice", "tts")):
        assert _output_route_key({"modality": modality, "task": task}, request) == (
            capability_route_key_for_output(modality, task)
        )


# ---------------------------------------------------------------------------
# THE MODALITY ROW IS A PARENT, NOT A REMNANT
# ---------------------------------------------------------------------------
#
# Operator question, 2026-08-01, looking at the Routes grid: "Why do we have
# output.image and output.video? Are those remnants of old code? We have t2i,
# i2i, t2v and i2v." They are not remnants -- they are the row that answers
# every task of that modality with no row of its own, which is exactly what the
# fresh-install seed writes. These tests pin the semantics end to end so the
# question cannot be answered by deleting a live fallback.


def test_modality_row_answers_when_the_task_row_is_absent() -> None:
    """Task row absent + modality row set -> the modality row answers."""

    import json
    import tempfile
    from pathlib import Path

    from abstractcore.core.generate_contract import resolve_capability_default_route

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "abstractcore.json"
        path.write_text(
            json.dumps(
                {
                    "capability_defaults": {
                        "version": 1,
                        "routes": {"output.image": {"provider": "mlx-gen", "model": "one-model"}},
                    }
                }
            ),
            encoding="utf-8",
        )
        for key in (
            "output.image.text_to_image",
            "output.image.image_to_image",
            "output.image.image_upscale",
        ):
            row = resolve_capability_default_route(key, config_file=str(path))
            assert row is not None and row["model"] == "one-model", key
            assert row["key"] == "output.image", "the answer names the row it came from"


def test_task_row_beats_the_modality_row_wholesale() -> None:
    """Both set -> the task row wins, and it wins as ONE identity."""

    from abstractcore.core.generate_contract import resolve_capability_default_route

    scoped = {
        "output.image": {"provider": "mlx-gen", "model": "broad", "options": {"steps": 4}},
        "output.image.image_upscale": {"provider": "mlx-gen", "model": "upscaler"},
    }
    row = resolve_capability_default_route("output.image.image_upscale", scoped_routes=scoped)
    assert row["model"] == "upscaler"
    assert "options" not in row, "the parent's options never ride along with a task row that won"


def test_pushed_not_configured_sentinel_does_not_veto_the_modality_row() -> None:
    """THE GATEWAY DEPLOYMENT REGRESSION.

    A pushed payload enumerates EVERY route, carrying
    ``source: "not_configured"`` for the unset ones. That sentinel exists to
    short-circuit the CONFIG-FILE read (the live payload already knows the
    answer) -- consuming it at the task key returned immediately, so
    `output.image` was never consulted for an unset `output.image.*` and the
    documented "exact-key then broad fallback" was false in every Gateway
    deployment.
    """

    from abstractcore.core.generate_contract import resolve_capability_default_route

    scoped = {
        "output.image": {"provider": "mlx-gen", "model": "one-model"},
        "output.image.text_to_image": {"key": "output.image.text_to_image", "source": "not_configured"},
    }
    row = resolve_capability_default_route("output.image.text_to_image", scoped_routes=scoped)
    assert row["model"] == "one-model"
    assert row["key"] == "output.image"

    # Both unset in the payload -> the sentinel still short-circuits the disk
    # read, reported against the key that was ASKED for.
    both_unset = {
        "output.image": {"key": "output.image", "source": "not_configured"},
        "output.image.text_to_image": {"key": "output.image.text_to_image", "source": "not_configured"},
    }
    row = resolve_capability_default_route("output.image.text_to_image", scoped_routes=both_unset)
    assert row == {"key": "output.image.text_to_image", "source": "not_configured"}


def test_nothing_reads_the_modality_row_once_every_task_row_is_set() -> None:
    """All tasks set + modality row unset -> the modality row is unreachable.

    This is what makes `not needed` an honest state in the grid rather than a
    cosmetic softening of `not configured`.
    """

    from abstractcore.config.capability_defaults import (
        capability_route_task_keys,
        capability_route_tasks_cover_broad,
    )
    from abstractcore.core.generate_contract import resolve_capability_default_route

    for modality in ("image", "video", "scene3d"):
        parent = f"output.{modality}"
        task_keys = capability_route_task_keys(parent)
        assert task_keys, f"{parent} must have task rows for this test to mean anything"
        scoped = {key: {"provider": "mlx-gen", "model": key} for key in task_keys}
        scoped[parent] = {"key": parent, "source": "not_configured"}
        assert capability_route_tasks_cover_broad(parent, scoped)

        # Every task the output table can route for this modality lands on a
        # task row, never on the parent.
        for _row_modality, aliases, route_key, source_image_key in _OUTPUT_ROUTE_TABLE:
            if _row_modality != modality:
                continue
            for candidate in {route_key, source_image_key} - {None, ""}:
                resolved = resolve_capability_default_route(candidate, scoped_routes=scoped)
                assert resolved["key"] != parent, (
                    f"{candidate} reached the parent row even though every task row is set"
                )


def test_task_rows_are_exactly_the_keys_the_output_table_can_produce() -> None:
    """The proof behind `covered_by_tasks`.

    If the table could route a modality to a 3-part key with no row in the
    grid, "every task row is set" would not imply "nothing reads the parent"
    and the benign state would be a lie.
    """

    from abstractcore.config.capability_defaults import capability_route_task_keys

    emitted: dict = {}
    for modality, _aliases, route_key, source_image_key in _OUTPUT_ROUTE_TABLE:
        for candidate in (route_key, source_image_key):
            if candidate and candidate.count(".") == 2:
                emitted.setdefault(modality, set()).add(candidate)
    for modality, keys in emitted.items():
        assert keys == set(capability_route_task_keys(f"output.{modality}")), modality


def test_modality_row_is_the_primary_key_where_no_task_rows_exist() -> None:
    """Why the row SHAPE can never be deleted: it is voice/sound/music's only key."""

    from abstractcore.config.capability_defaults import (
        capability_route_broad_key,
        capability_route_task_keys,
    )

    for modality in ("voice", "sound", "music"):
        key = f"output.{modality}"
        assert capability_route_task_keys(key) == ()
        assert capability_route_keys_for_output(modality, None) == (key, None)
        assert capability_route_broad_key(key) is None


def test_grid_payload_carries_the_hierarchy_for_every_surface() -> None:
    """ONE DERIVATION, FOUR GRIDS (web console, both TUIs, the CLI)."""

    import json
    import tempfile
    from pathlib import Path

    from abstractcore.config.manager import ConfigurationManager

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "abstractcore.json"
        path.write_text(
            json.dumps(
                {
                    "capability_defaults": {
                        "version": 1,
                        "routes": {
                            "output.image.text_to_image": {"provider": "mlx-gen", "model": "a"},
                            "output.image.image_to_image": {"provider": "mlx-gen", "model": "b"},
                            "output.image.image_upscale": {"provider": "mlx-gen", "model": "c"},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        rows = {
            row["key"]: row
            for row in ConfigurationManager(config_file=str(path), apply_env=False).list_capability_defaults()
        }

    parent = rows["output.image"]
    assert parent["task_keys"] == [
        "output.image.text_to_image",
        "output.image.image_to_image",
        "output.image.image_upscale",
    ]
    assert parent["covered_by_tasks"] is True, "all three task rows are set -- nothing reads the parent"
    assert parent["configured"] is False
    # `covered_by` drives READ-ONLY; this row must stay settable, because
    # setting it is the simple one-model-for-every-image-task path.
    assert "covered_by" not in parent and not parent.get("read_only")

    assert rows["output.image.text_to_image"]["broad_key"] == "output.image"
    # An unset parent whose tasks are NOT all set is still the missing setting.
    assert rows["output.video"]["task_keys"]
    assert not rows["output.video"].get("covered_by_tasks")
    # A modality with no task rows carries no hierarchy decoration at all.
    assert "task_keys" not in rows["output.voice"]
    assert "broad_key" not in rows["output.voice"]
    # Nothing inherits here -- no parent is configured.
    assert not any(row.get("inherits_broad") for row in rows.values())


def test_grid_payload_marks_task_rows_answered_by_a_configured_parent() -> None:
    """THE MIRROR, and the shape a FRESH INSTALL has.

    `RECOMMENDED_CAPABILITY_DEFAULT_ROUTES` seeds `output.image` alone, so a
    brand-new machine shows a working parent above three task rows with no
    value of their own. Painting those three `not configured` says "image
    editing is not set up" about a machine where it demonstrably is -- the same
    confusion the operator hit, in the opposite direction.
    """

    import json
    import tempfile
    from pathlib import Path

    from abstractcore.config.capability_defaults import RECOMMENDED_CAPABILITY_DEFAULT_ROUTES
    from abstractcore.config.manager import ConfigurationManager

    seeded = RECOMMENDED_CAPABILITY_DEFAULT_ROUTES["output.image"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "abstractcore.json"
        path.write_text(
            json.dumps(
                {
                    "capability_defaults": {
                        "version": 1,
                        "routes": {"output.image": {"provider": seeded.provider, "model": seeded.model}},
                    }
                }
            ),
            encoding="utf-8",
        )
        rows = {
            row["key"]: row
            for row in ConfigurationManager(config_file=str(path), apply_env=False).list_capability_defaults()
        }

    assert rows["output.image"]["configured"] is True
    assert not rows["output.image"].get("covered_by_tasks")
    for key in ("text_to_image", "image_to_image", "image_upscale"):
        row = rows[f"output.image.{key}"]
        assert row["configured"] is False
        assert row["inherits_broad"] is True, f"{key} is answered by the seeded parent"
    # The video parent is NOT configured, so its task rows inherit nothing and
    # stay honestly unconfigured.
    assert not rows["output.video.text_to_video"].get("inherits_broad")
