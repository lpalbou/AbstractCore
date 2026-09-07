"""The recommended starter kit is ADVICE FOR AN EMPTY ROUTE, not a standing debt.

`recommended_plan()` answers exactly one question: is the fresh-install model on
this disk? Every surface that rendered that answer raw -- `abstractcore models
status`, the AbstractCore console-TUI, both Gateway consoles -- told an operator
who had deliberately routed `input.text` at their own model that a model was
MISSING, with a download command to run. The only way to clear it was to install
the model they had chosen against, so it never cleared, and a status line that
cries wolf on a healthy machine teaches an operator to stop reading it.

`mark_recommended_route_gaps` answers the question a surface actually needs --
"is any recommended model missing for a route that has NOTHING else serving
it?" -- once, next to the plan it interprets. The counts and `would_download`
are deliberately untouched: `--dry-run` and `models download --recommended` ask
what the recommendation WOULD fetch, which is a different question.
"""

from __future__ import annotations

import pytest

from abstractcore.config import model_materializer as mm


@pytest.mark.parametrize(
    "row, answered, why",
    [
        ({"key": "input.text", "provider": "airelay", "model": "gpt-5.6-terra"}, True, "its own value"),
        ({"key": "input.image", "covered_by": "input.text"}, True, "the text model handles it"),
        ({"key": "output.image", "covered_by_tasks": True}, True, "every task row under it is set"),
        ({"key": "output.image.text_to_image", "inherits_broad": True}, True, "its parent is set"),
        ({"key": "output.voice", "provider": "", "model": ""}, False, "nothing serves it"),
        ({"key": "output.voice", "provider": "supertonic", "model": ""}, False, "half a route serves nothing"),
        ("not a row", False, "junk is not an answer"),
    ],
)
def test_a_route_is_answered_by_any_lane_that_actually_serves_it(row, answered, why):
    assert mm.route_is_answered(row) is answered, why


def _plan():
    return {
        "total": 3,
        "installed": 2,
        "absent": 1,
        "unknown": 0,
        "recommended": [
            {"route": "input.text", "provider": "lmstudio", "artifact": "qwen/qwen3.5-9b@4bit"},
            {"route": "output.voice", "provider": "supertonic", "artifact": "supertonic-3"},
            {"route": "output.image", "provider": "mlx-gen", "artifact": "flux.2-klein-4b-8bit"},
        ],
        "would_download": [
            {"route": "input.text", "provider": "lmstudio", "artifact": "qwen/qwen3.5-9b@4bit"},
        ],
    }


def test_a_configured_route_has_no_gap_however_absent_the_recommendation_is():
    """THE REPORTED DEFECT, pinned.

    `input.text` is served by the operator's own remote model. The recommended
    LM Studio build is nowhere on the machine and never will be. That is not a
    gap, and no surface may present it as one.
    """

    routes = [
        {"key": "input.text", "provider": "airelay", "model": "gpt-5.6-terra"},
        {"key": "output.voice", "provider": "supertonic", "model": "supertonic-3"},
        {"key": "output.image", "provider": "mlx-gen", "model": "flux"},
    ]
    plan = mm.mark_recommended_route_gaps(_plan(), routes)

    assert plan["gaps"] == []
    assert plan["routes_unanswered"] == 0
    # The raw catalog answer survives untouched -- only the INTERPRETATION is
    # added, because `--dry-run` still has to say what it would fetch.
    assert (plan["total"], plan["installed"], plan["absent"]) == (3, 2, 1)
    assert len(plan["would_download"]) == 1
    assert all(item["route_answered"] for item in plan["recommended"])


def test_an_empty_route_whose_model_is_absent_is_a_gap():
    """The fresh install the starter kit exists for still gets its offer."""

    routes = [
        {"key": "input.text", "provider": "", "model": "", "source": "not_configured"},
        {"key": "output.voice", "provider": "supertonic", "model": "supertonic-3"},
    ]
    plan = mm.mark_recommended_route_gaps(_plan(), routes)

    assert plan["routes_unanswered"] == 1
    assert plan["gaps"] == [
        {
            "route": "input.text",
            "provider": "lmstudio",
            "artifact": "qwen/qwen3.5-9b@4bit",
            "route_answered": False,
        }
    ]


def test_a_parent_covered_by_its_task_rows_is_not_a_gap():
    """`output.image` unset with every image task set is a WORKING machine."""

    plan = _plan()
    plan["would_download"].append(
        {"route": "output.image", "provider": "mlx-gen", "artifact": "flux.2-klein-4b-8bit"}
    )
    routes = [
        {"key": "input.text", "provider": "airelay", "model": "gpt-5.6-terra"},
        {"key": "output.image", "provider": "", "model": "", "covered_by_tasks": True},
        {"key": "output.image.text_to_image", "provider": "mlx-gen", "model": "flux"},
    ]

    assert mm.mark_recommended_route_gaps(plan, routes)["gaps"] == []


# ---------------------------------------------------------------------------
# `abstractcore models status`
# ---------------------------------------------------------------------------


def _status_text(capsys, plan) -> str:
    from abstractcore.config import main

    main._print_models_status({"routes": [], "recommended": plan})
    return capsys.readouterr().out


def test_the_cli_never_asks_for_a_model_no_route_needs(capsys):
    plan = mm.mark_recommended_route_gaps(
        _plan(), [{"key": "input.text", "provider": "airelay", "model": "gpt-5.6-terra"}]
    )
    out = _status_text(capsys, plan)

    # The count is a REPORT the operator asked for by typing the command, and
    # it stays. What goes is the instruction to act on it.
    assert "Recommended defaults: 2 of 3 present" in out
    assert "missing:" not in out
    assert "models download --recommended" not in out
    assert "nothing to download" in out


def test_the_cli_still_names_a_gap_and_the_command_that_fills_it(capsys):
    plan = mm.mark_recommended_route_gaps(
        _plan(), [{"key": "output.voice", "provider": "supertonic", "model": "supertonic-3"}]
    )
    out = _status_text(capsys, plan)

    assert "missing: lmstudio qwen/qwen3.5-9b@4bit  (input.text)" in out
    assert "models download --recommended" in out


def test_a_plan_from_an_older_caller_still_prints_its_missing_list(capsys):
    """No `gaps` key at all: fall back to the raw list rather than go silent."""

    out = _status_text(capsys, _plan())

    assert "missing: lmstudio qwen/qwen3.5-9b@4bit" in out


def test_the_status_payload_marks_gaps_against_the_whole_grid(monkeypatch):
    """`models status input.text` must not report every other route as empty.

    The `target` filter narrows what is PRINTED. Marking against the narrowed
    list would make the two routes it removed look unanswered, so a filtered
    status would invent gaps a full status does not have.
    """

    from abstractcore.config import main

    routes = [
        {"key": "input.text", "provider": "airelay", "model": "gpt-5.6-terra"},
        {"key": "output.voice", "provider": "supertonic", "model": "supertonic-3"},
        {"key": "output.image", "provider": "mlx-gen", "model": "flux"},
    ]

    class _Manager:
        config_file = "/tmp/abstractcore.json"

        class config:
            class capability_defaults:
                seeded = ""

        def list_capability_defaults(self):
            return routes

    monkeypatch.setattr(main, "_config_manager_for_cli", lambda args: _Manager())
    monkeypatch.setattr(mm, "annotate_route_availability", lambda rows: list(rows))
    monkeypatch.setattr(mm, "recommended_plan", _plan)
    monkeypatch.setattr(mm, "supported_providers", lambda: [], raising=False)

    class _Args:
        target = "input.text"
        all = False

    payload = main._models_status_payload(_Args())

    assert [row["key"] for row in payload["routes"]] == ["input.text"]
    assert payload["recommended"]["gaps"] == []
