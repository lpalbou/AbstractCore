"""Tests for centralized schema-aware tool-argument coercion (backlog 039).

Covers the coercion primitive, the security-flag cases that motivated the item, and the two
dispatch paths (AbstractCore registry + runtime mapping executor share one behavior).
"""
from __future__ import annotations

import pytest

from abstractcore.tools.arg_coercion import (
    ArgumentCoercionError,
    coerce_arguments,
    coerce_arguments_for_callable,
)
from abstractcore.tools.common_tools import edit_file, execute_command
from abstractcore.tools.core import ToolDefinition, tool
from abstractcore.tools.registry import ToolRegistry
from abstractcore.tools.core import ToolCall


# ---------------------------------------------------------------------------
# 1. Primitive coercion rules
# ---------------------------------------------------------------------------

BOOL_SCHEMA = {"flag": {"type": "boolean"}}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        (True, True), (False, False),
    ],
)
def test_bool_tokens(raw, expected) -> None:
    out, warnings = coerce_arguments(BOOL_SCHEMA, {"flag": raw})
    assert out["flag"] is expected
    # A real coercion (string->bool) emits a #FALLBACK note; an already-bool value does not.
    if isinstance(raw, str):
        assert any("#FALLBACK" in w and "flag" in w for w in warnings)
    else:
        assert warnings == []


def test_bool_ambiguous_raises() -> None:
    with pytest.raises(ArgumentCoercionError):
        coerce_arguments(BOOL_SCHEMA, {"flag": "maybe"})


def test_int_and_number_and_ambiguity() -> None:
    out, _ = coerce_arguments({"n": {"type": "integer"}}, {"n": "42"})
    assert out["n"] == 42 and isinstance(out["n"], int)
    out, _ = coerce_arguments({"x": {"type": "number"}}, {"x": "1.5"})
    assert out["x"] == 1.5
    with pytest.raises(ArgumentCoercionError):
        coerce_arguments({"n": {"type": "integer"}}, {"n": "3.7"})
    with pytest.raises(ArgumentCoercionError):
        coerce_arguments({"n": {"type": "integer"}}, {"n": "not-a-number"})


def test_container_types_are_conservative() -> None:
    # JSON-encoded string upgrades to the structured value.
    out, _ = coerce_arguments({"xs": {"type": "array"}}, {"xs": "[1, 2, 3]"})
    assert out["xs"] == [1, 2, 3]
    out, _ = coerce_arguments({"o": {"type": "object"}}, {"o": '{"a": 1}'})
    assert out["o"] == {"a": 1}
    # A non-JSON string for an array is left untouched (string-or-list tools keep working).
    out, _ = coerce_arguments({"xs": {"type": "array"}}, {"xs": "a.py"})
    assert out["xs"] == "a.py"


def test_no_mutation_and_unknown_keys_passthrough() -> None:
    src = {"flag": "true", "unknown": "keep"}
    out, _ = coerce_arguments(BOOL_SCHEMA, src)
    assert src == {"flag": "true", "unknown": "keep"}  # input untouched
    assert out["flag"] is True and out["unknown"] == "keep"


# ---------------------------------------------------------------------------
# 2. The security-flag cases that motivated the item (real tool schemas)
# ---------------------------------------------------------------------------

def test_edit_file_flags_coerce_via_callable_schema() -> None:
    out, warnings = coerce_arguments_for_callable(
        edit_file, {"use_regex": "false", "preview_only": "false", "max_replacements": "1"}
    )
    assert out["use_regex"] is False
    assert out["preview_only"] is False
    assert out["max_replacements"] == 1 and isinstance(out["max_replacements"], int)
    assert warnings  # coercions were applied and reported


def test_execute_command_allow_dangerous_false_is_false() -> None:
    out, _ = coerce_arguments_for_callable(execute_command, {"allow_dangerous": "false"})
    assert out["allow_dangerous"] is False


# ---------------------------------------------------------------------------
# 3. Registry dispatch path applies coercion and fails loudly on bad types
# ---------------------------------------------------------------------------

def test_registry_path_coerces_and_errors() -> None:
    seen: dict = {}

    @tool
    def _danger(allow_dangerous: bool = False) -> str:
        """test tool"""
        seen["allow_dangerous"] = allow_dangerous
        return "ran"

    reg = ToolRegistry()
    reg.register(_danger._tool_definition)

    # "false" must NOT enable the dangerous path.
    res = reg.execute_tool(ToolCall(name="_danger", arguments={"allow_dangerous": "false"}, call_id="1"))
    assert res.success is True
    assert seen["allow_dangerous"] is False

    # An un-coercible boolean value fails loudly (no silent default, no execution).
    seen.clear()
    res = reg.execute_tool(ToolCall(name="_danger", arguments={"allow_dangerous": "perhaps"}, call_id="2"))
    assert res.success is False
    assert "Invalid argument type" in (res.error or "")
    assert "allow_dangerous" not in seen  # never executed


def test_registry_typeerror_retry_path_still_coerces() -> None:
    """The registry's `except TypeError` retry branch must ALSO coerce (backlog 039).

    A stray kwarg trips the first invocation's TypeError; the retry strips it and re-invokes.
    That retry path previously skipped coercion, resurrecting the string-truthiness bug
    (preview_only="false" delivered as a truthy string). Regression guard for that branch.
    """
    seen: dict = {}

    @tool
    def _edit(preview_only: bool = False, use_regex: bool = False) -> str:
        """test tool with boolean flags (no **kwargs -> stray key raises TypeError)"""
        seen["preview_only"] = preview_only
        seen["use_regex"] = use_regex
        return "ran"

    reg = ToolRegistry()
    reg.register(_edit._tool_definition)

    # `extra_meta` is not a parameter -> first func(**args) raises TypeError -> retry branch.
    res = reg.execute_tool(
        ToolCall(
            name="_edit",
            arguments={"preview_only": "false", "use_regex": "false", "extra_meta": "x"},
            call_id="1",
        )
    )
    assert res.success is True
    # Coercion applied on the retry path: real bools, not truthy strings.
    assert seen["preview_only"] is False
    assert seen["use_regex"] is False
