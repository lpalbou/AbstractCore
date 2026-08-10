"""No shipped tool may carry a description that would get it dropped or ignored.

Item 4 of the 2026-08-07 tool-description work. `ToolDefinition` caps
`description` at 200 chars and the `@tool` decorator raises past it — but that
raise only fires when the module is IMPORTED. A tool added to a module the
inventory does not scan, or a description edited past the cap in a module that
nothing imports during a test run, would sail through CI and only surface as a
missing capability at runtime on a user's machine.

This test walks the authoritative inventory (`list_builtin_tool_inventory`,
which is derived from the live `@tool` definitions, never a copied list) and
asserts every shipped tool is actually shippable. It is deliberately about the
MECHANICAL contract — length, presence, uniqueness. Description QUALITY is a
review matter, not something a test can assert.
"""

from __future__ import annotations

import warnings

import pytest

from abstractcore.tools.core import (
    _MAX_TOOL_DESCRIPTION_CHARS,
    _MAX_TOOL_EXAMPLES,
    _MAX_TOOL_WHEN_TO_USE_CHARS,
    ToolDefinition,
)
from abstractcore.tools.inventory import list_builtin_tool_inventory
from abstractcore.tools.handler import UniversalToolHandler

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def _inventory():
    return list_builtin_tool_inventory()


def test_the_inventory_is_not_empty():
    """A silent import failure would make every other assertion here vacuous."""
    assert len(_inventory()) >= 25


@pytest.mark.parametrize("descriptor", _inventory(), ids=lambda d: d.name)
def test_every_shipped_description_is_present_and_within_the_cap(descriptor):
    description = descriptor.description
    assert description, f"{descriptor.name}: description is missing or empty"
    assert description.strip() == description, f"{descriptor.name}: description has stray whitespace"
    assert "\n" not in description, f"{descriptor.name}: description must be a single line"
    assert len(description) <= _MAX_TOOL_DESCRIPTION_CHARS, (
        f"{descriptor.name}: description is {len(description)} chars "
        f"(max {_MAX_TOOL_DESCRIPTION_CHARS}). Shorten it — the cap is not negotiable and an "
        f"over-long description on OUR OWN tool is a build error, not something to adapt around."
    )


@pytest.mark.parametrize("descriptor", _inventory(), ids=lambda d: d.name)
def test_every_shipped_tool_survives_the_conversion_path(descriptor):
    """The property that actually matters: it reaches the model.

    The cap is only one of the ways a tool can vanish. This drives the real
    converter and asserts the tool is present in BOTH lanes — the prompted lane
    (local models) and the native lane (API providers) — and that no warning was
    raised on the way, since every drop and every adaptation now warns.
    """
    from abstractcore.tools.inventory import _INVENTORY_MODULES, _scan_module_tool_definitions

    tool_def = None
    for module_path in _INVENTORY_MODULES:
        for candidate in _scan_module_tool_definitions(module_path):
            if candidate.name == descriptor.name:
                tool_def = candidate
                break
    assert tool_def is not None, f"{descriptor.name} not found in any scanned module"

    handler = UniversalToolHandler(MODEL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prompted = handler.format_tools_prompt([tool_def])
        native = handler.prepare_tools_for_native([tool_def])

    complaints = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not complaints, f"{descriptor.name} provoked a tool warning: {complaints}"
    assert descriptor.name in prompted, f"{descriptor.name} is missing from the prompted tool block"
    assert [t["function"]["name"] for t in native] == [descriptor.name]


@pytest.mark.parametrize("descriptor", _inventory(), ids=lambda d: d.name)
def test_every_shipped_when_to_use_is_within_its_cap(descriptor):
    from abstractcore.tools.inventory import _INVENTORY_MODULES, _scan_module_tool_definitions

    for module_path in _INVENTORY_MODULES:
        for candidate in _scan_module_tool_definitions(module_path):
            if candidate.name != descriptor.name:
                continue
            if candidate.when_to_use:
                assert len(candidate.when_to_use) <= _MAX_TOOL_WHEN_TO_USE_CHARS, (
                    f"{descriptor.name}: when_to_use is {len(candidate.when_to_use)} chars "
                    f"(max {_MAX_TOOL_WHEN_TO_USE_CHARS})"
                )
            assert len(candidate.examples) <= _MAX_TOOL_EXAMPLES


def test_shipped_descriptions_are_distinct():
    """Two tools with the same description cannot be told apart by a model."""
    seen: dict[str, str] = {}
    for descriptor in _inventory():
        clash = seen.get(descriptor.description)
        assert clash is None, (
            f"'{descriptor.name}' and '{clash}' ship the SAME description — a model cannot "
            f"choose between them: {descriptor.description!r}"
        )
        seen[descriptor.description] = descriptor.name


def test_the_whole_shipped_catalog_renders_together():
    """The realistic case: every builtin offered at once, nothing lost."""
    from abstractcore.tools.inventory import _INVENTORY_MODULES, _scan_module_tool_definitions

    tool_defs = []
    for module_path in _INVENTORY_MODULES:
        tool_defs.extend(_scan_module_tool_definitions(module_path))

    handler = UniversalToolHandler(MODEL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = handler.prepare_tools_for_native(tool_defs)

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(native) == len(tool_defs)


def test_the_guard_would_catch_a_regression():
    """Proof the assertions above are load-bearing, not vacuously true."""
    with pytest.raises(ValueError, match="description is too long"):
        ToolDefinition(
            name="regression_probe",
            description="y" * (_MAX_TOOL_DESCRIPTION_CHARS + 1),
            parameters={},
        )
