"""Tool parameters reach the model with their meaning, not as bare names.

2026-11-08: the schema built from a decorated function carried {type, default} only; the
docstring's `Args:` text never left the source file. search_files showed the model twelve
undocumented parameters, and the model ran a recursive shell grep over the whole workspace
instead. Now the `Args:` section fills each parameter's `description`; native payloads carry it
as JSON Schema, the prompted tool list shows its first sentence, an explicit description set on
the schema afterwards wins, and every built-in tool parameter is described.
"""

from __future__ import annotations

from abstractcore.tools.core import ToolDefinition, docstring_parameter_descriptions, tool
from abstractcore.tools.handler import UniversalToolHandler
from abstractcore.tools.inventory import list_builtin_tool_inventory
from abstractcore.tools.parser import _format_parameters_compact, format_tool_prompt


def test_google_args_section_fills_parameter_descriptions():
    @tool
    def demo(path: str, depth: int = 2, flag: bool = False, untold: str = "") -> str:
        """Do a demo thing.

        A second paragraph that is not an argument.

        Args:
            path: Where to look. Relative paths are resolved from the
                workspace root, absolute ones used as given.
            depth (int): How deep to go (default: 2).
            flag: Turn the thing on.

        Returns:
            Text.

        Raises:
            ValueError: on nonsense.
        """
        return path

    params = demo._tool_definition.parameters
    assert params["path"]["description"] == (
        "Where to look. Relative paths are resolved from the workspace root, absolute ones used as given."
    )
    assert params["depth"]["description"] == "How deep to go (default: 2)."
    assert params["flag"]["description"] == "Turn the thing on."
    assert "description" not in params["untold"]  # nothing invented for an undocumented parameter
    assert "ValueError" not in str(params)  # the Raises section is not an argument


def test_sphinx_param_lines_are_read_too():
    def demo(a: str, b: int = 1) -> str:
        """Summary.

        :param a: The first thing,
            continued on the next line.
        :param int b: The second thing.
        :returns: nothing
        """
        return a

    assert docstring_parameter_descriptions(demo.__doc__) == {
        "a": "The first thing, continued on the next line.",
        "b": "The second thing.",
    }
    assert docstring_parameter_descriptions(None) == {}
    assert docstring_parameter_descriptions("Just a sentence.") == {}


def test_explicit_schema_description_wins_over_the_docstring():
    from abstractcore.tools.common_tools import edit_file, execute_command

    timeout = execute_command._tool_definition.parameters["timeout"]
    assert timeout["description"].startswith("Seconds (max 600; values above are clamped).")
    assert edit_file._tool_definition.parameters["start_line"]["description"].startswith(
        "First line of the search scope (1-based, inclusive)."
    )


def test_search_files_parameters_are_all_explained():
    from abstractcore.tools.common_tools import search_files

    params = search_files._tool_definition.parameters
    for name, meta in params.items():
        assert meta.get("description"), f"search_files.{name} has no description"
    assert "default ignore set" in params["ignore_dirs"]["description"]
    # "required;" openers are dropped: the schema already says (required)
    assert params["pattern"]["description"] == "Regular expression pattern to search for."
    assert "files_with_matches" in params["output_mode"]["description"]


def test_every_builtin_tool_parameter_is_described():
    """Goes red the moment a tool gains a parameter without an Args entry (or an explicit description)."""
    missing = [
        f"{d.name}.{name}"
        for d in list_builtin_tool_inventory()
        for name, meta in (d.parameters or {}).items()
        if not (isinstance(meta, dict) and str(meta.get("description") or "").strip())
    ]
    assert missing == [], f"built-in tool parameters without a description: {missing}"


def test_parameter_gist_keeps_abbreviations_and_caps_length():
    from abstractcore.tools.parser import _parameter_gist

    assert _parameter_gist('Regional preference, e.g. "us-en" or "fr-fr". Second sentence.') == 'Regional preference, e.g. "us-en" or "fr-fr".'
    assert _parameter_gist("Seconds (max 600; values above are clamped). Time limit before the kill.") == "Seconds (max 600; values above are clamped)."
    assert _parameter_gist("Max. three items etc. are fine. Next.") == "Max. three items etc. are fine."
    long = "word " * 60
    gist = _parameter_gist(long)
    assert len(gist) <= 164 and gist.endswith("...")
    assert _parameter_gist(None) == "" and _parameter_gist("   ") == ""


def test_native_payload_and_prompt_carry_the_meaning():
    from abstractcore.tools.common_tools import search_files

    handler = UniversalToolHandler("gpt-4o")  # a native-tools model: the native lane is gated on support
    payload = handler.prepare_tools_for_native([search_files])
    props = payload[0]["function"]["parameters"]["properties"]
    assert props["ignore_dirs"]["description"].startswith("Comma-separated directory names to skip")

    compact = _format_parameters_compact(search_files._tool_definition.parameters)
    assert "ignore_dirs: string (optional) - Comma-separated directory names to skip" in compact
    # first sentence only in the prompt; the full text stays in the schema
    assert "Saves a follow-up read_file" not in compact
    assert "Saves a follow-up read_file" in search_files._tool_definition.parameters["context_lines"]["description"]

    prompt = format_tool_prompt([ToolDefinition.from_function(search_files)], model_name="qwen3.8-27b")
    assert "Comma-separated directory names to skip" in prompt
