"""analyze_code: the keyword-block engine seam + the languages riding it.

Companion to test_analyze_code_languages.py. That file pins the original
table and the never-refuse fallback; this one pins

  1. the ABSTRACTION — `end`-block extents and block-comment regions used to
     be hardcoded ruby-isms inside EndKeywordExtentIndex, which is why lua
     and elixir could not be added as DATA (the module's design rule 1).
     They are LanguageSpec fields now, so ruby must come through the
     refactor byte-identical and a new keyword-block language must stay a
     table entry;
  2. the languages added on top of that seam.

The bias under test throughout is the module's own: a MISS costs an outline
entry, a FALSE POSITIVE costs trust. Several tests below assert that
something is deliberately *absent*, and those are the load-bearing ones — a
wrong line range sends the agent's next read_file to the wrong block, which
is strictly worse than listing nothing.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from abstractcore.tools import code_analysis as ca
from abstractcore.tools.common_tools import analyze_code


def _lines(out: str) -> list[str]:
    return [l.rstrip() for l in out.split("\n")]


# ---------------------------------------------------------------------------
# 1. The seam: ruby parity, shared comment skipping, spec conformance
# ---------------------------------------------------------------------------


def test_ruby_survives_the_de_rubyfication_of_the_end_engine(tmp_path: Path) -> None:
    # Ruby's four hardcoded regexes became spec DATA. Its outline must not
    # move: this file exercises every one of them at once — a one-line
    # `def; end` (end_inline_close_re), an assigned `case` block
    # (end_assigned_openers), an `each do |x|` block (end_inline_block_re)
    # and a `=begin` comment (block_comment_pairs).
    f = tmp_path / "svc.rb"
    f.write_text(
        "class Svc\n"                      # 1
        "  def quick; :ok; end\n"          # 2  one-liner, self-closing
        "\n"                               # 3
        "  def pick(n)\n"                  # 4
        "    label = case n\n"             # 5  assigned opener
        "            when 1 then :one\n"   # 6
        "            end\n"                # 7
        "    label\n"                      # 8
        "  end\n"                          # 9
        "\n"                               # 10
        "  def walk(rows)\n"               # 11
        "    rows.each do |r|\n"           # 12 inline block opener
        "      puts r\n"                   # 13
        "    end\n"                        # 14
        "  end\n"                          # 15
        "end\n"                            # 16
    )
    out = analyze_code(file_path=str(f))
    got = _lines(out)
    assert "  - 2: quick()" in got, "one-line def must self-close, not swallow the next def"
    assert "  - 4-9: pick(n)" in got, "an assigned `case` must close before the def does"
    assert "  - 11-15: walk(rows)" in got, "a `do |r|` block must not leak its end"
    assert "  - 1-16: Svc" in got


def test_ruby_begin_comment_now_hides_declarations_from_the_outline(tmp_path: Path) -> None:
    # The extent index always skipped `=begin` blocks; the DECLARATION loop
    # did not, so a commented-out `def` was listed as live code with a range
    # pointing into a comment. One BlockCommentSkipper now serves both.
    f = tmp_path / "m.rb"
    f.write_text(
        "def live\n"
        "  1\n"
        "end\n"
        "=begin\n"
        "def commented_out\n"
        "  2\n"
        "end\n"
        "=end\n"
    )
    out = analyze_code(file_path=str(f))
    assert "live" in out
    assert "commented_out" not in out, "a def inside =begin/=end is a comment, not a declaration"


@pytest.mark.parametrize(
    "name, text, ghost",
    [
        ("m.rb", "=begin\ndef ghost_fn\nend\n=end\ndef real\nend\n", "ghost_fn"),
        ("m.lua", "--[[\nfunction ghost_fn()\nend\n]]\nfunction real()\nend\n", "ghost_fn"),
        ("m.pm", "=head1 NAME\n\nsub ghost_fn { }\n\n=cut\n\nsub real { 1 }\n", "ghost_fn"),
        ("m.ps1", "<#\nfunction ghost_fn { }\n#>\nfunction real { }\n", "ghost_fn"),
        ("m.hs", "{-\nghost_fn :: Int\n-}\nreal :: Int\n", "ghost_fn"),
        ("m.xml", "<root>\n<!-- <Item name=\"ghost_fn\" /> -->\n<Item name=\"real\" />\n</root>\n", "ghost_fn"),
    ],
)
def test_block_comment_regions_hide_declarations_in_every_language(
    tmp_path: Path, name: str, text: str, ghost: str
) -> None:
    # One skipper, six comment syntaxes. Each of these languages carries a
    # `block_comment_pairs` entry; if the shared seam breaks, they all break
    # together and this parametrisation says which.
    f = tmp_path / name
    f.write_text(text)
    out = analyze_code(file_path=str(f))
    assert "real" in out, f"{name}: live declaration must still be listed"
    assert ghost not in out, f"{name}: declaration inside a comment region must not be outlined"


def test_unterminated_block_comment_degrades_quietly_instead_of_crashing(tmp_path: Path) -> None:
    # An unclosed `--[[` swallows the rest of the file. That is the correct
    # reading of the source (lua would too), and the tool must still answer
    # rather than raise — never-refuse is the module's second design rule.
    f = tmp_path / "broken.lua"
    f.write_text("function head()\nend\n--[[ never closed\nfunction tail()\nend\n")
    out = analyze_code(file_path=str(f))
    assert not out.startswith("Error")
    assert "head" in out
    assert "tail" not in out


def test_keyword_block_specs_declare_the_data_the_engine_needs() -> None:
    # block_style="end" with no openers silently yields NO extents at all —
    # every declaration would collapse to a bare line number and nobody
    # would notice until an agent read the wrong range.
    for spec in ca._SPECS:
        if spec.block_style == "end":
            assert spec.end_block_openers, f"{spec.name}: block_style='end' needs end_block_openers"
    # And the reverse: the end-only knobs are meaningless on other families,
    # so a stray one means a spec was written against the wrong engine.
    for spec in ca._SPECS:
        if spec.block_style != "end":
            assert not spec.end_block_openers, f"{spec.name}: end_block_openers on a {spec.block_style} spec"
            assert not spec.end_assigned_openers, f"{spec.name}: end_assigned_openers on a {spec.block_style} spec"


def test_every_spec_pattern_exposes_the_name_group_the_engine_reads() -> None:
    # The engine renders `m.groupdict()["name"]`. A pattern without that
    # group matches, yields an empty name, and `break`s out of the decl loop
    # — silently suppressing every later pattern on that line.
    for spec in ca._SPECS:
        for pat in spec.import_patterns:
            assert "name" in pat.groupindex, f"{spec.name}: import pattern lacks (?P<name>…): {pat.pattern}"
        for decl in spec.decl_patterns:
            assert "name" in decl.pattern.groupindex, f"{spec.name}/{decl.kind}: decl lacks (?P<name>…)"


# ---------------------------------------------------------------------------
# 2. Per-language pins. Each targets the shape most likely to produce a
#    WRONG RANGE or a FALSE POSITIVE in that language, not a happy path.
# ---------------------------------------------------------------------------


def test_lua_anonymous_callback_end_does_not_steal_the_enclosing_range(tmp_path: Path) -> None:
    # `..., function()` … `end)` is the dominant shape in neovim/love2d
    # config. It is neither line-anchored nor an assignment, so without
    # end_inline_block_re the bare `end)` pops the ENCLOSING function and
    # every range below it shifts by one block.
    f = tmp_path / "init.lua"
    f.write_text(
        "local M = {}\n"                              # 1
        "function M.setup(opts)\n"                    # 2
        "  vim.keymap.set('n', 'x', function()\n"     # 3
        "    doit()\n"                                # 4
        "  end)\n"                                    # 5
        "  return opts\n"                             # 6
        "end\n"                                       # 7
        "\n"                                          # 8
        "function M.after()\n"                        # 9
        "  return 1\n"                                # 10
        "end\n"                                       # 11
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2-7: M.setup(opts)" in got
    assert "  - 9-11: M.after()" in got, "the callback's `end)` must not consume the next function"
    assert "  - 1: M" in got  # `local M = {}` lists as a table


def test_lua_forms_that_close_on_their_own_line(tmp_path: Path) -> None:
    f = tmp_path / "u.lua"
    f.write_text(
        "local util = require('util')\n"              # 1
        "local MAX = 10\n"                            # 2
        "M.quick = function(x) return x * 2 end\n"    # 3
        "local function slow(a, b)\n"                 # 4
        "  return a + b\n"                            # 5
        "end\n"                                       # 6
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: util" in got
    assert "  - 3: M.quick(x)" in got, "a one-line `f = function(...) ... end` must self-close"
    assert "  - 4-6: slow(a, b)" in got
    assert "  - 2: MAX" in got


def test_elixir_do_colon_oneliner_takes_no_range_and_steals_no_end(tmp_path: Path) -> None:
    # `def ping, do: :pong` is opener-SHAPED with no block. Pushing it would
    # hand the next `end` to it and corrupt every range below — the reason
    # end_bodyless_re exists.
    f = tmp_path / "app.ex"
    f.write_text(
        "defmodule App do\n"                     # 1
        "  def ping, do: :pong\n"                # 2
        "\n"                                     # 3
        "  def run(list) do\n"                   # 4
        "    Enum.map(list, fn x ->\n"           # 5
        "      x\n"                              # 6
        "    end)\n"                             # 7
        "  end\n"                                # 8
        "\n"                                     # 9
        "  def last do\n"                        # 10
        "    :ok\n"                              # 11
        "  end\n"                                # 12
        "end\n"                                  # 13
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2: ping()" in got, "a `, do:` one-liner has no block, so it gets a line, not a range"
    assert "  - 4-8: run(list)" in got, "`fn x ->` … `end)` must close inside its own def"
    assert "  - 10-12: last()" in got, "ranges below a one-liner and a callback must stay correct"
    assert "  - 1-13: App" in got


def test_dart_widget_tree_call_is_not_a_method_definition(tmp_path: Path) -> None:
    # `setState(() {` has a same-line `{` and would satisfy requires_brace.
    # Restricting params to `[^()]*` (no nested parens) is what rejects it —
    # at the documented cost of missing function-typed parameters.
    f = tmp_path / "w.dart"
    f.write_text(
        "class W extends StatefulWidget {\n"     # 1
        "  Widget build(BuildContext c) {\n"     # 2
        "    setState(() {\n"                    # 3
        "      counter++;\n"                     # 4
        "    });\n"                              # 5
        "    return Text('hi');\n"               # 6
        "  }\n"                                  # 7
        "}\n"                                    # 8
    )
    out = analyze_code(file_path=str(f))
    assert "  - 2-7: build(BuildContext c)" in _lines(out)
    assert "setState" not in out, "a call with a nested paren is not a definition"
    assert "Text" not in out


def test_scala_case_class_object_and_selector_import(tmp_path: Path) -> None:
    f = tmp_path / "S.scala"
    f.write_text(
        "import scala.collection.mutable.{Map => MMap}\n"   # 1
        "case class Started(id: Long)\n"                    # 2
        "object Svc {\n"                                    # 3
        "  def start(n: String): Long = {\n"                # 4
        "    1L\n"                                          # 5
        "  }\n"                                             # 6
        "}\n"                                               # 7
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: scala.collection.mutable.{Map => MMap}" in got, "the selector braces belong to the import name"
    assert "  - 2: Started" in got, "a brace-less case class must not borrow the next block's extent"
    assert "  - 3-7: Svc" in got
    assert "  - 4-6: start(n: String)" in got


def test_groovy_gradle_configuration_blocks_are_the_navigable_units(tmp_path: Path) -> None:
    f = tmp_path / "build.gradle"
    f.write_text(
        "plugins {\n"                          # 1
        "    id 'java'\n"                      # 2
        "}\n"                                  # 3
        "\n"                                   # 4
        "dependencies {\n"                     # 5
        "    implementation 'g:a:1'\n"         # 6
        "}\n"                                  # 7
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "language: groovy" in got
    assert "  - 1-3: plugins" in got
    assert "  - 5-7: dependencies" in got


def test_objectivec_interface_gets_no_brace_extent_and_methods_no_fake_parens(tmp_path: Path) -> None:
    # @interface/@implementation close with `@end`, not `}` — a brace extent
    # would point at the first ivar block. And an ObjC selector is not a
    # call, so it must not be rendered `- (void)foo()`.
    f = tmp_path / "V.m"
    f.write_text(
        "@implementation MyView\n"                 # 1
        "\n"                                       # 2
        "- (void)layoutSubviews {\n"               # 3
        "    [super layoutSubviews];\n"            # 4
        "}\n"                                      # 5
        "\n"                                       # 6
        "@end\n"                                   # 7
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: MyView" in got, "@implementation must carry a line number only"
    assert "  - 3-5: - (void)layoutSubviews" in got
    assert "- (void)layoutSubviews()" not in "\n".join(got)


def test_zig_lists_imports_tests_and_only_top_level_constants(tmp_path: Path) -> None:
    f = tmp_path / "b.zig"
    f.write_text(
        'const std = @import("std");\n'            # 1
        "const MAX: usize = 100;\n"                # 2
        "pub fn build(b: *std.Build) void {\n"     # 3
        "    const local = b.opts();\n"            # 4
        "    _ = local;\n"                         # 5
        "}\n"                                      # 6
        'test "adds" {\n'                          # 7
        "    try expect(1 == 1);\n"                # 8
        "}\n"                                      # 9
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: std" in got, "@import targets are imports, not constants"
    assert "  - 2: MAX" in got
    assert "  - 3-6: build(b: *std.Build)" in got
    assert '  - 7-9: test "adds"' in got
    assert "local" not in "\n".join(got), "function-body consts would flood the section cap"


def test_perl_lists_subs_and_packages_around_pod(tmp_path: Path) -> None:
    f = tmp_path / "R.pm"
    f.write_text(
        "package Acme::R;\n"                       # 1
        "use List::Util qw(sum);\n"                # 2
        "our $VERSION = '1.0';\n"                  # 3
        "\n"                                       # 4
        "sub total {\n"                            # 5
        "    return 1;\n"                          # 6
        "}\n"                                      # 7
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: Acme::R" in got
    assert "  - 2: List::Util" in got
    assert "  - 3: $VERSION" in got
    assert "  - 5-7: total()" in got


def test_haskell_signature_carries_its_type_and_instance_head_stops_at_where(tmp_path: Path) -> None:
    # Haskell is layout-scoped: there is no delimiter to measure, so entries
    # are line numbers only (block_style="none"). The TYPE is what makes the
    # entry worth reading, so it rides in the rendered signature.
    f = tmp_path / "T.hs"
    f.write_text(
        "module Acme.T (area) where\n"             # 1
        "import qualified Data.Map as M\n"         # 2
        "data Shape = Circle Double\n"             # 3
        "instance Show Shape where\n"              # 4
        "  show _ = \"s\"\n"                       # 5
        "area :: Shape -> Double\n"                # 6
        "area _ = 1.0\n"                           # 7
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: Acme.T" in got
    assert "  - 2: Data.Map" in got
    assert "  - 3: Shape" in got
    assert "  - 4: Show Shape" in got, "`where` is syntax, not part of the instance head"
    assert "  - 6: area(Shape -> Double)" in got


def test_powershell_lists_functions_filters_and_scoped_variables(tmp_path: Path) -> None:
    f = tmp_path / "D.ps1"
    f.write_text(
        "Import-Module Az.Accounts\n"              # 1
        "$script:RetryCount = 3\n"                 # 2
        "class Deployer {\n"                       # 3
        "    [string]$Name\n"                      # 4
        "}\n"                                      # 5
        "function Invoke-Deploy {\n"               # 6
        "    Write-Host 'go'\n"                    # 7
        "}\n"                                      # 8
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: Az.Accounts" in got
    assert "  - 2: RetryCount" in got
    assert "  - 3-5: Deployer" in got
    assert "  - 6-8: Invoke-Deploy()" in got


def test_powershell_script_without_functions_still_outlines_its_interface(tmp_path: Path) -> None:
    # Found against CPython's own msi build scripts: a standalone .ps1
    # declares NO functions, so a function-only spec answered with an empty
    # outline — a useless answer, not an honest one. For a script the
    # `param` block IS the interface and column-0 assignments ARE the
    # structure. The leading <# … #> help comment must still be skipped.
    f = tmp_path / "make_cat.ps1"
    f.write_text(
        "<#\n"                                          # 1
        ".Synopsis\n"                                   # 2
        "    function NotReal { }\n"                    # 3
        "#>\n"                                          # 4
        "param(\n"                                      # 5
        "    [string]$catalog,\n"                       # 6
        "    [switch]$sign\n"                           # 7
        ")\n"                                           # 8
        "\n"                                            # 9
        "$tools = Split-Path -parent $PSCommandPath\n"  # 10
        "Import-Module $tools\\sdktools.psm1\n"         # 11
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 5: param" in got, "a script's param block is its navigable interface"
    assert "  - 10: tools" in got
    assert "  - 11: $tools\\sdktools.psm1" in got, "imports may be variable paths"
    assert "NotReal" not in "\n".join(got), "the help comment is not code"


def test_ini_lists_sections_and_column_zero_keys_only(tmp_path: Path) -> None:
    # An indented line under a key is a CONTINUATION of that value, not a
    # new key — same column-0 rule the YAML lane documents.
    f = tmp_path / "setup.cfg"
    f.write_text(
        "[metadata]\n"                             # 1
        "name = acme\n"                            # 2
        "\n"                                       # 3
        "[options]\n"                              # 4
        "install_requires =\n"                     # 5
        "    requests>=2\n"                        # 6
        "    rich\n"                               # 7
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: [metadata]" in got
    assert "  - 4: [options]" in got
    assert "  - 2: name" in got
    assert "  - 5: install_requires" in got
    assert "requests" not in "\n".join(got), "a wrapped value is not a key"


def test_xml_lists_identity_attributes_and_stays_out_of_the_deep_tree(tmp_path: Path) -> None:
    f = tmp_path / "App.csproj"
    f.write_text(
        '<Project Sdk="Microsoft.NET.Sdk">\n'                              # 1
        '  <Import Project="..\\common.props" />\n'                        # 2
        "  <ItemGroup>\n"                                                  # 3
        '    <PackageReference Include="Newtonsoft.Json" Version="13" />\n' # 4
        "  </ItemGroup>\n"                                                 # 5
        "</Project>\n"                                                     # 6
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "language: xml" in got
    assert "  - 2: ..\\common.props" in got, "an <Import Project> is a reference, not a leaf element"
    assert "  - 1: Project" in got
    assert "  - 3: ItemGroup" in got
    assert "  - 4: PackageReference(Newtonsoft.Json)" in got


def test_graphql_brace_less_scalar_does_not_borrow_the_next_types_extent(tmp_path: Path) -> None:
    f = tmp_path / "s.graphql"
    f.write_text(
        "scalar DateTime\n"                        # 1
        "\n"                                       # 2
        "type User {\n"                            # 3
        "  id: ID!\n"                              # 4
        "}\n"                                      # 5
        "\n"                                       # 6
        "fragment UserFields on User {\n"          # 7
        "  id\n"                                   # 8
        "}\n"                                      # 9
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1: DateTime" in got, "a brace-less scalar must not take the next block's range"
    assert "  - 3-5: User" in got
    assert "  - 7-9: fragment UserFields" in got


# ---------------------------------------------------------------------------
# 3. Detection routing and cost
# ---------------------------------------------------------------------------


def test_new_specs_do_not_steal_extensions_from_the_deep_analyzers() -> None:
    # python/javascript/html/r keep bespoke analyzers in common_tools.py
    # (AST + ruff, import resolution, ids). If a table entry claimed one of
    # their extensions the file would silently lose that deeper analysis.
    for ext in (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".html", ".htm", ".xhtml", ".r", ".rmd"):
        assert ca.spec_for(path=Path("x" + ext)) is None, f"{ext} belongs to a deep analyzer lane"


def test_no_extension_is_claimed_by_two_specs() -> None:
    # _EXT_TO_SPEC uses setdefault, so a duplicate resolves silently by TABLE
    # ORDER — a reordering would then change behaviour with no test failing.
    seen: dict[str, str] = {}
    for spec in ca._SPECS:
        for ext in spec.extensions:
            assert ext not in seen, f"{ext} claimed by both {seen[ext]} and {spec.name}"
            seen[ext] = spec.name


def test_dotfiles_resolve_by_filename_because_their_suffix_is_empty() -> None:
    # Path(".editorconfig").suffix == "" — an extensions entry would never
    # fire, so these MUST be carried in `filenames`.
    assert Path(".editorconfig").suffix == ""
    for name, lang in ((".editorconfig", "ini"), ("tox.ini", "ini"), ("Dockerfile", "dockerfile")):
        spec = ca.spec_for(path=Path(name))
        assert spec is not None and spec.name == lang, f"{name} should resolve to {lang}"


def test_headers_stay_with_c_and_dot_m_is_decided_by_content(tmp_path: Path) -> None:
    # `.m` belongs to Objective-C and to MATLAB/Octave in roughly equal
    # measure. Claiming it outright labelled a MATLAB file `objectivec` and
    # then listed NOTHING — a confident lie, and worse than the generic lane
    # which at least surfaces its `function` lines. So the extension is
    # claimed only on positive evidence.
    assert ca.spec_for(path=Path("a.h")).name == "c", "ObjC must not steal .h from C"
    assert ca.spec_for(path=Path("a.m")) is None, "no text, no evidence, no claim"

    objc = tmp_path / "View.m"
    objc.write_text("#import <UIKit/UIKit.h>\n@implementation V\n- (void)go {\n}\n@end\n")
    out = analyze_code(file_path=str(objc))
    assert "language: objectivec" in out
    assert "  - 3-4: - (void)go" in _lines(out)

    matlab = tmp_path / "analyze.m"
    matlab.write_text(
        "% analyze.m -- MATLAB script\n"
        "function result = analyze(data, opts)\n"
        "    result = mean(data, 1);\n"
        "end\n"
    )
    out_m = analyze_code(file_path=str(matlab))
    assert "language: objectivec" not in out_m, "a MATLAB file must not be labelled Objective-C"
    assert not out_m.startswith("Error")
    assert "analyze" in out_m, "the generic lane still surfaces its top-level lines"


def test_no_spec_pattern_backtracks_catastrophically(tmp_path: Path) -> None:
    # This module has TWO prior ReDoS incidents in its comments (reviewer C,
    # P0-1 and P1-1): a plain C file of `* * * *` banner rows hung the tool.
    # Every spec is re-run here over input engineered to maximise ambiguity
    # for modifier-star and atomic-emulation patterns.
    hostile = "\n".join(
        [
            "*" + " *" * 200,
            "private " * 120 + "def x",
            "final static public " * 60 + "foo(",
            "a" * 400 + " " + "b" * 400 + "(",
            "const " * 150 + "X = struct {",
            "(" * 150 + ")" * 150,
            "-" * 300,
            "<" + "a" * 300 + " " + "name=" * 60,
            "instance " + "Show " * 120,
            "=" * 200,
        ]
        * 3
    )
    for spec in ca._SPECS:
        f = tmp_path / f"hostile_{spec.name}.txt"
        f.write_text(hostile)
        start = time.monotonic()
        out = analyze_code(file_path=str(f), language=spec.name)
        elapsed = time.monotonic() - start
        assert not out.startswith("Error")
        # Generous vs. a real machine, unmissable vs. exponential blowup.
        assert elapsed < 2.0, f"{spec.name} took {elapsed:.2f}s on 30 hostile lines"


def test_the_docstring_language_list_matches_the_live_table() -> None:
    # `language=` is a documented argument: the docstring IS the tool's
    # promise to the model about what it may pass. A table entry missing
    # from the list is an unreachable feature; a listed name missing from
    # the table is a lie that costs the model a turn to discover.
    import inspect
    import re

    doc = inspect.getdoc(analyze_code) or ""
    segment = doc.split("Outline engine:")[1].split("Anything")[0]
    documented = set(re.findall(r'"([a-z]+)"', segment))
    assert documented == set(ca.known_language_names())


# ---------------------------------------------------------------------------
# 4. The unified SkipRegion seam.
#
# Comments, long strings, heredocs, fences and `#if 0` blocks were five
# separate mechanisms; the heredoc regex alone existed in three copies and
# the block-comment skipper was a fourth mechanism with different matching
# rules. They disagreed, and every test below is one way that disagreement
# reached the user. One SkipRegion type now feeds all four passes — the
# declaration loop, both extent indexes, and the balance lint.
# ---------------------------------------------------------------------------


def test_indented_perl_equals_is_code_not_pod(tmp_path: Path) -> None:
    # THE WORST BUG THIS FILE GUARDS. Openers were matched on the LEFT-
    # STRIPPED line, so perl's POD directive `=for` matched the continuation
    # `=format_date($t)` — a prefix, indented, mid-expression. Two of three
    # subs silently vanished from a file `perl -c` calls valid, and the
    # summary counted them out. POD is column-0 only, which `^` expresses.
    f = tmp_path / "R.pm"
    f.write_text(
        "package My::Report;\n"          # 1
        "\n"                             # 2
        "sub header {\n"                 # 3
        "    my ($t) = @_;\n"            # 4
        "    my $when\n"                 # 5
        "        =format_date($t);\n"    # 6  indented `=` → code, not POD
        "    return $when;\n"            # 7
        "}\n"                            # 8
        "\n"                             # 9
        "sub body {\n"                   # 10
        "    return 'body';\n"           # 11
        "}\n"                            # 12
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 3-8: header()" in got
    assert "  - 10-12: body()" in got, "an indented `=` continuation must not open POD"
    assert "summary: imports=0; modules=1; functions=2" in got


def test_real_pod_block_is_still_skipped(tmp_path: Path) -> None:
    # The counterpart to the test above: fixing the false positive must not
    # cost the true one. A column-0 `=head1` really does open POD.
    f = tmp_path / "R.pm"
    f.write_text(
        "=head1 NAME\n"
        "\n"
        "sub documented_not_real { }\n"
        "\n"
        "=cut\n"
        "\n"
        "sub real { 1 }\n"
    )
    out = analyze_code(file_path=str(f))
    assert "documented_not_real" not in out
    assert "real()" in out


@pytest.mark.parametrize(
    "name, text",
    [
        ("m.rb", "=begin\nTODO: rewrite against the new API\n=end\ndef run\nend\n"),
        ("m.xml", "<root>\n  <!-- TODO: migrate this to v2 -->\n</root>\n"),
        ("m.lua", "--[[ TODO: finish the refactor ]]\nfunction go()\nend\n"),
    ],
)
def test_todo_markers_inside_comment_regions_are_still_reported(
    tmp_path: Path, name: str, text: str
) -> None:
    # A TODO lives in a COMMENT — that is the whole point of the marker.
    # Skipping comment regions before the TODO scan dropped every one of
    # them, silently turning `todo_markers=N` into `none`. The scan runs on
    # the raw line and must therefore run BEFORE the skip.
    f = tmp_path / name
    f.write_text(text)
    out = analyze_code(file_path=str(f))
    assert "todo_markers=1" in out, f"{name}: a TODO in a comment is still a TODO"
    assert "TODO" in out


def test_todo_inside_a_heredoc_body_is_data_not_a_marker(tmp_path: Path) -> None:
    # The other side of the rule: text inside a heredoc/string is DATA, and
    # counting it would invent TODOs that nobody wrote.
    f = tmp_path / "s.sh"
    f.write_text("cat <<EOF\nTODO: this is sample text, not a task\nEOF\n")
    out = analyze_code(file_path=str(f))
    assert "todo_markers" not in out


def test_elixir_multiline_guard_does_not_open_a_second_block(tmp_path: Path) -> None:
    # `mix format` breaks a long guard onto its own line, so `when … do` is
    # the FORMATTER'S OWN OUTPUT, not an exotic shape. Counting that `do`
    # pushed a second opener; the function then swallowed the module's `end`
    # and reported itself as spanning the whole file.
    f = tmp_path / "g.ex"
    f.write_text(
        "defmodule My.Guards do\n"                                  # 1
        "  def normalize(value)\n"                                  # 2
        "      when is_binary(value) and byte_size(value) > 0 do\n"  # 3
        "    String.trim(value)\n"                                  # 4
        "  end\n"                                                   # 5
        "\n"                                                        # 6
        "  def finish(x) do\n"                                      # 7
        "    x\n"                                                   # 8
        "  end\n"                                                   # 9
        "end\n"                                                     # 10
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2-5: normalize(value)" in got, "a `when … do` continuation opens no block"
    assert "  - 7-9: finish(x)" in got
    assert "  - 1-10: My.Guards" in got


def test_elixir_quote_do_colon_without_a_comma_is_still_bodyless(tmp_path: Path) -> None:
    # `quote do: :ok` reaches `do:` with no preceding comma, so a
    # comma-anchored rule missed it and the macro claimed the module's end.
    f = tmp_path / "m.ex"
    f.write_text(
        "defmodule My.Macros do\n"        # 1
        "  defmacro assert_ok(x) do\n"    # 2
        "    quote do: :ok\n"             # 3
        "  end\n"                         # 4
        "\n"                              # 5
        "  def after_macro(y) do\n"       # 6
        "    y\n"                         # 7
        "  end\n"                         # 8
        "end\n"                           # 9
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2-4: assert_ok(x)" in got
    assert "  - 6-8: after_macro(y)" in got


def test_elixir_triple_quote_heredoc_body_is_not_code(tmp_path: Path) -> None:
    # A bare `end` inside a docstring or embedded SQL closed the enclosing
    # def, so read_file(2,4) handed the agent a body cut off mid-string.
    f = tmp_path / "d.ex"
    f.write_text(
        "defmodule My.Doc do\n"            # 1
        "  def describe(x) do\n"           # 2
        '    text = """\n'                 # 3
        "    end\n"                        # 4  <- text, not a terminator
        '    """\n'                        # 5
        "    text <> to_string(x)\n"       # 6
        "  end\n"                          # 7
        "\n"                               # 8
        "  def other(y) do\n"              # 9
        "    y\n"                          # 10
        "  end\n"                          # 11
        "end\n"                            # 12
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2-7: describe(x)" in got
    assert "  - 9-11: other(y)" in got
    assert "  - 1-12: My.Doc" in got


def test_lua_long_string_contents_are_not_declarations(tmp_path: Path) -> None:
    # An embedded help/SQL string held lines that LOOK like declarations.
    # They were outlined as real ones — two functions that do not exist,
    # one with a range ending on an unrelated `end`. The module's stated
    # bias is to prefer misses; inventing declarations is the opposite.
    f = tmp_path / "h.lua"
    f.write_text(
        "local M = {}\n"                          # 1
        "function M.help()\n"                     # 2
        "  return [[\n"                           # 3
        "function old_api(x)\n"                   # 4  <- text
        "local helper = function(y)\n"            # 5  <- text
        "]]\n"                                    # 6
        "end\n"                                   # 7
        "\n"                                      # 8
        "function M.run()\n"                      # 9
        "  return 1\n"                            # 10
        "end\n"                                   # 11
    )
    out = analyze_code(file_path=str(f))
    got = _lines(out)
    assert "old_api" not in out and "helper" not in out, "string contents are not declarations"
    assert "  - 2-7: M.help()" in got
    assert "  - 9-11: M.run()" in got


def test_lua_level_one_long_bracket_comment_is_a_region(tmp_path: Path) -> None:
    # `--[=[` is the CONVENTIONAL form precisely when the body contains
    # `]]`, so covering only `--[[` and `--[==[` missed the middle case.
    f = tmp_path / "c.lua"
    f.write_text("--[=[\nfunction ghost(x)\nend\n]=]\nfunction real()\nend\n")
    out = analyze_code(file_path=str(f))
    assert "ghost" not in out
    assert "real" in out


def test_haskell_block_comments_nest(tmp_path: Path) -> None:
    # GHC nests `{- … -}`. Tracking a single closer let the INNER `-}` end
    # the outer region, so commented-out code below it was listed as live.
    f = tmp_path / "n.hs"
    f.write_text(
        "module Demo where\n"              # 1
        "\n"                               # 2
        "{- deprecated:\n"                 # 3
        "{- nested note -}\n"              # 4
        "area :: Double -> Double\n"       # 5
        "-}\n"                             # 6
        "\n"                               # 7
        "perimeter :: Double -> Double\n"  # 8
    )
    out = analyze_code(file_path=str(f))
    assert "area" not in out, "an inner -} must not close the outer comment"
    assert "  - 8: perimeter(Double -> Double)" in _lines(out)


def test_haskell_language_pragma_is_not_an_unclosed_comment(tmp_path: Path) -> None:
    # `{-# LANGUAGE … #-}` opens and closes on one line; treating it as an
    # open region would swallow the module.
    f = tmp_path / "p.hs"
    f.write_text("{-# LANGUAGE OverloadedStrings #-}\nmodule P where\narea :: Int\n")
    out = analyze_code(file_path=str(f))
    assert "unterminated" not in out
    assert "  - 2: P" in _lines(out)


@pytest.mark.parametrize(
    "name, text",
    [
        ("c.ps1", "<#\n.SYNOPSIS\n  Emits a literal like @{ Name = 'x'\n#>\nfunction Get-Config {\n    return @{ Name = 'x' }\n}\n"),
        ("c.pm", "=head1 NAME\n\nAn unbalanced { brace in prose\n\n=cut\n\nsub real { 1 }\n"),
    ],
)
def test_balance_lint_honours_the_same_regions_the_outline_does(
    tmp_path: Path, name: str, text: str
) -> None:
    # The lint and the outline are two passes over one file; when only the
    # outline skipped comments, the tool reported `unbalanced {}` in the
    # same breath as a note promising it had ignored that comment. Both
    # passes now share one tracker, so they cannot contradict each other.
    f = tmp_path / name
    f.write_text(text)
    out = analyze_code(file_path=str(f))
    assert "delimiters=ok" in out, f"{name}: braces inside a comment are not code"
    assert "unbalanced" not in out


def test_brace_extents_ignore_heredoc_bodies(tmp_path: Path) -> None:
    # BraceExtentIndex saw NO skip regions at all before the unification, so
    # a `{` inside a heredoc shifted every extent below it.
    f = tmp_path / "s.sh"
    f.write_text(
        "setup() {\n"              # 1
        "  cat <<EOF\n"            # 2
        "  { unbalanced brace\n"   # 3  <- heredoc body, not code
        "EOF\n"                    # 4
        "}\n"                      # 5
        "\n"                       # 6
        "teardown() {\n"           # 7
        "  :\n"                    # 8
        "}\n"                      # 9
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1-5: setup()" in got
    assert "  - 7-9: teardown()" in got


def test_ruby_end_marker_only_closes_at_column_zero(tmp_path: Path) -> None:
    # Matching the closer as a bare substring let prose containing
    # "foo=end_marker" end a `=begin` block early, resurrecting the
    # commented-out code beneath it.
    f = tmp_path / "m.rb"
    f.write_text(
        "def live\n"                            # 1
        "end\n"                                 # 2
        "=begin\n"                              # 3
        "mentions foo=end_marker in prose\n"    # 4
        "def commented_out\n"                   # 5
        "end\n"                                 # 6
        "=end\n"                                # 7
        "def after_comment\n"                   # 8
        "end\n"                                 # 9
    )
    out = analyze_code(file_path=str(f))
    assert "commented_out" not in out
    assert "  - 8-9: after_comment()" in _lines(out)


def test_ruby_then_end_oneliner_does_not_swallow_its_method(tmp_path: Path) -> None:
    f = tmp_path / "c.rb"
    f.write_text(
        "class Calc\n"                    # 1
        "  def compute(x)\n"              # 2
        "    if x > 0 then puts x end\n"  # 3
        "    x * 2\n"                     # 4
        "  end\n"                         # 5
        "\n"                              # 6
        "  def other(y)\n"                # 7
        "    y\n"                         # 8
        "  end\n"                         # 9
        "end\n"                           # 10
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2-5: compute(x)" in got
    assert "  - 7-9: other(y)" in got


def test_groovy_control_flow_is_not_a_configuration_block(tmp_path: Path) -> None:
    f = tmp_path / "b.gradle"
    f.write_text(
        "dependencies {\n"                     # 1
        "    implementation 'g:a:1'\n"         # 2
        "}\n"                                  # 3
        "\n"                                   # 4
        "try {\n"                              # 5
        "    build()\n"                        # 6
        "} catch (Exception e) {\n"            # 7
        "}\n"                                  # 8
        "\n"                                   # 9
        "static {\n"                           # 10
        "    init()\n"                         # 11
        "}\n"                                  # 12
    )
    out = analyze_code(file_path=str(f))
    blocks = out.split("blocks:")[-1]
    assert "  - 1-3: dependencies" in _lines(out)
    assert "try" not in blocks and "static" not in blocks


def test_unterminated_region_is_labelled_not_silently_dropped(tmp_path: Path) -> None:
    # Swallowing the rest of a file is the correct reading of an unclosed
    # `--[[`, but doing it SILENTLY leaves the model with a short outline
    # and no reason to doubt it.
    f = tmp_path / "u.lua"
    f.write_text("function head()\nend\n--[[ never closed\nfunction tail()\nend\n")
    out = analyze_code(file_path=str(f))
    assert "#TRUNCATION" in out and "unterminated" in out
    assert "line 3" in out
    assert "head" in out and "tail" not in out


def test_conf_is_routed_by_content_because_three_syntaxes_claim_it() -> None:
    # `.conf` means INI (systemd), XML (fontconfig) or a brace grammar
    # (nginx) depending entirely on the file.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ini = Path(d) / "systemd.conf"
        ini.write_text("[Unit]\nDescription=demo\n")
        assert "language: ini" in analyze_code(file_path=str(ini))

        xml = Path(d) / "fonts.conf"
        xml.write_text('<?xml version="1.0"?>\n<fontconfig>\n  <dir>/usr/share/fonts</dir>\n</fontconfig>\n')
        assert "language: xml" in analyze_code(file_path=str(xml))

        nginx = Path(d) / "nginx.conf"
        nginx.write_text("worker_processes 4;\nhttp {\n  server_name x;\n}\n")
        out = analyze_code(file_path=str(nginx))
        assert not out.startswith("Error")
        assert "language: ini" not in out and "language: xml" not in out


def test_skip_regions_are_internally_well_formed() -> None:
    # A region with neither `close` nor `close_group` can never end, so it
    # would eat every file in that language from its first opener onward.
    for spec in ca._SPECS:
        for region in spec.skip_regions:
            assert region.close is not None or region.close_group is not None, (
                f"{spec.name}: a SkipRegion with no terminator swallows the rest of the file"
            )
            if region.close_group is not None:
                assert region.close_group in region.open.groupindex, (
                    f"{spec.name}: close_group {region.close_group!r} is not a group of `open`"
                )


# ---------------------------------------------------------------------------
# 5. What a region hands BACK to its caller.
#
# A skip region is not only "these lines are inert" — the opening line
# usually carries real code beside the marker. Getting that wrong deleted
# declarations and invented lint errors on well-formed files, so each case
# below pins one side of the contract: the code before the opener, the code
# after a one-line closer, and whether the opener is looked for before or
# after line-comment stripping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, text, wanted, ghost",
    [
        # PowerShell: losing this line also loses its `{`, which then makes
        # the balance lint report an imbalance on a well-formed file.
        ("t.ps1", "function Get-Thing {  <# returns it #>\n    return 1\n}\n", "Get-Thing", None),
        ("t.xml", '<beans>\n  <bean id="alpha"/>   <!-- primary -->\n</beans>\n', "alpha", None),
        ("t.hs", "module H where\narea :: Double -> Double   {- see issue 123 -}\n", "area", None),
        ("t.lua", "local M = {}   --[[ module table ]]\nfunction M.run()\nend\n", "M.run", None),
    ],
)
def test_code_before_a_trailing_comment_survives(
    tmp_path: Path, name: str, text: str, wanted: str, ghost: object
) -> None:
    f = tmp_path / name
    f.write_text(text)
    out = analyze_code(file_path=str(f))
    assert wanted in out, f"{name}: a trailing comment must not delete the code before it"
    assert "unbalanced" not in out


def test_code_after_a_one_line_region_survives(tmp_path: Path) -> None:
    # `x = [[a]] .. tail` closes its region mid-line; the text on BOTH sides
    # of the region is code.
    tracker = ca.SkipRegionTracker(ca._SPEC_BY_NAME["lua"].skip_regions, ("--",))
    assert tracker.feed("local s = [[a]] .. tail") is not None


def test_a_comment_mentioning_a_heredoc_tag_opens_no_heredoc(tmp_path: Path) -> None:
    # Heredoc openers are found in the line-comment-STRIPPED text; every
    # other region here IS comment syntax and must see the raw line. Without
    # that split, a comment merely describing `cat <<EOF` swallowed the rest
    # of the file — the extent index and the old loop both stripped first.
    f = tmp_path / "d.sh"
    f.write_text(
        "#!/bin/bash\n"                                            # 1
        "# Historically this used `cat <<EOF` but we template now.\n"  # 2
        "render() {\n"                                             # 3
        "  echo hi\n"                                              # 4
        "}\n"                                                      # 5
        "\n"                                                       # 6
        "deploy() {\n"                                             # 7
        "  echo go\n"                                              # 8
        "}\n"                                                      # 9
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 3-5: render()" in got
    assert "  - 7-9: deploy()" in got, "a commented-out heredoc tag opens nothing"
    assert not any("unterminated" in l for l in got)


def test_text_after_a_heredoc_tag_is_not_counted_by_the_balance_lint(tmp_path: Path) -> None:
    # `cat <<XEOF | tr } ]` — everything after the tag belongs to the
    # command line, not to the brace count.
    f = tmp_path / "e.sh"
    f.write_text("emit() {\n  cat <<XEOF | tr } ]\nbody\nXEOF\n}\n")
    out = analyze_code(file_path=str(f))
    assert "delimiters=ok" in out
    assert "unbalanced" not in out


def test_ruby_one_line_def_without_a_semicolon_before_end(tmp_path: Path) -> None:
    # Ruby's own stdlib writes `def fu_windows?; true end` and
    # `def _do_nothing(*)end`. Requiring `; end` missed both, so the def
    # stole its module's `end` and every range below it shifted.
    f = tmp_path / "u.rb"
    f.write_text(
        "module StreamUtils_\n"                 # 1
        "  def fu_windows?; true end\n"         # 2
        "  def _do_nothing(*)end\n"             # 3
        "  def real(x)\n"                       # 4
        "    x\n"                               # 5
        "  end\n"                               # 6
        "end\n"                                 # 7
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1-7: StreamUtils_" in got, "the module must keep its own end"
    assert "  - 2: fu_windows?()" in got
    assert "  - 3: _do_nothing(*)" in got
    assert "  - 4-6: real(x)" in got


def test_ruby_while_begin_end_oneliner_still_opens_a_block(tmp_path: Path) -> None:
    # The counterpart: `while begin …; l or r end` opens TWO blocks and its
    # trailing `end` closes only the inner `begin`, so it must NOT
    # self-close. Widening the one-liner rule far enough to catch
    # `def x; true end` is exactly far enough to break this.
    f = tmp_path / "w.rb"
    f.write_text(
        "def summarize\n"                                  # 1
        "  while begin l = shift; r = shift; l or r end\n"  # 2
        "    emit(l, r)\n"                                 # 3
        "  end\n"                                          # 4
        "end\n"                                            # 5
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 1-5: summarize()" in got, "the inner `begin` end must not close the def"


def test_c_preprocessor_if_zero_is_matched_with_spacing(tmp_path: Path) -> None:
    # Real headers write `#        if 0` and `#if 0x…`. Matching the literal
    # prefix `#if 0` both missed the indented form (leaving disabled code in
    # the outline) and swallowed hex version comparisons (hiding live code).
    f = tmp_path / "h.c"
    f.write_text(
        "#        if 0\n"                       # 1
        "#          define DISABLED 1\n"        # 2
        "#        endif\n"                      # 3
        "#if 0x030700A1 <= PY_VERSION_HEX\n"    # 4
        "#  define LIVE 1\n"                    # 5
        "#endif\n"                              # 6
    )
    out = analyze_code(file_path=str(f))
    assert "DISABLED" not in out, "an indented `#  if 0` still disables its block"
    assert "LIVE" in out, "a hex version comparison is not a disabled block"


def test_elixir_wrapped_declaration_heads_of_every_shape(tmp_path: Path) -> None:
    # `mix format` wraps long heads in several ways: a `when` guard, a map
    # pattern closing with `}) do`, a list closing with `]) do`. Each puts
    # the `do` on a later line than the declaration, and counting it as a
    # second block handed the function's `end` away.
    f = tmp_path / "w.ex"
    f.write_text(
        "defmodule My.Wrap do\n"        # 1
        "  def render(%{\n"             # 2
        "        title: title\n"        # 3
        "      }) do\n"                 # 4
        "    title\n"                   # 5
        "  end\n"                       # 6
        "\n"                            # 7
        "  def pick([\n"                # 8
        "        head\n"                # 9
        "      ]) do\n"                 # 10
        "    head\n"                    # 11
        "  end\n"                       # 12
        "\n"                            # 13
        "  def last(x)\n"               # 14
        "      when is_map(x) do\n"     # 15
        "    x\n"                       # 16
        "  end\n"                       # 17
        "end\n"                         # 18
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2-6: render()" in got
    assert "  - 8-12: pick()" in got
    assert "  - 14-17: last(x)" in got
    assert "  - 1-18: My.Wrap" in got


def test_elixir_multiline_comprehension_do_colon_takes_no_block(tmp_path: Path) -> None:
    # A formatted comprehension puts `do:` on its own line, several lines
    # below the `for` that opened the head.
    f = tmp_path / "c.ex"
    f.write_text(
        "defmodule My.Comp do\n"          # 1
        "  def index(params) do\n"        # 2
        "    for {k, v} <- params,\n"     # 3
        "        is_binary(v),\n"         # 4
        "        into: %{},\n"            # 5
        "        do: {k, v}\n"            # 6
        "  end\n"                         # 7
        "\n"                              # 8
        "  def show(conn) do\n"           # 9
        "    conn\n"                      # 10
        "  end\n"                         # 11
        "end\n"                           # 12
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2-7: index(params)" in got
    assert "  - 9-11: show(conn)" in got


def test_elixir_pipeline_anonymous_function_still_opens_a_block(tmp_path: Path) -> None:
    # The guard against over-correcting: a `|>` line ending in `fn x ->` is
    # a REAL anonymous block closed by `end)`, not a wrapped head.
    f = tmp_path / "p.ex"
    f.write_text(
        "defmodule My.Pipe do\n"              # 1
        "  def run(list) do\n"                # 2
        "    list\n"                          # 3
        "    |> Enum.map(fn x ->\n"           # 4
        "      x * 2\n"                       # 5
        "    end)\n"                          # 6
        "  end\n"                             # 7
        "\n"                                  # 8
        "  def after_pipe(y) do\n"            # 9
        "    y\n"                             # 10
        "  end\n"                             # 11
        "end\n"                               # 12
    )
    got = _lines(analyze_code(file_path=str(f)))
    assert "  - 2-7: run(list)" in got
    assert "  - 9-11: after_pipe(y)" in got


def test_apache_conf_is_not_claimed_as_xml(tmp_path: Path) -> None:
    # Apache container directives (`<Directory …>`) look like XML tags. A
    # sniff that accepted any leading tag labelled the file `xml` and then
    # listed two "elements" while missing every real directive.
    f = tmp_path / "httpd.conf"
    f.write_text(
        'ServerRoot "/usr/local"\n'
        "Listen 80\n"
        "\n"
        '<Directory "/var/www">\n'
        "    AllowOverride None\n"
        "</Directory>\n"
    )
    out = analyze_code(file_path=str(f))
    assert "language: xml" not in out
    assert "language: ini" not in out
    assert not out.startswith("Error")


def test_conditional_extension_claims_are_disclosed_in_the_notes() -> None:
    # A spec that claims an extension only on a content sniff is making a
    # judgement the reader cannot see. The `notes:` line is where the tool
    # tells the model what it decided, so the condition has to be stated
    # there or the outline is quietly guessing.
    for spec in ca._SPECS:
        for ext in spec.sniff_extensions:
            assert ext in spec.notes, f"{spec.name}: notes must disclose its conditional claim on {ext}"
