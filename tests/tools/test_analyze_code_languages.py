"""analyze_code multi-language engine + never-refuse fallback.

Operator incident 2026-07-22: analyze_code refused `main.rs` ("Unsupported
code language") — a navigation tool that refuses makes the agent re-read
whole files raw. These pins hold the two design rules:

- 20+ languages outline through the declarative LanguageSpec engine
  (adding a language is adding DATA, never code);
- unknown-but-readable text NEVER errors: it gets a labeled generic outline;
  only binary content is an error.
"""

from __future__ import annotations

from pathlib import Path

from abstractcore.tools import code_analysis as ca
from abstractcore.tools.common_tools import analyze_code


def test_rust_incident_file_outlines_instead_of_refusing(tmp_path: Path) -> None:
    # The exact incident shape: a Rust source passed with language="rust".
    f = tmp_path / "main.rs"
    f.write_text(
        "use std::collections::HashMap;\n"
        "pub struct App { items: HashMap<String, u32> }\n"
        "impl App {\n"
        "    pub fn new() -> Self { Self { items: HashMap::new() } }\n"
        "}\n"
        "fn main() {\n"
        "    let _ = App::new();\n"
        "}\n"
    )
    out = analyze_code(file_path=str(f), language="rust")
    assert "Unsupported code language" not in out
    assert "language=rust" in out or "language: rust" in out
    assert "main()" in out
    assert "App" in out
    assert "std::collections::HashMap" in out


def test_rust_detected_by_extension_without_language_arg(tmp_path: Path) -> None:
    f = tmp_path / "lib.rs"
    f.write_text("pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
    out = analyze_code(file_path=str(f))
    assert "language: rust" in out
    assert "add(a: i32, b: i32)" in out


def test_rust_bodyless_trait_method_takes_no_block_extent(tmp_path: Path) -> None:
    # `fn draw(&self);` has no body — a forward brace scan would steal the
    # NEXT block's extent and misdirect read_file ranges.
    f = tmp_path / "t.rs"
    f.write_text("pub trait R {\n    fn draw(&self);\n}\nfn main() {\n    ()\n}\n")
    out = analyze_code(file_path=str(f))
    assert "  - 2: draw(&self)" in out
    assert "4-6: main()" in out


def test_go_receiver_methods_and_grouped_imports(tmp_path: Path) -> None:
    f = tmp_path / "svc.go"
    f.write_text(
        'package main\n\nimport (\n    "fmt"\n    "net/http"\n)\n\n'
        "type Server struct {\n    port int\n}\n\n"
        "func (s *Server) Start() error {\n    return nil\n}\n"
    )
    out = analyze_code(file_path=str(f))
    assert "language: go" in out
    assert "fmt" in out and "net/http" in out
    assert "(s *Server) Start()" in out
    assert "Server" in out


def test_java_class_and_methods(tmp_path: Path) -> None:
    f = tmp_path / "App.java"
    f.write_text(
        "import java.util.List;\n\n"
        "public class App {\n"
        "    private int count;\n"
        "    public static void main(String[] args) {\n"
        "        System.out.println(1);\n"
        "    }\n"
        "    private List<String> names(int limit) {\n"
        "        return null;\n"
        "    }\n"
        "}\n"
    )
    out = analyze_code(file_path=str(f))
    assert "language: java" in out
    assert "java.util.List" in out
    assert "main(String[] args)" in out
    assert "names(int limit)" in out


def test_c_definitions_not_prototypes(tmp_path: Path) -> None:
    f = tmp_path / "x.c"
    f.write_text(
        "#include <stdio.h>\n"
        "#define MAX 10\n"
        "int helper(int a);\n"  # prototype: must NOT be listed as a function
        "int helper(int a) {\n"
        "    return a * 2;\n"
        "}\n"
    )
    out = analyze_code(file_path=str(f))
    assert "language: c" in out
    assert "stdio.h" in out
    assert "MAX" in out
    functions_block = out.split("functions:")[1]
    assert "4-6: helper" in functions_block
    assert "  - 3:" not in functions_block, "prototypes must not appear as definitions"


def test_ruby_end_blocks(tmp_path: Path) -> None:
    f = tmp_path / "app.rb"
    f.write_text(
        "require 'json'\n\n"
        "class Greeter\n"
        "  def hello(name)\n"
        "    puts name\n"
        "  end\n"
        "end\n"
    )
    out = analyze_code(file_path=str(f))
    assert "language: ruby" in out
    assert "json" in out
    assert "3-7: Greeter" in out
    assert "4-6: hello(name)" in out


def test_shell_functions_and_shebang_detection(tmp_path: Path) -> None:
    f = tmp_path / "deploy"  # no extension: shebang must route it
    f.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\nMAX_RETRIES=3\n\nmain() {\n    echo hi\n}\n")
    out = analyze_code(file_path=str(f))
    assert "language: shell" in out
    assert "main()" in out
    assert "MAX_RETRIES" in out


def test_python_shebang_routes_to_deep_python_lane(tmp_path: Path) -> None:
    f = tmp_path / "tool"  # no extension
    f.write_text("#!/usr/bin/env python3\nimport os\n\ndef run():\n    return os.getcwd()\n")
    out = analyze_code(file_path=str(f))
    assert "language: python" in out
    assert "def run" in out or "run()" in out


def test_markdown_headings(tmp_path: Path) -> None:
    f = tmp_path / "README.md"
    f.write_text("# Title\n\ntext\n\n## Install\n\n### Notes\n")
    out = analyze_code(file_path=str(f))
    assert "language: markdown" in out
    assert "# Title" in out and "## Install" in out


def test_yaml_top_level_keys_only(tmp_path: Path) -> None:
    f = tmp_path / "config.yaml"
    f.write_text("server:\n  port: 8080\nlogging:\n  level: info\n")
    out = analyze_code(file_path=str(f))
    assert "language: yaml" in out
    keys_block = out.split("keys:")[1]
    assert "server" in keys_block and "logging" in keys_block
    assert "port" not in keys_block, "nested keys must not appear (top-level anchors only)"


def test_json_object_and_invalid(tmp_path: Path) -> None:
    good = tmp_path / "a.json"
    good.write_text('{"name": "x", "version": 1}')
    out = analyze_code(file_path=str(good))
    assert "parse=ok" in out and "name" in out and "version" in out

    bad = tmp_path / "b.json"
    bad.write_text('{"name": ')
    out_bad = analyze_code(file_path=str(bad))
    assert "parse=error" in out_bad


def test_sql_ddl_statements(tmp_path: Path) -> None:
    f = tmp_path / "schema.sql"
    f.write_text("CREATE TABLE users (id INT);\nALTER TABLE users ADD COLUMN name TEXT;\ncreate index idx_users on users(id);\n")
    out = analyze_code(file_path=str(f))
    assert "language: sql" in out
    assert "users" in out
    assert "idx_users" in out


def test_unknown_language_falls_back_generic_never_refuses(tmp_path: Path) -> None:
    f = tmp_path / "program.zig"
    f.write_text('const std = @import("std");\n\npub fn main() void {\n    // TODO: wire up\n}\n')
    out = analyze_code(file_path=str(f))
    assert not out.startswith("Error"), "readable text must never be refused"
    assert "GENERIC" in out or "generic" in out
    assert "TODO" in out


def test_unknown_explicit_language_labels_fallback(tmp_path: Path) -> None:
    f = tmp_path / "x.qqq"
    f.write_text("hello structural\n  indented\n")
    out = analyze_code(file_path=str(f), language="brainfuck")
    assert not out.startswith("Error")
    assert "#FALLBACK" in out and "brainfuck" in out


def test_binary_is_still_an_error(tmp_path: Path) -> None:
    f = tmp_path / "blob.rs"  # rust extension but binary content
    f.write_bytes(b"\x00\x01\x02\xff" * 64)
    out = analyze_code(file_path=str(f))
    assert "binary" in out.lower()
    assert out.startswith("Error")


def test_minified_file_says_so_instead_of_scanning(tmp_path: Path) -> None:
    f = tmp_path / "bundle.rs"
    f.write_text("fn a() {}" + " " * 6000 + "fn b() {}")
    out = analyze_code(file_path=str(f))
    assert "minified" in out or "generated" in out


def test_section_caps_are_labeled(tmp_path: Path) -> None:
    body = "\n".join(f"fn f{i}() {{}}" for i in range(80))
    f = tmp_path / "many.rs"
    f.write_text(body + "\n")
    out = analyze_code(file_path=str(f))
    assert "#TRUNCATION" in out
    assert f"({80 - ca.MAX_SECTION_ENTRIES} more)" in out


def test_legacy_lanes_unchanged(tmp_path: Path) -> None:
    # Python/JS/HTML/R keep their deep analyzers.
    py = tmp_path / "m.py"
    py.write_text("import os\n\nclass A:\n    def go(self):\n        return os.name\n")
    out = analyze_code(file_path=str(py))
    assert "language: python" in out and "classes:" in out

    js = tmp_path / "m.js"
    js.write_text("import x from './x';\nfunction go() { return 1; }\n")
    out_js = analyze_code(file_path=str(js))
    assert "language: javascript" in out_js


def test_every_spec_kind_is_emittable() -> None:
    # A DeclPattern.kind absent from the emit order silently drops its whole
    # section from the outline (truth bug class) — pin the table against the
    # render whitelist.
    for spec in ca._SPECS:
        for decl in spec.decl_patterns:
            assert decl.kind in ca._EMIT_KIND_ORDER, f"{spec.name} kind {decl.kind!r} missing from _EMIT_KIND_ORDER"


def test_long_line_jsonl_keeps_its_parse_lane(tmp_path: Path) -> None:
    # Reviewer B P1-1: json.loads is not line-anchored — a JSONL with a
    # >5000-char record must get validity + record counts, not the
    # minified-guard skip.
    f = tmp_path / "ledger.jsonl"
    big = '{"k": "' + "x" * 6000 + '"}'
    f.write_text(big + "\n" + '{"a": 1}' + "\n" + "not json\n")
    out = analyze_code(file_path=str(f))
    assert "jsonl_records=3" in out
    assert "invalid=1" in out
    assert "minified" not in out


def test_plaintext_alias_means_generic_without_fallback_label(tmp_path: Path) -> None:
    f = tmp_path / "notes.dat"
    f.write_text("meeting notes\n  - item\n")
    out = analyze_code(file_path=str(f), language="text")
    assert not out.startswith("Error")
    assert "#FALLBACK" not in out, "asking for text and getting the generic outline is not a fallback"


def test_unknown_hint_with_known_extension_honors_the_file(tmp_path: Path) -> None:
    f = tmp_path / "main.rs"
    f.write_text("fn main() {\n    ()\n}\n")
    out = analyze_code(file_path=str(f), language="rust-lang")
    assert "language: rust" in out
    assert "#FALLBACK" in out and "rust-lang" in out
    assert "main()" in out


def test_dockerfile_and_makefile_by_filename(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("FROM python:3.12 AS base\nRUN pip install x\nCMD [\"python\"]\n")
    out = analyze_code(file_path=str(df))
    assert "language: dockerfile" in out
    assert "python:3.12" in out and "base" in out

    mk = tmp_path / "Makefile"
    mk.write_text("CC = gcc\n\nbuild: deps\n\tgcc -o app main.c\n\ntest:\n\tpytest\n")
    out_mk = analyze_code(file_path=str(mk))
    assert "language: makefile" in out_mk
    assert "build" in out_mk and "test" in out_mk and "CC" in out_mk


def test_terraform_blocks(tmp_path: Path) -> None:
    f = tmp_path / "main.tf"
    f.write_text(
        'resource "aws_s3_bucket" "data" {\n  bucket = "x"\n}\n\n'
        'variable "region" {\n  default = "eu-west-1"\n}\n'
    )
    out = analyze_code(file_path=str(f))
    assert "language: terraform" in out
    assert 'resource "aws_s3_bucket" "data"' in out
    assert "1-3" in out


def test_markdown_heading_extents(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nintro\n\n## Install\n\nsteps\nmore\n\n## Usage\n\nrun it\n")
    out = analyze_code(file_path=str(f))
    # Install section spans from its heading to the line before ## Usage.
    assert "5-9: ## Install" in out


def test_rust_impl_rendered_distinct_from_struct(tmp_path: Path) -> None:
    f = tmp_path / "w.rs"
    f.write_text("pub struct Widget {\n    x: u32,\n}\n\nimpl Widget {\n    fn go(&self) {}\n}\n")
    out = analyze_code(file_path=str(f))
    types_block = out.split("types:")[1]
    assert "1-3: Widget" in types_block
    assert "impl Widget" in types_block, "impl blocks must be distinguishable from the struct definition"


# --- Reviewer A fidelity folds (wrong-edit-range class) ---


def test_brace_in_string_literal_does_not_end_extent(tmp_path: Path) -> None:
    # A F1: `String::from("}")` used to close the block early — an
    # edit_file() guided by that range replaces half a function silently.
    f = tmp_path / "f.rs"
    f.write_text('fn render() {\n    let s = String::from("}");\n    s\n}\n')
    out = analyze_code(file_path=str(f))
    assert "1-4: render()" in out


def test_brace_in_block_comment_neither_ends_extent_nor_lints(tmp_path: Path) -> None:
    # A F1: `/* } */` produced a wrong extent AND a false unbalanced lint.
    f = tmp_path / "f.c"
    f.write_text("int f(void) {\n    /* close later } */\n    return 1;\n}\n")
    out = analyze_code(file_path=str(f))
    assert "1-4: f" in out
    assert "unbalanced" not in out


def test_braceless_declaration_never_steals_next_block(tmp_path: Path) -> None:
    # A F2: kotlin `data class X(...)` scanned forward and adopted the next
    # class's brace as its own block.
    f = tmp_path / "m.kt"
    f.write_text("data class User(val name: String)\n\nclass Repo {\n    fun all() {}\n}\n")
    out = analyze_code(file_path=str(f))
    assert "  - 1: User" in out
    assert "3-5: Repo" in out


def test_ruby_one_liner_def_self_closes(tmp_path: Path) -> None:
    # A F3: `def name; @name; end` swallowed the next method.
    f = tmp_path / "a.rb"
    f.write_text("class A\n  def name; @name; end\n  def other\n    1\n  end\nend\n")
    out = analyze_code(file_path=str(f))
    assert "  - 2: name" in out
    assert "3-5: other" in out


def test_ruby_heredoc_body_never_closes_blocks(tmp_path: Path) -> None:
    # A F3: a heredoc line starting with the word "end" closed the def early.
    f = tmp_path / "m.rb"
    f.write_text(
        "class Mailer\n  def body\n    text = <<~TEXT\n      end of story\n    TEXT\n    text\n  end\nend\n"
    )
    out = analyze_code(file_path=str(f))
    assert "2-7: body" in out
    assert "1-8: Mailer" in out


def test_c_allman_brace_and_constructor_shapes(tmp_path: Path) -> None:
    # A F4: Allman style (brace on next line) and C++ init-list constructors
    # were whole missing categories.
    c = tmp_path / "a.c"
    c.write_text("int allman_main(void)\n{\n    return 0;\n}\n")
    assert "1-4: allman_main" in analyze_code(file_path=str(c))

    cpp = tmp_path / "w.cpp"
    cpp.write_text("class Widget {\npublic:\n    Widget(int w) : width_(w) {\n        init();\n    }\n};\n")
    assert "3-5: Widget(int w)" in analyze_code(file_path=str(cpp))


def test_shell_heredoc_body_not_outlined_and_case_arm_not_linted(tmp_path: Path) -> None:
    # A P2-11: heredoc data was outlined as live functions; `a)` case arms
    # produced a false unbalanced-parens lint.
    f = tmp_path / "d.sh"
    f.write_text(
        "#!/bin/bash\ncat <<EOF\ninner_fn() {\nEOF\ncase $1 in\n  a) echo a;;\nesac\nreal() {\n  echo r\n}\n"
    )
    out = analyze_code(file_path=str(f))
    assert "inner_fn" not in out
    assert "unbalanced" not in out
    assert "real()" in out


def test_c_if0_disabled_code_not_outlined(tmp_path: Path) -> None:
    f = tmp_path / "d.c"
    f.write_text("#if 0\nint dead_fn(void) {\n    return 1;\n}\n#endif\nint live(void) {\n    return 2;\n}\n")
    out = analyze_code(file_path=str(f))
    # Scope the negative assert past the header: pytest's tmp_path embeds the
    # TEST NAME in the display path, so a whole-output substring check can
    # collide with the path itself.
    body = out.split("functions:")[1]
    assert "dead_fn" not in body
    assert "6-8: live" in body


def test_markdown_fenced_comment_not_a_heading(tmp_path: Path) -> None:
    f = tmp_path / "d.md"
    f.write_text("# Real\n\n```bash\n# not a heading\n```\n\n## Also real\n")
    out = analyze_code(file_path=str(f))
    assert "not a heading" not in out
    assert "## Also real" in out


# --- Reviewer C performance/encoding folds ---


def test_c_comment_banner_never_hangs(tmp_path: Path) -> None:
    # C P0-1: `(?:[\w*]+[\s*]+)+` backtracked exponentially on `* * * *`
    # banner rows — an ordinary 8-line C file hung the tool >15s.
    import time

    f = tmp_path / "banner.c"
    f.write_text("/*\n * " + "* " * 60 + "\n */\nint main(void) {\n    return 0;\n}\n")
    t0 = time.perf_counter()
    out = analyze_code(file_path=str(f))
    assert time.perf_counter() - t0 < 5.0, "banner line must not trigger regex backtracking blowup"
    assert "main" in out


def test_many_unclosed_braces_stay_linear(tmp_path: Path) -> None:
    # C P0-2: per-decl forward scans were quadratic when braces never close
    # (4000 unclosed decls measured 91s); the one-pass index is O(total).
    import time

    f = tmp_path / "unclosed.rs"
    f.write_text("\n".join(f"fn f_{i}() {{" for i in range(3000)) + "\n")
    t0 = time.perf_counter()
    analyze_code(file_path=str(f))
    assert time.perf_counter() - t0 < 20.0

    rb = tmp_path / "many.rb"
    rb.write_text("\n".join(f"def m_{i}" for i in range(2000)) + "\n")
    t0 = time.perf_counter()
    analyze_code(file_path=str(rb))
    assert time.perf_counter() - t0 < 20.0


def test_utf16_bom_file_is_text_not_binary(tmp_path: Path) -> None:
    # C P2: Windows toolchains emit UTF-16 — "appears to be binary" was a
    # false claim.
    f = tmp_path / "wide.rs"
    f.write_bytes(b"\xff\xfe" + "fn wide() {}\n".encode("utf-16-le"))
    out = analyze_code(file_path=str(f))
    assert not out.startswith("Error")
    assert "wide()" in out
    assert "utf-16-le" in out, "the non-default decoding must be labeled"


def test_utf8_bom_does_not_hide_first_line(tmp_path: Path) -> None:
    f = tmp_path / "bom.go"
    f.write_bytes("\ufeff".encode("utf-8") + b'package main\nimport "fmt"\nfunc main() {\n}\n')
    out = analyze_code(file_path=str(f))
    assert "fmt" in out


def test_form_feed_line_numbers_match_read_file(tmp_path: Path) -> None:
    # C P2 truth bug: str.splitlines() splits on \f, drifting the reported
    # line numbers off the ones read_file(start_line) uses.
    f = tmp_path / "ff.c"
    f.write_text("int a(void) {\n    return 1;\n}\n\x0c\nint b(void) {\n    return 2;\n}\n")
    out = analyze_code(file_path=str(f))
    assert "5-7: b" in out


def test_rust_lifetime_with_line_comment(tmp_path: Path) -> None:
    # C adjacent: odd apostrophe counts (lifetimes) must not poison quote
    # tracking (rust drops `'` from its quote set).
    f = tmp_path / "lt.rs"
    f.write_text("fn f<'a>(x: &'a str) -> &'a str { // note\n    x\n}\n")
    out = analyze_code(file_path=str(f))
    assert "1-3: f" in out
    assert "unbalanced" not in out


def test_engine_spec_lookup_shapes() -> None:
    assert ca.spec_for("rust").name == "rust"
    assert ca.spec_for("c++").name == "cpp"
    assert ca.spec_for(None, Path("x/main.go")).name == "go"
    assert ca.spec_for(None, Path("Rakefile")).name == "ruby"
    assert ca.spec_for(None, Path("noext"), "#!/usr/bin/env bash") .name == "shell"
    assert ca.spec_for(None, Path("noext"), "#!/usr/bin/env fish") is None, "sh must not substring-match fish"
    assert ca.spec_for("qqq") is None
