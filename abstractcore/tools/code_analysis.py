"""Multi-language outline engine for the `analyze_code` tool.

Why this module exists (operator incident 2026-07-22): `analyze_code` refused
`main.rs` with "Unsupported code language" — a navigation tool that refuses is
worse than useless, because the agent then re-reads whole files raw (token-
expensive) or gives up. Two design rules fix the class, not the instance:

1. DECLARATIVE LANGUAGE SPECS — most brace-family languages share structure
   (imports are line-anchored patterns; declarations are line-anchored
   patterns with a kind label; block extent is brace matching). One generic
   engine + one small `LanguageSpec` per language means adding a language is
   adding DATA, never code. Keyword-block (`end`-delimited: ruby/lua/shell)
   and header-outline (markdown/yaml/toml/json/css/sql) families ride the
   same spec with a different extent strategy.

2. NEVER REFUSE READABLE TEXT — an unknown language degrades to an honest
   GENERIC outline (metrics + top-level structure sample + TODO markers),
   labeled as such. Only binary content is an error.

The four legacy lanes (python/javascript/html/r) keep their deeper bespoke
analyzers in common_tools.py (python AST + ruff, JS import resolution, HTML
ids, R sources) — this engine covers everything else and the fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

# Output bounds: outlines are prompt currency — cap list sections so a huge
# file cannot flood the model context (labels are honest about elision).
MAX_SECTION_ENTRIES = 50
# Read bound: analysis is line-oriented; a multi-MB artifact (bundle, log,
# generated code) is truncated with a label rather than freezing the tool.
MAX_ANALYZE_BYTES = 4 * 1024 * 1024
# A single enormous line (minified bundle) defeats line-anchored outlining;
# detect and say so instead of burning CPU on regexes over megabyte lines.
MAX_LINE_CHARS_FOR_OUTLINE = 5000

# Next-step guidance rendered at the top of EVERY outline (this module's
# engine lanes AND common_tools.py's deep lanes — one constant, no drift).
# It must teach edit_file's REAL contract: the default mode wants a short
# UNIQUE pattern with NO line params; start_line/end_line are 1-based scope
# limiters for disambiguating repeated matches (or bounding a range replace
# with pattern=""), not a required argument. And it must teach staleness:
# a live trace (2026-07-26) showed a model copying outline line numbers into
# edit_file wholesale — 0-based and stale after the first edit, burning turns.
ANALYZE_CODE_NEXT_STEP_HINT = (
    "Next step: read_file(start_line/end_line) around the target block, then edit_file with a short UNIQUE pattern "
    "and no line params (start_line/end_line are 1-based and only needed to disambiguate repeated matches, "
    "or with pattern=\"\" for a range replace). Line numbers go stale after every edit — re-run analyze_code or re-read before reusing them."
)


@dataclass
class DeclPattern:
    """One line-anchored declaration matcher.

    `kind` groups the match in the output ("functions", "types", ...);
    `pattern` must expose a named group `name`; optional `params` group is
    appended to the rendered signature. `requires_brace` demands a `{` on the
    match line OR as the first token of the next non-blank line (Allman
    style) — used by the C-family function patterns so prototypes and macro
    invocations never read as definitions (reviewer A, F4).
    """

    kind: str
    pattern: Pattern[str]
    block: bool = True  # whether a block extent should be computed
    requires_brace: bool = False


@dataclass
class LanguageSpec:
    """Declarative description of how to outline one language."""

    name: str
    aliases: Tuple[str, ...] = ()
    extensions: Tuple[str, ...] = ()
    filenames: Tuple[str, ...] = ()  # exact basenames (Makefile, Dockerfile)
    shebangs: Tuple[str, ...] = ()  # substrings matched against a #! first line
    line_comment: Tuple[str, ...] = ("//",)
    import_patterns: Tuple[Pattern[str], ...] = ()
    decl_patterns: Tuple[DeclPattern, ...] = ()
    # Block extent strategy: "brace" ({...}), "end" (def..end), "heading"
    # (markdown sections), or "none" (single-line declarations only).
    block_style: str = "brace"
    # Which characters open string literals for the brace/balance scanners.
    # Rust drops `'` (lifetimes make apostrophes non-string more often than
    # not; char literals rarely carry braces) — reviewer C.
    quote_chars: str = "'\"`"
    # Words that OPEN an end-delimited block (for block_style="end").
    end_block_openers: Tuple[str, ...] = ()
    # Whether the language has C-style /* */ block comments (drives the
    # string/comment-aware brace scanner; reviewer A, F1).
    c_block_comments: bool = True
    # Which bracket pairs the whole-file balance lint counts. Shell drops
    # parens (case arms `a)` are legal unbalanced parens; reviewer A, P2-11).
    balance_pairs: str = "{}()[]"
    # Cross-line regions the line scanner must skip: "heredoc" (shell/ruby
    # << tags), "fence" (markdown ``` blocks), "preproc_if0" (C-family
    # `#if 0` disabled code). Reviewer A F3/P2-5/P2-11/P2-12.
    skip_regions: Tuple[str, ...] = ()
    notes: str = ""


def _rx(p: str) -> Pattern[str]:
    return re.compile(p)


# ---------------------------------------------------------------------------
# Language table. Adding a language = adding an entry here.
# Patterns are LINE-ANCHORED and conservative: a miss costs an outline entry,
# a false positive costs trust — prefer misses.
# ---------------------------------------------------------------------------

_SPECS: Tuple[LanguageSpec, ...] = (
    LanguageSpec(
        name="rust",
        aliases=("rs",),
        extensions=(".rs",),
        line_comment=("//",),
        import_patterns=(_rx(r"^\s*(?:pub\s+)?use\s+(?P<name>[^;]+?)\s*;"), _rx(r"^\s*extern\s+crate\s+(?P<name>\w+)")),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+(?P<name>\w+)\s*(?:<[^>]*>)?\s*\((?P<params>[^)]*)?")),
            DeclPattern("types", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+(?P<name>\w+)")),
            DeclPattern("types", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+(?P<name>\w+)")),
            DeclPattern("types", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+(?P<name>\w+)")),
            # impl blocks render with the `impl` keyword kept in the name so
            # "types: App (struct)" and "types: impl App (methods)" are
            # distinguishable — an agent adding a FIELD must read the struct,
            # not the impl (reviewer B, P2-2).
            DeclPattern("types", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?P<name>impl(?:\s*<[^>]*>)?\s+[\w:<>, ]+?)\s*(?:\{|$)")),
            DeclPattern("modules", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(?P<name>\w+)")),
            DeclPattern("constants", _rx(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(?P<name>\w+)\s*:"), block=False),
            DeclPattern("macros", _rx(r"^\s*macro_rules!\s+(?P<name>\w+)")),
        ),
        quote_chars='"',
        notes="Rust outline is heuristic (line-anchored declarations, brace extents), not a full parse.",
    ),
    LanguageSpec(
        name="go",
        aliases=("golang",),
        extensions=(".go",),
        import_patterns=(_rx(r"^\s*import\s+(?:\w+\s+)?\"(?P<name>[^\"]+)\""), _rx(r"^\s*\"(?P<name>[\w./-]+)\"\s*$")),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*func\s+(?:\((?P<recv>[^)]*)\)\s+)?(?P<name>\w+)\s*(?:\[[^\]]*\])?\((?P<params>[^)]*)?")),
            DeclPattern("types", _rx(r"^\s*type\s+(?P<name>\w+)\s+(?:struct|interface)\b")),
            DeclPattern("types", _rx(r"^\s*type\s+(?P<name>\w+)\s+"), block=False),
            DeclPattern("constants", _rx(r"^\s*(?:const|var)\s+(?P<name>\w+)\b"), block=False),
        ),
        notes="Go outline is heuristic; grouped import blocks list one path per line.",
    ),
    LanguageSpec(
        name="java",
        extensions=(".java",),
        import_patterns=(_rx(r"^\s*import\s+(?:static\s+)?(?P<name>[\w.*]+)\s*;"),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+|static\s+|sealed\s+)*(?:class|interface|enum|record)\s+(?P<name>\w+)")),
            # Same-line annotations allowed (A P2-9); closing paren optional
            # so multi-line signatures still list at their start (A P2-10).
            DeclPattern("functions", _rx(r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?=((?:public\s+|private\s+|protected\s+|abstract\s+|final\s+|static\s+|synchronized\s+|native\s+|default\s+)+))\1(?=([\w<>\[\], ?.]+\s+))\2(?P<name>\w+)\s*\((?P<params>[^)]*)?")),
            # Constructors: modifier + ClassName( with no return type (A F4).
            DeclPattern("functions", _rx(r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:public|private|protected)\s+(?P<name>[A-Z]\w*)\s*\((?P<params>[^)]*)?")),
        ),
        notes="Java outline is heuristic; methods require a visibility/modifier keyword to reduce false positives.",
    ),
    LanguageSpec(
        name="c",
        extensions=(".c", ".h"),
        import_patterns=(_rx(r"^\s*#\s*include\s+[<\"](?P<name>[^>\"]+)[>\"]"),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*typedef\s+(?:struct|enum|union)\s*(?P<name>\w*)")),
            # `(?:\{|:|$)` after the name: a LOCAL variable of struct type
            # (`struct sockaddr_in addr;`) is not a type declaration (A P2-8).
            DeclPattern("types", _rx(r"^\s*(?:struct|enum|union)\s+(?P<name>\w+)\s*(?:\{|$)")),
            # Greedy params to the LAST `)` so function-pointer parameters
            # survive (A P2-7); requires_brace covers Allman style where the
            # `{` opens on the next line (A F4).
            # Disjoint repeated classes (reviewer C, P0-1: `[\w*]+[\s*]+`
            # backtracked exponentially on `* * * *` comment-banner rows —
            # a plain 8-line C file hung the tool).
            DeclPattern("functions", _rx(r"^(?!\s*(?:if|for|while|switch|return|else|do|sizeof)\b)\s*(?:[\w*]+\s+)+\**(?P<name>\w+)\s*\((?P<params>.*)\)\s*(?:\{|$)"), requires_brace=True),
            DeclPattern("macros", _rx(r"^\s*#\s*define\s+(?P<name>\w+)"), block=False),
        ),
        skip_regions=("preproc_if0",),
        notes="C outline is heuristic: function definitions need a brace on the same or next line (prototypes excluded); `#if 0` blocks are skipped.",
    ),
    LanguageSpec(
        name="cpp",
        aliases=("c++", "cxx"),
        extensions=(".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"),
        import_patterns=(_rx(r"^\s*#\s*include\s+[<\"](?P<name>[^>\"]+)[>\"]"),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct|enum(?:\s+class)?|union)\s+(?P<name>\w+)\s*(?:\{|:|;|$|final)")),
            DeclPattern("namespaces", _rx(r"^\s*namespace\s+(?P<name>[\w:]+)")),
            # catch/try excluded (A P2-6); optional `: init_list` trailer so
            # constructors with member-init lists match (A F4); greedy params
            # for fn-pointer args; requires_brace covers Allman (A F4).
            # Atomic-group emulation `(?=(...))\1` pins the prefix so star
            # runs cannot backtrack polynomially (reviewer C, P1-1; native
            # (?>...) needs Python 3.11, repo floor is lower).
            DeclPattern("functions", _rx(r"^(?!\s*(?:if|for|while|switch|return|else|do|sizeof|new|delete|catch|try)\b)\s*(?=((?:[\w:&<>,*~ ]+[\s*&]+)?))\1(?P<name>[\w:~]+)\s*\((?P<params>.*)\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?::[^{;]*)?(?:\{|$)"), requires_brace=True),
            DeclPattern("macros", _rx(r"^\s*#\s*define\s+(?P<name>\w+)"), block=False),
        ),
        skip_regions=("preproc_if0",),
        notes="C++ outline is heuristic: function definitions need a brace on the same or next line (prototypes excluded); `#if 0` blocks are skipped.",
    ),
    LanguageSpec(
        name="csharp",
        aliases=("c#", "cs"),
        extensions=(".cs",),
        import_patterns=(_rx(r"^\s*(?:global\s+)?using\s+(?:static\s+)?(?P<name>[\w.=\s]+?)\s*;"),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+|abstract\s+|sealed\s+|static\s+|partial\s+)*(?:class|interface|enum|record|struct)\s+(?P<name>\w+)")),
            DeclPattern("namespaces", _rx(r"^\s*namespace\s+(?P<name>[\w.]+)")),
            DeclPattern("functions", _rx(r"^\s*(?:\[[^\]]*\]\s*)*(?=((?:public\s+|private\s+|protected\s+|internal\s+|static\s+|virtual\s+|override\s+|async\s+|abstract\s+|sealed\s+|new\s+|extern\s+)+))\1(?=([\w<>\[\], ?.]+\s+))\2(?P<name>\w+)\s*(?:<[^>]*>)?\s*\((?P<params>[^)]*)?")),
            # Constructors: modifier + ClassName( with no return type (A F4).
            DeclPattern("functions", _rx(r"^\s*(?:public|private|protected|internal)\s+(?P<name>[A-Z]\w*)\s*\((?P<params>[^)]*)?")),
        ),
        notes="C# outline is heuristic; methods require a modifier keyword to reduce false positives.",
    ),
    LanguageSpec(
        name="swift",
        extensions=(".swift",),
        import_patterns=(_rx(r"^\s*import\s+(?P<name>[\w.]+)"),),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*(?:public\s+|private\s+|internal\s+|open\s+|fileprivate\s+|static\s+|class\s+|override\s+|final\s+|mutating\s+)*func\s+(?P<name>[\w`]+)\s*(?:<[^>]*>)?\s*\((?P<params>[^)]*)?")),
            DeclPattern("types", _rx(r"^\s*(?:public\s+|private\s+|internal\s+|open\s+|fileprivate\s+|final\s+|indirect\s+)*(?:class|struct|enum|protocol|actor|extension)\s+(?P<name>[\w.]+)")),
            # Initializers/deinitializers are functions too (A F4).
            DeclPattern("functions", _rx(r"^\s*(?:public\s+|private\s+|internal\s+|open\s+|fileprivate\s+|required\s+|convenience\s+|override\s+)*(?P<name>init\??|deinit)\s*(?:\((?P<params>[^)]*)?\)?)?")),
        ),
        notes="Swift outline is heuristic (line-anchored declarations, brace extents).",
    ),
    LanguageSpec(
        name="kotlin",
        aliases=("kt",),
        extensions=(".kt", ".kts"),
        import_patterns=(_rx(r"^\s*import\s+(?P<name>[\w.*]+)"),),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+|open\s+|override\s+|suspend\s+|inline\s+|operator\s+|infix\s+)*fun\s+(?:<[^>]*>\s*)?(?:[\w.<>?]+\.)?(?P<name>\w+)\s*\((?P<params>[^)]*)?")),
            DeclPattern("types", _rx(r"^\s*(?:public\s+|private\s+|internal\s+|open\s+|abstract\s+|sealed\s+|data\s+|inner\s+|annotation\s+|enum\s+)*(?:class|interface|object)\s+(?P<name>\w+)")),
        ),
        notes="Kotlin outline is heuristic (line-anchored declarations, brace extents).",
    ),
    LanguageSpec(
        name="ruby",
        aliases=("rb",),
        extensions=(".rb", ".rake"),
        filenames=("Rakefile", "Gemfile"),
        shebangs=("ruby",),
        line_comment=("#",),
        import_patterns=(_rx(r"^\s*require(?:_relative)?\s+['\"](?P<name>[^'\"]+)['\"]"),),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*def\s+(?P<name>[\w.?!=\[\]]+)\s*(?:\((?P<params>[^)]*)\))?")),
            DeclPattern("types", _rx(r"^\s*(?:class|module)\s+(?P<name>[\w:]+)")),
        ),
        block_style="end",
        end_block_openers=("def", "class", "module", "if", "unless", "case", "while", "until", "for", "begin", "do"),
        c_block_comments=False,
        skip_regions=("heredoc",),
        notes="Ruby outline is heuristic; def/end extents handle one-liners, heredocs and =begin comments; multi-line strings stay approximate.",
    ),
    LanguageSpec(
        name="php",
        extensions=(".php",),
        line_comment=("//", "#"),
        import_patterns=(_rx(r"^\s*use\s+(?P<name>[\w\\]+)"), _rx(r"^\s*(?:require|include)(?:_once)?\s*\(?\s*['\"](?P<name>[^'\"]+)['\"]")),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|abstract\s+|final\s+)*function\s+(?P<name>\w+)\s*\((?P<params>[^)]*)?")),
            DeclPattern("types", _rx(r"^\s*(?:abstract\s+|final\s+)*(?:class|interface|trait|enum)\s+(?P<name>\w+)")),
        ),
        notes="PHP outline is heuristic (line-anchored declarations, brace extents).",
    ),
    LanguageSpec(
        name="shell",
        aliases=("bash", "sh", "zsh"),
        extensions=(".sh", ".bash", ".zsh"),
        shebangs=("sh", "bash", "zsh", "ksh"),
        line_comment=("#",),
        import_patterns=(_rx(r"^\s*(?:source|\.)\s+(?P<name>[^\s;]+)"),),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*(?:function\s+)?(?P<name>[\w.-]+)\s*\(\)\s*\{?")),
            DeclPattern("constants", _rx(r"^\s*(?:export\s+|readonly\s+|declare\s+[-\w]*\s+)?(?P<name>[A-Z][A-Z0-9_]*)="), block=False),
        ),
        c_block_comments=False,
        balance_pairs="{}",
        skip_regions=("heredoc",),
        notes="Shell outline is heuristic; only `name() {` style functions and UPPER_CASE assignments are listed; heredoc bodies are skipped.",
    ),
    LanguageSpec(
        name="sql",
        extensions=(".sql",),
        line_comment=("--",),
        import_patterns=(),
        decl_patterns=(
            DeclPattern("statements", _rx(r"^\s*(?i:CREATE)\s+(?i:OR\s+REPLACE\s+)?(?i:TABLE|VIEW|INDEX|FUNCTION|PROCEDURE|TRIGGER|SCHEMA|DATABASE|TYPE|SEQUENCE|MATERIALIZED\s+VIEW)\s+(?i:IF\s+NOT\s+EXISTS\s+)?(?P<name>[\w.\"]+)"), block=False),
            DeclPattern("statements", _rx(r"^\s*(?i:ALTER)\s+(?i:TABLE|VIEW|INDEX|FUNCTION|SCHEMA|TYPE)\s+(?P<name>[\w.\"]+)"), block=False),
            DeclPattern("statements", _rx(r"^\s*(?i:DROP)\s+(?i:TABLE|VIEW|INDEX|FUNCTION|SCHEMA|TYPE)\s+(?i:IF\s+EXISTS\s+)?(?P<name>[\w.\"]+)"), block=False),
        ),
        block_style="none",
        notes="SQL outline lists DDL statements (CREATE/ALTER/DROP) with line numbers.",
    ),
    LanguageSpec(
        name="css",
        extensions=(".css", ".scss", ".less"),
        line_comment=("//",),
        import_patterns=(_rx(r"^\s*@import\s+(?:url\()?['\"]?(?P<name>[^'\")]+)"),),
        decl_patterns=(
            DeclPattern("rules", _rx(r"^\s*@(?P<name>media|keyframes|font-face|supports|layer)\b[^\{]*")),
            # Inline-closed rules (`.btn { color: red }`) count too (A P2-13).
            DeclPattern("rules", _rx(r"^(?P<name>[.#]?[\w-][^{;]{0,120}?)\s*\{")),
        ),
        notes="CSS outline lists top-level selectors and at-rules (heuristic; nested preprocessor rules list at their own lines).",
    ),
    LanguageSpec(
        name="markdown",
        aliases=("md",),
        extensions=(".md", ".markdown", ".mdx"),
        line_comment=(),
        import_patterns=(),
        decl_patterns=(DeclPattern("headings", _rx(r"^(?P<name>#{1,6}\s+.+?)\s*$"), block=False),),
        block_style="heading",
        skip_regions=("fence",),
        notes="Markdown outline lists headings; a heading's range ends where the next same-or-higher heading starts.",
    ),
    LanguageSpec(
        name="dockerfile",
        extensions=(".dockerfile",),
        filenames=("Dockerfile", "Containerfile"),
        line_comment=("#",),
        import_patterns=(_rx(r"^\s*(?i:FROM)\s+(?P<name>\S+)"),),
        decl_patterns=(
            DeclPattern("stages", _rx(r"^\s*(?i:FROM)\s+\S+\s+(?i:AS)\s+(?P<name>\S+)"), block=False),
            DeclPattern("statements", _rx(r"^\s*(?P<name>(?i:RUN|COPY|ADD|ENV|ARG|EXPOSE|ENTRYPOINT|CMD|WORKDIR|USER|VOLUME|LABEL|HEALTHCHECK))\b"), block=False),
        ),
        block_style="none",
        notes="Dockerfile outline lists base images, build stages, and instructions with line numbers.",
    ),
    LanguageSpec(
        name="makefile",
        aliases=("make",),
        extensions=(".mk",),
        filenames=("Makefile", "makefile", "GNUmakefile"),
        line_comment=("#",),
        import_patterns=(_rx(r"^\s*(?:include|-include)\s+(?P<name>\S+)"),),
        decl_patterns=(
            # Targets: name(s) at column 0 followed by ':' (not '=' — that's a var).
            DeclPattern("targets", _rx(r"^(?P<name>[^\s:=#][^:=#]*?)\s*:(?!=)"), block=False),
            DeclPattern("constants", _rx(r"^(?P<name>[A-Za-z_][\w.-]*)\s*[:?+]?="), block=False),
        ),
        block_style="none",
        notes="Makefile outline lists targets and variable assignments with line numbers.",
    ),
    LanguageSpec(
        name="terraform",
        aliases=("hcl", "tf"),
        extensions=(".tf", ".tfvars", ".hcl"),
        line_comment=("#", "//"),
        import_patterns=(),
        decl_patterns=(
            DeclPattern("blocks", _rx(r"^\s*(?P<name>(?:resource|data)\s+\"[^\"]+\"\s+\"[^\"]+\")\s*\{")),
            DeclPattern("blocks", _rx(r"^\s*(?P<name>(?:variable|output|module|provider)\s+\"[^\"]+\")\s*\{")),
            DeclPattern("blocks", _rx(r"^\s*(?P<name>terraform|locals)\s*\{")),
        ),
        notes="Terraform/HCL outline lists top-level blocks (resource/data/variable/output/module/provider) with brace extents.",
    ),
    LanguageSpec(
        name="proto",
        aliases=("protobuf",),
        extensions=(".proto",),
        import_patterns=(_rx(r"^\s*import\s+(?:public\s+)?\"(?P<name>[^\"]+)\""),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:message|enum|service)\s+(?P<name>\w+)")),
            DeclPattern("functions", _rx(r"^\s*rpc\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)")),
        ),
        notes="Protobuf outline lists messages/enums/services and rpc methods with brace extents.",
    ),
    LanguageSpec(
        name="yaml",
        aliases=("yml",),
        extensions=(".yaml", ".yml"),
        line_comment=("#",),
        import_patterns=(),
        decl_patterns=(DeclPattern("keys", _rx(r"^(?P<name>[\w.\"'/-]+)\s*:"), block=False),),
        block_style="none",
        notes="YAML outline lists TOP-LEVEL keys only (column-0 anchors) with line numbers.",
    ),
    LanguageSpec(
        name="toml",
        extensions=(".toml",),
        line_comment=("#",),
        import_patterns=(),
        decl_patterns=(DeclPattern("tables", _rx(r"^\s*(?P<name>\[\[?[^\]]+\]?\])\s*$"), block=False),),
        block_style="none",
        notes="TOML outline lists tables ([section]) with line numbers.",
    ),
    LanguageSpec(
        name="json",
        extensions=(".json", ".jsonl", ".ndjson"),
        line_comment=(),
        import_patterns=(),
        decl_patterns=(),
        block_style="none",
        notes="JSON outline reports validity and top-level keys.",
    ),
)

_SPEC_BY_NAME: Dict[str, LanguageSpec] = {}
for _spec in _SPECS:
    _SPEC_BY_NAME[_spec.name] = _spec
    for _a in _spec.aliases:
        _SPEC_BY_NAME[_a] = _spec

_EXT_TO_SPEC: Dict[str, LanguageSpec] = {}
for _spec in _SPECS:
    for _e in _spec.extensions:
        _EXT_TO_SPEC.setdefault(_e, _spec)

_FILENAME_TO_SPEC: Dict[str, LanguageSpec] = {}
for _spec in _SPECS:
    for _f in _spec.filenames:
        _FILENAME_TO_SPEC.setdefault(_f, _spec)


def known_language_names() -> List[str]:
    """Every language the ENGINE covers (the legacy lanes add their own)."""
    return sorted({s.name for s in _SPECS})


def spec_for(language: Optional[str] = None, path: Optional[Path] = None, first_line: str = "") -> Optional[LanguageSpec]:
    """Resolve a LanguageSpec from an explicit name, a path, or a shebang."""
    raw = str(language or "").strip().lower()
    if raw:
        return _SPEC_BY_NAME.get(raw)
    if path is not None:
        by_name = _FILENAME_TO_SPEC.get(path.name)
        if by_name is not None:
            return by_name
        by_ext = _EXT_TO_SPEC.get(path.suffix.lower())
        if by_ext is not None:
            return by_ext
    if first_line.startswith("#!"):
        lowered = first_line.lower()
        for s in _SPECS:
            for token in s.shebangs:
                # Match the interpreter token bounded (…/bash, env bash), not
                # any substring ("sh" must not match "fish").
                if re.search(rf"(?:^|[/\s]){re.escape(token)}(?:\s|$)", lowered):
                    return s
    return None


# ---------------------------------------------------------------------------
# Generic engine
# ---------------------------------------------------------------------------

def _strip_line_comment(line: str, markers: Tuple[str, ...]) -> str:
    """Remove a trailing line comment, respecting simple string quoting."""
    if not markers:
        return line
    in_s: Optional[str] = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_s:
            if ch == "\\":
                i += 2
                continue
            if ch == in_s:
                in_s = None
        elif ch in "'\"`":
            in_s = ch
        else:
            for m in markers:
                if line.startswith(m, i):
                    return line[:i]
        i += 1
    return line


def _code_brace_deltas(
    line: str,
    markers: Tuple[str, ...],
    *,
    c_block_comments: bool,
    in_block_comment: bool,
    quote_chars: str = "'\"`",
) -> Tuple[List[int], bool]:
    """Brace events (+1/-1) for the CODE characters of one line.

    String- and comment-aware (reviewer A, F1: a `"}"` literal or a
    `/* } */` comment used to end extents early while the balance lint said
    ok). Quote handling uses the CLOSES-ON-THIS-LINE rule: a quote char
    opens a string only if its closer appears later in the same line —
    this keeps apostrophes in prose from swallowing the rest of the line.
    `quote_chars` is per-language (rust drops `'` entirely: lifetimes make
    apostrophes non-string more often than not — reviewer C). Multi-line
    string literals remain a documented residual (state resets per line).
    Returns (events, block_comment_state_after_line).
    """
    # Fast path (reviewer C, P2 perf): most lines carry no brace, no
    # comment-open and no state — skip the char walk entirely.
    if not in_block_comment and "{" not in line and "}" not in line:
        if not (c_block_comments and "/*" in line):
            return [], False
    events: List[int] = []
    i = 0
    n = len(line)
    while i < n:
        if in_block_comment:
            end = line.find("*/", i)
            if end == -1:
                return events, True
            i = end + 2
            in_block_comment = False
            continue
        ch = line[i]
        if c_block_comments and line.startswith("/*", i):
            in_block_comment = True
            i += 2
            continue
        hit_marker = False
        for m in markers:
            if m and line.startswith(m, i):
                hit_marker = True
                break
        if hit_marker:
            break
        if ch in quote_chars:
            close = line.find(ch, i + 1)
            # Skip escaped closers.
            while close != -1 and close > 0 and line[close - 1] == "\\":
                close = line.find(ch, close + 1)
            if close != -1:
                i = close + 1
                continue
            i += 1
            continue
        if ch == "{":
            events.append(1)
        elif ch == "}":
            events.append(-1)
        i += 1
    return events, in_block_comment


class BraceExtentIndex:
    """ONE stack pass over the whole file answering every extent query.

    Reviewer C, P0-2: the per-declaration forward scan was O(n) per decl —
    a file where many declarations' braces never close (a truncated 4MB
    read, generated half-open code) walked to EOF per decl, measured
    quadratic (4000 unclosed decls = 91s). This index walks the lines ONCE
    (string/comment-aware) and records, for each line with an opening
    brace, the line where that brace's block closes — O(total) build, O(1)
    per query. Unclosed braces simply never record a close.
    """

    def __init__(self, lines: List[str], markers: Tuple[str, ...], *, c_block_comments: bool, quote_chars: str) -> None:
        self._lines = lines
        self._markers = markers
        # line index of first open event -> 1-based close line of THAT brace
        self._close_for_open_line: Dict[int, int] = {}
        # line index -> True if any open event occurs on it
        self._opens_on_line: Dict[int, bool] = {}
        stack: List[int] = []  # line indices of unmatched opens
        in_block_comment = False
        for j, raw in enumerate(lines):
            events, in_block_comment = _code_brace_deltas(
                raw,
                markers,
                c_block_comments=c_block_comments,
                in_block_comment=in_block_comment,
                quote_chars=quote_chars,
            )
            for delta in events:
                if delta > 0:
                    if j not in self._opens_on_line:
                        self._opens_on_line[j] = True
                    stack.append(j)
                else:
                    if stack:
                        open_line = stack.pop()
                        # Record the close for the FIRST open of that line
                        # only (a decl's block is its first brace).
                        self._close_for_open_line.setdefault(open_line, j + 1)

    def extent_from(
        self,
        start_idx: int,
        *,
        abort_patterns: Tuple[Pattern[str], ...] = (),
        max_lookahead: int = 10,
    ) -> Optional[int]:
        """End line of the block whose `{` opens at/after lines[start_idx].

        Before the block opens, a later line matching any declaration
        pattern aborts — a new declaration before your `{` means you have
        no block (reviewer A, F2: kotlin `data class X(...)` used to steal
        the next class's extent).
        """
        for j in range(start_idx, min(start_idx + max_lookahead + 1, len(self._lines))):
            if j > start_idx:
                code_line = _strip_line_comment(self._lines[j], self._markers)
                for pat in abort_patterns:
                    if pat.match(code_line):
                        return None
            if self._opens_on_line.get(j):
                return self._close_for_open_line.get(j)
        return None


class EndKeywordExtentIndex:
    """ONE pass over the file answering every `end`-block extent query.

    Same P0-2 fix as BraceExtentIndex, ruby edition (reviewer C measured
    2000 bodyless `def`s at 19.3s under the per-decl scan). Reviewer A (F3)
    hardening carried over: one-line `def x; ...; end` self-closes;
    assigned blocks (`label = case ... end`) open; heredoc bodies and
    `=begin/=end` comment blocks are skipped.
    """

    def __init__(self, lines: List[str], spec: LanguageSpec) -> None:
        opener_re = re.compile(r"^\s*(?:" + "|".join(re.escape(w) for w in spec.end_block_openers) + r")\b")
        assigned_opener_re = re.compile(r"=\s*(?:case|if|unless|begin)\b")
        inline_do_re = re.compile(r"\bdo\s*(?:\|[^|]*\|)?\s*$")
        end_re = re.compile(r"^\s*end\b")
        inline_end_re = re.compile(r";\s*end\b\s*$")
        heredoc_open_re = re.compile(r"<<[~-]?(?P<q>[\"'`]?)(?P<tag>\w+)(?P=q)")
        self._close_for_line: Dict[int, int] = {}
        stack: List[int] = []
        heredoc_tag: Optional[str] = None
        in_eq_comment = False
        for j, raw in enumerate(lines):
            if heredoc_tag is not None:
                if raw.strip() == heredoc_tag:
                    heredoc_tag = None
                continue
            if in_eq_comment:
                if raw.startswith("=end"):
                    in_eq_comment = False
                continue
            if raw.startswith("=begin"):
                in_eq_comment = True
                continue
            code = _strip_line_comment(raw, spec.line_comment).rstrip()
            stripped = code.strip()
            if not stripped:
                continue
            is_opener = bool(
                opener_re.match(code) or assigned_opener_re.search(code) or inline_do_re.search(code)
            )
            if is_opener:
                if inline_end_re.search(code):
                    # One-line `def name; body; end` self-closes.
                    self._close_for_line[j] = j + 1
                else:
                    stack.append(j)
            if end_re.match(code) is not None and stack:
                open_line = stack.pop()
                self._close_for_line[open_line] = j + 1
            m_heredoc = heredoc_open_re.search(code)
            if m_heredoc:
                heredoc_tag = m_heredoc.group("tag")

    def extent_from(self, start_idx: int) -> Optional[int]:
        return self._close_for_line.get(start_idx)


def _looks_binary(sample: bytes) -> bool:
    if b"\x00" in sample:
        return True
    # High ratio of non-text bytes → binary.
    text_chars = bytes(range(0x20, 0x7F)) + b"\n\r\t\f\b"
    if not sample:
        return False
    nontext = sum(1 for b in sample if b not in text_chars and b < 0x80)
    return (nontext / len(sample)) > 0.30


def read_text_bounded(path: Path) -> Tuple[Optional[str], Optional[str], bool, str]:
    """Read a file for analysis: (text, error, truncated, encoding_note).

    Reads at most MAX_ANALYZE_BYTES; refuses binary. Decoding order
    (reviewer C, P1-3/P2): UTF-16/32 by BOM (Windows toolchains emit these —
    "binary" would be a false claim), else UTF-8 via an INCREMENTAL decoder
    (a multi-byte character split at the truncation boundary must not fail
    the whole file into latin-1 mojibake), else latin-1 with a labeled note.
    """
    try:
        raw = path.read_bytes()
    except Exception as e:
        return None, f"Error reading file: {e}", False, ""
    truncated = False
    if len(raw) > MAX_ANALYZE_BYTES:
        raw = raw[:MAX_ANALYZE_BYTES]
        truncated = True

    encoding_note = ""
    text: Optional[str] = None
    # BOM-declared wide encodings BEFORE the binary sniff (their null bytes
    # are the encoding, not binary content). Order matters: UTF-32 BOMs
    # start with the UTF-16 LE BOM bytes.
    for bom, enc in (
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
    ):
        if raw.startswith(bom):
            try:
                text = raw[len(bom):].decode(enc, errors="replace")
                encoding_note = f"decoded as {enc} (BOM)"
            except Exception:
                return None, "binary", truncated, ""
            break

    if text is None:
        if _looks_binary(raw[:8192]):
            return None, "binary", truncated, ""
        try:
            import codecs

            # Incremental decode tolerates a truncation-split trailing
            # character (dropped, final=False) while still REFUSING genuinely
            # non-UTF-8 bytes mid-file (reviewer C: a plain decode() failed
            # on the split char and silently served latin-1 mojibake for 4MB).
            text = codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
                encoding_note = "decoded as latin-1 (not valid UTF-8) #FALLBACK"
            except Exception:
                return None, "binary", truncated, ""

    # Strip a UTF-8 BOM: it otherwise rides line 1 and silently defeats the
    # anchored first-line patterns and shebang routing (reviewer C, P2).
    if text.startswith("\ufeff"):
        text = text[1:]
    if truncated:
        # Never cut mid-line: drop the partial tail line.
        text = text.rsplit("\n", 1)[0]
    return text, None, truncated, encoding_note


def split_lines_like_read_file(text: str) -> List[str]:
    """Split on \\n only, tolerating \\r\\n — NEVER on \\f/NEL/U+2028.

    str.splitlines() splits on form feeds and unicode separators, drifting
    this tool's line numbers off the ones read_file(start_line) uses — and
    these numbers exist to feed read_file (reviewer C, P2 truth bug).
    """
    lines = text.split("\n")
    return [l[:-1] if l.endswith("\r") else l for l in lines]


def _scan_generic_delimiters(
    lines: List[str],
    markers: Tuple[str, ...],
    *,
    c_block_comments: bool = True,
    balance_pairs: str = "{}()[]",
    skip_heredocs: bool = False,
    quote_chars: str = "'\"`",
) -> List[str]:
    """Whole-file bracket balance (cheap sanity signal, not a parser).

    Shares the string/comment/heredoc rules with the extent scanner so the
    lint and the extents cannot contradict each other (reviewer A, F1: a
    `/* } */` comment produced BOTH a wrong extent and a false
    unbalanced-{} lint; heredoc bodies are data, not code).
    """
    counts: Dict[str, int] = {ch: 0 for ch in balance_pairs}
    in_block_comment = False
    heredoc_open_re = re.compile(r"<<[~-]?(?P<q>[\"'`]?)(?P<tag>\w+)(?P=q)") if skip_heredocs else None
    heredoc_tag: Optional[str] = None
    for raw in lines:
        if heredoc_tag is not None:
            if raw.strip() == heredoc_tag:
                heredoc_tag = None
            continue
        if heredoc_open_re is not None:
            m_h = heredoc_open_re.search(raw)
            if m_h:
                heredoc_tag = m_h.group("tag")
                raw = raw[: m_h.start()]  # the opener's prefix is still code
        # Fast path (reviewer C, P2 perf): skip lines with no countable char.
        if not in_block_comment and not any(ch in raw for ch in balance_pairs):
            if not (c_block_comments and "/*" in raw):
                continue
        i = 0
        n = len(raw)
        while i < n:
            if in_block_comment:
                end = raw.find("*/", i)
                if end == -1:
                    i = n
                    break
                i = end + 2
                in_block_comment = False
                continue
            ch = raw[i]
            if c_block_comments and raw.startswith("/*", i):
                in_block_comment = True
                i += 2
                continue
            hit_marker = False
            for m in markers:
                if m and raw.startswith(m, i):
                    hit_marker = True
                    break
            if hit_marker:
                break
            if ch in quote_chars:
                close = raw.find(ch, i + 1)
                while close != -1 and close > 0 and raw[close - 1] == "\\":
                    close = raw.find(ch, close + 1)
                if close != -1:
                    i = close + 1
                    continue
                i += 1
                continue
            if ch in counts:
                counts[ch] += 1
            i += 1
    issues: List[str] = []
    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        if opener not in counts or closer not in counts:
            continue
        if counts[opener] != counts[closer]:
            issues.append(f"  - unbalanced {opener}{closer}: {counts[opener]} open vs {counts[closer]} close")
    return issues


# Fixed render order for declaration sections. EVERY DeclPattern.kind in the
# spec table must be listed here (a missing kind would silently drop its
# section from the outline — pinned by test_every_spec_kind_is_emittable).
_EMIT_KIND_ORDER: Tuple[str, ...] = (
    "types",
    "namespaces",
    "modules",
    "functions",
    "constants",
    "macros",
    "rules",
    "statements",
    "stages",
    "targets",
    "blocks",
    "headings",
    "keys",
    "tables",
)


def _emit_section(out: List[str], label: str, entries: List[str]) -> None:
    out.append(f"{label}:" if entries else f"{label}: []")
    out.extend(entries[:MAX_SECTION_ENTRIES])
    if len(entries) > MAX_SECTION_ENTRIES:
        # Recovery path, not just honesty: the agent can still reach entries
        # beyond the cap (reviewer B, P2-4).
        out.append(
            f"  - ... ({len(entries) - MAX_SECTION_ENTRIES} more) #TRUNCATION — use search_files('<name>') for entries beyond the cap"
        )


def analyze_with_spec(
    path: Path,
    display_path: str,
    text: str,
    spec: LanguageSpec,
    *,
    truncated: bool = False,
    encoding_note: str = "",
) -> str:
    """Outline `text` per `spec`. Returns the formatted tool answer."""
    lines = split_lines_like_read_file(text)
    total_lines = len(lines)

    out: List[str] = [
        f"Code Analysis: {display_path} (language={spec.name}, lines={total_lines})",
        ANALYZE_CODE_NEXT_STEP_HINT,
        f"language: {spec.name}",
    ]
    if truncated:
        out.append(f"notice: #TRUNCATION analyzed only the first {MAX_ANALYZE_BYTES // (1024 * 1024)} MB of the file.")
    if encoding_note:
        out.append(f"notice: {encoding_note}")

    # JSON gets a real validity check instead of pattern scanning — and it
    # dispatches BEFORE the minified guard: json.loads is not line-anchored,
    # so a compact package-lock.json or a ledger JSONL with long records
    # still deserves its parse lane (reviewer B, P1-1).
    if spec.name == "json":
        return _analyze_json(out, path, text, lines)

    # Minified/generated guard: line-anchored outlining is meaningless.
    longest = max((len(l) for l in lines), default=0)
    if longest > MAX_LINE_CHARS_FOR_OUTLINE:
        out.append(
            f"diagnostics: longest_line={longest} chars — file looks generated/minified; line-anchored outline skipped."
        )
        out.append("notes: use search_files() for targeted lookups in generated files.")
        return "\n".join(out)

    imports: List[str] = []
    sections: Dict[str, List[str]] = {}
    todos: List[str] = []
    todo_re = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]?(.{0,80})")

    # Cross-line skip state (reviewer A: heredoc bodies, markdown fences and
    # `#if 0` blocks are DATA, not declarations — a shell heredoc containing
    # `inner() {` used to be outlined as a live function).
    heredoc_open_re = re.compile(r"<<[~-]?(?P<q>[\"'`]?)(?P<tag>\w+)(?P=q)")
    skip_heredoc = "heredoc" in spec.skip_regions
    skip_fence = "fence" in spec.skip_regions
    skip_if0 = "preproc_if0" in spec.skip_regions
    heredoc_tag: Optional[str] = None
    in_fence = False
    if0_depth = 0

    abort_patterns = tuple(d.pattern for d in spec.decl_patterns)

    # One-pass extent indexes (reviewer C, P0-2: per-decl forward scans were
    # quadratic when braces never close — 4000 unclosed decls took 91s).
    brace_index = (
        BraceExtentIndex(lines, spec.line_comment, c_block_comments=spec.c_block_comments, quote_chars=spec.quote_chars)
        if spec.block_style == "brace"
        else None
    )
    end_index = EndKeywordExtentIndex(lines, spec) if spec.block_style == "end" else None

    for idx, raw in enumerate(lines):
        line_no = idx + 1

        if heredoc_tag is not None:
            if raw.strip() == heredoc_tag:
                heredoc_tag = None
            continue
        if in_fence:
            if raw.lstrip().startswith("```"):
                in_fence = False
            continue
        if skip_fence and raw.lstrip().startswith("```"):
            in_fence = True
            continue
        if skip_if0:
            preproc = raw.strip()
            if if0_depth > 0:
                if preproc.startswith("#if"):
                    if0_depth += 1
                elif preproc.startswith("#endif"):
                    if0_depth -= 1
                continue
            if preproc.startswith("#if 0"):
                if0_depth = 1
                continue

        code = _strip_line_comment(raw, spec.line_comment)
        stripped = code.strip()

        # TODO markers ride the RAW line (they live in comments).
        m_todo = todo_re.search(raw)
        if m_todo and len(todos) < MAX_SECTION_ENTRIES + 10:
            todos.append(f"  - {line_no}: {m_todo.group(1)} {m_todo.group(2).strip()}".rstrip())

        if not stripped:
            continue
        if skip_heredoc:
            m_heredoc = heredoc_open_re.search(code)
            if m_heredoc:
                heredoc_tag = m_heredoc.group("tag")
                # The opener line itself is still code: fall through so a
                # declaration on it (rare) is not lost; the BODY is skipped.

        matched_import = False
        for pat in spec.import_patterns:
            m = pat.match(code)
            if m:
                name = (m.groupdict().get("name") or "").strip()
                if name:
                    imports.append(f"  - {line_no}: {name}")
                matched_import = True
                break
        if matched_import:
            continue

        for decl in spec.decl_patterns:
            m = decl.pattern.match(code)
            if not m:
                continue
            name = (m.groupdict().get("name") or "").strip()
            if not name:
                break
            if decl.requires_brace and "{" not in code:
                # Allman verification (reviewer A, F4): a definition without
                # a same-line brace is accepted only when the NEXT non-blank
                # line opens the block — otherwise it's a prototype/macro
                # shape and must not be listed.
                opens_next = False
                for j in range(idx + 1, min(idx + 3, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        continue
                    opens_next = nxt.startswith("{")
                    break
                if not opens_next:
                    continue
            params = (m.groupdict().get("params") or "").strip() if "params" in (m.groupdict() or {}) else ""
            recv = (m.groupdict().get("recv") or "").strip() if "recv" in (m.groupdict() or {}) else ""
            end_line: Optional[int] = None
            # A `;`-terminated declaration has NO body (trait/interface
            # method, forward decl) — scanning forward for a brace would
            # steal the NEXT block's extent.
            is_bodyless = stripped.endswith(";") and "{" not in stripped
            if decl.block and not is_bodyless and brace_index is not None:
                end_line = brace_index.extent_from(idx, abort_patterns=abort_patterns)
            elif decl.block and not is_bodyless and end_index is not None:
                end_line = end_index.extent_from(idx)
            elif spec.block_style == "heading":
                # A heading's range ends where the next same-or-higher
                # heading starts (reviewer B: needed to bound "edit the
                # Install section" reads).
                level = len(stripped) - len(stripped.lstrip("#"))
                for j in range(idx + 1, len(lines)):
                    nxt = lines[j].strip()
                    if nxt.startswith("#"):
                        nxt_level = len(nxt) - len(nxt.lstrip("#"))
                        if 0 < nxt_level <= level:
                            end_line = j  # section ends the line BEFORE the next heading
                            break
                else:
                    end_line = len(lines)
            rng = f"{line_no}-{end_line}" if end_line and end_line > line_no else f"{line_no}"
            sig = name
            if recv:
                sig = f"({recv}) {name}"
            if params or decl.kind == "functions":
                sig = f"{sig}({params})"
            sections.setdefault(decl.kind, []).append(f"  - {rng}: {sig}")
            break

    # Compute the delimiter scan ONCE (reviewer B, P2-1: this is a
    # char-by-char pass over up to 4MB — running it twice doubled the
    # dominant cost on large files).
    delimiter_issues: List[str] = (
        _scan_generic_delimiters(
            lines,
            spec.line_comment,
            c_block_comments=spec.c_block_comments,
            balance_pairs=spec.balance_pairs,
            skip_heredocs="heredoc" in spec.skip_regions,
            quote_chars=spec.quote_chars,
        )
        if spec.block_style == "brace"
        else []
    )

    diagnostics: List[str] = []
    if spec.block_style == "brace":
        diagnostics.append("delimiters=ok" if not delimiter_issues else f"delimiters={len(delimiter_issues)} issue(s)")
    if todos:
        diagnostics.append(f"todo_markers={len(todos)}")
    out.append("diagnostics: " + ("; ".join(diagnostics) if diagnostics else "none"))

    summary_bits = [f"imports={len(imports)}"] + [f"{k}={len(v)}" for k, v in sections.items()]
    out.append("summary: " + "; ".join(summary_bits))

    if spec.block_style == "brace":
        if delimiter_issues:
            out.append("lint:")
            out.extend(delimiter_issues)
        else:
            out.append("lint: []")

    _emit_section(out, "imports", imports)
    # NOTE: every DeclPattern.kind used by any LanguageSpec must appear here
    # or its section silently vanishes from the output (truth bug) — pinned
    # by tests against the live spec table.
    for kind in _EMIT_KIND_ORDER:
        if kind in sections:
            _emit_section(out, kind, sections[kind])
    if todos:
        _emit_section(out, "todo_markers", todos)

    if spec.notes:
        out.append(f"notes: {spec.notes}")
    return "\n".join(out)


def _analyze_json(out: List[str], path: Path, text: str, lines: List[str]) -> str:
    import json as _json

    is_jsonl = path.suffix.lower() in {".jsonl", ".ndjson"}
    if is_jsonl:
        bad: List[str] = []
        n_records = 0
        for i, raw in enumerate(lines, 1):
            s = raw.strip()
            if not s:
                continue
            n_records += 1
            try:
                _json.loads(s)
            except Exception as e:
                if len(bad) < 10:
                    bad.append(f"  - line {i}: {e}")
        out.append(f"diagnostics: jsonl_records={n_records}; invalid={len(bad)}")
        out.append("summary: " + (f"{n_records} records, {len(bad)} invalid" if bad else f"{n_records} records, all parse"))
        if bad:
            out.append("invalid_lines:")
            out.extend(bad)
        return "\n".join(out)

    try:
        doc = _json.loads(text)
    except Exception as e:
        out.append(f"diagnostics: parse=error ({e})")
        return "\n".join(out)
    out.append("diagnostics: parse=ok")
    if isinstance(doc, dict):
        keys = list(doc.keys())
        out.append(f"summary: object with {len(keys)} top-level key(s)")
        _emit_section(out, "keys", [f"  - {k}" for k in keys])
    elif isinstance(doc, list):
        out.append(f"summary: array with {len(doc)} element(s)")
    else:
        out.append(f"summary: top-level {type(doc).__name__}")
    return "\n".join(out)


def analyze_generic(
    display_path: str, text: str, *, language_hint: str = "", truncated: bool = False, encoding_note: str = ""
) -> str:
    """Never-refuse fallback: an honest structural sample for unknown text."""
    lines = split_lines_like_read_file(text)
    total_lines = len(lines)
    out: List[str] = [
        f"Code Analysis: {display_path} (language=unknown, lines={total_lines})",
        "notice: language not recognized — this is a GENERIC text outline (structure sample + metrics), not a parsed code outline.",
    ]
    if language_hint:
        out.append(
            f"notice: requested language '{language_hint}' is not in the analyzer's vocabulary; falling back to the generic outline. #FALLBACK"
        )
    if truncated:
        out.append(f"notice: #TRUNCATION analyzed only the first {MAX_ANALYZE_BYTES // (1024 * 1024)} MB of the file.")
    if encoding_note:
        out.append(f"notice: {encoding_note}")

    longest = max((len(l) for l in lines), default=0)
    non_blank = sum(1 for l in lines if l.strip())
    out.append(f"metrics: lines={total_lines}; non_blank={non_blank}; longest_line={longest} chars")

    if longest > MAX_LINE_CHARS_FOR_OUTLINE:
        out.append("notes: file looks generated/minified; use search_files() for targeted lookups.")
        return "\n".join(out)

    # Structure sample: column-0 lines that look like section anchors.
    anchors: List[str] = []
    for i, raw in enumerate(lines, 1):
        if not raw or raw[0].isspace():
            continue
        s = raw.strip()
        if len(s) < 3:
            continue
        anchors.append(f"  - {i}: {s[:120]}")
    _emit_section(out, "top_level_lines", anchors)

    todo_re = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]?(.{0,80})")
    todos = [
        f"  - {i}: {m.group(1)} {m.group(2).strip()}".rstrip()
        for i, raw in enumerate(lines, 1)
        if (m := todo_re.search(raw))
    ]
    if todos:
        _emit_section(out, "todo_markers", todos)

    out.append("notes: pass language=<name> to force a specific analyzer; supported names are listed in the tool description.")
    return "\n".join(out)
