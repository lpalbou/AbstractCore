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

# Output bounds: outlines are prompt currency, so a pathological file must not
# flood the model context. But a cap that hides data is a cost, not a feature:
# measured over 2429 real files / 8997 sections, section sizes run p50=2,
# p90=13, p99=57, p999=324, max=2293. A cap of 50 therefore truncated 1.3% of
# ALL sections — including the ones that matter most, like a 463-heading
# CHANGELOG whose whole point is navigation. At 500 that falls to 0.03%, and
# the worst realistic outline costs ~4k tokens, which is far cheaper than the
# re-reads a missing outline forces. Whenever this cap does bite, the section
# must say so AND say how to reach what it hid (see _emit_section).
MAX_SECTION_ENTRIES = 1000
# Read bound: analysis is line-oriented; a multi-MB artifact (bundle, log,
# generated code) is truncated with a label rather than freezing the tool.
MAX_ANALYZE_BYTES = 4 * 1024 * 1024
# A single enormous line (minified bundle) defeats line-anchored outlining;
# detect and say so instead of burning CPU on regexes over megabyte lines.
MAX_LINE_CHARS_FOR_OUTLINE = 5000
# Per-entry text bound (a signature, a TODO message, a generic anchor line).
# Unlike a section cap this one cannot lose a whole entry — the entry and its
# LINE NUMBER are still printed, so the full text is one read_file away — but
# it must still be visible when it bites, hence the ellipsis plus the notice
# that names the recovery.
MAX_ENTRY_TEXT_CHARS = 160
# read_file refuses a range larger than this, so a recovery hint that names a
# wider one is not a recovery at all. Kept in sync with read_file's own bound.
READ_FILE_LINE_BUDGET = 2000
# A per-section cap cannot bound N sections: four sections each just under the
# cap produced a 26k-token outline for a 45k-token file, and one hand-written
# module here reaches 16k tokens with NO section truncated. This is the
# backstop — sections are served in _EMIT_KIND_ORDER, so the kinds a reader
# navigates by (types, functions) are funded first, and whatever the budget
# cannot fund is reported in the trailing block like any other omission.
MAX_OUTLINE_CHARS = 60_000


@dataclass
class Truncation:
    """One thing the outline did not show, and the way to get it."""

    what: str
    shown: int
    total: int
    recovery: str


class TruncationLog:
    """Collects every omission so the answer can END with one actionable block.

    Scattering `notice:` lines and `(N more)` markers through a long outline
    makes the reader reconstruct the damage from fragments, and each fragment
    has to repeat the file path to stay runnable, which is neither concise nor
    reliable. One log, rendered last, states in a single place what is missing
    and the exact call that returns it.
    """

    __slots__ = ("_items", "_spent", "_budget")

    def __init__(self, budget_chars: int = MAX_OUTLINE_CHARS) -> None:
        self._items: List[Truncation] = []
        self._spent = 0
        self._budget = budget_chars

    def allowance(self, entries: List[str]) -> int:
        """How many of `entries` the remaining output budget can afford."""
        left = self._budget - self._spent
        if left <= 0:
            return 0
        taken = 0
        for entry in entries[:MAX_SECTION_ENTRIES]:
            cost = len(entry) + 1
            if taken and left - cost < 0:
                break
            left -= cost
            taken += 1
        return taken

    def spend(self, entries: List[str]) -> None:
        self._spent += sum(len(e) + 1 for e in entries)

    def add(self, what: str, recovery: str, *, shown: int = 0, total: int = 0) -> None:
        self._items.append(Truncation(what, shown, total, recovery))

    def __bool__(self) -> bool:
        return bool(self._items)

    def render(self) -> List[str]:
        """The trailing block. Empty when nothing was withheld."""
        if not self._items:
            return []
        out = [f"#TRUNCATION {len(self._items)} item(s) withheld — run these to see the rest:"]
        for t in self._items:
            count = f" ({t.shown} of {t.total} shown)" if t.total else ""
            out.append(f"  - {t.what}{count}: {t.recovery}")
        return out


def _elide(text: str, limit: int = MAX_ENTRY_TEXT_CHARS) -> Tuple[str, bool]:
    """Shorten one entry's text, reporting whether it was shortened."""
    if len(text) <= limit:
        return text, False
    return text[: limit - 1].rstrip() + "…", True

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
    # `end`-family knobs. These used to be hardcoded ruby-isms inside
    # EndKeywordExtentIndex, which is why lua/elixir could not be added as
    # DATA (the module's rule 1). Each language declares its own shapes:
    #   end_assigned_openers  words that open a block after `=`
    #                         (ruby `x = case`, lua `f = function`)
    #   end_inline_block_re   a trailing opener (ruby `do |x|`, elixir `fn ->`)
    #   end_inline_close_re   an opener that CLOSES on its own line
    #                         (ruby `def x; end`, lua `function f() return 1 end`)
    #   end_bodyless_re       opener-SHAPED lines with no block at all
    #                         (elixir `def foo, do: :ok`) — must not push
    end_assigned_openers: Tuple[str, ...] = ()
    end_inline_block_re: Optional[Pattern[str]] = None
    end_inline_close_re: Optional[Pattern[str]] = None
    end_bodyless_re: Optional[Pattern[str]] = None
    #   end_openers_need_terminator
    #                         an opener does not open a block until its
    #                         terminator (`end_inline_block_re`, i.e. a
    #                         trailing `do`) appears. Elixir declaration
    #                         HEADS wrap: `mix format` routinely emits
    #                         `def f(x)` / `when is_map(x) do`, and
    #                         `def f(%{` / `}) do`. Counting the wrapped
    #                         `do` as a SECOND block handed the function's
    #                         `end` away and made it span to the end of the
    #                         module. Holding the head pending until its
    #                         `do` arrives handles every wrap shape, and
    #                         lets a `, do:` on a later line cancel it.
    #                         Ruby leaves this off: `def foo` opens on its
    #                         own line, and its `each do |x|` really is a
    #                         second block.
    end_openers_need_terminator: bool = False
    # Whether the language has C-style /* */ block comments (drives the
    # string/comment-aware brace scanner; reviewer A, F1).
    c_block_comments: bool = True
    # Which bracket pairs the whole-file balance lint counts. Shell drops
    # parens (case arms `a)` are legal unbalanced parens; reviewer A, P2-11).
    balance_pairs: str = "{}()[]"
    # Cross-line regions every scanner reads as DATA: comments, long
    # strings, heredocs, fences, `#if 0` blocks. See SkipRegion — one
    # declarative type replaced a string enum whose three region kinds each
    # had their detection regex baked into the engine, which is why a
    # language needing a fourth kind (elixir `"""`, lua `[[ ]]`) could not
    # be added as data. Reviewer A F3/P2-5/P2-11/P2-12.
    skip_regions: Tuple[SkipRegion, ...] = ()
    # Extensions shared with an UNRELATED language, claimed only when
    # `sniff` matches the file head. `.m` is Objective-C or MATLAB/Octave
    # depending on content; labelling a MATLAB file `objectivec` and listing
    # nothing is a confident lie, while the generic lane at least surfaces
    # its `function` lines.
    sniff_extensions: Tuple[str, ...] = ()
    sniff: Optional[Pattern[str]] = None
    notes: str = ""


def _rx(p: str) -> Pattern[str]:
    return re.compile(p)


@dataclass(frozen=True)
class SkipRegion:
    """A cross-line region every scanner must read as DATA, not as code.

    Comments, long strings, heredocs, markdown fences and `#if 0` blocks are
    all one shape — a line opens the region, the lines inside are inert, a
    line closes it — and each used to be implemented separately. The heredoc
    detection regex alone appeared THREE times, and the newer block-comment
    skipper was a fourth mechanism with different matching rules. They
    disagreed, which is a truth bug: a `<# … #>` powershell comment holding
    `@{` was skipped by the outline and counted by the balance lint, so the
    tool reported an unbalanced brace in the same breath as a note promising
    it had ignored that comment.

    Fields:
      open   matched against the RAW line. ANCHOR IT YOURSELF: `^=begin` for
             a column-0-only marker, `--\\[\\[` for one that may be indented.
             Give it a boundary when a longer identifier could swallow it.
      close  matched against the raw line. Omit when `close_group` applies.
      close_group  names a group of the OPEN match whose text terminates the
             region — a heredoc tag is only known once the region starts.
      nest_open  when set the region counts depth: haskell's `{- {- -} -}`
             genuinely nests, and a C `#if 0` block is ended by the `#endif`
             matching its own nested `#if`, not the first one seen.
      counts_todos  a TODO in a COMMENT is a real TODO and is still reported;
             a TODO inside a heredoc body, a fenced sample or a disabled
             `#if 0` block is data, matching long-standing behaviour.
      after_line_comment  search for the opener in the line-comment-STRIPPED
             text. A shell comment that merely mentions `<<EOF` must not
             open a heredoc; only heredoc wants this, because every other
             opener here IS comment syntax and must see the raw line.

    Code OUTSIDE the region on the opening line is always returned to the
    caller — the prefix before the opener, plus the suffix after the closer
    when the region opens and closes on one line. Discarding those was a
    single boolean away from deleting `function Get-Thing {` because it
    carried a trailing `<# … #>` comment, taking its `{` with it and
    inventing an unbalanced-brace lint on a well-formed file.
    """

    open: Pattern[str]
    close: Optional[Pattern[str]] = None
    close_group: Optional[str] = None
    nest_open: Optional[Pattern[str]] = None
    counts_todos: bool = True
    after_line_comment: bool = False


# Regions shared by several languages. Defined once so the three former
# copies of the heredoc regex cannot drift apart again.
HEREDOC_REGION = SkipRegion(
    open=_rx(r"<<[~-]?(?P<q>[\"'`]?)(?P<tag>\w+)(?P=q)"),
    close_group="tag",
    counts_todos=False,
    after_line_comment=True,
)
FENCE_REGION = SkipRegion(open=_rx(r"^\s*```"), close=_rx(r"^\s*```"), counts_todos=False)
PREPROC_IF0_REGION = SkipRegion(
    open=_rx(r"^\s*#\s*if\s+0\b"),
    nest_open=_rx(r"^\s*#\s*if"),
    close=_rx(r"^\s*#\s*endif"),
    counts_todos=False,
)


class SkipRegionTracker:
    """One line-at-a-time state machine over a spec's SkipRegions.

    Every pass that walks lines owns one of these — the declaration loop,
    BOTH extent indexes, and the balance lint — so they cannot disagree
    about what counts as code.

    `feed()` returns the CODE TEXT of a line, or None when the line is
    inert. Callers that only need a yes/no can test `is None`.
    """

    __slots__ = (
        "_regions", "_markers", "_active", "_tag", "_depth",
        "_line_no", "_opened_at", "_line_comment",
    )

    def __init__(self, regions: Tuple[SkipRegion, ...], markers: Tuple[str, ...] = ()) -> None:
        self._regions = regions
        self._markers = markers
        self._active: Optional[SkipRegion] = None
        self._tag: Optional[str] = None
        self._depth = 0
        self._line_no = 0
        self._opened_at = 0
        self._line_comment = False

    @property
    def inside_comment(self) -> bool:
        """Whether the line just fed sat in a region whose TODOs count.

        Per-LINE, not per-state: a one-line `<!-- TODO … -->` opens and
        closes within the same call, so asking whether a region is still
        active afterwards misses exactly the comments TODOs live in.
        """
        return self._line_comment

    @property
    def unterminated_at(self) -> int:
        """1-based line where a still-open region began, else 0.

        An unclosed `--[[` or `=begin` swallows the rest of the file. That
        is the correct reading of the source, but swallowing it SILENTLY
        leaves the model with a short outline and no reason to doubt it.
        """
        return self._opened_at if self._active is not None else 0

    def _closes_at(self, region: SkipRegion, text: str, start: int) -> Optional[int]:
        """Walk `text` from `start` applying nest/close events in order.

        Returns the offset just past the closer that ended the region, or
        None if it is still open — the caller needs that offset to hand back
        the code following a region that opened and closed on one line.
        """
        events = []
        if region.close is not None:
            events.extend((m.start(), -1, m.end()) for m in region.close.finditer(text, start))
        if region.nest_open is not None:
            events.extend((m.start(), +1, m.end()) for m in region.nest_open.finditer(text, start))
        for _pos, delta, end in sorted(events):
            self._depth += delta
            if self._depth <= 0:
                return end
        return None

    def feed(self, raw: str) -> Optional[str]:
        self._line_no += 1
        self._line_comment = False
        if not self._regions:
            return raw
        if self._active is not None:
            region = self._active
            self._line_comment = region.counts_todos
            if region.close_group is not None:
                # A heredoc ends on a line that is EXACTLY its tag.
                if raw.strip() == self._tag:
                    self._active = None
                    self._tag = None
                return None
            closed_at = self._closes_at(region, raw, 0)
            if closed_at is None:
                return None
            self._active = None
            self._depth = 0
            # Code may follow the closer on the same line.
            tail = raw[closed_at:]
            return tail if tail.strip() else None

        # Choose the region opening EARLIEST on the line: lua's `--[[`
        # comment starts two characters before the `[[` long-string it
        # contains, so position — not table order — picks the right one.
        best: Optional[Tuple[int, SkipRegion, Any]] = None
        for region in self._regions:
            # Only heredoc looks past a line comment; every other opener here
            # IS comment syntax, so it must see the raw line. A shell comment
            # merely MENTIONING `<<EOF` must not open a heredoc and swallow
            # the rest of the file.
            hay = _strip_line_comment(raw, self._markers) if region.after_line_comment else raw
            m = region.open.search(hay)
            if m is not None and (best is None or m.start() < best[0]):
                best = (m.start(), region, m)
        if best is None:
            return raw
        _pos, region, match = best
        self._line_comment = region.counts_todos
        # Whatever precedes the opener is code: `sql = <<~SQL` declares
        # `sql`, and `function f {  <# note #>` declares f.
        head = raw[: match.start()]
        if region.close_group is not None:
            self._active = region
            self._tag = match.group(region.close_group)
            self._depth = 1
            self._opened_at = self._line_no
            return head if head.strip() else None
        self._depth = 1
        closed_at = self._closes_at(region, raw, match.end())
        if closed_at is not None:
            # A one-line region (`{-# LANGUAGE … #-}`, `x = [[a]]`): the code
            # on BOTH sides of it survives.
            self._depth = 0
            merged = head + raw[closed_at:]
            return merged if merged.strip() else None
        self._active = region
        self._opened_at = self._line_no
        return head if head.strip() else None


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
        skip_regions=(PREPROC_IF0_REGION,),
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
        skip_regions=(PREPROC_IF0_REGION,),
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
        name="objectivec",
        aliases=("objc", "objective-c"),
        extensions=(".mm",),
        sniff_extensions=(".m",),
        sniff=_rx(r"(?m)^\s*(?:@(?:interface|implementation|protocol|end)\b|#\s*import\b)"),
        import_patterns=(
            _rx(r"^\s*#\s*(?:import|include)\s+[<\"](?P<name>[^>\"]+)[>\"]"),
            _rx(r"^\s*@import\s+(?P<name>[\w.]+)"),
        ),
        decl_patterns=(
            # @interface/@implementation close with `@end`, NOT `}` — a brace
            # extent would point at the first ivar block and misdirect the
            # follow-up read_file, so these carry a line number only.
            DeclPattern("types", _rx(r"^\s*@(?:interface|implementation|protocol)\s+(?P<name>\w+)"), block=False),
            DeclPattern("methods", _rx(r"^\s*(?P<name>[-+]\s*\([^()]*\)\s*\w+:?)")),
            DeclPattern("functions", _rx(r"^(?!\s*(?:if|for|while|switch|return|else|do|sizeof)\b)\s*(?:[\w*]+\s+)+\**(?P<name>\w+)\s*\((?P<params>[^()]*)\)\s*(?:\{|$)"), requires_brace=True),
            DeclPattern("macros", _rx(r"^\s*#\s*define\s+(?P<name>\w+)"), block=False),
        ),
        skip_regions=(PREPROC_IF0_REGION,),
        notes="Objective-C outline lists @interface/@implementation (line only — they close with @end, not a brace), methods (first selector part), C functions and macros. `.h` headers analyse as C; `.m` is claimed only when the file shows Objective-C markers, so MATLAB/Octave `.m` falls to the generic outline.",
    ),
    LanguageSpec(
        name="dart",
        extensions=(".dart",),
        import_patterns=(_rx(r"^\s*(?:import|export|part)\s+['\"](?P<name>[^'\"]+)['\"]"),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:(?:abstract|base|final|interface|sealed|mixin)\s+)*(?:class|mixin|extension|enum)\s+(?P<name>\w+)")),
            DeclPattern("types", _rx(r"^\s*typedef\s+(?P<name>\w+)"), block=False),
            # Params are `[^()]*` on purpose: a widget-tree line such as
            # `setState(() {` has a NESTED paren and must not read as a
            # method definition. Cost: params holding a function type are
            # missed — a miss, never a false positive (module rule).
            DeclPattern("functions", _rx(r"^(?!\s*(?:if|for|while|switch|return|else|do|catch|assert|await|yield)\b)\s*(?:(?:static|final|const|external|abstract|factory|late)\s+)*(?=((?:[\w<>,\[\]?.]+\s+)?))\1(?P<name>[\w.]+)\s*(?:<[^>]*>\s*)?\((?P<params>[^()]*)\)"), requires_brace=True),
            DeclPattern("constants", _rx(r"^[ \t]{0,2}(?:final|const)\s+(?:[\w<>,\[\]?]+\s+)?(?P<name>[A-Za-z_]\w*)\s*="), block=False),
        ),
        notes="Dart outline is heuristic; definitions need a brace on the same or next line, so abstract members and expression-bodied members list without extents; only declaration-level (<=2 space indent) constants are listed.",
    ),
    LanguageSpec(
        name="zig",
        extensions=(".zig",),
        import_patterns=(_rx(r"^\s*(?:pub\s+)?const\s+\w+\s*=\s*@import\(\"(?P<name>[^\"]+)\"\)"),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:pub\s+)?const\s+(?P<name>\w+)\s*=\s*(?:packed\s+|extern\s+)?(?:struct|enum|union|opaque)\b")),
            DeclPattern("functions", _rx(r"^\s*(?:pub\s+)?(?:export\s+|inline\s+|noinline\s+|extern\s+(?:\"[^\"]*\"\s+)?)*fn\s+(?P<name>\w+)\s*\((?P<params>[^()]*)\)")),
            DeclPattern("blocks", _rx(r"^\s*(?P<name>test\s+\"[^\"]*\")")),
            DeclPattern("constants", _rx(r"^(?:pub\s+)?(?:const|var)\s+(?P<name>\w+)\s*[:=]"), block=False),
        ),
        quote_chars="\"'",
        notes="Zig outline is heuristic; `@import` targets list as imports, `const X = struct {…}` reads as a type, and only column-0 (top-level) constants are listed.",
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
        name="scala",
        extensions=(".scala", ".sbt"),
        import_patterns=(_rx(r"^\s*import\s+(?P<name>[\w.]+(?:\{[^}]*\})?)"),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:(?:private|protected|final|sealed|abstract|implicit|case|open)(?:\[\w+\])?\s+)*(?:class|trait|object|enum)\s+(?P<name>\w+)")),
            # Scala method names may be symbolic (`def +(x: Int)`), so the
            # name is either a word or a run of operator characters.
            DeclPattern("functions", _rx(r"^\s*(?:(?:private|protected|final|override|implicit|inline|abstract|lazy)(?:\[\w+\])?\s+)*def\s+(?P<name>\w+|[!#%&*+\-/:<=>?@^|~]+)\s*(?:\[[^\]]*\])?\s*(?:\((?P<params>[^()]*)\))?")),
            DeclPattern("types", _rx(r"^\s*(?:(?:private|protected|final)\s+)*type\s+(?P<name>\w+)"), block=False),
            DeclPattern("constants", _rx(r"^[ \t]{0,2}(?:(?:private|protected|final|lazy|implicit|override)\s+)*val\s+(?P<name>\w+)\s*[:=]"), block=False),
        ),
        notes="Scala outline is heuristic (line-anchored declarations, brace extents); Scala 3 indentation-only syntax lists declarations without extents.",
    ),
    LanguageSpec(
        name="groovy",
        aliases=("gradle",),
        extensions=(".groovy", ".gradle"),
        import_patterns=(_rx(r"^\s*import\s+(?P<name>[\w.*]+)"),),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*(?:class|interface|trait|enum)\s+(?P<name>\w+)")),
            # Balanced `(...)` params only (no nested parens): a Gradle DSL
            # call like `exclude(group: foo(x)` must not read as a method.
            DeclPattern("functions", _rx(r"^(?!\s*(?:if|for|while|switch|return|else|do|try|catch|new)\b)\s*(?:(?:public|private|protected|static|final|synchronized|abstract|def)\s+)*(?=((?:[\w.<>\[\]]+\s+)?))\1(?P<name>\w+)\s*\((?P<params>[^()]*)\)"), requires_brace=True),
            # Gradle configuration blocks (`dependencies {`, `android {`) are
            # the thing you actually navigate a build file by; column-0 only
            # so nested DSL noise stays out.
            DeclPattern("blocks", _rx(r"^(?!(?:if|for|while|switch|else|do|try|catch|finally|synchronized|static)\b)(?P<name>[a-zA-Z_][\w.]*)\s*\{\s*$")),
        ),
        notes="Groovy/Gradle outline lists classes, methods and column-0 configuration blocks (`dependencies {`); heuristic, brace extents.",
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
        end_assigned_openers=("case", "if", "unless", "begin"),
        end_inline_block_re=_rx(r"\bdo\s*(?:\|[^|]*\|)?\s*$"),
        # A one-liner self-closes either as `…; end` or as a single
        # statement ending in `end` (`def x; true end`, `if c then v end`).
        # The tempered dot is load-bearing: `while begin …; l or r end`
        # opens TWO blocks and its trailing `end` closes only the inner
        # `begin`, so a second opener keyword must block the self-close.
        end_inline_close_re=_rx(
            r"(?:;\s*end\b\s*$)"
            r"|(?:^\s*(?:def|if|unless|while|until|for|case|class|module|begin)\b"
            r"(?:(?!\b(?:def|if|unless|while|until|for|case|class|module|begin|do)\b).)*"
            r"\bend\b\s*$)"
        ),
        c_block_comments=False,
        skip_regions=(
            HEREDOC_REGION,
            # `=begin`/`=end` are COLUMN-0 only in ruby, hence `^`.
            SkipRegion(open=_rx(r"^=begin(?![\w])"), close=_rx(r"^=end(?![\w])")),
        ),
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
        name="lua",
        extensions=(".lua",),
        shebangs=("lua", "luajit"),
        line_comment=("--",),
        import_patterns=(_rx(r"^\s*(?:local\s+[\w{}\s,]+\s*=\s*)?require\s*\(?\s*['\"](?P<name>[^'\"]+)['\"]"),),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*(?:local\s+)?function\s+(?P<name>[\w.:]+)\s*\((?P<params>[^()]*)\)")),
            DeclPattern("functions", _rx(r"^\s*(?:local\s+)?(?P<name>[\w.:]+)\s*=\s*function\s*\((?P<params>[^()]*)\)")),
            DeclPattern("tables", _rx(r"^\s*local\s+(?P<name>\w+)\s*=\s*\{"), block=False),
            DeclPattern("constants", _rx(r"^\s*local\s+(?P<name>[A-Z][A-Z0-9_]*)\s*="), block=False),
        ),
        block_style="end",
        end_block_openers=("function", "local function", "if", "for", "while", "do"),
        end_assigned_openers=("function",),
        # `..., function()` at end of line opens an anonymous block that is
        # closed by a bare `end)` — without this the `end)` would pop an
        # OUTER opener and every range below it would be wrong. This shape
        # is the norm in neovim/love2d config code.
        end_inline_block_re=_rx(r"\bfunction\s*\([^()]*\)\s*$"),
        end_inline_close_re=_rx(r"\bend\b\s*[,;)\]}]?\s*$"),
        skip_regions=(
            # Long-bracket COMMENTS, longest level first.
            SkipRegion(open=_rx(r"--\[==\["), close=_rx(r"\]==\]")),
            SkipRegion(open=_rx(r"--\[=\["), close=_rx(r"\]=\]")),
            SkipRegion(open=_rx(r"--\[\["), close=_rx(r"\]\]")),
            # Long-bracket STRINGS. A bare `end` or a `function` line inside
            # an embedded SQL/help string is text: without these the string
            # body closed the enclosing function and its contents were
            # outlined as declarations that do not exist.
            SkipRegion(open=_rx(r"\[==\["), close=_rx(r"\]==\]"), counts_todos=False),
            SkipRegion(open=_rx(r"\[=\["), close=_rx(r"\]=\]"), counts_todos=False),
            SkipRegion(open=_rx(r"\[\["), close=_rx(r"\]\]"), counts_todos=False),
        ),
        c_block_comments=False,
        notes="Lua outline is heuristic; function/end extents cover `function`, `local function` and `f = function`; `repeat/until` blocks are not listed. Long brackets close at the first matching `]]` exactly as lua does — use `--[=[` for a comment containing `]]`.",
    ),
    LanguageSpec(
        name="perl",
        extensions=(".pl", ".pm"),
        shebangs=("perl",),
        line_comment=("#",),
        import_patterns=(_rx(r"^\s*(?:use|no|require)\s+(?P<name>[\w:]+)"),),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*sub\s+(?P<name>\w+)\s*(?:\((?P<params>[^()]*)\))?")),
            DeclPattern("modules", _rx(r"^\s*package\s+(?P<name>[\w:]+)"), block=False),
            DeclPattern("constants", _rx(r"^\s*our\s+(?P<name>[$@%]\w+)"), block=False),
        ),
        c_block_comments=False,
        skip_regions=(
            HEREDOC_REGION,
            # Perl's OWN rule: a column-0 `=` followed by a letter starts
            # POD, and `=cut` ends it. The `^` anchor is load-bearing — an
            # INDENTED `=` continues an expression (`my $x\n    =f($t);`
            # is valid perl), and matching it as POD deleted the rest of a
            # valid file from the outline.
            SkipRegion(open=_rx(r"^=[a-zA-Z]"), close=_rx(r"^=cut(?![\w])")),
        ),
        notes="Perl outline lists subs, packages and `our` variables; POD (=pod…=cut) and heredoc bodies are skipped. Regex literals containing braces can confuse the balance lint.",
    ),
    LanguageSpec(
        name="elixir",
        extensions=(".ex", ".exs"),
        line_comment=("#",),
        import_patterns=(_rx(r"^\s*(?:import|alias|require|use)\s+(?P<name>[A-Z][\w.]*(?:\{[^}]*\})?)"),),
        decl_patterns=(
            DeclPattern("modules", _rx(r"^\s*defmodule\s+(?P<name>[\w.]+)")),
            DeclPattern("functions", _rx(r"^\s*(?:def|defp|defmacro|defmacrop)\s+(?P<name>[\w?!]+)\s*(?:\((?P<params>[^()]*)\))?")),
            DeclPattern("types", _rx(r"^\s*(?:defprotocol|defimpl|defstruct|defexception)\s*(?P<name>[\w.]*)"), block=False),
        ),
        block_style="end",
        end_block_openers=(
            "defmodule", "defp", "def", "defmacrop", "defmacro", "defprotocol", "defimpl",
            "if", "unless", "case", "cond", "with", "for", "receive", "try", "quote",
            "test", "describe", "setup", "fn",
        ),
        end_assigned_openers=("case", "if", "cond", "fn", "with", "try", "receive", "quote", "unless", "for"),
        # A trailing `do` or a trailing `fn … ->` opens a block that a bare
        # `end` / `end)` closes — `Enum.map(list, fn x ->` is neither
        # line-anchored nor an assignment, and without this its `end)` would
        # pop the enclosing `def`.
        end_inline_block_re=_rx(r"(?:\bdo\s*$|\bfn\b[^>]*->\s*$)"),
        end_inline_close_re=_rx(r"\bend\b\s*[,;)\]}]?\s*$"),
        # `def foo, do: :ok` is opener-SHAPED with no block.
        # `do:` may be reached with or without a preceding comma, and
        # formatted multi-line comprehensions put it on its own line.
        end_bodyless_re=_rx(r"(?:^|[\s,])do:\s*\S"),
        end_openers_need_terminator=True,
        skip_regions=(
            # `"""` heredocs: a bare `end` inside a docstring or an
            # embedded SQL string is TEXT, and closed the enclosing def.
            SkipRegion(open=_rx(r'"""'), close=_rx(r'"""'), counts_todos=False),
        ),
        c_block_comments=False,
        notes="Elixir outline is heuristic; do/end extents cover def/defmodule/case/fn blocks, and `, do:` one-liners list without extents.",
    ),
    LanguageSpec(
        name="haskell",
        aliases=("hs",),
        extensions=(".hs",),
        line_comment=("--",),
        import_patterns=(_rx(r"^\s*import\s+(?:qualified\s+)?(?P<name>[\w.]+)"),),
        decl_patterns=(
            DeclPattern("modules", _rx(r"^\s*module\s+(?P<name>[\w.]+)"), block=False),
            DeclPattern("types", _rx(r"^\s*instance\s+(?:.*?=>\s*)?(?P<name>[\w.']+(?:\s+(?!where\b)[\w.'\[\]()]+)*)"), block=False),
            DeclPattern("types", _rx(r"^\s*(?:data|newtype|type|class)\s+(?P<name>[\w.']+)"), block=False),
            # Top-level type signatures at column 0 are the real index of a
            # Haskell module; equation bodies are layout-scoped and have no
            # delimiter to measure, hence block_style="none".
            # The signature rides in `params` so the entry renders as
            # `area(Shape -> Double)`: the engine appends `()` to every
            # "functions" entry anyway, and for haskell the TYPE is the
            # information worth spending those characters on.
            DeclPattern("functions", _rx(r"^(?P<name>[a-z_]\w*'?)\s*::\s*(?P<params>.*)"), block=False),
        ),
        block_style="none",
        c_block_comments=False,
        skip_regions=(
            # Haskell block comments nest: `{- {- x -} -}` is ONE comment,
            # so a single closer must not end the outer region.
            SkipRegion(open=_rx(r"\{-"), nest_open=_rx(r"\{-"), close=_rx(r"-\}")),
        ),
        notes="Haskell outline lists the module, imports, data/class/instance heads and column-0 type signatures; layout-scoped bodies have no measurable extent, so entries carry a line number only.",
    ),
    LanguageSpec(
        name="powershell",
        aliases=("ps1", "pwsh"),
        extensions=(".ps1", ".psm1", ".psd1"),
        shebangs=("pwsh", "powershell"),
        line_comment=("#",),
        import_patterns=(
            # `$`: real scripts dot-source and import via variable paths
            # (`Import-Module $tools\sdktools.psm1`).
            _rx(r"^\s*(?i:Import-Module|using\s+module|using\s+namespace)\s+(?P<name>[\w.$:/\\-]+)"),
            _rx(r"^\s*\.\s+(?P<name>[\w.$:/\\-]+\.ps1)"),
        ),
        decl_patterns=(
            DeclPattern("functions", _rx(r"^\s*(?i:function|filter|workflow)\s+(?P<name>[\w:-]+)")),
            DeclPattern("types", _rx(r"^\s*(?i:class|enum)\s+(?P<name>\w+)")),
            # A standalone SCRIPT (the common case — CPython's own build
            # scripts, CI steps) declares no functions at all: its `param`
            # block is its interface and its column-0 assignments are its
            # structure. Without these such a file outlined to nothing,
            # which is a useless answer rather than an honest one.
            DeclPattern("blocks", _rx(r"^(?P<name>(?i:param))\s*\("), block=False),
            DeclPattern("constants", _rx(r"^\s*\$(?i:script|global|env):(?P<name>\w+)\s*="), block=False),
            DeclPattern("constants", _rx(r"^\$(?P<name>\w+)\s*="), block=False),
        ),
        c_block_comments=False,
        skip_regions=(SkipRegion(open=_rx(r"<#"), close=_rx(r"#>")),),
        notes="PowerShell outline lists functions/filters, classes, the script `param` block, and scoped or column-0 variables; <# … #> comment blocks are skipped.",
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
        skip_regions=(HEREDOC_REGION,),
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
        skip_regions=(FENCE_REGION,),
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
        name="ini",
        aliases=("cfg", "properties", "conf"),
        extensions=(".ini", ".cfg", ".properties"),
        sniff_extensions=(".conf",),
        sniff=_rx(r"(?m)^\s*\[[^\]\n]+\]\s*$"),
        filenames=("setup.cfg", "tox.ini", "pytest.ini", ".editorconfig", ".gitconfig", ".flake8"),
        line_comment=("#", ";"),
        decl_patterns=(
            DeclPattern("tables", _rx(r"^\s*(?P<name>\[[^\]]+\])\s*$"), block=False),
            # Column-0 keys only, mirroring the YAML lane: an indented line
            # is a continuation of the value above it, not a new key.
            DeclPattern("keys", _rx(r"^\s*(?P<name>[\w.$-]+)\s*[=:]"), block=False),
        ),
        block_style="none",
        c_block_comments=False,
        notes="INI/properties outline lists [sections] and `key = value` lines with line numbers; a wrapped continuation line carries no separator and is not listed. `.conf` is claimed only when the file shows a [section] header.",
    ),
    LanguageSpec(
        name="xml",
        extensions=(
            ".xml", ".xsd", ".xsl", ".xslt", ".wsdl", ".plist",
            ".csproj", ".vbproj", ".fsproj", ".props", ".targets",
        ),
        sniff_extensions=(".conf",),
        sniff=_rx(r"(?m)^\s*<(?:\?xml|!DOCTYPE)"),
        line_comment=(),
        import_patterns=(
            _rx(r"^\s*<(?:xs:|xsd:)?(?:import|include)\b[^>]*schemaLocation=[\"'](?P<name>[^\"']+)"),
            _rx(r"^\s*<Import\b[^>]*Project=[\"'](?P<name>[^\"']+)"),
            _rx(r"^\s*<\?xml-stylesheet[^>]*href=[\"'](?P<name>[^\"']+)"),
            _rx(r"^\s*<!DOCTYPE\s+(?P<name>[^\s>\[]+)"),
        ),
        decl_patterns=(
            # Identity-carrying elements are what you navigate an XML file
            # by (a .csproj PackageReference, a spring bean, a plist key).
            DeclPattern("blocks", _rx(r"^\s*<(?P<name>[A-Za-z_][\w:.-]*)\b[^>]*\s(?:id|name|key|Include)=[\"'](?P<params>[^\"']*)"), block=False),
            # Everything else: only elements near the TOP of the tree, so a
            # deeply nested document does not flood the outline with leaves.
            DeclPattern("blocks", _rx(r"^[ \t]{0,4}<(?P<name>[A-Za-z_][\w:.-]*)"), block=False),
        ),
        block_style="none",
        c_block_comments=False,
        skip_regions=(SkipRegion(open=_rx(r"<!--"), close=_rx(r"-->")),),
        notes="XML outline lists elements carrying an id/name/key/Include attribute plus elements indented by at most 4 characters — a depth cut-off that tracks the file's own indent unit, not a tree level; repeated sibling elements list individually. <!-- --> regions are skipped. `.conf` is claimed only when the file opens with an XML declaration or DOCTYPE.",
    ),
    LanguageSpec(
        name="graphql",
        aliases=("gql",),
        extensions=(".graphql", ".gql"),
        line_comment=("#",),
        decl_patterns=(
            DeclPattern("types", _rx(r"^\s*(?:extend\s+)?(?:type|input|interface|enum|union|scalar|schema)\s+(?P<name>\w+)")),
            DeclPattern("blocks", _rx(r"^\s*(?P<name>(?:query|mutation|subscription|fragment)\s+\w+)")),
            DeclPattern("blocks", _rx(r"^\s*(?P<name>directive\s+@\w+)"), block=False),
        ),
        c_block_comments=False,
        quote_chars='"',
        notes="GraphQL outline lists type/input/interface/enum/union definitions, named operations and fragments; brace-less forms (`scalar`, `union`) carry a line number only.",
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

# How much of a file's head a contested-extension sniff may read.
SNIFF_BYTES = 65536
_SNIFF_EXT_TO_SPECS: Dict[str, Tuple[LanguageSpec, ...]] = {}
for _spec in _SPECS:
    for _e in _spec.sniff_extensions:
        _SNIFF_EXT_TO_SPECS[_e] = _SNIFF_EXT_TO_SPECS.get(_e, ()) + (_spec,)


def known_language_names() -> List[str]:
    """Every language the ENGINE covers (the legacy lanes add their own)."""
    return sorted({s.name for s in _SPECS})


def spec_for(
    language: Optional[str] = None,
    path: Optional[Path] = None,
    first_line: str = "",
    text: str = "",
) -> Optional[LanguageSpec]:
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
        # Contested extension: claim it only on positive evidence, and fall
        # through to the generic lane otherwise rather than mislabel.
        for cand in _SNIFF_EXT_TO_SPECS.get(path.suffix.lower(), ()):
            if cand.sniff is not None and text and cand.sniff.search(text[:SNIFF_BYTES]):
                return cand
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

    def __init__(
        self,
        lines: List[str],
        markers: Tuple[str, ...],
        *,
        c_block_comments: bool,
        quote_chars: str,
        skip_regions: Tuple[SkipRegion, ...] = (),
    ) -> None:
        self._lines = lines
        self._markers = markers
        # This pass used to see NO skip regions at all: braces inside a
        # heredoc body or a `<# … #>` comment moved every extent below them.
        skipper = SkipRegionTracker(skip_regions, markers)
        # line index of first open event -> 1-based close line of THAT brace
        self._close_for_open_line: Dict[int, int] = {}
        # line index -> True if any open event occurs on it
        self._opens_on_line: Dict[int, bool] = {}
        stack: List[int] = []  # line indices of unmatched opens
        in_block_comment = False
        for j, raw in enumerate(lines):
            code = skipper.feed(raw)
            if code is None:
                continue
            raw = code
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

    Same P0-2 fix as BraceExtentIndex, keyword-block edition (reviewer C
    measured 2000 bodyless `def`s at 19.3s under the per-decl scan). Reviewer
    A (F3) hardening carried over: one-line `def x; ...; end` self-closes;
    assigned blocks (`label = case ... end`) open; heredoc bodies and
    `=begin/=end` comment blocks are skipped.

    Every shape here used to be a hardcoded ruby-ism, which is why lua and
    elixir could not be added as DATA (the module's rule 1). They are now
    spec fields: ruby's data reproduces the old regexes exactly, and a new
    keyword-block language stays a table entry.
    """

    def __init__(self, lines: List[str], spec: LanguageSpec) -> None:
        opener_re = (
            re.compile(r"^\s*(?:" + "|".join(re.escape(w) for w in spec.end_block_openers) + r")\b")
            if spec.end_block_openers
            else None
        )
        assigned_opener_re = (
            re.compile(r"=\s*(?:" + "|".join(re.escape(w) for w in spec.end_assigned_openers) + r")\b")
            if spec.end_assigned_openers
            else None
        )
        inline_block_re = spec.end_inline_block_re
        inline_close_re = spec.end_inline_close_re
        bodyless_re = spec.end_bodyless_re
        needs_terminator = spec.end_openers_need_terminator
        end_re = re.compile(r"^\s*end\b")
        self._close_for_line: Dict[int, int] = {}
        stack: List[int] = []
        skipper = SkipRegionTracker(spec.skip_regions, spec.line_comment)
        # Line index of a declaration head seen but not yet opened.
        pending_head: Optional[int] = None
        for j, raw in enumerate(lines):
            region_code = skipper.feed(raw)
            if region_code is None:
                continue
            code = _strip_line_comment(region_code, spec.line_comment).rstrip()
            stripped = code.strip()
            if not stripped:
                continue
            # `def foo, do: :ok` is opener-SHAPED with no block at all.
            # Pushing it would hand the next real `end` to it, and EVERY
            # range below would be wrong — worse than listing no range.
            # When it lands on a LATER line (`for x <- list,` … `do: x`)
            # it also cancels the head still waiting to open.
            if bodyless_re is not None and bodyless_re.search(code):
                pending_head = None
                continue
            opens_here = opener_re is not None and opener_re.match(code) is not None
            terminates = inline_block_re is not None and inline_block_re.search(code) is not None
            if opens_here:
                # A new declaration abandons any incomplete head above it.
                pending_head = None
                if needs_terminator and not terminates:
                    pending_head = j
                    continue
                if inline_close_re is not None and inline_close_re.search(code):
                    # One-line `def name; body; end` self-closes.
                    self._close_for_line[j] = j + 1
                else:
                    stack.append(j)
            elif pending_head is not None and terminates:
                # The wrapped head finally opened: the block belongs to the
                # line the DECLARATION is on, not to this continuation.
                stack.append(pending_head)
                pending_head = None
            elif (assigned_opener_re is not None and assigned_opener_re.search(code)) or terminates:
                if inline_close_re is not None and inline_close_re.search(code):
                    self._close_for_line[j] = j + 1
                else:
                    stack.append(j)
            if end_re.match(code) is not None and stack:
                open_line = stack.pop()
                self._close_for_line[open_line] = j + 1

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
    # A trailing newline terminates the last line; it does not start a new
    # one. Keeping the empty string after it made `lines=N` one MORE than
    # read_file reports for the same file — for essentially every file, since
    # almost all end in a newline — which is precisely the drift this helper
    # exists to prevent.
    if lines and lines[-1] == "":
        lines.pop()
    return [l[:-1] if l.endswith("\r") else l for l in lines]


def _scan_generic_delimiters(
    lines: List[str],
    markers: Tuple[str, ...],
    *,
    c_block_comments: bool = True,
    balance_pairs: str = "{}()[]",
    skip_regions: Tuple[SkipRegion, ...] = (),
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
    skipper = SkipRegionTracker(skip_regions, markers)
    for raw in lines:
        code = skipper.feed(raw)
        if code is None:
            continue
        raw = code
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
    "methods",
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


def _entry_line(entry: str) -> Optional[int]:
    """The first line number an entry refers to, whatever its section shape.

    Sections render differently — `  - 12-40: name` in the engine lanes,
    `  - calls: a -> b (line 12)` in the python lane — and some (javascript
    `refs`) carry no line at all. The recovery hint has to work for all of
    them, so it asks here and degrades when the answer is None.
    """
    # The digits must sit in a POSITION that means "line": `- 12: name`,
    # `- 12-40: name`, `- line 12: err`, or a trailing `(line 12)`. A bare
    # `- 9000` is a JSON KEY, and reading it as a line produced a confident
    # `start_line=9500` for a 703-line file. Sections whose entries lead with
    # DATA must pass `entry_lines` rather than rely on this.
    m = re.match(r"^\s*-\s*(?:line\s+)?(\d+)(?:-\d+)?:", entry)
    if m:
        return int(m.group(1))
    m = re.search(r"\(line (\d+)\)", entry)
    return int(m.group(1)) if m else None


def _emit_section(
    out: List[str],
    label: str,
    entries: List[str],
    *,
    total: Optional[int] = None,
    path: str = "",
    lines_total: int = 0,
    log: Optional["TruncationLog"] = None,
    entry_lines: Optional[List[Optional[int]]] = None,
) -> None:
    """Print up to MAX_SECTION_ENTRIES, and report the REAL remainder.

    `total` is the true number of matches found, which may exceed `len(entries)`
    when the caller capped what it collected. Without it this function reports
    `len(entries) - MAX_SECTION_ENTRIES`, which is only correct for callers that
    collect everything.

    THE BUG THIS FIXES. The TODO collector capped at MAX_SECTION_ENTRIES + 10, so
    the remainder computed here was ALWAYS exactly 10 — for 61 matches and for
    5000 alike. A file with 500 TODOs reported "todo_markers=60" and "(10 more)",
    hiding 440 and stating two false numbers. Worse than a silent cap: the model
    was handed a bounded, closed problem ("10 missing, go fetch them") and had no
    reason to re-query, so any judgement about the file was off by an order of
    magnitude. Every other section here collects in full and was always honest;
    this function was correct and was being fed a pre-truncated list.
    """
    # The section cap is the first limit; the output budget is the backstop.
    cap = MAX_SECTION_ENTRIES if log is None else min(MAX_SECTION_ENTRIES, log.allowance(entries))
    shown = entries[:cap]
    if log is not None:
        log.spend(shown)
    out.append(f"{label}:" if entries else f"{label}: []")
    out.extend(shown)
    real_total = len(entries) if total is None else int(total)
    hidden = real_total - cap
    if hidden <= 0:
        return
    # The recovery must be something the model can RUN, and a PRECISE hint
    # that is wrong is worse than a vague one — the model has no reason to
    # doubt it. So the range is derived defensively:
    #
    #  * over the ACTUAL hidden slice (min/max, not first/last), because not
    #    every section is emitted in line order — the python lane walks
    #    classes before functions;
    #  * validated against the file's real length, because `_entry_line`
    #    reads a leading integer and some sections lead with DATA (a JSON
    #    file keyed "9000".. produced start_line=9500 for a 703-line file);
    #  * clamped to read_file's own per-call budget, or the call it suggests
    #    is refused outright.
    if entry_lines is not None:
        # Authoritative: the collector recorded these, no sniffing needed.
        hidden_lines = [n for n in entry_lines[cap:] if n is not None]
    else:
        hidden_lines = [n for n in (_entry_line(e) for e in entries[cap:]) if n is not None]
    if lines_total:
        hidden_lines = [n for n in hidden_lines if 0 < n <= lines_total]
    target = f'file_path="{path}"' if path else "this file"
    if hidden_lines:
        first_hidden, last_hidden = min(hidden_lines), max(hidden_lines)
        stop = min(last_hidden, first_hidden + READ_FILE_LINE_BUDGET - 1)
        more = f", then from {stop + 1}" if stop < last_hidden else ""
        recovery = f"read_file({target}, start_line={first_hidden}, end_line={stop}){more}"
    else:
        recovery = (
            f'no line anchors in this section — search_files(pattern="<name>", {target})'
        )
    if log is not None:
        # Detail rides in the trailing block; the inline marker stays short so
        # a long outline is not padded with a repeated sentence per section.
        log.add(label, recovery, shown=cap, total=real_total)
        out.append(f"  - ... ({hidden} more) #TRUNCATION")
    else:
        out.append(
            f"  - ... ({hidden} more) #TRUNCATION — {hidden} further {label} entries exist "
            f"and are NOT listed above: {recovery}."
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

    log = TruncationLog()
    out: List[str] = [
        f"Code Analysis: {display_path} (language={spec.name}, lines={total_lines})",
        ANALYZE_CODE_NEXT_STEP_HINT,
        f"language: {spec.name}",
    ]
    if truncated:
        log.add(
            f"file body past line {total_lines} (only the first {MAX_ANALYZE_BYTES // (1024 * 1024)} MB was read)",
            f'read_file(file_path="{display_path}", start_line={total_lines + 1})',
        )
    if encoding_note:
        out.append(f"notice: {encoding_note}")

    # JSON gets a real validity check instead of pattern scanning — and it
    # dispatches BEFORE the minified guard: json.loads is not line-anchored,
    # so a compact package-lock.json or a ledger JSONL with long records
    # still deserves its parse lane (reviewer B, P1-1).
    if spec.name == "json":
        return _analyze_json(out, path, display_path, text, lines, log)

    # Minified/generated guard: line-anchored outlining is meaningless.
    longest = max((len(l) for l in lines), default=0)
    if longest > MAX_LINE_CHARS_FOR_OUTLINE:
        # This skips the ENTIRE outline, which is the largest omission the
        # tool can make — it must carry the same marker a section cap does,
        # or a model scanning for #TRUNCATION concludes nothing was withheld.
        out.append(
            f"diagnostics: longest_line={longest} chars — file looks generated/minified; line-anchored outline skipped."
        )
        log.add(
            "all declarations (generated/minified file, no line-anchored outline)",
            f'search_files(pattern="<name>", file_path="{display_path}")',
        )
        out.append("notes: use search_files() for targeted lookups in generated files.")
        out.extend(log.render())
        return "\n".join(out)

    imports: List[str] = []
    sections: Dict[str, List[str]] = {}
    todos: List[str] = []
    elided_entries = 0
    # Every TODO seen, not just the ones kept for printing. The section cap is a
    # display limit; it must never become the reported count.
    todo_total = 0
    todo_re = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]?(.*)")

    # Cross-line skip state (reviewer A: heredoc bodies, markdown fences and
    # `#if 0` blocks are DATA, not declarations — a shell heredoc containing
    # `inner() {` used to be outlined as a live function). The same tracker
    # drives both extent indexes and the balance lint, so all four passes
    # agree on what is code.
    skipper = SkipRegionTracker(spec.skip_regions, spec.line_comment)

    abort_patterns = tuple(d.pattern for d in spec.decl_patterns)

    # One-pass extent indexes (reviewer C, P0-2: per-decl forward scans were
    # quadratic when braces never close — 4000 unclosed decls took 91s).
    brace_index = (
        BraceExtentIndex(
            lines,
            spec.line_comment,
            c_block_comments=spec.c_block_comments,
            quote_chars=spec.quote_chars,
            skip_regions=spec.skip_regions,
        )
        if spec.block_style == "brace"
        else None
    )
    end_index = EndKeywordExtentIndex(lines, spec) if spec.block_style == "end" else None

    for idx, raw in enumerate(lines):
        line_no = idx + 1

        region_code = skipper.feed(raw)

        # TODO markers ride the RAW line and live in COMMENTS, so this scan
        # must run before the skip: a `TODO:` inside ruby's `=begin` block or
        # an XML `<!-- TODO … -->` is exactly where such a marker belongs,
        # and skipping first silently dropped every one of them.
        if region_code is not None or skipper.inside_comment:
            m_todo = todo_re.search(raw)
            if m_todo:
                # Collect them ALL and cap at PRINT time, like every other
                # section. A pre-truncated list once made both the count and
                # the remainder wrong, and it cannot say where the hidden
                # entries live — which the recovery hint now needs.
                todo_total += 1
                body, cut = _elide(m_todo.group(2).strip())
                elided_entries += cut
                todos.append(f"  - {line_no}: {m_todo.group(1)} {body}".rstrip())

        if region_code is None:
            continue

        code = _strip_line_comment(region_code, spec.line_comment)
        stripped = code.strip()

        if not stripped:
            continue

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
            if params:
                params, cut = _elide(params)
                elided_entries += cut
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
            skip_regions=spec.skip_regions,
            quote_chars=spec.quote_chars,
        )
        if spec.block_style == "brace"
        else []
    )

    shown_elided = sum(
        1
        for kind in sections
        for entry in sections[kind][:MAX_SECTION_ENTRIES]
        if entry.endswith("…)") or entry.endswith("…")
    ) + sum(1 for entry in todos[:MAX_SECTION_ENTRIES] if entry.endswith("…"))
    if shown_elided:
        log.add(
            f"{shown_elided} over-long entry text(s), shortened with '…'",
            "read_file on the line number shown beside each one",
        )
    if skipper.unterminated_at:
        out.append(
            f"notice: #TRUNCATION an unterminated comment/string region opens at line "
            f"{skipper.unterminated_at}; declarations after it were read as data, not code."
        )

    diagnostics: List[str] = []
    if spec.block_style == "brace":
        diagnostics.append("delimiters=ok" if not delimiter_issues else f"delimiters={len(delimiter_issues)} issue(s)")
    if todos:
        diagnostics.append(f"todo_markers={todo_total}")
    out.append("diagnostics: " + ("; ".join(diagnostics) if diagnostics else "none"))

    summary_bits = [f"imports={len(imports)}"] + [f"{k}={len(v)}" for k, v in sections.items()]
    out.append("summary: " + "; ".join(summary_bits))

    if spec.block_style == "brace":
        if delimiter_issues:
            out.append("lint:")
            out.extend(delimiter_issues)
        else:
            out.append("lint: []")

    _emit_section(out, "imports", imports, path=display_path, lines_total=total_lines, log=log)
    # NOTE: every DeclPattern.kind used by any LanguageSpec must appear here
    # or its section silently vanishes from the output (truth bug) — pinned
    # by tests against the live spec table.
    for kind in _EMIT_KIND_ORDER:
        if kind in sections:
            _emit_section(out, kind, sections[kind], path=display_path, lines_total=total_lines, log=log)
    if todos:
        _emit_section(out, "todo_markers", todos, total=todo_total, path=display_path, lines_total=total_lines, log=log)

    if spec.notes:
        out.append(f"notes: {spec.notes}")
    out.extend(log.render())
    return "\n".join(out)


def _analyze_json(
    out: List[str], path: Path, display_path: str, text: str, lines: List[str], log: "TruncationLog"
) -> str:
    import json as _json

    is_jsonl = path.suffix.lower() in {".jsonl", ".ndjson"}
    if is_jsonl:
        bad: List[str] = []
        n_records = 0
        # COUNT every failure, COLLECT for printing. Conflating the two is the
        # bug _emit_section's docstring describes: `bad` was capped at 10, and
        # `invalid=len(bad)` then reported a 500-record file in which EVERY
        # record was malformed as `invalid=10` — a file 100% broken described
        # as 98% healthy, with no marker and no way to tell.
        n_invalid = 0
        for i, raw in enumerate(lines, 1):
            s = raw.strip()
            if not s:
                continue
            n_records += 1
            try:
                _json.loads(s)
            except Exception as e:
                n_invalid += 1
                bad.append(f"  - line {i}: {e}")
        out.append(f"diagnostics: jsonl_records={n_records}; invalid={n_invalid}")
        out.append(
            "summary: "
            + (f"{n_records} records, {n_invalid} invalid" if n_invalid else f"{n_records} records, all parse")
        )
        if bad:
            _emit_section(out, "invalid_lines", bad, total=n_invalid, path=display_path, lines_total=len(lines), log=log)
        out.extend(log.render())
        return "\n".join(out)

    try:
        doc = _json.loads(text)
    except Exception as e:
        out.append(f"diagnostics: parse=error ({e})")
        out.extend(log.render())
        return "\n".join(out)
    out.append("diagnostics: parse=ok")
    if isinstance(doc, dict):
        keys = list(doc.keys())
        out.append(f"summary: object with {len(keys)} top-level key(s)")
        _emit_section(out, "keys", [f"  - {k}" for k in keys], path=display_path, lines_total=len(lines), log=log)
    elif isinstance(doc, list):
        out.append(f"summary: array with {len(doc)} element(s)")
    else:
        out.append(f"summary: top-level {type(doc).__name__}")
    out.extend(log.render())
    return "\n".join(out)


def analyze_generic(
    display_path: str, text: str, *, language_hint: str = "", truncated: bool = False, encoding_note: str = ""
) -> str:
    """Never-refuse fallback: an honest structural sample for unknown text."""
    lines = split_lines_like_read_file(text)
    total_lines = len(lines)
    log = TruncationLog()
    out: List[str] = [
        f"Code Analysis: {display_path} (language=unknown, lines={total_lines})",
        "notice: language not recognized — this is a GENERIC text outline (structure sample + metrics), not a parsed code outline.",
    ]
    if language_hint:
        out.append(
            f"notice: requested language '{language_hint}' is not in the analyzer's vocabulary; falling back to the generic outline. #FALLBACK"
        )
    if truncated:
        log.add(
            f"file body past line {total_lines} (only the first {MAX_ANALYZE_BYTES // (1024 * 1024)} MB was read)",
            f'read_file(file_path="{display_path}", start_line={total_lines + 1})',
        )
    if encoding_note:
        out.append(f"notice: {encoding_note}")

    longest = max((len(l) for l in lines), default=0)
    non_blank = sum(1 for l in lines if l.strip())
    out.append(f"metrics: lines={total_lines}; non_blank={non_blank}; longest_line={longest} chars")

    if longest > MAX_LINE_CHARS_FOR_OUTLINE:
        out.append("notes: file looks generated/minified; use search_files() for targeted lookups.")
        log.add(
            "all structure (generated/minified file, no line-anchored outline)",
            f'search_files(pattern="<name>", file_path="{display_path}")',
        )
        out.extend(log.render())
        return "\n".join(out)

    # Structure sample: column-0 lines that look like section anchors.
    anchors: List[str] = []
    generic_elided = 0
    for i, raw in enumerate(lines, 1):
        if not raw or raw[0].isspace():
            continue
        s = raw.strip()
        if len(s) < 3:
            continue
        text_, cut = _elide(s)
        generic_elided += cut
        anchors.append(f"  - {i}: {text_}")
    _emit_section(out, "top_level_lines", anchors, path=display_path, lines_total=total_lines, log=log)

    todo_re = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s]?(.*)")
    todos = [
        f"  - {i}: {m.group(1)} {m.group(2).strip()}".rstrip()
        for i, raw in enumerate(lines, 1)
        if (m := todo_re.search(raw))
    ]
    if todos:
        _emit_section(out, "todo_markers", todos, path=display_path, lines_total=total_lines, log=log)

    if generic_elided:
        log.add(
            f"{generic_elided} over-long line(s), shortened with '…'",
            "read_file on the line number shown beside each one",
        )
    out.append("notes: pass language=<name> to force a specific analyzer; supported names are listed in the tool description.")
    out.extend(log.render())
    return "\n".join(out)
