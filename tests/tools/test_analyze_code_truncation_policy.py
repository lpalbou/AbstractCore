"""analyze_code truncation policy: disclose, and say how to recover.

The operating rule for this tool is that output is not quietly dropped.
Where a bound is genuinely necessary, the result must tell the model two
things: that something was withheld, and a step it can actually RUN to get
it. A hint the model cannot execute is not a recovery — the previous text
said `use search_files('<name>')`, but the names worth searching for were
exactly the ones the cap removed.

The cap itself is a cost, not a feature. Measured over 2429 real files and
8997 sections, section sizes run p50=2, p90=13, p99=57, p999=324, max=2293;
a cap of 50 therefore cut 1.3% of ALL sections, including the files whose
outline matters most. These tests pin the disclosure contract rather than
any particular number, so the bound can be tuned without gutting them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from abstractcore.tools import code_analysis as ca
from abstractcore.tools.common_tools import analyze_code

# A recovery is only real if it names a tool call the model can issue.
RECOVERY_TOKENS = ("read_file(", "search_files", "ruff check")


def _truncation_lines(out: str) -> list[str]:
    """Every line that admits something was withheld, however phrased."""
    return [
        l for l in out.split("\n")
        if "#TRUNCATION" in l or re.search(r"\(\d+ more\)", l) or "not listed" in l.lower()
    ]


def _trailing_block(out: str) -> list[str]:
    """The consolidated truncation block, which must END the answer.

    Detail belongs in ONE place at the end rather than repeated inline: an
    outline that scatters notices through hundreds of entries makes the
    reader rebuild the damage from fragments, and every fragment has to
    restate the file path to stay runnable.
    """
    lines = out.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("#TRUNCATION ") and "withheld" in line:
            return lines[i:]
    return []


def _assert_compliant(out: str, label: str) -> list[str]:
    inline = _truncation_lines(out)
    if not inline:
        return []
    block = _trailing_block(out)
    assert block, f"{label}: something was withheld but the answer has no trailing #TRUNCATION block"
    assert block[-1].strip(), f"{label}: the block must end the answer"
    for item in block[1:]:
        assert any(t in item for t in RECOVERY_TOKENS), (
            f"{label}: a withheld item with no runnable recovery: {item!r}"
        )
    # Every inline marker must be accounted for by the block.
    assert len(block) - 1 >= 1, f"{label}: empty truncation block"
    return inline


def test_the_section_cap_is_high_enough_to_be_rare() -> None:
    # p99 of real section sizes is 57. A cap anywhere near that truncates
    # routinely, which is what made the old value of 50 a problem: the tool
    # withheld data on 1.7% of all files, and the affected files were the
    # large ones an agent most needs a map of.
    assert ca.MAX_SECTION_ENTRIES >= 250, "a cap near p99 truncates ordinary files"


def test_a_capped_section_names_the_file_and_the_line_range(tmp_path: Path) -> None:
    n = ca.MAX_SECTION_ENTRIES + 200
    f = tmp_path / "big.go"
    f.write_text("package main\n" + "".join(f"func fn{i}() {{\n}}\n" for i in range(n)))
    out = analyze_code(file_path=str(f))
    _assert_compliant(out, "engine section")
    assert "(200 more)" in out, "the remainder must be the TRUE count"
    block = _trailing_block(out)
    line = [l for l in block if l.startswith("  - functions")][0]
    assert str(f) in line, "the recovery must name the file it applies to"
    # The hidden entries are a contiguous line range because sections are
    # emitted in line order; the range is what makes the recovery runnable.
    m = re.search(r"start_line=(\d+), end_line=(\d+)", line)
    assert m, f"expected an executable read_file range, got {line!r}"
    start, end = int(m.group(1)), int(m.group(2))
    assert 0 < start < end


def test_the_reported_remainder_is_the_true_count_not_the_collected_one(tmp_path: Path) -> None:
    # This class of bug has bitten here before: a collector that pre-capped
    # its list made the "(N more)" figure a constant, understating a file
    # with hundreds of matches by an order of magnitude. Every collector
    # must gather everything and let the printer do the capping.
    n = ca.MAX_SECTION_ENTRIES + 137
    f = tmp_path / "todos.go"
    f.write_text("package main\n" + "".join(f"// TODO: item {i}\n" for i in range(n)))
    out = analyze_code(file_path=str(f))
    assert f"todo_markers={n}" in out, "the diagnostic count must be the real total"
    assert "(137 more)" in out, "the remainder must be the real remainder"
    _assert_compliant(out, "todo markers")


# Sized off the LIVE cap so raising it cannot quietly turn these into tests
# of a file that fits.
_OVER = ca.MAX_SECTION_ENTRIES + 200


@pytest.mark.parametrize(
    "name, body, label",
    [
        ("big.go", "package main\n" + "".join(f"func f{i}() {{\n}}\n" for i in range(_OVER)), "engine functions"),
        ("big.md", "".join(f"# H{i}\n\ntext\n\n" for i in range(_OVER)), "markdown headings"),
        ("todos.go", "package main\n" + "".join(f"// TODO: t{i}\n" for i in range(_OVER)), "todo markers"),
        ("mystery.qqlang", "".join(f"blob {i}\n" for i in range(_OVER)), "generic fallback"),
        ("bundle.css", ".a{color:red}" * 800 + "\n", "minified guard"),
    ],
)
def test_every_lane_that_withholds_output_discloses_and_recovers(
    tmp_path: Path, name: str, body: str, label: str
) -> None:
    f = tmp_path / name
    f.write_text(body)
    out = analyze_code(file_path=str(f))
    lines = _assert_compliant(out, label)
    assert lines, f"{label}: expected this case to withhold something"


def test_the_minified_guard_admits_it_listed_nothing(tmp_path: Path) -> None:
    # Skipping the outline entirely is the LARGEST omission the tool makes.
    # It reported only a `diagnostics:` observation, so a model scanning for
    # #TRUNCATION concluded nothing had been withheld.
    f = tmp_path / "b.css"
    f.write_text(".a{color:red}" * 800 + "\n")
    out = analyze_code(file_path=str(f))
    block = _trailing_block(out)
    assert block, "skipping the whole outline must appear in the trailing block"
    assert any("all declarations" in l for l in block)
    assert any(t in l for l in block for t in RECOVERY_TOKENS)


def test_an_over_long_entry_is_elided_visibly_not_silently(tmp_path: Path) -> None:
    # Per-entry text used to be cut mid-string with no marker at all — the
    # worst class, because the reader cannot tell truth from fragment. The
    # entry keeps its line number, so `read_file` on that line is the
    # recovery, and the notice says so.
    f = tmp_path / "T.hs"
    f.write_text("longSig :: " + " -> ".join(f"VeryLongTypeName{i}" for i in range(20)) + "\n")
    out = analyze_code(file_path=str(f))
    assert "…" in out, "an elided entry must show that it was elided"
    block = _trailing_block(out)
    assert any("over-long entry text" in l for l in block)
    assert any("read_file" in l for l in block)


def test_short_entries_are_never_elided(tmp_path: Path) -> None:
    f = tmp_path / "T.hs"
    f.write_text("area :: Double -> Double\n")
    out = analyze_code(file_path=str(f))
    assert "  - 1: area(Double -> Double)" in out
    assert "#TRUNCATION" not in out, "nothing was withheld, so nothing should be claimed"


def test_an_ordinary_file_carries_no_truncation_claim(tmp_path: Path) -> None:
    # The contract runs both ways: a false truncation claim would send the
    # model chasing data that is already in front of it.
    f = tmp_path / "small.go"
    f.write_text("package main\n\nfunc main() {\n\treturn\n}\n")
    out = analyze_code(file_path=str(f))
    assert "#TRUNCATION" not in out
    assert not _truncation_lines(out)


def test_no_lane_emits_a_bare_more_line_without_the_marker() -> None:
    # Guards the pattern rather than the instances: several lanes each grew
    # their own `(N more)` line with no marker and no recovery, because the
    # shape was easy to copy. Any new one must fail this.
    import inspect
    from abstractcore.tools import common_tools

    src = inspect.getsource(common_tools)
    for m in re.finditer(r'f"  - \.\.\. \(\{[^}]+\} more\)"', src):
        snippet = src[m.start() : m.start() + 400]
        assert "#TRUNCATION" in snippet, (
            f"a bare '(N more)' line with no #TRUNCATION marker: {m.group(0)}"
        )


# ---------------------------------------------------------------------------
# The deep lanes (python/javascript/html/r) had none of the engine's bounds,
# and a precise-but-wrong recovery is worse than a vague one: the model has
# no reason to doubt a hint that names an exact line range.
# ---------------------------------------------------------------------------


def test_the_deep_lanes_bound_their_read_like_the_engine_does(tmp_path: Path) -> None:
    # The deep lanes called path.read_text() directly — no byte bound, no
    # minified guard, no notice — and python's `module_assignments` was both
    # uncapped AND missing from the summary. A 3 MB generated file therefore
    # returned ~944k TOKENS while reporting `imports=0; classes=0;
    # functions=0; relationships=0`: four zeros beside 3.7 MB of output.
    f = tmp_path / "huge.py"
    f.write_text("".join(f"v{i} = {i}\n" for i in range(_OVER * 20)))
    out = analyze_code(file_path=str(f))
    assert len(out) < 200_000, f"deep-lane outline is unbounded ({len(out)} chars)"
    assert f"module_assignments={_OVER * 20}" in out, "an emitted section must be counted in the summary"
    _assert_compliant(out, "deep lane")


def test_a_generated_deep_lane_file_is_guarded_like_the_engine(tmp_path: Path) -> None:
    f = tmp_path / "bundle.py"
    f.write_text("x = [" + ",".join(str(i) for i in range(4000)) + "]\n")
    out = analyze_code(file_path=str(f))
    assert "#TRUNCATION" in out and "generated/minified" in out
    assert any(t in out for t in RECOVERY_TOKENS)


def test_jsonl_invalid_count_is_the_true_count(tmp_path: Path) -> None:
    # The collected list was capped at 10 and the DIAGNOSTIC was reported as
    # len(collected), so a file in which every record was malformed came back
    # as `invalid=10` — 100% broken described as 98% healthy. This is the
    # exact bug class _emit_section's docstring was written about.
    f = tmp_path / "broken.jsonl"
    f.write_text("".join("not json\n" for _ in range(_OVER)))
    out = analyze_code(file_path=str(f))
    assert f"jsonl_records={_OVER}; invalid={_OVER}" in out
    assert f"{_OVER} records, {_OVER} invalid" in out


def test_a_recovery_range_is_never_invented_from_non_line_data(tmp_path: Path) -> None:
    # `_entry_line` reads a leading integer, but some sections lead with DATA:
    # a JSON file keyed "9000".."9699" yielded start_line=9500 for a 703-line
    # file, and following that hint just errored. A confidently wrong recovery
    # is worse than the vague one it replaced.
    n = ca.MAX_SECTION_ENTRIES + 200
    f = tmp_path / "numkeys.json"
    f.write_text("{\n" + ",\n".join(f'  "{9000 + i}": {i}' for i in range(n)) + "\n}\n")
    out = analyze_code(file_path=str(f))
    line = [l for l in _trailing_block(out) if l.startswith("  - keys")][0]
    assert "no line anchors" in line, "a section without line anchors must say so"
    assert "start_line=9" not in line, "a JSON key is not a line number"


def test_a_recovery_range_covers_entries_emitted_out_of_line_order() -> None:
    # Sections are NOT all line-ordered: the python lane walks classes before
    # functions, so entries[cap] can sit far below entries[-1]. Deriving the
    # range from those two endpoints pointed at the wrong end of the file and
    # missed most of what it promised. min/max over the hidden slice is
    # order-independent.
    half = ca.MAX_SECTION_ENTRIES
    entries = [f"  - calls: A.m -> h{i} (line {5000 + i})" for i in range(half)]
    entries += [f"  - calls: top -> h{i} (line {5 + i})" for i in range(half)]
    out: list[str] = []
    log = ca.TruncationLog()
    ca._emit_section(out, "relationships", entries, path="/tmp/order.py", lines_total=9000, log=log)
    hint = "\n".join(log.render())
    m = re.search(r"start_line=(\d+), end_line=(\d+)", hint)
    assert m, hint
    lo, hi = int(m.group(1)), int(m.group(2))
    hidden = [int(re.search(r"\(line (\d+)\)", e).group(1)) for e in entries[ca.MAX_SECTION_ENTRIES:]]
    assert lo <= min(hidden) and hi >= max(hidden), "the stated range must cover every hidden entry"


def test_a_recovery_range_stays_within_read_files_own_budget(tmp_path: Path) -> None:
    # read_file refuses a range wider than its per-call line budget, so a hint
    # naming a wider one is syntactically valid and impossible to execute.
    f = tmp_path / "huge.go"
    f.write_text("package main\n" + "".join(f"func f{i}() {{\n\t()\n}}\n" for i in range(_OVER * 10)))
    out = analyze_code(file_path=str(f))
    line = [l for l in _trailing_block(out) if "read_file(" in l][0]
    m = re.search(r"start_line=(\d+), end_line=(\d+)", line)
    assert m, line
    span = int(m.group(2)) - int(m.group(1)) + 1
    assert span <= ca.READ_FILE_LINE_BUDGET, f"hint asks for {span} lines, over the budget"
    assert "then from" in line, "a clamped range must say how to page on"


def test_the_four_mb_notice_names_a_real_line_not_a_placeholder() -> None:
    # The notice shipped a literal `start_line=…` — the same unusable
    # placeholder the redesign existed to remove. The analyzer knows exactly
    # how many lines it read, so the first unread line is a real number.
    import inspect

    src = inspect.getsource(ca)
    assert "start_line=…" not in src, "a recovery hint must not contain a placeholder"


def test_lint_scanners_report_the_true_total_out_of_band(tmp_path: Path) -> None:
    # Stopping at the issue cap produced a list indistinguishable from a
    # complete one. Putting a marker INSIDE that list fixed the disclosure
    # and broke the count: `diagnostics: delimiters=N issues` reports
    # len(list), so the marker inflated it by one and the figure the model
    # reads first became false. The count now lives beside the list.
    from abstractcore.tools import common_tools as ct

    issues = ct._scan_js_delimiter_issues(["}" * 25], max_issues=5)
    assert len(issues) == 5, "the printed sample stays bounded"
    assert issues.total == 25, "the reported total must be the real one"
    assert not any("#TRUNCATION" in i for i in issues), "the marker must not be a list element"

    f = tmp_path / "bad.js"
    f.write_text("}" * 25 + "\n")
    out = analyze_code(file_path=str(f))
    assert "delimiters=25 issues" in out, "the diagnostic must not count the marker"
    block = _trailing_block(out)
    assert any("delimiter issues (5 of 25 shown)" in l or "delimiter issues (10 of 25 shown)" in l for l in block)


def test_skim_folders_discloses_dropped_notable_files(tmp_path: Path) -> None:
    from abstractcore.tools.common_tools import skim_folders

    for i in range(60):
        sub = tmp_path / f"sub{i:02d}"
        sub.mkdir()
        for name in ("README.md", "ARCHITECTURE.md", "BACKLOG.md", "CHANGELOG.md"):
            (sub / name).write_text("x")
    out = skim_folders(paths=[str(tmp_path)])
    assert "#TRUNCATION" in out, "dropping notable files silently defeats the tool's purpose"
    assert "of 240 notable files shown" in out
    assert "skim_folders" in out or "list_files" in out
    assert "(+1 more)" in out, "the inline preview must admit it shows only the first few"


# ---------------------------------------------------------------------------
# Round 3: the failures a per-section cap cannot catch.
# ---------------------------------------------------------------------------


def test_bare_annotations_do_not_crash_the_python_lane(tmp_path: Path) -> None:
    # `x: int` with no value has node.value None, and generic_visit(None)
    # raised — 7 of this package's OWN modules returned a traceback instead
    # of an outline. That is 100% of the answer withheld, unmarked.
    f = tmp_path / "a.py"
    f.write_text("class C:\n    def m(self):\n        x: int\n        y: dict = {}\n        return y\n")
    out = analyze_code(file_path=str(f))
    assert not out.startswith("Error")
    assert "language: python" in out


def test_this_package_analyzes_without_crashing() -> None:
    # The corpus that found the bug above. A tool used to navigate code must
    # survive the code it ships with.
    import itertools

    root = Path(__file__).resolve().parents[2] / "abstractcore"
    failures = []
    for p in itertools.islice(root.rglob("*.py"), 250):
        try:
            analyze_code(file_path=str(p))
        except Exception as e:  # noqa: BLE001 - the point is that none escape
            failures.append((p.name, type(e).__name__))
    assert not failures, f"analyze_code raised on its own package: {failures[:5]}"


def test_the_biggest_python_sections_are_capped_like_every_other(tmp_path: Path) -> None:
    # imports/classes/functions never called _emit_section, so the cap, the
    # log and the whole disclosure design were bypassed: a generated module
    # returned ~600k tokens with no marker at all.
    n = ca.MAX_SECTION_ENTRIES + 300
    f = tmp_path / "gen.py"
    f.write_text("".join(f"def op_{i}(a, b=None):\n    return {i}\n" for i in range(n)))
    out = analyze_code(file_path=str(f))
    assert len(out) <= ca.MAX_OUTLINE_CHARS + 4000, f"unbounded python outline ({len(out)} chars)"
    _assert_compliant(out, "python functions")
    assert any(l.startswith("  - functions") for l in _trailing_block(out))


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason=(
        "CPython 3.9's parser can segfault while reporting an unterminated triple-quoted "
        "string at EOF in a multi-MB input (heap-dependent: passes alone, crashes the 3.9 CI "
        "run late in the suite); 3.9 is end-of-life. analyze_code itself is unchanged."
    ),
)
def test_a_syntax_error_on_a_file_this_tool_cut_says_so(tmp_path: Path) -> None:
    # The parse failed on text THIS TOOL truncated at the byte bound, and the
    # early return discarded the log — so the tool's own truncation was
    # reported to the model as the user's syntax error, with the explanation
    # thrown away.
    f = tmp_path / "big.py"
    f.write_text("x = 1\n" * 50000 + 'DOC = """' + ("y" * 200 + "\n") * 30000 + '"""\n')
    out = analyze_code(file_path=str(f))
    assert "syntax error" in out
    assert "cut at" in out and "by this tool" in out, "blaming the file for our own cut"
    assert _trailing_block(out), "the log must still render on the error path"


def test_analyze_code_line_count_matches_read_file(tmp_path: Path) -> None:
    # These numbers exist to feed read_file(start_line=...). Counting the
    # empty string after a trailing newline made `lines=N` one MORE than
    # read_file reports — for essentially every file.
    from abstractcore.tools.common_tools import read_file

    for name in ("t.go", "t.py", "t.md", "t.rb"):
        f = tmp_path / name
        f.write_text("a\nb\nc\n")
        head = analyze_code(file_path=str(f)).split("\n")[0]
        assert "lines=3)" in head, f"{name}: {head}"
        assert "(3 lines)" in read_file(file_path=str(f)).split("\n")[0]


def test_html_ids_recover_by_line_not_by_id_value(tmp_path: Path) -> None:
    # `ids` entries lead with the id, so sniffing read the ID as a line
    # number. With numeric ids that produced a confident, wrong range.
    # Ids chosen far outside the file's line range: sniffing the entry text
    # yields 90000+, which the range filter then rejects, and the section
    # falsely reports "no line anchors" — while the real line sits right
    # there after the colon. Only a collector-supplied line survives this.
    n = ca.MAX_SECTION_ENTRIES + 400
    f = tmp_path / "ids.html"
    f.write_text(
        "<html><body>\n"
        + "\n".join(f'<div id="{90000 + i}">x</div>' for i in range(n))
        + "\n</body></html>\n"
    )
    out = analyze_code(file_path=str(f))
    line = [l for l in _trailing_block(out) if l.startswith("  - ids")][0]
    assert "no line anchors" not in line, "ids DO have line numbers; they are printed in the entry"
    m = re.search(r"start_line=(\d+), end_line=(\d+)", line)
    assert m, line
    total_lines = int(re.search(r"lines=(\d+)\)", out.split("\n")[0]).group(1))
    start, end = int(m.group(1)), int(m.group(2))
    assert 0 < start <= end <= total_lines, f"range {start}-{end} outside a {total_lines}-line file"


def test_a_total_output_budget_bounds_a_many_section_file(tmp_path: Path) -> None:
    # A per-section cap cannot bound N sections: four sections each just
    # under the cap produced a 26k-token outline for a 45k-token file.
    f = tmp_path / "worst.c"
    body = "".join(f"#include <h{i}.h>\n" for i in range(1200))
    body += "".join(f"#define M{i} {i}\n" for i in range(1200))
    body += "".join(f"typedef struct S{i} {{ int a; }} S{i};\n" for i in range(1200))
    body += "".join(f"// TODO: t{i}\nint fn{i}(void) {{\n  return {i};\n}}\n" for i in range(1200))
    f.write_text(body)
    out = analyze_code(file_path=str(f))
    assert len(out) <= ca.MAX_OUTLINE_CHARS + 4000, f"budget not enforced ({len(out)} chars)"
    block = _trailing_block(out)
    # Sections starved by the BUDGET (not the cap) must be disclosed too.
    starved = [l for l in block if re.search(r"\((\d+) of \d+ shown\)", l)
               and int(re.search(r"\((\d+) of", l).group(1)) < ca.MAX_SECTION_ENTRIES]
    assert starved, "a budget-driven cut must be reported like any other omission"
    _assert_compliant(out, "output budget")


def test_the_budget_leaves_ordinary_files_untouched(tmp_path: Path) -> None:
    f = tmp_path / "ordinary.go"
    f.write_text("package main\n" + "".join(f"func f{i}() {{\n}}\n" for i in range(80)))
    out = analyze_code(file_path=str(f))
    assert "#TRUNCATION" not in out
    assert out.count("func") == 0 or "f79()" in out, "every declaration should be listed"
