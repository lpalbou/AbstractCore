from __future__ import annotations

from abstractcore.tools.common_tools import list_files, read_file, search_files


def test_list_files_empty_directory_reports_exists_but_empty(tmp_path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    out = list_files(directory_path=str(empty_dir), pattern="*")
    assert out == f"Directory '{empty_dir}' exists but is empty"


def test_list_files_hidden_only_directory_reports_hidden_entries(tmp_path) -> None:
    hidden_dir = tmp_path / "hidden_only"
    hidden_dir.mkdir()
    (hidden_dir / ".secret").write_text("x\n", encoding="utf-8")

    out = list_files(directory_path=str(hidden_dir), pattern="*")
    assert out == f"Directory '{hidden_dir}' exists but contains only hidden entries (use include_hidden=True)"


def test_list_files_truncation_note_suggests_increase_and_none(tmp_path) -> None:
    many = tmp_path / "many"
    many.mkdir()
    for i in range(40):
        (many / f"f{i:02d}.txt").write_text("x\n", encoding="utf-8")

    out = list_files(directory_path=str(many), pattern="*.txt", head_limit=10)
    assert "(showing 10 of 40 entries)" in out
    assert "Note: 30 more entries available" in out
    assert "increase head_limit to see more results" in out
    assert "set head_limit=None to show all results" in out
    assert "If you want to see more results, re-run: list_files(" in out
    assert "head_limit=20" in out


def test_read_file_inclusive_single_line_range_returns_line_number(tmp_path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        start_line=2,
        end_line=2,
    )

    assert out.startswith(f"File: {path} (1 lines)\n\n")
    assert "\n2: two\n" in out + "\n"


def test_read_file_inclusive_two_line_range_returns_both_lines(tmp_path) -> None:
    path = tmp_path / "demo2.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        start_line=2,
        end_line=3,
    )
    assert "\n2: b\n3: c\n" in out + "\n"


def test_read_file_preserves_trailing_spaces(tmp_path) -> None:
    path = tmp_path / "spaces.txt"
    path.write_text("a  \nb\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        start_line=1,
        end_line=1,
    )
    assert "1: a  " in out


def test_search_files_content_mode_line_prefix_is_line_number(tmp_path) -> None:
    path = tmp_path / "code.py"
    path.write_text("print('x')\n# TODO: fix\n# TODO: later\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.py", head_limit=None)
    assert f"\n📄 {path}:\n" in out
    assert "    2: # TODO: fix" in out
    assert "    3: # TODO: later" in out


def test_search_files_truncates_very_long_lines_and_keeps_match_visible(tmp_path) -> None:
    p = tmp_path / "one_line.txt"
    p.write_text("a" * 500 + "Maintenance Mode" + "b" * 600, encoding="utf-8")

    out = search_files("Maintenance Mode", path=str(tmp_path), file_pattern="*.txt", head_limit=1, max_hits=1)
    excerpt_lines = [ln for ln in out.splitlines() if ln.lstrip().startswith("1:")]
    assert excerpt_lines, f"expected a matching line excerpt, got:\n{out}"
    content = excerpt_lines[0].split(":", 1)[1].lstrip()
    assert len(content) <= 400
    assert "Maintenance Mode" in content
    assert "…" in content


def test_search_files_head_limit_is_per_file_not_global(tmp_path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("TODO a1\nTODO a2\nTODO a3\n", encoding="utf-8")
    b.write_text("TODO b1\nTODO b2\nTODO b3\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.txt", head_limit=2, max_hits=None)

    assert f"\n📄 {a}:\n" in out
    assert f"\n📄 {b}:\n" in out
    assert "    1: TODO a1" in out
    assert "    2: TODO a2" in out
    assert "TODO a3" not in out
    assert "    1: TODO b1" in out
    assert "    2: TODO b2" in out
    assert "TODO b3" not in out


def test_search_files_max_hits_limits_number_of_files(tmp_path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    a.write_text("TODO a\n", encoding="utf-8")
    b.write_text("TODO b\n", encoding="utf-8")
    c.write_text("TODO c\n", encoding="utf-8")

    out = search_files("TODO", path=str(tmp_path), file_pattern="*.txt", head_limit=1, max_hits=2)
    assert out.count("\n📄 ") == 2
    assert sum(int(str(p) in out) for p in (a, b, c)) == 2


def test_search_files_truncation_note_suggests_increase_and_none(tmp_path) -> None:
    for i in range(12):
        (tmp_path / f"m{i:02d}.txt").write_text("match\n", encoding="utf-8")

    out = search_files("match", path=str(tmp_path), file_pattern="*.txt", head_limit=1, max_hits=3)
    assert "(showing 3 of 12 matching files)" in out
    assert "Note: 9 more matching files available" in out
    assert "increase max_hits to see more results" in out
    assert "set max_hits=None to show all results" in out
    assert "If you want to see more results, re-run: search_files(" in out
    assert "max_hits=6" in out


def test_search_files_stops_per_file_work_at_max_hits_not_over_full_tree(tmp_path, monkeypatch) -> None:
    # search-perf incident 2026-07-23: the EXPENSIVE per-file work (is_ignored,
    # which resolve()d twice, + the 1KB binary sniff) must run only on files
    # the match loop REACHES — the old code did it eagerly on every file in the
    # tree before matching, so a 196k-file tree ran 8m39s. Count is_ignored
    # file-level calls as the direct measure: with max_hits + a bounded
    # remainder budget it stays far below the whole-tree count.
    for i in range(5):
        (tmp_path / f"hit{i}.txt").write_text("NEEDLE\n", encoding="utf-8")
    big = tmp_path / "tail"
    big.mkdir()
    for i in range(2000):
        (big / f"f{i:04d}.txt").write_text("NEEDLE\n", encoding="utf-8")

    from abstractcore.tools.abstractignore import AbstractIgnore

    real_is_ignored = AbstractIgnore.is_ignored
    file_calls = {"n": 0}

    def _counting_is_ignored(self, path, *, is_dir=None):
        if is_dir is False:
            file_calls["n"] += 1
        return real_is_ignored(self, path, is_dir=is_dir)

    monkeypatch.setattr(AbstractIgnore, "is_ignored", _counting_is_ignored)
    out = search_files("NEEDLE", path=str(tmp_path), file_pattern="*.txt", max_hits=3)
    assert "in 3 files" in out or "showing 3 of" in out
    # ~3 matched + up to the 500 remainder budget, never the full 2005-file tree.
    assert file_calls["n"] < 700, (
        f"is_ignored ran on {file_calls['n']} files — the per-file cost must stop near max_hits + the "
        "bounded remainder, not enumerate the whole tree (the 8m39s eager-enumeration regression)"
    )


def test_search_files_prunes_target_build_dir_by_default(tmp_path) -> None:
    # search-perf incident: target/ (Rust/JVM build tree) is now a default
    # ignore, the twin of node_modules/dist/build.
    (tmp_path / "src.rs").write_text("fn main() { LANDMARK }\n", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    (target / "artifact.rs").write_text("LANDMARK in build output\n", encoding="utf-8")

    out = search_files("LANDMARK", path=str(tmp_path))
    assert "src.rs" in out
    assert "target" not in out and "artifact.rs" not in out, "target/ must be pruned by default"


def test_list_files_stops_per_entry_work_at_head_limit_over_huge_tree(tmp_path, monkeypatch) -> None:
    # list-perf incident 2026-07-23 (operator dm 17:56): list_files(head_limit=100)
    # over a 130k-file tree walked + is_ignored'd + mtime-STAT-sorted all 130k
    # to show 100. Pin that the per-entry is_ignored work stops near
    # head_limit + the bounded budget, not over the whole tree.
    from abstractcore.tools.common_tools import list_files as _lf
    from abstractcore.tools.abstractignore import AbstractIgnore

    big = tmp_path / "tree"
    big.mkdir()
    pkg = big / "pkg"
    pkg.mkdir()
    for i in range(2000):
        (pkg / f"f{i:04d}.py").write_text("x\n", encoding="utf-8")

    real = AbstractIgnore.is_ignored
    calls = {"n": 0}

    def _counting(self, path, *, is_dir=None):
        if is_dir is False:
            calls["n"] += 1
        return real(self, path, is_dir=is_dir)

    monkeypatch.setattr(AbstractIgnore, "is_ignored", _counting)
    out = _lf(directory_path=str(big), pattern="*", recursive=True, head_limit=5)
    assert "showing 5 of many entries" in out
    assert "more entries exist" in out and "not fully scanned" in out
    assert calls["n"] < 700, (
        f"is_ignored ran on {calls['n']} entries — head_limit + the bounded budget must stop the walk, "
        "not enumerate the whole 2000-file tree (the eager-enumeration regression)"
    )


def test_list_files_small_tree_keeps_exact_count_and_recent_first(tmp_path) -> None:
    # The common case is UNCHANGED: a normal tree still reports the exact
    # total and the most-recent-first global order (the stream exhausts within
    # the budget, so the full matched set is sorted).
    import os
    import time

    for i in range(12):
        p = tmp_path / f"m{i:02d}.py"
        p.write_text("x\n", encoding="utf-8")
        os.utime(p, (time.time() + i, time.time() + i))  # m11 newest

    out = list_files(directory_path=str(tmp_path), pattern="*.py", head_limit=3)
    assert "showing 3 of 12 entries" in out
    assert "9 more entries available" in out
    lines = [l for l in out.splitlines() if l.startswith("  m")]
    assert lines[0].strip().startswith("m11.py"), "most-recent-first global sort must survive on normal trees"


def test_list_files_composition_summary_labeled_partial_on_large_tree(tmp_path) -> None:
    big = tmp_path / "tree"
    big.mkdir()
    pkg = big / "pkg"
    pkg.mkdir()
    for i in range(700):
        (pkg / f"f{i:04d}.py").write_text("x\n", encoding="utf-8")
    for i in range(30):
        (big / f"d{i}.md").write_text("x\n", encoding="utf-8")
    out = list_files(directory_path=str(big), pattern="*", recursive=True, head_limit=5)
    assert "Composition (of what was scanned):" in out, "large-tree summary must be labeled partial"
    assert ".py" in out and ".md" in out
    assert "subfolders:" in out and "pkg/" in out


def test_read_file_entire_file_small_returns_all_lines(tmp_path) -> None:
    path = tmp_path / "small.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    out = read_file(file_path=str(path))
    assert out.startswith(f"File: {path} (3 lines)\n\n")
    assert "\n1: one\n2: two\n3: three\n" in out + "\n"


def test_read_file_entire_file_refuses_when_over_line_limit(tmp_path) -> None:
    path = tmp_path / "many-lines.txt"
    path.write_text("\n".join(["x"] * 2001) + "\n", encoding="utf-8")

    out = read_file(file_path=str(path))
    assert out.startswith(f"Refused: File '{path}' is too large to read entirely")
    assert "> 2000 lines" in out
    assert "Next step:" in out


def test_read_file_entire_file_does_not_refuse_based_on_bytes_only(tmp_path) -> None:
    path = tmp_path / "large-bytes.txt"
    path.write_text("a" * 100_001, encoding="utf-8")

    out = read_file(file_path=str(path))
    assert out.startswith(f"File: {path} (1 lines)\n\n1: a")


def test_read_file_range_refuses_when_requested_lines_over_limit(tmp_path) -> None:
    path = tmp_path / "range-too-large.txt"
    path.write_text("\n".join([str(i) for i in range(1, 3001)]) + "\n", encoding="utf-8")

    out = read_file(
        file_path=str(path),
        start_line=1,
        end_line=2001,
    )

    assert out.startswith("Refused: Requested range would return 2001 lines")
    assert "> 2000 lines" in out


def _reassemble_partial_reads(path_str: str) -> tuple[str, int, bool]:
    """Walk the #TRUNCATION continuation chain the way a model does — copy the
    start_char byte-offset from each footer — and reassemble the file. Returns
    (joined_text, num_chunks, saw_replacement_char)."""
    import re as _re

    def body_of(out: str):
        if "[END OF FILE]" in out:
            return out.split("\n\n", 1)[1].rsplit("\n\n[END OF FILE]", 1)[0], None
        head_end = out.index("\n\n") + 2
        tail_start = out.rindex("\n\n#TRUNCATION")
        m = _re.search(r"start_char=(\d+)", out[tail_start:])
        assert m, "partial chunk must name its continuation offset"
        return out[head_end:tail_start], int(m.group(1))

    bodies: list[str] = []
    offsets: list[int] = []
    out = read_file(file_path=path_str)
    for _ in range(200):
        b, nxt = body_of(out)
        bodies.append(b)
        if nxt is None:
            break
        offsets.append(nxt)
        out = read_file(file_path=path_str, start_char=nxt)
    else:
        raise AssertionError("continuation chain did not terminate — overlap/no-progress bug")
    assert offsets == sorted(offsets) and len(set(offsets)) == len(offsets), "offsets must strictly increase"
    joined = "".join(bodies)
    return joined, len(bodies), ("\ufffd" in joined)


def test_read_file_partial_continuation_is_byte_true_on_multibyte(tmp_path) -> None:
    """Regression for audit item 0828: the char-offset continuation used a byte seek
    with character arithmetic, so on non-ASCII files each continuation overlapped the
    previous chunk and split multibyte codepoints into U+FFFD. Byte-true offsets +
    codepoint-boundary trimming must reassemble the file exactly."""
    # Few lines (< 2000) but > 120k CHARS, multibyte -> triggers the char-cap partial lane.
    line = "café résumé naïve €uro — accentué contenu " * 40
    content = "\n".join(f"{i}:{line}" for i in range(1, 120))
    assert len(content) > 120_000 and content.count("\n") + 1 < 2000
    path = tmp_path / "minified_multibyte.txt"
    path.write_text(content, encoding="utf-8")

    joined, chunks, saw_replacement = _reassemble_partial_reads(str(path))
    assert chunks >= 2, "test must exercise more than one chunk"
    assert not saw_replacement, "no U+FFFD may appear — codepoints must not be split at a seam"
    assert joined == content, "byte-true continuation must reassemble the file exactly (no overlap/loss)"


def test_read_file_partial_continuation_exact_on_ascii(tmp_path) -> None:
    """ASCII files (bytes == chars) must also reassemble exactly — guards against a
    regression that would only manifest on multibyte content."""
    line = "x" * 1700
    content = "\n".join(f"{i}:{line}" for i in range(1, 120))
    assert len(content) > 120_000 and content.count("\n") + 1 < 2000
    path = tmp_path / "minified_ascii.txt"
    path.write_text(content, encoding="utf-8")

    joined, chunks, saw_replacement = _reassemble_partial_reads(str(path))
    assert chunks >= 2 and not saw_replacement
    assert joined == content


def _write_sample_py(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text(
        "import os\n\ndef Process():\n    x = 1\n    return process_it(x)\n\n"
        "def process_it(y):\n    return y + 1\n",
        encoding="utf-8",
    )
    return p


def test_search_files_context_lines_shows_surrounding_lines(tmp_path) -> None:
    """Item 0831: context_lines was silently discarded; it must now emit surrounding
    lines ('-' separator) around the match line (':' separator)."""
    p = _write_sample_py(tmp_path)
    out = search_files(pattern="return process_it", path=str(tmp_path), file_pattern="*.py", context_lines=1)
    assert "5: " in out and "return process_it" in out  # the match line
    assert "4- " in out  # before-context
    assert "6- " in out  # after-context


def test_search_files_context_lines_zero_is_byte_identical_default(tmp_path) -> None:
    """Guard: context_lines=0 (default) must produce match-only ':' output with no
    context lines and no '--' separators (backward compatibility)."""
    p = _write_sample_py(tmp_path)
    out = search_files(pattern="def ", path=str(tmp_path), file_pattern="*.py")
    assert "3: def Process():" in out
    assert "7: def process_it(y):" in out
    assert "- " not in out.replace("process_it", "").replace("Process", "")  # no context-dash rows
    assert "--" not in out


def test_search_files_case_sensitive_is_honored(tmp_path) -> None:
    """Item 0831: case_sensitive was silently discarded; it must now restrict matches."""
    p = _write_sample_py(tmp_path)
    sensitive = search_files(pattern="Process", path=str(tmp_path), file_pattern="*.py", case_sensitive=True)
    assert "3: def Process():" in sensitive
    assert "process_it" not in sensitive  # lowercase not matched under case-sensitivity
    insensitive = search_files(pattern="process", path=str(tmp_path), file_pattern="*.py")
    assert "Process" in insensitive and "process_it" in insensitive  # both matched


def test_search_files_output_mode_files_with_matches_and_count(tmp_path) -> None:
    """Item 0831: output_mode was silently discarded; files_with_matches/count must work."""
    p = _write_sample_py(tmp_path)
    fwm = search_files(pattern="def ", path=str(tmp_path), file_pattern="*.py", output_mode="files_with_matches")
    assert "sample.py" in fwm and "def Process" not in fwm  # paths only, no excerpts
    cnt = search_files(pattern="def ", path=str(tmp_path), file_pattern="*.py", output_mode="count")
    assert "2\t" in cnt and "sample.py" in cnt  # 2 def matches in the file


def test_search_files_output_mode_invalid_is_refused_not_ignored(tmp_path) -> None:
    """Item 0831: an unsupported output_mode must be an explicit error, never silently
    treated as content mode."""
    p = _write_sample_py(tmp_path)
    out = search_files(pattern="def", path=str(tmp_path), output_mode="bogus")
    assert out.startswith("Error: output_mode must be one of")


def test_search_files_ignore_dirs_is_honored(tmp_path) -> None:
    """Item 0831: ignore_dirs was silently discarded; named dirs must be skipped."""
    (tmp_path / "keep").mkdir()
    (tmp_path / "skipme").mkdir()
    (tmp_path / "keep" / "a.py").write_text("NEEDLE here\n", encoding="utf-8")
    (tmp_path / "skipme" / "b.py").write_text("NEEDLE here\n", encoding="utf-8")
    out = search_files(pattern="NEEDLE", path=str(tmp_path), file_pattern="*.py", ignore_dirs="skipme")
    assert "keep" in out
    assert "skipme" not in out


def test_search_files_ignore_dirs_accepts_a_list(tmp_path) -> None:
    """Item 0831 review fix: models routinely send arrays for plural params; a list-typed
    ignore_dirs must work, not silently no-op via str(list).split(',')."""
    (tmp_path / "keep").mkdir()
    (tmp_path / "skipme").mkdir()
    (tmp_path / "keep" / "a.py").write_text("NEEDLE here\n", encoding="utf-8")
    (tmp_path / "skipme" / "b.py").write_text("NEEDLE here\n", encoding="utf-8")
    out = search_files(pattern="NEEDLE", path=str(tmp_path), file_pattern="*.py", ignore_dirs=["skipme"])
    assert "keep" in out and "skipme" not in out


def test_search_files_alt_modes_label_multiline_truncation(tmp_path, monkeypatch) -> None:
    """Item 0831 review NEEDS-FIX: files_with_matches/count must not silently undercount or
    return a false 'No matches' when a multiline scan is capped — the cap must be labeled.
    The cap is monkeypatched tiny so a small file exercises the truncation path."""
    from abstractcore.tools import common_tools as ct

    monkeypatch.setattr(ct, "_SEARCH_MAX_MULTILINE_BYTES", 40)  # tiny cap
    p = tmp_path / "big.txt"
    # First match within the cap; second match well past byte 40.
    p.write_text("HEADMATCH here\n" + ("filler line padding\n" * 5) + "TAILMATCH far past the cap\n", encoding="utf-8")

    # count: undercounts silently WITHOUT the fix; must carry a #TRUNCATION label now.
    cnt = search_files(pattern=r"\w+MATCH", path=str(tmp_path), file_pattern="*.txt", output_mode="count", multiline=True)
    assert "#TRUNCATION" in cnt, "capped multiline count must be labeled, never a silent undercount"

    # files_with_matches for a pattern that ONLY appears past the cap: without the fix this is a
    # false 'No matches'; with the fix the miss is labeled so it is not read as a clean negative.
    miss = search_files(pattern="TAILMATCH", path=str(tmp_path), file_pattern="*.txt", output_mode="files_with_matches", multiline=True)
    assert "#TRUNCATION" in miss, "a match past the cap must not read as a clean 'No matches'"


def test_search_files_alt_modes_no_false_truncation_label_on_small_file(tmp_path) -> None:
    """Guard: a file within the cap must NOT carry a truncation label (no false positive)."""
    p = tmp_path / "small.txt"
    p.write_text("alpha NEEDLE\nbeta\nNEEDLE gamma\n", encoding="utf-8")
    cnt = search_files(pattern="NEEDLE", path=str(tmp_path), file_pattern="*.txt", output_mode="count", multiline=True)
    assert "2\t" in cnt and "#TRUNCATION" not in cnt


def test_list_files_path_shaped_glob_returns_hint_not_empty(tmp_path) -> None:
    """Item 0835: list_files fnmatches basenames only, so a path-shaped glob used to match
    NOTHING silently. It must now return an explicit hint naming the fix."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x", encoding="utf-8")
    out = list_files(directory_path=str(tmp_path), pattern="src/*.py")
    assert out.startswith("Error:") and "NAMES only" in out and "directory_path" in out
    # recursive/globstar form is caught too
    out2 = list_files(directory_path=str(tmp_path), pattern="**/*.py")
    assert out2.startswith("Error:") and "NAMES only" in out2


def test_list_files_plain_pattern_still_lists(tmp_path) -> None:
    """Guard: a normal name pattern (no '/') is unaffected by the path-glob check."""
    (tmp_path / "keep.py").write_text("x", encoding="utf-8")
    out = list_files(directory_path=str(tmp_path), pattern="*.py")
    assert "keep.py" in out and not out.startswith("Error:")
