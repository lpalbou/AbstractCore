//! THE WIDTH POLICY — measure first, truncate only under real pressure.
//!
//! Shared verbatim with the AbstractGateway console-TUI
//! (`console-tui/src/ui/widths.rs` there): the gateway console is the
//! reference surface, and a column that reads one way here and another
//! way on its sibling is a drift bug waiting to be filed twice. Both
//! copies — module, policy and tests — must stay byte-identical below
//! this header.
//!
//! ## The defect this replaces
//!
//! Both routes screens sized their grid from CONSTANTS — `Cells(28)` for
//! the route key, `Cells(20)` for the source, plus a pre-render
//! `ellipsize(model, 40)` — and handed the leftover to a `Flex` model
//! column. On a 200-cell terminal that printed
//! `AbstractFramework/wan2.2-t2v-a14b-diffu…` beside seventy blank cells:
//! the payload was cut to a constant while the space to carry it whole sat
//! unused one column over. Worse, the cut fell at the HEAD-preserving end,
//! so `…t2v-a14b-diffusers-8bit` and `…i2v-a14b-diffusers-8bit` printed as
//! the same string and the column stopped telling its rows apart.
//!
//! ## The policy
//!
//! 1. **Measure.** A column's natural width is the widest thing it must
//!    say — every cell in the column plus its own header title.
//! 2. **Fit when it fits.** If the naturals tile inside the budget, every
//!    column gets its natural width and NOTHING is truncated. Leftover
//!    cells stay blank at the right edge; blank space is not a defect,
//!    a lie about a model name is.
//! 3. **Shrink proportionally under pressure.** Over budget, columns give
//!    up cells in proportion to the slack they actually have
//!    (`natural - min`), so no column collapses to a stub while its
//!    neighbours stay fat. Payload columns are protected by a generous
//!    `min`, never by a special case.
//! 4. **Keep the tail on identifiers.** Model names, route keys, source
//!    modules and provider ids differ in their LAST segment. Those columns
//!    truncate in the middle ([`middle_fit`]); prose columns keep the head.
//! 5. **Cut once.** The engine's `Table` truncates any cell wider than its
//!    column, head-first, with no say in the matter — so cells are fitted
//!    HERE, to the width solved here, and the engine's own cut becomes a
//!    no-op it never reaches.

use abstracttui::text::width as cells;
use abstracttui::widgets::{ColWidth, Column, Table};

/// The engine's `Table` draws exactly one blank cell between columns
/// (`solve_columns`: `usable = total - (n - 1)`), and the caller never
/// sees it. Budgeting it here keeps our solved widths and the engine's
/// tiling talking about the same cells.
const GAP: i32 = 1;

/// A long list grows a scrollbar in the right-most cell, and the table
/// does not tell us whether it drew one. Both route grids carry ~25 rows,
/// so reserve it unconditionally: over-solving costs the LAST column its
/// cells (the engine clamps it away), under-solving costs one blank
/// column nobody can see.
const SCROLLBAR: i32 = 1;

/// Cells a screen's bordered `Block` spends on itself, left plus right.
///
/// A grid inside one gets `viewport - BLOCK_CHROME`; a grid mounted bare
/// in PageHost's page region gets the viewport untouched. Measured
/// against the live consoles: a 200-cell terminal hands a blocked table
/// 198. Guessing high here is the expensive mistake — the engine clamps
/// the LAST column away — so when in doubt, budget less.
pub const BLOCK_CHROME: i32 = 2;

/// One column's appetite: what it is called, how far it may be squeezed,
/// and which end of its content discriminates.
#[derive(Clone, Copy, Debug)]
pub struct ColRule {
    pub title: &'static str,
    /// Never squeezed below this while any other column still has slack.
    pub min: i32,
    /// `true` — the content is an IDENTIFIER whose tail tells rows apart
    /// (`…-t2v-a14b-diffusers-8bit`); truncate in the middle.
    /// `false` — prose or a fixed vocabulary; keep the head.
    pub tail: bool,
}

impl ColRule {
    /// A column whose TAIL discriminates: model names, route keys,
    /// provider ids, source modules, URLs, file paths.
    pub const fn tail(title: &'static str, min: i32) -> ColRule {
        ColRule {
            title,
            min,
            tail: true,
        }
    }

    /// A column read from the left: state vocabularies, counts, prose.
    pub const fn head(title: &'static str, min: i32) -> ColRule {
        ColRule {
            title,
            min,
            tail: false,
        }
    }
}

/// Solve column widths for a table body `rect_w` cells wide.
///
/// `rows` is row-major and may be ragged; missing cells simply do not
/// widen their column. The returned widths are in the same order as
/// `rules` and always tile inside the budget (gap + scrollbar reserved).
///
/// The no-truncation guarantee: when the naturals fit, the result IS the
/// naturals — this function never invents a cap.
pub fn solve(rules: &[ColRule], rows: &[Vec<String>], rect_w: i32) -> Vec<i32> {
    let n = rules.len();
    if n == 0 {
        return Vec::new();
    }
    let usable = (rect_w - SCROLLBAR - GAP * (n as i32 - 1)).max(0);

    // NATURAL — the widest thing this column must say, its own header
    // included (a header the grid clips is the same defect one row up).
    let mut natural: Vec<i32> = rules.iter().map(|r| cells(r.title)).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate().take(n) {
            natural[i] = natural[i].max(cells(cell));
        }
    }
    // A column may never be solved below its own floor by MEASUREMENT —
    // an empty column still has to be clickable and titled.
    for (i, r) in rules.iter().enumerate() {
        natural[i] = natural[i].max(r.min.min(cells(r.title)));
    }

    let total: i32 = natural.iter().sum();
    if total <= usable {
        // THE ANSWER TO THE BUG REPORT: it fits, so nothing is cut.
        return natural;
    }

    // Over budget. Everyone gives up cells in proportion to the slack
    // they have; a column already at its floor contributes nothing.
    let slack: Vec<i32> = rules
        .iter()
        .enumerate()
        .map(|(i, r)| (natural[i] - r.min).max(0))
        .collect();
    let total_slack: i32 = slack.iter().sum();
    let mut out = natural;
    let deficit = total - usable;
    if total_slack > 0 {
        let take = deficit.min(total_slack);
        // Largest remainder: the shares must tile EXACTLY, or the
        // rounding crumbs re-create the overflow we are paying off.
        let mut shares = vec![0i32; n];
        let mut rems: Vec<(i64, usize)> = Vec::with_capacity(n);
        let mut given = 0i32;
        for (i, s) in slack.iter().enumerate() {
            let exact = i64::from(take) * i64::from(*s);
            let whole = (exact / i64::from(total_slack)) as i32;
            shares[i] = whole;
            given += whole;
            rems.push((exact % i64::from(total_slack), i));
        }
        rems.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut left = take - given;
        for (_, i) in rems {
            if left == 0 {
                break;
            }
            if shares[i] < slack[i] {
                shares[i] += 1;
                left -= 1;
            }
        }
        for (i, s) in shares.iter().enumerate() {
            out[i] -= s;
        }
    }

    // Every column sits on its floor and the budget is STILL short: this
    // terminal is genuinely too narrow. Shave the widest column a cell at
    // a time so the damage spreads, and leave every column at least one
    // cell — a column of width 0 is a column the operator cannot see is
    // there at all.
    let mut over: i32 = out.iter().sum::<i32>() - usable;
    while over > 0 {
        let Some(i) = widest(&out) else { break };
        if out[i] <= 1 {
            break;
        }
        out[i] -= 1;
        over -= 1;
    }
    out
}

fn widest(out: &[i32]) -> Option<usize> {
    out.iter()
        .enumerate()
        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(&a.0)))
        .map(|(i, _)| i)
}

/// Fit every cell to its solved column width — the ONE cut, made where
/// the rule says the content's discrimination lives.
pub fn fit_cells(rules: &[ColRule], rows: &mut [Vec<String>], widths: &[i32]) {
    for row in rows.iter_mut() {
        for (i, cell) in row.iter_mut().enumerate() {
            let (Some(w), Some(rule)) = (widths.get(i), rules.get(i)) else {
                continue;
            };
            if cells(cell) <= *w {
                continue;
            }
            *cell = if rule.tail {
                middle_fit(cell, *w)
            } else {
                head_fit(cell, *w)
            };
        }
    }
}

/// Solve, cut, and hand back engine columns — the one call a screen makes.
///
/// `rect_w` is the width of the table's own rect, NOT the terminal: a
/// screen that wraps its grid in a bordered block owes the border its
/// cells (see each call site's chrome constant).
pub fn columns(rules: &[ColRule], rows: &mut [Vec<String>], rect_w: i32) -> Vec<Column> {
    let widths = solve(rules, rows, rect_w);
    fit_cells(rules, rows, &widths);
    rules
        .iter()
        .zip(&widths)
        .map(|(r, w)| Column::new(r.title, ColWidth::Cells(*w)))
        .collect()
}

/// Build the table straight from the rules — solve, cut, construct.
pub fn table(rules: &[ColRule], mut rows: Vec<Vec<String>>, rect_w: i32) -> Table {
    let cols = columns(rules, &mut rows, rect_w);
    Table::new(cols).rows(rows)
}

/// Truncate keeping the HEAD, with an honest `…`. Cell-width aware:
/// `format!("{:.N}")` counts chars, and one CJK char is two cells.
pub fn head_fit(s: &str, max: i32) -> String {
    if max <= 0 {
        return String::new();
    }
    if cells(s) <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for ch in s.chars() {
        let w = cells(&ch.to_string());
        if used + w > max - 1 {
            break;
        }
        out.push(ch);
        used += w;
    }
    out.push('…');
    out
}

/// Truncate keeping the TAIL — the discriminating end of an identifier.
///
/// `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` and its `i2v` twin
/// share a 24-character prefix and differ only near the end, so a
/// head-preserving cut renders both rows identically and the column stops
/// being evidence.
///
/// The split is deliberately lopsided — a FIFTH of the budget to the head,
/// the rest to the tail (the operator's own ruling: `…t2v-a14b-diffusers-8bit`
/// beats `AbstractFramework/wan2.2-t2v…`). The head is there only to name
/// the family (`Abstr…` vs `stabi…`); the tail is the evidence. Below five
/// cells of body there is no room to do both, and the tail wins outright.
pub fn middle_fit(s: &str, max: i32) -> String {
    if max <= 0 {
        return String::new();
    }
    if cells(s) <= max {
        return s.to_string();
    }
    if max == 1 {
        return "…".to_string();
    }
    let body = max - 1; // the marker owns one cell
    let head_budget = if body >= 5 { body / 5 } else { 0 };
    let tail_budget = body - head_budget;

    // Head first, then the tail out of what the head did NOT consume —
    // taking both from the original string could print one grapheme
    // twice when the two windows meet.
    let mut head = String::new();
    let mut used = 0;
    let mut chars = s.chars().peekable();
    while let Some(&ch) = chars.peek() {
        let w = cells(&ch.to_string());
        if used + w > head_budget {
            break;
        }
        head.push(ch);
        used += w;
        chars.next();
    }
    let rest: Vec<char> = chars.collect();
    let mut tail_rev = String::new();
    let mut used = 0;
    for ch in rest.iter().rev() {
        let w = cells(&ch.to_string());
        if used + w > tail_budget {
            break;
        }
        tail_rev.push(*ch);
        used += w;
    }
    let tail: String = tail_rev.chars().rev().collect();
    format!("{head}…{tail}")
}

/// Budget a chrome line's ELASTIC segment.
///
/// A banner is a fixed sentence with one open-ended list wedged into it
/// (`missing: a, b, c  ·  w downloads …`). Drawn naively the list eats the
/// row and the engine clips whatever came last — which is the actionable
/// verb, the one span the operator needed. Reserve every fixed part, hand
/// the remainder to the list, and the verb always survives.
///
/// Returns the cells the elastic span may occupy — 0 when the fixed parts
/// already fill the row, which is the caller's cue to drop the list.
pub fn elastic_budget(fixed: &[&str], available: i32) -> i32 {
    let spent: i32 = fixed.iter().map(|s| cells(s)).sum();
    (available - spent).max(0)
}

/// Does this exact set of segments fit on a row `available` cells wide?
///
/// The cue for dropping an OPTIONAL segment whole rather than letting the
/// engine clip the row's last (and usually most actionable) span.
pub fn fits(parts: &[&str], available: i32) -> bool {
    parts.iter().map(|s| cells(s)).sum::<i32>() <= available
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(cells: &[&[&str]]) -> Vec<Vec<String>> {
        cells
            .iter()
            .map(|r| r.iter().map(|c| (*c).to_string()).collect())
            .collect()
    }

    /// THE BUG REPORT, as an assertion: room to the right means no cut.
    #[test]
    fn wide_terminal_truncates_nothing() {
        let rules = [
            ColRule::tail("route", 18),
            ColRule::head("state", 14),
            ColRule::tail("model", 22),
        ];
        let mut r = rows(&[
            &[
                "output.video.text_to_video",
                "configured",
                "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
            ],
            &["input.text", "configured", "gpt-5.4"],
        ]);
        let w = solve(&rules, &r, 200);
        assert_eq!(w[0], 26, "route sizes to its widest key, not a constant");
        assert_eq!(w[1], 10, "state sizes to 'configured'");
        assert_eq!(w[2], 48, "model sizes to the full artifact name");
        fit_cells(&rules, &mut r, &w);
        assert_eq!(
            r[0][2], "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
            "a model that fits is printed whole"
        );
        assert!(
            !r.iter().any(|row| row.iter().any(|c| c.contains('…'))),
            "no ellipsis anywhere at 200 cells: {r:?}"
        );
    }

    /// A column never shrinks below its own header, and the solved widths
    /// always tile inside the budget the engine will hand the table.
    #[test]
    fn solved_widths_always_tile_inside_the_budget() {
        let rules = [
            ColRule::tail("route", 18),
            ColRule::head("state", 14),
            ColRule::tail("provider", 10),
            ColRule::tail("model", 22),
            ColRule::head("weights", 14),
            ColRule::tail("source", 12),
        ];
        let r = rows(&[&[
            "output.scene3d.image_to_scene3d",
            "covered by input.text",
            "endpoint:airbender",
            "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
            "not downloaded",
            "abstractcore.capability_defaults",
        ]]);
        for w in [40, 60, 80, 100, 120, 160, 200, 300] {
            let solved = solve(&rules, &r, w);
            let used: i32 = solved.iter().sum::<i32>() + (rules.len() as i32 - 1) + 1;
            assert!(
                used <= w,
                "widths overflow the {w}-cell budget: {solved:?} -> {used}"
            );
            assert!(
                solved.iter().all(|c| *c >= 1),
                "every column keeps a cell at {w}: {solved:?}"
            );
        }
    }

    /// Under pressure the squeeze is SHARED — the old policy handed the
    /// whole deficit to one column and left it a stub.
    #[test]
    fn narrow_terminal_shares_the_squeeze() {
        let rules = [
            ColRule::tail("route", 18),
            ColRule::head("state", 14),
            ColRule::tail("model", 22),
            ColRule::tail("source", 12),
        ];
        let r = rows(&[&[
            "output.scene3d.image_to_scene3d",
            "covered by input.text",
            "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
            "abstractcore.capability_defaults",
        ]]);
        let wide = solve(&rules, &r, 200);
        let tight = solve(&rules, &r, 90);
        assert!(tight.iter().sum::<i32>() < wide.iter().sum::<i32>());
        for (i, rule) in rules.iter().enumerate() {
            assert!(
                tight[i] < wide[i],
                "{} carried none of the squeeze: {wide:?} -> {tight:?}",
                rule.title
            );
            assert!(
                tight[i] >= rule.min,
                "{} squeezed past its floor: {tight:?}",
                rule.title
            );
        }
    }

    /// The discriminating tail survives the cut — the whole reason
    /// identifier columns get a middle ellipsis.
    #[test]
    fn middle_fit_keeps_what_tells_rows_apart() {
        let t2v = "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit";
        let i2v = "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit";
        let a = middle_fit(t2v, 30);
        let b = middle_fit(i2v, 30);
        assert_eq!(cells(&a), 30, "fits exactly: {a:?}");
        assert!(a.ends_with("t2v-a14b-diffusers-8bit"), "tail survives: {a:?}");
        assert!(a.starts_with("Abstr"), "family still readable: {a:?}");
        assert!(a.contains('…'), "the cut is marked: {a:?}");
        assert_ne!(
            a, b,
            "two different models must still read differently: {a:?} vs {b:?}"
        );

        // The failure mode this replaces, on the operator's own rows: a
        // narrow column cut head-first prints ONE string for three
        // different models, and the grid stops being evidence.
        let live = [
            "AbstractFramework/flux.2-klein-9b-8bit",
            "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
            "AbstractFramework/seedvr2-7b-8bit",
        ];
        let headed: Vec<String> = live.iter().map(|m| head_fit(m, 15)).collect();
        assert!(
            headed[0] == headed[1] && headed[1] == headed[2],
            "head_fit really does collapse these — that is the bug: {headed:?}"
        );
        let middled: Vec<String> = live.iter().map(|m| middle_fit(m, 15)).collect();
        assert!(
            middled[0] != middled[1] && middled[1] != middled[2] && middled[0] != middled[2],
            "middle_fit keeps them apart at the same width: {middled:?}"
        );

        assert_eq!(middle_fit("short", 30), "short", "no cut when it fits");
        assert_eq!(middle_fit("anything", 1), "…");
        assert_eq!(middle_fit("anything", 0), "");
        for w in 1..12 {
            assert!(
                cells(&middle_fit(t2v, w)) <= w,
                "middle_fit overflows at {w}"
            );
        }
    }

    /// Wide glyphs are measured in CELLS, not chars — a char-counting cut
    /// slides every column to its right.
    #[test]
    fn cuts_are_cell_aware_not_char_aware() {
        let cjk = "模型模型模型模型模型模型";
        assert_eq!(cells(cjk), 24);
        for w in 2..24 {
            assert!(cells(&middle_fit(cjk, w)) <= w, "middle_fit at {w}");
            assert!(cells(&head_fit(cjk, w)) <= w, "head_fit at {w}");
        }
        let rules = [ColRule::tail("m", 4)];
        let r = rows(&[&[cjk]]);
        assert_eq!(solve(&rules, &r, 40)[0], 24, "natural width counts cells");
    }

    /// The banner's actionable verb outranks its open-ended list.
    #[test]
    fn elastic_budget_reserves_the_fixed_parts() {
        let head = "recommended models: 2 of 3 present";
        let mid = "  ·  missing: ";
        let verb = "  ·  w downloads the selected route's weights";
        let budget = elastic_budget(&[head, mid, verb], 120);
        assert_eq!(budget, 120 - cells(head) - cells(mid) - cells(verb));
        let list = middle_fit("lmstudio qwen/qwen3.5-9b@4bit", budget);
        assert!(list.ends_with("@4bit"), "the artifact tag survives: {list:?}");
        let total = cells(head) + cells(mid) + cells(&list) + cells(verb);
        assert!(total <= 120, "the whole line fits: {total}");
        assert_eq!(elastic_budget(&[head, mid, verb], 60), 0, "no room, no list");

        // `fits` is the cue for dropping an OPTIONAL segment whole
        // instead of letting the engine clip the actionable one.
        let count = "  ·  1 missing";
        assert!(fits(&[head, count, verb], 100), "the short form fits at 100");
        assert!(
            !fits(&[head, count, verb], 90),
            "at 90 even the count has to go, or the verb loses its end"
        );
        assert!(fits(&[head, verb], 90), "…and without it the verb survives");
    }
}
