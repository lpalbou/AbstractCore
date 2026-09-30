//! Shared UI helpers — plain functions over the token set. Everything
//! consumes theme tokens; no raw colors anywhere.

use abstracttui::base::{Point, Rgba};
use abstracttui::prelude::*;
use abstracttui::render::Style;
use abstracttui::widgets::{Badge, Tone};

use crate::cli::CliError;
use crate::config::FieldState;
use crate::store::Loadable;

/// A styled span for [`line()`].
pub type SpanSpec = (String, Rgba, bool);

pub fn span(text: impl Into<String>, ink: Rgba) -> SpanSpec {
    (text.into(), ink, false)
}

pub fn span_bold(text: impl Into<String>, ink: Rgba) -> SpanSpec {
    (text.into(), ink, true)
}

/// One row of styled spans (chrome-shaped multi-ink line).
pub fn line(spans: Vec<SpanSpec>) -> View {
    line_styled(LayoutStyle::line(1), spans)
}

pub fn line_styled(style: LayoutStyle, spans: Vec<SpanSpec>) -> View {
    Element::new()
        .style(style)
        .draw(move |canvas, rect| {
            // R4 from day one: a partially crushed rect arrives smaller
            // than asked — guard emptiness and clip on both axes.
            if rect.is_empty() {
                return;
            }
            let mut x = rect.x;
            let right = rect.x + rect.w;
            for (text, ink, bold) in &spans {
                if x >= right {
                    break;
                }
                let mut st = Style::new().fg(*ink);
                if *bold {
                    st = st.bold();
                }
                let budget = (right - x).max(0) as usize;
                let fitted = fit_width(text, budget);
                canvas.print_styled(Point::new(x, rect.y), &fitted, &st);
                x += abstracttui::text::width(&fitted);
            }
        })
        .build()
}

/// Pad (or truncate) to an exact CELL width — `format!("{:<N}")` counts
/// chars, so CJK/emoji values misalign every column to their right
/// (review P3-10).
pub fn pad_cells(text: &str, cells: usize) -> String {
    let fitted = fit_width(text, cells);
    let used = abstracttui::text::width(&fitted).max(0) as usize;
    let mut out = fitted;
    out.extend(std::iter::repeat_n(' ', cells.saturating_sub(used)));
    out
}

/// Cell-width-aware truncation with an honest `…` marker.
pub fn fit_width(text: &str, max_cells: usize) -> String {
    let max = max_cells as i32;
    if abstracttui::text::width(text) <= max {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0i32;
    let budget = max.saturating_sub(1); // room for the marker
    for ch in text.chars() {
        let w = abstracttui::text::width(&ch.to_string());
        if used + w > budget {
            break;
        }
        out.push(ch);
        used += w;
    }
    out.push('…');
    out
}

/// A labeled form row: fixed-width muted label, any child beside it.
pub fn field(t: &TokenSet, label: &str, child: View) -> View {
    field_w(t, label, 18, child)
}

pub fn field_w(t: &TokenSet, label: &str, label_w: i32, child: View) -> View {
    let ink = t.text_muted;
    let label = label.to_string();
    Element::new()
        .style(LayoutStyle::row().gap(1))
        .child(
            Element::new()
                .style(LayoutStyle::default().w(label_w).h(1).shrink(0.0))
                .draw(move |canvas, rect| {
                    if rect.is_empty() {
                        return;
                    }
                    let fitted = fit_width(&label, rect.w.max(0) as usize);
                    canvas.print(Point::new(rect.x, rect.y), &fitted, ink, Rgba::TRANSPARENT);
                })
                .build(),
        )
        .child(child)
        .build()
}

/// Key-hint row for the footer: pairs of (key, action).
pub fn hints(t: &TokenSet, pairs: &[(&str, &str)]) -> View {
    let mut spans = Vec::new();
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            spans.push(span("  ·  ", t.text_faint));
        }
        spans.push(span_bold((*k).to_string(), t.accent));
        spans.push(span(format!(" {v}"), t.text_muted));
    }
    line(spans)
}

pub fn badge(t: &TokenSet, label: &str, tone: Tone) -> View {
    Badge::new(label).tone(tone).element(t).build()
}

/// The honest-state panel around loadable data: distinct renders for
/// not-asked / loading / failed / ready-but-empty.
pub fn loadable_view<T>(
    t: &TokenSet,
    data: &Loadable<T>,
    empty_check: impl Fn(&T) -> bool,
    empty_text: &str,
    ready: impl FnOnce(&T) -> View,
) -> View {
    match data {
        Loadable::NotAsked => line(vec![span("— not loaded yet (r reloads)", t.text_muted)]),
        Loadable::Loading => line(vec![span("⟳ loading…", t.info)]),
        Loadable::Failed(e) => error_panel(t, e),
        Loadable::Ready(v) if empty_check(v) => {
            line(vec![span(format!("∅ {empty_text}"), t.text_muted)])
        }
        Loadable::Ready(v) => ready(v),
    }
}

/// Failure rendering that keeps error kinds distinct and always names
/// what the operator can do (refusals speak).
pub fn error_panel(t: &TokenSet, e: &CliError) -> View {
    Element::new()
        .style(LayoutStyle::column())
        .child(line(vec![span_bold(
            format!("✗ {}", e.headline()),
            t.error,
        )]))
        .child(line(vec![span(format!("  {}", e.message), t.text)]))
        .child(line(vec![span(format!("  {}", e.hint()), t.text_muted)]))
        .build()
}

/// The set/default/broken vocabulary, rendered one way everywhere.
pub fn state_spans(t: &TokenSet, state: &FieldState) -> Vec<SpanSpec> {
    match state {
        FieldState::Default => vec![span("default", t.text_faint)],
        FieldState::Set => vec![span_bold("set", t.ok)],
        FieldState::Broken(reason) => vec![
            span_bold("broken", t.error),
            span(format!(" — {reason}"), t.text_muted),
        ],
    }
}

/// `value` or an honest em-dash when absent.
pub fn or_dash(v: &Option<String>) -> String {
    match v {
        Some(s) if !s.is_empty() => s.clone(),
        _ => "—".into(),
    }
}

/// Truncate for table cells (display only, marked with …).
pub fn ellipsize(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{cut}…")
}

/// Human-readable byte size.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

// ---------------------------------------------------------------------
// State switches (the framework's state-toggle rule, terminal form).
// ---------------------------------------------------------------------

/// The state of one persistent on/off setting. A switch is labelled by
/// the FEATURE ("Agent email tools"), never by a verb ("Turn on"), and
/// its marker shows the current state: `[x]` on (accent, bold), `[ ]`
/// off (plain), `[-]` unavailable (faint, the reason after an em dash).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Switch {
    On,
    Off,
    /// Rendered, never actionable; the reason says what unlocks it.
    Unavailable(String),
}

impl Switch {
    pub fn from_bool(on: bool) -> Switch {
        if on {
            Switch::On
        } else {
            Switch::Off
        }
    }

    pub fn marker(&self) -> &'static str {
        match self {
            Switch::On => "[x]",
            Switch::Off => "[ ]",
            Switch::Unavailable(_) => "[-]",
        }
    }
}

/// `[x] Label` / `[ ] Label` / `[-] Label — reason`: the plain-text form
/// for table cells (which carry one ink per row).
pub fn switch_text(label: &str, state: &Switch) -> String {
    match state {
        Switch::Unavailable(reason) => format!("{} {label} — {reason}", state.marker()),
        _ => format!("{} {label}", state.marker()),
    }
}

/// The styled form: ON is accent + bold, OFF plain text, UNAVAILABLE
/// faint with its reason. The marker and the label share one ink so the
/// whole switch reads as its state.
pub fn switch_spans(t: &TokenSet, label: &str, state: &Switch) -> Vec<SpanSpec> {
    match state {
        Switch::On => vec![span_bold(format!("{} {label}", state.marker()), t.accent)],
        Switch::Off => vec![span(format!("{} {label}", state.marker()), t.text)],
        Switch::Unavailable(reason) => vec![span(
            format!("{} {label} — {reason}", state.marker()),
            t.text_faint,
        )],
    }
}

#[cfg(test)]
mod switch_tests {
    use super::*;

    fn tokens() -> TokenSet {
        abstracttui::theme::default_theme().tokens
    }

    /// ON is highlighted (accent + bold); removing the highlight or the
    /// marker turns this red.
    #[test]
    fn on_is_marked_and_highlighted() {
        let t = tokens();
        let spans = switch_spans(&t, "Agent email tools", &Switch::On);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].0, "[x] Agent email tools");
        assert_eq!(spans[0].1, t.accent, "ON wears the accent ink");
        assert!(spans[0].2, "ON is bold");
    }

    #[test]
    fn off_is_plain() {
        let t = tokens();
        let spans = switch_spans(&t, "Use this mailbox", &Switch::Off);
        assert_eq!(spans[0].0, "[ ] Use this mailbox");
        assert_eq!(spans[0].1, t.text, "OFF is plain text ink");
        assert!(!spans[0].2, "OFF is not bold");
        assert_ne!(t.text, t.accent, "the theme tells ON from OFF");
    }

    /// UNAVAILABLE is dimmed and says why, after an em dash.
    #[test]
    fn unavailable_is_dimmed_with_its_reason() {
        let t = tokens();
        let state = Switch::Unavailable("connect a mailbox first".into());
        let spans = switch_spans(&t, "Agent email tools", &state);
        assert_eq!(
            spans[0].0,
            "[-] Agent email tools — connect a mailbox first"
        );
        assert_eq!(spans[0].1, t.text_faint, "UNAVAILABLE is faint");
        assert!(!spans[0].2);
        assert_eq!(
            switch_text("Agent email tools", &state),
            "[-] Agent email tools — connect a mailbox first"
        );
        assert_eq!(switch_text("x", &Switch::from_bool(true)), "[x] x");
        assert_eq!(switch_text("x", &Switch::from_bool(false)), "[ ] x");
    }
}
