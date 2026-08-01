//! UI root: one PageHost shell over the 8 screens, two modes — browse
//! (free tabs) and wizard (a gated linear walk over the CLI wizard's
//! phases). Screens are plain component functions; durable UI state
//! (selections, wizard step) lives in [`UiState`] created at the root,
//! so screen remounts on tab switches lose nothing (PageHost disposes
//! page scopes on switch by design).

pub mod editors;
pub mod forms;
pub mod model_field;
pub mod overview;
pub mod providers;
pub mod review;
pub mod routes;
pub mod sections;
pub mod util;
pub mod widths;
pub mod wizard;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::Sender;
use std::time::Duration;

use abstracttui::app::{Modal, Overlays, ThemeSwitcher, Toast};
use abstracttui::prelude::*;
use abstracttui::reactive::IntervalHandle;
use abstracttui::widgets::PageHost;

use crate::config::FileState;
use crate::store::{Loadable, Store};
use crate::worker::Cmd;
use util::{fit_width, hints, line, span, span_bold};

pub const SCREENS: [&str; 8] = [
    "Overview",
    "Model",
    "Providers",
    "Routes",
    "Media",
    "Embeddings",
    "Server",
    "Review",
];

/// Stable PageHost page ids, parallel to `SCREENS`. `ui.screen: usize`
/// stays the source of truth; a two-way equality-guarded bridge keeps
/// PageHost's string `active` in lockstep.
pub const SCREEN_IDS: [&str; 8] = [
    "overview",
    "model",
    "providers",
    "routes",
    "media",
    "embeddings",
    "server",
    "review",
];

/// Which screen edits a given config section — the overview's
/// jump-to-owner map and the wizard's per-phase target.
pub fn screen_for_section(name: &str) -> usize {
    match name {
        "default_models" | "app_defaults" => 1,
        "provider_profiles" | "api_keys" => 2,
        "capability_defaults" => 3,
        "vision" | "audio" | "video" => 4,
        "embeddings" => 5,
        _ => 6, // server, logging, streaming, timeouts, offline, maintenance, email, cache
    }
}

/// Durable per-screen UI state (Copy: all signals).
#[derive(Clone, Copy)]
pub struct UiState {
    /// true = wizard chrome (gated steps); false = browse tabs.
    pub wizard: Signal<bool>,
    pub screen: Signal<usize>,
    /// The wizard's step index into `wizard::STEPS`.
    pub step: Signal<usize>,
    /// The section a wizard step edits — section pages filter to it.
    pub focus_section: Signal<Option<&'static str>>,

    pub overview_sel: Signal<usize>,
    pub route_sel: Signal<usize>,
    /// The Providers screen's ONE selection. It used to be two (a
    /// provider-inventory table plus a profiles table underneath);
    /// the screen has one unified list now, so it has one highlight.
    pub profile_sel: Signal<usize>,
    /// Field-table selections for the section pages.
    pub model_sel: Signal<usize>,
    pub media_sel: Signal<usize>,
    pub embeddings_sel: Signal<usize>,
    pub server_sel: Signal<usize>,

    /// (form_id, outcome) — write completions routed back to the one
    /// open form. Single slot is correct under the one-modal invariant.
    pub write_done: Signal<Option<(u64, Result<String, String>)>>,
    /// Bumped whenever the shared modal slot closes.
    pub modal_epoch: Signal<u64>,
}

impl UiState {
    pub fn create(cx: Scope) -> UiState {
        UiState {
            wizard: cx.signal(false),
            screen: cx.signal(0),
            step: cx.signal(0),
            focus_section: cx.signal(None),
            overview_sel: cx.signal(0),
            route_sel: cx.signal(0),
            profile_sel: cx.signal(0),
            model_sel: cx.signal(0),
            media_sel: cx.signal(0),
            embeddings_sel: cx.signal(0),
            server_sel: cx.signal(0),
            write_done: cx.signal(None),
            modal_epoch: cx.signal(0),
        }
    }
}

/// Cloneable UI context: the command channel, overlay store, quit and
/// the single-modal slot (stacked modals are an engine hazard — one at
/// a time, sequenced).
#[derive(Clone)]
pub struct Ctx {
    pub tx: Sender<Cmd>,
    pub overlays: Overlays,
    pub quitter: abstracttui::app::Quitter,
    pub store: Store,
    pub ui: UiState,
    pub modal: Rc<RefCell<Option<Modal>>>,
}

impl Ctx {
    pub fn send(&self, cmd: Cmd) {
        // A dropped worker only happens at quit; ignore then.
        let _ = self.tx.send(cmd);
    }

    pub fn close_modal(&self) {
        if let Some(m) = self.modal.borrow_mut().take() {
            m.close();
        }
        self.ui.modal_epoch.update(|e| *e += 1);
    }

    /// Send a test probe — single-flight: tests run real generations
    /// and live discoveries; a queued duplicate would silently double
    /// the cost behind one busy label. The flag is set HERE,
    /// synchronously (the UI thread owns signals): waiting for the
    /// worker's begin-post left every probe queued behind a busy
    /// worker invisible to the guard (M3 review P2-2). The worker
    /// still clears it on completion and on panic.
    pub fn send_probe(&self, spec: crate::probes::ProbeSpec) {
        if self.store.probe_busy.get_untracked() {
            self.store.notice.set(Some(
                "a test is already running — its result lands in the journal and on Review (8)"
                    .into(),
            ));
            return;
        }
        self.store.probe_busy.set(true);
        self.store.notice.set(Some(format!("⟳ {} …", spec.label)));
        self.send(Cmd::Probe(spec));
    }

    /// The file IDENTITY (mtime, ino, size) of the snapshot the
    /// operator is looking at — every WriteSpec carries it so the
    /// worker can refuse when another writer landed in between (no
    /// lock exists; mtime alone misses same-second rewrites).
    pub fn write_base(&self) -> Option<crate::config::FileStamp> {
        self.store.cfg.with_untracked(|c| {
            c.ready().and_then(|m| match &m.state {
                FileState::Ready(snap) => snap.stamp,
                _ => None,
            })
        })
    }

    /// Can a CLI-routed write run right now? Posts the reason and
    /// returns false when not — the corrupt/refused states are
    /// structural write-refusals at the UI door too (the worker
    /// re-checks on its own fresh read).
    pub fn writable_now(&self) -> bool {
        self.writable(true)
    }

    /// The RMW door: direct writes preserve every byte they don't
    /// touch, so a Python-REFUSED file stays editable through them
    /// (fixing the offending rows may even need it) — with a warning.
    /// Corrupt/unreadable still refuse everything (M2 review P2-3:
    /// the worker allowed this split but the door didn't, making the
    /// CHANGELOG claim dead code).
    pub fn writable_now_rmw(&self) -> bool {
        self.writable(false)
    }

    fn writable(&self, cli_routed: bool) -> bool {
        let refused = self.store.python_fallback.with_untracked(|f| f.is_some())
            || self.store.cfg.with_untracked(|c| {
                matches!(
                    c.ready().map(|m| &m.state),
                    Some(FileState::Ready(snap)) if !snap.python_refusals.is_empty()
                )
            });
        if refused {
            if cli_routed {
                self.store.notice.set(Some(
                    "Python refuses this file — CLI writes would RESET it; fix the named rows \
                     first (direct-write fields stay editable)"
                        .into(),
                ));
                return false;
            }
            self.store.notice.set(Some(
                "careful: Python refuses this file — this direct write preserves it, but fix \
                 the named rows (Overview) before any CLI write"
                    .into(),
            ));
            // fall through: RMW allowed
        }
        let verdict = self.store.cfg.with_untracked(|c| match c {
            Loadable::Ready(m) => match &m.state {
                FileState::Ready(_) | FileState::Missing => Ok(()),
                FileState::Corrupt { .. } => {
                    Err("the config file is corrupt — fix it first (Overview names the backups)"
                        .to_string())
                }
                FileState::Unreadable { error } => Err(format!("config unreadable: {error}")),
            },
            _ => Err("config not loaded yet — press r".to_string()),
        });
        match verdict {
            Ok(()) => true,
            Err(e) => {
                self.store.notice.set(Some(e));
                false
            }
        }
    }

    /// Reload everything: the file mirror and both derived views. One
    /// verb — this is a mirror of one file; partial staleness would
    /// lie. `reset_domains` first (the ONE exhaustive-destructure
    /// list), so caches with no Loading state of their own — the
    /// per-provider model lists — cannot serve stale entries across a
    /// reload (M2 review P3-9: the reset fn existed unwired).
    pub fn reload_all(&self) {
        self.store.reset_domains();
        self.store.cfg.set(Loadable::Loading);
        self.store.routes.set(Loadable::Loading);
        self.store.profiles.set(Loadable::Loading);
        self.store.availability.set(Loadable::Loading);
        self.send(Cmd::LoadConfig);
        self.send(Cmd::LoadRoutes);
        self.send(Cmd::LoadProfiles);
        // A reload is exactly when an operator expects a download that
        // just finished to show up in the weights column.
        self.send(Cmd::LoadAvailability);
    }
}

/// The root component.
pub fn root(cx: Scope, ctx: Ctx) -> View {
    let theme = use_theme(cx);
    let ui = ctx.ui;

    install_effects(cx, &ctx);

    let quit = ctx.quitter.clone();
    let ctx_q = ctx.clone();
    let ctx_refresh = ctx.clone();
    let ctx_next = ctx.clone();
    let ctx_back = ctx.clone();
    let ctx_esc = ctx.clone();
    let ctx_finish = ctx.clone();
    let ctx_wiz = ctx.clone();

    let mut root_el = Element::new()
        .style(LayoutStyle::column())
        .shortcut(KeyChord::plain(Key::Char('q')), move |_| {
            // q quits from browse; in wizard it is refused WITH A
            // REASON (a swallowed key reads as a dead app). A write in
            // flight also refuses — quitting mid-write abandons a
            // spawned setter that lands AFTER exit, unverified and
            // unjournaled (M2 review P3-4). Ctrl+C stays the force.
            if ui.wizard.get_untracked() {
                ctx_q.store.notice.set(Some(
                    "q quits in browse mode — Ctrl+C quits anywhere, f finishes the wizard".into(),
                ));
                return;
            }
            if ctx_q.store.busy.with_untracked(|b| !b.is_empty()) {
                ctx_q.store.notice.set(Some(
                    "an operation is in flight — wait for it (Ctrl+C force-quits; a running \
                     write would land unverified)"
                        .into(),
                ));
                return;
            }
            quit.quit();
        })
        .shortcut(KeyChord::plain(Key::Char('r')), move |_| {
            // The immediate ack: fast reloads repaint identically —
            // the notice IS the visible trace.
            ctx_refresh
                .store
                .notice
                .set(Some("⟳ reloading config file + derived views…".into()));
            ctx_refresh.reload_all();
        })
        // Ctrl chords survive focused text inputs; ]/[ work outside
        // them. Both advance the wizard in wizard mode and the tabs in
        // browse.
        .shortcut(KeyChord::new(Mods::CTRL, Key::Char('n')), move |_| {
            nav_next(&ctx_next);
        })
        .shortcut(KeyChord::new(Mods::CTRL, Key::Char('p')), move |_| {
            nav_back(&ctx_back);
        })
        .shortcut(KeyChord::plain(Key::Escape), move |_| {
            if ctx_esc.ui.wizard.get_untracked() {
                wizard::back(&ctx_esc);
            }
        })
        .shortcut(KeyChord::plain(Key::Char('f')), move |_| {
            if ctx_finish.ui.wizard.get_untracked() {
                wizard::finish(&ctx_finish);
            }
        })
        .shortcut(KeyChord::plain(Key::Char('g')), {
            let ctx_gen = ctx.clone();
            move |_| {
                // The definition-of-done verb: one cheap generation
                // over the CONFIGURED default route, from any screen.
                ctx_gen.send_probe(crate::probes::generate_default());
            }
        })
        .shortcut(KeyChord::plain(Key::Char('w')), move |_| {
            if !ctx_wiz.ui.wizard.get_untracked() {
                ctx_wiz.ui.wizard.set(true);
                wizard::apply_step(&ctx_wiz, 0);
                ctx_wiz.store.notice.set(Some(
                    "wizard mode — Ctrl+N next step, Esc back, f finish".into(),
                ));
            }
        });
    // (Ctrl+L rides the GLOBAL action registry — registered in lib.rs
    // so it stays live inside focus-trapped modals.)

    // Digit REFUSALS (wizard only): PageHost owns digit jumps in
    // browse; the wizard's refusal-with-a-reason stays a root shortcut
    // so a swallowed digit never reads as a dead app.
    for i in 0..SCREENS.len() {
        let ctx_i = ctx.clone();
        let key = char::from_digit(i as u32 + 1, 10).unwrap();
        root_el = root_el.shortcut(KeyChord::plain(Key::Char(key)), move |_| {
            if ctx_i.ui.wizard.get_untracked() {
                ctx_i.store.notice.set(Some(
                    "digit jumps work in browse mode — walk the wizard with Ctrl+N, finish with f"
                        .into(),
                ));
            }
        });
    }

    // ui.screen (usize, the wizard gate's truth) ⇄ PageHost's string
    // `active`. Both effects are equality-guarded — one hop, no
    // oscillation.
    let active =
        cx.signal(SCREEN_IDS[ui.screen.get_untracked().min(SCREEN_IDS.len() - 1)].to_string());
    cx.effect(move || {
        let id = SCREEN_IDS[ui.screen.get().min(SCREEN_IDS.len() - 1)];
        if active.with_untracked(|a| a != id) {
            active.set(id.to_string());
        }
    });
    cx.effect(move || {
        let pos = active.with(|a| SCREEN_IDS.iter().position(|s| s == a));
        if let Some(i) = pos {
            if ui.screen.get_untracked() != i {
                ui.screen.set(i);
            }
        }
    });

    // ONE PageHost carries the tab bar + page region. Free navigation
    // is ARMED in browse and fully DISARMED in wizard (empty chord
    // sets + number_jump(false)): the gate logic stays app-side. The
    // host rebuilds when the MODE flips; page state survives in
    // UiState.
    let host_ctx = ctx.clone();
    let host = dyn_view_scoped(LayoutStyle::default().grow(1.0), move |hcx| {
        let wizard_now = ui.wizard.get();
        let (prev_chords, next_chords) = if wizard_now {
            (Vec::new(), Vec::new())
        } else {
            (
                vec![
                    KeyChord::new(Mods::CTRL, Key::PageUp),
                    KeyChord::new(Mods::CTRL, Key::Char('p')),
                ],
                vec![
                    KeyChord::new(Mods::CTRL, Key::PageDown),
                    KeyChord::new(Mods::CTRL, Key::Char('n')),
                ],
            )
        };
        let c: Vec<Ctx> = (0..SCREEN_IDS.len()).map(|_| host_ctx.clone()).collect();
        let [c0, c1, c2, c3, c4, c5, c6, c7]: [Ctx; 8] = c.try_into().ok().expect("8 screens");
        PageHost::new()
            .page(SCREEN_IDS[0], "1 Overview", move |gcx| {
                overview::view(gcx, &c0, theme)
            })
            .page(SCREEN_IDS[1], "2 Model", move |gcx| {
                sections::page(
                    gcx,
                    &c1,
                    theme,
                    &["default_models", "app_defaults"],
                    c1.ui.model_sel,
                )
            })
            .page(SCREEN_IDS[2], "3 Providers", move |gcx| {
                providers::view(gcx, &c2, theme)
            })
            .page(SCREEN_IDS[3], "4 Routes", move |gcx| {
                routes::view(gcx, &c3, theme)
            })
            .page(SCREEN_IDS[4], "5 Media", move |gcx| {
                sections::page(
                    gcx,
                    &c4,
                    theme,
                    &["vision", "audio", "video"],
                    c4.ui.media_sel,
                )
            })
            .page(SCREEN_IDS[5], "6 Embeddings", move |gcx| {
                sections::embeddings_page(gcx, &c5, theme)
            })
            .page(SCREEN_IDS[6], "7 Server", move |gcx| {
                sections::page(
                    gcx,
                    &c6,
                    theme,
                    &[
                        "server",
                        "logging",
                        "streaming",
                        "timeouts",
                        "offline",
                        "maintenance",
                        "email",
                        "cache",
                    ],
                    c6.ui.server_sel,
                )
            })
            .page(SCREEN_IDS[7], "8 Review", move |gcx| {
                review::view(gcx, &c7, theme)
            })
            .active(active)
            .number_jump(!wizard_now)
            .chords(&prev_chords, &next_chords)
            .view(hcx)
    });

    // The wizard's step/goal line: auto-height — one row in wizard
    // mode, zero in browse (a permanently blank row is dead space at
    // 60x16).
    let goal = dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        if !ui.wizard.get() {
            return Element::new().style(LayoutStyle::default().h(0)).build();
        }
        let i = ui.step.get().min(wizard::STEPS.len() - 1);
        let step = &wizard::STEPS[i];
        line(vec![
            span_bold(
                format!(" Step {}/{} · {} — ", i + 1, wizard::STEPS.len(), step.title),
                t.accent,
            ),
            span(step.goal, t.text_muted),
        ])
    });

    root_el
        .child(header(cx, &ctx, theme))
        .child(
            Element::new()
                .style(LayoutStyle::default().grow(1.0))
                .child(host)
                .build(),
        )
        .child(goal)
        .child(footer(cx, &ctx, theme))
        .build()
}

fn nav_next(ctx: &Ctx) {
    if ctx.ui.wizard.get_untracked() {
        wizard::next(ctx);
    } else if ctx.ui.screen.get_untracked() + 1 < SCREENS.len() {
        ctx.ui.screen.update(|s| *s += 1);
    } else {
        ctx.store.notice.set(Some("already on the last screen".into()));
    }
}

fn nav_back(ctx: &Ctx) {
    if ctx.ui.wizard.get_untracked() {
        wizard::back(ctx);
    } else if ctx.ui.screen.get_untracked() > 0 {
        ctx.ui.screen.update(|s| *s -= 1);
    } else {
        ctx.store.notice.set(Some("already on the first screen".into()));
    }
}

fn install_effects(cx: Scope, ctx: &Ctx) {
    let store = ctx.store;
    let ui = ctx.ui;

    // Screen switches retire the footer notice — a stale toast line
    // must not outlive the context it was about.
    {
        let last_screen = Rc::new(std::cell::Cell::new(usize::MAX));
        cx.effect(move || {
            let s = ui.screen.get();
            if last_screen.get() != usize::MAX && last_screen.get() != s {
                store.notice.set(None);
            }
            last_screen.set(s);
        });
    }

    // Busy ticker: exists only while ops are in flight (zero idle cost).
    {
        let ticker: Rc<RefCell<Option<IntervalHandle>>> = Rc::new(RefCell::new(None));
        cx.effect(move || {
            let any = store.busy.with(|b| !b.is_empty());
            let mut slot = ticker.borrow_mut();
            match (any, slot.is_some()) {
                (true, false) => {
                    *slot = Some(abstracttui::reactive::interval(
                        cx,
                        Duration::from_millis(500),
                        move || store.tick.update(|t| *t += 1),
                    ));
                }
                (false, true) => {
                    if let Some(h) = slot.take() {
                        h.cancel();
                    }
                }
                _ => {}
            }
        });
    }

    // Notices → toast (and the footer mirrors the latest one).
    {
        let overlays = ctx.overlays.clone();
        cx.effect(move || {
            if let Some(n) = store.notice.get() {
                let viewport = abstracttui::app::use_viewport(cx).get_untracked();
                Toast::show(
                    &overlays,
                    cx,
                    viewport,
                    util::ellipsize(&n, (viewport.w as usize).saturating_sub(6).max(20)),
                    Duration::from_secs(4),
                );
            }
        });
    }
}

/// Humanize a raw engine startup notice for the operator footer: the
/// zero-collapse diagnostic reads as a crash log verbatim. The raw
/// line stays behind the debug env flag.
pub fn humanize_engine_notice(raw: &str) -> String {
    if raw.trim_start().starts_with("layout:") {
        if std::env::var("ABSTRACTCORE_CONSOLE_DEBUG").is_ok() {
            return raw.to_string();
        }
        return "display degraded — a panel over-demanded space; rows may overlap \
                (ABSTRACTCORE_CONSOLE_DEBUG=1 for details)"
            .to_string();
    }
    raw.to_string()
}

fn header(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    let ui = ctx.ui;
    let viewport = abstracttui::app::use_viewport(cx);
    // shrink(0.0): the title bar is CHROME — without the pin, a page
    // whose content minimum over-demands height flex-shrinks this
    // fixed row to zero and the tab bar paints at row 0 (the sibling's
    // vanishing-title-bar incident; engine finding 0240's class).
    dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        let avail = viewport.get().w;
        // A file the mirror parsed but PYTHON refuses (loader raise →
        // defaults + backup) must not wear the green dot — either
        // detection lane (fold-side refusals, CLI #FALLBACK stderr)
        // flips the header (review P1-1).
        let python_refuses = store.python_fallback.with(|f| f.is_some())
            || store.cfg.with(|c| {
                matches!(
                    c.ready().map(|m| &m.state),
                    Some(FileState::Ready(snap)) if !snap.python_refusals.is_empty()
                )
            });
        let (path, dot, dot_ink, label) = match store.cfg.get() {
            Loadable::Ready(m) => {
                let path = m.path.path.display().to_string();
                match &m.state {
                    FileState::Ready(_) if python_refuses => (
                        path,
                        "●",
                        t.error,
                        "loaded — but Python REFUSES it (Overview)".to_string(),
                    ),
                    FileState::Ready(_) => (path, "●", t.ok, "loaded".to_string()),
                    FileState::Missing => {
                        (path, "○", t.text_muted, "no config file yet".to_string())
                    }
                    FileState::Corrupt { .. } => (path, "●", t.error, "CORRUPT".to_string()),
                    FileState::Unreadable { .. } => (path, "●", t.error, "unreadable".to_string()),
                }
            }
            Loadable::Loading => (String::new(), "◌", t.info, "reading…".to_string()),
            _ => (String::new(), "○", t.text_muted, "not loaded".to_string()),
        };
        let mode = if ui.wizard.get() { "wizard" } else { "browse" };
        // STATE FIRST: the header truncates last-span-first, and at 60
        // cols a long path was evicting the state label — the single
        // most important span (a CORRUPT flag pushed off-screen is a
        // lying header).
        //
        // The path is the ELASTIC span, and its budget is the row it is
        // actually drawn on — a constant 46 cut `…/.abstract…nfig/
        // abstractcore.json` on a 200-cell terminal with a hundred cells
        // still free. The FILENAME is what identifies the file, so the
        // cut keeps the tail.
        let brand = " AbstractCore Console ";
        let state = format!("{dot} {label} ");
        let lead = format!("· {mode} · ");
        let path = widths::middle_fit(
            &path,
            widths::elastic_budget(&[brand, &state, &lead], avail),
        );
        line(vec![
            span_bold(brand.to_string(), t.accent),
            span(format!("{dot} "), dot_ink),
            span(format!("{label} "), t.text),
            span(format!("{lead}{path}"), t.text_muted),
        ])
    })
}

fn footer(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    let ui = ctx.ui;
    let engine_notices = abstracttui::app::use_startup_notices(cx);
    Element::new()
        // Chrome rows: pinned like the header — the hint line
        // disappearing under content pressure would take the app's
        // teachable surface with it.
        .style(LayoutStyle::column().shrink(0.0))
        .child(dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
            // Busy strip: in-flight ops with elapsed seconds; app
            // notice next; engine startup notices when idle (the lane
            // the engine names layout crushes into — unread means
            // invisible).
            let t = theme.get().tokens;
            let _ = store.tick.get();
            let ops = store.busy.get();
            if ops.is_empty() {
                let notice = store.notice.get();
                return match notice {
                    Some(n) => line(vec![span(format!(" {n}"), t.text_muted)]),
                    None => {
                        // Diagnostic engine notices only — the ambient
                        // caps summary reads as a permanent warning.
                        // Exact "caps:" prefix: a broader match would
                        // also hide a future diagnostic that merely
                        // starts with the word.
                        let last = engine_notices.with(|v| {
                            v.iter()
                                .rev()
                                .find(|n| !n.trim_start().starts_with("caps:"))
                                .cloned()
                        });
                        match last {
                            Some(en) => line(vec![span(
                                format!(" engine: {}", humanize_engine_notice(&en)),
                                t.warn,
                            )]),
                            None => line(vec![span(String::new(), t.text_muted)]),
                        }
                    }
                };
            }
            let mut parts = Vec::new();
            for (i, op) in ops.iter().enumerate() {
                if i > 0 {
                    parts.push(span(" · ", t.text_faint));
                }
                let secs = op.started.elapsed().as_secs();
                parts.push(span(format!(" ⟳ {}… {}s", op.label, secs), t.info));
            }
            line(parts)
        }))
        .child(
            Element::new()
                .style(LayoutStyle::row().shrink(0.0))
                .child(
                    Element::new()
                        .style(LayoutStyle::default().grow(1.0).h(1))
                        .child(dyn_view(LayoutStyle::line(1), move || {
                            let t = theme.get().tokens;
                            let screen = ui.screen.get();
                            let wizard = ui.wizard.get();
                            let mut pairs: Vec<(&str, &str)> = Vec::new();
                            // Universal pairs FIRST — the hint row
                            // truncates right-edge-first and the quit
                            // affordance must survive at 60 cols.
                            if wizard {
                                pairs.push(("Ctrl+N", "next step"));
                                pairs.push(("Esc", "back"));
                                pairs.push(("f", "finish"));
                                pairs.push(("Ctrl+C", "quit"));
                            } else {
                                pairs.push(("1-8", "screens"));
                                pairs.push(("q", "quit"));
                                pairs.push(("w", "wizard"));
                            }
                            pairs.push(("r", "reload"));
                            match screen {
                                0 => pairs.push(("Enter", "open section")),
                                1 | 4 | 5 | 6 => {
                                    pairs.push(("Enter/e", "edit field"));
                                    pairs.push(("x", "clear field"));
                                }
                                2 => {
                                    // ONE list, the gateway console's
                                    // verbs, in its order.
                                    pairs.push(("a", "add connection"));
                                    pairs.push(("e", "edit"));
                                    pairs.push(("d", "delete"));
                                    pairs.push(("m", "models"));
                                    pairs.push(("t", "test"));
                                    pairs.push(("k", "set key"));
                                }
                                3 => {
                                    pairs.push(("Enter/e", "edit route"));
                                    pairs.push(("x", "clear route"));
                                    pairs.push(("t", "test route"));
                                    // `d` is "delete" on Providers and
                                    // "download" here. The two never share
                                    // a screen, the hint row names the verb
                                    // for the screen you are on, and the
                                    // download confirms with the artifact
                                    // spelled out before it spends a byte.
                                    pairs.push(("w", "download weights"));
                                }
                                _ => {}
                            }
                            pairs.push(("g", "test default route"));
                            hints(&t, &pairs)
                        }))
                        .build(),
                )
                // The theme control: one cell of chrome, engine-owned
                // popup (opens upward from the footer automatically).
                .child(
                    Element::new()
                        .style(LayoutStyle::default().w(3).h(1).shrink(0.0))
                        .child(ThemeSwitcher::new().view(cx))
                        .build(),
                )
                .build(),
        )
        .build()
}

/// Path display shared by overview/review: name + where it came from.
pub fn config_identity_line(t: &TokenSet, store: &Store) -> Vec<util::SpanSpec> {
    match store.cfg.get() {
        Loadable::Ready(m) => {
            let mut spans = vec![
                span(m.path.path.display().to_string(), t.text),
                span(format!("  ({})", m.path.source.label()), t.text_faint),
            ];
            if let FileState::Ready(snap) = &m.state {
                spans.push(span(
                    format!("  · {}", util::human_bytes(snap.bytes)),
                    t.text_muted,
                ));
                if let Some(mode) = snap.mode {
                    let ink = if mode & 0o077 != 0 { t.warn } else { t.text_muted };
                    spans.push(span(format!("  · mode {mode:03o}"), ink));
                    if mode & 0o077 != 0 {
                        spans.push(span("  (secrets readable by others!)", t.warn));
                    }
                }
            }
            spans.push(span(format!("  · read {}", m.loaded_at), t.text_faint));
            spans
        }
        Loadable::Loading => vec![span("⟳ reading…", t.info)],
        _ => vec![span("— not loaded", t.text_muted)],
    }
}

/// One-line truncation helper for table cells built from spans.
pub fn cell(s: &str, max: usize) -> String {
    fit_width(s, max)
}
