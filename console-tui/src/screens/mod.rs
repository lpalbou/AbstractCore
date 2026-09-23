//! The shared **Models** and **Engines** screens (contract H) — one
//! implementation, mounted by both the core console (over
//! [`CliTransport`](crate::transport::CliTransport)) and the gateway
//! console (over its own HTTP transport).
//!
//! # Shape
//!
//! - [`ScreensStore`]: the screens' signals (catalog, installed, engines,
//!   host profile, the current job, filter state, selections). `Copy`.
//! - [`ScreensCtx`]: store + the command channel to the screens' OWN
//!   worker thread + the host label + the URL opener. `Clone`. Built once
//!   at mount with [`ScreensCtx::new`], which also spawns the worker.
//! - [`catalog()`] and [`engines()`]: the two screen builders. Each is a
//!   plain component `fn(Scope, &ScreensCtx) -> View`: mount it as a page
//!   of your `PageHost`. Entering a screen loads its data once; `r`
//!   re-reads.
//! - [`spawn_worker`] + [`schedule_job_poll`]: the worker lane and the
//!   job-poll helper (a timer thread re-sends [`ScreenCmd::PollJob`], the
//!   gateway console's `schedule_poll` pattern), exported for hosts that
//!   want to run the lane themselves.
//!
//! Every transport call runs on the worker; results come back as
//! closures posted through the engine's `WakeHandle` — the UI thread
//! never blocks.
//!
//! # Keys (contract H / G)
//!
//! | key | Models | Engines |
//! |---|---|---|
//! | `w` | download the selected artifact | — |
//! | `d` | delete (confirm; shows blockers) | — |
//! | `i` | — | install (confirm shows argv, host, sudo/UAC) |
//! | `o` | — | open the download page |
//! | `/` | filter | — |
//! | `f` | toggle fits-only | — |
//! | `e` | cycle engine filter | — |
//! | `v` | catalog ⇄ installed | — |
//! | `r` | refresh | refresh (probes local servers) |
//! | `c` | cancel the running job | cancel the running job |
//!
//! # Mounting (host crate)
//!
//! ```no_run
//! use std::sync::Arc;
//! use abstracttui::prelude::*;
//! use abstracttui::widgets::PageHost;
//! use abstractcore_console::screens::{self, ScreensCtx, ScreensOptions};
//! use abstractcore_console::transport::{CliTransport, ConsoleTransport};
//!
//! let mut app = App::new(Size::new(110, 32));
//! let overlays = app.overlays();
//! app.mount(move |cx| {
//!     let transport: Arc<dyn ConsoleTransport> =
//!         Arc::new(CliTransport::from_env().unwrap_or_else(|| CliTransport::new("abstractcore")));
//!     let sctx = ScreensCtx::new(cx, transport, overlays.clone(), ScreensOptions::default());
//!     let (a, b) = (sctx.clone(), sctx.clone());
//!     PageHost::new()
//!         .page("catalog", "9 Models", move |pcx| screens::catalog(pcx, &a))
//!         .page("engines", "0 Engines", move |pcx| screens::engines(pcx, &b))
//!         .view(cx)
//! })
//! .unwrap();
//! ```

pub mod catalog;
pub mod data;
pub mod engines;
mod worker;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use abstracttui::app::{ChoiceOption, ChoiceOutcome, ChoicePrompt, Modal, Overlays, Toast};
use abstracttui::prelude::*;
use abstracttui::widgets::Progress;

pub use catalog::catalog;
pub use data::{
    ArtifactRow, CatalogData, EngineRow, EnginesData, HostProfile, InstallPlan, InstalledData,
    InstalledRow, JobView,
};
pub use engines::engines;
pub use worker::{schedule_job_poll, spawn_worker};

use crate::transport::{ConsoleTransport, TransportError};
use crate::ui::util::{line, span, span_bold};

/// Stable page ids (contract principle 2): "Models" is `catalog`,
/// "Engines" is `engines` — never `models`/`runtimes`, which the
/// gateway console already uses for other things.
pub const CATALOG_ID: &str = "catalog";
pub const ENGINES_ID: &str = "engines";
pub const CATALOG_TITLE: &str = "Models";
pub const ENGINES_TITLE: &str = "Engines";

/// A remote read's honest state.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Remote<T> {
    #[default]
    NotAsked,
    Loading,
    Ready(T),
    Failed(TransportError),
}

impl<T> Remote<T> {
    pub fn ready(&self) -> Option<&T> {
        match self {
            Remote::Ready(t) => Some(t),
            _ => None,
        }
    }

    pub fn is_not_asked(&self) -> bool {
        matches!(self, Remote::NotAsked)
    }
}

/// Which list the Models screen shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CatalogView {
    /// Everything downloadable (contract C).
    #[default]
    Catalog,
    /// Everything already on disk (contract D).
    Installed,
}

/// The screens' signals. `Copy`: every field is a signal handle.
#[derive(Clone, Copy)]
pub struct ScreensStore {
    pub host: Signal<Remote<HostProfile>>,
    pub engines: Signal<Remote<EnginesData>>,
    pub catalog: Signal<Remote<CatalogData>>,
    pub installed: Signal<Remote<InstalledData>>,
    /// The latest job (running, or the last one to finish).
    pub job: Signal<Option<JobView>>,
    /// Free-text catalog filter (`/`).
    pub query: Signal<String>,
    /// Engine filter (`e` cycles through the catalog's providers).
    pub engine_filter: Signal<Option<String>>,
    /// `f`: only artifacts whose fit verdict is `fits`/`tight`.
    pub fits_only: Signal<bool>,
    pub view: Signal<CatalogView>,
    pub catalog_sel: Signal<usize>,
    pub installed_sel: Signal<usize>,
    pub engine_sel: Signal<usize>,
    /// Transient one-line notice. With
    /// [`ScreensOptions::notice`] this IS the host's own notice signal.
    pub notice: Signal<Option<String>>,
    /// Bumped on every catalog request; stale answers are dropped.
    pub catalog_gen: Signal<u64>,
    /// Engines seen in unfiltered catalog/installed answers — what `e`
    /// cycles through (a filtered answer only names one engine).
    pub providers_seen: Signal<Vec<String>>,
}

impl ScreensStore {
    /// Fresh signals; `notice` = the host's signal when it has one.
    pub fn create(cx: Scope, notice: Option<Signal<Option<String>>>) -> ScreensStore {
        ScreensStore {
            host: cx.signal(Remote::NotAsked),
            engines: cx.signal(Remote::NotAsked),
            catalog: cx.signal(Remote::NotAsked),
            installed: cx.signal(Remote::NotAsked),
            job: cx.signal(None),
            query: cx.signal(String::new()),
            engine_filter: cx.signal(None),
            fits_only: cx.signal(false),
            view: cx.signal(CatalogView::Catalog),
            catalog_sel: cx.signal(0),
            installed_sel: cx.signal(0),
            engine_sel: cx.signal(0),
            notice: notice.unwrap_or_else(|| cx.signal(None)),
            catalog_gen: cx.signal(0),
            providers_seen: cx.signal(Vec::new()),
        }
    }

    /// False once the scope that created these signals is disposed.
    pub fn alive(&self) -> bool {
        self.job.is_alive() && self.notice.is_alive() && self.catalog.is_alive()
    }

    pub fn job_active(&self) -> bool {
        self.job
            .with_untracked(|j| j.as_ref().is_some_and(JobView::is_active))
    }
}

/// Opens a URL on the machine running the console (`o`).
pub type Opener = Rc<dyn Fn(&str) -> Result<(), String>>;

/// Knobs for [`ScreensCtx::new`].
#[derive(Clone)]
pub struct ScreensOptions {
    /// Gap between job polls (default 1.5 s, the web console's cadence).
    pub poll_interval: Duration,
    /// How long the screens WATCH a job before handing it back (the job
    /// itself keeps running; default 60 min).
    pub poll_limit: Duration,
    /// Share the host's notice signal (its footer + toast already render
    /// it). `None` = the screens own one and toast it themselves.
    pub notice: Option<Signal<Option<String>>>,
    /// `o` opener; `None` = the system opener (`open` / `xdg-open` /
    /// `explorer`).
    pub opener: Option<Opener>,
}

impl Default for ScreensOptions {
    fn default() -> Self {
        ScreensOptions {
            poll_interval: Duration::from_millis(1500),
            poll_limit: Duration::from_secs(60 * 60),
            notice: None,
            opener: None,
        }
    }
}

/// Commands for the screens' worker lane.
#[derive(Clone, Debug)]
pub enum ScreenCmd {
    LoadHost,
    LoadEngines {
        probe: bool,
    },
    LoadCatalog {
        q: String,
        engine: Option<String>,
        fits_only: bool,
        generation: u64,
    },
    LoadInstalled,
    Download {
        provider: String,
        artifact: String,
    },
    Delete {
        provider: String,
        artifact: String,
        force: bool,
    },
    Install {
        engine: String,
        dry_run: bool,
    },
    PollJob(JobPoll),
    Cancel {
        job_id: String,
    },
}

/// One job watch, carried across polls.
#[derive(Clone, Debug)]
pub struct JobPoll {
    pub job_id: String,
    /// When the watch began (the poll limit counts from here).
    pub since: Instant,
    /// Consecutive failed polls (three in a row end the watch).
    pub errors: u32,
}

/// Everything a screen needs: store, worker lane, host label, opener.
#[derive(Clone)]
pub struct ScreensCtx {
    pub store: ScreensStore,
    pub tx: Sender<ScreenCmd>,
    pub overlays: Overlays,
    /// Where installs/deletes run ("this machine (mbp)", "gateway.lan").
    pub host_label: Rc<str>,
    opener: Opener,
    modal: Rc<RefCell<Option<Modal>>>,
}

impl ScreensCtx {
    /// Create the store, spawn the worker over `transport`, and (without
    /// a shared notice signal) install the toast effect. Call once, in
    /// the mount scope, on the UI thread.
    pub fn new(
        cx: Scope,
        transport: Arc<dyn ConsoleTransport>,
        overlays: Overlays,
        options: ScreensOptions,
    ) -> ScreensCtx {
        let own_notice = options.notice.is_none();
        let store = ScreensStore::create(cx, options.notice);
        let (tx, rx) = mpsc::channel::<ScreenCmd>();
        let host_label: Rc<str> = Rc::from(transport.host_label());
        let wake = abstracttui::reactive::wake_handle();
        let _worker = spawn_worker(
            transport,
            store,
            wake,
            tx.clone(),
            rx,
            worker::WorkerOptions {
                poll_interval: options.poll_interval,
                poll_limit: options.poll_limit,
            },
        );
        if own_notice {
            let overlays = overlays.clone();
            cx.effect(move || {
                if let Some(n) = store.notice.get() {
                    let viewport = abstracttui::app::use_viewport(cx).get_untracked();
                    Toast::show(&overlays, cx, viewport, n, Duration::from_secs(4));
                }
            });
        }
        ScreensCtx {
            store,
            tx,
            overlays,
            host_label,
            opener: options.opener.unwrap_or_else(|| Rc::new(system_open)),
            modal: Rc::new(RefCell::new(None)),
        }
    }

    pub fn send(&self, cmd: ScreenCmd) {
        let _ = self.tx.send(cmd);
    }

    fn notice(&self, msg: impl Into<String>) {
        self.store.notice.set(Some(msg.into()));
    }

    /// Load the Models screen's data if it was never asked for.
    pub fn ensure_catalog(&self) {
        let s = self.store;
        if s.host.with_untracked(Remote::is_not_asked) {
            s.host.set(Remote::Loading);
            self.send(ScreenCmd::LoadHost);
        }
        if s.catalog.with_untracked(Remote::is_not_asked) {
            self.reload_catalog();
        }
        if s.installed.with_untracked(Remote::is_not_asked) {
            s.installed.set(Remote::Loading);
            self.send(ScreenCmd::LoadInstalled);
        }
    }

    /// Load the Engines screen's data if it was never asked for.
    pub fn ensure_engines(&self) {
        let s = self.store;
        if s.engines.with_untracked(Remote::is_not_asked) {
            s.engines.set(Remote::Loading);
            self.send(ScreenCmd::LoadEngines { probe: false });
        }
        if s.host.with_untracked(Remote::is_not_asked) {
            s.host.set(Remote::Loading);
            self.send(ScreenCmd::LoadHost);
        }
    }

    /// Re-query the catalog with the current filters.
    pub fn reload_catalog(&self) {
        let s = self.store;
        let generation = s.catalog_gen.get_untracked() + 1;
        s.catalog_gen.set(generation);
        s.catalog.set(Remote::Loading);
        self.send(ScreenCmd::LoadCatalog {
            q: s.query.get_untracked(),
            engine: s.engine_filter.get_untracked(),
            fits_only: s.fits_only.get_untracked(),
            generation,
        });
    }

    /// `r` on Models: host profile, catalog and installed list.
    pub fn refresh_catalog(&self) {
        self.store.host.set(Remote::Loading);
        self.send(ScreenCmd::LoadHost);
        self.reload_catalog();
        self.store.installed.set(Remote::Loading);
        self.send(ScreenCmd::LoadInstalled);
    }

    /// `r` on Engines: a PROBING status read (contacts local servers).
    pub fn refresh_engines(&self) {
        self.store.engines.set(Remote::Loading);
        self.send(ScreenCmd::LoadEngines { probe: true });
    }

    /// Single-flight door for the three verbs.
    fn job_free(&self) -> bool {
        if let Some(j) = self.store.job.get_untracked().filter(JobView::is_active) {
            self.notice(format!(
                "{} {} is still running — c cancels it",
                j.verb(),
                j.subject()
            ));
            return false;
        }
        true
    }

    /// Mark a verb as started (the strip shows it before the backend
    /// answers) and send it.
    fn start(&self, placeholder: JobView, cmd: ScreenCmd) {
        self.store.job.set(Some(placeholder));
        self.send(cmd);
    }

    /// `w`: start downloading one artifact.
    pub fn download(&self, provider: &str, artifact: &str) {
        if !self.job_free() {
            return;
        }
        self.start(
            JobView {
                kind: "download".into(),
                status: "queued".into(),
                provider: Some(provider.into()),
                artifact: Some(artifact.into()),
                message: Some("starting…".into()),
                ..JobView::default()
            },
            ScreenCmd::Download {
                provider: provider.into(),
                artifact: artifact.into(),
            },
        );
    }

    /// `d` (after the confirm): delete one artifact.
    pub fn delete(&self, provider: &str, artifact: &str, force: bool) {
        if !self.job_free() {
            return;
        }
        self.start(
            JobView {
                kind: "delete".into(),
                status: "queued".into(),
                provider: Some(provider.into()),
                artifact: Some(artifact.into()),
                message: Some("deleting…".into()),
                ..JobView::default()
            },
            ScreenCmd::Delete {
                provider: provider.into(),
                artifact: artifact.into(),
                force,
            },
        );
    }

    /// `i` (after the confirm): install an engine, or dry-run it.
    pub fn install(&self, engine: &str, dry_run: bool) {
        if !self.job_free() {
            return;
        }
        self.start(
            JobView {
                kind: "engine_install".into(),
                status: "queued".into(),
                engine: Some(engine.into()),
                dry_run,
                message: Some(if dry_run {
                    "asking what would run…".into()
                } else {
                    "starting the installer…".into()
                }),
                ..JobView::default()
            },
            ScreenCmd::Install {
                engine: engine.into(),
                dry_run,
            },
        );
    }

    /// `c`: cancel the running job.
    pub fn cancel(&self) {
        match self.store.job.get_untracked() {
            Some(j) if j.is_active() && !j.job_id.is_empty() => {
                self.notice(format!("cancelling {} {}…", j.verb(), j.subject()));
                self.send(ScreenCmd::Cancel { job_id: j.job_id });
            }
            Some(j) if j.is_active() => {
                self.notice("the job has not started yet — press c again in a moment")
            }
            _ => self.notice("no job is running"),
        }
    }

    /// `o`: open a URL with the configured opener.
    pub fn open_url(&self, url: &str) {
        match (self.opener)(url) {
            Ok(()) => self.notice(format!("opened {url}")),
            Err(e) => self.notice(format!("could not open {url}: {e} — open it yourself")),
        }
    }

    /// Close the one modal these screens may hold (the filter input).
    fn close_modal(&self) {
        if let Some(m) = self.modal.borrow_mut().take() {
            m.close();
        }
    }
}

/// The system opener: fixed argv, the URL as ONE argument, never a shell.
pub fn system_open(url: &str) -> Result<(), String> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("only http(s) links are opened".into());
    }
    let (prog, args): (&str, Vec<&str>) = if cfg!(target_os = "macos") {
        ("open", vec![url])
    } else if cfg!(target_os = "windows") {
        ("explorer", vec![url])
    } else {
        ("xdg-open", vec![url])
    };
    std::process::Command::new(prog)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("{prog}: {e}"))
}

// ---------------------------------------------------------------------
// Shared UI pieces
// ---------------------------------------------------------------------

type ThemeSig = Signal<&'static abstracttui::theme::Theme>;

/// The job strip: two rows while a job exists (progress + last log
/// line / CLI equivalent), zero otherwise.
pub fn job_strip(sctx: &ScreensCtx, theme: ThemeSig) -> View {
    let store = sctx.store;
    dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        let Some(j) = store.job.get() else {
            return Element::new().style(LayoutStyle::default().h(0)).build();
        };
        let (glyph, ink) = match j.status.as_str() {
            "completed" => ("✓", t.ok),
            "failed" => ("✗", t.error),
            "cancelled" => ("⊘", t.warn),
            _ => ("⟳", t.info),
        };
        let mut facts = vec![span_bold(
            format!(" {glyph} {} {} ", j.verb(), j.subject()),
            ink,
        )];
        let mut info = Vec::new();
        if let Some(p) = j.percent {
            info.push(format!("{p:.0}%"));
        }
        if let (Some(d), Some(tot)) = (j.downloaded_bytes, j.total_bytes) {
            info.push(format!(
                "{} / {}",
                data::bytes_label(Some(d)),
                data::bytes_label(Some(tot))
            ));
        }
        info.push(j.status.clone());
        facts.push(span(info.join(" · "), t.text_muted));
        let bar = match j.fraction() {
            Some(fr) => Progress::new(fr)
                .layout(LayoutStyle::default().w(24).h(1).shrink(0.0))
                .element(&t)
                .build(),
            None => Element::new()
                .style(LayoutStyle::default().w(0).h(1))
                .build(),
        };
        let first = Element::new()
            .style(LayoutStyle::row().gap(1).h(1).shrink(0.0))
            .child(bar)
            .child(line(facts))
            .build();
        let detail = if let Some(e) = j.error.as_ref().filter(|_| j.status == "failed") {
            span(format!("   {e}"), t.error)
        } else if let Some(l) = j.log_tail.last().or(j.message.as_ref()) {
            span(format!("   {l}"), t.text_faint)
        } else if let Some(c) = &j.cli_equivalent {
            span(format!("   same as: {c}"), t.text_faint)
        } else {
            span(String::new(), t.text_faint)
        };
        let mut second = vec![detail];
        if j.is_active() {
            second.push(span("  · c cancels", t.text_faint));
        }
        Element::new()
            .style(LayoutStyle::column().shrink(0.0))
            .child(first)
            .child(line(second))
            .build()
    })
}

/// One honest line for a remote read that is not ready.
pub fn remote_line<T>(t: &TokenSet, what: &str, r: &Remote<T>) -> Option<View> {
    match r {
        Remote::Ready(_) => None,
        Remote::NotAsked => Some(line(vec![span(
            format!(" {what}: not read yet"),
            t.text_muted,
        )])),
        Remote::Loading => Some(line(vec![span(format!(" ⟳ reading {what}…"), t.info)])),
        Remote::Failed(e) => Some(line(vec![
            span_bold(format!(" {what} {}: ", e.headline()), t.error),
            span(e.message.clone(), t.text),
        ])),
    }
}

/// `d`'s confirm. Blockers change the question: `loaded` and
/// `shared_cache:*` can be overridden (force, danger-tinted), a
/// `remote_engine` cannot be deleted from here at all.
pub fn confirm_delete(
    cx: Scope,
    sctx: &ScreensCtx,
    provider: &str,
    artifact: &str,
    size: Option<u64>,
    blockers: Vec<String>,
) {
    if blockers.iter().any(|b| b == "remote_engine") {
        sctx.notice(format!(
            "{provider} {artifact} lives on a remote engine — delete it on that host"
        ));
        return;
    }
    let size = data::bytes_label(size);
    let mut body_lines: Vec<(String, bool)> = vec![
        (format!("provider  {provider}"), false),
        (format!("artifact  {artifact}"), false),
        (format!("frees     {size}"), false),
        (format!("runs on   {}", sctx.host_label), false),
    ];
    let forced = !blockers.is_empty();
    for b in &blockers {
        body_lines.push((format!("blocked   {}", blocker_text(b)), true));
    }
    let question = if forced {
        format!("Delete {artifact}? It is blocked — deleting anyway forces it.")
    } else {
        format!("Delete {artifact} from {provider}? This removes the files from disk.")
    };
    let label = if forced {
        "Delete anyway (force)"
    } else {
        "Delete"
    };
    let rows = body_lines.len() as i32;
    let theme = use_theme(cx);
    let prompt = ChoicePrompt::new(question)
        .body(move |_mcx| {
            let t = theme.get_untracked().tokens;
            let mut col = Element::new().style(LayoutStyle::column());
            for (l, warn) in body_lines {
                col = col.child(line(vec![span(l, if warn { t.warn } else { t.text })]));
            }
            col.build()
        })
        .body_rows(rows)
        .option_with(ChoiceOption::new("delete", label).danger(true))
        .option("keep", "Keep it")
        .initial("keep");
    let sctx2 = sctx.clone();
    let (p, a) = (provider.to_string(), artifact.to_string());
    prompt
        .on_resolve(move |outcome| {
            if let ChoiceOutcome::Answered(ans) = outcome {
                if ans.selected.iter().any(|s| s == "delete") {
                    sctx2.delete(&p, &a, forced);
                    return;
                }
            }
            sctx2.notice(format!("kept {a}"));
        })
        .open(cx);
}

fn blocker_text(b: &str) -> String {
    if b == "loaded" {
        return "loaded in memory right now (unload it, or force)".to_string();
    }
    if let Some(rest) = b.strip_prefix("shared_cache:") {
        return format!("shared cache — the same files serve {rest}");
    }
    b.to_string()
}

/// The host-safety notes the install confirm prints (principle 4).
pub fn install_notes(engine: &EngineRow, host_os: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    let method = engine.install.method.as_deref().unwrap_or("");
    let argv0 = engine
        .install
        .argv
        .first()
        .map(String::as_str)
        .unwrap_or("");
    let script_like = engine.install.argv.iter().any(|a| a.ends_with(".sh"));
    if argv0 == "sudo" || (method == "script" && host_os != Some("windows")) || script_like {
        out.push(
            "needs administrator rights (sudo): the installer may stop to ask for a password — \
             if it fails, run the command above yourself in a terminal"
                .to_string(),
        );
    }
    if method == "winget" || host_os == Some("windows") {
        out.push("Windows may show a UAC prompt on the host — approve it there".to_string());
    }
    match method {
        "brew" => out.push("Homebrew install — no sudo needed".to_string()),
        "pip" => out.push(
            "installs into the Python environment abstractcore runs from (no sudo)".to_string(),
        ),
        _ => {}
    }
    if let Some(n) = &engine.install.notes {
        out.push(n.clone());
    }
    if let Some(b) = engine.install.estimated_bytes {
        out.push(format!("downloads about {}", data::bytes_label(Some(b))));
    }
    out
}

/// `i`'s confirm: shows the exact argv, the host it runs on and the
/// sudo/UAC notes; offers Install, a dry run, or cancel (default).
pub fn confirm_install(cx: Scope, sctx: &ScreensCtx, engine: &EngineRow, host_os: Option<&str>) {
    let command = engine.install.argv.join(" ");
    let mut body_lines: Vec<(String, bool)> = vec![
        (format!("command   {command}"), false),
        (format!("runs on   {}", sctx.host_label), false),
        (
            format!(
                "method    {}",
                engine.install.method.as_deref().unwrap_or("unknown")
            ),
            false,
        ),
    ];
    for n in install_notes(engine, host_os) {
        body_lines.push((format!("note      {n}"), true));
    }
    let width = body_lines
        .iter()
        .map(|(l, _)| abstracttui::text::width(l) as i32)
        .max()
        .unwrap_or(40)
        .min(100);
    let rows = body_lines.len() as i32;
    let theme = use_theme(cx);
    let prompt = ChoicePrompt::new(format!("Install {} on {}?", engine.name, sctx.host_label))
        .body(move |_mcx| {
            let t = theme.get_untracked().tokens;
            let mut col = Element::new().style(LayoutStyle::column());
            for (l, warn) in body_lines {
                col = col.child(line(vec![span(l, if warn { t.warn } else { t.text })]));
            }
            col.build()
        })
        .body_rows(rows)
        .body_width(width + 2)
        .option_with(ChoiceOption::new("install", format!("Install {}", engine.name)).danger(true))
        .option("dry", "Dry run — show what would run")
        .option("cancel", "Cancel")
        .initial("cancel");
    let sctx2 = sctx.clone();
    let id = engine.id.clone();
    prompt
        .on_resolve(move |outcome| {
            if let ChoiceOutcome::Answered(ans) = outcome {
                match ans.selected.first().map(String::as_str) {
                    Some("install") => sctx2.install(&id, false),
                    Some("dry") => sctx2.install(&id, true),
                    _ => sctx2.notice("install cancelled — nothing ran"),
                }
            }
        })
        .open(cx);
}

/// `/`: a one-field modal; Enter applies (empty clears), Esc keeps.
pub fn open_filter(cx: Scope, sctx: &ScreensCtx) {
    sctx.close_modal();
    let viewport = abstracttui::app::use_viewport(cx).get_untracked();
    let value = cx.signal(sctx.store.query.get_untracked());
    let theme = use_theme(cx);
    let slot = sctx.modal.clone();
    let apply_ctx = sctx.clone();
    let esc_ctx = sctx.clone();
    let size = Size::new(56.min(viewport.w - 4).max(20), 5);
    let modal = Modal::open(&sctx.overlays, cx, viewport, size, move |mcx| {
        let t = theme.get_untracked().tokens;
        let apply = Rc::new(move || {
            let q = value.get_untracked().trim().to_string();
            apply_ctx.close_modal();
            apply_ctx.store.query.set(q.clone());
            apply_ctx.store.catalog_sel.set(0);
            apply_ctx.store.installed_sel.set(0);
            apply_ctx.reload_catalog();
            apply_ctx.notice(if q.is_empty() {
                "filter cleared".to_string()
            } else {
                format!("filter: {q}")
            });
        });
        Element::new()
            .style(LayoutStyle::column().grow(1.0))
            .shortcut(KeyChord::plain(Key::Escape), move |_| esc_ctx.close_modal())
            .child(line(vec![span_bold(" Filter models", t.accent)]))
            .child(
                TextInput::new()
                    .layout(LayoutStyle::default().grow(1.0).h(1))
                    .value(value)
                    .placeholder("name, family, tag or artifact — empty clears")
                    .on_submit(move |_| apply())
                    .view(mcx),
            )
            .child(line(vec![span(
                " Enter applies · Esc keeps the current filter",
                t.text_faint,
            )]))
            .build()
    });
    *slot.borrow_mut() = Some(modal);
}

/// Did the viewer's last refusal carry reasons? One line for a toast.
pub fn error_notice(verb: &str, subject: &str, e: &TransportError) -> String {
    let reasons = e.reasons();
    if reasons.is_empty() {
        format!("✗ {verb} {subject} {}: {}", e.headline(), e.message)
    } else {
        format!(
            "✗ {verb} {subject} {}: {} ({})",
            e.headline(),
            e.message,
            reasons.join(", ")
        )
    }
}
