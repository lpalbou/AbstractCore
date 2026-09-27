//! The screens' worker lane: one serial thread that owns every
//! transport call, posting results back to the UI thread.
//!
//! Job watching is the gateway console's `schedule_poll` pattern: after
//! each poll of a still-running job, a throwaway timer thread sleeps the
//! poll interval and re-sends [`ScreenCmd::PollJob`] — the lane stays
//! free for whatever the operator does next.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use abstracttui::reactive::WakeHandle;
use serde_json::Value;

use super::data::{
    download_jobs_from_value, text_default_from_value, CatalogData, EnginesData, HostProfile,
    InstallPlans, InstalledData, JobView, LocationPlan,
};
use super::{
    drop_placeholder, error_notice, is_download_kind, upsert_job, JobPoll, Remote, ScreenCmd,
    ScreensStore,
};
use crate::transport::{
    ConsoleTransport, InstallLocation, ServerAction, TransportError, TransportErrorKind,
};

/// Poll cadence and watch limit (from `ScreensOptions`).
#[derive(Clone, Copy, Debug)]
pub struct WorkerOptions {
    pub poll_interval: Duration,
    pub poll_limit: Duration,
}

/// Re-send one [`ScreenCmd::PollJob`] after `delay`, from a timer thread
/// that touches nothing but the channel. A closed channel (console
/// quitting) just drops the timer.
pub fn schedule_job_poll(tx: &Sender<ScreenCmd>, poll: JobPoll, delay: Duration) {
    let tx = tx.clone();
    if delay.is_zero() {
        let _ = tx.send(ScreenCmd::PollJob(poll));
        return;
    }
    let _ = std::thread::Builder::new()
        .name("screens-job-poll".into())
        .spawn(move || {
            std::thread::sleep(delay);
            let _ = tx.send(ScreenCmd::PollJob(poll));
        });
}

/// Spawn the lane. `tx` is the sender half of `rx` (the poll helper
/// re-sends through it). The thread ends when every sender is gone, or
/// when a posted result finds the store's signals disposed (the app
/// that owned them is gone — the lane holds a sender itself, so the
/// channel alone would never close).
pub fn spawn_worker(
    transport: Arc<dyn ConsoleTransport>,
    store: ScreensStore,
    wake: WakeHandle,
    tx: Sender<ScreenCmd>,
    rx: Receiver<ScreenCmd>,
    options: WorkerOptions,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("screens-worker".into())
        .spawn(move || {
            let mut w = Worker {
                transport,
                store,
                wake,
                tx,
                options,
                finished: HashSet::new(),
                watching: HashSet::new(),
                paused_seen: HashSet::new(),
                stop: Arc::new(AtomicBool::new(false)),
            };
            loop {
                if w.stop.load(Ordering::Relaxed) {
                    break;
                }
                let cmd = match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(cmd) => cmd,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => break,
                };
                let label = format!("{cmd:?}");
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| w.handle(cmd)));
                if outcome.is_err() {
                    let msg = format!("internal error in the models worker while handling {label}");
                    w.post(move |s| s.notice.set(Some(msg)));
                }
            }
        })
        .expect("spawn screens worker")
}

struct Worker {
    transport: Arc<dyn ConsoleTransport>,
    store: ScreensStore,
    wake: WakeHandle,
    tx: Sender<ScreenCmd>,
    options: WorkerOptions,
    /// Jobs already reported — a cancel and a poll racing to the same
    /// terminal state must produce ONE toast.
    finished: HashSet<String>,
    /// Jobs with a live poll chain (one chain per job, never two).
    watching: HashSet<String>,
    /// `job_id/state` pauses already announced (one toast per pause).
    paused_seen: HashSet<String>,
    /// Set (on the UI thread) when a post finds the store disposed.
    stop: Arc<AtomicBool>,
}

impl Worker {
    /// Run `f` on the UI thread — only if the store still exists. A
    /// result landing after its app was torn down (quit, or the next
    /// test's app on the same thread) is dropped, and the lane stops.
    fn post(&self, f: impl FnOnce(ScreensStore) + Send + 'static) {
        let s = self.store;
        let stop = self.stop.clone();
        self.wake.post(move || {
            if s.alive() {
                f(s)
            } else {
                stop.store(true, Ordering::Relaxed);
            }
        });
    }

    fn notice(&self, msg: String) {
        self.post(move |s| s.notice.set(Some(msg)));
    }

    fn handle(&mut self, cmd: ScreenCmd) {
        match cmd {
            ScreenCmd::LoadHost => {
                let r = remote(self.transport.host_profile(), HostProfile::from_value);
                self.post(move |s| s.host.set(r));
            }
            ScreenCmd::LoadEngines { probe } => {
                let r = remote(
                    self.transport.engines_status(probe),
                    EnginesData::from_value,
                );
                // A row naming a live install job this console does not
                // watch (started elsewhere, or before a restart — a
                // PAUSED one included): watch it, so its state and its
                // continue verb show up.
                if let Remote::Ready(d) = &r {
                    let ids: Vec<String> = d
                        .engines
                        .iter()
                        .filter_map(|e| e.active_job.as_ref().map(|j| j.job_id.clone()))
                        .collect();
                    for id in ids {
                        self.adopt(&id);
                    }
                }
                self.post(move |s| {
                    if let Remote::Ready(d) = &r {
                        let n = d.engines.len();
                        s.engine_sel.update(|i| *i = (*i).min(n.saturating_sub(1)));
                    }
                    s.engines.set(r)
                });
            }
            ScreenCmd::LoadCatalog {
                q,
                engine,
                fits_only,
                generation,
                hub,
            } => {
                let answer = if hub {
                    self.transport
                        .models_catalog_hub(&q, engine.as_deref(), fits_only)
                } else {
                    self.transport
                        .models_catalog(&q, engine.as_deref(), fits_only)
                };
                let r = remote(answer, CatalogData::from_value);
                self.post(move |s| {
                    if s.catalog_gen.get_untracked() != generation {
                        return; // a newer filter already asked
                    }
                    if let Remote::Ready(d) = &r {
                        // The catalog carries the profile it was fitted
                        // against — a host read that failed on its own
                        // still gets an answer.
                        if let Some(h) = &d.host {
                            if s.host.with_untracked(|x| x.ready().is_none()) {
                                s.host.set(Remote::Ready(h.clone()));
                            }
                        }
                        let n = d.rows.len();
                        s.catalog_sel.update(|i| *i = (*i).min(n.saturating_sub(1)));
                        remember_providers(&s, d.providers.iter());
                    }
                    s.catalog.set(r);
                });
            }
            ScreenCmd::LoadInstalled => {
                let r = remote(
                    self.transport.models_installed(None),
                    InstalledData::from_value,
                );
                self.post(move |s| {
                    if let Remote::Ready(d) = &r {
                        remember_providers(&s, d.rows.iter().map(|r| &r.provider));
                        let n = d.rows.len();
                        s.installed_sel
                            .update(|i| *i = (*i).min(n.saturating_sub(1)));
                    }
                    s.installed.set(r)
                });
            }
            ScreenCmd::LoadDownloads => match self.transport.download_jobs() {
                Ok(v) => {
                    let jobs = download_jobs_from_value(&v);
                    let n = jobs.len();
                    let mut live = Vec::new();
                    let mut done = Vec::new();
                    for j in jobs {
                        if j.is_active() && !self.finished.contains(&j.job_id) {
                            live.push(j);
                        } else {
                            done.push(j);
                        }
                    }
                    for j in &live {
                        // EVERY live download gets its own poll chain on
                        // the download route: a "download all" group
                        // (only that route knows it) and each of its
                        // children too. Nothing re-reads the feed on its
                        // own, so an unwatched job would stay "running"
                        // here forever — and hold the quit guard.
                        self.watch(&j.job_id, true);
                    }
                    self.post(move |s| {
                        // Live jobs always show; finished ones only when
                        // this console already listed them (history it
                        // never showed stays out — the web rule).
                        for j in live {
                            upsert_job(&s, j);
                        }
                        for j in done {
                            let known = s
                                .jobs
                                .with_untracked(|v| v.iter().any(|x| x.job_id == j.job_id));
                            if known {
                                s.jobs.update(|v| {
                                    if let Some(x) = v.iter_mut().find(|x| x.job_id == j.job_id) {
                                        *x = j;
                                    }
                                });
                            }
                        }
                        s.feed.set(Remote::Ready(n));
                    });
                }
                Err(e) => self.post(move |s| s.feed.set(Remote::Failed(e))),
            },
            ScreenCmd::LoadTextDefault => {
                let r = remote(
                    self.transport.capability_defaults(),
                    text_default_from_value,
                );
                self.post(move |s| s.text_default.set(r));
            }
            ScreenCmd::SetTextDefault { provider, model } => {
                match self.transport.set_text_default(&provider, &model) {
                    Ok(doc) => {
                        // Verify from the answer (a FRESH read of the
                        // defaults): the write counts only when the route
                        // now says what was asked.
                        let now = text_default_from_value(&doc);
                        let ok = now.as_ref() == Some(&(provider.clone(), model.clone()));
                        let msg = if ok {
                            format!("✓ default text model: {provider} · {model}")
                        } else {
                            format!(
                                "✗ the default text model was not changed: the route now says {}",
                                now.as_ref()
                                    .map(|(p, m)| format!("{p} · {m}"))
                                    .unwrap_or_else(|| "nothing".into())
                            )
                        };
                        self.post(move |s| {
                            s.text_default.set(Remote::Ready(now));
                            s.notice.set(Some(msg));
                        });
                    }
                    Err(e) => {
                        let msg = error_notice("set default", &format!("{provider} {model}"), &e);
                        self.notice(msg);
                    }
                }
            }
            ScreenCmd::LoadPlans { engine } => {
                // Two dry runs: nothing runs, and a dry run needs no
                // install permission (the gateway's rule).
                let one = |loc: InstallLocation| -> Result<LocationPlan, TransportError> {
                    let v = self.transport.engine_install_at(&engine, true, loc)?;
                    LocationPlan::from_dry_run(&v)
                        .ok_or_else(|| TransportError::protocol("the dry run returned no `plan`"))
                };
                let r = match (one(InstallLocation::User), one(InstallLocation::System)) {
                    (Ok(user), Ok(system)) => Remote::Ready(InstallPlans { user, system }),
                    (Err(e), _) | (_, Err(e)) => Remote::Failed(e),
                };
                self.post(move |s| {
                    s.plans.update(|m| {
                        m.insert(engine, r);
                    })
                });
            }
            ScreenCmd::Adopt { job_id } => self.adopt(&job_id),
            ScreenCmd::Continue { job_id, action } => {
                match self
                    .transport
                    .engine_job_continue(&job_id, action.as_deref())
                {
                    Ok(v) => {
                        let view = JobView::from_value(&v);
                        self.paused_seen
                            .retain(|k| !k.starts_with(&format!("{job_id}/")));
                        let msg = format!(
                            "continuing {} {}: {}",
                            view.verb(),
                            view.subject(),
                            view.message.as_deref().unwrap_or("working")
                        );
                        if view.is_active() {
                            self.post(move |s| {
                                upsert_job(&s, view);
                                s.notice.set(Some(msg));
                            });
                            self.watch(&job_id, false);
                        } else {
                            self.finish(view);
                        }
                    }
                    Err(e) => self.notice(error_notice("continue", &job_id, &e)),
                }
            }
            ScreenCmd::Server { engine, action } => {
                let verb = action.as_str();
                match self.transport.engine_server(&engine, action) {
                    Ok(v) => {
                        let running = v.get("running").and_then(Value::as_bool);
                        let ok = match action {
                            ServerAction::Start => running != Some(false),
                            ServerAction::Stop => running != Some(true),
                        };
                        let said = v
                            .get("message")
                            .and_then(Value::as_str)
                            .filter(|m| !m.trim().is_empty())
                            .map(|m| format!(" — {m}"))
                            .unwrap_or_default();
                        let msg = match (action, ok) {
                            (ServerAction::Start, true) => format!("✓ {engine} is running{said}"),
                            (ServerAction::Start, false) => {
                                format!("⚠ {engine} was started but is not answering yet{said}")
                            }
                            (ServerAction::Stop, true) => format!("✓ {engine} is stopped{said}"),
                            (ServerAction::Stop, false) => {
                                format!("⚠ {engine} is still running{said}")
                            }
                        };
                        self.notice(msg);
                    }
                    Err(e) => self.notice(error_notice(verb, &engine, &e)),
                }
                // Verify with a fresh PROBING read either way.
                self.post(|s| s.engines.set(Remote::Loading));
                let _ = self.tx.send(ScreenCmd::LoadEngines { probe: true });
            }
            ScreenCmd::Download {
                provider,
                artifact,
                expected_bytes,
            } => {
                let res = self
                    .transport
                    .start_download(&provider, &artifact, expected_bytes);
                let like = JobView {
                    kind: "download".into(),
                    provider: Some(provider.clone()),
                    artifact: Some(artifact.clone()),
                    ..JobView::default()
                };
                self.started("download", &format!("{provider} {artifact}"), res, like);
            }
            ScreenCmd::Delete {
                provider,
                artifact,
                force,
            } => {
                let res = self.transport.delete_model(&provider, &artifact, force);
                let like = JobView {
                    kind: "delete".into(),
                    provider: Some(provider.clone()),
                    artifact: Some(artifact.clone()),
                    ..JobView::default()
                };
                self.started("delete", &format!("{provider} {artifact}"), res, like);
            }
            ScreenCmd::Install {
                engine,
                dry_run,
                location,
            } => {
                let res = self.transport.engine_install_at(&engine, dry_run, location);
                let like = JobView {
                    kind: "engine_install".into(),
                    engine: Some(engine.clone()),
                    ..JobView::default()
                };
                self.started("install", &engine, res, like);
            }
            ScreenCmd::PollJob(poll) => self.poll(poll),
            ScreenCmd::Cancel { job_id, download } => match if download {
                self.transport.cancel_download(&job_id)
            } else {
                self.transport.cancel_job(&job_id)
            } {
                Ok(v) => {
                    let view = JobView::from_value(&v);
                    if view.is_active() {
                        self.post(move |s| upsert_job(&s, view));
                        self.watch(&job_id, download);
                    } else {
                        self.finish(view);
                    }
                }
                Err(e) => {
                    let msg = format!("✗ cancel {job_id} {}: {}", e.headline(), e.message);
                    self.post(move |s| s.notice.set(Some(msg)));
                }
            },
        }
    }

    /// Start a poll chain for `job_id` unless one runs (or it ended).
    /// `download` = poll it on the download route.
    fn watch(&mut self, job_id: &str, download: bool) {
        if job_id.is_empty()
            || self.finished.contains(job_id)
            || !self.watching.insert(job_id.to_string())
        {
            return;
        }
        schedule_job_poll(
            &self.tx,
            JobPoll::new(job_id, download),
            self.options.poll_interval,
        );
    }

    /// Read a job learned from elsewhere once, show it, and watch it.
    fn adopt(&mut self, job_id: &str) {
        if job_id.is_empty() || self.watching.contains(job_id) || self.finished.contains(job_id) {
            return;
        }
        match self.transport.job(job_id) {
            Ok(v) => {
                let mut view = JobView::from_value(&v);
                view.job_id = job_id.to_string();
                if view.is_active() {
                    let download = is_download_kind(&view.kind);
                    self.announce_pause(&view);
                    self.post(move |s| upsert_job(&s, view));
                    self.watch(job_id, download);
                }
            }
            Err(e) => {
                let msg = format!(
                    "could not read job {job_id} ({}: {})",
                    e.headline(),
                    e.message
                );
                self.notice(msg);
            }
        }
    }

    /// One toast per pause: the job now waits for a person.
    fn announce_pause(&mut self, view: &JobView) {
        if !view.is_paused() {
            return;
        }
        let key = format!("{}/{}", view.job_id, view.state.as_deref().unwrap_or(""));
        if self.paused_seen.insert(key) {
            let msg = view.outcome_line();
            self.notice(msg);
        }
    }

    /// A verb answered: show the job; watch it if it is still running.
    fn started(
        &mut self,
        verb: &str,
        subject: &str,
        res: Result<Value, TransportError>,
        like: JobView,
    ) {
        match res {
            Ok(v) => {
                let view = JobView::from_value(&v);
                if view.is_active() && !view.job_id.is_empty() {
                    let id = view.job_id.clone();
                    let download = is_download_kind(&like.kind);
                    self.announce_pause(&view);
                    self.post(move |s| {
                        // A joined job answers with an id already listed:
                        // the placeholder goes, the one row stays.
                        drop_placeholder(&s, &like);
                        upsert_job(&s, view);
                    });
                    self.watch(&id, download);
                } else {
                    self.finish(view);
                    self.post(move |s| drop_placeholder(&s, &like));
                }
            }
            Err(e) => {
                let msg = error_notice(verb, subject, &e);
                self.post(move |s| {
                    // Drop the "starting…" placeholder: nothing runs.
                    drop_placeholder(&s, &like);
                    s.notice.set(Some(msg));
                });
            }
        }
    }

    fn poll(&mut self, poll: JobPoll) {
        if self.finished.contains(&poll.job_id) {
            self.watching.remove(&poll.job_id);
            return;
        }
        let answer = if poll.download {
            self.transport.download_job(&poll.job_id)
        } else {
            self.transport.job(&poll.job_id)
        };
        match answer {
            Ok(v) => {
                let mut view = JobView::from_value(&v);
                // The answer is about the job ASKED for: a backend that
                // echoes some upstream id must not fork the row.
                view.job_id = poll.job_id.clone();
                if !view.is_active() {
                    self.watching.remove(&poll.job_id);
                    self.finish(view);
                    return;
                }
                self.announce_pause(&view);
                if poll.since.elapsed() > self.options.poll_limit {
                    self.watching.remove(&poll.job_id);
                    let msg = format!(
                        "{} {} is still running after {} min — it keeps going; r re-reads, \
                         c cancels",
                        view.verb(),
                        view.subject(),
                        self.options.poll_limit.as_secs() / 60
                    );
                    self.post(move |s| {
                        upsert_job(&s, view);
                        s.notice.set(Some(msg));
                    });
                    return;
                }
                // A job waiting for a person moves only when someone
                // acts: poll it at half the pace (the web console's 3 s).
                let delay = if view.is_paused() {
                    self.options.poll_interval * 2
                } else {
                    self.options.poll_interval
                };
                self.post(move |s| upsert_job(&s, view));
                schedule_job_poll(&self.tx, JobPoll { errors: 0, ..poll }, delay);
            }
            Err(e) if e.kind == TransportErrorKind::NotFound || poll.errors >= 2 => {
                let id = poll.job_id.clone();
                self.watching.remove(&id);
                self.finished.insert(id.clone());
                let msg = format!(
                    "lost track of job {id} ({}: {}) — r re-reads what is on disk",
                    e.headline(),
                    e.message
                );
                self.post(move |s| {
                    s.jobs.update(|v| {
                        if let Some(j) = v.iter_mut().find(|j| j.job_id == id) {
                            j.status = "unknown".into();
                        }
                    });
                    s.job.update(|j| {
                        if let Some(j) = j.as_mut().filter(|j| j.job_id == id) {
                            j.status = "unknown".into();
                        }
                    });
                    s.notice.set(Some(msg));
                });
            }
            Err(_) => schedule_job_poll(
                &self.tx,
                JobPoll {
                    errors: poll.errors + 1,
                    ..poll
                },
                self.options.poll_interval,
            ),
        }
    }

    /// A job reached a terminal state: toast it once, then re-read what
    /// it changed (installed + catalog for models, engines for installs).
    fn finish(&mut self, view: JobView) {
        if !view.job_id.is_empty() && !self.finished.insert(view.job_id.clone()) {
            return;
        }
        let msg = view.outcome_line();
        let kind = view.kind.clone();
        let changed = view.status == "completed" && !view.dry_run;
        let tx = self.tx.clone();
        self.post(move |s| {
            upsert_job(&s, view);
            s.notice.set(Some(msg));
            if !changed {
                return;
            }
            if kind == "engine_install" {
                s.engines.set(Remote::Loading);
                let _ = tx.send(ScreenCmd::LoadEngines { probe: true });
            } else {
                s.installed.set(Remote::Loading);
                let _ = tx.send(ScreenCmd::LoadInstalled);
                let generation = s.catalog_gen.get_untracked() + 1;
                s.catalog_gen.set(generation);
                // Keep the rows on screen while the refreshed presence
                // column loads — a download finishing must not blank
                // the list the operator is reading.
                let hub = s.hub.get_untracked();
                let _ = tx.send(ScreenCmd::LoadCatalog {
                    q: hub.clone().unwrap_or_else(|| s.query.get_untracked()),
                    engine: s.engine_filter.get_untracked(),
                    fits_only: s.fits_only.get_untracked(),
                    generation,
                    hub: hub.is_some(),
                });
            }
        });
    }
}

/// Add newly seen engine ids to the `e` cycle (UI thread).
fn remember_providers<'a>(s: &ScreensStore, ids: impl Iterator<Item = &'a String>) {
    let fresh: Vec<String> = s.providers_seen.with_untracked(|seen| {
        let mut out: Vec<String> = Vec::new();
        for id in ids {
            if !id.is_empty() && !seen.contains(id) && !out.contains(id) {
                out.push(id.clone());
            }
        }
        out
    });
    if !fresh.is_empty() {
        s.providers_seen.update(|seen| seen.extend(fresh));
    }
}

fn remote<T>(res: Result<Value, TransportError>, parse: fn(&Value) -> T) -> Remote<T> {
    match res {
        Ok(v) => Remote::Ready(parse(&v)),
        Err(e) => Remote::Failed(e),
    }
}
