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
use std::time::{Duration, Instant};

use abstracttui::reactive::WakeHandle;
use serde_json::Value;

use super::data::{CatalogData, EnginesData, HostProfile, InstalledData, JobView};
use super::{error_notice, JobPoll, Remote, ScreenCmd, ScreensStore};
use crate::transport::{ConsoleTransport, TransportError, TransportErrorKind};

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
            } => {
                let r = remote(
                    self.transport
                        .models_catalog(&q, engine.as_deref(), fits_only),
                    CatalogData::from_value,
                );
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
            ScreenCmd::Download { provider, artifact } => {
                let res = self.transport.start_download(&provider, &artifact);
                self.started("download", &format!("{provider} {artifact}"), res);
            }
            ScreenCmd::Delete {
                provider,
                artifact,
                force,
            } => {
                let res = self.transport.delete_model(&provider, &artifact, force);
                self.started("delete", &format!("{provider} {artifact}"), res);
            }
            ScreenCmd::Install { engine, dry_run } => {
                let res = self.transport.engine_install(&engine, dry_run);
                self.started("install", &engine, res);
            }
            ScreenCmd::PollJob(poll) => self.poll(poll),
            ScreenCmd::Cancel { job_id } => match self.transport.cancel_job(&job_id) {
                Ok(v) => {
                    let view = JobView::from_value(&v);
                    if view.is_active() {
                        self.post(move |s| s.job.set(Some(view)));
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

    /// A verb answered: show the job; watch it if it is still running.
    fn started(&mut self, verb: &str, subject: &str, res: Result<Value, TransportError>) {
        match res {
            Ok(v) => {
                let view = JobView::from_value(&v);
                if view.is_active() && !view.job_id.is_empty() {
                    let poll = JobPoll {
                        job_id: view.job_id.clone(),
                        since: Instant::now(),
                        errors: 0,
                    };
                    self.post(move |s| s.job.set(Some(view)));
                    schedule_job_poll(&self.tx, poll, self.options.poll_interval);
                } else {
                    self.finish(view);
                }
            }
            Err(e) => {
                let msg = error_notice(verb, subject, &e);
                self.post(move |s| {
                    // Drop the "starting…" placeholder: nothing runs.
                    if s.job
                        .with_untracked(|j| j.as_ref().is_some_and(|j| j.job_id.is_empty()))
                    {
                        s.job.set(None);
                    }
                    s.notice.set(Some(msg));
                });
            }
        }
    }

    fn poll(&mut self, poll: JobPoll) {
        if self.finished.contains(&poll.job_id) {
            return;
        }
        match self.transport.job(&poll.job_id) {
            Ok(v) => {
                let view = JobView::from_value(&v);
                if !view.is_active() {
                    self.finish(view);
                    return;
                }
                if poll.since.elapsed() > self.options.poll_limit {
                    let msg = format!(
                        "{} {} is still running after {} min — it keeps going; r re-reads, \
                         c cancels",
                        view.verb(),
                        view.subject(),
                        self.options.poll_limit.as_secs() / 60
                    );
                    self.post(move |s| {
                        s.job.set(Some(view));
                        s.notice.set(Some(msg));
                    });
                    return;
                }
                self.post(move |s| s.job.set(Some(view)));
                schedule_job_poll(
                    &self.tx,
                    JobPoll { errors: 0, ..poll },
                    self.options.poll_interval,
                );
            }
            Err(e) if e.kind == TransportErrorKind::NotFound || poll.errors >= 2 => {
                let id = poll.job_id.clone();
                self.finished.insert(id.clone());
                let msg = format!(
                    "lost track of job {id} ({}: {}) — r re-reads what is on disk",
                    e.headline(),
                    e.message
                );
                self.post(move |s| {
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
            s.job.set(Some(view));
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
                let _ = tx.send(ScreenCmd::LoadCatalog {
                    q: s.query.get_untracked(),
                    engine: s.engine_filter.get_untracked(),
                    fits_only: s.fits_only.get_untracked(),
                    generation,
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
