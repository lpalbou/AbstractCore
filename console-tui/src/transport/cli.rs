//! [`CliTransport`]: the core console's backend — `abstractcore … --json`
//! subprocesses on THIS machine.
//!
//! Reads (`host profile`, `engines status`, `models catalog|search|list`)
//! are one blocking subprocess each. Exit codes follow contract F:
//! `0` = the JSON document on stdout, `2` = REFUSED (the backend decided
//! no; its JSON body, when it printed one, rides along), anything else =
//! failed.
//!
//! Jobs. The abstractcore job registry lives inside ONE Python process,
//! so a job started by one CLI invocation cannot be polled by the next.
//! The long verbs (`models download`, a real `engines install`) therefore
//! run as a CHILD PROCESS this transport owns: `start_download` spawns
//! it and returns at once with a `host_job_v1` document whose `job_id`
//! names the child (`cli-<n>`), `job(id)` reports the child's state and
//! `cancel_job(id)` terminates it. While the child runs, any stdout line
//! that is itself a `host_job_v1` JSON object (NDJSON progress) updates
//! percent/bytes/message; stderr lines feed `log_tail`. When the child
//! exits, its final stdout JSON is adopted and the exit code decides the
//! status. A child that exits within the short refusal window with code
//! 2 is reported as a [`TransportErrorKind::Refused`] error straight from
//! the start call, like a gateway's 409/403.
//!
//! Short verbs (`models delete`, `engines install --dry-run`) run
//! synchronously and return the CLI's final job document.
//!
//! argv is FIXED: subcommand words from this file plus the provider /
//! artifact / engine ids from the backend's own payloads — never a shell
//! string, and an id starting with `-` is refused rather than risk being
//! parsed as a flag.
//!
//! ```no_run
//! use abstractcore_console::transport::{CliTransport, ConsoleTransport};
//!
//! let t = CliTransport::from_env().expect("abstractcore on PATH");
//! let engines = t.engines_status(false).unwrap();
//! assert_eq!(engines["schema"], "engines_status_v1");
//! ```

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use super::{ConsoleTransport, TransportError, TransportErrorKind};
use crate::cli::{self, CliErrorKind};

/// Jobs kept for `job(id)` after they finish (contract E keeps ≤ 40).
const MAX_RETAINED_JOBS: usize = 40;
/// stderr lines kept per child job for `log_tail`.
const LOG_TAIL_LINES: usize = 40;

/// The `abstractcore … --json` subprocess transport (see module docs).
pub struct CliTransport {
    bin: PathBuf,
    read_timeout: Duration,
    probe_timeout: Duration,
    verb_timeout: Duration,
    refusal_window: Duration,
    jobs: Mutex<Vec<Arc<ChildJob>>>,
    next_id: AtomicU64,
    host: OnceLock<String>,
}

impl CliTransport {
    /// A transport over an explicit `abstractcore` binary.
    pub fn new(bin: impl Into<PathBuf>) -> CliTransport {
        CliTransport {
            bin: bin.into(),
            read_timeout: Duration::from_secs(60),
            probe_timeout: Duration::from_secs(120),
            verb_timeout: Duration::from_secs(300),
            refusal_window: Duration::from_millis(1500),
            jobs: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
            host: OnceLock::new(),
        }
    }

    /// Resolve the binary like the rest of the console:
    /// `$ABSTRACTCORE_BIN` → `abstractcore` on PATH → the framework venv.
    /// `None` when nothing answers.
    pub fn from_env() -> Option<CliTransport> {
        cli::resolve_bin_from_env().map(|info| CliTransport::new(info.bin))
    }

    /// Deadline for read subprocesses (default 60 s; `--probe` gets 120 s).
    pub fn with_read_timeout(mut self, timeout: Duration) -> CliTransport {
        self.read_timeout = timeout;
        self.probe_timeout = timeout * 2;
        self
    }

    /// How long a freshly spawned job child is watched for an immediate
    /// refusal (exit 2) before the start call returns "running"
    /// (default 1.5 s).
    pub fn with_refusal_window(mut self, window: Duration) -> CliTransport {
        self.refusal_window = window;
        self
    }

    /// The binary this transport drives.
    pub fn bin(&self) -> &Path {
        &self.bin
    }

    fn run_sync(&self, args: &[&str], timeout: Duration) -> Result<Value, TransportError> {
        let label = format!("abstractcore {}", args.join(" "));
        let (status, stdout, stderr) =
            cli::run_raw_at(&self.bin, args, &label, timeout).map_err(|e| match e.kind {
                CliErrorKind::Timeout => {
                    TransportError::new(TransportErrorKind::Timeout, e.message)
                }
                _ => TransportError::unavailable(format!("could not start {}", e.message)),
            })?;
        let code = status.code().unwrap_or(-1);
        let body = parse_json_output(&stdout);
        match code {
            0 => body.ok_or_else(|| {
                TransportError::protocol(format!(
                    "`{label}` answered, but not with JSON — an abstractcore too old for this \
                     verb? first bytes: {}",
                    cli::head(&stdout, 120)
                ))
            }),
            2 => Err(refusal(code, body, &stdout, &stderr)),
            _ => {
                let message =
                    body_error(body.as_ref()).unwrap_or_else(|| cli::error_line(&stdout, &stderr));
                let mut e = TransportError::failed(message).with_code(code);
                e.body = body;
                Err(e)
            }
        }
    }

    /// Spawn a job child and watch it for the refusal window.
    fn start_child(&self, spec: ChildSpec) -> Result<Value, TransportError> {
        let id = format!("cli-{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let mut child = Command::new(&self.bin)
            .args(&spec.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                TransportError::unavailable(format!("could not start {}: {e}", self.bin.display()))
            })?;
        let state = Arc::new(Mutex::new(ChildState::default()));
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        {
            let state = state.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    let mut s = state.lock().expect("child state");
                    if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                        if v.get("schema").and_then(Value::as_str) == Some("host_job_v1") {
                            s.latest = Some(v);
                        }
                    }
                    s.stdout.push_str(&line);
                    s.stdout.push('\n');
                }
                state.lock().expect("child state").stdout_done = true;
            });
        }
        {
            let state = state.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let mut s = state.lock().expect("child state");
                    s.stderr_tail.push_back(line);
                    while s.stderr_tail.len() > LOG_TAIL_LINES {
                        s.stderr_tail.pop_front();
                    }
                }
                state.lock().expect("child state").stderr_done = true;
            });
        }
        let job = Arc::new(ChildJob {
            id,
            spec,
            started_at: now_iso(),
            child: Mutex::new(child),
            state,
        });
        {
            let mut jobs = self.jobs.lock().expect("jobs");
            jobs.push(job.clone());
            prune(&mut jobs);
        }
        let deadline = Instant::now() + self.refusal_window;
        while Instant::now() < deadline {
            if job.poll_exit().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let doc = job.snapshot();
        match job.exit_code() {
            Some(2) => {
                let (stdout, stderr) = job.outputs();
                Err(refusal(2, parse_json_output(&stdout), &stdout, &stderr))
            }
            Some(code) if code != 0 && !job.cancelled() => {
                let mut e = TransportError::failed(
                    doc.get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("the job exited early")
                        .to_string(),
                )
                .with_code(code);
                e.body = Some(doc);
                Err(e)
            }
            _ => Ok(doc),
        }
    }

    fn find(&self, id: &str) -> Option<Arc<ChildJob>> {
        self.jobs
            .lock()
            .expect("jobs")
            .iter()
            .find(|j| j.id == id)
            .cloned()
    }
}

impl ConsoleTransport for CliTransport {
    fn host_profile(&self) -> Result<Value, TransportError> {
        self.run_sync(&["host", "profile", "--json"], self.read_timeout)
    }

    fn engines_status(&self, probe: bool) -> Result<Value, TransportError> {
        let mut args = vec!["engines", "status"];
        if probe {
            args.push("--probe");
        }
        args.push("--json");
        let timeout = if probe {
            self.probe_timeout
        } else {
            self.read_timeout
        };
        self.run_sync(&args, timeout)
    }

    fn models_catalog(
        &self,
        q: &str,
        engine: Option<&str>,
        fits_only: bool,
    ) -> Result<Value, TransportError> {
        let q = q.trim();
        let mut args: Vec<&str> = if q.is_empty() {
            vec!["models", "catalog"]
        } else {
            refuse_flag_like("query", q)?;
            vec!["models", "search", q]
        };
        if let Some(e) = engine {
            refuse_flag_like("engine", e)?;
            args.extend(["--engine", e]);
        }
        if fits_only {
            args.push("--fits");
        }
        args.push("--json");
        self.run_sync(&args, self.read_timeout)
    }

    fn models_installed(&self, provider: Option<&str>) -> Result<Value, TransportError> {
        let mut args = vec!["models", "list"];
        if let Some(p) = provider {
            refuse_flag_like("provider", p)?;
            args.extend(["--provider", p]);
        }
        args.push("--json");
        self.run_sync(&args, self.read_timeout)
    }

    fn start_download(&self, provider: &str, artifact: &str) -> Result<Value, TransportError> {
        refuse_flag_like("provider", provider)?;
        refuse_flag_like("artifact", artifact)?;
        self.start_child(ChildSpec {
            kind: "download",
            args: vec![
                "models".into(),
                "download".into(),
                provider.into(),
                artifact.into(),
                "--json".into(),
            ],
            provider: Some(provider.into()),
            artifact: Some(artifact.into()),
            engine: None,
        })
    }

    fn delete_model(
        &self,
        provider: &str,
        artifact: &str,
        force: bool,
    ) -> Result<Value, TransportError> {
        refuse_flag_like("provider", provider)?;
        refuse_flag_like("artifact", artifact)?;
        let mut args = vec!["models", "delete", provider, artifact, "--yes"];
        if force {
            args.push("--force");
        }
        args.push("--json");
        self.run_sync(&args, self.verb_timeout)
    }

    fn engine_install(&self, id: &str, dry_run: bool) -> Result<Value, TransportError> {
        refuse_flag_like("engine", id)?;
        if dry_run {
            return self.run_sync(
                &["engines", "install", id, "--yes", "--dry-run", "--json"],
                self.read_timeout,
            );
        }
        self.start_child(ChildSpec {
            kind: "engine_install",
            args: vec![
                "engines".into(),
                "install".into(),
                id.into(),
                "--yes".into(),
                "--json".into(),
            ],
            provider: None,
            artifact: None,
            engine: Some(id.into()),
        })
    }

    fn job(&self, id: &str) -> Result<Value, TransportError> {
        if let Some(job) = self.find(id) {
            job.poll_exit();
            return Ok(job.snapshot());
        }
        // Not ours: a job some other process started. `models jobs`
        // lists that process's registry only when it IS this process's
        // registry, so this usually ends in an honest not-found.
        let listed = self.run_sync(&["models", "jobs", "--json"], self.read_timeout)?;
        let jobs = listed
            .get("jobs")
            .and_then(Value::as_array)
            .or_else(|| listed.as_array())
            .cloned()
            .unwrap_or_default();
        jobs.into_iter()
            .find(|j| j.get("job_id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| TransportError::not_found(format!("no job {id} on this machine")))
    }

    fn cancel_job(&self, id: &str) -> Result<Value, TransportError> {
        let job = self.find(id).ok_or_else(|| {
            TransportError::not_found(format!(
                "job {id} was not started by this console — cancel it where it runs"
            ))
        })?;
        job.cancel();
        Ok(job.snapshot())
    }

    fn host_label(&self) -> String {
        self.host
            .get_or_init(|| match local_hostname() {
                Some(h) => format!("this machine ({h})"),
                None => "this machine".to_string(),
            })
            .clone()
    }
}

// ---------------------------------------------------------------------
// Child jobs
// ---------------------------------------------------------------------

struct ChildSpec {
    kind: &'static str,
    args: Vec<String>,
    provider: Option<String>,
    artifact: Option<String>,
    engine: Option<String>,
}

#[derive(Default)]
struct ChildState {
    stdout: String,
    /// The last NDJSON `host_job_v1` progress line, if the CLI streams.
    latest: Option<Value>,
    stderr_tail: VecDeque<String>,
    exit: Option<i32>,
    finished_at: Option<String>,
    cancelled: bool,
    stdout_done: bool,
    stderr_done: bool,
}

struct ChildJob {
    id: String,
    spec: ChildSpec,
    started_at: String,
    child: Mutex<Child>,
    state: Arc<Mutex<ChildState>>,
}

impl ChildJob {
    /// Reap the child if it exited; returns the exit code once known.
    fn poll_exit(&self) -> Option<i32> {
        if let Some(code) = self.exit_code() {
            return Some(code);
        }
        let status = self
            .child
            .lock()
            .expect("child")
            .try_wait()
            .ok()
            .flatten()?;
        let code = status.code().unwrap_or(-1);
        // The readers finish when the pipes close; a grandchild holding
        // a pipe open must not hang the poll, so the wait is bounded.
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            let done = {
                let s = self.state.lock().expect("child state");
                s.stdout_done && s.stderr_done
            };
            if done {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut s = self.state.lock().expect("child state");
        s.exit = Some(code);
        s.finished_at = Some(now_iso());
        Some(code)
    }

    fn exit_code(&self) -> Option<i32> {
        self.state.lock().expect("child state").exit
    }

    fn cancelled(&self) -> bool {
        self.state.lock().expect("child state").cancelled
    }

    fn finished(&self) -> bool {
        self.exit_code().is_some()
    }

    fn outputs(&self) -> (String, String) {
        let s = self.state.lock().expect("child state");
        (
            s.stdout.clone(),
            s.stderr_tail.iter().cloned().collect::<Vec<_>>().join("\n"),
        )
    }

    fn cancel(&self) {
        if self.poll_exit().is_some() {
            return;
        }
        self.state.lock().expect("child state").cancelled = true;
        let mut child = self.child.lock().expect("child");
        let _ = child.kill();
        let _ = child.wait();
        drop(child);
        self.poll_exit();
    }

    /// The `host_job_v1` document for this child, now.
    fn snapshot(&self) -> Value {
        let s = self.state.lock().expect("child state");
        let command: Vec<Value> = std::iter::once(Value::from("abstractcore"))
            .chain(self.spec.args.iter().map(|a| Value::from(a.as_str())))
            .collect();
        let cli_equivalent = format!(
            "abstractcore {}",
            self.spec
                .args
                .iter()
                .filter(|a| a.as_str() != "--json")
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        );
        let mut doc = json!({
            "schema": "host_job_v1",
            "job_id": self.id,
            "kind": self.spec.kind,
            "status": "running",
            "provider": self.spec.provider,
            "artifact": self.spec.artifact,
            "engine": self.spec.engine,
            "percent": null,
            "downloaded_bytes": null,
            "total_bytes": null,
            "message": null,
            "log_tail": [],
            "command": command,
            "dry_run": false,
            "started_at": self.started_at,
            "finished_at": s.finished_at,
            "error": null,
            "joined": 0,
            "cli_equivalent": cli_equivalent,
        });
        // The CLI's own job document wins field by field (it knows the
        // percent, the vendor tool's argv, the error) — except the id,
        // which must stay the one this transport can answer for.
        let upstream = if s.exit.is_some() {
            parse_json_output(&s.stdout).or_else(|| s.latest.clone())
        } else {
            s.latest.clone()
        };
        let obj = doc.as_object_mut().expect("object");
        if let Some(Value::Object(up)) = &upstream {
            for (k, v) in up {
                match k.as_str() {
                    "job_id" => {
                        obj.insert("upstream_job_id".into(), v.clone());
                    }
                    "status" | "finished_at" => {}
                    _ => {
                        obj.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        let upstream_status = upstream
            .as_ref()
            .and_then(|u| u.get("status"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let last_stderr = s.stderr_tail.back().cloned();
        let status = match (s.exit, s.cancelled) {
            (None, _) => "running".to_string(),
            (Some(_), true) => "cancelled".to_string(),
            (Some(0), false) => match upstream_status.as_deref() {
                Some(st @ ("completed" | "failed" | "cancelled")) => st.to_string(),
                _ => "completed".to_string(),
            },
            (Some(_), false) => "failed".to_string(),
        };
        if status == "failed" && obj.get("error").is_none_or(Value::is_null) {
            let why = body_error(upstream.as_ref())
                .or(last_stderr.clone())
                .unwrap_or_else(|| format!("exited with {}", s.exit.unwrap_or(-1)));
            let why = if s.exit == Some(2) {
                format!("refused: {why}")
            } else {
                why
            };
            obj.insert("error".into(), Value::from(why));
        }
        if status == "cancelled" && obj.get("message").is_none_or(Value::is_null) {
            obj.insert("message".into(), Value::from("cancelled"));
        }
        if obj.get("message").is_none_or(Value::is_null) {
            if let Some(l) = last_stderr {
                obj.insert("message".into(), Value::from(l));
            }
        }
        let tail_empty = obj
            .get("log_tail")
            .and_then(Value::as_array)
            .is_none_or(|a| a.is_empty());
        if tail_empty && !s.stderr_tail.is_empty() {
            obj.insert(
                "log_tail".into(),
                Value::from(
                    s.stderr_tail
                        .iter()
                        .map(|l| Value::from(l.as_str()))
                        .collect::<Vec<_>>(),
                ),
            );
        }
        if status == "completed" && obj.get("percent").is_none_or(Value::is_null) {
            obj.insert("percent".into(), Value::from(100.0));
        }
        obj.insert("status".into(), Value::from(status));
        obj.insert("finished_at".into(), json!(s.finished_at));
        doc
    }
}

/// Keep at most `MAX_RETAINED_JOBS`, dropping the oldest FINISHED ones.
fn prune(jobs: &mut Vec<Arc<ChildJob>>) {
    while jobs.len() > MAX_RETAINED_JOBS {
        match jobs.iter().position(|j| j.finished()) {
            Some(i) => {
                jobs.remove(i);
            }
            None => break,
        }
    }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// The whole stdout as one JSON document, else its last line that is one
/// (a CLI that streams NDJSON progress ends with the final job).
fn parse_json_output(stdout: &str) -> Option<Value> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    trimmed
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
        .filter(|v| v.is_object())
}

fn body_error(body: Option<&Value>) -> Option<String> {
    let b = body?;
    for key in ["error", "message", "detail"] {
        if let Some(s) = b.get(key).and_then(Value::as_str) {
            if !s.trim().is_empty() {
                return Some(s.trim().to_string());
            }
        }
    }
    None
}

fn refusal(code: i32, body: Option<Value>, stdout: &str, stderr: &str) -> TransportError {
    let message = body_error(body.as_ref()).unwrap_or_else(|| cli::error_line(stdout, stderr));
    let mut e = TransportError::refused(message, body).with_code(code);
    if e.body.is_none() {
        e.body = Some(Value::Object(Map::new()));
    }
    e
}

/// Ids come from the backend's own payloads, but an id that starts
/// with `-` would reach argparse as a FLAG — refuse it outright.
fn refuse_flag_like(what: &str, value: &str) -> Result<(), TransportError> {
    if value.starts_with('-') || value.is_empty() {
        return Err(TransportError::refused(
            format!("{what} {value:?} is not a valid id"),
            None,
        ));
    }
    Ok(())
}

fn local_hostname() -> Option<String> {
    let out = Command::new("hostname")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!name.is_empty()).then_some(name)
}

/// ISO-8601 UTC, seconds precision (no chrono dependency).
fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil-from-days (Howard Hinnant), proleptic Gregorian.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_timestamp_shape() {
        let s = now_iso();
        assert_eq!(s.len(), 20, "{s}");
        assert!(s.ends_with('Z') && s.as_bytes()[10] == b'T', "{s}");
    }

    #[test]
    fn json_output_prefers_whole_then_last_line() {
        assert_eq!(parse_json_output("{\"a\":1}").unwrap()["a"], 1);
        let nd = "{\"schema\":\"host_job_v1\",\"percent\":10}\n{\"schema\":\"host_job_v1\",\"percent\":100}\n";
        assert_eq!(parse_json_output(nd).unwrap()["percent"], 100);
        assert!(parse_json_output("  ").is_none());
        assert!(parse_json_output("not json").is_none());
    }

    #[test]
    fn flag_like_ids_are_refused() {
        assert!(refuse_flag_like("artifact", "--force")
            .unwrap_err()
            .is_refused());
        assert!(refuse_flag_like("artifact", "qwen3:8b").is_ok());
    }
}
