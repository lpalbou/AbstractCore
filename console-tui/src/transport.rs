//! The seam between the shared Models / Engines screens and whatever
//! answers them (contract H).
//!
//! The screens never know WHERE the answers come from. The core
//! console's binary plugs in [`CliTransport`], which runs
//! `abstractcore … --json` subprocesses on this machine; the gateway
//! console plugs in its own `HttpTransport` over the gateway's
//! `/api/gateway/models/*` and `/api/gateway/engines/*` mirrors. Both
//! return the SAME JSON documents (contracts A–E in the mission's
//! `CONTRACTS.md`), so one set of screens renders both.
//!
//! Every method is BLOCKING and is only ever called from the screens'
//! worker thread ([`crate::screens::spawn_worker`]), never from the UI
//! thread — an implementation is free to do network or process I/O.
//!
//! # Implementing a transport
//!
//! ```
//! use abstractcore_console::transport::{ConsoleTransport, TransportError};
//! use serde_json::{json, Value};
//!
//! /// A transport that knows nothing: every read says so honestly.
//! struct Offline;
//!
//! impl ConsoleTransport for Offline {
//!     fn host_profile(&self) -> Result<Value, TransportError> {
//!         Err(TransportError::unavailable("offline"))
//!     }
//!     fn engines_status(&self, _probe: bool) -> Result<Value, TransportError> {
//!         Ok(json!({"schema": "engines_status_v1", "engines": []}))
//!     }
//!     fn models_catalog(&self, _q: &str, _engine: Option<&str>, _fits_only: bool)
//!         -> Result<Value, TransportError> {
//!         Ok(json!({"schema": "model_catalog_v1", "rows": []}))
//!     }
//!     fn models_installed(&self, _provider: Option<&str>) -> Result<Value, TransportError> {
//!         Ok(json!({"schema": "models_installed_v1", "rows": []}))
//!     }
//!     fn start_download(&self, _p: &str, _a: &str) -> Result<Value, TransportError> {
//!         Err(TransportError::refused("offline", None))
//!     }
//!     fn delete_model(&self, _p: &str, _a: &str, _force: bool) -> Result<Value, TransportError> {
//!         Err(TransportError::refused("offline", None))
//!     }
//!     fn engine_install(&self, _id: &str, _dry_run: bool) -> Result<Value, TransportError> {
//!         Err(TransportError::refused("offline", None))
//!     }
//!     fn job(&self, id: &str) -> Result<Value, TransportError> {
//!         Err(TransportError::not_found(format!("no job {id}")))
//!     }
//!     fn cancel_job(&self, id: &str) -> Result<Value, TransportError> {
//!         Err(TransportError::not_found(format!("no job {id}")))
//!     }
//! }
//!
//! let t: std::sync::Arc<dyn ConsoleTransport> = std::sync::Arc::new(Offline);
//! assert_eq!(t.host_label(), "this machine");
//! assert!(t.start_download("ollama", "qwen3:8b").unwrap_err().is_refused());
//! ```

pub mod cli;

use serde_json::Value;

pub use cli::CliTransport;

/// What went wrong talking to the backend — classified so the screens
/// can say what the operator can DO about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportErrorKind {
    /// The backend is not there at all: no `abstractcore` binary, the
    /// gateway is unreachable, the process could not be spawned.
    Unavailable,
    /// The backend answered and SAID NO: CLI exit code 2, HTTP 403/409.
    /// Not a failure of the machinery — a decision (a loaded model, a
    /// shared cache, installs disabled, a job already running). The
    /// payload's own explanation rides in [`TransportError::body`].
    Refused,
    /// The backend tried and failed: CLI exit 1, HTTP 5xx, a crash.
    Failed,
    /// The thing asked about does not exist (unknown job id, HTTP 404).
    NotFound,
    /// No answer within the deadline.
    Timeout,
    /// An answer arrived but was not the JSON the contract promises.
    Protocol,
    /// The caller is not allowed (HTTP 401): sign in / token.
    Unauthorized,
}

/// A transport failure: its class, one human line, and whatever JSON
/// the backend sent with it (a refusal's reasons, a job's error).
#[derive(Clone, Debug, PartialEq)]
pub struct TransportError {
    pub kind: TransportErrorKind,
    pub message: String,
    /// The exit code (CLI) or HTTP status (gateway), when there was one.
    pub code: Option<i32>,
    /// The backend's JSON body, if it sent one — a refusal's
    /// `delete_blockers`, `error`, or the job it refused to start.
    pub body: Option<Value>,
}

impl TransportError {
    pub fn new(kind: TransportErrorKind, message: impl Into<String>) -> TransportError {
        TransportError {
            kind,
            message: message.into(),
            code: None,
            body: None,
        }
    }

    pub fn unavailable(message: impl Into<String>) -> TransportError {
        TransportError::new(TransportErrorKind::Unavailable, message)
    }

    pub fn refused(message: impl Into<String>, body: Option<Value>) -> TransportError {
        TransportError {
            body,
            ..TransportError::new(TransportErrorKind::Refused, message)
        }
    }

    pub fn failed(message: impl Into<String>) -> TransportError {
        TransportError::new(TransportErrorKind::Failed, message)
    }

    pub fn not_found(message: impl Into<String>) -> TransportError {
        TransportError::new(TransportErrorKind::NotFound, message)
    }

    pub fn protocol(message: impl Into<String>) -> TransportError {
        TransportError::new(TransportErrorKind::Protocol, message)
    }

    pub fn with_code(mut self, code: i32) -> TransportError {
        self.code = Some(code);
        self
    }

    pub fn with_body(mut self, body: Value) -> TransportError {
        self.body = Some(body);
        self
    }

    pub fn is_refused(&self) -> bool {
        self.kind == TransportErrorKind::Refused
    }

    /// One short word for the class, as the screens print it.
    pub fn headline(&self) -> &'static str {
        match self.kind {
            TransportErrorKind::Unavailable => "unavailable",
            TransportErrorKind::Refused => "refused",
            TransportErrorKind::Failed => "failed",
            TransportErrorKind::NotFound => "not found",
            TransportErrorKind::Timeout => "timed out",
            TransportErrorKind::Protocol => "unexpected answer",
            TransportErrorKind::Unauthorized => "not authorized",
        }
    }

    /// The refusal's reasons, when the backend named them: the body's
    /// `delete_blockers` (contract D) and/or its `error` string.
    pub fn reasons(&self) -> Vec<String> {
        let Some(body) = &self.body else {
            return Vec::new();
        };
        let mut out: Vec<String> = body
            .get("delete_blockers")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if let Some(e) = body.get("error").and_then(Value::as_str) {
            if !e.is_empty() && !out.iter().any(|o| o == e) {
                out.push(e.to_string());
            }
        }
        out
    }
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.headline(), self.message)
    }
}

impl std::error::Error for TransportError {}

/// The Models / Engines backend (contract H). Every method returns the
/// contract document verbatim as a [`serde_json::Value`]; the screens
/// parse it tolerantly (unknown keys ignored, missing = unknown).
///
/// | method | CLI (`CliTransport`) | gateway route | contract |
/// |---|---|---|---|
/// | `host_profile` | `abstractcore host profile --json` | `GET /api/gateway/host/profile` | A |
/// | `engines_status` | `abstractcore engines status [--probe] --json` | `GET /api/gateway/engines` | B |
/// | `models_catalog` | `abstractcore models catalog\|search … --json` | `GET /api/gateway/models/catalog` | C |
/// | `models_installed` | `abstractcore models list [--provider X] --json` | `GET /api/gateway/models/installed` | D |
/// | `start_download` | `abstractcore models download P A --json` | `POST /api/gateway/models/download` | E |
/// | `delete_model` | `abstractcore models delete P A --yes [--force] --json` | `POST /api/gateway/models/delete` | E |
/// | `engine_install` | `abstractcore engines install ID --yes [--dry-run] --json` | `POST /api/gateway/engines/{id}/install` | E |
/// | `job` / `cancel_job` | the transport's own child jobs | `GET /api/gateway/jobs/{id}` / `POST …/cancel` | E |
///
/// The three verbs return a `host_job_v1` document; a job that is not
/// yet terminal is polled with [`ConsoleTransport::job`].
pub trait ConsoleTransport: Send + Sync {
    /// Contract A: the machine the models would run on.
    fn host_profile(&self) -> Result<Value, TransportError>;
    /// Contract B: every engine, installed or not. `probe` asks the
    /// backend to also contact local servers (slower, fresher).
    fn engines_status(&self, probe: bool) -> Result<Value, TransportError>;
    /// Contract C: the catalog, filtered by free text, engine and fit.
    fn models_catalog(
        &self,
        q: &str,
        engine: Option<&str>,
        fits_only: bool,
    ) -> Result<Value, TransportError>;
    /// Contract D: what is on disk now, per engine.
    fn models_installed(&self, provider: Option<&str>) -> Result<Value, TransportError>;
    /// Contract E: start (or join) a download job.
    fn start_download(&self, provider: &str, artifact: &str) -> Result<Value, TransportError>;
    /// Contract E: delete an installed artifact. Without `force` the
    /// backend refuses a loaded model or a shared cache
    /// ([`TransportErrorKind::Refused`], blockers in the body).
    fn delete_model(
        &self,
        provider: &str,
        artifact: &str,
        force: bool,
    ) -> Result<Value, TransportError>;
    /// Contract E: install an engine on the backend's HOST. `dry_run`
    /// returns the job that WOULD run (its `command`) without running it.
    fn engine_install(&self, id: &str, dry_run: bool) -> Result<Value, TransportError>;
    /// Contract E: one job's current state.
    fn job(&self, id: &str) -> Result<Value, TransportError>;
    /// Contract E: stop a running job; returns the job after the request.
    fn cancel_job(&self, id: &str) -> Result<Value, TransportError>;

    /// Where installs and deletes RUN, in words — the install prompt
    /// says "runs on host X". Additive to contract H with a default:
    /// the CLI transport names this machine, the gateway transport
    /// should name the gateway host.
    fn host_label(&self) -> String {
        "this machine".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reasons_collect_blockers_and_error() {
        let e = TransportError::refused(
            "model is loaded",
            Some(json!({"delete_blockers": ["loaded"], "error": "unload it first"})),
        );
        assert!(e.is_refused());
        assert_eq!(e.reasons(), vec!["loaded", "unload it first"]);
        assert_eq!(e.to_string(), "refused: model is loaded");
        assert!(TransportError::failed("x").reasons().is_empty());
    }
}
