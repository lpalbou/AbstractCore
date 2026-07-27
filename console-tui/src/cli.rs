//! The `abstractcore` CLI subprocess client — the console's only lane
//! into Python-derived views (and, in M2, into every coupled write).
//!
//! Runs ONLY on the worker thread. Machine surfaces only: JSON stdout
//! from `config … --json` subcommands and exit codes; human output
//! (`--status`, wizard prose) is never scraped (risk-map fact #9).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

/// The venv this workspace ships — the last-resort fallback when
/// nothing on PATH answers.
const KNOWN_VENV_BIN: &str = ".venv/bin/abstractcore";

#[derive(Clone, Debug, PartialEq)]
pub enum CliErrorKind {
    /// No abstractcore binary was found anywhere.
    NotFound,
    /// The process could not be spawned (permissions, broken venv).
    Spawn,
    /// The process outlived its deadline and was killed.
    Timeout,
    /// Nonzero exit; the message carries the CLI's own error line.
    Exit(i32),
    /// Exit 0 but stdout was not the JSON we asked for.
    BadJson,
}

#[derive(Clone, Debug)]
pub struct CliError {
    pub kind: CliErrorKind,
    pub message: String,
    /// Which binary failed — the chat lane's errors must not wear the
    /// core binary's name (M3 review P3-1: "abstractcore exited with
    /// 2" for an abstractcore-chat argparse refusal).
    pub program: &'static str,
}

impl CliError {
    pub fn core(kind: CliErrorKind, message: String) -> CliError {
        CliError {
            kind,
            message,
            program: "abstractcore",
        }
    }

    pub fn chat(kind: CliErrorKind, message: String) -> CliError {
        CliError {
            kind,
            message,
            program: "abstractcore-chat",
        }
    }

    pub fn headline(&self) -> String {
        let p = self.program;
        match self.kind {
            CliErrorKind::NotFound => format!("{p} CLI not found"),
            CliErrorKind::Spawn => format!("could not start {p}"),
            CliErrorKind::Timeout => format!("{p} timed out"),
            CliErrorKind::Exit(code) => format!("{p} exited with {code}"),
            CliErrorKind::BadJson => format!("{p} answered, but not with JSON"),
        }
    }

    /// What the operator can DO about it — refusals speak.
    pub fn hint(&self) -> &'static str {
        match self.kind {
            CliErrorKind::NotFound => {
                "set $ABSTRACTCORE_BIN, or install abstractcore on PATH — the file mirror still works"
            }
            CliErrorKind::Spawn => "check the binary is executable ($ABSTRACTCORE_BIN?)",
            CliErrorKind::Timeout => "the Python side hung — retry with r",
            CliErrorKind::Exit(_) => "the message above is the CLI's own error",
            CliErrorKind::BadJson => "an abstractcore too old for --json? check its version",
        }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.headline(), self.message)
    }
}

/// Where the binary came from — rendered in the header/overview so
/// "which abstractcore am I driving" is always answered.
#[derive(Clone, Debug)]
pub struct CliInfo {
    pub bin: PathBuf,
    pub source: &'static str,
}

/// `$ABSTRACTCORE_BIN` → `abstractcore` on PATH → the framework venv
/// fallback. None = not found (the mirror still works; derived views
/// and writes teach the fix).
pub fn resolve_bin(
    env: &dyn Fn(&str) -> Option<String>,
    home: &Path,
    exists: &dyn Fn(&Path) -> bool,
) -> Option<CliInfo> {
    if let Some(explicit) = env("ABSTRACTCORE_BIN").filter(|v| !v.trim().is_empty()) {
        // Explicit choice is honored even if the file check fails here
        // (it may be a PATH-relative name); spawn errors will name it.
        return Some(CliInfo {
            bin: PathBuf::from(explicit.trim()),
            source: "$ABSTRACTCORE_BIN",
        });
    }
    if let Some(paths) = env("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("abstractcore");
            if exists(&candidate) {
                return Some(CliInfo {
                    bin: candidate,
                    source: "PATH",
                });
            }
        }
    }
    let venv = home.join("tmp/abstractframework").join(KNOWN_VENV_BIN);
    if exists(&venv) {
        return Some(CliInfo {
            bin: venv,
            source: "framework venv fallback",
        });
    }
    None
}

pub fn resolve_bin_from_env() -> Option<CliInfo> {
    let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_default();
    resolve_bin(&|k| std::env::var(k).ok(), &home, &|p| p.is_file())
}

pub struct CoreCli {
    pub bin: PathBuf,
    /// `abstractcore-chat` — the one-shot generation lane (M3 test
    /// verbs). Resolved beside `bin` (same venv/bin dir) or on PATH;
    /// None = generation tests refuse with a teaching message.
    pub chat_bin: Option<PathBuf>,
    /// True when `chat_bin` came from the PATH fallback rather than
    /// beside `bin` — a different install may answer the generation
    /// test than the one serving discovery, and silence about it
    /// would contradict the sibling rationale (M3 review P3-2).
    pub chat_from_path: bool,
}

/// The chat binary ships in the same bin dir as `abstractcore`; a
/// sibling lookup keeps both binaries from the SAME install (a PATH
/// hit from a different venv would test a different abstractcore) —
/// the PATH fallback is flagged so the mismatch never goes silent.
pub fn resolve_chat_bin(
    core_bin: &Path,
    exists: &dyn Fn(&Path) -> bool,
) -> Option<(PathBuf, bool)> {
    let sibling = core_bin.with_file_name("abstractcore-chat");
    if exists(&sibling) {
        return Some((sibling, false));
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("abstractcore-chat");
            if exists(&candidate) {
                return Some((candidate, true));
            }
        }
    }
    None
}

/// A successful CLI run: the JSON payload PLUS any labeled degradation
/// the Python side printed to stderr while exiting 0. The
/// `#FALLBACK` lane is load-bearing (adversarial review P1-1): when
/// Python cannot load the config file it backs it up, runs on
/// DEFAULTS, prints one `#FALLBACK …` stderr line — and the JSON body
/// still says `ok: true`. Dropping that line made the mirror vouch
/// for a file Python refuses.
#[derive(Clone, Debug)]
pub struct CliOutput {
    pub value: Value,
    pub fallback_warnings: Vec<String>,
}

impl CoreCli {
    pub fn new(bin: PathBuf) -> CoreCli {
        let resolved = resolve_chat_bin(&bin, &|p| p.is_file());
        let chat_from_path = resolved.as_ref().is_some_and(|(_, from_path)| *from_path);
        CoreCli {
            bin,
            chat_bin: resolved.map(|(p, _)| p),
            chat_from_path,
        }
    }

    /// A CoreCli for tests: no chat sibling resolution.
    #[cfg(test)]
    pub fn bare(bin: PathBuf) -> CoreCli {
        CoreCli {
            bin,
            chat_bin: None,
            chat_from_path: false,
        }
    }

    /// Run `abstractcore-chat <args>` and return raw stdout + the
    /// stderr `#FALLBACK` lines. Exit-code truth only — the stdout
    /// verdict (reply vs `❌ Error:`) is the caller's fold
    /// (`probes::fold_generation`); this lane's argv never carries
    /// secrets (provider/model/prompt only).
    pub fn run_chat(&self, args: &[&str], timeout: Duration) -> Result<(String, Vec<String>), CliError> {
        let chat = self.chat_bin.as_ref().ok_or_else(|| {
            CliError::chat(
                CliErrorKind::NotFound,
                "abstractcore-chat not found beside abstractcore or on PATH".into(),
            )
        })?;
        let label = format!("abstractcore-chat {}", args.join(" "));
        let (status, stdout, stderr) = run_raw_at(chat, args, &label, timeout)
            .map_err(|mut e| {
                e.program = "abstractcore-chat";
                e
            })?;
        if !status.success() {
            return Err(CliError::chat(
                CliErrorKind::Exit(status.code().unwrap_or(-1)),
                error_line(&stdout, &stderr),
            ));
        }
        Ok((stdout, fallback_lines(&stderr)))
    }

    /// Run a SETTER invocation (human output, no JSON). The flags CLI
    /// exits 0 on refused writes (live-probed: `--set-server-port
    /// 99999` prints `❌ Error:` and exits 0) — so an error line on
    /// stdout is a failure REGARDLESS of the exit code. Returns the
    /// stderr `#FALLBACK` lines like the JSON lane.
    /// `redacted_label` names the invocation in error messages — argv
    /// may carry secrets here, so the label comes from the caller's
    /// already-redacted rendering, never from the args.
    pub fn run_setter(
        &self,
        args: &[&str],
        redacted_label: &str,
        timeout: Duration,
    ) -> Result<Vec<String>, CliError> {
        let (status, stdout, stderr) = self.run_raw(args, redacted_label, timeout)?;
        if !status.success() {
            return Err(CliError::core(
                CliErrorKind::Exit(status.code().unwrap_or(-1)),
                error_line(&stdout, &stderr),
            ));
        }
        if let Some(l) = stdout.lines().find(|l| l.contains("❌") || l.contains("Error:")) {
            return Err(CliError::core(
                CliErrorKind::Exit(0),
                format!("{} (the CLI still exited 0)", l.trim()),
            ));
        }
        Ok(fallback_lines(&stderr))
    }

    /// Run `abstractcore <args>` and parse stdout as JSON.
    ///
    /// Body over transport: a nonzero exit still tries to surface the
    /// CLI's own `❌ Error:` line (config subcommands print errors to
    /// stdout, main.py:1753-1760). Stdout/stderr are drained on reader
    /// threads so a large payload can never deadlock the pipe.
    pub fn run_json(&self, args: &[&str], timeout: Duration) -> Result<CliOutput, CliError> {
        // The read lane's argv never carries secrets — its own join is
        // an honest label.
        let label = format!("abstractcore {}", args.join(" "));
        let (status, stdout, stderr) = self.run_raw(args, &label, timeout)?;
        if !status.success() {
            let code = status.code().unwrap_or(-1);
            return Err(CliError::core(
                CliErrorKind::Exit(code),
                error_line(&stdout, &stderr),
            ));
        }
        let value = serde_json::from_str(&stdout).map_err(|e| CliError::core(
            CliErrorKind::BadJson,
            format!("{e} — first bytes: {}", head(&stdout, 120)),
        ))?;
        Ok(CliOutput {
            value,
            fallback_warnings: fallback_lines(&stderr),
        })
    }

    /// Shared subprocess mechanics: spawn, drain both pipes on reader
    /// threads (a large payload can never deadlock), wait with a
    /// deadline, kill on overrun.
    fn run_raw(
        &self,
        args: &[&str],
        redacted_label: &str,
        timeout: Duration,
    ) -> Result<(std::process::ExitStatus, String, String), CliError> {
        run_raw_at(&self.bin, args, redacted_label, timeout)
    }
}

fn run_raw_at(
    bin: &Path,
    args: &[&str],
    redacted_label: &str,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, String, String), CliError> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| CliError::core(
            CliErrorKind::Spawn,
            format!("{}: {e}", bin.display()),
        ))?;

    let stdout = child.stdout.take().expect("piped");
    let stderr = child.stderr.take().expect("piped");
    let out_h = std::thread::spawn(move || read_all(stdout));
    let err_h = std::thread::spawn(move || read_all(stderr));

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    // Join the readers so the threads never leak.
                    let _ = out_h.join();
                    let _ = err_h.join();
                    return Err(CliError::core(
                        CliErrorKind::Timeout,
                        format!("no answer within {}s: {redacted_label}", timeout.as_secs()),
                    ));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                return Err(CliError::core(
                    CliErrorKind::Spawn,
                    e.to_string(),
                ))
            }
        }
    };
    let stdout = out_h.join().unwrap_or_default();
    let stderr = err_h.join().unwrap_or_default();
    Ok((status, stdout, stderr))
}

/// The labeled degradations an exit-0 abstractcore run prints to
/// stderr. `#FALLBACK` is the framework-wide convention; the
/// corrupt-config warning uses it (manager.py:495-566).
fn fallback_lines(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .filter(|l| l.contains("#FALLBACK"))
        .map(|l| l.trim().to_string())
        .collect()
}

fn read_all(mut r: impl Read) -> String {
    let mut buf = Vec::new();
    let _ = r.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// The most useful single error line from a failed CLI run: the CLI's
/// own `❌ Error:` line (stdout) first, else the first non-empty stderr
/// line, else a generic head of whatever was printed.
fn error_line(stdout: &str, stderr: &str) -> String {
    if let Some(l) = stdout.lines().find(|l| l.contains("Error:")) {
        return l.trim().to_string();
    }
    if let Some(l) = stderr.lines().rev().find(|l| !l.trim().is_empty()) {
        return l.trim().to_string();
    }
    if let Some(l) = stdout.lines().find(|l| !l.trim().is_empty()) {
        return l.trim().to_string();
    }
    "(no output)".into()
}

fn head(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    t.replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_resolution_order() {
        let home = Path::new("/home/u");
        // Explicit env wins, even without an existence check.
        let info = resolve_bin(
            &|k| (k == "ABSTRACTCORE_BIN").then(|| "/opt/ac".to_string()),
            home,
            &|_| false,
        )
        .unwrap();
        assert_eq!(info.bin, PathBuf::from("/opt/ac"));
        assert_eq!(info.source, "$ABSTRACTCORE_BIN");

        // PATH scan finds the first hit.
        let info = resolve_bin(
            &|k| (k == "PATH").then(|| "/a:/b".to_string()),
            home,
            &|p| p == Path::new("/b/abstractcore"),
        )
        .unwrap();
        assert_eq!(info.bin, PathBuf::from("/b/abstractcore"));
        assert_eq!(info.source, "PATH");

        // Venv fallback.
        let info = resolve_bin(
            &|_| None,
            home,
            &|p| p == Path::new("/home/u/tmp/abstractframework/.venv/bin/abstractcore"),
        )
        .unwrap();
        assert_eq!(info.source, "framework venv fallback");

        // Nothing anywhere: honest None.
        assert!(resolve_bin(&|_| None, home, &|_| false).is_none());
    }

    #[test]
    fn error_line_prefers_the_cli_error() {
        let out = "some noise\n❌ Error: Unknown provider 'x'\nmore";
        assert_eq!(error_line(out, ""), "❌ Error: Unknown provider 'x'");
        assert_eq!(error_line("", "Traceback...\nValueError: boom"), "ValueError: boom");
        assert_eq!(error_line("", ""), "(no output)");
    }

    /// Subprocess mechanics against real processes (no network, no
    /// abstractcore needed): /bin/echo emits JSON; a sleep overruns the
    /// deadline and is killed.
    #[test]
    fn run_json_parses_and_times_out() {
        let echo = CoreCli::new(PathBuf::from("/bin/echo"));
        let out = echo
            .run_json(&["{\"ok\": true}"], Duration::from_secs(5))
            .unwrap();
        assert_eq!(out.value.get("ok").and_then(Value::as_bool), Some(true));
        assert!(out.fallback_warnings.is_empty());

        let sleep = CoreCli::new(PathBuf::from("/bin/sleep"));
        let err = sleep
            .run_json(&["5"], Duration::from_millis(200))
            .unwrap_err();
        assert_eq!(err.kind, CliErrorKind::Timeout);

        let missing = CoreCli::new(PathBuf::from("/nonexistent/bin"));
        let err = missing.run_json(&[], Duration::from_secs(1)).unwrap_err();
        assert_eq!(err.kind, CliErrorKind::Spawn);
    }

    /// The P1-1 signal lane: an exit-0 run whose stderr carries a
    /// `#FALLBACK` line surfaces it — the one honest signal Python
    /// gives when it refuses the config file while answering ok:true.
    #[test]
    fn exit_zero_stderr_fallback_is_surfaced() {
        let sh = CoreCli::new(PathBuf::from("/bin/sh"));
        let out = sh
            .run_json(
                &[
                    "-c",
                    "echo '#FALLBACK abstractcore config could not be parsed; \
                     falling back to DEFAULTS' >&2; echo '{\"ok\": true}'",
                ],
                Duration::from_secs(5),
            )
            .unwrap();
        assert_eq!(out.value.get("ok").and_then(Value::as_bool), Some(true));
        assert_eq!(out.fallback_warnings.len(), 1);
        assert!(out.fallback_warnings[0].contains("falling back to DEFAULTS"));
    }
}
