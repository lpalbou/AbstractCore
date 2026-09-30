//! The `abstractcore` CLI subprocess client — the console's only lane
//! into Python-derived views (and, in M2, into every coupled write).
//!
//! Runs ONLY on the worker thread. Machine surfaces only: JSON stdout
//! from `config … --json` subcommands and exit codes; human output
//! (`--status`, wizard prose) is never scraped (risk-map fact #9).

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;

/// The uv tool shim (`uv tool install abstractcore`), under `$HOME`.
const UV_TOOL_BIN: &str = ".local/bin/abstractcore";
/// A project virtualenv, relative to the CURRENT directory.
const LOCAL_VENV_BIN: &str = ".venv/bin/abstractcore";

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
    /// The person cancelled the command (it was killed).
    Cancelled,
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
            CliErrorKind::Cancelled => format!("{p} command cancelled"),
        }
    }

    /// What the operator can DO about it — refusals speak.
    pub fn hint(&self) -> &'static str {
        match self.kind {
            CliErrorKind::NotFound => {
                "set $ABSTRACTCORE_CLI, or install abstractcore on PATH — the file mirror still works"
            }
            CliErrorKind::Spawn => "check the binary is executable ($ABSTRACTCORE_CLI?)",
            CliErrorKind::Timeout => "the Python side hung — retry with r",
            CliErrorKind::Exit(_) => "the message above is the CLI's own error",
            CliErrorKind::BadJson => "an abstractcore too old for --json? check its version",
            CliErrorKind::Cancelled => "cancelled on request — nothing was stored",
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

/// How the console finds `abstractcore`, first hit wins:
///
/// 1. `$ABSTRACTCORE_CLI` — an explicit override (honored without an
///    existence check; a bad value fails loudly at spawn, naming it).
///    `$ABSTRACTCORE_BIN` is accepted as a legacy alias.
/// 2. `abstractcore` on `PATH`.
/// 3. `~/.local/bin/abstractcore` — the `uv tool install` shim, often
///    not on PATH in a fresh shell.
/// 4. `./.venv/bin/abstractcore` relative to the current directory.
///
/// None = not found (the mirror still works; derived views and writes
/// teach the fix).
pub fn resolve_bin(
    env: &dyn Fn(&str) -> Option<String>,
    home: &Path,
    cwd: &Path,
    exists: &dyn Fn(&Path) -> bool,
) -> Option<CliInfo> {
    for (var, source) in [
        ("ABSTRACTCORE_CLI", "$ABSTRACTCORE_CLI"),
        ("ABSTRACTCORE_BIN", "$ABSTRACTCORE_BIN (legacy alias)"),
    ] {
        if let Some(explicit) = env(var).filter(|v| !v.trim().is_empty()) {
            return Some(CliInfo {
                bin: PathBuf::from(explicit.trim()),
                source,
            });
        }
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
    if !home.as_os_str().is_empty() {
        let shim = home.join(UV_TOOL_BIN);
        if exists(&shim) {
            return Some(CliInfo {
                bin: shim,
                source: "~/.local/bin (uv tool)",
            });
        }
    }
    let venv = cwd.join(LOCAL_VENV_BIN);
    if exists(&venv) {
        return Some(CliInfo {
            bin: venv,
            source: "./.venv",
        });
    }
    None
}

pub fn resolve_bin_from_env() -> Option<CliInfo> {
    let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_default();
    let cwd = std::env::current_dir().unwrap_or_default();
    resolve_bin(&|k| std::env::var(k).ok(), &home, &cwd, &|p| p.is_file())
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
    pub fn run_chat(
        &self,
        args: &[&str],
        timeout: Duration,
    ) -> Result<(String, Vec<String>), CliError> {
        let chat = self.chat_bin.as_ref().ok_or_else(|| {
            CliError::chat(
                CliErrorKind::NotFound,
                "abstractcore-chat not found beside abstractcore or on PATH".into(),
            )
        })?;
        let label = format!("abstractcore-chat {}", args.join(" "));
        let (status, stdout, stderr) =
            run_raw_at(chat, args, &label, timeout).map_err(|mut e| {
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
        if let Some(l) = stdout
            .lines()
            .find(|l| l.contains("❌") || l.contains("Error:"))
        {
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
        let value = serde_json::from_str(&stdout).map_err(|e| {
            CliError::core(
                CliErrorKind::BadJson,
                format!("{e} — first bytes: {}", head(&stdout, 120)),
            )
        })?;
        Ok(CliOutput {
            value,
            fallback_warnings: fallback_lines(&stderr),
        })
    }

    /// Run a streaming `abstractcore <args> --json` verb and return its
    /// FINAL document.
    ///
    /// `models download <p> <a> --json` (and `engines install --json`)
    /// print one `host_job_v1` object per line while the job runs and the
    /// final job last; that last line also carries the legacy
    /// `ok`/`results` keys of the older one-document shape. A pretty-printed
    /// single document is accepted too. A nonzero exit reports the final
    /// job's own `error`/`message` when the last line is a job, else the
    /// CLI's error line.
    pub fn run_json_last(&self, args: &[&str], timeout: Duration) -> Result<CliOutput, CliError> {
        let label = format!("abstractcore {}", args.join(" "));
        let (status, stdout, stderr) = self.run_raw(args, &label, timeout)?;
        let last = last_json_object(&stdout);
        if !status.success() {
            let code = status.code().unwrap_or(-1);
            let msg = last
                .as_ref()
                .and_then(job_error_text)
                .unwrap_or_else(|| error_line(&stdout, &stderr));
            return Err(CliError::core(CliErrorKind::Exit(code), msg));
        }
        let value = last.ok_or_else(|| {
            CliError::core(
                CliErrorKind::BadJson,
                format!(
                    "no JSON object on stdout — first bytes: {}",
                    head(&stdout, 120)
                ),
            )
        })?;
        Ok(CliOutput {
            value,
            fallback_warnings: fallback_lines(&stderr),
        })
    }

    /// Run `abstractcore email <args>` (the caller includes `--json`).
    ///
    /// The email verbs print ONE JSON document and exit 0 (ok), 1
    /// (error) or 2 (refused: a failed connection test, a missing
    /// `--yes`). A secret never rides argv (readable by every local user
    /// while the command runs): it goes in `stdin` (`--password-stdin`,
    /// `--client-secret-stdin`), written as one line, then stdin is
    /// closed. Errors name `redacted_label`, never the args. A refusal's message
    /// is the CLI's own typed words: `cause — Fix: fix` from
    /// `{ok:false, error:{code, cause, fix}}`, or the first failed leg
    /// of a connection test.
    pub fn run_email(
        &self,
        args: &[&str],
        stdin: Option<&str>,
        redacted_label: &str,
        timeout: Duration,
    ) -> Result<CliOutput, CliError> {
        let (status, stdout, stderr) =
            run_raw_at_stdin(&self.bin, args, stdin, redacted_label, timeout)?;
        email_outcome(status, &stdout, &stderr)
    }

    /// `run_email` for a verb that waits on a person (`email connect
    /// --oauth …`): every stderr line that is `{"oauth_prompt": {…}}`
    /// reaches `on_prompt` WHILE the command runs (the sign-in code or
    /// address the person needs), and the command is killed as soon as
    /// `cancel` turns true.
    pub fn run_email_streaming(
        &self,
        args: &[&str],
        stdin: Option<&str>,
        redacted_label: &str,
        timeout: Duration,
        on_prompt: Box<dyn Fn(Value) + Send>,
        cancel: &AtomicBool,
    ) -> Result<CliOutput, CliError> {
        let on_line: Box<dyn FnMut(&str) + Send> = Box::new(move |line: &str| {
            if let Some(p) = oauth_prompt_of(line) {
                on_prompt(p);
            }
        });
        let (status, stdout, stderr) = run_raw_streaming_at(
            &self.bin,
            args,
            stdin,
            redacted_label,
            timeout,
            on_line,
            cancel,
        )?;
        email_outcome(status, &stdout, &stderr)
    }

    /// `run_raw_at` on this CLI's binary (no stdin).
    fn run_raw(
        &self,
        args: &[&str],
        redacted_label: &str,
        timeout: Duration,
    ) -> Result<(std::process::ExitStatus, String, String), CliError> {
        run_raw_at(&self.bin, args, redacted_label, timeout)
    }
}

/// The outcome of one email verb: its JSON document, or the CLI's own
/// typed words when it refused.
fn email_outcome(
    status: std::process::ExitStatus,
    stdout: &str,
    stderr: &str,
) -> Result<CliOutput, CliError> {
    let doc = last_json_object(stdout);
    if !status.success() {
        let code = status.code().unwrap_or(-1);
        let msg = doc
            .as_ref()
            .and_then(email_error_text)
            .unwrap_or_else(|| error_line(stdout, stderr));
        return Err(CliError::core(CliErrorKind::Exit(code), msg));
    }
    let value = doc.ok_or_else(|| {
        CliError::core(
            CliErrorKind::BadJson,
            format!(
                "no JSON object on stdout — first bytes: {}",
                head(stdout, 120)
            ),
        )
    })?;
    Ok(CliOutput {
        value,
        fallback_warnings: fallback_lines(stderr),
    })
}

/// The `oauth_prompt` object of one stderr line of `abstractcore email
/// connect --oauth … --json` (`{"oauth_prompt": {"flow": "device",
/// "user_code", "verification_uri", …}}` or `{"flow": "loopback",
/// "authorization_url", …}`); None for any other line.
pub fn oauth_prompt_of(line: &str) -> Option<Value> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    v.get("oauth_prompt").filter(|p| p.is_object()).cloned()
}

/// Run the command in its own process group (Unix), so a kill reaches
/// everything it started: a shell's children keep the output pipes open
/// otherwise, and the reader threads would wait for them.
fn own_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(not(unix))]
    {
        let _ = command;
    }
}

/// The subprocess every runner spawns: `bin args…`, stdout and stderr
/// piped, in its own process group. stdin is piped only when the caller
/// has data for it (a secret), else null.
pub(crate) fn build_command(bin: &Path, args: &[&str], stdin_piped: bool) -> Command {
    let mut command = Command::new(bin);
    command
        .args(args)
        .stdin(if stdin_piped {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    own_process_group(&mut command);
    command
}

/// Write `data` to the child's stdin on a thread, then close stdin (the
/// reader sees EOF after the line). A child that exits without reading
/// only breaks the pipe; the error is ignored.
fn feed_stdin(child: &mut std::process::Child, data: Option<&str>) {
    if let (Some(mut pipe), Some(data)) = (child.stdin.take(), data) {
        let data = data.to_string();
        std::thread::spawn(move || {
            let _ = pipe.write_all(data.as_bytes());
            drop(pipe);
        });
    }
}

/// Kill the command and every process in its group, then reap it.
fn kill_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        // The child is not reaped yet, so its pid (= its group id, from
        // `process_group(0)`) cannot have been reused. Signal the whole
        // group directly: no external `kill` binary, no PATH lookup.
        if let Ok(pgid) = i32::try_from(child.id()) {
            if pgid > 1 {
                // SAFETY: plain syscall with an integer group id.
                unsafe {
                    libc::kill(-pgid, libc::SIGKILL);
                }
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// `run_raw_at` that hands each stderr line to `on_line` as it arrives
/// and kills the process when `cancel` turns true.
pub(crate) fn run_raw_streaming_at(
    bin: &Path,
    args: &[&str],
    stdin: Option<&str>,
    redacted_label: &str,
    timeout: Duration,
    mut on_line: Box<dyn FnMut(&str) + Send>,
    cancel: &AtomicBool,
) -> Result<(std::process::ExitStatus, String, String), CliError> {
    let mut child = build_command(bin, args, stdin.is_some())
        .spawn()
        .map_err(|e| CliError::core(CliErrorKind::Spawn, format!("{}: {e}", bin.display())))?;
    feed_stdin(&mut child, stdin);

    let stdout = child.stdout.take().expect("piped");
    let stderr = child.stderr.take().expect("piped");
    let out_h = std::thread::spawn(move || read_all(stdout));
    let err_h = std::thread::spawn(move || {
        let mut all = String::new();
        let mut reader = BufReader::new(stderr);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf).into_owned();
                    on_line(&line);
                    all.push_str(&line);
                }
            }
        }
        all
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                let cancelled = cancel.load(Ordering::SeqCst);
                if cancelled || Instant::now() >= deadline {
                    kill_tree(&mut child);
                    // The readers end on their own once every writer of the
                    // pipes is gone; never block the caller on a straggler.
                    drop(out_h);
                    drop(err_h);
                    return Err(if cancelled {
                        CliError::core(
                            CliErrorKind::Cancelled,
                            format!("cancelled: {redacted_label}"),
                        )
                    } else {
                        CliError::core(
                            CliErrorKind::Timeout,
                            format!("no answer within {}s: {redacted_label}", timeout.as_secs()),
                        )
                    });
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(CliError::core(CliErrorKind::Spawn, e.to_string())),
        }
    };
    let stdout = out_h.join().unwrap_or_default();
    let stderr = err_h.join().unwrap_or_default();
    Ok((status, stdout, stderr))
}

/// Shared subprocess mechanics: spawn, drain both pipes on reader
/// threads (a large payload can never deadlock), wait with a deadline,
/// kill the whole process group on overrun.
pub(crate) fn run_raw_at(
    bin: &Path,
    args: &[&str],
    redacted_label: &str,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, String, String), CliError> {
    run_raw_at_stdin(bin, args, None, redacted_label, timeout)
}

/// `run_raw_at` that writes `stdin` (a secret line) to the child, then
/// closes its stdin.
pub(crate) fn run_raw_at_stdin(
    bin: &Path,
    args: &[&str],
    stdin: Option<&str>,
    redacted_label: &str,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, String, String), CliError> {
    let mut child = build_command(bin, args, stdin.is_some())
        .spawn()
        .map_err(|e| CliError::core(CliErrorKind::Spawn, format!("{}: {e}", bin.display())))?;
    feed_stdin(&mut child, stdin);

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
                    kill_tree(&mut child);
                    // The readers end on their own once every writer of the
                    // pipes is gone; never block the caller on a straggler.
                    drop(out_h);
                    drop(err_h);
                    return Err(CliError::core(
                        CliErrorKind::Timeout,
                        format!("no answer within {}s: {redacted_label}", timeout.as_secs()),
                    ));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(CliError::core(CliErrorKind::Spawn, e.to_string())),
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

/// The typed words of a refused email verb: `{error:{cause, fix}}`, a
/// refusal `message`, or the first failed leg of `email test`.
pub fn email_error_text(doc: &Value) -> Option<String> {
    let words = |v: &Value| -> Option<String> {
        let cause = v.get("cause").and_then(Value::as_str)?;
        Some(match v.get("fix").and_then(Value::as_str) {
            Some(fix) if !fix.is_empty() => format!("{cause} — Fix: {fix}"),
            _ => cause.to_string(),
        })
    };
    if let Some(e) = doc.get("error") {
        if let Some(w) = words(e) {
            return Some(w);
        }
    }
    for leg in ["imap", "smtp"] {
        if let Some(l) = doc.get(leg) {
            if l.get("ok").and_then(Value::as_bool) == Some(false) {
                let name = if leg == "imap" { "IMAP" } else { "SMTP" };
                return words(l).map(|w| format!("{name}: {w}"));
            }
        }
    }
    doc.get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn read_all(mut r: impl Read) -> String {
    let mut buf = Vec::new();
    let _ = r.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// The most useful single error line from a failed CLI run: the CLI's
/// own `❌ Error:` line (stdout) first, else the first non-empty stderr
/// line, else a generic head of whatever was printed.
pub(crate) fn error_line(stdout: &str, stderr: &str) -> String {
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

/// The whole stdout as one JSON object, else its LAST line that is one
/// (NDJSON streams end with the final document).
pub(crate) fn last_json_object(stdout: &str) -> Option<Value> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return v.is_object().then_some(v);
    }
    trimmed
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
        .filter(Value::is_object)
}

/// A job document's own failure text: `error` (a string or `{message}`),
/// else `message`.
fn job_error_text(v: &Value) -> Option<String> {
    let err = v.get("error");
    let text = err
        .and_then(Value::as_str)
        .or_else(|| err.and_then(|e| e.get("message")).and_then(Value::as_str))
        .or_else(|| v.get("message").and_then(Value::as_str))?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

pub(crate) fn head(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    t.replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_resolution_order() {
        let home = Path::new("/home/u");
        let cwd = Path::new("/work/proj");
        // Explicit override wins, even without an existence check —
        // and over the legacy alias when both are set.
        let info = resolve_bin(
            &|k| match k {
                "ABSTRACTCORE_CLI" => Some("/opt/ac".to_string()),
                "ABSTRACTCORE_BIN" => Some("/opt/legacy".to_string()),
                _ => None,
            },
            home,
            cwd,
            &|_| false,
        )
        .unwrap();
        assert_eq!(info.bin, PathBuf::from("/opt/ac"));
        assert_eq!(info.source, "$ABSTRACTCORE_CLI");
        let info = resolve_bin(
            &|k| (k == "ABSTRACTCORE_BIN").then(|| "/opt/legacy".to_string()),
            home,
            cwd,
            &|_| false,
        )
        .unwrap();
        assert_eq!(info.bin, PathBuf::from("/opt/legacy"));

        // PATH scan finds the first hit — before the shim and the venv.
        let info = resolve_bin(
            &|k| (k == "PATH").then(|| "/a:/b".to_string()),
            home,
            cwd,
            &|p| {
                p == Path::new("/b/abstractcore")
                    || p.starts_with("/home")
                    || p.starts_with("/work")
            },
        )
        .unwrap();
        assert_eq!(info.bin, PathBuf::from("/b/abstractcore"));
        assert_eq!(info.source, "PATH");

        // uv tool shim, then ./.venv relative to the CURRENT directory.
        let info = resolve_bin(&|_| None, home, cwd, &|p| {
            p == Path::new("/home/u/.local/bin/abstractcore")
                || p == Path::new("/work/proj/.venv/bin/abstractcore")
        })
        .unwrap();
        assert_eq!(info.bin, PathBuf::from("/home/u/.local/bin/abstractcore"));
        let info = resolve_bin(&|_| None, home, cwd, &|p| {
            p == Path::new("/work/proj/.venv/bin/abstractcore")
        })
        .unwrap();
        assert_eq!(info.bin, PathBuf::from("/work/proj/.venv/bin/abstractcore"));
        assert_eq!(info.source, "./.venv");

        // No personal-workspace path is probed any more.
        assert!(resolve_bin(&|_| None, home, cwd, &|p| {
            p == Path::new("/home/u/tmp/abstractframework/.venv/bin/abstractcore")
        })
        .is_none());

        // Nothing anywhere: honest None.
        assert!(resolve_bin(&|_| None, home, cwd, &|_| false).is_none());
    }

    #[test]
    fn email_refusals_speak_in_the_clis_typed_words() {
        use serde_json::json;
        let refused = json!({"ok": false, "error": {"code": "email_auth_failed",
            "cause": "The IMAP server rejected the user name or password.", "fix": "Check the password."}});
        assert_eq!(
            email_error_text(&refused).as_deref(),
            Some("The IMAP server rejected the user name or password. — Fix: Check the password.")
        );
        let test = json!({"imap": {"ok": true}, "smtp": {"ok": false, "cause": "TLS failed.", "fix": "Check the port."}, "ok": false});
        assert_eq!(
            email_error_text(&test).as_deref(),
            Some("SMTP: TLS failed. — Fix: Check the port.")
        );
        let msg =
            json!({"ok": false, "status": "refused", "message": "Re-run with --yes to confirm."});
        assert_eq!(
            email_error_text(&msg).as_deref(),
            Some("Re-run with --yes to confirm.")
        );
        assert_eq!(email_error_text(&json!({"ok": true})), None);
    }

    #[test]
    fn error_line_prefers_the_cli_error() {
        let out = "some noise\n❌ Error: Unknown provider 'x'\nmore";
        assert_eq!(error_line(out, ""), "❌ Error: Unknown provider 'x'");
        assert_eq!(
            error_line("", "Traceback...\nValueError: boom"),
            "ValueError: boom"
        );
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

    /// The streaming download verb prints `host_job_v1` NDJSON progress
    /// and the final job LAST; only that last line is the answer. A
    /// failed job exits 1 and its own error text is surfaced.
    #[test]
    fn run_json_last_takes_the_final_ndjson_line() {
        let sh = CoreCli::new(PathBuf::from("/bin/sh"));
        let ok = sh
            .run_json_last(
                &[
                    "-c",
                    r#"echo '{"schema":"host_job_v1","status":"running","percent":10.0}'
echo '{"schema":"host_job_v1","status":"running","percent":90.0}'
echo '{"schema":"host_job_v1","status":"completed","ok":true,"results":[{"status":"downloaded","message":"done"}]}'"#,
                ],
                Duration::from_secs(5),
            )
            .unwrap();
        assert_eq!(ok.value["status"], "completed");
        assert_eq!(ok.value["results"][0]["status"], "downloaded");

        // The old one-document reader would choke on exactly this stdout.
        assert_eq!(
            sh.run_json(
                &["-c", "echo '{\"a\":1}'; echo '{\"a\":2}'"],
                Duration::from_secs(5)
            )
            .unwrap_err()
            .kind,
            CliErrorKind::BadJson
        );

        let failed = sh
            .run_json_last(
                &[
                    "-c",
                    r#"echo '{"schema":"host_job_v1","status":"running"}'
echo '{"schema":"host_job_v1","status":"failed","error":"pull failed: manifest unknown"}'
exit 1"#,
                ],
                Duration::from_secs(5),
            )
            .unwrap_err();
        assert_eq!(failed.kind, CliErrorKind::Exit(1));
        assert!(failed.to_string().contains("manifest unknown"), "{failed}");

        let empty = sh
            .run_json_last(&["-c", "echo not json"], Duration::from_secs(5))
            .unwrap_err();
        assert_eq!(empty.kind, CliErrorKind::BadJson);
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

    /// An OAuth2 connect waits on a person: its `oauth_prompt` stderr
    /// line reaches the callback WHILE the command still runs (the
    /// script only exits after the prompt was delivered), other stderr
    /// lines do not, and the final document is the answer.
    #[test]
    fn email_streaming_delivers_the_prompt_before_the_answer() {
        use std::sync::mpsc;
        let sh = CoreCli::new(PathBuf::from("/bin/sh"));
        let (tx, rx) = mpsc::channel::<Value>();
        let never = AtomicBool::new(false);
        let out = sh
            .run_email_streaming(
                &[
                    "-c",
                    r#"echo 'plain noise' >&2
echo '{"oauth_prompt": {"flow": "device", "user_code": "WDJB-MJHT", "verification_uri": "https://example.test/device"}}' >&2
sleep 0.3
echo '{"ok": true, "schema": "email_settings_v1", "auth_kind": "oauth2"}'"#,
                ],
                None,
                "email connect --oauth",
                Duration::from_secs(10),
                Box::new(move |p| {
                    let _ = tx.send(p);
                }),
                &never,
            )
            .unwrap();
        assert_eq!(out.value["auth_kind"], "oauth2");
        let prompts: Vec<Value> = rx.try_iter().collect();
        assert_eq!(prompts.len(), 1, "{prompts:?}");
        assert_eq!(prompts[0]["user_code"], "WDJB-MJHT");
        assert_eq!(oauth_prompt_of("plain noise"), None);
        assert_eq!(
            oauth_prompt_of("{\"oauth_prompt\": \"not an object\"}"),
            None
        );
    }

    /// Cancel kills the waiting command: a typed `Cancelled`, not a
    /// timeout, and long before the deadline.
    #[test]
    fn email_streaming_cancel_kills_the_command() {
        let sh = CoreCli::new(PathBuf::from("/bin/sh"));
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            c2.store(true, Ordering::SeqCst);
        });
        let started = Instant::now();
        let err = sh
            .run_email_streaming(
                // `; true` keeps the shell alive as the parent of `sleep` on every
                // platform (a shell may exec a lone last command), so the test
                // proves the whole process group is killed.
                &["-c", "sleep 30; true"],
                None,
                "email connect --oauth",
                Duration::from_secs(60),
                Box::new(|_| {}),
                &cancel,
            )
            .unwrap_err();
        assert_eq!(err.kind, CliErrorKind::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(10));
    }
}

#[cfg(all(test, unix))]
mod stdin_secret_tests {
    use super::*;
    use crate::ui::email::{connect_action, oauth_action, ConnectFields, OAuthFields};
    use crate::worker::run_email_action;

    /// Every run here is bounded: a child left waiting on an open stdin
    /// fails the test in seconds, not after the 20-minute sign-in limit.
    const BOUND: Duration = Duration::from_secs(10);

    /// A fake `abstractcore`: reads ONE stdin line, then the rest until
    /// EOF (so it only answers once stdin is closed), and prints the hex
    /// of the line, of the rest and of its own argv.
    const FAKE_CLI: &str = r#"#!/bin/sh
[ -n "$FAKE_CLI_PROBE" ] && exit 0
IFS= read -r line
rest=$(cat)
hex() { printf '%s' "$1" | od -An -tx1 | tr -d ' \n'; }
printf '{"ok": true, "line": "%s", "rest": "%s", "argv": "%s"}\n' "$(hex "$line")" "$(hex "$rest")" "$(hex "$*")"
"#;

    /// The fake CLI as an executable file in a fresh temp directory.
    fn fake_cli() -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!(
            "acore-console-fake-cli-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("abstractcore");
        std::fs::write(&bin, FAKE_CLI).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        // Linux refuses to exec a file some process still holds open for
        // writing (ETXTBSY). Another test thread that forks while `write`
        // above had the file open briefly inherits that descriptor until
        // its own exec, so probe until the file runs once.
        for _ in 0..100 {
            match Command::new(&bin)
                .env("FAKE_CLI_PROBE", "1")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
            {
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                _ => break,
            }
        }
        bin
    }

    fn unhex(s: &str) -> String {
        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect();
        String::from_utf8(bytes).unwrap()
    }

    /// What the fake saw: (stdin line, rest of stdin, argv).
    fn seen(out: &CliOutput) -> (String, String, String) {
        let f = |k: &str| unhex(out.value[k].as_str().unwrap());
        (f("line"), f("rest"), f("argv"))
    }

    /// The TUI's Save and test action, run by the worker's runner against
    /// the fake CLI: the password reaches the child on stdin (exact bytes,
    /// spaces kept), stdin is closed after that one line, and argv never
    /// holds it.
    #[test]
    fn the_password_reaches_the_child_on_stdin_and_never_in_argv() {
        let pw = "  pw-SENTINEL-c0de é \"x\" ";
        let action = connect_action(
            &ConnectFields {
                address: "me@example.test".into(),
                password: pw.into(),
                imap_host: "imap.example.test".into(),
                ..ConnectFields::default()
            },
            None,
        )
        .unwrap();
        let cli = CoreCli::new(fake_cli());
        let never = AtomicBool::new(false);
        let out = run_email_action(&cli, &action, BOUND, Box::new(|_| {}), &never).unwrap();
        let (line, rest, argv) = seen(&out);
        assert_eq!(line, pw);
        assert_eq!(rest, "", "exactly one line on stdin, then EOF");
        assert!(
            argv.starts_with("email connect --address=me@example.test --password-stdin"),
            "{argv}"
        );
        assert!(!argv.contains("SENTINEL"), "{argv}");
    }

    /// The OAuth2 sign-in (the streaming runner): the client secret
    /// reaches the child on stdin, never in argv.
    #[test]
    fn the_client_secret_reaches_the_child_on_stdin_and_never_in_argv() {
        let action = oauth_action(
            &OAuthFields {
                address: "me@example.test".into(),
                client_id: "mine".into(),
                client_secret: "cs-SENTINEL-51".into(),
                ..OAuthFields::default()
            },
            None,
        )
        .unwrap();
        assert!(action.oauth);
        let cli = CoreCli::new(fake_cli());
        let never = AtomicBool::new(false);
        let out = run_email_action(&cli, &action, BOUND, Box::new(|_| {}), &never).unwrap();
        let (line, rest, argv) = seen(&out);
        assert_eq!(line, "cs-SENTINEL-51");
        assert_eq!(rest, "");
        assert!(argv.contains("--client-secret-stdin"), "{argv}");
        assert!(!argv.contains("SENTINEL"), "{argv}");
    }

    /// Without a secret stdin is null: the reader sees EOF at once.
    #[test]
    fn no_secret_means_null_stdin() {
        let cli = CoreCli::new(fake_cli());
        let out = cli
            .run_email(&["status"], None, "email status", Duration::from_secs(10))
            .unwrap();
        let (line, rest, argv) = seen(&out);
        assert_eq!(
            (line.as_str(), rest.as_str(), argv.as_str()),
            ("", "", "status")
        );
    }
}
