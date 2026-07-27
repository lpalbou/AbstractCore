//! The worker thread: owns every file read/write and every
//! `abstractcore` subprocess. The UI thread owns all signals; this
//! thread never touches them directly — it posts closures through
//! `WakeHandle`, which run on the UI thread in the next frame's user
//! phase.
//!
//! Serial by design: one command at a time, total order. Writes are
//! three-phase BY CONSTRUCTION — refuse-unless-writable → run verbs →
//! verify against a FRESH re-read (+ fresh derived views) → journal —
//! so a screen cannot forget the verification (the CLI's own success
//! signals lie; docs/backlog/proposed/0001_write_lane_design.md).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::time::Duration;

use abstracttui::reactive::WakeHandle;
use serde_json::Value;

use crate::cli::{CliError, CliErrorKind, CoreCli};
use crate::config::{self, ConfigPath, FileState};
use crate::store::{
    models_from_payload, ConfigMirror, JournalEntry, Loadable, ProfilesData, RoutesData, Store,
};
use crate::writes::{eval_file_expect, Arg, Expect, WriteSpec, WriteVerb};

static OP_ID: AtomicU64 = AtomicU64::new(1);

fn next_op() -> u64 {
    OP_ID.fetch_add(1, Ordering::Relaxed)
}

static FORM_ID: AtomicU64 = AtomicU64::new(1);

/// Correlates a write with the form modal that issued it (close on
/// success, stay open with data intact on failure).
pub fn next_form_id() -> u64 {
    FORM_ID.fetch_add(1, Ordering::Relaxed)
}

/// Commands the UI sends to the worker. Debug is test surface —
/// secrets ride `writes::Arg::Secret`, which redacts structurally.
#[derive(Clone, Debug)]
pub enum Cmd {
    /// Re-read + re-fold the config file.
    LoadConfig,
    /// `abstractcore config defaults --json` → routes.
    LoadRoutes,
    /// `abstractcore config providers --json` → profiles.
    LoadProfiles,
    /// Per-provider model list for pickers (`config models P --json`).
    LoadModels { provider: String },
    /// One verified write action.
    Write(Box<WriteSpec>),
    /// One test verb (M3): live model discovery, route membership, or
    /// a cheap generation. Evidence lands in `store.tests`.
    Probe(crate::probes::ProbeSpec),
}

/// Reads are Python-startup-bound (~1.5s observed); 30s is a hung
/// interpreter, not a slow one.
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// Model listing may do live discovery against a slow local server.
const MODELS_TIMEOUT: Duration = Duration::from_secs(60);
/// Setters are one Python startup + one file write.
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);
/// A generation may LOAD a local model first (21s observed for a 4B);
/// the busy strip shows elapsed the whole way.
const GENERATE_TIMEOUT: Duration = Duration::from_secs(120);
/// TCP reachability disambiguation — local endpoints answer instantly.
const REACH_TIMEOUT: Duration = Duration::from_millis(1500);

/// Form-completion sink (constructed in lib.rs over the UI signal).
pub type DoneSink = Box<dyn Fn(u64, Result<String, String>) + Send>;

pub fn spawn(
    store: Store,
    wake: WakeHandle,
    rx: Receiver<Cmd>,
    config_path: ConfigPath,
    cli: Option<CoreCli>,
    done: DoneSink,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while let Ok(cmd) = rx.recv() {
            // A panic in one command must not silently kill the lane:
            // catch, report loudly, keep serving.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                handle(&store, &wake, &config_path, cli.as_ref(), &cmd, &done);
            }));
            if let Err(p) = result {
                let msg = panic_text(&p);
                let cmd_dbg = format!("{cmd:?}");
                wake.post(move || {
                    // The worker is SERIAL: at most the panicked
                    // command's own ops are in flight — clearing all
                    // busy entries cannot cancel anyone else's, and a
                    // leaked ghost op would keep the 500ms ticker
                    // alive forever (review P3-13). The probe flag
                    // rides the same lifecycle: stuck true would
                    // refuse every future test verb.
                    store.busy.update(|ops| ops.clear());
                    store.probe_busy.set(false);
                    store.notice.set(Some(format!(
                        "internal error while handling {cmd_dbg}: {msg}"
                    )));
                });
            }
        }
    })
}

fn handle(
    store: &Store,
    wake: &WakeHandle,
    config_path: &ConfigPath,
    cli: Option<&CoreCli>,
    cmd: &Cmd,
    done: &DoneSink,
) {
    match cmd {
        Cmd::LoadConfig => {
            let op = next_op();
            begin(store, wake, op, "reading config file");
            let state = config::load(&config_path.path);
            let mirror = ConfigMirror {
                path: config_path.clone(),
                state,
                loaded_at: crate::store::now_hms(),
            };
            let store = *store;
            wake.post(move || {
                store.end_busy(op);
                store.cfg.set(Loadable::Ready(mirror.clone()));
            });
        }
        Cmd::LoadRoutes => {
            load_derived(
                store,
                wake,
                cli,
                "loading routes (config defaults --json)",
                &["config", "defaults", "--json"],
                |store, outcome| match outcome {
                    Ok(v) => store.routes.set(Loadable::Ready(RoutesData::from_value(&v))),
                    Err(e) => store.routes.set(Loadable::Failed(e)),
                },
            );
        }
        Cmd::LoadProfiles => {
            load_derived(
                store,
                wake,
                cli,
                "loading profiles (config providers --json)",
                &["config", "providers", "--json"],
                |store, outcome| match outcome {
                    Ok(v) => store
                        .profiles
                        .set(Loadable::Ready(ProfilesData::from_value(&v))),
                    Err(e) => store.profiles.set(Loadable::Failed(e)),
                },
            );
        }
        Cmd::LoadModels { provider } => {
            let provider = provider.clone();
            let outcome = match cli {
                None => Err(no_cli_error()),
                Some(cli) => {
                    let op = next_op();
                    begin(store, wake, op, &format!("listing models for {provider}"));
                    let r = cli
                        .run_json(&["config", "models", &provider, "--json"], MODELS_TIMEOUT)
                        .map(|o| models_from_payload(&o.value));
                    let store = *store;
                    wake.post(move || store.end_busy(op));
                    r
                }
            };
            let store = *store;
            wake.post(move || {
                store.models.update(|m| {
                    m.insert(
                        provider.clone(),
                        match &outcome {
                            Ok(models) => Loadable::Ready(models.clone()),
                            Err(e) => Loadable::Failed(e.clone()),
                        },
                    );
                });
            });
        }
        Cmd::Write(spec) => handle_write(store, wake, config_path, cli, spec, done),
        Cmd::Probe(spec) => handle_probe(store, wake, config_path, cli, spec),
    }
}

// ---------------------------------------------------------------------
// The probe lane (M3 test verbs).
// ---------------------------------------------------------------------

fn handle_probe(
    store: &Store,
    wake: &WakeHandle,
    config_path: &ConfigPath,
    cli: Option<&CoreCli>,
    spec: &crate::probes::ProbeSpec,
) {
    use crate::probes::{self, ProbeKind, Verdict};

    let op = next_op();
    begin(store, wake, op, &spec.label);
    {
        let store = *store;
        wake.post(move || store.probe_busy.set(true));
    }

    let (verdict, detail) = match (&spec.kind, cli) {
        (_, None) => (Verdict::Failed, no_cli_error().to_string()),
        (ProbeKind::ListModels { target, reach }, Some(cli)) => {
            match cli.run_json(&["config", "test-provider", target, "--json"], MODELS_TIMEOUT) {
                Err(e) => (Verdict::Failed, e.to_string()),
                Ok(out) => {
                    // TCP evidence only on the ambiguous branch.
                    let count = out.value.get("count").and_then(Value::as_u64).unwrap_or(0);
                    let evidence = if count == 0 {
                        reach.as_ref().map(|hp| (hp, tcp_probe(hp)))
                    } else {
                        None
                    };
                    let (v, mut d) = probes::fold_list_models(
                        &out.value,
                        evidence.as_ref().map(|(hp, r)| (*hp, r)),
                    );
                    // For keyed cloud providers the empty listing is
                    // ALSO the silent no-API-key answer (openai's
                    // list_available_models returns [] keyless) — name
                    // the likely cause where no endpoint exists to
                    // check (M3 review P3-6).
                    if v == Verdict::NotProven
                        && reach.is_none()
                        && crate::probes::KEYED_CLOUD_PROVIDERS.contains(&target.as_str())
                    {
                        d.push_str("; for this keyed cloud provider it is also the answer when \
                                    no API key resolves (config or env)");
                    }
                    (v, d)
                }
            }
        }
        (
            ProbeKind::RouteCheck {
                provider,
                model,
                reach,
                ..
            },
            Some(cli),
        ) => {
            match cli.run_json(
                &["config", "test-provider", provider, "--json"],
                MODELS_TIMEOUT,
            ) {
                Err(e) => (Verdict::Failed, e.to_string()),
                Ok(out) => {
                    let count = out.value.get("count").and_then(Value::as_u64).unwrap_or(0);
                    let evidence = if count == 0 {
                        reach.as_ref().map(|hp| (hp, tcp_probe(hp)))
                    } else {
                        None
                    };
                    let (v, d) = probes::fold_route_check(
                        &out.value,
                        model,
                        evidence.as_ref().map(|(hp, r)| (*hp, r)),
                    );
                    // The label names only the route (evidence replaces
                    // per target); the tested pair rides the detail
                    // AFTER the cause — notices truncate from the
                    // right and endpoint+HF-style pairs are long (M3
                    // review P3-11).
                    (v, format!("{d} — {provider}/{model}"))
                }
            }
        }
        (ProbeKind::Generate { provider, model }, Some(cli)) => {
            probe_generation(config_path, cli, provider.as_deref(), model.as_deref())
        }
    };

    let store = *store;
    let label = spec.label.clone();
    let result = crate::probes::TestResult {
        when: crate::store::now_hms(),
        label: label.clone(),
        verdict,
        detail: detail.clone(),
    };
    wake.post(move || {
        store.end_busy(op);
        store.probe_busy.set(false);
        store.record_test(result.clone());
        store.push_journal(JournalEntry {
            when: result.when.clone(),
            action: label.clone(),
            // "NOT PROVEN — " is a rendering contract with the journal
            // view: review.rs renders Err rows starting with it under
            // the ? glyph, not ✗ (M3 review P3-5). Change both or
            // neither.
            outcome: match verdict {
                Verdict::Proven => Ok(detail.clone()),
                Verdict::NotProven => Err(format!("NOT PROVEN — {detail}")),
                Verdict::Failed => Err(detail.clone()),
            },
        });
        store.notice.set(Some(format!(
            "{} {}: {}",
            verdict.glyph(),
            label,
            crate::ui::util::ellipsize(&detail, 90)
        )));
    });
}

/// The generation probe. For the DEFAULT route (`provider/model` both
/// None) the config file must actually name one — probed 2026-07-25:
/// `abstractcore-chat` on an empty config silently invents a
/// huggingface default, so without this pre-check the test would pass
/// while the operator's own route is unconfigured.
fn probe_generation(
    config_path: &ConfigPath,
    cli: &CoreCli,
    provider: Option<&str>,
    model: Option<&str>,
) -> (crate::probes::Verdict, String) {
    use crate::probes::{fold_generation, Verdict};

    // Every branch may need the fresh raw file: the default resolution
    // AND the endpoint-profile expansion below.
    let raw = match config::load_with_raw(&config_path.path) {
        (FileState::Ready(_), Some(raw)) => raw,
        (FileState::Missing, _) => Value::Object(Default::default()),
        _ => {
            return (
                Verdict::Failed,
                "the config file is corrupt/unreadable — fix it before testing".into(),
            )
        }
    };
    let (provider, model) = match (provider, model) {
        (Some(p), Some(m)) => (p.to_string(), m.to_string()),
        _ => {
            let get = |k: &str| {
                raw.get("default_models")
                    .and_then(|d| d.get(k))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
            };
            match (get("global_provider"), get("global_model")) {
                (Some(p), Some(m)) => (p, m),
                _ => {
                    return (
                        Verdict::Failed,
                        "no default route configured (default_models.global_provider/model) — \
                         set one first (wizard step 2); testing without it would exercise a \
                         built-in fallback, not YOUR route"
                            .into(),
                    )
                }
            }
        }
    };

    // `endpoint:<id>` routes: the chat CLI's --provider argparse knows
    // only the 10 static names (utils/cli.py:2558 — M3 review P1-1), so
    // naming the endpoint verbatim dies on a usage error and a WORKING
    // route reads ✗ FAILED. Expand keyless profiles to their family +
    // base_url (both flags exist); keyed profiles are refused honestly
    // — the chat CLI cannot receive their key (argv never carries
    // secrets), and a guessed key lane would mint fresh 401 lies.
    let mut argv_provider = provider.clone();
    let mut base_url: Option<String> = None;
    if let Some(id) = provider.strip_prefix("endpoint:") {
        let row = raw
            .get("provider_profiles")
            .and_then(|p| p.get("profiles"))
            .and_then(|p| p.get(id));
        let Some(row) = row else {
            return (
                Verdict::Failed,
                format!(
                    "the route names endpoint:{id}, but no such profile exists in the file — \
                     a on Providers adds one"
                ),
            );
        };
        let s = |k: &str| {
            row.get(k)
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or("")
                .to_string()
        };
        if row.get("enabled").and_then(Value::as_bool) == Some(false) {
            return (
                Verdict::NotProven,
                format!("profile {id} is disabled — enable it (e on Providers) before testing"),
            );
        }
        if !s("api_key").is_empty() || !s("api_key_env_var").is_empty() {
            return (
                Verdict::NotProven,
                format!(
                    "profile {id} carries an API key the chat CLI cannot receive \
                     (abstractcore-chat knows no endpoint: providers, and argv must never \
                     carry secrets) — t on Providers proves its models; generation over \
                     keyed endpoint profiles needs chat-CLI endpoint support"
                ),
            );
        }
        let family = s("provider_family");
        let url = s("base_url");
        if family.is_empty() || url.is_empty() {
            return (
                Verdict::NotProven,
                format!("profile {id} lacks a family/base_url — e on Providers completes it"),
            );
        }
        argv_provider = family;
        base_url = Some(url);
    }

    let started = std::time::Instant::now();
    let mut args = vec![
        "--provider",
        argv_provider.as_str(),
        "--model",
        model.as_str(),
        "--prompt",
        "Reply with exactly: PONG",
        "--max-output-tokens",
        "24",
    ];
    if let Some(u) = &base_url {
        args.push("--base-url");
        args.push(u.as_str());
    }
    let out = cli.run_chat(&args, GENERATE_TIMEOUT);
    // A PATH-resolved chat binary may belong to a DIFFERENT install
    // than the abstractcore serving discovery — say so in the
    // evidence rather than silently mixing installs (M3 review P3-2).
    let provenance = if cli.chat_from_path {
        " [chat binary from PATH, not beside abstractcore]"
    } else {
        ""
    };
    match out {
        Err(e) => (Verdict::Failed, format!("{e}{provenance}")),
        Ok((stdout, _warnings)) => {
            let (v, d) = fold_generation(&stdout, started.elapsed().as_secs());
            let via = base_url
                .as_ref()
                .map(|u| format!(" (via {argv_provider} @ {u})"))
                .unwrap_or_default();
            (v, format!("{provider}/{model} — {d}{via}{provenance}"))
        }
    }
}

/// The worker's one socket lane: connect against a KNOWN http endpoint
/// (profile base_url or a documented local default — the spec builder
/// enforces which; see probes::endpoint_for).
fn tcp_probe(hp: &crate::probes::HostPort) -> crate::probes::Reach {
    use std::net::ToSocketAddrs;
    let addrs = match (hp.host.as_str(), hp.port).to_socket_addrs() {
        Ok(a) => a.collect::<Vec<_>>(),
        Err(e) => return crate::probes::Reach::Unresolvable(e.to_string()),
    };
    probe_addr_list(&addrs, REACH_TIMEOUT)
}

/// One reachability verdict over ALL resolved addresses: Connected if
/// ANY accepts; Refused only when every one refuses. `localhost`
/// resolves `::1` FIRST on this class of machine while local servers
/// (LM Studio, ollama) often bind IPv4 only — probing just the first
/// address reported "the server looks DOWN" for an UP server, the
/// exact evidence the count==0 branch exists to give (M3 review P1-2).
/// The error text prefers the IPv4 attempt (the family local servers
/// actually serve).
fn probe_addr_list(addrs: &[std::net::SocketAddr], timeout: Duration) -> crate::probes::Reach {
    use crate::probes::Reach;
    if addrs.is_empty() {
        return Reach::Unresolvable("no addresses".into());
    }
    let mut v4_err: Option<String> = None;
    let mut last_err: Option<String> = None;
    for addr in addrs {
        match std::net::TcpStream::connect_timeout(addr, timeout) {
            Ok(_) => return Reach::Connected,
            Err(e) => {
                if addr.is_ipv4() && v4_err.is_none() {
                    v4_err = Some(e.to_string());
                }
                last_err = Some(e.to_string());
            }
        }
    }
    Reach::Refused(v4_err.or(last_err).unwrap_or_else(|| "refused".into()))
}

fn no_cli_error() -> CliError {
    CliError::core(
        CliErrorKind::NotFound,
        "no $ABSTRACTCORE_BIN, nothing on PATH, no venv fallback".into(),
    )
}

// ---------------------------------------------------------------------
// The write lane.
// ---------------------------------------------------------------------

fn handle_write(
    store: &Store,
    wake: &WakeHandle,
    config_path: &ConfigPath,
    cli: Option<&CoreCli>,
    spec: &WriteSpec,
    done: &DoneSink,
) {
    let op = next_op();
    begin(store, wake, op, &spec.label);

    let outcome = execute_write(config_path, cli, spec);

    // Post-write state refresh: the mirror ALWAYS re-reads (even after
    // failure — the operator must see what the file holds now), and
    // the derived views refresh when the write touched them.
    let mirror = ConfigMirror {
        path: config_path.clone(),
        state: config::load(&config_path.path),
        loaded_at: crate::store::now_hms(),
    };
    let mut routes_new: Option<Loadable<RoutesData>> = None;
    let mut profiles_new: Option<Loadable<ProfilesData>> = None;
    let mut fallback_new: Option<Option<String>> = None;
    if let Some(cli) = cli {
        if spec.needs_routes() || outcome.is_ok() {
            match cli.run_json(&["config", "defaults", "--json"], READ_TIMEOUT) {
                Ok(out) => {
                    fallback_new = Some(out.fallback_warnings.first().cloned());
                    routes_new = Some(Loadable::Ready(RoutesData::from_value(&out.value)));
                }
                Err(e) => routes_new = Some(Loadable::Failed(e)),
            }
        }
        // Profiles refresh only when the write can touch them (key/profile
        // specs declare profile expects) — an unconditional refresh added
        // ~5-15s of CLI tail to EVERY write for views it cannot change.
        if spec.needs_profiles() {
            match cli.run_json(&["config", "providers", "--json"], READ_TIMEOUT) {
                Ok(out) => {
                    if fallback_new.is_none() {
                        fallback_new = Some(out.fallback_warnings.first().cloned());
                    }
                    profiles_new = Some(Loadable::Ready(ProfilesData::from_value(&out.value)));
                }
                Err(e) => profiles_new = Some(Loadable::Failed(e)),
            }
        }
    }

    // Derived-view expectations get their verdict from the fresh
    // payloads; a failed file verify already decided the outcome.
    let routes_data = routes_new.as_ref().and_then(|l| match l {
        Loadable::Ready(d) => Some(d),
        _ => None,
    });
    let profiles_data = profiles_new.as_ref().and_then(|l| match l {
        Loadable::Ready(d) => Some(d),
        _ => None,
    });
    let outcome = outcome.and_then(|mut proofs| {
        for ex in &spec.expects {
            if ex.needs_routes() || ex.needs_profiles() {
                proofs.push(eval_derived_expect(ex, routes_data, profiles_data)?);
            }
        }
        Ok(proofs)
    });

    let store = *store;
    let label = spec.label.clone();
    let form_id = spec.form_id;
    let done_outcome = match &outcome {
        Ok(proofs) => Ok(format!("verified: {}", proofs.join("; "))),
        Err(e) => Err(e.clone()),
    };
    let journal_outcome = done_outcome.clone();
    wake.post(move || {
        store.end_busy(op);
        store.cfg.set(Loadable::Ready(mirror.clone()));
        if let Some(r) = routes_new.clone() {
            store.routes.set(r);
        }
        if let Some(p) = profiles_new.clone() {
            store.profiles.set(p);
        }
        if let Some(fb) = fallback_new.clone() {
            store.python_fallback.set(fb);
        }
        store.push_journal(JournalEntry {
            when: crate::store::now_hms(),
            action: label.clone(),
            outcome: journal_outcome.clone(),
        });
        match &journal_outcome {
            Ok(v) => store.notice.set(Some(format!("✓ {label} — {v}"))),
            Err(e) => store.notice.set(Some(format!("✗ {label} — {e}"))),
        }
    });
    // The sink marshals to the UI thread itself (lib.rs builds it over
    // wake.post) — calling it here on the worker is the contract.
    if let Some(fid) = form_id {
        done(fid, done_outcome);
    }
}

/// One derived-view expectation over the FRESH payloads — pure, so the
/// verification the routes/profiles writes rely on is unit-testable
/// (M2 review: this lane had zero coverage).
pub(crate) fn eval_derived_expect(
    ex: &Expect,
    routes: Option<&RoutesData>,
    profiles: Option<&ProfilesData>,
) -> Result<String, String> {
    match ex {
        Expect::RouteEq {
            key,
            provider,
            model,
        } => {
            let row = routes
                .and_then(|d| d.rows.iter().find(|r| &r.key == key))
                .ok_or_else(|| format!("route {key} not found in the fresh view (CLI reload failed?)"))?;
            if !row.configured
                || (provider.is_some() && row.provider != *provider)
                || (model.is_some() && row.model != *model)
            {
                return Err(format!(
                    "route {key} verifies as {}/{} (configured: {})",
                    row.provider.as_deref().unwrap_or("—"),
                    row.model.as_deref().unwrap_or("—"),
                    row.configured
                ));
            }
            Ok(format!(
                "route {key} = {}/{}",
                row.provider.as_deref().unwrap_or("—"),
                row.model.as_deref().unwrap_or("—")
            ))
        }
        Expect::RouteCleared { key } => {
            match routes.and_then(|d| d.rows.iter().find(|r| &r.key == key)) {
                Some(r) if !r.configured => Ok(format!("route {key} cleared")),
                Some(_) => Err(format!("route {key} is still configured")),
                None => Err(format!(
                    "route {key} not found in the fresh view (CLI reload failed?)"
                )),
            }
        }
        Expect::ProfileExists { id } => {
            if profiles.is_some_and(|d| d.profiles.iter().any(|p| &p.id == id)) {
                Ok(format!("profile {id} present"))
            } else {
                Err(format!("profile {id} not present in the fresh view"))
            }
        }
        Expect::ProfileAbsent { id } => {
            if profiles.is_some_and(|d| d.profiles.iter().any(|p| &p.id == id)) {
                Err(format!("profile {id} still present"))
            } else {
                Ok(format!("profile {id} gone"))
            }
        }
        other => Err(format!("not a derived expectation: {other:?}")),
    }
}

/// Pre-checks + verbs + FILE verification. Route/profile verification
/// happens in the caller over the fresh derived views.
fn execute_write(
    config_path: &ConfigPath,
    cli: Option<&CoreCli>,
    spec: &WriteSpec,
) -> Result<Vec<String>, String> {
    // A spec with CLI verbs and no CLI refuses UP FRONT — before ANY
    // verb applies. The alternative (failing at the CLI verb's turn)
    // half-applies multi-verb specs and reports an error that implies
    // nothing happened (M2 review P2-2).
    let has_cli_verb = spec.verbs.iter().any(|v| matches!(v, WriteVerb::Cli(_)));
    if has_cli_verb && cli.is_none() {
        return Err(
            "abstractcore CLI not found — this write needs it ($ABSTRACTCORE_BIN); \
             nothing was changed"
                .into(),
        );
    }
    // 1. Refuse-unless-writable, on a FRESH read (risk-map fact #4:
    // never write over corrupt; plus the drift guard — no lock exists,
    // whole-file rewrites are last-writer-wins).
    let (state, _) = config::load_with_raw(&config_path.path);
    match &state {
        FileState::Corrupt { .. } => {
            return Err(
                "the config file is CORRUPT — writes are disabled until it is fixed \
                 (see the Overview for backups)"
                    .into(),
            )
        }
        FileState::Unreadable { error } => {
            return Err(format!("the config file is unreadable: {error}"))
        }
        FileState::Ready(snap) => {
            if has_cli_verb && !snap.python_refusals.is_empty() {
                // A CLI setter against a Python-refused file LOADS
                // DEFAULTS then SAVES — the historical data-loss
                // incident, executed by us. Structurally refused.
                return Err(format!(
                    "Python refuses this file ({}) — a CLI write would RESET it to defaults; \
                     fix the named rows first (Overview)",
                    snap.python_refusals.first().cloned().unwrap_or_default()
                ));
            }
            if spec.base_stamp != snap.stamp {
                return Err(
                    "the file changed since you loaded it — press r to reload, review, \
                     then retry"
                        .into(),
                );
            }
        }
        FileState::Missing => {
            if spec.base_stamp.is_some() {
                return Err(
                    "the config file DISAPPEARED since you loaded it — press r to reload"
                        .into(),
                );
            }
        }
    }

    // 2. Verbs, in order.
    for verb in &spec.verbs {
        match verb {
            WriteVerb::Cli(args) => {
                let cli = cli.expect("pre-checked above");
                let argv: Vec<&str> = args.iter().map(Arg::value).collect();
                let warnings = cli
                    .run_setter(&argv, &spec.label, WRITE_TIMEOUT)
                    .map_err(|e| e.to_string())?;
                if let Some(w) = warnings.first() {
                    // The pre-check makes this unreachable in theory;
                    // if Python still fell back mid-write, the file
                    // may now be defaults+this-change — say so loudly.
                    return Err(format!(
                        "Python reported #FALLBACK during the write — the file may have been \
                         RESET; press r and inspect immediately ({w})"
                    ));
                }
            }
            WriteVerb::Rmw(op) => rmw_write(config_path, op)?,
        }
    }

    // 3. Verify every FILE expectation against a fresh re-read.
    let (_, raw) = config::load_with_raw(&config_path.path);
    let raw = raw.ok_or("post-write re-read failed — the file is missing or corrupt now?!")?;
    let mut proofs = Vec::new();
    for ex in &spec.expects {
        if ex.needs_routes() || ex.needs_profiles() {
            continue;
        }
        proofs.push(eval_file_expect(&raw, ex)?);
    }
    Ok(proofs)
}

/// Direct read-modify-write: fresh read (unknown keys ride along in
/// the raw value), mutate, write via a UNIQUE tmp name + rename, 0600.
/// (Python uses a fixed `<file>.tmp` — sharing it would interleave
/// writers.)
fn rmw_write(config_path: &ConfigPath, op: &crate::writes::RmwOp) -> Result<(), String> {
    let path = &config_path.path;
    let (state, raw) = config::load_with_raw(path);
    let mut raw = match (&state, raw) {
        (FileState::Ready(_), Some(raw)) => raw,
        (FileState::Missing, _) => Value::Object(Default::default()),
        _ => return Err("the file is not writable in its current state".into()),
    };
    op.apply(&mut raw)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(&raw).map_err(|e| e.to_string())?;
    bytes.push(b'\n'); // Python writes a trailing newline
    let tmp = path.with_file_name(format!(
        "{}.tmp-console-{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("abstractcore.json"),
        std::process::id()
    ));
    std::fs::write(&tmp, &bytes).map_err(|e| format!("write {} failed: {e}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("chmod failed: {e}"))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename into place failed: {e}")
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

fn load_derived(
    store: &Store,
    wake: &WakeHandle,
    cli: Option<&CoreCli>,
    label: &str,
    args: &[&str],
    apply: impl FnOnce(&Store, Result<serde_json::Value, CliError>) + Send + 'static,
) {
    let Some(cli) = cli else {
        let store = *store;
        wake.post(move || {
            apply(&store, Err(no_cli_error()));
        });
        return;
    };
    let op = next_op();
    begin(store, wake, op, label);
    let action = format!("abstractcore {}", args.join(" "));
    let outcome = cli.run_json(args, READ_TIMEOUT);
    let journal = match &outcome {
        Ok(_) => Ok("ok".to_string()),
        Err(e) => Err(e.to_string()),
    };
    let store = *store;
    wake.post(move || {
        store.end_busy(op);
        // Reads journal only on failure — a green load is not an event
        // worth a journal row, a red one is.
        if journal.is_err() {
            store.push_journal(JournalEntry {
                when: crate::store::now_hms(),
                action,
                outcome: journal,
            });
        }
        // The P1-1 lane: an exit-0 run that printed `#FALLBACK` means
        // Python REFUSED the config file and answered from defaults —
        // record it (both loads hit the same file, so last-write-wins
        // is one consistent truth), or clear it on a clean run.
        if let Ok(out) = &outcome {
            store
                .python_fallback
                .set(out.fallback_warnings.first().cloned());
        }
        apply(&store, outcome.map(|o| o.value));
    });
}

fn begin(store: &Store, wake: &WakeHandle, op: u64, label: &str) {
    let store = *store;
    let label = label.to_string();
    wake.post(move || store.begin_busy(op, &label));
}

fn panic_text(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{resolve_config_path, PathSource};
    use crate::writes::{Expect, RmwOp};
    use serde_json::json;
    use std::path::PathBuf;

    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let dir = std::env::temp_dir().join(format!(
                "acc-worker-{tag}-{}-{}",
                std::process::id(),
                OP_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Scratch { dir }
        }
        fn cfg_path(&self) -> ConfigPath {
            ConfigPath {
                path: self.dir.join("abstractcore.json"),
                source: PathSource::Default,
            }
        }
        fn write(&self, v: &serde_json::Value) {
            std::fs::write(
                self.cfg_path().path,
                serde_json::to_vec_pretty(v).unwrap(),
            )
            .unwrap();
        }
        fn read(&self) -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(self.cfg_path().path).unwrap()).unwrap()
        }
        fn stamp(&self) -> Option<crate::config::FileStamp> {
            crate::config::FileStamp::of(&self.cfg_path().path)
        }
        /// A fake abstractcore: a shell script so the CLI verb lane is
        /// testable without Python. `body` sees the argv.
        fn fake_cli(&self, body: &str) -> CoreCli {
            let p = self.dir.join("fake-abstractcore");
            std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            CoreCli::new(p)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn spec_rmw(op: RmwOp, expects: Vec<Expect>, base: Option<crate::config::FileStamp>) -> WriteSpec {
        WriteSpec {
            label: "test rmw".into(),
            verbs: vec![WriteVerb::Rmw(op)],
            expects,
            base_stamp: base,
            form_id: None,
        }
    }

    #[test]
    fn rmw_write_preserves_unknown_keys_and_verifies() {
        let s = Scratch::new("rmw");
        s.write(&json!({
            "future_section": {"keep": 1},
            "logging": {"verbatim_enabled": true, "unknown_knob": "x"},
        }));
        let base = s.stamp();
        let spec = spec_rmw(
            RmwOp::SetField {
                section: "logging".into(),
                key: "verbatim_enabled".into(),
                value: json!(false),
            },
            vec![Expect::Eq {
                path: vec!["logging".into(), "verbatim_enabled".into()],
                value: json!(false),
            }],
            base,
        );
        let proofs = execute_write(&s.cfg_path(), None, &spec).expect("write ok");
        assert!(proofs[0].contains("verbatim_enabled = false"), "{proofs:?}");
        let after = s.read();
        assert_eq!(after["future_section"]["keep"], json!(1), "unknown section kept");
        assert_eq!(after["logging"]["unknown_knob"], json!("x"), "unknown key kept");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(s.cfg_path().path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "write keeps the file private");
        }
    }

    #[test]
    fn drift_guard_refuses_and_leaves_the_file_alone() {
        let s = Scratch::new("drift");
        s.write(&json!({"video": {"max_frames": 3}}));
        // A stale base identity (someone else wrote since the UI loaded).
        let stale = Some(crate::config::FileStamp {
            mtime: Some(std::time::SystemTime::UNIX_EPOCH),
            ino: 1,
            size: 1,
        });
        let spec = spec_rmw(
            RmwOp::SetField {
                section: "video".into(),
                key: "max_frames".into(),
                value: json!(9),
            },
            vec![],
            stale,
        );
        let err = execute_write(&s.cfg_path(), None, &spec).unwrap_err();
        assert!(err.contains("changed since you loaded"), "{err}");
        assert_eq!(s.read()["video"]["max_frames"], json!(3), "untouched");
    }

    #[test]
    fn corrupt_file_refuses_all_writes() {
        let s = Scratch::new("corrupt");
        std::fs::write(s.cfg_path().path, b"{ not json").unwrap();
        let spec = spec_rmw(
            RmwOp::SetField {
                section: "video".into(),
                key: "max_frames".into(),
                value: json!(9),
            },
            vec![],
            None,
        );
        let err = execute_write(&s.cfg_path(), None, &spec).unwrap_err();
        assert!(err.contains("CORRUPT"), "{err}");
        assert_eq!(std::fs::read(s.cfg_path().path).unwrap(), b"{ not json");
    }

    #[test]
    fn python_refused_file_blocks_cli_verbs_but_not_rmw() {
        let s = Scratch::new("refused");
        s.write(&json!({
            "provider_profiles": {"profiles": {"p": {"id": "p", "future_field": true}}},
            "logging": {"verbatim_enabled": true},
        }));
        let base = s.stamp();
        // A CLI verb would make Python load defaults + save = the
        // incident. Refused.
        let cli_spec = WriteSpec {
            label: "cli write".into(),
            verbs: vec![WriteVerb::Cli(vec![Arg::p("--set-video-strategy"), Arg::p("auto")])],
            expects: vec![],
            base_stamp: base,
            form_id: None,
        };
        let fake = s.fake_cli("echo should-not-run; exit 0");
        let err = execute_write(&s.cfg_path(), Some(&fake), &cli_spec).unwrap_err();
        assert!(err.contains("RESET it to defaults"), "{err}");

        // RMW preserves everything — allowed even on a refused file.
        let rmw = spec_rmw(
            RmwOp::SetField {
                section: "logging".into(),
                key: "verbatim_enabled".into(),
                value: json!(false),
            },
            vec![Expect::Eq {
                path: vec!["logging".into(), "verbatim_enabled".into()],
                value: json!(false),
            }],
            base,
        );
        execute_write(&s.cfg_path(), None, &rmw).expect("rmw allowed");
        let after = s.read();
        assert_eq!(
            after["provider_profiles"]["profiles"]["p"]["future_field"],
            json!(true),
            "the refused row is preserved, not dropped"
        );
    }

    #[test]
    fn cli_verb_runs_and_verify_catches_lies() {
        let s = Scratch::new("cli");
        s.write(&json!({"video": {"max_frames": 3}}));
        let base = s.stamp();
        let cfg = s.cfg_path();

        // An honest fake setter: mutates the file like Python would.
        let honest = s.fake_cli(&format!(
            "python3 -c \"import json; p='{}'; v=json.load(open(p)); \
             v['video']['max_frames']=9; json.dump(v, open(p,'w'))\"",
            cfg.path.display()
        ));
        let spec = WriteSpec {
            label: "set video.max_frames = 9".into(),
            verbs: vec![WriteVerb::Cli(vec![Arg::p("--set-video-max-frames"), Arg::p("9")])],
            expects: vec![Expect::Eq {
                path: vec!["video".into(), "max_frames".into()],
                value: json!(9),
            }],
            base_stamp: base,
            form_id: None,
        };
        let proofs = execute_write(&cfg, Some(&honest), &spec).expect("verified");
        assert!(proofs[0].contains("max_frames = 9"));

        // A LYING setter (exit 0, ❌ line, no write) — the probed
        // --set-server-port class. Caught before verify.
        s.write(&json!({"video": {"max_frames": 3}}));
        let liar = s.fake_cli("echo '❌ Error: Invalid value'; exit 0");
        let spec2 = WriteSpec {
            base_stamp: s.stamp(),
            ..spec.clone()
        };
        let err = execute_write(&cfg, Some(&liar), &spec2).unwrap_err();
        assert!(err.contains("Invalid value"), "{err}");

        // A SILENT liar (✅, exit 0, writes nothing) — the probed
        // --set-app-default class. Caught by verify-by-re-read.
        let silent = s.fake_cli("echo '✅ Set'; exit 0");
        let spec3 = WriteSpec {
            base_stamp: s.stamp(),
            ..spec
        };
        let err = execute_write(&cfg, Some(&silent), &spec3).unwrap_err();
        assert!(err.contains("is 3 (expected 9)"), "{err}");
    }

    #[test]
    fn missing_file_writes_create_it() {
        let s = Scratch::new("missing");
        let spec = spec_rmw(
            RmwOp::SetField {
                section: "offline".into(),
                key: "allow_network".into(),
                value: json!(true),
            },
            vec![Expect::Eq {
                path: vec!["offline".into(), "allow_network".into()],
                value: json!(true),
            }],
            None, // Missing file = no base mtime
        );
        execute_write(&s.cfg_path(), None, &spec).expect("created");
        assert_eq!(s.read()["offline"]["allow_network"], json!(true));
    }

    #[test]
    fn resolve_helpers_are_reachable() {
        // Anchor: the worker's path type matches the resolver's.
        let p = resolve_config_path(&|_| None, std::path::Path::new("/home/u"));
        assert_eq!(p.source, PathSource::Default);
    }

    /// The M2-review P1 pin: the fallback-chain builders' array-index
    /// expectations must evaluate against the REAL post-write file —
    /// the walker used to return None for every numeric segment
    /// (false-fail on add, vacuous-pass on remove).
    #[test]
    fn fallback_chain_writes_verify_with_array_paths() {
        let s = Scratch::new("chain");
        s.write(&json!({"vision": {"fallback_chain": [
            {"provider": "ollama", "model": "llava2"}
        ]}}));
        // Add: an honest fake that appends like Python does.
        let cfg = s.cfg_path();
        let honest = s.fake_cli(&format!(
            "python3 -c \"import json; p='{}'; v=json.load(open(p)); \
             v['vision']['fallback_chain'].append({{'provider':'lmstudio','model':'qwen-vl'}}); \
             json.dump(v, open(p,'w'))\"",
            cfg.path.display()
        ));
        let spec = crate::writes::add_vision_fallback("lmstudio", "qwen-vl", 2, s.stamp(), None);
        let proofs = execute_write(&cfg, Some(&honest), &spec)
            .expect("a landed add VERIFIES (was: false-fail on every add)");
        assert!(
            proofs[0].contains("fallback_chain.1.provider"),
            "{proofs:?}"
        );

        // Remove: the RMW op + a REAL check that the slot is gone —
        // and the inverse: a remove whose expectation is checked
        // against a file where the entry survived must FAIL.
        let spec = crate::writes::remove_vision_fallback(1, 2, s.stamp(), None);
        execute_write(&cfg, None, &spec).expect("remove verifies");
        let after = s.read();
        assert_eq!(after["vision"]["fallback_chain"].as_array().unwrap().len(), 1);

        // Vacuous-pass guard: expecting slot 0 cleared while it still
        // exists must error.
        let bogus = spec_rmw(
            RmwOp::SetField {
                section: "audio".into(),
                key: "stt_language".into(),
                value: json!("fr"),
            },
            vec![Expect::Cleared {
                path: vec![
                    "vision".into(),
                    "fallback_chain".into(),
                    "0".into(),
                    "provider".into(),
                ],
            }],
            s.stamp(),
        );
        let err = execute_write(&cfg, None, &bogus).unwrap_err();
        assert!(err.contains("still holds"), "{err}");
    }

    /// #FALLBACK printed to stderr DURING a write = Python refused the
    /// file mid-write (the reset case) — the last line of defense for
    /// the historical incident must fire loudly (was: zero coverage).
    #[test]
    fn fallback_stderr_during_a_write_fails_loudly() {
        let s = Scratch::new("fbwrite");
        s.write(&json!({"video": {"max_frames": 3}}));
        let fake = s.fake_cli(
            "echo '#FALLBACK abstractcore config could not be parsed; falling back to \
             DEFAULTS' >&2; echo '✅ Set'; exit 0",
        );
        let spec = WriteSpec {
            label: "set video.max_frames = 9".into(),
            verbs: vec![WriteVerb::Cli(vec![
                Arg::p("--set-video-max-frames"),
                Arg::p("9"),
            ])],
            expects: vec![],
            base_stamp: s.stamp(),
            form_id: None,
        };
        let err = execute_write(&s.cfg_path(), Some(&fake), &spec).unwrap_err();
        assert!(err.contains("may have been RESET"), "{err}");
        assert!(err.contains("press r and inspect"), "{err}");
    }

    /// A nonzero-exit setter (argparse refusals: the leading-dash
    /// class) surfaces its error through execute_write.
    #[test]
    fn nonzero_exit_setter_fails_through_the_lane() {
        let s = Scratch::new("exit2");
        s.write(&json!({"video": {"max_frames": 3}}));
        let fake = s.fake_cli("echo 'error: expected one argument' >&2; exit 2");
        let spec = WriteSpec {
            label: "set something".into(),
            verbs: vec![WriteVerb::Cli(vec![Arg::p("--set-video-max-frames")])],
            expects: vec![],
            base_stamp: s.stamp(),
            form_id: None,
        };
        let err = execute_write(&s.cfg_path(), Some(&fake), &spec).unwrap_err();
        assert!(err.contains("expected one argument"), "{err}");
    }

    /// Multi-verb order (M2 review P2-2): the fallible CLI half of
    /// clear_global_default runs FIRST; with no CLI available the spec
    /// refuses UP FRONT and the RMW half never applies.
    #[test]
    fn clear_global_default_is_cli_first_and_refuses_without_cli() {
        let spec = crate::writes::clear_global_default(None, None);
        assert!(
            matches!(spec.verbs[0], WriteVerb::Cli(_)),
            "CLI verb first: {:?}",
            spec.verbs
        );
        assert!(matches!(spec.verbs[1], WriteVerb::Rmw(_)));

        let s = Scratch::new("noclip");
        s.write(&json!({"default_models": {"global_provider": "lmstudio",
                                             "global_model": "m"}}));
        let spec = crate::writes::clear_global_default(s.stamp(), None);
        let err = execute_write(&s.cfg_path(), None, &spec).unwrap_err();
        assert!(err.contains("nothing was changed"), "{err}");
        assert_eq!(
            s.read()["default_models"]["global_provider"],
            json!("lmstudio"),
            "the RMW half must NOT have applied"
        );
    }

    /// The derived-view verification fold, unit-tested (M2 review:
    /// this lane had zero coverage).
    #[test]
    fn derived_expects_evaluate_against_fresh_views() {
        let routes = RoutesData::from_value(&json!({
            "ok": true, "routes": [
                {"key": "input.text", "provider": "lmstudio", "model": "m",
                 "configured": true, "kind": "input", "modality": "text",
                 "label": "Text Input", "source": "x"},
                {"key": "input.voice", "configured": false, "kind": "input",
                 "modality": "voice", "label": "Voice", "source": "not_configured"}
            ]
        }));
        let profiles = ProfilesData::from_value(&json!({
            "ok": true, "profiles": [{"id": "acme"}]
        }));

        let ok = eval_derived_expect(
            &Expect::RouteEq {
                key: "input.text".into(),
                provider: Some("lmstudio".into()),
                model: Some("m".into()),
            },
            Some(&routes),
            None,
        )
        .unwrap();
        assert!(ok.contains("input.text = lmstudio/m"));

        let err = eval_derived_expect(
            &Expect::RouteEq {
                key: "input.text".into(),
                provider: Some("ollama".into()),
                model: None,
            },
            Some(&routes),
            None,
        )
        .unwrap_err();
        assert!(err.contains("verifies as lmstudio/m"), "{err}");

        assert!(eval_derived_expect(
            &Expect::RouteCleared {
                key: "input.voice".into()
            },
            Some(&routes),
            None
        )
        .is_ok());
        assert!(eval_derived_expect(
            &Expect::RouteCleared {
                key: "input.text".into()
            },
            Some(&routes),
            None
        )
        .is_err());
        // A missing fresh view is an honest failure, not a pass.
        assert!(eval_derived_expect(
            &Expect::RouteCleared {
                key: "input.text".into()
            },
            None,
            None
        )
        .is_err());

        assert!(eval_derived_expect(
            &Expect::ProfileExists { id: "acme".into() },
            None,
            Some(&profiles)
        )
        .is_ok());
        assert!(eval_derived_expect(
            &Expect::ProfileAbsent { id: "acme".into() },
            None,
            Some(&profiles)
        )
        .is_err());
        assert!(eval_derived_expect(
            &Expect::ProfileAbsent { id: "gone".into() },
            None,
            Some(&profiles)
        )
        .is_ok());
    }

    // -----------------------------------------------------------------
    // The probe lane (M3).
    // -----------------------------------------------------------------

    impl Scratch {
        /// A CoreCli whose CHAT binary is a controlled script — the
        /// generation probe's mechanics without any real model.
        fn fake_chat(&self, body: &str) -> CoreCli {
            let p = self.dir.join("fake-abstractcore-chat");
            std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            CoreCli {
                bin: self.dir.join("unused-abstractcore"),
                chat_bin: Some(p),
                chat_from_path: false,
            }
        }
    }

    /// The default-route probe must test the OPERATOR'S route, never
    /// the chat CLI's silent built-in fallback (live-probed: an empty
    /// config generates via an invented huggingface default).
    #[test]
    fn generation_probe_refuses_without_a_default_route() {
        let s = Scratch::new("probe-nodefault");
        s.write(&json!({"version": "1.0"}));
        let cli = s.fake_chat("echo SHOULD-NEVER-RUN");
        let (v, d) = probe_generation(&s.cfg_path(), &cli, None, None);
        assert_eq!(v, crate::probes::Verdict::Failed);
        assert!(d.contains("no default route configured"), "{d}");
        assert!(d.contains("wizard"), "teaches the fix: {d}");

        // A MISSING file is the same honest refusal (fresh machine).
        let s2 = Scratch::new("probe-missing");
        let (v, d) = probe_generation(&s2.cfg_path(), &cli, None, None);
        assert_eq!(v, crate::probes::Verdict::Failed);
        assert!(d.contains("no default route configured"), "{d}");
    }

    /// The chat CLI's liar class (exit 0 + ❌ on stdout) fails the
    /// probe; a real reply proves it; the configured default resolves
    /// from the FILE into the argv and the detail label.
    #[test]
    fn generation_probe_folds_reply_and_exit0_error() {
        let s = Scratch::new("probe-gen");
        s.write(&json!({"default_models": {
            "global_provider": "lmstudio", "global_model": "m1"}}));

        let lying = s.fake_chat("echo '❌ Error: API error: Connection refused'");
        let (v, d) = probe_generation(&s.cfg_path(), &lying, None, None);
        assert_eq!(v, crate::probes::Verdict::Failed);
        assert!(d.contains("Connection refused"), "{d}");

        // The happy fake echoes its argv tail so the assert can prove
        // the FILE's default reached the command line.
        let honest = s.fake_chat("echo \"PONG from $2/$4\"");
        let (v, d) = probe_generation(&s.cfg_path(), &honest, None, None);
        assert_eq!(v, crate::probes::Verdict::Proven, "{d}");
        assert!(d.starts_with("lmstudio/m1 — "), "label carries the route: {d}");
        assert!(d.contains("PONG from lmstudio/m1"), "{d}");

        // Explicit pair overrides the file.
        let (v, d) = probe_generation(&s.cfg_path(), &honest, Some("ollama"), Some("x"));
        assert_eq!(v, crate::probes::Verdict::Proven);
        assert!(d.starts_with("ollama/x — "), "{d}");

        // No chat binary: honest teaching refusal.
        let bare = CoreCli::bare(s.dir.join("unused"));
        let (v, d) = probe_generation(&s.cfg_path(), &bare, None, None);
        assert_eq!(v, crate::probes::Verdict::Failed);
        assert!(d.contains("abstractcore-chat not found"), "{d}");
    }

    /// The endpoint-default lane (M3 review P1-1): the chat CLI's
    /// argparse knows no `endpoint:` providers, so a keyless profile
    /// expands to `--provider <family> --base-url <url>`; keyed,
    /// disabled and missing profiles resolve without running the
    /// binary — a Failed verdict on a working route is a lie.
    #[test]
    fn generation_probe_expands_keyless_endpoint_profiles() {
        let s = Scratch::new("probe-endpoint");
        let profile = |api_key: &str, env_var: &str, enabled: bool| {
            json!({
                "default_models": {"global_provider": "endpoint:lab", "global_model": "m9"},
                "provider_profiles": {"profiles": {"lab": {
                    "id": "lab", "provider_family": "lmstudio",
                    "base_url": "http://localhost:9999/v1",
                    "api_key": api_key, "api_key_env_var": env_var,
                    "enabled": enabled}}}
            })
        };

        // Keyless + enabled: expands — the fake echoes argv so the
        // assert proves family + base_url REACHED THE COMMAND LINE
        // (the "via …" note alone is worker-constructed).
        s.write(&profile("", "", true));
        let honest = s.fake_chat("echo \"GEN $2 $4 $9 ${10}\"");
        let (v, d) = probe_generation(&s.cfg_path(), &honest, None, None);
        assert_eq!(v, crate::probes::Verdict::Proven, "{d}");
        assert!(
            d.starts_with("endpoint:lab/m9 — "),
            "the ROUTE stays the label: {d}"
        );
        assert!(
            d.contains("GEN lmstudio m9 --base-url"),
            "argv carried the expansion: {d}"
        );
        assert!(
            d.contains("via lmstudio @ http://localhost:9999/v1"),
            "expansion disclosed: {d}"
        );

        // Keyed profile: refused as NOT PROVEN (argv must never carry
        // secrets; a guessed key lane would mint 401 lies).
        s.write(&profile("sk-secret", "", true));
        let never = s.fake_chat("echo SHOULD-NEVER-RUN");
        let (v, d) = probe_generation(&s.cfg_path(), &never, None, None);
        assert_eq!(v, crate::probes::Verdict::NotProven);
        assert!(d.contains("cannot receive"), "{d}");
        assert!(!d.contains("sk-secret"), "the key never echoes: {d}");

        // $VAR-referenced keys refuse the same way.
        s.write(&profile("", "LAB_KEY", true));
        let (v, _) = probe_generation(&s.cfg_path(), &never, None, None);
        assert_eq!(v, crate::probes::Verdict::NotProven);

        // Disabled profile: NotProven with the enable teaching.
        s.write(&profile("", "", false));
        let (v, d) = probe_generation(&s.cfg_path(), &never, None, None);
        assert_eq!(v, crate::probes::Verdict::NotProven);
        assert!(d.contains("disabled"), "{d}");

        // A default naming a MISSING profile is a broken route: Failed.
        s.write(&json!({"default_models": {
            "global_provider": "endpoint:ghost", "global_model": "m"}}));
        let (v, d) = probe_generation(&s.cfg_path(), &never, None, None);
        assert_eq!(v, crate::probes::Verdict::Failed);
        assert!(d.contains("no such profile"), "{d}");
    }

    /// Reachability is judged over ALL resolved addresses (M3 review
    /// P1-2): `localhost` resolves `::1` first while local servers
    /// often bind IPv4 only — first-address-only probing reported
    /// "looks DOWN" for an UP server.
    #[test]
    fn reachability_connects_if_any_address_accepts() {
        use std::net::{SocketAddr, TcpListener};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let live = listener.local_addr().unwrap();
        // A port with no listener on either family: bind + drop.
        let dead_port = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let dead_v6: SocketAddr = format!("[::1]:{dead_port}").parse().unwrap();
        let dead_v4: SocketAddr = format!("127.0.0.1:{dead_port}").parse().unwrap();

        // The P1-2 shape: v6 refuses FIRST, v4 accepts → Connected.
        let r = probe_addr_list(&[dead_v6, live], Duration::from_millis(800));
        assert_eq!(r, crate::probes::Reach::Connected);

        // Every address refuses → Refused, preferring the IPv4 error.
        let r = probe_addr_list(&[dead_v6, dead_v4], Duration::from_millis(800));
        assert!(matches!(r, crate::probes::Reach::Refused(_)), "{r:?}");

        // No addresses at all → honest Unresolvable.
        let r = probe_addr_list(&[], Duration::from_millis(100));
        assert!(matches!(r, crate::probes::Reach::Unresolvable(_)), "{r:?}");
    }
}
