//! `CliTransport` against a FAKE `abstractcore` executable on PATH that
//! prints the contract fixtures — argv forwarding, exit-code mapping
//! (2 = refused), child jobs with NDJSON progress, and cancel.
//!
//! Its own test binary on purpose: it rewrites PATH for the process,
//! which must not race the other suites. Unix only (a `sh` script).
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use abstractcore_console::transport::{
    CliTransport, ConsoleTransport, InstallLocation, ServerAction, TransportErrorKind,
};
use serde_json::Value;

const FAKE: &str = r#"#!/bin/sh
D="$(dirname "$0")"
echo "$*" >> "$D/calls.log"
case "$1 $2" in
  "host profile") cat "$D/host_profile.json" ;;
  "engines status") cat "$D/engines_status.json" ;;
  "engines install")
    case "$*" in
      *--dry-run*) printf '{"schema":"host_job_v1","job_id":"ei_1","kind":"engine_install","status":"completed","engine":"%s","dry_run":true,"command":["brew","install","%s"]}\n' "$3" "$3" ;;
      *) echo "==> Downloading $3" >&2; exec sleep 30 ;;
    esac ;;
  "models catalog") cat "$D/model_catalog.json" ;;
  "models search")
    if [ "$3" = "notjson" ]; then echo "hello, human"; else cat "$D/model_catalog.json"; fi ;;
  "models list") cat "$D/models_installed.json" ;;
  "models delete")
    if [ "$4" = "loaded-model" ]; then
      case "$*" in
        *--force*) ;;
        *) echo '{"error":"model is loaded","delete_blockers":["loaded"]}'; exit 2 ;;
      esac
    fi
    printf '{"schema":"host_job_v1","job_id":"del_1","kind":"delete","status":"completed","provider":"%s","artifact":"%s"}\n' "$3" "$4" ;;
  "models download")
    if [ "$4" = "refused" ]; then echo '{"error":"downloads are disabled here"}'; exit 2; fi
    if [ "$4" = "broken" ]; then echo "Traceback: boom" >&2; exit 1; fi
    printf '{"schema":"host_job_v1","job_id":"dl_up","kind":"download","status":"running","percent":50.0,"downloaded_bytes":50,"total_bytes":100,"message":"pulling"}\n'
    sleep 1
    printf '{"schema":"host_job_v1","job_id":"dl_up","kind":"download","status":"completed","percent":100.0,"message":"success","command":["ollama","pull","%s"]}\n' "$4" ;;
  "models jobs") echo '{"jobs":[]}' ;;
  "config defaults")
    if [ -f "$D/route" ]; then R="$(cat "$D/route")"; else R='"provider":"lmstudio","model":"old","base_url":"http://localhost:1234/v1","reasoning":"high","options":{"speculation":{"mode":"native_mtp"}}'; fi
    printf '{"routes":[{"key":"input.text","provider":"mlx","model":"old"},{"key":"output.text",%s}]}\n' "$R" ;;
  "config set-default")
    # Core's merge rule: a flag not passed KEEPS the stored field, "" clears it.
    ROUTE="$3"; shift 3; P=""; M=""; CLEARED=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --provider) P="$2"; shift 2 ;;
        --model) M="$2"; shift 2 ;;
        --base-url|--reasoning|--option) if [ -z "$2" ]; then CLEARED="$CLEARED ${1#--}"; fi; shift 2 ;;
        *) shift ;;
      esac
    done
    if [ "$M" = "bad-model" ]; then echo "❌ Error: Failed to set capability default for $ROUTE"; exit 1; fi
    R="\"provider\":\"$P\",\"model\":\"$M\""
    case "$CLEARED" in *base-url*) ;; *) R="$R,\"base_url\":\"http://localhost:1234/v1\"" ;; esac
    case "$CLEARED" in *reasoning*) ;; *) R="$R,\"reasoning\":\"high\"" ;; esac
    case "$CLEARED" in *option*) ;; *) R="$R,\"options\":{\"speculation\":{\"mode\":\"native_mtp\"}}" ;; esac
    printf '%s' "$R" > "$D/route"
    echo "✅ Set capability default for $ROUTE" ;;
  *) echo "unknown: $*" >&2; exit 1 ;;
esac
"#;

fn setup() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "abstractcore-console-fake-{}-{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for name in [
        "host_profile",
        "engines_status",
        "model_catalog",
        "models_installed",
    ] {
        let src = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("{name}.json"));
        std::fs::copy(&src, dir.join(format!("{name}.json"))).unwrap();
    }
    let bin = dir.join("abstractcore");
    std::fs::write(&bin, FAKE).unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    dir
}

fn calls(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("calls.log")).unwrap_or_default()
}

fn wait_terminal(t: &CliTransport, id: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let j = t.job(id).expect("job reads");
        let st = j["status"].as_str().unwrap_or("").to_string();
        if !matches!(st.as_str(), "queued" | "running") {
            return j;
        }
        assert!(Instant::now() < deadline, "job {id} never finished: {j}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn cli_transport_against_a_fake_abstractcore_on_path() {
    let dir = setup();
    // The fake must be what PATH resolution finds: no explicit override.
    std::env::remove_var("ABSTRACTCORE_CLI");
    std::env::remove_var("ABSTRACTCORE_BIN");
    let path = std::env::var("PATH").unwrap_or_default();
    std::env::set_var("PATH", format!("{}:{path}", dir.display()));

    let t = CliTransport::from_env()
        .expect("the fake abstractcore resolves from PATH")
        .with_refusal_window(Duration::from_millis(300));
    assert_eq!(t.bin(), dir.join("abstractcore"));
    assert!(t.host_label().starts_with("this machine"));

    // Reads: the documents verbatim, argv forwarded exactly.
    assert_eq!(t.host_profile().unwrap()["schema"], "host_profile_v1");
    assert_eq!(
        t.engines_status(true).unwrap()["schema"],
        "engines_status_v1"
    );
    assert_eq!(
        t.models_catalog("", None, false).unwrap()["schema"],
        "model_catalog_v1"
    );
    t.models_catalog("qwen", Some("ollama"), true).unwrap();
    assert_eq!(
        t.models_installed(Some("mlx")).unwrap()["schema"],
        "models_installed_v1"
    );
    let log = calls(&dir);
    for line in [
        "host profile --json",
        "engines status --probe --json",
        "models catalog --json",
        "models search qwen --engine ollama --fits --json",
        "models list --provider mlx --json",
    ] {
        assert!(log.lines().any(|l| l == line), "{line:?} in:\n{log}");
    }

    // Exit 0 without JSON = a protocol error, not a crash.
    let e = t.models_catalog("notjson", None, false).unwrap_err();
    assert_eq!(e.kind, TransportErrorKind::Protocol, "{e}");

    // Exit 2 = REFUSED, the body's blockers ride along.
    let e = t.delete_model("mlx", "loaded-model", false).unwrap_err();
    assert!(e.is_refused(), "{e}");
    assert_eq!(e.code, Some(2));
    assert_eq!(e.message, "model is loaded");
    assert_eq!(e.reasons(), vec!["loaded", "model is loaded"]);
    let ok = t.delete_model("mlx", "loaded-model", true).unwrap();
    assert_eq!(ok["status"], "completed");
    assert!(calls(&dir).contains("models delete mlx loaded-model --yes --force --json"));

    // A flag-shaped id never reaches the CLI.
    assert!(t.start_download("ollama", "--rm", None).unwrap_err().is_refused());

    // Download: a child job. Immediate refusal → Refused from start.
    let e = t.start_download("ollama", "refused", None).unwrap_err();
    assert!(e.is_refused(), "{e}");
    assert_eq!(e.message, "downloads are disabled here");
    let e = t.start_download("ollama", "broken", Some(10)).unwrap_err();
    assert_eq!(e.kind, TransportErrorKind::Failed, "{e}");
    assert!(e.message.contains("boom"), "{e}");

    // A real one: running with the NDJSON progress, then completed.
    let j = t.start_download("ollama", "qwen3:8b", Some(4_600_000_000)).unwrap();
    assert_eq!(j["schema"], "host_job_v1");
    assert_eq!(j["status"], "running", "{j}");
    assert_eq!(j["kind"], "download");
    assert_eq!(j["percent"], 50.0, "NDJSON progress adopted: {j}");
    assert_eq!(
        j["cli_equivalent"],
        "abstractcore models download ollama qwen3:8b"
    );
    let id = j["job_id"].as_str().unwrap().to_string();
    assert!(id.starts_with("cli-"), "{id}");
    let done = wait_terminal(&t, &id);
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(done["percent"], 100.0);
    assert_eq!(done["job_id"], id, "the id stays ours");
    assert_eq!(done["upstream_job_id"], "dl_up");
    assert_eq!(done["command"][0], "ollama", "the CLI's own command wins");

    // Install: dry run is synchronous; a real one is a child we cancel.
    let dry = t.engine_install("ollama", true).unwrap();
    assert_eq!(dry["dry_run"], true);
    assert_eq!(dry["command"][1], "install");
    let run = t.engine_install("ollama", false).unwrap();
    assert_eq!(run["status"], "running", "{run}");
    let id = run["job_id"].as_str().unwrap().to_string();
    // The installer's stderr is the progress line.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let j = t.job(&id).unwrap();
        if j["message"] == "==> Downloading ollama" {
            break;
        }
        assert!(Instant::now() < deadline, "stderr never surfaced: {j}");
        std::thread::sleep(Duration::from_millis(20));
    }
    let c = t.cancel_job(&id).unwrap();
    assert_eq!(c["status"], "cancelled", "{c}");
    assert_eq!(t.job(&id).unwrap()["status"], "cancelled");

    // Jobs this transport never started: an honest not-found.
    let e = t.job("dl_elsewhere").unwrap_err();
    assert_eq!(e.kind, TransportErrorKind::NotFound, "{e}");
    let e = t.cancel_job("dl_elsewhere").unwrap_err();
    assert_eq!(e.kind, TransportErrorKind::NotFound, "{e}");

    // Optional verbs the CLI HAS: hub search, the text default, the feed.
    let caps = t.capabilities();
    assert!(caps.hub_search && caps.text_default && caps.downloads_feed);
    assert!(!caps.engine_server && !caps.engine_continue && !caps.install_location);
    t.models_catalog_hub("qwen", Some("ollama"), true).unwrap();
    assert!(
        calls(&dir).contains("models search qwen --hub --engine ollama --fits --json"),
        "{}",
        calls(&dir)
    );
    assert!(t.models_catalog_hub("  ", None, false).unwrap_err().is_refused());
    let feed = t.download_jobs().unwrap();
    let listed: Vec<&str> = feed["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j["kind"].as_str().unwrap())
        .collect();
    assert!(!listed.is_empty() && listed.iter().all(|k| *k == "download"), "{feed}");
    let before = t.capability_defaults().unwrap();
    assert_eq!(before["routes"][1]["model"], "old");
    let after = t.set_text_default("ollama", "qwen3:8b").unwrap();
    assert!(
        calls(&dir).contains("config set-default output.text --provider ollama --model qwen3:8b"),
        "{}",
        calls(&dir)
    );
    assert_eq!(after["routes"][1]["provider"], "ollama", "a FRESH read answers: {after}");
    assert_eq!(after["routes"][1]["model"], "qwen3:8b");
    // The route is EXACTLY the new model: the previous model's endpoint,
    // reasoning level and options are cleared, never inherited (an
    // Ollama model behind LM Studio's :1234 is a broken route).
    for field in ["base_url", "reasoning", "options"] {
        assert!(after["routes"][1].get(field).is_none(), "{field} survived: {after}");
    }
    let e = t.set_text_default("ollama", "bad-model").unwrap_err();
    assert_eq!(e.kind, TransportErrorKind::Failed, "{e}");
    assert!(e.message.contains("Failed to set"), "{e}");
    assert!(t.set_text_default("ollama", "--force").unwrap_err().is_refused());
    // …and the ones it has NOT: Unsupported, never a fake.
    for e in [
        t.engine_server("ollama", ServerAction::Start).unwrap_err(),
        t.engine_job_continue("cli-1", None).unwrap_err(),
        t.engine_install_at("ollama", false, InstallLocation::System).unwrap_err(),
    ] {
        assert_eq!(e.kind, TransportErrorKind::Unsupported, "{e}");
    }

    // A missing binary is "unavailable", never a panic.
    let gone = CliTransport::new(dir.join("nope"));
    let e = gone.host_profile().unwrap_err();
    assert_eq!(e.kind, TransportErrorKind::Unavailable, "{e}");
    assert_eq!(
        gone.start_download("ollama", "x", None).unwrap_err().kind,
        TransportErrorKind::Unavailable
    );

    let _ = std::fs::remove_dir_all(&dir);
}
