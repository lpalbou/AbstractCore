//! The M3 test vocabulary: probe specs (what to test) and pure verdict
//! folds (what the evidence means), mirroring `writes.rs`'s split —
//! builders + folds are pure and unit-tested; the worker owns the
//! subprocess/socket mechanics.
//!
//! Honesty rules this module encodes (charter lesson #3, M1 finding 5):
//! - `config test-provider` answers `ok:true, count:0, errors:[]`
//!   against a DEAD server (live-probed 2026-07-25) — zero-count
//!   success is never Proven, only NotProven, and a TCP reachability
//!   check on a KNOWN endpoint upgrades the message to a cause.
//! - `abstractcore-chat` exits 0 on failures and prints `❌ Error:` to
//!   stdout (the write lane's liar class, third appearance) — the fold
//!   scans the body, never trusts the exit code.
//! - A verdict is Proven only on positive evidence (models listed, a
//!   reply produced). Absence of failure is not success.

use serde_json::Value;

/// What a probe is allowed to reach over TCP: endpoints the CONFIG
/// names (profile base_url) or a provider's documented local default —
/// never cloud endpoints, never https (risk-map fact 5: the console
/// does its own networking only where a wrong guess cannot leak).
pub const LOCAL_DEFAULT_ENDPOINTS: &[(&str, &str)] = &[
    ("ollama", "http://localhost:11434"),
    ("lmstudio", "http://localhost:1234"),
];

/// Providers whose model listing REQUIRES an API key — their empty
/// listing is also the silent no-key answer (openai_provider.py
/// returns [] keyless without raising), so the ambiguous verdict names
/// that likely cause. huggingface/mlx list local caches (keyless) and
/// vllm/openai-compatible are local-server class — not in this set.
pub const KEYED_CLOUD_PROVIDERS: &[&str] = &["openai", "anthropic", "openrouter", "portkey"];

#[derive(Clone, Debug, PartialEq)]
pub enum ProbeKind {
    /// Live model discovery for a provider id / profile id /
    /// `endpoint:<id>` — `config test-provider <target> --json`.
    ListModels {
        target: String,
        /// Pre-resolved http endpoint for the count==0 disambiguation
        /// (None = no known endpoint; the fold says so).
        reach: Option<HostPort>,
    },
    /// The selected route's model must be IN the provider's live list —
    /// capability-agnostic (voice/image routes can't be chat-tested,
    /// but "the configured model exists on the provider" always can).
    RouteCheck {
        capability: String,
        provider: String,
        model: String,
        reach: Option<HostPort>,
    },
    /// One cheap generation. `None/None` = the CONFIGURED default
    /// route; the worker refuses when the file has none — probed:
    /// `abstractcore-chat` on an empty config silently invents a
    /// huggingface default, which would make the test lie about the
    /// operator's own route.
    Generate {
        provider: Option<String>,
        model: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbeSpec {
    pub label: String,
    pub kind: ProbeKind,
}

/// The three honest outcomes. NotProven is the load-bearing one: the
/// CLI's ambiguous successes land here, never in Proven.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    Proven,
    NotProven,
    Failed,
}

impl Verdict {
    pub fn glyph(&self) -> &'static str {
        match self {
            Verdict::Proven => "✓",
            Verdict::NotProven => "?",
            Verdict::Failed => "✗",
        }
    }
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Proven => "proven",
            Verdict::NotProven => "NOT PROVEN",
            Verdict::Failed => "FAILED",
        }
    }
}

/// One test's current evidence — the Review screen renders the latest
/// result per label (re-tests REPLACE: evidence is about NOW).
#[derive(Clone, Debug)]
pub struct TestResult {
    pub when: String,
    pub label: String,
    pub verdict: Verdict,
    pub detail: String,
}

/// An http host:port a probe may TCP-check.
#[derive(Clone, Debug, PartialEq)]
pub struct HostPort {
    pub host: String,
    pub port: u16,
}

/// Parse `http://host[:port][/path]` — https and anything else is
/// None (the console never speaks TLS; those endpoints get CLI-only
/// verdicts). Userinfo URLs are refused too: splitting them naively
/// would land `user:pass@host` — a possible secret — in `HostPort`,
/// which renders through probe details and the journal (M3 review
/// P3-3).
pub fn parse_http_host_port(url: &str) -> Option<HostPort> {
    let rest = url.trim().strip_prefix("http://")?;
    let authority = rest.split('/').next()?;
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().ok()?),
        None => (authority, 80),
    };
    if host.is_empty() {
        return None;
    }
    Some(HostPort {
        host: host.to_string(),
        port,
    })
}

/// The endpoint a provider test may reach: a configured profile
/// base_url wins; plain providers get their documented local default
/// (ollama/lmstudio) or nothing.
pub fn endpoint_for(target: &str, profile_base_url: Option<&str>) -> Option<HostPort> {
    if let Some(url) = profile_base_url {
        return parse_http_host_port(url);
    }
    LOCAL_DEFAULT_ENDPOINTS
        .iter()
        .find(|(p, _)| *p == target)
        .and_then(|(_, url)| parse_http_host_port(url))
}

/// TCP evidence, produced by the worker (the only socket owner) and
/// consumed by the folds as text.
#[derive(Clone, Debug, PartialEq)]
pub enum Reach {
    Connected,
    Refused(String),
    Unresolvable(String),
}

impl Reach {
    pub fn describe(&self, hp: &HostPort) -> String {
        match self {
            Reach::Connected => format!(
                "TCP {}:{} accepts connections — the server is UP but reports zero models",
                hp.host, hp.port
            ),
            Reach::Refused(e) => {
                format!("TCP {}:{} → {e} — the server looks DOWN", hp.host, hp.port)
            }
            Reach::Unresolvable(e) => format!("cannot resolve {}:{} — {e}", hp.host, hp.port),
        }
    }
}

/// Fold a `test-provider --json` payload. `reach` is the worker's TCP
/// evidence, gathered only when count==0 (the ambiguous branch).
pub fn fold_list_models(
    v: &Value,
    reach: Option<(&HostPort, &Reach)>,
) -> (Verdict, String) {
    let errors: Vec<String> = v
        .get("errors")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|e| match e.as_str() {
                    Some(s) => s.to_string(),
                    None => e.to_string(),
                })
                .collect()
        })
        .unwrap_or_default();
    if let Some(first) = errors.first() {
        return (Verdict::Failed, first.clone());
    }
    // Proven derives from the MODELS LIST, never the count field alone
    // — "Proven with no nameable model" would be a fold-level
    // fabrication if a payload ever drifts (M3 review P3-4; today's
    // CLI computes count = len(models), so the two agree live).
    // "available", not "served": huggingface/mlx list local caches —
    // no server exists to serve (P3-8).
    let models: Vec<&str> = v
        .get("models")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if !models.is_empty() {
        return (
            Verdict::Proven,
            format!("{} models available · e.g. {}", models.len(), models[0]),
        );
    }
    // The lying branch: ok:true, zero models, zero errors. The TCP
    // evidence LEADS when present — notices/rows truncate from the
    // right, and the cause is the part the operator must see.
    let base = "the CLI reports success with ZERO models (also its answer for a dead server)";
    match reach {
        Some((hp, r)) => (Verdict::NotProven, format!("{}; {base}", r.describe(hp))),
        None => (
            Verdict::NotProven,
            format!("{base}; no known endpoint to reach-check"),
        ),
    }
}

/// Fold a route check: the same payload, judged for membership.
pub fn fold_route_check(
    v: &Value,
    model: &str,
    reach: Option<(&HostPort, &Reach)>,
) -> (Verdict, String) {
    let (list_verdict, list_detail) = fold_list_models(v, reach);
    if list_verdict != Verdict::Proven {
        return (list_verdict, list_detail);
    }
    let models: Vec<&str> = v
        .get("models")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if models.contains(&model) {
        (
            Verdict::Proven,
            format!("model {model} is among the {} served", models.len()),
        )
    } else {
        (
            Verdict::Failed,
            format!(
                "model {model} is NOT among the {} the provider serves — edit the route (e) and pick from the live list",
                models.len()
            ),
        )
    }
}

/// Fold a generation run's stdout (exit code already vetted by the
/// runner; `❌ Error:` on an exit-0 stdout is the failure signal —
/// live-probed against a dead ollama).
///
/// DOCUMENTED COST (M3 review P3-7): the chat CLI has no machine
/// verdict channel, so this scrapes mixed stdout — a reply that
/// itself contains `❌` or opens a line with `Error:` will misread as
/// a failed generation. Accepted: the probe prompt asks for "PONG",
/// making such replies adversarial rather than expected.
pub fn fold_generation(stdout: &str, elapsed_s: u64) -> (Verdict, String) {
    if let Some(l) = stdout
        .lines()
        .find(|l| l.contains("❌") || l.trim_start().starts_with("Error:"))
    {
        return (Verdict::Failed, l.trim().to_string());
    }
    // Log-shaped lines ("HH:MM:SS [LEVEL] …") are noise, not reply.
    let reply: String = stdout
        .lines()
        .filter(|l| !is_log_line(l))
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if reply.is_empty() {
        return (
            Verdict::NotProven,
            "exit 0 but no reply text — nothing to show for it".into(),
        );
    }
    let head: String = reply.chars().take(60).collect();
    let ellipsis = if reply.chars().count() > 60 { "…" } else { "" };
    (
        Verdict::Proven,
        format!("replied in {elapsed_s}s: “{head}{ellipsis}”"),
    )
}

fn is_log_line(l: &str) -> bool {
    let t = l.trim_start();
    // "22:31:13 [ERROR] …" — the provider log format. `get(8..)`
    // (never `[8..]`): model output is arbitrary text, and a multibyte
    // char spanning byte 8 would panic a slice (M3 review P2-1).
    t.len() > 10
        && t.as_bytes()[2] == b':'
        && t.as_bytes()[5] == b':'
        && t.get(8..).is_some_and(|r| r.trim_start().starts_with('['))
}

// ---------------------------------------------------------------------
// Spec builders — every UI test verb goes through one of these.
// ---------------------------------------------------------------------

pub fn list_models(target: &str, profile_base_url: Option<&str>) -> ProbeSpec {
    ProbeSpec {
        label: format!("test {target}"),
        kind: ProbeKind::ListModels {
            target: target.to_string(),
            reach: endpoint_for(target, profile_base_url),
        },
    }
}

pub fn route_check(
    capability: &str,
    provider: &str,
    model: &str,
    profile_base_url: Option<&str>,
) -> ProbeSpec {
    // The label is the ROUTE alone: evidence replaces per target, and
    // a re-test after editing the route must supersede the old pair's
    // result, not sit beside it. The pair lives in the detail.
    ProbeSpec {
        label: format!("test route {capability}"),
        kind: ProbeKind::RouteCheck {
            capability: capability.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            reach: endpoint_for(provider, profile_base_url),
        },
    }
}

pub fn generate_default() -> ProbeSpec {
    ProbeSpec {
        label: "generation test (default route)".into(),
        kind: ProbeKind::Generate {
            provider: None,
            model: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn url_parse_is_http_only_and_exact() {
        assert_eq!(
            parse_http_host_port("http://localhost:11434"),
            Some(HostPort {
                host: "localhost".into(),
                port: 11434
            })
        );
        assert_eq!(
            parse_http_host_port("http://h/v1/models"),
            Some(HostPort {
                host: "h".into(),
                port: 80
            })
        );
        // TLS and garbage are honestly out of scope.
        assert_eq!(parse_http_host_port("https://api.openai.com/v1"), None);
        assert_eq!(parse_http_host_port("http://"), None);
        assert_eq!(parse_http_host_port("http://h:notaport"), None);
        assert_eq!(parse_http_host_port(""), None);
        // Userinfo would put `user:pass` into HostPort — a possible
        // secret on a rendered surface. Refused (M3 review P3-3).
        assert_eq!(parse_http_host_port("http://user:pass@h:1234"), None);
        // IPv6 literals keep their brackets and fail resolution
        // honestly (Unresolvable) rather than mis-splitting.
        assert_eq!(
            parse_http_host_port("http://[::1]:1234").map(|hp| hp.host),
            Some("[::1]".into())
        );
    }

    #[test]
    fn endpoints_config_first_then_local_defaults() {
        // Profile base_url wins.
        let hp = endpoint_for("ollama", Some("http://10.0.0.5:11434")).unwrap();
        assert_eq!(hp.host, "10.0.0.5");
        // Documented local default for the two local providers.
        assert_eq!(endpoint_for("lmstudio", None).unwrap().port, 1234);
        // Cloud/unknown providers get NOTHING to reach.
        assert_eq!(endpoint_for("openai", None), None);
        assert_eq!(endpoint_for("vllm", None), None);
    }

    /// The M1-finding-5 lane: ok:true + count:0 + errors:[] must never
    /// read as Proven, and TCP evidence names the cause.
    #[test]
    fn zero_count_success_is_not_proven() {
        let dead = json!({"ok": true, "count": 0, "errors": [], "models": []});
        let (v, d) = fold_list_models(&dead, None);
        assert_eq!(v, Verdict::NotProven);
        assert!(d.contains("ZERO models"), "{d}");

        let hp = HostPort {
            host: "localhost".into(),
            port: 11434,
        };
        let (v, d) =
            fold_list_models(&dead, Some((&hp, &Reach::Refused("connection refused".into()))));
        assert_eq!(v, Verdict::NotProven);
        assert!(d.contains("looks DOWN"), "{d}");

        let (v, d) = fold_list_models(&dead, Some((&hp, &Reach::Connected)));
        assert_eq!(v, Verdict::NotProven);
        assert!(d.contains("UP but reports zero"), "{d}");
    }

    #[test]
    fn listed_models_prove_and_errors_fail() {
        let live = json!({"ok": true, "count": 2, "errors": [], "models": ["a", "b"]});
        let (v, d) = fold_list_models(&live, None);
        assert_eq!(v, Verdict::Proven);
        assert!(d.contains("2 models available"), "{d}");

        let broken = json!({"ok": false, "count": 0, "errors": ["401 unauthorized"], "models": []});
        let (v, d) = fold_list_models(&broken, None);
        assert_eq!(v, Verdict::Failed);
        assert_eq!(d, "401 unauthorized");

        // Proven derives from the MODELS LIST — a count with no
        // nameable model is a payload drift, folded to the ambiguous
        // branch instead of fabricating "N models · e.g. ?" (P3-4).
        let drifted = json!({"ok": true, "count": 9, "errors": [], "models": []});
        let (v, _) = fold_list_models(&drifted, None);
        assert_eq!(v, Verdict::NotProven);
    }

    /// Route-test evidence replaces PER ROUTE: the label must not
    /// embed the pair, or editing the route and re-testing (the fix
    /// loop the Failed detail itself teaches) accumulates stale rows
    /// beside fresh ones (M3 review P2-3).
    #[test]
    fn route_check_label_is_pair_free() {
        let a = route_check("input.text", "lmstudio", "m1", None);
        let b = route_check("input.text", "ollama", "m2", None);
        assert_eq!(a.label, b.label);
        assert_eq!(a.label, "test route input.text");
    }

    #[test]
    fn route_check_judges_membership() {
        let live = json!({"ok": true, "count": 2, "errors": [], "models": ["m1", "m2"]});
        let (v, _) = fold_route_check(&live, "m1", None);
        assert_eq!(v, Verdict::Proven);
        let (v, d) = fold_route_check(&live, "ghost", None);
        assert_eq!(v, Verdict::Failed);
        assert!(d.contains("ghost"), "{d}");
        // Ambiguous listing stays ambiguous — membership is unjudgeable.
        let dead = json!({"ok": true, "count": 0, "errors": [], "models": []});
        let (v, _) = fold_route_check(&dead, "m1", None);
        assert_eq!(v, Verdict::NotProven);
    }

    /// The chat CLI's liar class: exit 0 + ❌ on stdout (live-probed).
    #[test]
    fn generation_folds_reply_error_and_silence() {
        let (v, d) = fold_generation("❌ Error: API error: [Errno 61] Connection refused", 1);
        assert_eq!(v, Verdict::Failed);
        assert!(d.contains("Connection refused"), "{d}");

        let (v, d) = fold_generation("22:31:13 [ERROR] OllamaProvider: boom\nOK\n", 3);
        assert_eq!(v, Verdict::Proven);
        assert!(d.contains("OK") && !d.contains("boom"), "log lines are noise: {d}");

        let (v, _) = fold_generation("   \n", 2);
        assert_eq!(v, Verdict::NotProven);

        // Model output is arbitrary text: a line shaped like a log
        // prefix with a multibyte char at byte 8 must not panic the
        // fold (M3 review P2-1 — `t[8..]` was a slice).
        let (v, d) = fold_generation("ab:cd:xé[test] whatever", 1);
        assert_eq!(v, Verdict::Proven, "{d}");

        // Long replies truncate for display, never panic on unicode.
        let long = "é".repeat(200);
        let (v, d) = fold_generation(&long, 9);
        assert_eq!(v, Verdict::Proven);
        assert!(d.contains('…'), "{d}");
    }
}
