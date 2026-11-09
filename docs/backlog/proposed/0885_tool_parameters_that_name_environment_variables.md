# 0885 — Tool parameters that ask the model for environment-variable names

> Type: tools + settings
> Created: 2026-11-08
> Priority: normal
> Labels: tools, comms, settings, adr
> Links: `abstractcore/tools/comms_tools.py`, `abstractcore/tools/telegram_tools.py`, [0884](../planned/0884_release_2_35_0_endpoint_unreachable_mcp_handshake_tool_descriptions.md)

## Summary

Nine built-in tool parameters are the *names* of environment variables the model is asked to supply:
`account_sid_env_var` and `auth_token_env_var` on `send_whatsapp_message`, `list_whatsapp_messages` and
`read_whatsapp_message`; `bot_token_env_var` on `send_telegram_message` and `send_telegram_artifact`;
`artifact_base_dir_env_var` on `send_telegram_artifact`. The 2026-11-08 rendering audit (tool parameter
descriptions now reach the model) made them visible in every tool list. They expose credential plumbing to
the model and contradict the operator rule that configuration is a setting or a launch parameter, never an
environment variable (see the launch-flags rulings in the root repository and ADR practice here).

## Current code reality

- The nine parameters are in the model-facing schema (not in `hide_args`) with defaults such as
  `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `ABSTRACT_TELEGRAM_BOT_TOKEN`, `ABSTRACTGATEWAY_DATA_DIR`.
- `_twilio_creds(...)` and `_resolve_required_env(...)` read the named variables at call time.
- The email tools already resolve their account through a host-installed resolver (`_EMAIL_RESOLVER`), which is
  the pattern to follow.

## Proposed scope

1. Hide the nine parameters from the schema (`hide_args`), keep them accepted for older Python callers.
2. Resolve Twilio and Telegram credentials and the artifact store through a host-installed resolver, as the
   email tools do; the gateway supplies them from its settings.
3. Environment variables remain only as the legacy fallback for bare-core use, reported as such.

## Non-goals

No change to what the tools do once credentials are resolved.

## Validation

- `tests/tools/test_tool_parameter_descriptions_unit.py::test_every_builtin_tool_parameter_is_described` stays
  green; a new test asserts no built-in parameter name ends with `_env_var`.
