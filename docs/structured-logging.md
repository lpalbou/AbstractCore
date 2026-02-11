# Structured Logging

AbstractCore uses Python logging throughout the library. You can control console verbosity and optional file logging via the centralized config CLI.

Default behavior (no overrides): **console shows only ERROR and above**.

## Configure with the CLI

```bash
# Show current config (including logging)
abstractcore --status

# Console verbosity
abstractcore --set-console-log-level DEBUG
abstractcore --set-console-log-level INFO
abstractcore --set-console-log-level WARNING
abstractcore --set-console-log-level ERROR
abstractcore --set-console-log-level NONE

# File logging (disabled by default)
abstractcore --enable-file-logging
abstractcore --disable-file-logging
abstractcore --set-log-base-dir ~/.abstractcore/logs

# Convenience
abstractcore --enable-debug-logging
abstractcore --disable-console-logging
```

Logging defaults live in `~/.abstractcore/config/abstractcore.json`. See [Centralized Config](centralized-config.md) for the schema.

## Verbatim capture (prompts/responses)

Some components can capture full prompts and responses in logs/traces. This is controlled by `verbatim_enabled` in the centralized config file (`~/.abstractcore/config/abstractcore.json`). Disable it if you may handle sensitive data.

## In-code usage

```python
from abstractcore.utils.structured_logging import get_logger

logger = get_logger(__name__)
logger.info("startup", component="my_app", version="1.0.0")
```
