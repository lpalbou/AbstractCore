### Fixed

- **Tool parameters carry their meaning** (`abstractcore/tools/core.py`, `abstractcore/tools/parser.py`). The tool
  schema built from a decorated function now takes each parameter's `description` from the docstring's `Args:`
  section (Google style, or `:param name:` lines); it used to emit the type and default only, so native tool
  payloads and prompted tool lists showed the model bare parameter names. Native payloads carry the full text as
  JSON Schema `description`; the prompted tool list shows each parameter's first sentence. An explicit
  `description` set on the schema afterwards still wins. Every built-in tool parameter is now described, and a
  test keeps it that way.
