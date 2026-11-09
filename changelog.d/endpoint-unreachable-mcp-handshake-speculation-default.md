### Fixed

- **MCP handshake accepted by every schema-validated server** (`abstractcore/mcp/client.py`). `initialize` now
  declares `capabilities: {}` as the specification and the reference client do; the previous envelope sent five
  capabilities set to `null`, which strict servers such as Context7 refused ("Invalid input: expected record,
  received null"), so their tools never reached the model. A `404` to a request carrying `MCP-Session-Id` now
  starts a new session with a fresh `initialize` and replays the request once, as the Streamable HTTP transport
  requires after a server restart. `McpHttpError` carries `status_code`.
- **An unreachable endpoint is reported as such, never as a spent timeout** (`abstractcore/providers/base.py`,
  `abstractcore/core/retry.py`, `abstractcore/exceptions`). When no TCP connection to the provider's endpoint can
  be made (machine off or asleep, connection refused, DNS failure), providers raise the new
  `EndpointUnreachableError` with `base_url`, the transport exception class (`cause`), `errno`, the measured
  `elapsed_s` and `retryable` (False when the connect attempt itself timed out). Hosts can import it by name to
  park the step and probe the endpoint instead of failing the run. Such failures no longer count toward the
  circuit breaker, and a connect that timed out is not retried (a second attempt would block for the same OS
  budget). Timeout detection is type-first: the message is consulted only for exceptions that carry no type
  information, so an `OSError` whose text says "Operation timed out" no longer prints
  "timed out after 7200.0s (configured timeout)". Every normalisation logs the original exception class, and
  the "Generation failed" line reports the elapsed time from the first attempt (it used to print ~0 ms for
  every failure).
- **A stored speculation default never fails a call** (`abstractcore/providers/speculation.py`). When
  `require_acceleration=True` comes from the provider-lifetime default (constructor `speculation=`, a stored
  profile or host setting) and the provider has no native MTP adapter, the call runs without speculation with
  one warning, and `metadata["speculation"]` carries `applied: false` and `degraded_from_default: true`. An
  explicit per-call `require_acceleration=True` still raises `SpeculationUnavailableError`.
- **Residency snapshot during an embeddings import** (`abstractcore/providers/process_residency.py`). The
  in-process residency view treated `abstractcore.embeddings.manager` as loaded as soon as it appeared in
  `sys.modules`, which is true while another thread is still importing it, and failed with "partially
  initialized module has no attribute". The backend counts as loaded only once its functions exist.
