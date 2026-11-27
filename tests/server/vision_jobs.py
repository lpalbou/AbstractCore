"""Waiting for a vision job without a clock: join the thread the server runs it on."""

from __future__ import annotations

import threading
from typing import Any, Dict

# Only stops a broken test from hanging the suite; no test relies on it for ordering.
HANG_GUARD_S = 120.0


def join_job(job_id: str) -> None:
    """Return once the server's runner of `job_id` (thread `vision-job-<id>`) has ended.

    The runner records the job's final state before it ends, so the state is final afterwards.
    A runner that already ended is no longer listed; its state is final as well.
    """
    for thread in threading.enumerate():
        if thread.name == f"vision-job-{job_id}":
            thread.join(HANG_GUARD_S)
            assert not thread.is_alive(), f"vision job {job_id} still running after {HANG_GUARD_S}s"


def finished_job(client: Any, job_id: str) -> Dict[str, Any]:
    """The job's body once its runner has ended."""
    join_job(job_id)
    resp = client.get(f"/v1/vision/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()
