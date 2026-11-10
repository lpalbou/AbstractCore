### Fixed

- **A release inside Context7's refresh window no longer ends with a failed job** (`.github/workflows/release.yml`,
  `context7-refresh.yml`). Context7 allows one re-index per project every 10 days; the 2.35.0 release asked two days
  after the previous refresh and the last job of the run failed with `HTTP 400 too-early` although everything was
  published. That answer is now a notice on the run (Context7 re-indexes on its own schedule, and the on-demand
  workflow can be run after the window); any other non-200 answer still fails the job visibly.
