# Proposed: Documentation site publishing strategy

## Metadata
- Created: 2026-05-06
- Status: Proposed
- Completed: N/A

## Context

AbstractCore currently has two different documentation/site surfaces:

- `mkdocs.yml` defines a conventional generated MkDocs site built from `docs/`.
- `abstractcore.ai` is served from the `gh-pages` branch and uses a custom website with
  `llms-full.txt`, root HTML pages, custom docs pages, and a `CNAME`.

The CI and release workflows install the docs dependencies with:

```bash
python -m pip install -e ".[docs]"
mkdocs build -q
```

This validates that the MkDocs documentation can build, but it does not deploy the generated
`site/` directory and does not update the current `abstractcore.ai` website.

## Problem

Deploying the default MkDocs `site/` output directly would create a second public documentation
surface that may compete with or overwrite the custom `abstractcore.ai` website. The custom site is
not a purely mechanical static-site build today: its update process depends on changes to
`llms-full.txt` being transformed into human-facing HTML pages, which requires active editorial work
from an AI agent.

Because that transformation is agent-driven, it cannot be automated safely by a normal static build
step alone.

## Proposal

Keep the MkDocs build as a CI validation step unless there is a deliberate decision to publish it as
a separate documentation surface.

Possible paths:

1. Keep MkDocs validation-only.
   - Continue using `mkdocs build -q` to catch broken Markdown, navigation, API references, and
     generated docs issues.
   - Do not deploy the generated `site/` directory.
2. Publish MkDocs as a secondary reference surface.
   - Deploy it under a scoped path such as `abstractcore.ai/reference/`, `abstractcore.ai/api/`, or
     a subdomain such as `docs.abstractcore.ai`.
   - Keep the root `abstractcore.ai` experience on the custom `gh-pages` site.
3. Build an agent-backed publishing pipeline for the custom site.
   - Treat `llms-full.txt` as the source input for site updates.
   - During CI/CD, call an authenticated agent endpoint that can review the diff, update the
     relevant HTML pages, and return or commit the generated changes.
   - Require explicit authentication, auditability, and guardrails because this would allow an
     automated system to rewrite the public website.

## Why

This avoids accidentally replacing the existing website with a generic generated docs site while
preserving the value of MkDocs as a documentation health check. It also makes the automation
boundary explicit: the current website publishing process needs an agent with enough context and
authority to edit content, not only a static-site generator.

## Evidence needed before promotion

Promote this to `planned/` only if at least one of these becomes true:

- There is a clear owner decision that MkDocs should be publicly deployed.
- Users need a generated API/reference section separate from the current website.
- The `llms-full.txt` to HTML update process becomes frequent enough that manual agent work is a
  bottleneck.
- A secure agent endpoint is available for CI/CD use, with authentication, logging, and review
  controls.

## Suggested implementation

For validation-only MkDocs:

- Keep the existing CI/release `mkdocs build -q` steps.
- Document that `site/` is a build artifact and is not the production website.

For secondary MkDocs publishing:

- Add an explicit deploy workflow that publishes `site/` to a non-root path or subdomain.
- Ensure it does not overwrite the `gh-pages` root, `CNAME`, or custom HTML pages.
- Add canonical links or navigation so users understand the relationship between the website and
  generated reference docs.

For agent-backed `abstractcore.ai` automation:

- Add a CI/CD job triggered by changes to `llms-full.txt` or manual dispatch.
- Provide the job with a short-lived token for an agent endpoint.
- Send the relevant diff and site constraints to the agent.
- Require the agent to return a patch or open a pull request against `gh-pages`.
- Keep a human review step before publishing unless the endpoint proves reliable enough for a
  tightly scoped automated path.

## Non-goals

- Do not deploy MkDocs output over the current `abstractcore.ai` root by accident.
- Do not assume `mkdocs build` can perform the AI-assisted editorial transformation.
- Do not give CI/CD broad, unaudited permission to rewrite public website pages.
- Do not maintain two divergent public docs sites without a clear user-facing distinction.

## Validation ideas

- Confirm CI still treats MkDocs as a build check only.
- If publishing MkDocs, test that `gh-pages` root files and `CNAME` remain intact.
- For an agent endpoint, test with a dry-run mode that returns a patch without committing.
- Add checks that generated HTML includes expected updates derived from `llms-full.txt`.

## Guidance for future agents

Before changing the docs publishing pipeline, inspect both `main` and `gh-pages`. Treat
`abstractcore.ai` as the canonical public site unless the project owner decides otherwise. If
automation is added, design it around an authenticated agent workflow rather than assuming the
MkDocs `site/` directory is the deployable website.
