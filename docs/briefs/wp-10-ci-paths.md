# CI-N — nalgebra-cairo: test jobs run only when their files change

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-02.
Light lot (workflow only, no Cairo build); runs on the VPS. Starts after #95 (pre-push, which edits the
same workflow) is merged; branch from the `origin/main` that contains it. Commit this brief verbatim
as `docs/briefs/wp-10-ci-paths.md` (first commit). Allowlist: that brief and
`.github/workflows/ci.yml` (not `split-measure.yml`, which runs only by manual dispatch).

## Mapping table (job → paths that trigger it on a pull request)

T = the toolchain set: `.tool-versions`, any `Scarb.toml`, any `Scarb.lock`, `.github/workflows/**`.
Prose `**/*.md` never triggers; checked `.md` artefacts do (§5).

| Job (name) | Group | Paths |
|---|---|---|
| `workspace` (Workspace: fmt, parity, generated code, lint, build) | `workspace` | T, `crates/**`, `scripts/**`, `tools/shapegen/**`, `tools/linalggen/**`, `tools/split/**`, `consumer_cost.toml` |
| `consumer-cost-shard` (Consumer cost shard N) and `consumer-cost` (Consumer cost) | `cost` | T, `crates/**`, `scripts/consumer_cost.py`, `scripts/packages_table.py`, `consumer_cost.toml` |
| `facade-features` (Facade features) | `facade` | T, `crates/**`, `scripts/facade_features.py` |
| `path-proof` (Path proof) | `paths` | T, `crates/**`, `tools/split/**` |
| `library` matrix (Library (…)) and `gas` (Gas snapshot) | `library` | T, `crates/**`, `gas/**`, `scripts/gas_report.py`, `scripts/shard_coverage.py` |
| `benchmarks` (Design benchmarks) | `benchmarks` | T, `benchmarks/**`, `scripts/gas_report.py`, and `crates/**` if any benchmark package depends on a workspace crate by path (check `benchmarks/**/Scarb.toml`) |
| `oracle` (Oracle (golden vectors)) | `oracle` | `tools/oracle/**`, `.github/workflows/**` (it builds with cargo, not scarb) |
| `changes` | — | always |
| `CI result` | — | always (summary, §6 below) |

`benchmarks` and `oracle` already compute their own `dorny/paths-filter` step: move their filters into
the `changes` job (same patterns, extended by the table) and gate the jobs with `if:`, so a skipped
job shows as skipped, not as a green job that did nothing; keep the inner step conditions only where
they still matter (e.g. the old-toolchain `benchmarks/libs` steps).

Read the workflow and every script a job runs before applying the table; if a job reads a file the
table misses, add it and say so in the PR body (a missing trigger is worse than an extra one).
## Common rules (both repositories)

Owner's CI rule (2026-10-02): "CI tests must absolutely run only if files related to the tests were
modified, so docs should skip all tests."

1. **Workflow file this lot changes: `.github/workflows/ci.yml`** (and nothing else under
   `.github/`). Edit it **with the file-editing tool only, never by a script rewrite** (no sed/awk/
   python rewriting the YAML).
2. A first job `changes` computes, **from the changed paths only** (never from a label), one boolean
   output per group of the table below, with `dorny/paths-filter` pinned by sha (nalgebra-cairo
   already pins `dorny/paths-filter@ceb8a2b8f2d89434be7ff52d3de7ec3738c5cc9d # v4.0.3`: use exactly
   that pin in both repositories). Each test job gets `needs: changes` and
   `if: github.event_name == 'push' || needs.changes.outputs.<group> == 'true'` (a job that already
   has `needs`/`if` combines them; a job that `needs` a skipped job must not fail because of it).
3. **Pushes to `main` keep their full CI**: every job runs on `push`, whatever changed.
4. **The toolchain set triggers everything**: `.tool-versions`, any `Scarb.toml`, any `Scarb.lock`,
   `.github/workflows/**`. Put it in every group.
5. **Prose documents trigger nothing; checked `.md` artefacts do.** A `.md` file that a job reads,
   regenerates or compares (for example `docs/API_PARITY.md` for the parity check,
   `docs/PACKAGES.md` for Consumer cost, `gas/*.md` for the gas check, `CHANGELOG.md` if a release
   check reads it, any table a script regenerates) triggers the job that checks it; every other
   `.md` triggers nothing. Find them yourself: grep every script and step a job runs for `.md`
   paths, and list in the PR body each checked `.md` with the job it triggers. A PR that changes only
   prose `.md` files (e.g. `docs/X.md`, `crates/core/README.md` if no job reads it) sets every
   output to false. The tables above say "`**/*.md` never triggers": read that as "prose `.md`".
6. **A final job that always runs**, named `CI result`, `needs:` every other job, `if: always()`.
   It fails when any job that ran failed or was cancelled, and passes only when every job either
   passed or was **skipped by the paths rule**, never skipped by error: for each job, `skipped` is
   accepted only if that job's `changes` output was `false` (and the event is a pull request); a job
   skipped while its output was `true` (e.g. because a job it needs failed or `changes` itself
   failed) fails `CI result`, and `CI result` fails if `changes` did not succeed. It is what
   `gh pr checks` always sees, so the standard's merge command never lacks checks. List which jobs
   it aggregates, with their groups, in a comment.
7. `concurrency`: keep the PR-only cancel line `cancel-in-progress: ${{ github.event_name ==
   'pull_request' }}`, and give each push to main a group of its own commit, because a group keeps
   only one pending run and would drop queued main runs even without cancelling:
   `group: ${{ github.workflow }}-${{ github.event_name == 'pull_request' && github.ref || github.sha }}`.
   Every main commit then keeps its full run.
8. Keep every existing job, step, pin, retry and `RAYON_NUM_THREADS` setting as it is; only add the
   `changes` job, the `needs`/`if` lines and `CI result`.
9. **Show it works**, in the PR body, with the real `changes` outputs from CI logs: (a) this PR itself
   (workflow changed → everything runs); then a throwaway **draft** PR or extra commit on a scratch
   branch is NOT allowed (it would spend the starved Actions queue) — instead, show the filter
   behaviour locally: run the same patterns through a small read-only check (e.g. a shell or Python
   snippet in the worktree, not committed) on three synthetic path lists: docs-only
   (`docs/X.md`, `crates/core/README.md`) → all false; one Cairo file → the expected groups true;
   `.tool-versions` → all true. State the tool's matching semantics you relied on.
10. Git commands: only reads of this repository, or in a sanitised environment (`env -u GIT_DIR -u
    GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_COMMON_DIR -u GIT_PREFIX`).
11. Push through the repository's pre-push hook (`git -c core.hooksPath=.githooks push` if the
    clone's config does not set it; never write the config). All fixes of one review loop in one
    push. CI: at most one GitHub call per PR every 5 minutes, or end your turn. Never merge, never
    launch a review or any agent yourself. Foreground only. `REPORT.md` at the worktree root (not
    committed), `## Summary` first, with the PR number and head sha. Prefer words over links to
    external issues in PR text and commit messages. Audit: none needed — CI gating, every job still
    runs on main.

Work autonomously, do not ask questions, do not widen the scope.
