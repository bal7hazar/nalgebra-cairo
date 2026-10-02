# WP 10-REL-FU — follow-ups of the 0.1.1 release in `scripts/release.py`

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-02.
Light lot (Python only, no Cairo build). Row `REL-FU` of `docs/PLAN.md`. Runs in parallel with
WP 10-TC (the toolchain bump), whose files are disjoint from yours except `ci.yml` (see §2).

## 1. Goal, context, files to read first

**Goal.** Close the three follow-ups that the Fable review of #89 noted on `scripts/release.py`:

1. `python3 scripts/release.py --self-test` runs in CI, in the Workspace job.
2. The continuation check (`continuation_problems` and the paths it is given) also covers
   (a) ignore files in **intermediate directories** between the workspace root and a published
   package (a `.gitignore` / `.scarbignore` in `crates/` changes what a package under it ships), and
   (b) a `readme = true` (or `readme` / `license-file`) **inherited from the workspace**
   (`readme.workspace = true` and the like), whose file lives outside the package.
3. A **lagging index for the package in flight**: when a run stopped after `scarb publish` succeeded
   for a package but before the registry index listed it, a resumed run must not treat that package
   as unpublished and publish it again, nor fail; it waits / re-checks the index (bounded, with the
   delay and retry count stated) and records it once listed.

Read first: `AGENTS.md`, `scripts/release.py` whole (docstring: lines 1-60; `plan_resume`,
`continuation_packages`, `continuation_problems`, `workspace`, `self_test`, `main`),
`docs/SPLIT.md` §21-§22 (how 0.1.1 was published: 12 packages from `3e5e4ba`, 42 from `b3915c7`),
`docs/briefs/` for the brief format, `.github/workflows/ci.yml` (Workspace job).

## 2. Scope: file allowlist

- This brief, committed verbatim as `docs/briefs/wp-10-rel-fu.md` (first commit).
- `scripts/release.py`: the three changes, each with self-test cases in `self_test()` (no network, no
  scarb): at least one case per change that fails on the old code and passes on the new.
- `.github/workflows/ci.yml`: **only** the Workspace job's step "Consumer cost script (self-test,
  consumer manifests of the dry run; no build)" — add `python3 scripts/release.py --self-test` to it
  (and rename the step if you must). Touch no other line: WP 10-TC edits the top-level `env:` block
  and the toolchain lines of the same file in parallel.
- `docs/ORCHESTRATOR.md` or the docstring of `release.py`: the lines that describe the continuation
  check and the resume, if they change.
- Nothing else: no Cairo file, no manifest, no `CHANGELOG.md`, `docs/PLAN.md`, `docs/SPLIT.md`.

## 3. Interfaces

`release.py`'s command line is unchanged (`--self-test`, `--publish`, and the other existing flags);
its state file format stays readable by the new code (a state file of 0.1.1,
`{"version": ..., "commit": ..., "published": [...]}` or whatever the current format is, must load).
**Never run `--publish`** and never call the registry with a write: publishing is reserved for the
owner. Read-only index queries in a manual check are allowed.

## 4. Tests

`python3 scripts/release.py --self-test` prints `self-test: ok`; each new case is shown to fail on
`origin/main`'s `release.py` (run the new cases against the old functions, or state the reason a case
cannot run there). A dry run without `--publish` on the current workspace (if the script has one)
behaves as before: show its output before and after.

## 5. Acceptance criteria

1. The three follow-ups implemented, each with self-test cases (§4).
2. CI's Workspace job runs the self-test (visible in the log), every CI check green.
3. No change outside the allowlist; no publication, no registry write.

## 6. Verification (definition of done)

Conventional commits, push, `gh pr create` per the template (body: what each follow-up does, the
self-test output, "Audit: none needed — release tooling, no publication in this lot; covered by
self-tests and the review"), `gh pr checks --watch` until green. Never merge. Foreground only.

## 7. Report expected

`REPORT.md` at the worktree root (not committed), `## Summary` first: each follow-up and its test
cases, the before/after dry-run output, Escalations, the PR number and head sha.

Work autonomously, do not ask questions, do not widen the scope.
