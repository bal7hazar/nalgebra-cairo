# Orchestration of the nalgebra track

What the orchestrator of the nalgebra track (repositories simba-cairo and nalgebra-cairo) adds to
its standard role, which Nexus keeps (`bal7hazar/nexus`), and to the operating document of the
project, [`slingfall/OPERATIONS.md`](https://github.com/bal7hazar/slingfall/blob/main/OPERATIONS.md)
(models §2, machine and capacity §3, launchers §4, merge gates and audits §6, releases §7). This
file never restates them; where it would contradict them, they win. Porter-side rules live in
`AGENTS.md`, design decisions in `docs/DESIGN.md`, sequencing in `docs/PLAN.md`, the package split
and its dated status in `docs/SPLIT.md`.

## Line and records

- The orchestrator session is created by the project manager of `slingfall` and reports to it
  through this repository: the dated status of the track in `docs/SPLIT.md` and `docs/PLAN.md`; a
  message only for a decision, a blocker, or an objective done. Escalations and cross-repository
  questions (the scalar in fixed-cairo, glam-cairo, rapier-cairo) go to the project manager.
- Machine-local archives, `~/orchestrator/nalgebra-cairo/`: `reports/` (every agent's `REPORT.md`,
  archived before its merge), `logs/`, `wt/` (the agents' worktrees), `refs/` (nalgebra-rs 0.35.0
  sources), `escalations/`, and `briefs/` (the briefs given before 2026-09-30, when they were not
  committed).

## Models

By kind of task, as `OPERATIONS.md` §2 says. The launcher takes a claude CLI model name
(`claude-opus-5-5`, or the alias `sonnet` for mechanical lots). The model that ran is read, never
assumed, from the agent's session transcript, and titles its unit, the orchestrator's background
task and the records:
`grep -oh '"model":"[^"]*"' ~/.claude/projects/-home-claude-orchestrator-nalgebra-cairo-wt-<wp>/*.jsonl | sort | uniq -c`.

## The brief

One brief per task, committed on `main` before the launch as `docs/briefs/<wp>.md` (a
documents-only change). The implementer's branch is cut from `main` after it, so the Codex reviewer
finds it on the pull request's branch (`nexus review ... --brief docs/briefs/<wp>.md`). Every brief
tells the agent to read first the blocks it shares with the others: `docs/briefs/_env.md` (the
machine: shims, foreground only, crate-scoped checks) and, for a package-split move,
`docs/briefs/_move_common.md`. Content, in this order:

1. Files to read first (`AGENTS.md`, `docs/DESIGN.md`, style precedents on `main`, the reports of
   the previous lots).
2. Strict scope: a file allowlist; everything else is forbidden. Shared files (`lib.cairo`,
   `Scarb.toml`, CI, scripts, docs, CHANGELOG, status) belong to the orchestrator unless the
   allowlist names them explicitly (the launcher's framing says the same): the agent lists its
   other needs in an "Escalations" section of the report instead of editing them.
3. Expected API (exact names from the source being ported), numeric semantics, what is
   explicitly deferred (DEFER).
4. Efficiency rules and numeric targets (gas / steps); variants to bench when the formulation is
   not obvious (the winner in the library, the losers in `benches::alt` with their benches).
5. Tests: table-driven, compile budget (max file size, max number of fuzz tests; the compile budget
   of the test crates is the first cause of CI failure observed), golden vectors from the reference
   oracle, panics with exact messages.
6. Definition of done: crate-scoped checks (the touched packages; never the whole-workspace gate on
   the shared machine, the pull-request CI is the full gate) run in the **foreground** (never a
   background command followed by the end of the turn: in headless mode the session stops), gas
   snapshots regenerated, conventional commits with the trailer, push, PR via `gh pr create`
   following the template, `gh pr checks --watch` until green, **never merge**, `REPORT.md` in the
   imposed format (summary, API, gas table, deviations, deferred items, requested re-exports,
   escalations, PR URL).
7. "Work autonomously, do not ask questions, do not widen the scope."

Parallel lots: pre-declare every stub (modules, tests, benches, golden files) in the shared files
before launching a wave, one gas snapshot per module, so that the allowlists of the lots running at
the same time do not overlap; a wave starts when its dependencies are merged.

## Launching, following and resuming an implementer

Implementers start with the track's launcher, `scripts/agent.sh`, as a systemd user unit
(`OPERATIONS.md` §4) until that document names `nexus` for the track; reviews and audits go through
`nexus`. Before every launch: the capacity rule of `OPERATIONS.md` §3 (`~/orchestrator/capacity.json`)
and `nexus resources` (the launcher and `nexus` do not count each other's agents).

```sh
wp=wp-9-r2; model=claude-opus-5-5; title="[Opus 5.5] WP 9-R2 <slug>"
wt=~/orchestrator/nalgebra-cairo/wt/$wp; log=~/orchestrator/nalgebra-cairo/logs/$wp.log
git fetch -q origin && git worktree add -b feat/$wp "$wt" origin/main
systemd-run --user --collect --unit=nalgebra-$wp --description="$title" \
  -p MemoryMax=14G -p OOMPolicy=continue \
  -E PATH="$HOME/orchestrator/shims:$HOME/.venvs/nalgebra/bin:$PATH" -E HOME="$HOME" \
  -E BASH_MAX_TIMEOUT_MS=3600000 -E BASH_DEFAULT_TIMEOUT_MS=1800000 \
  --working-directory="$wt" scripts/agent.sh "$wt" claude "$model" "$wt/docs/briefs/$wp.md" "$log"
```

- The shims serialise `scarb` / `snforge` builds through `flock ~/orchestrator/heavy-build.lock`;
  `~/.venvs/nalgebra` holds `mpmath` for the oracle scripts (system pip is blocked by PEP 668).
  A session whose `systemctl --user` fails with "org.freedesktop.systemd1 exited" is on another
  session bus: prefix the command with `DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus`.
- Follow: one background task of the orchestrator session per agent, titled with the model that ran
  (`[Opus 5.5] WP 9-R2 <slug>`), ending when the unit ends; then the orchestrator reads `REPORT.md`
  at the worktree root and the log, not the transcript.
- An interrupted agent, or one whose work needs a fix, is resumed in its worktree with its context,
  never started again: the same recipe with the unit `nalgebra-$wp-<k>` and
  `scripts/agent.sh "$wt" claude "$model" "$wt/docs/briefs/$wp.md" "$log" --resume "<what to do>"`.
- Test packages: an agent that needs a new test-only package (compile budget ≈ 7 GB each) adds the
  workspace member and its CI matrix entry; the orchestrator adds the new check to the required
  status checks at merge time (never before: a required check that does not run blocks every PR).

## Closing a task

1. Read `REPORT.md` and the pull request: scope within the allowlist, deviations, the gas / step
   table, and for a move the proofs of its checklist (below), checked in the CI logs. Archive
   `REPORT.md` to `~/orchestrator/nalgebra-cairo/reports/<wp>.md` before the merge
   (`gh pr merge --delete-branch` removes the worktree).
2. With every check green, the Codex review:
   `nexus review --project slingfall --task <9-R2> --repository nalgebra-cairo --branch feat/<wp> --brief docs/briefs/<wp>.md`,
   then `nexus wait <handle>` as a background task titled with the reviewer's model, and the audits
   `OPERATIONS.md` §6 requires for the kind of task. A finding is verified before it goes back to
   the implementer by `--resume`; a new review follows on the new head.
3. Squash merge, with the verdict in the merge body (`Codex review: <handle> PASS`, or
   `Codex review: none — <reason>` in the cases the standard allows); then the orchestrator alone
   updates re-exports, status, changelog and design decisions.

## Repository tooling

- `scripts/agent.sh <worktree> <claude|codex> <model> <brief.md> <log> [--resume "<follow-up>"]`
  wraps the CLI with the framing system prompt, `--name` and `REPORT.md`; implementation lots use
  `claude` only.
- Gas snapshots live in `gas/<module>.json` (one per CI shard); `snforge test -p <pkg> | python3
  scripts/gas_report.py --update gas/` regenerates a package's shards, `./scripts/check.sh --update`
  all of them (orchestrator only), CI checks each shard's file.
- `scripts/packages_table.py` (repository-agnostic): renders a `consumer_cost.py --json` file as the
  `docs/PACKAGES.md` table (lines and marginal cost with their margins against the gates, closures
  against their budget, run header); the `Consumer cost` job uploads it in the artifact
  `consumer-cost` and appends it to the job summary. A closure may carry its own budget
  (`budget = { seconds = 20, gb = 4.5 }` in `consumer_cost.toml`, or `--closure NAME=a,b::20s,4.5gb`).
- `scripts/consumer_cost.py` (repository-agnostic, copied unchanged by the sibling repositories):
  library lines of each published crate (inline tests excluded) and what an empty consumer of it
  adds to a cold build (time, peak memory), against the package granularity rule (40,000 lines,
  5 s / 1 GB, closures 15 s / 3 GB; `consumer_cost.toml`). `--lines-only --report-only` is the fast
  local proxy. CI (WP 9-NS11b): six `Consumer cost shard (k)` jobs (`--repeat 5 --interleave
  --shard k/6`: medians of the per-round differences, balanced shards, new crates included
  automatically) and the ENFORCING job `Consumer cost` that merges them into one verdict (the
  required check; `report_only` entries of `consumer_cost.toml`, the facade among them, are shown and
  never gated; `--report-only-marginals` is the transition switch that gates lines and closures
  only: ON in `ci.yml` until `nalgebra_dynamic` is cut under gate 2, owner decision 2026-09-29 and
  `docs/SPLIT.md` §20; `--merge` accepts it too, to re-judge downloaded shard files). A deeper
  measurement of a few crates: `--package A --package B --no-closures --repeat 9 --interleave`.
- `scripts/facade_features.py` (CI job `Facade features`): consumers of the workspace facade naming
  its five no-op features (`statistics`, `blas`, `dynamic`, `sparse`, `io`; docs/SPLIT.md §17) build,
  with and without the default features; `--resolve` is the cheap local form.
- `.github/PULL_REQUEST_TEMPLATE.md` is the PR format agents must follow.
- Package split (M9, `docs/SPLIT.md` §7, §11), tools of `tools/split/`:
  - `crates.toml`: the crate map (planned crates -> the package that hosts each one today, and the
    placement rules). Read by the generators, `api_parity.py`, `public_paths.py`,
    `rewrite_imports.py`; `NALGEBRA_CRATE_MAP=<file>` points them at another map.
    `cratemap.py --show` prints it; `--compare-plan` / `--compare-tree` compare it with NS1's plan.
  - `public_paths.py --check` (CI job `Path proof`): the facade exports exactly the 0.1.0 public
    paths (`public_paths_0.1.0.txt` + `public_paths_added.txt`) and a consumer naming each one
    builds. A PR that adds a public item appends its paths to `public_paths_added.txt`.
  - `gas_compare.py --base origin/main --head gas/`: zero-step proof of a move (package segment
    of the library packages dropped).
  - `rewrite_imports.py [--write] crates/<test package>`: `use nalgebra::X` -> the sub-crates.
- **Move-PR checklist** (NS3..NS11), in this order: (1) switch the moved crates to their package
  in `tools/split/crates.toml` (`core = "nalgebra_core"`), create the package (manifest, hand-written
  files moved with `git mv`, workspace member); (2) `python3 tools/shapegen/shapegen.py` and
  `python3 tools/linalggen/generate.py` (they write the generated items into the new package, the
  facade re-exports and module roots), then both `--check`; (3) `python3
  tools/split/rewrite_imports.py --write` on the test packages that import moved items, `scarb
  fmt`; (4) `python3 scripts/api_parity.py --check` (unchanged report) and `python3
  tools/split/public_paths.py --check`; (5) `snforge test -p <pkg> | python3 scripts/gas_report.py
  --update gas/` for each moved or rewritten package, then `python3 tools/split/gas_compare.py
  --base origin/main --head gas/` (0 changed, 0 missing, 0 added: paste the summary line in the
  PR); (6) `python3 scripts/consumer_cost.py --lines-only --report-only` for the new crates' lines;
  (7) `python3 tools/split/cratemap.py --anchors` (0 findings: each moved impl's module is its
  trait's or one of its argument types', and hand-written blocks are placed as the map says)
  and the `[internal]` table for former `pub(crate)` items the facade package still needs
  (`docs/SPLIT.md` §15); in-crate tests stay in the package hosting their methods.

## Releases (registry publication and tags)

On the project manager's written go, under the conditions of `OPERATIONS.md` §6-§7 (the owner's
delegation of 2026-09-25, `pm/decisions/2026-09-25-release-go-delegated-to-pm.md`). What this
repository adds:

1. CI green on `main` at the release commit (every shard, the Workspace job, gas snapshot and shard
   coverage: the whole-workspace gate), and this repository's release checklist followed (gas
   snapshots, `api_parity.py --check`, CHANGELOG entry, `repository` metadata, `scarb package`
   verified).
2. Version policy: a numeric change is a MINOR bump; pre-releases (`0.1.0-alpha.N`) for packages
   consumed before their API is stable; a release that changes numeric results is scheduled so
   that consumers regenerate their goldens.
3. Publication order follows the dependency chain (`fixed` → `simba` → `nalgebra`), each consumer
   bumping in its own PR; the registry token stays in the owner's environment and is never printed.

**The release script** `scripts/release.py` (WP 9-NS11b; dry run by default, `--publish` after the
go): the package set and the publication order come from `scarb metadata` (every package not marked
`publish = false`, topological order of the path dependencies, then the facade, then the bridges
such as `nalgebra_glam`); no name is hard-coded. It refuses to start unless the tree is clean, HEAD
is `origin/main`, every check run of that commit is green with `Consumer cost` among them, every
published package has the workspace version and that version is not on the registry yet. It
publishes one package at a time (`scarb publish -p`), waits for each on the registry index and
compares the index checksum with the local archive before the next one, and is resumable: the state
file `target/release/state-<version>.json` records the verified packages, so running the same
command again after a failure at package k skips the packages before k (after checking the index)
and restarts at k. It does not bump versions nor tag: the version bump is the release PR, the tag
`v<version>` follows the publication. Release sequence: release PR (version bump in the workspace
`Scarb.toml` and the intra-workspace requirements, CHANGELOG) merged with CI green → `python3
scripts/release.py` (dry run: no refusal) → the go → `python3 scripts/release.py
--publish` → tag.
