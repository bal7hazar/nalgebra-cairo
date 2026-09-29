# Sub-agent strategy (orchestrator)

Instructions for the orchestrator session. This file is meant to be pasted verbatim into the
prompt of an orchestrator of another repository (nalgebra-cairo, rapier-cairo). Porter-side rules
live in `AGENTS.md`, design decisions in `docs/DESIGN.md`, sequencing in `docs/PLAN.md`.

Role of the main session: orchestrate, split, brief, review, merge. Never implement anything
large directly.

## Execution: local CLIs, not the Agent tool

- The Agent tool burns the orchestrator session's quota: use it only for short, read-only
  research.
- Every sub-task runs in its own git worktree + branch (`feat/<module>`), launched in the
  background with its output redirected to a log file.
- Implementation work packages run on the **claude CLI only** (account distinct from the session),
  Opus 5.5 or Sonnet 5 by difficulty, never Fable:
  `claude -p "$(cat brief.md)" --model <sonnet|claude-opus-5-5> --dangerously-skip-permissions --name <task>`;
  resume with context: `claude --continue -p "<follow-up>"` in the same worktree.
- The codex CLI is used sparingly and **only for audits and second opinions** (a review of a PR, a
  numerics cross-check), never for an implementation lot (owner rule, restated 2026-09-25, recorded
  in the programme's `pm/decisions/2026-09-25-codex-audits-only.md`):
  `codex exec -C <worktree> -m <model> -c model_reasoning_effort=<low|medium|high|xhigh> --dangerously-bypass-approvals-and-sandbox -o REPORT.md "$(cat brief.md)"`.
- The agent writes a `REPORT.md` (not committed) at the root of its worktree: the orchestrator
  reads that file and the log, not the transcript.
- On a shared machine, launch each agent as a systemd user unit so that it survives desktop-session
  restarts, with an OOM policy, the build-lock shims and long Bash timeouts (a headless `claude -p`
  ends when its turn ends: agents must never background a command and stop):
  `systemd-run --user --collect --unit=<repo>-<wp> -p OOMPolicy=continue -E PATH="$HOME/orchestrator/shims:$PATH" -E BASH_MAX_TIMEOUT_MS=3600000 -E BASH_DEFAULT_TIMEOUT_MS=1800000 --working-directory=<wt> scripts/agent.sh <wt> claude <model> <brief> <log>`;
  the shims serialise `scarb` / `snforge` builds through `flock ~/orchestrator/heavy-build.lock`.
- Test packages: an agent that needs a new test-only package (compile budget ≈ 7 GB each) adds the
  workspace member and its CI matrix entry; the orchestrator adds the new check to the required
  status checks at merge time (never before: a required check that does not run blocks every PR).

## Model choice by difficulty

| difficulty | claude CLI (implementation) | codex CLI (audits only) | examples |
|---|---|---|---|
| mechanical, well framed | Sonnet 5 | `gpt-5.5` or `gpt-5.6-*` (effort `medium`) | template-generated code, test compaction, spec alignment, benching variants already identified |
| standard port with numerics | Opus 5.5 | `gpt-5.6-*` (effort `high`) | a new module: kernels, tests, golden vectors, benches |
| genuinely complex | Opus 5.5 (Fable is not used for sub-agents) | `gpt-6-astra` (effort `xhigh`) | novel numerics, hard debugging, cross-module design, API arbitration |

- The strong models are not the default, but do not rule them out when the problem warrants
  them.
- The smaller the model (or the lower the effort), the tighter the brief must be.
- The codex model tiering is inferred from the names (`gpt-6-astra` above `gpt-5.6-*`, which are
  above `gpt-5.5`); adjust it if the actual ranking is known. Observed on 2026-09-20: the ChatGPT
  account refuses `gpt-5.6` ("not supported when using Codex with a ChatGPT account"); `gpt-5.5`
  works.

## The brief (mandatory, in this order)

1. Files to read first (`AGENTS.md`, `docs/DESIGN.md`, style precedents on `main`).
2. Strict scope: a file allowlist; everything else is forbidden. Shared files (`lib.cairo`,
   `Scarb.toml`, CI, design docs, CHANGELOG, status) belong to the orchestrator: the agent lists
   its needs in an "Escalations" section of the report instead of editing them.
3. Expected API (exact names from the source being ported), numeric semantics, what is
   explicitly deferred (DEFER).
4. Efficiency rules and numeric targets (gas/steps); variants to bench when the formulation is
   not obvious (the winner in the library, the losers in `benches::alt` with their benches).
5. Tests: table-driven, compile budget (max file size, max number of fuzz tests), golden vectors
   from the reference oracle, panics with exact messages.
6. Definition of done: crate-scoped checks (the touched packages; never the whole-workspace gate on
   the shared machine, the pull-request CI is the full gate) run in the **foreground** (never a background command
   followed by the end of the turn: in headless mode the session stops), gas snapshots
   regenerated, conventional commits with the trailer, push, PR via `gh pr create` following the
   template, `gh pr checks --watch` until green, **never merge**, `REPORT.md` in the imposed
   format (summary, API, gas table, deviations, deferred items, requested re-exports,
   escalations, PR URL).
7. "Work autonomously, do not ask questions, do not widen the scope."

## Conflict-free parallelism

- Pre-declare every stub (modules, tests, benches, golden files) in the shared files before
  launching a wave; one gas snapshot per module. Parallel PRs then never touch a common file.
- Waves follow the dependency graph; a wave starts when its dependencies are merged.
- After each merge, the orchestrator alone updates re-exports, status, changelog and design
  decisions, then pushes to `main`.

## Quality control and quota

- Merge only on green CI + a review of the report (API parity, deviations, gas table).
- An interrupted agent (rate limit, end of turn) is resumed with `claude --continue -p` rather
  than relaunched from scratch.
- Watch the compile budget of the test crates: it is the first cause of CI failure observed.

## Repository tooling

- `scripts/agent.sh <worktree> <claude|codex> <model> <brief.md> <log> [--resume "<follow-up>"]`
  wraps the two CLIs with this strategy (framing system prompt, `--name`, `REPORT.md`).
- Gas snapshots live in `gas/<module>.json` (one per CI shard); `snforge test -p <pkg> | python3
  scripts/gas_report.py --update gas/` regenerates a package's shards, `./scripts/check.sh --update`
  all of them (orchestrator only), CI checks each shard's file.
- `scripts/consumer_cost.py` (repository-agnostic, copied unchanged by the sibling repositories):
  library lines of each published crate (inline tests excluded) and what an empty consumer of it
  adds to a cold build (time, peak memory), against the package granularity rule (40,000 lines,
  5 s / 1 GB, closures 15 s / 3 GB; `consumer_cost.toml`). `--lines-only --report-only` is the fast
  local proxy. CI (WP 9-NS11b): six `Consumer cost shard (k)` jobs (`--repeat 5 --interleave
  --shard k/6`: medians of the per-round differences, balanced shards, new crates included
  automatically) and the ENFORCING job `Consumer cost` that merges them into one verdict (the
  required check; `report_only` entries of `consumer_cost.toml`, the facade among them, are shown and
  never gated; `--report-only-marginals` is the transition switch that gates lines and closures
  only: ON in `ci.yml` until the re-cut of the crate map, owner decision 2026-09-29, because
  `nalgebra_dynamic`'s marginal median is 8.7 s on the current map; `--merge` accepts it too, to
  re-judge downloaded shard files). A deeper measurement of a few crates: `--package A --package B --no-closures --repeat 9
  --interleave`.
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

Standing delegation, confirmed by the owner in this repository's orchestrator session on
2026-09-25: the go for a registry release or a release tag of `simba` (simba-cairo) or `nalgebra`
(nalgebra-cairo) is given **in writing by the programme session ("Angry Birds Cairo
orchestration", the project lead)** on the owner's behalf, under the conditions of
`/home/claude/projects/pm/decisions/2026-09-25-release-go-delegated-to-pm.md`:

1. CI green on `main` at the release commit (every shard, the Workspace job, gas snapshot and shard
   coverage: the whole-workspace gate), and this repository's release checklist followed (gas snapshots, `api_parity.py --check`,
   CHANGELOG entry, `repository` metadata, `scarb package` verified).
2. Version policy: a numeric change is a MINOR bump; pre-releases (`0.1.0-alpha.N`) for packages
   consumed before their API is stable; a release that changes numeric results is scheduled so
   that consumers regenerate their goldens.
3. Publication order follows the dependency chain (`fixed` → `simba` → `nalgebra`), each consumer
   bumping in its own PR.
4. The go arrives as a cross-session message from the programme session and is recorded in its
   `decisions/` or `STATUS.md`; the orchestrator then tags and runs `scarb publish -p <package>`
   (the registry token stays in the owner's environment and is never printed).

Not delegated: money, accounts, credentials, and this session's permission settings.

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
