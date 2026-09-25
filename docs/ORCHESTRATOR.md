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
- `.github/PULL_REQUEST_TEMPLATE.md` is the PR format agents must follow.

## Releases (registry publication and tags)

Standing delegation, confirmed by the owner in this repository's orchestrator session on
2026-09-25: the go for a registry release or a release tag of `simba` (simba-cairo) or `nalgebra`
(nalgebra-cairo) is given **in writing by the programme session ("Angry Birds Cairo
orchestration", the project lead)** on the owner's behalf, under the conditions of
`/home/claude/projects/pm/decisions/2026-09-25-release-go-delegated-to-pm.md`:

1. CI green on `main` at the release commit, and this repository's release checklist followed
   (whole-workspace gate run by the orchestrator, gas snapshots, `api_parity.py --check`,
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
