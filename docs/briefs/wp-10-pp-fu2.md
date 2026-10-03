# PP-FU2 — pre-push review notes, both repositories (nalgebra-cairo)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Light lot (shell only); runs on the VPS. Commit this brief verbatim as `docs/briefs/wp-10-pp-fu2.md` (first
commit). Allowlist: the brief, `scripts/prepush.sh`, `.githooks/pre-push`. Nothing else.

## Fix (deferred notes of the reviews of #96)
1. A change of a manifest or lock file that is not a member of the root workspace (`benchmarks/**/Scarb.toml`,
   `benchmarks/**/Scarb.lock`, `tools/**/Scarb.toml|lock`) must not start a whole-workspace `snforge test`
   under the lock: treat an empty package map that is not caused by `gas/**` or `gas_report.py` as
   "nothing to compile" (fixed checks still follow their own triggers).
2. `stop_jobs` / kills: before signalling a group, check its leader is still a child of `$$`
   (`/proc/<pgid>/stat` field 4) or that it has not been waited yet, so a reaped id is never signalled.
3. Header: state that the script and hook need bash >= 4.4 (`wait $!` on a process substitution, empty
   arrays under `set -u`), and fail early with a clear message on an older bash.

## Rules
- Lock: the script holds `~/orchestrator/heavy-build.lock` itself with `flock -E 75 -w 90`, exports
  `HEAVY_BUILD_LOCK_HELD=1` inside, signals only process groups it started, never another holder; exit
  75 = busy → `heavy lock busy: Cairo compile left to CI`, pass. **Never probe the shared lock, not
  even with `flock -n`**: test lock cases on a private `HEAVY_BUILD_LOCK=$(mktemp)` file.
- Under `set -euo pipefail`, no function or trap may end on a `[[ … ]] && x` or other command that can
  be false; end trap helpers with `return 0`.
- Git: reads of this repository only, or a sanitised env. Never `git config core.hooksPath`.
- Before pushing, RUN at the new head and put in the PR body with exit codes: prose docs only (0, in
  seconds); a passing fixed check (0); a failing fixed check (1 with the FAILED line); a Cairo change on
  a private lock (0); the hook end to end once (0).
- Push through the hook. All fixes of one review loop in one push. CI: at most one GitHub call per PR
  every 5 minutes, or end your turn. Never merge, never launch a review or any agent. Foreground only.
  `REPORT.md` at the worktree root (not committed), `## Summary` first.

Work autonomously, do not ask questions, do not widen the scope.
