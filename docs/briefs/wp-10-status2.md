# WP 10-ST2 — status of the nalgebra track after the pre-push and CI lots (documents only)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Light lot (documents only); runs on the VPS. Commit this brief verbatim as
`docs/briefs/wp-10-status2.md` (first commit). Allowlist: that brief, `docs/PLAN.md`, `docs/SPLIT.md`
§21 (one new dated entry; earlier entries untouched except the two corrections below). No code.

## Facts to record (check each against its source: `gh pr view <n>`, the merged files, `git log`)

Merged on 2026-10-02/03:
- nalgebra-cairo #95 (`0e478c4`): pre-push check, hook, retries on the 12 setup steps, PR-only cancel.
- nalgebra-cairo #96 (`7ee554d`): pre-push follow-ups (untracked build files refused; fixed checks
  gated on their inputs; lock wait overlaps the checks; interrupt stops the Cairo block; ancestor-held
  lock; realpath; full gas comparison when gas/** changes with a package). Times in the PR body.
- nalgebra-cairo #97 (`7111163`): CI path gating (a `changes` job; 8 jobs gated; prose `.md`
  triggers nothing; `docs/API_PARITY.md` triggers the workspace job; pushes to main run everything in
  a per-commit concurrency group; `CI result` always runs).
- simba-cairo #6 (`9b7c065`) CI path gating; #7 (`d2a80f8`) pre-push follow-ups.

Corrections of #94 (review notes, do them in this lot):
1. SPLIT §21, the 2026-10-02 entry's WP 10-TC bullet: "every entry moved, uniformly within each test
   group" is not true for 31 groups (10 in gas/, 21 in benchmarks/gas/: the +100 builtin/dictionary
   tests and the exp4-6 baselines). Correct it in place, with "(corrected 2026-10-03)".
2. PLAN, row TC: the gas moves are not all the snforge harness offset: −7830 typical, other group
   offsets from setup code under Cairo 2.20.0 and the +100 builtin charge.

Plan rows: PP-N done (#95, #96); CI path gating done (both repos); a new deferred row PP-FU2 for the
pre-push review notes being fixed now in lot PP-FU2 (in progress); note the reading of the brief rule
"kills nothing" as "kills nothing the script did not start" (review of #96).

## Rules
Every fact checked; plan format kept; one PR `docs: …`; all fixes of one review loop in one push; CI:
at most one GitHub call per PR every 5 minutes, or end your turn; never merge, never launch a review or
any agent. Foreground only. `REPORT.md` at the worktree root (not committed), `## Summary` first.
Prefer words over links to external issues.

Work autonomously, do not ask questions, do not widen the scope.

