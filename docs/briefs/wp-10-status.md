# WP 10-ST — status of the nalgebra track after the toolchain lots (documents only)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-02.
Light lot (documents only, no build); runs on the VPS.

## 1. Goal

Bring `docs/PLAN.md` and `docs/SPLIT.md` §21 (the dated status of the track) up to date with what was
merged on 2026-10-02, and record the deferred follow-ups as plan rows, so the next lots start from
the repository and not from a session's memory. Facts below are the orchestrator's; check each
against its source (`gh pr view <n>`, the merged files, `git log`) and quote nothing you could not
confirm.

**Merged on 2026-10-02:**
- nalgebra-cairo #90 (`2880920`): post-release documents of 0.1.1 (reviews: Sonnet, then Opus PASS
  WITH FINDINGS at fa1543c).
- nalgebra-cairo #92 (`9dc3360`): REL-FU, `scripts/release.py` follow-ups (self-test in CI;
  continuation check covers ignore files of intermediate directories and inherited readme /
  license-file; a package in flight whose index lags is verified, never republished).
- simba-cairo #4 (`33fba0d`): TC-S, Scarb 2.20.1 / starknet-foundry 0.64.0 (Cairo 2.20.0), no
  release; results unchanged; gas moved by the snforge 0.64.0 harness offset (−7830 per bench).
- simba-cairo #5 (`811e3ab`): pre-push check, hook and CI retries for simba-cairo.
- nalgebra-cairo #93 (`d87c4e5`): WP 10-TC, Scarb 2.20.1 / starknet-foundry 0.64.0, no release;
  every test passes unchanged; path proof 9,289 / 9,289; gas: every entry moved, uniformly within each
  test group (−7830 typical, between −7830 and +8070 across groups); 89 margins over a baseline
  changed, all explained (snforge 0.64 charges +100 per test using a builtin or a dictionary; the
  exp4-6 baselines' setup moved) and accepted by the orchestrator; Consumer cost with the compiler's
  default threads: every gate passes, facade 35.8 s → 33.9 s; one compiler thread roughly doubles
  cold-build times on the runners, so timing jobs run on default threads and gas / hash / test jobs
  on one; `benchmarks/libs` stays on Scarb 2.19.4 / snforge 0.61.0 (vendored cubit and orion do not
  compile on corelib 2.20.0, which removed `u32_as_non_zero` / `u64_as_non_zero`).
- In progress: nalgebra-cairo's own pre-push lot (WP 10-PP).

**Follow-ups to record as plan rows (deferred, none blocking):**
1. `scripts/release.py` (review notes of #92): the docstring says 45 index checks where the code
   makes 46; `--verify-timeout` no longer caps wall-clock time (estimated up to ~38 min against a slow
   index); a small window between a successful publish and saving the `submitted` marker.
2. simba-cairo's pre-push hook (review notes of #5): a tag of a tree or blob gives a garbled
   message; the tag exemption does not check that the tag points at an already-checked commit; a
   stale "only deletions" comment. (A row here, since simba's plan is this track's.)
3. From #93: `tools/split/probes`, `tools/shapegen/proto`, `tools/shapegen/compare.py` still pin
   2.19.4 / 0.61.0 (no CI job builds them); the curated figures of `benchmarks/*/README.md` and
   `GAS.md` keep their 0.61.0 values; the `--md` option those READMEs name does not exist in
   `benchmarks/scripts/gas_report.py`; `benchmarks/libs` on the old toolchain until its vendored
   libraries are updated or retired.
4. Publishing is now done by the orchestrator by hand, one `scarb publish -p <package>` per
   package on the project manager's go naming package, version, commit and archive sha256
   (organisation standard, 2026-10-02): `scripts/release.py`'s publishing mode must be reconciled
   with that rule before the next release (row only, no change now).
5. A future Scarb release carrying the upstream fix that makes closure type names path-free changes
   Sierra / class hashes of anything holding a closure: such a bump gets its own lot. (Row only.)

## 2. Scope: file allowlist

This brief committed verbatim as `docs/briefs/wp-10-status.md` (first commit); `docs/PLAN.md` (the
TC, REL-FU rows marked done with their PRs; new rows for the follow-ups; a row for WP 10-PP in
progress); `docs/SPLIT.md` §21 (a new dated entry, 2026-10-02; earlier entries untouched). Nothing
else: no code, no CHANGELOG (no release), no other doc.

## 3. Acceptance criteria and report

Every fact checked against its source (list the commands); the plan's existing format kept; one PR
`docs: …`; CI green. All fixes of one review loop in one push. CI: at most one GitHub call per PR
every 5 minutes, or end your turn. Never merge, never launch a review or any agent yourself.
Foreground only. `REPORT.md` at the worktree root (not committed), `## Summary` first, with the PR
number and head sha. Prefer words over links to external issues in PR text and commit messages.

Work autonomously, do not ask questions, do not widen the scope.
