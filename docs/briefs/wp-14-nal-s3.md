# WP 14-NAL-S3 — nalgebra 0.3.0: simba 0.3.0, fixed 0.5.0, glam_core 0.5.0 (no result moves)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-05.
Runs on the **VPS**, crate-scoped only: never build or test the whole workspace locally (it peaks near 11 GB). Every
local `scarb`/`snforge` is `-p <crate>` under `prlimit --as=8589934592 -- …`; the facade `nalgebra` is not built
locally. The pre-push hook caps its own compile (#104) and leaves it to CI if the cap is reached. CI runs everything:
CI is the gate. This lot prepares the release; it publishes nothing (the request is a separate lot).
Commit this brief verbatim as `docs/briefs/wp-14-nal-s3.md` (first commit).

## 1. Goal

1. **Dependencies** (all three are on scarbs.xyz): `simba = "0.3.0"`, `fixed = "0.5.0"`, `glam_core = "0.5.0"` wherever
   the workspace pins them; `Scarb.lock` by scarb; `consumer_cost.toml` closures if they name the versions. 0.2.0 could
   not take simba 0.3.0 because `nalgebra_glam` pulled `glam_core` 0.4.1 (`fixed ^0.4.0`); `glam_core` 0.5.0 depends
   on `fixed ^0.5.0` and lifts it.
2. **No nalgebra result moves.** Every existing test, golden, oracle vector, gas and steps entry stays as it is — any
   change is a stop: report it with old/new values and escalate. Specifically:
   - simba 0.3.0's `is_sign_positive(0)` is now `true`. The 6 call sites (docs/research/rel-zero.md): site 418 of
     `crates/geometry3/src/geometry/unit_quaternion.cairo` already compares `tr > R::zero()` (nalgebra-rs's own
     comparison, pinned by `test_from_rotation_matrix_zero_trace`, 0.2.0); sites 203-209 of
     `crates/geometry3/src/internal/geometry/unit_quaternion.cairo` see a zero only on a non-unit input no caller
     passes; `unit_complex.cairo:755` catches zero first. Re-check each with simba 0.3.0 in place and say so.
   - glam_core 0.5.0 changes `Quat::to_axis_angle` / `to_scaled_axis` for vector parts of length in [2^-16, 2^-8):
     confirm nalgebra (and `nalgebra_glam`) never calls them.
   - fixed 0.5.0 is a pure addition (its CHANGELOG): confirm no delegated kernel's result changed.
3. **Version 0.3.0** for every published package (MINOR: simba's public trait moves 0.2 -> 0.3) and every inter-crate
   requirement. `CHANGELOG.md` 0.3.0 ("unreleased"): **Changed** — simba 0.3.0 / fixed 0.5.0 / glam_core 0.5.0, no
   nalgebra result moves (the sites above, citing docs/research/rel-zero.md); **Performance** — OPT-3 (#109): the
   shapegen probes' before/after from docs/STEPS.md (Matrix3 mul 186->161, determinant 82->65, try_inverse 430->412,
   Matrix6 mul_vec 192->143; the branch savings measured off-probe as STEPS.md says).
4. **Carried from the review of #108 (minor and note):** in `docs/releases/0.2.0.md` (~lines 222-224) and
   `docs/SPLIT.md` (~line 1461), replace "the Mac, or the VPS under a ruling like this one / a new ruling" by the
   Overseer's standing rule of 2026-10-04: a package whose `scarb publish` verification exceeds the 8 GiB cap is
   published by the orchestrator on the VPS inside the heavy flock, `RAYON_NUM_THREADS=1`,
   `prlimit --as=25769803776 -- /usr/bin/time -v scarb publish -p <pkg>`, started only at >= 20 GB free, its peak
   recorded in the release record; a package whose last recorded peak exceeded 16 GiB RSS, or whose verification failed
   once under that rule, goes to the owner on the Mac. In `docs/PLAN.md` row REL, add the missing ". " before
   "Released"; row NAL-S3 "in progress".

## 2. Allowlist

The brief; manifests (`version`, the three dependency pins and inter-crate requirements only) and `Scarb.lock` (by
scarb); `consumer_cost.toml` (versions only); `CHANGELOG.md`; `docs/releases/0.2.0.md` and `docs/SPLIT.md` (the
lines of item 4 only); `docs/PLAN.md` (rows REL, NAL-S3); `docs/PACKAGES.md` (only if regenerated from this PR's CI
artifact because line counts or costs changed — say so). No Cairo source change (none is expected; a forced one is an
escalation). Not: `.github/**`, scripts.

## 3. Verification and report

Crate-scoped local checks of `geometry2`, `geometry3`, `nalgebra_glam` and the simba-facing crates under the cap;
push through the hook (use the `ServerAliveInterval` form if it may run long); CI green. PR body: the three bumps,
the per-site re-check, the glam/fixed confirmations, `git diff --stat origin/main -- 'crates/tests_*' gas steps`
(empty), the package count at 0.3.0, the CHANGELOG. "Audit: none needed — dependency bump, no result moves".
Conventional commits; one push per review loop; never merge, never publish, never launch a review or any agent;
foreground only; `REPORT.md` at the worktree root (not committed), `## Summary` first.

Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call (if your thread cannot run it, name the job and run id in your report and end your turn: the orchestrator relays the log). Never `gh api …/logs`. To read a sibling repository (simba-cairo, fixed-cairo), use the Read tool on its checkout. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).
Picking up main: after any push (and until nexus #83 is installed), run `git merge origin/main`, alone in its call. `git rebase origin/main` in exactly that form is allowed only before your first push, once #83 is installed; every other rebase form and every force push stay refused.
Files: to remove an untracked file of your own worktree, use `git clean -f -- <exact path>` (nexus #92); a bare `rm` stays refused. NEVER run `git clean` with `-d`, `-x` or `-X`: it deletes your own .herdr-project folder (brief, report). `gh pr edit` works for a PR title and body (nexus #91).
Memory (organisation rule): the 8 GB cap (`prlimit --as=8589934592 -- /usr/bin/time -v <command>`) applies to Cairo builds and measures (scarb, snforge) on the VPS; every peak-memory measure stays capped; a Node suite, or a whole pre-push hook, runs uncapped only when each of its steps was measured well under 8 GB (`prlimit --as` kills Node at start-up). On the Mac there is no cap. Long pushes: if your pre-push hook may run long (several minutes), push with `git -c core.sshCommand='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=40' push ...` (writes no config), otherwise GitHub drops the SSH connection (exit 141) and nothing lands. Real peak memory: measure it uncapped only on the Mac (64 GB, no lock), never uncapped on the VPS; on the VPS, `prlimit --as=8589934592` (an address-space cap, which can kill a Cairo build at ~6.6 GB of real memory) stays the cap, used only for Cairo work known to fit well under it.

Work autonomously, do not ask questions, do not widen the scope.
