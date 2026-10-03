# WP 12-REL-b — nalgebra 0.2.0: simba 0.3.0, version bump, CHANGELOG (no result moves)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Runs on the **VPS**, crate-scoped only: never build or test the whole workspace locally (it peaks near
11 GB). Every local `scarb`/`snforge` is crate-scoped (`-p <crate>`) under `prlimit --as=8589934592 -- …`;
the facade package `nalgebra` is not built locally. The pre-push hook caps its own compile (#104) and
leaves it to CI if the cap is reached. CI runs the whole workspace and the goldens: CI is the gate; take
gas/steps snapshot updates from CI logs if a whole-workspace run is needed to produce them.
REL-a (#105) already did the release tooling and the doc lines: items 4 and 5 below are DONE, skip them.
This lot prepares the release; it publishes nothing. The release request (archive sha256 per package)
is a separate documents lot after this one merges.

## 1. Goal

1. **simba 0.3.0** (published 2026-10-03, scarbs.xyz): `simba = "0.3.0"` (and `fixed = "0.5.0"`, simba's
   dependency) wherever the workspace pins them; `Scarb.lock` by scarb; `consumer_cost.toml` closures if
   they name the versions.
2. **The one result change simba 0.3.0 brings**: `Real::is_sign_positive(0)` now returns `true` (was
   `false`), as simba-rs (`+0.0` is positive). nalgebra calls `R::is_sign_positive` at 6 sites; 5 can
   change a result at an exact zero (the review of simba #11 checked them):
   - `crates/geometry3/src/geometry/unit_quaternion.cairo:418` (trace branch of the rotation-matrix
     conversion, when the trace is exactly 0);
   - `crates/geometry3/src/internal/geometry/unit_quaternion.cairo:203-209` (sign canonicalisation on
     `w`, `i`, `j`, `k`);
   - `crates/geometry2/src/geometry/unit_complex.cairo:755` cannot (zero is caught one line earlier):
     confirm it.
   **Decided (REL-a's analysis, docs/research/rel-zero.md, accepted by the project manager):** at
   `crates/geometry3/src/geometry/unit_quaternion.cairo:418` replace `R::is_sign_positive(tr)` by upstream's
   own comparison `tr > R::zero()`, so `from_rotation_matrix` at a trace of exactly 0 keeps nalgebra-rs's
   result `(w,i,j,k) = (-0.5, 0.5, 0.5, 0.5)` (e.g. the -120° cyclic permutation), and add a test pinning it.
   The other sites cannot see a zero (209 only on a non-unit input no caller passes): no change there.
   **nalgebra 0.2.0 must move no result**: every existing test, golden and oracle vector passes unedited. **Every existing test, golden and oracle
   vector that changes** is listed with old and new values and the reason; a change not explained by this
   is a stop: escalate. No other result may change.
3. **Version 0.2.0** for every published package of the workspace (MINOR: a result change at exact zeros,
   bit-identical optimisations, no result change; no public API change — `python3 scripts/api_parity.py --check` unchanged),
   and inter-crate dependency requirements updated to match. `CHANGELOG.md` 0.2.0 ("unreleased"): under
   **Changed**, simba 0.3.0 / fixed 0.5.0, and that no nalgebra result moves: simba's `is_sign_positive(0)`
   change is absorbed by keeping nalgebra-rs's `tr > 0` in `from_rotation_matrix` (cite
   `docs/research/rel-zero.md`); under **Performance**, the Cairo-steps tables of OPT-1 (#102) and OPT-2 (#103) from
   `docs/STEPS.md` (before 0.1.1 → after), results bit-identical; the steps probes (#101); the CI and
   pre-push tooling (#95-#99) briefly.
4. (done in REL-a, #105.)
5. (done in REL-a, #105.)

## 2. Allowlist

This brief verbatim as `docs/briefs/wp-12-rel-b.md` (first commit); manifests (`version`, simba/fixed and
inter-crate requirements only) and `Scarb.lock` (by scarb); `consumer_cost.toml` (versions only); `crates/geometry3/src/geometry/unit_quaternion.cairo` (line 418 and its pinning test);
`gas/**`, `steps/**` (regenerated); `CHANGELOG.md`; `docs/STEPS.md`,
`docs/PACKAGES.md` (regenerated from this PR's CI artifact if line counts or costs changed);
`docs/PLAN.md` (rows TC2/PUB/REL-FU2/REL
status only). Not: `.github/**` (if a setup-scarb step must change, move it to wretry and say so).

## 3. Verification and report

Push through the hook (it runs `scripts/prepush.sh`, capped); if the hook may run long, push with the
`ServerAliveInterval` form of the addendum. CI green. `python3 scripts/api_parity.py --check` unchanged.
PR body: the site-418 change and its pinning test (nalgebra-rs's value), the statement that no existing test,
golden or oracle vector changed (show `git diff --stat origin/main -- 'crates/tests_*'` and the goldens),
the version list (count of packages), the steps tables.
"Audit: none needed — release preparation; no result moves (checked against nalgebra-rs at the one changed site)"
(the orchestrator may change it). Conventional commits; all fixes of one review loop in one push; never
merge, never publish, never launch a review or any agent; foreground only; `REPORT.md` at the worktree root
(not committed), `## Summary` first. Prefer words over links to external issues.
Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call (if your thread cannot run it, name the job and run id in your report and end your turn: the orchestrator relays the log). Never `gh api …/logs`. To read a sibling repository (simba-cairo, fixed-cairo), use the Read tool on its checkout. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).
Picking up main: after any push (and until nexus #83 is installed), run `git merge origin/main`, alone in its call. `git rebase origin/main` in exactly that form is allowed only before your first push, once #83 is installed; every other rebase form and every force push stay refused.
Files: to remove an untracked file of your own worktree, use `git clean -f -- <exact path>` (nexus #92); a bare `rm` stays refused. NEVER run `git clean` with `-d`, `-x` or `-X`: it deletes your own .herdr-project folder (brief, report). `gh pr edit` works for a PR title and body (nexus #91).
Memory (organisation rule): the 8 GB cap (`prlimit --as=8589934592 -- /usr/bin/time -v <command>`) applies to Cairo builds and measures (scarb, snforge) on the VPS; every peak-memory measure stays capped; a Node suite, or a whole pre-push hook, runs uncapped only when each of its steps was measured well under 8 GB (`prlimit --as` kills Node at start-up). On the Mac there is no cap. Long pushes: if your pre-push hook may run long (several minutes), push with `git -c core.sshCommand='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=40' push ...` (writes no config), otherwise GitHub drops the SSH connection (exit 141) and nothing lands. Real peak memory: measure it uncapped only on the Mac (64 GB, no lock), never uncapped on the VPS; on the VPS, `prlimit --as=8589934592` (an address-space cap, which can kill a Cairo build at ~6.6 GB of real memory) stays the cap, used only for Cairo work known to fit well under it.

Work autonomously, do not ask questions, do not widen the scope.
