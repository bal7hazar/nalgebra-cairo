# WP 12-REL-a — release tooling and the exact-zero analysis (no manifest change)

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Runs on the **VPS**, crate-scoped only: never build or test the whole workspace locally (it peaks near
11 GB). Every `scarb`/`snforge` command is crate-scoped (`-p <crate>`) and run under
`prlimit --as=8589934592 -- …`; the facade package `nalgebra` is never built locally (CI builds it).
**Change no manifest and no `Scarb.lock`** in this lot: the repository's pre-push hook treats a manifest
change as "everything" and would build the whole workspace on the VPS. The manifest part of REL (simba
0.3.0, version 0.2.0, the tests at exact zero that need simba 0.3.0) is lot REL-b, later, on the Mac.

## Scope of this lot (items of the REL plan below that need no manifest change)

- Item 4 (release tooling: `scripts/release.py` request and verify modes replacing the publish mode, with
  self-tests; the dry request run only on 2-3 small packages, never the facade).
- Item 5 (the slerp doc line; the garbled test doc line).
- Item 2 as **analysis only**: a report section (in the PR body and `docs/research/rel-zero.md`) giving, for
  each of the 6 `R::is_sign_positive` sites, what nalgebra-rs 0.35.0 does at the exact-zero input, what
  nalgebra-cairo returns today (simba 0.2.0) and will return with simba 0.3.0, and the test to add in REL-b.
  No code change at those sites in this lot.
- Not in this lot: items 1 and 3 (manifests, versions, CHANGELOG 0.2.0) — REL-b.

Allowlist for this lot: the brief verbatim as `docs/briefs/wp-12-rel-a.md` (first commit);
`scripts/release.py`; `docs/research/rel-zero.md`; the two doc lines' files
(`crates/geometry3/src/geometry/unit_quaternion.cairo` — doc only, `crates/linalg6/src/linalg/cholesky.cairo` —
test doc only); `docs/PLAN.md` (rows PUB / REL-FU2 / REL status only).

The full REL plan follows for context.

## REL plan (context): nalgebra 0.2.0: simba 0.3.0, the optimisations, the release tooling

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Heavy lot (whole-workspace builds and tests): runs on the Mac (`/Users/bal7hazar/git/nalgebra-cairo`).
Gas and steps snapshots are path- and platform-free and may be regenerated there; CI is the gate.
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
   For each site: what nalgebra-rs 0.35.0 does at the same input (read its source; `tools/oracle` if it
   helps), whether nalgebra-cairo's new result matches nalgebra-rs (expected: yes, closer than before),
   and a test at the exact-zero input pinning the new result. **Every existing test, golden and oracle
   vector that changes** is listed with old and new values and the reason; a change not explained by this
   is a stop: escalate. No other result may change.
3. **Version 0.2.0** for every published package of the workspace (MINOR: a result change at exact zeros,
   bit-identical optimisations; no public API change — `python3 scripts/api_parity.py --check` unchanged),
   and inter-crate dependency requirements updated to match. `CHANGELOG.md` 0.2.0 ("unreleased"): under
   **Changed**, the result change (the sites, the inputs, the nalgebra-rs alignment) and simba 0.3.0 /
   fixed 0.5.0; under **Performance**, the Cairo-steps tables of OPT-1 (#102) and OPT-2 (#103) from
   `docs/STEPS.md` (before 0.1.1 → after), results bit-identical; the steps probes (#101); the CI and
   pre-push tooling (#95-#99) briefly.
4. **Release tooling** (`scripts/release.py`, the PUB and REL-FU2 rows of `docs/PLAN.md`): publishing is
   now done by hand by the orchestrator, one `scarb publish -p <package>` per package, on the project
   manager's go naming package, version, commit and archive sha256 (several packages may share one
   request and one go). Replace the publishing mode with a **request mode**: from a checkout detached at a
   given commit, build every published package's archive (`scarb package -p <pkg>`, through the shims),
   compute sha256s, and write `docs/releases/<version>.md` (a table: package, version, commit, sha256,
   archive size compressed/unpacked, in dependency publication order) plus the ordered list of
   `scarb publish -p <pkg>` commands; and a **verify mode** that reads the registry index back and compares
   each package's checksum with the request. It never runs `scarb publish` itself. Self-tests for both.
   Fold the REL-FU2 docstring notes (45 vs 46 checks; `--verify-timeout` is not a wall-clock cap) if the
   code they describe survives.
5. **Small debt folded in**: the `slerp` doc line (a `MIN` component of `other` no longer panics, review
   of #102); the garbled doc line of `test_new_with_substitute_matches_reference` (review of #103).

## 2. Allowlist

This brief verbatim as `docs/briefs/wp-12-rel.md` (first commit); manifests (`version`, simba/fixed and
inter-crate requirements only) and `Scarb.lock` (by scarb); `consumer_cost.toml` (versions only); the 5
call sites' files (tests at the zero input; code only if needed to match nalgebra-rs — say why);
`gas/**`, `steps/**` (regenerated); `CHANGELOG.md`; `scripts/release.py`; `docs/STEPS.md`,
`docs/PACKAGES.md` (regenerated from this PR's CI artifact if line counts or costs changed);
`crates/geometry3/src/geometry/unit_quaternion.cairo` (the slerp doc line);
`crates/linalg6/src/linalg/cholesky.cairo` (the test doc line); `docs/PLAN.md` (rows TC2/PUB/REL-FU2/REL
status only). Not: `.github/**` (if a setup-scarb step must change, move it to wretry and say so).

## 3. Verification and report

`scripts/prepush.sh` before each push (the Mac clone has no hook configured; no lock on the Mac). CI green.
`python3 scripts/release.py --self-test` and a dry request run on this branch (not committed as a release
file). PR body: the result-change analysis per site (nalgebra-rs's behaviour, the new test), every changed
test/golden value, the version list (count of packages), the steps tables, the release tooling's new modes.
"Audit: none needed — release preparation; results checked against nalgebra-rs at the changed sites"
(the orchestrator may change it). Conventional commits; all fixes of one review loop in one push; never
merge, never publish, never launch a review or any agent; foreground only; `REPORT.md` at the worktree root
(not committed), `## Summary` first. Prefer words over links to external issues.
Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call (if your thread cannot run it, name the job and run id in your report and end your turn: the orchestrator relays the log). Never `gh api …/logs`. To read a sibling repository (simba-cairo, fixed-cairo), use the Read tool on its checkout. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).
Picking up main: after any push (and until nexus #83 is installed), run `git merge origin/main`, alone in its call. `git rebase origin/main` in exactly that form is allowed only before your first push, once #83 is installed; every other rebase form and every force push stay refused.
Files: to remove an untracked file of your own worktree, use `git clean -f -- <exact path>` (nexus #92); a bare `rm` stays refused. NEVER run `git clean` with `-d`, `-x` or `-X`: it deletes your own .herdr-project folder (brief, report). `gh pr edit` works for a PR title and body (nexus #91).
Memory (organisation rule): the 8 GB cap (`prlimit --as=8589934592 -- /usr/bin/time -v <command>`) applies to Cairo builds and measures (scarb, snforge) on the VPS; every peak-memory measure stays capped; a Node suite, or a whole pre-push hook, runs uncapped only when each of its steps was measured well under 8 GB (`prlimit --as` kills Node at start-up). On the Mac there is no cap. Long pushes: if your pre-push hook may run long (several minutes), push with `git -c core.sshCommand='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=40' push ...` (writes no config), otherwise GitHub drops the SSH connection (exit 141) and nothing lands. Real peak memory: measure it uncapped only on the Mac (64 GB, no lock), never uncapped on the VPS; on the VPS, `prlimit --as=8589934592` (an address-space cap, which can kill a Cairo build at ~6.6 GB of real memory) stays the cap, used only for Cairo work known to fit well under it.

Work autonomously, do not ask questions, do not widen the scope.
