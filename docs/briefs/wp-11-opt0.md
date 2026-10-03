# WP 11-OPT-0 — Cairo-steps probes and baseline for the hot primitives

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Programme scope (owner, 2026-10-03): the Cairo primitives — their optimisation is measured in **Cairo
steps on probes, with bit-identical results**. This lot builds the yardstick; it optimises nothing.
Runs on the Mac (`/Users/bal7hazar/git/nalgebra-cairo`): builds and tests there. Cairo steps are path-
and platform-free, so the steps snapshot may be generated on the Mac; the CI check is the gate.

## 1. Goal

1. A probe set, one probe per hot primitive, each a `#[test]` that runs the operation once on fixed
   inputs (through `crate::testing::black_box` or the repository's equivalent, as the gas benches do)
   and asserts its exact result (bit-identical, against an existing golden where one exists):
   - Vector3: dot, cross, norm, normalize; Matrix3: mul, mul_vec, transpose, determinant, try_inverse;
   - UnitQuaternion: mul (compose), transform_vector, from_axis_angle, inverse, slerp;
   - Isometry3: mul (compose), transform_point, inverse, inv_mul;
   - small linear algebra: Cholesky 3 and 6 (factor + solve), LU 3 and 6 (factor + solve),
     symmetric eigen 3, SVD 3; Matrix6 mul_vec.
   Each probe also has a `baseline` (inputs + assertion only) so the net steps of the operation are
   reported, as `bench_<group>__baseline` does for gas.
   The set is the orchestrator's choice: the primitives most used by nalgebra's own decomposition
   packages and consumers (rapier-cairo does not depend on nalgebra). If, while writing it, you find a
   primitive that the decomposition packages call far more often than one listed (read their sources),
   add it and say so in the PR body. Structure the probes so adding one is a few lines.
2. Where the probes live: a new test-only workspace member (e.g. `crates/probes_steps`, name it per
   the repository's crate-name rules in `docs/SPLIT.md`), `publish = false`, excluded from every
   release / consumer-cost / package-table / API-parity script that iterates members (check each:
   `scripts/release.py`, `scripts/consumer_cost.py`, `scripts/packages_table.py`,
   `scripts/api_parity.py`, `scripts/facade_features.py`, `tools/split/*`), depending only on the
   published sub-crates it measures.
3. A steps report and snapshot: `snforge test -p <probes> --tracked-resource cairo-steps` output turned
   into `steps/<probes>.json` + `.md` (net steps per probe, baseline subtracted), with `--update` /
   `--check` like `scripts/gas_report.py` (extend that script with a steps mode, or a sibling
   `scripts/steps_report.py`; reuse its conventions; add self-test cases).
4. CI (`.github/workflows/ci.yml`, **the one workflow file this lot changes**, edited with the file-editing
   tool only): a new job `Steps snapshot` that builds the probes and runs the `--check`; its own group in
   the `changes` job (triggers: the toolchain set, `crates/**` except prose `.md`, `steps/**`, the steps
   script), added to `CI result`'s `needs` and its group table; `RAYON_NUM_THREADS: "1"` like the other
   test jobs; setup steps written like the existing ones (the setup-snfoundry retry form of nexus #69:
   first attempt with an id and `continue-on-error: true`, last attempt the same action without it and
   `if: steps.<id>.outcome == 'failure'`; setup-scarb as **wretry**, the form the classifier accepts for
   scarb — read how glam-cairo or rapier-cairo write it if needed; no other `continue-on-error`).
5. `docs/STEPS.md`: how to read the snapshot, how to add a probe, the baseline table (net steps per probe
   at this head), the toolchain (Scarb 2.20.1 / snforge 0.64.0).

## 2. Allowlist

This brief verbatim as `docs/briefs/wp-11-opt0.md` (first commit); the new probes crate; root
`Scarb.toml` (members only) and `Scarb.lock` (by scarb); the steps script (or `scripts/gas_report.py`'s
steps mode) and its self-test; `steps/**`; `.github/workflows/ci.yml` (the new job, its `changes` group,
`CI result`); `docs/STEPS.md`; the member-iterating scripts above **only** to exclude the probes crate.
No library source changes (no `crates/<library>/src/**`).

## 3. Acceptance

1. Every probe asserts its exact result; all pass. 2. The steps snapshot is generated and `--check`
passes locally and in CI. 3. Exclusions shown: `scripts/release.py` dry run, `consumer_cost.py
--dry-run --report-only`, `api_parity.py --check`, `packages_table.py --self-test` unchanged in output
or with the probes crate absent. 4. The PR body lists every probe with its net steps, and the machine
and toolchain they were measured on. 5. CI green, with the new job running on this PR.

## 4. Rules

- The Mac clone has no pre-push hook configured: run `scripts/prepush.sh` yourself before each push
  (on the Mac there is no heavy lock; the compile always runs). Never set `git config` anywhere.
- Unit-test placement (owner, 2026-09-30): a module's unit tests go in its file under
  `#[cfg(test)] mod tests`.
- Conventional commits; all fixes of one review loop in one push; `gh pr create` per the template
  (body: "Audit: none needed — measurement tooling, no library change"). Never merge, never launch a
  review or any agent. Foreground only (long commands in the foreground with long timeouts).
  `REPORT.md` at the worktree root (not committed), `## Summary` first. Prefer words over links to
  external issues in PR text and commit messages.

Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call. Never `gh api .../actions/jobs/<id>/logs` (refused to threads); if you need a job log earlier, say so in your report and the orchestrator relays it. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).

Work autonomously, do not ask questions, do not widen the scope.

