# WP 11-OPT-1 — fewer Cairo steps on the geometry hot paths, results bit-identical

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-03.
Programme scope (owner, 2026-10-03): the Cairo primitives and their optimisation, measured in **Cairo
steps on the probes of `crates/probes_steps` (docs/STEPS.md), with bit-identical results**. Runs on the
Mac (`/Users/bal7hazar/git/nalgebra-cairo`); Cairo steps and gas are path- and platform-free, so
snapshots may be regenerated there; CI is the gate.

## 1. Goal

Reduce the net steps of these probes (baseline at main 791cd4d, from docs/STEPS.md):
`vector3_normalize` 100, `unit_quaternion_mul` 94, `unit_quaternion_transform_vector` 181,
`unit_quaternion_from_axis_angle` 307, `unit_quaternion_slerp` 920, `isometry3_mul` 292,
`isometry3_transform_point` 193, `isometry3_inverse` 200, `isometry3_inv_mul` 321 — and whatever they
share (the Vector3/Matrix3/Rotation3 code paths underneath). Rotation2/UnitComplex/Isometry2 are in scope
when the same change applies to them.

**Bit-identical is the hard rule.** Every result of every public function must stay exactly the same
bits for every input. So: removing redundant work, avoiding re-normalisation that is mathematically a
no-op only when it is also bit-exact, reusing computed sub-terms, cheaper control flow, avoiding
`Option`/`Result` round-trips, inlining (`#[inline(always)]` on methods of impls), better use of
`fixed`'s kernels **only where the kernel computes the same bits** as the code it replaces. A change
that would alter rounding (for example a fused `sum_prod`/`mul_add` replacing separate roundings) is a
numeric change: do NOT make it; list it in the report as a proposal with its expected steps saved and
the size of the result change, for the orchestrator to decide. A fused kernel that `fixed` lacks is
never written here: list it under Escalations; the orchestrator requests it from the fixed track.

## 2. How results are shown unchanged

- Every existing test, golden, oracle vector and gas/steps probe assertion passes **with no test edited**
  (`git diff --stat origin/main -- 'crates/tests_*' 'crates/probes_steps/src'` shows no assertion change).
- For each optimised function, an equivalence check against the previous implementation: keep the old
  body as a private `*_reference` function in a test module (under `#[cfg(test)] mod tests` in the file,
  owner's unit-test placement rule), and a test that compares old and new outputs bit for bit on a
  deterministic sweep of inputs (edge cases: zero, identity, near-180° rotations, unnormalised inputs
  where the API accepts them, extreme magnitudes of Q32.32) — at least 200 inputs per function. Remove
  nothing from the public API.

## 3. Snapshots

Regenerate `steps/` and the affected `gas/*.json|.md` with their scripts (`RAYON_NUM_THREADS=1`), and
report the before/after net steps per probe and the gas deltas (count, largest). The PR body has the
full steps table. A probe whose steps went up is a regression unless explained.

Carried notes from OPT-0's review: five probes (`matrix3_try_inverse`, `cholesky3/6`, `lu3/6` solve)
count an `Option` unwrap that their baseline does not build; if you touch a baseline, make it unwrap a
black-boxed `Option::Some(e)` too (a baseline change shifts net steps: report it separately, not as a
saving). The scalar/matrix6 expected values come from an exact-integer model: give its one-line
computation in docs/STEPS.md if you edit that file.

## 4. Allowlist

This brief verbatim as `docs/briefs/wp-11-opt1.md` (first commit); the source files of the geometry and
vector paths you optimise (`crates/**/src/**` of the crates that hold them; each listed in the report),
their in-file `#[cfg(test)] mod tests`; `steps/**`, `gas/**` (regenerated); `docs/STEPS.md` (table and
notes); `crates/probes_steps/src/**` only to fix the baselines named above. Not: `.github/**`, scripts,
manifests, the public API (no item added, removed or renamed: `python3 scripts/api_parity.py --check`
passes unchanged).

## 5. Verification and report

`scripts/prepush.sh` before each push (the Mac clone has no hook configured; no lock on the Mac). Every
CI check green. PR body: the steps table before/after, the equivalence-check counts per function, the
gas summary, the proposals that would change results (with expected savings), Escalations (kernels for
the fixed track). Conventional commits; all fixes of one review loop in one push; never merge, never
launch a review or any agent; foreground only; `REPORT.md` at the worktree root (not committed),
`## Summary` first; "Audit: none needed — results shown bit-identical by tests, goldens and equivalence
sweeps" (the orchestrator may change it). Prefer words over links to external issues.

Reading CI logs (programme rule): wait for the run to complete, then `gh run view <id> -R <owner>/<repo> --log-failed`, alone in its call. Never `gh api .../actions/jobs/<id>/logs` (refused to threads); if you need a job log earlier, say so in your report and the orchestrator relays it. A bare command refused: report its exact text, no other attempt. At most one GitHub call per PR every 5 minutes. Never probe the shared heavy-build lock. Mac repositories by absolute path (/Users/bal7hazar/git/<repo>).
Signals (programme rule after an incident): a thread, a test or a script signals only processes it started itself, by pid or pgid taken from `$!` or ids it recorded. Never find a pid by searching (`ps | grep`, `pgrep`, `pkill`, `killall`).

Work autonomously, do not ask questions, do not widen the scope.

