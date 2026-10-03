# Cairo steps: probes and snapshot (WP 11-OPT-0)

The Cairo primitives of the library are optimised in **Cairo steps, measured on probes, with
bit-identical results** (programme scope, owner, 2026-10-03). This document is the yardstick: one
probe per hot primitive, a snapshot of their net steps, and the check that keeps the snapshot honest.
It optimises nothing: the figures below are the baseline an optimisation lot must beat, with the same
results to the last bit.

Sierra gas (`gas/`, `docs/BENCHMARK.md`) stays the unit of the design benchmarks; steps are what a
prover pays. Both are deterministic, neither replaces the other.

## What a probe is

The probes live in the test-only package `nalgebra_probes_steps` (`crates/probes_steps`,
`publish = false`: no release, consumer-cost, package-table or API-parity script sees it; it depends
only on the published sub-crates it measures, not on the facade). A probe is a pair of tests
`probe_<group>__baseline` / `probe_<group>__op` over the same fixed inputs and the same exact
assertion:

- `op` runs the primitive once and asserts its **exact result** (bit for bit). The expected values are
  those of the gas benches (themselves checked against upstream nalgebra) where the primitive has one,
  an exact integer model for the scalar kernels and `Matrix6::mul_mat`, and, for the decompositions,
  the kernel's own result on an oracle input (agreeing with the oracle within its tolerance, checked
  when the probe was written). A change of any last bit fails the probe, so an optimisation cannot
  silently change a result.
- `baseline` builds the same inputs and runs the same assertion on the expected value against itself
  (`assert!(e == e)`): the net steps of the operation are `op - baseline`, as `bench_<group>__baseline`
  does for gas. Inputs go through `nalgebra_testing::black_box`, so the compiler cannot fold them.

Probes (one module per type, `crates/probes_steps/src/<module>.cairo`): Vector3 (`dot`, `cross`,
`norm`, `normalize`), Matrix3 (`mul`, `mul_vec`, `transpose`, `determinant`, `try_inverse`),
UnitQuaternion (`mul`, `transform_vector`, `from_axis_angle`, `inverse`, `slerp`), Isometry3 (`mul`,
`transform_point`, `inverse`, `inv_mul`), Cholesky 3 and 6 and LU 3 and 6 (factor and solve each),
symmetric eigen 3, SVD 3, Matrix6 `mul_vec`, and two scalar kernels.

**Added to the brief's list**: `Real::mul_add` and the fused sum of products (`wide_add_prod` then
`wide_rescale`). Counted over the sources of the decomposition packages (`linalg_core`, `linalg2..6`,
`linalg_pivot2..6`, `linalg_spectral2..6`, `linalg_svd_eigen2..6`), they are called far more often
than any shape operation of the list: about 3,900 `mul_add`, 4,400 `wide_add_prod` / `wide_sub_prod`
and 900 `wide_rescale` calls (against 244 `mul_mat`, 5 `transpose`, 3 `cross`, none of `dot` / `norm`; matching lines).
Their code lives in `simba` / `fixed` (a probe measures it as the library calls it).

## Running and reading the snapshot

```bash
snforge test -p nalgebra_probes_steps --tracked-resource cairo-steps --detailed-resources \
    | python3 scripts/steps_report.py            # print the report
... | python3 scripts/steps_report.py --update steps/   # rewrite the snapshot
... | python3 scripts/steps_report.py --check steps/    # what CI runs: exit 1 on any difference
python3 scripts/steps_report.py --self-test               # checks of the script itself
```

`--detailed-resources` is what makes snforge print `steps: N` under each `[PASS]` line;
`--tracked-resource cairo-steps` makes the run count steps rather than Sierra gas. Run it with
`RAYON_NUM_THREADS=1` like every snapshot (the compiler is not deterministic on several threads).

The snapshot is `steps/nalgebra_probes_steps.json` (`{module: {group: {variant: steps}}}`, raw steps of
every test) and `steps/nalgebra_probes_steps.md` (the same, with the baseline subtracted: `net steps`).
The check reads the `.json` only and compares every figure exactly, in both directions (a probe added
or removed is a difference). Cairo steps are path- and platform-free: the snapshot may be generated on
any machine (this lot's was generated on a Mac), and the CI check is the gate. The CI job `Steps snapshot`
runs the script's self-test, then the probes and the check, and is part of `CI result`.

A step count is a function of the Sierra the compiler emits for the test: it moves when the primitive
changes, but also when a probe's inputs, its assertion or the toolchain change. An optimisation lot
therefore reads `net steps` before and after on the same toolchain, regenerates the snapshot in its
own pull request and explains each moved figure there, as for `gas/`.

## Adding a probe

In the module of its type (or a new `mod` line in `crates/probes_steps/src/lib.cairo`), write the
inputs and the expected result as small functions and the pair of tests:

```cairo
#[test]
#[inline(never)]
fn probe_vector3_dot__baseline() {
    let _i = black_box((a(), b()));
    let e = black_box(fx(0x30000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_vector3_dot__op() {
    let (x, y) = black_box((a(), b()));
    let e = black_box(fx(0x30000000));
    assert!(x.dot(y) == e);
}
```

The names are the contract (`probe_<group>__baseline` and `probe_<group>__op`; the group may hold
underscores, the variant follows the last `__`); every test is `#[inline(never)]` (snforge otherwise
inlines some test bodies and not others, which shifts `raw - baseline`); the baseline holds every
input and the same assertion on the expected value. A result without `PartialEq` (a decomposition)
gets a field-wise impl in `crates/probes_steps/src/eq.cairo`. Then regenerate the snapshot with
`--update steps/` and add a row to the table below.

## Baseline at this head

Measured with Scarb 2.20.1 (Cairo 2.20.0) and snforge 0.64.0 (`.tool-versions`), `RAYON_NUM_THREADS=1`, on
macOS arm64 (Apple silicon), at the head of the pull request that adds the probes. `baseline` and `raw`
are the steps of the two tests; `net steps` is their difference, the cost of the operation.

| probe | primitive | module | baseline | raw | net steps |
|---|---|---|---:|---:|---:|
| `vector3_dot` | `Vector3::dot` | `vector3` | 90 | 107 | **17** |
| `vector3_cross` | `Vector3::cross` | `vector3` | 100 | 147 | **47** |
| `vector3_norm` | `Vector3::norm` | `vector3` | 84 | 102 | **18** |
| `vector3_normalize` | `Vector3::normalize` | `vector3` | 94 | 194 | **100** |
| `matrix3_mul` | `Matrix3 * Matrix3` | `matrix3` | 154 | 340 | **186** |
| `matrix3_mul_vec` | `Matrix3::mul_mat(Vector3)` | `matrix3` | 112 | 165 | **53** |
| `matrix3_transpose` | `Matrix3::transpose` | `matrix3` | 136 | 145 | **9** |
| `matrix3_determinant` | `Matrix3::determinant` | `matrix3` | 96 | 178 | **82** |
| `matrix3_try_inverse` | `Matrix3::try_inverse` | `matrix3` | 136 | 566 | **430** |
| `unit_quaternion_mul` | `UnitQuaternion * UnitQuaternion` | `unit_quaternion` | 109 | 203 | **94** |
| `unit_quaternion_transform_vector` | `UnitQuaternion::transform_vector` | `unit_quaternion` | 102 | 283 | **181** |
| `unit_quaternion_from_axis_angle` | `UnitQuaternion::from_axis_angle` | `unit_quaternion` | 101 | 408 | **307** |
| `unit_quaternion_inverse` | `UnitQuaternion::inverse` | `unit_quaternion` | 101 | 111 | **10** |
| `unit_quaternion_slerp` | `UnitQuaternion::slerp` | `unit_quaternion` | 111 | 1031 | **920** |
| `isometry3_mul` | `Isometry3 * Isometry3` | `isometry3` | 136 | 428 | **292** |
| `isometry3_transform_point` | `Isometry3::transform_point` | `isometry3` | 108 | 301 | **193** |
| `isometry3_inverse` | `Isometry3::inverse` | `isometry3` | 122 | 322 | **200** |
| `isometry3_inv_mul` | `Isometry3::inv_mul` | `isometry3` | 136 | 457 | **321** |
| `cholesky3_factor` | `Matrix3::cholesky` (factor) | `cholesky` | 121 | 341 | **220** |
| `cholesky3_solve` | `Cholesky3::solve` | `cholesky` | 106 | 352 | **246** |
| `cholesky6_factor` | `Matrix6::cholesky` (factor) | `cholesky` | 275 | 1172 | **897** |
| `cholesky6_solve` | `Cholesky6::solve` | `cholesky` | 157 | 726 | **569** |
| `lu3_factor` | `Matrix3::lu` (factor) | `lu` | 146 | 432 | **286** |
| `lu3_solve` | `Lu3::solve` | `lu` | 116 | 314 | **198** |
| `lu6_factor` | `Matrix6::lu` (factor) | `lu` | 471 | 2390 | **1919** |
| `lu6_solve` | `Lu6::solve` | `lu` | 197 | 739 | **542** |
| `symmetric_eigen3` | `Matrix3::symmetric_eigen` | `symmetric_eigen3` | 151 | 4200 | **4049** |
| `svd3` | `Matrix3::svd(true, true)` | `svd3` | 233 | 5483 | **5250** |
| `matrix6_mul_vec` | `Matrix6::mul_mat(Vector6)` | `matrix6` | 187 | 379 | **192** |
| `scalar_mul_add` | `Real::mul_add` (scalar kernel) | `scalar` | 84 | 99 | **15** |
| `scalar_wide_dot3` | fused sum of 3 products (`wide_add_prod` x 3, `wide_rescale`) | `scalar` | 90 | 107 | **17** |

## Limits

- A probe runs the primitive once on one input: the branches the input does not take are not measured
  (`try_inverse` takes its prescaled path, `slerp` the short arc, the factorisations the pivots of their
  input). A lot that optimises a branch adds a probe for that input.
- Steps are counted by the VM of snforge (`steps: N`); builtins (range check, bitwise, ...) and memory
  holes are in the raw output but not in the snapshot: a lot that trades steps for builtins must say
  so with the output of `--detailed-resources`.
- The decomposition probes pin the kernels' results at this head, not upstream's: the oracle tolerance
  is the correctness bound (`nalgebra_tests_linalg`), the probe's exact assertion is the
  bit-identity bound.
