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
  when the probe was written). The exact integer models, on raw values (`value = raw / 2^32`, Python's `>>`
  floors): `scalar_mul_add` is `c + (a * b >> 32)`, `scalar_wide_dot3` is
  `(a1 * b1 + a2 * b2 + a3 * b3) >> 32`, and each component of `matrix6_mul_vec` is
  `sum(a[i][j] * x[j] for j in range(6)) >> 32` (inputs in `crates/probes_steps/src/scalar.cairo` and
  `matrix6.cairo`). A change of any last bit fails the probe, so an optimisation cannot
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

## Snapshot at this head

Measured with Scarb 2.20.1 (Cairo 2.20.0) and snforge 0.64.0 (`.tool-versions`), `RAYON_NUM_THREADS=1`, on
macOS arm64 (Apple silicon). `baseline` and `raw` are the steps of the two tests; `net steps` is their
difference, the cost of the operation. The probes were added by WP 11-OPT-0; the rows changed since are
listed under "Optimisation lots" below.

| probe | primitive | module | baseline | raw | net steps |
|---|---|---|---:|---:|---:|
| `vector3_dot` | `Vector3::dot` | `vector3` | 90 | 107 | **17** |
| `vector3_cross` | `Vector3::cross` | `vector3` | 100 | 147 | **47** |
| `vector3_norm` | `Vector3::norm` | `vector3` | 84 | 102 | **18** |
| `vector3_normalize` | `Vector3::normalize` | `vector3` | 94 | 194 | **100** |
| `matrix3_mul` | `Matrix3 * Matrix3` | `matrix3` | 154 | 315 | **161** |
| `matrix3_mul_vec` | `Matrix3::mul_mat(Vector3)` | `matrix3` | 112 | 165 | **53** |
| `matrix3_transpose` | `Matrix3::transpose` | `matrix3` | 136 | 145 | **9** |
| `matrix3_determinant` | `Matrix3::determinant` | `matrix3` | 96 | 161 | **65** |
| `matrix3_try_inverse` | `Matrix3::try_inverse` | `matrix3` | 136 | 548 | **412** |
| `unit_quaternion_mul` | `UnitQuaternion * UnitQuaternion` | `unit_quaternion` | 109 | 188 | **79** |
| `unit_quaternion_transform_vector` | `UnitQuaternion::transform_vector` | `unit_quaternion` | 102 | 269 | **167** |
| `unit_quaternion_from_axis_angle` | `UnitQuaternion::from_axis_angle` | `unit_quaternion` | 101 | 397 | **296** |
| `unit_quaternion_inverse` | `UnitQuaternion::inverse` | `unit_quaternion` | 101 | 111 | **10** |
| `unit_quaternion_slerp` | `UnitQuaternion::slerp` | `unit_quaternion` | 111 | 1018 | **907** |
| `isometry3_mul` | `Isometry3 * Isometry3` | `isometry3` | 136 | 389 | **253** |
| `isometry3_transform_point` | `Isometry3::transform_point` | `isometry3` | 108 | 281 | **173** |
| `isometry3_inverse` | `Isometry3::inverse` | `isometry3` | 122 | 299 | **177** |
| `isometry3_inv_mul` | `Isometry3::inv_mul` | `isometry3` | 136 | 401 | **265** |
| `cholesky3_factor` | `Matrix3::cholesky` (factor) | `cholesky` | 121 | 326 | **205** |
| `cholesky3_solve` | `Cholesky3::solve` | `cholesky` | 106 | 332 | **226** |
| `cholesky6_factor` | `Matrix6::cholesky` (factor) | `cholesky` | 275 | 1097 | **822** |
| `cholesky6_solve` | `Cholesky6::solve` | `cholesky` | 157 | 686 | **529** |
| `lu3_factor` | `Matrix3::lu` (factor) | `lu` | 146 | 432 | **286** |
| `lu3_solve` | `Lu3::solve` | `lu` | 116 | 289 | **173** |
| `lu6_factor` | `Matrix6::lu` (factor) | `lu` | 471 | 2328 | **1857** |
| `lu6_solve` | `Lu6::solve` | `lu` | 197 | 680 | **483** |
| `symmetric_eigen3` | `Matrix3::symmetric_eigen` | `symmetric_eigen3` | 151 | 3683 | **3532** |
| `svd3` | `Matrix3::svd(true, true)` | `svd3` | 233 | 4946 | **4713** |
| `matrix6_mul_vec` | `Matrix6::mul_mat(Vector6)` | `matrix6` | 187 | 330 | **143** |
| `scalar_mul_add` | `Real::mul_add` (scalar kernel) | `scalar` | 84 | 99 | **15** |
| `scalar_wide_dot3` | fused sum of 3 products (`wide_add_prod` x 3, `wide_rescale`) | `scalar` | 90 | 107 | **17** |

## Optimisation lots

Net steps before and after each lot, same toolchain, every result bit-identical (all tests, goldens and
probes unchanged, plus in-file equivalence sweeps against the previous bodies).

**WP 11-OPT-1** (geometry hot paths): `#[inline(always)]` on the Hamilton product, `conj_mul`, the
quaternion sandwiches, `rotate_translate`, `Isometry2/3` `mul` / `inv_mul` / `transform_point`,
`from_axis_angle` and `slerp`; the shortest-arc flip of `try_slerp` folded into the sign of one weight
(`(-o)·tb = o·(-tb)` exactly: one negation instead of four); the negation of the translation in
`Isometry2/3::inverse` folded into the exact products of the sandwich.

| probe | before | after | saved |
|---|---:|---:|---:|
| `unit_quaternion_mul` | 94 | 79 | 15 |
| `unit_quaternion_transform_vector` | 181 | 167 | 14 |
| `unit_quaternion_from_axis_angle` | 307 | 296 | 11 |
| `unit_quaternion_slerp` | 920 | 907 | 13 |
| `isometry3_mul` | 292 | 253 | 39 |
| `isometry3_transform_point` | 193 | 173 | 20 |
| `isometry3_inverse` | 200 | 177 | 23 |
| `isometry3_inv_mul` | 321 | 265 | 56 |
| `vector3_normalize` | 100 | 100 | 0 (already `norm3` + one shared divisor, `div3`: all its steps are in `fixed`) |

**WP 11-OPT-2** (small linear algebra): `#[inline(always)]` on the Jacobi steps of
`SymmetricEigen3` (the three rotations with their eigenvector update, `sweep`, `sorted`, `finish`:
each call passed the 15 components of the state both ways) and on `Svd3::new`, on `Cholesky3`'s
`new_sym` and `solve`, `Cholesky6::new` and `solve`, `Lu3::solve`, `Lu6::new` and `solve`;
`Cholesky6::new` shares one prepared divisor per column for columns 2 and 3 (`Real::div4`, `div3`,
bit-identical to per-element division, as column 1 already did with `div5`); the row swap of step 1 of
`Lu6::new` is a `match` on the pivot row instead of a chain of comparisons (constant cost; on steps 2
to 4 the chain stays, cheaper when the pivot is in one of the first rows, as on this probe's input).
No baseline changed.

| probe | before | after | saved |
|---|---:|---:|---:|
| `cholesky3_factor` | 220 | 205 | 15 |
| `cholesky3_solve` | 246 | 226 | 20 |
| `cholesky6_factor` | 897 | 822 | 75 |
| `cholesky6_solve` | 569 | 529 | 40 |
| `lu3_factor` | 286 | 286 | 0 (inlining `new` saved 2: not kept, for the code it adds at every call site) |
| `lu3_solve` | 198 | 173 | 25 |
| `lu6_factor` | 1919 | 1857 | 62 |
| `lu6_solve` | 542 | 483 | 59 |
| `symmetric_eigen3` | 4049 | 3532 | 517 |
| `svd3` | 5250 | 4713 | 537 |
| `matrix3_determinant`, `matrix3_try_inverse`, `matrix3_mul`, `matrix6_mul_vec` | 82, 430, 186, 192 | unchanged | 0 (generated by `tools/shapegen`, outside the lot) |

**WP 13-OPT-3** (the generated shapes, `tools/shapegen`): `#[inline(always)]` on `*` of `Matrix2` and
`Matrix3` (the call passed 18 components in and 9 out around 9 `sum_prod3`; `Matrix4` and `Matrix6`
keep the call), on every matrix-vector product whose output is a column vector (before, only up to 4
components: `Matrix6 * Vector6` and the products into `Vector5` / `Vector6` were calls), and on
`Matrix3::determinant` and `try_inverse`. `try_inverse` also does less on the branches of a small
determinant: `k = floor(2 / f) >= 2` exactly when `f <= 1` (the division rounds to nearest), so
`2 / f` is computed on the pre-scaled branch only, and the six cofactors its determinant does not use
are computed after that branch, which never needs them (components at most 1 in magnitude: they
could not overflow there, so no panic is lost; before the singularity test, as before). Its previous
body is kept as `try_inverse_reference` in a generated test module of
`crates/nalgebra/src/base/matrix3.cairo`, compared bit for bit on 221 matrices (107 with
`|det| >= 1/2`, 58 pre-scaled, 56 with a small determinant and `f > 1`) and on a cofactor overflow
(both panic with `'Fixed: overflow'`). No baseline changed; no other probe moved.

| probe | before | after | saved |
|---|---:|---:|---:|
| `matrix3_mul` | 186 | 161 | 25 |
| `matrix3_determinant` | 82 | 65 | 17 |
| `matrix3_try_inverse` | 430 | 412 | 18 |
| `matrix6_mul_vec` | 192 | 143 | 49 |

The probe of `try_inverse` takes the branch `|det| >= 1/2` (its matrix has `f` about 1.87), where
only the inlining acts. The two other branches were measured the same way on an uncommitted
package (raw steps of one call, the previous body called as before against the new one inlined,
same harness): pre-scaled (`f` about 0.62) 804 -> 703 (-101), small determinant with `f > 1`
(det about 0.02) 593 -> 551 (-42), and this probe's matrix 519 -> 502 (-17). A probe per branch
would pin them (probes are hand-written, outside that lot).

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
