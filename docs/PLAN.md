# Execution plan

Execution model: one **orchestrator** session owns the plan, the workspace manifests, CI, the gas
snapshot and the merges. Work packages are delegated to **sub-agents running in parallel**, each
owning a disjoint set of files (one module directory per agent) and delivering through a PR that
must be green (fmt, build, tests, gas snapshot). Packages inside a milestone are independent
unless a dependency is stated; milestones are sequential only where an arrow says so.

```
M0 ─► M1 ─► M2 ─┬─► M3 ─┬─► M6
                └─► M4 ─┴─► M5
```

Priority inside each milestone follows what a game physics engine actually calls
([research/01](research/01-nalgebra-analysis.md) §3): vectors → unit quaternion / complex →
isometries → 2x2/3x3 matrices → scalar helpers → symmetric eigen → 6D → small factorizations →
dynamic → SVD.

## M0 — Foundations ✅

Research reports, four benchmark suites, design decisions, CI, agent conventions.

## M1 — `simba`: the scalar ✅

| WP | Content | Depends on |
|---|---|---|
| 1.1 | `Fixed` Q32.32: storage, constants, conversions, add/sub/neg/cmp, mul/div/rem, abs/signum/min/max/clamp/floor/ceil/round/recip, fused kernels (`sum_prod2/3/4`, `diff_prod`, `mul_add`, `lerp`, `norm2/3/4`), wide accumulator, `Real` trait, approx-eq in ulp, Python bit-exact model | — |
| 1.2 | Transcendentals: sqrt, inv_sqrt, sin/cos/sin_cos/tan, atan2/atan, acos/asin, exp/ln/pow (low priority) + `tools/` polynomial generator | 1.1 (API only) |
| 1.3 | `tools/oracle`: Rust generator of golden vectors from upstream nalgebra (f64, inputs quantised to Q32.32) | — |

## M2 — `nalgebra::base`: static vectors and matrices ✅

| WP | Content | Depends on |
|---|---|---|
| 2.1 | `Vector2/3/4`: constructors, ops, dot, cross, perp, norms, normalize, lerp, component-wise, min/max, angle, orthonormal basis | M1 |
| 2.2 | `Matrix2/3/4`: constructors (row-major `new`, from columns/rows/diagonal), ops, `mul_vec`, transpose, trace, determinant, `try_inverse`, `cross_matrix`, outer product | 2.1 |
| 2.3 | `SymMatrix2/3`, structured kernels (`quadform`, `mul_transpose`, sym inverse) | 2.1 |
| 2.4 | `Point2/3`, `Unit<V>` | 2.1 |
| 2.5 | `Vector6`, `Matrix6` (blocks of 3), rectangular `Matrix3x2`-style blocks only where M4/M5 need them | 2.2 |

## M3 — `nalgebra::geometry` ✅ (Similarity included; `Scale`, `Reflection`, `Transform` family deferred)

| WP | Content | Depends on |
|---|---|---|
| 3.1 | `UnitComplex`, `Rotation2` | 2.2, 2.4 |
| 3.2 | `Quaternion`, `UnitQuaternion` (axis-angle, `from_rotation_matrix`, `append_axisangle_linearized`, `renormalize_fast`, nlerp/slerp, rotation between vectors), `Rotation3` (euler angles, look_at) | 2.2, 2.4 |
| 3.3 | `Translation2/3`, `Isometry2/3` (`inv_mul`, `transform_point/vector`, inverse forms, lerp_slerp) | 3.1, 3.2 |
| 3.4 | `Similarity2/3` (in progress); `Scale`, `Reflection`, `Transform`/`Projective`/`Perspective`/`Orthographic` are deferred (rendering-oriented, unused by the physics stack) | 3.3 |

## M4 — `nalgebra::linalg`: small static decompositions ✅

| WP | Content | Depends on |
|---|---|---|
| 4.1 | `Cholesky` and `LDLᵀ` (upstream `UDU`) for 2/3/4/6, `solve`, `inverse` | 2.2, 2.5 |
| 4.2 | `LU` with partial pivoting for 2/3/4/6, `solve`, `determinant` | 2.2, 2.5 |
| 4.3 | `SymmetricEigen` 2x2 (closed form) and 3x3 (fixed-sweep Jacobi) | 2.3 |
| 4.4 | `SVD` 2x2/3x3, polar decomposition, `pseudo_inverse`; `QR` 2/3/4 | 4.3 |

## M5 — Dynamic algebra (scoped by multibody needs) — deferred

| WP | Content | Depends on |
|---|---|---|
| 5.1 | `DVector`, `DMatrix`: construction, element-wise ops, `gemv`, `gemm`, `axpy`, `tr_mul`, `quadform`, static-kernel dispatch | M2 |
| 5.2 | Dynamic `LU` / `Cholesky` solve | 5.1, M4 |

Gate: rapier.cairo v1 explicitly cuts multibody joints, IK and soft bodies (confirmed 2026-09-20),
so M5 stays deferred until a consumer exists.

## M4b — Consolidation

| WP | Content | Depends on |
|---|---|---|
| 4.5 | Promote `base::matrix_test_utils` to `pub(crate)` and delete the duplicated builders in `linalg`/`geometry`; fused `conj_mul` quaternion kernel (saves the 3 negations of `Isometry3::inv_mul`); `Wide × Fixed` accumulator op in `simba` for exact triple products (4x4 / 6x6 determinants); `Real::div` / `Real::rem` so that `normalize`/`unscale`/`new_normalize` no longer depend on the foreign scalar's `/` semantics (closes the one `simba_fixed` conformance gap); add `crates/simba_fixed/tools/gen_vectors.py --check` to the gate | M3, M4, 6.1 |

## M6 — Interop and release

| WP | Content | Depends on |
|---|---|---|
| 6.1 ✅ | `simba_fixed`: `Real` + `Transcendental` for glam.cairo's shared `fixed::Fixed` (git-pinned), bit-for-bit conformance suite between the two scalars, integration tests on `Vector3`/`SymMatrix3`/`UnitQuaternion`/`Isometry3<fixed::Fixed>` | M3, M4 |
| 6.2 | Conversions with glam.cairo types, `scarb doc`, publication on scarbs.xyz | 6.1 |

Conversions with glam.cairo, conformance suite against the oracle, `scarb doc`, publication of
`simba` and `nalgebra` on scarbs.xyz (tag-driven), upgrade policy (toolchain bumps are separate
PRs that re-run `benchmarks/` and review every ranking).

## Out of scope

Sparse, lapack, glm, macros, SIMD, complex numbers, generic dimension machinery, Schur,
Hessenberg, matrix exponential, convolution, serialization glue beyond `Serde`.

## Definition of done (every WP)

See [DESIGN.md](DESIGN.md) §D7 and [AGENTS.md](../AGENTS.md): doc comments, unit tests, oracle
vectors, gas benchmarks (with variants when the optimum is ambiguous), regenerated snapshot,
`scripts/check.sh` green.
