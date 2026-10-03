# Changelog

Numeric results are part of the API: any change of a result is a MINOR bump (pre-1.0), and a
release that changes results is scheduled so that consumers regenerate their goldens. Every
package of this repository (the `nalgebra` facade, its `nalgebra_*` sub-crates since 0.1.1, and
`nalgebra_glam`) is versioned together.

## 0.2.0 (unreleased)

No numeric result moves: every test, golden and oracle vector of 0.1.1 passes unedited. MINOR:
consumers need Cairo 2.20.0 (below).

### Changed

- **Version 0.2.0 for every package** (the 54 published ones and the unpublished test crates), and
  every inter-crate requirement `0.2.0`.
- Dependencies unchanged: simba 0.2.0 / fixed 0.4.0. simba 0.3.0 / fixed 0.5.0 are not taken yet:
  `nalgebra_glam` depends on `glam_core` 0.4.1, which requires `fixed ^0.4.0`, and Scarb resolves
  one `fixed` per build; they come in a later release once `glam_core` moves to fixed 0.5.0.
- **Toolchain**: Scarb 2.20.1 / Cairo 2.20.0 (`cairo-version = "2.20.0"`), snforge 0.64.0 (WP 10-TC,
  #93).
- **`UnitQuaternion::from_rotation_matrix`** selects its trace branch with nalgebra-rs's own
  `tr > 0` instead of `Real::is_sign_positive(tr)`. Same bits for every input today (simba 0.2.0's
  `is_sign_positive(0)` is false); it keeps nalgebra-rs's result at a trace of exactly 0 once simba
  0.3.0 (`is_sign_positive(0)` true, as simba-rs) is taken: the -120° turn about `(1, 1, 1)` gives
  `(w, i, j, k) = (-0.5, 0.5, 0.5, 0.5)`, pinned by
  `test_from_rotation_matrix_zero_trace` (`docs/research/rel-zero.md`: the other five
  `is_sign_positive` sites cannot see a zero a caller can pass).
- The facade features `statistics`, `blas`, `dynamic`, `sparse` and `io` (deprecated in 0.1.1) are
  still declared: their removal, announced for 0.2.0, would change the public API and is left to a
  later release.

### Performance

Fewer Cairo steps on the hot paths, every result bit-identical (all tests, goldens and probes
unchanged, plus in-file equivalence sweeps against the previous bodies). Net steps per call
(`docs/STEPS.md`, probes of `crates/probes_steps`, WP 11-OPT-0, #101), 0.1.1 → 0.2.0:

| probe | 0.1.1 | 0.2.0 | saved |
|---|---:|---:|---:|
| `unit_quaternion_mul` | 94 | 79 | 15 |
| `unit_quaternion_transform_vector` | 181 | 167 | 14 |
| `unit_quaternion_from_axis_angle` | 307 | 296 | 11 |
| `unit_quaternion_slerp` | 920 | 907 | 13 |
| `isometry3_mul` | 292 | 253 | 39 |
| `isometry3_transform_point` | 193 | 173 | 20 |
| `isometry3_inverse` | 200 | 177 | 23 |
| `isometry3_inv_mul` | 321 | 265 | 56 |
| `cholesky3_factor` | 220 | 205 | 15 |
| `cholesky3_solve` | 246 | 226 | 20 |
| `cholesky6_factor` | 897 | 822 | 75 |
| `cholesky6_solve` | 569 | 529 | 40 |
| `lu3_solve` | 198 | 173 | 25 |
| `lu6_factor` | 1919 | 1857 | 62 |
| `lu6_solve` | 542 | 483 | 59 |
| `symmetric_eigen3` | 4049 | 3532 | 517 |
| `svd3` | 5250 | 4713 | 537 |

- **Geometry** (WP 11-OPT-1, #102): inlined Hamilton product, quaternion sandwiches and isometry
  products; the shortest-arc flip of `try_slerp` folded into the sign of one weight; the negation of
  the translation in `Isometry2/3::inverse` folded into the sandwich. A removed panic, not a result
  change: `slerp` no longer panics when a component of `other` is the minimum of the scalar (the
  flip no longer negates it).
- **Small linear algebra** (WP 11-OPT-2, #103): inlined Jacobi steps of `SymmetricEigen3`,
  `Svd3::new`, the Cholesky 3 / 6 and LU 3 / 6 factor and solve paths; `Cholesky6::new` shares one
  prepared divisor per column (`Real::div3`, `div4`, bit-identical to per-element division); the
  row swap of step 1 of `Lu6::new` is a `match` on the pivot row.
- Unchanged: `vector3_normalize` (100), `lu3_factor` (286) and the generated `matrix3_*` /
  `matrix6_mul_vec` kernels.

### Tooling

- **Steps probes** (WP 11-OPT-0, #101): `crates/probes_steps` and the `steps/` snapshot, checked in
  CI, measure the net Cairo steps of the hot primitives (`docs/STEPS.md`).
- **CI and pre-push** (#95-#99, #104): `scripts/prepush.sh` behind `.githooks/pre-push` runs the
  checks of the files a push touches, with the Cairo compile under the shared build lock and an
  8 GB memory cap (left to CI when either is not available); CI test jobs run only when their files
  change, behind one always-run `CI result` check.
- **Release** (WP 12-REL-a, #105): `scripts/release.py` no longer publishes; `request` builds each
  published package's archive at a commit and writes `docs/releases/<version>.md`, `verify` reads the
  registry back against it.

## 0.1.1 (2026-10-01)

- **The package split** (docs/PLAN.md M9, docs/SPLIT.md): the library is cut into sub-crates
  `nalgebra_*` behind the `nalgebra` facade, published together at one version in dependency order
  (`scripts/release.py`). **No path, API or numeric change**: the facade re-exports every public path
  of 0.1.0 (9,289 paths, proved in CI by `tools/split/public_paths.py --check`, nothing more, nothing
  less), the sub-crates keep upstream's module paths, and every benchmark keeps its step count (zero
  step change, `tools/split/gas_compare.py` on every move). A 0.1.0 consumer updates the version and
  nothing else.
- **The 54 packages** (52 sub-crates, the facade and `nalgebra_glam`; a number in a name is exactly
  that dimension, the dimension of a rectangular shape being max(rows, columns); docs/SPLIT.md §18.1),
  lowest first:
  - `nalgebra_core`: The generic trait declarations (MatrixMul, MatrixTrMul, MatrixIndex, Norm, Normed / Unit, MatrixSolve, PermuteRows, GivensRotate, LuSteps, Householder / balancing, TransformMul), errors, fused kernels, and the dimension-1 types (Matrix1, Vector1, Point1, Translation1, Perm1, Reflection1 with its constructors and accessors)
  - `nalgebra_types2`: The types of dimension 2 (shapes with max(rows, cols) = 2, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls); Point2, Translation2, Perm2, Rotation2, Reflection2 (with its constructors and accessors), GivensRotation
  - `nalgebra_types3`: The types of dimension 3 (shapes with max(rows, cols) = 3, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls); Point3, Translation3, Perm3, Rotation3, Reflection3 (with its constructors and accessors)
  - `nalgebra_types4`: The types of dimension 4 (shapes with max(rows, cols) = 4, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls); Point4, Translation4, Perm4, Reflection4 (with its constructors and accessors)
  - `nalgebra_types5`: The types of dimension 5 (shapes with max(rows, cols) = 5, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls); Point5, Translation5, Perm5
  - `nalgebra_types6`: The types of dimension 6 (shapes with max(rows, cols) = 6, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls, without the edit kernels); Point6, Translation6
  - `nalgebra_static_core`: The methods of the dimension-1 shapes (Matrix1, Vector1, RowVector1, UnitVector1)
  - `nalgebra_static2`: The methods of the dimension-2 shapes (Matrix2, Vector2, RowVector2, Matrix1x2...)
  - `nalgebra_static3`: The methods of the dimension-3 shapes (Matrix3, Vector3, Matrix2x3, Matrix3x2...)
  - `nalgebra_static4`: The methods of the dimension-4 shapes (Matrix4, Vector4, Matrix2x4, Matrix4x3...)
  - `nalgebra_static5`: The methods of the dimension-5 shapes (Matrix5, Vector5, Matrix2x5, Matrix5x3...)
  - `nalgebra_static6_tall`: The methods of the shapes with 6 rows and fewer columns (Matrix6x1..Matrix6x5, Vector6); the dimension-6 edit kernels
  - `nalgebra_static6_wide`: The methods of the shapes with 6 columns (Matrix6, Matrix2x6..Matrix5x6, RowVector6); Lu6, Perm6
  - `nalgebra_geometry2`: The 1D / 2D geometry: UnitComplex, Rotation2 methods, isometries, similarities, points / translations / scales 1-2 methods, swizzles, Matrix2 / Matrix3 homogeneous (cg)
  - `nalgebra_geometry3`: The 3D geometry: quaternions, unit quaternions, Rotation3 methods, isometries, similarities, dual quaternions, AbstractRotation, point / translation / scale 3 methods, Matrix4 homogeneous (cg)
  - `nalgebra_geometry4`: The 4D geometry: point / translation / scale 4 methods, Matrix5 homogeneous (cg)
  - `nalgebra_geometry5`: The 5D geometry: points, translations, scales, reflections of dimension 5 (Reflection5 whole), Matrix6 homogeneous (cg)
  - `nalgebra_geometry6`: The 6D geometry: points, translations, scales, reflections of dimension 6 (Reflection6 whole)
  - `nalgebra_transform2`: Transform2, Projective2, Affine2 and their products (upstream geometry::transform)
  - `nalgebra_transform3`: Transform3, Projective3, Affine3, Perspective3, Orthographic3 and their products (upstream geometry::transform)
  - `nalgebra_blocks`: Row / column blocks, resize, pad / crop, Kronecker products, the fixed-size edition of the static shapes (upstream base::edition, base::blocks)
  - `nalgebra_views`: FixedView and its impls: fixed-size views of the static shapes (upstream base::matrix_view)
  - `nalgebra_norm`: The norm markers (EuclideanNorm, LpNorm, OneNorm, UniformNorm) and every Norm impl (upstream base::norm)
  - `nalgebra_statistics2`: base::statistics of the dimension-2 shapes (one inherent trait per shape)
  - `nalgebra_statistics3`: base::statistics of the dimension-3 shapes (one inherent trait per shape)
  - `nalgebra_statistics4`: base::statistics of the dimension-4 shapes (one inherent trait per shape)
  - `nalgebra_statistics5`: base::statistics of the dimension-5 shapes (one inherent trait per shape)
  - `nalgebra_statistics6`: base::statistics of the dimension-6 shapes (one inherent trait per shape)
  - `nalgebra_blas`: base::blas: MatrixGemm and its impls, the per-shape BLAS traits
  - `nalgebra_linalg_core`: Shared linalg kernels: the Householder kernels and steps (`clear_column_unchecked`, `assemble_q`), `reflection_axis_mut`, balancing, the SVD filters, the `exp` Padé message
  - `nalgebra_linalg2`: LU, Cholesky, LDLᵀ / UDU, QR, inverse of dimension 2 (and 1); the `Perm1` / `Perm2` methods
  - `nalgebra_linalg_svd_eigen2`: SVD and symmetric eigen of dimension 2 (and 1)
  - `nalgebra_linalg_pivot2`: Column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 2 (and 1)
  - `nalgebra_linalg_spectral2`: Bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 2 (and 1)
  - `nalgebra_linalg3`: LU, Cholesky (+ the column updates of dimension 2), LDLᵀ / UDU, QR, inverse of dimension 3; the `Perm3` methods
  - `nalgebra_linalg_svd_eigen3`: SVD and symmetric eigen of dimension 3
  - `nalgebra_linalg_pivot3`: Column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 3
  - `nalgebra_linalg_spectral3`: Bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 3
  - `nalgebra_linalg4`: LU, Cholesky (+ the column updates of dimension 3), LDLᵀ / UDU, QR, inverse of dimension 4; the `Perm4` methods
  - `nalgebra_linalg_svd_eigen4`: SVD and symmetric eigen of dimension 4
  - `nalgebra_linalg_pivot4`: Column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 4
  - `nalgebra_linalg_spectral4`: Bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 4
  - `nalgebra_linalg5`: QR of dimension 5; the `Perm5` methods (0.1.0 has no `Lu5`, `Cholesky5` or `Udu5`)
  - `nalgebra_linalg_svd_eigen5`: SVD and symmetric eigen of dimension 5
  - `nalgebra_linalg_pivot5`: Column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 5
  - `nalgebra_linalg_spectral5`: Bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 5
  - `nalgebra_linalg6`: Cholesky (+ its column updates), LDLᵀ / UDU, QR, inverse of dimension 6 (`Lu6` is in `static6_wide`)
  - `nalgebra_linalg_svd_eigen6`: SVD and symmetric eigen of dimension 6
  - `nalgebra_linalg_pivot6`: Column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 6
  - `nalgebra_linalg_spectral6`: Bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 6
  - `nalgebra_dynamic`: DMatrix, DVector, RowDVector, the dynamic forms of the static shapes
  - `nalgebra_sparse`: sparse (legacy CsMatrix, CsCholesky) and io (Matrix Market)
  - `nalgebra`: the facade: root functions, macros, `LuInvert`, `MatrixInfSup` and the 0.1.0 module tree, re-exporting every package above at its 0.1.0 paths.
  - `nalgebra_glam`: Conversions between nalgebra-cairo and glam-cairo types: the glam interop of the Rust nalgebra crate (its `convert-glam` features) as a separate package
- **Why**: a consumer that needs part of the library depends on the sub-crates it uses and pays
  only for them. Every sub-crate is at most 40,000 library lines and adds at most 5 s / 1 GB to a
  cold build of an empty consumer over its own dependencies; the typical closures cost at most
  15 s / 3 GB (package granularity rule, enforced by the CI job `Consumer cost`). Measured (GitHub
  runner, cold build over an empty consumer): the facade 29.8 s / 9.96 GB (0.1.0: 96.7 s / 10.4 GB
  with the default features); typical closures up to dimension 4, all gated at 15 s / 3 GB: 3.5 - 11.7 s / 1.05 - 2.83 GB
  (static 2-3 + SVD 5.5 s, static 2-4 + factorisations 8.3 s, static 2-4 + geometry 10.4 s,
  static 2-4 + BLAS 11.7 s / 2.83 GB; the dimension 5-6 closures have
  their own budget, below); `nalgebra_glam` 7.6 s / 1.69 GB (0.1.0 on `nalgebra` + `glam`: 37 s /
  5.5 GB). Figures of [CI run 36767869070](https://github.com/bal7hazar/nalgebra-cairo/actions/runs/36767869070) (5 interleaved rounds, one runner per shard); [docs/PACKAGES.md](docs/PACKAGES.md)
  holds the per-package table of the release commit's run ([36841314574](https://github.com/bal7hazar/nalgebra-cairo/actions/runs/36841314574)),
  whose medians differ from these by the runners' spread (same code, same line counts). The highest package marginal is 2.6 s
  (`nalgebra_dynamic`, `nalgebra_static5`) and 0.77 GB (`nalgebra_linalg_svd_eigen6`) in run 36767869070, and
  3.5 s / 0.77 GB (`nalgebra_linalg_svd_eigen6`) in the release run, under the gate of 5 s / 1 GB in both.
- **Dimensions 5 and 6** (docs/SPLIT.md §19): dimension k builds on every dimension below it, so a
  closure that includes dimension 5 or 6 costs more than all of dimensions 2-4. It has its own
  budget, **20 s / 4.5 GB** on the runner (15 s / 3 GB up to dimension 4), declared and enforced in
  `consumer_cost.toml`; gates 1 and 2 (40,000 lines, 5 s / 1 GB marginal) apply to every package.
  `nalgebra_geometry6` and `nalgebra_static6_wide` are no longer exceptions: they are covered by that
  budget. Two combined closures are documented in the README, not gated: static 6 + SVD / eigen 6 and
  static 6 + spectral 6 (11.6 s / 4.38 GB and 15.2 s / 4.13 GB in [CI run 36767869070](https://github.com/bal7hazar/nalgebra-cairo/actions/runs/36767869070); 21.5 s / 4.70 GB and
  21.5 s / 4.64 GB in the 15-round measurement of docs/SPLIT.md §18.2, the runners of two CI jobs
  differing by up to a factor of two in speed; the decomposition closures themselves are declared
  without the dimension-6 method crates). The README's "Dimensions 5 and 6" section has the figures of
  each closure, and the comparison with 0.1.0 (96.7 s / 10.4 GB in one package).
  `nalgebra_blocks`, `nalgebra_views` and `nalgebra_norm` (generic traits whose dimension 5-6 impls
  cannot be cut per dimension) are advanced use: "static 2-4 + family" is documented, not gated;
  the everyday methods (`norm()`, `normalize()`, `dot`, `transpose`...) do not need them.
- **`nalgebra_glam` 0.1.1** depends on the sub-crates it converts (`nalgebra_core`, `nalgebra_types2`
  .. `nalgebra_types4`, `nalgebra_static2`, `nalgebra_static3`, `nalgebra_geometry2`,
  `nalgebra_geometry3`) and on
  `glam_core` instead of `nalgebra` and `glam`: it needs **`glam` ≥ 0.4.1** (which re-exports
  `glam_core`); with the monolithic `glam` 0.4.0 the glam types differ and `.into()` does not compile.

### Deprecated

- The facade features **`statistics`, `blas`, `dynamic`, `sparse` and `io`** are no-ops: their code
  lives in sub-crates the facade always depends on (Scarb has no optional dependency), so leaving
  them out saves no compile work. They stay declared so that a manifest naming them keeps building
  (CI job `Facade features`), and they still gate the facade's re-exports as in 0.1.0. **They are
  removed in 0.2.0.** Replacement: depend on the sub-crates that hold the code: `statistics` ->
  `nalgebra_statistics2` .. `nalgebra_statistics6`; `blas` -> `nalgebra_blas`; `dynamic` ->
  `nalgebra_dynamic`; `sparse` and `io` -> `nalgebra_sparse` (with `nalgebra_dynamic`, which it pulls). `closures`, `macros` and the linalg features (`eigen`, `svd`, `qr`, `cholesky_update`,
  `full_piv_lu`, `col_piv_qr`, `lblt`, `hessenberg`, `bidiagonal`, `schur`, `exp`) keep gating code.

### Changed

- **The cost of `default-features = false` on the facade**: 28.7 s / 5.2 GB with 0.1.0 → 26.7 s /
  6.29 GB ([CI run 36767869070](https://github.com/bal7hazar/nalgebra-cairo/actions/runs/36767869070), `Consumer cost`, closure `facade_no_default_features`). The facade depends on every
  sub-crate unconditionally (Scarb has no optional dependency), so turning its features off no
  longer removes the code of the families from the build. Accepted by the programme session
  (docs/SPLIT.md §12.1): the facade is for parity with nalgebra-rs's paths; a light build depends on
  the sub-crates. No API change.

## 0.1.0 (2026-09-26)

- First release from this repository: a Cairo port of the Rust `nalgebra` crate (0.35.0) on the
  Q32.32 fixed point of fixed-cairo, designed gas-first (no loops in static code, every sum of
  products through a fused kernel of `simba::scalar::Real`).
- **Coverage of nalgebra-rs 0.35.0**: 1,885 of the 1,886 items of the coverage target (99.9 %,
  [docs/API_PARITY.md](docs/API_PARITY.md), checked in CI). The only item not ported is
  `io::cs_matrix_from_matrix_market(path)` (owner ruling: a Cairo program has no file system;
  `cs_matrix_from_matrix_market_str` is there). The documented exclusions are the Rust machinery
  with no Cairo counterpart (`simd`, `rayon`, `unsafe`, `borrow`, `fmt`, `glue`, `random`,
  `interop`, `generic-dim`, 550 items in all), each with its reason in the parity file.
- **Modules**: `base` (the 36 static shapes `Matrix1..6`, `MatrixRxC`, `Vector1..6`,
  `RowVector1..6` with their 54 aliases, points, unit vectors, the homogeneous helpers, views,
  blocks, edition, the functional and in-place variants; `DMatrix` / `DVector` and the partially
  dynamic aliases behind `dynamic`), `geometry` (rotations, quaternions, dual quaternions,
  isometries, similarities, translations, scale, reflection, transforms, projections),
  `linalg` (LU, full-pivot LU, Cholesky and rank-one updates, UDU, LDLT, LBLT, QR, column-pivot QR,
  symmetric eigen, SVD, Hessenberg, Bidiagonal, symmetric tridiagonal, Schur, Eigen, balancing,
  matrix exponential and integer power, the triangular and general solves), `sparse` (`CsMatrix`,
  `CsVector`, `CsCholesky`), `io` (Matrix Market), the crate-root functions and the construction
  macros (`matrix!`, `vector!`, `point!`, `stack!`, `dmatrix!`, `dvector!`).
- **Features** (DESIGN D9): `statistics`, `blas`, `closures`, `dynamic`, `sparse` (enables
  `dynamic`), `io` (enables `sparse`), `eigen`, `svd` (enables `eigen`), `qr`, `cholesky_update`,
  `full_piv_lu`, `col_piv_qr`, `lblt`, `macros`, `hessenberg`, `bidiagonal`, `schur` (enables
  `hessenberg`), `exp`. **`default` includes all of them**, so `nalgebra = "0.1.0"` exposes
  upstream's whole surface; `default-features = false` (plus `features = [..]`) is the lighter
  form for a dependent that needs part of it.
- **Compile cost** of a cold `scarb build -p nalgebra` (Scarb 2.19.4, a shared CPU-capped
  machine; peak resident memory / wall / CPU): 11.6 GB / 99 s / 170 s with the default
  features, 6.1 GB / 32 s / 62 s with `default-features = false`. The per-feature savings are in
  the [README](README.md#features).
- **Scalar**: fixed-cairo's `fixed::Fixed` 0.4.0 (Q32.32) through `simba` 0.2.0, the Cairo
  counterpart of Dimforge's simba (`Real`, `Transcendental`). Every type is generic over
  `simba::scalar::Real`.
- **Fidelity rules** (the scalar's rounding is the spec): division and reciprocal are rounded to
  nearest, ties to even (like `f64 /`); products and fused kernels are floored once per output
  scalar; overflow panics, nothing wraps; where upstream returns `NaN` (division by zero, the
  normalisation of a null vector, a domain error) the Cairo function panics with a stable message
  (`errors` modules, part of the API). Same inputs give bit-identical outputs forever.
- **`nalgebra_glam` 0.1.0**: the conversions between nalgebra-cairo and glam-cairo types (upstream's
  `convert-glam` features, 91 items) as a separate package (DESIGN D11: Scarb has no optional
  dependencies and `glam` costs every dependent of `nalgebra` compile memory). It depends on
  `nalgebra = "0.1.0"` without its default features, `glam = "0.4.0"` and `fixed = "0.4.0"`.
- The construction macros and the `Sum` / `Product` impls need the experimental Cairo features
  `user_defined_inline_macros` and `associated_item_constraints`, enabled in this package's
  manifest; a dependent needs neither (DESIGN D10).
- Publication order: `fixed` → `simba` → `nalgebra` → `nalgebra_glam`; `nalgebra_glam` can only be
  verified by `scarb package` once `nalgebra` is on the registry.
