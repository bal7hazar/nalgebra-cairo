# Changelog

Numeric results are part of the API: any change of a result is a MINOR bump (pre-1.0), and a
release that changes results is scheduled so that consumers regenerate their goldens. Every
package of this repository (the `nalgebra` facade, its `nalgebra_*` sub-crates since 0.1.1, and
`nalgebra_glam`) is versioned together.

## 0.1.1 (unreleased)

<!-- WP 9-NS11b: skeleton. The crate list and every figure marked TODO are filled in by the release
PR once the re-cut of the package split (owner decision, 2026-09-29) has landed; the costs come from
the `Consumer cost` CI job (GitHub runner, medians of interleaved cold builds). -->

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
  - `nalgebra_core`: generic trait declarations (`MatrixMul`, `MatrixTrMul`, `MatrixIndex`, `Norm`, `Normed` / `Unit`, `MatrixSolve`, `PermuteRows`, `GivensRotate`, `LuSteps`, Householder / balancing, `TransformMul`), errors, fused kernels; the dimension-1 types (`Matrix1`, `Vector1`, `Point1`, `Translation1`, `Perm1`, `Reflection1`)
  - `nalgebra_types2`: the types of dimension 2 (shapes with max(rows, cols) = 2, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls); `Point2`, `Translation2`, `Perm2`, `Rotation2`, `Reflection2`
  - `nalgebra_types3`: the types of dimension 3 (as `types2`); `Point3`, `Translation3`, `Perm3`, `Rotation3`, `Reflection3`
  - `nalgebra_types4`: the types of dimension 4 (as `types2`); `Point4`, `Translation4`, `Perm4`, `Reflection4`
  - `nalgebra_types5`: the types of dimension 5 (as `types2`); `Point5`, `Translation5`, `Perm5`
  - `nalgebra_types6`: the types of dimension 6 (as `types2`, without the edit kernels); `Point6`, `Translation6`
  - `nalgebra_static_core`: the methods of the dimension-1 shapes (`Matrix1`, `Vector1`, `RowVector1`, `UnitVector1`)
  - `nalgebra_static2`: the methods of the dimension-2 shapes (`Matrix2`, `Vector2`, `RowVector2`, `Matrix1x2`...)
  - `nalgebra_static3`: the methods of the dimension-3 shapes (`Matrix3`, `Vector3`, `Matrix2x3`, `Matrix3x2`...)
  - `nalgebra_static4`: the methods of the dimension-4 shapes
  - `nalgebra_static5`: the methods of the dimension-5 shapes
  - `nalgebra_static6_tall`: the methods of the shapes with 6 rows and fewer columns (`Matrix6x1`..`Matrix6x5`, `Vector6`); dimension-6 edit kernels
  - `nalgebra_static6_wide`: the methods of the shapes with 6 columns (`Matrix6`, `Matrix2x6`..`Matrix5x6`, `RowVector6`); `Lu6`, `Perm6`
  - `nalgebra_geometry2`: the 1D / 2D geometry: `UnitComplex`, `Rotation2` methods, isometries, similarities, points / translations / scales 1-2 methods, swizzles, `Matrix2` / `Matrix3` homogeneous (`cg`)
  - `nalgebra_geometry3`: the 3D geometry: quaternions, unit quaternions, `Rotation3` methods, isometries, similarities, dual quaternions, `AbstractRotation`, point / translation / scale / reflection 3 methods, `Matrix4` homogeneous
  - `nalgebra_geometry4`: point / translation / scale 4 methods, `Matrix5` homogeneous
  - `nalgebra_geometry5`: points, translations, scales, reflections of dimension 5 (`Reflection5` whole), `Matrix6` homogeneous
  - `nalgebra_geometry6`: points, translations, scales, reflections of dimension 6 (`Reflection6` whole)
  - `nalgebra_transform2`: `Transform2`, `Projective2`, `Affine2` and their products (upstream `geometry::transform`)
  - `nalgebra_transform3`: `Transform3`, `Projective3`, `Affine3`, `Perspective3`, `Orthographic3` and their products
  - `nalgebra_blocks`: row / column blocks, resize, pad / crop, Kronecker products
  - `nalgebra_views`: `FixedView` and its 441 impls
  - `nalgebra_norm`: the norm markers and every `Norm` impl
  - `nalgebra_statistics2`: `base::statistics` of the dimension-2 shapes (one inherent trait per shape)
  - `nalgebra_statistics3`: `base::statistics` of dimension 3
  - `nalgebra_statistics4`: `base::statistics` of dimension 4
  - `nalgebra_statistics5`: `base::statistics` of dimension 5
  - `nalgebra_statistics6`: `base::statistics` of dimension 6
  - `nalgebra_blas`: `base::blas` (`MatrixGemm` and its 216 impls, the per-shape BLAS traits)
  - `nalgebra_linalg_core`: shared linalg kernels: Householder kernels and steps (`clear_column_unchecked`, `assemble_q`), `reflection_axis_mut`, balancing, Givens methods, the SVD right-vector kernel trait
  - `nalgebra_linalg2`: LU, Cholesky, LDLᵀ / UDU, QR, inverse of dimension 2 (and 1)
  - `nalgebra_linalg_svd_eigen2`: SVD and symmetric eigen of dimension 2 (and 1)
  - `nalgebra_linalg_pivot2`: column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 2 (and 1)
  - `nalgebra_linalg_spectral2`: bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 2 (and 1)
  - `nalgebra_linalg3`: LU, Cholesky (+ the column updates of dimension 2), LDLᵀ / UDU, QR, inverse of dimension 3
  - `nalgebra_linalg_svd_eigen3`: SVD and symmetric eigen of dimension 3
  - `nalgebra_linalg_pivot3`: column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 3
  - `nalgebra_linalg_spectral3`: bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 3
  - `nalgebra_linalg4`: LU, Cholesky (+ the column updates of dimension 3), LDLᵀ / UDU, QR, inverse of dimension 4
  - `nalgebra_linalg_svd_eigen4`: SVD and symmetric eigen of dimension 4
  - `nalgebra_linalg_pivot4`: column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 4
  - `nalgebra_linalg_spectral4`: bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 4
  - `nalgebra_linalg5`: LU, Cholesky (+ the column updates of dimension 4), LDLᵀ / UDU, QR, inverse of dimension 5
  - `nalgebra_linalg_svd_eigen5`: SVD and symmetric eigen of dimension 5
  - `nalgebra_linalg_pivot5`: column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 5
  - `nalgebra_linalg_spectral5`: bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 5
  - `nalgebra_linalg6`: Cholesky, LDLᵀ / UDU, QR, inverse of dimension 6 (`Lu6` is in `static6_wide`)
  - `nalgebra_linalg_svd_eigen6`: SVD and symmetric eigen of dimension 6
  - `nalgebra_linalg_pivot6`: column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 6
  - `nalgebra_linalg_spectral6`: bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 6
  - `nalgebra_dynamic`: `DMatrix`, `DVector`, the dynamic forms
  - `nalgebra_sparse`: `sparse`, `io`
  - `nalgebra`: the facade: root functions, macros, `LuInvert`, `MatrixInfSup` and the 0.1.0 module tree, re-exporting every package above at its 0.1.0 paths.
  - `nalgebra_glam`: unchanged API
- **Why**: a consumer that needs part of the library depends on the sub-crates it uses and pays
  only for them. Every sub-crate is at most 40,000 library lines and adds at most 5 s / 1 GB to a
  cold build of an empty consumer over its own dependencies; the typical closures cost at most
  15 s / 3 GB (package granularity rule, enforced by the CI job `Consumer cost`). Measured (GitHub
  runner, cold build over an empty consumer): the facade TODO s / TODO GB (0.1.0: 96.7 s / 10.4 GB
  with the default features); typical closures TODO (the dimension 5-6 closures have their own budget, below); `nalgebra_glam` TODO s / TODO GB (0.1.0 on
  `nalgebra` + `glam`: 37 s / 5.5 GB).
- **Dimensions 5 and 6** (docs/SPLIT.md §19): dimension k builds on every dimension below it, so a
  closure that includes dimension 5 or 6 costs more than all of dimensions 2-4. It has its own
  budget, **20 s / 4.5 GB** on the runner (15 s / 3 GB up to dimension 4), declared and enforced in
  `consumer_cost.toml`; gates 1 and 2 (40,000 lines, 5 s / 1 GB marginal) apply to every package.
  `nalgebra_geometry6` and `nalgebra_static6_wide` are no longer exceptions: they are covered by that
  budget. Two combined closures are documented in the README, not gated: static 6 + SVD / eigen 6 and
  static 6 + spectral 6 (TODO s / TODO GB each; the decomposition closures themselves are declared
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

- **The cost of `default-features = false` on the facade**: 28.7 s / 5.2 GB with 0.1.0 → TODO s /
  TODO GB (CI `Consumer cost`, closure `facade_no_default_features`). The facade depends on every
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
