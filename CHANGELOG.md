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
- **The sub-crates**: TODO (one line per crate: name, content).
- **Why**: a consumer that needs part of the library depends on the sub-crates it uses and pays
  only for them. Every sub-crate is at most 40,000 library lines and adds at most 5 s / 1 GB to a
  cold build of an empty consumer over its own dependencies; the typical closures cost at most
  15 s / 3 GB (package granularity rule, enforced by the CI job `Consumer cost`). Measured (GitHub
  runner, cold build over an empty consumer): the facade TODO s / TODO GB (0.1.0: 96.7 s / 10.4 GB
  with the default features); typical closures TODO; `nalgebra_glam` TODO s / TODO GB (0.1.0 on
  `nalgebra` + `glam`: 37 s / 5.5 GB).
- **`nalgebra_glam` 0.1.1** depends on the sub-crates it converts (TODO: the crate list) and on
  `glam_core` instead of `nalgebra` and `glam`: it needs **`glam` ≥ 0.4.1** (which re-exports
  `glam_core`); with the monolithic `glam` 0.4.0 the glam types differ and `.into()` does not compile.

### Deprecated

- The facade features **`statistics`, `blas`, `dynamic`, `sparse` and `io`** are no-ops: their code
  lives in sub-crates the facade always depends on (Scarb has no optional dependency), so leaving
  them out saves no compile work. They stay declared so that a manifest naming them keeps building
  (CI job `Facade features`), and they still gate the facade's re-exports as in 0.1.0. **They are
  removed in 0.2.0.** Replacement: depend on the sub-crates that hold the code (TODO: the crate
  names). `closures`, `macros` and the linalg features (`eigen`, `svd`, `qr`, `cholesky_update`,
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
