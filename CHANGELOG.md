# Changelog

Numeric results are part of the API: any change of a result is a MINOR bump (pre-1.0), and a
release that changes results is scheduled so that consumers regenerate their goldens. Two packages
are versioned together from this repository: `nalgebra` and `nalgebra_glam`.

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
