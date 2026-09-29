# nalgebra_linalg_svd_eigen4

The symmetric eigen decomposition and the SVD up to 4x4, of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `linalg::symmetric_eigen1` .. `symmetric_eigen4`: `SymmetricEigen1` .. `SymmetricEigen4` (closed
  form in 2D, fixed-sweep Jacobi beyond) and `Matrix1SymmetricEigenTrait` ..;
- `linalg::svd2`, `linalg::svd3` and `linalg::svd`: `M = U·Σ·Vᵀ` of every shape up to 4 rows and 4
  columns (`Svd2`, `Svd3`, `Svd4`, `Svd2x3` ..), pseudo-inverse, rank, polar decomposition.

Upstream's module paths are kept: `nalgebra_linalg_svd_eigen4::linalg::svd3::Svd3` is
`nalgebra::linalg::svd3::Svd3` (docs/SPLIT.md §3.1). The dimension-5 / 6 forms are in the facade
`nalgebra` (until `nalgebra_linalg5` / `nalgebra_linalg6`). It depends on
[`nalgebra_core`](../core/README.md) and [`nalgebra_static3`](../static3/README.md) only.

## When to depend on it

Depend on `nalgebra_core` + `nalgebra_static3` + `nalgebra_linalg_svd_eigen4` (the declared closure
`static3_svd`) when you need the SVD or the eigen decomposition of the small shapes (polar
decompositions, principal axes) without the rest of the library. For upstream's whole API at
upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0
paths.

## Features

All on by default, as in `nalgebra` (whose features of the same names forward to these): `eigen`
(`linalg::symmetric_eigen1..4`) and `svd` (`linalg::svd*`, on top of `eigen`).

## `internal`

`nalgebra_linalg_svd_eigen4::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but
that the crates above `nalgebra_linalg_svd_eigen4` use: the SVD kernels (`SvdRightTrait`, the
filters of the pseudo-inverse), the symmetric `Sym4` input of the Jacobi kernel, and the internal
traits of `Svd2`, `Svd3`, `SymmetricEigen2`, `SymmetricEigen3` (`Jacobi3` with them), named by the
in-crate tests of `nalgebra`. They are public only for those crates: **internal, no stability
promise**, they may change or disappear in any release, and the facade never re-exports them.

## License

MIT.
