# nalgebra_linalg5

The QR decomposition, the SVD and the symmetric eigendecomposition of dimension 5, of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `linalg::qr`: `A = Q·R` by modified Gram-Schmidt of every shape with 5 rows or 5 columns and none
  larger (`Qr5`, `Qr5x1` .. `Qr5x4`, `Qr1x5` .. `Qr4x5`), with its dimension-5 kernels;
- `linalg::svd`: `M = U·Σ·Vᵀ` of the same shapes (`Svd5`, `Svd5x1` .. `Svd4x5`, pseudo-inverse,
  rank, polar decomposition);
- `linalg::symmetric_eigen5`: `SymmetricEigen5` (fixed-sweep Jacobi).

0.1.0 has no `Lu5`, `Cholesky5` or `Udu5` (the LU of dimension 5 is reached through
`FullPivLu5`, `nalgebra_linalg_pivot5`). Upstream's module paths are kept:
`nalgebra_linalg5::linalg::svd::svd5::Svd5` is `nalgebra::linalg::svd::svd5::Svd5`
(docs/SPLIT.md §3.1). It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_shapes5`](../shapes5/README.md) and [`nalgebra_linalg_svd_eigen4`](../linalg_svd_eigen4/README.md) (the SVD
filters and the Jacobi kernel of dimension 4).

## When to depend on it

Depend on it (and the crates it depends on) when you need the QR / SVD / symmetric eigen
decompositions of the dimension-5 shapes; the METHODS of those shapes are in
`nalgebra_static5`. For upstream's whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports
every sub-crate at the 0.1.0 paths.

## Features

All on by default, as in `nalgebra` (whose features of the same names forward to these): `qr`
(`linalg::qr`), `eigen` (`linalg::symmetric_eigen5`), `svd` (`linalg::svd`, on top of `eigen`).

## `internal`

`nalgebra_linalg5::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that
`nalgebra_linalg6` uses: the SVD kernel of dimension 5 (`SvdRightTrait5`) and the symmetric storage
of `SymmetricEigen5` (`Sym5`). They are public only for that crate: **internal, no stability
promise**, they may change or disappear in any release, and the facade never re-exports them.

## License

MIT.
