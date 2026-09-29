# nalgebra_linalg6

The Cholesky, UDU, QR, SVD and symmetric eigen decompositions of dimension 6, of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `linalg::cholesky` (`Cholesky6`, `Matrix6CholeskyTrait`), `linalg::cholesky_update` (its rank-one
  updates, column insertion / removal), `linalg::udu` (`Udu6`), `linalg::inverse`
  (`Matrix6InverseTrait`, on `Lu6`);
- `linalg::qr`: `A = Q·R` of every shape with 6 rows or 6 columns (`Qr6`, `Qr6x1` .. `Qr5x6`), with
  its dimension-6 kernels;
- `linalg::svd`: the SVD of the same shapes (`Svd6`, `Svd6x1` .. `Svd5x6`);
- `linalg::symmetric_eigen6`: `SymmetricEigen6`.

`Lu6` (with `Perm6`) is in `nalgebra_static6_wide`, because `Matrix6::determinant` /
`try_inverse` run it. Upstream's module paths are kept: `nalgebra_linalg6::linalg::cholesky::Cholesky6`
is `nalgebra::linalg::cholesky::Cholesky6` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_core`](../core/README.md), [`nalgebra_shapes5`](../shapes5/README.md), [`nalgebra_shapes6`](../shapes6/README.md), [`nalgebra_static6_wide`](../static6_wide/README.md), [`nalgebra_linalg_svd_eigen4`](../linalg_svd_eigen4/README.md) and [`nalgebra_linalg5`](../linalg5/README.md) (the SVD kernels of
dimensions 4 and 5, used by the shapes `6xC`).

## When to depend on it

Depend on it (and the crates it depends on) when you need the factorisations of the dimension-6
shapes (`Matrix6` solving, least squares, SVD); the METHODS of those shapes are in
`nalgebra_static6_tall` / `nalgebra_static6_wide`. For upstream's whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports
every sub-crate at the 0.1.0 paths.

## Features

All on by default, as in `nalgebra` (whose features of the same names forward to these): `qr`
(`linalg::qr`), `cholesky_update` (`linalg::cholesky_update`), `eigen` (`linalg::symmetric_eigen6`),
`svd` (`linalg::svd`, on top of `eigen`).

## `internal`

`nalgebra_linalg6::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
in-crate tests and benchmarks of `nalgebra` name: the `LDLᵀ` kernel of `udu` (`Ldlt6`) and the SVD
kernel of dimension 6 (`SvdRightTrait6`). They are public only for that reason: **internal, no
stability promise**, they may change or disappear in any release, and the facade never re-exports
them.

## License

MIT.
