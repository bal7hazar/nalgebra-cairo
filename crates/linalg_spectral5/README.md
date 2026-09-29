# nalgebra_linalg_spectral5

The bidiagonal, Hessenberg, symmetric tridiagonal and real Schur decompositions, the eigenvectors
and the matrix exponential / power of dimension 5, of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `linalg::hessenberg` (`Hessenberg5`), `linalg::symmetric_tridiagonal` (`SymmetricTridiagonal5`),
  `linalg::bidiagonal` (`A = U·D·Vᵀ`, every shape with 5 rows or 5 columns and none larger), all by
  unrolled Householder reflections;
- `linalg::schur` (`Schur5`) and `linalg::eigen` (`Eigen5`), upstream's Francis iteration;
- `linalg::exp` (`Matrix5ExpTrait`), `linalg::pow` (`Matrix5PowTrait`).

The balancing and Householder building blocks of dimension 5 (`balance_parlett_reinsch`,
`clear_column_unchecked`...) are generic functions of `nalgebra_linalg_spectral4` /
`nalgebra_linalg4` whose dimension-5 impls sit with their types in `nalgebra_shapes5`.
Upstream's module paths are kept: `nalgebra_linalg_spectral5::linalg::schur::schur5::Schur5` is
`nalgebra::linalg::schur::schur5::Schur5` (docs/SPLIT.md §3.1). It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_shapes5`](../shapes5/README.md), [`nalgebra_static5`](../static5/README.md), [`nalgebra_linalg4`](../linalg4/README.md) and [`nalgebra_linalg_spectral4`](../linalg_spectral4/README.md) (the
Householder kernels, the singular-Padé message of `exp`).

## When to depend on it

Depend on it (and the crates it depends on) when you need the spectral decompositions or the matrix
exponential of the dimension-5 shapes without the rest of the library. For upstream's whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports
every sub-crate at the 0.1.0 paths.

## Features

All on by default, as in `nalgebra` (whose features of the same names forward to these):
`hessenberg` (`hessenberg`, `symmetric_tridiagonal`), `bidiagonal`, `schur` (`schur`, `eigen`, on
top of `hessenberg`), `exp` (`exp`, `pow`).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
