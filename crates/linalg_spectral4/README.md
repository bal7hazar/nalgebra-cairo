# nalgebra_linalg_spectral4

The bidiagonal, Hessenberg, symmetric tridiagonal and real Schur decompositions, the eigenvectors,
the balancing and the matrix exponential / power up to 4x4, of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `linalg::hessenberg` (`A = Q·H·Qᵀ`), `linalg::symmetric_tridiagonal` (`A = Q·T·Qᵀ`),
  `linalg::bidiagonal` (`A = U·D·Vᵀ`, every shape up to 4 rows and columns), all by unrolled
  Householder reflections;
- `linalg::schur` (the real Schur form, the eigenvalues) and `linalg::eigen` (the eigenvectors of
  real simple spectra), upstream's Francis iteration;
- `linalg::balancing` (`balance_parlett_reinsch`, `unbalance`), `linalg::exp` (`Matrix1ExpTrait` ..
  `Matrix4ExpTrait`), `linalg::pow` (`Matrix1PowTrait` .. `Matrix4PowTrait`).

Upstream's module paths are kept: `nalgebra_linalg_spectral4::linalg::schur::schur3::Schur3` is
`nalgebra::linalg::schur::schur3::Schur3` (docs/SPLIT.md §3.1). The dimension-5 / 6 forms are in the
facade `nalgebra` (until `nalgebra_linalg_spectral5` / `nalgebra_linalg_spectral6`). It depends on
[`nalgebra_core`](../core/README.md), [`nalgebra_static3`](../static3/README.md),
[`nalgebra_shapes5`](../shapes5/README.md), [`nalgebra_static4`](../static4/README.md) and
[`nalgebra_linalg4`](../linalg4/README.md) (the Householder kernels, `Lu2` .. `Lu4` for `exp`).

## When to depend on it

Depend on it (and the crates it depends on) when you need the spectral decompositions or the matrix
exponential of the shapes up to 4x4 without the rest of the library. For upstream's whole API at
upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0
paths.

## Features

All on by default, as in `nalgebra` (whose features of the same names forward to these):
`hessenberg` (`hessenberg`, `symmetric_tridiagonal`, `balancing`), `bidiagonal`, `schur` (`schur`,
`eigen`, on top of `hessenberg`), `exp` (`exp`, `pow`).

## `internal`

`nalgebra_linalg_spectral4::internal` holds one item that `nalgebra` 0.1.0 keeps crate-private but
that the crates above use: the panic message of a singular Padé denominator of `exp`. It is public
only for those crates: **internal, no stability promise**, and the facade never re-exports it.

## License

MIT.
