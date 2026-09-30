# nalgebra_linalg_spectral6

Bidiagonal, Schur, eigen, Hessenberg, tridiagonal, `exp` / `pow` of dimension 6.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `linalg::bidiagonal`: bidiagonalisation;
- `linalg::eigen`: `Eigen` (general eigenvalues);
- `linalg::exp`: the matrix exponential;
- `linalg::hessenberg`: Hessenberg;
- `linalg::pow`: the integer matrix power;
- `linalg::schur`: Schur;
- `linalg::symmetric_tridiagonal`: symmetric tridiagonalisation.

Upstream's module paths are kept: `nalgebra_linalg_spectral6::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_static6_wide`](../static6_wide/README.md), [`nalgebra_linalg_core`](../linalg_core/README.md) and `simba`.

Features (all on by default, forwarded by the facade's features of the same name): `hessenberg`, `bidiagonal`, `schur`, `exp`. Each gates the module of that name; `nalgebra` with `default-features = false` leaves them out.

## When to depend on it

`nalgebra_linalg_spectral6` is a member of the declared closure `static6_spectral` (21.5 s / 4.64 GB, documented, not gated).

Depend on the `linalg_spectral` packages for the bidiagonal, Schur, eigen, Hessenberg, symmetric tridiagonal decompositions and the matrix `exp` / `pow`.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths (behind the features of the same names, on by default and still real: they gate code inside this package).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
