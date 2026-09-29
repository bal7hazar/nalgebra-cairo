# nalgebra_linalg_pivot4

The column-pivoting QR, the full-pivoting LU and the LBLᵀ decomposition up to 4x4,
of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `linalg::col_piv_qr`: `A·P = Q·R` by Householder reflections, every shape up to 4 rows and 4
  columns (`ColPivQr1` .. `ColPivQr4x3`);
- `linalg::full_piv_lu`: `P·A·Q = L·U` of the same shapes (`FullPivLu1` .. `FullPivLu4x3`);
- `linalg::lblt`: the Bunch-Kaufman `P·A·Pᵀ = L·B·Lᵀ` of the symmetric squares `Lblt1` .. `Lblt4`.

Upstream's module paths are kept: `nalgebra_linalg_pivot4::linalg::full_piv_lu::full_piv_lu3::FullPivLu3`
is `nalgebra::linalg::full_piv_lu::full_piv_lu3::FullPivLu3` (docs/SPLIT.md §3.1). The shapes with 5
or 6 rows or columns are in the facade `nalgebra` (until `nalgebra_linalg_pivot5` /
`nalgebra_linalg_pivot6`). It depends on [`nalgebra_core`](../core/README.md) only.

## When to depend on it

Depend on `nalgebra_core` + `nalgebra_linalg_pivot4` (the declared closure `core_pivot`) when you
need rank-revealing factorisations of the small shapes without the rest of the library. For
upstream's whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports every
sub-crate at the 0.1.0 paths.

## Features

All on by default, as in `nalgebra` (whose features of the same names forward to these):
`col_piv_qr`, `full_piv_lu`, `lblt` (one module each).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
