# nalgebra_linalg_pivot5

The column-pivoting QR, the full-pivoting LU and the LBLᵀ decomposition of dimension 5, of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `linalg::col_piv_qr`: `A·P = Q·R` by Householder reflections, every shape with 5 rows or 5
  columns and none larger (`ColPivQr5`, `ColPivQr5x1` ..);
- `linalg::full_piv_lu`: `P·A·Q = L·U` of the same shapes (`FullPivLu5`, `FullPivLu5x1` ..);
- `linalg::lblt`: the Bunch-Kaufman `P·A·Pᵀ = L·B·Lᵀ` of `Matrix5` (`Lblt5`).

Upstream's module paths are kept: `nalgebra_linalg_pivot5::linalg::full_piv_lu::full_piv_lu5::FullPivLu5`
is `nalgebra::linalg::full_piv_lu::full_piv_lu5::FullPivLu5` (docs/SPLIT.md §3.1). The shapes up
to 4x4 are in `nalgebra_linalg_pivot4`, those of dimension 6 in `nalgebra_linalg_pivot6`. It
depends on [`nalgebra_core`](../core/README.md) and [`nalgebra_shapes5`](../shapes5/README.md).

## When to depend on it

Depend on it (and the crates it depends on) when you need rank-revealing factorisations of the
dimension-5 shapes without the rest of the library. For upstream's whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports
every sub-crate at the 0.1.0 paths.

## Features

All on by default, as in `nalgebra` (whose features of the same names forward to these):
`col_piv_qr`, `full_piv_lu`, `lblt` (one module each).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
