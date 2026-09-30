# nalgebra_linalg_pivot4

Column-pivoting QR, full-pivoting LU, LBLᵀ of dimension 4.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `linalg::col_piv_qr`: column-pivoting QR;
- `linalg::full_piv_lu`: full-pivoting LU;
- `linalg::lblt`: LBLᵀ (Bunch-Kaufman).

Upstream's module paths are kept: `nalgebra_linalg_pivot4::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md) and `simba`.

Features (all on by default, forwarded by the facade's features of the same name): `full_piv_lu`, `col_piv_qr`, `lblt`. Each gates the module of that name; `nalgebra` with `default-features = false` leaves them out.

## When to depend on it

`nalgebra_linalg_pivot4` is a member of the declared closure `core_pivot` (3.5 s / 1.04 GB, 15 s / 3 GB).

Depend on the `linalg_pivot` packages for the column-pivoting QR, the full-pivoting LU and the LBLᵀ factorisation.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths (behind the features of the same names, on by default and still real: they gate code inside this package).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
