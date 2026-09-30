# nalgebra_linalg6

Cholesky (+ its column updates), LDLᵀ / UDU, QR, inverse of dimension 6 (`Lu6` is in `static6_wide`).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `linalg::cholesky`: Cholesky;
- `linalg::cholesky_update`: the Cholesky column updates (`rank_one_update`, `insert_column`, `remove_column`);
- `linalg::inverse`: the inverse;
- `linalg::qr`: QR;
- `linalg::udu`: UDU / LDLᵀ.

Upstream's module paths are kept: `nalgebra_linalg6::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_static6_wide`](../static6_wide/README.md) and `simba`.

Features (all on by default, forwarded by the facade's features of the same name): `qr`, `cholesky_update`. Each gates the module of that name; `nalgebra` with `default-features = false` leaves them out.

## When to depend on it

`nalgebra_linalg6` is a member of the declared closure `static6_factor` (17.7 s / 4.11 GB, 20 s / 4.5 GB).

Depend on the `linalg` packages for LU, Cholesky, LDLᵀ / UDU, QR and the inverse of the shapes of that dimension (`linalgN` pulls the lower ones for the column updates). `Lu6` is in `nalgebra_static6_wide`.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths (behind the features of the same names, on by default and still real: they gate code inside this package).

## `internal`

The modules under `internal` (`internal::linalg`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
