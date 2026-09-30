# nalgebra_linalg5

QR of dimension 5; the `Perm5` methods (0.1.0 has no `Lu5`, `Cholesky5` or `Udu5`).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `linalg::lu`: LU and the permutation (`PermN`) methods;
- `linalg::qr`: QR.

Upstream's module paths are kept: `nalgebra_linalg5::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md) and `simba`.

Features (all on by default, forwarded by the facade's features of the same name): `qr`. Each gates the module of that name; `nalgebra` with `default-features = false` leaves them out.

## When to depend on it

`nalgebra_linalg5` is a member of the declared closure `static5_factor` (10.9 s / 3.13 GB, 20 s / 4.5 GB).

Depend on the `linalg` packages for LU, Cholesky, LDLᵀ / UDU, QR and the inverse of the shapes of that dimension (`linalgN` pulls the lower ones for the column updates). Dimension 5 has no `Lu5`, `Cholesky5` or `Udu5` in upstream 0.35 / 0.1.0: this package holds the QR of dimension 5 only (and the `Perm5` methods).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths (behind the features of the same names, on by default and still real: they gate code inside this package).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
