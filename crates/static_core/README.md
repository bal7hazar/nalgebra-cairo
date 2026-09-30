# nalgebra_static_core

The methods of the dimension-1 shapes (Matrix1, Vector1, RowVector1, UnitVector1).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::matrix1`: the inherent methods of `Matrix1`, `Vector1`, `RowVector1` and `UnitVector1` (they cannot sit in `nalgebra_core`: `Matrix1::insert_row` builds a `Vector2`).

Upstream's module paths are kept: `nalgebra_static_core::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md) and `simba`.

Feature `closures` (on by default, forwarded by the facade's feature of the same name): the methods that take a closure (`map`, `fold`, `apply`, `zip_*`...).

## When to depend on it

It is pulled in by `static4_norm` (18.7 s / 4.04 GB, documented, not gated).

Depend on it for the methods of the 1x1 / 1xN shapes; no other static crate needs it.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
