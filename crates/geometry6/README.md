# nalgebra_geometry6

The 6D geometry: points, translations, scales, reflections of dimension 6 (Reflection6 whole).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `geometry::point6`, `geometry::translation6`, `geometry::scale6`, `geometry::reflection6`: points, translations, scales and reflections of dimension 6 (`Reflection6` whole).

Upstream's module paths are kept: `nalgebra_geometry6::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md) and `simba`.

## When to depend on it

`nalgebra_geometry6` is a member of the declared closure `static6_geometry` (18.0 s / 4.01 GB, 20 s / 4.5 GB).

Depend on it for the 6D points, translations, scales and reflections. Dimension 5-6 closures have their own budget (facade README, "Dimensions 5 and 6"); it was a documented exception to the 5 s / 1 GB marginal gate before the re-cut and is now covered by that budget.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
