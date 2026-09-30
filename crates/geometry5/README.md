# nalgebra_geometry5

The 5D geometry: points, translations, scales, reflections of dimension 5 (Reflection5 whole), Matrix6 homogeneous (cg).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `geometry::point5`, `geometry::translation5`, `geometry::scale5`, `geometry::reflection5`, `base::point_swizzle`: points, translations, scales and reflections of dimension 5 (`Reflection5` whole: it reflects shapes of dimension 5 only);
- `base::cg`: the homogeneous-coordinate methods of `Matrix6`.

Upstream's module paths are kept: `nalgebra_geometry5::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md) and `simba`.

## When to depend on it

`nalgebra_geometry5` is a member of the declared closure `static5_geometry` (13.3 s / 3.14 GB, 20 s / 4.5 GB).

Depend on it for the 5D points, translations, scales and reflections and the homogeneous methods of `Matrix6`.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
