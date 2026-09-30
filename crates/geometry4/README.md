# nalgebra_geometry4

The 4D geometry: point / translation / scale 4 methods, Matrix5 homogeneous (cg).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `geometry::point`, `geometry::translation4`, `geometry::scale4`, `base::point_swizzle`: the methods of the points, translations and scales of dimension 4 (`Reflection4` lives with its struct in `nalgebra_types4`);
- `base::cg`: the homogeneous-coordinate methods of `Matrix5`.

Upstream's module paths are kept: `nalgebra_geometry4::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md) and `simba`.

## When to depend on it

`nalgebra_geometry4` is a member of the declared closure `static4_geometry` (11.2 s / 2.44 GB, 15 s / 3 GB).

Depend on it for the methods of the 4D points, translations and scales and the homogeneous methods of `Matrix5`.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
