# nalgebra_transform3

Transform3, Projective3, Affine3, Perspective3, Orthographic3 and their products (upstream geometry::transform).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `geometry::transform3`, `geometry::projective3`, `geometry::affine3`, `geometry::perspective3`, `geometry::orthographic3`: `Transform3`, `Projective3`, `Affine3`, `Perspective3`, `Orthographic3`, their operators and their products (upstream `geometry::transform`).

Upstream's module paths are kept: `nalgebra_transform3::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_static4`](../static4/README.md), [`nalgebra_geometry3`](../geometry3/README.md), [`nalgebra_transform2`](../transform2/README.md) and `simba`.

## When to depend on it

`nalgebra_transform3` is a member of the declared closure `static4_geometry` (11.2 s / 2.44 GB, 15 s / 3 GB).

Depend on it for `Transform3`, `Projective3`, `Affine3`, `Perspective3`, `Orthographic3` (the general homogeneous transforms run the methods of dimension 4).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
