# nalgebra_transform2

Transform2, Projective2, Affine2 and their products (upstream geometry::transform).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `geometry::transform2`, `geometry::projective2`, `geometry::affine2`: `Transform2`, `Projective2`, `Affine2`, their operators and their products (upstream `geometry::transform`).

Upstream's module paths are kept: `nalgebra_transform2::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_static2`](../static2/README.md), [`nalgebra_static3`](../static3/README.md), [`nalgebra_geometry2`](../geometry2/README.md) and `simba`.

## When to depend on it

`nalgebra_transform2` is a member of the declared closure `static4_geometry` (11.2 s / 2.44 GB, 15 s / 3 GB).

Depend on it for `Transform2`, `Projective2`, `Affine2` (the general homogeneous transforms run the methods of dimension 3).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::geometry`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
