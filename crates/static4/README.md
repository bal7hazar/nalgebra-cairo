# nalgebra_static4

The methods of the dimension-4 shapes (Matrix4, Vector4, Matrix2x4, Matrix4x3...).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- the shapes of dimension 4 (`Matrix4`, `Vector4`, `RowVector4`, `Matrix2x4`, `Matrix4x3`...): one inherent trait per shape (`Matrix4Trait`...), with the constructors, accessors, products, determinant / inverse, norms, edition and the methods that take a closure (feature `closures`).

Upstream's module paths are kept: `nalgebra_static4::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_static2`](../static2/README.md), [`nalgebra_static3`](../static3/README.md) and `simba`.

Feature `closures` (on by default, forwarded by the facade's feature of the same name): the methods that take a closure (`map`, `fold`, `apply`, `zip_*`...).

## When to depend on it

`nalgebra_static4` is a member of the declared closures `static4_svd` (6.9 s / 2.11 GB, 15 s / 3 GB), `static4_geometry` (11.2 s / 2.44 GB, 15 s / 3 GB), `static4_factor` (8.9 s / 2.06 GB, 15 s / 3 GB), `static4_statistics` (8.6 s / 1.91 GB, 15 s / 3 GB), `static4_statistics_all` (11.9 s / 2.79 GB, 15 s / 3 GB), `static4_blas` (12.4 s / 2.88 GB, 15 s / 3 GB), `static4_blocks` (17.4 s / 3.87 GB, documented, not gated), `static4_views` (17.9 s / 4.18 GB, documented, not gated), `static4_norm` (18.7 s / 4.04 GB, documented, not gated). It is also pulled in by 13 other declared closures, the cheapest being `static5_pivot` (9.4 s / 3.26 GB, 20 s / 4.5 GB).

Depend on the `static` crates for the named methods of the shapes (`norm()`, `normalize()`, `dot`, `cross`, `transpose`, `inverse`, `insert_*` / `remove_*`...). This is the top of the everyday range (dimensions 2-4).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
