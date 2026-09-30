# nalgebra_static2

The methods of the dimension-2 shapes (Matrix2, Vector2, RowVector2, Matrix1x2...).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::matrix2`, `base::vector2`, `base::row_vector2` and the non-square shapes of dimension 2 (`Matrix1x2`, `Matrix2x1`): one inherent trait per shape (`Matrix2Trait`, `Vector2Trait`...), with the constructors, accessors, products, determinant / inverse, norms, edition and the methods that take a closure (feature `closures`).

Upstream's module paths are kept: `nalgebra_static2::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md) and `simba`.

Feature `closures` (on by default, forwarded by the facade's feature of the same name): the methods that take a closure (`map`, `fold`, `apply`, `zip_*`...).

## When to depend on it

It is pulled in by 26 declared closures, the cheapest being `static3_svd` (5.8 s / 1.23 GB, 15 s / 3 GB).

Depend on the `static` crates for the named methods of the shapes (`norm()`, `normalize()`, `dot`, `cross`, `transpose`, `inverse`, `insert_*` / `remove_*`...). `static2` is the dimension-2 step.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
