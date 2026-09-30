# nalgebra_static3

The methods of the dimension-3 shapes (Matrix3, Vector3, Matrix2x3, Matrix3x2...).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::matrix3`, `base::vector3`, `base::row_vector3`, `base::matrix2x3`, `base::matrix3x2` and `base::unit`: one inherent trait per shape (`Matrix3Trait`, `Vector3Trait` with `cross`...), with the constructors, accessors, products, determinant / inverse, norms, edition and the methods that take a closure (feature `closures`).

Upstream's module paths are kept: `nalgebra_static3::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_static2`](../static2/README.md) and `simba`.

Feature `closures` (on by default, forwarded by the facade's feature of the same name): the methods that take a closure (`map`, `fold`, `apply`, `zip_*`...).

## When to depend on it

`nalgebra_static3` is a member of the declared closures `static3_svd` (5.8 s / 1.23 GB, 15 s / 3 GB), `static3_geometry` (6.3 s / 1.43 GB, 15 s / 3 GB). It is also pulled in by 23 other declared closures, the cheapest being `static4_svd` (6.9 s / 2.11 GB, 15 s / 3 GB).

Depend on the `static` crates for the named methods of the shapes (`norm()`, `normalize()`, `dot`, `cross`, `transpose`, `inverse`, `insert_*` / `remove_*`...). Dimension k builds on every dimension below it.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
