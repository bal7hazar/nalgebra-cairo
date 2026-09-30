# nalgebra_types3

The types of dimension 3 (shapes with max(rows, cols) = 3, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls); Point3, Translation3, Perm3, Rotation3, Reflection3 (with its constructors and accessors).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- the shapes with max(rows, columns) = 3 (`Matrix3`, `Matrix2x3`, `Matrix3x2`, `Vector3`, `RowVector3`...), their core-trait impls, products, indexing and solve impls;
- `Point3`, `Translation3`, `Rotation3`, `Reflection3` (with its constructors and accessors), `Perm3` and the permutation / LU-step / Householder impls of dimension 3.

Upstream's module paths are kept: `nalgebra_types3::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md) and `simba`.

## When to depend on it

It is also pulled in by 26 other declared closures, the cheapest being `core_pivot` (3.5 s / 1.04 GB, 15 s / 3 GB).

Depend on the `types` crates when you need the types and their operators without the named methods (those are in the `static` crates). Dimension k builds on every dimension below it: this crate pulls [`nalgebra_types2`](../types2/README.md).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`, `internal::geometry`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
