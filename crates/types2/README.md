# nalgebra_types2

The types of dimension 2 (shapes with max(rows, cols) = 2, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls); Point2, Translation2, Perm2, Rotation2, Reflection2 (with its constructors and accessors), GivensRotation.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::matrix2`, `base::vector2`, `base::row_vector2`, `base::point2`: the shapes with max(rows, columns) = 2 (`Matrix2`, `Matrix1x2`, `Matrix2x1`, `Vector2`, `RowVector2`), their core-trait impls, products, indexing and solve impls;
- `geometry::rotation2`, `geometry::reflection2`, `geometry::translation2`: the `Rotation2`, `Reflection2` (with its constructors and accessors) and `Translation2` types; `Point2`;
- `linalg::lu`, `linalg::givens`: `Perm2`, the permutation, LU-step, Householder impls and `GivensRotation` with its methods.

Upstream's module paths are kept: `nalgebra_types2::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md) and `simba`.

## When to depend on it

It is pulled in by 27 declared closures, the cheapest being `core_pivot` (3.5 s / 1.04 GB, 15 s / 3 GB).

Depend on the `types` crates when you need the matrix / vector / point types and their operators, products and indexing, without the named methods (`norm()`, `transpose()`, `inverse()`: those are in the `static` crates). `nalgebra_types2` is the dimension-2 step.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`, `internal::geometry`, `internal::linalg`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
