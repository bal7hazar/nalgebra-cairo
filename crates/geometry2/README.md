# nalgebra_geometry2

The 1D / 2D geometry: UnitComplex, Rotation2 methods, isometries, similarities, points / translations / scales 1-2 methods, swizzles, Matrix2 / Matrix3 homogeneous (cg).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `geometry::unit_complex`, `geometry::rotation2` (methods), `geometry::isometry2`, `geometry::isometry_matrix2`, `geometry::similarity2`, `geometry::similarity_matrix2`: `UnitComplex`, `Isometry2`, `IsometryMatrix2`, `Similarity2`, `SimilarityMatrix2`;
- `geometry::point`, `geometry::point1`, `geometry::translation1`, `geometry::translation2`, `geometry::scale1`, `geometry::scale2`, `base::point2`, `base::point_swizzle`: the methods of points, translations and scales of dimension 1-2, and the swizzles;
- `base::cg`: the homogeneous-coordinate methods of `Matrix2` / `Matrix3`.

Upstream's module paths are kept: `nalgebra_geometry2::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md) and `simba`.

## When to depend on it

`nalgebra_geometry2` is a member of the declared closures `static4_geometry` (11.2 s / 2.44 GB, 15 s / 3 GB), `static3_geometry` (6.3 s / 1.43 GB, 15 s / 3 GB). It is also pulled in by `nalgebra_glam` (7.4 s / 1.77 GB, 15 s / 3 GB).

Depend on it for the 2D geometry (`Rotation2`, `UnitComplex`, `Isometry2`, `Similarity2`) and the 1D / 2D points, translations and scales.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`, `internal::geometry`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
