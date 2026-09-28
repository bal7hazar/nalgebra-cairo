# nalgebra_static3

Includes the 2D / 3D geometry: rotations, unit quaternions, isometries, similarities. A sub-crate of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust
`nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- the METHODS of the static shapes up to 3x3: `Matrix1Trait`, `Matrix2Trait`, `Matrix2x3Trait`,
  `Matrix3x2Trait`, `Matrix3Trait`, `Vector2Trait`, `Vector3Trait`, `RowVector2Trait`,
  `RowVector3Trait` (and their `...AngleTrait` forms: trigonometry, `exp`, `ln`...), the swizzles of
  `Vector2` / `Vector3`;
- the 2D / 3D geometry of upstream's `geometry` module: `Quaternion`, `UnitQuaternion`,
  `UnitComplex`, `Rotation2`, `Rotation3`, `Translation2`, `Translation3`, `Isometry2`, `Isometry3`,
  `IsometryMatrix2`, `IsometryMatrix3`, `Similarity2`, `Similarity3`, `SimilarityMatrix2`,
  `SimilarityMatrix3`, `DualQuaternion`, `UnitDualQuaternion`, `AbstractRotation`, and the
  completion of `Point2` / `Point3` (`Point2ExtTrait`, `Point3ExtTrait`);
- the conversions and products between those types and the shapes (`Rotation3` into `Matrix3` /
  `Matrix4`, `RowVector2 * Rotation2`...), in the module of the geometry type.

The types themselves (`Matrix3`, `Vector3`, `Point3`, `Unit`...) are in
[`nalgebra_core`](../core/README.md), which this crate depends on. Upstream's module paths are kept:
`nalgebra_static3::geometry::unit_quaternion::UnitQuaternion` is
`nalgebra::geometry::unit_quaternion::UnitQuaternion`, and `nalgebra_static3::base::matrix3::Matrix3Trait`
is `nalgebra::base::matrix3::Matrix3Trait` (docs/SPLIT.md §3.1).

## When to depend on it

`UnitQuaternion` / `Rotation3` / `Isometry3` → `nalgebra_core` + `nalgebra_static3`.

Depend on `nalgebra_core` + `nalgebra_static3` when you need the methods of the shapes up to 3x3 or
the 2D / 3D rotations and rigid-body transformations without the rest of the library (the shapes of
dimension 4 to 6, the decompositions, `dynamic`...): it is the build a 2D / 3D physics engine
needs, and much cheaper than the facade. For upstream's whole API at upstream's paths, depend on
the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0 paths.

## Features

`closures` (default): the methods of the shapes that take a closure (`map`, `fold`, `apply`,
`zip_*`...), as in `nalgebra` (whose feature of the same name forwards to this one).

## `internal`

`nalgebra_static3::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_static3` use (the crate-private kernels of `Matrix2` / `Matrix3` / `Vector3`
used by the decompositions, and of the rotations used by `nalgebra`'s in-crate tests). They are
public only for those crates: **internal, no stability promise**, they may change or disappear in
any release, and the facade never re-exports them.

## License

MIT.
