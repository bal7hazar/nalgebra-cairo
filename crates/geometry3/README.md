# nalgebra_geometry3

The 3D geometry: quaternions, unit quaternions, Rotation3 methods, isometries, similarities, dual quaternions, AbstractRotation, point / translation / scale 3 methods, Matrix4 homogeneous (cg).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `geometry::quaternion`, `geometry::unit_quaternion`, `geometry::rotation3` (methods), `geometry::dual_quaternion`, `geometry::unit_dual_quaternion`, `geometry::isometry3`, `geometry::isometry_matrix3`, `geometry::similarity3`, `geometry::similarity_matrix3`, `geometry::abstract_rotation`: the 3D rotations and rigid / similarity transforms;
- `geometry::translation3`, `geometry::scale3`, `base::point3`, `base::point_swizzle`: the methods of points, translations and scales of dimension 3;
- `base::cg`: the homogeneous-coordinate methods of `Matrix4`.

Upstream's module paths are kept: `nalgebra_geometry3::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_static3`](../static3/README.md), [`nalgebra_geometry2`](../geometry2/README.md) and `simba`.

## When to depend on it

`nalgebra_geometry3` is a member of the declared closures `static4_geometry` (11.2 s / 2.44 GB, 15 s / 3 GB), `static3_geometry` (6.3 s / 1.43 GB, 15 s / 3 GB). It is also pulled in by `nalgebra_glam` (7.4 s / 1.77 GB, 15 s / 3 GB).

Depend on it for the 3D geometry (quaternions, `Rotation3`, `Isometry3`, `Similarity3`, dual quaternions). It depends on `static3` but not on dimension 4 or above.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`, `internal::geometry`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
