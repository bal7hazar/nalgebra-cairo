# nalgebra_geometry4

The transforms, projections, scales and reflections up to dimension 4 of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- upstream's `geometry` module beyond the rigid-body types: the general, projective and affine
  transforms (`Transform2` / `Transform3`, `Projective2` / `Projective3`, `Affine2` / `Affine3`, the
  `TransformMul` / `TransformDiv` products), the projections (`Perspective3`, `Orthographic3`), the
  non-uniform scales `Scale1`..`Scale4` and the reflections `Reflection1`..`Reflection4`;
- upstream's `base/cg.rs` (homogeneous coordinates): `Matrix1CgTrait`..`Matrix5CgTrait` and their
  `...AngleTrait` forms (`new_rotation`, `look_at_rh`, `new_nonuniform_scaling`...).

The types they act on are in [`nalgebra_core`](../core/README.md) and
[`nalgebra_shapes5`](../shapes5/README.md), the methods of the shapes and the 2D / 3D rotations and
isometries in [`nalgebra_static3`](../static3/README.md) and
[`nalgebra_static4`](../static4/README.md): this crate depends on those four. Upstream's module paths
are kept: `nalgebra_geometry4::geometry::perspective3::Perspective3` is
`nalgebra::geometry::perspective3::Perspective3` (docs/SPLIT.md §3.1). The reflections of the
dimension-6 shapes (`Reflection2` on a `Matrix2x6`...) and `Matrix6CgTrait` sit with the shapes of
dimension 6, above.

## When to depend on it

`Perspective3` / `Transform3` / `Matrix4::look_at_rh` → `nalgebra_core` + `nalgebra_static3` +
`nalgebra_shapes5` + `nalgebra_static4` + `nalgebra_geometry4`.

Depend on it for cameras, projections and homogeneous transforms without the rest of the library
(the shapes of dimension 6, the decompositions, `dynamic`...). For upstream's whole API at
upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0
paths.

## License

MIT.
