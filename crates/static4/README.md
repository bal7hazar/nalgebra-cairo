# nalgebra_static4

The methods of the dimension-4 shapes of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- the METHODS of the static shapes of dimension 4: `Matrix4Trait`, `Matrix2x4Trait`,
  `Matrix3x4Trait`, `Matrix4x2Trait`, `Matrix4x3Trait`, `Vector4Trait`, `RowVector4Trait`,
  `UnitVector4Trait` and their `...AngleTrait` forms (trigonometry, `exp`, `ln`...): constructors,
  component-wise operations, reductions, norms, interpolation, inverse and determinant of
  `Matrix4`, the base completion of upstream's `Matrix` / `Vector` API.

The types themselves (`Matrix4`, `Vector4`...) are in [`nalgebra_core`](../core/README.md), those of
dimension 5 (which some methods return) in [`nalgebra_shapes5`](../shapes5/README.md); this crate
depends on both and on `nalgebra_static3`. Upstream's module paths are kept:
`nalgebra_static4::base::matrix4::Matrix4Trait` is `nalgebra::base::matrix4::Matrix4Trait`
(docs/SPLIT.md §3.1).

## When to depend on it

Depend on `nalgebra_core` + `nalgebra_static3` + `nalgebra_shapes5` + `nalgebra_static4` when you
need the methods of the shapes up to 4x4 (4x4 homogeneous transforms, 4-vectors) without the rest of
the library; add `nalgebra_geometry4` for the transforms and projections. For upstream's whole API
at upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0
paths.

## Features

`closures` (default): the methods of the shapes that take a closure (`map`, `fold`, `apply`,
`zip_*`...), as in `nalgebra` (whose feature of the same name forwards to this one).

## `internal`

`nalgebra_static4::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_static4` use (the crate-private kernels of `Matrix4` used by the
decompositions). They are public only for those crates: **internal, no stability promise**, they may
change or disappear in any release, and the facade never re-exports them.

## License

MIT.
