# nalgebra_glam

Conversions between [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo) and
[glam-cairo](https://github.com/bal7hazar/glam-cairo) types: the glam interop of the Rust
[nalgebra](https://nalgebra.rs) crate (`third_party/glam`, behind its `convert-glam0XX` features),
as a separate package because Scarb has no optional dependencies and `glam` costs every dependent
of `nalgebra` compile time and memory.

It depends on the sub-crates of nalgebra-cairo that define the types it converts:
[`nalgebra_core`](../core/README.md) (`Unit`, the dimension-1 types),
[`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md) and
[`nalgebra_types4`](../types4/README.md) (the shapes, `Point2..4`, `Translation2..4`, `Rotation2..3`),
[`nalgebra_static2`](../static2/README.md) and [`nalgebra_static3`](../static3/README.md) (their
methods up to 3x3) and [`nalgebra_geometry2`](../geometry2/README.md) and
[`nalgebra_geometry3`](../geometry3/README.md) (the 2D / 3D geometry: rotations, quaternions,
isometries, similarities), and on glam-cairo's `glam_core` 0.4.1 (every glam type it converts and
every impl it needs; it needs **`glam` >= 0.4.1**, which re-exports `glam_core`), not on the
`nalgebra` / `glam` facades: a much cheaper build. The types are the same as the facades' (`nalgebra` and `glam`
re-export them, glam from 0.4.1 on), so a dependent of the facades uses it unchanged; glam 0.4.0
defines its own types, so a lock file pinned to it needs `scarb update glam` (0.4.1):

```toml
[dependencies]
nalgebra = "0.1.1"          # or the sub-crates above
nalgebra_glam = "0.1.1"
glam = "0.4.1"              # or glam_core
fixed = "0.4.0"
```

Upstream's `From<X> for Y` is `Into<X, Y>` here, `TryFrom` is `TryInto` (returning an `Option`), and
the impls are found by the compiler only when in scope:

```cairo
use fixed::Fixed;
use glam::{Mat4, Quat, Vec3};
use nalgebra::{Isometry3, Vector3};
use nalgebra_glam::prelude::*;

let v: Vector3<Fixed> = vec3.into();                    // Vec3 -> Vector3<Fixed>
let iso: Isometry3<Fixed> = (vec3, quat).into();        // (Vec3, Quat) -> Isometry3
let mat: Mat4 = iso.into();                             // Isometry3 -> Mat4 (column-major)
let back: Option<Isometry3<Fixed>> = mat.try_into();    // None unless the matrix is rigid
```

## Coverage

Every conversion of upstream's `glam_matrix`, `glam_point`, `glam_translation`, `glam_quaternion`,
`glam_rotation`, `glam_unit_complex`, `glam_isometry` and `glam_similarity` for the types glam-cairo
has (91 items of [`docs/API_PARITY.md`](https://github.com/bal7hazar/nalgebra-cairo/blob/main/docs/API_PARITY.md)), one module per upstream file:

| glam-cairo | nalgebra-cairo |
|---|---|
| `Vec2..4` (`Fixed`) | `Vector2..4<Fixed>`, `Point2..4<Fixed>`, `Translation2..4<Fixed>`, `UnitVector2..4<Fixed>` (`TryInto`) |
| `IVec2..4` (`i32`), `UVec2..4` (`u32`), `BVec2..4` (`bool`) | `Vector2..4<T>`, `Point2..4<T>` |
| `Mat2..4` | `Matrix2..4<Fixed>`; `Mat2` also `Rotation2<Fixed>`, `UnitComplex<Fixed>`; `Mat3` / `Mat4` from `Isometry2..3`, `Similarity2..3` |
| `Quat` | `Quaternion<Fixed>`, `UnitQuaternion<Fixed>`, `Rotation3<Fixed>`, `Isometry3<Fixed>` |
| `(Vec2, Fixed)`, `(Vec3, Quat)` | `Isometry2<Fixed>`, `Isometry3<Fixed>` |

glam-cairo's `Mat2..4` are column-major (`x_axis` is the first COLUMN) and `Quat` is `(x, y, z, w)`
where nalgebra's `Quaternion::new` takes `(w, i, j, k)`: the conversions map them exactly. The types
glam-cairo does not have (`f64` `D*`, aligned `Vec3A`, other integer widths) have no impls.

Deviations from upstream (`Result` is an `Option`, zero-length inputs panic instead of giving NaN, the
2D `Similarity` conversion works where upstream panics) are documented in each module.

## When to depend on it

Depend on `nalgebra_glam` when you convert between nalgebra-cairo and glam-cairo types. It is the
declared closure `nalgebra_glam` (7.4 s / 1.77 GB on a GitHub runner over an empty consumer, budget
15 s / 3 GB; the reference `glam_core` 0.4.1 alone is 1.3 s / 0.56 GB), and it does not pull dimension
5 or 6. The facade [`nalgebra`](../../README.md) does not depend on it: add it next to the facade.

## `internal`

This crate has no `internal` module.

## License

MIT.
