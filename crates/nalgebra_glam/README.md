# nalgebra_glam

Conversions between [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo) and
[glam-cairo](https://github.com/bal7hazar/glam-cairo) types: the glam interop of the Rust
[nalgebra](https://nalgebra.rs) crate (`third_party/glam`, behind its `convert-glam0XX` features),
as a separate package because Scarb has no optional dependencies and `glam` costs every dependent
of `nalgebra` compile time and memory.

```toml
[dependencies]
nalgebra = "0.1.0"
nalgebra_glam = "0.1.0"
glam = "0.4.0"
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
has (91 items of [`docs/API_PARITY.md`](../../docs/API_PARITY.md)), one module per upstream file:

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

MIT licensed.
