//! `Isometry2` / `Isometry3` <-> tuples, translations, rotations and homogeneous matrices of glam
//! (upstream `third_party/glam/common/glam_isometry.rs`).
//!
//! - `Isometry2` <-> `(Vec2, Fixed)`: the translation and the angle of the rotation.
//!   `(tra, angle)` -> isometry is `Isometry2::new(tra, angle)` (one `sin_cos`); the reverse reads
//!   the angle back with `UnitComplex::angle` (one `atan2`, about 12 ulp, in `(-π, π]`).
//! - `Isometry2` from a `Vec2`: the pure translation (rotation exactly the identity).
//! - `Isometry3` <-> `(Vec3, Quat)`: translation and rotation. `Quat` -> isometry normalizes the
//!   quat (`UnitQuaternion::from`, see `glam_quaternion`); the reverse reads the stored components.
//! - `Isometry3` from a `Quat` (pure rotation, normalized) or a `Vec3` (pure translation).
//! - `Isometry2` -> `Mat3`, `Isometry3` -> `Mat4`: `to_homogeneous`, column-major (see
//!   `glam_matrix`).
//! - `Mat3` -> `Isometry2`, `Mat4` -> `Isometry3` (`TryInto`, upstream's `TryFrom`, whose error is
//!   `()`: `None` here): `None` unless the matrix is a rigid transform, exactly upstream's
//!   `SubsetOf<Matrix>::is_in_subset`: the bottom row is exactly `(0, .., 0, 1)` and the linear
//!   block is special orthogonal (`is_special_orthogonal`, tolerance `100` ulp: upstream's
//!   `default_epsilon() * 100` with `default_epsilon` = 1 ulp). Otherwise the isometry is read
//!   without further work: the translation is the last column and the rotation is taken from the
//!   linear block unchecked (`UnitComplex { re: m11, im: m21 }`, or Shepperd's
//!   `UnitQuaternion::from_rotation_matrix` in 3D). The bottom row is tested first and the rest
//!   runs in a function of its own: Cairo charges the most expensive path of the straight-line
//!   code of a function, so a matrix rejected by its bottom row costs the comparisons only
//!   (`bench_mat4_to_isometry3_reject`), and the result is the same in any order.
//!
//! glam-cairo has no `f64` types (`DVec*`, `DQuat`, `DMat*`): those impls stay excluded
//! (`interop`). The `Result<_, ()>` of upstream is an `Option` (Cairo form of `TryFrom`).
//!
//! Impls of `Into` / `TryInto` are found by the compiler only when in scope: `use
//! nalgebra_glam::glam_isometry::Isometry3FromVec3;` (or `use nalgebra_glam::prelude::*;`).

use fixed::{Fixed, ONE, ZERO};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use nalgebra::{
    Isometry2, Isometry2AngleTrait, Isometry2Trait, Isometry3, Isometry3Trait, Matrix2,
    Matrix2Trait, Matrix3, Matrix3Trait, Quaternion, Rotation3, Translation2, Translation3,
    UnitComplex, UnitComplexAngleTrait, UnitQuaternionTrait, Vector2, Vector3,
};
use crate::glam_matrix::{Matrix3IntoMat3, Matrix4IntoMat4};

/// The tolerance of the orthogonality check of `TryInto<Mat3 | Mat4, Isometry>`, in ulps:
/// upstream's `default_epsilon() * 100`, `default_epsilon` being 1 ulp of `Fixed`.
const ORTHOGONALITY_ULPS: u64 = 100;

/// The homogeneous matrix of an `Isometry2<Fixed>`. Upstream: `From<Isometry2<f32>> for Mat3`
/// (`to_homogeneous().into()`).
pub impl Isometry2IntoMat3 of Into<Isometry2<Fixed>, Mat3> {
    #[inline(always)]
    fn into(self: Isometry2<Fixed>) -> Mat3 {
        Isometry2Trait::to_homogeneous(self).into()
    }
}

/// The homogeneous matrix of an `Isometry3<Fixed>`. Upstream: `From<Isometry3<f32>> for Mat4`
/// (`to_homogeneous().into()`).
pub impl Isometry3IntoMat4 of Into<Isometry3<Fixed>, Mat4> {
    #[inline(always)]
    fn into(self: Isometry3<Fixed>) -> Mat4 {
        Isometry3Trait::to_homogeneous(self).into()
    }
}

/// `(translation, rotation)` of an `Isometry3<Fixed>`, the rotation as the stored quaternion.
/// Exact. Upstream: `From<Isometry3<f32>> for (Vec3, Quat)`.
pub impl Isometry3IntoVec3Quat of Into<Isometry3<Fixed>, (Vec3, Quat)> {
    #[inline(always)]
    fn into(self: Isometry3<Fixed>) -> (Vec3, Quat) {
        let t = self.translation.vector;
        let q = self.rotation.quaternion;
        (Vec3 { x: t.x, y: t.y, z: t.z }, Quat { x: q.i, y: q.j, z: q.k, w: q.w })
    }
}

/// `(translation, angle)` of an `Isometry2<Fixed>`, the angle in `(-π, π]` (one `atan2`).
/// Upstream: `From<Isometry2<f32>> for (Vec2, f32)`.
pub impl Isometry2IntoVec2Angle of Into<Isometry2<Fixed>, (Vec2, Fixed)> {
    #[inline(always)]
    fn into(self: Isometry2<Fixed>) -> (Vec2, Fixed) {
        let t = self.translation.vector;
        (Vec2 { x: t.x, y: t.y }, UnitComplexAngleTrait::angle(self.rotation))
    }
}

/// The isometry with translation `tra` and the rotation `rot` (normalized, see
/// `glam_quaternion`). Upstream: `From<(Vec3, Quat)> for Isometry3<f32>`
/// (`Isometry3::from_parts`).
pub impl Isometry3FromVec3Quat of Into<(Vec3, Quat), Isometry3<Fixed>> {
    #[inline(always)]
    fn into(self: (Vec3, Quat)) -> Isometry3<Fixed> {
        let (tra, rot) = self;
        Isometry3 {
            rotation: UnitQuaternionTrait::new_normalize(
                Quaternion { i: rot.x, j: rot.y, k: rot.z, w: rot.w },
            ),
            translation: Translation3 { vector: Vector3 { x: tra.x, y: tra.y, z: tra.z } },
        }
    }
}

/// The isometry with translation `tra` and rotation of angle `angle` (one `sin_cos`).
/// Upstream: `From<(Vec2, f32)> for Isometry2<f32>` (`Isometry2::new`).
pub impl Isometry2FromVec2Angle of Into<(Vec2, Fixed), Isometry2<Fixed>> {
    #[inline(always)]
    fn into(self: (Vec2, Fixed)) -> Isometry2<Fixed> {
        let (tra, angle) = self;
        Isometry2AngleTrait::new(Vector2 { x: tra.x, y: tra.y }, angle)
    }
}

/// The pure rotation of a `Quat` (normalized, see `glam_quaternion`). Upstream: `From<Quat> for
/// Isometry3<f32>`.
pub impl Isometry3FromQuat of Into<Quat, Isometry3<Fixed>> {
    #[inline(always)]
    fn into(self: Quat) -> Isometry3<Fixed> {
        Isometry3 {
            rotation: UnitQuaternionTrait::new_normalize(
                Quaternion { i: self.x, j: self.y, k: self.z, w: self.w },
            ),
            translation: Translation3 { vector: Vector3 { x: ZERO, y: ZERO, z: ZERO } },
        }
    }
}

/// The pure translation by a `Vec3` (rotation exactly the identity). Upstream: `From<Vec3> for
/// Isometry3<f32>`.
pub impl Isometry3FromVec3 of Into<Vec3, Isometry3<Fixed>> {
    #[inline(always)]
    fn into(self: Vec3) -> Isometry3<Fixed> {
        Isometry3 {
            rotation: UnitQuaternionTrait::identity(),
            translation: Translation3 { vector: Vector3 { x: self.x, y: self.y, z: self.z } },
        }
    }
}

/// The pure translation by a `Vec2` (rotation exactly the identity, upstream computes
/// `sin_cos(0)`). Upstream: `From<Vec2> for Isometry2<f32>` (`Isometry2::new(tra, 0.0)`).
pub impl Isometry2FromVec2 of Into<Vec2, Isometry2<Fixed>> {
    #[inline(always)]
    fn into(self: Vec2) -> Isometry2<Fixed> {
        Isometry2 {
            rotation: UnitComplex { re: ONE, im: ZERO },
            translation: Translation2 { vector: Vector2 { x: self.x, y: self.y } },
        }
    }
}

/// The rigid part of `Isometry2TryFromMat3`, run once the bottom row is known to be `(0, 0, 1)`:
/// kept out of line so that a rejected matrix costs the comparisons only (Cairo charges the most
/// expensive path of the straight-line code of a function, measured in `benches`).
#[inline(never)]
fn isometry2_of_rigid(x: Vec3, y: Vec3, z: Vec3) -> Option<Isometry2<Fixed>> {
    let rot = Matrix2 { m11: x.x, m21: x.y, m12: y.x, m22: y.y };
    if !Matrix2Trait::is_special_orthogonal(rot, ORTHOGONALITY_ULPS) {
        return Option::None;
    }
    Option::Some(
        Isometry2 {
            rotation: UnitComplex { re: x.x, im: x.y },
            translation: Translation2 { vector: Vector2 { x: z.x, y: z.y } },
        },
    )
}

/// The rigid part of `Isometry3TryFromMat4`, run once the bottom row is known to be `(0, 0, 0, 1)`
/// (out of line, see `isometry2_of_rigid`).
#[inline(never)]
fn isometry3_of_rigid(x: Vec4, y: Vec4, z: Vec4, w: Vec4) -> Option<Isometry3<Fixed>> {
    let rot = Matrix3 {
        m11: x.x, m21: x.y, m31: x.z, m12: y.x, m22: y.y, m32: y.z, m13: z.x, m23: z.y, m33: z.z,
    };
    if !Matrix3Trait::is_special_orthogonal(rot, ORTHOGONALITY_ULPS) {
        return Option::None;
    }
    Option::Some(
        Isometry3 {
            rotation: UnitQuaternionTrait::from_rotation_matrix(Rotation3 { matrix: rot }),
            translation: Translation3 { vector: Vector3 { x: w.x, y: w.y, z: w.z } },
        },
    )
}

/// `Some(isometry)` when `self` is a rigid transform (bottom row exactly `(0, 0, 1)`, linear
/// block special orthogonal within 100 ulp, see the module documentation), `None` otherwise
/// (upstream: `Err(())`). Upstream: `TryFrom<Mat3> for Isometry2<f32>`
/// (`nalgebra::try_convert`).
pub impl Isometry2TryFromMat3 of TryInto<Mat3, Isometry2<Fixed>> {
    #[inline(always)]
    fn try_into(self: Mat3) -> Option<Isometry2<Fixed>> {
        let (x, y, z) = (self.x_axis, self.y_axis, self.z_axis);
        if x.z != ZERO || y.z != ZERO || z.z != ONE {
            return Option::None;
        }
        isometry2_of_rigid(x, y, z)
    }
}

/// `Some(isometry)` when `self` is a rigid transform (bottom row exactly `(0, 0, 0, 1)`, linear
/// block special orthogonal within 100 ulp, see the module documentation), `None` otherwise
/// (upstream: `Err(())`). Upstream: `TryFrom<Mat4> for Isometry3<f32>`
/// (`nalgebra::try_convert`).
pub impl Isometry3TryFromMat4 of TryInto<Mat4, Isometry3<Fixed>> {
    #[inline(always)]
    fn try_into(self: Mat4) -> Option<Isometry3<Fixed>> {
        let (x, y, z, w) = (self.x_axis, self.y_axis, self.z_axis, self.w_axis);
        if x.w != ZERO || y.w != ZERO || z.w != ZERO || w.w != ONE {
            return Option::None;
        }
        isometry3_of_rigid(x, y, z, w)
    }
}
