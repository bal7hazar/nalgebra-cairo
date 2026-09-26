//! `Rotation2` <-> `Mat2` and `Rotation3` <-> `Quat` (upstream
//! `third_party/glam/common/glam_rotation.rs`).
//!
//! - `Rotation2` -> `Mat2`: the rotation matrix, exact (glam's axes are its columns).
//! - `Mat2` -> `Rotation2`: `UnitComplex::from(mat).to_rotation_matrix()`: the first column of
//!   the matrix is normalized (see `glam_unit_complex`), so a scaled or sheared `Mat2` gives the
//!   nearest rotation about its first column; panics with `Fixed: division by zero` when that
//!   column is zero (upstream: NaN).
//! - `Rotation3` -> `Quat`: `UnitQuaternion::from(rotation)` (`from_rotation_matrix`, Shepperd's
//!   method, one square root and three divisions) into a `Quat`.
//! - `Quat` -> `Rotation3`: `Rotation3::from(UnitQuaternion::from(quat))`: the quat is normalized
//!   (`new_normalize`; panics with `Fixed: division by zero` on the zero quat), then expanded into
//!   a rotation matrix (`to_rotation_matrix`).
//!
//! glam-cairo has no `DMat2` / `DQuat` (`f64`): those impls stay excluded (`interop`).
//!
//! Impls of `Into` are found by the compiler only when in scope: `use
//! nalgebra_glam::glam_rotation::Rotation2FromMat2;` (or `use nalgebra_glam::prelude::*;`).

use fixed::Fixed;
use glam::{Mat2, Quat};
use nalgebra::{Quaternion, Rotation2, Rotation3, UnitComplexTrait, UnitQuaternionTrait, Vector2};
use crate::glam_matrix::Matrix2IntoMat2;

/// The rotation matrix of a `Rotation2<Fixed>` as a `Mat2` (its columns are the axes). Exact.
/// Upstream: `From<Rotation2<f32>> for Mat2` (`into_inner().into()`).
pub impl Rotation2IntoMat2 of Into<Rotation2<Fixed>, Mat2> {
    #[inline(always)]
    fn into(self: Rotation2<Fixed>) -> Mat2 {
        self.matrix.into()
    }
}

/// The rotation about the first column of a `Mat2` (normalized), as a matrix. Upstream:
/// `From<Mat2> for Rotation2<f32>` (`UnitComplex::from(e).to_rotation_matrix()`).
pub impl Rotation2FromMat2 of Into<Mat2, Rotation2<Fixed>> {
    #[inline(always)]
    fn into(self: Mat2) -> Rotation2<Fixed> {
        UnitComplexTrait::to_rotation_matrix(
            UnitComplexTrait::new_normalize(Vector2 { x: self.x_axis.x, y: self.x_axis.y }),
        )
    }
}

/// The `Quat` of the rotation of a `Rotation3<Fixed>`: `UnitQuaternion::from_rotation_matrix`,
/// components `(x, y, z, w) = (i, j, k, w)`. The sign is upstream's (`q` and `-q` are the same
/// rotation). Upstream: `From<Rotation3<f32>> for Quat` (`UnitQuaternion::from(e).into()`).
pub impl Rotation3IntoQuat of Into<Rotation3<Fixed>, Quat> {
    #[inline(always)]
    fn into(self: Rotation3<Fixed>) -> Quat {
        let q = UnitQuaternionTrait::from_rotation_matrix(self).quaternion;
        Quat { x: q.i, y: q.j, z: q.k, w: q.w }
    }
}

/// The rotation matrix of the normalized `Quat` (see the module documentation). Upstream:
/// `From<Quat> for Rotation3<f32>` (`Rotation3::from(UnitQuaternion::from(e))`).
pub impl Rotation3FromQuat of Into<Quat, Rotation3<Fixed>> {
    #[inline(always)]
    fn into(self: Quat) -> Rotation3<Fixed> {
        UnitQuaternionTrait::to_rotation_matrix(
            UnitQuaternionTrait::new_normalize(
                Quaternion { i: self.x, j: self.y, k: self.z, w: self.w },
            ),
        )
    }
}
