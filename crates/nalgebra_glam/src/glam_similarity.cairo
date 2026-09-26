//! `Similarity2` / `Similarity3` <-> `Mat3` / `Mat4` (upstream
//! `third_party/glam/common/glam_similarity.rs`).
//!
//! - `Similarity2 -> Mat3`, `Similarity3 -> Mat4`: `to_homogeneous`, column-major (see
//!   `glam_matrix`), the scaling folded into the linear block.
//! - `Mat3 -> Similarity2`, `Mat4 -> Similarity3` (`TryInto`, upstream's `TryFrom` whose error is
//!   `()`: `None` here): upstream's `SubsetOf<Matrix>` conversion. `None` unless every column of
//!   the linear block is nonzero (`try_normalize(0)`) and the bottom row is exactly `(0, .., 0,
//!   1)`; otherwise the columns are normalized, the scaling is the MEAN of their norms
//!   (upstream: `(na + nb + nc) / 3`), the rotation is read from the normalized block without
//!   checking that it is orthogonal (upstream comments that check out: a sheared matrix is
//!   accepted, with the rotation of the Shepperd's method / first column), and when the block
//!   has a negative determinant the columns are negated and the scaling is negative, like
//!   upstream (3D: negating three columns flips the determinant, so `-s * R` is the reflection).
//!   The translation is the last column, unchanged.
//!
//! Deviations from upstream (documented, identical results wherever upstream terminates):
//!
//! - upstream runs the column normalization twice (`is_in_subset`, then
//!   `from_superset_unchecked`); it is run once here (same values, half the steps);
//! - for the 2D case upstream indexes a third column of the `2x2` block
//! (`fixed_columns_mut::<1>(2)`)
//!   and normalizes the translation column in `from_superset_unchecked`, so its `TryFrom<Mat3> for
//!   Similarity2` panics; here the mean is over the two columns of the linear block, `(na + nb) /
//!   2`.
//!
//! glam-cairo has no `f64` types (`DMat*`): those impls stay excluded (`interop`).
//!
//! Impls of `Into` / `TryInto` are found by the compiler only when in scope: `use
//! nalgebra_glam::glam_similarity::Similarity3IntoMat4;` (or `use nalgebra_glam::prelude::*;`).

use fixed::{Fixed, ONE, TWO, ZERO};
use glam::{Mat3, Mat4};
use nalgebra::{
    Isometry2, Isometry3, Matrix2, Matrix2Trait, Matrix3, Matrix3Trait, Rotation3, Similarity2,
    Similarity2Trait, Similarity3, Similarity3Trait, Translation2, Translation3, UnitComplex,
    UnitQuaternionTrait, Vector2, Vector2Trait, Vector3, Vector3Trait,
};
use crate::glam_matrix::{Matrix3IntoMat3, Matrix4IntoMat4};

/// 3 as a `Fixed`.
const THREE: Fixed = Fixed { raw: 0x300000000 };

/// The homogeneous matrix of a `Similarity2<Fixed>`. Upstream: `From<Similarity2<f32>> for Mat3`
/// (`to_homogeneous().into()`).
pub impl Similarity2IntoMat3 of Into<Similarity2<Fixed>, Mat3> {
    #[inline(always)]
    fn into(self: Similarity2<Fixed>) -> Mat3 {
        Similarity2Trait::to_homogeneous(self).into()
    }
}

/// The homogeneous matrix of a `Similarity3<Fixed>`. Upstream: `From<Similarity3<f32>> for Mat4`
/// (`to_homogeneous().into()`).
pub impl Similarity3IntoMat4 of Into<Similarity3<Fixed>, Mat4> {
    #[inline(always)]
    fn into(self: Similarity3<Fixed>) -> Mat4 {
        Similarity3Trait::to_homogeneous(self).into()
    }
}

/// `Some(similarity)` when no column of the linear block of `self` is zero, its determinant is not
/// negative and its bottom row is exactly `(0, 0, 1)`, `None` otherwise (upstream: `Err(())`); see
/// the module documentation for the scaling and the rotation. Upstream: `TryFrom<Mat3> for
/// Similarity2<f32>`
/// (`nalgebra::try_convert`).
pub impl Similarity2TryFromMat3 of TryInto<Mat3, Similarity2<Fixed>> {
    fn try_into(self: Mat3) -> Option<Similarity2<Fixed>> {
        let (x, y, z) = (self.x_axis, self.y_axis, self.z_axis);
        if x.z != ZERO || y.z != ZERO || z.z != ONE {
            return Option::None;
        }
        let mut a = Vector2 { x: x.x, y: x.y };
        let mut b = Vector2 { x: y.x, y: y.y };
        let Option::Some(na) = Vector2Trait::try_normalize_mut(ref a, ZERO) else {
            return Option::None;
        };
        let Option::Some(nb) = Vector2Trait::try_normalize_mut(ref b, ZERO) else {
            return Option::None;
        };
        let m = Matrix2 { m11: a.x, m21: a.y, m12: b.x, m22: b.y };
        if Matrix2Trait::determinant(m) < ZERO {
            return Option::None;
        }
        let scaling = (na + nb) / TWO;
        Option::Some(
            Similarity2 {
                isometry: Isometry2 {
                    rotation: UnitComplex { re: m.m11, im: m.m21 },
                    translation: Translation2 { vector: Vector2 { x: z.x, y: z.y } },
                },
                scaling,
            },
        )
    }
}

/// `Some(similarity)` when no column of the linear block of `self` is zero and its bottom row is
/// exactly `(0, 0, 0, 1)`, `None` otherwise (upstream: `Err(())`); see the module documentation
/// for the scaling and the rotation. Upstream: `TryFrom<Mat4> for Similarity3<f32>`
/// (`nalgebra::try_convert`).
pub impl Similarity3TryFromMat4 of TryInto<Mat4, Similarity3<Fixed>> {
    fn try_into(self: Mat4) -> Option<Similarity3<Fixed>> {
        let (x, y, z, w) = (self.x_axis, self.y_axis, self.z_axis, self.w_axis);
        if x.w != ZERO || y.w != ZERO || z.w != ZERO || w.w != ONE {
            return Option::None;
        }
        let mut a = Vector3 { x: x.x, y: x.y, z: x.z };
        let mut b = Vector3 { x: y.x, y: y.y, z: y.z };
        let mut c = Vector3 { x: z.x, y: z.y, z: z.z };
        let Option::Some(na) = Vector3Trait::try_normalize_mut(ref a, ZERO) else {
            return Option::None;
        };
        let Option::Some(nb) = Vector3Trait::try_normalize_mut(ref b, ZERO) else {
            return Option::None;
        };
        let Option::Some(nc) = Vector3Trait::try_normalize_mut(ref c, ZERO) else {
            return Option::None;
        };
        let mut m = Matrix3 {
            m11: a.x,
            m21: a.y,
            m31: a.z,
            m12: b.x,
            m22: b.y,
            m32: b.z,
            m13: c.x,
            m23: c.y,
            m33: c.z,
        };
        let mut scaling = (na + nb + nc) / THREE;
        if Matrix3Trait::determinant(m) < ZERO {
            m = -m;
            scaling = -scaling;
        }
        Option::Some(
            Similarity3 {
                isometry: Isometry3 {
                    rotation: UnitQuaternionTrait::from_rotation_matrix(Rotation3 { matrix: m }),
                    translation: Translation3 { vector: Vector3 { x: w.x, y: w.y, z: w.z } },
                },
                scaling,
            },
        )
    }
}
