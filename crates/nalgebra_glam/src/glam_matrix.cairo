//! `Vector2..4`, `UnitVector2..4` and `Matrix2..4` <-> the glam vectors and matrices (upstream
//! `third_party/glam/common/glam_matrix.rs`).
//!
//! # Component types
//!
//! glam-cairo's `Vec2..4` hold `fixed::Fixed`, so they convert to `Vector2..4<Fixed>` (upstream:
//! `f32`); `IVec2..4` hold `i32` (`Vector2..4<i32>`), `UVec2..4` hold `u32` (`Vector2..4<u32>`) and
//! `BVec2..4` hold `bool` (`Vector2..4<bool>`), like upstream. glam-cairo has no `f64` vectors
//! (`DVec*`, `DMat*`), no aligned `Vec3A` and no other integer widths: those impls stay excluded
//! (`interop`).
//!
//! # Layout
//!
//! Vectors convert component by component, in order (`x, y, z, w`), exact. glam's `Mat2..4` are
//! column-major (`x_axis, y_axis, ...` are the columns): `Mat3::x_axis` is the first COLUMN of
//! the `Matrix3` (`m11, m21, m31`), not its first row (DESIGN D8; upstream converts through
//! `to_cols_array_2d` / `from_cols`). Exact, no arithmetic.
//!
//! # Unit vectors
//!
//! `TryInto<Vec3, UnitVector3<Fixed>>` is upstream's `TryFrom<Vec3> for UnitVector3<f32>`:
//! `Unit::try_new(v, 0)`, so `None` (upstream: `Err("Normalization failed.")`, the Cairo form of
//! a `Result` is an `Option`) when the norm is zero, and otherwise the vector divided by its norm
//! (`Unit::try_new` rounding: norm floored once, one correctly rounded division per component).
//! The reverse conversion returns the wrapped vector unchanged.
//!
//! Impls of `Into` / `TryInto` are found by the compiler only when in scope: `use
//! nalgebra_glam::glam_matrix::Vector3FromVec3;` (or the whole prelude, `use
//! nalgebra_glam::prelude::*;`).

use fixed::Fixed;
use glam::{
    BVec2, BVec3, BVec4, IVec2, IVec3, IVec4, Mat2, Mat3, Mat4, UVec2, UVec3, UVec4, Vec2, Vec3,
    Vec4,
};
use nalgebra::{
    Matrix2, Matrix3, Matrix4, UnitTrait, UnitVector2, UnitVector3, UnitVector4, Vector2, Vector3,
    Vector4,
};

/// The `Vector2<Fixed>` with the components of `Vec2`, in order (`x, y`). Exact.
/// Upstream: `From<Vec2> for Vector2<f32>`.
pub impl Vector2FromVec2 of Into<Vec2, Vector2<Fixed>> {
    #[inline(always)]
    fn into(self: Vec2) -> Vector2<Fixed> {
        Vector2 { x: self.x, y: self.y }
    }
}

/// The `Vec2` with the components of `Vector2<Fixed>`, in order. Exact. Upstream:
/// `From<Vector2<f32>> for Vec2`.
pub impl Vector2IntoVec2 of Into<Vector2<Fixed>, Vec2> {
    #[inline(always)]
    fn into(self: Vector2<Fixed>) -> Vec2 {
        Vec2 { x: self.x, y: self.y }
    }
}

/// The `Vector3<Fixed>` with the components of `Vec3`, in order (`x, y, z`). Exact.
/// Upstream: `From<Vec3> for Vector3<f32>`.
pub impl Vector3FromVec3 of Into<Vec3, Vector3<Fixed>> {
    #[inline(always)]
    fn into(self: Vec3) -> Vector3<Fixed> {
        Vector3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Vec3` with the components of `Vector3<Fixed>`, in order. Exact. Upstream:
/// `From<Vector3<f32>> for Vec3`.
pub impl Vector3IntoVec3 of Into<Vector3<Fixed>, Vec3> {
    #[inline(always)]
    fn into(self: Vector3<Fixed>) -> Vec3 {
        Vec3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Vector4<Fixed>` with the components of `Vec4`, in order (`x, y, z, w`). Exact.
/// Upstream: `From<Vec4> for Vector4<f32>`.
pub impl Vector4FromVec4 of Into<Vec4, Vector4<Fixed>> {
    #[inline(always)]
    fn into(self: Vec4) -> Vector4<Fixed> {
        Vector4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `Vec4` with the components of `Vector4<Fixed>`, in order. Exact. Upstream:
/// `From<Vector4<f32>> for Vec4`.
pub impl Vector4IntoVec4 of Into<Vector4<Fixed>, Vec4> {
    #[inline(always)]
    fn into(self: Vector4<Fixed>) -> Vec4 {
        Vec4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `Vector2<i32>` with the components of `IVec2`, in order (`x, y`). Exact.
/// Upstream: `From<IVec2> for Vector2<i32>`.
pub impl Vector2FromIVec2 of Into<IVec2, Vector2<i32>> {
    #[inline(always)]
    fn into(self: IVec2) -> Vector2<i32> {
        Vector2 { x: self.x, y: self.y }
    }
}

/// The `IVec2` with the components of `Vector2<i32>`, in order. Exact. Upstream:
/// `From<Vector2<i32>> for IVec2`.
pub impl Vector2IntoIVec2 of Into<Vector2<i32>, IVec2> {
    #[inline(always)]
    fn into(self: Vector2<i32>) -> IVec2 {
        IVec2 { x: self.x, y: self.y }
    }
}

/// The `Vector3<i32>` with the components of `IVec3`, in order (`x, y, z`). Exact.
/// Upstream: `From<IVec3> for Vector3<i32>`.
pub impl Vector3FromIVec3 of Into<IVec3, Vector3<i32>> {
    #[inline(always)]
    fn into(self: IVec3) -> Vector3<i32> {
        Vector3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `IVec3` with the components of `Vector3<i32>`, in order. Exact. Upstream:
/// `From<Vector3<i32>> for IVec3`.
pub impl Vector3IntoIVec3 of Into<Vector3<i32>, IVec3> {
    #[inline(always)]
    fn into(self: Vector3<i32>) -> IVec3 {
        IVec3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Vector4<i32>` with the components of `IVec4`, in order (`x, y, z, w`). Exact.
/// Upstream: `From<IVec4> for Vector4<i32>`.
pub impl Vector4FromIVec4 of Into<IVec4, Vector4<i32>> {
    #[inline(always)]
    fn into(self: IVec4) -> Vector4<i32> {
        Vector4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `IVec4` with the components of `Vector4<i32>`, in order. Exact. Upstream:
/// `From<Vector4<i32>> for IVec4`.
pub impl Vector4IntoIVec4 of Into<Vector4<i32>, IVec4> {
    #[inline(always)]
    fn into(self: Vector4<i32>) -> IVec4 {
        IVec4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `Vector2<u32>` with the components of `UVec2`, in order (`x, y`). Exact.
/// Upstream: `From<UVec2> for Vector2<u32>`.
pub impl Vector2FromUVec2 of Into<UVec2, Vector2<u32>> {
    #[inline(always)]
    fn into(self: UVec2) -> Vector2<u32> {
        Vector2 { x: self.x, y: self.y }
    }
}

/// The `UVec2` with the components of `Vector2<u32>`, in order. Exact. Upstream:
/// `From<Vector2<u32>> for UVec2`.
pub impl Vector2IntoUVec2 of Into<Vector2<u32>, UVec2> {
    #[inline(always)]
    fn into(self: Vector2<u32>) -> UVec2 {
        UVec2 { x: self.x, y: self.y }
    }
}

/// The `Vector3<u32>` with the components of `UVec3`, in order (`x, y, z`). Exact.
/// Upstream: `From<UVec3> for Vector3<u32>`.
pub impl Vector3FromUVec3 of Into<UVec3, Vector3<u32>> {
    #[inline(always)]
    fn into(self: UVec3) -> Vector3<u32> {
        Vector3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `UVec3` with the components of `Vector3<u32>`, in order. Exact. Upstream:
/// `From<Vector3<u32>> for UVec3`.
pub impl Vector3IntoUVec3 of Into<Vector3<u32>, UVec3> {
    #[inline(always)]
    fn into(self: Vector3<u32>) -> UVec3 {
        UVec3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Vector4<u32>` with the components of `UVec4`, in order (`x, y, z, w`). Exact.
/// Upstream: `From<UVec4> for Vector4<u32>`.
pub impl Vector4FromUVec4 of Into<UVec4, Vector4<u32>> {
    #[inline(always)]
    fn into(self: UVec4) -> Vector4<u32> {
        Vector4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `UVec4` with the components of `Vector4<u32>`, in order. Exact. Upstream:
/// `From<Vector4<u32>> for UVec4`.
pub impl Vector4IntoUVec4 of Into<Vector4<u32>, UVec4> {
    #[inline(always)]
    fn into(self: Vector4<u32>) -> UVec4 {
        UVec4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `Vector2<bool>` with the components of `BVec2`, in order (`x, y`). Exact.
/// Upstream: `From<BVec2> for Vector2<bool>`.
pub impl Vector2FromBVec2 of Into<BVec2, Vector2<bool>> {
    #[inline(always)]
    fn into(self: BVec2) -> Vector2<bool> {
        Vector2 { x: self.x, y: self.y }
    }
}

/// The `BVec2` with the components of `Vector2<bool>`, in order. Exact. Upstream:
/// `From<Vector2<bool>> for BVec2`.
pub impl Vector2IntoBVec2 of Into<Vector2<bool>, BVec2> {
    #[inline(always)]
    fn into(self: Vector2<bool>) -> BVec2 {
        BVec2 { x: self.x, y: self.y }
    }
}

/// The `Vector3<bool>` with the components of `BVec3`, in order (`x, y, z`). Exact.
/// Upstream: `From<BVec3> for Vector3<bool>`.
pub impl Vector3FromBVec3 of Into<BVec3, Vector3<bool>> {
    #[inline(always)]
    fn into(self: BVec3) -> Vector3<bool> {
        Vector3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `BVec3` with the components of `Vector3<bool>`, in order. Exact. Upstream:
/// `From<Vector3<bool>> for BVec3`.
pub impl Vector3IntoBVec3 of Into<Vector3<bool>, BVec3> {
    #[inline(always)]
    fn into(self: Vector3<bool>) -> BVec3 {
        BVec3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Vector4<bool>` with the components of `BVec4`, in order (`x, y, z, w`). Exact.
/// Upstream: `From<BVec4> for Vector4<bool>`.
pub impl Vector4FromBVec4 of Into<BVec4, Vector4<bool>> {
    #[inline(always)]
    fn into(self: BVec4) -> Vector4<bool> {
        Vector4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `BVec4` with the components of `Vector4<bool>`, in order. Exact. Upstream:
/// `From<Vector4<bool>> for BVec4`.
pub impl Vector4IntoBVec4 of Into<Vector4<bool>, BVec4> {
    #[inline(always)]
    fn into(self: Vector4<bool>) -> BVec4 {
        BVec4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// `Some(v / |v|)`: the unit vector of `Vec2` `v`, or `None` when its norm is zero (upstream:
/// `Err`). `Unit::try_new(v, 0)`. Upstream: `TryFrom<Vec2> for UnitVector2<f32>`.
pub impl UnitVector2TryFromVec2 of TryInto<Vec2, UnitVector2<Fixed>> {
    #[inline(always)]
    fn try_into(self: Vec2) -> Option<UnitVector2<Fixed>> {
        UnitTrait::try_new(Vector2 { x: self.x, y: self.y }, Fixed { raw: 0 })
    }
}

/// The `Vec2` of the wrapped vector of a `UnitVector2<Fixed>`. Exact. Upstream:
/// `From<UnitVector2<f32>> for Vec2`.
pub impl UnitVector2IntoVec2 of Into<UnitVector2<Fixed>, Vec2> {
    #[inline(always)]
    fn into(self: UnitVector2<Fixed>) -> Vec2 {
        Vec2 { x: self.value.x, y: self.value.y }
    }
}

/// `Some(v / |v|)`: the unit vector of `Vec3` `v`, or `None` when its norm is zero (upstream:
/// `Err`). `Unit::try_new(v, 0)`. Upstream: `TryFrom<Vec3> for UnitVector3<f32>`.
pub impl UnitVector3TryFromVec3 of TryInto<Vec3, UnitVector3<Fixed>> {
    #[inline(always)]
    fn try_into(self: Vec3) -> Option<UnitVector3<Fixed>> {
        UnitTrait::try_new(Vector3 { x: self.x, y: self.y, z: self.z }, Fixed { raw: 0 })
    }
}

/// The `Vec3` of the wrapped vector of a `UnitVector3<Fixed>`. Exact. Upstream:
/// `From<UnitVector3<f32>> for Vec3`.
pub impl UnitVector3IntoVec3 of Into<UnitVector3<Fixed>, Vec3> {
    #[inline(always)]
    fn into(self: UnitVector3<Fixed>) -> Vec3 {
        Vec3 { x: self.value.x, y: self.value.y, z: self.value.z }
    }
}

/// `Some(v / |v|)`: the unit vector of `Vec4` `v`, or `None` when its norm is zero (upstream:
/// `Err`). `Unit::try_new(v, 0)`. Upstream: `TryFrom<Vec4> for UnitVector4<f32>`.
pub impl UnitVector4TryFromVec4 of TryInto<Vec4, UnitVector4<Fixed>> {
    #[inline(always)]
    fn try_into(self: Vec4) -> Option<UnitVector4<Fixed>> {
        UnitTrait::try_new(Vector4 { x: self.x, y: self.y, z: self.z, w: self.w }, Fixed { raw: 0 })
    }
}

/// The `Vec4` of the wrapped vector of a `UnitVector4<Fixed>`. Exact. Upstream:
/// `From<UnitVector4<f32>> for Vec4`.
pub impl UnitVector4IntoVec4 of Into<UnitVector4<Fixed>, Vec4> {
    #[inline(always)]
    fn into(self: UnitVector4<Fixed>) -> Vec4 {
        Vec4 { x: self.value.x, y: self.value.y, z: self.value.z, w: self.value.w }
    }
}

/// The `Matrix2<Fixed>` whose column `j` is the axis `j` of `Mat2` (glam is column-major). Exact.
/// Upstream: `From<Mat2> for Matrix2<f32>` (`to_cols_array_2d`).
pub impl Matrix2FromMat2 of Into<Mat2, Matrix2<Fixed>> {
    #[inline(always)]
    fn into(self: Mat2) -> Matrix2<Fixed> {
        Matrix2 { m11: self.x_axis.x, m21: self.x_axis.y, m12: self.y_axis.x, m22: self.y_axis.y }
    }
}

/// The `Mat2` whose axis `j` is the column `j` of a `Matrix2<Fixed>` (glam is column-major). Exact.
/// Upstream: `From<Matrix2<f32>> for Mat2` (`from_cols`).
pub impl Matrix2IntoMat2 of Into<Matrix2<Fixed>, Mat2> {
    #[inline(always)]
    fn into(self: Matrix2<Fixed>) -> Mat2 {
        Mat2 {
            x_axis: Vec2 { x: self.m11, y: self.m21 }, y_axis: Vec2 { x: self.m12, y: self.m22 },
        }
    }
}

/// The `Matrix3<Fixed>` whose column `j` is the axis `j` of `Mat3` (glam is column-major). Exact.
/// Upstream: `From<Mat3> for Matrix3<f32>` (`to_cols_array_2d`).
pub impl Matrix3FromMat3 of Into<Mat3, Matrix3<Fixed>> {
    #[inline(always)]
    fn into(self: Mat3) -> Matrix3<Fixed> {
        Matrix3 {
            m11: self.x_axis.x,
            m21: self.x_axis.y,
            m31: self.x_axis.z,
            m12: self.y_axis.x,
            m22: self.y_axis.y,
            m32: self.y_axis.z,
            m13: self.z_axis.x,
            m23: self.z_axis.y,
            m33: self.z_axis.z,
        }
    }
}

/// The `Mat3` whose axis `j` is the column `j` of a `Matrix3<Fixed>` (glam is column-major). Exact.
/// Upstream: `From<Matrix3<f32>> for Mat3` (`from_cols`).
pub impl Matrix3IntoMat3 of Into<Matrix3<Fixed>, Mat3> {
    #[inline(always)]
    fn into(self: Matrix3<Fixed>) -> Mat3 {
        Mat3 {
            x_axis: Vec3 { x: self.m11, y: self.m21, z: self.m31 },
            y_axis: Vec3 { x: self.m12, y: self.m22, z: self.m32 },
            z_axis: Vec3 { x: self.m13, y: self.m23, z: self.m33 },
        }
    }
}

/// The `Matrix4<Fixed>` whose column `j` is the axis `j` of `Mat4` (glam is column-major). Exact.
/// Upstream: `From<Mat4> for Matrix4<f32>` (`to_cols_array_2d`).
pub impl Matrix4FromMat4 of Into<Mat4, Matrix4<Fixed>> {
    #[inline(always)]
    fn into(self: Mat4) -> Matrix4<Fixed> {
        Matrix4 {
            m11: self.x_axis.x,
            m21: self.x_axis.y,
            m31: self.x_axis.z,
            m41: self.x_axis.w,
            m12: self.y_axis.x,
            m22: self.y_axis.y,
            m32: self.y_axis.z,
            m42: self.y_axis.w,
            m13: self.z_axis.x,
            m23: self.z_axis.y,
            m33: self.z_axis.z,
            m43: self.z_axis.w,
            m14: self.w_axis.x,
            m24: self.w_axis.y,
            m34: self.w_axis.z,
            m44: self.w_axis.w,
        }
    }
}

/// The `Mat4` whose axis `j` is the column `j` of a `Matrix4<Fixed>` (glam is column-major). Exact.
/// Upstream: `From<Matrix4<f32>> for Mat4` (`from_cols`).
pub impl Matrix4IntoMat4 of Into<Matrix4<Fixed>, Mat4> {
    #[inline(always)]
    fn into(self: Matrix4<Fixed>) -> Mat4 {
        Mat4 {
            x_axis: Vec4 { x: self.m11, y: self.m21, z: self.m31, w: self.m41 },
            y_axis: Vec4 { x: self.m12, y: self.m22, z: self.m32, w: self.m42 },
            z_axis: Vec4 { x: self.m13, y: self.m23, z: self.m33, w: self.m43 },
            w_axis: Vec4 { x: self.m14, y: self.m24, z: self.m34, w: self.m44 },
        }
    }
}
