//! Internal, no stability promise: the crate-private items of `base::matrix_view` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix1::Matrix1;
use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::matrix2x3::Matrix2x3;
use nalgebra_core::base::matrix2x4::Matrix2x4;
use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::matrix3x2::Matrix3x2;
use nalgebra_core::base::matrix3x4::Matrix3x4;
use nalgebra_core::base::matrix4::Matrix4;
use nalgebra_core::base::matrix4x2::Matrix4x2;
use nalgebra_core::base::matrix4x3::Matrix4x3;
use nalgebra_core::base::row_vector2::RowVector2;
use nalgebra_core::base::row_vector3::RowVector3;
use nalgebra_core::base::row_vector4::RowVector4;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::base::vector4::Vector4;
use nalgebra_shapes5::base::matrix2x5::Matrix2x5;
use nalgebra_shapes5::base::matrix3x5::Matrix3x5;
use nalgebra_shapes5::base::matrix4x5::Matrix4x5;
use nalgebra_shapes5::base::matrix5::Matrix5;
use nalgebra_shapes5::base::matrix5x2::Matrix5x2;
use nalgebra_shapes5::base::matrix5x3::Matrix5x3;
use nalgebra_shapes5::base::matrix5x4::Matrix5x4;
use nalgebra_shapes5::base::row_vector5::RowVector5;
use nalgebra_shapes5::base::vector5::Vector5;
use nalgebra_shapes6::base::matrix2x6::Matrix2x6;
use nalgebra_shapes6::base::matrix3x6::Matrix3x6;
use nalgebra_shapes6::base::matrix4x6::Matrix4x6;
use nalgebra_shapes6::base::matrix5x6::Matrix5x6;
use nalgebra_shapes6::base::matrix6::Matrix6;
use nalgebra_shapes6::base::matrix6x2::Matrix6x2;
use nalgebra_shapes6::base::matrix6x3::Matrix6x3;
use nalgebra_shapes6::base::matrix6x4::Matrix6x4;
use nalgebra_shapes6::base::matrix6x5::Matrix6x5;
use nalgebra_shapes6::base::row_vector6::RowVector6;
use nalgebra_shapes6::base::vector6::Vector6;

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix1PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix1<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix1<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: val,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: val,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix1` of a `Matrix6` (`FixedResize`).
pub impl Matrix1CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix1<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix1<T> {
        Matrix1 { x: m.m11 }
    }
}

/// `(1, 1)`: the size checks of the runtime-sized views.
pub impl Matrix1ShapeDims<T> of ShapeDims<Matrix1<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (1, 1)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl RowVector2PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<RowVector2<T>, T> {
    #[inline(always)]
    fn pad(self: RowVector2<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: val,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.y,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `RowVector2` of a `Matrix6` (`FixedResize`).
pub impl RowVector2CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<RowVector2<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> RowVector2<T> {
        RowVector2 { x: m.m11, y: m.m12 }
    }
}

/// `(1, 2)`: the size checks of the runtime-sized views.
pub impl RowVector2ShapeDims<T> of ShapeDims<RowVector2<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (1, 2)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl RowVector3PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<RowVector3<T>, T> {
    #[inline(always)]
    fn pad(self: RowVector3<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: val,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.y,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: self.z,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `RowVector3` of a `Matrix6` (`FixedResize`).
pub impl RowVector3CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<RowVector3<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> RowVector3<T> {
        RowVector3 { x: m.m11, y: m.m12, z: m.m13 }
    }
}

/// `(1, 3)`: the size checks of the runtime-sized views.
pub impl RowVector3ShapeDims<T> of ShapeDims<RowVector3<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (1, 3)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl RowVector4PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<RowVector4<T>, T> {
    #[inline(always)]
    fn pad(self: RowVector4<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: val,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.y,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: self.z,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: self.w,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `RowVector4` of a `Matrix6` (`FixedResize`).
pub impl RowVector4CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<RowVector4<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> RowVector4<T> {
        RowVector4 { x: m.m11, y: m.m12, z: m.m13, w: m.m14 }
    }
}

/// `(1, 4)`: the size checks of the runtime-sized views.
pub impl RowVector4ShapeDims<T> of ShapeDims<RowVector4<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (1, 4)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl RowVector5PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<RowVector5<T>, T> {
    #[inline(always)]
    fn pad(self: RowVector5<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: val,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.y,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: self.z,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: self.w,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: self.a,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `RowVector5` of a `Matrix6` (`FixedResize`).
pub impl RowVector5CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<RowVector5<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> RowVector5<T> {
        RowVector5 { x: m.m11, y: m.m12, z: m.m13, w: m.m14, a: m.m15 }
    }
}

/// `(1, 5)`: the size checks of the runtime-sized views.
pub impl RowVector5ShapeDims<T> of ShapeDims<RowVector5<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (1, 5)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl RowVector6PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<RowVector6<T>, T> {
    #[inline(always)]
    fn pad(self: RowVector6<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: val,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.y,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: self.z,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: self.w,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: self.a,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: self.b,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `RowVector6` of a `Matrix6` (`FixedResize`).
pub impl RowVector6CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<RowVector6<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> RowVector6<T> {
        RowVector6 { x: m.m11, y: m.m12, z: m.m13, w: m.m14, a: m.m15, b: m.m16 }
    }
}

/// `(1, 6)`: the size checks of the runtime-sized views.
pub impl RowVector6ShapeDims<T> of ShapeDims<RowVector6<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (1, 6)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Vector2PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Vector2<T>, T> {
    #[inline(always)]
    fn pad(self: Vector2<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: self.y,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: val,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Vector2` of a `Matrix6` (`FixedResize`).
pub impl Vector2CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Vector2<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Vector2<T> {
        Vector2 { x: m.m11, y: m.m21 }
    }
}

/// `(2, 1)`: the size checks of the runtime-sized views.
pub impl Vector2ShapeDims<T> of ShapeDims<Vector2<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (2, 1)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix2PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix2<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix2<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix2` of a `Matrix6` (`FixedResize`).
pub impl Matrix2CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix2<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix2<T> {
        Matrix2 { m11: m.m11, m21: m.m21, m12: m.m12, m22: m.m22 }
    }
}

/// `(2, 2)`: the size checks of the runtime-sized views.
pub impl Matrix2ShapeDims<T> of ShapeDims<Matrix2<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (2, 2)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix2x3PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix2x3<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix2x3<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix2x3` of a `Matrix6` (`FixedResize`).
pub impl Matrix2x3CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix2x3<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix2x3<T> {
        Matrix2x3 { m11: m.m11, m21: m.m21, m12: m.m12, m22: m.m22, m13: m.m13, m23: m.m23 }
    }
}

/// `(2, 3)`: the size checks of the runtime-sized views.
pub impl Matrix2x3ShapeDims<T> of ShapeDims<Matrix2x3<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (2, 3)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix2x4PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix2x4<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix2x4<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix2x4` of a `Matrix6` (`FixedResize`).
pub impl Matrix2x4CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix2x4<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix2x4<T> {
        Matrix2x4 {
            m11: m.m11,
            m21: m.m21,
            m12: m.m12,
            m22: m.m22,
            m13: m.m13,
            m23: m.m23,
            m14: m.m14,
            m24: m.m24,
        }
    }
}

/// `(2, 4)`: the size checks of the runtime-sized views.
pub impl Matrix2x4ShapeDims<T> of ShapeDims<Matrix2x4<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (2, 4)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix2x5PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix2x5<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix2x5<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: self.m15,
            m25: self.m25,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix2x5` of a `Matrix6` (`FixedResize`).
pub impl Matrix2x5CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix2x5<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix2x5<T> {
        Matrix2x5 {
            m11: m.m11,
            m21: m.m21,
            m12: m.m12,
            m22: m.m22,
            m13: m.m13,
            m23: m.m23,
            m14: m.m14,
            m24: m.m24,
            m15: m.m15,
            m25: m.m25,
        }
    }
}

/// `(2, 5)`: the size checks of the runtime-sized views.
pub impl Matrix2x5ShapeDims<T> of ShapeDims<Matrix2x5<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (2, 5)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix2x6PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix2x6<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix2x6<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: val,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: self.m15,
            m25: self.m25,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: self.m16,
            m26: self.m26,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix2x6` of a `Matrix6` (`FixedResize`).
pub impl Matrix2x6CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix2x6<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix2x6<T> {
        Matrix2x6 {
            m11: m.m11,
            m21: m.m21,
            m12: m.m12,
            m22: m.m22,
            m13: m.m13,
            m23: m.m23,
            m14: m.m14,
            m24: m.m24,
            m15: m.m15,
            m25: m.m25,
            m16: m.m16,
            m26: m.m26,
        }
    }
}

/// `(2, 6)`: the size checks of the runtime-sized views.
pub impl Matrix2x6ShapeDims<T> of ShapeDims<Matrix2x6<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (2, 6)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Vector3PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Vector3<T>, T> {
    #[inline(always)]
    fn pad(self: Vector3<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: self.y,
            m31: self.z,
            m41: val,
            m51: val,
            m61: val,
            m12: val,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Vector3` of a `Matrix6` (`FixedResize`).
pub impl Vector3CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Vector3<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Vector3<T> {
        Vector3 { x: m.m11, y: m.m21, z: m.m31 }
    }
}

/// `(3, 1)`: the size checks of the runtime-sized views.
pub impl Vector3ShapeDims<T> of ShapeDims<Vector3<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (3, 1)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix3x2PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix3x2<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix3x2<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix3x2` of a `Matrix6` (`FixedResize`).
pub impl Matrix3x2CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix3x2<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix3x2<T> {
        Matrix3x2 { m11: m.m11, m21: m.m21, m31: m.m31, m12: m.m12, m22: m.m22, m32: m.m32 }
    }
}

/// `(3, 2)`: the size checks of the runtime-sized views.
pub impl Matrix3x2ShapeDims<T> of ShapeDims<Matrix3x2<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (3, 2)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix3PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix3<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix3<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: val,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix3` of a `Matrix6` (`FixedResize`).
pub impl Matrix3CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix3<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix3<T> {
        Matrix3 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
        }
    }
}

/// `(3, 3)`: the size checks of the runtime-sized views.
pub impl Matrix3ShapeDims<T> of ShapeDims<Matrix3<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (3, 3)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix3x4PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix3x4<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix3x4<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: val,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: val,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix3x4` of a `Matrix6` (`FixedResize`).
pub impl Matrix3x4CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix3x4<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix3x4<T> {
        Matrix3x4 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
        }
    }
}

/// `(3, 4)`: the size checks of the runtime-sized views.
pub impl Matrix3x4ShapeDims<T> of ShapeDims<Matrix3x4<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (3, 4)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix3x5PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix3x5<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix3x5<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: val,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: val,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: val,
            m54: val,
            m64: val,
            m15: self.m15,
            m25: self.m25,
            m35: self.m35,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix3x5` of a `Matrix6` (`FixedResize`).
pub impl Matrix3x5CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix3x5<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix3x5<T> {
        Matrix3x5 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m15: m.m15,
            m25: m.m25,
            m35: m.m35,
        }
    }
}

/// `(3, 5)`: the size checks of the runtime-sized views.
pub impl Matrix3x5ShapeDims<T> of ShapeDims<Matrix3x5<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (3, 5)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix3x6PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix3x6<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix3x6<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: val,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: val,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: val,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: val,
            m54: val,
            m64: val,
            m15: self.m15,
            m25: self.m25,
            m35: self.m35,
            m45: val,
            m55: val,
            m65: val,
            m16: self.m16,
            m26: self.m26,
            m36: self.m36,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix3x6` of a `Matrix6` (`FixedResize`).
pub impl Matrix3x6CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix3x6<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix3x6<T> {
        Matrix3x6 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m15: m.m15,
            m25: m.m25,
            m35: m.m35,
            m16: m.m16,
            m26: m.m26,
            m36: m.m36,
        }
    }
}

/// `(3, 6)`: the size checks of the runtime-sized views.
pub impl Matrix3x6ShapeDims<T> of ShapeDims<Matrix3x6<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (3, 6)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Vector4PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Vector4<T>, T> {
    #[inline(always)]
    fn pad(self: Vector4<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: self.y,
            m31: self.z,
            m41: self.w,
            m51: val,
            m61: val,
            m12: val,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Vector4` of a `Matrix6` (`FixedResize`).
pub impl Vector4CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Vector4<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Vector4<T> {
        Vector4 { x: m.m11, y: m.m21, z: m.m31, w: m.m41 }
    }
}

/// `(4, 1)`: the size checks of the runtime-sized views.
pub impl Vector4ShapeDims<T> of ShapeDims<Vector4<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (4, 1)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix4x2PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix4x2<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix4x2<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix4x2` of a `Matrix6` (`FixedResize`).
pub impl Matrix4x2CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix4x2<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix4x2<T> {
        Matrix4x2 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
        }
    }
}

/// `(4, 2)`: the size checks of the runtime-sized views.
pub impl Matrix4x2ShapeDims<T> of ShapeDims<Matrix4x2<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (4, 2)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix4x3PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix4x3<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix4x3<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix4x3` of a `Matrix6` (`FixedResize`).
pub impl Matrix4x3CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix4x3<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix4x3<T> {
        Matrix4x3 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
        }
    }
}

/// `(4, 3)`: the size checks of the runtime-sized views.
pub impl Matrix4x3ShapeDims<T> of ShapeDims<Matrix4x3<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (4, 3)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix4PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix4<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix4<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix4` of a `Matrix6` (`FixedResize`).
pub impl Matrix4CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix4<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix4<T> {
        Matrix4 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
        }
    }
}

/// `(4, 4)`: the size checks of the runtime-sized views.
pub impl Matrix4ShapeDims<T> of ShapeDims<Matrix4<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (4, 4)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix4x5PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix4x5<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix4x5<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: val,
            m64: val,
            m15: self.m15,
            m25: self.m25,
            m35: self.m35,
            m45: self.m45,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix4x5` of a `Matrix6` (`FixedResize`).
pub impl Matrix4x5CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix4x5<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix4x5<T> {
        Matrix4x5 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
            m15: m.m15,
            m25: m.m25,
            m35: m.m35,
            m45: m.m45,
        }
    }
}

/// `(4, 5)`: the size checks of the runtime-sized views.
pub impl Matrix4x5ShapeDims<T> of ShapeDims<Matrix4x5<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (4, 5)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix4x6PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix4x6<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix4x6<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: val,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: val,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: val,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: val,
            m64: val,
            m15: self.m15,
            m25: self.m25,
            m35: self.m35,
            m45: self.m45,
            m55: val,
            m65: val,
            m16: self.m16,
            m26: self.m26,
            m36: self.m36,
            m46: self.m46,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix4x6` of a `Matrix6` (`FixedResize`).
pub impl Matrix4x6CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix4x6<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix4x6<T> {
        Matrix4x6 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
            m15: m.m15,
            m25: m.m25,
            m35: m.m35,
            m45: m.m45,
            m16: m.m16,
            m26: m.m26,
            m36: m.m36,
            m46: m.m46,
        }
    }
}

/// `(4, 6)`: the size checks of the runtime-sized views.
pub impl Matrix4x6ShapeDims<T> of ShapeDims<Matrix4x6<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (4, 6)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Vector5PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Vector5<T>, T> {
    #[inline(always)]
    fn pad(self: Vector5<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: self.y,
            m31: self.z,
            m41: self.w,
            m51: self.a,
            m61: val,
            m12: val,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Vector5` of a `Matrix6` (`FixedResize`).
pub impl Vector5CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Vector5<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Vector5<T> {
        Vector5 { x: m.m11, y: m.m21, z: m.m31, w: m.m41, a: m.m51 }
    }
}

/// `(5, 1)`: the size checks of the runtime-sized views.
pub impl Vector5ShapeDims<T> of ShapeDims<Vector5<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (5, 1)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix5x2PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix5x2<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix5x2<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix5x2` of a `Matrix6` (`FixedResize`).
pub impl Matrix5x2CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix5x2<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix5x2<T> {
        Matrix5x2 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
        }
    }
}

/// `(5, 2)`: the size checks of the runtime-sized views.
pub impl Matrix5x2ShapeDims<T> of ShapeDims<Matrix5x2<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (5, 2)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix5x3PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix5x3<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix5x3<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: self.m53,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix5x3` of a `Matrix6` (`FixedResize`).
pub impl Matrix5x3CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix5x3<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix5x3<T> {
        Matrix5x3 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m53: m.m53,
        }
    }
}

/// `(5, 3)`: the size checks of the runtime-sized views.
pub impl Matrix5x3ShapeDims<T> of ShapeDims<Matrix5x3<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (5, 3)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix5x4PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix5x4<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix5x4<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: self.m53,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: self.m54,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix5x4` of a `Matrix6` (`FixedResize`).
pub impl Matrix5x4CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix5x4<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix5x4<T> {
        Matrix5x4 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m53: m.m53,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
            m54: m.m54,
        }
    }
}

/// `(5, 4)`: the size checks of the runtime-sized views.
pub impl Matrix5x4ShapeDims<T> of ShapeDims<Matrix5x4<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (5, 4)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix5PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix5<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix5<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: self.m53,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: self.m54,
            m64: val,
            m15: self.m15,
            m25: self.m25,
            m35: self.m35,
            m45: self.m45,
            m55: self.m55,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix5` of a `Matrix6` (`FixedResize`).
pub impl Matrix5CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix5<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix5<T> {
        Matrix5 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m53: m.m53,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
            m54: m.m54,
            m15: m.m15,
            m25: m.m25,
            m35: m.m35,
            m45: m.m45,
            m55: m.m55,
        }
    }
}

/// `(5, 5)`: the size checks of the runtime-sized views.
pub impl Matrix5ShapeDims<T> of ShapeDims<Matrix5<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (5, 5)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix5x6PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix5x6<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix5x6<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: val,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: val,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: self.m53,
            m63: val,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: self.m54,
            m64: val,
            m15: self.m15,
            m25: self.m25,
            m35: self.m35,
            m45: self.m45,
            m55: self.m55,
            m65: val,
            m16: self.m16,
            m26: self.m26,
            m36: self.m36,
            m46: self.m46,
            m56: self.m56,
            m66: val,
        }
    }
}

/// The top-left `Matrix5x6` of a `Matrix6` (`FixedResize`).
pub impl Matrix5x6CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix5x6<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix5x6<T> {
        Matrix5x6 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m53: m.m53,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
            m54: m.m54,
            m15: m.m15,
            m25: m.m25,
            m35: m.m35,
            m45: m.m45,
            m55: m.m55,
            m16: m.m16,
            m26: m.m26,
            m36: m.m36,
            m46: m.m46,
            m56: m.m56,
        }
    }
}

/// `(5, 6)`: the size checks of the runtime-sized views.
pub impl Matrix5x6ShapeDims<T> of ShapeDims<Matrix5x6<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (5, 6)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Vector6PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Vector6<T>, T> {
    #[inline(always)]
    fn pad(self: Vector6<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.x,
            m21: self.y,
            m31: self.z,
            m41: self.w,
            m51: self.a,
            m61: self.b,
            m12: val,
            m22: val,
            m32: val,
            m42: val,
            m52: val,
            m62: val,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Vector6` of a `Matrix6` (`FixedResize`).
pub impl Vector6CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Vector6<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Vector6<T> {
        Vector6 { x: m.m11, y: m.m21, z: m.m31, w: m.m41, a: m.m51, b: m.m61 }
    }
}

/// `(6, 1)`: the size checks of the runtime-sized views.
pub impl Vector6ShapeDims<T> of ShapeDims<Vector6<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (6, 1)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix6x2PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix6x2<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix6x2<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: self.m61,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: self.m62,
            m13: val,
            m23: val,
            m33: val,
            m43: val,
            m53: val,
            m63: val,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix6x2` of a `Matrix6` (`FixedResize`).
pub impl Matrix6x2CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix6x2<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix6x2<T> {
        Matrix6x2 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m61: m.m61,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m62: m.m62,
        }
    }
}

/// `(6, 2)`: the size checks of the runtime-sized views.
pub impl Matrix6x2ShapeDims<T> of ShapeDims<Matrix6x2<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (6, 2)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix6x3PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix6x3<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix6x3<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: self.m61,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: self.m62,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: self.m53,
            m63: self.m63,
            m14: val,
            m24: val,
            m34: val,
            m44: val,
            m54: val,
            m64: val,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix6x3` of a `Matrix6` (`FixedResize`).
pub impl Matrix6x3CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix6x3<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix6x3<T> {
        Matrix6x3 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m61: m.m61,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m62: m.m62,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m53: m.m53,
            m63: m.m63,
        }
    }
}

/// `(6, 3)`: the size checks of the runtime-sized views.
pub impl Matrix6x3ShapeDims<T> of ShapeDims<Matrix6x3<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (6, 3)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix6x4PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix6x4<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix6x4<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: self.m61,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: self.m62,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: self.m53,
            m63: self.m63,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: self.m54,
            m64: self.m64,
            m15: val,
            m25: val,
            m35: val,
            m45: val,
            m55: val,
            m65: val,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix6x4` of a `Matrix6` (`FixedResize`).
pub impl Matrix6x4CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix6x4<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix6x4<T> {
        Matrix6x4 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m61: m.m61,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m62: m.m62,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m53: m.m53,
            m63: m.m63,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
            m54: m.m54,
            m64: m.m64,
        }
    }
}

/// `(6, 4)`: the size checks of the runtime-sized views.
pub impl Matrix6x4ShapeDims<T> of ShapeDims<Matrix6x4<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (6, 4)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix6x5PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix6x5<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix6x5<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: self.m61,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: self.m62,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: self.m53,
            m63: self.m63,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: self.m54,
            m64: self.m64,
            m15: self.m15,
            m25: self.m25,
            m35: self.m35,
            m45: self.m45,
            m55: self.m55,
            m65: self.m65,
            m16: val,
            m26: val,
            m36: val,
            m46: val,
            m56: val,
            m66: val,
        }
    }
}

/// The top-left `Matrix6x5` of a `Matrix6` (`FixedResize`).
pub impl Matrix6x5CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix6x5<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix6x5<T> {
        Matrix6x5 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m61: m.m61,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m62: m.m62,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m53: m.m53,
            m63: m.m63,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
            m54: m.m54,
            m64: m.m64,
            m15: m.m15,
            m25: m.m25,
            m35: m.m35,
            m45: m.m45,
            m55: m.m55,
            m65: m.m65,
        }
    }
}

/// `(6, 5)`: the size checks of the runtime-sized views.
pub impl Matrix6x5ShapeDims<T> of ShapeDims<Matrix6x5<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (6, 5)
    }
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).
pub impl Matrix6PadTo6<T, +Copy<T>, +Drop<T>> of PadTo6<Matrix6<T>, T> {
    #[inline(always)]
    fn pad(self: Matrix6<T>, val: T) -> Matrix6<T> {
        Matrix6 {
            m11: self.m11,
            m21: self.m21,
            m31: self.m31,
            m41: self.m41,
            m51: self.m51,
            m61: self.m61,
            m12: self.m12,
            m22: self.m22,
            m32: self.m32,
            m42: self.m42,
            m52: self.m52,
            m62: self.m62,
            m13: self.m13,
            m23: self.m23,
            m33: self.m33,
            m43: self.m43,
            m53: self.m53,
            m63: self.m63,
            m14: self.m14,
            m24: self.m24,
            m34: self.m34,
            m44: self.m44,
            m54: self.m54,
            m64: self.m64,
            m15: self.m15,
            m25: self.m25,
            m35: self.m35,
            m45: self.m45,
            m55: self.m55,
            m65: self.m65,
            m16: self.m16,
            m26: self.m26,
            m36: self.m36,
            m46: self.m46,
            m56: self.m56,
            m66: self.m66,
        }
    }
}

/// The top-left `Matrix6` of a `Matrix6` (`FixedResize`).
pub impl Matrix6CropFrom6<T, +Copy<T>, +Drop<T>> of CropFrom6<Matrix6<T>, T> {
    #[inline(always)]
    fn crop(m: Matrix6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: m.m11,
            m21: m.m21,
            m31: m.m31,
            m41: m.m41,
            m51: m.m51,
            m61: m.m61,
            m12: m.m12,
            m22: m.m22,
            m32: m.m32,
            m42: m.m42,
            m52: m.m52,
            m62: m.m62,
            m13: m.m13,
            m23: m.m23,
            m33: m.m33,
            m43: m.m43,
            m53: m.m53,
            m63: m.m63,
            m14: m.m14,
            m24: m.m24,
            m34: m.m34,
            m44: m.m44,
            m54: m.m54,
            m64: m.m64,
            m15: m.m15,
            m25: m.m25,
            m35: m.m35,
            m45: m.m45,
            m55: m.m55,
            m65: m.m65,
            m16: m.m16,
            m26: m.m26,
            m36: m.m36,
            m46: m.m46,
            m56: m.m56,
            m66: m.m66,
        }
    }
}

/// `(6, 6)`: the size checks of the runtime-sized views.
pub impl Matrix6ShapeDims<T> of ShapeDims<Matrix6<T>> {
    #[inline(always)]
    fn dims() -> (usize, usize) {
        (6, 6)
    }
}

/// The shape `(nrows, ncols)` of `S` (the size checks of the runtime-sized forms).
pub trait ShapeDims<S> {
    fn dims() -> (usize, usize);
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (the canvas of `FixedResize`).
pub trait PadTo6<M, T> {
    fn pad(self: M, val: T) -> Matrix6<T>;
}

/// The top-left block of a `Matrix6` of the shape `Out` (the canvas of `FixedResize`).
pub trait CropFrom6<Out, T> {
    fn crop(m: Matrix6<T>) -> Out;
}
