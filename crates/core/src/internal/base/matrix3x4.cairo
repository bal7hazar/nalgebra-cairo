//! Internal, no stability promise: the crate-private items of `base::matrix3x4` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use crate::base::errors;
use crate::base::matrix3x4::Matrix3x4;
use crate::base::row_vector4::RowVector4;
use crate::base::vector3::Vector3;

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl Matrix3x4EditImpl<T, +Copy<T>, +Drop<T>> of Matrix3x4EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: Matrix3x4<T>, index: (usize, usize), v: T) -> Matrix3x4<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => Matrix3x4 {
                    m11: v,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                1 => Matrix3x4 {
                    m11: self.m11,
                    m21: v,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                2 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: v,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            1 => match i {
                0 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: v,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                1 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: v,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                2 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: v,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            2 => match i {
                0 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: v,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                1 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: v,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                2 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: v,
                    m14: self.m14,
                    m24: self.m24,
                    m34: self.m34,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            3 => match i {
                0 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: v,
                    m24: self.m24,
                    m34: self.m34,
                },
                1 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: v,
                    m34: self.m34,
                },
                2 => Matrix3x4 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                    m13: self.m13,
                    m23: self.m23,
                    m33: self.m33,
                    m14: self.m14,
                    m24: self.m24,
                    m34: v,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `RowVector4`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: Matrix3x4<T>, i: usize) -> RowVector4<T> {
        match i {
            0 => RowVector4 { x: self.m11, y: self.m12, z: self.m13, w: self.m14 },
            1 => RowVector4 { x: self.m21, y: self.m22, z: self.m23, w: self.m24 },
            2 => RowVector4 { x: self.m31, y: self.m32, z: self.m33, w: self.m34 },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Vector3`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: Matrix3x4<T>, j: usize) -> Vector3<T> {
        match j {
            0 => Vector3 { x: self.m11, y: self.m21, z: self.m31 },
            1 => Vector3 { x: self.m12, y: self.m22, z: self.m32 },
            2 => Vector3 { x: self.m13, y: self.m23, z: self.m33 },
            3 => Vector3 { x: self.m14, y: self.m24, z: self.m34 },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
