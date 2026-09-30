//! Internal, no stability promise: the crate-private items of `base::matrix3x2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::errors;
use nalgebra_types2::base::row_vector2::RowVector2;
use crate::base::matrix3x2::Matrix3x2;
use crate::base::vector3::Vector3;

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl Matrix3x2EditImpl<T, +Copy<T>, +Drop<T>> of Matrix3x2EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: Matrix3x2<T>, index: (usize, usize), v: T) -> Matrix3x2<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => Matrix3x2 {
                    m11: v,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                },
                1 => Matrix3x2 {
                    m11: self.m11,
                    m21: v,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                },
                2 => Matrix3x2 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: v,
                    m12: self.m12,
                    m22: self.m22,
                    m32: self.m32,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            1 => match i {
                0 => Matrix3x2 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: v,
                    m22: self.m22,
                    m32: self.m32,
                },
                1 => Matrix3x2 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: v,
                    m32: self.m32,
                },
                2 => Matrix3x2 {
                    m11: self.m11,
                    m21: self.m21,
                    m31: self.m31,
                    m12: self.m12,
                    m22: self.m22,
                    m32: v,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `RowVector2`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: Matrix3x2<T>, i: usize) -> RowVector2<T> {
        match i {
            0 => RowVector2 { x: self.m11, y: self.m12 },
            1 => RowVector2 { x: self.m21, y: self.m22 },
            2 => RowVector2 { x: self.m31, y: self.m32 },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Vector3`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: Matrix3x2<T>, j: usize) -> Vector3<T> {
        match j {
            0 => Vector3 { x: self.m11, y: self.m21, z: self.m31 },
            1 => Vector3 { x: self.m12, y: self.m22, z: self.m32 },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
