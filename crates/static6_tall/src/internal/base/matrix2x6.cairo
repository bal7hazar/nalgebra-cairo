//! Internal, no stability promise: the crate-private items of `base::matrix2x6` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::errors;
use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types6::base::matrix2x6::Matrix2x6;
use nalgebra_types6::base::row_vector6::RowVector6;

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl Matrix2x6EditImpl<T, +Copy<T>, +Drop<T>> of Matrix2x6EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: Matrix2x6<T>, index: (usize, usize), v: T) -> Matrix2x6<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => Matrix2x6 {
                    m11: v,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                1 => Matrix2x6 {
                    m11: self.m11,
                    m21: v,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            1 => match i {
                0 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: v,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                1 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: v,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            2 => match i {
                0 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: v,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                1 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: v,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            3 => match i {
                0 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: v,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                1 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: v,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            4 => match i {
                0 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: v,
                    m25: self.m25,
                    m16: self.m16,
                    m26: self.m26,
                },
                1 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: v,
                    m16: self.m16,
                    m26: self.m26,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            5 => match i {
                0 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: v,
                    m26: self.m26,
                },
                1 => Matrix2x6 {
                    m11: self.m11,
                    m21: self.m21,
                    m12: self.m12,
                    m22: self.m22,
                    m13: self.m13,
                    m23: self.m23,
                    m14: self.m14,
                    m24: self.m24,
                    m15: self.m15,
                    m25: self.m25,
                    m16: self.m16,
                    m26: v,
                },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `RowVector6`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: Matrix2x6<T>, i: usize) -> RowVector6<T> {
        match i {
            0 => RowVector6 {
                x: self.m11, y: self.m12, z: self.m13, w: self.m14, a: self.m15, b: self.m16,
            },
            1 => RowVector6 {
                x: self.m21, y: self.m22, z: self.m23, w: self.m24, a: self.m25, b: self.m26,
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Vector2`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: Matrix2x6<T>, j: usize) -> Vector2<T> {
        match j {
            0 => Vector2 { x: self.m11, y: self.m21 },
            1 => Vector2 { x: self.m12, y: self.m22 },
            2 => Vector2 { x: self.m13, y: self.m23 },
            3 => Vector2 { x: self.m14, y: self.m24 },
            4 => Vector2 { x: self.m15, y: self.m25 },
            5 => Vector2 { x: self.m16, y: self.m26 },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
