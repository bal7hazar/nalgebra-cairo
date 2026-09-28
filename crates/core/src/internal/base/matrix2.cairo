//! Internal, no stability promise: the crate-private items of `base::matrix2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use crate::base::errors;
use crate::base::matrix2::Matrix2;
use crate::base::row_vector2::RowVector2;
use crate::base::vector2::Vector2;

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl Matrix2EditImpl<T, +Copy<T>, +Drop<T>> of Matrix2EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: Matrix2<T>, index: (usize, usize), v: T) -> Matrix2<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => Matrix2 { m11: v, m21: self.m21, m12: self.m12, m22: self.m22 },
                1 => Matrix2 { m11: self.m11, m21: v, m12: self.m12, m22: self.m22 },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            1 => match i {
                0 => Matrix2 { m11: self.m11, m21: self.m21, m12: v, m22: self.m22 },
                1 => Matrix2 { m11: self.m11, m21: self.m21, m12: self.m12, m22: v },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `RowVector2`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: Matrix2<T>, i: usize) -> RowVector2<T> {
        match i {
            0 => RowVector2 { x: self.m11, y: self.m12 },
            1 => RowVector2 { x: self.m21, y: self.m22 },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Vector2`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: Matrix2<T>, j: usize) -> Vector2<T> {
        match j {
            0 => Vector2 { x: self.m11, y: self.m21 },
            1 => Vector2 { x: self.m12, y: self.m22 },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
