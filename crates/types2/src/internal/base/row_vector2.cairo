//! Internal, no stability promise: the crate-private items of `base::row_vector2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::errors;
use nalgebra_core::base::matrix1::Matrix1;
use crate::base::row_vector2::RowVector2;

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl RowVector2EditImpl<T, +Copy<T>, +Drop<T>> of RowVector2EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: RowVector2<T>, index: (usize, usize), v: T) -> RowVector2<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => RowVector2 { x: v, y: self.y },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            1 => match i {
                0 => RowVector2 { x: self.x, y: v },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `RowVector2`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: RowVector2<T>, i: usize) -> RowVector2<T> {
        match i {
            0 => RowVector2 { x: self.x, y: self.y },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Matrix1`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: RowVector2<T>, j: usize) -> Matrix1<T> {
        match j {
            0 => Matrix1 { x: self.x },
            1 => Matrix1 { x: self.y },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
