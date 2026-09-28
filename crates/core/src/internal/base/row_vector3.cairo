//! Internal, no stability promise: the crate-private items of `base::row_vector3` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use crate::base::errors;
use crate::base::matrix1::Matrix1;
use crate::base::row_vector3::RowVector3;

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl RowVector3EditImpl<T, +Copy<T>, +Drop<T>> of RowVector3EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: RowVector3<T>, index: (usize, usize), v: T) -> RowVector3<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => RowVector3 { x: v, y: self.y, z: self.z },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            1 => match i {
                0 => RowVector3 { x: self.x, y: v, z: self.z },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            2 => match i {
                0 => RowVector3 { x: self.x, y: self.y, z: v },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `RowVector3`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: RowVector3<T>, i: usize) -> RowVector3<T> {
        match i {
            0 => RowVector3 { x: self.x, y: self.y, z: self.z },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Matrix1`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: RowVector3<T>, j: usize) -> Matrix1<T> {
        match j {
            0 => Matrix1 { x: self.x },
            1 => Matrix1 { x: self.y },
            2 => Matrix1 { x: self.z },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
