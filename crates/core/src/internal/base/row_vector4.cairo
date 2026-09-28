//! Internal, no stability promise: the crate-private items of `base::row_vector4` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use crate::base::errors;
use crate::base::matrix1::Matrix1;
use crate::base::row_vector4::RowVector4;

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl RowVector4EditImpl<T, +Copy<T>, +Drop<T>> of RowVector4EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: RowVector4<T>, index: (usize, usize), v: T) -> RowVector4<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => RowVector4 { x: v, y: self.y, z: self.z, w: self.w },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            1 => match i {
                0 => RowVector4 { x: self.x, y: v, z: self.z, w: self.w },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            2 => match i {
                0 => RowVector4 { x: self.x, y: self.y, z: v, w: self.w },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            3 => match i {
                0 => RowVector4 { x: self.x, y: self.y, z: self.z, w: v },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `RowVector4`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: RowVector4<T>, i: usize) -> RowVector4<T> {
        match i {
            0 => RowVector4 { x: self.x, y: self.y, z: self.z, w: self.w },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Matrix1`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: RowVector4<T>, j: usize) -> Matrix1<T> {
        match j {
            0 => Matrix1 { x: self.x },
            1 => Matrix1 { x: self.y },
            2 => Matrix1 { x: self.z },
            3 => Matrix1 { x: self.w },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
