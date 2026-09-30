//! Internal, no stability promise: the crate-private items of `base::row_vector6` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::errors;
use nalgebra_core::base::matrix1::Matrix1;
use nalgebra_types6::base::row_vector6::RowVector6;

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl RowVector6EditImpl<T, +Copy<T>, +Drop<T>> of RowVector6EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: RowVector6<T>, index: (usize, usize), v: T) -> RowVector6<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => RowVector6 { x: v, y: self.y, z: self.z, w: self.w, a: self.a, b: self.b },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            1 => match i {
                0 => RowVector6 { x: self.x, y: v, z: self.z, w: self.w, a: self.a, b: self.b },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            2 => match i {
                0 => RowVector6 { x: self.x, y: self.y, z: v, w: self.w, a: self.a, b: self.b },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            3 => match i {
                0 => RowVector6 { x: self.x, y: self.y, z: self.z, w: v, a: self.a, b: self.b },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            4 => match i {
                0 => RowVector6 { x: self.x, y: self.y, z: self.z, w: self.w, a: v, b: self.b },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            5 => match i {
                0 => RowVector6 { x: self.x, y: self.y, z: self.z, w: self.w, a: self.a, b: v },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `RowVector6`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: RowVector6<T>, i: usize) -> RowVector6<T> {
        match i {
            0 => RowVector6 { x: self.x, y: self.y, z: self.z, w: self.w, a: self.a, b: self.b },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Matrix1`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: RowVector6<T>, j: usize) -> Matrix1<T> {
        match j {
            0 => Matrix1 { x: self.x },
            1 => Matrix1 { x: self.y },
            2 => Matrix1 { x: self.z },
            3 => Matrix1 { x: self.w },
            4 => Matrix1 { x: self.a },
            5 => Matrix1 { x: self.b },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
