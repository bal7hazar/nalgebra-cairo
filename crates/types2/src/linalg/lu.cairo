//! LU factorisation with partial pivoting of the static square matrices (upstream
//! `nalgebra::linalg::lu`), one unrolled module per dimension: `Lu2`, `Lu3`, `Lu4`, `Lu6`.
//!
//! `LuN::new(a)` always succeeds, like upstream, and yields `P * A = L * U`: `L` unit lower
//! triangular, `U` upper triangular (both packed in one `MatrixN`, upstream's storage) and `P`
//! the row permutation, here the compact `PermN` defined below instead of upstream's
//! heap-allocated `PermutationSequence`. `solve`, `try_inverse` and `determinant` are then
//! `Option`-returning or total methods on the factorisation; `MatrixNLuTrait::lu` is the
//! upstream-named entry point (`matrix.lu()`), and `Matrix6LuTrait` additionally carries
//! `determinant`, `try_inverse` and `solve` for 6x6 matrices, which have no closed form worth
//! writing.
//!
//! Rejecting a singular matrix is not a saving: Cairo charges the worst-case path of a function,
//! so `solve` / `try_inverse` returning `None` costs what the successful call costs (measured:
//! `bench_luN_solve_singular__none` against `bench_luN_solve__substitution`, exactly as
//! `bench_matrix3_try_inverse_singular__none` already shows for the closed form). Check
//! `is_invertible` when the answer changes what the caller does, not to save gas.
//!
//! A `PermN` stores the transposition chosen at each of the `N - 1` elimination steps: `pK` is
//! the 1-based index of the row swapped with row `K` at step `K`, so `pK == K` means "no swap".
//! That is upstream's representation minus the allocation, it is `Copy`, and applying it (or its
//! inverse) is a fixed chain of comparisons and moves — the crate-internal `LuN::permute` /
//! `LuN::permute_rows` and the column permutation inside `LuN::try_inverse`. Building the `N x N`
//! permutation matrix is deliberately not offered: nothing in the library needs it, and it would
//! cost a full matrix product to use.

/// The row permutation of a 2x2 factorisation: the single transposition of step 1.
///
/// `p1` is the 1-based index of the row swapped with row 1 (`1` = no swap, `2` = rows 1 and 2
/// exchanged). Upstream: one entry of a `PermutationSequence`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm2 {
    /// Row swapped with row 1 at step 1, in `1..=2`.
    pub p1: u8,
}
use nalgebra_core::linalg::permutation_sequence::{PermuteColumns, PermuteRows};
use crate::base::row_vector2::RowVector2;
use crate::base::vector2::Vector2;

// crate-map: generated items (tools/split/cratemap.py) [shapegen]
// crate-map: from linalg/permutation_sequence.cairo
pub impl Perm2PermuteRowsVector2<T, +Copy<T>, +Drop<T>> of PermuteRows<Perm2, Vector2<T>> {
    fn permute_rows(self: Perm2, ref rhs: Vector2<T>) {
        let mut a00 = rhs.x;
        let mut a10 = rhs.y;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
        }
        rhs = Vector2 { x: a00, y: a10 };
    }

    fn inv_permute_rows(self: Perm2, ref rhs: Vector2<T>) {
        let mut a00 = rhs.x;
        let mut a10 = rhs.y;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
        }
        rhs = Vector2 { x: a00, y: a10 };
    }
}

pub impl Perm2PermuteColumnsRowVector2<
    T, +Copy<T>, +Drop<T>,
> of PermuteColumns<Perm2, RowVector2<T>> {
    fn permute_columns(self: Perm2, ref rhs: RowVector2<T>) {
        let mut a00 = rhs.x;
        let mut a01 = rhs.y;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
        }
        rhs = RowVector2 { x: a00, y: a01 };
    }

    fn inv_permute_columns(self: Perm2, ref rhs: RowVector2<T>) {
        let mut a00 = rhs.x;
        let mut a01 = rhs.y;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
        }
        rhs = RowVector2 { x: a00, y: a01 };
    }
}
// crate-map: end
