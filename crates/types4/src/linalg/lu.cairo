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

/// The row permutation of a 4x4 factorisation: the transpositions of steps 1 to 3.
///
/// `pK` is the 1-based index of the row swapped with row `K` at step `K` (`pK == K` = no swap),
/// so `P = T3 * T2 * T1`. Upstream: a `PermutationSequence`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm4 {
    /// Row swapped with row 1 at step 1, in `1..=4`.
    pub p1: u8,
    /// Row swapped with row 2 at step 2, in `2..=4`.
    pub p2: u8,
    /// Row swapped with row 3 at step 3, in `3..=4`.
    pub p3: u8,
}
use nalgebra_core::linalg::permutation_sequence::{PermuteColumns, PermuteRows};
use crate::base::row_vector4::RowVector4;
use crate::base::vector4::Vector4;

// crate-map: generated items (tools/split/cratemap.py) [shapegen]
// crate-map: from linalg/permutation_sequence.cairo
pub impl Perm4PermuteRowsVector4<T, +Copy<T>, +Drop<T>> of PermuteRows<Perm4, Vector4<T>> {
    fn permute_rows(self: Perm4, ref rhs: Vector4<T>) {
        let mut a00 = rhs.x;
        let mut a10 = rhs.y;
        let mut a20 = rhs.z;
        let mut a30 = rhs.w;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
        }
        rhs = Vector4 { x: a00, y: a10, z: a20, w: a30 };
    }

    fn inv_permute_rows(self: Perm4, ref rhs: Vector4<T>) {
        let mut a00 = rhs.x;
        let mut a10 = rhs.y;
        let mut a20 = rhs.z;
        let mut a30 = rhs.w;
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
        }
        rhs = Vector4 { x: a00, y: a10, z: a20, w: a30 };
    }
}

pub impl Perm4PermuteColumnsRowVector4<
    T, +Copy<T>, +Drop<T>,
> of PermuteColumns<Perm4, RowVector4<T>> {
    fn permute_columns(self: Perm4, ref rhs: RowVector4<T>) {
        let mut a00 = rhs.x;
        let mut a01 = rhs.y;
        let mut a02 = rhs.z;
        let mut a03 = rhs.w;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
        }
        rhs = RowVector4 { x: a00, y: a01, z: a02, w: a03 };
    }

    fn inv_permute_columns(self: Perm4, ref rhs: RowVector4<T>) {
        let mut a00 = rhs.x;
        let mut a01 = rhs.y;
        let mut a02 = rhs.z;
        let mut a03 = rhs.w;
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
        }
        rhs = RowVector4 { x: a00, y: a01, z: a02, w: a03 };
    }
}
// crate-map: end
