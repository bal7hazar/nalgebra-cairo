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

use nalgebra_core::base::errors::{INDEX_OUT_OF_BOUNDS, PERMUTATION_ORDER};
use nalgebra_types4::linalg::lu::Perm4;
use simba::scalar::Real;

/// Methods of `Perm4` (upstream `PermutationSequence<U4>`). The row / column permutations
/// are the generic `PermuteRows` / `PermuteColumns` (`linalg/permutation_sequence.cairo`).
#[generate_trait]
pub impl Perm4Impl of Perm4Trait {
    /// The identity permutation of a 4x4 factorisation (no swap). Upstream:
    /// `PermutationSequence::identity`.
    #[inline(always)]
    fn identity() -> Perm4 {
        Perm4 { p1: 1, p2: 2, p3: 3 }
    }

    /// Records the transposition of the rows (or columns) `i` and `i2` (0-based, like upstream)
    /// after those already recorded; `i == i2` records nothing. Upstream:
    /// `PermutationSequence::append_permutation`.
    ///
    /// The compact sequence stores ONE transposition per elimination step, `(k, p_k)` with `p_k >=
    /// k`, in step order: the transposition `(min, max)` is recorded as the step `min`, which
    /// must come after every step already recorded — what every upstream decomposition does
    /// (`LU`, `FullPivLU` and `ColPivQR` append `(i, piv)` with `piv >= i` at step `i`). Panics
    /// with `nalgebra: permutation order` otherwise (upstream's heap sequence only panics when it
    /// is full: "Maximum number of permutations exceeded."), and with `nalgebra: index out of
    /// bounds` when an index is `>= 4` (upstream panics when the permutation is applied).
    fn append_permutation(ref self: Perm4, i: usize, i2: usize) {
        if i != i2 {
            let (lo, hi) = if i < i2 {
                (i, i2)
            } else {
                (i2, i)
            };
            assert(hi < 4, INDEX_OUT_OF_BOUNDS);
            let last: usize = if self.p3 != 3 {
                3
            } else if self.p2 != 2 {
                2
            } else if self.p1 != 1 {
                1
            } else {
                0
            };
            assert(lo >= last, PERMUTATION_ORDER);
            let v: u8 = (hi + 1).try_into().unwrap();
            match lo {
                0 => self.p1 = v,
                1 => self.p2 = v,
                2 => self.p3 = v,
                _ => {},
            }
        }
    }

    /// The number of transpositions actually recorded (the steps `k` with `p_k != k`). Upstream:
    /// `PermutationSequence::len`.
    fn len(self: Perm4) -> usize {
        let mut n: usize = 0;
        if self.p1 != 1 {
            n += 1;
        }
        if self.p2 != 2 {
            n += 1;
        }
        if self.p3 != 3 {
            n += 1;
        }
        n
    }

    /// Whether no transposition is recorded (the identity). Upstream:
    /// `PermutationSequence::is_empty`.
    #[inline(always)]
    fn is_empty(self: Perm4) -> bool {
        self.p1 == 1 && self.p2 == 2 && self.p3 == 3
    }

    /// The determinant of the permutation: `1` for an even number of transpositions, `-1` for an
    /// odd one. Exact. Upstream: `PermutationSequence::determinant`.
    fn determinant<T, impl R: Real<T>, +Neg<T>, +Drop<T>>(self: Perm4) -> T {
        let mut odd = false;
        if self.p1 != 1 {
            odd = !odd;
        }
        if self.p2 != 2 {
            odd = !odd;
        }
        if self.p3 != 3 {
            odd = !odd;
        }
        if odd {
            -R::one()
        } else {
            R::one()
        }
    }
}

pub mod lu4;
