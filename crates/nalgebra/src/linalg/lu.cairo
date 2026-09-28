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

pub use lu2::{Lu2, Lu2Trait, Matrix2LuTrait};
pub use lu3::{Lu3, Lu3Trait, Matrix3LuTrait};
pub use lu4::{Lu4, Lu4Trait, Matrix4LuTrait};
pub use lu6::{Lu6, Lu6Trait, Matrix6LuTrait};
pub use perm1_5::{Perm1, Perm1Trait, Perm5, Perm5Trait};
use simba::scalar::Real;
use crate::base::errors::{INDEX_OUT_OF_BOUNDS, PERMUTATION_ORDER};
pub mod lu2;
pub mod lu3;
pub mod lu4;
pub mod lu6;
#[cfg(test)]
mod oracle_lu2;
#[cfg(test)]
mod oracle_lu3;
#[cfg(test)]
mod oracle_lu4;
#[cfg(test)]
mod oracle_lu6;
pub mod perm1_5;
pub use nalgebra_core::linalg::lu::*;

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm2PartialEq of PartialEq<Perm2> {
    fn eq(lhs: @Perm2, rhs: @Perm2) -> bool {
        lhs.p1 == rhs.p1
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm3PartialEq of PartialEq<Perm3> {
    fn eq(lhs: @Perm3, rhs: @Perm3) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm4PartialEq of PartialEq<Perm4> {
    fn eq(lhs: @Perm4, rhs: @Perm4) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2 && lhs.p3 == rhs.p3
    }
}

/// The row permutation of a 6x6 factorisation: the transpositions of steps 1 to 5.
///
/// `pK` is the 1-based index of the row swapped with row `K` at step `K` (`pK == K` = no swap),
/// so `P = T5 * T4 * T3 * T2 * T1`. Upstream: a `PermutationSequence`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm6 {
    /// Row swapped with row 1 at step 1, in `1..=6`.
    pub p1: u8,
    /// Row swapped with row 2 at step 2, in `2..=6`.
    pub p2: u8,
    /// Row swapped with row 3 at step 3, in `3..=6`.
    pub p3: u8,
    /// Row swapped with row 4 at step 4, in `4..=6`.
    pub p4: u8,
    /// Row swapped with row 5 at step 5, in `5..=6`.
    pub p5: u8,
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm6PartialEq of PartialEq<Perm6> {
    fn eq(lhs: @Perm6, rhs: @Perm6) -> bool {
        lhs.p1 == rhs.p1
            && lhs.p2 == rhs.p2
            && lhs.p3 == rhs.p3
            && lhs.p4 == rhs.p4
            && lhs.p5 == rhs.p5
    }
}

/// Methods of `Perm6` (upstream `PermutationSequence<U6>`). The row / column permutations
/// are the generic `PermuteRows` / `PermuteColumns` (`linalg/permutation_sequence.cairo`).
#[generate_trait]
pub impl Perm6Impl of Perm6Trait {
    /// The identity permutation of a 6x6 factorisation (no swap). Upstream:
    /// `PermutationSequence::identity`.
    #[inline(always)]
    fn identity() -> Perm6 {
        Perm6 { p1: 1, p2: 2, p3: 3, p4: 4, p5: 5 }
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
    /// bounds` when an index is `>= 6` (upstream panics when the permutation is applied).
    fn append_permutation(ref self: Perm6, i: usize, i2: usize) {
        if i != i2 {
            let (lo, hi) = if i < i2 {
                (i, i2)
            } else {
                (i2, i)
            };
            assert(hi < 6, INDEX_OUT_OF_BOUNDS);
            let last: usize = if self.p5 != 5 {
                5
            } else if self.p4 != 4 {
                4
            } else if self.p3 != 3 {
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
                3 => self.p4 = v,
                4 => self.p5 = v,
                _ => {},
            }
        }
    }

    /// The number of transpositions actually recorded (the steps `k` with `p_k != k`). Upstream:
    /// `PermutationSequence::len`.
    fn len(self: Perm6) -> usize {
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
        if self.p4 != 4 {
            n += 1;
        }
        if self.p5 != 5 {
            n += 1;
        }
        n
    }

    /// Whether no transposition is recorded (the identity). Upstream:
    /// `PermutationSequence::is_empty`.
    #[inline(always)]
    fn is_empty(self: Perm6) -> bool {
        self.p1 == 1 && self.p2 == 2 && self.p3 == 3 && self.p4 == 4 && self.p5 == 5
    }

    /// The determinant of the permutation: `1` for an even number of transpositions, `-1` for an
    /// odd one. Exact. Upstream: `PermutationSequence::determinant`.
    fn determinant<T, impl R: Real<T>, +Neg<T>, +Drop<T>>(self: Perm6) -> T {
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
        if self.p4 != 4 {
            odd = !odd;
        }
        if self.p5 != 5 {
            odd = !odd;
        }
        if odd {
            -R::one()
        } else {
            R::one()
        }
    }
}
use nalgebra_core::linalg::permutation_sequence::{PermuteColumns, PermuteRows};
use crate::base::row_vector6::RowVector6;
use crate::base::vector6::Vector6;

// crate-map: generated items (tools/split/cratemap.py) [shapegen]
// crate-map: from linalg/permutation_sequence.cairo
pub impl Perm6PermuteRowsVector6<T, +Copy<T>, +Drop<T>> of PermuteRows<Perm6, Vector6<T>> {
    fn permute_rows(self: Perm6, ref rhs: Vector6<T>) {
        let mut a00 = rhs.x;
        let mut a10 = rhs.y;
        let mut a20 = rhs.z;
        let mut a30 = rhs.w;
        let mut a40 = rhs.a;
        let mut a50 = rhs.b;
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
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
        }
        rhs = Vector6 { x: a00, y: a10, z: a20, w: a30, a: a40, b: a50 };
    }

    fn inv_permute_rows(self: Perm6, ref rhs: Vector6<T>) {
        let mut a00 = rhs.x;
        let mut a10 = rhs.y;
        let mut a20 = rhs.z;
        let mut a30 = rhs.w;
        let mut a40 = rhs.a;
        let mut a50 = rhs.b;
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
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
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
        }
        rhs = Vector6 { x: a00, y: a10, z: a20, w: a30, a: a40, b: a50 };
    }
}

pub impl Perm6PermuteColumnsRowVector6<
    T, +Copy<T>, +Drop<T>,
> of PermuteColumns<Perm6, RowVector6<T>> {
    fn permute_columns(self: Perm6, ref rhs: RowVector6<T>) {
        let mut a00 = rhs.x;
        let mut a01 = rhs.y;
        let mut a02 = rhs.z;
        let mut a03 = rhs.w;
        let mut a04 = rhs.a;
        let mut a05 = rhs.b;
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
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
        }
        rhs = RowVector6 { x: a00, y: a01, z: a02, w: a03, a: a04, b: a05 };
    }

    fn inv_permute_columns(self: Perm6, ref rhs: RowVector6<T>) {
        let mut a00 = rhs.x;
        let mut a01 = rhs.y;
        let mut a02 = rhs.z;
        let mut a03 = rhs.w;
        let mut a04 = rhs.a;
        let mut a05 = rhs.b;
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
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
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
        }
        rhs = RowVector6 { x: a00, y: a01, z: a02, w: a03, a: a04, b: a05 };
    }
}
// crate-map: end
