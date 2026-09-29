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
use nalgebra_core::linalg::permutation_sequence::{PermuteColumns, PermuteRows};
use nalgebra_shapes6::base::matrix2x6::Matrix2x6;
use nalgebra_shapes6::base::matrix3x6::Matrix3x6;
use nalgebra_shapes6::base::matrix4x6::Matrix4x6;
use nalgebra_shapes6::base::matrix5x6::Matrix5x6;
use nalgebra_shapes6::base::matrix6::Matrix6;
use nalgebra_shapes6::base::matrix6x2::Matrix6x2;
use nalgebra_shapes6::base::matrix6x3::Matrix6x3;
use nalgebra_shapes6::base::matrix6x4::Matrix6x4;
use nalgebra_shapes6::base::matrix6x5::Matrix6x5;
use nalgebra_shapes6::base::row_vector6::RowVector6;
use nalgebra_shapes6::base::vector6::Vector6;
use simba::scalar::Real;
pub mod lu6;

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

pub impl Perm6PermuteRowsMatrix6x2<T, +Copy<T>, +Drop<T>> of PermuteRows<Perm6, Matrix6x2<T>> {
    fn permute_rows(self: Perm6, ref rhs: Matrix6x2<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
        }
        rhs =
            Matrix6x2 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
            };
    }

    fn inv_permute_rows(self: Perm6, ref rhs: Matrix6x2<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
        }
        rhs =
            Matrix6x2 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
            };
    }
}

pub impl Perm6PermuteRowsMatrix6x3<T, +Copy<T>, +Drop<T>> of PermuteRows<Perm6, Matrix6x3<T>> {
    fn permute_rows(self: Perm6, ref rhs: Matrix6x3<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
            let tmp = a02;
            a02 = a12;
            a12 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
            let tmp = a02;
            a02 = a22;
            a22 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
            let tmp = a02;
            a02 = a32;
            a32 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
            let tmp = a02;
            a02 = a42;
            a42 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
            let tmp = a02;
            a02 = a52;
            a52 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
            let tmp = a12;
            a12 = a22;
            a22 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
            let tmp = a12;
            a12 = a32;
            a32 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
            let tmp = a12;
            a12 = a42;
            a42 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
            let tmp = a12;
            a12 = a52;
            a52 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
            let tmp = a22;
            a22 = a32;
            a32 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
            let tmp = a22;
            a22 = a42;
            a42 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
            let tmp = a22;
            a22 = a52;
            a52 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
            let tmp = a32;
            a32 = a42;
            a42 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
            let tmp = a32;
            a32 = a52;
            a52 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
            let tmp = a42;
            a42 = a52;
            a52 = tmp;
        }
        rhs =
            Matrix6x3 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
            };
    }

    fn inv_permute_rows(self: Perm6, ref rhs: Matrix6x3<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
            let tmp = a42;
            a42 = a52;
            a52 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
            let tmp = a32;
            a32 = a42;
            a42 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
            let tmp = a32;
            a32 = a52;
            a52 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
            let tmp = a22;
            a22 = a32;
            a32 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
            let tmp = a22;
            a22 = a42;
            a42 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
            let tmp = a22;
            a22 = a52;
            a52 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
            let tmp = a12;
            a12 = a22;
            a22 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
            let tmp = a12;
            a12 = a32;
            a32 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
            let tmp = a12;
            a12 = a42;
            a42 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
            let tmp = a12;
            a12 = a52;
            a52 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
            let tmp = a02;
            a02 = a12;
            a12 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
            let tmp = a02;
            a02 = a22;
            a22 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
            let tmp = a02;
            a02 = a32;
            a32 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
            let tmp = a02;
            a02 = a42;
            a42 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
            let tmp = a02;
            a02 = a52;
            a52 = tmp;
        }
        rhs =
            Matrix6x3 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
            };
    }
}

pub impl Perm6PermuteRowsMatrix6x4<T, +Copy<T>, +Drop<T>> of PermuteRows<Perm6, Matrix6x4<T>> {
    fn permute_rows(self: Perm6, ref rhs: Matrix6x4<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a53 = rhs.m64;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
            let tmp = a02;
            a02 = a12;
            a12 = tmp;
            let tmp = a03;
            a03 = a13;
            a13 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
            let tmp = a02;
            a02 = a22;
            a22 = tmp;
            let tmp = a03;
            a03 = a23;
            a23 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
            let tmp = a02;
            a02 = a32;
            a32 = tmp;
            let tmp = a03;
            a03 = a33;
            a33 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
            let tmp = a02;
            a02 = a42;
            a42 = tmp;
            let tmp = a03;
            a03 = a43;
            a43 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
            let tmp = a02;
            a02 = a52;
            a52 = tmp;
            let tmp = a03;
            a03 = a53;
            a53 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
            let tmp = a12;
            a12 = a22;
            a22 = tmp;
            let tmp = a13;
            a13 = a23;
            a23 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
            let tmp = a12;
            a12 = a32;
            a32 = tmp;
            let tmp = a13;
            a13 = a33;
            a33 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
            let tmp = a12;
            a12 = a42;
            a42 = tmp;
            let tmp = a13;
            a13 = a43;
            a43 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
            let tmp = a12;
            a12 = a52;
            a52 = tmp;
            let tmp = a13;
            a13 = a53;
            a53 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
            let tmp = a22;
            a22 = a32;
            a32 = tmp;
            let tmp = a23;
            a23 = a33;
            a33 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
            let tmp = a22;
            a22 = a42;
            a42 = tmp;
            let tmp = a23;
            a23 = a43;
            a43 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
            let tmp = a22;
            a22 = a52;
            a52 = tmp;
            let tmp = a23;
            a23 = a53;
            a53 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
            let tmp = a32;
            a32 = a42;
            a42 = tmp;
            let tmp = a33;
            a33 = a43;
            a43 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
            let tmp = a32;
            a32 = a52;
            a52 = tmp;
            let tmp = a33;
            a33 = a53;
            a53 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
            let tmp = a42;
            a42 = a52;
            a52 = tmp;
            let tmp = a43;
            a43 = a53;
            a53 = tmp;
        }
        rhs =
            Matrix6x4 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m64: a53,
            };
    }

    fn inv_permute_rows(self: Perm6, ref rhs: Matrix6x4<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a53 = rhs.m64;
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
            let tmp = a42;
            a42 = a52;
            a52 = tmp;
            let tmp = a43;
            a43 = a53;
            a53 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
            let tmp = a32;
            a32 = a42;
            a42 = tmp;
            let tmp = a33;
            a33 = a43;
            a43 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
            let tmp = a32;
            a32 = a52;
            a52 = tmp;
            let tmp = a33;
            a33 = a53;
            a53 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
            let tmp = a22;
            a22 = a32;
            a32 = tmp;
            let tmp = a23;
            a23 = a33;
            a33 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
            let tmp = a22;
            a22 = a42;
            a42 = tmp;
            let tmp = a23;
            a23 = a43;
            a43 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
            let tmp = a22;
            a22 = a52;
            a52 = tmp;
            let tmp = a23;
            a23 = a53;
            a53 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
            let tmp = a12;
            a12 = a22;
            a22 = tmp;
            let tmp = a13;
            a13 = a23;
            a23 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
            let tmp = a12;
            a12 = a32;
            a32 = tmp;
            let tmp = a13;
            a13 = a33;
            a33 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
            let tmp = a12;
            a12 = a42;
            a42 = tmp;
            let tmp = a13;
            a13 = a43;
            a43 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
            let tmp = a12;
            a12 = a52;
            a52 = tmp;
            let tmp = a13;
            a13 = a53;
            a53 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
            let tmp = a02;
            a02 = a12;
            a12 = tmp;
            let tmp = a03;
            a03 = a13;
            a13 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
            let tmp = a02;
            a02 = a22;
            a22 = tmp;
            let tmp = a03;
            a03 = a23;
            a23 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
            let tmp = a02;
            a02 = a32;
            a32 = tmp;
            let tmp = a03;
            a03 = a33;
            a33 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
            let tmp = a02;
            a02 = a42;
            a42 = tmp;
            let tmp = a03;
            a03 = a43;
            a43 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
            let tmp = a02;
            a02 = a52;
            a52 = tmp;
            let tmp = a03;
            a03 = a53;
            a53 = tmp;
        }
        rhs =
            Matrix6x4 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m64: a53,
            };
    }
}

pub impl Perm6PermuteRowsMatrix6x5<T, +Copy<T>, +Drop<T>> of PermuteRows<Perm6, Matrix6x5<T>> {
    fn permute_rows(self: Perm6, ref rhs: Matrix6x5<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a53 = rhs.m64;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a44 = rhs.m55;
        let mut a54 = rhs.m65;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
            let tmp = a02;
            a02 = a12;
            a12 = tmp;
            let tmp = a03;
            a03 = a13;
            a13 = tmp;
            let tmp = a04;
            a04 = a14;
            a14 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
            let tmp = a02;
            a02 = a22;
            a22 = tmp;
            let tmp = a03;
            a03 = a23;
            a23 = tmp;
            let tmp = a04;
            a04 = a24;
            a24 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
            let tmp = a02;
            a02 = a32;
            a32 = tmp;
            let tmp = a03;
            a03 = a33;
            a33 = tmp;
            let tmp = a04;
            a04 = a34;
            a34 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
            let tmp = a02;
            a02 = a42;
            a42 = tmp;
            let tmp = a03;
            a03 = a43;
            a43 = tmp;
            let tmp = a04;
            a04 = a44;
            a44 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
            let tmp = a02;
            a02 = a52;
            a52 = tmp;
            let tmp = a03;
            a03 = a53;
            a53 = tmp;
            let tmp = a04;
            a04 = a54;
            a54 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
            let tmp = a12;
            a12 = a22;
            a22 = tmp;
            let tmp = a13;
            a13 = a23;
            a23 = tmp;
            let tmp = a14;
            a14 = a24;
            a24 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
            let tmp = a12;
            a12 = a32;
            a32 = tmp;
            let tmp = a13;
            a13 = a33;
            a33 = tmp;
            let tmp = a14;
            a14 = a34;
            a34 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
            let tmp = a12;
            a12 = a42;
            a42 = tmp;
            let tmp = a13;
            a13 = a43;
            a43 = tmp;
            let tmp = a14;
            a14 = a44;
            a44 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
            let tmp = a12;
            a12 = a52;
            a52 = tmp;
            let tmp = a13;
            a13 = a53;
            a53 = tmp;
            let tmp = a14;
            a14 = a54;
            a54 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
            let tmp = a22;
            a22 = a32;
            a32 = tmp;
            let tmp = a23;
            a23 = a33;
            a33 = tmp;
            let tmp = a24;
            a24 = a34;
            a34 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
            let tmp = a22;
            a22 = a42;
            a42 = tmp;
            let tmp = a23;
            a23 = a43;
            a43 = tmp;
            let tmp = a24;
            a24 = a44;
            a44 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
            let tmp = a22;
            a22 = a52;
            a52 = tmp;
            let tmp = a23;
            a23 = a53;
            a53 = tmp;
            let tmp = a24;
            a24 = a54;
            a54 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
            let tmp = a32;
            a32 = a42;
            a42 = tmp;
            let tmp = a33;
            a33 = a43;
            a43 = tmp;
            let tmp = a34;
            a34 = a44;
            a44 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
            let tmp = a32;
            a32 = a52;
            a52 = tmp;
            let tmp = a33;
            a33 = a53;
            a53 = tmp;
            let tmp = a34;
            a34 = a54;
            a54 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
            let tmp = a42;
            a42 = a52;
            a52 = tmp;
            let tmp = a43;
            a43 = a53;
            a53 = tmp;
            let tmp = a44;
            a44 = a54;
            a54 = tmp;
        }
        rhs =
            Matrix6x5 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m64: a53,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m55: a44,
                m65: a54,
            };
    }

    fn inv_permute_rows(self: Perm6, ref rhs: Matrix6x5<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a53 = rhs.m64;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a44 = rhs.m55;
        let mut a54 = rhs.m65;
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
            let tmp = a42;
            a42 = a52;
            a52 = tmp;
            let tmp = a43;
            a43 = a53;
            a53 = tmp;
            let tmp = a44;
            a44 = a54;
            a54 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
            let tmp = a32;
            a32 = a42;
            a42 = tmp;
            let tmp = a33;
            a33 = a43;
            a43 = tmp;
            let tmp = a34;
            a34 = a44;
            a44 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
            let tmp = a32;
            a32 = a52;
            a52 = tmp;
            let tmp = a33;
            a33 = a53;
            a53 = tmp;
            let tmp = a34;
            a34 = a54;
            a54 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
            let tmp = a22;
            a22 = a32;
            a32 = tmp;
            let tmp = a23;
            a23 = a33;
            a33 = tmp;
            let tmp = a24;
            a24 = a34;
            a34 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
            let tmp = a22;
            a22 = a42;
            a42 = tmp;
            let tmp = a23;
            a23 = a43;
            a43 = tmp;
            let tmp = a24;
            a24 = a44;
            a44 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
            let tmp = a22;
            a22 = a52;
            a52 = tmp;
            let tmp = a23;
            a23 = a53;
            a53 = tmp;
            let tmp = a24;
            a24 = a54;
            a54 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
            let tmp = a12;
            a12 = a22;
            a22 = tmp;
            let tmp = a13;
            a13 = a23;
            a23 = tmp;
            let tmp = a14;
            a14 = a24;
            a24 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
            let tmp = a12;
            a12 = a32;
            a32 = tmp;
            let tmp = a13;
            a13 = a33;
            a33 = tmp;
            let tmp = a14;
            a14 = a34;
            a34 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
            let tmp = a12;
            a12 = a42;
            a42 = tmp;
            let tmp = a13;
            a13 = a43;
            a43 = tmp;
            let tmp = a14;
            a14 = a44;
            a44 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
            let tmp = a12;
            a12 = a52;
            a52 = tmp;
            let tmp = a13;
            a13 = a53;
            a53 = tmp;
            let tmp = a14;
            a14 = a54;
            a54 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
            let tmp = a02;
            a02 = a12;
            a12 = tmp;
            let tmp = a03;
            a03 = a13;
            a13 = tmp;
            let tmp = a04;
            a04 = a14;
            a14 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
            let tmp = a02;
            a02 = a22;
            a22 = tmp;
            let tmp = a03;
            a03 = a23;
            a23 = tmp;
            let tmp = a04;
            a04 = a24;
            a24 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
            let tmp = a02;
            a02 = a32;
            a32 = tmp;
            let tmp = a03;
            a03 = a33;
            a33 = tmp;
            let tmp = a04;
            a04 = a34;
            a34 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
            let tmp = a02;
            a02 = a42;
            a42 = tmp;
            let tmp = a03;
            a03 = a43;
            a43 = tmp;
            let tmp = a04;
            a04 = a44;
            a44 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
            let tmp = a02;
            a02 = a52;
            a52 = tmp;
            let tmp = a03;
            a03 = a53;
            a53 = tmp;
            let tmp = a04;
            a04 = a54;
            a54 = tmp;
        }
        rhs =
            Matrix6x5 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m64: a53,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m55: a44,
                m65: a54,
            };
    }
}

pub impl Perm6PermuteRowsMatrix6<T, +Copy<T>, +Drop<T>> of PermuteRows<Perm6, Matrix6<T>> {
    fn permute_rows(self: Perm6, ref rhs: Matrix6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a53 = rhs.m64;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a44 = rhs.m55;
        let mut a54 = rhs.m65;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        let mut a35 = rhs.m46;
        let mut a45 = rhs.m56;
        let mut a55 = rhs.m66;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
            let tmp = a02;
            a02 = a12;
            a12 = tmp;
            let tmp = a03;
            a03 = a13;
            a13 = tmp;
            let tmp = a04;
            a04 = a14;
            a14 = tmp;
            let tmp = a05;
            a05 = a15;
            a15 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
            let tmp = a02;
            a02 = a22;
            a22 = tmp;
            let tmp = a03;
            a03 = a23;
            a23 = tmp;
            let tmp = a04;
            a04 = a24;
            a24 = tmp;
            let tmp = a05;
            a05 = a25;
            a25 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
            let tmp = a02;
            a02 = a32;
            a32 = tmp;
            let tmp = a03;
            a03 = a33;
            a33 = tmp;
            let tmp = a04;
            a04 = a34;
            a34 = tmp;
            let tmp = a05;
            a05 = a35;
            a35 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
            let tmp = a02;
            a02 = a42;
            a42 = tmp;
            let tmp = a03;
            a03 = a43;
            a43 = tmp;
            let tmp = a04;
            a04 = a44;
            a44 = tmp;
            let tmp = a05;
            a05 = a45;
            a45 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
            let tmp = a02;
            a02 = a52;
            a52 = tmp;
            let tmp = a03;
            a03 = a53;
            a53 = tmp;
            let tmp = a04;
            a04 = a54;
            a54 = tmp;
            let tmp = a05;
            a05 = a55;
            a55 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
            let tmp = a12;
            a12 = a22;
            a22 = tmp;
            let tmp = a13;
            a13 = a23;
            a23 = tmp;
            let tmp = a14;
            a14 = a24;
            a24 = tmp;
            let tmp = a15;
            a15 = a25;
            a25 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
            let tmp = a12;
            a12 = a32;
            a32 = tmp;
            let tmp = a13;
            a13 = a33;
            a33 = tmp;
            let tmp = a14;
            a14 = a34;
            a34 = tmp;
            let tmp = a15;
            a15 = a35;
            a35 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
            let tmp = a12;
            a12 = a42;
            a42 = tmp;
            let tmp = a13;
            a13 = a43;
            a43 = tmp;
            let tmp = a14;
            a14 = a44;
            a44 = tmp;
            let tmp = a15;
            a15 = a45;
            a45 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
            let tmp = a12;
            a12 = a52;
            a52 = tmp;
            let tmp = a13;
            a13 = a53;
            a53 = tmp;
            let tmp = a14;
            a14 = a54;
            a54 = tmp;
            let tmp = a15;
            a15 = a55;
            a55 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
            let tmp = a22;
            a22 = a32;
            a32 = tmp;
            let tmp = a23;
            a23 = a33;
            a33 = tmp;
            let tmp = a24;
            a24 = a34;
            a34 = tmp;
            let tmp = a25;
            a25 = a35;
            a35 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
            let tmp = a22;
            a22 = a42;
            a42 = tmp;
            let tmp = a23;
            a23 = a43;
            a43 = tmp;
            let tmp = a24;
            a24 = a44;
            a44 = tmp;
            let tmp = a25;
            a25 = a45;
            a45 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
            let tmp = a22;
            a22 = a52;
            a52 = tmp;
            let tmp = a23;
            a23 = a53;
            a53 = tmp;
            let tmp = a24;
            a24 = a54;
            a54 = tmp;
            let tmp = a25;
            a25 = a55;
            a55 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
            let tmp = a32;
            a32 = a42;
            a42 = tmp;
            let tmp = a33;
            a33 = a43;
            a43 = tmp;
            let tmp = a34;
            a34 = a44;
            a44 = tmp;
            let tmp = a35;
            a35 = a45;
            a45 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
            let tmp = a32;
            a32 = a52;
            a52 = tmp;
            let tmp = a33;
            a33 = a53;
            a53 = tmp;
            let tmp = a34;
            a34 = a54;
            a54 = tmp;
            let tmp = a35;
            a35 = a55;
            a55 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
            let tmp = a42;
            a42 = a52;
            a52 = tmp;
            let tmp = a43;
            a43 = a53;
            a53 = tmp;
            let tmp = a44;
            a44 = a54;
            a54 = tmp;
            let tmp = a45;
            a45 = a55;
            a55 = tmp;
        }
        rhs =
            Matrix6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m64: a53,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m55: a44,
                m65: a54,
                m16: a05,
                m26: a15,
                m36: a25,
                m46: a35,
                m56: a45,
                m66: a55,
            };
    }

    fn inv_permute_rows(self: Perm6, ref rhs: Matrix6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a53 = rhs.m64;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a44 = rhs.m55;
        let mut a54 = rhs.m65;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        let mut a35 = rhs.m46;
        let mut a45 = rhs.m56;
        let mut a55 = rhs.m66;
        if self.p5 == 6 {
            let tmp = a40;
            a40 = a50;
            a50 = tmp;
            let tmp = a41;
            a41 = a51;
            a51 = tmp;
            let tmp = a42;
            a42 = a52;
            a52 = tmp;
            let tmp = a43;
            a43 = a53;
            a53 = tmp;
            let tmp = a44;
            a44 = a54;
            a54 = tmp;
            let tmp = a45;
            a45 = a55;
            a55 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a30;
            a30 = a40;
            a40 = tmp;
            let tmp = a31;
            a31 = a41;
            a41 = tmp;
            let tmp = a32;
            a32 = a42;
            a42 = tmp;
            let tmp = a33;
            a33 = a43;
            a43 = tmp;
            let tmp = a34;
            a34 = a44;
            a44 = tmp;
            let tmp = a35;
            a35 = a45;
            a45 = tmp;
        } else if self.p4 == 6 {
            let tmp = a30;
            a30 = a50;
            a50 = tmp;
            let tmp = a31;
            a31 = a51;
            a51 = tmp;
            let tmp = a32;
            a32 = a52;
            a52 = tmp;
            let tmp = a33;
            a33 = a53;
            a53 = tmp;
            let tmp = a34;
            a34 = a54;
            a54 = tmp;
            let tmp = a35;
            a35 = a55;
            a55 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a20;
            a20 = a30;
            a30 = tmp;
            let tmp = a21;
            a21 = a31;
            a31 = tmp;
            let tmp = a22;
            a22 = a32;
            a32 = tmp;
            let tmp = a23;
            a23 = a33;
            a33 = tmp;
            let tmp = a24;
            a24 = a34;
            a34 = tmp;
            let tmp = a25;
            a25 = a35;
            a35 = tmp;
        } else if self.p3 == 5 {
            let tmp = a20;
            a20 = a40;
            a40 = tmp;
            let tmp = a21;
            a21 = a41;
            a41 = tmp;
            let tmp = a22;
            a22 = a42;
            a42 = tmp;
            let tmp = a23;
            a23 = a43;
            a43 = tmp;
            let tmp = a24;
            a24 = a44;
            a44 = tmp;
            let tmp = a25;
            a25 = a45;
            a45 = tmp;
        } else if self.p3 == 6 {
            let tmp = a20;
            a20 = a50;
            a50 = tmp;
            let tmp = a21;
            a21 = a51;
            a51 = tmp;
            let tmp = a22;
            a22 = a52;
            a52 = tmp;
            let tmp = a23;
            a23 = a53;
            a53 = tmp;
            let tmp = a24;
            a24 = a54;
            a54 = tmp;
            let tmp = a25;
            a25 = a55;
            a55 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a10;
            a10 = a20;
            a20 = tmp;
            let tmp = a11;
            a11 = a21;
            a21 = tmp;
            let tmp = a12;
            a12 = a22;
            a22 = tmp;
            let tmp = a13;
            a13 = a23;
            a23 = tmp;
            let tmp = a14;
            a14 = a24;
            a24 = tmp;
            let tmp = a15;
            a15 = a25;
            a25 = tmp;
        } else if self.p2 == 4 {
            let tmp = a10;
            a10 = a30;
            a30 = tmp;
            let tmp = a11;
            a11 = a31;
            a31 = tmp;
            let tmp = a12;
            a12 = a32;
            a32 = tmp;
            let tmp = a13;
            a13 = a33;
            a33 = tmp;
            let tmp = a14;
            a14 = a34;
            a34 = tmp;
            let tmp = a15;
            a15 = a35;
            a35 = tmp;
        } else if self.p2 == 5 {
            let tmp = a10;
            a10 = a40;
            a40 = tmp;
            let tmp = a11;
            a11 = a41;
            a41 = tmp;
            let tmp = a12;
            a12 = a42;
            a42 = tmp;
            let tmp = a13;
            a13 = a43;
            a43 = tmp;
            let tmp = a14;
            a14 = a44;
            a44 = tmp;
            let tmp = a15;
            a15 = a45;
            a45 = tmp;
        } else if self.p2 == 6 {
            let tmp = a10;
            a10 = a50;
            a50 = tmp;
            let tmp = a11;
            a11 = a51;
            a51 = tmp;
            let tmp = a12;
            a12 = a52;
            a52 = tmp;
            let tmp = a13;
            a13 = a53;
            a53 = tmp;
            let tmp = a14;
            a14 = a54;
            a54 = tmp;
            let tmp = a15;
            a15 = a55;
            a55 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a10;
            a10 = tmp;
            let tmp = a01;
            a01 = a11;
            a11 = tmp;
            let tmp = a02;
            a02 = a12;
            a12 = tmp;
            let tmp = a03;
            a03 = a13;
            a13 = tmp;
            let tmp = a04;
            a04 = a14;
            a14 = tmp;
            let tmp = a05;
            a05 = a15;
            a15 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a20;
            a20 = tmp;
            let tmp = a01;
            a01 = a21;
            a21 = tmp;
            let tmp = a02;
            a02 = a22;
            a22 = tmp;
            let tmp = a03;
            a03 = a23;
            a23 = tmp;
            let tmp = a04;
            a04 = a24;
            a24 = tmp;
            let tmp = a05;
            a05 = a25;
            a25 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a30;
            a30 = tmp;
            let tmp = a01;
            a01 = a31;
            a31 = tmp;
            let tmp = a02;
            a02 = a32;
            a32 = tmp;
            let tmp = a03;
            a03 = a33;
            a33 = tmp;
            let tmp = a04;
            a04 = a34;
            a34 = tmp;
            let tmp = a05;
            a05 = a35;
            a35 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a40;
            a40 = tmp;
            let tmp = a01;
            a01 = a41;
            a41 = tmp;
            let tmp = a02;
            a02 = a42;
            a42 = tmp;
            let tmp = a03;
            a03 = a43;
            a43 = tmp;
            let tmp = a04;
            a04 = a44;
            a44 = tmp;
            let tmp = a05;
            a05 = a45;
            a45 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a50;
            a50 = tmp;
            let tmp = a01;
            a01 = a51;
            a51 = tmp;
            let tmp = a02;
            a02 = a52;
            a52 = tmp;
            let tmp = a03;
            a03 = a53;
            a53 = tmp;
            let tmp = a04;
            a04 = a54;
            a54 = tmp;
            let tmp = a05;
            a05 = a55;
            a55 = tmp;
        }
        rhs =
            Matrix6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m64: a53,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m55: a44,
                m65: a54,
                m16: a05,
                m26: a15,
                m36: a25,
                m46: a35,
                m56: a45,
                m66: a55,
            };
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

pub impl Perm6PermuteColumnsMatrix2x6<
    T, +Copy<T>, +Drop<T>,
> of PermuteColumns<Perm6, Matrix2x6<T>> {
    fn permute_columns(self: Perm6, ref rhs: Matrix2x6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
        }
        rhs =
            Matrix2x6 {
                m11: a00,
                m21: a10,
                m12: a01,
                m22: a11,
                m13: a02,
                m23: a12,
                m14: a03,
                m24: a13,
                m15: a04,
                m25: a14,
                m16: a05,
                m26: a15,
            };
    }

    fn inv_permute_columns(self: Perm6, ref rhs: Matrix2x6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
        }
        rhs =
            Matrix2x6 {
                m11: a00,
                m21: a10,
                m12: a01,
                m22: a11,
                m13: a02,
                m23: a12,
                m14: a03,
                m24: a13,
                m15: a04,
                m25: a14,
                m16: a05,
                m26: a15,
            };
    }
}

pub impl Perm6PermuteColumnsMatrix3x6<
    T, +Copy<T>, +Drop<T>,
> of PermuteColumns<Perm6, Matrix3x6<T>> {
    fn permute_columns(self: Perm6, ref rhs: Matrix3x6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
            let tmp = a20;
            a20 = a21;
            a21 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
            let tmp = a20;
            a20 = a22;
            a22 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
            let tmp = a20;
            a20 = a23;
            a23 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
            let tmp = a20;
            a20 = a24;
            a24 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
            let tmp = a20;
            a20 = a25;
            a25 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
            let tmp = a21;
            a21 = a22;
            a22 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
            let tmp = a21;
            a21 = a23;
            a23 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
            let tmp = a21;
            a21 = a24;
            a24 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
            let tmp = a21;
            a21 = a25;
            a25 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
            let tmp = a22;
            a22 = a23;
            a23 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
            let tmp = a22;
            a22 = a24;
            a24 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
            let tmp = a22;
            a22 = a25;
            a25 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
            let tmp = a23;
            a23 = a24;
            a24 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
            let tmp = a23;
            a23 = a25;
            a25 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
            let tmp = a24;
            a24 = a25;
            a25 = tmp;
        }
        rhs =
            Matrix3x6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m12: a01,
                m22: a11,
                m32: a21,
                m13: a02,
                m23: a12,
                m33: a22,
                m14: a03,
                m24: a13,
                m34: a23,
                m15: a04,
                m25: a14,
                m35: a24,
                m16: a05,
                m26: a15,
                m36: a25,
            };
    }

    fn inv_permute_columns(self: Perm6, ref rhs: Matrix3x6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
            let tmp = a24;
            a24 = a25;
            a25 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
            let tmp = a23;
            a23 = a24;
            a24 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
            let tmp = a23;
            a23 = a25;
            a25 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
            let tmp = a22;
            a22 = a23;
            a23 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
            let tmp = a22;
            a22 = a24;
            a24 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
            let tmp = a22;
            a22 = a25;
            a25 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
            let tmp = a21;
            a21 = a22;
            a22 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
            let tmp = a21;
            a21 = a23;
            a23 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
            let tmp = a21;
            a21 = a24;
            a24 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
            let tmp = a21;
            a21 = a25;
            a25 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
            let tmp = a20;
            a20 = a21;
            a21 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
            let tmp = a20;
            a20 = a22;
            a22 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
            let tmp = a20;
            a20 = a23;
            a23 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
            let tmp = a20;
            a20 = a24;
            a24 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
            let tmp = a20;
            a20 = a25;
            a25 = tmp;
        }
        rhs =
            Matrix3x6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m12: a01,
                m22: a11,
                m32: a21,
                m13: a02,
                m23: a12,
                m33: a22,
                m14: a03,
                m24: a13,
                m34: a23,
                m15: a04,
                m25: a14,
                m35: a24,
                m16: a05,
                m26: a15,
                m36: a25,
            };
    }
}

pub impl Perm6PermuteColumnsMatrix4x6<
    T, +Copy<T>, +Drop<T>,
> of PermuteColumns<Perm6, Matrix4x6<T>> {
    fn permute_columns(self: Perm6, ref rhs: Matrix4x6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        let mut a35 = rhs.m46;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
            let tmp = a20;
            a20 = a21;
            a21 = tmp;
            let tmp = a30;
            a30 = a31;
            a31 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
            let tmp = a20;
            a20 = a22;
            a22 = tmp;
            let tmp = a30;
            a30 = a32;
            a32 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
            let tmp = a20;
            a20 = a23;
            a23 = tmp;
            let tmp = a30;
            a30 = a33;
            a33 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
            let tmp = a20;
            a20 = a24;
            a24 = tmp;
            let tmp = a30;
            a30 = a34;
            a34 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
            let tmp = a20;
            a20 = a25;
            a25 = tmp;
            let tmp = a30;
            a30 = a35;
            a35 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
            let tmp = a21;
            a21 = a22;
            a22 = tmp;
            let tmp = a31;
            a31 = a32;
            a32 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
            let tmp = a21;
            a21 = a23;
            a23 = tmp;
            let tmp = a31;
            a31 = a33;
            a33 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
            let tmp = a21;
            a21 = a24;
            a24 = tmp;
            let tmp = a31;
            a31 = a34;
            a34 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
            let tmp = a21;
            a21 = a25;
            a25 = tmp;
            let tmp = a31;
            a31 = a35;
            a35 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
            let tmp = a22;
            a22 = a23;
            a23 = tmp;
            let tmp = a32;
            a32 = a33;
            a33 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
            let tmp = a22;
            a22 = a24;
            a24 = tmp;
            let tmp = a32;
            a32 = a34;
            a34 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
            let tmp = a22;
            a22 = a25;
            a25 = tmp;
            let tmp = a32;
            a32 = a35;
            a35 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
            let tmp = a23;
            a23 = a24;
            a24 = tmp;
            let tmp = a33;
            a33 = a34;
            a34 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
            let tmp = a23;
            a23 = a25;
            a25 = tmp;
            let tmp = a33;
            a33 = a35;
            a35 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
            let tmp = a24;
            a24 = a25;
            a25 = tmp;
            let tmp = a34;
            a34 = a35;
            a35 = tmp;
        }
        rhs =
            Matrix4x6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m16: a05,
                m26: a15,
                m36: a25,
                m46: a35,
            };
    }

    fn inv_permute_columns(self: Perm6, ref rhs: Matrix4x6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        let mut a35 = rhs.m46;
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
            let tmp = a24;
            a24 = a25;
            a25 = tmp;
            let tmp = a34;
            a34 = a35;
            a35 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
            let tmp = a23;
            a23 = a24;
            a24 = tmp;
            let tmp = a33;
            a33 = a34;
            a34 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
            let tmp = a23;
            a23 = a25;
            a25 = tmp;
            let tmp = a33;
            a33 = a35;
            a35 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
            let tmp = a22;
            a22 = a23;
            a23 = tmp;
            let tmp = a32;
            a32 = a33;
            a33 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
            let tmp = a22;
            a22 = a24;
            a24 = tmp;
            let tmp = a32;
            a32 = a34;
            a34 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
            let tmp = a22;
            a22 = a25;
            a25 = tmp;
            let tmp = a32;
            a32 = a35;
            a35 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
            let tmp = a21;
            a21 = a22;
            a22 = tmp;
            let tmp = a31;
            a31 = a32;
            a32 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
            let tmp = a21;
            a21 = a23;
            a23 = tmp;
            let tmp = a31;
            a31 = a33;
            a33 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
            let tmp = a21;
            a21 = a24;
            a24 = tmp;
            let tmp = a31;
            a31 = a34;
            a34 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
            let tmp = a21;
            a21 = a25;
            a25 = tmp;
            let tmp = a31;
            a31 = a35;
            a35 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
            let tmp = a20;
            a20 = a21;
            a21 = tmp;
            let tmp = a30;
            a30 = a31;
            a31 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
            let tmp = a20;
            a20 = a22;
            a22 = tmp;
            let tmp = a30;
            a30 = a32;
            a32 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
            let tmp = a20;
            a20 = a23;
            a23 = tmp;
            let tmp = a30;
            a30 = a33;
            a33 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
            let tmp = a20;
            a20 = a24;
            a24 = tmp;
            let tmp = a30;
            a30 = a34;
            a34 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
            let tmp = a20;
            a20 = a25;
            a25 = tmp;
            let tmp = a30;
            a30 = a35;
            a35 = tmp;
        }
        rhs =
            Matrix4x6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m16: a05,
                m26: a15,
                m36: a25,
                m46: a35,
            };
    }
}

pub impl Perm6PermuteColumnsMatrix5x6<
    T, +Copy<T>, +Drop<T>,
> of PermuteColumns<Perm6, Matrix5x6<T>> {
    fn permute_columns(self: Perm6, ref rhs: Matrix5x6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a44 = rhs.m55;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        let mut a35 = rhs.m46;
        let mut a45 = rhs.m56;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
            let tmp = a20;
            a20 = a21;
            a21 = tmp;
            let tmp = a30;
            a30 = a31;
            a31 = tmp;
            let tmp = a40;
            a40 = a41;
            a41 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
            let tmp = a20;
            a20 = a22;
            a22 = tmp;
            let tmp = a30;
            a30 = a32;
            a32 = tmp;
            let tmp = a40;
            a40 = a42;
            a42 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
            let tmp = a20;
            a20 = a23;
            a23 = tmp;
            let tmp = a30;
            a30 = a33;
            a33 = tmp;
            let tmp = a40;
            a40 = a43;
            a43 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
            let tmp = a20;
            a20 = a24;
            a24 = tmp;
            let tmp = a30;
            a30 = a34;
            a34 = tmp;
            let tmp = a40;
            a40 = a44;
            a44 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
            let tmp = a20;
            a20 = a25;
            a25 = tmp;
            let tmp = a30;
            a30 = a35;
            a35 = tmp;
            let tmp = a40;
            a40 = a45;
            a45 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
            let tmp = a21;
            a21 = a22;
            a22 = tmp;
            let tmp = a31;
            a31 = a32;
            a32 = tmp;
            let tmp = a41;
            a41 = a42;
            a42 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
            let tmp = a21;
            a21 = a23;
            a23 = tmp;
            let tmp = a31;
            a31 = a33;
            a33 = tmp;
            let tmp = a41;
            a41 = a43;
            a43 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
            let tmp = a21;
            a21 = a24;
            a24 = tmp;
            let tmp = a31;
            a31 = a34;
            a34 = tmp;
            let tmp = a41;
            a41 = a44;
            a44 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
            let tmp = a21;
            a21 = a25;
            a25 = tmp;
            let tmp = a31;
            a31 = a35;
            a35 = tmp;
            let tmp = a41;
            a41 = a45;
            a45 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
            let tmp = a22;
            a22 = a23;
            a23 = tmp;
            let tmp = a32;
            a32 = a33;
            a33 = tmp;
            let tmp = a42;
            a42 = a43;
            a43 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
            let tmp = a22;
            a22 = a24;
            a24 = tmp;
            let tmp = a32;
            a32 = a34;
            a34 = tmp;
            let tmp = a42;
            a42 = a44;
            a44 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
            let tmp = a22;
            a22 = a25;
            a25 = tmp;
            let tmp = a32;
            a32 = a35;
            a35 = tmp;
            let tmp = a42;
            a42 = a45;
            a45 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
            let tmp = a23;
            a23 = a24;
            a24 = tmp;
            let tmp = a33;
            a33 = a34;
            a34 = tmp;
            let tmp = a43;
            a43 = a44;
            a44 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
            let tmp = a23;
            a23 = a25;
            a25 = tmp;
            let tmp = a33;
            a33 = a35;
            a35 = tmp;
            let tmp = a43;
            a43 = a45;
            a45 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
            let tmp = a24;
            a24 = a25;
            a25 = tmp;
            let tmp = a34;
            a34 = a35;
            a35 = tmp;
            let tmp = a44;
            a44 = a45;
            a45 = tmp;
        }
        rhs =
            Matrix5x6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m55: a44,
                m16: a05,
                m26: a15,
                m36: a25,
                m46: a35,
                m56: a45,
            };
    }

    fn inv_permute_columns(self: Perm6, ref rhs: Matrix5x6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a44 = rhs.m55;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        let mut a35 = rhs.m46;
        let mut a45 = rhs.m56;
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
            let tmp = a24;
            a24 = a25;
            a25 = tmp;
            let tmp = a34;
            a34 = a35;
            a35 = tmp;
            let tmp = a44;
            a44 = a45;
            a45 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
            let tmp = a23;
            a23 = a24;
            a24 = tmp;
            let tmp = a33;
            a33 = a34;
            a34 = tmp;
            let tmp = a43;
            a43 = a44;
            a44 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
            let tmp = a23;
            a23 = a25;
            a25 = tmp;
            let tmp = a33;
            a33 = a35;
            a35 = tmp;
            let tmp = a43;
            a43 = a45;
            a45 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
            let tmp = a22;
            a22 = a23;
            a23 = tmp;
            let tmp = a32;
            a32 = a33;
            a33 = tmp;
            let tmp = a42;
            a42 = a43;
            a43 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
            let tmp = a22;
            a22 = a24;
            a24 = tmp;
            let tmp = a32;
            a32 = a34;
            a34 = tmp;
            let tmp = a42;
            a42 = a44;
            a44 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
            let tmp = a22;
            a22 = a25;
            a25 = tmp;
            let tmp = a32;
            a32 = a35;
            a35 = tmp;
            let tmp = a42;
            a42 = a45;
            a45 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
            let tmp = a21;
            a21 = a22;
            a22 = tmp;
            let tmp = a31;
            a31 = a32;
            a32 = tmp;
            let tmp = a41;
            a41 = a42;
            a42 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
            let tmp = a21;
            a21 = a23;
            a23 = tmp;
            let tmp = a31;
            a31 = a33;
            a33 = tmp;
            let tmp = a41;
            a41 = a43;
            a43 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
            let tmp = a21;
            a21 = a24;
            a24 = tmp;
            let tmp = a31;
            a31 = a34;
            a34 = tmp;
            let tmp = a41;
            a41 = a44;
            a44 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
            let tmp = a21;
            a21 = a25;
            a25 = tmp;
            let tmp = a31;
            a31 = a35;
            a35 = tmp;
            let tmp = a41;
            a41 = a45;
            a45 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
            let tmp = a20;
            a20 = a21;
            a21 = tmp;
            let tmp = a30;
            a30 = a31;
            a31 = tmp;
            let tmp = a40;
            a40 = a41;
            a41 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
            let tmp = a20;
            a20 = a22;
            a22 = tmp;
            let tmp = a30;
            a30 = a32;
            a32 = tmp;
            let tmp = a40;
            a40 = a42;
            a42 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
            let tmp = a20;
            a20 = a23;
            a23 = tmp;
            let tmp = a30;
            a30 = a33;
            a33 = tmp;
            let tmp = a40;
            a40 = a43;
            a43 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
            let tmp = a20;
            a20 = a24;
            a24 = tmp;
            let tmp = a30;
            a30 = a34;
            a34 = tmp;
            let tmp = a40;
            a40 = a44;
            a44 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
            let tmp = a20;
            a20 = a25;
            a25 = tmp;
            let tmp = a30;
            a30 = a35;
            a35 = tmp;
            let tmp = a40;
            a40 = a45;
            a45 = tmp;
        }
        rhs =
            Matrix5x6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m55: a44,
                m16: a05,
                m26: a15,
                m36: a25,
                m46: a35,
                m56: a45,
            };
    }
}

pub impl Perm6PermuteColumnsMatrix6<T, +Copy<T>, +Drop<T>> of PermuteColumns<Perm6, Matrix6<T>> {
    fn permute_columns(self: Perm6, ref rhs: Matrix6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a53 = rhs.m64;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a44 = rhs.m55;
        let mut a54 = rhs.m65;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        let mut a35 = rhs.m46;
        let mut a45 = rhs.m56;
        let mut a55 = rhs.m66;
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
            let tmp = a20;
            a20 = a21;
            a21 = tmp;
            let tmp = a30;
            a30 = a31;
            a31 = tmp;
            let tmp = a40;
            a40 = a41;
            a41 = tmp;
            let tmp = a50;
            a50 = a51;
            a51 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
            let tmp = a20;
            a20 = a22;
            a22 = tmp;
            let tmp = a30;
            a30 = a32;
            a32 = tmp;
            let tmp = a40;
            a40 = a42;
            a42 = tmp;
            let tmp = a50;
            a50 = a52;
            a52 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
            let tmp = a20;
            a20 = a23;
            a23 = tmp;
            let tmp = a30;
            a30 = a33;
            a33 = tmp;
            let tmp = a40;
            a40 = a43;
            a43 = tmp;
            let tmp = a50;
            a50 = a53;
            a53 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
            let tmp = a20;
            a20 = a24;
            a24 = tmp;
            let tmp = a30;
            a30 = a34;
            a34 = tmp;
            let tmp = a40;
            a40 = a44;
            a44 = tmp;
            let tmp = a50;
            a50 = a54;
            a54 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
            let tmp = a20;
            a20 = a25;
            a25 = tmp;
            let tmp = a30;
            a30 = a35;
            a35 = tmp;
            let tmp = a40;
            a40 = a45;
            a45 = tmp;
            let tmp = a50;
            a50 = a55;
            a55 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
            let tmp = a21;
            a21 = a22;
            a22 = tmp;
            let tmp = a31;
            a31 = a32;
            a32 = tmp;
            let tmp = a41;
            a41 = a42;
            a42 = tmp;
            let tmp = a51;
            a51 = a52;
            a52 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
            let tmp = a21;
            a21 = a23;
            a23 = tmp;
            let tmp = a31;
            a31 = a33;
            a33 = tmp;
            let tmp = a41;
            a41 = a43;
            a43 = tmp;
            let tmp = a51;
            a51 = a53;
            a53 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
            let tmp = a21;
            a21 = a24;
            a24 = tmp;
            let tmp = a31;
            a31 = a34;
            a34 = tmp;
            let tmp = a41;
            a41 = a44;
            a44 = tmp;
            let tmp = a51;
            a51 = a54;
            a54 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
            let tmp = a21;
            a21 = a25;
            a25 = tmp;
            let tmp = a31;
            a31 = a35;
            a35 = tmp;
            let tmp = a41;
            a41 = a45;
            a45 = tmp;
            let tmp = a51;
            a51 = a55;
            a55 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
            let tmp = a22;
            a22 = a23;
            a23 = tmp;
            let tmp = a32;
            a32 = a33;
            a33 = tmp;
            let tmp = a42;
            a42 = a43;
            a43 = tmp;
            let tmp = a52;
            a52 = a53;
            a53 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
            let tmp = a22;
            a22 = a24;
            a24 = tmp;
            let tmp = a32;
            a32 = a34;
            a34 = tmp;
            let tmp = a42;
            a42 = a44;
            a44 = tmp;
            let tmp = a52;
            a52 = a54;
            a54 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
            let tmp = a22;
            a22 = a25;
            a25 = tmp;
            let tmp = a32;
            a32 = a35;
            a35 = tmp;
            let tmp = a42;
            a42 = a45;
            a45 = tmp;
            let tmp = a52;
            a52 = a55;
            a55 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
            let tmp = a23;
            a23 = a24;
            a24 = tmp;
            let tmp = a33;
            a33 = a34;
            a34 = tmp;
            let tmp = a43;
            a43 = a44;
            a44 = tmp;
            let tmp = a53;
            a53 = a54;
            a54 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
            let tmp = a23;
            a23 = a25;
            a25 = tmp;
            let tmp = a33;
            a33 = a35;
            a35 = tmp;
            let tmp = a43;
            a43 = a45;
            a45 = tmp;
            let tmp = a53;
            a53 = a55;
            a55 = tmp;
        }
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
            let tmp = a24;
            a24 = a25;
            a25 = tmp;
            let tmp = a34;
            a34 = a35;
            a35 = tmp;
            let tmp = a44;
            a44 = a45;
            a45 = tmp;
            let tmp = a54;
            a54 = a55;
            a55 = tmp;
        }
        rhs =
            Matrix6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m64: a53,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m55: a44,
                m65: a54,
                m16: a05,
                m26: a15,
                m36: a25,
                m46: a35,
                m56: a45,
                m66: a55,
            };
    }

    fn inv_permute_columns(self: Perm6, ref rhs: Matrix6<T>) {
        let mut a00 = rhs.m11;
        let mut a10 = rhs.m21;
        let mut a20 = rhs.m31;
        let mut a30 = rhs.m41;
        let mut a40 = rhs.m51;
        let mut a50 = rhs.m61;
        let mut a01 = rhs.m12;
        let mut a11 = rhs.m22;
        let mut a21 = rhs.m32;
        let mut a31 = rhs.m42;
        let mut a41 = rhs.m52;
        let mut a51 = rhs.m62;
        let mut a02 = rhs.m13;
        let mut a12 = rhs.m23;
        let mut a22 = rhs.m33;
        let mut a32 = rhs.m43;
        let mut a42 = rhs.m53;
        let mut a52 = rhs.m63;
        let mut a03 = rhs.m14;
        let mut a13 = rhs.m24;
        let mut a23 = rhs.m34;
        let mut a33 = rhs.m44;
        let mut a43 = rhs.m54;
        let mut a53 = rhs.m64;
        let mut a04 = rhs.m15;
        let mut a14 = rhs.m25;
        let mut a24 = rhs.m35;
        let mut a34 = rhs.m45;
        let mut a44 = rhs.m55;
        let mut a54 = rhs.m65;
        let mut a05 = rhs.m16;
        let mut a15 = rhs.m26;
        let mut a25 = rhs.m36;
        let mut a35 = rhs.m46;
        let mut a45 = rhs.m56;
        let mut a55 = rhs.m66;
        if self.p5 == 6 {
            let tmp = a04;
            a04 = a05;
            a05 = tmp;
            let tmp = a14;
            a14 = a15;
            a15 = tmp;
            let tmp = a24;
            a24 = a25;
            a25 = tmp;
            let tmp = a34;
            a34 = a35;
            a35 = tmp;
            let tmp = a44;
            a44 = a45;
            a45 = tmp;
            let tmp = a54;
            a54 = a55;
            a55 = tmp;
        }
        if self.p4 == 5 {
            let tmp = a03;
            a03 = a04;
            a04 = tmp;
            let tmp = a13;
            a13 = a14;
            a14 = tmp;
            let tmp = a23;
            a23 = a24;
            a24 = tmp;
            let tmp = a33;
            a33 = a34;
            a34 = tmp;
            let tmp = a43;
            a43 = a44;
            a44 = tmp;
            let tmp = a53;
            a53 = a54;
            a54 = tmp;
        } else if self.p4 == 6 {
            let tmp = a03;
            a03 = a05;
            a05 = tmp;
            let tmp = a13;
            a13 = a15;
            a15 = tmp;
            let tmp = a23;
            a23 = a25;
            a25 = tmp;
            let tmp = a33;
            a33 = a35;
            a35 = tmp;
            let tmp = a43;
            a43 = a45;
            a45 = tmp;
            let tmp = a53;
            a53 = a55;
            a55 = tmp;
        }
        if self.p3 == 4 {
            let tmp = a02;
            a02 = a03;
            a03 = tmp;
            let tmp = a12;
            a12 = a13;
            a13 = tmp;
            let tmp = a22;
            a22 = a23;
            a23 = tmp;
            let tmp = a32;
            a32 = a33;
            a33 = tmp;
            let tmp = a42;
            a42 = a43;
            a43 = tmp;
            let tmp = a52;
            a52 = a53;
            a53 = tmp;
        } else if self.p3 == 5 {
            let tmp = a02;
            a02 = a04;
            a04 = tmp;
            let tmp = a12;
            a12 = a14;
            a14 = tmp;
            let tmp = a22;
            a22 = a24;
            a24 = tmp;
            let tmp = a32;
            a32 = a34;
            a34 = tmp;
            let tmp = a42;
            a42 = a44;
            a44 = tmp;
            let tmp = a52;
            a52 = a54;
            a54 = tmp;
        } else if self.p3 == 6 {
            let tmp = a02;
            a02 = a05;
            a05 = tmp;
            let tmp = a12;
            a12 = a15;
            a15 = tmp;
            let tmp = a22;
            a22 = a25;
            a25 = tmp;
            let tmp = a32;
            a32 = a35;
            a35 = tmp;
            let tmp = a42;
            a42 = a45;
            a45 = tmp;
            let tmp = a52;
            a52 = a55;
            a55 = tmp;
        }
        if self.p2 == 3 {
            let tmp = a01;
            a01 = a02;
            a02 = tmp;
            let tmp = a11;
            a11 = a12;
            a12 = tmp;
            let tmp = a21;
            a21 = a22;
            a22 = tmp;
            let tmp = a31;
            a31 = a32;
            a32 = tmp;
            let tmp = a41;
            a41 = a42;
            a42 = tmp;
            let tmp = a51;
            a51 = a52;
            a52 = tmp;
        } else if self.p2 == 4 {
            let tmp = a01;
            a01 = a03;
            a03 = tmp;
            let tmp = a11;
            a11 = a13;
            a13 = tmp;
            let tmp = a21;
            a21 = a23;
            a23 = tmp;
            let tmp = a31;
            a31 = a33;
            a33 = tmp;
            let tmp = a41;
            a41 = a43;
            a43 = tmp;
            let tmp = a51;
            a51 = a53;
            a53 = tmp;
        } else if self.p2 == 5 {
            let tmp = a01;
            a01 = a04;
            a04 = tmp;
            let tmp = a11;
            a11 = a14;
            a14 = tmp;
            let tmp = a21;
            a21 = a24;
            a24 = tmp;
            let tmp = a31;
            a31 = a34;
            a34 = tmp;
            let tmp = a41;
            a41 = a44;
            a44 = tmp;
            let tmp = a51;
            a51 = a54;
            a54 = tmp;
        } else if self.p2 == 6 {
            let tmp = a01;
            a01 = a05;
            a05 = tmp;
            let tmp = a11;
            a11 = a15;
            a15 = tmp;
            let tmp = a21;
            a21 = a25;
            a25 = tmp;
            let tmp = a31;
            a31 = a35;
            a35 = tmp;
            let tmp = a41;
            a41 = a45;
            a45 = tmp;
            let tmp = a51;
            a51 = a55;
            a55 = tmp;
        }
        if self.p1 == 2 {
            let tmp = a00;
            a00 = a01;
            a01 = tmp;
            let tmp = a10;
            a10 = a11;
            a11 = tmp;
            let tmp = a20;
            a20 = a21;
            a21 = tmp;
            let tmp = a30;
            a30 = a31;
            a31 = tmp;
            let tmp = a40;
            a40 = a41;
            a41 = tmp;
            let tmp = a50;
            a50 = a51;
            a51 = tmp;
        } else if self.p1 == 3 {
            let tmp = a00;
            a00 = a02;
            a02 = tmp;
            let tmp = a10;
            a10 = a12;
            a12 = tmp;
            let tmp = a20;
            a20 = a22;
            a22 = tmp;
            let tmp = a30;
            a30 = a32;
            a32 = tmp;
            let tmp = a40;
            a40 = a42;
            a42 = tmp;
            let tmp = a50;
            a50 = a52;
            a52 = tmp;
        } else if self.p1 == 4 {
            let tmp = a00;
            a00 = a03;
            a03 = tmp;
            let tmp = a10;
            a10 = a13;
            a13 = tmp;
            let tmp = a20;
            a20 = a23;
            a23 = tmp;
            let tmp = a30;
            a30 = a33;
            a33 = tmp;
            let tmp = a40;
            a40 = a43;
            a43 = tmp;
            let tmp = a50;
            a50 = a53;
            a53 = tmp;
        } else if self.p1 == 5 {
            let tmp = a00;
            a00 = a04;
            a04 = tmp;
            let tmp = a10;
            a10 = a14;
            a14 = tmp;
            let tmp = a20;
            a20 = a24;
            a24 = tmp;
            let tmp = a30;
            a30 = a34;
            a34 = tmp;
            let tmp = a40;
            a40 = a44;
            a44 = tmp;
            let tmp = a50;
            a50 = a54;
            a54 = tmp;
        } else if self.p1 == 6 {
            let tmp = a00;
            a00 = a05;
            a05 = tmp;
            let tmp = a10;
            a10 = a15;
            a15 = tmp;
            let tmp = a20;
            a20 = a25;
            a25 = tmp;
            let tmp = a30;
            a30 = a35;
            a35 = tmp;
            let tmp = a40;
            a40 = a45;
            a45 = tmp;
            let tmp = a50;
            a50 = a55;
            a55 = tmp;
        }
        rhs =
            Matrix6 {
                m11: a00,
                m21: a10,
                m31: a20,
                m41: a30,
                m51: a40,
                m61: a50,
                m12: a01,
                m22: a11,
                m32: a21,
                m42: a31,
                m52: a41,
                m62: a51,
                m13: a02,
                m23: a12,
                m33: a22,
                m43: a32,
                m53: a42,
                m63: a52,
                m14: a03,
                m24: a13,
                m34: a23,
                m44: a33,
                m54: a43,
                m64: a53,
                m15: a04,
                m25: a14,
                m35: a24,
                m45: a34,
                m55: a44,
                m65: a54,
                m16: a05,
                m26: a15,
                m36: a25,
                m46: a35,
                m56: a45,
                m66: a55,
            };
    }
}
// crate-map: end
