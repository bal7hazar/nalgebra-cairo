//! `Lu6`: the LU factorisation with partial pivoting of a `Matrix6` (upstream
//! `nalgebra::linalg::LU` on a 6x6 matrix).
//!
//! `P * A = L * U`, with `L` unit lower triangular, `U` upper triangular and `P` the product of the
//! 5 row transpositions chosen by partial pivoting. Both factors share one `Matrix6` like upstream
//! (strict lower triangle = `L`, whose unit diagonal is implicit; upper triangle = `U`) and the
//! permutation is the compact `Perm6`.
//!
//! Everything is unrolled (DESIGN D4: no loop in static code) and every sum of products goes
//! through a fused `Real` kernel — `mul_add` for a single product, the explicit `Real::Wide`
//! accumulator beyond that — so each output scalar is floored once and range-checked once
//! (AGENTS.md rule 4).
//!
//! Partial pivoting costs 15 `abs` and comparisons plus 5 conditional row swaps (moves only), and
//! every swap duplicates the row it moves, so it also costs Sierra statements. Dropping it would be
//! cheaper and shorter and it is not an option: without it the pivot of step `k` is whatever sits
//! at `a_kk`, nothing bounds `|l_ik|`, and a matrix as ordinary as a permuted identity factors with
//! a zero pivot. `bench_lu6_new__alt_no_pivot` and `test_no_pivot_candidate_is_wrong` keep the
//! measurement and the counter-example. Upstream has no unpivoted variant either, only `LU`
//! (partial pivoting) and `FullPivLU` (complete pivoting).

use simba::scalar::Real;
use crate::base::matrix3::Matrix3;
use crate::base::matrix6::Matrix6;
use crate::base::vector3::Vector3;
use crate::base::vector6::Vector6;
use super::Perm6;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;

/// The LU factorisation with partial pivoting of a `Matrix6<T>`: `P * A = L * U`.
///
/// `lu` packs both factors (strict lower triangle = `L` without its unit diagonal, upper triangle =
/// `U`) and `p` is the row permutation. Built by `Lu6Trait::new` or `Matrix6LuTrait::lu`. Upstream:
/// `nalgebra::linalg::LU`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Lu6<T> {
    /// `L` (strict lower triangle) and `U` (upper triangle) packed in one matrix.
    pub lu: Matrix6<T>,
    /// The row transpositions applied by partial pivoting.
    pub p: Perm6,
}

/// Methods of `Lu6<T>` for any `Real` scalar.
#[generate_trait]
pub impl Lu6Impl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Lu6Trait<T> {
    /// The LU factorisation of `matrix` with partial pivoting, fully unrolled. Upstream:
    /// `matrix.lu()` / `LU::new(matrix)`.
    ///
    /// Always succeeds, like upstream: a singular matrix simply leaves a zero on the diagonal of
    /// `U` (see `is_invertible`). At step `k`, the row of largest `|a_ik|` among rows `k..6` is
    /// swapped onto the diagonal (the FIRST such row, like upstream's `icamax`), the 15 multipliers
    /// `l_ik = a_ik / a_kk` are each one correctly rounded division, and the trailing submatrix is
    /// updated entry by entry with `Real::mul_add(-l_ik, a_kj, a_ij)`: ONE floor rounding and one
    /// overflow check per entry, never the two roundings of `a_ij - l_ik * a_kj`.
    ///
    /// A pivot column that is exactly zero is skipped — no swap, no permutation, zero multipliers
    /// —
    /// exactly like upstream's `continue`, so the division is never reached with a zero divisor.
    ///
    /// Error model: `l_ik` is off by at most 1 ulp (correctly rounded quotient) and every update
    /// floors once, so after each of the 5 steps an entry of `U` is within about `k * (1 + |a_kj|)`
    /// raw units of its exact value. Partial pivoting keeps `|l_ik| <= 1`, which is what bounds the
    /// growth of the trailing submatrix. Panics with the scalar's overflow error if an update does
    /// not fit.
    fn new(matrix: Matrix6<T>) -> Lu6<T> {
        let mut a11 = matrix.m11.m11;
        let mut a12 = matrix.m11.m12;
        let mut a13 = matrix.m11.m13;
        let mut a14 = matrix.m12.m11;
        let mut a15 = matrix.m12.m12;
        let mut a16 = matrix.m12.m13;
        let mut a21 = matrix.m11.m21;
        let mut a22 = matrix.m11.m22;
        let mut a23 = matrix.m11.m23;
        let mut a24 = matrix.m12.m21;
        let mut a25 = matrix.m12.m22;
        let mut a26 = matrix.m12.m23;
        let mut a31 = matrix.m11.m31;
        let mut a32 = matrix.m11.m32;
        let mut a33 = matrix.m11.m33;
        let mut a34 = matrix.m12.m31;
        let mut a35 = matrix.m12.m32;
        let mut a36 = matrix.m12.m33;
        let mut a41 = matrix.m21.m11;
        let mut a42 = matrix.m21.m12;
        let mut a43 = matrix.m21.m13;
        let mut a44 = matrix.m22.m11;
        let mut a45 = matrix.m22.m12;
        let mut a46 = matrix.m22.m13;
        let mut a51 = matrix.m21.m21;
        let mut a52 = matrix.m21.m22;
        let mut a53 = matrix.m21.m23;
        let mut a54 = matrix.m22.m21;
        let mut a55 = matrix.m22.m22;
        let mut a56 = matrix.m22.m23;
        let mut a61 = matrix.m21.m31;
        let mut a62 = matrix.m21.m32;
        let mut a63 = matrix.m21.m33;
        let mut a64 = matrix.m22.m31;
        let mut a65 = matrix.m22.m32;
        let mut a66 = matrix.m22.m33;
        // step 1: largest pivot among rows 1..6 of column 1
        let mut p1 = 1_u8;
        let mut piv = R::abs(a11);
        let c = R::abs(a21);
        if c > piv {
            piv = c;
            p1 = 2;
        }
        let c = R::abs(a31);
        if c > piv {
            piv = c;
            p1 = 3;
        }
        let c = R::abs(a41);
        if c > piv {
            piv = c;
            p1 = 4;
        }
        let c = R::abs(a51);
        if c > piv {
            piv = c;
            p1 = 5;
        }
        let c = R::abs(a61);
        if c > piv {
            piv = c;
            p1 = 6;
        }
        if p1 == 2 {
            let t = a11;
            a11 = a21;
            a21 = t;
            let t = a12;
            a12 = a22;
            a22 = t;
            let t = a13;
            a13 = a23;
            a23 = t;
            let t = a14;
            a14 = a24;
            a24 = t;
            let t = a15;
            a15 = a25;
            a25 = t;
            let t = a16;
            a16 = a26;
            a26 = t;
        } else if p1 == 3 {
            let t = a11;
            a11 = a31;
            a31 = t;
            let t = a12;
            a12 = a32;
            a32 = t;
            let t = a13;
            a13 = a33;
            a33 = t;
            let t = a14;
            a14 = a34;
            a34 = t;
            let t = a15;
            a15 = a35;
            a35 = t;
            let t = a16;
            a16 = a36;
            a36 = t;
        } else if p1 == 4 {
            let t = a11;
            a11 = a41;
            a41 = t;
            let t = a12;
            a12 = a42;
            a42 = t;
            let t = a13;
            a13 = a43;
            a43 = t;
            let t = a14;
            a14 = a44;
            a44 = t;
            let t = a15;
            a15 = a45;
            a45 = t;
            let t = a16;
            a16 = a46;
            a46 = t;
        } else if p1 == 5 {
            let t = a11;
            a11 = a51;
            a51 = t;
            let t = a12;
            a12 = a52;
            a52 = t;
            let t = a13;
            a13 = a53;
            a53 = t;
            let t = a14;
            a14 = a54;
            a54 = t;
            let t = a15;
            a15 = a55;
            a55 = t;
            let t = a16;
            a16 = a56;
            a56 = t;
        } else if p1 == 6 {
            let t = a11;
            a11 = a61;
            a61 = t;
            let t = a12;
            a12 = a62;
            a62 = t;
            let t = a13;
            a13 = a63;
            a63 = t;
            let t = a14;
            a14 = a64;
            a64 = t;
            let t = a15;
            a15 = a65;
            a65 = t;
            let t = a16;
            a16 = a66;
            a66 = t;
        }
        if piv != R::ZERO {
            let (l_a21, l_a31, l_a41, l_a51, l_a61) = R::div5(a21, a31, a41, a51, a61, a11);
            let l = l_a21;
            let nl = -l;
            a22 = R::mul_add(nl, a12, a22);
            a23 = R::mul_add(nl, a13, a23);
            a24 = R::mul_add(nl, a14, a24);
            a25 = R::mul_add(nl, a15, a25);
            a26 = R::mul_add(nl, a16, a26);
            a21 = l;
            let l = l_a31;
            let nl = -l;
            a32 = R::mul_add(nl, a12, a32);
            a33 = R::mul_add(nl, a13, a33);
            a34 = R::mul_add(nl, a14, a34);
            a35 = R::mul_add(nl, a15, a35);
            a36 = R::mul_add(nl, a16, a36);
            a31 = l;
            let l = l_a41;
            let nl = -l;
            a42 = R::mul_add(nl, a12, a42);
            a43 = R::mul_add(nl, a13, a43);
            a44 = R::mul_add(nl, a14, a44);
            a45 = R::mul_add(nl, a15, a45);
            a46 = R::mul_add(nl, a16, a46);
            a41 = l;
            let l = l_a51;
            let nl = -l;
            a52 = R::mul_add(nl, a12, a52);
            a53 = R::mul_add(nl, a13, a53);
            a54 = R::mul_add(nl, a14, a54);
            a55 = R::mul_add(nl, a15, a55);
            a56 = R::mul_add(nl, a16, a56);
            a51 = l;
            let l = l_a61;
            let nl = -l;
            a62 = R::mul_add(nl, a12, a62);
            a63 = R::mul_add(nl, a13, a63);
            a64 = R::mul_add(nl, a14, a64);
            a65 = R::mul_add(nl, a15, a65);
            a66 = R::mul_add(nl, a16, a66);
            a61 = l;
        }
        // step 2: largest pivot among rows 2..6 of column 2
        let mut p2 = 2_u8;
        let mut piv = R::abs(a22);
        let c = R::abs(a32);
        if c > piv {
            piv = c;
            p2 = 3;
        }
        let c = R::abs(a42);
        if c > piv {
            piv = c;
            p2 = 4;
        }
        let c = R::abs(a52);
        if c > piv {
            piv = c;
            p2 = 5;
        }
        let c = R::abs(a62);
        if c > piv {
            piv = c;
            p2 = 6;
        }
        if p2 == 3 {
            let t = a21;
            a21 = a31;
            a31 = t;
            let t = a22;
            a22 = a32;
            a32 = t;
            let t = a23;
            a23 = a33;
            a33 = t;
            let t = a24;
            a24 = a34;
            a34 = t;
            let t = a25;
            a25 = a35;
            a35 = t;
            let t = a26;
            a26 = a36;
            a36 = t;
        } else if p2 == 4 {
            let t = a21;
            a21 = a41;
            a41 = t;
            let t = a22;
            a22 = a42;
            a42 = t;
            let t = a23;
            a23 = a43;
            a43 = t;
            let t = a24;
            a24 = a44;
            a44 = t;
            let t = a25;
            a25 = a45;
            a45 = t;
            let t = a26;
            a26 = a46;
            a46 = t;
        } else if p2 == 5 {
            let t = a21;
            a21 = a51;
            a51 = t;
            let t = a22;
            a22 = a52;
            a52 = t;
            let t = a23;
            a23 = a53;
            a53 = t;
            let t = a24;
            a24 = a54;
            a54 = t;
            let t = a25;
            a25 = a55;
            a55 = t;
            let t = a26;
            a26 = a56;
            a56 = t;
        } else if p2 == 6 {
            let t = a21;
            a21 = a61;
            a61 = t;
            let t = a22;
            a22 = a62;
            a62 = t;
            let t = a23;
            a23 = a63;
            a63 = t;
            let t = a24;
            a24 = a64;
            a64 = t;
            let t = a25;
            a25 = a65;
            a65 = t;
            let t = a26;
            a26 = a66;
            a66 = t;
        }
        if piv != R::ZERO {
            let (l_a32, l_a42, l_a52, l_a62) = R::div4(a32, a42, a52, a62, a22);
            let l = l_a32;
            let nl = -l;
            a33 = R::mul_add(nl, a23, a33);
            a34 = R::mul_add(nl, a24, a34);
            a35 = R::mul_add(nl, a25, a35);
            a36 = R::mul_add(nl, a26, a36);
            a32 = l;
            let l = l_a42;
            let nl = -l;
            a43 = R::mul_add(nl, a23, a43);
            a44 = R::mul_add(nl, a24, a44);
            a45 = R::mul_add(nl, a25, a45);
            a46 = R::mul_add(nl, a26, a46);
            a42 = l;
            let l = l_a52;
            let nl = -l;
            a53 = R::mul_add(nl, a23, a53);
            a54 = R::mul_add(nl, a24, a54);
            a55 = R::mul_add(nl, a25, a55);
            a56 = R::mul_add(nl, a26, a56);
            a52 = l;
            let l = l_a62;
            let nl = -l;
            a63 = R::mul_add(nl, a23, a63);
            a64 = R::mul_add(nl, a24, a64);
            a65 = R::mul_add(nl, a25, a65);
            a66 = R::mul_add(nl, a26, a66);
            a62 = l;
        }
        // step 3: largest pivot among rows 3..6 of column 3
        let mut p3 = 3_u8;
        let mut piv = R::abs(a33);
        let c = R::abs(a43);
        if c > piv {
            piv = c;
            p3 = 4;
        }
        let c = R::abs(a53);
        if c > piv {
            piv = c;
            p3 = 5;
        }
        let c = R::abs(a63);
        if c > piv {
            piv = c;
            p3 = 6;
        }
        if p3 == 4 {
            let t = a31;
            a31 = a41;
            a41 = t;
            let t = a32;
            a32 = a42;
            a42 = t;
            let t = a33;
            a33 = a43;
            a43 = t;
            let t = a34;
            a34 = a44;
            a44 = t;
            let t = a35;
            a35 = a45;
            a45 = t;
            let t = a36;
            a36 = a46;
            a46 = t;
        } else if p3 == 5 {
            let t = a31;
            a31 = a51;
            a51 = t;
            let t = a32;
            a32 = a52;
            a52 = t;
            let t = a33;
            a33 = a53;
            a53 = t;
            let t = a34;
            a34 = a54;
            a54 = t;
            let t = a35;
            a35 = a55;
            a55 = t;
            let t = a36;
            a36 = a56;
            a56 = t;
        } else if p3 == 6 {
            let t = a31;
            a31 = a61;
            a61 = t;
            let t = a32;
            a32 = a62;
            a62 = t;
            let t = a33;
            a33 = a63;
            a63 = t;
            let t = a34;
            a34 = a64;
            a64 = t;
            let t = a35;
            a35 = a65;
            a65 = t;
            let t = a36;
            a36 = a66;
            a66 = t;
        }
        if piv != R::ZERO {
            let (l_a43, l_a53, l_a63) = R::div3(a43, a53, a63, a33);
            let l = l_a43;
            let nl = -l;
            a44 = R::mul_add(nl, a34, a44);
            a45 = R::mul_add(nl, a35, a45);
            a46 = R::mul_add(nl, a36, a46);
            a43 = l;
            let l = l_a53;
            let nl = -l;
            a54 = R::mul_add(nl, a34, a54);
            a55 = R::mul_add(nl, a35, a55);
            a56 = R::mul_add(nl, a36, a56);
            a53 = l;
            let l = l_a63;
            let nl = -l;
            a64 = R::mul_add(nl, a34, a64);
            a65 = R::mul_add(nl, a35, a65);
            a66 = R::mul_add(nl, a36, a66);
            a63 = l;
        }
        // step 4: largest pivot among rows 4..6 of column 4
        let mut p4 = 4_u8;
        let mut piv = R::abs(a44);
        let c = R::abs(a54);
        if c > piv {
            piv = c;
            p4 = 5;
        }
        let c = R::abs(a64);
        if c > piv {
            piv = c;
            p4 = 6;
        }
        if p4 == 5 {
            let t = a41;
            a41 = a51;
            a51 = t;
            let t = a42;
            a42 = a52;
            a52 = t;
            let t = a43;
            a43 = a53;
            a53 = t;
            let t = a44;
            a44 = a54;
            a54 = t;
            let t = a45;
            a45 = a55;
            a55 = t;
            let t = a46;
            a46 = a56;
            a56 = t;
        } else if p4 == 6 {
            let t = a41;
            a41 = a61;
            a61 = t;
            let t = a42;
            a42 = a62;
            a62 = t;
            let t = a43;
            a43 = a63;
            a63 = t;
            let t = a44;
            a44 = a64;
            a64 = t;
            let t = a45;
            a45 = a65;
            a65 = t;
            let t = a46;
            a46 = a66;
            a66 = t;
        }
        if piv != R::ZERO {
            let l = R::div(a54, a44);
            let nl = -l;
            a55 = R::mul_add(nl, a45, a55);
            a56 = R::mul_add(nl, a46, a56);
            a54 = l;
            let l = R::div(a64, a44);
            let nl = -l;
            a65 = R::mul_add(nl, a45, a65);
            a66 = R::mul_add(nl, a46, a66);
            a64 = l;
        }
        // step 5: largest pivot among rows 5..6 of column 5
        let mut p5 = 5_u8;
        let mut piv = R::abs(a55);
        let c = R::abs(a65);
        if c > piv {
            piv = c;
            p5 = 6;
        }
        if p5 == 6 {
            let t = a51;
            a51 = a61;
            a61 = t;
            let t = a52;
            a52 = a62;
            a62 = t;
            let t = a53;
            a53 = a63;
            a63 = t;
            let t = a54;
            a54 = a64;
            a64 = t;
            let t = a55;
            a55 = a65;
            a65 = t;
            let t = a56;
            a56 = a66;
            a66 = t;
        }
        if piv != R::ZERO {
            let l = R::div(a65, a55);
            let nl = -l;
            a66 = R::mul_add(nl, a56, a66);
            a65 = l;
        }
        Lu6 {
            lu: Matrix6 {
                m11: Matrix3 {
                    m11: a11,
                    m21: a21,
                    m31: a31,
                    m12: a12,
                    m22: a22,
                    m32: a32,
                    m13: a13,
                    m23: a23,
                    m33: a33,
                },
                m21: Matrix3 {
                    m11: a41,
                    m21: a51,
                    m31: a61,
                    m12: a42,
                    m22: a52,
                    m32: a62,
                    m13: a43,
                    m23: a53,
                    m33: a63,
                },
                m12: Matrix3 {
                    m11: a14,
                    m21: a24,
                    m31: a34,
                    m12: a15,
                    m22: a25,
                    m32: a35,
                    m13: a16,
                    m23: a26,
                    m33: a36,
                },
                m22: Matrix3 {
                    m11: a44,
                    m21: a54,
                    m31: a64,
                    m12: a45,
                    m22: a55,
                    m32: a65,
                    m13: a46,
                    m23: a56,
                    m33: a66,
                },
            },
            p: Perm6 { p1, p2, p3, p4, p5 },
        }
    }

    /// The unit lower triangular factor `L` (its diagonal of ones is implicit in the packed
    /// storage). Exact: moves only. Upstream: `LU::l`.
    #[inline(always)]
    fn l(self: Lu6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3 {
                m11: R::ONE,
                m21: self.lu.m11.m21,
                m31: self.lu.m11.m31,
                m12: R::ZERO,
                m22: R::ONE,
                m32: self.lu.m11.m32,
                m13: R::ZERO,
                m23: R::ZERO,
                m33: R::ONE,
            },
            m21: Matrix3 {
                m11: self.lu.m21.m11,
                m21: self.lu.m21.m21,
                m31: self.lu.m21.m31,
                m12: self.lu.m21.m12,
                m22: self.lu.m21.m22,
                m32: self.lu.m21.m32,
                m13: self.lu.m21.m13,
                m23: self.lu.m21.m23,
                m33: self.lu.m21.m33,
            },
            m12: Matrix3 {
                m11: R::ZERO,
                m21: R::ZERO,
                m31: R::ZERO,
                m12: R::ZERO,
                m22: R::ZERO,
                m32: R::ZERO,
                m13: R::ZERO,
                m23: R::ZERO,
                m33: R::ZERO,
            },
            m22: Matrix3 {
                m11: R::ONE,
                m21: self.lu.m22.m21,
                m31: self.lu.m22.m31,
                m12: R::ZERO,
                m22: R::ONE,
                m32: self.lu.m22.m32,
                m13: R::ZERO,
                m23: R::ZERO,
                m33: R::ONE,
            },
        }
    }

    /// The upper triangular factor `U`. Exact: moves only. Upstream: `LU::u`.
    #[inline(always)]
    fn u(self: Lu6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3 {
                m11: self.lu.m11.m11,
                m21: R::ZERO,
                m31: R::ZERO,
                m12: self.lu.m11.m12,
                m22: self.lu.m11.m22,
                m32: R::ZERO,
                m13: self.lu.m11.m13,
                m23: self.lu.m11.m23,
                m33: self.lu.m11.m33,
            },
            m21: Matrix3 {
                m11: R::ZERO,
                m21: R::ZERO,
                m31: R::ZERO,
                m12: R::ZERO,
                m22: R::ZERO,
                m32: R::ZERO,
                m13: R::ZERO,
                m23: R::ZERO,
                m33: R::ZERO,
            },
            m12: Matrix3 {
                m11: self.lu.m12.m11,
                m21: self.lu.m12.m21,
                m31: self.lu.m12.m31,
                m12: self.lu.m12.m12,
                m22: self.lu.m12.m22,
                m32: self.lu.m12.m32,
                m13: self.lu.m12.m13,
                m23: self.lu.m12.m23,
                m33: self.lu.m12.m33,
            },
            m22: Matrix3 {
                m11: self.lu.m22.m11,
                m21: R::ZERO,
                m31: R::ZERO,
                m12: self.lu.m22.m12,
                m22: self.lu.m22.m22,
                m32: R::ZERO,
                m13: self.lu.m22.m13,
                m23: self.lu.m22.m23,
                m33: self.lu.m22.m33,
            },
        }
    }

    /// The row permutation `P`, as the compact sequence of 5 transpositions `Perm6` — never as a
    /// matrix, and never as an array (DESIGN D4). Upstream: `LU::p`, which returns a heap-allocated
    /// `PermutationSequence`.
    #[inline(always)]
    fn p(self: Lu6<T>) -> Perm6 {
        self.p
    }

    /// `P * v`: the transpositions applied to the components of `v`, in factorisation order. Exact:
    /// moves and comparisons only. Upstream: `LU::p().permute_rows(&mut v)`.
    fn permute(self: Lu6<T>, v: Vector6<T>) -> Vector6<T> {
        let mut x1 = v.a.x;
        let mut x2 = v.a.y;
        let mut x3 = v.a.z;
        let mut x4 = v.b.x;
        let mut x5 = v.b.y;
        let mut x6 = v.b.z;
        if self.p.p1 == 2 {
            let t = x1;
            x1 = x2;
            x2 = t;
        } else if self.p.p1 == 3 {
            let t = x1;
            x1 = x3;
            x3 = t;
        } else if self.p.p1 == 4 {
            let t = x1;
            x1 = x4;
            x4 = t;
        } else if self.p.p1 == 5 {
            let t = x1;
            x1 = x5;
            x5 = t;
        } else if self.p.p1 == 6 {
            let t = x1;
            x1 = x6;
            x6 = t;
        }
        if self.p.p2 == 3 {
            let t = x2;
            x2 = x3;
            x3 = t;
        } else if self.p.p2 == 4 {
            let t = x2;
            x2 = x4;
            x4 = t;
        } else if self.p.p2 == 5 {
            let t = x2;
            x2 = x5;
            x5 = t;
        } else if self.p.p2 == 6 {
            let t = x2;
            x2 = x6;
            x6 = t;
        }
        if self.p.p3 == 4 {
            let t = x3;
            x3 = x4;
            x4 = t;
        } else if self.p.p3 == 5 {
            let t = x3;
            x3 = x5;
            x5 = t;
        } else if self.p.p3 == 6 {
            let t = x3;
            x3 = x6;
            x6 = t;
        }
        if self.p.p4 == 5 {
            let t = x4;
            x4 = x5;
            x5 = t;
        } else if self.p.p4 == 6 {
            let t = x4;
            x4 = x6;
            x6 = t;
        }
        if self.p.p5 == 6 {
            let t = x5;
            x5 = x6;
            x6 = t;
        }
        Vector6 { a: Vector3 { x: x1, y: x2, z: x3 }, b: Vector3 { x: x4, y: x5, z: x6 } }
    }

    /// `P * m`: the same transpositions applied to the ROWS of `m`, so that `lu.permute_rows(a)` is
    /// `lu.l() * lu.u()` up to the rounding of the factorisation (this is how the tests check the
    /// decomposition). Exact: moves and comparisons only. Upstream: `LU::p().permute_rows(&mut m)`.
    fn permute_rows(self: Lu6<T>, m: Matrix6<T>) -> Matrix6<T> {
        let mut a11 = m.m11.m11;
        let mut a12 = m.m11.m12;
        let mut a13 = m.m11.m13;
        let mut a14 = m.m12.m11;
        let mut a15 = m.m12.m12;
        let mut a16 = m.m12.m13;
        let mut a21 = m.m11.m21;
        let mut a22 = m.m11.m22;
        let mut a23 = m.m11.m23;
        let mut a24 = m.m12.m21;
        let mut a25 = m.m12.m22;
        let mut a26 = m.m12.m23;
        let mut a31 = m.m11.m31;
        let mut a32 = m.m11.m32;
        let mut a33 = m.m11.m33;
        let mut a34 = m.m12.m31;
        let mut a35 = m.m12.m32;
        let mut a36 = m.m12.m33;
        let mut a41 = m.m21.m11;
        let mut a42 = m.m21.m12;
        let mut a43 = m.m21.m13;
        let mut a44 = m.m22.m11;
        let mut a45 = m.m22.m12;
        let mut a46 = m.m22.m13;
        let mut a51 = m.m21.m21;
        let mut a52 = m.m21.m22;
        let mut a53 = m.m21.m23;
        let mut a54 = m.m22.m21;
        let mut a55 = m.m22.m22;
        let mut a56 = m.m22.m23;
        let mut a61 = m.m21.m31;
        let mut a62 = m.m21.m32;
        let mut a63 = m.m21.m33;
        let mut a64 = m.m22.m31;
        let mut a65 = m.m22.m32;
        let mut a66 = m.m22.m33;
        if self.p.p1 == 2 {
            let t = a11;
            a11 = a21;
            a21 = t;
            let t = a12;
            a12 = a22;
            a22 = t;
            let t = a13;
            a13 = a23;
            a23 = t;
            let t = a14;
            a14 = a24;
            a24 = t;
            let t = a15;
            a15 = a25;
            a25 = t;
            let t = a16;
            a16 = a26;
            a26 = t;
        } else if self.p.p1 == 3 {
            let t = a11;
            a11 = a31;
            a31 = t;
            let t = a12;
            a12 = a32;
            a32 = t;
            let t = a13;
            a13 = a33;
            a33 = t;
            let t = a14;
            a14 = a34;
            a34 = t;
            let t = a15;
            a15 = a35;
            a35 = t;
            let t = a16;
            a16 = a36;
            a36 = t;
        } else if self.p.p1 == 4 {
            let t = a11;
            a11 = a41;
            a41 = t;
            let t = a12;
            a12 = a42;
            a42 = t;
            let t = a13;
            a13 = a43;
            a43 = t;
            let t = a14;
            a14 = a44;
            a44 = t;
            let t = a15;
            a15 = a45;
            a45 = t;
            let t = a16;
            a16 = a46;
            a46 = t;
        } else if self.p.p1 == 5 {
            let t = a11;
            a11 = a51;
            a51 = t;
            let t = a12;
            a12 = a52;
            a52 = t;
            let t = a13;
            a13 = a53;
            a53 = t;
            let t = a14;
            a14 = a54;
            a54 = t;
            let t = a15;
            a15 = a55;
            a55 = t;
            let t = a16;
            a16 = a56;
            a56 = t;
        } else if self.p.p1 == 6 {
            let t = a11;
            a11 = a61;
            a61 = t;
            let t = a12;
            a12 = a62;
            a62 = t;
            let t = a13;
            a13 = a63;
            a63 = t;
            let t = a14;
            a14 = a64;
            a64 = t;
            let t = a15;
            a15 = a65;
            a65 = t;
            let t = a16;
            a16 = a66;
            a66 = t;
        }
        if self.p.p2 == 3 {
            let t = a21;
            a21 = a31;
            a31 = t;
            let t = a22;
            a22 = a32;
            a32 = t;
            let t = a23;
            a23 = a33;
            a33 = t;
            let t = a24;
            a24 = a34;
            a34 = t;
            let t = a25;
            a25 = a35;
            a35 = t;
            let t = a26;
            a26 = a36;
            a36 = t;
        } else if self.p.p2 == 4 {
            let t = a21;
            a21 = a41;
            a41 = t;
            let t = a22;
            a22 = a42;
            a42 = t;
            let t = a23;
            a23 = a43;
            a43 = t;
            let t = a24;
            a24 = a44;
            a44 = t;
            let t = a25;
            a25 = a45;
            a45 = t;
            let t = a26;
            a26 = a46;
            a46 = t;
        } else if self.p.p2 == 5 {
            let t = a21;
            a21 = a51;
            a51 = t;
            let t = a22;
            a22 = a52;
            a52 = t;
            let t = a23;
            a23 = a53;
            a53 = t;
            let t = a24;
            a24 = a54;
            a54 = t;
            let t = a25;
            a25 = a55;
            a55 = t;
            let t = a26;
            a26 = a56;
            a56 = t;
        } else if self.p.p2 == 6 {
            let t = a21;
            a21 = a61;
            a61 = t;
            let t = a22;
            a22 = a62;
            a62 = t;
            let t = a23;
            a23 = a63;
            a63 = t;
            let t = a24;
            a24 = a64;
            a64 = t;
            let t = a25;
            a25 = a65;
            a65 = t;
            let t = a26;
            a26 = a66;
            a66 = t;
        }
        if self.p.p3 == 4 {
            let t = a31;
            a31 = a41;
            a41 = t;
            let t = a32;
            a32 = a42;
            a42 = t;
            let t = a33;
            a33 = a43;
            a43 = t;
            let t = a34;
            a34 = a44;
            a44 = t;
            let t = a35;
            a35 = a45;
            a45 = t;
            let t = a36;
            a36 = a46;
            a46 = t;
        } else if self.p.p3 == 5 {
            let t = a31;
            a31 = a51;
            a51 = t;
            let t = a32;
            a32 = a52;
            a52 = t;
            let t = a33;
            a33 = a53;
            a53 = t;
            let t = a34;
            a34 = a54;
            a54 = t;
            let t = a35;
            a35 = a55;
            a55 = t;
            let t = a36;
            a36 = a56;
            a56 = t;
        } else if self.p.p3 == 6 {
            let t = a31;
            a31 = a61;
            a61 = t;
            let t = a32;
            a32 = a62;
            a62 = t;
            let t = a33;
            a33 = a63;
            a63 = t;
            let t = a34;
            a34 = a64;
            a64 = t;
            let t = a35;
            a35 = a65;
            a65 = t;
            let t = a36;
            a36 = a66;
            a66 = t;
        }
        if self.p.p4 == 5 {
            let t = a41;
            a41 = a51;
            a51 = t;
            let t = a42;
            a42 = a52;
            a52 = t;
            let t = a43;
            a43 = a53;
            a53 = t;
            let t = a44;
            a44 = a54;
            a54 = t;
            let t = a45;
            a45 = a55;
            a55 = t;
            let t = a46;
            a46 = a56;
            a56 = t;
        } else if self.p.p4 == 6 {
            let t = a41;
            a41 = a61;
            a61 = t;
            let t = a42;
            a42 = a62;
            a62 = t;
            let t = a43;
            a43 = a63;
            a63 = t;
            let t = a44;
            a44 = a64;
            a64 = t;
            let t = a45;
            a45 = a65;
            a65 = t;
            let t = a46;
            a46 = a66;
            a66 = t;
        }
        if self.p.p5 == 6 {
            let t = a51;
            a51 = a61;
            a61 = t;
            let t = a52;
            a52 = a62;
            a62 = t;
            let t = a53;
            a53 = a63;
            a63 = t;
            let t = a54;
            a54 = a64;
            a64 = t;
            let t = a55;
            a55 = a65;
            a65 = t;
            let t = a56;
            a56 = a66;
            a66 = t;
        }
        Matrix6 {
            m11: Matrix3 {
                m11: a11,
                m21: a21,
                m31: a31,
                m12: a12,
                m22: a22,
                m32: a32,
                m13: a13,
                m23: a23,
                m33: a33,
            },
            m21: Matrix3 {
                m11: a41,
                m21: a51,
                m31: a61,
                m12: a42,
                m22: a52,
                m32: a62,
                m13: a43,
                m23: a53,
                m33: a63,
            },
            m12: Matrix3 {
                m11: a14,
                m21: a24,
                m31: a34,
                m12: a15,
                m22: a25,
                m32: a35,
                m13: a16,
                m23: a26,
                m33: a36,
            },
            m22: Matrix3 {
                m11: a44,
                m21: a54,
                m31: a64,
                m12: a45,
                m22: a55,
                m32: a65,
                m13: a46,
                m23: a56,
                m33: a66,
            },
        }
    }

    /// Whether the factorisation is invertible: all 6 diagonal entries of `U` are EXACTLY nonzero
    /// (no epsilon), like upstream's `LU::is_invertible`.
    ///
    /// This is NOT the criterion of `Matrix6::try_inverse` (a computed determinant exactly zero),
    /// even though the determinant is the product of these very pivots: a matrix rejected here is
    /// rejected there too, but a matrix with 6 nonzero pivots can still have a determinant that
    /// underflows to zero in the product chain — then `Matrix6::try_inverse` gives `None` while
    /// the factorisation still solves. Neither criterion rejects a merely ill-conditioned matrix:
    /// it is factored and solved with the precision its conditioning allows.
    #[inline(always)]
    fn is_invertible(self: Lu6<T>) -> bool {
        self.lu.m11.m11 != R::ZERO
            && self.lu.m11.m22 != R::ZERO
            && self.lu.m11.m33 != R::ZERO
            && self.lu.m22.m11 != R::ZERO
            && self.lu.m22.m22 != R::ZERO
            && self.lu.m22.m33 != R::ZERO
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a pivot is exactly zero
    /// (`is_invertible`). Upstream: `LU::solve`.
    ///
    /// `b` is permuted (exactly), then `L y = P b` is solved by forward substitution and `U x = y`
    /// by back substitution. Each `y_i` costs ONE rounding — the whole sum of products is
    /// accumulated in `Real::Wide` and rescaled once — and each `x_i` costs TWO: the numerator,
    /// then the correctly rounded division by the pivot.
    ///
    /// The 6 correctly rounded divisions are kept rather than 6 reciprocals and 6 multiplications.
    /// That candidate loses on both counts here: a reciprocal plus a multiplication is dearer than
    /// a division and a single right-hand side amortises nothing, and rounding `1 / u_ii` before
    /// using it costs accuracy when `|u_ii| >> 1`. `bench_lu6_solve__alt_recip` and
    /// `test_solve_candidates_error` keep both measurements. `try_inverse` amortises a reciprocal
    /// over 6 columns and would be cheaper with one, and still does not use one (see there).
    ///
    /// Panics with the scalar's overflow error if a component of `x` does not fit.
    fn solve(self: Lu6<T>, b: Vector6<T>) -> Option<Vector6<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let pb = Self::permute(self, b);
        let y1 = pb.a.x;
        let y2 = R::mul_add(-self.lu.m11.m21, y1, pb.a.y);
        let y3 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.a.z), self.lu.m11.m31, y1),
                self.lu.m11.m32,
                y2,
            ),
        );
        let y4 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.b.x), self.lu.m21.m11, y1),
                    self.lu.m21.m12,
                    y2,
                ),
                self.lu.m21.m13,
                y3,
            ),
        );
        let y5 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.b.y), self.lu.m21.m21, y1),
                        self.lu.m21.m22,
                        y2,
                    ),
                    self.lu.m21.m23,
                    y3,
                ),
                self.lu.m22.m21,
                y4,
            ),
        );
        let y6 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_add(R::wide_zero(), pb.b.z), self.lu.m21.m31, y1,
                            ),
                            self.lu.m21.m32,
                            y2,
                        ),
                        self.lu.m21.m33,
                        y3,
                    ),
                    self.lu.m22.m31,
                    y4,
                ),
                self.lu.m22.m32,
                y5,
            ),
        );
        let x6 = R::div(y6, self.lu.m22.m33);
        let x5 = R::div(R::mul_add(-self.lu.m22.m23, x6, y5), self.lu.m22.m22);
        let x4 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y4), self.lu.m22.m12, x5),
                    self.lu.m22.m13,
                    x6,
                ),
            ),
            self.lu.m22.m11,
        );
        let x3 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), y3), self.lu.m12.m31, x4),
                        self.lu.m12.m32,
                        x5,
                    ),
                    self.lu.m12.m33,
                    x6,
                ),
            ),
            self.lu.m11.m33,
        );
        let x2 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_add(R::wide_zero(), y2), self.lu.m11.m23, x3),
                            self.lu.m12.m21,
                            x4,
                        ),
                        self.lu.m12.m22,
                        x5,
                    ),
                    self.lu.m12.m23,
                    x6,
                ),
            ),
            self.lu.m11.m22,
        );
        let x1 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(
                                    R::wide_add(R::wide_zero(), y1), self.lu.m11.m12, x2,
                                ),
                                self.lu.m11.m13,
                                x3,
                            ),
                            self.lu.m12.m11,
                            x4,
                        ),
                        self.lu.m12.m12,
                        x5,
                    ),
                    self.lu.m12.m13,
                    x6,
                ),
            ),
            self.lu.m11.m11,
        );
        Some(Vector6 { a: Vector3 { x: x1, y: x2, z: x3 }, b: Vector3 { x: x4, y: x5, z: x6 } })
    }

    /// The inverse, or `None` when a pivot is exactly zero (`is_invertible`). Upstream:
    /// `LU::try_inverse`.
    ///
    /// Computed as `A^-1 = (L U)^-1 P`, NOT as 6 calls to `solve`: the right-hand sides of `L U M =
    /// I` are then the STATIC unit vectors, whose leading zeros disappear at generation time (the
    /// forward substitution is 35 products instead of 90), and the permutation is applied at the
    /// end by swapping the COLUMNS of `M` in reverse factorisation order — moves only, exact.
    ///
    /// Rounding: one per entry of the forward substitution, two per entry of the back substitution
    /// (the numerator, then the correctly rounded division by the pivot). Panics with the scalar's
    /// overflow error if an entry of the inverse does not fit.
    ///
    /// Unlike `solve`, this one WOULD be cheaper with one reciprocal per pivot, which 6 columns
    /// amortise: 213 300 against 233 420 gas. It still divides, because `mul(x, recip(u))` rounds
    /// twice where `x / u` rounds once, which is the rule DESIGN D2 and `Vector6::unscale` already
    /// follow; the drift is small but real (3 ulp on the oracle inverses).
    /// `bench_lu6_try_inverse__alt_recip` and `test_try_inverse_candidates` keep the measurement.
    fn try_inverse(self: Lu6<T>) -> Option<Matrix6<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y21 = -self.lu.m11.m21;
        let y31 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m11.m31), self.lu.m11.m32, y21),
        );
        let y41 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub(R::wide_zero(), self.lu.m21.m11), self.lu.m21.m12, y21,
                ),
                self.lu.m21.m13,
                y31,
            ),
        );
        let y51 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub(R::wide_zero(), self.lu.m21.m21), self.lu.m21.m22, y21,
                    ),
                    self.lu.m21.m23,
                    y31,
                ),
                self.lu.m22.m21,
                y41,
            ),
        );
        let y61 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub(R::wide_zero(), self.lu.m21.m31), self.lu.m21.m32, y21,
                        ),
                        self.lu.m21.m33,
                        y31,
                    ),
                    self.lu.m22.m31,
                    y41,
                ),
                self.lu.m22.m32,
                y51,
            ),
        );
        let x61 = R::div(y61, self.lu.m22.m33);
        let x51 = R::div(R::mul_add(-self.lu.m22.m23, x61, y51), self.lu.m22.m22);
        let x41 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y41), self.lu.m22.m12, x51),
                    self.lu.m22.m13,
                    x61,
                ),
            ),
            self.lu.m22.m11,
        );
        let x31 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), y31), self.lu.m12.m31, x41),
                        self.lu.m12.m32,
                        x51,
                    ),
                    self.lu.m12.m33,
                    x61,
                ),
            ),
            self.lu.m11.m33,
        );
        let x21 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_add(R::wide_zero(), y21), self.lu.m11.m23, x31,
                            ),
                            self.lu.m12.m21,
                            x41,
                        ),
                        self.lu.m12.m22,
                        x51,
                    ),
                    self.lu.m12.m23,
                    x61,
                ),
            ),
            self.lu.m11.m22,
        );
        let x11 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(
                                    R::wide_add(R::wide_zero(), R::ONE), self.lu.m11.m12, x21,
                                ),
                                self.lu.m11.m13,
                                x31,
                            ),
                            self.lu.m12.m11,
                            x41,
                        ),
                        self.lu.m12.m12,
                        x51,
                    ),
                    self.lu.m12.m13,
                    x61,
                ),
            ),
            self.lu.m11.m11,
        );
        let y32 = -self.lu.m11.m32;
        let y42 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m21.m12), self.lu.m21.m13, y32),
        );
        let y52 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub(R::wide_zero(), self.lu.m21.m22), self.lu.m21.m23, y32,
                ),
                self.lu.m22.m21,
                y42,
            ),
        );
        let y62 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub(R::wide_zero(), self.lu.m21.m32), self.lu.m21.m33, y32,
                    ),
                    self.lu.m22.m31,
                    y42,
                ),
                self.lu.m22.m32,
                y52,
            ),
        );
        let x62 = R::div(y62, self.lu.m22.m33);
        let x52 = R::div(R::mul_add(-self.lu.m22.m23, x62, y52), self.lu.m22.m22);
        let x42 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y42), self.lu.m22.m12, x52),
                    self.lu.m22.m13,
                    x62,
                ),
            ),
            self.lu.m22.m11,
        );
        let x32 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), y32), self.lu.m12.m31, x42),
                        self.lu.m12.m32,
                        x52,
                    ),
                    self.lu.m12.m33,
                    x62,
                ),
            ),
            self.lu.m11.m33,
        );
        let x22 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_add(R::wide_zero(), R::ONE), self.lu.m11.m23, x32,
                            ),
                            self.lu.m12.m21,
                            x42,
                        ),
                        self.lu.m12.m22,
                        x52,
                    ),
                    self.lu.m12.m23,
                    x62,
                ),
            ),
            self.lu.m11.m22,
        );
        let x12 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(R::wide_zero(), self.lu.m11.m12, x22),
                                self.lu.m11.m13,
                                x32,
                            ),
                            self.lu.m12.m11,
                            x42,
                        ),
                        self.lu.m12.m12,
                        x52,
                    ),
                    self.lu.m12.m13,
                    x62,
                ),
            ),
            self.lu.m11.m11,
        );
        let y43 = -self.lu.m21.m13;
        let y53 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m21.m23), self.lu.m22.m21, y43),
        );
        let y63 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub(R::wide_zero(), self.lu.m21.m33), self.lu.m22.m31, y43,
                ),
                self.lu.m22.m32,
                y53,
            ),
        );
        let x63 = R::div(y63, self.lu.m22.m33);
        let x53 = R::div(R::mul_add(-self.lu.m22.m23, x63, y53), self.lu.m22.m22);
        let x43 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y43), self.lu.m22.m12, x53),
                    self.lu.m22.m13,
                    x63,
                ),
            ),
            self.lu.m22.m11,
        );
        let x33 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), R::ONE), self.lu.m12.m31, x43),
                        self.lu.m12.m32,
                        x53,
                    ),
                    self.lu.m12.m33,
                    x63,
                ),
            ),
            self.lu.m11.m33,
        );
        let x23 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m11.m23, x33),
                            self.lu.m12.m21,
                            x43,
                        ),
                        self.lu.m12.m22,
                        x53,
                    ),
                    self.lu.m12.m23,
                    x63,
                ),
            ),
            self.lu.m11.m22,
        );
        let x13 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(R::wide_zero(), self.lu.m11.m12, x23),
                                self.lu.m11.m13,
                                x33,
                            ),
                            self.lu.m12.m11,
                            x43,
                        ),
                        self.lu.m12.m12,
                        x53,
                    ),
                    self.lu.m12.m13,
                    x63,
                ),
            ),
            self.lu.m11.m11,
        );
        let y54 = -self.lu.m22.m21;
        let y64 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m22.m31), self.lu.m22.m32, y54),
        );
        let x64 = R::div(y64, self.lu.m22.m33);
        let x54 = R::div(R::mul_add(-self.lu.m22.m23, x64, y54), self.lu.m22.m22);
        let x44 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), R::ONE), self.lu.m22.m12, x54),
                    self.lu.m22.m13,
                    x64,
                ),
            ),
            self.lu.m22.m11,
        );
        let x34 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_zero(), self.lu.m12.m31, x44),
                        self.lu.m12.m32,
                        x54,
                    ),
                    self.lu.m12.m33,
                    x64,
                ),
            ),
            self.lu.m11.m33,
        );
        let x24 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m11.m23, x34),
                            self.lu.m12.m21,
                            x44,
                        ),
                        self.lu.m12.m22,
                        x54,
                    ),
                    self.lu.m12.m23,
                    x64,
                ),
            ),
            self.lu.m11.m22,
        );
        let x14 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(R::wide_zero(), self.lu.m11.m12, x24),
                                self.lu.m11.m13,
                                x34,
                            ),
                            self.lu.m12.m11,
                            x44,
                        ),
                        self.lu.m12.m12,
                        x54,
                    ),
                    self.lu.m12.m13,
                    x64,
                ),
            ),
            self.lu.m11.m11,
        );
        let y65 = -self.lu.m22.m32;
        let x65 = R::div(y65, self.lu.m22.m33);
        let x55 = R::div(R::mul_add(-self.lu.m22.m23, x65, R::ONE), self.lu.m22.m22);
        let x45 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_zero(), self.lu.m22.m12, x55), self.lu.m22.m13, x65,
                ),
            ),
            self.lu.m22.m11,
        );
        let x35 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_zero(), self.lu.m12.m31, x45),
                        self.lu.m12.m32,
                        x55,
                    ),
                    self.lu.m12.m33,
                    x65,
                ),
            ),
            self.lu.m11.m33,
        );
        let x25 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m11.m23, x35),
                            self.lu.m12.m21,
                            x45,
                        ),
                        self.lu.m12.m22,
                        x55,
                    ),
                    self.lu.m12.m23,
                    x65,
                ),
            ),
            self.lu.m11.m22,
        );
        let x15 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(R::wide_zero(), self.lu.m11.m12, x25),
                                self.lu.m11.m13,
                                x35,
                            ),
                            self.lu.m12.m11,
                            x45,
                        ),
                        self.lu.m12.m12,
                        x55,
                    ),
                    self.lu.m12.m13,
                    x65,
                ),
            ),
            self.lu.m11.m11,
        );
        let x66 = R::div(R::ONE, self.lu.m22.m33);
        let x56 = R::div(
            R::wide_rescale(R::wide_sub_prod(R::wide_zero(), self.lu.m22.m23, x66)),
            self.lu.m22.m22,
        );
        let x46 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_zero(), self.lu.m22.m12, x56), self.lu.m22.m13, x66,
                ),
            ),
            self.lu.m22.m11,
        );
        let x36 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_zero(), self.lu.m12.m31, x46),
                        self.lu.m12.m32,
                        x56,
                    ),
                    self.lu.m12.m33,
                    x66,
                ),
            ),
            self.lu.m11.m33,
        );
        let x26 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m11.m23, x36),
                            self.lu.m12.m21,
                            x46,
                        ),
                        self.lu.m12.m22,
                        x56,
                    ),
                    self.lu.m12.m23,
                    x66,
                ),
            ),
            self.lu.m11.m22,
        );
        let x16 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(R::wide_zero(), self.lu.m11.m12, x26),
                                self.lu.m11.m13,
                                x36,
                            ),
                            self.lu.m12.m11,
                            x46,
                        ),
                        self.lu.m12.m12,
                        x56,
                    ),
                    self.lu.m12.m13,
                    x66,
                ),
            ),
            self.lu.m11.m11,
        );
        let mut c11 = x11;
        let mut c12 = x12;
        let mut c13 = x13;
        let mut c14 = x14;
        let mut c15 = x15;
        let mut c16 = x16;
        let mut c21 = x21;
        let mut c22 = x22;
        let mut c23 = x23;
        let mut c24 = x24;
        let mut c25 = x25;
        let mut c26 = x26;
        let mut c31 = x31;
        let mut c32 = x32;
        let mut c33 = x33;
        let mut c34 = x34;
        let mut c35 = x35;
        let mut c36 = x36;
        let mut c41 = x41;
        let mut c42 = x42;
        let mut c43 = x43;
        let mut c44 = x44;
        let mut c45 = x45;
        let mut c46 = x46;
        let mut c51 = x51;
        let mut c52 = x52;
        let mut c53 = x53;
        let mut c54 = x54;
        let mut c55 = x55;
        let mut c56 = x56;
        let mut c61 = x61;
        let mut c62 = x62;
        let mut c63 = x63;
        let mut c64 = x64;
        let mut c65 = x65;
        let mut c66 = x66;
        if self.p.p5 == 6 {
            let t = c15;
            c15 = c16;
            c16 = t;
            let t = c25;
            c25 = c26;
            c26 = t;
            let t = c35;
            c35 = c36;
            c36 = t;
            let t = c45;
            c45 = c46;
            c46 = t;
            let t = c55;
            c55 = c56;
            c56 = t;
            let t = c65;
            c65 = c66;
            c66 = t;
        }
        if self.p.p4 == 5 {
            let t = c14;
            c14 = c15;
            c15 = t;
            let t = c24;
            c24 = c25;
            c25 = t;
            let t = c34;
            c34 = c35;
            c35 = t;
            let t = c44;
            c44 = c45;
            c45 = t;
            let t = c54;
            c54 = c55;
            c55 = t;
            let t = c64;
            c64 = c65;
            c65 = t;
        } else if self.p.p4 == 6 {
            let t = c14;
            c14 = c16;
            c16 = t;
            let t = c24;
            c24 = c26;
            c26 = t;
            let t = c34;
            c34 = c36;
            c36 = t;
            let t = c44;
            c44 = c46;
            c46 = t;
            let t = c54;
            c54 = c56;
            c56 = t;
            let t = c64;
            c64 = c66;
            c66 = t;
        }
        if self.p.p3 == 4 {
            let t = c13;
            c13 = c14;
            c14 = t;
            let t = c23;
            c23 = c24;
            c24 = t;
            let t = c33;
            c33 = c34;
            c34 = t;
            let t = c43;
            c43 = c44;
            c44 = t;
            let t = c53;
            c53 = c54;
            c54 = t;
            let t = c63;
            c63 = c64;
            c64 = t;
        } else if self.p.p3 == 5 {
            let t = c13;
            c13 = c15;
            c15 = t;
            let t = c23;
            c23 = c25;
            c25 = t;
            let t = c33;
            c33 = c35;
            c35 = t;
            let t = c43;
            c43 = c45;
            c45 = t;
            let t = c53;
            c53 = c55;
            c55 = t;
            let t = c63;
            c63 = c65;
            c65 = t;
        } else if self.p.p3 == 6 {
            let t = c13;
            c13 = c16;
            c16 = t;
            let t = c23;
            c23 = c26;
            c26 = t;
            let t = c33;
            c33 = c36;
            c36 = t;
            let t = c43;
            c43 = c46;
            c46 = t;
            let t = c53;
            c53 = c56;
            c56 = t;
            let t = c63;
            c63 = c66;
            c66 = t;
        }
        if self.p.p2 == 3 {
            let t = c12;
            c12 = c13;
            c13 = t;
            let t = c22;
            c22 = c23;
            c23 = t;
            let t = c32;
            c32 = c33;
            c33 = t;
            let t = c42;
            c42 = c43;
            c43 = t;
            let t = c52;
            c52 = c53;
            c53 = t;
            let t = c62;
            c62 = c63;
            c63 = t;
        } else if self.p.p2 == 4 {
            let t = c12;
            c12 = c14;
            c14 = t;
            let t = c22;
            c22 = c24;
            c24 = t;
            let t = c32;
            c32 = c34;
            c34 = t;
            let t = c42;
            c42 = c44;
            c44 = t;
            let t = c52;
            c52 = c54;
            c54 = t;
            let t = c62;
            c62 = c64;
            c64 = t;
        } else if self.p.p2 == 5 {
            let t = c12;
            c12 = c15;
            c15 = t;
            let t = c22;
            c22 = c25;
            c25 = t;
            let t = c32;
            c32 = c35;
            c35 = t;
            let t = c42;
            c42 = c45;
            c45 = t;
            let t = c52;
            c52 = c55;
            c55 = t;
            let t = c62;
            c62 = c65;
            c65 = t;
        } else if self.p.p2 == 6 {
            let t = c12;
            c12 = c16;
            c16 = t;
            let t = c22;
            c22 = c26;
            c26 = t;
            let t = c32;
            c32 = c36;
            c36 = t;
            let t = c42;
            c42 = c46;
            c46 = t;
            let t = c52;
            c52 = c56;
            c56 = t;
            let t = c62;
            c62 = c66;
            c66 = t;
        }
        if self.p.p1 == 2 {
            let t = c11;
            c11 = c12;
            c12 = t;
            let t = c21;
            c21 = c22;
            c22 = t;
            let t = c31;
            c31 = c32;
            c32 = t;
            let t = c41;
            c41 = c42;
            c42 = t;
            let t = c51;
            c51 = c52;
            c52 = t;
            let t = c61;
            c61 = c62;
            c62 = t;
        } else if self.p.p1 == 3 {
            let t = c11;
            c11 = c13;
            c13 = t;
            let t = c21;
            c21 = c23;
            c23 = t;
            let t = c31;
            c31 = c33;
            c33 = t;
            let t = c41;
            c41 = c43;
            c43 = t;
            let t = c51;
            c51 = c53;
            c53 = t;
            let t = c61;
            c61 = c63;
            c63 = t;
        } else if self.p.p1 == 4 {
            let t = c11;
            c11 = c14;
            c14 = t;
            let t = c21;
            c21 = c24;
            c24 = t;
            let t = c31;
            c31 = c34;
            c34 = t;
            let t = c41;
            c41 = c44;
            c44 = t;
            let t = c51;
            c51 = c54;
            c54 = t;
            let t = c61;
            c61 = c64;
            c64 = t;
        } else if self.p.p1 == 5 {
            let t = c11;
            c11 = c15;
            c15 = t;
            let t = c21;
            c21 = c25;
            c25 = t;
            let t = c31;
            c31 = c35;
            c35 = t;
            let t = c41;
            c41 = c45;
            c45 = t;
            let t = c51;
            c51 = c55;
            c55 = t;
            let t = c61;
            c61 = c65;
            c65 = t;
        } else if self.p.p1 == 6 {
            let t = c11;
            c11 = c16;
            c16 = t;
            let t = c21;
            c21 = c26;
            c26 = t;
            let t = c31;
            c31 = c36;
            c36 = t;
            let t = c41;
            c41 = c46;
            c46 = t;
            let t = c51;
            c51 = c56;
            c56 = t;
            let t = c61;
            c61 = c66;
            c66 = t;
        }
        Some(
            Matrix6 {
                m11: Matrix3 {
                    m11: c11,
                    m21: c21,
                    m31: c31,
                    m12: c12,
                    m22: c22,
                    m32: c32,
                    m13: c13,
                    m23: c23,
                    m33: c33,
                },
                m21: Matrix3 {
                    m11: c41,
                    m21: c51,
                    m31: c61,
                    m12: c42,
                    m22: c52,
                    m32: c62,
                    m13: c43,
                    m23: c53,
                    m33: c63,
                },
                m12: Matrix3 {
                    m11: c14,
                    m21: c24,
                    m31: c34,
                    m12: c15,
                    m22: c25,
                    m32: c35,
                    m13: c16,
                    m23: c26,
                    m33: c36,
                },
                m22: Matrix3 {
                    m11: c44,
                    m21: c54,
                    m31: c64,
                    m12: c45,
                    m22: c55,
                    m32: c65,
                    m13: c46,
                    m23: c56,
                    m33: c66,
                },
            },
        )
    }

    /// The determinant: the product of the 6 pivots, with the sign of the permutation. Upstream:
    /// `LU::determinant`.
    ///
    /// The sign is applied to the FIRST pivot — an exact negation — and the product is then a
    /// left-to-right chain of 5 floored multiplications, so the result stays a floor chain instead
    /// of the negation of one (`-floor(x)` is `ceil(-x)`, one ulp off). `Real` exposes no `Wide *
    /// T`, so a product of 6 scalars cannot be accumulated exactly: each intermediate is floored
    /// once, which adds at most 5 ulp to the error already carried by the pivots.
    ///
    /// Exactly zero for a singular matrix. Panics with the scalar's overflow error if an
    /// intermediate product does not fit; partial pivoting makes the 6 pivots comparable in
    /// magnitude, so the partial products grow monotonically toward the determinant and an
    /// intermediate overflow implies the determinant itself does not fit.
    fn determinant(self: Lu6<T>) -> T {
        let mut neg = false;
        if self.p.p1 != 1 {
            neg = !neg;
        }
        if self.p.p2 != 2 {
            neg = !neg;
        }
        if self.p.p3 != 3 {
            neg = !neg;
        }
        if self.p.p4 != 4 {
            neg = !neg;
        }
        if self.p.p5 != 5 {
            neg = !neg;
        }
        let d = if neg {
            -self.lu.m11.m11
        } else {
            self.lu.m11.m11
        };
        let d = d * self.lu.m11.m22;
        let d = d * self.lu.m11.m33;
        let d = d * self.lu.m22.m11;
        let d = d * self.lu.m22.m22;
        d * self.lu.m22.m33
    }
}

/// `Matrix6` methods that go through the LU factorisation; upstream carries them on the matrix
/// itself. Import `Matrix6LuTrait` to use them.
#[generate_trait]
pub impl Matrix6LuImpl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Matrix6LuTrait<T> {
    /// The LU factorisation with partial pivoting. Upstream: `Matrix6::lu`.
    #[inline(always)]
    fn lu(self: Matrix6<T>) -> Lu6<T> {
        Lu6Trait::new(self)
    }

    /// The determinant, through the LU factorisation: there is no closed form worth writing for a
    /// 6x6 (the cofactor expansion is 720 signed products of 6 factors, and `Real` cannot
    /// accumulate them exactly). Upstream: `Matrix6::determinant`, which also routes through `LU`
    /// above dimension 3.
    ///
    /// Convenience wrapper: keep the `Lu6` when several of `determinant`, `try_inverse` and `solve`
    /// are needed on the same matrix, as each call here re-factors it.
    fn determinant(self: Matrix6<T>) -> T {
        Lu6Trait::determinant(Lu6Trait::new(self))
    }

    /// The inverse, or `None` when a pivot of the factorisation is exactly zero. Upstream:
    /// `Matrix6::try_inverse`. Re-factors the matrix on every call.
    fn try_inverse(self: Matrix6<T>) -> Option<Matrix6<T>> {
        Lu6Trait::try_inverse(Lu6Trait::new(self))
    }

    /// The solution of `self * x = b`, or `None` when a pivot is exactly zero. Upstream:
    /// `matrix.lu().solve(&b)`. Re-factors the matrix on every call.
    fn solve(self: Matrix6<T>, b: Vector6<T>) -> Option<Vector6<T>> {
        Lu6Trait::solve(Lu6Trait::new(self), b)
    }
}
