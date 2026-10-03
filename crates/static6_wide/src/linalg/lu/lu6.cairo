//! Internal, no stability promise: the crate-private items of `linalg::lu::lu6` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_core::linalg::permutation_sequence::PermuteRows;
use nalgebra_types6::base::matrix6::Matrix6;
use nalgebra_types6::base::vector6::Vector6;
use simba::scalar::Real;
use crate::internal::linalg::lu::lu6::Lu6InternalTrait;
use crate::linalg::lu::Perm6;

/// The LU factorisation with partial pivoting of a `Matrix6<T>`: `P * A = L * U`.
///
/// `lu` packs both factors (strict lower triangle = `L` without its unit diagonal, upper triangle =
/// `U`) and `p` is the row permutation. Built by `Lu6Trait::new` or `Matrix6LuTrait::lu`. Upstream:
/// `nalgebra::linalg::LU`.
#[derive(Copy, Drop, Serde, Debug)]
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
    ///
    /// The row swap of step 1 is a `match` on the pivot row (constant cost, WP 11-OPT-2); the later
    /// steps keep the chain of comparisons, cheaper when the pivot is in one of the first rows (a
    /// `match` cost the probe's input 15 more steps there).
    #[inline(always)]
    fn new(matrix: Matrix6<T>) -> Lu6<T> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a13 = matrix.m13;
        let mut a14 = matrix.m14;
        let mut a15 = matrix.m15;
        let mut a16 = matrix.m16;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        let mut a23 = matrix.m23;
        let mut a24 = matrix.m24;
        let mut a25 = matrix.m25;
        let mut a26 = matrix.m26;
        let mut a31 = matrix.m31;
        let mut a32 = matrix.m32;
        let mut a33 = matrix.m33;
        let mut a34 = matrix.m34;
        let mut a35 = matrix.m35;
        let mut a36 = matrix.m36;
        let mut a41 = matrix.m41;
        let mut a42 = matrix.m42;
        let mut a43 = matrix.m43;
        let mut a44 = matrix.m44;
        let mut a45 = matrix.m45;
        let mut a46 = matrix.m46;
        let mut a51 = matrix.m51;
        let mut a52 = matrix.m52;
        let mut a53 = matrix.m53;
        let mut a54 = matrix.m54;
        let mut a55 = matrix.m55;
        let mut a56 = matrix.m56;
        let mut a61 = matrix.m61;
        let mut a62 = matrix.m62;
        let mut a63 = matrix.m63;
        let mut a64 = matrix.m64;
        let mut a65 = matrix.m65;
        let mut a66 = matrix.m66;
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
        match p1 {
            0 => {},
            1 => {},
            2 => {
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
            },
            3 => {
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
            },
            4 => {
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
            },
            5 => {
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
            },
            6 => {
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
            },
            _ => {},
        }
        if piv != R::zero() {
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
        if piv != R::zero() {
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
        if piv != R::zero() {
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
        if piv != R::zero() {
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
        if piv != R::zero() {
            let l = R::div(a65, a55);
            let nl = -l;
            a66 = R::mul_add(nl, a56, a66);
            a65 = l;
        }
        Lu6 {
            lu: Matrix6 {
                m11: a11,
                m21: a21,
                m31: a31,
                m12: a12,
                m22: a22,
                m32: a32,
                m13: a13,
                m23: a23,
                m33: a33,
                m41: a41,
                m51: a51,
                m61: a61,
                m42: a42,
                m52: a52,
                m62: a62,
                m43: a43,
                m53: a53,
                m63: a63,
                m14: a14,
                m24: a24,
                m34: a34,
                m15: a15,
                m25: a25,
                m35: a35,
                m16: a16,
                m26: a26,
                m36: a36,
                m44: a44,
                m54: a54,
                m64: a64,
                m45: a45,
                m55: a55,
                m65: a65,
                m46: a46,
                m56: a56,
                m66: a66,
            },
            p: Perm6 { p1, p2, p3, p4, p5 },
        }
    }

    /// The unit lower triangular factor `L` (its diagonal of ones is implicit in the packed
    /// storage). Exact: moves only. Upstream: `LU::l`.
    #[inline(always)]
    fn l(self: Lu6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: R::one(),
            m21: self.lu.m21,
            m31: self.lu.m31,
            m12: R::zero(),
            m22: R::one(),
            m32: self.lu.m32,
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m41: self.lu.m41,
            m51: self.lu.m51,
            m61: self.lu.m61,
            m42: self.lu.m42,
            m52: self.lu.m52,
            m62: self.lu.m62,
            m43: self.lu.m43,
            m53: self.lu.m53,
            m63: self.lu.m63,
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m15: R::zero(),
            m25: R::zero(),
            m35: R::zero(),
            m16: R::zero(),
            m26: R::zero(),
            m36: R::zero(),
            m44: R::one(),
            m54: self.lu.m54,
            m64: self.lu.m64,
            m45: R::zero(),
            m55: R::one(),
            m65: self.lu.m65,
            m46: R::zero(),
            m56: R::zero(),
            m66: R::one(),
        }
    }

    /// The upper triangular factor `U`. Exact: moves only. Upstream: `LU::u`.
    #[inline(always)]
    fn u(self: Lu6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: self.lu.m11,
            m21: R::zero(),
            m31: R::zero(),
            m12: self.lu.m12,
            m22: self.lu.m22,
            m32: R::zero(),
            m13: self.lu.m13,
            m23: self.lu.m23,
            m33: self.lu.m33,
            m41: R::zero(),
            m51: R::zero(),
            m61: R::zero(),
            m42: R::zero(),
            m52: R::zero(),
            m62: R::zero(),
            m43: R::zero(),
            m53: R::zero(),
            m63: R::zero(),
            m14: self.lu.m14,
            m24: self.lu.m24,
            m34: self.lu.m34,
            m15: self.lu.m15,
            m25: self.lu.m25,
            m35: self.lu.m35,
            m16: self.lu.m16,
            m26: self.lu.m26,
            m36: self.lu.m36,
            m44: self.lu.m44,
            m54: R::zero(),
            m64: R::zero(),
            m45: self.lu.m45,
            m55: self.lu.m55,
            m65: R::zero(),
            m46: self.lu.m46,
            m56: self.lu.m56,
            m66: self.lu.m66,
        }
    }

    /// The packed factors as the factorisation stores them: `L` (strict lower triangle, unit
    /// diagonal implicit) and `U` (upper triangle) in one matrix. Exact: a move. Upstream:
    /// `LU::lu_internal` (`#[doc(hidden)]`).
    #[inline(always)]
    fn lu_internal(self: Lu6<T>) -> Matrix6<T> {
        self.lu
    }

    /// The unit lower triangular factor `L`, consuming the factorisation: `l()` (exact, moves
    /// only). Upstream: `LU::l_unpack`.
    #[inline(always)]
    fn l_unpack(self: Lu6<T>) -> Matrix6<T> {
        Self::l(self)
    }

    /// The three factors `(P, L, U)` of `P * A = L * U`: `(p(), l(), u())`, exact. Upstream:
    /// `LU::unpack`.
    #[inline(always)]
    fn unpack(self: Lu6<T>) -> (Perm6, Matrix6<T>, Matrix6<T>) {
        (self.p, Self::l(self), Self::u(self))
    }

    /// Overwrites `b` (any shape with 6 rows: a vector or a matrix) with the solution `x` of `A *
    /// x = b` and returns `true`, or returns `false` and leaves `b` unchanged when a pivot is
    /// exactly zero (`is_invertible`; upstream returns `false` after overwriting `b` with an
    /// unspecified partial result). Upstream: `LU::solve_mut`.
    ///
    /// Upstream's steps, on every column of `b` at once: `P b` (moves), `L y = P b` by forward
    /// substitution on the implicit unit diagonal (one floor per component, no division: `L`'s
    /// quotients by 1 are exact, bit-identical to upstream's `solve_lower_triangular_with_diag_mut
    /// (b, 1)`), then `U x = y` by back substitution (one floor and one correctly rounded division
    /// per component, one prepared divisor per row from 3 columns on). On a vector it is
    /// bit-identical to `solve`. Panics on overflow.
    fn solve_mut<B, impl P: PermuteRows<Perm6, B>, impl K: SolveKernel<Matrix6<T>, B>, +Drop<B>>(
        self: Lu6<T>, ref b: B,
    ) -> bool {
        if !Self::is_invertible(self) {
            return false;
        }
        P::permute_rows(self.p, ref b);
        b = K::upper(self.lu, K::lower_unit(self.lu, b));
        true
    }

    /// Overwrites `out` with the inverse and returns `true`, or returns `false` and leaves `out`
    /// unchanged when a pivot is exactly zero (upstream fills `out` with the identity first and
    /// leaves a partial result). The inverse is `try_inverse`'s, bit for bit (upstream's
    /// `try_inverse` is `try_inverse_to` on the identity). Upstream: `LU::try_inverse_to`.
    #[inline(always)]
    fn try_inverse_to(self: Lu6<T>, ref out: Matrix6<T>) -> bool {
        match Self::try_inverse(self) {
            Option::Some(m) => {
                out = m;
                true
            },
            Option::None => false,
        }
    }

    /// The row permutation `P`, as the compact sequence of 5 transpositions `Perm6` — never as a
    /// matrix, and never as an array (DESIGN D4). Upstream: `LU::p`, which returns a heap-allocated
    /// `PermutationSequence`.
    #[inline(always)]
    fn p(self: Lu6<T>) -> Perm6 {
        self.p
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
        self.lu.m11 != R::zero()
            && self.lu.m22 != R::zero()
            && self.lu.m33 != R::zero()
            && self.lu.m44 != R::zero()
            && self.lu.m55 != R::zero()
            && self.lu.m66 != R::zero()
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
    #[inline(always)]
    fn solve(self: Lu6<T>, b: Vector6<T>) -> Option<Vector6<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let pb = Lu6InternalTrait::permute(self, b);
        let y1 = pb.x;
        let y2 = R::mul_add(-self.lu.m21, y1, pb.y);
        let y3 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.z), self.lu.m31, y1),
                self.lu.m32,
                y2,
            ),
        );
        let y4 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.w), self.lu.m41, y1),
                    self.lu.m42,
                    y2,
                ),
                self.lu.m43,
                y3,
            ),
        );
        let y5 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.a), self.lu.m51, y1),
                        self.lu.m52,
                        y2,
                    ),
                    self.lu.m53,
                    y3,
                ),
                self.lu.m54,
                y4,
            ),
        );
        let y6 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.b), self.lu.m61, y1),
                            self.lu.m62,
                            y2,
                        ),
                        self.lu.m63,
                        y3,
                    ),
                    self.lu.m64,
                    y4,
                ),
                self.lu.m65,
                y5,
            ),
        );
        let x6 = R::div(y6, self.lu.m66);
        let x5 = R::div(R::mul_add(-self.lu.m56, x6, y5), self.lu.m55);
        let x4 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y4), self.lu.m45, x5),
                    self.lu.m46,
                    x6,
                ),
            ),
            self.lu.m44,
        );
        let x3 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), y3), self.lu.m34, x4),
                        self.lu.m35,
                        x5,
                    ),
                    self.lu.m36,
                    x6,
                ),
            ),
            self.lu.m33,
        );
        let x2 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_add(R::wide_zero(), y2), self.lu.m23, x3),
                            self.lu.m24,
                            x4,
                        ),
                        self.lu.m25,
                        x5,
                    ),
                    self.lu.m26,
                    x6,
                ),
            ),
            self.lu.m22,
        );
        let x1 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(R::wide_add(R::wide_zero(), y1), self.lu.m12, x2),
                                self.lu.m13,
                                x3,
                            ),
                            self.lu.m14,
                            x4,
                        ),
                        self.lu.m15,
                        x5,
                    ),
                    self.lu.m16,
                    x6,
                ),
            ),
            self.lu.m11,
        );
        Some(Vector6 { x: x1, y: x2, z: x3, w: x4, a: x5, b: x6 })
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
    /// amortise: 208 240 against 271 680 gas (net, `fixed` 0.3.0). It still divides — upstream's
    /// `solve_mut` divides by the pivot — because `mul(x, recip(u))` rounds twice where `x / u`
    /// rounds once, which is the rule DESIGN D2 and `Vector6::unscale` already follow; the drift is
    /// small but real (17 ulp on the oracle inverses).
    /// `bench_lu6_try_inverse__alt_recip` and `test_try_inverse_candidates` keep the measurement.
    ///
    /// The back substitution runs ROW by row across the 6 columns, so the quotients that
    /// share a pivot go through ONE prepared divisor (`Real::div5` / `Real::div6`, bit-identical
    /// to per-element division, cheaper from 3 quotients) and the corner `1 / u_66` is
    /// `Real::recip` (WP 7.2).
    fn try_inverse(self: Lu6<T>) -> Option<Matrix6<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y21 = -self.lu.m21;
        let y31 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m31), self.lu.m32, y21),
        );
        let y41 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m41), self.lu.m42, y21),
                self.lu.m43,
                y31,
            ),
        );
        let y51 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m51), self.lu.m52, y21),
                    self.lu.m53,
                    y31,
                ),
                self.lu.m54,
                y41,
            ),
        );
        let y61 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub(R::wide_zero(), self.lu.m61), self.lu.m62, y21,
                        ),
                        self.lu.m63,
                        y31,
                    ),
                    self.lu.m64,
                    y41,
                ),
                self.lu.m65,
                y51,
            ),
        );
        let y32 = -self.lu.m32;
        let y42 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m42), self.lu.m43, y32),
        );
        let y52 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m52), self.lu.m53, y32),
                self.lu.m54,
                y42,
            ),
        );
        let y62 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m62), self.lu.m63, y32),
                    self.lu.m64,
                    y42,
                ),
                self.lu.m65,
                y52,
            ),
        );
        let y43 = -self.lu.m43;
        let y53 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m53), self.lu.m54, y43),
        );
        let y63 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m63), self.lu.m64, y43),
                self.lu.m65,
                y53,
            ),
        );
        let y54 = -self.lu.m54;
        let y64 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m64), self.lu.m65, y54),
        );
        let y65 = -self.lu.m65;
        let x66 = R::recip(self.lu.m66);
        let (x61, x62, x63, x64, x65) = R::div5(y61, y62, y63, y64, y65, self.lu.m66);
        let n51 = R::mul_add(-self.lu.m56, x61, y51);
        let n52 = R::mul_add(-self.lu.m56, x62, y52);
        let n53 = R::mul_add(-self.lu.m56, x63, y53);
        let n54 = R::mul_add(-self.lu.m56, x64, y54);
        let n55 = R::mul_add(-self.lu.m56, x65, R::one());
        let n56 = R::wide_rescale(R::wide_sub_prod(R::wide_zero(), self.lu.m56, x66));
        let (x51, x52, x53, x54, x55, x56) = R::div6(n51, n52, n53, n54, n55, n56, self.lu.m55);
        let n41 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), y41), self.lu.m45, x51),
                self.lu.m46,
                x61,
            ),
        );
        let n42 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), y42), self.lu.m45, x52),
                self.lu.m46,
                x62,
            ),
        );
        let n43 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), y43), self.lu.m45, x53),
                self.lu.m46,
                x63,
            ),
        );
        let n44 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), R::one()), self.lu.m45, x54),
                self.lu.m46,
                x64,
            ),
        );
        let n45 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), self.lu.m45, x55), self.lu.m46, x65),
        );
        let n46 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), self.lu.m45, x56), self.lu.m46, x66),
        );
        let (x41, x42, x43, x44, x45, x46) = R::div6(n41, n42, n43, n44, n45, n46, self.lu.m44);
        let n31 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y31), self.lu.m34, x41),
                    self.lu.m35,
                    x51,
                ),
                self.lu.m36,
                x61,
            ),
        );
        let n32 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y32), self.lu.m34, x42),
                    self.lu.m35,
                    x52,
                ),
                self.lu.m36,
                x62,
            ),
        );
        let n33 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), R::one()), self.lu.m34, x43),
                    self.lu.m35,
                    x53,
                ),
                self.lu.m36,
                x63,
            ),
        );
        let n34 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_zero(), self.lu.m34, x44), self.lu.m35, x54,
                ),
                self.lu.m36,
                x64,
            ),
        );
        let n35 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_zero(), self.lu.m34, x45), self.lu.m35, x55,
                ),
                self.lu.m36,
                x65,
            ),
        );
        let n36 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_zero(), self.lu.m34, x46), self.lu.m35, x56,
                ),
                self.lu.m36,
                x66,
            ),
        );
        let (x31, x32, x33, x34, x35, x36) = R::div6(n31, n32, n33, n34, n35, n36, self.lu.m33);
        let n21 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), y21), self.lu.m23, x31),
                        self.lu.m24,
                        x41,
                    ),
                    self.lu.m25,
                    x51,
                ),
                self.lu.m26,
                x61,
            ),
        );
        let n22 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), R::one()), self.lu.m23, x32),
                        self.lu.m24,
                        x42,
                    ),
                    self.lu.m25,
                    x52,
                ),
                self.lu.m26,
                x62,
            ),
        );
        let n23 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_zero(), self.lu.m23, x33), self.lu.m24, x43,
                    ),
                    self.lu.m25,
                    x53,
                ),
                self.lu.m26,
                x63,
            ),
        );
        let n24 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_zero(), self.lu.m23, x34), self.lu.m24, x44,
                    ),
                    self.lu.m25,
                    x54,
                ),
                self.lu.m26,
                x64,
            ),
        );
        let n25 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_zero(), self.lu.m23, x35), self.lu.m24, x45,
                    ),
                    self.lu.m25,
                    x55,
                ),
                self.lu.m26,
                x65,
            ),
        );
        let n26 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_zero(), self.lu.m23, x36), self.lu.m24, x46,
                    ),
                    self.lu.m25,
                    x56,
                ),
                self.lu.m26,
                x66,
            ),
        );
        let (x21, x22, x23, x24, x25, x26) = R::div6(n21, n22, n23, n24, n25, n26, self.lu.m22);
        let n11 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_add(R::wide_zero(), R::one()), self.lu.m12, x21,
                            ),
                            self.lu.m13,
                            x31,
                        ),
                        self.lu.m14,
                        x41,
                    ),
                    self.lu.m15,
                    x51,
                ),
                self.lu.m16,
                x61,
            ),
        );
        let n12 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m12, x22), self.lu.m13, x32,
                        ),
                        self.lu.m14,
                        x42,
                    ),
                    self.lu.m15,
                    x52,
                ),
                self.lu.m16,
                x62,
            ),
        );
        let n13 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m12, x23), self.lu.m13, x33,
                        ),
                        self.lu.m14,
                        x43,
                    ),
                    self.lu.m15,
                    x53,
                ),
                self.lu.m16,
                x63,
            ),
        );
        let n14 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m12, x24), self.lu.m13, x34,
                        ),
                        self.lu.m14,
                        x44,
                    ),
                    self.lu.m15,
                    x54,
                ),
                self.lu.m16,
                x64,
            ),
        );
        let n15 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m12, x25), self.lu.m13, x35,
                        ),
                        self.lu.m14,
                        x45,
                    ),
                    self.lu.m15,
                    x55,
                ),
                self.lu.m16,
                x65,
            ),
        );
        let n16 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_zero(), self.lu.m12, x26), self.lu.m13, x36,
                        ),
                        self.lu.m14,
                        x46,
                    ),
                    self.lu.m15,
                    x56,
                ),
                self.lu.m16,
                x66,
            ),
        );
        let (x11, x12, x13, x14, x15, x16) = R::div6(n11, n12, n13, n14, n15, n16, self.lu.m11);
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
                m11: c11,
                m21: c21,
                m31: c31,
                m12: c12,
                m22: c22,
                m32: c32,
                m13: c13,
                m23: c23,
                m33: c33,
                m41: c41,
                m51: c51,
                m61: c61,
                m42: c42,
                m52: c52,
                m62: c62,
                m43: c43,
                m53: c53,
                m63: c63,
                m14: c14,
                m24: c24,
                m34: c34,
                m15: c15,
                m25: c25,
                m35: c35,
                m16: c16,
                m26: c26,
                m36: c36,
                m44: c44,
                m54: c54,
                m64: c64,
                m45: c45,
                m55: c55,
                m65: c65,
                m46: c46,
                m56: c56,
                m66: c66,
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
            -self.lu.m11
        } else {
            self.lu.m11
        };
        let d = d * self.lu.m22;
        let d = d * self.lu.m33;
        let d = d * self.lu.m44;
        let d = d * self.lu.m55;
        d * self.lu.m66
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
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_types6::base::matrix6::Matrix6;
    use nalgebra_types6::base::vector6::Vector6;
    use simba::scalar::FixedReal as R;
    use crate::internal::linalg::lu::lu6::Lu6InternalTrait;
    use crate::linalg::lu::Perm6;
    use super::{Lu6, Lu6Trait};

    /// `Lu6Trait::new` before WP 11-OPT-2: the row swap of step 1 as a chain of `if p1 == k`
    /// (the new body matches on `p1`), called, not inlined.
    fn new_reference(matrix: Matrix6<Fixed>) -> Lu6<Fixed> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a13 = matrix.m13;
        let mut a14 = matrix.m14;
        let mut a15 = matrix.m15;
        let mut a16 = matrix.m16;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        let mut a23 = matrix.m23;
        let mut a24 = matrix.m24;
        let mut a25 = matrix.m25;
        let mut a26 = matrix.m26;
        let mut a31 = matrix.m31;
        let mut a32 = matrix.m32;
        let mut a33 = matrix.m33;
        let mut a34 = matrix.m34;
        let mut a35 = matrix.m35;
        let mut a36 = matrix.m36;
        let mut a41 = matrix.m41;
        let mut a42 = matrix.m42;
        let mut a43 = matrix.m43;
        let mut a44 = matrix.m44;
        let mut a45 = matrix.m45;
        let mut a46 = matrix.m46;
        let mut a51 = matrix.m51;
        let mut a52 = matrix.m52;
        let mut a53 = matrix.m53;
        let mut a54 = matrix.m54;
        let mut a55 = matrix.m55;
        let mut a56 = matrix.m56;
        let mut a61 = matrix.m61;
        let mut a62 = matrix.m62;
        let mut a63 = matrix.m63;
        let mut a64 = matrix.m64;
        let mut a65 = matrix.m65;
        let mut a66 = matrix.m66;
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
        if piv != R::zero() {
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
        if piv != R::zero() {
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
        if piv != R::zero() {
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
        if piv != R::zero() {
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
        if piv != R::zero() {
            let l = R::div(a65, a55);
            let nl = -l;
            a66 = R::mul_add(nl, a56, a66);
            a65 = l;
        }
        Lu6 {
            lu: Matrix6 {
                m11: a11,
                m21: a21,
                m31: a31,
                m12: a12,
                m22: a22,
                m32: a32,
                m13: a13,
                m23: a23,
                m33: a33,
                m41: a41,
                m51: a51,
                m61: a61,
                m42: a42,
                m52: a52,
                m62: a62,
                m43: a43,
                m53: a53,
                m63: a63,
                m14: a14,
                m24: a24,
                m34: a34,
                m15: a15,
                m25: a25,
                m35: a35,
                m16: a16,
                m26: a26,
                m36: a36,
                m44: a44,
                m54: a54,
                m64: a64,
                m45: a45,
                m55: a55,
                m65: a65,
                m46: a46,
                m56: a56,
                m66: a66,
            },
            p: Perm6 { p1, p2, p3, p4, p5 },
        }
    }

    /// `Lu6Trait::solve` before WP 11-OPT-2 (the same body, called, not inlined).
    fn solve_reference(self: Lu6<Fixed>, b: Vector6<Fixed>) -> Option<Vector6<Fixed>> {
        if !Lu6Trait::is_invertible(self) {
            return None;
        }
        let pb = Lu6InternalTrait::permute(self, b);
        let y1 = pb.x;
        let y2 = R::mul_add(-self.lu.m21, y1, pb.y);
        let y3 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.z), self.lu.m31, y1),
                self.lu.m32,
                y2,
            ),
        );
        let y4 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.w), self.lu.m41, y1),
                    self.lu.m42,
                    y2,
                ),
                self.lu.m43,
                y3,
            ),
        );
        let y5 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.a), self.lu.m51, y1),
                        self.lu.m52,
                        y2,
                    ),
                    self.lu.m53,
                    y3,
                ),
                self.lu.m54,
                y4,
            ),
        );
        let y6 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.b), self.lu.m61, y1),
                            self.lu.m62,
                            y2,
                        ),
                        self.lu.m63,
                        y3,
                    ),
                    self.lu.m64,
                    y4,
                ),
                self.lu.m65,
                y5,
            ),
        );
        let x6 = R::div(y6, self.lu.m66);
        let x5 = R::div(R::mul_add(-self.lu.m56, x6, y5), self.lu.m55);
        let x4 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y4), self.lu.m45, x5),
                    self.lu.m46,
                    x6,
                ),
            ),
            self.lu.m44,
        );
        let x3 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), y3), self.lu.m34, x4),
                        self.lu.m35,
                        x5,
                    ),
                    self.lu.m36,
                    x6,
                ),
            ),
            self.lu.m33,
        );
        let x2 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(R::wide_add(R::wide_zero(), y2), self.lu.m23, x3),
                            self.lu.m24,
                            x4,
                        ),
                        self.lu.m25,
                        x5,
                    ),
                    self.lu.m26,
                    x6,
                ),
            ),
            self.lu.m22,
        );
        let x1 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(
                            R::wide_sub_prod(
                                R::wide_sub_prod(R::wide_add(R::wide_zero(), y1), self.lu.m12, x2),
                                self.lu.m13,
                                x3,
                            ),
                            self.lu.m14,
                            x4,
                        ),
                        self.lu.m15,
                        x5,
                    ),
                    self.lu.m16,
                    x6,
                ),
            ),
            self.lu.m11,
        );
        Some(Vector6 { x: x1, y: x2, z: x3, w: x4, a: x5, b: x6 })
    }

    fn fx(raw: i64) -> Fixed {
        Fixed { raw }
    }

    /// Deterministic 64-bit LCG (Knuth's MMIX constants).
    fn next(ref state: u128) -> u128 {
        state = (state * 6364136223846793005 + 1442695040888963407) % 0x10000000000000000;
        state
    }

    /// A raw value uniform in `[-bound, bound]`.
    fn draw(ref state: u128, bound: u128) -> i64 {
        let r: i128 = (next(ref state) % (2 * bound + 1)).try_into().unwrap();
        let b: i128 = bound.try_into().unwrap();
        (r - b).try_into().unwrap()
    }

    /// A raw value uniform in `[lo, lo + span]`.
    fn draw_pos(ref state: u128, lo: u128, span: u128) -> i64 {
        (lo + next(ref state) % (span + 1)).try_into().unwrap()
    }

    /// A pivot of a hand-assembled factor: of magnitude `[1/2, 4]`, of the sign of the parity of
    /// `k`, and exactly zero on one draw in 16 (`solve` then returns `None`).
    fn pivot(ref state: u128, k: u32) -> i64 {
        let m = draw_pos(ref state, 0x80000000, 0x380000000);
        if next(ref state) % 16 == 0 {
            0
        } else if k % 2 == 0 {
            m
        } else {
            -m
        }
    }

    /// A row index uniform in `[lo, 3 or 6]`.
    fn row(ref state: u128, lo: u8, hi: u8) -> u8 {
        let span: u128 = (hi - lo + 1).into();
        lo + (next(ref state) % span).try_into().unwrap()
    }

    fn same(a: Lu6<Fixed>, b: Lu6<Fixed>) -> bool {
        a.lu == b.lu
            && a.p.p1 == b.p.p1
            && a.p.p2 == b.p.p2
            && a.p.p3 == b.p.p3
            && a.p.p4 == b.p.p4
            && a.p.p5 == b.p.p5
    }

    /// `m` with the first-column entry of row `r` (1-based) set to 7, above every other entry of
    /// that column (all below 1 in the probe's matrix): its step-1 pivot is row `r`, to reach every
    /// arm of the row swap.
    fn pivot_row(m: Matrix6<Fixed>, r: u32) -> Matrix6<Fixed> {
        let big = Fixed { raw: 0x700000000 };
        match r {
            1 => Matrix6 { m11: big, ..m },
            2 => Matrix6 { m21: big, ..m },
            3 => Matrix6 { m31: big, ..m },
            4 => Matrix6 { m41: big, ..m },
            5 => Matrix6 { m51: big, ..m },
            _ => Matrix6 { m61: big, ..m },
        }
    }

    fn diag(d: i64) -> Matrix6<Fixed> {
        let z = fx(0);
        let d = fx(d);
        Matrix6 {
            m11: d,
            m12: z,
            m13: z,
            m14: z,
            m15: z,
            m16: z,
            m21: z,
            m22: d,
            m23: z,
            m24: z,
            m25: z,
            m26: z,
            m31: z,
            m32: z,
            m33: d,
            m34: z,
            m35: z,
            m36: z,
            m41: z,
            m42: z,
            m43: z,
            m44: d,
            m45: z,
            m46: z,
            m51: z,
            m52: z,
            m53: z,
            m54: z,
            m55: d,
            m56: z,
            m61: z,
            m62: z,
            m63: z,
            m64: z,
            m65: z,
            m66: d,
        }
    }

    /// `new` against the reference, bit for bit: edge cases (zero, identity, a 1-ulp diagonal, the
    /// probe's matrix, a zero first column, the step-1 pivot forced to each of the 6 rows) and a
    /// deterministic sweep of matrices with entries from 2^-24 to 2^20 (every step-1 pivot row is
    /// reached: the test checks it).
    #[test]
    fn test_new_matches_reference() {
        let one: i64 = 0x100000000;
        let a = Matrix6 {
            m11: fx(-2527254097),
            m12: fx(4325213708),
            m13: fx(-1213189640),
            m14: fx(4008510819),
            m15: fx(2405075077),
            m16: fx(466428599),
            m21: fx(-700750028),
            m22: fx(-3536112306),
            m23: fx(1731465811),
            m24: fx(244560054),
            m25: fx(2632174365),
            m26: fx(1874080084),
            m31: fx(-4134312449),
            m32: fx(-1036601848),
            m33: fx(-444052936),
            m34: fx(-1797900266),
            m35: fx(-2730917964),
            m36: fx(404854629),
            m41: fx(-617580144),
            m42: fx(580405440),
            m43: fx(-3475076872),
            m44: fx(-924247759),
            m45: fx(-985644408),
            m46: fx(938584824),
            m51: fx(-1076219818),
            m52: fx(-1905997652),
            m53: fx(-1595683778),
            m54: fx(2067806730),
            m55: fx(-3245944919),
            m56: fx(3717734456),
            m61: fx(-308280913),
            m62: fx(-120179493),
            m63: fx(2005545225),
            m64: fx(-1090342151),
            m65: fx(932840990),
            m66: fx(2332476429),
        };
        let z = fx(0);
        let mut cases: Array<Matrix6<Fixed>> = array![
            diag(0), diag(one), diag(1), a,
            Matrix6 { m11: z, m21: z, m31: z, m41: z, m51: z, m61: z, ..a },
        ];
        for r in 1..7_u32 {
            cases.append(pivot_row(a, r));
        }
        let mut state: u128 = 0x1066;
        for k in 0..240_u32 {
            let ab: u128 = match k % 4 {
                0 => 0x100,
                1 => 0x100000000,
                2 => 0x10000000000,
                _ => 0x10000000000000,
            };
            cases
                .append(
                    Matrix6 {
                        m11: fx(draw(ref state, ab)),
                        m12: fx(draw(ref state, ab)),
                        m13: fx(draw(ref state, ab)),
                        m14: fx(draw(ref state, ab)),
                        m15: fx(draw(ref state, ab)),
                        m16: fx(draw(ref state, ab)),
                        m21: fx(draw(ref state, ab)),
                        m22: fx(draw(ref state, ab)),
                        m23: fx(draw(ref state, ab)),
                        m24: fx(draw(ref state, ab)),
                        m25: fx(draw(ref state, ab)),
                        m26: fx(draw(ref state, ab)),
                        m31: fx(draw(ref state, ab)),
                        m32: fx(draw(ref state, ab)),
                        m33: fx(draw(ref state, ab)),
                        m34: fx(draw(ref state, ab)),
                        m35: fx(draw(ref state, ab)),
                        m36: fx(draw(ref state, ab)),
                        m41: fx(draw(ref state, ab)),
                        m42: fx(draw(ref state, ab)),
                        m43: fx(draw(ref state, ab)),
                        m44: fx(draw(ref state, ab)),
                        m45: fx(draw(ref state, ab)),
                        m46: fx(draw(ref state, ab)),
                        m51: fx(draw(ref state, ab)),
                        m52: fx(draw(ref state, ab)),
                        m53: fx(draw(ref state, ab)),
                        m54: fx(draw(ref state, ab)),
                        m55: fx(draw(ref state, ab)),
                        m56: fx(draw(ref state, ab)),
                        m61: fx(draw(ref state, ab)),
                        m62: fx(draw(ref state, ab)),
                        m63: fx(draw(ref state, ab)),
                        m64: fx(draw(ref state, ab)),
                        m65: fx(draw(ref state, ab)),
                        m66: fx(draw(ref state, ab)),
                    },
                );
        }
        let mut n = 0_u32;
        let mut seen = 0_u32;
        for m in cases.span() {
            let got = Lu6Trait::new(*m);
            assert!(same(got, new_reference(*m)));
            let bit: u32 = match got.p.p1 {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 4,
                4 => 8,
                5 => 16,
                _ => 32,
            };
            seen = seen | bit;
            n += 1;
        }
        assert!(n >= 200);
        assert!(seen == 63);
    }

    /// `solve` against the reference, bit for bit: edge cases (the factors of the identity and of
    /// the probe's matrix; zero, 1-ulp and large right-hand sides) and a deterministic sweep of
    /// hand-assembled factors (every permutation, pivots of either sign, zero on one draw in 16,
    /// entries from 2^-24 to 1) with right-hand sides from 2^-32 to 2^8.
    #[test]
    fn test_solve_matches_reference() {
        let one: i64 = 0x100000000;
        let mut cases: Array<(Lu6<Fixed>, Vector6<Fixed>)> = array![];
        let a = Matrix6 {
            m11: fx(-2527254097),
            m12: fx(4325213708),
            m13: fx(-1213189640),
            m14: fx(4008510819),
            m15: fx(2405075077),
            m16: fx(466428599),
            m21: fx(-700750028),
            m22: fx(-3536112306),
            m23: fx(1731465811),
            m24: fx(244560054),
            m25: fx(2632174365),
            m26: fx(1874080084),
            m31: fx(-4134312449),
            m32: fx(-1036601848),
            m33: fx(-444052936),
            m34: fx(-1797900266),
            m35: fx(-2730917964),
            m36: fx(404854629),
            m41: fx(-617580144),
            m42: fx(580405440),
            m43: fx(-3475076872),
            m44: fx(-924247759),
            m45: fx(-985644408),
            m46: fx(938584824),
            m51: fx(-1076219818),
            m52: fx(-1905997652),
            m53: fx(-1595683778),
            m54: fx(2067806730),
            m55: fx(-3245944919),
            m56: fx(3717734456),
            m61: fx(-308280913),
            m62: fx(-120179493),
            m63: fx(2005545225),
            m64: fx(-1090342151),
            m65: fx(932840990),
            m66: fx(2332476429),
        };
        let rhs = array![
            Vector6 { x: fx(0), y: fx(0), z: fx(0), w: fx(0), a: fx(0), b: fx(0) },
            Vector6 { x: fx(1), y: fx(-1), z: fx(1), w: fx(-1), a: fx(1), b: fx(-1) },
            Vector6 {
                x: fx(3579353502),
                y: fx(7767965432),
                z: fx(6840630971),
                w: fx(-6086016248),
                a: fx(-4735434050),
                b: fx(-2809324573),
            },
            Vector6 {
                x: fx(0x10000000000),
                y: fx(-0x10000000000),
                z: fx(one),
                w: fx(0),
                a: fx(-one),
                b: fx(0x10000000000),
            },
        ];
        for b in rhs.span() {
            cases.append((new_reference(diag(one)), *b));
            cases.append((new_reference(a), *b));
        }
        let mut state: u128 = 0x5066;
        for k in 0..240_u32 {
            let u: u128 = match k % 3 {
                0 => 0x100,
                1 => 0x40000000,
                _ => 0x100000000,
            };
            let lu = Matrix6 {
                m11: fx(pivot(ref state, k)),
                m12: fx(draw(ref state, u)),
                m13: fx(draw(ref state, u)),
                m14: fx(draw(ref state, u)),
                m15: fx(draw(ref state, u)),
                m16: fx(draw(ref state, u)),
                m21: fx(draw(ref state, u)),
                m22: fx(pivot(ref state, k)),
                m23: fx(draw(ref state, u)),
                m24: fx(draw(ref state, u)),
                m25: fx(draw(ref state, u)),
                m26: fx(draw(ref state, u)),
                m31: fx(draw(ref state, u)),
                m32: fx(draw(ref state, u)),
                m33: fx(pivot(ref state, k)),
                m34: fx(draw(ref state, u)),
                m35: fx(draw(ref state, u)),
                m36: fx(draw(ref state, u)),
                m41: fx(draw(ref state, u)),
                m42: fx(draw(ref state, u)),
                m43: fx(draw(ref state, u)),
                m44: fx(pivot(ref state, k)),
                m45: fx(draw(ref state, u)),
                m46: fx(draw(ref state, u)),
                m51: fx(draw(ref state, u)),
                m52: fx(draw(ref state, u)),
                m53: fx(draw(ref state, u)),
                m54: fx(draw(ref state, u)),
                m55: fx(pivot(ref state, k)),
                m56: fx(draw(ref state, u)),
                m61: fx(draw(ref state, u)),
                m62: fx(draw(ref state, u)),
                m63: fx(draw(ref state, u)),
                m64: fx(draw(ref state, u)),
                m65: fx(draw(ref state, u)),
                m66: fx(pivot(ref state, k)),
            };
            let p = Perm6 {
                p1: row(ref state, 1, 6),
                p2: row(ref state, 2, 6),
                p3: row(ref state, 3, 6),
                p4: row(ref state, 4, 6),
                p5: row(ref state, 5, 6),
            };
            let bb: u128 = match k % 4 {
                0 => 0x100,
                1 => 0x100000000,
                2 => 0x10000000000,
                _ => 0x10000000000,
            };
            let v = Vector6 {
                x: fx(draw(ref state, bb)),
                y: fx(draw(ref state, bb)),
                z: fx(draw(ref state, bb)),
                w: fx(draw(ref state, bb)),
                a: fx(draw(ref state, bb)),
                b: fx(draw(ref state, bb)),
            };
            cases.append((Lu6 { lu, p }, v));
        }
        let mut n = 0_u32;
        for c in cases.span() {
            let (f, v) = *c;
            assert!(f.solve(v) == solve_reference(f, v));
            n += 1;
        }
        assert!(n >= 200);
    }
}
