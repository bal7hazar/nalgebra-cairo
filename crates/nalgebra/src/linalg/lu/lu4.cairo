//! `Lu4`: the LU factorisation with partial pivoting of a `Matrix4` (upstream
//! `nalgebra::linalg::LU` on a 4x4 matrix).
//!
//! `P * A = L * U`, with `L` unit lower triangular, `U` upper triangular and `P` the product of the
//! 3 row transpositions chosen by partial pivoting. Both factors share one `Matrix4` like upstream
//! (strict lower triangle = `L`, whose unit diagonal is implicit; upper triangle = `U`) and the
//! permutation is the compact `Perm4`.
//!
//! Everything is unrolled (DESIGN D4: no loop in static code) and every sum of products goes
//! through a fused `Real` kernel — `mul_add` for a single product, the explicit `Real::Wide`
//! accumulator beyond that — so each output scalar is floored once and range-checked once
//! (AGENTS.md rule 4).
//!
//! Partial pivoting costs 6 `abs` and comparisons plus 3 conditional row swaps (moves only), and
//! every swap duplicates the row it moves, so it also costs Sierra statements. Dropping it would be
//! cheaper and shorter and it is not an option: without it the pivot of step `k` is whatever sits
//! at `a_kk`, nothing bounds `|l_ik|`, and a matrix as ordinary as a permuted identity factors with
//! a zero pivot. `bench_lu4_new__alt_no_pivot` and `test_no_pivot_candidate_is_wrong` keep the
//! measurement and the counter-example. Upstream has no unpivoted variant either, only `LU`
//! (partial pivoting) and `FullPivLU` (complete pivoting).

use simba::scalar::Real;
use crate::base::matrix4::Matrix4;
use crate::base::vector4::Vector4;
use super::Perm4;

/// The LU factorisation with partial pivoting of a `Matrix4<T>`: `P * A = L * U`.
///
/// `lu` packs both factors (strict lower triangle = `L` without its unit diagonal, upper triangle =
/// `U`) and `p` is the row permutation. Built by `Lu4Trait::new` or `Matrix4LuTrait::lu`. Upstream:
/// `nalgebra::linalg::LU`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Lu4<T> {
    /// `L` (strict lower triangle) and `U` (upper triangle) packed in one matrix.
    pub lu: Matrix4<T>,
    /// The row transpositions applied by partial pivoting.
    pub p: Perm4,
}

/// Test-only field-wise equality (upstream `Lu4` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Lu4PartialEq<T, +PartialEq<T>> of PartialEq<Lu4<T>> {
    fn eq(lhs: @Lu4<T>, rhs: @Lu4<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

/// Methods of `Lu4<T>` for any `Real` scalar.
#[generate_trait]
pub impl Lu4Impl<
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
> of Lu4Trait<T> {
    /// The LU factorisation of `matrix` with partial pivoting, fully unrolled. Upstream:
    /// `matrix.lu()` / `LU::new(matrix)`.
    ///
    /// Always succeeds, like upstream: a singular matrix simply leaves a zero on the diagonal of
    /// `U` (see `is_invertible`). At step `k`, the row of largest `|a_ik|` among rows `k..4` is
    /// swapped onto the diagonal (the FIRST such row, like upstream's `icamax`), the 6 multipliers
    /// `l_ik = a_ik / a_kk` are each one correctly rounded division, and the trailing submatrix is
    /// updated entry by entry with `Real::mul_add(-l_ik, a_kj, a_ij)`: ONE floor rounding and one
    /// overflow check per entry, never the two roundings of `a_ij - l_ik * a_kj`.
    ///
    /// A pivot column that is exactly zero is skipped — no swap, no permutation, zero multipliers
    /// —
    /// exactly like upstream's `continue`, so the division is never reached with a zero divisor.
    ///
    /// Error model: `l_ik` is off by at most 1 ulp (correctly rounded quotient) and every update
    /// floors once, so after each of the 3 steps an entry of `U` is within about `k * (1 + |a_kj|)`
    /// raw units of its exact value. Partial pivoting keeps `|l_ik| <= 1`, which is what bounds the
    /// growth of the trailing submatrix. Panics with the scalar's overflow error if an update does
    /// not fit.
    fn new(matrix: Matrix4<T>) -> Lu4<T> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a13 = matrix.m13;
        let mut a14 = matrix.m14;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        let mut a23 = matrix.m23;
        let mut a24 = matrix.m24;
        let mut a31 = matrix.m31;
        let mut a32 = matrix.m32;
        let mut a33 = matrix.m33;
        let mut a34 = matrix.m34;
        let mut a41 = matrix.m41;
        let mut a42 = matrix.m42;
        let mut a43 = matrix.m43;
        let mut a44 = matrix.m44;
        // step 1: largest pivot among rows 1..4 of column 1
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
        }
        if piv != R::zero() {
            let (l_a21, l_a31, l_a41) = R::div3(a21, a31, a41, a11);
            let l = l_a21;
            let nl = -l;
            a22 = R::mul_add(nl, a12, a22);
            a23 = R::mul_add(nl, a13, a23);
            a24 = R::mul_add(nl, a14, a24);
            a21 = l;
            let l = l_a31;
            let nl = -l;
            a32 = R::mul_add(nl, a12, a32);
            a33 = R::mul_add(nl, a13, a33);
            a34 = R::mul_add(nl, a14, a34);
            a31 = l;
            let l = l_a41;
            let nl = -l;
            a42 = R::mul_add(nl, a12, a42);
            a43 = R::mul_add(nl, a13, a43);
            a44 = R::mul_add(nl, a14, a44);
            a41 = l;
        }
        // step 2: largest pivot among rows 2..4 of column 2
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
        }
        if piv != R::zero() {
            let l = R::div(a32, a22);
            let nl = -l;
            a33 = R::mul_add(nl, a23, a33);
            a34 = R::mul_add(nl, a24, a34);
            a32 = l;
            let l = R::div(a42, a22);
            let nl = -l;
            a43 = R::mul_add(nl, a23, a43);
            a44 = R::mul_add(nl, a24, a44);
            a42 = l;
        }
        // step 3: largest pivot among rows 3..4 of column 3
        let mut p3 = 3_u8;
        let mut piv = R::abs(a33);
        let c = R::abs(a43);
        if c > piv {
            piv = c;
            p3 = 4;
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
        }
        if piv != R::zero() {
            let l = R::div(a43, a33);
            let nl = -l;
            a44 = R::mul_add(nl, a34, a44);
            a43 = l;
        }
        Lu4 {
            lu: Matrix4 {
                m11: a11,
                m21: a21,
                m31: a31,
                m41: a41,
                m12: a12,
                m22: a22,
                m32: a32,
                m42: a42,
                m13: a13,
                m23: a23,
                m33: a33,
                m43: a43,
                m14: a14,
                m24: a24,
                m34: a34,
                m44: a44,
            },
            p: Perm4 { p1, p2, p3 },
        }
    }

    /// The unit lower triangular factor `L` (its diagonal of ones is implicit in the packed
    /// storage). Exact: moves only. Upstream: `LU::l`.
    #[inline(always)]
    fn l(self: Lu4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::one(),
            m21: self.lu.m21,
            m31: self.lu.m31,
            m41: self.lu.m41,
            m12: R::zero(),
            m22: R::one(),
            m32: self.lu.m32,
            m42: self.lu.m42,
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m43: self.lu.m43,
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
        }
    }

    /// The upper triangular factor `U`. Exact: moves only. Upstream: `LU::u`.
    #[inline(always)]
    fn u(self: Lu4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: self.lu.m11,
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m12: self.lu.m12,
            m22: self.lu.m22,
            m32: R::zero(),
            m42: R::zero(),
            m13: self.lu.m13,
            m23: self.lu.m23,
            m33: self.lu.m33,
            m43: R::zero(),
            m14: self.lu.m14,
            m24: self.lu.m24,
            m34: self.lu.m34,
            m44: self.lu.m44,
        }
    }

    /// The row permutation `P`, as the compact sequence of 3 transpositions `Perm4` — never as a
    /// matrix, and never as an array (DESIGN D4). Upstream: `LU::p`, which returns a heap-allocated
    /// `PermutationSequence`.
    #[inline(always)]
    fn p(self: Lu4<T>) -> Perm4 {
        self.p
    }

    /// Whether the factorisation is invertible: all 4 diagonal entries of `U` are EXACTLY nonzero
    /// (no epsilon), like upstream's `LU::is_invertible`.
    ///
    /// This is NOT the criterion of `Matrix4::try_inverse` (a computed determinant exactly zero),
    /// even though the determinant is the product of these very pivots: a matrix rejected here is
    /// rejected there too, but a matrix with 4 nonzero pivots can still have a determinant that
    /// underflows to zero in the product chain — then `Matrix4::try_inverse` gives `None` while
    /// the factorisation still solves. Neither criterion rejects a merely ill-conditioned matrix:
    /// it is factored and solved with the precision its conditioning allows.
    #[inline(always)]
    fn is_invertible(self: Lu4<T>) -> bool {
        self.lu.m11 != R::zero()
            && self.lu.m22 != R::zero()
            && self.lu.m33 != R::zero()
            && self.lu.m44 != R::zero()
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a pivot is exactly zero
    /// (`is_invertible`). Upstream: `LU::solve`.
    ///
    /// `b` is permuted (exactly), then `L y = P b` is solved by forward substitution and `U x = y`
    /// by back substitution. Each `y_i` costs ONE rounding — the whole sum of products is
    /// accumulated in `Real::Wide` and rescaled once — and each `x_i` costs TWO: the numerator,
    /// then the correctly rounded division by the pivot.
    ///
    /// The 4 correctly rounded divisions are kept rather than 4 reciprocals and 4 multiplications.
    /// That candidate loses on both counts here: a reciprocal plus a multiplication is dearer than
    /// a division and a single right-hand side amortises nothing, and rounding `1 / u_ii` before
    /// using it costs accuracy when `|u_ii| >> 1`. `bench_lu4_solve__alt_recip` and
    /// `test_solve_candidates_error` keep both measurements. `try_inverse` amortises a reciprocal
    /// over 4 columns and would be cheaper with one, and still does not use one (see there).
    ///
    /// Panics with the scalar's overflow error if a component of `x` does not fit.
    fn solve(self: Lu4<T>, b: Vector4<T>) -> Option<Vector4<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let pb = Lu4InternalTrait::permute(self, b);
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
        let x4 = R::div(y4, self.lu.m44);
        let x3 = R::div(R::mul_add(-self.lu.m34, x4, y3), self.lu.m33);
        let x2 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y2), self.lu.m23, x3),
                    self.lu.m24,
                    x4,
                ),
            ),
            self.lu.m22,
        );
        let x1 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(
                        R::wide_sub_prod(R::wide_add(R::wide_zero(), y1), self.lu.m12, x2),
                        self.lu.m13,
                        x3,
                    ),
                    self.lu.m14,
                    x4,
                ),
            ),
            self.lu.m11,
        );
        Some(Vector4 { x: x1, y: x2, z: x3, w: x4 })
    }

    /// The inverse, or `None` when a pivot is exactly zero (`is_invertible`). Upstream:
    /// `LU::try_inverse`.
    ///
    /// Computed as `A^-1 = (L U)^-1 P`, NOT as 4 calls to `solve`: the right-hand sides of `L U M =
    /// I` are then the STATIC unit vectors, whose leading zeros disappear at generation time (the
    /// forward substitution is 10 products instead of 24), and the permutation is applied at the
    /// end by swapping the COLUMNS of `M` in reverse factorisation order — moves only, exact.
    ///
    /// Rounding: one per entry of the forward substitution, two per entry of the back substitution
    /// (the numerator, then the correctly rounded division by the pivot). Panics with the scalar's
    /// overflow error if an entry of the inverse does not fit.
    ///
    /// Unlike `solve`, this one WOULD be cheaper with one reciprocal per pivot, which 4 columns
    /// amortise: 84 220 against 108 840 gas (net, `fixed` 0.3.0). It still divides — upstream's
    /// `solve_mut` divides by the pivot — because `mul(x, recip(u))` rounds twice where `x / u`
    /// rounds once, which is the rule DESIGN D2 and `Vector4::unscale` already follow; the drift is
    /// small but real (7 ulp on the oracle inverses).
    /// `bench_lu4_try_inverse__alt_recip` and `test_try_inverse_candidates` keep the measurement.
    ///
    /// The back substitution runs ROW by row across the 4 columns, so the quotients that
    /// share a pivot go through ONE prepared divisor (`Real::div3` / `Real::div4`, bit-identical
    /// to per-element division, cheaper from 3 quotients) and the corner `1 / u_44` is
    /// `Real::recip` (WP 7.2).
    fn try_inverse(self: Lu4<T>) -> Option<Matrix4<T>> {
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
        let y32 = -self.lu.m32;
        let y42 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m42), self.lu.m43, y32),
        );
        let y43 = -self.lu.m43;
        let x44 = R::recip(self.lu.m44);
        let (x41, x42, x43) = R::div3(y41, y42, y43, self.lu.m44);
        let n31 = R::mul_add(-self.lu.m34, x41, y31);
        let n32 = R::mul_add(-self.lu.m34, x42, y32);
        let n33 = R::mul_add(-self.lu.m34, x43, R::one());
        let n34 = R::wide_rescale(R::wide_sub_prod(R::wide_zero(), self.lu.m34, x44));
        let (x31, x32, x33, x34) = R::div4(n31, n32, n33, n34, self.lu.m33);
        let n21 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), y21), self.lu.m23, x31),
                self.lu.m24,
                x41,
            ),
        );
        let n22 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), R::one()), self.lu.m23, x32),
                self.lu.m24,
                x42,
            ),
        );
        let n23 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), self.lu.m23, x33), self.lu.m24, x43),
        );
        let n24 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), self.lu.m23, x34), self.lu.m24, x44),
        );
        let (x21, x22, x23, x24) = R::div4(n21, n22, n23, n24, self.lu.m22);
        let n11 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), R::one()), self.lu.m12, x21),
                    self.lu.m13,
                    x31,
                ),
                self.lu.m14,
                x41,
            ),
        );
        let n12 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_zero(), self.lu.m12, x22), self.lu.m13, x32,
                ),
                self.lu.m14,
                x42,
            ),
        );
        let n13 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_zero(), self.lu.m12, x23), self.lu.m13, x33,
                ),
                self.lu.m14,
                x43,
            ),
        );
        let n14 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_zero(), self.lu.m12, x24), self.lu.m13, x34,
                ),
                self.lu.m14,
                x44,
            ),
        );
        let (x11, x12, x13, x14) = R::div4(n11, n12, n13, n14, self.lu.m11);
        let mut c11 = x11;
        let mut c12 = x12;
        let mut c13 = x13;
        let mut c14 = x14;
        let mut c21 = x21;
        let mut c22 = x22;
        let mut c23 = x23;
        let mut c24 = x24;
        let mut c31 = x31;
        let mut c32 = x32;
        let mut c33 = x33;
        let mut c34 = x34;
        let mut c41 = x41;
        let mut c42 = x42;
        let mut c43 = x43;
        let mut c44 = x44;
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
        }
        Some(
            Matrix4 {
                m11: c11,
                m21: c21,
                m31: c31,
                m41: c41,
                m12: c12,
                m22: c22,
                m32: c32,
                m42: c42,
                m13: c13,
                m23: c23,
                m33: c33,
                m43: c43,
                m14: c14,
                m24: c24,
                m34: c34,
                m44: c44,
            },
        )
    }

    /// The determinant: the product of the 4 pivots, with the sign of the permutation. Upstream:
    /// `LU::determinant`.
    ///
    /// The sign is applied to the FIRST pivot — an exact negation — and the product is then a
    /// left-to-right chain of 3 floored multiplications, so the result stays a floor chain instead
    /// of the negation of one (`-floor(x)` is `ceil(-x)`, one ulp off). `Real` exposes no `Wide *
    /// T`, so a product of 4 scalars cannot be accumulated exactly: each intermediate is floored
    /// once, which adds at most 3 ulp to the error already carried by the pivots.
    ///
    /// Exactly zero for a singular matrix. Panics with the scalar's overflow error if an
    /// intermediate product does not fit; partial pivoting makes the 4 pivots comparable in
    /// magnitude, so the partial products grow monotonically toward the determinant and an
    /// intermediate overflow implies the determinant itself does not fit.
    ///
    /// PREFER `Matrix4::determinant` when the factorisation is not needed for something else: the
    /// closed form sums exact minors, where this one multiplies pivots that already carry the
    /// rounding of the elimination. Measured on the 3x3 oracle vectors, the gap is not subtle —
    /// worst error 556 ulp for the cofactors against 181 307 427 for the pivots, 4 of the 18 cases
    /// outside the oracle tolerance, and 10 660 gas against 41 240
    /// (`bench_lu3_vs_matrix3_determinant` and `test_try_inverse_versus_matrix3_cofactors` in
    /// `lu3`). This method earns its keep on the 6x6, which has no closed form, and whenever the
    /// factorisation is already in hand.
    fn determinant(self: Lu4<T>) -> T {
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
        let d = if neg {
            -self.lu.m11
        } else {
            self.lu.m11
        };
        let d = d * self.lu.m22;
        let d = d * self.lu.m33;
        d * self.lu.m44
    }
}

/// Crate-internal kernels of `Lu4<T>` (WP 8.0: the public API is strictly upstream's): the row
/// permutation applied to a vector or to the rows of a matrix, upstream's
/// `lu.p().permute_rows(&mut m)` (to be exposed as `Perm4::permute_rows` by the
/// `PermutationSequence` completion of docs/API_PARITY.md P14).
#[generate_trait]
pub(crate) impl Lu4InternalImpl<
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
> of Lu4InternalTrait<T> {
    /// `P * v`: the transpositions applied to the components of `v`, in factorisation order. Exact:
    /// moves and comparisons only. Upstream: `LU::p().permute_rows(&mut v)`.
    fn permute(self: Lu4<T>, v: Vector4<T>) -> Vector4<T> {
        let mut x1 = v.x;
        let mut x2 = v.y;
        let mut x3 = v.z;
        let mut x4 = v.w;
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
        }
        if self.p.p2 == 3 {
            let t = x2;
            x2 = x3;
            x3 = t;
        } else if self.p.p2 == 4 {
            let t = x2;
            x2 = x4;
            x4 = t;
        }
        if self.p.p3 == 4 {
            let t = x3;
            x3 = x4;
            x4 = t;
        }
        Vector4 { x: x1, y: x2, z: x3, w: x4 }
    }
    /// `P * m`: the same transpositions applied to the ROWS of `m`, so that `lu.permute_rows(a)` is
    /// `lu.l() * lu.u()` up to the rounding of the factorisation (this is how the tests check the
    /// decomposition). Exact: moves and comparisons only. Upstream: `LU::p().permute_rows(&mut m)`.
    fn permute_rows(self: Lu4<T>, m: Matrix4<T>) -> Matrix4<T> {
        let mut a11 = m.m11;
        let mut a12 = m.m12;
        let mut a13 = m.m13;
        let mut a14 = m.m14;
        let mut a21 = m.m21;
        let mut a22 = m.m22;
        let mut a23 = m.m23;
        let mut a24 = m.m24;
        let mut a31 = m.m31;
        let mut a32 = m.m32;
        let mut a33 = m.m33;
        let mut a34 = m.m34;
        let mut a41 = m.m41;
        let mut a42 = m.m42;
        let mut a43 = m.m43;
        let mut a44 = m.m44;
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
        }
        Matrix4 {
            m11: a11,
            m21: a21,
            m31: a31,
            m41: a41,
            m12: a12,
            m22: a22,
            m32: a32,
            m42: a42,
            m13: a13,
            m23: a23,
            m33: a33,
            m43: a43,
            m14: a14,
            m24: a24,
            m34: a34,
            m44: a44,
        }
    }
}

/// `Matrix4` methods that go through the LU factorisation; upstream carries them on the matrix
/// itself. Import `Matrix4LuTrait` to use them.
#[generate_trait]
pub impl Matrix4LuImpl<
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
> of Matrix4LuTrait<T> {
    /// The LU factorisation with partial pivoting. Upstream: `Matrix4::lu`.
    #[inline(always)]
    fn lu(self: Matrix4<T>) -> Lu4<T> {
        Lu4Trait::new(self)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Lu4`: an exactly representable factorisation, the identities (`P A = L U`, `A
    //! A^-1 = I`, `|l_ik| <= 1`), the singular cases, the overflow panic and the oracle vectors of
    //! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
    //!
    //! Tolerance of the oracle assertions: the oracle's `tol` plus ONE relative ulp of the expected
    //! value. `base::matrix_test_utils::oracle_tol` states why, with the measurement.
    //!
    //! Gas benchmarks of `Lu4` (`bench_lu4_<op>__<variant>`, net = raw - the `baseline` of the
    //! group), and the alternative implementations that lost, kept as evidence together with the
    //! tests that show why (AGENTS.md rule 8):
    //!
    //! - `alt_no_pivot`: the elimination without partial pivoting. Cheaper and shorter, and wrong
    //! on a matrix as ordinary as a permuted identity.
    //!
    //! - `alt_recip`: one reciprocal per pivot instead of one correctly rounded division per output
    //! scalar. DEARER for `solve`, where a single right-hand side does not amortise the reciprocal,
    //! cheaper for `try_inverse`, where 4 columns share it, and a second rounding per output in
    //! both.
    //!
    //! - `alt_solve_columns`: the inverse as 4 calls to `solve`. Bit-identical, dearer.

    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix4::{Matrix4, Matrix4Trait};
    use crate::base::matrix_test_utils::{
        abs_raw, fx, int, m4, max_abs_m4, max_abs_v4, max_ulp_diff4, max_ulp_diff_v4, oracle_tol,
        ulp_diff, v4it, v4t,
    };
    use crate::base::vector4::Vector4;
    use crate::linalg::lu::{Perm4, Perm4Trait, oracle_lu4 as oracle};
    use super::{Lu4, Lu4InternalTrait, Lu4Trait, Matrix4LuTrait};

    /// The oracle's first `unit` 4x4 case whose factorisation actually swaps rows, so every
    /// benchmark exercises the permutation.
    fn a_bench() -> Matrix4<Fixed> {
        m4(
            [
                [-125512283, -3597765021, -1339269412, -1828698387],
                [-3905117829, -101544735, 2917390324, -1089761038],
                [-220744962, 2678614957, -1309551624, -3799912375],
                [-2641031148, 493840096, -2692687512, 1437594188],
            ],
        )
    }

    /// Its right-hand side.
    fn b_bench() -> Vector4<Fixed> {
        v4t((7470372587, 3697509183, 6158142748, 5116133854))
    }

    /// `a_bench()` already factored, so that the benchmarks of the derived operations do not pay
    /// for `new`.
    fn f_bench() -> Lu4<Fixed> {
        Lu4 {
            lu: m4(
                [
                    [-3905117829, -101544735, 2917390324, -1089761038],
                    [138042224, -3594501327, -1433035679, -1793672967],
                    [2904686338, -672132918, -4889978813, 1893902051],
                    [242782019, -3207459345, 2235014795, -6063365662],
                ],
            ),
            p: Perm4 { p1: 2, p2: 2, p3: 4 },
        }
    }

    /// A matrix whose fixed-point factorisation is EXACT: `L` has at most 2 fractional bits, `U`
    /// integer entries, so no division and no update rounds. Built by `L * U` then a row shuffle,
    /// which partial pivoting undoes exactly because `|l_ik| < 1`.
    fn a_exact() -> Matrix4<Fixed> {
        m4(
            [
                [0, 3221225472, 7516192768, 9663676416],
                [-3221225472, 5368709120, -8589934592, -31138512896],
                [-12884901888, 12884901888, 8589934592, -17179869184],
                [3221225472, -7516192768, 10737418240, 8589934592],
            ],
        )
    }

    /// An exactly singular integer matrix (one row is an integer combination of the others, and
    /// every multiplier of the elimination is dyadic, so the last pivot is exactly zero).
    fn a_singular() -> Matrix4<Fixed> {
        m4(
            [
                [17179869184, 8589934592, 25769803776, 85899345920], [0, 0, 0, -21474836480],
                [-12884901888, -21474836480, -12884901888, 4294967296],
                [4294967296, 17179869184, 0, -25769803776],
            ],
        )
    }

    /// `a_singular()` already factored (its last pivot is exactly zero), so the benchmark of the
    /// rejection path measures the rejection and not `new`.
    fn f_singular() -> Lu4<Fixed> {
        Lu4 {
            lu: m4(
                [
                    [17179869184, 8589934592, 25769803776, 85899345920],
                    [-3221225472, -15032385536, 6442450944, 68719476736], [0, 0, 0, -21474836480],
                    [1073741824, -4294967296, 0, 21474836480],
                ],
            ),
            p: Perm4 { p1: 1, p2: 3, p3: 3 },
        }
    }

    /// The same elimination WITHOUT partial pivoting: no `abs`, no comparison, no row swap, and the
    /// identity permutation. Kept as evidence (AGENTS.md rule 8): it is cheaper and shorter, and it
    /// is wrong — the pivot of step `k` is whatever sits at `a_kk`, so a matrix as ordinary as a
    /// permuted identity factors with a zero pivot, and nothing bounds `|l_ik|`, which is what
    /// keeps `U` from growing. Upstream has no unpivoted variant either.
    fn new_no_pivot(matrix: Matrix4<Fixed>) -> Lu4<Fixed> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a13 = matrix.m13;
        let mut a14 = matrix.m14;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        let mut a23 = matrix.m23;
        let mut a24 = matrix.m24;
        let mut a31 = matrix.m31;
        let mut a32 = matrix.m32;
        let mut a33 = matrix.m33;
        let mut a34 = matrix.m34;
        let mut a41 = matrix.m41;
        let mut a42 = matrix.m42;
        let mut a43 = matrix.m43;
        let mut a44 = matrix.m44;
        if a11 != Real::zero() {
            let l = a21 / a11;
            let nl = -l;
            a22 = Real::mul_add(nl, a12, a22);
            a23 = Real::mul_add(nl, a13, a23);
            a24 = Real::mul_add(nl, a14, a24);
            a21 = l;
            let l = a31 / a11;
            let nl = -l;
            a32 = Real::mul_add(nl, a12, a32);
            a33 = Real::mul_add(nl, a13, a33);
            a34 = Real::mul_add(nl, a14, a34);
            a31 = l;
            let l = a41 / a11;
            let nl = -l;
            a42 = Real::mul_add(nl, a12, a42);
            a43 = Real::mul_add(nl, a13, a43);
            a44 = Real::mul_add(nl, a14, a44);
            a41 = l;
        }
        if a22 != Real::zero() {
            let l = a32 / a22;
            let nl = -l;
            a33 = Real::mul_add(nl, a23, a33);
            a34 = Real::mul_add(nl, a24, a34);
            a32 = l;
            let l = a42 / a22;
            let nl = -l;
            a43 = Real::mul_add(nl, a23, a43);
            a44 = Real::mul_add(nl, a24, a44);
            a42 = l;
        }
        if a33 != Real::zero() {
            let l = a43 / a33;
            let nl = -l;
            a44 = Real::mul_add(nl, a34, a44);
            a43 = l;
        }
        Lu4 {
            lu: Matrix4 {
                m11: a11,
                m21: a21,
                m31: a31,
                m41: a41,
                m12: a12,
                m22: a22,
                m32: a32,
                m42: a42,
                m13: a13,
                m23: a23,
                m33: a33,
                m43: a43,
                m14: a14,
                m24: a24,
                m34: a34,
                m44: a44,
            },
            p: Perm4 { p1: 1, p2: 2, p3: 3 },
        }
    }

    /// `solve` with ONE reciprocal per pivot and 4 multiplications instead of 4 correctly rounded
    /// divisions.
    /// Kept as evidence, and it loses on both counts: a reciprocal (2 190) plus a multiplication (1
    /// 750) is dearer than a division (2 740), and a single right-hand side gives nothing to
    /// amortise it over, so it costs 39 660 against 34 180 gas; and rounding `1 / u_ii` before
    /// using it puts 4 of the oracle cases outside the tolerance against 0
    /// (`test_solve_candidates_error`).
    fn solve_recip(f: Lu4<Fixed>, b: Vector4<Fixed>) -> Option<Vector4<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let r3 = Real::recip(f.lu.m33);
        let r4 = Real::recip(f.lu.m44);
        let pb = f.permute(b);
        let y1 = pb.x;
        let y2 = Real::mul_add(-f.lu.m21, y1, pb.y);
        let y3 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), pb.z), f.lu.m31, y1),
                f.lu.m32,
                y2,
            ),
        );
        let y4 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), pb.w), f.lu.m41, y1,
                    ),
                    f.lu.m42,
                    y2,
                ),
                f.lu.m43,
                y3,
            ),
        );
        let x4 = y4 * r4;
        let x3 = Real::mul_add(-f.lu.m34, x4, y3) * r3;
        let x2 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y2), f.lu.m23, x3),
                f.lu.m24,
                x4,
            ),
        )
            * r2;
        let x1 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), y1), f.lu.m12, x2,
                    ),
                    f.lu.m13,
                    x3,
                ),
                f.lu.m14,
                x4,
            ),
        )
            * r1;
        Some(Vector4 { x: x1, y: x2, z: x3, w: x4 })
    }

    /// `try_inverse` as 4 full calls to `solve` on the unit vectors — the obvious route, and the
    /// one upstream takes. Kept as evidence: it gives BIT-IDENTICAL results
    /// (`test_try_inverse_candidates_agree`) for strictly more gas, because it pays the permutation
    /// 4 times and multiplies by the leading zeros of each unit vector.
    fn try_inverse_solve_columns(f: Lu4<Fixed>) -> Option<Matrix4<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let c1 = f.solve(Vector4 { x: int(1), y: int(0), z: int(0), w: int(0) }).unwrap();
        let c2 = f.solve(Vector4 { x: int(0), y: int(1), z: int(0), w: int(0) }).unwrap();
        let c3 = f.solve(Vector4 { x: int(0), y: int(0), z: int(1), w: int(0) }).unwrap();
        let c4 = f.solve(Vector4 { x: int(0), y: int(0), z: int(0), w: int(1) }).unwrap();
        Some(
            Matrix4 {
                m11: c1.x,
                m21: c1.y,
                m31: c1.z,
                m41: c1.w,
                m12: c2.x,
                m22: c2.y,
                m32: c2.z,
                m42: c2.w,
                m13: c3.x,
                m23: c3.y,
                m33: c3.z,
                m43: c3.w,
                m14: c4.x,
                m24: c4.y,
                m34: c4.z,
                m44: c4.w,
            },
        )
    }

    /// `try_inverse` with ONE reciprocal per pivot instead of one division per entry of the back
    /// substitution. Here the reciprocal IS amortised (4 columns share it), so this is the
    /// candidate the gas argument favours; it is not shipped because it rounds `1 / u_ii` before
    /// using it, which is the second rounding per output scalar that AGENTS.md rule 4 and DESIGN D2
    /// forbid (the same call `Vector4::unscale` makes).
    fn try_inverse_recip(f: Lu4<Fixed>) -> Option<Matrix4<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let r3 = Real::recip(f.lu.m33);
        let r4 = Real::recip(f.lu.m44);
        let y21 = -f.lu.m21;
        let y31 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m31), f.lu.m32, y21,
            ),
        );
        let y41 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m41), f.lu.m42, y21,
                ),
                f.lu.m43,
                y31,
            ),
        );
        let x41 = y41 * r4;
        let x31 = Real::mul_add(-f.lu.m34, x41, y31) * r3;
        let x21 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y21), f.lu.m23, x31),
                f.lu.m24,
                x41,
            ),
        )
            * r2;
        let x11 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(
                        Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m12, x21,
                    ),
                    f.lu.m13,
                    x31,
                ),
                f.lu.m14,
                x41,
            ),
        )
            * r1;
        let y32 = -f.lu.m32;
        let y42 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m42), f.lu.m43, y32,
            ),
        );
        let x42 = y42 * r4;
        let x32 = Real::mul_add(-f.lu.m34, x42, y32) * r3;
        let x22 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m23, x32,
                ),
                f.lu.m24,
                x42,
            ),
        )
            * r2;
        let x12 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x22), f.lu.m13, x32,
                ),
                f.lu.m14,
                x42,
            ),
        )
            * r1;
        let y43 = -f.lu.m43;
        let x43 = y43 * r4;
        let x33 = Real::mul_add(-f.lu.m34, x43, int(1)) * r3;
        let x23 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m23, x33), f.lu.m24, x43,
            ),
        )
            * r2;
        let x13 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x23), f.lu.m13, x33,
                ),
                f.lu.m14,
                x43,
            ),
        )
            * r1;
        let x44 = int(1) * r4;
        let x34 = Real::wide_rescale(Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m34, x44))
            * r3;
        let x24 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m23, x34), f.lu.m24, x44,
            ),
        )
            * r2;
        let x14 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x24), f.lu.m13, x34,
                ),
                f.lu.m14,
                x44,
            ),
        )
            * r1;
        let mut c11 = x11;
        let mut c12 = x12;
        let mut c13 = x13;
        let mut c14 = x14;
        let mut c21 = x21;
        let mut c22 = x22;
        let mut c23 = x23;
        let mut c24 = x24;
        let mut c31 = x31;
        let mut c32 = x32;
        let mut c33 = x33;
        let mut c34 = x34;
        let mut c41 = x41;
        let mut c42 = x42;
        let mut c43 = x43;
        let mut c44 = x44;
        if f.p.p3 == 4 {
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
        }
        if f.p.p2 == 3 {
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
        } else if f.p.p2 == 4 {
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
        }
        if f.p.p1 == 2 {
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
        } else if f.p.p1 == 3 {
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
        } else if f.p.p1 == 4 {
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
        }
        Some(
            Matrix4 {
                m11: c11,
                m21: c21,
                m31: c31,
                m41: c41,
                m12: c12,
                m22: c22,
                m32: c32,
                m42: c42,
                m13: c13,
                m23: c23,
                m33: c33,
                m43: c43,
                m14: c14,
                m24: c24,
                m34: c34,
                m44: c44,
            },
        )
    }

    /// `(cases above the LU tolerance, worst error in ulp)` of a `solve` candidate: 0 = shipped
    /// (one division per pivot), 1 = one reciprocal per pivot, 2 = shipped substitution on the
    /// UNPIVOTED factorisation.
    fn solve_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle::lu4_solve_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let f = if variant == 2 {
                new_no_pivot(m4(a))
            } else {
                Lu4Trait::new(m4(a))
            };
            let got = if variant == 1 {
                solve_recip(f, v4t(b))
            } else {
                f.solve(v4t(b))
            };
            let e = v4t(expected);
            let err = max_ulp_diff_v4(got.unwrap(), e);
            if err > oracle_tol(max_abs_v4(e), tol) {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    #[test]
    fn test_new_is_exact_on_a_dyadic_matrix() {
        let f = Lu4Trait::new(a_exact());
        let lu = f.lu;
        assert!(
            lu == m4(
                [
                    [-12884901888, 12884901888, 8589934592, -17179869184],
                    [-1073741824, -4294967296, 12884901888, 4294967296],
                    [0, -3221225472, 17179869184, 12884901888],
                    [1073741824, -2147483648, -1073741824, -21474836480],
                ],
            ),
        );
        assert!(f.p() == Perm4 { p1: 3, p2: 4, p3: 3 });
        assert!(f.permute_rows(a_exact()) == f.l() * f.u());
        assert!(f.determinant() == fx(-257698037760));
    }

    #[test]
    fn test_new_reconstruction_oracle() {
        // `P A == L U` within 44 raw units on the lu4 vectors.
        let mut cases = oracle::lu4_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Lu4Trait::new(m4(a));
            worst = core::cmp::max(worst, max_ulp_diff4(f.permute_rows(m4(a)), f.l() * f.u()));
        }
        assert!(worst == 25, "reconstruction error {worst}");
    }

    #[test]
    fn test_new_multipliers_are_bounded_by_one() {
        // What partial pivoting buys: |l_ik| <= 1, which bounds the growth of `U`.
        let mut cases = oracle::lu4_solve_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            assert!(max_abs_m4(Lu4Trait::new(m4(a)).l()) <= 0x100000000);
        }
    }

    #[test]
    fn test_factors_permutation_and_accessors() {
        let f = Lu4Trait::new(a_exact());
        let l = f.l();
        assert!(
            l == m4(
                [
                    [4294967296, 0, 0, 0], [-1073741824, 4294967296, 0, 0],
                    [0, -3221225472, 4294967296, 0],
                    [1073741824, -2147483648, -1073741824, 4294967296],
                ],
            ),
        );
        let u = f.u();
        assert!(
            u == m4(
                [
                    [-12884901888, 12884901888, 8589934592, -17179869184],
                    [0, -4294967296, 12884901888, 4294967296], [0, 0, 17179869184, 12884901888],
                    [0, 0, 0, -21474836480],
                ],
            ),
        );
        let pa = f.permute_rows(a_exact());
        assert!(
            pa == m4(
                [
                    [-12884901888, 12884901888, 8589934592, -17179869184],
                    [3221225472, -7516192768, 10737418240, 8589934592],
                    [0, 3221225472, 7516192768, 9663676416],
                    [-3221225472, 5368709120, -8589934592, -31138512896],
                ],
            ),
        );
        assert!(f.permute(v4it((1, 2, 3, 4))) == v4it((3, 4, 1, 2)));
        // the identity factors without a single swap
        let id = Lu4Trait::new(Matrix4Trait::<Fixed>::identity());
        assert!(id.p() == Perm4Trait::identity());
        assert!(id.permute(b_bench()) == b_bench());
        assert!(id.permute_rows(a_bench()) == a_bench());
        assert!(id.l() == Matrix4Trait::<Fixed>::identity());
        assert!(id.u() == Matrix4Trait::<Fixed>::identity());
    }

    #[test]
    fn test_singular_is_rejected() {
        let f = Lu4Trait::new(a_singular());
        assert!(!f.is_invertible());
        assert!(f.solve(b_bench()).is_none());
        assert!(f.try_inverse().is_none());
        assert!(f.determinant() == int(0));
        let z = Lu4Trait::new(Matrix4Trait::<Fixed>::zeros());
        assert!(!z.is_invertible());
        assert!(z.try_inverse().is_none());
        assert!(z.determinant() == int(0));
        assert!(z.p() == Perm4Trait::identity());
        assert!(Lu4Trait::new(a_bench()).is_invertible());
    }

    #[test]
    fn test_solve_oracle() {
        let mut cases = oracle::lu4_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let x = Lu4Trait::new(m4(a)).solve(v4t(b)).unwrap();
            let e = v4t(expected);
            let err = max_ulp_diff_v4(x, e);
            assert!(err <= oracle_tol(max_abs_v4(e), tol), "solve error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 1451);
    }

    #[test]
    fn test_try_inverse_oracle() {
        let mut cases = oracle::lu4_inverse_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let inv = Lu4Trait::new(m4(a)).try_inverse().unwrap();
            let e = m4(expected);
            let err = max_ulp_diff4(inv, e);
            assert!(err <= oracle_tol(max_abs_m4(e), tol), "inverse error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 2619);
    }

    #[test]
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle::lu4_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let inv = Lu4Trait::new(m4(a)).try_inverse().unwrap();
            // Worst residual over the oracle: 117 ulp.
            assert!((m4(a) * inv).is_identity(117));
            assert!((inv * m4(a)).is_identity(117));
        }
    }

    #[test]
    fn test_try_inverse_candidates() {
        // The static unit right-hand sides give BIT-IDENTICAL results to 4 calls
        // to `solve`; the saving is pure gas.
        let mut cases = oracle::lu4_inverse_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let f = Lu4Trait::new(m4(a));
            assert!(f.try_inverse().unwrap() == try_inverse_solve_columns(f).unwrap());
            worst =
                core::cmp::max(
                    worst, max_ulp_diff4(f.try_inverse().unwrap(), try_inverse_recip(f).unwrap()),
                );
        }
        // ... and the reciprocal variant drifts by at most this many ulp from it.
        assert!(worst == 7);
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_overflow_panics() {
        // 2^-32 * I: every pivot is 1 raw unit, so the inverse is 2^32 * I.
        let _ = black_box(Matrix4Trait::from_diagonal_element(fx(1))).lu().try_inverse();
    }

    #[test]
    fn test_determinant_oracle() {
        let mut cases = oracle::lu4_determinant_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let det = Lu4Trait::new(m4(a)).determinant();
            let err = ulp_diff(det, fx(expected));
            assert!(err <= oracle_tol(abs_raw(fx(expected)), tol), "determinant error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 17456);
    }

    #[test]
    fn test_determinant_exact_and_sign() {
        assert!(Lu4Trait::new(Matrix4Trait::<Fixed>::identity()).determinant() == int(1));
        let d = Matrix4Trait::from_diagonal(v4it((2, -4, 3, -1)));
        assert!(Lu4Trait::new(d).determinant() == int(24));
        // Swapping two rows flips the sign exactly (the factorisation is exact here,
        // and the sign is applied to the first pivot before any rounding).
        let s = m4(
            [
                [-3221225472, 5368709120, -8589934592, -31138512896],
                [0, 3221225472, 7516192768, 9663676416],
                [-12884901888, 12884901888, 8589934592, -17179869184],
                [3221225472, -7516192768, 10737418240, 8589934592],
            ],
        );
        assert!(Lu4Trait::new(s).determinant() == -fx(-257698037760));
    }

    #[test]
    fn test_solve_candidates_error() {
        // Oracle, 15 well-conditioned matrices: (cases above the LU tolerance,
        // worst error in ulp) of the shipped divisions, of one reciprocal per
        // pivot, and of the shipped substitution on an UNPIVOTED factorisation.
        // The unpivoted figure is NOT a win: these matrices are random and
        // well-conditioned, so their leading entries happen to be usable pivots.
        // `test_no_pivot_candidate_is_wrong` shows the structural failure.
        assert!(solve_failures(0) == (0, 1451));
        assert!(solve_failures(1) == (3, 1450));
        assert!(solve_failures(2) == (0, 678));
    }

    #[test]
    fn test_no_pivot_candidate_is_wrong() {
        // A permuted identity: the unpivoted elimination finds a zero at `a11` and
        // gives up, where partial pivoting factors it exactly.
        let swapped = m4(
            [
                [0, 4294967296, 0, 0], [4294967296, 0, 0, 0], [0, 0, 4294967296, 0],
                [0, 0, 0, 4294967296],
            ],
        );
        assert!(!new_no_pivot(swapped).is_invertible());
        let f = Lu4Trait::new(swapped);
        assert!(f.is_invertible());
        assert!(f.determinant() == int(-1));
    }

    // --- gas benchmarks --------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_lu4_new__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(
            m4(
                [
                    [-3905117829, -101544735, 2917390324, -1089761038],
                    [138042224, -3594501327, -1433035679, -1793672967],
                    [2904686338, -672132918, -4889978813, 1893902051],
                    [242782019, -3207459345, 2235014795, -6063365662],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_new__pivot() {
        let a = black_box(a_bench());
        let e = black_box(
            m4(
                [
                    [-3905117829, -101544735, 2917390324, -1089761038],
                    [138042224, -3594501327, -1433035679, -1793672967],
                    [2904686339, -672132918, -4889978813, 1893902052],
                    [242782019, -3207459345, 2235014795, -6063365663],
                ],
            ),
        );
        assert!(Lu4Trait::new(a).lu == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_new__alt_no_pivot() {
        let a = black_box(a_bench());
        let e = black_box(
            m4(
                [
                    [-125512283, -3597765021, -1339269412, -1828698387],
                    [133631171083, 111837271070, 44586657535, 55807321098],
                    [7553781749, 345871238, -2544646850, -5077817040],
                    [90374759643, 2926289000, 8253522131, 11651805292],
                ],
            ),
        );
        assert!(new_no_pivot(a).lu == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_factors__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(
            m4(
                [
                    [4294967296, 0, 0, 0], [138042224, 4294967296, 0, 0],
                    [2904686338, -672132918, 4294967296, 0],
                    [242782019, -3207459345, 2235014795, 4294967296],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_factors__l() {
        let f = black_box(f_bench());
        let e = black_box(
            m4(
                [
                    [4294967296, 0, 0, 0], [138042224, 4294967296, 0, 0],
                    [2904686338, -672132918, 4294967296, 0],
                    [242782019, -3207459345, 2235014795, 4294967296],
                ],
            ),
        );
        assert!(f.l() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_factors__u() {
        let f = black_box(f_bench());
        let e = black_box(
            m4(
                [
                    [-3905117829, -101544735, 2917390324, -1089761038],
                    [0, -3594501327, -1433035679, -1793672967], [0, 0, -4889978813, 1893902051],
                    [0, 0, 0, -6063365662],
                ],
            ),
        );
        assert!(f.u() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_p__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(Perm4 { p1: 2, p2: 2, p3: 4 });
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_p__field() {
        let f = black_box(f_bench());
        let e = black_box(Perm4 { p1: 2, p2: 2, p3: 4 });
        assert!(f.p() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_permute__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(v4t((3697509183, 7470372587, 5116133854, 6158142748)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_permute__transpositions() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(v4t((3697509183, 7470372587, 5116133854, 6158142748)));
        assert!(f.permute(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_permute_rows__baseline() {
        let _f = black_box(f_bench());
        let _a = black_box(a_bench());
        let e = black_box(
            m4(
                [
                    [-3905117829, -101544735, 2917390324, -1089761038],
                    [-125512283, -3597765021, -1339269412, -1828698387],
                    [-2641031148, 493840096, -2692687512, 1437594188],
                    [-220744962, 2678614957, -1309551624, -3799912375],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_permute_rows__transpositions() {
        let f = black_box(f_bench());
        let a = black_box(a_bench());
        let e = black_box(
            m4(
                [
                    [-3905117829, -101544735, 2917390324, -1089761038],
                    [-125512283, -3597765021, -1339269412, -1828698387],
                    [-2641031148, 493840096, -2692687512, 1437594188],
                    [-220744962, 2678614957, -1309551624, -3799912375],
                ],
            ),
        );
        assert!(f.permute_rows(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_is_invertible__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_is_invertible__pivots() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.is_invertible() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_solve__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(Some(v4t((-6526739453, -3077920152, -5908376560, -6714764226))));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_solve__substitution() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v4t((-6526739453, -3077920151, -5908376560, -6714764226))));
        assert!(f.solve(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_solve__alt_recip() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v4t((-6526739453, -3077920152, -5908376560, -6714764225))));
        assert!(solve_recip(f, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_solve_singular__baseline() {
        let _s = black_box(f_singular());
        let _b = black_box(b_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_solve_singular__none() {
        let s = black_box(f_singular());
        let b = black_box(b_bench());
        let e = black_box(true);
        assert!(s.solve(b).is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_try_inverse__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(
            Some(
                m4(
                    [
                        [-368910955, -2803986567, -82972189, -2814138558],
                        [-3573914248, -315013115, 1987894369, 469481705],
                        [-1374341491, 2247339518, -1178301687, -3159192093],
                        [-2024239780, -833661343, -3042327497, 1583166179],
                    ],
                ),
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_try_inverse__columns() {
        let f = black_box(f_bench());
        let e = black_box(
            Some(
                m4(
                    [
                        [-368910954, -2803986567, -82972188, -2814138557],
                        [-3573914249, -315013115, 1987894368, 469481704],
                        [-1374341490, 2247339518, -1178301686, -3159192092],
                        [-2024239779, -833661342, -3042327496, 1583166180],
                    ],
                ),
            ),
        );
        assert!(f.try_inverse() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_try_inverse__alt_solve_columns() {
        let f = black_box(f_bench());
        let e = black_box(
            Some(
                m4(
                    [
                        [-368910954, -2803986567, -82972188, -2814138557],
                        [-3573914249, -315013115, 1987894368, 469481704],
                        [-1374341490, 2247339518, -1178301686, -3159192092],
                        [-2024239779, -833661342, -3042327496, 1583166180],
                    ],
                ),
            ),
        );
        assert!(try_inverse_solve_columns(f) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_try_inverse__alt_recip() {
        let f = black_box(f_bench());
        let e = black_box(
            Some(
                m4(
                    [
                        [-368910955, -2803986567, -82972190, -2814138558],
                        [-3573914248, -315013115, 1987894369, 469481705],
                        [-1374341491, 2247339518, -1178301687, -3159192093],
                        [-2024239780, -833661342, -3042327496, 1583166179],
                    ],
                ),
            ),
        );
        assert!(try_inverse_recip(f) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_determinant__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(fx(5253079147));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu4_determinant__pivots() {
        let f = black_box(f_bench());
        let e = black_box(fx(5253079147));
        assert!(f.determinant() == e);
    }
}
