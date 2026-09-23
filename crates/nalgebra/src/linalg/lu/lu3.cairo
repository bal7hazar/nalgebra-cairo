//! `Lu3`: the LU factorisation with partial pivoting of a `Matrix3` (upstream
//! `nalgebra::linalg::LU` on a 3x3 matrix).
//!
//! `P * A = L * U`, with `L` unit lower triangular, `U` upper triangular and `P` the product of the
//! 2 row transpositions chosen by partial pivoting. Both factors share one `Matrix3` like upstream
//! (strict lower triangle = `L`, whose unit diagonal is implicit; upper triangle = `U`) and the
//! permutation is the compact `Perm3`.
//!
//! Everything is unrolled (DESIGN D4: no loop in static code) and every sum of products goes
//! through a fused `Real` kernel — `mul_add` for a single product, the explicit `Real::Wide`
//! accumulator beyond that — so each output scalar is floored once and range-checked once
//! (AGENTS.md rule 4).
//!
//! Partial pivoting costs 3 `abs` and comparisons plus 2 conditional row swaps (moves only), and
//! every swap duplicates the row it moves, so it also costs Sierra statements. Dropping it would be
//! cheaper and shorter and it is not an option: without it the pivot of step `k` is whatever sits
//! at `a_kk`, nothing bounds `|l_ik|`, and a matrix as ordinary as a permuted identity factors with
//! a zero pivot. `bench_lu3_new__alt_no_pivot` and `test_no_pivot_candidate_is_wrong` keep the
//! measurement and the counter-example. Upstream has no unpivoted variant either, only `LU`
//! (partial pivoting) and `FullPivLU` (complete pivoting).

use simba::scalar::Real;
use crate::base::matrix3::Matrix3;
use crate::base::vector3::Vector3;
use super::Perm3;

/// The LU factorisation with partial pivoting of a `Matrix3<T>`: `P * A = L * U`.
///
/// `lu` packs both factors (strict lower triangle = `L` without its unit diagonal, upper triangle =
/// `U`) and `p` is the row permutation. Built by `Lu3Trait::new` or `Matrix3LuTrait::lu`. Upstream:
/// `nalgebra::linalg::LU`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Lu3<T> {
    /// `L` (strict lower triangle) and `U` (upper triangle) packed in one matrix.
    pub lu: Matrix3<T>,
    /// The row transpositions applied by partial pivoting.
    pub p: Perm3,
}

/// Methods of `Lu3<T>` for any `Real` scalar.
#[generate_trait]
pub impl Lu3Impl<
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
> of Lu3Trait<T> {
    /// The LU factorisation of `matrix` with partial pivoting, fully unrolled. Upstream:
    /// `matrix.lu()` / `LU::new(matrix)`.
    ///
    /// Always succeeds, like upstream: a singular matrix simply leaves a zero on the diagonal of
    /// `U` (see `is_invertible`). At step `k`, the row of largest `|a_ik|` among rows `k..3` is
    /// swapped onto the diagonal (the FIRST such row, like upstream's `icamax`), the 3 multipliers
    /// `l_ik = a_ik / a_kk` are each one correctly rounded division, and the trailing submatrix is
    /// updated entry by entry with `Real::mul_add(-l_ik, a_kj, a_ij)`: ONE floor rounding and one
    /// overflow check per entry, never the two roundings of `a_ij - l_ik * a_kj`.
    ///
    /// A pivot column that is exactly zero is skipped — no swap, no permutation, zero multipliers
    /// —
    /// exactly like upstream's `continue`, so the division is never reached with a zero divisor.
    ///
    /// Error model: `l_ik` is off by at most 1 ulp (correctly rounded quotient) and every update
    /// floors once, so after each of the 2 steps an entry of `U` is within about `k * (1 + |a_kj|)`
    /// raw units of its exact value. Partial pivoting keeps `|l_ik| <= 1`, which is what bounds the
    /// growth of the trailing submatrix. Panics with the scalar's overflow error if an update does
    /// not fit.
    fn new(matrix: Matrix3<T>) -> Lu3<T> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a13 = matrix.m13;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        let mut a23 = matrix.m23;
        let mut a31 = matrix.m31;
        let mut a32 = matrix.m32;
        let mut a33 = matrix.m33;
        // step 1: largest pivot among rows 1..3 of column 1
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
        }
        if piv != R::ZERO {
            let l = R::div(a21, a11);
            let nl = -l;
            a22 = R::mul_add(nl, a12, a22);
            a23 = R::mul_add(nl, a13, a23);
            a21 = l;
            let l = R::div(a31, a11);
            let nl = -l;
            a32 = R::mul_add(nl, a12, a32);
            a33 = R::mul_add(nl, a13, a33);
            a31 = l;
        }
        // step 2: largest pivot among rows 2..3 of column 2
        let mut p2 = 2_u8;
        let mut piv = R::abs(a22);
        let c = R::abs(a32);
        if c > piv {
            piv = c;
            p2 = 3;
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
        }
        if piv != R::ZERO {
            let l = R::div(a32, a22);
            let nl = -l;
            a33 = R::mul_add(nl, a23, a33);
            a32 = l;
        }
        Lu3 {
            lu: Matrix3 {
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
            p: Perm3 { p1, p2 },
        }
    }

    /// The unit lower triangular factor `L` (its diagonal of ones is implicit in the packed
    /// storage). Exact: moves only. Upstream: `LU::l`.
    #[inline(always)]
    fn l(self: Lu3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::ONE,
            m21: self.lu.m21,
            m31: self.lu.m31,
            m12: R::ZERO,
            m22: R::ONE,
            m32: self.lu.m32,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: R::ONE,
        }
    }

    /// The upper triangular factor `U`. Exact: moves only. Upstream: `LU::u`.
    #[inline(always)]
    fn u(self: Lu3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.lu.m11,
            m21: R::ZERO,
            m31: R::ZERO,
            m12: self.lu.m12,
            m22: self.lu.m22,
            m32: R::ZERO,
            m13: self.lu.m13,
            m23: self.lu.m23,
            m33: self.lu.m33,
        }
    }

    /// The row permutation `P`, as the compact sequence of 2 transpositions `Perm3` — never as a
    /// matrix, and never as an array (DESIGN D4). Upstream: `LU::p`, which returns a heap-allocated
    /// `PermutationSequence`.
    #[inline(always)]
    fn p(self: Lu3<T>) -> Perm3 {
        self.p
    }

    /// `P * v`: the transpositions applied to the components of `v`, in factorisation order. Exact:
    /// moves and comparisons only. Upstream: `LU::p().permute_rows(&mut v)`.
    fn permute(self: Lu3<T>, v: Vector3<T>) -> Vector3<T> {
        let mut x1 = v.x;
        let mut x2 = v.y;
        let mut x3 = v.z;
        if self.p.p1 == 2 {
            let t = x1;
            x1 = x2;
            x2 = t;
        } else if self.p.p1 == 3 {
            let t = x1;
            x1 = x3;
            x3 = t;
        }
        if self.p.p2 == 3 {
            let t = x2;
            x2 = x3;
            x3 = t;
        }
        Vector3 { x: x1, y: x2, z: x3 }
    }

    /// `P * m`: the same transpositions applied to the ROWS of `m`, so that `lu.permute_rows(a)` is
    /// `lu.l() * lu.u()` up to the rounding of the factorisation (this is how the tests check the
    /// decomposition). Exact: moves and comparisons only. Upstream: `LU::p().permute_rows(&mut m)`.
    fn permute_rows(self: Lu3<T>, m: Matrix3<T>) -> Matrix3<T> {
        let mut a11 = m.m11;
        let mut a12 = m.m12;
        let mut a13 = m.m13;
        let mut a21 = m.m21;
        let mut a22 = m.m22;
        let mut a23 = m.m23;
        let mut a31 = m.m31;
        let mut a32 = m.m32;
        let mut a33 = m.m33;
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
        }
        Matrix3 {
            m11: a11,
            m21: a21,
            m31: a31,
            m12: a12,
            m22: a22,
            m32: a32,
            m13: a13,
            m23: a23,
            m33: a33,
        }
    }

    /// Whether the factorisation is invertible: all 3 diagonal entries of `U` are EXACTLY nonzero
    /// (no epsilon), like upstream's `LU::is_invertible`.
    ///
    /// This is NOT the criterion of `Matrix3::try_inverse` (a computed determinant exactly zero),
    /// even though the determinant is the product of these very pivots: a matrix rejected here is
    /// rejected there too, but a matrix with 3 nonzero pivots can still have a determinant that
    /// underflows to zero in the product chain — then `Matrix3::try_inverse` gives `None` while
    /// the factorisation still solves. Neither criterion rejects a merely ill-conditioned matrix:
    /// it is factored and solved with the precision its conditioning allows.
    #[inline(always)]
    fn is_invertible(self: Lu3<T>) -> bool {
        self.lu.m11 != R::ZERO && self.lu.m22 != R::ZERO && self.lu.m33 != R::ZERO
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a pivot is exactly zero
    /// (`is_invertible`). Upstream: `LU::solve`.
    ///
    /// `b` is permuted (exactly), then `L y = P b` is solved by forward substitution and `U x = y`
    /// by back substitution. Each `y_i` costs ONE rounding — the whole sum of products is
    /// accumulated in `Real::Wide` and rescaled once — and each `x_i` costs TWO: the numerator,
    /// then the correctly rounded division by the pivot.
    ///
    /// The 3 correctly rounded divisions are kept rather than 3 reciprocals and 3 multiplications.
    /// That candidate loses on both counts here: a reciprocal plus a multiplication is dearer than
    /// a division and a single right-hand side amortises nothing, and rounding `1 / u_ii` before
    /// using it costs accuracy when `|u_ii| >> 1`. `bench_lu3_solve__alt_recip` and
    /// `test_solve_candidates_error` keep both measurements. `try_inverse` amortises a reciprocal
    /// over 3 columns and would be cheaper with one, and still does not use one (see there).
    ///
    /// Panics with the scalar's overflow error if a component of `x` does not fit.
    fn solve(self: Lu3<T>, b: Vector3<T>) -> Option<Vector3<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let pb = Self::permute(self, b);
        let y1 = pb.x;
        let y2 = R::mul_add(-self.lu.m21, y1, pb.y);
        let y3 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), pb.z), self.lu.m31, y1),
                self.lu.m32,
                y2,
            ),
        );
        let x3 = R::div(y3, self.lu.m33);
        let x2 = R::div(R::mul_add(-self.lu.m23, x3, y2), self.lu.m22);
        let x1 = R::div(
            R::wide_rescale(
                R::wide_sub_prod(
                    R::wide_sub_prod(R::wide_add(R::wide_zero(), y1), self.lu.m12, x2),
                    self.lu.m13,
                    x3,
                ),
            ),
            self.lu.m11,
        );
        Some(Vector3 { x: x1, y: x2, z: x3 })
    }

    /// The inverse, or `None` when a pivot is exactly zero (`is_invertible`). Upstream:
    /// `LU::try_inverse`.
    ///
    /// Computed as `A^-1 = (L U)^-1 P`, NOT as 3 calls to `solve`: the right-hand sides of `L U M =
    /// I` are then the STATIC unit vectors, whose leading zeros disappear at generation time (the
    /// forward substitution is 4 products instead of 9), and the permutation is applied at the end
    /// by swapping the COLUMNS of `M` in reverse factorisation order — moves only, exact.
    ///
    /// Rounding: one per entry of the forward substitution, two per entry of the back substitution
    /// (the numerator, then the correctly rounded division by the pivot). Panics with the scalar's
    /// overflow error if an entry of the inverse does not fit.
    ///
    /// Unlike `solve`, this one WOULD be cheaper with one reciprocal per pivot, which 3 columns
    /// amortise: 46 160 against 57 820 gas (net, `fixed` 0.3.0). It still divides — upstream's
    /// `solve_mut` divides by the pivot — because `mul(x, recip(u))` rounds twice where `x / u`
    /// rounds once, which is the rule DESIGN D2 and `Vector3::unscale` already follow; the drift is
    /// small but real (14 ulp on the oracle inverses).
    /// `bench_lu3_try_inverse__alt_recip` and `test_try_inverse_candidates` keep the measurement.
    ///
    /// The back substitution runs ROW by row across the 3 columns, so the quotients that
    /// share a pivot go through ONE prepared divisor (`Real::div3`, bit-identical
    /// to per-element division, cheaper from 3 quotients) and the corner `1 / u_33` is
    /// `Real::recip` (WP 7.2).
    ///
    /// Against `Matrix3::try_inverse` (cofactors with the integer pre-scaling): on the `matrix3`
    /// oracle vectors this one is 8 % DEARER since `fixed` 0.3.0 (95 760 against 88 360 gas,
    /// factorisation included; it was 11 % cheaper on the floor-division scalar) and less accurate
    /// (worst error 64 ulp against 27), both inside the oracle tolerance. Use the closed form when
    /// the inverse is the answer, this one when the same matrix is also solved against or its
    /// determinant is wanted, since the factorisation is then paid once
    /// (`bench_lu3_vs_matrix3_inverse`).
    fn try_inverse(self: Lu3<T>) -> Option<Matrix3<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y21 = -self.lu.m21;
        let y31 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub(R::wide_zero(), self.lu.m31), self.lu.m32, y21),
        );
        let y32 = -self.lu.m32;
        let x33 = R::recip(self.lu.m33);
        let x31 = R::div(y31, self.lu.m33);
        let x32 = R::div(y32, self.lu.m33);
        let n21 = R::mul_add(-self.lu.m23, x31, y21);
        let n22 = R::mul_add(-self.lu.m23, x32, R::ONE);
        let n23 = R::wide_rescale(R::wide_sub_prod(R::wide_zero(), self.lu.m23, x33));
        let (x21, x22, x23) = R::div3(n21, n22, n23, self.lu.m22);
        let n11 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), R::ONE), self.lu.m12, x21),
                self.lu.m13,
                x31,
            ),
        );
        let n12 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), self.lu.m12, x22), self.lu.m13, x32),
        );
        let n13 = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), self.lu.m12, x23), self.lu.m13, x33),
        );
        let (x11, x12, x13) = R::div3(n11, n12, n13, self.lu.m11);
        let mut c11 = x11;
        let mut c12 = x12;
        let mut c13 = x13;
        let mut c21 = x21;
        let mut c22 = x22;
        let mut c23 = x23;
        let mut c31 = x31;
        let mut c32 = x32;
        let mut c33 = x33;
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
        }
        Some(
            Matrix3 {
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
        )
    }

    /// The determinant: the product of the 3 pivots, with the sign of the permutation. Upstream:
    /// `LU::determinant`.
    ///
    /// The sign is applied to the FIRST pivot — an exact negation — and the product is then a
    /// left-to-right chain of 2 floored multiplications, so the result stays a floor chain instead
    /// of the negation of one (`-floor(x)` is `ceil(-x)`, one ulp off). `Real` exposes no `Wide *
    /// T`, so a product of 3 scalars cannot be accumulated exactly: each intermediate is floored
    /// once, which adds at most 2 ulp to the error already carried by the pivots.
    ///
    /// Exactly zero for a singular matrix. Panics with the scalar's overflow error if an
    /// intermediate product does not fit; partial pivoting makes the 3 pivots comparable in
    /// magnitude, so the partial products grow monotonically toward the determinant and an
    /// intermediate overflow implies the determinant itself does not fit.
    ///
    /// PREFER `Matrix3::determinant` when the factorisation is not needed for something else: the
    /// closed form sums exact minors, where this one multiplies pivots that already carry the
    /// rounding of the elimination. Measured on the 3x3 oracle vectors, the gap is not subtle —
    /// worst error 556 ulp for the cofactors against 181 307 427 for the pivots, 4 of the 18 cases
    /// outside the oracle tolerance, and 10 660 gas against 41 240
    /// (`bench_lu3_vs_matrix3_determinant` and `test_try_inverse_versus_matrix3_cofactors` in
    /// `lu3`). This method earns its keep on the 6x6, which has no closed form, and whenever the
    /// factorisation is already in hand.
    fn determinant(self: Lu3<T>) -> T {
        let mut neg = false;
        if self.p.p1 != 1 {
            neg = !neg;
        }
        if self.p.p2 != 2 {
            neg = !neg;
        }
        let d = if neg {
            -self.lu.m11
        } else {
            self.lu.m11
        };
        let d = d * self.lu.m22;
        d * self.lu.m33
    }
}

/// `Matrix3` methods that go through the LU factorisation; upstream carries them on the matrix
/// itself. Import `Matrix3LuTrait` to use them.
#[generate_trait]
pub impl Matrix3LuImpl<
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
> of Matrix3LuTrait<T> {
    /// The LU factorisation with partial pivoting. Upstream: `Matrix3::lu`.
    #[inline(always)]
    fn lu(self: Matrix3<T>) -> Lu3<T> {
        Lu3Trait::new(self)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Lu3`: an exactly representable factorisation, the identities (`P A = L U`, `A
    //! A^-1 = I`, `|l_ik| <= 1`), the singular cases, the overflow panic and the oracle vectors of
    //! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
    //!
    //! Tolerance of the oracle assertions: the oracle's `tol` plus ONE relative ulp of the expected
    //! value. `base::matrix_test_utils::oracle_tol` states why, with the measurement.
    //!
    //! Gas benchmarks of `Lu3` (`bench_lu3_<op>__<variant>`, net = raw - the `baseline` of the
    //! group), and the alternative implementations that lost, kept as evidence together with the
    //! tests that show why (AGENTS.md rule 8):
    //!
    //! - `alt_no_pivot`: the elimination without partial pivoting. Cheaper and shorter, and wrong
    //! on a matrix as ordinary as a permuted identity.
    //!
    //! - `alt_recip`: one reciprocal per pivot instead of one correctly rounded division per output
    //! scalar. DEARER for `solve`, where a single right-hand side does not amortise the reciprocal,
    //! cheaper for `try_inverse`, where 3 columns share it, and a second rounding per output in
    //! both.
    //!
    //! - `alt_solve_columns`: the inverse as 3 calls to `solve`. Bit-identical, dearer.

    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix3::{Matrix3, Matrix3Trait};
    use crate::base::matrix_test_utils::{
        abs_raw, fx, int, m3, max_abs_m3, max_abs_v3, max_ulp_diff3, max_ulp_diff_v3, oracle_tol,
        ulp_diff, v3it, v3t,
    };
    use crate::base::vector3::Vector3;
    use crate::linalg::lu::{
        Perm3, PermTrait, oracle_lu3 as oracle, oracle_matrix3_compare as compare,
    };
    use super::{Lu3, Lu3Trait, Matrix3LuTrait};

    /// The oracle's first `unit` 3x3 case whose factorisation actually swaps rows, so every
    /// benchmark exercises the permutation.
    fn a_bench() -> Matrix3<Fixed> {
        m3(
            [
                [-2414118097, 417657389, 4595817171], [6298117444, -2725477524, -1338161353],
                [2614037897, 2690897069, 3051067169],
            ],
        )
    }

    /// Its right-hand side.
    fn b_bench() -> Vector3<Fixed> {
        v3t((7757329492, -3622378744, 4147284176))
    }

    /// `a_bench()` already factored, so that the benchmarks of the derived operations do not pay
    /// for `new`.
    fn f_bench() -> Lu3<Fixed> {
        Lu3 {
            lu: m3(
                [
                    [6298117444, -2725477524, -1338161353], [1782629075, 3822108354, 3606471941],
                    [-1646294845, -704614973, 4674552576],
                ],
            ),
            p: Perm3 { p1: 2, p2: 3 },
        }
    }

    /// A matrix whose fixed-point factorisation is EXACT: `L` has at most 2 fractional bits, `U`
    /// integer entries, so no division and no update rounds. Built by `L * U` then a row shuffle,
    /// which partial pivoting undoes exactly because `|l_ik| < 1`.
    fn a_exact() -> Matrix3<Fixed> {
        m3(
            [
                [-3221225472, -13958643712, -2147483648], [-9663676416, 13958643712, 20401094656],
                [-12884901888, 12884901888, 8589934592],
            ],
        )
    }

    /// An exactly singular integer matrix (one row is an integer combination of the others, and
    /// every multiplier of the elimination is dyadic, so the last pivot is exactly zero).
    fn a_singular() -> Matrix3<Fixed> {
        m3(
            [
                [-25769803776, -12884901888, -25769803776], [12884901888, 12884901888, 0],
                [51539607552, 38654705664, 25769803776],
            ],
        )
    }

    /// `a_singular()` already factored (its last pivot is exactly zero), so the benchmark of the
    /// rejection path measures the rejection and not `new`.
    fn f_singular() -> Lu3<Fixed> {
        Lu3 {
            lu: m3(
                [
                    [51539607552, 38654705664, 25769803776],
                    [-2147483648, 6442450944, -12884901888], [1073741824, 2147483648, 0],
                ],
            ),
            p: Perm3 { p1: 3, p2: 3 },
        }
    }

    /// The same elimination WITHOUT partial pivoting: no `abs`, no comparison, no row swap, and the
    /// identity permutation. Kept as evidence (AGENTS.md rule 8): it is cheaper and shorter, and it
    /// is wrong — the pivot of step `k` is whatever sits at `a_kk`, so a matrix as ordinary as a
    /// permuted identity factors with a zero pivot, and nothing bounds `|l_ik|`, which is what
    /// keeps `U` from growing. Upstream has no unpivoted variant either.
    fn new_no_pivot(matrix: Matrix3<Fixed>) -> Lu3<Fixed> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a13 = matrix.m13;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        let mut a23 = matrix.m23;
        let mut a31 = matrix.m31;
        let mut a32 = matrix.m32;
        let mut a33 = matrix.m33;
        if a11 != Real::ZERO {
            let l = a21 / a11;
            let nl = -l;
            a22 = Real::mul_add(nl, a12, a22);
            a23 = Real::mul_add(nl, a13, a23);
            a21 = l;
            let l = a31 / a11;
            let nl = -l;
            a32 = Real::mul_add(nl, a12, a32);
            a33 = Real::mul_add(nl, a13, a33);
            a31 = l;
        }
        if a22 != Real::ZERO {
            let l = a32 / a22;
            let nl = -l;
            a33 = Real::mul_add(nl, a23, a33);
            a32 = l;
        }
        Lu3 {
            lu: Matrix3 {
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
            p: Perm3 { p1: 1, p2: 2 },
        }
    }

    /// `solve` with ONE reciprocal per pivot and 3 multiplications instead of 3 correctly rounded
    /// divisions.
    /// Kept as evidence, and it loses on both counts: a reciprocal (2 190) plus a multiplication (1
    /// 750) is dearer than a division (2 740), and a single right-hand side gives nothing to
    /// amortise it over, so it costs 27 560 against 23 560 gas; and rounding `1 / u_ii` before
    /// using it puts 3 of the oracle cases outside the tolerance against 0
    /// (`test_solve_candidates_error`).
    fn solve_recip(f: Lu3<Fixed>, b: Vector3<Fixed>) -> Option<Vector3<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let r3 = Real::recip(f.lu.m33);
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
        let x3 = y3 * r3;
        let x2 = Real::mul_add(-f.lu.m23, x3, y2) * r2;
        let x1 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), y1), f.lu.m12, x2),
                f.lu.m13,
                x3,
            ),
        )
            * r1;
        Some(Vector3 { x: x1, y: x2, z: x3 })
    }

    /// `try_inverse` as 3 full calls to `solve` on the unit vectors — the obvious route, and the
    /// one upstream takes. Kept as evidence: it gives BIT-IDENTICAL results
    /// (`test_try_inverse_candidates_agree`) for strictly more gas, because it pays the permutation
    /// 3 times and multiplies by the leading zeros of each unit vector.
    fn try_inverse_solve_columns(f: Lu3<Fixed>) -> Option<Matrix3<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let c1 = f.solve(Vector3 { x: int(1), y: int(0), z: int(0) }).unwrap();
        let c2 = f.solve(Vector3 { x: int(0), y: int(1), z: int(0) }).unwrap();
        let c3 = f.solve(Vector3 { x: int(0), y: int(0), z: int(1) }).unwrap();
        Some(
            Matrix3 {
                m11: c1.x,
                m21: c1.y,
                m31: c1.z,
                m12: c2.x,
                m22: c2.y,
                m32: c2.z,
                m13: c3.x,
                m23: c3.y,
                m33: c3.z,
            },
        )
    }

    /// `try_inverse` with ONE reciprocal per pivot instead of one division per entry of the back
    /// substitution. Here the reciprocal IS amortised (3 columns share it), so this is the
    /// candidate the gas argument favours; it is not shipped because it rounds `1 / u_ii` before
    /// using it, which is the second rounding per output scalar that AGENTS.md rule 4 and DESIGN D2
    /// forbid (the same call `Vector3::unscale` makes).
    fn try_inverse_recip(f: Lu3<Fixed>) -> Option<Matrix3<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let r3 = Real::recip(f.lu.m33);
        let y21 = -f.lu.m21;
        let y31 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub(Real::<Fixed>::wide_zero(), f.lu.m31), f.lu.m32, y21,
            ),
        );
        let x31 = y31 * r3;
        let x21 = Real::mul_add(-f.lu.m23, x31, y21) * r2;
        let x11 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(
                    Real::wide_add(Real::<Fixed>::wide_zero(), int(1)), f.lu.m12, x21,
                ),
                f.lu.m13,
                x31,
            ),
        )
            * r1;
        let y32 = -f.lu.m32;
        let x32 = y32 * r3;
        let x22 = Real::mul_add(-f.lu.m23, x32, int(1)) * r2;
        let x12 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x22), f.lu.m13, x32,
            ),
        )
            * r1;
        let x33 = int(1) * r3;
        let x23 = Real::wide_rescale(Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m23, x33))
            * r2;
        let x13 = Real::wide_rescale(
            Real::wide_sub_prod(
                Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x23), f.lu.m13, x33,
            ),
        )
            * r1;
        let mut c11 = x11;
        let mut c12 = x12;
        let mut c13 = x13;
        let mut c21 = x21;
        let mut c22 = x22;
        let mut c23 = x23;
        let mut c31 = x31;
        let mut c32 = x32;
        let mut c33 = x33;
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
        }
        Some(
            Matrix3 {
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
        )
    }

    /// `(cases above the LU tolerance, worst error in ulp)` of a `solve` candidate: 0 = shipped
    /// (one division per pivot), 1 = one reciprocal per pivot, 2 = shipped substitution on the
    /// UNPIVOTED factorisation.
    fn solve_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle::lu3_solve_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let f = if variant == 2 {
                new_no_pivot(m3(a))
            } else {
                Lu3Trait::new(m3(a))
            };
            let got = if variant == 1 {
                solve_recip(f, v3t(b))
            } else {
                f.solve(v3t(b))
            };
            let e = v3t(expected);
            let err = max_ulp_diff_v3(got.unwrap(), e);
            if err > oracle_tol(max_abs_v3(e), tol) {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    /// `(cases above the LU tolerance, worst error in ulp)` of the two 3x3 inverses on the
    /// `matrix3`
    /// oracle vectors: 0 = `Lu3::try_inverse`, 1 = `Matrix3::try_inverse` (cofactors with the
    /// integer pre-scaling).
    fn matrix3_inverse_failures(variant: u8) -> (u32, u128) {
        let mut cases = compare::matrix3_try_inverse_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = if variant == 0 {
                Lu3Trait::new(m3(a)).try_inverse()
            } else {
                m3(a).try_inverse()
            };
            let e = m3(expected);
            let err = max_ulp_diff3(got.unwrap(), e);
            if err > oracle_tol(max_abs_m3(e), tol) {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    /// The same for the determinant: 0 = product of the LU pivots, 1 = `Matrix3::determinant`
    /// (cofactor expansion on fused kernels).
    fn matrix3_determinant_failures(variant: u8) -> (u32, u128) {
        let mut cases = compare::matrix3_determinant_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = if variant == 0 {
                Lu3Trait::new(m3(a)).determinant()
            } else {
                m3(a).determinant()
            };
            let err = ulp_diff(got, fx(expected));
            if err > oracle_tol(abs_raw(fx(expected)), tol) {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    #[test]
    fn test_new_is_exact_on_a_dyadic_matrix() {
        let f = Lu3Trait::new(a_exact());
        let lu = f.lu;
        assert!(
            lu == m3(
                [
                    [-12884901888, 12884901888, 8589934592],
                    [1073741824, -17179869184, -4294967296], [3221225472, -1073741824, 12884901888],
                ],
            ),
        );
        assert!(f.p() == Perm3 { p1: 3, p2: 3 });
        assert!(f.permute_rows(a_exact()) == f.l() * f.u());
        assert!(f.determinant() == fx(154618822656));
    }

    #[test]
    fn test_new_reconstruction_oracle() {
        // `P A == L U` within 16 raw units on the lu3 vectors.
        let mut cases = oracle::lu3_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Lu3Trait::new(m3(a));
            worst = core::cmp::max(worst, max_ulp_diff3(f.permute_rows(m3(a)), f.l() * f.u()));
        }
        assert!(worst == 10, "reconstruction error {worst}");
    }

    #[test]
    fn test_new_multipliers_are_bounded_by_one() {
        // What partial pivoting buys: |l_ik| <= 1, which bounds the growth of `U`.
        let mut cases = oracle::lu3_solve_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            assert!(max_abs_m3(Lu3Trait::new(m3(a)).l()) <= 0x100000000);
        }
    }

    #[test]
    fn test_factors_permutation_and_accessors() {
        let f = Lu3Trait::new(a_exact());
        let l = f.l();
        assert!(
            l == m3(
                [
                    [4294967296, 0, 0], [1073741824, 4294967296, 0],
                    [3221225472, -1073741824, 4294967296],
                ],
            ),
        );
        let u = f.u();
        assert!(
            u == m3(
                [
                    [-12884901888, 12884901888, 8589934592], [0, -17179869184, -4294967296],
                    [0, 0, 12884901888],
                ],
            ),
        );
        let pa = f.permute_rows(a_exact());
        assert!(
            pa == m3(
                [
                    [-12884901888, 12884901888, 8589934592],
                    [-3221225472, -13958643712, -2147483648],
                    [-9663676416, 13958643712, 20401094656],
                ],
            ),
        );
        assert!(f.permute(v3it((1, 2, 3))) == v3it((3, 1, 2)));
        // the identity factors without a single swap
        let id = Lu3Trait::new(Matrix3Trait::<Fixed>::identity());
        assert!(id.p() == PermTrait::identity3());
        assert!(id.permute(b_bench()) == b_bench());
        assert!(id.permute_rows(a_bench()) == a_bench());
        assert!(id.l() == Matrix3Trait::<Fixed>::identity());
        assert!(id.u() == Matrix3Trait::<Fixed>::identity());
    }

    #[test]
    fn test_singular_is_rejected() {
        let f = Lu3Trait::new(a_singular());
        assert!(!f.is_invertible());
        assert!(f.solve(b_bench()).is_none());
        assert!(f.try_inverse().is_none());
        assert!(f.determinant() == int(0));
        let z = Lu3Trait::new(Matrix3Trait::<Fixed>::zeros());
        assert!(!z.is_invertible());
        assert!(z.try_inverse().is_none());
        assert!(z.determinant() == int(0));
        assert!(z.p() == PermTrait::identity3());
        assert!(Lu3Trait::new(a_bench()).is_invertible());
    }

    #[test]
    fn test_solve_oracle() {
        let mut cases = oracle::lu3_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let x = Lu3Trait::new(m3(a)).solve(v3t(b)).unwrap();
            let e = v3t(expected);
            let err = max_ulp_diff_v3(x, e);
            assert!(err <= oracle_tol(max_abs_v3(e), tol), "solve error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 436);
    }

    #[test]
    fn test_solve_near_singular_oracle() {
        // Condition number 1e2..1e4: the oracle flags these as behaviour, not
        // precision, tests. The tolerances are large by construction.
        let mut cases = oracle::lu3_solve_near_singular_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let x = Lu3Trait::new(m3(a)).solve(v3t(b)).unwrap();
            let e = v3t(expected);
            let err = max_ulp_diff_v3(x, e);
            assert!(err <= oracle_tol(max_abs_v3(e), tol), "solve error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 7603197);
    }

    #[test]
    fn test_try_inverse_oracle() {
        let mut cases = oracle::lu3_inverse_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let inv = Lu3Trait::new(m3(a)).try_inverse().unwrap();
            let e = m3(expected);
            let err = max_ulp_diff3(inv, e);
            assert!(err <= oracle_tol(max_abs_m3(e), tol), "inverse error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 1639);
    }

    #[test]
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle::lu3_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let inv = Lu3Trait::new(m3(a)).try_inverse().unwrap();
            // Worst residual over the oracle: 142 ulp.
            assert!((m3(a) * inv).is_identity(142));
            assert!((inv * m3(a)).is_identity(142));
        }
    }

    #[test]
    fn test_try_inverse_candidates() {
        // The static unit right-hand sides give BIT-IDENTICAL results to 3 calls
        // to `solve`; the saving is pure gas.
        let mut cases = oracle::lu3_inverse_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let f = Lu3Trait::new(m3(a));
            assert!(f.try_inverse().unwrap() == try_inverse_solve_columns(f).unwrap());
            worst =
                core::cmp::max(
                    worst, max_ulp_diff3(f.try_inverse().unwrap(), try_inverse_recip(f).unwrap()),
                );
        }
        // ... and the reciprocal variant drifts by at most this many ulp from it.
        assert!(worst == 14);
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_overflow_panics() {
        // 2^-32 * I: every pivot is 1 raw unit, so the inverse is 2^32 * I.
        let _ = black_box(Matrix3Trait::from_diagonal_element(fx(1))).lu().try_inverse();
    }

    #[test]
    fn test_determinant_oracle() {
        let mut cases = oracle::lu3_determinant_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let det = Lu3Trait::new(m3(a)).determinant();
            let err = ulp_diff(det, fx(expected));
            assert!(err <= oracle_tol(abs_raw(fx(expected)), tol), "determinant error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 326241);
    }

    #[test]
    fn test_determinant_exact_and_sign() {
        assert!(Lu3Trait::new(Matrix3Trait::<Fixed>::identity()).determinant() == int(1));
        let d = Matrix3Trait::from_diagonal(v3it((2, -4, 3)));
        assert!(Lu3Trait::new(d).determinant() == int(-24));
        // Swapping two rows flips the sign exactly (the factorisation is exact here,
        // and the sign is applied to the first pivot before any rounding).
        let s = m3(
            [
                [-9663676416, 13958643712, 20401094656], [-3221225472, -13958643712, -2147483648],
                [-12884901888, 12884901888, 8589934592],
            ],
        );
        assert!(Lu3Trait::new(s).determinant() == -fx(154618822656));
    }

    #[test]
    fn test_solve_candidates_error() {
        // Oracle, 18 well-conditioned matrices: (cases above the LU tolerance,
        // worst error in ulp) of the shipped divisions, of one reciprocal per
        // pivot, and of the shipped substitution on an UNPIVOTED factorisation.
        // The unpivoted figure is NOT a win: these matrices are random and
        // well-conditioned, so their leading entries happen to be usable pivots.
        // `test_no_pivot_candidate_is_wrong` shows the structural failure.
        assert!(solve_failures(0) == (0, 436));
        assert!(solve_failures(1) == (3, 374));
        assert!(solve_failures(2) == (0, 447));
    }

    #[test]
    fn test_no_pivot_candidate_is_wrong() {
        // A permuted identity: the unpivoted elimination finds a zero at `a11` and
        // gives up, where partial pivoting factors it exactly.
        let swapped = m3([[0, 4294967296, 0], [4294967296, 0, 0], [0, 0, 4294967296]]);
        assert!(!new_no_pivot(swapped).is_invertible());
        let f = Lu3Trait::new(swapped);
        assert!(f.is_invertible());
        assert!(f.determinant() == int(-1));
    }

    #[test]
    fn test_try_inverse_versus_matrix3_cofactors() {
        // Same 18 `matrix3_try_inverse` / `matrix3_determinant` oracle cases, same
        // tolerance predicate. The closed form wins on accuracy for a 3x3: its
        // determinant is a sum of exact 2x2 minors, where the LU determinant is a
        // product of pivots that already carry the rounding of the elimination.
        assert!(matrix3_inverse_failures(0) == (0, 64));
        assert!(matrix3_inverse_failures(1) == (0, 27));
        assert!(matrix3_determinant_failures(0) == (2, 27731535));
        assert!(matrix3_determinant_failures(1) == (0, 556));
    }

    // --- gas benchmarks --------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_lu3_new__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(
            m3(
                [
                    [6298117444, -2725477524, -1338161353], [1782629075, 3822108354, 3606471941],
                    [-1646294845, -704614973, 4674552576],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_new__pivot() {
        let a = black_box(a_bench());
        let e = black_box(
            m3(
                [
                    [6298117444, -2725477524, -1338161353], [1782629076, 3822108355, 3606471942],
                    [-1646294844, -704614971, 4674552574],
                ],
            ),
        );
        assert!(Lu3Trait::new(a).lu == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_new__alt_no_pivot() {
        let a = black_box(a_bench());
        let e = black_box(
            m3(
                [
                    [-2414118097, 417657389, 4595817171], [-11205006284, -1635864183, 10651722790],
                    [-4650645423, -8252330158, 28493647233],
                ],
            ),
        );
        assert!(new_no_pivot(a).lu == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_factors__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(
            m3(
                [
                    [4294967296, 0, 0], [1782629075, 4294967296, 0],
                    [-1646294845, -704614973, 4294967296],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_factors__l() {
        let f = black_box(f_bench());
        let e = black_box(
            m3(
                [
                    [4294967296, 0, 0], [1782629075, 4294967296, 0],
                    [-1646294845, -704614973, 4294967296],
                ],
            ),
        );
        assert!(f.l() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_factors__u() {
        let f = black_box(f_bench());
        let e = black_box(
            m3(
                [
                    [6298117444, -2725477524, -1338161353], [0, 3822108354, 3606471941],
                    [0, 0, 4674552576],
                ],
            ),
        );
        assert!(f.u() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_p__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(Perm3 { p1: 2, p2: 3 });
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_p__field() {
        let f = black_box(f_bench());
        let e = black_box(Perm3 { p1: 2, p2: 3 });
        assert!(f.p() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_permute__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(v3t((-3622378744, 4147284176, 7757329492)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_permute__transpositions() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(v3t((-3622378744, 4147284176, 7757329492)));
        assert!(f.permute(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_permute_rows__baseline() {
        let _f = black_box(f_bench());
        let _a = black_box(a_bench());
        let e = black_box(
            m3(
                [
                    [6298117444, -2725477524, -1338161353], [2614037897, 2690897069, 3051067169],
                    [-2414118097, 417657389, 4595817171],
                ],
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_permute_rows__transpositions() {
        let f = black_box(f_bench());
        let a = black_box(a_bench());
        let e = black_box(
            m3(
                [
                    [6298117444, -2725477524, -1338161353], [2614037897, 2690897069, 3051067169],
                    [-2414118097, 417657389, 4595817171],
                ],
            ),
        );
        assert!(f.permute_rows(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_is_invertible__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_is_invertible__pivots() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.is_invertible() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_solve__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(Some(v3t((-1035334030, 24604680, 6703439320))));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_solve__substitution() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v3t((-1035334029, 24604680, 6703439320))));
        assert!(f.solve(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_solve__alt_recip() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v3t((-1035334030, 24604680, 6703439320))));
        assert!(solve_recip(f, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_solve_singular__baseline() {
        let _s = black_box(f_singular());
        let _b = black_box(b_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_solve_singular__none() {
        let s = black_box(f_singular());
        let b = black_box(b_bench());
        let e = black_box(true);
        assert!(s.solve(b).is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_try_inverse__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(
            Some(
                m3(
                    [
                        [-772904016, 1818436374, 1961768289],
                        [-3723567548, -3176899586, 4215453383], [3946205283, 1243908434, 647398487],
                    ],
                ),
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_try_inverse__columns() {
        let f = black_box(f_bench());
        let e = black_box(
            Some(
                m3(
                    [
                        [-772904015, 1818436373, 1961768289],
                        [-3723567548, -3176899587, 4215453383], [3946205284, 1243908435, 647398487],
                    ],
                ),
            ),
        );
        assert!(f.try_inverse() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_try_inverse__alt_solve_columns() {
        let f = black_box(f_bench());
        let e = black_box(
            Some(
                m3(
                    [
                        [-772904015, 1818436373, 1961768289],
                        [-3723567548, -3176899587, 4215453383], [3946205284, 1243908435, 647398487],
                    ],
                ),
            ),
        );
        assert!(try_inverse_solve_columns(f) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_try_inverse__alt_recip() {
        let f = black_box(f_bench());
        let e = black_box(
            Some(
                m3(
                    [
                        [-772904016, 1818436374, 1961768288],
                        [-3723567548, -3176899586, 4215453382], [3946205284, 1243908434, 647398487],
                    ],
                ),
            ),
        );
        assert!(try_inverse_recip(f) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_determinant__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(fx(6100059567));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_determinant__pivots() {
        let f = black_box(f_bench());
        let e = black_box(fx(6100059567));
        assert!(f.determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_vs_matrix3_inverse__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(
            Some(
                m3(
                    [
                        [-772904016, 1818436374, 1961768289],
                        [-3723567548, -3176899586, 4215453383], [3946205283, 1243908434, 647398487],
                    ],
                ),
            ),
        );
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_vs_matrix3_inverse__lu() {
        let a = black_box(a_bench());
        let e = black_box(
            Some(
                m3(
                    [
                        [-772904016, 1818436373, 1961768289],
                        [-3723567549, -3176899587, 4215453384], [3946205285, 1243908435, 647398485],
                    ],
                ),
            ),
        );
        assert!(a.lu().try_inverse() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_vs_matrix3_inverse__cofactors() {
        let a = black_box(a_bench());
        let e = black_box(
            Some(
                m3(
                    [
                        [-772904015, 1818436374, 1961768289],
                        [-3723567548, -3176899587, 4215453384], [3946205284, 1243908435, 647398485],
                    ],
                ),
            ),
        );
        assert!(a.try_inverse() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_vs_matrix3_determinant__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(fx(6100059567));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_vs_matrix3_determinant__lu() {
        let a = black_box(a_bench());
        let e = black_box(fx(6100059565));
        assert!(a.lu().determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu3_vs_matrix3_determinant__cofactors() {
        let a = black_box(a_bench());
        let e = black_box(fx(6100059569));
        assert!(a.determinant() == e);
    }
}
