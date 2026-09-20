//! `Lu2`: the LU factorisation with partial pivoting of a `Matrix2` (upstream
//! `nalgebra::linalg::LU` on a 2x2 matrix).
//!
//! `P * A = L * U`, with `L` unit lower triangular, `U` upper triangular and `P` the product of the
//! 1 row transpositions chosen by partial pivoting. Both factors share one `Matrix2` like upstream
//! (strict lower triangle = `L`, whose unit diagonal is implicit; upper triangle = `U`) and the
//! permutation is the compact `Perm2`.
//!
//! Everything is unrolled (DESIGN D4: no loop in static code) and every sum of products goes
//! through a fused `Real` kernel — `mul_add` for a single product, the explicit `Real::Wide`
//! accumulator beyond that — so each output scalar is floored once and range-checked once
//! (AGENTS.md rule 4).
//!
//! Partial pivoting costs 1 `abs` and comparisons plus 1 conditional row swaps (moves only), and
//! every swap duplicates the row it moves, so it also costs Sierra statements. Dropping it would be
//! cheaper and shorter and it is not an option: without it the pivot of step `k` is whatever sits
//! at `a_kk`, nothing bounds `|l_ik|`, and a matrix as ordinary as a permuted identity factors with
//! a zero pivot. `bench_lu2_new__alt_no_pivot` and `test_no_pivot_candidate_is_wrong` keep the
//! measurement and the counter-example. Upstream has no unpivoted variant either, only `LU`
//! (partial pivoting) and `FullPivLU` (complete pivoting).

use simba::scalar::Real;
use crate::base::matrix2::Matrix2;
use crate::base::vector2::Vector2;
use super::Perm2;

/// The LU factorisation with partial pivoting of a `Matrix2<T>`: `P * A = L * U`.
///
/// `lu` packs both factors (strict lower triangle = `L` without its unit diagonal, upper triangle =
/// `U`) and `p` is the row permutation. Built by `Lu2Trait::new` or `Matrix2LuTrait::lu`. Upstream:
/// `nalgebra::linalg::LU`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Lu2<T> {
    /// `L` (strict lower triangle) and `U` (upper triangle) packed in one matrix.
    pub lu: Matrix2<T>,
    /// The row transpositions applied by partial pivoting.
    pub p: Perm2,
}

/// Methods of `Lu2<T>` for any `Real` scalar.
#[generate_trait]
pub impl Lu2Impl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Div<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Lu2Trait<T> {
    /// The LU factorisation of `matrix` with partial pivoting, fully unrolled. Upstream:
    /// `matrix.lu()` / `LU::new(matrix)`.
    ///
    /// Always succeeds, like upstream: a singular matrix simply leaves a zero on the diagonal of
    /// `U` (see `is_invertible`). At step `k`, the row of largest `|a_ik|` among rows `k..2` is
    /// swapped onto the diagonal (the FIRST such row, like upstream's `icamax`), the 1 multipliers
    /// `l_ik = a_ik / a_kk` are each one floor division, and the trailing submatrix is updated
    /// entry by entry with `Real::mul_add(-l_ik, a_kj, a_ij)`: ONE floor rounding and one overflow
    /// check per entry, never the two roundings of `a_ij - l_ik * a_kj`.
    ///
    /// A pivot column that is exactly zero is skipped — no swap, no permutation, zero multipliers
    /// —
    /// exactly like upstream's `continue`, so the division is never reached with a zero divisor.
    ///
    /// Error model: `l_ik` is off by at most 1 ulp (floored quotient) and every update floors once,
    /// so after step 1 an entry of `U` is within about `k * (1 + |a_kj|)` raw units of its exact
    /// value. Partial pivoting keeps `|l_ik| <= 1`, which is what bounds the growth of the trailing
    /// submatrix. Panics with the scalar's overflow error if an update does not fit.
    fn new(matrix: Matrix2<T>) -> Lu2<T> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        // step 1: largest pivot among rows 1..2 of column 1
        let mut p1 = 1_u8;
        let mut piv = R::abs(a11);
        let c = R::abs(a21);
        if c > piv {
            piv = c;
            p1 = 2;
        }
        if p1 == 2 {
            let t = a11;
            a11 = a21;
            a21 = t;
            let t = a12;
            a12 = a22;
            a22 = t;
        }
        if piv != R::ZERO {
            let l = a21 / a11;
            let nl = -l;
            a22 = R::mul_add(nl, a12, a22);
            a21 = l;
        }
        Lu2 { lu: Matrix2 { m11: a11, m21: a21, m12: a12, m22: a22 }, p: Perm2 { p1 } }
    }

    /// The unit lower triangular factor `L` (its diagonal of ones is implicit in the packed
    /// storage). Exact: moves only. Upstream: `LU::l`.
    #[inline(always)]
    fn l(self: Lu2<T>) -> Matrix2<T> {
        Matrix2 { m11: R::ONE, m21: self.lu.m21, m12: R::ZERO, m22: R::ONE }
    }

    /// The upper triangular factor `U`. Exact: moves only. Upstream: `LU::u`.
    #[inline(always)]
    fn u(self: Lu2<T>) -> Matrix2<T> {
        Matrix2 { m11: self.lu.m11, m21: R::ZERO, m12: self.lu.m12, m22: self.lu.m22 }
    }

    /// The row permutation `P`, as the compact sequence of 1 transpositions `Perm2` — never as a
    /// matrix, and never as an array (DESIGN D4). Upstream: `LU::p`, which returns a heap-allocated
    /// `PermutationSequence`.
    #[inline(always)]
    fn p(self: Lu2<T>) -> Perm2 {
        self.p
    }

    /// `P * v`: the transpositions applied to the components of `v`, in factorisation order. Exact:
    /// moves and comparisons only. Upstream: `LU::p().permute_rows(&mut v)`.
    fn permute(self: Lu2<T>, v: Vector2<T>) -> Vector2<T> {
        let mut x1 = v.x;
        let mut x2 = v.y;
        if self.p.p1 == 2 {
            let t = x1;
            x1 = x2;
            x2 = t;
        }
        Vector2 { x: x1, y: x2 }
    }

    /// `P * m`: the same transpositions applied to the ROWS of `m`, so that `lu.permute_rows(a)` is
    /// `lu.l() * lu.u()` up to the rounding of the factorisation (this is how the tests check the
    /// decomposition). Exact: moves and comparisons only. Upstream: `LU::p().permute_rows(&mut m)`.
    fn permute_rows(self: Lu2<T>, m: Matrix2<T>) -> Matrix2<T> {
        let mut a11 = m.m11;
        let mut a12 = m.m12;
        let mut a21 = m.m21;
        let mut a22 = m.m22;
        if self.p.p1 == 2 {
            let t = a11;
            a11 = a21;
            a21 = t;
            let t = a12;
            a12 = a22;
            a22 = t;
        }
        Matrix2 { m11: a11, m21: a21, m12: a12, m22: a22 }
    }

    /// Whether the factorisation is invertible: all 2 diagonal entries of `U` are EXACTLY nonzero
    /// (no epsilon), like upstream's `LU::is_invertible`.
    ///
    /// This is NOT the criterion of `Matrix2::try_inverse` (a computed determinant exactly zero),
    /// even though the determinant is the product of these very pivots: a matrix rejected here is
    /// rejected there too, but a matrix with 2 nonzero pivots can still have a determinant that
    /// underflows to zero in the product chain — then `Matrix2::try_inverse` gives `None` while
    /// the factorisation still solves. Neither criterion rejects a merely ill-conditioned matrix:
    /// it is factored and solved with the precision its conditioning allows.
    #[inline(always)]
    fn is_invertible(self: Lu2<T>) -> bool {
        self.lu.m11 != R::ZERO && self.lu.m22 != R::ZERO
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a pivot is exactly zero
    /// (`is_invertible`). Upstream: `LU::solve`.
    ///
    /// `b` is permuted (exactly), then `L y = P b` is solved by forward substitution and `U x = y`
    /// by back substitution. Each `y_i` costs ONE rounding — the whole sum of products is
    /// accumulated in `Real::Wide` and rescaled once — and each `x_i` costs TWO: the numerator,
    /// then the floor division by the pivot.
    ///
    /// The 2 floor divisions are kept rather than 2 reciprocals and 2 multiplications. That
    /// candidate loses on both counts here: a reciprocal plus a multiplication is dearer than a
    /// division and a single right-hand side amortises nothing, and rounding `1 / u_ii` before
    /// using it costs accuracy when `|u_ii| >> 1`. `bench_lu2_solve__alt_recip` and
    /// `test_solve_candidates_error` keep both measurements. `try_inverse` amortises a reciprocal
    /// over 2 columns and would be cheaper with one, and still does not use one (see there).
    ///
    /// Panics with the scalar's overflow error if a component of `x` does not fit.
    fn solve(self: Lu2<T>, b: Vector2<T>) -> Option<Vector2<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let pb = Self::permute(self, b);
        let y1 = pb.x;
        let y2 = R::mul_add(-self.lu.m21, y1, pb.y);
        let x2 = y2 / self.lu.m22;
        let x1 = R::mul_add(-self.lu.m12, x2, y1) / self.lu.m11;
        Some(Vector2 { x: x1, y: x2 })
    }

    /// The inverse, or `None` when a pivot is exactly zero (`is_invertible`). Upstream:
    /// `LU::try_inverse`.
    ///
    /// Computed as `A^-1 = (L U)^-1 P`, NOT as 2 calls to `solve`: the right-hand sides of `L U M =
    /// I` are then the STATIC unit vectors, whose leading zeros disappear at generation time (the
    /// forward substitution is 1 products instead of 2), and the permutation is applied at the end
    /// by swapping the COLUMNS of `M` in reverse factorisation order — moves only, exact.
    ///
    /// Rounding: one per entry of the forward substitution, two per entry of the back substitution
    /// (the numerator, then the floor division by the pivot). Panics with the scalar's overflow
    /// error if an entry of the inverse does not fit.
    ///
    /// Unlike `solve`, this one WOULD be cheaper with one reciprocal per pivot, which 2 columns
    /// amortise: 19 860 against 19 180 gas. It still divides, because `mul(x, recip(u))` rounds
    /// twice where `x / u` rounds once, which is the rule DESIGN D2 and `Vector2::unscale` already
    /// follow; the drift is small but real (2 ulp on the oracle inverses).
    /// `bench_lu2_try_inverse__alt_recip` and `test_try_inverse_candidates` keep the measurement.
    fn try_inverse(self: Lu2<T>) -> Option<Matrix2<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y21 = -self.lu.m21;
        let x21 = y21 / self.lu.m22;
        let x11 = R::mul_add(-self.lu.m12, x21, R::ONE) / self.lu.m11;
        let x22 = R::ONE / self.lu.m22;
        let x12 = R::wide_rescale(R::wide_sub_prod(R::wide_zero(), self.lu.m12, x22)) / self.lu.m11;
        let mut c11 = x11;
        let mut c12 = x12;
        let mut c21 = x21;
        let mut c22 = x22;
        if self.p.p1 == 2 {
            let t = c11;
            c11 = c12;
            c12 = t;
            let t = c21;
            c21 = c22;
            c22 = t;
        }
        Some(Matrix2 { m11: c11, m21: c21, m12: c12, m22: c22 })
    }

    /// The determinant: the product of the 2 pivots, with the sign of the permutation. Upstream:
    /// `LU::determinant`.
    ///
    /// The sign is applied to the FIRST pivot — an exact negation — and the product is then a
    /// left-to-right chain of 1 floored multiplications, so the result stays a floor chain instead
    /// of the negation of one (`-floor(x)` is `ceil(-x)`, one ulp off). `Real` exposes no `Wide *
    /// T`, so a product of 2 scalars cannot be accumulated exactly: each intermediate is floored
    /// once, which adds at most 1 ulp to the error already carried by the pivots.
    ///
    /// Exactly zero for a singular matrix. Panics with the scalar's overflow error if an
    /// intermediate product does not fit; partial pivoting makes the 2 pivots comparable in
    /// magnitude, so the partial products grow monotonically toward the determinant and an
    /// intermediate overflow implies the determinant itself does not fit.
    ///
    /// PREFER `Matrix2::determinant` when the factorisation is not needed for something else: the
    /// closed form sums exact minors, where this one multiplies pivots that already carry the
    /// rounding of the elimination. Measured on the 3x3 oracle vectors, the gap is not subtle —
    /// worst error 556 ulp for the cofactors against 181 307 427 for the pivots, 4 of the 18 cases
    /// outside the oracle tolerance, and 10 660 gas against 41 240
    /// (`bench_lu3_vs_matrix3_determinant` and `test_try_inverse_versus_matrix3_cofactors` in
    /// `lu3`). This method earns its keep on the 6x6, which has no closed form, and whenever the
    /// factorisation is already in hand.
    fn determinant(self: Lu2<T>) -> T {
        let mut neg = false;
        if self.p.p1 != 1 {
            neg = !neg;
        }
        let d = if neg {
            -self.lu.m11
        } else {
            self.lu.m11
        };
        d * self.lu.m22
    }
}

/// `Matrix2` methods that go through the LU factorisation; upstream carries them on the matrix
/// itself. Import `Matrix2LuTrait` to use them.
#[generate_trait]
pub impl Matrix2LuImpl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Div<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Matrix2LuTrait<T> {
    /// The LU factorisation with partial pivoting. Upstream: `Matrix2::lu`.
    #[inline(always)]
    fn lu(self: Matrix2<T>) -> Lu2<T> {
        Lu2Trait::new(self)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Lu2`: an exactly representable factorisation, the identities (`P A = L U`, `A
    //! A^-1 = I`, `|l_ik| <= 1`), the singular cases, the overflow panic and the oracle vectors of
    //! `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
    //!
    //! Tolerance of the oracle assertions: the oracle's `tol` plus ONE relative ulp of the expected
    //! value. `test_utils::lu_tol` states why, with the measurement.
    //!
    //! Gas benchmarks of `Lu2` (`bench_lu2_<op>__<variant>`, net = raw - the `baseline` of the
    //! group), and the alternative implementations that lost, kept as evidence together with the
    //! tests that show why (AGENTS.md rule 8):
    //!
    //! - `alt_no_pivot`: the elimination without partial pivoting. Cheaper and shorter, and wrong
    //! on a matrix as ordinary as a permuted identity.
    //!
    //! - `alt_recip`: one reciprocal per pivot instead of one floor division per output scalar.
    //! DEARER for `solve`, where a single right-hand side does not amortise the reciprocal, cheaper
    //! for `try_inverse`, where 2 columns share it, and a second rounding per output in both.
    //!
    //! - `alt_solve_columns`: the inverse as 2 calls to `solve`. Bit-identical, dearer.

    use nalgebra_testing::black_box;
    use simba::fixed::Fixed;
    use simba::scalar::Real;
    use crate::base::matrix2::{Matrix2, Matrix2Trait};
    use crate::base::vector2::Vector2;
    use crate::linalg::lu::test_utils::{
        abs_raw, fx, int, lu_tol, m2, max_abs_m2, max_abs_v2, max_ulp_diff_m2, max_ulp_diff_v2,
        ulp_diff, v2, v2i,
    };
    use crate::linalg::lu::{Perm2, PermTrait, oracle_lu2 as oracle};
    use super::{Lu2, Lu2Trait, Matrix2LuTrait};

    /// The oracle's first `unit` 2x2 case whose factorisation actually swaps rows, so every
    /// benchmark exercises the permutation.
    fn a_bench() -> Matrix2<Fixed> {
        m2([[-277028774, 4364373136], [3058251633, 1932838604]])
    }

    /// Its right-hand side.
    fn b_bench() -> Vector2<Fixed> {
        v2((-5886581674, -6536196560))
    }

    /// `a_bench()` already factored, so that the benchmarks of the derived operations do not pay
    /// for `new`.
    fn f_bench() -> Lu2<Fixed> {
        Lu2 { lu: m2([[3058251633, 1932838604], [-389055470, 4539457456]]), p: Perm2 { p1: 2 } }
    }

    /// A matrix whose fixed-point factorisation is EXACT: `L` has at most 2 fractional bits, `U`
    /// integer entries, so no division and no update rounds. Built by `L * U` then a row shuffle,
    /// which partial pivoting undoes exactly because `|l_ik| < 1`.
    fn a_exact() -> Matrix2<Fixed> {
        m2([[-3221225472, 16106127360], [12884901888, 4294967296]])
    }

    /// An exactly singular integer matrix (one row is an integer combination of the others, and
    /// every multiplier of the elimination is dyadic, so the last pivot is exactly zero).
    fn a_singular() -> Matrix2<Fixed> {
        m2([[4294967296, 0], [-8589934592, 0]])
    }

    /// `a_singular()` already factored (its last pivot is exactly zero), so the benchmark of the
    /// rejection path measures the rejection and not `new`.
    fn f_singular() -> Lu2<Fixed> {
        Lu2 { lu: m2([[-8589934592, 0], [-2147483648, 0]]), p: Perm2 { p1: 2 } }
    }

    /// The same elimination WITHOUT partial pivoting: no `abs`, no comparison, no row swap, and the
    /// identity permutation. Kept as evidence (AGENTS.md rule 8): it is cheaper and shorter, and it
    /// is wrong — the pivot of step `k` is whatever sits at `a_kk`, so a matrix as ordinary as a
    /// permuted identity factors with a zero pivot, and nothing bounds `|l_ik|`, which is what
    /// keeps `U` from growing. Upstream has no unpivoted variant either.
    fn new_no_pivot(matrix: Matrix2<Fixed>) -> Lu2<Fixed> {
        let mut a11 = matrix.m11;
        let mut a12 = matrix.m12;
        let mut a21 = matrix.m21;
        let mut a22 = matrix.m22;
        if a11 != Real::ZERO {
            let l = a21 / a11;
            let nl = -l;
            a22 = Real::mul_add(nl, a12, a22);
            a21 = l;
        }
        Lu2 { lu: Matrix2 { m11: a11, m21: a21, m12: a12, m22: a22 }, p: Perm2 { p1: 1 } }
    }

    /// `solve` with ONE reciprocal per pivot and 2 multiplications instead of 2 floor divisions.
    /// Kept as evidence, and it loses on both counts: a reciprocal (2 190) plus a multiplication (1
    /// 750) is dearer than a division (2 740), and a single right-hand side gives nothing to
    /// amortise it over, so it costs 16 760 against 14 240 gas; and rounding `1 / u_ii` before
    /// using it puts 4 of the oracle cases outside the tolerance against 0
    /// (`test_solve_candidates_error`).
    fn solve_recip(f: Lu2<Fixed>, b: Vector2<Fixed>) -> Option<Vector2<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let pb = f.permute(b);
        let y1 = pb.x;
        let y2 = Real::mul_add(-f.lu.m21, y1, pb.y);
        let x2 = y2 * r2;
        let x1 = Real::mul_add(-f.lu.m12, x2, y1) * r1;
        Some(Vector2 { x: x1, y: x2 })
    }

    /// `try_inverse` as 2 full calls to `solve` on the unit vectors — the obvious route, and the
    /// one upstream takes. Kept as evidence: it gives BIT-IDENTICAL results
    /// (`test_try_inverse_candidates_agree`) for strictly more gas, because it pays the permutation
    /// 2 times and multiplies by the leading zeros of each unit vector.
    fn try_inverse_solve_columns(f: Lu2<Fixed>) -> Option<Matrix2<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let c1 = f.solve(Vector2 { x: int(1), y: int(0) }).unwrap();
        let c2 = f.solve(Vector2 { x: int(0), y: int(1) }).unwrap();
        Some(Matrix2 { m11: c1.x, m21: c1.y, m12: c2.x, m22: c2.y })
    }

    /// `try_inverse` with ONE reciprocal per pivot instead of one division per entry of the back
    /// substitution. Here the reciprocal IS amortised (2 columns share it), so this is the
    /// candidate the gas argument favours; it is not shipped because it rounds `1 / u_ii` before
    /// using it, which is the second rounding per output scalar that AGENTS.md rule 4 and DESIGN D2
    /// forbid (the same call `Vector2::unscale` makes).
    fn try_inverse_recip(f: Lu2<Fixed>) -> Option<Matrix2<Fixed>> {
        if !f.is_invertible() {
            return None;
        }
        let r1 = Real::recip(f.lu.m11);
        let r2 = Real::recip(f.lu.m22);
        let y21 = -f.lu.m21;
        let x21 = y21 * r2;
        let x11 = Real::mul_add(-f.lu.m12, x21, int(1)) * r1;
        let x22 = int(1) * r2;
        let x12 = Real::wide_rescale(Real::wide_sub_prod(Real::<Fixed>::wide_zero(), f.lu.m12, x22))
            * r1;
        let mut c11 = x11;
        let mut c12 = x12;
        let mut c21 = x21;
        let mut c22 = x22;
        if f.p.p1 == 2 {
            let t = c11;
            c11 = c12;
            c12 = t;
            let t = c21;
            c21 = c22;
            c22 = t;
        }
        Some(Matrix2 { m11: c11, m21: c21, m12: c12, m22: c22 })
    }

    /// `(cases above the LU tolerance, worst error in ulp)` of a `solve` candidate: 0 = shipped
    /// (one division per pivot), 1 = one reciprocal per pivot, 2 = shipped substitution on the
    /// UNPIVOTED factorisation.
    fn solve_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle::lu2_solve_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let f = if variant == 2 {
                new_no_pivot(m2(a))
            } else {
                Lu2Trait::new(m2(a))
            };
            let got = if variant == 1 {
                solve_recip(f, v2(b))
            } else {
                f.solve(v2(b))
            };
            let e = v2(expected);
            let err = max_ulp_diff_v2(got.unwrap(), e);
            if err > lu_tol(max_abs_v2(e), tol) {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    #[test]
    fn test_new_is_exact_on_a_dyadic_matrix() {
        let f = Lu2Trait::new(a_exact());
        let lu = f.lu;
        assert!(lu == m2([[12884901888, 4294967296], [-1073741824, 17179869184]]));
        assert!(f.p() == Perm2 { p1: 2 });
        assert!(f.permute_rows(a_exact()) == f.l() * f.u());
        assert!(f.determinant() == fx(-51539607552));
    }

    #[test]
    fn test_new_reconstruction_oracle() {
        // `P A == L U` within 34 raw units on the lu2 vectors.
        let mut cases = oracle::lu2_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Lu2Trait::new(m2(a));
            worst = core::cmp::max(worst, max_ulp_diff_m2(f.permute_rows(m2(a)), f.l() * f.u()));
        }
        assert!(worst == 34, "reconstruction error {worst}");
    }

    #[test]
    fn test_new_multipliers_are_bounded_by_one() {
        // What partial pivoting buys: |l_ik| <= 1, which bounds the growth of `U`.
        let mut cases = oracle::lu2_solve_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            assert!(max_abs_m2(Lu2Trait::new(m2(a)).l()) <= 0x100000000);
        }
    }

    #[test]
    fn test_factors_permutation_and_accessors() {
        let f = Lu2Trait::new(a_exact());
        let l = f.l();
        assert!(l == m2([[4294967296, 0], [-1073741824, 4294967296]]));
        let u = f.u();
        assert!(u == m2([[12884901888, 4294967296], [0, 17179869184]]));
        let pa = f.permute_rows(a_exact());
        assert!(pa == m2([[12884901888, 4294967296], [-3221225472, 16106127360]]));
        assert!(f.permute(v2i((1, 2))) == v2i((2, 1)));
        // the identity factors without a single swap
        let id = Lu2Trait::new(Matrix2Trait::<Fixed>::identity());
        assert!(id.p() == PermTrait::identity2());
        assert!(id.permute(b_bench()) == b_bench());
        assert!(id.permute_rows(a_bench()) == a_bench());
        assert!(id.l() == Matrix2Trait::<Fixed>::identity());
        assert!(id.u() == Matrix2Trait::<Fixed>::identity());
    }

    #[test]
    fn test_singular_is_rejected() {
        let f = Lu2Trait::new(a_singular());
        assert!(!f.is_invertible());
        assert!(f.solve(b_bench()).is_none());
        assert!(f.try_inverse().is_none());
        assert!(f.determinant() == int(0));
        let z = Lu2Trait::new(Matrix2Trait::<Fixed>::zeros());
        assert!(!z.is_invertible());
        assert!(z.try_inverse().is_none());
        assert!(z.determinant() == int(0));
        assert!(z.p() == PermTrait::identity2());
        assert!(Lu2Trait::new(a_bench()).is_invertible());
    }

    #[test]
    fn test_solve_oracle() {
        let mut cases = oracle::lu2_solve_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let x = Lu2Trait::new(m2(a)).solve(v2(b)).unwrap();
            let e = v2(expected);
            let err = max_ulp_diff_v2(x, e);
            assert!(err <= lu_tol(max_abs_v2(e), tol), "solve error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 122);
    }

    #[test]
    fn test_try_inverse_oracle() {
        let mut cases = oracle::lu2_inverse_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let inv = Lu2Trait::new(m2(a)).try_inverse().unwrap();
            let e = m2(expected);
            let err = max_ulp_diff_m2(inv, e);
            assert!(err <= lu_tol(max_abs_m2(e), tol), "inverse error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 393);
    }

    #[test]
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle::lu2_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let inv = Lu2Trait::new(m2(a)).try_inverse().unwrap();
            // Worst residual over the oracle: 52 ulp.
            assert!((m2(a) * inv).is_identity(52));
            assert!((inv * m2(a)).is_identity(52));
        }
    }

    #[test]
    fn test_try_inverse_candidates() {
        // The static unit right-hand sides give BIT-IDENTICAL results to 2 calls
        // to `solve`; the saving is pure gas.
        let mut cases = oracle::lu2_inverse_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let f = Lu2Trait::new(m2(a));
            assert!(f.try_inverse().unwrap() == try_inverse_solve_columns(f).unwrap());
            worst =
                core::cmp::max(
                    worst, max_ulp_diff_m2(f.try_inverse().unwrap(), try_inverse_recip(f).unwrap()),
                );
        }
        // ... and the reciprocal variant drifts by at most this many ulp from it.
        assert!(worst == 2);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_try_inverse_overflow_panics() {
        // 2^-32 * I: every pivot is 1 raw unit, so the inverse is 2^32 * I.
        let _ = black_box(Matrix2Trait::from_diagonal_element(fx(1))).lu().try_inverse();
    }

    #[test]
    fn test_determinant_oracle() {
        let mut cases = oracle::lu2_determinant_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let det = Lu2Trait::new(m2(a)).determinant();
            let err = ulp_diff(det, fx(expected));
            assert!(err <= lu_tol(abs_raw(fx(expected)), tol), "determinant error {err}");
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 166);
    }

    #[test]
    fn test_determinant_exact_and_sign() {
        assert!(Lu2Trait::new(Matrix2Trait::<Fixed>::identity()).determinant() == int(1));
        let d = Matrix2Trait::from_diagonal(v2i((2, -4)));
        assert!(Lu2Trait::new(d).determinant() == int(-8));
        // Swapping two rows flips the sign exactly (the factorisation is exact here,
        // and the sign is applied to the first pivot before any rounding).
        let s = m2([[12884901888, 4294967296], [-3221225472, 16106127360]]);
        assert!(Lu2Trait::new(s).determinant() == -fx(-51539607552));
    }

    #[test]
    fn test_solve_candidates_error() {
        // Oracle, 18 well-conditioned matrices: (cases above the LU tolerance,
        // worst error in ulp) of the shipped divisions, of one reciprocal per
        // pivot, and of the shipped substitution on an UNPIVOTED factorisation.
        // The unpivoted figure is NOT a win: these matrices are random and
        // well-conditioned, so their leading entries happen to be usable pivots.
        // `test_no_pivot_candidate_is_wrong` shows the structural failure.
        assert!(solve_failures(0) == (0, 122));
        assert!(solve_failures(1) == (4, 767));
        assert!(solve_failures(2) == (0, 80));
    }

    #[test]
    fn test_no_pivot_candidate_is_wrong() {
        // A permuted identity: the unpivoted elimination finds a zero at `a11` and
        // gives up, where partial pivoting factors it exactly.
        let swapped = m2([[0, 4294967296], [4294967296, 0]]);
        assert!(!new_no_pivot(swapped).is_invertible());
        let f = Lu2Trait::new(swapped);
        assert!(f.is_invertible());
        assert!(f.determinant() == int(-1));
    }

    // --- gas benchmarks --------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_lu2_new__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(m2([[3058251633, 1932838604], [-389055470, 4539457456]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_new__pivot() {
        let a = black_box(a_bench());
        let e = black_box(m2([[3058251633, 1932838604], [-389055470, 4539457456]]));
        assert!(Lu2Trait::new(a).lu == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_new__alt_no_pivot() {
        let a = black_box(a_bench());
        let e = black_box(m2([[-277028774, 4364373136], [-47414174915, 50113217406]]));
        assert!(new_no_pivot(a).lu == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_factors__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(m2([[4294967296, 0], [-389055470, 4294967296]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_factors__l() {
        let f = black_box(f_bench());
        let e = black_box(m2([[4294967296, 0], [-389055470, 4294967296]]));
        assert!(f.l() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_factors__u() {
        let f = black_box(f_bench());
        let e = black_box(m2([[3058251633, 1932838604], [0, 4539457456]]));
        assert!(f.u() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_p__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(Perm2 { p1: 2 });
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_p__field() {
        let f = black_box(f_bench());
        let e = black_box(Perm2 { p1: 2 });
        assert!(f.p() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_permute__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(v2((-6536196560, -5886581674)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_permute__transpositions() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(v2((-6536196560, -5886581674)));
        assert!(f.permute(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_permute_rows__baseline() {
        let _f = black_box(f_bench());
        let _a = black_box(a_bench());
        let e = black_box(m2([[3058251633, 1932838604], [-277028774, 4364373136]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_permute_rows__transpositions() {
        let f = black_box(f_bench());
        let a = black_box(a_bench());
        let e = black_box(m2([[3058251633, 1932838604], [-277028774, 4364373136]]));
        assert!(f.permute_rows(a) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_is_invertible__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_is_invertible__pivots() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.is_invertible() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_solve__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(Some(v2((-5305313723, -6129723447))));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_solve__substitution() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v2((-5305313723, -6129723447))));
        assert!(f.solve(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_solve__alt_recip() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(Some(v2((-5305313725, -6129723446))));
        assert!(solve_recip(f, b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_solve_singular__baseline() {
        let _s = black_box(f_singular());
        let _b = black_box(b_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_solve_singular__none() {
        let s = black_box(f_singular());
        let b = black_box(b_bench());
        let e = black_box(true);
        assert!(s.solve(b).is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_try_inverse__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(Some(m2([[-2568255029, 5799151168], [4063645105, 368101372]])));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_try_inverse__columns() {
        let f = black_box(f_bench());
        let e = black_box(Some(m2([[-2568255029, 5799151168], [4063645105, 368101372]])));
        assert!(f.try_inverse() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_try_inverse__alt_solve_columns() {
        let f = black_box(f_bench());
        let e = black_box(Some(m2([[-2568255029, 5799151168], [4063645105, 368101372]])));
        assert!(try_inverse_solve_columns(f) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_try_inverse__alt_recip() {
        let f = black_box(f_bench());
        let e = black_box(Some(m2([[-2568255029, 5799151168], [4063645105, 368101372]])));
        assert!(try_inverse_recip(f) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_determinant__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(fx(-3232342000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_lu2_determinant__pivots() {
        let f = black_box(f_bench());
        let e = black_box(fx(-3232342000));
        assert!(f.determinant() == e);
    }
}
