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

use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_core::linalg::lu::Perm2;
use nalgebra_core::linalg::permutation_sequence::PermuteRows;
use simba::scalar::Real;
use crate::internal::linalg::lu::lu2::Lu2InternalTrait;

/// The LU factorisation with partial pivoting of a `Matrix2<T>`: `P * A = L * U`.
///
/// `lu` packs both factors (strict lower triangle = `L` without its unit diagonal, upper triangle =
/// `U`) and `p` is the row permutation. Built by `Lu2Trait::new` or `Matrix2LuTrait::lu`. Upstream:
/// `nalgebra::linalg::LU`.
#[derive(Copy, Drop, Serde, Debug)]
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
    /// `l_ik = a_ik / a_kk` are each one correctly rounded division, and the trailing submatrix is
    /// updated entry by entry with `Real::mul_add(-l_ik, a_kj, a_ij)`: ONE floor rounding and one
    /// overflow check per entry, never the two roundings of `a_ij - l_ik * a_kj`.
    ///
    /// A pivot column that is exactly zero is skipped — no swap, no permutation, zero multipliers
    /// —
    /// exactly like upstream's `continue`, so the division is never reached with a zero divisor.
    ///
    /// Error model: `l_ik` is off by at most 1 ulp (correctly rounded quotient) and every update
    /// floors once, so after step 1 an entry of `U` is within about `k * (1 + |a_kj|)` raw units of
    /// its exact value. Partial pivoting keeps `|l_ik| <= 1`, which is what bounds the growth of
    /// the trailing submatrix. Panics with the scalar's overflow error if an update does not fit.
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
        if piv != R::zero() {
            let l = R::div(a21, a11);
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
        Matrix2 { m11: R::one(), m21: self.lu.m21, m12: R::zero(), m22: R::one() }
    }

    /// The upper triangular factor `U`. Exact: moves only. Upstream: `LU::u`.
    #[inline(always)]
    fn u(self: Lu2<T>) -> Matrix2<T> {
        Matrix2 { m11: self.lu.m11, m21: R::zero(), m12: self.lu.m12, m22: self.lu.m22 }
    }

    /// The packed factors as the factorisation stores them: `L` (strict lower triangle, unit
    /// diagonal implicit) and `U` (upper triangle) in one matrix. Exact: a move. Upstream:
    /// `LU::lu_internal` (`#[doc(hidden)]`).
    #[inline(always)]
    fn lu_internal(self: Lu2<T>) -> Matrix2<T> {
        self.lu
    }

    /// The unit lower triangular factor `L`, consuming the factorisation: `l()` (exact, moves
    /// only). Upstream: `LU::l_unpack`.
    #[inline(always)]
    fn l_unpack(self: Lu2<T>) -> Matrix2<T> {
        Self::l(self)
    }

    /// The three factors `(P, L, U)` of `P * A = L * U`: `(p(), l(), u())`, exact. Upstream:
    /// `LU::unpack`.
    #[inline(always)]
    fn unpack(self: Lu2<T>) -> (Perm2, Matrix2<T>, Matrix2<T>) {
        (self.p, Self::l(self), Self::u(self))
    }

    /// Overwrites `b` (any shape with 2 rows: a vector or a matrix) with the solution `x` of `A *
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
    fn solve_mut<B, impl P: PermuteRows<Perm2, B>, impl K: SolveKernel<Matrix2<T>, B>, +Drop<B>>(
        self: Lu2<T>, ref b: B,
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
    fn try_inverse_to(self: Lu2<T>, ref out: Matrix2<T>) -> bool {
        match Self::try_inverse(self) {
            Option::Some(m) => {
                out = m;
                true
            },
            Option::None => false,
        }
    }

    /// The row permutation `P`, as the compact sequence of 1 transpositions `Perm2` — never as a
    /// matrix, and never as an array (DESIGN D4). Upstream: `LU::p`, which returns a heap-allocated
    /// `PermutationSequence`.
    #[inline(always)]
    fn p(self: Lu2<T>) -> Perm2 {
        self.p
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
        self.lu.m11 != R::zero() && self.lu.m22 != R::zero()
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a pivot is exactly zero
    /// (`is_invertible`). Upstream: `LU::solve`.
    ///
    /// `b` is permuted (exactly), then `L y = P b` is solved by forward substitution and `U x = y`
    /// by back substitution. Each `y_i` costs ONE rounding — the whole sum of products is
    /// accumulated in `Real::Wide` and rescaled once — and each `x_i` costs TWO: the numerator,
    /// then the correctly rounded division by the pivot.
    ///
    /// The 2 correctly rounded divisions are kept rather than 2 reciprocals and 2 multiplications.
    /// That candidate loses on both counts here: a reciprocal plus a multiplication is dearer than
    /// a division and a single right-hand side amortises nothing, and rounding `1 / u_ii` before
    /// using it costs accuracy when `|u_ii| >> 1`. `bench_lu2_solve__alt_recip` and
    /// `test_solve_candidates_error` keep both measurements. `try_inverse` amortises a reciprocal
    /// over 2 columns and would be cheaper with one, and still does not use one (see there).
    ///
    /// Panics with the scalar's overflow error if a component of `x` does not fit.
    fn solve(self: Lu2<T>, b: Vector2<T>) -> Option<Vector2<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let pb = Lu2InternalTrait::permute(self, b);
        let y1 = pb.x;
        let y2 = R::mul_add(-self.lu.m21, y1, pb.y);
        let x2 = R::div(y2, self.lu.m22);
        let x1 = R::div(R::mul_add(-self.lu.m12, x2, y1), self.lu.m11);
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
    /// (the numerator, then the correctly rounded division by the pivot). Panics with the scalar's
    /// overflow error if an entry of the inverse does not fit.
    ///
    /// Unlike `solve`, this one WOULD be cheaper with one reciprocal per pivot, which 2 columns
    /// amortise: 21 830 against 24 240 gas (net, `fixed` 0.3.0). It still divides — upstream's
    /// `solve_mut` divides by the pivot — because `mul(x, recip(u))` rounds twice where `x / u`
    /// rounds once, which is the rule DESIGN D2 and `Vector2::unscale` already follow; the drift is
    /// small but real (3 ulp on the oracle inverses).
    /// `bench_lu2_try_inverse__alt_recip` and `test_try_inverse_candidates` keep the measurement.
    ///
    /// The corner `1 / u_22` is `Real::recip`, bit-identical to `ONE / u_22` (WP 7.2).
    fn try_inverse(self: Lu2<T>) -> Option<Matrix2<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y21 = -self.lu.m21;
        let x22 = R::recip(self.lu.m22);
        let x21 = R::div(y21, self.lu.m22);
        let n11 = R::mul_add(-self.lu.m12, x21, R::one());
        let n12 = R::wide_rescale(R::wide_sub_prod(R::wide_zero(), self.lu.m12, x22));
        let x11 = R::div(n11, self.lu.m11);
        let x12 = R::div(n12, self.lu.m11);
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
