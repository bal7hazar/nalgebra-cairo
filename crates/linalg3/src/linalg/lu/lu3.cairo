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

use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_core::linalg::permutation_sequence::PermuteRows;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::linalg::lu::Perm3;
use simba::scalar::Real;
use crate::internal::linalg::lu::lu3::Lu3InternalTrait;

/// The LU factorisation with partial pivoting of a `Matrix3<T>`: `P * A = L * U`.
///
/// `lu` packs both factors (strict lower triangle = `L` without its unit diagonal, upper triangle =
/// `U`) and `p` is the row permutation. Built by `Lu3Trait::new` or `Matrix3LuTrait::lu`. Upstream:
/// `nalgebra::linalg::LU`.
#[derive(Copy, Drop, Serde, Debug)]
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
        if piv != R::zero() {
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
        if piv != R::zero() {
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
            m11: R::one(),
            m21: self.lu.m21,
            m31: self.lu.m31,
            m12: R::zero(),
            m22: R::one(),
            m32: self.lu.m32,
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
        }
    }

    /// The upper triangular factor `U`. Exact: moves only. Upstream: `LU::u`.
    #[inline(always)]
    fn u(self: Lu3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.lu.m11,
            m21: R::zero(),
            m31: R::zero(),
            m12: self.lu.m12,
            m22: self.lu.m22,
            m32: R::zero(),
            m13: self.lu.m13,
            m23: self.lu.m23,
            m33: self.lu.m33,
        }
    }

    /// The packed factors as the factorisation stores them: `L` (strict lower triangle, unit
    /// diagonal implicit) and `U` (upper triangle) in one matrix. Exact: a move. Upstream:
    /// `LU::lu_internal` (`#[doc(hidden)]`).
    #[inline(always)]
    fn lu_internal(self: Lu3<T>) -> Matrix3<T> {
        self.lu
    }

    /// The unit lower triangular factor `L`, consuming the factorisation: `l()` (exact, moves
    /// only). Upstream: `LU::l_unpack`.
    #[inline(always)]
    fn l_unpack(self: Lu3<T>) -> Matrix3<T> {
        Self::l(self)
    }

    /// The three factors `(P, L, U)` of `P * A = L * U`: `(p(), l(), u())`, exact. Upstream:
    /// `LU::unpack`.
    #[inline(always)]
    fn unpack(self: Lu3<T>) -> (Perm3, Matrix3<T>, Matrix3<T>) {
        (self.p, Self::l(self), Self::u(self))
    }

    /// Overwrites `b` (any shape with 3 rows: a vector or a matrix) with the solution `x` of `A *
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
    fn solve_mut<B, impl P: PermuteRows<Perm3, B>, impl K: SolveKernel<Matrix3<T>, B>, +Drop<B>>(
        self: Lu3<T>, ref b: B,
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
    fn try_inverse_to(self: Lu3<T>, ref out: Matrix3<T>) -> bool {
        match Self::try_inverse(self) {
            Option::Some(m) => {
                out = m;
                true
            },
            Option::None => false,
        }
    }

    /// The row permutation `P`, as the compact sequence of 2 transpositions `Perm3` — never as a
    /// matrix, and never as an array (DESIGN D4). Upstream: `LU::p`, which returns a heap-allocated
    /// `PermutationSequence`.
    #[inline(always)]
    fn p(self: Lu3<T>) -> Perm3 {
        self.p
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
        self.lu.m11 != R::zero() && self.lu.m22 != R::zero() && self.lu.m33 != R::zero()
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
    #[inline(always)]
    fn solve(self: Lu3<T>, b: Vector3<T>) -> Option<Vector3<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let pb = Lu3InternalTrait::permute(self, b);
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
        let n22 = R::mul_add(-self.lu.m23, x32, R::one());
        let n23 = R::wide_rescale(R::wide_sub_prod(R::wide_zero(), self.lu.m23, x33));
        let (x21, x22, x23) = R::div3(n21, n22, n23, self.lu.m22);
        let n11 = R::wide_rescale(
            R::wide_sub_prod(
                R::wide_sub_prod(R::wide_add(R::wide_zero(), R::one()), self.lu.m12, x21),
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
    use fixed::Fixed;
    use nalgebra_types3::base::matrix3::Matrix3;
    use nalgebra_types3::base::vector3::Vector3;
    use nalgebra_types3::linalg::lu::Perm3;
    use simba::scalar::FixedReal as R;
    use crate::internal::linalg::lu::lu3::Lu3InternalTrait;
    use super::{Lu3, Lu3Trait, Matrix3LuTrait};

    /// `Lu3Trait::solve` before WP 11-OPT-2 (the same body, called, not inlined).
    fn solve_reference(self: Lu3<Fixed>, b: Vector3<Fixed>) -> Option<Vector3<Fixed>> {
        if !Lu3Trait::is_invertible(self) {
            return None;
        }
        let pb = Lu3InternalTrait::permute(self, b);
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

    /// `solve` against the reference, bit for bit: edge cases (the factor of the identity, of the
    /// probe's matrix and of a singular matrix; zero, 1-ulp and large right-hand sides) and a
    /// deterministic sweep of hand-assembled factors (every permutation, pivots of either sign,
    /// zero on one draw in 16, entries from 2^-24 to 4) with right-hand sides of every magnitude.
    #[test]
    fn test_solve_matches_reference() {
        let one: i64 = 0x100000000;
        let id = Matrix3 {
            m11: fx(one),
            m21: fx(0),
            m31: fx(0),
            m12: fx(0),
            m22: fx(one),
            m32: fx(0),
            m13: fx(0),
            m23: fx(0),
            m33: fx(one),
        };
        let a = Matrix3 {
            m11: fx(-2414118097),
            m12: fx(417657389),
            m13: fx(4595817171),
            m21: fx(6298117444),
            m22: fx(-2725477524),
            m23: fx(-1338161353),
            m31: fx(2614037897),
            m32: fx(2690897069),
            m33: fx(3051067169),
        };
        let sing = Matrix3 {
            m11: fx(one),
            m21: fx(2 * one),
            m31: fx(0),
            m12: fx(2 * one),
            m22: fx(4 * one),
            m32: fx(0),
            m13: fx(0),
            m23: fx(0),
            m33: fx(one),
        };
        let mut cases: Array<(Lu3<Fixed>, Vector3<Fixed>)> = array![];
        let rhs = array![
            Vector3 { x: fx(0), y: fx(0), z: fx(0) }, Vector3 { x: fx(1), y: fx(-1), z: fx(1) },
            Vector3 { x: fx(7757329492), y: fx(-3622378744), z: fx(4147284176) },
            Vector3 { x: fx(0x40000000000000), y: fx(-0x40000000000000), z: fx(0x40000000000000) },
        ];
        for b in rhs.span() {
            cases.append((id.lu(), *b));
            cases.append((a.lu(), *b));
            cases.append((sing.lu(), *b));
        }
        let mut state: u128 = 0x1053;
        for k in 0..240_u32 {
            let u: u128 = match k % 3 {
                0 => 0x100,
                1 => 0x100000000,
                _ => 0x400000000,
            };
            let lu = Matrix3 {
                m11: fx(pivot(ref state, k)),
                m12: fx(draw(ref state, u)),
                m13: fx(draw(ref state, u)),
                m21: fx(draw(ref state, u)),
                m22: fx(pivot(ref state, k)),
                m23: fx(draw(ref state, u)),
                m31: fx(draw(ref state, u)),
                m32: fx(draw(ref state, u)),
                m33: fx(pivot(ref state, k)),
            };
            let p = Perm3 { p1: row(ref state, 1, 3), p2: row(ref state, 2, 3) };
            let bb: u128 = match k % 4 {
                0 => 0x100,
                1 => 0x100000000,
                2 => 0x10000000000,
                _ => 0x100000000000,
            };
            let v = Vector3 {
                x: fx(draw(ref state, bb)), y: fx(draw(ref state, bb)), z: fx(draw(ref state, bb)),
            };
            cases.append((Lu3 { lu, p }, v));
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
