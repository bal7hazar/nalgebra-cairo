//! `Qr2`: the QR factorisation of a `Matrix2` (upstream `nalgebra::linalg::QR` on a 2x2 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`.

use simba::scalar::Real;
use crate::base::MatrixTrMul;
use crate::base::matrix2::{Matrix2, Matrix2Trait};
use crate::base::vector2::Vector2;

/// The QR factorisation of a `Matrix2<T>`: `A = Q * R`.
///
/// `q` is orthonormal (a rotation or a reflection) and `r` is upper triangular with
/// `r_ii >= 0` and an exactly zero strict lower triangle. Built by `Qr2Trait::new` or
/// `Matrix2QrTrait::qr`. Upstream: `nalgebra::linalg::QR`, which packs the Householder reflectors
/// and the signed diagonal instead and rebuilds the two factors on demand.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Qr2<T> {
    /// The orthonormal factor `Q` (orthonormal iff `is_invertible`, see the module doc).
    pub q: Matrix2<T>,
    /// The upper triangular factor `R`, `r_ii >= 0`.
    pub r: Matrix2<T>,
}

/// Test-only field-wise equality (upstream `Qr2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Qr2PartialEq<T, +PartialEq<T>> of PartialEq<Qr2<T>> {
    fn eq(lhs: @Qr2<T>, rhs: @Qr2<T>) -> bool {
        lhs.q == rhs.q && lhs.r == rhs.r
    }
}

/// Methods of `Qr2<T>` for any `Real` scalar.
#[generate_trait]
pub impl Qr2Impl<
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
> of Qr2Trait<T> {
    /// The QR factorisation of `matrix` by modified Gram-Schmidt, fully unrolled. Upstream:
    /// `matrix.qr()` / `QR::new(matrix)` (Householder, same unpacked factors).
    ///
    /// ```text
    /// r11 = |a1|            q1 = a1 / r11
    /// r12 = <q1, a2>        w  = a2 - r12 q1
    /// r22 = |w|             q2 = w / r22
    /// ```
    ///
    /// Always succeeds. `r11` and `r22` are floored `norm2`, whose sum of squares is accumulated
    /// unscaled, so no intermediate can overflow and each norm is the exact floor of the true one;
    /// `r12` is one fused `sum_prod2`, `w` two fused `mul_add`, and each component of `q1` / `q2`
    /// one correctly rounded division. Five roundings per column.
    ///
    /// A column that is exactly zero leaves `r_ii = 0` and sets `q_i = 0` rather than completing
    /// the basis (module doc): `Q * R = A` still holds exactly, and `is_invertible` is false.
    ///
    /// **Measured** on the 30 vectors of the `qr2_q_r` oracle suite: the factors agree with
    /// upstream's unpacked Householder factors within **25 ulp** (`Q`) and **78 ulp** (`R`),
    /// every case inside the oracle tolerance; `|A - Q R| <= 2 ulp * max(1, max |a_ij|)` and
    /// `|QᵀQ - I| <= 54 ulp` (dominated by the cases whose `r22` is small — the suite
    /// goes down to 0.036 — since `q_2 = w / r22` inherits the rounding of `w` divided by it).
    ///
    /// Panics with the scalar's overflow error if a norm does not fit (a component of `q` cannot:
    /// it is bounded by 1).
    fn new(matrix: Matrix2<T>) -> Qr2<T> {
        let r11 = R::norm2(matrix.m11, matrix.m21);
        let (q11, q21) = if r11 == R::zero() {
            (R::zero(), R::zero())
        } else {
            (R::div(matrix.m11, r11), R::div(matrix.m21, r11))
        };
        let r12 = R::sum_prod2(q11, matrix.m12, q21, matrix.m22);
        let w1 = R::mul_add(-r12, q11, matrix.m12);
        let w2 = R::mul_add(-r12, q21, matrix.m22);
        let r22 = R::norm2(w1, w2);
        let (q12, q22) = if r22 == R::zero() {
            (R::zero(), R::zero())
        } else {
            (R::div(w1, r22), R::div(w2, r22))
        };
        Qr2 {
            q: Matrix2 { m11: q11, m21: q21, m12: q12, m22: q22 },
            r: Matrix2 { m11: r11, m21: R::zero(), m12: r12, m22: r22 },
        }
    }

    /// The orthonormal factor `Q`. Exact: a move. Upstream: `QR::q`, which rebuilds it from the
    /// stored reflectors.
    #[inline(always)]
    fn q(self: Qr2<T>) -> Matrix2<T> {
        self.q
    }

    /// The upper triangular factor `R`. Exact: a move. Upstream: `QR::r`.
    #[inline(always)]
    fn r(self: Qr2<T>) -> Matrix2<T> {
        self.r
    }

    /// `(Q, R)`. Exact: moves. Upstream: `QR::unpack`.
    #[inline(always)]
    fn unpack(self: Qr2<T>) -> (Matrix2<T>, Matrix2<T>) {
        (self.q, self.r)
    }

    /// Whether the factorisation is invertible: all 2 diagonal entries of `R` are EXACTLY nonzero
    /// (no epsilon), like upstream's `QR::is_invertible`. Equivalently, `Q` is orthonormal.
    #[inline(always)]
    fn is_invertible(self: Qr2<T>) -> bool {
        self.r.m11 != R::zero() && self.r.m22 != R::zero()
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a diagonal entry of `R`
    /// is exactly zero (`is_invertible`). Upstream: `QR::solve`.
    ///
    /// `x = R^-1 (Qᵀ b)`: one fused `tr_mul` for `Qᵀ b` (one rounding per component), then
    /// back substitution, each component costing one fused numerator and one correctly rounded
    /// division.
    /// Panics with the scalar's overflow error if a component of `x` does not fit.
    fn solve(self: Qr2<T>, b: Vector2<T>) -> Option<Vector2<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y = self.q.tr_mul(b);
        let x2 = R::div(y.y, self.r.m22);
        let x1 = R::div(R::mul_add(-self.r.m12, x2, y.x), self.r.m11);
        Some(Vector2 { x: x1, y: x2 })
    }

    /// The inverse, or `None` when a diagonal entry of `R` is exactly zero (`is_invertible`).
    /// Upstream: `QR::try_inverse`.
    ///
    /// `A^-1 = R^-1 Qᵀ`: the transpose is free (moves), and the back substitution runs on the two
    /// columns of `Qᵀ` at once. Two roundings per entry (the fused numerator, then the floor
    /// division by the pivot); a reciprocal per pivot would amortise over the 2 columns and is not
    /// used, for the reason `Lu2::try_inverse` documents (a second rounding per output scalar).
    /// Panics with the scalar's overflow error if an entry does not fit.
    fn try_inverse(self: Qr2<T>) -> Option<Matrix2<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        // Column j of the result solves `R x = (Qᵀ)_j`, and `(Qᵀ)_j` is row j of `Q`.
        let x21 = R::div(self.q.m12, self.r.m22);
        let x22 = R::div(self.q.m22, self.r.m22);
        let x11 = R::div(R::mul_add(-self.r.m12, x21, self.q.m11), self.r.m11);
        let x12 = R::div(R::mul_add(-self.r.m12, x22, self.q.m21), self.r.m11);
        Some(Matrix2 { m11: x11, m21: x21, m12: x12, m22: x22 })
    }
}

/// Crate-internal kernels of `Qr2<T>` (WP 8.0: the public API is strictly upstream's).
/// `determinant`
/// has no upstream counterpart (upstream's `QR::determinant` is commented out); the tests use it to
/// check the factors.
#[generate_trait]
pub(crate) impl Qr2InternalImpl<
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
> of Qr2InternalTrait<T> {
    /// The determinant: `det(Q) * r11 * r22`. Upstream: `QR::determinant`.
    ///
    /// `R` has a non-negative diagonal, so the sign lives entirely in `det(Q) = ±1`, read off
    /// `Matrix2::determinant(q)` — a single `diff_prod` on an orthonormal matrix, hence `±1`
    /// within the rounding of the factorisation, and its SIGN is what is used. The sign is applied
    /// to the FIRST factor (an exact negation) so that the product stays a floor chain, exactly as
    /// `Lu2::determinant` does.
    ///
    /// Exactly zero for a rank-deficient matrix. Panics with the scalar's overflow error if the
    /// product does not fit.
    ///
    /// PREFER `Matrix2::determinant` when the factorisation is not needed for something else: the
    /// closed form is one exactly-rounded `diff_prod`, where this one multiplies two norms that
    /// already carry the rounding of the orthogonalisation.
    fn determinant(self: Qr2<T>) -> T {
        let d = if self.q.determinant().is_sign_negative() {
            -self.r.m11
        } else {
            self.r.m11
        };
        d * self.r.m22
    }
}

/// `Matrix2` methods that go through the QR factorisation; upstream carries them on the matrix
/// itself. Import `Matrix2QrTrait` to use them.
#[generate_trait]
pub impl Matrix2QrImpl<
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
> of Matrix2QrTrait<T> {
    /// The QR factorisation. Upstream: `Matrix2::qr`.
    #[inline(always)]
    fn qr(self: Matrix2<T>) -> Qr2<T> {
        Qr2Trait::new(self)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Qr2`: the exact cases (identity, diagonal, permutation, rank deficient), the
    //! identities (`Q R = A`, `QᵀQ = I`, `A A^-1 = I`), and the oracle vectors of `tools/oracle`
    //! (upstream nalgebra 0.35 on the same raw inputs, unpacked factors, `r_ii >= 0`).

    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix2::{Matrix2, Matrix2InternalTrait, Matrix2Trait};
    use crate::base::matrix_test_utils::{
        amax_m2, excess, int, m2, max_abs_m2, max_abs_v2, max_ulp_diff2, max_ulp_diff_v2,
        oracle_tol, orthonormality_error_m2, ulp_diff, v2t,
    };
    use crate::base::vector2::Vector2;
    use crate::linalg::qr::oracle_qr2 as oracle;
    use super::{Matrix2QrTrait, Qr2, Qr2InternalTrait, Qr2Trait};

    /// The oracle's first `unit` 2x2 case: the benchmark input.
    fn a_bench() -> Matrix2<Fixed> {
        m2([[-3242700962, 4172136962], [-1063519290, -1993675480]])
    }

    /// Its right-hand side, from `qr2_solve`.
    fn b_bench() -> Vector2<Fixed> {
        v2t((-5886581674, -6536196560))
    }

    /// `a_bench()` already factored, so the benchmarks of the derived operations do not pay for
    /// `new`.
    fn f_bench() -> Qr2<Fixed> {
        Qr2Trait::new(a_bench())
    }

    /// A matrix whose QR is exact in fixed point: the columns `(0, 2)` and `(4, 0)` are orthogonal
    /// and their norms are powers of two, so the normalisation divides exactly.
    fn a_exact() -> Matrix2<Fixed> {
        Matrix2Trait::new(int(0), int(4), int(2), int(0))
    }

    /// A rank-1 matrix whose orthogonalisation is exact: the second column is three times the
    /// first, so `r22` is exactly zero.
    fn a_rank1() -> Matrix2<Fixed> {
        Matrix2Trait::new(int(2), int(6), int(0), int(0))
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_identity_and_diagonal_are_exact() {
        let f = Qr2Trait::new(Matrix2Trait::<Fixed>::identity());
        assert!(f.q() == Matrix2Trait::identity());
        assert!(f.r() == Matrix2Trait::identity());
        assert!(f.determinant() == int(1));
        // A positive diagonal is already its own R.
        let d = Matrix2Trait::new(int(2), int(0), int(0), int(5));
        let f = Qr2Trait::new(d);
        assert!(f.q() == Matrix2Trait::identity());
        assert!(f.r() == d);
        assert!(f.determinant() == int(10));
        // A negative diagonal entry moves its sign into Q (r_ii >= 0).
        let d = Matrix2Trait::new(int(2), int(0), int(0), int(-5));
        let f = Qr2Trait::new(d);
        assert!(f.q() == Matrix2Trait::new(int(1), int(0), int(0), int(-1)));
        assert!(f.r() == Matrix2Trait::new(int(2), int(0), int(0), int(5)));
        assert!(f.determinant() == int(-10));
    }

    #[test]
    fn test_new_permutation_is_exact() {
        // The swap matrix: Q is itself, R is the identity, det = -1.
        let p = Matrix2Trait::new(int(0), int(1), int(1), int(0));
        let f = Qr2Trait::new(p);
        assert!(f.q() == p);
        assert!(f.r() == Matrix2Trait::identity());
        assert!(f.determinant() == int(-1));
    }

    #[test]
    fn test_new_orthogonal_columns_are_exact() {
        let f = Qr2Trait::new(a_exact());
        assert!(f.r() == Matrix2Trait::new(int(2), int(0), int(0), int(4)));
        assert!(f.q() * f.r() == a_exact());
        assert!(f.determinant() == int(-8));
        assert!(f.determinant() == a_exact().determinant());
        assert!(orthonormality_error_m2(f.q()) == 0);
    }

    #[test]
    fn test_new_rank_one_leaves_a_zero_column() {
        let f = Qr2Trait::new(a_rank1());
        assert!(f.r().m22 == Real::zero());
        assert!(f.q().column2() == Vector2 { x: Real::zero(), y: Real::zero() });
        // `Q R = A` still holds exactly: row 2 of R is zero.
        assert!(f.q() * f.r() == a_rank1());
        assert!(!f.is_invertible());
        assert!(f.solve(b_bench()).is_none());
        assert!(f.try_inverse().is_none());
        assert!(f.determinant() == int(0));
    }

    #[test]
    fn test_new_zero_matrix_is_rejected() {
        let f = Qr2Trait::new(Matrix2Trait::<Fixed>::zeros());
        assert!(f.q() == Matrix2Trait::zeros());
        assert!(f.r() == Matrix2Trait::zeros());
        assert!(!f.is_invertible());
        assert!(f.determinant() == int(0));
    }

    #[test]
    fn test_unpack_matches_the_accessors() {
        let f = f_bench();
        let (q, r) = f.unpack();
        assert!(q == f.q() && r == f.r());
        assert!(a_bench().qr() == f);
    }

    // --- oracle --------------------------------------------------------------------------------

    #[test]
    fn test_new_factors_oracle() {
        let mut cases = oracle::qr2_q_r_cases();
        let (mut worst_q, mut worst_r, mut worst_ex) = (0, 0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, q, r, tol) = *case;
            let f = Qr2Trait::new(m2(a));
            let (eq, er) = (m2(q), m2(r));
            let (dq, dr) = (max_ulp_diff2(f.q(), eq), max_ulp_diff2(f.r(), er));
            worst_ex = core::cmp::max(worst_ex, excess(dq, oracle_tol(max_abs_m2(eq), tol)));
            worst_ex = core::cmp::max(worst_ex, excess(dr, oracle_tol(max_abs_m2(er), tol)));
            worst_q = core::cmp::max(worst_q, dq);
            worst_r = core::cmp::max(worst_r, dr);
        }
        // Measured worst cases over the 30 vectors: the MGS factors agree with upstream's
        // unpacked Householder factors entry by entry, no sign flip, and every case stays inside
        // the oracle tolerance (`worst_ex == 0`).
        assert!(
            (worst_q, worst_r, worst_ex) == (25, 15, 0),
            "regressed: {worst_q} {worst_r} {worst_ex}",
        );
    }

    #[test]
    fn test_new_reconstruction_and_orthonormality_oracle() {
        let mut cases = oracle::qr2_q_r_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Qr2Trait::new(m2(a));
            let rec = max_ulp_diff2(f.q() * f.r(), m2(a)) / amax_m2(m2(a));
            let orth = orthonormality_error_m2(f.q());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured: `|A - Q R| <= worst_rec ulp * max(1, max |a_ij|)` and `|QᵀQ - I| <=
        // worst_orth`.
        assert!(worst_rec == 2 && worst_orth == 52, "regressed: {worst_rec} {worst_orth}");
    }

    #[test]
    fn test_solve_oracle() {
        let mut cases = oracle::qr2_solve_cases();
        let (mut worst, mut worst_ex) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let x = Qr2Trait::new(m2(a)).solve(v2t(b)).unwrap();
            let e = v2t(expected);
            let err = max_ulp_diff_v2(x, e);
            worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v2(e), tol)));
            worst = core::cmp::max(worst, err);
        }
        assert!((worst, worst_ex) == (749, 4), "regressed: {worst} {worst_ex}");
    }

    #[test]
    fn test_try_inverse_product_is_identity_oracle() {
        let mut cases = oracle::qr2_q_r_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let inv = Qr2Trait::new(m2(a)).try_inverse().unwrap();
            let e = max_ulp_diff2(m2(a) * inv, Matrix2Trait::identity());
            worst = core::cmp::max(worst, e);
            worst = core::cmp::max(worst, max_ulp_diff2(inv * m2(a), Matrix2Trait::identity()));
        }
        // Measured residual of `A A^-1 - I` and `A^-1 A - I` over the 30 well-conditioned vectors.
        assert!(worst == 47, "regressed: {worst}");
    }

    #[test]
    fn test_determinant_versus_matrix2_cofactors() {
        // The closed form is one exactly-rounded `diff_prod`; this one multiplies two norms that
        // already carry the rounding of the orthogonalisation. The gap is recorded, not bounded by
        // the oracle: the `qr` suite has no determinant op.
        let mut cases = oracle::qr2_q_r_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let err = ulp_diff(Qr2Trait::new(m2(a)).determinant(), m2(a).determinant());
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 519, "regressed: {worst}");
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_overflow_panics() {
        // 2^-32 * I: both diagonal entries of R are 1 raw unit, so the inverse is 2^32 * I.
        let _ = black_box(Matrix2Trait::from_diagonal_element(Fixed { raw: 1 })).qr().try_inverse();
    }

    // --- gas benchmarks --------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_qr2_new__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(int(1));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_new__gram_schmidt() {
        let a = black_box(a_bench());
        let e = black_box(int(1));
        let f = Qr2Trait::new(a);
        assert!(f.r.m11 > Real::zero() && f.r.m22 > Real::zero() && f.r.m21 == Real::zero());
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_factors__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(int(1));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_factors__unpack() {
        let f = black_box(f_bench());
        let e = black_box(int(1));
        let (q, r) = f.unpack();
        assert!(q.m11 != Real::zero() && r.m11 != Real::zero());
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_is_invertible__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_is_invertible__pivots() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.is_invertible() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_solve__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_solve__substitution() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(true);
        assert!(f.solve(b).is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_solve_singular__baseline() {
        let _f = black_box(Qr2Trait::new(a_rank1()));
        let _b = black_box(b_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_solve_singular__none() {
        let f = black_box(Qr2Trait::new(a_rank1()));
        let b = black_box(b_bench());
        let e = black_box(true);
        assert!(f.solve(b).is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_try_inverse__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_try_inverse__substitution() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.try_inverse().is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_determinant__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr2_determinant__diagonal_product() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.determinant() != Real::zero()) == e);
    }
}
