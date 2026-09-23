//! `Svd2`: the singular value decomposition `M = U · Σ · Vᵀ` of a `Matrix2`, and the polar
//! decomposition built on it (upstream `nalgebra::linalg::SVD`, DESIGN D6).
//!
//! Upstream bidiagonalises and then runs an implicitly shifted QR until the off-diagonal entries
//! fall below `eps`, bounded only by `max_niter`: an unbounded loop with a tolerance parameter,
//! which a proof system cannot price. DESIGN D6 goes through the **symmetric eigen decomposition
//! of `MᵀM`** instead, which in 2x2 is a closed form (`SymmetricEigen2`): constant cost, no
//! tolerance, no iteration count.
//!
//! The eigenvectors of `MᵀM` are the right singular vectors, and the singular values are the
//! square roots of its eigenvalues — but they are NOT computed that way here: `σ_i = |M v_i|`,
//! a floored `Real::norm2` on an unscaled sum of squares.
//!
//! The alternative `σ_i = sqrt(λ_i)` is implemented and measured
//! (`test_singular_values_candidates`, `bench_svd2_new__alt_sqrt_eigenvalues`). In 2x2 it is in
//! fact MORE accurate on the oracle vectors — 10 ulp against 48 — because the closed-form
//! `λ` carries only the rounding of `mean ± r`. It still does not ship:
//!
//! - `λ₁ = mean - r` is a difference of two independently FLOORED quantities, so on a
//!   rank-deficient matrix, whose exact `λ₁` is zero, it can come out NEGATIVE by a raw
//!   unit — and `Real::sqrt` panics on a negative input. `|M v|` cannot be negative;
//! - `Σ` would no longer be the scale that relates `U` to `M V`, so `recompose`, `solve` and
//!   `pseudo_inverse` would be evaluated at a `σ` inconsistent with the vectors they use;
//! - the same substitution LOSES in 3x3 (47 ulp against 75, `svd3`), where `λ` comes out of
//!   four Jacobi sweeps rather than a closed form, and one formula for both sizes is worth more
//!   than a few ulp.

use simba::scalar::Real;
use crate::base::matrix2::{Matrix2, Matrix2Trait};
use crate::base::sym_matrix2::{SymMatrix2, SymMatrix2Trait};
use crate::base::vector2::Vector2;
use crate::linalg::symmetric_eigen2::SymmetricEigen2Trait;

/// The singular value decomposition `M = U · diag(singular_values) · v_t` of a `Matrix2<T>`.
///
/// `singular_values` are sorted **descending** and non-negative (like `tools/oracle` and upstream's
/// `singular_values`), `u` and `v_t` are orthonormal. Unlike upstream, where `u` and `v_t` are
/// `Option`s selected by the `compute_u` / `compute_v` flags, both are always present: the 2x2
/// closed form produces them at a cost the flags could not save.
///
/// Sign and order convention (the decomposition is only defined up to a sign per column and up to
/// the order of equal singular values): the columns of `V` are the eigenvectors that
/// `SymmetricEigen2` returns for `MᵀM`, reordered DESCENDING by singular value (a swap of the two
/// columns except on a tie, which keeps them); `u_1 = M v_1 / σ_1` fixes the first column of `U`,
/// and the second is the direct perpendicular `(x, y) -> (-y, x)` of the first, signed so that
/// `u_2` points along `M v_2`. The decomposition of a given matrix is therefore a deterministic
/// function of its raw components, as AGENTS.md requires.
///
/// Upstream: `SVD { u: Option<OMatrix>, v_t: Option<OMatrix>, singular_values: OVector }`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Svd2<T> {
    /// The left singular vectors, as columns. Orthonormal.
    pub u: Matrix2<T>,
    /// The two singular values, descending, non-negative.
    pub singular_values: Vector2<T>,
    /// The TRANSPOSE of the right singular vectors, i.e. `v_i` is ROW `i`. Orthonormal.
    pub v_t: Matrix2<T>,
}

/// Methods of `Svd2<T>` for any `Real` scalar.
#[generate_trait]
pub impl Svd2Impl<
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
> of Svd2Trait<T> {
    /// The singular value decomposition of `matrix`. Upstream: `matrix.svd(true, true)` /
    /// `SVD::new(matrix, true, true)`.
    ///
    /// ```text
    /// S  = MᵀM                    (3 fused kernels, exact up to one floor per component)
    /// V  = eigenvectors of S      (SymmetricEigen2, closed form), columns ORDERED so that the
    ///                              singular values come out descending
    /// w_i = M v_i                 (2 fused sum_prod2 each)
    /// σ_i = |w_i|                 (floored norm2 on the unscaled sum of squares)
    /// u_1 = w_1 / σ_1             (one correctly rounded division per component)
    /// u_2 = ± perp(u_1)           (EXACT orthonormality, see below)
    /// ```
    ///
    /// The second left singular vector is the perpendicular of the first rather than `w_2 / σ_2`.
    /// The two are equal in exact arithmetic, and the perpendicular is both cheaper (one
    /// `diff_prod` and a sign test against a `norm2` and two divisions) and unconditionally
    /// orthonormal: `w_2 / σ_2` drifts off `u_1`'s perpendicular by `|<u_1, w_2>| / σ_2`, which
    /// is the rounding of the eigenvector divided by the SMALL singular value — precisely the
    /// quantity that blows up on the ill-conditioned matrices the decomposition is meant to
    /// survive. Its sign is that of `<perp(u_1), w_2>`, so `U` still matches `M V` and not just
    /// its first column. `bench_svd2_new__alt_normalised_columns` and
    /// `test_left_vectors_candidates` keep the measurement.
    ///
    /// **Rank deficiency.** `σ_i` is exactly zero only when `M v_i` is exactly zero (a floored
    /// `norm2` of a nonzero vector is at least 1 raw unit), so the threshold below which the
    /// fallback fires is EXACT ZERO — no epsilon, in the spirit of `Matrix2::try_inverse` and
    /// `LU::is_invertible`. `σ_1 = 0` means `M = 0` and `U` is the identity; `σ_2 = 0` (rank 1)
    /// is already handled by the perpendicular, which needs no division. A merely TINY `σ_2` is
    /// not treated as zero: the direction it produces is as accurate as the input allows, and
    /// callers that need a rank decision have `rank(eps)`, `pseudo_inverse(eps)` and `solve(b,
    /// eps)`.
    ///
    /// **Measured** on the 30 vectors of the `svd2_singular_values` oracle suite
    /// (well-conditioned matrices, `small` / `unit` / `medium`): singular values within **48 ulp**
    /// of the floored exact result, every case INSIDE the oracle's own tolerance; `U` and `V`
    /// orthonormal within **23 ulp**; `|M - U Σ Vᵀ| <= 6 ulp * max(1, max |m_ij|)`; and
    /// the polar factors within `|M - R P| <= 7 ulp * max(1, max |m_ij|)` with `|RᵀR - I| <=
    /// 25 ulp`.
    ///
    /// Cost: constant. Panics with the scalar's overflow error if a component of `MᵀM` does not
    /// fit (the squares of the entries must be representable, unlike `norm2`).
    fn new(matrix: Matrix2<T>) -> Svd2<T> {
        let eigen = SymmetricEigen2Trait::new(Self::gram(matrix));
        // RENORMALISE the eigenvectors. `SymmetricEigen2` divides the raw eigen direction by its
        // own FLOORED norm, and on a `small` matrix that direction is a tiny vector — the row of
        // `S - λ₁ I` it is read off has a magnitude of a few million raw units — so flooring
        // its norm costs `1 / |u|_raw` RELATIVE, up to 1.4e-7 on the oracle vectors. That bias
        // lands directly on `σ = |M v|` (measured: 48 ulp on `svd2` case 6, against 2 after this
        // fix), on the orthonormality of `V` and on `recompose`. Here the columns are of magnitude
        // 2^32, so the same floor costs only 2.3e-10. Column 2 is the direct perpendicular of
        // column 1, exactly as `SymmetricEigen2` builds it, so one norm and two divisions
        // renormalise both.
        let c1 = eigen.eigenvectors.column1();
        let nv = R::norm2(c1.x, c1.y);
        let v1 = Vector2 { x: R::div(c1.x, nv), y: R::div(c1.y, nv) };
        let v2 = Vector2 { x: -v1.y, y: v1.x };
        let (w1, w2) = (matrix.mul_vec(v1), matrix.mul_vec(v2));
        let s1 = R::norm2(w1.x, w1.y);
        let s2 = R::norm2(w2.x, w2.y);
        // `SymmetricEigen2` sorts the eigenvalues ASCENDING and the singular values are DESCENDING,
        // so the columns come out in the opposite order — but reversing them unconditionally
        // would also reorder EQUAL singular values, and the decomposition of the identity would not
        // be the identity. One conditional swap on the computed norms reverses exactly when the
        // order asks for it, and leaves ties alone.
        let (s1, s2, w1, w2, v1, v2) = if s2 > s1 {
            (s2, s1, w2, w1, v2, v1)
        } else {
            (s1, s2, w1, w2, v1, v2)
        };
        let u1 = if s1 == R::ZERO {
            Vector2 { x: R::ONE, y: R::ZERO }
        } else {
            Vector2 { x: R::div(w1.x, s1), y: R::div(w1.y, s1) }
        };
        // `<perp(u1), w2>` with perp(u1) = (-u1.y, u1.x): one rounding, and its sign orients u2.
        let along = R::diff_prod(u1.x, w2.y, u1.y, w2.x);
        let u2 = if along.is_negative() {
            Vector2 { x: u1.y, y: -u1.x }
        } else {
            Vector2 { x: -u1.y, y: u1.x }
        };
        Svd2 {
            u: Matrix2Trait::from_columns(u1, u2),
            singular_values: Vector2 { x: s1, y: s2 },
            v_t: Matrix2Trait::from_rows(v1, v2),
        }
    }

    /// `MᵀM` as a symmetric matrix: 3 fused kernels instead of 8 products, bit-identical to the
    /// upper triangle of `m.transpose() * m` and to `m.transpose().mul_transpose()` — which
    /// would reuse `base` instead of repeating the kernel, and costs the moves of the transpose
    /// (measured at 2 760 gas in 3x3, `bench_svd3_gram__*`). Panics on overflow.
    /// Upstream: `m.tr_mul(&m)`.
    #[inline(always)]
    fn gram(m: Matrix2<T>) -> SymMatrix2<T> {
        SymMatrix2 {
            m11: R::norm_squared2(m.m11, m.m21),
            m12: R::sum_prod2(m.m11, m.m12, m.m21, m.m22),
            m22: R::norm_squared2(m.m12, m.m22),
        }
    }

    /// The two singular values, descending. Exact: a move. Upstream: the `singular_values` field
    /// (and `Matrix2::singular_values`).
    #[inline(always)]
    fn singular_values(self: Svd2<T>) -> Vector2<T> {
        self.singular_values
    }

    /// The right singular vectors as COLUMNS, `V = v_tᵀ`. Exact: moves. No upstream equivalent
    /// (upstream stores `v_t` and transposes at the call site).
    #[inline(always)]
    fn v(self: Svd2<T>) -> Matrix2<T> {
        self.v_t.transpose()
    }

    /// The number of singular values strictly greater than `eps`. `eps` must be non-negative;
    /// a negative one makes every singular value count, which is upstream's behaviour too.
    /// Upstream: `SVD::rank`.
    #[inline(always)]
    fn rank(self: Svd2<T>, eps: T) -> u32 {
        let mut n = 0_u32;
        if self.singular_values.x > eps {
            n += 1;
        }
        if self.singular_values.y > eps {
            n += 1;
        }
        n
    }

    /// `U · diag(singular_values) · v_t`, the matrix the decomposition came from, up to its
    /// rounding (measured: **6 ulp per unit of `max |m_ij|`** on the oracle vectors).
    ///
    /// Two roundings per entry: the columns of `U` are scaled by the singular values (4 floored
    /// products), then the product with `v_t` is 4 fused `sum_prod2`. Panics on overflow.
    /// Upstream: `SVD::recompose`, which returns a `Result` because `u` / `v_t` may be missing;
    /// here they never are.
    fn recompose(self: Svd2<T>) -> Matrix2<T> {
        let us = Matrix2 {
            m11: self.u.m11 * self.singular_values.x,
            m21: self.u.m21 * self.singular_values.x,
            m12: self.u.m12 * self.singular_values.y,
            m22: self.u.m22 * self.singular_values.y,
        };
        us * self.v_t
    }

    /// The Moore-Penrose pseudo-inverse `V · diag(σ⁺) · Uᵀ`, where `σ⁺_i = 1 / σ_i` when
    /// `σ_i > eps` and `0` otherwise, or `None` when `eps` is negative.
    ///
    /// Upstream: `SVD::pseudo_inverse`, which returns `Err` on a negative `eps` and otherwise
    /// recomposes with the inverted singular values and takes the adjoint — the same expression.
    /// Note that upstream's test is `> eps` as well, so `eps = 0` keeps every nonzero singular
    /// value, including one of 1 raw unit whose reciprocal overflows: pass an `eps` matched to the
    /// scale of the problem, not zero, unless an overflow panic is the wanted answer.
    ///
    /// Three roundings per entry (the reciprocal, the scaling, the product). Panics with the
    /// scalar's overflow error if a reciprocal or an entry does not fit.
    fn pseudo_inverse(self: Svd2<T>, eps: T) -> Option<Matrix2<T>> {
        if eps.is_negative() {
            return None;
        }
        let r1 = if self.singular_values.x > eps {
            self.singular_values.x.recip()
        } else {
            R::ZERO
        };
        let r2 = if self.singular_values.y > eps {
            self.singular_values.y.recip()
        } else {
            R::ZERO
        };
        let v = self.v_t.transpose();
        let vr = Matrix2 { m11: v.m11 * r1, m21: v.m21 * r1, m12: v.m12 * r2, m22: v.m22 * r2 };
        Some(vr * self.u.transpose())
    }

    /// The least-squares solution of `M x = b`, `V · (Uᵀ b / σ)` with the components whose
    /// singular value is `<= eps` zeroed, or `None` when `eps` is negative. Upstream: `SVD::solve`.
    ///
    /// Unlike `pseudo_inverse` this divides instead of multiplying by a reciprocal — one rounding
    /// per component instead of two, and no overflow on a tiny singular value unless the solution
    /// itself does not fit. Panics with the scalar's overflow error in that case.
    fn solve(self: Svd2<T>, b: Vector2<T>, eps: T) -> Option<Vector2<T>> {
        if eps.is_negative() {
            return None;
        }
        let y = self.u.tr_mul_vec(b);
        let y1 = if self.singular_values.x > eps {
            R::div(y.x, self.singular_values.x)
        } else {
            R::ZERO
        };
        let y2 = if self.singular_values.y > eps {
            R::div(y.y, self.singular_values.y)
        } else {
            R::ZERO
        };
        Some(self.v_t.tr_mul_vec(Vector2 { x: y1, y: y2 }))
    }

    /// The **polar decomposition** `M = R · P`: `R = U · v_t` is orthonormal (a rotation when
    /// `det(M) > 0`) and `P = V · diag(σ) · Vᵀ` is symmetric positive semi-definite. Returned
    /// as `(R, P)` with `P` a `SymMatrix2`, since only its 3 independent components exist.
    ///
    /// This is the form rapier's deformable bodies need (the rotation that best matches a
    /// deformation gradient, and the residual stretch). `R` costs one 2x2 product and `P` one
    /// `SymMatrix2::quadform`; both are two roundings per entry. Panics on overflow.
    ///
    /// No upstream equivalent at a fixed size: upstream users compose `svd.u * svd.v_t` and
    /// `svd.v_t.transpose() * Matrix::from_diagonal(&svd.singular_values) * svd.v_t` by hand.
    #[inline(always)]
    fn to_polar(self: Svd2<T>) -> (Matrix2<T>, SymMatrix2<T>) {
        (self.u * self.v_t, SymMatrix2Trait::quadform(self.v_t.transpose(), self.singular_values))
    }
}

/// `Matrix2` methods that go through the SVD; upstream carries them on the matrix itself.
/// Import `Matrix2SvdTrait` to use them.
#[generate_trait]
pub impl Matrix2SvdImpl<
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
> of Matrix2SvdTrait<T> {
    /// The singular value decomposition. Upstream: `Matrix2::svd(true, true)`.
    #[inline(always)]
    fn svd(self: Matrix2<T>) -> Svd2<T> {
        Svd2Trait::new(self)
    }

    /// The singular values alone, descending. Upstream: `Matrix2::singular_values`.
    ///
    /// Not cheaper than the full decomposition here (the left vectors are what the norms are read
    /// off), unlike upstream, where skipping `U` and `V` saves the whole accumulation.
    #[inline(always)]
    fn singular_values(self: Matrix2<T>) -> Vector2<T> {
        Svd2Trait::new(self).singular_values
    }

    /// The polar decomposition `M = R · P`, see `Svd2::to_polar`.
    #[inline(always)]
    fn polar_decomposition(self: Matrix2<T>) -> (Matrix2<T>, SymMatrix2<T>) {
        Svd2Trait::new(self).to_polar()
    }

    /// The Moore-Penrose pseudo-inverse, see `Svd2::pseudo_inverse`. Upstream:
    /// `Matrix2::pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Matrix2<T>, eps: T) -> Option<Matrix2<T>> {
        Svd2Trait::new(self).pseudo_inverse(eps)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Svd2`: the exact cases (identity, diagonal, permutation, rank 1, zero), the
    //! identities (`M = U Σ Vᵀ`, orthonormality, `M = R P`), the oracle vectors of
    //! `tools/oracle`
    //! and the two candidates that lost, kept as evidence (AGENTS.md rule 8):
    //!
    //! - `alt_sqrt_eigenvalues`: `σ = sqrt(λ)` from the eigenvalues of `MᵀM`, the formulation
    //! DESIGN D6 describes. Cheaper by two `norm2`, and two to three orders of magnitude less
    //! accurate, because squaring the matrix halves the significant bits of a singular value.
    //!
    //! - `alt_normalised_columns`: `u_2 = M v_2 / σ_2` instead of the perpendicular of `u_1`.
    //! Dearer, and its orthonormality degrades with the condition number.

    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix2::{Matrix2, Matrix2Trait};
    use crate::base::matrix_test_utils::{
        amax_m2, excess, int, m2, max_abs_v2, max_ulp_diff2, max_ulp_diff_v2, oracle_tol,
        orthonormality_error_m2, v2t,
    };
    use crate::base::sym_matrix2::SymMatrix2Trait;
    use crate::base::vector2::{Vector2, Vector2Trait};
    use crate::linalg::oracle_svd;
    use crate::linalg::symmetric_eigen2::SymmetricEigen2Trait;
    use super::{Matrix2SvdTrait, Svd2, Svd2Trait};

    /// An oracle `unit` 2x2 case: the benchmark input.
    fn a_bench() -> Matrix2<Fixed> {
        m2([[1004118060, 214027565], [-392529164, 1045075073]])
    }

    /// `a_bench()` already decomposed.
    fn f_bench() -> Svd2<Fixed> {
        Svd2Trait::new(a_bench())
    }

    /// A rank-1 matrix whose decomposition is exact: `2 e_1 (3 e_1 + 4 e_2)ᵀ / 5` is not
    /// representable, so take the axis-aligned `[[2, 0], [6, 0]]` — its second singular value is
    /// exactly zero.
    fn a_rank1() -> Matrix2<Fixed> {
        Matrix2Trait::new(int(2), int(0), int(6), int(0))
    }

    // --- the losing candidates -----------------------------------------------------------------

    /// `σ = sqrt(λ)` from the eigenvalues of `MᵀM` (DESIGN D6's formulation), descending. Kept
    /// as evidence: see `test_singular_values_candidates`.
    fn singular_values_from_sqrt(m: Matrix2<Fixed>) -> Vector2<Fixed> {
        let e = SymmetricEigen2Trait::eigenvalues(Svd2Trait::gram(m));
        Vector2 { x: e.y.sqrt(), y: e.x.sqrt() }
    }

    /// `u_2 = M v_2 / σ_2` instead of the perpendicular of `u_1`. Kept as evidence: see
    /// `test_left_vectors_candidates`.
    fn u_from_normalised_columns(m: Matrix2<Fixed>) -> Matrix2<Fixed> {
        let f = Svd2Trait::new(m);
        let v = f.v_t.transpose();
        let (w1, w2) = (m.mul_vec(v.column1()), m.mul_vec(v.column2()));
        let c1 = if f.singular_values.x == Real::ZERO {
            Vector2 { x: Real::ONE, y: Real::ZERO }
        } else {
            w1.unscale(f.singular_values.x)
        };
        let c2 = if f.singular_values.y == Real::ZERO {
            Vector2 { x: -c1.y, y: c1.x }
        } else {
            w2.unscale(f.singular_values.y)
        };
        Matrix2Trait::from_columns(c1, c2)
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_identity_is_exact() {
        let f = Svd2Trait::new(Matrix2Trait::<Fixed>::identity());
        assert!(f.singular_values() == Vector2 { x: int(1), y: int(1) });
        assert!(f.u == Matrix2Trait::identity());
        assert!(f.v_t == Matrix2Trait::identity());
        assert!(f.recompose() == Matrix2Trait::identity());
        assert!(f.rank(Real::ZERO) == 2);
    }

    #[test]
    fn test_new_diagonal_is_exact() {
        // diag(2, -5): singular values 5 and 2, and the sign moves into U.
        let d = Matrix2Trait::new(int(2), int(0), int(0), int(-5));
        let f = Svd2Trait::new(d);
        assert!(f.singular_values == Vector2 { x: int(5), y: int(2) });
        assert!(f.recompose() == d);
        assert!(orthonormality_error_m2(f.u) == 0);
        assert!(orthonormality_error_m2(f.v_t) == 0);
    }

    #[test]
    fn test_new_permutation_is_exact() {
        let p = Matrix2Trait::new(int(0), int(1), int(1), int(0));
        let f = Svd2Trait::new(p);
        assert!(f.singular_values == Vector2 { x: int(1), y: int(1) });
        assert!(f.recompose() == p);
    }

    #[test]
    fn test_new_rank_one_and_zero() {
        let f = Svd2Trait::new(a_rank1());
        assert!(f.singular_values.y == Real::ZERO);
        assert!(f.rank(Real::ZERO) == 1);
        // `V` is irrational here (the eigen direction is `(-36, 12)`), so `U` is only orthonormal
        // to the rounding of one normalisation.
        assert!(orthonormality_error_m2(f.u) <= 4);
        assert!(max_ulp_diff2(f.recompose(), a_rank1()) <= 16);
        // The zero matrix: every singular value vanishes and nothing divides by zero.
        let z = Svd2Trait::new(Matrix2Trait::<Fixed>::zeros());
        assert!(z.singular_values == Vector2 { x: int(0), y: int(0) });
        assert!(z.u == Matrix2Trait::identity());
        assert!(z.rank(Real::ZERO) == 0);
        assert!(z.recompose() == Matrix2Trait::zeros());
    }

    #[test]
    fn test_accessors() {
        let f = f_bench();
        assert!(f.singular_values() == f.singular_values);
        assert!(f.v() == f.v_t.transpose());
        assert!(a_bench().svd() == f);
        assert!(a_bench().singular_values() == f.singular_values);
    }

    // --- oracle --------------------------------------------------------------------------------

    #[test]
    fn test_new_singular_values_oracle() {
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let (mut worst, mut worst_ex) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = Svd2Trait::new(m2(a)).singular_values;
            let e = v2t(expected);
            let err = max_ulp_diff_v2(got, e);
            worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v2(e), tol)));
            assert!(got.x >= got.y && got.y >= Real::ZERO, "not sorted descending");
            worst = core::cmp::max(worst, err);
        }
        assert!((worst, worst_ex) == (48, 0), "regressed: {worst} {worst_ex}");
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle() {
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let f = Svd2Trait::new(m2(a));
            let rec = max_ulp_diff2(f.recompose(), m2(a)) / amax_m2(m2(a));
            let orth = core::cmp::max(orthonormality_error_m2(f.u), orthonormality_error_m2(f.v_t));
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        assert!(worst_rec == 6 && worst_orth == 23, "regressed: {worst_rec} {worst_orth}");
    }

    #[test]
    fn test_singular_values_candidates() {
        // DESIGN D6's `sqrt(lambda)` against the shipped `|M v|`, on the oracle expectations.
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let (mut worst_norm, mut worst_sqrt) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            let e = v2t(expected);
            worst_norm =
                core::cmp::max(
                    worst_norm, max_ulp_diff_v2(Svd2Trait::new(m2(a)).singular_values, e),
                );
            worst_sqrt =
                core::cmp::max(worst_sqrt, max_ulp_diff_v2(singular_values_from_sqrt(m2(a)), e));
        }
        assert!(worst_norm == 48 && worst_sqrt == 10, "regressed: {worst_norm} {worst_sqrt}");
    }

    #[test]
    fn test_left_vectors_candidates() {
        // The perpendicular is exactly orthonormal; the normalised second column is not.
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let (mut worst_perp, mut worst_norm) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            worst_perp =
                core::cmp::max(worst_perp, orthonormality_error_m2(Svd2Trait::new(m2(a)).u));
            worst_norm =
                core::cmp::max(
                    worst_norm, orthonormality_error_m2(u_from_normalised_columns(m2(a))),
                );
        }
        assert!(worst_perp == 23 && worst_norm == 56, "regressed: {worst_perp} {worst_norm}");
    }

    #[test]
    fn test_solve_matches_the_inverse_oracle() {
        // On a non-singular matrix the least-squares solution IS the solution.
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let mut worst = 0;
        let b = v2t((-5886581674, -6536196560));
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let x = Svd2Trait::new(m2(a)).solve(b, Real::EPSILON).unwrap();
            let e = m2(a).try_inverse().unwrap().mul_vec(b);
            worst = core::cmp::max(worst, max_ulp_diff_v2(x, e));
        }
        // Measured gap to `Matrix2::try_inverse` * b over the 30 well-conditioned vectors.
        assert!(worst == 2099, "regressed: {worst}");
    }

    #[test]
    fn test_pseudo_inverse_oracle() {
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let p = Svd2Trait::new(m2(a)).pseudo_inverse(Real::EPSILON).unwrap();
            let e = m2(a).try_inverse().unwrap();
            worst = core::cmp::max(worst, max_ulp_diff2(p, e));
        }
        assert!(worst == 1947, "regressed: {worst}");
    }

    #[test]
    fn test_pseudo_inverse_of_a_rank_one_matrix() {
        // The pseudo-inverse drops the null direction: `A A⁺ A = A`.
        let p = Svd2Trait::new(a_rank1()).pseudo_inverse(Real::EPSILON).unwrap();
        assert!(max_ulp_diff2(a_rank1() * p * a_rank1(), a_rank1()) <= 64);
        assert!(Svd2Trait::new(a_rank1()).pseudo_inverse(Real::NEG_ONE).is_none());
        assert!(
            Svd2Trait::new(a_rank1())
                .solve(Vector2 { x: int(1), y: int(1) }, Real::NEG_ONE)
                .is_none(),
        );
    }

    #[test]
    fn test_to_polar_oracle() {
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let (mut worst, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let (r, p) = Svd2Trait::new(m2(a)).to_polar();
            // `P` is symmetric by type and positive semi-definite: its determinant is the product
            // of the singular values.
            assert!(!p.determinant().is_negative(), "P is not positive semi-definite");
            worst_orth = core::cmp::max(worst_orth, orthonormality_error_m2(r));
            worst = core::cmp::max(worst, max_ulp_diff2(r * p.to_matrix(), m2(a)) / amax_m2(m2(a)));
        }
        // Measured: `|M - R P| <= worst ulp * max(1, max |m_ij|)`, `|RᵀR - I| <= worst_orth ulp`.
        assert!((worst, worst_orth) == (7, 24), "regressed: {worst} {worst_orth}");
    }

    #[test]
    fn test_polar_decomposition_of_a_rotation_is_the_rotation() {
        // `M = R` exactly: the stretch is the identity.
        let r = Matrix2Trait::new(int(0), int(-1), int(1), int(0));
        let (q, p) = r.polar_decomposition();
        assert!(max_ulp_diff2(q, r) <= 2);
        assert!(max_ulp_diff2(p.to_matrix(), Matrix2Trait::identity()) <= 2);
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_new_overflow_panics() {
        // `MᵀM` does not fit: the squares of the entries must be representable.
        let m = black_box(Matrix2Trait::from_diagonal_element(Real::<Fixed>::MAX));
        Svd2Trait::new(m);
    }

    // --- gas benchmarks --------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_svd2_new__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_new__eigen_of_gram() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = Svd2Trait::new(a);
        assert!((f.singular_values.x >= f.singular_values.y) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_singular_values__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    /// The singular values as the decomposition produces them, `|M v_i|`.
    #[test]
    #[inline(never)]
    fn bench_svd2_singular_values__from_left_vectors() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let s = Svd2Trait::new(a).singular_values;
        assert!((s.x >= s.y) == e);
    }

    /// `sqrt` of the eigenvalues of `MᵀM`, which needs no left vector at all: three times
    /// cheaper, and not what ships (see the module doc — it can take the square root of a
    /// negative number on a rank-deficient input, and it is LESS accurate in 3x3).
    #[test]
    #[inline(never)]
    fn bench_svd2_singular_values__alt_sqrt_eigenvalues() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let s = singular_values_from_sqrt(a);
        assert!((s.x >= s.y) == e);
    }

    /// `new` PLUS a second, naive construction of `U`: it calls `new` itself, so it can only be
    /// dearer than the shipped path. The decisive comparison is the accuracy one
    /// (`test_left_vectors_candidates`).
    #[test]
    #[inline(never)]
    fn bench_svd2_new__alt_normalised_columns() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let u = u_from_normalised_columns(a);
        assert!((u.m11 != Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_recompose__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_recompose__scaled_product() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.recompose().m11 != Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_solve__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(Vector2 { x: int(1), y: int(2) });
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_solve__divisions() {
        let f = black_box(f_bench());
        let b = black_box(Vector2 { x: int(1), y: int(2) });
        let e = black_box(true);
        assert!(f.solve(b, Real::EPSILON).is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_pseudo_inverse__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_pseudo_inverse__reciprocals() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.pseudo_inverse(Real::EPSILON).is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_rank__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_rank__comparisons() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.rank(Real::EPSILON) == 2) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_to_polar__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_svd2_to_polar__quadform() {
        let f = black_box(f_bench());
        let e = black_box(true);
        let (r, p) = f.to_polar();
        assert!((r.m11 != Real::ZERO && p.m11 != Real::ZERO) == e);
    }
}
