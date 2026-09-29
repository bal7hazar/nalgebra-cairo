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

use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::matrix_mul::MatrixMul;
use nalgebra_core::base::matrix_tr_mul::MatrixTrMul;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::internal::base::sym_matrix2::{SymMatrix2, SymMatrix2Trait};
use nalgebra_static3::base::matrix2::Matrix2Trait;
use nalgebra_static3::internal::base::matrix2::Matrix2InternalTrait;
use simba::scalar::Real;
use crate::internal::linalg::svd2::Svd2InternalTrait;
use crate::internal::linalg::symmetric_eigen2::SymmetricEigen2InternalTrait;

/// The singular value decomposition `M = U · diag(singular_values) · v_t` of a `Matrix2<T>`.
///
/// `singular_values` are sorted **descending** and non-negative (like `tools/oracle` and upstream's
/// `singular_values`), `u` and `v_t` are orthonormal, and `None` when the decomposition was built
/// without them (the `compute_u` / `compute_v` flags, like upstream).
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
#[derive(Copy, Drop, Serde, Debug)]
pub struct Svd2<T> {
    /// The left singular vectors, as columns, when computed. Orthonormal.
    pub u: Option<Matrix2<T>>,
    /// The two singular values, descending, non-negative.
    pub singular_values: Vector2<T>,
    /// The TRANSPOSE of the right singular vectors, i.e. `v_i` is ROW `i`, when computed.
    /// Orthonormal.
    pub v_t: Option<Matrix2<T>>,
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
    /// The singular value decomposition of `matrix`, with `u` when `compute_u` and `v_t` when
    /// `compute_v` (`None` otherwise; skipping `u` skips its two divisions and the perpendicular,
    /// `v_t` is free: the right vectors are what the singular values are read off). Upstream:
    /// `matrix.svd(compute_u, compute_v)` / `SVD::new(matrix, compute_u, compute_v)`.
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
    /// the left polar factors within `|M - P U| <= 6 ulp * max(1, max |m_ij|)` with
    /// `|UᵀU - I| <= 24 ulp`.
    ///
    /// Cost: constant. Panics with the scalar's overflow error if a component of `MᵀM` does not
    /// fit (the squares of the entries must be representable, unlike `norm2`).
    fn new(matrix: Matrix2<T>, compute_u: bool, compute_v: bool) -> Svd2<T> {
        let eigen = SymmetricEigen2InternalTrait::new_sym(Svd2InternalTrait::gram(matrix));
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
        let (w1, w2) = (matrix.mul_mat(v1), matrix.mul_mat(v2));
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
        let u = if compute_u {
            Some(Svd2InternalTrait::left(w1, w2, s1))
        } else {
            None
        };
        Svd2 {
            u,
            singular_values: Vector2 { x: s1, y: s2 },
            v_t: if compute_v {
                Some(Matrix2Trait::from_rows(v1, v2))
            } else {
                None
            },
        }
    }

    /// `new` (the closed form is always sorted). Upstream: `SVD::new_unordered`.
    #[inline(always)]
    fn new_unordered(matrix: Matrix2<T>, compute_u: bool, compute_v: bool) -> Svd2<T> {
        Self::new(matrix, compute_u, compute_v)
    }

    /// `Some(new(matrix))`: the 2x2 decomposition is a closed form, there is nothing to converge.
    /// `eps` and `max_niter` are accepted for signature parity and ignored. Upstream:
    /// `SVD::try_new(matrix, compute_u, compute_v, eps, max_niter)`.
    #[inline(always)]
    fn try_new(
        matrix: Matrix2<T>, compute_u: bool, compute_v: bool, eps: T, max_niter: usize,
    ) -> Option<Svd2<T>> {
        let _ = eps;
        let _ = max_niter;
        Some(Self::new(matrix, compute_u, compute_v))
    }

    /// `try_new` (always sorted). Upstream: `SVD::try_new_unordered`.
    #[inline(always)]
    fn try_new_unordered(
        matrix: Matrix2<T>, compute_u: bool, compute_v: bool, eps: T, max_niter: usize,
    ) -> Option<Svd2<T>> {
        Self::try_new(matrix, compute_u, compute_v, eps, max_niter)
    }

    /// Sorts the singular values DESCENDING, swapping the columns of `u` and the rows of `v_t`
    /// (those that were computed) with them (strict comparison: equal values keep their order).
    /// `new` already returns them sorted, so this only matters after the fields were edited.
    /// Upstream: `SVD::sort_by_singular_values`.
    fn sort_by_singular_values(ref self: Svd2<T>) {
        if self.singular_values.y > self.singular_values.x {
            let u = match self.u {
                Some(u) => Some(Matrix2 { m11: u.m12, m21: u.m22, m12: u.m11, m22: u.m21 }),
                None => None,
            };
            let v_t = match self.v_t {
                Some(v) => Some(Matrix2 { m11: v.m21, m21: v.m11, m12: v.m22, m22: v.m12 }),
                None => None,
            };
            self =
                Svd2 {
                    u,
                    singular_values: Vector2 {
                        x: self.singular_values.y, y: self.singular_values.x,
                    },
                    v_t,
                };
        }
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
    /// rounding (measured: **6 ulp per unit of `max |m_ij|`** on the oracle vectors), or `None`
    /// when `u` or `v_t` was not computed.
    ///
    /// Two roundings per entry: the columns of `U` are scaled by the singular values (4 floored
    /// products), then the product with `v_t` is 4 fused `sum_prod2`. Panics on overflow.
    /// Upstream: `SVD::recompose` (`Err` when a factor is missing; `None` here).
    fn recompose(self: Svd2<T>) -> Option<Matrix2<T>> {
        let u = self.u?;
        let v_t = self.v_t?;
        let us = Matrix2 {
            m11: u.m11 * self.singular_values.x,
            m21: u.m21 * self.singular_values.x,
            m12: u.m12 * self.singular_values.y,
            m22: u.m22 * self.singular_values.y,
        };
        Some(us * v_t)
    }

    /// The Moore-Penrose pseudo-inverse `V · diag(σ⁺) · Uᵀ`, where `σ⁺_i = 1 / σ_i` when
    /// `σ_i > eps` and `0` otherwise, or `None` when `eps` is negative or when `u` or `v_t` was
    /// not computed.
    ///
    /// Upstream: `SVD::pseudo_inverse`, which returns `Err` in those cases and otherwise
    /// recomposes with the inverted singular values and takes the adjoint — the same expression.
    /// Note that upstream's test is `> eps` as well, so `eps = 0` keeps every nonzero singular
    /// value, including one of 1 raw unit whose reciprocal overflows: pass an `eps` matched to the
    /// scale of the problem, not zero, unless an overflow panic is the wanted answer.
    ///
    /// Three roundings per entry (the reciprocal, the scaling, the product). Panics with the
    /// scalar's overflow error if a reciprocal or an entry does not fit.
    fn pseudo_inverse(self: Svd2<T>, eps: T) -> Option<Matrix2<T>> {
        if eps.is_sign_negative() {
            return None;
        }
        let u = self.u?;
        let v_t = self.v_t?;
        let r1 = if self.singular_values.x > eps {
            self.singular_values.x.recip()
        } else {
            R::zero()
        };
        let r2 = if self.singular_values.y > eps {
            self.singular_values.y.recip()
        } else {
            R::zero()
        };
        let v = v_t.transpose();
        let vr = Matrix2 { m11: v.m11 * r1, m21: v.m21 * r1, m12: v.m12 * r2, m22: v.m22 * r2 };
        Some(vr * u.transpose())
    }

    /// The least-squares solution of `M x = b`, `V · (Uᵀ b / σ)` with the components whose
    /// singular value is `<= eps` zeroed, or `None` when `eps` is negative or when `u` or `v_t`
    /// was not computed. Upstream: `SVD::solve` (`Err` in those cases).
    ///
    /// Unlike `pseudo_inverse` this divides instead of multiplying by a reciprocal — one rounding
    /// per component instead of two, and no overflow on a tiny singular value unless the solution
    /// itself does not fit. Panics with the scalar's overflow error in that case.
    fn solve(self: Svd2<T>, b: Vector2<T>, eps: T) -> Option<Vector2<T>> {
        if eps.is_sign_negative() {
            return None;
        }
        let u = self.u?;
        let v_t = self.v_t?;
        let y = u.tr_mul(b);
        let y1 = if self.singular_values.x > eps {
            R::div(y.x, self.singular_values.x)
        } else {
            R::zero()
        };
        let y2 = if self.singular_values.y > eps {
            R::div(y.y, self.singular_values.y)
        } else {
            R::zero()
        };
        Some(v_t.tr_mul(Vector2 { x: y1, y: y2 }))
    }

    /// The LEFT polar decomposition `M = P · U`, as `Some((P, U))`: `P = u · diag(σ) · uᵀ` is
    /// symmetric positive semi-definite and `U = u · v_t` is orthonormal (a rotation when
    /// `det(M) > 0`), or `None` when `u` or `v_t` was not computed (as upstream).
    ///
    /// `U` costs one 2x2 product and `P` one structured quadratic form (only its 3 independent
    /// components are computed, then mirrored); both are two roundings per entry. Panics on
    /// overflow. Upstream: `SVD::to_polar`.
    #[inline(always)]
    fn to_polar(self: Svd2<T>) -> Option<(Matrix2<T>, Matrix2<T>)> {
        let u = self.u?;
        let v_t = self.v_t?;
        Some((SymMatrix2Trait::quadform(u, self.singular_values).to_matrix(), u * v_t))
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
    /// The singular value decomposition, see `Svd2Trait::new`. Upstream: `Matrix2::svd`.
    #[inline(always)]
    fn svd(self: Matrix2<T>, compute_u: bool, compute_v: bool) -> Svd2<T> {
        Svd2Trait::new(self, compute_u, compute_v)
    }

    /// The singular values alone, descending: `svd(false, false)` (no left vectors). Upstream:
    /// `Matrix2::singular_values`.
    #[inline(always)]
    fn singular_values(self: Matrix2<T>) -> Vector2<T> {
        Svd2Trait::new(self, false, false).singular_values
    }

    /// The Moore-Penrose pseudo-inverse, see `Svd2::pseudo_inverse`. Upstream:
    /// `Matrix2::pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Matrix2<T>, eps: T) -> Option<Matrix2<T>> {
        Svd2Trait::new(self, true, true).pseudo_inverse(eps)
    }

    /// `svd`: the decomposition is always sorted (see `Svd2Trait::new_unordered`). Upstream:
    /// `Matrix::svd_unordered`.
    #[inline(always)]
    fn svd_unordered(self: Matrix2<T>, compute_u: bool, compute_v: bool) -> Svd2<T> {
        Svd2Trait::new(self, compute_u, compute_v)
    }

    /// See `Svd2Trait::try_new`. Upstream: `Matrix::try_svd`.
    #[inline(always)]
    fn try_svd(
        self: Matrix2<T>, compute_u: bool, compute_v: bool, eps: T, max_niter: usize,
    ) -> Option<Svd2<T>> {
        Svd2Trait::try_new(self, compute_u, compute_v, eps, max_niter)
    }

    /// See `Svd2Trait::try_new_unordered`. Upstream: `Matrix::try_svd_unordered`.
    #[inline(always)]
    fn try_svd_unordered(
        self: Matrix2<T>, compute_u: bool, compute_v: bool, eps: T, max_niter: usize,
    ) -> Option<Svd2<T>> {
        Svd2Trait::try_new(self, compute_u, compute_v, eps, max_niter)
    }

    /// `singular_values` (always sorted). Upstream: `Matrix::singular_values_unordered`.
    #[inline(always)]
    fn singular_values_unordered(self: Matrix2<T>) -> Vector2<T> {
        Svd2Trait::new(self, false, false).singular_values
    }

    /// The number of singular values strictly greater than `eps`. Upstream: `Matrix::rank`
    /// (which asserts `eps >= 0`; a negative `eps` counts every value here, like `Svd2::rank`).
    fn rank(self: Matrix2<T>, eps: T) -> usize {
        let s = Svd2Trait::new(self, false, false).singular_values;
        let mut n: usize = 0;
        if s.x > eps {
            n += 1;
        }
        if s.y > eps {
            n += 1;
        }
        n
    }

    /// The left polar decomposition `M = P · U`, see `Svd2Trait::to_polar`. Upstream:
    /// `Matrix::polar`.
    fn polar(self: Matrix2<T>) -> (Matrix2<T>, Matrix2<T>) {
        Svd2Trait::new(self, true, true).to_polar().unwrap()
    }

    /// `polar`, or `None` when the decomposition did not converge within `eps`, see
    /// `Svd2Trait::try_new`. Upstream: `Matrix::try_polar`.
    fn try_polar(self: Matrix2<T>, eps: T, max_niter: usize) -> Option<(Matrix2<T>, Matrix2<T>)> {
        match Svd2Trait::try_new(self, true, true, eps, max_niter) {
            Some(d) => d.to_polar(),
            None => None,
        }
    }
}

/// The ordered SVD of a `Matrix2`: `Svd2Trait::new(m, compute_u, compute_v)`. Upstream:
/// `nalgebra::linalg::svd_ordered2` (the closed form of the 2x2 SVD upstream uses, through
/// `atan2` / `sin_cos`). Here the 2x2 SVD is `Svd2`'s eigen decomposition of `MᵀM`, which needs
/// no transcendental function (steps criterion, see `Svd2`).
pub fn svd_ordered2<
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
>(
    m: Matrix2<T>, compute_u: bool, compute_v: bool,
) -> Svd2<T> {
    Svd2Trait::new(m, compute_u, compute_v)
}
