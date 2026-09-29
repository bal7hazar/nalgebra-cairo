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

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)
#[cfg(test)]
pub(crate) use nalgebra_linalg_svd_eigen4::internal::linalg::svd2::Svd2InternalTrait;
pub use nalgebra_linalg_svd_eigen4::linalg::svd2::*;

/// Test-only field-wise equality (upstream `Svd2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Svd2PartialEq<T, +PartialEq<T>> of PartialEq<Svd2<T>> {
    fn eq(lhs: @Svd2<T>, rhs: @Svd2<T>) -> bool {
        lhs.u == rhs.u && lhs.singular_values == rhs.singular_values && lhs.v_t == rhs.v_t
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Svd2`: the exact cases (identity, diagonal, permutation, rank 1, zero), the
    //! identities (`M = U Σ Vᵀ`, orthonormality, `M = P U`), the oracle vectors of
    //! `tools/oracle`
    //! and the two candidates that lost, kept as evidence (AGENTS.md rule 8):
    //!
    //! - `alt_sqrt_eigenvalues`: `σ = sqrt(λ)` from the eigenvalues of `MᵀM`, the formulation
    //! DESIGN D6 describes. Cheaper by two `norm2`, and two to three orders of magnitude less
    //! accurate, because squaring the matrix halves the significant bits of a singular value.
    //!
    //! - `alt_normalised_columns`: `u_2 = M v_2 / σ_2` instead of the perpendicular of `u_1`.
    //! Dearer, and its orthonormality degrades with the condition number.
    //!
    //! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API
    //! are in `crates/tests_linalg/src/svd2/tests.cairo`.

    use fixed::Fixed;
    use nalgebra_static3::internal::base::matrix2::Matrix2InternalTrait;
    use simba::scalar::Real;
    use crate::base::MatrixMul;
    use crate::base::matrix2::{Matrix2, Matrix2Trait};
    use crate::base::matrix_test_utils::{
        amax_m2, int, m2, max_ulp_diff2, max_ulp_diff_v2, orthonormality_error_m2, v2t,
    };
    use crate::base::vector2::{Vector2, Vector2Trait};
    use crate::linalg::oracle_svd;
    use crate::linalg::symmetric_eigen2::SymmetricEigen2InternalTrait;
    use crate::testing::black_box;
    use super::{Svd2InternalTrait, Svd2Trait};

    /// An oracle `unit` 2x2 case: the benchmark input.
    fn a_bench() -> Matrix2<Fixed> {
        m2([[1004118060, 214027565], [-392529164, 1045075073]])
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
        let e = SymmetricEigen2InternalTrait::eigenvalues(Svd2InternalTrait::gram(m));
        Vector2 { x: e.y.sqrt(), y: e.x.sqrt() }
    }

    /// `u_2 = M v_2 / σ_2` instead of the perpendicular of `u_1`. Kept as evidence: see
    /// `test_left_vectors_candidates`.
    fn u_from_normalised_columns(m: Matrix2<Fixed>) -> Matrix2<Fixed> {
        let f = Svd2Trait::new(m, true, true);
        let v = f.v_t.unwrap().transpose();
        let (w1, w2) = (m.mul_mat(v.column1()), m.mul_mat(v.column2()));
        let c1 = if f.singular_values.x == Real::zero() {
            Vector2 { x: Real::one(), y: Real::zero() }
        } else {
            w1.unscale(f.singular_values.x)
        };
        let c2 = if f.singular_values.y == Real::zero() {
            Vector2 { x: -c1.y, y: c1.x }
        } else {
            w2.unscale(f.singular_values.y)
        };
        Matrix2Trait::from_columns(c1, c2)
    }

    #[test]
    fn test_new_diagonal_is_exact() {
        // diag(2, -5): singular values 5 and 2, and the sign moves into U.
        let d = Matrix2Trait::new(int(2), int(0), int(0), int(-5));
        let f = Svd2Trait::new(d, true, true);
        assert!(f.singular_values == Vector2 { x: int(5), y: int(2) });
        assert!(f.recompose().unwrap() == d);
        assert!(orthonormality_error_m2(f.u.unwrap()) == 0);
        assert!(orthonormality_error_m2(f.v_t.unwrap()) == 0);
    }

    #[test]
    fn test_new_rank_one_and_zero() {
        let f = Svd2Trait::new(a_rank1(), true, true);
        assert!(f.singular_values.y == Real::zero());
        assert!(f.rank(Real::zero()) == 1);
        // `V` is irrational here (the eigen direction is `(-36, 12)`), so `U` is only orthonormal
        // to the rounding of one normalisation.
        assert!(orthonormality_error_m2(f.u.unwrap()) <= 4);
        assert!(max_ulp_diff2(f.recompose().unwrap(), a_rank1()) <= 16);
        // The zero matrix: every singular value vanishes and nothing divides by zero.
        let z = Svd2Trait::new(Matrix2Trait::<Fixed>::zeros(), true, true);
        assert!(z.singular_values == Vector2 { x: int(0), y: int(0) });
        assert!(z.u.unwrap() == Matrix2Trait::identity());
        assert!(z.rank(Real::zero()) == 0);
        assert!(z.recompose().unwrap() == Matrix2Trait::zeros());
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle() {
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let f = Svd2Trait::new(m2(a), true, true);
            let rec = max_ulp_diff2(f.recompose().unwrap(), m2(a)) / amax_m2(m2(a));
            let orth = core::cmp::max(
                orthonormality_error_m2(f.u.unwrap()), orthonormality_error_m2(f.v_t.unwrap()),
            );
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
                    worst_norm,
                    max_ulp_diff_v2(Svd2Trait::new(m2(a), true, true).singular_values, e),
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
                core::cmp::max(
                    worst_perp,
                    orthonormality_error_m2(Svd2Trait::new(m2(a), true, true).u.unwrap()),
                );
            worst_norm =
                core::cmp::max(
                    worst_norm, orthonormality_error_m2(u_from_normalised_columns(m2(a))),
                );
        }
        assert!(worst_perp == 23 && worst_norm == 56, "regressed: {worst_perp} {worst_norm}");
    }

    #[test]
    fn test_to_polar_oracle() {
        let mut cases = oracle_svd::svd2_singular_values_cases();
        let (mut worst, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let (p, u) = Svd2Trait::new(m2(a), true, true).to_polar().unwrap();
            // `P` is symmetric by construction and positive semi-definite: its determinant is the
            // product of the singular values.
            assert!(p == p.transpose(), "P is not symmetric");
            assert!(!p.determinant().is_sign_negative(), "P is not positive semi-definite");
            worst_orth = core::cmp::max(worst_orth, orthonormality_error_m2(u));
            worst = core::cmp::max(worst, max_ulp_diff2(p * u, m2(a)) / amax_m2(m2(a)));
        }
        // Measured: `|M - P U| <= worst ulp * max(1, max |m_ij|)`, `|UᵀU - I| <= worst_orth ulp`.
        assert!((worst, worst_orth) == (6, 24), "regressed: {worst} {worst_orth}");
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
        assert!((u.m11 != Real::zero()) == e);
    }
}
