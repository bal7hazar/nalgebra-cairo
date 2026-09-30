//! `Qr2`: the QR factorisation of a `Matrix2` (upstream `nalgebra::linalg::QR` on a 2x2 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`.

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)
#[cfg(test)]
pub(crate) use nalgebra_linalg4::internal::linalg::qr::qr2::Qr2InternalTrait;
pub use nalgebra_linalg4::linalg::qr::qr2::*;

/// Test-only field-wise equality (upstream `Qr2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Qr2PartialEq<T, +PartialEq<T>> of PartialEq<Qr2<T>> {
    fn eq(lhs: @Qr2<T>, rhs: @Qr2<T>) -> bool {
        lhs.q == rhs.q && lhs.r == rhs.r
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Qr2`: the exact cases (identity, diagonal, permutation, rank deficient), the
    //! identities (`Q R = A`, `QᵀQ = I`, `A A^-1 = I`), and the oracle vectors of `tools/oracle`
    //! (upstream nalgebra 0.35 on the same raw inputs, unpacked factors, `r_ii >= 0`).
    //!
    //! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API
    //! are in `crates/tests_linalg/src/qr/qr2/tests.cairo`.

    use fixed::Fixed;
    use nalgebra_static2::internal::base::matrix2::Matrix2InternalTrait;
    use simba::scalar::Real;
    use crate::base::matrix2::{Matrix2, Matrix2Trait};
    use crate::base::matrix_test_utils::{
        amax_m2, int, m2, max_ulp_diff2, orthonormality_error_m2, ulp_diff, v2t,
    };
    use crate::base::vector2::Vector2;
    use crate::linalg::qr::oracle_qr2 as oracle;
    use crate::testing::black_box;
    use super::{Qr2, Qr2InternalTrait, Qr2Trait};

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
    #[inline(never)]
    fn bench_qr2_determinant__diagonal_product() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.determinant() != Real::zero()) == e);
    }
}
