//! `SymmetricEigen2`: eigenvalues and eigenvectors of a symmetric 2x2 matrix, in closed form.
//!
//! Upstream `nalgebra::linalg::SymmetricEigen` tridiagonalises and then runs an implicitly shifted
//! symmetric QR with Wilkinson shifts until `|off| <= eps * (|d_m| + |d_n|)` — an unbounded loop
//! with a tolerance parameter, which a proof system cannot price. In 2x2 the whole problem is a
//! quadratic (upstream itself special-cases the trailing 2x2 block in closed form,
//! `symmetric_eigen.rs`), so this port solves it directly: constant cost, no tolerance, no
//! iteration count (DESIGN D6).
//!
//! `glamx::SymmetricEigen2` (what parry / rapier use in Rust) solves the same quadratic; the
//! difference here is the choice of the eigenvector row, made on the sign of `(m11 - m22) / 2` so
//! that no cancellation can occur, and the fixed-point kernels (one rounding per output scalar).

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)
#[cfg(test)]
pub(crate) use nalgebra_linalg_svd_eigen2::internal::linalg::symmetric_eigen2::SymmetricEigen2InternalTrait;
pub use nalgebra_linalg_svd_eigen2::linalg::symmetric_eigen2::*;

/// Test-only field-wise equality (upstream `SymmetricEigen2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl SymmetricEigen2PartialEq<T, +PartialEq<T>> of PartialEq<SymmetricEigen2<T>> {
    fn eq(lhs: @SymmetricEigen2<T>, rhs: @SymmetricEigen2<T>) -> bool {
        lhs.eigenvalues == rhs.eigenvalues && lhs.eigenvectors == rhs.eigenvectors
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_static2::internal::base::matrix2::Matrix2InternalTrait;
    use simba::scalar::Real;
    use crate::base::MatrixMul;
    use crate::base::matrix2::{Matrix2, Matrix2Trait};
    use crate::base::matrix_test_utils::{
        amax_s2, fx, int, max_ulp_diff_s2, max_ulp_diff_v2, s2i, s2r, ulp_diff, v2i, v2t,
    };
    use crate::base::sym_matrix2::{SymMatrix2, SymMatrix2Trait};
    use crate::base::vector2::{Vector2, Vector2Trait};
    use crate::linalg::oracle_symmetric_eigen;
    use crate::testing::black_box;
    use super::{
        Matrix2SymmetricEigenTrait, SymmetricEigen2, SymmetricEigen2InternalTrait,
        SymmetricEigen2Trait,
    };

    /// `|<c_i, c_j> - delta_ij|` over the two columns, in raw units.
    fn orthonormality_error(v: Matrix2<Fixed>) -> u128 {
        let (c1, c2) = (v.column1(), v.column2());
        let mut e = ulp_diff(c1.norm(), Real::one());
        e = core::cmp::max(e, ulp_diff(c2.norm(), Real::one()));
        core::cmp::max(e, ulp_diff(c1.dot(c2), Real::zero()))
    }

    /// `|S * c_i - lambda_i * c_i|` over the two columns, in raw units.
    fn residual_error(s: SymMatrix2<Fixed>, e: SymmetricEigen2<Fixed>) -> u128 {
        let (c1, c2) = (e.eigenvectors.column1(), e.eigenvectors.column2());
        let r1 = s.to_matrix().mul_mat(c1) - c1.scale(e.eigenvalues.x);
        let r2 = s.to_matrix().mul_mat(c2) - c2.scale(e.eigenvalues.y);
        core::cmp::max(
            max_ulp_diff_v2(r1, Vector2Trait::zeros()), max_ulp_diff_v2(r2, Vector2Trait::zeros()),
        )
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_diagonal_is_exact() {
        // Already ordered: columns are the axes, in order.
        let e = SymmetricEigen2InternalTrait::new_sym(s2i((2, 0, 7)));
        assert!(e.eigenvalues == v2i(2, 7));
        assert!(e.eigenvectors == Matrix2Trait::identity());
        // Reversed: the permutation that sorts ascending must keep `det = +1`.
        let e = SymmetricEigen2InternalTrait::new_sym(s2i((7, 0, 2)));
        assert!(e.eigenvalues == v2i(2, 7));
        assert!(e.eigenvectors == Matrix2Trait::new(int(0), int(-1), int(1), int(0)));
        assert!(e.eigenvectors.determinant() == Real::one());
        assert!(e.recompose_sym() == s2i((7, 0, 2)));
    }

    #[test]
    fn test_new_isotropic_is_exact() {
        let e = SymmetricEigen2InternalTrait::new_sym(s2i((3, 0, 3)));
        assert!(e.eigenvalues == v2i(3, 3));
        assert!(e.eigenvectors == Matrix2Trait::identity());
        assert!(e.recompose_sym() == s2i((3, 0, 3)));
        // Zero is isotropic too, and must not divide by zero.
        let e = SymmetricEigen2InternalTrait::new_sym(s2i((0, 0, 0)));
        assert!(e.eigenvalues == v2i(0, 0));
        assert!(e.eigenvectors == Matrix2Trait::identity());
    }

    #[test]
    fn test_new_anti_diagonal_is_exact() {
        // [[0, 1], [1, 0]]: eigenvalues -1, 1 for (1, -1)/sqrt(2), (1, 1)/sqrt(2).
        let e = SymmetricEigen2InternalTrait::new_sym(s2i((0, 1, 0)));
        assert!(e.eigenvalues == v2i(-1, 1));
        // `1/sqrt(2)` reached through a floored `norm2` and a correctly rounded division: 2 ulp of
        // the rounded constant.
        let h = Real::<Fixed>::FRAC_1_SQRT_2;
        assert!(max_ulp_diff_v2(e.eigenvectors.column1(), Vector2 { x: h, y: -h }) <= 2);
        assert!(max_ulp_diff_v2(e.eigenvectors.column2(), Vector2 { x: h, y: h }) <= 2);
        assert!(orthonormality_error(e.eigenvectors) <= 2);
    }

    #[test]
    fn test_new_integer_case_is_exact() {
        // [[5, 2], [2, 2]]: mean 3.5, d 1.5, r = sqrt(2.25 + 4) = 2.5 -> eigenvalues 1 and 6.
        let e = SymmetricEigen2InternalTrait::new_sym(s2i((5, 2, 2)));
        assert!(e.eigenvalues == v2i(1, 6));
        // The eigenvectors are (-1, 2)/sqrt(5) and (2, 1)/sqrt(5), irrational: 4 ulp of residual.
        assert!(residual_error(s2i((5, 2, 2)), e) <= 4);
    }

    #[test]
    fn test_new_is_deterministic_under_a_sign_flip_of_the_off_diagonal() {
        // `S` and its conjugate by diag(-1, 1) share the eigenvalues, and the sign convention is
        // on the dominant component (here `y`), so the columns mirror in `x` and the result is
        // reproducible rather than merely "some" basis. Floor rounding is not sign-symmetric
        // (`floor(-x) != -floor(x)`), so the mirrored component is only equal to 1 ulp.
        let a = SymmetricEigen2InternalTrait::new_sym(s2i((5, 2, 2)));
        let b = SymmetricEigen2InternalTrait::new_sym(s2i((5, -2, 2)));
        assert!(a.eigenvalues == b.eigenvalues);
        assert!(ulp_diff(a.eigenvectors.column1().x, -b.eigenvectors.column1().x) <= 1);
        assert!(a.eigenvectors.column1().y == b.eigenvectors.column1().y);
    }

    #[test]
    fn test_canonical_sign_picks_the_dominant_component() {
        assert!(SymmetricEigen2InternalTrait::canonical_sign(v2i(-3, 1)) == v2i(3, -1));
        assert!(SymmetricEigen2InternalTrait::canonical_sign(v2i(1, -3)) == v2i(-1, 3));
        // Tie (|x| == |y|): `x` decides.
        assert!(SymmetricEigen2InternalTrait::canonical_sign(v2i(-1, 1)) == v2i(1, -1));
        assert!(SymmetricEigen2InternalTrait::canonical_sign(v2i(0, -1)) == v2i(0, 1));
        assert!(SymmetricEigen2InternalTrait::<Fixed>::canonical_sign(v2i(0, 0)) == v2i(0, 0));
    }

    #[test]
    fn test_new_reads_the_lower_triangle() {
        // Lower triangle [[5, .], [2, 2]]; the upper entry -9 is ignored, like upstream.
        let m = Matrix2Trait::new(int(5), int(-9), int(2), int(2));
        let e = SymmetricEigen2Trait::new(m);
        assert!(e.eigenvalues == v2i(1, 6));
        let s = SymmetricEigen2InternalTrait::new_sym(s2i((5, 2, 2)));
        assert!(e.eigenvalues == s.eigenvalues && e.eigenvectors == s.eigenvectors);
        assert!(m.symmetric_eigen().eigenvalues == e.eigenvalues);
        assert!(m.symmetric_eigenvalues() == e.eigenvalues);
        assert!(e.recompose() == e.recompose_sym().to_matrix());
    }

    #[test]
    fn test_eigenvalues_matches_new() {
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s2r(a);
            assert!(
                SymmetricEigen2InternalTrait::eigenvalues(
                    s,
                ) == SymmetricEigen2InternalTrait::new_sym(s)
                    .eigenvalues,
            );
        }
    }

    // --- oracle --------------------------------------------------------------------------------

    #[test]
    fn test_new_eigenvalues_oracle() {
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = SymmetricEigen2InternalTrait::new_sym(s2r(a)).eigenvalues;
            let e = max_ulp_diff_v2(got, v2t(expected));
            assert!(e <= tol.into(), "eigenvalues off by more than the oracle tolerance");
            worst = core::cmp::max(worst, e);
        }
        // Measured: the closed form is within 1 ulp of the floored exact eigenvalues.
        assert!(worst <= 1, "worst case regressed");
    }

    #[test]
    fn test_new_eigenvalues_oracle_spd() {
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_spd_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = SymmetricEigen2InternalTrait::new_sym(s2r(a)).eigenvalues;
            let e = max_ulp_diff_v2(got, v2t(expected));
            assert!(e <= tol.into(), "eigenvalues off by more than the oracle tolerance");
            worst = core::cmp::max(worst, e);
        }
        assert!(worst <= 1, "worst case regressed");
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle() {
        let mut worst_rec: u128 = 0;
        let mut worst_orth: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s2r(a);
            let e = SymmetricEigen2InternalTrait::new_sym(s);
            // `det = +1` only up to the rounding of the normalised column.
            assert!(ulp_diff(e.eigenvectors.determinant(), Real::one()) <= 4);
            let orth = orthonormality_error(e.eigenvectors);
            assert!(orth <= 8, "columns are not orthonormal");
            let rec = max_ulp_diff_s2(e.recompose_sym(), s) / amax_s2(s);
            assert!(rec <= 8, "reconstruction is off");
            assert!(residual_error(s, e) <= 4 * amax_s2(s));
            assert!(e.recompose() == e.recompose_sym().to_matrix());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured worst cases over the 30 vectors.
        assert!(worst_rec <= 4 && worst_orth <= 2, "worst case regressed");
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle_spd() {
        // SPD vectors reach much closer to isotropy than the generic ones, and the eigenvector
        // `(d - r, m12)` is then a tiny vector whose direction is coarsely quantised: the unit
        // columns are a factor 20 less accurate here than on the generic vectors. The eigenvalues
        // are not affected (still 1 ulp), and neither is the residual `S c - lambda c`.
        let mut worst_rec: u128 = 0;
        let mut worst_orth: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_spd_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s2r(a);
            let e = SymmetricEigen2InternalTrait::new_sym(s);
            assert!(ulp_diff(e.eigenvectors.determinant(), Real::one()) <= 128);
            let orth = orthonormality_error(e.eigenvectors);
            assert!(orth <= 64, "columns are not orthonormal");
            let rec = max_ulp_diff_s2(e.recompose_sym(), s) / amax_s2(s);
            assert!(rec <= 32, "reconstruction is off");
            assert!(residual_error(s, e) <= 4 * amax_s2(s));
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        assert!(worst_rec <= 15 && worst_orth <= 38, "worst case regressed");
    }

    // --- overflow ------------------------------------------------------------------------------

    #[test]
    #[should_panic(expected: 'i64_add Overflow')]
    fn test_new_overflow_panics() {
        // `mean + r` leaves the representable range.
        let s = black_box(
            SymMatrix2 {
                m11: Real::<Fixed>::max_value().unwrap(),
                m12: Real::max_value().unwrap(),
                m22: Real::max_value().unwrap(),
            },
        );
        SymmetricEigen2InternalTrait::new_sym(s);
    }

    // --- gas -----------------------------------------------------------------------------------

    fn bench_input() -> SymMatrix2<Fixed> {
        SymMatrix2 { m11: fx(0x2c0000000), m12: fx(-0x180000000), m22: fx(0x140000000) }
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_new__baseline() {
        let _s = black_box(bench_input());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_new__closed_form() {
        let s = black_box(bench_input());
        let d = SymmetricEigen2InternalTrait::new_sym(s);
        assert!(d.eigenvalues.x < d.eigenvalues.y);
        assert!(d.eigenvectors.m11 != Real::zero());
    }

    /// The public entry point, `SymmetricEigen2::new` on a full `Matrix2` (lower triangle
    /// read): an inlined wrapper around the kernel measured by `new__closed_form`, which it
    /// should match (WP 8.0).
    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_new_matrix__baseline() {
        let _m = black_box(bench_input().to_matrix());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_new_matrix__public() {
        let m = black_box(bench_input().to_matrix());
        let d = SymmetricEigen2Trait::new(m);
        assert!(d.eigenvalues.x < d.eigenvalues.y);
        assert!(d.eigenvectors.m11 != Real::zero());
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_eigenvalues__baseline() {
        let _s = black_box(bench_input());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_eigenvalues__closed_form() {
        let s = black_box(bench_input());
        let v = SymmetricEigen2InternalTrait::eigenvalues(s);
        assert!(v.x < v.y);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_recompose__baseline() {
        let _d = black_box(SymmetricEigen2InternalTrait::new_sym(bench_input()));
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_recompose__quadform() {
        let d = black_box(SymmetricEigen2InternalTrait::new_sym(bench_input()));
        let s = d.recompose_sym();
        assert!(s.m11 != Real::zero());
    }
}
