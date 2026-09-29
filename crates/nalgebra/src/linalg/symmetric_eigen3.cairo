//! `SymmetricEigen3`: eigenvalues and eigenvectors of a symmetric 3x3 matrix, by a **fixed-sweep
//! cyclic Jacobi** rotation (DESIGN D6).
//!
//! Why not upstream's algorithm: `nalgebra::linalg::SymmetricEigen` tridiagonalises and then runs
//! an implicitly shifted symmetric QR until `|off| <= eps * (|d_m| + |d_n|)`, bounded only by a
//! `max_niter` argument. An unbounded loop has no place in a proven program: its gas depends on
//! the data, and the tolerance has no meaning in fixed point.
//!
//! Why not the closed form: `glamx::SymmetricEigen3` (parry / rapier's 3x3) uses Eberly's
//! trigonometric method, which needs `acos` and `cos` — `simba` has no `Transcendental<Fixed>`
//! impl yet, and `docs/research/01-nalgebra-analysis.md` records that it misbehaves on isotropic
//! inertia tensors, exactly the input rapier feeds it. A Cardano form needs the same `acos`. Even
//! ignoring that, the coefficients of the characteristic polynomial are the bottleneck: forming
//! `tr`, the sum of the principal 2x2 minors and `det` with the fused kernels and then solving the
//! cubic *exactly in f64* still costs up to 90 ulp on `small` matrices (cancellation in `det`),
//! against 9 ulp for the Jacobi sweeps below. The measurement is in the report of this work
//! package; the closed form is not ported.
//!
//! Cyclic Jacobi is a sequence of plane rotations, each one annihilating one off-diagonal entry
//! and each one an exact similarity transform in exact arithmetic. It converges quadratically, it
//! needs only `sqrt` and division, it is unconditionally stable, and **four sweeps of the three
//! rotations drive every off-diagonal entry of a 3x3 matrix to exactly zero** in Q32.32 —
//! measured over the oracle suite and 2 800 random / degenerate matrices (see `new`). The sweep
//! count is a constant, so the gas of the decomposition is a constant.

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)
#[cfg(test)]
pub(crate) use nalgebra_linalg_svd_eigen4::internal::linalg::symmetric_eigen3::{
    Jacobi3, Jacobi3Impl, Jacobi3Trait, SymmetricEigen3InternalTrait,
};
pub use nalgebra_linalg_svd_eigen4::linalg::symmetric_eigen3::*;

/// Test-only field-wise equality (upstream `SymmetricEigen3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl SymmetricEigen3PartialEq<T, +PartialEq<T>> of PartialEq<SymmetricEigen3<T>> {
    fn eq(lhs: @SymmetricEigen3<T>, rhs: @SymmetricEigen3<T>) -> bool {
        lhs.eigenvalues == rhs.eigenvalues && lhs.eigenvectors == rhs.eigenvectors
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use fixed::wide::{NormTrait, RecipTrait, norm3_wide};
    use nalgebra_static3::internal::base::matrix3::Matrix3InternalTrait;
    use simba::scalar::Real;
    use crate::base::MatrixMul;
    use crate::base::matrix3::{Matrix3, Matrix3Trait};
    use crate::base::matrix_test_utils::{
        amax_s3, fx, int, m3, max_ulp_diff_s3, max_ulp_diff_v3, s3i, s3r, ulp_diff, v3i, v3t,
    };
    use crate::base::sym_matrix3::{SymMatrix3, SymMatrix3Trait};
    use crate::base::vector3::{Vector3, Vector3Trait};
    use crate::linalg::oracle_symmetric_eigen;
    use crate::linalg::symmetric_eigen3::{Matrix3SymmetricEigenTrait, SymmetricEigen3InternalTrait};
    use crate::testing::black_box;
    use super::{Jacobi3, Jacobi3Impl, Jacobi3Trait, SymmetricEigen3, SymmetricEigen3Trait};

    // --- the losing candidates of the sweep-count study (kept as evidence) ----------------------

    /// `new` with three sweeps: leaves off-diagonal entries behind
    /// (see `test_three_sweeps_do_not_converge`).
    fn eigen_sweeps3(s: SymMatrix3<Fixed>) -> SymmetricEigen3<Fixed> {
        Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().finish()
    }

    /// `new` with five sweeps: the fifth sweep finds every off-diagonal entry already zero and
    /// returns the state unchanged (see `test_a_fifth_and_sixth_sweep_change_nothing`).
    fn eigen_sweeps5(s: SymMatrix3<Fixed>) -> SymmetricEigen3<Fixed> {
        Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().sweep().sweep().finish()
    }

    /// `new` with six sweeps.
    fn eigen_sweeps6(s: SymMatrix3<Fixed>) -> SymmetricEigen3<Fixed> {
        Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().sweep().sweep().sweep().finish()
    }

    /// `finish` without the final renormalisation of the columns: 2 `norm3` and 6 divisions
    /// cheaper, and measurably less orthonormal.
    fn finish_without_renormalisation(j: Jacobi3<Fixed>) -> SymmetricEigen3<Fixed> {
        let (eigenvalues, v) = j.sorted();
        let c1 = Jacobi3Impl::<Fixed>::canonical_sign(v.column1());
        let c2 = Jacobi3Impl::<Fixed>::canonical_sign(v.column2());
        SymmetricEigen3 {
            eigenvalues, eigenvectors: Matrix3Trait::from_columns(c1, c2, c1.cross(c2)),
        }
    }

    // --- error measures ------------------------------------------------------------------------

    /// `|<c_i, c_j> - delta_ij|` over the three columns, in raw units.
    fn orthonormality_error(v: Matrix3<Fixed>) -> u128 {
        let (c1, c2, c3) = (v.column1(), v.column2(), v.column3());
        let mut e = ulp_diff(c1.norm(), Real::one());
        e = core::cmp::max(e, ulp_diff(c2.norm(), Real::one()));
        e = core::cmp::max(e, ulp_diff(c3.norm(), Real::one()));
        e = core::cmp::max(e, ulp_diff(c1.dot(c2), Real::zero()));
        e = core::cmp::max(e, ulp_diff(c1.dot(c3), Real::zero()));
        core::cmp::max(e, ulp_diff(c2.dot(c3), Real::zero()))
    }

    /// `|S * c_i - lambda_i * c_i|` over the three columns, in raw units.
    fn residual_error(s: SymMatrix3<Fixed>, e: SymmetricEigen3<Fixed>) -> u128 {
        let (c1, c2, c3) = (
            e.eigenvectors.column1(), e.eigenvectors.column2(), e.eigenvectors.column3(),
        );
        let zero = Vector3Trait::zeros();
        let mut r = max_ulp_diff_v3(s.to_matrix().mul_mat(c1) - c1.scale(e.eigenvalues.x), zero);
        r =
            core::cmp::max(
                r, max_ulp_diff_v3(s.to_matrix().mul_mat(c2) - c2.scale(e.eigenvalues.y), zero),
            );
        core::cmp::max(
            r, max_ulp_diff_v3(s.to_matrix().mul_mat(c3) - c3.scale(e.eigenvalues.z), zero),
        )
    }

    /// Largest `|off-diagonal|` of the partially diagonalised matrix, in raw units.
    fn off_diagonal_error(j: Jacobi3<Fixed>) -> u128 {
        let mut e = ulp_diff(j.s.m12, Real::zero());
        e = core::cmp::max(e, ulp_diff(j.s.m13, Real::zero()));
        core::cmp::max(e, ulp_diff(j.s.m23, Real::zero()))
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_diagonal_is_exact() {
        let e = SymmetricEigen3InternalTrait::new_sym(s3i((-2, 0, 0, 1, 0, 7)));
        assert!(e.eigenvalues == v3i(-2, 1, 7));
        assert!(e.eigenvectors == Matrix3Trait::identity());
        assert!(e.recompose_sym() == s3i((-2, 0, 0, 1, 0, 7)));
    }

    #[test]
    fn test_new_diagonal_permutations_are_exact_and_right_handed() {
        // Every permutation of (-2, 1, 7) on the diagonal: the columns are signed axes, the
        // eigenvalues ascend and the determinant is exactly +1.
        let mut cases = [(-2, 1, 7), (-2, 7, 1), (1, -2, 7), (1, 7, -2), (7, -2, 1), (7, 1, -2)]
            .span();
        while let Some(case) = cases.pop_front() {
            let (a, b, c) = *case;
            let s = s3i((a, 0, 0, b, 0, c));
            let e = SymmetricEigen3InternalTrait::new_sym(s);
            assert!(e.eigenvalues == v3i(-2, 1, 7));
            assert!(e.eigenvectors.determinant() == Real::one());
            assert!(e.recompose_sym() == s);
            assert!(residual_error(s, e) == 0);
        }
    }

    #[test]
    fn test_new_isotropic_is_exact() {
        let e = SymmetricEigen3InternalTrait::new_sym(s3i((3, 0, 0, 3, 0, 3)));
        assert!(e.eigenvalues == v3i(3, 3, 3));
        assert!(e.eigenvectors == Matrix3Trait::identity());
        assert!(e.recompose_sym() == s3i((3, 0, 0, 3, 0, 3)));
        // Zero is isotropic too, and must not divide by zero.
        let e = SymmetricEigen3InternalTrait::new_sym(s3i((0, 0, 0, 0, 0, 0)));
        assert!(e.eigenvalues == v3i(0, 0, 0));
        assert!(e.eigenvectors == Matrix3Trait::identity());
    }

    #[test]
    fn test_new_repeated_eigenvalues_is_exact() {
        // diag(5, 3, 3): the (2, 3) eigenspace is a plane, any orthonormal basis of it is valid.
        let s = s3i((5, 0, 0, 3, 0, 3));
        let e = SymmetricEigen3InternalTrait::new_sym(s);
        assert!(e.eigenvalues == v3i(3, 3, 5));
        assert!(e.eigenvectors.determinant() == Real::one());
        assert!(residual_error(s, e) == 0);
        assert!(e.recompose_sym() == s);
    }

    #[test]
    fn test_new_rank_one_matrix() {
        // The all-ones matrix: eigenvalues 0, 0, 3 with (1, 1, 1)/sqrt(3) for 3.
        let s = s3i((1, 1, 1, 1, 1, 1));
        let e = SymmetricEigen3InternalTrait::new_sym(s);
        assert!(max_ulp_diff_v3(e.eigenvalues, v3i(0, 0, 3)) <= 2);
        // 1/sqrt(3) through `fixed`'s normalisation path (it has no `inv_sqrt`).
        let third = norm3_wide(Real::one(), Real::one(), Real::one()).recip().mul(Real::one());
        let c3 = e.eigenvectors.column3();
        assert!(max_ulp_diff_v3(c3.abs(), Vector3 { x: third, y: third, z: third }) <= 4);
        assert!(orthonormality_error(e.eigenvectors) <= 16);
        assert!(residual_error(s, e) <= 8);
    }

    #[test]
    fn test_new_block_diagonal_case_is_exact() {
        // [[5, 2, 0], [2, 2, 0], [0, 0, 4]]: the 2x2 block has eigenvalues 1 and 6.
        let s = s3i((5, 2, 0, 2, 0, 4));
        let e = SymmetricEigen3InternalTrait::new_sym(s);
        assert!(e.eigenvalues == v3i(1, 4, 6));
        // The eigenvectors of the block are irrational, so `det` is only +1 up to their rounding.
        assert!(ulp_diff(e.eigenvectors.determinant(), Real::one()) <= 4);
        assert!(residual_error(s, e) <= 8);
    }

    #[test]
    fn test_canonical_sign_picks_the_dominant_component() {
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(-3, 1, 2)) == v3i(3, -1, -2));
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(1, -3, 2)) == v3i(-1, 3, -2));
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(1, 2, -3)) == v3i(-1, -2, 3));
        // Ties go to the earlier component.
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(-1, 1, 1)) == v3i(1, -1, -1));
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(0, -1, 1)) == v3i(0, 1, -1));
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(0, 0, 0)) == v3i(0, 0, 0));
    }

    #[test]
    fn test_new_reads_the_lower_triangle() {
        // Lower triangle [[5, ., .], [2, 2, .], [0, 0, 4]]; the upper entries -9 are ignored, like
        // upstream.
        let m = Matrix3Trait::new(
            int(5), int(-9), int(-9), int(2), int(2), int(-9), int(0), int(0), int(4),
        );
        let e = SymmetricEigen3Trait::new(m);
        assert!(e.eigenvalues == v3i(1, 4, 6));
        assert!(m.symmetric_eigen().eigenvalues == e.eigenvalues);
        assert!(m.symmetric_eigenvalues() == e.eigenvalues);
        assert!(e.recompose() == e.recompose_sym().to_matrix());
    }

    #[test]
    fn test_eigenvalues_matches_new() {
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3r(a);
            assert!(
                SymmetricEigen3InternalTrait::eigenvalues(
                    s,
                ) == SymmetricEigen3InternalTrait::new_sym(s)
                    .eigenvalues,
            );
        }
    }

    // --- sweep count ---------------------------------------------------------------------------

    #[test]
    fn test_four_sweeps_reach_the_fixed_point() {
        // Every off-diagonal entry is exactly zero after four sweeps, on every oracle vector.
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let j = Jacobi3Impl::<Fixed>::start(s3r(a)).sweep().sweep().sweep().sweep();
            assert!(off_diagonal_error(j) == 0, "four sweeps left a non-zero off-diagonal entry");
        }
    }

    #[test]
    fn test_three_sweeps_do_not_converge() {
        // Evidence for the choice of 4: three sweeps leave a visible off-diagonal residue.
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let j = Jacobi3Impl::<Fixed>::start(s3r(a)).sweep().sweep().sweep();
            worst = core::cmp::max(worst, off_diagonal_error(j) / amax_s3(s3r(a)));
        }
        assert!(worst > 0, "three sweeps already converge: the fourth could be dropped");
    }

    #[test]
    fn test_a_fifth_and_sixth_sweep_change_nothing() {
        // Evidence that 4 is not merely sufficient but the fixed point: more sweeps are identical.
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3r(a);
            let four = SymmetricEigen3InternalTrait::new_sym(s);
            assert!(eigen_sweeps5(s) == four, "a fifth sweep changed the result");
            assert!(eigen_sweeps6(s) == four, "a sixth sweep changed the result");
        }
    }

    #[test]
    fn test_three_sweeps_are_less_accurate() {
        // The worst reconstruction error over the oracle suite, 3 sweeps against 4.
        let mut worst3: u128 = 0;
        let mut worst4: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3r(a);
            let scale = amax_s3(s);
            let e3 = max_ulp_diff_s3(eigen_sweeps3(s).recompose_sym(), s) / scale;
            let e4 = max_ulp_diff_s3(SymmetricEigen3InternalTrait::new_sym(s).recompose_sym(), s)
                / scale;
            worst3 = core::cmp::max(worst3, e3);
            worst4 = core::cmp::max(worst4, e4);
        }
        assert!(worst4 < worst3, "the fourth sweep did not improve the reconstruction");
        assert!(worst4 <= 26 && worst3 > 26, "worst case regressed");
    }

    #[test]
    fn test_renormalisation_improves_orthonormality() {
        let mut with: u128 = 0;
        let mut without: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3r(a);
            let j = Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().sweep();
            with = core::cmp::max(with, orthonormality_error(j.finish().eigenvectors));
            without =
                core::cmp::max(
                    without, orthonormality_error(finish_without_renormalisation(j).eigenvectors),
                );
        }
        assert!(with <= without, "renormalisation made the columns less orthonormal");
        assert!(with <= 16, "worst case regressed");
    }

    // --- oracle --------------------------------------------------------------------------------

    #[test]
    fn test_new_eigenvalues_oracle() {
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = SymmetricEigen3InternalTrait::new_sym(s3r(a)).eigenvalues;
            let e = max_ulp_diff_v3(got, v3t(expected));
            assert!(e <= tol.into(), "eigenvalues off by more than the oracle tolerance");
            worst = core::cmp::max(worst, e);
        }
        // Measured: 603 ulp worst case, against an oracle tolerance of up to 12 652.
        assert!(worst <= 603, "worst case regressed");
    }

    #[test]
    fn test_new_eigenvalues_oracle_spd() {
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_spd_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = SymmetricEigen3InternalTrait::new_sym(s3r(a)).eigenvalues;
            let e = max_ulp_diff_v3(got, v3t(expected));
            assert!(e <= tol.into(), "eigenvalues off by more than the oracle tolerance");
            worst = core::cmp::max(worst, e);
        }
        assert!(worst <= 264, "worst case regressed");
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle() {
        let mut worst_rec: u128 = 0;
        let mut worst_orth: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3r(a);
            let e = SymmetricEigen3InternalTrait::new_sym(s);
            assert!(ulp_diff(e.eigenvectors.determinant(), Real::one()) <= 4);
            let orth = orthonormality_error(e.eigenvectors);
            assert!(orth <= 32, "columns are not orthonormal");
            let rec = max_ulp_diff_s3(e.recompose_sym(), s) / amax_s3(s);
            assert!(rec <= 32, "reconstruction is off");
            assert!(e.recompose() == e.recompose_sym().to_matrix());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured worst cases over the 30 vectors.
        assert!(worst_rec <= 26 && worst_orth <= 16, "worst case regressed");
    }

    #[test]
    fn test_new_recompose_oracle_spd() {
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_spd_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3r(a);
            let e = SymmetricEigen3InternalTrait::new_sym(s);
            assert!(max_ulp_diff_s3(e.recompose_sym(), s) / amax_s3(s) <= 32);
            assert!(orthonormality_error(e.eigenvectors) <= 32);
            assert!(residual_error(s, e) <= 32 * amax_s3(s));
        }
    }

    #[test]
    fn test_new_eigenvectors_match_the_matrix_form() {
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let (e, f) = (
                SymmetricEigen3Trait::new(m3(a)), SymmetricEigen3InternalTrait::new_sym(s3r(a)),
            );
            assert!(e.eigenvalues == f.eigenvalues && e.eigenvectors == f.eigenvectors);
        }
    }

    // --- overflow ------------------------------------------------------------------------------

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_new_overflow_panics() {
        let s = black_box(
            SymMatrix3 {
                m11: Real::<Fixed>::max_value().unwrap(),
                m12: Real::max_value().unwrap(),
                m13: Real::max_value().unwrap(),
                m22: Real::max_value().unwrap(),
                m23: Real::max_value().unwrap(),
                m33: Real::max_value().unwrap(),
            },
        );
        SymmetricEigen3InternalTrait::new_sym(s);
    }

    // --- gas -----------------------------------------------------------------------------------

    /// A generic symmetric matrix: no off-diagonal entry is zero, so no rotation is skipped.
    fn bench_input() -> SymMatrix3<Fixed> {
        SymMatrix3 {
            m11: fx(0x2c0000000),
            m12: fx(-0x180000000),
            m13: fx(0x90000000),
            m22: fx(0x140000000),
            m23: fx(0x1e0000000),
            m33: fx(-0x240000000),
        }
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__baseline() {
        let _s = black_box(bench_input());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__jacobi_3_sweeps() {
        let s = black_box(bench_input());
        let d = eigen_sweeps3(s);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__jacobi_4_sweeps() {
        let s = black_box(bench_input());
        let d = SymmetricEigen3InternalTrait::new_sym(s);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    /// The public entry point, `SymmetricEigen3::new` on a full `Matrix3` (lower triangle
    /// read): an inlined wrapper around the kernel measured by `new__jacobi_4_sweeps`, which it
    /// should match (WP 8.0).
    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new_matrix__baseline() {
        let _m = black_box(bench_input().to_matrix());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new_matrix__public() {
        let m = black_box(bench_input().to_matrix());
        let d = SymmetricEigen3Trait::new(m);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__jacobi_5_sweeps() {
        let s = black_box(bench_input());
        let d = eigen_sweeps5(s);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__jacobi_6_sweeps() {
        let s = black_box(bench_input());
        let d = eigen_sweeps6(s);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__no_renormalisation() {
        let s = black_box(bench_input());
        let j = Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().sweep();
        let d = finish_without_renormalisation(j);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__diagonal_input() {
        // Every rotation is skipped: the cost of the early exits alone.
        let s = black_box(s3i((7, 0, 0, -2, 0, 1)));
        let d = SymmetricEigen3InternalTrait::new_sym(s);
        assert!(d.eigenvalues == v3i(-2, 1, 7));
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_eigenvalues__baseline() {
        let _s = black_box(bench_input());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_eigenvalues__without_eigenvectors() {
        let s = black_box(bench_input());
        let v = SymmetricEigen3InternalTrait::eigenvalues(s);
        assert!(v.x <= v.y && v.y <= v.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_eigenvalues__via_new() {
        let s = black_box(bench_input());
        let v = SymmetricEigen3InternalTrait::new_sym(s).eigenvalues;
        assert!(v.x <= v.y && v.y <= v.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_sweep__baseline() {
        let _j = black_box(Jacobi3Impl::<Fixed>::start(bench_input()));
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_sweep__three_rotations() {
        let j = black_box(Jacobi3Impl::<Fixed>::start(bench_input()));
        let j = j.sweep();
        // `m12` was zeroed by the first rotation and refilled by the next two.
        assert!(j.s.m12 != Real::zero());
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_sweep__three_rotations_without_eigenvectors() {
        let s = black_box(bench_input());
        let s = Jacobi3Impl::<Fixed>::sweep_s(s);
        assert!(s.m12 != Real::zero());
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_recompose__baseline() {
        let _d = black_box(SymmetricEigen3InternalTrait::new_sym(bench_input()));
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_recompose__quadform() {
        let d = black_box(SymmetricEigen3InternalTrait::new_sym(bench_input()));
        let s = d.recompose_sym();
        assert!(s.m11 != Real::zero());
    }
}
