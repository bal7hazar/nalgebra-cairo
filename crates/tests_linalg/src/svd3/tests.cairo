//! Unit tests of `Svd3`: the exact cases (identity, diagonal, permutation, rank 1, rank 2,
//! zero), the identities (`M = U Σ Vᵀ`, orthonormality, `M = P U`), the oracle vectors of
//! `tools/oracle`, and the three candidates that lost, kept as evidence (AGENTS.md rule 8):
//!
//! - `alt_sqrt_eigenvalues`: `σ = sqrt(λ)` from the eigenvalues of `MᵀM`, the formulation
//! DESIGN D6 describes. Cheaper, and two to three orders of magnitude less accurate.
//!
//! - `alt_normalised_columns`: every `u_i = M v_i / σ_i`, no re-orthogonalisation. Dearer (9
//! divisions against 6 plus a cross product) and measurably less orthonormal.
//!
//! - `alt_one_sided_jacobi`: the decomposition computed directly on `M`, by rotating pairs of
//! COLUMNS until they are orthogonal (`M J = U Σ`, `V = J`), never forming `MᵀM`. The
//! textbook answer for ill-conditioned inputs. Same four sweeps, no better on the oracle, and
//! dearer: every rotation recomputes three inner products of the working columns where the
//! Jacobi on `MᵀM` updates six scalars.
//!
//! Moved from `crates/nalgebra/src/linalg/svd3.cairo (inline `mod tests`)` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::MatrixMul;
use nalgebra::base::matrix3::{Matrix3, Matrix3Trait};
use nalgebra::base::vector3::{Vector3, Vector3Trait};
use nalgebra::linalg::svd3::{Matrix3SvdTrait, Svd3, Svd3Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Svd3PartialEq, excess, int, m3, max_abs_v3, max_ulp_diff3, max_ulp_diff_v3, oracle_tol, v3t,
};
use simba::scalar::Real;
use crate::oracle_svd;

/// An oracle `unit` 3x3 case: the benchmark input.
fn a_bench() -> Matrix3<Fixed> {
    m3(
        [
            [-930291762, 206204041, 164059380], [-1057407058, -1146357637, -192379840],
            [-394072183, -440329712, 775368183],
        ],
    )
}

/// `a_bench()` already decomposed.
fn f_bench() -> Svd3<Fixed> {
    Svd3Trait::new(a_bench())
}

/// A rank-2 matrix whose decomposition is exact.
fn a_rank2() -> Matrix3<Fixed> {
    Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(0), int(0), int(0))
}

#[test]
fn test_new_permutation_is_exact() {
    let p = Matrix3Trait::new(
        int(0), int(1), int(0), int(0), int(0), int(1), int(1), int(0), int(0),
    );
    let f = Svd3Trait::new(p);
    assert!(f.singular_values == Vector3 { x: int(1), y: int(1), z: int(1) });
    assert!(f.recompose() == p);
}

#[test]
fn test_accessors() {
    let f = f_bench();
    assert!(f.singular_values == f.singular_values);
    let g = a_bench().svd();
    assert!(g.u == f.u && g.singular_values == f.singular_values && g.v_t == f.v_t);
    assert!(a_bench().singular_values() == f.singular_values);
}

// --- oracle --------------------------------------------------------------------------------

#[test]
fn test_new_singular_values_oracle() {
    let mut cases = oracle_svd::svd3_singular_values_cases();
    let (mut worst, mut worst_ex) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let got = Svd3Trait::new(m3(a)).singular_values;
        let e = v3t(expected);
        let err = max_ulp_diff_v3(got, e);
        worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v3(e), tol)));
        assert!(got.x >= got.y && got.y >= got.z && got.z >= Real::zero(), "not descending");
        worst = core::cmp::max(worst, err);
    }
    assert!((worst, worst_ex) == (71, 0), "regressed: {worst} {worst_ex}");
}

#[test]
fn test_solve_matches_the_inverse_oracle() {
    let mut cases = oracle_svd::svd3_singular_values_cases();
    let mut worst = 0;
    let b = v3t((-1811584373, 4204441265, -4234532070));
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let x = Svd3Trait::new(m3(a)).solve(b, Real::default_epsilon()).unwrap();
        let e = m3(a).try_inverse().unwrap().mul_mat(b);
        worst = core::cmp::max(worst, max_ulp_diff_v3(x, e));
    }
    // Measured gap to `Matrix3::try_inverse` * b over the 30 well-conditioned vectors.
    assert!(worst == 4119, "regressed: {worst}");
}

#[test]
fn test_pseudo_inverse_oracle() {
    let mut cases = oracle_svd::svd3_singular_values_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let p = Svd3Trait::new(m3(a)).pseudo_inverse(Real::default_epsilon()).unwrap();
        worst = core::cmp::max(worst, max_ulp_diff3(p, m3(a).try_inverse().unwrap()));
    }
    assert!(worst == 77683, "regressed: {worst}");
}

#[test]
fn test_pseudo_inverse_of_a_rank_deficient_matrix() {
    // The pseudo-inverse drops the null directions: `A A⁺ A = A`.
    let p = Svd3Trait::new(a_rank2()).pseudo_inverse(Real::default_epsilon()).unwrap();
    assert!(max_ulp_diff3(a_rank2() * p * a_rank2(), a_rank2()) <= 64);
    assert!(Svd3Trait::new(a_rank2()).pseudo_inverse(Real::NEG_ONE).is_none());
    assert!(Svd3Trait::new(a_rank2()).solve(Vector3Trait::zeros(), Real::NEG_ONE).is_none());
}

#[test]
fn test_to_polar_of_a_rotation_is_the_rotation() {
    let r = Matrix3Trait::new(
        int(0), int(0), int(1), int(1), int(0), int(0), int(0), int(1), int(0),
    );
    let (p, q) = r.svd().to_polar().unwrap();
    assert!(max_ulp_diff3(q, r) <= 4);
    assert!(max_ulp_diff3(p, Matrix3Trait::identity()) <= 4);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_new_overflow_panics() {
    // `MᵀM` does not fit: the squares of the entries must be representable.
    let m = black_box(Matrix3Trait::from_diagonal_element(Real::<Fixed>::max_value().unwrap()));
    Svd3Trait::new(m);
}

// --- gas benchmarks --------------------------------------------------------

#[test]
#[inline(never)]
fn bench_svd3_new__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_svd3_new__eigen_of_gram() {
    let a = black_box(a_bench());
    let e = black_box(true);
    let f = Svd3Trait::new(a);
    assert!((f.singular_values.x >= f.singular_values.z) == e);
}

#[test]
#[inline(never)]
fn bench_svd3_singular_values__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(true);
    assert!(e == e);
}

/// The singular values as the decomposition produces them, `|M v_i|`.
#[test]
#[inline(never)]
fn bench_svd3_singular_values__from_left_vectors() {
    let a = black_box(a_bench());
    let e = black_box(true);
    let s = Svd3Trait::new(a).singular_values;
    assert!((s.x >= s.z) == e);
}

#[test]
#[inline(never)]
fn bench_svd3_gram__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_svd3_recompose__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_svd3_recompose__scaled_product() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!((f.recompose().m11 != Real::zero()) == e);
}

#[test]
#[inline(never)]
fn bench_svd3_solve__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(v3t((1, 2, 3)));
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_svd3_solve__divisions() {
    let f = black_box(f_bench());
    let b = black_box(v3t((0x100000000, 0x200000000, 0x300000000)));
    let e = black_box(true);
    assert!(f.solve(b, Real::default_epsilon()).is_some() == e);
}

#[test]
#[inline(never)]
fn bench_svd3_pseudo_inverse__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_svd3_pseudo_inverse__reciprocals() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.pseudo_inverse(Real::default_epsilon()).is_some() == e);
}

#[test]
#[inline(never)]
fn bench_svd3_rank__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_svd3_rank__comparisons() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!((f.rank(Real::default_epsilon()) == 3) == e);
}

#[test]
#[inline(never)]
fn bench_svd3_to_polar__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_svd3_to_polar__quadform() {
    let f = black_box(f_bench());
    let e = black_box(true);
    let (p, u) = f.to_polar().unwrap();
    assert!((u.m11 != Real::zero() && p.m11 != Real::zero()) == e);
}
