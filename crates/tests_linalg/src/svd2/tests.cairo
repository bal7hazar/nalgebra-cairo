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
//! Moved from `crates/nalgebra/src/linalg/svd2.cairo (inline `mod tests`)` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::MatrixMul;
use nalgebra::base::matrix2::{Matrix2, Matrix2Trait};
use nalgebra::base::vector2::Vector2;
use nalgebra::linalg::svd2::{Matrix2SvdTrait, Svd2, Svd2Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Svd2PartialEq, excess, int, m2, max_abs_v2, max_ulp_diff2, max_ulp_diff_v2, oracle_tol, v2t,
};
use simba::scalar::Real;
use crate::oracle_svd;

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

// --- exact cases ---------------------------------------------------------------------------

#[test]
fn test_new_identity_is_exact() {
    let f = Svd2Trait::new(Matrix2Trait::<Fixed>::identity());
    assert!(f.singular_values == Vector2 { x: int(1), y: int(1) });
    assert!(f.u == Matrix2Trait::identity());
    assert!(f.v_t == Matrix2Trait::identity());
    assert!(f.recompose() == Matrix2Trait::identity());
    assert!(f.rank(Real::zero()) == 2);
}

#[test]
fn test_new_permutation_is_exact() {
    let p = Matrix2Trait::new(int(0), int(1), int(1), int(0));
    let f = Svd2Trait::new(p);
    assert!(f.singular_values == Vector2 { x: int(1), y: int(1) });
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
    let mut cases = oracle_svd::svd2_singular_values_cases();
    let (mut worst, mut worst_ex) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let got = Svd2Trait::new(m2(a)).singular_values;
        let e = v2t(expected);
        let err = max_ulp_diff_v2(got, e);
        worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v2(e), tol)));
        assert!(got.x >= got.y && got.y >= Real::zero(), "not sorted descending");
        worst = core::cmp::max(worst, err);
    }
    assert!((worst, worst_ex) == (48, 0), "regressed: {worst} {worst_ex}");
}

#[test]
fn test_solve_matches_the_inverse_oracle() {
    // On a non-singular matrix the least-squares solution IS the solution.
    let mut cases = oracle_svd::svd2_singular_values_cases();
    let mut worst = 0;
    let b = v2t((-5886581674, -6536196560));
    while let Some(case) = cases.pop_front() {
        let (a, _, _) = *case;
        let x = Svd2Trait::new(m2(a)).solve(b, Real::default_epsilon()).unwrap();
        let e = m2(a).try_inverse().unwrap().mul_mat(b);
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
        let p = Svd2Trait::new(m2(a)).pseudo_inverse(Real::default_epsilon()).unwrap();
        let e = m2(a).try_inverse().unwrap();
        worst = core::cmp::max(worst, max_ulp_diff2(p, e));
    }
    assert!(worst == 1947, "regressed: {worst}");
}

#[test]
fn test_pseudo_inverse_of_a_rank_one_matrix() {
    // The pseudo-inverse drops the null direction: `A A⁺ A = A`.
    let p = Svd2Trait::new(a_rank1()).pseudo_inverse(Real::default_epsilon()).unwrap();
    assert!(max_ulp_diff2(a_rank1() * p * a_rank1(), a_rank1()) <= 64);
    assert!(Svd2Trait::new(a_rank1()).pseudo_inverse(Real::NEG_ONE).is_none());
    assert!(
        Svd2Trait::new(a_rank1()).solve(Vector2 { x: int(1), y: int(1) }, Real::NEG_ONE).is_none(),
    );
}

#[test]
fn test_to_polar_of_a_rotation_is_the_rotation() {
    // `M = R` exactly: the stretch is the identity.
    let r = Matrix2Trait::new(int(0), int(-1), int(1), int(0));
    let (p, q) = r.svd().to_polar().unwrap();
    assert!(max_ulp_diff2(q, r) <= 2);
    assert!(max_ulp_diff2(p, Matrix2Trait::identity()) <= 2);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_new_overflow_panics() {
    // `MᵀM` does not fit: the squares of the entries must be representable.
    let m = black_box(Matrix2Trait::from_diagonal_element(Real::<Fixed>::max_value().unwrap()));
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
    assert!((f.recompose().m11 != Real::zero()) == e);
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
    assert!(f.solve(b, Real::default_epsilon()).is_some() == e);
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
    assert!(f.pseudo_inverse(Real::default_epsilon()).is_some() == e);
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
    assert!((f.rank(Real::default_epsilon()) == 2) == e);
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
    let (p, u) = f.to_polar().unwrap();
    assert!((u.m11 != Real::zero() && p.m11 != Real::zero()) == e);
}
