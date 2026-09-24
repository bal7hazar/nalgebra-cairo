//! Unit tests of `Qr2`: the exact cases (identity, diagonal, permutation, rank deficient), the
//! identities (`Q R = A`, `QᵀQ = I`, `A A^-1 = I`), and the oracle vectors of `tools/oracle`
//! (upstream nalgebra 0.35 on the same raw inputs, unpacked factors, `r_ii >= 0`).
//!
//! Moved from `crates/nalgebra/src/linalg/qr/qr2.cairo (inline `mod tests`)` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::matrix2::{Matrix2, Matrix2Trait};
use nalgebra::base::vector2::Vector2;
use nalgebra::linalg::qr::qr2::{Matrix2QrTrait, Qr2, Qr2Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Qr2PartialEq, excess, int, m2, max_abs_m2, max_abs_v2, max_ulp_diff2, max_ulp_diff_v2,
    oracle_tol, v2t,
};
use simba::scalar::Real;
use crate::qr::oracle_qr2 as oracle;

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

/// A rank-1 matrix whose orthogonalisation is exact: the second column is three times the
/// first, so `r22` is exactly zero.
fn a_rank1() -> Matrix2<Fixed> {
    Matrix2Trait::new(int(2), int(6), int(0), int(0))
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
        (worst_q, worst_r, worst_ex) == (25, 15, 0), "regressed: {worst_q} {worst_r} {worst_ex}",
    );
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
