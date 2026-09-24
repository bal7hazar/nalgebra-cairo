//! Unit tests of `Qr4`: exact cases, the identities and the oracle vectors of `tools/oracle`.
//!
//! Moved from `crates/nalgebra/src/linalg/qr/qr4.cairo (inline `mod tests`)` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::matrix4::{Matrix4, Matrix4Trait};
use nalgebra::base::vector4::Vector4;
use nalgebra::linalg::qr::qr4::{Matrix4QrTrait, Qr4, Qr4Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Qr4PartialEq, excess, m4, max_abs_m4, max_abs_v4, max_ulp_diff4, max_ulp_diff_v4, oracle_tol,
    v4t,
};
use simba::scalar::Real;
use crate::qr::oracle_qr4 as oracle;

/// An oracle `unit` 4x4 case: the benchmark input.
fn a_bench() -> Matrix4<Fixed> {
    m4(
        [
            [-151519325, -831704190, -37020683, -258362626],
            [-309967323, -174899491, -990814222, 1188862699],
            [526702394, 89514843, 1516307715, 154933714],
            [1138644662, -23815325, 661466655, 61972869],
        ],
    )
}

/// Its right-hand side, from `qr4_solve`.
fn b_bench() -> Vector4<Fixed> {
    v4t((2614746463, 2804691241, -3498193820, 3365132750))
}

/// `a_bench()` already factored.
fn f_bench() -> Qr4<Fixed> {
    Qr4Trait::new(a_bench())
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
    let mut cases = oracle::qr4_q_r_cases();
    let (mut worst_q, mut worst_r, mut worst_ex) = (0, 0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, q, r, tol) = *case;
        let f = Qr4Trait::new(m4(a));
        let (eq, er) = (m4(q), m4(r));
        let (dq, dr) = (max_ulp_diff4(f.q(), eq), max_ulp_diff4(f.r(), er));
        worst_ex = core::cmp::max(worst_ex, excess(dq, oracle_tol(max_abs_m4(eq), tol)));
        worst_ex = core::cmp::max(worst_ex, excess(dr, oracle_tol(max_abs_m4(er), tol)));
        worst_q = core::cmp::max(worst_q, dq);
        worst_r = core::cmp::max(worst_r, dr);
    }
    assert!(
        (worst_q, worst_r, worst_ex) == (84, 15, 0), "regressed: {worst_q} {worst_r} {worst_ex}",
    );
}

#[test]
fn test_solve_oracle() {
    let mut cases = oracle::qr4_solve_cases();
    let (mut worst, mut worst_ex) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let x = Qr4Trait::new(m4(a)).solve(v4t(b)).unwrap();
        let e = v4t(expected);
        let err = max_ulp_diff_v4(x, e);
        worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v4(e), tol)));
        worst = core::cmp::max(worst, err);
    }
    assert!((worst, worst_ex) == (940, 0), "regressed: {worst} {worst_ex}");
}

#[test]
fn test_try_inverse_product_is_identity_oracle() {
    let mut cases = oracle::qr4_q_r_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _, _, _) = *case;
        let inv = Qr4Trait::new(m4(a)).try_inverse().unwrap();
        let id = Matrix4Trait::identity();
        worst = core::cmp::max(worst, max_ulp_diff4(m4(a) * inv, id));
        worst = core::cmp::max(worst, max_ulp_diff4(inv * m4(a), id));
    }
    // Measured residual of `A A^-1 - I` and `A^-1 A - I` over the 30 well-conditioned vectors.
    assert!(worst == 128, "regressed: {worst}");
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_try_inverse_overflow_panics() {
    // 2^-32 * I: every diagonal entry of R is 1 raw unit, so the inverse is 2^32 * I.
    let _ = black_box(Matrix4Trait::from_diagonal_element(Fixed { raw: 1 })).qr().try_inverse();
}

// --- gas benchmarks --------------------------------------------------------

#[test]
#[inline(never)]
fn bench_qr4_new__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr4_new__gram_schmidt() {
    let a = black_box(a_bench());
    let e = black_box(true);
    let f = Qr4Trait::new(a);
    assert!((f.r.m11 > Real::zero() && f.r.m44 > Real::zero()) == e);
}

#[test]
#[inline(never)]
fn bench_qr4_factors__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr4_factors__unpack() {
    let f = black_box(f_bench());
    let e = black_box(true);
    let (q, r) = f.unpack();
    assert!((q.m11 != Real::zero() && r.m11 != Real::zero()) == e);
}

#[test]
#[inline(never)]
fn bench_qr4_is_invertible__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr4_is_invertible__pivots() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.is_invertible() == e);
}

#[test]
#[inline(never)]
fn bench_qr4_solve__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr4_solve__substitution() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(true);
    assert!(f.solve(b).is_some() == e);
}

#[test]
#[inline(never)]
fn bench_qr4_try_inverse__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr4_try_inverse__substitution() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.try_inverse().is_some() == e);
}

#[test]
#[inline(never)]
fn bench_qr4_determinant__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}
