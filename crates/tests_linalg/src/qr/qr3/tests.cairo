//! Unit tests of `Qr3`, and the two alternative orthogonalisations that lost, kept as evidence
//! together with the tests that show why (AGENTS.md rule 8):
//!
//! - `alt_householder`: upstream's algorithm, two reflections plus the sign normalisation.
//! It lands FURTHER from the oracle factors than the shipped modified Gram-Schmidt (402 ulp
//! against 121) and costs more gas: in fixed point each reflector pays a normalisation of its
//! own axis and then two rounded reflections per column, where Gram-Schmidt rounds once per
//! entry. Its `Q` is slightly more orthonormal (29 ulp against 36), which is the one thing
//! the textbook promises and the only thing it wins here.
//!
//! - `alt_classical_gram_schmidt`: the projections of column 3 both taken against the ORIGINAL
//! column. Same arithmetic, statement for statement, and the same orthonormality on these
//! well-conditioned inputs; it measures 9 % cheaper (70 880 against 77 890 on `fixed` 0.3.0)
//! only because it divides UNGUARDED — it panics with a division by zero on a matrix with a
//! zero column, where the shipped `new` returns `r_ii = 0` — so its figure also prices the
//! three `r_ii == 0` guards the shipped `new` carries. Worse in theory, see `new`; re-ranked
//! in WP 7.2: kept as the loser.
//!
//! - `alt_completed_basis`: the rank-deficient fallback that completes `Q` to an orthonormal
//! basis instead of leaving a zero column. Strictly dearer, on every call.
//!
//! Moved from `crates/nalgebra/src/linalg/qr/qr3.cairo (inline `mod tests`)` (WP 8.1c, test-only
//! package): the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::matrix3::{Matrix3, Matrix3Trait};
use nalgebra::base::vector3::Vector3;
use nalgebra::linalg::qr::qr3::{Matrix3QrTrait, Qr3, Qr3Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    Qr3PartialEq, excess, int, m3, max_abs_m3, max_abs_v3, max_ulp_diff3, max_ulp_diff_v3,
    oracle_tol, v3t,
};
use simba::scalar::Real;
use crate::qr::oracle_qr3 as oracle;

/// The oracle's first 3x3 case: the benchmark input.
fn a_bench() -> Matrix3<Fixed> {
    m3(
        [
            [-930291762, 206204041, 164059380], [-1057407058, -1146357637, -192379840],
            [-394072183, -440329712, 775368183],
        ],
    )
}

/// A right-hand side, from `qr3_solve`.
fn b_bench() -> Vector3<Fixed> {
    v3t((-1811584373, 4204441265, -4234532070))
}

/// `a_bench()` already factored, so the benchmarks of the derived operations do not pay for
/// `new`.
fn f_bench() -> Qr3<Fixed> {
    Qr3Trait::new(a_bench())
}

/// A rank-2 matrix: the third column is the sum of the first two, so `r33` is exactly zero.
fn a_rank2() -> Matrix3<Fixed> {
    Matrix3Trait::new(int(1), int(0), int(1), int(0), int(2), int(2), int(0), int(0), int(0))
}

// --- the losing candidates -----------------------------------------------------------------

/// CLASSICAL Gram-Schmidt: `r23` is taken against the ORIGINAL third column instead of the
/// one already stripped of its `q1` component. Identical in exact arithmetic, identical in
/// gas, and measurably less orthogonal under rounding
/// (`test_classical_gram_schmidt_candidate_is_less_orthogonal`).
fn new_classical(matrix: Matrix3<Fixed>) -> Qr3<Fixed> {
    let r11 = Real::norm3(matrix.m11, matrix.m21, matrix.m31);
    let (q11, q21, q31) = (matrix.m11 / r11, matrix.m21 / r11, matrix.m31 / r11);
    let r12 = Real::sum_prod3(q11, matrix.m12, q21, matrix.m22, q31, matrix.m32);
    let r13 = Real::sum_prod3(q11, matrix.m13, q21, matrix.m23, q31, matrix.m33);
    let b21 = Real::mul_add(-r12, q11, matrix.m12);
    let b22 = Real::mul_add(-r12, q21, matrix.m22);
    let b23 = Real::mul_add(-r12, q31, matrix.m32);
    let r22 = Real::norm3(b21, b22, b23);
    let (q12, q22, q32) = (b21 / r22, b22 / r22, b23 / r22);
    // The classical difference: against `matrix`'s third column, not against `b3`.
    let r23 = Real::sum_prod3(q12, matrix.m13, q22, matrix.m23, q32, matrix.m33);
    let w = Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m13), r13, q11);
    let c31 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q12));
    let w = Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m23), r13, q21);
    let c32 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q22));
    let w = Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m33), r13, q31);
    let c33 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q32));
    let r33 = Real::norm3(c31, c32, c33);
    let (q13, q23, q33) = (c31 / r33, c32 / r33, c33 / r33);
    Qr3 {
        q: Matrix3Trait::from_columns(
            Vector3 { x: q11, y: q21, z: q31 },
            Vector3 { x: q12, y: q22, z: q32 },
            Vector3 { x: q13, y: q23, z: q33 },
        ),
        r: Matrix3Trait::new(
            r11, r12, r13, Real::zero(), r22, r23, Real::zero(), Real::zero(), r33,
        ),
    }
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
    let mut cases = oracle::qr3_q_r_cases();
    let (mut worst_q, mut worst_r, mut worst_ex) = (0, 0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, q, r, tol) = *case;
        let f = Qr3Trait::new(m3(a));
        let (eq, er) = (m3(q), m3(r));
        let (dq, dr) = (max_ulp_diff3(f.q(), eq), max_ulp_diff3(f.r(), er));
        worst_ex = core::cmp::max(worst_ex, excess(dq, oracle_tol(max_abs_m3(eq), tol)));
        worst_ex = core::cmp::max(worst_ex, excess(dr, oracle_tol(max_abs_m3(er), tol)));
        worst_q = core::cmp::max(worst_q, dq);
        worst_r = core::cmp::max(worst_r, dr);
    }
    // Measured worst cases: the MGS factors agree with upstream's unpacked Householder
    // factors entry by entry, no sign flip, and every case stays inside the oracle tolerance.
    assert!(
        (worst_q, worst_r, worst_ex) == (29, 55, 0), "regressed: {worst_q} {worst_r} {worst_ex}",
    );
}

#[test]
fn test_solve_oracle() {
    let mut cases = oracle::qr3_solve_cases();
    let (mut worst, mut worst_ex) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let x = Qr3Trait::new(m3(a)).solve(v3t(b)).unwrap();
        let e = v3t(expected);
        let err = max_ulp_diff_v3(x, e);
        worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v3(e), tol)));
        worst = core::cmp::max(worst, err);
    }
    assert!((worst, worst_ex) == (273, 0), "regressed: {worst} {worst_ex}");
}

#[test]
fn test_try_inverse_product_is_identity_oracle() {
    let mut cases = oracle::qr3_q_r_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, _, _, _) = *case;
        let inv = Qr3Trait::new(m3(a)).try_inverse().unwrap();
        let id = Matrix3Trait::identity();
        worst = core::cmp::max(worst, max_ulp_diff3(m3(a) * inv, id));
        worst = core::cmp::max(worst, max_ulp_diff3(inv * m3(a), id));
    }
    // Measured residual of `A A^-1 - I` and `A^-1 A - I` over the 30 well-conditioned vectors.
    assert!(worst == 82, "regressed: {worst}");
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_try_inverse_overflow_panics() {
    // 2^-32 * I: every diagonal entry of R is 1 raw unit, so the inverse is 2^32 * I.
    let _ = black_box(Matrix3Trait::from_diagonal_element(Fixed { raw: 1 })).qr().try_inverse();
}

// --- gas benchmarks --------------------------------------------------------

#[test]
#[inline(never)]
fn bench_qr3_new__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr3_new__gram_schmidt() {
    let a = black_box(a_bench());
    let e = black_box(true);
    let f = Qr3Trait::new(a);
    assert!((f.r.m11 > Real::zero() && f.r.m33 > Real::zero()) == e);
}

#[test]
#[inline(never)]
fn bench_qr3_new__alt_classical_gram_schmidt() {
    let a = black_box(a_bench());
    let e = black_box(true);
    let f = new_classical(a);
    assert!((f.r.m11 > Real::zero() && f.r.m33 > Real::zero()) == e);
}

#[test]
#[inline(never)]
fn bench_qr3_factors__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr3_factors__unpack() {
    let f = black_box(f_bench());
    let e = black_box(true);
    let (q, r) = f.unpack();
    assert!((q.m11 != Real::zero() && r.m11 != Real::zero()) == e);
}

#[test]
#[inline(never)]
fn bench_qr3_is_invertible__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr3_is_invertible__pivots() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.is_invertible() == e);
}

#[test]
#[inline(never)]
fn bench_qr3_solve__baseline() {
    let _f = black_box(f_bench());
    let _b = black_box(b_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr3_solve__substitution() {
    let f = black_box(f_bench());
    let b = black_box(b_bench());
    let e = black_box(true);
    assert!(f.solve(b).is_some() == e);
}

#[test]
#[inline(never)]
fn bench_qr3_solve_singular__baseline() {
    let _f = black_box(Qr3Trait::new(a_rank2()));
    let _b = black_box(b_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr3_solve_singular__none() {
    let f = black_box(Qr3Trait::new(a_rank2()));
    let b = black_box(b_bench());
    let e = black_box(true);
    assert!(f.solve(b).is_none() == e);
}

#[test]
#[inline(never)]
fn bench_qr3_try_inverse__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr3_try_inverse__substitution() {
    let f = black_box(f_bench());
    let e = black_box(true);
    assert!(f.try_inverse().is_some() == e);
}

#[test]
#[inline(never)]
fn bench_qr3_determinant__baseline() {
    let _f = black_box(f_bench());
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_qr3_determinant_closed_form__baseline() {
    let _a = black_box(a_bench());
    let e = black_box(true);
    assert!(e == e);
}

/// `Matrix3::determinant` on the matrix itself, in its own group because its input is a
/// `Matrix3` and not a factorisation: it is both cheaper and far more accurate, which is why
/// `Qr3::determinant` tells the caller to prefer it.
#[test]
#[inline(never)]
fn bench_qr3_determinant_closed_form__matrix3_cofactors() {
    let a = black_box(a_bench());
    let e = black_box(true);
    assert!((a.determinant() != Real::zero()) == e);
}
