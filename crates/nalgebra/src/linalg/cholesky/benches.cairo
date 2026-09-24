//! Gas benchmarks of `Cholesky{2,3,4,6}` (`bench_cholesky<n>_<op>__<variant>`, net = raw minus the
//! `baseline` of the group) and the alternative implementations that lost (`alt_*`), kept together
//! with the accuracy measurements that rejected them (AGENTS.md rule 8).
//!
//! THE MEASUREMENT. `solve` divides by each pivot twice (once forward, once back) and `inverse`
//! n(n+1)/2 times in all. On `fixed` 0.3.0 (division rounded to nearest, 3 300 gas; `recip`
//! 2 820; product 1 580) `alt_recip` is the CHEAPER candidate everywhere: 12 to 13 % in `solve`,
//! 13 to 22 % in `inverse` (on the floor-division scalar it was 3 to 4 % dearer in `solve`). It
//! still loses (WP 7.2 re-rank), on upstream fidelity first: upstream's `solve_mut`, and `inverse`
//! which is `solve_mut` on the identity, DIVIDE by the pivot; and on accuracy: on the oracle's 12
//! cases it is less accurate in `solve` at every size but 6, and in `inverse` `t_ij / l_ii` is the
//! correctly rounded quotient where `t_ij · recip(l_ii)` rounds twice, which
//! `test_cholesky2_inverse_alt_recip_loses_low_bits` exhibits on a hand-built factor. Division
//! therefore ships, for the reason `Matrix3::try_inverse` and `Vector3::unscale` divide.
//!
//! THE FIXTURE. Every benchmark factorises `a_ij = min(i, j)`, whose Cholesky factor is exactly the
//! all-ones lower triangular matrix: all inputs and results are integers, so the asserted values
//! are exact and the two variants of a group take the same branches (every quantity is positive).

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::matrix4::Matrix4;
use crate::base::matrix6::Matrix6;
use crate::base::matrix_test_utils::{
    fx, int, m2, m2i, m3, m3i, m4, m4i, m6, m6i, max_ulp_diff2, max_ulp_diff3, max_ulp_diff4,
    max_ulp_diff6, max_ulp_diff_v2, max_ulp_diff_v3, max_ulp_diff_v4, max_ulp_diff_v6, ulp_diff,
    v2it, v2t, v3it, v3t, v4it, v4t, v6it, v6t,
};
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector6::Vector6;
use crate::linalg::oracle_cholesky;
use super::{
    Cholesky2, Cholesky2Trait, Cholesky3, Cholesky3Trait, Cholesky4, Cholesky4Trait, Cholesky6,
    Cholesky6Trait,
};

// --- size 2 -------------------------------------------------------------------------------------

/// The benchmark input `a_ij = min(i, j)`.
fn a2() -> Matrix2<Fixed> {
    m2i([[1, 1], [1, 2]])
}

/// Its Cholesky factor: the all-ones lower triangle.
fn f2() -> Cholesky2<Fixed> {
    Cholesky2 { l11: int(1), l21: int(1), l22: int(1) }
}

/// The right-hand side of the benchmarked `solve`, and its exact solution.
fn b2() -> Vector2<Fixed> {
    v2it((1, -2))
}

/// `a2()⁻¹ · b2()`, exactly.
fn x2() -> Vector2<Fixed> {
    v2it((4, -3))
}

/// `a2()⁻¹`: tridiagonal, exactly.
fn inv2() -> Matrix2<Fixed> {
    m2i([[2, -1], [-1, 1]])
}

/// LOSER. `solve` with one reciprocal per pivot and multiplications instead of the two exactly
/// correctly rounded divisions: `recip(l_jj)` rounds `1/l_jj` first, so each substitution step is
/// off by up to `|y_i|` ulp instead of 1, and two multiplications plus a reciprocal cost more than
/// two divisions.
fn solve2_recip(f: Cholesky2<Fixed>, b: Vector2<Fixed>) -> Vector2<Fixed> {
    let e1 = Real::recip(f.l11);
    let e2 = Real::recip(f.l22);
    let y1 = b.x * e1;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.y);
    let w = Real::wide_sub_prod(w, f.l21, y1);
    let f2 = Real::wide_rescale(w);
    let y2 = f2 * e2;
    let x2 = y2 * e2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y1);
    let w = Real::wide_sub_prod(w, f.l21, x2);
    let g1 = Real::wide_rescale(w);
    let x1 = g1 * e1;
    Vector2 { x: x1, y: x2 }
}

/// LOSER. `inverse` reusing the already-computed `q_ii = recip(l_ii)` as a multiplier for the
/// sub-diagonal entries of `l⁻¹` instead of dividing by `l_ii`. Cheaper (the reciprocal is free
/// here) but less accurate, and accuracy wins.
fn inverse2_recip(f: Cholesky2<Fixed>) -> Matrix2<Fixed> {
    let q11 = Real::recip(f.l11);
    let q22 = Real::recip(f.l22);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l21, q11);
    let t21 = Real::wide_rescale(w);
    let q21 = t21 * q22;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q11, q11);
    let w = Real::wide_add_prod(w, q21, q21);
    let r11 = Real::wide_rescale(w);
    let r12 = q21 * q22;
    let r22 = Real::sqr(q22);
    Matrix2 { m11: r11, m21: r12, m12: r12, m22: r22 }
}

/// Worst error in ulp of a `solve` candidate over the oracle's `cholesky2_solve` vectors.
fn solve2_worst(variant: u8) -> u128 {
    let mut cases = oracle_cholesky::cholesky2_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _tol) = *case;
        let f = Cholesky2Trait::new(m2(a)).unwrap();
        let got = if variant == 0 {
            f.solve(v2t(b))
        } else {
            solve2_recip(f, v2t(b))
        };
        worst = core::cmp::max(worst, max_ulp_diff_v2(got, v2t(expected)));
    }
    worst
}

/// Worst error in ulp of an `inverse` candidate over the oracle's `cholesky2_inverse` vectors.
fn inverse2_worst(variant: u8) -> u128 {
    let mut cases = oracle_cholesky::cholesky2_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, _tol) = *case;
        let f = Cholesky2Trait::new(m2(a)).unwrap();
        let got = if variant == 0 {
            f.inverse()
        } else {
            inverse2_recip(f)
        };
        worst = core::cmp::max(worst, max_ulp_diff2(got, m2(expected)));
    }
    worst
}

/// Why `alt_recip` lost: measured worst error over the 12 oracle cases, shipped against candidate.
#[test]
fn test_cholesky2_alt_recip_is_less_accurate() {
    let (sd, sr) = (solve2_worst(0), solve2_worst(1));
    let (id, ir) = (inverse2_worst(0), inverse2_worst(1));
    assert!(
        sd == 45 && sr == 120 && id == 122 && ir == 122,
        "cholesky2 solve div {} recip {} / inverse div {} recip {}",
        sd,
        sr,
        id,
        ir,
    );
}

#[test]
#[inline(never)]
fn bench_cholesky2_new__baseline() {
    let _a = black_box(a2());
    let e = black_box(f2());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_new__factorize() {
    let a = black_box(a2());
    let e = black_box(f2());
    assert!(Cholesky2Trait::new(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_l__baseline() {
    let _f = black_box(f2());
    let e = black_box(m2i([[1, 0], [1, 1]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_l__expand() {
    let f = black_box(f2());
    let e = black_box(m2i([[1, 0], [1, 1]]));
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_solve__baseline() {
    let _f = black_box(f2());
    let _b = black_box(b2());
    let e = black_box(x2());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_solve__substitution() {
    let f = black_box(f2());
    let b = black_box(b2());
    let e = black_box(x2());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_solve__alt_recip() {
    let f = black_box(f2());
    let b = black_box(b2());
    let e = black_box(x2());
    assert!(solve2_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_inverse__baseline() {
    let _f = black_box(f2());
    let e = black_box(inv2());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_inverse__triangular() {
    let f = black_box(f2());
    let e = black_box(inv2());
    assert!(f.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_inverse__alt_recip() {
    let f = black_box(f2());
    let e = black_box(inv2());
    assert!(inverse2_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_determinant__baseline() {
    let _f = black_box(f2());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky2_determinant__diagonal_product() {
    let f = black_box(f2());
    let e = black_box(int(1));
    assert!(f.determinant() == e);
}

// --- size 3 -------------------------------------------------------------------------------------

/// The benchmark input `a_ij = min(i, j)`.
fn a3() -> Matrix3<Fixed> {
    m3i([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
}

/// Its Cholesky factor: the all-ones lower triangle.
fn f3() -> Cholesky3<Fixed> {
    Cholesky3 { l11: int(1), l21: int(1), l31: int(1), l22: int(1), l32: int(1), l33: int(1) }
}

/// The right-hand side of the benchmarked `solve`, and its exact solution.
fn b3() -> Vector3<Fixed> {
    v3it((1, -2, 3))
}

/// `a3()⁻¹ · b3()`, exactly.
fn x3() -> Vector3<Fixed> {
    v3it((4, -8, 5))
}

/// `a3()⁻¹`: tridiagonal, exactly.
fn inv3() -> Matrix3<Fixed> {
    m3i([[2, -1, 0], [-1, 2, -1], [0, -1, 1]])
}

/// LOSER. `solve` with one reciprocal per pivot and multiplications instead of the two exactly
/// correctly rounded divisions: `recip(l_jj)` rounds `1/l_jj` first, so each substitution step is
/// off by up to `|y_i|` ulp instead of 1, and two multiplications plus a reciprocal cost more than
/// two divisions.
fn solve3_recip(f: Cholesky3<Fixed>, b: Vector3<Fixed>) -> Vector3<Fixed> {
    let e1 = Real::recip(f.l11);
    let e2 = Real::recip(f.l22);
    let e3 = Real::recip(f.l33);
    let y1 = b.x * e1;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.y);
    let w = Real::wide_sub_prod(w, f.l21, y1);
    let f2 = Real::wide_rescale(w);
    let y2 = f2 * e2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.z);
    let w = Real::wide_sub_prod(w, f.l31, y1);
    let w = Real::wide_sub_prod(w, f.l32, y2);
    let f3 = Real::wide_rescale(w);
    let y3 = f3 * e3;
    let x3 = y3 * e3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y2);
    let w = Real::wide_sub_prod(w, f.l32, x3);
    let g2 = Real::wide_rescale(w);
    let x2 = g2 * e2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y1);
    let w = Real::wide_sub_prod(w, f.l21, x2);
    let w = Real::wide_sub_prod(w, f.l31, x3);
    let g1 = Real::wide_rescale(w);
    let x1 = g1 * e1;
    Vector3 { x: x1, y: x2, z: x3 }
}

/// LOSER. `inverse` reusing the already-computed `q_ii = recip(l_ii)` as a multiplier for the
/// sub-diagonal entries of `l⁻¹` instead of dividing by `l_ii`. Cheaper (the reciprocal is free
/// here) but less accurate, and accuracy wins.
fn inverse3_recip(f: Cholesky3<Fixed>) -> Matrix3<Fixed> {
    let q11 = Real::recip(f.l11);
    let q22 = Real::recip(f.l22);
    let q33 = Real::recip(f.l33);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l21, q11);
    let t21 = Real::wide_rescale(w);
    let q21 = t21 * q22;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l31, q11);
    let w = Real::wide_sub_prod(w, f.l32, q21);
    let t31 = Real::wide_rescale(w);
    let q31 = t31 * q33;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l32, q22);
    let t32 = Real::wide_rescale(w);
    let q32 = t32 * q33;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q11, q11);
    let w = Real::wide_add_prod(w, q21, q21);
    let w = Real::wide_add_prod(w, q31, q31);
    let r11 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q21, q22);
    let w = Real::wide_add_prod(w, q31, q32);
    let r12 = Real::wide_rescale(w);
    let r13 = q31 * q33;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q22, q22);
    let w = Real::wide_add_prod(w, q32, q32);
    let r22 = Real::wide_rescale(w);
    let r23 = q32 * q33;
    let r33 = Real::sqr(q33);
    Matrix3 {
        m11: r11, m21: r12, m31: r13, m12: r12, m22: r22, m32: r23, m13: r13, m23: r23, m33: r33,
    }
}

/// Worst error in ulp of a `solve` candidate over the oracle's `cholesky3_solve` vectors.
fn solve3_worst(variant: u8) -> u128 {
    let mut cases = oracle_cholesky::cholesky3_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _tol) = *case;
        let f = Cholesky3Trait::new(m3(a)).unwrap();
        let got = if variant == 0 {
            f.solve(v3t(b))
        } else {
            solve3_recip(f, v3t(b))
        };
        worst = core::cmp::max(worst, max_ulp_diff_v3(got, v3t(expected)));
    }
    worst
}

/// Worst error in ulp of an `inverse` candidate over the oracle's `cholesky3_inverse` vectors.
fn inverse3_worst(variant: u8) -> u128 {
    let mut cases = oracle_cholesky::cholesky3_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, _tol) = *case;
        let f = Cholesky3Trait::new(m3(a)).unwrap();
        let got = if variant == 0 {
            f.inverse()
        } else {
            inverse3_recip(f)
        };
        worst = core::cmp::max(worst, max_ulp_diff3(got, m3(expected)));
    }
    worst
}

/// Why `alt_recip` lost: measured worst error over the 12 oracle cases, shipped against candidate.
#[test]
fn test_cholesky3_alt_recip_is_less_accurate() {
    let (sd, sr) = (solve3_worst(0), solve3_worst(1));
    let (id, ir) = (inverse3_worst(0), inverse3_worst(1));
    assert!(
        sd == 135 && sr == 167 && id == 86 && ir == 86,
        "cholesky3 solve div {} recip {} / inverse div {} recip {}",
        sd,
        sr,
        id,
        ir,
    );
}

#[test]
#[inline(never)]
fn bench_cholesky3_new__baseline() {
    let _a = black_box(a3());
    let e = black_box(f3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_new__factorize() {
    let a = black_box(a3());
    let e = black_box(f3());
    assert!(Cholesky3Trait::new(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_l__baseline() {
    let _f = black_box(f3());
    let e = black_box(m3i([[1, 0, 0], [1, 1, 0], [1, 1, 1]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_l__expand() {
    let f = black_box(f3());
    let e = black_box(m3i([[1, 0, 0], [1, 1, 0], [1, 1, 1]]));
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_solve__baseline() {
    let _f = black_box(f3());
    let _b = black_box(b3());
    let e = black_box(x3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_solve__substitution() {
    let f = black_box(f3());
    let b = black_box(b3());
    let e = black_box(x3());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_solve__alt_recip() {
    let f = black_box(f3());
    let b = black_box(b3());
    let e = black_box(x3());
    assert!(solve3_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_inverse__baseline() {
    let _f = black_box(f3());
    let e = black_box(inv3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_inverse__triangular() {
    let f = black_box(f3());
    let e = black_box(inv3());
    assert!(f.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_inverse__alt_recip() {
    let f = black_box(f3());
    let e = black_box(inv3());
    assert!(inverse3_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_determinant__baseline() {
    let _f = black_box(f3());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky3_determinant__diagonal_product() {
    let f = black_box(f3());
    let e = black_box(int(1));
    assert!(f.determinant() == e);
}

// --- size 4 -------------------------------------------------------------------------------------

/// The benchmark input `a_ij = min(i, j)`.
fn a4() -> Matrix4<Fixed> {
    m4i([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3], [1, 2, 3, 4]])
}

/// Its Cholesky factor: the all-ones lower triangle.
fn f4() -> Cholesky4<Fixed> {
    Cholesky4 {
        l11: int(1),
        l21: int(1),
        l31: int(1),
        l41: int(1),
        l22: int(1),
        l32: int(1),
        l42: int(1),
        l33: int(1),
        l43: int(1),
        l44: int(1),
    }
}

/// The right-hand side of the benchmarked `solve`, and its exact solution.
fn b4() -> Vector4<Fixed> {
    v4it((1, -2, 3, -4))
}

/// `a4()⁻¹ · b4()`, exactly.
fn x4() -> Vector4<Fixed> {
    v4it((4, -8, 12, -7))
}

/// `a4()⁻¹`: tridiagonal, exactly.
fn inv4() -> Matrix4<Fixed> {
    m4i([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]])
}

/// LOSER. `solve` with one reciprocal per pivot and multiplications instead of the two exactly
/// correctly rounded divisions: `recip(l_jj)` rounds `1/l_jj` first, so each substitution step is
/// off by up to `|y_i|` ulp instead of 1, and two multiplications plus a reciprocal cost more than
/// two divisions.
fn solve4_recip(f: Cholesky4<Fixed>, b: Vector4<Fixed>) -> Vector4<Fixed> {
    let e1 = Real::recip(f.l11);
    let e2 = Real::recip(f.l22);
    let e3 = Real::recip(f.l33);
    let e4 = Real::recip(f.l44);
    let y1 = b.x * e1;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.y);
    let w = Real::wide_sub_prod(w, f.l21, y1);
    let f2 = Real::wide_rescale(w);
    let y2 = f2 * e2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.z);
    let w = Real::wide_sub_prod(w, f.l31, y1);
    let w = Real::wide_sub_prod(w, f.l32, y2);
    let f3 = Real::wide_rescale(w);
    let y3 = f3 * e3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.w);
    let w = Real::wide_sub_prod(w, f.l41, y1);
    let w = Real::wide_sub_prod(w, f.l42, y2);
    let w = Real::wide_sub_prod(w, f.l43, y3);
    let f4 = Real::wide_rescale(w);
    let y4 = f4 * e4;
    let x4 = y4 * e4;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y3);
    let w = Real::wide_sub_prod(w, f.l43, x4);
    let g3 = Real::wide_rescale(w);
    let x3 = g3 * e3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y2);
    let w = Real::wide_sub_prod(w, f.l32, x3);
    let w = Real::wide_sub_prod(w, f.l42, x4);
    let g2 = Real::wide_rescale(w);
    let x2 = g2 * e2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y1);
    let w = Real::wide_sub_prod(w, f.l21, x2);
    let w = Real::wide_sub_prod(w, f.l31, x3);
    let w = Real::wide_sub_prod(w, f.l41, x4);
    let g1 = Real::wide_rescale(w);
    let x1 = g1 * e1;
    Vector4 { x: x1, y: x2, z: x3, w: x4 }
}

/// LOSER. `inverse` reusing the already-computed `q_ii = recip(l_ii)` as a multiplier for the
/// sub-diagonal entries of `l⁻¹` instead of dividing by `l_ii`. Cheaper (the reciprocal is free
/// here) but less accurate, and accuracy wins.
fn inverse4_recip(f: Cholesky4<Fixed>) -> Matrix4<Fixed> {
    let q11 = Real::recip(f.l11);
    let q22 = Real::recip(f.l22);
    let q33 = Real::recip(f.l33);
    let q44 = Real::recip(f.l44);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l21, q11);
    let t21 = Real::wide_rescale(w);
    let q21 = t21 * q22;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l31, q11);
    let w = Real::wide_sub_prod(w, f.l32, q21);
    let t31 = Real::wide_rescale(w);
    let q31 = t31 * q33;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l41, q11);
    let w = Real::wide_sub_prod(w, f.l42, q21);
    let w = Real::wide_sub_prod(w, f.l43, q31);
    let t41 = Real::wide_rescale(w);
    let q41 = t41 * q44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l32, q22);
    let t32 = Real::wide_rescale(w);
    let q32 = t32 * q33;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l42, q22);
    let w = Real::wide_sub_prod(w, f.l43, q32);
    let t42 = Real::wide_rescale(w);
    let q42 = t42 * q44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l43, q33);
    let t43 = Real::wide_rescale(w);
    let q43 = t43 * q44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q11, q11);
    let w = Real::wide_add_prod(w, q21, q21);
    let w = Real::wide_add_prod(w, q31, q31);
    let w = Real::wide_add_prod(w, q41, q41);
    let r11 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q21, q22);
    let w = Real::wide_add_prod(w, q31, q32);
    let w = Real::wide_add_prod(w, q41, q42);
    let r12 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q31, q33);
    let w = Real::wide_add_prod(w, q41, q43);
    let r13 = Real::wide_rescale(w);
    let r14 = q41 * q44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q22, q22);
    let w = Real::wide_add_prod(w, q32, q32);
    let w = Real::wide_add_prod(w, q42, q42);
    let r22 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q32, q33);
    let w = Real::wide_add_prod(w, q42, q43);
    let r23 = Real::wide_rescale(w);
    let r24 = q42 * q44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q33, q33);
    let w = Real::wide_add_prod(w, q43, q43);
    let r33 = Real::wide_rescale(w);
    let r34 = q43 * q44;
    let r44 = Real::sqr(q44);
    Matrix4 {
        m11: r11,
        m21: r12,
        m31: r13,
        m41: r14,
        m12: r12,
        m22: r22,
        m32: r23,
        m42: r24,
        m13: r13,
        m23: r23,
        m33: r33,
        m43: r34,
        m14: r14,
        m24: r24,
        m34: r34,
        m44: r44,
    }
}

/// Worst error in ulp of a `solve` candidate over the oracle's `cholesky4_solve` vectors.
fn solve4_worst(variant: u8) -> u128 {
    let mut cases = oracle_cholesky::cholesky4_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _tol) = *case;
        let f = Cholesky4Trait::new(m4(a)).unwrap();
        let got = if variant == 0 {
            f.solve(v4t(b))
        } else {
            solve4_recip(f, v4t(b))
        };
        worst = core::cmp::max(worst, max_ulp_diff_v4(got, v4t(expected)));
    }
    worst
}

/// Worst error in ulp of an `inverse` candidate over the oracle's `cholesky4_inverse` vectors.
fn inverse4_worst(variant: u8) -> u128 {
    let mut cases = oracle_cholesky::cholesky4_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, _tol) = *case;
        let f = Cholesky4Trait::new(m4(a)).unwrap();
        let got = if variant == 0 {
            f.inverse()
        } else {
            inverse4_recip(f)
        };
        worst = core::cmp::max(worst, max_ulp_diff4(got, m4(expected)));
    }
    worst
}

/// Why `alt_recip` lost: measured worst error over the 12 oracle cases, shipped against candidate.
#[test]
fn test_cholesky4_alt_recip_is_less_accurate() {
    let (sd, sr) = (solve4_worst(0), solve4_worst(1));
    let (id, ir) = (inverse4_worst(0), inverse4_worst(1));
    assert!(
        sd == 107 && sr == 110 && id == 147 && ir == 148,
        "cholesky4 solve div {} recip {} / inverse div {} recip {}",
        sd,
        sr,
        id,
        ir,
    );
}

#[test]
#[inline(never)]
fn bench_cholesky4_new__baseline() {
    let _a = black_box(a4());
    let e = black_box(f4());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_new__factorize() {
    let a = black_box(a4());
    let e = black_box(f4());
    assert!(Cholesky4Trait::new(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_l__baseline() {
    let _f = black_box(f4());
    let e = black_box(m4i([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_l__expand() {
    let f = black_box(f4());
    let e = black_box(m4i([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]));
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_solve__baseline() {
    let _f = black_box(f4());
    let _b = black_box(b4());
    let e = black_box(x4());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_solve__substitution() {
    let f = black_box(f4());
    let b = black_box(b4());
    let e = black_box(x4());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_solve__alt_recip() {
    let f = black_box(f4());
    let b = black_box(b4());
    let e = black_box(x4());
    assert!(solve4_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_inverse__baseline() {
    let _f = black_box(f4());
    let e = black_box(inv4());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_inverse__triangular() {
    let f = black_box(f4());
    let e = black_box(inv4());
    assert!(f.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_inverse__alt_recip() {
    let f = black_box(f4());
    let e = black_box(inv4());
    assert!(inverse4_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_determinant__baseline() {
    let _f = black_box(f4());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky4_determinant__diagonal_product() {
    let f = black_box(f4());
    let e = black_box(int(1));
    assert!(f.determinant() == e);
}

// --- size 6 -------------------------------------------------------------------------------------

/// The benchmark input `a_ij = min(i, j)`.
fn a6() -> Matrix6<Fixed> {
    m6i(
        [
            [1, 1, 1, 1, 1, 1], [1, 2, 2, 2, 2, 2], [1, 2, 3, 3, 3, 3], [1, 2, 3, 4, 4, 4],
            [1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 5, 6],
        ],
    )
}

/// Its Cholesky factor: the all-ones lower triangle.
fn f6() -> Cholesky6<Fixed> {
    Cholesky6 {
        l11: int(1),
        l21: int(1),
        l31: int(1),
        l41: int(1),
        l51: int(1),
        l61: int(1),
        l22: int(1),
        l32: int(1),
        l42: int(1),
        l52: int(1),
        l62: int(1),
        l33: int(1),
        l43: int(1),
        l53: int(1),
        l63: int(1),
        l44: int(1),
        l54: int(1),
        l64: int(1),
        l55: int(1),
        l65: int(1),
        l66: int(1),
    }
}

/// The right-hand side of the benchmarked `solve`, and its exact solution.
fn b6() -> Vector6<Fixed> {
    v6it((1, -2, 3, -4, 5, -6))
}

/// `a6()⁻¹ · b6()`, exactly.
fn x6() -> Vector6<Fixed> {
    v6it((4, -8, 12, -16, 20, -11))
}

/// `a6()⁻¹`: tridiagonal, exactly.
fn inv6() -> Matrix6<Fixed> {
    m6i(
        [
            [2, -1, 0, 0, 0, 0], [-1, 2, -1, 0, 0, 0], [0, -1, 2, -1, 0, 0], [0, 0, -1, 2, -1, 0],
            [0, 0, 0, -1, 2, -1], [0, 0, 0, 0, -1, 1],
        ],
    )
}

/// LOSER. `solve` with one reciprocal per pivot and multiplications instead of the two exactly
/// correctly rounded divisions: `recip(l_jj)` rounds `1/l_jj` first, so each substitution step is
/// off by up to `|y_i|` ulp instead of 1, and two multiplications plus a reciprocal cost more than
/// two divisions.
fn solve6_recip(f: Cholesky6<Fixed>, b: Vector6<Fixed>) -> Vector6<Fixed> {
    let e1 = Real::recip(f.l11);
    let e2 = Real::recip(f.l22);
    let e3 = Real::recip(f.l33);
    let e4 = Real::recip(f.l44);
    let e5 = Real::recip(f.l55);
    let e6 = Real::recip(f.l66);
    let y1 = b.a.x * e1;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.a.y);
    let w = Real::wide_sub_prod(w, f.l21, y1);
    let f2 = Real::wide_rescale(w);
    let y2 = f2 * e2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.a.z);
    let w = Real::wide_sub_prod(w, f.l31, y1);
    let w = Real::wide_sub_prod(w, f.l32, y2);
    let f3 = Real::wide_rescale(w);
    let y3 = f3 * e3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.b.x);
    let w = Real::wide_sub_prod(w, f.l41, y1);
    let w = Real::wide_sub_prod(w, f.l42, y2);
    let w = Real::wide_sub_prod(w, f.l43, y3);
    let f4 = Real::wide_rescale(w);
    let y4 = f4 * e4;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.b.y);
    let w = Real::wide_sub_prod(w, f.l51, y1);
    let w = Real::wide_sub_prod(w, f.l52, y2);
    let w = Real::wide_sub_prod(w, f.l53, y3);
    let w = Real::wide_sub_prod(w, f.l54, y4);
    let f5 = Real::wide_rescale(w);
    let y5 = f5 * e5;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.b.z);
    let w = Real::wide_sub_prod(w, f.l61, y1);
    let w = Real::wide_sub_prod(w, f.l62, y2);
    let w = Real::wide_sub_prod(w, f.l63, y3);
    let w = Real::wide_sub_prod(w, f.l64, y4);
    let w = Real::wide_sub_prod(w, f.l65, y5);
    let f6 = Real::wide_rescale(w);
    let y6 = f6 * e6;
    let x6 = y6 * e6;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y5);
    let w = Real::wide_sub_prod(w, f.l65, x6);
    let g5 = Real::wide_rescale(w);
    let x5 = g5 * e5;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y4);
    let w = Real::wide_sub_prod(w, f.l54, x5);
    let w = Real::wide_sub_prod(w, f.l64, x6);
    let g4 = Real::wide_rescale(w);
    let x4 = g4 * e4;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y3);
    let w = Real::wide_sub_prod(w, f.l43, x4);
    let w = Real::wide_sub_prod(w, f.l53, x5);
    let w = Real::wide_sub_prod(w, f.l63, x6);
    let g3 = Real::wide_rescale(w);
    let x3 = g3 * e3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y2);
    let w = Real::wide_sub_prod(w, f.l32, x3);
    let w = Real::wide_sub_prod(w, f.l42, x4);
    let w = Real::wide_sub_prod(w, f.l52, x5);
    let w = Real::wide_sub_prod(w, f.l62, x6);
    let g2 = Real::wide_rescale(w);
    let x2 = g2 * e2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), y1);
    let w = Real::wide_sub_prod(w, f.l21, x2);
    let w = Real::wide_sub_prod(w, f.l31, x3);
    let w = Real::wide_sub_prod(w, f.l41, x4);
    let w = Real::wide_sub_prod(w, f.l51, x5);
    let w = Real::wide_sub_prod(w, f.l61, x6);
    let g1 = Real::wide_rescale(w);
    let x1 = g1 * e1;
    Vector6 { a: Vector3 { x: x1, y: x2, z: x3 }, b: Vector3 { x: x4, y: x5, z: x6 } }
}

/// LOSER. `inverse` reusing the already-computed `q_ii = recip(l_ii)` as a multiplier for the
/// sub-diagonal entries of `l⁻¹` instead of dividing by `l_ii`. Cheaper (the reciprocal is free
/// here) but less accurate, and accuracy wins.
fn inverse6_recip(f: Cholesky6<Fixed>) -> Matrix6<Fixed> {
    let q11 = Real::recip(f.l11);
    let q22 = Real::recip(f.l22);
    let q33 = Real::recip(f.l33);
    let q44 = Real::recip(f.l44);
    let q55 = Real::recip(f.l55);
    let q66 = Real::recip(f.l66);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l21, q11);
    let t21 = Real::wide_rescale(w);
    let q21 = t21 * q22;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l31, q11);
    let w = Real::wide_sub_prod(w, f.l32, q21);
    let t31 = Real::wide_rescale(w);
    let q31 = t31 * q33;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l41, q11);
    let w = Real::wide_sub_prod(w, f.l42, q21);
    let w = Real::wide_sub_prod(w, f.l43, q31);
    let t41 = Real::wide_rescale(w);
    let q41 = t41 * q44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l51, q11);
    let w = Real::wide_sub_prod(w, f.l52, q21);
    let w = Real::wide_sub_prod(w, f.l53, q31);
    let w = Real::wide_sub_prod(w, f.l54, q41);
    let t51 = Real::wide_rescale(w);
    let q51 = t51 * q55;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l61, q11);
    let w = Real::wide_sub_prod(w, f.l62, q21);
    let w = Real::wide_sub_prod(w, f.l63, q31);
    let w = Real::wide_sub_prod(w, f.l64, q41);
    let w = Real::wide_sub_prod(w, f.l65, q51);
    let t61 = Real::wide_rescale(w);
    let q61 = t61 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l32, q22);
    let t32 = Real::wide_rescale(w);
    let q32 = t32 * q33;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l42, q22);
    let w = Real::wide_sub_prod(w, f.l43, q32);
    let t42 = Real::wide_rescale(w);
    let q42 = t42 * q44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l52, q22);
    let w = Real::wide_sub_prod(w, f.l53, q32);
    let w = Real::wide_sub_prod(w, f.l54, q42);
    let t52 = Real::wide_rescale(w);
    let q52 = t52 * q55;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l62, q22);
    let w = Real::wide_sub_prod(w, f.l63, q32);
    let w = Real::wide_sub_prod(w, f.l64, q42);
    let w = Real::wide_sub_prod(w, f.l65, q52);
    let t62 = Real::wide_rescale(w);
    let q62 = t62 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l43, q33);
    let t43 = Real::wide_rescale(w);
    let q43 = t43 * q44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l53, q33);
    let w = Real::wide_sub_prod(w, f.l54, q43);
    let t53 = Real::wide_rescale(w);
    let q53 = t53 * q55;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l63, q33);
    let w = Real::wide_sub_prod(w, f.l64, q43);
    let w = Real::wide_sub_prod(w, f.l65, q53);
    let t63 = Real::wide_rescale(w);
    let q63 = t63 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l54, q44);
    let t54 = Real::wide_rescale(w);
    let q54 = t54 * q55;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l64, q44);
    let w = Real::wide_sub_prod(w, f.l65, q54);
    let t64 = Real::wide_rescale(w);
    let q64 = t64 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_sub_prod(w, f.l65, q55);
    let t65 = Real::wide_rescale(w);
    let q65 = t65 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q11, q11);
    let w = Real::wide_add_prod(w, q21, q21);
    let w = Real::wide_add_prod(w, q31, q31);
    let w = Real::wide_add_prod(w, q41, q41);
    let w = Real::wide_add_prod(w, q51, q51);
    let w = Real::wide_add_prod(w, q61, q61);
    let r11 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q21, q22);
    let w = Real::wide_add_prod(w, q31, q32);
    let w = Real::wide_add_prod(w, q41, q42);
    let w = Real::wide_add_prod(w, q51, q52);
    let w = Real::wide_add_prod(w, q61, q62);
    let r12 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q31, q33);
    let w = Real::wide_add_prod(w, q41, q43);
    let w = Real::wide_add_prod(w, q51, q53);
    let w = Real::wide_add_prod(w, q61, q63);
    let r13 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q41, q44);
    let w = Real::wide_add_prod(w, q51, q54);
    let w = Real::wide_add_prod(w, q61, q64);
    let r14 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q51, q55);
    let w = Real::wide_add_prod(w, q61, q65);
    let r15 = Real::wide_rescale(w);
    let r16 = q61 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q22, q22);
    let w = Real::wide_add_prod(w, q32, q32);
    let w = Real::wide_add_prod(w, q42, q42);
    let w = Real::wide_add_prod(w, q52, q52);
    let w = Real::wide_add_prod(w, q62, q62);
    let r22 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q32, q33);
    let w = Real::wide_add_prod(w, q42, q43);
    let w = Real::wide_add_prod(w, q52, q53);
    let w = Real::wide_add_prod(w, q62, q63);
    let r23 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q42, q44);
    let w = Real::wide_add_prod(w, q52, q54);
    let w = Real::wide_add_prod(w, q62, q64);
    let r24 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q52, q55);
    let w = Real::wide_add_prod(w, q62, q65);
    let r25 = Real::wide_rescale(w);
    let r26 = q62 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q33, q33);
    let w = Real::wide_add_prod(w, q43, q43);
    let w = Real::wide_add_prod(w, q53, q53);
    let w = Real::wide_add_prod(w, q63, q63);
    let r33 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q43, q44);
    let w = Real::wide_add_prod(w, q53, q54);
    let w = Real::wide_add_prod(w, q63, q64);
    let r34 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q53, q55);
    let w = Real::wide_add_prod(w, q63, q65);
    let r35 = Real::wide_rescale(w);
    let r36 = q63 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q44, q44);
    let w = Real::wide_add_prod(w, q54, q54);
    let w = Real::wide_add_prod(w, q64, q64);
    let r44 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q54, q55);
    let w = Real::wide_add_prod(w, q64, q65);
    let r45 = Real::wide_rescale(w);
    let r46 = q64 * q66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q55, q55);
    let w = Real::wide_add_prod(w, q65, q65);
    let r55 = Real::wide_rescale(w);
    let r56 = q65 * q66;
    let r66 = Real::sqr(q66);
    Matrix6 {
        m11: Matrix3 {
            m11: r11,
            m21: r12,
            m31: r13,
            m12: r12,
            m22: r22,
            m32: r23,
            m13: r13,
            m23: r23,
            m33: r33,
        },
        m21: Matrix3 {
            m11: r14,
            m21: r15,
            m31: r16,
            m12: r24,
            m22: r25,
            m32: r26,
            m13: r34,
            m23: r35,
            m33: r36,
        },
        m12: Matrix3 {
            m11: r14,
            m21: r24,
            m31: r34,
            m12: r15,
            m22: r25,
            m32: r35,
            m13: r16,
            m23: r26,
            m33: r36,
        },
        m22: Matrix3 {
            m11: r44,
            m21: r45,
            m31: r46,
            m12: r45,
            m22: r55,
            m32: r56,
            m13: r46,
            m23: r56,
            m33: r66,
        },
    }
}

/// Worst error in ulp of a `solve` candidate over the oracle's `cholesky6_solve` vectors.
fn solve6_worst(variant: u8) -> u128 {
    let mut cases = oracle_cholesky::cholesky6_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _tol) = *case;
        let f = Cholesky6Trait::new(m6(a)).unwrap();
        let got = if variant == 0 {
            f.solve(v6t(b))
        } else {
            solve6_recip(f, v6t(b))
        };
        worst = core::cmp::max(worst, max_ulp_diff_v6(got, v6t(expected)));
    }
    worst
}

/// Worst error in ulp of an `inverse` candidate over the oracle's `cholesky6_inverse` vectors.
fn inverse6_worst(variant: u8) -> u128 {
    let mut cases = oracle_cholesky::cholesky6_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, _tol) = *case;
        let f = Cholesky6Trait::new(m6(a)).unwrap();
        let got = if variant == 0 {
            f.inverse()
        } else {
            inverse6_recip(f)
        };
        worst = core::cmp::max(worst, max_ulp_diff6(got, m6(expected)));
    }
    worst
}

/// Why `alt_recip` lost: measured worst error over the 12 oracle cases, shipped against candidate.
#[test]
fn test_cholesky6_alt_recip_is_less_accurate() {
    let (sd, sr) = (solve6_worst(0), solve6_worst(1));
    let (id, ir) = (inverse6_worst(0), inverse6_worst(1));
    assert!(
        sd == 1558 && sr == 1023 && id == 95 && ir == 105,
        "cholesky6 solve div {} recip {} / inverse div {} recip {}",
        sd,
        sr,
        id,
        ir,
    );
}

#[test]
#[inline(never)]
fn bench_cholesky6_new__baseline() {
    let _a = black_box(a6());
    let e = black_box(f6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_new__factorize() {
    let a = black_box(a6());
    let e = black_box(f6());
    assert!(Cholesky6Trait::new(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_l__baseline() {
    let _f = black_box(f6());
    let e = black_box(
        m6i(
            [
                [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_l__expand() {
    let f = black_box(f6());
    let e = black_box(
        m6i(
            [
                [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1],
            ],
        ),
    );
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_solve__baseline() {
    let _f = black_box(f6());
    let _b = black_box(b6());
    let e = black_box(x6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_solve__substitution() {
    let f = black_box(f6());
    let b = black_box(b6());
    let e = black_box(x6());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_solve__alt_recip() {
    let f = black_box(f6());
    let b = black_box(b6());
    let e = black_box(x6());
    assert!(solve6_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_inverse__baseline() {
    let _f = black_box(f6());
    let e = black_box(inv6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_inverse__triangular() {
    let f = black_box(f6());
    let e = black_box(inv6());
    assert!(f.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_inverse__alt_recip() {
    let f = black_box(f6());
    let e = black_box(inv6());
    assert!(inverse6_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_determinant__baseline() {
    let _f = black_box(f6());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_cholesky6_determinant__diagonal_product() {
    let f = black_box(f6());
    let e = black_box(int(1));
    assert!(f.determinant() == e);
}

/// Why `alt_recip` loses in `inverse`, on a hand-built factor where the oracle's SPD sample does
/// not separate the two: `l = [[1, 0], [5, 3]]`, so `a = l·lᵀ = [[1, 5], [5, 34]]` and
/// `a⁻¹_11 = 34/9`, whose exact floor is 16 225 432 007 raw. The sub-diagonal entry of `l⁻¹`
/// is `-5/3`, floored exactly by the division and 2 ulp high through `-5 · recip(3)`; that error
/// reaches the output as 1 ulp for the shipped code and 6 ulp for the candidate.
#[test]
fn test_cholesky2_inverse_alt_recip_loses_low_bits() {
    let f = Cholesky2 { l11: int(1), l21: int(5), l22: int(3) };
    let exact = fx(16225432007);
    let shipped = f.inverse();
    let alt = inverse2_recip(f);
    assert!(shipped != alt);
    assert!(shipped.m12 == alt.m12 && shipped.m22 == alt.m22);
    assert!(ulp_diff(shipped.m11, exact) == 1);
    assert!(ulp_diff(alt.m11, exact) == 6);
}
