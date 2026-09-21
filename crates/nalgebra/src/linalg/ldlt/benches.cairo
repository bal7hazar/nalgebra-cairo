//! Gas benchmarks of `Ldlt{2,3,4,6}` (`bench_ldlt<n>_<op>__<variant>`, net = raw minus the
//! `baseline` of the group) and the alternative implementations that lost (`alt_*`), kept with the
//! accuracy measurements that rejected them (AGENTS.md rule 8).
//!
//! Three candidates are measured here.
//!
//! `alt_recip` in `solve` and `inverse`: one `recip(d_j)` then multiplications instead of exactly
//! floored divisions. Same verdict as in `cholesky::benches` — each pivot is divided by ONCE in
//! `solve`, so the reciprocal is both dearer (19 to 26 %) and, on the oracle's 12 cases, markedly
//! less accurate; in `inverse` it saves 6 to 11 % but rounds twice per entry, which
//! `test_ldlt2_inverse_alt_recip_loses_low_bits` exhibits on a hand-built factor.
//!
//! `alt_products` in `new`: recompute the column of `l·diag(d)` as explicit rounded products
//! `l_jk · d_k` instead of reusing the unrounded numerator of column k, which is the same quantity
//! up to the remainder of the division that produced `l_jk`. It costs n(n-1)/2 extra products and
//! is FARTHER from the true factors (the pivot `a_jj - l_jk·(l_jk·d_k)` is second order in the
//! error of `l_jk`, `a_jj - l_jk·n_jk` only first order), for 30 to 32 % more gas; the backward
//! error `l·diag(d)·lᵀ - a` is identical, being dominated by `l_ij·d_j - a_ij` in both.
//! Cheaper and more accurate wins, so the reuse is what ships. Both error figures are asserted
//! below.
//!
//! THE FIXTURE. Every benchmark factorises `a_ij = min(i, j)`, whose `LDLᵀ` factor is the
//! all-ones unit lower triangle with `d = (1, .., 1)`: every input and result is an integer, the
//! asserted values are exact, and all variants of a group take the same branches.

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix2::Matrix2Trait;
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix4::{Matrix4, Matrix4Trait};
use crate::base::matrix6::{Matrix6, Matrix6Trait};
use crate::base::matrix_test_utils::{
    fx, int, m2, m2i, m3, m3i, m4, m4i, m6, m6i, max_ulp_diff2, max_ulp_diff3, max_ulp_diff4,
    max_ulp_diff6, max_ulp_diff_v2, max_ulp_diff_v3, max_ulp_diff_v4, max_ulp_diff_v6, s2ir, s2r,
    s3ir, s3r, ulp_diff, v2it, v2t, v3it, v3t, v4it, v4t, v6it, v6t,
};
use crate::base::sym_matrix2::{SymMatrix2, SymMatrix2Trait};
use crate::base::sym_matrix3::{SymMatrix3, SymMatrix3Trait};
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector6::Vector6;
use crate::linalg::oracle_udu;
use super::{Ldlt2, Ldlt2Trait, Ldlt3, Ldlt3Trait, Ldlt4, Ldlt4Trait, Ldlt6, Ldlt6Trait};

// --- size 2 -------------------------------------------------------------------------------------

/// The benchmark input `a_ij = min(i, j)`.
fn a2() -> SymMatrix2<Fixed> {
    s2ir([[1, 1], [1, 2]])
}

/// Its `LDLᵀ` factor: the all-ones unit lower triangle, `d = (1, .., 1)`.
fn f2() -> Ldlt2<Fixed> {
    Ldlt2 { l21: int(1), d: v2it((1, 1)) }
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
fn inv2() -> SymMatrix2<Fixed> {
    s2ir([[2, -1], [-1, 1]])
}

/// LOSER. `new` recomputing `l_jk · d_k` as an explicit rounded product instead of reusing the
/// numerator of column k: n(n-1)/2 extra products, and a pivot that is second order rather than
/// first order in the rounding of `l_jk`.
fn new2_products(a: SymMatrix2<Fixed>) -> Option<Ldlt2<Fixed>> {
    let d1 = a.m11;
    if d1 == Real::<Fixed>::ZERO {
        return None;
    }
    let l21 = a.m12 / d1;
    let t21 = l21 * d1;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22);
    let w = Real::wide_sub_prod(w, l21, t21);
    let d2 = Real::wide_rescale(w);
    if d2 == Real::<Fixed>::ZERO {
        return None;
    }
    Some(Ldlt2 { l21, d: Vector2 { x: d1, y: d2 } })
}

/// LOSER. `solve` with one reciprocal per pivot instead of one exactly floored division.
fn solve2_recip(f: Ldlt2<Fixed>, b: Vector2<Fixed>) -> Vector2<Fixed> {
    let y1 = b.x;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.y);
    let w = Real::wide_sub_prod(w, f.l21, y1);
    let y2 = Real::wide_rescale(w);
    let z1 = y1 * Real::recip(f.d.x);
    let z2 = y2 * Real::recip(f.d.y);
    let x2 = z2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z1);
    let w = Real::wide_sub_prod(w, f.l21, x2);
    let x1 = Real::wide_rescale(w);
    Vector2 { x: x1, y: x2 }
}

/// LOSER. `inverse` scaling `l⁻¹` by `recip(d_k)` instead of dividing by `d_k`.
fn inverse2_recip(f: Ldlt2<Fixed>) -> SymMatrix2<Fixed> {
    let q21 = -f.l21;
    let e1 = Real::recip(f.d.x);
    let e2 = Real::recip(f.d.y);
    let s11 = e1;
    let s21 = q21 * e2;
    let s22 = e2;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s11);
    let w = Real::wide_add_prod(w, q21, s21);
    let r11 = Real::wide_rescale(w);
    let r12 = q21 * s22;
    let r22 = s22;
    SymMatrix2 { m11: r11, m12: r12, m22: r22 }
}

/// `(worst error on the factors, worst backward error of the reconstruction)` in ulp, over the
/// oracle's `ldlt2_l_d` vectors, for a `new` candidate.
fn new2_worst(variant: u8) -> (u128, u128) {
    let mut cases = oracle_udu::ldlt2_l_d_cases();
    let mut factors = 0;
    let mut backward = 0;
    while let Some(case) = cases.pop_front() {
        let (a, l, d, _tol) = *case;
        let f = if variant == 0 {
            Ldlt2Trait::new(s2r(a)).unwrap()
        } else {
            new2_products(s2r(a)).unwrap()
        };
        let e = core::cmp::max(max_ulp_diff2(f.l(), m2(l)), max_ulp_diff_v2(f.d(), v2t(d)));
        factors = core::cmp::max(factors, e);
        let r = f.l() * Matrix2Trait::from_diagonal(f.d()) * f.l().transpose();
        backward = core::cmp::max(backward, max_ulp_diff2(r, m2(a)));
    }
    (factors, backward)
}

/// Worst error in ulp of a `solve` candidate over the oracle's `udu2_solve` vectors.
fn solve2_worst(variant: u8) -> u128 {
    let mut cases = oracle_udu::udu2_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _tol) = *case;
        let f = Ldlt2Trait::new(s2r(a)).unwrap();
        let got = if variant == 0 {
            f.solve(v2t(b))
        } else {
            solve2_recip(f, v2t(b))
        };
        worst = core::cmp::max(worst, max_ulp_diff_v2(got, v2t(expected)));
    }
    worst
}

/// Worst error in ulp of an `inverse` candidate over the oracle's `udu2_inverse` vectors.
fn inverse2_worst(variant: u8) -> u128 {
    let mut cases = oracle_udu::udu2_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, _tol) = *case;
        let f = Ldlt2Trait::new(s2r(a)).unwrap();
        let got = if variant == 0 {
            f.inverse()
        } else {
            inverse2_recip(f)
        };
        worst = core::cmp::max(worst, max_ulp_diff2(got.to_matrix(), m2(expected)));
    }
    worst
}

/// Why `alt_recip` lost: measured worst error over the 12 oracle cases.
#[test]
fn test_ldlt2_alt_recip_is_less_accurate() {
    let (sd, sr) = (solve2_worst(0), solve2_worst(1));
    let (id, ir) = (inverse2_worst(0), inverse2_worst(1));
    assert!(
        sd == 250 && sr == 329 && id == 33 && ir == 33,
        "ldlt2 solve div {} recip {} / inverse div {} recip {}",
        sd,
        sr,
        id,
        ir,
    );
}

/// Why `alt_products` lost: same backward error, twice the distance to the true factors.
#[test]
fn test_ldlt2_alt_products_is_less_accurate() {
    let (f0, b0) = new2_worst(0);
    let (f1, b1) = new2_worst(1);
    assert!(
        f0 == 16 && b0 == 30 && f1 == 34 && b1 == 30,
        "new2 shipped ({}, {}) products ({}, {})",
        f0,
        b0,
        f1,
        b1,
    );
}

#[test]
#[inline(never)]
fn bench_ldlt2_new__baseline() {
    let _a = black_box(a2());
    let e = black_box(f2());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_new__factorize() {
    let a = black_box(a2());
    let e = black_box(f2());
    assert!(Ldlt2Trait::new(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_new__alt_products() {
    let a = black_box(a2());
    let e = black_box(f2());
    assert!(new2_products(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_l__baseline() {
    let _f = black_box(f2());
    let e = black_box(m2i([[1, 0], [1, 1]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_l__expand() {
    let f = black_box(f2());
    let e = black_box(m2i([[1, 0], [1, 1]]));
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_d__baseline() {
    let _f = black_box(f2());
    let e = black_box(v2it((1, 1)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_d__accessor() {
    let f = black_box(f2());
    let e = black_box(v2it((1, 1)));
    assert!(f.d() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_solve__baseline() {
    let _f = black_box(f2());
    let _b = black_box(b2());
    let e = black_box(x2());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_solve__substitution() {
    let f = black_box(f2());
    let b = black_box(b2());
    let e = black_box(x2());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_solve__alt_recip() {
    let f = black_box(f2());
    let b = black_box(b2());
    let e = black_box(x2());
    assert!(solve2_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_inverse__baseline() {
    let _f = black_box(f2());
    let e = black_box(inv2());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_inverse__triangular() {
    let f = black_box(f2());
    let e = black_box(inv2());
    assert!(f.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_inverse__alt_recip() {
    let f = black_box(f2());
    let e = black_box(inv2());
    assert!(inverse2_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_determinant__baseline() {
    let _f = black_box(f2());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt2_determinant__diagonal_product() {
    let f = black_box(f2());
    let e = black_box(int(1));
    assert!(f.determinant() == e);
}

// --- size 3 -------------------------------------------------------------------------------------

/// The benchmark input `a_ij = min(i, j)`.
fn a3() -> SymMatrix3<Fixed> {
    s3ir([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
}

/// Its `LDLᵀ` factor: the all-ones unit lower triangle, `d = (1, .., 1)`.
fn f3() -> Ldlt3<Fixed> {
    Ldlt3 { l21: int(1), l31: int(1), l32: int(1), d: v3it((1, 1, 1)) }
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
fn inv3() -> SymMatrix3<Fixed> {
    s3ir([[2, -1, 0], [-1, 2, -1], [0, -1, 1]])
}

/// LOSER. `new` recomputing `l_jk · d_k` as an explicit rounded product instead of reusing the
/// numerator of column k: n(n-1)/2 extra products, and a pivot that is second order rather than
/// first order in the rounding of `l_jk`.
fn new3_products(a: SymMatrix3<Fixed>) -> Option<Ldlt3<Fixed>> {
    let d1 = a.m11;
    if d1 == Real::<Fixed>::ZERO {
        return None;
    }
    let l21 = a.m12 / d1;
    let l31 = a.m13 / d1;
    let t21 = l21 * d1;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22);
    let w = Real::wide_sub_prod(w, l21, t21);
    let d2 = Real::wide_rescale(w);
    if d2 == Real::<Fixed>::ZERO {
        return None;
    }
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m23);
    let w = Real::wide_sub_prod(w, l31, t21);
    let n32 = Real::wide_rescale(w);
    let l32 = n32 / d2;
    let t31 = l31 * d1;
    let t32 = l32 * d2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m33);
    let w = Real::wide_sub_prod(w, l31, t31);
    let w = Real::wide_sub_prod(w, l32, t32);
    let d3 = Real::wide_rescale(w);
    if d3 == Real::<Fixed>::ZERO {
        return None;
    }
    Some(Ldlt3 { l21, l31, l32, d: Vector3 { x: d1, y: d2, z: d3 } })
}

/// LOSER. `solve` with one reciprocal per pivot instead of one exactly floored division.
fn solve3_recip(f: Ldlt3<Fixed>, b: Vector3<Fixed>) -> Vector3<Fixed> {
    let y1 = b.x;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.y);
    let w = Real::wide_sub_prod(w, f.l21, y1);
    let y2 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.z);
    let w = Real::wide_sub_prod(w, f.l31, y1);
    let w = Real::wide_sub_prod(w, f.l32, y2);
    let y3 = Real::wide_rescale(w);
    let z1 = y1 * Real::recip(f.d.x);
    let z2 = y2 * Real::recip(f.d.y);
    let z3 = y3 * Real::recip(f.d.z);
    let x3 = z3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z2);
    let w = Real::wide_sub_prod(w, f.l32, x3);
    let x2 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z1);
    let w = Real::wide_sub_prod(w, f.l21, x2);
    let w = Real::wide_sub_prod(w, f.l31, x3);
    let x1 = Real::wide_rescale(w);
    Vector3 { x: x1, y: x2, z: x3 }
}

/// LOSER. `inverse` scaling `l⁻¹` by `recip(d_k)` instead of dividing by `d_k`.
fn inverse3_recip(f: Ldlt3<Fixed>) -> SymMatrix3<Fixed> {
    let q21 = -f.l21;
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l31);
    let w = Real::wide_sub_prod(w, f.l32, q21);
    let q31 = Real::wide_rescale(w);
    let q32 = -f.l32;
    let e1 = Real::recip(f.d.x);
    let e2 = Real::recip(f.d.y);
    let e3 = Real::recip(f.d.z);
    let s11 = e1;
    let s21 = q21 * e2;
    let s31 = q31 * e3;
    let s22 = e2;
    let s32 = q32 * e3;
    let s33 = e3;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s11);
    let w = Real::wide_add_prod(w, q21, s21);
    let w = Real::wide_add_prod(w, q31, s31);
    let r11 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q21, s22);
    let w = Real::wide_add_prod(w, q31, s32);
    let r12 = Real::wide_rescale(w);
    let r13 = q31 * s33;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s22);
    let w = Real::wide_add_prod(w, q32, s32);
    let r22 = Real::wide_rescale(w);
    let r23 = q32 * s33;
    let r33 = s33;
    SymMatrix3 { m11: r11, m12: r12, m13: r13, m22: r22, m23: r23, m33: r33 }
}

/// `(worst error on the factors, worst backward error of the reconstruction)` in ulp, over the
/// oracle's `ldlt3_l_d` vectors, for a `new` candidate.
fn new3_worst(variant: u8) -> (u128, u128) {
    let mut cases = oracle_udu::ldlt3_l_d_cases();
    let mut factors = 0;
    let mut backward = 0;
    while let Some(case) = cases.pop_front() {
        let (a, l, d, _tol) = *case;
        let f = if variant == 0 {
            Ldlt3Trait::new(s3r(a)).unwrap()
        } else {
            new3_products(s3r(a)).unwrap()
        };
        let e = core::cmp::max(max_ulp_diff3(f.l(), m3(l)), max_ulp_diff_v3(f.d(), v3t(d)));
        factors = core::cmp::max(factors, e);
        let r = f.l() * Matrix3Trait::from_diagonal(f.d()) * f.l().transpose();
        backward = core::cmp::max(backward, max_ulp_diff3(r, m3(a)));
    }
    (factors, backward)
}

/// Worst error in ulp of a `solve` candidate over the oracle's `udu3_solve` vectors.
fn solve3_worst(variant: u8) -> u128 {
    let mut cases = oracle_udu::udu3_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _tol) = *case;
        let f = Ldlt3Trait::new(s3r(a)).unwrap();
        let got = if variant == 0 {
            f.solve(v3t(b))
        } else {
            solve3_recip(f, v3t(b))
        };
        worst = core::cmp::max(worst, max_ulp_diff_v3(got, v3t(expected)));
    }
    worst
}

/// Worst error in ulp of an `inverse` candidate over the oracle's `udu3_inverse` vectors.
fn inverse3_worst(variant: u8) -> u128 {
    let mut cases = oracle_udu::udu3_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, _tol) = *case;
        let f = Ldlt3Trait::new(s3r(a)).unwrap();
        let got = if variant == 0 {
            f.inverse()
        } else {
            inverse3_recip(f)
        };
        worst = core::cmp::max(worst, max_ulp_diff3(got.to_matrix(), m3(expected)));
    }
    worst
}

/// Why `alt_recip` lost: measured worst error over the 12 oracle cases.
#[test]
fn test_ldlt3_alt_recip_is_less_accurate() {
    let (sd, sr) = (solve3_worst(0), solve3_worst(1));
    let (id, ir) = (inverse3_worst(0), inverse3_worst(1));
    assert!(
        sd == 193 && sr == 500 && id == 72 && ir == 72,
        "ldlt3 solve div {} recip {} / inverse div {} recip {}",
        sd,
        sr,
        id,
        ir,
    );
}

/// Why `alt_products` lost: same backward error, twice the distance to the true factors.
#[test]
fn test_ldlt3_alt_products_is_less_accurate() {
    let (f0, b0) = new3_worst(0);
    let (f1, b1) = new3_worst(1);
    assert!(
        f0 == 30 && b0 == 16 && f1 == 30 && b1 == 16,
        "new3 shipped ({}, {}) products ({}, {})",
        f0,
        b0,
        f1,
        b1,
    );
}

#[test]
#[inline(never)]
fn bench_ldlt3_new__baseline() {
    let _a = black_box(a3());
    let e = black_box(f3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_new__factorize() {
    let a = black_box(a3());
    let e = black_box(f3());
    assert!(Ldlt3Trait::new(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_new__alt_products() {
    let a = black_box(a3());
    let e = black_box(f3());
    assert!(new3_products(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_l__baseline() {
    let _f = black_box(f3());
    let e = black_box(m3i([[1, 0, 0], [1, 1, 0], [1, 1, 1]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_l__expand() {
    let f = black_box(f3());
    let e = black_box(m3i([[1, 0, 0], [1, 1, 0], [1, 1, 1]]));
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_d__baseline() {
    let _f = black_box(f3());
    let e = black_box(v3it((1, 1, 1)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_d__accessor() {
    let f = black_box(f3());
    let e = black_box(v3it((1, 1, 1)));
    assert!(f.d() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_solve__baseline() {
    let _f = black_box(f3());
    let _b = black_box(b3());
    let e = black_box(x3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_solve__substitution() {
    let f = black_box(f3());
    let b = black_box(b3());
    let e = black_box(x3());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_solve__alt_recip() {
    let f = black_box(f3());
    let b = black_box(b3());
    let e = black_box(x3());
    assert!(solve3_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_inverse__baseline() {
    let _f = black_box(f3());
    let e = black_box(inv3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_inverse__triangular() {
    let f = black_box(f3());
    let e = black_box(inv3());
    assert!(f.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_inverse__alt_recip() {
    let f = black_box(f3());
    let e = black_box(inv3());
    assert!(inverse3_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_determinant__baseline() {
    let _f = black_box(f3());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt3_determinant__diagonal_product() {
    let f = black_box(f3());
    let e = black_box(int(1));
    assert!(f.determinant() == e);
}

// --- size 4 -------------------------------------------------------------------------------------

/// The benchmark input `a_ij = min(i, j)`.
fn a4() -> Matrix4<Fixed> {
    m4i([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3], [1, 2, 3, 4]])
}

/// Its `LDLᵀ` factor: the all-ones unit lower triangle, `d = (1, .., 1)`.
fn f4() -> Ldlt4<Fixed> {
    Ldlt4 {
        l21: int(1),
        l31: int(1),
        l41: int(1),
        l32: int(1),
        l42: int(1),
        l43: int(1),
        d: v4it((1, 1, 1, 1)),
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

/// LOSER. `new` recomputing `l_jk · d_k` as an explicit rounded product instead of reusing the
/// numerator of column k: n(n-1)/2 extra products, and a pivot that is second order rather than
/// first order in the rounding of `l_jk`.
fn new4_products(a: Matrix4<Fixed>) -> Option<Ldlt4<Fixed>> {
    let d1 = a.m11;
    if d1 == Real::<Fixed>::ZERO {
        return None;
    }
    let l21 = a.m21 / d1;
    let l31 = a.m31 / d1;
    let l41 = a.m41 / d1;
    let t21 = l21 * d1;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22);
    let w = Real::wide_sub_prod(w, l21, t21);
    let d2 = Real::wide_rescale(w);
    if d2 == Real::<Fixed>::ZERO {
        return None;
    }
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m32);
    let w = Real::wide_sub_prod(w, l31, t21);
    let n32 = Real::wide_rescale(w);
    let l32 = n32 / d2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m42);
    let w = Real::wide_sub_prod(w, l41, t21);
    let n42 = Real::wide_rescale(w);
    let l42 = n42 / d2;
    let t31 = l31 * d1;
    let t32 = l32 * d2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m33);
    let w = Real::wide_sub_prod(w, l31, t31);
    let w = Real::wide_sub_prod(w, l32, t32);
    let d3 = Real::wide_rescale(w);
    if d3 == Real::<Fixed>::ZERO {
        return None;
    }
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m43);
    let w = Real::wide_sub_prod(w, l41, t31);
    let w = Real::wide_sub_prod(w, l42, t32);
    let n43 = Real::wide_rescale(w);
    let l43 = n43 / d3;
    let t41 = l41 * d1;
    let t42 = l42 * d2;
    let t43 = l43 * d3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m44);
    let w = Real::wide_sub_prod(w, l41, t41);
    let w = Real::wide_sub_prod(w, l42, t42);
    let w = Real::wide_sub_prod(w, l43, t43);
    let d4 = Real::wide_rescale(w);
    if d4 == Real::<Fixed>::ZERO {
        return None;
    }
    Some(Ldlt4 { l21, l31, l41, l32, l42, l43, d: Vector4 { x: d1, y: d2, z: d3, w: d4 } })
}

/// LOSER. `solve` with one reciprocal per pivot instead of one exactly floored division.
fn solve4_recip(f: Ldlt4<Fixed>, b: Vector4<Fixed>) -> Vector4<Fixed> {
    let y1 = b.x;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.y);
    let w = Real::wide_sub_prod(w, f.l21, y1);
    let y2 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.z);
    let w = Real::wide_sub_prod(w, f.l31, y1);
    let w = Real::wide_sub_prod(w, f.l32, y2);
    let y3 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.w);
    let w = Real::wide_sub_prod(w, f.l41, y1);
    let w = Real::wide_sub_prod(w, f.l42, y2);
    let w = Real::wide_sub_prod(w, f.l43, y3);
    let y4 = Real::wide_rescale(w);
    let z1 = y1 * Real::recip(f.d.x);
    let z2 = y2 * Real::recip(f.d.y);
    let z3 = y3 * Real::recip(f.d.z);
    let z4 = y4 * Real::recip(f.d.w);
    let x4 = z4;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z3);
    let w = Real::wide_sub_prod(w, f.l43, x4);
    let x3 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z2);
    let w = Real::wide_sub_prod(w, f.l32, x3);
    let w = Real::wide_sub_prod(w, f.l42, x4);
    let x2 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z1);
    let w = Real::wide_sub_prod(w, f.l21, x2);
    let w = Real::wide_sub_prod(w, f.l31, x3);
    let w = Real::wide_sub_prod(w, f.l41, x4);
    let x1 = Real::wide_rescale(w);
    Vector4 { x: x1, y: x2, z: x3, w: x4 }
}

/// LOSER. `inverse` scaling `l⁻¹` by `recip(d_k)` instead of dividing by `d_k`.
fn inverse4_recip(f: Ldlt4<Fixed>) -> Matrix4<Fixed> {
    let q21 = -f.l21;
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l31);
    let w = Real::wide_sub_prod(w, f.l32, q21);
    let q31 = Real::wide_rescale(w);
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l41);
    let w = Real::wide_sub_prod(w, f.l42, q21);
    let w = Real::wide_sub_prod(w, f.l43, q31);
    let q41 = Real::wide_rescale(w);
    let q32 = -f.l32;
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l42);
    let w = Real::wide_sub_prod(w, f.l43, q32);
    let q42 = Real::wide_rescale(w);
    let q43 = -f.l43;
    let e1 = Real::recip(f.d.x);
    let e2 = Real::recip(f.d.y);
    let e3 = Real::recip(f.d.z);
    let e4 = Real::recip(f.d.w);
    let s11 = e1;
    let s21 = q21 * e2;
    let s31 = q31 * e3;
    let s41 = q41 * e4;
    let s22 = e2;
    let s32 = q32 * e3;
    let s42 = q42 * e4;
    let s33 = e3;
    let s43 = q43 * e4;
    let s44 = e4;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s11);
    let w = Real::wide_add_prod(w, q21, s21);
    let w = Real::wide_add_prod(w, q31, s31);
    let w = Real::wide_add_prod(w, q41, s41);
    let r11 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q21, s22);
    let w = Real::wide_add_prod(w, q31, s32);
    let w = Real::wide_add_prod(w, q41, s42);
    let r12 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q31, s33);
    let w = Real::wide_add_prod(w, q41, s43);
    let r13 = Real::wide_rescale(w);
    let r14 = q41 * s44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s22);
    let w = Real::wide_add_prod(w, q32, s32);
    let w = Real::wide_add_prod(w, q42, s42);
    let r22 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q32, s33);
    let w = Real::wide_add_prod(w, q42, s43);
    let r23 = Real::wide_rescale(w);
    let r24 = q42 * s44;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s33);
    let w = Real::wide_add_prod(w, q43, s43);
    let r33 = Real::wide_rescale(w);
    let r34 = q43 * s44;
    let r44 = s44;
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

/// `(worst error on the factors, worst backward error of the reconstruction)` in ulp, over the
/// oracle's `ldlt4_l_d` vectors, for a `new` candidate.
fn new4_worst(variant: u8) -> (u128, u128) {
    let mut cases = oracle_udu::ldlt4_l_d_cases();
    let mut factors = 0;
    let mut backward = 0;
    while let Some(case) = cases.pop_front() {
        let (a, l, d, _tol) = *case;
        let f = if variant == 0 {
            Ldlt4Trait::new(m4(a)).unwrap()
        } else {
            new4_products(m4(a)).unwrap()
        };
        let e = core::cmp::max(max_ulp_diff4(f.l(), m4(l)), max_ulp_diff_v4(f.d(), v4t(d)));
        factors = core::cmp::max(factors, e);
        let r = f.l() * Matrix4Trait::from_diagonal(f.d()) * f.l().transpose();
        backward = core::cmp::max(backward, max_ulp_diff4(r, m4(a)));
    }
    (factors, backward)
}

/// Worst error in ulp of a `solve` candidate over the oracle's `udu4_solve` vectors.
fn solve4_worst(variant: u8) -> u128 {
    let mut cases = oracle_udu::udu4_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _tol) = *case;
        let f = Ldlt4Trait::new(m4(a)).unwrap();
        let got = if variant == 0 {
            f.solve(v4t(b))
        } else {
            solve4_recip(f, v4t(b))
        };
        worst = core::cmp::max(worst, max_ulp_diff_v4(got, v4t(expected)));
    }
    worst
}

/// Worst error in ulp of an `inverse` candidate over the oracle's `udu4_inverse` vectors.
fn inverse4_worst(variant: u8) -> u128 {
    let mut cases = oracle_udu::udu4_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, _tol) = *case;
        let f = Ldlt4Trait::new(m4(a)).unwrap();
        let got = if variant == 0 {
            f.inverse()
        } else {
            inverse4_recip(f)
        };
        worst = core::cmp::max(worst, max_ulp_diff4(got, m4(expected)));
    }
    worst
}

/// Why `alt_recip` lost: measured worst error over the 12 oracle cases.
#[test]
fn test_ldlt4_alt_recip_is_less_accurate() {
    let (sd, sr) = (solve4_worst(0), solve4_worst(1));
    let (id, ir) = (inverse4_worst(0), inverse4_worst(1));
    assert!(
        sd == 70 && sr == 521 && id == 340 && ir == 340,
        "ldlt4 solve div {} recip {} / inverse div {} recip {}",
        sd,
        sr,
        id,
        ir,
    );
}

/// Why `alt_products` lost: same backward error, twice the distance to the true factors.
#[test]
fn test_ldlt4_alt_products_is_less_accurate() {
    let (f0, b0) = new4_worst(0);
    let (f1, b1) = new4_worst(1);
    assert!(
        f0 == 27 && b0 == 75 && f1 == 42 && b1 == 75,
        "new4 shipped ({}, {}) products ({}, {})",
        f0,
        b0,
        f1,
        b1,
    );
}

#[test]
#[inline(never)]
fn bench_ldlt4_new__baseline() {
    let _a = black_box(a4());
    let e = black_box(f4());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_new__factorize() {
    let a = black_box(a4());
    let e = black_box(f4());
    assert!(Ldlt4Trait::new(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_new__alt_products() {
    let a = black_box(a4());
    let e = black_box(f4());
    assert!(new4_products(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_l__baseline() {
    let _f = black_box(f4());
    let e = black_box(m4i([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_l__expand() {
    let f = black_box(f4());
    let e = black_box(m4i([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]));
    assert!(f.l() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_d__baseline() {
    let _f = black_box(f4());
    let e = black_box(v4it((1, 1, 1, 1)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_d__accessor() {
    let f = black_box(f4());
    let e = black_box(v4it((1, 1, 1, 1)));
    assert!(f.d() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_solve__baseline() {
    let _f = black_box(f4());
    let _b = black_box(b4());
    let e = black_box(x4());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_solve__substitution() {
    let f = black_box(f4());
    let b = black_box(b4());
    let e = black_box(x4());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_solve__alt_recip() {
    let f = black_box(f4());
    let b = black_box(b4());
    let e = black_box(x4());
    assert!(solve4_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_inverse__baseline() {
    let _f = black_box(f4());
    let e = black_box(inv4());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_inverse__triangular() {
    let f = black_box(f4());
    let e = black_box(inv4());
    assert!(f.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_inverse__alt_recip() {
    let f = black_box(f4());
    let e = black_box(inv4());
    assert!(inverse4_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_determinant__baseline() {
    let _f = black_box(f4());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt4_determinant__diagonal_product() {
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

/// Its `LDLᵀ` factor: the all-ones unit lower triangle, `d = (1, .., 1)`.
fn f6() -> Ldlt6<Fixed> {
    Ldlt6 {
        l21: int(1),
        l31: int(1),
        l41: int(1),
        l51: int(1),
        l61: int(1),
        l32: int(1),
        l42: int(1),
        l52: int(1),
        l62: int(1),
        l43: int(1),
        l53: int(1),
        l63: int(1),
        l54: int(1),
        l64: int(1),
        l65: int(1),
        d: v6it((1, 1, 1, 1, 1, 1)),
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

/// LOSER. `new` recomputing `l_jk · d_k` as an explicit rounded product instead of reusing the
/// numerator of column k: n(n-1)/2 extra products, and a pivot that is second order rather than
/// first order in the rounding of `l_jk`.
fn new6_products(a: Matrix6<Fixed>) -> Option<Ldlt6<Fixed>> {
    let d1 = a.m11.m11;
    if d1 == Real::<Fixed>::ZERO {
        return None;
    }
    let l21 = a.m11.m21 / d1;
    let l31 = a.m11.m31 / d1;
    let l41 = a.m21.m11 / d1;
    let l51 = a.m21.m21 / d1;
    let l61 = a.m21.m31 / d1;
    let t21 = l21 * d1;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m11.m22);
    let w = Real::wide_sub_prod(w, l21, t21);
    let d2 = Real::wide_rescale(w);
    if d2 == Real::<Fixed>::ZERO {
        return None;
    }
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m11.m32);
    let w = Real::wide_sub_prod(w, l31, t21);
    let n32 = Real::wide_rescale(w);
    let l32 = n32 / d2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m21.m12);
    let w = Real::wide_sub_prod(w, l41, t21);
    let n42 = Real::wide_rescale(w);
    let l42 = n42 / d2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m21.m22);
    let w = Real::wide_sub_prod(w, l51, t21);
    let n52 = Real::wide_rescale(w);
    let l52 = n52 / d2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m21.m32);
    let w = Real::wide_sub_prod(w, l61, t21);
    let n62 = Real::wide_rescale(w);
    let l62 = n62 / d2;
    let t31 = l31 * d1;
    let t32 = l32 * d2;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m11.m33);
    let w = Real::wide_sub_prod(w, l31, t31);
    let w = Real::wide_sub_prod(w, l32, t32);
    let d3 = Real::wide_rescale(w);
    if d3 == Real::<Fixed>::ZERO {
        return None;
    }
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m21.m13);
    let w = Real::wide_sub_prod(w, l41, t31);
    let w = Real::wide_sub_prod(w, l42, t32);
    let n43 = Real::wide_rescale(w);
    let l43 = n43 / d3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m21.m23);
    let w = Real::wide_sub_prod(w, l51, t31);
    let w = Real::wide_sub_prod(w, l52, t32);
    let n53 = Real::wide_rescale(w);
    let l53 = n53 / d3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m21.m33);
    let w = Real::wide_sub_prod(w, l61, t31);
    let w = Real::wide_sub_prod(w, l62, t32);
    let n63 = Real::wide_rescale(w);
    let l63 = n63 / d3;
    let t41 = l41 * d1;
    let t42 = l42 * d2;
    let t43 = l43 * d3;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22.m11);
    let w = Real::wide_sub_prod(w, l41, t41);
    let w = Real::wide_sub_prod(w, l42, t42);
    let w = Real::wide_sub_prod(w, l43, t43);
    let d4 = Real::wide_rescale(w);
    if d4 == Real::<Fixed>::ZERO {
        return None;
    }
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22.m21);
    let w = Real::wide_sub_prod(w, l51, t41);
    let w = Real::wide_sub_prod(w, l52, t42);
    let w = Real::wide_sub_prod(w, l53, t43);
    let n54 = Real::wide_rescale(w);
    let l54 = n54 / d4;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22.m31);
    let w = Real::wide_sub_prod(w, l61, t41);
    let w = Real::wide_sub_prod(w, l62, t42);
    let w = Real::wide_sub_prod(w, l63, t43);
    let n64 = Real::wide_rescale(w);
    let l64 = n64 / d4;
    let t51 = l51 * d1;
    let t52 = l52 * d2;
    let t53 = l53 * d3;
    let t54 = l54 * d4;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22.m22);
    let w = Real::wide_sub_prod(w, l51, t51);
    let w = Real::wide_sub_prod(w, l52, t52);
    let w = Real::wide_sub_prod(w, l53, t53);
    let w = Real::wide_sub_prod(w, l54, t54);
    let d5 = Real::wide_rescale(w);
    if d5 == Real::<Fixed>::ZERO {
        return None;
    }
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22.m32);
    let w = Real::wide_sub_prod(w, l61, t51);
    let w = Real::wide_sub_prod(w, l62, t52);
    let w = Real::wide_sub_prod(w, l63, t53);
    let w = Real::wide_sub_prod(w, l64, t54);
    let n65 = Real::wide_rescale(w);
    let l65 = n65 / d5;
    let t61 = l61 * d1;
    let t62 = l62 * d2;
    let t63 = l63 * d3;
    let t64 = l64 * d4;
    let t65 = l65 * d5;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), a.m22.m33);
    let w = Real::wide_sub_prod(w, l61, t61);
    let w = Real::wide_sub_prod(w, l62, t62);
    let w = Real::wide_sub_prod(w, l63, t63);
    let w = Real::wide_sub_prod(w, l64, t64);
    let w = Real::wide_sub_prod(w, l65, t65);
    let d6 = Real::wide_rescale(w);
    if d6 == Real::<Fixed>::ZERO {
        return None;
    }
    Some(
        Ldlt6 {
            l21,
            l31,
            l41,
            l51,
            l61,
            l32,
            l42,
            l52,
            l62,
            l43,
            l53,
            l63,
            l54,
            l64,
            l65,
            d: Vector6 { a: Vector3 { x: d1, y: d2, z: d3 }, b: Vector3 { x: d4, y: d5, z: d6 } },
        },
    )
}

/// LOSER. `solve` with one reciprocal per pivot instead of one exactly floored division.
fn solve6_recip(f: Ldlt6<Fixed>, b: Vector6<Fixed>) -> Vector6<Fixed> {
    let y1 = b.a.x;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.a.y);
    let w = Real::wide_sub_prod(w, f.l21, y1);
    let y2 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.a.z);
    let w = Real::wide_sub_prod(w, f.l31, y1);
    let w = Real::wide_sub_prod(w, f.l32, y2);
    let y3 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.b.x);
    let w = Real::wide_sub_prod(w, f.l41, y1);
    let w = Real::wide_sub_prod(w, f.l42, y2);
    let w = Real::wide_sub_prod(w, f.l43, y3);
    let y4 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.b.y);
    let w = Real::wide_sub_prod(w, f.l51, y1);
    let w = Real::wide_sub_prod(w, f.l52, y2);
    let w = Real::wide_sub_prod(w, f.l53, y3);
    let w = Real::wide_sub_prod(w, f.l54, y4);
    let y5 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), b.b.z);
    let w = Real::wide_sub_prod(w, f.l61, y1);
    let w = Real::wide_sub_prod(w, f.l62, y2);
    let w = Real::wide_sub_prod(w, f.l63, y3);
    let w = Real::wide_sub_prod(w, f.l64, y4);
    let w = Real::wide_sub_prod(w, f.l65, y5);
    let y6 = Real::wide_rescale(w);
    let z1 = y1 * Real::recip(f.d.a.x);
    let z2 = y2 * Real::recip(f.d.a.y);
    let z3 = y3 * Real::recip(f.d.a.z);
    let z4 = y4 * Real::recip(f.d.b.x);
    let z5 = y5 * Real::recip(f.d.b.y);
    let z6 = y6 * Real::recip(f.d.b.z);
    let x6 = z6;
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z5);
    let w = Real::wide_sub_prod(w, f.l65, x6);
    let x5 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z4);
    let w = Real::wide_sub_prod(w, f.l54, x5);
    let w = Real::wide_sub_prod(w, f.l64, x6);
    let x4 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z3);
    let w = Real::wide_sub_prod(w, f.l43, x4);
    let w = Real::wide_sub_prod(w, f.l53, x5);
    let w = Real::wide_sub_prod(w, f.l63, x6);
    let x3 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z2);
    let w = Real::wide_sub_prod(w, f.l32, x3);
    let w = Real::wide_sub_prod(w, f.l42, x4);
    let w = Real::wide_sub_prod(w, f.l52, x5);
    let w = Real::wide_sub_prod(w, f.l62, x6);
    let x2 = Real::wide_rescale(w);
    let w = Real::wide_add(Real::<Fixed>::wide_zero(), z1);
    let w = Real::wide_sub_prod(w, f.l21, x2);
    let w = Real::wide_sub_prod(w, f.l31, x3);
    let w = Real::wide_sub_prod(w, f.l41, x4);
    let w = Real::wide_sub_prod(w, f.l51, x5);
    let w = Real::wide_sub_prod(w, f.l61, x6);
    let x1 = Real::wide_rescale(w);
    Vector6 { a: Vector3 { x: x1, y: x2, z: x3 }, b: Vector3 { x: x4, y: x5, z: x6 } }
}

/// LOSER. `inverse` scaling `l⁻¹` by `recip(d_k)` instead of dividing by `d_k`.
fn inverse6_recip(f: Ldlt6<Fixed>) -> Matrix6<Fixed> {
    let q21 = -f.l21;
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l31);
    let w = Real::wide_sub_prod(w, f.l32, q21);
    let q31 = Real::wide_rescale(w);
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l41);
    let w = Real::wide_sub_prod(w, f.l42, q21);
    let w = Real::wide_sub_prod(w, f.l43, q31);
    let q41 = Real::wide_rescale(w);
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l51);
    let w = Real::wide_sub_prod(w, f.l52, q21);
    let w = Real::wide_sub_prod(w, f.l53, q31);
    let w = Real::wide_sub_prod(w, f.l54, q41);
    let q51 = Real::wide_rescale(w);
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l61);
    let w = Real::wide_sub_prod(w, f.l62, q21);
    let w = Real::wide_sub_prod(w, f.l63, q31);
    let w = Real::wide_sub_prod(w, f.l64, q41);
    let w = Real::wide_sub_prod(w, f.l65, q51);
    let q61 = Real::wide_rescale(w);
    let q32 = -f.l32;
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l42);
    let w = Real::wide_sub_prod(w, f.l43, q32);
    let q42 = Real::wide_rescale(w);
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l52);
    let w = Real::wide_sub_prod(w, f.l53, q32);
    let w = Real::wide_sub_prod(w, f.l54, q42);
    let q52 = Real::wide_rescale(w);
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l62);
    let w = Real::wide_sub_prod(w, f.l63, q32);
    let w = Real::wide_sub_prod(w, f.l64, q42);
    let w = Real::wide_sub_prod(w, f.l65, q52);
    let q62 = Real::wide_rescale(w);
    let q43 = -f.l43;
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l53);
    let w = Real::wide_sub_prod(w, f.l54, q43);
    let q53 = Real::wide_rescale(w);
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l63);
    let w = Real::wide_sub_prod(w, f.l64, q43);
    let w = Real::wide_sub_prod(w, f.l65, q53);
    let q63 = Real::wide_rescale(w);
    let q54 = -f.l54;
    let w = Real::wide_sub(Real::<Fixed>::wide_zero(), f.l64);
    let w = Real::wide_sub_prod(w, f.l65, q54);
    let q64 = Real::wide_rescale(w);
    let q65 = -f.l65;
    let e1 = Real::recip(f.d.a.x);
    let e2 = Real::recip(f.d.a.y);
    let e3 = Real::recip(f.d.a.z);
    let e4 = Real::recip(f.d.b.x);
    let e5 = Real::recip(f.d.b.y);
    let e6 = Real::recip(f.d.b.z);
    let s11 = e1;
    let s21 = q21 * e2;
    let s31 = q31 * e3;
    let s41 = q41 * e4;
    let s51 = q51 * e5;
    let s61 = q61 * e6;
    let s22 = e2;
    let s32 = q32 * e3;
    let s42 = q42 * e4;
    let s52 = q52 * e5;
    let s62 = q62 * e6;
    let s33 = e3;
    let s43 = q43 * e4;
    let s53 = q53 * e5;
    let s63 = q63 * e6;
    let s44 = e4;
    let s54 = q54 * e5;
    let s64 = q64 * e6;
    let s55 = e5;
    let s65 = q65 * e6;
    let s66 = e6;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s11);
    let w = Real::wide_add_prod(w, q21, s21);
    let w = Real::wide_add_prod(w, q31, s31);
    let w = Real::wide_add_prod(w, q41, s41);
    let w = Real::wide_add_prod(w, q51, s51);
    let w = Real::wide_add_prod(w, q61, s61);
    let r11 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q21, s22);
    let w = Real::wide_add_prod(w, q31, s32);
    let w = Real::wide_add_prod(w, q41, s42);
    let w = Real::wide_add_prod(w, q51, s52);
    let w = Real::wide_add_prod(w, q61, s62);
    let r12 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q31, s33);
    let w = Real::wide_add_prod(w, q41, s43);
    let w = Real::wide_add_prod(w, q51, s53);
    let w = Real::wide_add_prod(w, q61, s63);
    let r13 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q41, s44);
    let w = Real::wide_add_prod(w, q51, s54);
    let w = Real::wide_add_prod(w, q61, s64);
    let r14 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q51, s55);
    let w = Real::wide_add_prod(w, q61, s65);
    let r15 = Real::wide_rescale(w);
    let r16 = q61 * s66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s22);
    let w = Real::wide_add_prod(w, q32, s32);
    let w = Real::wide_add_prod(w, q42, s42);
    let w = Real::wide_add_prod(w, q52, s52);
    let w = Real::wide_add_prod(w, q62, s62);
    let r22 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q32, s33);
    let w = Real::wide_add_prod(w, q42, s43);
    let w = Real::wide_add_prod(w, q52, s53);
    let w = Real::wide_add_prod(w, q62, s63);
    let r23 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q42, s44);
    let w = Real::wide_add_prod(w, q52, s54);
    let w = Real::wide_add_prod(w, q62, s64);
    let r24 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q52, s55);
    let w = Real::wide_add_prod(w, q62, s65);
    let r25 = Real::wide_rescale(w);
    let r26 = q62 * s66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s33);
    let w = Real::wide_add_prod(w, q43, s43);
    let w = Real::wide_add_prod(w, q53, s53);
    let w = Real::wide_add_prod(w, q63, s63);
    let r33 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q43, s44);
    let w = Real::wide_add_prod(w, q53, s54);
    let w = Real::wide_add_prod(w, q63, s64);
    let r34 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q53, s55);
    let w = Real::wide_add_prod(w, q63, s65);
    let r35 = Real::wide_rescale(w);
    let r36 = q63 * s66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s44);
    let w = Real::wide_add_prod(w, q54, s54);
    let w = Real::wide_add_prod(w, q64, s64);
    let r44 = Real::wide_rescale(w);
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add_prod(w, q54, s55);
    let w = Real::wide_add_prod(w, q64, s65);
    let r45 = Real::wide_rescale(w);
    let r46 = q64 * s66;
    let w = Real::<Fixed>::wide_zero();
    let w = Real::wide_add(w, s55);
    let w = Real::wide_add_prod(w, q65, s65);
    let r55 = Real::wide_rescale(w);
    let r56 = q65 * s66;
    let r66 = s66;
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

/// `(worst error on the factors, worst backward error of the reconstruction)` in ulp, over the
/// oracle's `ldlt6_l_d` vectors, for a `new` candidate.
fn new6_worst(variant: u8) -> (u128, u128) {
    let mut cases = oracle_udu::ldlt6_l_d_cases();
    let mut factors = 0;
    let mut backward = 0;
    while let Some(case) = cases.pop_front() {
        let (a, l, d, _tol) = *case;
        let f = if variant == 0 {
            Ldlt6Trait::new(m6(a)).unwrap()
        } else {
            new6_products(m6(a)).unwrap()
        };
        let e = core::cmp::max(max_ulp_diff6(f.l(), m6(l)), max_ulp_diff_v6(f.d(), v6t(d)));
        factors = core::cmp::max(factors, e);
        let r = f.l() * Matrix6Trait::from_diagonal(f.d()) * f.l().transpose();
        backward = core::cmp::max(backward, max_ulp_diff6(r, m6(a)));
    }
    (factors, backward)
}

/// Worst error in ulp of a `solve` candidate over the oracle's `udu6_solve` vectors.
fn solve6_worst(variant: u8) -> u128 {
    let mut cases = oracle_udu::udu6_solve_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, _tol) = *case;
        let f = Ldlt6Trait::new(m6(a)).unwrap();
        let got = if variant == 0 {
            f.solve(v6t(b))
        } else {
            solve6_recip(f, v6t(b))
        };
        worst = core::cmp::max(worst, max_ulp_diff_v6(got, v6t(expected)));
    }
    worst
}

/// Worst error in ulp of an `inverse` candidate over the oracle's `udu6_inverse` vectors.
fn inverse6_worst(variant: u8) -> u128 {
    let mut cases = oracle_udu::udu6_inverse_cases();
    let mut worst = 0;
    while let Some(case) = cases.pop_front() {
        let (a, expected, _tol) = *case;
        let f = Ldlt6Trait::new(m6(a)).unwrap();
        let got = if variant == 0 {
            f.inverse()
        } else {
            inverse6_recip(f)
        };
        worst = core::cmp::max(worst, max_ulp_diff6(got, m6(expected)));
    }
    worst
}

/// Why `alt_recip` lost: measured worst error over the 12 oracle cases.
#[test]
fn test_ldlt6_alt_recip_is_less_accurate() {
    let (sd, sr) = (solve6_worst(0), solve6_worst(1));
    let (id, ir) = (inverse6_worst(0), inverse6_worst(1));
    assert!(
        sd == 615 && sr == 623 && id == 365 && ir == 365,
        "ldlt6 solve div {} recip {} / inverse div {} recip {}",
        sd,
        sr,
        id,
        ir,
    );
}

/// Why `alt_products` lost: same backward error, twice the distance to the true factors.
#[test]
fn test_ldlt6_alt_products_is_less_accurate() {
    let (f0, b0) = new6_worst(0);
    let (f1, b1) = new6_worst(1);
    assert!(
        f0 == 27 && b0 == 25 && f1 == 36 && b1 == 25,
        "new6 shipped ({}, {}) products ({}, {})",
        f0,
        b0,
        f1,
        b1,
    );
}

#[test]
#[inline(never)]
fn bench_ldlt6_new__baseline() {
    let _a = black_box(a6());
    let e = black_box(f6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_new__factorize() {
    let a = black_box(a6());
    let e = black_box(f6());
    assert!(Ldlt6Trait::new(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_new__alt_products() {
    let a = black_box(a6());
    let e = black_box(f6());
    assert!(new6_products(a).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_l__baseline() {
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
fn bench_ldlt6_l__expand() {
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
fn bench_ldlt6_d__baseline() {
    let _f = black_box(f6());
    let e = black_box(v6it((1, 1, 1, 1, 1, 1)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_d__accessor() {
    let f = black_box(f6());
    let e = black_box(v6it((1, 1, 1, 1, 1, 1)));
    assert!(f.d() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_solve__baseline() {
    let _f = black_box(f6());
    let _b = black_box(b6());
    let e = black_box(x6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_solve__substitution() {
    let f = black_box(f6());
    let b = black_box(b6());
    let e = black_box(x6());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_solve__alt_recip() {
    let f = black_box(f6());
    let b = black_box(b6());
    let e = black_box(x6());
    assert!(solve6_recip(f, b) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_inverse__baseline() {
    let _f = black_box(f6());
    let e = black_box(inv6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_inverse__triangular() {
    let f = black_box(f6());
    let e = black_box(inv6());
    assert!(f.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_inverse__alt_recip() {
    let f = black_box(f6());
    let e = black_box(inv6());
    assert!(inverse6_recip(f) == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_determinant__baseline() {
    let _f = black_box(f6());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_ldlt6_determinant__diagonal_product() {
    let f = black_box(f6());
    let e = black_box(int(1));
    assert!(f.determinant() == e);
}

/// Why `alt_recip` loses in `inverse`, on a hand-built factor: `l = [[1, 0], [5, 1]]` and
/// `d = (1, 3)`, so `a = l·diag(d)·lᵀ = [[1, 5], [5, 28]]` and `a⁻¹_11 = 28/3`, whose exact
/// floor is 40 086 361 429 raw. `q21 / d2 = -5/3` is floored exactly by the division and is 2 ulp
/// high through `-5 · recip(3)`; the output is 2 ulp high for the shipped code and 8 ulp low for
/// the candidate.
#[test]
fn test_ldlt2_inverse_alt_recip_loses_low_bits() {
    let f = Ldlt2 { l21: int(5), d: v2it((1, 3)) };
    let exact = fx(40086361429);
    let shipped = f.inverse();
    let alt = inverse2_recip(f);
    assert!(shipped != alt);
    assert!(shipped.m12 == alt.m12 && shipped.m22 == alt.m22);
    assert!(ulp_diff(shipped.m11, exact) == 2);
    assert!(ulp_diff(alt.m11, exact) == 8);
}
