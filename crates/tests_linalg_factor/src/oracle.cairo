//! Oracle tests (suite `solve` of `tools/oracle`, upstream nalgebra 0.35 in f64 on the same raw
//! inputs): every case within the oracle's tolerance plus one relative ulp of the expected
//! value (`oracle_tol`, the convention of the `linalg` oracle tests); a failure reports by how
//! much the worst case exceeds it.

use core::cmp::max;
use fixed::Fixed;
use nalgebra::linalg::{
    Cholesky2Trait, Cholesky3Trait, Cholesky4Trait, Cholesky6Trait, Lu2Trait, Lu3Trait, Lu4Trait,
    Lu6Trait, Qr2Trait, Qr3Trait, Qr4Trait, reflection_axis_mut,
};
use nalgebra::{
    GivensRotate, GivensRotateRows, GivensRotationTrait, Matrix2, Matrix2Trait, Matrix2x3,
    Matrix2x3Trait, Matrix3, Matrix3Trait, Matrix3x2, Matrix3x2Trait, Matrix4, Matrix4Trait,
    Matrix4x3, Matrix4x3Trait, Matrix6, Matrix6Trait, Matrix6x3, Matrix6x3Trait, Vector2,
    Vector2Trait, Vector3, Vector3Trait, Vector4, Vector4Trait, Vector5, Vector5Trait, Vector6,
    Vector6Trait,
};
use nalgebra_tests_utils::{abs_raw, excess, fx, oracle_tol, ulp_diff};
use crate::oracle_factor as oracle;

fn vec2(t: (i64, i64)) -> Vector2<Fixed> {
    let (a0, a1) = t;
    Vector2Trait::new(fx(a0), fx(a1))
}

/// `givens_new`: `GivensRotation::new(c, s)` against upstream (f64).
#[test]
fn test_oracle_givens_new() {
    let cases = oracle::givens_new_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let ((c, s), (ec, es, er), tol) = *cases.at(i);
        let (g, r) = GivensRotationTrait::new(fx(c), fx(s));
        let (err, mag) = (ulp_diff(g.c(), fx(ec)), 0);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(g.s(), fx(es)), 0);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(r, fx(er)), abs_raw(fx(er)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `givens_cancel_y`: `GivensRotation::cancel_y(v)` against upstream (f64).
#[test]
fn test_oracle_givens_cancel_y() {
    let cases = oracle::givens_cancel_y_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (rv, (ec, es, er), tol) = *cases.at(i);
        let (g, r) = GivensRotationTrait::cancel_y(vec2(rv)).unwrap();
        let (err, mag) = (ulp_diff(g.c(), fx(ec)), 0);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(g.s(), fx(es)), 0);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(r, fx(er)), abs_raw(fx(er)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `givens_cancel_x`: `GivensRotation::cancel_x(v)` against upstream (f64).
#[test]
fn test_oracle_givens_cancel_x() {
    let cases = oracle::givens_cancel_x_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (rv, (ec, es, er), tol) = *cases.at(i);
        let (g, r) = GivensRotationTrait::cancel_x(vec2(rv)).unwrap();
        let (err, mag) = (ulp_diff(g.c(), fx(ec)), 0);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(g.s(), fx(es)), 0);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(r, fx(er)), abs_raw(fx(er)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn mat_matrix2x3(m: [[i64; 3]; 2]) -> Matrix2x3<Fixed> {
    let [[a00, a01, a02], [a10, a11, a12]] = m;
    Matrix2x3Trait::new(fx(a00), fx(a01), fx(a02), fx(a10), fx(a11), fx(a12))
}

fn err_matrix2x3(x: Matrix2x3<Fixed>, e: [[i64; 3]; 2]) -> (u128, u128) {
    let [[e00, e01, e02], [e10, e11, e12]] = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.m11, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.m12, fx(e01)));
    mag = max(mag, abs_raw(fx(e01)));
    err = max(err, ulp_diff(x.m13, fx(e02)));
    mag = max(mag, abs_raw(fx(e02)));
    err = max(err, ulp_diff(x.m21, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.m22, fx(e11)));
    mag = max(mag, abs_raw(fx(e11)));
    err = max(err, ulp_diff(x.m23, fx(e12)));
    mag = max(mag, abs_raw(fx(e12)));
    (err, mag)
}

/// `givens_rotate`: `rotate` on a 2x3 matrix against upstream (f64).
#[test]
fn test_oracle_givens_rotate() {
    let cases = oracle::givens_rotate_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let ((c, s), rm, rx, tol) = *cases.at(i);
        let g = GivensRotationTrait::new_unchecked(fx(c), fx(s));
        let mut m = mat_matrix2x3(rm);
        g.rotate(ref m);
        let (err, mag) = err_matrix2x3(m, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn mat_matrix3x2(m: [[i64; 2]; 3]) -> Matrix3x2<Fixed> {
    let [[a00, a01], [a10, a11], [a20, a21]] = m;
    Matrix3x2Trait::new(fx(a00), fx(a01), fx(a10), fx(a11), fx(a20), fx(a21))
}

fn err_matrix3x2(x: Matrix3x2<Fixed>, e: [[i64; 2]; 3]) -> (u128, u128) {
    let [[e00, e01], [e10, e11], [e20, e21]] = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.m11, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.m12, fx(e01)));
    mag = max(mag, abs_raw(fx(e01)));
    err = max(err, ulp_diff(x.m21, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.m22, fx(e11)));
    mag = max(mag, abs_raw(fx(e11)));
    err = max(err, ulp_diff(x.m31, fx(e20)));
    mag = max(mag, abs_raw(fx(e20)));
    err = max(err, ulp_diff(x.m32, fx(e21)));
    mag = max(mag, abs_raw(fx(e21)));
    (err, mag)
}

/// `givens_rotate_rows`: `rotate_rows` on a 3x2 matrix against upstream (f64).
#[test]
fn test_oracle_givens_rotate_rows() {
    let cases = oracle::givens_rotate_rows_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let ((c, s), rm, rx, tol) = *cases.at(i);
        let g = GivensRotationTrait::new_unchecked(fx(c), fx(s));
        let mut m = mat_matrix3x2(rm);
        g.rotate_rows(ref m);
        let (err, mag) = err_matrix3x2(m, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn err_vector2_t(x: Vector2<Fixed>, e: (i64, i64)) -> (u128, u128) {
    let (e00, e10) = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.x, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.y, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    (err, mag)
}

/// `reflection_axis2`: `reflection_axis_mut` against upstream (f64).
#[test]
fn test_oracle_reflection_axis2() {
    let cases = oracle::reflection_axis2_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (rv, raxis, rr, tol) = *cases.at(i);
        let mut v = vec2(rv);
        let (r, ok) = reflection_axis_mut(ref v);
        assert!(ok);
        let (err, mag) = err_vector2_t(v, raxis);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(r, fx(rr)), abs_raw(fx(rr)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn vec3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (a0, a1, a2) = t;
    Vector3Trait::new(fx(a0), fx(a1), fx(a2))
}

fn err_vector3_t(x: Vector3<Fixed>, e: (i64, i64, i64)) -> (u128, u128) {
    let (e00, e10, e20) = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.x, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.y, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.z, fx(e20)));
    mag = max(mag, abs_raw(fx(e20)));
    (err, mag)
}

/// `reflection_axis3`: `reflection_axis_mut` against upstream (f64).
#[test]
fn test_oracle_reflection_axis3() {
    let cases = oracle::reflection_axis3_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (rv, raxis, rr, tol) = *cases.at(i);
        let mut v = vec3(rv);
        let (r, ok) = reflection_axis_mut(ref v);
        assert!(ok);
        let (err, mag) = err_vector3_t(v, raxis);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(r, fx(rr)), abs_raw(fx(rr)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn vec4(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (a0, a1, a2, a3) = t;
    Vector4Trait::new(fx(a0), fx(a1), fx(a2), fx(a3))
}

fn err_vector4_t(x: Vector4<Fixed>, e: (i64, i64, i64, i64)) -> (u128, u128) {
    let (e00, e10, e20, e30) = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.x, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.y, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.z, fx(e20)));
    mag = max(mag, abs_raw(fx(e20)));
    err = max(err, ulp_diff(x.w, fx(e30)));
    mag = max(mag, abs_raw(fx(e30)));
    (err, mag)
}

/// `reflection_axis4`: `reflection_axis_mut` against upstream (f64).
#[test]
fn test_oracle_reflection_axis4() {
    let cases = oracle::reflection_axis4_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (rv, raxis, rr, tol) = *cases.at(i);
        let mut v = vec4(rv);
        let (r, ok) = reflection_axis_mut(ref v);
        assert!(ok);
        let (err, mag) = err_vector4_t(v, raxis);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(r, fx(rr)), abs_raw(fx(rr)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn vec5(t: (i64, i64, i64, i64, i64)) -> Vector5<Fixed> {
    let (a0, a1, a2, a3, a4) = t;
    Vector5Trait::new(fx(a0), fx(a1), fx(a2), fx(a3), fx(a4))
}

fn err_vector5_t(x: Vector5<Fixed>, e: (i64, i64, i64, i64, i64)) -> (u128, u128) {
    let (e00, e10, e20, e30, e40) = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.x, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.y, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.z, fx(e20)));
    mag = max(mag, abs_raw(fx(e20)));
    err = max(err, ulp_diff(x.w, fx(e30)));
    mag = max(mag, abs_raw(fx(e30)));
    err = max(err, ulp_diff(x.a, fx(e40)));
    mag = max(mag, abs_raw(fx(e40)));
    (err, mag)
}

/// `reflection_axis5`: `reflection_axis_mut` against upstream (f64).
#[test]
fn test_oracle_reflection_axis5() {
    let cases = oracle::reflection_axis5_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (rv, raxis, rr, tol) = *cases.at(i);
        let mut v = vec5(rv);
        let (r, ok) = reflection_axis_mut(ref v);
        assert!(ok);
        let (err, mag) = err_vector5_t(v, raxis);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(r, fx(rr)), abs_raw(fx(rr)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn vec6(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (a0, a1, a2, a3, a4, a5) = t;
    Vector6Trait::new(fx(a0), fx(a1), fx(a2), fx(a3), fx(a4), fx(a5))
}

fn err_vector6_t(x: Vector6<Fixed>, e: (i64, i64, i64, i64, i64, i64)) -> (u128, u128) {
    let (e00, e10, e20, e30, e40, e50) = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.x, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.y, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.z, fx(e20)));
    mag = max(mag, abs_raw(fx(e20)));
    err = max(err, ulp_diff(x.w, fx(e30)));
    mag = max(mag, abs_raw(fx(e30)));
    err = max(err, ulp_diff(x.a, fx(e40)));
    mag = max(mag, abs_raw(fx(e40)));
    err = max(err, ulp_diff(x.b, fx(e50)));
    mag = max(mag, abs_raw(fx(e50)));
    (err, mag)
}

/// `reflection_axis6`: `reflection_axis_mut` against upstream (f64).
#[test]
fn test_oracle_reflection_axis6() {
    let cases = oracle::reflection_axis6_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (rv, raxis, rr, tol) = *cases.at(i);
        let mut v = vec6(rv);
        let (r, ok) = reflection_axis_mut(ref v);
        assert!(ok);
        let (err, mag) = err_vector6_t(v, raxis);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        let (err, mag) = (ulp_diff(r, fx(rr)), abs_raw(fx(rr)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn mat_matrix2(m: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [[a00, a01], [a10, a11]] = m;
    Matrix2Trait::new(fx(a00), fx(a01), fx(a10), fx(a11))
}

/// `lu2_solve_matrix`: `Lu2::solve_mut` on a 2x3 right-hand side against upstream (f64).
#[test]
fn test_oracle_lu2_solve_matrix() {
    let cases = oracle::lu2_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Lu2Trait::new(mat_matrix2(ra));
        let mut x = mat_matrix2x3(rb);
        assert!(f.solve_mut(ref x));
        let (err, mag) = err_matrix2x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `cholesky2_solve_matrix`: `Cholesky2::solve_mut` on a 2x3 right-hand side against upstream
/// (f64).
#[test]
fn test_oracle_cholesky2_solve_matrix() {
    let cases = oracle::cholesky2_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let c = Cholesky2Trait::new(mat_matrix2(ra)).unwrap();
        let mut x = mat_matrix2x3(rb);
        c.solve_mut(ref x);
        let (err, mag) = err_matrix2x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `cholesky2_ln_determinant`: `Cholesky2::ln_determinant` against upstream (f64).
#[test]
fn test_oracle_cholesky2_ln_determinant() {
    let cases = oracle::cholesky2_ln_determinant_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rl, tol) = *cases.at(i);
        let c = Cholesky2Trait::new(mat_matrix2(ra)).unwrap();
        let (err, mag) = (ulp_diff(c.ln_determinant(), fx(rl)), abs_raw(fx(rl)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `qr2_q_tr_mul`: `Qr2::q_tr_mul` on a 2x3 matrix against upstream (f64).
#[test]
fn test_oracle_qr2_q_tr_mul() {
    let cases = oracle::qr2_q_tr_mul_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Qr2Trait::new(mat_matrix2(ra));
        let mut x = mat_matrix2x3(rb);
        f.q_tr_mul(ref x);
        let (err, mag) = err_matrix2x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `qr2_solve_matrix`: `Qr2::solve_mut` on a 2x3 right-hand side against upstream (f64).
#[test]
fn test_oracle_qr2_solve_matrix() {
    let cases = oracle::qr2_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Qr2Trait::new(mat_matrix2(ra));
        let mut x = mat_matrix2x3(rb);
        assert!(f.solve_mut(ref x));
        let (err, mag) = err_matrix2x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn mat_matrix3(m: [[i64; 3]; 3]) -> Matrix3<Fixed> {
    let [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]] = m;
    Matrix3Trait::new(
        fx(a00), fx(a01), fx(a02), fx(a10), fx(a11), fx(a12), fx(a20), fx(a21), fx(a22),
    )
}

fn err_matrix3(x: Matrix3<Fixed>, e: [[i64; 3]; 3]) -> (u128, u128) {
    let [[e00, e01, e02], [e10, e11, e12], [e20, e21, e22]] = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.m11, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.m12, fx(e01)));
    mag = max(mag, abs_raw(fx(e01)));
    err = max(err, ulp_diff(x.m13, fx(e02)));
    mag = max(mag, abs_raw(fx(e02)));
    err = max(err, ulp_diff(x.m21, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.m22, fx(e11)));
    mag = max(mag, abs_raw(fx(e11)));
    err = max(err, ulp_diff(x.m23, fx(e12)));
    mag = max(mag, abs_raw(fx(e12)));
    err = max(err, ulp_diff(x.m31, fx(e20)));
    mag = max(mag, abs_raw(fx(e20)));
    err = max(err, ulp_diff(x.m32, fx(e21)));
    mag = max(mag, abs_raw(fx(e21)));
    err = max(err, ulp_diff(x.m33, fx(e22)));
    mag = max(mag, abs_raw(fx(e22)));
    (err, mag)
}

/// `lu3_solve_matrix`: `Lu3::solve_mut` on a 3x3 right-hand side against upstream (f64).
#[test]
fn test_oracle_lu3_solve_matrix() {
    let cases = oracle::lu3_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Lu3Trait::new(mat_matrix3(ra));
        let mut x = mat_matrix3(rb);
        assert!(f.solve_mut(ref x));
        let (err, mag) = err_matrix3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `cholesky3_solve_matrix`: `Cholesky3::solve_mut` on a 3x3 right-hand side against upstream
/// (f64).
#[test]
fn test_oracle_cholesky3_solve_matrix() {
    let cases = oracle::cholesky3_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let c = Cholesky3Trait::new(mat_matrix3(ra)).unwrap();
        let mut x = mat_matrix3(rb);
        c.solve_mut(ref x);
        let (err, mag) = err_matrix3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `cholesky3_ln_determinant`: `Cholesky3::ln_determinant` against upstream (f64).
#[test]
fn test_oracle_cholesky3_ln_determinant() {
    let cases = oracle::cholesky3_ln_determinant_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rl, tol) = *cases.at(i);
        let c = Cholesky3Trait::new(mat_matrix3(ra)).unwrap();
        let (err, mag) = (ulp_diff(c.ln_determinant(), fx(rl)), abs_raw(fx(rl)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `qr3_q_tr_mul`: `Qr3::q_tr_mul` on a 3x3 matrix against upstream (f64).
#[test]
fn test_oracle_qr3_q_tr_mul() {
    let cases = oracle::qr3_q_tr_mul_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Qr3Trait::new(mat_matrix3(ra));
        let mut x = mat_matrix3(rb);
        f.q_tr_mul(ref x);
        let (err, mag) = err_matrix3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `qr3_solve_matrix`: `Qr3::solve_mut` on a 3x3 right-hand side against upstream (f64).
#[test]
fn test_oracle_qr3_solve_matrix() {
    let cases = oracle::qr3_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Qr3Trait::new(mat_matrix3(ra));
        let mut x = mat_matrix3(rb);
        assert!(f.solve_mut(ref x));
        let (err, mag) = err_matrix3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn mat_matrix4(m: [[i64; 4]; 4]) -> Matrix4<Fixed> {
    let [[a00, a01, a02, a03], [a10, a11, a12, a13], [a20, a21, a22, a23], [a30, a31, a32, a33]] =
        m;
    Matrix4Trait::new(
        fx(a00),
        fx(a01),
        fx(a02),
        fx(a03),
        fx(a10),
        fx(a11),
        fx(a12),
        fx(a13),
        fx(a20),
        fx(a21),
        fx(a22),
        fx(a23),
        fx(a30),
        fx(a31),
        fx(a32),
        fx(a33),
    )
}

fn mat_matrix4x3(m: [[i64; 3]; 4]) -> Matrix4x3<Fixed> {
    let [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22], [a30, a31, a32]] = m;
    Matrix4x3Trait::new(
        fx(a00),
        fx(a01),
        fx(a02),
        fx(a10),
        fx(a11),
        fx(a12),
        fx(a20),
        fx(a21),
        fx(a22),
        fx(a30),
        fx(a31),
        fx(a32),
    )
}

fn err_matrix4x3(x: Matrix4x3<Fixed>, e: [[i64; 3]; 4]) -> (u128, u128) {
    let [[e00, e01, e02], [e10, e11, e12], [e20, e21, e22], [e30, e31, e32]] = e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.m11, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.m12, fx(e01)));
    mag = max(mag, abs_raw(fx(e01)));
    err = max(err, ulp_diff(x.m13, fx(e02)));
    mag = max(mag, abs_raw(fx(e02)));
    err = max(err, ulp_diff(x.m21, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.m22, fx(e11)));
    mag = max(mag, abs_raw(fx(e11)));
    err = max(err, ulp_diff(x.m23, fx(e12)));
    mag = max(mag, abs_raw(fx(e12)));
    err = max(err, ulp_diff(x.m31, fx(e20)));
    mag = max(mag, abs_raw(fx(e20)));
    err = max(err, ulp_diff(x.m32, fx(e21)));
    mag = max(mag, abs_raw(fx(e21)));
    err = max(err, ulp_diff(x.m33, fx(e22)));
    mag = max(mag, abs_raw(fx(e22)));
    err = max(err, ulp_diff(x.m41, fx(e30)));
    mag = max(mag, abs_raw(fx(e30)));
    err = max(err, ulp_diff(x.m42, fx(e31)));
    mag = max(mag, abs_raw(fx(e31)));
    err = max(err, ulp_diff(x.m43, fx(e32)));
    mag = max(mag, abs_raw(fx(e32)));
    (err, mag)
}

/// `lu4_solve_matrix`: `Lu4::solve_mut` on a 4x3 right-hand side against upstream (f64).
#[test]
fn test_oracle_lu4_solve_matrix() {
    let cases = oracle::lu4_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Lu4Trait::new(mat_matrix4(ra));
        let mut x = mat_matrix4x3(rb);
        assert!(f.solve_mut(ref x));
        let (err, mag) = err_matrix4x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `cholesky4_solve_matrix`: `Cholesky4::solve_mut` on a 4x3 right-hand side against upstream
/// (f64).
#[test]
fn test_oracle_cholesky4_solve_matrix() {
    let cases = oracle::cholesky4_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let c = Cholesky4Trait::new(mat_matrix4(ra)).unwrap();
        let mut x = mat_matrix4x3(rb);
        c.solve_mut(ref x);
        let (err, mag) = err_matrix4x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `cholesky4_ln_determinant`: `Cholesky4::ln_determinant` against upstream (f64).
#[test]
fn test_oracle_cholesky4_ln_determinant() {
    let cases = oracle::cholesky4_ln_determinant_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rl, tol) = *cases.at(i);
        let c = Cholesky4Trait::new(mat_matrix4(ra)).unwrap();
        let (err, mag) = (ulp_diff(c.ln_determinant(), fx(rl)), abs_raw(fx(rl)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `qr4_q_tr_mul`: `Qr4::q_tr_mul` on a 4x3 matrix against upstream (f64).
#[test]
fn test_oracle_qr4_q_tr_mul() {
    let cases = oracle::qr4_q_tr_mul_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Qr4Trait::new(mat_matrix4(ra));
        let mut x = mat_matrix4x3(rb);
        f.q_tr_mul(ref x);
        let (err, mag) = err_matrix4x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `qr4_solve_matrix`: `Qr4::solve_mut` on a 4x3 right-hand side against upstream (f64).
#[test]
fn test_oracle_qr4_solve_matrix() {
    let cases = oracle::qr4_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Qr4Trait::new(mat_matrix4(ra));
        let mut x = mat_matrix4x3(rb);
        assert!(f.solve_mut(ref x));
        let (err, mag) = err_matrix4x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn mat_matrix6(m: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [
        [a00, a01, a02, a03, a04, a05],
        [a10, a11, a12, a13, a14, a15],
        [a20, a21, a22, a23, a24, a25],
        [a30, a31, a32, a33, a34, a35],
        [a40, a41, a42, a43, a44, a45],
        [a50, a51, a52, a53, a54, a55],
    ] =
        m;
    Matrix6Trait::new(
        fx(a00),
        fx(a01),
        fx(a02),
        fx(a03),
        fx(a04),
        fx(a05),
        fx(a10),
        fx(a11),
        fx(a12),
        fx(a13),
        fx(a14),
        fx(a15),
        fx(a20),
        fx(a21),
        fx(a22),
        fx(a23),
        fx(a24),
        fx(a25),
        fx(a30),
        fx(a31),
        fx(a32),
        fx(a33),
        fx(a34),
        fx(a35),
        fx(a40),
        fx(a41),
        fx(a42),
        fx(a43),
        fx(a44),
        fx(a45),
        fx(a50),
        fx(a51),
        fx(a52),
        fx(a53),
        fx(a54),
        fx(a55),
    )
}

fn mat_matrix6x3(m: [[i64; 3]; 6]) -> Matrix6x3<Fixed> {
    let [
        [a00, a01, a02],
        [a10, a11, a12],
        [a20, a21, a22],
        [a30, a31, a32],
        [a40, a41, a42],
        [a50, a51, a52],
    ] =
        m;
    Matrix6x3Trait::new(
        fx(a00),
        fx(a01),
        fx(a02),
        fx(a10),
        fx(a11),
        fx(a12),
        fx(a20),
        fx(a21),
        fx(a22),
        fx(a30),
        fx(a31),
        fx(a32),
        fx(a40),
        fx(a41),
        fx(a42),
        fx(a50),
        fx(a51),
        fx(a52),
    )
}

fn err_matrix6x3(x: Matrix6x3<Fixed>, e: [[i64; 3]; 6]) -> (u128, u128) {
    let [
        [e00, e01, e02],
        [e10, e11, e12],
        [e20, e21, e22],
        [e30, e31, e32],
        [e40, e41, e42],
        [e50, e51, e52],
    ] =
        e;
    let mut err: u128 = 0;
    let mut mag: u128 = 0;
    err = max(err, ulp_diff(x.m11, fx(e00)));
    mag = max(mag, abs_raw(fx(e00)));
    err = max(err, ulp_diff(x.m12, fx(e01)));
    mag = max(mag, abs_raw(fx(e01)));
    err = max(err, ulp_diff(x.m13, fx(e02)));
    mag = max(mag, abs_raw(fx(e02)));
    err = max(err, ulp_diff(x.m21, fx(e10)));
    mag = max(mag, abs_raw(fx(e10)));
    err = max(err, ulp_diff(x.m22, fx(e11)));
    mag = max(mag, abs_raw(fx(e11)));
    err = max(err, ulp_diff(x.m23, fx(e12)));
    mag = max(mag, abs_raw(fx(e12)));
    err = max(err, ulp_diff(x.m31, fx(e20)));
    mag = max(mag, abs_raw(fx(e20)));
    err = max(err, ulp_diff(x.m32, fx(e21)));
    mag = max(mag, abs_raw(fx(e21)));
    err = max(err, ulp_diff(x.m33, fx(e22)));
    mag = max(mag, abs_raw(fx(e22)));
    err = max(err, ulp_diff(x.m41, fx(e30)));
    mag = max(mag, abs_raw(fx(e30)));
    err = max(err, ulp_diff(x.m42, fx(e31)));
    mag = max(mag, abs_raw(fx(e31)));
    err = max(err, ulp_diff(x.m43, fx(e32)));
    mag = max(mag, abs_raw(fx(e32)));
    err = max(err, ulp_diff(x.m51, fx(e40)));
    mag = max(mag, abs_raw(fx(e40)));
    err = max(err, ulp_diff(x.m52, fx(e41)));
    mag = max(mag, abs_raw(fx(e41)));
    err = max(err, ulp_diff(x.m53, fx(e42)));
    mag = max(mag, abs_raw(fx(e42)));
    err = max(err, ulp_diff(x.m61, fx(e50)));
    mag = max(mag, abs_raw(fx(e50)));
    err = max(err, ulp_diff(x.m62, fx(e51)));
    mag = max(mag, abs_raw(fx(e51)));
    err = max(err, ulp_diff(x.m63, fx(e52)));
    mag = max(mag, abs_raw(fx(e52)));
    (err, mag)
}

/// `lu6_solve_matrix`: `Lu6::solve_mut` on a 6x3 right-hand side against upstream (f64).
#[test]
fn test_oracle_lu6_solve_matrix() {
    let cases = oracle::lu6_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let f = Lu6Trait::new(mat_matrix6(ra));
        let mut x = mat_matrix6x3(rb);
        assert!(f.solve_mut(ref x));
        let (err, mag) = err_matrix6x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `cholesky6_solve_matrix`: `Cholesky6::solve_mut` on a 6x3 right-hand side against upstream
/// (f64).
#[test]
fn test_oracle_cholesky6_solve_matrix() {
    let cases = oracle::cholesky6_solve_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let c = Cholesky6Trait::new(mat_matrix6(ra)).unwrap();
        let mut x = mat_matrix6x3(rb);
        c.solve_mut(ref x);
        let (err, mag) = err_matrix6x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `cholesky6_ln_determinant`: `Cholesky6::ln_determinant` against upstream (f64).
#[test]
fn test_oracle_cholesky6_ln_determinant() {
    let cases = oracle::cholesky6_ln_determinant_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rl, tol) = *cases.at(i);
        let c = Cholesky6Trait::new(mat_matrix6(ra)).unwrap();
        let (err, mag) = (ulp_diff(c.ln_determinant(), fx(rl)), abs_raw(fx(rl)));
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}
