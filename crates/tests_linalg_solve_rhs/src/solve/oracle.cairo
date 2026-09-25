//! Oracle tests (suite `solve` of `tools/oracle`, upstream nalgebra 0.35 in f64 on the same raw
//! inputs): every case within the oracle's tolerance plus one relative ulp of the expected
//! value (`oracle_tol`, the convention of the `linalg` oracle tests); a failure reports by how
//! much the worst case exceeds it.

use core::cmp::max;
use fixed::Fixed;
use nalgebra::{
    Matrix3, Matrix3Trait, Matrix6, Matrix6Trait, Matrix6x3, Matrix6x3Trait, MatrixSolve,
};
use nalgebra_tests_utils::{abs_raw, excess, fx, oracle_tol, ulp_diff};
use crate::oracle_solve_rhs as oracle;

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

/// `solve3_lower_matrix`: `a.solve_lower_triangular(b).unwrap()` on a 3x3 right-hand side against
/// upstream (f64).
#[test]
fn test_oracle_solve3_lower_matrix() {
    let cases = oracle::solve3_lower_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix3(ra);
        let b = mat_matrix3(rb);
        let x = a.solve_lower_triangular(b).unwrap();
        let (err, mag) = err_matrix3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve3_upper_matrix`: `a.solve_upper_triangular(b).unwrap()` on a 3x3 right-hand side against
/// upstream (f64).
#[test]
fn test_oracle_solve3_upper_matrix() {
    let cases = oracle::solve3_upper_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix3(ra);
        let b = mat_matrix3(rb);
        let x = a.solve_upper_triangular(b).unwrap();
        let (err, mag) = err_matrix3(x, rx);
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

/// `solve6_lower_matrix`: `a.solve_lower_triangular(b).unwrap()` on a 6x3 right-hand side against
/// upstream (f64).
#[test]
fn test_oracle_solve6_lower_matrix() {
    let cases = oracle::solve6_lower_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix6(ra);
        let b = mat_matrix6x3(rb);
        let x = a.solve_lower_triangular(b).unwrap();
        let (err, mag) = err_matrix6x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve6_upper_matrix`: `a.solve_upper_triangular(b).unwrap()` on a 6x3 right-hand side against
/// upstream (f64).
#[test]
fn test_oracle_solve6_upper_matrix() {
    let cases = oracle::solve6_upper_matrix_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix6(ra);
        let b = mat_matrix6x3(rb);
        let x = a.solve_upper_triangular(b).unwrap();
        let (err, mag) = err_matrix6x3(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}
