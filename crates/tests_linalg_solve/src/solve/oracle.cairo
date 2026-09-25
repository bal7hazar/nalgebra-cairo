//! Oracle tests (suite `solve` of `tools/oracle`, upstream nalgebra 0.35 in f64 on the same raw
//! inputs): every case within the oracle's tolerance plus one relative ulp of the expected
//! value (`oracle_tol`, the convention of the `linalg` oracle tests); a failure reports by how
//! much the worst case exceeds it.

use core::cmp::max;
use fixed::Fixed;
use nalgebra::{
    Matrix2, Matrix2Trait, Matrix3, Matrix3Trait, Matrix4, Matrix4Trait, Matrix5, Matrix5Trait,
    Matrix6, Matrix6Trait, MatrixSolve, Vector2, Vector2Trait, Vector3, Vector3Trait, Vector4,
    Vector4Trait, Vector5, Vector5Trait, Vector6, Vector6Trait,
};
use nalgebra_tests_utils::{abs_raw, excess, fx, oracle_tol, ulp_diff};
use crate::oracle_solve as oracle;

fn mat_matrix2(m: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [[a00, a01], [a10, a11]] = m;
    Matrix2Trait::new(fx(a00), fx(a01), fx(a10), fx(a11))
}

fn vec2(t: (i64, i64)) -> Vector2<Fixed> {
    let (a0, a1) = t;
    Vector2Trait::new(fx(a0), fx(a1))
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

/// `solve2_lower`: `a.solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve2_lower() {
    let cases = oracle::solve2_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix2(ra);
        let b = vec2(rb);
        let x = a.solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector2_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve2_upper`: `a.solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve2_upper() {
    let cases = oracle::solve2_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix2(ra);
        let b = vec2(rb);
        let x = a.solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector2_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve2_tr_lower`: `a.tr_solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve2_tr_lower() {
    let cases = oracle::solve2_tr_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix2(ra);
        let b = vec2(rb);
        let x = a.tr_solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector2_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve2_tr_upper`: `a.tr_solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve2_tr_upper() {
    let cases = oracle::solve2_tr_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix2(ra);
        let b = vec2(rb);
        let x = a.tr_solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector2_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve2_lower_with_diag`: `solve_lower_triangular_with_diag_mut` against upstream (f64).
#[test]
fn test_oracle_solve2_lower_with_diag() {
    let cases = oracle::solve2_lower_with_diag_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rd, rx, tol) = *cases.at(i);
        let a = mat_matrix2(ra);
        let mut x = vec2(rb);
        assert!(a.solve_lower_triangular_with_diag_mut(ref x, fx(rd)));
        let (err, mag) = err_vector2_t(x, rx);
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

/// `solve3_lower`: `a.solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve3_lower() {
    let cases = oracle::solve3_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix3(ra);
        let b = vec3(rb);
        let x = a.solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector3_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve3_upper`: `a.solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve3_upper() {
    let cases = oracle::solve3_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix3(ra);
        let b = vec3(rb);
        let x = a.solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector3_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve3_tr_lower`: `a.tr_solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve3_tr_lower() {
    let cases = oracle::solve3_tr_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix3(ra);
        let b = vec3(rb);
        let x = a.tr_solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector3_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve3_tr_upper`: `a.tr_solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve3_tr_upper() {
    let cases = oracle::solve3_tr_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix3(ra);
        let b = vec3(rb);
        let x = a.tr_solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector3_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve3_lower_with_diag`: `solve_lower_triangular_with_diag_mut` against upstream (f64).
#[test]
fn test_oracle_solve3_lower_with_diag() {
    let cases = oracle::solve3_lower_with_diag_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rd, rx, tol) = *cases.at(i);
        let a = mat_matrix3(ra);
        let mut x = vec3(rb);
        assert!(a.solve_lower_triangular_with_diag_mut(ref x, fx(rd)));
        let (err, mag) = err_vector3_t(x, rx);
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

/// `solve4_lower`: `a.solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve4_lower() {
    let cases = oracle::solve4_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix4(ra);
        let b = vec4(rb);
        let x = a.solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector4_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve4_upper`: `a.solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve4_upper() {
    let cases = oracle::solve4_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix4(ra);
        let b = vec4(rb);
        let x = a.solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector4_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve4_tr_lower`: `a.tr_solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve4_tr_lower() {
    let cases = oracle::solve4_tr_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix4(ra);
        let b = vec4(rb);
        let x = a.tr_solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector4_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve4_tr_upper`: `a.tr_solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve4_tr_upper() {
    let cases = oracle::solve4_tr_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix4(ra);
        let b = vec4(rb);
        let x = a.tr_solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector4_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve4_lower_with_diag`: `solve_lower_triangular_with_diag_mut` against upstream (f64).
#[test]
fn test_oracle_solve4_lower_with_diag() {
    let cases = oracle::solve4_lower_with_diag_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rd, rx, tol) = *cases.at(i);
        let a = mat_matrix4(ra);
        let mut x = vec4(rb);
        assert!(a.solve_lower_triangular_with_diag_mut(ref x, fx(rd)));
        let (err, mag) = err_vector4_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

fn mat_matrix5(m: [[i64; 5]; 5]) -> Matrix5<Fixed> {
    let [
        [a00, a01, a02, a03, a04],
        [a10, a11, a12, a13, a14],
        [a20, a21, a22, a23, a24],
        [a30, a31, a32, a33, a34],
        [a40, a41, a42, a43, a44],
    ] =
        m;
    Matrix5Trait::new(
        fx(a00),
        fx(a01),
        fx(a02),
        fx(a03),
        fx(a04),
        fx(a10),
        fx(a11),
        fx(a12),
        fx(a13),
        fx(a14),
        fx(a20),
        fx(a21),
        fx(a22),
        fx(a23),
        fx(a24),
        fx(a30),
        fx(a31),
        fx(a32),
        fx(a33),
        fx(a34),
        fx(a40),
        fx(a41),
        fx(a42),
        fx(a43),
        fx(a44),
    )
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

/// `solve5_lower`: `a.solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve5_lower() {
    let cases = oracle::solve5_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix5(ra);
        let b = vec5(rb);
        let x = a.solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector5_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve5_upper`: `a.solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve5_upper() {
    let cases = oracle::solve5_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix5(ra);
        let b = vec5(rb);
        let x = a.solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector5_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve5_tr_lower`: `a.tr_solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve5_tr_lower() {
    let cases = oracle::solve5_tr_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix5(ra);
        let b = vec5(rb);
        let x = a.tr_solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector5_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve5_tr_upper`: `a.tr_solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve5_tr_upper() {
    let cases = oracle::solve5_tr_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix5(ra);
        let b = vec5(rb);
        let x = a.tr_solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector5_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve5_lower_with_diag`: `solve_lower_triangular_with_diag_mut` against upstream (f64).
#[test]
fn test_oracle_solve5_lower_with_diag() {
    let cases = oracle::solve5_lower_with_diag_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rd, rx, tol) = *cases.at(i);
        let a = mat_matrix5(ra);
        let mut x = vec5(rb);
        assert!(a.solve_lower_triangular_with_diag_mut(ref x, fx(rd)));
        let (err, mag) = err_vector5_t(x, rx);
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

/// `solve6_lower`: `a.solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve6_lower() {
    let cases = oracle::solve6_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix6(ra);
        let b = vec6(rb);
        let x = a.solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector6_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve6_upper`: `a.solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve6_upper() {
    let cases = oracle::solve6_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix6(ra);
        let b = vec6(rb);
        let x = a.solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector6_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve6_tr_lower`: `a.tr_solve_lower_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve6_tr_lower() {
    let cases = oracle::solve6_tr_lower_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix6(ra);
        let b = vec6(rb);
        let x = a.tr_solve_lower_triangular(b).unwrap();
        let (err, mag) = err_vector6_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve6_tr_upper`: `a.tr_solve_upper_triangular(b).unwrap()` against upstream (f64).
#[test]
fn test_oracle_solve6_tr_upper() {
    let cases = oracle::solve6_tr_upper_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rx, tol) = *cases.at(i);
        let a = mat_matrix6(ra);
        let b = vec6(rb);
        let x = a.tr_solve_upper_triangular(b).unwrap();
        let (err, mag) = err_vector6_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}

/// `solve6_lower_with_diag`: `solve_lower_triangular_with_diag_mut` against upstream (f64).
#[test]
fn test_oracle_solve6_lower_with_diag() {
    let cases = oracle::solve6_lower_with_diag_cases();
    let mut i = 0;
    let mut worst: u128 = 0;
    while i < cases.len() {
        let (ra, rb, rd, rx, tol) = *cases.at(i);
        let a = mat_matrix6(ra);
        let mut x = vec6(rb);
        assert!(a.solve_lower_triangular_with_diag_mut(ref x, fx(rd)));
        let (err, mag) = err_vector6_t(x, rx);
        worst = max(worst, excess(err, oracle_tol(mag, tol)));
        i += 1;
    }
    assert!(worst == 0, "oracle tolerance exceeded by {}", worst);
}
