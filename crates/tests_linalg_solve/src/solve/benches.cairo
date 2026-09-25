//! Gas benchmarks of `MatrixSolve` (`bench_<group>__<variant>`, net = raw - the group's
//! `baseline`): the vector solves of the 2, 3, 4 and 6 squares (lower, upper, transposed,
//! unchecked, `with_diag`) and the square right-hand side, with the per-column alternative
//! (`alt_columns`) that lost to the shared prepared divisor of the generated kernels.

use fixed::Fixed;
use nalgebra::{
    Matrix2, Matrix2Trait, Matrix3, Matrix3Trait, Matrix4, Matrix4Trait, Matrix6, Matrix6Trait,
    MatrixSolve, Vector2, Vector2Trait, Vector3, Vector3Trait, Vector4, Vector4Trait, Vector6,
    Vector6Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;

fn l2() -> Matrix2<Fixed> {
    Matrix2Trait::new(int(2), int(0), int(0), int(4))
}

fn b2() -> Vector2<Fixed> {
    Vector2Trait::new(int(-6), int(8))
}

fn bm2() -> Matrix2<Fixed> {
    Matrix2Trait::new(int(-6), int(0), int(8), int(-16))
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_vector2__baseline() {
    let l = black_box(l2());
    let b = black_box(b2());
    assert!(l.m11 != b.x);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_vector2__lower() {
    let l = black_box(l2());
    let b = black_box(b2());
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_vector2__lower_unchecked() {
    let l = black_box(l2());
    let b = black_box(b2());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_vector2__upper() {
    let l = black_box(l2());
    let b = black_box(b2());
    let x = l.transpose().solve_upper_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_vector2__tr_lower() {
    let l = black_box(l2());
    let b = black_box(b2());
    let x = l.tr_solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_vector2__with_diag() {
    let l = black_box(l2());
    let mut b = black_box(b2());
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(l.m11 != b.x);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_matrix2__baseline() {
    let l = black_box(l2());
    let b = black_box(bm2());
    assert!(l.m11 != b.m12);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_matrix2__lower() {
    let l = black_box(l2());
    let b = black_box(bm2());
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.m12);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_matrix2__lower_unchecked() {
    let l = black_box(l2());
    let b = black_box(bm2());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(l.m11 != x.m12);
}

fn l3() -> Matrix3<Fixed> {
    Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1))
}

fn b3() -> Vector3<Fixed> {
    Vector3Trait::new(int(-6), int(8), int(-9))
}

fn bm3() -> Matrix3<Fixed> {
    Matrix3Trait::new(int(-6), int(0), int(6), int(8), int(-16), int(-4), int(-9), int(-3), int(12))
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_vector3__baseline() {
    let l = black_box(l3());
    let b = black_box(b3());
    assert!(l.m11 != b.x);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_vector3__lower() {
    let l = black_box(l3());
    let b = black_box(b3());
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_vector3__lower_unchecked() {
    let l = black_box(l3());
    let b = black_box(b3());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_vector3__upper() {
    let l = black_box(l3());
    let b = black_box(b3());
    let x = l.transpose().solve_upper_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_vector3__tr_lower() {
    let l = black_box(l3());
    let b = black_box(b3());
    let x = l.tr_solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_vector3__with_diag() {
    let l = black_box(l3());
    let mut b = black_box(b3());
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(l.m11 != b.x);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_matrix3__baseline() {
    let l = black_box(l3());
    let b = black_box(bm3());
    assert!(l.m11 != b.m12);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_matrix3__lower() {
    let l = black_box(l3());
    let b = black_box(bm3());
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.m12);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_matrix3__lower_unchecked() {
    let l = black_box(l3());
    let b = black_box(bm3());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(l.m11 != x.m12);
}

/// The multi-column solve as 3 single-column solves (the `columns` form of `gemm`, a sixth of
/// the generated code): bit-identical, dearer, because each row's 3 quotients no longer share
/// one prepared divisor (`Real::div3`). Kept as evidence (AGENTS.md rule 8); the bit-identity is
/// checked by `test_solve_alt_columns_is_bit_identical`.
#[test]
#[inline(never)]
fn bench_matrix3_solve_matrix3__alt_columns() {
    let l = black_box(l3());
    let b = black_box(bm3());
    let c0 = l.solve_lower_triangular_unchecked(Vector3Trait::new(b.m11, b.m21, b.m31));
    let c1 = l.solve_lower_triangular_unchecked(Vector3Trait::new(b.m12, b.m22, b.m32));
    let c2 = l.solve_lower_triangular_unchecked(Vector3Trait::new(b.m13, b.m23, b.m33));
    let x = Matrix3 {
        m11: c0.x,
        m21: c0.y,
        m31: c0.z,
        m12: c1.x,
        m22: c1.y,
        m32: c1.z,
        m13: c2.x,
        m23: c2.y,
        m33: c2.z,
    };
    assert!(l.m11 != x.m12);
}

fn l4() -> Matrix4<Fixed> {
    Matrix4Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
    )
}

fn b4() -> Vector4<Fixed> {
    Vector4Trait::new(int(-6), int(8), int(-9), int(17))
}

fn bm4() -> Matrix4<Fixed> {
    Matrix4Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(17),
        int(-10),
        int(8),
        int(17),
    )
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_vector4__baseline() {
    let l = black_box(l4());
    let b = black_box(b4());
    assert!(l.m11 != b.x);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_vector4__lower() {
    let l = black_box(l4());
    let b = black_box(b4());
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_vector4__lower_unchecked() {
    let l = black_box(l4());
    let b = black_box(b4());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_vector4__upper() {
    let l = black_box(l4());
    let b = black_box(b4());
    let x = l.transpose().solve_upper_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_vector4__tr_lower() {
    let l = black_box(l4());
    let b = black_box(b4());
    let x = l.tr_solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_vector4__with_diag() {
    let l = black_box(l4());
    let mut b = black_box(b4());
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(l.m11 != b.x);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_matrix4__baseline() {
    let l = black_box(l4());
    let b = black_box(bm4());
    assert!(l.m11 != b.m12);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_matrix4__lower() {
    let l = black_box(l4());
    let b = black_box(bm4());
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.m12);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_matrix4__lower_unchecked() {
    let l = black_box(l4());
    let b = black_box(bm4());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(l.m11 != x.m12);
}

/// The multi-column solve as 4 single-column solves (the `columns` form of `gemm`, a sixth of
/// the generated code): bit-identical, dearer, because each row's 4 quotients no longer share
/// one prepared divisor (`Real::div4`). Kept as evidence (AGENTS.md rule 8); the bit-identity is
/// checked by `test_solve_alt_columns_is_bit_identical`.
#[test]
#[inline(never)]
fn bench_matrix4_solve_matrix4__alt_columns() {
    let l = black_box(l4());
    let b = black_box(bm4());
    let c0 = l.solve_lower_triangular_unchecked(Vector4Trait::new(b.m11, b.m21, b.m31, b.m41));
    let c1 = l.solve_lower_triangular_unchecked(Vector4Trait::new(b.m12, b.m22, b.m32, b.m42));
    let c2 = l.solve_lower_triangular_unchecked(Vector4Trait::new(b.m13, b.m23, b.m33, b.m43));
    let c3 = l.solve_lower_triangular_unchecked(Vector4Trait::new(b.m14, b.m24, b.m34, b.m44));
    let x = Matrix4 {
        m11: c0.x,
        m21: c0.y,
        m31: c0.z,
        m41: c0.w,
        m12: c1.x,
        m22: c1.y,
        m32: c1.z,
        m42: c1.w,
        m13: c2.x,
        m23: c2.y,
        m33: c2.z,
        m43: c2.w,
        m14: c3.x,
        m24: c3.y,
        m34: c3.z,
        m44: c3.w,
    };
    assert!(l.m11 != x.m12);
}

fn l6() -> Matrix6<Fixed> {
    Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(1),
        int(1),
        int(0),
        int(0),
        int(0),
        int(-1),
        int(-3),
        int(2),
        int(8),
        int(0),
        int(0),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(2),
        int(0),
        int(-2),
        int(3),
        int(1),
        int(-1),
        int(-3),
        int(4),
    )
}

fn b6() -> Vector6<Fixed> {
    Vector6Trait::new(int(-6), int(8), int(-9), int(17), int(5), int(26))
}

fn bm6() -> Matrix6<Fixed> {
    Matrix6Trait::new(
        int(-6),
        int(0),
        int(6),
        int(-6),
        int(0),
        int(6),
        int(8),
        int(-16),
        int(-4),
        int(8),
        int(-16),
        int(-4),
        int(-9),
        int(-3),
        int(12),
        int(-9),
        int(-3),
        int(12),
        int(17),
        int(-10),
        int(8),
        int(17),
        int(-10),
        int(8),
        int(5),
        int(-7),
        int(-10),
        int(5),
        int(-7),
        int(-10),
        int(26),
        int(-22),
        int(11),
        int(26),
        int(-22),
        int(11),
    )
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_vector6__baseline() {
    let l = black_box(l6());
    let b = black_box(b6());
    assert!(l.m11 != b.x);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_vector6__lower() {
    let l = black_box(l6());
    let b = black_box(b6());
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_vector6__lower_unchecked() {
    let l = black_box(l6());
    let b = black_box(b6());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_vector6__upper() {
    let l = black_box(l6());
    let b = black_box(b6());
    let x = l.transpose().solve_upper_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_vector6__tr_lower() {
    let l = black_box(l6());
    let b = black_box(b6());
    let x = l.tr_solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_vector6__with_diag() {
    let l = black_box(l6());
    let mut b = black_box(b6());
    assert!(l.solve_lower_triangular_with_diag_mut(ref b, int(2)));
    assert!(l.m11 != b.x);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_matrix6__baseline() {
    let l = black_box(l6());
    let b = black_box(bm6());
    assert!(l.m11 != b.m12);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_matrix6__lower() {
    let l = black_box(l6());
    let b = black_box(bm6());
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(l.m11 != x.m12);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_matrix6__lower_unchecked() {
    let l = black_box(l6());
    let b = black_box(bm6());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(l.m11 != x.m12);
}

/// The multi-column solve as 6 single-column solves (the `columns` form of `gemm`, a sixth of
/// the generated code): bit-identical, dearer, because each row's 6 quotients no longer share
/// one prepared divisor (`Real::div6`). Kept as evidence (AGENTS.md rule 8); the bit-identity is
/// checked by `test_solve_alt_columns_is_bit_identical`.
#[test]
#[inline(never)]
fn bench_matrix6_solve_matrix6__alt_columns() {
    let l = black_box(l6());
    let b = black_box(bm6());
    let c0 = l
        .solve_lower_triangular_unchecked(
            Vector6Trait::new(b.m11, b.m21, b.m31, b.m41, b.m51, b.m61),
        );
    let c1 = l
        .solve_lower_triangular_unchecked(
            Vector6Trait::new(b.m12, b.m22, b.m32, b.m42, b.m52, b.m62),
        );
    let c2 = l
        .solve_lower_triangular_unchecked(
            Vector6Trait::new(b.m13, b.m23, b.m33, b.m43, b.m53, b.m63),
        );
    let c3 = l
        .solve_lower_triangular_unchecked(
            Vector6Trait::new(b.m14, b.m24, b.m34, b.m44, b.m54, b.m64),
        );
    let c4 = l
        .solve_lower_triangular_unchecked(
            Vector6Trait::new(b.m15, b.m25, b.m35, b.m45, b.m55, b.m65),
        );
    let c5 = l
        .solve_lower_triangular_unchecked(
            Vector6Trait::new(b.m16, b.m26, b.m36, b.m46, b.m56, b.m66),
        );
    let x = Matrix6 {
        m11: c0.x,
        m21: c0.y,
        m31: c0.z,
        m41: c0.w,
        m51: c0.a,
        m61: c0.b,
        m12: c1.x,
        m22: c1.y,
        m32: c1.z,
        m42: c1.w,
        m52: c1.a,
        m62: c1.b,
        m13: c2.x,
        m23: c2.y,
        m33: c2.z,
        m43: c2.w,
        m53: c2.a,
        m63: c2.b,
        m14: c3.x,
        m24: c3.y,
        m34: c3.z,
        m44: c3.w,
        m54: c3.a,
        m64: c3.b,
        m15: c4.x,
        m25: c4.y,
        m35: c4.z,
        m45: c4.w,
        m55: c4.a,
        m65: c4.b,
        m16: c5.x,
        m26: c5.y,
        m36: c5.z,
        m46: c5.w,
        m56: c5.a,
        m66: c5.b,
    };
    assert!(l.m11 != x.m12);
}
