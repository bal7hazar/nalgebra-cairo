//! Gas benchmarks of `MatrixSolve` (`bench_<group>__<variant>`, net = raw - the group's
//! `baseline`): the vector solves of the 2, 3, 4 and 6 squares (lower, upper, transposed,
//! unchecked, `with_diag`).

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
    let b = black_box(b2());
    let mut x = b;
    assert!(l.solve_lower_triangular_with_diag_mut(ref x, int(2)));
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix2_solve_vector2__baseline() {
    let l = black_box(l2());
    let x = black_box(b2());
    assert!(l.m11 != x.x);
}

fn l3() -> Matrix3<Fixed> {
    Matrix3Trait::new(int(2), int(0), int(0), int(0), int(4), int(0), int(3), int(1), int(1))
}

fn b3() -> Vector3<Fixed> {
    Vector3Trait::new(int(-6), int(8), int(-9))
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
    let b = black_box(b3());
    let mut x = b;
    assert!(l.solve_lower_triangular_with_diag_mut(ref x, int(2)));
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix3_solve_vector3__baseline() {
    let l = black_box(l3());
    let x = black_box(b3());
    assert!(l.m11 != x.x);
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
    let b = black_box(b4());
    let mut x = b;
    assert!(l.solve_lower_triangular_with_diag_mut(ref x, int(2)));
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix4_solve_vector4__baseline() {
    let l = black_box(l4());
    let x = black_box(b4());
    assert!(l.m11 != x.x);
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
    let b = black_box(b6());
    let mut x = b;
    assert!(l.solve_lower_triangular_with_diag_mut(ref x, int(2)));
    assert!(l.m11 != x.x);
}

#[test]
#[inline(never)]
fn bench_matrix6_solve_vector6__baseline() {
    let l = black_box(l6());
    let x = black_box(b6());
    assert!(l.m11 != x.x);
}
