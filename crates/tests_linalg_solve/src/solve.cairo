use fixed::Fixed;
use nalgebra::{Matrix3, Matrix3Trait, MatrixSolve, Vector3, Vector3Trait};
use nalgebra_tests_utils::int;

#[test]
fn test_probe() {
    let l = Matrix3Trait::new(
        int(2), int(0), int(0), int(1), int(4), int(0), int(3), int(-1), int(8),
    );
    let b = Vector3Trait::new(int(2), int(9), int(-7));
    let x: Vector3<Fixed> = l.solve_lower_triangular(b).unwrap();
    assert!(x == Vector3Trait::new(int(1), int(2), int(-1)));
    let mut bb = b;
    assert!(l.solve_lower_triangular_mut(ref bb));
    let _m: Matrix3<Fixed> = l;
}
use nalgebra::{Matrix3x2, Matrix6, Matrix6Trait, Vector6, Vector6Trait};
use nalgebra_testing::black_box;

fn l6() -> Matrix6<Fixed> {
    Matrix6Trait::new(
        int(2),
        int(0),
        int(0),
        int(0),
        int(0),
        int(0),
        int(1),
        int(4),
        int(0),
        int(0),
        int(0),
        int(0),
        int(3),
        int(-1),
        int(8),
        int(0),
        int(0),
        int(0),
        int(1),
        int(2),
        int(3),
        int(3),
        int(0),
        int(0),
        int(-1),
        int(2),
        int(1),
        int(3),
        int(5),
        int(0),
        int(1),
        int(1),
        int(1),
        int(1),
        int(1),
        int(7),
    )
}

#[test]
#[inline(never)]
fn bench_probe6__baseline() {
    let l = black_box(l6());
    let b = black_box(l6());
    assert!(l.m11 == b.m11);
}

#[test]
#[inline(never)]
fn bench_probe6__direct() {
    let l = black_box(l6());
    let b = black_box(l6());
    let x = l.solve_lower_triangular_unchecked(b);
    assert!(x.m11 == int(1));
}

fn col(m: Matrix6<Fixed>, j: u32) -> Vector6<Fixed> {
    m.column(j)
}

#[test]
#[inline(never)]
fn bench_probe6__columns() {
    let l = black_box(l6());
    let b = black_box(l6());
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
    assert!(
        c0.x == int(1)
            && c1.x == int(0)
            && c2.y == int(0)
            && c3.x == int(0)
            && c4.x == int(0)
            && c5.b == int(1),
    );
}

#[test]
#[inline(never)]
fn bench_probe6__tr_direct() {
    let l = black_box(l6());
    let b = black_box(l6());
    let x = l.tr_solve_lower_triangular_unchecked(b);
    assert!(x.m66 == int(1));
}

#[test]
#[inline(never)]
fn bench_probe6v__baseline() {
    let l = black_box(l6());
    let b = black_box(Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)));
    assert!(l.m11 == int(2) && b.x == int(1));
}

#[test]
#[inline(never)]
fn bench_probe6v__lower() {
    let l = black_box(l6());
    let b = black_box(Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)));
    let x = l.solve_lower_triangular(b).unwrap();
    assert!(x.x != int(7));
}

#[test]
#[inline(never)]
fn bench_probe6v__tr_lower() {
    let l = black_box(l6());
    let b = black_box(Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6)));
    let x = l.tr_solve_lower_triangular(b).unwrap();
    assert!(x.x != int(7));
}
