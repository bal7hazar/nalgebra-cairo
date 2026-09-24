//! Gas benchmarks of the WP 8.4-P09a completion of `Rotation2` (`bench_rotation2_<op>__<variant>`,
//! net = raw - `baseline` of the group). Inputs: the rotations of 0.75 and -2 rad and a skewed
//! matrix; expected values are the kernels' own results, checked against upstream in
//! `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::matrix1::Matrix1;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::geometry::rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
use nalgebra::geometry::unit_complex::UnitComplex;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{ONE_RAW, fx, m2};
use simba::scalar::Real;

fn a() -> Rotation2<Fixed> {
    Rotation2AngleTrait::new(fx(0xc0000000))
}

fn b() -> Rotation2<Fixed> {
    Rotation2AngleTrait::new(fx(-0x200000000))
}

fn skewed() -> Matrix2<Fixed> {
    m2([[ONE_RAW * 3, ONE_RAW], [-ONE_RAW, ONE_RAW * 2]])
}

fn c() -> UnitComplex<Fixed> {
    b().into()
}

// --- from_matrix_eps

#[test]
#[inline(never)]
fn bench_rotation2_from_matrix_eps__baseline() {
    let _m = black_box(skewed());
    let e = black_box(Rotation2Trait::from_matrix(skewed()));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_matrix_eps__closed_form() {
    let m = black_box(skewed());
    let e = black_box(Rotation2Trait::from_matrix(skewed()));
    let id = Rotation2Trait::identity();
    assert!(Rotation2AngleTrait::from_matrix_eps(m, Real::default_epsilon(), 0, id) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_matrix_eps__iterate_8() {
    let m = black_box(skewed());
    let id = Rotation2Trait::identity();
    let e = black_box(
        Rotation2AngleTrait::from_matrix_eps(skewed(), Real::default_epsilon(), 8, id),
    );
    assert!(Rotation2AngleTrait::from_matrix_eps(m, Real::default_epsilon(), 8, id) == e);
}

// --- slerp

#[test]
#[inline(never)]
fn bench_rotation2_slerp__baseline() {
    let (_x, _y) = (black_box(a()), black_box(b()));
    let e = black_box(a().slerp(b(), Real::HALF));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_slerp__through_unit_complex() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().slerp(b(), Real::HALF));
    assert!(x.slerp(y, Real::HALF) == e);
}

// --- rotation_to, division

#[test]
#[inline(never)]
fn bench_rotation2_rotation_to__baseline() {
    let (_x, _y) = (black_box(a()), black_box(b()));
    let e = black_box(a().rotation_to(b()));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_rotation_to__four_kernels() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().rotation_to(b()));
    assert!(x.rotation_to(y) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_rotation_to__div() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().rotation_to(b()));
    assert!(y / x == e);
}

// --- products with unit complex numbers

#[test]
#[inline(never)]
fn bench_rotation2_mul_unit_complex__baseline() {
    let (_r, _u) = (black_box(a()), black_box(c()));
    let e = black_box(a().mul_unit_complex(c()));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_mul_unit_complex__first_column() {
    let (r, u) = (black_box(a()), black_box(c()));
    let e = black_box(a().mul_unit_complex(c()));
    assert!(r.mul_unit_complex(u) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_mul_unit_complex__div() {
    let (r, u) = (black_box(a()), black_box(c()));
    let e = black_box(a().div_unit_complex(c()));
    assert!(r.div_unit_complex(u) == e);
}

// --- scaled axis

#[test]
#[inline(never)]
fn bench_rotation2_from_scaled_axis__baseline() {
    let _v = black_box(Matrix1 { x: fx(0xc0000000) });
    let e = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_scaled_axis__new() {
    let v = black_box(Matrix1 { x: fx(0xc0000000) });
    let e = black_box(a());
    assert!(Rotation2AngleTrait::from_scaled_axis(v) == e);
}
