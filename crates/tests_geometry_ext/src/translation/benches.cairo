//! Gas benchmarks of the translation types of WP 8.4-P09a (`bench_translation<N>_<op>__<variant>`,
//! net = raw - `baseline` of the group): the heterogeneous products and the division of
//! `Translation2` / `Translation3`, and the size-dependent operations of `Translation4` /
//! `Translation6`.

use fixed::Fixed;
use nalgebra::geometry::point4::{Point4, Point4Trait};
use nalgebra::geometry::translation2::Translation2Trait;
use nalgebra::geometry::translation3::Translation3Trait;
use nalgebra::geometry::translation4::{Translation4, Translation4Trait};
use nalgebra::geometry::translation6::{Translation6, Translation6Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{int, iso3, t2, t3, uc, uq};
use simba::scalar::Real;

fn t4() -> Translation4<Fixed> {
    Translation4Trait::new(int(1), int(-2), int(3), int(4))
}

fn t6() -> Translation6<Fixed> {
    Translation6Trait::new(int(1), int(-2), int(3), int(4), int(-5), int(6))
}

fn p4() -> Point4<Fixed> {
    Point4Trait::new(int(10), int(20), int(30), int(40))
}

// --- Translation3

#[test]
#[inline(never)]
fn bench_translation3_mul_isometry__baseline() {
    let t = black_box(t3(0x100000000, -0x200000000, 0x300000000));
    let i = black_box(iso3((0, 0, 0x100000000), (0x80000000, 0x80000000, 0x80000000, 0x80000000)));
    assert!(t == t && i == i);
}

#[test]
#[inline(never)]
fn bench_translation3_mul_isometry__add_translation() {
    let t = black_box(t3(0x100000000, -0x200000000, 0x300000000));
    let i = black_box(iso3((0, 0, 0x100000000), (0x80000000, 0x80000000, 0x80000000, 0x80000000)));
    assert!(t.mul_isometry(i).rotation == i.rotation);
}

#[test]
#[inline(never)]
fn bench_translation3_mul_isometry__mul_unit_quaternion() {
    let t = black_box(t3(0x100000000, -0x200000000, 0x300000000));
    let q = black_box(uq(0x80000000, 0x80000000, 0x80000000, 0x80000000));
    assert!(t.mul_unit_quaternion(q).translation == t);
}

#[test]
#[inline(never)]
fn bench_translation3_div__baseline() {
    let (a, b) = (black_box(t3(0x100000000, 0, 0)), black_box(t3(0, 0x100000000, 0)));
    let e = black_box(t3(0x100000000, -0x100000000, 0));
    assert!(a == a && b == b && e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_div__subtract() {
    let (a, b) = (black_box(t3(0x100000000, 0, 0)), black_box(t3(0, 0x100000000, 0)));
    let e = black_box(t3(0x100000000, -0x100000000, 0));
    assert!(a / b == e);
}

#[test]
#[inline(never)]
fn bench_translation3_div__relative_eq() {
    let (a, b) = (black_box(t3(0x100000000, 0, 0)), black_box(t3(0, 0x100000000, 0)));
    let e = black_box(t3(0x100000000, -0x100000000, 0));
    assert!((a / b).relative_eq(e, 1, Real::zero()));
}

// --- Translation2

#[test]
#[inline(never)]
fn bench_translation2_mul_unit_complex__baseline() {
    let (t, c) = (black_box(t2(0x100000000, 0)), black_box(uc(0, 0x100000000)));
    assert!(t == t && c == c);
}

#[test]
#[inline(never)]
fn bench_translation2_mul_unit_complex__wrap() {
    let (t, c) = (black_box(t2(0x100000000, 0)), black_box(uc(0, 0x100000000)));
    assert!(t.mul_unit_complex(c).rotation == c);
}

// --- Translation4, Translation6

#[test]
#[inline(never)]
fn bench_translation4_transform_point__baseline() {
    let (t, p) = (black_box(t4()), black_box(p4()));
    let e = black_box(Point4Trait::new(int(11), int(18), int(33), int(44)));
    assert!(t == t && p == p && e == e);
}

#[test]
#[inline(never)]
fn bench_translation4_transform_point__add() {
    let (t, p) = (black_box(t4()), black_box(p4()));
    let e = black_box(Point4Trait::new(int(11), int(18), int(33), int(44)));
    assert!(t.transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_translation4_to_homogeneous__baseline() {
    let t = black_box(t4());
    assert!(t == t);
}

#[test]
#[inline(never)]
fn bench_translation4_to_homogeneous__matrix5() {
    let t = black_box(t4());
    assert!(t.to_homogeneous().m45 == int(4));
}

#[test]
#[inline(never)]
fn bench_translation6_mul__baseline() {
    let (a, b) = (black_box(t6()), black_box(t6()));
    assert!(a == b);
}

#[test]
#[inline(never)]
fn bench_translation6_mul__add() {
    let (a, b) = (black_box(t6()), black_box(t6()));
    assert!((a * b).vector.b == int(12));
}
