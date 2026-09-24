//! Gas benchmarks of the point types of WP 8.4-P09a (`bench_point<N>_<op>__<variant>`, net = raw -
//! `baseline` of the group): the operations of `Point4` / `Point6` whose cost depends on the size
//! (`lerp`, `unscale`, `from_homogeneous`, the comparisons, `p[i]`, `from_slice`) and the
//! completion of `Point3`.

use fixed::Fixed;
use nalgebra::base::point3::{Point3, Point3Trait};
use nalgebra::base::vector5::Vector5;
use nalgebra::geometry::point::{Point3ExtTrait, Point3Index};
use nalgebra::geometry::point4::{Point4, Point4Trait};
use nalgebra::geometry::point6::{Point6, Point6Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int};
use simba::scalar::Real;

fn p() -> Point4<Fixed> {
    Point4Trait::new(fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(0x10000000))
}

fn q() -> Point4<Fixed> {
    Point4Trait::new(fx(-0x80000000), fx(0x100000000), fx(0x20000000), fx(0x300000000))
}

fn p6() -> Point6<Fixed> {
    Point6Trait::new(int(1), int(-2), int(3), int(-4), int(5), int(-6))
}

fn p3() -> Point3<Fixed> {
    Point3Trait::new(fx(0x180000000), fx(-0x240000000), fx(0x3c0000000))
}

// --- Point4

#[test]
#[inline(never)]
fn bench_point4_lerp__baseline() {
    let (_a, _b) = (black_box(p()), black_box(q()));
    let e = black_box(p().lerp(q(), Real::HALF));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point4_lerp__fused() {
    let (a, b) = (black_box(p()), black_box(q()));
    let e = black_box(p().lerp(q(), Real::HALF));
    assert!(a.lerp(b, Real::HALF) == e);
}

#[test]
#[inline(never)]
fn bench_point4_unscale__baseline() {
    let _a = black_box(p());
    let e = black_box(p().unscale(int(3)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point4_unscale__div4() {
    let a = black_box(p());
    let e = black_box(p().unscale(int(3)));
    assert!(a.unscale(int(3)) == e);
}

#[test]
#[inline(never)]
fn bench_point4_from_homogeneous__baseline() {
    let _h: Vector5<Fixed> = black_box(p().to_homogeneous());
    let e = black_box(p());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point4_from_homogeneous__div4() {
    let h: Vector5<Fixed> = black_box(p().to_homogeneous());
    let e = black_box(p());
    assert!(Point4Trait::from_homogeneous(h) == Some(e));
}

#[test]
#[inline(never)]
fn bench_point4_relative_eq__baseline() {
    let (_a, _b) = (black_box(p()), black_box(p()));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_point4_relative_eq__component_wise() {
    let (a, b) = (black_box(p()), black_box(p()));
    assert!(a.relative_eq(b, 1, Real::zero()));
}

#[test]
#[inline(never)]
fn bench_point4_relative_eq__ulps_eq() {
    let (a, b) = (black_box(p()), black_box(p()));
    assert!(a.ulps_eq(b, 1, 4));
}

#[test]
#[inline(never)]
fn bench_point4_partial_ord__baseline() {
    let (_a, _b) = (black_box(p()), black_box(p().sup(q())));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_point4_partial_ord__le() {
    let (a, b) = (black_box(p()), black_box(p().sup(q())));
    assert!(a <= b);
}

#[test]
#[inline(never)]
fn bench_point4_index__baseline() {
    let _a = black_box(p());
    let e = black_box(p().w);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point4_index__match() {
    let mut a = black_box(p());
    let e = black_box(p().w);
    assert!(a[3] == e);
}

// --- Point6

#[test]
#[inline(never)]
fn bench_point6_lerp__baseline() {
    let (_a, _b) = (black_box(p6()), black_box(-p6()));
    let e = black_box(p6().lerp(-p6(), Real::HALF));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point6_lerp__fused() {
    let (a, b) = (black_box(p6()), black_box(-p6()));
    let e = black_box(p6().lerp(-p6(), Real::HALF));
    assert!(a.lerp(b, Real::HALF) == e);
}

// --- Point3 completion

#[test]
#[inline(never)]
fn bench_point3_from_slice__baseline() {
    let _s = black_box(array![fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
    let e = black_box(p3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_from_slice__checked() {
    let s = black_box(array![fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
    let e = black_box(p3());
    assert!(s.len() == 3 && Point3ExtTrait::from_slice(s.span()) == e);
}

#[test]
#[inline(never)]
fn bench_point3_index__baseline() {
    let _a = black_box(p3());
    let e = black_box(p3().z);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_index__match() {
    let mut a = black_box(p3());
    let e = black_box(p3().z);
    assert!(a[2] == e);
}

#[test]
#[inline(never)]
fn bench_point3_relative_eq__baseline() {
    let (_a, _b) = (black_box(p3()), black_box(p3()));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_point3_relative_eq__component_wise() {
    let (a, b) = (black_box(p3()), black_box(p3()));
    assert!(a.relative_eq(b, 1, Real::zero()));
}
