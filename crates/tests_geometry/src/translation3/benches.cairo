//! Gas benchmarks of `Translation3` (`bench_translation3_<op>__<variant>`, net = raw - `baseline`
//! of the group).
//!
//! Every operation of this type is an exact addition or negation, so there is no alternative
//! implementation to measure against (AGENTS.md rule 8), except the inverse transform, which can
//! either subtract directly or materialise the inverse translation first. The benchmarks exist to
//! keep the cost of the pose plumbing visible in `GAS.md`, next to the isometries that carry it.
//!
//! The translations used are `(1.25, -0.375, 2.5)` and `(-0.25, 2.5, -1.5)`, the point
//! `(-2.5, 3.75, 0.75)`.
//!
//! Moved from `crates/nalgebra/src/geometry/translation3/benches.cairo` (WP 8.1c, test-only
//! package).

use fixed::Fixed;
use nalgebra::base::point3::Point3;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::translation3::{Translation3, Translation3Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, p3, t3};

/// `(1.25, -0.375, 2.5)`.
fn a() -> Translation3<Fixed> {
    t3(0x140000000, -0x60000000, 0x280000000)
}

/// `(-0.25, 2.5, -1.5)`.
fn b() -> Translation3<Fixed> {
    t3(-0x40000000, 0x280000000, -0x180000000)
}

/// `(-2.5, 3.75, 0.75)`.
fn p() -> Point3<Fixed> {
    p3(-0x280000000, 0x3c0000000, 0xc0000000)
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_translation3_identity__baseline() {
    let _a: Translation3<Fixed> = black_box(a());
    let e: Translation3<Fixed> = black_box(t3(0, 0, 0));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_identity__const() {
    let _a: Translation3<Fixed> = black_box(a());
    let e: Translation3<Fixed> = black_box(t3(0, 0, 0));
    assert!(Translation3Trait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_translation3_new__baseline() {
    let _v: Vector3<Fixed> = black_box(a().vector);
    let e: Translation3<Fixed> = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_new__from_components() {
    let v: Vector3<Fixed> = black_box(a().vector);
    let e: Translation3<Fixed> = black_box(a());
    assert!(Translation3Trait::new(v.x, v.y, v.z) == e);
}

#[test]
#[inline(never)]
fn bench_translation3_new__from_vector() {
    let v: Vector3<Fixed> = black_box(a().vector);
    let e: Translation3<Fixed> = black_box(a());
    assert!(Translation3Trait::from_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_translation3_inverse__baseline() {
    let _t: Translation3<Fixed> = black_box(a());
    let e: Translation3<Fixed> = black_box(t3(-0x140000000, 0x60000000, -0x280000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_inverse__negate() {
    let t: Translation3<Fixed> = black_box(a());
    let e: Translation3<Fixed> = black_box(t3(-0x140000000, 0x60000000, -0x280000000));
    assert!(t.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_translation3_mul__baseline() {
    let _x: Translation3<Fixed> = black_box(a());
    let _y: Translation3<Fixed> = black_box(b());
    let e: Translation3<Fixed> = black_box(t3(0x100000000, 0x220000000, 0x100000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_mul__compose() {
    let x: Translation3<Fixed> = black_box(a());
    let y: Translation3<Fixed> = black_box(b());
    let e: Translation3<Fixed> = black_box(t3(0x100000000, 0x220000000, 0x100000000));
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_translation3_transform_point__baseline() {
    let _t: Translation3<Fixed> = black_box(a());
    let _q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-0x140000000, 0x360000000, 0x340000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_transform_point__add() {
    let t: Translation3<Fixed> = black_box(a());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-0x140000000, 0x360000000, 0x340000000));
    assert!(t.transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_translation3_inverse_transform_point__baseline() {
    let _t: Translation3<Fixed> = black_box(a());
    let _q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-0x3c0000000, 0x420000000, -0x1c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_inverse_transform_point__subtract() {
    let t: Translation3<Fixed> = black_box(a());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-0x3c0000000, 0x420000000, -0x1c0000000));
    assert!(t.inverse_transform_point(q) == e);
}

/// The loser: the inverse translation is materialised first (one extra negation per component,
/// and it panics on `-MIN` where the direct subtraction does not).
#[test]
#[inline(never)]
fn bench_translation3_inverse_transform_point__alt_inverse_then_transform() {
    let t: Translation3<Fixed> = black_box(a());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-0x3c0000000, 0x420000000, -0x1c0000000));
    assert!(t.inverse().transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_translation3_to_homogeneous__baseline() {
    let _t: Translation3<Fixed> = black_box(a());
    let e: Fixed = black_box(fx(0x140000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_to_homogeneous__matrix4() {
    let t: Translation3<Fixed> = black_box(a());
    let e: Fixed = black_box(fx(0x140000000));
    assert!(t.to_homogeneous().m14 == e);
}

#[test]
#[inline(never)]
fn bench_translation3_abs_diff_eq__baseline() {
    let _x: Translation3<Fixed> = black_box(a());
    let _y: Translation3<Fixed> = black_box(t3(0x140000003, -0x60000002, 0x280000001));
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation3_abs_diff_eq__ulps() {
    let x: Translation3<Fixed> = black_box(a());
    let y: Translation3<Fixed> = black_box(t3(0x140000003, -0x60000002, 0x280000001));
    let e: bool = black_box(true);
    assert!(x.abs_diff_eq(y, 3) == e);
}
