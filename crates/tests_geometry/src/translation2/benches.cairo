//! Gas benchmarks of `Translation2` (`bench_translation2_<op>__<variant>`, net = raw - `baseline`
//! of the group).
//!
//! Every operation of this type is an exact addition or negation, so there is no alternative
//! implementation to measure against (AGENTS.md rule 8): the benchmarks exist to keep the cost of
//! the pose plumbing visible in `GAS.md`, next to the isometries that carry it.
//!
//! The translations used are `(1.25, -0.375)` and `(-0.25, 2.5)`, the point `(-2.5, 3.75)`.
//!
//! Moved from `crates/nalgebra/src/geometry/translation2/benches.cairo` (WP 8.1c, test-only
//! package).

use fixed::Fixed;
use nalgebra::base::point2::Point2;
use nalgebra::base::vector2::Vector2;
use nalgebra::geometry::translation2::{Translation2, Translation2Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, p2, t2};

/// `(1.25, -0.375)`.
fn a() -> Translation2<Fixed> {
    t2(0x140000000, -0x60000000)
}

/// `(-0.25, 2.5)`.
fn b() -> Translation2<Fixed> {
    t2(-0x40000000, 0x280000000)
}

/// `(-2.5, 3.75)`.
fn p() -> Point2<Fixed> {
    p2(-0x280000000, 0x3c0000000)
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_translation2_identity__baseline() {
    let _a: Translation2<Fixed> = black_box(a());
    let e: Translation2<Fixed> = black_box(t2(0, 0));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation2_identity__const() {
    let _a: Translation2<Fixed> = black_box(a());
    let e: Translation2<Fixed> = black_box(t2(0, 0));
    assert!(Translation2Trait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_translation2_new__baseline() {
    let _v: Vector2<Fixed> = black_box(a().vector);
    let e: Translation2<Fixed> = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation2_new__from_components() {
    let v: Vector2<Fixed> = black_box(a().vector);
    let e: Translation2<Fixed> = black_box(a());
    assert!(Translation2Trait::new(v.x, v.y) == e);
}

#[test]
#[inline(never)]
fn bench_translation2_new__from_vector() {
    let v: Vector2<Fixed> = black_box(a().vector);
    let e: Translation2<Fixed> = black_box(a());
    assert!(Translation2Trait::from_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_translation2_inverse__baseline() {
    let _t: Translation2<Fixed> = black_box(a());
    let e: Translation2<Fixed> = black_box(t2(-0x140000000, 0x60000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation2_inverse__negate() {
    let t: Translation2<Fixed> = black_box(a());
    let e: Translation2<Fixed> = black_box(t2(-0x140000000, 0x60000000));
    assert!(t.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_translation2_mul__baseline() {
    let _x: Translation2<Fixed> = black_box(a());
    let _y: Translation2<Fixed> = black_box(b());
    let e: Translation2<Fixed> = black_box(t2(0x100000000, 0x220000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation2_mul__compose() {
    let x: Translation2<Fixed> = black_box(a());
    let y: Translation2<Fixed> = black_box(b());
    let e: Translation2<Fixed> = black_box(t2(0x100000000, 0x220000000));
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_translation2_transform_point__baseline() {
    let _t: Translation2<Fixed> = black_box(a());
    let _q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-0x140000000, 0x360000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation2_transform_point__add() {
    let t: Translation2<Fixed> = black_box(a());
    let q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-0x140000000, 0x360000000));
    assert!(t.transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_translation2_inverse_transform_point__baseline() {
    let _t: Translation2<Fixed> = black_box(a());
    let _q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-0x3c0000000, 0x420000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation2_inverse_transform_point__subtract() {
    let t: Translation2<Fixed> = black_box(a());
    let q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-0x3c0000000, 0x420000000));
    assert!(t.inverse_transform_point(q) == e);
}

/// The loser: the inverse translation is materialised first (one extra negation per component,
/// and it panics on `-MIN` where the direct subtraction does not).
#[test]
#[inline(never)]
fn bench_translation2_inverse_transform_point__alt_inverse_then_transform() {
    let t: Translation2<Fixed> = black_box(a());
    let q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-0x3c0000000, 0x420000000));
    assert!(t.inverse().transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_translation2_to_homogeneous__baseline() {
    let _t: Translation2<Fixed> = black_box(a());
    let e: Fixed = black_box(fx(0x140000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation2_to_homogeneous__matrix3() {
    let t: Translation2<Fixed> = black_box(a());
    let e: Fixed = black_box(fx(0x140000000));
    assert!(t.to_homogeneous().m13 == e);
}

#[test]
#[inline(never)]
fn bench_translation2_abs_diff_eq__baseline() {
    let _x: Translation2<Fixed> = black_box(a());
    let _y: Translation2<Fixed> = black_box(t2(0x140000003, -0x60000002));
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_translation2_abs_diff_eq__ulps() {
    let x: Translation2<Fixed> = black_box(a());
    let y: Translation2<Fixed> = black_box(t2(0x140000003, -0x60000002));
    let e: bool = black_box(true);
    assert!(x.abs_diff_eq(y, 3) == e);
}
