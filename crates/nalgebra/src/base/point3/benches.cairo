//! Gas benchmarks of `Point3` (`bench_point3_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/point3/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, p3};
use super::{Point3, Point3InternalTrait, Point3Trait};

/// `center` as `lerp(rhs, 1/2)`: an exact difference, then one fused product-sum per coordinate.
#[inline(always)]
fn alt_center_lerp(p: Point3<Fixed>, q: Point3<Fixed>) -> Point3<Fixed> {
    p.lerp(q, Real::HALF)
}

/// `center` as upstream's `(p + q) * 0.5`: one addition and one product per coordinate. The sum
/// overflows for coordinates above `2^30`.
#[inline(always)]
fn alt_center_add_scale(p: Point3<Fixed>, q: Point3<Fixed>) -> Point3<Fixed> {
    let h: Fixed = Real::HALF;
    Point3 { x: (p.x + q.x) * h, y: (p.y + q.y) * h, z: (p.z + q.z) * h }
}

/// `center` as `(p + q) / 2`: one addition and one division per coordinate. The sum overflows for
/// coordinates above `2^30`.
#[inline(always)]
fn alt_center_add_div(p: Point3<Fixed>, q: Point3<Fixed>) -> Point3<Fixed> {
    let two: Fixed = Real::TWO;
    Point3 { x: (p.x + q.x) / two, y: (p.y + q.y) / two, z: (p.z + q.z) / two }
}

/// `distance_squared` with one rounding and one overflow check per product (what AGENTS.md rule 4
/// forbids).
#[inline(always)]
fn alt_distance_squared_unfused(p: Point3<Fixed>, q: Point3<Fixed>) -> Fixed {
    let (dx, dy, dz) = (p.x - q.x, p.y - q.y, p.z - q.z);
    dx * dx + dy * dy + dz * dz
}

/// `distance` as the square root of the squared distance: overflows above a distance of 46 340
/// and loses the low bits of short distances.
#[inline(always)]
fn alt_distance_sqrt(p: Point3<Fixed>, q: Point3<Fixed>) -> Fixed {
    p.distance_squared(q).sqrt()
}

#[test]
fn test_center_alts_agree_with_center() {
    let (p, q) = (
        p3(0x180000000, -0x240000000, 0x3c0000000), p3(-0x480000000, 0x40000000, 0x200000000),
    );
    assert!(p.center(q) == alt_center_lerp(p, q));
    assert!(p.center(q) == alt_center_add_scale(p, q));
    assert!(p.center(q) == alt_center_add_div(p, q));
}

#[test]
fn test_center_does_not_overflow_where_the_alts_do() {
    assert!(
        p3(0x4000000000000000, 0x4000000000000000, 0x4000000000000000)
            .center(
                p3(0x4000000000000000, 0x4000000000000000, 0x4000000000000000),
            ) == p3(0x4000000000000000, 0x4000000000000000, 0x4000000000000000),
    );
}

#[test]
fn test_distance_alt_sqrt_is_less_accurate_on_short_distances() {
    // A distance of 3 ulp: its square (9 * 2^-64) floors to 0, so the alternative returns 0.
    let (p, q) = (p3(0x3, 0x0, 0x0), p3(0x0, 0x0, 0x0));
    assert!(p.distance(q) == fx(3));
    assert!(alt_distance_sqrt(p, q) == fx(0));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_distance_alt_sqrt_overflows_on_long_distances() {
    // Distance 100 000: `distance` is fine, its square does not fit.
    let _ = alt_distance_sqrt(p3(0x186a000000000, 0x0, 0x0), p3(0x0, 0x0, 0x0));
}

#[test]
fn test_distance_squared_alt_unfused_is_less_accurate() {
    // Three 0.5-ulp squares: the fused kernel adds them exactly before flooring, the unfused
    // version floors each to 0.
    let (p, q) = (p3(0xb504f33, 0xb504f33, 0xb504f33), p3(0x0, 0x0, 0x0));
    assert!(p.distance_squared(q) != alt_distance_squared_unfused(p, q));
}

#[test]
#[inline(never)]
fn bench_point3_center__sum_prod2() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(p.center(q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_distance_squared__fused() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x2d50000000));
    assert!(p.distance_squared(q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_distance__norm() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x6bb40b374));
    assert!(p.distance(q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_distance__alt_sqrt() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x6bb40b374));
    assert!(alt_distance_sqrt(p, q) == e);
}
