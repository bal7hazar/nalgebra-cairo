//! Gas benchmarks of `Point2` (`bench_point2_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::point3::Point3;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use super::{Point2, Point2Trait};

fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

fn v2(x: i64, y: i64) -> Vector2<Fixed> {
    Vector2 { x: fx(x), y: fx(y) }
}

fn v3(x: i64, y: i64, z: i64) -> Vector3<Fixed> {
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

fn p2(x: i64, y: i64) -> Point2<Fixed> {
    Point2 { x: fx(x), y: fx(y) }
}

fn p3(x: i64, y: i64, z: i64) -> Point3<Fixed> {
    Point3 { x: fx(x), y: fx(y), z: fx(z) }
}

// --- alternative implementations (losers)

/// `from_homogeneous` through one reciprocal of `w`: cheaper, but `1 / w` is rounded before being
/// amplified by the coordinates.
#[inline(always)]
fn alt_from_homogeneous_recip(v: Vector3<Fixed>) -> Option<Point2<Fixed>> {
    if v.z == Real::ZERO {
        None
    } else {
        let r = v.z.recip();
        Some(Point2 { x: v.x * r, y: v.y * r })
    }
}

/// `center` as `lerp(rhs, 1/2)`: an exact difference, then one fused product-sum per coordinate.
#[inline(always)]
fn alt_center_lerp(p: Point2<Fixed>, q: Point2<Fixed>) -> Point2<Fixed> {
    p.lerp(q, Real::HALF)
}

/// `center` as upstream's `(p + q) * 0.5`: one addition and one product per coordinate. The sum
/// overflows for coordinates above `2^30`.
#[inline(always)]
fn alt_center_add_scale(p: Point2<Fixed>, q: Point2<Fixed>) -> Point2<Fixed> {
    let h: Fixed = Real::HALF;
    Point2 { x: (p.x + q.x) * h, y: (p.y + q.y) * h }
}

/// `center` as `(p + q) / 2`: one addition and one division per coordinate. The sum overflows for
/// coordinates above `2^30`.
#[inline(always)]
fn alt_center_add_div(p: Point2<Fixed>, q: Point2<Fixed>) -> Point2<Fixed> {
    let two: Fixed = Real::TWO;
    Point2 { x: (p.x + q.x) / two, y: (p.y + q.y) / two }
}

/// `distance_squared` with one rounding and one overflow check per product (what AGENTS.md rule 4
/// forbids).
#[inline(always)]
fn alt_distance_squared_unfused(p: Point2<Fixed>, q: Point2<Fixed>) -> Fixed {
    let (dx, dy) = (p.x - q.x, p.y - q.y);
    dx * dx + dy * dy
}

/// `distance` as the square root of the squared distance: overflows above a distance of 46 340
/// and loses the low bits of short distances.
#[inline(always)]
fn alt_distance_sqrt(p: Point2<Fixed>, q: Point2<Fixed>) -> Fixed {
    p.distance_squared(q).sqrt()
}

// --- why the alternatives lost

#[test]
fn test_from_homogeneous_alt_recip_is_less_accurate() {
    // (1.5, ..) / 3 = 0.5: exact through the division, 1 ulp short through `recip(3)` (whose own
    // rounding is amplified by the coordinates: up to `|x|` ulp).
    let v = v3(0x180000000, -0x240000000, 0x300000000);
    assert!(Point2Trait::<Fixed>::from_homogeneous(v) == Some(p2(0x80000000, -0xc0000000)));
    assert!(alt_from_homogeneous_recip(v) == Some(p2(0x7fffffff, -0xc0000000)));
}

#[test]
fn test_center_alts_agree_with_center() {
    let (p, q) = (p2(0x180000000, -0x240000000), p2(-0x480000000, 0x40000000));
    assert!(p.center(q) == alt_center_lerp(p, q));
    assert!(p.center(q) == alt_center_add_scale(p, q));
    assert!(p.center(q) == alt_center_add_div(p, q));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_center_alt_add_scale_overflows() {
    // The exact midpoint fits, the sum does not.
    let _ = alt_center_add_scale(
        p2(0x4000000000000000, 0x4000000000000000), p2(0x4000000000000000, 0x4000000000000000),
    );
}

#[test]
fn test_center_does_not_overflow_where_the_alts_do() {
    assert!(
        p2(0x4000000000000000, 0x4000000000000000)
            .center(
                p2(0x4000000000000000, 0x4000000000000000),
            ) == p2(0x4000000000000000, 0x4000000000000000),
    );
}

#[test]
fn test_distance_alt_sqrt_is_less_accurate_on_short_distances() {
    // A distance of 3 ulp: its square (9 * 2^-64) floors to 0, so the alternative returns 0.
    let (p, q) = (p2(0x3, 0x0), p2(0x0, 0x0));
    assert!(p.distance(q) == fx(3));
    assert!(alt_distance_sqrt(p, q) == fx(0));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_distance_alt_sqrt_overflows_on_long_distances() {
    // Distance 100 000: `distance` is fine, its square does not fit.
    let _ = alt_distance_sqrt(p2(0x186a000000000, 0x0), p2(0x0, 0x0));
}

#[test]
fn test_distance_squared_alt_unfused_is_less_accurate() {
    // Three 0.5-ulp squares: the fused kernel adds them exactly before flooring, the unfused
    // version floors each to 0.
    let (p, q) = (p2(0xb504f33, 0xb504f33), p2(0x0, 0x0));
    assert!(p.distance_squared(q) != alt_distance_squared_unfused(p, q));
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_point2_new__baseline() {
    let _x: Fixed = black_box(fx(0x180000000));
    let _y: Fixed = black_box(fx(-0x240000000));
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_new__new() {
    let x: Fixed = black_box(fx(0x180000000));
    let y: Fixed = black_box(fx(-0x240000000));
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    assert!(Point2Trait::<Fixed>::new(x, y) == e);
}

#[test]
#[inline(never)]
fn bench_point2_origin__baseline() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Point2<Fixed> = black_box(p2(0x0, 0x0));
    let r = black_box(e);
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point2_origin__origin() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Point2<Fixed> = black_box(p2(0x0, 0x0));
    let r = black_box(Point2Trait::<Fixed>::origin());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point2_from__baseline() {
    let _v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _t: (Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000)));
    let _arr: [Fixed; 2] = black_box([fx(0x180000000), fx(-0x240000000)]);
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_from__from_coordinates() {
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _t: (Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000)));
    let _arr: [Fixed; 2] = black_box([fx(0x180000000), fx(-0x240000000)]);
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    assert!(Point2Trait::<Fixed>::from_coordinates(v) == e);
}

#[test]
#[inline(never)]
fn bench_point2_from__vector() {
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _t: (Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000)));
    let _arr: [Fixed; 2] = black_box([fx(0x180000000), fx(-0x240000000)]);
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let res: Point2<Fixed> = v.into();
    assert!(res == e);
}

#[test]
#[inline(never)]
fn bench_point2_from__tuple() {
    let _v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let t: (Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000)));
    let _arr: [Fixed; 2] = black_box([fx(0x180000000), fx(-0x240000000)]);
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let res: Point2<Fixed> = t.into();
    assert!(res == e);
}

#[test]
#[inline(never)]
fn bench_point2_from__array() {
    let _v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _t: (Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000)));
    let arr: [Fixed; 2] = black_box([fx(0x180000000), fx(-0x240000000)]);
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let res: Point2<Fixed> = arr.into();
    assert!(res == e);
}

#[test]
#[inline(never)]
fn bench_point2_coords__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_coords__coords() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    assert!(p.coords() == e);
}

#[test]
#[inline(never)]
fn bench_point2_coords__into_vector() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let res: Vector2<Fixed> = p.into();
    assert!(res == e);
}

#[test]
#[inline(never)]
fn bench_point2_into__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Fixed = black_box(fx(0x180000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_into__tuple() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Fixed = black_box(fx(0x180000000));
    let (x, _): (Fixed, Fixed) = p.into();
    assert!(x == e);
}

#[test]
#[inline(never)]
fn bench_point2_into__array() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Fixed = black_box(fx(0x180000000));
    let r: [Fixed; 2] = p.into();
    let [x, _] = r;
    assert!(x == e);
}

#[test]
#[inline(never)]
fn bench_point2_push__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _z: Fixed = black_box(fx(0x3c0000000));
    let e: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_push__push() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let z: Fixed = black_box(fx(0x3c0000000));
    let e: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(p.push(z) == e);
}

#[test]
#[inline(never)]
fn bench_point2_to_homogeneous__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x100000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_to_homogeneous__to_homogeneous() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x100000000));
    assert!(p.to_homogeneous() == e);
}

#[test]
#[inline(never)]
fn bench_point2_from_homogeneous__baseline() {
    let _v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x300000000));
    let e: Option<Point2<Fixed>> = black_box(Some(p2(0x80000000, -0xc0000000)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_from_homogeneous__from_homogeneous() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x300000000));
    let e: Option<Point2<Fixed>> = black_box(Some(p2(0x80000000, -0xc0000000)));
    assert!(Point2Trait::<Fixed>::from_homogeneous(v) == e);
}

#[test]
#[inline(never)]
fn bench_point2_from_homogeneous__alt_recip() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x300000000));
    let e: Option<Point2<Fixed>> = black_box(Some(p2(0x7fffffff, -0xc0000000)));
    assert!(alt_from_homogeneous_recip(v) == e);
}

#[test]
#[inline(never)]
fn bench_point2_sub_point__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Vector2<Fixed> = black_box(v2(0x600000000, -0x280000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_sub_point__sub_point() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Vector2<Fixed> = black_box(v2(0x600000000, -0x280000000));
    assert!(p.sub_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_add_vector__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _v: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x300000000, -0x200000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_add_vector__add_vector() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let v: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x300000000, -0x200000000));
    assert!(p.add_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_point2_add_vector__add_assign() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let v: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x300000000, -0x200000000));
    let mut r = p;
    r += v;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point2_sub_vector__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _v: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(0x600000000, -0x280000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_sub_vector__sub_vector() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let v: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(0x600000000, -0x280000000));
    assert!(p.sub_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_point2_sub_vector__sub_assign() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let v: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(0x600000000, -0x280000000));
    let mut r = p;
    r -= v;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point2_scale__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Point2<Fixed> = black_box(p2(0x3c0000000, -0x5a0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_scale__scale() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Point2<Fixed> = black_box(p2(0x3c0000000, -0x5a0000000));
    assert!(p.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_point2_scale__mul_assign() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Point2<Fixed> = black_box(p2(0x3c0000000, -0x5a0000000));
    let mut r = p;
    r *= k;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point2_unscale__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Point2<Fixed> = black_box(p2(0x99999999, -0xe6666667));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_unscale__unscale() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Point2<Fixed> = black_box(p2(0x99999999, -0xe6666667));
    assert!(p.unscale(k) == e);
}

#[test]
#[inline(never)]
fn bench_point2_unscale__div_assign() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Point2<Fixed> = black_box(p2(0x99999999, -0xe6666667));
    let mut r = p;
    r /= k;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point2_neg__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, 0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_neg__neg() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, 0x240000000));
    assert!(-p == e);
}

#[test]
#[inline(never)]
fn bench_point2_inf_sup__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x480000000, -0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_inf_sup__inf() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x480000000, -0x240000000));
    assert!(p.inf(q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_inf_sup__sup() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(0x180000000, 0x40000000));
    assert!(p.sup(q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_inf_sup__inf_sup() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: (Point2<Fixed>, Point2<Fixed>) = black_box(
        (p2(-0x480000000, -0x240000000), p2(0x180000000, 0x40000000)),
    );
    assert!(p.inf_sup(q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_abs_diff_eq__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _q: Point2<Fixed> = black_box(p2(0x180000003, -0x23ffffffd));
    let _ulps: u64 = black_box(3);
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_abs_diff_eq__within() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(0x180000003, -0x23ffffffd));
    let ulps: u64 = black_box(3);
    let e: bool = black_box(true);
    assert!(p.abs_diff_eq(q, ulps) == e);
}

#[test]
#[inline(never)]
fn bench_point2_abs_diff_eq__outside() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(0x180000003, -0x23ffffffd));
    let ulps: u64 = black_box(3);
    let e: bool = black_box(false);
    assert!(p.abs_diff_eq(q, ulps - 1) == e);
}

#[test]
#[inline(never)]
fn bench_point2_lerp__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let _t: Fixed = black_box(fx(0x80000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, -0x100000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_lerp__lerp() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let t: Fixed = black_box(fx(0x80000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, -0x100000000));
    assert!(p.lerp(q, t) == e);
}

#[test]
#[inline(never)]
fn bench_point2_center__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, -0x100000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_center__sum_prod2() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, -0x100000000));
    assert!(p.center(q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_center__alt_lerp() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, -0x100000000));
    assert!(alt_center_lerp(p, q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_center__alt_add_scale() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, -0x100000000));
    assert!(alt_center_add_scale(p, q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_center__alt_add_div() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Point2<Fixed> = black_box(p2(-0x180000000, -0x100000000));
    assert!(alt_center_add_div(p, q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_distance_squared__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Fixed = black_box(fx(0x2a40000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_distance_squared__fused() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Fixed = black_box(fx(0x2a40000000));
    assert!(p.distance_squared(q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_distance_squared__alt_unfused() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Fixed = black_box(fx(0x2a40000000));
    assert!(alt_distance_squared_unfused(p, q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_distance__baseline() {
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let _q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Fixed = black_box(fx(0x680000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point2_distance__norm() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Fixed = black_box(fx(0x680000000));
    assert!(p.distance(q) == e);
}

#[test]
#[inline(never)]
fn bench_point2_distance__alt_sqrt() {
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let q: Point2<Fixed> = black_box(p2(-0x480000000, 0x40000000));
    let e: Fixed = black_box(fx(0x680000000));
    assert!(alt_distance_sqrt(p, q) == e);
}
