//! Gas benchmarks of `Point3` (`bench_point3_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).
//!
//! Moved from `crates/nalgebra/src/base/point3/benches.cairo` (WP 8.1c, test-only package): the
//! tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::{Point3, Point3Trait};
use nalgebra::base::vector3::Vector3;
use nalgebra::base::vector4::Vector4;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, p2, p3, v3, v4};
use simba::scalar::Real;

// --- alternative implementations (losers)

/// `from_homogeneous` through one reciprocal of `w`: cheaper, but `1 / w` is rounded before being
/// amplified by the coordinates.
#[inline(always)]
fn alt_from_homogeneous_recip(v: Vector4<Fixed>) -> Option<Point3<Fixed>> {
    if v.w == Real::zero() {
        None
    } else {
        let r = v.w.recip();
        Some(Point3 { x: v.x * r, y: v.y * r, z: v.z * r })
    }
}

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

// --- why the alternatives lost

#[test]
fn test_from_homogeneous_alt_recip_is_less_accurate() {
    // (1.5, ..) / 3 = 0.5: exact through the division, 1 ulp short through `recip(3)` (whose own
    // rounding is amplified by the coordinates: up to `|x|` ulp).
    let v = v4(0x180000000, -0x240000000, 0x3c0000000, 0x300000000);
    assert!(
        Point3Trait::<Fixed>::from_homogeneous(v) == Some(p3(0x80000000, -0xc0000000, 0x140000000)),
    );
    assert!(alt_from_homogeneous_recip(v) == Some(p3(0x7fffffff, -0xc0000000, 0x13ffffffe)));
}

#[test]
#[should_panic(expected: 'i64_add Overflow')]
fn test_center_alt_add_scale_overflows() {
    // The exact midpoint fits, the sum does not.
    let _ = alt_center_add_scale(
        p3(0x4000000000000000, 0x4000000000000000, 0x4000000000000000),
        p3(0x4000000000000000, 0x4000000000000000, 0x4000000000000000),
    );
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_point3_new__baseline() {
    let _x: Fixed = black_box(fx(0x180000000));
    let _y: Fixed = black_box(fx(-0x240000000));
    let _z: Fixed = black_box(fx(0x3c0000000));
    let e: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_new__new() {
    let x: Fixed = black_box(fx(0x180000000));
    let y: Fixed = black_box(fx(-0x240000000));
    let z: Fixed = black_box(fx(0x3c0000000));
    let e: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(Point3Trait::<Fixed>::new(x, y, z) == e);
}

#[test]
#[inline(never)]
fn bench_point3_origin__baseline() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Point3<Fixed> = black_box(p3(0x0, 0x0, 0x0));
    let r = black_box(e);
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point3_origin__origin() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Point3<Fixed> = black_box(p3(0x0, 0x0, 0x0));
    let r = black_box(Point3Trait::<Fixed>::origin());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point3_from__baseline() {
    let _v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _t: (Fixed, Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)));
    let _arr: [Fixed; 3] = black_box([fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
    let e: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_from__from_coordinates() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _t: (Fixed, Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)));
    let _arr: [Fixed; 3] = black_box([fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
    let e: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(Point3Trait::<Fixed>::from_coordinates(v) == e);
}

#[test]
#[inline(never)]
fn bench_point3_from__vector() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _t: (Fixed, Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)));
    let _arr: [Fixed; 3] = black_box([fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
    let e: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let res: Point3<Fixed> = v.into();
    assert!(res == e);
}

#[test]
#[inline(never)]
fn bench_point3_from__array() {
    let _v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _t: (Fixed, Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)));
    let arr: [Fixed; 3] = black_box([fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
    let e: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let res: Point3<Fixed> = arr.into();
    assert!(res == e);
}

#[test]
#[inline(never)]
fn bench_point3_coords__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_coords__coords() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(p.coords() == e);
}

#[test]
#[inline(never)]
fn bench_point3_coords__into_vector() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let res: Vector3<Fixed> = p.into();
    assert!(res == e);
}

#[test]
#[inline(never)]
fn bench_point3_into__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x180000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_into__array() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x180000000));
    let r: [Fixed; 3] = p.into();
    let [x, _, _] = r;
    assert!(x == e);
}

#[test]
#[inline(never)]
fn bench_point3_xy__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_xy__xy() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    assert!(p.xy() == e);
}

#[test]
#[inline(never)]
fn bench_point3_to_homogeneous__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x100000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_to_homogeneous__to_homogeneous() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x100000000));
    assert!(p.to_homogeneous() == e);
}

#[test]
#[inline(never)]
fn bench_point3_from_homogeneous__baseline() {
    let _v: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x300000000));
    let e: Option<Point3<Fixed>> = black_box(Some(p3(0x80000000, -0xc0000000, 0x140000000)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_from_homogeneous__from_homogeneous() {
    let v: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x300000000));
    let e: Option<Point3<Fixed>> = black_box(Some(p3(0x80000000, -0xc0000000, 0x140000000)));
    assert!(Point3Trait::<Fixed>::from_homogeneous(v) == e);
}

#[test]
#[inline(never)]
fn bench_point3_from_homogeneous__alt_recip() {
    let v: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x300000000));
    let e: Option<Point3<Fixed>> = black_box(Some(p3(0x7fffffff, -0xc0000000, 0x13ffffffe)));
    assert!(alt_from_homogeneous_recip(v) == e);
}

#[test]
#[inline(never)]
fn bench_point3_sub_point__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(0x600000000, -0x280000000, 0x1c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_sub_point__sub_point() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(0x600000000, -0x280000000, 0x1c0000000));
    assert!(p.sub_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_add_vector__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _v: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x300000000, -0x200000000, 0x5c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_add_vector__add_vector() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let v: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x300000000, -0x200000000, 0x5c0000000));
    assert!(p.add_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_point3_add_vector__add_assign() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let v: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x300000000, -0x200000000, 0x5c0000000));
    let mut r = p;
    r += v;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point3_sub_vector__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _v: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(0x600000000, -0x280000000, 0x1c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_sub_vector__sub_vector() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let v: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(0x600000000, -0x280000000, 0x1c0000000));
    assert!(p.sub_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_point3_sub_vector__sub_assign() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let v: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(0x600000000, -0x280000000, 0x1c0000000));
    let mut r = p;
    r -= v;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point3_scale__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Point3<Fixed> = black_box(p3(0x3c0000000, -0x5a0000000, 0x960000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_scale__scale() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Point3<Fixed> = black_box(p3(0x3c0000000, -0x5a0000000, 0x960000000));
    assert!(p.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_point3_scale__mul_assign() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Point3<Fixed> = black_box(p3(0x3c0000000, -0x5a0000000, 0x960000000));
    let mut r = p;
    r *= k;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point3_unscale__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Point3<Fixed> = black_box(p3(0x99999999, -0xe6666667, 0x180000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_unscale__unscale() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Point3<Fixed> = black_box(p3(0x9999999a, -0xe6666666, 0x180000000));
    assert!(p.unscale(k) == e);
}

#[test]
#[inline(never)]
fn bench_point3_unscale__div_assign() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Point3<Fixed> = black_box(p3(0x9999999a, -0xe6666666, 0x180000000));
    let mut r = p;
    r /= k;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_point3_neg__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, 0x240000000, -0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_neg__neg() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, 0x240000000, -0x3c0000000));
    assert!(-p == e);
}

#[test]
#[inline(never)]
fn bench_point3_inf_sup__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x480000000, -0x240000000, 0x200000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_inf_sup__inf() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x480000000, -0x240000000, 0x200000000));
    assert!(p.inf(q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_inf_sup__sup() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(0x180000000, 0x40000000, 0x3c0000000));
    assert!(p.sup(q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_inf_sup__inf_sup() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: (Point3<Fixed>, Point3<Fixed>) = black_box(
        (p3(-0x480000000, -0x240000000, 0x200000000), p3(0x180000000, 0x40000000, 0x3c0000000)),
    );
    assert!(p.inf_sup(q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_abs_diff_eq__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _q: Point3<Fixed> = black_box(p3(0x180000003, -0x23ffffffd, 0x3c0000003));
    let _ulps: u64 = black_box(3);
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_abs_diff_eq__within() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(0x180000003, -0x23ffffffd, 0x3c0000003));
    let ulps: u64 = black_box(3);
    let e: bool = black_box(true);
    assert!(p.abs_diff_eq(q, ulps) == e);
}

#[test]
#[inline(never)]
fn bench_point3_abs_diff_eq__outside() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(0x180000003, -0x23ffffffd, 0x3c0000003));
    let ulps: u64 = black_box(3);
    let e: bool = black_box(false);
    assert!(p.abs_diff_eq(q, ulps - 1) == e);
}

#[test]
#[inline(never)]
fn bench_point3_lerp__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let _t: Fixed = black_box(fx(0x80000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_lerp__lerp() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let t: Fixed = black_box(fx(0x80000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(p.lerp(q, t) == e);
}

#[test]
#[inline(never)]
fn bench_point3_center__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_center__alt_lerp() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(alt_center_lerp(p, q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_center__alt_add_scale() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(alt_center_add_scale(p, q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_center__alt_add_div() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Point3<Fixed> = black_box(p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(alt_center_add_div(p, q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_distance_squared__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x2d50000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_point3_distance_squared__alt_unfused() {
    let p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x2d50000000));
    assert!(alt_distance_squared_unfused(p, q) == e);
}

#[test]
#[inline(never)]
fn bench_point3_distance__baseline() {
    let _p: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x3c0000000));
    let _q: Point3<Fixed> = black_box(p3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x6bb40b374));
    assert!(e == e);
}
