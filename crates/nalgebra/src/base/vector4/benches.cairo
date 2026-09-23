//! Gas benchmarks of `Vector4` (`bench_vector4_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix_test_utils::{fx, v2, v3, v4};
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use super::{Vector4, Vector4AngleTrait, Vector4Trait};

// --- alternative implementations (losers)

/// `unscale` through one reciprocal: cheaper, but `1 / k` is rounded before being amplified by
/// the components.
#[inline(always)]
fn alt_unscale_recip(v: Vector4<Fixed>, k: Fixed) -> Vector4<Fixed> {
    v.scale(k.recip())
}

/// `normalize` through the reciprocal of the norm: off by about `norm` ulp, overflows for norms
/// up to `2^-31`.
#[inline(always)]
fn alt_normalize_recip(v: Vector4<Fixed>) -> Vector4<Fixed> {
    v.scale(v.norm().recip())
}

/// `normalize` through `recip(sqrt(norm_squared))`: the squared norm overflows above 46 340 and
/// has no precision left for short vectors.
#[inline(always)]
fn alt_normalize_recip_sqrt(v: Vector4<Fixed>) -> Vector4<Fixed> {
    v.scale(v.norm_squared().sqrt().recip())
}

/// `cap_magnitude` as `normalize().scale(max)`: accurate to about `max` ulp, one more division
/// per component.
#[inline(always)]
fn alt_cap_magnitude_normalize(v: Vector4<Fixed>, max: Fixed) -> Vector4<Fixed> {
    let n = v.norm();
    if n <= max {
        v
    } else {
        v.unscale(n).scale(max)
    }
}

/// `dot` with one rounding and one overflow check per product (what AGENTS.md rule 4 forbids).
#[inline(always)]
fn alt_dot_unfused(a: Vector4<Fixed>, b: Vector4<Fixed>) -> Fixed {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

// --- why the alternatives lost

#[test]
fn test_unscale_alt_recip_is_less_accurate() {
    // (3000, -4000, ..) / 3: exact floors vs an error of 1334 ulp through `recip(3)`.
    let v = v4(0xbb800000000, -0xfa000000000, 0, 0);
    assert!(v.unscale(fx(0x300000000)) == v4(0x3e800000000, -5726623061333, 0, 0));
    assert!(alt_unscale_recip(v, fx(0x300000000)) == v4(4294967295000, -5726623060000, 0, 0));
    assert!(!alt_unscale_recip(v, fx(0x300000000)).abs_diff_eq(v.unscale(fx(0x300000000)), 1000));
}

#[test]
fn test_normalize_alt_recip_is_less_accurate() {
    // (30000, -40000, ..) / 50000 = (0.6, -0.8, ..): exact floors vs 13837 ulp through
    // `recip(50000)`.
    let v = v4(0x753000000000, -0x9c4000000000, 0, 0);
    assert!(v.normalize() == v4(2576980377, -3435973836, 0, 0));
    assert!(alt_normalize_recip(v) == v4(2576970000, -3435960000, 0, 0));
    assert!(!alt_normalize_recip(v).abs_diff_eq(v.normalize(), 10000));
}

#[test]
fn test_normalize_tiny_is_exact() {
    // (3, -4, ..) ulp has a norm of 5 ulp: the divisions still give (0.6, -0.8, ..) floored.
    assert!(v4(3, -4, 0, 0).normalize() == v4(2576980377, -3435973836, 0, 0));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_normalize_alt_recip_overflows_on_tiny() {
    // Norm 1 ulp: `normalize` gives (1, -1, ..), the reciprocal of the norm does not fit.
    assert!(v4(1, -1, 0, 0).normalize() == v4(0x100000000, -0x100000000, 0, 0));
    let _ = alt_normalize_recip(black_box(v4(1, -1, 0, 0)));
}

#[test]
fn test_normalize_huge_does_not_overflow() {
    // Norm 141 421.35: its square does not fit Q32.32, the norm does.
    assert!(
        v4(0x186a000000000, -0x186a000000000, 0, 0)
            .normalize() == v4(3037000499, -3037000499, 0, 0),
    );
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_normalize_alt_recip_sqrt_overflows_on_huge() {
    let _ = alt_normalize_recip_sqrt(black_box(v4(0x186a000000000, -0x186a000000000, 0, 0)));
}

#[test]
fn test_normalize_alt_recip_sqrt_is_wrong_on_short() {
    // (3, -4, ..) * 2^-18 has a squared norm of 25 * 2^-36, floored to 1 ulp: (0.75, -1, ..)
    // instead of (0.6, -0.8, ..).
    let v = v4(49152, -65536, 0, 0);
    assert!(v.normalize() == v4(2576980377, -3435973836, 0, 0));
    assert!(alt_normalize_recip_sqrt(v) == v4(0xc0000000, -0x100000000, 0, 0));
}

#[test]
fn test_cap_magnitude_alt_normalize_is_more_accurate() {
    // (30000, -40000, ..) capped to 1: upstream's `scale(max / norm)` is 13837 ulp short of
    // (0.6, -0.8, ..), the price of flooring `1 / 50000`; `normalize().scale(max)` is exact here.
    let v = v4(0x753000000000, -0x9c4000000000, 0, 0);
    assert!(v.cap_magnitude(fx(0x100000000)) == v4(2576970000, -3435960000, 0, 0));
    assert!(alt_cap_magnitude_normalize(v, fx(0x100000000)) == v4(2576980377, -3435973836, 0, 0));
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_vector4_new__baseline() {
    let _x: Fixed = black_box(fx(0x180000000));
    let _y: Fixed = black_box(fx(-0x240000000));
    let _z: Fixed = black_box(fx(0x3c0000000));
    let _w: Fixed = black_box(fx(-0x80000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_new__new() {
    let x: Fixed = black_box(fx(0x180000000));
    let y: Fixed = black_box(fx(-0x240000000));
    let z: Fixed = black_box(fx(0x3c0000000));
    let w: Fixed = black_box(fx(-0x80000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    assert!(Vector4Trait::<Fixed>::new(x, y, z, w) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_fill__baseline() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0, 0, 0, 0));
    let r = black_box(e);
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_fill__zeros() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0, 0, 0, 0));
    let r = black_box(Vector4Trait::<Fixed>::zeros());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_fill__repeat() {
    let s: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x280000000, 0x280000000, 0x280000000, 0x280000000));
    let r = black_box(Vector4Trait::<Fixed>::repeat(s));
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_fill__from_element() {
    let s: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x280000000, 0x280000000, 0x280000000, 0x280000000));
    let r = black_box(Vector4Trait::<Fixed>::from_element(s));
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_fill__x() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x100000000, 0, 0, 0));
    let r = black_box(Vector4Trait::<Fixed>::x());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_fill__y() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0, 0x100000000, 0, 0));
    let r = black_box(Vector4Trait::<Fixed>::y());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_fill__z() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0, 0, 0x100000000, 0));
    let r = black_box(Vector4Trait::<Fixed>::z());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_fill__w() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0, 0, 0, 0x100000000));
    let r = black_box(Vector4Trait::<Fixed>::w());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_xy__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_xy__xy() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    assert!(a.xy() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_xyz__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_xyz__xyz() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(a.xyz() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_from__baseline() {
    let _t: (Fixed, Fixed, Fixed, Fixed) = black_box(
        (fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000)),
    );
    let _r: [Fixed; 4] = black_box(
        [fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000)],
    );
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_from__tuple() {
    let t: (Fixed, Fixed, Fixed, Fixed) = black_box(
        (fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000)),
    );
    let _r: [Fixed; 4] = black_box(
        [fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000)],
    );
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    assert!(t.into() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_from__array() {
    let _t: (Fixed, Fixed, Fixed, Fixed) = black_box(
        (fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000)),
    );
    let r: [Fixed; 4] = black_box(
        [fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000)],
    );
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    assert!(r.into() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_into__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x180000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_into__tuple() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x180000000));
    let (x, _, _, _): (Fixed, Fixed, Fixed, Fixed) = a.into();
    assert!(x == e);
}

#[test]
#[inline(never)]
fn bench_vector4_into__array() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x180000000));
    let [x, _, _, _]: [Fixed; 4] = a.into();
    assert!(x == e);
}

#[test]
#[inline(never)]
fn bench_vector4_add__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-0x300000000, -0x200000000, 0x5c0000000, 0x780000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_add__add() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-0x300000000, -0x200000000, 0x5c0000000, 0x780000000));
    assert!(a + b == e);
}

#[test]
#[inline(never)]
fn bench_vector4_add__add_assign() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-0x300000000, -0x200000000, 0x5c0000000, 0x780000000));
    let mut r = a;
    r += b;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_sub__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(0x600000000, -0x280000000, 0x1c0000000, -0x880000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_sub__sub() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(0x600000000, -0x280000000, 0x1c0000000, -0x880000000));
    assert!(a - b == e);
}

#[test]
#[inline(never)]
fn bench_vector4_sub__sub_assign() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(0x600000000, -0x280000000, 0x1c0000000, -0x880000000));
    let mut r = a;
    r -= b;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_neg__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector4<Fixed> = black_box(v4(-0x180000000, 0x240000000, -0x3c0000000, 0x80000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_neg__neg() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector4<Fixed> = black_box(v4(-0x180000000, 0x240000000, -0x3c0000000, 0x80000000));
    assert!(-a == e);
}

#[test]
#[inline(never)]
fn bench_vector4_scale__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x3c0000000, -0x5a0000000, 0x960000000, -0x140000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_scale__scale() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x3c0000000, -0x5a0000000, 0x960000000, -0x140000000));
    assert!(a.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_scale__mul_assign() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x3c0000000, -0x5a0000000, 0x960000000, -0x140000000));
    let mut r = a;
    r *= k;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_unscale__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(2576980377, -3865470567, 0x180000000, -858993460));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_unscale__unscale() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(2576980377, -3865470566, 0x180000000, -858993459));
    assert!(a.unscale(k) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_unscale__div_assign() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(2576980377, -3865470566, 0x180000000, -858993459));
    let mut r = a;
    r /= k;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector4_unscale__alt_recip() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(2576980377, -3865470566, 6442450942, -858993459));
    assert!(alt_unscale_recip(a, k) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_component_mul__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-0x6c0000000, -0x90000000, 0x780000000, -0x400000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_component_mul__component_mul() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-0x6c0000000, -0x90000000, 0x780000000, -0x400000000));
    assert!(a.component_mul(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_component_div__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-1431655766, -0x900000000, 0x1e0000000, -0x10000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_component_div__component_div() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-1431655765, -0x900000000, 0x1e0000000, -0x10000000));
    assert!(a.component_div(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_abs__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, 0x240000000, 0x3c0000000, 0x80000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_abs__abs() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, 0x240000000, 0x3c0000000, 0x80000000));
    assert!(a.abs() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_inf__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-0x480000000, -0x240000000, 0x200000000, -0x80000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_inf__inf() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(-0x480000000, -0x240000000, 0x200000000, -0x80000000));
    assert!(a.inf(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_inf__sup() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, 0x40000000, 0x3c0000000, 0x800000000));
    assert!(a.sup(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_inf_sup__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: (Vector4<Fixed>, Vector4<Fixed>) = black_box(
        (
            v4(-0x480000000, -0x240000000, 0x200000000, -0x80000000),
            v4(0x180000000, 0x40000000, 0x3c0000000, 0x800000000),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_inf_sup__inf_sup() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: (Vector4<Fixed>, Vector4<Fixed>) = black_box(
        (
            v4(-0x480000000, -0x240000000, 0x200000000, -0x80000000),
            v4(0x180000000, 0x40000000, 0x3c0000000, 0x800000000),
        ),
    );
    assert!(a.inf_sup(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_min__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(-0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_min__min() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(-0x240000000));
    assert!(a.min() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_min__max() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x3c0000000));
    assert!(a.max() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_min__amin() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x80000000));
    assert!(a.amin() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_min__amax() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x3c0000000));
    assert!(a.amax() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_imin__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: usize = black_box(1);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_imin__imin() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: usize = black_box(1);
    assert!(a.imin() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_imin__imax() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: usize = black_box(2);
    assert!(a.imax() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_imin__iamin() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: usize = black_box(3);
    assert!(a.iamin() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_imin__iamax() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: usize = black_box(2);
    assert!(a.iamax() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_sum__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x280000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_sum__sum() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x280000000));
    assert!(a.sum() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_is_zero__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: bool = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_is_zero__is_zero() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: bool = black_box(false);
    assert!(a.is_zero() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_abs_diff_eq__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(6442450947, -9663676413, 16106127363, -2147483645));
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_abs_diff_eq__abs_diff_eq() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(6442450947, -9663676413, 16106127363, -2147483645));
    let e: bool = black_box(true);
    assert!(a.abs_diff_eq(b, 3) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_dot__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Fixed = black_box(fx(-0x3d0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_dot__fused() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Fixed = black_box(fx(-0x3d0000000));
    assert!(a.dot(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_dot__alt_unfused() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Fixed = black_box(fx(-0x3d0000000));
    assert!(alt_dot_unfused(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_norm_squared__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x15a0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_norm_squared__norm_squared() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x15a0000000));
    assert!(a.norm_squared() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_norm_squared__magnitude_squared() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(0x15a0000000));
    assert!(a.magnitude_squared() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_norm__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(19972752454));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_norm__norm() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(19972752454));
    assert!(a.norm() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_norm__magnitude() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Fixed = black_box(fx(19972752454));
    assert!(a.magnitude() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_metric_distance__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Fixed = black_box(fx(46568716432));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_metric_distance__metric_distance() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Fixed = black_box(fx(46568716432));
    assert!(a.metric_distance(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_normalize__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector4<Fixed> = black_box(v4(1385393233, -2078089851, 3463483084, -461797745));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_normalize__unscale() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector4<Fixed> = black_box(v4(1385393233, -2078089850, 3463483084, -461797744));
    assert!(a.normalize() == e);
}

#[test]
#[inline(never)]
fn bench_vector4_normalize__alt_recip() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector4<Fixed> = black_box(v4(1385393233, -2078089851, 3463483083, -461797745));
    assert!(alt_normalize_recip(a) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_normalize__alt_recip_sqrt() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let e: Vector4<Fixed> = black_box(v4(1385393233, -2078089851, 3463483083, -461797745));
    assert!(alt_normalize_recip_sqrt(a) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_try_normalize__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _min_norm: Fixed = black_box(fx(65536));
    let e: Option<Vector4<Fixed>> = black_box(
        Some(v4(1385393233, -2078089851, 3463483084, -461797745)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_try_normalize__some() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let min_norm: Fixed = black_box(fx(65536));
    let e: Option<Vector4<Fixed>> = black_box(
        Some(v4(1385393233, -2078089850, 3463483084, -461797744)),
    );
    assert!(a.try_normalize(min_norm) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_try_normalize__none() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let min_norm: Fixed = black_box(fx(0x6400000000));
    let e: Option<Vector4<Fixed>> = black_box(None);
    assert!(a.try_normalize(min_norm) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_cap_magnitude__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _max: Fixed = black_box(fx(0x200000000));
    let e: Vector4<Fixed> = black_box(v4(2770786467, -4156179701, 6926966167, -923595489));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_cap_magnitude__capped() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let max: Fixed = black_box(fx(0x200000000));
    let e: Vector4<Fixed> = black_box(v4(2770786467, -4156179701, 6926966167, -923595489));
    assert!(a.cap_magnitude(max) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_cap_magnitude__unchanged() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let max: Fixed = black_box(fx(0x6400000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    assert!(a.cap_magnitude(max) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_cap_magnitude__alt_normalize() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let max: Fixed = black_box(fx(0x200000000));
    let e: Vector4<Fixed> = black_box(v4(2770786466, -4156179700, 6926966168, -923595488));
    assert!(alt_cap_magnitude_normalize(a, max) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_lerp__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let _t: Fixed = black_box(fx(0x40000000));
    let e: Vector4<Fixed> = black_box(v4(0, -0x1a0000000, 0x350000000, 0x1a0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_lerp__lerp() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let t: Fixed = black_box(fx(0x40000000));
    let e: Vector4<Fixed> = black_box(v4(0, -0x1a0000000, 0x350000000, 0x1a0000000));
    assert!(a.lerp(b, t) == e);
}

// --- angle: the half-angle form against upstream's `acos` of the normalized dot product

/// `angle` the way upstream writes it: `acos(a·b / (|a|·|b|))`. `acos` is cheaper than `atan2`
/// (11 430 against 15 400 gas), but the product of the norms overflows for long vectors, and near
/// 0 the cosine has no bits left for the angle (`cos θ = 1 - θ²/2`).
#[inline(always)]
fn alt_angle_acos(a: Vector4<Fixed>, b: Vector4<Fixed>) -> Fixed {
    let n = a.norm() * b.norm();
    if n == Real::ZERO {
        return Real::ZERO;
    }
    Transcendental::acos(a.dot(b) / n)
}

#[test]
fn test_angle_alt_acos_loses_precision_on_close_directions() {
    // Two directions 2^-20 rad apart: the half-angle form gets the angle, `acos` returns exactly
    // zero, because its cosine `1 - 2^-41` floors to 1.
    let (a, b) = (v4(0x100000000, 0, 0, 0), v4(0x100000000, 0x1000, 0, 0));
    assert!(Real::abs_diff_eq(a.angle(b), fx(4096), 4));
    assert!(alt_angle_acos(a, b) == Real::ZERO);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_angle_alt_acos_overflows_on_long_vectors() {
    // Norms of 1e6: their product does not fit Q32.32, while the half-angle form normalizes
    // first and answers pi/2.
    let (a, b) = (v4(0xf424000000000, 0, 0, 0), v4(0, 0xf424000000000, 0, 0));
    assert!(Real::abs_diff_eq(a.angle(b), Real::FRAC_PI_2, 4));
    let _ = alt_angle_acos(black_box(a), black_box(b));
}

#[test]
#[inline(never)]
fn bench_vector4_angle__baseline() {
    let _a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let _b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Fixed = black_box(fx(7121693092));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector4_angle__half_angle() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Fixed = black_box(fx(7121693092));
    assert!(a.angle(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector4_angle__alt_acos() {
    let a: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000));
    let b: Vector4<Fixed> = black_box(v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000));
    let e: Fixed = black_box(fx(7121693093));
    assert!(alt_angle_acos(a, b) == e);
}
