//! Gas benchmarks of `Vector3` (`bench_vector3_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix_test_utils::{fx, v2, v3, v4};
use crate::base::vector2::Vector2;
use crate::base::vector4::Vector4;
use super::{Vector3, Vector3AngleTrait, Vector3InternalTrait, Vector3Trait};

// --- alternative implementations (losers)

/// `unscale` through one reciprocal: cheaper, but `1 / k` is rounded before being amplified by
/// the components.
#[inline(always)]
fn alt_unscale_recip(v: Vector3<Fixed>, k: Fixed) -> Vector3<Fixed> {
    v.scale(k.recip())
}

/// `normalize` through the reciprocal of the norm: off by about `norm` ulp, overflows for norms
/// up to `2^-31`.
#[inline(always)]
fn alt_normalize_recip(v: Vector3<Fixed>) -> Vector3<Fixed> {
    v.scale(v.norm().recip())
}

/// `normalize` through `recip(sqrt(norm_squared))`: the squared norm overflows above 46 340 and
/// has no precision left for short vectors.
#[inline(always)]
fn alt_normalize_recip_sqrt(v: Vector3<Fixed>) -> Vector3<Fixed> {
    v.scale(v.norm_squared().sqrt().recip())
}

/// `cap_magnitude` as `normalize().scale(max)`: accurate to about `max` ulp, one more division
/// per component.
#[inline(always)]
fn alt_cap_magnitude_normalize(v: Vector3<Fixed>, max: Fixed) -> Vector3<Fixed> {
    let n = v.norm();
    if n <= max {
        v
    } else {
        v.unscale(n).scale(max)
    }
}

/// `dot` with one rounding and one overflow check per product (what AGENTS.md rule 4 forbids).
#[inline(always)]
fn alt_dot_unfused(a: Vector3<Fixed>, b: Vector3<Fixed>) -> Fixed {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// Upstream's `orthonormal_subspace_basis` for one 3D vector: a vector orthogonal to `v` built
/// from its two largest-leverage components, normalized, and its cross product with `v`
/// (specialised for the zero component).
#[inline(always)]
fn alt_orthonormal_basis_upstream(v: Vector3<Fixed>) -> (Vector3<Fixed>, Vector3<Fixed>) {
    let zero: Fixed = Real::ZERO;
    if v.x.abs() > v.y.abs() {
        let n = Real::norm2(v.z, v.x);
        let (ax, az) = (v.z / n, (-v.x) / n);
        let u = Vector3 { x: -(az * v.y), y: Real::diff_prod(az, v.x, ax, v.z), z: ax * v.y };
        (u, Vector3 { x: ax, y: zero, z: az })
    } else {
        let n = Real::norm2(v.z, v.y);
        let (ay, az) = ((-v.z) / n, v.y / n);
        let u = Vector3 { x: Real::diff_prod(ay, v.z, az, v.y), y: az * v.x, z: -(ay * v.x) };
        (u, Vector3 { x: zero, y: ay, z: az })
    }
}

// --- why the alternatives lost

#[test]
fn test_unscale_alt_recip_is_less_accurate() {
    // (3000, -4000, ..) / 3: exact floors vs an error of 1334 ulp through `recip(3)`.
    let v = v3(0xbb800000000, -0xfa000000000, 0);
    assert!(v.unscale(fx(0x300000000)) == v3(0x3e800000000, -5726623061333, 0));
    assert!(alt_unscale_recip(v, fx(0x300000000)) == v3(4294967295000, -5726623060000, 0));
    assert!(!alt_unscale_recip(v, fx(0x300000000)).abs_diff_eq(v.unscale(fx(0x300000000)), 1000));
}

#[test]
fn test_normalize_alt_recip_is_less_accurate() {
    // (30000, -40000, ..) / 50000 = (0.6, -0.8, ..): exact floors vs 13837 ulp through
    // `recip(50000)`.
    let v = v3(0x753000000000, -0x9c4000000000, 0);
    assert!(v.normalize() == v3(2576980378, -3435973837, 0));
    assert!(alt_normalize_recip(v) == v3(2576970000, -3435960000, 0));
    assert!(!alt_normalize_recip(v).abs_diff_eq(v.normalize(), 10000));
}

#[test]
fn test_normalize_tiny_is_exact() {
    // (3, -4, ..) ulp has a norm of 5 ulp: the divisions still give (0.6, -0.8, ..) floored.
    assert!(v3(3, -4, 0).normalize() == v3(2576980378, -3435973837, 0));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_normalize_alt_recip_overflows_on_tiny() {
    // Norm 1 ulp: `normalize` gives (1, -1, ..), the reciprocal of the norm does not fit.
    assert!(v3(1, -1, 0).normalize() == v3(0x100000000, -0x100000000, 0));
    let _ = alt_normalize_recip(black_box(v3(1, -1, 0)));
}

#[test]
fn test_normalize_huge_does_not_overflow() {
    // Norm 141 421.35: its square does not fit Q32.32, the norm does.
    assert!(v3(0x186a000000000, -0x186a000000000, 0).normalize() == v3(3037000500, -3037000500, 0));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_normalize_alt_recip_sqrt_overflows_on_huge() {
    let _ = alt_normalize_recip_sqrt(black_box(v3(0x186a000000000, -0x186a000000000, 0)));
}

#[test]
fn test_normalize_alt_recip_sqrt_is_wrong_on_short() {
    // (3, -4, ..) * 2^-18 has a squared norm of 25 * 2^-36, floored to 1 ulp: (0.75, -1, ..)
    // instead of (0.6, -0.8, ..).
    let v = v3(49152, -65536, 0);
    assert!(v.normalize() == v3(2576980378, -3435973837, 0));
    assert!(alt_normalize_recip_sqrt(v) == v3(0xc0000000, -0x100000000, 0));
}

#[test]
fn test_cap_magnitude_alt_normalize_is_more_accurate() {
    // (30000, -40000, ..) capped to 1: upstream's `scale(max / norm)` is 13837 ulp short of
    // (0.6, -0.8, ..), the price of flooring `1 / 50000`; `normalize().scale(max)` is exact here.
    let v = v3(0x753000000000, -0x9c4000000000, 0);
    assert!(v.cap_magnitude(fx(0x100000000)) == v3(2576970000, -3435960000, 0));
    assert!(alt_cap_magnitude_normalize(v, fx(0x100000000)) == v3(2576980378, -3435973837, 0));
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_vector3_new__baseline() {
    let _x: Fixed = black_box(fx(0x180000000));
    let _y: Fixed = black_box(fx(-0x240000000));
    let _z: Fixed = black_box(fx(0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_new__new() {
    let x: Fixed = black_box(fx(0x180000000));
    let y: Fixed = black_box(fx(-0x240000000));
    let z: Fixed = black_box(fx(0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(Vector3Trait::<Fixed>::new(x, y, z) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_fill__baseline() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0, 0, 0));
    let r = black_box(e);
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_fill__zeros() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0, 0, 0));
    let r = black_box(Vector3Trait::<Fixed>::zeros());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_fill__repeat() {
    let s: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0x280000000, 0x280000000, 0x280000000));
    let r = black_box(Vector3Trait::<Fixed>::repeat(s));
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_fill__from_element() {
    let s: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0x280000000, 0x280000000, 0x280000000));
    let r = black_box(Vector3Trait::<Fixed>::from_element(s));
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_fill__x() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0x100000000, 0, 0));
    let r = black_box(Vector3Trait::<Fixed>::x());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_fill__y() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let r = black_box(Vector3Trait::<Fixed>::y());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_fill__z() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0, 0, 0x100000000));
    let r = black_box(Vector3Trait::<Fixed>::z());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_xy__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_xy__xy() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    assert!(a.xy() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_push__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x280000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_push__push() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x280000000));
    assert!(a.push(k) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_push__to_homogeneous() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0));
    assert!(a.to_homogeneous() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_from__baseline() {
    let _t: (Fixed, Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)));
    let _r: [Fixed; 3] = black_box([fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_from__array() {
    let _t: (Fixed, Fixed, Fixed) = black_box((fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)));
    let r: [Fixed; 3] = black_box([fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(r.into() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_into__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x180000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_into__array() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x180000000));
    let [x, _, _]: [Fixed; 3] = a.into();
    assert!(x == e);
}

#[test]
#[inline(never)]
fn bench_vector3_add__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x300000000, -0x200000000, 0x5c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_add__add() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x300000000, -0x200000000, 0x5c0000000));
    assert!(a + b == e);
}

#[test]
#[inline(never)]
fn bench_vector3_add__add_assign() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x300000000, -0x200000000, 0x5c0000000));
    let mut r = a;
    r += b;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_sub__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(0x600000000, -0x280000000, 0x1c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_sub__sub() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(0x600000000, -0x280000000, 0x1c0000000));
    assert!(a - b == e);
}

#[test]
#[inline(never)]
fn bench_vector3_sub__sub_assign() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(0x600000000, -0x280000000, 0x1c0000000));
    let mut r = a;
    r -= b;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_neg__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(-0x180000000, 0x240000000, -0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_neg__neg() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(-0x180000000, 0x240000000, -0x3c0000000));
    assert!(-a == e);
}

#[test]
#[inline(never)]
fn bench_vector3_scale__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0x3c0000000, -0x5a0000000, 0x960000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_scale__scale() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0x3c0000000, -0x5a0000000, 0x960000000));
    assert!(a.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_scale__mul_assign() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0x3c0000000, -0x5a0000000, 0x960000000));
    let mut r = a;
    r *= k;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_unscale__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(2576980377, -3865470567, 0x180000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_unscale__unscale() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(2576980378, -3865470566, 0x180000000));
    assert!(a.unscale(k) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_unscale__div_assign() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(2576980378, -3865470566, 0x180000000));
    let mut r = a;
    r /= k;
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector3_unscale__alt_recip() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(2576980377, -3865470566, 6442450942));
    assert!(alt_unscale_recip(a, k) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_component_mul__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x6c0000000, -0x90000000, 0x780000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_component_mul__component_mul() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x6c0000000, -0x90000000, 0x780000000));
    assert!(a.component_mul(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_component_div__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-1431655766, -0x900000000, 0x1e0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_component_div__component_div() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-1431655765, -0x900000000, 0x1e0000000));
    assert!(a.component_div(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_abs__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, 0x240000000, 0x3c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_abs__abs() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, 0x240000000, 0x3c0000000));
    assert!(a.abs() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_inf__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x480000000, -0x240000000, 0x200000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_inf__inf() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x480000000, -0x240000000, 0x200000000));
    assert!(a.inf(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_inf__sup() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, 0x40000000, 0x3c0000000));
    assert!(a.sup(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_inf_sup__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(-0x480000000, -0x240000000, 0x200000000), v3(0x180000000, 0x40000000, 0x3c0000000)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_inf_sup__inf_sup() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(-0x480000000, -0x240000000, 0x200000000), v3(0x180000000, 0x40000000, 0x3c0000000)),
    );
    assert!(a.inf_sup(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_min__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(-0x240000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_min__min() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(-0x240000000));
    assert!(a.min() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_min__max() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x3c0000000));
    assert!(a.max() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_min__amin() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x180000000));
    assert!(a.amin() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_min__amax() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x3c0000000));
    assert!(a.amax() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_imin__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: usize = black_box(1);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_imin__imin() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: usize = black_box(1);
    assert!(a.imin() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_imin__imax() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: usize = black_box(2);
    assert!(a.imax() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_imin__iamin() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: usize = black_box(0);
    assert!(a.iamin() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_imin__iamax() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: usize = black_box(2);
    assert!(a.iamax() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_sum__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x300000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_sum__sum() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x300000000));
    assert!(a.sum() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_is_zero__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: bool = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_is_zero__is_zero() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: bool = black_box(false);
    assert!(a.is_zero() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_abs_diff_eq__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(6442450947, -9663676413, 16106127363));
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_abs_diff_eq__abs_diff_eq() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(6442450947, -9663676413, 16106127363));
    let e: bool = black_box(true);
    assert!(a.abs_diff_eq(b, 3) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_dot__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x30000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_dot__fused() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x30000000));
    assert!(a.dot(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_dot__alt_unfused() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(0x30000000));
    assert!(alt_dot_unfused(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_cross__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x570000000, -0x13e0000000, -0x9c0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_cross__cross() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Vector3<Fixed> = black_box(v3(-0x570000000, -0x13e0000000, -0x9c0000000));
    assert!(a.cross(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_norm_squared__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x1560000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_norm_squared__norm_squared() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x1560000000));
    assert!(a.norm_squared() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_norm_squared__magnitude_squared() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(0x1560000000));
    assert!(a.magnitude_squared() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_norm__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(19856967406));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_norm__norm() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(19856967406));
    assert!(a.norm() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_norm__magnitude() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Fixed = black_box(fx(19856967406));
    assert!(a.magnitude() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_metric_distance__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(28911383412));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_metric_distance__metric_distance() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(28911383412));
    assert!(a.metric_distance(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_normalize__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, 3483678492));
    assert!(e == e);
}

/// `normalize` with one `Real::div` per component instead of `Real::div3` (the shared prepared
/// divisor of the shipped `unscale`): the same bits, three divisor preparations.
#[inline(always)]
fn alt_normalize_per_element_div(v: Vector3<Fixed>) -> Vector3<Fixed> {
    let n = v.norm();
    Vector3 { x: Real::div(v.x, n), y: Real::div(v.y, n), z: Real::div(v.z, n) }
}

#[test]
#[inline(never)]
fn bench_vector3_normalize__alt_per_element_div() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(1393471397, -2090207095, 3483678492));
    assert!(alt_normalize_per_element_div(a) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_normalize__unscale() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(1393471397, -2090207095, 3483678492));
    assert!(a.normalize() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_normalize__alt_recip() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(1393471396, -2090207095, 3483678491));
    assert!(alt_normalize_recip(a) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_normalize__alt_recip_sqrt() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Vector3<Fixed> = black_box(v3(1393471396, -2090207095, 3483678491));
    assert!(alt_normalize_recip_sqrt(a) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_try_normalize__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _min_norm: Fixed = black_box(fx(65536));
    let e: Option<Vector3<Fixed>> = black_box(Some(v3(1393471396, -2090207096, 3483678492)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_try_normalize__some() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let min_norm: Fixed = black_box(fx(65536));
    let e: Option<Vector3<Fixed>> = black_box(Some(v3(1393471397, -2090207095, 3483678492)));
    assert!(a.try_normalize(min_norm) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_try_normalize__none() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let min_norm: Fixed = black_box(fx(0x6400000000));
    let e: Option<Vector3<Fixed>> = black_box(None);
    assert!(a.try_normalize(min_norm) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_cap_magnitude__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _max: Fixed = black_box(fx(0x200000000));
    let e: Vector3<Fixed> = black_box(v3(2786942793, -4180414190, 6967356982));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_cap_magnitude__capped() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let max: Fixed = black_box(fx(0x200000000));
    let e: Vector3<Fixed> = black_box(v3(2786942794, -4180414192, 6967356986));
    assert!(a.cap_magnitude(max) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_cap_magnitude__unchanged() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let max: Fixed = black_box(fx(0x6400000000));
    let e: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    assert!(a.cap_magnitude(max) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_cap_magnitude__alt_normalize() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let max: Fixed = black_box(fx(0x200000000));
    let e: Vector3<Fixed> = black_box(v3(2786942794, -4180414190, 6967356984));
    assert!(alt_cap_magnitude_normalize(a, max) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_lerp__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let _t: Fixed = black_box(fx(0x40000000));
    let e: Vector3<Fixed> = black_box(v3(0, -0x1a0000000, 0x350000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_lerp__lerp() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let t: Fixed = black_box(fx(0x40000000));
    let e: Vector3<Fixed> = black_box(v3(0, -0x1a0000000, 0x350000000));
    assert!(a.lerp(b, t) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_orthonormal_basis_zpos__baseline() {
    let _u: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, 3483678492));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(4045339972, 374440986, -1393471396), v3(374440986, 3733305815, 2090207096)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_orthonormal_basis_zpos__duff() {
    let u: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, 3483678492));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(4045339972, 374440987, -1393471396), v3(374440987, 3733305816, 2090207096)),
    );
    assert!(u.orthonormal_basis() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_orthonormal_basis_zpos__alt_upstream() {
    let u: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, 3483678492));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(-4062632342, -716935119, 1194891865), v3(0, -3682904072, -2209742444)),
    );
    assert!(alt_orthonormal_basis_upstream(u) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_orthonormal_basis_zneg__baseline() {
    let _u: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, -3483678493));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(4045339972, 374440986, 1393471396), v3(-374440986, -3733305816, 2090207096)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_orthonormal_basis_zneg__duff() {
    let u: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, -3483678493));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(4045339972, 374440987, 1393471396), v3(-374440987, -3733305816, 2090207096)),
    );
    assert!(u.orthonormal_basis() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_orthonormal_basis_zneg__alt_upstream() {
    let u: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, -3483678493));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(-4062632342, -716935119, -1194891864), v3(0, 3682904072, -2209742444)),
    );
    assert!(alt_orthonormal_basis_upstream(u) == e);
}

// --- angle: the half-angle form against upstream's `acos` of the normalized dot product

/// `angle` the way upstream writes it: `acos(a·b / (|a|·|b|))`. `acos` is cheaper than `atan2`
/// (11 430 against 15 400 gas), but the product of the norms overflows for long vectors, and near
/// 0 the cosine has no bits left for the angle (`cos θ = 1 - θ²/2`).
#[inline(always)]
fn alt_angle_acos(a: Vector3<Fixed>, b: Vector3<Fixed>) -> Fixed {
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
    let (a, b) = (v3(0x100000000, 0, 0), v3(0x100000000, 0x1000, 0));
    assert!(Real::abs_diff_eq(a.angle(b), fx(4096), 4));
    assert!(alt_angle_acos(a, b) == Real::ZERO);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_angle_alt_acos_overflows_on_long_vectors() {
    // Norms of 1e6: their product does not fit Q32.32, while the half-angle form normalizes
    // first and answers pi/2.
    let (a, b) = (v3(0xf424000000000, 0, 0), v3(0, 0xf424000000000, 0));
    assert!(Real::abs_diff_eq(a.angle(b), Real::FRAC_PI_2, 4));
    let _ = alt_angle_acos(black_box(a), black_box(b));
}

#[test]
#[inline(never)]
fn bench_vector3_angle__baseline() {
    let _a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(6711192552));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector3_angle__half_angle() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(6711192552));
    assert!(a.angle(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector3_angle__alt_acos() {
    let a: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let b: Vector3<Fixed> = black_box(v3(-0x480000000, 0x40000000, 0x200000000));
    let e: Fixed = black_box(fx(6711192551));
    assert!(alt_angle_acos(a, b) == e);
}
