//! Gas benchmarks of `Unit` (`bench_unit<n>_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, u2, u3, u4, v2, v3, v4};
use crate::base::unit::UnitInternalTrait;
use crate::base::vector2::Vector2;
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::base::vector4::Vector4;
use super::{Unit, Unit2Trait, Unit3Trait, Unit4Trait, UnitTrait};

// --- alternative implementations (losers)

/// `renormalize_fast` as upstream writes it: `v * (1/2 * (3 - |v|²))`, one product per component.
/// Bit-identical to `renormalize_fast` (whose `mul_add(s, -1/2, 3/2)` floors the same value as
/// `1/2 * (3 - s)`), only the gas differs: one subtraction and one rescaled product more.
#[inline(always)]
fn alt_renormalize_fast_upstream(u: Unit<Vector3<Fixed>>) -> Unit<Vector3<Fixed>> {
    let v = u.value;
    let three: Fixed = Real::TWO + Real::ONE;
    let f = Real::HALF * (three - v.norm_squared());
    Unit { value: Vector3 { x: v.x * f, y: v.y * f, z: v.z * f } }
}

/// `renormalize_fast` as `v + v * (1/2 * (1 - s))`: one fused `mul_add` per component (dearer than
/// one product per component).
#[inline(always)]
fn alt_renormalize_fast_mul_add(u: Unit<Vector3<Fixed>>) -> Unit<Vector3<Fixed>> {
    let v = u.value;
    let f = Real::HALF * (Real::ONE - v.norm_squared());
    Unit {
        value: Vector3 {
            x: Real::mul_add(v.x, f, v.x),
            y: Real::mul_add(v.y, f, v.y),
            z: Real::mul_add(v.z, f, v.z),
        },
    }
}

/// `renormalize_fast` as `v * (1 + 1/2 * (1 - s))`: the same factor without the constant 3.
#[inline(always)]
fn alt_renormalize_fast_scale(u: Unit<Vector3<Fixed>>) -> Unit<Vector3<Fixed>> {
    let v = u.value;
    let f = Real::ONE + Real::HALF * (Real::ONE - v.norm_squared());
    Unit { value: Vector3 { x: v.x * f, y: v.y * f, z: v.z * f } }
}

/// `renormalize_fast` through `lerp`: `v + (0 - v) * -f` per component (an extra exact
/// difference per component).
#[inline(always)]
fn alt_renormalize_fast_lerp(u: Unit<Vector3<Fixed>>) -> Unit<Vector3<Fixed>> {
    let v = u.value;
    let f = Real::HALF * (Real::ONE - v.norm_squared());
    Unit {
        value: Vector3 {
            x: Real::lerp(v.x, Real::ZERO, -f),
            y: Real::lerp(v.y, Real::ZERO, -f),
            z: Real::lerp(v.z, Real::ZERO, -f),
        },
    }
}

/// `dot` with one rounding and one overflow check per product (what AGENTS.md rule 4 forbids).
#[inline(always)]
fn alt_dot_unfused(a: Unit<Vector3<Fixed>>, b: Unit<Vector3<Fixed>>) -> Fixed {
    a.value.x * b.value.x + a.value.y * b.value.y + a.value.z * b.value.z
}

// --- why the alternatives lost

#[test]
fn test_renormalize_fast_alts_are_bit_identical() {
    // `floor((3 - s) / 2) = 1 + floor((1 - s) / 2)` in raw units, and `v + floor(v * f)` floors
    // once like `floor(v * (1 + f))`: the variants only differ in gas.
    let mut units = array![
        u3(1393475576, -2090213367, 3483688943), u3(1393464429, -2090196645, 3483661074),
        u3(1394864867, -2092297303, 3487162170), u3(1189375559, -1585834079, 4757502235),
        u3(0, 0, 0), u3(-0x100000000, 5, 7),
    ]
        .span();
    while let Some(u) = units.pop_front() {
        let r = (*u).renormalized_fast();
        assert!(alt_renormalize_fast_upstream(*u) == r);
        assert!(alt_renormalize_fast_mul_add(*u) == r);
        assert!(alt_renormalize_fast_scale(*u) == r);
        assert!(alt_renormalize_fast_lerp(*u) == r);
    }
}

#[test]
fn test_dot_alt_unfused_is_less_accurate() {
    // Three products of 0.5 ulp: the fused kernel adds them exactly before flooring, the unfused
    // version floors each to 0.
    let a = u3(46341, 46341, 46341);
    assert!(a.dot(a) == fx(1));
    assert!(alt_dot_unfused(a, a) == fx(0));
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_unit3_new_unchecked__baseline() {
    let _v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Unit<Vector3<Fixed>> = black_box(
        Unit { value: v3(0x180000000, -0x240000000, 0x3c0000000) },
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_new_unchecked__new_unchecked() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let e: Unit<Vector3<Fixed>> = black_box(
        Unit { value: v3(0x180000000, -0x240000000, 0x3c0000000) },
    );
    assert!(UnitTrait::new_unchecked(v) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_new_normalize__baseline() {
    let _v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _min_norm: Fixed = black_box(fx(0x10000));
    let _n: Fixed = black_box(fx(0x49f9146ee));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_new_normalize__new_normalize() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _min_norm: Fixed = black_box(fx(0x10000));
    let _n: Fixed = black_box(fx(0x49f9146ee));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa5, -0x7c960777, 0xcfa4b71c));
    assert!(UnitTrait::new_normalize(v) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_new_normalize__new_and_get() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let _min_norm: Fixed = black_box(fx(0x10000));
    let n: Fixed = black_box(fx(0x49f9146ee));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa5, -0x7c960777, 0xcfa4b71c));
    let (u, m) = UnitTrait::new_and_get(v);
    assert!(u == e);
    assert!(m == n);
}

#[test]
#[inline(never)]
fn bench_unit3_new_normalize__try_new() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let min_norm: Fixed = black_box(fx(0x10000));
    let _n: Fixed = black_box(fx(0x49f9146ee));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa5, -0x7c960777, 0xcfa4b71c));
    assert!(UnitTrait::try_new(v, min_norm) == Some(e));
}

#[test]
#[inline(never)]
fn bench_unit3_new_normalize__try_new_and_get() {
    let v: Vector3<Fixed> = black_box(v3(0x180000000, -0x240000000, 0x3c0000000));
    let min_norm: Fixed = black_box(fx(0x10000));
    let n: Fixed = black_box(fx(0x49f9146ee));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa5, -0x7c960777, 0xcfa4b71c));
    assert!(UnitTrait::try_new_and_get(v, min_norm) == Some((e, n)));
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize_fast__baseline() {
    let _u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa3, -0x7c960779, 0xcfa4b71b));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize_fast__fma_factor() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa3, -0x7c960779, 0xcfa4b71b));
    let mut renormalized = u;
    renormalized.renormalize_fast();
    assert!(renormalized == e);
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize_fast__alt_upstream() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa3, -0x7c960779, 0xcfa4b71b));
    assert!(alt_renormalize_fast_upstream(u) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize_fast__alt_mul_add() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa3, -0x7c960779, 0xcfa4b71b));
    assert!(alt_renormalize_fast_mul_add(u) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize_fast__alt_scale() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa3, -0x7c960779, 0xcfa4b71b));
    assert!(alt_renormalize_fast_scale(u) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize_fast__alt_lerp() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa3, -0x7c960779, 0xcfa4b71b));
    assert!(alt_renormalize_fast_lerp(u) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize_fast__alt_exact() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960779, 0xcfa4b71d));
    let mut renormalized = u;
    let _ = renormalized.renormalize();
    assert!(renormalized == e);
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize__baseline() {
    let _u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa3, -0x7c960779, 0xcfa4b71c));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_renormalize__renormalize() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530ebff8, -0x7c961ff7, 0xcfa4dfef));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960779, 0xcfa4b71d));
    let mut renormalized = u;
    let _ = renormalized.renormalize();
    assert!(renormalized == e);
}

#[test]
#[inline(never)]
fn bench_unit3_neg__baseline() {
    let _u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let e: Unit<Vector3<Fixed>> = black_box(u3(-0x530eafa4, 0x7c960778, -0xcfa4b71c));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_neg__neg() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let e: Unit<Vector3<Fixed>> = black_box(u3(-0x530eafa4, 0x7c960778, -0xcfa4b71c));
    assert!(-u == e);
}

#[test]
#[inline(never)]
fn bench_unit3_dot__baseline() {
    let _u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let _w: Unit<Vector3<Fixed>> = black_box(u3(0xcfa4b71c, -0x7c960778, 0x530eafa4));
    let _v: Vector3<Fixed> = black_box(v3(0xcfa4b71c, -0x7c960778, 0x530eafa4));
    let e: Fixed = black_box(fx(0xc35e50d6));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_dot__dot() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let w: Unit<Vector3<Fixed>> = black_box(u3(0xcfa4b71c, -0x7c960778, 0x530eafa4));
    let _v: Vector3<Fixed> = black_box(v3(0xcfa4b71c, -0x7c960778, 0x530eafa4));
    let e: Fixed = black_box(fx(0xc35e50d6));
    assert!(u.dot(w) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_dot__alt_unfused() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let w: Unit<Vector3<Fixed>> = black_box(u3(0xcfa4b71c, -0x7c960778, 0x530eafa4));
    let _v: Vector3<Fixed> = black_box(v3(0xcfa4b71c, -0x7c960778, 0x530eafa4));
    let e: Fixed = black_box(fx(0xc35e50d5));
    assert!(alt_dot_unfused(u, w) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_scale__baseline() {
    let _u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let _k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0xcfa4b71a, -0x1377712ac, 0x2071bc9c6));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_scale__scale() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let k: Fixed = black_box(fx(0x280000000));
    let e: Vector3<Fixed> = black_box(v3(0xcfa4b71a, -0x1377712ac, 0x2071bc9c6));
    assert!(u.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_abs_diff_eq__baseline() {
    let _u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let _w: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa7, -0x7c960775, 0xcfa4b71f));
    let _ulps: u64 = black_box(3);
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_abs_diff_eq__within() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let w: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa7, -0x7c960775, 0xcfa4b71f));
    let ulps: u64 = black_box(3);
    let e: bool = black_box(true);
    assert!(u.abs_diff_eq(w, ulps) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_abs_diff_eq__outside() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let w: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa7, -0x7c960775, 0xcfa4b71f));
    let ulps: u64 = black_box(3);
    let e: bool = black_box(false);
    assert!(u.abs_diff_eq(w, ulps - 1) == e);
}

#[test]
#[inline(never)]
fn bench_unit3_axes__baseline() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x100000000, 0x0, 0x0));
    let r = black_box(e);
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit3_axes__x_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x100000000, 0x0, 0x0));
    let r = black_box(Unit3Trait::<Fixed>::x_axis());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit3_axes__y_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x0, 0x100000000, 0x0));
    let r = black_box(Unit3Trait::<Fixed>::y_axis());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit3_axes__z_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector3<Fixed>> = black_box(u3(0x0, 0x0, 0x100000000));
    let r = black_box(Unit3Trait::<Fixed>::z_axis());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit3_into_inner__baseline() {
    let _u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let e: Vector3<Fixed> = black_box(v3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit3_into_inner__into_inner() {
    let u: Unit<Vector3<Fixed>> = black_box(u3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    let e: Vector3<Fixed> = black_box(v3(0x530eafa4, -0x7c960778, 0xcfa4b71c));
    assert!(u.into_inner() == e);
}

#[test]
#[inline(never)]
fn bench_unit2_new_normalize__baseline() {
    let _v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _min_norm: Fixed = black_box(fx(0x10000));
    let _n: Fixed = black_box(fx(0x2b4440e69));
    let e: Unit<Vector2<Fixed>> = black_box(u2(0x8e00d501, -0xd5013f83));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit2_new_normalize__new_normalize() {
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _min_norm: Fixed = black_box(fx(0x10000));
    let _n: Fixed = black_box(fx(0x2b4440e69));
    let e: Unit<Vector2<Fixed>> = black_box(u2(0x8e00d502, -0xd5013f83));
    assert!(UnitTrait::new_normalize(v) == e);
}

#[test]
#[inline(never)]
fn bench_unit2_renormalize_fast__baseline() {
    let _u: Unit<Vector2<Fixed>> = black_box(u2(0x8e00f0ec, -0xd5016964));
    let e: Unit<Vector2<Fixed>> = black_box(u2(0x8e00d500, -0xd5013f83));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit2_renormalize_fast__fma_factor() {
    let u: Unit<Vector2<Fixed>> = black_box(u2(0x8e00f0ec, -0xd5016964));
    let e: Unit<Vector2<Fixed>> = black_box(u2(0x8e00d500, -0xd5013f83));
    let mut renormalized = u;
    renormalized.renormalize_fast();
    assert!(renormalized == e);
}

#[test]
#[inline(never)]
fn bench_unit2_dot__baseline() {
    let _u: Unit<Vector2<Fixed>> = black_box(u2(0x8e00d501, -0xd5013f83));
    let _w: Unit<Vector2<Fixed>> = black_box(u2(-0xd5013f83, 0x8e00d501));
    let _v: Vector2<Fixed> = black_box(v2(-0xd5013f83, 0x8e00d501));
    let e: Fixed = black_box(fx(-0xec4ec4ec));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit2_dot__dot() {
    let u: Unit<Vector2<Fixed>> = black_box(u2(0x8e00d501, -0xd5013f83));
    let w: Unit<Vector2<Fixed>> = black_box(u2(-0xd5013f83, 0x8e00d501));
    let _v: Vector2<Fixed> = black_box(v2(-0xd5013f83, 0x8e00d501));
    let e: Fixed = black_box(fx(-0xec4ec4ec));
    assert!(u.dot(w) == e);
}

#[test]
#[inline(never)]
fn bench_unit2_axes__baseline() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector2<Fixed>> = black_box(u2(0x100000000, 0x0));
    let r = black_box(e);
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit2_axes__x_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector2<Fixed>> = black_box(u2(0x100000000, 0x0));
    let r = black_box(Unit2Trait::<Fixed>::x_axis());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit2_axes__y_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector2<Fixed>> = black_box(u2(0x0, 0x100000000));
    let r = black_box(Unit2Trait::<Fixed>::y_axis());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit4_new_normalize__baseline() {
    let _v: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x40000000));
    let _min_norm: Fixed = black_box(fx(0x10000));
    let _n: Fixed = black_box(fx(0x4a14bed26));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x52efab16, -0x7c6780a3, 0xcf572bb9, 0xdd29c83));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit4_new_normalize__new_normalize() {
    let v: Vector4<Fixed> = black_box(v4(0x180000000, -0x240000000, 0x3c0000000, 0x40000000));
    let _min_norm: Fixed = black_box(fx(0x10000));
    let _n: Fixed = black_box(fx(0x4a14bed26));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x52efab17, -0x7c6780a2, 0xcf572bb9, 0xdd29c84));
    assert!(UnitTrait::new_normalize(v) == e);
}

#[test]
#[inline(never)]
fn bench_unit4_renormalize_fast__baseline() {
    let _u: Unit<Vector4<Fixed>> = black_box(u4(0x52efbb64, -0x7c679918, 0xcf57547d, 0xdd29f3b));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x52efab15, -0x7c6780a3, 0xcf572bb9, 0xdd29c83));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit4_renormalize_fast__fma_factor() {
    let u: Unit<Vector4<Fixed>> = black_box(u4(0x52efbb64, -0x7c679918, 0xcf57547d, 0xdd29f3b));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x52efab15, -0x7c6780a3, 0xcf572bb9, 0xdd29c83));
    let mut renormalized = u;
    renormalized.renormalize_fast();
    assert!(renormalized == e);
}

#[test]
#[inline(never)]
fn bench_unit4_dot__baseline() {
    let _u: Unit<Vector4<Fixed>> = black_box(u4(0x52efab16, -0x7c6780a3, 0xcf572bb9, 0xdd29c83));
    let _w: Unit<Vector4<Fixed>> = black_box(u4(-0xa77f761d, 0x9d077eba, -0x5e37b270, 0x3ecfcc4a));
    let _v: Vector4<Fixed> = black_box(v4(-0xa77f761d, 0x9d077eba, -0x5e37b270, 0x3ecfcc4a));
    let e: Fixed = black_box(fx(-0xcb7da625));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit4_dot__dot() {
    let u: Unit<Vector4<Fixed>> = black_box(u4(0x52efab16, -0x7c6780a3, 0xcf572bb9, 0xdd29c83));
    let w: Unit<Vector4<Fixed>> = black_box(u4(-0xa77f761d, 0x9d077eba, -0x5e37b270, 0x3ecfcc4a));
    let _v: Vector4<Fixed> = black_box(v4(-0xa77f761d, 0x9d077eba, -0x5e37b270, 0x3ecfcc4a));
    let e: Fixed = black_box(fx(-0xcb7da625));
    assert!(u.dot(w) == e);
}

#[test]
#[inline(never)]
fn bench_unit4_axes__baseline() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x100000000, 0x0, 0x0, 0x0));
    let r = black_box(e);
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit4_axes__x_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x100000000, 0x0, 0x0, 0x0));
    let r = black_box(Unit4Trait::<Fixed>::x_axis());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit4_axes__y_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x0, 0x100000000, 0x0, 0x0));
    let r = black_box(Unit4Trait::<Fixed>::y_axis());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit4_axes__z_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x0, 0x0, 0x100000000, 0x0));
    let r = black_box(Unit4Trait::<Fixed>::z_axis());
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_unit4_axes__w_axis() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Unit<Vector4<Fixed>> = black_box(u4(0x0, 0x0, 0x0, 0x100000000));
    let r = black_box(Unit4Trait::<Fixed>::w_axis());
    assert!(r == e);
}
