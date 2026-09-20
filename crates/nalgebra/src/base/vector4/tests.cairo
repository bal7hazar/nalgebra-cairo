//! Unit tests of `Vector4`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, robustness at both ends of the range, identities, and
//! the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution,
//! every op of the suite but `angle`, which waits for `Transcendental<Fixed>`) with
//! `cargo run --release -- emit-cairo vector4 --from vectors --max-per-dist 4 --out <oracle.cairo>
//! --ops <list>`, `<list>` being the comma-separated `vector4_<op>` names of the oracle tests
//! at the bottom of this file.

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::{FixedReal, Real, Transcendental};
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use super::{Vector4, Vector4AngleImpl, Vector4Trait, oracle};

const MAX: i64 = 0x7fffffffffffffff;
const MIN: i64 = -0x8000000000000000;

fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

fn v2(x: i64, y: i64) -> Vector2<Fixed> {
    Vector2 { x: fx(x), y: fx(y) }
}

fn v3(x: i64, y: i64, z: i64) -> Vector3<Fixed> {
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

fn v4(x: i64, y: i64, z: i64, w: i64) -> Vector4<Fixed> {
    Vector4 { x: fx(x), y: fx(y), z: fx(z), w: fx(w) }
}

/// (1.5, -2.25, 3.75, -0.5)
fn a() -> Vector4<Fixed> {
    v4(0x180000000, -0x240000000, 0x3c0000000, -0x80000000)
}

/// (-4.5, 0.25, 2, 8)
fn b() -> Vector4<Fixed> {
    v4(-0x480000000, 0x40000000, 0x200000000, 0x800000000)
}

/// (1, -2, 2, 4), of norm 5
fn p() -> Vector4<Fixed> {
    v4(0x100000000, -0x200000000, 0x200000000, 0x400000000)
}

/// A vector from an oracle tuple of raws.
fn vt(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (x, y, z, w) = t;
    v4(x, y, z, w)
}

// --- constructors, resizing, conversions

#[test]
fn test_new_sets_fields() {
    let r = Vector4Trait::<
        Fixed,
    >::new(fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000));
    assert!(
        r.x == fx(0x180000000)
            && r.y == fx(-0x240000000)
            && r.z == fx(0x3c0000000)
            && r.w == fx(-0x80000000),
    );
    assert!(r == a());
}

#[test]
fn test_zeros_is_zero() {
    assert!(Vector4Trait::<Fixed>::zeros() == v4(0, 0, 0, 0));
    assert!(Vector4Trait::<Fixed>::zeros() == Default::default());
    assert!(Vector4Trait::<Fixed>::zeros().is_zero());
}

#[test]
fn test_repeat_fills_components() {
    assert!(
        Vector4Trait::<
            Fixed,
        >::repeat(fx(-0x240000000)) == v4(-0x240000000, -0x240000000, -0x240000000, -0x240000000),
    );
    assert!(
        Vector4Trait::<
            Fixed,
        >::from_element(
            fx(-0x240000000),
        ) == v4(-0x240000000, -0x240000000, -0x240000000, -0x240000000),
    );
}

#[test]
fn test_axes_are_orthonormal() {
    assert!(Vector4Trait::<Fixed>::x() == v4(0x100000000, 0, 0, 0));
    assert!(Vector4Trait::<Fixed>::y() == v4(0, 0x100000000, 0, 0));
    assert!(Vector4Trait::<Fixed>::z() == v4(0, 0, 0x100000000, 0));
    assert!(Vector4Trait::<Fixed>::w() == v4(0, 0, 0, 0x100000000));
    // Orthonormal: unit norms, zero dot products.
    assert!(Vector4Trait::<Fixed>::x().norm() == Real::ONE);
    assert!(Vector4Trait::<Fixed>::x().dot(Vector4Trait::<Fixed>::y()) == Real::ZERO);
    assert!(Vector4Trait::<Fixed>::x().dot(Vector4Trait::<Fixed>::z()) == Real::ZERO);
    assert!(Vector4Trait::<Fixed>::x().dot(Vector4Trait::<Fixed>::w()) == Real::ZERO);
    assert!(Vector4Trait::<Fixed>::y().norm() == Real::ONE);
    assert!(Vector4Trait::<Fixed>::y().dot(Vector4Trait::<Fixed>::z()) == Real::ZERO);
    assert!(Vector4Trait::<Fixed>::y().dot(Vector4Trait::<Fixed>::w()) == Real::ZERO);
    assert!(Vector4Trait::<Fixed>::z().norm() == Real::ONE);
    assert!(Vector4Trait::<Fixed>::z().dot(Vector4Trait::<Fixed>::w()) == Real::ZERO);
    assert!(Vector4Trait::<Fixed>::w().norm() == Real::ONE);
}

#[test]
fn test_xy_keeps_first_components() {
    assert!(a().xy() == v2(0x180000000, -0x240000000));
}

#[test]
fn test_xyz_keeps_first_components() {
    assert!(a().xyz() == v3(0x180000000, -0x240000000, 0x3c0000000));
}

#[test]
fn test_tuple_conversions_roundtrip() {
    let r: Vector4<Fixed> = (fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000))
        .into();
    assert!(r == a());
    let t: (Fixed, Fixed, Fixed, Fixed) = r.into();
    assert!(t == (fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000)));
}

#[test]
fn test_array_conversions_roundtrip() {
    let r: Vector4<Fixed> = [fx(0x180000000), fx(-0x240000000), fx(0x3c0000000), fx(-0x80000000)]
        .into();
    assert!(r == a());
    let [x, y, z, w]: [Fixed; 4] = r.into();
    assert!(
        x == fx(0x180000000)
            && y == fx(-0x240000000)
            && z == fx(0x3c0000000)
            && w == fx(-0x80000000),
    );
}

// --- operators

#[test]
fn test_add_exact() {
    // (1.5, -2.25, 3.75, -0.5) + (-4.5, 0.25, 2, 8) = (-3, -2, 5.75, 7.5)
    assert!(a() + b() == v4(-0x300000000, -0x200000000, 0x5c0000000, 0x780000000));
}

#[test]
fn test_sub_exact() {
    // (1.5, -2.25, 3.75, -0.5) - (-4.5, 0.25, 2, 8) = (6, -2.5, 1.75, -8.5)
    assert!(a() - b() == v4(0x600000000, -0x280000000, 0x1c0000000, -0x880000000));
}

#[test]
fn test_neg_exact() {
    assert!(-a() == v4(-0x180000000, 0x240000000, -0x3c0000000, 0x80000000));
    assert!(-(-a()) == a());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_add_overflow() {
    let _ = black_box(v4(0, 0, 0, 9223372036854775807)) + v4(0, 0, 0, 1);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_sub_overflow() {
    let _ = black_box(v4(0, 0, 0, -0x8000000000000000)) - v4(0, 0, 0, 1);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_neg_overflow() {
    let _ = -black_box(v4(0, 0, 0, MIN));
}

#[test]
fn test_add_assign_matches_add() {
    let mut r = a();
    r += b();
    assert!(r == a() + b());
}

#[test]
fn test_sub_assign_matches_sub() {
    let mut r = a();
    r -= b();
    assert!(r == a() - b());
}

#[test]
fn test_mul_assign_matches_scale() {
    let mut r = a();
    r *= fx(0x280000000);
    assert!(r == a().scale(fx(0x280000000)));
}

#[test]
fn test_div_assign_matches_unscale() {
    let mut r = a();
    r /= fx(0x280000000);
    assert!(r == a().unscale(fx(0x280000000)));
}

// --- component-wise

#[test]
fn test_scale_exact() {
    // (1.5, -2.25, 3.75, -0.5) * 2.5 = (3.75, -5.625, 9.375, -1.25)
    assert!(a().scale(fx(0x280000000)) == v4(0x3c0000000, -0x5a0000000, 0x960000000, -0x140000000));
    assert!(a().scale(Real::ONE) == a());
    assert!(a().scale(Real::ZERO).is_zero());
}

#[test]
fn test_scale_rounds_toward_negative_infinity() {
    // [3, -3, 1, -1] ulp * 0.5: floor, whatever the sign.
    assert!(v4(3, -3, 1, -1).scale(fx(0x80000000)) == v4(1, -2, 0, -1));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_scale_overflow() {
    let _ = black_box(v4(0, 0, 0, 0x4000000000000000)).scale(fx(0x200000000));
}

#[test]
fn test_unscale_exact() {
    // (1.5, -2.25, 3.75, -0.5) / 2.5 = (0.5999999999, -0.9000000001, 1.5, -0.2000000002)
    assert!(a().unscale(fx(0x280000000)) == v4(2576980377, -3865470567, 0x180000000, -858993460));
    assert!(a().unscale(Real::ONE) == a());
}

#[test]
fn test_unscale_rounds_toward_negative_infinity() {
    // +-1 / 3, +-2 / 3: exact floors.
    assert!(
        v4(0x100000000, -0x100000000, 0x200000000, -0x200000000)
            .unscale(fx(0x300000000)) == v4(1431655765, -1431655766, 2863311530, -2863311531),
    );
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_unscale_by_zero() {
    let _ = black_box(a()).unscale(Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_unscale_overflow() {
    let _ = black_box(v4(0, 0, 0, 0x4000000000000000)).unscale(fx(0x80000000));
}

#[test]
fn test_component_mul_exact() {
    // (1.5, -2.25, 3.75, -0.5) .* (-4.5, 0.25, 2, 8) = (-6.75, -0.5625, 7.5, -4)
    assert!(a().component_mul(b()) == v4(-0x6c0000000, -0x90000000, 0x780000000, -0x400000000));
    assert!(a().component_mul(Vector4Trait::<Fixed>::repeat(Real::ONE)) == a());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_component_mul_overflow() {
    let _ = black_box(v4(0, 0, 0, 0x1000000000000)).component_mul(v4(0, 0, 0, 0x800000000000));
}

#[test]
fn test_component_div_exact() {
    // (1.5, -2.25, 3.75, -0.5) ./ (-4.5, 0.25, 2, 8) = floor of the exact quotients
    assert!(a().component_div(b()) == v4(-1431655766, -0x900000000, 0x1e0000000, -0x10000000));
    assert!(a().component_div(Vector4Trait::<Fixed>::repeat(Real::ONE)) == a());
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_component_div_by_zero() {
    let _ = black_box(a()).component_div(v4(0x100000000, 0x100000000, 0x100000000, 0));
}

#[test]
fn test_abs_exact() {
    assert!(a().abs() == v4(0x180000000, 0x240000000, 0x3c0000000, 0x80000000));
    assert!(a().abs().abs() == a().abs());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_abs_overflow() {
    let _ = black_box(v4(MIN, 0, 0, 0)).abs();
}

#[test]
fn test_inf_sup_exact() {
    // (1.5, -2.25, 3.75, -0.5) and (-4.5, 0.25, 2, 8)
    assert!(a().inf(b()) == v4(-0x480000000, -0x240000000, 0x200000000, -0x80000000));
    assert!(a().sup(b()) == v4(0x180000000, 0x40000000, 0x3c0000000, 0x800000000));
    assert!(
        a()
            .inf_sup(
                b(),
            ) == (
                v4(-0x480000000, -0x240000000, 0x200000000, -0x80000000),
                v4(0x180000000, 0x40000000, 0x3c0000000, 0x800000000),
            ),
    );
    assert!(b().inf(a()) == v4(-0x480000000, -0x240000000, 0x200000000, -0x80000000));
    assert!(a().inf(a()) == a() && a().sup(a()) == a());
    assert!(v4(MIN, MIN, MIN, MIN).inf(v4(MAX, MAX, MAX, MAX)) == v4(MIN, MIN, MIN, MIN));
}

// --- reductions

#[test]
fn test_min_max_exact() {
    // (1.5, -2.25, 3.75, -0.5)
    assert!(a().min() == fx(-0x240000000));
    assert!(a().max() == fx(0x3c0000000));
    assert!(a().amin() == fx(0x80000000));
    assert!(a().amax() == fx(0x3c0000000));
}

#[test]
fn test_min_max_every_position() {
    assert!(
        v4(-0x700000000, 0x500000000, 0x500000000, 0x500000000).min() == fx(-0x700000000)
            && v4(-0x700000000, 0x500000000, 0x500000000, 0x500000000).imin() == 0,
    );
    assert!(
        v4(0x700000000, -0x500000000, -0x500000000, -0x500000000).max() == fx(0x700000000)
            && v4(0x700000000, -0x500000000, -0x500000000, -0x500000000).imax() == 0,
    );
    assert!(
        v4(-0x700000000, 0x500000000, 0x500000000, 0x500000000).amax() == fx(0x700000000)
            && v4(-0x700000000, 0x500000000, 0x500000000, 0x500000000).iamax() == 0,
    );
    assert!(
        v4(0x300000000, -0x500000000, -0x500000000, -0x500000000).amin() == fx(0x300000000)
            && v4(0x300000000, -0x500000000, -0x500000000, -0x500000000).iamin() == 0,
    );
    assert!(
        v4(0x500000000, -0x700000000, 0x500000000, 0x500000000).min() == fx(-0x700000000)
            && v4(0x500000000, -0x700000000, 0x500000000, 0x500000000).imin() == 1,
    );
    assert!(
        v4(-0x500000000, 0x700000000, -0x500000000, -0x500000000).max() == fx(0x700000000)
            && v4(-0x500000000, 0x700000000, -0x500000000, -0x500000000).imax() == 1,
    );
    assert!(
        v4(0x500000000, -0x700000000, 0x500000000, 0x500000000).amax() == fx(0x700000000)
            && v4(0x500000000, -0x700000000, 0x500000000, 0x500000000).iamax() == 1,
    );
    assert!(
        v4(-0x500000000, 0x300000000, -0x500000000, -0x500000000).amin() == fx(0x300000000)
            && v4(-0x500000000, 0x300000000, -0x500000000, -0x500000000).iamin() == 1,
    );
    assert!(
        v4(0x500000000, 0x500000000, -0x700000000, 0x500000000).min() == fx(-0x700000000)
            && v4(0x500000000, 0x500000000, -0x700000000, 0x500000000).imin() == 2,
    );
    assert!(
        v4(-0x500000000, -0x500000000, 0x700000000, -0x500000000).max() == fx(0x700000000)
            && v4(-0x500000000, -0x500000000, 0x700000000, -0x500000000).imax() == 2,
    );
    assert!(
        v4(0x500000000, 0x500000000, -0x700000000, 0x500000000).amax() == fx(0x700000000)
            && v4(0x500000000, 0x500000000, -0x700000000, 0x500000000).iamax() == 2,
    );
    assert!(
        v4(-0x500000000, -0x500000000, 0x300000000, -0x500000000).amin() == fx(0x300000000)
            && v4(-0x500000000, -0x500000000, 0x300000000, -0x500000000).iamin() == 2,
    );
    assert!(
        v4(0x500000000, 0x500000000, 0x500000000, -0x700000000).min() == fx(-0x700000000)
            && v4(0x500000000, 0x500000000, 0x500000000, -0x700000000).imin() == 3,
    );
    assert!(
        v4(-0x500000000, -0x500000000, -0x500000000, 0x700000000).max() == fx(0x700000000)
            && v4(-0x500000000, -0x500000000, -0x500000000, 0x700000000).imax() == 3,
    );
    assert!(
        v4(0x500000000, 0x500000000, 0x500000000, -0x700000000).amax() == fx(0x700000000)
            && v4(0x500000000, 0x500000000, 0x500000000, -0x700000000).iamax() == 3,
    );
    assert!(
        v4(-0x500000000, -0x500000000, -0x500000000, 0x300000000).amin() == fx(0x300000000)
            && v4(-0x500000000, -0x500000000, -0x500000000, 0x300000000).iamin() == 3,
    );
}

#[test]
fn test_imin_imax_ties_pick_first() {
    // Upstream returns the first extremum.
    assert!(
        v4(0x100000000, 0x100000000, 0x100000000, 0x100000000).imax() == 0
            && v4(0x100000000, 0x100000000, 0x100000000, 0x100000000).imin() == 0,
    );
    assert!(
        v4(-0x100000000, -0x100000000, -0x100000000, -0x100000000).iamax() == 0
            && v4(-0x100000000, -0x100000000, -0x100000000, -0x100000000).iamin() == 0,
    );
    assert!(
        v4(0x80000000, 0x100000000, 0x100000000, 0x100000000).imax() == 1
            && v4(0x200000000, 0x100000000, 0x100000000, 0x100000000).imin() == 1,
    );
    assert!(
        v4(-0x80000000, -0x100000000, -0x100000000, -0x100000000).iamax() == 1
            && v4(-0x200000000, -0x100000000, -0x100000000, -0x100000000).iamin() == 1,
    );
    assert!(
        v4(0x80000000, 0x80000000, 0x100000000, 0x100000000).imax() == 2
            && v4(0x200000000, 0x200000000, 0x100000000, 0x100000000).imin() == 2,
    );
    assert!(
        v4(-0x80000000, -0x80000000, -0x100000000, -0x100000000).iamax() == 2
            && v4(-0x200000000, -0x200000000, -0x100000000, -0x100000000).iamin() == 2,
    );
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_amax_overflow() {
    let _ = black_box(v4(0, 0, 0, MIN)).amax();
}

#[test]
fn test_sum_exact() {
    // (1.5, -2.25, 3.75, -0.5)
    assert!(a().sum() == fx(0x280000000));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_sum_overflow() {
    let _ = black_box(v4(0x4000000000000000, 0x4000000000000000, 0, 0)).sum();
}

#[test]
fn test_is_zero_cases() {
    assert!(v4(0, 0, 0, 0).is_zero());
    assert!(!v4(1, 0, 0, 0).is_zero());
    assert!(!v4(0, 1, 0, 0).is_zero());
    assert!(!v4(0, 0, 1, 0).is_zero());
    assert!(!v4(0, 0, 0, 1).is_zero());
}

#[test]
fn test_abs_diff_eq_counts_ulps() {
    let r = a();
    assert!(r.abs_diff_eq(r, 0));
    assert!(
        r.abs_diff_eq(v4(6442450947, -0x240000000, 0x3c0000000, -0x80000000), 3)
            && !r.abs_diff_eq(v4(6442450947, -0x240000000, 0x3c0000000, -0x80000000), 2),
    );
    assert!(
        r.abs_diff_eq(v4(0x180000000, -9663676419, 0x3c0000000, -0x80000000), 3)
            && !r.abs_diff_eq(v4(0x180000000, -9663676419, 0x3c0000000, -0x80000000), 2),
    );
    assert!(
        r.abs_diff_eq(v4(0x180000000, -0x240000000, 16106127363, -0x80000000), 3)
            && !r.abs_diff_eq(v4(0x180000000, -0x240000000, 16106127363, -0x80000000), 2),
    );
    assert!(
        r.abs_diff_eq(v4(0x180000000, -0x240000000, 0x3c0000000, -2147483651), 3)
            && !r.abs_diff_eq(v4(0x180000000, -0x240000000, 0x3c0000000, -2147483651), 2),
    );
    // The difference itself may exceed the scalar range.
    assert!(!v4(MIN, MIN, MIN, MIN).abs_diff_eq(v4(MAX, MAX, MAX, MAX), 0xfffffffffffffffe));
    assert!(v4(MIN, MIN, MIN, MIN).abs_diff_eq(v4(MAX, MAX, MAX, MAX), 0xffffffffffffffff));
}

// --- products

#[test]
fn test_dot_exact() {
    // (1.5, -2.25, 3.75, -0.5) . (-4.5, 0.25, 2, 8) = -3.8125
    assert!(a().dot(b()) == fx(-0x3d0000000));
    assert!(b().dot(a()) == fx(-0x3d0000000));
    assert!(a().dot(a()) == a().norm_squared());
}

#[test]
fn test_dot_rounds_once() {
    // 4 products of 0.5 ulp each: the fused sum is floor(2.0) = 2 ulp, where rounding every
    // product would give 0.
    assert!(v4(0x80000000, 0x80000000, 0x80000000, 0x80000000).dot(v4(1, 1, 1, 1)) == fx(2));
}

#[test]
fn test_dot_intermediate_products_may_overflow() {
    // 60000^2 = 3.6e9 does not fit Q32.32, the cancelled sum does.
    assert!(
        v4(0xea6000000000, 0xea6000000000, 0, 0)
            .dot(v4(0xea6000000000, -0xea6000000000, 0x100000000, 0x100000000)) == fx(0),
    );
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_dot_overflow() {
    let _ = black_box(v4(0xea6000000000, 0xea6000000000, 0, 0))
        .dot(v4(0xea6000000000, 0xea6000000000, 0, 0));
}

// --- norms

#[test]
fn test_norm_squared_exact() {
    // |(1, -2, 2, 4)|^2 = 25
    assert!(p().norm_squared() == fx(0x1900000000));
    assert!(p().magnitude_squared() == fx(0x1900000000));
    assert!(a().norm_squared() == fx(0x15a0000000));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_norm_squared_overflow() {
    let _ = black_box(v4(0xea6000000000, 0, 0, 0)).norm_squared();
}

#[test]
fn test_norm_exact() {
    // |(1, -2, 2, 4)| = 5
    assert!(p().norm() == fx(0x500000000));
    assert!(p().magnitude() == fx(0x500000000));
    assert!(v4(0, 0, 0, 0).norm() == Real::ZERO);
    assert!(a().norm() == fx(19972752454));
}

#[test]
fn test_norm_large_magnitude_does_not_overflow() {
    // |(1e6, ..)| = 2000000: the squared norm (4e12) is far outside Q32.32.
    assert!(
        v4(0xf424000000000, 0xf424000000000, 0xf424000000000, 0xf424000000000)
            .norm() == fx(0x1e848000000000),
    );
    // Largest components.
    assert!(v4(MAX, 0, 0, 0).norm() == fx(MAX));
}

#[test]
fn test_norm_tiny_magnitude_keeps_precision() {
    // |(3, -4, ..) ulp| = 5 ulp: the squares vanish in Q32.32, not in the unscaled kernel.
    assert!(v4(3, -4, 0, 0).norm() == fx(5));
    assert!(v4(3, -4, 0, 0).norm_squared() == Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_norm_overflow() {
    let _ = black_box(v4(MAX, MAX, 0, 0)).norm();
}

#[test]
fn test_metric_distance_exact() {
    // Distance between (2.5, -4.25, 5.75, 3.5) and (1.5, -2.25, 3.75, -0.5).
    assert!(
        v4(0x280000000, -0x440000000, 0x5c0000000, 0x380000000)
            .metric_distance(a()) == fx(0x500000000),
    );
    assert!(
        a()
            .metric_distance(
                v4(0x280000000, -0x440000000, 0x5c0000000, 0x380000000),
            ) == fx(0x500000000),
    );
    assert!(a().metric_distance(a()) == Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_metric_distance_overflow() {
    let _ = black_box(v4(MAX, 0, 0, 0)).metric_distance(v4(-1, 0, 0, 0));
}

#[test]
fn test_normalize_exact() {
    // (1, -2, 2, 4) / 5, floored.
    assert!(p().normalize() == v4(858993459, -1717986919, 1717986918, 3435973836));
    assert!(v4(-0x500000000, 0, 0, 0).normalize() == -Vector4Trait::<Fixed>::x());
    assert!(v4(0, -0x500000000, 0, 0).normalize() == -Vector4Trait::<Fixed>::y());
    assert!(v4(0, 0, -0x500000000, 0).normalize() == -Vector4Trait::<Fixed>::z());
    assert!(v4(0, 0, 0, -0x500000000).normalize() == -Vector4Trait::<Fixed>::w());
}

#[test]
fn test_normalize_large_magnitude() {
    // (1e6, ..) / |(1e6, ..)|: no overflow, every component is 1 / sqrt(4) within 1 ulp.
    assert!(
        v4(0xf424000000000, 0xf424000000000, 0xf424000000000, 0xf424000000000)
            .normalize() == v4(0x80000000, 0x80000000, 0x80000000, 0x80000000),
    );
    assert!(
        v4(0xf424000000000, 0xf424000000000, 0xf424000000000, 0xf424000000000)
            .normalize()
            .abs_diff_eq(Vector4Trait::<Fixed>::repeat(fx(0x80000000)), 1),
    );
}

#[test]
fn test_normalize_tiny_magnitude() {
    // (3, -4, ..) ulp: norm 5 ulp, result (0.6, -0.8, ..) floored.
    assert!(v4(3, -4, 0, 0).normalize() == v4(2576980377, -3435973837, 0, 0));
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_normalize_zero() {
    let _ = black_box(Vector4Trait::<Fixed>::zeros()).normalize();
}

#[test]
fn test_normalize_is_unit_within_tolerance() {
    let r = a().normalize();
    assert!(r == v4(1385393233, -2078089851, 3463483084, -461797745));
    assert!(r.norm().abs_diff_eq(Real::ONE, 8));
}

#[test]
fn test_try_normalize_some() {
    assert!(p().try_normalize(Real::ZERO) == Some(p().normalize()));
    assert!(p().try_normalize(fx(0x500000000) - Real::EPSILON) == Some(p().normalize()));
}

#[test]
fn test_try_normalize_none() {
    assert!(Vector4Trait::<Fixed>::zeros().try_normalize(Real::ZERO) == None);
    // The threshold is inclusive, like upstream (`norm <= min_norm`).
    assert!(p().try_normalize(fx(0x500000000)) == None);
    assert!(v4(3, -4, 0, 0).try_normalize(Real::EPSILON) == Some(v4(3, -4, 0, 0).normalize()));
    assert!(v4(3, -4, 0, 0).try_normalize(fx(5)) == None);
}

#[test]
fn test_cap_magnitude_exact() {
    // |(1, -2, 2, 4)| = 5 capped to 2.5: the ratio 0.5 is exact.
    assert!(
        p()
            .cap_magnitude(
                fx(0x280000000),
            ) == v4(0x80000000, -0x100000000, 0x100000000, 0x200000000),
    );
    assert!(p().cap_magnitude(fx(0x280000000)).norm() == fx(0x280000000));
    // Not longer than the cap: unchanged (inclusive).
    assert!(p().cap_magnitude(fx(0x500000000)) == p());
    assert!(p().cap_magnitude(fx(MAX)) == p());
    assert!(p().cap_magnitude(Real::ZERO).is_zero());
    assert!(Vector4Trait::<Fixed>::zeros().cap_magnitude(Real::ZERO).is_zero());
}

#[test]
fn test_cap_magnitude_never_exceeds_cap_by_more_than_rounding() {
    let r = a().cap_magnitude(fx(0x200000000));
    assert!(r == v4(2770786467, -4156179701, 6926966167, -923595489));
    assert!(r.norm() <= fx(0x200000000) && r.norm().abs_diff_eq(fx(0x200000000), 16));
}

#[test]
fn test_cap_magnitude_large_magnitude() {
    assert!(
        v4(0xf424000000000, 0xf424000000000, 0xf424000000000, 0xf424000000000)
            .cap_magnitude(
                fx(0xa00000000),
            ) == v4(21474000000, 21474000000, 21474000000, 21474000000),
    );
}

// --- interpolation

#[test]
fn test_lerp_exact() {
    // From (1.5, -2.25, 3.75, -0.5) to (-4.5, 0.25, 2, 8).
    assert!(a().lerp(b(), Real::ZERO) == a());
    assert!(a().lerp(b(), Real::ONE) == b());
    assert!(a().lerp(b(), Real::HALF) == v4(-0x180000000, -0x100000000, 0x2e0000000, 0x3c0000000));
    assert!(a().lerp(b(), fx(0x40000000)) == v4(0, -0x1a0000000, 0x350000000, 0x1a0000000));
    // Not clamped: extrapolation.
    assert!(a().lerp(b(), Real::TWO) == v4(-0xa80000000, 0x2c0000000, 0x40000000, 0x1080000000));
    assert!(
        a().lerp(b(), Real::NEG_ONE) == v4(0x780000000, -0x4c0000000, 0x580000000, -0x900000000),
    );
}

#[test]
fn test_lerp_full_range_does_not_overflow() {
    // `rhs - self` does not fit the scalar; the kernel works on the exact difference.
    let (lo, hi) = (v4(MIN, MAX, MAX, MAX), v4(MAX, MIN, MIN, MIN));
    assert!(lo.lerp(hi, Real::ONE) == hi);
    assert!(lo.lerp(hi, Real::HALF) == v4(-1, -1, -1, -1));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_lerp_overflow() {
    let _ = black_box(v4(0, 0, 0, 0))
        .lerp(
            v4(0x4000000000000000, 0x4000000000000000, 0x4000000000000000, 0x4000000000000000),
            fx(0x400000000),
        );
}

// --- angle (behind `Transcendental`)

/// Test double for `Transcendental`: `atan2` is exact at the cardinal configurations used below
/// and panics elsewhere. It is passed EXPLICITLY to `Vector4AngleImpl`, so these tests do not
/// depend on (nor conflict with) the real `Transcendental<Fixed>` impl of `simba`.
impl CardinalAtan2 of Transcendental<Fixed> {
    fn sin(self: Fixed) -> Fixed {
        core::panic_with_felt252('unused')
    }
    fn cos(self: Fixed) -> Fixed {
        core::panic_with_felt252('unused')
    }
    fn tan(self: Fixed) -> Fixed {
        core::panic_with_felt252('unused')
    }
    fn asin(self: Fixed) -> Fixed {
        core::panic_with_felt252('unused')
    }
    fn acos(self: Fixed) -> Fixed {
        core::panic_with_felt252('unused')
    }
    fn atan(self: Fixed) -> Fixed {
        core::panic_with_felt252('unused')
    }
    fn exp(self: Fixed) -> Fixed {
        core::panic_with_felt252('unused')
    }
    fn ln(self: Fixed) -> Fixed {
        core::panic_with_felt252('unused')
    }
    fn sin_cos(self: Fixed) -> (Fixed, Fixed) {
        core::panic_with_felt252('unused')
    }
    fn atan2(y: Fixed, x: Fixed) -> Fixed {
        if y == Real::ZERO {
            Real::ZERO
        } else if x == Real::ZERO {
            Real::FRAC_PI_2
        } else if x == y {
            Real::FRAC_PI_4
        } else {
            core::panic_with_felt252('not cardinal')
        }
    }
}

fn angle(a: Vector4<Fixed>, b: Vector4<Fixed>) -> Fixed {
    Vector4AngleImpl::<Fixed, FixedReal, CardinalAtan2>::angle(a, b)
}

#[test]
fn test_angle_cardinal_directions() {
    // 2 * atan2(|u - v|, |u + v|) on the normalized vectors, whatever their lengths.
    assert!(angle(v4(0x300000000, 0, 0, 0), v4(0x300000000, 0, 0, 0)) == Real::ZERO);
    assert!(angle(v4(0x300000000, 0, 0, 0), v4(0x1500000000, 0, 0, 0)) == Real::ZERO);
    assert!(
        angle(v4(0x300000000, 0, 0, 0), v4(0, 0x80000000, 0, 0)) == Real::<Fixed>::FRAC_PI_4
            + Real::FRAC_PI_4,
    );
    assert!(
        angle(v4(0, 0x80000000, 0, 0), v4(-0x300000000, 0, 0, 0)) == Real::<Fixed>::FRAC_PI_4
            + Real::FRAC_PI_4,
    );
    assert!(
        angle(v4(0x300000000, 0, 0, 0), v4(-0x300000000, 0, 0, 0)) == Real::<Fixed>::FRAC_PI_2
            + Real::FRAC_PI_2,
    );
    // Lengths whose product overflows (upstream divides by `|a| * |b|`).
    assert!(
        angle(v4(0xf424000000000, 0, 0, 0), v4(0, 0xf424000000000, 0, 0)) == Real::<
            Fixed,
        >::FRAC_PI_4
            + Real::FRAC_PI_4,
    );
}

#[test]
fn test_angle_of_zero_vector_is_zero() {
    assert!(angle(Vector4Trait::<Fixed>::zeros(), v4(0x300000000, 0, 0, 0)) == Real::ZERO);
    assert!(angle(v4(0x300000000, 0, 0, 0), Vector4Trait::<Fixed>::zeros()) == Real::ZERO);
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp, 0 = bit for bit)

#[test]
fn test_add_oracle() {
    let mut cases = oracle::vector4_add_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (vt(a), vt(b));
        assert!((a + b).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_sub_oracle() {
    let mut cases = oracle::vector4_sub_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (vt(a), vt(b));
        assert!((a - b).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_neg_oracle() {
    let mut cases = oracle::vector4_neg_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = vt(a);
        assert!((-a).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_scale_oracle() {
    let mut cases = oracle::vector4_scale_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, k, expected, tol) = *case;
        let a = vt(a);
        assert!((a.scale(fx(k))).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_dot_oracle() {
    let mut cases = oracle::vector4_dot_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (vt(a), vt(b));
        assert!((a.dot(b)).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_norm_squared_oracle() {
    let mut cases = oracle::vector4_norm_squared_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = vt(a);
        assert!((a.norm_squared()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_norm_oracle() {
    let mut cases = oracle::vector4_norm_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = vt(a);
        assert!((a.norm()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_normalize_oracle() {
    let mut cases = oracle::vector4_normalize_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = vt(a);
        assert!((a.normalize()).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_lerp_oracle() {
    let mut cases = oracle::vector4_lerp_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, t, expected, tol) = *case;
        let (a, b) = (vt(a), vt(b));
        assert!((a.lerp(b, fx(t))).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_component_mul_oracle() {
    let mut cases = oracle::vector4_component_mul_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (vt(a), vt(b));
        assert!((a.component_mul(b)).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_component_div_oracle() {
    let mut cases = oracle::vector4_component_div_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (vt(a), vt(b));
        assert!((a.component_div(b)).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_abs_oracle() {
    let mut cases = oracle::vector4_abs_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = vt(a);
        assert!((a.abs()).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_min_oracle() {
    let mut cases = oracle::vector4_min_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = vt(a);
        assert!((a.min()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_max_oracle() {
    let mut cases = oracle::vector4_max_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = vt(a);
        assert!((a.max()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_inf_oracle() {
    let mut cases = oracle::vector4_inf_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (vt(a), vt(b));
        assert!((a.inf(b)).abs_diff_eq(vt(expected), tol));
    }
}

#[test]
fn test_sup_oracle() {
    let mut cases = oracle::vector4_sup_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (vt(a), vt(b));
        assert!((a.sup(b)).abs_diff_eq(vt(expected), tol));
    }
}
