//! Unit tests of `Vector3`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, robustness at both ends of the range, identities, and
//! the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution,
//! every op of the suite) with
//! `cargo run --release -- emit-cairo vector3 --from vectors --max-per-dist 4 --out
//! <oracle.cairo>`.

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, v2, v3, v3t, v4};
use crate::base::vector2::Vector2Trait;
use crate::base::vector4::Vector4Trait;
use super::{Vector3, Vector3AngleTrait, Vector3Trait, oracle};

const MAX: i64 = 0x7fffffffffffffff;
const MIN: i64 = -0x8000000000000000;

/// (1.5, -2.25, 3.75)
fn a() -> Vector3<Fixed> {
    v3(0x180000000, -0x240000000, 0x3c0000000)
}

/// (-4.5, 0.25, 2)
fn b() -> Vector3<Fixed> {
    v3(-0x480000000, 0x40000000, 0x200000000)
}

/// (3, -4, 12), of norm 13
fn p() -> Vector3<Fixed> {
    v3(0x300000000, -0x400000000, 0xc00000000)
}

// --- constructors, resizing, conversions

#[test]
fn test_new_sets_fields() {
    let r = Vector3Trait::<Fixed>::new(fx(0x180000000), fx(-0x240000000), fx(0x3c0000000));
    assert!(r.x == fx(0x180000000) && r.y == fx(-0x240000000) && r.z == fx(0x3c0000000));
    assert!(r == a());
}

#[test]
fn test_zeros_is_zero() {
    assert!(Vector3Trait::<Fixed>::zeros() == v3(0, 0, 0));
    assert!(Vector3Trait::<Fixed>::zeros() == Default::default());
    assert!(Vector3Trait::<Fixed>::zeros().is_zero());
}

#[test]
fn test_repeat_fills_components() {
    assert!(
        Vector3Trait::<
            Fixed,
        >::repeat(fx(-0x240000000)) == v3(-0x240000000, -0x240000000, -0x240000000),
    );
    assert!(
        Vector3Trait::<
            Fixed,
        >::from_element(fx(-0x240000000)) == v3(-0x240000000, -0x240000000, -0x240000000),
    );
}

#[test]
fn test_axes_are_orthonormal() {
    assert!(Vector3Trait::<Fixed>::x() == v3(0x100000000, 0, 0));
    assert!(Vector3Trait::<Fixed>::y() == v3(0, 0x100000000, 0));
    assert!(Vector3Trait::<Fixed>::z() == v3(0, 0, 0x100000000));
    // Orthonormal: unit norms, zero dot products.
    assert!(Vector3Trait::<Fixed>::x().norm() == Real::ONE);
    assert!(Vector3Trait::<Fixed>::x().dot(Vector3Trait::<Fixed>::y()) == Real::ZERO);
    assert!(Vector3Trait::<Fixed>::x().dot(Vector3Trait::<Fixed>::z()) == Real::ZERO);
    assert!(Vector3Trait::<Fixed>::y().norm() == Real::ONE);
    assert!(Vector3Trait::<Fixed>::y().dot(Vector3Trait::<Fixed>::z()) == Real::ZERO);
    assert!(Vector3Trait::<Fixed>::z().norm() == Real::ONE);
}

#[test]
fn test_xy_keeps_first_components() {
    assert!(a().xy() == v2(0x180000000, -0x240000000));
}

#[test]
fn test_push_appends_component() {
    assert!(a().push(fx(0x700000000)) == v4(0x180000000, -0x240000000, 0x3c0000000, 0x700000000));
}

#[test]
fn test_to_homogeneous_appends_zero() {
    assert!(a().to_homogeneous() == v4(0x180000000, -0x240000000, 0x3c0000000, 0));
}

#[test]
fn test_push_xy_roundtrip() {
    assert!(a().push(fx(1)).xyz() == a());
    assert!(v2(0x180000000, -0x240000000).push(fx(0x3c0000000)) == a());
}

#[test]
fn test_tuple_conversions_roundtrip() {
    let r: Vector3<Fixed> = (fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)).into();
    assert!(r == a());
    let t: (Fixed, Fixed, Fixed) = r.into();
    assert!(t == (fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)));
}

#[test]
fn test_array_conversions_roundtrip() {
    let r: Vector3<Fixed> = [fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)].into();
    assert!(r == a());
    let [x, y, z]: [Fixed; 3] = r.into();
    assert!(x == fx(0x180000000) && y == fx(-0x240000000) && z == fx(0x3c0000000));
}

// --- operators

#[test]
fn test_add_exact() {
    // (1.5, -2.25, 3.75) + (-4.5, 0.25, 2) = (-3, -2, 5.75)
    assert!(a() + b() == v3(-0x300000000, -0x200000000, 0x5c0000000));
}

#[test]
fn test_sub_exact() {
    // (1.5, -2.25, 3.75) - (-4.5, 0.25, 2) = (6, -2.5, 1.75)
    assert!(a() - b() == v3(0x600000000, -0x280000000, 0x1c0000000));
}

#[test]
fn test_neg_exact() {
    assert!(-a() == v3(-0x180000000, 0x240000000, -0x3c0000000));
    assert!(-(-a()) == a());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_add_overflow() {
    let _ = black_box(v3(0, 0, 9223372036854775807)) + v3(0, 0, 1);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_sub_overflow() {
    let _ = black_box(v3(0, 0, -0x8000000000000000)) - v3(0, 0, 1);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_neg_overflow() {
    let _ = -black_box(v3(0, 0, MIN));
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
    // (1.5, -2.25, 3.75) * 2.5 = (3.75, -5.625, 9.375)
    assert!(a().scale(fx(0x280000000)) == v3(0x3c0000000, -0x5a0000000, 0x960000000));
    assert!(a().scale(Real::ONE) == a());
    assert!(a().scale(Real::ZERO).is_zero());
}

#[test]
fn test_scale_rounds_toward_negative_infinity() {
    // [3, -3, 1] ulp * 0.5: floor, whatever the sign.
    assert!(v3(3, -3, 1).scale(fx(0x80000000)) == v3(1, -2, 0));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_scale_overflow() {
    let _ = black_box(v3(0, 0, 0x4000000000000000)).scale(fx(0x200000000));
}

#[test]
fn test_unscale_exact() {
    // (1.5, -2.25, 3.75) / 2.5 = (0.5999999999, -0.9000000001, 1.5)
    assert!(a().unscale(fx(0x280000000)) == v3(2576980377, -3865470567, 0x180000000));
    assert!(a().unscale(Real::ONE) == a());
}

#[test]
fn test_unscale_rounds_toward_negative_infinity() {
    // +-1 / 3, +-2 / 3: exact floors.
    assert!(
        v3(0x100000000, -0x100000000, 0x200000000)
            .unscale(fx(0x300000000)) == v3(1431655765, -1431655766, 2863311530),
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
    let _ = black_box(v3(0, 0, 0x4000000000000000)).unscale(fx(0x80000000));
}

#[test]
fn test_component_mul_exact() {
    // (1.5, -2.25, 3.75) .* (-4.5, 0.25, 2) = (-6.75, -0.5625, 7.5)
    assert!(a().component_mul(b()) == v3(-0x6c0000000, -0x90000000, 0x780000000));
    assert!(a().component_mul(Vector3Trait::<Fixed>::repeat(Real::ONE)) == a());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_component_mul_overflow() {
    let _ = black_box(v3(0, 0, 0x1000000000000)).component_mul(v3(0, 0, 0x800000000000));
}

#[test]
fn test_component_div_exact() {
    // (1.5, -2.25, 3.75) ./ (-4.5, 0.25, 2) = floor of the exact quotients
    assert!(a().component_div(b()) == v3(-1431655766, -0x900000000, 0x1e0000000));
    assert!(a().component_div(Vector3Trait::<Fixed>::repeat(Real::ONE)) == a());
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_component_div_by_zero() {
    let _ = black_box(a()).component_div(v3(0x100000000, 0x100000000, 0));
}

#[test]
fn test_abs_exact() {
    assert!(a().abs() == v3(0x180000000, 0x240000000, 0x3c0000000));
    assert!(a().abs().abs() == a().abs());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_abs_overflow() {
    let _ = black_box(v3(MIN, 0, 0)).abs();
}

#[test]
fn test_inf_sup_exact() {
    // (1.5, -2.25, 3.75) and (-4.5, 0.25, 2)
    assert!(a().inf(b()) == v3(-0x480000000, -0x240000000, 0x200000000));
    assert!(a().sup(b()) == v3(0x180000000, 0x40000000, 0x3c0000000));
    assert!(
        a()
            .inf_sup(
                b(),
            ) == (
                v3(-0x480000000, -0x240000000, 0x200000000),
                v3(0x180000000, 0x40000000, 0x3c0000000),
            ),
    );
    assert!(b().inf(a()) == v3(-0x480000000, -0x240000000, 0x200000000));
    assert!(a().inf(a()) == a() && a().sup(a()) == a());
    assert!(v3(MIN, MIN, MIN).inf(v3(MAX, MAX, MAX)) == v3(MIN, MIN, MIN));
}

// --- reductions

#[test]
fn test_min_max_exact() {
    // (1.5, -2.25, 3.75)
    assert!(a().min() == fx(-0x240000000));
    assert!(a().max() == fx(0x3c0000000));
    assert!(a().amin() == fx(0x180000000));
    assert!(a().amax() == fx(0x3c0000000));
}

#[test]
fn test_min_max_every_position() {
    assert!(
        v3(-0x700000000, 0x500000000, 0x500000000).min() == fx(-0x700000000)
            && v3(-0x700000000, 0x500000000, 0x500000000).imin() == 0,
    );
    assert!(
        v3(0x700000000, -0x500000000, -0x500000000).max() == fx(0x700000000)
            && v3(0x700000000, -0x500000000, -0x500000000).imax() == 0,
    );
    assert!(
        v3(-0x700000000, 0x500000000, 0x500000000).amax() == fx(0x700000000)
            && v3(-0x700000000, 0x500000000, 0x500000000).iamax() == 0,
    );
    assert!(
        v3(0x300000000, -0x500000000, -0x500000000).amin() == fx(0x300000000)
            && v3(0x300000000, -0x500000000, -0x500000000).iamin() == 0,
    );
    assert!(
        v3(0x500000000, -0x700000000, 0x500000000).min() == fx(-0x700000000)
            && v3(0x500000000, -0x700000000, 0x500000000).imin() == 1,
    );
    assert!(
        v3(-0x500000000, 0x700000000, -0x500000000).max() == fx(0x700000000)
            && v3(-0x500000000, 0x700000000, -0x500000000).imax() == 1,
    );
    assert!(
        v3(0x500000000, -0x700000000, 0x500000000).amax() == fx(0x700000000)
            && v3(0x500000000, -0x700000000, 0x500000000).iamax() == 1,
    );
    assert!(
        v3(-0x500000000, 0x300000000, -0x500000000).amin() == fx(0x300000000)
            && v3(-0x500000000, 0x300000000, -0x500000000).iamin() == 1,
    );
    assert!(
        v3(0x500000000, 0x500000000, -0x700000000).min() == fx(-0x700000000)
            && v3(0x500000000, 0x500000000, -0x700000000).imin() == 2,
    );
    assert!(
        v3(-0x500000000, -0x500000000, 0x700000000).max() == fx(0x700000000)
            && v3(-0x500000000, -0x500000000, 0x700000000).imax() == 2,
    );
    assert!(
        v3(0x500000000, 0x500000000, -0x700000000).amax() == fx(0x700000000)
            && v3(0x500000000, 0x500000000, -0x700000000).iamax() == 2,
    );
    assert!(
        v3(-0x500000000, -0x500000000, 0x300000000).amin() == fx(0x300000000)
            && v3(-0x500000000, -0x500000000, 0x300000000).iamin() == 2,
    );
}

#[test]
fn test_imin_imax_ties_pick_first() {
    // Upstream returns the first extremum.
    assert!(
        v3(0x100000000, 0x100000000, 0x100000000).imax() == 0
            && v3(0x100000000, 0x100000000, 0x100000000).imin() == 0,
    );
    assert!(
        v3(-0x100000000, -0x100000000, -0x100000000).iamax() == 0
            && v3(-0x100000000, -0x100000000, -0x100000000).iamin() == 0,
    );
    assert!(
        v3(0x80000000, 0x100000000, 0x100000000).imax() == 1
            && v3(0x200000000, 0x100000000, 0x100000000).imin() == 1,
    );
    assert!(
        v3(-0x80000000, -0x100000000, -0x100000000).iamax() == 1
            && v3(-0x200000000, -0x100000000, -0x100000000).iamin() == 1,
    );
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_amax_overflow() {
    let _ = black_box(v3(0, 0, MIN)).amax();
}

#[test]
fn test_sum_exact() {
    // (1.5, -2.25, 3.75)
    assert!(a().sum() == fx(0x300000000));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_sum_overflow() {
    let _ = black_box(v3(0x4000000000000000, 0x4000000000000000, 0)).sum();
}

#[test]
fn test_is_zero_cases() {
    assert!(v3(0, 0, 0).is_zero());
    assert!(!v3(1, 0, 0).is_zero());
    assert!(!v3(0, 1, 0).is_zero());
    assert!(!v3(0, 0, 1).is_zero());
}

#[test]
fn test_abs_diff_eq_counts_ulps() {
    let r = a();
    assert!(r.abs_diff_eq(r, 0));
    assert!(
        r.abs_diff_eq(v3(6442450947, -0x240000000, 0x3c0000000), 3)
            && !r.abs_diff_eq(v3(6442450947, -0x240000000, 0x3c0000000), 2),
    );
    assert!(
        r.abs_diff_eq(v3(0x180000000, -9663676419, 0x3c0000000), 3)
            && !r.abs_diff_eq(v3(0x180000000, -9663676419, 0x3c0000000), 2),
    );
    assert!(
        r.abs_diff_eq(v3(0x180000000, -0x240000000, 16106127363), 3)
            && !r.abs_diff_eq(v3(0x180000000, -0x240000000, 16106127363), 2),
    );
    // The difference itself may exceed the scalar range.
    assert!(!v3(MIN, MIN, MIN).abs_diff_eq(v3(MAX, MAX, MAX), 0xfffffffffffffffe));
    assert!(v3(MIN, MIN, MIN).abs_diff_eq(v3(MAX, MAX, MAX), 0xffffffffffffffff));
}

// --- products

#[test]
fn test_dot_exact() {
    // (1.5, -2.25, 3.75) . (-4.5, 0.25, 2) = 0.1875
    assert!(a().dot(b()) == fx(0x30000000));
    assert!(b().dot(a()) == fx(0x30000000));
    assert!(a().dot(a()) == a().norm_squared());
}

#[test]
fn test_dot_rounds_once() {
    // 3 products of 0.5 ulp each: the fused sum is floor(1.5) = 1 ulp, where rounding every
    // product would give 0.
    assert!(v3(0x80000000, 0x80000000, 0x80000000).dot(v3(1, 1, 1)) == fx(1));
}

#[test]
fn test_dot_intermediate_products_may_overflow() {
    // 60000^2 = 3.6e9 does not fit Q32.32, the cancelled sum does.
    assert!(
        v3(0xea6000000000, 0xea6000000000, 0)
            .dot(v3(0xea6000000000, -0xea6000000000, 0x100000000)) == fx(0),
    );
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_dot_overflow() {
    let _ = black_box(v3(0xea6000000000, 0xea6000000000, 0))
        .dot(v3(0xea6000000000, 0xea6000000000, 0));
}

#[test]
fn test_cross_exact() {
    // (1.5, -2.25, 3.75) x (-4.5, 0.25, 2) = (-5.4375, -19.875, -9.75)
    assert!(a().cross(b()) == v3(-0x570000000, -0x13e0000000, -0x9c0000000));
    assert!(b().cross(a()) == v3(0x570000000, 0x13e0000000, 0x9c0000000));
    assert!(a().cross(a()).is_zero());
}

#[test]
fn test_cross_axes_are_right_handed() {
    assert!(
        Vector3Trait::<Fixed>::x().cross(Vector3Trait::<Fixed>::y()) == Vector3Trait::<Fixed>::z(),
    );
    assert!(
        Vector3Trait::<Fixed>::y().cross(Vector3Trait::<Fixed>::z()) == Vector3Trait::<Fixed>::x(),
    );
    assert!(
        Vector3Trait::<Fixed>::z().cross(Vector3Trait::<Fixed>::x()) == Vector3Trait::<Fixed>::y(),
    );
}

#[test]
fn test_cross_is_orthogonal_to_operands() {
    // Quarter-valued inputs: every product is exact.
    let r = a().cross(b());
    assert!(r.dot(a()) == Real::ZERO && r.dot(b()) == Real::ZERO);
}

#[test]
fn test_cross_intermediate_products_may_overflow() {
    // 60000 * 60001 - 60000 * 60000 = 60000: only the result must fit.
    assert!(
        v3(0xea6000000000, 0xea6000000000, 0)
            .cross(v3(0xea6000000000, 0xea6100000000, 0)) == v3(0, 0, 0xea6000000000),
    );
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_cross_overflow() {
    let _ = black_box(v3(0xea6000000000, 0xea6000000000, 0))
        .cross(v3(-0xea6000000000, 0xea6000000000, 0));
}

// --- norms

#[test]
fn test_norm_squared_exact() {
    // |(3, -4, 12)|^2 = 169
    assert!(p().norm_squared() == fx(0xa900000000));
    assert!(p().magnitude_squared() == fx(0xa900000000));
    assert!(a().norm_squared() == fx(0x1560000000));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_norm_squared_overflow() {
    let _ = black_box(v3(0xea6000000000, 0, 0)).norm_squared();
}

#[test]
fn test_norm_exact() {
    // |(3, -4, 12)| = 13
    assert!(p().norm() == fx(0xd00000000));
    assert!(p().magnitude() == fx(0xd00000000));
    assert!(v3(0, 0, 0).norm() == Real::ZERO);
    assert!(a().norm() == fx(19856967406));
}

#[test]
fn test_norm_large_magnitude_does_not_overflow() {
    // |(1e6, ..)| = 1732050.808: the squared norm (3e12) is far outside Q32.32.
    assert!(v3(0xf424000000000, 0xf424000000000, 0xf424000000000).norm() == fx(7439101573518717));
    // Largest components.
    assert!(v3(MAX, 0, 0).norm() == fx(MAX));
}

#[test]
fn test_norm_tiny_magnitude_keeps_precision() {
    // |(3, -4, ..) ulp| = 5 ulp: the squares vanish in Q32.32, not in the unscaled kernel.
    assert!(v3(3, -4, 0).norm() == fx(5));
    assert!(v3(3, -4, 0).norm_squared() == Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_norm_overflow() {
    let _ = black_box(v3(MAX, MAX, 0)).norm();
}

#[test]
fn test_metric_distance_exact() {
    // Distance between (4.5, -6.25, 15.75) and (1.5, -2.25, 3.75).
    assert!(v3(0x480000000, -0x640000000, 0xfc0000000).metric_distance(a()) == fx(0xd00000000));
    assert!(a().metric_distance(v3(0x480000000, -0x640000000, 0xfc0000000)) == fx(0xd00000000));
    assert!(a().metric_distance(a()) == Real::ZERO);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_metric_distance_overflow() {
    let _ = black_box(v3(MAX, 0, 0)).metric_distance(v3(-1, 0, 0));
}

#[test]
fn test_normalize_exact() {
    // (3, -4, 12) / 13, floored.
    assert!(p().normalize() == v3(991146299, -1321528399, 3964585196));
    assert!(v3(-0x500000000, 0, 0).normalize() == -Vector3Trait::<Fixed>::x());
    assert!(v3(0, -0x500000000, 0).normalize() == -Vector3Trait::<Fixed>::y());
    assert!(v3(0, 0, -0x500000000).normalize() == -Vector3Trait::<Fixed>::z());
}

#[test]
fn test_normalize_large_magnitude() {
    // (1e6, ..) / |(1e6, ..)|: no overflow, every component is 1 / sqrt(3) within 1 ulp.
    assert!(
        v3(0xf424000000000, 0xf424000000000, 0xf424000000000)
            .normalize() == v3(2479700524, 2479700524, 2479700524),
    );
    assert!(
        v3(0xf424000000000, 0xf424000000000, 0xf424000000000)
            .normalize()
            .abs_diff_eq(Vector3Trait::<Fixed>::repeat(fx(2479700524)), 1),
    );
}

#[test]
fn test_normalize_tiny_magnitude() {
    // (3, -4, ..) ulp: norm 5 ulp, result (0.6, -0.8, ..) floored.
    assert!(v3(3, -4, 0).normalize() == v3(2576980377, -3435973837, 0));
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_normalize_zero() {
    let _ = black_box(Vector3Trait::<Fixed>::zeros()).normalize();
}

#[test]
fn test_normalize_is_unit_within_tolerance() {
    let r = a().normalize();
    assert!(r == v3(1393471396, -2090207096, 3483678492));
    assert!(r.norm().abs_diff_eq(Real::ONE, 6));
}

#[test]
fn test_try_normalize_some() {
    assert!(p().try_normalize(Real::ZERO) == Some(p().normalize()));
    assert!(p().try_normalize(fx(0xd00000000) - Real::EPSILON) == Some(p().normalize()));
}

#[test]
fn test_try_normalize_none() {
    assert!(Vector3Trait::<Fixed>::zeros().try_normalize(Real::ZERO) == None);
    // The threshold is inclusive, like upstream (`norm <= min_norm`).
    assert!(p().try_normalize(fx(0xd00000000)) == None);
    assert!(v3(3, -4, 0).try_normalize(Real::EPSILON) == Some(v3(3, -4, 0).normalize()));
    assert!(v3(3, -4, 0).try_normalize(fx(5)) == None);
}

#[test]
fn test_cap_magnitude_exact() {
    // |(3, -4, 12)| = 13 capped to 6.5: the ratio 0.5 is exact.
    assert!(p().cap_magnitude(fx(0x680000000)) == v3(0x180000000, -0x200000000, 0x600000000));
    assert!(p().cap_magnitude(fx(0x680000000)).norm() == fx(0x680000000));
    // Not longer than the cap: unchanged (inclusive).
    assert!(p().cap_magnitude(fx(0xd00000000)) == p());
    assert!(p().cap_magnitude(fx(MAX)) == p());
    assert!(p().cap_magnitude(Real::ZERO).is_zero());
    assert!(Vector3Trait::<Fixed>::zeros().cap_magnitude(Real::ZERO).is_zero());
}

#[test]
fn test_cap_magnitude_never_exceeds_cap_by_more_than_rounding() {
    let r = a().cap_magnitude(fx(0x200000000));
    assert!(r == v3(2786942793, -4180414190, 6967356982));
    assert!(r.norm() <= fx(0x200000000) && r.norm().abs_diff_eq(fx(0x200000000), 16));
}

#[test]
fn test_cap_magnitude_large_magnitude() {
    assert!(
        v3(0xf424000000000, 0xf424000000000, 0xf424000000000)
            .cap_magnitude(fx(0xa00000000)) == v3(24797000000, 24797000000, 24797000000),
    );
}

// --- interpolation, basis

#[test]
fn test_lerp_exact() {
    // From (1.5, -2.25, 3.75) to (-4.5, 0.25, 2).
    assert!(a().lerp(b(), Real::ZERO) == a());
    assert!(a().lerp(b(), Real::ONE) == b());
    assert!(a().lerp(b(), Real::HALF) == v3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(a().lerp(b(), fx(0x40000000)) == v3(0, -0x1a0000000, 0x350000000));
    // Not clamped: extrapolation.
    assert!(a().lerp(b(), Real::TWO) == v3(-0xa80000000, 0x2c0000000, 0x40000000));
    assert!(a().lerp(b(), Real::NEG_ONE) == v3(0x780000000, -0x4c0000000, 0x580000000));
}

#[test]
fn test_lerp_full_range_does_not_overflow() {
    // `rhs - self` does not fit the scalar; the kernel works on the exact difference.
    let (lo, hi) = (v3(MIN, MAX, MAX), v3(MAX, MIN, MIN));
    assert!(lo.lerp(hi, Real::ONE) == hi);
    assert!(lo.lerp(hi, Real::HALF) == v3(-1, -1, -1));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_lerp_overflow() {
    let _ = black_box(v3(0, 0, 0))
        .lerp(v3(0x4000000000000000, 0x4000000000000000, 0x4000000000000000), fx(0x400000000));
}

#[test]
fn test_orthonormal_basis_of_axes() {
    assert!(
        Vector3Trait::<Fixed>::x()
            .orthonormal_basis() == (v3(0, 0, -0x100000000), v3(0, 0x100000000, 0)),
    );
    assert!(
        Vector3Trait::<Fixed>::y()
            .orthonormal_basis() == (v3(0x100000000, 0, 0), v3(0, 0, -0x100000000)),
    );
    assert!(
        Vector3Trait::<Fixed>::z()
            .orthonormal_basis() == (v3(0x100000000, 0, 0), v3(0, 0x100000000, 0)),
    );
    assert!(
        (-Vector3Trait::<Fixed>::z())
            .orthonormal_basis() == (v3(0x100000000, 0, 0), v3(0, -0x100000000, 0)),
    );
}

/// `(v, u, w)` is orthonormal and right-handed (`u x w = v`) within `tol` ulp.
fn assert_orthonormal_basis(v: Vector3<Fixed>, tol: u64) {
    let (u, w) = v.orthonormal_basis();
    assert!(u.norm().abs_diff_eq(Real::ONE, tol) && w.norm().abs_diff_eq(Real::ONE, tol));
    assert!(u.dot(w).abs_diff_eq(Real::ZERO, tol));
    assert!(u.dot(v).abs_diff_eq(Real::ZERO, tol) && w.dot(v).abs_diff_eq(Real::ZERO, tol));
    assert!(u.cross(w).abs_diff_eq(v, tol));
}

#[test]
fn test_orthonormal_basis_is_orthonormal_and_right_handed() {
    // Every octant, directions close to the poles `z = +-1` and to the equator, axes.
    let mut dirs = array![
        v3(0x100000000, 0x200000000, 0x300000000), // (1, 2, 3)
        v3(-0x100000000, 0x200000000, 0x300000000), // (-1, 2, 3)
        v3(0x100000000, -0x200000000, 0x300000000), // (1, -2, 3)
        v3(0x100000000, 0x200000000, -0x300000000), // (1, 2, -3)
        v3(-0x100000000, -0x200000000, 0x300000000), // (-1, -2, 3)
        v3(-0x100000000, 0x200000000, -0x300000000), // (-1, 2, -3)
        v3(0x100000000, -0x200000000, -0x300000000), // (1, -2, -3)
        v3(-0x100000000, -0x200000000, -0x300000000), // (-1, -2, -3)
        v3(0x100000000, 0x100000000, 0x186a000000000), // (1, 1, 100000)
        v3(-0x100000000, 0x100000000, -0x186a000000000), // (-1, 1, -100000)
        v3(0x186a000000000, 0x300000000, 0x100000000), // (100000, 3, 1)
        v3(0x300000000, -0x186a000000000, -0x100000000), // (3, -100000, -1)
        v3(0, 0x500000000, 0), // (0, 5, 0)
        v3(0x700000000, 0, 0), // (7, 0, 0)
        v3(0x100000000, 0x100000000, 0), // (1, 1, 0)
        v3(0x40000000, -0x80000000, 0x20000000) // (0.25, -0.5, 0.125)
    ]
        .span();
    while let Some(dir) = dirs.pop_front() {
        assert_orthonormal_basis((*dir).normalize(), 2);
    }
}

#[test]
fn test_orthonormal_basis_model_values() {
    // Unit vector of (1.5, -2.25, 3.75), then with z negated (other branch).
    assert!(
        v3(1393471396, -2090207096, 3483678492)
            .orthonormal_basis() == (
                v3(4045339972, 374440986, -1393471396), v3(374440986, 3733305815, 2090207096),
            ),
    );
    assert!(
        v3(1393471396, -2090207096, -3483678493)
            .orthonormal_basis() == (
                v3(4045339972, 374440986, 1393471396), v3(-374440986, -3733305816, 2090207096),
            ),
    );
}

// --- angle

#[test]
fn test_angle_cardinal_directions() {
    // 2 * atan2(|u - v|, |u + v|) on the normalized vectors, whatever their lengths. `atan2` is
    // accurate to 1.12 ulp (DESIGN D6) and normalizing the inputs costs a couple more, so the
    // cardinal angles land within 4 ulp of the exact 0, pi/2 and pi. Parallel vectors give
    // EXACTLY zero (`|u - v| = 0` and `atan2(0, x) = 0` for `x > 0`).
    let half_turn = Real::<Fixed>::PI;
    let quarter_turn = Real::<Fixed>::FRAC_PI_2;
    assert!(v3(0x300000000, 0, 0).angle(v3(0x300000000, 0, 0)) == Real::ZERO);
    assert!(v3(0x300000000, 0, 0).angle(v3(0x1500000000, 0, 0)) == Real::ZERO);
    assert!(Real::abs_diff_eq(v3(0x300000000, 0, 0).angle(v3(0, 0x80000000, 0)), quarter_turn, 4));
    assert!(Real::abs_diff_eq(v3(0, 0x80000000, 0).angle(v3(-0x300000000, 0, 0)), quarter_turn, 4));
    assert!(Real::abs_diff_eq(v3(0x300000000, 0, 0).angle(v3(-0x300000000, 0, 0)), half_turn, 4));
    // Lengths whose product overflows (upstream divides by `|a| * |b|`).
    assert!(
        Real::abs_diff_eq(
            v3(0xf424000000000, 0, 0).angle(v3(0, 0xf424000000000, 0)), quarter_turn, 4,
        ),
    );
}

#[test]
fn test_angle_of_zero_vector_is_zero() {
    assert!(Vector3Trait::<Fixed>::zeros().angle(v3(0x300000000, 0, 0)) == Real::ZERO);
    assert!(v3(0x300000000, 0, 0).angle(Vector3Trait::<Fixed>::zeros()) == Real::ZERO);
}

#[test]
fn test_angle_oracle() {
    let mut cases = oracle::vector3_angle_cases();
    assert!(cases.len() >= 12);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(Real::abs_diff_eq(v3t(a).angle(v3t(b)), fx(expected), tol));
    }
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp, 0 = bit for bit)

#[test]
fn test_add_oracle() {
    let mut cases = oracle::vector3_add_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a + b).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_sub_oracle() {
    let mut cases = oracle::vector3_sub_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a - b).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_neg_oracle() {
    let mut cases = oracle::vector3_neg_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v3t(a);
        assert!((-a).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_scale_oracle() {
    let mut cases = oracle::vector3_scale_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, k, expected, tol) = *case;
        let a = v3t(a);
        assert!((a.scale(fx(k))).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_dot_oracle() {
    let mut cases = oracle::vector3_dot_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a.dot(b)).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_norm_squared_oracle() {
    let mut cases = oracle::vector3_norm_squared_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v3t(a);
        assert!((a.norm_squared()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_norm_oracle() {
    let mut cases = oracle::vector3_norm_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v3t(a);
        assert!((a.norm()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_normalize_oracle() {
    let mut cases = oracle::vector3_normalize_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v3t(a);
        assert!((a.normalize()).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_lerp_oracle() {
    let mut cases = oracle::vector3_lerp_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, t, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a.lerp(b, fx(t))).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_component_mul_oracle() {
    let mut cases = oracle::vector3_component_mul_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a.component_mul(b)).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_component_div_oracle() {
    let mut cases = oracle::vector3_component_div_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a.component_div(b)).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_abs_oracle() {
    let mut cases = oracle::vector3_abs_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v3t(a);
        assert!((a.abs()).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_min_oracle() {
    let mut cases = oracle::vector3_min_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v3t(a);
        assert!((a.min()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_max_oracle() {
    let mut cases = oracle::vector3_max_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, expected, tol) = *case;
        let a = v3t(a);
        assert!((a.max()).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_inf_oracle() {
    let mut cases = oracle::vector3_inf_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a.inf(b)).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_sup_oracle() {
    let mut cases = oracle::vector3_sup_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a.sup(b)).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_cross_oracle() {
    let mut cases = oracle::vector3_cross_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        let (a, b) = (v3t(a), v3t(b));
        assert!((a.cross(b)).abs_diff_eq(v3t(expected), tol));
    }
}
